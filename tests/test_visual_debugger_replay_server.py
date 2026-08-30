"""Core-free replay-mode HTTP coordinator tests with an injected fake service."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from pathlib import Path
from threading import Event, Thread
from typing import Annotated, Literal, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field
from scripts.dev.visual_debugger.replay_protocol import (
    ReplayApiErrorV1,
    ReplayCommandRequestV1,
    ReplayCommandResponseV1,
    ReplayNextFrameCommandV1,
    SharedObsAgentPovReplayTimelineV1,
    SharedObsAgentPovReplayViewerFrameV1,
)
from scripts.dev.visual_debugger.replay_service import ReplayViewerService
from scripts.dev.visual_debugger.server import (
    LIVE_HTTP_ROUTES,
    REPLAY_HTTP_ROUTES,
    DebuggerHTTPServer,
    HttpCoordinatorBinding,
    HttpCoordinatorReplacement,
    HttpCoordinatorRouter,
    HttpRouteSet,
    create_server,
    serve_browser_debugger,
)
from tests.evaluation_fixtures import captured_evaluation_trajectory
from tests.export_visual_debugger_replay_artifacts import export_artifacts

from marl_battlegrounds.evaluation.metrics import build_evaluation_observer_v1
from marl_battlegrounds.evaluation.replay import (
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.replay_io import (
    LoadedReplayBundleV1,
    canonical_metric_report_artifact_json_bytes_v1,
    load_replay_bundle_v1,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ASSET_ROOT = _REPOSITORY_ROOT / "web" / "visual_debugger"
_TOKEN_HEADER = "X-MARL-Debugger-Token"
_TOKEN = "test-replay-capability"

type _ErrorCode = Literal[
    "audience_unavailable",
    "command_id_conflict",
    "forbidden_origin",
    "internal_error",
    "invalid_cursor",
    "invalid_request",
    "method_not_allowed",
    "not_found",
    "payload_too_large",
    "server_shutting_down",
    "stale_revision",
    "unauthorized",
    "unsupported_media_type",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _FakeReplayFrame(_StrictModel):
    schema_version: Literal[1] = 1
    frame_kind: Literal["researcher_replay_viewer"] = "researcher_replay_viewer"
    revision: Annotated[int, Field(ge=0)]
    frame_index: Annotated[int, Field(ge=0)]


class _FakeReplayTimeline(_StrictModel):
    schema_version: Literal[1] = 1
    timeline_kind: Literal["researcher_replay_timeline"] = "researcher_replay_timeline"
    current_frame_index: Annotated[int, Field(ge=0)]
    frame_indices: tuple[Annotated[int, Field(ge=0)], ...]


class _FakeReplayPresentation(_StrictModel):
    schema_version: Literal[1] = 1
    presentation_kind: Literal["replay_oracle"] = "replay_oracle"
    source_revision: Annotated[int, Field(ge=0)]
    source_frame_index: Annotated[int, Field(ge=0)]


class _FakeLivePresentation(_StrictModel):
    schema_version: Literal[1] = 1
    presentation_kind: Literal["live_oracle"] = "live_oracle"
    source_revision: Annotated[int, Field(ge=0)]
    source_frame_index: Annotated[int, Field(ge=0)]


class _FakePresentationError(_StrictModel):
    schema_version: Literal[1] = 1
    error_code: Literal["audience_unavailable"] = "audience_unavailable"
    message: str = Field(min_length=1)


class _FakeSeekCommand(_StrictModel):
    command_type: Literal["seek"] = "seek"
    frame_index: Annotated[int, Field(ge=0)]


class _FakeExitCommand(_StrictModel):
    command_type: Literal["exit"] = "exit"


type _FakeCommand = Annotated[
    _FakeSeekCommand | _FakeExitCommand,
    Field(discriminator="command_type"),
]


class _FakeReplayRequest(_StrictModel):
    schema_version: Literal[1] = 1
    client_id: str = Field(min_length=1)
    command_id: str = Field(min_length=1)
    base_revision: Annotated[int, Field(ge=0)]
    command: _FakeCommand


class _FakeReplayResponse(_StrictModel):
    schema_version: Literal[1] = 1
    result: Literal["applied", "duplicate", "shutdown_scheduled"]
    frame: _FakeReplayFrame


class _FakeReplayError(_StrictModel):
    schema_version: Literal[1] = 1
    error_code: _ErrorCode
    message: str = Field(min_length=1)
    latest_frame: _FakeReplayFrame | None = None


class _FakeLiveFrame(_StrictModel):
    schema_version: Literal[2] = 2
    frame_kind: Literal["live_debugger"] = "live_debugger"
    revision: Annotated[int, Field(ge=0)]


class _FakeLiveRequest(_StrictModel):
    schema_version: Literal[2] = 2
    command_type: Literal["advance"] = "advance"


class _FakeLiveResponse(_StrictModel):
    schema_version: Literal[2] = 2
    result: Literal["applied"] = "applied"
    frame: _FakeLiveFrame


class _FakeLiveError(_StrictModel):
    schema_version: Literal[2] = 2
    error_code: _ErrorCode
    message: str = Field(min_length=1)
    latest_frame: _FakeLiveFrame | None = None


@dataclass(frozen=True, slots=True)
class _FakeServiceResult:
    outcome: str
    payload: _FakeReplayResponse | _FakeReplayError
    shutdown_requested: bool = False


@dataclass(frozen=True, slots=True)
class _FakeLiveServiceResult:
    outcome: str
    payload: _FakeLiveResponse | _FakeLiveError
    shutdown_requested: bool = False


@dataclass(frozen=True, slots=True)
class _FakePresentationResult:
    outcome: Literal["response", "audience_unavailable"]
    payload: _FakeReplayPresentation | _FakeLivePresentation | _FakePresentationError


@dataclass(frozen=True, slots=True)
class _FakeMetricReportResult:
    outcome: str
    payload: bytes | None
    filename: str | None


class _FakeReplayService:
    """Small transport fake; no production replay or simulator imports."""

    def __init__(self) -> None:
        self.revision = 0
        self.frame_index = 0
        self.frame_calls = 0
        self.timeline_calls = 0
        self.presentation_calls = 0
        self.metric_report_calls = 0
        self.command_calls = 0
        self.cursor_mutations = 0
        self.private_replay = {
            "hidden_events": ["must-never-cross-http"],
            "metric_report": {"secret": 1},
        }
        self._commands: dict[tuple[str, str], str] = {}
        self.presentation_available = True
        self.metric_report_result: object = _FakeMetricReportResult(
            outcome="available",
            payload=b'{"canonical":true}',
            filename="fake-replay.marlbg-metrics.json",
        )

    def current_frame(self) -> _FakeReplayFrame:
        self.frame_calls += 1
        return self._frame()

    def current_timeline(self) -> _FakeReplayTimeline:
        self.timeline_calls += 1
        return _FakeReplayTimeline(
            current_frame_index=self.frame_index,
            frame_indices=(0, 1, 2),
        )

    def current_presentation(self) -> _FakePresentationResult:
        self.presentation_calls += 1
        if not self.presentation_available:
            return _FakePresentationResult(
                outcome="audience_unavailable",
                payload=_FakePresentationError(
                    message=(
                        "Authorized presentation is unavailable for the active "
                        "audience."
                    )
                ),
            )
        return _FakePresentationResult(
            outcome="response",
            payload=_FakeReplayPresentation(
                source_revision=self.revision,
                source_frame_index=self.frame_index,
            ),
        )

    def current_live_presentation(self) -> _FakePresentationResult:
        self.presentation_calls += 1
        return _FakePresentationResult(
            outcome="response",
            payload=_FakeLivePresentation(
                source_revision=self.revision,
                source_frame_index=self.frame_index,
            ),
        )

    def current_metric_report(self) -> _FakeMetricReportResult:
        self.metric_report_calls += 1
        return cast(_FakeMetricReportResult, self.metric_report_result)

    def apply_command(self, request: _FakeReplayRequest) -> _FakeServiceResult:
        self.command_calls += 1
        key = (request.client_id, request.command_id)
        fingerprint = request.model_dump_json()
        previous = self._commands.get(key)
        if previous is not None:
            if previous != fingerprint:
                return self._error_result(
                    "command_id_conflict",
                    "Command ID was reused with a different request.",
                    outcome="command_id_conflict",
                )
            return _FakeServiceResult(
                outcome="response",
                payload=_FakeReplayResponse(
                    result="duplicate",
                    frame=self._frame(),
                ),
            )
        if request.base_revision != self.revision:
            return self._error_result(
                "stale_revision",
                "Replay revision is stale.",
                outcome="stale_revision",
            )
        if request.command.command_type == "exit":
            self._commands[key] = fingerprint
            return _FakeServiceResult(
                outcome="response",
                payload=_FakeReplayResponse(
                    result="shutdown_scheduled",
                    frame=self._frame(),
                ),
                shutdown_requested=True,
            )
        if request.command.frame_index > 2:
            return self._error_result(
                "invalid_cursor",
                "Replay cursor is outside the artifact.",
                outcome="invalid_cursor",
            )

        self.frame_index = request.command.frame_index
        self.revision += 1
        self.cursor_mutations += 1
        self._commands[key] = fingerprint
        return _FakeServiceResult(
            outcome="response",
            payload=_FakeReplayResponse(result="applied", frame=self._frame()),
        )

    def _frame(self) -> _FakeReplayFrame:
        return _FakeReplayFrame(
            revision=self.revision,
            frame_index=self.frame_index,
        )

    def _error_result(
        self,
        error_code: _ErrorCode,
        message: str,
        *,
        outcome: str,
    ) -> _FakeServiceResult:
        return _FakeServiceResult(
            outcome=outcome,
            payload=_FakeReplayError(
                error_code=error_code,
                message=message,
                latest_frame=self._frame(),
            ),
        )


class _BlockingLiveService:
    """Transport fake whose selected operation remains inside one request."""

    def __init__(
        self,
        *,
        block_frame_with_error: bool = False,
        block_presentation: bool = False,
    ) -> None:
        self.block_frame_with_error = block_frame_with_error
        self.block_presentation = block_presentation
        self.entered = Event()
        self.release = Event()
        self.received_request: _FakeLiveRequest | None = None
        self.on_apply: Callable[[], None] | None = None
        self.presentation_calls = 0

    def current_frame(self) -> _FakeLiveFrame:
        if self.block_frame_with_error:
            self._wait_for_release()
            raise RuntimeError("synthetic live failure")
        return _FakeLiveFrame(revision=0)

    def current_presentation(self) -> _FakePresentationResult:
        self.presentation_calls += 1
        if self.block_presentation:
            self._wait_for_release()
        return _FakePresentationResult(
            outcome="response",
            payload=_FakeLivePresentation(
                source_revision=0,
                source_frame_index=0,
            ),
        )

    def apply_command(self, request: _FakeLiveRequest) -> _FakeLiveServiceResult:
        self.received_request = request
        self._wait_for_release()
        if self.on_apply is not None:
            self.on_apply()
        return _FakeLiveServiceResult(
            outcome="response",
            payload=_FakeLiveResponse(frame=_FakeLiveFrame(revision=1)),
        )

    def _wait_for_release(self) -> None:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("synthetic live request was not released")


def _error_factory(*, error_code: str, message: str) -> _FakeReplayError:
    return _FakeReplayError(
        error_code=cast(_ErrorCode, error_code),
        message=message,
    )


def _live_error_factory(*, error_code: str, message: str) -> _FakeLiveError:
    return _FakeLiveError(
        error_code=cast(_ErrorCode, error_code),
        message=message,
    )


def _live_coordinator(service: _BlockingLiveService) -> HttpCoordinatorBinding:
    return HttpCoordinatorBinding(
        mode="live",
        routes=LIVE_HTTP_ROUTES,
        request_model=_FakeLiveRequest,
        error_factory=_live_error_factory,
        current_frame=service.current_frame,
        apply_command=service.apply_command,
        current_presentation=service.current_presentation,
    )


def _coordinator(
    service: _FakeReplayService,
    *,
    mode: Literal["live", "replay"] = "replay",
) -> HttpCoordinatorBinding:
    return HttpCoordinatorBinding(
        mode=mode,
        routes=REPLAY_HTTP_ROUTES if mode == "replay" else LIVE_HTTP_ROUTES,
        request_model=_FakeReplayRequest,
        error_factory=_error_factory,
        current_frame=service.current_frame,
        apply_command=service.apply_command,
        current_timeline=(service.current_timeline if mode == "replay" else None),
        current_presentation=(
            service.current_presentation
            if mode == "replay"
            else service.current_live_presentation
        ),
        current_metric_report=(
            service.current_metric_report if mode == "replay" else None
        ),
    )


@pytest.fixture
def running_replay_server() -> Iterator[
    tuple[DebuggerHTTPServer, _FakeReplayService, Thread]
]:
    service = _FakeReplayService()
    server = create_server(
        service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=_coordinator(service),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, service, thread
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _exchange(
    server: DebuggerHTTPServer,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[HTTPResponse, bytes]:
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response, payload


def _raw_exchange(server: DebuggerHTTPServer, request: bytes) -> bytes:
    with socket.create_connection(
        ("127.0.0.1", server.server_port),
        timeout=5,
    ) as connection:
        connection.settimeout(5)
        connection.sendall(request)
        chunks: list[bytes] = []
        while True:
            chunk = connection.recv(64 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)


def _authorized_headers(**extra: str) -> dict[str, str]:
    return {_TOKEN_HEADER: _TOKEN, **extra}


def _stable_headers(response: HTTPResponse) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name.lower(), value)
            for name, value in response.getheaders()
            if name.lower() != "date"
        )
    )


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        record = cast(dict[str, object], value)
        return set(record) | {
            key for child in record.values() for key in _recursive_keys(child)
        }
    if isinstance(value, list):
        sequence = cast(list[object], value)
        return {key for child in sequence for key in _recursive_keys(child)}
    return set()


def _command_body(
    command_id: str,
    *,
    base_revision: int,
    frame_index: int,
) -> bytes:
    return (
        _FakeReplayRequest(
            client_id="replay-browser",
            command_id=command_id,
            base_revision=base_revision,
            command=_FakeSeekCommand(frame_index=frame_index),
        )
        .model_dump_json()
        .encode()
    )


def _real_loaded_bundle(
    *,
    episode_id: str,
    execution_information_mode: Literal["no_shared_obs", "shared_obs"],
) -> LoadedReplayBundleV1:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        execution_information_mode=execution_information_mode,
        episode_id=episode_id,
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    observer.start(trajectory.frames[0])
    observer.append(trajectory.transitions[0], trajectory.frames[1])
    report = observer.finalize(completion_state="complete")
    bundle = build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=RuntimeProvenanceV1(
            python_version="3.14.0",
            package_version="0.0.0",
            jax_version="0.7.0",
            jaxlib_version="0.7.0",
            numpy_version="2.3.0",
            pydantic_version="2.11.0",
            platform="linux",
            machine="x86_64",
            backend="cpu",
            device="generic-cpu",
            precision="float32",
            environment_count=1,
            batch_shape=(1,),
            policy_execution_included=False,
        ),
    )
    return LoadedReplayBundleV1(
        replay=bundle.replay,
        metric_report_artifact=bundle.metric_report_artifact,
        status="complete",
    )


def _real_replay_binding(service: ReplayViewerService) -> HttpCoordinatorBinding:
    return HttpCoordinatorBinding(
        mode="replay",
        routes=REPLAY_HTTP_ROUTES,
        request_model=ReplayCommandRequestV1,
        error_factory=ReplayApiErrorV1,
        current_frame=service.current_frame,
        apply_command=service.apply_command,
        current_timeline=service.current_timeline,
        current_presentation=service.current_presentation,
        current_metric_report=service.current_metric_report,
    )


def _real_metric_exchange(
    service: ReplayViewerService,
) -> tuple[HTTPResponse, bytes]:
    server = create_server(
        service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=_real_replay_binding(service),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        return _exchange(
            server,
            "GET",
            "/api/replay/metric-report",
            headers=_authorized_headers(),
        )
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_replay_bootstrap_exposes_only_the_replay_product_identity(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server

    response, body = _exchange(server, "GET", "/bootstrap.js")

    assert response.status == HTTPStatus.OK
    assert response.getheader("Content-Type") == "text/javascript; charset=utf-8"
    assert response.getheader("Cache-Control") == "no-store"
    assert body == (
        b"globalThis.__MARL_DEBUGGER_BOOTSTRAP__ = Object.freeze("
        b'{"product_kind":"replay_viewer","schema_version":1});\n'
    )
    assert service.frame_calls == 0
    assert service.timeline_calls == 0


def test_replay_metric_report_preserves_exact_bytes_and_attachment_headers(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server

    response, body = _exchange(
        server,
        "GET",
        "/api/replay/metric-report",
        headers=_authorized_headers(),
    )

    assert response.status == HTTPStatus.OK
    assert response.getheader("Content-Type") == "application/json; charset=utf-8"
    assert response.getheader("Content-Length") == str(len(b'{"canonical":true}'))
    assert response.getheader("Cache-Control") == "no-store"
    assert response.getheader("Content-Disposition") == (
        'attachment; filename="fake-replay.marlbg-metrics.json"'
    )
    assert body == b'{"canonical":true}'
    assert service.metric_report_calls == 1
    assert service.frame_calls == service.timeline_calls == service.command_calls == 0


def test_replay_metric_missing_uses_the_exact_non_disclosing_envelope(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server
    service.metric_report_result = _FakeMetricReportResult("missing", None, None)

    missing, missing_body = _exchange(
        server,
        "GET",
        "/api/replay/metric-report",
        headers=_authorized_headers(),
    )

    assert missing.status == HTTPStatus.NOT_FOUND
    assert json.loads(missing_body) == {
        "schema_version": 1,
        "error_code": "not_found",
        "message": "No metric report is available for this replay.",
        "latest_frame": None,
    }


def test_real_agent_metric_routes_match_oracle_availability() -> None:
    no_shared = _real_loaded_bundle(
        episode_id="privacy-no-shared",
        execution_information_mode="no_shared_obs",
    )
    shared = _real_loaded_bundle(
        episode_id="privacy-shared",
        execution_information_mode="shared_obs",
    )
    cases = (
        no_shared,
        LoadedReplayBundleV1(
            replay=no_shared.replay,
            metric_report_artifact=None,
            status="metric_report_missing",
        ),
        shared,
        LoadedReplayBundleV1(
            replay=shared.replay,
            metric_report_artifact=None,
            status="metric_report_missing",
        ),
    )
    services = tuple(
        ReplayViewerService(
            bundle,
            view_mode="pov",
            pov_global_slot=0,
            viewer_session_id=f"real-agent-metric-route-{index}",
        )
        for index, bundle in enumerate(cases)
    )

    outcomes = tuple(_real_metric_exchange(service) for service in services)

    for index in (0, 2):
        response, body = outcomes[index]
        artifact = cases[index].metric_report_artifact
        assert artifact is not None
        assert response.status == HTTPStatus.OK
        assert body == canonical_metric_report_artifact_json_bytes_v1(artifact)
        episode_id = cases[index].replay.header.context.identity.episode_id
        assert response.getheader("Content-Disposition") == (
            f'attachment; filename="{episode_id}.marlbg-metrics.json"'
        )
    for index in (1, 3):
        response, body = outcomes[index]
        assert response.status == HTTPStatus.NOT_FOUND
        assert json.loads(body) == {
            "schema_version": 1,
            "error_code": "not_found",
            "message": "No metric report is available for this replay.",
            "latest_frame": None,
        }
        assert response.getheader("Content-Disposition") is None
    assert tuple(service.revision for service in services) == (0, 0, 0, 0)


def test_real_service_reserved_suffix_metric_download_is_repeatable() -> None:
    loaded = _real_loaded_bundle(
        episode_id="safe.marlbg-metrics.json",
        execution_information_mode="no_shared_obs",
    )
    artifact = loaded.metric_report_artifact
    assert artifact is not None
    service = ReplayViewerService(
        loaded,
        viewer_session_id="reserved-suffix-http-regression",
    )
    server = create_server(
        service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=_real_replay_binding(service),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        first = _exchange(
            server,
            "GET",
            "/api/replay/metric-report",
            headers=_authorized_headers(),
        )
        second = _exchange(
            server,
            "GET",
            "/api/replay/metric-report",
            headers=_authorized_headers(),
        )
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    expected = canonical_metric_report_artifact_json_bytes_v1(artifact)
    first_response, first_body = first
    second_response, second_body = second
    assert first_response.status == second_response.status == HTTPStatus.OK
    assert first_body == second_body == expected
    assert first_response.getheader("Content-Disposition") == (
        'attachment; filename="safe.marlbg-metrics.json"'
    )
    assert second_response.getheader("Content-Disposition") == (
        first_response.getheader("Content-Disposition")
    )
    assert _stable_headers(first_response) == _stable_headers(second_response)
    assert service.revision == 0


def test_metric_route_checks_origin_and_token_before_callback(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server

    responses = (
        _exchange(server, "GET", "/api/replay/metric-report")[0],
        _exchange(
            server,
            "GET",
            "/api/replay/metric-report",
            headers={_TOKEN_HEADER: "wrong"},
        )[0],
        _exchange(
            server,
            "GET",
            "/api/replay/metric-report",
            headers={"Host": "example.test"},
        )[0],
        _exchange(
            server,
            "GET",
            "/api/replay/metric-report",
            headers=_authorized_headers(Host="example.test"),
        )[0],
        _exchange(
            server,
            "GET",
            "/api/replay/metric-report",
            headers=_authorized_headers(Origin="null"),
        )[0],
        _exchange(
            server,
            "GET",
            "/api/replay/metric-report",
            headers=_authorized_headers(**{"Sec-Fetch-Site": "cross-site"}),
        )[0],
    )

    assert tuple(response.status for response in responses) == (
        HTTPStatus.UNAUTHORIZED,
        HTTPStatus.UNAUTHORIZED,
        HTTPStatus.FORBIDDEN,
        HTTPStatus.FORBIDDEN,
        HTTPStatus.FORBIDDEN,
        HTTPStatus.FORBIDDEN,
    )
    assert service.metric_report_calls == 0


@pytest.mark.parametrize(
    "path",
    (
        "/api/replay/metric-report?",
        "/api/replay/metric-report#",
        "/api/replay/metric-report?download=1",
        "/api/replay/metric-report#fragment",
        "/api/replay/metric-report?download=1#fragment",
    ),
)
def test_metric_route_rejects_every_raw_query_or_fragment_target(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
    path: str,
) -> None:
    server, service, _ = running_replay_server

    response, body = _exchange(
        server,
        "GET",
        path,
        headers=_authorized_headers(),
    )

    assert response.status == HTTPStatus.NOT_FOUND
    assert json.loads(body)["error_code"] == "not_found"
    assert service.metric_report_calls == 0


def test_metric_route_rejects_absolute_form_raw_target_before_callback(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server
    raw_target = f"http://{server.expected_host}/api/replay/metric-report"
    response = _raw_exchange(
        server,
        (
            f"GET {raw_target} HTTP/1.1\r\n"
            f"Host: {server.expected_host}\r\n"
            f"{_TOKEN_HEADER}: {_TOKEN}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii"),
    )

    _, _, body = response.partition(b"\r\n\r\n")
    assert response.startswith(b"HTTP/1.1 404 ")
    assert json.loads(body)["error_code"] == "not_found"
    assert service.metric_report_calls == 0


def test_authenticated_metric_post_is_method_not_allowed_without_callback(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server

    unauthorized, _ = _exchange(server, "POST", "/api/replay/metric-report")
    response, body = _exchange(
        server,
        "POST",
        "/api/replay/metric-report",
        headers=_authorized_headers(),
    )

    assert unauthorized.status == HTTPStatus.UNAUTHORIZED
    assert response.status == HTTPStatus.METHOD_NOT_ALLOWED
    assert json.loads(body)["error_code"] == "method_not_allowed"
    assert service.metric_report_calls == 0


@pytest.mark.parametrize(
    "forged",
    (
        object(),
        _FakeMetricReportResult("unknown", None, None),
        _FakeMetricReportResult(
            "available",
            cast(bytes, bytearray(b"{}")),
            "safe.marlbg-metrics.json",
        ),
        _FakeMetricReportResult("available", b"{}", "../unsafe.marlbg-metrics.json"),
        _FakeMetricReportResult("available", b"{}", "unsafe\\path.marlbg-metrics.json"),
        _FakeMetricReportResult(
            "available",
            b"{}",
            "unsafe\r\nX-Forged-Header:true.marlbg-metrics.json",
        ),
        _FakeMetricReportResult("available", b"{}", 'unsafe".marlbg-metrics.json'),
        _FakeMetricReportResult("available", b"{}", "unsafe;.marlbg-metrics.json"),
        _FakeMetricReportResult("available", b"{}", "é.marlbg-metrics.json"),
        _FakeMetricReportResult("available", b"{}", "unsafe.json"),
        _FakeMetricReportResult(
            "available",
            b"{}",
            "unsafe.marlbg-metrics.json.marlbg-metrics.json",
        ),
        _FakeMetricReportResult("available", b"{}", "unsafe.marlbg-metrics.json.evil"),
        _FakeMetricReportResult("missing", b"{}", None),
        _FakeMetricReportResult("forbidden", None, "unsafe.marlbg-metrics.json"),
    ),
)
def test_forged_metric_results_fail_generically_before_attachment_headers(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
    forged: object,
) -> None:
    server, service, _ = running_replay_server
    service.metric_report_result = forged

    response, body = _exchange(
        server,
        "GET",
        "/api/replay/metric-report",
        headers=_authorized_headers(),
    )

    assert response.status == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.getheader("Content-Disposition") is None
    assert json.loads(body) == {
        "schema_version": 1,
        "error_code": "internal_error",
        "message": "The debugger could not process this request.",
        "latest_frame": None,
    }
    assert service.metric_report_calls == 1


def test_server_import_is_core_jax_and_protocol_family_free() -> None:
    code = """
import sys
import scripts.dev.visual_debugger.server

forbidden = (
    'jax',
    'jaxlib',
    'numpy',
    'marl_battlegrounds.core',
    'marl_battlegrounds.evaluation.capture',
    'scripts.dev.visual_debugger.control',
    'scripts.dev.visual_debugger.protocol',
    'scripts.dev.visual_debugger.presentation',
    'scripts.dev.visual_debugger.presentation_protocol',
    'scripts.dev.visual_debugger.replay_protocol',
    'scripts.dev.visual_debugger.replay_service',
    'scripts.dev.visual_debugger.scenarios',
    'scripts.dev.visual_debugger.service',
    'marl_battlegrounds.rendering.authorized_presentation',
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden)
)
assert loaded == [], loaded
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_live_request_and_replay_replacement_are_serialized_and_coherent() -> None:
    live_service = _BlockingLiveService()
    server = create_server(
        live_service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=_live_coordinator(live_service),
    )
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    expected = server.coordinator_snapshot()
    original_origin = server.expected_origin
    original_host = server.expected_host
    original_token = server.capability_token
    live_metric, live_metric_body = _exchange(
        server,
        "GET",
        "/api/replay/metric-report",
        headers=_authorized_headers(),
    )
    old_responses: list[tuple[HTTPResponse, bytes]] = []
    old_request = Thread(
        target=lambda: old_responses.append(
            _exchange(
                server,
                "POST",
                "/api/command",
                body=b'{"schema_version":2,"command_type":"advance"}',
                headers=_authorized_headers(**{"Content-Type": "application/json"}),
            )
        ),
        daemon=True,
    )
    replay_service = _FakeReplayService()
    replay_binding = _coordinator(replay_service)
    replacement = HttpCoordinatorReplacement(
        service=replay_service,
        binding=replay_binding,
    )
    swap_started = Event()
    swap_finished = Event()
    swap_results: list[bool] = []

    def install_replay() -> None:
        swap_started.set()
        swap_results.append(
            server.install_replay_coordinator(
                expected=expected,
                replacement=replacement,
            )
        )
        swap_finished.set()

    swap_thread = Thread(target=install_replay, daemon=True)
    try:
        old_request.start()
        assert live_service.entered.wait(timeout=2)
        swap_thread.start()
        assert swap_started.wait(timeout=2)
        assert not swap_finished.wait(timeout=0.1)

        live_service.release.set()
        old_request.join(timeout=2)
        swap_thread.join(timeout=2)
        assert not old_request.is_alive()
        assert not swap_thread.is_alive()

        old_response, old_body = old_responses[0]
        replay_frame, replay_frame_body = _exchange(
            server,
            "GET",
            "/api/frame",
            headers=_authorized_headers(),
        )
        replay_timeline, _ = _exchange(
            server,
            "GET",
            "/api/replay/timeline",
            headers=_authorized_headers(),
        )
        replay_presentation, replay_presentation_body = _exchange(
            server,
            "GET",
            "/api/presentation/frame",
            headers=_authorized_headers(),
        )
        replay_metric, replay_metric_body = _exchange(
            server,
            "GET",
            "/api/replay/metric-report",
            headers=_authorized_headers(),
        )
        replay_bootstrap, replay_bootstrap_body = _exchange(
            server,
            "GET",
            "/bootstrap.js",
        )
        removed_live_route, removed_live_body = _exchange(
            server,
            "POST",
            "/api/command",
            body=b"{}",
            headers=_authorized_headers(**{"Content-Type": "application/json"}),
        )

        assert old_response.status == HTTPStatus.OK
        assert live_metric.status == HTTPStatus.NOT_FOUND
        assert json.loads(live_metric_body)["error_code"] == "not_found"
        assert json.loads(old_body) == {
            "schema_version": 2,
            "result": "applied",
            "frame": {
                "schema_version": 2,
                "frame_kind": "live_debugger",
                "revision": 1,
            },
        }
        assert isinstance(live_service.received_request, _FakeLiveRequest)
        assert swap_results == [True]
        assert (
            replay_frame.status
            == replay_timeline.status
            == replay_presentation.status
            == HTTPStatus.OK
        )
        assert replay_metric.status == HTTPStatus.OK
        assert replay_metric_body == b'{"canonical":true}'
        assert replay_service.metric_report_calls == 1
        assert json.loads(replay_presentation_body)["presentation_kind"] == (
            "replay_oracle"
        )
        assert replay_bootstrap.status == HTTPStatus.OK
        assert replay_bootstrap_body == (
            b"globalThis.__MARL_DEBUGGER_BOOTSTRAP__ = Object.freeze("
            b'{"product_kind":"replay_viewer","schema_version":1});\n'
        )
        assert json.loads(replay_frame_body)["schema_version"] == 1
        assert removed_live_route.status == HTTPStatus.NOT_FOUND
        assert json.loads(removed_live_body)["schema_version"] == 1
        active = server.coordinator_snapshot()
        assert active.generation == 1
        assert active.service is replay_service
        assert active.binding is replay_binding
        assert server.debugger_service is replay_service
        assert server.coordinator is replay_binding
        assert server.expected_origin == original_origin
        assert server.expected_host == original_host
        assert server.capability_token == original_token
    finally:
        live_service.release.set()
        if server_thread.is_alive():
            server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        old_request.join(timeout=2)
        swap_thread.join(timeout=2)


def test_in_flight_live_presentation_pins_old_binding_until_replay_cas() -> None:
    live_service = _BlockingLiveService(block_presentation=True)
    server = create_server(
        live_service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=_live_coordinator(live_service),
    )
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    expected = server.coordinator_snapshot()
    replay_service = _FakeReplayService()
    replay_binding = _coordinator(replay_service)
    replacement = HttpCoordinatorReplacement(
        service=replay_service,
        binding=replay_binding,
    )
    old_responses: list[tuple[HTTPResponse, bytes]] = []
    old_request = Thread(
        target=lambda: old_responses.append(
            _exchange(
                server,
                "GET",
                "/api/presentation/frame",
                headers=_authorized_headers(),
            )
        ),
        daemon=True,
    )
    swap_finished = Event()
    swap_results: list[bool] = []

    def install_replay() -> None:
        swap_results.append(
            server.install_replay_coordinator(
                expected=expected,
                replacement=replacement,
            )
        )
        swap_finished.set()

    swap_thread = Thread(target=install_replay, daemon=True)
    try:
        old_request.start()
        assert live_service.entered.wait(timeout=2)
        swap_thread.start()
        assert not swap_finished.wait(timeout=0.1)

        live_service.release.set()
        old_request.join(timeout=2)
        swap_thread.join(timeout=2)
        assert not old_request.is_alive()
        assert not swap_thread.is_alive()
        assert swap_results == [True]

        old_response, old_body = old_responses[0]
        assert old_response.status == HTTPStatus.OK
        assert json.loads(old_body)["presentation_kind"] == "live_oracle"
        assert live_service.presentation_calls == 1

        next_response, next_body = _exchange(
            server,
            "GET",
            "/api/presentation/frame",
            headers=_authorized_headers(),
        )
        assert next_response.status == HTTPStatus.OK
        assert json.loads(next_body)["presentation_kind"] == "replay_oracle"
        assert replay_service.presentation_calls == 1
        active = server.coordinator_snapshot()
        assert active.generation == 1
        assert active.service is replay_service
        assert active.binding is replay_binding
    finally:
        live_service.release.set()
        if server_thread.is_alive():
            server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        old_request.join(timeout=2)
        swap_thread.join(timeout=2)


def test_live_request_can_reentrantly_install_replay_before_live_response() -> None:
    live_service = _BlockingLiveService()
    live_service.release.set()
    live_binding = _live_coordinator(live_service)
    router = HttpCoordinatorRouter(
        service=live_service,
        binding=live_binding,
    )
    server = create_server(
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator_router=router,
    )
    expected = router.snapshot()
    replay_service = _FakeReplayService()
    replay_binding = _coordinator(replay_service)
    replacement = HttpCoordinatorReplacement(
        service=replay_service,
        binding=replay_binding,
    )
    install_results: list[bool] = []

    def install_from_live_request() -> None:
        install_results.append(
            router.compare_and_swap(
                expected=expected,
                replacement=replacement,
            )
        )

    live_service.on_apply = install_from_live_request
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        handoff_response, handoff_body = _exchange(
            server,
            "POST",
            "/api/command",
            body=b'{"schema_version":2,"command_type":"advance"}',
            headers=_authorized_headers(**{"Content-Type": "application/json"}),
        )
        replay_timeline, _ = _exchange(
            server,
            "GET",
            "/api/replay/timeline",
            headers=_authorized_headers(),
        )

        assert install_results == [True]
        assert handoff_response.status == HTTPStatus.OK
        assert json.loads(handoff_body)["schema_version"] == 2
        assert json.loads(handoff_body)["frame"]["frame_kind"] == "live_debugger"
        assert replay_timeline.status == HTTPStatus.OK
        active = server.coordinator_snapshot()
        assert active.generation == 1
        assert active.service is replay_service
        assert active.binding is replay_binding
        assert server.coordinator_router is router
    finally:
        if server_thread.is_alive():
            server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


def test_preconstructed_router_requires_exact_coherent_redundant_inputs() -> None:
    live_service = _BlockingLiveService()
    live_binding = _live_coordinator(live_service)
    router = HttpCoordinatorRouter(
        service=live_service,
        binding=live_binding,
    )
    server = create_server(
        live_service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=live_binding,
        coordinator_router=router,
    )
    try:
        assert server.coordinator_router is router
        assert server.debugger_service is live_service
        assert server.coordinator is live_binding
    finally:
        server.server_close()

    with pytest.raises(ValueError, match="supplied debugger service"):
        create_server(
            _BlockingLiveService(),
            asset_root=_ASSET_ROOT,
            port=0,
            coordinator_router=router,
        )
    with pytest.raises(ValueError, match="supplied coordinator binding"):
        create_server(
            live_service,
            asset_root=_ASSET_ROOT,
            port=0,
            coordinator=_live_coordinator(live_service),
            coordinator_router=router,
        )
    with pytest.raises(TypeError, match="exact HttpCoordinatorRouter"):
        create_server(
            asset_root=_ASSET_ROOT,
            port=0,
            coordinator_router=cast(HttpCoordinatorRouter, object()),
        )
    with pytest.raises(TypeError, match="service is required"):
        create_server(
            asset_root=_ASSET_ROOT,
            port=0,
        )


def test_serve_browser_debugger_accepts_router_only_launch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    replay_service = _FakeReplayService()
    router = HttpCoordinatorRouter(
        service=replay_service,
        binding=_coordinator(replay_service),
    )

    def interrupt_server(
        server: DebuggerHTTPServer,
        *,
        poll_interval: float,
    ) -> None:
        assert server.coordinator_router is router
        del poll_interval
        raise KeyboardInterrupt

    monkeypatch.setattr(DebuggerHTTPServer, "serve_forever", interrupt_server)

    result = serve_browser_debugger(
        asset_root=_ASSET_ROOT,
        port=0,
        open_browser=False,
        coordinator_router=router,
    )

    assert result == 0
    assert "MARL-BattleGrounds Replay Viewer stopped." in capsys.readouterr().out


def test_shutdown_title_uses_the_active_product_after_live_to_replay_handoff(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    live_service = _BlockingLiveService()
    live_binding = _live_coordinator(live_service)
    router = HttpCoordinatorRouter(service=live_service, binding=live_binding)
    replay_service = _FakeReplayService()
    replacement = HttpCoordinatorReplacement(
        service=replay_service,
        binding=_coordinator(replay_service),
    )

    def install_then_interrupt(
        server: DebuggerHTTPServer,
        *,
        poll_interval: float,
    ) -> None:
        del poll_interval
        assert server.install_replay_coordinator(
            expected=router.snapshot(),
            replacement=replacement,
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(
        DebuggerHTTPServer,
        "serve_forever",
        install_then_interrupt,
    )

    result = serve_browser_debugger(
        asset_root=_ASSET_ROOT,
        port=0,
        open_browser=False,
        coordinator_router=router,
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "MARL-BattleGrounds Combat Debugger: http://127.0.0.1:" in output
    assert "MARL-BattleGrounds Replay Viewer stopped." in output


def test_in_flight_error_uses_the_pinned_protocol_family() -> None:
    live_service = _BlockingLiveService(block_frame_with_error=True)
    server = create_server(
        live_service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=_live_coordinator(live_service),
    )
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    expected = server.coordinator_snapshot()
    old_responses: list[tuple[HTTPResponse, bytes]] = []
    old_request = Thread(
        target=lambda: old_responses.append(
            _exchange(
                server,
                "GET",
                "/api/frame",
                headers=_authorized_headers(),
            )
        ),
        daemon=True,
    )
    replay_service = _FakeReplayService()
    replacement = HttpCoordinatorReplacement(
        service=replay_service,
        binding=_coordinator(replay_service),
    )
    swap_finished = Event()

    def install_replay() -> None:
        assert server.install_replay_coordinator(
            expected=expected,
            replacement=replacement,
        )
        swap_finished.set()

    swap_thread = Thread(target=install_replay, daemon=True)
    try:
        old_request.start()
        assert live_service.entered.wait(timeout=2)
        swap_thread.start()
        assert not swap_finished.wait(timeout=0.1)

        live_service.release.set()
        old_request.join(timeout=2)
        swap_thread.join(timeout=2)
        assert not old_request.is_alive()
        assert not swap_thread.is_alive()

        old_response, old_body = old_responses[0]
        replay_error, replay_error_body = _exchange(
            server,
            "GET",
            "/api/frame?query=forbidden",
        )

        assert old_response.status == HTTPStatus.INTERNAL_SERVER_ERROR
        assert json.loads(old_body) == {
            "schema_version": 2,
            "error_code": "internal_error",
            "message": "The debugger could not process this request.",
            "latest_frame": None,
        }
        assert replay_error.status == HTTPStatus.NOT_FOUND
        assert json.loads(replay_error_body)["schema_version"] == 1
        assert json.loads(replay_error_body)["error_code"] == "not_found"
    finally:
        live_service.release.set()
        if server_thread.is_alive():
            server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        old_request.join(timeout=2)
        swap_thread.join(timeout=2)


def test_coordinator_cas_failure_never_partially_swaps_active_pair() -> None:
    live_service = _BlockingLiveService()
    server = create_server(
        live_service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=_live_coordinator(live_service),
    )
    expected = server.coordinator_snapshot()
    first_replay_service = _FakeReplayService()
    first_replay_binding = _coordinator(first_replay_service)
    first_replacement = HttpCoordinatorReplacement(
        service=first_replay_service,
        binding=first_replay_binding,
    )
    second_replay_service = _FakeReplayService()
    second_replacement = HttpCoordinatorReplacement(
        service=second_replay_service,
        binding=_coordinator(second_replay_service),
    )
    try:
        assert server.install_replay_coordinator(
            expected=expected,
            replacement=first_replacement,
        )
        installed = server.coordinator_snapshot()

        assert not server.install_replay_coordinator(
            expected=expected,
            replacement=second_replacement,
        )
        after_stale_cas = server.coordinator_snapshot()
        assert after_stale_cas is installed
        assert after_stale_cas.service is first_replay_service
        assert after_stale_cas.binding is first_replay_binding

        replacement_live_service = _BlockingLiveService()
        replacement_live = HttpCoordinatorReplacement(
            service=replacement_live_service,
            binding=_live_coordinator(replacement_live_service),
        )
        with pytest.raises(ValueError, match="monotonic live-to-replay"):
            server.install_replay_coordinator(
                expected=installed,
                replacement=replacement_live,
            )

        after_forbidden_swap = server.coordinator_snapshot()
        assert after_forbidden_swap is installed
        assert after_forbidden_swap.generation == 1
        assert after_forbidden_swap.service is first_replay_service
        assert after_forbidden_swap.binding is first_replay_binding
    finally:
        server.server_close()


def test_binding_rejects_route_or_timeline_cross_mode_configuration() -> None:
    service = _FakeReplayService()
    with pytest.raises(ValueError, match="exact HTTP route set"):
        HttpCoordinatorBinding(
            mode="replay",
            routes=LIVE_HTTP_ROUTES,
            request_model=_FakeReplayRequest,
            error_factory=_error_factory,
            current_frame=service.current_frame,
            apply_command=service.apply_command,
            current_presentation=service.current_presentation,
            current_timeline=service.current_timeline,
            current_metric_report=service.current_metric_report,
        )
    with pytest.raises(ValueError, match="cannot expose a replay timeline"):
        HttpCoordinatorBinding(
            mode="live",
            routes=LIVE_HTTP_ROUTES,
            request_model=_FakeReplayRequest,
            error_factory=_error_factory,
            current_frame=service.current_frame,
            apply_command=service.apply_command,
            current_presentation=service.current_live_presentation,
            current_timeline=service.current_timeline,
        )
    with pytest.raises(ValueError, match="requires a timeline"):
        HttpCoordinatorBinding(
            mode="replay",
            routes=REPLAY_HTTP_ROUTES,
            request_model=_FakeReplayRequest,
            error_factory=_error_factory,
            current_frame=service.current_frame,
            apply_command=service.apply_command,
            current_presentation=service.current_presentation,
            current_metric_report=service.current_metric_report,
        )
    with pytest.raises(ValueError, match="requires a metric-report"):
        HttpCoordinatorBinding(
            mode="replay",
            routes=REPLAY_HTTP_ROUTES,
            request_model=_FakeReplayRequest,
            error_factory=_error_factory,
            current_frame=service.current_frame,
            apply_command=service.apply_command,
            current_presentation=service.current_presentation,
            current_timeline=service.current_timeline,
        )
    with pytest.raises(ValueError, match="cannot expose replay metrics"):
        HttpCoordinatorBinding(
            mode="live",
            routes=LIVE_HTTP_ROUTES,
            request_model=_FakeReplayRequest,
            error_factory=_error_factory,
            current_frame=service.current_frame,
            apply_command=service.apply_command,
            current_presentation=service.current_live_presentation,
            current_metric_report=service.current_metric_report,
        )
    with pytest.raises(TypeError, match="current_presentation"):
        HttpCoordinatorBinding(  # pyright: ignore[reportCallIssue]
            mode="replay",
            routes=REPLAY_HTTP_ROUTES,
            request_model=_FakeReplayRequest,
            error_factory=_error_factory,
            current_frame=service.current_frame,
            apply_command=service.apply_command,
            current_timeline=service.current_timeline,
        )
    with pytest.raises(TypeError, match="current_presentation must be callable"):
        HttpCoordinatorBinding(
            mode="live",
            routes=LIVE_HTTP_ROUTES,
            request_model=_FakeReplayRequest,
            error_factory=_error_factory,
            current_frame=service.current_frame,
            apply_command=service.apply_command,
            current_presentation=None,  # pyright: ignore[reportArgumentType]
        )
    with pytest.raises(ValueError, match="exact HTTP route set"):
        HttpCoordinatorBinding(
            mode="live",
            routes=HttpRouteSet(
                frame="/api/frame",
                command="/api/replay/command",
                timeline=None,
                presentation="/api/presentation/frame",
                metric_report=None,
            ),
            request_model=_FakeReplayRequest,
            error_factory=_error_factory,
            current_frame=service.current_frame,
            apply_command=service.apply_command,
            current_presentation=service.current_live_presentation,
        )


def test_replay_frame_presentation_and_timeline_are_authenticated_bounded_models(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server
    missing, missing_body = _exchange(server, "GET", "/api/frame")
    frame, frame_body = _exchange(
        server,
        "GET",
        "/api/frame",
        headers=_authorized_headers(),
    )
    presentation, presentation_body = _exchange(
        server,
        "GET",
        "/api/presentation/frame",
        headers=_authorized_headers(),
    )
    timeline, timeline_body = _exchange(
        server,
        "GET",
        "/api/replay/timeline",
        headers=_authorized_headers(),
    )

    assert missing.status == HTTPStatus.UNAUTHORIZED
    assert json.loads(missing_body)["error_code"] == "unauthorized"
    assert frame.status == presentation.status == timeline.status == HTTPStatus.OK
    assert frame.getheader("Cache-Control") == "no-store"
    assert presentation.getheader("Cache-Control") == "no-store"
    assert timeline.getheader("Cache-Control") == "no-store"
    assert json.loads(frame_body) == {
        "schema_version": 1,
        "frame_kind": "researcher_replay_viewer",
        "revision": 0,
        "frame_index": 0,
    }
    assert json.loads(timeline_body) == {
        "schema_version": 1,
        "timeline_kind": "researcher_replay_timeline",
        "current_frame_index": 0,
        "frame_indices": [0, 1, 2],
    }
    assert json.loads(presentation_body) == {
        "schema_version": 1,
        "presentation_kind": "replay_oracle",
        "source_revision": 0,
        "source_frame_index": 0,
    }
    serialized = frame_body + presentation_body + timeline_body
    assert b"must-never-cross-http" not in serialized
    assert b"hidden_events" not in serialized
    assert b"metric_report" not in serialized
    assert (
        service.frame_calls == service.presentation_calls == service.timeline_calls == 1
    )


def test_actual_shared_replay_http_keeps_fog_private_beside_artifact_facts(
    tmp_path: Path,
) -> None:
    artifacts = export_artifacts(tmp_path / "standalone-shared-artifacts")
    service = ReplayViewerService(
        load_replay_bundle_v1(
            Path(artifacts["shared"]),
            require_metric_report=True,
        ),
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="server-shared-private",
    )
    binding = HttpCoordinatorBinding(
        mode="replay",
        routes=REPLAY_HTTP_ROUTES,
        request_model=ReplayCommandRequestV1,
        error_factory=ReplayApiErrorV1,
        current_frame=service.current_frame,
        apply_command=service.apply_command,
        current_timeline=service.current_timeline,
        current_presentation=service.current_presentation,
        current_metric_report=service.current_metric_report,
    )
    server = create_server(
        service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=binding,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    headers = _authorized_headers(**{"Content-Type": "application/json"})
    request = ReplayCommandRequestV1(
        client_id="shared-http-client",
        command_id="shared-next",
        base_revision=0,
        command=ReplayNextFrameCommandV1(),
    )
    stale_request = ReplayCommandRequestV1(
        client_id="shared-http-client",
        command_id="shared-stale",
        base_revision=0,
        command=ReplayNextFrameCommandV1(),
    )
    try:
        frame_response, frame_body = _exchange(
            server,
            "GET",
            "/api/frame",
            headers=_authorized_headers(),
        )
        timeline_response, timeline_body = _exchange(
            server,
            "GET",
            "/api/replay/timeline",
            headers=_authorized_headers(),
        )
        applied_response, applied_body = _exchange(
            server,
            "POST",
            "/api/replay/command",
            body=request.model_dump_json().encode(),
            headers=headers,
        )
        duplicate_response, duplicate_body = _exchange(
            server,
            "POST",
            "/api/replay/command",
            body=request.model_dump_json().encode(),
            headers=headers,
        )
        stale_response, stale_body = _exchange(
            server,
            "POST",
            "/api/replay/command",
            body=stale_request.model_dump_json().encode(),
            headers=headers,
        )
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert frame_response.status == timeline_response.status == HTTPStatus.OK
    assert applied_response.status == duplicate_response.status == HTTPStatus.OK
    assert stale_response.status == HTTPStatus.CONFLICT
    assert frame_response.getheader("Cache-Control") == "no-store"
    assert timeline_response.getheader("Cache-Control") == "no-store"

    frame = SharedObsAgentPovReplayViewerFrameV1.model_validate_json(frame_body)
    timeline = SharedObsAgentPovReplayTimelineV1.model_validate_json(timeline_body)
    applied = ReplayCommandResponseV1.model_validate_json(applied_body)
    duplicate = ReplayCommandResponseV1.model_validate_json(duplicate_body)
    stale = ReplayApiErrorV1.model_validate_json(stale_body)
    assert type(applied.frame) is SharedObsAgentPovReplayViewerFrameV1
    assert type(duplicate.frame) is SharedObsAgentPovReplayViewerFrameV1
    assert type(stale.latest_frame) is SharedObsAgentPovReplayViewerFrameV1
    assert frame.cursor.frame_index == 0
    assert timeline.rows[0].recipient_frame_id == frame.recipient_frame_id
    assert applied.frame.cursor.frame_index == 1
    assert duplicate.frame == applied.frame
    assert stale.latest_frame == applied.frame

    bodies = (frame_body, timeline_body, applied_body, duplicate_body, stale_body)
    payloads = [json.loads(body) for body in bodies]
    fact_roots = (
        payloads[0].pop("artifact_facts"),
        payloads[2]["frame"].pop("artifact_facts"),
        payloads[3]["frame"].pop("artifact_facts"),
        payloads[4]["latest_frame"].pop("artifact_facts"),
    )
    assert all(facts == fact_roots[0] for facts in fact_roots)
    assert fact_roots[0]["artifact_summary"]["metric_report_availability"] == (
        "available"
    )
    forbidden_keys = {
        "global_slot",
        "metric_report_availability",
        "observation_materialization",
        "processing",
        "projection",
        "replay_reference",
        "selected_global_slot",
        "source_material_frame_id",
    }
    for payload in payloads:
        assert _recursive_keys(payload).isdisjoint(forbidden_keys)
    serialized_battlefield = json.dumps(payloads, sort_keys=True)
    assert "shared_obs_source_material" not in serialized_battlefield
    assert "must-never-cross-http" not in serialized_battlefield


def test_replay_presentation_unavailable_is_typed_and_non_disclosing(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server
    service.presentation_available = False

    response, body = _exchange(
        server,
        "GET",
        "/api/presentation/frame",
        headers=_authorized_headers(),
    )

    assert response.status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.getheader("Cache-Control") == "no-store"
    assert json.loads(body) == {
        "schema_version": 1,
        "error_code": "audience_unavailable",
        "message": "Authorized presentation is unavailable for the active audience.",
    }
    assert b"latest_frame" not in body
    assert b"must-never-cross-http" not in body
    assert service.presentation_calls == 1
    assert service.frame_calls == service.timeline_calls == service.command_calls == 0


def test_command_between_raw_and_presentation_gets_creates_detectable_join_race(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server
    raw_response, raw_body = _exchange(
        server,
        "GET",
        "/api/frame",
        headers=_authorized_headers(),
    )
    command_response, _ = _exchange(
        server,
        "POST",
        "/api/replay/command",
        body=_command_body("between-gets", base_revision=0, frame_index=1),
        headers=_authorized_headers(**{"Content-Type": "application/json"}),
    )
    presentation_response, presentation_body = _exchange(
        server,
        "GET",
        "/api/presentation/frame",
        headers=_authorized_headers(),
    )

    raw = json.loads(raw_body)
    presentation = json.loads(presentation_body)
    assert raw_response.status == command_response.status == HTTPStatus.OK
    assert presentation_response.status == HTTPStatus.OK
    assert (raw["revision"], raw["frame_index"]) == (0, 0)
    assert (
        presentation["source_revision"],
        presentation["source_frame_index"],
    ) == (1, 1)
    assert raw["revision"] != presentation["source_revision"]
    assert service.frame_calls == service.presentation_calls == 1
    assert service.command_calls == service.cursor_mutations == 1


def test_replay_command_parser_rejects_bad_requests_before_service_entry(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server
    malformed, _ = _exchange(
        server,
        "POST",
        "/api/replay/command",
        body=b"{",
        headers=_authorized_headers(**{"Content-Type": "application/json"}),
    )
    invalid, invalid_body = _exchange(
        server,
        "POST",
        "/api/replay/command",
        body=b'{"schema_version":1}',
        headers=_authorized_headers(**{"Content-Type": "application/json"}),
    )
    wrong_media, _ = _exchange(
        server,
        "POST",
        "/api/replay/command",
        body=b"{}",
        headers=_authorized_headers(**{"Content-Type": "text/plain"}),
    )

    assert malformed.status == HTTPStatus.BAD_REQUEST
    assert invalid.status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert b"_FakeReplayRequest" in invalid_body
    assert wrong_media.status == HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    assert service.command_calls == 0
    assert service.cursor_mutations == 0


def test_replay_command_idempotency_and_error_statuses_are_preserved(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server
    headers = _authorized_headers(**{"Content-Type": "application/json"})
    body = _command_body("seek-once", base_revision=0, frame_index=1)
    applied, applied_body = _exchange(
        server,
        "POST",
        "/api/replay/command",
        body=body,
        headers=headers,
    )
    duplicate, duplicate_body = _exchange(
        server,
        "POST",
        "/api/replay/command",
        body=body,
        headers=headers,
    )
    conflict, conflict_body = _exchange(
        server,
        "POST",
        "/api/replay/command",
        body=_command_body("seek-once", base_revision=1, frame_index=2),
        headers=headers,
    )
    stale, stale_body = _exchange(
        server,
        "POST",
        "/api/replay/command",
        body=_command_body("stale", base_revision=0, frame_index=2),
        headers=headers,
    )
    invalid, invalid_body = _exchange(
        server,
        "POST",
        "/api/replay/command",
        body=_command_body("outside", base_revision=1, frame_index=99),
        headers=headers,
    )

    assert applied.status == duplicate.status == HTTPStatus.OK
    assert json.loads(applied_body)["result"] == "applied"
    assert json.loads(duplicate_body)["result"] == "duplicate"
    assert conflict.status == stale.status == HTTPStatus.CONFLICT
    assert json.loads(conflict_body)["error_code"] == "command_id_conflict"
    assert json.loads(stale_body)["error_code"] == "stale_revision"
    assert invalid.status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert json.loads(invalid_body)["error_code"] == "invalid_cursor"
    assert service.cursor_mutations == 1
    assert service.frame_index == 1


def test_replay_routes_reuse_host_origin_token_and_size_protections(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server
    bad_host, _ = _exchange(
        server,
        "GET",
        "/api/replay/timeline",
        headers=_authorized_headers(Host="example.invalid"),
    )
    bad_origin, _ = _exchange(
        server,
        "GET",
        "/api/replay/timeline",
        headers=_authorized_headers(Origin="null"),
    )
    cross_site, _ = _exchange(
        server,
        "GET",
        "/api/replay/timeline",
        headers=_authorized_headers(**{"Sec-Fetch-Site": "cross-site"}),
    )
    unauthorized, _ = _exchange(
        server,
        "POST",
        "/api/replay/command",
        body=b"{}",
        headers={"Content-Type": "application/json"},
    )
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.putrequest("POST", "/api/replay/command", skip_host=True)
    connection.putheader("Host", server.expected_host)
    connection.putheader(_TOKEN_HEADER, _TOKEN)
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", str(64 * 1024 + 1))
    connection.endheaders()
    oversized = connection.getresponse()
    oversized.read()
    connection.close()

    assert bad_host.status == bad_origin.status == cross_site.status == 403
    assert unauthorized.status == HTTPStatus.UNAUTHORIZED
    assert oversized.status == HTTPStatus.CONTENT_TOO_LARGE
    assert oversized.getheader("Connection") == "close"
    assert service.timeline_calls == service.command_calls == 0


def test_presentation_get_reuses_auth_origin_query_and_method_boundaries(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    server, service, _ = running_replay_server

    unauthorized, _ = _exchange(server, "GET", "/api/presentation/frame")
    bad_host, _ = _exchange(
        server,
        "GET",
        "/api/presentation/frame",
        headers=_authorized_headers(Host="example.invalid"),
    )
    bad_origin, _ = _exchange(
        server,
        "GET",
        "/api/presentation/frame",
        headers=_authorized_headers(Origin="null"),
    )
    query, _ = _exchange(
        server,
        "GET",
        "/api/presentation/frame?forbidden=1",
        headers=_authorized_headers(),
    )
    post, _ = _exchange(
        server,
        "POST",
        "/api/presentation/frame",
        body=b"{}",
        headers=_authorized_headers(**{"Content-Type": "application/json"}),
    )
    head, _ = _exchange(
        server,
        "HEAD",
        "/api/presentation/frame",
        headers=_authorized_headers(),
    )

    assert unauthorized.status == HTTPStatus.UNAUTHORIZED
    assert bad_host.status == bad_origin.status == HTTPStatus.FORBIDDEN
    assert query.status == post.status == HTTPStatus.NOT_FOUND
    assert head.status == HTTPStatus.METHOD_NOT_ALLOWED
    assert service.presentation_calls == 0


def test_live_and_replay_routes_are_mode_isolated(
    running_replay_server: tuple[DebuggerHTTPServer, _FakeReplayService, Thread],
) -> None:
    replay_server, replay_service, _ = running_replay_server
    replay_live_command, _ = _exchange(
        replay_server,
        "POST",
        "/api/command",
        body=b"{}",
        headers=_authorized_headers(**{"Content-Type": "application/json"}),
    )
    invented_replay_frame, _ = _exchange(
        replay_server,
        "GET",
        "/api/replay/frame",
        headers=_authorized_headers(),
    )

    live_service = _FakeReplayService()
    live_server = create_server(
        live_service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=_coordinator(live_service, mode="live"),
    )
    live_thread = Thread(target=live_server.serve_forever, daemon=True)
    live_thread.start()
    try:
        replay_timeline, _ = _exchange(
            live_server,
            "GET",
            "/api/replay/timeline",
            headers=_authorized_headers(),
        )
        replay_command, _ = _exchange(
            live_server,
            "POST",
            "/api/replay/command",
            body=b"{}",
            headers=_authorized_headers(**{"Content-Type": "application/json"}),
        )
        live_presentation, live_presentation_body = _exchange(
            live_server,
            "GET",
            "/api/presentation/frame",
            headers=_authorized_headers(),
        )
    finally:
        live_server.shutdown()
        live_server.server_close()
        live_thread.join(timeout=2)

    assert replay_live_command.status == HTTPStatus.NOT_FOUND
    assert invented_replay_frame.status == HTTPStatus.NOT_FOUND
    assert replay_timeline.status == replay_command.status == HTTPStatus.NOT_FOUND
    assert live_presentation.status == HTTPStatus.OK
    assert json.loads(live_presentation_body)["presentation_kind"] == "live_oracle"
    assert replay_service.command_calls == 0
    assert live_service.timeline_calls == live_service.command_calls == 0
    assert live_service.presentation_calls == 1


def test_replay_exit_response_is_flushed_before_server_shutdown() -> None:
    service = _FakeReplayService()
    server = create_server(
        service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=_coordinator(service),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            _FakeReplayRequest(
                client_id="replay-browser",
                command_id="exit",
                base_revision=0,
                command=_FakeExitCommand(),
            )
            .model_dump_json()
            .encode()
        )
        response, payload = _exchange(
            server,
            "POST",
            "/api/replay/command",
            body=body,
            headers=_authorized_headers(**{"Content-Type": "application/json"}),
        )
        thread.join(timeout=2)

        assert response.status == HTTPStatus.OK
        assert json.loads(payload)["result"] == "shutdown_scheduled"
        assert server.shutdown_started.is_set()
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
