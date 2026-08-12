"""Core-free replay-mode HTTP coordinator tests with an injected fake service."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from pathlib import Path
from threading import Thread
from typing import Annotated, Literal, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field
from scripts.dev.visual_debugger.server import (
    LIVE_HTTP_ROUTES,
    REPLAY_HTTP_ROUTES,
    DebuggerHTTPServer,
    HttpCoordinatorBinding,
    HttpRouteSet,
    create_server,
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


@dataclass(frozen=True, slots=True)
class _FakeServiceResult:
    outcome: str
    payload: _FakeReplayResponse | _FakeReplayError
    shutdown_requested: bool = False


class _FakeReplayService:
    """Small transport fake; no production replay or simulator imports."""

    def __init__(self) -> None:
        self.revision = 0
        self.frame_index = 0
        self.frame_calls = 0
        self.timeline_calls = 0
        self.command_calls = 0
        self.cursor_mutations = 0
        self.private_replay = {
            "hidden_events": ["must-never-cross-http"],
            "metric_report": {"secret": 1},
        }
        self._commands: dict[tuple[str, str], str] = {}

    def current_frame(self) -> _FakeReplayFrame:
        self.frame_calls += 1
        return self._frame()

    def current_timeline(self) -> _FakeReplayTimeline:
        self.timeline_calls += 1
        return _FakeReplayTimeline(
            current_frame_index=self.frame_index,
            frame_indices=(0, 1, 2),
        )

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


def _error_factory(*, error_code: str, message: str) -> _FakeReplayError:
    return _FakeReplayError(
        error_code=cast(_ErrorCode, error_code),
        message=message,
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


def _authorized_headers(**extra: str) -> dict[str, str]:
    return {_TOKEN_HEADER: _TOKEN, **extra}


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
    'scripts.dev.visual_debugger.replay_protocol',
    'scripts.dev.visual_debugger.replay_service',
    'scripts.dev.visual_debugger.scenarios',
    'scripts.dev.visual_debugger.service',
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
            current_timeline=service.current_timeline,
        )
    with pytest.raises(ValueError, match="cannot expose a replay timeline"):
        HttpCoordinatorBinding(
            mode="live",
            routes=LIVE_HTTP_ROUTES,
            request_model=_FakeReplayRequest,
            error_factory=_error_factory,
            current_frame=service.current_frame,
            apply_command=service.apply_command,
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
        )
    with pytest.raises(ValueError, match="exact HTTP route set"):
        HttpCoordinatorBinding(
            mode="live",
            routes=HttpRouteSet(
                frame="/api/frame",
                command="/api/replay/command",
                timeline=None,
            ),
            request_model=_FakeReplayRequest,
            error_factory=_error_factory,
            current_frame=service.current_frame,
            apply_command=service.apply_command,
        )


def test_replay_frame_and_timeline_are_authenticated_bounded_models(
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
    timeline, timeline_body = _exchange(
        server,
        "GET",
        "/api/replay/timeline",
        headers=_authorized_headers(),
    )

    assert missing.status == HTTPStatus.UNAUTHORIZED
    assert json.loads(missing_body)["error_code"] == "unauthorized"
    assert frame.status == timeline.status == HTTPStatus.OK
    assert frame.getheader("Cache-Control") == "no-store"
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
    serialized = frame_body + timeline_body
    assert b"must-never-cross-http" not in serialized
    assert b"hidden_events" not in serialized
    assert b"metric_report" not in serialized
    assert service.frame_calls == service.timeline_calls == 1


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
    finally:
        live_server.shutdown()
        live_server.server_close()
        live_thread.join(timeout=2)

    assert replay_live_command.status == HTTPStatus.NOT_FOUND
    assert invented_replay_frame.status == HTTPStatus.NOT_FOUND
    assert replay_timeline.status == replay_command.status == HTTPStatus.NOT_FOUND
    assert replay_service.command_calls == 0
    assert live_service.timeline_calls == live_service.command_calls == 0


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
