"""Actual-service HTTP proofs for the live authorized presentation resource."""

from __future__ import annotations

import json
from collections.abc import Iterator
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from pathlib import Path
from threading import Thread

import pytest
from scripts.dev.visual_debugger.presentation_protocol import (
    LiveNoSharedObsAuthorizedPresentationFrameV1,
    LiveOracleAuthorizedPresentationFrameV1,
    PresentationResourceResultV1,
)
from scripts.dev.visual_debugger.protocol import (
    ActorPovLiveDebuggerFrameV2,
    CommandRequestV1,
    CommandResponseV2,
    KeyboardCommandV1,
    ResearcherLiveDebuggerFrameV2,
    ResetCommandV1,
    SetViewCommandV1,
)
from scripts.dev.visual_debugger.server import DebuggerHTTPServer, create_server
from scripts.dev.visual_debugger.service import DebuggerService
from tests.test_visual_debugger_service import (
    _service,  # pyright: ignore[reportPrivateUsage]
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ASSET_ROOT = _REPOSITORY_ROOT / "web" / "visual_debugger"
_TOKEN_HEADER = "X-MARL-Debugger-Token"
_TOKEN = "test-live-presentation-capability"
_PRESENTATION_PATH = "/api/presentation/frame"


@pytest.fixture
def running_live_presentation_server() -> Iterator[
    tuple[DebuggerHTTPServer, DebuggerService, Thread]
]:
    service = _service()
    server = create_server(
        service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
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
    command: KeyboardCommandV1 | ResetCommandV1 | SetViewCommandV1,
) -> bytes:
    return (
        CommandRequestV1(
            client_id="live-presentation-browser",
            command_id=command_id,
            base_revision=base_revision,
            command=command,
        )
        .model_dump_json()
        .encode()
    )


def _post_command(
    server: DebuggerHTTPServer,
    command_id: str,
    *,
    base_revision: int,
    command: KeyboardCommandV1 | ResetCommandV1 | SetViewCommandV1,
) -> CommandResponseV2:
    response, body = _exchange(
        server,
        "POST",
        "/api/command",
        body=_command_body(
            command_id,
            base_revision=base_revision,
            command=command,
        ),
        headers=_authorized_headers(**{"Content-Type": "application/json"}),
    )
    assert response.status == HTTPStatus.OK
    return CommandResponseV2.model_validate_json(body)


def _assert_json_security_headers(response: HTTPResponse, body: bytes) -> None:
    assert response.getheader("Content-Type") == "application/json; charset=utf-8"
    assert response.getheader("Content-Length") == str(len(body))
    assert response.getheader("Connection") == "close"
    assert response.getheader("Cache-Control") == "no-store"
    assert response.getheader("X-Content-Type-Options") == "nosniff"
    assert response.getheader("X-Frame-Options") == "DENY"
    assert response.getheader("Referrer-Policy") == "no-referrer"
    assert response.getheader("Content-Security-Policy") == (
        "default-src 'self'; "
        "base-uri 'none'; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "frame-ancestors 'none'; "
        "img-src 'self' data:; "
        "object-src 'none'; "
        "script-src 'self'; "
        "style-src 'self'"
    )
    assert response.getheader("Access-Control-Allow-Origin") is None


def _duplicate_token_exchange(
    server: DebuggerHTTPServer,
) -> tuple[HTTPResponse, bytes]:
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.putrequest("GET", _PRESENTATION_PATH)
    connection.putheader(_TOKEN_HEADER, _TOKEN)
    connection.putheader(_TOKEN_HEADER, _TOKEN)
    connection.endheaders()
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response, payload


def test_live_oracle_frame_zero_get_is_exact_strict_and_source_joined(
    running_live_presentation_server: tuple[
        DebuggerHTTPServer,
        DebuggerService,
        Thread,
    ],
) -> None:
    server, service, _ = running_live_presentation_server

    raw_response, raw_body = _exchange(
        server,
        "GET",
        "/api/frame",
        headers=_authorized_headers(),
    )
    expected = service.current_presentation()
    presentation_response, presentation_body = _exchange(
        server,
        "GET",
        _PRESENTATION_PATH,
        headers=_authorized_headers(),
    )

    assert raw_response.status == presentation_response.status == HTTPStatus.OK
    assert expected.outcome == "response"
    assert type(expected.payload) is LiveOracleAuthorizedPresentationFrameV1
    assert presentation_body == expected.payload.model_dump_json().encode()
    raw = ResearcherLiveDebuggerFrameV2.model_validate_json(raw_body)
    presentation = LiveOracleAuthorizedPresentationFrameV1.model_validate_json(
        presentation_body
    )
    assert type(presentation) is LiveOracleAuthorizedPresentationFrameV1
    assert presentation.presentation_kind == "live_oracle"
    assert presentation.product_kind == "combat_debugger"
    assert presentation.source.source_revision == raw.revision == 0
    assert presentation.source.source_authority_epoch == raw.revision
    assert presentation.source.source_frame_index == raw.frame_index == 0
    assert presentation.source.source_frame_id == raw.frame_id
    assert presentation.source.source_session_id == raw.session_id
    assert presentation.source.source_run_generation == raw.run_generation
    assert presentation.current_endpoint.frame_id == raw.frame_id
    assert presentation.latest_events is None
    assert presentation.latest_transition is None
    _assert_json_security_headers(presentation_response, presentation_body)


def test_live_no_shared_get_after_pov_switch_is_exact_strict_and_source_joined(
    running_live_presentation_server: tuple[
        DebuggerHTTPServer,
        DebuggerService,
        Thread,
    ],
) -> None:
    server, service, _ = running_live_presentation_server
    command = _post_command(
        server,
        "switch-to-pov",
        base_revision=0,
        command=SetViewCommandV1(view_mode="pov"),
    )
    assert command.result == "applied"
    assert type(command.frame) is ActorPovLiveDebuggerFrameV2

    raw_response, raw_body = _exchange(
        server,
        "GET",
        "/api/frame",
        headers=_authorized_headers(),
    )
    expected = service.current_presentation()
    presentation_response, presentation_body = _exchange(
        server,
        "GET",
        _PRESENTATION_PATH,
        headers=_authorized_headers(),
    )

    assert raw_response.status == presentation_response.status == HTTPStatus.OK
    assert expected.outcome == "response"
    assert type(expected.payload) is LiveNoSharedObsAuthorizedPresentationFrameV1
    assert presentation_body == expected.payload.model_dump_json().encode()
    raw = ActorPovLiveDebuggerFrameV2.model_validate_json(raw_body)
    presentation = LiveNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
        presentation_body
    )
    assert type(presentation) is LiveNoSharedObsAuthorizedPresentationFrameV1
    assert presentation.presentation_kind == "live_no_shared_obs_agent_pov"
    assert presentation.product_kind == "combat_debugger"
    assert presentation.source.source_revision == raw.revision
    assert presentation.source.source_authority_epoch == raw.revision
    assert presentation.source.source_frame_index == raw.frame_index
    assert presentation.source.source_session_id == raw.session_id
    assert presentation.source.source_run_generation == raw.run_generation
    recipient = raw.projection.scene.self_actor.public_agent_id
    assert presentation.source.source_recipient_public_agent_id == recipient
    assert presentation.source.source_recipient_frame_id == (
        f"{raw.episode_id}:actor-pov:{recipient}:frame:{raw.frame_index}"
    )
    assert (
        presentation.current_endpoint.parts.source_recipient_frame_id
        == presentation.source.source_recipient_frame_id
    )
    assert "oracle_" not in presentation_body.decode()
    _assert_json_security_headers(presentation_response, presentation_body)


def test_scripted_reset_http_is_epoch_preserving_and_nonmutating() -> None:
    service = _service("basic_support")
    server = create_server(
        service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        before_session = service.session
        before_frame = service.current_frame()

        blocked = _post_command(
            server,
            "scripted-reset",
            base_revision=0,
            command=ResetCommandV1(),
        )

        assert blocked.result == "no_op"
        assert blocked.frame == before_frame
        assert blocked.notice is not None
        assert "inspection-only" in blocked.notice
        assert service.session is before_session
        assert service.revision == 0
        assert service.session.run_generation == 0
        assert service.session.current_evaluation_frame.frame_index == 0

    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_command_between_raw_and_presentation_gets_is_detectable_by_source_epoch(
    running_live_presentation_server: tuple[
        DebuggerHTTPServer,
        DebuggerService,
        Thread,
    ],
) -> None:
    server, _, _ = running_live_presentation_server
    raw_response, raw_body = _exchange(
        server,
        "GET",
        "/api/frame",
        headers=_authorized_headers(),
    )
    command = _post_command(
        server,
        "advance-between-gets",
        base_revision=0,
        command=KeyboardCommandV1(key=" "),
    )
    presentation_response, presentation_body = _exchange(
        server,
        "GET",
        _PRESENTATION_PATH,
        headers=_authorized_headers(),
    )

    assert raw_response.status == presentation_response.status == HTTPStatus.OK
    raw = ResearcherLiveDebuggerFrameV2.model_validate_json(raw_body)
    assert type(command.frame) is ResearcherLiveDebuggerFrameV2
    presentation = LiveOracleAuthorizedPresentationFrameV1.model_validate_json(
        presentation_body
    )
    assert (raw.revision, raw.frame_index) == (0, 0)
    assert raw.frame_id == f"{raw.episode_id}:frame:0"
    assert command.frame.revision == presentation.source.source_revision
    assert command.frame.frame_index == presentation.source.source_frame_index
    assert command.frame.frame_id == presentation.source.source_frame_id
    assert (
        raw.revision,
        raw.frame_index,
        raw.frame_id,
    ) != (
        presentation.source.source_revision,
        presentation.source.source_frame_index,
        presentation.source.source_frame_id,
    )
    assert presentation.latest_transition is not None
    assert presentation.latest_transition.incoming_transition_index == 0


def test_presentation_request_boundaries_never_invoke_live_getter(
    running_live_presentation_server: tuple[
        DebuggerHTTPServer,
        DebuggerService,
        Thread,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, service, _ = running_live_presentation_server
    calls = 0
    original = service.current_presentation

    def counted_presentation() -> PresentationResourceResultV1:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(service, "current_presentation", counted_presentation)

    missing, _ = _exchange(server, "GET", _PRESENTATION_PATH)
    wrong, _ = _exchange(
        server,
        "GET",
        _PRESENTATION_PATH,
        headers={_TOKEN_HEADER: "wrong-capability"},
    )
    duplicate, _ = _duplicate_token_exchange(server)
    bad_host, _ = _exchange(
        server,
        "GET",
        _PRESENTATION_PATH,
        headers=_authorized_headers(Host="example.invalid"),
    )
    bad_origin, _ = _exchange(
        server,
        "GET",
        _PRESENTATION_PATH,
        headers=_authorized_headers(Origin="null"),
    )
    cross_site, _ = _exchange(
        server,
        "GET",
        _PRESENTATION_PATH,
        headers=_authorized_headers(**{"Sec-Fetch-Site": "cross-site"}),
    )
    query, _ = _exchange(
        server,
        "GET",
        f"{_PRESENTATION_PATH}?forbidden=1",
        headers=_authorized_headers(),
    )
    post, _ = _exchange(
        server,
        "POST",
        _PRESENTATION_PATH,
        body=b"{}",
        headers=_authorized_headers(**{"Content-Type": "application/json"}),
    )
    head, _ = _exchange(
        server,
        "HEAD",
        _PRESENTATION_PATH,
        headers=_authorized_headers(),
    )

    assert missing.status == wrong.status == duplicate.status == HTTPStatus.UNAUTHORIZED
    assert (
        bad_host.status
        == bad_origin.status
        == cross_site.status
        == HTTPStatus.FORBIDDEN
    )
    assert query.status == post.status == HTTPStatus.NOT_FOUND
    assert head.status == HTTPStatus.METHOD_NOT_ALLOWED
    assert calls == 0
    assert service.revision == 0
    assert service.session.current_evaluation_frame.frame_index == 0


def test_live_presentation_getter_failure_returns_generic_non_disclosing_500(
    running_live_presentation_server: tuple[
        DebuggerHTTPServer,
        DebuggerService,
        Thread,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, service, _ = running_live_presentation_server
    calls = 0

    def fail_presentation() -> PresentationResourceResultV1:
        nonlocal calls
        calls += 1
        raise RuntimeError("private live presentation failure detail")

    monkeypatch.setattr(service, "current_presentation", fail_presentation)
    response, body = _exchange(
        server,
        "GET",
        _PRESENTATION_PATH,
        headers=_authorized_headers(),
    )

    assert response.status == HTTPStatus.INTERNAL_SERVER_ERROR
    assert json.loads(body) == {
        "schema_version": 2,
        "error_code": "internal_error",
        "message": "The debugger could not process this request.",
        "latest_frame": None,
    }
    assert b"private live presentation failure detail" not in body
    assert b"RuntimeError" not in body
    assert calls == 1
    assert service.revision == 0
    assert service.session.current_evaluation_frame.frame_index == 0
    _assert_json_security_headers(response, body)
