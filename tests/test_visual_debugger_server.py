"""Focused HTTP security, routing, and lifecycle tests for the debugger."""

import json
import socket
from collections.abc import Iterator
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from pathlib import Path
from threading import Thread

import pytest
import scripts.dev.visual_debugger.server as server_module
from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.protocol import (
    CommandRequestV1,
    ExitCommandV1,
    KeyboardCommandV1,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.server import (
    DebuggerHTTPServer,
    DebuggerRequestHandler,
    GracefulCloseResult,
    build_static_manifest,
    create_server,
    serve_browser_debugger,
)
from scripts.dev.visual_debugger.service import DebuggerService
from tests.visual_debugger_fixtures import debugger_test_launch_specification

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ASSET_ROOT = _REPOSITORY_ROOT / "web" / "visual_debugger"
_TOKEN = "test-capability"


def _service() -> DebuggerService:
    session = create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    return DebuggerService(
        session,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        session_id="server-test-session",
    )


@pytest.fixture
def running_server() -> Iterator[tuple[DebuggerHTTPServer, Thread]]:
    server = create_server(
        _service(),
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, thread
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


_TOKEN_HEADER = "X-MARL-Debugger-Token"


def _command_body(
    command_id: str,
    *,
    base_revision: int,
    key: str,
) -> bytes:
    request = CommandRequestV1(
        client_id="browser-client",
        command_id=command_id,
        base_revision=base_revision,
        command=KeyboardCommandV1(key=key),
    )
    return request.model_dump_json().encode()


def _complete_runtime_asset_root(tmp_path: Path) -> Path:
    """Copy the validated runtime allowlist into one isolated test root."""
    asset_root = tmp_path / "web"
    for route, asset in build_static_manifest(_ASSET_ROOT).items():
        if route == "/":
            continue
        destination = asset_root / route.removeprefix("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(asset.body)
    return asset_root


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


def _raw_command_prefix(server: DebuggerHTTPServer) -> bytes:
    return (
        b"POST /api/command HTTP/1.1\r\n"
        + f"Host: {server.expected_host}\r\n".encode()
        + f"{_TOKEN_HEADER}: {_TOKEN}\r\n".encode()
        + b"Content-Type: application/json\r\n"
    )


def test_index_and_allowlisted_assets_use_security_headers(
    running_server: tuple[DebuggerHTTPServer, Thread],
) -> None:
    server, _ = running_server
    for path, expected_type in (
        ("/", "text/html"),
        ("/styles.css", "text/css"),
        ("/src/main.js", "text/javascript"),
        ("/src/replay-controls.js", "text/javascript"),
        ("/src/replay-frame-normalizer.js", "text/javascript"),
        ("/src/explanations.js", "text/javascript"),
        ("/src/tooltip.js", "text/javascript"),
        (
            "/assets/fonts/AtkinsonHyperlegible-Regular.woff2",
            "font/woff2",
        ),
        (
            "/assets/fonts/AtkinsonHyperlegible-Bold.woff2",
            "font/woff2",
        ),
        ("/assets/fonts/OFL.txt", "text/plain"),
    ):
        response, body = _exchange(server, "GET", path)
        assert response.status == 200
        assert response.getheader("Content-Type", "").startswith(expected_type)
        assert response.getheader("Cache-Control") == "no-store"
        assert response.getheader("X-Content-Type-Options") == "nosniff"
        assert "frame-ancestors 'none'" in response.getheader(
            "Content-Security-Policy", ""
        )
        assert body


def test_frame_api_requires_token_and_never_mutates_service(
    running_server: tuple[DebuggerHTTPServer, Thread],
) -> None:
    server, _ = running_server
    initial = server.debugger_service.session

    missing, missing_body = _exchange(server, "GET", "/api/frame")
    wrong, wrong_body = _exchange(
        server,
        "GET",
        "/api/frame",
        headers={_TOKEN_HEADER: "wrong"},
    )
    accepted, accepted_body = _exchange(
        server,
        "GET",
        "/api/frame",
        headers=_authorized_headers(),
    )

    assert missing.status == wrong.status == 401
    assert json.loads(missing_body) == json.loads(wrong_body)
    assert accepted.status == 200
    assert json.loads(accepted_body)["revision"] == 0
    assert server.debugger_service.session is initial


def test_host_origin_and_cross_site_requests_are_rejected(
    running_server: tuple[DebuggerHTTPServer, Thread],
) -> None:
    server, _ = running_server
    bad_host, _ = _exchange(
        server,
        "GET",
        "/api/frame",
        headers=_authorized_headers(Host="example.test"),
    )
    bad_origin, _ = _exchange(
        server,
        "GET",
        "/api/frame",
        headers=_authorized_headers(Origin="null"),
    )
    cross_site, _ = _exchange(
        server,
        "GET",
        "/api/frame",
        headers=_authorized_headers(**{"Sec-Fetch-Site": "cross-site"}),
    )

    assert bad_host.status == bad_origin.status == cross_site.status == 403


@pytest.mark.parametrize(
    "path",
    (
        "/../pyproject.toml",
        "/%2e%2e/pyproject.toml",
        "/..%2fpyproject.toml",
        r"/..\pyproject.toml",
        "/package.json",
        "/tests/control-parity.spec.js",
    ),
)
def test_static_routes_cannot_escape_the_runtime_allowlist(
    running_server: tuple[DebuggerHTTPServer, Thread],
    path: str,
) -> None:
    server, _ = running_server
    response, _ = _exchange(server, "GET", path)
    assert response.status == 404


def test_malformed_invalid_and_wrong_media_commands_never_reach_service(
    running_server: tuple[DebuggerHTTPServer, Thread],
) -> None:
    server, _ = running_server
    malformed, _ = _exchange(
        server,
        "POST",
        "/api/command",
        body=b"{",
        headers=_authorized_headers(**{"Content-Type": "application/json"}),
    )
    invalid, _ = _exchange(
        server,
        "POST",
        "/api/command",
        body=b'{"schema_version":1}',
        headers=_authorized_headers(**{"Content-Type": "application/json"}),
    )
    wrong_media, _ = _exchange(
        server,
        "POST",
        "/api/command",
        body=b"{}",
        headers=_authorized_headers(**{"Content-Type": "text/plain"}),
    )

    assert malformed.status == 400
    assert invalid.status == 422
    assert wrong_media.status == 415
    assert server.debugger_service.revision == 0
    assert int(server.debugger_service.session.state.step_count) == 0


@pytest.mark.parametrize(
    ("framing_headers", "body", "expected_status"),
    (
        (
            b"Content-Length: 2\r\nContent-Length: 2\r\n",
            b"{}",
            b" 411 ",
        ),
        (
            b"Transfer-Encoding: chunked\r\nContent-Length: 2\r\n",
            b"0\r\n\r\n",
            b" 400 ",
        ),
    ),
)
def test_ambiguous_command_framing_is_rejected_and_connection_closed(
    running_server: tuple[DebuggerHTTPServer, Thread],
    framing_headers: bytes,
    body: bytes,
    expected_status: bytes,
) -> None:
    server, _ = running_server
    response = _raw_exchange(
        server,
        _raw_command_prefix(server) + framing_headers + b"\r\n" + body,
    )

    assert expected_status in response.split(b"\r\n", maxsplit=1)[0]
    assert b"Connection: close\r\n" in response
    assert response.count(b"HTTP/1.1 ") == 1
    assert int(server.debugger_service.session.state.step_count) == 0


@pytest.mark.parametrize(
    ("duplicate_headers", "expected_status"),
    (
        (b"Host: duplicate.invalid\r\n", b" 403 "),
        (
            b"Origin: http://duplicate.invalid\r\nOrigin: http://duplicate.invalid\r\n",
            b" 403 ",
        ),
        (
            f"{_TOKEN_HEADER}: {_TOKEN}\r\n".encode(),
            b" 401 ",
        ),
        (b"Content-Type: application/json\r\n", b" 415 "),
    ),
)
def test_security_sensitive_duplicate_headers_are_rejected(
    running_server: tuple[DebuggerHTTPServer, Thread],
    duplicate_headers: bytes,
    expected_status: bytes,
) -> None:
    server, _ = running_server
    request = (
        _raw_command_prefix(server) + duplicate_headers + b"Content-Length: 2\r\n\r\n{}"
    )

    response = _raw_exchange(server, request)

    assert expected_status in response.split(b"\r\n", maxsplit=1)[0]
    assert b"Connection: close\r\n" in response
    assert int(server.debugger_service.session.state.step_count) == 0


def test_oversized_body_cannot_desynchronize_a_second_request(
    running_server: tuple[DebuggerHTTPServer, Thread],
) -> None:
    server, _ = running_server
    pipelined = (
        _raw_command_prefix(server)
        + b"Content-Length: 65537\r\n\r\n"
        + b"{}"
        + b"GET /api/frame HTTP/1.1\r\n"
        + f"Host: {server.expected_host}\r\n".encode()
        + f"{_TOKEN_HEADER}: {_TOKEN}\r\n\r\n".encode()
    )

    response = _raw_exchange(server, pipelined)

    assert b" 413 " in response.split(b"\r\n", maxsplit=1)[0]
    assert b"Connection: close\r\n" in response
    assert response.count(b"HTTP/1.1 ") == 1
    assert server.debugger_service.revision == 0


def test_incomplete_command_body_times_out_without_reaching_service(
    running_server: tuple[DebuggerHTTPServer, Thread],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, _ = running_server
    monkeypatch.setattr(server_module, "_CLIENT_SOCKET_TIMEOUT_SECONDS", 0.1)
    request = _raw_command_prefix(server) + b"Content-Length: 10\r\n\r\n" + b"{"

    response = _raw_exchange(server, request)

    assert b" 408 " in response.split(b"\r\n", maxsplit=1)[0]
    assert b"Connection: close\r\n" in response
    assert int(server.debugger_service.session.state.step_count) == 0


def test_identical_http_submit_body_is_applied_then_returned_as_duplicate(
    running_server: tuple[DebuggerHTTPServer, Thread],
) -> None:
    server, _ = running_server
    body = _command_body("identical-submit", base_revision=0, key="Enter")
    headers = _authorized_headers(**{"Content-Type": "application/json"})

    applied, applied_body = _exchange(
        server,
        "POST",
        "/api/command",
        body=body,
        headers=headers,
    )
    duplicate, duplicate_body = _exchange(
        server,
        "POST",
        "/api/command",
        body=body,
        headers=headers,
    )

    applied_payload = json.loads(applied_body)
    duplicate_payload = json.loads(duplicate_body)
    assert applied.status == duplicate.status == 200
    assert applied_payload["result"] == "applied"
    assert duplicate_payload["result"] == "duplicate"
    assert applied_payload["frame"]["revision"] == 1
    assert duplicate_payload["frame"]["revision"] == 1
    assert applied_payload["frame"]["simulator_step_count"] == 1
    assert duplicate_payload["frame"]["simulator_step_count"] == 1
    assert applied_payload["frame"]["incoming_transition_index"] == 0
    assert duplicate_payload["frame"]["incoming_transition_index"] == 0
    assert server.debugger_service.revision == 1
    assert int(server.debugger_service.session.state.step_count) == 1


def test_command_and_stale_service_results_map_to_http(
    running_server: tuple[DebuggerHTTPServer, Thread],
) -> None:
    server, _ = running_server
    headers = _authorized_headers(**{"Content-Type": "application/json"})
    applied, applied_body = _exchange(
        server,
        "POST",
        "/api/command",
        body=_command_body("move", base_revision=0, key="d"),
        headers=headers,
    )
    stale, stale_body = _exchange(
        server,
        "POST",
        "/api/command",
        body=_command_body("stale", base_revision=0, key="enter"),
        headers=headers,
    )

    assert applied.status == 200
    assert json.loads(applied_body)["frame"]["revision"] == 1
    assert stale.status == 409
    assert json.loads(stale_body)["error_code"] == "stale_revision"
    assert json.loads(stale_body)["latest_frame"]["revision"] == 1
    assert int(server.debugger_service.session.state.step_count) == 0


def test_service_shutdown_fence_maps_to_503_without_stepping(
    running_server: tuple[DebuggerHTTPServer, Thread],
) -> None:
    server, _ = running_server
    accepted_exit = server.debugger_service.apply_command(
        CommandRequestV1(
            client_id="exiting-client",
            command_id="exit-directly",
            base_revision=0,
            command=ExitCommandV1(),
        )
    )
    assert accepted_exit.shutdown_requested

    response, payload = _exchange(
        server,
        "POST",
        "/api/command",
        body=_command_body("submit-after-exit", base_revision=0, key="Enter"),
        headers=_authorized_headers(**{"Content-Type": "application/json"}),
    )

    assert response.status == 503
    assert json.loads(payload)["error_code"] == "server_shutting_down"
    assert int(server.debugger_service.session.state.step_count) == 0


def test_unknown_routes_and_methods_are_not_cors_enabled(
    running_server: tuple[DebuggerHTTPServer, Thread],
) -> None:
    server, _ = running_server
    missing, _ = _exchange(server, "GET", "/missing")
    options, _ = _exchange(server, "OPTIONS", "/api/command")
    replay_timeline, _ = _exchange(
        server,
        "GET",
        "/api/replay/timeline",
        headers=_authorized_headers(),
    )
    replay_command, _ = _exchange(
        server,
        "POST",
        "/api/replay/command",
        body=b"{}",
        headers=_authorized_headers(**{"Content-Type": "application/json"}),
    )

    assert missing.status == 404
    assert options.status == 405
    assert options.getheader("Access-Control-Allow-Origin") is None
    assert replay_timeline.status == replay_command.status == 404
    assert server.coordinator.mode == "live"
    assert server.debugger_service.revision == 0


def test_authenticated_exit_delivers_response_then_stops_server() -> None:
    server = create_server(
        _service(),
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            CommandRequestV1(
                client_id="browser-client",
                command_id="exit",
                base_revision=0,
                command={"command_type": "exit"},  # pyright: ignore[reportArgumentType]
            )
            .model_dump_json()
            .encode()
        )
        response, payload = _exchange(
            server,
            "POST",
            "/api/command",
            body=body,
            headers=_authorized_headers(**{"Content-Type": "application/json"}),
        )
        thread.join(timeout=2)

        assert response.status == 200
        assert json.loads(payload)["result"] == "shutdown_scheduled"
        assert not thread.is_alive()
        assert server.shutdown_started.is_set()
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()


def test_accepted_exit_stops_server_when_response_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_send_model = DebuggerRequestHandler._send_model  # pyright: ignore[reportPrivateUsage]

    def fail_exit_response(
        handler: DebuggerRequestHandler,
        status: HTTPStatus,
        model: object,
    ) -> None:
        if getattr(model, "result", None) == "shutdown_scheduled":
            raise BrokenPipeError
        real_send_model(handler, status, model)

    monkeypatch.setattr(DebuggerRequestHandler, "_send_model", fail_exit_response)
    server = create_server(
        _service(),
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            CommandRequestV1(
                client_id="browser-client",
                command_id="exit-write-failure",
                base_revision=0,
                command=ExitCommandV1(),
            )
            .model_dump_json()
            .encode()
        )
        response = _raw_exchange(
            server,
            _raw_command_prefix(server)
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body,
        )
        thread.join(timeout=2)

        assert response == b""
        assert not thread.is_alive()
        assert server.shutdown_started.is_set()
        assert server.debugger_service.shutting_down
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()


def test_unexpected_service_failure_returns_generic_internal_error(
    running_server: tuple[DebuggerHTTPServer, Thread],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, _ = running_server

    def fail_command(request: object) -> object:
        del request
        raise RuntimeError("sensitive synthetic detail")

    monkeypatch.setattr(server.debugger_service, "apply_command", fail_command)
    response, payload = _exchange(
        server,
        "POST",
        "/api/command",
        body=_command_body("failure", base_revision=0, key="Enter"),
        headers=_authorized_headers(**{"Content-Type": "application/json"}),
    )

    decoded = json.loads(payload)
    assert response.status == 500
    assert decoded["error_code"] == "internal_error"
    assert "sensitive synthetic detail" not in payload.decode()
    assert int(server.debugger_service.session.state.step_count) == 0


@pytest.mark.parametrize(
    "relative_path",
    (
        Path("src") / "main.js",
        Path("src") / "display.js",
        Path("src") / "explanations.js",
        Path("src") / "tooltip.js",
        Path("assets") / "fonts" / "AtkinsonHyperlegible-Regular.woff2",
    ),
    ids=(
        "main-module",
        "display-module",
        "explanations-module",
        "tooltip-module",
        "font",
    ),
)
def test_missing_required_runtime_asset_fails_before_browser_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: Path,
) -> None:
    asset_root = _complete_runtime_asset_root(tmp_path)
    (asset_root / relative_path).unlink()
    opened_urls: list[str] = []

    def record_browser_open(url: str) -> bool:
        opened_urls.append(url)
        return True

    monkeypatch.setattr(server_module.webbrowser, "open", record_browser_open)

    with pytest.raises(ValueError, match="required browser asset is missing"):
        serve_browser_debugger(
            _service(),
            asset_root=asset_root,
            port=0,
            open_browser=True,
        )

    assert opened_urls == []


def test_occupied_port_fails_before_browser_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened_urls: list[str] = []

    def record_browser_open(url: str) -> bool:
        opened_urls.append(url)
        return True

    monkeypatch.setattr(server_module.webbrowser, "open", record_browser_open)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = int(occupied.getsockname()[1])

        with pytest.raises(OSError):
            serve_browser_debugger(
                _service(),
                asset_root=_ASSET_ROOT,
                port=port,
                open_browser=True,
            )

    assert opened_urls == []


def test_static_manifest_rejects_symlinked_runtime_assets(tmp_path: Path) -> None:
    outside = tmp_path / "outside.html"
    outside.write_text("<html></html>", encoding="utf-8")
    asset_root = _complete_runtime_asset_root(tmp_path)
    (asset_root / "index.html").unlink()
    (asset_root / "index.html").symlink_to(outside)

    with pytest.raises(ValueError, match="symlinks"):
        build_static_manifest(asset_root)


def test_static_manifest_snapshots_asset_bytes_at_startup(tmp_path: Path) -> None:
    asset_root = _complete_runtime_asset_root(tmp_path)
    index = asset_root / "index.html"
    index.write_text("<html>initial</html>", encoding="utf-8")

    manifest = build_static_manifest(asset_root)
    index.write_text("<html>replaced</html>", encoding="utf-8")

    assert manifest["/"].body == b"<html>initial</html>"


def test_keyboard_interrupt_without_close_hook_preserves_legacy_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupt_server(
        server: DebuggerHTTPServer,
        *,
        poll_interval: float,
    ) -> None:
        del server, poll_interval
        raise KeyboardInterrupt

    monkeypatch.setattr(DebuggerHTTPServer, "serve_forever", interrupt_server)

    result = serve_browser_debugger(
        _service(),
        asset_root=_ASSET_ROOT,
        port=0,
        open_browser=False,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Visual Debugger and Analyzer stopped." in captured.out


def test_keyboard_interrupt_invokes_graceful_close_once_under_router_lock(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupt_server(
        server: DebuggerHTTPServer,
        *,
        poll_interval: float,
    ) -> None:
        del server, poll_interval
        raise KeyboardInterrupt

    observed_servers: list[DebuggerHTTPServer] = []
    real_run_graceful_close = DebuggerHTTPServer.run_graceful_close

    def capture_server_and_close(
        server: DebuggerHTTPServer,
        callback: server_module.GracefulCloseCallback,
    ) -> GracefulCloseResult:
        observed_servers.append(server)
        return real_run_graceful_close(server, callback)

    monkeypatch.setattr(DebuggerHTTPServer, "serve_forever", interrupt_server)
    monkeypatch.setattr(
        DebuggerHTTPServer,
        "run_graceful_close",
        capture_server_and_close,
    )
    calls: list[int] = []

    def graceful_close() -> GracefulCloseResult:
        calls.append(1)
        server = observed_servers[0]
        with server.coordinator_router.pinned_snapshot() as nested:
            assert nested.binding.mode == "live"
            assert nested is server.coordinator_snapshot()
        return GracefulCloseResult(
            exit_code=7,
            message="synthetic recovery publication failed",
        )

    result = serve_browser_debugger(
        _service(),
        asset_root=_ASSET_ROOT,
        port=0,
        open_browser=False,
        graceful_close=graceful_close,
    )

    captured = capsys.readouterr()
    assert result == 7
    assert calls == [1]
    assert len(observed_servers) == 1
    assert "synthetic recovery publication failed" in captured.err
