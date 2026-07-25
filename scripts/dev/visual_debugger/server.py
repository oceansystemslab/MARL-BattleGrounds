"""Authenticated loopback HTTP transport for the live visual debugger."""

from __future__ import annotations

import json
import secrets
import sys
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socket import socket
from threading import Event, Lock, Thread
from typing import cast
from urllib.parse import urlsplit

from pydantic import ValidationError

from scripts.dev.visual_debugger.protocol import (
    ApiErrorCode,
    ApiErrorV1,
    CommandRequestV1,
)
from scripts.dev.visual_debugger.service import DebuggerService

_TOKEN_HEADER = "X-MARL-Debugger-Token"
_MAX_COMMAND_BODY_BYTES = 64 * 1024
_CLIENT_SOCKET_TIMEOUT_SECONDS = 2.0
_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "base-uri 'none'; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "frame-ancestors 'none'; "
        "img-src 'self' data:; "
        "object-src 'none'; "
        "script-src 'self'; "
        "style-src 'self'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}
_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".woff2": "font/woff2",
}


@dataclass(frozen=True, slots=True)
class StaticAsset:
    """One exact URL-to-file entry in the runtime asset allowlist."""

    body: bytes
    content_type: str


def build_static_manifest(asset_root: Path) -> dict[str, StaticAsset]:
    """Validate and return the exact browser-runtime static asset allowlist."""
    root = asset_root.resolve(strict=True)
    required = (root / "index.html", root / "styles.css")
    candidates = [*required]
    source_root = root / "src"
    if source_root.is_dir():
        candidates.extend(sorted(source_root.rglob("*.js")))
    bundled_root = root / "assets"
    if bundled_root.is_dir():
        candidates.extend(
            sorted(
                path
                for path in bundled_root.rglob("*")
                if path.is_file() and path.suffix.lower() in _CONTENT_TYPES
            )
        )

    manifest: dict[str, StaticAsset] = {}
    for candidate in candidates:
        if not candidate.is_file():
            raise ValueError(f"required browser asset is missing: {candidate}")
        relative = candidate.relative_to(root)
        if candidate.is_symlink() or any(
            parent.is_symlink() for parent in candidate.parents if parent != root
        ):
            raise ValueError(f"browser assets must not use symlinks: {candidate}")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"browser asset escapes its root: {candidate}")
        content_type = _CONTENT_TYPES.get(candidate.suffix.lower())
        if content_type is None:
            raise ValueError(f"unsupported browser asset type: {candidate}")
        try:
            body = resolved.read_bytes()
        except OSError as exc:
            raise ValueError(f"browser asset could not be loaded: {candidate}") from exc
        route = f"/{relative.as_posix()}"
        manifest[route] = StaticAsset(body, content_type)

    manifest["/"] = manifest["/index.html"]
    return manifest


class DebuggerHTTPServer(ThreadingHTTPServer):
    """Threaded loopback server carrying one authoritative debugger service."""

    daemon_threads = True
    block_on_close = True

    def __init__(
        self,
        *,
        service: DebuggerService,
        capability_token: str,
        static_manifest: dict[str, StaticAsset],
        port: int,
    ) -> None:
        self.debugger_service = service
        self.capability_token = capability_token
        self.static_manifest = static_manifest
        self.shutdown_started = Event()
        self._shutdown_lock = Lock()
        super().__init__(("127.0.0.1", port), DebuggerRequestHandler)
        host, actual_port = cast(tuple[str, int], self.server_address)
        self.expected_host = f"{host}:{actual_port}"
        self.expected_origin = f"http://{self.expected_host}"

    def request_shutdown(self) -> None:
        """Schedule one non-blocking shutdown after an Exit response."""
        with self._shutdown_lock:
            if self.shutdown_started.is_set():
                return
            self.shutdown_started.set()
        Thread(
            target=self.shutdown,
            name="visual-debugger-shutdown",
            daemon=True,
        ).start()

    def get_request(self) -> tuple[socket, tuple[str, int]]:
        """Bound incomplete local requests without timing simulator execution."""
        request, client_address = super().get_request()
        request.settimeout(_CLIENT_SOCKET_TIMEOUT_SECONDS)
        return request, cast(tuple[str, int], client_address)


class DebuggerRequestHandler(BaseHTTPRequestHandler):
    """Exact route handler with no filesystem fallback or CORS surface."""

    protocol_version = "HTTP/1.1"
    server_version = "MARLDebugger/1"
    sys_version = ""

    def version_string(self) -> str:
        """Avoid disclosing the host Python version in HTTP responses."""
        return self.server_version

    @property
    def debugger_server(self) -> DebuggerHTTPServer:
        return cast(DebuggerHTTPServer, self.server)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request targets so capability material can never be logged."""
        del format, args

    def do_GET(self) -> None:
        route = self._route()
        if route is None or not self._valid_request_origin():
            return
        if route == "/api/frame":
            if not self._authenticated():
                return
            try:
                frame = self.debugger_server.debugger_service.current_frame()
            except Exception:
                self._send_internal_error()
                return
            try:
                self._send_model(HTTPStatus.OK, frame)
            except BrokenPipeError, ConnectionError, TimeoutError:
                self.close_connection = True
            except Exception:
                self._send_internal_error()
            return
        asset = self.debugger_server.static_manifest.get(route)
        if asset is None:
            self._send_api_error(
                HTTPStatus.NOT_FOUND,
                error_code="not_found",
                message="No such debugger route.",
            )
            return
        self._send_bytes(HTTPStatus.OK, asset.body, content_type=asset.content_type)

    def do_POST(self) -> None:
        route = self._route()
        if route is None or not self._valid_request_origin():
            return
        if route != "/api/command":
            self._send_api_error(
                HTTPStatus.NOT_FOUND,
                error_code="not_found",
                message="No such debugger route.",
            )
            return
        if not self._authenticated():
            return
        body = self._read_command_body()
        if body is None:
            return
        try:
            raw_request = json.loads(body)
        except json.JSONDecodeError, UnicodeDecodeError:
            self._send_api_error(
                HTTPStatus.BAD_REQUEST,
                error_code="invalid_request",
                message="Request body is not valid JSON.",
            )
            return
        try:
            request = CommandRequestV1.model_validate(raw_request)
        except ValidationError:
            self._send_api_error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                error_code="invalid_request",
                message="Request body does not match CommandRequestV1.",
            )
            return

        try:
            result = self.debugger_server.debugger_service.apply_command(request)
        except Exception:
            self._send_internal_error()
            return
        status = {
            "stale_revision": HTTPStatus.CONFLICT,
            "command_id_conflict": HTTPStatus.CONFLICT,
            "server_shutting_down": HTTPStatus.SERVICE_UNAVAILABLE,
            "service_faulted": HTTPStatus.INTERNAL_SERVER_ERROR,
        }.get(result.outcome, HTTPStatus.OK)
        try:
            self._send_model(status, result.payload)
            if result.shutdown_requested:
                self.wfile.flush()
        except BrokenPipeError, ConnectionError, TimeoutError:
            self.close_connection = True
        except Exception:
            self._send_internal_error()
        finally:
            if result.shutdown_requested:
                self.debugger_server.request_shutdown()

    def do_OPTIONS(self) -> None:
        self._method_not_allowed()

    def do_HEAD(self) -> None:
        self._method_not_allowed()

    def do_PUT(self) -> None:
        self._method_not_allowed()

    def do_PATCH(self) -> None:
        self._method_not_allowed()

    def do_DELETE(self) -> None:
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self._send_api_error(
            HTTPStatus.METHOD_NOT_ALLOWED,
            error_code="method_not_allowed",
            message="Method is not supported by this debugger.",
        )

    def _route(self) -> str | None:
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment:
            self._send_api_error(
                HTTPStatus.NOT_FOUND,
                error_code="not_found",
                message="No such debugger route.",
            )
            return None
        return parsed.path

    def _valid_request_origin(self) -> bool:
        hosts = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        fetch_sites = self.headers.get_all("Sec-Fetch-Site", [])
        if len(hosts) != 1 or hosts[0] != self.debugger_server.expected_host:
            self._send_api_error(
                HTTPStatus.FORBIDDEN,
                error_code="forbidden_origin",
                message="Request origin is not authorized.",
            )
            return False
        if len(origins) > 1 or (
            origins and origins[0] != self.debugger_server.expected_origin
        ):
            self._send_api_error(
                HTTPStatus.FORBIDDEN,
                error_code="forbidden_origin",
                message="Request origin is not authorized.",
            )
            return False
        if len(fetch_sites) > 1 or (fetch_sites and fetch_sites[0] == "cross-site"):
            self._send_api_error(
                HTTPStatus.FORBIDDEN,
                error_code="forbidden_origin",
                message="Request origin is not authorized.",
            )
            return False
        return True

    def _authenticated(self) -> bool:
        supplied = self.headers.get_all(_TOKEN_HEADER, [])
        if len(supplied) != 1 or not secrets.compare_digest(
            supplied[0],
            self.debugger_server.capability_token,
        ):
            self._send_api_error(
                HTTPStatus.UNAUTHORIZED,
                error_code="unauthorized",
                message="Debugger capability is missing or invalid.",
            )
            return False
        return True

    def _read_command_body(self) -> bytes | None:
        if self.headers.get_all("Transfer-Encoding", []):
            self._send_api_error(
                HTTPStatus.BAD_REQUEST,
                error_code="invalid_request",
                message="Chunked command requests are not supported.",
            )
            return None
        content_types = self.headers.get_all("Content-Type", [])
        if (
            len(content_types) != 1
            or self.headers.get_content_type() != "application/json"
        ):
            self._send_api_error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                error_code="unsupported_media_type",
                message="Command requests require application/json.",
            )
            return None
        raw_lengths = self.headers.get_all("Content-Length", [])
        if len(raw_lengths) != 1:
            self._send_api_error(
                HTTPStatus.LENGTH_REQUIRED,
                error_code="invalid_request",
                message="Command requests require one Content-Length value.",
            )
            return None
        raw_length = raw_lengths[0]
        if not raw_length.isascii() or not raw_length.isdecimal():
            self._send_api_error(
                HTTPStatus.BAD_REQUEST,
                error_code="invalid_request",
                message="Content-Length must be a non-negative integer.",
            )
            return None
        length = int(raw_length)
        if length > _MAX_COMMAND_BODY_BYTES:
            self._send_api_error(
                HTTPStatus.CONTENT_TOO_LARGE,
                error_code="payload_too_large",
                message="Command request exceeds the debugger body limit.",
            )
            return None
        try:
            body = self.rfile.read(length)
        except TimeoutError, OSError:
            self._send_api_error(
                HTTPStatus.REQUEST_TIMEOUT,
                error_code="invalid_request",
                message="Command request body was not received in time.",
            )
            return None
        if len(body) != length:
            self._send_api_error(
                HTTPStatus.BAD_REQUEST,
                error_code="invalid_request",
                message="Command request body ended before Content-Length.",
            )
            return None
        return body

    def _send_internal_error(self) -> None:
        self._send_api_error(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            error_code="internal_error",
            message="The debugger could not process this request.",
        )

    def _send_api_error(
        self,
        status: HTTPStatus,
        *,
        error_code: ApiErrorCode,
        message: str,
    ) -> None:
        self._send_model(
            status,
            ApiErrorV1(
                error_code=error_code,
                message=message,
                latest_frame=None,
            ),
        )

    def _send_model(self, status: HTTPStatus, model: object) -> None:
        model_dump_json = getattr(model, "model_dump_json", None)
        if not callable(model_dump_json):
            raise TypeError("HTTP JSON payloads must be Pydantic protocol models.")
        body = cast(str, model_dump_json()).encode("utf-8")
        self._send_bytes(
            status,
            body,
            content_type="application/json; charset=utf-8",
        )

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        *,
        content_type: str,
    ) -> None:
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for name, value in _SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


def create_server(
    service: DebuggerService,
    *,
    asset_root: Path,
    port: int,
    capability_token: str | None = None,
) -> DebuggerHTTPServer:
    """Validate assets, then bind one debugger server to loopback."""
    if not 0 <= port <= 65535:
        raise ValueError(f"port must be in [0, 65535]; got {port}.")
    manifest = build_static_manifest(asset_root)
    return DebuggerHTTPServer(
        service=service,
        capability_token=capability_token or secrets.token_urlsafe(32),
        static_manifest=manifest,
        port=port,
    )


def serve_browser_debugger(
    service: DebuggerService,
    *,
    asset_root: Path,
    port: int,
    open_browser: bool,
) -> int:
    """Bind, announce, optionally open, serve, and always close cleanly."""
    server = create_server(service, asset_root=asset_root, port=port)
    with server:
        url = f"{server.expected_origin}/#token={server.capability_token}"
        print(f"Visual debugger: {url}")
        if open_browser:
            try:
                opened = webbrowser.open(url)
            except Exception as exc:  # Browser integration is platform-owned.
                print(
                    f"warning: browser could not be opened ({exc}); use the URL above.",
                    file=sys.stderr,
                )
            else:
                if not opened:
                    print(
                        "warning: browser could not be opened; use the URL above.",
                        file=sys.stderr,
                    )
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            print("Visual debugger stopped.")
    return 0
