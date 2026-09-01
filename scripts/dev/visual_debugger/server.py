"""Core-free authenticated HTTP coordinator for debugger browser modes."""

from __future__ import annotations

import json
import re
import secrets
import sys
import webbrowser
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from inspect import Parameter, Signature, signature
from pathlib import Path
from socket import socket
from threading import Event, Lock, RLock, Thread
from typing import Any, Literal, Protocol, cast, get_args

from pydantic import BaseModel, ValidationError

_TOKEN_HEADER = "X-MARL-Debugger-Token"
_MAX_COMMAND_BODY_BYTES = 64 * 1024
_CLIENT_SOCKET_TIMEOUT_SECONDS = 2.0
_METRIC_REPORT_ROUTE = "/api/replay/metric-report"
_AUTHORING_COMMAND_ROUTE = "/api/dev/authoring/command"
_METRIC_REPORT_SUFFIX = ".marlbg-metrics.json"
_METRIC_REPORT_CONTENT_TYPE = "application/json; charset=utf-8"
_SAFE_METRIC_REPORT_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
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
_REQUIRED_RUNTIME_ASSET_PATHS = (
    "index.html",
    "styles.css",
    "src/api.js",
    "src/authorized-presentation-adapter.js",
    "src/authorized-presentation-normalizer.js",
    "src/authorized-presentation-schema.js",
    "src/choreography-painter.js",
    "src/choreography-plan.js",
    "src/choreography.js",
    "src/controls.js",
    "src/display.js",
    "src/explanations.js",
    "src/frame-normalizer.js",
    "src/icons.js",
    "src/layout.js",
    "src/main.js",
    "src/panels.js",
    "src/presentation-install.js",
    "src/replay-controls.js",
    "src/replay-export.js",
    "src/replay-frame-normalizer.js",
    "src/routes.js",
    "src/scene.js",
    "src/tooltip.js",
    "src/vocabulary.js",
    "assets/fonts/AtkinsonHyperlegible-Regular.woff2",
    "assets/fonts/AtkinsonHyperlegible-Bold.woff2",
)

type DebuggerServerMode = Literal["live", "replay"]
type DebuggerProductKind = Literal["combat_debugger", "replay_viewer"]

_BOOTSTRAP_ROUTE = "/bootstrap.js"
_PRODUCT_KIND_BY_MODE: dict[DebuggerServerMode, DebuggerProductKind] = {
    "live": "combat_debugger",
    "replay": "replay_viewer",
}
_PRODUCT_TITLE_BY_KIND: dict[DebuggerProductKind, str] = {
    "combat_debugger": "MARL-BattleGrounds DevClient",
    "replay_viewer": "MARL-BattleGrounds Replay Viewer",
}


@dataclass(frozen=True, slots=True)
class HttpRouteSet:
    """Exact API routes exposed by one debugger server mode."""

    frame: str
    command: str
    timeline: str | None
    presentation: str
    metric_report: str | None


LIVE_HTTP_ROUTES = HttpRouteSet(
    frame="/api/frame",
    command="/api/command",
    timeline=None,
    presentation="/api/presentation/frame",
    metric_report=None,
)
REPLAY_HTTP_ROUTES = HttpRouteSet(
    frame="/api/frame",
    command="/api/replay/command",
    timeline="/api/replay/timeline",
    presentation="/api/presentation/frame",
    metric_report=_METRIC_REPORT_ROUTE,
)


class HttpPayloadResult(Protocol):
    """Structural payload result shared by read and command operations."""

    @property
    def outcome(self) -> str: ...

    @property
    def payload(self) -> object: ...


class HttpCommandResult(HttpPayloadResult, Protocol):
    """Payload result whose command may request host shutdown."""

    @property
    def shutdown_requested(self) -> bool: ...


class HttpMetricReportResult(Protocol):
    """Structural result for one canonical replay metric artifact."""

    @property
    def outcome(self) -> str: ...

    @property
    def payload(self) -> bytes | None: ...

    @property
    def filename(self) -> str | None: ...


class _LegacyServiceOperations(Protocol):
    def current_frame(self) -> object: ...

    def current_presentation(self) -> HttpPayloadResult: ...

    def apply_command(self, request: BaseModel) -> HttpCommandResult: ...


def _default_result_status(result: HttpPayloadResult) -> HTTPStatus:
    """Preserve the established service-outcome HTTP mapping."""
    outcome = _require_string(result.outcome, name="outcome")
    return {
        "invalid_cursor": HTTPStatus.UNPROCESSABLE_ENTITY,
        "audience_unavailable": HTTPStatus.UNPROCESSABLE_ENTITY,
        "stale_revision": HTTPStatus.CONFLICT,
        "command_id_conflict": HTTPStatus.CONFLICT,
        "server_shutting_down": HTTPStatus.SERVICE_UNAVAILABLE,
        "service_faulted": HTTPStatus.INTERNAL_SERVER_ERROR,
    }.get(outcome, HTTPStatus.OK)


def _presentation_result_status(result: HttpPayloadResult) -> HTTPStatus:
    """Map the fixed additive presentation-resource outcome contract."""
    outcome = _require_string(result.outcome, name="outcome")
    if outcome == "response":
        return HTTPStatus.OK
    if outcome == "audience_unavailable":
        return HTTPStatus.UNPROCESSABLE_ENTITY
    raise ValueError("Unknown presentation resource outcome.")


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Debugger service results require a string {name}.")
    return value


def _require_boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"Debugger service results require a boolean {name}.")
    return value


def _require_http_status(value: object) -> HTTPStatus:
    if not isinstance(value, HTTPStatus):
        raise TypeError("Result status resolver must return HTTPStatus.")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class HttpAuthoringBinding:
    """Strict live-only command boundary for DevClient authoring operations."""

    request_model: type[BaseModel]
    apply_command: Callable[[BaseModel], BaseModel]

    def __post_init__(self) -> None:
        request_model = cast(object, self.request_model)
        if not isinstance(request_model, type) or not issubclass(
            request_model, BaseModel
        ):
            raise TypeError(
                "authoring request_model must be an exact Pydantic model class."
            )
        if not callable(self.apply_command):
            raise TypeError("authoring apply_command must be callable.")


def _validate_metric_report_filename(filename: object) -> str:
    """Validate the complete ASCII attachment basename at the HTTP boundary."""
    if type(filename) is not str or not filename.isascii():
        raise TypeError("Metric report filename must be an ASCII string.")
    if not filename.endswith(_METRIC_REPORT_SUFFIX):
        raise ValueError("Metric report filename has the wrong suffix.")
    if filename.count(_METRIC_REPORT_SUFFIX) != 1:
        raise ValueError("Metric report filename contains an injected suffix.")
    stem = filename.removesuffix(_METRIC_REPORT_SUFFIX)
    if (
        _SAFE_METRIC_REPORT_STEM.fullmatch(stem) is None
        or stem.strip("._-") != stem
        or stem in (".", "..")
    ):
        raise ValueError("Metric report filename is not a safe basename.")
    return filename


def _validate_metric_report_result(
    result: object,
) -> tuple[Literal["available", "missing", "forbidden"], bytes | None, str | None]:
    """Validate a structural service result before any HTTP response bytes."""
    typed_result = cast(HttpMetricReportResult, result)
    outcome = typed_result.outcome
    payload = typed_result.payload
    filename = typed_result.filename
    if type(outcome) is not str or outcome not in (
        "available",
        "missing",
        "forbidden",
    ):
        raise ValueError("Unknown metric report outcome.")
    if outcome == "available":
        if type(payload) is not bytes:
            raise TypeError("Available metric report payload must be immutable bytes.")
        return outcome, payload, _validate_metric_report_filename(filename)
    if payload is not None or filename is not None:
        raise ValueError("Unavailable metric report cannot carry response data.")
    return outcome, None, None


def _validate_response_headers(
    headers: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Validate the finite optional response-header surface before status write."""
    if type(headers) is not tuple:
        raise TypeError("Optional response headers must be an immutable tuple.")
    seen: set[str] = set()
    for header in headers:
        if type(header) is not tuple or len(header) != 2:
            raise TypeError("Each optional response header must be an exact pair.")
        name, value = header
        if type(name) is not str or type(value) is not str:
            raise TypeError(
                "Optional response header names and values must be strings."
            )
        if name != "Content-Disposition" or name in seen:
            raise ValueError("Optional response header is not allowlisted.")
        if not value.isascii() or "\r" in value or "\n" in value:
            raise ValueError("Optional response header value is unsafe.")
        seen.add(name)
    return headers


@dataclass(frozen=True, slots=True, kw_only=True)
class HttpCoordinatorBinding:
    """Injected protocol and service operations for one isolated server mode."""

    mode: DebuggerServerMode
    routes: HttpRouteSet
    request_model: type[BaseModel]
    error_factory: Callable[..., object]
    current_frame: Callable[[], object]
    apply_command: Callable[..., HttpCommandResult]
    current_presentation: Callable[[], HttpPayloadResult]
    current_timeline: Callable[[], object] | None = None
    current_metric_report: Callable[[], HttpMetricReportResult] | None = None
    result_status: Callable[[HttpCommandResult], HTTPStatus] = _default_result_status
    authoring: HttpAuthoringBinding | None = None

    @property
    def product_kind(self) -> DebuggerProductKind:
        """Return the public product identity owned by this exact route binding."""
        return _PRODUCT_KIND_BY_MODE[self.mode]

    def __post_init__(self) -> None:
        if self.mode not in ("live", "replay"):
            raise ValueError("debugger server mode must be 'live' or 'replay'.")
        expected_routes = (
            LIVE_HTTP_ROUTES if self.mode == "live" else REPLAY_HTTP_ROUTES
        )
        if self.routes != expected_routes:
            raise ValueError(
                f"{self.mode} debugger mode requires its exact HTTP route set."
            )
        request_model = cast(object, self.request_model)
        if not isinstance(request_model, type) or not issubclass(
            request_model, BaseModel
        ):
            raise TypeError("request_model must be an exact Pydantic model class.")
        operations = {
            "error_factory": self.error_factory,
            "current_frame": self.current_frame,
            "apply_command": self.apply_command,
            "current_presentation": self.current_presentation,
            "result_status": self.result_status,
        }
        for name, operation in operations.items():
            if not callable(operation):
                raise TypeError(f"{name} must be callable.")
        if self.mode == "live" and self.current_timeline is not None:
            raise ValueError("live debugger mode cannot expose a replay timeline.")
        if self.mode == "replay" and not callable(self.current_timeline):
            raise ValueError("replay debugger mode requires a timeline operation.")
        if self.mode == "live" and self.current_metric_report is not None:
            raise ValueError("live debugger mode cannot expose replay metrics.")
        if self.mode == "replay" and not callable(self.current_metric_report):
            raise ValueError("replay debugger mode requires a metric-report operation.")
        if self.mode == "replay" and self.authoring is not None:
            raise ValueError("replay debugger mode cannot expose DevClient authoring.")
        authoring = cast(object, self.authoring)
        if authoring is not None and not isinstance(authoring, HttpAuthoringBinding):
            raise TypeError("authoring must be an exact HttpAuthoringBinding.")


@dataclass(frozen=True, slots=True, kw_only=True)
class HttpCoordinatorReplacement:
    """Fully constructed service/binding pair eligible for atomic installation."""

    service: object
    binding: HttpCoordinatorBinding

    def __post_init__(self) -> None:
        binding = cast(object, self.binding)
        if not isinstance(binding, HttpCoordinatorBinding):
            raise TypeError("coordinator replacement requires an exact binding.")


@dataclass(frozen=True, slots=True, kw_only=True)
class HttpCoordinatorSnapshot:
    """Immutable identity for one active coordinator generation."""

    generation: int
    service: object
    binding: HttpCoordinatorBinding

    def __post_init__(self) -> None:
        generation = cast(object, self.generation)
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError("coordinator generation must be a non-negative integer.")
        binding = cast(object, self.binding)
        if not isinstance(binding, HttpCoordinatorBinding):
            raise TypeError("coordinator snapshot requires an exact binding.")


class HttpCoordinatorRouter:
    """Serialize requests and one monotonic live-to-replay binding replacement."""

    def __init__(self, *, service: object, binding: HttpCoordinatorBinding) -> None:
        raw_binding = cast(object, binding)
        if not isinstance(raw_binding, HttpCoordinatorBinding):
            raise TypeError("coordinator router requires an exact binding.")
        self._lock = RLock()
        self._active = HttpCoordinatorSnapshot(
            generation=0,
            service=service,
            binding=binding,
        )

    def snapshot(self) -> HttpCoordinatorSnapshot:
        """Return one coherent active service/binding identity."""
        with self._lock:
            return self._active

    @contextmanager
    def pinned_snapshot(self) -> Generator[HttpCoordinatorSnapshot]:
        """Pin one generation while a request is routed and answered."""
        with self._lock:
            yield self._active

    def compare_and_swap(
        self,
        *,
        expected: HttpCoordinatorSnapshot,
        replacement: HttpCoordinatorReplacement,
    ) -> bool:
        """Install one complete replay pair iff the expected live pair is active."""
        raw_expected = cast(object, expected)
        if not isinstance(raw_expected, HttpCoordinatorSnapshot):
            raise TypeError("expected coordinator state must be an exact snapshot.")
        raw_replacement = cast(object, replacement)
        if not isinstance(raw_replacement, HttpCoordinatorReplacement):
            raise TypeError(
                "replacement coordinator state must be an exact replacement."
            )
        if replacement.binding.mode != "replay":
            raise ValueError(
                "coordinator replacement must be monotonic live-to-replay."
            )
        with self._lock:
            active = self._active
            if (
                active.generation != expected.generation
                or active.service is not expected.service
                or active.binding is not expected.binding
            ):
                return False
            if active.binding.mode != "live":
                return False
            self._active = HttpCoordinatorSnapshot(
                generation=active.generation + 1,
                service=replacement.service,
                binding=replacement.binding,
            )
            return True


@dataclass(frozen=True, slots=True, kw_only=True)
class GracefulCloseResult:
    """Typed launcher outcome produced by an optional graceful-close callback."""

    exit_code: int
    message: str | None = None

    def __post_init__(self) -> None:
        exit_code = cast(object, self.exit_code)
        if (
            isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or not 0 <= exit_code <= 255
        ):
            raise ValueError("graceful-close exit_code must be an integer in [0, 255].")
        message = cast(object, self.message)
        if message is not None and (not isinstance(message, str) or not message):
            raise ValueError("graceful-close message must be null or non-empty.")


type GracefulCloseCallback = Callable[[], GracefulCloseResult]


def _legacy_live_binding(service: object) -> HttpCoordinatorBinding:
    """Derive the existing live wire types without importing their modules here."""
    current_frame = getattr(service, "current_frame", None)
    current_presentation = getattr(service, "current_presentation", None)
    apply_command = getattr(service, "apply_command", None)
    if (
        not callable(current_frame)
        or not callable(current_presentation)
        or not callable(apply_command)
    ):
        raise TypeError(
            "Debugger services require callable current_frame, current_presentation, "
            "and apply_command."
        )

    try:
        apply_signature = signature(apply_command)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Could not inspect the live debugger service contract."
        ) from exc
    request_model = _request_model_from_signature(apply_signature)
    error_model = _error_model_from_signature(apply_signature)
    typed_service = cast(_LegacyServiceOperations, service)

    def create_error(*, error_code: str, message: str) -> object:
        return error_model(
            error_code=error_code,
            message=message,
            latest_frame=None,
        )

    def call_current_frame() -> object:
        return typed_service.current_frame()

    def call_current_presentation() -> HttpPayloadResult:
        return typed_service.current_presentation()

    def call_apply_command(request: BaseModel) -> HttpCommandResult:
        return typed_service.apply_command(request)

    return HttpCoordinatorBinding(
        mode="live",
        routes=LIVE_HTTP_ROUTES,
        request_model=request_model,
        error_factory=create_error,
        current_frame=call_current_frame,
        apply_command=call_apply_command,
        current_presentation=call_current_presentation,
        current_metric_report=None,
    )


def _request_model_from_signature(apply_signature: Signature) -> type[BaseModel]:
    parameters = tuple(apply_signature.parameters.values())
    request_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    )
    if len(request_parameters) != 1:
        raise TypeError("Live apply_command must accept exactly one request model.")
    request_model = request_parameters[0].annotation
    if not isinstance(request_model, type) or not issubclass(request_model, BaseModel):
        raise TypeError("Live apply_command must annotate its exact request model.")
    return request_model


def _error_model_from_signature(apply_signature: Signature) -> type[BaseModel]:
    result_type = apply_signature.return_annotation
    payload_type = getattr(result_type, "__annotations__", {}).get("payload")
    for candidate in get_args(payload_type):
        if not isinstance(candidate, type) or not issubclass(candidate, BaseModel):
            continue
        if {"error_code", "message", "latest_frame"} <= set(candidate.model_fields):
            return candidate
    raise TypeError(
        "Live apply_command must return a result annotated with its API error model."
    )


@dataclass(frozen=True, slots=True)
class StaticAsset:
    """One exact URL-to-file entry in the runtime asset allowlist."""

    body: bytes
    content_type: str


def build_static_manifest(asset_root: Path) -> dict[str, StaticAsset]:
    """Validate and return the exact browser-runtime static asset allowlist."""
    root = asset_root.resolve(strict=True)
    required = tuple(root / relative for relative in _REQUIRED_RUNTIME_ASSET_PATHS)
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
    for candidate in dict.fromkeys(candidates):
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
    """Threaded loopback server carrying one monotonic active coordinator."""

    daemon_threads = True
    block_on_close = True

    def __init__(
        self,
        *,
        service: object,
        coordinator: HttpCoordinatorBinding,
        coordinator_router: HttpCoordinatorRouter | None = None,
        capability_token: str,
        static_manifest: dict[str, StaticAsset],
        port: int,
    ) -> None:
        if coordinator_router is None:
            self._coordinator_router = HttpCoordinatorRouter(
                service=service,
                binding=coordinator,
            )
        else:
            raw_router = cast(object, coordinator_router)
            if type(raw_router) is not HttpCoordinatorRouter:
                raise TypeError(
                    "coordinator_router must be an exact HttpCoordinatorRouter."
                )
            snapshot = coordinator_router.snapshot()
            if snapshot.service is not service or snapshot.binding is not coordinator:
                raise ValueError(
                    "coordinator_router must own the supplied service and binding."
                )
            self._coordinator_router = coordinator_router
        self.capability_token = capability_token
        self.static_manifest = static_manifest
        self.shutdown_started = Event()
        self._shutdown_lock = Lock()
        super().__init__(("127.0.0.1", port), DebuggerRequestHandler)
        host, actual_port = cast(tuple[str, int], self.server_address)
        self.expected_host = f"{host}:{actual_port}"
        self.expected_origin = f"http://{self.expected_host}"

    @property
    def coordinator_router(self) -> HttpCoordinatorRouter:
        """Expose the narrow dynamic coordinator for lifecycle integration."""
        return self._coordinator_router

    @property
    def coordinator(self) -> HttpCoordinatorBinding:
        """Preserve the legacy active-binding inspection surface."""
        return self._coordinator_router.snapshot().binding

    @property
    def debugger_service(self) -> Any:  # noqa: ANN401
        """Preserve the deliberately untyped active-service inspection surface."""
        return cast(Any, self._coordinator_router.snapshot().service)

    def coordinator_snapshot(self) -> HttpCoordinatorSnapshot:
        """Return one coherent service/binding generation for a future CAS."""
        return self._coordinator_router.snapshot()

    def install_replay_coordinator(
        self,
        *,
        expected: HttpCoordinatorSnapshot,
        replacement: HttpCoordinatorReplacement,
    ) -> bool:
        """Atomically replace the expected live coordinator with replay."""
        return self._coordinator_router.compare_and_swap(
            expected=expected,
            replacement=replacement,
        )

    def run_graceful_close(
        self,
        callback: GracefulCloseCallback,
    ) -> GracefulCloseResult:
        """Invoke one close callback while excluding HTTP requests and swaps."""
        if not callable(callback):
            raise TypeError("graceful-close callback must be callable.")
        with self._coordinator_router.pinned_snapshot():
            result = callback()
            raw_result = cast(object, result)
            if not isinstance(raw_result, GracefulCloseResult):
                raise TypeError(
                    "graceful-close callback must return GracefulCloseResult."
                )
            return result

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
        with self.debugger_server.coordinator_router.pinned_snapshot() as snapshot:
            self._handle_get(snapshot.binding)

    def _handle_get(self, coordinator: HttpCoordinatorBinding) -> None:
        route = self._route(coordinator)
        if route is None or not self._valid_request_origin(coordinator):
            return
        if route == _BOOTSTRAP_ROUTE:
            body = (
                "globalThis.__MARL_DEBUGGER_BOOTSTRAP__ = Object.freeze("
                + json.dumps(
                    {
                        "authoring_available": coordinator.authoring is not None,
                        "schema_version": 1,
                        "product_kind": coordinator.product_kind,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + ");\n"
            ).encode("ascii")
            self._send_bytes(
                HTTPStatus.OK,
                body,
                content_type="text/javascript; charset=utf-8",
            )
            return
        if route == coordinator.routes.frame:
            if not self._authenticated(coordinator):
                return
            try:
                frame = coordinator.current_frame()
            except Exception:
                self._send_internal_error(coordinator)
                return
            try:
                self._send_model(HTTPStatus.OK, frame)
            except BrokenPipeError, ConnectionError, TimeoutError:
                self.close_connection = True
            except Exception:
                self._send_internal_error(coordinator)
            return
        presentation_route = coordinator.routes.presentation
        if route == presentation_route:
            if not self._authenticated(coordinator):
                return
            try:
                result = coordinator.current_presentation()
                status = _presentation_result_status(result)
                payload = result.payload
            except Exception:
                self._send_internal_error(coordinator)
                return
            try:
                self._send_model(status, payload)
            except BrokenPipeError, ConnectionError, TimeoutError:
                self.close_connection = True
            except Exception:
                self._send_internal_error(coordinator)
            return
        metric_report_route = coordinator.routes.metric_report
        if metric_report_route is not None and route == metric_report_route:
            if not self._authenticated(coordinator):
                return
            current_metric_report = coordinator.current_metric_report
            if current_metric_report is None:
                self._send_internal_error(coordinator)
                return
            try:
                outcome, payload, filename = _validate_metric_report_result(
                    current_metric_report()
                )
            except Exception:
                self._send_internal_error(coordinator)
                return
            if outcome == "missing":
                self._send_api_error(
                    coordinator,
                    HTTPStatus.NOT_FOUND,
                    error_code="not_found",
                    message="No metric report is available for this replay.",
                )
                return
            if outcome == "forbidden":
                self._send_api_error(
                    coordinator,
                    HTTPStatus.FORBIDDEN,
                    error_code="audience_unavailable",
                    message="Metric reports are available only in Oracle View.",
                )
                return
            if payload is None or filename is None:
                self._send_internal_error(coordinator)
                return
            try:
                self._send_bytes(
                    HTTPStatus.OK,
                    payload,
                    content_type=_METRIC_REPORT_CONTENT_TYPE,
                    response_headers=(
                        (
                            "Content-Disposition",
                            f'attachment; filename="{filename}"',
                        ),
                    ),
                )
            except BrokenPipeError, ConnectionError, TimeoutError:
                self.close_connection = True
            except Exception:
                self._send_internal_error(coordinator)
            return
        if route == coordinator.routes.timeline:
            if not self._authenticated(coordinator):
                return
            current_timeline = coordinator.current_timeline
            if current_timeline is None:
                self._send_internal_error(coordinator)
                return
            try:
                timeline = current_timeline()
            except Exception:
                self._send_internal_error(coordinator)
                return
            try:
                self._send_model(HTTPStatus.OK, timeline)
            except BrokenPipeError, ConnectionError, TimeoutError:
                self.close_connection = True
            except Exception:
                self._send_internal_error(coordinator)
            return
        asset = self.debugger_server.static_manifest.get(route)
        if asset is None:
            self._send_api_error(
                coordinator,
                HTTPStatus.NOT_FOUND,
                error_code="not_found",
                message="No such debugger route.",
            )
            return
        self._send_bytes(HTTPStatus.OK, asset.body, content_type=asset.content_type)

    def do_POST(self) -> None:
        with self.debugger_server.coordinator_router.pinned_snapshot() as snapshot:
            self._handle_post(snapshot.binding)

    def _handle_post(self, coordinator: HttpCoordinatorBinding) -> None:
        route = self._route(coordinator)
        if route is None or not self._valid_request_origin(coordinator):
            return
        metric_report_route = coordinator.routes.metric_report
        if metric_report_route is not None and route == metric_report_route:
            if not self._authenticated(coordinator):
                return
            self._send_api_error(
                coordinator,
                HTTPStatus.METHOD_NOT_ALLOWED,
                error_code="method_not_allowed",
                message="Method is not supported by this debugger.",
            )
            return
        if route == _AUTHORING_COMMAND_ROUTE:
            self._handle_authoring_post(coordinator)
            return
        if route != coordinator.routes.command:
            self._send_api_error(
                coordinator,
                HTTPStatus.NOT_FOUND,
                error_code="not_found",
                message="No such debugger route.",
            )
            return
        if not self._authenticated(coordinator):
            return
        body = self._read_command_body(coordinator)
        if body is None:
            return
        try:
            request = coordinator.request_model.model_validate_json(body)
        except ValidationError as error:
            malformed_json = any(
                detail.get("type") == "json_invalid" for detail in error.errors()
            )
            self._send_api_error(
                coordinator,
                (
                    HTTPStatus.BAD_REQUEST
                    if malformed_json
                    else HTTPStatus.UNPROCESSABLE_ENTITY
                ),
                error_code="invalid_request",
                message=(
                    "Request body is not valid JSON."
                    if malformed_json
                    else (
                        "Request body does not match "
                        f"{coordinator.request_model.__name__}."
                    )
                ),
            )
            return

        try:
            result = coordinator.apply_command(request)
            shutdown_requested = _require_boolean(
                result.shutdown_requested,
                name="shutdown flag",
            )
        except Exception:
            self._send_internal_error(coordinator)
            return
        try:
            status = _require_http_status(coordinator.result_status(result))
            payload = result.payload
            self._send_model(status, payload)
            if shutdown_requested:
                self.wfile.flush()
        except BrokenPipeError, ConnectionError, TimeoutError:
            self.close_connection = True
        except Exception:
            self._send_internal_error(coordinator)
        finally:
            if shutdown_requested:
                self.debugger_server.request_shutdown()

    def _handle_authoring_post(self, coordinator: HttpCoordinatorBinding) -> None:
        """Apply one whole-draft command only for an installed live binding."""
        authoring = coordinator.authoring
        if coordinator.mode != "live" or authoring is None:
            self._send_api_error(
                coordinator,
                HTTPStatus.NOT_FOUND,
                error_code="not_found",
                message="No such debugger route.",
            )
            return
        if not self._authenticated(coordinator):
            return
        body = self._read_command_body(coordinator)
        if body is None:
            return
        try:
            request = authoring.request_model.model_validate_json(body)
        except ValidationError as error:
            malformed_json = any(
                detail.get("type") == "json_invalid" for detail in error.errors()
            )
            self._send_api_error(
                coordinator,
                HTTPStatus.BAD_REQUEST
                if malformed_json
                else HTTPStatus.UNPROCESSABLE_ENTITY,
                error_code="invalid_request",
                message=(
                    "Request body is not valid JSON."
                    if malformed_json
                    else (
                        "Request body does not match "
                        f"{authoring.request_model.__name__}."
                    )
                ),
            )
            return
        try:
            response = authoring.apply_command(request)
            self._send_model(HTTPStatus.OK, response)
        except BrokenPipeError, ConnectionError, TimeoutError:
            self.close_connection = True
        except Exception:
            self._send_internal_error(coordinator)

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
        with self.debugger_server.coordinator_router.pinned_snapshot() as snapshot:
            self._send_api_error(
                snapshot.binding,
                HTTPStatus.METHOD_NOT_ALLOWED,
                error_code="method_not_allowed",
                message="Method is not supported by this debugger.",
            )

    def _route(self, coordinator: HttpCoordinatorBinding) -> str | None:
        raw_target = self.path
        if (
            not raw_target.startswith("/")
            or raw_target.startswith("//")
            or "?" in raw_target
            or "#" in raw_target
        ):
            self._send_api_error(
                coordinator,
                HTTPStatus.NOT_FOUND,
                error_code="not_found",
                message="No such debugger route.",
            )
            return None
        return raw_target

    def _valid_request_origin(self, coordinator: HttpCoordinatorBinding) -> bool:
        hosts = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        fetch_sites = self.headers.get_all("Sec-Fetch-Site", [])
        if len(hosts) != 1 or hosts[0] != self.debugger_server.expected_host:
            self._send_api_error(
                coordinator,
                HTTPStatus.FORBIDDEN,
                error_code="forbidden_origin",
                message="Request origin is not authorized.",
            )
            return False
        if len(origins) > 1 or (
            origins and origins[0] != self.debugger_server.expected_origin
        ):
            self._send_api_error(
                coordinator,
                HTTPStatus.FORBIDDEN,
                error_code="forbidden_origin",
                message="Request origin is not authorized.",
            )
            return False
        if len(fetch_sites) > 1 or (fetch_sites and fetch_sites[0] == "cross-site"):
            self._send_api_error(
                coordinator,
                HTTPStatus.FORBIDDEN,
                error_code="forbidden_origin",
                message="Request origin is not authorized.",
            )
            return False
        return True

    def _authenticated(self, coordinator: HttpCoordinatorBinding) -> bool:
        supplied = self.headers.get_all(_TOKEN_HEADER, [])
        if len(supplied) != 1 or not secrets.compare_digest(
            supplied[0],
            self.debugger_server.capability_token,
        ):
            self._send_api_error(
                coordinator,
                HTTPStatus.UNAUTHORIZED,
                error_code="unauthorized",
                message="Debugger capability is missing or invalid.",
            )
            return False
        return True

    def _read_command_body(
        self,
        coordinator: HttpCoordinatorBinding,
    ) -> bytes | None:
        if self.headers.get_all("Transfer-Encoding", []):
            self._send_api_error(
                coordinator,
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
                coordinator,
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                error_code="unsupported_media_type",
                message="Command requests require application/json.",
            )
            return None
        raw_lengths = self.headers.get_all("Content-Length", [])
        if len(raw_lengths) != 1:
            self._send_api_error(
                coordinator,
                HTTPStatus.LENGTH_REQUIRED,
                error_code="invalid_request",
                message="Command requests require one Content-Length value.",
            )
            return None
        raw_length = raw_lengths[0]
        if not raw_length.isascii() or not raw_length.isdecimal():
            self._send_api_error(
                coordinator,
                HTTPStatus.BAD_REQUEST,
                error_code="invalid_request",
                message="Content-Length must be a non-negative integer.",
            )
            return None
        length = int(raw_length)
        if length > _MAX_COMMAND_BODY_BYTES:
            self._send_api_error(
                coordinator,
                HTTPStatus.CONTENT_TOO_LARGE,
                error_code="payload_too_large",
                message="Command request exceeds the debugger body limit.",
            )
            return None
        try:
            body = self.rfile.read(length)
        except TimeoutError, OSError:
            self._send_api_error(
                coordinator,
                HTTPStatus.REQUEST_TIMEOUT,
                error_code="invalid_request",
                message="Command request body was not received in time.",
            )
            return None
        if len(body) != length:
            self._send_api_error(
                coordinator,
                HTTPStatus.BAD_REQUEST,
                error_code="invalid_request",
                message="Command request body ended before Content-Length.",
            )
            return None
        return body

    def _send_internal_error(self, coordinator: HttpCoordinatorBinding) -> None:
        self._send_api_error(
            coordinator,
            HTTPStatus.INTERNAL_SERVER_ERROR,
            error_code="internal_error",
            message="The debugger could not process this request.",
        )

    def _send_api_error(
        self,
        coordinator: HttpCoordinatorBinding,
        status: HTTPStatus,
        *,
        error_code: str,
        message: str,
    ) -> None:
        error = coordinator.error_factory(
            error_code=error_code,
            message=message,
        )
        self._send_model(status, error)

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
        response_headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        validated_headers = _validate_response_headers(response_headers)
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for name, value in validated_headers:
            self.send_header(name, value)
        for name, value in _SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


def create_server(
    service: object | None = None,
    *,
    asset_root: Path,
    port: int,
    capability_token: str | None = None,
    coordinator: HttpCoordinatorBinding | None = None,
    coordinator_router: HttpCoordinatorRouter | None = None,
) -> DebuggerHTTPServer:
    """Validate assets, then bind one debugger server to loopback."""
    if not 0 <= port <= 65535:
        raise ValueError(f"port must be in [0, 65535]; got {port}.")
    if coordinator_router is None:
        if service is None:
            raise TypeError(
                "service is required when coordinator_router is not supplied."
            )
        resolved_service = service
        resolved_coordinator = coordinator or _legacy_live_binding(service)
    else:
        raw_router = cast(object, coordinator_router)
        if type(raw_router) is not HttpCoordinatorRouter:
            raise TypeError(
                "coordinator_router must be an exact HttpCoordinatorRouter."
            )
        snapshot = coordinator_router.snapshot()
        if service is not None and snapshot.service is not service:
            raise ValueError(
                "coordinator_router does not own the supplied debugger service."
            )
        if coordinator is not None and snapshot.binding is not coordinator:
            raise ValueError(
                "coordinator_router does not own the supplied coordinator binding."
            )
        resolved_service = snapshot.service
        resolved_coordinator = snapshot.binding
    manifest = build_static_manifest(asset_root)
    return DebuggerHTTPServer(
        service=resolved_service,
        coordinator=resolved_coordinator,
        coordinator_router=coordinator_router,
        capability_token=capability_token or secrets.token_urlsafe(32),
        static_manifest=manifest,
        port=port,
    )


def serve_browser_debugger(
    service: object | None = None,
    *,
    asset_root: Path,
    port: int,
    open_browser: bool,
    coordinator: HttpCoordinatorBinding | None = None,
    coordinator_router: HttpCoordinatorRouter | None = None,
    authoring: HttpAuthoringBinding | None = None,
    graceful_close: GracefulCloseCallback | None = None,
) -> int:
    """Bind, announce, optionally open, serve, and always close cleanly."""
    if authoring is not None:
        if coordinator_router is not None:
            raise ValueError(
                "router-backed servers must install authoring on their live binding."
            )
        if coordinator is None:
            if service is None:
                raise TypeError("authoring requires a live debugger service.")
            coordinator = _legacy_live_binding(service)
        if coordinator.mode != "live" or coordinator.authoring is not None:
            raise ValueError("authoring can be installed once on a live binding.")
        coordinator = replace(coordinator, authoring=authoring)
    server = create_server(
        service,
        asset_root=asset_root,
        port=port,
        coordinator=coordinator,
        coordinator_router=coordinator_router,
    )
    exit_code = 0
    with server:
        url = f"{server.expected_origin}/#token={server.capability_token}"
        startup_title = _PRODUCT_TITLE_BY_KIND[server.coordinator.product_kind]
        print(f"{startup_title}: {url}")
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
            active_title = _PRODUCT_TITLE_BY_KIND[server.coordinator.product_kind]
            if graceful_close is None:
                print(f"{active_title} stopped.")
            else:
                try:
                    close_result = server.run_graceful_close(graceful_close)
                except Exception:
                    print(
                        f"error: {active_title} graceful close failed.",
                        file=sys.stderr,
                    )
                    exit_code = 1
                else:
                    if close_result.message is not None:
                        print(
                            close_result.message,
                            file=(
                                sys.stdout
                                if close_result.exit_code == 0
                                else sys.stderr
                            ),
                        )
                    exit_code = close_result.exit_code
    return exit_code
