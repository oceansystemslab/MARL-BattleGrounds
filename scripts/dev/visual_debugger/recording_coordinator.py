"""Same-process live-recording to read-only replay HTTP handoff."""

from __future__ import annotations

from scripts.dev.visual_debugger.protocol import ApiErrorV2, CommandRequestV1
from scripts.dev.visual_debugger.replay_protocol import (
    ReplayApiErrorV1,
    ReplayCommandRequestV1,
)
from scripts.dev.visual_debugger.replay_service import ReplayViewerService
from scripts.dev.visual_debugger.server import (
    LIVE_HTTP_ROUTES,
    REPLAY_HTTP_ROUTES,
    GracefulCloseResult,
    HttpAuthoringBinding,
    HttpCoordinatorBinding,
    HttpCoordinatorReplacement,
    HttpCoordinatorRouter,
)
from scripts.dev.visual_debugger.service import (
    DebuggerService,
    ServiceCommandResult,
)


class RecordingDebuggerCoordinator:
    """Own one monotonic live-recording to replay coordinator transition."""

    __slots__ = ("_live_binding", "_router", "_service")

    def __init__(
        self,
        service: DebuggerService,
        *,
        authoring: HttpAuthoringBinding | None = None,
    ) -> None:
        if type(service) is not DebuggerService:
            raise TypeError("recording coordinator requires exact DebuggerService")
        if service.recording_status is None:
            raise ValueError("recording coordinator requires recording-enabled service")
        self._service = service
        self._live_binding = HttpCoordinatorBinding(
            mode="live",
            routes=LIVE_HTTP_ROUTES,
            request_model=CommandRequestV1,
            error_factory=ApiErrorV2,
            current_frame=service.current_frame,
            apply_command=self.apply_command,
            current_presentation=service.current_presentation,
            current_metric_report=None,
            authoring=authoring,
        )
        self._router = HttpCoordinatorRouter(
            service=service,
            binding=self._live_binding,
        )

    @property
    def router(self) -> HttpCoordinatorRouter:
        """Return the exact router to attach before the loopback server binds."""
        return self._router

    @property
    def service(self) -> DebuggerService:
        """Return the live service owned until a replay handoff commits."""
        return self._service

    def apply_command(self, request: CommandRequestV1) -> ServiceCommandResult:
        """Apply one live command and install a prepared replay before response."""
        result = self._service.apply_command(request)
        handoff = result.replay_handoff
        if handoff is None:
            return result
        if type(handoff) is not ReplayViewerService:
            raise TypeError("recording handoff must be exact ReplayViewerService")

        replay_binding = HttpCoordinatorBinding(
            mode="replay",
            routes=REPLAY_HTTP_ROUTES,
            request_model=ReplayCommandRequestV1,
            error_factory=ReplayApiErrorV1,
            current_frame=handoff.current_frame,
            apply_command=handoff.apply_command,
            current_timeline=handoff.current_timeline,
            current_presentation=handoff.current_presentation,
            current_metric_report=handoff.current_metric_report,
        )
        expected = self._router.snapshot()
        if (
            expected.service is not self._service
            or expected.binding is not self._live_binding
            or not self._router.compare_and_swap(
                expected=expected,
                replacement=HttpCoordinatorReplacement(
                    service=handoff,
                    binding=replay_binding,
                ),
            )
        ):
            raise RuntimeError("recording replay handoff lost coordinator authority")
        return result

    def graceful_close(self) -> GracefulCloseResult:
        """Map the service's host-only Ctrl-C closeout to launcher semantics."""
        close_result = self._service.close_recording_for_keyboard_interrupt()
        return GracefulCloseResult(
            exit_code=0 if close_result.saved else 1,
            message=close_result.message,
        )


__all__ = ["RecordingDebuggerCoordinator"]
