"""Switch one loopback server from live recording to saved replay review.

The launcher installs ``RecordingDebuggerCoordinator.router`` before serving.
A successful recording handoff replaces the router binding once, before its
response is returned. Commands and recording behavior remain owned by their
existing services. This module has no independent CLI or file writer.
"""

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
    """Coordinate a recording-enabled debugger and its eventual replay viewer.

    Parameters
    ----------
    service : DebuggerService
        Exact live service with recording enabled.
    authoring : HttpAuthoringBinding or None, optional
        Optional map/scenario authoring routes available while the service is live.

    Raises
    ------
    TypeError
        If ``service`` is not the exact supported service type.
    ValueError
        If the live service has no recording status.

    Notes
    -----
    Construction creates the router binding but does not open a socket or write data.
    """

    __slots__ = ("_live_binding", "_router", "_service")

    def __init__(
        self,
        service: DebuggerService,
        *,
        authoring: HttpAuthoringBinding | None = None,
    ) -> None:
        """Check the live recording service and bind its initial HTTP route set."""
        if type(service) is not DebuggerService:
            raise TypeError("recording coordinator requires exact DebuggerService")
        if service.recording_status is None:
            raise ValueError("recording coordinator requires recording-enabled service")
        self._service = service
        self._live_binding = HttpCoordinatorBinding(
            mode="live",
            initial_show_ranges=service.session.show_ranges,
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
        """Return the router to install before the loopback server opens its socket."""
        return self._router

    @property
    def service(self) -> DebuggerService:
        """Return the original live service that owns recording operations."""
        return self._service

    def apply_command(self, request: CommandRequestV1) -> ServiceCommandResult:
        """Apply a live command and install any ready replay view before returning.

        Parameters
        ----------
        request : CommandRequestV1
            Validated live request handled by the debugger service.

        Returns
        -------
        ServiceCommandResult
            The live service response, after any requested replay handoff is committed.

        Raises
        ------
        TypeError
            If the prepared replay handoff has an unexpected service type.
        RuntimeError
            If another owner changed the router before this handoff could be installed.

        Notes
        -----
        The delegated command may advance the session or save a recording. Router
        changes
        use the exact prior binding so a stale handoff cannot replace newer state.
        """
        result = self._service.apply_command(request)
        handoff = result.replay_handoff
        if handoff is None:
            return result
        if type(handoff) is not ReplayViewerService:
            raise TypeError("recording handoff must be exact ReplayViewerService")

        replay_binding = HttpCoordinatorBinding(
            mode="replay",
            initial_show_ranges=handoff.show_ranges,
            routes=REPLAY_HTTP_ROUTES,
            request_model=ReplayCommandRequestV1,
            error_factory=ReplayApiErrorV1,
            current_frame=handoff.current_frame,
            apply_command=handoff.apply_command,
            current_timeline=handoff.current_timeline,
            current_presentation=handoff.current_presentation,
            current_metric_report=handoff.current_metric_report,
            metric_analysis=handoff.metric_analysis,
            metric_catalog=handoff.metric_catalog,
            episode_details=handoff.episode_details,
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
        """Close recording after a host keyboard interrupt and return launcher status.

        Returns
        -------
        GracefulCloseResult
            Exit code zero when recording was saved, otherwise one, with the service's
            explanatory message.

        Notes
        -----
        Delegates finalization and persistence to the live recording service.
        """
        close_result = self._service.close_recording_for_keyboard_interrupt()
        return GracefulCloseResult(
            exit_code=0 if close_result.saved else 1,
            message=close_result.message,
        )


__all__ = ["RecordingDebuggerCoordinator"]
