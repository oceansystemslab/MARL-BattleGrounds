"""Own the live browser session and serialize its commands.

``DebuggerService`` keeps one committed session and response frame under a
reentrant lock. Candidate frames and authorized presentations are checked
before installation. A bounded command cache prevents repeated execution
while an ID remains cached. Optional recording captures accepted transitions,
saves through the recorder, and hands verified output to the replay service.
The HTTP server owns transport and authentication.
"""

import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from secrets import token_urlsafe
from threading import RLock
from typing import Literal, cast

from marl_battlegrounds.rendering.evaluation_adapter import (
    build_visual_event_batch_v2,
)
from marl_battlegrounds.viewer.no_shared_visual import (
    build_live_no_shared_obs_visual_adjacent_slice_v1,
    build_live_no_shared_obs_visual_current_slice_v1,
)
from marl_battlegrounds.viewer.presentation_protocol import (
    PresentationResourceResultV1,
)
from marl_battlegrounds.viewer.replay_service import ReplayViewerService
from scripts.dev.visual_debugger.control import (
    DebuggerTransitionFailureStageV1,
    DebuggerTransitionFailureV1,
    set_combat_configuration,
    switch_scenario,
)
from scripts.dev.visual_debugger.frame import LiveDebuggerFrame, build_debugger_frame
from scripts.dev.visual_debugger.input import (
    InputDispatchResult,
    dispatch_command,
    normalize_key,
    recording_restart_intent_v1,
)
from scripts.dev.visual_debugger.live_presentation import (
    build_live_no_shared_obs_authorized_presentation_v1,
    build_live_oracle_authorized_presentation_v1,
    build_live_researcher_space_v1,
    build_live_shared_obs_authorized_presentation_v1,
)
from scripts.dev.visual_debugger.model import (
    DebuggerScenario,
    DebuggerSession,
    is_system_controller,
)
from scripts.dev.visual_debugger.protocol import (
    ActorPovLiveDebuggerFrameV2,
    ApiErrorV2,
    CancelSystemCommandV1,
    CommandRequestV1,
    CommandResponseV2,
    ConfirmDiscardAndReplaceCommandV1,
    DebuggerCommandV1,
    ExitCommandV1,
    FinishAndReviewCommandV1,
    FinishSystemCommandV1,
    KeyboardCommandV1,
    Preset,
    RecordingLifecycleV1,
    RecordingPersistenceErrorCodeV1,
    RecordingStatusV1,
    ResearcherLiveDebuggerFrameV2,
    RetrySaveCommandV1,
    ReviewReplayCommandV1,
    SaveAsCommandV1,
    SetCombatConfigurationCommandV1,
    SetPresetCommandV1,
    SetViewCommandV1,
    SharedObsAgentPovLiveDebuggerFrameV2,
    SystemChoiceV1,
    SystemOperationV1,
    ViewMode,
)
from scripts.dev.visual_debugger.recording import (
    DebuggerRecordingCloseCauseV1,
)
from scripts.dev.visual_debugger.replay_recorder import DebuggerReplayRecorder
from scripts.dev.visual_debugger.scenarios import STRESS_SCENARIOS
from scripts.dev.visual_debugger.system_worker import (
    PendingSystemOperation,
    SystemInstallation,
    SystemWorker,
)

_COMMAND_RECORD_LIMIT = 256
_CLOSED_RECORDING_PRESENTATION_KEYS = frozenset(("g", "v", "p", "?"))

type ServiceOutcome = Literal[
    "response",
    "stale_revision",
    "command_id_conflict",
    "server_shutting_down",
    "service_faulted",
]


@dataclass(frozen=True, slots=True)
class ServiceCommandResult:
    """One transport-neutral command outcome and its validated payload."""

    outcome: ServiceOutcome
    payload: CommandResponseV2 | ApiErrorV2
    shutdown_requested: bool = False
    replay_handoff: ReplayViewerService | None = None


@dataclass(frozen=True, slots=True)
class RecordingCloseResult:
    """Host-only result of one keyboard-interrupt recording closeout."""

    saved: bool
    message: str

    def __post_init__(self) -> None:
        """Require an exact boolean saved flag and a nonempty host message."""
        if type(self.saved) is not bool:
            raise TypeError("recording close saved flag must be a Python bool.")
        if type(self.message) is not str or not self.message:
            raise ValueError("recording close message must be nonempty.")


@dataclass(frozen=True, slots=True)
class _CommandRecord:
    """Remember one request fingerprint and whether it requested shutdown."""

    fingerprint: str
    shutdown_requested: bool


@dataclass(frozen=True, slots=True)
class _RecordingResponseCandidate:
    """Hold a validated frame, response, and cache update before state is committed."""

    revision: int
    frame: LiveDebuggerFrame
    response: CommandResponseV2
    command_records: OrderedDict[tuple[str, str], _CommandRecord]
    shutdown_requested: bool


@dataclass(frozen=True, slots=True)
class _EndpointResponseCandidates:
    """Hold checked success and save-failure responses for a sealed episode."""

    saved: _RecordingResponseCandidate
    failures: dict[
        RecordingPersistenceErrorCodeV1,
        _RecordingResponseCandidate,
    ]


@dataclass(frozen=True, slots=True)
class _FailureCloseoutCandidates:
    """Hold checked responses for saving the last accepted prefix after a failure."""

    saved: _RecordingResponseCandidate
    failures: dict[
        RecordingPersistenceErrorCodeV1,
        _RecordingResponseCandidate,
    ]


class DebuggerService:
    """Serialize commands around one immutable debugger session."""

    def __init__(
        self,
        session: DebuggerSession,
        *,
        view_mode: ViewMode,
        preset: Preset | Literal["technical", "debug"],
        include_stress: bool,
        session_id: str | None = None,
        recorder: DebuggerReplayRecorder | None = None,
        offered_systems: Mapping[str, str] | None = None,
    ) -> None:
        """Build the live service around an existing immutable session.

        Parameters
        ----------
        session : DebuggerSession
            Current live episode and matching captured frame.
        view_mode : ViewMode
            Researcher or actor view passed to the frame builder.
        preset : Preset or {'technical', 'debug'}
            Accepted compatibility preset. All accepted values start in analysis mode.
        include_stress : bool
            Whether registered stress scenarios may be selected.
        session_id : str or None, optional
            Session identity for protocol frames. Missing or empty values create a
            random identity. Defaults to None.
        recorder : DebuggerReplayRecorder or None, optional
            Exact recorder owning this open episode prefix. None requires the debug
            capture profile; recording requires metric-complete capture.

        offered_systems : mapping of str to str, optional
            Launcher names mapped to trusted module:function factories. Only a
            selected name loads. Browser input cannot supply a factory or URL.

        Raises
        ------
        TypeError
            The recorder has an unsupported exact type.
        ValueError
            Scenario access, preset, capture profile, or recorder identity is invalid.

        Notes
        -----
        Construction builds the initial host frame and lock. It does not advance the
        simulator or publish a replay file.
        """
        if session.scenario_name in STRESS_SCENARIOS and not include_stress:
            msg = (
                f"stress scenario {session.scenario_name!r} requires "
                "include_stress=True."
            )
            raise ValueError(msg)
        if preset not in ("presentation", "analysis", "technical", "debug"):
            raise ValueError("unknown debugger preset")
        self._session: DebuggerSession = session
        self._view_mode: ViewMode = view_mode
        self._preset: Preset = "analysis"
        self._include_stress = include_stress
        self._session_id = session_id or token_urlsafe(24)
        self._revision = 0
        self._shutting_down = False
        self._faulted = False
        self._lock = RLock()
        self._command_records: OrderedDict[
            tuple[str, str],
            _CommandRecord,
        ] = OrderedDict()
        self._recorder: DebuggerReplayRecorder | None = self._validated_recorder(
            self._session,
            recorder,
        )
        self._offered_systems = dict(offered_systems or {})
        if any(
            not is_system_controller(f"system:{name}") or not reference.strip()
            for name, reference in self._offered_systems.items()
        ):
            raise ValueError(
                "Offered Systems need a bounded name and factory reference"
            )
        self._system_choices = tuple(
            SystemChoiceV1(
                id=f"system:{name}",
                label=name.replace("_", " ").replace("-", " ").title(),
            )
            for name in self._offered_systems
        )
        self._system_worker: SystemWorker | None = None
        self._system_evidence: SystemInstallation | None = None
        self._system_confirmed_discard = False
        self._system_failed = False
        self._frame = self._build_frame()

    @staticmethod
    def _validated_recorder(
        session: DebuggerSession,
        recorder: DebuggerReplayRecorder | None,
    ) -> DebuggerReplayRecorder | None:
        """Require the recorder to own this exact open capture, or require debug
        mode.
        """
        if recorder is None:
            if session.evaluation_context.capture_profile != "debug":
                raise ValueError(
                    "unrecorded live debugger sessions require the debug profile."
                )
            return None
        raw_recorder = cast(object, recorder)
        if type(raw_recorder) is not DebuggerReplayRecorder:
            raise TypeError("recorder must be an exact supported debugger recorder.")
        if session.evaluation_context.capture_profile != "evaluation_metric_complete":
            raise ValueError("recording sessions require metric-complete capture.")
        if (
            recorder.lifecycle != "recording"
            or recorder.context != session.evaluation_context
            or recorder.current_frame != session.current_evaluation_frame
            or recorder.validated_transition_count
            != session.current_evaluation_frame.frame_index
        ):
            raise ValueError(
                "recorder must own the current open debugger episode prefix."
            )
        return recorder

    @property
    def session(self) -> DebuggerSession:
        """Return the current immutable session for diagnostics and tests."""
        with self._lock:
            return self._session

    def load_scenario(
        self,
        scenario: DebuggerScenario,
        *,
        on_installed: Callable[[], None] | None = None,
    ) -> DebuggerSession | PendingSystemOperation:
        """Replace an unrecorded live episode after validating its full response.

        Parameters
        ----------
        scenario : DebuggerScenario
            Exact compiled scenario to load into a reset episode.
        on_installed : callable or None, optional
            Trusted host callback called once after successful adoption. None does
            no extra work. Cancellation and failed candidates never call it.

        Returns
        -------
        DebuggerSession or PendingSystemOperation
            Newly installed session, or a declared System setup awaiting explicit
            completion. Pending work leaves the old game intact. Response and
            presentation checks precede replacement. Synchronous loads clear old
            command IDs; deferred loads retain the normal bounded command cache.

        Raises
        ------
        TypeError
            The scenario has the wrong exact type.
        RuntimeError
            The service is stopping, faulted, or owns a replay recording.
        ValueError
            Scenario setup or its candidate response fails validation.

        Notes
        -----
        The operation holds the service lock and performs reset work. It does not
        write an authored scenario file.
        """
        if type(scenario) is not DebuggerScenario:
            raise TypeError("scenario must be the exact DebuggerScenario type.")
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("the debugger is shutting down")
            if self._faulted:
                raise RuntimeError("the debugger service is faulted")
            if self._recorder is not None:
                raise RuntimeError(
                    "authored scenarios cannot replace a replay-recording session"
                )

            worker = self._system_worker
            if worker is not None:
                if worker.operation is not None:
                    worker.cancel(worker.operation)
                    raise RuntimeError(
                        "System work is being cancelled; retry after cleanup"
                    )
                base = self._session
                controllers = (base.team_a_controller, base.team_b_controller)
                if any(is_system_controller(value) for value in controllers):
                    operation = worker.prepare(
                        base,
                        self._offered_systems,
                        controllers,
                        lambda pair: switch_scenario(base, scenario, systems=pair),
                        directory=None,
                    )
                    operation.on_installed = on_installed
                    self._system_failed = False
                    self._system_confirmed_discard = False
                    self._revision += 1
                    self._frame = self._build_frame()
                    return operation
            candidate_session = switch_scenario(self._session, scenario)
            candidate_revision = self._revision + 1
            candidate_frame = self._build_frame(
                session=candidate_session,
                revision=candidate_revision,
            )
            self._build_authorized_presentation(
                session=candidate_session,
                raw_frame=candidate_frame,
                view_mode=self._view_mode,
            )

            self._session = candidate_session
            self._revision = candidate_revision
            self._frame = candidate_frame
            self._command_records.clear()
            if on_installed is not None:
                on_installed()
            return candidate_session

    @property
    def revision(self) -> int:
        """Return the current service/frame revision."""
        with self._lock:
            return self._revision

    @property
    def command_cache_size(self) -> int:
        """Return the bounded idempotency-record count."""
        with self._lock:
            return len(self._command_records)

    @property
    def evaluation_validated_transition_count(self) -> int:
        """Return the count from the committed episode frame."""
        with self._lock:
            if self._recorder is not None:
                return self._recorder.validated_transition_count
            return self._session.current_evaluation_frame.frame_index

    @property
    def evaluation_observer_lifecycle_state(self) -> str:
        """Return whether the committed episode is open or finished."""
        with self._lock:
            if self._recorder is not None:
                return self._recorder.observer_lifecycle_state
            incoming = self._session.incoming_evaluation_view
            if self._session.reached_declared_horizon or (
                incoming is not None
                and (incoming.transition.terminated or incoming.transition.truncated)
            ):
                return "sealed"
            return "open"

    @property
    def recording_status(self) -> RecordingStatusV1 | None:
        """Return the path-free recorder status without exposing its mutator."""
        with self._lock:
            return None if self._recorder is None else self._recorder.status

    def close_recording_for_keyboard_interrupt(self) -> RecordingCloseResult:
        """Try to save active capture before the hosting process exits.

        Returns
        -------
        RecordingCloseResult
            Whether capture is safe to leave and a host-facing status message.
            No active capture and an already saved replay both count as success.

        Notes
        -----
        The service lock covers response preparation and saving. A failed initial
        save is retried; a failed retry tries a deterministic recovery sibling.
        The revision and displayed recording status reflect the final result.
        Unexpected invariant or response-validation failures still propagate.
        """
        with self._lock:
            recorder = self._recorder
            if recorder is None or recorder.lifecycle == "discarded":
                return RecordingCloseResult(
                    saved=True,
                    message="No replay recording was active.",
                )
            if recorder.lifecycle in ("saved", "reviewing"):
                saved = recorder.saved_bundle
                if saved is None:
                    raise AssertionError("saved recording is missing its artifact.")
                return RecordingCloseResult(
                    saved=True,
                    message=(
                        f"Replay recording was already saved at {saved.replay_path}."
                    ),
                )

            close_cause: DebuggerRecordingCloseCauseV1 | None = (
                "keyboard_interrupt"
                if recorder.lifecycle == "recording"
                else recorder.close_cause
            )
            if close_cause is None:
                raise AssertionError("recording closeout is missing its close cause.")
            completion_reason = recorder.status.completion_reason
            candidate_revision = self._revision + 1
            captured_transition_count = recorder.validated_transition_count
            saved_frame = self._build_frame(
                revision=candidate_revision,
                recording_status=recorder.preview_status_v1(
                    captured_transition_count=captured_transition_count,
                    lifecycle="saved",
                    close_cause=close_cause,
                    completion_reason=completion_reason,
                ),
            )
            failure_frames = {
                error_code: self._build_frame(
                    revision=candidate_revision,
                    recording_status=recorder.preview_status_v1(
                        captured_transition_count=captured_transition_count,
                        lifecycle="persistence_failed",
                        close_cause=close_cause,
                        completion_reason=completion_reason,
                        persistence_error_code=error_code,
                    ),
                )
                for error_code in cast(
                    tuple[RecordingPersistenceErrorCodeV1, ...],
                    (
                        "target_unavailable",
                        "publication_failed",
                        "verification_failed",
                    ),
                )
            }
            self._build_authorized_presentation(
                session=self._session,
                raw_frame=saved_frame,
                view_mode=self._view_mode,
            )
            for failure_frame in failure_frames.values():
                self._build_authorized_presentation(
                    session=self._session,
                    raw_frame=failure_frame,
                    view_mode=self._view_mode,
                )

            if recorder.lifecycle == "persistence_failed":
                outcome = recorder.retry_save()
            else:
                outcome = recorder.finalize_and_save(close_cause)
                if outcome == "persistence_failed":
                    outcome = recorder.retry_save()
            if outcome == "persistence_failed":
                outcome = recorder.save_recovery_copy()

            self._revision = candidate_revision
            if outcome == "saved":
                saved = recorder.saved_bundle
                if saved is None:
                    raise AssertionError("successful closeout lacks a saved artifact.")
                self._frame = saved_frame
                return RecordingCloseResult(
                    saved=True,
                    message=f"Replay recording saved at {saved.replay_path}.",
                )

            error_code = recorder.persistence_error_code
            if error_code is None:
                raise AssertionError("failed closeout lacks a persistence error code.")
            self._frame = failure_frames[error_code]
            return RecordingCloseResult(
                saved=False,
                message=(
                    "Replay recording could not be saved; no recovery copy was written."
                ),
            )

    def _validated_transition_count(self) -> int:
        """Read the committed recorder count, or the unrecorded current frame index."""
        if self._recorder is not None:
            return self._recorder.validated_transition_count
        return self._session.current_evaluation_frame.frame_index

    @property
    def shutting_down(self) -> bool:
        """Return whether an accepted Exit has fenced new commands."""
        with self._lock:
            return self._shutting_down

    @property
    def faulted(self) -> bool:
        """Return whether an internal failure fenced all later commands."""
        with self._lock:
            return self._faulted

    def current_frame(self) -> LiveDebuggerFrame:
        """Read the current immutable transport frame under the service lock.

        Returns
        -------
        LiveDebuggerFrame
            The already built frame for the committed revision; no simulator work
            or new recording I/O is performed.
        """
        with self._lock:
            return self._frame

    def current_presentation(self) -> PresentationResourceResultV1:
        """Build the authorized display resource for the committed live snapshot.

        Returns
        -------
        PresentationResourceResultV1
            Checked presentation for the current researcher or actor view.

        Raises
        ------
        RuntimeError
            The retained transport frame differs from the service-owned snapshot.
        ValueError
            An authority, epoch, or display contract fails validation.

        Notes
        -----
        This holds the service lock while deriving host display data. It does not
        advance the simulator or expand an actor's information rights.
        """
        with self._lock:
            raw_frame = self._frame
            expected_frame = self._build_frame()
            if (
                type(raw_frame) is not type(expected_frame)
                or raw_frame != expected_frame
            ):
                raise RuntimeError(
                    "committed live frame diverged from service-owned state."
                )
            return self._build_authorized_presentation(
                session=self._session,
                raw_frame=raw_frame,
                view_mode=self._view_mode,
            )

    def _build_authorized_presentation(
        self,
        *,
        session: DebuggerSession,
        raw_frame: LiveDebuggerFrame,
        view_mode: ViewMode,
    ) -> PresentationResourceResultV1:
        """Build and strictly validate one parameterized live presentation."""
        context = session.evaluation_context
        current = session.current_evaluation_frame
        incoming = session.incoming_evaluation_view
        if view_mode == "researcher":
            if type(raw_frame) is not ResearcherLiveDebuggerFrameV2:
                raise RuntimeError(
                    "researcher live service lacks its committed Oracle frame."
                )
            presentation = build_live_oracle_authorized_presentation_v1(
                context,
                current,
                incoming,
                raw_frame,
            )
        else:
            if type(raw_frame) not in (
                ActorPovLiveDebuggerFrameV2,
                CancelSystemCommandV1,
                FinishSystemCommandV1,
                SetCombatConfigurationCommandV1,
                SystemChoiceV1,
                SystemOperationV1,
                SharedObsAgentPovLiveDebuggerFrameV2,
            ):
                raise RuntimeError(
                    "POV live service lacks its committed recipient frame."
                )
            recipient = session.controlled_global_slot
            oracle_raw_frame = self._build_frame(
                session=session,
                revision=raw_frame.revision,
                view_mode="researcher",
                preset=raw_frame.preset,
                recording_status=raw_frame.recording,
            )
            if type(oracle_raw_frame) is not ResearcherLiveDebuggerFrameV2:
                raise RuntimeError(
                    "POV live service cannot build its researcher-space epoch."
                )
            oracle_presentation = build_live_oracle_authorized_presentation_v1(
                context,
                current,
                incoming,
                oracle_raw_frame,
            )
            researcher_space = build_live_researcher_space_v1(oracle_presentation)
            if type(raw_frame) is SharedObsAgentPovLiveDebuggerFrameV2:
                presentation = build_live_shared_obs_authorized_presentation_v1(
                    context,
                    current,
                    incoming,
                    raw_frame,
                    authorized_recipient_global_slot=recipient,
                    pending_action=session.pending_actions[recipient],
                    researcher_space=researcher_space,
                )
            else:
                if type(raw_frame) is not ActorPovLiveDebuggerFrameV2:
                    raise AssertionError("POV raw-frame narrowing failed.")
                current_slice = build_live_no_shared_obs_visual_current_slice_v1(
                    context,
                    current,
                    global_slot=recipient,
                    incoming_transition_view=incoming,
                )
                carrier = (
                    None
                    if incoming is None
                    else build_live_no_shared_obs_visual_adjacent_slice_v1(
                        incoming,
                        global_slot=recipient,
                    )
                )
                presentation = build_live_no_shared_obs_authorized_presentation_v1(
                    current_slice,
                    carrier,
                    raw_frame,
                    global_context=context,
                    current_global_frame=current,
                    previous_global_frame=(
                        None if incoming is None else incoming.start_frame
                    ),
                    public_catalog=context.static_mechanics_catalog,
                    incoming_visual_events=(
                        None
                        if incoming is None
                        else build_visual_event_batch_v2(incoming)
                    ),
                    researcher_space=researcher_space,
                )
        return PresentationResourceResultV1(
            outcome="response",
            payload=presentation,
        )

    def _recording_no_op(
        self,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        notice: str,
    ) -> ServiceCommandResult:
        """Remember an unavailable recording command and return the unchanged frame."""
        record = _CommandRecord(
            fingerprint=fingerprint,
            shutdown_requested=False,
        )
        self._remember_command(command_key, record)
        return ServiceCommandResult(
            outcome="response",
            payload=CommandResponseV2(
                result="no_op",
                frame=self._frame,
                notice=notice,
            ),
        )

    def _build_replay_handoff(self) -> ReplayViewerService:
        """Create a replay service from verified saved capture and current view
        choices.
        """
        recorder = self._recorder
        if recorder is None or recorder.verified_loaded_bundle is None:
            raise RuntimeError("replay handoff requires a verified recording.")
        reference_global_slot: int | None = None
        selected_global_slot: int | None = None
        armed_lane: Literal[0, 1] | None = None
        if self._view_mode == "researcher":
            reference_global_slot = self._session.controlled_global_slot
            pending = self._session.pending_actions[reference_global_slot]
            selected_global_slot = (
                reference_global_slot
                if pending.selected_global_target_slot is None
                else pending.selected_global_target_slot
            )
            armed_lane = pending.armed_lane
        return ReplayViewerService(
            recorder.verified_loaded_bundle,
            initial_frame_index=0,
            view_mode=self._view_mode,
            reference_global_slot=reference_global_slot,
            selected_global_slot=selected_global_slot,
            armed_lane=armed_lane,
            pov_global_slot=self._session.controlled_global_slot,
            preset=self._preset,
            show_ranges=self._session.show_ranges,
            verbose=False,
        )

    def _prepare_recording_response(
        self,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        recording_status: RecordingStatusV1,
        result: Literal["applied", "no_op", "shutdown_scheduled"],
        notice: str,
        changed: bool,
        shutdown_requested: bool = False,
    ) -> _RecordingResponseCandidate:
        """Validate a proposed recording response and copied command cache before
        mutation.
        """
        candidate_revision = self._revision + int(changed)
        candidate_frame = self._build_frame(
            revision=candidate_revision,
            recording_status=recording_status,
        )
        self._build_authorized_presentation(
            session=self._session,
            raw_frame=candidate_frame,
            view_mode=self._view_mode,
        )
        response = CommandResponseV2(
            result=result,
            frame=candidate_frame,
            notice=notice,
        )
        record = _CommandRecord(
            fingerprint=fingerprint,
            shutdown_requested=shutdown_requested,
        )
        records = self._command_records.copy()
        self._remember_command_in(records, command_key, record)
        return _RecordingResponseCandidate(
            revision=candidate_revision,
            frame=candidate_frame,
            response=response,
            command_records=records,
            shutdown_requested=shutdown_requested,
        )

    def _commit_recording_response(
        self,
        candidate: _RecordingResponseCandidate,
        *,
        replay_handoff: ReplayViewerService | None = None,
    ) -> ServiceCommandResult:
        """Install a prepared response and optional shutdown flag; return any replay
        handoff.
        """
        self._acknowledge_saved_system_prefix()
        self._revision = candidate.revision
        self._frame = candidate.frame
        self._command_records = candidate.command_records
        if candidate.shutdown_requested:
            self._shutting_down = True
        return ServiceCommandResult(
            outcome="response",
            payload=candidate.response,
            shutdown_requested=candidate.shutdown_requested,
            replay_handoff=replay_handoff,
        )

    def _preview_recording_status(
        self,
        *,
        lifecycle: RecordingLifecycleV1,
        close_cause: DebuggerRecordingCloseCauseV1 | None = None,
        completion_reason: str | None = None,
        persistence_error_code: RecordingPersistenceErrorCodeV1 | None = None,
    ) -> RecordingStatusV1:
        """Build proposed lifecycle facts using the current accepted capture count."""
        recorder = self._recorder
        if recorder is None:
            raise AssertionError("recording status preview requires a recorder.")
        return recorder.preview_status_v1(
            captured_transition_count=recorder.validated_transition_count,
            lifecycle=lifecycle,
            close_cause=close_cause,
            completion_reason=completion_reason,
            persistence_error_code=persistence_error_code,
        )

    def _apply_recording_lifecycle_command(
        self,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        command: object,
    ) -> ServiceCommandResult | None:
        """Handle save, review, retry, and exit commands for an attached recorder.

        Return None when ordinary command dispatch should continue. Build response
        candidates before recorder mutations and retain a saved fallback if review
        construction fails.
        """
        recorder = self._recorder
        if recorder is None:
            return None
        if isinstance(command, ConfirmDiscardAndReplaceCommandV1):
            return None
        if isinstance(command, FinishAndReviewCommandV1):
            if recorder.lifecycle != "recording":
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice="Finish & Review is unavailable in the current lifecycle.",
                )
            success, saved_fallback, failures = self._prepare_publication_responses(
                command_key=command_key,
                fingerprint=fingerprint,
                close_cause="finish_and_review",
                success_notice="Replay saved; opening review.",
                failure_notice="Replay save failed; choose Retry Save or Save As.",
            )
            outcome = recorder.finalize_and_save("finish_and_review")
            return self._commit_publication_outcome(
                outcome=outcome,
                success=success,
                saved_fallback=saved_fallback,
                failures=failures,
                begin_review=True,
            )
        if isinstance(command, ReviewReplayCommandV1):
            if recorder.lifecycle != "saved":
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice="Replay review is unavailable until saving succeeds.",
                )
            reviewing = self._prepare_recording_response(
                command_key=command_key,
                fingerprint=fingerprint,
                recording_status=self._preview_recording_status(
                    lifecycle="reviewing",
                    close_cause=recorder.close_cause,
                    completion_reason=recorder.status.completion_reason,
                ),
                result="applied",
                notice="Opening the saved replay.",
                changed=True,
            )
            saved_fallback = self._prepare_recording_response(
                command_key=command_key,
                fingerprint=fingerprint,
                recording_status=recorder.status,
                result="no_op",
                notice=(
                    "The replay remains saved, but review could not open; retry "
                    "Review Replay."
                ),
                changed=False,
            )
            try:
                handoff = self._build_replay_handoff()
                recorder.begin_review()
            except Exception:
                return self._commit_recording_response(saved_fallback)
            return self._commit_recording_response(
                reviewing,
                replay_handoff=handoff,
            )
        if isinstance(command, RetrySaveCommandV1):
            if recorder.lifecycle != "persistence_failed":
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice="Retry is unavailable because no save failure is pending.",
                )
            success, saved_fallback, failures = self._prepare_publication_responses(
                command_key=command_key,
                fingerprint=fingerprint,
                close_cause=recorder.close_cause,
                completion_reason=recorder.status.completion_reason,
                success_notice="Replay saved; opening review.",
                failure_notice="Replay save still failed; try Save As.",
            )
            return self._commit_publication_outcome(
                outcome=recorder.retry_save(),
                success=success,
                saved_fallback=saved_fallback,
                failures=failures,
                begin_review=True,
            )
        if isinstance(command, SaveAsCommandV1):
            if recorder.lifecycle != "persistence_failed":
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice="Save As is unavailable because no save failure is pending.",
                )
            success, saved_fallback, failures = self._prepare_publication_responses(
                command_key=command_key,
                fingerprint=fingerprint,
                close_cause=recorder.close_cause,
                completion_reason=recorder.status.completion_reason,
                success_notice="Replay saved under the new name; opening review.",
                failure_notice="Replay could not be saved under the requested name.",
            )
            return self._commit_publication_outcome(
                outcome=recorder.save_as(command.file_name),
                success=success,
                saved_fallback=saved_fallback,
                failures=failures,
                begin_review=True,
            )
        if isinstance(command, ExitCommandV1):
            if recorder.lifecycle == "recording":
                success, _saved_fallback, failures = (
                    self._prepare_publication_responses(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        close_cause="user_exit",
                        success_notice="Replay saved; debugger shutdown requested.",
                        failure_notice=(
                            "Replay save failed; the debugger remains open for "
                            "recovery."
                        ),
                        success_result="shutdown_scheduled",
                        shutdown_requested=True,
                        success_lifecycle="saved",
                    )
                )
                outcome = recorder.finalize_and_save("user_exit")
            elif recorder.lifecycle == "persistence_failed":
                success, _saved_fallback, failures = (
                    self._prepare_publication_responses(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        close_cause=recorder.close_cause,
                        completion_reason=recorder.status.completion_reason,
                        success_notice="Replay saved; debugger shutdown requested.",
                        failure_notice=(
                            "Replay save failed; the debugger remains open for "
                            "recovery."
                        ),
                        success_result="shutdown_scheduled",
                        shutdown_requested=True,
                        success_lifecycle="saved",
                    )
                )
                outcome = recorder.retry_save()
            elif recorder.lifecycle in ("saved", "reviewing"):
                shutdown = self._prepare_recording_response(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    recording_status=recorder.status,
                    result="shutdown_scheduled",
                    notice="Replay saved; debugger shutdown requested.",
                    changed=False,
                    shutdown_requested=True,
                )
                return self._commit_recording_response(shutdown)
            else:
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice=(
                        "Exit is unavailable while recording closeout is incomplete."
                    ),
                )
            if outcome == "persistence_failed":
                return self._commit_recording_response(
                    self._failure_response_for_current_status(failures)
                )
            return self._commit_recording_response(success)
        return None

    @staticmethod
    def _closed_recording_command_is_allowed(command: object) -> bool:
        """Allow only lifecycle and presentation authority after capture closes."""
        if isinstance(
            command,
            (SetViewCommandV1, SetPresetCommandV1, ConfirmDiscardAndReplaceCommandV1),
        ):
            return True
        if not isinstance(command, KeyboardCommandV1):
            return False
        return (
            normalize_key(command.key, shift_key=command.shift_key)
            in _CLOSED_RECORDING_PRESENTATION_KEYS
        )

    def _prepare_publication_responses(
        self,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        close_cause: DebuggerRecordingCloseCauseV1 | None,
        completion_reason: str | None = None,
        success_notice: str,
        failure_notice: str,
        success_result: Literal["applied", "shutdown_scheduled"] = "applied",
        shutdown_requested: bool = False,
        success_lifecycle: Literal["saved", "reviewing"] = "reviewing",
    ) -> tuple[
        _RecordingResponseCandidate,
        _RecordingResponseCandidate,
        dict[RecordingPersistenceErrorCodeV1, _RecordingResponseCandidate],
    ]:
        """Validate success, review-fallback, and each expected save-error response
        first.
        """
        success = self._prepare_recording_response(
            command_key=command_key,
            fingerprint=fingerprint,
            recording_status=self._preview_recording_status(
                lifecycle=success_lifecycle,
                close_cause=close_cause,
                completion_reason=completion_reason,
            ),
            result=success_result,
            notice=success_notice,
            changed=True,
            shutdown_requested=shutdown_requested,
        )
        saved_fallback = self._prepare_recording_response(
            command_key=command_key,
            fingerprint=fingerprint,
            recording_status=self._preview_recording_status(
                lifecycle="saved",
                close_cause=close_cause,
                completion_reason=completion_reason,
            ),
            result="applied",
            notice=(
                "Replay saved, but review could not open; use Review Replay to retry."
            ),
            changed=True,
        )
        error_codes: tuple[RecordingPersistenceErrorCodeV1, ...] = (
            "target_unavailable",
            "publication_failed",
            "verification_failed",
        )
        failures: dict[
            RecordingPersistenceErrorCodeV1,
            _RecordingResponseCandidate,
        ] = {
            error_code: self._prepare_recording_response(
                command_key=command_key,
                fingerprint=fingerprint,
                recording_status=self._preview_recording_status(
                    lifecycle="persistence_failed",
                    close_cause=close_cause,
                    completion_reason=completion_reason,
                    persistence_error_code=error_code,
                ),
                result="applied",
                notice=failure_notice,
                changed=True,
            )
            for error_code in error_codes
        }
        return success, saved_fallback, failures

    def _failure_response_for_current_status(
        self,
        candidates: dict[
            RecordingPersistenceErrorCodeV1,
            _RecordingResponseCandidate,
        ],
    ) -> _RecordingResponseCandidate:
        """Select the prebuilt response matching the recorder's public save error."""
        recorder = self._recorder
        if recorder is None or recorder.persistence_error_code is None:
            raise AssertionError(
                "persistence failure did not expose a canonical error code."
            )
        return candidates[recorder.persistence_error_code]

    def _commit_publication_outcome(
        self,
        *,
        outcome: Literal["saved", "persistence_failed"],
        success: _RecordingResponseCandidate,
        saved_fallback: _RecordingResponseCandidate,
        failures: dict[
            RecordingPersistenceErrorCodeV1,
            _RecordingResponseCandidate,
        ],
        begin_review: bool,
    ) -> ServiceCommandResult:
        """Install the matching save result, keeping saved status if review cannot
        open.
        """
        if outcome == "persistence_failed":
            return self._commit_recording_response(
                self._failure_response_for_current_status(failures)
            )
        recorder = self._recorder
        if recorder is None:
            raise AssertionError("recording service lost its recorder.")
        if not begin_review:
            return self._commit_recording_response(success)
        try:
            handoff = self._build_replay_handoff()
            recorder.begin_review()
        except Exception:
            return self._commit_recording_response(saved_fallback)
        return self._commit_recording_response(success, replay_handoff=handoff)

    def _finalize_endpoint_after_transition(
        self,
        *,
        candidates: _EndpointResponseCandidates,
    ) -> ServiceCommandResult:
        """Save a committed sealed capture and install its prevalidated endpoint
        response.
        """
        recorder = self._recorder
        if recorder is None or recorder.lifecycle != "sealed":
            raise AssertionError("endpoint finalization requires a sealed recorder.")
        close_cause = recorder.close_cause
        if close_cause not in ("endpoint", "truncation"):
            raise AssertionError("sealed recorder has an invalid close cause.")
        outcome = recorder.finalize_and_save(close_cause)
        if outcome == "persistence_failed":
            return self._install_endpoint_response(
                self._failure_response_for_current_status(candidates.failures)
            )
        return self._install_endpoint_response(candidates.saved)

    def _install_endpoint_response(
        self,
        candidate: _RecordingResponseCandidate,
    ) -> ServiceCommandResult:
        """Install endpoint save status only at the already committed transition
        revision.
        """
        self._acknowledge_saved_system_prefix()
        if candidate.revision != self._revision:
            raise AssertionError("endpoint response revision drifted after commit.")
        self._frame = candidate.frame
        self._command_records = candidate.command_records
        return ServiceCommandResult(
            outcome="response",
            payload=candidate.response,
        )

    def _prepare_endpoint_responses(
        self,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        session: DebuggerSession,
        revision: int,
        view_mode: ViewMode,
        preset: Preset,
        transition_close_status: RecordingStatusV1,
    ) -> _EndpointResponseCandidates:
        """Validate endpoint success and save-failure frames before accepting the
        transition.
        """
        if transition_close_status.lifecycle != "sealed":
            raise ValueError("endpoint response candidates require sealed status.")
        completion_state = transition_close_status.completion_state
        close_cause: DebuggerRecordingCloseCauseV1 = (
            "endpoint" if completion_state == "complete" else "truncation"
        )
        completion_reason = transition_close_status.completion_reason

        def prepare(
            *,
            status: RecordingStatusV1,
            notice: str,
        ) -> _RecordingResponseCandidate:
            """Build one endpoint response from proposed recording status and public
            notice.
            """
            frame = self._build_frame(
                session=session,
                revision=revision,
                view_mode=view_mode,
                preset=preset,
                recording_status=status,
            )
            self._build_authorized_presentation(
                session=session,
                raw_frame=frame,
                view_mode=view_mode,
            )
            response = CommandResponseV2(
                result="applied",
                frame=frame,
                notice=notice,
            )
            records = self._command_records.copy()
            self._remember_command_in(
                records,
                command_key,
                _CommandRecord(
                    fingerprint=fingerprint,
                    shutdown_requested=False,
                ),
            )
            return _RecordingResponseCandidate(
                revision=revision,
                frame=frame,
                response=response,
                command_records=records,
                shutdown_requested=False,
            )

        recorder = self._recorder
        if recorder is None:
            raise AssertionError("endpoint response preview requires a recorder.")
        saved = prepare(
            status=recorder.preview_status_v1(
                captured_transition_count=transition_close_status.captured_transition_count,
                lifecycle="saved",
                close_cause=close_cause,
                completion_reason=completion_reason,
            ),
            notice="The episode ended and its replay was saved.",
        )
        error_codes: tuple[RecordingPersistenceErrorCodeV1, ...] = (
            "target_unavailable",
            "publication_failed",
            "verification_failed",
        )
        failures: dict[
            RecordingPersistenceErrorCodeV1,
            _RecordingResponseCandidate,
        ] = {
            error_code: prepare(
                status=recorder.preview_status_v1(
                    captured_transition_count=(
                        transition_close_status.captured_transition_count
                    ),
                    lifecycle="persistence_failed",
                    close_cause=close_cause,
                    completion_reason=completion_reason,
                    persistence_error_code=error_code,
                ),
                notice=(
                    "The episode ended, but replay saving failed; choose Retry "
                    "Save or Save As."
                ),
            )
            for error_code in error_codes
        }
        return _EndpointResponseCandidates(saved=saved, failures=failures)

    def _prepare_failure_closeout_responses(
        self,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        close_cause: Literal[
            "simulation_failure",
            "policy_failure",
            "capture_failure",
            "validation_failure",
            "processing_failure",
        ],
        session: DebuggerSession | None = None,
        captured_transition_count: int | None = None,
    ) -> _FailureCloseoutCandidates:
        """Validate save outcomes for the accepted capture prefix before failure
        closeout.
        """
        recorder = self._recorder
        if recorder is None:
            raise AssertionError("failure closeout requires a recorder.")
        count = (
            recorder.validated_transition_count
            if captured_transition_count is None
            else captured_transition_count
        )
        resolved_session = self._session if session is None else session
        candidate_revision = self._revision + 1
        completion_reason = (
            "evaluation_processing_failure"
            if close_cause == "processing_failure"
            else close_cause
        )

        def prepare(
            *,
            lifecycle: Literal["saved", "persistence_failed"],
            notice: str,
            persistence_error_code: RecordingPersistenceErrorCodeV1 | None = None,
        ) -> _RecordingResponseCandidate:
            """Build a failed-step closeout response for one proposed publication
            result.
            """
            status = recorder.preview_status_v1(
                captured_transition_count=count,
                lifecycle=lifecycle,
                close_cause=close_cause,
                completion_reason=completion_reason,
                persistence_error_code=persistence_error_code,
            )
            frame = self._build_frame(
                session=resolved_session,
                revision=candidate_revision,
                recording_status=status,
            )
            self._build_authorized_presentation(
                session=resolved_session,
                raw_frame=frame,
                view_mode=self._view_mode,
            )
            response = CommandResponseV2(
                result="applied",
                frame=frame,
                notice=notice,
            )
            records = self._command_records.copy()
            self._remember_command_in(
                records,
                command_key,
                _CommandRecord(
                    fingerprint=fingerprint,
                    shutdown_requested=False,
                ),
            )
            return _RecordingResponseCandidate(
                revision=candidate_revision,
                frame=frame,
                response=response,
                command_records=records,
                shutdown_requested=False,
            )

        saved = prepare(
            lifecycle="saved",
            notice=(
                "The transition failed; the last validated replay prefix was saved."
            ),
        )
        error_codes: tuple[RecordingPersistenceErrorCodeV1, ...] = (
            "target_unavailable",
            "publication_failed",
            "verification_failed",
        )
        failures: dict[
            RecordingPersistenceErrorCodeV1,
            _RecordingResponseCandidate,
        ] = {
            error_code: prepare(
                lifecycle="persistence_failed",
                persistence_error_code=error_code,
                notice=(
                    "The transition failed and replay saving also failed; choose "
                    "Retry Save or Save As."
                ),
            )
            for error_code in error_codes
        }
        return _FailureCloseoutCandidates(saved=saved, failures=failures)

    def _close_failed_recording(
        self,
        *,
        close_cause: Literal[
            "simulation_failure",
            "policy_failure",
            "capture_failure",
            "validation_failure",
            "processing_failure",
        ],
        candidates: _FailureCloseoutCandidates,
    ) -> ServiceCommandResult:
        """Save the accepted prefix, install its result, and fence unexpected
        closeout failures.
        """
        recorder = self._recorder
        if recorder is None:
            raise AssertionError("failure closeout requires a recorder.")
        try:
            outcome = recorder.finalize_and_save(close_cause)
        except Exception:
            self._faulted = True
            self._command_records = candidates.saved.command_records
            raise
        if outcome == "persistence_failed":
            return self._commit_recording_response(
                self._failure_response_for_current_status(candidates.failures)
            )
        return self._commit_recording_response(candidates.saved)

    def apply_command(self, request: CommandRequestV1) -> ServiceCommandResult:
        """Apply one revision-matched command under the service lock.

        Parameters
        ----------
        request : CommandRequestV1
            Validated request containing client ID, command ID, base revision, and
            one supported live or recording command.

        Returns
        -------
        ServiceCommandResult
            Response or typed rejection with the current frame. A saved replay may
            include a prepared replay-service handoff. Shutdown is only requested
            in the result; the HTTP host owns stopping its request loop.

        Notes
        -----
        The latest 256 request IDs are retained. An exact repeat while cached returns
        the current frame without repeating the command; changed content under that
        ID is rejected. Stale revisions, shutdown, and fault state reject new work.
        Step and reset commands may execute JAX work; recording commands may write
        files. Candidate response validation precedes installation. Unexpected
        internal failures can fence further commands and propagate.
        """
        command_key = (request.client_id, request.command_id)
        fingerprint = request.model_dump_json()
        with self._lock:
            previous = self._command_records.get(command_key)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    return ServiceCommandResult(
                        outcome="command_id_conflict",
                        payload=ApiErrorV2(
                            error_code="command_id_conflict",
                            message=(
                                "This client reused a command_id for a "
                                "different request."
                            ),
                            latest_frame=self._frame,
                        ),
                    )
                self._command_records.move_to_end(command_key)
                return ServiceCommandResult(
                    outcome="response",
                    payload=CommandResponseV2(
                        result="duplicate",
                        frame=self._frame,
                        notice="Command already processed; current frame returned.",
                    ),
                    shutdown_requested=previous.shutdown_requested,
                )

            if self._faulted:
                self._remember_command(
                    command_key,
                    _CommandRecord(
                        fingerprint=fingerprint,
                        shutdown_requested=False,
                    ),
                )
                return ServiceCommandResult(
                    outcome="service_faulted",
                    payload=ApiErrorV2(
                        error_code="internal_error",
                        message=(
                            "The debugger entered a safe fault state; restart it "
                            "before sending another command."
                        ),
                        latest_frame=self._frame,
                    ),
                )

            if self._shutting_down:
                self._remember_command(
                    command_key,
                    _CommandRecord(
                        fingerprint=fingerprint,
                        shutdown_requested=False,
                    ),
                )
                return ServiceCommandResult(
                    outcome="server_shutting_down",
                    payload=ApiErrorV2(
                        error_code="server_shutting_down",
                        message=(
                            "The debugger is shutting down; this command was not "
                            "applied."
                        ),
                        latest_frame=self._frame,
                    ),
                )

            if request.base_revision != self._revision:
                self._remember_command(
                    command_key,
                    _CommandRecord(
                        fingerprint=fingerprint,
                        shutdown_requested=False,
                    ),
                )
                return ServiceCommandResult(
                    outcome="stale_revision",
                    payload=ApiErrorV2(
                        error_code="stale_revision",
                        message=(
                            "The debugger advanced after this client frame; "
                            "the latest frame is attached."
                        ),
                        latest_frame=self._frame,
                    ),
                )

            command = request.command
            if isinstance(command, (FinishSystemCommandV1, CancelSystemCommandV1)):
                return self._finish_system_command(command, command_key, fingerprint)
            effective = (
                command.replacement
                if isinstance(command, ConfirmDiscardAndReplaceCommandV1)
                else command
            )
            if isinstance(effective, SetCombatConfigurationCommandV1):
                selected = (effective.team_a_controller, effective.team_b_controller)
                if any(
                    is_system_controller(value)
                    and value.removeprefix("system:") not in self._offered_systems
                    for value in selected
                ):
                    return self._recording_no_op(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        notice="Choose a System declared when DevClient started.",
                    )
            worker = self._system_worker
            operation = None if worker is None else worker.operation
            if operation is not None and not isinstance(
                command, (SetViewCommandV1, SetPresetCommandV1)
            ):
                if isinstance(command, (ExitCommandV1, FinishAndReviewCommandV1)):
                    assert worker is not None
                    worker.cancel(operation)
                else:
                    return self._system_response(
                        command_key,
                        fingerprint,
                        "A System is still working. "
                        "Cancel it before changing the match or drafts.",
                    )
            confirmed_discard = False
            if self._recorder is not None:
                lifecycle_result = self._apply_recording_lifecycle_command(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    command=command,
                )
                if lifecycle_result is not None:
                    return lifecycle_result
                if (
                    self._recorder.lifecycle != "recording"
                    and not self._closed_recording_command_is_allowed(command)
                ):
                    return self._recording_no_op(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        notice=(
                            "Scientific controls are fenced because this recording "
                            "is no longer capturing transitions."
                        ),
                    )
                if isinstance(command, ConfirmDiscardAndReplaceCommandV1):
                    confirmed_discard = True
                    command = command.replacement
                restart_intent = recording_restart_intent_v1(
                    self._session,
                    command,
                    view_mode=self._view_mode,
                    include_stress=self._include_stress,
                )
                if confirmed_discard and (
                    restart_intent is None
                    or self._recorder.lifecycle != "recording"
                    or self._recorder.validated_transition_count == 0
                ):
                    return self._recording_no_op(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        notice=(
                            "Discard confirmation requires a captured prefix and "
                            "an effective episode replacement."
                        ),
                    )
                if (
                    not confirmed_discard
                    and restart_intent is not None
                    and (
                        self._recorder.lifecycle != "recording"
                        or self._recorder.validated_transition_count > 0
                    )
                ):
                    return self._recording_no_op(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        notice=(
                            "Replay recording has captured progress; Finish & Review "
                            "or explicitly discard it before replacing the episode."
                        ),
                    )

            asynchronous = self._start_system_command(
                command, command_key, fingerprint, confirmed_discard
            )
            if asynchronous is not None:
                return asynchronous
            try:
                dispatched = dispatch_command(
                    self._session,
                    command,
                    view_mode=self._view_mode,
                    preset=self._preset,
                    include_stress=self._include_stress,
                )
            except DebuggerTransitionFailureV1 as error:
                recorder = self._recorder
                if recorder is None:
                    self._faulted = True
                    self._remember_command(
                        command_key,
                        _CommandRecord(
                            fingerprint=fingerprint,
                            shutdown_requested=False,
                        ),
                    )
                    raise
                return self._close_transition_failure(error, command_key, fingerprint)
            except Exception:
                self._faulted = True
                self._remember_command(
                    command_key,
                    _CommandRecord(
                        fingerprint=fingerprint,
                        shutdown_requested=False,
                    ),
                )
                raise

            return self._accept_dispatch_result(
                dispatched,
                command_key=command_key,
                fingerprint=fingerprint,
                confirmed_discard=confirmed_discard,
            )

    def _close_transition_failure(
        self,
        error: DebuggerTransitionFailureV1,
        command_key: tuple[str, str],
        fingerprint: str,
    ) -> ServiceCommandResult:
        """Use the recorder's existing failure cause and accepted-prefix closeout."""
        failure_causes: dict[
            DebuggerTransitionFailureStageV1,
            Literal[
                "simulation_failure",
                "policy_failure",
                "capture_failure",
                "validation_failure",
            ],
        ] = {
            "action_build": "policy_failure",
            "simulation": "simulation_failure",
            "capture": "capture_failure",
            "validation": "validation_failure",
        }
        close_cause: Literal[
            "simulation_failure",
            "policy_failure",
            "capture_failure",
            "validation_failure",
        ] = failure_causes[error.stage]
        candidates = self._prepare_failure_closeout_responses(
            command_key=command_key,
            fingerprint=fingerprint,
            close_cause=close_cause,
        )
        return self._close_failed_recording(
            close_cause=close_cause,
            candidates=candidates,
        )

    @property
    def system_evidence_directory(self) -> Path | None:
        """Return the current match's call folder for host tools, never the browser."""
        installation = self._system_evidence
        return None if installation is None else installation.directory

    def close(self) -> None:
        """Fence pending results and queue owned cleanup without stopping a server.

        Arbitrary Python factories can delay thread exit. The worker retains
        cleanup futures so host tools can inspect failures; supplied Systems are
        caller-owned and are not closed here.
        """
        with self._lock:
            if self._system_worker is not None:
                self._system_worker.close()

    def _system_response(
        self, key: tuple[str, str], fingerprint: str, notice: str
    ) -> ServiceCommandResult:
        """Publish current operation status without changing the accepted game."""
        if self._frame.system_operation != self._system_operation_status():
            self._revision += 1
            self._frame = self._build_frame()
        return self._recording_no_op(
            command_key=key, fingerprint=fingerprint, notice=notice
        )

    def _start_system_command(
        self,
        command: object,
        key: tuple[str, str],
        fingerprint: str,
        confirmed_discard: bool,
    ) -> ServiceCommandResult | None:
        """Submit selected setup or one joint turn; leave ordinary dispatch unchanged.

        Called after recording replacement checks. The captured session and draft
        belong to this operation until an explicit completion command accepts it.
        No model call, factory or initialization runs under the service lock.
        """
        session = self._session
        restart = recording_restart_intent_v1(
            session,
            cast("DebuggerCommandV1", command),
            view_mode=self._view_mode,
            include_stress=self._include_stress,
        )
        controllers = (session.team_a_controller, session.team_b_controller)
        mode = session.evaluation_context.execution_information_mode
        if isinstance(command, SetCombatConfigurationCommandV1):
            controllers = (command.team_a_controller, command.team_b_controller)
            mode = command.execution_information_mode
        selected = any(is_system_controller(value) for value in controllers)
        worker = self._system_worker
        if restart is not None and (
            selected or (worker is not None and worker.active is not None)
        ):
            if worker is None:
                worker = self._system_worker = SystemWorker()
            if worker.operation is not None:
                return self._system_response(
                    key,
                    fingerprint,
                    "Wait for System cleanup before loading a replacement.",
                )
            self._system_failed = False
            self._system_confirmed_discard = confirmed_discard
            directory = None
            if self._recorder is not None:
                directory = (
                    self._recorder.evidence_directory
                    / f"generation-{session.run_generation + 1}"
                )
            worker.prepare(
                session,
                self._offered_systems,
                controllers,
                lambda pair: set_combat_configuration(
                    session,
                    team_a_controller=controllers[0],
                    team_b_controller=controllers[1],
                    execution_information_mode=mode,
                    systems=pair,
                ),
                directory=directory,
            )
            return self._system_response(
                key, fingerprint, "Loading the selected Systems."
            )
        submission = (
            isinstance(command, KeyboardCommandV1)
            and not (
                command.ctrl_key
                or command.alt_key
                or command.meta_key
                or command.repeat
            )
            and normalize_key(command.key, shift_key=command.shift_key)
            in ("space", "enter")
            and session.scenario.mode == "interactive"
        )
        if submission and selected and worker is not None:
            if worker.active is None or worker.active.closed or self._system_failed:
                return self._system_response(
                    key,
                    fingerprint,
                    "Load a fresh System before playing again. "
                    "The accepted replay can still be saved.",
                )
            view, preset = self._view_mode, self._preset
            worker.decide(
                session,
                lambda: dispatch_command(
                    session,
                    cast("DebuggerCommandV1", command),
                    view_mode=view,
                    preset=preset,
                    include_stress=self._include_stress,
                ),
            )
            return self._system_response(
                key, fingerprint, "The Systems are choosing this turn."
            )
        return None

    def _finish_system_command(
        self,
        command: FinishSystemCommandV1 | CancelSystemCommandV1,
        key: tuple[str, str],
        fingerprint: str,
    ) -> ServiceCommandResult:
        """Poll, cancel or accept one operation without waiting on unfinished work."""
        worker = self._system_worker
        operation = None if worker is None else worker.operation
        if (
            worker is None
            or operation is None
            or operation.identifier != command.operation_id
        ):
            return self._system_response(
                key, fingerprint, "This System operation is no longer pending."
            )
        if isinstance(command, CancelSystemCommandV1) and not operation.adopted:
            worker.cancel(operation)
        try:
            if operation.adopted:
                complete = worker.finish_accept(operation)
                return self._system_response(
                    key,
                    fingerprint,
                    "System turn accepted."
                    if complete
                    else "Finishing System records and cleanup.",
                )
            if operation.cancelled:
                complete = worker.finish_cancel(operation)
                return self._system_response(
                    key,
                    fingerprint,
                    "System work failed. Load fresh methods before playing again."
                    if complete and self._system_failed
                    else "System work cancelled. The accepted game is unchanged."
                    if complete
                    else "Waiting for cancelled System work to finish cleanup.",
                )
            if not operation.future.done():
                return self._system_response(
                    key, fingerprint, "The System is still working."
                )
            prepared = operation.future.result()
            base, current = operation.base, self._session
            if (
                base.raw_continuation_identity is not current.raw_continuation_identity
                or base.pending_actions != current.pending_actions
                or base.evaluation_context is not current.evaluation_context
                or base.run_generation != current.run_generation
                or base.current_evaluation_frame is not current.current_evaluation_frame
            ):
                worker.cancel(operation)
                return self._system_response(
                    key,
                    fingerprint,
                    "The match changed; this System result was abandoned.",
                )
            if isinstance(prepared, SystemInstallation):
                assert prepared.session is not None
                candidate = prepared.session
                dispatched = InputDispatchResult(
                    candidate,
                    self._view_mode,
                    self._preset,
                    True,
                    True,
                    episode_restarted=True,
                    raw_continuation_identity=candidate.raw_continuation_identity,
                )
            else:
                dispatched = prepared
            candidate = replace(
                dispatched.session,
                controlled_global_slot=current.controlled_global_slot
                if dispatched.session.scenario is current.scenario
                else dispatched.session.controlled_global_slot,
                show_ranges=current.show_ranges,
                verbose_logging=current.verbose_logging,
            )
            dispatched = replace(
                dispatched,
                session=candidate,
                view_mode=self._view_mode,
                preset=self._preset,
            )
            result = self._accept_dispatch_result(
                dispatched,
                command_key=key,
                fingerprint=fingerprint,
                confirmed_discard=self._system_confirmed_discard
                if operation.kind == "setup"
                else False,
                system_operation=operation,
            )
            if not operation.adopted:
                worker.cancel(operation)
            self._frame = self._build_frame()
            if isinstance(result.payload, CommandResponseV2):
                result = replace(
                    result,
                    payload=result.payload.model_copy(update={"frame": self._frame}),
                )
            return result
        except DebuggerTransitionFailureV1 as error:
            logging.getLogger(__name__).exception("DevClient System decision failed")
            worker.cancel(operation)
            self._system_failed = True
            if self._recorder is not None:
                return self._close_transition_failure(error, key, fingerprint)
            return self._system_response(
                key,
                fingerprint,
                "System decision failed. Load fresh methods before playing again.",
            )
        except Exception:
            logging.getLogger(__name__).exception("DevClient System operation failed")
            if not operation.adopted:
                worker.cancel(operation)
            self._system_failed = operation.kind == "decision" or operation.adopted
            return self._system_response(
                key,
                fingerprint,
                "System work failed. The accepted game and replay remain available; "
                "check the host error log.",
            )

    def _acknowledge_saved_system_prefix(self) -> None:
        """Queue the recorder's verified save for its exact call-evidence owner."""
        recorder, worker, installation = (
            self._recorder,
            self._system_worker,
            self._system_evidence,
        )
        if recorder is None or worker is None or installation is None:
            return
        saved = recorder.saved_bundle
        if saved is not None:
            assert recorder.verified_loaded_bundle is not None
            worker.record_saved_prefix(
                installation=installation,
                replay_id=recorder.verified_loaded_bundle.replay.artifact_id,
                path=saved.replay_path,
                transitions=recorder.validated_transition_count,
            )

    def _adopt_system_operation(
        self, operation: PendingSystemOperation | None, session: DebuggerSession
    ) -> None:
        """Acknowledge each real adoption before later recording work can fail."""
        if operation is None:
            return
        worker = self._system_worker
        assert worker is not None
        self._system_evidence = (
            operation.installation if session.systems is not None else None
        )
        worker.accept(operation, session)
        if operation.on_installed is not None:
            operation.on_installed()

    def _accept_dispatch_result(
        self,
        dispatched: InputDispatchResult,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        confirmed_discard: bool = False,
        system_operation: PendingSystemOperation | None = None,
    ) -> ServiceCommandResult:
        """Validate and adopt one prepared result under the existing service lock.

        Both ordinary dispatch and later worker completion use this one path.
        command_key/fingerprint identify the request to remember. confirmed_discard
        permits the already approved replacement of an accepted recording prefix.
        Validate frames before installation; recorder acceptance remains the owner
        of saved progress, including processing errors after append succeeds.
        """
        candidate_session = dispatched.session
        candidate_view_mode = dispatched.view_mode
        candidate_preset = dispatched.preset
        candidate_revision = self._revision + int(dispatched.changed)
        previous_scientific_state = (
            self._session.scenario_name,
            self._session.seed,
            self._session.run_generation,
            self._session.scenario_default_movement_scale,
            self._session.evaluation_context,
            self._session.current_evaluation_frame,
            self._session.incoming_evaluation_view,
            self._session.status_source_evidence_state,
            self._session.last_submission_kind,
            self._session.last_report_actor_slots,
            self._session.next_script_frame_index,
        )
        candidate_scientific_state = (
            candidate_session.scenario_name,
            candidate_session.seed,
            candidate_session.run_generation,
            candidate_session.scenario_default_movement_scale,
            candidate_session.evaluation_context,
            candidate_session.current_evaluation_frame,
            candidate_session.incoming_evaluation_view,
            candidate_session.status_source_evidence_state,
            candidate_session.last_submission_kind,
            candidate_session.last_report_actor_slots,
            candidate_session.next_script_frame_index,
        )
        previous_continuation = (
            self._session.config,
            self._session.key,
            self._session.state,
            self._session.observation,
            self._session.action_mask,
            self._session.raw_continuation_identity,
        )
        candidate_continuation = (
            candidate_session.config,
            candidate_session.key,
            candidate_session.state,
            candidate_session.observation,
            candidate_session.action_mask,
            candidate_session.raw_continuation_identity,
        )
        previous_system_parts = (
            self._session.systems,
            self._session.system_ids,
            self._session.environment_state,
            self._session.system_memory,
        )
        candidate_system_parts = (
            candidate_session.systems,
            candidate_session.system_ids,
            candidate_session.environment_state,
            candidate_session.system_memory,
        )
        if dispatched.transition_applied is not None:
            if (
                candidate_session.systems is not self._session.systems
                or candidate_session.system_ids is not self._session.system_ids
            ):
                self._faulted = True
                raise RuntimeError("A transition cannot replace installed Systems")
            if (
                self._session.environment_state is not None
                and candidate_session.environment_state
                is self._session.environment_state
            ):
                self._faulted = True
                raise RuntimeError("A System transition needs its successor wrapper")
            transition_view = dispatched.transition_applied
            scenario = self._session.scenario
            expected_submission_kind: Literal["interactive", "scripted"] = (
                "scripted" if scenario.mode == "scripted" else "interactive"
            )
            expected_script_cursor = (
                self._session.next_script_frame_index + 1
                if scenario.mode == "scripted"
                else self._session.next_script_frame_index
            )
            expected_report_slots = (
                tuple(
                    sorted(
                        command.actor_global_slot
                        for command in scenario.frames[
                            self._session.next_script_frame_index
                        ].commands
                    )
                )
                if scenario.mode == "scripted"
                else tuple(
                    row.global_slot
                    for row in self._session.evaluation_context.roster
                    if row.configured_active
                )
            )
            if (
                transition_view.context != self._session.evaluation_context
                or transition_view.start_frame != self._session.current_evaluation_frame
                or transition_view.successor_frame
                != candidate_session.current_evaluation_frame
                or candidate_session.incoming_evaluation_view != transition_view
                or candidate_session.run_generation != self._session.run_generation
                or candidate_session.scenario_name != self._session.scenario_name
                or candidate_session.seed != self._session.seed
                or candidate_session.scenario_default_movement_scale
                != self._session.scenario_default_movement_scale
                or candidate_session.last_submission_kind != expected_submission_kind
                or candidate_session.last_report_actor_slots != expected_report_slots
                or candidate_session.next_script_frame_index != expected_script_cursor
                or dispatched.raw_continuation_identity
                is not candidate_session.raw_continuation_identity
                or candidate_session.config is not self._session.config
                or any(
                    candidate is previous
                    for candidate, previous in zip(
                        candidate_continuation[1:],
                        previous_continuation[1:],
                        strict=True,
                    )
                )
            ):
                self._faulted = True
                raise RuntimeError(
                    "transition marker does not describe exactly one session advance"
                )
        elif dispatched.episode_restarted:
            if (
                candidate_session.run_generation != self._session.run_generation + 1
                or candidate_session.current_evaluation_frame.frame_index != 0
                or candidate_session.incoming_evaluation_view is not None
                or candidate_session.evaluation_context.identity.episode_id
                == self._session.evaluation_context.identity.episode_id
                or candidate_session.seed != self._session.seed
                or candidate_session.last_submission_kind is not None
                or candidate_session.last_report_actor_slots
                or candidate_session.next_script_frame_index != 0
                or dispatched.raw_continuation_identity
                is not candidate_session.raw_continuation_identity
                or any(
                    candidate is previous
                    for candidate, previous in zip(
                        candidate_continuation[1:],
                        previous_continuation[1:],
                        strict=True,
                    )
                )
            ):
                self._faulted = True
                raise RuntimeError(
                    "restart marker does not describe a fresh episode generation"
                )
        elif (
            any(
                candidate is not previous
                for candidate, previous in zip(
                    candidate_system_parts, previous_system_parts, strict=True
                )
            )
            or candidate_scientific_state != previous_scientific_state
            or any(
                candidate is not previous
                for candidate, previous in zip(
                    candidate_continuation,
                    previous_continuation,
                    strict=True,
                )
            )
        ):
            self._faulted = True
            raise RuntimeError(
                "scientific session state changed without transition or restart marker"
            )
        candidate_recorder = self._recorder
        endpoint_candidates: _EndpointResponseCandidates | None = None
        validation_failure_candidates: _FailureCloseoutCandidates | None = None
        processing_failure_candidates: _FailureCloseoutCandidates | None = None
        candidate_recording_status = (
            None if self._recorder is None else self._recorder.status
        )
        candidate_frame = self._frame
        if dispatched.changed:
            try:
                if self._recorder is not None:
                    if dispatched.transition_applied is not None:
                        candidate_recording_status = (
                            self._recorder.preview_status_after_append_v1(
                                dispatched.transition_applied.transition
                            )
                        )
                        validation_failure_candidates = (
                            self._prepare_failure_closeout_responses(
                                command_key=command_key,
                                fingerprint=fingerprint,
                                close_cause="validation_failure",
                            )
                        )
                        processing_failure_candidates = (
                            self._prepare_failure_closeout_responses(
                                command_key=command_key,
                                fingerprint=fingerprint,
                                close_cause="processing_failure",
                                session=candidate_session,
                                captured_transition_count=(
                                    self._recorder.validated_transition_count + 1
                                ),
                            )
                        )
                    elif dispatched.episode_restarted:
                        candidate_recorder = self._recorder.replacement_for(
                            candidate_session.evaluation_context,
                            candidate_session.current_evaluation_frame,
                        )
                        candidate_recording_status = candidate_recorder.status
                candidate_frame = self._build_frame(
                    session=candidate_session,
                    revision=candidate_revision,
                    view_mode=candidate_view_mode,
                    preset=candidate_preset,
                    recording_status=candidate_recording_status,
                )
                self._build_authorized_presentation(
                    session=candidate_session,
                    raw_frame=candidate_frame,
                    view_mode=candidate_view_mode,
                )
                if (
                    self._recorder is not None
                    and dispatched.transition_applied is not None
                    and candidate_recording_status is not None
                    and candidate_recording_status.lifecycle == "sealed"
                ):
                    endpoint_candidates = self._prepare_endpoint_responses(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        session=candidate_session,
                        revision=candidate_revision,
                        view_mode=candidate_view_mode,
                        preset=candidate_preset,
                        transition_close_status=candidate_recording_status,
                    )
            except Exception:
                self._faulted = True
                self._remember_command(
                    command_key,
                    _CommandRecord(
                        fingerprint=fingerprint,
                        shutdown_requested=False,
                    ),
                )
                raise

        result_kind: Literal[
            "applied",
            "no_op",
            "shutdown_scheduled",
        ] = (
            "shutdown_scheduled"
            if dispatched.shutdown_requested
            else "applied"
            if dispatched.changed
            else "no_op"
        )
        try:
            candidate_response = CommandResponseV2(
                result=result_kind,
                frame=candidate_frame,
                notice=dispatched.notice,
            )
            candidate_record = _CommandRecord(
                fingerprint=fingerprint,
                shutdown_requested=dispatched.shutdown_requested,
            )
            candidate_result = ServiceCommandResult(
                outcome="response",
                payload=candidate_response,
                shutdown_requested=dispatched.shutdown_requested,
            )
            candidate_command_records = self._command_records.copy()
            self._remember_command_in(
                candidate_command_records,
                command_key,
                candidate_record,
            )
        except Exception:
            self._faulted = True
            self._remember_command(
                command_key,
                _CommandRecord(
                    fingerprint=fingerprint,
                    shutdown_requested=False,
                ),
            )
            raise

        if dispatched.transition_applied is not None:
            transition_view = dispatched.transition_applied
            if (
                candidate_session.current_evaluation_frame
                != transition_view.successor_frame
                or candidate_session.evaluation_context != transition_view.context
                or transition_view.context != self._session.evaluation_context
                or transition_view.start_frame != self._session.current_evaluation_frame
                or self._validated_transition_count()
                != transition_view.transition.transition_index
            ):
                self._faulted = True
                self._remember_command(
                    command_key,
                    _CommandRecord(
                        fingerprint=fingerprint,
                        shutdown_requested=False,
                    ),
                )
                raise RuntimeError(
                    "candidate transition and committed observer epoch diverged"
                )
            try:
                if self._recorder is not None:
                    self._recorder.append(
                        transition_view.transition,
                        candidate_session.current_evaluation_frame,
                    )
            except Exception as error:
                recorder = self._recorder
                if recorder is None:
                    self._faulted = True
                    self._remember_command(
                        command_key,
                        _CommandRecord(
                            fingerprint=fingerprint,
                            shutdown_requested=False,
                        ),
                    )
                    raise
                previous_count = transition_view.transition.transition_index
                if recorder.validated_transition_count == previous_count:
                    if validation_failure_candidates is None:
                        raise AssertionError(
                            "validation failure response was not prebuilt."
                        ) from error
                    return self._close_failed_recording(
                        close_cause="validation_failure",
                        candidates=validation_failure_candidates,
                    )
                if recorder.validated_transition_count != previous_count + 1:
                    self._faulted = True
                    raise RuntimeError(
                        "recording append failed with incoherent validated progress"
                    ) from error
                self._session = candidate_session
                self._adopt_system_operation(system_operation, candidate_session)
                self._view_mode = candidate_view_mode
                self._preset = candidate_preset
                self._revision = candidate_revision
                self._frame = candidate_frame
                self._command_records = candidate_command_records
                if recorder.lifecycle == "sealed":
                    if endpoint_candidates is None:
                        raise AssertionError(
                            "sealed processing failure lacks endpoint candidates."
                        ) from error
                    return self._finalize_endpoint_after_transition(
                        candidates=endpoint_candidates,
                    )
                if processing_failure_candidates is None:
                    raise AssertionError(
                        "processing failure response was not prebuilt."
                    ) from error
                return self._close_failed_recording(
                    close_cause="processing_failure",
                    candidates=processing_failure_candidates,
                )

        if (
            confirmed_discard
            and dispatched.episode_restarted
            and self._recorder is not None
        ):
            self._recorder.discard()

        self._session = candidate_session
        self._adopt_system_operation(system_operation, candidate_session)
        self._view_mode = candidate_view_mode
        self._preset = candidate_preset
        if dispatched.shutdown_requested:
            self._shutting_down = True
        if dispatched.changed:
            self._revision = candidate_revision
            self._frame = candidate_frame
        self._recorder = candidate_recorder
        self._command_records = candidate_command_records
        if endpoint_candidates is not None:
            return self._finalize_endpoint_after_transition(
                candidates=endpoint_candidates,
            )
        return candidate_result

    def _system_operation_status(self) -> SystemOperationV1 | None:
        """Read bounded worker state without polling or changing the game."""
        worker = self._system_worker
        operation = None if worker is None else worker.operation
        if operation is None:
            return None
        finishing = (
            operation.acceptance_future
            if operation.adopted
            else operation.cleanup_future
        )
        cleanup_failed = (
            finishing is not None
            and finishing.done()
            and finishing.exception() is not None
        )
        return SystemOperationV1(
            operation_id=operation.identifier,
            state="failed"
            if cleanup_failed
            else "finishing"
            if operation.adopted
            else "cancelling"
            if operation.cancelled
            else "loading"
            if operation.kind == "setup"
            else "thinking",
        )

    def _build_frame(
        self,
        *,
        session: DebuggerSession | None = None,
        revision: int | None = None,
        view_mode: ViewMode | None = None,
        preset: Preset | None = None,
        recording_status: RecordingStatusV1 | None = None,
    ) -> LiveDebuggerFrame:
        """Build a candidate frame, using committed values wherever overrides are
        None.
        """
        resolved_recording_status = recording_status
        if resolved_recording_status is None and self._recorder is not None:
            resolved_recording_status = self._recorder.status
        frame = build_debugger_frame(
            self._session if session is None else session,
            session_id=self._session_id,
            revision=self._revision if revision is None else revision,
            view_mode=self._view_mode if view_mode is None else view_mode,
            preset=self._preset if preset is None else preset,
            include_stress=self._include_stress,
            recording_status=resolved_recording_status,
        )
        status = self._system_operation_status()
        return frame.model_copy(
            update={"system_choices": self._system_choices, "system_operation": status}
        )

    def _remember_command(
        self,
        key: tuple[str, str],
        record: _CommandRecord,
    ) -> None:
        """Insert a request record into the current bounded recent-command cache."""
        self._remember_command_in(self._command_records, key, record)

    @staticmethod
    def _remember_command_in(
        records: OrderedDict[tuple[str, str], _CommandRecord],
        key: tuple[str, str],
        record: _CommandRecord,
    ) -> None:
        """Prepare a bounded cache mutation before scientific state commits."""
        records[key] = record
        records.move_to_end(key)
        while len(records) > _COMMAND_RECORD_LIMIT:
            records.popitem(last=False)
