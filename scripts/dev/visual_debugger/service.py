"""Locked authoritative session ownership for the live browser debugger."""

from collections import OrderedDict
from dataclasses import dataclass
from secrets import token_urlsafe
from threading import RLock
from typing import Literal, cast

from marl_battlegrounds.evaluation.metrics import EvaluationEpisodeObserverV1
from marl_battlegrounds.evaluation.pov import (
    build_actor_pov_adjacent_transition_slice_v1,
    build_actor_pov_current_slice_v1,
)
from scripts.dev.visual_debugger.control import (
    DebuggerTransitionFailureStageV1,
    DebuggerTransitionFailureV1,
)
from scripts.dev.visual_debugger.frame import LiveDebuggerFrame, build_debugger_frame
from scripts.dev.visual_debugger.input import (
    dispatch_command,
    normalize_key,
    recording_restart_intent_v1,
    sanitize_pov_pending_target,
)
from scripts.dev.visual_debugger.live_presentation import (
    build_live_no_shared_obs_authorized_presentation_v1,
    build_live_oracle_authorized_presentation_v1,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.presentation_protocol import (
    PresentationResourceResultV1,
)
from scripts.dev.visual_debugger.protocol import (
    ActorPovLiveDebuggerFrameV2,
    ApiErrorV2,
    CommandRequestV1,
    CommandResponseV2,
    ConfirmDiscardAndReplaceCommandV1,
    ExitCommandV1,
    FinishAndReviewCommandV1,
    KeyboardCommandV1,
    Preset,
    RecordingLifecycleV1,
    RecordingPersistenceErrorCodeV1,
    RecordingStatusV1,
    ResearcherLiveDebuggerFrameV2,
    RetrySaveCommandV1,
    ReviewReplayCommandV1,
    SaveAsCommandV1,
    SetPresetCommandV1,
    SetViewCommandV1,
    ViewMode,
)
from scripts.dev.visual_debugger.recording import (
    DebuggerRecordingCloseCauseV1,
    DebuggerReplayRecorderV1,
)
from scripts.dev.visual_debugger.replay_service import ReplayViewerService
from scripts.dev.visual_debugger.scenarios import STRESS_SCENARIOS, get_scenario

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
        if type(self.saved) is not bool:
            raise TypeError("recording close saved flag must be a Python bool.")
        if type(self.message) is not str or not self.message:
            raise ValueError("recording close message must be nonempty.")


@dataclass(frozen=True, slots=True)
class _CommandRecord:
    fingerprint: str
    shutdown_requested: bool


@dataclass(frozen=True, slots=True)
class _RecordingResponseCandidate:
    revision: int
    frame: LiveDebuggerFrame
    response: CommandResponseV2
    command_records: OrderedDict[tuple[str, str], _CommandRecord]
    shutdown_requested: bool


@dataclass(frozen=True, slots=True)
class _EndpointResponseCandidates:
    saved: _RecordingResponseCandidate
    failures: dict[
        RecordingPersistenceErrorCodeV1,
        _RecordingResponseCandidate,
    ]


@dataclass(frozen=True, slots=True)
class _FailureCloseoutCandidates:
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
        recorder: DebuggerReplayRecorderV1 | None = None,
    ) -> None:
        if session.scenario_name in STRESS_SCENARIOS and not include_stress:
            msg = (
                f"stress scenario {session.scenario_name!r} requires "
                "include_stress=True."
            )
            raise ValueError(msg)
        if preset not in ("presentation", "analysis", "technical", "debug"):
            raise ValueError("unknown debugger preset")
        self._session = (
            sanitize_pov_pending_target(session) if view_mode == "pov" else session
        )
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
        self._recorder: DebuggerReplayRecorderV1 | None = self._validated_recorder(
            self._session,
            recorder,
        )
        self._evaluation_observer: EvaluationEpisodeObserverV1 | None = (
            self._new_evaluation_observer(self._session)
            if self._recorder is None
            else None
        )
        self._frame = self._build_frame()

    @staticmethod
    def _validated_recorder(
        session: DebuggerSession,
        recorder: DebuggerReplayRecorderV1 | None,
    ) -> DebuggerReplayRecorderV1 | None:
        if recorder is None:
            if session.evaluation_context.capture_profile != "debug":
                raise ValueError(
                    "unrecorded live debugger sessions require the debug profile."
                )
            return None
        raw_recorder = cast(object, recorder)
        if type(raw_recorder) is not DebuggerReplayRecorderV1:
            raise TypeError("recorder must be exact DebuggerReplayRecorderV1.")
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

    @staticmethod
    def _new_evaluation_observer(
        session: DebuggerSession,
    ) -> EvaluationEpisodeObserverV1:
        """Start an unretaining zero-reducer observer for one session epoch."""
        if session.evaluation_context.capture_profile != "debug":
            raise ValueError("live debugger observers require the debug profile.")
        observer = EvaluationEpisodeObserverV1(
            session.evaluation_context,
            reducers=(),
        )
        observer.start(session.current_evaluation_frame)
        if (
            observer.retained_frames is not None
            or observer.retained_transitions is not None
        ):
            raise AssertionError("debug observers must not retain trajectory history.")
        return observer

    @property
    def session(self) -> DebuggerSession:
        """Return the current immutable session for diagnostics and tests."""
        with self._lock:
            return self._session

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
        """Return immutable observer progress without exposing its mutator."""
        with self._lock:
            if self._recorder is not None:
                return self._recorder.validated_transition_count
            observer = self._evaluation_observer
            if observer is None:
                raise AssertionError("unrecorded service is missing its observer.")
            return observer.validated_transition_count

    @property
    def evaluation_observer_lifecycle_state(self) -> str:
        """Return the service-owned observer lifecycle label."""
        with self._lock:
            if self._recorder is not None:
                return self._recorder.observer_lifecycle_state
            observer = self._evaluation_observer
            if observer is None:
                raise AssertionError("unrecorded service is missing its observer.")
            return observer.lifecycle_state

    @property
    def recording_status(self) -> RecordingStatusV1 | None:
        """Return the path-free recorder status without exposing its mutator."""
        with self._lock:
            return None if self._recorder is None else self._recorder.status

    def close_recording_for_keyboard_interrupt(self) -> RecordingCloseResult:
        """Best-effort durable closeout for the hosting process's Ctrl-C path."""
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
        if self._recorder is not None:
            return self._recorder.validated_transition_count
        observer = self._evaluation_observer
        if observer is None:
            raise AssertionError("unrecorded service is missing its observer.")
        return observer.validated_transition_count

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
        """Return the current coherent frame without mutating the session."""
        with self._lock:
            return self._frame

    def current_presentation(self) -> PresentationResourceResultV1:
        """Build one authorized presentation from the committed live snapshot."""
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
            session = self._session
            context = session.evaluation_context
            current = session.current_evaluation_frame
            incoming = session.incoming_evaluation_view
            if self._view_mode == "researcher":
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
                if type(raw_frame) is not ActorPovLiveDebuggerFrameV2:
                    raise RuntimeError(
                        "POV live service lacks its committed recipient frame."
                    )
                recipient = session.controlled_global_slot
                current_slice = build_actor_pov_current_slice_v1(
                    context,
                    current,
                    global_slot=recipient,
                    incoming_transition_view=incoming,
                )
                carrier = (
                    None
                    if incoming is None
                    else build_actor_pov_adjacent_transition_slice_v1(
                        incoming,
                        global_slot=recipient,
                    )
                )
                presentation = build_live_no_shared_obs_authorized_presentation_v1(
                    current_slice,
                    carrier,
                    raw_frame,
                    public_catalog=context.static_mechanics_catalog,
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
        candidate_revision = self._revision + int(changed)
        candidate_frame = self._build_frame(
            revision=candidate_revision,
            recording_status=recording_status,
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
            frame = self._build_frame(
                session=session,
                revision=revision,
                view_mode=view_mode,
                preset=preset,
                recording_status=status,
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
        """Apply at most one command under duplicate and revision guards."""
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
            if dispatched.transition_applied is not None:
                transition_view = dispatched.transition_applied
                scenario = get_scenario(self._session.scenario_name)
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
                    else (self._session.controlled_global_slot,)
                    if self._view_mode == "pov"
                    else tuple(
                        row.global_slot
                        for row in self._session.evaluation_context.roster
                        if row.configured_active
                    )
                )
                if (
                    transition_view.context != self._session.evaluation_context
                    or transition_view.start_frame
                    != self._session.current_evaluation_frame
                    or transition_view.successor_frame
                    != candidate_session.current_evaluation_frame
                    or candidate_session.incoming_evaluation_view != transition_view
                    or candidate_session.run_generation != self._session.run_generation
                    or candidate_session.scenario_name != self._session.scenario_name
                    or candidate_session.seed != self._session.seed
                    or candidate_session.scenario_default_movement_scale
                    != self._session.scenario_default_movement_scale
                    or candidate_session.last_submission_kind
                    != expected_submission_kind
                    or candidate_session.last_report_actor_slots
                    != expected_report_slots
                    or candidate_session.next_script_frame_index
                    != expected_script_cursor
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
                        "transition marker does not describe exactly one "
                        "session advance"
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
            elif candidate_scientific_state != previous_scientific_state or any(
                candidate is not previous
                for candidate, previous in zip(
                    candidate_continuation,
                    previous_continuation,
                    strict=True,
                )
            ):
                self._faulted = True
                raise RuntimeError(
                    "scientific session state changed without transition or "
                    "restart marker"
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

            candidate_observer = self._evaluation_observer
            if dispatched.transition_applied is not None:
                transition_view = dispatched.transition_applied
                if (
                    candidate_session.current_evaluation_frame
                    != transition_view.successor_frame
                    or candidate_session.evaluation_context != transition_view.context
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
                    if self._recorder is None:
                        if self._evaluation_observer is None:
                            raise AssertionError(
                                "unrecorded service is missing its observer."
                            )
                        self._evaluation_observer.append(
                            transition_view.transition,
                            transition_view.successor_frame,
                        )
                    else:
                        self._recorder.append(
                            transition_view.transition,
                            transition_view.successor_frame,
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
            elif dispatched.episode_restarted and self._recorder is None:
                try:
                    candidate_observer = self._new_evaluation_observer(
                        candidate_session
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

            if (
                confirmed_discard
                and dispatched.episode_restarted
                and self._recorder is not None
            ):
                self._recorder.discard()

            self._session = candidate_session
            self._view_mode = candidate_view_mode
            self._preset = candidate_preset
            if dispatched.shutdown_requested:
                self._shutting_down = True
            if dispatched.changed:
                self._revision = candidate_revision
                self._frame = candidate_frame
            self._evaluation_observer = candidate_observer
            self._recorder = candidate_recorder
            self._command_records = candidate_command_records
            if endpoint_candidates is not None:
                return self._finalize_endpoint_after_transition(
                    candidates=endpoint_candidates,
                )
            return candidate_result

    def _build_frame(
        self,
        *,
        session: DebuggerSession | None = None,
        revision: int | None = None,
        view_mode: ViewMode | None = None,
        preset: Preset | None = None,
        recording_status: RecordingStatusV1 | None = None,
    ) -> LiveDebuggerFrame:
        resolved_recording_status = recording_status
        if resolved_recording_status is None and self._recorder is not None:
            resolved_recording_status = self._recorder.status
        return build_debugger_frame(
            self._session if session is None else session,
            session_id=self._session_id,
            revision=self._revision if revision is None else revision,
            view_mode=self._view_mode if view_mode is None else view_mode,
            preset=self._preset if preset is None else preset,
            include_stress=self._include_stress,
            recording_status=resolved_recording_status,
        )

    def _remember_command(
        self,
        key: tuple[str, str],
        record: _CommandRecord,
    ) -> None:
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
