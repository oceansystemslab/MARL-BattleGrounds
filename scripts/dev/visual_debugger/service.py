"""Locked authoritative session ownership for the live browser debugger."""

from collections import OrderedDict
from dataclasses import dataclass
from secrets import token_urlsafe
from threading import RLock
from typing import Literal

from marl_battlegrounds.evaluation.metrics import EvaluationEpisodeObserverV1
from scripts.dev.visual_debugger.frame import LiveDebuggerFrame, build_debugger_frame
from scripts.dev.visual_debugger.input import (
    dispatch_command,
    sanitize_pov_pending_target,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.protocol import (
    ApiErrorV2,
    CommandRequestV1,
    CommandResponseV2,
    Preset,
    ViewMode,
)
from scripts.dev.visual_debugger.scenarios import STRESS_SCENARIOS, get_scenario

_COMMAND_RECORD_LIMIT = 256

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


@dataclass(frozen=True, slots=True)
class _CommandRecord:
    fingerprint: str
    shutdown_requested: bool


class DebuggerService:
    """Serialize commands around one immutable debugger session."""

    def __init__(
        self,
        session: DebuggerSession,
        *,
        view_mode: ViewMode,
        preset: Preset,
        include_stress: bool,
        session_id: str | None = None,
    ) -> None:
        if session.scenario_name in STRESS_SCENARIOS and not include_stress:
            msg = (
                f"stress scenario {session.scenario_name!r} requires "
                "include_stress=True."
            )
            raise ValueError(msg)
        self._session = (
            sanitize_pov_pending_target(session) if view_mode == "pov" else session
        )
        self._view_mode: ViewMode = view_mode
        self._preset: Preset = preset
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
        self._frame = self._build_frame()
        self._evaluation_observer = self._new_evaluation_observer(self._session)

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
            return self._evaluation_observer.validated_transition_count

    @property
    def evaluation_observer_lifecycle_state(self) -> str:
        """Return the service-owned observer lifecycle label."""
        with self._lock:
            return self._evaluation_observer.lifecycle_state

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

            try:
                dispatched = dispatch_command(
                    self._session,
                    request.command,
                    view_mode=self._view_mode,
                    preset=self._preset,
                    include_stress=self._include_stress,
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
            candidate_frame = self._frame
            if dispatched.changed:
                try:
                    candidate_frame = self._build_frame(
                        session=candidate_session,
                        revision=candidate_revision,
                        view_mode=candidate_view_mode,
                        preset=candidate_preset,
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
                    or self._evaluation_observer.validated_transition_count
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
                    self._evaluation_observer.append(
                        transition_view.transition,
                        transition_view.successor_frame,
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
            elif dispatched.episode_restarted:
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

            self._session = candidate_session
            self._view_mode = candidate_view_mode
            self._preset = candidate_preset
            if dispatched.shutdown_requested:
                self._shutting_down = True
            if dispatched.changed:
                self._revision = candidate_revision
                self._frame = candidate_frame
            self._evaluation_observer = candidate_observer
            self._command_records = candidate_command_records
            return candidate_result

    def _build_frame(
        self,
        *,
        session: DebuggerSession | None = None,
        revision: int | None = None,
        view_mode: ViewMode | None = None,
        preset: Preset | None = None,
    ) -> LiveDebuggerFrame:
        return build_debugger_frame(
            self._session if session is None else session,
            session_id=self._session_id,
            revision=self._revision if revision is None else revision,
            view_mode=self._view_mode if view_mode is None else view_mode,
            preset=self._preset if preset is None else preset,
            include_stress=self._include_stress,
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
