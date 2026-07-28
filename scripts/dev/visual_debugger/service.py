"""Locked authoritative session ownership for the live browser debugger."""

from collections import OrderedDict
from dataclasses import dataclass
from secrets import token_urlsafe
from threading import RLock
from typing import Literal

from scripts.dev.visual_debugger.frame import build_debugger_frame
from scripts.dev.visual_debugger.input import (
    dispatch_command,
    sanitize_pov_pending_target,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.protocol import (
    ApiErrorV1,
    CommandRequestV1,
    CommandResponseV1,
    DebuggerFrameV1,
    Preset,
    ViewMode,
)
from scripts.dev.visual_debugger.scenarios import STRESS_SCENARIOS

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
    payload: CommandResponseV1 | ApiErrorV1
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
    def shutting_down(self) -> bool:
        """Return whether an accepted Exit has fenced new commands."""
        with self._lock:
            return self._shutting_down

    @property
    def faulted(self) -> bool:
        """Return whether an internal failure fenced all later commands."""
        with self._lock:
            return self._faulted

    def current_frame(self) -> DebuggerFrameV1:
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
                        payload=ApiErrorV1(
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
                    payload=CommandResponseV1(
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
                    payload=ApiErrorV1(
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
                    payload=ApiErrorV1(
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
                    payload=ApiErrorV1(
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

            self._session = candidate_session
            self._view_mode = candidate_view_mode
            self._preset = candidate_preset
            if dispatched.shutdown_requested:
                self._shutting_down = True
            if dispatched.changed:
                self._revision = candidate_revision
                self._frame = candidate_frame

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
            response = CommandResponseV1(
                result=result_kind,
                frame=self._frame,
                notice=dispatched.notice,
            )
            self._remember_command(
                command_key,
                _CommandRecord(
                    fingerprint=fingerprint,
                    shutdown_requested=dispatched.shutdown_requested,
                ),
            )
            return ServiceCommandResult(
                outcome="response",
                payload=response,
                shutdown_requested=dispatched.shutdown_requested,
            )

    def _build_frame(
        self,
        *,
        session: DebuggerSession | None = None,
        revision: int | None = None,
        view_mode: ViewMode | None = None,
        preset: Preset | None = None,
    ) -> DebuggerFrameV1:
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
        self._command_records[key] = record
        self._command_records.move_to_end(key)
        while len(self._command_records) > _COMMAND_RECORD_LIMIT:
            self._command_records.popitem(last=False)
