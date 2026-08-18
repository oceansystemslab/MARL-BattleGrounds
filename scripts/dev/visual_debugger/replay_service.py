"""Locked read-only service for one validated semantic replay bundle."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from secrets import token_urlsafe
from threading import RLock
from typing import Literal

from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV1,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovReplayArtifactV1,
    export_actor_pov_replay_v1,
)
from marl_battlegrounds.evaluation.replay import ReplayArtifactReferenceV1
from marl_battlegrounds.evaluation.replay_io import (
    LoadedReplayBundleV1,
    canonical_metric_report_artifact_json_bytes_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    EvaluationScenePresentationStateV1,
    SharedObsSourceMaterialProjectionV1,
    build_researcher_analyzer_projection_v2,
    build_shared_obs_authority_source_material_projection_v1,
    build_status_source_evidence_index_v2,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovProjectionIndexV1,
    build_actor_pov_analyzer_projection_v1,
    build_actor_pov_projection_index_v1,
)
from scripts.dev.visual_debugger.presentation import (
    build_replay_no_shared_obs_authorized_presentation_v1,
    build_replay_oracle_authorized_presentation_v1,
    build_replay_shared_obs_authorized_presentation_v1,
)
from scripts.dev.visual_debugger.presentation_protocol import (
    PresentationResourceResultV1,
)
from scripts.dev.visual_debugger.replay_protocol import (
    ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1,
    ActorPovProcessingDisclosureV1,
    ActorPovReplayCompletionBadgeV1,
    ActorPovReplayTimelineRowV1,
    ActorPovReplayTimelineV1,
    ActorPovReplayViewerFrameV1,
    ReplayAbsoluteSeekCommandV1,
    ReplayApiErrorV1,
    ReplayArtifactSummaryV1,
    ReplayCommandRequestV1,
    ReplayCommandResponseV1,
    ReplayCompletionBadgeV1,
    ReplayCursorV1,
    ReplayExitCommandV1,
    ReplayFirstFrameCommandV1,
    ReplayLastFrameCommandV1,
    ReplayNextFrameCommandV1,
    ReplayPresetV1,
    ReplayPreviousFrameCommandV1,
    ReplayProcessingBadgeV1,
    ReplaySelectAgentCommandV1,
    ReplaySetPovActorCommandV1,
    ReplaySetPresetCommandV1,
    ReplaySetRangesCommandV1,
    ReplaySetVerbosityCommandV1,
    ReplaySetViewCommandV1,
    ReplayTimelineEndpointKindV1,
    ReplayTimelineV1,
    ReplayViewerFrameV1,
    ReplayViewModeV1,
    ResearcherReplayTimelineRowV1,
    ResearcherReplayTimelineV1,
    ResearcherReplayViewerFrameV1,
    SharedObsAgentPovReplayArtifactSummaryV1,
    SharedObsAgentPovReplayTimelineRowV1,
    SharedObsAgentPovReplayTimelineV1,
    SharedObsAgentPovReplayViewerFrameV1,
)

_COMMAND_RECORD_LIMIT = 256
_METRIC_REPORT_SUFFIX = ".marlbg-metrics.json"

type ReplayServiceOutcomeV1 = Literal[
    "response",
    "invalid_cursor",
    "audience_unavailable",
    "stale_revision",
    "command_id_conflict",
    "server_shutting_down",
    "service_faulted",
]

type ReplayMetricReportOutcomeV1 = Literal[
    "available",
    "missing",
    "forbidden",
]


@dataclass(frozen=True, slots=True)
class ReplayServiceCommandResultV1:
    """One transport-neutral replay command result."""

    outcome: ReplayServiceOutcomeV1
    payload: ReplayCommandResponseV1 | ReplayApiErrorV1
    shutdown_requested: bool = False


@dataclass(frozen=True, slots=True)
class ReplayMetricReportResultV1:
    """One transport-neutral canonical metric-report retrieval result."""

    outcome: ReplayMetricReportOutcomeV1
    payload: bytes | None
    filename: str | None

    def __post_init__(self) -> None:
        if self.outcome not in ("available", "missing", "forbidden"):
            raise ValueError("unknown replay metric-report outcome")
        if self.outcome == "available":
            if type(self.payload) is not bytes:
                raise TypeError("available metric report requires immutable bytes")
            if type(self.filename) is not str or not self.filename:
                raise TypeError("available metric report requires a filename")
            return
        if self.payload is not None or self.filename is not None:
            raise ValueError(
                "unavailable metric report cannot carry bytes or a filename"
            )


@dataclass(frozen=True, slots=True)
class _CommandRecord:
    fingerprint: str
    shutdown_requested: bool


@dataclass(frozen=True, slots=True)
class _PovCacheEntry:
    artifact: ActorPovReplayArtifactV1
    projection_index: ActorPovProjectionIndexV1
    timeline: ActorPovReplayTimelineV1


def _safe_metric_report_filename(episode_id: str) -> str:
    """Return the bounded attachment basename derived from one episode ID."""
    if type(episode_id) is not str:
        raise TypeError("episode_id must be a string")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", episode_id).strip("._-")
    stem = stem[:96].strip("._-")
    stem = stem.replace(_METRIC_REPORT_SUFFIX, "-").strip("._-") or "replay"
    return f"{stem}{_METRIC_REPORT_SUFFIX}"


def _replay_reference(bundle: LoadedReplayBundleV1) -> ReplayArtifactReferenceV1:
    replay = bundle.replay
    return ReplayArtifactReferenceV1(
        artifact_id=replay.artifact_id,
        episode_id=replay.header.context.identity.episode_id,
        context_digest_sha256=replay.header.context_digest_sha256,
        trajectory_content_digest_sha256=replay.trajectory_content_digest_sha256,
        canonical_digest_sha256=replay.canonical_digest_sha256,
        canonical_byte_length=len(canonical_json_bytes(replay)),
    )


def _completion_badge(bundle: LoadedReplayBundleV1) -> ReplayCompletionBadgeV1:
    completion = bundle.replay.completion
    return ReplayCompletionBadgeV1(
        episode_id=completion.episode_id,
        completion_state=completion.completion_state,
        expected_transition_count=completion.expected_transition_count,
        validated_transition_count=completion.validated_transition_count,
        last_valid_frame_index=completion.last_valid_frame_index,
        last_valid_frame_id=completion.last_valid_frame_id,
        terminated=completion.terminated,
        truncated=completion.truncated,
        completion_bases=completion.completion_bases,
        end_or_failure_reason=completion.end_or_failure_reason,
        failure_origin=completion.failure_origin,
    )


def _processing_badge(bundle: LoadedReplayBundleV1) -> ReplayProcessingBadgeV1:
    processing = bundle.replay.processing_status
    failure = processing.failure
    return ReplayProcessingBadgeV1(
        status=processing.status,
        processed_transition_count=processing.processed_transition_count,
        failure_stage=None if failure is None else failure.stage,
        failure_code=None if failure is None else failure.code,
        attempted_transition_index=(
            None if failure is None else failure.attempted_transition_index
        ),
    )


def _pov_completion_badge(
    artifact: ActorPovReplayArtifactV1,
) -> ActorPovReplayCompletionBadgeV1:
    content = artifact.content
    completion = content.completion
    return ActorPovReplayCompletionBadgeV1(
        episode_id=content.episode_id,
        completion_state=completion.completion_state,
        expected_transition_count=completion.expected_transition_count,
        captured_transition_count=completion.captured_transition_count,
        terminated=completion.terminated,
        truncated=completion.truncated,
        completion_bases=completion.completion_bases,
        public_end_or_failure_reason=completion.public_end_or_failure_reason,
    )


def _shared_completion_badge(
    bundle: LoadedReplayBundleV1,
) -> ActorPovReplayCompletionBadgeV1:
    """Project physical SharedObs completion without researcher failure detail."""
    completion = bundle.replay.completion
    return ActorPovReplayCompletionBadgeV1(
        episode_id=completion.episode_id,
        completion_state=completion.completion_state,
        expected_transition_count=completion.expected_transition_count,
        captured_transition_count=completion.validated_transition_count,
        terminated=completion.terminated,
        truncated=completion.truncated,
        completion_bases=completion.completion_bases,
        public_end_or_failure_reason=(
            None if completion.completion_state == "complete" else "captured_prefix"
        ),
    )


def _endpoint_kind(
    completion: ReplayCompletionBadgeV1 | ActorPovReplayCompletionBadgeV1,
) -> ReplayTimelineEndpointKindV1:
    if completion.completion_state != "complete":
        return "captured_prefix"
    if completion.completion_bases == ("task_terminal", "declared_horizon"):
        return "task_terminal_and_declared_horizon"
    if completion.completion_bases == ("task_terminal",):
        return "task_terminal"
    if completion.completion_bases == ("declared_horizon",):
        return "declared_horizon"
    raise ValueError("complete replay has an unsupported endpoint basis")


class ReplayViewerService:
    """Serialize immutable cursor and viewer-owned presentation around one replay.

    Initial reference, selection, and lane values describe the viewer handoff;
    they are not historical fields recovered independently at each replay cursor.
    """

    def __init__(
        self,
        bundle: LoadedReplayBundleV1,
        *,
        initial_frame_index: int = 0,
        view_mode: ReplayViewModeV1 = "researcher",
        reference_global_slot: int | None = None,
        selected_global_slot: int | None = None,
        armed_lane: Literal[0, 1] | None = None,
        pov_global_slot: int | None = None,
        preset: ReplayPresetV1 | Literal["technical", "debug"] = "analysis",
        show_ranges: bool = True,
        verbose: bool = False,
        viewer_session_id: str | None = None,
    ) -> None:
        if type(bundle) is not LoadedReplayBundleV1:
            raise TypeError("bundle must be the exact LoadedReplayBundleV1 root")
        if bundle.status not in ("complete", "metric_report_missing") or (
            bundle.metric_report_artifact is None
        ) != (bundle.status == "metric_report_missing"):
            raise ValueError(
                "loaded replay status must match metric-report sidecar availability"
            )
        if type(initial_frame_index) is not int or not (
            0 <= initial_frame_index < len(bundle.replay.frames)
        ):
            raise ValueError("initial frame index is outside the captured replay")
        if view_mode not in ("researcher", "pov"):
            raise ValueError("unknown replay view mode")
        if preset not in ("presentation", "analysis", "technical", "debug"):
            raise ValueError("unknown replay preset")
        if type(show_ranges) is not bool or type(verbose) is not bool:
            raise TypeError("replay presentation flags must be booleans")
        if viewer_session_id is not None and (
            type(viewer_session_id) is not str or not viewer_session_id.strip()
        ):
            raise ValueError("viewer_session_id must be a nonempty string when set")

        self._bundle = bundle
        self._replay = bundle.replay
        self._context = bundle.replay.header.context
        self._active_slots = tuple(
            row.global_slot for row in self._context.roster if row.configured_active
        )
        focal_slots = tuple(
            row.global_slot
            for row in self._context.policy_assignments
            if isinstance(row, AssignedPolicySlotV1) and row.evaluation_role == "focal"
        )
        default_slot = min(focal_slots) if focal_slots else None
        researcher_slot = (
            default_slot if reference_global_slot is None else reference_global_slot
        )
        inspection_slot = selected_global_slot
        actor_slot = default_slot if pov_global_slot is None else pov_global_slot
        self._require_active_or_none(
            researcher_slot,
            name="reference_global_slot",
        )
        self._require_active_or_none(inspection_slot, name="selected_global_slot")
        self._require_active_or_none(actor_slot, name="pov_global_slot")
        if armed_lane is not None and (
            type(armed_lane) is not int or armed_lane not in (0, 1)
        ):
            raise ValueError("armed_lane must be the Python int zero or one, or None")
        if inspection_slot is not None and researcher_slot is None:
            raise ValueError("selected_global_slot requires a researcher reference")
        if armed_lane is not None and inspection_slot is None:
            raise ValueError("armed_lane requires a selected_global_slot")
        if view_mode == "pov" and actor_slot is None:
            raise ValueError("POV replay view requires a configured-active actor")

        self._artifact_summary = ReplayArtifactSummaryV1(
            replay_reference=_replay_reference(bundle),
            expected_transition_count=self._replay.header.expected_transition_count,
            recorded_transition_count=len(self._replay.transitions),
            recorded_frame_count=len(self._replay.frames),
            metric_report_availability=(
                "available" if bundle.metric_report_artifact is not None else "missing"
            ),
        )
        self._pov_artifact_summary = ReplayArtifactSummaryV1(
            replay_reference=self._artifact_summary.replay_reference,
            expected_transition_count=self._artifact_summary.expected_transition_count,
            recorded_transition_count=self._artifact_summary.recorded_transition_count,
            recorded_frame_count=self._artifact_summary.recorded_frame_count,
            metric_report_availability=ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1,
        )
        self._completion = _completion_badge(bundle)
        self._shared_completion = _shared_completion_badge(bundle)
        self._processing = _processing_badge(bundle)
        self._status_index = build_status_source_evidence_index_v2(
            self._context,
            self._replay.frames,
            self._replay.transitions,
        )
        self._researcher_timeline = self._build_researcher_timeline()
        self._pov_cache: dict[int, _PovCacheEntry] = {}
        self._shared_timeline_cache: dict[int, SharedObsAgentPovReplayTimelineV1] = {}

        self._viewer_session_id = (
            token_urlsafe(24) if viewer_session_id is None else viewer_session_id
        )
        self._revision = 0
        self._frame_index = initial_frame_index
        self._cursor_generation = 0
        self._choreography_generation = 0
        self._view_mode: ReplayViewModeV1 = view_mode
        self._reference_global_slot = researcher_slot
        self._inspection_global_slot = inspection_slot
        self._armed_lane = armed_lane
        self._pov_global_slot = actor_slot
        self._preset: ReplayPresetV1 = "analysis"
        self._show_ranges = show_ranges
        self._verbose = False
        self._shutting_down = False
        self._faulted = False
        self._lock = RLock()
        self._command_records: OrderedDict[tuple[str, str], _CommandRecord] = (
            OrderedDict()
        )
        self._frame = self._build_frame(
            revision=self._revision,
            frame_index=self._frame_index,
            cursor_generation=self._cursor_generation,
            choreography_generation=self._choreography_generation,
            view_mode=self._view_mode,
            inspection_global_slot=self._inspection_global_slot,
            armed_lane=self._armed_lane,
            pov_global_slot=self._pov_global_slot,
            preset=self._preset,
            show_ranges=self._show_ranges,
            verbose=self._verbose,
            pov_cache=self._pov_cache,
            shared_timeline_cache=self._shared_timeline_cache,
        )

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def command_cache_size(self) -> int:
        with self._lock:
            return len(self._command_records)

    @property
    def pov_index_build_count(self) -> int:
        with self._lock:
            return len(self._pov_cache)

    @property
    def shared_timeline_build_count(self) -> int:
        with self._lock:
            return len(self._shared_timeline_cache)

    @property
    def shutting_down(self) -> bool:
        with self._lock:
            return self._shutting_down

    @property
    def faulted(self) -> bool:
        with self._lock:
            return self._faulted

    def current_frame(self) -> ReplayViewerFrameV1:
        with self._lock:
            return self._frame

    def current_metric_report(self) -> ReplayMetricReportResultV1:
        """Return canonical metric bytes only for the locked Oracle authority."""
        with self._lock:
            if self._view_mode != "researcher":
                return ReplayMetricReportResultV1(
                    outcome="forbidden",
                    payload=None,
                    filename=None,
                )
            artifact = self._bundle.metric_report_artifact
            if artifact is None:
                return ReplayMetricReportResultV1(
                    outcome="missing",
                    payload=None,
                    filename=None,
                )
            return ReplayMetricReportResultV1(
                outcome="available",
                payload=canonical_metric_report_artifact_json_bytes_v1(artifact),
                filename=_safe_metric_report_filename(
                    self._context.identity.episode_id
                ),
            )

    def current_presentation(self) -> PresentationResourceResultV1:
        """Build the authorized resource from one committed replay snapshot."""
        with self._lock:
            if self._view_mode != "researcher":
                if self._context.execution_information_mode == "no_shared_obs":
                    raw_frame = self._frame
                    if type(raw_frame) is not ActorPovReplayViewerFrameV1:
                        raise RuntimeError(
                            "NoSharedObs replay view does not hold its committed "
                            "recipient frame"
                        )
                    self._validate_committed_presentation_snapshot(
                        raw_frame,
                        oracle=False,
                    )
                    if self._pov_global_slot is None:
                        raise RuntimeError(
                            "NoSharedObs replay view has no fixed POV recipient"
                        )
                    entry = self._pov_cache.get(self._pov_global_slot)
                    if entry is None:
                        raise RuntimeError(
                            "committed NoSharedObs replay frame has no POV cache entry"
                        )
                    if (
                        entry.projection_index.content.selected_global_slot
                        != self._pov_global_slot
                    ):
                        raise RuntimeError(
                            "committed NoSharedObs cache entry does not join the "
                            "fixed POV recipient"
                        )
                    frame = build_replay_no_shared_obs_authorized_presentation_v1(
                        entry.projection_index,
                        raw_frame,
                        public_catalog=self._context.static_mechanics_catalog,
                        source_authority_epoch=raw_frame.revision,
                    )
                    return PresentationResourceResultV1(
                        outcome="response",
                        payload=frame,
                    )
                raw_frame = self._frame
                if type(raw_frame) is not SharedObsAgentPovReplayViewerFrameV1:
                    raise RuntimeError(
                        "SharedObs replay view does not hold its committed "
                        "private recipient frame"
                    )
                self._validate_committed_presentation_snapshot(
                    raw_frame,
                    oracle=False,
                )
                if self._pov_global_slot is None:
                    raise RuntimeError(
                        "SharedObs replay view has no fixed POV recipient"
                    )
                frame_index = raw_frame.cursor.frame_index
                current_recipient, current_nonrecipient = (
                    self._shared_authority_sources(
                        frame_index=frame_index,
                        recipient_global_slot=self._pov_global_slot,
                    )
                )
                if frame_index == 0:
                    previous_recipient = None
                    previous_nonrecipient: tuple[
                        SharedObsSourceMaterialProjectionV1, ...
                    ] = ()
                    incoming_transition = None
                else:
                    previous_recipient, previous_nonrecipient = (
                        self._shared_authority_sources(
                            frame_index=frame_index - 1,
                            recipient_global_slot=self._pov_global_slot,
                        )
                    )
                    incoming_transition = self._replay.transitions[frame_index - 1]
                outgoing_transition = (
                    None
                    if frame_index == len(self._replay.transitions)
                    else self._replay.transitions[frame_index]
                )
                frame = build_replay_shared_obs_authorized_presentation_v1(
                    raw_frame,
                    public_catalog=self._context.static_mechanics_catalog,
                    source_authority_epoch=raw_frame.revision,
                    authorized_recipient_global_slot=self._pov_global_slot,
                    current_recipient_source_material=current_recipient,
                    current_active_nonrecipient_source_material=(current_nonrecipient),
                    previous_recipient_source_material=previous_recipient,
                    previous_active_nonrecipient_source_material=(
                        previous_nonrecipient
                    ),
                    incoming_transition=incoming_transition,
                    outgoing_transition=outgoing_transition,
                )
                return PresentationResourceResultV1(outcome="response", payload=frame)

            raw_frame = self._frame
            if type(raw_frame) is not ResearcherReplayViewerFrameV1:
                raise RuntimeError(
                    "Oracle replay view does not hold its committed researcher frame"
                )
            self._validate_committed_presentation_snapshot(raw_frame, oracle=True)
            source_frame_index = raw_frame.cursor.frame_index
            current_frame = self._replay.frames[source_frame_index]
            incoming_transition = (
                None
                if source_frame_index == 0
                else self._replay.transitions[source_frame_index - 1]
            )
            outgoing_transition = (
                self._replay.transitions[source_frame_index]
                if self._inspection_global_slot is not None
                and source_frame_index < len(self._replay.transitions)
                else None
            )
            frame = build_replay_oracle_authorized_presentation_v1(
                self._context,
                current_frame,
                raw_frame,
                source_authority_epoch=raw_frame.revision,
                selected_internal_slot=self._inspection_global_slot,
                incoming_transition=incoming_transition,
                outgoing_transition=outgoing_transition,
            )
            return PresentationResourceResultV1(
                outcome="response",
                payload=frame,
            )

    def _validate_committed_presentation_snapshot(
        self,
        raw_frame: ReplayViewerFrameV1,
        *,
        oracle: bool,
    ) -> None:
        """Require the raw envelope to equal this locked service snapshot."""
        cursor = raw_frame.cursor
        if type(cursor) is not ReplayCursorV1:
            raise RuntimeError("committed replay frame has an invalid cursor root")
        if (
            type(raw_frame.viewer_session_id) is not str
            or raw_frame.viewer_session_id != self._viewer_session_id
            or type(raw_frame.revision) is not int
            or raw_frame.revision != self._revision
            or type(cursor.frame_index) is not int
            or cursor.frame_index != self._frame_index
            or type(cursor.final_frame_index) is not int
            or cursor.final_frame_index != len(self._replay.transitions)
        ):
            raise RuntimeError("committed replay frame does not join service state")
        if oracle and (
            type(cursor.cursor_generation) is not int
            or cursor.cursor_generation != self._cursor_generation
            or type(cursor.choreography_generation) is not int
            or cursor.choreography_generation != self._choreography_generation
        ):
            raise RuntimeError("committed Oracle cursor generations are stale")
        if not oracle and self._context.execution_information_mode == "shared_obs":
            if type(raw_frame) is not SharedObsAgentPovReplayViewerFrameV1:
                raise RuntimeError(
                    "committed SharedObs frame has the wrong product root"
                )
            if self._pov_global_slot is None:
                raise RuntimeError("committed SharedObs frame has no fixed recipient")
            timeline = self._shared_timeline_cache.get(self._pov_global_slot)
            if timeline is None:
                raise RuntimeError(
                    "committed SharedObs frame has no private timeline cache entry"
                )
            roster = self._context.roster[self._pov_global_slot]
            episode_id = self._context.identity.episode_id
            prefix = f"{episode_id}:shared-obs-visual-union:{roster.public_agent_id}"
            expected_incoming_id = (
                None
                if self._frame_index == 0
                else f"{prefix}:transition:{self._frame_index - 1}"
            )
            source_frame = self._replay.frames[self._frame_index]
            if (
                type(raw_frame.artifact_summary)
                is not SharedObsAgentPovReplayArtifactSummaryV1
                or raw_frame.artifact_summary != timeline.artifact_summary
                or raw_frame.timeline_id != timeline.timeline_id
                or raw_frame.public_agent_id != roster.public_agent_id
                or raw_frame.recipient_frame_id != f"{prefix}:frame:{self._frame_index}"
                or raw_frame.simulator_step_count != source_frame.simulator_step_count
                or raw_frame.incoming_recipient_transition_id != expected_incoming_id
                or type(raw_frame.completion) is not ActorPovReplayCompletionBadgeV1
                or raw_frame.completion != self._shared_completion
            ):
                raise RuntimeError(
                    "committed SharedObs transport identity does not join service "
                    "authority"
                )
        if oracle:
            if type(raw_frame) is not ResearcherReplayViewerFrameV1:
                raise RuntimeError("committed Oracle frame has the wrong product root")
            recorded_scale = (
                self._context.resolved_env_config.ordinary_movement_distance_scale
            )
            source_frame = self._replay.frames[self._frame_index]
            incoming = self._transition_view(self._frame_index)
            expected_projection = build_researcher_analyzer_projection_v2(
                self._context,
                source_frame,
                transition_view=incoming,
                presentation=EvaluationScenePresentationStateV1(
                    controlled_global_slot=self._reference_global_slot,
                    selected_global_slot=self._inspection_global_slot,
                    armed_lane=self._armed_lane,
                    show_ranges=self._show_ranges,
                ),
                status_source_evidence_state=self._status_index.state_for_frame(
                    self._frame_index
                ),
            )
            if (
                type(raw_frame.artifact_summary) is not ReplayArtifactSummaryV1
                or raw_frame.artifact_summary is not self._artifact_summary
                or type(raw_frame.timeline_id) is not str
                or raw_frame.timeline_id != self._researcher_timeline.timeline_id
                or type(raw_frame.recorded_ordinary_movement_distance_scale)
                is not float
                or raw_frame.recorded_ordinary_movement_distance_scale != recorded_scale
                or type(raw_frame.show_ranges) is not bool
                or raw_frame.show_ranges != self._show_ranges
                or type(raw_frame.projection) is not type(expected_projection)
                or raw_frame.projection != expected_projection
            ):
                raise RuntimeError(
                    "committed Oracle provenance does not join service authority"
                )

    def current_timeline(self) -> ReplayTimelineV1:
        with self._lock:
            if self._view_mode == "researcher":
                return self._researcher_timeline
            if self._pov_global_slot is None:
                raise RuntimeError("POV replay view has no selected actor")
            if self._context.execution_information_mode == "no_shared_obs":
                return self._pov_entry(
                    self._pov_global_slot,
                    cache=self._pov_cache,
                ).timeline
            return self._shared_timeline(
                self._pov_global_slot,
                cache=self._shared_timeline_cache,
            )

    def apply_command(
        self,
        request: ReplayCommandRequestV1,
    ) -> ReplayServiceCommandResultV1:
        """Apply at most one replay command under revision/idempotency guards."""
        if type(request) is not ReplayCommandRequestV1:
            raise TypeError("request must be the exact ReplayCommandRequestV1 root")
        command_key = (request.client_id, request.command_id)
        fingerprint = request.model_dump_json()
        with self._lock:
            previous = self._command_records.get(command_key)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    return self._error_result(
                        "command_id_conflict",
                        "command_id_conflict",
                        "This client reused a command_id for a different request.",
                    )
                result = ReplayServiceCommandResultV1(
                    outcome="response",
                    payload=ReplayCommandResponseV1(
                        result="duplicate",
                        frame=self._frame,
                        notice="Command already processed; current frame returned.",
                        animate_incoming=False,
                    ),
                    shutdown_requested=previous.shutdown_requested,
                )
                records = self._command_records.copy()
                records.move_to_end(command_key)
                self._command_records = records
                return result
            if self._faulted:
                return self._record_error(
                    command_key,
                    fingerprint,
                    "service_faulted",
                    "internal_error",
                    "The replay viewer entered a safe fault state; restart it.",
                )
            if self._shutting_down:
                return self._record_error(
                    command_key,
                    fingerprint,
                    "server_shutting_down",
                    "server_shutting_down",
                    "The replay viewer is shutting down; the command was not applied.",
                )
            if request.base_revision != self._revision:
                return self._record_error(
                    command_key,
                    fingerprint,
                    "stale_revision",
                    "stale_revision",
                    "The replay cursor advanced; the latest frame is attached.",
                )

            command = request.command
            frame_index = self._frame_index
            cursor_generation = self._cursor_generation
            choreography_generation = self._choreography_generation
            view_mode = self._view_mode
            inspection_global_slot = self._inspection_global_slot
            armed_lane = self._armed_lane
            pov_global_slot = self._pov_global_slot
            preset = self._preset
            show_ranges = self._show_ranges
            verbose = self._verbose
            pov_cache = self._pov_cache.copy()
            shared_timeline_cache = self._shared_timeline_cache.copy()
            changed = False
            animate_incoming = False
            shutdown_requested = False
            notice: str | None = None

            if type(command) is ReplayAbsoluteSeekCommandV1:
                if (
                    command.frame_index
                    > self._artifact_summary.recorded_transition_count
                ):
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "invalid_cursor",
                        "invalid_cursor",
                        "The requested frame is outside the captured replay prefix.",
                    )
                frame_index = command.frame_index
                cursor_generation += 1
                changed = True
            elif type(command) is ReplayFirstFrameCommandV1:
                frame_index = 0
                cursor_generation += 1
                changed = True
            elif type(command) is ReplayLastFrameCommandV1:
                frame_index = self._artifact_summary.recorded_transition_count
                cursor_generation += 1
                changed = True
            elif type(command) is ReplayPreviousFrameCommandV1:
                if frame_index == 0:
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "invalid_cursor",
                        "invalid_cursor",
                        "The replay cursor cannot move before frame zero.",
                    )
                frame_index -= 1
                cursor_generation += 1
                changed = True
            elif type(command) is ReplayNextFrameCommandV1:
                if frame_index == self._artifact_summary.recorded_transition_count:
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "invalid_cursor",
                        "invalid_cursor",
                        "The replay cursor cannot move past the captured prefix.",
                    )
                frame_index += 1
                cursor_generation += 1
                choreography_generation += 1
                changed = True
                animate_incoming = True
            elif type(command) is ReplaySelectAgentCommandV1:
                if view_mode != "researcher":
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "audience_unavailable",
                        "audience_unavailable",
                        "Researcher selection is unavailable in POV replay.",
                    )
                try:
                    self._require_active_or_none(
                        command.selected_global_slot,
                        name="selected_global_slot",
                    )
                except ValueError:
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "audience_unavailable",
                        "audience_unavailable",
                        "The requested researcher selection is unavailable.",
                    )
                if command.selected_global_slot != inspection_global_slot:
                    inspection_global_slot = command.selected_global_slot
                    armed_lane = None
                    changed = True
            elif type(command) is ReplaySetViewCommandV1:
                if command.view_mode == "pov" and pov_global_slot is None:
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "audience_unavailable",
                        "audience_unavailable",
                        "No configured-active actor is available for POV replay.",
                    )
                if command.view_mode != view_mode:
                    view_mode = command.view_mode
                    changed = True
            elif type(command) is ReplaySetPovActorCommandV1:
                try:
                    self._require_active_or_none(
                        command.global_slot,
                        name="pov_global_slot",
                    )
                except ValueError:
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "audience_unavailable",
                        "audience_unavailable",
                        "The requested POV actor is unavailable.",
                    )
                if command.global_slot != pov_global_slot:
                    pov_global_slot = command.global_slot
                    changed = True
            elif type(command) is ReplaySetPresetCommandV1:
                # V1 compatibility command. All legacy preset requests
                # canonicalize to the single product Analysis presentation.
                preset = "analysis"
            elif type(command) is ReplaySetRangesCommandV1:
                if view_mode != "researcher":
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "audience_unavailable",
                        "audience_unavailable",
                        "Researcher ranges are unavailable in POV replay.",
                    )
                if command.show_ranges != show_ranges:
                    show_ranges = command.show_ranges
                    changed = True
            elif type(command) is ReplaySetVerbosityCommandV1:
                # Retained as a V1 compatibility command. Verbose presentation
                # is no longer a product mode, so both values are a fixed no-op.
                verbose = False
            elif type(command) is ReplayExitCommandV1:
                shutdown_requested = True
            else:  # pragma: no cover - exact discriminated union is exhaustive.
                raise TypeError("unsupported replay command root")

            candidate_revision = self._revision + int(changed)
            try:
                candidate_frame = (
                    self._build_frame(
                        revision=candidate_revision,
                        frame_index=frame_index,
                        cursor_generation=cursor_generation,
                        choreography_generation=choreography_generation,
                        view_mode=view_mode,
                        inspection_global_slot=inspection_global_slot,
                        armed_lane=armed_lane,
                        pov_global_slot=pov_global_slot,
                        preset=preset,
                        show_ranges=show_ranges,
                        verbose=verbose,
                        pov_cache=pov_cache,
                        shared_timeline_cache=shared_timeline_cache,
                    )
                    if changed
                    else self._frame
                )
                result_kind: Literal["applied", "no_op", "shutdown_scheduled"] = (
                    "shutdown_scheduled"
                    if shutdown_requested
                    else "applied"
                    if changed
                    else "no_op"
                )
                response = ReplayCommandResponseV1(
                    result=result_kind,
                    frame=candidate_frame,
                    notice=notice,
                    animate_incoming=animate_incoming,
                )
                records = self._command_records.copy()
                self._remember_in(records, command_key, fingerprint, shutdown_requested)
                result = ReplayServiceCommandResultV1(
                    outcome="response",
                    payload=response,
                    shutdown_requested=shutdown_requested,
                )
            except Exception:
                records = self._command_records.copy()
                self._remember_in(records, command_key, fingerprint, False)
                self._faulted = True
                self._command_records = records
                raise

            self._revision = candidate_revision
            self._frame_index = frame_index
            self._cursor_generation = cursor_generation
            self._choreography_generation = choreography_generation
            self._view_mode = view_mode
            self._inspection_global_slot = inspection_global_slot
            self._armed_lane = armed_lane
            self._pov_global_slot = pov_global_slot
            self._preset = preset
            self._show_ranges = show_ranges
            self._verbose = verbose
            self._pov_cache = pov_cache
            self._shared_timeline_cache = shared_timeline_cache
            if shutdown_requested:
                self._shutting_down = True
            self._frame = candidate_frame
            self._command_records = records
            return result

    def _build_frame(
        self,
        *,
        revision: int,
        frame_index: int,
        cursor_generation: int,
        choreography_generation: int,
        view_mode: ReplayViewModeV1,
        inspection_global_slot: int | None,
        armed_lane: Literal[0, 1] | None,
        pov_global_slot: int | None,
        preset: ReplayPresetV1,
        show_ranges: bool,
        verbose: bool,
        pov_cache: dict[int, _PovCacheEntry],
        shared_timeline_cache: dict[int, SharedObsAgentPovReplayTimelineV1],
    ) -> ReplayViewerFrameV1:
        cursor = ReplayCursorV1(
            frame_index=frame_index,
            final_frame_index=self._artifact_summary.recorded_transition_count,
            cursor_generation=cursor_generation,
            choreography_generation=choreography_generation,
        )
        frame = self._replay.frames[frame_index]
        if view_mode == "researcher":
            incoming = self._transition_view(frame_index)
            projection = build_researcher_analyzer_projection_v2(
                self._context,
                frame,
                transition_view=incoming,
                presentation=EvaluationScenePresentationStateV1(
                    controlled_global_slot=self._reference_global_slot,
                    selected_global_slot=inspection_global_slot,
                    armed_lane=armed_lane,
                    show_ranges=show_ranges,
                ),
                status_source_evidence_state=self._status_index.state_for_frame(
                    frame_index
                ),
            )
            return ResearcherReplayViewerFrameV1(
                viewer_session_id=self._viewer_session_id,
                revision=revision,
                artifact_summary=self._artifact_summary,
                timeline_id=self._researcher_timeline.timeline_id,
                cursor=cursor,
                preset=preset,
                verbose=False,
                frame_id=frame.frame_id,
                simulator_step_count=frame.simulator_step_count,
                incoming_transition_index=(
                    None if incoming is None else incoming.transition.transition_index
                ),
                incoming_transition_id=(
                    None if incoming is None else incoming.transition.transition_id
                ),
                completion=self._completion,
                processing=self._processing,
                show_ranges=show_ranges,
                recorded_ordinary_movement_distance_scale=(
                    self._context.resolved_env_config.ordinary_movement_distance_scale
                ),
                projection=projection,
            )
        if pov_global_slot is None:
            raise ValueError("POV replay view requires a configured-active actor")
        roster = self._context.roster[pov_global_slot]
        if self._context.execution_information_mode == "no_shared_obs":
            incoming = self._transition_view(frame_index)
            entry = self._pov_entry(pov_global_slot, cache=pov_cache)
            projection = build_actor_pov_analyzer_projection_v1(
                entry.projection_index,
                frame_index=frame_index,
            )
            pov_frame = entry.artifact.content.frames[frame_index]
            return ActorPovReplayViewerFrameV1(
                viewer_session_id=self._viewer_session_id,
                revision=revision,
                artifact_summary=self._pov_artifact_summary,
                timeline_id=entry.timeline.timeline_id,
                cursor=cursor,
                preset=preset,
                verbose=False,
                pov_global_slot=pov_global_slot,
                public_agent_id=roster.public_agent_id,
                pov_frame_id=pov_frame.pov_frame_id,
                simulator_step_count=pov_frame.simulator_step_count,
                incoming_pov_transition_id=projection.incoming_transition_id,
                completion=_pov_completion_badge(entry.artifact),
                processing_disclosure=ActorPovProcessingDisclosureV1(),
                projection=projection,
            )
        timeline = self._shared_timeline(
            pov_global_slot,
            cache=shared_timeline_cache,
        )
        prefix = (
            f"{self._context.identity.episode_id}:shared-obs-visual-union:"
            f"{roster.public_agent_id}"
        )
        return SharedObsAgentPovReplayViewerFrameV1(
            schema_version=1,
            frame_kind="shared_obs_agent_pov_replay_viewer",
            viewer_session_id=self._viewer_session_id,
            revision=revision,
            artifact_summary=timeline.artifact_summary,
            timeline_id=timeline.timeline_id,
            cursor=cursor,
            preset="analysis",
            verbose=False,
            view_mode="pov",
            public_agent_id=roster.public_agent_id,
            recipient_frame_id=f"{prefix}:frame:{frame_index}",
            simulator_step_count=frame.simulator_step_count,
            incoming_recipient_transition_id=(
                None if frame_index == 0 else f"{prefix}:transition:{frame_index - 1}"
            ),
            completion=self._shared_completion,
        )

    def _build_researcher_timeline(self) -> ResearcherReplayTimelineV1:
        endpoint = _endpoint_kind(self._completion)
        final = len(self._replay.frames) - 1
        rows = tuple(
            ResearcherReplayTimelineRowV1(
                frame_index=index,
                frame_id=frame.frame_id,
                simulator_step_count=frame.simulator_step_count,
                incoming_transition_id=(
                    None
                    if index == 0
                    else self._replay.transitions[index - 1].transition_id
                ),
                incoming_event_count=(
                    0 if index == 0 else len(self._replay.transitions[index - 1].events)
                ),
                endpoint_kind=endpoint if index == final else "none",
            )
            for index, frame in enumerate(self._replay.frames)
        )
        return ResearcherReplayTimelineV1(
            timeline_id=f"{self._replay.artifact_id}:timeline:researcher",
            artifact_summary=self._artifact_summary,
            final_frame_index=final,
            completion=self._completion,
            rows=rows,
        )

    def _pov_entry(
        self,
        global_slot: int,
        *,
        cache: dict[int, _PovCacheEntry],
    ) -> _PovCacheEntry:
        cached = cache.get(global_slot)
        if cached is not None:
            return cached
        artifact = export_actor_pov_replay_v1(
            self._replay,
            global_slot=global_slot,
        )
        projection_index = build_actor_pov_projection_index_v1(artifact.content)
        completion = _pov_completion_badge(artifact)
        endpoint = _endpoint_kind(completion)
        final = len(artifact.content.frames) - 1
        rows = tuple(
            ActorPovReplayTimelineRowV1(
                frame_index=index,
                pov_frame_id=frame.pov_frame_id,
                simulator_step_count=frame.simulator_step_count,
                incoming_pov_transition_id=(
                    None
                    if index == 0
                    else artifact.content.transitions[index - 1].pov_transition_id
                ),
                incoming_cue_count=(
                    0
                    if index == 0
                    else len(artifact.content.transitions[index - 1].cues)
                ),
                endpoint_kind=endpoint if index == final else "none",
            )
            for index, frame in enumerate(artifact.content.frames)
        )
        timeline = ActorPovReplayTimelineV1(
            timeline_id=(
                f"{self._replay.artifact_id}:timeline:actor-pov:"
                f"{artifact.content.public_agent_id}"
            ),
            artifact_summary=self._pov_artifact_summary,
            final_frame_index=final,
            pov_global_slot=global_slot,
            public_agent_id=artifact.content.public_agent_id,
            completion=completion,
            rows=rows,
        )
        entry = _PovCacheEntry(
            artifact=artifact,
            projection_index=projection_index,
            timeline=timeline,
        )
        cache[global_slot] = entry
        return entry

    def _shared_authority_sources(
        self,
        *,
        frame_index: int,
        recipient_global_slot: int,
    ) -> tuple[
        SharedObsSourceMaterialProjectionV1,
        tuple[SharedObsSourceMaterialProjectionV1, ...],
    ]:
        """Build one uncached, same-epoch fixed-recipient authority source set."""
        if self._context.execution_information_mode != "shared_obs":
            raise RuntimeError("Shared authority sources require a SharedObs replay")
        if type(frame_index) is not int or not (
            0 <= frame_index < len(self._replay.frames)
        ):
            raise RuntimeError("Shared authority source frame is outside the replay")
        if (
            type(recipient_global_slot) is not int
            or recipient_global_slot not in self._active_slots
        ):
            raise RuntimeError("Shared authority recipient is not configured active")
        frame = self._replay.frames[frame_index]

        def build(global_slot: int) -> SharedObsSourceMaterialProjectionV1:
            return build_shared_obs_authority_source_material_projection_v1(
                self._context,
                frame,
                selected_global_slot=global_slot,
            )

        recipient = build(recipient_global_slot)
        contributors = tuple(
            build(global_slot)
            for global_slot in self._active_slots
            if global_slot != recipient_global_slot
        )
        return recipient, contributors

    def _shared_timeline(
        self,
        global_slot: int,
        *,
        cache: dict[int, SharedObsAgentPovReplayTimelineV1],
    ) -> SharedObsAgentPovReplayTimelineV1:
        cached = cache.get(global_slot)
        if cached is not None:
            return cached
        roster = self._context.roster[global_slot]
        summary = self._shared_artifact_summary(roster.public_agent_id)
        endpoint = _endpoint_kind(self._shared_completion)
        final = len(self._replay.frames) - 1
        prefix = (
            f"{self._context.identity.episode_id}:shared-obs-visual-union:"
            f"{roster.public_agent_id}"
        )
        rows = tuple(
            SharedObsAgentPovReplayTimelineRowV1(
                frame_index=index,
                recipient_frame_id=f"{prefix}:frame:{index}",
                simulator_step_count=frame.simulator_step_count,
                incoming_recipient_transition_id=(
                    None if index == 0 else f"{prefix}:transition:{index - 1}"
                ),
                endpoint_kind=endpoint if index == final else "none",
            )
            for index, frame in enumerate(self._replay.frames)
        )
        timeline = SharedObsAgentPovReplayTimelineV1(
            schema_version=1,
            timeline_kind="shared_obs_agent_pov",
            timeline_id=f"{prefix}:timeline",
            artifact_summary=summary,
            final_frame_index=final,
            completion=self._shared_completion,
            rows=rows,
        )
        cache[global_slot] = timeline
        return timeline

    def _shared_artifact_summary(
        self,
        public_agent_id: str,
    ) -> SharedObsAgentPovReplayArtifactSummaryV1:
        episode_id = self._context.identity.episode_id
        return SharedObsAgentPovReplayArtifactSummaryV1(
            schema_version=1,
            recipient_replay_id=(
                f"{episode_id}:shared-obs-visual-union:{public_agent_id}:replay"
            ),
            episode_id=episode_id,
            public_agent_id=public_agent_id,
            expected_transition_count=self._replay.header.expected_transition_count,
            captured_transition_count=len(self._replay.transitions),
            captured_frame_count=len(self._replay.frames),
        )

    def _transition_view(self, frame_index: int) -> EvaluationTransitionViewV1 | None:
        if frame_index == 0:
            return None
        return EvaluationTransitionViewV1(
            context=self._context,
            start_frame=self._replay.frames[frame_index - 1],
            transition=self._replay.transitions[frame_index - 1],
            successor_frame=self._replay.frames[frame_index],
        )

    def _require_active_or_none(self, value: int | None, *, name: str) -> None:
        if value is None:
            return
        if type(value) is not int or value not in self._active_slots:
            raise ValueError(f"{name} must name a configured-active replay actor")

    def _error_result(
        self,
        outcome: ReplayServiceOutcomeV1,
        error_code: Literal[
            "invalid_cursor",
            "audience_unavailable",
            "stale_revision",
            "command_id_conflict",
            "server_shutting_down",
            "internal_error",
        ],
        message: str,
    ) -> ReplayServiceCommandResultV1:
        return ReplayServiceCommandResultV1(
            outcome=outcome,
            payload=ReplayApiErrorV1(
                error_code=error_code,
                message=message,
                latest_frame=self._frame,
            ),
        )

    def _record_error(
        self,
        key: tuple[str, str],
        fingerprint: str,
        outcome: ReplayServiceOutcomeV1,
        error_code: Literal[
            "invalid_cursor",
            "audience_unavailable",
            "stale_revision",
            "command_id_conflict",
            "server_shutting_down",
            "internal_error",
        ],
        message: str,
    ) -> ReplayServiceCommandResultV1:
        result = self._error_result(outcome, error_code, message)
        records = self._command_records.copy()
        self._remember_in(
            records,
            key,
            fingerprint,
            False,
        )
        self._command_records = records
        return result

    @staticmethod
    def _remember_in(
        records: OrderedDict[tuple[str, str], _CommandRecord],
        key: tuple[str, str],
        fingerprint: str,
        shutdown_requested: bool,
    ) -> None:
        records[key] = _CommandRecord(
            fingerprint=fingerprint,
            shutdown_requested=shutdown_requested,
        )
        records.move_to_end(key)
        while len(records) > _COMMAND_RECORD_LIMIT:
            records.popitem(last=False)


__all__ = [
    "PresentationResourceResultV1",
    "ReplayServiceCommandResultV1",
    "ReplayServiceOutcomeV1",
    "ReplayViewerService",
]
