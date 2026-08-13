"""Locked read-only service for one validated semantic replay bundle."""

from __future__ import annotations

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
from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundleV1
from marl_battlegrounds.rendering.evaluation_adapter import (
    EvaluationScenePresentationStateV1,
    build_researcher_analyzer_projection_v2,
    build_shared_obs_source_material_projection_v1,
    build_status_source_evidence_index_v2,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovProjectionIndexV1,
    build_actor_pov_analyzer_projection_v1,
    build_actor_pov_projection_index_v1,
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
    SharedObsSourceMaterialReplayTimelineRowV1,
    SharedObsSourceMaterialReplayTimelineV1,
    SharedObsSourceMaterialReplayViewerFrameV1,
)

_COMMAND_RECORD_LIMIT = 256

type ReplayServiceOutcomeV1 = Literal[
    "response",
    "invalid_cursor",
    "audience_unavailable",
    "stale_revision",
    "command_id_conflict",
    "server_shutting_down",
    "service_faulted",
]


@dataclass(frozen=True, slots=True)
class ReplayServiceCommandResultV1:
    """One transport-neutral replay command result."""

    outcome: ReplayServiceOutcomeV1
    payload: ReplayCommandResponseV1 | ReplayApiErrorV1
    shutdown_requested: bool = False


@dataclass(frozen=True, slots=True)
class _CommandRecord:
    fingerprint: str
    shutdown_requested: bool


@dataclass(frozen=True, slots=True)
class _PovCacheEntry:
    artifact: ActorPovReplayArtifactV1
    projection_index: ActorPovProjectionIndexV1
    timeline: ActorPovReplayTimelineV1


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
        preset: ReplayPresetV1 = "analysis",
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
        if preset not in ("presentation", "analysis", "debug"):
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
        reference_slot = selected_global_slot
        actor_slot = default_slot if pov_global_slot is None else pov_global_slot
        self._require_active_or_none(
            researcher_slot,
            name="reference_global_slot",
        )
        self._require_active_or_none(reference_slot, name="selected_global_slot")
        self._require_active_or_none(actor_slot, name="pov_global_slot")
        if armed_lane is not None and (
            type(armed_lane) is not int or armed_lane not in (0, 1)
        ):
            raise ValueError("armed_lane must be the Python int zero or one, or None")
        if reference_slot is not None and researcher_slot is None:
            raise ValueError("selected_global_slot requires a researcher reference")
        if armed_lane is not None and reference_slot is None:
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
        self._processing = _processing_badge(bundle)
        self._status_index = build_status_source_evidence_index_v2(
            self._context,
            self._replay.frames,
            self._replay.transitions,
        )
        self._researcher_timeline = self._build_researcher_timeline()
        self._pov_cache: dict[int, _PovCacheEntry] = {}
        self._shared_timeline_cache: dict[
            int, SharedObsSourceMaterialReplayTimelineV1
        ] = {}

        self._viewer_session_id = (
            token_urlsafe(24) if viewer_session_id is None else viewer_session_id
        )
        self._revision = 0
        self._frame_index = initial_frame_index
        self._cursor_generation = 0
        self._choreography_generation = 0
        self._view_mode: ReplayViewModeV1 = view_mode
        self._reference_global_slot = researcher_slot
        self._selected_global_slot = reference_slot
        self._armed_lane = armed_lane
        self._pov_global_slot = actor_slot
        self._preset: ReplayPresetV1 = preset
        self._show_ranges = show_ranges
        self._verbose = verbose
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
            selected_global_slot=self._selected_global_slot,
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
            selected_global_slot = self._selected_global_slot
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
                if command.selected_global_slot != selected_global_slot:
                    selected_global_slot = command.selected_global_slot
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
                if command.preset != preset:
                    preset = command.preset
                    changed = True
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
                if command.verbose != verbose:
                    verbose = command.verbose
                    changed = True
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
                        selected_global_slot=selected_global_slot,
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
            self._selected_global_slot = selected_global_slot
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
        selected_global_slot: int | None,
        armed_lane: Literal[0, 1] | None,
        pov_global_slot: int | None,
        preset: ReplayPresetV1,
        show_ranges: bool,
        verbose: bool,
        pov_cache: dict[int, _PovCacheEntry],
        shared_timeline_cache: dict[
            int,
            SharedObsSourceMaterialReplayTimelineV1,
        ],
    ) -> ReplayViewerFrameV1:
        cursor = ReplayCursorV1(
            frame_index=frame_index,
            final_frame_index=self._artifact_summary.recorded_transition_count,
            cursor_generation=cursor_generation,
            choreography_generation=choreography_generation,
        )
        frame = self._replay.frames[frame_index]
        incoming = self._transition_view(frame_index)
        if view_mode == "researcher":
            projection = build_researcher_analyzer_projection_v2(
                self._context,
                frame,
                transition_view=incoming,
                presentation=EvaluationScenePresentationStateV1(
                    controlled_global_slot=self._reference_global_slot,
                    selected_global_slot=selected_global_slot,
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
                verbose=verbose,
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
                projection=projection,
            )
        if pov_global_slot is None:
            raise ValueError("POV replay view requires a configured-active actor")
        roster = self._context.roster[pov_global_slot]
        if self._context.execution_information_mode == "no_shared_obs":
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
                verbose=verbose,
                pov_global_slot=pov_global_slot,
                public_agent_id=roster.public_agent_id,
                pov_frame_id=pov_frame.pov_frame_id,
                simulator_step_count=pov_frame.simulator_step_count,
                incoming_pov_transition_id=projection.incoming_transition_id,
                completion=_pov_completion_badge(entry.artifact),
                processing_disclosure=ActorPovProcessingDisclosureV1(),
                projection=projection,
            )
        projection = build_shared_obs_source_material_projection_v1(
            self._context,
            frame,
            selected_global_slot=pov_global_slot,
            transition_view=incoming,
        )
        timeline = self._shared_timeline(
            pov_global_slot,
            cache=shared_timeline_cache,
        )
        return SharedObsSourceMaterialReplayViewerFrameV1(
            viewer_session_id=self._viewer_session_id,
            revision=revision,
            artifact_summary=self._artifact_summary,
            timeline_id=timeline.timeline_id,
            cursor=cursor,
            preset=preset,
            verbose=verbose,
            selected_global_slot=pov_global_slot,
            public_agent_id=roster.public_agent_id,
            source_material_frame_id=projection.base_sensor_frame.source_material_frame_id,
            source_frame_id=frame.frame_id,
            simulator_step_count=frame.simulator_step_count,
            incoming_transition_id=projection.incoming_transition_id,
            completion=self._completion,
            processing=self._processing,
            projection=projection,
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

    def _shared_timeline(
        self,
        global_slot: int,
        *,
        cache: dict[int, SharedObsSourceMaterialReplayTimelineV1],
    ) -> SharedObsSourceMaterialReplayTimelineV1:
        cached = cache.get(global_slot)
        if cached is not None:
            return cached
        roster = self._context.roster[global_slot]
        endpoint = _endpoint_kind(self._completion)
        final = len(self._replay.frames) - 1
        rows = tuple(
            SharedObsSourceMaterialReplayTimelineRowV1(
                frame_index=index,
                source_material_frame_id=(
                    f"{self._context.identity.episode_id}:shared-obs-source-material:"
                    f"{roster.public_agent_id}:frame:{index}"
                ),
                simulator_step_count=frame.simulator_step_count,
                incoming_transition_id=(
                    None
                    if index == 0
                    else self._replay.transitions[index - 1].transition_id
                ),
                endpoint_kind=endpoint if index == final else "none",
            )
            for index, frame in enumerate(self._replay.frames)
        )
        timeline = SharedObsSourceMaterialReplayTimelineV1(
            timeline_id=(
                f"{self._replay.artifact_id}:timeline:shared-obs-source-material:"
                f"{roster.public_agent_id}"
            ),
            artifact_summary=self._artifact_summary,
            final_frame_index=final,
            selected_global_slot=global_slot,
            public_agent_id=roster.public_agent_id,
            completion=self._completion,
            rows=rows,
        )
        cache[global_slot] = timeline
        return timeline

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
    "ReplayServiceCommandResultV1",
    "ReplayServiceOutcomeV1",
    "ReplayViewerService",
]
