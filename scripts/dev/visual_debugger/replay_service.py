"""Serve a loaded replay without changing its captured trajectory.

``ReplayViewerService`` owns the current cursor and display choices. It
serializes commands with a lock, rejects stale revisions, and caches recent
command identities. Researcher analysis and actor-specific projections are
built lazily and reused. The HTTP layer owns authentication and file responses;
this service returns checked models or encoded bytes and runs no simulator.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from secrets import token_urlsafe
from threading import RLock
from typing import TYPE_CHECKING, Literal, cast

from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_V2,
)
from marl_battlegrounds.evaluation.metric_catalog import METRIC_SCHEMA_VERSION
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV1,
    AssignedPolicySlotV2,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovReplayContent,
    export_actor_pov_replay_v1,
    export_actor_pov_replay_v2,
)
from marl_battlegrounds.evaluation.replay import (
    ReplayArtifactReferenceV1,
    ReplayArtifactV1,
)
from marl_battlegrounds.evaluation.replay_io import (
    LoadedReplay,
    LoadedReplayBundle,
    LoadedReplayBundleV1,
    canonical_metric_report_artifact_json_bytes_v1,
)
from marl_battlegrounds.evaluation.replay_v2 import (
    ReplayArtifactReferenceV2,
    ReplayArtifactV2,
    replay_reference_v2,
)
from marl_battlegrounds.evaluation.replay_v3 import (
    ReplayArtifactReferenceV3,
    ReplayArtifactV3,
    replay_reference_v3,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    EvaluationScenePresentationStateV1,
    SharedObsSourceMaterialProjection,
    build_researcher_analyzer_projection_v2,
    build_shared_obs_authority_source_material_projection_v1,
    build_status_source_evidence_index_v2,
    build_visual_event_batch_v2,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovProjectionIndexV1,
    build_actor_pov_analyzer_projection_v1,
    build_actor_pov_projection_index_v1,
)
from scripts.dev.visual_debugger.no_shared_visual import (
    build_replay_no_shared_obs_visual_content_v1,
)
from scripts.dev.visual_debugger.presentation import (
    build_replay_no_shared_obs_authorized_presentation_v1,
    build_replay_oracle_authorized_presentation_v1,
    build_replay_researcher_space_v1,
    build_replay_shared_obs_authorized_presentation_v1,
)
from scripts.dev.visual_debugger.presentation_protocol import (
    PresentationResourceResultV1,
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    ReplayResearcherSpaceV1,
    ReplaySharedObsAuthorizedPresentationFrameV1,
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
    ReplayArtifactFactsV1,
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

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.analysis import ReplayAnalysis

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
        """Require attachment bytes and a name only when the metric report is
        available.
        """
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
    """Retain a request fingerprint and shutdown result for duplicate detection."""

    fingerprint: str
    shutdown_requested: bool


@dataclass(frozen=True, slots=True)
class _PovCacheEntry:
    """Keep one actor's projected capture, timeline, index, and exact-export
    capability.
    """

    content: ActorPovReplayContent
    completion: ActorPovReplayCompletionBadgeV1
    projection_index: ActorPovProjectionIndexV1
    timeline: ActorPovReplayTimelineV1
    exact_actor_input_export_available: bool


def _safe_metric_report_filename(episode_id: str) -> str:
    """Return the bounded attachment basename derived from one episode ID."""
    if type(episode_id) is not str:
        raise TypeError("episode_id must be a string")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", episode_id).strip("._-")
    stem = stem[:96].strip("._-")
    stem = stem.replace(_METRIC_REPORT_SUFFIX, "-").strip("._-") or "replay"
    return f"{stem}{_METRIC_REPORT_SUFFIX}"


def _replay_reference(
    bundle: LoadedReplayBundle,
) -> ReplayArtifactReferenceV1 | ReplayArtifactReferenceV2 | ReplayArtifactReferenceV3:
    """Build the version-appropriate immutable replay identity and byte-length facts."""
    replay = bundle.replay
    if type(replay) is ReplayArtifactV3:
        return replay_reference_v3(replay)
    if type(replay) is ReplayArtifactV2:
        return replay_reference_v2(replay)
    return ReplayArtifactReferenceV1(
        artifact_id=replay.artifact_id,
        episode_id=replay.header.context.identity.episode_id,
        context_digest_sha256=replay.header.context_digest_sha256,
        trajectory_content_digest_sha256=replay.trajectory_content_digest_sha256,
        canonical_digest_sha256=replay.canonical_digest_sha256,
        canonical_byte_length=len(canonical_json_bytes(replay)),
    )


def _completion_badge(bundle: LoadedReplayBundle) -> ReplayCompletionBadgeV1:
    """Copy recorded completion and failure facts into the researcher badge."""
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


def _processing_badge(bundle: LoadedReplayBundle) -> ReplayProcessingBadgeV1:
    """Report legacy metric processing; newer capture-only formats report not
    requested.
    """
    if type(bundle.replay) in (ReplayArtifactV2, ReplayArtifactV3):
        return ReplayProcessingBadgeV1(
            status="not_requested", processed_transition_count=0
        )
    processing = cast(ReplayArtifactV1, bundle.replay).processing_status
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
    content: ActorPovReplayContent,
) -> ActorPovReplayCompletionBadgeV1:
    """Copy only the actor-export completion facts into a POV badge."""
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
    bundle: LoadedReplayBundle,
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
    """Map complete endpoint bases or an incomplete prefix to the timeline end
    marker.
    """
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
        bundle: LoadedReplayBundle,
        *,
        initial_frame_index: int = 0,
        view_mode: ReplayViewModeV1 = "researcher",
        reference_global_slot: int | None = None,
        selected_global_slot: int | None = None,
        armed_lane: Literal[0, 1] | None = None,
        pov_global_slot: int | None = None,
        preset: ReplayPresetV1 | Literal["technical", "debug"] = "analysis",
        show_ranges: bool = False,
        verbose: bool = False,
        viewer_session_id: str | None = None,
    ) -> None:
        """Open a viewer state around a validated in-memory replay.

        Parameters
        ----------
        bundle : LoadedReplayBundle
            Exact LoadedReplay or LoadedReplayBundleV1 root from the public loader.
        initial_frame_index : int, optional
            Captured frame index, from zero through the last retained frame. Defaults
            to zero; the final frame index equals the number of captured transitions.
        view_mode : {'researcher', 'pov'}, optional
            Initial display authority. Defaults to researcher.
        reference_global_slot : int or None, optional
            Configured-active researcher reference. None uses the first focal slot,
            if one exists. This is viewer state, not a recorded historical selection.
        selected_global_slot : int or None, optional
            Initial active inspection target. Requires a reference when supplied.
            Defaults to None.
        armed_lane : {0, 1} or None, optional
            Viewer inspection lane. Requires a selected target. Defaults to None.
        pov_global_slot : int or None, optional
            Active actor whose POV is displayed. None uses the first focal slot.
            POV mode requires a resolved actor.
        preset : ReplayPresetV1 or {'technical', 'debug'}, optional
            Accepted compatibility preset. All accepted values use analysis mode.
            Defaults to analysis.
        show_ranges : bool, optional
            Initial range-overlay preference. Defaults to False.
        verbose : bool, optional
            Accepted compatibility flag; current viewer verbosity remains False.
        viewer_session_id : str or None, optional
            Nonblank viewer identity. None creates a random identity.

        Raises
        ------
        TypeError
            The loaded root or boolean flags have unsupported exact types.
        ValueError
            Status, cursor, view, slot selection, lane, or viewer identity is invalid.

        Notes
        -----
        Construction builds the researcher timeline and status-evidence index. Actor
        projections are cached as needed. It performs no replay write or simulation.
        Initial viewer choices are not reconstructed from each historical frame.
        """
        if type(bundle) not in (LoadedReplayBundleV1, LoadedReplay):
            raise TypeError("bundle must be an exact loaded replay root")
        if bundle.status not in (
            "complete",
            "metric_report_missing",
            "not_recorded",
        ) or (bundle.metric_report_artifact is None) != (bundle.status != "complete"):
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
        self._metric_analysis: ReplayAnalysis | None = None
        self._metric_analysis_lock = RLock()
        self._metric_catalog_bytes: bytes | None = None
        self._replay = bundle.replay
        self._context = bundle.replay.header.context
        self._active_slots: tuple[int, ...] = tuple(
            row.global_slot for row in self._context.roster if row.configured_active
        )
        focal_slots: tuple[int, ...] = tuple(
            row.global_slot
            for row in self._context.policy_assignments
            if isinstance(row, (AssignedPolicySlotV1, AssignedPolicySlotV2))
            and row.evaluation_role == "focal"
        )
        default_slot: int | None = min(focal_slots) if focal_slots else None
        researcher_slot: int | None = (
            default_slot if reference_global_slot is None else reference_global_slot
        )
        inspection_slot: int | None = selected_global_slot
        actor_slot: int | None = (
            default_slot if pov_global_slot is None else pov_global_slot
        )
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
                "not_recorded"
                if type(bundle.replay) in (ReplayArtifactV2, ReplayArtifactV3)
                else "available"
                if bundle.metric_report_artifact is not None
                else "missing"
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
        self._artifact_facts = ReplayArtifactFactsV1(
            artifact_summary=self._artifact_summary,
            completion=self._completion,
            processing=self._processing,
        )
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
        self._reference_global_slot: int | None = researcher_slot
        self._inspection_global_slot: int | None = inspection_slot
        self._armed_lane = armed_lane
        self._pov_global_slot: int | None = actor_slot
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
    def show_ranges(self) -> bool:
        """Return the viewer preference without exposing an actor observation."""
        with self._lock:
            return self._show_ranges

    @property
    def revision(self) -> int:
        """Return the committed viewer revision under the service lock."""
        with self._lock:
            return self._revision

    @property
    def command_cache_size(self) -> int:
        """Return how many recent command identities are cached, at most 256."""
        with self._lock:
            return len(self._command_records)

    @property
    def pov_index_build_count(self) -> int:
        """Return the number of cached actor projection indexes."""
        with self._lock:
            return len(self._pov_cache)

    @property
    def shared_timeline_build_count(self) -> int:
        """Return the number of cached SharedObs recipient timelines."""
        with self._lock:
            return len(self._shared_timeline_cache)

    @property
    def shutting_down(self) -> bool:
        """Return whether an accepted exit command fenced new commands."""
        with self._lock:
            return self._shutting_down

    @property
    def faulted(self) -> bool:
        """Return whether an internal failure fenced new commands."""
        with self._lock:
            return self._faulted

    def current_frame(self) -> ReplayViewerFrameV1:
        """Read the current immutable viewer frame under the service lock.

        Returns
        -------
        ReplayViewerFrameV1
            The already built frame for the committed cursor and display choices.
        """
        with self._lock:
            return self._frame

    def current_metric_report(self) -> ReplayMetricReportResultV1:
        """Read the recorded metric attachment independently of battlefield view.

        Returns
        -------
        ReplayMetricReportResultV1
            Canonical JSON bytes and a safe basename when a metric artifact exists;
            otherwise a missing result with no bytes or filename.

        Notes
        -----
        This serializes the recorded artifact under the service lock. It does not
        compute a new analysis or write a file. Actor battlefield view does not hide
        researcher analysis resources from the researcher using the viewer.
        """
        with self._lock:
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

    def episode_details(self) -> tuple[bytes, str]:
        """Encode recorded identity and configuration without trajectory arrays.

        Returns
        -------
        tuple of bytes and str
            Canonical JSON content and the filename episode-details.json. Content
            includes context, completion, runtime, wrappers, and captured count.

        Notes
        -----
        This creates an in-memory attachment; the HTTP caller handles delivery.
        """
        replay = self._replay
        return canonical_json_bytes(
            {
                "schema_id": "marlbg.replay.episode_details",
                "schema_version": 1,
                "source_replay": self._artifact_summary.replay_reference,
                "context": self._context,
                "completion": replay.completion,
                "runtime_provenance": replay.header.runtime_provenance,
                "wrapper_stack": replay.header.wrapper_stack,
                "captured_transition_count": len(replay.transitions),
            }
        ), "episode-details.json"

    def metric_analysis(
        self, frame_index: int, scope: str, format_: str
    ) -> tuple[bytes, str | None]:
        """Encode researcher metrics at a cursor or at the final captured frame.

        Parameters
        ----------
        frame_index : int
            Valid captured frame index, checked even for final scope.
        scope : {'cursor', 'final'}
            Whether to summarize the selected frame or all captured transitions.
        format_ : {'json', 'csv'}
            Output encoding. JSON includes applicable statistics; CSV keeps its
            complete fixed column schema.

        Returns
        -------
        tuple of bytes and str or None
            Encoded content and an attachment filename for CSV; JSON returns None
            as the filename.

        Raises
        ------
        ValueError
            Scope or format is unsupported.
        IndexError
            The requested frame is outside the captured replay.

        Notes
        -----
        The first request computes and caches full analysis. Subsequent requests
        reuse it. This creates no file and changes no replay or cursor state.
        """
        if scope not in ("cursor", "final") or format_ not in ("json", "csv"):
            raise ValueError("unsupported metric scope or format")
        self._bundle.frame_at(frame_index)
        analysis = self._analyzed_metrics()
        if format_ == "json":
            summary = analysis.summary(frame_index, scope=scope)
            statistics = cast(list[dict[str, object]], summary["statistics"])
            # The CSV keeps its fixed schema. Transport only rows this roster can
            # display; undefined ratios and reachable zeros remain available.
            summary["statistics"] = [row for row in statistics if row["applicable"]]
            return canonical_json_bytes(summary), None
        selected = analysis.frame_count - 1 if scope == "final" else frame_index
        episode = _safe_metric_report_filename(
            self._context.identity.episode_id
        ).removesuffix(_METRIC_REPORT_SUFFIX)
        return (
            analysis.csv(frame_index, scope=scope).encode("utf-8"),
            f"tdm-metrics__episode-{episode}__schema-{METRIC_SCHEMA_VERSION}"
            f"__{scope}__frame-{selected}.csv",
        )

    def _analyzed_metrics(self) -> ReplayAnalysis:
        """Share one immutable analysis across values, downloads and search."""
        with self._metric_analysis_lock:
            if self._metric_analysis is None:
                from marl_battlegrounds.evaluation.analysis import analyze_replay

                self._metric_analysis = analyze_replay(self._bundle, full=True)
            return self._metric_analysis

    def metric_catalog(self) -> bytes:
        """Encode and cache measurement descriptions for this replay.

        Returns
        -------
        bytes
            Canonical JSON catalog shared across later requests.

        Notes
        -----
        The first call may build full replay analysis; later calls reuse both the
        analysis and encoded catalog under a separate analysis lock.
        """
        with self._metric_analysis_lock:
            if self._metric_catalog_bytes is None:
                self._metric_catalog_bytes = canonical_json_bytes(
                    self._analyzed_metrics().catalog()
                )
            return self._metric_catalog_bytes

    def current_presentation(self) -> PresentationResourceResultV1:
        """Build a checked display resource from the locked viewer snapshot.

        Returns
        -------
        PresentationResourceResultV1
            Researcher, NoSharedObs, or SharedObs presentation for the current cursor.

        Raises
        ------
        RuntimeError
            The committed frame, selected actor, or source authority is inconsistent.
        ValueError
            An epoch, model, or authorized display contract fails validation.

        Notes
        -----
        This may populate actor projection caches. It does not advance a simulator,
        change the cursor, or expand an actor's allowed information.
        """
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
                        global_context=self._context,
                        current_global_frame=self._replay.frames[
                            raw_frame.cursor.frame_index
                        ],
                        previous_global_frame=(
                            None
                            if raw_frame.cursor.frame_index == 0
                            else self._replay.frames[raw_frame.cursor.frame_index - 1]
                        ),
                        public_catalog=self._context.static_mechanics_catalog,
                        source_authority_epoch=raw_frame.revision,
                        incoming_visual_events=(
                            None
                            if raw_frame.cursor.frame_index == 0
                            else build_visual_event_batch_v2(
                                cast(
                                    EvaluationTransitionViewV1,
                                    self._transition_view(raw_frame.cursor.frame_index),
                                )
                            )
                        ),
                        researcher_space=self._researcher_space_for_frame(
                            frame_index=raw_frame.cursor.frame_index,
                            selected_global_slot=self._pov_global_slot,
                        ),
                        exact_actor_input_export_available=(
                            entry.exact_actor_input_export_available
                        ),
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
                        SharedObsSourceMaterialProjection, ...
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
                    global_context=self._context,
                    current_global_frame=self._replay.frames[frame_index],
                    previous_global_frame=(
                        None
                        if frame_index == 0
                        else self._replay.frames[frame_index - 1]
                    ),
                    public_catalog=self._context.static_mechanics_catalog,
                    source_authority_epoch=raw_frame.revision,
                    authorized_recipient_global_slot=self._pov_global_slot,
                    current_recipient_source_material=current_recipient,
                    current_active_nonrecipient_source_material=(current_nonrecipient),
                    previous_recipient_source_material=previous_recipient,
                    previous_active_nonrecipient_source_material=(
                        previous_nonrecipient
                    ),
                    incoming_visual_events=(
                        None
                        if frame_index == 0
                        else build_visual_event_batch_v2(
                            cast(
                                EvaluationTransitionViewV1,
                                self._transition_view(frame_index),
                            )
                        )
                    ),
                    incoming_transition=incoming_transition,
                    outgoing_transition=outgoing_transition,
                    researcher_space=self._researcher_space_for_frame(
                        frame_index=frame_index,
                        selected_global_slot=self._pov_global_slot,
                    ),
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
                None
                if source_frame_index == len(self._replay.transitions)
                else self._replay.transitions[source_frame_index]
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
        if not oracle:
            if type(raw_frame) not in (
                ActorPovReplayViewerFrameV1,
                SharedObsAgentPovReplayViewerFrameV1,
            ):
                raise RuntimeError(
                    "committed Agent replay artifact facts use the wrong root"
                )
            agent_raw_frame = cast(
                ActorPovReplayViewerFrameV1 | SharedObsAgentPovReplayViewerFrameV1,
                raw_frame,
            )
            if (
                type(agent_raw_frame.artifact_facts) is not ReplayArtifactFactsV1
                or agent_raw_frame.artifact_facts != self._artifact_facts
            ):
                raise RuntimeError(
                    "committed Agent replay artifact facts do not join service state"
                )
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
                or raw_frame.artifact_facts != self._artifact_facts
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
        """Read the timeline allowed for the current view and actor.

        Returns
        -------
        ReplayTimelineV1
            Researcher timeline or the selected actor's cached timeline. SharedObs
            timelines use recipient-local identities and completion facts.

        Raises
        ------
        RuntimeError
            POV mode has no selected actor.

        Notes
        -----
        The service lock covers any first-time actor timeline construction.
        """
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
        """Apply a checked viewer command without changing recorded replay data.

        Parameters
        ----------
        request : ReplayCommandRequestV1
            Exact validated root with client ID, command ID, base revision, and a
            supported cursor or display command.

        Returns
        -------
        ReplayServiceCommandResultV1
            New frame, duplicate response, or typed rejection. Exit requests are
            returned for the HTTP host to act on after writing the response.

        Raises
        ------
        TypeError
            The request is not the exact supported model.

        Notes
        -----
        The lock covers validation and installation. The latest 256 command IDs are
        remembered; exact repeats while cached do not apply twice. Stale revisions
        and reused IDs with different contents are rejected. Projection caches may
        be built, but no simulator step or replay write occurs. Unexpected internal
        failures can fence further commands and propagate.
        """
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
                requested_selection: int | None = command.selected_global_slot
                try:
                    self._require_active_or_none(
                        requested_selection,
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
                selection_changed = requested_selection != inspection_global_slot
                pov_changed = (
                    requested_selection is not None
                    and requested_selection != pov_global_slot
                )
                if selection_changed or pov_changed:
                    inspection_global_slot = requested_selection
                    if requested_selection is not None:
                        pov_global_slot = requested_selection
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
                    if command.view_mode == "pov":
                        if (
                            inspection_global_slot is not None
                            and inspection_global_slot != pov_global_slot
                        ):
                            pov_global_slot = inspection_global_slot
                    elif inspection_global_slot != pov_global_slot:
                        inspection_global_slot = pov_global_slot
                        armed_lane = None
                    view_mode = command.view_mode
                    changed = True
            elif type(command) is ReplaySetPovActorCommandV1:
                requested_slot: int | None = command.global_slot
                try:
                    if command.presentation_key is not None:
                        if view_mode != "pov":
                            raise ValueError(
                                "presentation-key actor switching requires POV"
                            )
                        requested_slot = self._visible_pov_actor_slot(
                            command.presentation_key
                        )
                    self._require_active_or_none(
                        requested_slot,
                        name="pov_global_slot",
                    )
                    if requested_slot is None:  # pragma: no cover - model invariant.
                        raise ValueError("POV actor reference is absent")
                except ValueError:
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "audience_unavailable",
                        "audience_unavailable",
                        "The requested POV actor is unavailable.",
                    )
                if (
                    requested_slot != pov_global_slot
                    or requested_slot != inspection_global_slot
                ):
                    pov_global_slot = requested_slot
                    inspection_global_slot = requested_slot
                    armed_lane = None
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
        """Build and validate one proposed cursor/view frame using caller-owned
        caches.
        """
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
            pov_frame = entry.content.frames[frame_index]
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
                completion=entry.completion,
                processing_disclosure=ActorPovProcessingDisclosureV1(),
                artifact_facts=self._artifact_facts,
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
            artifact_facts=self._artifact_facts,
        )

    def _build_researcher_timeline(self) -> ResearcherReplayTimelineV1:
        """Build ordered frame and incoming-event rows with one final endpoint
        marker.
        """
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
        """Get or build one actor's immutable visual projection and timeline.

        Populate the supplied cache only after construction. Preserve whether the
        replay version supports exact actor-input export or only a visual projection.
        """
        cached = cache.get(global_slot)
        if cached is not None:
            return cached
        if type(self._replay) is ReplayArtifactV3:
            content = export_actor_pov_replay_v2(
                self._replay, global_slot=global_slot
            ).content
            exact_actor_input_export_available = True
        elif (
            type(self._replay) is ReplayArtifactV2
            or self._context.actor_projection == NO_SHARED_OBS_ACTOR_PROJECTION_V2
        ):
            content = build_replay_no_shared_obs_visual_content_v1(
                self._replay,
                global_slot=global_slot,
            )
            exact_actor_input_export_available = False
        else:
            artifact = export_actor_pov_replay_v1(
                cast(ReplayArtifactV1, self._replay),
                global_slot=global_slot,
            )
            content = artifact.content
            exact_actor_input_export_available = True
        projection_index = build_actor_pov_projection_index_v1(content)
        completion = _pov_completion_badge(content)
        endpoint = _endpoint_kind(completion)
        final = len(content.frames) - 1
        rows = tuple(
            ActorPovReplayTimelineRowV1(
                frame_index=index,
                pov_frame_id=frame.pov_frame_id,
                simulator_step_count=frame.simulator_step_count,
                incoming_pov_transition_id=(
                    None
                    if index == 0
                    else content.transitions[index - 1].pov_transition_id
                ),
                incoming_cue_count=(
                    0 if index == 0 else len(content.transitions[index - 1].cues)
                ),
                endpoint_kind=endpoint if index == final else "none",
            )
            for index, frame in enumerate(content.frames)
        )
        timeline = ActorPovReplayTimelineV1(
            timeline_id=(
                f"{self._replay.artifact_id}:timeline:actor-pov:"
                f"{content.public_agent_id}"
            ),
            artifact_summary=self._pov_artifact_summary,
            final_frame_index=final,
            pov_global_slot=global_slot,
            public_agent_id=content.public_agent_id,
            completion=completion,
            rows=rows,
        )
        entry = _PovCacheEntry(
            content=content,
            completion=completion,
            projection_index=projection_index,
            timeline=timeline,
            exact_actor_input_export_available=(exact_actor_input_export_available),
        )
        cache[global_slot] = entry
        return entry

    def _shared_authority_sources(
        self,
        *,
        frame_index: int,
        recipient_global_slot: int,
    ) -> tuple[
        SharedObsSourceMaterialProjection,
        tuple[SharedObsSourceMaterialProjection, ...],
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

        def build(global_slot: int) -> SharedObsSourceMaterialProjection:
            """Project one active slot's source material at the already selected
            frame.
            """
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
        """Get or cache a recipient-local SharedObs timeline with no global event
        counts.
        """
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
        """Build recipient-local replay identity and capture counts without global
        hashes.
        """
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
        """Join the selected frame to its incoming transition; frame zero returns
        None.
        """
        if frame_index == 0:
            return None
        return EvaluationTransitionViewV1(
            context=self._context,
            start_frame=self._replay.frames[frame_index - 1],
            transition=self._replay.transitions[frame_index - 1],
            successor_frame=self._replay.frames[frame_index],
        )

    def _researcher_space_for_frame(
        self,
        *,
        frame_index: int,
        selected_global_slot: int,
    ) -> ReplayResearcherSpaceV1:
        """Build geometry-free global researcher facts for one selected frame and
        agent.
        """
        incoming_view = self._transition_view(frame_index)
        projection = build_researcher_analyzer_projection_v2(
            self._context,
            self._replay.frames[frame_index],
            transition_view=incoming_view,
            presentation=EvaluationScenePresentationStateV1(
                controlled_global_slot=self._reference_global_slot,
                selected_global_slot=selected_global_slot,
                armed_lane=None,
                show_ranges=self._show_ranges,
            ),
            status_source_evidence_state=self._status_index.state_for_frame(
                frame_index
            ),
        )
        return build_replay_researcher_space_v1(
            self._context,
            projection.scene,
            authority_session_id=self._viewer_session_id,
            final_frame_index=self._artifact_summary.recorded_transition_count,
            selected_global_slot=selected_global_slot,
            incoming_transition=(
                None if incoming_view is None else incoming_view.transition
            ),
            outgoing_transition=(
                None
                if frame_index == len(self._replay.transitions)
                else self._replay.transitions[frame_index]
            ),
        )

    def _require_active_or_none(self, value: int | None, *, name: str) -> None:
        """Allow None or an exact integer naming a configured-active roster slot."""
        if value is None:
            return
        if type(value) is not int or value not in self._active_slots:
            raise ValueError(f"{name} must name a configured-active replay actor")

    def _visible_pov_actor_slot(self, presentation_key: str) -> int:
        """Resolve one current POV-scene key without exposing slot topology."""
        result = self.current_presentation()
        presentation = result.payload
        if result.outcome != "response" or type(presentation) not in (
            ReplayNoSharedObsAuthorizedPresentationFrameV1,
            ReplaySharedObsAuthorizedPresentationFrameV1,
        ):
            raise ValueError("current POV presentation is unavailable")
        agent_presentation = cast(
            ReplayNoSharedObsAuthorizedPresentationFrameV1
            | ReplaySharedObsAuthorizedPresentationFrameV1,
            presentation,
        )
        matching_agents = tuple(
            agent
            for agent in agent_presentation.current_endpoint.parts.scene.agents
            if agent.presentation_key == presentation_key
        )
        if len(matching_agents) != 1:
            raise ValueError("POV actor key is not visible in current authority")
        public_agent_id = matching_agents[0].public_agent_id
        matching_slots = tuple(
            row.global_slot
            for row in self._context.roster
            if row.configured_active and row.public_agent_id == public_agent_id
        )
        if len(matching_slots) != 1 or matching_slots[0] not in self._active_slots:
            raise ValueError("POV actor key does not join one active roster row")
        return matching_slots[0]

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
        """Build a typed rejection carrying the latest committed viewer frame."""
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
        """Build a rejection and retain its request identity in the bounded cache."""
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
        """Add a request record and evict oldest entries beyond the 256-record limit."""
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
