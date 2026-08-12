"""Strict core-free wire contracts for the read-only replay viewer."""

from __future__ import annotations

from typing import Annotated, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from marl_battlegrounds.evaluation.replay import ReplayArtifactReferenceV1
from marl_battlegrounds.rendering.evaluation_adapter import (
    SharedObsSourceMaterialProjectionV1,
)
from marl_battlegrounds.rendering.pov_scene import ActorPovAnalyzerProjectionV1
from marl_battlegrounds.rendering.scene import ResearcherAnalyzerProjectionV2

REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION = 1
ACTOR_POV_PROCESSING_DISCLOSURE_V1 = "not_available_in_actor_pov"
ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1 = "not_available_in_actor_pov"

type ReplayViewModeV1 = Literal["researcher", "pov"]
type ReplayPresetV1 = Literal["presentation", "analysis", "debug"]
type ReplayCommandResultV1 = Literal[
    "applied",
    "duplicate",
    "no_op",
    "shutdown_scheduled",
]
type ReplayApiErrorCodeV1 = Literal[
    "invalid_request",
    "invalid_cursor",
    "audience_unavailable",
    "unauthorized",
    "forbidden_origin",
    "not_found",
    "method_not_allowed",
    "payload_too_large",
    "unsupported_media_type",
    "stale_revision",
    "command_id_conflict",
    "server_shutting_down",
    "internal_error",
]
type ReplayCompletionStateV1 = Literal[
    "complete",
    "partial",
    "interrupted",
    "failed",
]
type ReplayCompletionBasisV1 = Literal["task_terminal", "declared_horizon"]
type ReplayRolloutFailureOriginV1 = Literal[
    "simulation",
    "policy",
    "validation",
    "capture",
]
type ReplayProcessingStateV1 = Literal["succeeded", "failed"]
type ReplayProcessingFailureStageV1 = Literal[
    "initial_validation",
    "reducer_initialize",
    "transition_validation",
    "reducer_advance",
    "completion_validation",
    "reducer_finalize",
    "statistic_materialization",
    "report_validation",
    "lifecycle",
]
type ReplayTimelineEndpointKindV1 = Literal[
    "none",
    "task_terminal",
    "declared_horizon",
    "task_terminal_and_declared_horizon",
    "captured_prefix",
]

_OpaqueId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
]
_ScientificId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=512,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    ),
]
_PublicAgentId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    ),
]
_Message = Annotated[str, StringConstraints(min_length=1, max_length=2048)]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(gt=0)]
_GlobalSlot = Annotated[int, Field(ge=0, lt=10)]


class _ReplayProtocolModel(BaseModel):
    """Strict immutable base for values crossing the replay HTTP boundary."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class ReplayArtifactSummaryV1(_ReplayProtocolModel):
    """Path-free replay provenance and bounded captured-prefix counts."""

    schema_version: Literal[1] = REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION
    replay_reference: ReplayArtifactReferenceV1
    expected_transition_count: _PositiveInt
    recorded_transition_count: _NonNegativeInt
    recorded_frame_count: _PositiveInt
    metric_report_availability: Literal[
        "available",
        "missing",
        "not_available_in_actor_pov",
    ]

    @model_validator(mode="after")
    def _validate_summary(self) -> Self:
        if type(self.replay_reference) is not ReplayArtifactReferenceV1:
            raise ValueError("replay_reference must be its exact V1 root.")
        if self.recorded_frame_count != self.recorded_transition_count + 1:
            raise ValueError("replay summary requires exact T+1/T counts.")
        if self.recorded_transition_count > self.expected_transition_count:
            raise ValueError("recorded transitions cannot exceed the expected horizon.")
        return self


def _validate_researcher_artifact_summary(summary: ReplayArtifactSummaryV1) -> None:
    if summary.metric_report_availability == ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1:
        raise ValueError(
            "researcher/source-material summary cannot use actor-POV non-disclosure."
        )


def _validate_actor_pov_artifact_summary(summary: ReplayArtifactSummaryV1) -> None:
    if summary.metric_report_availability != ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1:
        raise ValueError(
            "actor POV summary must not disclose metric-report availability."
        )


class ReplayCompletionBadgeV1(_ReplayProtocolModel):
    """Researcher-safe physical rollout completion, separate from processing."""

    schema_version: Literal[1] = REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION
    episode_id: _ScientificId
    completion_state: ReplayCompletionStateV1
    expected_transition_count: _PositiveInt
    validated_transition_count: _NonNegativeInt
    last_valid_frame_index: _NonNegativeInt
    last_valid_frame_id: _ScientificId
    terminated: bool
    truncated: bool
    completion_bases: tuple[ReplayCompletionBasisV1, ...]
    end_or_failure_reason: _Message | None = None
    failure_origin: ReplayRolloutFailureOriginV1 | None = None

    @model_validator(mode="after")
    def _validate_completion(self) -> Self:
        if self.validated_transition_count > self.expected_transition_count:
            raise ValueError(
                "validated transitions cannot exceed the expected horizon."
            )
        if self.last_valid_frame_index != self.validated_transition_count:
            raise ValueError("last valid frame index must equal validated progress.")
        if self.last_valid_frame_id != (
            f"{self.episode_id}:frame:{self.validated_transition_count}"
        ):
            raise ValueError("last valid frame ID is not canonical.")
        if self.validated_transition_count == 0 and (self.terminated or self.truncated):
            raise ValueError("a zero-transition prefix cannot carry done flags.")
        expected_bases: list[ReplayCompletionBasisV1] = []
        if self.terminated:
            expected_bases.append("task_terminal")
        if self.validated_transition_count == self.expected_transition_count:
            expected_bases.append("declared_horizon")
        if self.completion_state == "complete":
            if not expected_bases or self.completion_bases != tuple(expected_bases):
                raise ValueError("complete rollout bases must preserve exact evidence.")
            if self.failure_origin is not None:
                raise ValueError("complete rollout forbids a failure origin.")
        else:
            if expected_bases or self.completion_bases:
                raise ValueError("an incomplete prefix cannot carry completion bases.")
            if self.end_or_failure_reason is None:
                raise ValueError("an incomplete prefix requires a reason.")
            if self.completion_state == "failed":
                if self.failure_origin is None:
                    raise ValueError("failed rollout requires a failure origin.")
            elif self.failure_origin is not None:
                raise ValueError("only failed rollout may carry a failure origin.")
        return self


class ActorPovReplayCompletionBadgeV1(_ReplayProtocolModel):
    """Recipient-safe completion copied from the canonical POV content."""

    schema_version: Literal[1] = REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION
    episode_id: _ScientificId
    completion_state: ReplayCompletionStateV1
    expected_transition_count: _PositiveInt
    captured_transition_count: _NonNegativeInt
    terminated: bool
    truncated: bool
    completion_bases: tuple[ReplayCompletionBasisV1, ...]
    public_end_or_failure_reason: _Message | None = None

    @model_validator(mode="after")
    def _validate_completion(self) -> Self:
        if self.captured_transition_count > self.expected_transition_count:
            raise ValueError("captured transitions cannot exceed the expected horizon.")
        if self.captured_transition_count == 0 and (self.terminated or self.truncated):
            raise ValueError("a zero-transition POV prefix cannot carry done flags.")
        expected_bases: list[ReplayCompletionBasisV1] = []
        if self.terminated:
            expected_bases.append("task_terminal")
        if self.captured_transition_count == self.expected_transition_count:
            expected_bases.append("declared_horizon")
        if self.completion_state == "complete":
            if not expected_bases or self.completion_bases != tuple(expected_bases):
                raise ValueError("complete POV bases must preserve exact evidence.")
        else:
            if expected_bases or self.completion_bases:
                raise ValueError(
                    "an incomplete POV prefix cannot carry completion bases."
                )
            if self.public_end_or_failure_reason is None:
                raise ValueError("an incomplete POV prefix requires a public reason.")
        return self


class ReplayProcessingBadgeV1(_ReplayProtocolModel):
    """Minimal researcher processing progress without exception detail."""

    schema_version: Literal[1] = REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION
    status: ReplayProcessingStateV1
    processed_transition_count: _NonNegativeInt
    failure_stage: ReplayProcessingFailureStageV1 | None = None
    failure_code: _ScientificId | None = None
    attempted_transition_index: _NonNegativeInt | None = None

    @model_validator(mode="after")
    def _validate_processing(self) -> Self:
        if self.status == "succeeded":
            if any(
                value is not None
                for value in (
                    self.failure_stage,
                    self.failure_code,
                    self.attempted_transition_index,
                )
            ):
                raise ValueError("successful processing forbids failure fields.")
            return self
        if self.failure_stage is None or self.failure_code is None:
            raise ValueError("failed processing requires a stage and stable code.")
        permits_attempted_index = self.failure_stage in (
            "transition_validation",
            "reducer_advance",
        )
        if not permits_attempted_index and self.attempted_transition_index is not None:
            raise ValueError("this processing stage forbids an attempted index.")
        if (
            self.failure_stage == "reducer_advance"
            and self.attempted_transition_index is None
        ):
            raise ValueError("reducer advance failure requires an attempted index.")
        return self


class ActorPovProcessingDisclosureV1(_ReplayProtocolModel):
    """Constant disclosure preventing privileged processing metadata leakage."""

    schema_version: Literal[1] = REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION
    disclosure: Literal["not_available_in_actor_pov"] = (
        ACTOR_POV_PROCESSING_DISCLOSURE_V1
    )


class ReplayCursorV1(_ReplayProtocolModel):
    """Durable selected-frame progress; animation intent is not stored here."""

    schema_version: Literal[1] = REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION
    frame_index: _NonNegativeInt
    final_frame_index: _NonNegativeInt
    cursor_generation: _NonNegativeInt
    choreography_generation: _NonNegativeInt

    @model_validator(mode="after")
    def _validate_cursor(self) -> Self:
        if self.frame_index > self.final_frame_index:
            raise ValueError("replay cursor exceeds the captured prefix.")
        if self.choreography_generation > self.cursor_generation:
            raise ValueError("choreography generation cannot exceed cursor generation.")
        return self


def _expected_endpoint_kind(
    completion: ReplayCompletionBadgeV1 | ActorPovReplayCompletionBadgeV1,
) -> ReplayTimelineEndpointKindV1:
    if completion.completion_state != "complete":
        return "captured_prefix"
    bases = completion.completion_bases
    if bases == ("task_terminal", "declared_horizon"):
        return "task_terminal_and_declared_horizon"
    if bases == ("task_terminal",):
        return "task_terminal"
    if bases == ("declared_horizon",):
        return "declared_horizon"
    raise ValueError("complete timeline requires a supported completion basis.")


def _validate_completion_summary_join(
    summary: ReplayArtifactSummaryV1,
    completion: ReplayCompletionBadgeV1 | ActorPovReplayCompletionBadgeV1,
) -> None:
    if completion.episode_id != summary.replay_reference.episode_id:
        raise ValueError("timeline completion must join replay identity.")
    if completion.expected_transition_count != summary.expected_transition_count:
        raise ValueError("timeline completion horizon must equal replay summary.")
    if type(completion) is ReplayCompletionBadgeV1:
        captured_count = completion.validated_transition_count
    else:
        captured_count = cast(
            ActorPovReplayCompletionBadgeV1,
            completion,
        ).captured_transition_count
    if captured_count != summary.recorded_transition_count:
        raise ValueError("timeline completion must equal the captured replay prefix.")


def _validate_adjacent_simulator_step(
    *,
    previous_step: int | None,
    current_step: int,
) -> None:
    if previous_step is not None and current_step != previous_step + 1:
        raise ValueError("timeline simulator epochs must remain adjacent.")


class ResearcherReplayTimelineRowV1(_ReplayProtocolModel):
    """One compact researcher timeline row with canonical CP2 identity."""

    frame_index: _NonNegativeInt
    frame_id: _ScientificId
    simulator_step_count: _NonNegativeInt
    incoming_transition_id: _ScientificId | None
    incoming_event_count: _NonNegativeInt
    endpoint_kind: ReplayTimelineEndpointKindV1 = "none"


class ActorPovReplayTimelineRowV1(_ReplayProtocolModel):
    """One recipient-local timeline row without privileged CP2 event identity."""

    frame_index: _NonNegativeInt
    pov_frame_id: _ScientificId
    simulator_step_count: _NonNegativeInt
    incoming_pov_transition_id: _ScientificId | None
    incoming_cue_count: _NonNegativeInt
    endpoint_kind: ReplayTimelineEndpointKindV1 = "none"


class SharedObsSourceMaterialReplayTimelineRowV1(_ReplayProtocolModel):
    """One explicitly source-only SharedObs timeline row."""

    frame_index: _NonNegativeInt
    source_material_frame_id: _ScientificId
    simulator_step_count: _NonNegativeInt
    incoming_transition_id: _ScientificId | None
    endpoint_kind: ReplayTimelineEndpointKindV1 = "none"


class _ReplayTimelineBaseV1(_ReplayProtocolModel):
    schema_version: Literal[1] = REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION
    timeline_id: _ScientificId
    artifact_summary: ReplayArtifactSummaryV1
    final_frame_index: _NonNegativeInt

    @model_validator(mode="after")
    def _validate_base(self) -> Self:
        if type(self.artifact_summary) is not ReplayArtifactSummaryV1:
            raise ValueError("artifact_summary must be the exact replay root.")
        if self.final_frame_index != self.artifact_summary.recorded_transition_count:
            raise ValueError("timeline endpoint must equal the captured prefix.")
        return self


class ResearcherReplayTimelineV1(_ReplayTimelineBaseV1):
    timeline_kind: Literal["researcher"] = "researcher"
    completion: ReplayCompletionBadgeV1
    rows: tuple[ResearcherReplayTimelineRowV1, ...]

    @model_validator(mode="after")
    def _validate_timeline(self) -> Self:
        summary = self.artifact_summary
        _validate_researcher_artifact_summary(summary)
        completion = self.completion
        episode_id = summary.replay_reference.episode_id
        if type(completion) is not ReplayCompletionBadgeV1:
            raise ValueError("completion must be the exact researcher badge.")
        _validate_completion_summary_join(summary, completion)
        if self.timeline_id != (
            f"{summary.replay_reference.artifact_id}:timeline:researcher"
        ):
            raise ValueError("researcher timeline ID is not canonical.")
        if len(self.rows) != summary.recorded_frame_count:
            raise ValueError("researcher timeline must contain exactly T+1 rows.")
        endpoint = _expected_endpoint_kind(completion)
        previous_step: int | None = None
        for index, row in enumerate(self.rows):
            if (
                row.frame_index != index
                or row.frame_id != f"{episode_id}:frame:{index}"
            ):
                raise ValueError("researcher timeline frame identity is not canonical.")
            expected_transition_id = (
                None if index == 0 else f"{episode_id}:transition:{index - 1}"
            )
            if row.incoming_transition_id != expected_transition_id:
                raise ValueError("researcher timeline transition identity is invalid.")
            if index == 0 and row.incoming_event_count != 0:
                raise ValueError("frame zero cannot carry incoming events.")
            expected_endpoint = endpoint if index == self.final_frame_index else "none"
            if row.endpoint_kind != expected_endpoint:
                raise ValueError("researcher timeline endpoint marker is invalid.")
            _validate_adjacent_simulator_step(
                previous_step=previous_step,
                current_step=row.simulator_step_count,
            )
            previous_step = row.simulator_step_count
        return self


class ActorPovReplayTimelineV1(_ReplayTimelineBaseV1):
    timeline_kind: Literal["actor_pov"] = "actor_pov"
    pov_global_slot: _GlobalSlot
    public_agent_id: _PublicAgentId
    completion: ActorPovReplayCompletionBadgeV1
    rows: tuple[ActorPovReplayTimelineRowV1, ...]

    @model_validator(mode="after")
    def _validate_timeline(self) -> Self:
        summary = self.artifact_summary
        _validate_actor_pov_artifact_summary(summary)
        completion = self.completion
        episode_id = summary.replay_reference.episode_id
        if type(completion) is not ActorPovReplayCompletionBadgeV1:
            raise ValueError("completion must be the exact POV badge.")
        _validate_completion_summary_join(summary, completion)
        if self.timeline_id != (
            f"{summary.replay_reference.artifact_id}:timeline:actor-pov:"
            f"{self.public_agent_id}"
        ):
            raise ValueError("actor POV timeline ID is not canonical.")
        if len(self.rows) != summary.recorded_frame_count:
            raise ValueError("actor POV timeline must contain exactly T+1 rows.")
        endpoint = _expected_endpoint_kind(completion)
        previous_step: int | None = None
        for index, row in enumerate(self.rows):
            expected_frame_id = (
                f"{episode_id}:actor-pov:{self.public_agent_id}:frame:{index}"
            )
            if row.frame_index != index or row.pov_frame_id != expected_frame_id:
                raise ValueError("actor POV timeline frame identity is not canonical.")
            expected_transition_id = (
                None
                if index == 0
                else (
                    f"{episode_id}:actor-pov:{self.public_agent_id}:"
                    f"transition:{index - 1}"
                )
            )
            if row.incoming_pov_transition_id != expected_transition_id:
                raise ValueError("actor POV timeline transition identity is invalid.")
            if index == 0 and row.incoming_cue_count != 0:
                raise ValueError("POV frame zero cannot carry incoming cues.")
            expected_endpoint = endpoint if index == self.final_frame_index else "none"
            if row.endpoint_kind != expected_endpoint:
                raise ValueError("actor POV timeline endpoint marker is invalid.")
            _validate_adjacent_simulator_step(
                previous_step=previous_step,
                current_step=row.simulator_step_count,
            )
            previous_step = row.simulator_step_count
        return self


class SharedObsSourceMaterialReplayTimelineV1(_ReplayTimelineBaseV1):
    timeline_kind: Literal["shared_obs_source_material"] = "shared_obs_source_material"
    selected_global_slot: _GlobalSlot
    public_agent_id: _PublicAgentId
    observation_materialization: Literal["source_material_only"] = (
        "source_material_only"
    )
    completion: ReplayCompletionBadgeV1
    rows: tuple[SharedObsSourceMaterialReplayTimelineRowV1, ...]

    @model_validator(mode="after")
    def _validate_timeline(self) -> Self:
        summary = self.artifact_summary
        _validate_researcher_artifact_summary(summary)
        completion = self.completion
        episode_id = summary.replay_reference.episode_id
        if type(completion) is not ReplayCompletionBadgeV1:
            raise ValueError("completion must be the exact researcher badge.")
        _validate_completion_summary_join(summary, completion)
        if self.timeline_id != (
            f"{summary.replay_reference.artifact_id}:timeline:"
            f"shared-obs-source-material:{self.public_agent_id}"
        ):
            raise ValueError("SharedObs source-material timeline ID is not canonical.")
        if len(self.rows) != summary.recorded_frame_count:
            raise ValueError("source-material timeline must contain exactly T+1 rows.")
        endpoint = _expected_endpoint_kind(completion)
        previous_step: int | None = None
        for index, row in enumerate(self.rows):
            expected_frame_id = (
                f"{episode_id}:shared-obs-source-material:"
                f"{self.public_agent_id}:frame:{index}"
            )
            if (
                row.frame_index != index
                or row.source_material_frame_id != expected_frame_id
            ):
                raise ValueError("source-material frame identity is not canonical.")
            expected_transition_id = (
                None if index == 0 else f"{episode_id}:transition:{index - 1}"
            )
            if row.incoming_transition_id != expected_transition_id:
                raise ValueError("source-material transition identity is invalid.")
            expected_endpoint = endpoint if index == self.final_frame_index else "none"
            if row.endpoint_kind != expected_endpoint:
                raise ValueError("source-material endpoint marker is invalid.")
            _validate_adjacent_simulator_step(
                previous_step=previous_step,
                current_step=row.simulator_step_count,
            )
            previous_step = row.simulator_step_count
        return self


type ReplayTimelineV1 = Annotated[
    ResearcherReplayTimelineV1
    | ActorPovReplayTimelineV1
    | SharedObsSourceMaterialReplayTimelineV1,
    Field(discriminator="timeline_kind"),
]


class _ReplayViewerFrameBaseV1(_ReplayProtocolModel):
    schema_version: Literal[1] = REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION
    viewer_session_id: _OpaqueId
    revision: _NonNegativeInt
    artifact_summary: ReplayArtifactSummaryV1
    timeline_id: _ScientificId
    cursor: ReplayCursorV1
    preset: ReplayPresetV1
    verbose: bool

    @model_validator(mode="after")
    def _validate_base(self) -> Self:
        if type(self.artifact_summary) is not ReplayArtifactSummaryV1:
            raise ValueError("artifact_summary must be the exact replay root.")
        if type(self.cursor) is not ReplayCursorV1:
            raise ValueError("cursor must be the exact replay cursor root.")
        if (
            self.cursor.final_frame_index
            != self.artifact_summary.recorded_transition_count
        ):
            raise ValueError("cursor endpoint must equal the captured prefix.")
        return self


def _validate_researcher_progress(
    summary: ReplayArtifactSummaryV1,
    completion: ReplayCompletionBadgeV1,
    processing: ReplayProcessingBadgeV1,
) -> None:
    reference = summary.replay_reference
    if completion.episode_id != reference.episode_id:
        raise ValueError("completion must join replay identity.")
    if completion.expected_transition_count != summary.expected_transition_count:
        raise ValueError("completion horizon must equal replay summary.")
    if completion.validated_transition_count != summary.recorded_transition_count:
        raise ValueError("completion progress must equal the retained replay prefix.")
    if processing.processed_transition_count > completion.validated_transition_count:
        raise ValueError("processed progress cannot exceed validated progress.")
    if (
        processing.status == "succeeded"
        and processing.processed_transition_count
        != completion.validated_transition_count
    ):
        raise ValueError("successful processing must cover the validated prefix.")


class ResearcherReplayViewerFrameV1(_ReplayViewerFrameBaseV1):
    """Read-only researcher frame carrying the accepted CP7.1 projection."""

    frame_kind: Literal["researcher_replay_viewer"] = "researcher_replay_viewer"
    view_mode: Literal["researcher"] = "researcher"
    frame_id: _ScientificId
    simulator_step_count: _NonNegativeInt
    incoming_transition_index: _NonNegativeInt | None
    incoming_transition_id: _ScientificId | None
    completion: ReplayCompletionBadgeV1
    processing: ReplayProcessingBadgeV1
    show_ranges: bool
    projection: ResearcherAnalyzerProjectionV2

    @model_validator(mode="after")
    def _validate_frame(self) -> Self:
        _validate_researcher_artifact_summary(self.artifact_summary)
        _validate_researcher_progress(
            self.artifact_summary,
            self.completion,
            self.processing,
        )
        reference = self.artifact_summary.replay_reference
        frame_index = self.cursor.frame_index
        if self.timeline_id != f"{reference.artifact_id}:timeline:researcher":
            raise ValueError("researcher frame timeline ID is not canonical.")
        if self.frame_id != f"{reference.episode_id}:frame:{frame_index}":
            raise ValueError("researcher frame ID is not canonical.")
        expected_transition_index = None if frame_index == 0 else frame_index - 1
        expected_transition_id = (
            None
            if expected_transition_index is None
            else f"{reference.episode_id}:transition:{expected_transition_index}"
        )
        if (
            self.incoming_transition_index != expected_transition_index
            or self.incoming_transition_id != expected_transition_id
        ):
            raise ValueError("incoming researcher transition does not enter the frame.")
        if type(self.projection) is not ResearcherAnalyzerProjectionV2:
            raise ValueError("projection must be exact ResearcherAnalyzerProjectionV2.")
        scene = self.projection.scene
        if (
            scene.episode_id != reference.episode_id
            or scene.frame_index != frame_index
            or scene.frame_id != self.frame_id
            or scene.simulator_step_count != self.simulator_step_count
            or scene.incoming_transition_id != self.incoming_transition_id
        ):
            raise ValueError("researcher projection must join the replay envelope.")
        return self


class ActorPovReplayViewerFrameV1(_ReplayViewerFrameBaseV1):
    """Exact NoSharedObs recipient frame with no privileged processing truth."""

    frame_kind: Literal["actor_pov_replay_viewer"] = "actor_pov_replay_viewer"
    view_mode: Literal["pov"] = "pov"
    pov_global_slot: _GlobalSlot
    public_agent_id: _PublicAgentId
    pov_frame_id: _ScientificId
    simulator_step_count: _NonNegativeInt
    incoming_pov_transition_id: _ScientificId | None
    completion: ActorPovReplayCompletionBadgeV1
    processing_disclosure: ActorPovProcessingDisclosureV1
    projection: ActorPovAnalyzerProjectionV1

    @model_validator(mode="after")
    def _validate_frame(self) -> Self:
        summary = self.artifact_summary
        _validate_actor_pov_artifact_summary(summary)
        reference = summary.replay_reference
        completion = self.completion
        frame_index = self.cursor.frame_index
        if completion.episode_id != reference.episode_id:
            raise ValueError("POV completion must join replay identity.")
        if completion.expected_transition_count != summary.expected_transition_count:
            raise ValueError("POV horizon must equal replay summary.")
        if completion.captured_transition_count != summary.recorded_transition_count:
            raise ValueError("POV completion must equal the captured replay prefix.")
        if type(self.processing_disclosure) is not ActorPovProcessingDisclosureV1:
            raise ValueError("POV processing must use the non-disclosure root.")
        if self.timeline_id != (
            f"{reference.artifact_id}:timeline:actor-pov:{self.public_agent_id}"
        ):
            raise ValueError("actor POV frame timeline ID is not canonical.")
        expected_frame_id = (
            f"{reference.episode_id}:actor-pov:"
            f"{self.public_agent_id}:frame:{frame_index}"
        )
        if self.pov_frame_id != expected_frame_id:
            raise ValueError("actor POV frame ID is not canonical.")
        expected_transition_id = (
            None
            if frame_index == 0
            else (
                f"{reference.episode_id}:actor-pov:{self.public_agent_id}:"
                f"transition:{frame_index - 1}"
            )
        )
        if self.incoming_pov_transition_id != expected_transition_id:
            raise ValueError("incoming POV transition does not enter the frame.")
        if type(self.projection) is not ActorPovAnalyzerProjectionV1:
            raise ValueError("projection must be exact ActorPovAnalyzerProjectionV1.")
        scene = self.projection.scene
        if (
            scene.episode_id != reference.episode_id
            or scene.frame_index != frame_index
            or scene.pov_frame_id != self.pov_frame_id
            or scene.simulator_step_count != self.simulator_step_count
            or scene.self_actor.global_slot != self.pov_global_slot
            or scene.self_actor.public_agent_id != self.public_agent_id
            or self.projection.incoming_transition_id != self.incoming_pov_transition_id
        ):
            raise ValueError("actor POV projection must join the replay envelope.")
        return self


class SharedObsSourceMaterialReplayViewerFrameV1(_ReplayViewerFrameBaseV1):
    """Labelled SharedObs base-sensor source material, never composed input."""

    frame_kind: Literal["shared_obs_source_material_replay_viewer"] = (
        "shared_obs_source_material_replay_viewer"
    )
    view_mode: Literal["pov"] = "pov"
    selected_global_slot: _GlobalSlot
    public_agent_id: _PublicAgentId
    observation_materialization: Literal["source_material_only"] = (
        "source_material_only"
    )
    source_material_frame_id: _ScientificId
    source_frame_id: _ScientificId
    simulator_step_count: _NonNegativeInt
    incoming_transition_id: _ScientificId | None
    completion: ReplayCompletionBadgeV1
    processing: ReplayProcessingBadgeV1
    projection: SharedObsSourceMaterialProjectionV1

    @model_validator(mode="after")
    def _validate_frame(self) -> Self:
        _validate_researcher_artifact_summary(self.artifact_summary)
        _validate_researcher_progress(
            self.artifact_summary,
            self.completion,
            self.processing,
        )
        reference = self.artifact_summary.replay_reference
        frame_index = self.cursor.frame_index
        if self.timeline_id != (
            f"{reference.artifact_id}:timeline:shared-obs-source-material:"
            f"{self.public_agent_id}"
        ):
            raise ValueError("source-material frame timeline ID is not canonical.")
        expected_material_id = (
            f"{reference.episode_id}:shared-obs-source-material:"
            f"{self.public_agent_id}:frame:{frame_index}"
        )
        if self.source_material_frame_id != expected_material_id:
            raise ValueError("source-material frame ID is not canonical.")
        if self.source_frame_id != f"{reference.episode_id}:frame:{frame_index}":
            raise ValueError("SharedObs source frame ID is not canonical.")
        expected_transition_id = (
            None
            if frame_index == 0
            else f"{reference.episode_id}:transition:{frame_index - 1}"
        )
        if self.incoming_transition_id != expected_transition_id:
            raise ValueError("incoming source-material transition is invalid.")
        if type(self.projection) is not SharedObsSourceMaterialProjectionV1:
            raise ValueError(
                "projection must be exact SharedObsSourceMaterialProjectionV1."
            )
        projection = self.projection
        base = projection.base_sensor_frame
        if (
            base.episode_id != reference.episode_id
            or base.frame_index != frame_index
            or base.source_material_frame_id != self.source_material_frame_id
            or base.source_frame_id != self.source_frame_id
            or base.simulator_step_count != self.simulator_step_count
            or base.public_agent_id != self.public_agent_id
            or projection.base_sensor_scene.self_actor.global_slot
            != self.selected_global_slot
            or projection.incoming_transition_id != self.incoming_transition_id
        ):
            raise ValueError("SharedObs projection must join the replay envelope.")
        return self


type ReplayViewerFrameV1 = Annotated[
    ResearcherReplayViewerFrameV1
    | ActorPovReplayViewerFrameV1
    | SharedObsSourceMaterialReplayViewerFrameV1,
    Field(discriminator="frame_kind"),
]


class ReplayAbsoluteSeekCommandV1(_ReplayProtocolModel):
    command_type: Literal["absolute_seek"] = "absolute_seek"
    frame_index: _NonNegativeInt


class ReplayFirstFrameCommandV1(_ReplayProtocolModel):
    command_type: Literal["first_frame"] = "first_frame"


class ReplayPreviousFrameCommandV1(_ReplayProtocolModel):
    command_type: Literal["previous_frame"] = "previous_frame"


class ReplayNextFrameCommandV1(_ReplayProtocolModel):
    command_type: Literal["next_frame"] = "next_frame"


class ReplayLastFrameCommandV1(_ReplayProtocolModel):
    command_type: Literal["last_frame"] = "last_frame"


class ReplaySelectAgentCommandV1(_ReplayProtocolModel):
    command_type: Literal["select_agent"] = "select_agent"
    selected_global_slot: _GlobalSlot | None


class ReplaySetViewCommandV1(_ReplayProtocolModel):
    command_type: Literal["set_view"] = "set_view"
    view_mode: ReplayViewModeV1


class ReplaySetPovActorCommandV1(_ReplayProtocolModel):
    command_type: Literal["set_pov_actor"] = "set_pov_actor"
    global_slot: _GlobalSlot


class ReplaySetPresetCommandV1(_ReplayProtocolModel):
    command_type: Literal["set_preset"] = "set_preset"
    preset: ReplayPresetV1


class ReplaySetRangesCommandV1(_ReplayProtocolModel):
    command_type: Literal["set_ranges"] = "set_ranges"
    show_ranges: bool


class ReplaySetVerbosityCommandV1(_ReplayProtocolModel):
    command_type: Literal["set_verbosity"] = "set_verbosity"
    verbose: bool


class ReplayExitCommandV1(_ReplayProtocolModel):
    command_type: Literal["exit"] = "exit"


type ReplayCommandV1 = Annotated[
    ReplayAbsoluteSeekCommandV1
    | ReplayFirstFrameCommandV1
    | ReplayPreviousFrameCommandV1
    | ReplayNextFrameCommandV1
    | ReplayLastFrameCommandV1
    | ReplaySelectAgentCommandV1
    | ReplaySetViewCommandV1
    | ReplaySetPovActorCommandV1
    | ReplaySetPresetCommandV1
    | ReplaySetRangesCommandV1
    | ReplaySetVerbosityCommandV1
    | ReplayExitCommandV1,
    Field(discriminator="command_type"),
]


class ReplayCommandRequestV1(_ReplayProtocolModel):
    schema_version: Literal[1] = REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION
    client_id: _OpaqueId
    command_id: _OpaqueId
    base_revision: _NonNegativeInt
    command: ReplayCommandV1


class ReplayCommandResponseV1(_ReplayProtocolModel):
    schema_version: Literal[1] = REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION
    result: ReplayCommandResultV1
    frame: ReplayViewerFrameV1
    notice: _Message | None = None
    animate_incoming: bool = False

    @model_validator(mode="after")
    def _validate_animation_intent(self) -> Self:
        if self.animate_incoming:
            if self.result != "applied":
                raise ValueError("only an applied command may request choreography.")
            if self.frame.cursor.frame_index == 0:
                raise ValueError("frame zero has no incoming transition to animate.")
            if self.frame.cursor.choreography_generation == 0:
                raise ValueError("animation requires an advanced choreography epoch.")
        return self


class ReplayApiErrorV1(_ReplayProtocolModel):
    schema_version: Literal[1] = REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION
    error_code: ReplayApiErrorCodeV1
    message: _Message
    latest_frame: ReplayViewerFrameV1 | None = None


__all__ = [
    "ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1",
    "ACTOR_POV_PROCESSING_DISCLOSURE_V1",
    "REPLAY_VIEWER_PROTOCOL_SCHEMA_VERSION",
    "ActorPovProcessingDisclosureV1",
    "ActorPovReplayCompletionBadgeV1",
    "ActorPovReplayTimelineRowV1",
    "ActorPovReplayTimelineV1",
    "ActorPovReplayViewerFrameV1",
    "ReplayAbsoluteSeekCommandV1",
    "ReplayApiErrorCodeV1",
    "ReplayApiErrorV1",
    "ReplayArtifactSummaryV1",
    "ReplayCommandRequestV1",
    "ReplayCommandResponseV1",
    "ReplayCommandResultV1",
    "ReplayCommandV1",
    "ReplayCompletionBadgeV1",
    "ReplayCompletionBasisV1",
    "ReplayCompletionStateV1",
    "ReplayCursorV1",
    "ReplayExitCommandV1",
    "ReplayFirstFrameCommandV1",
    "ReplayLastFrameCommandV1",
    "ReplayNextFrameCommandV1",
    "ReplayPresetV1",
    "ReplayPreviousFrameCommandV1",
    "ReplayProcessingBadgeV1",
    "ReplayProcessingFailureStageV1",
    "ReplayProcessingStateV1",
    "ReplayRolloutFailureOriginV1",
    "ReplaySelectAgentCommandV1",
    "ReplaySetPovActorCommandV1",
    "ReplaySetPresetCommandV1",
    "ReplaySetRangesCommandV1",
    "ReplaySetVerbosityCommandV1",
    "ReplaySetViewCommandV1",
    "ReplayTimelineEndpointKindV1",
    "ReplayTimelineV1",
    "ReplayViewModeV1",
    "ReplayViewerFrameV1",
    "ResearcherReplayTimelineRowV1",
    "ResearcherReplayTimelineV1",
    "ResearcherReplayViewerFrameV1",
    "SharedObsSourceMaterialReplayTimelineRowV1",
    "SharedObsSourceMaterialReplayTimelineV1",
    "SharedObsSourceMaterialReplayViewerFrameV1",
]
