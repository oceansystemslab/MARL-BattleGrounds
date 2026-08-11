"""Opt-in host-side metric streaming over validated evaluation records.

This module deliberately contains no simulator, JAX, persistence, logging, or
official metric-formula authority.  It provides strict raw component records,
an immutable accumulator, and a transactional observer seam for trusted pure
reducers that consume already validated CP2 transition units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, cast

from pydantic import BeforeValidator, Field, StringConstraints, model_validator

from marl_battlegrounds.evaluation.models import (
    REQUIRED_SCHEMA_BINDINGS_V1,
    AggregationKeyV1,
    AssignedPolicySlotV1,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationModel,
    EvaluationTransitionV1,
    SchemaVersionEntryV1,
)
from marl_battlegrounds.evaluation.validation import (
    validate_declared_model_tree,
    validate_evaluation_transition_unit_v1,
    validate_initial_evaluation_frame_v1,
)

EPISODE_COMPLETION_SCHEMA_ID = "marl_battlegrounds.evaluation.episode_completion"
PROCESSING_STATUS_SCHEMA_ID = "marl_battlegrounds.evaluation.processing_status"
RAW_SUFFICIENT_STATISTIC_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.raw_sufficient_statistic"
)
METRIC_REPORT_SCHEMA_ID = "marl_battlegrounds.evaluation.metric_report"
CP3_SCHEMA_VERSION: Literal[1] = 1

type CompletionState = Literal["complete", "partial", "interrupted", "failed"]
type CompletionBasis = Literal["task_terminal", "declared_horizon"]
type RolloutFailureOrigin = Literal["simulation", "policy", "validation", "capture"]
type ProcessingState = Literal["succeeded", "failed"]
type ProcessingFailureStage = Literal[
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
type ObserverLifecycleState = Literal[
    "awaiting_initial",
    "open",
    "sealed",
    "poisoned",
    "finalized",
]
type StatisticCompletionScope = Literal[
    "any_gap_free_prefix",
    "complete_episode",
]
type StatisticResultStatus = Literal[
    "invalid_artifact",
    "structurally_inapplicable",
    "ambiguous_attribution",
    "insufficient_data",
    "zero_opportunity",
    "defined",
]
type EndpointObservationStatus = Literal[
    "not_applicable",
    "observed",
    "right_censored",
    "competing_event",
    "unavailable",
]
type HealthAmountStage = Literal[
    "raw_source",
    "source_modified_gross",
    "recipient_modified_gross",
    "combat_resolution_health",
    "realized_net_health_change",
    "actual_regeneration",
]

_AsciiText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[\x20-\x7e]+$",
    ),
]
_AsciiIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]*$",
    ),
]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(gt=0)]
_FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
_NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
_GlobalSlot = Annotated[int, Field(ge=0, lt=10)]
_TeamId = Annotated[int, Field(ge=1, le=2)]
_ClassId = Annotated[int, Field(ge=1, le=5)]


def _require_binary_int(value: object) -> object:
    if type(value) is not int or value not in (0, 1):
        raise ValueError("value must be an exact integer 0 or 1")
    return value


def _require_schema_version_one(value: object) -> object:
    if type(value) is not int or value != 1:
        raise ValueError("schema_version must be the exact integer 1")
    return value


_BinaryInt = Annotated[int, BeforeValidator(_require_binary_int)]
_EpisodeEligibility = _BinaryInt
_Cp3SchemaVersion = Annotated[
    Literal[1],
    BeforeValidator(_require_schema_version_one),
]

_RESERVED_DIMENSION_NAMES = frozenset(
    {
        "actor_projection",
        "amount_stage",
        "aggregation_keys",
        "algorithm_id",
        "assignment_status",
        "canonical_reward_mode",
        "capture_profile",
        "canonical_digest",
        "checkpoint_digest",
        "class_id",
        "code_revision",
        "component",
        "component_name",
        "component_type",
        "completion_bases",
        "completion_scope",
        "completion_state",
        "configured_active",
        "configured_team_id",
        "commit_sha",
        "cooperative_partner_seed",
        "critic_information_regime",
        "curriculum",
        "dirty_patch_digest",
        "dimensions",
        "denominator",
        "eligible_episode_count",
        "eligible_steps",
        "end_or_failure_reason",
        "endpoint_observation_status",
        "environment_seed",
        "episode_id",
        "episode_seed",
        "evaluation_seed",
        "evaluation_id",
        "evaluation_role",
        "evaluation_suite",
        "expected_horizon",
        "expected_transition_count",
        "execution_information_mode",
        "execution_mode",
        "experiment_manifest",
        "global_slot",
        "identity",
        "is_dirty",
        "layout",
        "layout_seed",
        "last_valid_frame_id",
        "last_valid_frame_index",
        "match_id",
        "matchup_id",
        "metric_id",
        "metric_version",
        "normalization",
        "numerator",
        "observation_count",
        "observations",
        "opportunity_count",
        "ordinal",
        "package_version",
        "paired_comparison_key",
        "parameter_sharing_group_id",
        "policy_content_digest",
        "policy_id",
        "policy_assignments",
        "policy_kind",
        "population_member_id",
        "preprocessing",
        "primary_global_slot",
        "processed_transition_count",
        "processing_status",
        "public_agent_id",
        "qualifying_steps",
        "reducer_id",
        "reducer_version",
        "report_id",
        "result_status",
        "resolved_env_config",
        "run_id",
        "root_seed",
        "roster",
        "rollout_completion",
        "scenario",
        "scenario_seed",
        "schema_id",
        "schema_version",
        "schema_versions",
        "secondary_global_slot",
        "seed",
        "seed_protocol",
        "shaping_configuration",
        "source_observation_id",
        "source_schema_versions",
        "source_tree_digest",
        "status_reason",
        "static_mechanics_catalog",
        "subject",
        "subject_type",
        "supports_right_censoring",
        "task",
        "team_id",
        "team_local_slot",
        "terminated",
        "training_run_id",
        "training_step",
        "truncated",
        "units",
        "validated_transition_count",
        "value",
        "zero_opportunity_occurrence",
        "adversarial_opponent_seed",
        "count",
        "failure_origin",
        "focal_policy_seed",
    }
)


class StatisticDimensionV1(EvaluationModel):
    """One metric-defined categorical coordinate, excluding context truth."""

    name: _AsciiIdentifier
    value: _AsciiIdentifier

    @model_validator(mode="after")
    def _reject_context_shadowing(self) -> StatisticDimensionV1:
        if self.name in _RESERVED_DIMENSION_NAMES:
            raise ValueError(f"statistic dimension {self.name!r} shadows context truth")
        return self


class EpisodeStatisticSubjectV1(EvaluationModel):
    """The episode as a whole."""

    subject_type: Literal["episode"] = "episode"


class TeamStatisticSubjectV1(EvaluationModel):
    """One configured team."""

    subject_type: Literal["team"] = "team"
    team_id: _TeamId


class AgentStatisticSubjectV1(EvaluationModel):
    """One configured-active global agent slot."""

    subject_type: Literal["agent"] = "agent"
    global_slot: _GlobalSlot


class TeamClassStatisticSubjectV1(EvaluationModel):
    """One team/class stratum, including a declared absent class."""

    subject_type: Literal["team_class"] = "team_class"
    team_id: _TeamId
    class_id: _ClassId


class AgentPairStatisticSubjectV1(EvaluationModel):
    """One ordered pair of distinct configured-active agent slots."""

    subject_type: Literal["agent_pair"] = "agent_pair"
    primary_global_slot: _GlobalSlot
    secondary_global_slot: _GlobalSlot

    @model_validator(mode="after")
    def _require_distinct_slots(self) -> AgentPairStatisticSubjectV1:
        if self.primary_global_slot == self.secondary_global_slot:
            raise ValueError("ordered agent-pair subjects require two distinct slots")
        return self


type StatisticSubjectV1 = Annotated[
    EpisodeStatisticSubjectV1
    | TeamStatisticSubjectV1
    | AgentStatisticSubjectV1
    | TeamClassStatisticSubjectV1
    | AgentPairStatisticSubjectV1,
    Field(discriminator="subject_type"),
]


def _require_stable_nested_model(
    model: EvaluationModel,
    *,
    record_name: str,
    expected_types: tuple[type[EvaluationModel], ...],
) -> None:
    """Reject undeclared subtypes and unchecked Pydantic model copies."""
    if type(model) not in expected_types:
        expected_names = ", ".join(row.__name__ for row in expected_types)
        raise ValueError(
            f"{record_name} must use an exact declared schema type: {expected_names}"
        )
    validate_declared_model_tree(
        model,
        record_name=record_name,
        expected_type=type(model),
    )


class DistributionObservationV1(EvaluationModel):
    """One linked long-form observation retained without moment reduction."""

    source_observation_id: _AsciiIdentifier
    ordinal: _NonNegativeInt
    value: _FiniteFloat


class CountComponentV1(EvaluationModel):
    """An additive event/row count with episode eligibility exposure."""

    component_type: Literal["count"] = "count"
    count: _NonNegativeInt
    eligible_episode_count: _EpisodeEligibility


class SumComponentV1(EvaluationModel):
    """A finite additive total with observation and episode exposure."""

    component_type: Literal["sum"] = "sum"
    value: _FiniteFloat
    observation_count: _NonNegativeInt
    eligible_episode_count: _EpisodeEligibility

    @model_validator(mode="after")
    def _validate_empty_sum(self) -> SumComponentV1:
        if self.observation_count == 0 and self.value != 0.0:
            raise ValueError("zero observations require a zero sum")
        return self


class RatioComponentV1(EvaluationModel):
    """One episode-local numerator/denominator pair, never a computed ratio."""

    component_type: Literal["ratio"] = "ratio"
    numerator: _FiniteFloat
    denominator: _NonNegativeFloat
    zero_opportunity_occurrence: _BinaryInt
    eligible_episode_count: _EpisodeEligibility

    @model_validator(mode="after")
    def _validate_denominator(self) -> RatioComponentV1:
        if self.denominator == 0.0:
            if self.numerator != 0.0 or self.zero_opportunity_occurrence != 1:
                raise ValueError(
                    "zero denominator requires zero numerator and one "
                    "zero-opportunity occurrence"
                )
        elif self.zero_opportunity_occurrence != 0:
            raise ValueError(
                "positive denominator forbids a zero-opportunity occurrence"
            )
        return self


class DurationComponentV1(EvaluationModel):
    """Qualifying and eligible discrete transition durations."""

    component_type: Literal["duration"] = "duration"
    qualifying_steps: _NonNegativeInt
    eligible_steps: _NonNegativeInt
    eligible_episode_count: _EpisodeEligibility

    @model_validator(mode="after")
    def _validate_duration(self) -> DurationComponentV1:
        if self.qualifying_steps > self.eligible_steps:
            raise ValueError("qualifying duration cannot exceed eligible duration")
        return self


class OpportunityComponentV1(EvaluationModel):
    """An explicit opportunity denominator with episode exposure."""

    component_type: Literal["opportunity"] = "opportunity"
    opportunity_count: _NonNegativeInt
    eligible_episode_count: _EpisodeEligibility


class DistributionComponentV1(EvaluationModel):
    """Ordered long-form finite observations with episode exposure."""

    component_type: Literal["distribution"] = "distribution"
    observations: tuple[DistributionObservationV1, ...]
    eligible_episode_count: _EpisodeEligibility

    @model_validator(mode="after")
    def _validate_observations(self) -> DistributionComponentV1:
        for observation in self.observations:
            _require_stable_nested_model(
                observation,
                record_name="distribution observation",
                expected_types=(DistributionObservationV1,),
            )
        ordinals = tuple(row.ordinal for row in self.observations)
        if ordinals != tuple(range(len(self.observations))):
            raise ValueError("distribution observation ordinals must be gap-free")
        source_ids = tuple(row.source_observation_id for row in self.observations)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("distribution source observation IDs must be unique")
        return self


type SufficientStatisticComponentV1 = Annotated[
    CountComponentV1
    | SumComponentV1
    | RatioComponentV1
    | DurationComponentV1
    | OpportunityComponentV1
    | DistributionComponentV1,
    Field(discriminator="component_type"),
]


def _dimensions_key(
    dimensions: tuple[StatisticDimensionV1, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple((row.name, row.value) for row in dimensions)


def _subject_key(subject: StatisticSubjectV1) -> str:
    return subject.model_dump_json()


def _component_has_zero_opportunity(
    component: SufficientStatisticComponentV1,
) -> bool:
    if isinstance(component, CountComponentV1):
        return False
    if isinstance(component, SumComponentV1):
        return component.eligible_episode_count > 0 and component.observation_count == 0
    if isinstance(component, RatioComponentV1):
        return component.eligible_episode_count > 0 and component.denominator == 0.0
    if isinstance(component, DurationComponentV1):
        return component.eligible_episode_count > 0 and component.eligible_steps == 0
    if isinstance(component, OpportunityComponentV1):
        return component.eligible_episode_count > 0 and component.opportunity_count == 0
    return component.eligible_episode_count > 0 and len(component.observations) == 0


class SufficientStatisticDraftV1(EvaluationModel):
    """Reducer-owned statistic semantics before observer provenance joins."""

    metric_id: _AsciiIdentifier
    metric_version: _PositiveInt
    component_name: _AsciiIdentifier
    reducer_id: _AsciiIdentifier
    reducer_version: _PositiveInt
    units: _AsciiIdentifier
    amount_stage: HealthAmountStage | None = None
    subject: StatisticSubjectV1
    dimensions: tuple[StatisticDimensionV1, ...] = ()
    completion_scope: StatisticCompletionScope
    supports_right_censoring: bool
    result_status: StatisticResultStatus
    status_reason: _AsciiText | None = None
    endpoint_observation_status: EndpointObservationStatus
    component: SufficientStatisticComponentV1 | None

    @model_validator(mode="after")
    def _validate_draft(self) -> SufficientStatisticDraftV1:
        _require_stable_nested_model(
            self.subject,
            record_name="statistic subject",
            expected_types=(
                EpisodeStatisticSubjectV1,
                TeamStatisticSubjectV1,
                AgentStatisticSubjectV1,
                TeamClassStatisticSubjectV1,
                AgentPairStatisticSubjectV1,
            ),
        )
        for dimension in self.dimensions:
            _require_stable_nested_model(
                dimension,
                record_name="statistic dimension",
                expected_types=(StatisticDimensionV1,),
            )
        if self.component is not None:
            _require_stable_nested_model(
                self.component,
                record_name="statistic component",
                expected_types=(
                    CountComponentV1,
                    SumComponentV1,
                    RatioComponentV1,
                    DurationComponentV1,
                    OpportunityComponentV1,
                    DistributionComponentV1,
                ),
            )
        if not self.metric_id.endswith(f".v{self.metric_version}"):
            raise ValueError("metric_id must end with its declared .vN version")
        dimension_key = _dimensions_key(self.dimensions)
        if dimension_key != tuple(sorted(dimension_key)):
            raise ValueError("statistic dimensions must be canonically sorted")
        names = tuple(name for name, _value in dimension_key)
        if len(names) != len(set(names)):
            raise ValueError("statistic dimension names must be unique")
        if self.result_status == "defined":
            if self.component is None:
                raise ValueError("defined statistics require a raw component")
            if self.status_reason is not None:
                raise ValueError("defined statistics forbid a status reason")
        else:
            if self.status_reason is None:
                raise ValueError("non-defined statistics require a status reason")
            if self.result_status == "zero_opportunity" and (
                self.component is None
                or not _component_has_zero_opportunity(self.component)
            ):
                raise ValueError(
                    "zero-opportunity status requires zero-opportunity evidence "
                    "in its component"
                )
        if (
            self.endpoint_observation_status == "right_censored"
            and not self.supports_right_censoring
        ):
            raise ValueError(
                "right-censored drafts must declare right-censoring support"
            )
        return self


def _draft_row_key(
    draft: SufficientStatisticDraftV1,
) -> tuple[object, ...]:
    return (
        draft.metric_id,
        draft.metric_version,
        draft.component_name,
        _subject_key(draft.subject),
        _dimensions_key(draft.dimensions),
    )


def _draft_metadata_key(
    draft: SufficientStatisticDraftV1,
) -> tuple[object, ...]:
    return (
        *_draft_row_key(draft),
        draft.reducer_id,
        draft.reducer_version,
        draft.units,
        draft.amount_stage,
        draft.completion_scope,
        draft.supports_right_censoring,
        draft.result_status,
        draft.status_reason,
        draft.endpoint_observation_status,
        None if draft.component is None else draft.component.component_type,
    )


def _merge_components(
    left: SufficientStatisticComponentV1,
    right: SufficientStatisticComponentV1,
) -> SufficientStatisticComponentV1:
    if type(left) is not type(right):
        raise ValueError("sufficient-statistic component families must match")
    if isinstance(left, CountComponentV1) and isinstance(right, CountComponentV1):
        return CountComponentV1(
            count=left.count + right.count,
            eligible_episode_count=max(
                left.eligible_episode_count,
                right.eligible_episode_count,
            ),
        )
    if isinstance(left, SumComponentV1) and isinstance(right, SumComponentV1):
        return SumComponentV1(
            value=left.value + right.value,
            observation_count=left.observation_count + right.observation_count,
            eligible_episode_count=max(
                left.eligible_episode_count,
                right.eligible_episode_count,
            ),
        )
    if isinstance(left, RatioComponentV1) and isinstance(right, RatioComponentV1):
        denominator = left.denominator + right.denominator
        return RatioComponentV1(
            numerator=left.numerator + right.numerator,
            denominator=denominator,
            zero_opportunity_occurrence=0 if denominator > 0.0 else 1,
            eligible_episode_count=max(
                left.eligible_episode_count,
                right.eligible_episode_count,
            ),
        )
    if isinstance(left, DurationComponentV1) and isinstance(right, DurationComponentV1):
        return DurationComponentV1(
            qualifying_steps=left.qualifying_steps + right.qualifying_steps,
            eligible_steps=left.eligible_steps + right.eligible_steps,
            eligible_episode_count=max(
                left.eligible_episode_count,
                right.eligible_episode_count,
            ),
        )
    if isinstance(left, OpportunityComponentV1) and isinstance(
        right, OpportunityComponentV1
    ):
        return OpportunityComponentV1(
            opportunity_count=left.opportunity_count + right.opportunity_count,
            eligible_episode_count=max(
                left.eligible_episode_count,
                right.eligible_episode_count,
            ),
        )
    if not isinstance(left, DistributionComponentV1) or not isinstance(
        right, DistributionComponentV1
    ):
        raise TypeError("unrecognized sufficient-statistic component family")
    source_ids = {
        row.source_observation_id for row in (*left.observations, *right.observations)
    }
    if len(source_ids) != len(left.observations) + len(right.observations):
        raise ValueError("distribution merges forbid duplicate observation IDs")
    observations = tuple(
        row.model_copy(update={"ordinal": ordinal})
        for ordinal, row in enumerate((*left.observations, *right.observations))
    )
    return DistributionComponentV1(
        observations=observations,
        eligible_episode_count=max(
            left.eligible_episode_count,
            right.eligible_episode_count,
        ),
    )


def _merge_drafts(
    left: SufficientStatisticDraftV1,
    right: SufficientStatisticDraftV1,
) -> SufficientStatisticDraftV1:
    if _draft_row_key(left) != _draft_row_key(right):
        raise ValueError("only identical statistic row keys may merge")
    if _draft_metadata_key(left) != _draft_metadata_key(right):
        raise ValueError("statistic metadata conflicts for the same row key")
    if left.component is None or right.component is None:
        if left.component is not None or right.component is not None:
            raise ValueError("undefined statistic component presence must match")
        return left
    merged_component = _merge_components(left.component, right.component)
    merged_status = left.result_status
    merged_reason = left.status_reason
    if merged_status == "zero_opportunity" and not _component_has_zero_opportunity(
        merged_component
    ):
        merged_status = "defined"
        merged_reason = None
    return _replace_draft(
        left,
        component=merged_component,
        result_status=merged_status,
        status_reason=merged_reason,
    )


class SufficientStatisticAccumulatorV1(EvaluationModel):
    """Immutable episode-local collection of compatible raw statistic drafts.

    Cross-episode aggregation consumes finalized ``RawSufficientStatisticV1``
    rows downstream. It never merges these reducer-state drafts, so a ratio's
    zero-opportunity occurrence remains the canonical per-episode 0/1 value.
    """

    entries: tuple[SufficientStatisticDraftV1, ...] = ()

    @model_validator(mode="after")
    def _validate_entries(self) -> SufficientStatisticAccumulatorV1:
        for entry in self.entries:
            _require_stable_nested_model(
                entry,
                record_name="accumulator draft",
                expected_types=(SufficientStatisticDraftV1,),
            )
        keys = tuple(_draft_row_key(row) for row in self.entries)
        if keys != tuple(sorted(keys)):
            raise ValueError("accumulator entries must be canonically sorted")
        if len(keys) != len(set(keys)):
            raise ValueError("accumulator entries must have unique statistic keys")
        return self

    def add(
        self,
        draft: SufficientStatisticDraftV1,
    ) -> SufficientStatisticAccumulatorV1:
        """Return a replacement accumulator with one draft added or merged."""
        replacement = list(self.entries)
        new_key = _draft_row_key(draft)
        for index, existing in enumerate(replacement):
            if _draft_row_key(existing) == new_key:
                replacement[index] = _merge_drafts(existing, draft)
                break
        else:
            replacement.append(draft)
        replacement.sort(key=_draft_row_key)
        return SufficientStatisticAccumulatorV1(entries=tuple(replacement))

    def merge(
        self,
        other: SufficientStatisticAccumulatorV1,
    ) -> SufficientStatisticAccumulatorV1:
        """Return a replacement containing every compatible entry in ``other``."""
        replacement = self
        for draft in other.entries:
            replacement = replacement.add(draft)
        return replacement


class EvaluationMetricReducerStateV1(EvaluationModel):
    """Frozen identity-bearing base for trusted reducer replacement state."""

    reducer_id: _AsciiIdentifier
    reducer_version: _PositiveInt


class EvaluationEpisodeCompletionV1(EvaluationModel):
    """Rollout completion truth, independent of host metric processing."""

    schema_id: Literal["marl_battlegrounds.evaluation.episode_completion"] = (
        EPISODE_COMPLETION_SCHEMA_ID
    )
    schema_version: _Cp3SchemaVersion = CP3_SCHEMA_VERSION
    episode_id: _AsciiIdentifier
    completion_state: CompletionState
    expected_transition_count: _PositiveInt
    validated_transition_count: _NonNegativeInt
    last_valid_frame_index: _NonNegativeInt
    last_valid_frame_id: _AsciiIdentifier
    terminated: bool
    truncated: bool
    completion_bases: tuple[CompletionBasis, ...]
    end_or_failure_reason: _AsciiText | None = None
    failure_origin: RolloutFailureOrigin | None = None

    @model_validator(mode="after")
    def _validate_completion(self) -> EvaluationEpisodeCompletionV1:
        if self.validated_transition_count > self.expected_transition_count:
            raise ValueError("validated transitions cannot exceed the declared horizon")
        if self.validated_transition_count == 0 and (self.terminated or self.truncated):
            raise ValueError(
                "zero-transition completion cannot carry transition done flags"
            )
        if self.last_valid_frame_index != self.validated_transition_count:
            raise ValueError(
                "last valid frame index must equal validated transition count"
            )
        expected_frame_id = f"{self.episode_id}:frame:{self.validated_transition_count}"
        if self.last_valid_frame_id != expected_frame_id:
            raise ValueError("last valid frame ID is not canonical")
        expected_bases: list[CompletionBasis] = []
        if self.terminated:
            expected_bases.append("task_terminal")
        if self.validated_transition_count == self.expected_transition_count:
            expected_bases.append("declared_horizon")
        if self.completion_state == "complete":
            if not expected_bases:
                raise ValueError(
                    "complete rollout requires task termination or declared horizon"
                )
            if self.completion_bases != tuple(expected_bases):
                raise ValueError(
                    "completion bases must exactly preserve terminal/horizon evidence"
                )
            if self.failure_origin is not None:
                raise ValueError("complete rollout forbids a rollout failure origin")
        else:
            if expected_bases:
                raise ValueError(
                    "terminal or horizon-complete rollout must be labeled complete"
                )
            if self.completion_bases:
                raise ValueError("non-complete rollout forbids completion bases")
            if self.end_or_failure_reason is None:
                raise ValueError(
                    "partial, interrupted, and failed rollouts require a reason"
                )
            if self.completion_state == "failed":
                if self.failure_origin is None:
                    raise ValueError("failed rollout requires a failure origin")
            elif self.failure_origin is not None:
                raise ValueError(
                    "only failed rollout completion may carry a failure origin"
                )
        return self


class EvaluationProcessingFailureV1(EvaluationModel):
    """One stable failure of host observation or metric processing."""

    stage: ProcessingFailureStage
    code: _AsciiIdentifier
    reducer_id: _AsciiIdentifier | None = None
    reducer_version: _PositiveInt | None = None
    attempted_transition_index: _NonNegativeInt | None = None
    detail: _AsciiText

    @model_validator(mode="after")
    def _validate_reducer_identity(self) -> EvaluationProcessingFailureV1:
        if (self.reducer_id is None) != (self.reducer_version is None):
            raise ValueError(
                "processing failure reducer ID and version must appear together"
            )
        return self


class EvaluationProcessingStatusV1(EvaluationModel):
    """Metric-processing progress, separate from physical rollout completion."""

    schema_id: Literal["marl_battlegrounds.evaluation.processing_status"] = (
        PROCESSING_STATUS_SCHEMA_ID
    )
    schema_version: _Cp3SchemaVersion = CP3_SCHEMA_VERSION
    status: ProcessingState
    processed_transition_count: _NonNegativeInt
    failure: EvaluationProcessingFailureV1 | None = None

    @model_validator(mode="after")
    def _validate_status(self) -> EvaluationProcessingStatusV1:
        if self.failure is not None:
            _require_stable_nested_model(
                self.failure,
                record_name="processing failure",
                expected_types=(EvaluationProcessingFailureV1,),
            )
        if self.status == "succeeded":
            if self.failure is not None:
                raise ValueError("successful processing forbids a failure record")
        elif self.failure is None:
            raise ValueError("failed processing requires a failure record")
        return self


_STATUS_PRECEDENCE: dict[StatisticResultStatus, int] = {
    "invalid_artifact": 0,
    "structurally_inapplicable": 1,
    "ambiguous_attribution": 2,
    "insufficient_data": 3,
    "zero_opportunity": 4,
    "defined": 5,
}

_FAILURE_STAGES_WITHOUT_STATISTICS = frozenset(
    {
        "initial_validation",
        "reducer_initialize",
        "completion_validation",
        "reducer_finalize",
        "statistic_materialization",
        "report_validation",
    }
)


def _validate_processing_progress(
    validated_transition_count: int,
    processing_status: EvaluationProcessingStatusV1,
) -> None:
    processed_transition_count = processing_status.processed_transition_count
    if processed_transition_count > validated_transition_count:
        raise ValueError("processed count cannot exceed validated count")
    if processing_status.status == "succeeded":
        if processed_transition_count != validated_transition_count:
            raise ValueError(
                "successful processing requires equal validated and processed counts"
            )
        return

    failure = processing_status.failure
    if failure is None:
        raise ValueError("failed processing requires its typed failure record")
    if failure.stage == "reducer_initialize":
        if validated_transition_count != 0 or processed_transition_count != 0:
            raise ValueError(
                "reducer initialization failure requires a zero-transition prefix"
            )
        return
    if failure.stage == "reducer_advance":
        if validated_transition_count != processed_transition_count + 1:
            raise ValueError(
                "reducer advance failure requires exactly one unprocessed "
                "validated transition"
            )
        return
    if processed_transition_count != validated_transition_count:
        raise ValueError(
            "this processing failure stage requires equal validated and "
            "processed counts"
        )


def _replace_draft(
    draft: SufficientStatisticDraftV1,
    **updates: object,
) -> SufficientStatisticDraftV1:
    payload = draft.model_dump(mode="python")
    payload.update(updates)
    return SufficientStatisticDraftV1.model_validate(payload)


def _with_episode_eligibility(
    draft: SufficientStatisticDraftV1,
    eligible_episode_count: _EpisodeEligibility,
) -> SufficientStatisticDraftV1:
    component = draft.component
    if component is None or component.eligible_episode_count == eligible_episode_count:
        return draft
    component_payload = component.model_dump(mode="python")
    component_payload["eligible_episode_count"] = eligible_episode_count
    replacement_component = type(component).model_validate(component_payload)
    return _replace_draft(draft, component=replacement_component)


def _apply_episode_eligibility(
    draft: SufficientStatisticDraftV1,
    completion: EvaluationEpisodeCompletionV1,
    processing_status: EvaluationProcessingStatusV1,
) -> SufficientStatisticDraftV1:
    endpoint = draft.endpoint_observation_status
    is_complete = completion.completion_state == "complete"
    reached_horizon = "declared_horizon" in completion.completion_bases
    processed_prefix_is_complete = (
        processing_status.processed_transition_count
        == completion.validated_transition_count
    )
    if endpoint == "right_censored" and (
        not is_complete or not reached_horizon or not draft.supports_right_censoring
    ):
        raise ValueError(
            "right censoring requires a complete declared horizon and "
            "censor-aware statistic"
        )
    if endpoint == "competing_event" and not is_complete:
        raise ValueError("competing-event endpoint requires a complete rollout")

    requires_complete = draft.completion_scope == "complete_episode"
    if requires_complete and (not is_complete or not processed_prefix_is_complete):
        if (
            _STATUS_PRECEDENCE[draft.result_status]
            > _STATUS_PRECEDENCE["insufficient_data"]
        ):
            draft = _replace_draft(
                draft,
                result_status="insufficient_data",
                status_reason=("complete episode and fully processed prefix required"),
                endpoint_observation_status="unavailable",
            )
        return _with_episode_eligibility(draft, 0)

    if draft.result_status in (
        "invalid_artifact",
        "structurally_inapplicable",
        "ambiguous_attribution",
        "insufficient_data",
    ):
        return _with_episode_eligibility(draft, 0)

    component = draft.component
    if (
        draft.result_status == "defined"
        and component is not None
        and component.eligible_episode_count == 0
    ):
        raise ValueError("defined final statistics require one eligible episode")
    if (
        draft.result_status == "defined"
        and isinstance(component, RatioComponentV1)
        and component.zero_opportunity_occurrence == 1
    ):
        return _replace_draft(
            draft,
            result_status="zero_opportunity",
            status_reason="no genuine opportunities in eligible episode",
        )

    return draft


class RawSufficientStatisticV1(SufficientStatisticDraftV1):
    """One raw statistic plus observer-owned episode/progress provenance."""

    schema_id: Literal["marl_battlegrounds.evaluation.raw_sufficient_statistic"] = (
        RAW_SUFFICIENT_STATISTIC_SCHEMA_ID
    )
    schema_version: _Cp3SchemaVersion = CP3_SCHEMA_VERSION
    episode_id: _AsciiIdentifier
    aggregation_keys: tuple[AggregationKeyV1, ...]
    source_schema_versions: tuple[SchemaVersionEntryV1, ...]
    rollout_completion: EvaluationEpisodeCompletionV1
    validated_transition_count: _NonNegativeInt
    processed_transition_count: _NonNegativeInt
    processing_status: EvaluationProcessingStatusV1

    @model_validator(mode="after")
    def _validate_provenance(self) -> RawSufficientStatisticV1:
        for aggregation_key in self.aggregation_keys:
            _require_stable_nested_model(
                aggregation_key,
                record_name="raw statistic aggregation key",
                expected_types=(AggregationKeyV1,),
            )
        for schema_binding in self.source_schema_versions:
            _require_stable_nested_model(
                schema_binding,
                record_name="raw statistic source schema binding",
                expected_types=(SchemaVersionEntryV1,),
            )
        _require_stable_nested_model(
            self.rollout_completion,
            record_name="raw statistic completion",
            expected_types=(EvaluationEpisodeCompletionV1,),
        )
        _require_stable_nested_model(
            self.processing_status,
            record_name="raw statistic processing status",
            expected_types=(EvaluationProcessingStatusV1,),
        )
        aggregation_names = tuple(row.name for row in self.aggregation_keys)
        if aggregation_names != tuple(sorted(aggregation_names)):
            raise ValueError("raw statistic aggregation keys must be sorted")
        if len(aggregation_names) != len(set(aggregation_names)):
            raise ValueError("raw statistic aggregation keys must be unique")
        source_bindings = tuple(
            (row.schema_id, row.schema_version) for row in self.source_schema_versions
        )
        if source_bindings != REQUIRED_SCHEMA_BINDINGS_V1:
            raise ValueError("raw statistic must bind the exact eight CP2 V1 roots")
        dimension_names = {row.name for row in self.dimensions}
        shadowed_names = dimension_names.intersection(aggregation_names)
        if shadowed_names:
            raise ValueError(
                "statistic dimensions cannot shadow context aggregation keys: "
                f"{', '.join(sorted(shadowed_names))}"
            )
        if self.episode_id != self.rollout_completion.episode_id:
            raise ValueError("raw statistic must join its rollout completion")
        if (
            self.validated_transition_count
            != self.rollout_completion.validated_transition_count
        ):
            raise ValueError("raw statistic validated count must match completion")
        if (
            self.processed_transition_count
            != self.processing_status.processed_transition_count
        ):
            raise ValueError("raw statistic processed count must match processing")
        _validate_processing_progress(
            self.validated_transition_count,
            self.processing_status,
        )
        if (
            self.processing_status.failure is not None
            and self.processing_status.failure.stage
            in _FAILURE_STAGES_WITHOUT_STATISTICS
        ):
            raise ValueError(
                "this processing failure stage cannot publish a raw statistic"
            )
        draft = SufficientStatisticDraftV1.model_validate(
            {
                field_name: getattr(self, field_name)
                for field_name in SufficientStatisticDraftV1.model_fields
            }
        )
        normalized = _apply_episode_eligibility(
            draft,
            self.rollout_completion,
            self.processing_status,
        )
        if normalized != draft:
            raise ValueError(
                "raw statistic result and endpoint status must reflect episode "
                "and processing eligibility"
            )
        return self


def _raw_row_key(row: RawSufficientStatisticV1) -> tuple[object, ...]:
    return _draft_row_key(row)


def _validate_subject_join(
    context: EvaluationEpisodeContextV1,
    row: RawSufficientStatisticV1,
) -> None:
    subject = row.subject
    if isinstance(subject, EpisodeStatisticSubjectV1):
        return
    if isinstance(subject, TeamStatisticSubjectV1):
        return
    if isinstance(subject, AgentStatisticSubjectV1):
        slots = (subject.global_slot,)
    elif isinstance(subject, AgentPairStatisticSubjectV1):
        slots = (subject.primary_global_slot, subject.secondary_global_slot)
    else:
        class_is_present = any(
            roster_row.configured_active
            and roster_row.configured_team_id == subject.team_id
            and roster_row.class_id == subject.class_id
            for roster_row in context.roster
        )
        if not class_is_present and (
            _STATUS_PRECEDENCE[row.result_status]
            > _STATUS_PRECEDENCE["structurally_inapplicable"]
        ):
            raise ValueError(
                "absent team/class subject must be invalid or structurally inapplicable"
            )
        return
    for global_slot in slots:
        roster_row = context.roster[global_slot]
        policy_row = context.policy_assignments[global_slot]
        if not roster_row.configured_active or not isinstance(
            policy_row, AssignedPolicySlotV1
        ):
            raise ValueError(
                "agent statistic subjects must join active assigned-policy slots"
            )


class EvaluationMetricReportV1(EvaluationModel):
    """Immutable CP3 metric result; trajectory persistence remains Step 6."""

    schema_id: Literal["marl_battlegrounds.evaluation.metric_report"] = (
        METRIC_REPORT_SCHEMA_ID
    )
    schema_version: _Cp3SchemaVersion = CP3_SCHEMA_VERSION
    report_id: _AsciiIdentifier
    context: EvaluationEpisodeContextV1
    completion: EvaluationEpisodeCompletionV1
    processing_status: EvaluationProcessingStatusV1
    statistics: tuple[RawSufficientStatisticV1, ...]

    @model_validator(mode="after")
    def _validate_report(self) -> EvaluationMetricReportV1:
        _require_stable_nested_model(
            self.context,
            record_name="metric report context",
            expected_types=(EvaluationEpisodeContextV1,),
        )
        _require_stable_nested_model(
            self.completion,
            record_name="metric report completion",
            expected_types=(EvaluationEpisodeCompletionV1,),
        )
        _require_stable_nested_model(
            self.processing_status,
            record_name="metric report processing status",
            expected_types=(EvaluationProcessingStatusV1,),
        )
        for statistic in self.statistics:
            _require_stable_nested_model(
                statistic,
                record_name="metric report raw statistic",
                expected_types=(RawSufficientStatisticV1,),
            )
        episode_id = self.context.identity.episode_id
        if self.report_id != f"{episode_id}:metric-report":
            raise ValueError("metric report ID is not canonical")
        if self.completion.episode_id != episode_id:
            raise ValueError("metric report completion must join the context episode")
        if self.completion.expected_transition_count != self.context.expected_horizon:
            raise ValueError("metric report completion must use the context horizon")
        _validate_processing_progress(
            self.completion.validated_transition_count,
            self.processing_status,
        )
        failure = self.processing_status.failure
        if failure is not None and failure.stage in (
            "initial_validation",
            "completion_validation",
        ):
            raise ValueError(
                "initial or completion validation failure cannot produce a "
                "metric report"
            )
        statistic_keys = tuple(_raw_row_key(row) for row in self.statistics)
        if statistic_keys != tuple(sorted(statistic_keys)):
            raise ValueError("metric report statistics must be canonically sorted")
        if len(statistic_keys) != len(set(statistic_keys)):
            raise ValueError("metric report statistics must have unique row keys")
        for row in self.statistics:
            if row.episode_id != episode_id:
                raise ValueError("raw statistic must join the report episode")
            if row.aggregation_keys != self.context.aggregation_keys:
                raise ValueError("raw statistic aggregation keys must equal context")
            if row.source_schema_versions != self.context.schema_versions:
                raise ValueError("raw statistic source schemas must equal context")
            if row.rollout_completion != self.completion:
                raise ValueError(
                    "raw statistic completion must equal report completion"
                )
            if row.processing_status != self.processing_status:
                raise ValueError(
                    "raw statistic processing must equal report processing"
                )
            _validate_subject_join(self.context, row)
        return self


@dataclass(frozen=True, slots=True)
class EvaluationTransitionViewV1:
    """One fully validated context/start/transition/successor consumer view."""

    context: EvaluationEpisodeContextV1
    start_frame: EvaluationFrameV1
    transition: EvaluationTransitionV1
    successor_frame: EvaluationFrameV1

    def __post_init__(self) -> None:
        validate_evaluation_transition_unit_v1(
            self.context,
            self.start_frame,
            self.transition,
            self.successor_frame,
        )
        canonical_context = cast(
            EvaluationEpisodeContextV1,
            validate_declared_model_tree(
                self.context,
                record_name="transition view context",
                expected_type=EvaluationEpisodeContextV1,
            ),
        )
        canonical_start = cast(
            EvaluationFrameV1,
            validate_declared_model_tree(
                self.start_frame,
                record_name="transition view start frame",
                expected_type=EvaluationFrameV1,
            ),
        )
        canonical_transition = cast(
            EvaluationTransitionV1,
            validate_declared_model_tree(
                self.transition,
                record_name="transition view transition",
                expected_type=EvaluationTransitionV1,
            ),
        )
        canonical_successor = cast(
            EvaluationFrameV1,
            validate_declared_model_tree(
                self.successor_frame,
                record_name="transition view successor frame",
                expected_type=EvaluationFrameV1,
            ),
        )
        object.__setattr__(self, "context", canonical_context)
        object.__setattr__(self, "start_frame", canonical_start)
        object.__setattr__(self, "transition", canonical_transition)
        object.__setattr__(self, "successor_frame", canonical_successor)


class EvaluationMetricReducerV1(Protocol):
    """Trusted deterministic reducer using explicit frozen replacement state.

    The observer can make state replacement atomic, but Python cannot undo a
    plugin that violates this protocol by mutating external state. Reducer
    implementations therefore must not use mutable internals, RNG, clocks,
    global discovery, files, network access, logging, or callbacks.
    """

    reducer_id: str
    reducer_version: int

    def initialize(
        self,
        context: EvaluationEpisodeContextV1,
        initial_frame: EvaluationFrameV1,
    ) -> EvaluationMetricReducerStateV1: ...

    def advance(
        self,
        previous_state: EvaluationMetricReducerStateV1,
        view: EvaluationTransitionViewV1,
    ) -> EvaluationMetricReducerStateV1: ...

    def finalize(
        self,
        state: EvaluationMetricReducerStateV1,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]: ...


def _ascii_failure_detail(error: BaseException) -> str:
    try:
        raw_detail = str(error)
    except Exception:
        raw_detail = type(error).__name__
    collapsed = " ".join(raw_detail.split()) or type(error).__name__
    ascii_detail = collapsed.encode("ascii", errors="backslashreplace").decode("ascii")
    return "".join(
        character if 0x20 <= ord(character) <= 0x7E else f"\\x{ord(character):02x}"
        for character in ascii_detail
    )


def _validate_reducer_registration(
    reducer: EvaluationMetricReducerV1,
) -> tuple[str, int]:
    reducer_id = reducer.reducer_id
    reducer_version = reducer.reducer_version
    if (
        type(reducer_id) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+\-]*", reducer_id) is None
    ):
        raise ValueError("reducer_id must use the durable ASCII identifier vocabulary")
    if type(reducer_version) is not int or reducer_version <= 0:
        raise ValueError("reducer_version must be a positive strict integer")
    return reducer_id, reducer_version


def _uses_strict_frozen_model_config(model_type: type[EvaluationModel]) -> bool:
    model_config = model_type.model_config
    return (
        model_config.get("allow_inf_nan") is False
        and model_config.get("extra") == "forbid"
        and model_config.get("frozen") is True
        and model_config.get("strict") is True
    )


def _revalidate_reducer_state(
    state: object,
    *,
    reducer_id: str,
    reducer_version: int,
    expected_type: type[EvaluationMetricReducerStateV1] | None,
) -> EvaluationMetricReducerStateV1:
    if not isinstance(state, EvaluationMetricReducerStateV1):
        raise TypeError("reducers must return EvaluationMetricReducerStateV1")
    if not _uses_strict_frozen_model_config(type(state)):
        raise TypeError(
            "reducer state subclasses must retain the strict frozen model config"
        )
    reconstructed = type(state).model_validate(state.model_dump(mode="python"))
    if reconstructed != state:
        raise ValueError("reducer state changes under structural revalidation")
    if not _is_frozen_state_value(state):
        raise TypeError("reducer state fields must be scalar or tuple-backed models")
    if state.reducer_id != reducer_id or state.reducer_version != reducer_version:
        raise ValueError("reducer state identity/version must match its reducer")
    if expected_type is not None and type(state) is not expected_type:
        raise TypeError("reducer replacement state type must remain exact")
    return state


def _is_frozen_state_value(value: object) -> bool:
    """Return whether reducer state recursively excludes mutable/device values."""
    if isinstance(value, EvaluationModel):
        if not _uses_strict_frozen_model_config(type(value)):
            return False
        # Pydantic private attributes are deliberately absent from model dumps
        # and equality.  Allowing them would let a nominally frozen reducer
        # state retain hidden mutable data outside the replacement-state
        # transaction, so reducer state models may not declare them at all.
        if getattr(type(value), "__private_attributes__", {}):
            return False
        if getattr(value, "__pydantic_private__", None):
            return False
        return all(
            _is_frozen_state_value(getattr(value, field_name))
            for field_name in type(value).model_fields
        )
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return all(_is_frozen_state_value(item) for item in items)
    return value is None or type(value) in (bool, int, float, str)


def _materialize_raw_statistic(
    context: EvaluationEpisodeContextV1,
    draft: SufficientStatisticDraftV1,
    completion: EvaluationEpisodeCompletionV1,
    processing_status: EvaluationProcessingStatusV1,
) -> RawSufficientStatisticV1:
    eligible = _apply_episode_eligibility(draft, completion, processing_status)
    payload = eligible.model_dump(mode="python")
    return RawSufficientStatisticV1.model_validate(
        {
            **payload,
            "episode_id": context.identity.episode_id,
            "aggregation_keys": context.aggregation_keys,
            "source_schema_versions": context.schema_versions,
            "rollout_completion": completion,
            "validated_transition_count": completion.validated_transition_count,
            "processed_transition_count": (
                processing_status.processed_transition_count
            ),
            "processing_status": processing_status,
        }
    )


class EvaluationEpisodeObserverV1:
    """Mutable transaction coordinator around immutable host evaluation records."""

    __slots__ = (
        "_context",
        "_current_frame",
        "_finalize_attempted",
        "_last_transition",
        "_lifecycle_state",
        "_processed_transition_count",
        "_processing_failure",
        "_reducer_state_types",
        "_reducer_states",
        "_reducers",
        "_retained_frames",
        "_retained_transitions",
        "_validated_transition_count",
    )

    def __init__(
        self,
        context: EvaluationEpisodeContextV1,
        reducers: tuple[EvaluationMetricReducerV1, ...] = (),
    ) -> None:
        if type(context) is not EvaluationEpisodeContextV1:
            raise ValueError(
                "observer context must use exact declared root type "
                "EvaluationEpisodeContextV1"
            )
        reconstructed_context = cast(
            EvaluationEpisodeContextV1,
            validate_declared_model_tree(
                context,
                record_name="observer context",
                expected_type=EvaluationEpisodeContextV1,
            ),
        )
        if type(reducers) is not tuple:
            raise TypeError("reducers must be supplied as an immutable tuple")
        ordered = tuple(sorted(reducers, key=_validate_reducer_registration))
        reducer_ids = tuple(reducer.reducer_id for reducer in ordered)
        if len(reducer_ids) != len(set(reducer_ids)):
            raise ValueError("observer reducer IDs must be unique")

        self._context = reconstructed_context
        self._reducers = ordered
        self._lifecycle_state: ObserverLifecycleState = "awaiting_initial"
        self._current_frame: EvaluationFrameV1 | None = None
        self._finalize_attempted = False
        self._last_transition: EvaluationTransitionV1 | None = None
        self._validated_transition_count = 0
        self._processed_transition_count = 0
        self._processing_failure: EvaluationProcessingFailureV1 | None = None
        self._reducer_states: tuple[EvaluationMetricReducerStateV1, ...] | None = None
        self._reducer_state_types: (
            tuple[type[EvaluationMetricReducerStateV1], ...] | None
        ) = None
        retain_trajectory = reconstructed_context.capture_profile in (
            "evaluation_metric_complete",
            "scenario_metric_complete",
        )
        self._retained_frames: list[EvaluationFrameV1] | None = (
            [] if retain_trajectory else None
        )
        self._retained_transitions: list[EvaluationTransitionV1] | None = (
            [] if retain_trajectory else None
        )

    @property
    def context(self) -> EvaluationEpisodeContextV1:
        """Return the immutable episode context owned by this observer."""
        return self._context

    @property
    def lifecycle_state(self) -> ObserverLifecycleState:
        """Return the current observer lifecycle state."""
        return self._lifecycle_state

    @property
    def validated_transition_count(self) -> int:
        """Return the count of semantically validated CP2 transition units."""
        return self._validated_transition_count

    @property
    def processed_transition_count(self) -> int:
        """Return the count atomically consumed by every reducer."""
        return self._processed_transition_count

    @property
    def retained_frames(self) -> tuple[EvaluationFrameV1, ...] | None:
        """Return metric-complete frame history, or ``None`` when not retained."""
        if self._retained_frames is None:
            return None
        return tuple(self._retained_frames)

    @property
    def retained_transitions(self) -> tuple[EvaluationTransitionV1, ...] | None:
        """Return metric-complete transition history, or ``None`` otherwise."""
        if self._retained_transitions is None:
            return None
        return tuple(self._retained_transitions)

    @property
    def reducer_states(self) -> tuple[EvaluationMetricReducerStateV1, ...] | None:
        """Return the last atomically committed frozen reducer states."""
        return self._reducer_states

    def _set_failure(
        self,
        *,
        stage: ProcessingFailureStage,
        code: str,
        detail: str,
        reducer: EvaluationMetricReducerV1 | None = None,
        attempted_transition_index: int | None = None,
        replace: bool = False,
    ) -> None:
        if self._processing_failure is not None:
            if replace:
                first = self._processing_failure
                self._processing_failure = EvaluationProcessingFailureV1.model_validate(
                    {
                        **first.model_dump(mode="python"),
                        "detail": f"{first.detail}; secondary {stage}/{code}: {detail}",
                    }
                )
            return
        self._processing_failure = EvaluationProcessingFailureV1(
            stage=stage,
            code=code,
            reducer_id=None if reducer is None else reducer.reducer_id,
            reducer_version=None if reducer is None else reducer.reducer_version,
            attempted_transition_index=attempted_transition_index,
            detail=detail,
        )

    def _reject_lifecycle(self, operation: str) -> None:
        previous = self._lifecycle_state
        if previous != "finalized":
            self._set_failure(
                stage="lifecycle",
                code=f"{operation}_not_allowed",
                detail=f"{operation} is not allowed while observer is {previous}",
            )
            self._lifecycle_state = "poisoned"
        raise RuntimeError(f"{operation} is not allowed while observer is {previous}")

    def start(self, initial_frame: EvaluationFrameV1) -> None:
        """Accept frame zero and atomically initialize every reducer once."""
        if self._lifecycle_state != "awaiting_initial":
            self._reject_lifecycle("start")
        try:
            validate_initial_evaluation_frame_v1(self._context, initial_frame)
            canonical_initial_frame = cast(
                EvaluationFrameV1,
                validate_declared_model_tree(
                    initial_frame,
                    record_name="observer initial frame",
                    expected_type=EvaluationFrameV1,
                ),
            )
        except Exception as error:
            self._set_failure(
                stage="initial_validation",
                code="invalid_initial_frame",
                detail=_ascii_failure_detail(error),
            )
            self._lifecycle_state = "poisoned"
            raise

        self._current_frame = canonical_initial_frame
        if self._retained_frames is not None:
            self._retained_frames.append(canonical_initial_frame)

        candidate_states: list[EvaluationMetricReducerStateV1] = []
        candidate_types: list[type[EvaluationMetricReducerStateV1]] = []
        for reducer in self._reducers:
            try:
                candidate = _revalidate_reducer_state(
                    reducer.initialize(self._context, canonical_initial_frame),
                    reducer_id=reducer.reducer_id,
                    reducer_version=reducer.reducer_version,
                    expected_type=None,
                )
            except Exception as error:
                self._set_failure(
                    stage="reducer_initialize",
                    code="reducer_initialize_failed",
                    detail=_ascii_failure_detail(error),
                    reducer=reducer,
                )
                self._lifecycle_state = "poisoned"
                raise RuntimeError(
                    f"reducer {reducer.reducer_id} initialization failed"
                ) from error
            candidate_states.append(candidate)
            candidate_types.append(type(candidate))
        self._reducer_states = tuple(candidate_states)
        self._reducer_state_types = tuple(candidate_types)
        self._lifecycle_state = "open"

    def append(
        self,
        transition: EvaluationTransitionV1,
        successor_frame: EvaluationFrameV1,
    ) -> None:
        """Validate one unit, then atomically replace every reducer state."""
        if self._lifecycle_state != "open":
            self._reject_lifecycle("append")
        if self._current_frame is None:
            raise RuntimeError("open observer is missing its current frame")
        attempted_index: int | None = None
        try:
            raw_attempted_index = getattr(transition, "transition_index", None)
            attempted_index = (
                raw_attempted_index
                if type(raw_attempted_index) is int and raw_attempted_index >= 0
                else None
            )
            view = EvaluationTransitionViewV1(
                context=self._context,
                start_frame=self._current_frame,
                transition=transition,
                successor_frame=successor_frame,
            )
        except Exception as error:
            self._set_failure(
                stage="transition_validation",
                code="invalid_transition_unit",
                detail=_ascii_failure_detail(error),
                attempted_transition_index=attempted_index,
            )
            self._lifecycle_state = "poisoned"
            raise

        # Validation is authoritative physical/artifact evidence and commits
        # independently from reducer progress.
        self._current_frame = view.successor_frame
        self._last_transition = view.transition
        self._validated_transition_count += 1
        if self._retained_transitions is not None:
            self._retained_transitions.append(view.transition)
        if self._retained_frames is not None:
            self._retained_frames.append(view.successor_frame)

        states = self._reducer_states
        state_types = self._reducer_state_types
        if states is None or state_types is None:
            raise RuntimeError("open observer is missing initialized reducer states")
        candidates: list[EvaluationMetricReducerStateV1] = []
        for reducer, previous_state, expected_type in zip(
            self._reducers,
            states,
            state_types,
            strict=True,
        ):
            try:
                candidate = _revalidate_reducer_state(
                    reducer.advance(previous_state, view),
                    reducer_id=reducer.reducer_id,
                    reducer_version=reducer.reducer_version,
                    expected_type=expected_type,
                )
            except Exception as error:
                self._set_failure(
                    stage="reducer_advance",
                    code="reducer_advance_failed",
                    detail=_ascii_failure_detail(error),
                    reducer=reducer,
                    attempted_transition_index=attempted_index,
                )
                self._lifecycle_state = "poisoned"
                raise RuntimeError(
                    f"reducer {reducer.reducer_id} advance failed"
                ) from error
            candidates.append(candidate)
        self._reducer_states = tuple(candidates)
        self._processed_transition_count += 1
        if (
            view.transition.terminated
            or view.transition.truncated
            or self._validated_transition_count == self._context.expected_horizon
        ):
            self._lifecycle_state = "sealed"

    def _build_completion(
        self,
        *,
        completion_state: CompletionState,
        end_or_failure_reason: str | None,
        failure_origin: RolloutFailureOrigin | None,
    ) -> EvaluationEpisodeCompletionV1:
        if self._current_frame is None:
            raise RuntimeError("a valid initial frame is required before finalization")
        terminated = (
            False if self._last_transition is None else self._last_transition.terminated
        )
        truncated = (
            False if self._last_transition is None else self._last_transition.truncated
        )
        bases: list[CompletionBasis] = []
        if terminated:
            bases.append("task_terminal")
        if self._validated_transition_count == self._context.expected_horizon:
            bases.append("declared_horizon")
        authoritative_end_reason = (
            None
            if self._last_transition is None
            else self._last_transition.owning_task_end_reason
        )
        if authoritative_end_reason is not None:
            if (
                end_or_failure_reason is not None
                and end_or_failure_reason != authoritative_end_reason
            ):
                raise ValueError(
                    "completion reason must agree with authoritative task end reason"
                )
            end_or_failure_reason = authoritative_end_reason
        return EvaluationEpisodeCompletionV1(
            episode_id=self._context.identity.episode_id,
            completion_state=completion_state,
            expected_transition_count=self._context.expected_horizon,
            validated_transition_count=self._validated_transition_count,
            last_valid_frame_index=self._current_frame.frame_index,
            last_valid_frame_id=self._current_frame.frame_id,
            terminated=terminated,
            truncated=truncated,
            completion_bases=tuple(bases),
            end_or_failure_reason=end_or_failure_reason,
            failure_origin=failure_origin,
        )

    def _processing_status(self) -> EvaluationProcessingStatusV1:
        if self._processing_failure is None:
            if self._processed_transition_count != self._validated_transition_count:
                self._set_failure(
                    stage="statistic_materialization",
                    code="processing_progress_mismatch",
                    detail=(
                        "processed and validated progress diverged without a failure"
                    ),
                )
            else:
                return EvaluationProcessingStatusV1(
                    status="succeeded",
                    processed_transition_count=self._processed_transition_count,
                )
        return EvaluationProcessingStatusV1(
            status="failed",
            processed_transition_count=self._processed_transition_count,
            failure=self._processing_failure,
        )

    def finalize(
        self,
        *,
        completion_state: CompletionState,
        end_or_failure_reason: str | None = None,
        failure_origin: RolloutFailureOrigin | None = None,
    ) -> EvaluationMetricReportV1:
        """Atomically finalize raw rows and return the only CP3 report seam."""
        if self._lifecycle_state in ("awaiting_initial", "finalized"):
            self._reject_lifecycle("finalize")
        if self._current_frame is None:
            raise RuntimeError(
                "invalid initial frame cannot produce an observer report"
            )
        if self._finalize_attempted:
            self._reject_lifecycle("finalize")
        self._finalize_attempted = True
        try:
            completion = self._build_completion(
                completion_state=completion_state,
                end_or_failure_reason=end_or_failure_reason,
                failure_origin=failure_origin,
            )
        except Exception as error:
            self._set_failure(
                stage="completion_validation",
                code="invalid_completion_request",
                detail=_ascii_failure_detail(error),
            )
            self._lifecycle_state = "poisoned"
            raise

        processing_status = self._processing_status()
        drafts: tuple[SufficientStatisticDraftV1, ...] = ()
        states = self._reducer_states
        if states is not None:
            collected: list[SufficientStatisticDraftV1] = []
            finalization_failed = False
            for reducer, state in zip(self._reducers, states, strict=True):
                try:
                    reducer_drafts = reducer.finalize(
                        state,
                        completion,
                        processing_status,
                    )
                    if type(reducer_drafts) is not tuple:
                        raise TypeError(
                            "reducer finalize must return an immutable tuple"
                        )
                    for draft in reducer_drafts:
                        reconstructed = SufficientStatisticDraftV1.model_validate(
                            draft.model_dump(mode="python")
                        )
                        if reconstructed != draft:
                            raise ValueError(
                                "statistic draft changes under structural revalidation"
                            )
                        if (
                            draft.reducer_id != reducer.reducer_id
                            or draft.reducer_version != reducer.reducer_version
                        ):
                            raise ValueError(
                                "statistic draft reducer identity/version mismatch"
                            )
                        collected.append(draft)
                except Exception as error:
                    self._set_failure(
                        stage="reducer_finalize",
                        code="reducer_finalize_failed",
                        detail=_ascii_failure_detail(error),
                        reducer=reducer,
                        replace=True,
                    )
                    finalization_failed = True
                    break
            if not finalization_failed:
                drafts = tuple(collected)

        processing_status = self._processing_status()
        rows: tuple[RawSufficientStatisticV1, ...] = ()
        if states is not None and (
            self._processing_failure is None
            or self._processing_failure.stage != "reducer_finalize"
        ):
            try:
                materialized = tuple(
                    _materialize_raw_statistic(
                        self._context,
                        draft,
                        completion,
                        processing_status,
                    )
                    for draft in drafts
                )
                keys = tuple(_raw_row_key(row) for row in materialized)
                if len(keys) != len(set(keys)):
                    raise ValueError("reducers produced duplicate statistic row keys")
                rows = tuple(sorted(materialized, key=_raw_row_key))
            except Exception as error:
                self._set_failure(
                    stage="statistic_materialization",
                    code="statistic_materialization_failed",
                    detail=_ascii_failure_detail(error),
                    replace=True,
                )
                processing_status = self._processing_status()
                rows = ()

        try:
            report = EvaluationMetricReportV1(
                report_id=f"{self._context.identity.episode_id}:metric-report",
                context=self._context,
                completion=completion,
                processing_status=processing_status,
                statistics=rows,
            )
        except Exception as error:
            self._set_failure(
                stage="report_validation",
                code="metric_report_validation_failed",
                detail=_ascii_failure_detail(error),
                replace=True,
            )
            processing_status = self._processing_status()
            report = EvaluationMetricReportV1(
                report_id=f"{self._context.identity.episode_id}:metric-report",
                context=self._context,
                completion=completion,
                processing_status=processing_status,
                statistics=(),
            )
        self._lifecycle_state = "finalized"
        return report


def build_evaluation_observer_v1(
    context: EvaluationEpisodeContextV1,
    reducers: tuple[EvaluationMetricReducerV1, ...] = (),
) -> EvaluationEpisodeObserverV1:
    """Build one explicitly enabled observer; disabled callers retain ``None``."""
    return EvaluationEpisodeObserverV1(context=context, reducers=reducers)


__all__ = [
    "AgentPairStatisticSubjectV1",
    "AgentStatisticSubjectV1",
    "CompletionBasis",
    "CompletionState",
    "CountComponentV1",
    "DistributionComponentV1",
    "DistributionObservationV1",
    "DurationComponentV1",
    "EndpointObservationStatus",
    "EpisodeStatisticSubjectV1",
    "EvaluationEpisodeCompletionV1",
    "EvaluationEpisodeObserverV1",
    "EvaluationMetricReducerStateV1",
    "EvaluationMetricReducerV1",
    "EvaluationMetricReportV1",
    "EvaluationProcessingFailureV1",
    "EvaluationProcessingStatusV1",
    "EvaluationTransitionViewV1",
    "HealthAmountStage",
    "ObserverLifecycleState",
    "OpportunityComponentV1",
    "ProcessingFailureStage",
    "ProcessingState",
    "RatioComponentV1",
    "RawSufficientStatisticV1",
    "RolloutFailureOrigin",
    "StatisticCompletionScope",
    "StatisticDimensionV1",
    "StatisticResultStatus",
    "StatisticSubjectV1",
    "SufficientStatisticAccumulatorV1",
    "SufficientStatisticComponentV1",
    "SufficientStatisticDraftV1",
    "SumComponentV1",
    "TeamClassStatisticSubjectV1",
    "TeamStatisticSubjectV1",
    "build_evaluation_observer_v1",
]
