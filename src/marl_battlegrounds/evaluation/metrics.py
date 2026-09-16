"""Define historical metric records and the validated host observer lifecycle.

V1 reports store raw counts, sums, ratios, durations, opportunities, and samples
with explicit episode eligibility. The observer keeps valid captured trajectory
progress separate from reducer progress, so a metric failure cannot erase a
valid transition. It serves historical recording and debugger analysis; current
high-throughput metrics use the numerical evaluation workflow.

All models and reducers here run on the host. Reducers are trusted deterministic
code with immutable replacement state. Their raw statistics are analysis data,
not additional information an acting policy may consume.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from types import UnionType
from typing import (
    Annotated,
    Literal,
    Protocol,
    TypeAliasType,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import BeforeValidator, Field, StringConstraints, model_validator

from marl_battlegrounds.evaluation.models import (
    REQUIRED_SCHEMA_BINDINGS_V1,
    AggregationKeyV1,
    AssignedPolicySlotV1,
    EvaluationEpisodeContext,
    EvaluationEpisodeContextV1,
    EvaluationEpisodeContextV2,
    EvaluationEpisodeContextV3,
    EvaluationFrame,
    EvaluationFrameV1,
    EvaluationFrameV2,
    EvaluationModel,
    EvaluationTransitionV1,
    SchemaVersionEntryV1,
)
from marl_battlegrounds.evaluation.validation import (
    _canonicalize_evaluation_transition_unit_v1,  # pyright: ignore[reportPrivateUsage]
    validate_declared_model_tree,
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
    "offline_reducer_initialize",
    "offline_reducer_advance",
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
    """Return exact integer 0 or 1; reject bool and other values with ValueError."""
    if type(value) is not int or value not in (0, 1):
        raise ValueError("value must be an exact integer 0 or 1")
    return value


def _require_schema_version_one(value: object) -> object:
    """Return exact integer 1; reject bool and other versions with ValueError."""
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
    """Add one metric-specific category without replacing context-owned facts.

    Attributes
    ----------
    name : _AsciiIdentifier
        Nonempty ASCII identifier absent from the reserved context/metric field names.
    value : _AsciiIdentifier
        Nonempty ASCII category identifier.

    Notes
    -----
    Drafts require dimensions to be sorted and unique by name. Raw rows also
    reject names used by the context's aggregation keys.
    """

    name: _AsciiIdentifier
    value: _AsciiIdentifier

    @model_validator(mode="after")
    def _reject_context_shadowing(self) -> StatisticDimensionV1:
        """Reject reserved dimension names with ValueError; return this row
        otherwise.
        """
        if self.name in _RESERVED_DIMENSION_NAMES:
            raise ValueError(f"statistic dimension {self.name!r} shadows context truth")
        return self


class EpisodeStatisticSubjectV1(EvaluationModel):
    """Select the whole episode as a statistic's subject.

    Attributes
    ----------
    subject_type : Literal['episode']
        Fixed discriminator "episode"; defaults to "episode".
    """

    subject_type: Literal["episode"] = "episode"


class TeamStatisticSubjectV1(EvaluationModel):
    """Select one configured team as a statistic's subject.

    Attributes
    ----------
    subject_type : Literal['team']
        Fixed discriminator "team"; defaults to "team".
    team_id : _TeamId
        1 for Team A or 2 for Team B.
    """

    subject_type: Literal["team"] = "team"
    team_id: _TeamId


class AgentStatisticSubjectV1(EvaluationModel):
    """Select one global agent slot as a statistic's subject.

    Attributes
    ----------
    subject_type : Literal['agent']
        Fixed discriminator "agent"; defaults to "agent".
    global_slot : _GlobalSlot
        Slot 0-9. Report validation additionally requires an active slot
        with an assigned V1 policy record.
    """

    subject_type: Literal["agent"] = "agent"
    global_slot: _GlobalSlot


class TeamClassStatisticSubjectV1(EvaluationModel):
    """Select one class within a team, including a declared absent class.

    Attributes
    ----------
    subject_type : Literal['team_class']
        Fixed discriminator "team_class"; defaults to "team_class".
    team_id : _TeamId
        1 for Team A or 2 for Team B.
    class_id : _ClassId
        Class 1-5. An absent class must be reported as invalid_artifact
        or structurally_inapplicable when joined to context.
    """

    subject_type: Literal["team_class"] = "team_class"
    team_id: _TeamId
    class_id: _ClassId


class AgentPairStatisticSubjectV1(EvaluationModel):
    """Select an ordered pair of distinct agent slots.

    Attributes
    ----------
    subject_type : Literal['agent_pair']
        Fixed discriminator "agent_pair"; defaults to "agent_pair".
    primary_global_slot : _GlobalSlot
        First slot, 0-9.
    secondary_global_slot : _GlobalSlot
        Different second slot, 0-9.

    Notes
    -----
    Pair order is meaningful. Report validation requires both slots active and assigned.
    """

    subject_type: Literal["agent_pair"] = "agent_pair"
    primary_global_slot: _GlobalSlot
    secondary_global_slot: _GlobalSlot

    @model_validator(mode="after")
    def _require_distinct_slots(self) -> AgentPairStatisticSubjectV1:
        """Reject a self-pair with ValueError; preserve the supplied order otherwise."""
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
    """Require an exact allowed model class and unchanged strict revalidation.

    Raise ValueError for undeclared subclasses or unchecked/coerced fields.
    Do not mutate or silently repair the supplied model.
    """
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
    """Keep one finite sample and its stable source link.

    Attributes
    ----------
    source_observation_id : _AsciiIdentifier
        Nonempty ASCII source identifier, unique within its distribution.
    ordinal : _NonNegativeInt
        Zero-based position, required gap-free by the owning distribution.
    value : _FiniteFloat
        Finite sample value in the owning statistic's units.
    """

    source_observation_id: _AsciiIdentifier
    ordinal: _NonNegativeInt
    value: _FiniteFloat


class CountComponentV1(EvaluationModel):
    """Store an additive count and whether this episode contributes to it.

    Attributes
    ----------
    component_type : Literal['count']
        Fixed discriminator "count"; defaults to "count".
    count : _NonNegativeInt
        Nonnegative number of qualifying events or rows.
    eligible_episode_count : _EpisodeEligibility
        Exact integer 0 or 1, not bool.

    Notes
    -----
    A zero count is a valid measured result and does not imply zero opportunity.
    """

    component_type: Literal["count"] = "count"
    count: _NonNegativeInt
    eligible_episode_count: _EpisodeEligibility


class SumComponentV1(EvaluationModel):
    """Store a finite sum with its observation and episode counts.

    Attributes
    ----------
    component_type : Literal['sum']
        Fixed discriminator "sum"; defaults to "sum".
    value : _FiniteFloat
        Finite additive total in the statistic's units; negative totals are allowed.
    observation_count : _NonNegativeInt
        Nonnegative number of contributing observations.
    eligible_episode_count : _EpisodeEligibility
        Exact integer 0 or 1.

    Notes
    -----
    Zero observations require a zero sum. Empty eligible sums carry zero-opportunity
    evidence rather than an invented observation.
    """

    component_type: Literal["sum"] = "sum"
    value: _FiniteFloat
    observation_count: _NonNegativeInt
    eligible_episode_count: _EpisodeEligibility

    @model_validator(mode="after")
    def _validate_empty_sum(self) -> SumComponentV1:
        """Require a zero total when observation_count is zero; otherwise raise
        ValueError.
        """
        if self.observation_count == 0 and self.value != 0.0:
            raise ValueError("zero observations require a zero sum")
        return self


class RatioComponentV1(EvaluationModel):
    """Store a numerator and denominator before choosing a cross-episode reduction.

    Attributes
    ----------
    component_type : Literal['ratio']
        Fixed discriminator "ratio"; defaults to "ratio".
    numerator : _FiniteFloat
        Finite numerator; not a precomputed ratio.
    denominator : _NonNegativeFloat
        Finite nonnegative exposure or opportunity amount.
    zero_opportunity_occurrence : _BinaryInt
        Exact integer 1 when denominator is zero, else 0.
    eligible_episode_count : _EpisodeEligibility
        Exact integer 0 or 1.

    Notes
    -----
    A zero denominator requires a zero numerator. Retaining both terms lets a
    later analysis choose episode means or pooled ratios explicitly.
    """

    component_type: Literal["ratio"] = "ratio"
    numerator: _FiniteFloat
    denominator: _NonNegativeFloat
    zero_opportunity_occurrence: _BinaryInt
    eligible_episode_count: _EpisodeEligibility

    @model_validator(mode="after")
    def _validate_denominator(self) -> RatioComponentV1:
        """Check numerator and zero-opportunity flag against denominator presence.

        Zero denominator needs numerator zero and flag one; positive denominator needs
        flag zero. Return this component or raise ValueError.
        """
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
    """Store qualifying and eligible durations in discrete transition ticks.

    Attributes
    ----------
    component_type : Literal['duration']
        Fixed discriminator "duration"; defaults to "duration".
    qualifying_steps : _NonNegativeInt
        Nonnegative count of qualifying ticks.
    eligible_steps : _NonNegativeInt
        Nonnegative eligible tick count, at least qualifying_steps.
    eligible_episode_count : _EpisodeEligibility
        Exact integer 0 or 1.
    """

    component_type: Literal["duration"] = "duration"
    qualifying_steps: _NonNegativeInt
    eligible_steps: _NonNegativeInt
    eligible_episode_count: _EpisodeEligibility

    @model_validator(mode="after")
    def _validate_duration(self) -> DurationComponentV1:
        """Reject qualifying ticks greater than eligible ticks with ValueError."""
        if self.qualifying_steps > self.eligible_steps:
            raise ValueError("qualifying duration cannot exceed eligible duration")
        return self


class OpportunityComponentV1(EvaluationModel):
    """Store a count of actual opportunities and episode eligibility.

    Attributes
    ----------
    component_type : Literal['opportunity']
        Fixed discriminator "opportunity"; defaults to "opportunity".
    opportunity_count : _NonNegativeInt
        Nonnegative number of opportunities defined by the metric.
    eligible_episode_count : _EpisodeEligibility
        Exact integer 0 or 1.
    """

    component_type: Literal["opportunity"] = "opportunity"
    opportunity_count: _NonNegativeInt
    eligible_episode_count: _EpisodeEligibility


class DistributionComponentV1(EvaluationModel):
    """Keep ordered finite samples instead of reducing them to summary moments.

    Attributes
    ----------
    component_type : Literal['distribution']
        Fixed discriminator "distribution"; defaults to "distribution".
    observations : tuple[DistributionObservationV1, ...]
        Immutable sample tuple with ordinals 0 onward and unique source IDs.
    eligible_episode_count : _EpisodeEligibility
        Exact integer 0 or 1.

    Notes
    -----
    Empty samples in an eligible episode provide zero-opportunity evidence.
    Construction checks exact nested sample types.
    """

    component_type: Literal["distribution"] = "distribution"
    observations: tuple[DistributionObservationV1, ...]
    eligible_episode_count: _EpisodeEligibility

    @model_validator(mode="after")
    def _validate_observations(self) -> DistributionComponentV1:
        """Check strict sample models, gap-free ordinals, and unique source IDs.

        Return this distribution or raise ValueError; preserve the declared sample
        order.
        """
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
    """Return ordered name/value coordinates for stable comparison without sorting."""
    return tuple((row.name, row.value) for row in dimensions)


def _subject_key(subject: StatisticSubjectV1) -> str:
    """Serialize a strict subject to stable JSON text used in row ordering."""
    return subject.model_dump_json()


def _component_has_zero_opportunity(
    component: SufficientStatisticComponentV1,
) -> bool:
    """Recognize an eligible component with no observations or denominator.

    Counts deliberately return false because a measured zero count is not an
    opportunity claim. Other component kinds use their own exposure field.
    """
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
    """Describe one reducer-owned raw statistic before adding episode provenance.

    Attributes
    ----------
    metric_id : _AsciiIdentifier
        Stable identifier ending in ".vN" for metric_version.
    metric_version : _PositiveInt
        Positive version of the metric definition.
    component_name : _AsciiIdentifier
        Identifier distinguishing this component of the metric.
    reducer_id : _AsciiIdentifier
        Owning reducer's stable identifier.
    reducer_version : _PositiveInt
        Positive reducer version.
    units : _AsciiIdentifier
        Identifier naming the component's measurement units.
    amount_stage : HealthAmountStage | None
        Optional health-accounting stage; defaults to None.
    subject : StatisticSubjectV1
        Episode, team, agent, team/class, or ordered agent-pair subject.
    dimensions : tuple[StatisticDimensionV1, ...]
        Sorted unique metric-specific categories; defaults to an empty tuple.
    completion_scope : StatisticCompletionScope
        any_gap_free_prefix or complete_episode.
    supports_right_censoring : bool
        Whether the metric defines right-censored endpoints.
    result_status : StatisticResultStatus
        defined, zero_opportunity, insufficient_data,
        ambiguous_attribution, structurally_inapplicable, or invalid_artifact.
    status_reason : _AsciiText | None
        None for defined results; required printable reason otherwise.
    endpoint_observation_status : EndpointObservationStatus
        not_applicable, observed, right_censored,
        competing_event, or unavailable.
    component : SufficientStatisticComponentV1 | None
        Raw typed component, required for defined and zero_opportunity
        results; other statuses may omit it.

    Notes
    -----
    Context truth is added by the observer. Construction checks strict nested
    types, dimensions, version suffix, status evidence, and censoring support.
    """

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
        """Check draft identity, dimensions, typed payload, status, and censoring
        support.

        Raise ValueError on a contradiction. Episode-level eligibility is applied later
        because a draft does not yet own completion or processing progress.
        """
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
    """Return metric/version/component/subject/dimensions as the unique row key.

    Reducer identity is not part of this key, so two reducers cannot publish the
    same metric row under different producer names.
    """
    return (
        draft.metric_id,
        draft.metric_version,
        draft.component_name,
        _subject_key(draft.subject),
        _dimensions_key(draft.dimensions),
    )


class EvaluationMetricReducerStateV1(EvaluationModel):
    """Provide the identity fields for an immutable reducer-specific state model.

    Attributes
    ----------
    reducer_id : _AsciiIdentifier
        Stable owner identifier matching the registered reducer.
    reducer_version : _PositiveInt
        Positive owner version matching the registered reducer.

    Notes
    -----
    Subclasses add only scalar, tuple, or strict frozen model fields. Reducers
    return replacement instances and keep one exact state class after initialization.
    """

    reducer_id: _AsciiIdentifier
    reducer_version: _PositiveInt


class EvaluationEpisodeCompletionV1(EvaluationModel):
    """Record captured rollout completion independently of metric-processing success.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.episode_completion']
        Fixed episode-completion schema identifier.
    schema_version : _Cp3SchemaVersion
        Exact integer 1; defaults to 1.
    episode_id : _AsciiIdentifier
        Episode owning this captured prefix.
    completion_state : CompletionState
        complete, partial, interrupted, or failed.
    expected_transition_count : _PositiveInt
        Positive declared artifact horizon.
    validated_transition_count : _NonNegativeInt
        Valid prefix length from zero through the horizon.
    last_valid_frame_index : _NonNegativeInt
        Equal to validated_transition_count.
    last_valid_frame_id : _AsciiIdentifier
        Canonical episode/frame ID at that index.
    terminated : bool
        Recorded task-termination flag at the valid tail.
    truncated : bool
        Recorded truncation flag at the valid tail.
    completion_bases : tuple[CompletionBasis, ...]
        Ordered task_terminal when terminated, then declared_horizon
        when the validated count reaches its declared horizon; empty otherwise.
    end_or_failure_reason : _AsciiText | None
        Required for incomplete states; defaults to None.
    failure_origin : RolloutFailureOrigin | None
        simulation, policy, validation, or capture only for failed
        rollouts; defaults to None and is required when failed.

    Notes
    -----
    Truncated alone does not establish a declared horizon. A complete record
    needs termination or the full declared count. Zero-transition prefixes
    cannot carry either done flag.
    """

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
        """Check counts, canonical tail identity, exact completion evidence, and failure
        fields.

        Require complete labeling once task termination or the declared horizon is
        observed. Reject incomplete records without reasons and contradictory failure
        origins with ValueError.
        """
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
    """Record the first stable failure in host validation or metric processing.

    Attributes
    ----------
    stage : ProcessingFailureStage
        Named processing stage, from initial validation through lifecycle checks.
    code : _AsciiIdentifier
        Stable machine-readable failure identifier.
    reducer_id : _AsciiIdentifier | None
        Reducer identifier for reducer stages, otherwise None.
    reducer_version : _PositiveInt | None
        Positive reducer version paired with reducer_id, otherwise None.
    attempted_transition_index : _NonNegativeInt | None
        Optional zero-based failed transition coordinate;
        required for reducer advance and forbidden outside transition-related stages.
    detail : _AsciiText
        Nonempty printable ASCII explanation.

    Notes
    -----
    This is processing failure metadata, separate from a policy or simulator
    rollout failure. Optional fields default to None.
    """

    stage: ProcessingFailureStage
    code: _AsciiIdentifier
    reducer_id: _AsciiIdentifier | None = None
    reducer_version: _PositiveInt | None = None
    attempted_transition_index: _NonNegativeInt | None = None
    detail: _AsciiText

    @model_validator(mode="after")
    def _validate_reducer_identity(self) -> EvaluationProcessingFailureV1:
        """Require reducer identity and attempted-index fields only for their owning
        stages.

        Return this failure or raise ValueError when required pairs/coordinates are
        absent
        or a nonowning stage carries them.
        """
        if (self.reducer_id is None) != (self.reducer_version is None):
            raise ValueError(
                "processing failure reducer ID and version must appear together"
            )
        reducer_stage = self.stage in (
            "reducer_initialize",
            "reducer_advance",
            "offline_reducer_initialize",
            "offline_reducer_advance",
            "reducer_finalize",
        )
        if reducer_stage and self.reducer_id is None:
            raise ValueError("reducer failure stages require reducer identity")
        if not reducer_stage and self.reducer_id is not None:
            raise ValueError("non-reducer failure stages forbid reducer identity")
        transition_stage = self.stage in (
            "transition_validation",
            "reducer_advance",
            "offline_reducer_advance",
        )
        if (
            self.stage in ("reducer_advance", "offline_reducer_advance")
            and self.attempted_transition_index is None
        ):
            raise ValueError("reducer advance failure requires an attempted index")
        if not transition_stage and self.attempted_transition_index is not None:
            raise ValueError(
                "only transition validation or reducer advance may carry an "
                "attempted transition index"
            )
        return self


class EvaluationProcessingStatusV1(EvaluationModel):
    """Record how much validated trajectory every reducer processed successfully.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.processing_status']
        Fixed processing-status schema identifier.
    schema_version : _Cp3SchemaVersion
        Exact integer 1; defaults to 1.
    status : ProcessingState
        succeeded or failed.
    processed_transition_count : _NonNegativeInt
        Nonnegative fully committed prefix length.
    failure : EvaluationProcessingFailureV1 | None
        Required strict failure model when failed, otherwise None; defaults to None.

    Notes
    -----
    This model checks failure presence. The separate progress validator compares
    this count with the captured trajectory's validated count.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.processing_status"] = (
        PROCESSING_STATUS_SCHEMA_ID
    )
    schema_version: _Cp3SchemaVersion = CP3_SCHEMA_VERSION
    status: ProcessingState
    processed_transition_count: _NonNegativeInt
    failure: EvaluationProcessingFailureV1 | None = None

    @model_validator(mode="after")
    def _validate_status(self) -> EvaluationProcessingStatusV1:
        """Require failure metadata exactly when status is failed.

        Revalidate the nested failure type and return this status or raise ValueError.
        """
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
        "offline_reducer_initialize",
        "completion_validation",
        "reducer_finalize",
        "statistic_materialization",
        "report_validation",
    }
)


def validate_evaluation_processing_progress_v1(
    validated_transition_count: int,
    processing_status: EvaluationProcessingStatusV1,
) -> None:
    """Check processing progress against the valid captured trajectory.

    Parameters
    ----------
    validated_transition_count : int
        Exact nonnegative Python int counting valid transitions.
    processing_status : EvaluationProcessingStatusV1
        Strict V1 status with committed reducer count and any failure.

    Returns
    -------
    None
        None.

    Raises
    ------
    ValueError
        Types, counts, failure stage, or attempted-transition coordinate disagree.

    Notes
    -----
    Success requires equal counts. Online advance failure leaves exactly one
    validated but unprocessed transition. Offline analysis may leave a longer
    unprocessed suffix, with the failed index naming its first transition.
    """
    if type(validated_transition_count) is not int or validated_transition_count < 0:
        raise ValueError("validated transition count must be a nonnegative integer")
    _require_stable_nested_model(
        processing_status,
        record_name="processing progress status",
        expected_types=(EvaluationProcessingStatusV1,),
    )
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
    if failure.stage == "offline_reducer_initialize":
        if processed_transition_count != 0:
            raise ValueError("offline initialization failure forbids processed steps")
        return
    if failure.stage == "offline_reducer_advance":
        if (
            processed_transition_count >= validated_transition_count
            or failure.attempted_transition_index != processed_transition_count
        ):
            raise ValueError(
                "offline advance failure must identify the first unprocessed step"
            )
        return
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
        if failure.attempted_transition_index != processed_transition_count:
            raise ValueError(
                "reducer advance attempted index must equal processed progress"
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
    """Make an unchecked immutable draft copy with observer-owned field updates.

    Only trusted eligibility normalization calls this helper; external drafts still
    need strict model admission.
    """
    return draft.model_copy(update=updates)


def _with_episode_eligibility(
    draft: SufficientStatisticDraftV1,
    eligible_episode_count: _EpisodeEligibility,
) -> SufficientStatisticDraftV1:
    """Replace a present component's zero/one episode exposure without changing its
    data.

    Return the original draft when no component or no change is needed.
    """
    component = draft.component
    if component is None or component.eligible_episode_count == eligible_episode_count:
        return draft
    replacement_component = component.model_copy(
        update={"eligible_episode_count": eligible_episode_count}
    )
    return _replace_draft(draft, component=replacement_component)


def _apply_episode_eligibility(
    draft: SufficientStatisticDraftV1,
    completion: EvaluationEpisodeCompletionV1,
    processing_status: EvaluationProcessingStatusV1,
) -> SufficientStatisticDraftV1:
    """Apply completion, processing-prefix, and endpoint rules to a trusted draft.

    Preserve stronger failure statuses. Complete-episode rows need a complete fully
    processed trajectory; otherwise make them ineligible and, when appropriate,
    insufficient_data. Right censoring requires a complete declared horizon.
    Raise ValueError for contradictory endpoint or defined-result exposure claims.
    """
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
    """Add observer-owned episode and processing facts to a statistic draft.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.raw_sufficient_statistic']
        Fixed raw-statistic schema identifier.
    schema_version : _Cp3SchemaVersion
        Exact integer 1; defaults to 1.
    episode_id : _AsciiIdentifier
        Owning episode ID.
    aggregation_keys : tuple[AggregationKeyV1, ...]
        Sorted unique experiment coordinates copied from context.
    source_schema_versions : tuple[SchemaVersionEntryV1, ...]
        Exact ordered V1 source bindings copied from context.
    rollout_completion : EvaluationEpisodeCompletionV1
        Captured prefix's completion record.
    validated_transition_count : _NonNegativeInt
        Valid count, equal to rollout_completion.
    processed_transition_count : _NonNegativeInt
        Committed reducer count, equal to processing_status.
    processing_status : EvaluationProcessingStatusV1
        Metric-processing result for the prefix.

    Notes
    -----
    All semantic fields from SufficientStatisticDraftV1 also apply. Construction
    checks joins, forbidden dimension shadowing, publishing eligibility, and
    failure stages. It does not average or normalize raw components.
    """

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
        """Check copied context coordinates, counts, failure publishing rules, and
        eligibility.

        Require final status to equal episode-aware normalization of the inherited
        draft.
        Return this row or raise ValueError; do not silently repair inconsistent
        provenance.
        """
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
        validate_evaluation_processing_progress_v1(
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
    """Use the draft's metric/component/subject/dimensions key for final row
    ordering.
    """
    return _draft_row_key(row)


def _validate_subject_join(
    context: EvaluationEpisodeContextV1,
    row: RawSufficientStatisticV1,
) -> None:
    """Check statistic subjects against active roster/policy assignments.

    Agent and pair subjects require active V1 assigned slots. An absent team/class
    requires invalid or structurally inapplicable status. Episode/team subjects need
    no additional roster join. Raise ValueError for a contradiction.
    """
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
    """Keep one historical episode's raw statistics and processing result.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.metric_report']
        Fixed metric-report schema identifier.
    schema_version : _Cp3SchemaVersion
        Exact integer 1; defaults to 1.
    report_id : _AsciiIdentifier
        Canonical episode ID followed by ":metric-report".
    context : EvaluationEpisodeContextV1
        Exact historical V1 context.
    completion : EvaluationEpisodeCompletionV1
        Physical rollout completion or valid partial-prefix status.
    processing_status : EvaluationProcessingStatusV1
        Separate metric-processing progress and failure status.
    statistics : tuple[RawSufficientStatisticV1, ...]
        Immutable, uniquely keyed, canonically sorted raw rows joined to
        the same episode, context coordinates, completion, and progress.

    Notes
    -----
    A report can preserve processing failure. Initial or completion validation
    failure cannot produce a structurally valid report. Replay persistence lives
    in the replay and replay_io modules.
    """

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
        """Check strict record types, canonical report ID, progress, and every statistic
        join.

        Require sorted unique row keys and eligible active subjects. Reject inconsistent
        or unreportable failure metadata with ValueError.
        """
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
        validate_evaluation_processing_progress_v1(
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
    """Join one context, start frame, transition, and successor for a host consumer.

    Attributes
    ----------
    context : EvaluationEpisodeContext
        Supported episode context.
    start_frame : EvaluationFrame
        Decision frame before the transition.
    transition : EvaluationTransitionV1
        V1 transition linking the two frames.
    successor_frame : EvaluationFrame
        Matching-version frame one simulator tick after start_frame.

    Notes
    -----
    Context V1 construction performs full semantic validation and stores detached
    canonical copies. Context V2/V3 construction checks version/episode/frame
    links and adjacent simulator ticks only, retaining supplied records; their
    replay admission path owns full integrity checks. The class name alone
    does not promise equally deep validation for every context version.
    """

    context: EvaluationEpisodeContext
    start_frame: EvaluationFrame
    transition: EvaluationTransitionV1
    successor_frame: EvaluationFrame

    def __post_init__(self) -> None:
        """Validate the context-specific join and canonicalize historical V1 records.

        Newer contexts use the narrow captured-record join; V1 uses full semantic
        validation and detached copies. Raise ValueError on an invalid join.
        """
        if type(self.context) in (
            EvaluationEpisodeContextV2,
            EvaluationEpisodeContextV3,
        ):
            # V2 carries captured facts; this view joins records without replaying
            # the legacy simulator/metric semantic-validation pipeline.
            expected_frame = (
                EvaluationFrameV2
                if type(self.context) is EvaluationEpisodeContextV3
                else EvaluationFrameV1
            )
            if (
                type(self.start_frame) is not expected_frame
                or type(self.successor_frame) is not expected_frame
            ):
                raise ValueError(
                    "transition view frame versions must match their context"
                )
            episode_id = self.context.identity.episode_id
            if (
                self.start_frame.episode_id != episode_id
                or self.successor_frame.episode_id != episode_id
                or self.transition.episode_id != episode_id
                or self.transition.start_frame_id != self.start_frame.frame_id
                or self.transition.successor_frame_id != self.successor_frame.frame_id
                or self.successor_frame.simulator_step_count
                != self.start_frame.simulator_step_count + 1
            ):
                raise ValueError(
                    "captured transition view must join adjacent episode frames"
                )
            return
        (
            canonical_context,
            canonical_start,
            canonical_transition,
            canonical_successor,
        ) = _canonicalize_evaluation_transition_unit_v1(
            self.context,
            self.start_frame,
            self.transition,
            self.successor_frame,
        )
        object.__setattr__(self, "context", canonical_context)
        object.__setattr__(self, "start_frame", canonical_start)
        object.__setattr__(self, "transition", canonical_transition)
        object.__setattr__(self, "successor_frame", canonical_successor)


def _view_from_owned_records(
    context: EvaluationEpisodeContext,
    start_frame: EvaluationFrame,
    transition: EvaluationTransitionV1,
    successor_frame: EvaluationFrame,
) -> EvaluationTransitionViewV1:
    """Build a view from observer-owned validated records without revalidating.

    Caller must already own a coherent unit. This deliberately bypasses dataclass
    post-init to avoid repeating the same transition check.
    """
    view = object.__new__(EvaluationTransitionViewV1)
    object.__setattr__(view, "context", context)
    object.__setattr__(view, "start_frame", start_frame)
    object.__setattr__(view, "transition", transition)
    object.__setattr__(view, "successor_frame", successor_frame)
    return view


class EvaluationMetricReducerV1(Protocol):
    """Define a trusted deterministic reducer with immutable replacement state.

    Attributes
    ----------
    reducer_id : str
        Stable ASCII identifier, unique within an observer.
    reducer_version : int
        Positive exact Python int identifying reducer behavior.

    Notes
    -----
    Implement initialize, advance, and finalize without mutable internals, RNG,
    clocks, discovery, files, network, logging, or callbacks. The observer can
    commit state replacements atomically, but cannot undo external side effects
    from a reducer that breaks this protocol.
    """

    reducer_id: str
    reducer_version: int

    def initialize(
        self,
        context: EvaluationEpisodeContextV1,
        initial_frame: EvaluationFrameV1,
    ) -> EvaluationMetricReducerStateV1:
        """Create this reducer's immutable state from the accepted initial frame.

        Parameters
        ----------
        context : EvaluationEpisodeContextV1
            Validated V1 episode metadata.
        initial_frame : EvaluationFrameV1
            Validated artifact frame zero, possibly at a nonzero simulator tick.

        Returns
        -------
        EvaluationMetricReducerStateV1
            Frozen scalar/tuple-backed state bearing this reducer's exact ID/version.

        Notes
        -----
        The observer calls this once at start and checks the declared state schema.
        Do not mutate inputs or store hidden evolving state on the reducer.
        """
        ...

    def advance(
        self,
        previous_state: EvaluationMetricReducerStateV1,
        view: EvaluationTransitionViewV1,
    ) -> EvaluationMetricReducerStateV1:
        """Return replacement state after consuming one validated transition view.

        Parameters
        ----------
        previous_state : EvaluationMetricReducerStateV1
            Last committed state of this reducer's exact initialized class.
        view : EvaluationTransitionViewV1
            Owned context/start/transition/successor records for the next transition.

        Returns
        -------
        EvaluationMetricReducerStateV1
            New state with the same exact type and reducer identity.

        Notes
        -----
        Do not mutate previous_state or external objects. If any reducer fails, the
        observer discards all candidate reducer replacements for this transition.
        """
        ...

    def finalize(
        self,
        state: EvaluationMetricReducerStateV1,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        """Project immutable reducer state into raw statistic drafts.

        Parameters
        ----------
        state : EvaluationMetricReducerStateV1
            Last committed reducer state.
        completion : EvaluationEpisodeCompletionV1
            Captured rollout completion or partial-prefix facts.
        processing_status : EvaluationProcessingStatusV1
            Shared reducer progress and any processing failure.

        Returns
        -------
        tuple[SufficientStatisticDraftV1, ...]
            Immutable tuple of exact SufficientStatisticDraftV1 rows carrying this
            reducer's identity/version. An empty tuple is allowed.

        Notes
        -----
        Must be deterministic and side-effect free because previews may call it
        before finalization. The observer supplies episode provenance and final
        eligibility; reducers own valid metric meaning and raw component values.
        """
        ...


def _ascii_failure_detail(error: BaseException) -> str:
    """Turn an exception into one printable ASCII diagnostic without raising from str.

    Collapse whitespace, escape non-ASCII/control characters, and fall back to the
    exception class name when its message is missing or cannot be rendered.
    """
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
    """Return a reducer's validated ASCII ID and positive exact integer version.

    Raise ValueError for malformed registration. The observer uses the pair to
    sort reducers and separately rejects duplicate IDs.
    """
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
    """Check that a state model forbids extras/nonfinite values and is strict/frozen."""
    model_config = model_type.model_config
    return (
        model_config.get("allow_inf_nan") is False
        and model_config.get("extra") == "forbid"
        and model_config.get("frozen") is True
        and model_config.get("strict") is True
    )


def _immutable_state_schema(
    annotation: object, seen: set[type[EvaluationModel]]
) -> bool:
    """Check declared state types once without traversing accumulated values.

    Allow scalar leaves, tuples, literals, unions, aliases, and strict frozen
    EvaluationModel trees without private attributes. Track seen model classes to
    handle recursive declarations and reject mutable or arbitrary-object fields.
    """
    if any(annotation is scalar for scalar in (type(None), bool, int, float, str)):
        return True
    if isinstance(annotation, TypeAliasType):
        return _immutable_state_schema(annotation.__value__, seen)
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Annotated:
        return _immutable_state_schema(arguments[0], seen)
    if origin in (Union, UnionType):
        return all(_immutable_state_schema(item, seen) for item in arguments)
    if origin is Literal:
        return all(
            type(item) in (type(None), bool, int, float, str) for item in arguments
        )
    if origin is tuple:
        return all(
            item is Ellipsis or _immutable_state_schema(item, seen)
            for item in arguments
        )
    if not isinstance(annotation, type) or not issubclass(annotation, EvaluationModel):
        return False
    if not _uses_strict_frozen_model_config(annotation) or getattr(
        annotation, "__private_attributes__", {}
    ):
        return False
    if annotation in seen:
        return True
    seen.add(annotation)
    return all(
        _immutable_state_schema(field.annotation, seen)
        for field in annotation.model_fields.values()
    )


def _validate_reducer_state(
    state: object,
    *,
    reducer_id: str,
    reducer_version: int,
    expected_type: type[EvaluationMetricReducerStateV1] | None,
) -> EvaluationMetricReducerStateV1:
    """Check a trusted replacement state's schema, exact type, and reducer identity.

    At initialization inspect immutable field declarations; later replacements must
    use the established exact class. Reject wrong types/private or mutable top-level
    fields with TypeError and identity mismatches with ValueError. This does not
    deeply revalidate all accumulated component values on every step.
    """
    if not isinstance(state, EvaluationMetricReducerStateV1):
        raise TypeError("reducers must return EvaluationMetricReducerStateV1")
    if not _uses_strict_frozen_model_config(type(state)):
        raise TypeError(
            "reducer state subclasses must retain the strict frozen model config"
        )
    if expected_type is None and not _immutable_state_schema(type(state), set()):
        raise TypeError("reducer state fields must be scalar or tuple-backed models")
    if (
        type(state.reducer_id) is not str
        or type(state.reducer_version) is not int
        or state.reducer_id != reducer_id
        or state.reducer_version != reducer_version
    ):
        raise ValueError("reducer state identity/version must match its reducer")
    if expected_type is not None and type(state) is not expected_type:
        raise TypeError("reducer replacement state type must remain exact")
    if getattr(state, "__pydantic_private__", None) or any(
        type(getattr(state, name)) not in (type(None), bool, int, float, str, tuple)
        and not isinstance(getattr(state, name), EvaluationModel)
        for name in type(state).model_fields
    ):
        raise TypeError("reducer state fields must be scalar or tuple-backed models")
    return state


def _materialize_raw_statistic(
    context: EvaluationEpisodeContextV1,
    draft: SufficientStatisticDraftV1,
    completion: EvaluationEpisodeCompletionV1,
    processing_status: EvaluationProcessingStatusV1,
) -> RawSufficientStatisticV1:
    """Add context/progress provenance to an eligibility-adjusted trusted draft.

    Reject dimensions that shadow aggregation keys. Reuse validated immutable values
    instead of serializing accumulated history; external admission still performs
    full raw-row validation.
    """
    eligible = _apply_episode_eligibility(draft, completion, processing_status)
    if {dimension.name for dimension in eligible.dimensions}.intersection(
        key.name for key in context.aggregation_keys
    ):
        raise ValueError("statistic dimensions cannot shadow context aggregation keys")
    # Reducers own valid immutable drafts; the observer owns all provenance.
    # Reuse those values instead of serializing and rebuilding their history.
    return RawSufficientStatisticV1.model_construct(
        **{
            name: getattr(eligible, name)
            for name in SufficientStatisticDraftV1.model_fields
        },
        episode_id=context.identity.episode_id,
        aggregation_keys=context.aggregation_keys,
        source_schema_versions=context.schema_versions,
        rollout_completion=completion,
        validated_transition_count=completion.validated_transition_count,
        processed_transition_count=processing_status.processed_transition_count,
        processing_status=processing_status,
    )


def _reducer_drafts(
    reducer: EvaluationMetricReducerV1,
    state: EvaluationMetricReducerStateV1,
    completion: EvaluationEpisodeCompletionV1,
    processing: EvaluationProcessingStatusV1,
) -> tuple[SufficientStatisticDraftV1, ...]:
    """Call a trusted reducer's finalize and check tuple, row type, and owner identity.

    Reject mutable containers or wrong types with TypeError and mismatched reducer
    identity with ValueError. Do not traverse accumulated component values again.
    """
    drafts = reducer.finalize(state, completion, processing)
    if type(drafts) is not tuple:
        raise TypeError("reducer finalize must return an immutable tuple")
    for draft in drafts:
        if type(draft) is not SufficientStatisticDraftV1:
            raise TypeError("reducer finalize must return exact statistic drafts")
        if (
            draft.reducer_id != reducer.reducer_id
            or draft.reducer_version != reducer.reducer_version
        ):
            raise ValueError("statistic draft reducer identity/version mismatch")
    return drafts


class EvaluationEpisodeObserverV1:
    """Validate a historical episode and coordinate atomic host metric updates.

    Parameters
    ----------
    context : object
        Exact V1 episode context; construction retains a detached validated copy.
    reducers : object
        Immutable tuple of trusted reducers, sorted by ID/version internally.
        IDs must be unique. Defaults to an empty tuple for capture without metrics.

    Attributes
    ----------
    context : object
        Detached copy of the immutable owned context.
    lifecycle_state : object
        awaiting_initial, open, sealed, poisoned, or finalized.
    validated_transition_count : object
        Number of accepted physical/artifact transition units.
    processed_transition_count : object
        Number atomically consumed by every reducer.
    retained_frames : object
        Detached T+1 frame tuple for metric-complete profiles, otherwise None.
    retained_transitions : object
        Detached T transition tuple for those profiles, otherwise None.
    finalized_report : object
        Detached committed report after finalization, otherwise None.
    reducer_states : object
        Detached tuple of last committed immutable states, otherwise None.

    Notes
    -----
    Call start once, append consecutive transitions, then finalize once. Completion
    seals appends. Invalid input or reducer failure poisons the observer while
    retaining its last valid trajectory and committed reducer prefix.
    Only evaluation_metric_complete and scenario_metric_complete retain history.
    This is a host analysis boundary, not the numerical rollout API or actor input.
    """

    __slots__ = (
        "_context",
        "_current_frame",
        "_finalize_attempted",
        "_finalized_report",
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
        """Create an observer without accepting an initial frame or running reducers.

        Parameters
        ----------
        context : EvaluationEpisodeContextV1
            Exact, structurally valid historical V1 context.
        reducers : tuple[EvaluationMetricReducerV1, ...]
            Immutable tuple of trusted reducers with unique valid IDs and
            positive versions. Defaults to no reducers.

        Returns
        -------
        None
            None.

        Raises
        ------
        TypeError
            reducers is not an immutable tuple.
        ValueError
            Context admission or reducer registration fails.

        Notes
        -----
        Copies the context, orders reducers, sets both progress counts to zero, and
        chooses history retention from capture_profile. start initializes reducer
        states.
        """
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
        self._finalized_report: EvaluationMetricReportV1 | None = None
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
        """Return a detached deep copy of the immutable context.

        Repeated access copies the record tree; internal processing uses the owned copy.
        """
        return deepcopy(self._context)

    @property
    def lifecycle_state(self) -> ObserverLifecycleState:
        """Return the current lifecycle label without changing or copying observer
        state.

        The labels are awaiting_initial, open, sealed, poisoned, and finalized.
        """
        return self._lifecycle_state

    @property
    def validated_transition_count(self) -> int:
        """Return the count of accepted transition units, even after a reducer failure.

        This counts physical/artifact validation, not completed metric processing.
        """
        return self._validated_transition_count

    @property
    def processed_transition_count(self) -> int:
        """Return the prefix length committed successfully by every reducer.

        A failing transition can increase validated progress without increasing this
        count.
        """
        return self._processed_transition_count

    @property
    def retained_frames(self) -> tuple[EvaluationFrameV1, ...] | None:
        """Return a detached tuple of retained frames, or None when history is disabled.

        Metric-complete profiles retain frame zero and each validated successor. Before
        start, their tuple is empty. Reading this property copies the retained history.
        """
        if self._retained_frames is None:
            return None
        return deepcopy(tuple(self._retained_frames))

    @property
    def retained_transitions(self) -> tuple[EvaluationTransitionV1, ...] | None:
        """Return a detached tuple of validated transitions, or None without retention.

        The tuple includes a validated transition whose reducer update later failed.
        Reading it copies the retained history.
        """
        if self._retained_transitions is None:
            return None
        return deepcopy(tuple(self._retained_transitions))

    @property
    def finalized_report(self) -> EvaluationMetricReportV1 | None:
        """Return a detached deep copy of the committed report, or None before
        finalization.

        The caller cannot mutate the observer's owned report through this result.
        """
        return deepcopy(self._finalized_report)

    @property
    def reducer_states(self) -> tuple[EvaluationMetricReducerStateV1, ...] | None:
        """Return detached last-committed reducer states, or None before successful
        start.

        Candidate states from a failing update are never exposed as committed states.
        """
        return deepcopy(self._reducer_states)

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
        """Keep the first processing failure and optionally append a later diagnostic.

        Optional reducer and attempted index default to None. replace=False preserves
        an existing failure; replace=True appends secondary detail while retaining the
        first failure's stage, code, identity, and progress boundary.
        """
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
        """Reject an operation and poison unfinished observers with a lifecycle failure.

        A finalized observer remains finalized. Raise RuntimeError naming the operation
        and previous state; retain the first processing failure when one already exists.
        """
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
        """Accept frame zero and initialize all reducer states as one transaction.

        Parameters
        ----------
        initial_frame : EvaluationFrameV1
            Exact V1 frame zero matching context and any remaining TDM horizon.

        Returns
        -------
        None
            None.

        Raises
        ------
        RuntimeError
            Called outside awaiting_initial, or a reducer fails initialization.
        ValueError
            Initial frame/context semantic validation fails.

        Notes
        -----
        Retains a detached valid initial frame before reducer initialization.
        Reducer states commit only when every initializer succeeds. Any failure
        poisons the observer; invalid initial data cannot later produce a report.
        """
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
                candidate = _validate_reducer_state(
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
        """Accept one adjacent transition, then atomically advance all reducers.

        Parameters
        ----------
        transition : EvaluationTransitionV1
            Exact V1 transition from the observer's current frame.
        successor_frame : EvaluationFrameV1
            Exact V1 successor frame one artifact index and simulator tick later.

        Returns
        -------
        None
            None.

        Raises
        ------
        RuntimeError
            Observer is not open or a reducer update fails.
        ValueError
            Strict transition-unit validation fails.

        Notes
        -----
        A valid transition and its successor commit to trajectory history before
        reducers run. All reducer replacements commit together; on failure their
        previous states remain and processed count does not advance. Invalid inputs
        or reducer errors poison further capture. Done or declared horizon seals it.
        """
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

        self._append_validated_view(view)

    def _append_validated_view(self, view: EvaluationTransitionViewV1) -> None:
        """Commit an owned valid trajectory unit before the atomic reducer transaction.

        Caller supplies a previously validated V1 view. On reducer failure, preserve the
        new validated frame/history and old reducer states, then poison the observer.
        Successful terminal or horizon-reaching updates seal further appends.
        """
        if self._lifecycle_state != "open":
            self._reject_lifecycle("append")
        attempted_index = view.transition.transition_index

        # Validation is authoritative physical/artifact evidence and commits
        # independently from reducer progress.
        # This observer accepts only V1 contexts and validated V1 frames.
        successor = cast(EvaluationFrameV1, view.successor_frame)
        self._current_frame = successor
        self._last_transition = view.transition
        self._validated_transition_count += 1
        if self._retained_transitions is not None:
            self._retained_transitions.append(view.transition)
        if self._retained_frames is not None:
            self._retained_frames.append(successor)

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
                candidate = _validate_reducer_state(
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
        """Build completion from the last valid frame and task-owned tail facts.

        Require a valid initial frame. The caller's requested state/reason/origin must
        fit captured termination and declared-horizon evidence. An authoritative task
        reason replaces an omitted reason and rejects a conflicting one.
        """
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
        """Build status from committed progress, recording an unexplained mismatch as
        failure.

        If no failure exists, equal validated/processed counts succeed; unequal counts
        create a statistic_materialization failure. Preserve any earlier failure.
        """
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

    def evaluate_retained(
        self,
        reducers: tuple[EvaluationMetricReducerV1, ...],
    ) -> None:
        """Run reducers over captured history without rerunning the simulator.

        Parameters
        ----------
        reducers : tuple[EvaluationMetricReducerV1, ...]
            Immutable tuple of trusted reducers to attach to a capture created
            without reducers.

        Returns
        -------
        None
            None.

        Raises
        ------
        RuntimeError
            Reducers are already attached, finalization was attempted,
            or no initial frame/history was retained.
        ValueError
            Reducer registration or an unhandled retained-record invariant fails.

        Notes
        -----
        Uses a temporary observer and copies its reducer state/progress back.
        Reducer failures are recorded as offline failures; the original valid
        trajectory is never shortened to the processed prefix. If capture already
        has a processing failure, this method leaves it unchanged and returns.
        It is host analysis and does not reset or step Core.
        """
        if self._reducers or self._finalize_attempted:
            raise RuntimeError("offline evaluation requires an unfinalized capture")
        frames = self._retained_frames
        transitions = self._retained_transitions
        if not frames or transitions is None:
            raise RuntimeError("offline evaluation requires retained capture")
        if self._processing_failure is not None:
            # Preserve the original failure and its validated/processed boundary.
            # Replay analysis may separately recover metrics from this prefix.
            return
        evaluator = EvaluationEpisodeObserverV1(self._context, reducers=reducers)
        try:
            evaluator.start(frames[0])
            for start, transition, successor in zip(
                frames[:-1], transitions, frames[1:], strict=True
            ):
                evaluator._append_validated_view(
                    _view_from_owned_records(
                        evaluator._context, start, transition, successor
                    )
                )
        except RuntimeError, ValueError, TypeError:
            if evaluator._processing_failure is None:
                raise
        self._reducers = evaluator._reducers
        self._reducer_states = evaluator._reducer_states
        self._reducer_state_types = evaluator._reducer_state_types
        self._processed_transition_count = evaluator._processed_transition_count
        if self._processing_failure is None:
            failure = evaluator._processing_failure
            if failure is not None and failure.stage in (
                "reducer_initialize",
                "reducer_advance",
            ):
                payload = failure.model_dump(mode="python")
                payload["stage"] = f"offline_{failure.stage}"
                failure = EvaluationProcessingFailureV1.model_validate(payload)
            self._processing_failure = failure

    def preview_statistics(
        self,
        *,
        completion_state: CompletionState = "partial",
        end_or_failure_reason: str | None = None,
        failure_origin: RolloutFailureOrigin | None = None,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        """Project current raw statistics without finalizing or changing the observer.

        Parameters
        ----------
        completion_state : CompletionState
            Requested captured-prefix state. Defaults to "partial";
            use "complete" only when completion evidence is already present.
        end_or_failure_reason : str | None
            Optional reason. Partial previews default to
            "cursor_prefix"; other incomplete states require an explicit reason.
        failure_origin : RolloutFailureOrigin | None
            Required only for a failed rollout; defaults to None.

        Returns
        -------
        tuple[SufficientStatisticDraftV1, ...]
            Immutable, sorted tuple of eligibility-adjusted drafts with unique row keys.

        Raises
        ------
        RuntimeError
            No initialized reducer states are available.
        ValueError
            Completion evidence, endpoint rules, draft identity, or row uniqueness
            fails.

        Notes
        -----
        Reducer exceptions also propagate. A failing preview does not poison capture.
        Complete-episode rows keep final-report eligibility rules. These statistics
        are researcher analysis, not authorized policy observations.
        """
        if self._reducer_states is None:
            raise RuntimeError("a valid initial frame is required for a preview")
        completion = self._build_completion(
            completion_state=completion_state,
            end_or_failure_reason=(
                "cursor_prefix"
                if completion_state == "partial" and end_or_failure_reason is None
                else end_or_failure_reason
            ),
            failure_origin=failure_origin,
        )
        processing = EvaluationProcessingStatusV1(
            status="succeeded" if self._processing_failure is None else "failed",
            processed_transition_count=self._processed_transition_count,
            failure=self._processing_failure,
        )
        rows: list[SufficientStatisticDraftV1] = []
        for reducer, state in zip(self._reducers, self._reducer_states, strict=True):
            for draft in _reducer_drafts(reducer, state, completion, processing):
                rows.append(_apply_episode_eligibility(draft, completion, processing))
        keys = tuple(_draft_row_key(row) for row in rows)
        if len(keys) != len(set(keys)):
            raise ValueError("reducers produced duplicate statistic row keys")
        return tuple(sorted(rows, key=_draft_row_key))

    def finalize(
        self,
        *,
        completion_state: CompletionState,
        end_or_failure_reason: str | None = None,
        failure_origin: RolloutFailureOrigin | None = None,
    ) -> EvaluationMetricReportV1:
        """Commit the observer's one report using captured completion evidence.

        Parameters
        ----------
        completion_state : CompletionState
            Required state: complete, partial, interrupted, or failed.
            Must agree with the valid transition tail and declared horizon.
        end_or_failure_reason : str | None
            Optional reason; required for incomplete rollout states.
            A recorded task end reason is used when omitted and must match if supplied.
        failure_origin : RolloutFailureOrigin | None
            Required for failed rollouts and forbidden otherwise; defaults to None.

        Returns
        -------
        EvaluationMetricReportV1
            Detached EvaluationMetricReportV1 with owned context, completion, processing
            status, and sorted raw rows. Metric failure may yield an empty statistic
            tuple
            while preserving valid rollout progress.

        Raises
        ------
        RuntimeError
            No valid start exists, finalization was already attempted, or
            lifecycle disallows finalization.
        ValueError
            Requested completion conflicts with captured evidence.

        Notes
        -----
        Finalization is single-use, including failed completion requests. Reducer
        finalization/materialization/report failures are recorded in processing
        status; they do not rewrite physical completion. The committed observer
        becomes finalized and cannot accept more transitions. Performs no file I/O.
        """
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
                    collected.extend(
                        _reducer_drafts(reducer, state, completion, processing_status)
                    )
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
            validate_evaluation_processing_progress_v1(
                completion.validated_transition_count, processing_status
            )
            for row in rows:
                _validate_subject_join(self._context, row)
        except Exception as error:
            self._set_failure(
                stage="report_validation",
                code="metric_report_validation_failed",
                detail=_ascii_failure_detail(error),
                replace=True,
            )
            processing_status = self._processing_status()
            rows = ()
        report = EvaluationMetricReportV1.model_construct(
            report_id=f"{self._context.identity.episode_id}:metric-report",
            context=self._context,
            completion=completion,
            processing_status=processing_status,
            statistics=rows,
        )
        # The caller receives its own tree; finalized observer state stays owned.
        self._lifecycle_state = "finalized"
        self._finalized_report = report
        return deepcopy(report)


def build_evaluation_observer_v1(
    context: EvaluationEpisodeContextV1,
    reducers: tuple[EvaluationMetricReducerV1, ...] = (),
) -> EvaluationEpisodeObserverV1:
    """Create an explicitly enabled historical host observer.

    Parameters
    ----------
    context : EvaluationEpisodeContextV1
        Exact valid V1 episode context.
    reducers : tuple[EvaluationMetricReducerV1, ...]
        Immutable tuple of trusted reducers; defaults to no reducers.

    Returns
    -------
    EvaluationEpisodeObserverV1
        New observer awaiting its initial frame. History retention follows
        context.capture_profile.

    Raises
    ------
    TypeError
        reducers is not a tuple.
    ValueError
        Context or reducer registration is invalid.

    Notes
    -----
    Disabled integrations should retain None instead of constructing an observer.
    This helper does not start capture, run reducers, or write artifacts.
    """
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
    "SufficientStatisticComponentV1",
    "SufficientStatisticDraftV1",
    "SumComponentV1",
    "TeamClassStatisticSubjectV1",
    "TeamStatisticSubjectV1",
    "build_evaluation_observer_v1",
    "validate_evaluation_processing_progress_v1",
]
