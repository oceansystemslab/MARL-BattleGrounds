"""Versioned controlled-scenario specifications and evaluation records.

This module stores declared scenario semantics and caller-supplied results.  It
does not implement measurements, success predicates, simulator rules, or metric
formulas.  Replay and metric-report artifacts remain the source evidence and
are joined through immutable content-addressed references.
"""

from __future__ import annotations

from typing import Annotated, Literal, cast

from pydantic import BeforeValidator, Field, StringConstraints, model_validator

from marl_battlegrounds.evaluation.models import (
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    AssignedPolicySlotV1,
    ContentAddressedIdentityV1,
    EvaluationFrameV1,
    EvaluationModel,
    EvaluationSeedProtocolV1,
    NotApplicablePolicySlotV1,
    RosterSlotV1,
    VersionedIdentityV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.replay import (
    EvaluationMetricReportArtifactV1,
    MetricReportReferenceV1,
    ReplayArtifactReferenceV1,
    ReplayArtifactV1,
    _build_replay_artifact_reference_from_validated_v1,  # pyright: ignore[reportPrivateUsage]
    _validate_metric_report_artifact_against_validated_replay_v1,  # pyright: ignore[reportPrivateUsage]
    validate_replay_artifact_v1,
)
from marl_battlegrounds.evaluation.validation import validate_declared_model_tree

SCENARIO_SPECIFICATION_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.resolved_scenario_specification"
)
SCENARIO_MEASUREMENT_DEFINITION_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.scenario_measurement_definition"
)
SCENARIO_MEASUREMENT_RESULT_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.scenario_measurement_result"
)
SCENARIO_VIOLATION_DEFINITION_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.scenario_violation_definition"
)
SCENARIO_VIOLATION_RESULT_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.scenario_violation_result"
)
SCENARIO_PREDICATE_RESULT_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.scenario_predicate_result"
)
SCENARIO_EVALUATION_RECORD_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.scenario_evaluation_record"
)
SCENARIO_SEED_SCHEDULE_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.scenario_seed_schedule"
)
SCENARIO_SCHEMA_VERSION: Literal[1] = 1
SCENARIO_SCHEMA_VERSION_V2: Literal[2] = 2

type ScenarioClassification = Literal["official", "custom"]
type ScenarioEvaluationRole = Literal[
    "focal",
    "cooperative_partner",
    "adversarial_opponent",
]
type ScenarioFixedSlotRoleV2 = ScenarioEvaluationRole | Literal["not_applicable"]
type ScenarioMeasurementRole = Literal["primary", "secondary"]
type ScenarioValueType = Literal["boolean", "count", "scalar"]
type ScenarioCompletionScope = Literal[
    "any_gap_free_prefix",
    "complete_episode",
]
type ScenarioResultStatus = Literal[
    "defined",
    "insufficient_data",
    "unavailable",
]
type ScenarioEndpointObservationStatus = Literal[
    "not_applicable",
    "observed",
    "right_censored",
    "competing_event",
    "unavailable",
]
type ScenarioPredicateStatus = Literal[
    "satisfied",
    "not_satisfied",
    "unavailable",
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
_Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(gt=0)]
_FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


def _require_schema_version_one(value: object) -> object:
    if type(value) is not int or value != 1:
        raise ValueError("schema_version must be the exact integer 1")
    return value


_ScenarioSchemaVersion = Annotated[
    Literal[1],
    BeforeValidator(_require_schema_version_one),
]


def _require_schema_version_two(value: object) -> object:
    if type(value) is not int or value != 2:
        raise ValueError("schema_version must be the exact integer 2")
    return value


_ScenarioSchemaVersionV2 = Annotated[
    Literal[2],
    BeforeValidator(_require_schema_version_two),
]


def _require_stable_nested_model(
    model: EvaluationModel,
    *,
    record_name: str,
    expected_types: tuple[type[EvaluationModel], ...],
) -> None:
    """Reject unchecked Pydantic copies and undeclared nested subtypes."""
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


class ScenarioParameterV1(EvaluationModel):
    """One canonically ordered, strictly typed scenario parameter."""

    name: _AsciiIdentifier
    value: bool | int | _FiniteFloat | _AsciiText

    @model_validator(mode="after")
    def _validate_value(self) -> ScenarioParameterV1:
        if type(self.value) not in (bool, int, float, str):
            raise ValueError("scenario parameter must use one exact JSON scalar type")
        return self


class ScenarioMeasurementDefinitionV1(EvaluationModel):
    """One declared primary endpoint or secondary scenario margin."""

    schema_id: Literal[
        "marl_battlegrounds.evaluation.scenario_measurement_definition"
    ] = SCENARIO_MEASUREMENT_DEFINITION_SCHEMA_ID
    schema_version: _ScenarioSchemaVersion = SCENARIO_SCHEMA_VERSION
    measurement_id: _AsciiIdentifier
    measurement_version: _PositiveInt
    role: ScenarioMeasurementRole
    value_type: ScenarioValueType
    units: _AsciiIdentifier
    completion_scope: ScenarioCompletionScope
    supports_right_censoring: bool


class ScenarioViolationDefinitionV1(EvaluationModel):
    """One declared safety, role, or behavior violation measurement."""

    schema_id: Literal[
        "marl_battlegrounds.evaluation.scenario_violation_definition"
    ] = SCENARIO_VIOLATION_DEFINITION_SCHEMA_ID
    schema_version: _ScenarioSchemaVersion = SCENARIO_SCHEMA_VERSION
    violation_id: _AsciiIdentifier
    violation_version: _PositiveInt
    value_type: ScenarioValueType
    units: _AsciiIdentifier
    completion_scope: ScenarioCompletionScope
    supports_right_censoring: bool


class ScenarioBooleanValueV1(EvaluationModel):
    """A defined Boolean scenario result value."""

    value_type: Literal["boolean"] = "boolean"
    value: bool


class ScenarioCountValueV1(EvaluationModel):
    """A defined nonnegative count scenario result value."""

    value_type: Literal["count"] = "count"
    value: _NonNegativeInt


class ScenarioScalarValueV1(EvaluationModel):
    """A defined finite scalar scenario result value."""

    value_type: Literal["scalar"] = "scalar"
    value: _FiniteFloat


type ScenarioResultValueV1 = Annotated[
    ScenarioBooleanValueV1 | ScenarioCountValueV1 | ScenarioScalarValueV1,
    Field(discriminator="value_type"),
]


def _validate_result_payload(
    *,
    result_status: ScenarioResultStatus,
    endpoint_observation_status: ScenarioEndpointObservationStatus,
    value: ScenarioResultValueV1 | None,
    reason: str | None,
) -> None:
    if value is not None:
        _require_stable_nested_model(
            value,
            record_name="scenario result value",
            expected_types=(
                ScenarioBooleanValueV1,
                ScenarioCountValueV1,
                ScenarioScalarValueV1,
            ),
        )
    if result_status == "defined":
        if value is None:
            raise ValueError("defined scenario results require a typed value")
        if reason is not None:
            raise ValueError("defined scenario results forbid a reason")
        if endpoint_observation_status == "unavailable":
            raise ValueError(
                "defined scenario results cannot have an unavailable endpoint"
            )
        return
    if value is not None:
        raise ValueError("undefined scenario results must not carry a value")
    if reason is None:
        raise ValueError("undefined scenario results require a reason")
    if result_status == "unavailable" and endpoint_observation_status != "unavailable":
        raise ValueError("unavailable scenario results require an unavailable endpoint")


class ScenarioMeasurementResultV1(EvaluationModel):
    """One caller-materialized scenario measurement result."""

    schema_id: Literal["marl_battlegrounds.evaluation.scenario_measurement_result"] = (
        SCENARIO_MEASUREMENT_RESULT_SCHEMA_ID
    )
    schema_version: _ScenarioSchemaVersion = SCENARIO_SCHEMA_VERSION
    measurement_id: _AsciiIdentifier
    measurement_version: _PositiveInt
    result_status: ScenarioResultStatus
    endpoint_observation_status: ScenarioEndpointObservationStatus
    value: ScenarioResultValueV1 | None
    reason: _AsciiText | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> ScenarioMeasurementResultV1:
        _validate_result_payload(
            result_status=self.result_status,
            endpoint_observation_status=self.endpoint_observation_status,
            value=self.value,
            reason=self.reason,
        )
        return self


class ScenarioViolationResultV1(EvaluationModel):
    """One caller-materialized scenario violation result."""

    schema_id: Literal["marl_battlegrounds.evaluation.scenario_violation_result"] = (
        SCENARIO_VIOLATION_RESULT_SCHEMA_ID
    )
    schema_version: _ScenarioSchemaVersion = SCENARIO_SCHEMA_VERSION
    violation_id: _AsciiIdentifier
    violation_version: _PositiveInt
    result_status: ScenarioResultStatus
    endpoint_observation_status: ScenarioEndpointObservationStatus
    value: ScenarioResultValueV1 | None
    reason: _AsciiText | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> ScenarioViolationResultV1:
        _validate_result_payload(
            result_status=self.result_status,
            endpoint_observation_status=self.endpoint_observation_status,
            value=self.value,
            reason=self.reason,
        )
        return self


class ScenarioPredicateResultV1(EvaluationModel):
    """The supplied outcome of one versioned scenario success predicate."""

    schema_id: Literal["marl_battlegrounds.evaluation.scenario_predicate_result"] = (
        SCENARIO_PREDICATE_RESULT_SCHEMA_ID
    )
    schema_version: _ScenarioSchemaVersion = SCENARIO_SCHEMA_VERSION
    predicate_id: _AsciiIdentifier
    predicate_version: _PositiveInt
    status: ScenarioPredicateStatus
    reason: _AsciiText | None = None

    @model_validator(mode="after")
    def _validate_status(self) -> ScenarioPredicateResultV1:
        if self.status == "unavailable":
            if self.reason is None:
                raise ValueError("unavailable predicate results require a reason")
        elif self.reason is not None:
            raise ValueError("available predicate results forbid a reason")
        return self


class ResolvedScenarioSpecificationV1(EvaluationModel):
    """Frozen pre-rollout semantics for one controlled scenario version."""

    schema_id: Literal[
        "marl_battlegrounds.evaluation.resolved_scenario_specification"
    ] = SCENARIO_SPECIFICATION_SCHEMA_ID
    schema_version: _ScenarioSchemaVersion = SCENARIO_SCHEMA_VERSION
    canonical_digest_sha256: _Sha256Hex
    scenario_id: _AsciiIdentifier
    scenario_version: _PositiveInt
    classification: ScenarioClassification
    hypothesis: _AsciiText
    eligible_roles: Annotated[
        tuple[ScenarioEvaluationRole, ...],
        Field(min_length=1, max_length=3),
    ]
    authored_initial_condition: ContentAddressedIdentityV1
    parameters: tuple[ScenarioParameterV1, ...] = ()
    resolved_config_digest_sha256: _Sha256Hex
    horizon: _PositiveInt
    pressure_protocol: ContentAddressedIdentityV1 | None = None
    primary_measurement: ScenarioMeasurementDefinitionV1
    secondary_measurements: Annotated[
        tuple[ScenarioMeasurementDefinitionV1, ...],
        Field(max_length=2),
    ] = ()
    violations: tuple[ScenarioViolationDefinitionV1, ...] = ()
    success_predicate: VersionedIdentityV1
    completion_policy: VersionedIdentityV1
    partial_result_policy: VersionedIdentityV1

    @model_validator(mode="after")
    def _validate_specification(self) -> ResolvedScenarioSpecificationV1:
        _require_stable_nested_model(
            self.authored_initial_condition,
            record_name="authored initial-condition identity",
            expected_types=(ContentAddressedIdentityV1,),
        )
        if self.pressure_protocol is not None:
            _require_stable_nested_model(
                self.pressure_protocol,
                record_name="pressure-protocol identity",
                expected_types=(ContentAddressedIdentityV1,),
            )
        for identity_name, identity in (
            ("success-predicate identity", self.success_predicate),
            ("completion-policy identity", self.completion_policy),
            ("partial-result-policy identity", self.partial_result_policy),
        ):
            _require_stable_nested_model(
                identity,
                record_name=identity_name,
                expected_types=(VersionedIdentityV1,),
            )
        for parameter in self.parameters:
            _require_stable_nested_model(
                parameter,
                record_name="scenario parameter",
                expected_types=(ScenarioParameterV1,),
            )
        for definition in (
            self.primary_measurement,
            *self.secondary_measurements,
        ):
            _require_stable_nested_model(
                definition,
                record_name="scenario measurement definition",
                expected_types=(ScenarioMeasurementDefinitionV1,),
            )
        for definition in self.violations:
            _require_stable_nested_model(
                definition,
                record_name="scenario violation definition",
                expected_types=(ScenarioViolationDefinitionV1,),
            )

        role_order = {
            "focal": 0,
            "cooperative_partner": 1,
            "adversarial_opponent": 2,
        }
        expected_roles = tuple(sorted(self.eligible_roles, key=role_order.__getitem__))
        if self.eligible_roles != expected_roles:
            raise ValueError("eligible scenario roles must be canonically sorted")
        if len(self.eligible_roles) != len(set(self.eligible_roles)):
            raise ValueError("eligible scenario roles must be unique")

        parameter_names = tuple(row.name for row in self.parameters)
        if parameter_names != tuple(sorted(parameter_names)):
            raise ValueError("scenario parameters must be canonically sorted")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("scenario parameter names must be unique")

        if self.primary_measurement.role != "primary":
            raise ValueError("primary_measurement must declare the primary role")
        if any(row.role != "secondary" for row in self.secondary_measurements):
            raise ValueError("secondary_measurements must declare the secondary role")
        secondary_keys = tuple(
            (row.measurement_id, row.measurement_version)
            for row in self.secondary_measurements
        )
        if secondary_keys != tuple(sorted(secondary_keys)):
            raise ValueError("secondary measurements must be canonically sorted")
        measurement_ids = (
            self.primary_measurement.measurement_id,
            *(row.measurement_id for row in self.secondary_measurements),
        )
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError("scenario measurement IDs must be unique")

        violation_keys = tuple(
            (row.violation_id, row.violation_version) for row in self.violations
        )
        if violation_keys != tuple(sorted(violation_keys)):
            raise ValueError("scenario violations must be canonically sorted")
        violation_ids = tuple(row.violation_id for row in self.violations)
        if len(violation_ids) != len(set(violation_ids)):
            raise ValueError("scenario violation IDs must be unique")

        expected_digest = canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        )
        if self.canonical_digest_sha256 != expected_digest:
            raise ValueError("resolved scenario specification digest mismatch")
        return self


def _measurement_definition_key(
    definition: ScenarioMeasurementDefinitionV1,
) -> tuple[str, int]:
    return definition.measurement_id, definition.measurement_version


def _measurement_result_key(
    result: ScenarioMeasurementResultV1,
) -> tuple[str, int]:
    return result.measurement_id, result.measurement_version


def _violation_definition_key(
    definition: ScenarioViolationDefinitionV1,
) -> tuple[str, int]:
    return definition.violation_id, definition.violation_version


def _violation_result_key(
    result: ScenarioViolationResultV1,
) -> tuple[str, int]:
    return result.violation_id, result.violation_version


class ScenarioEvaluationRecordV1(EvaluationModel):
    """Content-addressed scenario results joined to replay/report evidence."""

    schema_id: Literal["marl_battlegrounds.evaluation.scenario_evaluation_record"] = (
        SCENARIO_EVALUATION_RECORD_SCHEMA_ID
    )
    schema_version: _ScenarioSchemaVersion = SCENARIO_SCHEMA_VERSION
    record_id: _AsciiIdentifier
    canonical_digest_sha256: _Sha256Hex
    specification: ResolvedScenarioSpecificationV1
    replay_reference: ReplayArtifactReferenceV1
    metric_report_reference: MetricReportReferenceV1
    realized_initial_frame_digest_sha256: _Sha256Hex
    measurement_results: tuple[ScenarioMeasurementResultV1, ...]
    violation_results: tuple[ScenarioViolationResultV1, ...]
    predicate_result: ScenarioPredicateResultV1

    @model_validator(mode="after")
    def _validate_record(self) -> ScenarioEvaluationRecordV1:
        _require_stable_nested_model(
            self.specification,
            record_name="scenario specification",
            expected_types=(ResolvedScenarioSpecificationV1,),
        )
        _require_stable_nested_model(
            self.replay_reference,
            record_name="scenario replay reference",
            expected_types=(ReplayArtifactReferenceV1,),
        )
        _require_stable_nested_model(
            self.metric_report_reference,
            record_name="scenario metric-report reference",
            expected_types=(MetricReportReferenceV1,),
        )
        for result in self.measurement_results:
            _require_stable_nested_model(
                result,
                record_name="scenario measurement result",
                expected_types=(ScenarioMeasurementResultV1,),
            )
        for result in self.violation_results:
            _require_stable_nested_model(
                result,
                record_name="scenario violation result",
                expected_types=(ScenarioViolationResultV1,),
            )
        _require_stable_nested_model(
            self.predicate_result,
            record_name="scenario predicate result",
            expected_types=(ScenarioPredicateResultV1,),
        )

        episode_id = self.replay_reference.episode_id
        if self.record_id != f"{episode_id}:scenario-evaluation":
            raise ValueError("scenario evaluation record ID is not canonical")
        if self.metric_report_reference.episode_id != episode_id:
            raise ValueError("scenario replay and metric report episodes must match")
        if (
            self.metric_report_reference.trajectory_content_digest_sha256
            != self.replay_reference.trajectory_content_digest_sha256
        ):
            raise ValueError(
                "scenario replay and metric report must join the same trajectory"
            )

        definitions = (
            self.specification.primary_measurement,
            *self.specification.secondary_measurements,
        )
        definition_keys = tuple(map(_measurement_definition_key, definitions))
        result_keys = tuple(map(_measurement_result_key, self.measurement_results))
        if result_keys != definition_keys:
            raise ValueError(
                "scenario measurement results must exactly follow their definitions"
            )
        for definition, result in zip(
            definitions,
            self.measurement_results,
            strict=True,
        ):
            if (
                result.value is not None
                and result.value.value_type != definition.value_type
            ):
                raise ValueError(
                    "scenario measurement result value type must match its definition"
                )
            if (
                result.endpoint_observation_status == "right_censored"
                and not definition.supports_right_censoring
            ):
                raise ValueError("right-censored measurement requires declared support")

        violation_definition_keys = tuple(
            map(_violation_definition_key, self.specification.violations)
        )
        violation_result_keys = tuple(
            map(_violation_result_key, self.violation_results)
        )
        if violation_result_keys != violation_definition_keys:
            raise ValueError(
                "scenario violation results must exactly follow their definitions"
            )
        for definition, result in zip(
            self.specification.violations,
            self.violation_results,
            strict=True,
        ):
            if (
                result.value is not None
                and result.value.value_type != definition.value_type
            ):
                raise ValueError(
                    "scenario violation result value type must match its definition"
                )
            if (
                result.endpoint_observation_status == "right_censored"
                and not definition.supports_right_censoring
            ):
                raise ValueError("right-censored violation requires declared support")
        if (
            self.predicate_result.predicate_id
            != self.specification.success_predicate.identifier
            or self.predicate_result.predicate_version
            != self.specification.success_predicate.version
        ):
            raise ValueError(
                "scenario predicate result must match the declared success predicate"
            )

        expected_digest = canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        )
        if self.canonical_digest_sha256 != expected_digest:
            raise ValueError("scenario evaluation record digest mismatch")
        return self


def _validate_scenario_evaluation_record_against_validated_replay_v1(
    canonical_record: ScenarioEvaluationRecordV1,
    replay: ReplayArtifactV1,
    expected_replay_reference: ReplayArtifactReferenceV1,
) -> None:
    """Validate scenario joins after replay and sidecar validation."""
    if canonical_record.replay_reference != expected_replay_reference:
        raise ValueError("scenario replay reference does not match replay content")
    if canonical_record.metric_report_reference != replay.metric_report_reference:
        raise ValueError(
            "scenario metric-report reference does not match replay content"
        )

    context = replay.header.context
    if context.capture_profile != "scenario_metric_complete":
        raise ValueError(
            "scenario evaluation requires the scenario_metric_complete profile"
        )
    scenario_identity = context.identity.scenario
    if scenario_identity is None:
        raise ValueError("scenario replay context requires a scenario identity")
    specification = canonical_record.specification
    if (
        scenario_identity.identifier != specification.scenario_id
        or scenario_identity.version != specification.scenario_version
        or scenario_identity.canonical_digest != specification.canonical_digest_sha256
    ):
        raise ValueError(
            "scenario specification must equal the context scenario identity"
        )
    if (
        specification.resolved_config_digest_sha256
        != context.resolved_env_config.canonical_digest_sha256
    ):
        raise ValueError("scenario specification must join the resolved config")
    if specification.horizon != context.expected_horizon:
        raise ValueError("scenario horizon must equal the replay context horizon")

    configured_roles = {
        assignment.evaluation_role
        for assignment in context.policy_assignments
        if isinstance(assignment, AssignedPolicySlotV1)
    }
    if not set(specification.eligible_roles).issubset(configured_roles):
        raise ValueError("scenario eligible roles must join assigned policies")

    expected_initial_digest = canonical_digest_sha256(replay.frames[0])
    if canonical_record.realized_initial_frame_digest_sha256 != expected_initial_digest:
        raise ValueError(
            "scenario realized initial-frame digest does not match replay frame zero"
        )

    if replay.completion.completion_state != "complete":
        definitions = (
            specification.primary_measurement,
            *specification.secondary_measurements,
        )
        for definition, result in zip(
            definitions,
            canonical_record.measurement_results,
            strict=True,
        ):
            if (
                definition.completion_scope == "complete_episode"
                and result.result_status == "defined"
            ):
                raise ValueError(
                    "complete-episode measurement cannot be defined from a partial "
                    "replay"
                )
        for definition, result in zip(
            specification.violations,
            canonical_record.violation_results,
            strict=True,
        ):
            if (
                definition.completion_scope == "complete_episode"
                and result.result_status == "defined"
            ):
                raise ValueError(
                    "complete-episode violation cannot be defined from a partial replay"
                )

    is_complete = replay.completion.completion_state == "complete"
    has_declared_horizon = "declared_horizon" in replay.completion.completion_bases
    for result in (
        *canonical_record.measurement_results,
        *canonical_record.violation_results,
    ):
        if result.endpoint_observation_status == "right_censored" and (
            not is_complete or not has_declared_horizon
        ):
            raise ValueError(
                "right-censored scenario results require complete declared-horizon "
                "evidence"
            )
        if result.endpoint_observation_status == "competing_event" and not is_complete:
            raise ValueError(
                "competing-event scenario results require a complete rollout"
            )


def validate_scenario_evaluation_record_v1(
    record: ScenarioEvaluationRecordV1,
    replay: ReplayArtifactV1,
    metric_report_artifact: EvaluationMetricReportArtifactV1,
) -> None:
    """Validate all scenario joins without evaluating scenario semantics."""
    canonical_record = cast(
        ScenarioEvaluationRecordV1,
        validate_declared_model_tree(
            record,
            record_name="scenario evaluation record",
            expected_type=ScenarioEvaluationRecordV1,
        ),
    )
    validate_replay_artifact_v1(replay)
    _validate_metric_report_artifact_against_validated_replay_v1(
        metric_report_artifact,
        replay,
    )
    _validate_scenario_evaluation_record_against_validated_replay_v1(
        canonical_record,
        replay,
        _build_replay_artifact_reference_from_validated_v1(replay),
    )


def build_scenario_evaluation_record_v1(
    specification: ResolvedScenarioSpecificationV1,
    replay: ReplayArtifactV1,
    metric_report_artifact: EvaluationMetricReportArtifactV1,
    *,
    measurement_results: tuple[ScenarioMeasurementResultV1, ...],
    violation_results: tuple[ScenarioViolationResultV1, ...],
    predicate_result: ScenarioPredicateResultV1,
) -> ScenarioEvaluationRecordV1:
    """Build and cross-validate a content-addressed scenario result record."""
    validate_replay_artifact_v1(replay)
    _validate_metric_report_artifact_against_validated_replay_v1(
        metric_report_artifact,
        replay,
    )
    replay_reference = _build_replay_artifact_reference_from_validated_v1(replay)
    episode_id = replay.header.context.identity.episode_id
    payload: dict[str, object] = {
        "schema_id": SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "record_id": f"{episode_id}:scenario-evaluation",
        "specification": specification,
        "replay_reference": replay_reference,
        "metric_report_reference": replay.metric_report_reference,
        "realized_initial_frame_digest_sha256": canonical_digest_sha256(
            replay.frames[0]
        ),
        "measurement_results": measurement_results,
        "violation_results": violation_results,
        "predicate_result": predicate_result,
    }
    record = ScenarioEvaluationRecordV1.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )
    _validate_scenario_evaluation_record_against_validated_replay_v1(
        record,
        replay,
        replay_reference,
    )
    return record


_RESOLVED_INITIAL_STATE_DIGEST_DOMAIN_V2 = (
    "marl_battlegrounds.evaluation.resolved_scenario_initial_state.v2"
)


def _resolved_initial_state_digest_from_validated_frame_v2(
    frame: EvaluationFrameV1,
) -> str:
    return canonical_digest_sha256(
        {
            "digest_domain": _RESOLVED_INITIAL_STATE_DIGEST_DOMAIN_V2,
            "simulator_step_count": frame.simulator_step_count,
            "snapshot": frame.snapshot,
        }
    )


def resolved_initial_state_digest_sha256_v2(frame: EvaluationFrameV1) -> str:
    """Hash only the stable simulator epoch and privileged state at frame zero."""
    canonical_frame = cast(
        EvaluationFrameV1,
        validate_declared_model_tree(
            frame,
            record_name="scenario resolved initial-state frame",
            expected_type=EvaluationFrameV1,
        ),
    )
    return _resolved_initial_state_digest_from_validated_frame_v2(canonical_frame)


class ScenarioSeedScheduleV2(EvaluationModel):
    """One immutable ordered schedule of matched realized seed records."""

    schema_id: Literal["marl_battlegrounds.evaluation.scenario_seed_schedule"] = (
        SCENARIO_SEED_SCHEDULE_SCHEMA_ID
    )
    schema_version: _ScenarioSchemaVersionV2 = SCENARIO_SCHEMA_VERSION_V2
    schedule_id: _AsciiIdentifier
    schedule_version: _PositiveInt
    canonical_digest_sha256: _Sha256Hex
    realized_seed_protocols: Annotated[
        tuple[EvaluationSeedProtocolV1, ...],
        Field(min_length=1),
    ]

    @model_validator(mode="after")
    def _validate_schedule(self) -> ScenarioSeedScheduleV2:
        for seed_protocol in self.realized_seed_protocols:
            _require_stable_nested_model(
                seed_protocol,
                record_name="scenario schedule seed protocol",
                expected_types=(EvaluationSeedProtocolV1,),
            )

        protocol_identity = self.realized_seed_protocols[0].seed_protocol
        if any(
            row.seed_protocol != protocol_identity
            for row in self.realized_seed_protocols[1:]
        ):
            raise ValueError(
                "scenario schedule rows must share one seed-protocol identity"
            )
        row_digests = tuple(
            canonical_digest_sha256(row) for row in self.realized_seed_protocols
        )
        if len(row_digests) != len(set(row_digests)):
            raise ValueError("scenario schedule rows must be unique")

        expected_digest = canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        )
        if self.canonical_digest_sha256 != expected_digest:
            raise ValueError("scenario seed schedule digest mismatch")
        return self


class ResolvedScenarioSpecificationV2(EvaluationModel):
    """Frozen fixed-slot scenario semantics independent of runtime policies."""

    schema_id: Literal[
        "marl_battlegrounds.evaluation.resolved_scenario_specification"
    ] = SCENARIO_SPECIFICATION_SCHEMA_ID
    schema_version: _ScenarioSchemaVersionV2 = SCENARIO_SCHEMA_VERSION_V2
    canonical_digest_sha256: _Sha256Hex
    scenario_id: _AsciiIdentifier
    scenario_version: _PositiveInt
    classification: ScenarioClassification
    hypothesis: _AsciiText
    layout: ContentAddressedIdentityV1
    authored_initial_condition: ContentAddressedIdentityV1
    resolved_initial_state_digest_sha256: _Sha256Hex
    parameters: tuple[ScenarioParameterV1, ...] = ()
    resolved_config_digest_sha256: _Sha256Hex
    roster_template: Annotated[
        tuple[RosterSlotV1, ...],
        Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS),
    ]
    role_template: Annotated[
        tuple[ScenarioFixedSlotRoleV2, ...],
        Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS),
    ]
    seed_schedule: ScenarioSeedScheduleV2
    horizon: _PositiveInt
    pressure_protocol: ContentAddressedIdentityV1 | None = None
    primary_measurement: ScenarioMeasurementDefinitionV1
    secondary_measurements: Annotated[
        tuple[ScenarioMeasurementDefinitionV1, ...],
        Field(max_length=2),
    ] = ()
    violations: tuple[ScenarioViolationDefinitionV1, ...] = ()
    success_predicate: VersionedIdentityV1
    completion_policy: VersionedIdentityV1
    partial_result_policy: VersionedIdentityV1

    @model_validator(mode="after")
    def _validate_specification(self) -> ResolvedScenarioSpecificationV2:
        for identity_name, identity in (
            ("layout identity", self.layout),
            ("authored initial-condition identity", self.authored_initial_condition),
        ):
            _require_stable_nested_model(
                identity,
                record_name=identity_name,
                expected_types=(ContentAddressedIdentityV1,),
            )
        if self.pressure_protocol is not None:
            _require_stable_nested_model(
                self.pressure_protocol,
                record_name="pressure-protocol identity",
                expected_types=(ContentAddressedIdentityV1,),
            )
        for identity_name, identity in (
            ("success-predicate identity", self.success_predicate),
            ("completion-policy identity", self.completion_policy),
            ("partial-result-policy identity", self.partial_result_policy),
        ):
            _require_stable_nested_model(
                identity,
                record_name=identity_name,
                expected_types=(VersionedIdentityV1,),
            )
        for parameter in self.parameters:
            _require_stable_nested_model(
                parameter,
                record_name="scenario parameter",
                expected_types=(ScenarioParameterV1,),
            )
        for definition in (
            self.primary_measurement,
            *self.secondary_measurements,
        ):
            _require_stable_nested_model(
                definition,
                record_name="scenario measurement definition",
                expected_types=(ScenarioMeasurementDefinitionV1,),
            )
        for definition in self.violations:
            _require_stable_nested_model(
                definition,
                record_name="scenario violation definition",
                expected_types=(ScenarioViolationDefinitionV1,),
            )
        _require_stable_nested_model(
            self.seed_schedule,
            record_name="scenario seed schedule",
            expected_types=(ScenarioSeedScheduleV2,),
        )
        for roster_row in self.roster_template:
            _require_stable_nested_model(
                roster_row,
                record_name="scenario roster row",
                expected_types=(RosterSlotV1,),
            )

        parameter_names = tuple(row.name for row in self.parameters)
        if parameter_names != tuple(sorted(parameter_names)):
            raise ValueError("scenario parameters must be canonically sorted")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("scenario parameter names must be unique")

        if self.primary_measurement.role != "primary":
            raise ValueError("primary_measurement must declare the primary role")
        if any(row.role != "secondary" for row in self.secondary_measurements):
            raise ValueError("secondary_measurements must declare the secondary role")
        secondary_keys = tuple(
            (row.measurement_id, row.measurement_version)
            for row in self.secondary_measurements
        )
        if secondary_keys != tuple(sorted(secondary_keys)):
            raise ValueError("secondary measurements must be canonically sorted")
        measurement_ids = (
            self.primary_measurement.measurement_id,
            *(row.measurement_id for row in self.secondary_measurements),
        )
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError("scenario measurement IDs must be unique")

        violation_keys = tuple(
            (row.violation_id, row.violation_version) for row in self.violations
        )
        if violation_keys != tuple(sorted(violation_keys)):
            raise ValueError("scenario violations must be canonically sorted")
        violation_ids = tuple(row.violation_id for row in self.violations)
        if len(violation_ids) != len(set(violation_ids)):
            raise ValueError("scenario violation IDs must be unique")

        slots = tuple(row.global_slot for row in self.roster_template)
        if slots != tuple(range(MAX_AGENT_SLOTS)):
            raise ValueError("scenario roster must contain ordered fixed global slots")
        public_agent_ids = tuple(row.public_agent_id for row in self.roster_template)
        if len(public_agent_ids) != len(set(public_agent_ids)):
            raise ValueError("scenario roster public agent IDs must be unique")
        for roster_row, role in zip(
            self.roster_template,
            self.role_template,
            strict=True,
        ):
            if (
                roster_row.team_local_slot
                != roster_row.global_slot % MAX_AGENTS_PER_TEAM
            ):
                raise ValueError("scenario roster must follow fixed team-local slots")
            expected_team_id = 1 if roster_row.global_slot < MAX_AGENTS_PER_TEAM else 2
            if roster_row.configured_active:
                if (
                    roster_row.configured_team_id != expected_team_id
                    or roster_row.class_id == 0
                ):
                    raise ValueError(
                        "active scenario roster rows require fixed team and class"
                    )
                if role == "not_applicable":
                    raise ValueError(
                        "active scenario roster rows require one evaluation role"
                    )
            else:
                if roster_row.configured_team_id != 0 or roster_row.class_id != 0:
                    raise ValueError(
                        "inactive scenario roster rows require neutral team and class"
                    )
                if role != "not_applicable":
                    raise ValueError(
                        "inactive scenario roster rows require not_applicable role"
                    )

        active_roles = set(self.role_template) - {"not_applicable"}
        if "focal" not in active_roles:
            raise ValueError("scenario role template requires at least one focal slot")
        for seed_row in self.seed_schedule.realized_seed_protocols:
            role_seed_pairs = (
                (
                    "cooperative_partner",
                    seed_row.cooperative_partner_seed,
                ),
                (
                    "adversarial_opponent",
                    seed_row.adversarial_opponent_seed,
                ),
            )
            for role, seed in role_seed_pairs:
                if (role in active_roles) != (seed != "not_applicable"):
                    raise ValueError(
                        f"{role} seed presence must match scenario role template"
                    )
            if seed_row.scenario_seed == "not_applicable":
                raise ValueError("scenario schedule rows require a scenario seed")

        expected_digest = canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        )
        if self.canonical_digest_sha256 != expected_digest:
            raise ValueError("resolved scenario specification digest mismatch")
        return self


class ScenarioEvaluationRecordV2(EvaluationModel):
    """V2 scenario results joined to fixed-slot and matched-seed evidence."""

    schema_id: Literal["marl_battlegrounds.evaluation.scenario_evaluation_record"] = (
        SCENARIO_EVALUATION_RECORD_SCHEMA_ID
    )
    schema_version: _ScenarioSchemaVersionV2 = SCENARIO_SCHEMA_VERSION_V2
    record_id: _AsciiIdentifier
    canonical_digest_sha256: _Sha256Hex
    specification: ResolvedScenarioSpecificationV2
    schedule_coordinate: _NonNegativeInt
    replay_reference: ReplayArtifactReferenceV1
    metric_report_reference: MetricReportReferenceV1
    realized_initial_frame_digest_sha256: _Sha256Hex
    measurement_results: tuple[ScenarioMeasurementResultV1, ...]
    violation_results: tuple[ScenarioViolationResultV1, ...]
    predicate_result: ScenarioPredicateResultV1

    @model_validator(mode="after")
    def _validate_record(self) -> ScenarioEvaluationRecordV2:
        _require_stable_nested_model(
            self.specification,
            record_name="scenario specification",
            expected_types=(ResolvedScenarioSpecificationV2,),
        )
        _require_stable_nested_model(
            self.replay_reference,
            record_name="scenario replay reference",
            expected_types=(ReplayArtifactReferenceV1,),
        )
        _require_stable_nested_model(
            self.metric_report_reference,
            record_name="scenario metric-report reference",
            expected_types=(MetricReportReferenceV1,),
        )
        for result in self.measurement_results:
            _require_stable_nested_model(
                result,
                record_name="scenario measurement result",
                expected_types=(ScenarioMeasurementResultV1,),
            )
        for result in self.violation_results:
            _require_stable_nested_model(
                result,
                record_name="scenario violation result",
                expected_types=(ScenarioViolationResultV1,),
            )
        _require_stable_nested_model(
            self.predicate_result,
            record_name="scenario predicate result",
            expected_types=(ScenarioPredicateResultV1,),
        )

        episode_id = self.replay_reference.episode_id
        if self.record_id != f"{episode_id}:scenario-evaluation":
            raise ValueError("scenario evaluation record ID is not canonical")
        if self.metric_report_reference.episode_id != episode_id:
            raise ValueError("scenario replay and metric report episodes must match")
        if (
            self.metric_report_reference.trajectory_content_digest_sha256
            != self.replay_reference.trajectory_content_digest_sha256
        ):
            raise ValueError(
                "scenario replay and metric report must join the same trajectory"
            )
        if self.schedule_coordinate >= len(
            self.specification.seed_schedule.realized_seed_protocols
        ):
            raise ValueError("scenario schedule coordinate is out of range")

        definitions = (
            self.specification.primary_measurement,
            *self.specification.secondary_measurements,
        )
        definition_keys = tuple(map(_measurement_definition_key, definitions))
        result_keys = tuple(map(_measurement_result_key, self.measurement_results))
        if result_keys != definition_keys:
            raise ValueError(
                "scenario measurement results must exactly follow their definitions"
            )
        for definition, result in zip(
            definitions,
            self.measurement_results,
            strict=True,
        ):
            if (
                result.value is not None
                and result.value.value_type != definition.value_type
            ):
                raise ValueError(
                    "scenario measurement result value type must match its definition"
                )
            if (
                result.endpoint_observation_status == "right_censored"
                and not definition.supports_right_censoring
            ):
                raise ValueError("right-censored measurement requires declared support")

        violation_definition_keys = tuple(
            map(_violation_definition_key, self.specification.violations)
        )
        violation_result_keys = tuple(
            map(_violation_result_key, self.violation_results)
        )
        if violation_result_keys != violation_definition_keys:
            raise ValueError(
                "scenario violation results must exactly follow their definitions"
            )
        for definition, result in zip(
            self.specification.violations,
            self.violation_results,
            strict=True,
        ):
            if (
                result.value is not None
                and result.value.value_type != definition.value_type
            ):
                raise ValueError(
                    "scenario violation result value type must match its definition"
                )
            if (
                result.endpoint_observation_status == "right_censored"
                and not definition.supports_right_censoring
            ):
                raise ValueError("right-censored violation requires declared support")
        if (
            self.predicate_result.predicate_id
            != self.specification.success_predicate.identifier
            or self.predicate_result.predicate_version
            != self.specification.success_predicate.version
        ):
            raise ValueError(
                "scenario predicate result must match the declared success predicate"
            )

        expected_digest = canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        )
        if self.canonical_digest_sha256 != expected_digest:
            raise ValueError("scenario evaluation record digest mismatch")
        return self


def _validate_scenario_evaluation_record_against_validated_replay_v2(
    canonical_record: ScenarioEvaluationRecordV2,
    replay: ReplayArtifactV1,
    expected_replay_reference: ReplayArtifactReferenceV1,
) -> None:
    """Validate V2 evidence joins after one replay and sidecar scan."""
    if canonical_record.replay_reference != expected_replay_reference:
        raise ValueError("scenario replay reference does not match replay content")
    if canonical_record.metric_report_reference != replay.metric_report_reference:
        raise ValueError(
            "scenario metric-report reference does not match replay content"
        )

    context = replay.header.context
    if context.capture_profile != "scenario_metric_complete":
        raise ValueError(
            "scenario evaluation requires the scenario_metric_complete profile"
        )
    scenario_identity = context.identity.scenario
    if scenario_identity is None:
        raise ValueError("scenario replay context requires a scenario identity")
    specification = canonical_record.specification
    if (
        scenario_identity.identifier != specification.scenario_id
        or scenario_identity.version != specification.scenario_version
        or scenario_identity.canonical_digest != specification.canonical_digest_sha256
    ):
        raise ValueError(
            "scenario specification must equal the context scenario identity"
        )
    if context.identity.layout != specification.layout:
        raise ValueError("scenario specification must join the context layout identity")
    if (
        specification.resolved_config_digest_sha256
        != context.resolved_env_config.canonical_digest_sha256
    ):
        raise ValueError("scenario specification must join the resolved config")
    if specification.roster_template != context.roster:
        raise ValueError(
            "scenario roster template must equal the replay context roster"
        )
    if specification.horizon != context.expected_horizon:
        raise ValueError("scenario horizon must equal the replay context horizon")

    for frozen_role, policy_assignment in zip(
        specification.role_template,
        context.policy_assignments,
        strict=True,
    ):
        if frozen_role == "not_applicable":
            if not isinstance(policy_assignment, NotApplicablePolicySlotV1):
                raise ValueError(
                    "inactive scenario role slots require not-applicable policies"
                )
        elif (
            not isinstance(policy_assignment, AssignedPolicySlotV1)
            or policy_assignment.evaluation_role != frozen_role
        ):
            raise ValueError(
                "assigned policy role must equal the frozen scenario slot role"
            )

    expected_seed_protocol = specification.seed_schedule.realized_seed_protocols[
        canonical_record.schedule_coordinate
    ]
    if context.seed_protocol != expected_seed_protocol:
        raise ValueError(
            "scenario schedule coordinate must select the context seed protocol"
        )

    initial_frame = replay.frames[0]
    expected_resolved_initial_state_digest = (
        _resolved_initial_state_digest_from_validated_frame_v2(initial_frame)
    )
    if (
        specification.resolved_initial_state_digest_sha256
        != expected_resolved_initial_state_digest
    ):
        raise ValueError(
            "scenario resolved initial-state digest does not match replay frame zero"
        )
    expected_initial_frame_digest = canonical_digest_sha256(initial_frame)
    if (
        canonical_record.realized_initial_frame_digest_sha256
        != expected_initial_frame_digest
    ):
        raise ValueError(
            "scenario realized initial-frame digest does not match replay frame zero"
        )

    if replay.completion.completion_state != "complete":
        definitions = (
            specification.primary_measurement,
            *specification.secondary_measurements,
        )
        for definition, result in zip(
            definitions,
            canonical_record.measurement_results,
            strict=True,
        ):
            if (
                definition.completion_scope == "complete_episode"
                and result.result_status == "defined"
            ):
                raise ValueError(
                    "complete-episode measurement cannot be defined from a partial "
                    "replay"
                )
        for definition, result in zip(
            specification.violations,
            canonical_record.violation_results,
            strict=True,
        ):
            if (
                definition.completion_scope == "complete_episode"
                and result.result_status == "defined"
            ):
                raise ValueError(
                    "complete-episode violation cannot be defined from a partial replay"
                )

    is_complete = replay.completion.completion_state == "complete"
    has_declared_horizon = "declared_horizon" in replay.completion.completion_bases
    for result in (
        *canonical_record.measurement_results,
        *canonical_record.violation_results,
    ):
        if result.endpoint_observation_status == "right_censored" and (
            not is_complete or not has_declared_horizon
        ):
            raise ValueError(
                "right-censored scenario results require complete declared-horizon "
                "evidence"
            )
        if result.endpoint_observation_status == "competing_event" and not is_complete:
            raise ValueError(
                "competing-event scenario results require a complete rollout"
            )


def _validate_official_scenario_context_for_replay_v2(
    replay: ReplayArtifactV1,
) -> None:
    from marl_battlegrounds.evaluation.catalog import (
        _validate_official_scenario_context_v2,  # pyright: ignore[reportPrivateUsage]
    )

    _validate_official_scenario_context_v2(
        replay.header.context,
        replay.frames[0],
    )


def _validate_scenario_evaluation_record_and_evidence_v2(
    record: ScenarioEvaluationRecordV2,
    replay: ReplayArtifactV1,
    metric_report_artifact: EvaluationMetricReportArtifactV1,
    *,
    require_official_context: bool,
) -> None:
    canonical_record = cast(
        ScenarioEvaluationRecordV2,
        validate_declared_model_tree(
            record,
            record_name="scenario evaluation record",
            expected_type=ScenarioEvaluationRecordV2,
        ),
    )
    validate_replay_artifact_v1(replay)
    _validate_metric_report_artifact_against_validated_replay_v1(
        metric_report_artifact,
        replay,
    )
    _validate_scenario_evaluation_record_against_validated_replay_v2(
        canonical_record,
        replay,
        _build_replay_artifact_reference_from_validated_v1(replay),
    )
    if require_official_context:
        _validate_official_scenario_context_for_replay_v2(replay)


def validate_scenario_evaluation_record_v2(
    record: ScenarioEvaluationRecordV2,
    replay: ReplayArtifactV1,
    metric_report_artifact: EvaluationMetricReportArtifactV1,
) -> None:
    """Validate V2 artifact/evidence joins without importing simulator code."""
    _validate_scenario_evaluation_record_and_evidence_v2(
        record,
        replay,
        metric_report_artifact,
        require_official_context=False,
    )


def validate_official_scenario_evaluation_record_v2(
    record: ScenarioEvaluationRecordV2,
    replay: ReplayArtifactV1,
    metric_report_artifact: EvaluationMetricReportArtifactV1,
) -> None:
    """Validate V2 evidence and the carried context against live core rules."""
    _validate_scenario_evaluation_record_and_evidence_v2(
        record,
        replay,
        metric_report_artifact,
        require_official_context=True,
    )


def build_scenario_evaluation_record_v2(
    specification: ResolvedScenarioSpecificationV2,
    replay: ReplayArtifactV1,
    metric_report_artifact: EvaluationMetricReportArtifactV1,
    *,
    schedule_coordinate: int,
    measurement_results: tuple[ScenarioMeasurementResultV1, ...],
    violation_results: tuple[ScenarioViolationResultV1, ...],
    predicate_result: ScenarioPredicateResultV1,
) -> ScenarioEvaluationRecordV2:
    """Build V2 scenario evidence and require official context validity."""
    validate_replay_artifact_v1(replay)
    _validate_metric_report_artifact_against_validated_replay_v1(
        metric_report_artifact,
        replay,
    )
    replay_reference = _build_replay_artifact_reference_from_validated_v1(replay)
    episode_id = replay.header.context.identity.episode_id
    payload: dict[str, object] = {
        "schema_id": SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
        "schema_version": SCENARIO_SCHEMA_VERSION_V2,
        "record_id": f"{episode_id}:scenario-evaluation",
        "specification": specification,
        "schedule_coordinate": schedule_coordinate,
        "replay_reference": replay_reference,
        "metric_report_reference": replay.metric_report_reference,
        "realized_initial_frame_digest_sha256": canonical_digest_sha256(
            replay.frames[0]
        ),
        "measurement_results": measurement_results,
        "violation_results": violation_results,
        "predicate_result": predicate_result,
    }
    record = ScenarioEvaluationRecordV2.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )
    _validate_scenario_evaluation_record_against_validated_replay_v2(
        record,
        replay,
        replay_reference,
    )
    _validate_official_scenario_context_for_replay_v2(replay)
    return record


__all__ = [
    "SCENARIO_EVALUATION_RECORD_SCHEMA_ID",
    "SCENARIO_MEASUREMENT_DEFINITION_SCHEMA_ID",
    "SCENARIO_MEASUREMENT_RESULT_SCHEMA_ID",
    "SCENARIO_PREDICATE_RESULT_SCHEMA_ID",
    "SCENARIO_SCHEMA_VERSION",
    "SCENARIO_SCHEMA_VERSION_V2",
    "SCENARIO_SEED_SCHEDULE_SCHEMA_ID",
    "SCENARIO_SPECIFICATION_SCHEMA_ID",
    "SCENARIO_VIOLATION_DEFINITION_SCHEMA_ID",
    "SCENARIO_VIOLATION_RESULT_SCHEMA_ID",
    "ResolvedScenarioSpecificationV1",
    "ResolvedScenarioSpecificationV2",
    "ScenarioBooleanValueV1",
    "ScenarioClassification",
    "ScenarioCompletionScope",
    "ScenarioCountValueV1",
    "ScenarioEndpointObservationStatus",
    "ScenarioEvaluationRecordV1",
    "ScenarioEvaluationRecordV2",
    "ScenarioEvaluationRole",
    "ScenarioFixedSlotRoleV2",
    "ScenarioMeasurementDefinitionV1",
    "ScenarioMeasurementResultV1",
    "ScenarioMeasurementRole",
    "ScenarioParameterV1",
    "ScenarioPredicateResultV1",
    "ScenarioPredicateStatus",
    "ScenarioResultStatus",
    "ScenarioResultValueV1",
    "ScenarioScalarValueV1",
    "ScenarioSeedScheduleV2",
    "ScenarioValueType",
    "ScenarioViolationDefinitionV1",
    "ScenarioViolationResultV1",
    "build_scenario_evaluation_record_v1",
    "build_scenario_evaluation_record_v2",
    "resolved_initial_state_digest_sha256_v2",
    "validate_official_scenario_evaluation_record_v2",
    "validate_scenario_evaluation_record_v1",
    "validate_scenario_evaluation_record_v2",
]
