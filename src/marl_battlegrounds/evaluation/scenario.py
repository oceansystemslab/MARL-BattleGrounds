"""Declare controlled scenarios and join supplied results to replay evidence.

Specifications freeze the hypothesis, initial conditions, roles, seed schedule,
and endpoint contracts before a rollout. Result records store caller-computed
measurements and predicate outcomes. This module checks types, identities,
completion eligibility, and evidence joins; it does not compute a tactical
measurement or decide whether a hypothesis is true.

Record V1/V2 use replay V1 plus a metric sidecar. Record V3 uses replay V2.
Record V4 uses replay V3 and frame V2 (before Red Zone). Current record V5
uses replay V4, whose context records the Team Deathmatch Red Zone depth.
Generic validation stays on the host without current simulator admission.
Official validation additionally imports current catalog/Core checks and
requires canonical SharedObs coverage.
"""

from __future__ import annotations

from typing import Annotated, Literal, cast

from pydantic import BeforeValidator, Field, StringConstraints, model_validator

from marl_battlegrounds.evaluation.models import (
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    AssignedPolicySlotV1,
    AssignedPolicySlotV2,
    ContentAddressedIdentityV1,
    EvaluationFrame,
    EvaluationModel,
    EvaluationSeedProtocolV1,
    EvaluationSeedProtocolV2,
    GlobalAnalysisSnapshotV1,
    NotApplicablePolicySlotV1,
    RosterSlotV1,
    VersionedIdentityV1,
    canonical_digest_sha256,
    evaluation_frame_type,
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
from marl_battlegrounds.evaluation.replay_v4 import (
    ReplayArtifactReferenceV4,
    ReplayArtifactV4,
    replay_reference_v4,
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
SCENARIO_SCHEMA_VERSION_V3: Literal[3] = 3
SCENARIO_SCHEMA_VERSION_V4: Literal[4] = 4
SCENARIO_SCHEMA_VERSION_V5: Literal[5] = 5

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
    """Return exact Python integer 1; reject bool and other values with ValueError."""
    if type(value) is not int or value != 1:
        raise ValueError("schema_version must be the exact integer 1")
    return value


_ScenarioSchemaVersion = Annotated[
    Literal[1],
    BeforeValidator(_require_schema_version_one),
]


def _require_schema_version_two(value: object) -> object:
    """Return exact Python integer 2; reject bool and other values with ValueError."""
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
    """Require an exact allowed record class and stable structural revalidation.

    Raise ValueError for undeclared subtypes, unchecked copies, or invalid fields.
    Return nothing and leave the supplied record unchanged.
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


class ScenarioParameterV1(EvaluationModel):
    """Store one explicitly typed scenario parameter.

    Attributes
    ----------
    name : _AsciiIdentifier
        Nonempty ASCII identifier; specifications require sorted unique names.
    value : bool | int | _FiniteFloat | _AsciiText
        Exact bool, int, finite float, or nonempty printable ASCII string.
    """

    name: _AsciiIdentifier
    value: bool | int | _FiniteFloat | _AsciiText

    @model_validator(mode="after")
    def _validate_value(self) -> ScenarioParameterV1:
        """Require an exact bool, int, float, or str parameter value.

        Return this parameter or raise ValueError; typed fields also forbid nonfinite
        floats.
        """
        if type(self.value) not in (bool, int, float, str):
            raise ValueError("scenario parameter must use one exact JSON scalar type")
        return self


class ScenarioMeasurementDefinitionV1(EvaluationModel):
    """Declare one measurement before collecting scenario results.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.scenario_measurement_definition']
        Fixed scenario-measurement-definition identifier.
    schema_version : _ScenarioSchemaVersion
        Exact integer 1; the default.
    measurement_id : _AsciiIdentifier
        Stable nonempty ASCII definition identifier.
    measurement_version : _PositiveInt
        Positive definition version.
    role : ScenarioMeasurementRole
        primary or secondary.
    value_type : ScenarioValueType
        boolean, count, or scalar.
    units : _AsciiIdentifier
        Nonempty identifier naming measurement units.
    completion_scope : ScenarioCompletionScope
        any_gap_free_prefix or complete_episode.
    supports_right_censoring : bool
        Whether this endpoint has defined right-censoring semantics.

    Notes
    -----
    This is a contract, not an implementation of the measurement.
    """

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
    """Declare one violation before collecting scenario results.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.scenario_violation_definition']
        Fixed scenario-violation-definition identifier.
    schema_version : _ScenarioSchemaVersion
        Exact integer 1; the default.
    violation_id : _AsciiIdentifier
        Stable nonempty ASCII definition identifier.
    violation_version : _PositiveInt
        Positive definition version.
    value_type : ScenarioValueType
        boolean, count, or scalar.
    units : _AsciiIdentifier
        Nonempty identifier naming measurement units.
    completion_scope : ScenarioCompletionScope
        any_gap_free_prefix or complete_episode.
    supports_right_censoring : bool
        Whether this endpoint has defined right-censoring semantics.

    Notes
    -----
    This is a contract, not an implementation of the measurement.
    """

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
    """Store a defined boolean scenario value.

    Attributes
    ----------
    value_type : Literal['boolean']
        Fixed "boolean" discriminator; the default.
    value : bool
        Exact bool outcome.

    Notes
    -----
    Units and meaning come from the owning definition.
    """

    value_type: Literal["boolean"] = "boolean"
    value: bool


class ScenarioCountValueV1(EvaluationModel):
    """Store a defined count scenario value.

    Attributes
    ----------
    value_type : Literal['count']
        Fixed "count" discriminator; the default.
    value : _NonNegativeInt
        Nonnegative integer count.

    Notes
    -----
    Units and meaning come from the owning definition.
    """

    value_type: Literal["count"] = "count"
    value: _NonNegativeInt


class ScenarioScalarValueV1(EvaluationModel):
    """Store a defined scalar scenario value.

    Attributes
    ----------
    value_type : Literal['scalar']
        Fixed "scalar" discriminator; the default.
    value : _FiniteFloat
        Finite floating value; negative values are allowed.

    Notes
    -----
    Units and meaning come from the owning definition.
    """

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
    """Require typed values for defined results and reasons for missing results.

    Defined endpoints cannot be unavailable and must omit reason. Undefined results
    must omit value and include a reason; unavailable status also requires an
    unavailable endpoint. Raise ValueError on contradictions or invalid nested values.
    """
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
    """Store one caller-computed scenario measurement result.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.scenario_measurement_result']
        Fixed scenario-measurement-result identifier.
    schema_version : _ScenarioSchemaVersion
        Exact integer 1; the default.
    measurement_id : _AsciiIdentifier
        Identifier of the matching definition.
    measurement_version : _PositiveInt
        Positive version of the matching definition.
    result_status : ScenarioResultStatus
        defined, insufficient_data, or unavailable.
    endpoint_observation_status : ScenarioEndpointObservationStatus
        not_applicable, observed, right_censored, competing_event, or unavailable.
    value : ScenarioResultValueV1 | None
        Typed result when defined, otherwise None. Required argument even when None.
    reason : _AsciiText | None
        Printable explanation for undefined results; None for defined results, the
        default.

    Notes
    -----
    The enclosing record checks definition, value kind, and completion eligibility. This
    model does not calculate the value.
    """

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
        """Check value/reason/endpoint consistency through the shared payload validator.

        Return this result or raise ValueError; declaration and replay joins happen
        later.
        """
        _validate_result_payload(
            result_status=self.result_status,
            endpoint_observation_status=self.endpoint_observation_status,
            value=self.value,
            reason=self.reason,
        )
        return self


class ScenarioViolationResultV1(EvaluationModel):
    """Store one caller-computed scenario violation result.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.scenario_violation_result']
        Fixed scenario-violation-result identifier.
    schema_version : _ScenarioSchemaVersion
        Exact integer 1; the default.
    violation_id : _AsciiIdentifier
        Identifier of the matching definition.
    violation_version : _PositiveInt
        Positive version of the matching definition.
    result_status : ScenarioResultStatus
        defined, insufficient_data, or unavailable.
    endpoint_observation_status : ScenarioEndpointObservationStatus
        not_applicable, observed, right_censored, competing_event, or unavailable.
    value : ScenarioResultValueV1 | None
        Typed result when defined, otherwise None. Required argument even when None.
    reason : _AsciiText | None
        Printable explanation for undefined results; None for defined results, the
        default.

    Notes
    -----
    The enclosing record checks definition, value kind, and completion eligibility. This
    model does not calculate the value.
    """

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
        """Check value/reason/endpoint consistency through the shared payload validator.

        Return this result or raise ValueError; declaration and replay joins happen
        later.
        """
        _validate_result_payload(
            result_status=self.result_status,
            endpoint_observation_status=self.endpoint_observation_status,
            value=self.value,
            reason=self.reason,
        )
        return self


class ScenarioPredicateResultV1(EvaluationModel):
    """Store the supplied result of a named scenario success rule.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.scenario_predicate_result']
        Fixed scenario-predicate-result identifier.
    schema_version : _ScenarioSchemaVersion
        Exact integer 1; the default.
    predicate_id : _AsciiIdentifier
        Identifier of the declared success predicate.
    predicate_version : _PositiveInt
        Positive predicate version.
    status : ScenarioPredicateStatus
        satisfied, not_satisfied, or unavailable.
    reason : _AsciiText | None
        Required only when unavailable; defaults to None.

    Notes
    -----
    This model records a conclusion; it does not run the success rule.
    """

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
        """Require a reason only when predicate status is unavailable.

        Return this result or raise ValueError; do not evaluate the predicate itself.
        """
        if self.status == "unavailable":
            if self.reason is None:
                raise ValueError("unavailable predicate results require a reason")
        elif self.reason is not None:
            raise ValueError("available predicate results forbid a reason")
        return self


class ResolvedScenarioSpecificationV1(EvaluationModel):
    """Freeze a historical role-based scenario definition before rollout.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.resolved_scenario_specification']
        Fixed resolved-scenario-specification identifier.
    canonical_digest_sha256 : _Sha256Hex
        SHA-256 of all other specification fields.
    scenario_id : _AsciiIdentifier
        Stable scenario identifier.
    scenario_version : _PositiveInt
        Positive authored scenario version.
    classification : ScenarioClassification
        official or custom label.
    hypothesis : _AsciiText
        Nonempty printable description of the scientific hypothesis.
    authored_initial_condition : ContentAddressedIdentityV1
        Content identity of authored initial-condition data.
    parameters : tuple[ScenarioParameterV1, ...]
        Sorted unique typed parameter tuple; defaults to empty.
    resolved_config_digest_sha256 : _Sha256Hex
        SHA-256 of the resolved runtime config.
    horizon : _PositiveInt
        Positive expected artifact transition count.
    pressure_protocol : ContentAddressedIdentityV1 | None
        Optional content identity of opposing pressure rules; defaults to None.
    primary_measurement : ScenarioMeasurementDefinitionV1
        The one primary endpoint definition.
    secondary_measurements : Annotated[tuple[ScenarioMeasurementDefinitionV1, ...],
    Field(max_length=2)]
        At most two sorted secondary definitions; defaults to empty.
    violations : tuple[ScenarioViolationDefinitionV1, ...]
        Sorted unique violation definitions; defaults to empty.
    success_predicate : VersionedIdentityV1
        Identifier/version of the declared success rule.
    completion_policy : VersionedIdentityV1
        Identifier/version of the completion contract.
    partial_result_policy : VersionedIdentityV1
        Identifier/version of the incomplete-result contract.
    schema_version : _ScenarioSchemaVersion
        Exact integer 1; the default.
    eligible_roles : Annotated[tuple[ScenarioEvaluationRole, ...], Field(min_length=1,
    max_length=3)]
        One to three unique roles ordered focal, cooperative_partner,
        adversarial_opponent.

    Notes
    -----
    V1 does not bind a fixed roster, seed coordinate, or identity-independent initial
    state.
    """

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
        """Check exact nested declarations, sorted unique roles/parameters/endpoints,
        and digest.

        Require one primary and at most two secondary definitions with their declared
        roles. Return this historical specification or raise ValueError.
        """
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
    """Return the measurement identifier/version pair used to join definition rows.

    Preserve declaration order; this helper does not sort or evaluate values.
    """
    return definition.measurement_id, definition.measurement_version


def _measurement_result_key(
    result: ScenarioMeasurementResultV1,
) -> tuple[str, int]:
    """Return the measurement identifier/version pair used to join result rows.

    Preserve declaration order; this helper does not sort or evaluate values.
    """
    return result.measurement_id, result.measurement_version


def _violation_definition_key(
    definition: ScenarioViolationDefinitionV1,
) -> tuple[str, int]:
    """Return the violation identifier/version pair used to join definition rows.

    Preserve declaration order; this helper does not sort or evaluate values.
    """
    return definition.violation_id, definition.violation_version


def _violation_result_key(
    result: ScenarioViolationResultV1,
) -> tuple[str, int]:
    """Return the violation identifier/version pair used to join result rows.

    Preserve declaration order; this helper does not sort or evaluate values.
    """
    return result.violation_id, result.violation_version


class ScenarioEvaluationRecordV1(EvaluationModel):
    """Join historical role-based scenario results to replay and metric evidence.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.scenario_evaluation_record']
        Fixed scenario-evaluation-record identifier.
    record_id : _AsciiIdentifier
        Canonical replay episode ID followed by :scenario-evaluation.
    canonical_digest_sha256 : _Sha256Hex
        SHA-256 of all other record fields.
    realized_initial_frame_digest_sha256 : _Sha256Hex
        Digest of the complete recorded frame zero, including recording identity.
    measurement_results : tuple[ScenarioMeasurementResultV1, ...]
        Results in exact primary-then-secondary definition order.
    violation_results : tuple[ScenarioViolationResultV1, ...]
        Results in exact violation-definition order.
    predicate_result : ScenarioPredicateResultV1
        Supplied result matching the declared success-predicate identity.
    schema_version : _ScenarioSchemaVersion
        Exact integer 1; the default.
    specification : ResolvedScenarioSpecificationV1
        Exact role-based specification V1.
    replay_reference : ReplayArtifactReferenceV1
        V1 replay content reference.
    metric_report_reference : MetricReportReferenceV1
        V1 sidecar reference for the same episode and trajectory.

    Notes
    -----
    Construction checks local references and result definitions. Use
    validate_scenario_evaluation_record_v1 with actual evidence to check joins.
    """

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
        """Check exact nested references, result-definition order/types, predicate
        identity, and digest.

        Require replay/report episode and trajectory joins. Full evidence and completion
        checks need the actual replay and sidecar, not references alone.
        """
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
    """Check legacy scenario identity, config, horizon, roles, frame-zero digest, and
    endpoints.

    Caller has already validated replay and sidecar. Require scenario_metric_complete
    capture and exact content references. Incomplete trajectories cannot supply
    complete-episode values; censored/competing endpoints need their completion
    evidence. Raise ValueError without computing any supplied measurement.
    """
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
    """Validate scenario record V1 against its actual recorded evidence.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV1
        Exact ScenarioEvaluationRecordV1 with supplied endpoint results.
    replay : ReplayArtifactV1
        Exact ReplayArtifactV1 carrying the referenced trajectory.
    metric_report_artifact : EvaluationMetricReportArtifactV1
        Exact V1 sidecar referenced by the replay.

    Returns
    -------
    None
        None.

    Raises
    ------
    TypeError
        A required root type is unsupported.
    ValueError
        Record validity, content references, scenario/config/role/state
        joins, or endpoint completion eligibility fails.

    Notes
    -----
    Host-only validation does not compute supplied measurement values or
    predicate truth. Includes V1 replay and sidecar validation.
    Current official product admission requires the separate official validator.
    """
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
    """Build and validate a content-addressed scenario record V1.

    Parameters
    ----------
    specification : ResolvedScenarioSpecificationV1
        Exact ResolvedScenarioSpecificationV1 declared before rollout.
    replay : ReplayArtifactV1
        Exact ReplayArtifactV1 containing the realized evidence.
    metric_report_artifact : EvaluationMetricReportArtifactV1
        Matching V1 metric-report sidecar.
    measurement_results : tuple[ScenarioMeasurementResultV1, ...]
        Caller-computed results in primary-then-secondary definition order.
    violation_results : tuple[ScenarioViolationResultV1, ...]
        Caller-computed results in exact violation-definition order.
    predicate_result : ScenarioPredicateResultV1
        Caller-computed outcome of the declared success predicate.

    Returns
    -------
    ScenarioEvaluationRecordV1
        Strict ScenarioEvaluationRecordV1 with canonical ID/digest, content references,
        and complete initial-frame digest.

    Raises
    ------
    TypeError
        Required evidence types are unsupported.
    ValueError
        Evidence, declaration/result joins, completion eligibility fails.

    Notes
    -----
    This historical builder checks evidence joins without live official context
    admission.
    Performs no file I/O and does not compute measurements, predicate truth, or
    a simulator rollout. All supplied endpoint values remain the caller's
    responsibility.
    """
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


def resolved_initial_state_digest_sha256(
    simulator_step_count: int,
    snapshot: GlobalAnalysisSnapshotV1,
) -> str:
    """Hash simulator tick and privileged state independently of recording identity.

    Parameters
    ----------
    simulator_step_count : int
        Simulator tick belonging to the supplied snapshot.
    snapshot : GlobalAnalysisSnapshotV1
        Validated GlobalAnalysisSnapshotV1.

    Returns
    -------
    str
        SHA-256 of the fixed V2 initial-state domain, tick, and snapshot.

    Raises
    ------
    TypeError
        A supplied value cannot be serialized canonically.
    ValueError
        A supplied value is nonfinite or circular.

    Notes
    -----
    Does not check that the tick is nonnegative or a frame-zero coordinate.
    Callers own admission. Episode IDs, observations, and action masks are excluded.
    """
    return canonical_digest_sha256(
        {
            "digest_domain": _RESOLVED_INITIAL_STATE_DIGEST_DOMAIN_V2,
            "simulator_step_count": simulator_step_count,
            "snapshot": snapshot,
        }
    )


def _resolved_initial_state_digest_from_validated_frame_v2(
    frame: EvaluationFrame,
) -> str:
    """Hash a validated frame's simulator tick and snapshot without recording
    identities.

    Caller owns validation and deciding whether this frame is the initial condition.
    """
    return resolved_initial_state_digest_sha256(
        frame.simulator_step_count, frame.snapshot
    )


def resolved_initial_state_digest_sha256_v2(frame: EvaluationFrame) -> str:
    """Validate a frame and hash its simulator tick and privileged snapshot.

    Parameters
    ----------
    frame : EvaluationFrame
        Exact supported frame V1 or V2.

    Returns
    -------
    str
        Identity-independent V2 state digest; actor observations and episode IDs are
        excluded.

    Raises
    ------
    TypeError
        The frame root type is unsupported.
    ValueError
        Strict frame revalidation fails.

    Notes
    -----
    Intended for scenario initial conditions, but the helper does not require
    frame_index zero. The enclosing scenario/replay join establishes that fact.
    """
    canonical_frame = cast(
        EvaluationFrame,
        validate_declared_model_tree(
            frame,
            record_name="scenario resolved initial-state frame",
            expected_type=evaluation_frame_type(frame),
        ),
    )
    return _resolved_initial_state_digest_from_validated_frame_v2(canonical_frame)


def _require_schema_version_three(value: object) -> object:
    """Return exact Python integer 3; reject bool and other values with ValueError."""
    if type(value) is not int or value != 3:
        raise ValueError("schema_version must be the exact integer 3")
    return value


_ScenarioSchemaVersionV3 = Annotated[
    Literal[3], BeforeValidator(_require_schema_version_three)
]


class _ScenarioSeedScheduleFields(EvaluationModel):
    """Share stable identity fields for an ordered matched-seed schedule.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.scenario_seed_schedule']
        Fixed scenario-seed-schedule identifier.
    schedule_id : _AsciiIdentifier
        Stable schedule identifier.
    schedule_version : _PositiveInt
        Positive schedule-content version.
    canonical_digest_sha256 : _Sha256Hex
        SHA-256 over all other fields including ordered seed rows.

    Notes
    -----
    Concrete models supply nonempty realized_seed_protocols and their exact schema
    version.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.scenario_seed_schedule"] = (
        SCENARIO_SEED_SCHEDULE_SCHEMA_ID
    )
    schedule_id: _AsciiIdentifier
    schedule_version: _PositiveInt
    canonical_digest_sha256: _Sha256Hex

    def _check_schedule(
        self,
        rows: tuple[EvaluationSeedProtocolV1, ...]
        | tuple[EvaluationSeedProtocolV2, ...],
        expected_type: type[EvaluationSeedProtocolV1] | type[EvaluationSeedProtocolV2],
    ) -> None:
        """Check nonempty exact seed rows, one protocol identity, unique rows, and
        digest.

        The concrete schedule model ensures nonempty input. Preserve declared row order;
        it is the stable coordinate used by scenario records. Raise ValueError on
        mismatch.
        """
        for seed_protocol in rows:
            _require_stable_nested_model(
                seed_protocol,
                record_name="scenario schedule seed protocol",
                expected_types=(expected_type,),
            )

        protocol_identity = rows[0].seed_protocol
        if any(row.seed_protocol != protocol_identity for row in rows[1:]):
            raise ValueError(
                "scenario schedule rows must share one seed-protocol identity"
            )
        row_digests = tuple(canonical_digest_sha256(row) for row in rows)
        if len(row_digests) != len(set(row_digests)):
            raise ValueError("scenario schedule rows must be unique")

        expected_digest = canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        )
        if self.canonical_digest_sha256 != expected_digest:
            raise ValueError("scenario seed schedule digest mismatch")


class ScenarioSeedScheduleV2(_ScenarioSeedScheduleFields):
    """Historical complete seed provenance in a stable ordered schedule.

    Attributes
    ----------
    schema_version : _ScenarioSchemaVersionV2
        Exact integer 2; the default.
    realized_seed_protocols : Annotated[tuple[EvaluationSeedProtocolV1, ...],
    Field(min_length=1)]
        Nonempty tuple of exact EvaluationSeedProtocolV1 rows with one shared protocol
        identity and unique content.

    Notes
    -----
    Inherits schedule identity/digest fields. Row order is significant; no RNG stream is
    generated here.
    """

    schema_version: _ScenarioSchemaVersionV2 = SCENARIO_SCHEMA_VERSION_V2
    realized_seed_protocols: Annotated[
        tuple[EvaluationSeedProtocolV1, ...], Field(min_length=1)
    ]

    @model_validator(mode="after")
    def _validate_schedule(self) -> ScenarioSeedScheduleV2:
        """Validate the shared schedule using exact V1 seed records.

        Return this schedule or raise ValueError; no rows are reordered.
        """
        self._check_schedule(self.realized_seed_protocols, EvaluationSeedProtocolV1)
        return self


class ScenarioSeedScheduleV3(_ScenarioSeedScheduleFields):
    """Current optional seed provenance in a stable ordered schedule.

    Attributes
    ----------
    schema_version : _ScenarioSchemaVersionV3
        Exact integer 3; the default.
    realized_seed_protocols : Annotated[tuple[EvaluationSeedProtocolV2, ...],
    Field(min_length=1)]
        Nonempty tuple of exact EvaluationSeedProtocolV2 rows with one shared protocol
        identity and unique content.

    Notes
    -----
    Inherits schedule identity/digest fields. Row order is significant; no RNG stream is
    generated here.
    """

    schema_version: _ScenarioSchemaVersionV3 = 3
    realized_seed_protocols: Annotated[
        tuple[EvaluationSeedProtocolV2, ...], Field(min_length=1)
    ]

    @model_validator(mode="after")
    def _validate_schedule(self) -> ScenarioSeedScheduleV3:
        """Validate the shared schedule using exact V2 seed records.

        Return this schedule or raise ValueError; no rows are reordered.
        """
        self._check_schedule(self.realized_seed_protocols, EvaluationSeedProtocolV2)
        return self


class _ResolvedScenarioSpecificationFields(EvaluationModel):
    """Share fixed-slot scenario fields across specification versions.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.resolved_scenario_specification']
        Fixed resolved-scenario-specification identifier.
    canonical_digest_sha256 : _Sha256Hex
        SHA-256 of all other specification fields.
    scenario_id : _AsciiIdentifier
        Stable scenario identifier.
    scenario_version : _PositiveInt
        Positive authored scenario version.
    classification : ScenarioClassification
        official or custom label.
    hypothesis : _AsciiText
        Nonempty printable description of the scientific hypothesis.
    authored_initial_condition : ContentAddressedIdentityV1
        Content identity of authored initial-condition data.
    parameters : tuple[ScenarioParameterV1, ...]
        Sorted unique typed parameter tuple; defaults to empty.
    resolved_config_digest_sha256 : _Sha256Hex
        SHA-256 of the resolved runtime config.
    horizon : _PositiveInt
        Positive expected artifact transition count.
    pressure_protocol : ContentAddressedIdentityV1 | None
        Optional content identity of opposing pressure rules; defaults to None.
    primary_measurement : ScenarioMeasurementDefinitionV1
        The one primary endpoint definition.
    secondary_measurements : Annotated[tuple[ScenarioMeasurementDefinitionV1, ...],
    Field(max_length=2)]
        At most two sorted secondary definitions; defaults to empty.
    violations : tuple[ScenarioViolationDefinitionV1, ...]
        Sorted unique violation definitions; defaults to empty.
    success_predicate : VersionedIdentityV1
        Identifier/version of the declared success rule.
    completion_policy : VersionedIdentityV1
        Identifier/version of the completion contract.
    partial_result_policy : VersionedIdentityV1
        Identifier/version of the incomplete-result contract.
    layout : ContentAddressedIdentityV1
        Exact content identity of layout.
    resolved_initial_state_digest_sha256 : _Sha256Hex
        Digest of simulator tick and privileged snapshot, excluding recording identity.
    roster_template : Annotated[tuple[RosterSlotV1, ...],
    Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS)]
        Ten global roster rows, Team A 0-4 then Team B 5-9.
    role_template : Annotated[tuple[ScenarioFixedSlotRoleV2, ...],
    Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS)]
        Ten aligned focal/partner/opponent roles or not_applicable for inactive slots.

    Notes
    -----
    Concrete V2/V3 specifications add their exact seed_schedule version. Templates
    freeze scientific conditions independently of the policy implementation.
    """

    schema_id: Literal[
        "marl_battlegrounds.evaluation.resolved_scenario_specification"
    ] = SCENARIO_SPECIFICATION_SCHEMA_ID
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

    def _check_specification(
        self,
        seed_schedule: ScenarioSeedScheduleV2 | ScenarioSeedScheduleV3,
        schedule_type: type[ScenarioSeedScheduleV2] | type[ScenarioSeedScheduleV3],
    ) -> None:
        """Check fixed-slot scenario declarations, exact schedule, role joins, and
        content digest.

        Require ten ordered roster/role rows, neutral inactive slots, a focal role,
        sorted unique parameters/endpoints, and role-compatible schedule fields.
        None seed facts remain permitted for present roles in V3; absent optional roles
        must be marked not_applicable. Raise ValueError without running a scenario.
        """
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
            seed_schedule,
            record_name="scenario seed schedule",
            expected_types=(schedule_type,),
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
        for seed_row in seed_schedule.realized_seed_protocols:
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


class ResolvedScenarioSpecificationV2(_ResolvedScenarioSpecificationFields):
    """Historical fixed-slot scenario contract with matched seed coordinates.

    Attributes
    ----------
    schema_version : _ScenarioSchemaVersionV2
        Exact integer 2; the default.
    seed_schedule : ScenarioSeedScheduleV2
        Exact ScenarioSeedScheduleV2 with all declared coordinates.

    Notes
    -----
    Inherits the complete fields of _ResolvedScenarioSpecificationFields: layout,
    state/config digests, roster/role templates, hypothesis, endpoints, policies, and
    optional pressure identity.
    """

    schema_version: _ScenarioSchemaVersionV2 = SCENARIO_SCHEMA_VERSION_V2
    seed_schedule: ScenarioSeedScheduleV2

    @model_validator(mode="after")
    def _validate_specification(self) -> ResolvedScenarioSpecificationV2:
        """Check shared fixed-slot declarations using exact schedule V2.

        Return this specification or raise ValueError on any identity, role, seed, or
        digest mismatch.
        """
        self._check_specification(self.seed_schedule, ScenarioSeedScheduleV2)
        return self


class ResolvedScenarioSpecificationV3(_ResolvedScenarioSpecificationFields):
    """Current fixed-slot scenario contract with matched seed coordinates.

    Attributes
    ----------
    schema_version : _ScenarioSchemaVersionV3
        Exact integer 3; the default.
    seed_schedule : ScenarioSeedScheduleV3
        Exact ScenarioSeedScheduleV3 with all declared coordinates.

    Notes
    -----
    Inherits the complete fields of _ResolvedScenarioSpecificationFields: layout,
    state/config digests, roster/role templates, hypothesis, endpoints, policies, and
    optional pressure identity.
    """

    schema_version: _ScenarioSchemaVersionV3 = 3
    seed_schedule: ScenarioSeedScheduleV3

    @model_validator(mode="after")
    def _validate_specification(self) -> ResolvedScenarioSpecificationV3:
        """Check shared fixed-slot declarations using exact schedule V3.

        Return this specification or raise ValueError on any identity, role, seed, or
        digest mismatch.
        """
        self._check_specification(self.seed_schedule, ScenarioSeedScheduleV3)
        return self


class _ScenarioEvaluationRecordFields(EvaluationModel):
    """Share result fields for fixed-slot, matched-seed scenario records.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.scenario_evaluation_record']
        Fixed scenario-evaluation-record identifier.
    record_id : _AsciiIdentifier
        Canonical replay episode ID followed by :scenario-evaluation.
    canonical_digest_sha256 : _Sha256Hex
        SHA-256 of all other record fields.
    realized_initial_frame_digest_sha256 : _Sha256Hex
        Digest of the complete recorded frame zero, including recording identity.
    measurement_results : tuple[ScenarioMeasurementResultV1, ...]
        Results in exact primary-then-secondary definition order.
    violation_results : tuple[ScenarioViolationResultV1, ...]
        Results in exact violation-definition order.
    predicate_result : ScenarioPredicateResultV1
        Supplied result matching the declared success-predicate identity.
    schedule_coordinate : _NonNegativeInt
        Zero-based coordinate into specification.seed_schedule.realized_seed_protocols.

    Notes
    -----
    Concrete records add exact specification/replay reference versions and, for V2, a
    metric sidecar.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.scenario_evaluation_record"] = (
        SCENARIO_EVALUATION_RECORD_SCHEMA_ID
    )
    record_id: _AsciiIdentifier
    canonical_digest_sha256: _Sha256Hex
    schedule_coordinate: _NonNegativeInt
    realized_initial_frame_digest_sha256: _Sha256Hex
    measurement_results: tuple[ScenarioMeasurementResultV1, ...]
    violation_results: tuple[ScenarioViolationResultV1, ...]
    predicate_result: ScenarioPredicateResultV1

    def _check_record(
        self,
        specification: ResolvedScenarioSpecificationV2
        | ResolvedScenarioSpecificationV3,
        replay_reference: ReplayArtifactReferenceV1
        | ReplayArtifactReferenceV2
        | ReplayArtifactReferenceV3
        | ReplayArtifactReferenceV4,
    ) -> None:
        """Check schedule bounds, exact endpoint order/types, predicate identity, and
        digest.

        Caller validates concrete specification/reference versions. This shared check
        uses the replay reference's episode ID but does not load replay content.
        """
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

        episode_id = replay_reference.episode_id
        if self.record_id != f"{episode_id}:scenario-evaluation":
            raise ValueError("scenario evaluation record ID is not canonical")
        if self.schedule_coordinate >= len(
            specification.seed_schedule.realized_seed_protocols
        ):
            raise ValueError("scenario schedule coordinate is out of range")

        definitions = (
            specification.primary_measurement,
            *specification.secondary_measurements,
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
            map(_violation_definition_key, specification.violations)
        )
        violation_result_keys = tuple(
            map(_violation_result_key, self.violation_results)
        )
        if violation_result_keys != violation_definition_keys:
            raise ValueError(
                "scenario violation results must exactly follow their definitions"
            )
        for definition, result in zip(
            specification.violations,
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
            != specification.success_predicate.identifier
            or self.predicate_result.predicate_version
            != specification.success_predicate.version
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


class ScenarioEvaluationRecordV2(_ScenarioEvaluationRecordFields):
    """Historical scenario endpoint evidence joined to replay V1.

    Attributes
    ----------
    schema_version : _ScenarioSchemaVersionV2
        Exact integer 2; the default.
    specification : ResolvedScenarioSpecificationV2
        Exact fixed-slot specification V2.
    replay_reference : ReplayArtifactReferenceV1
        Exact replay reference V1.
    metric_report_reference : MetricReportReferenceV1
        V1 sidecar reference sharing episode and trajectory.

    Notes
    -----
    Inherits record identity, digest, schedule coordinate, initial-frame digest, and
    endpoint results from _ScenarioEvaluationRecordFields. Actual evidence joins require
    the public validator.
    """

    schema_version: _ScenarioSchemaVersionV2 = SCENARIO_SCHEMA_VERSION_V2
    specification: ResolvedScenarioSpecificationV2
    replay_reference: ReplayArtifactReferenceV1
    metric_report_reference: MetricReportReferenceV1

    @model_validator(mode="after")
    def _validate_record(self) -> ScenarioEvaluationRecordV2:
        """Check exact specification/reference versions and shared result/digest
        constraints.

        Also require the report reference to share replay episode and trajectory. Return
        this record or raise ValueError; no files are loaded.
        """
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
        episode_id = self.replay_reference.episode_id
        if self.metric_report_reference.episode_id != episode_id:
            raise ValueError("scenario replay and metric report episodes must match")
        if (
            self.metric_report_reference.trajectory_content_digest_sha256
            != self.replay_reference.trajectory_content_digest_sha256
        ):
            raise ValueError(
                "scenario replay and metric report must join the same trajectory"
            )
        self._check_record(self.specification, self.replay_reference)
        return self


class ScenarioEvaluationRecordV3(_ScenarioEvaluationRecordFields):
    """Historical scenario endpoint evidence joined to replay V2.

    Attributes
    ----------
    schema_version : _ScenarioSchemaVersionV3
        Exact integer 3; the default.
    specification : ResolvedScenarioSpecificationV3
        Exact fixed-slot specification V3.
    replay_reference : ReplayArtifactReferenceV2
        Exact replay reference V2.

    Notes
    -----
    Inherits record identity, digest, schedule coordinate, initial-frame digest, and
    endpoint results from _ScenarioEvaluationRecordFields. Actual evidence joins require
    the public validator.
    """

    schema_version: _ScenarioSchemaVersionV3 = 3
    specification: ResolvedScenarioSpecificationV3
    replay_reference: ReplayArtifactReferenceV2

    @model_validator(mode="after")
    def _validate_record(self) -> ScenarioEvaluationRecordV3:
        """Check exact specification/reference versions and shared result/digest
        constraints.

        Return this record or raise ValueError; no files are loaded.
        """
        _require_stable_nested_model(
            self.specification,
            record_name="scenario specification",
            expected_types=(ResolvedScenarioSpecificationV3,),
        )
        _require_stable_nested_model(
            self.replay_reference,
            record_name="scenario replay reference",
            expected_types=(ReplayArtifactReferenceV2,),
        )
        self._check_record(self.specification, self.replay_reference)
        return self


def _require_schema_version_four(value: object) -> object:
    """Return exact Python integer 4; reject bool and other values with ValueError."""
    if type(value) is not int or value != 4:
        raise ValueError("schema_version must be the exact integer 4")
    return value


class ScenarioEvaluationRecordV4(_ScenarioEvaluationRecordFields):
    """Current scenario endpoint evidence joined to replay V3.

    Attributes
    ----------
    schema_version : Annotated[Literal[4],
    BeforeValidator(_require_schema_version_four)]
        Exact integer 4; the default.
    specification : ResolvedScenarioSpecificationV3
        Exact fixed-slot specification V3.
    replay_reference : ReplayArtifactReferenceV3
        Exact replay reference V3.

    Notes
    -----
    Inherits record identity, digest, schedule coordinate, initial-frame digest, and
    endpoint results from _ScenarioEvaluationRecordFields. Actual evidence joins require
    the public validator.
    """

    schema_version: Annotated[
        Literal[4], BeforeValidator(_require_schema_version_four)
    ] = 4
    specification: ResolvedScenarioSpecificationV3
    replay_reference: ReplayArtifactReferenceV3

    @model_validator(mode="after")
    def _validate_record(self) -> ScenarioEvaluationRecordV4:
        """Check exact specification/reference versions and shared result/digest
        constraints.

        Return this record or raise ValueError; no files are loaded.
        """
        _require_stable_nested_model(
            self.specification,
            record_name="scenario specification",
            expected_types=(ResolvedScenarioSpecificationV3,),
        )
        _require_stable_nested_model(
            self.replay_reference,
            record_name="scenario replay reference",
            expected_types=(ReplayArtifactReferenceV3,),
        )
        self._check_record(self.specification, self.replay_reference)
        return self


def _require_schema_version_five(value: object) -> object:
    """Return exact Python integer 5; reject bool and other values with ValueError."""
    if type(value) is not int or value != 5:
        raise ValueError("schema_version must be the exact integer 5")
    return value


class ScenarioEvaluationRecordV5(_ScenarioEvaluationRecordFields):
    """Current scenario endpoint evidence joined to replay V4.

    Attributes
    ----------
    schema_version : Annotated[Literal[5],
    BeforeValidator(_require_schema_version_five)]
        Exact integer 5; the default.
    specification : ResolvedScenarioSpecificationV3
        Exact fixed-slot specification V3 (its resolved_config_digest_sha256
        is the resolved config V2 digest, which covers the Red Zone depth).
    replay_reference : ReplayArtifactReferenceV4
        Exact replay reference V4.

    Notes
    -----
    Same result fields and checks as ScenarioEvaluationRecordV4; only the
    replay version differs. V1-V4 stay readable with their original meanings.
    Actual evidence joins require the public validator.
    """

    schema_version: Annotated[
        Literal[5], BeforeValidator(_require_schema_version_five)
    ] = 5
    specification: ResolvedScenarioSpecificationV3
    replay_reference: ReplayArtifactReferenceV4

    @model_validator(mode="after")
    def _validate_record(self) -> ScenarioEvaluationRecordV5:
        """Check exact specification/reference versions and shared result/digest
        constraints.

        Return this record or raise ValueError; no files are loaded.
        """
        _require_stable_nested_model(
            self.specification,
            record_name="scenario specification",
            expected_types=(ResolvedScenarioSpecificationV3,),
        )
        _require_stable_nested_model(
            self.replay_reference,
            record_name="scenario replay reference",
            expected_types=(ReplayArtifactReferenceV4,),
        )
        self._check_record(self.specification, self.replay_reference)
        return self


def _validate_scenario_evaluation_record_against_validated_replay_v2(
    canonical_record: ScenarioEvaluationRecordV2,
    replay: ReplayArtifactV1,
    expected_replay_reference: ReplayArtifactReferenceV1,
) -> None:
    """Check V2 replay/report references and scenario capture before shared scientific
    joins.

    Caller has already scanned the V1 replay and sidecar. Require
    scenario_metric_complete capture, then exact roster/role/seed/state matching.
    """
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
    _validate_scenario_replay_joins(canonical_record, replay, AssignedPolicySlotV1)


def _validate_scenario_replay_joins(
    canonical_record: ScenarioEvaluationRecordV2
    | ScenarioEvaluationRecordV3
    | ScenarioEvaluationRecordV4
    | ScenarioEvaluationRecordV5,
    replay: ReplayArtifactV1 | ReplayArtifactV2 | ReplayArtifactV3 | ReplayArtifactV4,
    assigned_type: type[AssignedPolicySlotV1] | type[AssignedPolicySlotV2],
) -> None:
    """Join scenario declaration to actual layout, config, roster, roles, seeds, and
    state.

    Require the schedule coordinate's exact seed record and both initial-state and
    whole-frame digests. Enforce completion restrictions for defined, censored, and
    competing-event endpoints. Raise ValueError on disagreement; supplied numeric
    results and predicate truth are not independently recomputed.
    """
    context = replay.header.context
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
            not isinstance(policy_assignment, assigned_type)
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


def _validate_official_scenario_replay_v2(
    replay: ReplayArtifactV1 | ReplayArtifactV2 | ReplayArtifactV3 | ReplayArtifactV4,
) -> None:
    """Check recorded context/state against current official rules for each replay
    version.

    Load catalog/Core admission only here. Require current mechanics, product config,
    valid initial state, the matching SharedObs projection, and complete same-team
    source availability at every frame, excluding self and inactive slots.
    Raise TypeError or ValueError on failure; do not run simulator transitions.
    """
    from marl_battlegrounds.evaluation.catalog import (
        _validate_official_scenario_context_v2,  # pyright: ignore[reportPrivateUsage]
        _validate_official_scenario_context_v3,  # pyright: ignore[reportPrivateUsage]
        _validate_official_scenario_context_v4,  # pyright: ignore[reportPrivateUsage]
        _validate_official_scenario_context_v5,  # pyright: ignore[reportPrivateUsage]
    )

    context = replay.header.context
    # V4 subclasses V2 (not V3), so it must be matched first.
    if isinstance(replay, ReplayArtifactV4):
        _validate_official_scenario_context_v5(replay.header.context, replay.frames[0])
    elif isinstance(replay, ReplayArtifactV3):
        _validate_official_scenario_context_v4(replay.header.context, replay.frames[0])
    elif isinstance(replay, ReplayArtifactV2):
        _validate_official_scenario_context_v3(replay.header.context, replay.frames[0])
    else:
        _validate_official_scenario_context_v2(replay.header.context, replay.frames[0])

    expected_availability = tuple(
        tuple(
            recipient_slot != sensor_source_slot
            and recipient.configured_active
            and sensor_source.configured_active
            and recipient.configured_team_id == sensor_source.configured_team_id
            for sensor_source_slot, sensor_source in enumerate(context.roster)
        )
        for recipient_slot, recipient in enumerate(context.roster)
    )
    for frame in replay.frames:
        if (
            frame.shared_obs_information_availability_by_recipient_and_sensor_source
            != expected_availability
        ):
            raise ValueError(
                "official scenario evaluation requires canonical SharedObs "
                f"availability at frame_index {frame.frame_index}"
            )


def _validate_scenario_evaluation_record_and_evidence_v2(
    record: ScenarioEvaluationRecordV2,
    replay: ReplayArtifactV1,
    metric_report_artifact: EvaluationMetricReportArtifactV1,
    *,
    require_official_context: bool,
) -> None:
    """Validate V2 record, V1 replay, sidecar, and optional current official admission.

    require_official_context controls only the final live-rule check. Common identity
    and endpoint joins always apply; no measurement or predicate is computed.
    """
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
        _validate_official_scenario_replay_v2(replay)


def validate_scenario_evaluation_record_v2(
    record: ScenarioEvaluationRecordV2,
    replay: ReplayArtifactV1,
    metric_report_artifact: EvaluationMetricReportArtifactV1,
) -> None:
    """Validate scenario record V2 against its actual recorded evidence.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV2
        Exact ScenarioEvaluationRecordV2 with supplied endpoint results.
    replay : ReplayArtifactV1
        Exact ReplayArtifactV1 carrying the referenced trajectory.
    metric_report_artifact : EvaluationMetricReportArtifactV1
        Exact V1 sidecar referenced by the replay.

    Returns
    -------
    None
        None.

    Raises
    ------
    TypeError
        A required root type is unsupported.
    ValueError
        Record validity, content references, scenario/config/role/state
        joins, or endpoint completion eligibility fails.

    Notes
    -----
    Host-only validation does not compute supplied measurement values or
    predicate truth. Includes V1 replay and sidecar validation.
    Current official product admission requires the separate official validator.
    """
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
    """Validate scenario evidence and its context against current official rules.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV2
        Exact ScenarioEvaluationRecordV2 with supplied endpoint results.
    replay : ReplayArtifactV1
        Exact ReplayArtifactV1 carrying the referenced trajectory.
    metric_report_artifact : EvaluationMetricReportArtifactV1
        Exact V1 sidecar referenced by the replay.

    Returns
    -------
    None
        None.

    Raises
    ------
    TypeError
        A record or context type is unsupported.
    ValueError
        Evidence joins, current mechanics/product config, initial-state
        validity, projection, or canonical SharedObs availability fails.

    Notes
    -----
    Adds live catalog/Core admission to generic scenario validation. This can
    allocate JAX arrays while reconstructing recorded config/state; it does not
    rerun the episode or prove the supplied endpoint calculation.
    """
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
    """Build and validate a content-addressed scenario record V2.

    Parameters
    ----------
    specification : ResolvedScenarioSpecificationV2
        Exact ResolvedScenarioSpecificationV2 declared before rollout.
    replay : ReplayArtifactV1
        Exact ReplayArtifactV1 containing the realized evidence.
    metric_report_artifact : EvaluationMetricReportArtifactV1
        Matching V1 metric-report sidecar.
    schedule_coordinate : int
        Zero-based matched-seed coordinate within the specification.
    measurement_results : tuple[ScenarioMeasurementResultV1, ...]
        Caller-computed results in primary-then-secondary definition order.
    violation_results : tuple[ScenarioViolationResultV1, ...]
        Caller-computed results in exact violation-definition order.
    predicate_result : ScenarioPredicateResultV1
        Caller-computed outcome of the declared success predicate.

    Returns
    -------
    ScenarioEvaluationRecordV2
        Strict ScenarioEvaluationRecordV2 with canonical ID/digest, content references,
        and complete initial-frame digest.

    Raises
    ------
    TypeError
        Required evidence types are unsupported.
    ValueError
        Evidence, declaration/result joins, completion eligibility, or current official
        context admission fails.

    Notes
    -----
    Always applies official live-rule checks, including canonical SharedObs, even though
    specification classification is a recorded field.
    Performs no file I/O and does not compute measurements, predicate truth, or
    a simulator rollout. All supplied endpoint values remain the caller's
    responsibility.
    """
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
    _validate_official_scenario_replay_v2(replay)
    return record


def validate_scenario_evaluation_record_v3(
    record: ScenarioEvaluationRecordV3,
    replay: ReplayArtifactV2,
) -> None:
    """Validate scenario record V3 against its actual recorded evidence.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV3
        Exact ScenarioEvaluationRecordV3 with supplied endpoint results.
    replay : ReplayArtifactV2
        Exact ReplayArtifactV2 carrying the referenced trajectory.

    Returns
    -------
    None
        None.

    Raises
    ------
    TypeError
        A required root type is unsupported.
    ValueError
        Record validity, content references, scenario/config/role/state
        joins, or endpoint completion eligibility fails.

    Notes
    -----
    Host-only validation does not compute supplied measurement values or
    predicate truth. No legacy metric sidecar is needed.
    Current official product admission requires the separate official validator.
    """
    canonical_record = cast(
        ScenarioEvaluationRecordV3,
        validate_declared_model_tree(
            record,
            record_name="scenario evaluation record",
            expected_type=ScenarioEvaluationRecordV3,
        ),
    )
    validate_declared_model_tree(
        replay, record_name="scenario replay", expected_type=ReplayArtifactV2
    )
    if canonical_record.replay_reference != replay_reference_v2(replay):
        raise ValueError("scenario replay reference does not match replay content")
    _validate_scenario_replay_joins(canonical_record, replay, AssignedPolicySlotV2)


def validate_official_scenario_evaluation_record_v3(
    record: ScenarioEvaluationRecordV3,
    replay: ReplayArtifactV2,
) -> None:
    """Validate scenario evidence and its context against current official rules.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV3
        Exact ScenarioEvaluationRecordV3 with supplied endpoint results.
    replay : ReplayArtifactV2
        Exact ReplayArtifactV2 carrying the referenced trajectory.

    Returns
    -------
    None
        None.

    Raises
    ------
    TypeError
        A record or context type is unsupported.
    ValueError
        Evidence joins, current mechanics/product config, initial-state
        validity, projection, or canonical SharedObs availability fails.

    Notes
    -----
    Adds live catalog/Core admission to generic scenario validation. This can
    allocate JAX arrays while reconstructing recorded config/state; it does not
    rerun the episode or prove the supplied endpoint calculation.
    """
    validate_scenario_evaluation_record_v3(record, replay)
    _validate_official_scenario_replay_v2(replay)


def build_scenario_evaluation_record_v3(
    specification: ResolvedScenarioSpecificationV3,
    replay: ReplayArtifactV2,
    *,
    schedule_coordinate: int,
    measurement_results: tuple[ScenarioMeasurementResultV1, ...],
    violation_results: tuple[ScenarioViolationResultV1, ...],
    predicate_result: ScenarioPredicateResultV1,
) -> ScenarioEvaluationRecordV3:
    """Build and validate a content-addressed scenario record V3.

    Parameters
    ----------
    specification : ResolvedScenarioSpecificationV3
        Exact ResolvedScenarioSpecificationV3 declared before rollout.
    replay : ReplayArtifactV2
        Exact ReplayArtifactV2 containing the realized evidence.
    schedule_coordinate : int
        Zero-based matched-seed coordinate within the specification.
    measurement_results : tuple[ScenarioMeasurementResultV1, ...]
        Caller-computed results in primary-then-secondary definition order.
    violation_results : tuple[ScenarioViolationResultV1, ...]
        Caller-computed results in exact violation-definition order.
    predicate_result : ScenarioPredicateResultV1
        Caller-computed outcome of the declared success predicate.

    Returns
    -------
    ScenarioEvaluationRecordV3
        Strict ScenarioEvaluationRecordV3 with canonical ID/digest, content references,
        and complete initial-frame digest.

    Raises
    ------
    TypeError
        Required evidence types are unsupported.
    ValueError
        Evidence, declaration/result joins, completion eligibility, or current official
        context admission fails.

    Notes
    -----
    Always applies official live-rule checks, including canonical SharedObs, even though
    specification classification is a recorded field.
    Performs no file I/O and does not compute measurements, predicate truth, or
    a simulator rollout. All supplied endpoint values remain the caller's
    responsibility.
    """
    payload = {
        "schema_id": SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
        "schema_version": 3,
        "record_id": f"{replay.header.context.identity.episode_id}:scenario-evaluation",
        "specification": specification,
        "schedule_coordinate": schedule_coordinate,
        "replay_reference": replay_reference_v2(replay),
        "realized_initial_frame_digest_sha256": canonical_digest_sha256(
            replay.frames[0]
        ),
        "measurement_results": measurement_results,
        "violation_results": violation_results,
        "predicate_result": predicate_result,
    }
    record = ScenarioEvaluationRecordV3.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )
    validate_official_scenario_evaluation_record_v3(record, replay)
    return record


def validate_scenario_evaluation_record_v4(
    record: ScenarioEvaluationRecordV4,
    replay: ReplayArtifactV3,
) -> None:
    """Validate scenario record V4 against its actual recorded evidence.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV4
        Exact ScenarioEvaluationRecordV4 with supplied endpoint results.
    replay : ReplayArtifactV3
        Exact ReplayArtifactV3 carrying the referenced trajectory.

    Returns
    -------
    None
        None.

    Raises
    ------
    TypeError
        A required root type is unsupported.
    ValueError
        Record validity, content references, scenario/config/role/state
        joins, or endpoint completion eligibility fails.

    Notes
    -----
    Host-only validation does not compute supplied measurement values or
    predicate truth. No legacy metric sidecar is needed.
    Current official product admission requires the separate official validator.
    """
    canonical_record = cast(
        ScenarioEvaluationRecordV4,
        validate_declared_model_tree(
            record,
            record_name="scenario evaluation record",
            expected_type=ScenarioEvaluationRecordV4,
        ),
    )
    validate_declared_model_tree(
        replay, record_name="scenario replay", expected_type=ReplayArtifactV3
    )
    if canonical_record.replay_reference != replay_reference_v3(replay):
        raise ValueError("scenario replay reference does not match replay content")
    _validate_scenario_replay_joins(canonical_record, replay, AssignedPolicySlotV2)


def validate_official_scenario_evaluation_record_v4(
    record: ScenarioEvaluationRecordV4,
    replay: ReplayArtifactV3,
) -> None:
    """Validate scenario evidence and its context against current official rules.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV4
        Exact ScenarioEvaluationRecordV4 with supplied endpoint results.
    replay : ReplayArtifactV3
        Exact ReplayArtifactV3 carrying the referenced trajectory.

    Returns
    -------
    None
        None.

    Raises
    ------
    TypeError
        A record or context type is unsupported.
    ValueError
        Evidence joins, current mechanics/product config, initial-state
        validity, projection, or canonical SharedObs availability fails.

    Notes
    -----
    Adds live catalog/Core admission to generic scenario validation. This can
    allocate JAX arrays while reconstructing recorded config/state; it does not
    rerun the episode or prove the supplied endpoint calculation.
    """
    validate_scenario_evaluation_record_v4(record, replay)
    _validate_official_scenario_replay_v2(replay)


def build_scenario_evaluation_record_v4(
    specification: ResolvedScenarioSpecificationV3,
    replay: ReplayArtifactV3,
    *,
    schedule_coordinate: int,
    measurement_results: tuple[ScenarioMeasurementResultV1, ...],
    violation_results: tuple[ScenarioViolationResultV1, ...],
    predicate_result: ScenarioPredicateResultV1,
) -> ScenarioEvaluationRecordV4:
    """Build and validate a content-addressed scenario record V4.

    Parameters
    ----------
    specification : ResolvedScenarioSpecificationV3
        Exact ResolvedScenarioSpecificationV3 declared before rollout.
    replay : ReplayArtifactV3
        Exact ReplayArtifactV3 containing the realized evidence.
    schedule_coordinate : int
        Zero-based matched-seed coordinate within the specification.
    measurement_results : tuple[ScenarioMeasurementResultV1, ...]
        Caller-computed results in primary-then-secondary definition order.
    violation_results : tuple[ScenarioViolationResultV1, ...]
        Caller-computed results in exact violation-definition order.
    predicate_result : ScenarioPredicateResultV1
        Caller-computed outcome of the declared success predicate.

    Returns
    -------
    ScenarioEvaluationRecordV4
        Strict ScenarioEvaluationRecordV4 with canonical ID/digest, content references,
        and complete initial-frame digest.

    Raises
    ------
    TypeError
        Required evidence types are unsupported.
    ValueError
        Evidence, declaration/result joins, completion eligibility, or current official
        context admission fails.

    Notes
    -----
    Always applies official live-rule checks, including canonical SharedObs, even though
    specification classification is a recorded field.
    Performs no file I/O and does not compute measurements, predicate truth, or
    a simulator rollout. All supplied endpoint values remain the caller's
    responsibility.
    """
    payload = {
        "schema_id": SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
        "schema_version": 4,
        "record_id": f"{replay.header.context.identity.episode_id}:scenario-evaluation",
        "specification": specification,
        "schedule_coordinate": schedule_coordinate,
        "replay_reference": replay_reference_v3(replay),
        "realized_initial_frame_digest_sha256": canonical_digest_sha256(
            replay.frames[0]
        ),
        "measurement_results": measurement_results,
        "violation_results": violation_results,
        "predicate_result": predicate_result,
    }
    record = ScenarioEvaluationRecordV4.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )
    validate_official_scenario_evaluation_record_v4(record, replay)
    return record


def validate_scenario_evaluation_record_v5(
    record: ScenarioEvaluationRecordV5,
    replay: ReplayArtifactV4,
) -> None:
    """Validate scenario record V5 against its actual recorded evidence.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV5
        Exact ScenarioEvaluationRecordV5 with supplied endpoint results.
    replay : ReplayArtifactV4
        Exact ReplayArtifactV4 carrying the referenced trajectory.

    Returns
    -------
    None
        None.

    Raises
    ------
    TypeError
        A required root type is unsupported.
    ValueError
        Record validity, content references, scenario/config/role/state
        joins, or endpoint completion eligibility fails.

    Notes
    -----
    Host-only validation does not compute supplied measurement values or
    predicate truth. No legacy metric sidecar is needed.
    Current official product admission requires the separate official validator.
    """
    canonical_record = cast(
        ScenarioEvaluationRecordV5,
        validate_declared_model_tree(
            record,
            record_name="scenario evaluation record",
            expected_type=ScenarioEvaluationRecordV5,
        ),
    )
    validate_declared_model_tree(
        replay, record_name="scenario replay", expected_type=ReplayArtifactV4
    )
    if canonical_record.replay_reference != replay_reference_v4(replay):
        raise ValueError("scenario replay reference does not match replay content")
    _validate_scenario_replay_joins(canonical_record, replay, AssignedPolicySlotV2)


def validate_official_scenario_evaluation_record_v5(
    record: ScenarioEvaluationRecordV5,
    replay: ReplayArtifactV4,
) -> None:
    """Validate scenario evidence and its context against current official rules.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV5
        Exact ScenarioEvaluationRecordV5 with supplied endpoint results.
    replay : ReplayArtifactV4
        Exact ReplayArtifactV4 carrying the referenced trajectory.

    Returns
    -------
    None
        None.

    Raises
    ------
    TypeError
        A record or context type is unsupported.
    ValueError
        Evidence joins, current mechanics/product config, initial-state
        validity, projection, or canonical SharedObs availability fails.

    Notes
    -----
    Adds live catalog/Core admission to generic scenario validation. This can
    allocate JAX arrays while reconstructing recorded config/state; it does not
    rerun the episode or prove the supplied endpoint calculation.
    """
    validate_scenario_evaluation_record_v5(record, replay)
    _validate_official_scenario_replay_v2(replay)


def build_scenario_evaluation_record_v5(
    specification: ResolvedScenarioSpecificationV3,
    replay: ReplayArtifactV4,
    *,
    schedule_coordinate: int,
    measurement_results: tuple[ScenarioMeasurementResultV1, ...],
    violation_results: tuple[ScenarioViolationResultV1, ...],
    predicate_result: ScenarioPredicateResultV1,
) -> ScenarioEvaluationRecordV5:
    """Build and validate a content-addressed scenario record V5.

    Parameters
    ----------
    specification : ResolvedScenarioSpecificationV3
        Exact ResolvedScenarioSpecificationV3 declared before rollout.
    replay : ReplayArtifactV4
        Exact ReplayArtifactV4 containing the realized evidence.
    schedule_coordinate : int
        Zero-based matched-seed coordinate within the specification.
    measurement_results : tuple[ScenarioMeasurementResultV1, ...]
        Caller-computed results in primary-then-secondary definition order.
    violation_results : tuple[ScenarioViolationResultV1, ...]
        Caller-computed results in exact violation-definition order.
    predicate_result : ScenarioPredicateResultV1
        Caller-computed outcome of the declared success predicate.

    Returns
    -------
    ScenarioEvaluationRecordV5
        Strict ScenarioEvaluationRecordV5 with canonical ID/digest, content references,
        and complete initial-frame digest.

    Raises
    ------
    TypeError
        Required evidence types are unsupported.
    ValueError
        Evidence, declaration/result joins, completion eligibility, or current official
        context admission fails.

    Notes
    -----
    Always applies official live-rule checks, including canonical SharedObs, even though
    specification classification is a recorded field.
    Performs no file I/O and does not compute measurements, predicate truth, or
    a simulator rollout. All supplied endpoint values remain the caller's
    responsibility.
    """
    payload = {
        "schema_id": SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
        "schema_version": 5,
        "record_id": f"{replay.header.context.identity.episode_id}:scenario-evaluation",
        "specification": specification,
        "schedule_coordinate": schedule_coordinate,
        "replay_reference": replay_reference_v4(replay),
        "realized_initial_frame_digest_sha256": canonical_digest_sha256(
            replay.frames[0]
        ),
        "measurement_results": measurement_results,
        "violation_results": violation_results,
        "predicate_result": predicate_result,
    }
    record = ScenarioEvaluationRecordV5.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )
    validate_official_scenario_evaluation_record_v5(record, replay)
    return record


__all__ = [
    "SCENARIO_EVALUATION_RECORD_SCHEMA_ID",
    "SCENARIO_MEASUREMENT_DEFINITION_SCHEMA_ID",
    "SCENARIO_MEASUREMENT_RESULT_SCHEMA_ID",
    "SCENARIO_PREDICATE_RESULT_SCHEMA_ID",
    "SCENARIO_SCHEMA_VERSION",
    "SCENARIO_SCHEMA_VERSION_V2",
    "SCENARIO_SCHEMA_VERSION_V3",
    "SCENARIO_SCHEMA_VERSION_V4",
    "SCENARIO_SCHEMA_VERSION_V5",
    "SCENARIO_SEED_SCHEDULE_SCHEMA_ID",
    "SCENARIO_SPECIFICATION_SCHEMA_ID",
    "SCENARIO_VIOLATION_DEFINITION_SCHEMA_ID",
    "SCENARIO_VIOLATION_RESULT_SCHEMA_ID",
    "ResolvedScenarioSpecificationV1",
    "ResolvedScenarioSpecificationV2",
    "ResolvedScenarioSpecificationV3",
    "ScenarioBooleanValueV1",
    "ScenarioClassification",
    "ScenarioCompletionScope",
    "ScenarioCountValueV1",
    "ScenarioEndpointObservationStatus",
    "ScenarioEvaluationRecordV1",
    "ScenarioEvaluationRecordV2",
    "ScenarioEvaluationRecordV3",
    "ScenarioEvaluationRecordV4",
    "ScenarioEvaluationRecordV5",
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
    "ScenarioSeedScheduleV3",
    "ScenarioValueType",
    "ScenarioViolationDefinitionV1",
    "ScenarioViolationResultV1",
    "build_scenario_evaluation_record_v1",
    "build_scenario_evaluation_record_v2",
    "build_scenario_evaluation_record_v3",
    "build_scenario_evaluation_record_v4",
    "build_scenario_evaluation_record_v5",
    "resolved_initial_state_digest_sha256",
    "resolved_initial_state_digest_sha256_v2",
    "validate_official_scenario_evaluation_record_v2",
    "validate_official_scenario_evaluation_record_v3",
    "validate_official_scenario_evaluation_record_v4",
    "validate_official_scenario_evaluation_record_v5",
    "validate_scenario_evaluation_record_v1",
    "validate_scenario_evaluation_record_v2",
    "validate_scenario_evaluation_record_v3",
    "validate_scenario_evaluation_record_v4",
    "validate_scenario_evaluation_record_v5",
]
