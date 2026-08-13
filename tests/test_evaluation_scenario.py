"""Focused contract tests for versioned controlled-scenario records."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError
from tests.evaluation_fixtures import captured_evaluation_trajectory

import marl_battlegrounds.evaluation.scenario as scenario_module
from marl_battlegrounds.evaluation.metrics import build_evaluation_observer_v1
from marl_battlegrounds.evaluation.models import (
    ContentAddressedIdentityV1,
    EvaluationEpisodeContextV1,
    EvaluationEpisodeIdentityV1,
    VersionedIdentityV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.replay import (
    EvaluationMetricReportArtifactV1,
    ReplayArtifactReferenceV1,
    ReplayArtifactV1,
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.scenario import (
    SCENARIO_SPECIFICATION_SCHEMA_ID,
    ResolvedScenarioSpecificationV1,
    ScenarioBooleanValueV1,
    ScenarioCompletionScope,
    ScenarioCountValueV1,
    ScenarioEvaluationRecordV1,
    ScenarioMeasurementDefinitionV1,
    ScenarioMeasurementResultV1,
    ScenarioParameterV1,
    ScenarioPredicateResultV1,
    ScenarioScalarValueV1,
    ScenarioViolationDefinitionV1,
    ScenarioViolationResultV1,
    build_scenario_evaluation_record_v1,
    validate_scenario_evaluation_record_v1,
)

_DIGEST_A = "1" * 64
_DIGEST_B = "2" * 64
_DIGEST_C = "3" * 64


def _runtime_provenance() -> RuntimeProvenanceV1:
    return RuntimeProvenanceV1(
        python_version="3.14",
        package_version="0.0.0",
        jax_version="test",
        jaxlib_version="test",
        numpy_version="test",
        pydantic_version="test",
        platform="test-platform",
        machine="test-machine",
        backend="cpu",
        device="test-device",
        precision="float32",
        environment_count=1,
        batch_shape=(),
        policy_execution_included=True,
    )


def _specification(
    context: EvaluationEpisodeContextV1,
    *,
    primary_completion_scope: ScenarioCompletionScope = "complete_episode",
) -> ResolvedScenarioSpecificationV1:
    payload: dict[str, object] = {
        "schema_id": SCENARIO_SPECIFICATION_SCHEMA_ID,
        "schema_version": 1,
        "scenario_id": "scenario",
        "scenario_version": 1,
        "classification": "official",
        "hypothesis": "The focal policy realizes the declared endpoint.",
        "eligible_roles": (
            "focal",
            "cooperative_partner",
            "adversarial_opponent",
        ),
        "authored_initial_condition": ContentAddressedIdentityV1(
            identifier="initial-condition",
            version=1,
            canonical_digest=_DIGEST_A,
        ),
        "parameters": (
            ScenarioParameterV1(name="pressure", value=1.5),
            ScenarioParameterV1(name="seed_family", value="matched"),
        ),
        "resolved_config_digest_sha256": (
            context.resolved_env_config.canonical_digest_sha256
        ),
        "horizon": context.expected_horizon,
        "pressure_protocol": ContentAddressedIdentityV1(
            identifier="pressure-protocol",
            version=1,
            canonical_digest=_DIGEST_B,
        ),
        "primary_measurement": ScenarioMeasurementDefinitionV1(
            measurement_id="test.endpoint",
            measurement_version=1,
            role="primary",
            value_type="boolean",
            units="boolean",
            completion_scope=primary_completion_scope,
            supports_right_censoring=True,
        ),
        "secondary_measurements": (
            ScenarioMeasurementDefinitionV1(
                measurement_id="test.margin",
                measurement_version=1,
                role="secondary",
                value_type="scalar",
                units="world_units",
                completion_scope="any_gap_free_prefix",
                supports_right_censoring=False,
            ),
        ),
        "violations": (
            ScenarioViolationDefinitionV1(
                violation_id="test.role_violation",
                violation_version=1,
                value_type="count",
                units="count",
                completion_scope="complete_episode",
                supports_right_censoring=False,
            ),
        ),
        "success_predicate": VersionedIdentityV1(
            identifier="test.success",
            version=1,
        ),
        "completion_policy": VersionedIdentityV1(
            identifier="test.completion",
            version=1,
        ),
        "partial_result_policy": VersionedIdentityV1(
            identifier="test.partial-results",
            version=1,
        ),
    }
    return ResolvedScenarioSpecificationV1.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )


def _context_with_specification(
    context: EvaluationEpisodeContextV1,
    specification: ResolvedScenarioSpecificationV1,
) -> EvaluationEpisodeContextV1:
    identity_payload = context.identity.model_dump(mode="python")
    identity_payload["scenario"] = ContentAddressedIdentityV1(
        identifier=specification.scenario_id,
        version=specification.scenario_version,
        canonical_digest=specification.canonical_digest_sha256,
    )
    identity = EvaluationEpisodeIdentityV1.model_validate(identity_payload)
    context_payload = context.model_dump(mode="python")
    context_payload["identity"] = identity
    return EvaluationEpisodeContextV1.model_validate(context_payload)


def _defined_measurement_results() -> tuple[ScenarioMeasurementResultV1, ...]:
    return (
        ScenarioMeasurementResultV1(
            measurement_id="test.endpoint",
            measurement_version=1,
            result_status="defined",
            endpoint_observation_status="observed",
            value=ScenarioBooleanValueV1(value=True),
        ),
        ScenarioMeasurementResultV1(
            measurement_id="test.margin",
            measurement_version=1,
            result_status="defined",
            endpoint_observation_status="not_applicable",
            value=ScenarioScalarValueV1(value=1.25),
        ),
    )


def _defined_violation_results() -> tuple[ScenarioViolationResultV1, ...]:
    return (
        ScenarioViolationResultV1(
            violation_id="test.role_violation",
            violation_version=1,
            result_status="defined",
            endpoint_observation_status="observed",
            value=ScenarioCountValueV1(value=0),
        ),
    )


def _predicate_result() -> ScenarioPredicateResultV1:
    return ScenarioPredicateResultV1(
        predicate_id="test.success",
        predicate_version=1,
        status="satisfied",
    )


@dataclass(frozen=True, slots=True)
class _ScenarioArtifacts:
    specification: ResolvedScenarioSpecificationV1
    replay: ReplayArtifactV1
    report_artifact: EvaluationMetricReportArtifactV1
    record: ScenarioEvaluationRecordV1


@pytest.fixture(scope="module")
def scenario_artifacts() -> _ScenarioArtifacts:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        capture_profile="scenario_metric_complete",
        expected_horizon=1,
        with_scenario=True,
    )
    specification = _specification(trajectory.context)
    context = _context_with_specification(trajectory.context, specification)
    observer = build_evaluation_observer_v1(context)
    observer.start(trajectory.frames[0])
    observer.append(trajectory.transitions[0], trajectory.frames[1])
    report = observer.finalize(completion_state="complete")
    bundle = build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=_runtime_provenance(),
    )
    record = build_scenario_evaluation_record_v1(
        specification,
        bundle.replay,
        bundle.metric_report_artifact,
        measurement_results=_defined_measurement_results(),
        violation_results=_defined_violation_results(),
        predicate_result=_predicate_result(),
    )
    return _ScenarioArtifacts(
        specification=specification,
        replay=bundle.replay,
        report_artifact=bundle.metric_report_artifact,
        record=record,
    )


def _redigest_specification(
    payload: dict[str, object],
) -> ResolvedScenarioSpecificationV1:
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )
    return ResolvedScenarioSpecificationV1.model_validate(payload)


def _redigest_record(payload: dict[str, object]) -> ScenarioEvaluationRecordV1:
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )
    return ScenarioEvaluationRecordV1.model_validate(payload)


def test_scenario_specification_and_record_round_trip_strictly(
    scenario_artifacts: _ScenarioArtifacts,
) -> None:
    specification = scenario_artifacts.specification
    record = scenario_artifacts.record

    assert (
        ResolvedScenarioSpecificationV1.model_validate_json(
            specification.model_dump_json()
        )
        == specification
    )
    assert (
        ScenarioEvaluationRecordV1.model_validate_json(record.model_dump_json())
        == record
    )
    assert record.record_id == "episode-001:scenario-evaluation"
    assert record.replay_reference.episode_id == "episode-001"
    assert (
        record.metric_report_reference.trajectory_content_digest_sha256
        == record.replay_reference.trajectory_content_digest_sha256
    )
    validate_scenario_evaluation_record_v1(
        record,
        scenario_artifacts.replay,
        scenario_artifacts.report_artifact,
    )


def test_specification_requires_canonical_roles_parameters_and_definitions(
    scenario_artifacts: _ScenarioArtifacts,
) -> None:
    payload = scenario_artifacts.specification.model_dump(mode="python")

    unsorted_roles = dict(payload)
    unsorted_roles["eligible_roles"] = (
        "adversarial_opponent",
        "focal",
    )
    with pytest.raises(ValidationError, match="roles must be canonically sorted"):
        _redigest_specification(unsorted_roles)

    duplicate_parameters = dict(payload)
    duplicate_parameters["parameters"] = (
        ScenarioParameterV1(name="pressure", value=1),
        ScenarioParameterV1(name="pressure", value=2),
    )
    with pytest.raises(ValidationError, match="parameter names must be unique"):
        _redigest_specification(duplicate_parameters)

    wrong_primary = dict(payload)
    primary = scenario_artifacts.specification.primary_measurement
    wrong_primary["primary_measurement"] = primary.model_copy(
        update={"role": "secondary"}
    )
    with pytest.raises(ValidationError, match="primary role"):
        _redigest_specification(wrong_primary)

    too_many_secondary = dict(payload)
    secondary = scenario_artifacts.specification.secondary_measurements[0]
    too_many_secondary["secondary_measurements"] = (
        secondary,
        secondary.model_copy(update={"measurement_id": "test.margin_b"}),
        secondary.model_copy(update={"measurement_id": "test.margin_c"}),
    )
    with pytest.raises(ValidationError):
        _redigest_specification(too_many_secondary)

    duplicate_violations = dict(payload)
    violation = scenario_artifacts.specification.violations[0]
    duplicate_violations["violations"] = (violation, violation)
    with pytest.raises(ValidationError, match="violation IDs must be unique"):
        _redigest_specification(duplicate_violations)


def test_scenario_roots_reject_lists_bool_as_int_nonfinite_and_digest_tampering(
    scenario_artifacts: _ScenarioArtifacts,
) -> None:
    specification_payload = scenario_artifacts.specification.model_dump(mode="python")

    list_payload = dict(specification_payload)
    list_payload["eligible_roles"] = ["focal"]
    with pytest.raises(ValidationError):
        _redigest_specification(list_payload)

    bool_horizon = dict(specification_payload)
    bool_horizon["horizon"] = True
    with pytest.raises(ValidationError):
        _redigest_specification(bool_horizon)

    with pytest.raises(ValidationError):
        ScenarioScalarValueV1(value=float("nan"))
    with pytest.raises(ValidationError):
        ScenarioCountValueV1(value=True)

    tampered_digest = dict(specification_payload)
    tampered_digest["canonical_digest_sha256"] = _DIGEST_C
    with pytest.raises(ValidationError, match="digest mismatch"):
        ResolvedScenarioSpecificationV1.model_validate(tampered_digest)

    unknown_version = dict(specification_payload)
    unknown_version["schema_version"] = 2
    with pytest.raises(ValidationError):
        ResolvedScenarioSpecificationV1.model_validate(unknown_version)


def test_result_values_statuses_and_predicate_reasons_are_coherent() -> None:
    with pytest.raises(ValidationError, match="require a typed value"):
        ScenarioMeasurementResultV1(
            measurement_id="test.endpoint",
            measurement_version=1,
            result_status="defined",
            endpoint_observation_status="observed",
            value=None,
        )
    with pytest.raises(ValidationError, match="must not carry a value"):
        ScenarioViolationResultV1(
            violation_id="test.violation",
            violation_version=1,
            result_status="insufficient_data",
            endpoint_observation_status="unavailable",
            value=ScenarioCountValueV1(value=1),
            reason="partial prefix",
        )
    with pytest.raises(ValidationError, match="unavailable endpoint"):
        ScenarioMeasurementResultV1(
            measurement_id="test.endpoint",
            measurement_version=1,
            result_status="unavailable",
            endpoint_observation_status="observed",
            value=None,
            reason="source absent",
        )
    with pytest.raises(ValidationError, match="require a reason"):
        ScenarioPredicateResultV1(
            predicate_id="test.success",
            predicate_version=1,
            status="unavailable",
        )
    with pytest.raises(ValidationError, match="forbid a reason"):
        ScenarioPredicateResultV1(
            predicate_id="test.success",
            predicate_version=1,
            status="satisfied",
            reason="not allowed",
        )


def test_every_independently_serialized_scenario_root_rejects_bad_versions(
    scenario_artifacts: _ScenarioArtifacts,
) -> None:
    roots = (
        scenario_artifacts.specification.primary_measurement,
        scenario_artifacts.record.measurement_results[0],
        scenario_artifacts.specification.violations[0],
        scenario_artifacts.record.violation_results[0],
        scenario_artifacts.record.predicate_result,
    )
    for root in roots:
        for invalid_version in (True, 1.0, 2):
            payload = root.model_dump(mode="python")
            payload["schema_version"] = invalid_version
            with pytest.raises(ValidationError):
                type(root).model_validate(payload)


def test_record_requires_exact_definition_order_types_and_predicate_identity(
    scenario_artifacts: _ScenarioArtifacts,
) -> None:
    payload = scenario_artifacts.record.model_dump(mode="python")

    reordered = dict(payload)
    reordered["measurement_results"] = tuple(
        reversed(scenario_artifacts.record.measurement_results)
    )
    with pytest.raises(ValidationError, match="exactly follow their definitions"):
        _redigest_record(reordered)

    wrong_type = dict(payload)
    wrong_type["measurement_results"] = (
        scenario_artifacts.record.measurement_results[0].model_copy(
            update={"value": ScenarioCountValueV1(value=1)}
        ),
        scenario_artifacts.record.measurement_results[1],
    )
    with pytest.raises(ValidationError, match="value type must match"):
        _redigest_record(wrong_type)

    wrong_violation_type = dict(payload)
    wrong_violation_type["violation_results"] = (
        scenario_artifacts.record.violation_results[0].model_copy(
            update={"value": ScenarioBooleanValueV1(value=False)}
        ),
    )
    with pytest.raises(ValidationError, match="violation result value type"):
        _redigest_record(wrong_violation_type)

    wrong_predicate = dict(payload)
    wrong_predicate["predicate_result"] = ScenarioPredicateResultV1(
        predicate_id="test.other",
        predicate_version=1,
        status="satisfied",
    )
    with pytest.raises(ValidationError, match="declared success predicate"):
        _redigest_record(wrong_predicate)

    missing_violation = dict(payload)
    missing_violation["violation_results"] = ()
    with pytest.raises(ValidationError, match="exactly follow their definitions"):
        _redigest_record(missing_violation)


def test_censored_results_require_declared_definition_support(
    scenario_artifacts: _ScenarioArtifacts,
) -> None:
    payload = scenario_artifacts.record.model_dump(mode="python")
    primary_result, secondary_result = scenario_artifacts.record.measurement_results

    supported_primary = dict(payload)
    supported_primary["measurement_results"] = (
        primary_result.model_copy(
            update={"endpoint_observation_status": "right_censored"}
        ),
        secondary_result,
    )
    record = _redigest_record(supported_primary)
    validate_scenario_evaluation_record_v1(
        record,
        scenario_artifacts.replay,
        scenario_artifacts.report_artifact,
    )

    unsupported_secondary = dict(payload)
    unsupported_secondary["measurement_results"] = (
        primary_result,
        secondary_result.model_copy(
            update={"endpoint_observation_status": "right_censored"}
        ),
    )
    with pytest.raises(ValidationError, match="requires declared support"):
        _redigest_record(unsupported_secondary)

    insufficient_secondary = dict(payload)
    insufficient_secondary["measurement_results"] = (
        primary_result,
        ScenarioMeasurementResultV1(
            measurement_id=secondary_result.measurement_id,
            measurement_version=secondary_result.measurement_version,
            result_status="insufficient_data",
            endpoint_observation_status="right_censored",
            value=None,
            reason="the endpoint was censored before enough observations accrued",
        ),
    )
    with pytest.raises(ValidationError, match="requires declared support"):
        _redigest_record(insufficient_secondary)

    unsupported_violation = dict(payload)
    unsupported_violation["violation_results"] = (
        scenario_artifacts.record.violation_results[0].model_copy(
            update={"endpoint_observation_status": "right_censored"}
        ),
    )
    with pytest.raises(ValidationError, match="violation requires declared support"):
        _redigest_record(unsupported_violation)


def test_cross_validation_rejects_wrong_replay_and_initial_frame_references(
    scenario_artifacts: _ScenarioArtifacts,
) -> None:
    payload = scenario_artifacts.record.model_dump(mode="python")

    wrong_initial = dict(payload)
    wrong_initial["realized_initial_frame_digest_sha256"] = _DIGEST_C
    record = _redigest_record(wrong_initial)
    with pytest.raises(ValueError, match="initial-frame digest"):
        validate_scenario_evaluation_record_v1(
            record,
            scenario_artifacts.replay,
            scenario_artifacts.report_artifact,
        )

    replay_reference = scenario_artifacts.record.replay_reference
    wrong_reference = ReplayArtifactReferenceV1(
        artifact_id=replay_reference.artifact_id,
        episode_id=replay_reference.episode_id,
        context_digest_sha256=replay_reference.context_digest_sha256,
        trajectory_content_digest_sha256=(
            replay_reference.trajectory_content_digest_sha256
        ),
        canonical_digest_sha256=_DIGEST_C,
        canonical_byte_length=replay_reference.canonical_byte_length,
    )
    wrong_replay = dict(payload)
    wrong_replay["replay_reference"] = wrong_reference
    record = _redigest_record(wrong_replay)
    with pytest.raises(ValueError, match="does not match replay content"):
        validate_scenario_evaluation_record_v1(
            record,
            scenario_artifacts.replay,
            scenario_artifacts.report_artifact,
        )


def test_partial_replay_enforces_completion_scope_and_censoring_eligibility() -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=0,
        capture_profile="scenario_metric_complete",
        expected_horizon=1,
        with_scenario=True,
    )
    specification = _specification(trajectory.context)
    context = _context_with_specification(trajectory.context, specification)
    observer = build_evaluation_observer_v1(context)
    observer.start(trajectory.frames[0])
    report = observer.finalize(
        completion_state="partial",
        end_or_failure_reason="controlled partial prefix",
    )
    bundle = build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=_runtime_provenance(),
    )

    with pytest.raises(ValueError, match="complete-episode measurement"):
        build_scenario_evaluation_record_v1(
            specification,
            bundle.replay,
            bundle.metric_report_artifact,
            measurement_results=_defined_measurement_results(),
            violation_results=_defined_violation_results(),
            predicate_result=_predicate_result(),
        )

    insufficient_primary = ScenarioMeasurementResultV1(
        measurement_id="test.endpoint",
        measurement_version=1,
        result_status="insufficient_data",
        endpoint_observation_status="right_censored",
        value=None,
        reason="horizon was not observed",
    )
    prefix_secondary = _defined_measurement_results()[1]
    insufficient_violation = ScenarioViolationResultV1(
        violation_id="test.role_violation",
        violation_version=1,
        result_status="insufficient_data",
        endpoint_observation_status="unavailable",
        value=None,
        reason="complete episode required",
    )
    with pytest.raises(ValueError, match="complete declared-horizon evidence"):
        build_scenario_evaluation_record_v1(
            specification,
            bundle.replay,
            bundle.metric_report_artifact,
            measurement_results=(insufficient_primary, prefix_secondary),
            violation_results=(insufficient_violation,),
            predicate_result=ScenarioPredicateResultV1(
                predicate_id="test.success",
                predicate_version=1,
                status="unavailable",
                reason="complete episode required",
            ),
        )

    competing_primary = insufficient_primary.model_copy(
        update={"endpoint_observation_status": "competing_event"}
    )
    with pytest.raises(ValueError, match="require a complete rollout"):
        build_scenario_evaluation_record_v1(
            specification,
            bundle.replay,
            bundle.metric_report_artifact,
            measurement_results=(competing_primary, prefix_secondary),
            violation_results=(insufficient_violation,),
            predicate_result=ScenarioPredicateResultV1(
                predicate_id="test.success",
                predicate_version=1,
                status="unavailable",
                reason="complete episode required",
            ),
        )


def test_scenario_build_and_validation_each_scan_replay_once(
    scenario_artifacts: _ScenarioArtifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_passes = 0
    original_validate = scenario_module.validate_replay_artifact_v1

    def counted_validate(replay: ReplayArtifactV1) -> None:
        nonlocal semantic_passes
        semantic_passes += 1
        original_validate(replay)

    monkeypatch.setattr(
        scenario_module,
        "validate_replay_artifact_v1",
        counted_validate,
    )
    rebuilt = build_scenario_evaluation_record_v1(
        scenario_artifacts.specification,
        scenario_artifacts.replay,
        scenario_artifacts.report_artifact,
        measurement_results=_defined_measurement_results(),
        violation_results=_defined_violation_results(),
        predicate_result=_predicate_result(),
    )
    assert semantic_passes == 1

    semantic_passes = 0
    validate_scenario_evaluation_record_v1(
        rebuilt,
        scenario_artifacts.replay,
        scenario_artifacts.report_artifact,
    )
    assert semantic_passes == 1


def test_specification_and_record_are_frozen(
    scenario_artifacts: _ScenarioArtifacts,
) -> None:
    with pytest.raises(ValidationError):
        scenario_artifacts.specification.horizon = 99
    with pytest.raises(ValidationError):
        scenario_artifacts.record.record_id = "changed"
