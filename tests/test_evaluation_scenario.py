"""Focused contract tests for versioned controlled-scenario records."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    evaluation_env_config,
)

import marl_battlegrounds.evaluation.catalog as catalog_module
import marl_battlegrounds.evaluation.scenario as scenario_module
from marl_battlegrounds.evaluation.metrics import build_evaluation_observer_v1
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV1,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContextV1,
    EvaluationEpisodeIdentityV1,
    EvaluationFrameV1,
    EvaluationSeedProtocolV1,
    ResolvedEnvConfigV1,
    StaticMechanicsCatalogV1,
    VersionedIdentityV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.replay import (
    EvaluationMetricReportArtifactV1,
    ReplayArtifactReferenceV1,
    ReplayArtifactV1,
    ReplayBundleV1,
    RuntimeProvenanceV1,
    build_replay_artifact_reference_v1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.scenario import (
    SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
    SCENARIO_SEED_SCHEDULE_SCHEMA_ID,
    SCENARIO_SPECIFICATION_SCHEMA_ID,
    ResolvedScenarioSpecificationV1,
    ResolvedScenarioSpecificationV2,
    ScenarioBooleanValueV1,
    ScenarioCompletionScope,
    ScenarioCountValueV1,
    ScenarioEvaluationRecordV1,
    ScenarioEvaluationRecordV2,
    ScenarioMeasurementDefinitionV1,
    ScenarioMeasurementResultV1,
    ScenarioParameterV1,
    ScenarioPredicateResultV1,
    ScenarioScalarValueV1,
    ScenarioSeedScheduleV2,
    ScenarioViolationDefinitionV1,
    ScenarioViolationResultV1,
    build_scenario_evaluation_record_v1,
    build_scenario_evaluation_record_v2,
    resolved_initial_state_digest_sha256_v2,
    validate_official_scenario_evaluation_record_v2,
    validate_scenario_evaluation_record_v1,
    validate_scenario_evaluation_record_v2,
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


def _seed_protocol_with_updates(
    row: EvaluationSeedProtocolV1,
    **updates: object,
) -> EvaluationSeedProtocolV1:
    payload = row.model_dump(mode="python")
    payload.update(updates)
    return EvaluationSeedProtocolV1.model_validate(payload)


def _seed_schedule_v2(
    realized_seed_protocols: tuple[EvaluationSeedProtocolV1, ...],
) -> ScenarioSeedScheduleV2:
    payload: dict[str, object] = {
        "schema_id": SCENARIO_SEED_SCHEDULE_SCHEMA_ID,
        "schema_version": 2,
        "schedule_id": "matched-seeds",
        "schedule_version": 1,
        "realized_seed_protocols": realized_seed_protocols,
    }
    return ScenarioSeedScheduleV2.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )


def _role_template_v2(
    context: EvaluationEpisodeContextV1,
) -> tuple[str, ...]:
    return tuple(
        assignment.evaluation_role
        if isinstance(assignment, AssignedPolicySlotV1)
        else "not_applicable"
        for assignment in context.policy_assignments
    )


def _specification_v2(
    context: EvaluationEpisodeContextV1,
    initial_frame: EvaluationFrameV1,
    seed_schedule: ScenarioSeedScheduleV2,
) -> ResolvedScenarioSpecificationV2:
    payload: dict[str, object] = {
        "schema_id": SCENARIO_SPECIFICATION_SCHEMA_ID,
        "schema_version": 2,
        "scenario_id": "scenario",
        "scenario_version": 1,
        "classification": "official",
        "hypothesis": "The frozen slots realize the declared matched endpoint.",
        "layout": context.identity.layout,
        "authored_initial_condition": ContentAddressedIdentityV1(
            identifier="authored-initial-condition",
            version=1,
            canonical_digest=_DIGEST_C,
        ),
        "resolved_initial_state_digest_sha256": (
            resolved_initial_state_digest_sha256_v2(initial_frame)
        ),
        "parameters": (
            ScenarioParameterV1(name="pressure", value=1.5),
            ScenarioParameterV1(name="seed_family", value="matched"),
        ),
        "resolved_config_digest_sha256": (
            context.resolved_env_config.canonical_digest_sha256
        ),
        "roster_template": context.roster,
        "role_template": _role_template_v2(context),
        "seed_schedule": seed_schedule,
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
            completion_scope="complete_episode",
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
    return ResolvedScenarioSpecificationV2.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )


def _context_with_specification_v2(
    context: EvaluationEpisodeContextV1,
    specification: ResolvedScenarioSpecificationV2,
    *,
    schedule_coordinate: int,
    policy_assignments: tuple[object, ...] | None = None,
) -> EvaluationEpisodeContextV1:
    identity_payload = context.identity.model_dump(mode="python")
    identity_payload["scenario"] = ContentAddressedIdentityV1(
        identifier=specification.scenario_id,
        version=specification.scenario_version,
        canonical_digest=specification.canonical_digest_sha256,
    )
    context_payload = context.model_dump(mode="python")
    context_payload.update(
        {
            "identity": EvaluationEpisodeIdentityV1.model_validate(identity_payload),
            "seed_protocol": (
                specification.seed_schedule.realized_seed_protocols[schedule_coordinate]
            ),
        }
    )
    if policy_assignments is not None:
        context_payload["policy_assignments"] = policy_assignments
    return EvaluationEpisodeContextV1.model_validate(context_payload)


def _bundle_for_context(
    trajectory: CapturedEvaluationTrajectory,
    context: EvaluationEpisodeContextV1,
    *,
    partial: bool = False,
) -> ReplayBundleV1:
    observer = build_evaluation_observer_v1(context)
    observer.start(trajectory.frames[0])
    for transition, successor_frame in zip(
        trajectory.transitions,
        trajectory.frames[1:],
        strict=True,
    ):
        observer.append(transition, successor_frame)
    report = (
        observer.finalize(
            completion_state="partial",
            end_or_failure_reason="controlled hostile initial-state fixture",
        )
        if partial
        else observer.finalize(completion_state="complete")
    )
    return build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=_runtime_provenance(),
    )


@dataclass(frozen=True, slots=True)
class _ScenarioArtifactsV2:
    trajectories: tuple[CapturedEvaluationTrajectory, CapturedEvaluationTrajectory]
    contexts: tuple[EvaluationEpisodeContextV1, EvaluationEpisodeContextV1]
    specification: ResolvedScenarioSpecificationV2
    bundles: tuple[ReplayBundleV1, ReplayBundleV1]
    records: tuple[ScenarioEvaluationRecordV2, ScenarioEvaluationRecordV2]


@pytest.fixture(scope="module")
def scenario_artifacts_v2() -> _ScenarioArtifactsV2:
    trajectories = (
        captured_evaluation_trajectory(
            transition_count=1,
            capture_profile="scenario_metric_complete",
            expected_horizon=1,
            with_scenario=True,
            episode_id="episode-v2-a",
        ),
        captured_evaluation_trajectory(
            transition_count=1,
            capture_profile="scenario_metric_complete",
            expected_horizon=1,
            with_scenario=True,
            episode_id="episode-v2-b",
        ),
    )
    first_seed = trajectories[0].context.seed_protocol
    second_seed = _seed_protocol_with_updates(
        first_seed,
        root_seed=101,
        episode_seed=102,
        layout_seed=103,
        environment_seed=104,
        focal_policy_seed=105,
        evaluation_seed=106,
        cooperative_partner_seed=107,
        adversarial_opponent_seed=108,
        scenario_seed=109,
    )
    schedule = _seed_schedule_v2((first_seed, second_seed))
    specification = _specification_v2(
        trajectories[0].context,
        trajectories[0].frames[0],
        schedule,
    )
    contexts = (
        _context_with_specification_v2(
            trajectories[0].context,
            specification,
            schedule_coordinate=0,
        ),
        _context_with_specification_v2(
            trajectories[1].context,
            specification,
            schedule_coordinate=1,
        ),
    )
    bundles = (
        _bundle_for_context(trajectories[0], contexts[0]),
        _bundle_for_context(trajectories[1], contexts[1]),
    )
    records = (
        build_scenario_evaluation_record_v2(
            specification,
            bundles[0].replay,
            bundles[0].metric_report_artifact,
            schedule_coordinate=0,
            measurement_results=_defined_measurement_results(),
            violation_results=_defined_violation_results(),
            predicate_result=_predicate_result(),
        ),
        build_scenario_evaluation_record_v2(
            specification,
            bundles[1].replay,
            bundles[1].metric_report_artifact,
            schedule_coordinate=1,
            measurement_results=_defined_measurement_results(),
            violation_results=_defined_violation_results(),
            predicate_result=_predicate_result(),
        ),
    )
    return _ScenarioArtifactsV2(
        trajectories=trajectories,
        contexts=contexts,
        specification=specification,
        bundles=bundles,
        records=records,
    )


def _redigest_seed_schedule_v2(
    payload: dict[str, object],
) -> ScenarioSeedScheduleV2:
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )
    return ScenarioSeedScheduleV2.model_validate(payload)


def _resolved_config_with_updates(
    config: ResolvedEnvConfigV1,
    **updates: object,
) -> ResolvedEnvConfigV1:
    payload = config.model_dump(mode="python")
    payload.update(updates)
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )
    return ResolvedEnvConfigV1.model_validate(payload)


def _catalog_with_updates(
    catalog: StaticMechanicsCatalogV1,
    **updates: object,
) -> StaticMechanicsCatalogV1:
    payload = catalog.model_dump(mode="python")
    payload.update(updates)
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )
    return StaticMechanicsCatalogV1.model_validate(payload)


def _redigest_specification_v2(
    payload: dict[str, object],
) -> ResolvedScenarioSpecificationV2:
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )
    return ResolvedScenarioSpecificationV2.model_validate(payload)


def _redigest_record_v2(
    payload: dict[str, object],
) -> ScenarioEvaluationRecordV2:
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )
    return ScenarioEvaluationRecordV2.model_validate(payload)


def _specification_with_updates_v2(
    specification: ResolvedScenarioSpecificationV2,
    **updates: object,
) -> ResolvedScenarioSpecificationV2:
    payload = specification.model_dump(mode="python")
    payload.update(updates)
    return _redigest_specification_v2(payload)


def _build_record_v2(
    specification: ResolvedScenarioSpecificationV2,
    bundle: ReplayBundleV1,
    *,
    schedule_coordinate: int,
) -> ScenarioEvaluationRecordV2:
    return build_scenario_evaluation_record_v2(
        specification,
        bundle.replay,
        bundle.metric_report_artifact,
        schedule_coordinate=schedule_coordinate,
        measurement_results=_defined_measurement_results(),
        violation_results=_defined_violation_results(),
        predicate_result=_predicate_result(),
    )


def _record_without_official_gate_v2(
    specification: ResolvedScenarioSpecificationV2,
    bundle: ReplayBundleV1,
    *,
    schedule_coordinate: int,
    measurement_results: tuple[ScenarioMeasurementResultV1, ...] | None = None,
    violation_results: tuple[ScenarioViolationResultV1, ...] | None = None,
    predicate_result: ScenarioPredicateResultV1 | None = None,
) -> ScenarioEvaluationRecordV2:
    replay = bundle.replay
    replay_reference = build_replay_artifact_reference_v1(replay)
    payload: dict[str, object] = {
        "schema_id": SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
        "schema_version": 2,
        "record_id": f"{replay.header.context.identity.episode_id}:scenario-evaluation",
        "specification": specification,
        "schedule_coordinate": schedule_coordinate,
        "replay_reference": replay_reference,
        "metric_report_reference": replay.metric_report_reference,
        "realized_initial_frame_digest_sha256": canonical_digest_sha256(
            replay.frames[0]
        ),
        "measurement_results": (
            _defined_measurement_results()
            if measurement_results is None
            else measurement_results
        ),
        "violation_results": (
            _defined_violation_results()
            if violation_results is None
            else violation_results
        ),
        "predicate_result": (
            _predicate_result() if predicate_result is None else predicate_result
        ),
    }
    return ScenarioEvaluationRecordV2.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )


def _pure_v2_artifacts_for_context(
    trajectory: CapturedEvaluationTrajectory,
    context: EvaluationEpisodeContextV1,
    seed_schedule: ScenarioSeedScheduleV2,
) -> tuple[
    ResolvedScenarioSpecificationV2,
    ReplayBundleV1,
    ScenarioEvaluationRecordV2,
]:
    specification = _specification_v2(
        context,
        trajectory.frames[0],
        seed_schedule,
    )
    bound_context = _context_with_specification_v2(
        context,
        specification,
        schedule_coordinate=0,
    )
    bundle = _bundle_for_context(trajectory, bound_context)
    record = _record_without_official_gate_v2(
        specification,
        bundle,
        schedule_coordinate=0,
    )
    validate_scenario_evaluation_record_v2(
        record,
        bundle.replay,
        bundle.metric_report_artifact,
    )
    return specification, bundle, record


def test_v2_roots_round_trip_strictly_and_each_coordinate_selects_one_seed_row(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
) -> None:
    specification = scenario_artifacts_v2.specification
    schedule = specification.seed_schedule

    assert (
        ScenarioSeedScheduleV2.model_validate_json(schedule.model_dump_json())
        == schedule
    )
    assert (
        ResolvedScenarioSpecificationV2.model_validate_json(
            specification.model_dump_json()
        )
        == specification
    )
    for coordinate, (context, bundle, record) in enumerate(
        zip(
            scenario_artifacts_v2.contexts,
            scenario_artifacts_v2.bundles,
            scenario_artifacts_v2.records,
            strict=True,
        )
    ):
        assert (
            ScenarioEvaluationRecordV2.model_validate_json(record.model_dump_json())
            == record
        )
        assert record.schema_version == 2
        assert record.schedule_coordinate == coordinate
        assert context.seed_protocol == schedule.realized_seed_protocols[coordinate]
        assert record.specification == specification
        validate_scenario_evaluation_record_v2(
            record,
            bundle.replay,
            bundle.metric_report_artifact,
        )
        validate_official_scenario_evaluation_record_v2(
            record,
            bundle.replay,
            bundle.metric_report_artifact,
        )


def test_v2_resolved_state_digest_excludes_attempt_and_policy_material(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
) -> None:
    frame = scenario_artifacts_v2.bundles[0].replay.frames[0]
    expected_digest = resolved_initial_state_digest_sha256_v2(frame)

    renamed_attempt = frame.model_copy(
        update={
            "episode_id": "another-attempt",
            "frame_id": "another-attempt:frame:0",
        }
    )
    self_features = [list(row) for row in frame.base_observation.self_features]
    self_features[0][0] += 0.25
    changed_policy_material = frame.model_copy(
        update={
            "base_observation": frame.base_observation.model_copy(
                update={"self_features": tuple(tuple(row) for row in self_features)}
            )
        }
    )

    assert resolved_initial_state_digest_sha256_v2(renamed_attempt) == expected_digest
    assert (
        resolved_initial_state_digest_sha256_v2(changed_policy_material)
        == expected_digest
    )

    health = list(frame.snapshot.current_health)
    health[0] += 1.0
    changed_snapshot = frame.model_copy(
        update={
            "snapshot": frame.snapshot.model_copy(
                update={"current_health": tuple(health)}
            )
        }
    )
    changed_epoch = frame.model_copy(update={"simulator_step_count": 1})
    assert resolved_initial_state_digest_sha256_v2(changed_snapshot) != expected_digest
    assert resolved_initial_state_digest_sha256_v2(changed_epoch) != expected_digest


def test_v2_schedule_and_fixed_slot_templates_enforce_owned_invariants(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
) -> None:
    specification = scenario_artifacts_v2.specification
    schedule = specification.seed_schedule
    schedule_payload = schedule.model_dump(mode="python")

    duplicate_rows = dict(schedule_payload)
    duplicate_rows["realized_seed_protocols"] = (
        schedule.realized_seed_protocols[0],
        schedule.realized_seed_protocols[0],
    )
    with pytest.raises(ValidationError, match="schedule rows must be unique"):
        _redigest_seed_schedule_v2(duplicate_rows)

    mixed_protocols = dict(schedule_payload)
    mixed_protocols["realized_seed_protocols"] = (
        schedule.realized_seed_protocols[0],
        _seed_protocol_with_updates(
            schedule.realized_seed_protocols[1],
            seed_protocol=VersionedIdentityV1(
                identifier="another-split",
                version=1,
            ),
        ),
    )
    with pytest.raises(ValidationError, match="share one seed-protocol identity"):
        _redigest_seed_schedule_v2(mixed_protocols)

    specification_payload = specification.model_dump(mode="python")
    list_roles = dict(specification_payload)
    list_roles["role_template"] = list(specification.role_template)
    with pytest.raises(ValidationError):
        _redigest_specification_v2(list_roles)

    wrong_slot_order = dict(specification_payload)
    wrong_slot_order["roster_template"] = (
        specification.roster_template[1],
        specification.roster_template[0],
        *specification.roster_template[2:],
    )
    with pytest.raises(ValidationError, match="ordered fixed global slots"):
        _redigest_specification_v2(wrong_slot_order)

    inactive_with_role = dict(specification_payload)
    roles = list(specification.role_template)
    inactive_slot = next(
        index
        for index, row in enumerate(specification.roster_template)
        if not row.configured_active
    )
    roles[inactive_slot] = "focal"
    inactive_with_role["role_template"] = tuple(roles)
    with pytest.raises(ValidationError, match=r"inactive.*not_applicable"):
        _redigest_specification_v2(inactive_with_role)

    missing_partner_seed = dict(specification_payload)
    rows = list(schedule.realized_seed_protocols)
    rows[0] = _seed_protocol_with_updates(
        rows[0], cooperative_partner_seed="not_applicable"
    )
    missing_partner_seed["seed_schedule"] = _seed_schedule_v2(tuple(rows))
    with pytest.raises(ValidationError, match="cooperative_partner seed presence"):
        _redigest_specification_v2(missing_partner_seed)

    out_of_range_record = scenario_artifacts_v2.records[0].model_dump(mode="python")
    out_of_range_record["schedule_coordinate"] = len(schedule.realized_seed_protocols)
    with pytest.raises(ValidationError, match="coordinate is out of range"):
        _redigest_record_v2(out_of_range_record)


@pytest.mark.parametrize("root_name", ("seed_schedule", "specification"))
@pytest.mark.parametrize("invalid_version", (True, 1, 3))
def test_v2_nested_roots_require_exact_schema_version_two(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
    root_name: str,
    invalid_version: object,
) -> None:
    if root_name == "seed_schedule":
        payload = scenario_artifacts_v2.specification.seed_schedule.model_dump(
            mode="python"
        )
        payload["schema_version"] = invalid_version
        with pytest.raises(ValidationError):
            _redigest_seed_schedule_v2(payload)
    else:
        payload = scenario_artifacts_v2.specification.model_dump(mode="python")
        payload["schema_version"] = invalid_version
        with pytest.raises(ValidationError):
            _redigest_specification_v2(payload)


@pytest.mark.parametrize(
    ("owned_field", "expected_message"),
    (
        ("resolved_config_digest_sha256", "resolved config"),
        ("roster_template", "roster template"),
        ("role_template", "frozen scenario slot role"),
        ("horizon", "context horizon"),
        ("resolved_initial_state_digest_sha256", "initial-state digest"),
    ),
)
def test_v2_builder_rejects_independent_scenario_owned_join_tampering(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
    owned_field: str,
    expected_message: str,
) -> None:
    specification = scenario_artifacts_v2.specification
    if owned_field == "resolved_config_digest_sha256":
        replacement: object = "4" * 64
    elif owned_field == "roster_template":
        roster = list(specification.roster_template)
        roster[0] = roster[0].model_copy(
            update={"public_agent_id": "replacement-agent-zero"}
        )
        replacement = tuple(roster)
    elif owned_field == "role_template":
        roles = list(specification.role_template)
        roles[1] = "focal"
        replacement = tuple(roles)
    elif owned_field == "horizon":
        replacement = specification.horizon + 1
    else:
        replacement = "4" * 64

    changed_specification = _specification_with_updates_v2(
        specification,
        **{owned_field: replacement},
    )
    context = _context_with_specification_v2(
        scenario_artifacts_v2.trajectories[0].context,
        changed_specification,
        schedule_coordinate=0,
    )
    bundle = _bundle_for_context(scenario_artifacts_v2.trajectories[0], context)

    with pytest.raises(ValueError, match=expected_message):
        _build_record_v2(
            changed_specification,
            bundle,
            schedule_coordinate=0,
        )


@pytest.mark.parametrize(
    ("identity_field", "replacement"),
    (
        ("identifier", "another-layout"),
        ("version", 2),
        ("canonical_digest", "4" * 64),
    ),
)
def test_v2_each_layout_identity_coordinate_is_independently_joined(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
    identity_field: str,
    replacement: object,
) -> None:
    specification = scenario_artifacts_v2.specification
    changed_layout = specification.layout.model_copy(
        update={identity_field: replacement}
    )
    changed_specification = _specification_with_updates_v2(
        specification,
        layout=changed_layout,
    )
    trajectory = scenario_artifacts_v2.trajectories[0]
    context = _context_with_specification_v2(
        trajectory.context,
        changed_specification,
        schedule_coordinate=0,
    )
    bundle = _bundle_for_context(trajectory, context)

    with pytest.raises(ValueError, match="layout identity"):
        _build_record_v2(
            changed_specification,
            bundle,
            schedule_coordinate=0,
        )


@pytest.mark.parametrize(
    ("identity_field", "replacement"),
    (
        ("identifier", "another-authored-state"),
        ("version", 2),
        ("canonical_digest", "4" * 64),
    ),
)
def test_v2_each_authored_initial_condition_identity_coordinate_is_bound(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
    identity_field: str,
    replacement: object,
) -> None:
    authored_identity = (
        scenario_artifacts_v2.specification.authored_initial_condition.model_copy(
            update={identity_field: replacement}
        )
    )
    changed_specification = _specification_with_updates_v2(
        scenario_artifacts_v2.specification,
        authored_initial_condition=authored_identity,
    )
    bundle = scenario_artifacts_v2.bundles[0]

    with pytest.raises(ValueError, match="context scenario identity"):
        _build_record_v2(changed_specification, bundle, schedule_coordinate=0)


@pytest.mark.parametrize(
    ("identity_field", "replacement"),
    (
        ("identifier", "another-scenario"),
        ("version", 2),
        ("canonical_digest", "4" * 64),
    ),
)
def test_v2_resealed_context_scenario_identity_tampering_is_rejected(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
    identity_field: str,
    replacement: object,
) -> None:
    specification = scenario_artifacts_v2.specification
    trajectory = scenario_artifacts_v2.trajectories[0]
    context = scenario_artifacts_v2.contexts[0]
    identity_payload = context.identity.model_dump(mode="python")
    scenario_identity = context.identity.scenario
    assert scenario_identity is not None
    identity_payload["scenario"] = scenario_identity.model_copy(
        update={identity_field: replacement}
    )
    context_payload = context.model_dump(mode="python")
    context_payload["identity"] = EvaluationEpisodeIdentityV1.model_validate(
        identity_payload
    )
    changed_context = EvaluationEpisodeContextV1.model_validate(context_payload)
    changed_bundle = _bundle_for_context(trajectory, changed_context)

    with pytest.raises(ValueError, match="context scenario identity"):
        _build_record_v2(specification, changed_bundle, schedule_coordinate=0)


@pytest.mark.parametrize(
    ("scenario_field", "replacement"),
    (("scenario_id", "another-scenario"), ("scenario_version", 2)),
)
def test_v2_explicit_scenario_identity_coordinates_are_frozen_by_context(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
    scenario_field: str,
    replacement: object,
) -> None:
    changed_specification = _specification_with_updates_v2(
        scenario_artifacts_v2.specification,
        **{scenario_field: replacement},
    )
    bundle = scenario_artifacts_v2.bundles[0]

    with pytest.raises(ValueError, match="context scenario identity"):
        _build_record_v2(changed_specification, bundle, schedule_coordinate=0)


@pytest.mark.parametrize("roster_mutation", ("class_id", "configured_active"))
def test_v2_roster_class_and_active_flag_are_independently_joined(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
    roster_mutation: str,
) -> None:
    specification = scenario_artifacts_v2.specification
    roster = list(specification.roster_template)
    roles = list(specification.role_template)
    if roster_mutation == "class_id":
        roster[0] = roster[0].model_copy(
            update={"class_id": 2 if roster[0].class_id != 2 else 3}
        )
    else:
        roster[1] = roster[1].model_copy(
            update={
                "configured_team_id": 0,
                "class_id": 0,
                "configured_active": False,
            }
        )
        roles[1] = "not_applicable"
    changed_specification = _specification_with_updates_v2(
        specification,
        roster_template=tuple(roster),
        role_template=tuple(roles),
    )
    trajectory = scenario_artifacts_v2.trajectories[0]
    context = _context_with_specification_v2(
        trajectory.context,
        changed_specification,
        schedule_coordinate=0,
    )
    bundle = _bundle_for_context(trajectory, context)

    with pytest.raises(ValueError, match="roster template"):
        _build_record_v2(
            changed_specification,
            bundle,
            schedule_coordinate=0,
        )


@pytest.mark.parametrize(
    ("schedule_field", "replacement"),
    (("schedule_id", "another-schedule"), ("schedule_version", 2)),
)
def test_v2_seed_schedule_identity_and_version_are_bound_by_scenario_identity(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
    schedule_field: str,
    replacement: object,
) -> None:
    specification = scenario_artifacts_v2.specification
    schedule_payload = specification.seed_schedule.model_dump(mode="python")
    schedule_payload[schedule_field] = replacement
    changed_schedule = _redigest_seed_schedule_v2(schedule_payload)
    changed_specification = _specification_with_updates_v2(
        specification,
        seed_schedule=changed_schedule,
    )
    bundle = scenario_artifacts_v2.bundles[0]

    with pytest.raises(ValueError, match="context scenario identity"):
        _build_record_v2(changed_specification, bundle, schedule_coordinate=0)


@pytest.mark.parametrize(
    "seed_field",
    (
        "root_seed",
        "episode_seed",
        "layout_seed",
        "environment_seed",
        "focal_policy_seed",
        "evaluation_seed",
        "cooperative_partner_seed",
        "adversarial_opponent_seed",
        "scenario_seed",
    ),
)
def test_v2_schedule_coordinate_rejects_every_changed_seed_category(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
    seed_field: str,
) -> None:
    specification = scenario_artifacts_v2.specification
    original_row = specification.seed_schedule.realized_seed_protocols[0]
    original_value = getattr(original_row, seed_field)
    assert type(original_value) is int
    rows = list(specification.seed_schedule.realized_seed_protocols)
    rows[0] = _seed_protocol_with_updates(
        original_row,
        **{seed_field: original_value + 1},
    )
    changed_schedule = _seed_schedule_v2(tuple(rows))
    changed_specification = _specification_with_updates_v2(
        specification,
        seed_schedule=changed_schedule,
    )
    context = _context_with_specification_v2(
        scenario_artifacts_v2.trajectories[0].context,
        changed_specification,
        schedule_coordinate=0,
    )
    context_payload = context.model_dump(mode="python")
    context_payload["seed_protocol"] = original_row
    context = EvaluationEpisodeContextV1.model_validate(context_payload)
    bundle = _bundle_for_context(scenario_artifacts_v2.trajectories[0], context)

    with pytest.raises(ValueError, match="select the context seed protocol"):
        _build_record_v2(
            changed_specification,
            bundle,
            schedule_coordinate=0,
        )


def test_v2_schedule_rejects_protocol_identity_order_and_coordinate_aliasing(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
) -> None:
    specification = scenario_artifacts_v2.specification
    rows = specification.seed_schedule.realized_seed_protocols
    changed_protocol_rows = tuple(
        _seed_protocol_with_updates(
            row,
            seed_protocol=VersionedIdentityV1(
                identifier="another-split",
                version=1,
            ),
        )
        for row in rows
    )
    for changed_schedule in (
        _seed_schedule_v2(changed_protocol_rows),
        _seed_schedule_v2(tuple(reversed(rows))),
    ):
        changed_specification = _specification_with_updates_v2(
            specification,
            seed_schedule=changed_schedule,
        )
        context = _context_with_specification_v2(
            scenario_artifacts_v2.trajectories[0].context,
            changed_specification,
            schedule_coordinate=0,
        )
        context_payload = context.model_dump(mode="python")
        context_payload["seed_protocol"] = rows[0]
        context = EvaluationEpisodeContextV1.model_validate(context_payload)
        bundle = _bundle_for_context(scenario_artifacts_v2.trajectories[0], context)
        with pytest.raises(ValueError, match="select the context seed protocol"):
            _build_record_v2(
                changed_specification,
                bundle,
                schedule_coordinate=0,
            )

    record_payload = scenario_artifacts_v2.records[0].model_dump(mode="python")
    record_payload["schedule_coordinate"] = 1
    aliased_coordinate = _redigest_record_v2(record_payload)
    bundle = scenario_artifacts_v2.bundles[0]
    with pytest.raises(ValueError, match="select the context seed protocol"):
        validate_scenario_evaluation_record_v2(
            aliased_coordinate,
            bundle.replay,
            bundle.metric_report_artifact,
        )


def test_v2_policy_identity_is_replaceable_but_fixed_slot_role_is_not(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
) -> None:
    specification = scenario_artifacts_v2.specification
    trajectory = scenario_artifacts_v2.trajectories[0]
    replacements = tuple(
        assignment.model_copy(
            update={
                "policy_id": f"replacement-policy-{assignment.global_slot}",
                "policy_content_digest": "4" * 64,
                "checkpoint_digest": "5" * 64,
            }
        )
        if isinstance(assignment, AssignedPolicySlotV1)
        else assignment
        for assignment in trajectory.context.policy_assignments
    )
    replacement_context = _context_with_specification_v2(
        trajectory.context,
        specification,
        schedule_coordinate=0,
        policy_assignments=replacements,
    )
    replacement_bundle = _bundle_for_context(trajectory, replacement_context)
    replacement_record = _build_record_v2(
        specification,
        replacement_bundle,
        schedule_coordinate=0,
    )
    validate_official_scenario_evaluation_record_v2(
        replacement_record,
        replacement_bundle.replay,
        replacement_bundle.metric_report_artifact,
    )

    wrong_roles = list(trajectory.context.policy_assignments)
    first = wrong_roles[0]
    second = wrong_roles[1]
    assert isinstance(first, AssignedPolicySlotV1)
    assert isinstance(second, AssignedPolicySlotV1)
    wrong_roles[0] = first.model_copy(update={"evaluation_role": "cooperative_partner"})
    wrong_roles[1] = second.model_copy(update={"evaluation_role": "focal"})
    wrong_role_context = _context_with_specification_v2(
        trajectory.context,
        specification,
        schedule_coordinate=0,
        policy_assignments=tuple(wrong_roles),
    )
    wrong_role_bundle = _bundle_for_context(trajectory, wrong_role_context)
    with pytest.raises(ValueError, match="frozen scenario slot role"):
        _build_record_v2(
            specification,
            wrong_role_bundle,
            schedule_coordinate=0,
        )


def test_v2_validation_rejects_resealed_full_initial_frame_digest_tampering(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
) -> None:
    record_payload = scenario_artifacts_v2.records[0].model_dump(mode="python")
    record_payload["realized_initial_frame_digest_sha256"] = "4" * 64
    changed_record = _redigest_record_v2(record_payload)
    bundle = scenario_artifacts_v2.bundles[0]

    with pytest.raises(ValueError, match="initial-frame digest"):
        validate_scenario_evaluation_record_v2(
            changed_record,
            bundle.replay,
            bundle.metric_report_artifact,
        )


@pytest.mark.parametrize(
    ("hostile_case", "expected_message"),
    (
        ("non_product_movement", "product ordinary_movement_distance_scale"),
        ("int32_overflow", "representable as int32"),
        ("invalid_geometry", "outside radius-adjusted map bounds"),
        ("invalid_spawn_speed", "spawn_shield_movement_speed"),
        ("float32_narrowing", "losslessly representable as float32"),
    ),
)
def test_v2_pure_validation_accepts_wire_contexts_that_official_gate_rejects(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
    hostile_case: str,
    expected_message: str,
) -> None:
    trajectory = scenario_artifacts_v2.trajectories[0]
    resolved = trajectory.context.resolved_env_config
    if hostile_case == "non_product_movement":
        changed_resolved = _resolved_config_with_updates(
            resolved,
            ordinary_movement_distance_scale=0.5,
        )
    elif hostile_case == "int32_overflow":
        changed_resolved = _resolved_config_with_updates(
            resolved,
            spawn_shield_duration_steps=2**31,
        )
    elif hostile_case == "invalid_geometry":
        changed_resolved = _resolved_config_with_updates(resolved, map_width=5.0)
    elif hostile_case == "invalid_spawn_speed":
        changed_resolved = _resolved_config_with_updates(
            resolved,
            spawn_shield_movement_speed=0.0,
        )
    else:
        pads = [
            [list(center) for center in team]
            for team in resolved.team_spawn_pad_positions
        ]
        pads[0][0][0] = 1.1
        changed_resolved = _resolved_config_with_updates(
            resolved,
            team_spawn_pad_positions=tuple(
                tuple(tuple(center) for center in team) for team in pads
            ),
        )

    context_payload = trajectory.context.model_dump(mode="python")
    context_payload["resolved_env_config"] = changed_resolved
    changed_context = EvaluationEpisodeContextV1.model_validate(context_payload)
    _, bundle, record = _pure_v2_artifacts_for_context(
        trajectory,
        changed_context,
        scenario_artifacts_v2.specification.seed_schedule,
    )

    with pytest.raises(ValueError, match=expected_message):
        validate_official_scenario_evaluation_record_v2(
            record,
            bundle.replay,
            bundle.metric_report_artifact,
        )


@pytest.mark.parametrize(
    ("source_slot", "promoted_slot"),
    ((1, 4), (5, 9)),
)
def test_v2_official_gate_rejects_noncontiguous_active_prefix_in_each_team(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
    source_slot: int,
    promoted_slot: int,
) -> None:
    source_trajectory = scenario_artifacts_v2.trajectories[0]
    source_context = source_trajectory.context
    roster = list(source_context.roster)
    source_roster = roster[source_slot]
    roster[promoted_slot] = roster[promoted_slot].model_copy(
        update={
            "configured_team_id": source_roster.configured_team_id,
            "class_id": source_roster.class_id,
            "configured_active": True,
        }
    )
    mechanics = list(source_context.resolved_env_config.slot_mechanics)
    mechanics[promoted_slot] = mechanics[source_slot].model_copy(
        update={"global_slot": promoted_slot}
    )
    resolved = _resolved_config_with_updates(
        source_context.resolved_env_config,
        slot_mechanics=tuple(mechanics),
    )
    policies = list(source_context.policy_assignments)
    source_policy = policies[source_slot]
    assert isinstance(source_policy, AssignedPolicySlotV1)
    policies[promoted_slot] = source_policy.model_copy(
        update={
            "global_slot": promoted_slot,
            "policy_id": f"hostile-active-hole-policy-{promoted_slot}",
            "training_run_id": f"hostile-active-hole-run-{promoted_slot}",
            "population_member_id": f"hostile-active-hole-member-{promoted_slot}",
        }
    )
    context_payload = source_context.model_dump(mode="python")
    context_payload.update(
        {
            "resolved_env_config": resolved,
            "roster": tuple(roster),
            "policy_assignments": tuple(policies),
        }
    )
    changed_context = EvaluationEpisodeContextV1.model_validate(context_payload)
    changed_frames: list[EvaluationFrameV1] = []
    for frame in source_trajectory.frames:
        context_features = [
            list(row) for row in frame.base_observation.context_features
        ]
        context_features[promoted_slot] = list(context_features[source_slot])
        changed_frames.append(
            frame.model_copy(
                update={
                    "base_observation": frame.base_observation.model_copy(
                        update={
                            "context_features": tuple(
                                tuple(row) for row in context_features
                            )
                        }
                    )
                }
            )
        )
    trajectory = CapturedEvaluationTrajectory(
        context=source_context,
        frames=tuple(changed_frames),
        transitions=source_trajectory.transitions,
    )
    _, bundle, record = _pure_v2_artifacts_for_context(
        trajectory,
        changed_context,
        scenario_artifacts_v2.specification.seed_schedule,
    )

    with pytest.raises(ValueError, match="contiguous active prefix"):
        validate_official_scenario_evaluation_record_v2(
            record,
            bundle.replay,
            bundle.metric_report_artifact,
        )


def test_v1_context_transport_rejects_roster_mechanics_disagreement_before_v2_gate(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
) -> None:
    context = scenario_artifacts_v2.contexts[0]
    roster = list(context.roster)
    roster[0] = roster[0].model_copy(
        update={"class_id": 2 if roster[0].class_id != 2 else 3}
    )
    context_payload = context.model_dump(mode="python")
    context_payload["roster"] = tuple(roster)

    with pytest.raises(
        ValidationError,
        match="roster profile disagrees with mechanics catalog",
    ):
        EvaluationEpisodeContextV1.model_validate(context_payload)


def test_v2_official_gate_rejects_aligned_wire_catalog_profile_drift(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = scenario_artifacts_v2.trajectories[0]
    source_context = trajectory.context
    target_class_id = source_context.roster[0].class_id

    class_mechanics = list(source_context.static_mechanics_catalog.class_mechanics)
    class_mechanics[target_class_id] = class_mechanics[target_class_id].model_copy(
        update={"body_radius": 0.75}
    )
    changed_catalog = _catalog_with_updates(
        source_context.static_mechanics_catalog,
        class_mechanics=tuple(class_mechanics),
    )

    slot_mechanics = list(source_context.resolved_env_config.slot_mechanics)
    for global_slot, roster_row in enumerate(source_context.roster):
        if roster_row.class_id == target_class_id:
            slot_mechanics[global_slot] = slot_mechanics[global_slot].model_copy(
                update={"body_radius": 0.75}
            )
    changed_resolved = _resolved_config_with_updates(
        source_context.resolved_env_config,
        slot_mechanics=tuple(slot_mechanics),
    )
    context_payload = source_context.model_dump(mode="python")
    context_payload.update(
        {
            "static_mechanics_catalog": changed_catalog,
            "resolved_env_config": changed_resolved,
        }
    )
    changed_context = EvaluationEpisodeContextV1.model_validate(context_payload)
    _, bundle, record = _pure_v2_artifacts_for_context(
        trajectory,
        changed_context,
        scenario_artifacts_v2.specification.seed_schedule,
    )

    monkeypatch.setattr(
        catalog_module,
        "build_static_mechanics_catalog_v1",
        lambda: changed_catalog,
    )
    with pytest.raises(
        ValueError,
        match=r"agent_profile\.agent_radii must match the resolved class catalog",
    ):
        validate_official_scenario_evaluation_record_v2(
            record,
            bundle.replay,
            bundle.metric_report_artifact,
        )


def test_v2_official_gate_rejects_carried_catalog_that_disagrees_with_live_code(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
) -> None:
    trajectory = scenario_artifacts_v2.trajectories[0]
    source_context = trajectory.context
    changed_catalog = _catalog_with_updates(
        source_context.static_mechanics_catalog,
        global_slow_floor=(
            0.25
            if source_context.static_mechanics_catalog.global_slow_floor != 0.25
            else 0.5
        ),
    )
    context_payload = source_context.model_dump(mode="python")
    context_payload["static_mechanics_catalog"] = changed_catalog
    changed_context = EvaluationEpisodeContextV1.model_validate(context_payload)
    _, bundle, record = _pure_v2_artifacts_for_context(
        trajectory,
        changed_context,
        scenario_artifacts_v2.specification.seed_schedule,
    )

    with pytest.raises(ValueError, match="disagrees with live catalog"):
        validate_official_scenario_evaluation_record_v2(
            record,
            bundle.replay,
            bundle.metric_report_artifact,
        )


def test_v2_official_gate_accepts_non_float32_exact_python_scalar_dimensions(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
) -> None:
    config = evaluation_env_config()._replace(map_width=20.1)
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        capture_profile="scenario_metric_complete",
        expected_horizon=1,
        with_scenario=True,
        episode_id="episode-v2-scalar-map-width",
        config=config,
    )
    specification = _specification_v2(
        trajectory.context,
        trajectory.frames[0],
        scenario_artifacts_v2.specification.seed_schedule,
    )
    context = _context_with_specification_v2(
        trajectory.context,
        specification,
        schedule_coordinate=0,
    )
    bundle = _bundle_for_context(trajectory, context)

    record = _build_record_v2(specification, bundle, schedule_coordinate=0)
    validate_official_scenario_evaluation_record_v2(
        record,
        bundle.replay,
        bundle.metric_report_artifact,
    )


def test_v2_official_gate_rejects_wire_valid_but_invalid_curated_state(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
) -> None:
    source = captured_evaluation_trajectory(
        transition_count=0,
        capture_profile="scenario_metric_complete",
        expected_horizon=1,
        with_scenario=True,
        episode_id="episode-v2-hostile-state",
    )
    frame = source.frames[0]
    health = list(frame.snapshot.current_health)
    health[0] = (
        source.context.resolved_env_config.slot_mechanics[0].maximum_health + 1.0
    )
    changed_frame = frame.model_copy(
        update={
            "snapshot": frame.snapshot.model_copy(
                update={"current_health": tuple(health)}
            )
        }
    )
    trajectory = CapturedEvaluationTrajectory(
        context=source.context,
        frames=(changed_frame,),
        transitions=(),
    )
    specification = _specification_v2(
        trajectory.context,
        changed_frame,
        scenario_artifacts_v2.specification.seed_schedule,
    )
    context = _context_with_specification_v2(
        trajectory.context,
        specification,
        schedule_coordinate=0,
    )
    bundle = _bundle_for_context(trajectory, context, partial=True)
    measurement_results = (
        ScenarioMeasurementResultV1(
            measurement_id="test.endpoint",
            measurement_version=1,
            result_status="insufficient_data",
            endpoint_observation_status="unavailable",
            value=None,
            reason="complete episode required",
        ),
        _defined_measurement_results()[1],
    )
    violation_results = (
        ScenarioViolationResultV1(
            violation_id="test.role_violation",
            violation_version=1,
            result_status="insufficient_data",
            endpoint_observation_status="unavailable",
            value=None,
            reason="complete episode required",
        ),
    )
    record = _record_without_official_gate_v2(
        specification,
        bundle,
        schedule_coordinate=0,
        measurement_results=measurement_results,
        violation_results=violation_results,
        predicate_result=ScenarioPredicateResultV1(
            predicate_id="test.success",
            predicate_version=1,
            status="unavailable",
            reason="complete episode required",
        ),
    )
    validate_scenario_evaluation_record_v2(
        record,
        bundle.replay,
        bundle.metric_report_artifact,
    )
    with pytest.raises(ValueError, match="current_health must not exceed max_health"):
        validate_official_scenario_evaluation_record_v2(
            record,
            bundle.replay,
            bundle.metric_report_artifact,
        )


def test_v2_build_and_each_validation_gate_scan_replay_once(
    scenario_artifacts_v2: _ScenarioArtifactsV2,
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
    bundle = scenario_artifacts_v2.bundles[0]
    rebuilt = _build_record_v2(
        scenario_artifacts_v2.specification,
        bundle,
        schedule_coordinate=0,
    )
    assert semantic_passes == 1

    semantic_passes = 0
    validate_scenario_evaluation_record_v2(
        rebuilt,
        bundle.replay,
        bundle.metric_report_artifact,
    )
    assert semantic_passes == 1

    semantic_passes = 0
    validate_official_scenario_evaluation_record_v2(
        rebuilt,
        bundle.replay,
        bundle.metric_report_artifact,
    )
    assert semantic_passes == 1


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
