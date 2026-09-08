"""Approved eight-scenario definitions joined to the existing V2 evidence contract.

The endpoint is observed terminal Team A reward, not an inferred tactical skill.
The small qualification schedule is plumbing evidence, not manuscript sampling.
"""

from __future__ import annotations

from marl_battlegrounds._tdm_assets import scenario_content
from marl_battlegrounds.evaluation.catalog import build_roster_v1
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV1,
    ContentAddressedIdentityV1,
    EvaluationFrameV1,
    EvaluationSeedProtocolV1,
    VersionedIdentityV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.replay import (
    EvaluationMetricReportArtifactV1,
    ReplayArtifactV1,
)
from marl_battlegrounds.evaluation.scenario import (
    ResolvedScenarioSpecificationV2,
    ScenarioEvaluationRecordV2,
    ScenarioMeasurementDefinitionV1,
    ScenarioMeasurementResultV1,
    ScenarioPredicateResultV1,
    ScenarioScalarValueV1,
    ScenarioSeedScheduleV2,
    build_scenario_evaluation_record_v2,
    resolved_initial_state_digest_sha256_v2,
)
from marl_battlegrounds.policies.reactive_tdm_alpha import (
    reactive_tdm_alpha_controller_descriptor,
)
from marl_battlegrounds.policies.reactive_tdm_beta import (
    reactive_tdm_beta_controller_descriptor,
)
from marl_battlegrounds.tasks import load_tdm_scenario

TDM_SCENARIO_PUBLIC_AGENT_IDS = tuple(
    f"agent-{team}{slot}" for team in ("a", "b") for slot in range(1, 6)
)
TDM_BETA_SCENARIO_IDS = (3, 5, 8)
_HYPOTHESES = (
    "Coordinate a last heal, healer disable, cover and two finishing attacks.",
    "Coordinate protection and a timed rescue heal while preserving the healer.",
    "Sustain a moving body screen while a vulnerable Hunter kites and attacks.",
    "Use shared sightings, displacement and kill order to coordinate an ambush.",
    "Coordinate a moving screen, healing allocation, Burst and finishing attacks.",
    "Preserve a trapped opponent while coordinating healed bait and a hidden finisher.",
    "Coordinate protective control, healing priorities and a delayed Burst.",
    "Protect a healer and preserve enemy control until allied Ultimates are ready.",
)


def tdm_scenario_pressure_identity(scenario_id: int) -> ContentAddressedIdentityV1:
    """Freeze the current ALPHA/BETA rule descriptor selected by approved Notes."""
    load_tdm_scenario(scenario_id)
    descriptor = (
        reactive_tdm_beta_controller_descriptor()
        if scenario_id in TDM_BETA_SCENARIO_IDS
        else reactive_tdm_alpha_controller_descriptor()
    )
    return ContentAddressedIdentityV1.model_validate(
        {
            "identifier": descriptor["policy_id"],
            "version": descriptor["version"],
            "canonical_digest": canonical_digest_sha256(descriptor),
        }
    )


def build_tdm_qualification_seed_schedule() -> ScenarioSeedScheduleV2:
    """Two fixed coordinates for deterministic retry and matched plumbing checks.

    The reference rollout consumes environment_seed as its root and splits epoch
    environment/actor keys internally. Both diagnostic controllers ignore actor
    keys; the policy seed fields identify that same rollout root, not independent
    streams that were never consumed. There is no cooperative-partner role.
    """
    rows = tuple(
        EvaluationSeedProtocolV1(
            seed_protocol=VersionedIdentityV1(
                identifier="tdm-qualification-reference-rollout", version=1
            ),
            root_seed=seed,
            episode_seed=seed,
            layout_seed=seed,
            environment_seed=seed,
            focal_policy_seed=seed,
            evaluation_seed=seed,
            cooperative_partner_seed="not_applicable",
            adversarial_opponent_seed=seed,
            scenario_seed=seed,
        )
        for seed in (0, 1)
    )
    payload: dict[str, object] = {
        "schema_id": "marl_battlegrounds.evaluation.scenario_seed_schedule",
        "schema_version": 2,
        "schedule_id": "tdm-closeout-plumbing",
        "schedule_version": 1,
        "realized_seed_protocols": rows,
    }
    return ScenarioSeedScheduleV2.model_validate(
        {**payload, "canonical_digest_sha256": canonical_digest_sha256(payload)}
    )


def build_tdm_scenario_specification(
    scenario_id: int,
    initial_frame: EvaluationFrameV1,
    seed_schedule: ScenarioSeedScheduleV2,
) -> ResolvedScenarioSpecificationV2:
    """Bind approved content to an explicit matched schedule and exact frame zero."""
    scenario = load_tdm_scenario(scenario_id)
    content = scenario_content(scenario.info)
    if (
        initial_frame.frame_index != 0
        or initial_frame.simulator_step_count != content.step_count
        or initial_frame.snapshot != content.initial_snapshot
    ):
        raise ValueError(
            "scenario initial frame must equal its approved packaged state"
        )
    roster = build_roster_v1(scenario.config, TDM_SCENARIO_PUBLIC_AGENT_IDS)
    payload: dict[str, object] = {
        "schema_id": "marl_battlegrounds.evaluation.resolved_scenario_specification",
        "schema_version": 2,
        "scenario_id": f"tdm-scenario-{scenario_id}",
        "scenario_version": 1,
        "classification": "official",
        "hypothesis": _HYPOTHESES[scenario_id - 1],
        "layout": ContentAddressedIdentityV1(
            identifier=f"tdm-scenario-{scenario_id}-map",
            version=1,
            canonical_digest=scenario.info.map_semantic_digest,
        ),
        "authored_initial_condition": ContentAddressedIdentityV1(
            identifier=scenario.info.source.asset_id,
            version=scenario.info.approved_source.revision,
            canonical_digest=scenario.info.source.semantic_digest,
        ),
        "resolved_initial_state_digest_sha256": (
            resolved_initial_state_digest_sha256_v2(initial_frame)
        ),
        "resolved_config_digest_sha256": scenario.info.resolved_configuration_digest,
        "roster_template": roster,
        "role_template": tuple(
            "not_applicable"
            if not row.configured_active
            else "focal"
            if row.configured_team_id == 1
            else "adversarial_opponent"
            for row in roster
        ),
        "seed_schedule": seed_schedule,
        "horizon": scenario.info.horizon,
        "pressure_protocol": tdm_scenario_pressure_identity(scenario_id),
        "primary_measurement": ScenarioMeasurementDefinitionV1(
            measurement_id="tdm.terminal_team_a_reward",
            measurement_version=1,
            role="primary",
            value_type="scalar",
            units="reward",
            completion_scope="complete_episode",
            supports_right_censoring=False,
        ),
        "success_predicate": VersionedIdentityV1(
            identifier="tdm.terminal_team_a_win", version=1
        ),
        "completion_policy": VersionedIdentityV1(
            identifier="tdm.task_terminal_or_declared_horizon", version=1
        ),
        "partial_result_policy": VersionedIdentityV1(
            identifier="tdm.incomplete_endpoint_unavailable", version=1
        ),
    }
    # Include defaults in the content digest, as required by the V2 authority.
    payload.update(parameters=(), secondary_measurements=(), violations=())
    return ResolvedScenarioSpecificationV2.model_validate(
        {**payload, "canonical_digest_sha256": canonical_digest_sha256(payload)}
    )


def build_tdm_scenario_evaluation_record(
    scenario_id: int,
    specification: ResolvedScenarioSpecificationV2,
    replay: ReplayArtifactV1,
    metric_report: EvaluationMetricReportArtifactV1,
    *,
    schedule_coordinate: int,
) -> ScenarioEvaluationRecordV2:
    """Materialize the frozen reward endpoint and require all official V2 joins."""
    expected = build_tdm_scenario_specification(
        scenario_id, replay.frames[0], specification.seed_schedule
    )
    if expected != specification:
        raise ValueError(
            "scenario specification disagrees with approved TDM definition"
        )
    pressure = expected.pressure_protocol
    assert pressure is not None
    for assignment in replay.header.context.policy_assignments:
        if (
            isinstance(assignment, AssignedPolicySlotV1)
            and assignment.evaluation_role == "adversarial_opponent"
            and (
                assignment.policy_content_digest != pressure.canonical_digest
                or assignment.algorithm_id != pressure.identifier
            )
        ):
            raise ValueError("scenario opponent must match its frozen pressure rules")
    complete = replay.completion.completion_state == "complete"
    terminal = replay.transitions[-1] if replay.transitions else None
    available = (
        complete
        and terminal is not None
        and (terminal.terminated or terminal.truncated)
    )
    reward: float | None = None
    if available and terminal is not None:
        rewards = {
            terminal.canonical_reward_by_agent[row.global_slot]
            for row in specification.roster_template
            if row.configured_active and row.configured_team_id == 1
        }
        if len(rewards) != 1 or not rewards <= {-1.0, 0.0, 1.0}:
            raise ValueError("terminal Team A reward must be one canonical team value")
        reward = rewards.pop()
    measurement = ScenarioMeasurementResultV1(
        measurement_id=specification.primary_measurement.measurement_id,
        measurement_version=1,
        result_status="defined" if reward is not None else "unavailable",
        endpoint_observation_status="observed" if reward is not None else "unavailable",
        value=ScenarioScalarValueV1(value=reward) if reward is not None else None,
        reason=None
        if reward is not None
        else "Complete terminal TDM reward unavailable.",
    )
    predicate = ScenarioPredicateResultV1(
        predicate_id=specification.success_predicate.identifier,
        predicate_version=1,
        status=(
            "unavailable"
            if reward is None
            else "satisfied"
            if reward == 1.0
            else "not_satisfied"
        ),
        reason=None
        if reward is not None
        else "Complete terminal TDM reward unavailable.",
    )
    return build_scenario_evaluation_record_v2(
        specification,
        replay,
        metric_report,
        schedule_coordinate=schedule_coordinate,
        measurement_results=(measurement,),
        violation_results=(),
        predicate_result=predicate,
    )
