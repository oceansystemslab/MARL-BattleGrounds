"""Approved eight-scenario definitions joined to current replay evidence.

The endpoint is observed terminal Team A reward, not an inferred tactical skill.
The small qualification schedule is plumbing evidence, not manuscript sampling.
"""

from __future__ import annotations

from marl_battlegrounds._tdm_assets import scenario_content
from marl_battlegrounds.evaluation.catalog import build_roster_v1
from marl_battlegrounds.evaluation.models import (
    ContentAddressedIdentityV1,
    EvaluationFrame,
    EvaluationSeedProtocolV2,
    VersionedIdentityV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3
from marl_battlegrounds.evaluation.scenario import (
    ResolvedScenarioSpecificationV3,
    ScenarioEvaluationRecordV4,
    ScenarioMeasurementDefinitionV1,
    ScenarioMeasurementResultV1,
    ScenarioPredicateResultV1,
    ScenarioScalarValueV1,
    ScenarioSeedScheduleV3,
    build_scenario_evaluation_record_v4,
    resolved_initial_state_digest_sha256,
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


def build_tdm_qualification_seed_schedule() -> ScenarioSeedScheduleV3:
    """Two coordinates in the shared executor's actual fold-in RNG protocol.

    Named substream keys are derived by the executor, not independent seeds.
    Their scalar seed fields therefore remain unrecorded. No cooperative role
    exists in these scenario definitions; Team A is the focal policy team.
    """
    rows = tuple(
        EvaluationSeedProtocolV2(
            seed_protocol=VersionedIdentityV1(
                identifier="episode-fold-in-v1", version=1
            ),
            root_seed=0,
            episode_seed=coordinate,
            cooperative_partner_seed="not_applicable",
        )
        for coordinate in (0, 1)
    )
    payload: dict[str, object] = {
        "schema_id": "marl_battlegrounds.evaluation.scenario_seed_schedule",
        "schema_version": 3,
        "schedule_id": "tdm-closeout-plumbing",
        "schedule_version": 2,
        "realized_seed_protocols": rows,
    }
    return ScenarioSeedScheduleV3.model_validate(
        {**payload, "canonical_digest_sha256": canonical_digest_sha256(payload)}
    )


def build_tdm_scenario_specification(
    scenario_id: int,
    seed_schedule: ScenarioSeedScheduleV3,
    *,
    initial_frame: EvaluationFrame | None = None,
) -> ResolvedScenarioSpecificationV3:
    """Bind approved content to an explicit matched schedule and exact frame zero."""
    scenario = load_tdm_scenario(scenario_id)
    content = scenario_content(scenario.info)
    if initial_frame is not None and (
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
        "schema_version": 3,
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
            resolved_initial_state_digest_sha256(
                content.step_count, content.initial_snapshot
            )
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
    # Include defaults in the content digest, as required by the scenario authority.
    payload.update(parameters=(), secondary_measurements=(), violations=())
    return ResolvedScenarioSpecificationV3.model_validate(
        {**payload, "canonical_digest_sha256": canonical_digest_sha256(payload)}
    )


def build_tdm_scenario_evaluation_record(
    scenario_id: int,
    specification: ResolvedScenarioSpecificationV3,
    replay: ReplayArtifactV3,
    *,
    schedule_coordinate: int,
) -> ScenarioEvaluationRecordV4:
    """Record the frozen reward endpoint and check its official replay joins."""
    expected = build_tdm_scenario_specification(
        scenario_id, specification.seed_schedule, initial_frame=replay.frames[0]
    )
    if expected != specification:
        raise ValueError(
            "scenario specification disagrees with approved TDM definition"
        )
    pressure = expected.pressure_protocol
    assert pressure is not None
    identities = {row.name: row.value for row in replay.header.context.aggregation_keys}
    recorded_pressure = identities.get("team_b_controller_identity")
    if (
        recorded_pressure is None
        or ContentAddressedIdentityV1.model_validate_json(recorded_pressure) != pressure
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
    return build_scenario_evaluation_record_v4(
        specification,
        replay,
        schedule_coordinate=schedule_coordinate,
        measurement_results=(measurement,),
        violation_results=(),
        predicate_result=predicate,
    )
