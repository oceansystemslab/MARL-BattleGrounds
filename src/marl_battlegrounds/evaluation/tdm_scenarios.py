"""Bind the eight packaged TDM scenarios to reproducible evaluation records.

These host helpers record the approved layout, exact initial state, fixed opposing
controller, and seed schedule. The measured endpoint is terminal Team A reward.
A win supports that endpoint; it does not by itself prove the proposed tactical
skill was learned. The two-coordinate qualification schedule checks recording
plumbing and is not a manuscript sampling plan.
"""

from __future__ import annotations

from marl_battlegrounds._tdm_assets import ScenarioContent, scenario_content
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
from marl_battlegrounds.tasks import TDMScenario, load_tdm_scenario

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
    """Identify the fixed opposing controller for one packaged scenario.

    Parameters
    ----------
    scenario_id : int
        Packaged scenario number, 1 through 8.

    Returns
    -------
    ContentAddressedIdentityV1
        Controller identifier, version, and SHA-256 digest of its complete current
        rule descriptor. Scenarios 3, 5, and 8 use BETA; the others use ALPHA.

    Raises
    ------
    ValueError
        The scenario number is not supported or its packaged content is invalid.

    Notes
    -----
    Loads and validates scenario content on the host. The identity describes
    controller rules, not learned weights or a claim about their performance.
    """
    load_tdm_scenario(scenario_id)
    return _pressure_identity(scenario_id)


def _pressure_identity(scenario_id: int) -> ContentAddressedIdentityV1:
    """Identify pressure rules for an already validated packaged scenario ID.

    scenario_id must be 1 through 8, checked by the owning scenario loader.
    Return the current ALPHA/BETA descriptor identity without loading or running
    the scenario again. Only validated preparation paths call this helper.
    """
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
    """Build the small seed schedule used to check scenario recording.

    Returns
    -------
    ScenarioSeedScheduleV3
        Version 3 schedule named tdm-closeout-plumbing, schedule version 2,
        with root seed 0 and episode coordinates 0 and 1. Its digest includes the
        complete schedule payload.

    Notes
    -----
    The executor derives named substream keys through episode-fold-in-v1.
    These keys are not separate scalar seeds, so those seed fields remain
    unrecorded. Team A is focal; there is no cooperative-partner role.
    The two coordinates test execution and artifact joins, not statistical power.
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
    """Describe one approved scenario and the exact experiment to record.

    Parameters
    ----------
    scenario_id : int
        Packaged scenario number, 1 through 8.
    seed_schedule : ScenarioSeedScheduleV3
        Explicit version 3 matched schedule, including its content digest.
    initial_frame : EvaluationFrame | None
        Optional captured frame zero. If supplied, its simulator tick
        and snapshot must exactly equal the packaged initial state. Defaults to None.

    Returns
    -------
    ResolvedScenarioSpecificationV3
        Version 3 official specification with ten global roster slots, Team A focal
        roles, Team B opposing roles, approved source/configuration identities,
        initial-state digest, horizon, controller identity, and terminal-reward
        endpoint.

    Raises
    ------
    ValueError
        The scenario is invalid or the supplied initial frame disagrees
        with its packaged state; model validation can also reject malformed metadata.

    Notes
    -----
    Inactive slots have no scenario role. Partial episodes leave the endpoint
    unavailable. The success predicate means terminal Team A reward equals 1;
    it does not infer a tactical explanation. This host helper performs no rollout.
    """
    scenario = load_tdm_scenario(scenario_id)
    content = scenario_content(scenario.info)
    return _prepared_scenario_specification(
        scenario, content, seed_schedule, initial_frame=initial_frame
    )


def _prepared_scenario_specification(
    scenario: TDMScenario,
    content: ScenarioContent,
    seed_schedule: ScenarioSeedScheduleV3,
    *,
    initial_frame: EvaluationFrame | None = None,
) -> ResolvedScenarioSpecificationV3:
    """Build the shared specification from an already verified scenario pair.

    scenario and content must come from the same validated loader call. Use the
    exact authored snapshot, not a newly rounded copy. seed_schedule and the
    optional initial_frame have the public builder's contracts. Return the same
    frozen specification without resource reads or repeated scenario validation.
    Reject a supplied frame that differs from the exact authored start.
    """
    scenario_id = scenario.info.scenario_id
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
        "pressure_protocol": _pressure_identity(scenario_id),
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
    """Join one scenario replay to its declared terminal-reward endpoint.

    Parameters
    ----------
    scenario_id : int
        Packaged scenario number, 1 through 8.
    specification : ResolvedScenarioSpecificationV3
        Exact current official specification for this scenario.
    replay : ReplayArtifactV3
        Version 3 replay with the approved frame zero and recorded Team B
        controller identity.
    schedule_coordinate : int
        Zero-based index into the specification's matched schedule.

    Returns
    -------
    ScenarioEvaluationRecordV4
        Version 4 record joining specification, replay, and schedule coordinate.
        A complete replay ending with terminated or truncated reports Team A reward
        and whether it equals 1. Other replays report an unavailable endpoint.

    Raises
    ------
    ValueError
        The specification, initial frame, opposing controller, schedule
        join, or terminal reward is inconsistent. Active Team A rewards must
        agree on one value from -1, 0, and 1.

    Notes
    -----
    Uses recorded rewards and the shared scenario-record validator. It does not
    rerun physics, infer success from visual behavior, or write a file.
    """
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
