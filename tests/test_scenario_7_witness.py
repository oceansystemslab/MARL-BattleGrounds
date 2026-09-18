"""Check Scenario 7's coordinated retreat and four matched healing omissions."""

from collections.abc import Callable
from typing import NamedTuple, cast
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.authoring_compiler import CompiledDevScenarioV1
from tests.scenario_controller_fixtures import (
    SCENARIO_7_SEMANTIC_DIGEST,
    load_scenario_7,
    load_scenario_7_draft,
)

from marl_battlegrounds.core.env import step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_NORTHEAST,
    MOVE_NORTHWEST,
    MOVE_STAY,
    MOVE_WEST,
    STUN_CHANNEL_WARRIOR_CHARGE,
    TEAM_B_ID,
    Action,
    ActionMask,
    DoneFlags,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.reactive_tdm_alpha import reactive_tdm_alpha_policy
from marl_battlegrounds.policies.shared_obs import (
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)

_TEAM = cast(
    Callable[..., ActorAction],
    jax.jit(execute_shared_obs_team_policy, static_argnums=(5, 6)),
)
_STEP = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(step),
)


class WitnessTransition(NamedTuple):
    before: EnvState
    observation: Observation
    mask: ActionMask
    action: Action
    after: EnvState
    successor_observation: Observation
    successor_mask: ActionMask
    reward: Reward
    done: DoneFlags
    info: Info


@pytest.fixture(scope="module")
def scenario() -> CompiledDevScenarioV1:
    return load_scenario_7()


def _run_witness(
    scenario: CompiledDevScenarioV1,
    omit_heal_tick: int | None = None,
) -> list[WitnessTransition]:
    assert omit_heal_tick is None or 1 <= omit_heal_tick <= 4
    state, observation, mask = (
        scenario.initial_state,
        scenario.observation,
        scenario.action_mask,
    )
    availability = build_default_shared_obs_information_availability(
        scenario.config.agent_profile.active_mask,
        scenario.config.agent_profile.team_ids,
    )
    # Team A: Mage, Warrior, dead Hunter, dead Rogue, Priest. The author's
    # unspecified tick-4 movements and Warrior combat are fixed to Stay/no-combat
    # for this witness, not claimed as the only possible winning completion.
    moves = (
        (MOVE_NORTHEAST, MOVE_NORTH, 0, 0, MOVE_NORTHEAST),
        (MOVE_EAST, MOVE_NORTHWEST, 0, 0, MOVE_EAST),
        (MOVE_NORTHEAST, MOVE_WEST, 0, 0, MOVE_NORTHEAST),
        (MOVE_STAY, MOVE_STAY, 0, 0, MOVE_STAY),
    )
    # Relative targets: 1 Mage-A; 2 Warrior-A; 5 Priest-A; 6 Mage-B;
    # 8 Hunter-B; 9 Rogue-B. Each intervention omits just one Priest action.
    targets = ((6, 9, 0, 0, 2), (0, 9, 0, 0, 5), (9, 0, 0, 0, 1), (8, 0, 0, 0, 1))
    ultimates = ((0, 1, 0, 0, 0), (1, 0, 0, 0, 0)) + ((0, 0, 0, 0, 0),) * 2
    trajectory: list[WitnessTransition] = []
    slots = jnp.arange(10)
    for tick in range(4):
        living = state.alive_mask
        np.testing.assert_array_equal(
            observation.self_features[living, AGENT_FEATURE_X : AGENT_FEATURE_Y + 1],
            state.agent_positions[living],
        )
        np.testing.assert_array_equal(
            observation.self_features[living, AGENT_FEATURE_CURRENT_HEALTH],
            state.current_health[living],
        )
        # Team B sees only this epoch's authorized observations and exact masks,
        # never the staged Team A sequence or privileged simulator state.
        bank = build_shared_obs_sensor_source_bank(observation)
        team_b = _TEAM(
            observation,
            mask,
            jax.random.split(jax.random.key(tick), 10),
            bank,
            availability,
            reactive_tdm_alpha_policy,
            TEAM_B_ID,
        )
        current_targets = jnp.asarray(targets[tick], dtype=jnp.int32)
        if omit_heal_tick == tick + 1:
            current_targets = current_targets.at[4].set(0)
        team_a = ActorAction(
            jnp.asarray(moves[tick], dtype=jnp.int32),
            current_targets,
            jnp.asarray(ultimates[tick], dtype=jnp.int32),
        )
        joint = build_joint_action_from_actor_actions(team_a, team_b)
        assert bool(jnp.all(mask.move_mask[slots, joint.move]))
        assert bool(
            jnp.all(
                mask.select_target_use_ultimate_joint_mask[
                    slots, joint.select_target, joint.use_ultimate
                ]
            )
        )
        before, before_observation, before_mask = state, observation, mask
        state, observation, reward, done, mask, info = _STEP(
            scenario.config, state, mask, joint, jax.random.key(tick)
        )
        assert int(info.transition_facts.transition_start_step_count) == int(
            before.step_count
        )
        assert int(state.step_count) == int(before.step_count) + 1
        accepted = info.transition_facts.action_acceptance_facts.accepted_joint_action
        for submitted, actual in zip(joint, accepted, strict=True):
            np.testing.assert_array_equal(actual, submitted)
        trajectory.append(
            WitnessTransition(
                before,
                before_observation,
                before_mask,
                joint,
                state,
                observation,
                mask,
                reward,
                done,
                info,
            )
        )
        if bool(done.terminated) or bool(done.truncated):
            break
    return trajectory


@pytest.fixture(scope="module")
def trajectory(scenario: CompiledDevScenarioV1) -> list[WitnessTransition]:
    return _run_witness(scenario)


def test_scenario_7_fixture_binds_the_approved_revision_25(
    scenario: CompiledDevScenarioV1,
) -> None:
    draft = load_scenario_7_draft()
    assert draft.asset_id == "scenario_7"
    assert draft.revision == 25
    assert draft.content.description == draft.content.notes == ""
    assert len(draft.content.embedded_map.obstacles) == 11
    assert scenario.semantic_digest == SCENARIO_7_SEMANTIC_DIGEST
    np.testing.assert_array_equal(
        scenario.initial_state.team_deathmatch_scores, [17, 19]
    )
    np.testing.assert_array_equal(
        scenario.initial_state.current_health, [21, 1, 0, 0, 5, 14, 0, 20, 54, 0]
    )
    assert int(scenario.initial_state.step_count) == 295
    assert scenario.config.max_steps == 300
    assert not bool(scenario.initial_state.has_previous_timestep_joint_action)


def test_witness_repeats_and_wins_on_tick_4_without_a_team_a_death(
    scenario: CompiledDevScenarioV1, trajectory: list[WitnessTransition]
) -> None:
    repeated = _run_witness(scenario)
    for first, second in zip(
        jax.tree.leaves(trajectory), jax.tree.leaves(repeated), strict=True
    ):
        np.testing.assert_array_equal(first, second)
    assert len(trajectory) == 4
    scores = ((18, 19), (18, 19), (19, 19), (20, 19))
    health = (
        (1.938749, 3.9, 5),
        (1.938749, 3.9, 7.9),
        (4.838749, 3.9, 7.9),
        (6.838749, 3.9, 7.9),
    )
    killed = ((5,), (), (8,), (7,))
    for tick, transition in enumerate(trajectory):
        deaths = transition.info.transition_facts.death_facts.is_newly_dead_by_recipient
        np.testing.assert_array_equal(np.flatnonzero(deaths), killed[tick])
        assert not bool(jnp.any(deaths[:5]))
        np.testing.assert_array_equal(
            transition.after.team_deathmatch_scores, scores[tick]
        )
        np.testing.assert_allclose(
            transition.after.current_health[jnp.asarray([0, 1, 4])],
            health[tick],
            rtol=0,
            atol=1e-5,
        )
        assert bool(transition.done.terminated) == (tick == 3)
        assert not bool(transition.done.truncated)
        np.testing.assert_array_equal(
            transition.reward.rewards[:5], [1 if tick == 3 else 0] * 5
        )
        assert not bool(
            jnp.any(
                transition.info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent
            )
        )
    final = trajectory[-1]
    assert int(final.after.step_count) == 299
    np.testing.assert_allclose(
        final.after.agent_positions[jnp.asarray([0, 1, 4, 7])],
        [
            [12.424541, 6.202107],
            [8.797300, 5.133029],
            [13.511398, 5.480935],
            [14.5, 7.821322],
        ],
        rtol=0,
        atol=1e-5,
    )


def test_team_b_responses_are_generated_by_alpha(
    trajectory: list[WitnessTransition],
) -> None:
    moves = (
        (MOVE_EAST, 0, MOVE_STAY, MOVE_NORTHEAST, 0),
        (0, 0, MOVE_NORTHEAST, 0, 0),
        (0, 0, MOVE_NORTHEAST, MOVE_NORTHWEST, 0),
        (0, 0, MOVE_NORTHEAST, 0, 0),
    )
    targets = ((6, 0, 7, 0, 0), (0, 0, 10, 0, 0), (0, 0, 6, 0, 0), (0, 0, 6, 0, 0))
    for tick, transition in enumerate(trajectory):
        np.testing.assert_array_equal(transition.action.move[5:], moves[tick])
        np.testing.assert_array_equal(
            transition.action.select_target[5:], targets[tick]
        )
        np.testing.assert_array_equal(transition.action.use_ultimate[5:], [0] * 5)


def test_charge_and_burst_control_subsequent_decisions_not_accepted_combat(
    trajectory: list[WitnessTransition],
) -> None:
    first, second, third, _ = trajectory
    first_combat = first.info.transition_facts.combat_transition_facts
    # Killing Mage-B this transition does not erase its precommitted Basic.
    assert bool(first_combat.basic_effect_is_activated_by_source[5])
    assert float(first_combat.total_effective_damage_by_recipient[0]) > 0
    assert not bool(first.after.alive_mask[5])
    assert int(second.before.stun_durations[8, STUN_CHANNEL_WARRIOR_CHARGE]) > 0
    assert tuple(int(leaf[8]) for leaf in second.action) == (MOVE_STAY, 0, 0)
    assert int(third.before.stun_durations[8, STUN_CHANNEL_WARRIOR_CHARGE]) == 0
    second_combat = second.info.transition_facts.combat_transition_facts
    assert bool(second_combat.mage_burst_damage_amplification_is_applied_by_source[0])
    assert not bool(second_combat.basic_effect_is_activated_by_source[0])
    assert float(second_combat.source_modified_damage_output_by_source[0]) == 0
    assert float(third.before.current_health[0]) == pytest.approx(1.938749, abs=1e-5)
    third_combat = third.info.transition_facts.combat_transition_facts
    assert bool(third_combat.basic_effect_is_activated_by_source[0])
    assert float(third_combat.source_modified_damage_output_by_source[0]) > float(
        first_combat.source_modified_damage_output_by_source[0]
    )


@pytest.mark.parametrize(
    ("omit_heal_tick", "score", "reward", "team_a_death"),
    (
        (1, (18, 20), -1, 1),
        (2, (18, 20), -1, 4),
        (3, (19, 20), -1, 0),
        (4, (20, 20), 0, 0),
    ),
)
def test_omitting_one_heal_changes_the_matched_sequence_outcome(
    scenario: CompiledDevScenarioV1,
    trajectory: list[WitnessTransition],
    omit_heal_tick: int,
    score: tuple[int, int],
    reward: int,
    team_a_death: int,
) -> None:
    counterfactual = _run_witness(scenario, omit_heal_tick)
    assert len(counterfactual) == omit_heal_tick
    # This is one omitted action with otherwise identical inputs/actions until
    # that transition, not an exhaustive search for alternative winning plans.
    for actual, baseline in zip(
        counterfactual, trajectory[:omit_heal_tick], strict=True
    ):
        for first, second in zip(
            jax.tree.leaves((actual.before, actual.observation, actual.mask)),
            jax.tree.leaves((baseline.before, baseline.observation, baseline.mask)),
            strict=True,
        ):
            np.testing.assert_array_equal(first, second)
    final = counterfactual[-1]
    expected_action = trajectory[omit_heal_tick - 1].action
    np.testing.assert_array_equal(final.action.move, expected_action.move)
    np.testing.assert_array_equal(
        final.action.use_ultimate, expected_action.use_ultimate
    )
    np.testing.assert_array_equal(
        final.action.select_target, expected_action.select_target.at[4].set(0)
    )
    assert bool(final.done.terminated)
    assert not bool(final.done.truncated)
    assert int(final.after.step_count) == 295 + omit_heal_tick
    np.testing.assert_array_equal(final.after.team_deathmatch_scores, score)
    np.testing.assert_array_equal(final.reward.rewards[:5], [reward] * 5)
    deaths = final.info.transition_facts.death_facts.is_newly_dead_by_recipient
    np.testing.assert_array_equal(np.flatnonzero(deaths[:5]), [team_a_death])
    assert (
        float(
            final.info.transition_facts.combat_transition_facts.total_effective_healing_by_recipient[
                team_a_death
            ]
        )
        == 0
    )


def test_each_decision_has_one_current_bank_policy_assembly_and_transition(
    scenario: CompiledDevScenarioV1,
) -> None:
    calls = Mock()
    for name, original in (
        ("bank", build_shared_obs_sensor_source_bank),
        ("policy", _TEAM),
        ("assembly", build_joint_action_from_actor_actions),
        ("step", _STEP),
    ):
        calls.attach_mock(Mock(wraps=original), name)
    with (
        patch(f"{__name__}.build_shared_obs_sensor_source_bank", calls.bank),
        patch(f"{__name__}._TEAM", calls.policy),
        patch(f"{__name__}.build_joint_action_from_actor_actions", calls.assembly),
        patch(f"{__name__}._STEP", calls.step),
    ):
        trajectory = _run_witness(scenario)
    assert [call[0] for call in calls.mock_calls] == [
        "bank",
        "policy",
        "assembly",
        "step",
    ] * 4
    for tick, transition in enumerate(trajectory):
        assert calls.bank.call_args_list[tick].args[0] is transition.observation
        policy_args = calls.policy.call_args_list[tick].args
        assert policy_args[0] is transition.observation
        assert policy_args[1] is transition.mask
        assert policy_args[5] is reactive_tdm_alpha_policy
        assert policy_args[6] == TEAM_B_ID
        assert calls.step.call_args_list[tick].args[1] is transition.before
        assert calls.step.call_args_list[tick].args[2] is transition.mask
        assert calls.step.call_args_list[tick].args[3] is transition.action
