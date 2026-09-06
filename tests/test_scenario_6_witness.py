"""Scenario 6's authored bait-and-punish witness and two matched interventions."""

from collections.abc import Callable
from operator import itemgetter
from typing import Literal, NamedTuple, cast
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.authoring_compiler import CompiledDevScenarioV1
from tests.scenario_controller_fixtures import (
    SCENARIO_6_SEMANTIC_DIGEST,
    load_scenario_6,
    load_scenario_6_draft,
)

from marl_battlegrounds.core.env import step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    MOVE_NORTHWEST,
    MOVE_SOUTH,
    MOVE_SOUTHEAST,
    MOVE_SOUTHWEST,
    MOVE_STAY,
    MOVE_WEST,
    STUN_CHANNEL_HUNTER_TRAP,
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
    compose_shared_obs_unit_features,
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
_Intervention = Literal["none", "early_hunter_basic", "omit_tick_4_heal"]


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
    return load_scenario_6()


def _run_witness(
    scenario: CompiledDevScenarioV1,
    intervention: _Intervention = "none",
) -> list[WitnessTransition]:
    state, observation, mask = (
        scenario.initial_state,
        scenario.observation,
        scenario.action_mask,
    )
    availability = build_default_shared_obs_information_availability(
        scenario.config.agent_profile.active_mask,
        scenario.config.agent_profile.team_ids,
    )
    # Team A: Mage, dead Warrior, Hunter, dead Rogue, Priest.
    moves = (
        (MOVE_NORTHWEST, 0, MOVE_NORTHWEST, 0, MOVE_SOUTHWEST),
        (MOVE_SOUTHEAST, 0, MOVE_SOUTHWEST, 0, MOVE_SOUTHWEST),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    # Relative targets: 3 Hunter-A; 6 Mage-B; 7 Warrior-B; 9 Rogue-B.
    targets = (
        (0, 0, 9, 0, 3),
        (6, 0, 9 if intervention == "early_hunter_basic" else 0, 0, 3),
        (9, 0, 0, 0, 3),
        (0, 0, 0, 0, 0 if intervention == "omit_tick_4_heal" else 3),
        (7, 0, 0, 0, 3),
    )
    ultimates = ((1, 0, 1, 0, 0),) + ((0, 0, 0, 0, 0),) * 4
    trajectory: list[WitnessTransition] = []
    slots = jnp.arange(10)
    for tick in range(5):
        living = state.alive_mask
        np.testing.assert_array_equal(
            observation.self_features[living, AGENT_FEATURE_X : AGENT_FEATURE_Y + 1],
            state.agent_positions[living],
        )
        np.testing.assert_array_equal(
            observation.self_features[living, AGENT_FEATURE_CURRENT_HEALTH],
            state.current_health[living],
        )
        # One current-epoch bank and policy call. Team B never receives the
        # scripted Team A choices, privileged facts, or a future observation.
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
        team_a = ActorAction(
            jnp.asarray(moves[tick], dtype=jnp.int32),
            jnp.asarray(targets[tick], dtype=jnp.int32),
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
        # Counterfactuals may end sooner. Never stage a later heal on a dead
        # recipient or step after termination merely to complete the tape.
        if bool(done.terminated) or bool(done.truncated):
            break
    return trajectory


@pytest.fixture(scope="module")
def trajectory(scenario: CompiledDevScenarioV1) -> list[WitnessTransition]:
    return _run_witness(scenario)


def test_scenario_6_fixture_binds_the_author_approved_score_correction(
    scenario: CompiledDevScenarioV1,
) -> None:
    draft = load_scenario_6_draft()
    assert draft.asset_id == "scenario_6"
    assert draft.revision == 10  # Source revision, with the named score correction.
    assert draft.content.description == draft.content.notes == ""
    assert len(draft.content.embedded_map.obstacles) == 3
    assert scenario.semantic_digest == SCENARIO_6_SEMANTIC_DIGEST
    np.testing.assert_array_equal(
        scenario.initial_state.team_deathmatch_scores, [17, 19]
    )
    assert int(scenario.initial_state.step_count) == 295
    assert scenario.config.max_steps == 300
    assert not bool(scenario.initial_state.has_previous_timestep_joint_action)


def test_witness_repeats_and_wins_only_on_the_final_transition(
    scenario: CompiledDevScenarioV1, trajectory: list[WitnessTransition]
) -> None:
    repeated = _run_witness(scenario)
    for first, second in zip(
        jax.tree.leaves(trajectory), jax.tree.leaves(repeated), strict=True
    ):
        np.testing.assert_array_equal(first, second)
    assert len(trajectory) == 5
    scores = ((17, 19), (18, 19), (19, 19), (19, 19), (20, 19))
    health = ((23, 3, 1), (8.05, 11, 1), (8.05, 19, 1), (8.05, 7, 1), (8.05, 7, 1))
    killed = ((), (5,), (8,), (), (6,))
    for tick, transition in enumerate(trajectory):
        deaths = transition.info.transition_facts.death_facts.is_newly_dead_by_recipient
        np.testing.assert_array_equal(np.flatnonzero(deaths), killed[tick])
        np.testing.assert_array_equal(
            transition.after.team_deathmatch_scores, scores[tick]
        )
        np.testing.assert_allclose(
            transition.after.current_health[jnp.asarray([0, 2, 4])],
            health[tick],
            rtol=0,
            atol=1e-5,
        )
        assert bool(transition.done.terminated) == (tick == 4)
        assert bool(transition.done.truncated) == (tick == 4)
        np.testing.assert_array_equal(
            transition.reward.rewards[:5], [1 if tick == 4 else 0] * 5
        )
    final = trajectory[-1]
    assert int(final.after.step_count) == 300
    np.testing.assert_array_equal(
        final.info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent,
        [False, True, False, True, False, True, False, True, True, True],
    )
    assert not bool(final.after.alive_mask[6])  # Newly killed Warrior misses this wave.


def test_team_b_responses_are_generated_by_alpha(
    trajectory: list[WitnessTransition],
) -> None:
    moves = (
        (MOVE_SOUTH, MOVE_SOUTHWEST, 0, MOVE_WEST, 0),
        (MOVE_NORTHWEST, MOVE_SOUTHWEST, 0, 0, 0),
        (0, MOVE_SOUTHWEST, 0, 0, 0),
        (0, MOVE_SOUTHWEST, 0, 0, 0),
        (0, MOVE_SOUTHWEST, 0, 0, 0),
    )
    targets = (
        (0, 0, 0, 8, 0),
        (6, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 8, 0, 0, 0),
        (0, 8, 0, 0, 0),
    )
    for tick, transition in enumerate(trajectory):
        np.testing.assert_array_equal(transition.action.move[5:], moves[tick])
        np.testing.assert_array_equal(
            transition.action.select_target[5:], targets[tick]
        )
        np.testing.assert_array_equal(
            transition.action.use_ultimate[5:], [0, int(tick == 3), 0, 0, 0]
        )


def test_trap_burst_and_simultaneous_healing_have_the_documented_epochs(
    trajectory: list[WitnessTransition],
) -> None:
    first, second, third, fourth, fifth = trajectory
    first_combat = first.info.transition_facts.combat_transition_facts
    assert bool(first_combat.mage_burst_damage_amplification_is_applied_by_source[0])
    assert not bool(first_combat.basic_effect_is_activated_by_source[0])
    assert float(first_combat.source_modified_damage_output_by_source[0]) == 0
    assert bool(first_combat.basic_effect_is_activated_by_source[8])
    assert float(first_combat.total_effective_damage_by_recipient[2]) == 12
    assert float(first_combat.total_effective_healing_by_recipient[2]) == 8
    assert int(first.after.stun_durations[8, STUN_CHANNEL_HUNTER_TRAP]) > 0
    assert int(second.before.stun_durations[8, STUN_CHANNEL_HUNTER_TRAP]) > 0
    assert int(third.before.stun_durations[8, STUN_CHANNEL_HUNTER_TRAP]) > 0
    assert tuple(int(leaf[8]) for leaf in second.action) == (0, 0, 0)
    assert tuple(int(leaf[8]) for leaf in third.action) == (0, 0, 0)
    assert float(
        second.info.transition_facts.combat_transition_facts.source_modified_damage_output_by_source[
            0
        ]
    ) == pytest.approx(22.425, abs=1e-5)
    charge = fourth.info.transition_facts.combat_transition_facts
    assert bool(charge.ultimate_effect_is_activated_by_source[6])
    assert float(fourth.before.current_health[2]) == 19
    assert float(charge.total_effective_damage_by_recipient[2]) == 20
    assert float(charge.total_effective_healing_by_recipient[2]) == 8
    assert float(charge.health_after_combat_resolution_by_recipient[2]) == 7
    assert int(fifth.before.stun_durations[2, STUN_CHANNEL_WARRIOR_CHARGE]) == 1
    assert tuple(int(leaf[2]) for leaf in fifth.action) == (MOVE_STAY, 0, 0)
    final_combat = fifth.info.transition_facts.combat_transition_facts
    assert float(final_combat.total_effective_damage_by_recipient[2]) == 8
    assert float(final_combat.total_effective_healing_by_recipient[2]) == 8


def test_warrior_is_new_information_and_hunter_is_the_only_charge_target(
    scenario: CompiledDevScenarioV1, trajectory: list[WitnessTransition]
) -> None:
    availability = build_default_shared_obs_information_availability(
        scenario.config.agent_profile.active_mask,
        scenario.config.agent_profile.team_ids,
    )
    for tick in (0, 1):
        observation = trajectory[tick].observation
        bank = build_shared_obs_sensor_source_bank(observation)
        for actor in (0, 2, 4):
            scalar = jax.tree.map(itemgetter(actor), observation)
            _, _, _, enemies_visible = compose_shared_obs_unit_features(
                scalar, bank, availability[actor], jnp.int32(actor)
            )
            assert bool(enemies_visible[1]) == (tick == 1)
    charge_targets = np.flatnonzero(
        trajectory[3].mask.select_target_use_ultimate_joint_mask[6, :, 1]
    )
    np.testing.assert_array_equal(charge_targets, [8])


@pytest.mark.parametrize(
    ("intervention", "terminal_tick"),
    (("early_hunter_basic", 3), ("omit_tick_4_heal", 4)),
)
def test_matched_interventions_lose_without_a_post_terminal_continuation(
    scenario: CompiledDevScenarioV1,
    intervention: _Intervention,
    terminal_tick: int,
) -> None:
    trajectory = _run_witness(scenario, intervention)
    assert len(trajectory) == terminal_tick
    final = trajectory[-1]
    assert bool(final.done.terminated)
    assert not bool(final.done.truncated)
    np.testing.assert_array_equal(final.after.team_deathmatch_scores, [19, 20])
    np.testing.assert_array_equal(final.reward.rewards[:5], [-1] * 5)
    assert bool(final.info.transition_facts.death_facts.is_newly_dead_by_recipient[2])
    if intervention == "early_hunter_basic":
        second = trajectory[1]
        assert int(second.before.stun_durations[8, STUN_CHANNEL_HUNTER_TRAP]) > 0
        assert int(second.after.stun_durations[8, STUN_CHANNEL_HUNTER_TRAP]) == 0
        assert float(second.after.current_health[8]) == 4
        assert tuple(int(leaf[8]) for leaf in second.action) == (0, 0, 0)
        assert int(final.action.use_ultimate[8]) == 1
    else:
        combat = final.info.transition_facts.combat_transition_facts
        assert float(final.before.current_health[2]) == 19
        assert float(combat.total_effective_damage_by_recipient[2]) == 20
        assert float(combat.total_effective_healing_by_recipient[2]) == 0


def test_each_decision_has_one_current_bank_assembly_and_transition(
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
    ] * 5
    for tick, transition in enumerate(trajectory):
        assert calls.bank.call_args_list[tick].args[0] is transition.observation
        assert calls.policy.call_args_list[tick].args[0] is transition.observation
        assert calls.policy.call_args_list[tick].args[1] is transition.mask
        assert calls.step.call_args_list[tick].args[1] is transition.before
        assert calls.step.call_args_list[tick].args[2] is transition.mask
        assert calls.step.call_args_list[tick].args[3] is transition.action
