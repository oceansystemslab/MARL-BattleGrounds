"""Check Scenario 3's saved eight-turn body-screening solution.

The file also checks the collision regression and missed-movement variants.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.scenario_witness_assertions import assert_witness_tick

from marl_battlegrounds.core.env import initialize_scenario_state, step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_SOUTHEAST,
    MOVE_SOUTHWEST,
    MOVE_STAY,
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
from marl_battlegrounds.policies.reactive_tdm_beta import (
    reactive_tdm_beta_controller_descriptor,
    reactive_tdm_beta_policy,
)
from marl_battlegrounds.policies.shared_obs import (
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)
from marl_battlegrounds.tasks import TDMScenario, load_tdm_scenario

_TEAM = cast(
    Callable[..., ActorAction],
    jax.jit(execute_shared_obs_team_policy, static_argnums=(5, 6)),
)
_STEP = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(step),
)
_EXPECTED_PATH = Path(__file__).parent / "fixtures" / "scenario_3_witness.json"
# The user lists Hunter first and Warrior second. Simulator slots are 2 and 1.
_HUNTER_WARRIOR_MOVES = (
    (MOVE_NORTH, MOVE_NORTH),
    (MOVE_NORTH, MOVE_SOUTHEAST),
    (MOVE_SOUTH, MOVE_EAST),
    (MOVE_SOUTH, MOVE_SOUTH),
    (MOVE_NORTH, MOVE_NORTH),
    (MOVE_NORTH, MOVE_SOUTH),
    (MOVE_NORTH, MOVE_NORTH),
    (MOVE_EAST, MOVE_SOUTHWEST),
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
def scenario() -> TDMScenario:
    return load_tdm_scenario(3)


def _run_witness(
    scenario: TDMScenario, *, omit_south_south: bool = False
) -> tuple[WitnessTransition, ...]:
    state, observation, mask, _ = initialize_scenario_state(
        scenario.initial_state, scenario.config
    )
    profile = scenario.config.agent_profile
    availability = build_default_shared_obs_information_availability(
        profile.active_mask, profile.team_ids
    )
    moves = _HUNTER_WARRIOR_MOVES
    if omit_south_south:
        # Remove turn 4, then finish the shortened line with Stay and Basics.
        # This is one declared continuation, not every possible alternative.
        moves = (*moves[:3], *moves[4:], (MOVE_STAY, MOVE_STAY))
    trajectory: list[WitnessTransition] = []
    for tick, (hunter_move, warrior_move) in enumerate(moves):
        bank = build_shared_obs_sensor_source_bank(observation)
        team_b = _TEAM(
            observation,
            mask,
            jax.random.split(jax.random.key(tick), 10),
            bank,
            availability,
            reactive_tdm_beta_policy,
            TEAM_B_ID,
        )
        # Both use Basic against Rogue-B (relative target 9) when legal now.
        # Other Team A slots keep Stay/no-combat, including after respawn.
        targets = [0] * 5
        for slot in (1, 2):
            targets[slot] = (
                9 if bool(mask.select_target_use_ultimate_joint_mask[slot, 9, 0]) else 0
            )
        team_a = ActorAction(
            jnp.asarray([0, warrior_move, hunter_move, 0, 0], dtype=jnp.int32),
            jnp.asarray(targets, dtype=jnp.int32),
            jnp.zeros(5, dtype=jnp.int32),
        )
        joint = build_joint_action_from_actor_actions(team_a, team_b)
        before, before_observation, before_mask = state, observation, mask
        state, observation, reward, done, mask, info = _STEP(
            scenario.config, state, mask, joint, jax.random.key(tick)
        )
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
    return tuple(trajectory)


@pytest.fixture(scope="module")
def trajectory(scenario: TDMScenario) -> tuple[WitnessTransition, ...]:
    return _run_witness(scenario)


def test_scenario_3_binds_the_approved_setup(scenario: TDMScenario) -> None:
    assert scenario.info.approved_source.revision == 24
    assert scenario.info.source.revision == 26
    assert scenario.info.source.semantic_digest == (
        "c5c409e4f64deacc5e54b7c09ee62c59667342138786303b2fc52103a5dc3459"
    )
    assert scenario.info.resolved_initial_state_digest == (
        "43bf4bbe5b6f70ff14d85533b9aaec4d626580a55ac6b1cf665e70ded33fea8a"
    )
    assert reactive_tdm_beta_controller_descriptor()["version"] == 5
    np.testing.assert_array_equal(
        scenario.initial_state.team_deathmatch_scores, [19, 19]
    )
    np.testing.assert_array_equal(
        scenario.initial_state.current_health, [0, 81, 1, 0, 0, 0, 0, 0, 100, 0]
    )
    assert int(scenario.initial_state.step_count) == 290
    assert scenario.config.max_steps == 300


def test_known_sequence_wins_at_step_298(
    trajectory: tuple[WitnessTransition, ...],
) -> None:
    assert len(trajectory) == 8
    for tick, transition in enumerate(trajectory, start=1):
        final = tick == 8
        assert int(transition.before.step_count) == 289 + tick
        assert int(transition.after.step_count) == 290 + tick
        np.testing.assert_array_equal(
            transition.after.team_deathmatch_scores, [20 if final else 19, 19]
        )
        np.testing.assert_array_equal(
            np.flatnonzero(
                transition.info.transition_facts.death_facts.is_newly_dead_by_recipient
            ),
            [8] if final else [],
        )
        assert bool(transition.done.terminated) == final
        assert not bool(transition.done.truncated)
        np.testing.assert_array_equal(
            transition.reward.rewards,
            [1] * 5 + [-1] * 5 if final else [0] * 10,
        )
        assert int(transition.info.transition_facts.team_deathmatch_facts.outcome) == (
            1 if final else 0
        )
    np.testing.assert_allclose(
        trajectory[-1].after.current_health[jnp.asarray([1, 2, 8])],
        [9.6, 1, 0],
        rtol=0,
        atol=1e-5,
    )


def test_all_slots_keep_the_recorded_positions_health_and_actions(
    trajectory: tuple[WitnessTransition, ...],
) -> None:
    expected = json.loads(_EXPECTED_PATH.read_text())
    assert expected["scenario_id"] == 3
    assert expected["approved_revision"] == 24
    assert expected["source_revision"] == 26
    slots = jnp.arange(10)
    for transition, row in zip(trajectory, expected["ticks"], strict=True):
        assert_witness_tick(
            row,
            state=transition.after,
            action=transition.action,
            reward=transition.reward,
            done=transition.done,
            info=transition.info,
            case_name="scenario_3",
        )
        assert bool(jnp.all(transition.mask.move_mask[slots, transition.action.move]))
        assert bool(
            jnp.all(
                transition.mask.select_target_use_ultimate_joint_mask[
                    slots,
                    transition.action.select_target,
                    transition.action.use_ultimate,
                ]
            )
        )
        living = transition.before.alive_mask
        np.testing.assert_array_equal(
            transition.observation.self_features[
                living, AGENT_FEATURE_X : AGENT_FEATURE_Y + 1
            ],
            transition.before.agent_positions[living],
        )
        np.testing.assert_array_equal(
            transition.observation.self_features[living, AGENT_FEATURE_CURRENT_HEALTH],
            transition.before.current_health[living],
        )


def test_final_turn_gives_hunter_a_safe_finishing_attack(
    trajectory: tuple[WitnessTransition, ...],
) -> None:
    final = trajectory[-1]
    assert bool(final.mask.select_target_use_ultimate_joint_mask[2, 9, 0])
    assert not bool(final.mask.select_target_use_ultimate_joint_mask[1, 9, 0])
    assert not bool(final.mask.select_target_use_ultimate_joint_mask[8, 7, 0])
    combat = final.info.transition_facts.combat_transition_facts
    np.testing.assert_array_equal(
        combat.basic_effect_is_activated_by_source[jnp.asarray([1, 2, 8])],
        [False, True, False],
    )
    np.testing.assert_array_equal(
        combat.total_effective_damage_by_recipient[jnp.asarray([1, 2, 8])], [0, 0, 6]
    )
    assert float(final.before.current_health[1]) == pytest.approx(9.6)
    assert float(final.after.current_health[1]) == pytest.approx(9.6)


def test_respawned_agents_keep_the_declared_behavior(
    trajectory: tuple[WitnessTransition, ...],
) -> None:
    revived = [0, 3, 4, 5, 6, 7, 9]
    for tick, transition in enumerate(trajectory, start=1):
        respawn = transition.info.transition_facts.respawn_facts
        np.testing.assert_array_equal(
            np.flatnonzero(respawn.was_respawned_this_transition_by_agent),
            revived if tick == 5 else [],
        )
        np.testing.assert_array_equal(
            respawn.respawn_wave_occurred_this_transition_by_team, [tick == 5] * 2
        )
        np.testing.assert_array_equal(
            transition.after.spawn_shield_durations[jnp.asarray(revived)],
            [8 - tick if tick >= 5 else 0] * len(revived),
        )
        for head in transition.action:
            np.testing.assert_array_equal(head[jnp.asarray([0, 3, 4])], [0, 0, 0])


def test_omitting_south_south_loses_hunter_before_the_finish(
    scenario: TDMScenario, trajectory: tuple[WitnessTransition, ...]
) -> None:
    without_pause = _run_witness(scenario, omit_south_south=True)
    assert len(without_pause) == 7
    for original, changed in zip(trajectory[:3], without_pause[:3], strict=True):
        for left, right in zip(
            jax.tree.leaves(original), jax.tree.leaves(changed), strict=True
        ):
            np.testing.assert_array_equal(left, right)
    final = without_pause[-1]
    assert int(final.after.step_count) == 297
    np.testing.assert_array_equal(final.after.team_deathmatch_scores, [19, 20])
    np.testing.assert_array_equal(
        np.flatnonzero(
            final.info.transition_facts.death_facts.is_newly_dead_by_recipient
        ),
        [2],
    )
    np.testing.assert_array_equal(final.reward.rewards, [-1] * 5 + [1] * 5)
    assert bool(final.done.terminated)
    assert not bool(final.done.truncated)
    assert int(final.info.transition_facts.team_deathmatch_facts.outcome) == 2
