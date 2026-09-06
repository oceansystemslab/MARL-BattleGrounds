"""Real shoulder contact and matched strict-steering comparisons."""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.authoring_compiler import (
    CompiledDevScenarioV1,
    compile_dev_scenario,
)
from tests.scenario_controller_fixtures import load_scenario_3_draft

from marl_battlegrounds.core.env import step
from marl_battlegrounds.core.types import (
    MOVE_NORTH,
    MOVE_NORTHWEST,
    MOVE_SOUTH,
    MOVE_SOUTHWEST,
    MOVE_STAY,
    MOVE_WEST,
    TEAM_B_ID,
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
from marl_battlegrounds.policies.scenario_3 import scenario_3_policy
from marl_battlegrounds.policies.scenario_5 import scenario_5_policy
from marl_battlegrounds.policies.shared_obs import (
    SharedObsPolicy,
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


@pytest.fixture(scope="module", params=("hunter", "priest"))
def scenario(request: pytest.FixtureRequest) -> CompiledDevScenarioV1:
    draft = load_scenario_3_draft()
    # Also exercise Priest pursuit with the same test-only body-blocking setup.
    roster = list(draft.content.roster)
    roster[2] = roster[2].model_copy(update={"class_name": request.param})
    return compile_dev_scenario(
        draft.model_copy(
            update={
                "content": draft.content.model_copy(update={"roster": tuple(roster)})
            }
        )
    )


def _run(
    scenario: CompiledDevScenarioV1,
    policy: SharedObsPolicy,
    first_warrior_move: int,
) -> list[tuple[ActorAction, EnvState]]:
    state, obs, mask = (
        scenario.initial_state,
        scenario.observation,
        scenario.action_mask,
    )
    availability = build_default_shared_obs_information_availability(
        scenario.config.agent_profile.active_mask,
        scenario.config.agent_profile.team_ids,
    )
    result: list[tuple[ActorAction, EnvState]] = []
    slots = jnp.arange(10)
    for tick in range(4):
        b = _TEAM(
            obs,
            mask,
            jax.random.split(jax.random.key(tick), 10),
            build_shared_obs_sensor_source_bank(obs),
            availability,
            policy,
            TEAM_B_ID,
        )
        neutral = jnp.zeros(5, dtype=jnp.int32)
        a = ActorAction(
            neutral.at[1].set(first_warrior_move if tick == 0 else MOVE_STAY),
            neutral,
            neutral,
        )
        joint = build_joint_action_from_actor_actions(a, b)
        assert bool(jnp.all(mask.move_mask[slots, joint.move]))
        assert bool(
            jnp.all(
                mask.select_target_use_ultimate_joint_mask[
                    slots, joint.select_target, joint.use_ultimate
                ]
            )
        )
        state, obs, _, _, mask, info = _STEP(
            scenario.config, state, mask, joint, jax.random.key(tick)
        )
        accepted = info.transition_facts.action_acceptance_facts.accepted_joint_action
        for submitted, actual in zip(joint, accepted, strict=True):
            np.testing.assert_array_equal(actual, submitted)
        result.append((b, state))
    return result


@pytest.mark.parametrize(
    "warrior_move,expected_moves",
    [
        (MOVE_STAY, [MOVE_NORTHWEST, MOVE_WEST, MOVE_SOUTH]),
        (MOVE_NORTH, [MOVE_NORTHWEST, MOVE_SOUTHWEST, MOVE_WEST]),
    ],
)
def test_shoulder_bypass_reaches_priority_prey_one_decision_before_strict_avoidance(
    scenario: CompiledDevScenarioV1,
    warrior_move: int,
    expected_moves: list[int],
) -> None:
    # Only Rogue is alive on Team B before the first wave, so both policies
    # pursue the same vulnerable prey and choose independent combat targets.
    strict = _run(scenario, scenario_3_policy, warrior_move)
    bypass = _run(scenario, scenario_5_policy, warrior_move)
    assert [int(b.move[3]) for b, _ in bypass[:3]] == expected_moves
    assert [int(b.select_target[3]) for b, _ in bypass[:3]] == [7, 7, 8]
    assert [int(b.select_target[3]) for b, _ in strict] == [7, 7, 0, 8]
    assert float(bypass[1][1].current_health[2]) == 1
    assert float(bypass[2][1].current_health[2]) == 0
    assert float(strict[2][1].current_health[2]) == 1
    assert float(strict[3][1].current_health[2]) == 0

    # The Warrior submits Stay on the second tick. Its observed displacement
    # proves real body contact, not a hypothetical endpoint passing a predicate.
    before, after = bypass[0][1], bypass[1][1]
    displacement = np.asarray(after.agent_positions - before.agent_positions)
    assert float(np.linalg.norm(displacement[1])) > 0.01
    assert float(np.linalg.norm(displacement[8])) > 1.2
    assert float(after.current_health[8]) == 47
    np.testing.assert_array_equal(bypass[2][1].team_deathmatch_scores, [0, 1])


def test_bypass_responds_to_observed_moving_blocker_not_pending_actions(
    scenario: CompiledDevScenarioV1,
) -> None:
    stationary = _run(scenario, scenario_5_policy, MOVE_STAY)
    moving = _run(scenario, scenario_5_policy, MOVE_NORTH)
    for left, right in zip(stationary[0][0], moving[0][0], strict=True):
        np.testing.assert_array_equal(left, right)
    assert int(stationary[1][0].move[3]) == MOVE_WEST
    assert int(moving[1][0].move[3]) == MOVE_SOUTHWEST
