"""Check the current five-turn healing solution for Scenario 5.

Historical reference cases remain separate from the current scenario.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.scenario_controller_fixtures import (
    SCENARIO_5_SEMANTIC_DIGEST,
    load_scenario_5,
    load_scenario_5_draft,
)

from marl_battlegrounds.core.env import initialize_scenario_state, step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    STUN_CHANNEL_ROGUE_POISON,
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
from marl_battlegrounds.policies.reactive_tdm_beta import reactive_tdm_beta_policy
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


class WitnessTransition(NamedTuple):
    before: EnvState
    action: Action
    after: EnvState
    reward: Reward
    done: DoneFlags
    info: Info


@pytest.fixture(scope="module")
def scenario() -> TDMScenario:
    return load_tdm_scenario(5)


def _run_witness(scenario: TDMScenario) -> list[WitnessTransition]:
    state, observation, mask, _ = initialize_scenario_state(
        scenario.initial_state, scenario.config
    )
    availability = build_default_shared_obs_information_availability(
        scenario.config.agent_profile.active_mask,
        scenario.config.agent_profile.team_ids,
    )
    expected = json.loads(
        (Path(__file__).parent / "fixtures" / "scenario_5_witness.json").read_text()
    )
    assert scenario.info.source.revision == expected["source_revision"]
    assert scenario.info.source.semantic_digest == expected["semantic_digest"]
    commands = expected["ticks"]
    assert len(commands) == 5
    trajectory: list[WitnessTransition] = []
    slots = jnp.arange(10)
    for tick in range(5):
        # The actual Team B policy receives the current epoch, never Team A's
        # pending action or the successor. No Team B actions are supplied here.
        living = state.alive_mask
        np.testing.assert_array_equal(
            observation.self_features[living, AGENT_FEATURE_X : AGENT_FEATURE_Y + 1],
            state.agent_positions[living],
        )
        np.testing.assert_array_equal(
            observation.self_features[living, AGENT_FEATURE_CURRENT_HEALTH],
            state.current_health[living],
        )
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
        team_a = ActorAction(
            *(
                jnp.asarray(commands[tick][head][:5], dtype=jnp.int32)
                for head in ActorAction._fields
            )
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
        before = state
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
        trajectory.append(WitnessTransition(before, joint, state, reward, done, info))
    return trajectory


def test_scenario_5_fixture_preserves_the_approved_physical_revision() -> None:
    # Keep the original r9 fixture separate from the current packaged solution.
    scenario = load_scenario_5()
    draft = load_scenario_5_draft()
    assert draft.asset_id == "scenario_5"
    assert draft.revision == 9
    assert draft.content.description == draft.content.notes == ""
    assert len(draft.content.embedded_map.obstacles) == 7
    assert scenario.semantic_digest == SCENARIO_5_SEMANTIC_DIGEST
    assert int(scenario.initial_state.step_count) == 295
    assert scenario.config.max_steps == 300
    assert not bool(scenario.initial_state.has_previous_timestep_joint_action)


def test_coordinated_healing_witness_wins_on_the_final_transition(
    scenario: TDMScenario,
) -> None:
    trajectory = _run_witness(scenario)
    repeated = _run_witness(scenario)
    for first, second in zip(
        jax.tree.leaves(trajectory), jax.tree.leaves(repeated), strict=True
    ):
        np.testing.assert_array_equal(first, second)

    expected_scores = ((17, 19), (17, 19), (18, 19), (18, 19), (20, 19))
    expected_health = (
        (15, 44, 1),
        (0.575001, 44, 1),
        (80, 13.4, 1),
        (57.575001, 7.199999, 1),
        (35.150002, 0.999998, 1),
    )
    for tick, transition in enumerate(trajectory):
        deaths = transition.info.transition_facts.death_facts.is_newly_dead_by_recipient
        assert not bool(jnp.any(deaths[:5]))
        np.testing.assert_array_equal(
            transition.after.team_deathmatch_scores, expected_scores[tick]
        )
        np.testing.assert_allclose(
            transition.after.current_health[jnp.asarray([0, 1, 4])],
            expected_health[tick],
            rtol=0,
            atol=1e-5,
        )
        assert bool(transition.done.terminated) == (tick == 4)
        assert bool(transition.done.truncated) == (tick == 4)
        np.testing.assert_array_equal(
            transition.reward.rewards[:5], [1 if tick == 4 else 0] * 5
        )
        # Priest-B never uses Ultimate: cooldown reaches zero only after death.
        assert int(transition.action.use_ultimate[9]) == 0
    assert [int(t.before.ultimate_cooldowns[9]) for t in trajectory] == [3, 2, 1, 0, 0]
    assert float(trajectory[2].before.current_health[0]) == pytest.approx(
        0.575001, abs=1e-5
    )
    assert bool(
        trajectory[2].info.transition_facts.death_facts.is_newly_dead_by_recipient[9]
    )
    assert not bool(trajectory[3].before.alive_mask[9])
    assert int(trajectory[3].before.stun_durations[1, STUN_CHANNEL_ROGUE_POISON]) == 1
    assert tuple(int(leaf[1]) for leaf in trajectory[3].action) == (0, 0, 0)
    final = trajectory[-1]
    assert int(final.after.step_count) == 300
    assert bool(
        final.info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent[
            9
        ]
    )
    assert bool(final.after.alive_mask[9])
    assert not bool(
        final.info.transition_facts.death_facts.is_newly_dead_by_recipient[9]
    )
    np.testing.assert_array_equal(
        final.info.transition_facts.death_facts.is_newly_dead_by_recipient[5:],
        [True, False, False, True, False],
    )
