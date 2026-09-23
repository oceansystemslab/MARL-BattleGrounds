"""Check Scenario 8's current route and seven matched timing interventions."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, cast
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.scenario_controller_fixtures import (
    SCENARIO_8_SEMANTIC_DIGEST,
    load_scenario_8,
    load_scenario_8_draft,
)

from marl_battlegrounds.core.env import initialize_scenario_state, step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    MOVE_EAST,
    MOVE_SOUTH,
    MOVE_SOUTHEAST,
    MOVE_STAY,
    MOVE_WEST,
    STUN_CHANNEL_HUNTER_TRAP,
    STUN_CHANNEL_ROGUE_POISON,
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


class ActionChange(NamedTuple):
    tick: int
    slot: int
    move: int
    target: int
    ultimate: int


@pytest.fixture(scope="module")
def scenario() -> TDMScenario:
    return load_tdm_scenario(8)


def _team_a_action(
    tick: int, command: ActorAction, changes: tuple[ActionChange, ...]
) -> ActorAction:
    heads = list(command)
    for change in changes:
        if change.tick == tick:
            heads = [
                head.at[change.slot].set(value)
                for head, value in zip(heads, change[2:], strict=True)
            ]
    return ActorAction(*heads)


def _run_witness(
    scenario: TDMScenario,
    changes: tuple[ActionChange, ...] = (),
) -> list[WitnessTransition]:
    state, observation, mask, _ = initialize_scenario_state(
        scenario.initial_state, scenario.config
    )
    availability = build_default_shared_obs_information_availability(
        scenario.config.agent_profile.active_mask,
        scenario.config.agent_profile.team_ids,
    )
    expected = json.loads(
        (Path(__file__).parent / "fixtures" / "scenario_8_witness.json").read_text()
    )
    assert scenario.info.source.revision == expected["source_revision"]
    assert scenario.info.source.semantic_digest == expected["semantic_digest"]
    commands = expected["ticks"]
    assert len(commands) == 5
    trajectory: list[WitnessTransition] = []
    slots = jnp.arange(10)
    for tick in range(1, 6):
        living = state.alive_mask
        np.testing.assert_array_equal(
            observation.self_features[living, AGENT_FEATURE_X : AGENT_FEATURE_Y + 1],
            state.agent_positions[living],
        )
        np.testing.assert_array_equal(
            observation.self_features[living, AGENT_FEATURE_CURRENT_HEALTH],
            state.current_health[living],
        )
        # State is inspected only by the proof and simulator; the real BETA
        # receives no raw state, pending Team A actions or future information.
        bank = build_shared_obs_sensor_source_bank(observation)
        team_b = _TEAM(
            observation,
            mask,
            jax.random.split(jax.random.key(tick - 1), 10),
            bank,
            availability,
            reactive_tdm_beta_policy,
            TEAM_B_ID,
        )
        command = ActorAction(
            *(
                jnp.asarray(commands[tick - 1][head][:5], dtype=jnp.int32)
                for head in ActorAction._fields
            )
        )
        joint = build_joint_action_from_actor_actions(
            _team_a_action(tick, command, changes), team_b
        )
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
            scenario.config, state, mask, joint, jax.random.key(tick - 1)
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
def trajectory(scenario: TDMScenario) -> list[WitnessTransition]:
    return _run_witness(scenario)


def test_scenario_8_fixture_binds_approved_revision_14_and_current_beta() -> None:
    # Keep the old immutable fixture separate from the current packaged route.
    scenario = load_scenario_8()
    draft = load_scenario_8_draft()
    assert draft.asset_id == "scenario_8"
    assert draft.revision == 14
    assert draft.content.description == draft.content.notes == ""
    assert len(draft.content.embedded_map.obstacles) == 4
    assert scenario.semantic_digest == SCENARIO_8_SEMANTIC_DIGEST
    assert reactive_tdm_beta_controller_descriptor()["version"] == 5
    np.testing.assert_array_equal(
        scenario.initial_state.team_deathmatch_scores, [18, 19]
    )
    np.testing.assert_array_equal(
        scenario.initial_state.current_health, [0, 1, 0, 21, 1, 0, 0, 0, 56, 9]
    )
    assert int(scenario.initial_state.step_count) == 295
    assert scenario.config.max_steps == 300
    assert not bool(scenario.initial_state.has_previous_timestep_joint_action)


def test_witness_repeats_and_wins_at_truncation_without_a_team_a_death(
    scenario: TDMScenario, trajectory: list[WitnessTransition]
) -> None:
    repeated = _run_witness(scenario)
    for first, second in zip(
        jax.tree.leaves(trajectory), jax.tree.leaves(repeated), strict=True
    ):
        np.testing.assert_array_equal(first, second)
    assert len(trajectory) == 5
    scores = ((18, 19), (18, 19), (18, 19), (19, 19), (20, 19))
    rogue_health = (29, 1, 5, 93, 85)
    killed = ((), (), (), (9,), (8,))
    for index, transition in enumerate(trajectory):
        facts = transition.info.transition_facts
        deaths = facts.death_facts.is_newly_dead_by_recipient
        np.testing.assert_array_equal(np.flatnonzero(deaths), killed[index])
        assert not bool(jnp.any(deaths[:5]))
        np.testing.assert_array_equal(
            transition.after.team_deathmatch_scores, scores[index]
        )
        assert float(transition.after.current_health[3]) == pytest.approx(
            rogue_health[index], rel=0, abs=1e-5
        )
        assert bool(transition.done.terminated) == (index == 4)
        assert bool(transition.done.truncated) == (index == 4)
        np.testing.assert_array_equal(
            transition.reward.rewards[:5], [1 if index == 4 else 0] * 5
        )
        respawned = facts.respawn_facts.was_respawned_this_transition_by_agent
        np.testing.assert_array_equal(
            np.flatnonzero(respawned), (0, 2, 5, 6, 7, 9) if index == 4 else ()
        )
    final = trajectory[-1]
    assert int(final.after.step_count) == 300
    np.testing.assert_allclose(
        final.after.current_health[jnp.asarray([1, 3, 4])],
        [1, 85, 1],
        rtol=0,
        atol=1e-5,
    )
    assert not bool(trajectory[3].after.alive_mask[9])
    assert bool(final.after.alive_mask[9])
    assert not bool(final.after.alive_mask[8])


def test_team_b_responses_are_generated_by_beta(
    trajectory: list[WitnessTransition],
) -> None:
    rogue_moves = (MOVE_SOUTH, MOVE_SOUTH, MOVE_STAY, MOVE_SOUTH, MOVE_SOUTHEAST)
    rogue_targets = (0, 9, 0, 9, 9)
    for index, transition in enumerate(trajectory):
        np.testing.assert_array_equal(
            transition.action.move[5:], [0, 0, 0, rogue_moves[index], 0]
        )
        np.testing.assert_array_equal(
            transition.action.select_target[5:], [0, 0, 0, rogue_targets[index], 0]
        )
        np.testing.assert_array_equal(
            transition.action.use_ultimate[5:], [0, 0, 0, int(index == 1), 0]
        )


def test_poison_is_immediate_damage_then_observable_stun_and_anti_heal(
    trajectory: list[WitnessTransition],
) -> None:
    _, second, third, fourth, _ = trajectory
    combat = second.info.transition_facts.combat_transition_facts
    for slot in (3, 8):
        assert bool(combat.ultimate_effect_is_activated_by_source[slot])
        assert not bool(combat.basic_effect_is_activated_by_source[slot])
        assert float(combat.total_effective_damage_by_recipient[slot]) == 36
        assert bool(combat.rogue_poison_anti_heal_is_applied_by_source[slot])
        assert int(third.before.stun_durations[slot, STUN_CHANNEL_ROGUE_POISON]) == 1
        assert int(third.before.rogue_poison_anti_heal_durations[slot]) == 4
        assert tuple(int(head[slot]) for head in third.action) == (MOVE_STAY, 0, 0)
        assert int(fourth.before.stun_durations[slot, STUN_CHANNEL_ROGUE_POISON]) == 0
    # Effects selected on tick 2 use tick-2 status: newly inflicted anti-heal
    # reduces later healing, not the already selected heal on the Poison tick.
    assert float(combat.total_effective_healing_by_recipient[3]) == 8
    later = third.info.transition_facts.combat_transition_facts
    np.testing.assert_array_equal(later.total_effective_damage_by_recipient, [0] * 10)
    assert float(later.total_effective_healing_by_recipient[3]) == 4
    assert float(third.after.current_health[3]) == 5
    assert float(third.after.current_health[8]) == 20


def _assert_matched_intervention(
    actual: list[WitnessTransition],
    baseline: list[WitnessTransition],
    changes: tuple[ActionChange, ...],
) -> None:
    first_tick = min(change.tick for change in changes)
    for index, transition in enumerate(actual):
        if index < first_tick - 1:
            for first, second in zip(
                jax.tree.leaves(transition),
                jax.tree.leaves(baseline[index]),
                strict=True,
            ):
                np.testing.assert_array_equal(first, second)
        expected = [head[:5] for head in baseline[index].action]
        for change in changes:
            if change.tick == index + 1:
                expected = [
                    head.at[change.slot].set(value)
                    for head, value in zip(expected, change[2:], strict=True)
                ]
        for submitted, intended in zip(transition.action, expected, strict=True):
            np.testing.assert_array_equal(submitted[:5], intended)
    first_actual, first_baseline = actual[first_tick - 1], baseline[first_tick - 1]
    for first, second in zip(
        jax.tree.leaves(
            (first_actual.before, first_actual.observation, first_actual.mask)
        ),
        jax.tree.leaves(
            (first_baseline.before, first_baseline.observation, first_baseline.mask)
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(first, second)
    for first, second in zip(first_actual.action, first_baseline.action, strict=True):
        np.testing.assert_array_equal(first[5:], second[5:])


def test_preserving_trap_prevents_the_enemy_healers_early_self_ultimate(
    scenario: TDMScenario, trajectory: list[WitnessTransition]
) -> None:
    changes = (ActionChange(1, 1, MOVE_STAY, 10, 0),)
    actual = _run_witness(scenario, changes)
    _assert_matched_intervention(actual, trajectory, changes)
    for transition in trajectory[:4]:
        assert int(transition.before.stun_durations[9, STUN_CHANNEL_HUNTER_TRAP]) > 0
        assert tuple(int(head[9]) for head in transition.action) == (MOVE_STAY, 0, 0)
    assert int(trajectory[1].before.ultimate_cooldowns[9]) == 0
    assert int(actual[1].before.stun_durations[9, STUN_CHANNEL_HUNTER_TRAP]) == 0
    assert float(actual[1].before.current_health[9]) == 1
    assert int(actual[1].action.select_target[9]) == 5
    assert int(actual[1].action.use_ultimate[9]) == 1
    assert float(actual[1].after.current_health[9]) == 100
    assert len(actual) == 5
    final = actual[-1]
    np.testing.assert_array_equal(final.after.team_deathmatch_scores, [19, 19])
    np.testing.assert_array_equal(final.reward.rewards, [0] * 10)
    assert not bool(final.done.terminated)
    assert bool(final.done.truncated)


def test_late_charge_cannot_cancel_the_enemy_healers_precommitted_self_heal(
    scenario: TDMScenario, trajectory: list[WitnessTransition]
) -> None:
    changes = (
        ActionChange(4, 1, MOVE_STAY, 0, 0),
        ActionChange(5, 1, MOVE_STAY, 10, 1),
    )
    actual = _run_witness(scenario, changes)
    _assert_matched_intervention(actual, trajectory, changes)
    assert len(actual) == 5
    final = actual[-1]
    assert int(final.before.stun_durations[9, STUN_CHANNEL_HUNTER_TRAP]) == 0
    assert tuple(int(head[1]) for head in final.action) == (MOVE_STAY, 10, 1)
    assert tuple(int(head[9]) for head in final.action) == (MOVE_SOUTH, 5, 1)
    # Both effects were chosen before this turn's Charge. The healer now heals
    # itself; this route does not show a dying healer saving its ally.
    combat = final.info.transition_facts.combat_transition_facts
    assert bool(combat.ultimate_effect_is_activated_by_source[1])
    assert int(combat.combat_effect_recipient_global_slot_by_source[1]) == 9
    assert bool(
        combat.stun_is_applied_by_source_and_channel[1, STUN_CHANNEL_WARRIOR_CHARGE]
    )
    assert bool(combat.ultimate_effect_is_activated_by_source[9])
    assert int(combat.combat_effect_recipient_global_slot_by_source[9]) == 9
    assert float(combat.total_effective_healing_by_recipient[9]) == 200
    assert float(final.after.current_health[9]) == 100
    assert float(combat.total_effective_healing_by_recipient[8]) == 0
    assert bool(final.info.transition_facts.death_facts.is_newly_dead_by_recipient[8])
    assert not bool(
        final.info.transition_facts.death_facts.is_newly_dead_by_recipient[9]
    )
    assert not bool(final.after.alive_mask[8])
    assert bool(final.after.alive_mask[9])
    np.testing.assert_array_equal(final.after.team_deathmatch_scores, [19, 19])
    np.testing.assert_array_equal(final.reward.rewards, [0] * 10)
    assert not bool(final.done.terminated)
    assert bool(final.done.truncated)


@pytest.mark.parametrize(
    ("change", "terminal_tick", "score"),
    (
        pytest.param(
            ActionChange(2, 3, MOVE_WEST, 9, 0), 3, (18, 20), id="basic-not-poison"
        ),
        pytest.param(
            ActionChange(1, 4, MOVE_SOUTH, 5, 0), 2, (18, 20), id="self-heal-tick-1"
        ),
        pytest.param(
            ActionChange(2, 4, MOVE_SOUTH, 5, 0), 2, (18, 20), id="self-heal-tick-2"
        ),
        pytest.param(
            ActionChange(4, 4, MOVE_EAST, 5, 1), 4, (19, 20), id="self-ultimate"
        ),
    ),
)
def test_matched_damage_or_healing_change_loses_rogue_a(
    scenario: TDMScenario,
    trajectory: list[WitnessTransition],
    change: ActionChange,
    terminal_tick: int,
    score: tuple[int, int],
) -> None:
    actual = _run_witness(scenario, (change,))
    _assert_matched_intervention(actual, trajectory, (change,))
    assert len(actual) == terminal_tick
    final = actual[-1]
    assert int(final.after.step_count) == 295 + terminal_tick
    assert bool(final.done.terminated)
    assert not bool(final.done.truncated)
    np.testing.assert_array_equal(final.after.team_deathmatch_scores, score)
    np.testing.assert_array_equal(final.reward.rewards[:5], [-1] * 5)
    deaths = final.info.transition_facts.death_facts.is_newly_dead_by_recipient
    np.testing.assert_array_equal(np.flatnonzero(deaths[:5]), [3])
    assert not bool(final.after.alive_mask[3])


def test_tick_3_heal_is_not_necessary_for_this_matched_win(
    scenario: TDMScenario, trajectory: list[WitnessTransition]
) -> None:
    changes = (ActionChange(3, 4, MOVE_SOUTH, 0, 0),)
    actual = _run_witness(scenario, changes)
    _assert_matched_intervention(actual, trajectory, changes)
    assert len(actual) == 5
    for transition in actual:
        assert not bool(
            jnp.any(
                transition.info.transition_facts.death_facts.is_newly_dead_by_recipient[
                    :5
                ]
            )
        )
    final = actual[-1]
    assert bool(final.done.terminated) and bool(final.done.truncated)
    np.testing.assert_array_equal(final.after.team_deathmatch_scores, [20, 19])
    np.testing.assert_array_equal(final.reward.rewards[:5], [1] * 5)
    assert float(final.after.current_health[3]) == pytest.approx(81, rel=0, abs=1e-5)


def test_each_decision_has_one_current_bank_policy_assembly_and_transition(
    scenario: TDMScenario,
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
        policy_args = calls.policy.call_args_list[tick].args
        assert policy_args[0] is transition.observation
        assert policy_args[1] is transition.mask
        assert policy_args[5] is reactive_tdm_beta_policy
        assert policy_args[6] == TEAM_B_ID
        assert calls.step.call_args_list[tick].args[1] is transition.before
        assert calls.step.call_args_list[tick].args[2] is transition.mask
        assert calls.step.call_args_list[tick].args[3] is transition.action
