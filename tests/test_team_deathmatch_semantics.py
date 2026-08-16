"""Public trajectory proofs for authoritative Team Deathmatch semantics."""

from collections.abc import Iterable
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import initialize_scenario_state, reset, step
from marl_battlegrounds.core.types import (
    CLASS_NEUTRAL,
    CONTEXT_FEATURE_IS_TDM,
    CONTEXT_FEATURE_TDM_ALLY_SCORE,
    CONTEXT_FEATURE_TDM_ENEMY_SCORE,
    CONTEXT_FEATURE_TDM_SCORE_THRESHOLD,
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_STAY,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    OBSTACLE_FEATURES,
    TASK_MODE_NEUTRAL,
    TASK_MODE_OUTCOME_DRAW,
    TASK_MODE_OUTCOME_ONGOING,
    TASK_MODE_OUTCOME_TEAM_A_WIN,
    TASK_MODE_OUTCOME_TEAM_B_WIN,
    TASK_MODE_TDM,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)

_TEAM_A_FIRST_SLOT = 0
_TEAM_A_SECOND_SLOT = 1
_TEAM_B_FIRST_SLOT = MAX_AGENTS_PER_TEAM
_TEAM_B_SECOND_SLOT = MAX_AGENTS_PER_TEAM + 1
_TARGET_NONE = 0
_FIRST_ENEMY_TARGET = 1 + MAX_AGENTS_PER_TEAM
_SECOND_ENEMY_TARGET = _FIRST_ENEMY_TARGET + 1


def _task_config(
    *,
    team_sizes: tuple[int, int] = (1, 1),
    task_mode: int = TASK_MODE_TDM,
    score_threshold: int = 5,
    max_steps: int = 20,
    respawn_periods: tuple[int, int] = (5, 5),
) -> EnvConfig:
    """Build a catalog-valid deterministic task configuration."""
    class_ids = jnp.full((MAX_AGENT_SLOTS,), CLASS_NEUTRAL, dtype=jnp.int32)
    class_ids = class_ids.at[: team_sizes[0]].set(HUNTER_CLASS_ID)
    class_ids = class_ids.at[
        MAX_AGENTS_PER_TEAM : MAX_AGENTS_PER_TEAM + team_sizes[1]
    ].set(HUNTER_CLASS_ID)
    profile = resolve_agent_profile(
        class_ids,
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )

    spawn_positions = jnp.zeros(
        (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32
    )
    for local_slot in range(MAX_AGENTS_PER_TEAM):
        y_position = 2.0 + 2.0 * local_slot
        spawn_positions = spawn_positions.at[local_slot].set((2.0, y_position))
        spawn_positions = spawn_positions.at[MAX_AGENTS_PER_TEAM + local_slot].set(
            (10.0, y_position)
        )

    return EnvConfig(
        task_mode=task_mode,
        team_deathmatch_score_threshold=score_threshold,
        max_steps=max_steps,
        map_width=12.0,
        map_height=12.0,
        obstacles=jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32),
        agent_profile=profile,
        ordinary_movement_distance_scale=1.0,
        team_spawn_pad_positions=spawn_positions.reshape(
            (2, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS)
        ),
        spawn_shield_duration_steps=0,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray(
            respawn_periods, dtype=jnp.int32
        ),
    )


def _combat_positions(team_sizes: tuple[int, int]) -> Array:
    """Place opposing active rows within Hunter Basic range without overlap."""
    positions = jnp.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
    for local_slot in range(team_sizes[0]):
        positions = positions.at[local_slot].set((4.0, 2.0 + 2.0 * local_slot))
    for local_slot in range(team_sizes[1]):
        positions = positions.at[MAX_AGENTS_PER_TEAM + local_slot].set(
            (7.0, 2.0 + 2.0 * local_slot)
        )
    return positions


def _scenario(
    config: EnvConfig,
    *,
    scores: tuple[int, int] = (0, 0),
    step_count: int = 0,
    low_health_slots: Iterable[int] = (),
    dead_slots: Iterable[int] = (),
    due_respawn_teams: Iterable[int] = (),
) -> tuple[EnvState, Observation, ActionMask, Info]:
    """Author one valid preterminal TDM state and expose its paired mask."""
    state, _, _, _ = reset(config, jax.random.key(0))
    team_sizes = (
        int(jnp.sum(config.agent_profile.active_mask[:MAX_AGENTS_PER_TEAM])),
        int(jnp.sum(config.agent_profile.active_mask[MAX_AGENTS_PER_TEAM:])),
    )
    state = state._replace(
        team_deathmatch_scores=jnp.asarray(scores, dtype=jnp.int32),
        step_count=jnp.asarray(step_count, dtype=jnp.int32),
        agent_positions=_combat_positions(team_sizes),
    )
    for slot in low_health_slots:
        state = state._replace(current_health=state.current_health.at[slot].set(1.0))
    for slot in dead_slots:
        state = state._replace(
            alive_mask=state.alive_mask.at[slot].set(False),
            current_health=state.current_health.at[slot].set(0.0),
        )
    for team_index in due_respawn_teams:
        state = state._replace(
            team_respawn_wave_countdowns=(
                state.team_respawn_wave_countdowns.at[team_index].set(0)
            )
        )
    return initialize_scenario_state(state, config)


def _joint_action(*target_rows: tuple[int, int]) -> Action:
    """Build a canonical joint action with selected Basic targets."""
    target_actions = jnp.full((MAX_AGENT_SLOTS,), _TARGET_NONE, dtype=jnp.int32)
    for actor_slot, target_action in target_rows:
        target_actions = target_actions.at[actor_slot].set(target_action)
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=target_actions,
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _take_step(
    config: EnvConfig,
    state: EnvState,
    action_mask: ActionMask,
    action: Action,
    *,
    key_index: int = 1,
) -> tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]:
    """Advance one deterministic transition from a paired state and mask."""
    return step(config, state, action_mask, action, jax.random.key(key_index))


def _assert_task_result(
    reward: Reward,
    done_flags: DoneFlags,
    info: Info,
    *,
    outcome: int,
    terminated: bool,
    truncated: bool,
) -> None:
    """Assert the three public result surfaces agree on one transition."""
    assert int(info.transition_facts.team_deathmatch_facts.outcome) == outcome
    assert bool(done_flags.terminated) is terminated
    assert bool(done_flags.truncated) is truncated
    assert reward.rewards.shape == (MAX_AGENT_SLOTS,)
    assert reward.rewards.dtype == jnp.float32


def test_reset_exposes_zero_score_task_context_and_neutral_outcome() -> None:
    """Reset publishes task configuration and canonical zero dynamic truth."""
    config = _task_config(team_sizes=(2, 1), score_threshold=7)
    state, observation, action_mask, info = reset(config, jax.random.key(2))

    assert state.team_deathmatch_scores.shape == (2,)
    assert state.team_deathmatch_scores.dtype == jnp.int32
    assert bool(jnp.all(state.team_deathmatch_scores == 0))
    assert int(info.transition_facts.team_deathmatch_facts.outcome) == (
        TASK_MODE_OUTCOME_ONGOING
    )
    assert info.transition_facts.team_deathmatch_facts.outcome.dtype == jnp.int32
    assert bool(jnp.all(observation.context_features[:2, CONTEXT_FEATURE_IS_TDM] == 1))
    assert bool(
        jnp.all(
            observation.context_features[:2, CONTEXT_FEATURE_TDM_SCORE_THRESHOLD] == 7
        )
    )
    assert bool(jnp.all(observation.context_features[2:5] == 0.0))
    assert bool(jnp.all(observation.context_features[6:] == 0.0))
    assert action_mask.move_mask.shape == (MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS)
    assert action_mask.select_target_mask.shape == (
        MAX_AGENT_SLOTS,
        NUM_TARGET_ACTIONS,
    )
    assert action_mask.use_ultimate_mask.shape == (
        MAX_AGENT_SLOTS,
        NUM_ULTIMATE_ACTIONS,
    )


@pytest.mark.parametrize(
    ("victim_slot", "actor_slot", "expected_scores"),
    (
        pytest.param(
            _TEAM_B_FIRST_SLOT,
            _TEAM_A_FIRST_SLOT,
            (1, 0),
            id="team-b-death-scores-for-team-a",
        ),
        pytest.param(
            _TEAM_A_FIRST_SLOT,
            _TEAM_B_FIRST_SLOT,
            (0, 1),
            id="team-a-death-scores-for-team-b",
        ),
    ),
)
def test_one_new_recipient_death_scores_once_for_the_opposing_team(
    victim_slot: int,
    actor_slot: int,
    expected_scores: tuple[int, int],
) -> None:
    """Score authority follows recipient death rather than contributor identity."""
    config = _task_config(score_threshold=5)
    state, _, action_mask, _ = _scenario(
        config,
        low_health_slots=(victim_slot,),
    )

    next_state, observation, reward, done_flags, _, info = _take_step(
        config,
        state,
        action_mask,
        _joint_action((actor_slot, _FIRST_ENEMY_TARGET)),
    )

    assert bool(
        jnp.array_equal(
            next_state.team_deathmatch_scores,
            jnp.asarray(expected_scores, dtype=jnp.int32),
        )
    )
    assert bool(
        info.transition_facts.death_facts.is_newly_dead_by_recipient[victim_slot]
    )
    assert bool(
        jnp.array_equal(
            observation.context_features[
                _TEAM_A_FIRST_SLOT,
                CONTEXT_FEATURE_TDM_ALLY_SCORE : CONTEXT_FEATURE_TDM_ENEMY_SCORE + 1,
            ],
            jnp.asarray(expected_scores, dtype=jnp.float32),
        )
    )
    assert bool(
        jnp.array_equal(
            observation.context_features[
                _TEAM_B_FIRST_SLOT,
                CONTEXT_FEATURE_TDM_ALLY_SCORE : CONTEXT_FEATURE_TDM_ENEMY_SCORE + 1,
            ],
            jnp.asarray(expected_scores[::-1], dtype=jnp.float32),
        )
    )
    _assert_task_result(
        reward,
        done_flags,
        info,
        outcome=TASK_MODE_OUTCOME_ONGOING,
        terminated=False,
        truncated=False,
    )
    assert bool(jnp.all(reward.rewards == 0.0))


def test_bilateral_multi_death_updates_both_scores_before_outcome() -> None:
    """Several simultaneous recipient deaths reduce into one order-free score."""
    config = _task_config(team_sizes=(2, 2), score_threshold=5)
    state, _, action_mask, _ = _scenario(
        config,
        low_health_slots=(
            _TEAM_A_FIRST_SLOT,
            _TEAM_B_FIRST_SLOT,
            _TEAM_B_SECOND_SLOT,
        ),
    )

    next_state, _, reward, done_flags, _, info = _take_step(
        config,
        state,
        action_mask,
        _joint_action(
            (_TEAM_A_FIRST_SLOT, _FIRST_ENEMY_TARGET),
            (_TEAM_A_SECOND_SLOT, _SECOND_ENEMY_TARGET),
            (_TEAM_B_FIRST_SLOT, _FIRST_ENEMY_TARGET),
        ),
    )

    assert bool(
        jnp.array_equal(
            next_state.team_deathmatch_scores,
            jnp.asarray((2, 1), dtype=jnp.int32),
        )
    )
    expected_deaths = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.bool_)
    expected_deaths = expected_deaths.at[
        jnp.asarray(
            (_TEAM_A_FIRST_SLOT, _TEAM_B_FIRST_SLOT, _TEAM_B_SECOND_SLOT),
            dtype=jnp.int32,
        )
    ].set(True)
    assert bool(
        jnp.array_equal(
            info.transition_facts.death_facts.is_newly_dead_by_recipient,
            expected_deaths,
        )
    )
    _assert_task_result(
        reward,
        done_flags,
        info,
        outcome=TASK_MODE_OUTCOME_ONGOING,
        terminated=False,
        truncated=False,
    )


def test_dead_body_does_not_score_again_on_a_later_transition() -> None:
    """Only the alive-to-dead edge can change Team Deathmatch score."""
    config = _task_config(score_threshold=5)
    state, _, action_mask, _ = _scenario(
        config,
        low_health_slots=(_TEAM_B_FIRST_SLOT,),
    )
    dead_state, _, _, _, dead_mask, _ = _take_step(
        config,
        state,
        action_mask,
        _joint_action((_TEAM_A_FIRST_SLOT, _FIRST_ENEMY_TARGET)),
    )

    next_state, _, reward, done_flags, _, info = _take_step(
        config,
        dead_state,
        dead_mask,
        _joint_action((_TEAM_A_FIRST_SLOT, _FIRST_ENEMY_TARGET)),
        key_index=2,
    )

    assert bool(
        jnp.array_equal(
            next_state.team_deathmatch_scores,
            jnp.asarray((1, 0), dtype=jnp.int32),
        )
    )
    assert not bool(
        jnp.any(info.transition_facts.death_facts.is_newly_dead_by_recipient)
    )
    _assert_task_result(
        reward,
        done_flags,
        info,
        outcome=TASK_MODE_OUTCOME_ONGOING,
        terminated=False,
        truncated=False,
    )


def test_due_respawn_wave_does_not_erase_a_different_new_death_score() -> None:
    """Retained death facts score even when lifecycle also respawns a teammate."""
    config = _task_config(team_sizes=(2, 2), score_threshold=5)
    state, _, action_mask, _ = _scenario(
        config,
        low_health_slots=(_TEAM_B_FIRST_SLOT,),
        dead_slots=(_TEAM_B_SECOND_SLOT,),
        due_respawn_teams=(1,),
    )

    next_state, _, _, _, _, info = _take_step(
        config,
        state,
        action_mask,
        _joint_action((_TEAM_A_FIRST_SLOT, _FIRST_ENEMY_TARGET)),
    )

    assert bool(
        jnp.array_equal(
            next_state.team_deathmatch_scores,
            jnp.asarray((1, 0), dtype=jnp.int32),
        )
    )
    assert bool(
        info.transition_facts.death_facts.is_newly_dead_by_recipient[_TEAM_B_FIRST_SLOT]
    )
    assert bool(
        info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent[
            _TEAM_B_SECOND_SLOT
        ]
    )
    assert not bool(next_state.alive_mask[_TEAM_B_FIRST_SLOT])
    assert bool(next_state.alive_mask[_TEAM_B_SECOND_SLOT])


@pytest.mark.parametrize(
    ("winning_team", "expected_outcome"),
    (
        pytest.param("team-a", TASK_MODE_OUTCOME_TEAM_A_WIN, id="team-a"),
        pytest.param("team-b", TASK_MODE_OUTCOME_TEAM_B_WIN, id="team-b"),
    ),
)
def test_threshold_win_emits_one_shared_reward_pulse_for_configured_team(
    winning_team: str,
    expected_outcome: int,
) -> None:
    """Threshold completion rewards configured membership, not alive state."""
    config = _task_config(team_sizes=(2, 1), score_threshold=1)
    victim_slot = _TEAM_B_FIRST_SLOT if winning_team == "team-a" else _TEAM_A_FIRST_SLOT
    actor_slot = _TEAM_A_FIRST_SLOT if winning_team == "team-a" else _TEAM_B_FIRST_SLOT
    dead_teammate_slots = (_TEAM_A_SECOND_SLOT,) if winning_team == "team-a" else ()
    state, _, action_mask, _ = _scenario(
        config,
        low_health_slots=(victim_slot,),
        dead_slots=dead_teammate_slots,
    )

    next_state, observation, reward, done_flags, next_mask, info = _take_step(
        config,
        state,
        action_mask,
        _joint_action((actor_slot, _FIRST_ENEMY_TARGET)),
    )

    _assert_task_result(
        reward,
        done_flags,
        info,
        outcome=expected_outcome,
        terminated=True,
        truncated=False,
    )
    if winning_team == "team-a":
        assert bool(jnp.array_equal(reward.rewards[:2], jnp.ones((2,))))
        assert float(reward.rewards[_TEAM_B_FIRST_SLOT]) == -1.0
        expected_scores = (1, 0)
    else:
        assert float(reward.rewards[_TEAM_A_FIRST_SLOT]) == -1.0
        assert float(reward.rewards[_TEAM_A_SECOND_SLOT]) == -1.0
        assert float(reward.rewards[_TEAM_B_FIRST_SLOT]) == 1.0
        expected_scores = (0, 1)
    inactive_mask = jnp.logical_not(config.agent_profile.active_mask)
    assert bool(jnp.all(reward.rewards[inactive_mask] == 0.0))
    assert bool(
        jnp.array_equal(
            next_state.team_deathmatch_scores,
            jnp.asarray(expected_scores, dtype=jnp.int32),
        )
    )
    assert bool(
        jnp.array_equal(
            observation.context_features[
                actor_slot,
                CONTEXT_FEATURE_TDM_ALLY_SCORE : CONTEXT_FEATURE_TDM_ENEMY_SCORE + 1,
            ],
            jnp.asarray(
                expected_scores if winning_team == "team-a" else expected_scores[::-1],
                dtype=jnp.float32,
            ),
        )
    )
    assert not bool(next_state.alive_mask[victim_slot])
    assert bool(next_mask.move_mask[victim_slot, MOVE_STAY])


def test_equal_simultaneous_threshold_crossing_is_a_draw() -> None:
    """A mutual first-to-K trade compares complete successor scores."""
    config = _task_config(score_threshold=1)
    state, _, action_mask, _ = _scenario(
        config,
        low_health_slots=(_TEAM_A_FIRST_SLOT, _TEAM_B_FIRST_SLOT),
    )

    next_state, _, reward, done_flags, _, info = _take_step(
        config,
        state,
        action_mask,
        _joint_action(
            (_TEAM_A_FIRST_SLOT, _FIRST_ENEMY_TARGET),
            (_TEAM_B_FIRST_SLOT, _FIRST_ENEMY_TARGET),
        ),
    )

    assert bool(
        jnp.array_equal(
            next_state.team_deathmatch_scores,
            jnp.asarray((1, 1), dtype=jnp.int32),
        )
    )
    _assert_task_result(
        reward,
        done_flags,
        info,
        outcome=TASK_MODE_OUTCOME_DRAW,
        terminated=True,
        truncated=False,
    )
    assert bool(jnp.all(reward.rewards == 0.0))


def test_unequal_bilateral_threshold_overshoot_selects_higher_successor_score() -> None:
    """Both teams may reach K, but simultaneous overshoot remains unclamped."""
    config = _task_config(team_sizes=(2, 2), score_threshold=2)
    state, _, action_mask, _ = _scenario(
        config,
        scores=(1, 1),
        low_health_slots=(
            _TEAM_A_FIRST_SLOT,
            _TEAM_B_FIRST_SLOT,
            _TEAM_B_SECOND_SLOT,
        ),
    )

    next_state, _, reward, done_flags, _, info = _take_step(
        config,
        state,
        action_mask,
        _joint_action(
            (_TEAM_A_FIRST_SLOT, _FIRST_ENEMY_TARGET),
            (_TEAM_A_SECOND_SLOT, _SECOND_ENEMY_TARGET),
            (_TEAM_B_FIRST_SLOT, _FIRST_ENEMY_TARGET),
        ),
    )

    assert bool(
        jnp.array_equal(
            next_state.team_deathmatch_scores,
            jnp.asarray((3, 2), dtype=jnp.int32),
        )
    )
    _assert_task_result(
        reward,
        done_flags,
        info,
        outcome=TASK_MODE_OUTCOME_TEAM_A_WIN,
        terminated=True,
        truncated=False,
    )
    assert bool(jnp.all(reward.rewards[:2] == 1.0))
    assert bool(jnp.all(reward.rewards[5:7] == -1.0))


@pytest.mark.parametrize(
    "scores",
    (
        pytest.param((2, 0), id="team-a-leads"),
        pytest.param((0, 2), id="team-b-leads"),
        pytest.param((2, 2), id="scores-equal"),
    ),
)
def test_horizon_without_threshold_is_always_a_draw(
    scores: tuple[int, int],
) -> None:
    """First-to-K semantics do not convert a horizon lead into victory."""
    config = _task_config(score_threshold=5, max_steps=3)
    state, _, action_mask, _ = _scenario(
        config,
        scores=scores,
        step_count=2,
    )

    _, _, reward, done_flags, _, info = _take_step(
        config,
        state,
        action_mask,
        _joint_action(),
    )

    _assert_task_result(
        reward,
        done_flags,
        info,
        outcome=TASK_MODE_OUTCOME_DRAW,
        terminated=False,
        truncated=True,
    )
    assert bool(jnp.all(reward.rewards == 0.0))


def test_threshold_on_final_allowed_action_retains_both_completion_bases() -> None:
    """Termination and truncation independently describe one final transition."""
    config = _task_config(score_threshold=1, max_steps=1)
    state, _, action_mask, _ = _scenario(
        config,
        low_health_slots=(_TEAM_B_FIRST_SLOT,),
    )

    _, _, reward, done_flags, _, info = _take_step(
        config,
        state,
        action_mask,
        _joint_action((_TEAM_A_FIRST_SLOT, _FIRST_ENEMY_TARGET)),
    )

    _assert_task_result(
        reward,
        done_flags,
        info,
        outcome=TASK_MODE_OUTCOME_TEAM_A_WIN,
        terminated=True,
        truncated=True,
    )


def test_neutral_mode_retains_zero_task_truth_and_horizon_only_truncation() -> None:
    """Team Deathmatch additions do not alter the task-neutral simulator path."""
    config = _task_config(
        task_mode=TASK_MODE_NEUTRAL,
        score_threshold=0,
        max_steps=1,
    )
    state, observation, action_mask, reset_info = reset(config, jax.random.key(3))
    assert bool(jnp.all(state.team_deathmatch_scores == 0))
    assert bool(jnp.all(observation.context_features[:, CONTEXT_FEATURE_IS_TDM] == 0))
    assert int(reset_info.transition_facts.team_deathmatch_facts.outcome) == (
        TASK_MODE_OUTCOME_ONGOING
    )

    next_state, next_observation, reward, done_flags, _, info = _take_step(
        config,
        state,
        action_mask,
        _joint_action(),
    )

    assert bool(jnp.all(next_state.team_deathmatch_scores == 0))
    assert bool(
        jnp.all(next_observation.context_features[:, CONTEXT_FEATURE_IS_TDM] == 0)
    )
    _assert_task_result(
        reward,
        done_flags,
        info,
        outcome=TASK_MODE_OUTCOME_ONGOING,
        terminated=False,
        truncated=True,
    )
    assert bool(jnp.all(reward.rewards == 0.0))


def test_tdm_public_trajectory_is_stable_under_jit_vmap_and_real_scan() -> None:
    """Compiled and batched execution preserve score, fact, and done semantics."""
    config = _task_config(score_threshold=5, max_steps=10)
    state, _, action_mask, _ = _scenario(
        config,
        low_health_slots=(_TEAM_B_FIRST_SLOT,),
    )
    action = _joint_action((_TEAM_A_FIRST_SLOT, _FIRST_ENEMY_TARGET))

    eager = _take_step(config, state, action_mask, action)
    compiled = cast(
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        jax.jit(step)(
            config,
            state,
            action_mask,
            action,
            jax.random.key(1),
        ),
    )
    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager),
        jax.tree_util.tree_leaves(compiled),
        strict=True,
    ):
        assert bool(jnp.array_equal(eager_leaf, compiled_leaf))

    batched = jax.vmap(step, in_axes=(None, None, None, None, 0))(
        config,
        state,
        action_mask,
        action,
        jax.random.split(jax.random.key(4), 3),
    )
    assert bool(
        jnp.all(
            batched[0].team_deathmatch_scores == jnp.asarray((1, 0), dtype=jnp.int32)
        )
    )
    assert bool(
        jnp.all(
            batched[-1].transition_facts.team_deathmatch_facts.outcome
            == TASK_MODE_OUTCOME_ONGOING
        )
    )

    def _scan_step(
        carry: tuple[EnvState, ActionMask], key: Array
    ) -> tuple[tuple[EnvState, ActionMask], tuple[EnvState, Info]]:
        current_state, current_mask = carry
        next_state, _, _, _, next_mask, info = step(
            config,
            current_state,
            current_mask,
            action,
            key,
        )
        return (next_state, next_mask), (next_state, info)

    _, (scanned_states, scanned_infos) = jax.lax.scan(
        _scan_step,
        (state, action_mask),
        jax.random.split(jax.random.key(5), 3),
    )
    assert bool(
        jnp.all(
            scanned_states.team_deathmatch_scores
            == jnp.asarray((1, 0), dtype=jnp.int32)
        )
    )
    scanned_deaths = (
        scanned_infos.transition_facts.death_facts.is_newly_dead_by_recipient[
            :, _TEAM_B_FIRST_SLOT
        ]
    )
    assert bool(
        jnp.array_equal(
            scanned_deaths,
            jnp.asarray((True, False, False), dtype=jnp.bool_),
        )
    )
