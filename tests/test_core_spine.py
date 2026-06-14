"""Core simulator spine contract tests."""

from typing import cast

import jax.numpy as jnp
import jax.random
import pytest
from jax import Array

from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    NUM_MOVE_ACTIONS,
    NUM_OBSERVATION_FEATURES,
    NUM_TARGET_ACTIONS,
    NUM_TEAMS,
    NUM_ULTIMATE_ACTIONS,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)

VALID_TEAM_SIZES = (1, 2, 3, 4, 5)


def _bool_vector(values: tuple[int, ...]) -> Array:
    return jnp.array(values, dtype=bool)


def _int_vector(values: tuple[int, ...]) -> Array:
    return jnp.array(values, dtype=jnp.int32)


def _zero_action() -> Action:
    return Action(
        move=jnp.zeros(shape=(MAX_AGENT_SLOTS,), dtype=jnp.int32),
        target=jnp.zeros(shape=(MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros(shape=(MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _assert_all_action_heads_gate_padded_rows(
    action_mask: ActionMask,
    active_indices: Array,
    inactive_indices: Array,
) -> None:
    assert jnp.all(action_mask.move[active_indices, :])
    assert not jnp.any(action_mask.move[inactive_indices, :])

    assert jnp.all(action_mask.target[active_indices, :])
    assert not jnp.any(action_mask.target[inactive_indices, :])

    assert jnp.all(action_mask.use_ultimate[active_indices, :])
    assert not jnp.any(action_mask.use_ultimate[inactive_indices, :])


def test_static_shape_constants_are_consistent() -> None:
    assert NUM_TEAMS == 2
    assert MAX_AGENTS_PER_TEAM == 5
    assert MAX_AGENT_SLOTS == NUM_TEAMS * MAX_AGENTS_PER_TEAM
    assert NUM_TARGET_ACTIONS == MAX_AGENT_SLOTS + 1
    assert NUM_MOVE_ACTIONS == 9
    assert NUM_ULTIMATE_ACTIONS == 2
    assert ENVIRONMENT_DIMENSIONS == 2


def test_env_config_stores_static_episode_settings() -> None:
    env_config = EnvConfig(team_size=5, max_steps=10000)

    assert env_config.team_size == 5
    assert env_config.max_steps == 10000


def test_env_state_stores_slot_aligned_arrays() -> None:
    env_state = EnvState(
        step_count=jnp.array(1, dtype=jnp.int32),
        agent_positions=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32
        ),
        team_ids=jnp.ones(shape=(MAX_AGENT_SLOTS,), dtype=jnp.int32),
        active_mask=jnp.ones(shape=(MAX_AGENT_SLOTS,), dtype=bool),
        alive_mask=jnp.ones(shape=(MAX_AGENT_SLOTS,), dtype=bool),
    )

    assert env_state.step_count.shape == ()
    assert env_state.step_count.dtype == jnp.int32

    assert env_state.agent_positions.shape == (
        MAX_AGENT_SLOTS,
        ENVIRONMENT_DIMENSIONS,
    )
    assert env_state.agent_positions.dtype == jnp.float32

    assert env_state.team_ids.shape == (MAX_AGENT_SLOTS,)
    assert env_state.team_ids.dtype == jnp.int32

    assert env_state.active_mask.shape == (MAX_AGENT_SLOTS,)
    assert env_state.active_mask.dtype == bool

    assert env_state.alive_mask.shape == (MAX_AGENT_SLOTS,)
    assert env_state.alive_mask.dtype == bool


def test_action_stores_one_discrete_choice_per_agent_slot() -> None:
    joint_action = Action(
        move=jnp.ones(shape=(MAX_AGENT_SLOTS,), dtype=jnp.int32),
        target=jnp.ones(shape=(MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.ones(shape=(MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )

    assert joint_action.move.shape == (MAX_AGENT_SLOTS,)
    assert joint_action.move.dtype == jnp.int32

    assert joint_action.target.shape == (MAX_AGENT_SLOTS,)
    assert joint_action.target.dtype == jnp.int32

    assert joint_action.use_ultimate.shape == (MAX_AGENT_SLOTS,)
    assert joint_action.use_ultimate.dtype == jnp.int32


def test_action_mask_stores_validity_for_each_action_head() -> None:
    action_mask = ActionMask(
        move=jnp.ones(shape=(MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS), dtype=bool),
        target=jnp.ones(shape=(MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS), dtype=bool),
        use_ultimate=jnp.ones(
            shape=(MAX_AGENT_SLOTS, NUM_ULTIMATE_ACTIONS), dtype=bool
        ),
    )

    assert action_mask.move.shape == (MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS)
    assert action_mask.move.dtype == bool

    assert action_mask.target.shape == (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)
    assert action_mask.target.dtype == bool

    assert action_mask.use_ultimate.shape == (MAX_AGENT_SLOTS, NUM_ULTIMATE_ACTIONS)
    assert action_mask.use_ultimate.dtype == bool


def test_observation_stores_one_vector_per_agent_slot() -> None:
    observation = Observation(
        observation_vectors=jnp.ones(
            shape=(MAX_AGENT_SLOTS, NUM_OBSERVATION_FEATURES), dtype=jnp.float32
        ),
    )

    assert observation.observation_vectors.shape == (
        MAX_AGENT_SLOTS,
        NUM_OBSERVATION_FEATURES,
    )
    assert observation.observation_vectors.dtype == jnp.float32


@pytest.mark.parametrize(
    ("terminated", "truncated", "expected"),
    [
        (jnp.array(False), jnp.array(False), jnp.array(False)),
        (jnp.array(True), jnp.array(False), jnp.array(True)),
        (jnp.array(False), jnp.array(True), jnp.array(True)),
        (jnp.array(True), jnp.array(True), jnp.array(True)),
    ],
)
def test_done_flags_derive_done_from_termination_or_truncation(
    terminated: Array, truncated: Array, expected: Array
) -> None:
    done_flags = DoneFlags(terminated=terminated, truncated=truncated)

    assert jnp.array_equal(done_flags.done, expected)


def test_info_is_empty_placeholder_for_auxiliary_diagnostics() -> None:
    info = Info()

    assert len(info) == 0


@pytest.mark.parametrize("team_size", VALID_TEAM_SIZES)
def test_reset_returns_fixed_shape_core_outputs(team_size: int) -> None:
    config = EnvConfig(team_size=team_size, max_steps=1000)
    key = jax.random.key(42)

    state, observation, action_mask, info = reset(config, key)

    assert state.step_count.shape == ()
    assert state.step_count.dtype == jnp.int32
    assert jnp.array_equal(state.step_count, jnp.array(0, dtype=jnp.int32))

    assert state.agent_positions.shape == (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS)
    assert state.agent_positions.dtype == jnp.float32

    assert state.team_ids.shape == (MAX_AGENT_SLOTS,)
    assert state.team_ids.dtype == jnp.int32

    assert state.active_mask.shape == (MAX_AGENT_SLOTS,)
    assert state.active_mask.dtype == bool

    assert state.alive_mask.shape == (MAX_AGENT_SLOTS,)
    assert state.alive_mask.dtype == bool

    assert observation.observation_vectors.shape == (
        MAX_AGENT_SLOTS,
        NUM_OBSERVATION_FEATURES,
    )
    assert observation.observation_vectors.dtype == jnp.float32

    assert action_mask.move.shape == (MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS)
    assert action_mask.move.dtype == bool

    assert action_mask.target.shape == (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)
    assert action_mask.target.dtype == bool

    assert action_mask.use_ultimate.shape == (MAX_AGENT_SLOTS, NUM_ULTIMATE_ACTIONS)
    assert action_mask.use_ultimate.dtype == bool

    assert isinstance(info, Info)


@pytest.mark.parametrize(
    ("team_size", "expected_active_mask", "active_indices", "inactive_indices"),
    [
        (
            1,
            _bool_vector((1, 0, 0, 0, 0, 1, 0, 0, 0, 0)),
            _int_vector((0, 5)),
            _int_vector((1, 2, 3, 4, 6, 7, 8, 9)),
        ),
        (
            3,
            _bool_vector((1, 1, 1, 0, 0, 1, 1, 1, 0, 0)),
            _int_vector((0, 1, 2, 5, 6, 7)),
            _int_vector((3, 4, 8, 9)),
        ),
        (
            5,
            _bool_vector((1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
            _int_vector((0, 1, 2, 3, 4, 5, 6, 7, 8, 9)),
            _int_vector(()),
        ),
    ],
)
def test_reset_marks_only_active_team_slots_as_alive_and_actionable(
    team_size: int,
    expected_active_mask: Array,
    active_indices: Array,
    inactive_indices: Array,
) -> None:
    config = EnvConfig(team_size=team_size, max_steps=10000)
    key = jax.random.key(42)

    state, _, action_mask, _ = reset(config, key)

    assert jnp.array_equal(state.active_mask, expected_active_mask)
    assert jnp.array_equal(state.alive_mask, expected_active_mask)

    _assert_all_action_heads_gate_padded_rows(
        action_mask=action_mask,
        active_indices=active_indices,
        inactive_indices=inactive_indices,
    )


def test_reset_preserves_stable_team_id_blocks_for_padded_slots() -> None:
    config = EnvConfig(team_size=3, max_steps=10000)
    key = jax.random.key(42)

    state, _, _, _ = reset(config, key)

    expected_team_ids = _int_vector((0, 0, 0, 0, 0, 1, 1, 1, 1, 1))

    assert jnp.array_equal(state.team_ids, expected_team_ids)


def test_reward_stores_one_scalar_per_agent_slot() -> None:
    reward_obj = Reward(rewards=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32))
    rewards = reward_obj.rewards

    assert rewards.shape == (MAX_AGENT_SLOTS,)
    assert rewards.dtype == jnp.float32


def test_step_increments_step_count() -> None:
    config = EnvConfig(team_size=3, max_steps=1000)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    next_state, _, _, _, _, _ = step(config, state, _zero_action(), key)

    assert jnp.array_equal(next_state.step_count, jnp.array(1, dtype=jnp.int32))


def test_step_preserves_slot_aligned_state_arrays() -> None:
    config = EnvConfig(team_size=3, max_steps=1000)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    next_state, _, _, _, _, _ = step(config, state, _zero_action(), key)

    assert jnp.array_equal(next_state.agent_positions, state.agent_positions)
    assert jnp.array_equal(next_state.team_ids, state.team_ids)
    assert jnp.array_equal(next_state.active_mask, state.active_mask)
    assert jnp.array_equal(next_state.alive_mask, state.alive_mask)


def test_step_returns_fixed_shape_core_outputs() -> None:
    config = EnvConfig(team_size=3, max_steps=1000)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    next_state, observation, reward, done_flags, action_mask, info = step(
        config, state, _zero_action(), key
    )

    assert next_state.step_count.shape == ()
    assert next_state.step_count.dtype == jnp.int32

    assert next_state.agent_positions.shape == (
        MAX_AGENT_SLOTS,
        ENVIRONMENT_DIMENSIONS,
    )
    assert next_state.agent_positions.dtype == jnp.float32

    assert next_state.team_ids.shape == (MAX_AGENT_SLOTS,)
    assert next_state.team_ids.dtype == jnp.int32

    assert next_state.active_mask.shape == (MAX_AGENT_SLOTS,)
    assert next_state.active_mask.dtype == bool

    assert next_state.alive_mask.shape == (MAX_AGENT_SLOTS,)
    assert next_state.alive_mask.dtype == bool

    assert observation.observation_vectors.shape == (
        MAX_AGENT_SLOTS,
        NUM_OBSERVATION_FEATURES,
    )
    assert observation.observation_vectors.dtype == jnp.float32

    assert reward.rewards.shape == (MAX_AGENT_SLOTS,)
    assert reward.rewards.dtype == jnp.float32

    assert done_flags.terminated.shape == ()
    assert done_flags.terminated.dtype == bool

    assert done_flags.truncated.shape == ()
    assert done_flags.truncated.dtype == bool

    assert action_mask.move.shape == (MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS)
    assert action_mask.move.dtype == bool

    assert action_mask.target.shape == (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)
    assert action_mask.target.dtype == bool

    assert action_mask.use_ultimate.shape == (MAX_AGENT_SLOTS, NUM_ULTIMATE_ACTIONS)
    assert action_mask.use_ultimate.dtype == bool

    assert isinstance(info, Info)


def test_step_returns_zero_rewards_for_all_agent_slots() -> None:
    config = EnvConfig(team_size=3, max_steps=1000)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    _, _, reward, _, _, _ = step(config, state, _zero_action(), key)

    assert jnp.array_equal(
        reward.rewards, jnp.zeros(shape=(MAX_AGENT_SLOTS,), dtype=jnp.float32)
    )


def test_step_action_masks_are_gated_by_active_slots() -> None:
    config = EnvConfig(team_size=3, max_steps=1000)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    _, _, _, _, action_mask, _ = step(config, state, _zero_action(), key)

    _assert_all_action_heads_gate_padded_rows(
        action_mask=action_mask,
        active_indices=_int_vector((0, 1, 2, 5, 6, 7)),
        inactive_indices=_int_vector((3, 4, 8, 9)),
    )


def test_step_does_not_truncate_before_horizon() -> None:
    config = EnvConfig(team_size=3, max_steps=2)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    _, _, _, done_flags, _, _ = step(config, state, _zero_action(), key)

    assert jnp.array_equal(done_flags.terminated, jnp.array(False))
    assert jnp.array_equal(done_flags.truncated, jnp.array(False))
    assert jnp.array_equal(done_flags.done, jnp.array(False))


def test_step_truncates_when_incremented_step_reaches_horizon() -> None:
    config = EnvConfig(team_size=3, max_steps=1)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    _, _, _, done_flags, _, _ = step(config, state, _zero_action(), key)

    assert jnp.array_equal(done_flags.terminated, jnp.array(False))
    assert jnp.array_equal(done_flags.truncated, jnp.array(True))
    assert jnp.array_equal(done_flags.done, jnp.array(True))


def test_that_step_can_be_jit_compiled() -> None:
    step_jitted = jax.jit(step)

    config = EnvConfig(team_size=3, max_steps=1)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    next_state, observation, reward, done_flags, action_mask, info = cast(
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        step_jitted(config, state, _zero_action(), key),
    )

    assert next_state.step_count.shape == ()
    assert next_state.step_count.dtype == jnp.int32

    assert next_state.agent_positions.shape == (
        MAX_AGENT_SLOTS,
        ENVIRONMENT_DIMENSIONS,
    )
    assert next_state.agent_positions.dtype == jnp.float32

    assert next_state.team_ids.shape == (MAX_AGENT_SLOTS,)
    assert next_state.team_ids.dtype == jnp.int32

    assert next_state.active_mask.shape == (MAX_AGENT_SLOTS,)
    assert next_state.active_mask.dtype == bool

    assert next_state.alive_mask.shape == (MAX_AGENT_SLOTS,)
    assert next_state.alive_mask.dtype == bool

    assert observation.observation_vectors.shape == (
        MAX_AGENT_SLOTS,
        NUM_OBSERVATION_FEATURES,
    )
    assert observation.observation_vectors.dtype == jnp.float32

    assert reward.rewards.shape == (MAX_AGENT_SLOTS,)
    assert reward.rewards.dtype == jnp.float32

    assert done_flags.terminated.shape == ()
    assert done_flags.terminated.dtype == bool

    assert done_flags.truncated.shape == ()
    assert done_flags.truncated.dtype == bool

    assert action_mask.move.shape == (MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS)
    assert action_mask.move.dtype == bool

    assert action_mask.target.shape == (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)
    assert action_mask.target.dtype == bool

    assert action_mask.use_ultimate.shape == (MAX_AGENT_SLOTS, NUM_ULTIMATE_ACTIONS)
    assert action_mask.use_ultimate.dtype == bool

    assert isinstance(info, Info)
