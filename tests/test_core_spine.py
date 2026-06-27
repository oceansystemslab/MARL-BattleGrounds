"""Core simulator spine contract tests."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.types import (
    CONTEXT_FEATURES,
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBJECTIVE_SLOTS,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_NORTHEAST,
    MOVE_NORTHWEST,
    MOVE_SOUTH,
    MOVE_SOUTHEAST,
    MOVE_SOUTHWEST,
    MOVE_STAY,
    MOVE_WEST,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_TEAMS,
    NUM_ULTIMATE_ACTIONS,
    OBJECTIVE_FEATURES,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_FEATURES,
    OBSTACLE_TYPE_NONE,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    SELF_FEATURES,
    UNIT_FEATURES,
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

# Test helpers ---


def _bool_vector(values: tuple[int, ...]) -> Array:
    """Return a boolean JAX vector from integer test literals."""
    return jnp.array(values, dtype=bool)


def _int_vector(values: tuple[int, ...]) -> Array:
    """Return an int32 JAX vector from integer test literals."""
    return jnp.array(values, dtype=jnp.int32)


def _empty_obstacles() -> Array:
    """Return a zero-filled obstacle feature table."""
    return jnp.zeros(shape=(MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _sample_obstacles() -> Array:
    """Return an obstacle table with slot 0 set to a deterministic pillar."""
    obstacles = _empty_obstacles()
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_X].set(5.0)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_Y].set(4.0)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_RADIUS].set(1.5)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_ACTIVE].set(1.0)
    return obstacles


def _config(
    team_size: int = 3,
    max_steps: int = 1000,
    obstacles: Array | None = None,
) -> EnvConfig:
    """Return a deterministic test config."""
    return EnvConfig(
        team_size=team_size,
        max_steps=max_steps,
        map_width=20.0,
        map_height=12.0,
        movement_speed=1.0,
        observation_radius=8.0,
        target_range=6.0,
        default_agent_radius=0.5,
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
    )


def _zero_action() -> Action:
    """Return a no-op action for every agent slot."""
    return Action(
        move=jnp.zeros(shape=(MAX_AGENT_SLOTS,), dtype=jnp.int32),
        target=jnp.zeros(shape=(MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros(shape=(MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _zero_observation() -> Observation:
    """Return a zero-filled observation."""
    return Observation(
        self_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, SELF_FEATURES), dtype=jnp.float32
        ),
        ally_unit_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES),
            dtype=jnp.float32,
        ),
        enemy_unit_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES),
            dtype=jnp.float32,
        ),
        map_obstacle_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
            dtype=jnp.float32,
        ),
        objective_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_OBJECTIVE_SLOTS, OBJECTIVE_FEATURES),
            dtype=jnp.float32,
        ),
        context_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, CONTEXT_FEATURES), dtype=jnp.float32
        ),
        ally_visibility_mask=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM), dtype=bool
        ),
        enemy_visibility_mask=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM), dtype=bool
        ),
        ally_targetability_mask=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM), dtype=bool
        ),
        enemy_targetability_mask=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM), dtype=bool
        ),
    )


def _expected_self_visibility(active_alive_mask: Array) -> Array:
    """Return the Step 1 self-visibility mask."""
    team_self_visibility = jnp.identity(MAX_AGENTS_PER_TEAM, dtype=bool)
    all_slots_self_visibility = jnp.vstack((team_self_visibility, team_self_visibility))
    return jnp.logical_and(all_slots_self_visibility, active_alive_mask.reshape(-1, 1))


def _expected_step1_target_rows(num_rows: int) -> Array:
    """Return Step 1 target-mask rows."""
    none_column = jnp.ones(shape=(num_rows, 1), dtype=bool)
    unit_columns = jnp.zeros(shape=(num_rows, NUM_TARGET_ACTIONS - 1), dtype=bool)
    return jnp.concat((none_column, unit_columns), axis=1)


def _expected_step1_ultimate_rows(num_rows: int) -> Array:
    """Return Step 1 ultimate-mask rows."""
    noop_column = jnp.ones(shape=(num_rows, 1), dtype=bool)
    use_column = jnp.zeros(shape=(num_rows, NUM_ULTIMATE_ACTIONS - 1), dtype=bool)
    return jnp.concat((noop_column, use_column), axis=1)


def _assert_state_contract(state: EnvState) -> None:
    """Assert the EnvState shape and dtype contract."""
    assert state.step_count.shape == ()
    assert state.step_count.dtype == jnp.int32

    assert state.agent_positions.shape == (
        MAX_AGENT_SLOTS,
        ENVIRONMENT_DIMENSIONS,
    )
    assert state.agent_positions.dtype == jnp.float32

    assert state.agent_radii.shape == (MAX_AGENT_SLOTS,)
    assert state.agent_radii.dtype == jnp.float32

    assert state.team_ids.shape == (MAX_AGENT_SLOTS,)
    assert state.team_ids.dtype == jnp.int32

    assert state.active_mask.shape == (MAX_AGENT_SLOTS,)
    assert state.active_mask.dtype == bool

    assert state.alive_mask.shape == (MAX_AGENT_SLOTS,)
    assert state.alive_mask.dtype == bool


def _assert_observation_contract(observation: Observation) -> None:
    """Assert the Observation shape and dtype contract."""
    assert observation.self_features.shape == (MAX_AGENT_SLOTS, SELF_FEATURES)
    assert observation.self_features.dtype == jnp.float32

    assert observation.ally_unit_features.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
        UNIT_FEATURES,
    )
    assert observation.ally_unit_features.dtype == jnp.float32

    assert observation.enemy_unit_features.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
        UNIT_FEATURES,
    )
    assert observation.enemy_unit_features.dtype == jnp.float32

    assert observation.map_obstacle_features.shape == (
        MAX_AGENT_SLOTS,
        MAX_OBSTACLE_SLOTS,
        OBSTACLE_FEATURES,
    )
    assert observation.map_obstacle_features.dtype == jnp.float32

    assert observation.objective_features.shape == (
        MAX_AGENT_SLOTS,
        MAX_OBJECTIVE_SLOTS,
        OBJECTIVE_FEATURES,
    )
    assert observation.objective_features.dtype == jnp.float32

    assert observation.context_features.shape == (MAX_AGENT_SLOTS, CONTEXT_FEATURES)
    assert observation.context_features.dtype == jnp.float32

    assert observation.ally_visibility_mask.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
    )
    assert observation.ally_visibility_mask.dtype == bool

    assert observation.enemy_visibility_mask.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
    )
    assert observation.enemy_visibility_mask.dtype == bool

    assert observation.ally_targetability_mask.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
    )
    assert observation.ally_targetability_mask.dtype == bool

    assert observation.enemy_targetability_mask.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
    )
    assert observation.enemy_targetability_mask.dtype == bool


def _assert_action_mask_contract(action_mask: ActionMask) -> None:
    """Assert the ActionMask shape and dtype contract."""
    assert action_mask.move.shape == (MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS)
    assert action_mask.move.dtype == bool

    assert action_mask.target.shape == (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)
    assert action_mask.target.dtype == bool

    assert action_mask.use_ultimate.shape == (
        MAX_AGENT_SLOTS,
        NUM_ULTIMATE_ACTIONS,
    )
    assert action_mask.use_ultimate.dtype == bool


def _assert_step1_action_mask_values(
    action_mask: ActionMask,
    active_indices: Array,
    inactive_indices: Array,
) -> None:
    """Assert Step 1 action-mask values."""
    active_count = int(active_indices.shape[0])

    assert jnp.all(action_mask.move[active_indices, :])
    assert not jnp.any(action_mask.move[inactive_indices, :])

    assert jnp.array_equal(
        action_mask.target[active_indices, :],
        _expected_step1_target_rows(active_count),
    )
    assert not jnp.any(action_mask.target[inactive_indices, :])

    assert jnp.array_equal(
        action_mask.use_ultimate[active_indices, :],
        _expected_step1_ultimate_rows(active_count),
    )
    assert not jnp.any(action_mask.use_ultimate[inactive_indices, :])


def _assert_step1_observation_values(
    observation: Observation,
    config: EnvConfig,
    active_alive_mask: Array,
) -> None:
    """Assert Step 1 observation values."""
    expected_map_features = jnp.broadcast_to(
        config.obstacles[None, :, :],
        (MAX_AGENT_SLOTS, MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
    )

    assert jnp.array_equal(observation.map_obstacle_features, expected_map_features)
    assert jnp.array_equal(
        observation.ally_visibility_mask,
        _expected_self_visibility(active_alive_mask),
    )
    assert not jnp.any(observation.enemy_visibility_mask)
    assert not jnp.any(observation.ally_targetability_mask)
    assert not jnp.any(observation.enemy_targetability_mask)


def test_static_shape_constants_are_consistent() -> None:
    assert NUM_TEAMS == 2
    assert MAX_AGENTS_PER_TEAM == 5
    assert MAX_AGENT_SLOTS == NUM_TEAMS * MAX_AGENTS_PER_TEAM
    assert NUM_TARGET_ACTIONS == MAX_AGENT_SLOTS + 1
    assert NUM_MOVE_ACTIONS == 9
    assert NUM_ULTIMATE_ACTIONS == 2
    assert ENVIRONMENT_DIMENSIONS == 2

    assert MAX_OBSTACLE_SLOTS == 16
    assert OBSTACLE_FEATURES == 8
    assert OBSTACLE_TYPE_NONE == 0
    assert OBSTACLE_TYPE_PILLAR == 1
    assert OBSTACLE_TYPE_WALL == 2

    assert OBSTACLE_FEATURE_TYPE == 0
    assert OBSTACLE_FEATURE_X == 1
    assert OBSTACLE_FEATURE_Y == 2
    assert OBSTACLE_FEATURE_RADIUS == 3
    assert OBSTACLE_FEATURE_WIDTH == 4
    assert OBSTACLE_FEATURE_HEIGHT == 5
    assert OBSTACLE_FEATURE_THETA == 6
    assert OBSTACLE_FEATURE_ACTIVE == 7

    assert tuple(range(NUM_MOVE_ACTIONS)) == (
        MOVE_STAY,
        MOVE_NORTH,
        MOVE_SOUTH,
        MOVE_EAST,
        MOVE_WEST,
        MOVE_NORTHEAST,
        MOVE_NORTHWEST,
        MOVE_SOUTHEAST,
        MOVE_SOUTHWEST,
    )

    assert SELF_FEATURES == 12
    assert UNIT_FEATURES == 16
    assert MAX_OBJECTIVE_SLOTS == 8
    assert OBJECTIVE_FEATURES == 12
    assert CONTEXT_FEATURES == 8


def test_env_config_stores_static_episode_settings() -> None:
    obstacles = _sample_obstacles()
    env_config = _config(team_size=5, max_steps=10000, obstacles=obstacles)

    assert env_config.team_size == 5
    assert env_config.max_steps == 10000
    assert env_config.map_width == 20.0
    assert env_config.map_height == 12.0
    assert env_config.movement_speed == 1.0
    assert env_config.observation_radius == 8.0
    assert env_config.target_range == 6.0
    assert env_config.default_agent_radius == 0.5

    assert env_config.obstacles.shape == (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES)
    assert env_config.obstacles.dtype == jnp.float32
    assert jnp.array_equal(env_config.obstacles, obstacles)


def test_env_state_stores_slot_aligned_arrays() -> None:
    env_state = EnvState(
        step_count=jnp.array(1, dtype=jnp.int32),
        agent_positions=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32
        ),
        agent_radii=jnp.ones(shape=(MAX_AGENT_SLOTS,), dtype=jnp.float32),
        team_ids=jnp.ones(shape=(MAX_AGENT_SLOTS,), dtype=jnp.int32),
        active_mask=jnp.ones(shape=(MAX_AGENT_SLOTS,), dtype=bool),
        alive_mask=jnp.ones(shape=(MAX_AGENT_SLOTS,), dtype=bool),
    )

    _assert_state_contract(env_state)


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

    _assert_action_mask_contract(action_mask)


def test_observation_stores_structured_families() -> None:
    _assert_observation_contract(_zero_observation())


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
    config = _config(team_size=team_size)
    key = jax.random.key(42)

    state, observation, action_mask, info = reset(config, key)

    _assert_state_contract(state)
    assert jnp.array_equal(state.step_count, jnp.array(0, dtype=jnp.int32))
    assert jnp.all(state.agent_radii == config.default_agent_radius)

    _assert_observation_contract(observation)
    _assert_step1_observation_values(
        observation=observation,
        config=config,
        active_alive_mask=jnp.logical_and(state.active_mask, state.alive_mask),
    )

    _assert_action_mask_contract(action_mask)

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
    config = _config(team_size=team_size, max_steps=10000)
    key = jax.random.key(42)

    state, observation, action_mask, _ = reset(config, key)

    assert jnp.array_equal(state.active_mask, expected_active_mask)
    assert jnp.array_equal(state.alive_mask, expected_active_mask)
    assert jnp.array_equal(
        observation.ally_visibility_mask,
        _expected_self_visibility(expected_active_mask),
    )

    _assert_step1_action_mask_values(
        action_mask=action_mask,
        active_indices=active_indices,
        inactive_indices=inactive_indices,
    )


def test_reset_preserves_stable_team_id_blocks_for_padded_slots() -> None:
    config = _config(team_size=3, max_steps=10000)
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
    config = _config(team_size=3, max_steps=1000)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    next_state, _, _, _, _, _ = step(config, state, _zero_action(), key)

    assert jnp.array_equal(next_state.step_count, jnp.array(1, dtype=jnp.int32))


def test_step_preserves_slot_aligned_state_arrays() -> None:
    config = _config(team_size=3, max_steps=1000)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    next_state, _, _, _, _, _ = step(config, state, _zero_action(), key)

    assert jnp.array_equal(next_state.agent_positions, state.agent_positions)
    assert jnp.array_equal(next_state.agent_radii, state.agent_radii)
    assert jnp.array_equal(next_state.team_ids, state.team_ids)
    assert jnp.array_equal(next_state.active_mask, state.active_mask)
    assert jnp.array_equal(next_state.alive_mask, state.alive_mask)


def test_step_returns_fixed_shape_core_outputs() -> None:
    config = _config(team_size=3, max_steps=1000)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    next_state, observation, reward, done_flags, action_mask, info = step(
        config, state, _zero_action(), key
    )

    _assert_state_contract(next_state)
    _assert_observation_contract(observation)
    _assert_step1_observation_values(
        observation=observation,
        config=config,
        active_alive_mask=jnp.logical_and(
            next_state.active_mask, next_state.alive_mask
        ),
    )

    assert reward.rewards.shape == (MAX_AGENT_SLOTS,)
    assert reward.rewards.dtype == jnp.float32

    assert done_flags.terminated.shape == ()
    assert done_flags.terminated.dtype == bool

    assert done_flags.truncated.shape == ()
    assert done_flags.truncated.dtype == bool

    _assert_action_mask_contract(action_mask)

    assert isinstance(info, Info)


def test_step_returns_zero_rewards_for_all_agent_slots() -> None:
    config = _config(team_size=3, max_steps=1000)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    _, _, reward, _, _, _ = step(config, state, _zero_action(), key)

    assert jnp.array_equal(
        reward.rewards, jnp.zeros(shape=(MAX_AGENT_SLOTS,), dtype=jnp.float32)
    )


def test_step_action_masks_are_gated_by_active_alive_slots() -> None:
    config = _config(team_size=3, max_steps=1000)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    _, _, _, _, action_mask, _ = step(config, state, _zero_action(), key)

    _assert_step1_action_mask_values(
        action_mask=action_mask,
        active_indices=_int_vector((0, 1, 2, 5, 6, 7)),
        inactive_indices=_int_vector((3, 4, 8, 9)),
    )


def test_step_does_not_truncate_before_horizon() -> None:
    config = _config(team_size=3, max_steps=2)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    _, _, _, done_flags, _, _ = step(config, state, _zero_action(), key)

    assert jnp.array_equal(done_flags.terminated, jnp.array(False))
    assert jnp.array_equal(done_flags.truncated, jnp.array(False))
    assert jnp.array_equal(done_flags.done, jnp.array(False))


def test_step_truncates_when_incremented_step_reaches_horizon() -> None:
    config = _config(team_size=3, max_steps=1)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    _, _, _, done_flags, _, _ = step(config, state, _zero_action(), key)

    assert jnp.array_equal(done_flags.terminated, jnp.array(False))
    assert jnp.array_equal(done_flags.truncated, jnp.array(True))
    assert jnp.array_equal(done_flags.done, jnp.array(True))


def test_that_step_can_be_jit_compiled() -> None:
    step_jitted = jax.jit(step)

    config = _config(team_size=3, max_steps=1)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)

    next_state, observation, reward, done_flags, action_mask, info = cast(
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        step_jitted(config, state, _zero_action(), key),
    )

    _assert_state_contract(next_state)
    _assert_observation_contract(observation)

    assert reward.rewards.shape == (MAX_AGENT_SLOTS,)
    assert reward.rewards.dtype == jnp.float32

    assert done_flags.terminated.shape == ()
    assert done_flags.terminated.dtype == bool

    assert done_flags.truncated.shape == ()
    assert done_flags.truncated.dtype == bool

    _assert_action_mask_contract(action_mask)

    assert isinstance(info, Info)


def test_step_can_run_in_scanned_rollout() -> None:
    horizon = 3
    config = _config(team_size=5, max_steps=1000)
    key = jax.random.key(42)
    state, _, _, _ = reset(config, key)
    joint_action = _zero_action()

    def _step_wrapper(
        state: EnvState,
        key: Array,
        config: EnvConfig = config,
        joint_action: Action = joint_action,
    ) -> tuple[EnvState, Array]:
        """Run one rollout step for scan."""
        new_state, _, _, _, _, _ = step(config, state, joint_action, key)
        return new_state, new_state.step_count

    keys = jax.random.split(key, horizon)
    assert keys.shape == (horizon,)

    new_state, history = jax.lax.scan(
        f=_step_wrapper, init=state, xs=keys, length=horizon
    )

    assert new_state.step_count == horizon
    assert new_state.step_count.dtype == jnp.int32
    assert history.shape == (horizon,)
    assert history.dtype == jnp.int32
