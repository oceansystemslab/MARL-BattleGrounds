"""Movement integration tests for Milestone 4 Step 3."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.env import step
from marl_battlegrounds.core.geometry import GEOMETRY_TOLERANCE
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
    NUM_ULTIMATE_ACTIONS,
    OBJECTIVE_FEATURES,
    OBSTACLE_FEATURES,
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


def _empty_obstacles() -> Array:
    """Create a padded all-inactive obstacle table."""
    return jnp.zeros(
        (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
        dtype=jnp.float32,
    )


def _deterministic_config(
    *,
    team_size: int = 3,
    max_steps: int = 1000,
    movement_speed: float = 1.0,
    obstacles: Array | None = None,
) -> EnvConfig:
    """Create a deterministic config for movement integration tests."""
    return EnvConfig(
        team_size=team_size,
        max_steps=max_steps,
        map_width=20.0,
        map_height=12.0,
        movement_speed=movement_speed,
        observation_radius=8.0,
        target_range=6.0,
        default_agent_radius=0.5,
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
    )


def _team_ids() -> Array:
    """Create the canonical fixed-slot team-id vector."""
    return jnp.concatenate(
        (
            jnp.zeros((MAX_AGENTS_PER_TEAM,), dtype=jnp.int32),
            jnp.ones((MAX_AGENTS_PER_TEAM,), dtype=jnp.int32),
        ),
        axis=0,
    )


def _agent_positions_array_with_rows(*rows: tuple[int, Array]) -> Array:
    """Create a padded agent-position array with selected slots populated."""
    agent_positions = jnp.zeros(
        (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        dtype=jnp.float32,
    )

    for slot, agent_position in rows:
        assert 0 <= slot < MAX_AGENT_SLOTS
        agent_positions = agent_positions.at[slot].set(agent_position)

    return agent_positions


def _agent_radii_array_with_rows(*rows: tuple[int, Array | float]) -> Array:
    """Create a padded agent-radius vector with selected slots populated."""
    radii = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32)

    for slot, radius in rows:
        assert 0 <= slot < MAX_AGENT_SLOTS
        radii = radii.at[slot].set(radius)

    return radii


def _mask_with_true_slots(*slots: int) -> Array:
    """Create a slot mask with only selected slots marked true."""
    mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)

    for slot in slots:
        assert 0 <= slot < MAX_AGENT_SLOTS
        mask = mask.at[slot].set(True)

    return mask


def _state_with_single_active_alive_agent(
    position: Array,
    *,
    radius: float = 0.5,
    step_count: int = 0,
) -> EnvState:
    """Create a valid state with slot 0 active and alive."""
    return EnvState(
        step_count=jnp.array(step_count, dtype=jnp.int32),
        agent_positions=_agent_positions_array_with_rows((0, position)),
        agent_radii=_agent_radii_array_with_rows((0, radius)),
        team_ids=_team_ids(),
        active_mask=_mask_with_true_slots(0),
        alive_mask=_mask_with_true_slots(0),
    )


def _joint_action_with_moves(*rows: tuple[int, int]) -> Action:
    """Create a slot-aligned joint action with selected movement overrides."""
    joint_action_moves = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    joint_action_targets = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    joint_action_ults = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)

    for slot, move_action in rows:
        assert 0 <= slot < MAX_AGENT_SLOTS
        joint_action_moves = joint_action_moves.at[slot].set(move_action)

    return Action(
        move=joint_action_moves,
        target=joint_action_targets,
        use_ultimate=joint_action_ults,
    )


def _assert_center_close(result: Array, expected: Array) -> None:
    """Assert that a position row matches the expected float32 center."""
    assert result.shape == (ENVIRONMENT_DIMENSIONS,)
    assert result.dtype == jnp.float32
    assert bool(
        jnp.allclose(
            result,
            expected,
            atol=GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
    )


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


def _assert_reward_contract(reward: Reward) -> None:
    """Assert the placeholder reward contract."""
    assert reward.rewards.shape == (MAX_AGENT_SLOTS,)
    assert reward.rewards.dtype == jnp.float32
    assert bool(jnp.all(reward.rewards == 0.0))


def _assert_done_flags_contract(
    done_flags: DoneFlags,
    *,
    expected_truncated: bool,
) -> None:
    """Assert the placeholder done-flag contract."""
    assert done_flags.terminated.shape == ()
    assert done_flags.terminated.dtype == bool
    assert bool(done_flags.terminated) is False

    assert done_flags.truncated.shape == ()
    assert done_flags.truncated.dtype == bool
    assert bool(done_flags.truncated) is expected_truncated


def _assert_placeholder_action_mask_semantics(action_mask: ActionMask) -> None:
    """Assert current placeholder action-mask semantics for one active slot."""
    assert bool(jnp.all(action_mask.move[0]))
    assert bool(jnp.all(~action_mask.move[1:]))

    assert bool(action_mask.target[0, 0])
    assert bool(jnp.all(~action_mask.target[0, 1:]))
    assert bool(jnp.all(~action_mask.target[1:]))

    assert bool(action_mask.use_ultimate[0, 0])
    assert bool(~action_mask.use_ultimate[0, 1])
    assert bool(jnp.all(~action_mask.use_ultimate[1:]))


def test_move_stay_preserves_valid_position_in_free_space() -> None:
    config = _deterministic_config()
    key = jax.random.key(42)
    start = jnp.array([10.0, 10.0], dtype=jnp.float32)
    state = _state_with_single_active_alive_agent(start)
    joint_action = _joint_action_with_moves((0, MOVE_STAY))

    next_state, *_ = step(config, state, joint_action, key)

    _assert_center_close(next_state.agent_positions[0], start)


@pytest.mark.parametrize(
    ("move_action", "expected_position"),
    [
        pytest.param(
            MOVE_NORTH, jnp.array([10.0, 11.0], dtype=jnp.float32), id="north"
        ),
        pytest.param(MOVE_SOUTH, jnp.array([10.0, 9.0], dtype=jnp.float32), id="south"),
        pytest.param(MOVE_EAST, jnp.array([11.0, 10.0], dtype=jnp.float32), id="east"),
        pytest.param(MOVE_WEST, jnp.array([9.0, 10.0], dtype=jnp.float32), id="west"),
    ],
)
def test_cardinal_moves_update_position_in_free_space(
    move_action: int,
    expected_position: Array,
) -> None:
    config = _deterministic_config()
    key = jax.random.key(42)
    state = _state_with_single_active_alive_agent(
        jnp.array([10.0, 10.0], dtype=jnp.float32)
    )
    joint_action = _joint_action_with_moves((0, move_action))

    next_state, obs, reward, done_flags, action_mask, info = step(
        config,
        state,
        joint_action,
        key,
    )

    _assert_center_close(next_state.agent_positions[0], expected_position)
    _assert_state_contract(next_state)
    _assert_observation_contract(obs)
    _assert_reward_contract(reward)
    _assert_done_flags_contract(done_flags, expected_truncated=False)
    _assert_action_mask_contract(action_mask)
    _assert_placeholder_action_mask_semantics(action_mask)
    assert isinstance(info, Info)


@pytest.mark.parametrize(
    ("move_action", "expected_delta"),
    [
        pytest.param(
            MOVE_NORTHEAST,
            jnp.array([1.0, 1.0], dtype=jnp.float32) / jnp.sqrt(2.0),
            id="northeast",
        ),
        pytest.param(
            MOVE_NORTHWEST,
            jnp.array([-1.0, 1.0], dtype=jnp.float32) / jnp.sqrt(2.0),
            id="northwest",
        ),
        pytest.param(
            MOVE_SOUTHEAST,
            jnp.array([1.0, -1.0], dtype=jnp.float32) / jnp.sqrt(2.0),
            id="southeast",
        ),
        pytest.param(
            MOVE_SOUTHWEST,
            jnp.array([-1.0, -1.0], dtype=jnp.float32) / jnp.sqrt(2.0),
            id="southwest",
        ),
    ],
)
def test_diagonal_moves_are_normalized_in_free_space(
    move_action: int,
    expected_delta: Array,
) -> None:
    config = _deterministic_config()
    key = jax.random.key(42)
    start = jnp.array([10.0, 10.0], dtype=jnp.float32)
    state = _state_with_single_active_alive_agent(start)
    joint_action = _joint_action_with_moves((0, move_action))

    next_state, *_ = step(config, state, joint_action, key)

    displacement = next_state.agent_positions[0] - start
    expected_position = start + expected_delta

    _assert_center_close(next_state.agent_positions[0], expected_position)
    assert bool(
        jnp.isclose(
            cast(Array, jnp.linalg.norm(displacement)),
            jnp.asarray(config.movement_speed, dtype=jnp.float32),
            atol=GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
    )


def test_step_preserves_placeholder_contracts_after_non_stay_movement() -> None:
    config = _deterministic_config(max_steps=10)
    key = jax.random.key(42)
    state = _state_with_single_active_alive_agent(
        jnp.array([10.0, 10.0], dtype=jnp.float32)
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, obs, reward, done_flags, action_mask, info = step(
        config,
        state,
        joint_action,
        key,
    )

    assert next_state.step_count == state.step_count + 1
    _assert_state_contract(next_state)
    _assert_observation_contract(obs)
    _assert_reward_contract(reward)
    _assert_done_flags_contract(done_flags, expected_truncated=False)
    _assert_action_mask_contract(action_mask)
    _assert_placeholder_action_mask_semantics(action_mask)
    assert isinstance(info, Info)


def test_step_truncates_after_incremented_step_count() -> None:
    config = _deterministic_config(max_steps=1)
    key = jax.random.key(42)
    state = _state_with_single_active_alive_agent(
        jnp.array([10.0, 10.0], dtype=jnp.float32),
        step_count=0,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    _, _, _, done_flags, _, _ = step(config, state, joint_action, key)

    _assert_done_flags_contract(done_flags, expected_truncated=True)
