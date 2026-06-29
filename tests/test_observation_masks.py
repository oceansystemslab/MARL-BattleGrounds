"""Observation feature-contract tests for Milestone 4 Step 4."""

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_TEAM_ID,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    OBSTACLE_FEATURES,
    SELF_FEATURES,
    UNIT_FEATURES,
    Action,
    EnvConfig,
    EnvState,
    Observation,
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
    map_width: float = 20.0,
    map_height: float = 12.0,
    obstacles: Array | None = None,
) -> EnvConfig:
    """Create a deterministic config for observation-mask tests."""
    return EnvConfig(
        team_size=team_size,
        max_steps=max_steps,
        map_width=map_width,
        map_height=map_height,
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


def _agent_radii_array_with_rows(*rows: tuple[int, float]) -> Array:
    """Create a padded agent-radius vector with selected slots populated."""
    radii = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32)

    for slot, radius in rows:
        assert 0 <= slot < MAX_AGENT_SLOTS
        radii = radii.at[slot].set(radius)

    return radii


def _state_two_versus_two_game(
    *,
    agent_a_position: Array,
    agent_b_position: Array,
    agent_c_position: Array,
    agent_d_position: Array,
    agent_a_active_flag: bool = True,
    agent_a_alive_flag: bool = True,
    agent_b_active_flag: bool = True,
    agent_b_alive_flag: bool = True,
    agent_c_active_flag: bool = True,
    agent_c_alive_flag: bool = True,
    agent_d_active_flag: bool = True,
    agent_d_alive_flag: bool = True,
    radius: float = 0.5,
    step_count: int = 0,
) -> EnvState:
    """Create a fixed two-versus-two state in slots 0, 1, 5, and 6."""
    agent_a_index = 0
    agent_b_index = 1
    agent_c_index = MAX_AGENTS_PER_TEAM
    agent_d_index = MAX_AGENTS_PER_TEAM + 1

    active_mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)
    alive_mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)

    active_mask = active_mask.at[agent_a_index].set(agent_a_active_flag)
    alive_mask = alive_mask.at[agent_a_index].set(agent_a_alive_flag)

    active_mask = active_mask.at[agent_b_index].set(agent_b_active_flag)
    alive_mask = alive_mask.at[agent_b_index].set(agent_b_alive_flag)

    active_mask = active_mask.at[agent_c_index].set(agent_c_active_flag)
    alive_mask = alive_mask.at[agent_c_index].set(agent_c_alive_flag)

    active_mask = active_mask.at[agent_d_index].set(agent_d_active_flag)
    alive_mask = alive_mask.at[agent_d_index].set(agent_d_alive_flag)

    return EnvState(
        step_count=jnp.array(step_count, dtype=jnp.int32),
        agent_positions=_agent_positions_array_with_rows(
            (agent_a_index, agent_a_position),
            (agent_b_index, agent_b_position),
            (agent_c_index, agent_c_position),
            (agent_d_index, agent_d_position),
        ),
        agent_radii=_agent_radii_array_with_rows(
            (agent_a_index, radius),
            (agent_b_index, radius),
            (agent_c_index, radius),
            (agent_d_index, radius),
        ),
        team_ids=_team_ids(),
        active_mask=active_mask,
        alive_mask=alive_mask,
    )


def _assert_self_features_match_state_base_fields(
    observation: Observation,
    state: EnvState,
) -> None:
    """Assert that self features expose the state's shared base agent fields."""
    assert observation.self_features.shape == (MAX_AGENT_SLOTS, SELF_FEATURES)
    assert observation.self_features.dtype == jnp.float32

    assert bool(
        jnp.allclose(
            observation.self_features[:, AGENT_FEATURE_X : AGENT_FEATURE_Y + 1],
            state.agent_positions,
            atol=0.0,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.allclose(
            observation.self_features[:, AGENT_FEATURE_RADIUS],
            state.agent_radii,
            atol=0.0,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.allclose(
            observation.self_features[:, AGENT_FEATURE_TEAM_ID],
            state.team_ids.astype(jnp.float32),
            atol=0.0,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.allclose(
            observation.self_features[:, AGENT_FEATURE_ACTIVE],
            state.active_mask.astype(jnp.float32),
            atol=0.0,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.allclose(
            observation.self_features[:, AGENT_FEATURE_ALIVE],
            state.alive_mask.astype(jnp.float32),
            atol=0.0,
            rtol=0.0,
        )
    )


def _assert_unused_self_feature_columns_are_zero(observation: Observation) -> None:
    """Assert that currently unused self-feature columns remain zero."""
    unused_start = AGENT_FEATURE_ALIVE + 1

    assert bool(
        jnp.allclose(
            observation.self_features[:, unused_start:],
            jnp.zeros(
                (MAX_AGENT_SLOTS, SELF_FEATURES - unused_start),
                dtype=jnp.float32,
            ),
            atol=0.0,
            rtol=0.0,
        )
    )


def _assert_unit_features_remain_checkpoint_2_placeholders(
    observation: Observation,
) -> None:
    """Assert that unit-candidate features are still zero placeholders."""
    assert observation.ally_unit_features.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
        UNIT_FEATURES,
    )
    assert observation.enemy_unit_features.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
        UNIT_FEATURES,
    )

    assert observation.ally_unit_features.dtype == jnp.float32
    assert observation.enemy_unit_features.dtype == jnp.float32

    assert bool(jnp.all(observation.ally_unit_features == 0.0))
    assert bool(jnp.all(observation.enemy_unit_features == 0.0))


def test_reset_self_features_match_reset_state_base_fields() -> None:
    config = _deterministic_config()
    key = jax.random.key(42)

    initial_state, observation, *_ = reset(config, key)

    _assert_self_features_match_state_base_fields(observation, initial_state)
    _assert_unused_self_feature_columns_are_zero(observation)
    _assert_unit_features_remain_checkpoint_2_placeholders(observation)


def test_step_self_features_match_next_state_base_fields() -> None:
    config = _deterministic_config()
    key = jax.random.key(42)

    state = _state_two_versus_two_game(
        agent_a_position=jnp.array([5.0, 5.0], dtype=jnp.float32),
        agent_b_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
        agent_c_position=jnp.array([5.0, 8.0], dtype=jnp.float32),
        agent_d_position=jnp.array([8.0, 8.0], dtype=jnp.float32),
        agent_a_active_flag=True,
        agent_a_alive_flag=True,
        agent_b_active_flag=False,
        agent_b_alive_flag=True,
        agent_c_active_flag=True,
        agent_c_alive_flag=False,
        agent_d_active_flag=False,
        agent_d_alive_flag=False,
    )

    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, observation, *_ = step(config, state, joint_action, key)

    _assert_self_features_match_state_base_fields(observation, next_state)
    _assert_unused_self_feature_columns_are_zero(observation)
    _assert_unit_features_remain_checkpoint_2_placeholders(observation)
