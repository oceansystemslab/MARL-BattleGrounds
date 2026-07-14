"""Observation, visibility, and mask-contract tests for Milestone 4 Step 4.

This file owns env-level proof that Step 4 consumes geometry LOS correctly.
Low-level segment/obstacle geometry remains covered in ``test_geometry.py``.
"""
# pyright: reportPrivateUsage=false

from typing import TypedDict, cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_BASE_MOVEMENT_SPEED,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_OBSERVATION_RADIUS,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_TEAM_ID,
    AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    CLASS_NEUTRAL,
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    NO_TEAM_ID,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_FEATURES,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    SELF_FEATURES,
    TEAM_A_ID,
    TEAM_B_ID,
    UNIT_FEATURES,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    EnvConfig,
    EnvState,
    Observation,
)

# Test Helpers ---


class _CombatStateFields(TypedDict):
    """Keyword fields for inert combat state in test EnvState constructors."""

    current_health: Array
    ultimate_cooldowns: Array
    slow_durations: Array
    stun_durations: Array
    rogue_poison_anti_heal_durations: Array
    mage_burst_damage_amplification_durations: Array
    priest_blessing_of_freedom_slow_floor_durations: Array


def _inert_combat_state_fields() -> _CombatStateFields:
    """Return neutral combat fields for direct EnvState constructors."""
    return {
        "current_health": jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32),
        "ultimate_cooldowns": jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        "slow_durations": jnp.zeros(
            (MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32
        ),
        "stun_durations": jnp.zeros(
            (MAX_AGENT_SLOTS, NUM_STUN_CHANNELS), dtype=jnp.int32
        ),
        "rogue_poison_anti_heal_durations": jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        "mage_burst_damage_amplification_durations": jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        "priest_blessing_of_freedom_slow_floor_durations": jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
    }


def _empty_obstacles() -> Array:
    """Create a padded all-inactive obstacle table."""
    return jnp.zeros(
        (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
        dtype=jnp.float32,
    )


def _obstacle_array_with_rows(*rows: tuple[int, Array]) -> Array:
    """Create a padded obstacle array with selected slots populated."""
    obstacles = jnp.zeros(
        (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
        dtype=jnp.float32,
    )

    for slot, obstacle in rows:
        assert 0 <= slot < MAX_OBSTACLE_SLOTS
        obstacles = obstacles.at[slot].set(obstacle)

    return obstacles


def _pillar_obstacle(
    *,
    x: float,
    y: float,
    radius: float,
    active: bool = True,
) -> Array:
    """Create one pillar obstacle row."""
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(x)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(y)
    obstacle = obstacle.at[OBSTACLE_FEATURE_RADIUS].set(radius)
    obstacle = obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(float(active))
    return obstacle


def _wall_obstacle(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    theta: float = 0.0,
    active: bool = True,
) -> Array:
    """Create one wall obstacle row."""
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_WALL)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(x)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(y)
    obstacle = obstacle.at[OBSTACLE_FEATURE_WIDTH].set(width)
    obstacle = obstacle.at[OBSTACLE_FEATURE_HEIGHT].set(height)
    obstacle = obstacle.at[OBSTACLE_FEATURE_THETA].set(theta)
    obstacle = obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(float(active))
    return obstacle


def _relation_visibility_masks_with_rows(
    *,
    ally_rows: tuple[tuple[int, Array], ...] = (),
    enemy_rows: tuple[tuple[int, Array], ...] = (),
) -> tuple[Array, Array]:
    """Create expected relation-local visibility masks from sparse rows."""
    ally_mask = jnp.zeros((MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM), dtype=bool)
    enemy_mask = jnp.zeros((MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM), dtype=bool)

    for observer_slot, row in ally_rows:
        assert 0 <= observer_slot < MAX_AGENT_SLOTS
        assert row.shape == (MAX_AGENTS_PER_TEAM,)
        ally_mask = ally_mask.at[observer_slot].set(row)

    for observer_slot, row in enemy_rows:
        assert 0 <= observer_slot < MAX_AGENT_SLOTS
        assert row.shape == (MAX_AGENTS_PER_TEAM,)
        enemy_mask = enemy_mask.at[observer_slot].set(row)

    return (ally_mask, enemy_mask)


def _assert_visibility_masks_match(
    observation: Observation,
    expected_ally: Array,
    expected_enemy: Array,
) -> None:
    """Assert visibility masks keep their fixed contract and exact values."""
    assert observation.ally_visibility_mask.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
    )
    assert observation.enemy_visibility_mask.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
    )
    assert observation.ally_visibility_mask.dtype == bool
    assert observation.enemy_visibility_mask.dtype == bool

    assert bool(jnp.array_equal(observation.ally_visibility_mask, expected_ally))
    assert bool(jnp.array_equal(observation.enemy_visibility_mask, expected_enemy))


def _deterministic_config(
    *,
    team_size: int = 3,
    max_steps: int = 1000,
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
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
        agent_profile=resolve_agent_profile(
            jnp.full((MAX_AGENT_SLOTS,), CLASS_NEUTRAL, dtype=jnp.int32),
            jnp.asarray((team_size, team_size), dtype=jnp.int32),
        ),
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


def _slot_float_vector(
    default_value: float,
    *rows: tuple[int, Array | float],
) -> Array:
    """Create a float32 slot vector with selected overrides."""
    values = jnp.full((MAX_AGENT_SLOTS,), default_value, dtype=jnp.float32)

    for slot, value in rows:
        assert 0 <= slot < MAX_AGENT_SLOTS
        values = values.at[slot].set(value)

    return values


def _neutral_class_ids() -> Array:
    """Create a fixed neutral class-ID vector."""
    return jnp.full((MAX_AGENT_SLOTS,), CLASS_NEUTRAL, dtype=jnp.int32)


def _state_two_versus_two_game(
    config: EnvConfig,
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
    agent_a_class_id: int = CLASS_NEUTRAL,
    agent_b_class_id: int = CLASS_NEUTRAL,
    agent_c_class_id: int = CLASS_NEUTRAL,
    agent_d_class_id: int = CLASS_NEUTRAL,
    radius: float = 0.5,
    effective_movement_speed: float = 1.0,
    effective_observation_radius: float = 8.0,
    effective_basic_interaction_radius: float = 6.0,
    effective_ultimate_interaction_radius: float = 9.0,
    step_count: int = 0,
) -> tuple[EnvConfig, EnvState]:
    """Create an exact fixed two-versus-two config/state pair.

    Team A occupies global slots 0 and 1 in these scenarios. Team B occupies
    global slots 5 and 6, which become relation-local enemy slots 0 and 1 for
    Team A observers.
    """
    agent_a_index = 0
    agent_b_index = 1
    agent_c_index = MAX_AGENTS_PER_TEAM
    agent_d_index = MAX_AGENTS_PER_TEAM + 1

    active_mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)
    alive_mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)

    active_mask = active_mask.at[agent_a_index].set(agent_a_active_flag)
    alive_mask = alive_mask.at[agent_a_index].set(
        agent_a_active_flag and agent_a_alive_flag
    )

    active_mask = active_mask.at[agent_b_index].set(agent_b_active_flag)
    alive_mask = alive_mask.at[agent_b_index].set(
        agent_b_active_flag and agent_b_alive_flag
    )

    active_mask = active_mask.at[agent_c_index].set(agent_c_active_flag)
    alive_mask = alive_mask.at[agent_c_index].set(
        agent_c_active_flag and agent_c_alive_flag
    )

    active_mask = active_mask.at[agent_d_index].set(agent_d_active_flag)
    alive_mask = alive_mask.at[agent_d_index].set(
        agent_d_active_flag and agent_d_alive_flag
    )

    requested_class_ids = jnp.full((MAX_AGENT_SLOTS,), CLASS_NEUTRAL, dtype=jnp.int32)
    requested_class_ids = requested_class_ids.at[agent_a_index].set(
        agent_a_class_id if agent_a_active_flag else CLASS_NEUTRAL
    )
    requested_class_ids = requested_class_ids.at[agent_b_index].set(
        agent_b_class_id if agent_b_active_flag else CLASS_NEUTRAL
    )
    requested_class_ids = requested_class_ids.at[agent_c_index].set(
        agent_c_class_id if agent_c_active_flag else CLASS_NEUTRAL
    )
    requested_class_ids = requested_class_ids.at[agent_d_index].set(
        agent_d_class_id if agent_d_active_flag else CLASS_NEUTRAL
    )
    resolved_profile = resolve_agent_profile(
        requested_class_ids,
        jnp.asarray((2, 2), dtype=jnp.int32),
    )

    slot_team_ids = jnp.concatenate(
        (
            jnp.full((MAX_AGENTS_PER_TEAM,), TEAM_A_ID, dtype=jnp.int32),
            jnp.full((MAX_AGENTS_PER_TEAM,), TEAM_B_ID, dtype=jnp.int32),
        )
    )
    profile = resolved_profile._replace(
        team_ids=jnp.where(active_mask, slot_team_ids, NO_TEAM_ID),
        active_mask=active_mask,
        agent_radii=_agent_radii_array_with_rows(
            (agent_a_index, radius),
            (agent_b_index, radius),
            (agent_c_index, radius),
            (agent_d_index, radius),
        ),
        base_movement_speeds=_slot_float_vector(
            0.0,
            (agent_a_index, effective_movement_speed),
            (agent_b_index, effective_movement_speed),
            (agent_c_index, effective_movement_speed),
            (agent_d_index, effective_movement_speed),
        ),
        observation_radii=_slot_float_vector(
            0.0,
            (agent_a_index, effective_observation_radius),
            (agent_b_index, effective_observation_radius),
            (agent_c_index, effective_observation_radius),
            (agent_d_index, effective_observation_radius),
        ),
        basic_interaction_radii=_slot_float_vector(
            0.0,
            (agent_a_index, effective_basic_interaction_radius),
            (agent_b_index, effective_basic_interaction_radius),
            (agent_c_index, effective_basic_interaction_radius),
            (agent_d_index, effective_basic_interaction_radius),
        ),
        ultimate_interaction_radii=_slot_float_vector(
            0.0,
            (agent_a_index, effective_ultimate_interaction_radius),
            (agent_b_index, effective_ultimate_interaction_radius),
            (agent_c_index, effective_ultimate_interaction_radius),
            (agent_d_index, effective_ultimate_interaction_radius),
        ),
    )
    config = config._replace(agent_profile=profile)

    state = EnvState(
        step_count=jnp.array(step_count, dtype=jnp.int32),
        agent_positions=_agent_positions_array_with_rows(
            (agent_a_index, agent_a_position),
            (agent_b_index, agent_b_position),
            (agent_c_index, agent_c_position),
            (agent_d_index, agent_d_position),
        ),
        alive_mask=alive_mask,
        **_inert_combat_state_fields(),
    )
    return config, state


def _assert_targetability_masks_match(
    observation: Observation,
    expected_ally: Array,
    expected_enemy: Array,
) -> None:
    """Assert relation-local targetability masks have exact expected values."""
    assert observation.ally_targetability_mask.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
    )
    assert observation.enemy_targetability_mask.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
    )
    assert observation.ally_targetability_mask.dtype == bool
    assert observation.enemy_targetability_mask.dtype == bool

    assert bool(jnp.array_equal(observation.ally_targetability_mask, expected_ally))
    assert bool(jnp.array_equal(observation.enemy_targetability_mask, expected_enemy))


def _assert_targetability_never_exceeds_visibility(
    observation: Observation,
) -> None:
    """Assert targetability is always a subset of visibility."""
    illegal_ally_targetability = jnp.logical_and(
        observation.ally_targetability_mask,
        jnp.logical_not(observation.ally_visibility_mask),
    )
    illegal_enemy_targetability = jnp.logical_and(
        observation.enemy_targetability_mask,
        jnp.logical_not(observation.enemy_visibility_mask),
    )

    assert not bool(jnp.any(illegal_ally_targetability))
    assert not bool(jnp.any(illegal_enemy_targetability))


def _assert_flat_target_mask_matches_relation_targetability(
    *,
    action_mask: ActionMask,
    state: EnvState,
    config: EnvConfig,
    expected_ally: Array,
    expected_enemy: Array,
) -> None:
    """Assert flat target mask derives from none + ally/enemy targetability."""
    active_alive_mask_bc = jnp.logical_and(
        config.agent_profile.active_mask, state.alive_mask
    )[:, None]

    expected_target_mask = jnp.concatenate(
        (
            jnp.ones((MAX_AGENT_SLOTS, 1), dtype=bool),
            expected_ally,
            expected_enemy,
        ),
        axis=1,
    )
    expected_target_mask = jnp.logical_and(
        expected_target_mask,
        active_alive_mask_bc,
    )

    assert action_mask.target.shape == (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)
    assert action_mask.target.dtype == bool
    assert bool(jnp.array_equal(action_mask.target, expected_target_mask))

    assert bool(
        jnp.array_equal(
            action_mask.target[:, 0],
            jnp.logical_and(config.agent_profile.active_mask, state.alive_mask),
        )
    )
    assert bool(
        jnp.array_equal(
            action_mask.target[:, 1 : 1 + MAX_AGENTS_PER_TEAM],
            jnp.logical_and(expected_ally, active_alive_mask_bc),
        )
    )
    assert bool(
        jnp.array_equal(
            action_mask.target[:, 1 + MAX_AGENTS_PER_TEAM :],
            jnp.logical_and(expected_enemy, active_alive_mask_bc),
        )
    )


def _assert_self_features_match_state_base_fields(
    observation: Observation,
    state: EnvState,
    config: EnvConfig,
) -> None:
    """Assert that self features expose the shared agent spatial/state fields."""
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
            config.agent_profile.agent_radii,
            atol=0.0,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.allclose(
            observation.self_features[:, AGENT_FEATURE_TEAM_ID],
            config.agent_profile.team_ids.astype(jnp.float32),
            atol=0.0,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.allclose(
            observation.self_features[:, AGENT_FEATURE_ACTIVE],
            config.agent_profile.active_mask.astype(jnp.float32),
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


def _assert_unit_feature_row_matches_self_row(
    observation: Observation,
    *,
    expected_global_slot: int,
    result_row: Array,
) -> None:
    """Assert a visible relation-local candidate row matches its global self row."""
    assert result_row.shape == (UNIT_FEATURES,)
    assert observation.self_features[expected_global_slot].shape == (SELF_FEATURES,)
    assert bool(
        jnp.allclose(
            result_row,
            observation.self_features[expected_global_slot],
            atol=0.0,
            rtol=0.0,
        )
    )


def _assert_unit_feature_row_is_zero(row: Array) -> None:
    """Assert a hidden relation-local candidate row leaks no feature values."""
    assert row.shape == (UNIT_FEATURES,)
    assert bool(
        jnp.allclose(
            row,
            jnp.zeros((UNIT_FEATURES,), dtype=jnp.float32),
            atol=0.0,
            rtol=0.0,
        )
    )


def _assert_action_mask_row_is_false(row: Array) -> None:
    """Assert one action-mask row contains no valid actions."""
    assert bool(jnp.all(jnp.logical_not(row)))


def _assert_self_features_match_state_effective_fields(
    observation: Observation,
    config: EnvConfig,
) -> None:
    """Assert that self features expose the shared agent class/stat fields."""
    assert bool(
        jnp.allclose(
            observation.self_features[:, AGENT_FEATURE_CLASS_ID],
            config.agent_profile.class_ids.astype(jnp.float32),
            atol=0.0,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.allclose(
            observation.self_features[:, AGENT_FEATURE_BASE_MOVEMENT_SPEED],
            config.agent_profile.base_movement_speeds,
            atol=0.0,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.allclose(
            observation.self_features[:, AGENT_FEATURE_OBSERVATION_RADIUS],
            config.agent_profile.observation_radii,
            atol=0.0,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.allclose(
            observation.self_features[:, AGENT_FEATURE_BASIC_INTERACTION_RADIUS],
            config.agent_profile.basic_interaction_radii,
            atol=0.0,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.allclose(
            observation.self_features[:, AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS],
            config.agent_profile.ultimate_interaction_radii,
            atol=0.0,
            rtol=0.0,
        )
    )


# Tests ---


def test_reset_self_features_match_reset_state_base_fields() -> None:
    config = _deterministic_config()
    key = jax.random.key(42)

    initial_state, observation, *_ = reset(config, key)

    _assert_self_features_match_state_base_fields(observation, initial_state, config)
    _assert_self_features_match_state_effective_fields(observation, config)


def test_step_self_features_match_next_state_base_fields() -> None:
    config = _deterministic_config()
    key = jax.random.key(42)

    config, state = _state_two_versus_two_game(
        config,
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

    _assert_self_features_match_state_base_fields(observation, next_state, config)
    _assert_self_features_match_state_effective_fields(observation, config)


def test_reset_padded_self_rows_are_deterministic_inactive_dummy_rows() -> None:
    config = _deterministic_config(team_size=3)
    first_state, first_observation, first_action_mask, _ = reset(
        config, jax.random.key(1)
    )
    second_state, second_observation, second_action_mask, _ = reset(
        config, jax.random.key(2)
    )
    inactive_mask = jnp.logical_not(config.agent_profile.active_mask)

    assert bool(
        jnp.all(
            first_observation.self_features[:, AGENT_FEATURE_ACTIVE]
            == config.agent_profile.active_mask.astype(jnp.float32)
        )
    )
    assert bool(
        jnp.all(
            first_observation.self_features[:, AGENT_FEATURE_ALIVE]
            == first_state.alive_mask.astype(jnp.float32)
        )
    )
    assert bool(jnp.all(first_observation.self_features[inactive_mask] >= 0.0))
    assert bool(
        jnp.array_equal(
            first_observation.self_features[inactive_mask],
            second_observation.self_features[inactive_mask],
        )
    )
    assert bool(jnp.array_equal(first_state.alive_mask, second_state.alive_mask))
    assert not bool(jnp.any(first_action_mask.move[inactive_mask, :]))
    assert not bool(jnp.any(first_action_mask.target[inactive_mask, :]))
    assert not bool(jnp.any(first_action_mask.use_ultimate[inactive_mask, :]))
    assert not bool(jnp.any(second_action_mask.move[inactive_mask, :]))
    assert not bool(jnp.any(second_action_mask.target[inactive_mask, :]))
    assert not bool(jnp.any(second_action_mask.use_ultimate[inactive_mask, :]))


def test_active_dead_self_rows_remain_active_but_not_actionable() -> None:
    config = _deterministic_config()
    key = jax.random.key(42)
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([5.0, 5.0], dtype=jnp.float32),
        agent_b_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
        agent_c_position=jnp.array([5.0, 8.0], dtype=jnp.float32),
        agent_d_position=jnp.array([8.0, 8.0], dtype=jnp.float32),
        agent_a_active_flag=True,
        agent_a_alive_flag=False,
        agent_b_active_flag=False,
        agent_c_active_flag=False,
        agent_d_active_flag=False,
    )

    _, observation, _, _, action_mask, _ = step(
        config, state, _joint_action_with_moves(), key
    )

    assert observation.self_features[0, AGENT_FEATURE_ACTIVE] == 1.0
    assert observation.self_features[0, AGENT_FEATURE_ALIVE] == 0.0
    assert not bool(jnp.any(action_mask.move[0, :]))
    assert not bool(jnp.any(action_mask.target[0, :]))
    assert not bool(jnp.any(action_mask.use_ultimate[0, :]))


def test_visibility_uses_state_observation_radii_not_config_default() -> None:
    """Visibility must consume effective per-slot state radii."""
    config = _deterministic_config()
    key = jax.random.key(42)
    joint_action = _joint_action_with_moves()
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([7.0, 2.0], dtype=jnp.float32),
        agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
        agent_b_active_flag=False,
        agent_d_active_flag=False,
        effective_observation_radius=3.0,
    )

    _, observation, *_ = step(config, state, joint_action, key)

    expected_ally, expected_enemy = _relation_visibility_masks_with_rows(
        ally_rows=(
            (0, jnp.array([True, False, False, False, False])),
            (5, jnp.array([True, False, False, False, False])),
        )
    )

    _assert_visibility_masks_match(observation, expected_ally, expected_enemy)


def test_visibility_is_directed_by_each_observer_radius() -> None:
    """A can see B does not imply B can see A when radii differ."""
    config = _deterministic_config()
    key = jax.random.key(42)
    joint_action = _joint_action_with_moves()
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([7.0, 2.0], dtype=jnp.float32),
        agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
        agent_b_active_flag=False,
        agent_d_active_flag=False,
    )
    config = config._replace(
        agent_profile=config.agent_profile._replace(
            observation_radii=_slot_float_vector(
                0.0,
                (0, 6.0),
                (MAX_AGENTS_PER_TEAM, 3.0),
            )
        )
    )

    _, observation, *_ = step(config, state, joint_action, key)

    expected_ally, expected_enemy = _relation_visibility_masks_with_rows(
        ally_rows=(
            (0, jnp.array([True, False, False, False, False])),
            (5, jnp.array([True, False, False, False, False])),
        ),
        enemy_rows=((0, jnp.array([True, False, False, False, False])),),
    )

    _assert_visibility_masks_match(observation, expected_ally, expected_enemy)


def test_visibility_ignores_basic_and_ultimate_interaction_radii() -> None:
    """Visibility radius must stay independent from interaction radii."""
    config = _deterministic_config()
    key = jax.random.key(42)
    joint_action = _joint_action_with_moves()
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([7.0, 2.0], dtype=jnp.float32),
        agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
        agent_b_active_flag=False,
        agent_d_active_flag=False,
        effective_observation_radius=8.0,
        effective_basic_interaction_radius=0.25,
        effective_ultimate_interaction_radius=0.25,
    )

    _, observation, *_ = step(config, state, joint_action, key)

    expected_ally, expected_enemy = _relation_visibility_masks_with_rows(
        ally_rows=(
            (0, jnp.array([True, False, False, False, False])),
            (5, jnp.array([True, False, False, False, False])),
        ),
        enemy_rows=(
            (0, jnp.array([True, False, False, False, False])),
            (5, jnp.array([True, False, False, False, False])),
        ),
    )

    _assert_visibility_masks_match(observation, expected_ally, expected_enemy)


@pytest.mark.parametrize(
    ("obstacles", "scenario", "expected_ally", "expected_enemy"),
    [
        pytest.param(
            _empty_obstacles(),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
                agent_c_position=jnp.array([2.0, 5.0], dtype=jnp.float32),
                agent_d_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
                effective_observation_radius=8.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=(
                    (0, jnp.array([True, True, False, False, False])),
                    (1, jnp.array([True, True, False, False, False])),
                    (5, jnp.array([True, True, False, False, False])),
                    (6, jnp.array([True, True, False, False, False])),
                ),
                enemy_rows=(
                    (0, jnp.array([True, True, False, False, False])),
                    (1, jnp.array([True, True, False, False, False])),
                    (5, jnp.array([True, True, False, False, False])),
                    (6, jnp.array([True, True, False, False, False])),
                ),
            ),
            id="clear_los_inside_radius_visible",
        ),
        pytest.param(
            _empty_obstacles(),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
                agent_c_position=jnp.array([15.0, 10.0], dtype=jnp.float32),
                agent_d_position=jnp.array([16.0, 10.0], dtype=jnp.float32),
                effective_observation_radius=3.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=(
                    (0, jnp.array([True, True, False, False, False])),
                    (1, jnp.array([True, True, False, False, False])),
                    (5, jnp.array([True, True, False, False, False])),
                    (6, jnp.array([True, True, False, False, False])),
                ),
                enemy_rows=(),
            ),
            id="outside_observation_radius_not_visible",
        ),
        pytest.param(
            _empty_obstacles(),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
                agent_c_position=jnp.array([2.0, 5.0], dtype=jnp.float32),
                agent_d_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
                agent_b_active_flag=False,
                agent_c_alive_flag=False,
                effective_observation_radius=8.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=(
                    (0, jnp.array([True, False, False, False, False])),
                    (6, jnp.array([False, True, False, False, False])),
                ),
                enemy_rows=(
                    (0, jnp.array([False, True, False, False, False])),
                    (6, jnp.array([True, False, False, False, False])),
                ),
            ),
            id="inactive_or_dead_candidates_not_visible",
        ),
        pytest.param(
            _empty_obstacles(),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
                agent_c_position=jnp.array([2.0, 5.0], dtype=jnp.float32),
                agent_d_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
                agent_a_alive_flag=False,
                agent_d_active_flag=False,
                effective_observation_radius=8.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=(
                    (1, jnp.array([False, True, False, False, False])),
                    (5, jnp.array([True, False, False, False, False])),
                ),
                enemy_rows=(
                    (1, jnp.array([True, False, False, False, False])),
                    (5, jnp.array([False, True, False, False, False])),
                ),
            ),
            id="inactive_or_dead_observers_see_nothing",
        ),
        pytest.param(
            _obstacle_array_with_rows(
                (
                    0,
                    _pillar_obstacle(
                        x=5.0,
                        y=2.0,
                        radius=0.75,
                        active=True,
                    ),
                )
            ),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
                agent_c_position=jnp.array([8.0, 2.0], dtype=jnp.float32),
                agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
                agent_b_active_flag=False,
                agent_d_active_flag=False,
                effective_observation_radius=10.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=(
                    (0, jnp.array([True, False, False, False, False])),
                    (5, jnp.array([True, False, False, False, False])),
                ),
                enemy_rows=(),
            ),
            id="active_pillar_blocks_visibility",
        ),
        pytest.param(
            _obstacle_array_with_rows(
                (
                    0,
                    _pillar_obstacle(
                        x=5.0,
                        y=2.0,
                        radius=0.75,
                        active=False,
                    ),
                )
            ),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
                agent_c_position=jnp.array([8.0, 2.0], dtype=jnp.float32),
                agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
                agent_b_active_flag=False,
                agent_d_active_flag=False,
                effective_observation_radius=10.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=(
                    (0, jnp.array([True, False, False, False, False])),
                    (5, jnp.array([True, False, False, False, False])),
                ),
                enemy_rows=(
                    (0, jnp.array([True, False, False, False, False])),
                    (5, jnp.array([True, False, False, False, False])),
                ),
            ),
            id="inactive_pillar_does_not_block_visibility",
        ),
        pytest.param(
            _obstacle_array_with_rows(
                (
                    0,
                    _wall_obstacle(
                        x=5.0,
                        y=2.0,
                        width=0.5,
                        height=4.0,
                        theta=0.0,
                        active=True,
                    ),
                )
            ),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
                agent_c_position=jnp.array([8.0, 2.0], dtype=jnp.float32),
                agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
                agent_b_active_flag=False,
                agent_d_active_flag=False,
                effective_observation_radius=10.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=(
                    (0, jnp.array([True, False, False, False, False])),
                    (5, jnp.array([True, False, False, False, False])),
                ),
                enemy_rows=(),
            ),
            id="active_wall_blocks_visibility",
        ),
        pytest.param(
            _obstacle_array_with_rows(
                (
                    0,
                    _wall_obstacle(
                        x=5.0,
                        y=2.0,
                        width=0.5,
                        height=4.0,
                        theta=0.0,
                        active=False,
                    ),
                )
            ),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
                agent_c_position=jnp.array([8.0, 2.0], dtype=jnp.float32),
                agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
                agent_b_active_flag=False,
                agent_d_active_flag=False,
                effective_observation_radius=10.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=(
                    (0, jnp.array([True, False, False, False, False])),
                    (5, jnp.array([True, False, False, False, False])),
                ),
                enemy_rows=(
                    (0, jnp.array([True, False, False, False, False])),
                    (5, jnp.array([True, False, False, False, False])),
                ),
            ),
            id="inactive_wall_does_not_block_visibility",
        ),
        pytest.param(
            _obstacle_array_with_rows(
                (
                    0,
                    _wall_obstacle(
                        x=5.0,
                        y=2.0,
                        width=0.5,
                        height=4.0,
                        theta=0.7853982,
                        active=True,
                    ),
                )
            ),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
                agent_c_position=jnp.array([8.0, 2.0], dtype=jnp.float32),
                agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
                agent_b_active_flag=False,
                agent_d_active_flag=False,
                effective_observation_radius=10.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=(
                    (0, jnp.array([True, False, False, False, False])),
                    (5, jnp.array([True, False, False, False, False])),
                ),
                enemy_rows=(),
            ),
            id="active_rotated_wall_blocks_visibility",
        ),
        pytest.param(
            _empty_obstacles(),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([5.0, 2.0], dtype=jnp.float32),
                agent_c_position=jnp.array([8.0, 2.0], dtype=jnp.float32),
                agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
                agent_d_active_flag=False,
                effective_observation_radius=10.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=(
                    (0, jnp.array([True, True, False, False, False])),
                    (1, jnp.array([True, True, False, False, False])),
                    (5, jnp.array([True, False, False, False, False])),
                ),
                enemy_rows=(
                    (0, jnp.array([True, False, False, False, False])),
                    (1, jnp.array([True, False, False, False, False])),
                    (5, jnp.array([True, True, False, False, False])),
                ),
            ),
            id="agents_do_not_block_visibility",
        ),
    ],
)
def test_visibility_masks(
    obstacles: Array,
    scenario: tuple[EnvConfig, EnvState],
    expected_ally: Array,
    expected_enemy: Array,
) -> None:
    """Assert env-level LOS-gated visibility across representative scenarios."""
    config, state = scenario
    config = config._replace(obstacles=obstacles)
    key = jax.random.key(42)
    joint_action = _joint_action_with_moves()

    _, observation, *_ = step(config, state, joint_action, key)

    _assert_visibility_masks_match(observation, expected_ally, expected_enemy)


def test_visible_candidate_rows_match_shared_self_feature_schema() -> None:
    """Visible relation-local rows expose the candidate's shared agent schema."""
    config = _deterministic_config(obstacles=_empty_obstacles())
    key = jax.random.key(42)
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([2.0, 5.0], dtype=jnp.float32),
        agent_d_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
        effective_observation_radius=8.0,
    )
    profile = config.agent_profile._replace(
        agent_radii=_slot_float_vector(
            0.5,
            (0, 0.35),
            (1, 0.45),
            (MAX_AGENTS_PER_TEAM, 0.55),
            (MAX_AGENTS_PER_TEAM + 1, 0.65),
        ),
        class_ids=_neutral_class_ids()
        .at[0]
        .set(MAGE_CLASS_ID)
        .at[1]
        .set(WARRIOR_CLASS_ID)
        .at[MAX_AGENTS_PER_TEAM]
        .set(HUNTER_CLASS_ID)
        .at[MAX_AGENTS_PER_TEAM + 1]
        .set(ROGUE_CLASS_ID),
        base_movement_speeds=_slot_float_vector(
            0.0,
            (0, 1.25),
            (1, 1.50),
            (MAX_AGENTS_PER_TEAM, 1.75),
            (MAX_AGENTS_PER_TEAM + 1, 2.00),
        ),
        observation_radii=_slot_float_vector(
            0.0,
            (0, 8.0),
            (1, 8.5),
            (MAX_AGENTS_PER_TEAM, 9.0),
            (MAX_AGENTS_PER_TEAM + 1, 9.5),
        ),
        basic_interaction_radii=_slot_float_vector(
            0.0,
            (0, 3.25),
            (1, 3.50),
            (MAX_AGENTS_PER_TEAM, 3.75),
            (MAX_AGENTS_PER_TEAM + 1, 4.00),
        ),
        ultimate_interaction_radii=_slot_float_vector(
            0.0,
            (0, 5.25),
            (1, 5.50),
            (MAX_AGENTS_PER_TEAM, 5.75),
            (MAX_AGENTS_PER_TEAM + 1, 6.00),
        ),
    )
    config = config._replace(agent_profile=profile)

    _, observation, *_ = step(config, state, _joint_action_with_moves(), key)

    _assert_unit_feature_row_matches_self_row(
        observation,
        expected_global_slot=0,
        result_row=observation.ally_unit_features[0, 0],
    )
    _assert_unit_feature_row_matches_self_row(
        observation,
        expected_global_slot=1,
        result_row=observation.ally_unit_features[0, 1],
    )
    _assert_unit_feature_row_matches_self_row(
        observation,
        expected_global_slot=MAX_AGENTS_PER_TEAM,
        result_row=observation.enemy_unit_features[0, 0],
    )
    _assert_unit_feature_row_matches_self_row(
        observation,
        expected_global_slot=MAX_AGENTS_PER_TEAM + 1,
        result_row=observation.enemy_unit_features[0, 1],
    )

    _assert_unit_feature_row_matches_self_row(
        observation,
        expected_global_slot=MAX_AGENTS_PER_TEAM,
        result_row=observation.ally_unit_features[MAX_AGENTS_PER_TEAM, 0],
    )
    _assert_unit_feature_row_matches_self_row(
        observation,
        expected_global_slot=MAX_AGENTS_PER_TEAM + 1,
        result_row=observation.ally_unit_features[MAX_AGENTS_PER_TEAM, 1],
    )
    _assert_unit_feature_row_matches_self_row(
        observation,
        expected_global_slot=0,
        result_row=observation.enemy_unit_features[MAX_AGENTS_PER_TEAM, 0],
    )
    _assert_unit_feature_row_matches_self_row(
        observation,
        expected_global_slot=1,
        result_row=observation.enemy_unit_features[MAX_AGENTS_PER_TEAM, 1],
    )


def test_visible_candidate_rows_preserve_non_boolean_numeric_values() -> None:
    """Feature masking must preserve float values rather than booleanizing rows."""
    config = _deterministic_config(obstacles=_empty_obstacles())
    key = jax.random.key(42)
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([1.25, 1.75], dtype=jnp.float32),
        agent_b_position=jnp.array([2.50, 1.25], dtype=jnp.float32),
        agent_c_position=jnp.array([7.50, 1.50], dtype=jnp.float32),
        agent_d_position=jnp.array([8.25, 1.75], dtype=jnp.float32),
        effective_observation_radius=20.0,
        effective_movement_speed=1.75,
        effective_basic_interaction_radius=4.25,
        effective_ultimate_interaction_radius=6.75,
    )

    next_state, observation, *_ = step(config, state, _joint_action_with_moves(), key)

    candidate_slot = MAX_AGENTS_PER_TEAM
    candidate_row = observation.enemy_unit_features[0, 0]
    assert bool(observation.enemy_visibility_mask[0, 0])
    assert bool(
        jnp.isclose(
            candidate_row[AGENT_FEATURE_X],
            next_state.agent_positions[candidate_slot, 0],
        )
    )
    assert bool(
        jnp.isclose(
            candidate_row[AGENT_FEATURE_Y],
            next_state.agent_positions[candidate_slot, 1],
        )
    )
    assert bool(
        jnp.isclose(
            candidate_row[AGENT_FEATURE_BASE_MOVEMENT_SPEED],
            config.agent_profile.base_movement_speeds[candidate_slot],
        )
    )
    assert bool(
        jnp.isclose(
            candidate_row[AGENT_FEATURE_BASIC_INTERACTION_RADIUS],
            config.agent_profile.basic_interaction_radii[candidate_slot],
        )
    )
    assert bool(
        jnp.isclose(
            candidate_row[AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS],
            config.agent_profile.ultimate_interaction_radii[candidate_slot],
        )
    )
    _assert_unit_feature_row_matches_self_row(
        observation,
        expected_global_slot=candidate_slot,
        result_row=candidate_row,
    )


def test_los_blocked_candidate_rows_are_fully_zero() -> None:
    """LOS-blocked candidates must not leak any dynamic unit feature values."""
    obstacles = _obstacle_array_with_rows(
        (
            0,
            _wall_obstacle(
                x=5.0,
                y=2.0,
                width=0.5,
                height=4.0,
                theta=0.0,
                active=True,
            ),
        )
    )
    config = _deterministic_config(obstacles=obstacles)
    key = jax.random.key(42)
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
        agent_c_position=jnp.array([8.0, 2.0], dtype=jnp.float32),
        agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
        agent_b_active_flag=False,
        agent_d_active_flag=False,
        effective_observation_radius=10.0,
    )

    _, observation, *_ = step(config, state, _joint_action_with_moves(), key)

    assert not bool(observation.enemy_visibility_mask[0, 0])
    assert not bool(observation.enemy_visibility_mask[MAX_AGENTS_PER_TEAM, 0])
    _assert_unit_feature_row_is_zero(observation.enemy_unit_features[0, 0])
    _assert_unit_feature_row_is_zero(
        observation.enemy_unit_features[MAX_AGENTS_PER_TEAM, 0]
    )


def test_out_of_radius_candidate_rows_are_fully_zero() -> None:
    """Out-of-radius candidates must not leak any dynamic unit feature values."""
    config = _deterministic_config(obstacles=_empty_obstacles())
    key = jax.random.key(42)
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([15.0, 10.0], dtype=jnp.float32),
        agent_d_position=jnp.array([16.0, 10.0], dtype=jnp.float32),
        effective_observation_radius=3.0,
    )

    _, observation, *_ = step(config, state, _joint_action_with_moves(), key)

    assert not bool(observation.enemy_visibility_mask[0, 0])
    assert not bool(observation.enemy_visibility_mask[0, 1])
    _assert_unit_feature_row_is_zero(observation.enemy_unit_features[0, 0])
    _assert_unit_feature_row_is_zero(observation.enemy_unit_features[0, 1])


def test_inactive_dead_and_padded_candidate_rows_are_fully_zero() -> None:
    """Inactive, dead, and padded candidates must have zero candidate rows."""
    config = _deterministic_config(obstacles=_empty_obstacles())
    key = jax.random.key(42)
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([2.0, 5.0], dtype=jnp.float32),
        agent_d_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
        agent_b_active_flag=False,
        agent_c_alive_flag=False,
        effective_observation_radius=8.0,
    )

    _, observation, *_ = step(config, state, _joint_action_with_moves(), key)

    assert not bool(observation.ally_visibility_mask[0, 1])
    _assert_unit_feature_row_is_zero(observation.ally_unit_features[0, 1])

    assert not bool(observation.enemy_visibility_mask[0, 0])
    _assert_unit_feature_row_is_zero(observation.enemy_unit_features[0, 0])

    assert not bool(observation.ally_visibility_mask[0, 2])
    _assert_unit_feature_row_is_zero(observation.ally_unit_features[0, 2])

    assert not bool(observation.enemy_visibility_mask[0, 2])
    _assert_unit_feature_row_is_zero(observation.enemy_unit_features[0, 2])


def test_candidate_visibility_masking_does_not_alter_self_features() -> None:
    """Candidate visibility masking must not mutate canonical self rows."""
    obstacles = _obstacle_array_with_rows(
        (
            0,
            _wall_obstacle(
                x=5.0,
                y=2.0,
                width=0.5,
                height=4.0,
                theta=0.0,
                active=True,
            ),
        )
    )
    config = _deterministic_config(obstacles=obstacles)
    key = jax.random.key(42)
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
        agent_c_position=jnp.array([8.0, 2.0], dtype=jnp.float32),
        agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
        agent_b_active_flag=False,
        agent_d_active_flag=False,
        effective_observation_radius=10.0,
    )

    next_state, observation, *_ = step(config, state, _joint_action_with_moves(), key)

    _assert_self_features_match_state_base_fields(observation, next_state, config)
    _assert_self_features_match_state_effective_fields(observation, config)


def test_active_dead_rows_remain_active_but_candidate_rows_actions_are_masked() -> None:
    """Dead active slots keep self state but cannot act or appear as candidates."""
    config = _deterministic_config(obstacles=_empty_obstacles())
    key = jax.random.key(42)
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([2.0, 5.0], dtype=jnp.float32),
        agent_d_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
        agent_a_active_flag=True,
        agent_a_alive_flag=False,
        effective_observation_radius=8.0,
    )

    _, observation, _, _, action_mask, _ = step(
        config,
        state,
        _joint_action_with_moves(),
        key,
    )

    assert observation.self_features[0, AGENT_FEATURE_ACTIVE] == 1.0
    assert observation.self_features[0, AGENT_FEATURE_ALIVE] == 0.0

    _assert_action_mask_row_is_false(action_mask.move[0])
    _assert_action_mask_row_is_false(action_mask.target[0])
    _assert_action_mask_row_is_false(action_mask.use_ultimate[0])

    assert not bool(observation.enemy_visibility_mask[MAX_AGENTS_PER_TEAM, 0])
    _assert_unit_feature_row_is_zero(
        observation.enemy_unit_features[MAX_AGENTS_PER_TEAM, 0]
    )


def test_expanded_unit_feature_columns_obey_full_row_visibility_masking() -> None:
    """Expanded combat columns remain present when visible and zero when hidden."""
    config = _deterministic_config(obstacles=_empty_obstacles())
    key = jax.random.key(42)
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([2.0, 5.0], dtype=jnp.float32),
        agent_d_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
        effective_observation_radius=8.0,
    )

    _, observation, *_ = step(config, state, _joint_action_with_moves(), key)

    expanded_start = AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS + 1
    relation_families = (
        (
            observation.ally_unit_features,
            observation.ally_visibility_mask,
        ),
        (
            observation.enemy_unit_features,
            observation.enemy_visibility_mask,
        ),
    )
    for features, visibility in relation_families:
        assert bool(jnp.any(features[visibility, expanded_start:] != 0.0))
        assert bool(jnp.all(features[jnp.logical_not(visibility)] == 0.0))


def test_map_obstacle_features_remain_globally_observed_after_candidate_masking() -> (
    None
):
    """Static map geometry remains globally observed despite dynamic-unit masking."""
    obstacles = _obstacle_array_with_rows(
        (0, _pillar_obstacle(x=2.0, y=2.0, radius=0.5, active=True)),
        (
            1,
            _wall_obstacle(
                x=5.0,
                y=5.0,
                width=1.0,
                height=3.0,
                theta=0.25,
                active=True,
            ),
        ),
    )
    config = _deterministic_config(obstacles=obstacles)
    key = jax.random.key(42)
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([15.0, 10.0], dtype=jnp.float32),
        agent_d_position=jnp.array([16.0, 10.0], dtype=jnp.float32),
        effective_observation_radius=3.0,
    )

    _, observation, *_ = step(config, state, _joint_action_with_moves(), key)

    assert bool(
        jnp.allclose(
            observation.map_obstacle_features,
            jnp.broadcast_to(
                obstacles[None, :, :],
                observation.map_obstacle_features.shape,
            ),
            atol=0.0,
            rtol=0.0,
        )
    )


@pytest.mark.parametrize(
    ("config", "scenario", "expected_ally", "expected_enemy"),
    [
        pytest.param(
            _deterministic_config(),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([3.0, 2.0], dtype=jnp.float32),
                agent_c_position=jnp.array([2.0, 4.0], dtype=jnp.float32),
                agent_d_position=jnp.array([4.0, 4.0], dtype=jnp.float32),
                agent_a_class_id=MAGE_CLASS_ID,
                agent_b_class_id=PRIEST_CLASS_ID,
                agent_c_class_id=MAGE_CLASS_ID,
                agent_d_class_id=PRIEST_CLASS_ID,
                effective_observation_radius=10.0,
                effective_basic_interaction_radius=5.0,
                effective_ultimate_interaction_radius=9.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=(
                    (1, jnp.array([True, True, False, False, False])),
                    (6, jnp.array([True, True, False, False, False])),
                ),
                enemy_rows=(
                    (0, jnp.array([True, True, False, False, False])),
                    (5, jnp.array([True, True, False, False, False])),
                ),
            ),
            id="visible_inside_basic_radius_candidates_are_targetable",
        ),
        pytest.param(
            _deterministic_config(),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([3.0, 2.0], dtype=jnp.float32),
                agent_c_position=jnp.array([7.0, 2.0], dtype=jnp.float32),
                agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
                agent_d_active_flag=False,
                agent_a_class_id=MAGE_CLASS_ID,
                agent_b_class_id=PRIEST_CLASS_ID,
                agent_c_class_id=MAGE_CLASS_ID,
                effective_observation_radius=10.0,
                effective_basic_interaction_radius=2.5,
                effective_ultimate_interaction_radius=9.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=((1, jnp.array([True, True, False, False, False])),),
                enemy_rows=(),
            ),
            id="visible_outside_basic_radius_candidates_are_not_targetable",
        ),
        pytest.param(
            _deterministic_config(
                obstacles=_obstacle_array_with_rows(
                    (
                        0,
                        _wall_obstacle(
                            x=5.0,
                            y=2.0,
                            width=0.5,
                            height=4.0,
                            theta=0.0,
                            active=True,
                        ),
                    )
                )
            ),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([4.0, 5.0], dtype=jnp.float32),
                agent_c_position=jnp.array([8.0, 2.0], dtype=jnp.float32),
                agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
                agent_b_active_flag=False,
                agent_d_active_flag=False,
                agent_a_class_id=MAGE_CLASS_ID,
                agent_c_class_id=MAGE_CLASS_ID,
                effective_observation_radius=10.0,
                effective_basic_interaction_radius=10.0,
                effective_ultimate_interaction_radius=10.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=(),
                enemy_rows=(),
            ),
            id="los_blocked_candidates_are_not_targetable",
        ),
        pytest.param(
            _deterministic_config(),
            _state_two_versus_two_game(
                _deterministic_config(),
                agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
                agent_b_position=jnp.array([3.0, 2.0], dtype=jnp.float32),
                agent_c_position=jnp.array([2.0, 4.0], dtype=jnp.float32),
                agent_d_position=jnp.array([3.0, 4.0], dtype=jnp.float32),
                agent_b_active_flag=False,
                agent_c_alive_flag=False,
                agent_a_class_id=MAGE_CLASS_ID,
                agent_c_class_id=MAGE_CLASS_ID,
                agent_d_class_id=PRIEST_CLASS_ID,
                effective_observation_radius=8.0,
                effective_basic_interaction_radius=5.0,
                effective_ultimate_interaction_radius=9.0,
            ),
            *_relation_visibility_masks_with_rows(
                ally_rows=((6, jnp.array([False, True, False, False, False])),),
                enemy_rows=((0, jnp.array([False, True, False, False, False])),),
            ),
            id="inactive_dead_and_padded_candidates_are_not_targetable",
        ),
    ],
)
def test_basic_targetability_masks(
    config: EnvConfig,
    scenario: tuple[EnvConfig, EnvState],
    expected_ally: Array,
    expected_enemy: Array,
) -> None:
    """Assert class-aware basic targetability across spatial scenarios."""
    scenario_config, state = scenario
    config = scenario_config._replace(obstacles=config.obstacles)
    next_state, observation, _, _, action_mask, _ = step(
        config,
        state,
        _joint_action_with_moves(),
        jax.random.key(42),
    )

    _assert_targetability_masks_match(observation, expected_ally, expected_enemy)
    _assert_targetability_never_exceeds_visibility(observation)
    _assert_flat_target_mask_matches_relation_targetability(
        action_mask=action_mask,
        state=next_state,
        config=config,
        expected_ally=expected_ally,
        expected_enemy=expected_enemy,
    )


def test_basic_targetability_uses_observer_specific_basic_interaction_radius() -> None:
    """Assert each observer uses its own current basic interaction radius."""
    config = _deterministic_config()
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([6.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([4.0, 2.0], dtype=jnp.float32),
        agent_d_position=jnp.array([9.0, 2.0], dtype=jnp.float32),
        agent_d_active_flag=False,
        agent_a_class_id=MAGE_CLASS_ID,
        agent_b_class_id=MAGE_CLASS_ID,
        agent_c_class_id=PRIEST_CLASS_ID,
        effective_observation_radius=10.0,
        effective_basic_interaction_radius=6.0,
    )
    config = config._replace(
        agent_profile=config.agent_profile._replace(
            basic_interaction_radii=_slot_float_vector(
                0.0,
                (0, 1.5),
                (1, 3.0),
                (MAX_AGENTS_PER_TEAM, 1.0),
            )
        )
    )

    expected_ally, expected_enemy = _relation_visibility_masks_with_rows(
        ally_rows=(
            (MAX_AGENTS_PER_TEAM, jnp.array([True, False, False, False, False])),
        ),
        enemy_rows=((1, jnp.array([True, False, False, False, False])),),
    )

    next_state, observation, _, _, action_mask, _ = step(
        config,
        state,
        _joint_action_with_moves(),
        jax.random.key(42),
    )

    _assert_targetability_masks_match(observation, expected_ally, expected_enemy)
    _assert_flat_target_mask_matches_relation_targetability(
        action_mask=action_mask,
        state=next_state,
        config=config,
        expected_ally=expected_ally,
        expected_enemy=expected_enemy,
    )


def test_observation_radius_does_not_substitute_for_basic_interaction_radius() -> None:
    """Assert visible units outside basic interaction radius are not targetable."""
    config = _deterministic_config()
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([3.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([7.0, 2.0], dtype=jnp.float32),
        agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
        agent_b_active_flag=False,
        agent_d_active_flag=False,
        agent_a_class_id=MAGE_CLASS_ID,
        agent_c_class_id=MAGE_CLASS_ID,
        effective_observation_radius=10.0,
        effective_basic_interaction_radius=2.0,
        effective_ultimate_interaction_radius=9.0,
    )

    _, observation, _, _, action_mask, _ = step(
        config,
        state,
        _joint_action_with_moves(),
        jax.random.key(42),
    )

    assert bool(observation.enemy_visibility_mask[0, 0])
    assert not bool(observation.enemy_targetability_mask[0, 0])
    assert not bool(action_mask.target[0, 1 + MAX_AGENTS_PER_TEAM])


def test_ultimate_interaction_radius_does_not_affect_basic_targetability() -> None:
    """Assert M4 basic targetability ignores ultimate interaction radius."""
    config = _deterministic_config()
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([3.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([7.0, 2.0], dtype=jnp.float32),
        agent_d_position=jnp.array([8.0, 5.0], dtype=jnp.float32),
        agent_b_active_flag=False,
        agent_d_active_flag=False,
        agent_a_class_id=MAGE_CLASS_ID,
        agent_c_class_id=MAGE_CLASS_ID,
        effective_observation_radius=10.0,
        effective_basic_interaction_radius=2.0,
        effective_ultimate_interaction_radius=10.0,
    )

    _, observation, _, _, action_mask, _ = step(
        config,
        state,
        _joint_action_with_moves(),
        jax.random.key(42),
    )

    assert bool(observation.enemy_visibility_mask[0, 0])
    assert not bool(observation.enemy_targetability_mask[0, 0])
    assert not bool(action_mask.target[0, 1 + MAX_AGENTS_PER_TEAM])


def test_inactive_and_dead_observers_have_no_target_choices() -> None:
    """Assert inactive/dead observers cannot select none or unit targets."""
    config = _deterministic_config()
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([3.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([2.0, 4.0], dtype=jnp.float32),
        agent_d_position=jnp.array([3.0, 4.0], dtype=jnp.float32),
        agent_a_alive_flag=False,
        agent_b_active_flag=False,
        agent_a_class_id=MAGE_CLASS_ID,
        agent_c_class_id=MAGE_CLASS_ID,
        agent_d_class_id=PRIEST_CLASS_ID,
        effective_observation_radius=8.0,
        effective_basic_interaction_radius=5.0,
    )

    _, observation, _, _, action_mask, _ = step(
        config,
        state,
        _joint_action_with_moves(),
        jax.random.key(42),
    )

    _assert_action_mask_row_is_false(action_mask.target[0])
    _assert_action_mask_row_is_false(action_mask.target[1])

    assert not bool(jnp.any(observation.ally_targetability_mask[0]))
    assert not bool(jnp.any(observation.enemy_targetability_mask[0]))
    assert not bool(jnp.any(observation.ally_targetability_mask[1]))
    assert not bool(jnp.any(observation.enemy_targetability_mask[1]))


def test_none_target_selection_is_valid_only_for_active_alive_observers() -> None:
    """Assert target column zero is valid exactly for active alive observers."""
    config = _deterministic_config()
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([3.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([2.0, 4.0], dtype=jnp.float32),
        agent_d_position=jnp.array([3.0, 4.0], dtype=jnp.float32),
        agent_a_alive_flag=False,
        agent_b_active_flag=False,
        effective_observation_radius=8.0,
        effective_basic_interaction_radius=5.0,
    )

    next_state, _, _, _, action_mask, _ = step(
        config,
        state,
        _joint_action_with_moves(),
        jax.random.key(42),
    )

    expected_none_column = jnp.logical_and(
        config.agent_profile.active_mask,
        next_state.alive_mask,
    )

    assert bool(jnp.array_equal(action_mask.target[:, 0], expected_none_column))


def test_ultimate_use_remains_unavailable_in_milestone_5_step_3() -> None:
    """Assert Step 3 does not introduce ultimate availability."""
    config = _deterministic_config()
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([3.0, 2.0], dtype=jnp.float32),
        agent_c_position=jnp.array([2.0, 4.0], dtype=jnp.float32),
        agent_d_position=jnp.array([3.0, 4.0], dtype=jnp.float32),
        agent_a_class_id=MAGE_CLASS_ID,
        agent_b_class_id=PRIEST_CLASS_ID,
        agent_c_class_id=MAGE_CLASS_ID,
        agent_d_class_id=PRIEST_CLASS_ID,
        effective_observation_radius=8.0,
        effective_basic_interaction_radius=5.0,
        effective_ultimate_interaction_radius=10.0,
    )

    next_state, _, _, _, action_mask, _ = step(
        config,
        state,
        _joint_action_with_moves(),
        jax.random.key(42),
    )

    active_alive = jnp.logical_and(
        config.agent_profile.active_mask, next_state.alive_mask
    )

    assert bool(jnp.array_equal(action_mask.use_ultimate[:, 0], active_alive))
    assert not bool(jnp.any(action_mask.use_ultimate[:, 1]))


def test_jitted_nonstay_step_preserves_observation_mask_contracts() -> None:
    """Assert compiled non-stay step matches eager observation-mask behavior."""
    config = _deterministic_config(
        obstacles=_obstacle_array_with_rows(
            (
                0,
                _pillar_obstacle(
                    x=5.0,
                    y=2.0,
                    radius=0.75,
                    active=True,
                ),
            )
        )
    )
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([2.0, 5.0], dtype=jnp.float32),
        agent_c_position=jnp.array([7.0, 2.0], dtype=jnp.float32),
        agent_d_position=jnp.array([7.0, 5.0], dtype=jnp.float32),
        agent_a_class_id=MAGE_CLASS_ID,
        agent_b_class_id=MAGE_CLASS_ID,
        agent_c_class_id=MAGE_CLASS_ID,
        agent_d_class_id=MAGE_CLASS_ID,
        effective_observation_radius=10.0,
        effective_basic_interaction_radius=6.0,
    )
    joint_action = _joint_action_with_moves(
        (0, MOVE_EAST),
        (MAX_AGENTS_PER_TEAM, MOVE_EAST),
    )
    key = jax.random.key(42)

    eager_state, eager_observation, _, _, eager_action_mask, _ = step(
        config,
        state,
        joint_action,
        key,
    )
    compiled_state, compiled_observation, _, _, compiled_action_mask, _ = cast(
        tuple[EnvState, Observation, object, object, ActionMask, object],
        jax.jit(step)(
            config,
            state,
            joint_action,
            key,
        ),
    )

    assert bool(
        jnp.allclose(
            compiled_state.agent_positions,
            eager_state.agent_positions,
            atol=0.0,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.array_equal(
            compiled_observation.ally_visibility_mask,
            eager_observation.ally_visibility_mask,
        )
    )
    assert bool(
        jnp.array_equal(
            compiled_observation.enemy_visibility_mask,
            eager_observation.enemy_visibility_mask,
        )
    )
    assert bool(
        jnp.array_equal(
            compiled_observation.ally_targetability_mask,
            eager_observation.ally_targetability_mask,
        )
    )
    assert bool(
        jnp.array_equal(
            compiled_observation.enemy_targetability_mask,
            eager_observation.enemy_targetability_mask,
        )
    )
    assert bool(jnp.array_equal(compiled_action_mask.target, eager_action_mask.target))
    assert bool(jnp.any(compiled_observation.enemy_targetability_mask))

    _assert_targetability_never_exceeds_visibility(compiled_observation)
    _assert_flat_target_mask_matches_relation_targetability(
        action_mask=compiled_action_mask,
        state=compiled_state,
        config=config,
        expected_ally=compiled_observation.ally_targetability_mask,
        expected_enemy=compiled_observation.enemy_targetability_mask,
    )


def test_scanned_rollout_emits_stable_observation_mask_history() -> None:
    """Assert scan keeps observation and action-mask structures stable."""
    horizon = 4
    config = _deterministic_config(max_steps=1000)
    config, state = _state_two_versus_two_game(
        config,
        agent_a_position=jnp.array([2.0, 2.0], dtype=jnp.float32),
        agent_b_position=jnp.array([2.0, 5.0], dtype=jnp.float32),
        agent_c_position=jnp.array([7.0, 2.0], dtype=jnp.float32),
        agent_d_position=jnp.array([7.0, 5.0], dtype=jnp.float32),
        agent_a_class_id=MAGE_CLASS_ID,
        agent_b_class_id=PRIEST_CLASS_ID,
        agent_c_class_id=MAGE_CLASS_ID,
        agent_d_class_id=PRIEST_CLASS_ID,
        effective_observation_radius=10.0,
        effective_basic_interaction_radius=6.0,
    )
    joint_action = _joint_action_with_moves(
        (0, MOVE_EAST),
        (1, MOVE_EAST),
    )
    key = jax.random.key(123)
    keys = jax.random.split(key, horizon)

    def _rollout_step(
        carry: EnvState,
        step_key: Array,
    ) -> tuple[EnvState, tuple[Array, Array, Array, Array, Array, Array]]:
        next_state, observation, _, _, action_mask, _ = step(
            config,
            carry,
            joint_action,
            step_key,
        )
        return next_state, (
            observation.ally_visibility_mask,
            observation.enemy_visibility_mask,
            observation.ally_targetability_mask,
            observation.enemy_targetability_mask,
            action_mask.target,
            action_mask.use_ultimate,
        )

    def _rollout(
        initial_state: EnvState,
        step_keys: Array,
    ) -> tuple[EnvState, tuple[Array, Array, Array, Array, Array, Array]]:
        """Run a compiled fixed-horizon rollout with mask history."""
        return jax.lax.scan(
            _rollout_step,
            initial_state,
            step_keys,
            length=horizon,
        )

    final_state, history = cast(
        tuple[EnvState, tuple[Array, Array, Array, Array, Array, Array]],
        jax.jit(_rollout)(state, keys),
    )
    (
        ally_visibility_history,
        enemy_visibility_history,
        ally_targetability_history,
        enemy_targetability_history,
        target_mask_history,
        ultimate_mask_history,
    ) = history

    assert final_state.step_count == horizon
    assert ally_visibility_history.shape == (
        horizon,
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
    )
    assert enemy_visibility_history.shape == (
        horizon,
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
    )
    assert ally_targetability_history.shape == ally_visibility_history.shape
    assert enemy_targetability_history.shape == enemy_visibility_history.shape
    assert target_mask_history.shape == (horizon, MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)
    assert ultimate_mask_history.shape == (
        horizon,
        MAX_AGENT_SLOTS,
        NUM_ULTIMATE_ACTIONS,
    )

    assert ally_visibility_history.dtype == bool
    assert enemy_visibility_history.dtype == bool
    assert ally_targetability_history.dtype == bool
    assert enemy_targetability_history.dtype == bool
    assert target_mask_history.dtype == bool
    assert ultimate_mask_history.dtype == bool

    assert not bool(
        jnp.any(
            jnp.logical_and(
                ally_targetability_history,
                jnp.logical_not(ally_visibility_history),
            )
        )
    )
    assert not bool(
        jnp.any(
            jnp.logical_and(
                enemy_targetability_history,
                jnp.logical_not(enemy_visibility_history),
            )
        )
    )
    assert bool(jnp.any(ally_targetability_history))
    assert bool(jnp.any(enemy_targetability_history))

    expected_target_history = jnp.concatenate(
        (
            jnp.ones((horizon, MAX_AGENT_SLOTS, 1), dtype=bool),
            ally_targetability_history,
            enemy_targetability_history,
        ),
        axis=2,
    )
    active_alive = jnp.logical_and(
        config.agent_profile.active_mask, final_state.alive_mask
    )
    expected_target_history = jnp.logical_and(
        expected_target_history,
        active_alive[None, :, None],
    )

    assert bool(jnp.array_equal(target_mask_history, expected_target_history))
    assert not bool(jnp.any(ultimate_mask_history[:, :, 1]))
