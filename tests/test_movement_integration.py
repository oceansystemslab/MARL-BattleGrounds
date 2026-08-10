"""Movement integration tests across the simulator's movement contracts."""
# pyright: reportPrivateUsage=false

from typing import TypedDict, cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import _build_observation_and_action_mask, step
from marl_battlegrounds.core.geometry import GEOMETRY_TOLERANCE
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    CLASS_NEUTRAL,
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
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
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
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    SELF_FEATURES,
    STUN_CHANNEL_HUNTER_TRAP,
    STUN_CHANNEL_ROGUE_POISON,
    STUN_CHANNEL_WARRIOR_CHARGE,
    UNIT_FEATURES,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)

# Test Helpers ---


class _CombatStateFields(TypedDict):
    """Keyword fields for inert combat and action-history test state."""

    current_health: Array
    ultimate_cooldowns: Array
    slow_durations: Array
    stun_durations: Array
    rogue_poison_anti_heal_durations: Array
    mage_burst_damage_amplification_durations: Array
    priest_blessing_of_freedom_slow_floor_durations: Array
    steps_until_out_of_combat: Array
    previous_timestep_move_actions: Array
    previous_timestep_select_target_actions: Array
    previous_timestep_use_ultimate_actions: Array
    has_previous_timestep_joint_action: Array


def _inert_combat_state_fields(living_mask: Array) -> _CombatStateFields:
    """Return neutral fields with coherent positive health for living slots."""
    return {
        "current_health": living_mask.astype(jnp.float32),
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
        "steps_until_out_of_combat": jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        "previous_timestep_move_actions": jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        "previous_timestep_select_target_actions": jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        "previous_timestep_use_ultimate_actions": jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        "has_previous_timestep_joint_action": jnp.asarray(False),
    }


def _current_action_mask(config: EnvConfig, state: EnvState) -> ActionMask:
    """Return the action mask paired with an explicitly built test state."""
    _, action_mask = _build_observation_and_action_mask(state, config)
    return action_mask


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


def _empty_obstacles() -> Array:
    """Create a padded all-inactive obstacle table."""
    return jnp.zeros(
        (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
        dtype=jnp.float32,
    )


def _empty_obstacle() -> Array:
    """Create an inactive padding obstacle row."""
    return jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)


def _pillar_obstacle(
    pillar_center: Array,
    pillar_radius: Array | float,
    *,
    active: bool = True,
) -> Array:
    """Create a pillar obstacle row."""
    pillar = _empty_obstacle()

    pillar = pillar.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    pillar = pillar.at[OBSTACLE_FEATURE_ACTIVE].set(float(active))

    x_coordinate, y_coordinate = pillar_center
    pillar = pillar.at[OBSTACLE_FEATURE_X].set(x_coordinate)
    pillar = pillar.at[OBSTACLE_FEATURE_Y].set(y_coordinate)
    pillar = pillar.at[OBSTACLE_FEATURE_RADIUS].set(pillar_radius)

    return pillar


def _wall_obstacle(
    wall_center: Array,
    width: Array | float,
    height: Array | float,
    theta: Array | float = 0.0,
    *,
    active: bool = True,
) -> Array:
    """Create a wall obstacle row parameterized by center, size, and rotation."""
    wall = _empty_obstacle()

    wall = wall.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_WALL)
    wall = wall.at[OBSTACLE_FEATURE_ACTIVE].set(float(active))

    x_coordinate, y_coordinate = wall_center
    wall = wall.at[OBSTACLE_FEATURE_X].set(x_coordinate)
    wall = wall.at[OBSTACLE_FEATURE_Y].set(y_coordinate)
    wall = wall.at[OBSTACLE_FEATURE_WIDTH].set(width)
    wall = wall.at[OBSTACLE_FEATURE_HEIGHT].set(height)
    wall = wall.at[OBSTACLE_FEATURE_THETA].set(theta)

    return wall


def _deterministic_config(
    *,
    team_size: int = 3,
    max_steps: int = 1000,
    map_width: float = 20.0,
    map_height: float = 12.0,
    obstacles: Array | None = None,
    ordinary_movement_distance_scale: float = 1.0,
    spawn_shield_duration_steps: int = 3,
    spawn_shield_movement_speed: float = 2.0,
) -> EnvConfig:
    """Create a deterministic config for movement integration tests."""
    profile = resolve_agent_profile(
        jnp.full((MAX_AGENT_SLOTS,), CLASS_NEUTRAL, dtype=jnp.int32),
        jnp.asarray((team_size, team_size), dtype=jnp.int32),
    )
    positions = jnp.asarray(
        (
            (0.5, 0.5),
            (2.5, 0.5),
            (4.5, 0.5),
            (6.5, 0.5),
            (8.5, 0.5),
            (0.5, 7.5),
            (2.5, 7.5),
            (4.5, 7.5),
            (6.5, 7.5),
            (8.5, 7.5),
        ),
        dtype=jnp.float32,
    )
    return EnvConfig(
        max_steps=max_steps,
        map_width=map_width,
        map_height=map_height,
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
        agent_profile=profile,
        ordinary_movement_distance_scale=ordinary_movement_distance_scale,
        team_spawn_pad_positions=positions.reshape(
            (NUM_TEAMS, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS)
        ),
        spawn_shield_duration_steps=spawn_shield_duration_steps,
        spawn_shield_movement_speed=spawn_shield_movement_speed,
        team_respawn_wave_period_step_count=jnp.asarray((5, 5), dtype=jnp.int32),
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


def _slot_int_vector(
    default_value: int,
    *rows: tuple[int, Array | int],
) -> Array:
    """Create an int32 slot vector with selected overrides."""
    values = jnp.full((MAX_AGENT_SLOTS,), default_value, dtype=jnp.int32)

    for slot, value in rows:
        assert 0 <= slot < MAX_AGENT_SLOTS
        values = values.at[slot].set(value)

    return values


def _mask_with_true_slots(*slots: int) -> Array:
    """Create a slot mask with only selected slots marked true."""
    mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)

    for slot in slots:
        assert 0 <= slot < MAX_AGENT_SLOTS
        mask = mask.at[slot].set(True)

    return mask


def _state_with_single_active_alive_agent(
    config: EnvConfig,
    position: Array,
    *,
    radius: float = 0.5,
    effective_movement_speed: float = 1.0,
    effective_observation_radius: float = 8.0,
    effective_basic_interaction_radius: float = 6.0,
    effective_ultimate_interaction_radius: float = 9.0,
    spawn_shield_duration: int = 0,
    step_count: int = 0,
) -> tuple[EnvConfig, EnvState]:
    """Create an exact config/state pair with only slot 0 participating."""
    active_mask = _mask_with_true_slots(0)
    profile = config.agent_profile._replace(
        active_mask=active_mask,
        agent_radii=_agent_radii_array_with_rows((0, radius)),
        base_movement_speeds=_slot_float_vector(0.0, (0, effective_movement_speed)),
        observation_radii=_slot_float_vector(0.0, (0, effective_observation_radius)),
        basic_interaction_radii=_slot_float_vector(
            0.0, (0, effective_basic_interaction_radius)
        ),
        ultimate_interaction_radii=_slot_float_vector(
            0.0, (0, effective_ultimate_interaction_radius)
        ),
        max_health=active_mask.astype(jnp.float32),
    )
    config = config._replace(agent_profile=profile)
    state = EnvState(
        step_count=jnp.array(step_count, dtype=jnp.int32),
        agent_positions=_agent_positions_array_with_rows((0, position)),
        alive_mask=active_mask,
        team_respawn_wave_countdowns=config.team_respawn_wave_period_step_count - 1,
        spawn_shield_durations=_slot_int_vector(
            0,
            (0, spawn_shield_duration),
        ),
        **_inert_combat_state_fields(active_mask),
    )
    return config, state


def _state_with_two_agents(
    config: EnvConfig,
    agent_a_position: Array,
    agent_b_position: Array,
    agent_a_active_flag: bool = True,
    agent_a_alive_flag: bool = True,
    agent_b_active_flag: bool = True,
    agent_b_alive_flag: bool = True,
    *,
    agent_b_slot: int = 1,
    radius: float = 0.5,
    agent_a_effective_movement_speed: float = 1.0,
    agent_b_effective_movement_speed: float = 1.0,
    effective_observation_radius: float = 8.0,
    effective_basic_interaction_radius: float = 6.0,
    effective_ultimate_interaction_radius: float = 9.0,
    agent_a_spawn_shield_duration: int = 0,
    agent_b_spawn_shield_duration: int = 0,
    step_count: int = 0,
) -> tuple[EnvConfig, EnvState]:
    """Create an exact low-level pair for movement-kernel boundary tests.

    The helper intentionally permits catalog overrides and noncanonical payload
    in inactive rows so masking is tested rather than assumed. Such cases are
    adversarial kernel inputs, not official host-validated state evidence.
    """
    assert 0 < agent_b_slot < MAX_AGENT_SLOTS

    active_mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)
    alive_mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)

    active_mask = active_mask.at[0].set(agent_a_active_flag)
    alive_mask = alive_mask.at[0].set(agent_a_alive_flag)

    active_mask = active_mask.at[agent_b_slot].set(agent_b_active_flag)
    alive_mask = alive_mask.at[agent_b_slot].set(agent_b_alive_flag)

    profile = config.agent_profile._replace(
        active_mask=active_mask,
        agent_radii=_agent_radii_array_with_rows(
            (0, radius),
            (agent_b_slot, radius),
        ),
        base_movement_speeds=_slot_float_vector(
            0.0,
            (0, agent_a_effective_movement_speed),
            (agent_b_slot, agent_b_effective_movement_speed),
        ),
        observation_radii=_slot_float_vector(
            0.0,
            (0, effective_observation_radius),
            (agent_b_slot, effective_observation_radius),
        ),
        basic_interaction_radii=_slot_float_vector(
            0.0,
            (0, effective_basic_interaction_radius),
            (agent_b_slot, effective_basic_interaction_radius),
        ),
        ultimate_interaction_radii=_slot_float_vector(
            0.0,
            (0, effective_ultimate_interaction_radius),
            (agent_b_slot, effective_ultimate_interaction_radius),
        ),
        max_health=active_mask.astype(jnp.float32),
    )
    config = config._replace(agent_profile=profile)
    state = EnvState(
        step_count=jnp.array(step_count, dtype=jnp.int32),
        agent_positions=_agent_positions_array_with_rows(
            (0, agent_a_position),
            (agent_b_slot, agent_b_position),
        ),
        alive_mask=alive_mask,
        team_respawn_wave_countdowns=config.team_respawn_wave_period_step_count - 1,
        spawn_shield_durations=_slot_int_vector(
            0,
            (0, agent_a_spawn_shield_duration),
            (agent_b_slot, agent_b_spawn_shield_duration),
        ),
        **_inert_combat_state_fields(alive_mask),
    )
    return config, state


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
        select_target=joint_action_targets,
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


def _assert_agents_do_not_overlap(
    agent_positions: Array,
    *,
    slot_a: int,
    slot_b: int,
    radius_a: float,
    radius_b: float,
) -> None:
    """Assert that two agent discs are separated."""
    center_a = agent_positions[slot_a]
    center_b = agent_positions[slot_b]

    distance = cast(Array, jnp.linalg.norm(center_a - center_b))
    minimum_valid_distance = radius_a + radius_b

    assert bool(distance >= minimum_valid_distance - GEOMETRY_TOLERANCE)


def _assert_agent_positions_are_finite(agent_positions: Array) -> None:
    """Assert that all slot-aligned positions are finite."""
    assert bool(jnp.all(jnp.isfinite(agent_positions)))


def _assert_position_history_contract(position_history: Array, *, horizon: int) -> None:
    """Assert the scan-emitted position history shape and dtype."""
    assert position_history.shape == (
        horizon,
        MAX_AGENT_SLOTS,
        ENVIRONMENT_DIMENSIONS,
    )
    assert position_history.dtype == jnp.float32


def _assert_state_contract(state: EnvState) -> None:
    """Assert the EnvState shape and dtype contract."""
    assert state.step_count.shape == ()
    assert state.step_count.dtype == jnp.int32

    assert state.agent_positions.shape == (
        MAX_AGENT_SLOTS,
        ENVIRONMENT_DIMENSIONS,
    )
    assert state.agent_positions.dtype == jnp.float32

    assert state.alive_mask.shape == (MAX_AGENT_SLOTS,)
    assert state.alive_mask.dtype == bool

    assert state.spawn_shield_durations.shape == (MAX_AGENT_SLOTS,)
    assert state.spawn_shield_durations.dtype == jnp.int32


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

    assert "ally_targetability_mask" not in Observation._fields
    assert "enemy_targetability_mask" not in Observation._fields


def _assert_action_mask_contract(action_mask: ActionMask) -> None:
    """Assert the ActionMask shape and dtype contract."""
    assert action_mask.move_mask.shape == (MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS)
    assert action_mask.move_mask.dtype == bool

    assert action_mask.select_target_mask.shape == (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)
    assert action_mask.select_target_mask.dtype == bool

    assert action_mask.use_ultimate_mask.shape == (
        MAX_AGENT_SLOTS,
        NUM_ULTIMATE_ACTIONS,
    )
    assert action_mask.use_ultimate_mask.dtype == bool

    assert action_mask.select_target_use_ultimate_joint_mask.shape == (
        MAX_AGENT_SLOTS,
        NUM_TARGET_ACTIONS,
        NUM_ULTIMATE_ACTIONS,
    )
    assert action_mask.select_target_use_ultimate_joint_mask.dtype == bool

    assert jnp.array_equal(
        action_mask.select_target_mask,
        jnp.any(action_mask.select_target_use_ultimate_joint_mask, axis=-1),
    )
    assert jnp.array_equal(
        action_mask.use_ultimate_mask,
        jnp.any(action_mask.select_target_use_ultimate_joint_mask, axis=1),
    )
    assert bool(jnp.all(jnp.any(action_mask.move_mask, axis=-1)))
    assert bool(jnp.all(jnp.any(action_mask.select_target_mask, axis=-1)))
    assert bool(jnp.all(jnp.any(action_mask.use_ultimate_mask, axis=-1)))
    assert bool(
        jnp.all(
            jnp.any(
                action_mask.select_target_use_ultimate_joint_mask,
                axis=(-2, -1),
            )
        )
    )


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


def _assert_single_agent_action_mask_semantics(action_mask: ActionMask) -> None:
    """Assert one living actor plus canonical no-op rows for padded slots."""
    assert bool(jnp.all(action_mask.move_mask[0]))
    assert bool(jnp.all(action_mask.move_mask[1:, MOVE_STAY]))
    assert bool(jnp.all(jnp.sum(action_mask.move_mask[1:], axis=-1) == 1))

    assert bool(action_mask.select_target_mask[0, 0])
    assert bool(jnp.all(~action_mask.select_target_mask[0, 1:]))
    assert bool(jnp.all(action_mask.select_target_mask[1:, 0]))
    assert bool(jnp.all(jnp.sum(action_mask.select_target_mask[1:], axis=-1) == 1))

    assert bool(action_mask.use_ultimate_mask[0, 0])
    assert bool(~action_mask.use_ultimate_mask[0, 1])
    assert bool(jnp.all(action_mask.use_ultimate_mask[1:, 0]))
    assert bool(jnp.all(jnp.sum(action_mask.use_ultimate_mask[1:], axis=-1) == 1))

    joint_mask = action_mask.select_target_use_ultimate_joint_mask
    assert bool(jnp.all(joint_mask[1:, 0, 0]))
    assert bool(jnp.all(jnp.sum(joint_mask[1:], axis=(-2, -1)) == 1))


def _assert_center_inside_bounds(
    center: Array,
    *,
    radius: float,
    config: EnvConfig,
) -> None:
    """Assert that an agent center satisfies the map-boundary invariant."""
    assert float(center[0]) >= radius - GEOMETRY_TOLERANCE
    assert float(center[0]) <= config.map_width - radius + GEOMETRY_TOLERANCE
    assert float(center[1]) >= radius - GEOMETRY_TOLERANCE
    assert float(center[1]) <= config.map_height - radius + GEOMETRY_TOLERANCE


def _assert_center_outside_pillar(
    center: Array,
    *,
    agent_radius: float,
    pillar_center: Array,
    pillar_radius: float,
) -> None:
    """Assert that an agent disc does not overlap a circular pillar."""
    distance = cast(Array, jnp.linalg.norm(center - pillar_center))
    minimum_valid_distance = agent_radius + pillar_radius

    assert bool(distance >= minimum_valid_distance - GEOMETRY_TOLERANCE)


def _assert_center_outside_axis_aligned_wall(
    center: Array,
    *,
    agent_radius: float,
    wall_center: Array,
    wall_width: float,
    wall_height: float,
) -> None:
    """Assert that an agent disc does not overlap an axis-aligned wall."""
    half_width = wall_width / 2.0
    half_height = wall_height / 2.0

    dx = jnp.abs(center[0] - wall_center[0])
    dy = jnp.abs(center[1] - wall_center[1])

    outside_x = dx >= half_width + agent_radius - GEOMETRY_TOLERANCE
    outside_y = dy >= half_height + agent_radius - GEOMETRY_TOLERANCE

    assert bool(jnp.logical_or(outside_x, outside_y))


def _assert_center_outside_rotated_wall(
    center: Array,
    *,
    agent_radius: float,
    wall_center: Array,
    wall_width: float,
    wall_height: float,
    theta: float,
) -> None:
    """Assert that an agent disc does not overlap a rotated rectangular wall."""
    relative = center - wall_center

    cos_theta = jnp.cos(-theta)
    sin_theta = jnp.sin(-theta)

    local_x = cos_theta * relative[0] - sin_theta * relative[1]
    local_y = sin_theta * relative[0] + cos_theta * relative[1]

    half_width = wall_width / 2.0
    half_height = wall_height / 2.0

    outside_x = jnp.abs(local_x) >= half_width + agent_radius - GEOMETRY_TOLERANCE
    outside_y = jnp.abs(local_y) >= half_height + agent_radius - GEOMETRY_TOLERANCE

    assert bool(jnp.logical_or(outside_x, outside_y))


# Tests ---


def test_move_stay_preserves_valid_position_in_free_space() -> None:
    config = _deterministic_config()
    key = jax.random.key(42)
    start = jnp.array([10.0, 10.0], dtype=jnp.float32)
    config, state = _state_with_single_active_alive_agent(config, start)
    joint_action = _joint_action_with_moves((0, MOVE_STAY))

    next_state, _, _, _, _, info = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_close(next_state.agent_positions[0], start)
    physical_facts = info.transition_facts.physical_facts
    assert bool(jnp.all(physical_facts.charge_phase_displacement_by_agent == 0.0))
    assert bool(
        jnp.all(physical_facts.ordinary_movement_phase_displacement_by_agent == 0.0)
    )


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
    config, state = _state_with_single_active_alive_agent(
        config, jnp.array([10.0, 10.0], dtype=jnp.float32)
    )
    joint_action = _joint_action_with_moves((0, move_action))

    next_state, obs, reward, done_flags, action_mask, info = step(
        config,
        state,
        _current_action_mask(config, state),
        joint_action,
        key,
    )

    _assert_center_close(next_state.agent_positions[0], expected_position)
    _assert_state_contract(next_state)
    _assert_observation_contract(obs)
    _assert_reward_contract(reward)
    _assert_done_flags_contract(done_flags, expected_truncated=False)
    _assert_action_mask_contract(action_mask)
    _assert_single_agent_action_mask_semantics(action_mask)
    assert isinstance(info, Info)
    physical_facts = info.transition_facts.physical_facts
    assert bool(jnp.all(physical_facts.charge_phase_displacement_by_agent == 0.0))
    assert bool(
        jnp.allclose(
            physical_facts.ordinary_movement_phase_displacement_by_agent,
            next_state.agent_positions - state.agent_positions,
        )
    )


@pytest.mark.parametrize(
    ("move_action", "expected_position"),
    [
        pytest.param(MOVE_NORTH, jnp.array([10.0, 8.5], dtype=jnp.float32), id="north"),
        pytest.param(MOVE_EAST, jnp.array([12.5, 6.0], dtype=jnp.float32), id="east"),
    ],
)
def test_cardinal_moves_scale_by_effective_state_movement_speed(
    move_action: int,
    expected_position: Array,
) -> None:
    config = _deterministic_config()
    key = jax.random.key(42)
    config, state = _state_with_single_active_alive_agent(
        config,
        jnp.array([10.0, 6.0], dtype=jnp.float32),
        effective_movement_speed=2.5,
    )
    joint_action = _joint_action_with_moves((0, move_action))

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_close(next_state.agent_positions[0], expected_position)


@pytest.mark.parametrize(
    ("move_action", "expected_direction"),
    (
        pytest.param(MOVE_NORTH, (0.0, 1.0), id="north"),
        pytest.param(MOVE_SOUTH, (0.0, -1.0), id="south"),
        pytest.param(MOVE_EAST, (1.0, 0.0), id="east"),
        pytest.param(MOVE_WEST, (-1.0, 0.0), id="west"),
        pytest.param(MOVE_NORTHEAST, (1.0, 1.0), id="northeast"),
        pytest.param(MOVE_NORTHWEST, (-1.0, 1.0), id="northwest"),
        pytest.param(MOVE_SOUTHEAST, (1.0, -1.0), id="southeast"),
        pytest.param(MOVE_SOUTHWEST, (-1.0, -1.0), id="southwest"),
    ),
)
def test_movement_calibration_scales_every_nonstay_direction(
    move_action: int,
    expected_direction: tuple[float, float],
) -> None:
    """Prove calibrated displacement preserves direction and unit normalization."""
    movement_scale = 0.1
    config = _deterministic_config(ordinary_movement_distance_scale=movement_scale)
    start = jnp.asarray((10.0, 6.0), dtype=jnp.float32)
    config, state = _state_with_single_active_alive_agent(config, start)

    next_state, observation, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        _joint_action_with_moves((0, move_action)),
        jax.random.key(60),
    )

    direction = jnp.asarray(expected_direction, dtype=jnp.float32)
    normalized_direction = direction / cast(Array, jnp.linalg.norm(direction))
    displacement = next_state.agent_positions[0] - start
    expected_displacement = normalized_direction * movement_scale

    assert bool(jnp.allclose(displacement, expected_displacement, atol=1e-6))
    assert bool(
        jnp.isclose(
            cast(Array, jnp.linalg.norm(displacement)),
            movement_scale,
            atol=2e-6,
        )
    )
    assert (
        observation.self_features[0, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
        == movement_scale
    )


def test_movement_calibration_matches_representative_lane_budget_under_scan() -> None:
    """Prove exact float32 engagement and seven-unit traversal cadence."""
    movement_scale = 0.1
    horizon = 70
    config = _deterministic_config(ordinary_movement_distance_scale=movement_scale)
    start = jnp.asarray((0.5, 6.0), dtype=jnp.float32)
    stationary_target_x = jnp.float32(7.5)
    config, initial_state = _state_with_single_active_alive_agent(config, start)
    initial_mask = _current_action_mask(config, initial_state)
    action = _joint_action_with_moves((0, MOVE_EAST))
    keys = jax.random.split(jax.random.key(61), horizon)

    def _rollout(
        state: EnvState,
        action_mask: ActionMask,
        rollout_keys: Array,
    ) -> tuple[Array, Array]:
        def _scan_step(
            carry: tuple[EnvState, ActionMask],
            key: Array,
        ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array]]:
            current_state, current_mask = carry
            next_state, observation, _, _, next_mask, _ = step(
                config,
                current_state,
                current_mask,
                action,
                key,
            )
            return (
                (next_state, next_mask),
                (
                    next_state.agent_positions[0],
                    observation.self_features[
                        0, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED
                    ],
                ),
            )

        _, history = jax.lax.scan(
            _scan_step,
            (state, action_mask),
            rollout_keys,
        )
        return history

    position_history, speed_history = _rollout(initial_state, initial_mask, keys)
    compiled_positions, compiled_speeds = cast(
        tuple[Array, Array],
        jax.jit(_rollout)(initial_state, initial_mask, keys),
    )
    distance_history = stationary_target_x - position_history[:, 0]

    # Four float32 geometry substeps leave the nominal 10- and 15-step
    # boundaries just outside range; the following decision crosses them.
    assert bool(distance_history[9] > 6.0)
    assert bool(distance_history[10] <= 6.0)
    assert bool(distance_history[14] > 5.5)
    assert bool(distance_history[15] <= 5.5)
    assert bool(distance_history[18] > 5.0)
    assert bool(distance_history[19] <= 5.0)
    assert bool(
        jnp.isclose(
            position_history[-1, 0] - start[0],
            7.0,
            atol=1e-5,
        )
    )
    assert bool(jnp.all(speed_history == movement_scale))
    assert bool(jnp.array_equal(position_history, compiled_positions))
    assert bool(jnp.array_equal(speed_history, compiled_speeds))


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
    config, state = _state_with_single_active_alive_agent(config, start)
    joint_action = _joint_action_with_moves((0, move_action))

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    displacement = next_state.agent_positions[0] - start
    expected_position = start + expected_delta

    _assert_center_close(next_state.agent_positions[0], expected_position)
    assert bool(
        jnp.isclose(
            cast(Array, jnp.linalg.norm(displacement)),
            config.agent_profile.base_movement_speeds[0],
            atol=GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
    )


def test_diagonal_moves_scale_by_effective_state_movement_speed() -> None:
    effective_movement_speed = 2.0
    config = _deterministic_config()
    key = jax.random.key(42)
    start = jnp.array([10.0, 10.0], dtype=jnp.float32)
    config, state = _state_with_single_active_alive_agent(
        config, start, effective_movement_speed=effective_movement_speed
    )
    joint_action = _joint_action_with_moves((0, MOVE_NORTHEAST))

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    displacement = next_state.agent_positions[0] - start

    assert bool(
        jnp.isclose(
            cast(Array, jnp.linalg.norm(displacement)),
            jnp.asarray(effective_movement_speed, dtype=jnp.float32),
            atol=GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
    )
    assert bool(displacement[0] > 0.0)
    assert bool(displacement[1] > 0.0)


def test_same_move_action_uses_per_slot_movement_speeds() -> None:
    config = _deterministic_config()
    key = jax.random.key(42)
    agent_a_start = jnp.array([5.0, 5.0], dtype=jnp.float32)
    agent_b_start = jnp.array([12.0, 5.0], dtype=jnp.float32)
    config, state = _state_with_two_agents(
        config,
        agent_a_start,
        agent_b_start,
        agent_a_effective_movement_speed=1.0,
        agent_b_effective_movement_speed=2.5,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST), (1, MOVE_EAST))

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_close(
        next_state.agent_positions[0],
        jnp.array([6.0, 5.0], dtype=jnp.float32),
    )
    _assert_center_close(
        next_state.agent_positions[1],
        jnp.array([14.5, 5.0], dtype=jnp.float32),
    )


def test_step_uses_state_movement_speed_after_state_exists() -> None:
    config = _deterministic_config()
    key = jax.random.key(42)
    start = jnp.array([5.0, 5.0], dtype=jnp.float32)
    config, state = _state_with_single_active_alive_agent(
        config, start, effective_movement_speed=1.25
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_close(
        next_state.agent_positions[0],
        jnp.array([6.25, 5.0], dtype=jnp.float32),
    )


def test_step_preserves_current_contracts_after_non_stay_movement() -> None:
    config = _deterministic_config(max_steps=10)
    key = jax.random.key(42)
    config, state = _state_with_single_active_alive_agent(
        config, jnp.array([10.0, 10.0], dtype=jnp.float32)
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, obs, reward, done_flags, action_mask, info = step(
        config,
        state,
        _current_action_mask(config, state),
        joint_action,
        key,
    )

    assert next_state.step_count == state.step_count + 1
    assert jnp.array_equal(next_state.alive_mask, state.alive_mask)
    _assert_state_contract(next_state)
    _assert_observation_contract(obs)
    _assert_reward_contract(reward)
    _assert_done_flags_contract(done_flags, expected_truncated=False)
    _assert_action_mask_contract(action_mask)
    _assert_single_agent_action_mask_semantics(action_mask)
    assert isinstance(info, Info)


def test_step_truncates_after_incremented_step_count() -> None:
    config = _deterministic_config(max_steps=1)
    key = jax.random.key(42)
    config, state = _state_with_single_active_alive_agent(
        config,
        jnp.array([10.0, 10.0], dtype=jnp.float32),
        step_count=0,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    _, _, _, done_flags, _, _ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_done_flags_contract(done_flags, expected_truncated=True)


@pytest.mark.parametrize(
    ("start", "move_action"),
    [
        pytest.param(jnp.array([0.6, 6.0], dtype=jnp.float32), MOVE_WEST, id="west"),
        pytest.param(jnp.array([19.4, 6.0], dtype=jnp.float32), MOVE_EAST, id="east"),
        pytest.param(jnp.array([10.0, 0.6], dtype=jnp.float32), MOVE_SOUTH, id="south"),
        pytest.param(
            jnp.array([10.0, 11.4], dtype=jnp.float32), MOVE_NORTH, id="north"
        ),
    ],
)
def test_step_projects_active_alive_agent_inside_bounds(
    start: Array,
    move_action: int,
) -> None:
    config = _deterministic_config(map_width=20.0, map_height=12.0)
    key = jax.random.key(42)

    agent_radius = 0.5
    config, state = _state_with_single_active_alive_agent(
        config,
        start,
        radius=agent_radius,
    )
    joint_action = _joint_action_with_moves((0, move_action))

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_inside_bounds(
        next_state.agent_positions[0],
        radius=agent_radius,
        config=config,
    )


def test_step_projects_active_alive_agent_outside_active_pillar() -> None:
    pillar_center = jnp.array([11.0, 10.0], dtype=jnp.float32)
    pillar_radius = 0.5

    obstacles = _obstacle_array_with_rows(
        (0, _pillar_obstacle(pillar_center, pillar_radius, active=True))
    )
    config = _deterministic_config(obstacles=obstacles)
    key = jax.random.key(42)

    agent_radius = 0.5
    config, state = _state_with_single_active_alive_agent(
        config,
        jnp.array([10.0, 10.0], dtype=jnp.float32),
        radius=agent_radius,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, _, _, _, _, info = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_outside_pillar(
        next_state.agent_positions[0],
        agent_radius=agent_radius,
        pillar_center=pillar_center,
        pillar_radius=pillar_radius,
    )
    physical_facts = info.transition_facts.physical_facts
    assert bool(jnp.all(physical_facts.charge_phase_displacement_by_agent == 0.0))
    assert bool(
        jnp.allclose(
            physical_facts.ordinary_movement_phase_displacement_by_agent,
            next_state.agent_positions - state.agent_positions,
        )
    )


def test_step_projects_active_alive_agent_outside_active_wall() -> None:
    wall_center = jnp.array([11.0, 10.0], dtype=jnp.float32)
    wall_width = 1.0
    wall_height = 2.0

    obstacles = _obstacle_array_with_rows(
        (0, _wall_obstacle(wall_center, wall_width, wall_height, active=True))
    )
    config = _deterministic_config(obstacles=obstacles)
    key = jax.random.key(42)

    agent_radius = 0.5
    config, state = _state_with_single_active_alive_agent(
        config,
        jnp.array([10.0, 10.0], dtype=jnp.float32),
        radius=agent_radius,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_outside_axis_aligned_wall(
        next_state.agent_positions[0],
        agent_radius=agent_radius,
        wall_center=wall_center,
        wall_width=wall_width,
        wall_height=wall_height,
    )


def test_step_ignores_inactive_obstacle_rows() -> None:
    pillar_center = jnp.array([11.0, 10.0], dtype=jnp.float32)

    obstacles = _obstacle_array_with_rows(
        (0, _pillar_obstacle(pillar_center, 0.5, active=False))
    )
    config = _deterministic_config(obstacles=obstacles)
    key = jax.random.key(42)

    config, state = _state_with_single_active_alive_agent(
        config, jnp.array([10.0, 10.0], dtype=jnp.float32)
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_close(
        next_state.agent_positions[0],
        jnp.array([11.0, 10.0], dtype=jnp.float32),
    )


def test_step_ignores_inactive_wall_rows() -> None:
    wall_center = jnp.array([11.0, 10.0], dtype=jnp.float32)
    wall_width = 1.0
    wall_height = 2.0

    obstacles = _obstacle_array_with_rows(
        (0, _wall_obstacle(wall_center, wall_width, wall_height, active=False))
    )
    config = _deterministic_config(obstacles=obstacles)
    key = jax.random.key(42)

    config, state = _state_with_single_active_alive_agent(
        config, jnp.array([10.0, 10.0], dtype=jnp.float32)
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_close(
        next_state.agent_positions[0],
        jnp.array([11.0, 10.0], dtype=jnp.float32),
    )


def test_step_uses_active_obstacles_in_late_padded_slots() -> None:
    pillar_center = jnp.array([11.0, 10.0], dtype=jnp.float32)
    pillar_radius = 0.5
    late_slot = MAX_OBSTACLE_SLOTS - 1

    obstacles = _obstacle_array_with_rows(
        (late_slot, _pillar_obstacle(pillar_center, pillar_radius, active=True))
    )
    config = _deterministic_config(obstacles=obstacles)
    key = jax.random.key(42)

    agent_radius = 0.5
    config, state = _state_with_single_active_alive_agent(
        config,
        jnp.array([10.0, 10.0], dtype=jnp.float32),
        radius=agent_radius,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_outside_pillar(
        next_state.agent_positions[0],
        agent_radius=agent_radius,
        pillar_center=pillar_center,
        pillar_radius=pillar_radius,
    )


def test_step_projects_active_alive_agent_outside_rotated_wall() -> None:
    wall_center = jnp.array([11.0, 10.0], dtype=jnp.float32)
    wall_width = 1.0
    wall_height = 3.0
    theta = jnp.pi / 4.0

    obstacles = _obstacle_array_with_rows(
        (0, _wall_obstacle(wall_center, wall_width, wall_height, theta, active=True))
    )
    config = _deterministic_config(obstacles=obstacles)
    key = jax.random.key(42)

    agent_radius = 0.5
    config, state = _state_with_single_active_alive_agent(
        config,
        jnp.array([10.0, 10.0], dtype=jnp.float32),
        radius=agent_radius,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_outside_rotated_wall(
        next_state.agent_positions[0],
        agent_radius=agent_radius,
        wall_center=wall_center,
        wall_width=wall_width,
        wall_height=wall_height,
        theta=float(theta),
    )


def test_inactive_slots_with_nonstay_action_preserve_original_positions() -> None:
    """Prove malformed inactive payload remains physically inert at kernel level."""
    config = _deterministic_config()
    key = jax.random.key(42)

    agent_a_position = jnp.array([10.0, 10.0], dtype=jnp.float32)
    agent_b_position = jnp.array([12.0, 10.0], dtype=jnp.float32)

    config, state = _state_with_two_agents(
        config,
        agent_a_position,
        agent_b_position,
        agent_a_active_flag=False,
        agent_a_alive_flag=False,
        agent_b_active_flag=False,
        agent_b_alive_flag=False,
    )

    joint_action = _joint_action_with_moves(
        (0, MOVE_EAST),
        (1, MOVE_WEST),
    )

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_close(next_state.agent_positions[0], agent_a_position)
    _assert_center_close(next_state.agent_positions[1], agent_b_position)


def test_dead_slots_with_nonstay_action_preserve_original_positions() -> None:
    config = _deterministic_config()
    key = jax.random.key(42)

    agent_a_position = jnp.array([10.0, 10.0], dtype=jnp.float32)
    agent_b_position = jnp.array([12.0, 10.0], dtype=jnp.float32)

    config, state = _state_with_two_agents(
        config,
        agent_a_position,
        agent_b_position,
        agent_a_active_flag=True,
        agent_a_alive_flag=False,
        agent_b_active_flag=True,
        agent_b_alive_flag=False,
    )

    joint_action = _joint_action_with_moves(
        (0, MOVE_EAST),
        (1, MOVE_WEST),
    )

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_close(next_state.agent_positions[0], agent_a_position)
    _assert_center_close(next_state.agent_positions[1], agent_b_position)


@pytest.mark.parametrize(
    ("agent_b_active_flag", "agent_b_alive_flag"),
    [
        pytest.param(False, False, id="inactive_neighbor"),
        pytest.param(True, False, id="active_dead_neighbor"),
    ],
)
def test_active_alive_slot_moves_while_nonparticipant_neighbor_is_preserved(
    agent_b_active_flag: bool,
    agent_b_alive_flag: bool,
) -> None:
    config = _deterministic_config()
    key = jax.random.key(42)

    agent_a_position = jnp.array([10.0, 10.0], dtype=jnp.float32)
    agent_b_position = jnp.array([11.0, 10.0], dtype=jnp.float32)

    config, state = _state_with_two_agents(
        config,
        agent_a_position,
        agent_b_position,
        agent_a_active_flag=True,
        agent_a_alive_flag=True,
        agent_b_active_flag=agent_b_active_flag,
        agent_b_alive_flag=agent_b_alive_flag,
    )

    joint_action = _joint_action_with_moves(
        (0, MOVE_EAST),
        (1, MOVE_WEST),
    )

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_center_close(
        next_state.agent_positions[0],
        jnp.array([11.0, 10.0], dtype=jnp.float32),
    )
    _assert_center_close(next_state.agent_positions[1], agent_b_position)


@pytest.mark.parametrize(
    "blocker_slot",
    [
        pytest.param(1, id="same_team_blocker"),
        pytest.param(MAX_AGENTS_PER_TEAM, id="enemy_team_blocker"),
    ],
)
def test_active_alive_overlapping_agents_separate_in_free_space(
    blocker_slot: int,
) -> None:
    config = _deterministic_config()
    key = jax.random.key(42)

    radius = 0.5
    agent_a_position = jnp.array([10.0, 10.0], dtype=jnp.float32)
    blocker_position = jnp.array([11.25, 10.0], dtype=jnp.float32)
    active_mask = _mask_with_true_slots(0, blocker_slot)
    config = config._replace(
        agent_profile=config.agent_profile._replace(
            active_mask=active_mask,
            agent_radii=_agent_radii_array_with_rows(
                (0, radius),
                (blocker_slot, radius),
            ),
            base_movement_speeds=_slot_float_vector(
                0.0,
                (0, 1.0),
                (blocker_slot, 1.0),
            ),
            observation_radii=_slot_float_vector(0.0),
            basic_interaction_radii=_slot_float_vector(0.0),
            ultimate_interaction_radii=_slot_float_vector(0.0),
            max_health=active_mask.astype(jnp.float32),
        )
    )

    state = EnvState(
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=_agent_positions_array_with_rows(
            (0, agent_a_position),
            (blocker_slot, blocker_position),
        ),
        alive_mask=active_mask,
        team_respawn_wave_countdowns=config.team_respawn_wave_period_step_count - 1,
        spawn_shield_durations=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        **_inert_combat_state_fields(active_mask),
    )

    joint_action = _joint_action_with_moves(
        (0, MOVE_EAST),
        (blocker_slot, MOVE_STAY),
    )

    next_state, *_ = step(
        config, state, _current_action_mask(config, state), joint_action, key
    )

    _assert_agent_positions_are_finite(next_state.agent_positions)
    _assert_agents_do_not_overlap(
        next_state.agent_positions,
        slot_a=0,
        slot_b=blocker_slot,
        radius_a=radius,
        radius_b=radius,
    )


@pytest.mark.parametrize(
    ("configured_duration", "current_duration", "expected_next_duration"),
    (
        pytest.param(0, 0, 0, id="disabled"),
        pytest.param(1, 1, 0, id="expires"),
        pytest.param(3, 3, 2, id="official-duration"),
        pytest.param(7, 7, 6, id="larger-valid-duration"),
    ),
)
def test_spawn_shield_counter_decrements_once_without_underflow(
    configured_duration: int,
    current_duration: int,
    expected_next_duration: int,
) -> None:
    """Prove one current-state counter controls one transition of protection."""
    config = _deterministic_config(
        team_size=1,
        spawn_shield_duration_steps=configured_duration,
    )
    start = jnp.asarray((10.0, 6.0), dtype=jnp.float32)
    config, state = _state_with_single_active_alive_agent(
        config,
        start,
        spawn_shield_duration=current_duration,
    )

    next_state, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        _joint_action_with_moves((0, MOVE_STAY)),
        jax.random.key(70),
    )

    _assert_center_close(next_state.agent_positions[0], start)
    assert next_state.spawn_shield_durations.dtype == jnp.int32
    assert int(next_state.spawn_shield_durations[0]) == expected_next_duration
    assert bool(jnp.all(next_state.spawn_shield_durations[1:] == 0))


def test_spawn_shield_counter_clears_dead_and_inactive_rows() -> None:
    """Prove nonparticipating slots cannot retain malformed shield counters."""
    config = _deterministic_config(
        team_size=1,
        spawn_shield_duration_steps=3,
    )
    config, state = _state_with_two_agents(
        config,
        jnp.asarray((5.0, 5.0), dtype=jnp.float32),
        jnp.asarray((8.0, 5.0), dtype=jnp.float32),
        agent_a_active_flag=False,
        agent_a_alive_flag=False,
        agent_b_active_flag=True,
        agent_b_alive_flag=False,
        agent_b_slot=MAX_AGENTS_PER_TEAM,
        agent_a_spawn_shield_duration=3,
        agent_b_spawn_shield_duration=3,
    )

    next_state, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        _joint_action_with_moves(),
        jax.random.key(71),
    )

    assert bool(jnp.all(next_state.spawn_shield_durations == 0))


@pytest.mark.parametrize(
    ("move_action", "expected_direction"),
    (
        pytest.param(MOVE_NORTH, (0.0, 1.0), id="north"),
        pytest.param(MOVE_SOUTH, (0.0, -1.0), id="south"),
        pytest.param(MOVE_EAST, (1.0, 0.0), id="east"),
        pytest.param(MOVE_WEST, (-1.0, 0.0), id="west"),
        pytest.param(MOVE_NORTHEAST, (1.0, 1.0), id="northeast"),
        pytest.param(MOVE_NORTHWEST, (-1.0, 1.0), id="northwest"),
        pytest.param(MOVE_SOUTHEAST, (1.0, -1.0), id="southeast"),
        pytest.param(MOVE_SOUTHWEST, (-1.0, -1.0), id="southwest"),
    ),
)
@pytest.mark.parametrize(
    "spawn_shield_movement_speed",
    (
        pytest.param(2.0, id="official-speed"),
        pytest.param(1.375, id="nondefault-speed"),
    ),
)
def test_spawn_shield_uses_absolute_speed_for_every_nonstay_heading(
    move_action: int,
    expected_direction: tuple[float, float],
    spawn_shield_movement_speed: float,
) -> None:
    """Prove shield movement bypasses profile, status, and ordinary scaling."""
    config = _deterministic_config(
        team_size=1,
        ordinary_movement_distance_scale=0.125,
        spawn_shield_movement_speed=spawn_shield_movement_speed,
    )
    start = jnp.asarray((10.0, 6.0), dtype=jnp.float32)
    config, state = _state_with_single_active_alive_agent(
        config,
        start,
        effective_movement_speed=7.0,
        spawn_shield_duration=3,
    )
    state = state._replace(
        slow_durations=state.slow_durations.at[0].set(5),
    )
    observation, current_action_mask = _build_observation_and_action_mask(state, config)

    next_state, next_observation, *_ = step(
        config,
        state,
        current_action_mask,
        _joint_action_with_moves((0, move_action)),
        jax.random.key(72),
    )

    expected_heading = jnp.asarray(expected_direction, dtype=jnp.float32)
    expected_heading = expected_heading / cast(
        Array,
        jnp.linalg.norm(expected_heading),
    )
    displacement = next_state.agent_positions[0] - start

    assert bool(
        jnp.allclose(
            displacement,
            expected_heading * spawn_shield_movement_speed,
            atol=2 * GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
    )
    assert (
        observation.self_features[0, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
        == spawn_shield_movement_speed
    )
    assert (
        next_observation.self_features[0, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
        == spawn_shield_movement_speed
    )
    assert int(next_state.spawn_shield_durations[0]) == 2


def test_spawn_shield_stay_is_zero_movement_and_consumes_one_step() -> None:
    """Prove the speed override never turns Stay into forced movement."""
    config = _deterministic_config(
        team_size=1,
        spawn_shield_movement_speed=2.0,
    )
    start = jnp.asarray((10.0, 6.0), dtype=jnp.float32)
    config, state = _state_with_single_active_alive_agent(
        config,
        start,
        spawn_shield_duration=3,
    )

    next_state, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        _joint_action_with_moves((0, MOVE_STAY)),
        jax.random.key(73),
    )

    _assert_center_close(next_state.agent_positions[0], start)
    assert int(next_state.spawn_shield_durations[0]) == 2


@pytest.mark.parametrize(
    ("agent_a_duration", "agent_b_duration"),
    (
        pytest.param(3, 3, id="shielded-versus-shielded"),
        pytest.param(3, 0, id="shielded-versus-unshielded"),
    ),
)
def test_spawn_shielded_body_neither_pushes_nor_receives_displacement(
    agent_a_duration: int,
    agent_b_duration: int,
) -> None:
    """Prove one exempt endpoint makes the entire agent pair noncolliding."""
    config = _deterministic_config(
        team_size=5,
        spawn_shield_movement_speed=2.0,
    )
    agent_a_start = jnp.asarray((5.0, 5.0), dtype=jnp.float32)
    agent_b_start = jnp.asarray((7.0, 5.0), dtype=jnp.float32)
    config, state = _state_with_two_agents(
        config,
        agent_a_start,
        agent_b_start,
        agent_a_spawn_shield_duration=agent_a_duration,
        agent_b_spawn_shield_duration=agent_b_duration,
    )

    next_state, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        _joint_action_with_moves((0, MOVE_EAST), (1, MOVE_STAY)),
        jax.random.key(74),
    )

    _assert_center_close(next_state.agent_positions[0], agent_b_start)
    _assert_center_close(next_state.agent_positions[1], agent_b_start)


@pytest.mark.parametrize(
    ("spawn_shield_duration", "collision_is_expected"),
    (
        pytest.param(1, True, id="expiring-at-final-position"),
        pytest.param(2, False, id="still-shielded-at-final-position"),
    ),
)
def test_spawn_shield_expiry_controls_only_final_endpoint_collision(
    spawn_shield_duration: int,
    collision_is_expected: bool,
) -> None:
    """Prove counter one rejoins collision after collision-exempt traversal."""
    config = _deterministic_config(
        team_size=5,
        spawn_shield_movement_speed=2.0,
    )
    raw_shared_endpoint = jnp.asarray((6.0, 5.0), dtype=jnp.float32)
    config, state = _state_with_two_agents(
        config,
        jnp.asarray((4.0, 5.0), dtype=jnp.float32),
        raw_shared_endpoint,
        agent_a_spawn_shield_duration=spawn_shield_duration,
    )

    next_state, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        _joint_action_with_moves((0, MOVE_EAST), (1, MOVE_STAY)),
        jax.random.key(75),
    )

    if collision_is_expected:
        assert not bool(
            jnp.array_equal(next_state.agent_positions[0], raw_shared_endpoint)
        )
        assert not bool(
            jnp.array_equal(next_state.agent_positions[1], raw_shared_endpoint)
        )
        _assert_agents_do_not_overlap(
            next_state.agent_positions,
            slot_a=0,
            slot_b=1,
            radius_a=0.5,
            radius_b=0.5,
        )
    else:
        _assert_center_close(next_state.agent_positions[0], raw_shared_endpoint)
        _assert_center_close(next_state.agent_positions[1], raw_shared_endpoint)


def test_expiring_spawn_shield_rejects_charge_but_preserves_movement() -> None:
    """Keep an expiring shield movement-only for the complete transition."""
    config = _deterministic_config(
        team_size=5,
        ordinary_movement_distance_scale=0.1,
        spawn_shield_movement_speed=2.0,
    )
    blocker_slot = 1
    target_slot = MAX_AGENTS_PER_TEAM
    active_mask = _mask_with_true_slots(0, blocker_slot, target_slot)
    profile = config.agent_profile._replace(
        class_ids=config.agent_profile.class_ids.at[0].set(WARRIOR_CLASS_ID),
        active_mask=active_mask,
        agent_radii=_agent_radii_array_with_rows(
            (0, 0.5),
            (blocker_slot, 0.5),
            (target_slot, 0.5),
        ),
        base_movement_speeds=_slot_float_vector(0.0, (0, 0.25)),
        observation_radii=_slot_float_vector(
            0.0,
            (0, 20.0),
            (blocker_slot, 20.0),
            (target_slot, 20.0),
        ),
        basic_interaction_radii=_slot_float_vector(
            0.0,
            (0, 20.0),
            (blocker_slot, 20.0),
            (target_slot, 20.0),
        ),
        ultimate_interaction_radii=_slot_float_vector(
            0.0,
            (0, 20.0),
            (blocker_slot, 20.0),
            (target_slot, 20.0),
        ),
        max_health=active_mask.astype(jnp.float32),
    )
    config = config._replace(agent_profile=profile)
    charger_start = jnp.asarray((2.0, 6.0), dtype=jnp.float32)
    blocker_start = jnp.asarray((6.0, 6.0), dtype=jnp.float32)
    target_start = jnp.asarray((7.0, 6.0), dtype=jnp.float32)
    state = EnvState(
        step_count=jnp.asarray(0, dtype=jnp.int32),
        agent_positions=_agent_positions_array_with_rows(
            (0, charger_start),
            (blocker_slot, blocker_start),
            (target_slot, target_start),
        ),
        alive_mask=active_mask,
        team_respawn_wave_countdowns=config.team_respawn_wave_period_step_count - 1,
        spawn_shield_durations=_slot_int_vector(0, (0, 1)),
        **_inert_combat_state_fields(active_mask),
    )
    movement_only_action = _joint_action_with_moves((0, MOVE_NORTH))
    charge_action = movement_only_action._replace(
        select_target=movement_only_action.select_target.at[0].set(
            1 + MAX_AGENTS_PER_TEAM
        ),
        use_ultimate=movement_only_action.use_ultimate.at[0].set(1),
    )
    current_action_mask = _current_action_mask(config, state)

    charge_next_state, *_, charge_info = step(
        config,
        state,
        current_action_mask,
        charge_action,
        jax.random.key(81),
    )
    no_charge_next_state, *_ = step(
        config,
        state,
        current_action_mask,
        movement_only_action,
        jax.random.key(82),
    )

    # Shielding is still active when this action is accepted, so the combat
    # pair is rejected while its independently legal movement is preserved.
    _assert_center_close(
        no_charge_next_state.agent_positions[0],
        jnp.asarray((2.0, 8.0), dtype=jnp.float32),
    )
    _assert_center_close(
        charge_next_state.agent_positions[0],
        no_charge_next_state.agent_positions[0],
    )
    _assert_center_close(
        charge_next_state.agent_positions[blocker_slot],
        blocker_start,
    )
    _assert_center_close(
        no_charge_next_state.agent_positions[blocker_slot],
        blocker_start,
    )
    assert int(charge_next_state.spawn_shield_durations[0]) == 0
    assert int(no_charge_next_state.spawn_shield_durations[0]) == 0
    acceptance = charge_info.transition_facts.action_acceptance_facts
    combat_facts = charge_info.transition_facts.combat_transition_facts
    assert bool(acceptance.in_domain_combat_action_pair_is_rejected_by_actor[0])
    assert int(acceptance.accepted_joint_action.select_target[0]) == 0
    assert int(acceptance.accepted_joint_action.use_ultimate[0]) == 0
    assert not bool(combat_facts.ultimate_effect_is_activated_by_source[0])
    assert int(charge_next_state.ultimate_cooldowns[0]) == 0


@pytest.mark.parametrize(
    ("spawn_shield_duration", "collision_is_expected"),
    (
        pytest.param(1, True, id="expiring-stay"),
        pytest.param(2, False, id="protected-stay"),
    ),
)
def test_spawn_shield_stay_rejoins_collision_only_on_expiry(
    spawn_shield_duration: int,
    collision_is_expected: bool,
) -> None:
    """Prove zero movement still reaches the existing final collision pass."""
    config = _deterministic_config(team_size=5)
    first_start = jnp.asarray((5.0, 5.0), dtype=jnp.float32)
    second_start = jnp.asarray((5.5, 5.0), dtype=jnp.float32)
    config, state = _state_with_two_agents(
        config,
        first_start,
        second_start,
        agent_a_spawn_shield_duration=spawn_shield_duration,
    )

    next_state, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        _joint_action_with_moves(),
        jax.random.key(76),
    )

    if collision_is_expected:
        _assert_agents_do_not_overlap(
            next_state.agent_positions,
            slot_a=0,
            slot_b=1,
            radius_a=0.5,
            radius_b=0.5,
        )
    else:
        _assert_center_close(next_state.agent_positions[0], first_start)
        _assert_center_close(next_state.agent_positions[1], second_start)


def test_simultaneous_spawn_shield_expiry_is_team_and_slot_independent() -> None:
    """Prove equivalent same-team and opposing-team pairs resolve identically."""
    pair_results: list[Array] = []

    for second_slot in (1, MAX_AGENTS_PER_TEAM):
        config = _deterministic_config(team_size=5)
        config, state = _state_with_two_agents(
            config,
            jnp.asarray((5.0, 5.0), dtype=jnp.float32),
            jnp.asarray((5.5, 5.0), dtype=jnp.float32),
            agent_b_slot=second_slot,
            agent_a_spawn_shield_duration=1,
            agent_b_spawn_shield_duration=1,
        )

        next_state, *_ = step(
            config,
            state,
            _current_action_mask(config, state),
            _joint_action_with_moves((0, MOVE_STAY), (second_slot, MOVE_STAY)),
            jax.random.key(77),
        )
        pair_results.append(
            jnp.stack(
                (
                    next_state.agent_positions[0],
                    next_state.agent_positions[second_slot],
                )
            )
        )

    assert bool(jnp.array_equal(pair_results[0], pair_results[1]))


def test_spawn_shield_movement_keeps_bounds_and_obstacles_authoritative() -> None:
    """Prove collision exemption does not bypass static world geometry."""
    pillar_center = jnp.asarray((10.0, 6.0), dtype=jnp.float32)
    pillar_radius = 0.5
    obstacles = _obstacle_array_with_rows(
        (0, _pillar_obstacle(pillar_center, pillar_radius))
    )
    config = _deterministic_config(
        team_size=5,
        obstacles=obstacles,
        spawn_shield_movement_speed=2.0,
    )
    config, state = _state_with_two_agents(
        config,
        jnp.asarray((18.75, 10.0), dtype=jnp.float32),
        jnp.asarray((8.5, 6.0), dtype=jnp.float32),
        agent_a_spawn_shield_duration=3,
        agent_b_spawn_shield_duration=3,
    )

    next_state, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        _joint_action_with_moves((0, MOVE_EAST), (1, MOVE_EAST)),
        jax.random.key(78),
    )

    _assert_center_inside_bounds(
        next_state.agent_positions[0],
        radius=0.5,
        config=config,
    )
    _assert_center_outside_pillar(
        next_state.agent_positions[1],
        agent_radius=0.5,
        pillar_center=pillar_center,
        pillar_radius=pillar_radius,
    )
    assert bool(jnp.all(next_state.spawn_shield_durations[:2] == 2))


def test_spawn_shield_duration_paths_match_eager_jit_and_vmap() -> None:
    """Prove mixed counters batch through one fixed-shape transition program."""
    durations = (0, 1, 3, 7)
    config = _deterministic_config(
        team_size=1,
        ordinary_movement_distance_scale=0.25,
        spawn_shield_duration_steps=max(durations),
        spawn_shield_movement_speed=2.0,
    )
    states: list[EnvState] = []
    action_masks: list[ActionMask] = []

    for duration in durations:
        config, state = _state_with_single_active_alive_agent(
            config,
            jnp.asarray((3.0, 5.0), dtype=jnp.float32),
            effective_movement_speed=1.0,
            spawn_shield_duration=duration,
        )
        states.append(state)
        action_masks.append(_current_action_mask(config, state))

    batched_states = jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)
    batched_action_masks = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *action_masks,
    )
    action = _joint_action_with_moves((0, MOVE_EAST))
    keys = jax.random.split(jax.random.key(80), len(durations))

    def _batched_transition(
        states_batch: EnvState,
        masks_batch: ActionMask,
        step_keys: Array,
    ) -> tuple[Array, Array, Array, Array]:
        def _one_transition(
            state: EnvState,
            action_mask: ActionMask,
            step_key: Array,
        ) -> tuple[Array, Array, Array, Array]:
            next_state, _, _, _, _, info = step(
                config,
                state,
                action_mask,
                action,
                step_key,
            )
            physical_facts = info.transition_facts.physical_facts
            return (
                next_state.agent_positions[0],
                next_state.spawn_shield_durations[0],
                physical_facts.charge_phase_displacement_by_agent,
                physical_facts.ordinary_movement_phase_displacement_by_agent,
            )

        return jax.vmap(_one_transition)(
            states_batch,
            masks_batch,
            step_keys,
        )

    eager_positions, eager_durations, eager_charge, eager_ordinary = (
        _batched_transition(
            batched_states,
            batched_action_masks,
            keys,
        )
    )
    (
        compiled_positions,
        compiled_durations,
        compiled_charge,
        compiled_ordinary,
    ) = cast(
        tuple[Array, Array, Array, Array],
        jax.jit(_batched_transition)(
            batched_states,
            batched_action_masks,
            keys,
        ),
    )

    assert bool(jnp.array_equal(compiled_positions, eager_positions))
    assert bool(jnp.array_equal(compiled_durations, eager_durations))
    assert bool(jnp.array_equal(compiled_charge, eager_charge))
    assert bool(jnp.array_equal(compiled_ordinary, eager_ordinary))
    assert eager_charge.shape == (
        len(durations),
        MAX_AGENT_SLOTS,
        ENVIRONMENT_DIMENSIONS,
    )
    assert eager_ordinary.shape == eager_charge.shape
    assert bool(jnp.all(eager_charge == 0.0))
    assert bool(
        jnp.allclose(
            eager_ordinary[:, 0, 0],
            jnp.asarray((0.25, 2.0, 2.0, 2.0), dtype=jnp.float32),
        )
    )
    assert bool(jnp.all(eager_ordinary[:, 0, 1] == 0.0))
    assert bool(jnp.all(eager_ordinary[:, 1:] == 0.0))
    assert bool(
        jnp.allclose(
            eager_positions[:, 0],
            jnp.asarray((3.25, 5.0, 5.0, 5.0), dtype=jnp.float32),
            atol=2 * GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.array_equal(
            eager_durations,
            jnp.asarray((0, 0, 2, 6), dtype=jnp.int32),
        )
    )


def test_spawn_shield_trajectory_matches_eager_jit_and_scan() -> None:
    """Prove three protected moves then ordinary movement in a compiled rollout."""
    horizon = 4
    config = _deterministic_config(
        team_size=1,
        ordinary_movement_distance_scale=0.25,
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
    )
    config, initial_state = _state_with_single_active_alive_agent(
        config,
        jnp.asarray((3.0, 5.0), dtype=jnp.float32),
        effective_movement_speed=1.0,
        spawn_shield_duration=3,
    )
    initial_action_mask = _current_action_mask(config, initial_state)
    action = _joint_action_with_moves((0, MOVE_EAST))
    keys = jax.random.split(jax.random.key(79), horizon)

    def _rollout(
        state: EnvState,
        action_mask: ActionMask,
        step_keys: Array,
    ) -> tuple[Array, Array, Array]:
        def _scan_step(
            carry: tuple[EnvState, ActionMask],
            step_key: Array,
        ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array]]:
            current_state, current_mask = carry
            next_state, observation, _, _, next_mask, _ = step(
                config,
                current_state,
                current_mask,
                action,
                step_key,
            )
            return (next_state, next_mask), (
                next_state.agent_positions[0],
                next_state.spawn_shield_durations[0],
                observation.self_features[0, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED],
            )

        _, outputs = jax.lax.scan(
            _scan_step,
            (state, action_mask),
            step_keys,
        )
        return outputs

    eager_outputs = _rollout(initial_state, initial_action_mask, keys)
    compiled_outputs = cast(
        tuple[Array, Array, Array],
        jax.jit(_rollout)(initial_state, initial_action_mask, keys),
    )

    for eager_output, compiled_output in zip(
        eager_outputs,
        compiled_outputs,
        strict=True,
    ):
        assert bool(jnp.array_equal(eager_output, compiled_output))

    position_history, duration_history, speed_history = eager_outputs
    assert bool(
        jnp.allclose(
            position_history[:, 0],
            jnp.asarray((5.0, 7.0, 9.0, 9.25), dtype=jnp.float32),
            atol=2 * GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.array_equal(
            duration_history,
            jnp.asarray((2, 1, 0, 0), dtype=jnp.int32),
        )
    )
    assert bool(
        jnp.array_equal(
            speed_history,
            jnp.asarray((2.0, 2.0, 0.25, 0.25), dtype=jnp.float32),
        )
    )


@pytest.mark.parametrize(
    "active_stun_channels",
    (
        pytest.param((STUN_CHANNEL_WARRIOR_CHARGE,), id="warrior-charge"),
        pytest.param((STUN_CHANNEL_HUNTER_TRAP,), id="hunter-trap"),
        pytest.param((STUN_CHANNEL_ROGUE_POISON,), id="rogue-poison"),
        pytest.param(
            (
                STUN_CHANNEL_WARRIOR_CHARGE,
                STUN_CHANNEL_HUNTER_TRAP,
                STUN_CHANNEL_ROGUE_POISON,
            ),
            id="concurrent-stuns",
        ),
    ),
)
def test_current_stun_exposes_stay_only_and_suppresses_voluntary_movement(
    active_stun_channels: tuple[int, ...],
) -> None:
    """Prove every current stun source aligns speed, mask, and movement intent."""
    config = _deterministic_config(team_size=1)
    start = jnp.asarray((10.0, 6.0), dtype=jnp.float32)
    config, state = _state_with_single_active_alive_agent(config, start)
    for channel in active_stun_channels:
        state = state._replace(
            stun_durations=state.stun_durations.at[0, channel].set(1)
        )

    observation, current_action_mask = _build_observation_and_action_mask(state, config)

    assert observation.self_features[0, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED] == 0.0
    assert bool(current_action_mask.move_mask[0, MOVE_STAY])
    assert int(jnp.sum(current_action_mask.move_mask[0])) == 1

    next_state, next_observation, _, _, next_action_mask, _ = step(
        config,
        state,
        current_action_mask,
        _joint_action_with_moves((0, MOVE_EAST)),
        jax.random.key(50),
    )

    _assert_center_close(next_state.agent_positions[0], start)
    assert bool(jnp.all(next_state.stun_durations[0] == 0))
    assert (
        next_observation.self_features[0, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
        == config.agent_profile.base_movement_speeds[0]
    )
    assert bool(jnp.all(next_action_mask.move_mask[0]))


def test_forged_movement_mask_cannot_restore_stunned_voluntary_movement() -> None:
    """Prove direct transition enforcement survives a stale or forged mask."""
    config = _deterministic_config(team_size=1)
    start = jnp.asarray((10.0, 6.0), dtype=jnp.float32)
    config, state = _state_with_single_active_alive_agent(config, start)
    state = state._replace(
        stun_durations=state.stun_durations.at[0, STUN_CHANNEL_HUNTER_TRAP].set(2)
    )
    current_action_mask = _current_action_mask(config, state)
    forged_action_mask = current_action_mask._replace(
        move_mask=current_action_mask.move_mask.at[0].set(True)
    )

    next_state, next_observation, _, _, next_action_mask, _ = step(
        config,
        state,
        forged_action_mask,
        _joint_action_with_moves((0, MOVE_EAST)),
        jax.random.key(51),
    )

    _assert_center_close(next_state.agent_positions[0], start)
    assert next_state.stun_durations[0, STUN_CHANNEL_HUNTER_TRAP] == 1
    assert (
        next_observation.self_features[0, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED] == 0.0
    )
    assert bool(next_action_mask.move_mask[0, MOVE_STAY])
    assert int(jnp.sum(next_action_mask.move_mask[0])) == 1


def test_collision_projection_may_displace_a_stunned_zero_intent_body() -> None:
    """Prove stun removes voluntary agency without freezing physical geometry."""
    config = _deterministic_config(team_size=1)
    stunned_start = jnp.asarray((5.0, 5.0), dtype=jnp.float32)
    neighbor_start = jnp.asarray((5.5, 5.0), dtype=jnp.float32)
    config, state = _state_with_two_agents(
        config,
        stunned_start,
        neighbor_start,
        radius=0.5,
    )
    state = state._replace(
        stun_durations=state.stun_durations.at[0, STUN_CHANNEL_HUNTER_TRAP].set(2)
    )

    next_state, next_observation, _, _, _, info = step(
        config,
        state,
        _current_action_mask(config, state),
        _joint_action_with_moves(),
        jax.random.key(52),
    )

    assert not bool(jnp.array_equal(next_state.agent_positions[0], stunned_start))
    assert (
        next_observation.self_features[0, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED] == 0.0
    )
    _assert_agents_do_not_overlap(
        next_state.agent_positions,
        slot_a=0,
        slot_b=1,
        radius_a=0.5,
        radius_b=0.5,
    )
    physical_facts = info.transition_facts.physical_facts
    assert bool(jnp.all(physical_facts.charge_phase_displacement_by_agent == 0.0))
    assert bool(
        jnp.allclose(
            physical_facts.ordinary_movement_phase_displacement_by_agent,
            next_state.agent_positions - state.agent_positions,
        )
    )


def test_stun_control_trajectory_matches_jit_and_scan() -> None:
    """Prove paired masks preserve D=2 stun control across compiled rollout."""
    horizon = 3
    config = _deterministic_config(team_size=1)
    start = jnp.asarray((5.0, 5.0), dtype=jnp.float32)
    config, initial_state = _state_with_single_active_alive_agent(config, start)
    initial_state = initial_state._replace(
        stun_durations=initial_state.stun_durations.at[0, STUN_CHANNEL_HUNTER_TRAP].set(
            2
        )
    )
    initial_action_mask = _current_action_mask(config, initial_state)
    action = _joint_action_with_moves((0, MOVE_EAST))
    keys = jax.random.split(jax.random.key(53), horizon)

    def _rollout(
        state: EnvState, action_mask: ActionMask, step_keys: Array
    ) -> tuple[Array, Array, Array, Array]:
        def _scan_step(
            carry: tuple[EnvState, ActionMask], key: Array
        ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array, Array]]:
            current_state, current_mask = carry
            next_state, observation, _, _, next_mask, _ = step(
                config, current_state, current_mask, action, key
            )
            outputs = (
                next_state.agent_positions[0],
                next_state.stun_durations[0, STUN_CHANNEL_HUNTER_TRAP],
                observation.self_features[0, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED],
                next_mask.move_mask[0],
            )
            return (next_state, next_mask), outputs

        _, outputs = jax.lax.scan(_scan_step, (state, action_mask), step_keys)
        return outputs

    eager_outputs = _rollout(initial_state, initial_action_mask, keys)
    compiled_outputs = cast(
        tuple[Array, Array, Array, Array],
        jax.jit(_rollout)(initial_state, initial_action_mask, keys),
    )

    for eager_output, compiled_output in zip(
        eager_outputs, compiled_outputs, strict=True
    ):
        assert bool(jnp.array_equal(eager_output, compiled_output))

    position_history, duration_history, speed_history, move_mask_history = eager_outputs
    assert bool(
        jnp.array_equal(
            position_history[:, 0],
            jnp.asarray((5.0, 5.0, 6.0), dtype=jnp.float32),
        )
    )
    assert bool(jnp.array_equal(duration_history, jnp.asarray((1, 0, 0))))
    assert bool(
        jnp.array_equal(
            speed_history,
            jnp.asarray((0.0, 1.0, 1.0), dtype=jnp.float32),
        )
    )
    assert int(jnp.sum(move_mask_history[0])) == 1
    assert bool(jnp.all(move_mask_history[1:]))


def test_step_can_be_jit_compiled_with_non_stay_movement() -> None:
    jitted_step = jax.jit(step)

    config = _deterministic_config(team_size=3, max_steps=10)
    key = jax.random.key(42)

    agent_a_start = jnp.array([5.0, 5.0], dtype=jnp.float32)
    agent_b_start = jnp.array([15.0, 5.0], dtype=jnp.float32)

    config, state = _state_with_two_agents(
        config,
        agent_a_start,
        agent_b_start,
        agent_a_active_flag=True,
        agent_a_alive_flag=True,
        agent_b_active_flag=True,
        agent_b_alive_flag=True,
    )
    joint_action = _joint_action_with_moves(
        (0, MOVE_EAST),
        (1, MOVE_STAY),
    )

    next_state, observation, reward, done_flags, action_mask, info = cast(
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        jitted_step(
            config,
            state,
            _current_action_mask(config, state),
            joint_action,
            key,
        ),
    )

    _assert_center_close(
        next_state.agent_positions[0],
        jnp.array([6.0, 5.0], dtype=jnp.float32),
    )
    _assert_center_close(next_state.agent_positions[1], agent_b_start)

    _assert_state_contract(next_state)
    _assert_observation_contract(observation)
    _assert_reward_contract(reward)
    _assert_done_flags_contract(done_flags, expected_truncated=False)
    _assert_action_mask_contract(action_mask)

    assert isinstance(info, Info)


def test_step_can_run_non_stay_movement_in_jitted_scanned_rollout() -> None:
    horizon = 3
    config = _deterministic_config(team_size=5, max_steps=1000)
    key = jax.random.key(42)

    agent_a_start = jnp.array([5.0, 5.0], dtype=jnp.float32)
    agent_b_start = jnp.array([15.0, 5.0], dtype=jnp.float32)

    config, state = _state_with_two_agents(
        config,
        agent_a_start,
        agent_b_start,
        agent_a_active_flag=True,
        agent_a_alive_flag=True,
        agent_b_active_flag=True,
        agent_b_alive_flag=True,
    )

    joint_action = _joint_action_with_moves(
        (0, MOVE_EAST),
        (1, MOVE_STAY),
    )
    initial_action_mask = _current_action_mask(config, state)

    def _rollout(
        initial_state: EnvState,
        initial_mask: ActionMask,
        step_keys: Array,
    ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array]]:
        """Run a compiled fixed-horizon rollout with stable scan outputs."""

        def _step_wrapper(
            carry: tuple[EnvState, ActionMask],
            step_key: Array,
        ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array]]:
            current_state, current_action_mask = carry
            new_state, _, _, _, next_action_mask, info = step(
                config,
                current_state,
                current_action_mask,
                joint_action,
                step_key,
            )
            return (new_state, next_action_mask), (
                new_state.step_count,
                new_state.agent_positions,
                info.transition_facts.physical_facts.ordinary_movement_phase_displacement_by_agent,
            )

        return jax.lax.scan(
            f=_step_wrapper,
            init=(initial_state, initial_mask),
            xs=step_keys,
            length=horizon,
        )

    keys = jax.random.split(key, horizon)
    assert keys.shape == (horizon,)

    (final_state, final_action_mask), history = cast(
        tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array]],
        jax.jit(_rollout)(state, initial_action_mask, keys),
    )
    step_count_history, position_history, ordinary_displacement_history = history

    expected_agent_a_position = jnp.array([8.0, 5.0], dtype=jnp.float32)
    expected_agent_a_history = jnp.array(
        [
            [6.0, 5.0],
            [7.0, 5.0],
            [8.0, 5.0],
        ],
        dtype=jnp.float32,
    )

    _assert_center_close(final_state.agent_positions[0], expected_agent_a_position)
    _assert_center_close(final_state.agent_positions[1], agent_b_start)

    assert final_state.step_count == horizon
    assert final_state.step_count.dtype == jnp.int32
    assert jax.tree_util.tree_structure(
        final_action_mask
    ) == jax.tree_util.tree_structure(initial_action_mask)

    assert step_count_history.shape == (horizon,)
    assert step_count_history.dtype == jnp.int32
    assert bool(jnp.all(step_count_history == jnp.array([1, 2, 3], dtype=jnp.int32)))

    _assert_position_history_contract(position_history, horizon=horizon)
    assert bool(
        jnp.allclose(
            position_history[:, 0, :],
            expected_agent_a_history,
            atol=GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
    )
    assert bool(
        jnp.allclose(
            position_history[:, 1, :],
            jnp.broadcast_to(agent_b_start, (horizon, ENVIRONMENT_DIMENSIONS)),
            atol=GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
    )
    expected_ordinary_displacement_history = (
        jnp.zeros((horizon, MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
        .at[:, 0, 0]
        .set(1.0)
    )
    assert ordinary_displacement_history.shape == (
        horizon,
        MAX_AGENT_SLOTS,
        ENVIRONMENT_DIMENSIONS,
    )
    assert ordinary_displacement_history.dtype == jnp.float32
    assert bool(
        jnp.array_equal(
            ordinary_displacement_history,
            expected_ordinary_displacement_history,
        )
    )

    _assert_state_contract(final_state)
