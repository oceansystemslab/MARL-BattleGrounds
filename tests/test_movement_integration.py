"""Movement integration tests for Milestone 4 Step 3."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.env import step
from marl_battlegrounds.core.geometry import GEOMETRY_TOLERANCE
from marl_battlegrounds.core.types import (
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
    NUM_TARGET_ACTIONS,
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

# Test Helpers ---


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
    default_movement_speed: float = 1.0,
    map_width: float = 20.0,
    map_height: float = 12.0,
    obstacles: Array | None = None,
) -> EnvConfig:
    """Create a deterministic config for movement integration tests."""
    return EnvConfig(
        team_size=team_size,
        max_steps=max_steps,
        map_width=map_width,
        map_height=map_height,
        default_agent_radius=0.5,
        default_movement_speed=default_movement_speed,
        default_observation_radius=8.0,
        default_basic_interaction_radius=6.0,
        default_ultimate_interaction_radius=9.0,
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
    """Create the placeholder neutral class-id vector."""
    return jnp.full((MAX_AGENT_SLOTS,), CLASS_NEUTRAL, dtype=jnp.int32)


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
    effective_movement_speed: float = 1.0,
    effective_observation_radius: float = 8.0,
    effective_basic_interaction_radius: float = 6.0,
    effective_ultimate_interaction_radius: float = 9.0,
    step_count: int = 0,
) -> EnvState:
    """Create a valid state with slot 0 active and alive."""
    return EnvState(
        step_count=jnp.array(step_count, dtype=jnp.int32),
        agent_positions=_agent_positions_array_with_rows((0, position)),
        agent_radii=_agent_radii_array_with_rows((0, radius)),
        team_ids=_team_ids(),
        class_ids=_neutral_class_ids(),
        movement_speeds=_slot_float_vector(1.0, (0, effective_movement_speed)),
        observation_radii=_slot_float_vector(8.0, (0, effective_observation_radius)),
        basic_interaction_radii=_slot_float_vector(
            6.0, (0, effective_basic_interaction_radius)
        ),
        ultimate_interaction_radii=_slot_float_vector(
            9.0, (0, effective_ultimate_interaction_radius)
        ),
        active_mask=_mask_with_true_slots(0),
        alive_mask=_mask_with_true_slots(0),
    )


def _state_with_two_agents(
    agent_a_position: Array,
    agent_b_position: Array,
    agent_a_active_flag: bool = True,
    agent_a_alive_flag: bool = True,
    agent_b_active_flag: bool = True,
    agent_b_alive_flag: bool = True,
    *,
    radius: float = 0.5,
    agent_a_effective_movement_speed: float = 1.0,
    agent_b_effective_movement_speed: float = 1.0,
    effective_observation_radius: float = 8.0,
    effective_basic_interaction_radius: float = 6.0,
    effective_ultimate_interaction_radius: float = 9.0,
    step_count: int = 0,
) -> EnvState:
    """Create a valid state with two agents and configurable participation flags."""
    active_mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)
    alive_mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)

    active_mask = active_mask.at[0].set(agent_a_active_flag)
    alive_mask = alive_mask.at[0].set(agent_a_alive_flag)

    active_mask = active_mask.at[1].set(agent_b_active_flag)
    alive_mask = alive_mask.at[1].set(agent_b_alive_flag)

    return EnvState(
        step_count=jnp.array(step_count, dtype=jnp.int32),
        agent_positions=_agent_positions_array_with_rows(
            (0, agent_a_position),
            (1, agent_b_position),
        ),
        agent_radii=_agent_radii_array_with_rows((0, radius), (1, radius)),
        team_ids=_team_ids(),
        class_ids=_neutral_class_ids(),
        movement_speeds=_slot_float_vector(
            1.0,
            (0, agent_a_effective_movement_speed),
            (1, agent_b_effective_movement_speed),
        ),
        observation_radii=_slot_float_vector(
            8.0,
            (0, effective_observation_radius),
            (1, effective_observation_radius),
        ),
        basic_interaction_radii=_slot_float_vector(
            6.0,
            (0, effective_basic_interaction_radius),
            (1, effective_basic_interaction_radius),
        ),
        ultimate_interaction_radii=_slot_float_vector(
            9.0,
            (0, effective_ultimate_interaction_radius),
            (1, effective_ultimate_interaction_radius),
        ),
        active_mask=active_mask,
        alive_mask=alive_mask,
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

    assert state.agent_radii.shape == (MAX_AGENT_SLOTS,)
    assert state.agent_radii.dtype == jnp.float32

    assert state.team_ids.shape == (MAX_AGENT_SLOTS,)
    assert state.team_ids.dtype == jnp.int32

    assert state.class_ids.shape == (MAX_AGENT_SLOTS,)
    assert state.class_ids.dtype == jnp.int32

    assert state.movement_speeds.shape == (MAX_AGENT_SLOTS,)
    assert state.movement_speeds.dtype == jnp.float32

    assert state.observation_radii.shape == (MAX_AGENT_SLOTS,)
    assert state.observation_radii.dtype == jnp.float32

    assert state.basic_interaction_radii.shape == (MAX_AGENT_SLOTS,)
    assert state.basic_interaction_radii.dtype == jnp.float32

    assert state.ultimate_interaction_radii.shape == (MAX_AGENT_SLOTS,)
    assert state.ultimate_interaction_radii.dtype == jnp.float32

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


def _assert_single_agent_action_mask_semantics(action_mask: ActionMask) -> None:
    """Assert current action-mask semantics for one active alive slot."""
    assert bool(jnp.all(action_mask.move[0]))
    assert bool(jnp.all(~action_mask.move[1:]))

    assert bool(action_mask.target[0, 0])
    assert bool(action_mask.target[0, 1])
    assert bool(jnp.all(~action_mask.target[0, 2:]))
    assert bool(jnp.all(~action_mask.target[1:]))

    assert bool(action_mask.use_ultimate[0, 0])
    assert bool(~action_mask.use_ultimate[0, 1])
    assert bool(jnp.all(~action_mask.use_ultimate[1:]))


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
    _assert_single_agent_action_mask_semantics(action_mask)
    assert isinstance(info, Info)


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
    state = _state_with_single_active_alive_agent(
        jnp.array([10.0, 6.0], dtype=jnp.float32),
        effective_movement_speed=2.5,
    )
    joint_action = _joint_action_with_moves((0, move_action))

    next_state, *_ = step(config, state, joint_action, key)

    _assert_center_close(next_state.agent_positions[0], expected_position)


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
            state.movement_speeds[0],
            atol=GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
    )


def test_diagonal_moves_scale_by_effective_state_movement_speed() -> None:
    effective_movement_speed = 2.0
    config = _deterministic_config()
    key = jax.random.key(42)
    start = jnp.array([10.0, 10.0], dtype=jnp.float32)
    state = _state_with_single_active_alive_agent(
        start, effective_movement_speed=effective_movement_speed
    )
    joint_action = _joint_action_with_moves((0, MOVE_NORTHEAST))

    next_state, *_ = step(config, state, joint_action, key)

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
    state = _state_with_two_agents(
        agent_a_start,
        agent_b_start,
        agent_a_effective_movement_speed=1.0,
        agent_b_effective_movement_speed=2.5,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST), (1, MOVE_EAST))

    next_state, *_ = step(config, state, joint_action, key)

    _assert_center_close(
        next_state.agent_positions[0],
        jnp.array([6.0, 5.0], dtype=jnp.float32),
    )
    _assert_center_close(
        next_state.agent_positions[1],
        jnp.array([14.5, 5.0], dtype=jnp.float32),
    )


def test_step_ignores_config_default_movement_speed_after_state_exists() -> None:
    config = _deterministic_config(default_movement_speed=9.0)
    key = jax.random.key(42)
    start = jnp.array([5.0, 5.0], dtype=jnp.float32)
    state = _state_with_single_active_alive_agent(start, effective_movement_speed=1.25)
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(config, state, joint_action, key)

    _assert_center_close(
        next_state.agent_positions[0],
        jnp.array([6.25, 5.0], dtype=jnp.float32),
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
    assert jnp.array_equal(next_state.agent_radii, state.agent_radii)
    assert jnp.array_equal(next_state.team_ids, state.team_ids)
    assert jnp.array_equal(next_state.class_ids, state.class_ids)
    assert jnp.array_equal(next_state.movement_speeds, state.movement_speeds)
    assert jnp.array_equal(next_state.observation_radii, state.observation_radii)
    assert jnp.array_equal(
        next_state.basic_interaction_radii, state.basic_interaction_radii
    )
    assert jnp.array_equal(
        next_state.ultimate_interaction_radii, state.ultimate_interaction_radii
    )
    assert jnp.array_equal(next_state.active_mask, state.active_mask)
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
    state = _state_with_single_active_alive_agent(
        jnp.array([10.0, 10.0], dtype=jnp.float32),
        step_count=0,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    _, _, _, done_flags, _, _ = step(config, state, joint_action, key)

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
    state = _state_with_single_active_alive_agent(
        start,
        radius=agent_radius,
    )
    joint_action = _joint_action_with_moves((0, move_action))

    next_state, *_ = step(config, state, joint_action, key)

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
    state = _state_with_single_active_alive_agent(
        jnp.array([10.0, 10.0], dtype=jnp.float32),
        radius=agent_radius,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(config, state, joint_action, key)

    _assert_center_outside_pillar(
        next_state.agent_positions[0],
        agent_radius=agent_radius,
        pillar_center=pillar_center,
        pillar_radius=pillar_radius,
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
    state = _state_with_single_active_alive_agent(
        jnp.array([10.0, 10.0], dtype=jnp.float32),
        radius=agent_radius,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(config, state, joint_action, key)

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

    state = _state_with_single_active_alive_agent(
        jnp.array([10.0, 10.0], dtype=jnp.float32)
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(config, state, joint_action, key)

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

    state = _state_with_single_active_alive_agent(
        jnp.array([10.0, 10.0], dtype=jnp.float32)
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(config, state, joint_action, key)

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
    state = _state_with_single_active_alive_agent(
        jnp.array([10.0, 10.0], dtype=jnp.float32),
        radius=agent_radius,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(config, state, joint_action, key)

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
    state = _state_with_single_active_alive_agent(
        jnp.array([10.0, 10.0], dtype=jnp.float32),
        radius=agent_radius,
    )
    joint_action = _joint_action_with_moves((0, MOVE_EAST))

    next_state, *_ = step(config, state, joint_action, key)

    _assert_center_outside_rotated_wall(
        next_state.agent_positions[0],
        agent_radius=agent_radius,
        wall_center=wall_center,
        wall_width=wall_width,
        wall_height=wall_height,
        theta=float(theta),
    )


def test_inactive_slots_with_nonstay_action_preserve_original_positions() -> None:
    config = _deterministic_config()
    key = jax.random.key(42)

    agent_a_position = jnp.array([10.0, 10.0], dtype=jnp.float32)
    agent_b_position = jnp.array([12.0, 10.0], dtype=jnp.float32)

    state = _state_with_two_agents(
        agent_a_position,
        agent_b_position,
        agent_a_active_flag=False,
        agent_a_alive_flag=True,
        agent_b_active_flag=False,
        agent_b_alive_flag=True,
    )

    joint_action = _joint_action_with_moves(
        (0, MOVE_EAST),
        (1, MOVE_WEST),
    )

    next_state, *_ = step(config, state, joint_action, key)

    _assert_center_close(next_state.agent_positions[0], agent_a_position)
    _assert_center_close(next_state.agent_positions[1], agent_b_position)


def test_dead_slots_with_nonstay_action_preserve_original_positions() -> None:
    config = _deterministic_config()
    key = jax.random.key(42)

    agent_a_position = jnp.array([10.0, 10.0], dtype=jnp.float32)
    agent_b_position = jnp.array([12.0, 10.0], dtype=jnp.float32)

    state = _state_with_two_agents(
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

    next_state, *_ = step(config, state, joint_action, key)

    _assert_center_close(next_state.agent_positions[0], agent_a_position)
    _assert_center_close(next_state.agent_positions[1], agent_b_position)


@pytest.mark.parametrize(
    ("agent_b_active_flag", "agent_b_alive_flag"),
    [
        pytest.param(False, True, id="inactive_alive_neighbor"),
        pytest.param(True, False, id="active_dead_neighbor"),
        pytest.param(False, False, id="inactive_dead_neighbor"),
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

    state = _state_with_two_agents(
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

    next_state, *_ = step(config, state, joint_action, key)

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

    state = EnvState(
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=_agent_positions_array_with_rows(
            (0, agent_a_position),
            (blocker_slot, blocker_position),
        ),
        agent_radii=_agent_radii_array_with_rows(
            (0, radius),
            (blocker_slot, radius),
        ),
        team_ids=_team_ids(),
        class_ids=_neutral_class_ids(),
        movement_speeds=_slot_float_vector(1.0),
        observation_radii=_slot_float_vector(8.0),
        basic_interaction_radii=_slot_float_vector(6.0),
        ultimate_interaction_radii=_slot_float_vector(9.0),
        active_mask=_mask_with_true_slots(0, blocker_slot),
        alive_mask=_mask_with_true_slots(0, blocker_slot),
    )

    joint_action = _joint_action_with_moves(
        (0, MOVE_EAST),
        (blocker_slot, MOVE_STAY),
    )

    next_state, *_ = step(config, state, joint_action, key)

    _assert_agent_positions_are_finite(next_state.agent_positions)
    _assert_agents_do_not_overlap(
        next_state.agent_positions,
        slot_a=0,
        slot_b=blocker_slot,
        radius_a=radius,
        radius_b=radius,
    )


def test_step_can_be_jit_compiled_with_non_stay_movement() -> None:
    jitted_step = jax.jit(step)

    config = _deterministic_config(team_size=3, max_steps=10)
    key = jax.random.key(42)

    agent_a_start = jnp.array([5.0, 5.0], dtype=jnp.float32)
    agent_b_start = jnp.array([15.0, 5.0], dtype=jnp.float32)

    state = _state_with_two_agents(
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
        jitted_step(config, state, joint_action, key),
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

    state = _state_with_two_agents(
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

    def _rollout(
        initial_state: EnvState,
        step_keys: Array,
    ) -> tuple[EnvState, tuple[Array, Array]]:
        """Run a compiled fixed-horizon rollout with stable scan outputs."""

        def _step_wrapper(
            carry: EnvState,
            step_key: Array,
        ) -> tuple[EnvState, tuple[Array, Array]]:
            new_state, _, _, _, _, _ = step(config, carry, joint_action, step_key)
            return new_state, (new_state.step_count, new_state.agent_positions)

        return jax.lax.scan(
            f=_step_wrapper,
            init=initial_state,
            xs=step_keys,
            length=horizon,
        )

    keys = jax.random.split(key, horizon)
    assert keys.shape == (horizon,)

    final_state, history = cast(
        tuple[EnvState, tuple[Array, Array]],
        jax.jit(_rollout)(state, keys),
    )
    step_count_history, position_history = history

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

    _assert_state_contract(final_state)
