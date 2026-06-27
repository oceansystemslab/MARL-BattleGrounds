"""Tests for shared simulator geometry helpers."""

# pyright: reportPrivateUsage=false

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

# Step 2 tests private kernels directly because the design requires low-level
# geometry coverage; production modules should call the public composed API.
from marl_battlegrounds.core.geometry import (
    GEOMETRY_TOLERANCE,
    _project_disc_out_of_obstacle,
    _project_disc_out_of_obstacles,
    _project_disc_out_of_pillar,
    _project_disc_out_of_wall,
    _project_disc_to_bounds,
    _resolve_agent_agent_overlaps,
    _segment_intersects_circle,
    _segment_intersects_rotated_rect,
    has_clear_line_of_sight,
    project_movement_with_geometry,
)
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_OBSTACLE_SLOTS,
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
)

# Test helpers ---


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


def _active_none_obstacle_with_geometry() -> Array:
    """Create an active none row with geometry fields that should be ignored."""
    obstacle = _empty_obstacle()

    obstacle = obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_NONE)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(0.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(0.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_RADIUS].set(10.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_WIDTH].set(10.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_HEIGHT].set(10.0)

    return obstacle


def _create_obstacle_array() -> Array:
    """Create a mixed fixed-size obstacle array for projection tests."""
    late_pillar_slot = MAX_OBSTACLE_SLOTS - 1
    axis_wall_slot = MAX_OBSTACLE_SLOTS - 3
    inactive_pillar_slot = MAX_OBSTACLE_SLOTS - 6
    inactive_rotated_wall_slot = MAX_OBSTACLE_SLOTS - 7
    rotated_wall_slot = MAX_OBSTACLE_SLOTS - 10

    assert rotated_wall_slot >= 0

    return _obstacle_array_with_rows(
        (
            late_pillar_slot,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                1.0,
            ),
        ),
        (
            axis_wall_slot,
            _wall_obstacle(
                jnp.array([4.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
        ),
        (
            inactive_pillar_slot,
            _pillar_obstacle(
                jnp.array([0.0, 4.0], dtype=jnp.float32),
                1.0,
                active=False,
            ),
        ),
        (
            inactive_rotated_wall_slot,
            _wall_obstacle(
                jnp.array([4.0, 4.0], dtype=jnp.float32),
                width=2.0,
                height=1.0,
                theta=jnp.pi / 4.0,
                active=False,
            ),
        ),
        (
            rotated_wall_slot,
            _wall_obstacle(
                jnp.array([-4.0, 0.0], dtype=jnp.float32),
                width=1.0,
                height=3.0,
                theta=jnp.pi / 6.0,
            ),
        ),
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
    """Create a slot mask with only the provided slots marked true."""
    mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)

    for slot in slots:
        assert 0 <= slot < MAX_AGENT_SLOTS
        mask = mask.at[slot].set(True)

    return mask


def _movement_deltas_array_with_rows(*rows: tuple[int, Array]) -> Array:
    """Create a padded movement-delta array with selected slots populated."""
    intended_movement_deltas = jnp.zeros(
        (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        dtype=jnp.float32,
    )

    for slot, movement_delta in rows:
        assert 0 <= slot < MAX_AGENT_SLOTS
        intended_movement_deltas = intended_movement_deltas.at[slot].set(movement_delta)

    return intended_movement_deltas


def _assert_center_close(result: Array, expected: Array) -> None:
    """Assert that a geometry helper returned the expected float32 center."""
    assert result.shape == (2,)
    assert result.dtype == jnp.float32
    assert bool(
        jnp.allclose(
            result,
            expected,
            atol=GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
    )


def _assert_scalar_bool(result: Array, expected: bool) -> None:
    """Assert that a geometry predicate returned the expected scalar bool."""
    assert result.shape == ()
    assert result.dtype == bool
    assert bool(result) is expected


def _assert_agent_positions_close(result: Array, expected: Array) -> None:
    """Assert that slot-aligned agent positions match the expected values."""
    assert result.shape == (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS)
    assert result.dtype == jnp.float32
    assert bool(
        jnp.allclose(
            result,
            expected,
            atol=GEOMETRY_TOLERANCE,
            rtol=0.0,
        )
    )


def _assert_agent_positions_are_finite(agent_positions: Array) -> None:
    """Assert that projected positions contain no NaNs or infinities."""
    assert bool(jnp.all(jnp.isfinite(agent_positions)))


def _assert_active_alive_agents_inside_bounds(
    agent_positions: Array,
    agent_radii: Array,
    active_mask: Array,
    alive_mask: Array,
    map_width: Array | float,
    map_height: Array | float,
) -> None:
    """Assert the hard map-boundary invariant for active alive agents."""
    width = float(map_width)
    height = float(map_height)

    for slot in range(MAX_AGENT_SLOTS):
        participates = bool(active_mask[slot]) and bool(alive_mask[slot])
        if not participates:
            continue

        radius = float(agent_radii[slot])
        x_position = float(agent_positions[slot, 0])
        y_position = float(agent_positions[slot, 1])

        assert x_position >= radius - GEOMETRY_TOLERANCE
        assert x_position <= width - radius + GEOMETRY_TOLERANCE
        assert y_position >= radius - GEOMETRY_TOLERANCE
        assert y_position <= height - radius + GEOMETRY_TOLERANCE


def _max_active_alive_agent_overlap_residual(
    agent_positions: Array,
    agent_radii: Array,
    active_mask: Array,
    alive_mask: Array,
) -> float:
    """Return the largest fixed-pass residual overlap among active alive agents."""
    max_residual = 0.0

    for first_slot in range(MAX_AGENT_SLOTS):
        first_participates = bool(active_mask[first_slot]) and bool(
            alive_mask[first_slot]
        )
        if not first_participates:
            continue

        for second_slot in range(first_slot + 1, MAX_AGENT_SLOTS):
            second_participates = bool(active_mask[second_slot]) and bool(
                alive_mask[second_slot]
            )
            if not second_participates:
                continue

            center_distance = cast(
                Array,
                jnp.linalg.norm(
                    agent_positions[first_slot] - agent_positions[second_slot]
                ),
            )
            minimum_distance = agent_radii[first_slot] + agent_radii[second_slot]
            residual = float(
                jnp.maximum(0.0, minimum_distance - center_distance),
            )
            max_residual = max(max_residual, residual)

    return max_residual


# Tests ---


@pytest.mark.parametrize(
    ("agent_center", "agent_radius", "map_width", "map_height", "expected"),
    [
        pytest.param(
            jnp.array([-1.0, 2.0], dtype=jnp.float32),
            1.0,
            5.0,
            4.0,
            jnp.array([1.0, 2.0], dtype=jnp.float32),
            id="outside-left",
        ),
        pytest.param(
            jnp.array([6.0, 2.0], dtype=jnp.float32),
            1.0,
            5.0,
            4.0,
            jnp.array([4.0, 2.0], dtype=jnp.float32),
            id="outside-right",
        ),
        pytest.param(
            jnp.array([2.0, -1.0], dtype=jnp.float32),
            1.0,
            5.0,
            4.0,
            jnp.array([2.0, 1.0], dtype=jnp.float32),
            id="below-bottom",
        ),
        pytest.param(
            jnp.array([2.0, 5.0], dtype=jnp.float32),
            1.0,
            5.0,
            4.0,
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            id="above-top",
        ),
        pytest.param(
            jnp.array([2.5, 2.0], dtype=jnp.float32),
            1.0,
            5.0,
            4.0,
            jnp.array([2.5, 2.0], dtype=jnp.float32),
            id="already-valid",
        ),
        pytest.param(
            jnp.array([-2.0, 9.0], dtype=jnp.float32),
            1.5,
            10.0,
            6.0,
            jnp.array([1.5, 4.5], dtype=jnp.float32),
            id="non-square-map-both-axes",
        ),
    ],
)
def test_project_disc_to_bounds(
    agent_center: Array,
    agent_radius: Array | float,
    map_width: Array | float,
    map_height: Array | float,
    expected: Array,
) -> None:
    result = _project_disc_to_bounds(agent_center, agent_radius, map_width, map_height)

    _assert_center_close(result, expected)


@pytest.mark.parametrize(
    ("agent_center", "agent_radius", "obstacle", "expected"),
    [
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                1.0,
            ),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            id="active-pillar-overlap-projects-positive-x",
        ),
        pytest.param(
            jnp.array([-1.5, 0.0], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                1.0,
            ),
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            id="active-pillar-overlap-projects-negative-x",
        ),
        pytest.param(
            jnp.array([0.0, 1.5], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                1.0,
            ),
            jnp.array([0.0, 2.0], dtype=jnp.float32),
            id="active-pillar-overlap-projects-positive-y",
        ),
        pytest.param(
            jnp.array([0.0, -1.5], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                1.0,
            ),
            jnp.array([0.0, -2.0], dtype=jnp.float32),
            id="active-pillar-overlap-projects-negative-y",
        ),
        pytest.param(
            jnp.array([0.6, 0.8], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                1.0,
            ),
            jnp.array([1.2, 1.6], dtype=jnp.float32),
            id="active-pillar-diagonal-overlap-projects-along-ray",
        ),
        pytest.param(
            jnp.array([4.5, 4.0], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([3.0, 4.0], dtype=jnp.float32),
                1.0,
            ),
            jnp.array([5.0, 4.0], dtype=jnp.float32),
            id="shifted-active-pillar-overlap-projects-relative-to-pillar",
        ),
        pytest.param(
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                1.0,
            ),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            id="active-pillar-tangent-unchanged",
        ),
        pytest.param(
            jnp.array([3.0, 0.0], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                1.0,
            ),
            jnp.array([3.0, 0.0], dtype=jnp.float32),
            id="active-pillar-separated-unchanged",
        ),
        pytest.param(
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                1.0,
            ),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            id="coincident-centers-use-positive-x-fallback",
        ),
        pytest.param(
            jnp.array([3.0, 4.0], dtype=jnp.float32),
            0.5,
            _pillar_obstacle(
                jnp.array([3.0, 4.0], dtype=jnp.float32),
                1.25,
            ),
            jnp.array([4.75, 4.0], dtype=jnp.float32),
            id="shifted-coincident-centers-use-positive-x-fallback",
        ),
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                1.0,
                active=False,
            ),
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            id="inactive-pillar-unchanged",
        ),
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _empty_obstacle(),
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            id="empty-obstacle-row-unchanged",
        ),
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            id="active-wall-unchanged",
        ),
    ],
)
def test_project_disc_out_of_pillar(
    agent_center: Array,
    agent_radius: Array | float,
    obstacle: Array,
    expected: Array,
) -> None:
    result = _project_disc_out_of_pillar(agent_center, agent_radius, obstacle)

    _assert_center_close(result, expected)


@pytest.mark.parametrize(
    ("agent_center", "agent_radius", "obstacle", "expected"),
    [
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            id="axis-aligned-wall-overlap-projects-right",
        ),
        pytest.param(
            jnp.array([-1.5, 0.0], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            id="axis-aligned-wall-overlap-projects-left",
        ),
        pytest.param(
            jnp.array([0.0, 1.5], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([0.0, 2.0], dtype=jnp.float32),
            id="axis-aligned-wall-overlap-projects-top",
        ),
        pytest.param(
            jnp.array([0.0, -1.5], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([0.0, -2.0], dtype=jnp.float32),
            id="axis-aligned-wall-overlap-projects-bottom",
        ),
        pytest.param(
            jnp.array([2.5, 0.0], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([2.5, 0.0], dtype=jnp.float32),
            id="axis-aligned-wall-separated-unchanged",
        ),
        pytest.param(
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            id="axis-aligned-wall-tangent-unchanged",
        ),
        pytest.param(
            jnp.array([1.6, 1.8], dtype=jnp.float32),
            1.5,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([1.9, 2.2], dtype=jnp.float32),
            id="axis-aligned-wall-corner-overlap-projects-diagonal",
        ),
        pytest.param(
            jnp.array([-0.75, 0.25], dtype=jnp.float32),
            0.5,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([-1.5, 0.25], dtype=jnp.float32),
            id="inside-wall-projects-through-left-face",
        ),
        pytest.param(
            jnp.array([0.75, 0.25], dtype=jnp.float32),
            0.5,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([1.5, 0.25], dtype=jnp.float32),
            id="inside-wall-projects-through-right-face",
        ),
        pytest.param(
            jnp.array([0.2, -0.8], dtype=jnp.float32),
            0.25,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([0.2, -1.25], dtype=jnp.float32),
            id="inside-wall-projects-through-bottom-face",
        ),
        pytest.param(
            jnp.array([0.2, 0.8], dtype=jnp.float32),
            0.25,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([0.2, 1.25], dtype=jnp.float32),
            id="inside-wall-projects-through-top-face",
        ),
        pytest.param(
            jnp.array([1.0, 0.0], dtype=jnp.float32),
            0.5,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            id="boundary-center-projects-through-nearest-face",
        ),
        pytest.param(
            jnp.array([0.0, 2.5], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=4.0,
                height=2.0,
                theta=jnp.pi / 2.0,
            ),
            jnp.array([0.0, 3.0], dtype=jnp.float32),
            id="rotated-wall-overlap-projects-in-world-frame",
        ),
        pytest.param(
            jnp.array([1.0606601, 1.0606601], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=jnp.pi / 4.0,
            ),
            jnp.array([1.4142135, 1.4142135], dtype=jnp.float32),
            id="non-right-angle-wall-projects-along-local-normal",
        ),
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
                active=False,
            ),
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            id="inactive-wall-unchanged",
        ),
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                2.0,
            ),
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            id="active-non-wall-pillar-unchanged",
        ),
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _empty_obstacle(),
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            id="empty-obstacle-row-unchanged",
        ),
    ],
)
def test_project_disc_out_of_wall(
    agent_center: Array,
    agent_radius: Array | float,
    obstacle: Array,
    expected: Array,
) -> None:
    result = _project_disc_out_of_wall(agent_center, agent_radius, obstacle)

    _assert_center_close(result, expected)


def test_projection_helpers_can_be_jit_compiled() -> None:
    pillar = _pillar_obstacle(
        jnp.array([0.0, 0.0], dtype=jnp.float32),
        1.0,
    )
    wall = _wall_obstacle(
        jnp.array([0.0, 0.0], dtype=jnp.float32),
        width=2.0,
        height=2.0,
        theta=jnp.pi / 4.0,
    )

    bounds_result = cast(
        Array,
        jax.jit(_project_disc_to_bounds)(
            jnp.array([-1.0, 5.0], dtype=jnp.float32),
            0.5,
            4.0,
            4.0,
        ),
    )
    pillar_result = cast(
        Array,
        jax.jit(_project_disc_out_of_pillar)(
            jnp.array([1.25, 0.0], dtype=jnp.float32),
            1.0,
            pillar,
        ),
    )
    wall_result = cast(
        Array,
        jax.jit(_project_disc_out_of_wall)(
            jnp.array([1.0606601, 1.0606601], dtype=jnp.float32),
            1.0,
            wall,
        ),
    )

    _assert_center_close(bounds_result, jnp.array([0.5, 3.5], dtype=jnp.float32))
    _assert_center_close(pillar_result, jnp.array([2.0, 0.0], dtype=jnp.float32))
    _assert_center_close(
        wall_result,
        jnp.array([1.4142135, 1.4142135], dtype=jnp.float32),
    )


@pytest.mark.parametrize(
    ("agent_center", "agent_radius", "obstacle", "expected"),
    [
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
            ),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            id="active-wall-row-dispatches-to-wall-projection",
        ),
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                1.0,
            ),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            id="active-pillar-row-dispatches-to-pillar-projection",
        ),
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _empty_obstacle(),
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            id="none-row-leaves-center-unchanged",
        ),
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=2.0,
                height=2.0,
                theta=0.0,
                active=False,
            ),
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            id="inactive-wall-row-leaves-center-unchanged",
        ),
        pytest.param(
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            1.0,
            _pillar_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                1.0,
                active=False,
            ),
            jnp.array([1.5, 0.0], dtype=jnp.float32),
            id="inactive-pillar-row-leaves-center-unchanged",
        ),
    ],
)
def test_project_disc_out_of_obstacle(
    agent_center: Array,
    agent_radius: Array | float,
    obstacle: Array,
    expected: Array,
) -> None:
    center = _project_disc_out_of_obstacle(agent_center, agent_radius, obstacle)
    _assert_center_close(center, expected)


_OBSTACLE_ARRAY_PROJECTION_CASES = [
    pytest.param(
        jnp.array([2.5, 2.5], dtype=jnp.float32),
        0.5,
        jnp.zeros(
            (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
            dtype=jnp.float32,
        ),
        jnp.array([2.5, 2.5], dtype=jnp.float32),
        id="empty-obstacle-array-unchanged",
    ),
    pytest.param(
        jnp.array([1.5, 0.0], dtype=jnp.float32),
        1.0,
        _create_obstacle_array(),
        jnp.array([2.0, 0.0], dtype=jnp.float32),
        id="mixed-sparse-array-late-pillar-projects",
    ),
    pytest.param(
        jnp.array([5.5, 0.0], dtype=jnp.float32),
        1.0,
        _create_obstacle_array(),
        jnp.array([6.0, 0.0], dtype=jnp.float32),
        id="mixed-sparse-array-axis-wall-projects",
    ),
    pytest.param(
        jnp.array([0.0, 4.5], dtype=jnp.float32),
        1.0,
        _create_obstacle_array(),
        jnp.array([0.0, 4.5], dtype=jnp.float32),
        id="mixed-sparse-array-inactive-pillar-ignored",
    ),
    pytest.param(
        jnp.array([4.0, 4.0], dtype=jnp.float32),
        0.5,
        _create_obstacle_array(),
        jnp.array([4.0, 4.0], dtype=jnp.float32),
        id="mixed-sparse-array-inactive-rotated-wall-ignored",
    ),
    pytest.param(
        jnp.array([-4.4, 0.0], dtype=jnp.float32),
        1.0,
        _create_obstacle_array(),
        _project_disc_out_of_wall(
            jnp.array([-4.4, 0.0], dtype=jnp.float32),
            1.0,
            _wall_obstacle(
                jnp.array([-4.0, 0.0], dtype=jnp.float32),
                width=1.0,
                height=3.0,
                theta=jnp.pi / 6.0,
            ),
        ),
        id="mixed-sparse-array-active-rotated-wall-projects",
    ),
    pytest.param(
        jnp.array([1.5, 0.0], dtype=jnp.float32),
        1.0,
        _obstacle_array_with_rows(
            (
                0,
                _pillar_obstacle(
                    jnp.array([0.0, 0.0], dtype=jnp.float32),
                    1.0,
                ),
            ),
            (
                1,
                _pillar_obstacle(
                    jnp.array([3.0, 0.0], dtype=jnp.float32),
                    1.0,
                ),
            ),
        ),
        jnp.array([1.0, 0.0], dtype=jnp.float32),
        id="active-obstacles-project-in-fixed-slot-order",
    ),
]


@pytest.mark.parametrize(
    ("agent_center", "agent_radius", "obstacles", "expected"),
    _OBSTACLE_ARRAY_PROJECTION_CASES,
)
def test_project_disc_out_of_obstacles(
    agent_center: Array,
    agent_radius: Array | float,
    obstacles: Array,
    expected: Array,
) -> None:
    center = _project_disc_out_of_obstacles(
        agent_center,
        agent_radius,
        obstacles,
    )

    _assert_center_close(center, expected)


@pytest.mark.parametrize(
    ("agent_center", "agent_radius", "obstacles", "expected"),
    _OBSTACLE_ARRAY_PROJECTION_CASES,
)
def test_project_disc_out_of_obstacles_jit_compiles(
    agent_center: Array,
    agent_radius: Array | float,
    obstacles: Array,
    expected: Array,
) -> None:
    center = cast(
        Array,
        jax.jit(_project_disc_out_of_obstacles)(
            agent_center,
            agent_radius,
            obstacles,
        ),
    )

    _assert_center_close(center, expected)


@pytest.mark.parametrize(
    ("segment_start", "segment_end", "circle_center", "circle_radius", "expected"),
    [
        pytest.param(
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            1.0,
            True,
            id="segment-crosses-circle",
        ),
        pytest.param(
            jnp.array([-2.0, 2.0], dtype=jnp.float32),
            jnp.array([2.0, 2.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            1.0,
            False,
            id="segment-clear-above-circle",
        ),
        pytest.param(
            jnp.array([-2.0, 1.0], dtype=jnp.float32),
            jnp.array([2.0, 1.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            1.0,
            True,
            id="segment-tangent-to-circle",
        ),
        pytest.param(
            jnp.array([-2.0, 1.0 + GEOMETRY_TOLERANCE / 2.0], dtype=jnp.float32),
            jnp.array([2.0, 1.0 + GEOMETRY_TOLERANCE / 2.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            1.0,
            True,
            id="near-tangent-segment-counts-as-blocked",
        ),
        pytest.param(
            jnp.array([0.5, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            1.0,
            True,
            id="start-endpoint-inside-circle",
        ),
        pytest.param(
            jnp.array([1.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            1.0,
            True,
            id="start-endpoint-tangent-to-circle",
        ),
        pytest.param(
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            jnp.array([4.0, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            1.0,
            False,
            id="projection-before-segment-and-endpoint-clear",
        ),
        pytest.param(
            jnp.array([-4.0, 0.0], dtype=jnp.float32),
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            1.0,
            False,
            id="projection-after-segment-and-endpoint-clear",
        ),
        pytest.param(
            jnp.array([0.5, 0.0], dtype=jnp.float32),
            jnp.array([0.5, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            1.0,
            True,
            id="zero-length-segment-inside-circle",
        ),
        pytest.param(
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            1.0,
            False,
            id="zero-length-segment-outside-circle",
        ),
        pytest.param(
            jnp.array([-2.0, -2.0], dtype=jnp.float32),
            jnp.array([2.0, 2.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            0.5,
            True,
            id="diagonal-segment-crosses-circle",
        ),
    ],
)
def test_segment_intersects_circle(
    segment_start: Array,
    segment_end: Array,
    circle_center: Array,
    circle_radius: Array | float,
    expected: bool,
) -> None:
    result = _segment_intersects_circle(
        segment_start,
        segment_end,
        circle_center,
        circle_radius,
    )

    _assert_scalar_bool(result, expected)


def test_segment_intersects_circle_jit_compiles() -> None:
    result = cast(
        Array,
        jax.jit(_segment_intersects_circle)(
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            1.0,
        ),
    )

    _assert_scalar_bool(result, True)


@pytest.mark.parametrize(
    (
        "segment_start",
        "segment_end",
        "rectangle_center",
        "width",
        "height",
        "theta",
        "expected",
    ),
    [
        pytest.param(
            jnp.array([-2.0, 0.0]),
            jnp.array([2.0, 0.0]),
            jnp.array([0.0, 0.0]),
            2.0,
            2.0,
            0.0,
            True,
            id="axis_aligned_horizontal_crosses_rectangle",
        ),
        pytest.param(
            jnp.array([-2.0, 2.0]),
            jnp.array([2.0, 2.0]),
            jnp.array([0.0, 0.0]),
            2.0,
            2.0,
            0.0,
            False,
            id="axis_aligned_horizontal_clear_miss",
        ),
        pytest.param(
            jnp.array([-2.0, 1.0]),
            jnp.array([2.0, 1.0]),
            jnp.array([0.0, 0.0]),
            2.0,
            2.0,
            0.0,
            True,
            id="axis_aligned_horizontal_touches_top_edge",
        ),
        pytest.param(
            jnp.array([1.0, -2.0]),
            jnp.array([1.0, 2.0]),
            jnp.array([0.0, 0.0]),
            2.0,
            2.0,
            0.0,
            True,
            id="axis_aligned_vertical_touches_right_edge",
        ),
        pytest.param(
            jnp.array([1.0, 1.0]),
            jnp.array([2.0, 2.0]),
            jnp.array([0.0, 0.0]),
            2.0,
            2.0,
            0.0,
            True,
            id="axis_aligned_segment_touches_corner",
        ),
        pytest.param(
            jnp.array([2.0, -2.0]),
            jnp.array([2.0, 2.0]),
            jnp.array([0.0, 0.0]),
            2.0,
            2.0,
            0.0,
            False,
            id="axis_aligned_vertical_clear_miss",
        ),
        pytest.param(
            jnp.array([0.0, 0.0]),
            jnp.array([2.0, 2.0]),
            jnp.array([0.0, 0.0]),
            2.0,
            2.0,
            0.0,
            True,
            id="segment_start_inside_rectangle",
        ),
        pytest.param(
            jnp.array([-2.0, -2.0]),
            jnp.array([0.0, 0.0]),
            jnp.array([0.0, 0.0]),
            2.0,
            2.0,
            0.0,
            True,
            id="segment_end_inside_rectangle",
        ),
        pytest.param(
            jnp.array([0.5, 0.5]),
            jnp.array([0.5, 0.5]),
            jnp.array([0.0, 0.0]),
            2.0,
            2.0,
            0.0,
            True,
            id="zero_length_segment_inside_rectangle",
        ),
        pytest.param(
            jnp.array([2.0, 2.0]),
            jnp.array([2.0, 2.0]),
            jnp.array([0.0, 0.0]),
            2.0,
            2.0,
            0.0,
            False,
            id="zero_length_segment_outside_rectangle",
        ),
        pytest.param(
            jnp.array([-2.0, 0.0]),
            jnp.array([2.0, 0.0]),
            jnp.array([0.0, 0.0]),
            2.0,
            1.0,
            jnp.pi / 4.0,
            True,
            id="rotated_rectangle_crosses_center",
        ),
        pytest.param(
            jnp.array([-2.1213202, -0.70710677]),
            jnp.array([0.70710677, 2.1213202]),
            jnp.array([0.0, 0.0]),
            2.0,
            1.0,
            jnp.pi / 4.0,
            False,
            id="rotated_rectangle_clear_miss",
        ),
    ],
)
def test_segment_intersects_rotated_rect(
    segment_start: Array,
    segment_end: Array,
    rectangle_center: Array,
    width: Array | float,
    height: Array | float,
    theta: Array | float,
    expected: bool,
) -> None:
    result = _segment_intersects_rotated_rect(
        segment_start,
        segment_end,
        rectangle_center,
        width,
        height,
        theta,
    )

    _assert_scalar_bool(result, expected)


def test_segment_intersects_rotated_rect_jit_compiles() -> None:
    result = cast(
        Array,
        jax.jit(_segment_intersects_rotated_rect)(
            jnp.array([-2.0, 1.0]),
            jnp.array([2.0, 1.0]),
            jnp.array([0.0, 0.0]),
            2.0,
            2.0,
            0.0,
        ),
    )

    _assert_scalar_bool(result, True)


@pytest.mark.parametrize(
    ("agent_center_a", "agent_center_b", "obstacles", "expected"),
    [
        pytest.param(
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            _obstacle_array_with_rows(),
            True,
            id="empty_obstacle_array_clear",
        ),
        pytest.param(
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            _obstacle_array_with_rows(
                (
                    0,
                    _pillar_obstacle(
                        jnp.array([0.0, 0.0], dtype=jnp.float32),
                        1.0,
                        active=False,
                    ),
                ),
            ),
            True,
            id="inactive_pillar_on_los_ignored",
        ),
        pytest.param(
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            _obstacle_array_with_rows(
                (
                    0,
                    _wall_obstacle(
                        jnp.array([0.0, 0.0], dtype=jnp.float32),
                        width=2.0,
                        height=2.0,
                        theta=0.0,
                        active=False,
                    ),
                ),
            ),
            True,
            id="inactive_wall_on_los_ignored",
        ),
        pytest.param(
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            _obstacle_array_with_rows(
                (0, _active_none_obstacle_with_geometry()),
            ),
            True,
            id="active_none_obstacle_with_geometry_ignored",
        ),
        pytest.param(
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            _obstacle_array_with_rows(
                (
                    0,
                    _pillar_obstacle(
                        jnp.array([0.0, 0.0], dtype=jnp.float32),
                        0.5,
                    ),
                ),
            ),
            False,
            id="active_pillar_blocks",
        ),
        pytest.param(
            jnp.array([-2.0, 1.0], dtype=jnp.float32),
            jnp.array([2.0, 1.0], dtype=jnp.float32),
            _obstacle_array_with_rows(
                (
                    0,
                    _pillar_obstacle(
                        jnp.array([0.0, 0.0], dtype=jnp.float32),
                        1.0,
                    ),
                ),
            ),
            False,
            id="active_pillar_tangent_blocks",
        ),
        pytest.param(
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            _obstacle_array_with_rows(
                (
                    0,
                    _pillar_obstacle(
                        jnp.array([0.0, 0.0], dtype=jnp.float32),
                        0.5,
                    ),
                ),
            ),
            False,
            id="zero_length_segment_inside_active_pillar_blocks",
        ),
        pytest.param(
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            _obstacle_array_with_rows(
                (
                    0,
                    _wall_obstacle(
                        jnp.array([0.0, 0.0], dtype=jnp.float32),
                        width=1.0,
                        height=1.0,
                        theta=0.0,
                    ),
                ),
            ),
            False,
            id="active_axis_aligned_wall_blocks",
        ),
        pytest.param(
            jnp.array([-2.0, 0.5], dtype=jnp.float32),
            jnp.array([2.0, 0.5], dtype=jnp.float32),
            _obstacle_array_with_rows(
                (
                    0,
                    _wall_obstacle(
                        jnp.array([0.0, 0.0], dtype=jnp.float32),
                        width=1.0,
                        height=1.0,
                        theta=0.0,
                    ),
                ),
            ),
            False,
            id="active_wall_edge_touch_blocks",
        ),
        pytest.param(
            jnp.array([3.0, 3.0], dtype=jnp.float32),
            jnp.array([3.0, 3.0], dtype=jnp.float32),
            _obstacle_array_with_rows(
                (
                    0,
                    _wall_obstacle(
                        jnp.array([0.0, 0.0], dtype=jnp.float32),
                        width=1.0,
                        height=1.0,
                        theta=0.0,
                    ),
                ),
            ),
            True,
            id="zero_length_segment_outside_active_wall_clear",
        ),
        pytest.param(
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            _obstacle_array_with_rows(
                (
                    0,
                    _wall_obstacle(
                        jnp.array([0.0, 0.0], dtype=jnp.float32),
                        width=2.0,
                        height=1.0,
                        theta=jnp.pi / 4.0,
                    ),
                ),
            ),
            False,
            id="active_rotated_wall_blocks",
        ),
        pytest.param(
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            _obstacle_array_with_rows(
                (
                    0,
                    _pillar_obstacle(
                        jnp.array([0.0, 3.0], dtype=jnp.float32),
                        0.5,
                    ),
                ),
                (
                    1,
                    _wall_obstacle(
                        jnp.array([0.0, -3.0], dtype=jnp.float32),
                        width=1.0,
                        height=1.0,
                        theta=0.0,
                    ),
                ),
            ),
            True,
            id="active_obstacles_clear_miss",
        ),
        pytest.param(
            jnp.array([-2.0, 0.0], dtype=jnp.float32),
            jnp.array([2.0, 0.0], dtype=jnp.float32),
            _obstacle_array_with_rows(
                (
                    MAX_OBSTACLE_SLOTS - 1,
                    _pillar_obstacle(
                        jnp.array([0.0, 0.0], dtype=jnp.float32),
                        0.5,
                    ),
                ),
            ),
            False,
            id="late_active_slot_blocks",
        ),
    ],
)
def test_has_clear_line_of_sight(
    agent_center_a: Array,
    agent_center_b: Array,
    obstacles: Array,
    expected: bool,
) -> None:
    result = has_clear_line_of_sight(
        agent_center_a,
        agent_center_b,
        obstacles,
    )

    _assert_scalar_bool(result, expected)


def test_has_clear_line_of_sight_jit_compiles() -> None:
    agent_center_a = jnp.array([-2.0, 0.0], dtype=jnp.float32)
    agent_center_b = jnp.array([2.0, 0.0], dtype=jnp.float32)
    obstacles = _obstacle_array_with_rows(
        (
            0,
            _pillar_obstacle(
                jnp.array([0.0, 3.0], dtype=jnp.float32),
                0.5,
            ),
        ),
        (
            1,
            _wall_obstacle(
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                width=1.0,
                height=1.0,
                theta=0.0,
            ),
        ),
    )
    expected = False

    result = cast(
        Array,
        jax.jit(has_clear_line_of_sight)(
            agent_center_a,
            agent_center_b,
            obstacles,
        ),
    )

    _assert_scalar_bool(result, expected)


@pytest.mark.parametrize(
    (
        "agent_positions",
        "agent_radii",
        "active_mask",
        "alive_mask",
        "projection_passes",
        "expected",
    ),
    [
        pytest.param(
            _agent_positions_array_with_rows(),
            _agent_radii_array_with_rows(),
            _mask_with_true_slots(),
            _mask_with_true_slots(),
            1,
            _agent_positions_array_with_rows(),
            id="empty_agent_slots_unchanged",
        ),
        pytest.param(
            _agent_positions_array_with_rows(
                (0, jnp.array([-1.0, 0.0], dtype=jnp.float32)),
                (1, jnp.array([1.0, 0.0], dtype=jnp.float32)),
            ),
            _agent_radii_array_with_rows((0, 0.5), (1, 0.5)),
            _mask_with_true_slots(0, 1),
            _mask_with_true_slots(0, 1),
            1,
            _agent_positions_array_with_rows(
                (0, jnp.array([-1.0, 0.0], dtype=jnp.float32)),
                (1, jnp.array([1.0, 0.0], dtype=jnp.float32)),
            ),
            id="active_alive_non_overlapping_agents_unchanged",
        ),
        pytest.param(
            _agent_positions_array_with_rows(
                (0, jnp.array([-0.25, 0.0], dtype=jnp.float32)),
                (1, jnp.array([0.25, 0.0], dtype=jnp.float32)),
            ),
            _agent_radii_array_with_rows((0, 0.5), (1, 0.5)),
            _mask_with_true_slots(0, 1),
            _mask_with_true_slots(0, 1),
            1,
            _agent_positions_array_with_rows(
                (0, jnp.array([-0.5, 0.0], dtype=jnp.float32)),
                (1, jnp.array([0.5, 0.0], dtype=jnp.float32)),
            ),
            id="active_alive_overlapping_agents_separated",
        ),
        pytest.param(
            _agent_positions_array_with_rows(
                (0, jnp.array([-0.25, 0.0], dtype=jnp.float32)),
                (1, jnp.array([0.25, 0.0], dtype=jnp.float32)),
            ),
            _agent_radii_array_with_rows((0, 0.5), (1, 0.5)),
            _mask_with_true_slots(0),
            _mask_with_true_slots(0, 1),
            1,
            _agent_positions_array_with_rows(
                (0, jnp.array([-0.25, 0.0], dtype=jnp.float32)),
                (1, jnp.array([0.25, 0.0], dtype=jnp.float32)),
            ),
            id="inactive_agent_does_not_push_or_get_pushed",
        ),
        pytest.param(
            _agent_positions_array_with_rows(
                (0, jnp.array([-0.25, 0.0], dtype=jnp.float32)),
                (1, jnp.array([0.25, 0.0], dtype=jnp.float32)),
            ),
            _agent_radii_array_with_rows((0, 0.5), (1, 0.5)),
            _mask_with_true_slots(0, 1),
            _mask_with_true_slots(0),
            1,
            _agent_positions_array_with_rows(
                (0, jnp.array([-0.25, 0.0], dtype=jnp.float32)),
                (1, jnp.array([0.25, 0.0], dtype=jnp.float32)),
            ),
            id="dead_agent_does_not_push_or_get_pushed",
        ),
        pytest.param(
            _agent_positions_array_with_rows(
                (0, jnp.array([0.0, 0.0], dtype=jnp.float32)),
                (1, jnp.array([0.0, 0.0], dtype=jnp.float32)),
            ),
            _agent_radii_array_with_rows((0, 0.5), (1, 0.5)),
            _mask_with_true_slots(0, 1),
            _mask_with_true_slots(0, 1),
            1,
            _agent_positions_array_with_rows(
                (0, jnp.array([0.5, 0.0], dtype=jnp.float32)),
                (1, jnp.array([-0.5, 0.0], dtype=jnp.float32)),
            ),
            id="coincident_centers_use_deterministic_fallback",
        ),
    ],
)
def test_resolve_agent_agent_overlaps(
    agent_positions: Array,
    agent_radii: Array,
    active_mask: Array,
    alive_mask: Array,
    projection_passes: int,
    expected: Array,
) -> None:
    result = _resolve_agent_agent_overlaps(
        agent_positions,
        agent_radii,
        active_mask,
        alive_mask,
        projection_passes,
    )

    _assert_agent_positions_close(result, expected)


def test_resolve_agent_agent_overlaps_multi_agent_chain_improves() -> None:
    agent_positions = _agent_positions_array_with_rows(
        (0, jnp.array([-0.4, 0.0], dtype=jnp.float32)),
        (1, jnp.array([0.0, 0.0], dtype=jnp.float32)),
        (2, jnp.array([0.4, 0.0], dtype=jnp.float32)),
    )
    agent_radii = _agent_radii_array_with_rows((0, 0.3), (1, 0.3), (2, 0.3))
    active_mask = _mask_with_true_slots(0, 1, 2)
    alive_mask = _mask_with_true_slots(0, 1, 2)

    result = _resolve_agent_agent_overlaps(
        agent_positions,
        agent_radii,
        active_mask,
        alive_mask,
        projection_passes=8,
    )

    assert result.shape == (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS)
    assert result.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(result)))

    distance_01 = cast(Array, jnp.linalg.norm(result[0] - result[1]))
    distance_12 = cast(Array, jnp.linalg.norm(result[1] - result[2]))

    assert bool(distance_01 >= 0.6 - GEOMETRY_TOLERANCE)
    assert bool(distance_12 >= 0.6 - GEOMETRY_TOLERANCE)


def test_resolve_agent_agent_overlaps_jit_compiles() -> None:
    agent_positions = _agent_positions_array_with_rows(
        (0, jnp.array([-0.25, 0.0], dtype=jnp.float32)),
        (1, jnp.array([0.25, 0.0], dtype=jnp.float32)),
    )
    agent_radii = _agent_radii_array_with_rows((0, 0.5), (1, 0.5))
    active_mask = _mask_with_true_slots(0, 1)
    alive_mask = _mask_with_true_slots(0, 1)

    result = cast(
        Array,
        jax.jit(_resolve_agent_agent_overlaps, static_argnames=("projection_passes",))(
            agent_positions,
            agent_radii,
            active_mask,
            alive_mask,
            projection_passes=1,
        ),
    )

    expected = _agent_positions_array_with_rows(
        (0, jnp.array([-0.5, 0.0], dtype=jnp.float32)),
        (1, jnp.array([0.5, 0.0], dtype=jnp.float32)),
    )

    _assert_agent_positions_close(result, expected)


@pytest.mark.parametrize(
    (
        "intended_movement_deltas",
        "agent_positions",
        "agent_radii",
        "active_mask",
        "alive_mask",
        "map_width",
        "map_height",
        "obstacles",
        "expected",
        "agent_agent_overlap_projection_passes",
        "collision_projection_passes",
        "movement_substeps",
    ),
    [
        pytest.param(
            _movement_deltas_array_with_rows(),
            _agent_positions_array_with_rows(
                (0, jnp.array([5.0, 5.0], dtype=jnp.float32)),
            ),
            _agent_radii_array_with_rows((0, 0.5)),
            _mask_with_true_slots(0),
            _mask_with_true_slots(0),
            20.0,
            20.0,
            _obstacle_array_with_rows(),
            _agent_positions_array_with_rows(
                (0, jnp.array([5.0, 5.0], dtype=jnp.float32)),
            ),
            1,
            1,
            1,
            id="zero_movement_preserves_valid_active_alive_position",
        ),
        pytest.param(
            _movement_deltas_array_with_rows(
                (0, jnp.array([4.0, -1.0], dtype=jnp.float32)),
            ),
            _agent_positions_array_with_rows(
                (0, jnp.array([5.0, 5.0], dtype=jnp.float32)),
            ),
            _agent_radii_array_with_rows((0, 0.5)),
            _mask_with_true_slots(0),
            _mask_with_true_slots(0),
            20.0,
            20.0,
            _obstacle_array_with_rows(),
            _agent_positions_array_with_rows(
                (0, jnp.array([9.0, 4.0], dtype=jnp.float32)),
            ),
            1,
            1,
            4,
            id="active_alive_agent_moves_by_full_delta_across_substeps",
        ),
        pytest.param(
            _movement_deltas_array_with_rows(
                (0, jnp.array([10.0, 0.0], dtype=jnp.float32)),
                (1, jnp.array([0.0, 10.0], dtype=jnp.float32)),
            ),
            _agent_positions_array_with_rows(
                (0, jnp.array([5.0, 5.0], dtype=jnp.float32)),
                (1, jnp.array([6.0, 6.0], dtype=jnp.float32)),
            ),
            _agent_radii_array_with_rows((0, 0.5), (1, 0.5)),
            _mask_with_true_slots(1),
            _mask_with_true_slots(0),
            20.0,
            20.0,
            _obstacle_array_with_rows(),
            _agent_positions_array_with_rows(
                (0, jnp.array([5.0, 5.0], dtype=jnp.float32)),
                (1, jnp.array([6.0, 6.0], dtype=jnp.float32)),
            ),
            1,
            1,
            4,
            id="inactive_and_dead_slots_preserve_original_positions",
        ),
        pytest.param(
            _movement_deltas_array_with_rows(
                (0, jnp.array([5.0, 0.0], dtype=jnp.float32)),
            ),
            _agent_positions_array_with_rows(
                (0, jnp.array([19.0, 10.0], dtype=jnp.float32)),
            ),
            _agent_radii_array_with_rows((0, 0.5)),
            _mask_with_true_slots(0),
            _mask_with_true_slots(0),
            20.0,
            20.0,
            _obstacle_array_with_rows(),
            _agent_positions_array_with_rows(
                (0, jnp.array([19.5, 10.0], dtype=jnp.float32)),
            ),
            1,
            1,
            1,
            id="movement_past_right_boundary_is_projected_inside",
        ),
        pytest.param(
            _movement_deltas_array_with_rows(
                (0, jnp.array([1.0, 0.0], dtype=jnp.float32)),
            ),
            _agent_positions_array_with_rows(
                (0, jnp.array([8.0, 10.0], dtype=jnp.float32)),
            ),
            _agent_radii_array_with_rows((0, 0.5)),
            _mask_with_true_slots(0),
            _mask_with_true_slots(0),
            20.0,
            20.0,
            _obstacle_array_with_rows(
                (
                    0,
                    _pillar_obstacle(
                        jnp.array([10.0, 10.0], dtype=jnp.float32),
                        1.0,
                    ),
                ),
            ),
            _agent_positions_array_with_rows(
                (0, jnp.array([8.5, 10.0], dtype=jnp.float32)),
            ),
            1,
            1,
            1,
            id="movement_into_pillar_is_projected_out",
        ),
        pytest.param(
            _movement_deltas_array_with_rows(
                (1, jnp.array([-1.5, 0.0], dtype=jnp.float32)),
            ),
            _agent_positions_array_with_rows(
                (0, jnp.array([5.0, 5.0], dtype=jnp.float32)),
                (1, jnp.array([7.0, 5.0], dtype=jnp.float32)),
            ),
            _agent_radii_array_with_rows((0, 0.5), (1, 0.5)),
            _mask_with_true_slots(0, 1),
            _mask_with_true_slots(0, 1),
            20.0,
            20.0,
            _obstacle_array_with_rows(),
            _agent_positions_array_with_rows(
                (0, jnp.array([4.75, 5.0], dtype=jnp.float32)),
                (1, jnp.array([5.75, 5.0], dtype=jnp.float32)),
            ),
            1,
            1,
            1,
            id="movement_into_active_alive_agent_resolves_overlap",
        ),
    ],
)
def test_project_movement_with_geometry(
    intended_movement_deltas: Array,
    agent_positions: Array,
    agent_radii: Array,
    active_mask: Array,
    alive_mask: Array,
    map_width: Array | float,
    map_height: Array | float,
    obstacles: Array,
    expected: Array,
    agent_agent_overlap_projection_passes: int,
    collision_projection_passes: int,
    movement_substeps: int,
) -> None:
    result = project_movement_with_geometry(
        agent_positions,
        agent_radii,
        intended_movement_deltas,
        active_mask,
        alive_mask,
        map_width,
        map_height,
        obstacles,
        agent_agent_overlap_projection_passes,
        collision_projection_passes,
        movement_substeps,
    )

    _assert_agent_positions_close(result, expected)


def test_project_movement_with_geometry_projects_agent_out_of_wall() -> None:
    wall = _wall_obstacle(
        jnp.array([10.0, 10.0], dtype=jnp.float32),
        width=2.0,
        height=2.0,
        theta=0.0,
    )

    intended_movement_deltas = _movement_deltas_array_with_rows(
        (0, jnp.array([2.25, 0.0], dtype=jnp.float32)),
    )
    agent_positions = _agent_positions_array_with_rows(
        (0, jnp.array([7.0, 10.0], dtype=jnp.float32)),
    )
    agent_radii = _agent_radii_array_with_rows((0, 0.5))
    active_mask = _mask_with_true_slots(0)
    alive_mask = _mask_with_true_slots(0)

    result = project_movement_with_geometry(
        agent_positions,
        agent_radii,
        intended_movement_deltas,
        active_mask,
        alive_mask,
        20.0,
        20.0,
        _obstacle_array_with_rows((0, wall)),
        1,
        1,
        1,
    )

    expected = _agent_positions_array_with_rows(
        (0, jnp.array([8.5, 10.0], dtype=jnp.float32)),
    )

    _assert_agent_positions_close(result, expected)


def test_project_movement_keeps_static_validity_with_bounded_agent_residual() -> None:
    intended_movement_deltas = _movement_deltas_array_with_rows(
        (1, jnp.array([-1.0, 0.0], dtype=jnp.float32)),
    )
    agent_positions = _agent_positions_array_with_rows(
        (0, jnp.array([0.5, 10.0], dtype=jnp.float32)),
        (1, jnp.array([2.0, 10.0], dtype=jnp.float32)),
    )
    agent_radii = _agent_radii_array_with_rows((0, 0.5), (1, 0.5))
    active_mask = _mask_with_true_slots(0, 1)
    alive_mask = _mask_with_true_slots(0, 1)

    result = project_movement_with_geometry(
        agent_positions,
        agent_radii,
        intended_movement_deltas,
        active_mask,
        alive_mask,
        20.0,
        20.0,
        _obstacle_array_with_rows(),
        1,
        1,
        1,
    )

    _assert_agent_positions_are_finite(result)
    _assert_active_alive_agents_inside_bounds(
        result,
        agent_radii,
        active_mask,
        alive_mask,
        20.0,
        20.0,
    )

    residual = _max_active_alive_agent_overlap_residual(
        result,
        agent_radii,
        active_mask,
        alive_mask,
    )

    # This is the boundary-pinned ordering hazard: final bounds cleanup keeps the
    # agents static-valid, but it can reintroduce a small body-blocking residual.
    assert residual > 0.0
    assert residual <= 0.25 + GEOMETRY_TOLERANCE


def test_project_movement_more_collision_passes_reduce_boundary_residual() -> None:
    intended_movement_deltas = _movement_deltas_array_with_rows(
        (1, jnp.array([-1.0, 0.0], dtype=jnp.float32)),
    )
    agent_positions = _agent_positions_array_with_rows(
        (0, jnp.array([0.5, 10.0], dtype=jnp.float32)),
        (1, jnp.array([2.0, 10.0], dtype=jnp.float32)),
    )
    agent_radii = _agent_radii_array_with_rows((0, 0.5), (1, 0.5))
    active_mask = _mask_with_true_slots(0, 1)
    alive_mask = _mask_with_true_slots(0, 1)
    obstacles = _obstacle_array_with_rows()

    one_pass_result = project_movement_with_geometry(
        agent_positions,
        agent_radii,
        intended_movement_deltas,
        active_mask,
        alive_mask,
        20.0,
        20.0,
        obstacles,
        1,
        1,
        1,
    )
    four_pass_result = project_movement_with_geometry(
        agent_positions,
        agent_radii,
        intended_movement_deltas,
        active_mask,
        alive_mask,
        20.0,
        20.0,
        obstacles,
        1,
        4,
        1,
    )

    _assert_agent_positions_are_finite(one_pass_result)
    _assert_agent_positions_are_finite(four_pass_result)
    _assert_active_alive_agents_inside_bounds(
        one_pass_result,
        agent_radii,
        active_mask,
        alive_mask,
        20.0,
        20.0,
    )
    _assert_active_alive_agents_inside_bounds(
        four_pass_result,
        agent_radii,
        active_mask,
        alive_mask,
        20.0,
        20.0,
    )

    one_pass_residual = _max_active_alive_agent_overlap_residual(
        one_pass_result,
        agent_radii,
        active_mask,
        alive_mask,
    )
    four_pass_residual = _max_active_alive_agent_overlap_residual(
        four_pass_result,
        agent_radii,
        active_mask,
        alive_mask,
    )

    # Extra fixed passes should improve the residual in this pressure case even
    # though exact separation is not the hard final invariant.
    assert one_pass_residual > 0.0
    assert four_pass_residual < one_pass_residual


def test_project_movement_overconstrained_crowd_stays_finite_and_static_valid() -> None:
    intended_movement_deltas = _movement_deltas_array_with_rows()
    agent_positions = _agent_positions_array_with_rows(
        (0, jnp.array([0.5, 0.5], dtype=jnp.float32)),
        (1, jnp.array([0.5, 0.5], dtype=jnp.float32)),
        (2, jnp.array([0.5, 0.5], dtype=jnp.float32)),
    )
    agent_radii = _agent_radii_array_with_rows((0, 0.5), (1, 0.5), (2, 0.5))
    active_mask = _mask_with_true_slots(0, 1, 2)
    alive_mask = _mask_with_true_slots(0, 1, 2)

    result = project_movement_with_geometry(
        agent_positions,
        agent_radii,
        intended_movement_deltas,
        active_mask,
        alive_mask,
        1.0,
        1.0,
        _obstacle_array_with_rows(),
        1,
        4,
        1,
    )

    _assert_agent_positions_are_finite(result)
    _assert_active_alive_agents_inside_bounds(
        result,
        agent_radii,
        active_mask,
        alive_mask,
        1.0,
        1.0,
    )

    residual = _max_active_alive_agent_overlap_residual(
        result,
        agent_radii,
        active_mask,
        alive_mask,
    )

    # The solver must stay finite and static-valid even when the map cannot fit
    # every active body without overlap.
    assert residual > 0.0
    assert residual <= 1.0 + GEOMETRY_TOLERANCE


def test_project_movement_with_geometry_jit_compiles_with_static_args() -> None:
    intended_movement_deltas = _movement_deltas_array_with_rows(
        (0, jnp.array([4.0, -1.0], dtype=jnp.float32)),
    )
    agent_positions = _agent_positions_array_with_rows(
        (0, jnp.array([5.0, 5.0], dtype=jnp.float32)),
    )
    agent_radii = _agent_radii_array_with_rows((0, 0.5))
    active_mask = _mask_with_true_slots(0)
    alive_mask = _mask_with_true_slots(0)
    obstacles = _obstacle_array_with_rows()

    compiled_project_movement = jax.jit(
        project_movement_with_geometry,
        static_argnames=(
            "agent_agent_overlap_projection_passes",
            "collision_projection_passes",
            "movement_substeps",
        ),
    )

    result = cast(
        Array,
        compiled_project_movement(
            agent_positions,
            agent_radii,
            intended_movement_deltas,
            active_mask,
            alive_mask,
            20.0,
            20.0,
            obstacles,
            agent_agent_overlap_projection_passes=1,
            collision_projection_passes=1,
            movement_substeps=4,
        ),
    )

    expected = _agent_positions_array_with_rows(
        (0, jnp.array([9.0, 4.0], dtype=jnp.float32)),
    )

    _assert_agent_positions_close(result, expected)
