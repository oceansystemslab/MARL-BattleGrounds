"""Tests for shared simulator geometry helpers."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.geometry import (
    GEOMETRY_TOLERANCE,
    has_clear_line_of_sight,
    project_disc_out_of_obstacle,
    project_disc_out_of_obstacles,
    project_disc_out_of_pillar,
    project_disc_out_of_wall,
    project_disc_to_bounds,
    segment_intersects_circle,
    segment_intersects_rotated_rect,
)
from marl_battlegrounds.core.types import (
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
    result = project_disc_to_bounds(agent_center, agent_radius, map_width, map_height)

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
    result = project_disc_out_of_pillar(agent_center, agent_radius, obstacle)

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
    result = project_disc_out_of_wall(agent_center, agent_radius, obstacle)

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
        jax.jit(project_disc_to_bounds)(
            jnp.array([-1.0, 5.0], dtype=jnp.float32),
            0.5,
            4.0,
            4.0,
        ),
    )
    pillar_result = cast(
        Array,
        jax.jit(project_disc_out_of_pillar)(
            jnp.array([1.25, 0.0], dtype=jnp.float32),
            1.0,
            pillar,
        ),
    )
    wall_result = cast(
        Array,
        jax.jit(project_disc_out_of_wall)(
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
    center = project_disc_out_of_obstacle(agent_center, agent_radius, obstacle)
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
        project_disc_out_of_wall(
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
    center = project_disc_out_of_obstacles(
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
        jax.jit(project_disc_out_of_obstacles)(
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
    result = segment_intersects_circle(
        segment_start,
        segment_end,
        circle_center,
        circle_radius,
    )

    _assert_scalar_bool(result, expected)


def test_segment_intersects_circle_jit_compiles() -> None:
    result = cast(
        Array,
        jax.jit(segment_intersects_circle)(
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
    result = segment_intersects_rotated_rect(
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
        jax.jit(segment_intersects_rotated_rect)(
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
