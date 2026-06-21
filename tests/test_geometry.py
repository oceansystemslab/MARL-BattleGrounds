"""Tests for shared simulator geometry helpers."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.geometry import (
    GEOMETRY_TOLERANCE,
    project_disc_out_of_pillar,
    project_disc_out_of_wall,
    project_disc_to_bounds,
)
from marl_battlegrounds.core.types import (
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
)

# Test helpers ---


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
    """Geometry projection helpers should remain usable inside JIT transitions."""
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
