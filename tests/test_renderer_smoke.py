"""Smoke tests for optional geometry rendering helpers."""

import sys
from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec
from typing import TypedDict, cast

import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.types import (
    CLASS_NEUTRAL,
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
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
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.rendering.geometry import (
    RenderResult,
    redraw_geometry,
    render_geometry,
)


class _CombatStateFields(TypedDict):
    """Keyword fields for inert combat state in test EnvState constructors."""

    current_health: Array
    max_health: Array
    ultimate_cooldowns: Array
    slow_multipliers: Array
    slow_durations: Array
    stun_durations: Array
    anti_heal_multipliers: Array
    anti_heal_durations: Array
    damage_amplification_multipliers: Array
    damage_amplification_durations: Array
    blessing_of_freedom_durations: Array


def _inert_combat_state_fields() -> _CombatStateFields:
    """Return neutral combat fields for direct EnvState constructors."""
    return {
        "current_health": jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32),
        "max_health": jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32),
        "ultimate_cooldowns": jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        "slow_multipliers": jnp.ones(
            (MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.float32
        ),
        "slow_durations": jnp.zeros(
            (MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32
        ),
        "stun_durations": jnp.zeros(
            (MAX_AGENT_SLOTS, NUM_STUN_CHANNELS), dtype=jnp.int32
        ),
        "anti_heal_multipliers": jnp.ones((MAX_AGENT_SLOTS,), dtype=jnp.float32),
        "anti_heal_durations": jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        "damage_amplification_multipliers": jnp.ones(
            (MAX_AGENT_SLOTS,), dtype=jnp.float32
        ),
        "damage_amplification_durations": jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        "blessing_of_freedom_durations": jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    }


def _skip_if_matplotlib_unavailable() -> None:
    """Skip optional rendering checks when Matplotlib is unavailable."""
    try:
        has_matplotlib = find_spec("matplotlib") is not None
        has_pyplot = has_matplotlib and find_spec("matplotlib.pyplot") is not None
    except ModuleNotFoundError:
        has_pyplot = False

    if not has_pyplot:
        pytest.skip("matplotlib is not installed")


def _close_render_result(result: RenderResult) -> None:
    """Close a Matplotlib figure created by renderer smoke tests."""
    pyplot = import_module("matplotlib.pyplot")
    close_figure = cast(Callable[[object], object], pyplot.close)
    close_figure(result.figure)


def _empty_obstacles() -> Array:
    """Create a padded all-inactive obstacle table."""
    return jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _pillar_obstacle() -> Array:
    """Create one active pillar obstacle row."""
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(4.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(3.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_RADIUS].set(0.75)
    obstacle = obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)
    return obstacle


def _wall_obstacle() -> Array:
    """Create one active rotated wall obstacle row."""
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_WALL)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(8.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(5.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_WIDTH].set(2.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_HEIGHT].set(0.5)
    obstacle = obstacle.at[OBSTACLE_FEATURE_THETA].set(0.5)
    obstacle = obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)
    return obstacle


def _sample_config() -> EnvConfig:
    """Create a renderer smoke-test config with mixed active obstacles."""
    obstacles = _empty_obstacles()
    obstacles = obstacles.at[0].set(_pillar_obstacle())
    obstacles = obstacles.at[1].set(_wall_obstacle())

    return EnvConfig(
        team_size=2,
        max_steps=100,
        map_width=12.0,
        map_height=8.0,
        obstacles=obstacles,
        initial_class_ids=jnp.full((MAX_AGENT_SLOTS,), CLASS_NEUTRAL, dtype=jnp.int32),
    )


def _sample_state() -> EnvState:
    """Create a renderer smoke-test state with active slots on both teams."""
    positions = jnp.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
    positions = positions.at[0].set(jnp.array((2.0, 2.0), dtype=jnp.float32))
    positions = positions.at[1].set(jnp.array((3.0, 2.5), dtype=jnp.float32))
    positions = positions.at[MAX_AGENTS_PER_TEAM].set(
        jnp.array((10.0, 6.0), dtype=jnp.float32)
    )
    positions = positions.at[MAX_AGENTS_PER_TEAM + 1].set(
        jnp.array((9.0, 5.5), dtype=jnp.float32)
    )

    active_mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)
    active_mask = active_mask.at[0].set(True)
    active_mask = active_mask.at[1].set(True)
    active_mask = active_mask.at[MAX_AGENTS_PER_TEAM].set(True)
    active_mask = active_mask.at[MAX_AGENTS_PER_TEAM + 1].set(True)

    alive_mask = active_mask.at[MAX_AGENTS_PER_TEAM + 1].set(False)

    team_ids = jnp.concatenate(
        (
            jnp.zeros((MAX_AGENTS_PER_TEAM,), dtype=jnp.int32),
            jnp.ones((MAX_AGENTS_PER_TEAM,), dtype=jnp.int32),
        ),
        axis=0,
    )

    return EnvState(
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=positions,
        agent_radii=jnp.full((MAX_AGENT_SLOTS,), 0.5, dtype=jnp.float32),
        team_ids=team_ids,
        class_ids=jnp.full((MAX_AGENT_SLOTS,), CLASS_NEUTRAL, dtype=jnp.int32),
        movement_speeds=jnp.full((MAX_AGENT_SLOTS,), 1.0, dtype=jnp.float32),
        observation_radii=jnp.full((MAX_AGENT_SLOTS,), 8.0, dtype=jnp.float32),
        basic_interaction_radii=jnp.full((MAX_AGENT_SLOTS,), 6.0, dtype=jnp.float32),
        ultimate_interaction_radii=jnp.full((MAX_AGENT_SLOTS,), 9.0, dtype=jnp.float32),
        active_mask=active_mask,
        alive_mask=alive_mask,
        **_inert_combat_state_fields(),
    )


def _sample_state_with_shifted_first_agent() -> EnvState:
    """Create a second renderer state for redraw smoke coverage."""
    state = _sample_state()
    shifted_positions = state.agent_positions.at[0].set(
        jnp.array((4.0, 4.0), dtype=jnp.float32)
    )

    return state._replace(agent_positions=shifted_positions)


def test_rendering_package_imports_without_visualization_dependency() -> None:
    """The rendering package should import without eagerly importing Matplotlib."""
    rendering_package = import_module("marl_battlegrounds.rendering")
    geometry_module = import_module("marl_battlegrounds.rendering.geometry")

    assert rendering_package.__name__ == "marl_battlegrounds.rendering"
    assert geometry_module.__name__ == "marl_battlegrounds.rendering.geometry"


def test_matplotlib_skip_helper_handles_missing_parent_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Renderer smoke tests should skip cleanly when Matplotlib is absent."""

    def fake_find_spec(name: str) -> object | None:
        assert name == "matplotlib"
        return None

    monkeypatch.setattr(sys.modules[__name__], "find_spec", fake_find_spec)

    with pytest.raises(pytest.skip.Exception):
        _skip_if_matplotlib_unavailable()


def test_render_geometry_constructs_figure_when_matplotlib_is_available() -> None:
    """The optional renderer should construct a render result or skip cleanly."""
    _skip_if_matplotlib_unavailable()

    result = render_geometry(_sample_config(), _sample_state())

    try:
        assert isinstance(result, RenderResult)
        assert hasattr(result.figure, "savefig")
        assert hasattr(result.axes, "clear")
    finally:
        _close_render_result(result)


def test_redraw_geometry_reuses_existing_render_result() -> None:
    """Redraw should reuse the figure/axes pair instead of creating a new one."""
    _skip_if_matplotlib_unavailable()

    result = render_geometry(_sample_config(), _sample_state())

    try:
        redrawn = redraw_geometry(
            _sample_config(),
            _sample_state_with_shifted_first_agent(),
            result,
            show_agent_indices=False,
        )

        assert redrawn is result
        assert redrawn.figure is result.figure
        assert redrawn.axes is result.axes
    finally:
        _close_render_result(result)
