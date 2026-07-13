"""Minimal read-only renderer for Milestone 4 simulator geometry."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from marl_battlegrounds.core.types import (
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
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    TEAM_A_ID,
    EnvConfig,
    EnvState,
)


class _AxesLike(Protocol):
    """Small subset of the Matplotlib axes API used by the renderer."""

    def add_patch(self, patch: object) -> object: ...

    def clear(self) -> object: ...

    def text(self, x: float, y: float, s: str, **kwargs: object) -> object: ...

    def set_aspect(self, aspect: str, adjustable: str | None = None) -> object: ...

    def set_xlim(self, left: float, right: float) -> object: ...

    def set_ylim(self, bottom: float, top: float) -> object: ...

    def set_xlabel(self, xlabel: str) -> object: ...

    def set_ylabel(self, ylabel: str) -> object: ...

    def set_title(self, label: str) -> object: ...


class _PyplotLike(Protocol):
    """Small subset of the Matplotlib pyplot API used by the renderer."""

    def subplots(self) -> tuple[object, _AxesLike]: ...


_PatchFactory = Callable[..., object]
_ObstacleRow = npt.NDArray[np.float32]


@dataclass(frozen=True)
class RenderResult:
    """Matplotlib objects created by geometry rendering helpers.

    The fields stay typed as ``object`` so importing this module does not require
    Matplotlib type imports.
    """

    figure: object
    axes: object


def render_geometry(
    config: EnvConfig,
    state: EnvState,
    *,
    show_agent_indices: bool = True,
) -> RenderResult:
    """Render map boundaries, active agents, and active static obstacles.

    The renderer consumes simulator config/state read-only. It is a visual
    inspection helper, not a source of collision, line-of-sight, observation, or
    action-mask semantics.

    Args:
        config: Static episode configuration containing map and obstacle data.
        state: Dynamic simulator state containing agent slots and masks.
        show_agent_indices: Whether to annotate active agent slots.

    Returns:
        A ``RenderResult`` containing the Matplotlib figure and axes when the
        optional visualization dependency is installed.

    Raises:
        ImportError: If Matplotlib is not installed. Install the ``viz`` extra
        to use this renderer.
    """
    pyplot, circle_factory, polygon_factory = _load_matplotlib()
    figure, axes = pyplot.subplots()

    result = RenderResult(figure=figure, axes=axes)
    _draw_geometry(
        axes,
        circle_factory,
        polygon_factory,
        config,
        state,
        show_agent_indices=show_agent_indices,
    )

    return result


def redraw_geometry(
    config: EnvConfig,
    state: EnvState,
    result: RenderResult,
    *,
    show_agent_indices: bool = True,
) -> RenderResult:
    """Redraw geometry into an existing render result.

    This keeps the static renderer and interactive debug harness on one drawing
    path. It clears and redraws the existing axes; simulator state remains
    read-only.

    Args:
        config: Static episode configuration containing map and obstacle data.
        state: Dynamic simulator state containing agent slots and masks.
        result: Existing Matplotlib figure/axes pair to redraw into.
        show_agent_indices: Whether to annotate active agent slots.

    Returns:
        The same ``RenderResult`` object, after its axes have been redrawn.

    Raises:
        ImportError: If Matplotlib is not installed. Install the ``viz`` extra
        to use rendering helpers.
    """
    _, circle_factory, polygon_factory = _load_matplotlib()
    _draw_geometry(
        cast(_AxesLike, result.axes),
        circle_factory,
        polygon_factory,
        config,
        state,
        show_agent_indices=show_agent_indices,
    )

    return result


def _draw_geometry(
    axes: _AxesLike,
    circle_factory: _PatchFactory,
    polygon_factory: _PatchFactory,
    config: EnvConfig,
    state: EnvState,
    *,
    show_agent_indices: bool,
) -> None:
    """Draw one complete geometry frame into ``axes``."""
    axes.clear()
    _draw_map_boundary(axes, polygon_factory, config)
    _draw_obstacles(axes, circle_factory, polygon_factory, config)
    _draw_agents(
        axes,
        circle_factory,
        config,
        state,
        show_agent_indices=show_agent_indices,
    )
    _style_axes(axes, config)


def _load_matplotlib() -> tuple[_PyplotLike, _PatchFactory, _PatchFactory]:
    """Load Matplotlib lazily so package import stays dependency-light."""
    try:
        pyplot = cast(_PyplotLike, import_module("matplotlib.pyplot"))
        patches = import_module("matplotlib.patches")
    except ImportError as exc:
        msg = (
            "Rendering helpers require the optional visualization dependency "
            "'matplotlib'. Install marl-battlegrounds with the 'viz' extra to "
            "use them."
        )
        raise ImportError(msg) from exc

    circle_factory = cast(_PatchFactory, patches.Circle)
    polygon_factory = cast(_PatchFactory, patches.Polygon)

    return (pyplot, circle_factory, polygon_factory)


def _draw_map_boundary(
    axes: _AxesLike,
    polygon_factory: _PatchFactory,
    config: EnvConfig,
) -> None:
    """Draw the rectangular map bounds from ``EnvConfig``."""
    width = float(config.map_width)
    height = float(config.map_height)
    corners = np.array(
        (
            (0.0, 0.0),
            (width, 0.0),
            (width, height),
            (0.0, height),
        ),
        dtype=np.float32,
    )
    axes.add_patch(
        polygon_factory(
            corners,
            closed=True,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
        )
    )


def _draw_obstacles(
    axes: _AxesLike,
    circle_factory: _PatchFactory,
    polygon_factory: _PatchFactory,
    config: EnvConfig,
) -> None:
    """Draw active pillar and wall obstacle rows."""
    obstacles = np.asarray(config.obstacles, dtype=np.float32)

    for obstacle_slot in range(MAX_OBSTACLE_SLOTS):
        obstacle = obstacles[obstacle_slot]
        is_active = bool(obstacle[OBSTACLE_FEATURE_ACTIVE] == 1.0)
        if not is_active:
            continue

        obstacle_type = int(obstacle[OBSTACLE_FEATURE_TYPE])
        if obstacle_type == OBSTACLE_TYPE_PILLAR:
            _draw_pillar(axes, circle_factory, obstacle)
        elif obstacle_type == OBSTACLE_TYPE_WALL:
            _draw_wall(axes, polygon_factory, obstacle)


def _draw_pillar(
    axes: _AxesLike,
    circle_factory: _PatchFactory,
    obstacle: _ObstacleRow,
) -> None:
    """Draw one active circular pillar obstacle row."""
    center = (
        float(obstacle[OBSTACLE_FEATURE_X]),
        float(obstacle[OBSTACLE_FEATURE_Y]),
    )
    radius = float(obstacle[OBSTACLE_FEATURE_RADIUS])

    axes.add_patch(
        circle_factory(
            center,
            radius,
            fill=False,
            edgecolor="slategray",
            linewidth=1.5,
        )
    )


def _draw_wall(
    axes: _AxesLike,
    polygon_factory: _PatchFactory,
    obstacle: _ObstacleRow,
) -> None:
    """Draw one active rotated wall obstacle row."""
    center = np.array(
        (
            float(obstacle[OBSTACLE_FEATURE_X]),
            float(obstacle[OBSTACLE_FEATURE_Y]),
        ),
        dtype=np.float32,
    )
    width = float(obstacle[OBSTACLE_FEATURE_WIDTH])
    height = float(obstacle[OBSTACLE_FEATURE_HEIGHT])
    theta = float(obstacle[OBSTACLE_FEATURE_THETA])

    half_width = width / 2.0
    half_height = height / 2.0
    local_corners = np.array(
        (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ),
        dtype=np.float32,
    )
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation = np.array(
        (
            (cos_theta, -sin_theta),
            (sin_theta, cos_theta),
        ),
        dtype=np.float32,
    )
    world_corners = local_corners @ rotation.T + center

    axes.add_patch(
        polygon_factory(
            world_corners,
            closed=True,
            fill=False,
            edgecolor="dimgray",
            linewidth=1.5,
        )
    )


def _draw_agents(
    axes: _AxesLike,
    circle_factory: _PatchFactory,
    config: EnvConfig,
    state: EnvState,
    *,
    show_agent_indices: bool,
) -> None:
    """Draw profile-static and state-dynamic facts for active agent slots."""
    positions = np.asarray(state.agent_positions)
    radii = np.asarray(config.agent_profile.agent_radii)
    team_ids = np.asarray(config.agent_profile.team_ids)
    active_mask = np.asarray(config.agent_profile.active_mask)
    alive_mask = np.asarray(state.alive_mask)

    for agent_slot in range(MAX_AGENT_SLOTS):
        if not bool(active_mask[agent_slot]):
            continue

        center = (
            float(positions[agent_slot, 0]),
            float(positions[agent_slot, 1]),
        )
        radius = float(radii[agent_slot])
        team_id = int(team_ids[agent_slot])
        is_alive = bool(alive_mask[agent_slot])
        edgecolor = "tab:blue" if team_id == TEAM_A_ID else "tab:red"
        linestyle = "-" if is_alive else "--"
        alpha = 1.0 if is_alive else 0.45

        axes.add_patch(
            circle_factory(
                center,
                radius,
                fill=False,
                edgecolor=edgecolor,
                linewidth=1.5,
                linestyle=linestyle,
                alpha=alpha,
            )
        )

        if show_agent_indices:
            axes.text(
                center[0],
                center[1],
                str(agent_slot),
                color=edgecolor,
                ha="center",
                va="center",
                fontsize=8,
                alpha=alpha,
            )


def _style_axes(axes: _AxesLike, config: EnvConfig) -> None:
    """Apply stable axis styling for geometry inspection."""
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim(0.0, float(config.map_width))
    axes.set_ylim(0.0, float(config.map_height))
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("MARL-BattleGrounds Geometry")
