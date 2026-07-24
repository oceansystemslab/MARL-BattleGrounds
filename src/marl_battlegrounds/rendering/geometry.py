"""Stateless battlefield presentation for snapshots and described effects."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
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
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.rendering.visuals import (
    BASIC_COLOR,
    DAMAGE_COLOR,
    HEALING_COLOR,
    HISTORY_COLOR,
    HUNTER_COLOR,
    MAGE_COLOR,
    PRIEST_COLOR,
    ROGUE_COLOR,
    TARGET_COLOR,
    ULTIMATE_COLOR,
    UNAVAILABLE_COLOR,
    WARRIOR_COLOR,
    ActivationVisual,
    AuraCueVisual,
    BattlefieldOverlays,
    HealthDeltaVisual,
    LaneMarkerVisual,
    ObserverVisibilityVisual,
    PersistentEffectVisual,
    PreviousAcceptedActionVisual,
    RejectedActionVisual,
    SelectionVisual,
    StatusCueVisual,
    class_color,
    source_effect_color,
    team_color,
)


class _AxesLike(Protocol):
    def add_patch(self, patch: object) -> object: ...

    def clear(self) -> object: ...

    def text(self, x: float, y: float, s: str, **kwargs: object) -> object: ...

    def plot(
        self,
        x: object,
        y: object,
        *args: object,
        **kwargs: object,
    ) -> object: ...

    def annotate(
        self,
        text: str,
        xy: tuple[float, float],
        **kwargs: object,
    ) -> object: ...

    def set_aspect(self, aspect: str, adjustable: str | None = None) -> object: ...

    def set_xlim(self, left: float, right: float) -> object: ...

    def set_ylim(self, bottom: float, top: float) -> object: ...

    def set_xlabel(self, xlabel: str) -> object: ...

    def set_ylabel(self, ylabel: str) -> object: ...

    def set_title(self, label: str) -> object: ...


class _PyplotLike(Protocol):
    def subplots(self) -> tuple[object, _AxesLike]: ...


_PatchFactory = Callable[..., object]
_ObstacleRow = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _MatplotlibParts:
    pyplot: _PyplotLike
    circle: _PatchFactory
    polygon: _PatchFactory
    wedge: _PatchFactory
    arc: _PatchFactory
    regular_polygon: _PatchFactory


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Matplotlib objects created by battlefield rendering helpers."""

    figure: object
    axes: object


def draw_geometry(
    axes: object,
    config: EnvConfig,
    state: EnvState,
    *,
    overlays: BattlefieldOverlays | None = None,
    show_agent_indices: bool = True,
) -> None:
    """Clear and draw exactly one stateless battlefield frame."""
    parts = _load_matplotlib()
    typed_axes = cast(_AxesLike, axes)
    supplied_overlays = overlays or BattlefieldOverlays()

    typed_axes.clear()
    _draw_map_boundary(typed_axes, parts.polygon, config)
    _draw_obstacles(typed_axes, parts.circle, parts.polygon, config)
    _draw_world_overlays(typed_axes, parts, config, state, supplied_overlays)
    _draw_agents(
        typed_axes,
        parts,
        config,
        state,
        supplied_overlays,
        show_agent_indices=show_agent_indices,
    )
    _style_axes(typed_axes, config)


def render_geometry(
    config: EnvConfig,
    state: EnvState,
    *,
    overlays: BattlefieldOverlays | None = None,
    show_agent_indices: bool = True,
) -> RenderResult:
    """Create a figure and render one battlefield snapshot."""
    parts = _load_matplotlib()
    figure, axes = parts.pyplot.subplots()
    result = RenderResult(figure=figure, axes=axes)
    draw_geometry(
        axes,
        config,
        state,
        overlays=overlays,
        show_agent_indices=show_agent_indices,
    )
    return result


def redraw_geometry(
    config: EnvConfig,
    state: EnvState,
    result: RenderResult,
    *,
    overlays: BattlefieldOverlays | None = None,
    show_agent_indices: bool = True,
) -> RenderResult:
    """Redraw an existing axes and return the identical result object."""
    draw_geometry(
        result.axes,
        config,
        state,
        overlays=overlays,
        show_agent_indices=show_agent_indices,
    )
    return result


def _load_matplotlib() -> _MatplotlibParts:
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

    return _MatplotlibParts(
        pyplot=pyplot,
        circle=cast(_PatchFactory, patches.Circle),
        polygon=cast(_PatchFactory, patches.Polygon),
        wedge=cast(_PatchFactory, patches.Wedge),
        arc=cast(_PatchFactory, patches.Arc),
        regular_polygon=cast(_PatchFactory, patches.RegularPolygon),
    )


def _draw_map_boundary(
    axes: _AxesLike,
    polygon_factory: _PatchFactory,
    config: EnvConfig,
) -> None:
    corners = np.asarray(
        (
            (0.0, 0.0),
            (float(config.map_width), 0.0),
            (float(config.map_width), float(config.map_height)),
            (0.0, float(config.map_height)),
        ),
        dtype=np.float32,
    )
    axes.add_patch(
        polygon_factory(
            corners,
            closed=True,
            facecolor="#F8F9FA",
            edgecolor="none",
            zorder=0,
        )
    )
    axes.add_patch(
        polygon_factory(
            corners,
            closed=True,
            facecolor="none",
            edgecolor="#111111",
            linewidth=2.0,
            zorder=10,
        )
    )


def _draw_obstacles(
    axes: _AxesLike,
    circle_factory: _PatchFactory,
    polygon_factory: _PatchFactory,
    config: EnvConfig,
) -> None:
    obstacles = np.asarray(config.obstacles, dtype=np.float32)
    for obstacle_slot in range(MAX_OBSTACLE_SLOTS):
        obstacle = obstacles[obstacle_slot]
        if not bool(obstacle[OBSTACLE_FEATURE_ACTIVE] == 1.0):
            continue

        obstacle_type = int(obstacle[OBSTACLE_FEATURE_TYPE])
        if obstacle_type == OBSTACLE_TYPE_PILLAR:
            center = (
                float(obstacle[OBSTACLE_FEATURE_X]),
                float(obstacle[OBSTACLE_FEATURE_Y]),
            )
            axes.add_patch(
                circle_factory(
                    center,
                    float(obstacle[OBSTACLE_FEATURE_RADIUS]),
                    facecolor="#B0B5BC",
                    edgecolor="#59636E",
                    linewidth=1.5,
                    hatch="//",
                    zorder=10,
                )
            )
        elif obstacle_type == OBSTACLE_TYPE_WALL:
            _draw_wall(axes, polygon_factory, obstacle)


def _draw_wall(
    axes: _AxesLike,
    polygon_factory: _PatchFactory,
    obstacle: _ObstacleRow,
) -> None:
    center = np.asarray(
        (
            float(obstacle[OBSTACLE_FEATURE_X]),
            float(obstacle[OBSTACLE_FEATURE_Y]),
        ),
        dtype=np.float32,
    )
    half_width = float(obstacle[OBSTACLE_FEATURE_WIDTH]) / 2.0
    half_height = float(obstacle[OBSTACLE_FEATURE_HEIGHT]) / 2.0
    theta = float(obstacle[OBSTACLE_FEATURE_THETA])
    local_corners = np.asarray(
        (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ),
        dtype=np.float32,
    )
    rotation = np.asarray(
        (
            (np.cos(theta), -np.sin(theta)),
            (np.sin(theta), np.cos(theta)),
        ),
        dtype=np.float32,
    )
    axes.add_patch(
        polygon_factory(
            local_corners @ rotation.T + center,
            closed=True,
            facecolor="#A7ADB5",
            edgecolor="#4C5560",
            linewidth=1.5,
            hatch="///",
            zorder=10,
        )
    )


def _draw_world_overlays(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    config: EnvConfig,
    state: EnvState,
    overlays: BattlefieldOverlays,
) -> None:
    positions = np.asarray(state.agent_positions, dtype=np.float32)

    range_styles = {
        "observation": (UNAVAILABLE_COLOR, "--", 0.35),
        "basic": (BASIC_COLOR, "-", 0.45),
        "ultimate": (ULTIMATE_COLOR, ":", 0.50),
    }
    for visual in overlays.ranges:
        color, linestyle, alpha = range_styles[visual.kind]
        axes.add_patch(
            parts.circle(
                visual.center,
                visual.radius,
                fill=False,
                edgecolor=color,
                linewidth=1.4,
                linestyle=linestyle,
                alpha=alpha,
                zorder=20,
            )
        )

    for visual in overlays.charge_trails:
        axes.annotate(
            "",
            xy=visual.end,
            xytext=visual.start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": WARRIOR_COLOR,
                "linewidth": 2.7,
                "alpha": visual.opacity,
            },
            zorder=30 if visual.opacity == 1.0 else 25,
        )

    for visual in overlays.target_links:
        source = positions[visual.source_global_slot]
        target = positions[visual.target_global_slot]
        color = BASIC_COLOR if visual.lane == 0 else ULTIMATE_COLOR
        if not visual.legal:
            color = DAMAGE_COLOR
        axes.plot(
            (float(source[0]), float(target[0])),
            (float(source[1]), float(target[1])),
            color=color,
            linewidth=2.0,
            linestyle="-" if visual.legal else "--",
            alpha=0.8,
            zorder=30,
        )

    for visual in overlays.rejections:
        if (
            visual.component != "combat"
            or visual.target_global_slot is None
            or visual.lane is None
        ):
            continue
        source = positions[visual.actor_global_slot]
        target = positions[visual.target_global_slot]
        axes.plot(
            (float(source[0]), float(target[0])),
            (float(source[1]), float(target[1])),
            color=DAMAGE_COLOR,
            linewidth=2.0,
            linestyle="--",
            alpha=0.9,
            zorder=30,
        )

    for visual in overlays.activations:
        if visual.target_global_slot is None or visual.kind not in (
            "basic_damage",
            "basic_heal",
            "holy_word",
        ):
            continue
        source = positions[visual.source_global_slot]
        target = positions[visual.target_global_slot]
        color = (
            source_effect_color(visual.source_class_id)
            if visual.kind == "basic_damage"
            else PRIEST_COLOR
        )
        axes.plot(
            (float(source[0]), float(target[0])),
            (float(source[1]), float(target[1])),
            color=color,
            linewidth=1.7 if visual.kind != "holy_word" else 2.5,
            linestyle="-",
            alpha=0.75,
            zorder=30,
        )

    _ = config


def _draw_agents(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    config: EnvConfig,
    state: EnvState,
    overlays: BattlefieldOverlays,
    *,
    show_agent_indices: bool,
) -> None:
    positions = np.asarray(state.agent_positions, dtype=np.float32)
    radii = np.asarray(config.agent_profile.agent_radii, dtype=np.float32)
    class_ids = np.asarray(config.agent_profile.class_ids, dtype=np.int32)
    team_ids = np.asarray(config.agent_profile.team_ids, dtype=np.int32)
    active_mask = np.asarray(config.agent_profile.active_mask, dtype=bool)
    alive_mask = np.asarray(state.alive_mask, dtype=bool)
    current_health = np.asarray(state.current_health, dtype=np.float32)
    max_health = np.asarray(config.agent_profile.max_health, dtype=np.float32)

    statuses = _group_by_slot(overlays.statuses, "global_slot")
    auras = _group_by_slot(overlays.auras, "global_slot")
    persistent = _group_by_slot(overlays.persistent_effects, "global_slot")
    lanes = _group_by_slot(overlays.lane_markers, "candidate_global_slot")
    selections = _group_by_slot(overlays.selections, "global_slot")
    visibility = _group_by_slot(
        overlays.observer_visibility,
        "candidate_global_slot",
    )
    previous_actions = _group_by_slot(
        overlays.previous_actions,
        "actor_global_slot",
    )
    health_deltas = _group_by_slot(overlays.health_deltas, "global_slot")
    activations = _group_by_slot(overlays.activations, "target_global_slot")
    source_activations = _group_by_slot(overlays.activations, "source_global_slot")
    rejections = _group_by_slot(overlays.rejections, "actor_global_slot")

    for global_slot in range(MAX_AGENT_SLOTS):
        if not active_mask[global_slot]:
            continue

        center = (
            float(positions[global_slot, 0]),
            float(positions[global_slot, 1]),
        )
        radius = float(radii[global_slot])
        class_id = int(class_ids[global_slot])
        is_alive = bool(alive_mask[global_slot])
        alpha = 1.0 if is_alive else 0.45

        axes.add_patch(
            parts.circle(
                center,
                radius,
                facecolor=class_color(class_id),
                edgecolor="none",
                alpha=0.58 * alpha,
                zorder=40,
            )
        )
        _draw_health(
            axes,
            parts,
            center,
            radius,
            float(current_health[global_slot]),
            float(max_health[global_slot]),
            alpha,
        )
        _draw_auras(axes, parts, center, radius, auras[global_slot])
        _draw_statuses(axes, center, radius, statuses[global_slot])
        _draw_persistent_effects(
            axes,
            parts,
            center,
            radius,
            persistent[global_slot],
        )

        axes.add_patch(
            parts.circle(
                center,
                radius,
                fill=False,
                edgecolor=team_color(int(team_ids[global_slot])),
                linewidth=2.2,
                linestyle="-" if is_alive else "--",
                alpha=alpha,
                zorder=42,
            )
        )

        _draw_lane_markers(axes, parts, center, radius, lanes[global_slot])
        _draw_selections(axes, parts, center, radius, selections[global_slot])
        _draw_previous_action(
            axes,
            center,
            radius,
            previous_actions[global_slot],
        )

        if (
            visibility[global_slot]
            and not cast(
                ObserverVisibilityVisual,
                visibility[global_slot][-1],
            ).observer_visible
        ):
            axes.add_patch(
                parts.circle(
                    center,
                    radius * 0.96,
                    facecolor="#6B7280",
                    edgecolor="none",
                    alpha=0.32,
                    zorder=42.5,
                )
            )
            axes.add_patch(
                parts.wedge(
                    center,
                    radius * 0.96,
                    0,
                    360,
                    width=radius * 0.46,
                    facecolor="#6B7280",
                    edgecolor="none",
                    alpha=0.28,
                    hatch="....",
                    zorder=44,
                )
            )

        _draw_transients(
            axes,
            parts,
            center,
            radius,
            health_deltas[global_slot],
            activations[global_slot],
            source_activations[global_slot],
            rejections[global_slot],
        )

        class_letter = {
            1: "M",
            2: "W",
            3: "H",
            4: "R",
            5: "P",
        }[class_id]
        axes.text(
            center[0],
            center[1],
            class_letter,
            color="#111111",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            alpha=alpha,
            zorder=45,
        )
        if show_agent_indices:
            axes.text(
                center[0],
                center[1] - radius * 0.34,
                f"g{global_slot}",
                color="#111111",
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold",
                alpha=alpha,
                zorder=45,
            )


def _group_by_slot(
    values: tuple[object, ...],
    attribute: str,
) -> defaultdict[int, list[object]]:
    grouped: defaultdict[int, list[object]] = defaultdict(list)
    for value in values:
        slot = getattr(value, attribute)
        if slot is not None:
            grouped[int(slot)].append(value)
    return grouped


def _draw_health(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    center: tuple[float, float],
    radius: float,
    current: float,
    maximum: float,
    alpha: float,
) -> None:
    fraction = float(np.clip(current / maximum if maximum > 0 else 0.0, 0.0, 1.0))
    axes.add_patch(
        parts.wedge(
            center,
            radius * 0.86,
            0,
            360,
            width=radius * 0.13,
            facecolor="#30343B",
            edgecolor="none",
            alpha=0.55 * alpha,
            zorder=41,
        )
    )
    health_color = (
        "#2E7D32" if fraction > 0.5 else "#F9A825" if fraction > 0.25 else DAMAGE_COLOR
    )
    if fraction > 0:
        axes.add_patch(
            parts.wedge(
                center,
                radius * 0.86,
                90,
                90 + 360 * fraction,
                width=radius * 0.13,
                facecolor=health_color,
                edgecolor="none",
                alpha=alpha,
                zorder=41,
            )
        )


def _draw_auras(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    center: tuple[float, float],
    radius: float,
    values: list[object],
) -> None:
    for value in values:
        visual = cast(AuraCueVisual, value)
        if visual.kind == "mage_amplification":
            theta1, theta2, color, hatch = 0, 180, MAGE_COLOR, "..."
        else:
            theta1, theta2, color, hatch = 180, 360, WARRIOR_COLOR, "///"
        axes.add_patch(
            parts.wedge(
                center,
                radius * 0.69,
                theta1,
                theta2,
                width=radius * 0.08,
                facecolor="none",
                edgecolor=color,
                linewidth=1.5,
                hatch=hatch,
                zorder=41,
            )
        )


def _draw_statuses(
    axes: _AxesLike,
    center: tuple[float, float],
    radius: float,
    values: list[object],
) -> None:
    x_offsets = (-0.28, 0.0, 0.28)
    for value in values:
        visual = cast(StatusCueVisual, value)
        x = center[0] + radius * x_offsets[visual.channel_index]
        y_sign = 1.0 if visual.family == "stun" else -1.0
        y = center[1] + y_sign * radius * 0.38
        color = source_effect_color(visual.source_class_id)
        if visual.family == "stun":
            axes.text(
                x,
                y,
                "◆",
                color=color,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                zorder=41,
            )
        else:
            for y_offset in (0.035, -0.035):
                axes.text(
                    x,
                    y + radius * y_offset,
                    "▼",
                    color=color,
                    ha="center",
                    va="center",
                    fontsize=5,
                    fontweight="bold",
                    zorder=41,
                )


def _draw_persistent_effects(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    center: tuple[float, float],
    radius: float,
    values: list[object],
) -> None:
    for value in values:
        visual = cast(PersistentEffectVisual, value)
        if visual.kind == "rogue_anti_heal":
            axes.text(
                center[0] - radius * 0.43,
                center[1],
                "✚̸",
                color=ROGUE_COLOR,
                ha="center",
                va="center",
                fontsize=7,
                zorder=41,
            )
        elif visual.kind == "priest_freedom":
            axes.add_patch(
                parts.regular_polygon(
                    (center[0] + radius * 0.38, center[1]),
                    numVertices=5,
                    radius=radius * 0.10,
                    orientation=np.pi,
                    facecolor="white",
                    edgecolor=PRIEST_COLOR,
                    linewidth=1.4,
                    zorder=41,
                )
            )
        else:
            axes.add_patch(
                parts.regular_polygon(
                    center,
                    numVertices=8,
                    radius=radius * 0.38,
                    orientation=np.pi / 8,
                    fill=False,
                    edgecolor=MAGE_COLOR,
                    linewidth=1.3,
                    zorder=41,
                )
            )


def _draw_lane_markers(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    center: tuple[float, float],
    radius: float,
    values: list[object],
) -> None:
    for value in values:
        visual = cast(LaneMarkerVisual, value)
        color = (
            (BASIC_COLOR if visual.lane == 0 else ULTIMATE_COLOR)
            if visual.available
            else UNAVAILABLE_COLOR
        )
        theta1, theta2 = (205, 255) if visual.lane == 0 else (285, 335)
        axes.add_patch(
            parts.arc(
                center,
                radius * 1.10,
                radius * 1.10,
                theta1=theta1,
                theta2=theta2,
                color=color,
                linewidth=3.0 if visual.selected else 1.4,
                alpha=1.0 if visual.available else 0.6,
                zorder=43,
            )
        )


def _draw_selections(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    center: tuple[float, float],
    radius: float,
    values: list[object],
) -> None:
    for value in values:
        visual = cast(SelectionVisual, value)
        if visual.role == "target":
            axes.plot(
                (center[0] - radius * 0.28, center[0] + radius * 0.28),
                (center[1], center[1]),
                color=TARGET_COLOR,
                linewidth=1.8,
                zorder=43,
            )
            axes.plot(
                (center[0], center[0]),
                (center[1] - radius * 0.28, center[1] + radius * 0.28),
                color=TARGET_COLOR,
                linewidth=1.8,
                zorder=43,
            )
            axes.add_patch(
                parts.circle(
                    center,
                    radius * 0.24,
                    fill=False,
                    edgecolor=TARGET_COLOR,
                    linewidth=1.5,
                    zorder=43,
                )
            )
        else:
            for x_sign, y_sign in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                x_values = (
                    center[0] + x_sign * radius * 0.50,
                    center[0] + x_sign * radius * 0.44,
                )
                y_values = (
                    center[1] + y_sign * radius * 0.50,
                    center[1] + y_sign * radius * 0.44,
                )
                for color, linewidth, zorder in (
                    ("#111111", 4.0, 42.9),
                    ("white", 2.2, 43),
                ):
                    axes.plot(
                        x_values,
                        y_values,
                        color=color,
                        linewidth=linewidth,
                        solid_capstyle="round",
                        zorder=zorder,
                    )


def _draw_previous_action(
    axes: _AxesLike,
    center: tuple[float, float],
    radius: float,
    values: list[object],
) -> None:
    if not values:
        return
    visual = cast(PreviousAcceptedActionVisual, values[-1])
    movement = {
        MOVE_STAY: "·",
        MOVE_NORTH: "↑",
        MOVE_SOUTH: "↓",
        MOVE_EAST: "→",
        MOVE_WEST: "←",
        MOVE_NORTHEAST: "↗",
        MOVE_NORTHWEST: "↖",
        MOVE_SOUTHEAST: "↘",
        MOVE_SOUTHWEST: "↙",
    }.get(visual.move_action, "?")
    badge = (
        "U" if visual.use_ultimate == 1 else "B" if visual.target_action != 0 else "·"
    )
    axes.text(
        center[0],
        center[1] + radius * 0.25,
        f"{movement}{badge}",
        color=HISTORY_COLOR,
        ha="center",
        va="center",
        fontsize=5,
        zorder=43,
    )


def _draw_transients(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    center: tuple[float, float],
    radius: float,
    health_values: list[object],
    target_activations: list[object],
    source_activations: list[object],
    rejection_values: list[object],
) -> None:
    for value in health_values[-1:]:
        visual = cast(HealthDeltaVisual, value)
        color = HEALING_COLOR if visual.net_delta > 0 else DAMAGE_COLOR
        axes.text(
            center[0],
            center[1] + radius * 1.25,
            f"{visual.net_delta:+.2f}",
            color=color,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            zorder=50,
        )

    recipient_kinds = {
        cast(ActivationVisual, value).kind for value in target_activations
    }
    if len(recipient_kinds) > 1:
        ordered_kinds = tuple(sorted(recipient_kinds))
        segment_size = 360.0 / len(ordered_kinds)
        for index, kind in enumerate(ordered_kinds):
            axes.add_patch(
                parts.arc(
                    center,
                    radius * 2.28,
                    radius * 2.28,
                    theta1=index * segment_size + 3.0,
                    theta2=(index + 1) * segment_size - 3.0,
                    color=_activation_color(kind),
                    linewidth=3.0,
                    alpha=0.9,
                    zorder=50,
                )
            )
    elif recipient_kinds:
        _draw_single_activation(
            axes,
            parts,
            center,
            radius,
            next(iter(recipient_kinds)),
        )

    if any(
        cast(ActivationVisual, value).kind == "mage_burst"
        and cast(ActivationVisual, value).target_global_slot is None
        for value in source_activations
    ):
        axes.add_patch(
            parts.regular_polygon(
                center,
                numVertices=8,
                radius=radius * 1.25,
                orientation=np.pi / 8,
                fill=False,
                edgecolor=MAGE_COLOR,
                linewidth=2.5,
                alpha=0.9,
                zorder=50,
            )
        )

    if rejection_values:
        components = {
            cast(RejectedActionVisual, value).component for value in rejection_values
        }
        labels = {
            "movement": "M\u00d7",
            "combat": "C\u00d7",
            "complete_tuple_domain": "TUPLE\u00d7",
        }
        label = " ".join(
            labels[component]
            for component in (
                "complete_tuple_domain",
                "movement",
                "combat",
            )
            if component in components
        )
        axes.text(
            center[0],
            center[1] - radius * 1.22,
            label,
            color=DAMAGE_COLOR,
            ha="center",
            va="top",
            fontsize=7,
            fontweight="bold",
            zorder=50,
        )


def _activation_color(kind: str) -> str:
    return {
        "basic_damage": DAMAGE_COLOR,
        "basic_heal": HEALING_COLOR,
        "holy_word": PRIEST_COLOR,
        "mage_burst": MAGE_COLOR,
        "warrior_charge": WARRIOR_COLOR,
        "hunter_trap": HUNTER_COLOR,
        "rogue_poison": ROGUE_COLOR,
    }[kind]


def _draw_single_activation(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    center: tuple[float, float],
    radius: float,
    kind: str,
) -> None:
    if kind == "holy_word":
        for scale, color in ((1.10, HEALING_COLOR), (1.20, PRIEST_COLOR)):
            axes.add_patch(
                parts.circle(
                    center,
                    radius * scale,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.2,
                    alpha=0.88,
                    zorder=50,
                )
            )
    elif kind == "hunter_trap":
        axes.add_patch(
            parts.regular_polygon(
                center,
                numVertices=4,
                radius=radius * 1.15,
                orientation=np.pi / 4,
                fill=False,
                edgecolor=HUNTER_COLOR,
                linewidth=2.6,
                zorder=50,
            )
        )
    elif kind == "rogue_poison":
        for x_offset in (-0.18, 0.0, 0.18):
            axes.add_patch(
                parts.regular_polygon(
                    (center[0] + radius * x_offset, center[1]),
                    numVertices=3,
                    radius=radius * 0.16,
                    orientation=np.pi,
                    facecolor=ROGUE_COLOR,
                    edgecolor=ROGUE_COLOR,
                    alpha=0.9,
                    zorder=50,
                )
            )
    elif kind == "warrior_charge":
        axes.add_patch(
            parts.regular_polygon(
                center,
                numVertices=3,
                radius=radius * 1.15,
                orientation=-np.pi / 2,
                fill=False,
                edgecolor=WARRIOR_COLOR,
                linewidth=2.8,
                zorder=50,
            )
        )
    elif kind == "basic_damage":
        axes.text(
            center[0],
            center[1] + radius * 0.10,
            "✦",
            color=DAMAGE_COLOR,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            zorder=50,
        )
    else:
        axes.add_patch(
            parts.circle(
                center,
                radius * 1.12,
                fill=False,
                edgecolor=HEALING_COLOR,
                linewidth=2.4,
                alpha=0.88,
                zorder=50,
            )
        )


def _style_axes(axes: _AxesLike, config: EnvConfig) -> None:
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim(0.0, float(config.map_width))
    axes.set_ylim(0.0, float(config.map_height))
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("MARL-BattleGrounds")
