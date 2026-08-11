"""Stateless Matplotlib rendering for authorized scene and event records."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from math import cos, hypot, sin
from typing import Protocol, cast

from marl_battlegrounds.rendering.scene import (
    AcceptedActivationEventV1,
    AgentSceneV1,
    AgentSceneV2,
    BattlefieldSceneV1,
    BattlefieldSceneV2,
    ChargeDisplacementEventV1,
    NetHealthEventV1,
    RejectedActionEventV1,
    StatusLifecycleEventV1,
    VisualEventBatchV1,
)
from marl_battlegrounds.rendering.vocabulary import (
    class_token_from_id,
    lookup_activation_token,
    lookup_lifecycle_token,
    lookup_modifier_token,
    lookup_status_token,
    team_token_from_id,
)

_BACKGROUND = "#0B1020"
_BATTLEFIELD = "#111827"
_TEXT = "#F4F7FB"
_MUTED = "#9AA7B8"
_TEAM_A = "#3B82F6"
_TEAM_B = "#F05A67"
_CLASS_COLORS = {
    "mage": "#22D3EE",
    "warrior": "#D18B47",
    "hunter": "#84CC16",
    "rogue": "#FACC15",
    "priest": "#F472B6",
}
_DAMAGE = "#FB7185"
_HEALING = "#34D399"
_BASIC = "#2DD4BF"
_ULTIMATE = "#A78BFA"
_UNAVAILABLE = "#64748B"
_TARGET = "#F472B6"

type BattlefieldScene = BattlefieldSceneV1 | BattlefieldSceneV2


def _aura_color(token_id: str) -> str:
    """Return a neutral fallback for future aura presentation tokens."""
    return {
        "mage_amplification": "#22D3EE",
        "warrior_mitigation": "#D18B47",
    }.get(token_id, _MUTED)


def _format_display_number(value: float) -> str:
    """Format a human-visible scalar without exposing binary-float spill."""
    formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if formatted == "-0" else formatted


class _ArtistLike(Protocol):
    def set_gid(self, gid: str) -> object: ...


class _AxesLike(Protocol):
    transAxes: object  # noqa: N815 - Matplotlib public attribute.

    def add_patch(self, patch: object) -> object: ...

    def annotate(
        self,
        text: str,
        xy: tuple[float, float],
        **kwargs: object,
    ) -> object: ...

    def clear(self) -> object: ...

    def set_aspect(self, aspect: str, adjustable: str | None = None) -> object: ...

    def set_facecolor(self, color: str) -> object: ...

    def set_title(self, label: str, **kwargs: object) -> object: ...

    def set_xlim(self, left: float, right: float) -> object: ...

    def set_ylim(self, bottom: float, top: float) -> object: ...

    def text(self, x: float, y: float, s: str, **kwargs: object) -> object: ...


class _PyplotLike(Protocol):
    def subplots(self) -> tuple[object, _AxesLike]: ...


_PatchFactory = Callable[..., object]


@dataclass(frozen=True, slots=True)
class _MatplotlibParts:
    pyplot: _PyplotLike
    circle: _PatchFactory
    polygon: _PatchFactory
    rectangle: _PatchFactory
    wedge: _PatchFactory


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Matplotlib objects created by scene-native static rendering helpers."""

    figure: object
    axes: object


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneRenderOptions:
    """Presentation-only switches for one scene-native static render."""

    show_agent_ids: bool = False
    show_ranges: bool = True
    show_statuses: bool = True
    show_modifiers: bool = True
    show_observer_visibility: bool = False
    show_events: bool = True

    def __post_init__(self) -> None:
        for name in (
            "show_agent_ids",
            "show_ranges",
            "show_statuses",
            "show_modifiers",
            "show_observer_visibility",
            "show_events",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a Python bool.")


def draw_scene_geometry(
    axes: object,
    scene: BattlefieldScene,
    *,
    event_batch: VisualEventBatchV1 | None = None,
    options: SceneRenderOptions | None = None,
) -> None:
    """Clear and draw one authorized scene plus its latest static event batch."""
    if type(scene) not in (BattlefieldSceneV1, BattlefieldSceneV2):
        raise TypeError("scene must be BattlefieldSceneV1 or BattlefieldSceneV2.")
    if event_batch is not None and type(event_batch) is not VisualEventBatchV1:
        raise TypeError("event_batch must be VisualEventBatchV1 or None.")
    if options is not None and type(options) is not SceneRenderOptions:
        raise TypeError("options must be SceneRenderOptions or None.")
    render_options = options or SceneRenderOptions()
    parts = _load_matplotlib()
    typed_axes = cast(_AxesLike, axes)

    if type(scene) is BattlefieldSceneV2:
        if event_batch is not None:
            raise ValueError(
                "BattlefieldSceneV2 does not accept legacy VisualEventBatchV1."
            )
        _draw_scene_v2(typed_axes, parts, scene, render_options)
        return

    legacy_scene = cast(BattlefieldSceneV1, scene)
    typed_axes.clear()
    _draw_map(typed_axes, parts, legacy_scene)
    _draw_fields(typed_axes, parts, legacy_scene, render_options)
    _draw_pending_route(typed_axes, legacy_scene)
    _draw_obstacles(typed_axes, parts, legacy_scene)
    _draw_agents(typed_axes, parts, legacy_scene, render_options)
    _draw_selected_legality(typed_axes, legacy_scene)
    if render_options.show_observer_visibility:
        _draw_observer_visibility(typed_axes, legacy_scene)
    if render_options.show_events and event_batch is not None:
        _draw_events(typed_axes, legacy_scene, event_batch)
    _draw_audience_badge(typed_axes, legacy_scene)
    _style_axes(typed_axes, legacy_scene)


def render_scene_geometry(
    scene: BattlefieldScene,
    *,
    event_batch: VisualEventBatchV1 | None = None,
    options: SceneRenderOptions | None = None,
) -> RenderResult:
    """Create a Matplotlib figure for one authorized scene/event snapshot."""
    parts = _load_matplotlib()
    figure, axes = parts.pyplot.subplots()
    result = RenderResult(figure=figure, axes=axes)
    draw_scene_geometry(
        axes,
        scene,
        event_batch=event_batch,
        options=options,
    )
    set_facecolor = getattr(figure, "set_facecolor", None)
    if callable(set_facecolor):
        set_facecolor(_BACKGROUND)
    return result


def redraw_scene_geometry(
    scene: BattlefieldScene,
    result: RenderResult,
    *,
    event_batch: VisualEventBatchV1 | None = None,
    options: SceneRenderOptions | None = None,
) -> RenderResult:
    """Redraw an existing scene figure and return the identical result object."""
    draw_scene_geometry(
        result.axes,
        scene,
        event_batch=event_batch,
        options=options,
    )
    return result


def _draw_scene_v2(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    scene: BattlefieldSceneV2,
    options: SceneRenderOptions,
) -> None:
    """Draw one canonical-record durable scene without inferred events."""
    axes.clear()
    _draw_map(axes, parts, scene)
    _draw_v2_spawn_pads(axes, parts, scene)
    _draw_v2_aura_fields(axes, parts, scene)
    if options.show_ranges:
        _draw_v2_ranges(axes, parts, scene)
    _draw_obstacles(axes, parts, scene)
    _draw_v2_agents(axes, parts, scene, options)
    _draw_v2_wave_clocks(axes, scene)
    _draw_audience_badge(axes, scene)
    _style_axes(axes, scene)
    axes.set_title(
        (
            "MARL-BattleGrounds · Replay Analyzer · "
            f"Frame {scene.frame_index} · step {scene.simulator_step_count}"
        ),
        color=_TEXT,
        fontsize=10,
    )


def _draw_v2_spawn_pads(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    scene: BattlefieldSceneV2,
) -> None:
    for pad in scene.spawn_pads:
        color = _TEAM_A if pad.team_id == 1 else _TEAM_B
        marker = parts.circle(
            pad.position,
            0.22,
            facecolor="none",
            edgecolor=color,
            linewidth=1.0,
            linestyle=":",
            alpha=0.65,
            zorder=4,
        )
        axes.add_patch(
            _tag(
                marker,
                f"scene:v2:spawn-pad:{pad.assigned_public_agent_id}",
            )
        )


def _draw_v2_aura_fields(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    scene: BattlefieldSceneV2,
) -> None:
    for aura in scene.aura_fields:
        if not aura.source_alive:
            continue
        color = "#22D3EE" if aura.aura_id == "mage_damage_amplification" else "#D18B47"
        field_patch = parts.circle(
            aura.center,
            aura.radius,
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
            alpha=0.12,
            zorder=3,
        )
        axes.add_patch(
            _tag(
                field_patch,
                f"scene:v2:aura:{aura.source_public_agent_id}:{aura.aura_id}",
            )
        )


def _draw_v2_ranges(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    scene: BattlefieldSceneV2,
) -> None:
    colors = {
        "observation": _UNAVAILABLE,
        "basic": _BASIC,
        "ultimate": _ULTIMATE,
    }
    for range_row in scene.ranges:
        if range_row.radius <= 0.0:
            continue
        patch = parts.circle(
            range_row.center,
            range_row.radius,
            facecolor="none",
            edgecolor=colors[range_row.kind],
            linewidth=0.8,
            linestyle="--",
            alpha=0.55,
            zorder=5,
        )
        axes.add_patch(
            _tag(
                patch,
                f"scene:v2:range:{range_row.global_slot}:{range_row.kind}",
            )
        )


def _draw_v2_agents(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    scene: BattlefieldSceneV2,
    options: SceneRenderOptions,
) -> None:
    selection = scene.selection
    for agent in scene.agents:
        show_identity = options.show_agent_ids or (
            selection is not None
            and agent.global_slot
            in (
                selection.controlled_global_slot,
                selection.selected_global_slot,
            )
        )
        _draw_v2_agent_body(
            axes,
            parts,
            agent,
            options,
            show_identity=show_identity,
        )
        if selection is not None and (
            agent.global_slot == selection.controlled_global_slot
        ):
            halo = parts.circle(
                agent.position,
                agent.radius * 1.32,
                facecolor="none",
                edgecolor=_TEXT,
                linewidth=2.5,
                zorder=31,
            )
            axes.add_patch(
                _tag(halo, f"scene:v2:selection:controlled:{agent.public_agent_id}")
            )
        if selection is not None and (
            agent.global_slot == selection.selected_global_slot
        ):
            reticle = parts.circle(
                agent.position,
                agent.radius * 1.48,
                facecolor="none",
                edgecolor=_TARGET,
                linewidth=2.2,
                linestyle="--",
                zorder=32,
            )
            axes.add_patch(
                _tag(reticle, f"scene:v2:selection:target:{agent.public_agent_id}")
            )


def _draw_v2_agent_body(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    agent: AgentSceneV2,
    options: SceneRenderOptions,
    *,
    show_identity: bool,
) -> None:
    class_token = class_token_from_id(agent.class_id)
    team_token = team_token_from_id(agent.team_id)
    class_color = _CLASS_COLORS.get(class_token.token_id, _MUTED)
    team_color = _TEAM_A if agent.team_id == 1 else _TEAM_B
    alive = agent.life_state == "alive"
    body = parts.circle(
        agent.position,
        agent.radius,
        facecolor=class_color,
        edgecolor=team_color,
        linewidth=3.0,
        alpha=0.95 if alive else 0.35,
        zorder=24,
    )
    axes.add_patch(_tag(body, f"scene:v2:agent:{agent.public_agent_id}:body"))
    health_fraction = min(max(agent.current_health / agent.max_health, 0.0), 1.0)
    if health_fraction > 0.0:
        health = parts.wedge(
            agent.position,
            agent.radius * 0.86,
            90.0,
            90.0 + 360.0 * health_fraction,
            width=max(agent.radius * 0.10, 0.02),
            facecolor=_HEALING,
            edgecolor="none",
            zorder=27,
        )
        axes.add_patch(_tag(health, f"scene:v2:agent:{agent.public_agent_id}:health"))
    class_artist = axes.text(
        agent.position[0],
        agent.position[1],
        "DEAD" if not alive else class_token.fallback,
        color=_TEXT,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=28,
    )
    _tag(class_artist, f"scene:v2:agent:{agent.public_agent_id}:class")
    if show_identity:
        identity = axes.annotate(
            f"Agent ID {agent.public_agent_id}",
            xy=agent.position,
            xytext=(0, -16),
            textcoords="offset points",
            color=_TEXT,
            fontsize=6,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=29,
        )
        _tag(identity, f"scene:v2:agent:{agent.public_agent_id}:identity")
    if agent.spawn_shield_remaining > 0:
        shield = parts.circle(
            agent.position,
            agent.radius * 1.18,
            facecolor="none",
            edgecolor="#67E8F9",
            linewidth=1.8,
            alpha=0.9,
            zorder=30,
        )
        axes.add_patch(
            _tag(shield, f"scene:v2:agent:{agent.public_agent_id}:spawn-shield")
        )
    countdown = axes.annotate(
        (
            f"U {agent.ultimate_cooldown_remaining} · "
            f"OOC {agent.steps_until_out_of_combat}"
        ),
        xy=agent.position,
        xytext=(0, 14),
        textcoords="offset points",
        color=_MUTED,
        fontsize=5.5,
        ha="center",
        va="bottom",
        zorder=34,
    )
    _tag(countdown, f"scene:v2:agent:{agent.public_agent_id}:countdowns")
    if options.show_statuses:
        for index, status in enumerate(agent.statuses):
            source = class_token_from_id(status.source_class_id)
            status_color = _CLASS_COLORS.get(source.token_id, _MUTED)
            source_suffix = (
                ""
                if not status.direct_source_evidence
                else " · "
                + ",".join(
                    row.source_public_agent_id for row in status.direct_source_evidence
                )
            )
            status_artist = axes.annotate(
                f"{status.status_id} {status.remaining_duration}{source_suffix}",
                xy=agent.position,
                xytext=(12, 16 + index * 11),
                textcoords="offset points",
                color=status_color,
                fontsize=5.2,
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": _BACKGROUND,
                    "edgecolor": status_color,
                    "linewidth": 0.7,
                    "alpha": 0.94,
                },
                zorder=36,
            )
            _tag(
                status_artist,
                f"scene:v2:agent:{agent.public_agent_id}:status:{status.status_id}",
            )
    if options.show_modifiers:
        visible_modifiers = tuple(
            row for row in agent.aura_modifiers if row.multiplier != 1.0
        )
        for index, modifier in enumerate(visible_modifiers):
            modifier_artist = axes.annotate(
                f"{modifier.aura_id} x{_format_display_number(modifier.multiplier)}",
                xy=agent.position,
                xytext=(-12, 16 + index * 11),
                textcoords="offset points",
                color=_BASIC,
                fontsize=5.2,
                ha="right",
                va="bottom",
                zorder=36,
            )
            _tag(
                modifier_artist,
                f"scene:v2:agent:{agent.public_agent_id}:aura:{modifier.aura_id}",
            )
    del team_token


def _draw_v2_wave_clocks(
    axes: _AxesLike,
    scene: BattlefieldSceneV2,
) -> None:
    label = " · ".join(
        f"Team {wave.team_id} wave {wave.countdown_steps}/{wave.period_steps}"
        for wave in scene.respawn_waves
    )
    artist = axes.text(
        0.99,
        0.99,
        label,
        transform=axes.transAxes,
        color=_MUTED,
        fontsize=6.5,
        ha="right",
        va="top",
        zorder=60,
    )
    _tag(artist, f"scene:v2:waves:{scene.frame_id}")


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
        rectangle=cast(_PatchFactory, patches.Rectangle),
        wedge=cast(_PatchFactory, patches.Wedge),
    )


def _tag(artist: object, gid: str) -> object:
    cast(_ArtistLike, artist).set_gid(gid)
    return artist


def _draw_map(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    scene: BattlefieldScene,
) -> None:
    background = parts.rectangle(
        (0.0, 0.0),
        scene.map.width,
        scene.map.height,
        facecolor=_BATTLEFIELD,
        edgecolor="#49617F",
        linewidth=2.0,
        zorder=0,
    )
    axes.add_patch(_tag(background, "scene:map:field"))


def _draw_fields(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    scene: BattlefieldSceneV1,
    options: SceneRenderOptions,
) -> None:
    for aura in scene.aura_fields:
        color = _aura_color(aura.token_id)
        patch = parts.circle(
            aura.center,
            aura.radius,
            facecolor=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.12,
            zorder=2,
        )
        axes.add_patch(
            _tag(
                patch,
                f"scene:aura:{aura.source_global_slot}:{aura.token_id}",
            )
        )
    if not options.show_ranges:
        return
    agents = {agent.global_slot: agent for agent in scene.agents}
    for range_record in scene.ranges:
        owner = agents.get(range_record.global_slot)
        owner_class_color = (
            _MUTED
            if owner is None
            else _CLASS_COLORS.get(
                class_token_from_id(owner.class_id).token_id,
                _MUTED,
            )
        )
        edgecolor, linestyle = {
            "observation": (_TEXT, ":"),
            "basic": (owner_class_color, "--"),
            "ultimate": (_ULTIMATE, "-."),
        }[range_record.kind]
        patch = parts.circle(
            range_record.center,
            range_record.radius,
            facecolor="none",
            edgecolor=edgecolor,
            linewidth=1.0,
            linestyle=linestyle,
            alpha=0.78,
            zorder=3,
        )
        axes.add_patch(
            _tag(
                patch,
                f"scene:range:{range_record.global_slot}:{range_record.kind}",
            )
        )


def _draw_pending_route(
    axes: _AxesLike,
    scene: BattlefieldSceneV1,
) -> None:
    route = scene.pending_route
    if route is None:
        return
    color = (_BASIC, _ULTIMATE)[route.lane] if route.legal else _DAMAGE
    artist = axes.annotate(
        f"PENDING {'0/B' if route.lane == 0 else '1/U'}",
        xy=route.target_anchor,
        xytext=route.source_anchor,
        color=color,
        fontsize=7,
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linestyle": "--",
            "linewidth": 1.5,
        },
        zorder=18,
    )
    _tag(
        artist,
        (
            f"scene:pending:{route.source_global_slot}:"
            f"{route.target_global_slot}:lane:{route.lane}"
        ),
    )


def _draw_obstacles(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    scene: BattlefieldScene,
) -> None:
    for obstacle in scene.map.obstacles:
        if obstacle.kind == "pillar":
            assert obstacle.radius is not None
            patch = parts.circle(
                obstacle.center,
                obstacle.radius,
                facecolor="#334155",
                edgecolor="#94A3B8",
                linewidth=1.5,
                zorder=20,
            )
        else:
            assert obstacle.width is not None
            assert obstacle.height is not None
            half_width = obstacle.width / 2.0
            half_height = obstacle.height / 2.0
            local_corners = (
                (-half_width, -half_height),
                (half_width, -half_height),
                (half_width, half_height),
                (-half_width, half_height),
            )
            cosine = cos(obstacle.theta)
            sine = sin(obstacle.theta)
            corners = tuple(
                (
                    obstacle.center[0] + x * cosine - y * sine,
                    obstacle.center[1] + x * sine + y * cosine,
                )
                for x, y in local_corners
            )
            patch = parts.polygon(
                corners,
                closed=True,
                facecolor="#334155",
                edgecolor="#94A3B8",
                linewidth=1.5,
                zorder=20,
            )
        axes.add_patch(_tag(patch, f"scene:obstacle:{obstacle.obstacle_id}:shape"))


def _draw_agents(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    scene: BattlefieldSceneV1,
    options: SceneRenderOptions,
) -> None:
    selection = scene.selection
    for agent in scene.agents:
        show_identity = options.show_agent_ids or (
            selection is not None
            and agent.global_slot
            in (
                selection.controlled_global_slot,
                selection.selected_global_slot,
            )
        )
        _draw_agent_body(
            axes,
            parts,
            agent,
            options,
            show_identity=show_identity,
        )
        if (
            selection is not None
            and agent.global_slot == selection.controlled_global_slot
        ):
            halo = parts.circle(
                agent.position,
                agent.radius * 1.32,
                facecolor="none",
                edgecolor=_TEXT,
                linewidth=2.5,
                zorder=31,
            )
            axes.add_patch(
                _tag(
                    halo,
                    f"scene:selection:controlled:{agent.global_slot}",
                )
            )
        if (
            selection is not None
            and agent.global_slot == selection.selected_global_slot
        ):
            reticle = parts.circle(
                agent.position,
                agent.radius * 1.48,
                facecolor="none",
                edgecolor=_TARGET,
                linewidth=2.2,
                linestyle="--",
                zorder=32,
            )
            axes.add_patch(
                _tag(
                    reticle,
                    f"scene:selection:target:{agent.global_slot}",
                )
            )


def _draw_agent_body(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    agent: AgentSceneV1,
    options: SceneRenderOptions,
    *,
    show_identity: bool,
) -> None:
    class_token = class_token_from_id(agent.class_id)
    team_token = team_token_from_id(agent.team_id)
    class_color = _CLASS_COLORS.get(class_token.token_id, _MUTED)
    team_color, team_line_style = {
        "team_a": (_TEAM_A, "-"),
        "team_b": (_TEAM_B, "-"),
    }.get(team_token.token_id, (_MUTED, ":"))
    body = parts.circle(
        agent.position,
        agent.radius,
        facecolor=class_color,
        edgecolor=_BACKGROUND,
        linewidth=1.0,
        alpha=0.95 if agent.alive else 0.42,
        zorder=24,
    )
    axes.add_patch(_tag(body, f"scene:agent:{agent.global_slot}:body"))
    team_ring = parts.circle(
        agent.position,
        agent.radius,
        facecolor="none",
        edgecolor=team_color,
        linewidth=3.0,
        linestyle=team_line_style,
        alpha=0.95 if agent.alive else 0.42,
        zorder=25,
    )
    axes.add_patch(_tag(team_ring, f"scene:agent:{agent.global_slot}:team"))
    if team_token.token_id == "team_b":
        marker = parts.polygon(
            (
                (
                    agent.position[0] + agent.radius * 0.52,
                    agent.position[1] - agent.radius * 0.28,
                ),
                (
                    agent.position[0] + agent.radius * 0.82,
                    agent.position[1],
                ),
                (
                    agent.position[0] + agent.radius * 0.52,
                    agent.position[1] + agent.radius * 0.28,
                ),
            ),
            closed=False,
            facecolor="none",
            edgecolor=team_color,
            linewidth=1.5,
            zorder=30,
        )
        axes.add_patch(
            _tag(
                marker,
                f"scene:agent:{agent.global_slot}:team-marker",
            )
        )

    health_track = parts.wedge(
        agent.position,
        agent.radius * 0.86,
        0.0,
        360.0,
        width=max(agent.radius * 0.10, 0.02),
        facecolor="#080D18",
        edgecolor="none",
        zorder=26,
    )
    axes.add_patch(_tag(health_track, f"scene:agent:{agent.global_slot}:health:track"))
    health_fraction = (
        0.0
        if agent.max_health <= 0.0
        else min(max(agent.current_health / agent.max_health, 0.0), 1.0)
    )
    if health_fraction > 0.0:
        health = parts.wedge(
            agent.position,
            agent.radius * 0.86,
            90.0,
            90.0 + 360.0 * health_fraction,
            width=max(agent.radius * 0.10, 0.02),
            facecolor=_HEALING,
            edgecolor="none",
            zorder=27,
        )
        axes.add_patch(_tag(health, f"scene:agent:{agent.global_slot}:health:value"))

    class_artist = axes.text(
        agent.position[0],
        agent.position[1],
        class_token.fallback,
        color=_TEXT,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=28,
    )
    _tag(class_artist, f"scene:agent:{agent.global_slot}:class")
    if show_identity:
        identity = axes.annotate(
            f"id_{agent.global_slot}",
            xy=agent.position,
            xytext=(0, -16),
            textcoords="offset points",
            color=_TEXT,
            fontsize=6,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=29,
        )
        _tag(identity, f"scene:agent:{agent.global_slot}:identity")
    if options.show_statuses:
        for index, status in enumerate(agent.statuses):
            token = lookup_status_token(status.token_id)
            source_class = class_token_from_id(status.source_class_id)
            status_color = _CLASS_COLORS.get(source_class.token_id, _MUTED)
            column = index % 3
            row = index // 3
            artist = axes.annotate(
                f"{token.glyph}{source_class.fallback} {status.duration}",
                xy=agent.position,
                xytext=(12 + column * 35, 16 + row * 13),
                textcoords="offset points",
                color=status_color,
                fontsize=5.5,
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": _BACKGROUND,
                    "edgecolor": status_color,
                    "linewidth": 0.8,
                    "alpha": 0.94,
                },
                zorder=36,
            )
            _tag(
                artist,
                (f"scene:agent:{agent.global_slot}:status:{status.token_id}"),
            )
    if options.show_modifiers:
        for index, modifier in enumerate(agent.modifiers):
            token = lookup_modifier_token(modifier.token_id)
            artist = axes.annotate(
                (f"{token.short_label} x{_format_display_number(modifier.multiplier)}"),
                xy=agent.position,
                xytext=(12 + (index % 2) * 42, -18 - (index // 2) * 13),
                textcoords="offset points",
                color=_TEXT,
                fontsize=5.5,
                ha="left",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": _BACKGROUND,
                    "edgecolor": _BASIC,
                    "linestyle": "--",
                    "linewidth": 0.8,
                    "alpha": 0.94,
                },
                zorder=35,
            )
            _tag(
                artist,
                (f"scene:agent:{agent.global_slot}:modifier:{modifier.token_id}"),
            )


def _draw_selected_legality(
    axes: _AxesLike,
    scene: BattlefieldSceneV1,
) -> None:
    legality = scene.selected_legality
    if legality is None:
        return
    target = next(
        (
            agent
            for agent in scene.agents
            if agent.global_slot == legality.target_global_slot
        ),
        None,
    )
    if target is None:
        return
    for lane, available in (
        (0, legality.lane_0_available),
        (1, legality.lane_1_available),
    ):
        armed = legality.armed_lane == lane
        color = (_BASIC, _ULTIMATE)[lane] if available else _UNAVAILABLE
        artist = axes.annotate(
            (
                f"{'0/B' if lane == 0 else '1/U'} "
                f"{'1' if available else '0'}"
                f"{' ARMED' if armed else ''}"
            ),
            xy=target.position,
            xytext=(-28 + lane * 34, -28),
            textcoords="offset points",
            color=color,
            fontsize=6,
            fontweight="bold" if armed else "normal",
            ha="center",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": _BACKGROUND,
                "edgecolor": color,
                "linestyle": "-" if available else "--",
                "linewidth": 1.0,
                "alpha": 0.95,
            },
            zorder=38,
        )
        _tag(
            artist,
            (
                f"scene:legality:{legality.controlled_global_slot}:"
                f"{legality.target_global_slot}:lane:{lane}"
            ),
        )


def _draw_observer_visibility(
    axes: _AxesLike,
    scene: BattlefieldSceneV1,
) -> None:
    agents = {agent.global_slot: agent for agent in scene.agents}
    for record in scene.observer_visibility:
        candidate = agents.get(record.candidate_global_slot)
        if candidate is None:
            continue
        artist = axes.annotate(
            "V" if record.visible else "H",
            xy=candidate.position,
            xytext=(11, 11),
            textcoords="offset points",
            color=_HEALING if record.visible else _UNAVAILABLE,
            fontsize=6,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=39,
        )
        _tag(
            artist,
            (
                f"scene:visibility:{record.observer_global_slot}:"
                f"{record.candidate_global_slot}"
            ),
        )


def _draw_events(
    axes: _AxesLike,
    scene: BattlefieldSceneV1,
    event_batch: VisualEventBatchV1,
) -> None:
    activation_ordinal = 0
    for ordinal, event in enumerate(event_batch.events):
        fallback = (
            0.25,
            max(scene.map.height - 0.35 - ordinal * 0.24, 0.25),
        )
        if type(event) is AcceptedActivationEventV1:
            gid = f"event:accepted_activation:{event.event_id}"
            token = lookup_activation_token(event.token_id)
            source = event.source_anchor or fallback
            target = event.target_anchor or source
            target_label = (
                f"id_{event.target_global_slot}"
                if event.target_global_slot is not None
                else "source-local"
                if event.target_disclosure == "target_none"
                else event.target_disclosure
            )
            color = (
                _HEALING
                if event.token_id in ("basic_heal", "holy_word")
                else _CLASS_COLORS.get(
                    class_token_from_id(event.source_class_id).token_id,
                    _DAMAGE,
                )
                if event.token_id == "basic_damage"
                else _ULTIMATE
            )
            radius = ((activation_ordinal % 5) - 2) * 0.08
            activation_ordinal += 1
            arrowprops = (
                None
                if event.target_anchor is None or event.source_anchor is None
                else {
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 1.6,
                    "connectionstyle": f"arc3,rad={radius:g}",
                }
            )
            if arrowprops is None:
                artist = axes.annotate(
                    f"{token.short_label} → {target_label}",
                    xy=source,
                    xytext=(0, 22 + (activation_ordinal % 3) * 11),
                    textcoords="offset points",
                    color=color,
                    fontsize=6,
                    ha="center",
                    va="bottom",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": _BACKGROUND,
                        "edgecolor": color,
                        "linewidth": 0.8,
                        "alpha": 0.92,
                    },
                    zorder=45,
                )
            else:
                artist = axes.annotate(
                    "",
                    xy=target,
                    xytext=source,
                    arrowprops=arrowprops,
                    zorder=45,
                )
                midpoint = (
                    (source[0] + target[0]) / 2.0,
                    (source[1] + target[1]) / 2.0,
                )
                label = axes.annotate(
                    token.short_label,
                    xy=midpoint,
                    xytext=(0, 7 + (activation_ordinal % 3) * 9),
                    textcoords="offset points",
                    color=color,
                    fontsize=5.5,
                    ha="center",
                    va="bottom",
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": _BACKGROUND,
                        "edgecolor": color,
                        "linewidth": 0.7,
                        "alpha": 0.88,
                    },
                    zorder=46,
                )
                _tag(label, f"scene:event-label:{event.event_id}")
                impact_semantic = (
                    ("\N{MINUS SIGN}", _DAMAGE)
                    if event.token_id
                    in (
                        "basic_damage",
                        "warrior_charge",
                        "hunter_trap",
                        "rogue_poison",
                    )
                    else ("+", _HEALING)
                    if event.token_id in ("basic_heal", "holy_word")
                    else None
                )
                if impact_semantic is not None:
                    symbol, impact_color = impact_semantic
                    recipient = next(
                        (
                            agent
                            for agent in scene.agents
                            if agent.global_slot == event.target_global_slot
                        ),
                        None,
                    )
                    source_to_target = (
                        target[0] - source[0],
                        target[1] - source[1],
                    )
                    source_to_target_distance = hypot(*source_to_target)
                    impact_anchor = target
                    if recipient is not None and source_to_target_distance > 0.0:
                        perimeter_distance = recipient.radius * 1.25
                        impact_anchor = (
                            target[0]
                            - source_to_target[0]
                            / source_to_target_distance
                            * perimeter_distance,
                            target[1]
                            - source_to_target[1]
                            / source_to_target_distance
                            * perimeter_distance,
                        )
                    impact = axes.annotate(
                        symbol,
                        xy=impact_anchor,
                        xytext=(0, 0),
                        textcoords="offset points",
                        color=impact_color,
                        fontsize=7,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        bbox={
                            "boxstyle": "circle,pad=0.12",
                            "facecolor": _BACKGROUND,
                            "edgecolor": impact_color,
                            "linewidth": 1.0,
                            "alpha": 0.96,
                        },
                        zorder=47,
                    )
                    _tag(impact, f"scene:event-impact:{event.event_id}")
        elif type(event) is NetHealthEventV1:
            gid = f"event:net_health:{event.event_id}"
            anchor = event.recipient_anchor or fallback
            if event.net_delta == 0:
                health_label = "HP unchanged"
            else:
                sign = "+" if event.net_delta > 0 else ""
                health_label = f"NET {sign}{_format_display_number(event.net_delta)}"
            artist = axes.annotate(
                health_label,
                xy=anchor,
                xytext=(0, -24 - (ordinal % 3) * 11),
                textcoords="offset points",
                color=(
                    _DAMAGE
                    if event.net_delta < 0
                    else _HEALING
                    if event.net_delta > 0
                    else _TEXT
                ),
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="top",
                zorder=48,
            )
        elif type(event) is ChargeDisplacementEventV1:
            gid = f"event:charge_displacement:{event.event_id}"
            label = (
                "Charge + ordinary movement · realized endpoint change"
                if event.path_kind == "combined_charge_and_movement"
                else "Charge · realized endpoint change"
            )
            artist = axes.annotate(
                "",
                xy=event.end,
                xytext=event.start,
                arrowprops={
                    "arrowstyle": "->",
                    "color": _ULTIMATE,
                    "linestyle": "--",
                    "linewidth": 1.4,
                },
                zorder=44,
            )
            midpoint = (
                (event.start[0] + event.end[0]) / 2.0,
                (event.start[1] + event.end[1]) / 2.0,
            )
            label_artist = axes.annotate(
                label,
                xy=midpoint,
                xytext=(0, -14 - (ordinal % 2) * 9),
                textcoords="offset points",
                color=_ULTIMATE,
                fontsize=5.5,
                ha="center",
                va="top",
                zorder=45,
            )
            _tag(label_artist, f"scene:event-label:{event.event_id}")
        elif type(event) is StatusLifecycleEventV1:
            gid = f"event:status_lifecycle:{event.event_id}"
            status = lookup_status_token(event.token_id)
            lifecycle = lookup_lifecycle_token(event.change)
            anchor = event.recipient_anchor or fallback
            artist = axes.annotate(
                (
                    f"{status.short_label} {lifecycle.short_label} "
                    f"{event.duration_before}→{event.duration_after}"
                ),
                xy=anchor,
                xytext=(8, -20 - (ordinal % 3) * 11),
                textcoords="offset points",
                color=_TEXT,
                fontsize=5.5,
                ha="left",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": _BACKGROUND,
                    "edgecolor": _ULTIMATE,
                    "linewidth": 0.8,
                    "alpha": 0.92,
                },
                zorder=47,
            )
        elif type(event) is RejectedActionEventV1:
            gid = f"event:rejected_action:{event.event_id}"
            source = event.actor_anchor or fallback
            target = event.target_anchor or source
            arrowprops = (
                None
                if event.actor_anchor is None or event.target_anchor is None
                else {
                    "arrowstyle": "-|>",
                    "color": _DAMAGE,
                    "linestyle": "--",
                    "linewidth": 1.2,
                }
            )
            label = (
                f"REJECTED {event.component} "
                f"M={int(event.movement_mask_value)} "
                f"P={int(event.pair_mask_value)}"
            )
            artist = axes.annotate(
                "" if arrowprops is not None else label,
                xy=target,
                xytext=source if arrowprops is not None else (0, 22),
                textcoords=None if arrowprops is not None else "offset points",
                color=_DAMAGE,
                fontsize=5.5,
                ha="center",
                va="center",
                arrowprops=arrowprops,
                zorder=46,
            )
            if arrowprops is not None:
                midpoint = (
                    (source[0] + target[0]) / 2.0,
                    (source[1] + target[1]) / 2.0,
                )
                label_artist = axes.annotate(
                    label,
                    xy=midpoint,
                    xytext=(0, 7),
                    textcoords="offset points",
                    color=_DAMAGE,
                    fontsize=5.5,
                    ha="center",
                    va="bottom",
                    zorder=47,
                )
                _tag(label_artist, f"scene:event-label:{event.event_id}")
        else:
            raise AssertionError(f"unknown visual event type: {type(event).__name__}")
        _tag(artist, gid)


def _draw_audience_badge(
    axes: _AxesLike,
    scene: BattlefieldScene,
) -> None:
    artist = axes.text(
        0.01,
        0.99,
        scene.audience_badge,
        transform=axes.transAxes,
        color=_TEXT,
        fontsize=7,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": _BACKGROUND,
            "edgecolor": _TEAM_A if scene.audience == "researcher" else _BASIC,
            "linewidth": 1.0,
            "alpha": 0.96,
        },
        zorder=60,
    )
    _tag(artist, f"scene:audience:{scene.audience}")


def _style_axes(
    axes: _AxesLike,
    scene: BattlefieldScene,
) -> None:
    axes.set_facecolor(_BACKGROUND)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim(0.0, scene.map.width)
    axes.set_ylim(0.0, scene.map.height)
    axes.set_title(
        "MARL-BattleGrounds · Visual Debugger and Analyzer · Static scene",
        color=_TEXT,
        fontsize=10,
    )
