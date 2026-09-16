"""Draw authorized scene and event records with optional Matplotlib.

render_scene_geometry creates a figure, draw_scene_geometry replaces an
existing axes, and redraw_scene_geometry reuses a RenderResult. V1 supports
legacy debugger scenes; V2 paints canonical analyzer records. The caller
owns audience authorization, matching scene/event versions, and figure
saving, display and cleanup.

These host helpers only paint supplied facts. They do not run simulator
rules, recover hidden information, infer causal events or animate time.
Matplotlib is imported lazily when drawing starts, so importing rendering
does not require its optional visualization dependency.
"""

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
    VisualAgentAnchorV2,
    VisualEventBatchV1,
    VisualEventBatchV2,
    VisualEventV2,
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
type VisualEventBatch = VisualEventBatchV1 | VisualEventBatchV2


def _aura_color(token_id: str) -> str:
    """Choose the legacy aura color for token_id.

    Return a hex color for Mage amplification or Warrior mitigation, or the
    muted color for any other ID. This lookup does not validate token text.
    """
    return {
        "mage_amplification": "#22D3EE",
        "warrior_mitigation": "#D18B47",
    }.get(token_id, _MUTED)


def _format_display_number(value: float) -> str:
    """Format value to at most two decimal places for a label.

    Strip trailing zeroes and a trailing point; change the rounded text -0 to
    0. The caller supplies a finite scalar. No scientific value is changed.
    """
    formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if formatted == "-0" else formatted


class _ArtistLike(Protocol):
    """Describe the Matplotlib artist method used for stable output IDs.

    This typing-only interface does not implement or validate a renderer.
    """

    def set_gid(self, gid: str) -> object:
        """Set the artist's string group ID to gid.

        The concrete Matplotlib artist owns the mutation. Its return is ignored.
        """
        ...


class _AxesLike(Protocol):
    """Describe the Matplotlib axes operations used by the painter.

    Methods mutate the caller's axes or add artists. This protocol adds no
    runtime adapter or validation; keyword arguments pass to Matplotlib.
    """

    transAxes: object  # noqa: N815 - Matplotlib public attribute.
    """Matplotlib transform from axes-relative coordinates to display coordinates."""

    def add_patch(self, patch: object) -> object:
        """Add patch to these axes and return Matplotlib's result."""
        ...

    def annotate(
        self,
        text: str,
        xy: tuple[float, float],
        **kwargs: object,
    ) -> object:
        """Add text anchored at world point xy, using Matplotlib kwargs.

        Return the annotation artist; kwargs may set text offsets, colors and arrows.
        """
        ...

    def clear(self) -> object:
        """Remove the existing contents of these axes; return is ignored."""
        ...

    def set_aspect(self, aspect: str, adjustable: str | None = None) -> object:
        """Set the aspect ratio and optional adjustable axes policy.

        The painter passes equal and box so both world axes use the same scale.
        The concrete method's return is ignored.
        """
        ...

    def set_facecolor(self, color: str) -> object:
        """Set the axes background to color; return is ignored."""
        ...

    def set_title(self, label: str, **kwargs: object) -> object:
        """Set title label using Matplotlib styling kwargs.

        The concrete method's return is ignored.
        """
        ...

    def set_xlim(self, left: float, right: float) -> object:
        """Set horizontal world bounds to left and right; ignore return."""
        ...

    def set_ylim(self, bottom: float, top: float) -> object:
        """Set vertical world bounds to bottom and top; ignore return."""
        ...

    def text(self, x: float, y: float, s: str, **kwargs: object) -> object:
        """Add string s at x, y using Matplotlib styling kwargs.

        Coordinates use world units unless kwargs supplies a transform. Return the
        text artist so the caller can assign its stable group ID.
        """
        ...


class _PyplotLike(Protocol):
    """Describe the figure-and-axes factory needed by render_scene_geometry."""

    def subplots(self) -> tuple[object, _AxesLike]:
        """Create and return one figure and its axes, in that order."""
        ...


_PatchFactory = Callable[..., object]


@dataclass(frozen=True, slots=True)
class _MatplotlibParts:
    """Keep the lazily imported Matplotlib factories used by the painter.

    This frozen, slotted record holds modules/callables, not a scene or figure.
    Construction performs no validation.
    """

    pyplot: _PyplotLike
    """Imported plotting module used to create a figure and axes."""
    circle: _PatchFactory
    """Factory for circular patch artists."""
    polygon: _PatchFactory
    """Factory for polygon patch artists."""
    rectangle: _PatchFactory
    """Factory for rectangular patch artists."""
    wedge: _PatchFactory
    """Factory for wedge patch artists, including health rings."""


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Keep the figure and axes owned by one static scene rendering.

    Attributes
    ----------
    figure : object
        Matplotlib Figure created by render_scene_geometry.
    axes : object
        Matplotlib Axes receiving the scene's artists.

    Notes
    -----
    The record is frozen and slotted; the contained Matplotlib objects remain
    mutable. The caller owns saving, showing and closing the figure. Redrawing
    clears these axes and returns the same RenderResult. Construction does not
    validate arbitrary objects passed directly to this record.
    """

    figure: object
    """Mutable Matplotlib figure; the caller owns saving, showing and closing it."""
    axes: object
    """Mutable Matplotlib axes that redraw_scene_geometry clears and repaints."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneRenderOptions:
    """Choose which optional labels and overlays a static render draws.

    All fields are keyword-only Python bools. This frozen, slotted record changes
    presentation only. It neither changes the scene nor authorizes more data.

    Attributes
    ----------
    show_agent_ids : bool, default False
        Show every agent ID. Selected and controlled IDs are shown regardless.
    show_ranges : bool, default True
        Show observation, Basic and Ultimate range outlines; aura fields remain.
    show_statuses : bool, default True
        Show agent status labels and remaining durations.
    show_modifiers : bool, default True
        Show supplied modifier labels. V2 omits aura multipliers equal to 1.0.
    show_observer_visibility : bool, default False
        Show supplied legacy V1 visibility labels. V2 does not use this switch.
    show_events : bool, default True
        Draw supplied event cues. False still enforces the V2 scene/batch join.

    Raises
    ------
    ValueError
        Any option is not an exact Python bool.
    """

    show_agent_ids: bool = False
    """Show all agent IDs; default False. Selection still reveals selected IDs."""
    show_ranges: bool = True
    """Draw range outlines; default True. Aura fields draw independently."""
    show_statuses: bool = True
    """Draw status labels and durations; default True."""
    show_modifiers: bool = True
    """Draw supplied modifier labels; default True."""
    show_observer_visibility: bool = False
    """Draw legacy V1 observer-visibility labels; default False."""
    show_events: bool = True
    """Draw supplied event cues; default True. V2 joins are always checked."""

    def __post_init__(self) -> None:
        """Reject any option that is not an exact Python bool.

        Dataclass construction calls this host check once. Raise ValueError on a
        bad field; otherwise return None without changing the supplied switches.
        """
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
    event_batch: VisualEventBatch | None = None,
    options: SceneRenderOptions | None = None,
) -> None:
    """Replace the contents of existing axes with one authorized scene.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Caller-owned axes to clear and draw into. Must supply the Matplotlib
        methods used here; no separate runtime type check is performed.

    scene : BattlefieldSceneV1 or BattlefieldSceneV2
        Validated host scene already restricted to its declared audience. The
        renderer does not grant permission or remove hidden data. Coordinates
        and radii use world units; wall angles use radians.
    event_batch : VisualEventBatchV1 or VisualEventBatchV2 or None, optional
        Latest events for the scene, or None to omit event cues. Supply the same
        schema version as the scene. For V2, transition ID, successor frame ID
        and ordered event IDs must match the scene, even when events are hidden.
    options : SceneRenderOptions or None, optional
        Presentation switches. None uses SceneRenderOptions() defaults.

    Returns
    -------
    None
        Artists are added to axes. Scene, event batch and options are unchanged.

    Raises
    ------
    TypeError
        Scene, event batch or options has the wrong exact record type.
    ValueError
        A V2 scene receives a V1 batch or a V2 batch with mismatched join IDs.
    ImportError
        The optional Matplotlib dependency cannot be imported. Install the viz
        extra before using these drawing helpers.

    Notes
    -----
    This is host drawing, outside JAX compilation and simulator transitions.
    It does not show, save, close or animate a figure. Draws may partially change
    axes before a Matplotlib error. Legacy V1 does not check event join IDs;
    a wrong-version batch may fail later during event drawing. Supply a matching
    batch rather than relying on the legacy path to validate it.
    """
    if type(scene) not in (BattlefieldSceneV1, BattlefieldSceneV2):
        raise TypeError("scene must be BattlefieldSceneV1 or BattlefieldSceneV2.")
    if event_batch is not None and type(event_batch) not in (
        VisualEventBatchV1,
        VisualEventBatchV2,
    ):
        raise TypeError("event_batch must be VisualEventBatchV1/V2 or None.")
    if options is not None and type(options) is not SceneRenderOptions:
        raise TypeError("options must be SceneRenderOptions or None.")
    render_options = options or SceneRenderOptions()
    parts = _load_matplotlib()
    typed_axes = cast(_AxesLike, axes)

    if type(scene) is BattlefieldSceneV2:
        if event_batch is not None and type(event_batch) is not VisualEventBatchV2:
            raise ValueError(
                "BattlefieldSceneV2 does not accept legacy VisualEventBatchV1."
            )
        typed_v2_batch = event_batch
        if typed_v2_batch is not None and (
            typed_v2_batch.transition_id != scene.incoming_transition_id
            or typed_v2_batch.successor_frame_id != scene.frame_id
            or tuple(event.event_id for event in typed_v2_batch.events)
            != scene.incoming_event_ids
        ):
            raise ValueError("VisualEventBatchV2 must join its BattlefieldSceneV2.")
        _draw_scene_v2(
            typed_axes,
            parts,
            scene,
            render_options,
            typed_v2_batch,
        )
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
        _draw_events(
            typed_axes,
            legacy_scene,
            cast(VisualEventBatchV1, event_batch),
        )
    _draw_audience_badge(typed_axes, legacy_scene)
    _style_axes(typed_axes, legacy_scene)


def render_scene_geometry(
    scene: BattlefieldScene,
    *,
    event_batch: VisualEventBatch | None = None,
    options: SceneRenderOptions | None = None,
) -> RenderResult:
    """Create a new Matplotlib figure and draw one authorized scene.

    Parameters
    ----------

    scene : BattlefieldSceneV1 or BattlefieldSceneV2
        Validated host scene already restricted to its declared audience. The
        renderer does not grant permission or remove hidden data. Coordinates
        and radii use world units; wall angles use radians.
    event_batch : VisualEventBatchV1 or VisualEventBatchV2 or None, optional
        Latest events for the scene, or None to omit event cues. Supply the same
        schema version as the scene. For V2, transition ID, successor frame ID
        and ordered event IDs must match the scene, even when events are hidden.
    options : SceneRenderOptions or None, optional
        Presentation switches. None uses SceneRenderOptions() defaults.

    Returns
    -------
    RenderResult
        New figure and axes. The caller owns showing, saving and closing them.

    Raises
    ------
    TypeError
        Scene, event batch or options has the wrong exact record type.
    ValueError
        A V2 scene receives a V1 batch or a V2 batch with mismatched join IDs.
    ImportError
        The optional Matplotlib dependency cannot be imported. Install the viz
        extra before using these drawing helpers.

    Notes
    -----
    This host helper allocates a figure, calls draw_scene_geometry and sets its
    background. It does not show or save output. Validation occurs after figure
    creation; a drawing failure does not automatically close the new figure.
    Use redraw_scene_geometry to reuse an existing figure.
    """
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
    event_batch: VisualEventBatch | None = None,
    options: SceneRenderOptions | None = None,
) -> RenderResult:
    """Replace an existing render while retaining its figure and result.

    Parameters
    ----------

    scene : BattlefieldSceneV1 or BattlefieldSceneV2
        Validated host scene already restricted to its declared audience. The
        renderer does not grant permission or remove hidden data. Coordinates
        and radii use world units; wall angles use radians.
    event_batch : VisualEventBatchV1 or VisualEventBatchV2 or None, optional
        Latest events for the scene, or None to omit event cues. Supply the same
        schema version as the scene. For V2, transition ID, successor frame ID
        and ordered event IDs must match the scene, even when events are hidden.
    options : SceneRenderOptions or None, optional
        Presentation switches. None uses SceneRenderOptions() defaults.

    result : RenderResult
        Existing result whose axes will be cleared and redrawn.

    Returns
    -------
    RenderResult
        The identical result object, now containing the new axes contents.

    Raises
    ------
    TypeError
        Scene, event batch or options has the wrong exact record type.
    ValueError
        A V2 scene receives a V1 batch or a V2 batch with mismatched join IDs.
    ImportError
        The optional Matplotlib dependency cannot be imported. Install the viz
        extra before using these drawing helpers.

    Notes
    -----
    No figure is created, shown, saved or closed. Errors can leave axes partly
    redrawn. Scene and event records remain unchanged.
    """
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
    event_batch: VisualEventBatchV2 | None,
) -> None:
    """Clear axes and paint the validated V2 scene in layer order.

    parts supplies Matplotlib factories; options chooses optional ranges,
    labels and events. event_batch is None or a batch already joined by the
    public caller. Draw pads, living aura fields, bodies, clocks and direct
    event cues without inferring missing events. Mutate axes and return None.
    """
    axes.clear()
    _draw_map(axes, parts, scene)
    _draw_v2_spawn_pads(axes, parts, scene)
    _draw_v2_aura_fields(axes, parts, scene)
    if options.show_ranges:
        _draw_v2_ranges(axes, parts, scene)
    _draw_obstacles(axes, parts, scene)
    _draw_v2_agents(axes, parts, scene, options)
    _draw_v2_wave_clocks(axes, scene)
    if options.show_events and event_batch is not None:
        _draw_events_v2(axes, event_batch)
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


def _v2_primary_anchor(event: VisualEventV2) -> VisualAgentAnchorV2 | None:
    """Select the first direct agent anchor stored on event.

    Check recipient, agent, actor, source, then end anchor fields. Return the
    first exact VisualAgentAnchorV2 or None for an event without one. Do not
    join other events or reconstruct hidden coordinates.
    """
    for field_name in (
        "recipient_anchor",
        "agent_anchor",
        "actor_anchor",
        "source_anchor",
        "end_anchor",
    ):
        anchor = getattr(event, field_name, None)
        if type(anchor) is VisualAgentAnchorV2:
            return anchor
    return None


def _draw_events_v2(
    axes: _AxesLike,
    event_batch: VisualEventBatchV2,
) -> None:
    """Add one static label per canonical event to axes in batch order.

    Use event_batch's direct anchor when present, otherwise an axes-relative
    fallback. Tag each artist with the event ID. No trajectory is inferred;
    this helper mutates axes and returns None.
    """
    for event in event_batch.events:
        anchor = _v2_primary_anchor(event)
        label = event.event_type.replace("_", " ").upper()
        color = {
            "action_rejected": _DAMAGE,
            "source_damage_output": _DAMAGE,
            "recipient_health_resolution": _DAMAGE,
            "health_regenerated": _HEALING,
            "source_healing_output": _HEALING,
            "ability_activated": _ULTIMATE,
            "agent_respawned": _HEALING,
        }.get(event.event_type, _TEXT)
        if anchor is None:
            artist = axes.text(
                0.01,
                max(0.02, 0.96 - event.ordinal * 0.028),
                label,
                transform=axes.transAxes,
                color=color,
                fontsize=5.0,
                ha="left",
                va="top",
                zorder=80,
            )
        else:
            artist = axes.annotate(
                label,
                xy=anchor.position,
                xytext=(5, 5 + (event.ordinal % 3) * 7),
                textcoords="offset points",
                color=color,
                fontsize=4.8,
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": _BACKGROUND,
                    "edgecolor": color,
                    "linewidth": 0.5,
                    "alpha": 0.88,
                },
                zorder=80,
            )
        _tag(artist, f"scene:v2:event:{event.event_id}")


def _draw_v2_spawn_pads(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    scene: BattlefieldSceneV2,
) -> None:
    """Add every supplied V2 spawn-pad marker to axes.

    parts supplies circle factories. Read scene's fixed positions and team IDs;
    tag each marker by assigned public agent ID. Return None.
    """
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
    """Add V2 aura circles for sources marked alive in scene.

    parts supplies circles. Skip sources marked dead; use the supplied center
    and radius without recomputing recipients or effects. Mutate axes and
    return None.
    """
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
    """Add positive-radius V2 range outlines from scene to axes.

    parts supplies circles. Color outlines by observation, Basic or Ultimate;
    zero-radius entries draw nothing. This is a display of supplied ranges,
    not a target-legality calculation. Return None.
    """
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
    """Paint V2 agents and selection markers from scene onto axes.

    parts supplies patches; options controls labels. Selected and controlled
    agents show identity even when all-ID labels are off. Both markers can
    appear on the same agent. Return None without changing scene.
    """
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
    """Paint one V2 agent, health ring, clocks and optional labels.

    axes receives artists built with parts. agent supplies already authorized
    position and facts. options controls status/modifier labels; show_identity
    adds the public ID. Corpse bodies fade, positive shield counters add a ring,
    and only non-neutral aura modifiers are labelled. Return None.
    """
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
    """Write scene's V2 team wave countdowns to the top-right of axes.

    Use the supplied countdown/period step values in tuple order. Tag the label
    with frame ID; do not advance or recompute a clock. Return None.
    """
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
    """Import optional Matplotlib modules and return their drawing factories.

    Imports may initialize Matplotlib's chosen backend. Raise ImportError with
    installation guidance when imports fail. No figure is created here.
    """
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
    """Set artist's group ID to gid and return that same artist.

    artist must expose set_gid. The mutation supports stable SVG/test identity;
    this helper does not check types or uniqueness.
    """
    cast(_ArtistLike, artist).set_gid(gid)
    return artist


def _draw_map(
    axes: _AxesLike,
    parts: _MatplotlibParts,
    scene: BattlefieldScene,
) -> None:
    """Add scene's rectangular map background to axes using parts.

    Read positive width and height in world units, with origin (0, 0). This
    helper adds one tagged patch and returns None.
    """
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
    """Draw supplied V1 aura fields and optional range outlines on axes.

    parts creates circles. Auras always draw; options.show_ranges gates range
    outlines only. Use scene's records without computing visibility or legal
    targets. Return None.
    """
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
    """Draw scene's pending V1 combat route as a dashed arrow on axes.

    Return None without drawing when pending_route is absent. Its supplied
    lane and legal flag choose the label/color; this is pending intent, not
    an accepted simulator event.
    """
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
    """Add supplied pillar circles and rotated wall polygons to axes.

    parts supplies factories; scene supplies validated dimensions, world
    centers and wall angles in radians. No collision rule is run here.
    Return None; missing kind-specific dimensions fail an assertion.
    """
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
    """Paint V1 scene agents and controlled/selected markers on axes.

    parts supplies patches; options controls labels. Selected or controlled
    agents show IDs even when show_agent_ids is False. This helper neither
    filters audience data nor changes scene. Return None.
    """
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
    """Paint a V1 body, team marker, health ring and optional labels.

    axes receives artists from parts. agent supplies position and authorized
    facts; options gates status and modifier labels. show_identity adds its
    global slot ID. Dead bodies fade, and health fractions are clipped only
    for drawing. Return None without changing the record.
    """
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
    """Label the selected V1 target's supplied Basic/Ultimate availability.

    axes receives labels from scene.selected_legality. Missing selection or
    an absent target draws nothing. Armed state changes emphasis only; no
    legality is computed. Return None.
    """
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
    """Label visible/hidden V1 candidates that are present in scene.

    axes receives V/H labels from supplied observer_visibility records. Skip
    candidates absent from scene instead of finding hidden positions. This
    helper does not change audience permissions. Return None.
    """
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
    """Paint supplied legacy V1 event cues onto axes in batch order.

    scene supplies fallback map positions and displayed body radii; event_batch
    supplies authorized anchors and facts. Missing anchors use local labels,
    not inferred target paths. Health labels describe net changes. Charge
    arrows describe recorded endpoints, not the traversed path. Return None;
    raise AssertionError for an unsupported event record type.
    """
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
    """Write scene's declared audience badge at the top-left of axes.

    The badge describes supplied authority; it does not check or grant access.
    Tag the resulting artist and return None.
    """
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
    """Set map bounds, equal world-unit scale and static title on axes.

    scene supplies width and height. Mutate axes presentation and return None;
    this does not alter map geometry or save the figure.
    """
    axes.set_facecolor(_BACKGROUND)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim(0.0, scene.map.width)
    axes.set_ylim(0.0, scene.map.height)
    axes.set_title(
        "MARL-BattleGrounds · Visual Debugger and Analyzer · Static scene",
        color=_TEXT,
        fontsize=10,
    )
