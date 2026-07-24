"""Smoke tests for optional geometry rendering helpers."""

import sys
from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec
from typing import Protocol, TypedDict, cast

import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    MAGE_CLASS_ID,
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
    draw_geometry,
    redraw_geometry,
    render_geometry,
)
from marl_battlegrounds.rendering.visuals import (
    DAMAGE_COLOR,
    ActivationVisual,
    AuraCueVisual,
    BattlefieldOverlays,
    ChargeTrailVisual,
    HealthDeltaVisual,
    LaneMarkerVisual,
    ObserverVisibilityVisual,
    PersistentEffectKind,
    PersistentEffectVisual,
    RangeVisual,
    RejectedActionVisual,
    SelectionVisual,
    StatusCueVisual,
    StatusFamily,
    TargetLinkVisual,
)


class _CombatStateFields(TypedDict):
    """Keyword fields for inert combat and action-history test state."""

    current_health: Array
    ultimate_cooldowns: Array
    slow_durations: Array
    stun_durations: Array
    rogue_poison_anti_heal_durations: Array
    mage_burst_damage_amplification_durations: Array
    priest_blessing_of_freedom_slow_floor_durations: Array
    previous_timestep_move_actions: Array
    previous_timestep_select_target_actions: Array
    previous_timestep_use_ultimate_actions: Array
    has_previous_timestep_joint_action: Array


class _BboxLike(Protocol):
    def overlaps(self, other: object) -> bool: ...


class _TextLike(Protocol):
    def get_text(self) -> str: ...

    def get_gid(self) -> str | None: ...

    def get_zorder(self) -> float: ...

    def get_window_extent(self) -> _BboxLike: ...


class _CanvasDrawLike(Protocol):
    def draw(self) -> object: ...


class _FigureWithCanvas(Protocol):
    canvas: _CanvasDrawLike


class _ArtistLike(Protocol):
    def get_zorder(self) -> float: ...

    def get_gid(self) -> str | None: ...


class _LineLike(_ArtistLike, Protocol):
    def get_color(self) -> str: ...

    def get_linestyle(self) -> str: ...


class _AxesIntrospection(Protocol):
    patches: list[_ArtistLike]
    lines: list[_LineLike]
    texts: list[_TextLike]


def _inert_combat_state_fields() -> _CombatStateFields:
    """Return neutral combat fields for direct EnvState constructors."""
    return {
        "current_health": jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32),
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

    profile = resolve_agent_profile(
        jnp.full((MAX_AGENT_SLOTS,), MAGE_CLASS_ID, dtype=jnp.int32),
        jnp.asarray((2, 2), dtype=jnp.int32),
    )
    positions = jnp.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
    positions = positions.at[0].set(jnp.asarray((2.0, 2.0), dtype=jnp.float32))
    positions = positions.at[1].set(jnp.asarray((3.0, 2.5), dtype=jnp.float32))
    positions = positions.at[MAX_AGENTS_PER_TEAM].set(
        jnp.asarray((10.0, 6.0), dtype=jnp.float32)
    )
    positions = positions.at[MAX_AGENTS_PER_TEAM + 1].set(
        jnp.asarray((9.0, 5.5), dtype=jnp.float32)
    )
    return EnvConfig(
        max_steps=100,
        map_width=12.0,
        map_height=8.0,
        obstacles=obstacles,
        agent_profile=profile,
        initial_agent_positions=positions,
        ordinary_movement_distance_scale=1.0,
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

    return EnvState(
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=positions,
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
        axes = cast(_AxesIntrospection, result.axes)
        assert axes.patches[0].get_zorder() == 0
        assert axes.patches[1].get_zorder() == 10
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


def _all_overlay_families() -> BattlefieldOverlays:
    return BattlefieldOverlays(
        selections=(
            SelectionVisual(global_slot=0, role="controlled"),
            SelectionVisual(global_slot=MAX_AGENTS_PER_TEAM, role="target"),
        ),
        observer_visibility=(
            ObserverVisibilityVisual(
                observer_global_slot=0,
                candidate_global_slot=0,
                observer_visible=True,
            ),
            ObserverVisibilityVisual(
                observer_global_slot=0,
                candidate_global_slot=MAX_AGENTS_PER_TEAM,
                observer_visible=False,
            ),
        ),
        ranges=(
            RangeVisual(0, (2.0, 2.0), 6.0, "observation"),
            RangeVisual(0, (2.0, 2.0), 5.0, "basic"),
            RangeVisual(0, (2.0, 2.0), 3.0, "ultimate"),
        ),
        target_links=(
            TargetLinkVisual(0, MAX_AGENTS_PER_TEAM, 0, True),
            TargetLinkVisual(1, MAX_AGENTS_PER_TEAM + 1, 1, False),
        ),
        lane_markers=(
            LaneMarkerVisual(MAX_AGENTS_PER_TEAM, 0, True, True),
            LaneMarkerVisual(MAX_AGENTS_PER_TEAM, 1, False, False),
        ),
        statuses=(
            StatusCueVisual(0, "stun", 2, 0, 1),
            StatusCueVisual(0, "stun", 3, 1, 2),
            StatusCueVisual(0, "stun", 4, 2, 3),
            StatusCueVisual(0, "slow", 2, 0, 4),
            StatusCueVisual(0, "slow", 3, 1, 5),
            StatusCueVisual(0, "slow", 4, 2, 6),
        ),
        auras=(
            AuraCueVisual(0, "mage_amplification", 1.15),
            AuraCueVisual(0, "warrior_mitigation", 0.85),
        ),
        persistent_effects=(
            PersistentEffectVisual(0, "rogue_anti_heal", 4, 0.5),
            PersistentEffectVisual(0, "priest_freedom", 1, 0.85),
            PersistentEffectVisual(0, "mage_burst", 5, 1.5),
        ),
        health_deltas=(HealthDeltaVisual(MAX_AGENTS_PER_TEAM, -8.0),),
        activations=(
            ActivationVisual("basic_damage", 0, MAX_AGENTS_PER_TEAM, 1),
            ActivationVisual("mage_burst", 0, None, 1),
            ActivationVisual("warrior_charge", 1, MAX_AGENTS_PER_TEAM + 1, 2),
            ActivationVisual("hunter_trap", 0, MAX_AGENTS_PER_TEAM, 3),
            ActivationVisual("rogue_poison", 0, MAX_AGENTS_PER_TEAM, 4),
            ActivationVisual("basic_heal", 0, 0, 5),
            ActivationVisual("holy_word", 0, 0, 5),
        ),
        charge_trails=(
            ChargeTrailVisual(
                1,
                (3.0, 2.5),
                (8.0, 5.0),
                MAX_AGENTS_PER_TEAM + 1,
                "combined_charge_and_movement",
                0.65,
            ),
        ),
        rejections=(
            RejectedActionVisual(0, "movement", None, None),
            RejectedActionVisual(1, "combat", MAX_AGENTS_PER_TEAM + 1, 1),
        ),
    )


def test_draw_geometry_draws_every_overlay_family() -> None:
    _skip_if_matplotlib_unavailable()
    config = _sample_config()
    state = _sample_state()
    result = render_geometry(config, state)

    try:
        axes = cast(_AxesIntrospection, result.axes)
        baseline_patch_count = len(axes.patches)
        baseline_line_count = len(axes.lines)
        baseline_text_count = len(axes.texts)

        draw_geometry(
            axes,
            config,
            state,
            overlays=_all_overlay_families(),
        )

        assert len(axes.patches) > baseline_patch_count
        assert len(axes.lines) > baseline_line_count
        assert len(axes.texts) > baseline_text_count
        assert {
            text.get_text()
            for text in axes.texts
            if text.get_gid() and ":status-chip:" in cast(str, text.get_gid())
        } == {
            "CHARGE-STUN 1",
            "TRAP 2",
            "POISON-STUN 3",
            "CHARGE-SLOW 4",
            "HUNTER-SLOW 5",
            "POISON-SLOW 6",
            "ANTI-HEAL 4",
            "FREEDOM 1",
            "BURST 5",
        }
        assert not any(line.get_zorder() == 42.9 for line in axes.lines)
        assert {"0", "1"} <= {text.get_text() for text in axes.texts}
        assert any(
            line.get_color() == DAMAGE_COLOR and line.get_linestyle() == "--"
            for line in axes.lines
        )
    finally:
        _close_render_result(result)


def test_identical_snapshot_and_overlays_redraw_equivalently() -> None:
    _skip_if_matplotlib_unavailable()
    config = _sample_config()
    state = _sample_state()
    overlays = _all_overlay_families()
    result = render_geometry(config, state, overlays=overlays)

    try:
        axes = cast(_AxesIntrospection, result.axes)
        first_counts = (len(axes.patches), len(axes.lines), len(axes.texts))

        redrawn = redraw_geometry(config, state, result, overlays=overlays)
        redrawn_axes = cast(_AxesIntrospection, redrawn.axes)
        second_counts = (
            len(redrawn_axes.patches),
            len(redrawn_axes.lines),
            len(redrawn_axes.texts),
        )

        assert second_counts == first_counts
    finally:
        _close_render_result(result)


def test_observer_visibility_dims_only_supplied_nonvisible_bodies() -> None:
    _skip_if_matplotlib_unavailable()
    config = _sample_config()
    state = _sample_state()
    visible_overlays = BattlefieldOverlays(
        observer_visibility=(ObserverVisibilityVisual(0, MAX_AGENTS_PER_TEAM, True),),
    )
    hidden_overlays = BattlefieldOverlays(
        observer_visibility=(ObserverVisibilityVisual(0, MAX_AGENTS_PER_TEAM, False),),
    )
    result = render_geometry(config, state, overlays=visible_overlays)

    try:
        axes = cast(_AxesIntrospection, result.axes)
        visible_patch_count = len(axes.patches)
        redraw_geometry(config, state, result, overlays=hidden_overlays)

        assert len(axes.patches) == visible_patch_count + 2
        assert any(
            text.get_text() == f"id_{MAX_AGENTS_PER_TEAM}" for text in axes.texts
        )
    finally:
        _close_render_result(result)


def test_visibility_dimming_stays_below_every_protected_semantic_artist() -> None:
    _skip_if_matplotlib_unavailable()
    config = _sample_config()
    state = _sample_state()
    target = MAX_AGENTS_PER_TEAM
    overlays = BattlefieldOverlays(
        selections=(SelectionVisual(target, "target"),),
        observer_visibility=(ObserverVisibilityVisual(0, target, False),),
        lane_markers=(
            LaneMarkerVisual(target, 0, True, True),
            LaneMarkerVisual(target, 1, False, False),
        ),
        statuses=(StatusCueVisual(target, "stun", 3, 1, 4),),
        auras=(AuraCueVisual(target, "mage_amplification", 1.15),),
        activations=(ActivationVisual("hunter_trap", 0, target, 3),),
    )
    result = render_geometry(config, state, overlays=overlays)

    try:
        axes = cast(_AxesIntrospection, result.axes)
        artists = [*axes.patches, *axes.lines, *axes.texts]
        by_gid = {
            gid: artist for artist in artists if (gid := artist.get_gid()) is not None
        }
        dim_zorders = [
            artist.get_zorder()
            for gid, artist in by_gid.items()
            if gid.startswith(f"agent:{target}:visibility-dim")
        ]
        protected_fragments = (
            ":health-",
            ":aura:",
            ":team-outline",
            ":lane:",
            ":target-reticle",
            ":id-label",
            ":status-chip:",
        )
        protected_zorders = [
            artist.get_zorder()
            for gid, artist in by_gid.items()
            if gid.startswith(f"agent:{target}")
            and any(fragment in gid for fragment in protected_fragments)
        ]
        protected_zorders.extend(
            artist.get_zorder()
            for gid, artist in by_gid.items()
            if gid.startswith("transient:")
        )

        assert dim_zorders
        assert protected_zorders
        assert max(dim_zorders) < min(protected_zorders)
    finally:
        _close_render_result(result)


def test_dense_nearby_status_stacks_use_readable_nonoverlapping_chips() -> None:
    _skip_if_matplotlib_unavailable()
    config = _sample_config()
    state = _sample_state()
    status_values = tuple(
        StatusCueVisual(
            slot,
            cast(StatusFamily, family),
            source_class,
            channel,
            duration,
        )
        for slot in (0, 1)
        for family, source_class, channel, duration in (
            ("stun", 2, 0, 1),
            ("stun", 3, 1, 4),
            ("stun", 4, 2, 1),
            ("slow", 2, 0, 5),
            ("slow", 3, 1, 1),
            ("slow", 4, 2, 5),
        )
    )
    persistent_values = tuple(
        PersistentEffectVisual(
            slot,
            cast(PersistentEffectKind, kind),
            duration,
            magnitude,
        )
        for slot in (0, 1)
        for kind, duration, magnitude in (
            ("rogue_anti_heal", 4, 0.5),
            ("priest_freedom", 1, 0.85),
            ("mage_burst", 5, 1.5),
        )
    )
    overlays = BattlefieldOverlays(
        statuses=status_values,
        persistent_effects=persistent_values,
    )
    result = render_geometry(config, state, overlays=overlays)

    try:
        canvas = cast(_FigureWithCanvas, result.figure).canvas
        canvas.draw()
        axes = cast(_AxesIntrospection, result.axes)
        chips_by_slot = {
            slot: [
                text
                for text in axes.texts
                if (text.get_gid() or "").startswith(f"agent:{slot}:status-chip:")
            ]
            for slot in (0, 1)
        }

        assert all(len(chips) == 9 for chips in chips_by_slot.values())
        assert {text.get_text() for text in chips_by_slot[0]} == {
            "CHARGE-STUN 1",
            "TRAP 4",
            "POISON-STUN 1",
            "CHARGE-SLOW 5",
            "HUNTER-SLOW 1",
            "POISON-SLOW 5",
            "ANTI-HEAL 4",
            "FREEDOM 1",
            "BURST 5",
        }
        assert not any(
            left.get_window_extent().overlaps(right.get_window_extent())
            for left in chips_by_slot[0]
            for right in chips_by_slot[1]
        )

        redraw_geometry(config, state, result, overlays=overlays)
        redrawn_axes = cast(_AxesIntrospection, result.axes)
        assert (
            sum(
                text.get_gid() is not None
                and ":status-chip:" in cast(str, text.get_gid())
                for text in redrawn_axes.texts
            )
            == 18
        )
    finally:
        _close_render_result(result)


@pytest.mark.parametrize(
    ("visual", "minimum_patch_delta", "minimum_line_delta", "required_text"),
    (
        (
            ActivationVisual("basic_damage", 0, MAX_AGENTS_PER_TEAM, 1),
            0,
            1,
            "✦",
        ),
        (
            ActivationVisual("basic_heal", 1, 0, 5),
            1,
            1,
            None,
        ),
        (
            ActivationVisual("holy_word", 1, 0, 5),
            0,
            1,
            "HOLY WORD!",
        ),
        (
            ActivationVisual("mage_burst", 0, None, 1),
            0,
            0,
            "BURST!",
        ),
        (
            ActivationVisual("warrior_charge", 0, MAX_AGENTS_PER_TEAM, 2),
            0,
            0,
            "CHARGE!",
        ),
        (
            ActivationVisual("hunter_trap", 0, MAX_AGENTS_PER_TEAM, 3),
            0,
            0,
            "TRAP!",
        ),
        (
            ActivationVisual("rogue_poison", 0, MAX_AGENTS_PER_TEAM, 4),
            0,
            0,
            "POISON!",
        ),
    ),
)
def test_each_activation_kind_has_a_distinct_drawable_primitive(
    visual: ActivationVisual,
    minimum_patch_delta: int,
    minimum_line_delta: int,
    required_text: str | None,
) -> None:
    _skip_if_matplotlib_unavailable()
    config = _sample_config()
    state = _sample_state()
    result = render_geometry(config, state)

    try:
        axes = cast(_AxesIntrospection, result.axes)
        baseline = (len(axes.patches), len(axes.lines))
        draw_geometry(
            axes,
            config,
            state,
            overlays=BattlefieldOverlays(activations=(visual,)),
        )

        assert len(axes.patches) >= baseline[0] + minimum_patch_delta
        assert len(axes.lines) >= baseline[1] + minimum_line_delta
        if required_text is not None:
            assert any(text.get_text() == required_text for text in axes.texts)
    finally:
        _close_render_result(result)


def test_multiple_recipient_effects_keep_labels_and_segmented_composition() -> None:
    _skip_if_matplotlib_unavailable()
    config = _sample_config()
    state = _sample_state()
    result = render_geometry(config, state)
    target = MAX_AGENTS_PER_TEAM
    overlays = BattlefieldOverlays(
        activations=(
            ActivationVisual("basic_damage", 0, target, 1),
            ActivationVisual("hunter_trap", 1, target, 3),
            ActivationVisual("rogue_poison", 0, target, 4),
        )
    )

    try:
        axes = cast(_AxesIntrospection, result.axes)
        baseline_patch_count = len(axes.patches)
        draw_geometry(axes, config, state, overlays=overlays)

        assert len(axes.patches) == baseline_patch_count + 3
        labels = {text.get_text() for text in axes.texts}
        assert {"✦", "TRAP!", "POISON!"} <= labels
    finally:
        _close_render_result(result)


def test_simultaneous_movement_and_combat_rejections_share_one_label() -> None:
    _skip_if_matplotlib_unavailable()
    config = _sample_config()
    state = _sample_state()
    overlays = BattlefieldOverlays(
        rejections=(
            RejectedActionVisual(0, "movement", None, None),
            RejectedActionVisual(0, "combat", MAX_AGENTS_PER_TEAM, 1),
        )
    )
    result = render_geometry(config, state, overlays=overlays)

    try:
        axes = cast(_AxesIntrospection, result.axes)
        labels = [text.get_text() for text in axes.texts]
        assert labels.count("M\u00d7 C\u00d7") == 1
        assert (
            sum(
                line.get_color() == DAMAGE_COLOR and line.get_linestyle() == "--"
                for line in axes.lines
            )
            == 1
        )
    finally:
        _close_render_result(result)
