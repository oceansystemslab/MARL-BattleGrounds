"""Event-driven Matplotlib application for the comprehensive visual debugger."""

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Protocol, cast

import numpy as np

from marl_battlegrounds.core.types import (
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_NORTHEAST,
    MOVE_NORTHWEST,
    MOVE_SOUTH,
    MOVE_SOUTHEAST,
    MOVE_SOUTHWEST,
    MOVE_WEST,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.rendering import draw_geometry
from scripts.dev.visual_debugger.control import (
    arm_basic,
    arm_ultimate,
    clear_pending_target,
    create_session,
    cycle_controlled_actor,
    reset_session,
    select_clicked_target,
    set_pending_movement,
    submit_interactive,
    submit_next_script_frame,
    switch_scenario,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.presentation import (
    build_debugger_overlays,
    draw_hud,
)
from scripts.dev.visual_debugger.scenarios import (
    cycle_scenario_name,
    get_scenario,
)

_MOVEMENT_KEYS = {
    "w": MOVE_NORTH,
    "s": MOVE_SOUTH,
    "d": MOVE_EAST,
    "a": MOVE_WEST,
    "q": MOVE_NORTHWEST,
    "e": MOVE_NORTHEAST,
    "z": MOVE_SOUTHWEST,
    "c": MOVE_SOUTHEAST,
}
_RESERVED_KEYS = frozenset(
    (
        *_MOVEMENT_KEYS,
        "tab",
        "shift+tab",
        "1",
        "2",
        "space",
        "enter",
        "n",
        "r",
        "g",
        "v",
        "[",
        "]",
        "escape",
    )
)


class _CanvasLike(Protocol):
    def mpl_connect(self, event_name: str, callback: Callable[..., None]) -> int: ...

    def mpl_disconnect(self, connection_id: int) -> object: ...

    def draw_idle(self) -> object: ...


class _FigureLike(Protocol):
    canvas: _CanvasLike


class _PyplotLike(Protocol):
    rcParams: MutableMapping[str, object]  # noqa: N815 - Matplotlib API name.

    def subplots(self, *args: object, **kwargs: object) -> tuple[object, object]: ...

    def show(self) -> object: ...


class _KeyEventLike(Protocol):
    key: str | None


class _MouseEventLike(Protocol):
    inaxes: object | None
    xdata: float | None
    ydata: float | None
    button: object


def hit_test_active_agent(
    config: EnvConfig,
    state: EnvState,
    x: float,
    y: float,
) -> int | None:
    """Return the closest normalized active body hit with stable slot tie-break."""
    positions = np.asarray(state.agent_positions, dtype=np.float32)
    radii = np.asarray(config.agent_profile.agent_radii, dtype=np.float32)
    active_mask = np.asarray(config.agent_profile.active_mask, dtype=bool)
    candidates: list[tuple[float, int]] = []
    point = np.asarray((x, y), dtype=np.float32)
    for global_slot in np.flatnonzero(active_mask):
        slot = int(global_slot)
        radius = float(radii[slot])
        if radius <= 0:
            continue
        normalized_distance = float(np.linalg.norm(point - positions[slot]) / radius)
        if normalized_distance <= 1.0:
            candidates.append((normalized_distance, slot))
    if not candidates:
        return None
    return min(candidates)[1]


def _load_pyplot() -> _PyplotLike:
    try:
        return cast(_PyplotLike, import_module("matplotlib.pyplot"))
    except ImportError as exc:
        msg = (
            "Matplotlib is required for the visual debugger. "
            "Run 'uv sync --extra viz --extra dev'."
        )
        raise ImportError(msg) from exc


def _normalize_key(key: str | None) -> str | None:
    if key is None:
        return None
    normalized = key.lower()
    if normalized == " ":
        return "space"
    if normalized in ("return",):
        return "enter"
    return normalized


def _reserve_keymaps(pyplot: _PyplotLike) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for name, value in tuple(pyplot.rcParams.items()):
        if not name.startswith("keymap.") or not isinstance(value, list):
            continue
        entries = cast(list[object], value)
        filtered = [
            entry
            for entry in entries
            if not (isinstance(entry, str) and _normalize_key(entry) in _RESERVED_KEYS)
        ]
        if filtered != entries:
            snapshot[name] = list(entries)
            pyplot.rcParams[name] = filtered
    return snapshot


def _restore_keymaps(
    pyplot: _PyplotLike,
    snapshot: dict[str, object],
) -> None:
    for name, value in snapshot.items():
        pyplot.rcParams[name] = value


def _is_terminal(session: DebuggerSession) -> bool:
    return bool(session.done_flags.terminated) or bool(session.done_flags.truncated)


@dataclass(slots=True)
class VisualDebuggerApp:
    """Thin mutable Matplotlib adapter around one immutable debugger session."""

    session: DebuggerSession
    figure: object
    battlefield_axes: object
    hud_axes: object
    connection_ids: list[int]
    keymap_snapshot: dict[str, object]
    closed: bool = False

    def connect(self) -> None:
        if self.closed or self.connection_ids:
            return
        canvas = cast(_FigureLike, self.figure).canvas

        def _on_close(_event: object) -> None:
            self.close()

        try:
            for event_name, callback in (
                ("key_press_event", self.on_key_press),
                ("button_press_event", self.on_mouse_press),
                ("close_event", _on_close),
            ):
                self.connection_ids.append(canvas.mpl_connect(event_name, callback))
        except Exception:
            for connection_id in self.connection_ids:
                canvas.mpl_disconnect(connection_id)
            self.connection_ids.clear()
            raise

    def redraw(self) -> None:
        draw_geometry(
            self.battlefield_axes,
            self.session.config,
            self.session.state,
            overlays=build_debugger_overlays(self.session),
        )
        draw_hud(self.hud_axes, self.session)
        cast(_FigureLike, self.figure).canvas.draw_idle()

    def on_key_press(self, event: object) -> None:
        key = _normalize_key(cast(_KeyEventLike, event).key)
        if key is None or self.closed:
            return
        scenario = get_scenario(self.session.scenario_name)
        terminal = _is_terminal(self.session)

        if key == "tab":
            self.session = cycle_controlled_actor(self.session, 1)
        elif key == "shift+tab":
            self.session = cycle_controlled_actor(self.session, -1)
        elif key == "escape":
            self.session = clear_pending_target(self.session)
        elif key in _MOVEMENT_KEYS and not terminal:
            self.session = set_pending_movement(
                self.session,
                _MOVEMENT_KEYS[key],
            )
        elif key == "1" and not terminal:
            self.session = arm_basic(self.session)
        elif key == "2" and not terminal:
            self.session = arm_ultimate(self.session)
        elif key in ("space", "enter"):
            self.session = (
                submit_next_script_frame(self.session)
                if scenario.mode == "scripted"
                else submit_interactive(self.session)
            )
        elif key == "n":
            self.session = submit_next_script_frame(self.session)
        elif key == "r":
            self.session = reset_session(self.session)
        elif key == "g":
            self.session = replace(
                self.session,
                show_ranges=not self.session.show_ranges,
            )
        elif key == "v":
            self.session = replace(
                self.session,
                verbose_logging=not self.session.verbose_logging,
            )
        elif key in ("[", "]"):
            direction = -1 if key == "[" else 1
            next_name = cycle_scenario_name(
                self.session.scenario_name,
                direction,
            )
            self.session = switch_scenario(
                self.session,
                get_scenario(next_name),
            )
        else:
            return
        self.redraw()

    def on_mouse_press(self, event: object) -> None:
        if self.closed:
            return
        mouse_event = cast(_MouseEventLike, event)
        if mouse_event.inaxes is not self.battlefield_axes:
            return
        if mouse_event.xdata is None or mouse_event.ydata is None:
            return
        try:
            button = int(mouse_event.button)  # type: ignore[arg-type]
        except TypeError, ValueError:
            return
        if button == 3:
            self.session = clear_pending_target(self.session)
        elif button == 1:
            target = hit_test_active_agent(
                self.session.config,
                self.session.state,
                mouse_event.xdata,
                mouse_event.ydata,
            )
            if target is None:
                return
            self.session = select_clicked_target(self.session, target)
        else:
            return
        self.redraw()

    def close(self) -> None:
        if self.closed:
            return
        canvas = cast(_FigureLike, self.figure).canvas
        for connection_id in self.connection_ids:
            canvas.mpl_disconnect(connection_id)
        self.connection_ids.clear()
        _restore_keymaps(_load_pyplot(), self.keymap_snapshot)
        self.keymap_snapshot.clear()
        self.closed = True


def _create_figure(pyplot: _PyplotLike) -> tuple[object, object, object]:
    figure, axes = pyplot.subplots(
        1,
        2,
        figsize=(15, 8),
        gridspec_kw={"width_ratios": (3, 2)},
    )
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    if axes_array.size != 2:
        msg = "visual debugger expected exactly two Matplotlib axes."
        raise RuntimeError(msg)
    return figure, axes_array[0], axes_array[1]


def run_visual_debugger(
    *,
    scenario_name: str,
    seed: int,
    controlled_global_slot: int | None,
    static: bool,
    verbose: bool,
    show_ranges: bool,
) -> int:
    """Create, optionally connect, display, and cleanly close the debugger."""
    pyplot = _load_pyplot()
    scenario = get_scenario(scenario_name)
    if controlled_global_slot is not None:
        config = scenario.build_config()
        if not (
            0 <= controlled_global_slot < len(config.agent_profile.active_mask)
            and bool(config.agent_profile.active_mask[controlled_global_slot])
        ):
            msg = (
                f"controlled slot g{controlled_global_slot} is not active in "
                f"scenario {scenario_name!r}."
            )
            raise ValueError(msg)
    session = create_session(
        scenario,
        seed=seed,
        controlled_global_slot=controlled_global_slot,
        show_ranges=show_ranges,
        verbose_logging=verbose,
    )
    keymap_snapshot = _reserve_keymaps(pyplot)
    try:
        figure, battlefield_axes, hud_axes = _create_figure(pyplot)
    except Exception:
        _restore_keymaps(pyplot, keymap_snapshot)
        raise
    app = VisualDebuggerApp(
        session=session,
        figure=figure,
        battlefield_axes=battlefield_axes,
        hud_axes=hud_axes,
        connection_ids=[],
        keymap_snapshot=keymap_snapshot,
    )
    try:
        if not static:
            app.connect()
        app.redraw()
        pyplot.show()
        return 0
    finally:
        app.close()
