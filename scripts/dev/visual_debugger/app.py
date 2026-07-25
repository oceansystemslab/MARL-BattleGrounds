"""Event-driven Matplotlib application for the comprehensive visual debugger."""

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

import numpy as np

from marl_battlegrounds.rendering import draw_geometry
from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.input import (
    dispatch_command,
    hit_test_active_agent,
    normalize_key,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.presentation import (
    build_debugger_overlays,
    draw_hud,
)
from scripts.dev.visual_debugger.protocol import (
    BattlefieldPointerCommandV1,
    KeyboardCommandV1,
)
from scripts.dev.visual_debugger.scenarios import get_scenario

__all__ = ["VisualDebuggerApp", "hit_test_active_agent", "run_visual_debugger"]

_RESERVED_KEYS = frozenset(
    (
        "w",
        "s",
        "d",
        "a",
        "q",
        "e",
        "z",
        "c",
        "up",
        "down",
        "left",
        "right",
        "tab",
        "shift+tab",
        "shift+r",
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
    guiEvent: object | None  # noqa: N815 - Matplotlib event API name.


class _MouseEventLike(Protocol):
    inaxes: object | None
    xdata: float | None
    ydata: float | None
    button: object
    modifiers: frozenset[str] | None


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
    """Compatibility wrapper around the renderer-independent normalizer."""
    return normalize_key(key)


def _schedule_canvas_focus(event: object, canvas: object) -> None:
    """Restore canvas focus after native traversal has completed."""
    gui_event = getattr(event, "guiEvent", None)
    widget = getattr(gui_event, "widget", None)
    after_idle = getattr(widget, "after_idle", None)
    focus_set = getattr(widget, "focus_set", None)
    if callable(after_idle) and callable(focus_set):
        after_idle(focus_set)
        return
    set_focus = getattr(canvas, "setFocus", None)
    if callable(set_focus):
        set_focus()


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


@dataclass(slots=True)
class VisualDebuggerApp:
    """Thin mutable Matplotlib adapter around one immutable debugger session."""

    session: DebuggerSession
    figure: object
    battlefield_axes: object
    hud_axes: object
    connection_ids: list[int]
    keymap_snapshot: dict[str, object]
    include_stress: bool = False
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
        result = dispatch_command(
            self.session,
            KeyboardCommandV1(
                key=key,
                shift_key=key in ("shift+tab", "shift+r"),
            ),
            view_mode="researcher",
            preset="analysis",
            include_stress=self.include_stress,
        )
        if result.notice is not None:
            print(result.notice)
        if not result.handled:
            return
        self.session = result.session
        if key in ("tab", "shift+tab"):
            _schedule_canvas_focus(
                event,
                cast(_FigureLike, self.figure).canvas,
            )
        if result.changed:
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
        if button not in (1, 3):
            return
        modifiers = getattr(mouse_event, "modifiers", None) or ()
        result = dispatch_command(
            self.session,
            BattlefieldPointerCommandV1(
                world_x=float(mouse_event.xdata),
                world_y=float(mouse_event.ydata),
                button="primary" if button == 1 else "secondary",
                shift_key="shift" in modifiers,
            ),
            view_mode="researcher",
            preset="analysis",
            include_stress=self.include_stress,
        )
        if result.notice is not None:
            print(result.notice)
        if not result.handled:
            return
        self.session = result.session
        if result.changed:
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
        figsize=(17, 9),
        gridspec_kw={"width_ratios": (58, 42)},
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
    include_stress: bool = False,
) -> int:
    """Create, optionally connect, display, and cleanly close the debugger."""
    scenario = get_scenario(scenario_name)
    if scenario.audience == "stress" and not include_stress:
        msg = f"stress scenario {scenario_name!r} requires the --include-stress option."
        raise ValueError(msg)
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
    pyplot = _load_pyplot()
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
        include_stress=include_stress,
    )
    try:
        if not static:
            app.connect()
        app.redraw()
        pyplot.show()
        return 0
    finally:
        app.close()
