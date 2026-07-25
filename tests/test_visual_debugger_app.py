"""Headless application tests for callbacks, hit testing, and static mode."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
import scripts.dev.visual_debugger.app as app_module
from scripts.dev.visual_debugger.app import (
    VisualDebuggerApp,
    hit_test_active_agent,
    run_visual_debugger,
)
from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.scenarios import get_scenario

from marl_battlegrounds.core.types import (
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_NORTHEAST,
    MOVE_NORTHWEST,
    MOVE_SOUTH,
    MOVE_SOUTHEAST,
    MOVE_SOUTHWEST,
    MOVE_WEST,
    DoneFlags,
)


class _FakeCanvas:
    def __init__(self) -> None:
        self.callbacks: dict[int, tuple[str, Callable[..., None]]] = {}
        self.next_id = 1
        self.draw_idle_calls = 0
        self.disconnected: list[int] = []
        self.timer_requested = False
        self.set_focus_calls = 0

    def mpl_connect(self, event_name: str, callback: Callable[..., None]) -> int:
        connection_id = self.next_id
        self.next_id += 1
        self.callbacks[connection_id] = (event_name, callback)
        return connection_id

    def mpl_disconnect(self, connection_id: int) -> None:
        self.disconnected.append(connection_id)

    def draw_idle(self) -> None:
        self.draw_idle_calls += 1

    def new_timer(self, *_args: object, **_kwargs: object) -> object:
        self.timer_requested = True
        raise AssertionError("the explicit-submit debugger must not create a timer")

    def setFocus(self) -> None:  # noqa: N802 - Qt canvas compatibility surface.
        self.set_focus_calls += 1


class _FailingConnectCanvas(_FakeCanvas):
    def mpl_connect(self, event_name: str, callback: Callable[..., None]) -> int:
        if self.next_id == 2:
            raise RuntimeError("callback registration failed")
        return super().mpl_connect(event_name, callback)


@dataclass
class _FakeFigure:
    canvas: _FakeCanvas


class _FakePyplot:
    def __init__(self) -> None:
        self.rcParams: dict[str, object] = {
            "keymap.save": ["s", "ctrl+s"],
            "keymap.quit": ["q"],
            "other": ["w"],
        }
        self.show_calls = 0
        self.subplots_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def show(self) -> None:
        self.show_calls += 1

    def subplots(
        self,
        *args: object,
        **kwargs: object,
    ) -> tuple[object, tuple[object, object]]:
        self.subplots_calls.append((args, kwargs))
        return object(), (object(), object())


def _ignore_render(*_args: object, **_kwargs: object) -> None:
    return None


def _session(
    name: str = "arena_5v5",
    controlled_slot: int | None = None,
) -> DebuggerSession:
    scenario = get_scenario(name)
    return create_session(
        scenario,
        seed=0,
        controlled_global_slot=controlled_slot,
        show_ranges=True,
        verbose_logging=False,
    )


def _app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scenario_name: str = "arena_5v5",
    controlled_slot: int | None = None,
) -> tuple[VisualDebuggerApp, _FakeCanvas, object, _FakePyplot]:
    canvas = _FakeCanvas()
    figure = _FakeFigure(canvas)
    battlefield_axes = object()
    hud_axes = object()
    pyplot = _FakePyplot()
    monkeypatch.setattr(app_module, "_load_pyplot", lambda: pyplot)
    monkeypatch.setattr(app_module, "draw_geometry", _ignore_render)
    monkeypatch.setattr(app_module, "draw_hud", _ignore_render)
    app = VisualDebuggerApp(
        session=_session(scenario_name, controlled_slot),
        figure=figure,
        battlefield_axes=battlefield_axes,
        hud_axes=hud_axes,
        connection_ids=[],
        keymap_snapshot={"keymap.save": ["s", "ctrl+s"]},
    )
    return app, canvas, battlefield_axes, pyplot


def test_hit_testing_uses_true_active_discs_and_ignores_padding() -> None:
    session = _session("basic_support")
    assert (
        hit_test_active_agent(
            session.config,
            session.state,
            4.0,
            3.0,
        )
        == 0
    )
    assert (
        hit_test_active_agent(
            session.config,
            session.state,
            7.0,
            3.0,
        )
        == 5
    )
    assert (
        hit_test_active_agent(
            session.config,
            session.state,
            0.0,
            0.0,
        )
        is None
    )
    # Padded slot positions are zero but cannot be clicked.
    assert not bool(session.config.agent_profile.active_mask[3])


def test_hit_testing_uses_normalized_distance_then_lowest_slot_tie_break() -> None:
    session = _session("basic_support")
    overlapping = session.state._replace(
        agent_positions=session.state.agent_positions.at[5].set(
            session.state.agent_positions[0]
        )
    )
    assert hit_test_active_agent(session.config, overlapping, 4.0, 3.0) == 0

    nearer_to_five = overlapping._replace(
        agent_positions=overlapping.agent_positions.at[5].set(
            jnp.asarray((4.1, 3.0), dtype=jnp.float32)
        )
    )
    assert hit_test_active_agent(session.config, nearer_to_five, 4.09, 3.0) == 5


def test_app_registers_expected_callbacks_without_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, canvas, _, _ = _app(monkeypatch)
    app.connect()
    app.connect()

    assert [event for event, _ in canvas.callbacks.values()] == [
        "key_press_event",
        "button_press_event",
        "close_event",
    ]
    assert len(app.connection_ids) == 3
    assert not canvas.timer_requested


@pytest.mark.parametrize(
    ("key", "expected_move"),
    (
        ("w", MOVE_NORTH),
        ("s", MOVE_SOUTH),
        ("d", MOVE_EAST),
        ("a", MOVE_WEST),
        ("q", MOVE_NORTHWEST),
        ("e", MOVE_NORTHEAST),
        ("z", MOVE_SOUTHWEST),
        ("c", MOVE_SOUTHEAST),
        ("W", MOVE_NORTH),
        ("S", MOVE_SOUTH),
        ("D", MOVE_EAST),
        ("A", MOVE_WEST),
        ("Q", MOVE_NORTHWEST),
        ("E", MOVE_NORTHEAST),
        ("Z", MOVE_SOUTHWEST),
        ("C", MOVE_SOUTHEAST),
    ),
)
def test_movement_key_normalization_changes_pending_without_step(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    expected_move: int,
) -> None:
    app, _, _, _ = _app(monkeypatch)
    initial_state = app.session.state
    initial_key = app.session.key
    app.on_key_press(SimpleNamespace(key=key))

    assert app.session.pending_action.move_action == expected_move
    assert app.session.state is initial_state
    assert bool(jnp.array_equal(app.session.key, initial_key))


def test_keyboard_callbacks_cover_cycle_arm_clear_toggles_and_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, _ = _app(monkeypatch, scenario_name="basic_support")

    app.on_key_press(SimpleNamespace(key="tab"))
    assert app.session.controlled_global_slot == 1
    app.on_key_press(SimpleNamespace(key="shift+tab"))
    assert app.session.controlled_global_slot == 0
    app.on_key_press(SimpleNamespace(key="2"))
    assert app.session.pending_action.armed_lane == 1
    app.on_key_press(SimpleNamespace(key="escape"))
    assert app.session.pending_action.selected_global_target_slot is None
    app.on_key_press(SimpleNamespace(key="g"))
    assert not app.session.show_ranges
    app.on_key_press(SimpleNamespace(key="v"))
    assert app.session.verbose_logging
    app.on_key_press(SimpleNamespace(key="]"))
    assert app.session.scenario_name == "ultimate_showcase"
    app.on_key_press(SimpleNamespace(key="["))
    assert app.session.scenario_name == "basic_support"


@pytest.mark.parametrize(
    "key",
    ("shift+tab", "backtab", "iso_left_tab", "shift+iso_left_tab"),
)
def test_backward_tab_variants_cycle_and_schedule_tk_focus_restoration(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    app, canvas, _, _ = _app(monkeypatch, scenario_name="basic_support")
    callbacks: list[Callable[[], None]] = []

    class _Widget:
        def after_idle(self, callback: Callable[[], None]) -> None:
            callbacks.append(callback)

        def focus_set(self) -> None:
            canvas.set_focus_calls += 1

    widget = _Widget()
    event = SimpleNamespace(
        key=key,
        guiEvent=SimpleNamespace(widget=widget),
    )
    app.on_key_press(event)

    assert app.session.controlled_global_slot == 7
    assert callbacks == [widget.focus_set]
    assert canvas.set_focus_calls == 0
    callbacks[0]()
    assert canvas.set_focus_calls == 1


def test_tab_focus_restoration_falls_back_to_canvas_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, canvas, _, _ = _app(monkeypatch, scenario_name="basic_support")
    app.on_key_press(SimpleNamespace(key="tab"))
    assert canvas.set_focus_calls == 1


@pytest.mark.parametrize("submit_key", (" ", "space", "enter", "return"))
def test_submit_key_variants_advance_exactly_one_step(
    monkeypatch: pytest.MonkeyPatch,
    submit_key: str,
) -> None:
    app, _, _, _ = _app(monkeypatch, scenario_name="arena_5v5")
    app.on_key_press(SimpleNamespace(key=submit_key))
    assert int(app.session.state.step_count) == 1


def test_n_submits_reference_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, _ = _app(monkeypatch, scenario_name="basic_support")
    app.on_key_press(SimpleNamespace(key="n"))
    assert int(app.session.state.step_count) == 1
    assert app.session.next_script_frame_index == 1


def test_scripted_space_submits_multi_actor_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, _ = _app(monkeypatch, scenario_name="basic_support")
    app.on_key_press(SimpleNamespace(key="space"))
    transition = app.session.last_transition
    assert transition is not None
    assert transition.submission_kind == "scripted"
    assert transition.report_actor_slots == (0, 1, 7)


def test_terminal_mode_keys_do_not_mutate_pending_but_inspection_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, _, _ = _app(monkeypatch, scenario_name="basic_support")
    app.session = replace(
        app.session,
        done_flags=DoneFlags(
            terminated=jnp.asarray(True),
            truncated=jnp.asarray(False),
        ),
    )
    original_pending = app.session.pending_action
    app.on_key_press(SimpleNamespace(key="d"))
    app.on_key_press(SimpleNamespace(key="2"))
    assert app.session.pending_action == original_pending
    app.on_key_press(SimpleNamespace(key="tab"))
    assert app.session.controlled_global_slot == 1


def test_mouse_callbacks_select_clear_and_ignore_empty_or_other_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, battlefield_axes, _ = _app(
        monkeypatch,
        scenario_name="arena_5v5",
    )
    app.on_mouse_press(
        SimpleNamespace(
            inaxes=battlefield_axes,
            xdata=15.0,
            ydata=10.0,
            button=1,
            modifiers=frozenset(),
        )
    )
    assert app.session.pending_action.selected_global_target_slot == 5
    selected = app.session
    app.on_mouse_press(
        SimpleNamespace(
            inaxes=battlefield_axes,
            xdata=8.0,
            ydata=1.0,
            button=1,
            modifiers=frozenset(),
        )
    )
    assert app.session is selected
    app.on_mouse_press(
        SimpleNamespace(
            inaxes=object(),
            xdata=7.0,
            ydata=3.0,
            button=1,
            modifiers=frozenset(),
        )
    )
    assert app.session is selected
    app.on_mouse_press(
        SimpleNamespace(
            inaxes=battlefield_axes,
            xdata=7.0,
            ydata=3.0,
            button=3,
            modifiers=frozenset(),
        )
    )
    assert app.session.pending_action.selected_global_target_slot is None


def test_shift_click_selects_actor_while_stale_mouse_key_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, battlefield_axes, _ = _app(
        monkeypatch,
        scenario_name="basic_support",
    )
    initial_state = app.session.state
    initial_key = app.session.key

    app.on_mouse_press(
        SimpleNamespace(
            inaxes=battlefield_axes,
            xdata=7.0,
            ydata=3.0,
            button=1,
            modifiers=frozenset(),
            key="shift+d",
        )
    )
    assert app.session.controlled_global_slot == 0
    assert app.session.pending_action.selected_global_target_slot == 5

    app.on_mouse_press(
        SimpleNamespace(
            inaxes=battlefield_axes,
            xdata=4.0,
            ydata=6.0,
            button=1,
            modifiers=frozenset({"shift"}),
            key="d",
        )
    )
    assert app.session.controlled_global_slot == 1
    assert app.session.pending_action.selected_global_target_slot == 5
    assert app.session.state is initial_state
    assert bool(jnp.array_equal(app.session.key, initial_key))


def test_target_actor_movement_target_arm_submit_sequence_needs_no_dummy_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _, battlefield_axes, _ = _app(
        monkeypatch,
        scenario_name="arena_5v5",
    )
    app.on_mouse_press(
        SimpleNamespace(
            inaxes=battlefield_axes,
            xdata=15.0,
            ydata=10.0,
            button=1,
            modifiers=frozenset(),
        )
    )
    app.on_mouse_press(
        SimpleNamespace(
            inaxes=battlefield_axes,
            xdata=3.0,
            ydata=6.0,
            button=1,
            modifiers=frozenset({"shift"}),
        )
    )
    app.on_key_press(SimpleNamespace(key="d"))
    app.on_mouse_press(
        SimpleNamespace(
            inaxes=battlefield_axes,
            xdata=15.0,
            ydata=6.0,
            button=1,
            modifiers=frozenset(),
        )
    )
    app.on_key_press(SimpleNamespace(key="2"))
    app.on_key_press(SimpleNamespace(key="enter"))

    assert int(app.session.state.step_count) == 1
    transition = app.session.last_transition
    assert transition is not None
    assert transition.report_actor_slots == (2,)


def test_shift_r_is_inert_and_lowercase_r_resets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app, _, _, _ = _app(monkeypatch, scenario_name="basic_support")
    app.on_key_press(SimpleNamespace(key="space"))
    advanced = app.session
    app.on_key_press(SimpleNamespace(key="R"))

    assert app.session is advanced
    assert "no public coherent snapshot-rebuild API exists" in capsys.readouterr().out

    app.on_key_press(SimpleNamespace(key="r"))
    assert int(app.session.state.step_count) == 0


def test_redraw_never_steps_and_only_requests_canvas_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, canvas, _, _ = _app(monkeypatch)
    state = app.session.state
    key = app.session.key
    app.redraw()
    app.redraw()

    assert app.session.state is state
    assert bool(jnp.array_equal(app.session.key, key))
    assert canvas.draw_idle_calls == 2


def test_close_disconnects_callbacks_restores_keymaps_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, canvas, _, pyplot = _app(monkeypatch)
    app.connect()
    connection_ids = tuple(app.connection_ids)
    app.close()
    app.close()

    assert app.closed
    assert tuple(canvas.disconnected) == connection_ids
    assert pyplot.rcParams["keymap.save"] == ["s", "ctrl+s"]
    assert app.connection_ids == []
    assert app.keymap_snapshot == {}


def test_static_mode_does_not_register_callbacks_or_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canvas = _FakeCanvas()
    figure = _FakeFigure(canvas)
    pyplot = _FakePyplot()
    monkeypatch.setattr(app_module, "_load_pyplot", lambda: pyplot)

    def fake_create_figure(_pyplot: object) -> tuple[object, object, object]:
        return figure, object(), object()

    monkeypatch.setattr(
        app_module,
        "_create_figure",
        fake_create_figure,
    )
    monkeypatch.setattr(app_module, "draw_geometry", _ignore_render)
    monkeypatch.setattr(app_module, "draw_hud", _ignore_render)

    exit_code = run_visual_debugger(
        scenario_name="arena_5v5",
        seed=0,
        controlled_global_slot=None,
        static=True,
        verbose=False,
        show_ranges=True,
    )

    assert exit_code == 0
    assert pyplot.show_calls == 1
    assert canvas.callbacks == {}
    assert not canvas.timer_requested
    assert canvas.draw_idle_calls == 1


def test_figure_reserves_forty_two_percent_for_the_hud() -> None:
    pyplot = _FakePyplot()

    _, battlefield_axes, hud_axes = app_module._create_figure(  # pyright: ignore[reportPrivateUsage]
        pyplot  # pyright: ignore[reportArgumentType]
    )

    assert battlefield_axes is not hud_axes
    assert pyplot.subplots_calls == [
        (
            (1, 2),
            {
                "figsize": (17, 9),
                "gridspec_kw": {"width_ratios": (58, 42)},
            },
        )
    ]


def test_run_rejects_inactive_controlled_slot_before_figure_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyplot = _FakePyplot()
    monkeypatch.setattr(app_module, "_load_pyplot", lambda: pyplot)
    with pytest.raises(ValueError, match="not active"):
        run_visual_debugger(
            scenario_name="basic_support",
            seed=0,
            controlled_global_slot=3,
            static=True,
            verbose=False,
            show_ranges=True,
        )


def test_figure_creation_failure_restores_reserved_matplotlib_keymaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyplot = _FakePyplot()
    original_save_keymap = list(pyplot.rcParams["keymap.save"])  # type: ignore[arg-type]
    monkeypatch.setattr(app_module, "_load_pyplot", lambda: pyplot)

    def fail_figure_creation(_pyplot: object) -> tuple[object, object, object]:
        raise RuntimeError("backend failed")

    monkeypatch.setattr(app_module, "_create_figure", fail_figure_creation)
    with pytest.raises(RuntimeError, match="backend failed"):
        run_visual_debugger(
            scenario_name="arena_5v5",
            seed=0,
            controlled_global_slot=None,
            static=False,
            verbose=False,
            show_ranges=True,
        )

    assert pyplot.rcParams["keymap.save"] == original_save_keymap


def test_partial_callback_failure_disconnects_registered_callbacks_and_restores_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canvas = _FailingConnectCanvas()
    figure = _FakeFigure(canvas)
    pyplot = _FakePyplot()
    original_save_keymap = list(pyplot.rcParams["keymap.save"])  # type: ignore[arg-type]
    monkeypatch.setattr(app_module, "_load_pyplot", lambda: pyplot)

    def create_partial_failure_figure(
        _pyplot: object,
    ) -> tuple[object, object, object]:
        return figure, object(), object()

    monkeypatch.setattr(
        app_module,
        "_create_figure",
        create_partial_failure_figure,
    )

    with pytest.raises(RuntimeError, match="callback registration failed"):
        run_visual_debugger(
            scenario_name="arena_5v5",
            seed=0,
            controlled_global_slot=None,
            static=False,
            verbose=False,
            show_ranges=True,
        )

    assert canvas.disconnected == [1]
    assert pyplot.rcParams["keymap.save"] == original_save_keymap
