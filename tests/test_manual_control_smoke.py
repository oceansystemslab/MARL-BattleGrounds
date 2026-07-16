"""Smoke tests for optional manual-control rendering helpers."""

from collections.abc import Callable
from importlib import import_module
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

import marl_battlegrounds.rendering.manual_control as manual_control_module
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.types import (
    CLASS_NEUTRAL,
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
    NUM_MOVE_ACTIONS,
    OBSTACLE_FEATURES,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.rendering.geometry import RenderResult
from marl_battlegrounds.rendering.manual_control import (
    KEY_TO_MOVE_ACTION,
    build_manual_joint_action,
    movement_from_key,
    step_manual_control,
)

_EXPECTED_KEY_TO_MOVE_ACTION: dict[str, int] = {
    "w": MOVE_NORTH,
    "s": MOVE_SOUTH,
    "d": MOVE_EAST,
    "a": MOVE_WEST,
    "q": MOVE_NORTHWEST,
    "e": MOVE_NORTHEAST,
    "z": MOVE_SOUTHWEST,
    "c": MOVE_SOUTHEAST,
}

_SUPPORTED_MOVEMENT_KEY_CASES: tuple[tuple[str, int], ...] = tuple(
    _EXPECTED_KEY_TO_MOVE_ACTION.items()
) + tuple(
    (key.upper(), move_action)
    for key, move_action in _EXPECTED_KEY_TO_MOVE_ACTION.items()
)

_STAY_FALLBACK_KEYS: tuple[str | None, ...] = (
    None,
    "",
    "space",
    "x",
    "ctrl+w",
)

_VALID_MANUAL_ACTION_CASES: tuple[tuple[int, int], ...] = (
    (0, MOVE_STAY),
    (2, MOVE_NORTHEAST),
    (MAX_AGENT_SLOTS - 1, MOVE_SOUTHWEST),
)

_INVALID_CONTROLLED_SLOTS: tuple[int, ...] = (-1, MAX_AGENT_SLOTS)
_INVALID_MOVE_ACTIONS: tuple[int, ...] = (-1, NUM_MOVE_ACTIONS)

type CoreStepOutput = tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]


class _FakeKeyEvent:
    """Minimal key event used by the fake interactive loop."""

    def __init__(self, key: str | None) -> None:
        self.key = key


class _FakeTimer:
    """Minimal timer used to exercise manual-control callbacks."""

    def __init__(self) -> None:
        self.callback: Callable[[], bool] | None = None
        self.started = False

    def add_callback(self, func: Callable[[], bool]) -> object:
        self.callback = func
        return func

    def start(self) -> object:
        self.started = True
        return None


class _FakeCanvas:
    """Minimal canvas used to register keypress and timer callbacks."""

    def __init__(self, timer: _FakeTimer) -> None:
        self.timer = timer
        self.draw_idle_count = 0
        self.key_callback: Callable[[_FakeKeyEvent], None] | None = None
        self.timer_interval: int | None = None

    def mpl_connect(
        self, event_name: str, callback: Callable[[_FakeKeyEvent], None]
    ) -> int:
        assert event_name == "key_press_event"
        self.key_callback = callback
        return 1

    def new_timer(self, *, interval: int | None = None) -> _FakeTimer:
        self.timer_interval = interval
        return self.timer

    def draw_idle(self) -> object:
        self.draw_idle_count += 1
        return None


class _FakeFigure:
    """Minimal figure wrapper exposing a fake canvas."""

    def __init__(self, canvas: _FakeCanvas) -> None:
        self.canvas = canvas


class _FakePyplot:
    """Minimal pyplot facade that drives two timer ticks without a GUI."""

    def __init__(self, canvas: _FakeCanvas) -> None:
        self.canvas = canvas
        self.show_count = 0
        self.rcParams: dict[str, object] = {
            "keymap.back": ["left", "c", "backspace"],
            "keymap.quit": ["q", "ctrl+w"],
            "keymap.save": ["s", "ctrl+s"],
            "keymap.pan": ["p"],
            "figure.max_open_warning": 20,
        }
        self.rc_params_during_show: dict[str, object] | None = None

    def show(self) -> object:
        self.show_count += 1
        self.rc_params_during_show = {
            key: list(cast(list[object], value)) if isinstance(value, list) else value
            for key, value in self.rcParams.items()
        }

        if self.canvas.key_callback is not None:
            self.canvas.key_callback(_FakeKeyEvent("D"))

        if self.canvas.timer.callback is not None:
            assert self.canvas.timer.callback()
            assert self.canvas.timer.callback()

        return None


def _empty_obstacles() -> Array:
    """Create a padded all-inactive obstacle table."""
    return jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _sample_config() -> EnvConfig:
    """Create a small deterministic manual-control smoke-test config."""
    return EnvConfig(
        team_size=1,
        max_steps=10,
        map_width=12.0,
        map_height=8.0,
        obstacles=_empty_obstacles(),
        agent_profile=resolve_agent_profile(
            jnp.full((MAX_AGENT_SLOTS,), CLASS_NEUTRAL, dtype=jnp.int32),
            jnp.asarray((1, 1), dtype=jnp.int32),
        ),
    )


def _expected_manual_move_choices(controlled_slot: int, move_action: int) -> Array:
    """Build the expected fixed-slot movement head for manual action tests."""
    expected_moves = jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32)
    return expected_moves.at[controlled_slot].set(move_action)


def _assert_int_action_head_contract(action_head: Array) -> None:
    """Assert the shared fixed-slot integer action-head contract."""
    assert action_head.shape == (MAX_AGENT_SLOTS,)
    assert action_head.dtype == jnp.int32


def _assert_manual_joint_action_contract(action: Action) -> None:
    """Assert manual action construction preserves the core action contract."""
    _assert_int_action_head_contract(action.move)
    _assert_int_action_head_contract(action.select_target)
    _assert_int_action_head_contract(action.use_ultimate)


def _assert_manual_step_output_matches_expected(
    actual: manual_control_module.ManualStepOutput,
    expected_next_key: Array,
    expected_step_output: CoreStepOutput,
) -> None:
    """Assert manual stepping returns the next key and core step outputs."""
    next_key, next_state, obs, reward, done_flags, action_mask, info = actual
    (
        expected_state,
        expected_obs,
        expected_reward,
        expected_done_flags,
        expected_action_mask,
        expected_info,
    ) = expected_step_output

    assert bool(jnp.array_equal(next_key, expected_next_key))
    assert bool(jnp.array_equal(next_state.step_count, expected_state.step_count))
    assert bool(
        jnp.array_equal(next_state.agent_positions, expected_state.agent_positions)
    )
    assert bool(jnp.array_equal(obs.self_features, expected_obs.self_features))
    assert bool(jnp.array_equal(reward.rewards, expected_reward.rewards))
    assert bool(jnp.array_equal(done_flags.terminated, expected_done_flags.terminated))
    assert bool(jnp.array_equal(done_flags.truncated, expected_done_flags.truncated))
    assert bool(jnp.array_equal(action_mask.move_mask, expected_action_mask.move_mask))
    assert bool(
        jnp.array_equal(
            action_mask.select_target_mask, expected_action_mask.select_target_mask
        )
    )
    assert bool(
        jnp.array_equal(
            action_mask.use_ultimate_mask,
            expected_action_mask.use_ultimate_mask,
        )
    )
    assert isinstance(info, Info)
    assert isinstance(expected_info, Info)


def test_manual_control_module_imports_without_visualization_dependency() -> None:
    """Manual-control helpers should import without importing Matplotlib."""
    manual_control_module = import_module("marl_battlegrounds.rendering.manual_control")

    assert (
        manual_control_module.__name__ == "marl_battlegrounds.rendering.manual_control"
    )


def test_key_to_move_action_contains_only_expected_lowercase_bindings() -> None:
    """Manual movement bindings should stay explicit and lowercase."""
    assert dict(KEY_TO_MOVE_ACTION) == _EXPECTED_KEY_TO_MOVE_ACTION


@pytest.mark.parametrize(("key", "expected_action"), _SUPPORTED_MOVEMENT_KEY_CASES)
def test_movement_from_key_maps_supported_keyboard_inputs_to_move_actions(
    key: str, expected_action: int
) -> None:
    """Supported movement keys should map to existing move action ids."""
    assert movement_from_key(key) == expected_action


@pytest.mark.parametrize("key", _STAY_FALLBACK_KEYS)
def test_movement_from_key_maps_absent_or_unknown_input_to_stay(
    key: str | None,
) -> None:
    """No input and unsupported keys should emit an explicit stay action."""
    assert movement_from_key(key) == MOVE_STAY


@pytest.mark.parametrize(("controlled_slot", "move_action"), _VALID_MANUAL_ACTION_CASES)
def test_build_manual_joint_action_controls_only_selected_slot(
    controlled_slot: int, move_action: int
) -> None:
    """The action builder should leave non-controlled slots on safe defaults."""
    action = build_manual_joint_action(
        _sample_config(),
        controlled_slot,
        move_action,
    )

    _assert_manual_joint_action_contract(action)
    assert bool(
        jnp.array_equal(
            action.move,
            _expected_manual_move_choices(controlled_slot, move_action),
        )
    )
    assert bool(
        jnp.array_equal(action.select_target, jnp.zeros_like(action.select_target))
    )
    assert bool(
        jnp.array_equal(action.use_ultimate, jnp.zeros_like(action.use_ultimate))
    )


@pytest.mark.parametrize("controlled_slot", _INVALID_CONTROLLED_SLOTS)
def test_build_manual_joint_action_rejects_invalid_controlled_slots(
    controlled_slot: int,
) -> None:
    """Host-side slot validation should reject impossible global slots."""
    with pytest.raises(ValueError, match="controlled_slot"):
        build_manual_joint_action(_sample_config(), controlled_slot, MOVE_STAY)


@pytest.mark.parametrize("move_action", _INVALID_MOVE_ACTIONS)
def test_build_manual_joint_action_rejects_invalid_move_actions(
    move_action: int,
) -> None:
    """Host-side movement validation should reject impossible move action ids."""
    with pytest.raises(ValueError, match="move_action"):
        build_manual_joint_action(_sample_config(), 0, move_action)


def test_step_manual_control_returns_next_key_and_step_outputs() -> None:
    """The single-step helper should drive the real simulator transition."""
    config = _sample_config()
    state, _obs, _action_mask, _info = reset(config, jax.random.key(0))
    key = jax.random.key(123)

    result = step_manual_control(
        config,
        state,
        key,
        controlled_slot=0,
        input_key="d",
    )
    next_key, next_state, obs, reward, done_flags, action_mask, info = result
    expected_next_key, _step_key = jax.random.split(key)

    assert next_key.shape == key.shape
    assert bool(jnp.array_equal(next_key, expected_next_key))
    assert next_state.agent_positions.shape == state.agent_positions.shape
    assert obs.self_features.shape[0] == MAX_AGENT_SLOTS
    assert reward.rewards.shape == (MAX_AGENT_SLOTS,)
    assert done_flags.terminated.shape == ()
    assert done_flags.truncated.shape == ()
    assert action_mask.move_mask.shape == (MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS)
    assert isinstance(info, Info)


def test_step_manual_control_matches_direct_core_step_outputs() -> None:
    """Manual stepping should be a thin wrapper around the core step."""
    config = _sample_config()
    state, _obs, _action_mask, _info = reset(config, jax.random.key(0))
    key = jax.random.key(123)
    next_key, step_key = jax.random.split(key)
    expected_action = build_manual_joint_action(config, 0, MOVE_EAST)
    expected_step_output = step(config, state, expected_action, step_key)

    actual = step_manual_control(
        config,
        state,
        key,
        controlled_slot=0,
        input_key="d",
    )

    _assert_manual_step_output_matches_expected(
        actual,
        next_key,
        expected_step_output,
    )


def test_run_manual_control_consumes_pending_key_and_redraws_without_gui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interactive driver should wire callbacks without opening a GUI."""
    config = _sample_config()
    initial_state, obs, action_mask, info = reset(config, jax.random.key(0))
    reward = Reward(rewards=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32))
    done_flags = DoneFlags(
        terminated=jnp.array(False),
        truncated=jnp.array(False),
    )
    timer = _FakeTimer()
    canvas = _FakeCanvas(timer)
    figure = _FakeFigure(canvas)
    pyplot = _FakePyplot(canvas)
    render_result = RenderResult(figure=figure, axes=object())
    step_input_keys: list[str | None] = []
    redraw_states: list[EnvState] = []

    def fake_load_pyplot() -> _FakePyplot:
        return pyplot

    def fake_render_geometry(
        config_arg: EnvConfig,
        state_arg: EnvState,
        *,
        show_agent_indices: bool = True,
    ) -> RenderResult:
        assert config_arg is config
        assert state_arg is initial_state
        assert show_agent_indices is False
        return render_result

    def fake_step_manual_control(
        config_arg: EnvConfig,
        state_arg: EnvState,
        key_arg: Array,
        *,
        controlled_slot: int = 0,
        input_key: str | None = None,
    ) -> manual_control_module.ManualStepOutput:
        assert config_arg is config
        assert controlled_slot == 3
        step_input_keys.append(input_key)
        next_key, _step_key = jax.random.split(key_arg)
        next_state = state_arg._replace(step_count=state_arg.step_count + 1)
        return (
            next_key,
            next_state,
            obs,
            reward,
            done_flags,
            action_mask,
            info,
        )

    def fake_redraw_geometry(
        config_arg: EnvConfig,
        state_arg: EnvState,
        result_arg: RenderResult,
        *,
        show_agent_indices: bool = True,
    ) -> RenderResult:
        assert config_arg is config
        assert result_arg is render_result
        assert show_agent_indices is False
        redraw_states.append(state_arg)
        return result_arg

    monkeypatch.setattr(manual_control_module, "_load_pyplot", fake_load_pyplot)
    monkeypatch.setattr(manual_control_module, "render_geometry", fake_render_geometry)
    monkeypatch.setattr(
        manual_control_module,
        "step_manual_control",
        fake_step_manual_control,
    )
    monkeypatch.setattr(manual_control_module, "redraw_geometry", fake_redraw_geometry)

    final_state = manual_control_module.run_manual_control(
        config,
        initial_state,
        jax.random.key(99),
        controlled_slot=3,
        step_interval_ms=123,
        show_agent_indices=False,
    )

    assert timer.started
    assert canvas.timer_interval == 123
    assert canvas.draw_idle_count == 2
    assert pyplot.show_count == 1
    assert pyplot.rc_params_during_show == {
        "keymap.back": ["left", "backspace"],
        "keymap.quit": ["ctrl+w"],
        "keymap.save": ["ctrl+s"],
        "keymap.pan": ["p"],
        "figure.max_open_warning": 20,
    }
    assert pyplot.rcParams == {
        "keymap.back": ["left", "c", "backspace"],
        "keymap.quit": ["q", "ctrl+w"],
        "keymap.save": ["s", "ctrl+s"],
        "keymap.pan": ["p"],
        "figure.max_open_warning": 20,
    }
    assert step_input_keys == ["D", None]
    assert len(redraw_states) == 2
    assert final_state is redraw_states[-1]
    assert bool(jnp.array_equal(final_state.step_count, initial_state.step_count + 2))
