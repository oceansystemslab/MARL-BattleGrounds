"""Development-only manual-control harness for simulator debugging."""

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.env import step
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
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
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.rendering.geometry import redraw_geometry, render_geometry

type ManualStepOutput = tuple[
    Array,
    EnvState,
    Observation,
    Reward,
    DoneFlags,
    ActionMask,
    Info,
]

KEY_TO_MOVE_ACTION: Mapping[str, int] = {
    "w": MOVE_NORTH,
    "s": MOVE_SOUTH,
    "d": MOVE_EAST,
    "a": MOVE_WEST,
    "q": MOVE_NORTHWEST,
    "e": MOVE_NORTHEAST,
    "z": MOVE_SOUTHWEST,
    "c": MOVE_SOUTHEAST,
}
_MANUAL_CONTROL_KEYS: frozenset[str] = frozenset(KEY_TO_MOVE_ACTION)
_KeymapSnapshot = dict[str, object]


class _KeyEventLike(Protocol):
    """Small subset of Matplotlib key events used by manual control."""

    key: str | None


class _TimerLike(Protocol):
    """Small subset of Matplotlib timers used by manual control."""

    def add_callback(self, func: Callable[[], bool]) -> object: ...

    def start(self) -> object: ...


class _CanvasLike(Protocol):
    """Small subset of the Matplotlib canvas API used by manual control."""

    def mpl_connect(
        self, event_name: str, callback: Callable[[_KeyEventLike], None]
    ) -> int: ...

    def new_timer(self, *, interval: int | None = None) -> _TimerLike: ...

    def draw_idle(self) -> object: ...


class _FigureLike(Protocol):
    """Small subset of the Matplotlib figure API used by manual control."""

    canvas: _CanvasLike


class _PyplotLike(Protocol):
    """Small subset of Matplotlib pyplot used by manual control."""

    rcParams: MutableMapping[str, object]  # noqa: N815 - mirrors Matplotlib API.

    def show(self) -> object: ...


@dataclass
class _ManualLoopState:
    """Mutable host-side state for the interactive debug loop."""

    key: Array
    state: EnvState
    action_mask: ActionMask
    pending_input_key: str | None = None


def movement_from_key(key: str | None) -> int:
    """Map a keyboard input to a movement action id.

    Supported movement keys are case-insensitive. Unknown keys, empty strings,
    and missing input all map to ``MOVE_STAY`` so an unpressed timestep is an
    explicit no-op.
    """
    if key is None:
        return MOVE_STAY

    return KEY_TO_MOVE_ACTION.get(key.lower(), MOVE_STAY)


def build_manual_joint_action(
    config: EnvConfig, controlled_slot: int, move_action: int
) -> Action:
    """Build a joint action with one manually controlled movement slot.

    The helper only constructs the factored action object. It does not inspect
    masks or decide whether an action is legal; the simulator transition owns
    those semantics. ``config`` is accepted for API consistency with the other
    manual-control helpers, but active-slot legality is not inferred here.
    """
    _ = config
    _validate_controlled_slot(controlled_slot)
    _validate_move_action(move_action)

    move = jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32)
    move = move.at[controlled_slot].set(move_action)

    return Action(
        move=move,
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def step_manual_control(
    config: EnvConfig,
    current_state: EnvState,
    current_action_mask: ActionMask,
    key: Array,
    *,
    controlled_slot: int = 0,
    input_key: str | None = None,
) -> ManualStepOutput:
    """Advance one manual-control timestep through the real simulator step.

    The returned key is the next key for the manual-control loop; the remaining
    outputs are exactly the current core ``step`` outputs. The supplied mask
    must be the one paired with ``current_state``.
    """
    move_action = movement_from_key(input_key)
    action = build_manual_joint_action(config, controlled_slot, move_action)
    next_key, step_key = jax.random.split(key)
    next_state, obs, reward, done_flags, action_mask, info = step(
        config,
        current_state,
        current_action_mask,
        action,
        step_key,
    )

    return (next_key, next_state, obs, reward, done_flags, action_mask, info)


def run_manual_control(
    config: EnvConfig,
    initial_state: EnvState,
    initial_action_mask: ActionMask,
    key: Array,
    *,
    controlled_slot: int = 0,
    step_interval_ms: int = 200,
    show_agent_indices: bool = True,
) -> EnvState:
    """Run a Matplotlib manual-control debug loop.

    Key presses are consumed one timestep at a time. If no supported movement
    key is pending when the timer fires, the controlled agent emits
    ``MOVE_STAY`` for that timestep.
    """
    _validate_controlled_slot(controlled_slot)
    pyplot = _load_pyplot()
    keymap_snapshot = _reserve_manual_control_keys(pyplot)

    try:
        result = render_geometry(
            config,
            initial_state,
            show_agent_indices=show_agent_indices,
        )
        figure = cast(_FigureLike, result.figure)
        loop_state = _ManualLoopState(
            key=key,
            state=initial_state,
            action_mask=initial_action_mask,
        )

        def _on_key_press(event: _KeyEventLike) -> None:
            loop_state.pending_input_key = event.key

        def _on_timer() -> bool:
            (
                loop_state.key,
                loop_state.state,
                _observation,
                _reward,
                done_flags,
                loop_state.action_mask,
                _info,
            ) = step_manual_control(
                config,
                loop_state.state,
                loop_state.action_mask,
                loop_state.key,
                controlled_slot=controlled_slot,
                input_key=loop_state.pending_input_key,
            )
            loop_state.pending_input_key = None
            redraw_geometry(
                config,
                loop_state.state,
                result,
                show_agent_indices=show_agent_indices,
            )
            figure.canvas.draw_idle()

            return not bool(done_flags.done)

        figure.canvas.mpl_connect("key_press_event", _on_key_press)
        timer = figure.canvas.new_timer(interval=step_interval_ms)
        timer.add_callback(_on_timer)
        timer.start()
        pyplot.show()

        return loop_state.state
    finally:
        _restore_matplotlib_keymaps(pyplot, keymap_snapshot)


def _validate_controlled_slot(controlled_slot: int) -> None:
    """Validate host-side manual-control slot selection."""
    if not 0 <= controlled_slot < MAX_AGENT_SLOTS:
        msg = (
            "controlled_slot must be a valid global agent slot in "
            f"[0, {MAX_AGENT_SLOTS}); got {controlled_slot}."
        )
        raise ValueError(msg)


def _validate_move_action(move_action: int) -> None:
    """Validate host-side manual movement action selection."""
    if not 0 <= move_action < NUM_MOVE_ACTIONS:
        msg = (
            "move_action must be a valid movement action id in "
            f"[0, {NUM_MOVE_ACTIONS}); got {move_action}."
        )
        raise ValueError(msg)


def _reserve_manual_control_keys(pyplot: _PyplotLike) -> _KeymapSnapshot:
    """Temporarily remove movement keys from Matplotlib default shortcuts."""
    original_keymaps: _KeymapSnapshot = {}

    for rc_param_name, rc_param_value in tuple(pyplot.rcParams.items()):
        if not rc_param_name.startswith("keymap."):
            continue
        if not isinstance(rc_param_value, list):
            continue

        keymap_entries = cast(list[object], rc_param_value)
        filtered_keymap: list[object] = [
            key
            for key in keymap_entries
            if not (isinstance(key, str) and key.lower() in _MANUAL_CONTROL_KEYS)
        ]
        if filtered_keymap == keymap_entries:
            continue

        original_keymaps[rc_param_name] = list(keymap_entries)
        pyplot.rcParams[rc_param_name] = filtered_keymap

    return original_keymaps


def _restore_matplotlib_keymaps(
    pyplot: _PyplotLike, keymap_snapshot: _KeymapSnapshot
) -> None:
    """Restore Matplotlib shortcut keymaps after the manual loop exits."""
    for rc_param_name, rc_param_value in keymap_snapshot.items():
        pyplot.rcParams[rc_param_name] = rc_param_value


def _load_pyplot() -> _PyplotLike:
    """Load Matplotlib lazily for the interactive entrypoint."""
    try:
        return cast(_PyplotLike, import_module("matplotlib.pyplot"))
    except ImportError as exc:
        msg = (
            "run_manual_control requires the optional visualization dependency "
            "'matplotlib'. Install marl-battlegrounds with the 'viz' extra to "
            "use the manual-control harness."
        )
        raise ImportError(msg) from exc
