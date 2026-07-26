"""Renderer-independent debugger input normalization and dispatch."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import isfinite

import numpy as np

from marl_battlegrounds.core.types import (
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_NORTHEAST,
    MOVE_NORTHWEST,
    MOVE_SOUTH,
    MOVE_SOUTHEAST,
    MOVE_SOUTHWEST,
    MOVE_STAY,
    MOVE_WEST,
)
from marl_battlegrounds.rendering.scene import AgentSceneV1
from scripts.dev.visual_debugger.control import (
    arm_basic,
    arm_ultimate,
    clear_pending_target,
    reset_session,
    select_clicked_target,
    select_controlled_actor,
    set_pending_movement,
    submit_interactive,
    submit_next_script_frame,
    switch_scenario,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.protocol import (
    BattlefieldPointerCommandV1,
    DebuggerCommandV1,
    KeyboardCommandV1,
    Preset,
    ResetCommandV1,
    RosterSelectionCommandV1,
    ScenarioSwitchCommandV1,
    SetPresetCommandV1,
    SetViewCommandV1,
    ViewMode,
)
from scripts.dev.visual_debugger.scenarios import (
    cycle_scenario_name,
    get_scenario,
    list_scenarios,
)
from scripts.dev.visual_debugger.scene_adapter import build_battlefield_scene

_MOVEMENT_KEYS = {
    "w": MOVE_NORTH,
    "s": MOVE_SOUTH,
    "d": MOVE_EAST,
    "a": MOVE_WEST,
    "q": MOVE_NORTHWEST,
    "e": MOVE_NORTHEAST,
    "z": MOVE_SOUTHWEST,
    "c": MOVE_SOUTHEAST,
    "x": MOVE_STAY,
    "arrowup": MOVE_NORTH,
    "arrowdown": MOVE_SOUTH,
    "arrowright": MOVE_EAST,
    "arrowleft": MOVE_WEST,
}
_SUBMISSION_KEYS = frozenset(("space", "enter", "n"))
_SHIFT_R_NOTICE = (
    "Shift+R cooldown clearing is unavailable because no public coherent "
    "snapshot-rebuild API exists; use R for a full reset."
)


@dataclass(frozen=True, slots=True)
class InputDispatchResult:
    """One authoritative input outcome for the browser debugger service."""

    session: DebuggerSession
    view_mode: ViewMode
    preset: Preset
    handled: bool
    changed: bool
    notice: str | None = None
    shutdown_requested: bool = False


def normalize_key(
    key: str | None,
    *,
    shift_key: bool | None = None,
) -> str | None:
    """Normalize supported keyboard aliases to debugger commands."""
    if key is None:
        return None
    if shift_key is None and key == "R":
        return "shift+r"
    normalized = key.lower()
    if normalized in (
        "shift+tab",
        "backtab",
        "iso_left_tab",
        "shift+iso_left_tab",
    ):
        return "shift+tab"
    if normalized in (" ", "spacebar"):
        return "space"
    if normalized == "return":
        return "enter"
    if normalized == "esc":
        return "escape"
    if normalized in ("up", "arrowup"):
        return "arrowup"
    if normalized in ("down", "arrowdown"):
        return "arrowdown"
    if normalized in ("right", "arrowright"):
        return "arrowright"
    if normalized in ("left", "arrowleft"):
        return "arrowleft"
    if normalized == "tab" and shift_key:
        return "shift+tab"
    if normalized in ("r", "shift+r") and (shift_key or normalized == "shift+r"):
        return "shift+r"
    return normalized


def _hit_test_rows(
    rows: Iterable[tuple[int, tuple[float, float], float, bool]],
    x: float,
    y: float,
) -> int | None:
    if not isfinite(x) or not isfinite(y):
        return None
    point = np.asarray((x, y), dtype=np.float32)
    candidates: list[tuple[float, int]] = []
    for global_slot, position, radius, active in rows:
        if not active or radius <= 0:
            continue
        center = np.asarray(position, dtype=np.float32)
        normalized_distance = float(np.linalg.norm(point - center) / radius)
        if normalized_distance <= 1.0:
            candidates.append((normalized_distance, global_slot))
    if not candidates:
        return None
    return min(candidates)[1]


def hit_test_scene_agents(
    agents: Iterable[AgentSceneV1],
    x: float,
    y: float,
) -> int | None:
    """Hit-test only agents authorized in the current serialized scene."""
    return _hit_test_rows(
        (
            (agent.global_slot, agent.position, agent.radius, agent.active)
            for agent in agents
        ),
        x,
        y,
    )


def sanitize_pov_pending_target(session: DebuggerSession) -> DebuggerSession:
    """Clear a pending target absent from the controlled actor's safe POV."""
    target = session.pending_action.selected_global_target_slot
    if target is None:
        return session
    scene = build_battlefield_scene(session, audience="agent_pov")
    authorized_slots = {agent.global_slot for agent in scene.agents}
    if target in authorized_slots:
        return session
    sanitized = clear_pending_target(session)
    if sanitized.pending_action == session.pending_action:
        return session
    return sanitized


def _result(
    session: DebuggerSession,
    *,
    view_mode: ViewMode,
    preset: Preset,
    handled: bool,
    changed: bool,
    notice: str | None = None,
    shutdown_requested: bool = False,
) -> InputDispatchResult:
    return InputDispatchResult(
        session=session,
        view_mode=view_mode,
        preset=preset,
        handled=handled,
        changed=changed,
        notice=notice,
        shutdown_requested=shutdown_requested,
    )


def _pending_edit_result(
    before: DebuggerSession,
    after: DebuggerSession,
    *,
    view_mode: ViewMode,
    preset: Preset,
) -> InputDispatchResult:
    changed = (
        before.controlled_global_slot != after.controlled_global_slot
        or before.pending_actions != after.pending_actions
    )
    return _result(
        after if changed else before,
        view_mode=view_mode,
        preset=preset,
        handled=True,
        changed=changed,
    )


def _is_terminal(session: DebuggerSession) -> bool:
    return bool(session.done_flags.terminated) or bool(session.done_flags.truncated)


def _terminal_notice(session: DebuggerSession) -> str:
    reason = "terminated" if bool(session.done_flags.terminated) else "truncated"
    return f"Episode is {reason}; reset or switch scenario to continue."


def _dispatch_keyboard(
    session: DebuggerSession,
    command: KeyboardCommandV1,
    *,
    view_mode: ViewMode,
    preset: Preset,
    include_stress: bool,
) -> InputDispatchResult:
    if command.ctrl_key or command.alt_key or command.meta_key:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=False,
            changed=False,
        )
    key = normalize_key(command.key, shift_key=command.shift_key)
    if key is None:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=False,
            changed=False,
        )
    if command.repeat and key in _SUBMISSION_KEYS:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice="Repeated submission input ignored.",
        )

    if key in ("tab", "shift+tab"):
        direction = -1 if key == "shift+tab" else 1
        active_slots = tuple(
            int(slot)
            for slot in np.flatnonzero(
                np.asarray(session.config.agent_profile.active_mask, dtype=bool)
            )
        )
        current_index = active_slots.index(session.controlled_global_slot)
        controlled_slot = active_slots[(current_index + direction) % len(active_slots)]
        edited = select_controlled_actor(session, controlled_slot)
        if view_mode == "pov":
            edited = sanitize_pov_pending_target(edited)
        return _pending_edit_result(
            session,
            edited,
            view_mode=view_mode,
            preset=preset,
        )

    if key == "escape":
        return _pending_edit_result(
            session,
            clear_pending_target(session),
            view_mode=view_mode,
            preset=preset,
        )

    scenario = get_scenario(session.scenario_name)
    terminal = _is_terminal(session)
    if scenario.mode == "scripted" and (
        key in _MOVEMENT_KEYS or key in ("1", "2", "space", "enter")
    ):
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=(
                "Scripted playback is inspection-only; press N to advance "
                "the registered frame."
            ),
        )
    if key in _MOVEMENT_KEYS:
        if terminal:
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=_terminal_notice(session),
            )
        return _pending_edit_result(
            session,
            set_pending_movement(session, _MOVEMENT_KEYS[key]),
            view_mode=view_mode,
            preset=preset,
        )
    if key in ("1", "2"):
        if terminal:
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=_terminal_notice(session),
            )
        edited = arm_basic(session) if key == "1" else arm_ultimate(session)
        return _pending_edit_result(
            session,
            edited,
            view_mode=view_mode,
            preset=preset,
        )

    if key == "n":
        if scenario.mode != "scripted":
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice="N advances scripted playback only.",
            )
        edited = submit_next_script_frame(session)
        changed = edited is not session
        notice = (
            _terminal_notice(session)
            if terminal
            else "Script is already complete."
            if edited is session
            else None
        )
        return _result(
            edited,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=changed,
            notice=notice,
        )
    if key in ("space", "enter"):
        edited = submit_interactive(
            session,
            actor_global_slots=(
                (session.controlled_global_slot,) if view_mode == "pov" else None
            ),
        )
        if view_mode == "pov":
            edited = sanitize_pov_pending_target(edited)
        changed = edited is not session
        notice = (
            _terminal_notice(session)
            if terminal
            else "Script is already complete."
            if edited is session
            else None
        )
        return _result(
            edited,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=changed,
            notice=notice,
        )
    if key == "r":
        return _result(
            reset_session(session),
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=True,
        )
    if key == "shift+r":
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=_SHIFT_R_NOTICE,
        )
    if key == "g":
        return _result(
            replace(session, show_ranges=not session.show_ranges),
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=True,
        )
    if key == "v":
        return _result(
            replace(session, verbose_logging=not session.verbose_logging),
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=True,
        )
    if key in ("[", "]"):
        direction = -1 if key == "[" else 1
        next_name = cycle_scenario_name(
            session.scenario_name,
            direction,
            include_stress=include_stress,
        )
        edited = switch_scenario(session, get_scenario(next_name))
        if view_mode == "pov":
            edited = sanitize_pov_pending_target(edited)
        return _result(
            edited,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=True,
        )
    return _result(
        session,
        view_mode=view_mode,
        preset=preset,
        handled=False,
        changed=False,
    )


def _dispatch_pointer(
    session: DebuggerSession,
    command: BattlefieldPointerCommandV1,
    *,
    view_mode: ViewMode,
    preset: Preset,
) -> InputDispatchResult:
    if command.ctrl_key or command.alt_key or command.meta_key:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=False,
            changed=False,
        )
    if command.button == "secondary":
        return _pending_edit_result(
            session,
            clear_pending_target(session),
            view_mode=view_mode,
            preset=preset,
        )
    scene = build_battlefield_scene(
        session,
        audience="researcher" if view_mode == "researcher" else "agent_pov",
    )
    target = hit_test_scene_agents(
        scene.agents,
        command.world_x,
        command.world_y,
    )
    if target is None:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
        )
    edited = (
        select_controlled_actor(session, target)
        if command.shift_key
        else select_clicked_target(session, target)
    )
    if view_mode == "pov" and command.shift_key:
        edited = sanitize_pov_pending_target(edited)
    return _pending_edit_result(
        session,
        edited,
        view_mode=view_mode,
        preset=preset,
    )


def _dispatch_roster_selection(
    session: DebuggerSession,
    command: RosterSelectionCommandV1,
    *,
    view_mode: ViewMode,
    preset: Preset,
) -> InputDispatchResult:
    scene = build_battlefield_scene(
        session,
        audience="researcher" if view_mode == "researcher" else "agent_pov",
    )
    if command.global_slot not in {agent.global_slot for agent in scene.agents}:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=f"Agent g{command.global_slot} is unavailable in this view.",
        )
    edited = (
        select_controlled_actor(session, command.global_slot)
        if command.role == "control"
        else select_clicked_target(session, command.global_slot)
    )
    if view_mode == "pov" and command.role == "control":
        edited = sanitize_pov_pending_target(edited)
    return _pending_edit_result(
        session,
        edited,
        view_mode=view_mode,
        preset=preset,
    )


def dispatch_command(
    session: DebuggerSession,
    command: DebuggerCommandV1,
    *,
    view_mode: ViewMode,
    preset: Preset,
    include_stress: bool,
) -> InputDispatchResult:
    """Apply one validated input without owning RNG or simulator semantics."""
    if isinstance(command, KeyboardCommandV1):
        return _dispatch_keyboard(
            session,
            command,
            view_mode=view_mode,
            preset=preset,
            include_stress=include_stress,
        )
    if isinstance(command, BattlefieldPointerCommandV1):
        return _dispatch_pointer(
            session,
            command,
            view_mode=view_mode,
            preset=preset,
        )
    if isinstance(command, RosterSelectionCommandV1):
        return _dispatch_roster_selection(
            session,
            command,
            view_mode=view_mode,
            preset=preset,
        )
    if isinstance(command, ScenarioSwitchCommandV1):
        allowed_names = {
            scenario.name for scenario in list_scenarios(include_stress=include_stress)
        }
        if command.scenario_name not in allowed_names:
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=f"Scenario {command.scenario_name!r} is unavailable.",
            )
        if command.scenario_name == session.scenario_name:
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
            )
        edited = switch_scenario(session, get_scenario(command.scenario_name))
        if view_mode == "pov":
            edited = sanitize_pov_pending_target(edited)
        return _result(
            edited,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=True,
        )
    if isinstance(command, ResetCommandV1):
        edited = reset_session(session)
        if view_mode == "pov":
            edited = sanitize_pov_pending_target(edited)
        return _result(
            edited,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=True,
        )
    if isinstance(command, SetViewCommandV1):
        edited = (
            sanitize_pov_pending_target(session)
            if command.view_mode == "pov"
            else session
        )
        changed = command.view_mode != view_mode or edited is not session
        return _result(
            edited,
            view_mode=command.view_mode,
            preset=preset,
            handled=True,
            changed=changed,
        )
    if isinstance(command, SetPresetCommandV1):
        return _result(
            session,
            view_mode=view_mode,
            preset=command.preset,
            handled=True,
            changed=command.preset != preset,
        )
    return _result(
        session,
        view_mode=view_mode,
        preset=preset,
        handled=True,
        changed=False,
        notice="Debugger shutdown requested.",
        shutdown_requested=True,
    )
