"""Renderer-independent debugger input normalization and dispatch."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import hypot, isfinite
from typing import Literal

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
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.pov import build_actor_pov_current_slice_v1
from marl_battlegrounds.rendering.evaluation_adapter import (
    EvaluationScenePresentationStateV1,
    build_researcher_analyzer_projection_v2,
)
from marl_battlegrounds.rendering.pov_scene import (
    build_actor_pov_analyzer_projection_v1,
)
from marl_battlegrounds.rendering.scene import AgentSceneV1, AgentSceneV2
from scripts.dev.visual_debugger.control import (
    DebuggerTransitionFailureV1,
    arm_basic,
    arm_ultimate,
    clear_pending_target,
    lane_availability,
    reset_session,
    select_clicked_target,
    select_controlled_actor,
    select_no_combat,
    set_pending_movement,
    submit_interactive,
    submit_next_script_frame,
)
from scripts.dev.visual_debugger.model import DebuggerSession, RawContinuationIdentity
from scripts.dev.visual_debugger.protocol import (
    ActorPovTargetActionCommandV1,
    BattlefieldPointerCommandV1,
    DebuggerCommandV1,
    ExitCommandV1,
    KeyboardCommandV1,
    Preset,
    ResetCommandV1,
    RosterSelectionCommandV1,
    ScenarioSwitchCommandV1,
    SetPresetCommandV1,
    SetViewCommandV1,
    ViewMode,
)
from scripts.dev.visual_debugger.scenarios import get_scenario

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

type RecordingRestartIntentV1 = Literal["reset"]


def recording_restart_intent_v1(
    session: DebuggerSession,
    command: DebuggerCommandV1,
    *,
    view_mode: ViewMode,
    include_stress: bool,
) -> RecordingRestartIntentV1 | None:
    """Classify the sole public episode replacement before dispatch constructs it."""
    del session, view_mode, include_stress
    if isinstance(command, KeyboardCommandV1):
        if command.ctrl_key or command.alt_key or command.meta_key:
            return None
        key = normalize_key(command.key, shift_key=command.shift_key)
        if key == "r":
            return "reset"
        return None
    if isinstance(command, ResetCommandV1):
        return "reset"
    return None


_SUBMISSION_KEYS = frozenset(("space", "enter", "n"))

_SCRIPTED_INSPECTION_NOTICE = (
    "Scripted playback is inspection-only; press N without modifiers to advance "
    "the registered frame."
)


def _scripted_inspection_command_is_allowed(command: DebuggerCommandV1) -> bool:
    """Keep scripted playback to its closed advance and presentation surface."""
    if isinstance(command, KeyboardCommandV1):
        if (
            command.shift_key
            or command.ctrl_key
            or command.alt_key
            or command.meta_key
            or command.repeat
        ):
            return False
        return normalize_key(command.key, shift_key=False) in ("n", "g")
    return isinstance(command, (SetViewCommandV1, ExitCommandV1))


@dataclass(frozen=True, slots=True)
class InputDispatchResult:
    """One authoritative input outcome for the browser debugger service."""

    session: DebuggerSession
    view_mode: ViewMode
    preset: Preset
    handled: bool
    changed: bool
    transition_applied: EvaluationTransitionViewV1 | None = None
    episode_restarted: bool = False
    raw_continuation_identity: RawContinuationIdentity | None = None
    notice: str | None = None
    shutdown_requested: bool = False

    def __post_init__(self) -> None:
        transition = self.transition_applied
        if transition is not None:
            if type(transition) is not EvaluationTransitionViewV1:
                raise TypeError("transition_applied must be an exact coherent V1 view.")
            if self.episode_restarted:
                raise ValueError(
                    "one input result cannot both apply a transition and restart."
                )
            if not self.changed or self.session.incoming_evaluation_view != transition:
                raise ValueError(
                    "transition_applied must identify the candidate session view."
                )
        if type(self.episode_restarted) is not bool:
            raise TypeError("episode_restarted must be a Python bool.")
        if self.episode_restarted and (
            not self.changed
            or self.session.current_evaluation_frame.frame_index != 0
            or self.session.incoming_evaluation_view is not None
        ):
            raise ValueError(
                "restarted results must expose a fresh frame-zero episode."
            )
        has_scientific_marker = transition is not None or self.episode_restarted
        if has_scientific_marker:
            if (
                type(self.raw_continuation_identity) is not RawContinuationIdentity
                or self.raw_continuation_identity
                is not self.session.raw_continuation_identity
            ):
                raise ValueError(
                    "transition and restart results must bind the candidate raw "
                    "continuation identity."
                )
        elif self.raw_continuation_identity is not None:
            raise ValueError(
                "UI-only results must not carry a raw continuation identity."
            )


def normalize_key(
    key: str | None,
    *,
    shift_key: bool | None = None,
) -> str | None:
    """Normalize supported keyboard aliases to debugger commands."""
    if key is None:
        return None
    normalized = key.lower()
    if normalized == "shift+r" or (normalized == "r" and shift_key is True):
        return None
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
    return normalized


def _hit_test_rows(
    rows: Iterable[tuple[int, tuple[float, float], float, bool]],
    x: float,
    y: float,
) -> int | None:
    if not isfinite(x) or not isfinite(y):
        return None
    candidates: list[tuple[float, int]] = []
    for global_slot, position, radius, active in rows:
        if not active or radius <= 0:
            continue
        normalized_distance = hypot(x - position[0], y - position[1]) / radius
        if normalized_distance <= 1.0:
            candidates.append((normalized_distance, global_slot))
    if not candidates:
        return None
    return min(candidates)[1]


def hit_test_scene_agents(
    agents: Iterable[AgentSceneV1 | AgentSceneV2],
    x: float,
    y: float,
) -> int | None:
    """Hit-test only agents authorized in the current serialized scene."""
    return _hit_test_rows(
        (
            (
                agent.global_slot,
                agent.position,
                agent.radius,
                agent.active if isinstance(agent, AgentSceneV1) else True,
            )
            for agent in agents
        ),
        x,
        y,
    )


def _result(
    session: DebuggerSession,
    *,
    view_mode: ViewMode,
    preset: Preset,
    handled: bool,
    changed: bool,
    transition_applied: EvaluationTransitionViewV1 | None = None,
    episode_restarted: bool = False,
    notice: str | None = None,
    shutdown_requested: bool = False,
) -> InputDispatchResult:
    return InputDispatchResult(
        session=session,
        view_mode=view_mode,
        preset=preset,
        handled=handled,
        changed=changed,
        transition_applied=transition_applied,
        episode_restarted=episode_restarted,
        raw_continuation_identity=(
            session.raw_continuation_identity
            if transition_applied is not None or episode_restarted
            else None
        ),
        notice=notice,
        shutdown_requested=shutdown_requested,
    )


def _applied_transition_result(
    session: DebuggerSession,
    *,
    view_mode: ViewMode,
    preset: Preset,
) -> InputDispatchResult:
    """Package one captured transition behind the typed validation boundary."""
    try:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=True,
            transition_applied=session.incoming_evaluation_view,
        )
    except DebuggerTransitionFailureV1:
        raise
    except Exception as error:
        raise DebuggerTransitionFailureV1(
            "validation",
            "transition_packaging_failed",
        ) from error


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
    return session.episode_sealed


def _terminal_notice(session: DebuggerSession) -> str:
    reason = (
        "terminated"
        if session.terminated
        else "truncated"
        if session.truncated
        else "at its declared horizon"
    )
    return f"Episode is {reason}; reset or switch scenario to continue."


def _target_action(
    session: DebuggerSession,
    actor_global_slot: int,
    target_global_slot: int | None,
) -> int:
    if target_global_slot is None:
        return 0
    catalog = session.evaluation_context.static_mechanics_catalog
    mapping = catalog.global_recipient_slot_by_actor_and_target_action[
        actor_global_slot
    ]
    try:
        return mapping.index(target_global_slot)
    except ValueError as error:
        raise ValueError("target is absent from the serialized action axis") from error


def _authorized_pointer_rows(
    session: DebuggerSession,
    *,
    view_mode: ViewMode,
) -> tuple[tuple[int, tuple[float, float], float, bool], ...]:
    if view_mode == "researcher":
        projection = build_researcher_analyzer_projection_v2(
            session.evaluation_context,
            session.current_evaluation_frame,
            transition_view=session.incoming_evaluation_view,
            presentation=EvaluationScenePresentationStateV1(
                controlled_global_slot=session.controlled_global_slot,
                selected_global_slot=session.controlled_global_slot,
                show_ranges=session.show_ranges,
            ),
            status_source_evidence_state=session.status_source_evidence_state,
        )
        return tuple(
            (agent.global_slot, agent.position, agent.radius, True)
            for agent in projection.scene.agents
        )
    slice_ = build_actor_pov_current_slice_v1(
        session.evaluation_context,
        session.current_evaluation_frame,
        global_slot=session.controlled_global_slot,
        incoming_transition_view=session.incoming_evaluation_view,
    )
    projection = build_actor_pov_analyzer_projection_v1(slice_)
    slot_by_public_id = {
        row.public_agent_id: row.global_slot
        for row in session.evaluation_context.roster
    }
    self_actor = projection.scene.self_actor
    return (
        (
            self_actor.global_slot,
            self_actor.position,
            self_actor.radius,
            True,
        ),
        *(
            (
                slot_by_public_id[body.public_agent_id],
                body.position,
                body.radius,
                True,
            )
            for body in projection.scene.visible_bodies
        ),
    )


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
            row.global_slot
            for row in session.evaluation_context.roster
            if row.configured_active
        )
        current_index = active_slots.index(session.controlled_global_slot)
        controlled_slot = active_slots[(current_index + direction) % len(active_slots)]
        edited = select_controlled_actor(session, controlled_slot)
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
        key in _MOVEMENT_KEYS or key in ("0", "1", "2", "space", "enter")
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
        move_action = _MOVEMENT_KEYS[key]
        if not bool(
            session.current_evaluation_frame.action_mask.move_mask[
                session.controlled_global_slot
            ][move_action]
        ):
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=(
                    f"Movement action {move_action} is unavailable in the "
                    "current action mask; pending action unchanged."
                ),
            )
        return _pending_edit_result(
            session,
            set_pending_movement(session, move_action),
            view_mode=view_mode,
            preset=preset,
        )
    if key in ("0", "1", "2"):
        if terminal:
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=_terminal_notice(session),
            )
        if key == "0":
            return _pending_edit_result(
                session,
                select_no_combat(session),
                view_mode=view_mode,
                preset=preset,
            )
        edited = arm_basic(session) if key == "1" else arm_ultimate(session)
        edited_pending = edited.pending_action
        target_action = _target_action(
            edited,
            edited.controlled_global_slot,
            edited_pending.selected_global_target_slot,
        )
        if key == "1" and target_action == 0:
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=(
                    "Basic requires a selected target; target-none lane zero is "
                    "the canonical no-combat tuple. Pending action unchanged."
                ),
            )
        availability = lane_availability(
            session.current_evaluation_frame.action_mask,
            session.controlled_global_slot,
            target_action,
            edited_pending.armed_lane,
        )
        if not availability.armed_pair_legal:
            ability = "Basic" if key == "1" else "Ultimate"
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=(
                    f"{ability} is unavailable for the selected target in the "
                    "current action mask; pending action unchanged."
                ),
            )
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
        if changed:
            return _applied_transition_result(
                edited,
                view_mode=view_mode,
                preset=preset,
            )
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
        edited = submit_interactive(session, actor_global_slots=None)
        transition_applied = edited is not session
        if transition_applied:
            return _applied_transition_result(
                edited,
                view_mode=view_mode,
                preset=preset,
            )
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
            episode_restarted=True,
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
        edited = (
            replace(session, verbose_logging=False)
            if session.verbose_logging
            else session
        )
        return _result(
            edited,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=edited is not session,
        )
    if key in ("[", "]"):
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=("Scenario navigation moved to the read-only Replay Viewer."),
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
    target = _hit_test_rows(
        _authorized_pointer_rows(session, view_mode=view_mode),
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
        select_clicked_target(session, target)
        if command.shift_key
        else select_controlled_actor(session, target)
    )
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
    authorized_slots = {
        row.global_slot
        for row in session.evaluation_context.roster
        if row.configured_active
    }
    if command.global_slot not in authorized_slots:
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
    return _pending_edit_result(
        session,
        edited,
        view_mode=view_mode,
        preset=preset,
    )


def _dispatch_actor_pov_target_action(
    session: DebuggerSession,
    command: ActorPovTargetActionCommandV1,
    *,
    view_mode: ViewMode,
    preset: Preset,
) -> InputDispatchResult:
    """Resolve one recipient-relative POV target action on the trusted host."""
    if view_mode != "pov":
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice="Actor-relative target selection is available only in POV.",
        )
    if command.target_action == 0:
        return _pending_edit_result(
            session,
            clear_pending_target(session),
            view_mode=view_mode,
            preset=preset,
        )

    mapping = session.evaluation_context.static_mechanics_catalog
    recipients = mapping.global_recipient_slot_by_actor_and_target_action[
        session.controlled_global_slot
    ]
    target_global_slot = recipients[command.target_action]
    authorized_slots = {
        row.global_slot
        for row in session.evaluation_context.roster
        if row.configured_active
    }
    if target_global_slot not in authorized_slots:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=(
                f"Target action {command.target_action} is unavailable in the "
                "current authorized POV."
            ),
        )
    return _pending_edit_result(
        session,
        select_clicked_target(session, target_global_slot),
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
    scripted_inspection = get_scenario(session.scenario_name).mode == "scripted"
    if scripted_inspection and not _scripted_inspection_command_is_allowed(command):
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=_SCRIPTED_INSPECTION_NOTICE,
        )
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
    if isinstance(command, ActorPovTargetActionCommandV1):
        return _dispatch_actor_pov_target_action(
            session,
            command,
            view_mode=view_mode,
            preset=preset,
        )
    if isinstance(command, ScenarioSwitchCommandV1):
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=("Scenario switching moved to the read-only Replay Viewer."),
        )
    if isinstance(command, ResetCommandV1):
        edited = reset_session(session)
        return _result(
            edited,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=True,
            episode_restarted=True,
        )
    if isinstance(command, SetViewCommandV1):
        edited = session
        changed = command.view_mode != view_mode
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
    if isinstance(command, ExitCommandV1):
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice="Debugger shutdown requested.",
            shutdown_requested=True,
        )
    return _result(
        session,
        view_mode=view_mode,
        preset=preset,
        handled=True,
        changed=False,
        notice="Replay recording is not enabled for this debugger session.",
    )
