"""Pure debugger state transitions and the single simulator submission boundary."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core.config import validate_env_config
from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.types import (
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MOVE_STAY,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
)
from scripts.dev.visual_debugger.diagnostics import (
    extract_transition_view,
    format_concise_transition,
    format_reset,
    format_verbose_transition,
)
from scripts.dev.visual_debugger.model import (
    MOVEMENT_SCALE_MAXIMUM,
    MOVEMENT_SCALE_MINIMUM,
    DebuggerScenario,
    DebuggerSession,
    Lane,
    LaneAvailability,
    PendingAction,
    ScenarioFrame,
    SubmissionKind,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.targeting import global_slot_to_target_action


def make_neutral_joint_action() -> Action:
    """Return the canonical fixed-shape neutral joint action."""
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _validate_active_slot(
    config: EnvConfig,
    global_slot: int,
    *,
    name: str,
) -> None:
    if not 0 <= global_slot < MAX_AGENT_SLOTS:
        msg = f"{name} must be in [0, {MAX_AGENT_SLOTS}); got {global_slot}."
        raise ValueError(msg)
    if not bool(config.agent_profile.active_mask[global_slot]):
        msg = f"{name} g{global_slot} is inactive."
        raise ValueError(msg)


def _active_global_slots(config: EnvConfig) -> tuple[int, ...]:
    """Return active fixed slots in deterministic global-slot order."""
    return tuple(
        int(slot)
        for slot in np.flatnonzero(
            np.asarray(config.agent_profile.active_mask, dtype=bool)
        )
    )


def lane_availability(
    action_mask: ActionMask,
    actor_global_slot: int,
    target_action: int,
    armed_lane: Lane | None,
) -> LaneAvailability:
    """Read exact lane availability from the authoritative joint mask."""
    if not 0 <= actor_global_slot < MAX_AGENT_SLOTS:
        msg = f"actor_global_slot must be in [0, {MAX_AGENT_SLOTS})."
        raise ValueError(msg)
    if not 0 <= target_action < NUM_TARGET_ACTIONS:
        msg = f"target_action must be in [0, {NUM_TARGET_ACTIONS})."
        raise ValueError(msg)
    if armed_lane not in (None, 0, 1):
        msg = f"armed_lane must be None, 0, or 1; got {armed_lane}."
        raise ValueError(msg)
    lane_values = action_mask.select_target_use_ultimate_joint_mask[
        actor_global_slot,
        target_action,
    ]
    lane_0_available = bool(lane_values[0])
    lane_1_available = bool(lane_values[1])
    armed_pair_legal = (
        False
        if armed_lane is None
        else lane_0_available
        if armed_lane == 0
        else lane_1_available
    )
    return LaneAvailability(
        target_action=target_action,
        lane_0_available=lane_0_available,
        lane_1_available=lane_1_available,
        armed_lane=armed_lane,
        armed_pair_legal=armed_pair_legal,
    )


def _default_pending_actions(
    config: EnvConfig,
    action_mask: ActionMask,
) -> tuple[PendingAction, ...]:
    """Build one exact fixed-slot draft tuple for a fresh decision epoch."""
    pending_actions: list[PendingAction] = []
    for actor_slot in range(MAX_AGENT_SLOTS):
        if not bool(config.agent_profile.active_mask[actor_slot]):
            pending_actions.append(PendingAction(armed_lane=None, arm_origin=None))
            continue
        basic_available = lane_availability(
            action_mask,
            actor_slot,
            0,
            0,
        ).lane_0_available
        pending_actions.append(
            PendingAction(
                armed_lane=0 if basic_available else None,
                arm_origin="automatic" if basic_available else None,
            )
        )
    return tuple(pending_actions)


def _replace_controlled_pending_action(
    session: DebuggerSession,
    pending_action: PendingAction,
) -> DebuggerSession:
    """Replace only the controlled actor's row in the immutable draft tuple."""
    pending_actions = list(session.pending_actions)
    pending_actions[session.controlled_global_slot] = pending_action
    return replace(session, pending_actions=tuple(pending_actions))


def build_interactive_joint_action(
    config: EnvConfig,
    pending_actions: tuple[PendingAction, ...],
    *,
    actor_global_slots: tuple[int, ...],
) -> Action:
    """Build one authorized joint request without pre-filtering it by the mask."""
    if len(pending_actions) != MAX_AGENT_SLOTS:
        msg = (
            f"pending_actions must contain {MAX_AGENT_SLOTS} fixed-slot rows; "
            f"got {len(pending_actions)}."
        )
        raise ValueError(msg)
    if not actor_global_slots:
        raise ValueError("actor_global_slots must contain at least one active actor.")
    if len(actor_global_slots) != len(set(actor_global_slots)):
        raise ValueError("actor_global_slots must not contain duplicates.")

    action = make_neutral_joint_action()
    move = action.move
    target = action.select_target
    ultimate = action.use_ultimate
    for actor_slot in actor_global_slots:
        _validate_active_slot(config, actor_slot, name="submission actor")
        pending_action = pending_actions[actor_slot]
        move = move.at[actor_slot].set(pending_action.move_action)
        if pending_action.armed_lane is None:
            continue
        if pending_action.selected_global_target_slot is not None:
            _validate_active_slot(
                config,
                pending_action.selected_global_target_slot,
                name="pending target",
            )
        target_action = global_slot_to_target_action(
            actor_slot,
            pending_action.selected_global_target_slot,
        )
        target = target.at[actor_slot].set(target_action)
        ultimate = ultimate.at[actor_slot].set(pending_action.armed_lane)
    return Action(move=move, select_target=target, use_ultimate=ultimate)


def build_scripted_joint_action(
    config: EnvConfig,
    frame: ScenarioFrame,
) -> Action:
    """Build a potentially multi-actor scripted request from neutral defaults."""
    action = make_neutral_joint_action()
    move = action.move
    target = action.select_target
    ultimate = action.use_ultimate
    seen_slots: set[int] = set()
    for command in frame.commands:
        if command.actor_global_slot in seen_slots:
            msg = (
                f"frame {frame.label!r} contains duplicate command for "
                f"g{command.actor_global_slot}."
            )
            raise ValueError(msg)
        seen_slots.add(command.actor_global_slot)
        _validate_active_slot(
            config,
            command.actor_global_slot,
            name="command actor",
        )
        if command.target_global_slot is not None:
            _validate_active_slot(
                config,
                command.target_global_slot,
                name="command target",
            )
        move = move.at[command.actor_global_slot].set(command.move_action)
        target = target.at[command.actor_global_slot].set(
            global_slot_to_target_action(
                command.actor_global_slot,
                command.target_global_slot,
            )
        )
        ultimate = ultimate.at[command.actor_global_slot].set(command.use_ultimate)
    return Action(move=move, select_target=target, use_ultimate=ultimate)


def _fresh_snapshot(
    scenario: DebuggerScenario,
    seed: int,
    *,
    movement_scale: float | None = None,
) -> tuple[float, EnvConfig, Array, EnvState, Observation, ActionMask, Info]:
    authored_config = scenario.build_config()
    scenario_default_movement_scale = authored_config.ordinary_movement_distance_scale
    effective_movement_scale = (
        scenario_default_movement_scale if movement_scale is None else movement_scale
    )
    config = authored_config
    if effective_movement_scale != scenario_default_movement_scale:
        config = authored_config._replace(
            ordinary_movement_distance_scale=effective_movement_scale
        )
        validate_env_config(config)
    master_key = jax.random.key(seed)
    next_key, reset_key = jax.random.split(master_key)
    state, observation, action_mask, info = reset(config, reset_key)
    return (
        scenario_default_movement_scale,
        config,
        next_key,
        state,
        observation,
        action_mask,
        info,
    )


def create_session(
    scenario: DebuggerScenario,
    *,
    seed: int,
    controlled_global_slot: int | None,
    show_ranges: bool,
    verbose_logging: bool,
) -> DebuggerSession:
    """Create one deterministic immutable debugger session."""
    (
        scenario_default_movement_scale,
        config,
        next_key,
        state,
        observation,
        action_mask,
        info,
    ) = _fresh_snapshot(scenario, seed)
    requested_slot = (
        scenario.default_controlled_slot
        if controlled_global_slot is None
        else controlled_global_slot
    )
    if not (
        0 <= requested_slot < MAX_AGENT_SLOTS
        and bool(config.agent_profile.active_mask[requested_slot])
    ):
        requested_slot = scenario.default_controlled_slot
    _validate_active_slot(config, requested_slot, name="controlled_global_slot")

    session = DebuggerSession(
        scenario_name=scenario.name,
        seed=seed,
        run_generation=0,
        scenario_default_movement_scale=scenario_default_movement_scale,
        config=config,
        key=next_key,
        state=state,
        observation=observation,
        action_mask=action_mask,
        last_reward=None,
        done_flags=DoneFlags(
            terminated=jnp.asarray(False),
            truncated=jnp.asarray(False),
        ),
        info=info,
        controlled_global_slot=requested_slot,
        pending_actions=_default_pending_actions(config, action_mask),
        next_script_frame_index=0,
        last_transition=None,
        show_ranges=show_ranges,
        verbose_logging=verbose_logging,
    )
    print(format_reset(session))
    return session


def select_clicked_target(
    session: DebuggerSession,
    target_global_slot: int,
) -> DebuggerSession:
    """Select an active target and auto-arm Basic only when exact lane zero is legal."""
    _validate_active_slot(session.config, target_global_slot, name="clicked target")
    target_action = global_slot_to_target_action(
        session.controlled_global_slot,
        target_global_slot,
    )
    availability = lane_availability(
        session.action_mask,
        session.controlled_global_slot,
        target_action,
        0,
    )
    pending = PendingAction(
        move_action=session.pending_action.move_action,
        selected_global_target_slot=target_global_slot,
        armed_lane=0 if availability.lane_0_available else None,
        arm_origin="automatic" if availability.lane_0_available else None,
    )
    return _replace_controlled_pending_action(session, pending)


def clear_pending_target(session: DebuggerSession) -> DebuggerSession:
    """Clear target selection while preserving an explicit Mage Burst arm."""
    class_id = int(
        session.config.agent_profile.class_ids[session.controlled_global_slot]
    )
    keep_mage_ultimate = (
        class_id == MAGE_CLASS_ID
        and session.pending_action.armed_lane == 1
        and session.pending_action.arm_origin == "explicit"
    )
    pending = PendingAction(
        move_action=session.pending_action.move_action,
        selected_global_target_slot=None,
        armed_lane=1 if keep_mage_ultimate else 0,
        arm_origin="explicit" if keep_mage_ultimate else "automatic",
    )
    return _replace_controlled_pending_action(session, pending)


def arm_basic(session: DebuggerSession) -> DebuggerSession:
    """Explicitly arm lane zero even when the current pair is unavailable."""
    return _replace_controlled_pending_action(
        session,
        replace(
            session.pending_action,
            armed_lane=0,
            arm_origin="explicit",
        ),
    )


def arm_ultimate(session: DebuggerSession) -> DebuggerSession:
    """Explicitly arm lane one; Mage Burst always uses target-none."""
    class_id = int(
        session.config.agent_profile.class_ids[session.controlled_global_slot]
    )
    return _replace_controlled_pending_action(
        session,
        replace(
            session.pending_action,
            selected_global_target_slot=(
                None
                if class_id == MAGE_CLASS_ID
                else session.pending_action.selected_global_target_slot
            ),
            armed_lane=1,
            arm_origin="explicit",
        ),
    )


def select_no_combat(session: DebuggerSession) -> DebuggerSession:
    """Stage no-combat intent while preserving movement and target context."""
    return _replace_controlled_pending_action(
        session,
        replace(
            session.pending_action,
            armed_lane=None,
            arm_origin=None,
        ),
    )


def set_pending_movement(
    session: DebuggerSession,
    move_action: int,
) -> DebuggerSession:
    """Set one in-domain pending movement category without inspecting legality."""
    if not 0 <= move_action < NUM_MOVE_ACTIONS:
        msg = f"move_action must be in [0, {NUM_MOVE_ACTIONS}); got {move_action}."
        raise ValueError(msg)
    return _replace_controlled_pending_action(
        session,
        replace(session.pending_action, move_action=move_action),
    )


def cycle_controlled_actor(
    session: DebuggerSession,
    direction: int,
) -> DebuggerSession:
    """Cycle active fixed slots through direct controlled-actor selection."""
    if direction not in (-1, 1):
        msg = f"direction must be -1 or 1; got {direction}."
        raise ValueError(msg)
    active_slots = _active_global_slots(session.config)
    current_index = active_slots.index(session.controlled_global_slot)
    controlled_slot = active_slots[(current_index + direction) % len(active_slots)]
    return select_controlled_actor(session, controlled_slot)


def select_controlled_actor(
    session: DebuggerSession,
    global_slot: int,
) -> DebuggerSession:
    """Select an active actor without changing any staged draft or simulator epoch."""
    _validate_active_slot(session.config, global_slot, name="controlled actor")
    return replace(session, controlled_global_slot=global_slot)


def _validate_joint_action(action: Action) -> None:
    for name, head in zip(Action._fields, action, strict=True):
        if head.shape != (MAX_AGENT_SLOTS,):
            msg = (
                f"action.{name} must have shape ({MAX_AGENT_SLOTS},); got {head.shape}."
            )
            raise ValueError(msg)
        if head.dtype != jnp.int32:
            msg = f"action.{name} must have dtype int32; got {head.dtype}."
            raise ValueError(msg)


def _terminal_reason(done_flags: DoneFlags) -> str | None:
    if bool(done_flags.terminated):
        return "terminated"
    if bool(done_flags.truncated):
        return "truncated"
    return None


def _post_submit_pending(
    session: DebuggerSession,
    action_mask: ActionMask,
) -> tuple[PendingAction, ...]:
    pending_actions: list[PendingAction] = []
    for actor_slot in range(MAX_AGENT_SLOTS):
        if not bool(session.config.agent_profile.active_mask[actor_slot]):
            pending_actions.append(PendingAction(armed_lane=None, arm_origin=None))
            continue
        target_slot = session.pending_actions[actor_slot].selected_global_target_slot
        target_action = global_slot_to_target_action(actor_slot, target_slot)
        availability = lane_availability(
            action_mask,
            actor_slot,
            target_action,
            0,
        )
        pending_actions.append(
            PendingAction(
                move_action=MOVE_STAY,
                selected_global_target_slot=target_slot,
                armed_lane=0 if availability.lane_0_available else None,
                arm_origin="automatic" if availability.lane_0_available else None,
            )
        )
    return tuple(pending_actions)


def submit_joint_action(
    session: DebuggerSession,
    submitted_action: Action,
    *,
    submission_kind: SubmissionKind,
    report_actor_slots: tuple[int, ...],
) -> DebuggerSession:
    """Split once, step once, diagnose once, and advance all paired epoch fields."""
    terminal_reason = _terminal_reason(session.done_flags)
    if terminal_reason is not None:
        print(
            f"SUBMIT BLOCKED: episode is {terminal_reason}; press R or switch scenario."
        )
        return session
    _validate_joint_action(submitted_action)
    if len(report_actor_slots) != len(set(report_actor_slots)):
        raise ValueError("report_actor_slots must not contain duplicates.")
    for actor_slot in report_actor_slots:
        _validate_active_slot(session.config, actor_slot, name="report actor")

    next_key, step_key = jax.random.split(session.key)
    (
        next_state,
        next_observation,
        reward,
        done_flags,
        next_action_mask,
        info,
    ) = step(
        session.config,
        session.state,
        session.action_mask,
        submitted_action,
        step_key,
    )
    transition = extract_transition_view(
        scenario_name=session.scenario_name,
        submission_kind=submission_kind,
        report_actor_slots=report_actor_slots,
        before_state=session.state,
        before_observation=session.observation,
        before_action_mask=session.action_mask,
        submitted_action=submitted_action,
        after_state=next_state,
        after_observation=next_observation,
        after_action_mask=next_action_mask,
        reward=reward,
        done_flags=done_flags,
        info=info,
    )
    next_session = replace(
        session,
        key=next_key,
        state=next_state,
        observation=next_observation,
        action_mask=next_action_mask,
        last_reward=reward,
        done_flags=done_flags,
        info=info,
        pending_actions=_post_submit_pending(session, next_action_mask),
        last_transition=transition,
    )
    print(
        format_verbose_transition(transition)
        if session.verbose_logging
        else format_concise_transition(transition)
    )
    return next_session


def submit_interactive(
    session: DebuggerSession,
    *,
    actor_global_slots: tuple[int, ...] | None = None,
) -> DebuggerSession:
    """Submit one authorized collection of same-epoch pending actor rows."""
    submission_slots = (
        _active_global_slots(session.config)
        if actor_global_slots is None
        else actor_global_slots
    )
    action = build_interactive_joint_action(
        session.config,
        session.pending_actions,
        actor_global_slots=submission_slots,
    )
    return submit_joint_action(
        session,
        action,
        submission_kind="interactive",
        report_actor_slots=submission_slots,
    )


def submit_next_script_frame(
    session: DebuggerSession,
) -> DebuggerSession:
    """Submit the next registered multi-actor frame through the shared boundary."""
    scenario = get_scenario(session.scenario_name)
    if session.next_script_frame_index >= len(scenario.frames):
        print("SCRIPT COMPLETE: press R to replay or [ / ] to change scenario.")
        return session
    frame = scenario.frames[session.next_script_frame_index]
    action = build_scripted_joint_action(session.config, frame)
    report_slots = tuple(
        sorted(command.actor_global_slot for command in frame.commands)
    )
    submitted = submit_joint_action(
        session,
        action,
        submission_kind="scripted",
        report_actor_slots=report_slots,
    )
    if int(submitted.state.step_count) == int(session.state.step_count):
        return submitted
    return replace(
        submitted,
        next_script_frame_index=session.next_script_frame_index + 1,
    )


def reset_session(
    session: DebuggerSession,
) -> DebuggerSession:
    """Recreate the deterministic initial epoch at the current effective scale."""
    scenario = get_scenario(session.scenario_name)
    return _restart_session(
        session,
        scenario,
        movement_scale=session.config.ordinary_movement_distance_scale,
        preserve_controlled_slot=True,
    )


def _restart_session(
    session: DebuggerSession,
    scenario: DebuggerScenario,
    *,
    movement_scale: float | None,
    preserve_controlled_slot: bool,
) -> DebuggerSession:
    """Build one coherent fresh epoch without entering the simulator step seam."""
    (
        scenario_default_movement_scale,
        config,
        next_key,
        state,
        observation,
        action_mask,
        info,
    ) = _fresh_snapshot(
        scenario,
        session.seed,
        movement_scale=movement_scale,
    )
    controlled_slot = (
        session.controlled_global_slot
        if preserve_controlled_slot
        else scenario.default_controlled_slot
    )
    if not (
        0 <= controlled_slot < MAX_AGENT_SLOTS
        and bool(config.agent_profile.active_mask[controlled_slot])
    ):
        controlled_slot = scenario.default_controlled_slot
    restarted = replace(
        session,
        scenario_name=scenario.name,
        run_generation=session.run_generation + 1,
        scenario_default_movement_scale=scenario_default_movement_scale,
        config=config,
        key=next_key,
        state=state,
        observation=observation,
        action_mask=action_mask,
        last_reward=None,
        done_flags=DoneFlags(
            terminated=jnp.asarray(False),
            truncated=jnp.asarray(False),
        ),
        info=info,
        controlled_global_slot=controlled_slot,
        pending_actions=_default_pending_actions(config, action_mask),
        next_script_frame_index=0,
        last_transition=None,
    )
    print(format_reset(restarted))
    return restarted


def set_movement_scale(
    session: DebuggerSession,
    movement_scale: float | None,
) -> DebuggerSession:
    """Reset at one exact scale, or at the active scenario's authored default."""
    if movement_scale is not None:
        if type(movement_scale) is not float:
            msg = "movement_scale must be a Python float or None."
            raise TypeError(msg)
        if not np.isfinite(movement_scale) or not (
            MOVEMENT_SCALE_MINIMUM <= movement_scale <= MOVEMENT_SCALE_MAXIMUM
        ):
            msg = (
                "movement_scale must be finite and in "
                f"[{MOVEMENT_SCALE_MINIMUM}, {MOVEMENT_SCALE_MAXIMUM}]."
            )
            raise ValueError(msg)
    effective_scale = (
        session.scenario_default_movement_scale
        if movement_scale is None
        else movement_scale
    )
    if effective_scale == session.config.ordinary_movement_distance_scale:
        return session
    return _restart_session(
        session,
        get_scenario(session.scenario_name),
        movement_scale=effective_scale,
        preserve_controlled_slot=True,
    )


def switch_scenario(
    session: DebuggerSession,
    scenario: DebuggerScenario,
) -> DebuggerSession:
    """Start another scenario at its authored movement-scale default."""
    return _restart_session(
        session,
        scenario,
        movement_scale=None,
        preserve_controlled_slot=False,
    )
