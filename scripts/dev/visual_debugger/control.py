"""Pure debugger state transitions and the single simulator submission boundary."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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
    age_transient_history,
    derive_transient_entries,
    extract_transition_view,
    format_concise_transition,
    format_reset,
    format_verbose_transition,
)
from scripts.dev.visual_debugger.model import (
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


def build_interactive_joint_action(
    config: EnvConfig,
    controlled_global_slot: int,
    pending_action: PendingAction,
) -> Action:
    """Build one controlled-actor request without pre-filtering it by the mask."""
    _validate_active_slot(
        config,
        controlled_global_slot,
        name="controlled_global_slot",
    )
    action = make_neutral_joint_action()
    move = action.move.at[controlled_global_slot].set(pending_action.move_action)
    target = action.select_target
    ultimate = action.use_ultimate
    if pending_action.armed_lane is not None:
        if pending_action.selected_global_target_slot is not None:
            _validate_active_slot(
                config,
                pending_action.selected_global_target_slot,
                name="pending target",
            )
        target_action = global_slot_to_target_action(
            controlled_global_slot,
            pending_action.selected_global_target_slot,
        )
        target = target.at[controlled_global_slot].set(target_action)
        ultimate = ultimate.at[controlled_global_slot].set(pending_action.armed_lane)
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
) -> tuple[EnvConfig, Array, EnvState, Observation, ActionMask, Info]:
    config = scenario.build_config()
    master_key = jax.random.key(seed)
    next_key, reset_key = jax.random.split(master_key)
    state, observation, action_mask, info = reset(config, reset_key)
    return config, next_key, state, observation, action_mask, info


def create_session(
    scenario: DebuggerScenario,
    *,
    seed: int,
    controlled_global_slot: int | None,
    show_ranges: bool,
    verbose_logging: bool,
) -> DebuggerSession:
    """Create one deterministic immutable debugger session."""
    config, next_key, state, observation, action_mask, info = _fresh_snapshot(
        scenario,
        seed,
    )
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
        pending_action=PendingAction(),
        next_script_frame_index=0,
        last_transition=None,
        transient_history=(),
        next_transient_sequence_number=0,
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
    return replace(session, pending_action=pending)


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
    return replace(session, pending_action=pending)


def arm_basic(session: DebuggerSession) -> DebuggerSession:
    """Explicitly arm lane zero even when the current pair is unavailable."""
    return replace(
        session,
        pending_action=replace(
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
    return replace(
        session,
        pending_action=replace(
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


def set_pending_movement(
    session: DebuggerSession,
    move_action: int,
) -> DebuggerSession:
    """Set one in-domain pending movement category without inspecting legality."""
    if not 0 <= move_action < NUM_MOVE_ACTIONS:
        msg = f"move_action must be in [0, {NUM_MOVE_ACTIONS}); got {move_action}."
        raise ValueError(msg)
    return replace(
        session,
        pending_action=replace(session.pending_action, move_action=move_action),
    )


def cycle_controlled_actor(
    session: DebuggerSession,
    direction: int,
) -> DebuggerSession:
    """Cycle active fixed slots through direct controlled-actor selection."""
    if direction not in (-1, 1):
        msg = f"direction must be -1 or 1; got {direction}."
        raise ValueError(msg)
    active_slots = tuple(
        int(slot)
        for slot in np.flatnonzero(
            np.asarray(session.config.agent_profile.active_mask, dtype=bool)
        )
    )
    current_index = active_slots.index(session.controlled_global_slot)
    controlled_slot = active_slots[(current_index + direction) % len(active_slots)]
    return select_controlled_actor(session, controlled_slot)


def select_controlled_actor(
    session: DebuggerSession,
    global_slot: int,
) -> DebuggerSession:
    """Select an active actor without advancing the simulator decision epoch."""
    _validate_active_slot(session.config, global_slot, name="controlled actor")
    target_slot = session.pending_action.selected_global_target_slot
    target_action = global_slot_to_target_action(global_slot, target_slot)
    availability = lane_availability(
        session.action_mask,
        global_slot,
        target_action,
        0,
    )
    pending = PendingAction(
        move_action=MOVE_STAY,
        selected_global_target_slot=target_slot,
        armed_lane=0 if availability.lane_0_available else None,
        arm_origin="automatic" if availability.lane_0_available else None,
    )
    return replace(
        session,
        controlled_global_slot=global_slot,
        pending_action=pending,
    )


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
) -> PendingAction:
    target_slot = session.pending_action.selected_global_target_slot
    target_action = global_slot_to_target_action(
        session.controlled_global_slot,
        target_slot,
    )
    availability = lane_availability(
        action_mask,
        session.controlled_global_slot,
        target_action,
        0,
    )
    return PendingAction(
        move_action=MOVE_STAY,
        selected_global_target_slot=target_slot,
        armed_lane=0 if availability.lane_0_available else None,
        arm_origin="automatic" if availability.lane_0_available else None,
    )


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
    aged_history = age_transient_history(session.transient_history)
    new_entries = derive_transient_entries(
        transition,
        first_sequence_number=session.next_transient_sequence_number,
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
        pending_action=_post_submit_pending(session, next_action_mask),
        last_transition=transition,
        transient_history=(*aged_history, *new_entries),
        next_transient_sequence_number=(
            session.next_transient_sequence_number + len(new_entries)
        ),
    )
    print(
        format_verbose_transition(transition)
        if session.verbose_logging
        else format_concise_transition(transition)
    )
    return next_session


def submit_interactive(session: DebuggerSession) -> DebuggerSession:
    """Submit the one-controlled-actor pending request."""
    action = build_interactive_joint_action(
        session.config,
        session.controlled_global_slot,
        session.pending_action,
    )
    return submit_joint_action(
        session,
        action,
        submission_kind="interactive",
        report_actor_slots=(session.controlled_global_slot,),
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
    """Recreate the deterministic initial key sequence and clear live history."""
    scenario = get_scenario(session.scenario_name)
    config, next_key, state, observation, action_mask, info = _fresh_snapshot(
        scenario,
        session.seed,
    )
    controlled_slot = session.controlled_global_slot
    if not bool(config.agent_profile.active_mask[controlled_slot]):
        controlled_slot = scenario.default_controlled_slot
    reset_result = replace(
        session,
        scenario_name=scenario.name,
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
        pending_action=PendingAction(),
        next_script_frame_index=0,
        last_transition=None,
        transient_history=(),
        next_transient_sequence_number=0,
    )
    print(format_reset(reset_result))
    return reset_result


def switch_scenario(
    session: DebuggerSession,
    scenario: DebuggerScenario,
) -> DebuggerSession:
    """Start another deterministic scenario using the original CLI seed."""
    config, next_key, state, observation, action_mask, info = _fresh_snapshot(
        scenario,
        session.seed,
    )
    switched = DebuggerSession(
        scenario_name=scenario.name,
        seed=session.seed,
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
        controlled_global_slot=scenario.default_controlled_slot,
        pending_action=PendingAction(),
        next_script_frame_index=0,
        last_transition=None,
        transient_history=(),
        next_transient_sequence_number=0,
        show_ranges=session.show_ranges,
        verbose_logging=session.verbose_logging,
    )
    print(format_reset(switched))
    return switched
