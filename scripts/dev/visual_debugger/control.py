"""Pure debugger state transitions and the single simulator submission boundary."""

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from marl_battlegrounds.core.env import initialize_scenario_state, step
from marl_battlegrounds.core.types import (
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MOVE_STAY,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    Action,
    ActionMask,
    EnvConfig,
    EnvState,
    Observation,
)
from marl_battlegrounds.evaluation.capture import (
    capture_evaluation_transition_unit_v1,
    capture_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    ActionMaskV1,
    EvaluationEpisodeContextV1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    advance_status_source_evidence_v2,
    initialize_status_source_evidence_v2,
)
from scripts.dev.visual_debugger.evaluation_bridge import (
    DebuggerActionSourceKindV1,
    DebuggerEvaluationLaunchSpecificationV1,
    build_debugger_evaluation_context_v1,
    build_debugger_evaluation_launch_specification_v1,
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


def make_neutral_joint_action() -> Action:
    """Return the canonical fixed-shape neutral joint action."""
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _validate_active_context_slot(
    context: EvaluationEpisodeContextV1,
    global_slot: int,
    *,
    name: str,
) -> None:
    if not 0 <= global_slot < MAX_AGENT_SLOTS:
        raise ValueError(f"{name} must be in [0, {MAX_AGENT_SLOTS}).")
    if not context.roster[global_slot].configured_active:
        raise ValueError(f"{name} g{global_slot} is inactive.")


def _active_context_slots(session: DebuggerSession) -> tuple[int, ...]:
    """Return active slots from the already-host canonical roster."""
    return tuple(
        row.global_slot
        for row in session.evaluation_context.roster
        if row.configured_active
    )


def _target_action_from_context(
    context: EvaluationEpisodeContextV1,
    actor_global_slot: int,
    target_global_slot: int | None,
) -> int:
    """Resolve one target category through the serialized catalog authority."""
    if target_global_slot is None:
        return 0
    catalog = context.static_mechanics_catalog
    mapping = catalog.global_recipient_slot_by_actor_and_target_action[
        actor_global_slot
    ]
    try:
        return mapping.index(target_global_slot)
    except ValueError as error:
        raise ValueError("target is absent from the serialized action axis") from error


def lane_availability(
    action_mask: ActionMask | ActionMaskV1,
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
    if type(action_mask) is ActionMaskV1:
        lane_values = action_mask.select_target_use_ultimate_joint_mask[
            actor_global_slot
        ][target_action]
    else:
        core_mask = cast(ActionMask, action_mask)
        lane_values = core_mask.select_target_use_ultimate_joint_mask[
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
    context: EvaluationEpisodeContextV1,
    action_mask: ActionMask | ActionMaskV1,
) -> tuple[PendingAction, ...]:
    """Build one exact fixed-slot draft tuple for a fresh decision epoch."""
    pending_actions: list[PendingAction] = []
    for actor_slot in range(MAX_AGENT_SLOTS):
        if not context.roster[actor_slot].configured_active:
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
    context: EvaluationEpisodeContextV1,
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
        _validate_active_context_slot(
            context,
            actor_slot,
            name="submission actor",
        )
        pending_action = pending_actions[actor_slot]
        move = move.at[actor_slot].set(pending_action.move_action)
        if pending_action.armed_lane is None:
            continue
        if pending_action.selected_global_target_slot is not None:
            _validate_active_context_slot(
                context,
                pending_action.selected_global_target_slot,
                name="pending target",
            )
        target_action = _target_action_from_context(
            context,
            actor_slot,
            pending_action.selected_global_target_slot,
        )
        target = target.at[actor_slot].set(target_action)
        ultimate = ultimate.at[actor_slot].set(pending_action.armed_lane)
    return Action(move=move, select_target=target, use_ultimate=ultimate)


def build_scripted_joint_action(
    context: EvaluationEpisodeContextV1,
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
        _validate_active_context_slot(
            context,
            command.actor_global_slot,
            name="command actor",
        )
        if command.target_global_slot is not None:
            _validate_active_context_slot(
                context,
                command.target_global_slot,
                name="command target",
            )
        move = move.at[command.actor_global_slot].set(command.move_action)
        target = target.at[command.actor_global_slot].set(
            _target_action_from_context(
                context,
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
) -> tuple[float, EnvConfig, EnvState, Observation, ActionMask]:
    """Validate and expose an authored scenario without replacing its state."""
    authored_config, authored_state = scenario.build_scenario()
    scenario_default_movement_scale = authored_config.ordinary_movement_distance_scale
    effective_movement_scale = (
        scenario_default_movement_scale if movement_scale is None else movement_scale
    )
    config = authored_config
    if effective_movement_scale != scenario_default_movement_scale:
        config = authored_config._replace(
            ordinary_movement_distance_scale=effective_movement_scale
        )
    del seed
    state, observation, action_mask, _info = initialize_scenario_state(
        authored_state,
        config,
    )
    return (
        scenario_default_movement_scale,
        config,
        state,
        observation,
        action_mask,
    )


def _debugger_action_source_kind(
    scenario: DebuggerScenario,
) -> DebuggerActionSourceKindV1:
    """Return the only action source authorized by the scenario mode."""
    return "manual" if scenario.mode == "interactive" else "scripted"


def _debugger_expected_horizon(scenario: DebuggerScenario, config: EnvConfig) -> int:
    """Use the resolved script length or the interactive config maximum."""
    return len(scenario.frames) if scenario.mode == "scripted" else config.max_steps


def create_session(
    scenario: DebuggerScenario,
    *,
    seed: int,
    evaluation_launch_specification: DebuggerEvaluationLaunchSpecificationV1,
    controlled_global_slot: int | None,
    show_ranges: bool,
    verbose_logging: bool,
) -> DebuggerSession:
    """Create one deterministic immutable debugger session."""
    if seed != evaluation_launch_specification.root_seed:
        raise ValueError("seed must equal the debugger evaluation launch root seed.")
    (
        scenario_default_movement_scale,
        config,
        state,
        observation,
        action_mask,
    ) = _fresh_snapshot(scenario, seed)
    evaluation_context = build_debugger_evaluation_context_v1(
        evaluation_launch_specification,
        scenario=scenario,
        config=config,
        run_generation=0,
        action_source_kind=_debugger_action_source_kind(scenario),
        expected_horizon=_debugger_expected_horizon(scenario, config),
    )
    next_key = jax.random.key(evaluation_context.seed_protocol.environment_seed)
    initial_frame = capture_initial_evaluation_frame_v1(
        evaluation_context,
        state,
        observation,
        action_mask,
    )
    status_source_evidence_state = initialize_status_source_evidence_v2(
        evaluation_context,
        initial_frame,
    )
    requested_slot = (
        scenario.default_controlled_slot
        if controlled_global_slot is None
        else controlled_global_slot
    )
    if not (
        0 <= requested_slot < MAX_AGENT_SLOTS
        and evaluation_context.roster[requested_slot].configured_active
    ):
        requested_slot = scenario.default_controlled_slot
    _validate_active_context_slot(
        evaluation_context,
        requested_slot,
        name="controlled_global_slot",
    )

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
        raw_continuation_identity=None,
        evaluation_context=evaluation_context,
        current_evaluation_frame=initial_frame,
        incoming_evaluation_view=None,
        status_source_evidence_state=status_source_evidence_state,
        last_submission_kind=None,
        last_report_actor_slots=(),
        controlled_global_slot=requested_slot,
        pending_actions=_default_pending_actions(
            evaluation_context,
            initial_frame.action_mask,
        ),
        next_script_frame_index=0,
        show_ranges=show_ranges,
        verbose_logging=verbose_logging,
    )
    return session


def select_clicked_target(
    session: DebuggerSession,
    target_global_slot: int,
) -> DebuggerSession:
    """Select an active target and auto-arm Basic only when exact lane zero is legal."""
    _validate_active_context_slot(
        session.evaluation_context,
        target_global_slot,
        name="clicked target",
    )
    target_action = _target_action_from_context(
        session.evaluation_context,
        session.controlled_global_slot,
        target_global_slot,
    )
    availability = lane_availability(
        session.current_evaluation_frame.action_mask,
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
    class_id = session.evaluation_context.roster[
        session.controlled_global_slot
    ].class_id
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
    class_id = session.evaluation_context.roster[
        session.controlled_global_slot
    ].class_id
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
    active_slots = _active_context_slots(session)
    current_index = active_slots.index(session.controlled_global_slot)
    controlled_slot = active_slots[(current_index + direction) % len(active_slots)]
    return select_controlled_actor(session, controlled_slot)


def select_controlled_actor(
    session: DebuggerSession,
    global_slot: int,
) -> DebuggerSession:
    """Select an active actor without changing any staged draft or simulator epoch."""
    _validate_active_context_slot(
        session.evaluation_context,
        global_slot,
        name="controlled actor",
    )
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


def _terminal_reason(session: DebuggerSession) -> str | None:
    if session.terminated:
        return "terminated"
    if session.truncated:
        return "truncated"
    if session.reached_declared_horizon:
        return "at its declared horizon"
    return None


def _post_submit_pending(
    session: DebuggerSession,
    action_mask: ActionMaskV1,
) -> tuple[PendingAction, ...]:
    pending_actions: list[PendingAction] = []
    for actor_slot in range(MAX_AGENT_SLOTS):
        if not session.evaluation_context.roster[actor_slot].configured_active:
            pending_actions.append(PendingAction(armed_lane=None, arm_origin=None))
            continue
        target_slot = session.pending_actions[actor_slot].selected_global_target_slot
        target_action = _target_action_from_context(
            session.evaluation_context,
            actor_slot,
            target_slot,
        )
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
    terminal_reason = _terminal_reason(session)
    if terminal_reason is not None:
        return session
    _validate_joint_action(submitted_action)
    if len(report_actor_slots) != len(set(report_actor_slots)):
        raise ValueError("report_actor_slots must not contain duplicates.")
    for actor_slot in report_actor_slots:
        _validate_active_context_slot(
            session.evaluation_context,
            actor_slot,
            name="report actor",
        )

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
    transition, successor_frame = capture_evaluation_transition_unit_v1(
        session.evaluation_context,
        session.current_evaluation_frame,
        next_state,
        next_observation,
        next_action_mask,
        info.transition_facts,
        reward,
        done_flags,
    )
    coherent_view = EvaluationTransitionViewV1(
        context=session.evaluation_context,
        start_frame=session.current_evaluation_frame,
        transition=transition,
        successor_frame=successor_frame,
    )
    status_source_evidence_state = advance_status_source_evidence_v2(
        session.status_source_evidence_state,
        coherent_view,
    )
    next_session = replace(
        session,
        key=next_key,
        state=next_state,
        observation=next_observation,
        action_mask=next_action_mask,
        evaluation_context=coherent_view.context,
        current_evaluation_frame=coherent_view.successor_frame,
        incoming_evaluation_view=coherent_view,
        status_source_evidence_state=status_source_evidence_state,
        last_submission_kind=submission_kind,
        last_report_actor_slots=tuple(sorted(report_actor_slots)),
        raw_continuation_identity=None,
        pending_actions=_post_submit_pending(
            session,
            coherent_view.successor_frame.action_mask,
        ),
    )
    return next_session


def submit_interactive(
    session: DebuggerSession,
    *,
    actor_global_slots: tuple[int, ...] | None = None,
) -> DebuggerSession:
    """Submit one authorized collection of same-epoch pending actor rows."""
    submission_slots = (
        _active_context_slots(session)
        if actor_global_slots is None
        else actor_global_slots
    )
    action = build_interactive_joint_action(
        session.evaluation_context,
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
        return session
    frame = scenario.frames[session.next_script_frame_index]
    action = build_scripted_joint_action(session.evaluation_context, frame)
    report_slots = tuple(
        sorted(command.actor_global_slot for command in frame.commands)
    )
    submitted = submit_joint_action(
        session,
        action,
        submission_kind="scripted",
        report_actor_slots=report_slots,
    )
    if (
        submitted.current_evaluation_frame.frame_index
        == session.current_evaluation_frame.frame_index
    ):
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
        movement_scale=(
            session.evaluation_context.resolved_env_config.ordinary_movement_distance_scale
        ),
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
        state,
        observation,
        action_mask,
    ) = _fresh_snapshot(
        scenario,
        session.seed,
        movement_scale=movement_scale,
    )
    run_generation = session.run_generation + 1
    launch_specification = build_debugger_evaluation_launch_specification_v1(
        root_seed=session.evaluation_context.seed_protocol.root_seed,
        code_revision=session.evaluation_context.code_revision,
    )
    evaluation_context = build_debugger_evaluation_context_v1(
        launch_specification,
        scenario=scenario,
        config=config,
        run_generation=run_generation,
        action_source_kind=_debugger_action_source_kind(scenario),
        expected_horizon=_debugger_expected_horizon(scenario, config),
    )
    next_key = jax.random.key(evaluation_context.seed_protocol.environment_seed)
    initial_frame = capture_initial_evaluation_frame_v1(
        evaluation_context,
        state,
        observation,
        action_mask,
    )
    status_source_evidence_state = initialize_status_source_evidence_v2(
        evaluation_context,
        initial_frame,
    )
    controlled_slot = (
        session.controlled_global_slot
        if preserve_controlled_slot
        else scenario.default_controlled_slot
    )
    if not (
        0 <= controlled_slot < MAX_AGENT_SLOTS
        and evaluation_context.roster[controlled_slot].configured_active
    ):
        controlled_slot = scenario.default_controlled_slot
    _validate_active_context_slot(
        evaluation_context,
        controlled_slot,
        name="controlled_global_slot",
    )
    restarted = replace(
        session,
        scenario_name=scenario.name,
        run_generation=run_generation,
        scenario_default_movement_scale=scenario_default_movement_scale,
        config=config,
        key=next_key,
        state=state,
        observation=observation,
        action_mask=action_mask,
        raw_continuation_identity=None,
        evaluation_context=evaluation_context,
        current_evaluation_frame=initial_frame,
        incoming_evaluation_view=None,
        status_source_evidence_state=status_source_evidence_state,
        last_submission_kind=None,
        last_report_actor_slots=(),
        controlled_global_slot=controlled_slot,
        pending_actions=_default_pending_actions(
            evaluation_context,
            initial_frame.action_mask,
        ),
        next_script_frame_index=0,
    )
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
    recorded_scale = (
        session.evaluation_context.resolved_env_config
    ).ordinary_movement_distance_scale
    if effective_scale == recorded_scale:
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
