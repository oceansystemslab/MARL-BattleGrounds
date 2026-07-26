"""Allowlisted adapter from one debugger epoch to one browser frame."""

import numpy as np

from marl_battlegrounds.core.types import (
    MAX_AGENTS_PER_TEAM,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
)
from marl_battlegrounds.rendering.scene import (
    AgentSceneV1,
    BattlefieldSceneV1,
    SceneAudience,
    TargetDisclosure,
    VisualEventBatchV1,
)
from scripts.dev.visual_debugger.diagnostics import format_ability_name
from scripts.dev.visual_debugger.model import ActorTransition, DebuggerSession
from scripts.dev.visual_debugger.protocol import (
    ActionTupleCardV1,
    ActorActionResultV1,
    CandidateLegalityCardV1,
    DebuggerFrameV1,
    DiagnosticFactV1,
    HudFrameV1,
    LatestTransitionCardV1,
    MovementLegalityCardV1,
    PendingActionCardV1,
    PendingSubmissionScope,
    Preset,
    ScenarioMetadataV1,
    ScenarioOptionV1,
    TargetReferenceV1,
    TerminalStateV1,
    ViewMode,
)
from scripts.dev.visual_debugger.scenarios import (
    get_scenario,
    list_scenarios,
)
from scripts.dev.visual_debugger.scene_adapter import (
    build_battlefield_scene,
    build_visual_event_batch,
)
from scripts.dev.visual_debugger.targeting import (
    global_slot_to_target_action,
    target_action_to_global_slot,
)

_MOVE_NAMES = (
    "Stay",
    "North",
    "South",
    "East",
    "West",
    "Northeast",
    "Northwest",
    "Southeast",
    "Southwest",
)


def _move_name(move_action: int) -> str:
    return (
        _MOVE_NAMES[move_action]
        if 0 <= move_action < len(_MOVE_NAMES)
        else "Invalid move"
    )


def _scenario_option(name: str) -> ScenarioOptionV1:
    scenario = get_scenario(name)
    return ScenarioOptionV1(
        name=scenario.name,
        title=scenario.title,
        description=scenario.description,
        mode=scenario.mode,
        audience=scenario.audience,
    )


def _scenario_metadata(session: DebuggerSession) -> ScenarioMetadataV1:
    scenario = get_scenario(session.scenario_name)
    effective_movement_scale = session.config.ordinary_movement_distance_scale
    movement_scale_overridden = (
        effective_movement_scale != session.scenario_default_movement_scale
    )
    if scenario.mode == "interactive":
        return ScenarioMetadataV1(
            **_scenario_option(scenario.name).model_dump(),
            ordinary_movement_distance_scale=effective_movement_scale,
            scenario_default_movement_scale=session.scenario_default_movement_scale,
            movement_scale_overridden=movement_scale_overridden,
            completed_frame_count=0,
            frame_count=0,
            next_frame_index=None,
            next_frame_label=None,
            next_frame_description=None,
            script_complete=False,
        )

    completed = min(session.next_script_frame_index, len(scenario.frames))
    next_frame = (
        scenario.frames[completed] if completed < len(scenario.frames) else None
    )
    return ScenarioMetadataV1(
        **_scenario_option(scenario.name).model_dump(),
        ordinary_movement_distance_scale=effective_movement_scale,
        scenario_default_movement_scale=session.scenario_default_movement_scale,
        movement_scale_overridden=movement_scale_overridden,
        completed_frame_count=completed,
        frame_count=len(scenario.frames),
        next_frame_index=None if next_frame is None else completed,
        next_frame_label=None if next_frame is None else next_frame.label,
        next_frame_description=None if next_frame is None else next_frame.description,
        script_complete=next_frame is None,
    )


def _terminal_state(session: DebuggerSession) -> TerminalStateV1:
    terminated = bool(session.done_flags.terminated)
    truncated = bool(session.done_flags.truncated)
    reason = "terminated" if terminated else "truncated" if truncated else None
    return TerminalStateV1(
        is_terminal=reason is not None,
        terminated=terminated,
        truncated=truncated,
        reason=reason,
    )


def _scene_agents(scene: BattlefieldSceneV1) -> dict[int, AgentSceneV1]:
    return {agent.global_slot: agent for agent in scene.agents}


def _target_reference(
    *,
    actor_global_slot: int,
    target_action: int,
    scene: BattlefieldSceneV1,
    view_mode: ViewMode,
) -> tuple[int | None, TargetReferenceV1]:
    if not 0 <= target_action < NUM_TARGET_ACTIONS:
        return target_action, TargetReferenceV1(
            disclosure="invalid",
            global_slot=None,
        )
    target_global_slot = target_action_to_global_slot(
        actor_global_slot,
        target_action,
    )
    if target_global_slot is None:
        return 0, TargetReferenceV1(
            disclosure="target_none",
            global_slot=None,
        )
    authorized_slots = {agent.global_slot for agent in scene.agents}
    if target_global_slot in authorized_slots:
        return target_action, TargetReferenceV1(
            disclosure="public",
            global_slot=target_global_slot,
        )
    disclosure: TargetDisclosure = "redacted" if view_mode == "pov" else "invalid"
    return (
        None if disclosure == "redacted" else target_action,
        TargetReferenceV1(
            disclosure=disclosure,
            global_slot=None,
        ),
    )


def _action_summary(
    *,
    actor_class_id: int,
    move_action: int,
    target: TargetReferenceV1,
    use_ultimate_action: int,
) -> str:
    move = _move_name(move_action)
    if target.disclosure == "target_none" and use_ultimate_action == 0:
        return f"{move} + NO COMBAT"
    ability = format_ability_name(actor_class_id, use_ultimate_action)
    target_label = (
        f"id_{target.global_slot}"
        if target.global_slot is not None
        else "self"
        if target.disclosure == "target_none" and use_ultimate_action == 1
        else "redacted target"
        if target.disclosure == "redacted"
        else "invalid target"
    )
    return f"{move} + {ability} → {target_label}"


def _action_card(
    *,
    actor: AgentSceneV1,
    move_action: int,
    target_action: int,
    use_ultimate_action: int,
    scene: BattlefieldSceneV1,
    view_mode: ViewMode,
) -> ActionTupleCardV1:
    disclosed_target_action, target = _target_reference(
        actor_global_slot=actor.global_slot,
        target_action=target_action,
        scene=scene,
        view_mode=view_mode,
    )
    return ActionTupleCardV1(
        move_action=move_action,
        target_action=disclosed_target_action,
        use_ultimate_action=use_ultimate_action,
        target=target,
        summary=_action_summary(
            actor_class_id=actor.class_id,
            move_action=move_action,
            target=target,
            use_ultimate_action=use_ultimate_action,
        ),
    )


def _pending_action_card(
    session: DebuggerSession,
    scene: BattlefieldSceneV1,
    *,
    actor_global_slot: int,
    view_mode: ViewMode,
    inspection_only: bool,
) -> PendingActionCardV1:
    actor_slot = actor_global_slot
    actor = _scene_agents(scene)[actor_slot]
    pending_action = session.pending_actions[actor_slot]
    raw_target = pending_action.selected_global_target_slot
    authorized_slots = {agent.global_slot for agent in scene.agents}
    if raw_target is None:
        target_action: int | None = 0
        target = TargetReferenceV1(
            disclosure="target_none",
            global_slot=None,
        )
    elif raw_target in authorized_slots:
        target_action = global_slot_to_target_action(actor_slot, raw_target)
        target = TargetReferenceV1(
            disclosure="public",
            global_slot=raw_target,
        )
    else:
        target_action = None
        target = TargetReferenceV1(
            disclosure="redacted",
            global_slot=None,
        )

    move_action = pending_action.move_action
    movement_mask_value = bool(session.action_mask.move_mask[actor_slot, move_action])
    armed_lane = pending_action.armed_lane
    pair_mask_value = (
        None
        if armed_lane is None or target_action is None
        else bool(
            session.action_mask.select_target_use_ultimate_joint_mask[
                actor_slot,
                target_action,
                armed_lane,
            ]
        )
    )
    if armed_lane is None or (armed_lane == 0 and raw_target is None):
        summary = f"{_move_name(move_action)} + NO COMBAT"
    else:
        summary = _action_summary(
            actor_class_id=actor.class_id,
            move_action=move_action,
            target=target,
            use_ultimate_action=armed_lane,
        )
    return PendingActionCardV1(
        label=(
            "PLAYBACK / INSPECTION ONLY" if inspection_only else "PENDING / WILL SUBMIT"
        ),
        actor_global_slot=actor_slot,
        move_action=move_action,
        target_action=target_action,
        armed_lane=armed_lane,
        arm_origin=pending_action.arm_origin,
        target=target,
        movement_mask_value=movement_mask_value,
        pair_mask_value=pair_mask_value,
        summary=summary,
    )


def _pending_action_cards(
    session: DebuggerSession,
    scene: BattlefieldSceneV1,
    *,
    view_mode: ViewMode,
) -> tuple[PendingSubmissionScope, tuple[PendingActionCardV1, ...]]:
    scenario = get_scenario(session.scenario_name)
    if scenario.mode == "scripted":
        scope: PendingSubmissionScope = "scripted_playback"
        actor_slots = (session.controlled_global_slot,)
    elif view_mode == "pov":
        scope = "controlled_actor"
        actor_slots = (session.controlled_global_slot,)
    else:
        scope = "joint_turn"
        actor_slots = tuple(agent.global_slot for agent in scene.agents)
    return scope, tuple(
        _pending_action_card(
            session,
            scene,
            actor_global_slot=actor_slot,
            view_mode=view_mode,
            inspection_only=scope == "scripted_playback",
        )
        for actor_slot in actor_slots
    )


def _actor_transition(
    session: DebuggerSession,
    actor_global_slot: int,
) -> ActorTransition:
    transition = session.last_transition
    if transition is None:
        raise AssertionError("latest action card requires a transition")
    return next(
        actor
        for actor in transition.actor_transitions
        if actor.actor_global_slot == actor_global_slot
    )


def _previous_action_row(
    session: DebuggerSession,
    scene: BattlefieldSceneV1,
    actor_global_slot: int,
) -> tuple[int, int, int] | None:
    """Decode only observer-authorized successor previous-action one-hots."""
    agents = _scene_agents(scene)
    observer_slot = session.controlled_global_slot
    observer = agents[observer_slot]
    actor = agents.get(actor_global_slot)
    if actor is None:
        return None

    same_team = observer.team_id == actor.team_id
    if same_team:
        row = (
            actor_global_slot
            if observer_slot < MAX_AGENTS_PER_TEAM
            else actor_global_slot - MAX_AGENTS_PER_TEAM
        )
    else:
        row = (
            actor_global_slot - MAX_AGENTS_PER_TEAM
            if observer_slot < MAX_AGENTS_PER_TEAM
            else actor_global_slot
        )

    previous = session.observation.previous_timestep_actions
    if same_team:
        heads = (
            previous.ally_previous_timestep_move_actions_one_hot[
                observer_slot,
                row,
            ],
            previous.ally_previous_timestep_select_target_actions_one_hot[
                observer_slot,
                row,
            ],
            previous.ally_previous_timestep_use_ultimate_actions_one_hot[
                observer_slot,
                row,
            ],
        )
    else:
        heads = (
            previous.enemy_previous_timestep_move_actions_one_hot[
                observer_slot,
                row,
            ],
            previous.enemy_previous_timestep_select_target_actions_one_hot[
                observer_slot,
                row,
            ],
            previous.enemy_previous_timestep_use_ultimate_actions_one_hot[
                observer_slot,
                row,
            ],
        )
    values = tuple(np.asarray(head) for head in heads)
    if not all(bool(value.sum() == 1) for value in values):
        return None
    decoded = tuple(int(np.argmax(value)) for value in values)
    move_action, target_action, use_ultimate_action = decoded
    if not (
        0 <= move_action < NUM_MOVE_ACTIONS
        and 0 <= target_action < NUM_TARGET_ACTIONS
        and 0 <= use_ultimate_action < NUM_ULTIMATE_ACTIONS
    ):
        raise AssertionError("authorized previous action decoded outside its domain")
    return move_action, target_action, use_ultimate_action


def _actor_action_result(
    session: DebuggerSession,
    scene: BattlefieldSceneV1,
    *,
    actor_slot: int,
    view_mode: ViewMode,
) -> ActorActionResultV1 | None:
    actor = _scene_agents(scene)[actor_slot]
    actor_transition = _actor_transition(session, actor_slot)
    submitted_values = (
        actor_transition.submitted_move_action,
        actor_transition.submitted_target_action,
        actor_transition.submitted_use_ultimate,
    )
    if view_mode == "researcher":
        accepted_values = (
            actor_transition.accepted_move_action,
            actor_transition.accepted_target_action,
            actor_transition.accepted_use_ultimate,
        )
        movement_accepted = actor_transition.movement_accepted
        combat_accepted = actor_transition.combat_pair_accepted
    else:
        accepted_values = _previous_action_row(session, scene, actor_slot)
        if accepted_values is None:
            return None
        movement_accepted = (
            actor_transition.submitted_tuple_in_domain
            and actor_transition.submitted_move_mask_value
            and submitted_values[0] == accepted_values[0]
        )
        combat_accepted = (
            actor_transition.submitted_tuple_in_domain
            and actor_transition.submitted_pair_mask_value
            and submitted_values[1:] == accepted_values[1:]
        )

    submitted = _action_card(
        actor=actor,
        move_action=submitted_values[0],
        target_action=submitted_values[1],
        use_ultimate_action=submitted_values[2],
        scene=scene,
        view_mode=view_mode,
    )
    accepted = _action_card(
        actor=actor,
        move_action=accepted_values[0],
        target_action=accepted_values[1],
        use_ultimate_action=accepted_values[2],
        scene=scene,
        view_mode=view_mode,
    )
    combat_is_disclosed = not (
        view_mode == "pov"
        and (
            submitted.target.disclosure == "redacted"
            or accepted.target.disclosure == "redacted"
        )
    )
    combat_result = (
        "undisclosed"
        if not combat_is_disclosed
        else "canonical_noop"
        if submitted_values[1] == 0 and submitted_values[2] == 0 and combat_accepted
        else "accepted"
        if combat_accepted
        else "rejected"
    )
    return ActorActionResultV1(
        actor_global_slot=actor_slot,
        submitted=submitted,
        accepted=accepted,
        movement_mask_value=actor_transition.submitted_move_mask_value,
        pair_mask_value=(
            actor_transition.submitted_pair_mask_value if combat_is_disclosed else None
        ),
        movement_accepted=movement_accepted,
        combat_result=combat_result,
    )


def _latest_transition_card(
    session: DebuggerSession,
    scene: BattlefieldSceneV1,
    *,
    view_mode: ViewMode,
) -> LatestTransitionCardV1 | None:
    transition = session.last_transition
    if transition is None:
        return None

    report_slots = transition.report_actor_slots
    actor_slots = (
        tuple(
            actor_slot
            for actor_slot in report_slots
            if actor_slot == session.controlled_global_slot
        )
        if view_mode == "pov"
        else report_slots
    )
    scene_slots = set(_scene_agents(scene))
    actor_results: list[ActorActionResultV1] = []
    for actor_slot in actor_slots:
        if actor_slot not in scene_slots:
            continue
        result = _actor_action_result(
            session,
            scene,
            actor_slot=actor_slot,
            view_mode=view_mode,
        )
        if result is not None:
            actor_results.append(result)
    actors = tuple(actor_results)
    if not actors:
        return None
    return LatestTransitionCardV1(
        transition_id=int(transition.after_state.step_count),
        submission_kind=transition.submission_kind,
        actors=actors,
    )


def _diagnostics(
    session: DebuggerSession,
    scene: BattlefieldSceneV1,
    *,
    revision: int,
) -> tuple[DiagnosticFactV1, ...]:
    facts = [
        DiagnosticFactV1(
            fact_id="simulator_step",
            label="Simulator step",
            value=str(int(session.state.step_count)),
            technical=True,
        ),
        DiagnosticFactV1(
            fact_id="run_generation",
            label="Run generation",
            value=str(session.run_generation),
            technical=True,
        ),
        DiagnosticFactV1(
            fact_id="revision",
            label="Frame revision",
            value=str(revision),
            technical=True,
        ),
        DiagnosticFactV1(
            fact_id="logging_mode",
            label="Terminal logging",
            value="verbose" if session.verbose_logging else "concise",
            technical=True,
        ),
    ]
    legality = scene.selected_legality
    if legality is not None:
        facts.append(
            DiagnosticFactV1(
                fact_id="selected_pair",
                label="Selected exact lanes",
                value=(
                    f"Basic={int(legality.lane_0_available)} "
                    f"Ultimate={int(legality.lane_1_available)} "
                    f"Armed={int(legality.armed_pair_legal)}"
                ),
                technical=True,
            )
        )
    return tuple(facts)


def _candidate_legalities(
    session: DebuggerSession,
    scene: BattlefieldSceneV1,
) -> tuple[CandidateLegalityCardV1, ...]:
    """Copy exact current-mask lanes for target-none and authorized scene slots."""
    controlled = session.controlled_global_slot
    exact_mask = session.action_mask.select_target_use_ultimate_joint_mask
    rows = [
        CandidateLegalityCardV1(
            target_action=0,
            target=TargetReferenceV1(
                disclosure="target_none",
                global_slot=None,
            ),
            lane_0_available=bool(exact_mask[controlled, 0, 0]),
            lane_1_available=bool(exact_mask[controlled, 0, 1]),
            basic_available=False,
            ultimate_available=bool(exact_mask[controlled, 0, 1]),
        )
    ]
    for agent in scene.agents:
        target_action = global_slot_to_target_action(
            controlled,
            agent.global_slot,
        )
        rows.append(
            CandidateLegalityCardV1(
                target_action=target_action,
                target=TargetReferenceV1(
                    disclosure="public",
                    global_slot=agent.global_slot,
                ),
                lane_0_available=bool(exact_mask[controlled, target_action, 0]),
                lane_1_available=bool(exact_mask[controlled, target_action, 1]),
                basic_available=bool(exact_mask[controlled, target_action, 0]),
                ultimate_available=bool(exact_mask[controlled, target_action, 1]),
            )
        )
    return tuple(rows)


def _movement_legalities(
    session: DebuggerSession,
) -> tuple[MovementLegalityCardV1, ...]:
    """Copy the controlled actor's exact current movement-mask row."""
    controlled = session.controlled_global_slot
    exact_mask = session.action_mask.move_mask
    return tuple(
        MovementLegalityCardV1(
            move_action=move_action,
            available=bool(exact_mask[controlled, move_action]),
        )
        for move_action in range(NUM_MOVE_ACTIONS)
    )


def _hud_frame(
    session: DebuggerSession,
    scene: BattlefieldSceneV1,
    *,
    view_mode: ViewMode,
    revision: int,
) -> HudFrameV1:
    selection = scene.selection
    if selection is None:
        raise AssertionError("live debugger scenes require a selection record")
    pending_submission_scope, pending_actions = _pending_action_cards(
        session,
        scene,
        view_mode=view_mode,
    )
    pending_action = next(
        pending
        for pending in pending_actions
        if pending.actor_global_slot == selection.controlled_global_slot
    )
    return HudFrameV1(
        roster_global_slots=tuple(agent.global_slot for agent in scene.agents),
        controlled_global_slot=selection.controlled_global_slot,
        selected_global_slot=selection.selected_global_slot,
        pending_submission_scope=pending_submission_scope,
        pending_actions=pending_actions,
        pending_action=pending_action,
        latest_transition=_latest_transition_card(
            session,
            scene,
            view_mode=view_mode,
        ),
        movement_legalities=_movement_legalities(session),
        candidate_legalities=_candidate_legalities(session, scene),
        diagnostics=_diagnostics(session, scene, revision=revision),
    )


def build_debugger_frame(
    session: DebuggerSession,
    *,
    session_id: str,
    revision: int,
    view_mode: ViewMode,
    preset: Preset,
    include_stress: bool,
) -> DebuggerFrameV1:
    """Build one validated frame without serializing simulator-owned objects."""
    audience: SceneAudience = "researcher" if view_mode == "researcher" else "agent_pov"
    scene = build_battlefield_scene(session, audience=audience)
    event_batch: VisualEventBatchV1 | None = build_visual_event_batch(
        session,
        audience=audience,
    )
    transition_id = None if event_batch is None else event_batch.transition_id
    return DebuggerFrameV1(
        session_id=session_id,
        run_generation=session.run_generation,
        revision=revision,
        simulator_step=int(session.state.step_count),
        transition_id=transition_id,
        view_mode=view_mode,
        preset=preset,
        scenario=_scenario_metadata(session),
        available_scenarios=tuple(
            _scenario_option(scenario.name)
            for scenario in list_scenarios(include_stress=include_stress)
        ),
        terminal=_terminal_state(session),
        scene=scene,
        event_batch=event_batch,
        hud=_hud_frame(
            session,
            scene,
            view_mode=view_mode,
            revision=revision,
        ),
    )
