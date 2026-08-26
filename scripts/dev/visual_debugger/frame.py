"""Canonical CP2/CP3 projections for live debugger browser frames."""

from typing import Literal

from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV1
from marl_battlegrounds.evaluation.pov import (
    ActorPovAxisMappingV1,
    ActorPovCurrentSliceV1,
    build_actor_pov_current_slice_v1,
)
from marl_battlegrounds.evaluation.wire_shapes import (
    NUM_MOVE_ACTIONS_V1,
    NUM_TARGET_ACTIONS_V1,
    NUM_ULTIMATE_ACTIONS_V1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    EvaluationScenePresentationStateV1,
    build_researcher_analyzer_projection_v2,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovAnalyzerProjectionV1,
    build_actor_pov_analyzer_projection_v1,
)
from marl_battlegrounds.rendering.scene import (
    BattlefieldSceneV2,
    ResearcherAnalyzerProjectionV2,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.protocol import (
    ActionTupleCardV1,
    ActorActionResultV1,
    ActorPovActionResultV1,
    ActorPovActionTupleCardV1,
    ActorPovCandidateLegalityCardV1,
    ActorPovHudFrameV1,
    ActorPovLatestTransitionCardV1,
    ActorPovLiveDebuggerFrameV2,
    ActorPovPendingActionCardV1,
    ActorPovTargetReferenceV1,
    CandidateLegalityCardV1,
    DiagnosticFactV1,
    LatestTransitionCardV2,
    MovementLegalityCardV1,
    PendingActionCardV1,
    PendingSubmissionScope,
    Preset,
    RecordingStatusV1,
    ResearcherHudFrameV2,
    ResearcherLiveDebuggerFrameV2,
    ScenarioMetadataV1,
    ScenarioOptionV1,
    TargetReferenceV1,
    TerminalStateV2,
    ViewMode,
)
from scripts.dev.visual_debugger.scenarios import get_scenario, list_scenarios

type LiveDebuggerFrame = ResearcherLiveDebuggerFrameV2 | ActorPovLiveDebuggerFrameV2


def _actor_pov_recording_status(
    status: RecordingStatusV1 | None,
) -> RecordingStatusV1 | None:
    """Project reducer-processing closeout into one audience-safe stable reason."""
    if status is None or status.completion_reason != "evaluation_processing_failure":
        return status
    return RecordingStatusV1.model_validate(
        {
            **status.model_dump(mode="python"),
            "completion_reason": "evaluation_unavailable",
        }
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
    effective_scale = (
        session.evaluation_context.resolved_env_config.ordinary_movement_distance_scale
    )
    if scenario.mode == "interactive":
        return ScenarioMetadataV1(
            **_scenario_option(scenario.name).model_dump(),
            ordinary_movement_distance_scale=effective_scale,
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
        ordinary_movement_distance_scale=effective_scale,
        completed_frame_count=completed,
        frame_count=len(scenario.frames),
        next_frame_index=None if next_frame is None else completed,
        next_frame_label=None if next_frame is None else next_frame.label,
        next_frame_description=None if next_frame is None else next_frame.description,
        script_complete=next_frame is None,
    )


def _terminal_state(session: DebuggerSession) -> TerminalStateV2:
    terminated = session.terminated
    truncated = session.truncated
    horizon = session.reached_declared_horizon
    reason = (
        "terminated"
        if terminated
        else "truncated"
        if truncated
        else "declared_horizon"
        if horizon
        else None
    )
    return TerminalStateV2(
        is_sealed=reason is not None,
        terminated=terminated,
        truncated=truncated,
        reached_declared_horizon=horizon,
        reason=reason,
    )


def _target_action_for_slot(
    context: EvaluationEpisodeContextV1,
    actor_global_slot: int,
    target_global_slot: int | None,
) -> int:
    if target_global_slot is None:
        return 0
    catalog = context.static_mechanics_catalog
    mapping = catalog.global_recipient_slot_by_actor_and_target_action[
        actor_global_slot
    ]
    try:
        return mapping.index(target_global_slot)
    except ValueError as error:
        raise ValueError(
            "pending target is absent from the serialized target axis"
        ) from error


def _researcher_target_reference(
    context: EvaluationEpisodeContextV1,
    *,
    actor_global_slot: int,
    target_action: int,
) -> TargetReferenceV1:
    if not 0 <= target_action < NUM_TARGET_ACTIONS_V1:
        return TargetReferenceV1(disclosure="invalid", global_slot=None)
    catalog = context.static_mechanics_catalog
    target_slot = catalog.global_recipient_slot_by_actor_and_target_action[
        actor_global_slot
    ][target_action]
    if target_slot is None:
        return TargetReferenceV1(disclosure="target_none", global_slot=None)
    if not context.roster[target_slot].configured_active:
        return TargetReferenceV1(disclosure="invalid", global_slot=None)
    return TargetReferenceV1(disclosure="public", global_slot=target_slot)


def _researcher_action_card(
    context: EvaluationEpisodeContextV1,
    *,
    actor_global_slot: int,
    move_action: int,
    target_action: int,
    use_ultimate_action: int,
) -> ActionTupleCardV1:
    target = _researcher_target_reference(
        context,
        actor_global_slot=actor_global_slot,
        target_action=target_action,
    )
    target_label = (
        f"Agent ID {context.roster[target.global_slot].public_agent_id}"
        if target.global_slot is not None
        else "No target"
        if target.disclosure == "target_none"
        else "Invalid target"
    )
    move_name = (
        context.static_mechanics_catalog.movement_action_name_by_id[move_action]
        if 0 <= move_action < NUM_MOVE_ACTIONS_V1
        else "Invalid move"
    )
    component = (
        context.static_mechanics_catalog.use_ultimate_action_name_by_id[
            use_ultimate_action
        ]
        if 0 <= use_ultimate_action < NUM_ULTIMATE_ACTIONS_V1
        else "Invalid ability"
    )
    summary = (
        f"{move_name} + NO COMBAT"
        if target_action == 0 and use_ultimate_action == 0
        else f"{move_name} + {component} → {target_label}"
    )
    return ActionTupleCardV1(
        move_action=move_action,
        target_action=(None if target.disclosure == "redacted" else target_action),
        use_ultimate_action=use_ultimate_action,
        target=target,
        summary=summary,
    )


def _researcher_pending_card(
    session: DebuggerSession,
    scene: BattlefieldSceneV2,
    *,
    actor_global_slot: int,
    inspection_only: bool,
) -> PendingActionCardV1:
    pending = session.pending_actions[actor_global_slot]
    target_action = _target_action_for_slot(
        session.evaluation_context,
        actor_global_slot,
        pending.selected_global_target_slot,
    )
    target = _researcher_target_reference(
        session.evaluation_context,
        actor_global_slot=actor_global_slot,
        target_action=target_action,
    )
    mask = session.current_evaluation_frame.action_mask
    pair_value = (
        None
        if pending.armed_lane is None
        else mask.select_target_use_ultimate_joint_mask[actor_global_slot][
            target_action
        ][pending.armed_lane]
    )
    action = _researcher_action_card(
        session.evaluation_context,
        actor_global_slot=actor_global_slot,
        move_action=pending.move_action,
        target_action=target_action,
        use_ultimate_action=0 if pending.armed_lane is None else pending.armed_lane,
    )
    scene_slots = {agent.global_slot for agent in scene.agents}
    if target.global_slot is not None and target.global_slot not in scene_slots:
        raise ValueError("researcher pending target must occur in the scene roster")
    return PendingActionCardV1(
        label=(
            "PLAYBACK / INSPECTION ONLY" if inspection_only else "PENDING / WILL SUBMIT"
        ),
        actor_global_slot=actor_global_slot,
        move_action=pending.move_action,
        target_action=target_action,
        armed_lane=pending.armed_lane,
        arm_origin=pending.arm_origin,
        target=target,
        movement_mask_value=mask.move_mask[actor_global_slot][pending.move_action],
        pair_mask_value=pair_value,
        summary=action.summary,
    )


def _researcher_latest_transition(
    session: DebuggerSession,
) -> LatestTransitionCardV2 | None:
    view = session.incoming_evaluation_view
    if view is None:
        return None
    facts = view.transition.facts.action_acceptance_facts
    mask = view.start_frame.action_mask
    actors: list[ActorActionResultV1] = []
    for actor_slot in session.last_report_actor_slots:
        submitted = facts.submitted_joint_action
        accepted = facts.accepted_joint_action
        submitted_values = (
            submitted.move[actor_slot],
            submitted.select_target[actor_slot],
            submitted.use_ultimate[actor_slot],
        )
        accepted_values = (
            accepted.move[actor_slot],
            accepted.select_target[actor_slot],
            accepted.use_ultimate[actor_slot],
        )
        out_of_domain = facts.submitted_action_tuple_is_out_of_domain_by_actor[
            actor_slot
        ]
        move_in_domain = 0 <= submitted_values[0] < NUM_MOVE_ACTIONS_V1
        pair_in_domain = (
            0 <= submitted_values[1] < NUM_TARGET_ACTIONS_V1
            and 0 <= submitted_values[2] < NUM_ULTIMATE_ACTIONS_V1
        )
        movement_rejected = facts.in_domain_move_action_is_rejected_by_actor[actor_slot]
        combat_rejected = facts.in_domain_combat_action_pair_is_rejected_by_actor[
            actor_slot
        ]
        combat_result: Literal[
            "accepted", "rejected", "canonical_noop", "undisclosed"
        ] = (
            "rejected"
            if out_of_domain or combat_rejected
            else "canonical_noop"
            if accepted_values[1:] == (0, 0)
            else "accepted"
        )
        actors.append(
            ActorActionResultV1(
                actor_global_slot=actor_slot,
                submitted=_researcher_action_card(
                    session.evaluation_context,
                    actor_global_slot=actor_slot,
                    move_action=submitted_values[0],
                    target_action=submitted_values[1],
                    use_ultimate_action=submitted_values[2],
                ),
                accepted=_researcher_action_card(
                    session.evaluation_context,
                    actor_global_slot=actor_slot,
                    move_action=accepted_values[0],
                    target_action=accepted_values[1],
                    use_ultimate_action=accepted_values[2],
                ),
                movement_mask_value=(
                    mask.move_mask[actor_slot][submitted_values[0]]
                    if move_in_domain
                    else False
                ),
                pair_mask_value=(
                    mask.select_target_use_ultimate_joint_mask[actor_slot][
                        submitted_values[1]
                    ][submitted_values[2]]
                    if pair_in_domain
                    else None
                ),
                movement_accepted=not (out_of_domain or movement_rejected),
                combat_result=combat_result,
            )
        )
    if not actors:
        return None
    submission_kind = session.last_submission_kind
    if submission_kind is None:
        raise ValueError("incoming researcher transition requires submission metadata")
    return LatestTransitionCardV2(
        transition_index=view.transition.transition_index,
        transition_id=view.transition.transition_id,
        submission_kind=submission_kind,
        actors=tuple(actors),
    )


def _movement_legalities(
    mask_row: tuple[bool, ...],
) -> tuple[MovementLegalityCardV1, ...]:
    return tuple(
        MovementLegalityCardV1(move_action=move_action, available=available)
        for move_action, available in enumerate(mask_row)
    )


def _researcher_candidates(
    session: DebuggerSession,
) -> tuple[CandidateLegalityCardV1, ...]:
    controlled = session.controlled_global_slot
    mask = session.current_evaluation_frame.action_mask
    joint = mask.select_target_use_ultimate_joint_mask[controlled]
    rows: list[CandidateLegalityCardV1] = []
    for target_action, lanes in enumerate(joint):
        target = _researcher_target_reference(
            session.evaluation_context,
            actor_global_slot=controlled,
            target_action=target_action,
        )
        if target_action > 0 and target.disclosure != "public":
            continue
        rows.append(
            CandidateLegalityCardV1(
                target_action=target_action,
                target=target,
                lane_0_available=lanes[0],
                lane_1_available=lanes[1],
                basic_available=target_action > 0 and lanes[0],
                ultimate_available=lanes[1],
            )
        )
    return tuple(rows)


def _diagnostics(
    session: DebuggerSession,
    *,
    revision: int,
    include_episode_id: bool,
) -> tuple[DiagnosticFactV1, ...]:
    frame = session.current_evaluation_frame
    facts = [
        DiagnosticFactV1(
            fact_id="simulator_step",
            label="Simulator step",
            value=str(frame.simulator_step_count),
            technical=True,
        ),
        DiagnosticFactV1(
            fact_id="frame_index",
            label="Evaluation frame",
            value=str(frame.frame_index),
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
            label="Browser revision",
            value=str(revision),
            technical=True,
        ),
    ]
    if include_episode_id:
        facts.append(
            DiagnosticFactV1(
                fact_id="episode_id",
                label="Evaluation episode",
                value=session.evaluation_context.identity.episode_id,
                technical=True,
            )
        )
    return tuple(facts)


def _build_researcher_hud(
    session: DebuggerSession,
    projection: ResearcherAnalyzerProjectionV2,
    *,
    revision: int,
) -> ResearcherHudFrameV2:
    scene = projection.scene
    selection = scene.selection
    if selection is None:
        raise ValueError("live researcher projections require a selection record")
    scenario = get_scenario(session.scenario_name)
    scope: PendingSubmissionScope = (
        "scripted_playback" if scenario.mode == "scripted" else "joint_turn"
    )
    actor_slots = (
        (session.controlled_global_slot,)
        if scope == "scripted_playback"
        else tuple(agent.global_slot for agent in scene.agents)
    )
    pending = tuple(
        _researcher_pending_card(
            session,
            scene,
            actor_global_slot=actor_slot,
            inspection_only=scope == "scripted_playback",
        )
        for actor_slot in actor_slots
    )
    controlled_pending = next(
        row
        for row in pending
        if row.actor_global_slot == session.controlled_global_slot
    )
    mask = session.current_evaluation_frame.action_mask
    return ResearcherHudFrameV2(
        roster_global_slots=tuple(agent.global_slot for agent in scene.agents),
        controlled_global_slot=session.controlled_global_slot,
        selected_global_slot=selection.selected_global_slot,
        pending_submission_scope=scope,
        pending_actions=pending,
        pending_action=controlled_pending,
        latest_transition=_researcher_latest_transition(session),
        movement_legalities=_movement_legalities(
            mask.move_mask[session.controlled_global_slot]
        ),
        candidate_legalities=_researcher_candidates(session),
        diagnostics=_diagnostics(session, revision=revision, include_episode_id=True),
    )


def _pov_target_reference(
    axis: ActorPovAxisMappingV1,
    target_action: int,
) -> ActorPovTargetReferenceV1:
    public_id = (
        axis.target_action_recipient_public_agent_id_by_id[target_action]
        if 0 <= target_action < NUM_TARGET_ACTIONS_V1
        else None
    )
    return ActorPovTargetReferenceV1(
        target_action=target_action,
        public_agent_id=public_id,
    )


def _pov_action_card(
    axis: ActorPovAxisMappingV1,
    *,
    move_action: int,
    target_action: int,
    use_ultimate_action: int,
) -> ActorPovActionTupleCardV1:
    target = _pov_target_reference(axis, target_action)
    move_name = (
        axis.movement_action_name_by_id[move_action]
        if 0 <= move_action < NUM_MOVE_ACTIONS_V1
        else "Invalid move"
    )
    component = (
        axis.use_ultimate_action_name_by_id[use_ultimate_action]
        if 0 <= use_ultimate_action < NUM_ULTIMATE_ACTIONS_V1
        else "Invalid ability"
    )
    target_label = (
        f"Agent ID {target.public_agent_id}"
        if target.public_agent_id is not None
        else "No target"
        if target_action == 0
        else "Invalid target"
    )
    summary = (
        f"{move_name} + NO COMBAT"
        if target_action == 0 and use_ultimate_action == 0
        else f"{move_name} + {component} → {target_label}"
    )
    return ActorPovActionTupleCardV1(
        move_action=move_action,
        target=target,
        use_ultimate_action=use_ultimate_action,
        summary=summary,
    )


def _pov_latest_transition(
    session: DebuggerSession,
    slice_: ActorPovCurrentSliceV1,
) -> ActorPovLatestTransitionCardV1 | None:
    incoming = slice_.incoming_transition
    if incoming is None:
        return None
    submitted = incoming.submitted_action
    accepted = incoming.accepted_action
    submission_kind = session.last_submission_kind
    if submission_kind is None:
        raise ValueError("incoming POV transition requires submission metadata")
    return ActorPovLatestTransitionCardV1(
        transition_index=incoming.transition_index,
        pov_transition_id=incoming.pov_transition_id,
        submission_kind=submission_kind,
        actor=ActorPovActionResultV1(
            actor_public_agent_id=slice_.public_agent_id,
            submitted=_pov_action_card(
                slice_.axis_mapping,
                move_action=submitted.move,
                target_action=submitted.select_target,
                use_ultimate_action=submitted.use_ultimate,
            ),
            accepted=_pov_action_card(
                slice_.axis_mapping,
                move_action=accepted.move,
                target_action=accepted.select_target,
                use_ultimate_action=accepted.use_ultimate,
            ),
            submitted_tuple_is_out_of_domain=(
                incoming.submitted_action_tuple_is_out_of_domain
            ),
            movement_rejected=incoming.in_domain_move_action_is_rejected,
            combat_pair_rejected=(incoming.in_domain_combat_action_pair_is_rejected),
            movement_accepted=not (
                incoming.submitted_action_tuple_is_out_of_domain
                or incoming.in_domain_move_action_is_rejected
            ),
            combat_result=(
                "rejected"
                if (
                    incoming.submitted_action_tuple_is_out_of_domain
                    or incoming.in_domain_combat_action_pair_is_rejected
                )
                else "canonical_noop"
                if (accepted.select_target, accepted.use_ultimate) == (0, 0)
                else "accepted"
            ),
        ),
    )


def _build_pov_hud(
    session: DebuggerSession,
    slice_: ActorPovCurrentSliceV1,
    *,
    revision: int,
) -> ActorPovHudFrameV1:
    pending = session.pending_actions[session.controlled_global_slot]
    target_action = _target_action_for_slot(
        session.evaluation_context,
        session.controlled_global_slot,
        pending.selected_global_target_slot,
    )
    mask = slice_.frame.action_mask
    target = _pov_target_reference(slice_.axis_mapping, target_action)
    pair_value = (
        None
        if pending.armed_lane is None
        else mask.select_target_use_ultimate_joint[target_action][pending.armed_lane]
    )
    action = _pov_action_card(
        slice_.axis_mapping,
        move_action=pending.move_action,
        target_action=target_action,
        use_ultimate_action=0 if pending.armed_lane is None else pending.armed_lane,
    )
    scenario = get_scenario(session.scenario_name)
    scope = "scripted_playback" if scenario.mode == "scripted" else "joint_turn"
    return ActorPovHudFrameV1(
        controlled_public_agent_id=slice_.public_agent_id,
        pending_submission_scope=scope,
        pending_action=ActorPovPendingActionCardV1(
            label=(
                "PLAYBACK / INSPECTION ONLY"
                if scope == "scripted_playback"
                else "PENDING / WILL SUBMIT"
            ),
            actor_public_agent_id=slice_.public_agent_id,
            move_action=pending.move_action,
            target=target,
            armed_lane=pending.armed_lane,
            arm_origin=pending.arm_origin,
            movement_mask_value=mask.move[pending.move_action],
            pair_mask_value=pair_value,
            summary=action.summary,
        ),
        latest_transition=_pov_latest_transition(session, slice_),
        movement_legalities=_movement_legalities(mask.move),
        candidate_legalities=tuple(
            ActorPovCandidateLegalityCardV1(
                target=_pov_target_reference(slice_.axis_mapping, target_index),
                lane_0_available=lanes[0],
                lane_1_available=lanes[1],
                basic_available=target_index > 0 and lanes[0],
                ultimate_available=lanes[1],
            )
            for target_index, lanes in enumerate(mask.select_target_use_ultimate_joint)
        ),
        diagnostics=_diagnostics(
            session,
            revision=revision,
            include_episode_id=False,
        ),
    )


def build_debugger_frame(
    session: DebuggerSession,
    *,
    session_id: str,
    revision: int,
    view_mode: ViewMode,
    preset: Preset,
    include_stress: bool,
    recording_status: RecordingStatusV1 | None = None,
) -> LiveDebuggerFrame:
    """Build one audience-exact frame from canonical evaluation records."""
    context = session.evaluation_context
    frame = session.current_evaluation_frame
    terminal = _terminal_state(session)
    if view_mode == "researcher":
        selected_target = session.pending_actions[
            session.controlled_global_slot
        ].selected_global_target_slot
        projection = build_researcher_analyzer_projection_v2(
            context,
            frame,
            transition_view=session.incoming_evaluation_view,
            presentation=EvaluationScenePresentationStateV1(
                controlled_global_slot=session.controlled_global_slot,
                selected_global_slot=(
                    session.controlled_global_slot
                    if selected_target is None
                    else selected_target
                ),
                armed_lane=session.pending_actions[
                    session.controlled_global_slot
                ].armed_lane,
                show_ranges=session.show_ranges,
            ),
            status_source_evidence_state=session.status_source_evidence_state,
        )
        incoming = session.incoming_evaluation_view
        return ResearcherLiveDebuggerFrameV2(
            session_id=session_id,
            run_generation=session.run_generation,
            revision=revision,
            episode_id=context.identity.episode_id,
            frame_index=frame.frame_index,
            frame_id=frame.frame_id,
            simulator_step_count=frame.simulator_step_count,
            incoming_transition_index=(
                None if incoming is None else incoming.transition.transition_index
            ),
            incoming_transition_id=(
                None if incoming is None else incoming.transition.transition_id
            ),
            preset=preset,
            verbose=False,
            show_ranges=session.show_ranges,
            terminal=terminal,
            recording=recording_status,
            scenario=_scenario_metadata(session),
            available_scenarios=tuple(
                _scenario_option(scenario.name)
                for scenario in list_scenarios(include_stress=include_stress)
            ),
            projection=projection,
            hud=_build_researcher_hud(session, projection, revision=revision),
        )

    slice_ = build_actor_pov_current_slice_v1(
        context,
        frame,
        global_slot=session.controlled_global_slot,
        incoming_transition_view=session.incoming_evaluation_view,
    )
    pov_projection: ActorPovAnalyzerProjectionV1 = (
        build_actor_pov_analyzer_projection_v1(slice_)
    )
    return ActorPovLiveDebuggerFrameV2(
        session_id=session_id,
        run_generation=session.run_generation,
        revision=revision,
        episode_id=context.identity.episode_id,
        frame_index=frame.frame_index,
        frame_id=frame.frame_id,
        simulator_step_count=frame.simulator_step_count,
        preset=preset,
        verbose=False,
        terminal=terminal,
        recording=_actor_pov_recording_status(recording_status),
        incoming_pov_transition_id=pov_projection.incoming_transition_id,
        projection=pov_projection,
        hud=_build_pov_hud(session, slice_, revision=revision),
    )


__all__ = ["LiveDebuggerFrame", "build_debugger_frame"]
