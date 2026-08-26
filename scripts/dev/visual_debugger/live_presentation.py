"""Live packaging seams for the authorized presentation resource.

The Combat Debugger owns two presentation audiences: the full researcher
Oracle and a selected-recipient NoSharedObs Agent POV.  The Agent battlefield
remains fog-authorized while a separate geometry-free researcher branch owns
global controls and panels.  This module packages only already-committed live
epochs.  It does not step the simulator, retain history, read replay artifacts,
or expose a live SharedObs product.
"""

from __future__ import annotations

from dataclasses import replace

from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    StaticMechanicsCatalogV1,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovAdjacentTransitionSliceV1,
    ActorPovCurrentSliceV1,
)
from marl_battlegrounds.rendering.authorized_incoming import (
    build_live_no_shared_obs_incoming_summary_v1,
)
from marl_battlegrounds.rendering.authorized_inspection import (
    AuthorizedAxisOnlyTargetActionV1,
    AuthorizedNoTargetActionV1,
    AuthorizedVisibleTargetActionV1,
    DraftArmedLaneV1,
    build_live_no_shared_obs_draft_inspection_v1,
    build_live_oracle_draft_inspection_v1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    build_no_shared_obs_authorized_scene_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AcceptedActionTupleV1,
    SubmittedActionTupleV1,
    build_agent_pov_visual_incoming_summary_v1,
    build_replay_oracle_presentation_parts_v1,
    oracle_presentation_key_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    build_visual_event_batch_v2,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovAnalyzerProjectionV1,
    build_actor_pov_analyzer_projection_v1,
)
from marl_battlegrounds.rendering.scene import (
    ResearcherAnalyzerProjectionV2,
    VisualEventBatchV2,
)
from scripts.dev.visual_debugger.presentation_protocol import (
    AgentPovActionAxisV1,
    LatestTransitionActionRowV1,
    LiveEditableDraftInspectionV1,
    LiveNoSharedObsAuthorizedPresentationFrameV1,
    LiveNoSharedObsInspectionEnvelopeV1,
    LiveNoSharedObsPresentationSourceIdentityV1,
    LiveNoSharedObsTechnicalFrameV1,
    LiveOracleAuthorizedPresentationFrameV1,
    LiveOracleInspectionEnvelopeV1,
    LiveOraclePresentationSourceIdentityV1,
    LiveOracleTechnicalFrameV1,
    LivePendingActionTupleV1,
    LivePendingJointActionRowV1,
    LivePendingJointActionV1,
    LiveResearcherDraftInspectionV1,
    LiveResearcherEditableDraftInspectionV1,
    LiveResearcherSpaceV1,
    LiveScriptedPlaybackInspectionV1,
    NoSharedObsLatestTransitionV1,
    NoSharedObsPresentationAuthorityV1,
    OracleLatestTransitionV1,
    OraclePresentationAuthorityV1,
    ReplayResearcherRosterAgentV1,
    build_no_shared_obs_authorized_current_endpoint_v1,
    build_oracle_authorized_current_endpoint_v1,
)
from scripts.dev.visual_debugger.protocol import (
    ActorPovHudFrameV1,
    ActorPovLiveDebuggerFrameV2,
    ActorPovPendingActionCardV1,
    ActorPovTargetReferenceV1,
    PendingActionCardV1,
    ResearcherHudFrameV2,
    ResearcherLiveDebuggerFrameV2,
    TargetReferenceV1,
)


def _require_live_header(
    raw_frame: ResearcherLiveDebuggerFrameV2 | ActorPovLiveDebuggerFrameV2,
    *,
    researcher: bool,
) -> None:
    expected_type = (
        ResearcherLiveDebuggerFrameV2 if researcher else ActorPovLiveDebuggerFrameV2
    )
    expected_kind = (
        "researcher_live_debugger" if researcher else "actor_pov_live_debugger"
    )
    expected_mode = "researcher" if researcher else "pov"
    if type(raw_frame) is not expected_type:
        raise TypeError(f"raw_frame must be the exact {expected_type.__name__} root.")
    if (
        type(raw_frame.schema_version) is not int
        or raw_frame.schema_version != 2
        or type(raw_frame.frame_kind) is not str
        or raw_frame.frame_kind != expected_kind
        or type(raw_frame.view_mode) is not str
        or raw_frame.view_mode != expected_mode
    ):
        raise ValueError("raw live frame does not retain its exact wire identity.")
    for name in (
        "session_id",
        "episode_id",
        "frame_id",
    ):
        value = getattr(raw_frame, name)
        if type(value) is not str or not value:
            raise TypeError(f"raw live {name} must be a nonempty exact string.")
    for name in (
        "run_generation",
        "revision",
        "frame_index",
        "simulator_step_count",
    ):
        value = getattr(raw_frame, name)
        if type(value) is not int or value < 0:
            raise TypeError(f"raw live {name} must be a non-negative exact int.")
    if raw_frame.frame_id != f"{raw_frame.episode_id}:frame:{raw_frame.frame_index}":
        raise ValueError("raw live frame ID is not canonical.")


def _require_oracle_used_containers(raw_frame: ResearcherLiveDebuggerFrameV2) -> None:
    if type(raw_frame.projection) is not ResearcherAnalyzerProjectionV2:
        raise TypeError("raw Oracle projection must use its exact V2 root.")
    if type(raw_frame.hud) is not ResearcherHudFrameV2:
        raise TypeError("raw Oracle HUD must use its exact V2 root.")
    pending = raw_frame.hud.pending_action
    if type(pending) is not PendingActionCardV1:
        raise TypeError("raw Oracle pending action must use its exact V1 root.")
    if type(pending.target) is not TargetReferenceV1:
        raise TypeError("raw Oracle pending target must use its exact V1 root.")


def _require_no_shared_used_containers(
    raw_frame: ActorPovLiveDebuggerFrameV2,
) -> None:
    if type(raw_frame.projection) is not ActorPovAnalyzerProjectionV1:
        raise TypeError("raw POV projection must use its exact V1 root.")
    if type(raw_frame.hud) is not ActorPovHudFrameV1:
        raise TypeError("raw POV HUD must use its exact V1 root.")
    pending = raw_frame.hud.pending_action
    if type(pending) is not ActorPovPendingActionCardV1:
        raise TypeError("raw POV pending action must use its exact V1 root.")
    if type(pending.target) is not ActorPovTargetReferenceV1:
        raise TypeError("raw POV pending target must use its exact V1 root.")


def _canonical_live_view(
    context: EvaluationEpisodeContextV1,
    current_frame: EvaluationFrameV1,
    incoming_transition_view: EvaluationTransitionViewV1 | None,
) -> EvaluationTransitionViewV1 | None:
    if type(context) is not EvaluationEpisodeContextV1:
        raise TypeError("context must be the exact EvaluationEpisodeContextV1 root.")
    if type(current_frame) is not EvaluationFrameV1:
        raise TypeError("current_frame must be the exact EvaluationFrameV1 root.")
    if current_frame.frame_index == 0:
        if incoming_transition_view is not None:
            raise ValueError("live frame zero cannot carry incoming T_(n-1).")
        return None
    if type(incoming_transition_view) is not EvaluationTransitionViewV1:
        raise TypeError("nonzero live frames require an exact incoming view.")
    view = EvaluationTransitionViewV1(
        context=incoming_transition_view.context,
        start_frame=incoming_transition_view.start_frame,
        transition=incoming_transition_view.transition,
        successor_frame=incoming_transition_view.successor_frame,
    )
    if view.context != context or view.successor_frame != current_frame:
        raise ValueError("incoming live view must enter the selected current frame.")
    return view


def _draft_lane_v1(
    *,
    target_action: int,
    raw_armed_lane: int | None,
) -> DraftArmedLaneV1:
    """Translate the raw UI lane into the presentation's named draft state."""
    if type(target_action) is not int or target_action < 0:
        raise TypeError("draft target action must be a non-negative exact int.")
    if raw_armed_lane is not None and (
        type(raw_armed_lane) is not int or raw_armed_lane not in (0, 1)
    ):
        raise TypeError("raw draft lane must be exact 0, 1, or None.")
    if raw_armed_lane is None or (raw_armed_lane == 0 and target_action == 0):
        return "none"
    return "basic" if raw_armed_lane == 0 else "ultimate"


def _pending_joint_action_v1(
    context: EvaluationEpisodeContextV1,
    raw_frame: ResearcherLiveDebuggerFrameV2,
) -> LivePendingJointActionV1 | None:
    """Project the exact next joint submission without slots or geometry."""
    if raw_frame.hud.pending_submission_scope != "joint_turn":
        return None
    rows: list[LivePendingJointActionRowV1] = []
    for pending in raw_frame.hud.pending_actions:
        roster = context.roster[pending.actor_global_slot]
        if not roster.configured_active:
            raise ValueError("Pending Joint Action cannot include an inactive actor.")
        target_action = pending.target_action
        if type(target_action) is not int:
            raise ValueError("researcher pending target action must be public.")
        armed_lane = _draft_lane_v1(
            target_action=target_action,
            raw_armed_lane=pending.armed_lane,
        )
        if armed_lane == "none":
            submitted_target_action = 0
            submitted_ultimate_action = 0
        else:
            submitted_target_action = target_action
            submitted_ultimate_action = pending.armed_lane
            if type(submitted_ultimate_action) is not int:
                raise ValueError("an armed pending action requires an exact lane.")
        rows.append(
            LivePendingJointActionRowV1(
                actor_presentation_key=oracle_presentation_key_v1(
                    authority_session_id=raw_frame.session_id,
                    public_agent_id=roster.public_agent_id,
                ),
                actor_public_agent_id=roster.public_agent_id,
                pending_action=LivePendingActionTupleV1(
                    move_action=pending.move_action,
                    target_action=submitted_target_action,
                    use_ultimate_action=submitted_ultimate_action,
                ),
            )
        )
    return LivePendingJointActionV1(
        current_simulator_step_count=raw_frame.simulator_step_count,
        action_rows=tuple(rows),
    )


def _oracle_latest_transition_v1(
    context: EvaluationEpisodeContextV1,
    incoming_view: EvaluationTransitionViewV1 | None,
    *,
    authority_session_id: str,
) -> OracleLatestTransitionV1 | None:
    if incoming_view is None:
        return None
    transition = incoming_view.transition
    acceptance = transition.facts.action_acceptance_facts
    submitted = acceptance.submitted_joint_action
    accepted = acceptance.accepted_joint_action
    catalog = context.static_mechanics_catalog
    rows: list[LatestTransitionActionRowV1] = []
    for actor_slot, roster in enumerate(context.roster):
        if not roster.configured_active:
            continue
        target_slots = catalog.global_recipient_slot_by_actor_and_target_action[
            actor_slot
        ]
        rows.append(
            LatestTransitionActionRowV1(
                actor_presentation_key=oracle_presentation_key_v1(
                    authority_session_id=authority_session_id,
                    public_agent_id=roster.public_agent_id,
                ),
                actor_public_agent_id=roster.public_agent_id,
                target_action_recipient_public_agent_id_by_id=tuple(
                    None
                    if target_slot is None
                    else context.roster[target_slot].public_agent_id
                    for target_slot in target_slots
                ),
                submitted_action=SubmittedActionTupleV1(
                    move_action=submitted.move[actor_slot],
                    target_action=submitted.select_target[actor_slot],
                    use_ultimate_action=submitted.use_ultimate[actor_slot],
                ),
                accepted_action=AcceptedActionTupleV1(
                    move_action=accepted.move[actor_slot],
                    target_action=accepted.select_target[actor_slot],
                    use_ultimate_action=accepted.use_ultimate[actor_slot],
                ),
            )
        )
    facts = transition.facts
    return OracleLatestTransitionV1(
        transition_kind="oracle_incoming_submitted_accepted",
        episode_id=transition.episode_id,
        incoming_transition_index=transition.transition_index,
        incoming_transition_id=transition.transition_id,
        incoming_start_frame_id=transition.start_frame_id,
        incoming_successor_frame_id=transition.successor_frame_id,
        incoming_start_simulator_step_count=facts.transition_start_step_count,
        incoming_successor_simulator_step_count=(
            incoming_view.successor_frame.simulator_step_count
        ),
        action_rows=tuple(rows),
    )


def _no_shared_latest_transition_v1(
    carrier: ActorPovAdjacentTransitionSliceV1 | None,
    *,
    endpoint_action_axis: AgentPovActionAxisV1,
) -> NoSharedObsLatestTransitionV1 | None:
    if carrier is None:
        return None
    action_axis = endpoint_action_axis
    transition = carrier.transition
    start = carrier.start_frame
    successor = carrier.successor_frame
    return NoSharedObsLatestTransitionV1(
        transition_kind="no_shared_obs_incoming_submitted_accepted",
        episode_id=transition.episode_id,
        incoming_transition_index=transition.transition_index,
        incoming_transition_id=transition.pov_transition_id,
        incoming_start_frame_id=transition.start_pov_frame_id,
        incoming_successor_frame_id=transition.successor_pov_frame_id,
        incoming_start_simulator_step_count=start.simulator_step_count,
        incoming_successor_simulator_step_count=successor.simulator_step_count,
        action_rows=(
            LatestTransitionActionRowV1(
                actor_presentation_key=action_axis.owner_presentation_key,
                actor_public_agent_id=action_axis.owner_public_agent_id,
                target_action_recipient_public_agent_id_by_id=(
                    action_axis.target_public_agent_id_by_action
                ),
                submitted_action=SubmittedActionTupleV1(
                    move_action=transition.submitted_action.move,
                    target_action=transition.submitted_action.select_target,
                    use_ultimate_action=transition.submitted_action.use_ultimate,
                ),
                accepted_action=AcceptedActionTupleV1(
                    move_action=transition.accepted_action.move,
                    target_action=transition.accepted_action.select_target,
                    use_ultimate_action=transition.accepted_action.use_ultimate,
                ),
            ),
        ),
        recipient_public_agent_id=action_axis.owner_public_agent_id,
        recipient_presentation_key=action_axis.owner_presentation_key,
    )


def build_live_oracle_authorized_presentation_v1(
    context: EvaluationEpisodeContextV1,
    current_frame: EvaluationFrameV1,
    incoming_transition_view: EvaluationTransitionViewV1 | None,
    raw_frame: ResearcherLiveDebuggerFrameV2,
) -> LiveOracleAuthorizedPresentationFrameV1:
    """Package one committed live Oracle ``s_n`` and its incoming/draft siblings."""
    _require_live_header(
        raw_frame,
        researcher=True,
    )
    _require_oracle_used_containers(raw_frame)
    incoming_view = _canonical_live_view(
        context,
        current_frame,
        incoming_transition_view,
    )
    if (
        raw_frame.episode_id != context.identity.episode_id
        or raw_frame.episode_id != current_frame.episode_id
        or raw_frame.frame_index != current_frame.frame_index
        or raw_frame.frame_id != current_frame.frame_id
        or raw_frame.simulator_step_count != current_frame.simulator_step_count
    ):
        raise ValueError("raw Oracle frame does not join the current live authority.")
    expected_events = (
        None if incoming_view is None else build_visual_event_batch_v2(incoming_view)
    )
    if raw_frame.projection.incoming_events != expected_events:
        raise ValueError("raw Oracle incoming events diverge from T_(n-1).")
    parts = build_replay_oracle_presentation_parts_v1(
        context,
        raw_frame.projection.scene,
        expected_events,
        authority_session_id=raw_frame.session_id,
        final_frame_index=current_frame.frame_index,
        selected_internal_slot=None,
        outgoing_transition=None,
    )
    controlled_slot = raw_frame.hud.controlled_global_slot
    endpoint = build_oracle_authorized_current_endpoint_v1(
        context=context,
        source_scene=raw_frame.projection.scene,
        authority_session_id=raw_frame.session_id,
        selected_internal_slot=controlled_slot,
    )
    if endpoint.scene != parts.current_scene:
        raise ValueError("Oracle endpoint scene diverged from the incoming packager.")
    submission_scope = raw_frame.hud.pending_submission_scope
    if submission_scope == "joint_turn":
        pending = raw_frame.hud.pending_action
        pending_target_action = (
            0 if pending.target.global_slot is None else pending.target_action
        )
        if pending_target_action is None:
            raise ValueError("controlled Oracle draft target action is unavailable.")
        input_inspection = LiveEditableDraftInspectionV1(
            inspection_kind="editable_live_draft",
            submission_scope="joint_turn",
            draft=build_live_oracle_draft_inspection_v1(
                context,
                current_frame,
                endpoint.scene,
                controlled_internal_slot=controlled_slot,
                draft_move_action=pending.move_action,
                draft_target_internal_slot=pending.target.global_slot,
                draft_armed_lane=_draft_lane_v1(
                    target_action=pending_target_action,
                    raw_armed_lane=pending.armed_lane,
                ),
            ),
        )
    elif submission_scope == "scripted_playback":
        input_inspection = LiveScriptedPlaybackInspectionV1(
            inspection_kind="scripted_playback_inspection",
            submission_scope="scripted_playback",
            editable_draft_available=False,
            advance_semantics="registered_script_frame",
        )
    else:
        raise ValueError("Oracle live frame has an invalid submission scope.")
    latest_transition = _oracle_latest_transition_v1(
        context,
        incoming_view,
        authority_session_id=raw_frame.session_id,
    )
    source = LiveOraclePresentationSourceIdentityV1(
        source_kind="live_oracle_frame",
        source_session_id=raw_frame.session_id,
        source_run_generation=raw_frame.run_generation,
        source_revision=raw_frame.revision,
        source_authority_epoch=raw_frame.revision,
        episode_id=raw_frame.episode_id,
        source_frame_index=raw_frame.frame_index,
        source_frame_id=raw_frame.frame_id,
        source_simulator_step_count=raw_frame.simulator_step_count,
        source_submission_scope=submission_scope,
        source_authorized_endpoint_digest_sha256=(
            endpoint.authorized_endpoint_digest_sha256
        ),
    )
    return LiveOracleAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="live_oracle",
        product_kind="combat_debugger",
        source=source,
        authority=OraclePresentationAuthorityV1(
            authority_kind="oracle",
            projection_basis="global_evaluation_projection",
        ),
        analysis_mode="analysis",
        current_endpoint=endpoint,
        latest_events=parts.incoming_summary,
        latest_transition=latest_transition,
        pending_joint_action=_pending_joint_action_v1(context, raw_frame),
        technical_frame=LiveOracleTechnicalFrameV1(
            technical_kind="live_oracle_technical_frame",
            episode_id=source.episode_id,
            evaluation_frame_index=source.source_frame_index,
            simulator_step_count=source.source_simulator_step_count,
            incoming_transition_id=(
                None
                if latest_transition is None
                else latest_transition.incoming_transition_id
            ),
        ),
        live_inspection=LiveOracleInspectionEnvelopeV1(
            envelope_kind="live_oracle_source_bound_inspection",
            source_session_id=source.source_session_id,
            source_run_generation=source.source_run_generation,
            source_revision=source.source_revision,
            source_authority_epoch=source.source_authority_epoch,
            episode_id=source.episode_id,
            source_frame_index=source.source_frame_index,
            source_frame_id=source.source_frame_id,
            source_simulator_step_count=source.source_simulator_step_count,
            inspection=input_inspection,
        ),
    )


def build_live_researcher_space_v1(
    oracle: LiveOracleAuthorizedPresentationFrameV1,
) -> LiveResearcherSpaceV1:
    """Project one validated Oracle epoch into non-battlefield researcher UI."""
    if type(oracle) is not LiveOracleAuthorizedPresentationFrameV1:
        raise TypeError("oracle must be an exact live Oracle presentation.")
    endpoint = oracle.current_endpoint
    action_axis = endpoint.action_axis
    if action_axis is None:
        raise ValueError("live researcher controls require a selected Oracle actor.")
    source = oracle.source
    directory_by_id = {
        row.public_agent_id: row for row in endpoint.identity_directory.identities
    }
    inspection = oracle.live_inspection.inspection
    if type(inspection) is LiveEditableDraftInspectionV1:
        source_draft = inspection.draft
        no_target = source_draft.decision_mask.target_actions[0]
        if type(no_target) is not AuthorizedNoTargetActionV1:
            raise ValueError("Oracle draft target zero changed its exact variant.")
        positive_targets: list[AuthorizedAxisOnlyTargetActionV1] = []
        for row in source_draft.decision_mask.target_actions[1:]:
            if (
                type(row) is AuthorizedVisibleTargetActionV1
                or type(row) is AuthorizedAxisOnlyTargetActionV1
            ):
                target_public_agent_id = row.target_public_agent_id
            else:
                raise ValueError("Oracle draft positive target changed its variant.")
            positive_targets.append(
                AuthorizedAxisOnlyTargetActionV1(
                    target_kind="axis_only_authorized_agent",
                    target_action=row.target_action,
                    display_name=row.display_name,
                    target_public_agent_id=target_public_agent_id,
                )
            )
        geometry_free_targets = (no_target, *positive_targets)
        geometry_free_mask = replace(
            source_draft.decision_mask,
            target_actions=geometry_free_targets,
        )
        pending_inspection = LiveResearcherEditableDraftInspectionV1(
            inspection_kind="editable_live_draft",
            submission_scope="joint_turn",
            draft=LiveResearcherDraftInspectionV1(
                schema_version=source_draft.schema_version,
                inspection_kind="live_draft_action",
                current_simulator_step_count=(
                    source_draft.current_simulator_step_count
                ),
                actor_presentation_key=source_draft.actor_presentation_key,
                actor_public_agent_id=source_draft.actor_public_agent_id,
                decision_mask=geometry_free_mask,
                draft_action=source_draft.draft_action,
                draft_target=geometry_free_targets[
                    source_draft.draft_action.target_action
                ],
                draft_legality=source_draft.draft_legality,
            ),
        )
    elif type(inspection) is LiveScriptedPlaybackInspectionV1:
        pending_inspection = inspection
    else:  # pragma: no cover - the Oracle root owns this exact union.
        raise AssertionError("validated Oracle inspection variant disappeared.")

    return LiveResearcherSpaceV1(
        researcher_space_kind="global_live_researcher_space",
        source_session_id=source.source_session_id,
        source_run_generation=source.source_run_generation,
        source_revision=source.source_revision,
        source_authority_epoch=source.source_authority_epoch,
        episode_id=source.episode_id,
        frame_index=source.source_frame_index,
        simulator_step_count=source.source_simulator_step_count,
        selected_public_agent_id=action_axis.owner_public_agent_id,
        identity_directory=endpoint.identity_directory,
        roster_agents=tuple(
            ReplayResearcherRosterAgentV1(
                presentation_key=agent.presentation_key,
                public_agent_id=agent.public_agent_id,
                team_id=agent.team_id,
                team_local_slot=directory_by_id[agent.public_agent_id].team_local_slot,
                class_id=agent.class_id,
                class_name=agent.class_name,
                life_state=agent.life_state,
                current_health=agent.current_health,
                maximum_health=agent.maximum_health,
                effective_movement_speed=agent.effective_movement_speed,
                ultimate_cooldown_remaining=agent.ultimate_cooldown_remaining,
                spawn_shield_remaining=agent.spawn_shield_remaining,
                steps_until_out_of_combat=agent.steps_until_out_of_combat,
                out_of_combat_delay_steps=agent.out_of_combat_delay_steps,
                statuses=agent.statuses,
                aura_modifiers=agent.aura_modifiers,
            )
            for agent in endpoint.scene.agents
        ),
        class_mechanics=endpoint.scene.class_mechanics,
        latest_transition=oracle.latest_transition,
        pending_joint_action=oracle.pending_joint_action,
        technical_frame=oracle.technical_frame,
        pending_inspection=pending_inspection,
    )


def build_live_no_shared_obs_authorized_presentation_v1(
    current_slice: ActorPovCurrentSliceV1,
    incoming_carrier: ActorPovAdjacentTransitionSliceV1 | None,
    raw_frame: ActorPovLiveDebuggerFrameV2,
    *,
    public_catalog: StaticMechanicsCatalogV1,
    incoming_visual_events: VisualEventBatchV2 | None,
    researcher_space: LiveResearcherSpaceV1,
) -> LiveNoSharedObsAuthorizedPresentationFrameV1:
    """Package one committed live NoSharedObs battlefield and researcher UI."""
    _require_live_header(
        raw_frame,
        researcher=False,
    )
    _require_no_shared_used_containers(raw_frame)
    if type(current_slice) is not ActorPovCurrentSliceV1:
        raise TypeError("current_slice must be the exact ActorPovCurrentSliceV1 root.")
    if type(public_catalog) is not StaticMechanicsCatalogV1:
        raise TypeError("public_catalog must be the exact V1 catalog root.")
    current_slice = ActorPovCurrentSliceV1.model_validate(
        current_slice.model_dump(mode="python")
    )
    expected_projection = build_actor_pov_analyzer_projection_v1(current_slice)
    if raw_frame.projection != expected_projection:
        raise ValueError("raw POV projection does not join the current slice.")
    if (
        raw_frame.episode_id != current_slice.episode_id
        or raw_frame.frame_index != current_slice.frame.frame_index
        or raw_frame.frame_id != current_slice.frame.source_frame_id
        or raw_frame.simulator_step_count != current_slice.frame.simulator_step_count
        or raw_frame.hud.controlled_public_agent_id != current_slice.public_agent_id
        or raw_frame.incoming_pov_transition_id
        != expected_projection.incoming_transition_id
    ):
        raise ValueError("raw POV frame does not join the current live recipient.")
    if current_slice.frame.frame_index == 0:
        if incoming_carrier is not None:
            raise ValueError("live NoSharedObs frame zero cannot carry a carrier.")
        carrier = None
    else:
        if type(incoming_carrier) is not ActorPovAdjacentTransitionSliceV1:
            raise TypeError("nonzero live NoSharedObs frames require an exact carrier.")
        carrier = ActorPovAdjacentTransitionSliceV1.model_validate(
            incoming_carrier.model_dump(mode="python")
        )
        if (
            carrier.episode_id != current_slice.episode_id
            or carrier.public_agent_id != current_slice.public_agent_id
            or carrier.selected_global_slot != current_slice.selected_global_slot
            or carrier.selected_team_local_slot
            != current_slice.selected_team_local_slot
            or carrier.configured_team_id != current_slice.configured_team_id
            or carrier.class_id != current_slice.class_id
            or carrier.axis_mapping != current_slice.axis_mapping
            or carrier.successor_frame != current_slice.frame
            or carrier.transition != current_slice.incoming_transition
        ):
            raise ValueError(
                "live NoSharedObs carrier does not enter the current slice."
            )
    parts = build_no_shared_obs_authorized_scene_v1(
        current_slice,
        public_catalog=public_catalog,
        authority_session_id=raw_frame.session_id,
    )
    endpoint = build_no_shared_obs_authorized_current_endpoint_v1(
        parts=parts,
        axis_mapping=current_slice.axis_mapping,
    )
    if carrier is None:
        if incoming_visual_events is not None:
            raise ValueError("live NoSharedObs frame zero cannot carry visual events.")
        visual_events = None
    else:
        if type(incoming_visual_events) is not VisualEventBatchV2:
            raise TypeError(
                "non-initial live NoSharedObs frames require exact visual events."
            )
        previous = build_no_shared_obs_authorized_scene_v1(
            carrier,
            public_catalog=public_catalog,
            authority_session_id=raw_frame.session_id,
            frame_index=carrier.start_frame.frame_index,
        )
        visual_events = build_agent_pov_visual_incoming_summary_v1(
            incoming_visual_events,
            transition_start_scene=previous.scene,
            successor_scene=parts.scene,
            recipient_public_agent_id=parts.recipient_public_agent_id,
            incoming_recipient_transition_id=carrier.transition.pov_transition_id,
            incoming_start_recipient_frame_id=carrier.transition.start_pov_frame_id,
            incoming_successor_recipient_frame_id=(
                carrier.transition.successor_pov_frame_id
            ),
        )
    latest_events = (
        None
        if carrier is None
        else build_live_no_shared_obs_incoming_summary_v1(
            carrier,
            public_catalog=public_catalog,
            authority_session_id=raw_frame.session_id,
        )
    )
    latest_transition = _no_shared_latest_transition_v1(
        carrier,
        endpoint_action_axis=endpoint.action_axis,
    )
    submission_scope = raw_frame.hud.pending_submission_scope
    if submission_scope == "joint_turn":
        pending = raw_frame.hud.pending_action
        pending_target_action = pending.target.target_action
        input_inspection = LiveEditableDraftInspectionV1(
            inspection_kind="editable_live_draft",
            submission_scope="joint_turn",
            draft=build_live_no_shared_obs_draft_inspection_v1(
                current_slice,
                parts,
                draft_move_action=pending.move_action,
                draft_target_action=pending_target_action,
                draft_armed_lane=_draft_lane_v1(
                    target_action=pending_target_action,
                    raw_armed_lane=pending.armed_lane,
                ),
            ),
        )
    elif submission_scope == "scripted_playback":
        input_inspection = LiveScriptedPlaybackInspectionV1(
            inspection_kind="scripted_playback_inspection",
            submission_scope="scripted_playback",
            editable_draft_available=False,
            advance_semantics="registered_script_frame",
        )
    else:
        raise ValueError("NoSharedObs live frame has an invalid submission scope.")
    source = LiveNoSharedObsPresentationSourceIdentityV1(
        source_kind="live_no_shared_obs_frame",
        source_session_id=raw_frame.session_id,
        source_run_generation=raw_frame.run_generation,
        source_revision=raw_frame.revision,
        source_authority_epoch=raw_frame.revision,
        episode_id=current_slice.episode_id,
        source_frame_index=current_slice.frame.frame_index,
        source_recipient_public_agent_id=current_slice.public_agent_id,
        source_recipient_frame_id=parts.source_recipient_frame_id,
        source_simulator_step_count=current_slice.frame.simulator_step_count,
        source_submission_scope=submission_scope,
        source_authorized_endpoint_digest_sha256=(
            endpoint.authorized_endpoint_digest_sha256
        ),
    )
    return LiveNoSharedObsAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="live_no_shared_obs_agent_pov",
        product_kind="combat_debugger",
        source=source,
        authority=NoSharedObsPresentationAuthorityV1(
            authority_kind="agent_pov",
            observation_mode="no_shared_obs",
            recipient_public_agent_id=parts.recipient_public_agent_id,
            recipient_presentation_key=parts.recipient_presentation_key,
            projection_basis="recipient_own_recorded_observation",
            exact_actor_input_export_available=True,
        ),
        analysis_mode="analysis",
        current_endpoint=endpoint,
        latest_events=latest_events,
        visual_events=visual_events,
        latest_transition=latest_transition,
        technical_frame=LiveNoSharedObsTechnicalFrameV1(
            technical_kind="live_no_shared_obs_technical_frame",
            episode_id=source.episode_id,
            recipient_frame_index=source.source_frame_index,
            simulator_step_count=source.source_simulator_step_count,
            incoming_recipient_transition_id=(
                None
                if latest_transition is None
                else latest_transition.incoming_transition_id
            ),
        ),
        live_inspection=LiveNoSharedObsInspectionEnvelopeV1(
            envelope_kind="live_no_shared_obs_source_bound_inspection",
            source_session_id=source.source_session_id,
            source_run_generation=source.source_run_generation,
            source_revision=source.source_revision,
            source_authority_epoch=source.source_authority_epoch,
            episode_id=source.episode_id,
            source_frame_index=source.source_frame_index,
            source_recipient_public_agent_id=(source.source_recipient_public_agent_id),
            source_recipient_frame_id=source.source_recipient_frame_id,
            source_simulator_step_count=source.source_simulator_step_count,
            inspection=input_inspection,
        ),
        researcher_space=researcher_space,
    )


__all__ = [
    "build_live_no_shared_obs_authorized_presentation_v1",
    "build_live_oracle_authorized_presentation_v1",
    "build_live_researcher_space_v1",
]
