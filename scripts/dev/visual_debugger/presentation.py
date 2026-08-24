"""Replay packaging seams for the authorized presentation resource."""

from __future__ import annotations

from typing import cast

from marl_battlegrounds.evaluation.models import (
    ActionAcceptanceFactsV1,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationTransitionV1,
    JointActionV1,
    StaticMechanicsCatalogV1,
    TransitionFactsV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.rendering.authorized_incoming import (
    build_replay_no_shared_obs_incoming_summary_v1,
    build_shared_obs_incoming_summary_v1,
)
from marl_battlegrounds.rendering.authorized_inspection import (
    build_replay_no_shared_obs_inspection_v1,
    build_replay_oracle_inspection_v1,
    build_replay_shared_obs_inspection_v1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    SharedObsAuthorizedScenePartsV1,
    build_no_shared_obs_authorized_scene_v1,
    build_shared_obs_authorized_scene_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AcceptedActionTupleV1,
    SubmittedActionTupleV1,
    build_agent_pov_visual_incoming_summary_v1,
    build_replay_oracle_presentation_parts_v1,
    oracle_presentation_key_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    SharedObsSourceMaterialProjectionV1,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovAnalyzerProjectionV1,
    ActorPovProjectionIndexV1,
    build_actor_pov_analyzer_projection_v1,
)
from marl_battlegrounds.rendering.scene import VisualEventBatchV2
from scripts.dev.visual_debugger.presentation_protocol import (
    AgentPovActionAxisV1,
    LatestTransitionActionRowV1,
    NoSharedObsLatestTransitionV1,
    NoSharedObsPresentationAuthorityV1,
    OracleLatestTransitionV1,
    OraclePresentationAuthorityV1,
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    ReplayNoSharedObsPresentationSourceIdentityV1,
    ReplayNoSharedObsTechnicalFrameV1,
    ReplayOracleAuthorizedPresentationFrameV1,
    ReplayOraclePresentationSourceIdentityV1,
    ReplayOracleTechnicalFrameV1,
    ReplaySharedObsAuthorizedPresentationFrameV1,
    ReplaySharedObsPresentationSourceIdentityV1,
    ReplaySharedObsTechnicalFrameV1,
    SharedObsLatestTransitionV1,
    SharedObsPresentationAuthorityV1,
    build_no_shared_obs_authorized_current_endpoint_v1,
    build_oracle_authorized_current_endpoint_v1,
    build_shared_obs_authorized_current_endpoint_v1,
)
from scripts.dev.visual_debugger.replay_protocol import (
    ActorPovReplayCompletionBadgeV1,
    ActorPovReplayViewerFrameV1,
    ReplayCursorV1,
    ResearcherReplayViewerFrameV1,
    SharedObsAgentPovReplayArtifactSummaryV1,
    SharedObsAgentPovReplayViewerFrameV1,
)


def _oracle_latest_transition_v1(
    context: EvaluationEpisodeContextV1,
    incoming_transition: EvaluationTransitionV1 | None,
    *,
    authority_session_id: str,
) -> OracleLatestTransitionV1 | None:
    if incoming_transition is None:
        return None
    if type(incoming_transition) is not EvaluationTransitionV1:
        raise TypeError("incoming_transition must use its exact evaluation root.")
    transition = EvaluationTransitionV1.model_validate(
        incoming_transition.model_dump(mode="python")
    )
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
    start_tick = transition.facts.transition_start_step_count
    return OracleLatestTransitionV1(
        transition_kind="oracle_incoming_submitted_accepted",
        episode_id=transition.episode_id,
        incoming_transition_index=transition.transition_index,
        incoming_transition_id=transition.transition_id,
        incoming_start_frame_id=transition.start_frame_id,
        incoming_successor_frame_id=transition.successor_frame_id,
        incoming_start_simulator_step_count=start_tick,
        incoming_successor_simulator_step_count=start_tick + 1,
        action_rows=tuple(rows),
    )


def _no_shared_obs_latest_transition_v1(
    source: ActorPovProjectionIndexV1,
    *,
    successor_frame_index: int,
    action_axis: AgentPovActionAxisV1,
) -> NoSharedObsLatestTransitionV1 | None:
    if type(source) is not ActorPovProjectionIndexV1:
        raise TypeError("source must use the exact POV projection index root.")
    source = ActorPovProjectionIndexV1(content=source.content)
    if type(successor_frame_index) is not int or not (
        0 <= successor_frame_index < len(source.content.frames)
    ):
        raise IndexError("successor_frame_index is outside the captured POV prefix.")
    if successor_frame_index == 0:
        return None
    transition = source.content.transitions[successor_frame_index - 1]
    start = source.content.frames[successor_frame_index - 1]
    successor = source.content.frames[successor_frame_index]
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


def build_replay_no_shared_obs_authorized_presentation_v1(
    source: ActorPovProjectionIndexV1,
    raw_frame: ActorPovReplayViewerFrameV1,
    *,
    public_catalog: StaticMechanicsCatalogV1,
    source_authority_epoch: int,
    incoming_visual_events: VisualEventBatchV2 | None,
) -> ReplayNoSharedObsAuthorizedPresentationFrameV1:
    """Package one committed recipient-local NoSharedObs replay frame."""
    if type(source) is not ActorPovProjectionIndexV1:
        raise TypeError("source must use the exact POV projection index root.")
    if type(raw_frame) is not ActorPovReplayViewerFrameV1:
        raise TypeError("raw_frame must use the exact ActorPov replay root.")
    if type(public_catalog) is not StaticMechanicsCatalogV1:
        raise TypeError("public_catalog must use its exact evaluation root.")
    if type(raw_frame.cursor) is not ReplayCursorV1:
        raise TypeError("raw_frame cursor must use the exact replay cursor root.")
    if type(raw_frame.projection) is not ActorPovAnalyzerProjectionV1:
        raise TypeError("raw_frame projection must use its exact POV root.")
    if (
        type(raw_frame.schema_version) is not int
        or raw_frame.schema_version != 1
        or type(raw_frame.cursor.schema_version) is not int
        or raw_frame.cursor.schema_version != 1
        or type(raw_frame.frame_kind) is not str
        or raw_frame.frame_kind != "actor_pov_replay_viewer"
        or type(raw_frame.view_mode) is not str
        or raw_frame.view_mode != "pov"
    ):
        raise ValueError("raw_frame must retain its exact NoSharedObs wire identity.")
    for name, value in (
        ("revision", raw_frame.revision),
        ("frame_index", raw_frame.cursor.frame_index),
        ("final_frame_index", raw_frame.cursor.final_frame_index),
        ("simulator_step_count", raw_frame.simulator_step_count),
    ):
        if type(value) is not int or value < 0:
            raise TypeError(f"raw_frame {name} must be a nonnegative Python int.")
    for name, value in (
        ("viewer_session_id", raw_frame.viewer_session_id),
        ("public_agent_id", raw_frame.public_agent_id),
        ("pov_frame_id", raw_frame.pov_frame_id),
    ):
        if type(value) is not str or not value.strip():
            raise TypeError(f"raw_frame {name} must be a nonempty Python string.")
    source = ActorPovProjectionIndexV1(content=source.content)
    public_catalog = StaticMechanicsCatalogV1.model_validate_json(
        public_catalog.model_dump_json()
    )
    content = source.content
    frame_index = raw_frame.cursor.frame_index
    final_frame_index = len(content.transitions)
    if not 0 <= frame_index < len(content.frames):
        raise ValueError("committed NoSharedObs cursor is outside POV content.")
    if (
        raw_frame.cursor.final_frame_index != final_frame_index
        or raw_frame.public_agent_id != content.public_agent_id
    ):
        raise ValueError("committed NoSharedObs frame does not join its POV content.")
    recorded_frame = content.frames[frame_index]
    expected_projection = build_actor_pov_analyzer_projection_v1(
        source,
        frame_index=frame_index,
    )
    if (
        raw_frame.pov_frame_id != recorded_frame.pov_frame_id
        or raw_frame.simulator_step_count != recorded_frame.simulator_step_count
        or raw_frame.projection != expected_projection
    ):
        raise ValueError("committed NoSharedObs projection does not join s_n.")

    current = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=public_catalog,
        authority_session_id=raw_frame.viewer_session_id,
        frame_index=frame_index,
    )
    endpoint = build_no_shared_obs_authorized_current_endpoint_v1(
        parts=current,
        axis_mapping=content.axis_mapping,
    )
    if frame_index == 0:
        if incoming_visual_events is not None:
            raise ValueError("NoSharedObs frame zero cannot carry visual events.")
        visual_events = None
    else:
        if type(incoming_visual_events) is not VisualEventBatchV2:
            raise TypeError(
                "non-initial NoSharedObs frames require exact visual events."
            )
        previous = build_no_shared_obs_authorized_scene_v1(
            source,
            public_catalog=public_catalog,
            authority_session_id=raw_frame.viewer_session_id,
            frame_index=frame_index - 1,
        )
        transition = content.transitions[frame_index - 1]
        visual_events = build_agent_pov_visual_incoming_summary_v1(
            incoming_visual_events,
            transition_start_scene=previous.scene,
            successor_scene=current.scene,
            recipient_public_agent_id=current.recipient_public_agent_id,
            incoming_recipient_transition_id=transition.pov_transition_id,
            incoming_start_recipient_frame_id=transition.start_pov_frame_id,
            incoming_successor_recipient_frame_id=transition.successor_pov_frame_id,
        )
    latest_events = build_replay_no_shared_obs_incoming_summary_v1(
        source,
        successor_frame_index=frame_index,
        public_catalog=public_catalog,
        authority_session_id=raw_frame.viewer_session_id,
    )
    latest_transition = _no_shared_obs_latest_transition_v1(
        source,
        successor_frame_index=frame_index,
        action_axis=endpoint.action_axis,
    )
    replay_inspection = build_replay_no_shared_obs_inspection_v1(source, current)
    return ReplayNoSharedObsAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="replay_no_shared_obs_agent_pov",
        product_kind="replay_viewer",
        source=ReplayNoSharedObsPresentationSourceIdentityV1(
            source_kind="replay_no_shared_obs_frame",
            source_session_id=raw_frame.viewer_session_id,
            source_revision=raw_frame.revision,
            source_authority_epoch=source_authority_epoch,
            episode_id=content.episode_id,
            source_frame_index=frame_index,
            source_final_frame_index=final_frame_index,
            source_recipient_public_agent_id=current.recipient_public_agent_id,
            source_recipient_frame_id=current.source_recipient_frame_id,
            source_simulator_step_count=current.source_simulator_step_count,
            source_observation_mode="no_shared_obs",
            source_authorized_endpoint_digest_sha256=(
                endpoint.authorized_endpoint_digest_sha256
            ),
        ),
        authority=NoSharedObsPresentationAuthorityV1(
            authority_kind="agent_pov",
            observation_mode="no_shared_obs",
            recipient_public_agent_id=current.recipient_public_agent_id,
            recipient_presentation_key=current.recipient_presentation_key,
            projection_basis="recipient_own_recorded_observation",
            exact_actor_input_export_available=True,
        ),
        analysis_mode="analysis",
        current_endpoint=endpoint,
        latest_events=latest_events,
        visual_events=visual_events,
        latest_transition=latest_transition,
        technical_frame=ReplayNoSharedObsTechnicalFrameV1(
            technical_kind="replay_no_shared_obs_technical_frame",
            frame_index=frame_index,
            simulator_step_count=current.source_simulator_step_count,
            incoming_recipient_transition_id=(
                None
                if latest_transition is None
                else latest_transition.incoming_transition_id
            ),
        ),
        replay_inspection=replay_inspection,
    )


def _shared_obs_latest_transition_v1(
    incoming_transition: EvaluationTransitionV1 | None,
    *,
    successor: SharedObsAuthorizedScenePartsV1,
    action_axis: AgentPovActionAxisV1,
    authorized_recipient_global_slot: int,
) -> SharedObsLatestTransitionV1 | None:
    if type(authorized_recipient_global_slot) is not int or not (
        0 <= authorized_recipient_global_slot < 10
    ):
        raise ValueError(
            "authorized_recipient_global_slot must be an exact V1 actor slot."
        )
    frame_index = successor.source_frame_index
    if frame_index == 0:
        if incoming_transition is not None:
            raise ValueError("SharedObs frame zero cannot receive incoming T_(n-1).")
        return None
    if type(incoming_transition) is not EvaluationTransitionV1:
        raise TypeError("non-initial SharedObs presentation requires exact T_(n-1).")
    transition = incoming_transition
    if (
        type(transition.schema_id) is not str
        or transition.schema_id != "marl_battlegrounds.evaluation.transition"
        or type(transition.schema_version) is not int
        or transition.schema_version != 1
    ):
        raise ValueError("incoming transition must retain its exact V1 schema.")
    for name in (
        "episode_id",
        "transition_id",
        "start_frame_id",
        "successor_frame_id",
    ):
        value = getattr(transition, name)
        if type(value) is not str or not value.strip():
            raise TypeError(f"incoming transition {name} must be a nonempty string.")
    if type(transition.transition_index) is not int:
        raise TypeError("incoming transition index must be an exact Python int.")
    if type(transition.facts) is not TransitionFactsV1:
        raise TypeError("incoming transition facts must use their exact V1 root.")
    facts = transition.facts
    if (
        type(facts.schema_id) is not str
        or facts.schema_id != "marl_battlegrounds.evaluation.transition_facts"
        or type(facts.schema_version) is not int
        or facts.schema_version != 1
        or facts.has_transition is not True
        or type(facts.transition_start_step_count) is not int
    ):
        raise ValueError("incoming transition facts must retain exact used headers.")
    expected_index = frame_index - 1
    episode_id = successor.source_episode_id
    if (
        transition.episode_id != episode_id
        or transition.transition_index != expected_index
        or transition.transition_id != f"{episode_id}:transition:{expected_index}"
        or transition.start_frame_id != f"{episode_id}:frame:{expected_index}"
        or transition.successor_frame_id != f"{episode_id}:frame:{frame_index}"
        or facts.transition_start_step_count + 1
        != successor.source_simulator_step_count
    ):
        raise ValueError("incoming transition must be exact T_(n-1).")
    if type(facts.action_acceptance_facts) is not ActionAcceptanceFactsV1:
        raise TypeError("incoming action acceptance must use its exact V1 root.")
    acceptance = facts.action_acceptance_facts
    submitted = acceptance.submitted_joint_action
    accepted = acceptance.accepted_joint_action
    if type(submitted) is not JointActionV1 or type(accepted) is not JointActionV1:
        raise TypeError("incoming actions must use exact joint-action roots.")
    for owner, name in ((submitted, "submitted"), (accepted, "accepted")):
        for values, field_name in (
            (owner.move, "move"),
            (owner.select_target, "select_target"),
            (owner.use_ultimate, "use_ultimate"),
        ):
            if type(values) is not tuple or len(values) != 10:
                raise ValueError(
                    f"incoming {name} {field_name} must retain ten actor rows."
                )
    submitted_row = SubmittedActionTupleV1(
        move_action=submitted.move[authorized_recipient_global_slot],
        target_action=submitted.select_target[authorized_recipient_global_slot],
        use_ultimate_action=submitted.use_ultimate[authorized_recipient_global_slot],
    )
    accepted_row = AcceptedActionTupleV1(
        move_action=accepted.move[authorized_recipient_global_slot],
        target_action=accepted.select_target[authorized_recipient_global_slot],
        use_ultimate_action=accepted.use_ultimate[authorized_recipient_global_slot],
    )
    prefix = (
        f"{episode_id}:shared-obs-visual-union:{successor.recipient_public_agent_id}"
    )
    return SharedObsLatestTransitionV1(
        transition_kind="shared_obs_incoming_submitted_accepted",
        episode_id=episode_id,
        incoming_transition_index=expected_index,
        incoming_transition_id=f"{prefix}:transition:{expected_index}",
        incoming_start_frame_id=f"{prefix}:frame:{expected_index}",
        incoming_successor_frame_id=f"{prefix}:frame:{frame_index}",
        incoming_start_simulator_step_count=facts.transition_start_step_count,
        incoming_successor_simulator_step_count=(successor.source_simulator_step_count),
        action_rows=(
            LatestTransitionActionRowV1(
                actor_presentation_key=action_axis.owner_presentation_key,
                actor_public_agent_id=action_axis.owner_public_agent_id,
                target_action_recipient_public_agent_id_by_id=(
                    action_axis.target_public_agent_id_by_action
                ),
                submitted_action=submitted_row,
                accepted_action=accepted_row,
            ),
        ),
        recipient_public_agent_id=successor.recipient_public_agent_id,
        recipient_presentation_key=successor.recipient_presentation_key,
    )


def build_replay_shared_obs_authorized_presentation_v1(
    raw_frame: SharedObsAgentPovReplayViewerFrameV1,
    *,
    public_catalog: StaticMechanicsCatalogV1,
    source_authority_epoch: int,
    authorized_recipient_global_slot: int,
    current_recipient_source_material: SharedObsSourceMaterialProjectionV1,
    current_active_nonrecipient_source_material: tuple[
        SharedObsSourceMaterialProjectionV1, ...
    ],
    previous_recipient_source_material: SharedObsSourceMaterialProjectionV1 | None,
    previous_active_nonrecipient_source_material: tuple[
        SharedObsSourceMaterialProjectionV1, ...
    ],
    incoming_visual_events: VisualEventBatchV2 | None,
    incoming_transition: EvaluationTransitionV1 | None,
    outgoing_transition: EvaluationTransitionV1 | None,
) -> ReplaySharedObsAuthorizedPresentationFrameV1:
    """Package one fixed-recipient SharedObs visual-union replay frame."""
    if type(raw_frame) is not SharedObsAgentPovReplayViewerFrameV1:
        raise TypeError("raw_frame must use the exact private SharedObs replay root.")
    if type(raw_frame.cursor) is not ReplayCursorV1:
        raise TypeError("raw_frame cursor must use the exact replay cursor root.")
    if (
        type(raw_frame.artifact_summary) is not SharedObsAgentPovReplayArtifactSummaryV1
        or type(raw_frame.completion) is not ActorPovReplayCompletionBadgeV1
    ):
        raise TypeError("raw_frame must use exact private SharedObs nested roots.")
    if (
        type(raw_frame.schema_version) is not int
        or raw_frame.schema_version != 1
        or type(raw_frame.cursor.schema_version) is not int
        or raw_frame.cursor.schema_version != 1
        or type(raw_frame.frame_kind) is not str
        or raw_frame.frame_kind != "shared_obs_agent_pov_replay_viewer"
        or type(raw_frame.view_mode) is not str
        or raw_frame.view_mode != "pov"
        or type(raw_frame.preset) is not str
        or raw_frame.preset != "analysis"
        or type(raw_frame.verbose) is not bool
        or raw_frame.verbose is not False
    ):
        raise ValueError("raw_frame must retain its exact SharedObs wire identity.")
    for name, value in (
        ("revision", raw_frame.revision),
        ("frame_index", raw_frame.cursor.frame_index),
        ("final_frame_index", raw_frame.cursor.final_frame_index),
        ("simulator_step_count", raw_frame.simulator_step_count),
    ):
        if type(value) is not int or value < 0:
            raise TypeError(f"raw_frame {name} must be a nonnegative Python int.")
    for name, value in (
        ("viewer_session_id", raw_frame.viewer_session_id),
        ("public_agent_id", raw_frame.public_agent_id),
        ("recipient_frame_id", raw_frame.recipient_frame_id),
        ("timeline_id", raw_frame.timeline_id),
    ):
        if type(value) is not str or not value.strip():
            raise TypeError(f"raw_frame {name} must be a nonempty Python string.")
    if type(public_catalog) is not StaticMechanicsCatalogV1:
        raise TypeError("public_catalog must use its exact evaluation root.")
    public_catalog = StaticMechanicsCatalogV1.model_validate_json(
        public_catalog.model_dump_json()
    )
    if (
        type(current_recipient_source_material)
        is not SharedObsSourceMaterialProjectionV1
    ):
        raise TypeError("current recipient source must use its exact SharedObs root.")
    if (
        type(current_active_nonrecipient_source_material) is not tuple
        or any(
            type(row) is not SharedObsSourceMaterialProjectionV1
            for row in current_active_nonrecipient_source_material
        )
        or type(previous_active_nonrecipient_source_material) is not tuple
        or any(
            type(row) is not SharedObsSourceMaterialProjectionV1
            for row in previous_active_nonrecipient_source_material
        )
    ):
        raise TypeError("SharedObs contributor sources must use exact tuples/roots.")
    frame_index = raw_frame.cursor.frame_index
    current_base = current_recipient_source_material.base_sensor_frame
    current = build_shared_obs_authorized_scene_v1(
        current_recipient_source_material,
        all_active_nonrecipient_source_material=(
            current_active_nonrecipient_source_material
        ),
        public_catalog=public_catalog,
        authority_session_id=raw_frame.viewer_session_id,
    )
    summary = raw_frame.artifact_summary
    local_prefix = (
        f"{summary.episode_id}:shared-obs-visual-union:{summary.public_agent_id}"
    )
    expected_incoming_id = (
        None if frame_index == 0 else f"{local_prefix}:transition:{frame_index - 1}"
    )
    if (
        raw_frame.public_agent_id != current_base.public_agent_id
        or raw_frame.public_agent_id != current.recipient_public_agent_id
        or summary.public_agent_id != current.recipient_public_agent_id
        or summary.episode_id != current.source_episode_id
        or raw_frame.recipient_frame_id != current.source_recipient_frame_id
        or raw_frame.recipient_frame_id != f"{local_prefix}:frame:{frame_index}"
        or raw_frame.timeline_id != f"{local_prefix}:timeline"
        or raw_frame.simulator_step_count != current_base.simulator_step_count
        or raw_frame.simulator_step_count != current.source_simulator_step_count
        or frame_index != current_base.frame_index
        or frame_index != current.source_frame_index
        or raw_frame.incoming_recipient_transition_id != expected_incoming_id
    ):
        raise ValueError("private SharedObs raw identity does not join authorized s_n.")
    previous: SharedObsAuthorizedScenePartsV1 | None
    if frame_index == 0:
        if (
            previous_recipient_source_material is not None
            or previous_active_nonrecipient_source_material
        ):
            raise ValueError("SharedObs frame zero cannot receive prior sources.")
        previous = None
    else:
        if (
            type(previous_recipient_source_material)
            is not SharedObsSourceMaterialProjectionV1
        ):
            raise TypeError(
                "non-initial SharedObs frames require an exact prior source."
            )
        previous = build_shared_obs_authorized_scene_v1(
            previous_recipient_source_material,
            all_active_nonrecipient_source_material=(
                previous_active_nonrecipient_source_material
            ),
            public_catalog=public_catalog,
            authority_session_id=raw_frame.viewer_session_id,
        )
    if previous is None:
        if incoming_visual_events is not None:
            raise ValueError("SharedObs frame zero cannot carry visual events.")
        visual_events = None
    else:
        if type(incoming_visual_events) is not VisualEventBatchV2:
            raise TypeError("non-initial SharedObs frames require exact visual events.")
        visual_events = build_agent_pov_visual_incoming_summary_v1(
            incoming_visual_events,
            transition_start_scene=previous.scene,
            successor_scene=current.scene,
            recipient_public_agent_id=current.recipient_public_agent_id,
            incoming_recipient_transition_id=cast(
                str,
                raw_frame.incoming_recipient_transition_id,
            ),
            incoming_start_recipient_frame_id=(
                f"{local_prefix}:frame:{frame_index - 1}"
            ),
            incoming_successor_recipient_frame_id=current.source_recipient_frame_id,
        )
    latest_events = build_shared_obs_incoming_summary_v1(previous, current)
    endpoint = build_shared_obs_authorized_current_endpoint_v1(
        parts=current,
        axis_mapping=current_recipient_source_material.axis_mapping,
    )
    latest_transition = _shared_obs_latest_transition_v1(
        incoming_transition,
        successor=current,
        action_axis=endpoint.action_axis,
        authorized_recipient_global_slot=authorized_recipient_global_slot,
    )
    replay_inspection = build_replay_shared_obs_inspection_v1(
        current,
        current_recipient_source_material,
        authorized_recipient_global_slot=authorized_recipient_global_slot,
        outgoing_transition=outgoing_transition,
        final_frame_index=raw_frame.cursor.final_frame_index,
    )
    return ReplaySharedObsAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="replay_shared_obs_agent_pov",
        product_kind="replay_viewer",
        source=ReplaySharedObsPresentationSourceIdentityV1(
            source_kind="replay_shared_obs_visual_union_frame",
            source_session_id=raw_frame.viewer_session_id,
            source_revision=raw_frame.revision,
            source_authority_epoch=source_authority_epoch,
            episode_id=current.source_episode_id,
            source_frame_index=frame_index,
            source_final_frame_index=raw_frame.cursor.final_frame_index,
            source_recipient_public_agent_id=current.recipient_public_agent_id,
            source_recipient_frame_id=current.source_recipient_frame_id,
            source_simulator_step_count=current.source_simulator_step_count,
            source_observation_mode="shared_obs_visual_union",
            source_authorized_endpoint_digest_sha256=(
                endpoint.authorized_endpoint_digest_sha256
            ),
        ),
        authority=SharedObsPresentationAuthorityV1(
            authority_kind="agent_pov",
            observation_mode="shared_obs_visual_union",
            recipient_public_agent_id=current.recipient_public_agent_id,
            recipient_presentation_key=current.recipient_presentation_key,
            projection_basis="authorized_same_epoch_sensor_source_visual_union",
            exact_actor_input_export_available=False,
        ),
        analysis_mode="analysis",
        current_endpoint=endpoint,
        latest_events=latest_events,
        visual_events=visual_events,
        latest_transition=latest_transition,
        technical_frame=ReplaySharedObsTechnicalFrameV1(
            technical_kind="replay_shared_obs_technical_frame",
            frame_index=frame_index,
            simulator_step_count=current.source_simulator_step_count,
            incoming_recipient_transition_id=(
                None
                if latest_transition is None
                else latest_transition.incoming_transition_id
            ),
        ),
        replay_inspection=replay_inspection,
    )


def build_replay_oracle_authorized_presentation_v1(
    context: EvaluationEpisodeContextV1,
    current_frame: EvaluationFrameV1,
    raw_frame: ResearcherReplayViewerFrameV1,
    *,
    source_authority_epoch: int,
    selected_internal_slot: int | None,
    incoming_transition: EvaluationTransitionV1 | None,
    outgoing_transition: EvaluationTransitionV1 | None,
) -> ReplayOracleAuthorizedPresentationFrameV1:
    """Package one committed Oracle ``s_n`` with incoming and outgoing siblings."""
    if type(raw_frame) is not ResearcherReplayViewerFrameV1:
        raise TypeError(
            "raw_frame must be the exact ResearcherReplayViewerFrameV1 root."
        )
    if type(context) is not EvaluationEpisodeContextV1:
        raise TypeError("context must be the exact EvaluationEpisodeContextV1 root.")
    if type(current_frame) is not EvaluationFrameV1:
        raise TypeError("current_frame must be the exact EvaluationFrameV1 root.")
    context = EvaluationEpisodeContextV1.model_validate(
        context.model_dump(mode="python")
    )
    current_frame = EvaluationFrameV1.model_validate(
        current_frame.model_dump(mode="python")
    )
    reference = raw_frame.artifact_summary.replay_reference
    if canonical_digest_sha256(context) != reference.context_digest_sha256:
        raise ValueError("context must match the replay artifact reference digest.")
    if (
        current_frame.episode_id != reference.episode_id
        or current_frame.frame_index != raw_frame.cursor.frame_index
        or current_frame.frame_id != raw_frame.frame_id
        or current_frame.simulator_step_count != raw_frame.simulator_step_count
    ):
        raise ValueError("current evaluation frame must identity-join the raw frame.")
    frame_index = raw_frame.cursor.frame_index
    if frame_index == 0:
        if incoming_transition is not None:
            raise ValueError("frame zero cannot receive an incoming transition.")
    elif (
        type(incoming_transition) is not EvaluationTransitionV1
        or incoming_transition.transition_index != frame_index - 1
        or incoming_transition.successor_frame_id != raw_frame.frame_id
        or incoming_transition.facts.transition_start_step_count + 1
        != raw_frame.simulator_step_count
    ):
        raise ValueError("incoming transition must be exact T_(n-1).")

    parts = build_replay_oracle_presentation_parts_v1(
        context,
        raw_frame.projection.scene,
        raw_frame.projection.incoming_events,
        authority_session_id=raw_frame.viewer_session_id,
        final_frame_index=raw_frame.cursor.final_frame_index,
        selected_internal_slot=None,
        outgoing_transition=None,
    )
    inspection = build_replay_oracle_inspection_v1(
        context,
        current_frame,
        parts.current_scene,
        inspection_internal_slot=selected_internal_slot,
        outgoing_transition=outgoing_transition,
        final_frame_index=raw_frame.cursor.final_frame_index,
    )
    endpoint = build_oracle_authorized_current_endpoint_v1(
        context=context,
        source_scene=raw_frame.projection.scene,
        authority_session_id=raw_frame.viewer_session_id,
        selected_internal_slot=selected_internal_slot,
    )
    if endpoint.scene != parts.current_scene:
        raise ValueError("Oracle endpoint scene diverged from the incoming packager.")
    latest_transition = _oracle_latest_transition_v1(
        context,
        incoming_transition,
        authority_session_id=raw_frame.viewer_session_id,
    )
    source = ReplayOraclePresentationSourceIdentityV1(
        source_kind="replay_oracle_frame",
        source_session_id=raw_frame.viewer_session_id,
        source_revision=raw_frame.revision,
        source_authority_epoch=source_authority_epoch,
        source_artifact_id=reference.artifact_id,
        source_timeline_id=raw_frame.timeline_id,
        source_replay_schema_version=reference.replay_schema_version,
        source_context_digest_sha256=reference.context_digest_sha256,
        source_trajectory_content_digest_sha256=(
            reference.trajectory_content_digest_sha256
        ),
        source_artifact_digest_sha256=reference.canonical_digest_sha256,
        episode_id=reference.episode_id,
        source_frame_index=frame_index,
        source_final_frame_index=raw_frame.cursor.final_frame_index,
        source_frame_id=raw_frame.frame_id,
        source_simulator_step_count=raw_frame.simulator_step_count,
        source_cursor_generation=raw_frame.cursor.cursor_generation,
        source_choreography_generation=raw_frame.cursor.choreography_generation,
        source_recorded_ordinary_movement_distance_scale=(
            raw_frame.recorded_ordinary_movement_distance_scale
        ),
        source_authorized_endpoint_digest_sha256=(
            endpoint.authorized_endpoint_digest_sha256
        ),
    )
    return ReplayOracleAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="replay_oracle",
        product_kind="replay_viewer",
        source=source,
        authority=OraclePresentationAuthorityV1(
            authority_kind="oracle",
            projection_basis="global_evaluation_projection",
        ),
        analysis_mode="analysis",
        current_endpoint=endpoint,
        latest_events=parts.incoming_summary,
        latest_transition=latest_transition,
        technical_frame=ReplayOracleTechnicalFrameV1(
            technical_kind="replay_oracle_technical_frame",
            artifact_digest_prefix=(reference.canonical_digest_sha256[:12]),
            frame_index=frame_index,
            simulator_step_count=raw_frame.simulator_step_count,
            incoming_transition_id=(
                None
                if latest_transition is None
                else latest_transition.incoming_transition_id
            ),
            recorded_ordinary_movement_distance_scale=(
                raw_frame.recorded_ordinary_movement_distance_scale
            ),
        ),
        replay_inspection=inspection,
    )


__all__ = [
    "build_replay_no_shared_obs_authorized_presentation_v1",
    "build_replay_oracle_authorized_presentation_v1",
    "build_replay_shared_obs_authorized_presentation_v1",
]
