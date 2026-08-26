"""Focused CP2.5-B five-leaf protocol, integrity, and privacy proofs."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Literal, TypeGuard, cast

import pytest
from pydantic import TypeAdapter, ValidationError
from scripts.dev.visual_debugger.control import (
    create_session,
    select_controlled_actor,
    submit_next_script_frame,
)
from scripts.dev.visual_debugger.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
)
from scripts.dev.visual_debugger.frame import build_debugger_frame
from scripts.dev.visual_debugger.live_presentation import (
    build_live_oracle_authorized_presentation_v1,
    build_live_researcher_space_v1,
)
from scripts.dev.visual_debugger.presentation import (
    build_replay_oracle_authorized_presentation_v1,
    build_replay_researcher_space_v1,
)
from scripts.dev.visual_debugger.presentation_protocol import (
    AgentPovActionAxisV1,
    AuthorizedPresentationFrameV1,
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
    LiveScriptedPlaybackInspectionV1,
    MovementActionDisplayRowV1,
    NoSharedObsLatestTransitionV1,
    NoSharedObsPresentationAuthorityV1,
    NoSharedObsUpcomingTransitionV1,
    OracleActionAxisV1,
    OracleAuthorizedCurrentEndpointV1,
    PresentationApiErrorV1,
    PresentationResourceResultV1,
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    ReplayNoSharedObsPresentationSourceIdentityV1,
    ReplayNoSharedObsTechnicalFrameV1,
    ReplayOracleAuthorizedPresentationFrameV1,
    ReplayOraclePresentationSourceIdentityV1,
    ReplayResearcherSpaceV1,
    ReplaySharedObsAuthorizedPresentationFrameV1,
    ReplaySharedObsPresentationSourceIdentityV1,
    ReplaySharedObsTechnicalFrameV1,
    SharedObsLatestTransitionV1,
    SharedObsPresentationAuthorityV1,
    SharedObsUpcomingTransitionV1,
    TargetAgentActionDisplayRowV1,
    TargetNoneActionDisplayRowV1,
    UltimateChoiceDisplayRowV1,
    build_no_shared_obs_authorized_current_endpoint_v1,
    build_oracle_authorized_current_endpoint_v1,
    build_shared_obs_authorized_current_endpoint_v1,
    canonical_authorized_endpoint_digest_sha256,
)
from scripts.dev.visual_debugger.protocol import ResearcherLiveDebuggerFrameV2
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.evaluation_fixtures import CapturedEvaluationTrajectory
from tests.test_rendering_authorized_inspection import (
    _InspectionCases,
    _no_shared_current,
    _pov_index,
    _pov_slice,
    _shared_current,
)
from tests.test_rendering_authorized_inspection import (
    inspection_cases as _inspection_cases_fixture,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from tests.test_visual_debugger_authorized_presentation import _build_raw_frames
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV1
from marl_battlegrounds.evaluation.pov import (
    ActorPovAxisMappingV1,
    ActorPovTransitionV1,
    build_actor_pov_adjacent_transition_slice_v1,
)
from marl_battlegrounds.rendering.authorized_incoming import (
    build_live_no_shared_obs_incoming_summary_v1,
    build_replay_no_shared_obs_incoming_summary_v1,
    build_shared_obs_incoming_summary_v1,
)
from marl_battlegrounds.rendering.authorized_inspection import (
    AuthorizedAxisOnlyTargetActionV1,
    AuthorizedDecisionMaskV1,
    AuthorizedNoTargetActionV1,
    AuthorizedVisibleTargetActionV1,
    ReplayInspectionPresentationV1,
    build_live_no_shared_obs_draft_inspection_v1,
    build_live_oracle_draft_inspection_v1,
    build_replay_no_shared_obs_inspection_v1,
    build_replay_shared_obs_inspection_v1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    build_no_shared_obs_authorized_scene_v1,
    pov_presentation_key_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AcceptedActionTupleV1,
    AgentPovVisualIncomingSummaryV1,
    AuthorizedAgentV1,
    AuthorizedBattlefieldSceneV1,
    SubmittedActionTupleV1,
    build_agent_pov_visual_incoming_summary_v1,
    build_oracle_authorized_scene_v1,
    build_replay_oracle_presentation_parts_v1,
    oracle_presentation_key_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import build_visual_event_batch_v2
from marl_battlegrounds.rendering.scene import AgentSceneV2, BattlefieldSceneV2


def _replay_researcher_space_at(
    trajectory: CapturedEvaluationTrajectory,
    *,
    frame_index: int,
    selected_global_slot: int,
    session: str,
) -> ReplayResearcherSpaceV1:
    raw = _build_raw_frames(trajectory, viewer_session_id=session)[frame_index]
    final_frame_index = len(trajectory.transitions)
    return build_replay_researcher_space_v1(
        trajectory.context,
        raw.projection.scene,
        authority_session_id=session,
        final_frame_index=final_frame_index,
        selected_global_slot=selected_global_slot,
        incoming_transition=(
            None if frame_index == 0 else trajectory.transitions[frame_index - 1]
        ),
        outgoing_transition=(
            None
            if frame_index == final_frame_index
            else trajectory.transitions[frame_index]
        ),
    )


class _PoisonAuthorizedAgentV1(AuthorizedAgentV1):
    pass


class _PoisonBattlefieldSceneV2(BattlefieldSceneV2):
    pass


class _PoisonAgentSceneV2(AgentSceneV2):
    pass


def _agent_visual_events(
    trajectory: CapturedEvaluationTrajectory,
    *,
    transition_index: int,
    transition_start_scene: AuthorizedBattlefieldSceneV1,
    successor_scene: AuthorizedBattlefieldSceneV1,
    recipient_public_agent_id: str,
    incoming_recipient_transition_id: str,
    incoming_start_recipient_frame_id: str,
    incoming_successor_recipient_frame_id: str,
) -> AgentPovVisualIncomingSummaryV1:
    return build_agent_pov_visual_incoming_summary_v1(
        build_visual_event_batch_v2(
            EvaluationTransitionViewV1(
                context=trajectory.context,
                start_frame=trajectory.frames[transition_index],
                transition=trajectory.transitions[transition_index],
                successor_frame=trajectory.frames[transition_index + 1],
            )
        ),
        transition_start_scene=transition_start_scene,
        successor_scene=successor_scene,
        recipient_public_agent_id=recipient_public_agent_id,
        incoming_recipient_transition_id=incoming_recipient_transition_id,
        incoming_start_recipient_frame_id=incoming_start_recipient_frame_id,
        incoming_successor_recipient_frame_id=(incoming_successor_recipient_frame_id),
    )


class _PoisonEvaluationEpisodeContextV1(EvaluationEpisodeContextV1):
    pass


class _PoisonReplayOracleSourceIdentityV1(ReplayOraclePresentationSourceIdentityV1):
    pass


class _PoisonReplayOracleAuthorizedPresentationFrameV1(
    ReplayOracleAuthorizedPresentationFrameV1
):
    pass


class _PoisonPresentationApiErrorV1(PresentationApiErrorV1):
    pass


class _PoisonPresentationResourceResultV1(PresentationResourceResultV1):
    pass


class _OutcomeString(str):
    pass


@dataclass(frozen=True, slots=True)
class _FiveFrames:
    cases: _InspectionCases
    live_oracle: LiveOracleAuthorizedPresentationFrameV1
    live_no_shared: LiveNoSharedObsAuthorizedPresentationFrameV1
    replay_oracle: ReplayOracleAuthorizedPresentationFrameV1
    replay_no_shared: ReplayNoSharedObsAuthorizedPresentationFrameV1
    replay_shared: ReplaySharedObsAuthorizedPresentationFrameV1

    @property
    def rows(
        self,
    ) -> tuple[
        LiveOracleAuthorizedPresentationFrameV1
        | LiveNoSharedObsAuthorizedPresentationFrameV1
        | ReplayOracleAuthorizedPresentationFrameV1
        | ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1,
        ...,
    ]:
        return (
            self.live_oracle,
            self.live_no_shared,
            self.replay_oracle,
            self.replay_no_shared,
            self.replay_shared,
        )


_PUBLIC_FIELD_BY_PRESENTATION_KEY_FIELD = {
    "presentation_key": "public_agent_id",
    "source_presentation_key": "source_public_agent_id",
    "agent_presentation_key": "agent_public_agent_id",
    "recipient_presentation_key": "recipient_public_agent_id",
    "owner_presentation_key": "owner_public_agent_id",
    "target_presentation_key": "target_public_agent_id",
    "assigned_presentation_key": "assigned_public_agent_id",
    "actor_presentation_key": "actor_public_agent_id",
}


def _presentation_key_paths(
    value: object,
    *,
    path: tuple[str | int, ...] = (),
) -> tuple[tuple[tuple[str | int, ...], str | None, str | None], ...]:
    rows: list[tuple[tuple[str | int, ...], str | None, str | None]] = []
    if isinstance(value, dict):
        node: dict[str, object] = cast(dict[str, object], value)
        for name, nested in node.items():
            if name == "presentation_key" or name.endswith("_presentation_key"):
                public_field = _PUBLIC_FIELD_BY_PRESENTATION_KEY_FIELD[name]
                key = nested
                public_id = node[public_field]
                assert key is None or type(key) is str
                assert public_id is None or type(public_id) is str
                rows.append(((*path, name), key, public_id))
            rows.extend(_presentation_key_paths(nested, path=(*path, name)))
    elif isinstance(value, list):
        sequence: list[object] = cast(list[object], value)
        for index, nested in enumerate(sequence):
            rows.extend(_presentation_key_paths(nested, path=(*path, index)))
    return tuple(rows)


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    return type(value) is dict


def _is_json_array(value: object) -> TypeGuard[list[object]]:
    return type(value) is list


def _replace_path(
    root: dict[str, object],
    path: tuple[str | int, ...],
    value: object,
) -> None:
    node: object = root
    for element in path[:-1]:
        if isinstance(element, str):
            if not _is_json_object(node):
                raise AssertionError("serialized key path left its object owner")
            node = node[element]
        else:
            if not _is_json_array(node):
                raise AssertionError("serialized key path left its array owner")
            node = node[element]
    final = path[-1]
    if isinstance(final, str):
        if not _is_json_object(node):
            raise AssertionError("serialized key path has no object owner")
        node[final] = value
    else:
        if not _is_json_array(node):
            raise AssertionError("serialized key path has no array owner")
        node[final] = value


def _serialized_string_values(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    result: set[str] = set()
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        for nested in mapping.values():
            result.update(_serialized_string_values(nested))
    elif isinstance(value, list):
        sequence = cast(list[object], value)
        for nested in sequence:
            result.update(_serialized_string_values(nested))
    return result


def _nested_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        result.update(mapping)
        for nested in mapping.values():
            result.update(_nested_keys(nested))
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            result.update(_nested_keys(nested))
    return result


def _axis_from_mask(
    mask: AuthorizedDecisionMaskV1,
    *,
    oracle_team_by_public_id: dict[str, int] | None = None,
) -> OracleActionAxisV1 | AgentPovActionAxisV1:
    targets: list[TargetNoneActionDisplayRowV1 | TargetAgentActionDisplayRowV1] = []
    owner_team = (
        None
        if oracle_team_by_public_id is None
        else oracle_team_by_public_id[mask.owner_public_agent_id]
    )
    for row in mask.target_actions:
        if type(row) is AuthorizedNoTargetActionV1:
            targets.append(
                TargetNoneActionDisplayRowV1(
                    target_kind="target_none",
                    target_action=0,
                    display_name=row.display_name,
                )
            )
            continue
        assert type(row) in (
            AuthorizedVisibleTargetActionV1,
            AuthorizedAxisOnlyTargetActionV1,
        )
        target_row = cast(
            AuthorizedVisibleTargetActionV1 | AuthorizedAxisOnlyTargetActionV1,
            row,
        )
        relation = (
            "same_team"
            if (oracle_team_by_public_id is None and target_row.target_action <= 5)
            or (
                oracle_team_by_public_id is not None
                and oracle_team_by_public_id[target_row.target_public_agent_id]
                == owner_team
            )
            else "opponent"
        )
        targets.append(
            TargetAgentActionDisplayRowV1(
                target_kind="public_agent",
                target_action=target_row.target_action,
                display_name=target_row.display_name,
                target_public_agent_id=target_row.target_public_agent_id,
                target_relation=relation,
            )
        )
    movement_actions = tuple(
        MovementActionDisplayRowV1(
            move_action=index,
            display_name=name,
        )
        for index, name in enumerate(mask.movement_action_display_names)
    )
    target_actions = tuple(targets)
    ultimate_choices = tuple(
        UltimateChoiceDisplayRowV1(
            use_ultimate_action=index,
            display_name=name,
        )
        for index, name in enumerate(mask.use_ultimate_action_display_names)
    )
    if oracle_team_by_public_id is not None:
        return OracleActionAxisV1(
            axis_kind="oracle_actor_action_axis",
            owner_presentation_key=mask.owner_presentation_key,
            owner_public_agent_id=mask.owner_public_agent_id,
            movement_actions=movement_actions,
            target_actions=target_actions,
            ultimate_choices=ultimate_choices,
        )
    return AgentPovActionAxisV1(
        axis_kind="agent_pov_action_axis",
        owner_presentation_key=mask.owner_presentation_key,
        owner_public_agent_id=mask.owner_public_agent_id,
        movement_actions=movement_actions,
        target_actions=target_actions,
        ultimate_choices=ultimate_choices,
    )


def _agent_latest_transition(
    transition: ActorPovTransitionV1,
    *,
    axis: AgentPovActionAxisV1,
    start_tick: int,
) -> NoSharedObsLatestTransitionV1:
    row = LatestTransitionActionRowV1(
        actor_presentation_key=axis.owner_presentation_key,
        actor_public_agent_id=axis.owner_public_agent_id,
        target_action_recipient_public_agent_id_by_id=(
            axis.target_public_agent_id_by_action
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
    )
    return NoSharedObsLatestTransitionV1(
        transition_kind="no_shared_obs_incoming_submitted_accepted",
        episode_id=transition.episode_id,
        incoming_transition_index=transition.transition_index,
        incoming_transition_id=transition.pov_transition_id,
        incoming_start_frame_id=transition.start_pov_frame_id,
        incoming_successor_frame_id=transition.successor_pov_frame_id,
        incoming_start_simulator_step_count=start_tick,
        incoming_successor_simulator_step_count=start_tick + 1,
        action_rows=(row,),
        recipient_public_agent_id=axis.owner_public_agent_id,
        recipient_presentation_key=axis.owner_presentation_key,
    )


def _shared_latest_transition(
    cases: _InspectionCases,
    *,
    axis: AgentPovActionAxisV1,
    incoming_transition_index: int,
    start_tick: int,
) -> SharedObsLatestTransitionV1:
    transition = cases.shared.transitions[incoming_transition_index]
    acceptance = transition.facts.action_acceptance_facts
    submitted = acceptance.submitted_joint_action
    accepted = acceptance.accepted_joint_action
    row = LatestTransitionActionRowV1(
        actor_presentation_key=axis.owner_presentation_key,
        actor_public_agent_id=axis.owner_public_agent_id,
        target_action_recipient_public_agent_id_by_id=(
            axis.target_public_agent_id_by_action
        ),
        submitted_action=SubmittedActionTupleV1(
            move_action=submitted.move[0],
            target_action=submitted.select_target[0],
            use_ultimate_action=submitted.use_ultimate[0],
        ),
        accepted_action=AcceptedActionTupleV1(
            move_action=accepted.move[0],
            target_action=accepted.select_target[0],
            use_ultimate_action=accepted.use_ultimate[0],
        ),
    )
    episode = transition.episode_id
    recipient = axis.owner_public_agent_id
    prefix = f"{episode}:shared-obs-visual-union:{recipient}"
    return SharedObsLatestTransitionV1(
        transition_kind="shared_obs_incoming_submitted_accepted",
        episode_id=episode,
        incoming_transition_index=incoming_transition_index,
        incoming_transition_id=f"{prefix}:transition:{incoming_transition_index}",
        incoming_start_frame_id=f"{prefix}:frame:{incoming_transition_index}",
        incoming_successor_frame_id=(f"{prefix}:frame:{incoming_transition_index + 1}"),
        incoming_start_simulator_step_count=start_tick,
        incoming_successor_simulator_step_count=start_tick + 1,
        action_rows=(row,),
        recipient_public_agent_id=recipient,
        recipient_presentation_key=axis.owner_presentation_key,
    )


def _no_shared_upcoming_transition(
    inspection: ReplayInspectionPresentationV1 | None,
    *,
    axis: AgentPovActionAxisV1,
) -> NoSharedObsUpcomingTransitionV1 | None:
    if inspection is None:
        return None
    reference = inspection.transition_reference
    return NoSharedObsUpcomingTransitionV1(
        transition_kind="no_shared_obs_outgoing_submitted_accepted",
        episode_id=inspection.episode_id,
        outgoing_transition_index=inspection.outgoing_transition_index,
        outgoing_transition_id=reference.transition_id,
        outgoing_start_frame_id=reference.start_frame_id,
        outgoing_successor_frame_id=reference.successor_frame_id,
        outgoing_start_simulator_step_count=inspection.current_simulator_step_count,
        outgoing_successor_simulator_step_count=(
            inspection.current_simulator_step_count + 1
        ),
        action_rows=(
            LatestTransitionActionRowV1(
                actor_presentation_key=axis.owner_presentation_key,
                actor_public_agent_id=axis.owner_public_agent_id,
                target_action_recipient_public_agent_id_by_id=(
                    axis.target_public_agent_id_by_action
                ),
                submitted_action=inspection.submitted_action,
                accepted_action=inspection.accepted_action,
            ),
        ),
        recipient_public_agent_id=axis.owner_public_agent_id,
        recipient_presentation_key=axis.owner_presentation_key,
    )


def _shared_upcoming_transition(
    inspection: ReplayInspectionPresentationV1 | None,
    *,
    axis: AgentPovActionAxisV1,
) -> SharedObsUpcomingTransitionV1 | None:
    if inspection is None:
        return None
    reference = inspection.transition_reference
    return SharedObsUpcomingTransitionV1(
        transition_kind="shared_obs_outgoing_submitted_accepted",
        episode_id=inspection.episode_id,
        outgoing_transition_index=inspection.outgoing_transition_index,
        outgoing_transition_id=reference.transition_id,
        outgoing_start_frame_id=reference.start_frame_id,
        outgoing_successor_frame_id=reference.successor_frame_id,
        outgoing_start_simulator_step_count=inspection.current_simulator_step_count,
        outgoing_successor_simulator_step_count=(
            inspection.current_simulator_step_count + 1
        ),
        action_rows=(
            LatestTransitionActionRowV1(
                actor_presentation_key=axis.owner_presentation_key,
                actor_public_agent_id=axis.owner_public_agent_id,
                target_action_recipient_public_agent_id_by_id=(
                    axis.target_public_agent_id_by_action
                ),
                submitted_action=inspection.submitted_action,
                accepted_action=inspection.accepted_action,
            ),
        ),
        recipient_public_agent_id=axis.owner_public_agent_id,
        recipient_presentation_key=axis.owner_presentation_key,
    )


@pytest.fixture(scope="module")
def five_frames(
    _inspection_cases_fixture: _InspectionCases,  # noqa: F811
) -> _FiveFrames:
    inspection_cases = _inspection_cases_fixture
    no_shared = inspection_cases.no_shared
    raw = _build_raw_frames(
        no_shared,
        viewer_session_id="cp2-5-b-oracle",
    )[1]
    replay_oracle = build_replay_oracle_authorized_presentation_v1(
        no_shared.context,
        no_shared.frames[1],
        raw,
        source_authority_epoch=raw.revision,
        selected_internal_slot=0,
        incoming_transition=no_shared.transitions[0],
        outgoing_transition=no_shared.transitions[1],
    )

    live_draft = build_live_oracle_draft_inspection_v1(
        no_shared.context,
        no_shared.frames[1],
        replay_oracle.current_endpoint.scene,
        controlled_internal_slot=0,
        draft_move_action=0,
        draft_target_internal_slot=None,
        draft_armed_lane="none",
    )
    team_by_id = {
        row.public_agent_id: row.team_id
        for row in replay_oracle.current_endpoint.identity_directory.identities
    }
    live_oracle_axis = cast(
        OracleActionAxisV1,
        _axis_from_mask(
            live_draft.decision_mask,
            oracle_team_by_public_id=team_by_id,
        ),
    )
    live_oracle_endpoint = build_oracle_authorized_current_endpoint_v1(
        context=no_shared.context,
        source_scene=raw.projection.scene,
        authority_session_id=raw.viewer_session_id,
        selected_internal_slot=0,
    )
    assert live_oracle_endpoint.action_axis == live_oracle_axis
    live_oracle = LiveOracleAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="live_oracle",
        product_kind="combat_debugger",
        source=LiveOraclePresentationSourceIdentityV1(
            source_kind="live_oracle_frame",
            source_session_id=raw.viewer_session_id,
            source_run_generation=0,
            source_revision=1,
            source_authority_epoch=1,
            episode_id=no_shared.context.identity.episode_id,
            source_frame_index=1,
            source_frame_id=no_shared.frames[1].frame_id,
            source_simulator_step_count=no_shared.frames[1].simulator_step_count,
            source_submission_scope="joint_turn",
            source_authorized_endpoint_digest_sha256=(
                live_oracle_endpoint.authorized_endpoint_digest_sha256
            ),
        ),
        authority=replay_oracle.authority,
        analysis_mode="analysis",
        current_endpoint=live_oracle_endpoint,
        latest_events=replay_oracle.latest_events,
        latest_transition=replay_oracle.latest_transition,
        technical_frame=LiveOracleTechnicalFrameV1(
            technical_kind="live_oracle_technical_frame",
            episode_id=no_shared.context.identity.episode_id,
            evaluation_frame_index=1,
            simulator_step_count=no_shared.frames[1].simulator_step_count,
            incoming_transition_id=no_shared.transitions[0].transition_id,
        ),
        live_inspection=LiveOracleInspectionEnvelopeV1(
            envelope_kind="live_oracle_source_bound_inspection",
            source_session_id=raw.viewer_session_id,
            source_run_generation=0,
            source_revision=1,
            source_authority_epoch=1,
            episode_id=no_shared.context.identity.episode_id,
            source_frame_index=1,
            source_frame_id=no_shared.frames[1].frame_id,
            source_simulator_step_count=(no_shared.frames[1].simulator_step_count),
            inspection=LiveEditableDraftInspectionV1(
                inspection_kind="editable_live_draft",
                submission_scope="joint_turn",
                draft=live_draft,
            ),
        ),
    )

    no_shared_index = _pov_index(no_shared, actor_slot=0)
    no_shared_session = "cp2-5-b-no-shared"
    no_shared_current = _no_shared_current(
        no_shared,
        no_shared_index,
        frame_index=1,
        authority=no_shared_session,
    )
    no_shared_start = _no_shared_current(
        no_shared,
        no_shared_index,
        frame_index=0,
        authority=no_shared_session,
    )
    replay_no_shared_inspection = build_replay_no_shared_obs_inspection_v1(
        no_shared_index,
        no_shared_current,
    )
    assert replay_no_shared_inspection is not None
    no_shared_axis = cast(
        AgentPovActionAxisV1,
        _axis_from_mask(replay_no_shared_inspection.decision_mask),
    )
    no_shared_endpoint = build_no_shared_obs_authorized_current_endpoint_v1(
        parts=no_shared_current,
        axis_mapping=no_shared_index.content.axis_mapping,
    )
    no_shared_events = build_replay_no_shared_obs_incoming_summary_v1(
        no_shared_index,
        successor_frame_index=1,
        public_catalog=no_shared.context.static_mechanics_catalog,
        authority_session_id=no_shared_session,
    )
    assert no_shared_events is not None
    no_shared_visual_events = _agent_visual_events(
        no_shared,
        transition_index=0,
        transition_start_scene=no_shared_start.scene,
        successor_scene=no_shared_current.scene,
        recipient_public_agent_id=no_shared_current.recipient_public_agent_id,
        incoming_recipient_transition_id=(
            no_shared_events.incoming_recipient_transition_id
        ),
        incoming_start_recipient_frame_id=(
            no_shared_events.incoming_start_recipient_frame_id
        ),
        incoming_successor_recipient_frame_id=(
            no_shared_events.incoming_successor_recipient_frame_id
        ),
    )
    no_shared_transition = _agent_latest_transition(
        no_shared_index.content.transitions[0],
        axis=no_shared_axis,
        start_tick=no_shared.frames[0].simulator_step_count,
    )
    replay_no_shared = ReplayNoSharedObsAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="replay_no_shared_obs_agent_pov",
        product_kind="replay_viewer",
        source=ReplayNoSharedObsPresentationSourceIdentityV1(
            source_kind="replay_no_shared_obs_frame",
            source_session_id=no_shared_session,
            source_revision=1,
            source_authority_epoch=1,
            episode_id=no_shared_current.source_episode_id,
            source_frame_index=1,
            source_final_frame_index=3,
            source_recipient_public_agent_id=(
                no_shared_current.recipient_public_agent_id
            ),
            source_recipient_frame_id=no_shared_current.source_recipient_frame_id,
            source_simulator_step_count=(no_shared_current.source_simulator_step_count),
            source_observation_mode="no_shared_obs",
            source_authorized_endpoint_digest_sha256=(
                no_shared_endpoint.authorized_endpoint_digest_sha256
            ),
        ),
        authority=NoSharedObsPresentationAuthorityV1(
            authority_kind="agent_pov",
            observation_mode="no_shared_obs",
            recipient_public_agent_id=no_shared_current.recipient_public_agent_id,
            recipient_presentation_key=no_shared_current.recipient_presentation_key,
            projection_basis="recipient_own_recorded_observation",
            exact_actor_input_export_available=True,
        ),
        analysis_mode="analysis",
        current_endpoint=no_shared_endpoint,
        latest_events=no_shared_events,
        visual_events=no_shared_visual_events,
        latest_transition=no_shared_transition,
        upcoming_transition=_no_shared_upcoming_transition(
            replay_no_shared_inspection,
            axis=no_shared_endpoint.action_axis,
        ),
        researcher_space=_replay_researcher_space_at(
            no_shared,
            frame_index=1,
            selected_global_slot=0,
            session=no_shared_session,
        ),
        technical_frame=ReplayNoSharedObsTechnicalFrameV1(
            technical_kind="replay_no_shared_obs_technical_frame",
            frame_index=1,
            simulator_step_count=no_shared_current.source_simulator_step_count,
            incoming_recipient_transition_id=(
                no_shared_transition.incoming_transition_id
            ),
        ),
        replay_inspection=replay_no_shared_inspection,
    )

    live_session = raw.viewer_session_id
    live_current = _no_shared_current(
        no_shared,
        no_shared_index,
        frame_index=1,
        authority=live_session,
    )
    live_start = _no_shared_current(
        no_shared,
        no_shared_index,
        frame_index=0,
        authority=live_session,
    )
    live_slice = _pov_slice(no_shared, actor_slot=0, frame_index=1)
    live_inspection = build_live_no_shared_obs_draft_inspection_v1(
        live_slice,
        live_current,
        draft_move_action=0,
        draft_target_action=0,
        draft_armed_lane="none",
    )
    live_axis = cast(
        AgentPovActionAxisV1,
        _axis_from_mask(live_inspection.decision_mask),
    )
    live_endpoint = build_no_shared_obs_authorized_current_endpoint_v1(
        parts=live_current,
        axis_mapping=live_slice.axis_mapping,
    )
    live_events = build_replay_no_shared_obs_incoming_summary_v1(
        no_shared_index,
        successor_frame_index=1,
        public_catalog=no_shared.context.static_mechanics_catalog,
        authority_session_id=live_session,
    )
    assert live_events is not None
    live_visual_events = _agent_visual_events(
        no_shared,
        transition_index=0,
        transition_start_scene=live_start.scene,
        successor_scene=live_current.scene,
        recipient_public_agent_id=live_current.recipient_public_agent_id,
        incoming_recipient_transition_id=(live_events.incoming_recipient_transition_id),
        incoming_start_recipient_frame_id=(
            live_events.incoming_start_recipient_frame_id
        ),
        incoming_successor_recipient_frame_id=(
            live_events.incoming_successor_recipient_frame_id
        ),
    )
    live_transition = _agent_latest_transition(
        no_shared_index.content.transitions[0],
        axis=live_axis,
        start_tick=no_shared.frames[0].simulator_step_count,
    )
    live_no_shared = LiveNoSharedObsAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="live_no_shared_obs_agent_pov",
        product_kind="combat_debugger",
        source=LiveNoSharedObsPresentationSourceIdentityV1(
            source_kind="live_no_shared_obs_frame",
            source_session_id=live_session,
            source_run_generation=0,
            source_revision=1,
            source_authority_epoch=1,
            episode_id=live_current.source_episode_id,
            source_frame_index=1,
            source_recipient_public_agent_id=live_current.recipient_public_agent_id,
            source_recipient_frame_id=live_current.source_recipient_frame_id,
            source_simulator_step_count=live_current.source_simulator_step_count,
            source_submission_scope="joint_turn",
            source_authorized_endpoint_digest_sha256=(
                live_endpoint.authorized_endpoint_digest_sha256
            ),
        ),
        authority=NoSharedObsPresentationAuthorityV1(
            authority_kind="agent_pov",
            observation_mode="no_shared_obs",
            recipient_public_agent_id=live_current.recipient_public_agent_id,
            recipient_presentation_key=live_current.recipient_presentation_key,
            projection_basis="recipient_own_recorded_observation",
            exact_actor_input_export_available=True,
        ),
        analysis_mode="analysis",
        current_endpoint=live_endpoint,
        latest_events=live_events,
        visual_events=live_visual_events,
        latest_transition=live_transition,
        technical_frame=LiveNoSharedObsTechnicalFrameV1(
            technical_kind="live_no_shared_obs_technical_frame",
            episode_id=live_current.source_episode_id,
            recipient_frame_index=1,
            simulator_step_count=live_current.source_simulator_step_count,
            incoming_recipient_transition_id=live_transition.incoming_transition_id,
        ),
        live_inspection=LiveNoSharedObsInspectionEnvelopeV1(
            envelope_kind="live_no_shared_obs_source_bound_inspection",
            source_session_id=live_session,
            source_run_generation=0,
            source_revision=1,
            source_authority_epoch=1,
            episode_id=live_current.source_episode_id,
            source_frame_index=1,
            source_recipient_public_agent_id=live_current.recipient_public_agent_id,
            source_recipient_frame_id=live_current.source_recipient_frame_id,
            source_simulator_step_count=(live_current.source_simulator_step_count),
            inspection=LiveEditableDraftInspectionV1(
                inspection_kind="editable_live_draft",
                submission_scope="joint_turn",
                draft=live_inspection,
            ),
        ),
        researcher_space=build_live_researcher_space_v1(live_oracle),
    )

    shared_session = "cp2-5-b-shared"
    shared_start, _ = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=0,
        authority=shared_session,
    )
    shared_current, shared_source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
        authority=shared_session,
    )
    shared_inspection = build_replay_shared_obs_inspection_v1(
        shared_current,
        shared_source,
        authorized_recipient_global_slot=0,
        outgoing_transition=inspection_cases.shared.transitions[1],
        final_frame_index=3,
    )
    assert shared_inspection is not None
    shared_axis = cast(
        AgentPovActionAxisV1,
        _axis_from_mask(shared_inspection.decision_mask),
    )
    shared_endpoint = build_shared_obs_authorized_current_endpoint_v1(
        parts=shared_current,
        axis_mapping=shared_source.axis_mapping,
    )
    shared_events = build_shared_obs_incoming_summary_v1(
        shared_start,
        shared_current,
    )
    assert shared_events is not None
    shared_visual_events = _agent_visual_events(
        inspection_cases.shared,
        transition_index=0,
        transition_start_scene=shared_start.scene,
        successor_scene=shared_current.scene,
        recipient_public_agent_id=shared_current.recipient_public_agent_id,
        incoming_recipient_transition_id=(
            shared_events.incoming_recipient_transition_id
        ),
        incoming_start_recipient_frame_id=(
            shared_events.incoming_start_recipient_frame_id
        ),
        incoming_successor_recipient_frame_id=(
            shared_events.incoming_successor_recipient_frame_id
        ),
    )
    shared_transition = _shared_latest_transition(
        inspection_cases,
        axis=shared_axis,
        incoming_transition_index=0,
        start_tick=inspection_cases.shared.frames[0].simulator_step_count,
    )
    replay_shared = ReplaySharedObsAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="replay_shared_obs_agent_pov",
        product_kind="replay_viewer",
        source=ReplaySharedObsPresentationSourceIdentityV1(
            source_kind="replay_shared_obs_visual_union_frame",
            source_session_id=shared_session,
            source_revision=1,
            source_authority_epoch=1,
            episode_id=shared_current.source_episode_id,
            source_frame_index=1,
            source_final_frame_index=3,
            source_recipient_public_agent_id=(shared_current.recipient_public_agent_id),
            source_recipient_frame_id=shared_current.source_recipient_frame_id,
            source_simulator_step_count=shared_current.source_simulator_step_count,
            source_observation_mode="shared_obs_visual_union",
            source_authorized_endpoint_digest_sha256=(
                shared_endpoint.authorized_endpoint_digest_sha256
            ),
        ),
        authority=SharedObsPresentationAuthorityV1(
            authority_kind="agent_pov",
            observation_mode="shared_obs_visual_union",
            recipient_public_agent_id=shared_current.recipient_public_agent_id,
            recipient_presentation_key=shared_current.recipient_presentation_key,
            projection_basis="authorized_same_epoch_sensor_source_visual_union",
            exact_actor_input_export_available=False,
        ),
        analysis_mode="analysis",
        current_endpoint=shared_endpoint,
        latest_events=shared_events,
        visual_events=shared_visual_events,
        latest_transition=shared_transition,
        upcoming_transition=_shared_upcoming_transition(
            shared_inspection,
            axis=shared_endpoint.action_axis,
        ),
        researcher_space=_replay_researcher_space_at(
            inspection_cases.shared,
            frame_index=1,
            selected_global_slot=0,
            session=shared_session,
        ),
        technical_frame=ReplaySharedObsTechnicalFrameV1(
            technical_kind="replay_shared_obs_technical_frame",
            frame_index=1,
            simulator_step_count=shared_current.source_simulator_step_count,
            incoming_recipient_transition_id=(shared_transition.incoming_transition_id),
        ),
        replay_inspection=shared_inspection,
    )
    return _FiveFrames(
        cases=inspection_cases,
        live_oracle=live_oracle,
        live_no_shared=live_no_shared,
        replay_oracle=replay_oracle,
        replay_no_shared=replay_no_shared,
        replay_shared=replay_shared,
    )


def _replay_oracle_at(
    cases: _InspectionCases,
    *,
    frame_index: int,
    selected_internal_slot: int | None,
    session: str,
) -> ReplayOracleAuthorizedPresentationFrameV1:
    raw = _build_raw_frames(cases.no_shared, viewer_session_id=session)[frame_index]
    final_frame_index = len(cases.no_shared.transitions)
    return build_replay_oracle_authorized_presentation_v1(
        cases.no_shared.context,
        cases.no_shared.frames[frame_index],
        raw,
        source_authority_epoch=raw.revision,
        selected_internal_slot=selected_internal_slot,
        incoming_transition=(
            None if frame_index == 0 else cases.no_shared.transitions[frame_index - 1]
        ),
        outgoing_transition=(
            None
            if frame_index == final_frame_index
            else cases.no_shared.transitions[frame_index]
        ),
    )


def _replay_no_shared_at(
    cases: _InspectionCases,
    *,
    frame_index: int,
    session: str,
    actor_slot: int = 0,
) -> ReplayNoSharedObsAuthorizedPresentationFrameV1:
    trajectory = cases.no_shared
    index = _pov_index(trajectory, actor_slot=actor_slot)
    current = _no_shared_current(
        trajectory,
        index,
        frame_index=frame_index,
        authority=session,
    )
    endpoint = build_no_shared_obs_authorized_current_endpoint_v1(
        parts=current,
        axis_mapping=index.content.axis_mapping,
    )
    latest_events = (
        None
        if frame_index == 0
        else build_replay_no_shared_obs_incoming_summary_v1(
            index,
            successor_frame_index=frame_index,
            public_catalog=trajectory.context.static_mechanics_catalog,
            authority_session_id=session,
        )
    )
    latest_transition = (
        None
        if frame_index == 0
        else _agent_latest_transition(
            index.content.transitions[frame_index - 1],
            axis=endpoint.action_axis,
            start_tick=trajectory.frames[frame_index - 1].simulator_step_count,
        )
    )
    visual_events = None
    if frame_index > 0:
        assert latest_events is not None
        start = _no_shared_current(
            trajectory,
            index,
            frame_index=frame_index - 1,
            authority=session,
        )
        visual_events = _agent_visual_events(
            trajectory,
            transition_index=frame_index - 1,
            transition_start_scene=start.scene,
            successor_scene=current.scene,
            recipient_public_agent_id=current.recipient_public_agent_id,
            incoming_recipient_transition_id=(
                latest_events.incoming_recipient_transition_id
            ),
            incoming_start_recipient_frame_id=(
                latest_events.incoming_start_recipient_frame_id
            ),
            incoming_successor_recipient_frame_id=(
                latest_events.incoming_successor_recipient_frame_id
            ),
        )
    replay_inspection = build_replay_no_shared_obs_inspection_v1(index, current)
    return ReplayNoSharedObsAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="replay_no_shared_obs_agent_pov",
        product_kind="replay_viewer",
        source=ReplayNoSharedObsPresentationSourceIdentityV1(
            source_kind="replay_no_shared_obs_frame",
            source_session_id=session,
            source_revision=7,
            source_authority_epoch=7,
            episode_id=current.source_episode_id,
            source_frame_index=frame_index,
            source_final_frame_index=len(trajectory.transitions),
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
        upcoming_transition=_no_shared_upcoming_transition(
            replay_inspection,
            axis=endpoint.action_axis,
        ),
        researcher_space=_replay_researcher_space_at(
            trajectory,
            frame_index=frame_index,
            selected_global_slot=actor_slot,
            session=session,
        ),
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


def _replay_shared_at(
    cases: _InspectionCases,
    *,
    frame_index: int,
    session: str,
    recipient_slot: int = 0,
) -> ReplaySharedObsAuthorizedPresentationFrameV1:
    trajectory = cases.shared
    current, source_material = _shared_current(
        trajectory,
        recipient_slot=recipient_slot,
        frame_index=frame_index,
        authority=session,
    )
    endpoint = build_shared_obs_authorized_current_endpoint_v1(
        parts=current,
        axis_mapping=source_material.axis_mapping,
    )
    latest_events = None
    latest_transition = None
    visual_events = None
    if frame_index > 0:
        start, _ = _shared_current(
            trajectory,
            recipient_slot=recipient_slot,
            frame_index=frame_index - 1,
            authority=session,
        )
        latest_events = build_shared_obs_incoming_summary_v1(start, current)
        assert latest_events is not None
        visual_events = _agent_visual_events(
            trajectory,
            transition_index=frame_index - 1,
            transition_start_scene=start.scene,
            successor_scene=current.scene,
            recipient_public_agent_id=current.recipient_public_agent_id,
            incoming_recipient_transition_id=(
                latest_events.incoming_recipient_transition_id
            ),
            incoming_start_recipient_frame_id=(
                latest_events.incoming_start_recipient_frame_id
            ),
            incoming_successor_recipient_frame_id=(
                latest_events.incoming_successor_recipient_frame_id
            ),
        )
        latest_transition = _shared_latest_transition(
            cases,
            axis=endpoint.action_axis,
            incoming_transition_index=frame_index - 1,
            start_tick=trajectory.frames[frame_index - 1].simulator_step_count,
        )
    final_frame_index = len(trajectory.transitions)
    replay_inspection = build_replay_shared_obs_inspection_v1(
        current,
        source_material,
        authorized_recipient_global_slot=recipient_slot,
        outgoing_transition=(
            None
            if frame_index == final_frame_index
            else trajectory.transitions[frame_index]
        ),
        final_frame_index=final_frame_index,
    )
    return ReplaySharedObsAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="replay_shared_obs_agent_pov",
        product_kind="replay_viewer",
        source=ReplaySharedObsPresentationSourceIdentityV1(
            source_kind="replay_shared_obs_visual_union_frame",
            source_session_id=session,
            source_revision=7,
            source_authority_epoch=7,
            episode_id=current.source_episode_id,
            source_frame_index=frame_index,
            source_final_frame_index=final_frame_index,
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
        upcoming_transition=_shared_upcoming_transition(
            replay_inspection,
            axis=endpoint.action_axis,
        ),
        researcher_space=_replay_researcher_space_at(
            trajectory,
            frame_index=frame_index,
            selected_global_slot=recipient_slot,
            session=session,
        ),
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


def _live_oracle_at(
    cases: _InspectionCases,
    *,
    frame_index: int,
    session: str,
) -> LiveOracleAuthorizedPresentationFrameV1:
    trajectory = cases.no_shared
    raw = _build_raw_frames(trajectory, viewer_session_id=session)[frame_index]
    replay_projection = build_replay_oracle_authorized_presentation_v1(
        trajectory.context,
        trajectory.frames[frame_index],
        raw,
        source_authority_epoch=raw.revision,
        selected_internal_slot=0,
        incoming_transition=(
            None if frame_index == 0 else trajectory.transitions[frame_index - 1]
        ),
        outgoing_transition=trajectory.transitions[frame_index],
    )
    draft = build_live_oracle_draft_inspection_v1(
        trajectory.context,
        trajectory.frames[frame_index],
        replay_projection.current_endpoint.scene,
        controlled_internal_slot=0,
        draft_move_action=0,
        draft_target_internal_slot=None,
        draft_armed_lane="none",
    )
    endpoint = build_oracle_authorized_current_endpoint_v1(
        context=trajectory.context,
        source_scene=raw.projection.scene,
        authority_session_id=session,
        selected_internal_slot=0,
    )
    incoming_id = (
        None
        if replay_projection.latest_transition is None
        else replay_projection.latest_transition.incoming_transition_id
    )
    return LiveOracleAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="live_oracle",
        product_kind="combat_debugger",
        source=LiveOraclePresentationSourceIdentityV1(
            source_kind="live_oracle_frame",
            source_session_id=session,
            source_run_generation=0,
            source_revision=frame_index,
            source_authority_epoch=frame_index,
            episode_id=trajectory.context.identity.episode_id,
            source_frame_index=frame_index,
            source_frame_id=trajectory.frames[frame_index].frame_id,
            source_simulator_step_count=(
                trajectory.frames[frame_index].simulator_step_count
            ),
            source_submission_scope="joint_turn",
            source_authorized_endpoint_digest_sha256=(
                endpoint.authorized_endpoint_digest_sha256
            ),
        ),
        authority=replay_projection.authority,
        analysis_mode="analysis",
        current_endpoint=endpoint,
        latest_events=replay_projection.latest_events,
        latest_transition=replay_projection.latest_transition,
        technical_frame=LiveOracleTechnicalFrameV1(
            technical_kind="live_oracle_technical_frame",
            episode_id=trajectory.context.identity.episode_id,
            evaluation_frame_index=frame_index,
            simulator_step_count=trajectory.frames[frame_index].simulator_step_count,
            incoming_transition_id=incoming_id,
        ),
        live_inspection=LiveOracleInspectionEnvelopeV1(
            envelope_kind="live_oracle_source_bound_inspection",
            source_session_id=session,
            source_run_generation=0,
            source_revision=frame_index,
            source_authority_epoch=frame_index,
            episode_id=trajectory.context.identity.episode_id,
            source_frame_index=frame_index,
            source_frame_id=trajectory.frames[frame_index].frame_id,
            source_simulator_step_count=(
                trajectory.frames[frame_index].simulator_step_count
            ),
            inspection=LiveEditableDraftInspectionV1(
                inspection_kind="editable_live_draft",
                submission_scope="joint_turn",
                draft=draft,
            ),
        ),
    )


def _live_no_shared_at(
    cases: _InspectionCases,
    *,
    frame_index: int,
    session: str,
) -> LiveNoSharedObsAuthorizedPresentationFrameV1:
    trajectory = cases.no_shared
    index = _pov_index(trajectory, actor_slot=0)
    current = _no_shared_current(
        trajectory,
        index,
        frame_index=frame_index,
        authority=session,
    )
    source_slice = _pov_slice(trajectory, actor_slot=0, frame_index=frame_index)
    draft = build_live_no_shared_obs_draft_inspection_v1(
        source_slice,
        current,
        draft_move_action=0,
        draft_target_action=0,
        draft_armed_lane="none",
    )
    endpoint = build_no_shared_obs_authorized_current_endpoint_v1(
        parts=current,
        axis_mapping=source_slice.axis_mapping,
    )
    latest_events = build_replay_no_shared_obs_incoming_summary_v1(
        index,
        successor_frame_index=frame_index,
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id=session,
    )
    latest_transition = (
        None
        if frame_index == 0
        else _agent_latest_transition(
            index.content.transitions[frame_index - 1],
            axis=endpoint.action_axis,
            start_tick=trajectory.frames[frame_index - 1].simulator_step_count,
        )
    )
    incoming_id = (
        None if latest_transition is None else latest_transition.incoming_transition_id
    )
    visual_events = None
    if frame_index > 0:
        assert latest_events is not None
        start = _no_shared_current(
            trajectory,
            index,
            frame_index=frame_index - 1,
            authority=session,
        )
        visual_events = _agent_visual_events(
            trajectory,
            transition_index=frame_index - 1,
            transition_start_scene=start.scene,
            successor_scene=current.scene,
            recipient_public_agent_id=current.recipient_public_agent_id,
            incoming_recipient_transition_id=(
                latest_events.incoming_recipient_transition_id
            ),
            incoming_start_recipient_frame_id=(
                latest_events.incoming_start_recipient_frame_id
            ),
            incoming_successor_recipient_frame_id=(
                latest_events.incoming_successor_recipient_frame_id
            ),
        )
    return LiveNoSharedObsAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="live_no_shared_obs_agent_pov",
        product_kind="combat_debugger",
        source=LiveNoSharedObsPresentationSourceIdentityV1(
            source_kind="live_no_shared_obs_frame",
            source_session_id=session,
            source_run_generation=0,
            source_revision=frame_index,
            source_authority_epoch=frame_index,
            episode_id=current.source_episode_id,
            source_frame_index=frame_index,
            source_recipient_public_agent_id=current.recipient_public_agent_id,
            source_recipient_frame_id=current.source_recipient_frame_id,
            source_simulator_step_count=current.source_simulator_step_count,
            source_submission_scope="joint_turn",
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
        technical_frame=LiveNoSharedObsTechnicalFrameV1(
            technical_kind="live_no_shared_obs_technical_frame",
            episode_id=current.source_episode_id,
            recipient_frame_index=frame_index,
            simulator_step_count=current.source_simulator_step_count,
            incoming_recipient_transition_id=incoming_id,
        ),
        live_inspection=LiveNoSharedObsInspectionEnvelopeV1(
            envelope_kind="live_no_shared_obs_source_bound_inspection",
            source_session_id=session,
            source_run_generation=0,
            source_revision=frame_index,
            source_authority_epoch=frame_index,
            episode_id=current.source_episode_id,
            source_frame_index=frame_index,
            source_recipient_public_agent_id=current.recipient_public_agent_id,
            source_recipient_frame_id=current.source_recipient_frame_id,
            source_simulator_step_count=current.source_simulator_step_count,
            inspection=LiveEditableDraftInspectionV1(
                inspection_kind="editable_live_draft",
                submission_scope="joint_turn",
                draft=draft,
            ),
        ),
        researcher_space=build_live_researcher_space_v1(
            _live_oracle_at(
                cases,
                frame_index=frame_index,
                session=session,
            )
        ),
    )


def test_union_has_exactly_five_discriminated_leaves() -> None:
    schema = TypeAdapter(AuthorizedPresentationFrameV1).json_schema()
    assert len(cast(list[object], schema["oneOf"])) == 5
    mapping = cast(dict[str, object], schema["discriminator"])["mapping"]
    assert set(cast(dict[str, str], mapping)) == {
        "live_oracle",
        "live_no_shared_obs_agent_pov",
        "replay_oracle",
        "replay_no_shared_obs_agent_pov",
        "replay_shared_obs_agent_pov",
    }
    assert not any("live_shared" in key for key in cast(dict[str, str], mapping))


def test_recursive_schema_is_closed_required_and_key_catalog_is_exhaustive() -> None:
    schema = TypeAdapter(AuthorizedPresentationFrameV1).json_schema()
    definitions = cast(dict[str, dict[str, object]], schema["$defs"])
    key_pairs = {
        "presentation_key": "public_agent_id",
        "source_presentation_key": "source_public_agent_id",
        "agent_presentation_key": "agent_public_agent_id",
        "recipient_presentation_key": "recipient_public_agent_id",
        "owner_presentation_key": "owner_public_agent_id",
        "target_presentation_key": "target_public_agent_id",
        "assigned_presentation_key": "assigned_public_agent_id",
        "actor_presentation_key": "actor_public_agent_id",
    }
    encountered_key_fields: set[str] = set()
    one_of_count = 0

    def visit(value: object, *, path: str) -> None:
        nonlocal one_of_count
        if type(value) is list:
            for index, item in enumerate(cast(list[object], value)):
                visit(item, path=f"{path}[{index}]")
            return
        if type(value) is not dict:
            return
        node = cast(dict[str, object], value)
        properties = node.get("properties")
        if type(properties) is dict:
            property_names = set(cast(dict[str, object], properties))
            assert set(cast(list[str], node.get("required", []))) == property_names, (
                path
            )
            assert node.get("additionalProperties") is False, path
            for name in property_names:
                if name == "presentation_key" or name.endswith("_presentation_key"):
                    encountered_key_fields.add(name)
                    assert name in key_pairs, path
                    assert key_pairs[name] in property_names, path
        if "oneOf" in node:
            one_of_count += 1
            assert "discriminator" in node, path
        for name, nested in node.items():
            visit(nested, path=f"{path}.{name}")

    visit(schema, path="root")
    assert encountered_key_fields == set(key_pairs)
    assert one_of_count > 1
    assert len(definitions) >= 100


def test_live_researcher_latest_rejects_reordered_target_identity_axis(
    five_frames: _FiveFrames,
) -> None:
    payload = json.loads(five_frames.live_no_shared.model_dump_json())
    target_axis = payload["researcher_space"]["latest_transition"]["action_rows"][0][
        "target_action_recipient_public_agent_id_by_id"
    ]
    target_axis[1], target_axis[2] = target_axis[2], target_axis[1]

    with pytest.raises(
        ValidationError,
        match="live researcher Latest action identity changed",
    ):
        LiveNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


def test_live_researcher_rejects_changed_fog_authorized_class_mechanics(
    five_frames: _FiveFrames,
) -> None:
    payload = json.loads(five_frames.live_no_shared.model_dump_json())
    local_class_id = payload["current_endpoint"]["parts"]["scene"]["class_mechanics"][
        0
    ]["class_id"]
    global_class = next(
        row
        for row in payload["researcher_space"]["class_mechanics"]
        if row["class_id"] == local_class_id
    )
    global_class["basic_raw_damage"] += 1.0

    with pytest.raises(
        ValidationError,
        match="live researcher class mechanics changed a fog-authorized class",
    ):
        LiveNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


def test_live_and_replay_researcher_spaces_reject_hidden_standalone_fact_poison(
    five_frames: _FiveFrames,
) -> None:
    cases = (
        (
            "live NoSharedObs",
            five_frames.live_no_shared,
            LiveNoSharedObsAuthorizedPresentationFrameV1,
        ),
        (
            "replay NoSharedObs",
            five_frames.replay_no_shared,
            ReplayNoSharedObsAuthorizedPresentationFrameV1,
        ),
        (
            "replay SharedObs",
            five_frames.replay_shared,
            ReplaySharedObsAuthorizedPresentationFrameV1,
        ),
    )
    for leaf_name, frame, model in cases:
        payload = json.loads(frame.model_dump_json())
        local_public_ids = {
            row["public_agent_id"]
            for row in payload["current_endpoint"]["parts"]["scene"]["agents"]
        }
        hidden_index = next(
            index
            for index, row in enumerate(payload["researcher_space"]["roster_agents"])
            if row["public_agent_id"] not in local_public_ids
        )
        hidden_class_id = payload["researcher_space"]["roster_agents"][hidden_index][
            "class_id"
        ]
        status_template = next(
            status
            for row in payload["researcher_space"]["roster_agents"]
            for status in row["statuses"]
        )
        aura_template = next(
            aura
            for row in payload["researcher_space"]["roster_agents"]
            for aura in row["aura_modifiers"]
        )
        poisoned: list[tuple[str, object]] = []

        health = copy.deepcopy(payload)
        hidden = health["researcher_space"]["roster_agents"][hidden_index]
        hidden["current_health"] = hidden["maximum_health"] + 1.0
        poisoned.append(("health above maximum", health))

        maximum = copy.deepcopy(payload)
        maximum["researcher_space"]["roster_agents"][hidden_index][
            "maximum_health"
        ] += 1.0
        poisoned.append(("maximum health outside class mechanics", maximum))

        duplicate_status = copy.deepcopy(payload)
        duplicate_status["researcher_space"]["roster_agents"][hidden_index][
            "statuses"
        ].extend((copy.deepcopy(status_template), copy.deepcopy(status_template)))
        poisoned.append(("duplicate hidden status channel", duplicate_status))

        wrong_status_family = copy.deepcopy(payload)
        status = copy.deepcopy(status_template)
        status["family"] = "stun" if status["family"] == "slow" else "slow"
        wrong_status_family["researcher_space"]["roster_agents"][hidden_index][
            "statuses"
        ].append(status)
        poisoned.append(
            ("hidden status outside its catalog family", wrong_status_family)
        )

        duplicate_aura = copy.deepcopy(payload)
        duplicate_aura["researcher_space"]["roster_agents"][hidden_index][
            "aura_modifiers"
        ].extend((copy.deepcopy(aura_template), copy.deepcopy(aura_template)))
        poisoned.append(("duplicate hidden aura", duplicate_aura))

        wrong_class_name = copy.deepcopy(payload)
        next(
            row
            for row in wrong_class_name["researcher_space"]["class_mechanics"]
            if row["class_id"] == hidden_class_id
        )["class_name"] = "Wrong Class"
        poisoned.append(("hidden class name outside the catalog", wrong_class_name))

        mixed_versions = copy.deepcopy(payload)
        hidden_mechanics = next(
            row
            for row in mixed_versions["researcher_space"]["class_mechanics"]
            if row["class_id"] == hidden_class_id
        )
        del hidden_mechanics["mechanics_version"]
        del hidden_mechanics["documentation_profile"]
        poisoned.append(("mixed hidden V1 and V2 mechanics", mixed_versions))

        wrong_status_owner = copy.deepcopy(payload)
        mechanics = wrong_status_owner["researcher_space"]["class_mechanics"]
        hidden_mechanics = next(
            row for row in mechanics if row["class_id"] == hidden_class_id
        )
        hidden_channels = {
            row["status_channel"] for row in hidden_mechanics["status_mechanics"]
        }
        foreign_status = next(
            status
            for row in mechanics
            if row["class_id"] != hidden_class_id
            for status in row["status_mechanics"]
            if status["status_channel"] not in hidden_channels
        )
        status_mechanics = cast(
            list[dict[str, object]],
            hidden_mechanics["status_mechanics"],
        )
        status_mechanics.append(copy.deepcopy(foreign_status))
        status_mechanics.sort(
            key=lambda row: cast(int, row["status_channel"]),
        )
        poisoned.append(("hidden class with a foreign status", wrong_status_owner))

        wrong_aura_owner = copy.deepcopy(payload)
        mechanics = wrong_aura_owner["researcher_space"]["class_mechanics"]
        hidden_mechanics = next(
            row for row in mechanics if row["class_id"] == hidden_class_id
        )
        foreign_aura = next(
            aura
            for row in mechanics
            if row["class_id"] != hidden_class_id
            for aura in row["aura_mechanics"]
        )
        hidden_mechanics["aura_mechanics"].append(copy.deepcopy(foreign_aura))
        poisoned.append(("hidden class with a foreign aura", wrong_aura_owner))

        for mutation_name, candidate in poisoned:
            try:
                model.model_validate_json(json.dumps(candidate))
            except ValidationError:
                pass
            else:
                pytest.fail(f"{leaf_name}: {mutation_name} unexpectedly passed")


def test_agent_endpoint_factories_derive_exact_axis_from_accepted_mapping(
    five_frames: _FiveFrames,
) -> None:
    trajectory = five_frames.cases.no_shared
    index = _pov_index(trajectory, actor_slot=0)
    current = _no_shared_current(
        trajectory,
        index,
        frame_index=1,
        authority="cp2-5-b-axis-factory",
    )
    mapping = index.content.axis_mapping
    mapping_before = TypeAdapter(ActorPovAxisMappingV1).dump_json(mapping)
    parts_before = TypeAdapter(type(current)).dump_json(current)
    endpoint = build_no_shared_obs_authorized_current_endpoint_v1(
        parts=current,
        axis_mapping=mapping,
    )
    assert tuple(row.display_name for row in endpoint.action_axis.movement_actions) == (
        mapping.movement_action_name_by_id
    )
    assert endpoint.action_axis.target_public_agent_id_by_action == (
        mapping.target_action_recipient_public_agent_id_by_id
    )
    assert tuple(row.display_name for row in endpoint.action_axis.target_actions) == (
        mapping.target_action_name_by_id
    )
    assert tuple(row.display_name for row in endpoint.action_axis.ultimate_choices) == (
        mapping.use_ultimate_action_name_by_id
    )
    assert (
        tuple(
            cast(TargetAgentActionDisplayRowV1, row).target_relation
            for row in endpoint.action_axis.target_actions[1:]
        )
        == ("same_team",) * 5 + ("opponent",) * 5
    )
    assert TypeAdapter(ActorPovAxisMappingV1).dump_json(mapping) == mapping_before
    assert TypeAdapter(type(current)).dump_json(current) == parts_before

    visible_ids = {row.public_agent_id for row in current.scene.agents}
    hidden_action = next(
        action
        for action, public_id in enumerate(
            mapping.target_action_recipient_public_agent_id_by_id
        )
        if action > 0 and public_id not in visible_ids
    )
    forged_id = "forged-hidden-axis-agent"
    forged_payload = mapping.model_dump(mode="python")
    forged_target_axis = list(mapping.target_action_recipient_public_agent_id_by_id)
    forged_target_axis[hidden_action] = forged_id
    forged_payload["target_action_recipient_public_agent_id_by_id"] = tuple(
        forged_target_axis
    )
    relation_field = (
        "ally_observation_row_public_agent_id_by_id"
        if hidden_action <= 5
        else "enemy_observation_row_public_agent_id_by_id"
    )
    relation_axis = list(cast(tuple[str, ...], forged_payload[relation_field]))
    relation_axis[(hidden_action - 1) % 5] = forged_id
    forged_payload[relation_field] = tuple(relation_axis)
    forged_mapping = ActorPovAxisMappingV1.model_validate(forged_payload)
    forged_endpoint = build_no_shared_obs_authorized_current_endpoint_v1(
        parts=current,
        axis_mapping=forged_mapping,
    )
    assert forged_endpoint.action_axis.target_public_agent_id_by_action[
        hidden_action
    ] == (forged_id)
    assert endpoint.action_axis.target_public_agent_id_by_action[hidden_action] != (
        forged_id
    )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cast(
            Callable[..., object],
            build_no_shared_obs_authorized_current_endpoint_v1,
        )(
            parts=current,
            action_axis=forged_endpoint.action_axis,
        )

    shared_current, shared_source = _shared_current(
        five_frames.cases.shared,
        recipient_slot=0,
        frame_index=1,
        authority="cp2-5-b-shared-axis-factory",
    )
    shared_mapping_before = TypeAdapter(ActorPovAxisMappingV1).dump_json(
        shared_source.axis_mapping
    )
    shared_endpoint = build_shared_obs_authorized_current_endpoint_v1(
        parts=shared_current,
        axis_mapping=shared_source.axis_mapping,
    )
    assert shared_endpoint.action_axis.target_public_agent_id_by_action == (
        shared_source.axis_mapping.target_action_recipient_public_agent_id_by_id
    )
    assert (
        TypeAdapter(ActorPovAxisMappingV1).dump_json(shared_source.axis_mapping)
        == shared_mapping_before
    )


def test_oracle_scene_wrapper_and_endpoint_factory_own_authority_inputs(
    five_frames: _FiveFrames,
) -> None:
    trajectory = five_frames.cases.no_shared
    session_a = "cp2-5-b-oracle-wrapper-a"
    session_b = "cp2-5-b-oracle-wrapper-b"
    raw_frames = _build_raw_frames(trajectory, viewer_session_id=session_a)
    context_bytes = trajectory.context.model_dump_json()

    for frame_index in (0, 1, len(raw_frames) - 1):
        raw = raw_frames[frame_index]
        source_scene = raw.projection.scene
        scene_bytes = TypeAdapter(BattlefieldSceneV2).dump_json(source_scene)
        wrapped = build_oracle_authorized_scene_v1(
            trajectory.context,
            source_scene,
            authority_session_id=session_a,
        )
        existing_parts = build_replay_oracle_presentation_parts_v1(
            trajectory.context,
            source_scene,
            raw.projection.incoming_events,
            authority_session_id=session_a,
            final_frame_index=len(raw_frames) - 1,
            selected_internal_slot=None,
            outgoing_transition=None,
        )
        assert wrapped == existing_parts.current_scene
        endpoint = build_oracle_authorized_current_endpoint_v1(
            context=trajectory.context,
            source_scene=source_scene,
            authority_session_id=session_a,
            selected_internal_slot=0,
        )
        assert endpoint.scene == wrapped
        assert endpoint.episode_id == source_scene.episode_id
        assert endpoint.frame_index == source_scene.frame_index
        assert endpoint.frame_id == source_scene.frame_id
        assert endpoint.simulator_step_count == source_scene.simulator_step_count
        assert trajectory.context.model_dump_json() == context_bytes
        assert TypeAdapter(BattlefieldSceneV2).dump_json(source_scene) == scene_bytes

    source_scene = raw_frames[1].projection.scene
    namespace_a = build_oracle_authorized_scene_v1(
        trajectory.context,
        source_scene,
        authority_session_id=session_a,
    )
    namespace_b = build_oracle_authorized_scene_v1(
        trajectory.context,
        source_scene,
        authority_session_id=session_b,
    )
    assert tuple(row.public_agent_id for row in namespace_a.agents) == tuple(
        row.public_agent_id for row in namespace_b.agents
    )
    assert tuple(row.presentation_key for row in namespace_a.agents) != tuple(
        row.presentation_key for row in namespace_b.agents
    )
    for selected_internal_slot in (None, 0):
        endpoint_a = build_oracle_authorized_current_endpoint_v1(
            context=trajectory.context,
            source_scene=source_scene,
            authority_session_id=session_a,
            selected_internal_slot=selected_internal_slot,
        )
        endpoint_b = build_oracle_authorized_current_endpoint_v1(
            context=trajectory.context,
            source_scene=source_scene,
            authority_session_id=session_b,
            selected_internal_slot=selected_internal_slot,
        )
        assert endpoint_a.authorized_endpoint_digest_sha256 != (
            endpoint_b.authorized_endpoint_digest_sha256
        )

    with pytest.raises(TypeError, match="exact BattlefieldSceneV2"):
        build_oracle_authorized_scene_v1(
            trajectory.context,
            _PoisonBattlefieldSceneV2(
                **{
                    field.name: getattr(source_scene, field.name)
                    for field in fields(source_scene)
                }
            ),
            authority_session_id=session_a,
        )
    with pytest.raises(TypeError, match="exact EvaluationEpisodeContextV1"):
        build_oracle_authorized_scene_v1(
            _PoisonEvaluationEpisodeContextV1.model_construct(
                **{
                    name: getattr(trajectory.context, name)
                    for name in type(trajectory.context).model_fields
                }
            ),
            source_scene,
            authority_session_id=session_a,
        )

    mismatched_context = trajectory.context.model_copy(
        update={
            "identity": trajectory.context.identity.model_copy(
                update={"episode_id": "different-episode"}
            )
        }
    )
    with pytest.raises(ValueError, match="join one episode"):
        build_oracle_authorized_scene_v1(
            mismatched_context,
            source_scene,
            authority_session_id=session_a,
        )

    root_list_poison = copy.copy(source_scene)
    object.__setattr__(
        root_list_poison,
        "observer_visibility",
        list(source_scene.observer_visibility),
    )
    with pytest.raises(ValueError, match="exact runtime wire types"):
        build_oracle_authorized_scene_v1(
            trajectory.context,
            root_list_poison,
            authority_session_id=session_a,
        )

    nested_subclass = _PoisonAgentSceneV2(
        **{
            field.name: getattr(source_scene.agents[0], field.name)
            for field in fields(source_scene.agents[0])
        }
    )
    nested_subclass_poison = copy.copy(source_scene)
    object.__setattr__(
        nested_subclass_poison,
        "agents",
        (nested_subclass, *source_scene.agents[1:]),
    )
    with pytest.raises(ValueError, match="exact runtime wire types"):
        build_oracle_authorized_scene_v1(
            trajectory.context,
            nested_subclass_poison,
            authority_session_id=session_a,
        )

    nested_invalid = copy.copy(source_scene)
    invalid_map = copy.copy(source_scene.map)
    object.__setattr__(invalid_map, "width", -1.0)
    object.__setattr__(nested_invalid, "map", invalid_map)
    with pytest.raises((ValueError, ValidationError)):
        build_oracle_authorized_scene_v1(
            trajectory.context,
            nested_invalid,
            authority_session_id=session_a,
        )

    invalid_context_values = {
        name: getattr(trajectory.context, name)
        for name in type(trajectory.context).model_fields
    }
    invalid_context_values["execution_information_mode"] = "invalid-mode"
    invalid_context = EvaluationEpisodeContextV1.model_construct(
        **invalid_context_values
    )
    with pytest.raises(ValidationError):
        build_oracle_authorized_scene_v1(
            invalid_context,
            source_scene,
            authority_session_id=session_a,
        )

    hidden_extra_context = trajectory.context.model_copy(deep=True)
    object.__setattr__(hidden_extra_context, "hidden_extra", "not-in-schema")
    with pytest.raises(ValueError, match="exact runtime wire types"):
        build_oracle_authorized_scene_v1(
            hidden_extra_context,
            source_scene,
            authority_session_id=session_a,
        )

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cast(Callable[..., object], build_oracle_authorized_current_endpoint_v1)(
            context=trajectory.context,
            source_scene=source_scene,
            authority_session_id=session_a,
            selected_internal_slot=0,
            identity_directory=five_frames.replay_oracle.current_endpoint.identity_directory,
            action_axis=five_frames.replay_oracle.current_endpoint.action_axis,
        )


def test_all_five_leaves_strictly_round_trip_and_result_revalidates(
    five_frames: _FiveFrames,
) -> None:
    adapter: TypeAdapter[AuthorizedPresentationFrameV1] = TypeAdapter(
        AuthorizedPresentationFrameV1
    )
    for frame in five_frames.rows:
        encoded = adapter.dump_json(frame)
        reparsed = adapter.validate_json(encoded)
        assert type(reparsed) is type(frame)
        assert adapter.dump_json(reparsed) == encoded
        result = PresentationResourceResultV1(outcome="response", payload=frame)
        assert type(result.payload) is type(frame)
        assert (
            adapter.dump_json(cast(AuthorizedPresentationFrameV1, result.payload))
            == encoded
        )


def test_replay_oracle_selected_axis_state_matrix(
    five_frames: _FiveFrames,
) -> None:
    trajectory = five_frames.cases.no_shared
    raw_frames = _build_raw_frames(
        trajectory,
        viewer_session_id="cp2-5-b-state-matrix",
    )
    final_index = len(raw_frames) - 1
    selected_axes: dict[int, OracleActionAxisV1] = {}
    for frame_index in (0, 1, final_index):
        for selected_internal_slot in (None, 0):
            raw = raw_frames[frame_index]
            frame = build_replay_oracle_authorized_presentation_v1(
                trajectory.context,
                trajectory.frames[frame_index],
                raw,
                source_authority_epoch=raw.revision,
                selected_internal_slot=selected_internal_slot,
                incoming_transition=(
                    None
                    if frame_index == 0
                    else trajectory.transitions[frame_index - 1]
                ),
                outgoing_transition=(
                    None
                    if frame_index == final_index
                    else trajectory.transitions[frame_index]
                ),
            )
            if frame_index == final_index:
                assert frame.upcoming_transition is None
            else:
                upcoming = frame.upcoming_transition
                assert upcoming is not None
                assert upcoming.outgoing_transition_index == frame_index
                assert len(upcoming.action_rows) == sum(
                    roster.configured_active for roster in trajectory.context.roster
                )
            if selected_internal_slot is None:
                assert frame.current_endpoint.action_axis is None
                assert frame.replay_inspection is None
                continue
            axis = frame.current_endpoint.action_axis
            assert axis is not None
            selected_axes[frame_index] = axis
            assert (
                axis.owner_public_agent_id
                == trajectory.context.roster[0].public_agent_id
            )
            if frame_index == final_index:
                assert frame.replay_inspection is None
            else:
                assert frame.replay_inspection is not None
                assert frame.upcoming_transition is not None
                selected_upcoming = next(
                    row
                    for row in frame.upcoming_transition.action_rows
                    if row.actor_public_agent_id == axis.owner_public_agent_id
                )
                assert selected_upcoming.submitted_action == (
                    frame.replay_inspection.submitted_action
                )
                assert selected_upcoming.accepted_action == (
                    frame.replay_inspection.accepted_action
                )
                assert (
                    frame.replay_inspection.decision_mask.owner_public_agent_id
                    == axis.owner_public_agent_id
                )
            axis_payload = axis.model_dump(mode="json")
            assert not {
                "submitted_action",
                "accepted_action",
                "transition_reference",
                "movement_action_mask",
                "target_action_mask",
            } & set(axis_payload)
    assert selected_axes[0].model_dump(mode="json") == selected_axes[
        final_index
    ].model_dump(mode="json")

    final_raw = raw_frames[final_index]
    with pytest.raises(ValueError, match="final Oracle frames cannot receive T_n"):
        build_replay_oracle_authorized_presentation_v1(
            trajectory.context,
            trajectory.frames[final_index],
            final_raw,
            source_authority_epoch=final_raw.revision,
            selected_internal_slot=0,
            incoming_transition=trajectory.transitions[final_index - 1],
            outgoing_transition=trajectory.transitions[0],
        )


@pytest.mark.parametrize(
    ("builder", "frame_type"),
    (
        (
            _replay_no_shared_at,
            ReplayNoSharedObsAuthorizedPresentationFrameV1,
        ),
        (
            _replay_shared_at,
            ReplaySharedObsAuthorizedPresentationFrameV1,
        ),
    ),
)
def test_replay_agent_actual_frame_zero_middle_final_state_matrix(
    five_frames: _FiveFrames,
    builder: Callable[..., object],
    frame_type: type[
        ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1
    ],
) -> None:
    final_frame_index = len(five_frames.cases.no_shared.transitions)
    for frame_index in (0, 1, final_frame_index):
        frame = cast(
            ReplayNoSharedObsAuthorizedPresentationFrameV1
            | ReplaySharedObsAuthorizedPresentationFrameV1,
            builder(
                five_frames.cases,
                frame_index=frame_index,
                session=f"cp2-5-b-actual-{frame_type.__name__}",
            ),
        )
        assert type(frame) is frame_type
        assert frame.source.source_frame_index == frame_index
        assert frame.current_endpoint.parts.source_frame_index == frame_index
        assert frame.current_endpoint.action_axis.owner_public_agent_id == (
            frame.source.source_recipient_public_agent_id
        )
        if frame_index == 0:
            assert frame.latest_events is None
            assert frame.latest_transition is None
        else:
            assert frame.latest_events is not None
            assert frame.latest_transition is not None
            assert frame.latest_transition.incoming_transition_index == frame_index - 1
        if frame_index == final_frame_index:
            assert frame.replay_inspection is None
            assert frame.upcoming_transition is None
        else:
            assert frame.replay_inspection is not None
            assert frame.replay_inspection.outgoing_transition_index == frame_index
            assert frame.upcoming_transition is not None
            assert frame.upcoming_transition.outgoing_transition_index == frame_index
            assert len(frame.upcoming_transition.action_rows) == 1
            assert frame.upcoming_transition.action_rows[0].submitted_action == (
                frame.replay_inspection.submitted_action
            )
            assert frame.upcoming_transition.action_rows[0].accepted_action == (
                frame.replay_inspection.accepted_action
            )
        encoded = frame.model_dump_json()
        assert frame_type.model_validate_json(encoded) == frame


@pytest.mark.parametrize(
    ("builder", "frame_type", "session", "local_identity"),
    (
        (
            _live_oracle_at,
            LiveOracleAuthorizedPresentationFrameV1,
            "cp2-5-b-live-oracle-state",
            False,
        ),
        (
            _live_no_shared_at,
            LiveNoSharedObsAuthorizedPresentationFrameV1,
            "cp2-5-b-live-agent-state",
            True,
        ),
    ),
)
def test_live_actual_frame_zero_and_nonzero_state_matrix(
    five_frames: _FiveFrames,
    builder: Callable[..., object],
    frame_type: type[
        LiveOracleAuthorizedPresentationFrameV1
        | LiveNoSharedObsAuthorizedPresentationFrameV1
    ],
    session: str,
    local_identity: bool,
) -> None:
    built_frames: list[
        LiveOracleAuthorizedPresentationFrameV1
        | LiveNoSharedObsAuthorizedPresentationFrameV1
    ] = []
    for frame_index in (0, 1):
        frame = cast(
            LiveOracleAuthorizedPresentationFrameV1
            | LiveNoSharedObsAuthorizedPresentationFrameV1,
            builder(
                five_frames.cases,
                frame_index=frame_index,
                session=session,
            ),
        )
        repeated = cast(
            LiveOracleAuthorizedPresentationFrameV1
            | LiveNoSharedObsAuthorizedPresentationFrameV1,
            builder(
                five_frames.cases,
                frame_index=frame_index,
                session=session,
            ),
        )
        encoded = frame.model_dump_json()
        assert repeated.model_dump_json() == encoded
        assert frame_type.model_validate_json(encoded).model_dump_json() == encoded
        assert frame.source.source_frame_index == frame_index
        assert frame.current_endpoint.authorized_endpoint_digest_sha256 == (
            frame.source.source_authorized_endpoint_digest_sha256
        )
        assert frame.current_endpoint.action_axis is not None
        assert frame.live_inspection is not None
        assert frame.live_inspection.source_frame_index == frame_index
        assert frame.live_inspection.source_simulator_step_count == (
            frame.source.source_simulator_step_count
        )
        if local_identity:
            assert isinstance(frame, LiveNoSharedObsAuthorizedPresentationFrameV1)
            assert frame.source.source_recipient_frame_id.endswith(
                f":frame:{frame_index}"
            )
            assert frame.current_endpoint.parts.source_recipient_frame_id == (
                frame.source.source_recipient_frame_id
            )
        else:
            assert isinstance(frame, LiveOracleAuthorizedPresentationFrameV1)
            assert frame.source.source_frame_id == (
                f"{frame.source.episode_id}:frame:{frame_index}"
            )
            assert frame.current_endpoint.frame_id == frame.source.source_frame_id
        built_frames.append(frame)

    zero, nonzero = built_frames
    assert zero.latest_events is None
    assert zero.latest_transition is None
    if isinstance(zero, LiveOracleAuthorizedPresentationFrameV1):
        assert zero.technical_frame.incoming_transition_id is None
    else:
        assert zero.technical_frame.incoming_recipient_transition_id is None
    assert nonzero.latest_events is not None
    assert nonzero.latest_transition is not None
    incoming_id = nonzero.latest_transition.incoming_transition_id
    if isinstance(nonzero, LiveOracleAuthorizedPresentationFrameV1):
        assert nonzero.technical_frame.incoming_transition_id == incoming_id
        assert incoming_id == five_frames.cases.no_shared.transitions[0].transition_id
    else:
        assert nonzero.technical_frame.incoming_recipient_transition_id == incoming_id
        assert incoming_id.endswith(":transition:0")


@pytest.mark.parametrize(
    "frame_attribute,frame_type",
    (
        (
            "replay_no_shared",
            ReplayNoSharedObsAuthorizedPresentationFrameV1,
        ),
        (
            "replay_shared",
            ReplaySharedObsAuthorizedPresentationFrameV1,
        ),
    ),
)
def test_replay_agent_final_and_nonfinal_inspection_state_machine(
    five_frames: _FiveFrames,
    frame_attribute: str,
    frame_type: type[
        ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1
    ],
) -> None:
    frame = cast(
        ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1,
        getattr(five_frames, frame_attribute),
    )
    payload = frame.model_dump(mode="json")
    payload["replay_inspection"] = None
    with pytest.raises(ValidationError, match="non-final"):
        frame_type.model_validate_json(json.dumps(payload))

    final_payload = frame.model_dump(mode="json")
    final_payload["source"]["source_final_frame_index"] = (
        frame.source.source_frame_index
    )
    final_payload["replay_inspection"] = None
    final_payload["upcoming_transition"] = None
    final_payload["researcher_space"]["final_frame_index"] = (
        frame.source.source_frame_index
    )
    final_payload["researcher_space"]["upcoming_transition"] = None
    final_frame = frame_type.model_validate_json(json.dumps(final_payload))
    assert final_frame.current_endpoint.action_axis.owner_public_agent_id == (
        frame.current_endpoint.action_axis.owner_public_agent_id
    )
    assert final_frame.replay_inspection is None

    forbidden_final = frame.model_dump(mode="json")
    forbidden_final["source"]["source_final_frame_index"] = (
        frame.source.source_frame_index
    )
    forbidden_final["researcher_space"]["final_frame_index"] = (
        frame.source.source_frame_index
    )
    forbidden_final["researcher_space"]["upcoming_transition"] = None
    with pytest.raises(ValidationError, match="final replay Agent"):
        frame_type.model_validate_json(json.dumps(forbidden_final))


@pytest.mark.parametrize(
    ("frame_attribute", "frame_type"),
    (
        ("replay_oracle", ReplayOracleAuthorizedPresentationFrameV1),
        ("replay_no_shared", ReplayNoSharedObsAuthorizedPresentationFrameV1),
        ("replay_shared", ReplaySharedObsAuthorizedPresentationFrameV1),
    ),
)
def test_replay_upcoming_transition_rejects_missing_epoch_and_inspection_drift(
    five_frames: _FiveFrames,
    frame_attribute: str,
    frame_type: type[
        ReplayOracleAuthorizedPresentationFrameV1
        | ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1
    ],
) -> None:
    frame = cast(
        ReplayOracleAuthorizedPresentationFrameV1
        | ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1,
        getattr(five_frames, frame_attribute),
    )
    missing = frame.model_dump(mode="json")
    missing["upcoming_transition"] = None
    with pytest.raises(ValidationError, match="upcoming_transition"):
        frame_type.model_validate_json(json.dumps(missing))

    wrong_epoch = frame.model_dump(mode="json")
    wrong_epoch["upcoming_transition"]["outgoing_transition_index"] += 1
    with pytest.raises(ValidationError, match="Upcoming Transition"):
        frame_type.model_validate_json(json.dumps(wrong_epoch))

    inspection_drift = frame.model_dump(mode="json")
    accepted = inspection_drift["upcoming_transition"]["action_rows"][0][
        "accepted_action"
    ]
    accepted["move_action"] = (accepted["move_action"] + 1) % 9
    with pytest.raises(ValidationError, match=r"inspection.*Upcoming Transition"):
        frame_type.model_validate_json(json.dumps(inspection_drift))


def test_oracle_endpoint_digest_and_source_reject_nested_scene_swap(
    five_frames: _FiveFrames,
) -> None:
    payload = five_frames.replay_oracle.model_dump(mode="json")
    original_scene = copy.deepcopy(payload["current_endpoint"]["scene"])
    payload["current_endpoint"]["scene"]["map"]["width"] += 0.25
    assert payload["current_endpoint"]["scene"] != original_scene
    with pytest.raises(ValidationError, match="digest"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )

    payload = five_frames.replay_oracle.model_dump(mode="json")
    payload["source"]["source_authorized_endpoint_digest_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="digest"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


def test_same_session_frame_zero_final_branch_swaps_reject(
    five_frames: _FiveFrames,
) -> None:
    session = "cp2-5-b-cross-frame-oracle"
    oracle_zero = _replay_oracle_at(
        five_frames.cases,
        frame_index=0,
        selected_internal_slot=0,
        session=session,
    )
    oracle_final = _replay_oracle_at(
        five_frames.cases,
        frame_index=3,
        selected_internal_slot=0,
        session=session,
    )
    oracle_other_actor = _replay_oracle_at(
        five_frames.cases,
        frame_index=0,
        selected_internal_slot=1,
        session=session,
    )
    zero_payload = oracle_zero.model_dump(mode="json")
    final_payload = oracle_final.model_dump(mode="json")
    assert zero_payload["current_endpoint"] != final_payload["current_endpoint"]
    assert (
        zero_payload["current_endpoint"]["scene"]
        != final_payload["current_endpoint"]["scene"]
    )
    assert (
        zero_payload["current_endpoint"]["identity_directory"]
        == final_payload["current_endpoint"]["identity_directory"]
    )

    oracle_poisons: list[dict[str, object]] = []
    whole_endpoint = copy.deepcopy(zero_payload)
    whole_endpoint["current_endpoint"] = copy.deepcopy(
        final_payload["current_endpoint"]
    )
    oracle_poisons.append(whole_endpoint)

    nested_scene = copy.deepcopy(zero_payload)
    nested_scene["current_endpoint"]["scene"] = copy.deepcopy(
        final_payload["current_endpoint"]["scene"]
    )
    oracle_poisons.append(nested_scene)

    directory = copy.deepcopy(zero_payload)
    directory_rows = directory["current_endpoint"]["identity_directory"]["identities"]
    directory_rows[0]["public_agent_id"], directory_rows[1]["public_agent_id"] = (
        directory_rows[1]["public_agent_id"],
        directory_rows[0]["public_agent_id"],
    )
    oracle_poisons.append(directory)

    other_axis = copy.deepcopy(zero_payload)
    other_axis["current_endpoint"]["action_axis"] = copy.deepcopy(
        oracle_other_actor.model_dump(mode="json")["current_endpoint"]["action_axis"]
    )
    oracle_poisons.append(other_axis)

    endpoint_digest = copy.deepcopy(zero_payload)
    endpoint_digest["current_endpoint"]["authorized_endpoint_digest_sha256"] = (
        final_payload["current_endpoint"]["authorized_endpoint_digest_sha256"]
    )
    oracle_poisons.append(endpoint_digest)

    source_digest = copy.deepcopy(zero_payload)
    source_digest["source"]["source_authorized_endpoint_digest_sha256"] = final_payload[
        "source"
    ]["source_authorized_endpoint_digest_sha256"]
    oracle_poisons.append(source_digest)

    technical = copy.deepcopy(zero_payload)
    technical["technical_frame"] = copy.deepcopy(final_payload["technical_frame"])
    oracle_poisons.append(technical)

    coherent_source_endpoint = copy.deepcopy(zero_payload)
    coherent_source_endpoint["source"] = copy.deepcopy(final_payload["source"])
    coherent_source_endpoint["current_endpoint"] = copy.deepcopy(
        final_payload["current_endpoint"]
    )
    oracle_poisons.append(coherent_source_endpoint)

    coherent_source_endpoint_technical = copy.deepcopy(coherent_source_endpoint)
    coherent_source_endpoint_technical["technical_frame"] = copy.deepcopy(
        final_payload["technical_frame"]
    )
    oracle_poisons.append(coherent_source_endpoint_technical)

    reverse_source_endpoint = copy.deepcopy(final_payload)
    reverse_source_endpoint["source"] = copy.deepcopy(zero_payload["source"])
    reverse_source_endpoint["current_endpoint"] = copy.deepcopy(
        zero_payload["current_endpoint"]
    )
    oracle_poisons.append(reverse_source_endpoint)

    for poisoned in oracle_poisons:
        with pytest.raises(ValidationError):
            ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
                json.dumps(poisoned)
            )

    agent_rows = (
        (
            _replay_no_shared_at,
            ReplayNoSharedObsAuthorizedPresentationFrameV1,
            "cp2-5-b-cross-frame-no-shared",
        ),
        (
            _replay_shared_at,
            ReplaySharedObsAuthorizedPresentationFrameV1,
            "cp2-5-b-cross-frame-shared",
        ),
    )
    for builder, frame_type, agent_session in agent_rows:
        zero = builder(
            five_frames.cases,
            frame_index=0,
            session=agent_session,
        )
        final = builder(
            five_frames.cases,
            frame_index=3,
            session=agent_session,
        )
        other_actor = builder(
            five_frames.cases,
            frame_index=0,
            session=agent_session,
            **(
                {"actor_slot": 1}
                if builder is _replay_no_shared_at
                else {"recipient_slot": 1}
            ),
        )
        zero_payload = zero.model_dump(mode="json")
        final_payload = final.model_dump(mode="json")
        assert zero_payload["current_endpoint"] != final_payload["current_endpoint"]
        assert (
            zero_payload["current_endpoint"]["parts"]
            != final_payload["current_endpoint"]["parts"]
        )

        agent_poisons: list[dict[str, object]] = []
        whole_endpoint = copy.deepcopy(zero_payload)
        whole_endpoint["current_endpoint"] = copy.deepcopy(
            final_payload["current_endpoint"]
        )
        agent_poisons.append(whole_endpoint)

        nested_parts = copy.deepcopy(zero_payload)
        nested_parts["current_endpoint"]["parts"] = copy.deepcopy(
            final_payload["current_endpoint"]["parts"]
        )
        agent_poisons.append(nested_parts)

        nested_scene = copy.deepcopy(zero_payload)
        nested_scene["current_endpoint"]["parts"]["scene"] = copy.deepcopy(
            final_payload["current_endpoint"]["parts"]["scene"]
        )
        agent_poisons.append(nested_scene)

        other_axis = copy.deepcopy(zero_payload)
        other_axis["current_endpoint"]["action_axis"] = copy.deepcopy(
            other_actor.model_dump(mode="json")["current_endpoint"]["action_axis"]
        )
        agent_poisons.append(other_axis)

        endpoint_digest = copy.deepcopy(zero_payload)
        endpoint_digest["current_endpoint"]["authorized_endpoint_digest_sha256"] = (
            final_payload["current_endpoint"]["authorized_endpoint_digest_sha256"]
        )
        agent_poisons.append(endpoint_digest)

        source_digest = copy.deepcopy(zero_payload)
        source_digest["source"]["source_authorized_endpoint_digest_sha256"] = (
            final_payload["source"]["source_authorized_endpoint_digest_sha256"]
        )
        agent_poisons.append(source_digest)

        technical = copy.deepcopy(zero_payload)
        technical["technical_frame"] = copy.deepcopy(final_payload["technical_frame"])
        agent_poisons.append(technical)

        coherent_source_endpoint = copy.deepcopy(zero_payload)
        coherent_source_endpoint["source"] = copy.deepcopy(final_payload["source"])
        coherent_source_endpoint["current_endpoint"] = copy.deepcopy(
            final_payload["current_endpoint"]
        )
        agent_poisons.append(coherent_source_endpoint)

        coherent_source_endpoint_technical = copy.deepcopy(coherent_source_endpoint)
        coherent_source_endpoint_technical["technical_frame"] = copy.deepcopy(
            final_payload["technical_frame"]
        )
        agent_poisons.append(coherent_source_endpoint_technical)

        reverse_source_endpoint = copy.deepcopy(final_payload)
        reverse_source_endpoint["source"] = copy.deepcopy(zero_payload["source"])
        reverse_source_endpoint["current_endpoint"] = copy.deepcopy(
            zero_payload["current_endpoint"]
        )
        agent_poisons.append(reverse_source_endpoint)

        for poisoned in agent_poisons:
            with pytest.raises(ValidationError):
                frame_type.model_validate_json(json.dumps(poisoned))


def test_same_frame_coherent_other_session_branches_reject(
    five_frames: _FiveFrames,
) -> None:
    rows: tuple[
        tuple[
            Callable[..., AuthorizedPresentationFrameV1],
            type[
                ReplayOracleAuthorizedPresentationFrameV1
                | ReplayNoSharedObsAuthorizedPresentationFrameV1
                | ReplaySharedObsAuthorizedPresentationFrameV1
            ],
            dict[str, object],
        ],
        ...,
    ] = (
        (
            cast(Callable[..., AuthorizedPresentationFrameV1], _replay_oracle_at),
            ReplayOracleAuthorizedPresentationFrameV1,
            {"selected_internal_slot": 0},
        ),
        (
            cast(Callable[..., AuthorizedPresentationFrameV1], _replay_no_shared_at),
            ReplayNoSharedObsAuthorizedPresentationFrameV1,
            {},
        ),
        (
            cast(Callable[..., AuthorizedPresentationFrameV1], _replay_shared_at),
            ReplaySharedObsAuthorizedPresentationFrameV1,
            {},
        ),
    )
    for builder, frame_type, builder_kwargs in rows:
        session_a = builder(
            five_frames.cases,
            frame_index=1,
            session="cp2-5-b-session-a",
            **builder_kwargs,
        )
        session_b = builder(
            five_frames.cases,
            frame_index=1,
            session="cp2-5-b-session-b",
            **builder_kwargs,
        )
        payload_a = session_a.model_dump(mode="json")
        payload_b = session_b.model_dump(mode="json")
        assert payload_a["current_endpoint"] != payload_b["current_endpoint"]

        coherent_b_under_a_source = copy.deepcopy(payload_b)
        coherent_b_under_a_source["source"] = copy.deepcopy(payload_a["source"])
        coherent_b_under_a_source["source"][
            "source_authorized_endpoint_digest_sha256"
        ] = payload_b["current_endpoint"]["authorized_endpoint_digest_sha256"]
        with pytest.raises(ValidationError):
            frame_type.model_validate_json(json.dumps(coherent_b_under_a_source))


def test_oracle_target_axis_permutations_reject(
    five_frames: _FiveFrames,
) -> None:
    payload = five_frames.replay_oracle.model_dump(mode="json")
    target_rows = payload["current_endpoint"]["action_axis"]["target_actions"]
    target_rows[1], target_rows[2] = target_rows[2], target_rows[1]
    with pytest.raises(ValidationError, match=r"order|digest"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


@pytest.mark.parametrize(
    ("frame_attribute", "frame_type"),
    (
        ("live_oracle", LiveOracleAuthorizedPresentationFrameV1),
        ("live_no_shared", LiveNoSharedObsAuthorizedPresentationFrameV1),
        ("replay_oracle", ReplayOracleAuthorizedPresentationFrameV1),
        ("replay_no_shared", ReplayNoSharedObsAuthorizedPresentationFrameV1),
        ("replay_shared", ReplaySharedObsAuthorizedPresentationFrameV1),
    ),
)
def test_decision_target_variant_exactly_matches_current_scene_membership(
    five_frames: _FiveFrames,
    frame_attribute: str,
    frame_type: type[
        LiveOracleAuthorizedPresentationFrameV1
        | LiveNoSharedObsAuthorizedPresentationFrameV1
        | ReplayOracleAuthorizedPresentationFrameV1
        | ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1
    ],
) -> None:
    frame = cast(
        LiveOracleAuthorizedPresentationFrameV1
        | LiveNoSharedObsAuthorizedPresentationFrameV1
        | ReplayOracleAuthorizedPresentationFrameV1
        | ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1,
        getattr(five_frames, frame_attribute),
    )
    base = frame.model_dump(mode="json")
    inspection = (
        base["live_inspection"]["inspection"]["draft"]
        if frame_attribute.startswith("live_")
        else base["replay_inspection"]
    )
    mask = inspection["decision_mask"]
    visible_row = next(
        row
        for row in mask["target_actions"]
        if row["target_kind"] == "visible_authorized_agent"
    )
    hidden_row = next(
        row
        for row in mask["target_actions"]
        if row["target_kind"] == "axis_only_authorized_agent"
    )

    visible_as_axis_only = copy.deepcopy(base)
    poisoned_inspection = (
        visible_as_axis_only["live_inspection"]["inspection"]["draft"]
        if frame_attribute.startswith("live_")
        else visible_as_axis_only["replay_inspection"]
    )
    poisoned_visible = poisoned_inspection["decision_mask"]["target_actions"][
        visible_row["target_action"]
    ]
    poisoned_visible["target_kind"] = "axis_only_authorized_agent"
    poisoned_visible.pop("target_presentation_key")
    poisoned_visible.pop("target_anchor")
    with pytest.raises(ValidationError, match=r"present.*visible target"):
        frame_type.model_validate_json(json.dumps(visible_as_axis_only))

    hidden_as_visible = copy.deepcopy(base)
    poisoned_inspection = (
        hidden_as_visible["live_inspection"]["inspection"]["draft"]
        if frame_attribute.startswith("live_")
        else hidden_as_visible["replay_inspection"]
    )
    poisoned_hidden = poisoned_inspection["decision_mask"]["target_actions"][
        hidden_row["target_action"]
    ]
    hidden_public_id = hidden_row["target_public_agent_id"]
    if isinstance(
        frame,
        LiveOracleAuthorizedPresentationFrameV1
        | ReplayOracleAuthorizedPresentationFrameV1,
    ):
        hidden_key = oracle_presentation_key_v1(
            authority_session_id=frame.source.source_session_id,
            public_agent_id=hidden_public_id,
        )
    else:
        hidden_key = pov_presentation_key_v1(
            authority_session_id=frame.source.source_session_id,
            recipient_public_agent_id=(frame.source.source_recipient_public_agent_id),
            public_agent_id=hidden_public_id,
        )
    poisoned_hidden["target_kind"] = "visible_authorized_agent"
    poisoned_hidden["target_presentation_key"] = hidden_key
    poisoned_hidden["target_anchor"] = [0.5, 0.5]
    with pytest.raises(ValidationError, match=r"absent.*axis-only|does not join"):
        frame_type.model_validate_json(json.dumps(hidden_as_visible))

    payload = five_frames.replay_oracle.model_dump(mode="json")
    axis = payload["latest_transition"]["action_rows"][0][
        "target_action_recipient_public_agent_id_by_id"
    ]
    axis[6], axis[7] = axis[7], axis[6]
    with pytest.raises(ValidationError, match="order"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


def test_oracle_rejection_events_exactly_match_rejected_action_rows(
    five_frames: _FiveFrames,
) -> None:
    missing_event = five_frames.replay_oracle.model_dump(mode="json")
    first_row = missing_event["latest_transition"]["action_rows"][0]
    assert first_row["submitted_action"] == first_row["accepted_action"]
    first_row["submitted_action"]["move_action"] = 1
    with pytest.raises(ValidationError, match=r"rejection events.*action rows"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(missing_event)
        )

    spurious_event = five_frames.replay_oracle.model_dump(mode="json")
    latest_events = spurious_event["latest_events"]
    action_row = spurious_event["latest_transition"]["action_rows"][0]
    actor_public_id = action_row["actor_public_agent_id"]
    trajectory = next(
        row
        for row in latest_events["agent_phase_trajectories"]
        if row["agent_public_agent_id"] == actor_public_id
    )
    transition_id = latest_events["incoming_transition_id"]
    rejection = {
        "event_id": f"{transition_id}:event:0000",
        "ordinal": 0,
        "phase_rank": 10,
        "event_kind": "action_rejected",
        "actor_identity": {
            "identity_kind": "authorized_agent",
            "presentation_key": action_row["actor_presentation_key"],
            "public_agent_id": actor_public_id,
        },
        "actor_configured_active": True,
        "rejection_component": "movement",
        "submitted_action": copy.deepcopy(action_row["submitted_action"]),
        "actor_anchor": copy.deepcopy(trajectory["transition_start"]),
    }
    for ordinal, event in enumerate(latest_events["events"], start=1):
        event["ordinal"] = ordinal
        event["event_id"] = f"{transition_id}:event:{ordinal:04d}"
    latest_events["events"].insert(0, rejection)
    latest_events["event_count"] += 1
    latest_events["ordered_event_ids"] = [
        f"{transition_id}:event:{ordinal:04d}"
        for ordinal in range(latest_events["event_count"])
    ]
    latest_events["ordered_event_kinds"].insert(0, "action_rejected")
    with pytest.raises(ValidationError, match=r"rejection events.*action rows"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(spurious_event)
        )


def _agent_payload_with_rejection_components(
    frame: LiveNoSharedObsAuthorizedPresentationFrameV1
    | ReplayNoSharedObsAuthorizedPresentationFrameV1
    | ReplaySharedObsAuthorizedPresentationFrameV1,
    *,
    submitted_action: dict[str, int],
    rejection_components: tuple[str, ...],
) -> dict[str, object]:
    payload = frame.model_dump(mode="json")
    latest_transition = cast(dict[str, object], payload["latest_transition"])
    action_rows = cast(list[dict[str, object]], latest_transition["action_rows"])
    action_rows[0]["submitted_action"] = copy.deepcopy(submitted_action)
    latest_events = cast(dict[str, object], payload["latest_events"])
    if latest_events.get("summary_kind") == "no_shared_obs_recipient_cues":
        cues = cast(list[dict[str, object]], latest_events["cues"])
        cues[0]["outcome"] = (
            "accepted"
            if submitted_action
            == cast(dict[str, object], action_rows[0]["accepted_action"])
            else "rejected"
        )

    visual = cast(dict[str, object], payload["visual_events"])
    trajectories = cast(list[dict[str, object]], visual["agent_phase_trajectories"])
    recipient_public_id = cast(str, visual["recipient_public_agent_id"])
    recipient_key = cast(str, visual["recipient_presentation_key"])
    trajectory = next(
        row
        for row in trajectories
        if row["agent_public_agent_id"] == recipient_public_id
    )
    events = [
        event
        for event in cast(list[dict[str, object]], visual["events"])
        if event["event_kind"] != "action_rejected"
    ]
    rejection_events = [
        {
            "event_id": "reindexed below",
            "ordinal": 0,
            "phase_rank": 10,
            "event_kind": "action_rejected",
            "actor_identity": {
                "identity_kind": "authorized_agent",
                "presentation_key": recipient_key,
                "public_agent_id": recipient_public_id,
            },
            "actor_configured_active": True,
            "rejection_component": component,
            "submitted_action": copy.deepcopy(submitted_action),
            "actor_anchor": copy.deepcopy(trajectory["transition_start"]),
        }
        for component in rejection_components
    ]
    events = [*rejection_events, *events]
    transition_id = cast(str, visual["incoming_recipient_transition_id"])
    for ordinal, event in enumerate(events):
        event["ordinal"] = ordinal
        event["event_id"] = f"{transition_id}:visual-event:{ordinal:04d}"
    visual["events"] = events
    visual["event_count"] = len(events)
    visual["ordered_event_ids"] = [event["event_id"] for event in events]
    visual["ordered_event_kinds"] = [event["event_kind"] for event in events]
    return payload


@pytest.mark.parametrize(
    ("frame_name", "frame_type"),
    (
        ("replay_no_shared", ReplayNoSharedObsAuthorizedPresentationFrameV1),
        ("replay_shared", ReplaySharedObsAuthorizedPresentationFrameV1),
    ),
)
@pytest.mark.parametrize(
    ("submitted_action", "rejection_components"),
    (
        (
            {"move_action": 0, "target_action": 0, "use_ultimate_action": 0},
            (),
        ),
        (
            {"move_action": 1, "target_action": 0, "use_ultimate_action": 0},
            ("movement",),
        ),
        (
            {"move_action": 0, "target_action": 6, "use_ultimate_action": 1},
            ("combat_pair",),
        ),
        (
            {"move_action": 1, "target_action": 6, "use_ultimate_action": 1},
            ("movement", "combat_pair"),
        ),
        (
            {"move_action": 99, "target_action": 0, "use_ultimate_action": 0},
            ("domain",),
        ),
    ),
)
def test_agent_rejection_events_retain_each_rejected_head_group(
    five_frames: _FiveFrames,
    frame_name: str,
    frame_type: type[
        LiveNoSharedObsAuthorizedPresentationFrameV1
        | ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1
    ],
    submitted_action: dict[str, int],
    rejection_components: tuple[str, ...],
) -> None:
    payload = _agent_payload_with_rejection_components(
        cast(
            LiveNoSharedObsAuthorizedPresentationFrameV1
            | ReplayNoSharedObsAuthorizedPresentationFrameV1
            | ReplaySharedObsAuthorizedPresentationFrameV1,
            getattr(five_frames, frame_name),
        ),
        submitted_action=submitted_action,
        rejection_components=rejection_components,
    )

    restored = frame_type.model_validate_json(json.dumps(payload))
    assert restored.visual_events is not None
    assert (
        tuple(
            event.rejection_component
            for event in restored.visual_events.events
            if event.event_kind == "action_rejected"
        )
        == rejection_components
    )


@pytest.mark.parametrize(
    ("submitted_action", "rejection_components", "mismatch_second_tuple"),
    (
        (
            {"move_action": 1, "target_action": 6, "use_ultimate_action": 1},
            ("movement",),
            False,
        ),
        (
            {"move_action": 1, "target_action": 6, "use_ultimate_action": 1},
            ("movement", "movement"),
            False,
        ),
        (
            {"move_action": 1, "target_action": 6, "use_ultimate_action": 1},
            ("combat_pair", "movement"),
            False,
        ),
        (
            {"move_action": 99, "target_action": 0, "use_ultimate_action": 0},
            ("movement",),
            False,
        ),
        (
            {"move_action": 1, "target_action": 6, "use_ultimate_action": 1},
            ("movement", "combat_pair"),
            True,
        ),
    ),
)
def test_agent_rejection_events_reject_inexact_head_group_inventory(
    five_frames: _FiveFrames,
    submitted_action: dict[str, int],
    rejection_components: tuple[str, ...],
    mismatch_second_tuple: bool,
) -> None:
    payload = _agent_payload_with_rejection_components(
        five_frames.replay_no_shared,
        submitted_action=submitted_action,
        rejection_components=rejection_components,
    )
    if mismatch_second_tuple:
        visual = cast(dict[str, object], payload["visual_events"])
        events = cast(list[dict[str, object]], visual["events"])
        events[1]["submitted_action"] = {
            "move_action": 1,
            "target_action": 6,
            "use_ultimate_action": 0,
        }

    with pytest.raises(ValidationError, match="visual rejection"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


def test_wrong_session_bare_scene_agent_and_incoming_anchor_keys_reject(
    five_frames: _FiveFrames,
) -> None:
    for path in ("scene_agent", "incoming_anchor"):
        payload = five_frames.replay_oracle.model_dump(mode="json")
        if path == "scene_agent":
            payload["current_endpoint"]["scene"]["agents"][0]["presentation_key"] = (
                "oracle_" + "f" * 64
            )
        else:
            payload["latest_events"]["agent_phase_trajectories"][0]["successor"][
                "presentation_key"
            ] = "oracle_" + "f" * 64
        with pytest.raises(
            ValidationError,
            match=r"key|digest|trajectory|aura field|source agent",
        ):
            ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
                json.dumps(payload)
            )


def test_every_serialized_presentation_key_is_recomputed_for_its_authority(
    five_frames: _FiveFrames,
) -> None:
    adapter: TypeAdapter[AuthorizedPresentationFrameV1] = TypeAdapter(
        AuthorizedPresentationFrameV1
    )
    expected_counts = {
        "live_oracle": 53,
        "live_no_shared_obs_agent_pov": 57,
        "replay_oracle": 58,
        "replay_no_shared_obs_agent_pov": 62,
        "replay_shared_obs_agent_pov": 77,
    }
    for frame in five_frames.rows:
        payload = frame.model_dump(mode="json")
        key_paths = _presentation_key_paths(payload)
        assert len(key_paths) == expected_counts[frame.presentation_kind]
        source = payload["source"]
        source_session_id = cast(str, source["source_session_id"])
        recipient_public_agent_id = source.get("source_recipient_public_agent_id")
        for path, current_key, public_agent_id in key_paths:
            researcher_key = path[0] == "researcher_space"
            oracle_key = "oracle" in frame.presentation_kind or researcher_key
            if current_key is None:
                assert public_agent_id is None
                wrong = (
                    oracle_presentation_key_v1(
                        authority_session_id=source_session_id,
                        public_agent_id="poison-inactive-agent",
                    )
                    if oracle_key
                    else pov_presentation_key_v1(
                        authority_session_id=source_session_id,
                        recipient_public_agent_id=cast(
                            str,
                            recipient_public_agent_id,
                        ),
                        public_agent_id="poison-inactive-agent",
                    )
                )
                poisoned = copy.deepcopy(payload)
                _replace_path(poisoned, path, wrong)
                with pytest.raises(ValidationError):
                    adapter.validate_json(json.dumps(poisoned))
                continue
            assert public_agent_id is not None
            if oracle_key:
                expected = oracle_presentation_key_v1(
                    authority_session_id=source_session_id,
                    public_agent_id=public_agent_id,
                )
                wrong = oracle_presentation_key_v1(
                    authority_session_id=f"{source_session_id}-wrong",
                    public_agent_id=public_agent_id,
                )
            else:
                assert type(recipient_public_agent_id) is str
                expected = pov_presentation_key_v1(
                    authority_session_id=source_session_id,
                    recipient_public_agent_id=recipient_public_agent_id,
                    public_agent_id=public_agent_id,
                )
                wrong = pov_presentation_key_v1(
                    authority_session_id=f"{source_session_id}-wrong",
                    recipient_public_agent_id=recipient_public_agent_id,
                    public_agent_id=public_agent_id,
                )
            assert current_key == expected, path
            poisoned = copy.deepcopy(payload)
            _replace_path(poisoned, path, wrong)
            with pytest.raises(ValidationError):
                adapter.validate_json(json.dumps(poisoned))


def test_agent_sources_and_technical_frames_have_exact_privacy_allowlists(
    five_frames: _FiveFrames,
) -> None:
    forbidden = (
        "global_slot",
        "artifact_digest",
        "context_digest",
        "trajectory",
        "timeline",
        "cursor_generation",
        "choreography_generation",
        "metric",
        "processing",
        "completion",
    )
    for frame in five_frames.rows:
        if frame.authority.authority_kind != "agent_pov":
            continue
        source_keys = set(frame.source.model_dump(mode="json"))
        technical_keys = set(frame.technical_frame.model_dump(mode="json"))
        assert not any(
            token in key for key in source_keys | technical_keys for token in forbidden
        )
        payload = frame.model_dump(mode="json")
        researcher = payload.pop("researcher_space", None)
        assert "oracle_" not in json.dumps(payload)
        if researcher is not None:
            assert "oracle_" in json.dumps(researcher)
            researcher_keys = _nested_keys(researcher)
            forbidden_researcher_keys = {
                "position",
                "map",
                "spawn_pads",
                "respawn_waves",
                "aura_fields",
                "latest_events",
                "visual_events",
                "artifact_id",
                "timeline_id",
                "processing",
                "completion",
                "metric_report",
            }
            leaked_researcher_keys = researcher_keys & forbidden_researcher_keys
            assert not leaked_researcher_keys, leaked_researcher_keys

    oracle_source = five_frames.replay_oracle.source
    forbidden_oracle_values = {
        oracle_source.source_artifact_id,
        oracle_source.source_timeline_id,
        oracle_source.source_context_digest_sha256,
        oracle_source.source_trajectory_content_digest_sha256,
        oracle_source.source_artifact_digest_sha256,
        *(frame.frame_id for frame in five_frames.cases.no_shared.frames),
        *(
            transition.transition_id
            for transition in five_frames.cases.no_shared.transitions
        ),
        *(
            event.event_id
            for transition in five_frames.cases.no_shared.transitions
            for event in transition.events
        ),
    }
    assert forbidden_oracle_values
    for frame in five_frames.rows:
        if frame.authority.authority_kind != "agent_pov":
            continue
        payload = frame.model_dump(mode="json")
        payload.pop("researcher_space", None)
        serialized_values = _serialized_string_values(payload)
        assert serialized_values.isdisjoint(forbidden_oracle_values)


def test_agent_inspection_reference_mask_and_anchor_cross_swaps_reject(
    five_frames: _FiveFrames,
) -> None:
    no_shared = five_frames.replay_no_shared.model_dump(mode="json")
    shared = five_frames.replay_shared.model_dump(mode="json")

    payload = copy.deepcopy(no_shared)
    payload["replay_inspection"]["transition_reference"] = shared["replay_inspection"][
        "transition_reference"
    ]
    with pytest.raises(ValidationError, match="transition_reference"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )

    payload = copy.deepcopy(no_shared)
    payload["replay_inspection"]["actor_anchor"][0] += 0.25
    with pytest.raises(ValidationError, match="anchor"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )

    payload = copy.deepcopy(no_shared)
    movement = payload["replay_inspection"]["decision_mask"]["movement_action_mask"]
    movement[0] = not movement[0]
    with pytest.raises(ValidationError, match=r"mask|legality|legal under"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


def test_replay_inspection_transition_references_cannot_cross_authorities(
    five_frames: _FiveFrames,
) -> None:
    rows = (
        (
            five_frames.replay_oracle,
            ReplayOracleAuthorizedPresentationFrameV1,
        ),
        (
            five_frames.replay_no_shared,
            ReplayNoSharedObsAuthorizedPresentationFrameV1,
        ),
        (
            five_frames.replay_shared,
            ReplaySharedObsAuthorizedPresentationFrameV1,
        ),
    )
    references = tuple(
        cast(dict[str, object], frame.model_dump(mode="json"))["replay_inspection"]
        for frame, _ in rows
    )
    for target_index, (frame, frame_type) in enumerate(rows):
        for source_index, source_inspection in enumerate(references):
            if source_index == target_index:
                continue
            payload = frame.model_dump(mode="json")
            payload["replay_inspection"]["transition_reference"] = cast(
                dict[str, object],
                source_inspection,
            )["transition_reference"]
            with pytest.raises(ValidationError, match="transition_reference"):
                frame_type.model_validate_json(json.dumps(payload))


def test_agent_latest_events_reject_fabricated_axis_external_identities(
    five_frames: _FiveFrames,
) -> None:
    fabricated_id = "fabricated-agent-cue-identity"

    no_shared = five_frames.replay_no_shared
    no_shared_payload = no_shared.model_dump(mode="json")
    cue = next(
        row
        for row in no_shared_payload["latest_events"]["cues"]
        if "agent_public_agent_id" in row
    )
    fabricated_no_shared_key = pov_presentation_key_v1(
        authority_session_id=no_shared.source.source_session_id,
        recipient_public_agent_id=(no_shared.source.source_recipient_public_agent_id),
        public_agent_id=fabricated_id,
    )
    cue["agent_public_agent_id"] = fabricated_id
    cue["agent_presentation_key"] = fabricated_no_shared_key
    for observation_name in ("start_observation", "successor_observation"):
        if cue.get(observation_name) is not None:
            cue[observation_name]["public_agent_id"] = fabricated_id
            cue[observation_name]["presentation_key"] = fabricated_no_shared_key
    with pytest.raises(ValidationError, match="outside its action axis"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(no_shared_payload)
        )

    shared = five_frames.replay_shared
    shared_payload = shared.model_dump(mode="json")
    delta = shared_payload["latest_events"]["deltas"][0]
    fabricated_shared_key = pov_presentation_key_v1(
        authority_session_id=shared.source.source_session_id,
        recipient_public_agent_id=shared.source.source_recipient_public_agent_id,
        public_agent_id=fabricated_id,
    )
    delta["agent_public_agent_id"] = fabricated_id
    delta["agent_presentation_key"] = fabricated_shared_key
    for observation_name in ("start_observation", "successor_observation"):
        if delta.get(observation_name) is not None:
            delta[observation_name]["public_agent_id"] = fabricated_id
            delta[observation_name]["presentation_key"] = fabricated_shared_key
    with pytest.raises(ValidationError, match="outside its action axis"):
        ReplaySharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(shared_payload)
        )


def test_agent_incoming_observation_relation_and_team_join_axis(
    five_frames: _FiveFrames,
) -> None:
    frame = five_frames.replay_no_shared
    base = frame.model_dump(mode="json")
    cue = next(
        row
        for row in base["latest_events"]["cues"]
        if row.get("cue_type") == "visible_body_observation_changed"
    )
    assert cue["agent_public_agent_id"] != frame.source.source_recipient_public_agent_id

    def observations(payload_cue: dict[str, object]) -> tuple[dict[str, object], ...]:
        return tuple(
            cast(dict[str, object], payload_cue[name])
            for name in ("start_observation", "successor_observation")
            if payload_cue.get(name) is not None
        )

    opponent_as_ally = copy.deepcopy(base)
    opponent_cue = next(
        row
        for row in opponent_as_ally["latest_events"]["cues"]
        if row.get("cue_type") == "visible_body_observation_changed"
    )
    opponent_id = frame.current_endpoint.action_axis.target_public_agent_id_by_action[6]
    assert opponent_id is not None
    opponent_key = pov_presentation_key_v1(
        authority_session_id=frame.source.source_session_id,
        recipient_public_agent_id=frame.source.source_recipient_public_agent_id,
        public_agent_id=opponent_id,
    )
    opponent_cue["agent_public_agent_id"] = opponent_id
    opponent_cue["agent_presentation_key"] = opponent_key
    for observation in observations(opponent_cue):
        observation["public_agent_id"] = opponent_id
        observation["presentation_key"] = opponent_key
    with pytest.raises(ValidationError, match=r"relation/team.*axis"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(opponent_as_ally)
        )

    ally_as_opponent = copy.deepcopy(base)
    ally_cue = next(
        row
        for row in ally_as_opponent["latest_events"]["cues"]
        if row.get("cue_type") == "visible_body_observation_changed"
    )
    for observation in observations(ally_cue):
        observation["relation"] = "opponent"
        observation["team_id"] = 2
    with pytest.raises(ValidationError, match=r"relation/team.*axis"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(ally_as_opponent)
        )

    wrong_team = copy.deepcopy(base)
    team_cue = next(
        row
        for row in wrong_team["latest_events"]["cues"]
        if row.get("cue_type") == "visible_body_observation_changed"
    )
    for observation in observations(team_cue):
        observation["team_id"] = 2
    with pytest.raises(ValidationError, match=r"relation/team.*axis"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(wrong_team)
        )

    recipient_as_ally = copy.deepcopy(base)
    recipient_cue = next(
        row
        for row in recipient_as_ally["latest_events"]["cues"]
        if row.get("cue_type") == "visible_body_observation_changed"
    )
    recipient_id = frame.source.source_recipient_public_agent_id
    recipient_key = frame.authority.recipient_presentation_key
    recipient_cue["agent_public_agent_id"] = recipient_id
    recipient_cue["agent_presentation_key"] = recipient_key
    for observation in observations(recipient_cue):
        observation["public_agent_id"] = recipient_id
        observation["presentation_key"] = recipient_key
    with pytest.raises(ValidationError, match=r"self|recipient|relation/team"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(recipient_as_ally)
        )

    shared = five_frames.replay_shared.model_dump(mode="json")
    shared_delta = shared["latest_events"]["deltas"][0]
    for name in ("start_observation", "successor_observation"):
        observation = shared_delta.get(name)
        if observation is not None:
            observation["relation"] = "opponent"
            observation["team_id"] = 2
    with pytest.raises(ValidationError, match=r"relation/team|current scene"):
        ReplaySharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(shared)
        )


def test_agent_incoming_successors_join_authorized_current_endpoint(
    five_frames: _FiveFrames,
) -> None:
    no_shared = five_frames.replay_no_shared.model_dump(mode="json")
    body_cue = next(
        row
        for row in no_shared["latest_events"]["cues"]
        if row.get("cue_type") == "visible_body_observation_changed"
    )
    body_cue["successor_observation"]["position"][0] += 0.125
    with pytest.raises(
        ValidationError,
        match=r"successor observation.*current scene",
    ):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(no_shared)
        )

    retained_as_disappearance = five_frames.replay_no_shared.model_dump(mode="json")
    body_cue = next(
        row
        for row in retained_as_disappearance["latest_events"]["cues"]
        if row.get("cue_type") == "visible_body_observation_changed"
    )
    body_cue["observation_change_kind"] = "disappearance"
    body_cue["successor_observation"] = None
    with pytest.raises(ValidationError, match="disappearance identity remains"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(retained_as_disappearance)
        )

    shared = five_frames.replay_shared.model_dump(mode="json")
    delta = shared["latest_events"]["deltas"][0]
    delta["successor_observation"]["current_health"] = 199.0
    delta["changed_dynamic_fields"] = ["current_health", "statuses"]
    with pytest.raises(
        ValidationError,
        match=r"successor observation.*current scene",
    ):
        ReplaySharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(shared)
        )

    shared_disappearance = five_frames.replay_shared.model_dump(mode="json")
    delta = shared_disappearance["latest_events"]["deltas"][0]
    delta["delta_kind"] = "disappearance"
    delta["start_observation_sources"] = shared_disappearance["current_endpoint"][
        "parts"
    ]["agent_observation_provenance"][1]["observation_sources"]
    delta.pop("changed_dynamic_fields")
    delta.pop("successor_observation")
    with pytest.raises(ValidationError, match="disappearance identity remains"):
        ReplaySharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(shared_disappearance)
        )


@pytest.mark.parametrize(
    ("cue_type", "cue_fields", "poison_field", "poison_value"),
    (
        (
            "own_position_changed",
            {"start_position": [1.0, 1.5], "successor_position": [1.5, 1.5]},
            "successor_position",
            [1.625, 1.5],
        ),
        (
            "own_health_changed",
            {"start_health": 78.0, "successor_health": 80.0},
            "successor_health",
            79.0,
        ),
        (
            "own_cooldown_changed",
            {"start_remaining_ticks": 1, "successor_remaining_ticks": 0},
            "successor_remaining_ticks",
            2,
        ),
        (
            "own_lifecycle_changed",
            {
                "start_active": True,
                "successor_active": True,
                "start_life_state": "alive",
                "successor_life_state": "alive",
                "start_spawn_shield_remaining_ticks": 1,
                "successor_spawn_shield_remaining_ticks": 0,
            },
            "successor_life_state",
            "corpse",
        ),
    ),
)
def test_no_shared_own_successor_cues_join_current_self(
    five_frames: _FiveFrames,
    cue_type: str,
    cue_fields: dict[str, object],
    poison_field: str,
    poison_value: object,
) -> None:
    payload = five_frames.replay_no_shared.model_dump(mode="json")
    summary = payload["latest_events"]
    transition_id = summary["incoming_recipient_transition_id"]
    body_cue = summary["cues"][1]
    body_cue["ordinal"] = 2
    body_cue["cue_id"] = f"{transition_id}:cue:2"
    own_cue = {
        "cue_type": cue_type,
        "cue_id": f"{transition_id}:cue:1",
        "pov_transition_id": transition_id,
        "ordinal": 1,
        **cue_fields,
    }
    summary["cues"].insert(1, own_cue)
    summary["cue_count"] = 3
    frame = ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
        json.dumps(payload)
    )
    assert frame.latest_events is not None

    poisoned = copy.deepcopy(payload)
    poisoned["latest_events"]["cues"][1][poison_field] = poison_value
    with pytest.raises(ValidationError, match=r"own .* successor.*self"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(poisoned)
        )


def test_no_shared_own_status_successor_joins_current_self(
    five_frames: _FiveFrames,
) -> None:
    payload = five_frames.replay_no_shared.model_dump(mode="json")
    summary = payload["latest_events"]
    transition_id = summary["incoming_recipient_transition_id"]
    body_cue = summary["cues"][1]
    foreign_statuses = copy.deepcopy(body_cue["successor_observation"]["statuses"])
    assert foreign_statuses
    body_cue["ordinal"] = 2
    body_cue["cue_id"] = f"{transition_id}:cue:2"
    summary["cues"].insert(
        1,
        {
            "cue_type": "own_status_changed",
            "cue_id": f"{transition_id}:cue:1",
            "pov_transition_id": transition_id,
            "ordinal": 1,
            "start_statuses": [],
            "successor_statuses": foreign_statuses,
        },
    )
    summary["cue_count"] = 3
    with pytest.raises(ValidationError, match=r"own status successor.*self"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


def test_real_death_successor_keeps_selected_recipient_corpse_authorized() -> None:
    launch = debugger_test_launch_specification(0)
    session = create_session(
        get_scenario("death_respawn_cycle"),
        seed=0,
        evaluation_launch_specification=(
            build_debugger_evaluation_launch_specification_v1(
                root_seed=0,
                code_revision=launch.code_revision,
                capture_profile="evaluation_metric_complete",
            )
        ),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    session = submit_next_script_frame(session)
    session = select_controlled_actor(session, 5)
    view = session.incoming_evaluation_view
    assert view is not None
    death = build_actor_pov_adjacent_transition_slice_v1(view, global_slot=5)
    assert death.start_frame.self_features[5] == 1.0
    assert death.successor_frame.self_features[5] == 0.0

    authority = "cp2-5-b-real-death"
    parts = build_no_shared_obs_authorized_scene_v1(
        death,
        public_catalog=session.evaluation_context.static_mechanics_catalog,
        authority_session_id=authority,
        frame_index=death.successor_frame.frame_index,
    )
    self_agent = next(
        row
        for row in parts.scene.agents
        if row.public_agent_id == parts.recipient_public_agent_id
    )
    assert self_agent.relation == "self"
    assert self_agent.life_state == "corpse"
    assert all(
        row.life_state == "alive"
        for row in parts.scene.agents
        if row.relation != "self"
    )
    self_pad = next(
        row
        for row in parts.scene.spawn_pads
        if row.assigned_public_agent_id == parts.recipient_public_agent_id
    )
    assert self_pad.configured_active
    assert not self_pad.currently_alive

    latest_events = build_live_no_shared_obs_incoming_summary_v1(
        death,
        public_catalog=session.evaluation_context.static_mechanics_catalog,
        authority_session_id=authority,
    )
    lifecycle = next(
        cue for cue in latest_events.cues if cue.cue_type == "own_lifecycle_changed"
    )
    assert lifecycle.successor_active
    assert lifecycle.successor_life_state == "corpse"

    endpoint = build_no_shared_obs_authorized_current_endpoint_v1(
        parts=parts,
        axis_mapping=death.axis_mapping,
    )
    latest_transition = _agent_latest_transition(
        death.transition,
        axis=endpoint.action_axis,
        start_tick=death.start_frame.simulator_step_count,
    )
    start_parts = build_no_shared_obs_authorized_scene_v1(
        death,
        public_catalog=session.evaluation_context.static_mechanics_catalog,
        authority_session_id=authority,
        frame_index=death.start_frame.frame_index,
    )
    visual_events = build_agent_pov_visual_incoming_summary_v1(
        build_visual_event_batch_v2(view),
        transition_start_scene=start_parts.scene,
        successor_scene=parts.scene,
        recipient_public_agent_id=parts.recipient_public_agent_id,
        incoming_recipient_transition_id=(
            latest_events.incoming_recipient_transition_id
        ),
        incoming_start_recipient_frame_id=(
            latest_events.incoming_start_recipient_frame_id
        ),
        incoming_successor_recipient_frame_id=(
            latest_events.incoming_successor_recipient_frame_id
        ),
    )
    frame_index = death.successor_frame.frame_index
    oracle_raw = build_debugger_frame(
        session,
        session_id=authority,
        revision=frame_index,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    assert type(oracle_raw) is ResearcherLiveDebuggerFrameV2
    oracle_presentation = build_live_oracle_authorized_presentation_v1(
        session.evaluation_context,
        session.current_evaluation_frame,
        session.incoming_evaluation_view,
        oracle_raw,
    )
    frame = LiveNoSharedObsAuthorizedPresentationFrameV1(
        schema_version=1,
        presentation_kind="live_no_shared_obs_agent_pov",
        product_kind="combat_debugger",
        source=LiveNoSharedObsPresentationSourceIdentityV1(
            source_kind="live_no_shared_obs_frame",
            source_session_id=authority,
            source_run_generation=0,
            source_revision=frame_index,
            source_authority_epoch=frame_index,
            episode_id=parts.source_episode_id,
            source_frame_index=frame_index,
            source_recipient_public_agent_id=parts.recipient_public_agent_id,
            source_recipient_frame_id=parts.source_recipient_frame_id,
            source_simulator_step_count=parts.source_simulator_step_count,
            source_submission_scope="scripted_playback",
            source_authorized_endpoint_digest_sha256=(
                endpoint.authorized_endpoint_digest_sha256
            ),
        ),
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
            episode_id=parts.source_episode_id,
            recipient_frame_index=frame_index,
            simulator_step_count=parts.source_simulator_step_count,
            incoming_recipient_transition_id=(latest_transition.incoming_transition_id),
        ),
        live_inspection=LiveNoSharedObsInspectionEnvelopeV1(
            envelope_kind="live_no_shared_obs_source_bound_inspection",
            source_session_id=authority,
            source_run_generation=0,
            source_revision=frame_index,
            source_authority_epoch=frame_index,
            episode_id=parts.source_episode_id,
            source_frame_index=frame_index,
            source_recipient_public_agent_id=parts.recipient_public_agent_id,
            source_recipient_frame_id=parts.source_recipient_frame_id,
            source_simulator_step_count=parts.source_simulator_step_count,
            inspection=LiveScriptedPlaybackInspectionV1(
                inspection_kind="scripted_playback_inspection",
                submission_scope="scripted_playback",
                editable_draft_available=False,
                advance_semantics="registered_script_frame",
            ),
        ),
        researcher_space=build_live_researcher_space_v1(oracle_presentation),
    )
    encoded = frame.model_dump_json()
    parsed = LiveNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(encoded)
    parsed_self = next(
        row
        for row in parsed.current_endpoint.parts.scene.agents
        if row.public_agent_id == parsed.source.source_recipient_public_agent_id
    )
    assert parsed_self.life_state == "corpse"


@pytest.mark.parametrize("delta_kind", ("appearance", "observation_provenance_change"))
def test_shared_successor_provenance_joins_current_endpoint(
    five_frames: _FiveFrames,
    delta_kind: str,
) -> None:
    payload = five_frames.replay_shared.model_dump(mode="json")
    delta = payload["latest_events"]["deltas"][0]
    current_sources = payload["current_endpoint"]["parts"][
        "agent_observation_provenance"
    ][1]["observation_sources"]
    assert len(current_sources) == 3
    if delta_kind == "appearance":
        delta["delta_kind"] = "appearance"
        delta["successor_observation_sources"] = copy.deepcopy(current_sources)
        delta.pop("changed_dynamic_fields")
        delta.pop("start_observation")
    else:
        delta["delta_kind"] = "observation_provenance_change"
        delta["start_observation_sources"] = copy.deepcopy(current_sources[:1])
        delta["successor_observation_sources"] = copy.deepcopy(current_sources)
        delta.pop("changed_dynamic_fields")
        delta.pop("start_observation")
        delta.pop("successor_observation")
    frame = ReplaySharedObsAuthorizedPresentationFrameV1.model_validate_json(
        json.dumps(payload)
    )
    assert frame.latest_events is not None

    poisoned = copy.deepcopy(payload)
    poisoned["latest_events"]["deltas"][0]["successor_observation_sources"] = (
        copy.deepcopy(current_sources[:2])
    )
    with pytest.raises(ValidationError, match=r"provenance.*current endpoint"):
        ReplaySharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(poisoned)
        )


def test_shared_sensor_sources_remain_recipient_and_teammate_authorized(
    five_frames: _FiveFrames,
) -> None:
    frame = five_frames.replay_shared
    base = frame.model_dump(mode="json")
    sources = base["current_endpoint"]["parts"]["authorized_sensor_sources"]
    assert sources[0]["source_kind"] == "recipient_base"
    assert sources[1]["source_kind"] == "shared_sensor_source"

    wrong_recipient = copy.deepcopy(base)
    wrong_sources = wrong_recipient["current_endpoint"]["parts"][
        "authorized_sensor_sources"
    ]
    wrong_sources[0]["source_public_agent_id"] = wrong_sources[1][
        "source_public_agent_id"
    ]
    wrong_sources[0]["source_presentation_key"] = wrong_sources[1][
        "source_presentation_key"
    ]
    with pytest.raises(ValidationError, match=r"recipient-base|recipient"):
        ReplaySharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(wrong_recipient)
        )

    recipient_as_shared = copy.deepcopy(base)
    recipient_as_shared["current_endpoint"]["parts"]["authorized_sensor_sources"][1] = (
        copy.deepcopy(sources[0])
    )
    recipient_as_shared["current_endpoint"]["parts"]["authorized_sensor_sources"][1][
        "source_kind"
    ] = "shared_sensor_source"
    with pytest.raises(ValidationError, match=r"unique|nonrecipient|authorized source"):
        ReplaySharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(recipient_as_shared)
        )

    opponent_as_shared = copy.deepcopy(base)
    opponent_id = frame.current_endpoint.action_axis.target_public_agent_id_by_action[6]
    assert opponent_id is not None
    opponent_source = opponent_as_shared["current_endpoint"]["parts"][
        "authorized_sensor_sources"
    ][-1]
    opponent_source["source_public_agent_id"] = opponent_id
    opponent_source["source_presentation_key"] = pov_presentation_key_v1(
        authority_session_id=frame.source.source_session_id,
        recipient_public_agent_id=frame.source.source_recipient_public_agent_id,
        public_agent_id=opponent_id,
    )
    with pytest.raises(
        ValidationError,
        match=r"canonical|authorized source|ally|teammate|authorized body",
    ):
        ReplaySharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(opponent_as_shared)
        )

    incoming_source = copy.deepcopy(base)
    delta = incoming_source["latest_events"]["deltas"][0]
    current_sources = incoming_source["current_endpoint"]["parts"][
        "agent_observation_provenance"
    ][1]["observation_sources"]
    delta["delta_kind"] = "observation_provenance_change"
    delta["start_observation_sources"] = copy.deepcopy(current_sources[:1])
    delta["successor_observation_sources"] = copy.deepcopy(current_sources)
    delta.pop("changed_dynamic_fields")
    delta.pop("start_observation")
    delta.pop("successor_observation")
    forged_start = delta["start_observation_sources"][0]
    forged_start["source_kind"] = "shared_sensor_source"
    forged_start["source_public_agent_id"] = opponent_id
    forged_start["source_presentation_key"] = pov_presentation_key_v1(
        authority_session_id=frame.source.source_session_id,
        recipient_public_agent_id=frame.source.source_recipient_public_agent_id,
        public_agent_id=opponent_id,
    )
    with pytest.raises(ValidationError, match=r"teammate|nonrecipient"):
        ReplaySharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(incoming_source)
        )


def test_no_shared_action_outcome_cue_joins_latest_transition(
    five_frames: _FiveFrames,
) -> None:
    frame = five_frames.replay_no_shared
    mismatch_payload = frame.model_dump(mode="json")
    action_row = mismatch_payload["latest_transition"]["action_rows"][0]
    assert action_row["submitted_action"] == action_row["accepted_action"]
    assert mismatch_payload["latest_events"]["cues"][0]["outcome"] == "accepted"
    action_row["submitted_action"]["move_action"] = 1
    with pytest.raises(ValidationError, match=r"outcome cue.*Latest Transition"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(mismatch_payload)
        )

    rejected_payload = frame.model_dump(mode="json")
    rejected_payload["latest_events"]["cues"][0]["outcome"] = "rejected"
    with pytest.raises(ValidationError, match=r"outcome cue.*Latest Transition"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(rejected_payload)
        )


def test_agent_status_sources_are_always_empty_even_with_valid_pov_key(
    five_frames: _FiveFrames,
) -> None:
    frame = five_frames.replay_no_shared
    payload = frame.model_dump(mode="json")
    agent_with_status = next(
        row
        for row in payload["current_endpoint"]["parts"]["scene"]["agents"]
        if row["statuses"]
    )
    status = agent_with_status["statuses"][0]
    source_public_id = next(
        row["public_agent_id"]
        for row in payload["current_endpoint"]["parts"]["scene"]["agents"]
        if row["class_id"] == status["source_class_id"]
    )
    status["direct_sources"] = [
        {
            "source_presentation_key": pov_presentation_key_v1(
                authority_session_id=frame.source.source_session_id,
                recipient_public_agent_id=(
                    frame.source.source_recipient_public_agent_id
                ),
                public_agent_id=source_public_id,
            ),
            "source_public_agent_id": source_public_id,
        }
    ]
    with pytest.raises(ValidationError, match="cannot disclose direct source"):
        ReplayNoSharedObsAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


@pytest.mark.parametrize(
    "frame_attribute,frame_type",
    (
        ("live_no_shared", LiveNoSharedObsAuthorizedPresentationFrameV1),
        (
            "replay_no_shared",
            ReplayNoSharedObsAuthorizedPresentationFrameV1,
        ),
        (
            "replay_shared",
            ReplaySharedObsAuthorizedPresentationFrameV1,
        ),
    ),
)
def test_agent_projected_mask_schema_identity_is_required(
    five_frames: _FiveFrames,
    frame_attribute: str,
    frame_type: type[
        LiveNoSharedObsAuthorizedPresentationFrameV1
        | ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1
    ],
) -> None:
    frame = cast(
        LiveNoSharedObsAuthorizedPresentationFrameV1
        | ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1,
        getattr(five_frames, frame_attribute),
    )
    for field_name in ("schema_id", "schema_version"):
        payload = frame.model_dump(mode="json")
        del payload["current_endpoint"]["parts"]["next_decision_action_mask"][
            field_name
        ]
        with pytest.raises(ValidationError, match="Field required"):
            frame_type.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "frame_attribute,frame_type",
    (
        (
            "replay_no_shared",
            ReplayNoSharedObsAuthorizedPresentationFrameV1,
        ),
        (
            "replay_shared",
            ReplaySharedObsAuthorizedPresentationFrameV1,
        ),
    ),
)
@pytest.mark.parametrize(
    "surface",
    (
        "movement_action_mask",
        "target_action_mask",
        "use_ultimate_action_mask",
        "target_use_ultimate_joint_mask",
    ),
)
def test_each_agent_inspection_legality_surface_mismatch_rejects(
    five_frames: _FiveFrames,
    frame_attribute: str,
    frame_type: type[
        ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1
    ],
    surface: str,
) -> None:
    frame = cast(
        ReplayNoSharedObsAuthorizedPresentationFrameV1
        | ReplaySharedObsAuthorizedPresentationFrameV1,
        getattr(five_frames, frame_attribute),
    )
    payload = frame.model_dump(mode="json")
    mask = payload["replay_inspection"]["decision_mask"]
    if surface == "target_use_ultimate_joint_mask":
        mask[surface][10][1] = not mask[surface][10][1]
    else:
        mask[surface][-1] = not mask[surface][-1]
    with pytest.raises(
        ValidationError,
        match=r"mask|marginal|legality|legal under",
    ):
        frame_type.model_validate_json(json.dumps(payload))


def test_json_number_semantics_and_non_numeric_coercions(
    five_frames: _FiveFrames,
) -> None:
    payload = five_frames.replay_oracle.model_dump(mode="json")
    payload["current_endpoint"]["scene"]["map"]["width"] = 20
    parsed = ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
        json.dumps(payload)
    )
    assert parsed.current_endpoint.scene.map.width == 20.0
    assert type(parsed.current_endpoint.scene.map.width) is float
    assert (
        parsed.current_endpoint.authorized_endpoint_digest_sha256
        == five_frames.replay_oracle.current_endpoint.authorized_endpoint_digest_sha256
    )

    for value in ("20", True):
        invalid = five_frames.replay_oracle.model_dump(mode="json")
        invalid["current_endpoint"]["scene"]["map"]["width"] = value
        with pytest.raises(ValidationError):
            ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
                json.dumps(invalid)
            )

    fractional_index = five_frames.replay_oracle.model_dump(mode="json")
    fractional_index["source"]["source_frame_index"] = 1.5
    with pytest.raises(ValidationError, match="int"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(fractional_index)
        )


@pytest.mark.parametrize(
    "frame_attribute,frame_type",
    (
        ("live_oracle", LiveOracleAuthorizedPresentationFrameV1),
        ("live_no_shared", LiveNoSharedObsAuthorizedPresentationFrameV1),
    ),
)
def test_live_inspection_envelope_rejects_stale_run_and_revision(
    five_frames: _FiveFrames,
    frame_attribute: str,
    frame_type: type[
        LiveOracleAuthorizedPresentationFrameV1
        | LiveNoSharedObsAuthorizedPresentationFrameV1
    ],
) -> None:
    frame = cast(
        LiveOracleAuthorizedPresentationFrameV1
        | LiveNoSharedObsAuthorizedPresentationFrameV1,
        getattr(five_frames, frame_attribute),
    )
    stale = frame.model_dump(mode="json")
    stale["source"]["source_run_generation"] += 1
    stale["source"]["source_revision"] += 1
    stale["source"]["source_authority_epoch"] += 1
    with pytest.raises(ValidationError, match=r"inspection envelope.*source epoch"):
        frame_type.model_validate_json(json.dumps(stale))

    rebound = copy.deepcopy(stale)
    rebound["live_inspection"]["source_run_generation"] = stale["source"][
        "source_run_generation"
    ]
    rebound["live_inspection"]["source_revision"] = stale["source"]["source_revision"]
    rebound["live_inspection"]["source_authority_epoch"] = stale["source"][
        "source_authority_epoch"
    ]
    if frame_attribute == "live_no_shared":
        rebound["researcher_space"]["source_run_generation"] = stale["source"][
            "source_run_generation"
        ]
        rebound["researcher_space"]["source_revision"] = stale["source"][
            "source_revision"
        ]
        rebound["researcher_space"]["source_authority_epoch"] = stale["source"][
            "source_authority_epoch"
        ]
    parsed = frame_type.model_validate_json(json.dumps(rebound))
    assert parsed.live_inspection.source_revision == parsed.source.source_revision
    assert (
        parsed.live_inspection.source_run_generation
        == parsed.source.source_run_generation
    )


def test_result_rejects_nested_subclass_and_list_model_construct_poison(
    five_frames: _FiveFrames,
) -> None:
    base = five_frames.replay_oracle
    agent = base.current_endpoint.scene.agents[0]
    poison_agent = _PoisonAuthorizedAgentV1(
        **{field.name: getattr(agent, field.name) for field in fields(agent)}
    )
    poison_scene = copy.copy(base.current_endpoint.scene)
    object.__setattr__(
        poison_scene,
        "agents",
        (poison_agent, *base.current_endpoint.scene.agents[1:]),
    )
    poison_endpoint = OracleAuthorizedCurrentEndpointV1.model_construct(
        endpoint_kind=base.current_endpoint.endpoint_kind,
        episode_id=base.current_endpoint.episode_id,
        frame_index=base.current_endpoint.frame_index,
        frame_id=base.current_endpoint.frame_id,
        simulator_step_count=base.current_endpoint.simulator_step_count,
        scene=poison_scene,
        identity_directory=base.current_endpoint.identity_directory,
        action_axis=base.current_endpoint.action_axis,
        authorized_endpoint_digest_sha256=(
            base.current_endpoint.authorized_endpoint_digest_sha256
        ),
    )
    poison_root = ReplayOracleAuthorizedPresentationFrameV1.model_construct(
        schema_version=base.schema_version,
        presentation_kind=base.presentation_kind,
        product_kind=base.product_kind,
        source=base.source,
        authority=base.authority,
        analysis_mode=base.analysis_mode,
        current_endpoint=poison_endpoint,
        latest_events=base.latest_events,
        latest_transition=base.latest_transition,
        upcoming_transition=base.upcoming_transition,
        technical_frame=base.technical_frame,
        replay_inspection=base.replay_inspection,
    )
    with pytest.raises(ValueError, match="exact runtime type"):
        canonical_authorized_endpoint_digest_sha256(poison_endpoint)
    with pytest.raises(ValueError, match="exact runtime type"):
        PresentationResourceResultV1(outcome="response", payload=poison_root)

    list_scene = copy.copy(base.current_endpoint.scene)
    object.__setattr__(
        list_scene,
        "agents",
        list(base.current_endpoint.scene.agents),
    )
    list_endpoint = OracleAuthorizedCurrentEndpointV1.model_construct(
        endpoint_kind=base.current_endpoint.endpoint_kind,
        episode_id=base.current_endpoint.episode_id,
        frame_index=base.current_endpoint.frame_index,
        frame_id=base.current_endpoint.frame_id,
        simulator_step_count=base.current_endpoint.simulator_step_count,
        scene=list_scene,
        identity_directory=base.current_endpoint.identity_directory,
        action_axis=base.current_endpoint.action_axis,
        authorized_endpoint_digest_sha256=(
            base.current_endpoint.authorized_endpoint_digest_sha256
        ),
    )
    list_root = ReplayOracleAuthorizedPresentationFrameV1.model_construct(
        schema_version=base.schema_version,
        presentation_kind=base.presentation_kind,
        product_kind=base.product_kind,
        source=base.source,
        authority=base.authority,
        analysis_mode=base.analysis_mode,
        current_endpoint=list_endpoint,
        latest_events=base.latest_events,
        latest_transition=base.latest_transition,
        upcoming_transition=base.upcoming_transition,
        technical_frame=base.technical_frame,
        replay_inspection=base.replay_inspection,
    )
    with pytest.raises(ValueError, match="replaced by a list"):
        PresentationResourceResultV1(outcome="response", payload=list_root)
    with pytest.raises(ValueError, match="replaced by a list"):
        canonical_authorized_endpoint_digest_sha256(list_endpoint)

    poison_source = _PoisonReplayOracleSourceIdentityV1(
        **base.source.model_dump(mode="python")
    )
    poison_source_root = ReplayOracleAuthorizedPresentationFrameV1.model_construct(
        schema_version=base.schema_version,
        presentation_kind=base.presentation_kind,
        product_kind=base.product_kind,
        source=poison_source,
        authority=base.authority,
        analysis_mode=base.analysis_mode,
        current_endpoint=base.current_endpoint,
        latest_events=base.latest_events,
        latest_transition=base.latest_transition,
        upcoming_transition=base.upcoming_transition,
        technical_frame=base.technical_frame,
        replay_inspection=base.replay_inspection,
    )
    with pytest.raises(ValueError, match="exact runtime type"):
        PresentationResourceResultV1(
            outcome="response",
            payload=poison_source_root,
        )


def test_endpoint_digest_is_stable_nonmutating_and_revalidates_surrogates(
    five_frames: _FiveFrames,
) -> None:
    endpoint = five_frames.replay_oracle.current_endpoint
    before = endpoint.model_dump_json()
    first = canonical_authorized_endpoint_digest_sha256(endpoint)
    second = canonical_authorized_endpoint_digest_sha256(endpoint)
    assert first == second == endpoint.authorized_endpoint_digest_sha256
    assert endpoint.model_dump_json() == before

    canonical_values = {
        name: getattr(endpoint, name) for name in type(endpoint).model_fields
    }

    missing_values = dict(canonical_values)
    del missing_values["frame_id"]
    missing = OracleAuthorizedCurrentEndpointV1.model_construct(**missing_values)
    with pytest.raises(ValueError, match="runtime fields"):
        canonical_authorized_endpoint_digest_sha256(missing)

    for scalar in ("1", True, 1.0):
        scalar_values = dict(canonical_values)
        scalar_values["frame_index"] = scalar
        scalar_poison = OracleAuthorizedCurrentEndpointV1.model_construct(
            **scalar_values
        )
        with pytest.raises((ValidationError, ValueError)):
            canonical_authorized_endpoint_digest_sha256(scalar_poison)

    extra = OracleAuthorizedCurrentEndpointV1.model_construct(**canonical_values)
    object.__setattr__(extra, "unexpected", True)
    with pytest.raises(ValueError, match="runtime fields"):
        canonical_authorized_endpoint_digest_sha256(extra)

    nonfinite_map = copy.copy(endpoint.scene.map)
    object.__setattr__(nonfinite_map, "width", float("nan"))
    nonfinite_scene = copy.copy(endpoint.scene)
    object.__setattr__(nonfinite_scene, "map", nonfinite_map)
    nonfinite_values = dict(canonical_values)
    nonfinite_values["scene"] = nonfinite_scene
    nonfinite = OracleAuthorizedCurrentEndpointV1.model_construct(**nonfinite_values)
    with pytest.raises((ValidationError, ValueError)):
        canonical_authorized_endpoint_digest_sha256(nonfinite)

    integer_map = copy.copy(endpoint.scene.map)
    object.__setattr__(integer_map, "width", 20)
    integer_scene = copy.copy(endpoint.scene)
    object.__setattr__(integer_scene, "map", integer_map)
    integer_values = dict(canonical_values)
    integer_values["scene"] = integer_scene
    integer_width = OracleAuthorizedCurrentEndpointV1.model_construct(**integer_values)
    with pytest.raises(ValueError, match="runtime type changed"):
        canonical_authorized_endpoint_digest_sha256(integer_width)


def test_result_wrapper_and_outcome_require_exact_runtime_types(
    five_frames: _FiveFrames,
) -> None:
    error = PresentationApiErrorV1(
        schema_version=1,
        error_code="audience_unavailable",
        message="Authorized presentation is unavailable for the active audience.",
    )
    with pytest.raises(TypeError, match="exact root"):
        _PoisonPresentationResourceResultV1(
            outcome="response",
            payload=five_frames.replay_oracle,
        )
    with pytest.raises(TypeError, match="exact root"):
        _PoisonPresentationResourceResultV1(
            outcome="audience_unavailable",
            payload=error,
        )
    with pytest.raises(TypeError, match="exact string"):
        PresentationResourceResultV1(
            outcome=cast(str, _OutcomeString("response")),  # pyright: ignore[reportArgumentType]
            payload=five_frames.replay_oracle,
        )


def test_result_pairings_error_payloads_and_top_level_poison_reject(
    five_frames: _FiveFrames,
) -> None:
    from scripts.dev.visual_debugger.replay_service import (
        PresentationResourceResultV1 as ReplayServicePresentationResourceResultV1,
    )

    assert ReplayServicePresentationResourceResultV1 is PresentationResourceResultV1
    error = PresentationApiErrorV1(
        schema_version=1,
        error_code="audience_unavailable",
        message="Authorized presentation is unavailable for the active audience.",
    )
    accepted_error = PresentationResourceResultV1(
        outcome="audience_unavailable",
        payload=error,
    )
    assert type(accepted_error.payload) is PresentationApiErrorV1
    assert accepted_error.payload.model_dump_json() == error.model_dump_json()

    with pytest.raises(TypeError, match="response requires"):
        PresentationResourceResultV1(outcome="response", payload=error)
    with pytest.raises(TypeError, match="unavailable presentation"):
        PresentationResourceResultV1(
            outcome="audience_unavailable",
            payload=five_frames.replay_oracle,
        )
    with pytest.raises(ValueError, match="unknown"):
        PresentationResourceResultV1(
            outcome=cast(
                Literal["response", "audience_unavailable"],
                "unknown",
            ),
            payload=five_frames.replay_oracle,
        )

    poison_error_subclass = _PoisonPresentationApiErrorV1(
        schema_version=1,
        error_code="audience_unavailable",
        message="Authorized presentation is unavailable for the active audience.",
    )
    with pytest.raises(TypeError, match="exact API error"):
        PresentationResourceResultV1(
            outcome="audience_unavailable",
            payload=poison_error_subclass,
        )

    poison_message = PresentationApiErrorV1.model_construct(
        schema_version=1,
        error_code="audience_unavailable",
        message=_OutcomeString(
            "Authorized presentation is unavailable for the active audience."
        ),
    )
    with pytest.raises(ValueError, match=r"unsupported|exact runtime type"):
        PresentationResourceResultV1(
            outcome="audience_unavailable",
            payload=poison_message,
        )

    missing_message = PresentationApiErrorV1.model_construct(
        schema_version=1,
        error_code="audience_unavailable",
    )
    with pytest.raises(ValueError, match="runtime fields"):
        PresentationResourceResultV1(
            outcome="audience_unavailable",
            payload=missing_message,
        )

    wrong_error = PresentationApiErrorV1.model_construct(
        schema_version=1,
        error_code="unknown",
        message="Authorized presentation is unavailable for the active audience.",
    )
    with pytest.raises(ValueError, match="failed full strict revalidation"):
        PresentationResourceResultV1(
            outcome="audience_unavailable",
            payload=wrong_error,
        )

    base = five_frames.replay_oracle
    root_values = {name: getattr(base, name) for name in type(base).model_fields}
    response_subclass = (
        _PoisonReplayOracleAuthorizedPresentationFrameV1.model_construct(**root_values)
    )
    with pytest.raises(TypeError, match="exact authorized frame root"):
        PresentationResourceResultV1(
            outcome="response",
            payload=response_subclass,
        )

    missing_root_values = dict(root_values)
    del missing_root_values["technical_frame"]
    missing_root = ReplayOracleAuthorizedPresentationFrameV1.model_construct(
        **missing_root_values
    )
    with pytest.raises(ValueError, match="runtime fields"):
        PresentationResourceResultV1(outcome="response", payload=missing_root)

    wrong_discriminator_values = dict(root_values)
    wrong_discriminator_values["presentation_kind"] = "live_oracle"
    wrong_discriminator = ReplayOracleAuthorizedPresentationFrameV1.model_construct(
        **wrong_discriminator_values
    )
    with pytest.raises(ValueError, match="failed full strict revalidation"):
        PresentationResourceResultV1(
            outcome="response",
            payload=wrong_discriminator,
        )

    coerced_values = dict(root_values)
    coerced_values["schema_version"] = "1"
    coerced_root = ReplayOracleAuthorizedPresentationFrameV1.model_construct(
        **coerced_values
    )
    with pytest.raises(ValueError, match="failed full strict revalidation"):
        PresentationResourceResultV1(outcome="response", payload=coerced_root)

    boolean_endpoint_values = {
        name: getattr(base.current_endpoint, name)
        for name in type(base.current_endpoint).model_fields
    }
    boolean_endpoint_values["frame_index"] = True
    boolean_endpoint = OracleAuthorizedCurrentEndpointV1.model_construct(
        **boolean_endpoint_values
    )
    boolean_root_values = dict(root_values)
    boolean_root_values["current_endpoint"] = boolean_endpoint
    boolean_root = ReplayOracleAuthorizedPresentationFrameV1.model_construct(
        **boolean_root_values
    )
    with pytest.raises(ValueError, match="runtime type changed"):
        PresentationResourceResultV1(outcome="response", payload=boolean_root)

    integer_map = copy.copy(base.current_endpoint.scene.map)
    object.__setattr__(integer_map, "width", 20)
    integer_scene = copy.copy(base.current_endpoint.scene)
    object.__setattr__(integer_scene, "map", integer_map)
    integer_endpoint_values = dict(boolean_endpoint_values)
    integer_endpoint_values["frame_index"] = base.current_endpoint.frame_index
    integer_endpoint_values["scene"] = integer_scene
    integer_endpoint = OracleAuthorizedCurrentEndpointV1.model_construct(
        **integer_endpoint_values
    )
    integer_root_values = dict(root_values)
    integer_root_values["current_endpoint"] = integer_endpoint
    integer_root = ReplayOracleAuthorizedPresentationFrameV1.model_construct(
        **integer_root_values
    )
    with pytest.raises(ValueError, match="runtime type changed"):
        PresentationResourceResultV1(outcome="response", payload=integer_root)

    extra_root = ReplayOracleAuthorizedPresentationFrameV1.model_construct(
        **root_values
    )
    object.__setattr__(extra_root, "unexpected", True)
    with pytest.raises(ValueError, match="runtime fields"):
        PresentationResourceResultV1(outcome="response", payload=extra_root)


def test_technical_frame_missing_extra_and_unrecorded_scale_reject(
    five_frames: _FiveFrames,
) -> None:
    payload = five_frames.replay_oracle.model_dump(mode="json")
    del payload["technical_frame"]["incoming_transition_id"]
    with pytest.raises(ValidationError, match="Field required"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )

    payload = five_frames.replay_oracle.model_dump(mode="json")
    payload["technical_frame"]["revision"] = 1
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )

    payload = five_frames.replay_oracle.model_dump(mode="json")
    payload["technical_frame"]["recorded_ordinary_movement_distance_scale"] += 0.5
    with pytest.raises(ValidationError, match="Technical Frame"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )
