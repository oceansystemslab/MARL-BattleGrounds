"""Focused CP2.4 outgoing replay-inspection and live-draft proofs."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    neutral_action,
)

from marl_battlegrounds.core.types import Action
from marl_battlegrounds.evaluation import pov as pov_module
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    ActionAcceptanceFactsV1,
    EvaluationFrameV1,
    EvaluationTransitionV1,
    JointActionV1,
    TransitionFactsV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.pov import (
    ACTOR_POV_CONTENT_SCHEMA_ID,
    ACTOR_POV_SCHEMA_VERSION,
    ActorPovAcceptedActionV1,
    ActorPovActionMaskV1,
    ActorPovCurrentSliceV1,
    ActorPovEpisodeCompletionV1,
    ActorPovFrameV1,
    ActorPovReplayContentV1,
    ActorPovTransitionV1,
    build_actor_pov_current_slice_v1,
)
from marl_battlegrounds.rendering import authorized_inspection as inspection_module
from marl_battlegrounds.rendering import evaluation_adapter as evaluation_adapter_module
from marl_battlegrounds.rendering import evaluation_wire_features as wire
from marl_battlegrounds.rendering.authorized_inspection import (
    AuthorizedAxisOnlyTargetActionV1,
    AuthorizedDecisionMaskV1,
    AuthorizedTargetActionV1,
    AuthorizedVisibleTargetActionV1,
    LiveDraftActionTupleV1,
    LiveDraftInspectionPresentationV1,
    LiveDraftLegalityV1,
    NoSharedObsReplayTransitionReferenceV1,
    OracleReplayTransitionReferenceV1,
    ReplayInspectionPresentationV1,
    SharedObsReplayTransitionReferenceV1,
    build_live_no_shared_obs_draft_inspection_v1,
    build_live_oracle_draft_inspection_v1,
    build_live_shared_obs_draft_inspection_v1,
    build_replay_no_shared_obs_inspection_v1,
    build_replay_oracle_inspection_v1,
    build_replay_shared_obs_inspection_v1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    NoSharedObsAuthorizedScenePartsV1,
    SharedObsAuthorizedScenePartsV1,
    build_no_shared_obs_authorized_scene_v1,
    build_shared_obs_authorized_scene_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AuthorizedBattlefieldSceneV1,
    build_replay_oracle_presentation_parts_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    EvaluationScenePresentationStateV1,
    SharedObsSourceMaterialProjectionV1,
    build_researcher_analyzer_projection_v2,
    build_shared_obs_source_material_projection_v1,
    build_status_source_evidence_index_v2,
)
from marl_battlegrounds.rendering.evaluation_wire_features import AGENT_FEATURE_X_V1
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovProjectionIndexV1,
    build_actor_pov_projection_index_v1,
)
from marl_battlegrounds.rendering.vocabulary import (
    status_sort_key,
    status_token_id_from_catalog_status_id,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class _InspectionCases:
    no_shared: CapturedEvaluationTrajectory
    shared: CapturedEvaluationTrajectory
    oracle_scenes: tuple[AuthorizedBattlefieldSceneV1, ...]


def _actions() -> tuple[Action, ...]:
    base = neutral_action()
    basic = Action(
        move=base.move,
        select_target=base.select_target.at[2].set(2),
        use_ultimate=base.use_ultimate,
    )
    ultimate = Action(
        move=base.move,
        select_target=base.select_target,
        use_ultimate=base.use_ultimate.at[0].set(1),
    )
    rejected = Action(
        move=base.move,
        select_target=base.select_target.at[1].set(99),
        use_ultimate=base.use_ultimate,
    )
    return basic, ultimate, rejected


def _oracle_scenes(
    trajectory: CapturedEvaluationTrajectory,
) -> tuple[AuthorizedBattlefieldSceneV1, ...]:
    status_index = build_status_source_evidence_index_v2(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
    )
    scenes: list[AuthorizedBattlefieldSceneV1] = []
    final_index = len(trajectory.transitions)
    for frame_index, frame in enumerate(trajectory.frames):
        incoming = (
            None
            if frame_index == 0
            else EvaluationTransitionViewV1(
                context=trajectory.context,
                start_frame=trajectory.frames[frame_index - 1],
                transition=trajectory.transitions[frame_index - 1],
                successor_frame=frame,
            )
        )
        raw = build_researcher_analyzer_projection_v2(
            trajectory.context,
            frame,
            transition_view=incoming,
            presentation=EvaluationScenePresentationStateV1(
                controlled_global_slot=0,
                selected_global_slot=0,
                show_ranges=True,
            ),
            status_source_evidence_state=status_index.state_for_frame(frame_index),
        )
        parts = build_replay_oracle_presentation_parts_v1(
            trajectory.context,
            raw.scene,
            raw.incoming_events,
            authority_session_id="cp2-4-oracle-authority",
            final_frame_index=final_index,
            selected_internal_slot=None,
            outgoing_transition=None,
        )
        scenes.append(parts.current_scene)
    return tuple(scenes)


@pytest.fixture(scope="module")
def inspection_cases() -> _InspectionCases:
    actions = _actions()
    no_shared = captured_evaluation_trajectory(
        transition_count=3,
        expected_horizon=3,
        actions=actions,
    )
    shared = captured_evaluation_trajectory(
        transition_count=3,
        expected_horizon=3,
        execution_information_mode="shared_obs",
        actions=actions,
    )
    return _InspectionCases(
        no_shared=no_shared,
        shared=shared,
        oracle_scenes=_oracle_scenes(no_shared),
    )


def _pov_slice(
    trajectory: CapturedEvaluationTrajectory,
    *,
    actor_slot: int,
    frame_index: int,
) -> ActorPovCurrentSliceV1:
    incoming = (
        None
        if frame_index == 0
        else EvaluationTransitionViewV1(
            context=trajectory.context,
            start_frame=trajectory.frames[frame_index - 1],
            transition=trajectory.transitions[frame_index - 1],
            successor_frame=trajectory.frames[frame_index],
        )
    )
    return build_actor_pov_current_slice_v1(
        trajectory.context,
        trajectory.frames[frame_index],
        global_slot=actor_slot,
        incoming_transition_view=incoming,
    )


def _pov_index(
    trajectory: CapturedEvaluationTrajectory,
    *,
    actor_slot: int,
) -> ActorPovProjectionIndexV1:
    slices = tuple(
        _pov_slice(
            trajectory,
            actor_slot=actor_slot,
            frame_index=frame_index,
        )
        for frame_index in range(len(trajectory.frames))
    )
    transitions = tuple(
        cast(ActorPovTransitionV1, row.incoming_transition) for row in slices[1:]
    )
    first = slices[0]
    content_payload: dict[str, object] = {
        "schema_id": ACTOR_POV_CONTENT_SCHEMA_ID,
        "schema_version": ACTOR_POV_SCHEMA_VERSION,
        "content_id": (f"{first.episode_id}:actor-pov:{first.public_agent_id}:content"),
        "episode_id": first.episode_id,
        "selected_global_slot": first.selected_global_slot,
        "selected_team_local_slot": first.selected_team_local_slot,
        "public_agent_id": first.public_agent_id,
        "configured_team_id": first.configured_team_id,
        "class_id": first.class_id,
        "observation_materialization": "exact_no_shared_obs_actor_input",
        "axis_mapping": first.axis_mapping,
        "completion": ActorPovEpisodeCompletionV1(
            completion_state="complete",
            expected_transition_count=len(transitions),
            captured_transition_count=len(transitions),
            terminated=False,
            truncated=False,
            completion_bases=("declared_horizon",),
        ),
        "frames": tuple(row.frame for row in slices),
        "transitions": transitions,
    }
    content = ActorPovReplayContentV1.model_validate(
        {
            **content_payload,
            "canonical_digest_sha256": canonical_digest_sha256(content_payload),
        }
    )
    return build_actor_pov_projection_index_v1(content)


def _no_shared_current(
    trajectory: CapturedEvaluationTrajectory,
    source: ActorPovProjectionIndexV1,
    *,
    frame_index: int,
    authority: str = "cp2-4-no-shared-authority",
) -> NoSharedObsAuthorizedScenePartsV1:
    return build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id=authority,
        frame_index=frame_index,
    )


def _shared_current(
    trajectory: CapturedEvaluationTrajectory,
    *,
    recipient_slot: int,
    frame_index: int,
    authority: str = "cp2-4-shared-authority",
) -> tuple[
    SharedObsAuthorizedScenePartsV1,
    SharedObsSourceMaterialProjectionV1,
]:
    transition_view = (
        None
        if frame_index == 0
        else EvaluationTransitionViewV1(
            context=trajectory.context,
            start_frame=trajectory.frames[frame_index - 1],
            transition=trajectory.transitions[frame_index - 1],
            successor_frame=trajectory.frames[frame_index],
        )
    )
    projections = {
        roster.global_slot: build_shared_obs_source_material_projection_v1(
            trajectory.context,
            trajectory.frames[frame_index],
            selected_global_slot=roster.global_slot,
            transition_view=transition_view,
        )
        for roster in trajectory.context.roster
        if roster.configured_active
    }
    recipient = projections[recipient_slot]
    current = build_shared_obs_authorized_scene_v1(
        recipient,
        all_active_nonrecipient_source_material=tuple(
            projection
            for slot, projection in projections.items()
            if slot != recipient_slot
        ),
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id=authority,
    )
    return current, recipient


def _trajectory_with_public_effects(
    trajectory: CapturedEvaluationTrajectory,
    *,
    global_slot: int,
    active_status_channels: tuple[int, ...],
    active_aura_ids: tuple[str, ...],
) -> CapturedEvaluationTrajectory:
    """Install one coherent public multi-effect row at frame zero."""
    frame = trajectory.frames[0]
    context = trajectory.context
    catalog = context.static_mechanics_catalog
    active_statuses = set(active_status_channels)
    active_auras = set(active_aura_ids)
    if len(active_statuses) != len(active_status_channels):
        raise AssertionError("test status channels must be unique")
    if len(active_auras) != len(active_aura_ids):
        raise AssertionError("test aura IDs must be unique")

    aura_by_id = {row.aura_id: row for row in catalog.aura_mechanics}

    def with_effects(row: tuple[float, ...]) -> tuple[float, ...]:
        values = list(row)
        for channel, mechanic in enumerate(catalog.status_channels):
            duration_column = wire.AGENT_STATUS_REMAINING_DURATION_COLUMN_BY_CHANNEL_V1[
                channel
            ]
            values[duration_column] = 1.0 if channel in active_statuses else 0.0
            magnitude_column = wire.AGENT_STATUS_ACTIVE_MAGNITUDE_COLUMN_BY_CHANNEL_V1[
                channel
            ]
            if channel in active_statuses and magnitude_column is not None:
                assert mechanic.magnitude is not None
                values[magnitude_column] = mechanic.magnitude
        values[wire.AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1] = (
            aura_by_id["mage_damage_amplification"].per_emitter_multiplier
            if "mage_damage_amplification" in active_auras
            else 1.0
        )
        values[wire.AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1] = (
            aura_by_id["warrior_damage_mitigation"].per_emitter_multiplier
            if "warrior_damage_mitigation" in active_auras
            else 1.0
        )
        return tuple(values)

    base_payload = frame.base_observation.model_dump(mode="python")
    self_rows = list(cast(tuple[tuple[float, ...], ...], base_payload["self_features"]))
    self_rows[global_slot] = with_effects(self_rows[global_slot])
    base_payload["self_features"] = tuple(self_rows)

    target = context.roster[global_slot]
    for relation, mask_name, row_name in (
        ("ally", "ally_visibility_mask", "ally_unit_features"),
        ("enemy", "enemy_visibility_mask", "enemy_unit_features"),
    ):
        masks = cast(tuple[tuple[bool, ...], ...], base_payload[mask_name])
        observer_rows = list(
            cast(tuple[tuple[tuple[float, ...], ...], ...], base_payload[row_name])
        )
        for observer_slot, observer in enumerate(context.roster):
            expected_relation = (
                "ally"
                if observer.configured_team_id == target.configured_team_id
                else "enemy"
            )
            if (
                not observer.configured_active
                or relation != expected_relation
                or not masks[observer_slot][target.team_local_slot]
            ):
                continue
            rows = list(observer_rows[observer_slot])
            rows[target.team_local_slot] = with_effects(rows[target.team_local_slot])
            observer_rows[observer_slot] = tuple(rows)
        base_payload[row_name] = tuple(observer_rows)

    snapshot_payload = frame.snapshot.model_dump(mode="python")
    slow_rows = list(
        cast(tuple[tuple[int, ...], ...], snapshot_payload["slow_durations"])
    )
    slow_rows[global_slot] = tuple(
        1 if channel in active_statuses else 0 for channel in range(3)
    )
    snapshot_payload["slow_durations"] = tuple(slow_rows)
    stun_rows = list(
        cast(tuple[tuple[int, ...], ...], snapshot_payload["stun_durations"])
    )
    stun_rows[global_slot] = tuple(
        1 if channel + 3 in active_statuses else 0 for channel in range(3)
    )
    snapshot_payload["stun_durations"] = tuple(stun_rows)
    for field_name, channel in (
        ("rogue_poison_anti_heal_durations", 6),
        ("mage_burst_damage_amplification_durations", 7),
        ("priest_blessing_of_freedom_slow_floor_durations", 8),
    ):
        values = list(cast(tuple[int, ...], snapshot_payload[field_name]))
        values[global_slot] = 1 if channel in active_statuses else 0
        snapshot_payload[field_name] = tuple(values)

    frame_payload = frame.model_dump(mode="python")
    frame_payload["base_observation"] = base_payload
    frame_payload["snapshot"] = snapshot_payload
    changed_frame = EvaluationFrameV1.model_validate(frame_payload)
    return CapturedEvaluationTrajectory(
        context=context,
        frames=(changed_frame,),
        transitions=(),
    )


def _canonical_bytes(value: object) -> bytes:
    return TypeAdapter(type(value)).dump_json(value)


def _json_payload(value: object) -> dict[str, object]:
    payload = json.loads(_canonical_bytes(value))
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


@pytest.mark.parametrize(
    ("active_status_channels", "active_aura_ids"),
    (
        ((0, 3), ()),
        ((4, 6, 7), ("mage_damage_amplification",)),
        ((0, 3, 8), ("warrior_damage_mitigation",)),
        (
            tuple(range(9)),
            (
                "mage_damage_amplification",
                "warrior_damage_mitigation",
            ),
        ),
    ),
    ids=("charge", "control-and-burst", "charge-and-freedom", "all-effects"),
)
def test_live_and_replay_agent_effect_rows_match_local_oracle_order(
    inspection_cases: _InspectionCases,
    active_status_channels: tuple[int, ...],
    active_aura_ids: tuple[str, ...],
) -> None:
    global_slot = 1
    base_no_shared = CapturedEvaluationTrajectory(
        context=inspection_cases.no_shared.context,
        frames=(inspection_cases.no_shared.frames[0],),
        transitions=(),
    )
    base_shared = CapturedEvaluationTrajectory(
        context=inspection_cases.shared.context,
        frames=(inspection_cases.shared.frames[0],),
        transitions=(),
    )
    no_shared = _trajectory_with_public_effects(
        base_no_shared,
        global_slot=global_slot,
        active_status_channels=active_status_channels,
        active_aura_ids=active_aura_ids,
    )
    shared = _trajectory_with_public_effects(
        base_shared,
        global_slot=global_slot,
        active_status_channels=active_status_channels,
        active_aura_ids=active_aura_ids,
    )
    public_agent_id = no_shared.context.roster[global_slot].public_agent_id
    catalog = no_shared.context.static_mechanics_catalog
    expected_status_ids = tuple(
        catalog.status_channels[channel].status_id
        for channel in sorted(
            active_status_channels,
            key=lambda channel: status_sort_key(
                status_token_id_from_catalog_status_id(
                    catalog.status_channels[channel].status_id
                )
            ),
        )
    )
    expected_aura_ids = tuple(
        aura_id
        for aura_id in (
            "mage_damage_amplification",
            "warrior_damage_mitigation",
        )
        if aura_id in active_aura_ids
    )

    oracle_no_shared = next(
        row
        for row in _oracle_scenes(no_shared)[0].agents
        if row.public_agent_id == public_agent_id
    )
    live_slice = _pov_slice(no_shared, actor_slot=global_slot, frame_index=0)
    live_no_shared = build_no_shared_obs_authorized_scene_v1(
        live_slice,
        public_catalog=catalog,
        authority_session_id="effect-order-live-no-shared",
    )
    replay_source = _pov_index(
        inspection_cases.no_shared,
        actor_slot=global_slot,
    )
    replay_frames = list(replay_source.content.frames)
    replay_frames[0] = live_slice.frame
    replay_transitions = _rederive_pov_transitions(
        replay_source,
        tuple(replay_frames),
        replay_source.content.transitions,
    )
    replay_source = _rebuild_pov_index(
        replay_source,
        frames=tuple(replay_frames),
        transitions=replay_transitions,
    )
    replay_no_shared = _no_shared_current(
        no_shared,
        replay_source,
        frame_index=0,
        authority="effect-order-replay-no-shared",
    )
    shared_current, _ = _shared_current(
        shared,
        recipient_slot=global_slot,
        frame_index=0,
        authority="effect-order-replay-shared",
    )
    oracle_shared = next(
        row
        for row in _oracle_scenes(shared)[0].agents
        if row.public_agent_id == public_agent_id
    )

    local_rows = (
        next(
            row
            for row in live_no_shared.scene.agents
            if row.public_agent_id == public_agent_id
        ),
        next(
            row
            for row in replay_no_shared.scene.agents
            if row.public_agent_id == public_agent_id
        ),
        next(
            row
            for row in shared_current.scene.agents
            if row.public_agent_id == public_agent_id
        ),
    )
    assert tuple(row.status_id for row in oracle_no_shared.statuses) == (
        expected_status_ids
    )
    assert tuple(row.aura_id for row in oracle_no_shared.aura_modifiers) == (
        expected_aura_ids
    )
    assert oracle_shared.statuses == oracle_no_shared.statuses
    assert oracle_shared.aura_modifiers == oracle_no_shared.aura_modifiers
    for local in local_rows:
        assert local.statuses == oracle_no_shared.statuses
        assert local.aura_modifiers == oracle_no_shared.aura_modifiers


def _rebuild_pov_index(
    source: ActorPovProjectionIndexV1,
    *,
    frames: tuple[object, ...] | None = None,
    transitions: tuple[object, ...] | None = None,
) -> ActorPovProjectionIndexV1:
    payload = source.content.model_dump(mode="python")
    payload["frames"] = source.content.frames if frames is None else frames
    payload["transitions"] = (
        source.content.transitions if transitions is None else transitions
    )
    payload.pop("canonical_digest_sha256")
    content = ActorPovReplayContentV1.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )
    return build_actor_pov_projection_index_v1(content)


def _rederive_pov_transitions(
    source: ActorPovProjectionIndexV1,
    frames: tuple[ActorPovFrameV1, ...],
    transitions: tuple[ActorPovTransitionV1, ...],
) -> tuple[ActorPovTransitionV1, ...]:
    derive_cues = vars(pov_module)["_derive_cues"]
    result: list[ActorPovTransitionV1] = []
    for index, transition in enumerate(transitions):
        payload = transition.model_dump(mode="python")
        payload["cues"] = derive_cues(
            episode_id=source.content.episode_id,
            public_agent_id=source.content.public_agent_id,
            transition_index=index,
            team_local_slot=source.content.selected_team_local_slot,
            start_frame=frames[index],
            successor_frame=frames[index + 1],
            has_any_rejection=(
                transition.submitted_action_tuple_is_out_of_domain
                or transition.in_domain_move_action_is_rejected
                or transition.in_domain_combat_action_pair_is_rejected
            ),
            terminated=transition.terminated,
            truncated=transition.truncated,
            public_end_reason=transition.public_end_reason,
        )
        result.append(ActorPovTransitionV1.model_validate(payload))
    return tuple(result)


def _pov_frame_with_shifted_self_position(
    frame: ActorPovFrameV1,
    *,
    delta_x: float,
) -> ActorPovFrameV1:
    payload = frame.model_dump(mode="python")
    self_features = list(frame.self_features)
    self_features[AGENT_FEATURE_X_V1] += delta_x
    payload["self_features"] = tuple(self_features)
    return ActorPovFrameV1.model_validate(payload)


def _different_valid_pov_mask(mask: ActorPovActionMaskV1) -> ActorPovActionMaskV1:
    payload = mask.model_dump(mode="python")
    joint = [list(row) for row in mask.select_target_use_ultimate_joint]
    target, ultimate = next(
        (target, ultimate)
        for target, row in enumerate(joint)
        for ultimate, allowed in enumerate(row)
        if not allowed
    )
    joint[target][ultimate] = True
    payload["select_target_use_ultimate_joint"] = tuple(tuple(row) for row in joint)
    payload["select_target"] = tuple(any(row) for row in joint)
    payload["use_ultimate"] = tuple(
        any(joint[target][ultimate] for target in range(len(joint)))
        for ultimate in range(2)
    )
    return ActorPovActionMaskV1.model_validate(payload)


def _pov_mask_with_legal_pair(
    mask: ActorPovActionMaskV1,
    *,
    target_action: int,
    ultimate_action: int,
) -> ActorPovActionMaskV1:
    payload = mask.model_dump(mode="python")
    joint = [list(row) for row in mask.select_target_use_ultimate_joint]
    joint[target_action][ultimate_action] = True
    payload["select_target_use_ultimate_joint"] = tuple(tuple(row) for row in joint)
    payload["select_target"] = tuple(any(row) for row in joint)
    payload["use_ultimate"] = tuple(
        any(joint[target][ultimate] for target in range(len(joint)))
        for ultimate in range(2)
    )
    return ActorPovActionMaskV1.model_validate(payload)


def _assert_exact_decision_mask(
    projected: AuthorizedDecisionMaskV1,
    source: ActorPovActionMaskV1,
) -> None:
    assert projected.movement_action_mask == source.move
    assert projected.target_action_mask == source.select_target
    assert projected.use_ultimate_action_mask == source.use_ultimate
    assert (
        projected.target_use_ultimate_joint_mask
        == source.select_target_use_ultimate_joint
    )


def _recursive_pairs(value: object) -> tuple[tuple[str, object], ...]:
    pairs: list[tuple[str, object]] = []

    def visit(item: object) -> None:
        if type(item) is dict:
            mapping = cast(dict[str, object], item)
            for key, child in mapping.items():
                pairs.append((key, child))
                visit(child)
        elif type(item) is list:
            for child in cast(list[object], item):
                visit(child)

    visit(value)
    return tuple(pairs)


def _replace_scene_actor_position(
    current: NoSharedObsAuthorizedScenePartsV1 | SharedObsAuthorizedScenePartsV1,
    *,
    public_agent_id: str,
) -> NoSharedObsAuthorizedScenePartsV1 | SharedObsAuthorizedScenePartsV1:
    agents = tuple(
        replace(
            row,
            position=(row.position[0] + 0.25, row.position[1]),
        )
        if row.public_agent_id == public_agent_id
        else row
        for row in current.scene.agents
    )
    aura_fields = tuple(
        replace(
            row,
            center=(row.center[0] + 0.25, row.center[1]),
        )
        if row.source_public_agent_id == public_agent_id
        else row
        for row in current.scene.aura_fields
    )
    scene = replace(current.scene, agents=agents, aura_fields=aura_fields)
    return replace(current, scene=scene)


def _oracle_corpse_frame_and_scene(
    frame: EvaluationFrameV1,
    scene: AuthorizedBattlefieldSceneV1,
    *,
    public_agent_id: str,
    internal_slot: int,
) -> tuple[EvaluationFrameV1, AuthorizedBattlefieldSceneV1]:
    snapshot = frame.snapshot
    snapshot_payload = snapshot.model_dump(mode="python")
    alive = list(cast(tuple[bool, ...], snapshot_payload["alive_mask"]))
    alive[internal_slot] = False
    snapshot_payload["alive_mask"] = tuple(alive)
    corpse_snapshot = type(snapshot).model_validate(snapshot_payload)
    frame_payload = frame.model_dump(mode="python")
    frame_payload["snapshot"] = corpse_snapshot
    corpse_frame = EvaluationFrameV1.model_validate(frame_payload)

    agents = tuple(
        replace(agent, life_state="corpse")
        if agent.public_agent_id == public_agent_id
        else agent
        for agent in scene.agents
    )
    aura_fields = tuple(
        replace(field, source_alive=False)
        if field.source_public_agent_id == public_agent_id
        else field
        for field in scene.aura_fields
    )
    spawn_pads = tuple(
        replace(pad, currently_alive=False)
        if pad.assigned_public_agent_id == public_agent_id
        else pad
        for pad in scene.spawn_pads
    )
    return corpse_frame, replace(
        scene,
        agents=agents,
        aura_fields=aura_fields,
        spawn_pads=spawn_pads,
    )


def _oracle(
    cases: _InspectionCases,
    *,
    frame_index: int,
    actor_slot: int | None,
) -> ReplayInspectionPresentationV1 | None:
    outgoing = (
        None
        if actor_slot is None or frame_index == len(cases.no_shared.transitions)
        else cases.no_shared.transitions[frame_index]
    )
    return build_replay_oracle_inspection_v1(
        cases.no_shared.context,
        cases.no_shared.frames[frame_index],
        cases.oracle_scenes[frame_index],
        inspection_internal_slot=actor_slot,
        outgoing_transition=outgoing,
        final_frame_index=len(cases.no_shared.transitions),
    )


def test_oracle_frame_zero_middle_final_and_exact_lane_truth(
    inspection_cases: _InspectionCases,
) -> None:
    basic = _oracle(inspection_cases, frame_index=0, actor_slot=2)
    ultimate = _oracle(inspection_cases, frame_index=1, actor_slot=0)
    no_combat = _oracle(inspection_cases, frame_index=2, actor_slot=1)
    assert basic is not None and ultimate is not None and no_combat is not None
    assert basic.outgoing_transition_index == 0
    assert basic.combat_lane == "basic"
    assert basic.accepted_action.target_action == 2
    assert ultimate.outgoing_transition_index == 1
    assert ultimate.combat_lane == "ultimate"
    assert ultimate.accepted_action.target_action == 0
    assert no_combat.combat_lane == "none"
    assert no_combat.submitted_action.target_action == 99
    assert no_combat.accepted_action.target_action == 0
    assert no_combat.route_display_basis == "accepted_action"

    assert _oracle(inspection_cases, frame_index=3, actor_slot=0) is None
    assert _oracle(inspection_cases, frame_index=1, actor_slot=None) is None


def test_oracle_exact_mask_target_axis_and_current_anchor(
    inspection_cases: _InspectionCases,
) -> None:
    result = _oracle(inspection_cases, frame_index=0, actor_slot=2)
    assert result is not None
    mask = result.decision_mask
    assert len(mask.movement_action_mask) == 9
    assert len(mask.target_actions) == 11
    assert len(mask.use_ultimate_action_mask) == 2
    assert tuple(len(row) for row in mask.target_use_ultimate_joint_mask) == (2,) * 11
    assert mask.target_action_mask == tuple(
        any(row) for row in mask.target_use_ultimate_joint_mask
    )
    assert (
        result.actor_anchor
        == inspection_cases.no_shared.frames[0].snapshot.agent_positions[2]
    )
    assert any(
        isinstance(row, AuthorizedAxisOnlyTargetActionV1) for row in mask.target_actions
    )
    for row in mask.target_actions:
        if isinstance(row, AuthorizedAxisOnlyTargetActionV1):
            assert not hasattr(row, "target_presentation_key")
            assert not hasattr(row, "target_anchor")


def test_oracle_no_shared_and_shared_masks_equal_exact_recorded_m_n_values(
    inspection_cases: _InspectionCases,
) -> None:
    oracle = _oracle(inspection_cases, frame_index=0, actor_slot=2)
    assert oracle is not None
    oracle_mask = inspection_cases.no_shared.frames[0].action_mask
    assert oracle.decision_mask.movement_action_mask == oracle_mask.move_mask[2]
    assert oracle.decision_mask.target_action_mask == oracle_mask.select_target_mask[2]
    assert (
        oracle.decision_mask.use_ultimate_action_mask
        == oracle_mask.use_ultimate_mask[2]
    )
    assert (
        oracle.decision_mask.target_use_ultimate_joint_mask
        == oracle_mask.select_target_use_ultimate_joint_mask[2]
    )

    no_shared_source = _pov_index(inspection_cases.no_shared, actor_slot=2)
    no_shared_current = _no_shared_current(
        inspection_cases.no_shared,
        no_shared_source,
        frame_index=0,
    )
    no_shared = build_replay_no_shared_obs_inspection_v1(
        no_shared_source,
        no_shared_current,
    )
    assert no_shared is not None
    _assert_exact_decision_mask(
        no_shared.decision_mask,
        no_shared_source.content.frames[0].action_mask,
    )

    shared_current, shared_source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    shared = build_replay_shared_obs_inspection_v1(
        shared_current,
        shared_source,
        authorized_recipient_global_slot=0,
        outgoing_transition=inspection_cases.shared.transitions[1],
        final_frame_index=3,
    )
    assert shared is not None
    _assert_exact_decision_mask(
        shared.decision_mask,
        shared_source.base_sensor_frame.action_mask,
    )


def test_agent_builders_reject_wrong_current_owner_mask(
    inspection_cases: _InspectionCases,
) -> None:
    no_shared_source = _pov_index(inspection_cases.no_shared, actor_slot=2)
    no_shared_current = _no_shared_current(
        inspection_cases.no_shared,
        no_shared_source,
        frame_index=0,
    )
    no_shared_wrong_mask = replace(
        no_shared_current,
        next_decision_action_mask=_different_valid_pov_mask(
            no_shared_current.next_decision_action_mask
        ),
    )
    with pytest.raises(ValueError, match="recipient frame"):
        build_replay_no_shared_obs_inspection_v1(
            no_shared_source,
            no_shared_wrong_mask,
        )

    shared_current, shared_source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    shared_wrong_mask = replace(
        shared_current,
        next_decision_action_mask=_different_valid_pov_mask(
            shared_current.next_decision_action_mask
        ),
    )
    with pytest.raises(ValueError, match="do not join recipient"):
        build_replay_shared_obs_inspection_v1(
            shared_wrong_mask,
            shared_source,
            authorized_recipient_global_slot=0,
            outgoing_transition=inspection_cases.shared.transitions[1],
            final_frame_index=3,
        )

    oracle = _oracle(inspection_cases, frame_index=0, actor_slot=2)
    assert oracle is not None
    payload = _json_payload(oracle)
    decision = cast(dict[str, object], payload["decision_mask"])
    decision["owner_public_agent_id"] = "agent-slot-0"
    with pytest.raises(ValidationError, match="owner"):
        TypeAdapter(ReplayInspectionPresentationV1).validate_json(json.dumps(payload))


def test_oracle_rejects_wrong_epoch_and_every_invalid_selector(
    inspection_cases: _InspectionCases,
) -> None:
    with pytest.raises(ValueError, match="displayed s_n/m_n"):
        build_replay_oracle_inspection_v1(
            inspection_cases.no_shared.context,
            inspection_cases.no_shared.frames[1],
            inspection_cases.oracle_scenes[1],
            inspection_internal_slot=0,
            outgoing_transition=inspection_cases.no_shared.transitions[0],
            final_frame_index=3,
        )
    for invalid_slot in (3, 10):
        with pytest.raises(ValueError):
            build_replay_oracle_inspection_v1(
                inspection_cases.no_shared.context,
                inspection_cases.no_shared.frames[3],
                inspection_cases.oracle_scenes[3],
                inspection_internal_slot=invalid_slot,
                outgoing_transition=None,
                final_frame_index=3,
            )
    with pytest.raises(ValueError, match="configured active"):
        build_replay_oracle_inspection_v1(
            inspection_cases.no_shared.context,
            inspection_cases.no_shared.frames[1],
            inspection_cases.oracle_scenes[1],
            inspection_internal_slot=3,
            outgoing_transition=inspection_cases.no_shared.transitions[1],
            final_frame_index=3,
        )


def test_oracle_configured_active_corpse_produces_nonfinal_inspection_and_is_final_safe(
    inspection_cases: _InspectionCases,
) -> None:
    nonfinal_frame, nonfinal_scene = _oracle_corpse_frame_and_scene(
        inspection_cases.no_shared.frames[2],
        inspection_cases.oracle_scenes[2],
        public_agent_id=inspection_cases.no_shared.context.roster[1].public_agent_id,
        internal_slot=1,
    )
    nonfinal = build_replay_oracle_inspection_v1(
        inspection_cases.no_shared.context,
        nonfinal_frame,
        nonfinal_scene,
        inspection_internal_slot=1,
        outgoing_transition=inspection_cases.no_shared.transitions[2],
        final_frame_index=3,
    )
    assert nonfinal is not None
    assert nonfinal.actor_public_agent_id == (
        inspection_cases.no_shared.context.roster[1].public_agent_id
    )
    assert nonfinal.actor_anchor == nonfinal_frame.snapshot.agent_positions[1]
    assert nonfinal.accepted_action.target_action == 0
    assert nonfinal.decision_mask.target_use_ultimate_joint_mask[0][0] is True

    final_index = len(inspection_cases.no_shared.transitions)
    roster = inspection_cases.no_shared.context.roster[2]
    corpse_frame, corpse_scene = _oracle_corpse_frame_and_scene(
        inspection_cases.no_shared.frames[final_index],
        inspection_cases.oracle_scenes[final_index],
        public_agent_id=roster.public_agent_id,
        internal_slot=roster.global_slot,
    )
    assert (
        build_replay_oracle_inspection_v1(
            inspection_cases.no_shared.context,
            corpse_frame,
            corpse_scene,
            inspection_internal_slot=2,
            outgoing_transition=None,
            final_frame_index=final_index,
        )
        is None
    )


def test_no_shared_replay_is_fixed_recipient_local_and_final_safe(
    inspection_cases: _InspectionCases,
) -> None:
    source = _pov_index(inspection_cases.no_shared, actor_slot=2)
    current = _no_shared_current(
        inspection_cases.no_shared,
        source,
        frame_index=0,
    )
    result = build_replay_no_shared_obs_inspection_v1(source, current)
    assert result is not None
    assert result.actor_public_agent_id == source.content.public_agent_id
    assert result.decision_mask.owner_public_agent_id == source.content.public_agent_id
    assert result.combat_lane == "basic"
    assert isinstance(
        result.transition_reference,
        NoSharedObsReplayTransitionReferenceV1,
    )
    assert ":actor-pov:" in result.transition_reference.transition_id
    assert (
        f"{source.content.episode_id}:transition:"
        not in _canonical_bytes(result).decode()
    )

    final = _no_shared_current(
        inspection_cases.no_shared,
        source,
        frame_index=3,
    )
    assert build_replay_no_shared_obs_inspection_v1(source, final) is None


def test_no_shared_submitted_and_accepted_stay_distinct(
    inspection_cases: _InspectionCases,
) -> None:
    source = _pov_index(inspection_cases.no_shared, actor_slot=1)
    current = _no_shared_current(
        inspection_cases.no_shared,
        source,
        frame_index=2,
    )
    result = build_replay_no_shared_obs_inspection_v1(source, current)
    assert result is not None
    assert result.submitted_action.target_action == 99
    assert result.accepted_action.target_action == 0
    assert result.combat_lane == "none"
    assert result.accepted_target.target_kind == "no_target"


def test_no_shared_off_epoch_scene_anchor_rejects(
    inspection_cases: _InspectionCases,
) -> None:
    source = _pov_index(inspection_cases.no_shared, actor_slot=0)
    current = _no_shared_current(
        inspection_cases.no_shared,
        source,
        frame_index=1,
    )
    poisoned = _replace_scene_actor_position(
        current,
        public_agent_id=current.recipient_public_agent_id,
    )
    assert isinstance(poisoned, NoSharedObsAuthorizedScenePartsV1)
    with pytest.raises(ValueError, match="recorded s_n"):
        build_replay_no_shared_obs_inspection_v1(source, poisoned)


def test_no_shared_replay_can_accept_axis_only_target_without_key_anchor_or_path(
    inspection_cases: _InspectionCases,
) -> None:
    source = _pov_index(inspection_cases.no_shared, actor_slot=0)
    current = _no_shared_current(
        inspection_cases.no_shared,
        source,
        frame_index=0,
    )
    draft_axis = build_live_no_shared_obs_draft_inspection_v1(
        _pov_slice(inspection_cases.no_shared, actor_slot=0, frame_index=0),
        current,
        draft_move_action=0,
        draft_target_action=0,
        draft_armed_lane="none",
    ).decision_mask.target_actions
    hidden_action = next(
        row.target_action
        for row in draft_axis
        if isinstance(row, AuthorizedAxisOnlyTargetActionV1)
    )

    frames = list(source.content.frames)
    frame_payload = frames[0].model_dump(mode="python")
    frame_payload["action_mask"] = _pov_mask_with_legal_pair(
        frames[0].action_mask,
        target_action=hidden_action,
        ultimate_action=0,
    )
    frames[0] = ActorPovFrameV1.model_validate(frame_payload)
    transitions = list(source.content.transitions)
    transition_payload = transitions[0].model_dump(mode="python")
    submitted_payload = transitions[0].submitted_action.model_dump(mode="python")
    submitted_payload["select_target"] = hidden_action
    transition_payload["submitted_action"] = type(
        transitions[0].submitted_action
    ).model_validate(submitted_payload)
    accepted_payload = transitions[0].accepted_action.model_dump(mode="python")
    accepted_payload["select_target"] = hidden_action
    transition_payload["accepted_action"] = ActorPovAcceptedActionV1.model_validate(
        accepted_payload
    )
    transitions[0] = ActorPovTransitionV1.model_validate(transition_payload)
    changed_source = _rebuild_pov_index(
        source,
        frames=tuple(frames),
        transitions=tuple(transitions),
    )
    changed_current = _no_shared_current(
        inspection_cases.no_shared,
        changed_source,
        frame_index=0,
    )
    result = build_replay_no_shared_obs_inspection_v1(
        changed_source,
        changed_current,
    )
    assert result is not None
    assert result.combat_lane == "basic"
    assert result.accepted_action.target_action == hidden_action
    assert isinstance(result.accepted_target, AuthorizedAxisOnlyTargetActionV1)
    accepted_payload_json = _json_payload(result)["accepted_target"]
    assert isinstance(accepted_payload_json, dict)
    accepted_mapping = cast(dict[str, object], accepted_payload_json)
    assert set(accepted_mapping) == {
        "target_kind",
        "target_action",
        "display_name",
        "target_public_agent_id",
    }
    assert "path" not in json.dumps(_json_payload(result), sort_keys=True)


def test_shared_replay_uses_only_recipient_row_and_local_ids(
    inspection_cases: _InspectionCases,
) -> None:
    current, source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    result = build_replay_shared_obs_inspection_v1(
        current,
        source,
        authorized_recipient_global_slot=0,
        outgoing_transition=inspection_cases.shared.transitions[1],
        final_frame_index=3,
    )
    assert result is not None
    assert result.actor_public_agent_id == source.base_sensor_frame.public_agent_id
    assert result.combat_lane == "ultimate"
    reference = result.transition_reference
    assert reference.reference_kind == "shared_obs_visual_union_transition"
    assert ":shared-obs-visual-union:" in reference.transition_id
    payload = _canonical_bytes(result).decode()
    assert f"{source.base_sensor_frame.episode_id}:transition:1" not in payload
    assert source.incoming_transition_id is not None
    assert source.incoming_transition_id not in payload


def test_shared_team_b_recipient_uses_its_exact_team_block_owner_row(
    inspection_cases: _InspectionCases,
) -> None:
    current, source = _shared_current(
        inspection_cases.shared,
        recipient_slot=5,
        frame_index=1,
    )
    result = build_replay_shared_obs_inspection_v1(
        current,
        source,
        authorized_recipient_global_slot=5,
        outgoing_transition=inspection_cases.shared.transitions[1],
        final_frame_index=3,
    )
    assert result is not None
    assert source.base_sensor_scene.self_actor.team_id == 2
    assert source.base_sensor_scene.self_actor.team_local_slot == 0
    assert source.base_sensor_scene.self_actor.global_slot == 5
    assert result.actor_public_agent_id == (
        inspection_cases.shared.context.roster[5].public_agent_id
    )
    accepted = inspection_cases.shared.transitions[
        1
    ].facts.action_acceptance_facts.accepted_joint_action
    assert result.accepted_action.move_action == accepted.move[5]
    assert result.accepted_action.target_action == accepted.select_target[5]
    assert result.accepted_action.use_ultimate_action == accepted.use_ultimate[5]
    _assert_exact_decision_mask(
        result.decision_mask,
        source.base_sensor_frame.action_mask,
    )
    assert "authorized_recipient_global_slot" not in _canonical_bytes(result).decode()
    assert "global_slot" not in _canonical_bytes(result).decode()


def test_shared_final_and_off_epoch_anchor_reject(
    inspection_cases: _InspectionCases,
) -> None:
    final, final_source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=3,
    )
    assert (
        build_replay_shared_obs_inspection_v1(
            final,
            final_source,
            authorized_recipient_global_slot=0,
            outgoing_transition=None,
            final_frame_index=3,
        )
        is None
    )

    current, source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    poisoned = _replace_scene_actor_position(
        current,
        public_agent_id=current.recipient_public_agent_id,
    )
    assert isinstance(poisoned, SharedObsAuthorizedScenePartsV1)
    with pytest.raises(ValueError, match="recorded s_n"):
        build_replay_shared_obs_inspection_v1(
            poisoned,
            source,
            authorized_recipient_global_slot=0,
            outgoing_transition=inspection_cases.shared.transitions[1],
            final_frame_index=3,
        )


def test_shared_self_slot_and_relation_axis_forgeries_reject(
    inspection_cases: _InspectionCases,
) -> None:
    current, source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    transition = inspection_cases.shared.transitions[1]

    wrong_slot = copy.copy(source)
    wrong_scene = copy.copy(source.base_sensor_scene)
    wrong_self = copy.copy(source.base_sensor_scene.self_actor)
    object.__setattr__(wrong_self, "global_slot", 1)
    object.__setattr__(wrong_scene, "self_actor", wrong_self)
    object.__setattr__(wrong_slot, "base_sensor_scene", wrong_scene)
    with pytest.raises(ValueError, match="authorized recipient row"):
        build_replay_shared_obs_inspection_v1(
            current,
            wrong_slot,
            authorized_recipient_global_slot=0,
            outgoing_transition=transition,
            final_frame_index=3,
        )

    swapped_axis = copy.copy(source)
    ally_axis = list(source.ally_observation_row_global_slot_by_id)
    ally_axis[0], ally_axis[1] = ally_axis[1], ally_axis[0]
    object.__setattr__(
        swapped_axis,
        "ally_observation_row_global_slot_by_id",
        tuple(ally_axis),
    )
    with pytest.raises(ValueError, match="team-local ordering"):
        build_replay_shared_obs_inspection_v1(
            current,
            swapped_axis,
            authorized_recipient_global_slot=0,
            outgoing_transition=transition,
            final_frame_index=3,
        )

    coordinated_swap = copy.copy(source)
    coordinated_scene = copy.copy(source.base_sensor_scene)
    coordinated_self = copy.copy(source.base_sensor_scene.self_actor)
    object.__setattr__(coordinated_self, "global_slot", 1)
    object.__setattr__(coordinated_self, "team_local_slot", 1)
    object.__setattr__(coordinated_scene, "self_actor", coordinated_self)
    object.__setattr__(coordinated_swap, "base_sensor_scene", coordinated_scene)
    axis_payload = source.axis_mapping.model_dump(mode="python")
    ally_public_ids = list(
        source.axis_mapping.ally_observation_row_public_agent_id_by_id
    )
    ally_public_ids[0], ally_public_ids[1] = ally_public_ids[1], ally_public_ids[0]
    axis_payload["ally_observation_row_public_agent_id_by_id"] = tuple(ally_public_ids)
    axis_payload["target_action_recipient_public_agent_id_by_id"] = (
        None,
        *ally_public_ids,
        *source.axis_mapping.enemy_observation_row_public_agent_id_by_id,
    )
    object.__setattr__(
        coordinated_swap,
        "axis_mapping",
        type(source.axis_mapping).model_validate(axis_payload),
    )
    with pytest.raises(ValueError, match="authorized recipient row"):
        build_replay_shared_obs_inspection_v1(
            current,
            coordinated_swap,
            authorized_recipient_global_slot=0,
            outgoing_transition=transition,
            final_frame_index=3,
        )


def test_shared_authority_credential_rejects_wrong_local_cross_block_and_invalid_slots(
    inspection_cases: _InspectionCases,
) -> None:
    current, source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    transition = inspection_cases.shared.transitions[1]
    for wrong_authorized_slot in (1, 5):
        with pytest.raises(ValueError, match="authorized recipient row"):
            build_replay_shared_obs_inspection_v1(
                current,
                source,
                authorized_recipient_global_slot=wrong_authorized_slot,
                outgoing_transition=transition,
                final_frame_index=3,
            )
    for invalid_authorized_slot in (-1, 10, False):
        with pytest.raises(ValueError, match="authorized_recipient_global_slot"):
            build_live_shared_obs_draft_inspection_v1(
                current,
                source,
                authorized_recipient_global_slot=cast(int, invalid_authorized_slot),
                draft_move_action=0,
                draft_target_action=0,
                draft_armed_lane="none",
            )


def test_shared_fully_valid_joint_forgery_cannot_redirect_service_authorized_row(
    inspection_cases: _InspectionCases,
) -> None:
    current, source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    axis_payload = source.axis_mapping.model_dump(mode="python")
    ally_public_ids = list(
        source.axis_mapping.ally_observation_row_public_agent_id_by_id
    )
    ally_public_ids[0], ally_public_ids[1] = ally_public_ids[1], ally_public_ids[0]
    axis_payload["ally_observation_row_public_agent_id_by_id"] = tuple(ally_public_ids)
    axis_payload["target_action_recipient_public_agent_id_by_id"] = (
        None,
        *ally_public_ids,
        *source.axis_mapping.enemy_observation_row_public_agent_id_by_id,
    )
    forged_axis = type(source.axis_mapping).model_validate(axis_payload)
    build_base_scene = vars(evaluation_adapter_module)["_shared_obs_base_sensor_scene"]
    forged_base_scene = build_base_scene(
        source.base_sensor_frame,
        selected_global_slot=1,
        selected_team_local_slot=1,
        configured_team_id=1,
        class_id=source.base_sensor_scene.self_actor.class_id,
        axis_mapping=forged_axis,
    )
    availability = list(source.sensor_source_availability)
    row_zero = availability[0]
    row_one = availability[1]
    availability[0] = replace(
        row_zero,
        sensor_source_public_agent_id=row_one.sensor_source_public_agent_id,
        relation_to_recipient="ally",
        recorded_available=row_one.recorded_available,
    )
    availability[1] = replace(
        row_one,
        sensor_source_public_agent_id=row_zero.sensor_source_public_agent_id,
        relation_to_recipient="self",
        recorded_available=False,
    )
    forged_source = replace(
        source,
        axis_mapping=forged_axis,
        base_sensor_scene=forged_base_scene,
        sensor_source_availability=tuple(availability),
    )
    forged_source.__post_init__()

    agent_by_public_id = {row.public_agent_id: row for row in current.scene.agents}
    public_zero = source.axis_mapping.ally_observation_row_public_agent_id_by_id[0]
    public_one = source.axis_mapping.ally_observation_row_public_agent_id_by_id[1]
    agent_zero = agent_by_public_id[public_zero]
    agent_one = agent_by_public_id[public_one]
    forged_pads = tuple(
        replace(
            pad,
            assigned_presentation_key=(
                agent_one.presentation_key
                if pad.team_id == 1 and pad.team_local_slot == 0
                else agent_zero.presentation_key
            ),
            assigned_public_agent_id=(
                agent_one.public_agent_id
                if pad.team_id == 1 and pad.team_local_slot == 0
                else agent_zero.public_agent_id
            ),
            currently_alive=(
                agent_one.life_state == "alive"
                if pad.team_id == 1 and pad.team_local_slot == 0
                else agent_zero.life_state == "alive"
            ),
            spawn_shield_remaining=(
                agent_one.spawn_shield_remaining
                if pad.team_id == 1 and pad.team_local_slot == 0
                else agent_zero.spawn_shield_remaining
            ),
        )
        if pad.team_id == 1 and pad.team_local_slot in (0, 1)
        else pad
        for pad in current.scene.spawn_pads
    )
    forged_scene = replace(current.scene, spawn_pads=forged_pads)
    forged_current = replace(current, scene=forged_scene)
    forged_current.__post_init__()

    with pytest.raises(ValueError, match="authorized recipient row"):
        build_replay_shared_obs_inspection_v1(
            forged_current,
            forged_source,
            authorized_recipient_global_slot=0,
            outgoing_transition=inspection_cases.shared.transitions[1],
            final_frame_index=3,
        )


def _mutate_other_actor_transition(
    transition: EvaluationTransitionV1,
    *,
    recipient_slot: int,
) -> EvaluationTransitionV1:
    other_slot = 1 if recipient_slot != 1 else 0
    acceptance = transition.facts.action_acceptance_facts
    submitted_payload = acceptance.submitted_joint_action.model_dump(mode="python")
    submitted_moves = list(cast(tuple[int, ...], submitted_payload["move"]))
    submitted_moves[other_slot] = (submitted_moves[other_slot] + 1) % 9
    submitted_payload["move"] = tuple(submitted_moves)
    submitted = JointActionV1.model_validate(submitted_payload)
    acceptance_payload = acceptance.model_dump(mode="python")
    acceptance_payload["submitted_joint_action"] = submitted
    changed_acceptance = ActionAcceptanceFactsV1.model_validate(acceptance_payload)
    facts_payload = transition.facts.model_dump(mode="python")
    facts_payload["action_acceptance_facts"] = changed_acceptance
    facts = TransitionFactsV1.model_validate(facts_payload)
    payload = transition.model_dump(mode="python")
    payload["facts"] = facts
    rewards = list(cast(tuple[float, ...], payload["canonical_reward_by_agent"]))
    rewards[other_slot] += 17.0
    payload["canonical_reward_by_agent"] = tuple(rewards)
    payload["events"] = ()
    return EvaluationTransitionV1.model_validate(payload)


def _mutate_physical_successor_facts(
    transition: EvaluationTransitionV1,
    *,
    actor_slot: int,
) -> EvaluationTransitionV1:
    physical = transition.facts.physical_facts
    physical_payload = physical.model_dump(mode="python")
    displacement = [
        list(row) for row in physical.ordinary_movement_phase_displacement_by_agent
    ]
    displacement[actor_slot][0] += 0.5
    physical_payload["ordinary_movement_phase_displacement_by_agent"] = tuple(
        tuple(row) for row in displacement
    )
    facts_payload = transition.facts.model_dump(mode="python")
    facts_payload["physical_facts"] = type(physical).model_validate(physical_payload)
    payload = transition.model_dump(mode="python")
    payload["facts"] = TransitionFactsV1.model_validate(facts_payload)
    return EvaluationTransitionV1.model_validate(payload)


def test_oracle_outgoing_bytes_ignore_successor_displacement_facts(
    inspection_cases: _InspectionCases,
) -> None:
    context = inspection_cases.no_shared.context
    frame = inspection_cases.no_shared.frames[0]
    scene = inspection_cases.oracle_scenes[0]
    baseline = build_replay_oracle_inspection_v1(
        context,
        frame,
        scene,
        inspection_internal_slot=2,
        outgoing_transition=inspection_cases.no_shared.transitions[0],
        final_frame_index=3,
    )
    changed = build_replay_oracle_inspection_v1(
        context,
        frame,
        scene,
        inspection_internal_slot=2,
        outgoing_transition=_mutate_physical_successor_facts(
            inspection_cases.no_shared.transitions[0],
            actor_slot=2,
        ),
        final_frame_index=3,
    )
    assert baseline is not None and changed is not None
    assert _canonical_bytes(changed) == _canonical_bytes(baseline)
    assert (
        "successor_frame"
        not in inspect.signature(build_replay_oracle_inspection_v1).parameters
    )


def test_no_shared_successor_incoming_reward_cues_and_other_frames_are_inert(
    inspection_cases: _InspectionCases,
) -> None:
    source = _pov_index(inspection_cases.no_shared, actor_slot=2)
    current = _no_shared_current(
        inspection_cases.no_shared,
        source,
        frame_index=0,
    )
    baseline = build_replay_no_shared_obs_inspection_v1(source, current)
    assert baseline is not None

    frames = list(source.content.frames)
    frames[1] = _pov_frame_with_shifted_self_position(
        frames[1],
        delta_x=0.375,
    )
    last_payload = frames[-1].model_dump(mode="python")
    objectives = [list(row) for row in frames[-1].objective_features]
    objectives[0][0] += 0.125
    last_payload["objective_features"] = tuple(tuple(row) for row in objectives)
    frames[-1] = ActorPovFrameV1.model_validate(last_payload)
    transitions = list(source.content.transitions)
    outgoing_payload = transitions[0].model_dump(mode="python")
    outgoing_payload["canonical_reward"] = transitions[0].canonical_reward + 7.0
    transitions[0] = ActorPovTransitionV1.model_validate(outgoing_payload)
    coherent_transitions = _rederive_pov_transitions(
        source,
        tuple(frames),
        tuple(transitions),
    )
    assert coherent_transitions[0].cues != source.content.transitions[0].cues
    changed_source = _rebuild_pov_index(
        source,
        frames=tuple(frames),
        transitions=coherent_transitions,
    )
    changed_current = _no_shared_current(
        inspection_cases.no_shared,
        changed_source,
        frame_index=0,
    )
    changed = build_replay_no_shared_obs_inspection_v1(
        changed_source,
        changed_current,
    )
    assert changed is not None
    assert _canonical_bytes(changed) == _canonical_bytes(baseline)

    frame_one_current = _no_shared_current(
        inspection_cases.no_shared,
        source,
        frame_index=1,
    )
    frame_one_baseline = build_replay_no_shared_obs_inspection_v1(
        source,
        frame_one_current,
    )
    incoming = list(source.content.transitions)
    incoming_payload = incoming[0].model_dump(mode="python")
    incoming_payload["canonical_reward"] = incoming[0].canonical_reward - 3.0
    incoming[0] = ActorPovTransitionV1.model_validate(incoming_payload)
    incoming_source = _rebuild_pov_index(source, transitions=tuple(incoming))
    incoming_current = _no_shared_current(
        inspection_cases.no_shared,
        incoming_source,
        frame_index=1,
    )
    frame_one_changed = build_replay_no_shared_obs_inspection_v1(
        incoming_source,
        incoming_current,
    )
    assert frame_one_baseline is not None and frame_one_changed is not None
    assert _canonical_bytes(frame_one_changed) == _canonical_bytes(frame_one_baseline)


def test_shared_other_actor_history_reward_event_and_diagnostic_branches_are_inert(
    inspection_cases: _InspectionCases,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )

    def build(
        current_value: SharedObsAuthorizedScenePartsV1,
        source_value: SharedObsSourceMaterialProjectionV1,
        transition: EvaluationTransitionV1,
    ) -> bytes:
        result = build_replay_shared_obs_inspection_v1(
            current_value,
            source_value,
            authorized_recipient_global_slot=0,
            outgoing_transition=transition,
            final_frame_index=3,
        )
        assert result is not None
        return _canonical_bytes(result)

    transition = inspection_cases.shared.transitions[1]
    baseline = build(current, source, transition)
    assert (
        build(
            current,
            source,
            _mutate_other_actor_transition(transition, recipient_slot=0),
        )
        == baseline
    )

    hidden_source = copy.copy(source)
    object.__setattr__(hidden_source, "incoming_transition_id", "oracle-hidden-id")
    object.__setattr__(hidden_source, "sensor_source_availability", ())
    hidden_frame = copy.copy(hidden_source.base_sensor_frame)
    object.__setattr__(hidden_frame, "objective_features", ())
    object.__setattr__(hidden_frame, "previous_timestep_actions", object())
    object.__setattr__(hidden_source, "base_sensor_frame", hidden_frame)

    hidden_current = copy.copy(current)
    object.__setattr__(hidden_current, "authorized_sensor_sources", ())
    object.__setattr__(hidden_current, "agent_observation_provenance", ())
    assert build(hidden_current, hidden_source, transition) == baseline

    available_index = next(
        index
        for index, row in enumerate(source.sensor_source_availability)
        if row.recorded_available
    )
    availability = list(source.sensor_source_availability)
    availability[available_index] = replace(
        availability[available_index],
        recorded_available=False,
    )
    objective = [list(row) for row in source.base_sensor_frame.objective_features]
    objective[0][0] += 0.25
    changed_frame = replace(
        source.base_sensor_frame,
        objective_features=tuple(tuple(row) for row in objective),
    )
    valid_hidden_source = replace(
        source,
        base_sensor_frame=changed_frame,
        sensor_source_availability=tuple(availability),
    )
    valid_hidden_source.__post_init__()
    assert build(current, valid_hidden_source, transition) == baseline

    provenance_index, removable_index = next(
        (row_index, source_index)
        for row_index, row in enumerate(current.agent_observation_provenance)
        for source_index, provenance_source in enumerate(row.observation_sources)
        if len(row.observation_sources) > 1
        and provenance_source.source_public_agent_id != row.agent_public_agent_id
    )
    provenance = list(current.agent_observation_provenance)
    provenance_row = provenance[provenance_index]
    provenance[provenance_index] = replace(
        provenance_row,
        observation_sources=tuple(
            source_row
            for index, source_row in enumerate(provenance_row.observation_sources)
            if index != removable_index
        ),
    )
    valid_hidden_current = replace(
        current,
        agent_observation_provenance=tuple(provenance),
    )
    valid_hidden_current.__post_init__()
    assert build(valid_hidden_current, source, transition) == baseline

    def forbidden(_value: object) -> None:
        raise AssertionError("wide Shared diagnostic validator was called")

    monkeypatch.setattr(SharedObsAuthorizedScenePartsV1, "__post_init__", forbidden)
    monkeypatch.setattr(
        SharedObsSourceMaterialProjectionV1,
        "__post_init__",
        forbidden,
    )
    assert build(current, source, transition) == baseline


def test_shared_outgoing_structurally_poisoned_forbidden_branches_are_inert(
    inspection_cases: _InspectionCases,
) -> None:
    current, source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    transition = inspection_cases.shared.transitions[1]
    baseline = build_replay_shared_obs_inspection_v1(
        current,
        source,
        authorized_recipient_global_slot=0,
        outgoing_transition=transition,
        final_frame_index=3,
    )
    assert baseline is not None
    acceptance = transition.facts.action_acceptance_facts
    submitted = acceptance.submitted_joint_action
    accepted = acceptance.accepted_joint_action
    hidden_submitted_move = tuple(
        value if index == 0 else object() for index, value in enumerate(submitted.move)
    )
    hidden_accepted_target = tuple(
        value if index == 0 else object()
        for index, value in enumerate(accepted.select_target)
    )
    poisoned_submitted = submitted.model_copy(update={"move": hidden_submitted_move})
    poisoned_accepted = accepted.model_copy(
        update={"select_target": hidden_accepted_target}
    )
    poisoned_acceptance = acceptance.model_copy(
        update={
            "submitted_joint_action": poisoned_submitted,
            "accepted_joint_action": poisoned_accepted,
            "submitted_action_tuple_is_out_of_domain_by_actor": object(),
            "in_domain_move_action_is_rejected_by_actor": object(),
            "in_domain_combat_action_pair_is_rejected_by_actor": object(),
        }
    )
    poisoned_facts = transition.facts.model_copy(
        update={
            "action_acceptance_facts": poisoned_acceptance,
            "combat_transition_facts": object(),
            "death_facts": object(),
            "spawn_shield_facts": object(),
            "respawn_facts": object(),
            "regeneration_facts": object(),
            "physical_facts": object(),
            "aura_facts": object(),
            "status_lifecycle_facts": object(),
        }
    )
    poisoned = transition.model_copy(
        update={
            "facts": poisoned_facts,
            "events": object(),
            "canonical_reward_by_agent": object(),
            "canonical_reward_by_team": object(),
            "terminated": object(),
            "truncated": object(),
            "owning_task_end_reason": object(),
        }
    )

    result = build_replay_shared_obs_inspection_v1(
        current,
        source,
        authorized_recipient_global_slot=0,
        outgoing_transition=poisoned,
        final_frame_index=3,
    )

    assert result is not None
    assert _canonical_bytes(result) == _canonical_bytes(baseline)


def test_shared_outgoing_selective_used_epoch_and_action_poisons_reject(
    inspection_cases: _InspectionCases,
) -> None:
    current, source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    transition = inspection_cases.shared.transitions[1]
    acceptance = transition.facts.action_acceptance_facts
    submitted = acceptance.submitted_joint_action
    accepted = acceptance.accepted_joint_action

    def with_acceptance(value: object) -> EvaluationTransitionV1:
        facts = transition.facts.model_copy(update={"action_acceptance_facts": value})
        return transition.model_copy(update={"facts": facts})

    submitted_list = submitted.model_copy(update={"move": list(submitted.move)})
    submitted_bool_values = list(submitted.move)
    submitted_bool_values[0] = True
    submitted_bool = submitted.model_copy(update={"move": tuple(submitted_bool_values)})
    accepted_values = list(accepted.select_target)
    accepted_values[0] = 99
    accepted_out_of_domain = accepted.model_copy(
        update={"select_target": tuple(accepted_values)}
    )
    poisoned_roots = (
        transition.model_copy(update={"schema_id": "wrong.transition"}),
        transition.model_copy(update={"schema_version": True}),
        transition.model_copy(
            update={
                "facts": transition.facts.model_copy(
                    update={"schema_id": "wrong.transition_facts"}
                )
            }
        ),
        transition.model_copy(
            update={
                "facts": transition.facts.model_copy(update={"schema_version": True})
            }
        ),
        transition.model_copy(update={"transition_index": True}),
        transition.model_copy(
            update={
                "facts": transition.facts.model_copy(
                    update={"transition_start_step_count": True}
                )
            }
        ),
        with_acceptance(object()),
        with_acceptance(
            acceptance.model_copy(update={"submitted_joint_action": object()})
        ),
        with_acceptance(
            acceptance.model_copy(update={"submitted_joint_action": submitted_list})
        ),
        with_acceptance(
            acceptance.model_copy(update={"submitted_joint_action": submitted_bool})
        ),
        with_acceptance(
            acceptance.model_copy(
                update={"accepted_joint_action": accepted_out_of_domain}
            )
        ),
    )

    for poison_index, poisoned in enumerate(poisoned_roots):
        try:
            build_replay_shared_obs_inspection_v1(
                current,
                source,
                authorized_recipient_global_slot=0,
                outgoing_transition=poisoned,
                final_frame_index=3,
            )
        except TypeError, ValueError:
            continue
        pytest.fail(f"Shared used-path poison {poison_index} was accepted")


def test_shared_recipient_action_mutation_changes_bytes(
    inspection_cases: _InspectionCases,
) -> None:
    current, source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    transition = inspection_cases.shared.transitions[1]
    baseline = build_replay_shared_obs_inspection_v1(
        current,
        source,
        authorized_recipient_global_slot=0,
        outgoing_transition=transition,
        final_frame_index=3,
    )
    assert baseline is not None
    acceptance = transition.facts.action_acceptance_facts
    submitted_payload = acceptance.submitted_joint_action.model_dump(mode="python")
    moves = list(cast(tuple[int, ...], submitted_payload["move"]))
    moves[0] = 1
    submitted_payload["move"] = tuple(moves)
    changed_submitted = JointActionV1.model_validate(submitted_payload)
    acceptance_payload = acceptance.model_dump(mode="python")
    acceptance_payload["submitted_joint_action"] = changed_submitted
    changed_acceptance = ActionAcceptanceFactsV1.model_validate(acceptance_payload)
    facts_payload = transition.facts.model_dump(mode="python")
    facts_payload["action_acceptance_facts"] = changed_acceptance
    payload = transition.model_dump(mode="python")
    payload["facts"] = TransitionFactsV1.model_validate(facts_payload)
    changed_transition = EvaluationTransitionV1.model_validate(payload)
    changed = build_replay_shared_obs_inspection_v1(
        current,
        source,
        authorized_recipient_global_slot=0,
        outgoing_transition=changed_transition,
        final_frame_index=3,
    )
    assert changed is not None
    assert _canonical_bytes(changed) != _canonical_bytes(baseline)


def test_decision_mask_strict_shape_marginal_order_and_identity_poison(
    inspection_cases: _InspectionCases,
) -> None:
    result = _oracle(inspection_cases, frame_index=0, actor_slot=2)
    assert result is not None
    adapter = TypeAdapter(AuthorizedDecisionMaskV1)
    baseline = _json_payload(result.decision_mask)
    assert adapter.validate_json(json.dumps(baseline)) == result.decision_mask

    poisons: list[dict[str, object]] = []
    extra = copy.deepcopy(baseline)
    extra["extra"] = True
    poisons.append(extra)
    coerced = copy.deepcopy(baseline)
    cast(list[object], coerced["movement_action_mask"])[0] = 1
    poisons.append(coerced)
    short = copy.deepcopy(baseline)
    short["movement_action_display_names"] = cast(
        list[object], short["movement_action_display_names"]
    )[:-1]
    poisons.append(short)
    short_move_mask = copy.deepcopy(baseline)
    short_move_mask["movement_action_mask"] = cast(
        list[object], short_move_mask["movement_action_mask"]
    )[:-1]
    poisons.append(short_move_mask)
    short_targets = copy.deepcopy(baseline)
    short_targets["target_actions"] = cast(
        list[object], short_targets["target_actions"]
    )[:-1]
    poisons.append(short_targets)
    short_target_mask = copy.deepcopy(baseline)
    short_target_mask["target_action_mask"] = cast(
        list[object], short_target_mask["target_action_mask"]
    )[:-1]
    poisons.append(short_target_mask)
    short_ultimate_names = copy.deepcopy(baseline)
    short_ultimate_names["use_ultimate_action_display_names"] = cast(
        list[object], short_ultimate_names["use_ultimate_action_display_names"]
    )[:-1]
    poisons.append(short_ultimate_names)
    short_ultimate_mask = copy.deepcopy(baseline)
    short_ultimate_mask["use_ultimate_action_mask"] = cast(
        list[object], short_ultimate_mask["use_ultimate_action_mask"]
    )[:-1]
    poisons.append(short_ultimate_mask)
    short_joint_outer = copy.deepcopy(baseline)
    short_joint_outer["target_use_ultimate_joint_mask"] = cast(
        list[object], short_joint_outer["target_use_ultimate_joint_mask"]
    )[:-1]
    poisons.append(short_joint_outer)
    short_joint_inner = copy.deepcopy(baseline)
    joint_rows = cast(
        list[list[object]],
        short_joint_inner["target_use_ultimate_joint_mask"],
    )
    joint_rows[0] = joint_rows[0][:-1]
    poisons.append(short_joint_inner)
    marginal = copy.deepcopy(baseline)
    cast(list[object], marginal["target_action_mask"])[0] = False
    poisons.append(marginal)
    reordered = copy.deepcopy(baseline)
    targets = cast(list[object], reordered["target_actions"])
    targets[1], targets[2] = targets[2], targets[1]
    poisons.append(reordered)
    duplicate_key = copy.deepcopy(baseline)
    duplicate_targets = cast(list[dict[str, object]], duplicate_key["target_actions"])
    visible = [
        row
        for row in duplicate_targets
        if row["target_kind"] == "visible_authorized_agent"
    ]
    assert len(visible) >= 2
    visible[1]["target_presentation_key"] = visible[0]["target_presentation_key"]
    poisons.append(duplicate_key)
    duplicate_public = copy.deepcopy(baseline)
    duplicate_public_targets = cast(
        list[dict[str, object]], duplicate_public["target_actions"]
    )
    duplicate_public_targets[2]["target_public_agent_id"] = duplicate_public_targets[1][
        "target_public_agent_id"
    ]
    poisons.append(duplicate_public)
    zero_wrong_variant = copy.deepcopy(baseline)
    zero_targets = cast(list[dict[str, object]], zero_wrong_variant["target_actions"])
    replacement_zero = copy.deepcopy(zero_targets[1])
    replacement_zero["target_action"] = 0
    zero_targets[0] = replacement_zero
    poisons.append(zero_wrong_variant)
    wrong_target_discriminator = copy.deepcopy(baseline)
    wrong_discriminator_targets = cast(
        list[dict[str, object]], wrong_target_discriminator["target_actions"]
    )
    wrong_discriminator_targets[1]["target_kind"] = "unknown"
    poisons.append(wrong_target_discriminator)
    missing_nested = copy.deepcopy(baseline)
    missing_targets = cast(list[dict[str, object]], missing_nested["target_actions"])
    del missing_targets[1]["target_public_agent_id"]
    poisons.append(missing_nested)
    extra_nested = copy.deepcopy(baseline)
    extra_targets = cast(list[dict[str, object]], extra_nested["target_actions"])
    extra_targets[1]["unexpected"] = True
    poisons.append(extra_nested)
    coerced_nested = copy.deepcopy(baseline)
    coerced_targets = cast(list[dict[str, object]], coerced_nested["target_actions"])
    coerced_targets[1]["target_action"] = "1"
    poisons.append(coerced_nested)
    for poison in poisons:
        with pytest.raises(ValidationError):
            adapter.validate_json(json.dumps(poison))


def test_target_union_visible_axis_only_surfaces_are_exact(
    inspection_cases: _InspectionCases,
) -> None:
    result = _oracle(inspection_cases, frame_index=0, actor_slot=2)
    assert result is not None
    adapter: TypeAdapter[AuthorizedTargetActionV1] = TypeAdapter(
        AuthorizedTargetActionV1
    )
    visible = next(
        row
        for row in result.decision_mask.target_actions
        if isinstance(row, AuthorizedVisibleTargetActionV1)
    )
    axis_only = next(
        row
        for row in result.decision_mask.target_actions
        if isinstance(row, AuthorizedAxisOnlyTargetActionV1)
    )
    visible_payload = json.loads(adapter.dump_json(visible))
    axis_payload = json.loads(adapter.dump_json(axis_only))
    assert set(visible_payload) == {
        "target_kind",
        "target_action",
        "display_name",
        "target_presentation_key",
        "target_public_agent_id",
        "target_anchor",
    }
    assert set(axis_payload) == {
        "target_kind",
        "target_action",
        "display_name",
        "target_public_agent_id",
    }
    axis_payload["target_anchor"] = [0.0, 0.0]
    with pytest.raises(ValidationError):
        adapter.validate_json(json.dumps(axis_payload))
    del visible_payload["target_anchor"]
    with pytest.raises(ValidationError):
        adapter.validate_json(json.dumps(visible_payload))


def test_reference_namespaces_index_and_adjacency_are_strict(
    inspection_cases: _InspectionCases,
) -> None:
    source = _pov_index(inspection_cases.no_shared, actor_slot=2)
    current = _no_shared_current(
        inspection_cases.no_shared,
        source,
        frame_index=0,
    )
    result = build_replay_no_shared_obs_inspection_v1(source, current)
    assert result is not None
    adapter = TypeAdapter(ReplayInspectionPresentationV1)
    baseline = _json_payload(result)

    canonical_oracle = copy.deepcopy(baseline)
    reference = cast(dict[str, object], canonical_oracle["transition_reference"])
    reference["transition_id"] = f"{source.content.episode_id}:transition:0"
    reference["start_frame_id"] = f"{source.content.episode_id}:frame:0"
    reference["successor_frame_id"] = f"{source.content.episode_id}:frame:1"
    with pytest.raises(ValidationError):
        adapter.validate_json(json.dumps(canonical_oracle))

    wrong_index = copy.deepcopy(baseline)
    wrong_index["outgoing_transition_index"] = 1
    with pytest.raises(ValidationError):
        adapter.validate_json(json.dumps(wrong_index))

    wrong_successor = copy.deepcopy(baseline)
    successor_ref = cast(dict[str, object], wrong_successor["transition_reference"])
    successor_ref["successor_frame_id"] = str(successor_ref["start_frame_id"])
    with pytest.raises(ValidationError):
        adapter.validate_json(json.dumps(wrong_successor))


def test_oracle_no_shared_and_shared_reference_namespaces_are_independently_strict(
    inspection_cases: _InspectionCases,
) -> None:
    oracle = _oracle(inspection_cases, frame_index=0, actor_slot=2)
    no_shared_source = _pov_index(inspection_cases.no_shared, actor_slot=2)
    no_shared = build_replay_no_shared_obs_inspection_v1(
        no_shared_source,
        _no_shared_current(
            inspection_cases.no_shared,
            no_shared_source,
            frame_index=0,
        ),
    )
    shared_current, shared_source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    shared = build_replay_shared_obs_inspection_v1(
        shared_current,
        shared_source,
        authorized_recipient_global_slot=0,
        outgoing_transition=inspection_cases.shared.transitions[1],
        final_frame_index=3,
    )
    assert oracle is not None and no_shared is not None and shared is not None
    assert isinstance(oracle.transition_reference, OracleReplayTransitionReferenceV1)
    assert isinstance(
        no_shared.transition_reference,
        NoSharedObsReplayTransitionReferenceV1,
    )
    assert isinstance(
        shared.transition_reference,
        SharedObsReplayTransitionReferenceV1,
    )

    adapter = TypeAdapter(ReplayInspectionPresentationV1)
    for result in (oracle, no_shared, shared):
        baseline = _json_payload(result)
        index = result.outgoing_transition_index
        episode_id = inspection_cases.no_shared.context.identity.episode_id

        wrong_index = copy.deepcopy(baseline)
        wrong_index["outgoing_transition_index"] = index + 1

        wrong_start = copy.deepcopy(baseline)
        wrong_start_ref = cast(dict[str, object], wrong_start["transition_reference"])
        wrong_start_ref["start_frame_id"] = str(wrong_start_ref["successor_frame_id"])

        wrong_successor = copy.deepcopy(baseline)
        wrong_successor_ref = cast(
            dict[str, object], wrong_successor["transition_reference"]
        )
        wrong_successor_ref["successor_frame_id"] = str(
            wrong_successor_ref["start_frame_id"]
        )

        wrong_namespace = copy.deepcopy(baseline)
        wrong_namespace_ref = cast(
            dict[str, object], wrong_namespace["transition_reference"]
        )
        if isinstance(result.transition_reference, OracleReplayTransitionReferenceV1):
            wrong_prefix = f"{episode_id}:actor-pov:{result.actor_public_agent_id}"
        else:
            wrong_prefix = episode_id
        wrong_namespace_ref["transition_id"] = f"{wrong_prefix}:transition:{index}"
        wrong_namespace_ref["start_frame_id"] = f"{wrong_prefix}:frame:{index}"
        wrong_namespace_ref["successor_frame_id"] = f"{wrong_prefix}:frame:{index + 1}"

        wrong_discriminator = copy.deepcopy(baseline)
        wrong_discriminator_ref = cast(
            dict[str, object], wrong_discriminator["transition_reference"]
        )
        wrong_discriminator_ref["reference_kind"] = "unknown"

        poisons = [
            wrong_index,
            wrong_start,
            wrong_successor,
            wrong_namespace,
            wrong_discriminator,
        ]
        if not isinstance(
            result.transition_reference,
            OracleReplayTransitionReferenceV1,
        ):
            wrong_recipient = copy.deepcopy(baseline)
            wrong_recipient_ref = cast(
                dict[str, object], wrong_recipient["transition_reference"]
            )
            wrong_recipient_ref["recipient_public_agent_id"] = "agent-slot-9"
            poisons.append(wrong_recipient)
        for poison in poisons:
            with pytest.raises(ValidationError):
                adapter.validate_json(json.dumps(poison))


def test_replay_root_and_nested_action_contracts_reject_all_strict_poison(
    inspection_cases: _InspectionCases,
) -> None:
    result = _oracle(inspection_cases, frame_index=0, actor_slot=2)
    assert result is not None
    adapter = TypeAdapter(ReplayInspectionPresentationV1)
    baseline = _json_payload(result)
    assert adapter.validate_json(json.dumps(baseline)) == result

    allowed_episode = "A.b_c:/+d-e"
    allowed = copy.deepcopy(baseline)
    allowed["episode_id"] = allowed_episode
    allowed_reference = cast(dict[str, object], allowed["transition_reference"])
    allowed_reference["transition_id"] = f"{allowed_episode}:transition:0"
    allowed_reference["start_frame_id"] = f"{allowed_episode}:frame:0"
    allowed_reference["successor_frame_id"] = f"{allowed_episode}:frame:1"
    assert adapter.validate_json(json.dumps(allowed)).episode_id == allowed_episode

    poisons: list[dict[str, object]] = []
    missing_episode = copy.deepcopy(baseline)
    del missing_episode["episode_id"]
    poisons.append(missing_episode)
    coerced_episode = copy.deepcopy(baseline)
    coerced_episode["episode_id"] = 1
    poisons.append(coerced_episode)
    for invalid_episode in (
        "",
        " episode-001",
        "episode-001 ",
        "episode 001",
        "épisode-001",
    ):
        invalid_episode_payload = copy.deepcopy(baseline)
        invalid_episode_payload["episode_id"] = invalid_episode
        poisons.append(invalid_episode_payload)
    mismatched_episode = copy.deepcopy(baseline)
    mismatched_episode["episode_id"] = "episode-999"
    poisons.append(mismatched_episode)
    extra_episode = copy.deepcopy(baseline)
    cast(dict[str, object], extra_episode["transition_reference"])["episode_id"] = (
        baseline["episode_id"]
    )
    poisons.append(extra_episode)
    missing = copy.deepcopy(baseline)
    del missing["accepted_action"]
    poisons.append(missing)
    extra = copy.deepcopy(baseline)
    extra["unexpected"] = True
    poisons.append(extra)
    coerced = copy.deepcopy(baseline)
    coerced["outgoing_transition_index"] = "0"
    poisons.append(coerced)
    wrong_discriminator = copy.deepcopy(baseline)
    wrong_discriminator["inspection_kind"] = "live_draft_action"
    poisons.append(wrong_discriminator)
    wrong_route = copy.deepcopy(baseline)
    wrong_route["route_display_basis"] = "submitted_action"
    poisons.append(wrong_route)
    wrong_lane = copy.deepcopy(baseline)
    wrong_lane["combat_lane"] = "none"
    poisons.append(wrong_lane)
    wrong_target = copy.deepcopy(baseline)
    decision_targets = cast(
        list[dict[str, object]],
        cast(dict[str, object], wrong_target["decision_mask"])["target_actions"],
    )
    wrong_target["accepted_target"] = decision_targets[0]
    poisons.append(wrong_target)
    wrong_accepted_axis = copy.deepcopy(baseline)
    accepted = cast(dict[str, object], wrong_accepted_axis["accepted_action"])
    accepted["target_action"] = 0
    poisons.append(wrong_accepted_axis)
    illegal_move = copy.deepcopy(baseline)
    accepted_move = cast(
        int,
        cast(dict[str, object], illegal_move["accepted_action"])["move_action"],
    )
    move_mask = cast(
        list[bool],
        cast(dict[str, object], illegal_move["decision_mask"])["movement_action_mask"],
    )
    move_mask[accepted_move] = False
    poisons.append(illegal_move)
    wrong_owner_key = copy.deepcopy(baseline)
    cast(dict[str, object], wrong_owner_key["decision_mask"])[
        "owner_presentation_key"
    ] = "different-key"
    poisons.append(wrong_owner_key)
    submitted_extra = copy.deepcopy(baseline)
    cast(dict[str, object], submitted_extra["submitted_action"])["unexpected"] = True
    poisons.append(submitted_extra)
    accepted_out_of_domain = copy.deepcopy(baseline)
    cast(dict[str, object], accepted_out_of_domain["accepted_action"])[
        "use_ultimate_action"
    ] = 2
    poisons.append(accepted_out_of_domain)
    reference_extra = copy.deepcopy(baseline)
    cast(dict[str, object], reference_extra["transition_reference"])["unexpected"] = (
        True
    )
    poisons.append(reference_extra)

    for poison in poisons:
        with pytest.raises(ValidationError):
            adapter.validate_json(json.dumps(poison))


def test_live_drafts_are_disjoint_unarmed_basic_and_ultimate(
    inspection_cases: _InspectionCases,
) -> None:
    context = inspection_cases.no_shared.context
    frame = inspection_cases.no_shared.frames[0]
    scene = inspection_cases.oracle_scenes[0]
    unarmed = build_live_oracle_draft_inspection_v1(
        context,
        frame,
        scene,
        controlled_internal_slot=2,
        draft_move_action=0,
        draft_target_internal_slot=1,
        draft_armed_lane="none",
    )
    basic = build_live_oracle_draft_inspection_v1(
        context,
        frame,
        scene,
        controlled_internal_slot=2,
        draft_move_action=0,
        draft_target_internal_slot=1,
        draft_armed_lane="basic",
    )
    ultimate = build_live_oracle_draft_inspection_v1(
        context,
        frame,
        scene,
        controlled_internal_slot=0,
        draft_move_action=0,
        draft_target_internal_slot=None,
        draft_armed_lane="ultimate",
    )
    assert unarmed.draft_legality.armed_lane_is_legal is None
    assert unarmed.draft_legality.combat_pair_is_legal is None
    assert basic.draft_legality.combat_pair_is_legal is True
    assert ultimate.draft_legality.combat_pair_is_legal is True
    assert all(
        row.route_display_basis == "draft_action" for row in (unarmed, basic, ultimate)
    )

    payload = _json_payload(basic)
    recursive = json.dumps(payload, sort_keys=True)
    for forbidden in ("accepted_action", "transition_id", "successor_frame_id"):
        assert forbidden not in recursive
    with pytest.raises(ValidationError):
        TypeAdapter(ReplayInspectionPresentationV1).validate_json(
            _canonical_bytes(basic)
        )
    replay = _oracle(inspection_cases, frame_index=0, actor_slot=2)
    assert replay is not None
    with pytest.raises(ValidationError):
        TypeAdapter(LiveDraftInspectionPresentationV1).validate_json(
            _canonical_bytes(replay)
        )


def test_live_root_action_target_and_legality_contracts_reject_strict_poison(
    inspection_cases: _InspectionCases,
) -> None:
    result = build_live_oracle_draft_inspection_v1(
        inspection_cases.no_shared.context,
        inspection_cases.no_shared.frames[0],
        inspection_cases.oracle_scenes[0],
        controlled_internal_slot=2,
        draft_move_action=0,
        draft_target_internal_slot=1,
        draft_armed_lane="basic",
    )
    adapter = TypeAdapter(LiveDraftInspectionPresentationV1)
    baseline = _json_payload(result)
    assert adapter.validate_json(json.dumps(baseline)) == result

    poisons: list[dict[str, object]] = []
    missing = copy.deepcopy(baseline)
    del missing["draft_action"]
    poisons.append(missing)
    extra = copy.deepcopy(baseline)
    extra["accepted_action"] = {}
    poisons.append(extra)
    coerced = copy.deepcopy(baseline)
    coerced["current_simulator_step_count"] = "0"
    poisons.append(coerced)
    wrong_discriminator = copy.deepcopy(baseline)
    wrong_discriminator["inspection_kind"] = "replay_recorded_outgoing_action"
    poisons.append(wrong_discriminator)
    wrong_route = copy.deepcopy(baseline)
    wrong_route["route_display_basis"] = "accepted_action"
    poisons.append(wrong_route)
    wrong_owner = copy.deepcopy(baseline)
    cast(dict[str, object], wrong_owner["decision_mask"])["owner_public_agent_id"] = (
        "agent-slot-0"
    )
    poisons.append(wrong_owner)
    wrong_draft_target = copy.deepcopy(baseline)
    wrong_draft_decision = cast(dict[str, object], wrong_draft_target["decision_mask"])
    wrong_draft_targets = cast(list[object], wrong_draft_decision["target_actions"])
    wrong_draft_target["draft_target"] = cast(
        dict[str, object],
        wrong_draft_targets[0],
    )
    poisons.append(wrong_draft_target)
    wrong_draft_axis = copy.deepcopy(baseline)
    cast(dict[str, object], wrong_draft_axis["draft_action"])["target_action"] = 0
    poisons.append(wrong_draft_axis)
    wrong_legality = copy.deepcopy(baseline)
    legality = cast(dict[str, object], wrong_legality["draft_legality"])
    legality["move_action_is_legal"] = not cast(bool, legality["move_action_is_legal"])
    poisons.append(wrong_legality)
    absent_armed_legality = copy.deepcopy(baseline)
    absent_legality = cast(dict[str, object], absent_armed_legality["draft_legality"])
    absent_legality["armed_lane_is_legal"] = None
    absent_legality["combat_pair_is_legal"] = None
    poisons.append(absent_armed_legality)
    wrong_lane = copy.deepcopy(baseline)
    cast(dict[str, object], wrong_lane["draft_action"])["armed_lane"] = "other"
    poisons.append(wrong_lane)
    nested_extra = copy.deepcopy(baseline)
    cast(dict[str, object], nested_extra["draft_action"])["unexpected"] = True
    poisons.append(nested_extra)
    nested_coerced = copy.deepcopy(baseline)
    cast(dict[str, object], nested_coerced["draft_action"])["move_action"] = False
    poisons.append(nested_coerced)
    wrong_target_discriminator = copy.deepcopy(baseline)
    cast(dict[str, object], wrong_target_discriminator["draft_target"])[
        "target_kind"
    ] = "unknown"
    poisons.append(wrong_target_discriminator)
    coerced_legality = copy.deepcopy(baseline)
    cast(dict[str, object], coerced_legality["draft_legality"])[
        "combat_pair_is_legal"
    ] = 1
    poisons.append(coerced_legality)

    for poison in poisons:
        with pytest.raises(ValidationError):
            adapter.validate_json(json.dumps(poison))

    action_adapter = TypeAdapter(LiveDraftActionTupleV1)
    with pytest.raises(ValidationError):
        action_adapter.validate_json(
            '{"move_action":false,"target_action":0,"armed_lane":"none"}'
        )
    legality_adapter = TypeAdapter(LiveDraftLegalityV1)
    with pytest.raises(ValidationError):
        legality_adapter.validate_json(
            "{"
            '"move_action_is_legal":true,'
            '"target_action_is_legal":true,'
            '"armed_lane_is_legal":null,'
            '"combat_pair_is_legal":true'
            "}"
        )


def test_agent_live_drafts_fix_owner_and_hide_unseen_target_anchor(
    inspection_cases: _InspectionCases,
) -> None:
    source = _pov_slice(
        inspection_cases.no_shared,
        actor_slot=0,
        frame_index=0,
    )
    current = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=inspection_cases.no_shared.context.static_mechanics_catalog,
        authority_session_id="cp2-4-live-no-shared",
    )
    hidden_action = next(
        row.target_action
        for row in build_live_no_shared_obs_draft_inspection_v1(
            source,
            current,
            draft_move_action=0,
            draft_target_action=0,
            draft_armed_lane="none",
        ).decision_mask.target_actions
        if isinstance(row, AuthorizedAxisOnlyTargetActionV1)
    )
    no_shared = build_live_no_shared_obs_draft_inspection_v1(
        source,
        current,
        draft_move_action=0,
        draft_target_action=hidden_action,
        draft_armed_lane="basic",
    )
    assert no_shared.actor_public_agent_id == source.public_agent_id
    assert isinstance(no_shared.draft_target, AuthorizedAxisOnlyTargetActionV1)
    assert not hasattr(no_shared.draft_target, "target_anchor")

    shared_current, shared_source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=0,
        authority="cp2-4-live-shared",
    )
    shared = build_live_shared_obs_draft_inspection_v1(
        shared_current,
        shared_source,
        authorized_recipient_global_slot=0,
        draft_move_action=0,
        draft_target_action=hidden_action,
        draft_armed_lane="none",
    )
    assert shared.actor_public_agent_id == shared_current.recipient_public_agent_id
    assert isinstance(shared.draft_target, AuthorizedAxisOnlyTargetActionV1)


def test_agent_roots_recursively_omit_oracle_slots_ids_rewards_events_and_diagnostics(
    inspection_cases: _InspectionCases,
) -> None:
    no_shared_index = _pov_index(inspection_cases.no_shared, actor_slot=0)
    no_shared_current = _no_shared_current(
        inspection_cases.no_shared,
        no_shared_index,
        frame_index=0,
    )
    no_shared_replay = build_replay_no_shared_obs_inspection_v1(
        no_shared_index,
        no_shared_current,
    )
    no_shared_slice = _pov_slice(
        inspection_cases.no_shared,
        actor_slot=0,
        frame_index=0,
    )
    no_shared_live = build_live_no_shared_obs_draft_inspection_v1(
        no_shared_slice,
        no_shared_current,
        draft_move_action=0,
        draft_target_action=0,
        draft_armed_lane="none",
    )
    shared_current, shared_source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    shared_replay = build_replay_shared_obs_inspection_v1(
        shared_current,
        shared_source,
        authorized_recipient_global_slot=0,
        outgoing_transition=inspection_cases.shared.transitions[1],
        final_frame_index=3,
    )
    shared_live = build_live_shared_obs_draft_inspection_v1(
        shared_current,
        shared_source,
        authorized_recipient_global_slot=0,
        draft_move_action=0,
        draft_target_action=0,
        draft_armed_lane="none",
    )
    assert no_shared_replay is not None and shared_replay is not None

    forbidden_keys = {
        "global_slot",
        "internal_slot",
        "source_frame_id",
        "incoming_transition_id",
        "canonical_reward",
        "canonical_reward_by_agent",
        "canonical_reward_by_team",
        "events",
        "processing",
        "source_evidence",
    }
    for root, episode_id in (
        (no_shared_replay, inspection_cases.no_shared.context.identity.episode_id),
        (no_shared_live, inspection_cases.no_shared.context.identity.episode_id),
        (shared_replay, inspection_cases.shared.context.identity.episode_id),
        (shared_live, inspection_cases.shared.context.identity.episode_id),
    ):
        payload = _json_payload(root)
        pairs = _recursive_pairs(payload)
        assert not ({key for key, _value in pairs} & forbidden_keys)
        canonical_oracle_ids = {
            *(f"{episode_id}:frame:{index}" for index in range(4)),
            *(f"{episode_id}:transition:{index}" for index in range(3)),
        }
        assert not (
            {value for _key, value in pairs if isinstance(value, str)}
            & canonical_oracle_ids
        )


def test_agent_builder_signatures_fix_recipient_and_cross_recipient_inputs_reject(
    inspection_cases: _InspectionCases,
) -> None:
    for builder in (
        build_replay_no_shared_obs_inspection_v1,
        build_replay_shared_obs_inspection_v1,
        build_live_no_shared_obs_draft_inspection_v1,
        build_live_shared_obs_draft_inspection_v1,
    ):
        parameter_names = set(inspect.signature(builder).parameters)
        assert not any(
            token in name
            for name in parameter_names
            for token in ("controlled", "inspection", "actor_slot", "recipient_slot")
        )
    for shared_builder in (
        build_replay_shared_obs_inspection_v1,
        build_live_shared_obs_draft_inspection_v1,
    ):
        credential = inspect.signature(shared_builder).parameters[
            "authorized_recipient_global_slot"
        ]
        assert credential.kind is inspect.Parameter.KEYWORD_ONLY
        assert credential.default is inspect.Parameter.empty

    source_zero = _pov_index(inspection_cases.no_shared, actor_slot=0)
    source_two = _pov_index(inspection_cases.no_shared, actor_slot=2)
    current_two = _no_shared_current(
        inspection_cases.no_shared,
        source_two,
        frame_index=0,
    )
    with pytest.raises(ValueError, match="recipient frame"):
        build_replay_no_shared_obs_inspection_v1(source_zero, current_two)
    with pytest.raises(ValueError, match="recipient frame"):
        build_live_no_shared_obs_draft_inspection_v1(
            _pov_slice(
                inspection_cases.no_shared,
                actor_slot=0,
                frame_index=0,
            ),
            current_two,
            draft_move_action=0,
            draft_target_action=0,
            draft_armed_lane="none",
        )

    shared_zero, shared_source_zero = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
    )
    _shared_one, shared_source_one = _shared_current(
        inspection_cases.shared,
        recipient_slot=1,
        frame_index=1,
    )
    with pytest.raises(
        ValueError,
        match=r"authorized recipient row|current actor|do not join recipient",
    ):
        build_replay_shared_obs_inspection_v1(
            shared_zero,
            shared_source_one,
            authorized_recipient_global_slot=0,
            outgoing_transition=inspection_cases.shared.transitions[1],
            final_frame_index=3,
        )
    with pytest.raises(
        ValueError,
        match=r"authorized recipient row|current actor|do not join recipient",
    ):
        build_live_shared_obs_draft_inspection_v1(
            shared_zero,
            shared_source_one,
            authorized_recipient_global_slot=0,
            draft_move_action=0,
            draft_target_action=0,
            draft_armed_lane="none",
        )
    assert shared_source_zero.base_sensor_frame.public_agent_id != (
        shared_source_one.base_sensor_frame.public_agent_id
    )


def test_repeated_bytes_and_raw_inputs_are_not_mutated(
    inspection_cases: _InspectionCases,
) -> None:
    context_before = inspection_cases.no_shared.context.model_dump_json()
    frame_before = inspection_cases.no_shared.frames[0].model_dump_json()
    transition_before = inspection_cases.no_shared.transitions[0].model_dump_json()
    first = _oracle(inspection_cases, frame_index=0, actor_slot=2)
    second = _oracle(inspection_cases, frame_index=0, actor_slot=2)
    assert first is not None and second is not None
    assert _canonical_bytes(first) == _canonical_bytes(second)
    assert inspection_cases.no_shared.context.model_dump_json() == context_before
    assert inspection_cases.no_shared.frames[0].model_dump_json() == frame_before
    assert (
        inspection_cases.no_shared.transitions[0].model_dump_json() == transition_before
    )


def test_both_agent_replay_and_live_builders_repeat_bytes_without_mutating_raw_roots(
    inspection_cases: _InspectionCases,
) -> None:
    no_shared_index = _pov_index(inspection_cases.no_shared, actor_slot=2)
    no_shared_replay_current = _no_shared_current(
        inspection_cases.no_shared,
        no_shared_index,
        frame_index=0,
        authority="cp2-4-no-shared-replay-repeat",
    )
    index_before = _canonical_bytes(no_shared_index)
    replay_current_before = _canonical_bytes(no_shared_replay_current)
    first_no_shared_replay = build_replay_no_shared_obs_inspection_v1(
        no_shared_index,
        no_shared_replay_current,
    )
    second_no_shared_replay = build_replay_no_shared_obs_inspection_v1(
        no_shared_index,
        no_shared_replay_current,
    )
    assert first_no_shared_replay is not None
    assert second_no_shared_replay is not None
    assert _canonical_bytes(first_no_shared_replay) == _canonical_bytes(
        second_no_shared_replay
    )
    assert _canonical_bytes(no_shared_index) == index_before
    assert _canonical_bytes(no_shared_replay_current) == replay_current_before

    no_shared_slice = _pov_slice(
        inspection_cases.no_shared,
        actor_slot=2,
        frame_index=0,
    )
    no_shared_live_current = build_no_shared_obs_authorized_scene_v1(
        no_shared_slice,
        public_catalog=(inspection_cases.no_shared.context.static_mechanics_catalog),
        authority_session_id="cp2-4-no-shared-live-repeat",
    )
    slice_before = no_shared_slice.model_dump_json()
    live_current_before = _canonical_bytes(no_shared_live_current)
    first_no_shared_live = build_live_no_shared_obs_draft_inspection_v1(
        no_shared_slice,
        no_shared_live_current,
        draft_move_action=0,
        draft_target_action=0,
        draft_armed_lane="none",
    )
    second_no_shared_live = build_live_no_shared_obs_draft_inspection_v1(
        no_shared_slice,
        no_shared_live_current,
        draft_move_action=0,
        draft_target_action=0,
        draft_armed_lane="none",
    )
    assert _canonical_bytes(first_no_shared_live) == _canonical_bytes(
        second_no_shared_live
    )
    assert no_shared_slice.model_dump_json() == slice_before
    assert _canonical_bytes(no_shared_live_current) == live_current_before

    shared_current, shared_source = _shared_current(
        inspection_cases.shared,
        recipient_slot=0,
        frame_index=1,
        authority="cp2-4-shared-repeat",
    )
    shared_transition = inspection_cases.shared.transitions[1]
    shared_current_before = _canonical_bytes(shared_current)
    shared_source_before = _canonical_bytes(shared_source)
    shared_transition_before = shared_transition.model_dump_json()
    first_shared_replay = build_replay_shared_obs_inspection_v1(
        shared_current,
        shared_source,
        authorized_recipient_global_slot=0,
        outgoing_transition=shared_transition,
        final_frame_index=3,
    )
    second_shared_replay = build_replay_shared_obs_inspection_v1(
        shared_current,
        shared_source,
        authorized_recipient_global_slot=0,
        outgoing_transition=shared_transition,
        final_frame_index=3,
    )
    assert first_shared_replay is not None and second_shared_replay is not None
    assert _canonical_bytes(first_shared_replay) == _canonical_bytes(
        second_shared_replay
    )
    first_shared_live = build_live_shared_obs_draft_inspection_v1(
        shared_current,
        shared_source,
        authorized_recipient_global_slot=0,
        draft_move_action=0,
        draft_target_action=0,
        draft_armed_lane="none",
    )
    second_shared_live = build_live_shared_obs_draft_inspection_v1(
        shared_current,
        shared_source,
        authorized_recipient_global_slot=0,
        draft_move_action=0,
        draft_target_action=0,
        draft_armed_lane="none",
    )
    assert _canonical_bytes(first_shared_live) == _canonical_bytes(second_shared_live)
    assert _canonical_bytes(shared_current) == shared_current_before
    assert _canonical_bytes(shared_source) == shared_source_before
    assert shared_transition.model_dump_json() == shared_transition_before


def test_module_has_six_exact_builders_and_no_forbidden_imports() -> None:
    expected_builders = {
        "build_replay_oracle_inspection_v1",
        "build_replay_no_shared_obs_inspection_v1",
        "build_replay_shared_obs_inspection_v1",
        "build_live_oracle_draft_inspection_v1",
        "build_live_no_shared_obs_draft_inspection_v1",
        "build_live_shared_obs_draft_inspection_v1",
    }
    assert expected_builders <= set(inspection_module.__all__)
    for name in expected_builders:
        assert inspect.isfunction(getattr(inspection_module, name))

    source = Path(inspection_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in imported
        for prefix in ("jax", "jaxlib", "numpy", "marl_battlegrounds.core", "scripts")
    )
    shared_validator_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "__post_init__" not in {
        node.func.attr
        for node in ast.walk(
            next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name == "_validate_shared_current"
            )
        )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert shared_validator_calls


def test_authorized_inspection_import_is_core_jax_numpy_isolated() -> None:
    code = """
import sys
import marl_battlegrounds.rendering.authorized_inspection
forbidden = ('jax', 'jaxlib', 'numpy', 'marl_battlegrounds.core')
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden)
)
assert loaded == [], loaded
print('forbidden', loaded)
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cuda"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "forbidden []"
