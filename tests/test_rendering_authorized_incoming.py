"""Focused CP2.3 recipient-safe incoming-summary proofs."""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import TypeAdapter, ValidationError
from scripts.dev.visual_debugger.control import (
    create_session,
    submit_next_script_frame,
)
from scripts.dev.visual_debugger.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    neutral_action,
)
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.types import Action
from marl_battlegrounds.evaluation import pov as pov_module
from marl_battlegrounds.evaluation.metrics import (
    EvaluationTransitionViewV1,
    build_evaluation_observer_v1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationFrameV1,
    EvaluationTransitionV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovAdjacentTransitionSliceV1,
    ActorPovAxisMappingV1,
    ActorPovFrameV1,
    ActorPovReplayContentV1,
    ActorPovTransitionV1,
    ActorPovVisibleBodyObservationChangedCueV1,
    build_actor_pov_adjacent_transition_slice_v1,
    export_actor_pov_replay_v1,
)
from marl_battlegrounds.evaluation.replay import (
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.rendering.authorized_incoming import (
    NoSharedObsIncomingSummaryV1,
    NoSharedObsVisibleBodyChangedIncomingCueV1,
    SharedObsIncomingSummaryV1,
    SharedObsObservedValuesIncomingDeltaV1,
    build_live_no_shared_obs_incoming_summary_v1,
    build_replay_no_shared_obs_incoming_summary_v1,
    build_shared_obs_incoming_summary_v1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    SharedObsAuthorizedScenePartsV1,
    build_shared_obs_authorized_scene_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AuthorizedAgentV1,
    AuthorizedSpawnShieldMechanicsAvailableV2,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    SharedObsSourceMaterialProjectionV1,
    build_shared_obs_source_material_projection_v1,
)
from marl_battlegrounds.rendering.evaluation_wire_features import (
    AGENT_FEATURE_CURRENT_HEALTH_V1,
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION_V1,
    AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1,
    AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1,
    AGENT_FEATURE_X_V1,
    CONTEXT_FEATURE_MAP_WIDTH_V1,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovProjectionIndexV1,
    build_actor_pov_projection_index_v1,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = (
    _REPOSITORY_ROOT
    / "src"
    / "marl_battlegrounds"
    / "rendering"
    / "authorized_incoming.py"
)


def _runtime_provenance() -> RuntimeProvenanceV1:
    return RuntimeProvenanceV1(
        python_version="3.14.0",
        package_version="0.0.0",
        jax_version="0.7.0",
        jaxlib_version="0.7.0",
        numpy_version="2.3.0",
        pydantic_version="2.11.0",
        platform="linux",
        machine="x86_64",
        backend="cpu",
        device="generic-cpu",
        precision="float32",
        environment_count=1,
        batch_shape=(1,),
        policy_execution_included=False,
    )


def _transition_view(
    trajectory: CapturedEvaluationTrajectory,
    transition_index: int,
) -> EvaluationTransitionViewV1:
    return EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[transition_index],
        transition=trajectory.transitions[transition_index],
        successor_frame=trajectory.frames[transition_index + 1],
    )


def _live_carrier(
    trajectory: CapturedEvaluationTrajectory,
    *,
    transition_index: int,
    global_slot: int = 0,
) -> ActorPovAdjacentTransitionSliceV1:
    return build_actor_pov_adjacent_transition_slice_v1(
        _transition_view(trajectory, transition_index),
        global_slot=global_slot,
    )


def _replay_index(
    trajectory: CapturedEvaluationTrajectory,
    *,
    global_slot: int,
) -> ActorPovProjectionIndexV1:
    observer = build_evaluation_observer_v1(trajectory.context)
    observer.start(trajectory.frames[0])
    for transition, successor in zip(
        trajectory.transitions,
        trajectory.frames[1:],
        strict=True,
    ):
        observer.append(transition, successor)
    report = observer.finalize(
        completion_state="partial",
        end_or_failure_reason="incoming-summary-fixture",
    )
    runtime_payload = _runtime_provenance().model_dump(mode="python")
    runtime_payload["package_version"] = (
        trajectory.context.code_revision.package_version
    )
    runtime = RuntimeProvenanceV1.model_validate(runtime_payload)
    replay = build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=runtime,
    ).replay
    artifact = export_actor_pov_replay_v1(replay, global_slot=global_slot)
    return build_actor_pov_projection_index_v1(artifact.content)


def _replace_tuple_item[T](
    values: tuple[T, ...],
    index: int,
    value: T,
) -> tuple[T, ...]:
    rows = list(values)
    rows[index] = value
    return tuple(rows)


def _rewrap_content(
    source: ActorPovReplayContentV1,
    *,
    frames: tuple[ActorPovFrameV1, ...] | None = None,
    transitions: tuple[ActorPovTransitionV1, ...] | None = None,
    axis_mapping: ActorPovAxisMappingV1 | None = None,
    completion_updates: dict[str, object] | None = None,
) -> ActorPovReplayContentV1:
    payload = source.model_dump(mode="python")
    payload.pop("canonical_digest_sha256")
    if frames is not None:
        payload["frames"] = frames
    if transitions is not None:
        payload["transitions"] = transitions
    if axis_mapping is not None:
        payload["axis_mapping"] = axis_mapping
    if completion_updates is not None:
        completion = source.completion.model_dump(mode="python")
        completion.update(completion_updates)
        payload["completion"] = completion
    payload["canonical_digest_sha256"] = canonical_digest_sha256(payload)
    return ActorPovReplayContentV1.model_validate(payload)


def _exhaustive_no_shared_index(
    source: ActorPovProjectionIndexV1,
) -> ActorPovProjectionIndexV1:
    content = source.content
    original_start = content.frames[0]
    original_successor = content.frames[1]
    selected_row = content.selected_team_local_slot

    start_payload = original_start.model_dump(mode="python")
    start_visibility = _replace_tuple_item(
        original_start.ally_visibility_mask,
        selected_row,
        True,
    )
    start_payload["ally_visibility_mask"] = start_visibility
    start = ActorPovFrameV1.model_validate(start_payload)

    successor_self = list(start.self_features)
    successor_self[AGENT_FEATURE_X_V1] += 0.25
    successor_self[AGENT_FEATURE_CURRENT_HEALTH_V1] -= 1.0
    successor_self[AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1] = 1.0
    successor_self[AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1] = 1.0
    successor_self_tuple = tuple(successor_self)
    successor_ally_rows = _replace_tuple_item(
        start.ally_unit_features,
        selected_row,
        successor_self_tuple,
    )
    lifecycle_payload = start.spawn_lifecycle.model_dump(mode="python")
    shield_rows = list(start.spawn_lifecycle.spawn_shield_actual_durations_by_team)
    own_shields = list(shield_rows[0])
    start_shield = own_shields[selected_row]
    own_shields[selected_row] = start_shield - 1 if start_shield > 0 else 1
    shield_rows[0] = tuple(own_shields)
    lifecycle_payload["spawn_shield_actual_durations_by_team"] = tuple(shield_rows)
    successor_lifecycle = type(start.spawn_lifecycle).model_validate(lifecycle_payload)
    successor_payload = original_successor.model_dump(mode="python")
    successor_payload.update(
        {
            "self_features": successor_self_tuple,
            "ally_unit_features": successor_ally_rows,
            "enemy_unit_features": start.enemy_unit_features,
            "ally_visibility_mask": start_visibility,
            "enemy_visibility_mask": start.enemy_visibility_mask,
            "map_obstacle_features": start.map_obstacle_features,
            "objective_features": start.objective_features,
            "context_features": start.context_features,
            "spawn_lifecycle": successor_lifecycle,
        }
    )
    successor = ActorPovFrameV1.model_validate(successor_payload)

    source_transition = content.transitions[0]
    cues = pov_module._derive_cues(  # pyright: ignore[reportPrivateUsage]
        episode_id=content.episode_id,
        public_agent_id=content.public_agent_id,
        transition_index=0,
        team_local_slot=content.selected_team_local_slot,
        start_frame=start,
        successor_frame=successor,
        has_any_rejection=(
            source_transition.submitted_action_tuple_is_out_of_domain
            or source_transition.in_domain_move_action_is_rejected
            or source_transition.in_domain_combat_action_pair_is_rejected
        ),
        terminated=False,
        truncated=True,
        public_end_reason="exhaustive recipient-local fixture",
    )
    transition_payload = source_transition.model_dump(mode="python")
    transition_payload.update(
        {
            "truncated": True,
            "public_end_reason": "exhaustive recipient-local fixture",
            "cues": cues,
        }
    )
    transition = ActorPovTransitionV1.model_validate(transition_payload)
    rebuilt = _rewrap_content(
        content,
        frames=(start, successor),
        transitions=(transition,),
        completion_updates={
            "truncated": True,
            "public_end_or_failure_reason": "exhaustive recipient-local fixture",
        },
    )
    return build_actor_pov_projection_index_v1(rebuilt)


def _exhaustive_live_carrier(
    trajectory: CapturedEvaluationTrajectory,
) -> ActorPovAdjacentTransitionSliceV1:
    source = _live_carrier(trajectory, transition_index=1)
    selected_row = source.selected_team_local_slot
    start = source.start_frame

    successor_self = list(start.self_features)
    successor_self[AGENT_FEATURE_X_V1] += 0.25
    successor_self[AGENT_FEATURE_CURRENT_HEALTH_V1] -= 1.0
    successor_self[AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1] = 1.0
    successor_self[AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1] = 1.0
    successor_self_tuple = tuple(successor_self)
    successor_ally_rows = _replace_tuple_item(
        start.ally_unit_features,
        selected_row,
        successor_self_tuple,
    )
    lifecycle_payload = start.spawn_lifecycle.model_dump(mode="python")
    shield_rows = list(start.spawn_lifecycle.spawn_shield_actual_durations_by_team)
    own_shields = list(shield_rows[0])
    own_shields[selected_row] = own_shields[selected_row] + 1
    shield_rows[0] = tuple(own_shields)
    lifecycle_payload["spawn_shield_actual_durations_by_team"] = tuple(shield_rows)
    successor_lifecycle = type(start.spawn_lifecycle).model_validate(lifecycle_payload)
    successor_payload = source.successor_frame.model_dump(mode="python")
    successor_payload.update(
        {
            "self_features": successor_self_tuple,
            "ally_unit_features": successor_ally_rows,
            "enemy_unit_features": start.enemy_unit_features,
            "ally_visibility_mask": start.ally_visibility_mask,
            "enemy_visibility_mask": start.enemy_visibility_mask,
            "map_obstacle_features": start.map_obstacle_features,
            "objective_features": start.objective_features,
            "context_features": start.context_features,
            "spawn_lifecycle": successor_lifecycle,
        }
    )
    successor = ActorPovFrameV1.model_validate(successor_payload)
    transition_payload = source.transition.model_dump(mode="python")
    transition_payload.update(
        {
            "truncated": True,
            "public_end_reason": "exhaustive live recipient-local fixture",
            "cues": pov_module._derive_cues(  # pyright: ignore[reportPrivateUsage]
                episode_id=source.episode_id,
                public_agent_id=source.public_agent_id,
                transition_index=source.transition.transition_index,
                team_local_slot=source.selected_team_local_slot,
                start_frame=start,
                successor_frame=successor,
                has_any_rejection=(
                    source.transition.submitted_action_tuple_is_out_of_domain
                    or source.transition.in_domain_move_action_is_rejected
                    or source.transition.in_domain_combat_action_pair_is_rejected
                ),
                terminated=False,
                truncated=True,
                public_end_reason="exhaustive live recipient-local fixture",
            ),
        }
    )
    transition = ActorPovTransitionV1.model_validate(transition_payload)
    carrier_payload = source.model_dump(mode="python")
    carrier_payload.update(
        {
            "start_frame": start,
            "transition": transition,
            "successor_frame": successor,
        }
    )
    return ActorPovAdjacentTransitionSliceV1.model_validate(carrier_payload)


def _with_live_reward_mutation(
    source: ActorPovAdjacentTransitionSliceV1,
) -> ActorPovAdjacentTransitionSliceV1:
    payload = source.model_dump(mode="python")
    transition = dict(cast(dict[str, object], payload["transition"]))
    transition["canonical_reward"] = cast(float, transition["canonical_reward"]) + 123.0
    payload["transition"] = transition
    return ActorPovAdjacentTransitionSliceV1.model_validate(payload)


def _with_live_source_only_evidence_mutation(
    source: ActorPovAdjacentTransitionSliceV1,
) -> ActorPovAdjacentTransitionSliceV1:
    payload = source.model_dump(mode="python")
    for endpoint_name in ("start_frame", "successor_frame"):
        endpoint = dict(cast(dict[str, object], payload[endpoint_name]))
        previous = dict(cast(dict[str, object], endpoint["previous_timestep_actions"]))
        rows = list(
            cast(tuple[tuple[float, ...], ...], previous["ally_move_actions_one_hot"])
        )
        row = list(rows[1])
        row[0] = 0.25 if row[0] != 0.25 else 0.5
        rows[1] = tuple(row)
        previous["ally_move_actions_one_hot"] = tuple(rows)
        endpoint["previous_timestep_actions"] = previous
        payload[endpoint_name] = endpoint
    return ActorPovAdjacentTransitionSliceV1.model_validate(payload)


def _with_live_masked_relation_payload(
    source: ActorPovAdjacentTransitionSliceV1,
    *,
    endpoint_name: Literal["start_frame", "successor_frame"],
    relation: Literal["ally", "enemy"],
    observation_row: int,
    replacement: tuple[float, ...],
) -> ActorPovAdjacentTransitionSliceV1:
    endpoint = getattr(source, endpoint_name)
    visibility = getattr(endpoint, f"{relation}_visibility_mask")
    if visibility[observation_row]:
        raise AssertionError("test poison requires a masked relation row")
    payload = source.model_dump(mode="python")
    endpoint_payload = dict(cast(dict[str, object], payload[endpoint_name]))
    field_name = f"{relation}_unit_features"
    rows = list(cast(tuple[tuple[float, ...], ...], endpoint_payload[field_name]))
    rows[observation_row] = replacement
    endpoint_payload[field_name] = tuple(rows)
    payload[endpoint_name] = endpoint_payload
    return ActorPovAdjacentTransitionSliceV1.model_validate(payload)


def _with_no_shared_reward_mutation(
    source: ActorPovProjectionIndexV1,
) -> ActorPovProjectionIndexV1:
    transition_payload = source.content.transitions[0].model_dump(mode="python")
    transition_payload["canonical_reward"] += 123.0
    transition = ActorPovTransitionV1.model_validate(transition_payload)
    content = _rewrap_content(source.content, transitions=(transition,))
    return build_actor_pov_projection_index_v1(content)


def _with_no_shared_source_only_evidence_mutation(
    source: ActorPovProjectionIndexV1,
) -> ActorPovProjectionIndexV1:
    frames: list[ActorPovFrameV1] = []
    for frame in source.content.frames:
        previous = frame.previous_timestep_actions
        previous_payload = previous.model_dump(mode="python")
        rows = list(previous.ally_move_actions_one_hot)
        row = list(rows[1])
        row[0] = 0.25 if row[0] != 0.25 else 0.5
        rows[1] = tuple(row)
        previous_payload["ally_move_actions_one_hot"] = tuple(rows)
        frame_payload = frame.model_dump(mode="python")
        frame_payload["previous_timestep_actions"] = type(previous).model_validate(
            previous_payload
        )
        frames.append(ActorPovFrameV1.model_validate(frame_payload))
    content = _rewrap_content(source.content, frames=tuple(frames))
    return build_actor_pov_projection_index_v1(content)


def _with_hidden_no_shared_mutations(
    source: ActorPovProjectionIndexV1,
) -> ActorPovProjectionIndexV1:
    content = source.content
    hidden_rows = tuple(
        row
        for row in range(len(content.frames[0].enemy_visibility_mask))
        if not content.frames[0].enemy_visibility_mask[row]
        and not content.frames[1].enemy_visibility_mask[row]
    )
    if len(hidden_rows) < 2:
        raise AssertionError("fixture requires two persistently hidden enemy rows.")
    first, second = hidden_rows[:2]
    hidden_row = list(content.frames[0].enemy_unit_features[first])
    hidden_row[AGENT_FEATURE_X_V1] += 7.25
    replacement_row = tuple(hidden_row)
    frames: list[ActorPovFrameV1] = []
    for frame in content.frames:
        payload = frame.model_dump(mode="python")
        payload["enemy_unit_features"] = _replace_tuple_item(
            frame.enemy_unit_features,
            first,
            replacement_row,
        )
        frames.append(ActorPovFrameV1.model_validate(payload))

    axis_payload = content.axis_mapping.model_dump(mode="python")
    enemy_ids = list(content.axis_mapping.enemy_observation_row_public_agent_id_by_id)
    enemy_ids[first], enemy_ids[second] = enemy_ids[second], enemy_ids[first]
    enemy_ids_tuple = tuple(enemy_ids)
    axis_payload["enemy_observation_row_public_agent_id_by_id"] = enemy_ids_tuple
    axis_payload["target_action_recipient_public_agent_id_by_id"] = (
        None,
        *content.axis_mapping.ally_observation_row_public_agent_id_by_id,
        *enemy_ids_tuple,
    )
    axis = ActorPovAxisMappingV1.model_validate(axis_payload)
    rebuilt = _rewrap_content(
        content,
        frames=tuple(frames),
        axis_mapping=axis,
    )
    return build_actor_pov_projection_index_v1(rebuilt)


def _with_no_shared_successor_map_drift(
    source: ActorPovProjectionIndexV1,
) -> ActorPovProjectionIndexV1:
    frames = list(source.content.frames)
    successor_payload = frames[1].model_dump(mode="python")
    context = list(frames[1].context_features)
    context[CONTEXT_FEATURE_MAP_WIDTH_V1] += 1.0
    successor_payload["context_features"] = tuple(context)
    frames[1] = ActorPovFrameV1.model_validate(successor_payload)
    rebuilt = _rewrap_content(source.content, frames=tuple(frames))
    return build_actor_pov_projection_index_v1(rebuilt)


def _with_no_shared_lifecycle_static_drift(
    source: ActorPovProjectionIndexV1,
    *,
    field: str,
) -> ActorPovProjectionIndexV1:
    frames = list(source.content.frames)
    successor = frames[1]
    lifecycle = successor.spawn_lifecycle
    lifecycle_payload = lifecycle.model_dump(mode="python")
    if field == "spawn_shield_configured_duration":
        lifecycle_payload[field] += 1
    elif field == "spawn_shield_speed":
        lifecycle_payload[field] += 0.25
    elif field == "spawn_pad_positions_by_team":
        teams = [list(team) for team in lifecycle.spawn_pad_positions_by_team]
        position = list(teams[0][1])
        position[0] += 0.125
        teams[0][1] = tuple(position)
        lifecycle_payload[field] = tuple(tuple(team) for team in teams)
    elif field == "respawn_wave_period_step_count_by_team":
        periods = list(lifecycle.respawn_wave_period_step_count_by_team)
        periods[0] += 1
        lifecycle_payload[field] = tuple(periods)
    else:  # pragma: no cover - closed test helper domain.
        raise AssertionError(f"unknown static lifecycle poison: {field}")
    successor_payload = successor.model_dump(mode="python")
    successor_payload["spawn_lifecycle"] = type(lifecycle).model_validate(
        lifecycle_payload
    )
    frames[1] = ActorPovFrameV1.model_validate(successor_payload)
    rebuilt = _rewrap_content(source.content, frames=tuple(frames))
    return build_actor_pov_projection_index_v1(rebuilt)


def _with_no_shared_health_and_status_values(
    source: ActorPovProjectionIndexV1,
    *,
    start_health: float,
    successor_health: float,
    start_status_remaining: float,
    successor_status_remaining: float,
) -> ActorPovProjectionIndexV1:
    content = source.content
    selected_row = content.selected_team_local_slot
    frames: list[ActorPovFrameV1] = []
    for frame, health, status_remaining in zip(
        content.frames,
        (start_health, successor_health),
        (start_status_remaining, successor_status_remaining),
        strict=True,
    ):
        self_features = list(frame.self_features)
        self_features[AGENT_FEATURE_CURRENT_HEALTH_V1] = health
        self_features[AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1] = 0.0
        self_features[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION_V1] = status_remaining
        self_row = tuple(self_features)
        payload = frame.model_dump(mode="python")
        payload["self_features"] = self_row
        payload["ally_unit_features"] = _replace_tuple_item(
            frame.ally_unit_features,
            selected_row,
            self_row,
        )
        frames.append(ActorPovFrameV1.model_validate(payload))
    source_transition = content.transitions[0]
    cues = pov_module._derive_cues(  # pyright: ignore[reportPrivateUsage]
        episode_id=content.episode_id,
        public_agent_id=content.public_agent_id,
        transition_index=0,
        team_local_slot=content.selected_team_local_slot,
        start_frame=frames[0],
        successor_frame=frames[1],
        has_any_rejection=(
            source_transition.submitted_action_tuple_is_out_of_domain
            or source_transition.in_domain_move_action_is_rejected
            or source_transition.in_domain_combat_action_pair_is_rejected
        ),
        terminated=source_transition.terminated,
        truncated=source_transition.truncated,
        public_end_reason=source_transition.public_end_reason,
    )
    transition_payload = source_transition.model_dump(mode="python")
    transition_payload["cues"] = cues
    transition = ActorPovTransitionV1.model_validate(transition_payload)
    rebuilt = _rewrap_content(
        content,
        frames=tuple(frames),
        transitions=(transition,),
    )
    return build_actor_pov_projection_index_v1(rebuilt)


@pytest.fixture(scope="module")
def no_shared_case() -> tuple[CapturedEvaluationTrajectory, ActorPovProjectionIndexV1]:
    neutral = neutral_action()
    moving = Action(
        move=neutral.move.at[0].set(1),
        select_target=neutral.select_target,
        use_ultimate=neutral.use_ultimate,
    )
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        actions=(moving,),
    )
    return trajectory, _replay_index(trajectory, global_slot=0)


def _shared_parts_at_frame_zero() -> SharedObsAuthorizedScenePartsV1:
    trajectory = captured_evaluation_trajectory(
        transition_count=0,
        execution_information_mode="shared_obs",
    )
    active_slots = tuple(
        row.global_slot for row in trajectory.context.roster if row.configured_active
    )
    projection_by_slot = {
        slot: build_shared_obs_source_material_projection_v1(
            trajectory.context,
            trajectory.frames[0],
            selected_global_slot=slot,
        )
        for slot in active_slots
    }
    return build_shared_obs_authorized_scene_v1(
        projection_by_slot[0],
        all_active_nonrecipient_source_material=tuple(
            projection_by_slot[slot] for slot in active_slots if slot != 0
        ),
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id="incoming-shared-authority",
    )


@pytest.fixture(scope="module")
def shared_start() -> SharedObsAuthorizedScenePartsV1:
    return _shared_parts_at_frame_zero()


@pytest.fixture(scope="module")
def shared_trajectory() -> CapturedEvaluationTrajectory:
    return captured_evaluation_trajectory(
        transition_count=1,
        execution_information_mode="shared_obs",
    )


def _shared_parts_from_frame(
    trajectory: CapturedEvaluationTrajectory,
    frame: EvaluationFrameV1,
    *,
    projection_replacements: dict[int, SharedObsSourceMaterialProjectionV1]
    | None = None,
) -> tuple[
    SharedObsAuthorizedScenePartsV1,
    dict[int, SharedObsSourceMaterialProjectionV1],
]:
    active_slots = tuple(
        row.global_slot for row in trajectory.context.roster if row.configured_active
    )
    frame_index = frame.frame_index
    transition_view = (
        None
        if frame_index == 0
        else EvaluationTransitionViewV1(
            context=trajectory.context,
            start_frame=trajectory.frames[frame_index - 1],
            transition=trajectory.transitions[frame_index - 1],
            successor_frame=frame,
        )
    )
    projection_by_slot = {
        slot: build_shared_obs_source_material_projection_v1(
            trajectory.context,
            frame,
            selected_global_slot=slot,
            transition_view=transition_view,
        )
        for slot in active_slots
    }
    if projection_replacements is not None:
        projection_by_slot.update(projection_replacements)
    return (
        build_shared_obs_authorized_scene_v1(
            projection_by_slot[0],
            all_active_nonrecipient_source_material=tuple(
                projection_by_slot[slot] for slot in active_slots if slot != 0
            ),
            public_catalog=trajectory.context.static_mechanics_catalog,
            authority_session_id="incoming-real-shared-authority",
        ),
        projection_by_slot,
    )


def _with_shared_availability(
    frame: EvaluationFrameV1,
    *,
    recipient_slot: int,
    source_slot: int,
    available: bool,
) -> EvaluationFrameV1:
    availability = (
        frame.shared_obs_information_availability_by_recipient_and_sensor_source
    )
    assert availability is not None
    rows = list(availability)
    rows[recipient_slot] = _replace_tuple_item(
        rows[recipient_slot],
        source_slot,
        available,
    )
    payload = frame.model_dump(mode="python")
    payload["shared_obs_information_availability_by_recipient_and_sensor_source"] = (
        tuple(rows)
    )
    return EvaluationFrameV1.model_validate(payload)


def _with_inert_previous_action_mutation(
    projection: SharedObsSourceMaterialProjectionV1,
) -> SharedObsSourceMaterialProjectionV1:
    previous = projection.base_sensor_frame.previous_timestep_actions
    rows = list(previous.ally_move_actions_one_hot)
    row = list(rows[0])
    row[0] = 0.25 if row[0] != 0.25 else 0.5
    rows[0] = tuple(row)
    previous_payload = previous.model_dump(mode="python")
    previous_payload["ally_move_actions_one_hot"] = tuple(rows)
    changed_previous = type(previous).model_validate(previous_payload)
    return replace(
        projection,
        base_sensor_frame=replace(
            projection.base_sensor_frame,
            previous_timestep_actions=changed_previous,
        ),
    )


def _as_successor(
    start: SharedObsAuthorizedScenePartsV1,
    **updates: object,
) -> SharedObsAuthorizedScenePartsV1:
    frame_index = start.source_frame_index + 1
    return replace(
        start,
        source_frame_index=frame_index,
        source_recipient_frame_id=(
            f"{start.source_episode_id}:shared-obs-visual-union:"
            f"{start.recipient_public_agent_id}:frame:{frame_index}"
        ),
        source_simulator_step_count=start.source_simulator_step_count + 1,
        **updates,
    )


def _replace_agent(
    parts: SharedObsAuthorizedScenePartsV1,
    replacement: AuthorizedAgentV1,
) -> SharedObsAuthorizedScenePartsV1:
    agents = tuple(
        replacement if row.public_agent_id == replacement.public_agent_id else row
        for row in parts.scene.agents
    )
    return _as_successor(parts, scene=replace(parts.scene, agents=agents))


def _without_agent(
    parts: SharedObsAuthorizedScenePartsV1,
    *,
    public_agent_id: str,
) -> SharedObsAuthorizedScenePartsV1:
    removed = next(
        row for row in parts.scene.agents if row.public_agent_id == public_agent_id
    )
    agents = tuple(
        row for row in parts.scene.agents if row.public_agent_id != public_agent_id
    )
    represented_classes = {row.class_id for row in agents}
    scene = replace(
        parts.scene,
        agents=agents,
        aura_fields=tuple(
            row
            for row in parts.scene.aura_fields
            if row.source_public_agent_id != public_agent_id
        ),
        class_mechanics=tuple(
            row
            for row in parts.scene.class_mechanics
            if row.class_id in represented_classes
        ),
        spawn_pads=tuple(
            replace(
                row,
                assigned_presentation_key=None,
                assigned_public_agent_id=None,
            )
            if row.assigned_public_agent_id == public_agent_id
            else row
            for row in parts.scene.spawn_pads
        ),
    )
    sources = tuple(
        row
        for row in parts.authorized_sensor_sources
        if row.source_public_agent_id != public_agent_id
    )
    provenance = tuple(
        replace(
            row,
            observation_sources=tuple(
                source
                for source in row.observation_sources
                if source.source_public_agent_id != public_agent_id
            ),
        )
        for row in parts.agent_observation_provenance
        if row.agent_public_agent_id != public_agent_id
    )
    assert removed.class_id not in represented_classes
    return _as_successor(
        parts,
        scene=scene,
        authorized_sensor_sources=sources,
        agent_observation_provenance=provenance,
    )


def _canonical_bytes(value: object) -> bytes:
    return TypeAdapter(type(value)).dump_json(value)


def _recursive_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in cast(dict[object, object], value).items():
            keys.append(str(key))
            keys.extend(_recursive_keys(child))
    elif isinstance(value, list):
        for child in cast(list[object], value):
            keys.extend(_recursive_keys(child))
    return tuple(keys)


def _recursive_strings(value: object) -> tuple[str, ...]:
    rows: list[str] = []
    if isinstance(value, str):
        rows.append(value)
    elif isinstance(value, dict):
        for child in cast(dict[object, object], value).values():
            rows.extend(_recursive_strings(child))
    elif isinstance(value, list):
        for child in cast(list[object], value):
            rows.extend(_recursive_strings(child))
    return tuple(rows)


def _json_clone(value: object) -> object:
    return json.loads(json.dumps(value))


def test_no_shared_public_replay_entry_owns_frame_zero_and_adjacent_join(
    no_shared_case: tuple[
        CapturedEvaluationTrajectory,
        ActorPovProjectionIndexV1,
    ],
) -> None:
    trajectory, index = no_shared_case
    catalog = trajectory.context.static_mechanics_catalog

    assert (
        build_replay_no_shared_obs_incoming_summary_v1(
            index,
            successor_frame_index=0,
            public_catalog=catalog,
            authority_session_id="incoming-no-shared-authority",
        )
        is None
    )
    summary = build_replay_no_shared_obs_incoming_summary_v1(
        index,
        successor_frame_index=1,
        public_catalog=catalog,
        authority_session_id="incoming-no-shared-authority",
    )
    assert summary is not None
    source_transition = index.content.transitions[0]
    assert summary.incoming_transition_index == 0
    assert summary.incoming_recipient_transition_id == (
        source_transition.pov_transition_id
    )
    assert summary.incoming_start_recipient_frame_id == (
        index.content.frames[0].pov_frame_id
    )
    assert summary.incoming_successor_recipient_frame_id == (
        index.content.frames[1].pov_frame_id
    )
    assert summary.incoming_successor_simulator_step_count == (
        summary.incoming_start_simulator_step_count + 1
    )
    assert tuple(cue.cue_id for cue in summary.cues) == tuple(
        cue.cue_id for cue in source_transition.cues
    )
    assert tuple(cue.cue_type for cue in summary.cues) == tuple(
        cue.cue_type for cue in source_transition.cues
    )
    assert summary.cue_count == len(source_transition.cues)
    assert "own_position_changed" in tuple(cue.cue_type for cue in summary.cues)

    with pytest.raises(TypeError, match="exact POV index"):
        build_replay_no_shared_obs_incoming_summary_v1(
            cast(ActorPovProjectionIndexV1, index.content),
            successor_frame_index=1,
            public_catalog=catalog,
            authority_session_id="incoming-no-shared-authority",
        )
    with pytest.raises(IndexError, match="outside"):
        build_replay_no_shared_obs_incoming_summary_v1(
            index,
            successor_frame_index=True,
            public_catalog=catalog,
            authority_session_id="incoming-no-shared-authority",
        )


def test_public_no_shared_mapper_preserves_all_eight_local_cue_kinds(
    no_shared_case: tuple[
        CapturedEvaluationTrajectory,
        ActorPovProjectionIndexV1,
    ],
) -> None:
    trajectory, source = no_shared_case
    index = _exhaustive_no_shared_index(source)
    transition = index.content.transitions[0]
    assert set(cue.cue_type for cue in transition.cues) == {
        "own_action_outcome",
        "own_position_changed",
        "own_health_changed",
        "own_status_changed",
        "own_cooldown_changed",
        "own_lifecycle_changed",
        "visible_body_observation_changed",
        "episode_ended",
    }

    summary = build_replay_no_shared_obs_incoming_summary_v1(
        index,
        successor_frame_index=1,
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id="exhaustive-no-shared-authority",
    )
    assert summary is not None
    assert tuple(cue.cue_id for cue in summary.cues) == tuple(
        cue.cue_id for cue in transition.cues
    )
    assert tuple(cue.ordinal for cue in summary.cues) == tuple(
        cue.ordinal for cue in transition.cues
    )
    assert tuple(cue.cue_type for cue in summary.cues) == tuple(
        cue.cue_type for cue in transition.cues
    )
    assert summary.cue_count == len(transition.cues)
    body_rows = tuple(
        cue
        for cue in summary.cues
        if type(cue) is NoSharedObsVisibleBodyChangedIncomingCueV1
    )
    assert body_rows
    assert all(row.start_observation is not None for row in body_rows)
    assert all(row.successor_observation is not None for row in body_rows)
    payload = json.loads(_canonical_bytes(summary))
    assert set(cue["cue_type"] for cue in payload["cues"]) == {
        "own_action_outcome",
        "own_position_changed",
        "own_health_changed",
        "own_status_changed",
        "own_cooldown_changed",
        "own_lifecycle_changed",
        "visible_body_observation_changed",
        "episode_ended",
    }
    assert set(_recursive_keys(payload)).isdisjoint(
        {
            "event_id",
            "effect_source",
            "damage_event",
            "healing_event",
            "status_application",
            "status_expiry",
            "respawn_wave_event",
        }
    )


def test_live_carrier_preserves_all_eight_local_cues_at_nonzero_index() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=2)
    carrier = _exhaustive_live_carrier(trajectory)
    assert carrier.transition.transition_index == 1
    cue_kinds = tuple(cue.cue_type for cue in carrier.transition.cues)
    assert set(cue_kinds) == {
        "own_action_outcome",
        "own_position_changed",
        "own_health_changed",
        "own_status_changed",
        "own_cooldown_changed",
        "own_lifecycle_changed",
        "visible_body_observation_changed",
        "episode_ended",
    }
    assert cue_kinds[:6] == (
        "own_action_outcome",
        "own_position_changed",
        "own_health_changed",
        "own_status_changed",
        "own_cooldown_changed",
        "own_lifecycle_changed",
    )
    assert cue_kinds[-1] == "episode_ended"
    assert set(cue_kinds[6:-1]) == {"visible_body_observation_changed"}

    summary = build_live_no_shared_obs_incoming_summary_v1(
        carrier,
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id="exhaustive-live-authority",
    )
    assert tuple(cue.cue_id for cue in summary.cues) == tuple(
        cue.cue_id for cue in carrier.transition.cues
    )
    assert tuple(cue.cue_type for cue in summary.cues) == tuple(
        cue.cue_type for cue in carrier.transition.cues
    )
    assert summary.cue_count == len(carrier.transition.cues)


def test_live_summary_is_repeatable_read_only_and_matches_replay_path() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=2)
    carrier = _live_carrier(trajectory, transition_index=1)
    index = _replay_index(trajectory, global_slot=0)
    authority = "live-replay-parity-authority"
    catalog = trajectory.context.static_mechanics_catalog
    carrier_before = carrier.model_dump_json()
    catalog_before = catalog.model_dump_json()

    first = build_live_no_shared_obs_incoming_summary_v1(
        carrier,
        public_catalog=catalog,
        authority_session_id=authority,
    )
    second = build_live_no_shared_obs_incoming_summary_v1(
        carrier,
        public_catalog=catalog,
        authority_session_id=authority,
    )
    replay = build_replay_no_shared_obs_incoming_summary_v1(
        index,
        successor_frame_index=2,
        public_catalog=catalog,
        authority_session_id=authority,
    )
    assert replay is not None
    assert _canonical_bytes(first) == _canonical_bytes(second)
    assert _canonical_bytes(first) == _canonical_bytes(replay)
    assert carrier.model_dump_json() == carrier_before
    assert catalog.model_dump_json() == catalog_before


def test_live_hidden_events_reward_and_source_evidence_are_byte_inert() -> None:
    neutral = neutral_action()
    hidden_rejection = Action(
        move=neutral.move.at[5].set(-7),
        select_target=neutral.select_target,
        use_ultimate=neutral.use_ultimate,
    )
    first = captured_evaluation_trajectory(transition_count=1, actions=(neutral,))
    second = captured_evaluation_trajectory(
        transition_count=1,
        actions=(hidden_rejection,),
    )
    assert first.frames == second.frames
    assert first.transitions[0].events != second.transitions[0].events
    first_carrier = _live_carrier(first, transition_index=0)
    second_carrier = _live_carrier(second, transition_index=0)
    authority = "live-noninterference-authority"

    def build(source: ActorPovAdjacentTransitionSliceV1) -> bytes:
        return _canonical_bytes(
            build_live_no_shared_obs_incoming_summary_v1(
                source,
                public_catalog=first.context.static_mechanics_catalog,
                authority_session_id=authority,
            )
        )

    baseline = build(first_carrier)
    assert build(second_carrier) == baseline
    assert build(_with_live_reward_mutation(first_carrier)) == baseline
    assert build(_with_live_source_only_evidence_mutation(first_carrier)) == baseline
    payload = json.loads(baseline)
    assert set(_recursive_keys(payload)).isdisjoint(
        {
            "events",
            "canonical_reward",
            "previous_timestep_actions",
            "status_source_evidence",
            "effect_source",
        }
    )


def test_real_death_and_respawn_keep_recipient_self_authorized_in_live_and_replay() -> (
    None
):
    base_launch = debugger_test_launch_specification(0)
    session = create_session(
        get_scenario("death_respawn_cycle"),
        seed=0,
        evaluation_launch_specification=(
            build_debugger_evaluation_launch_specification_v1(
                root_seed=0,
                code_revision=base_launch.code_revision,
                capture_profile="evaluation_metric_complete",
            )
        ),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    frames = [session.current_evaluation_frame]
    transitions: list[EvaluationTransitionV1] = []
    carriers: list[ActorPovAdjacentTransitionSliceV1] = []
    for _ in range(3):
        session = submit_next_script_frame(session)
        view = session.incoming_evaluation_view
        assert view is not None
        frames.append(view.successor_frame)
        transitions.append(view.transition)
        carriers.append(
            build_actor_pov_adjacent_transition_slice_v1(view, global_slot=5)
        )
    death, _corpse_wait, respawn = carriers
    assert death.start_frame.self_features[5] == 1.0
    assert death.successor_frame.self_features[5] == 0.0
    assert not death.successor_frame.ally_visibility_mask[0]
    assert respawn.start_frame.self_features[5] == 0.0
    assert not respawn.start_frame.ally_visibility_mask[0]
    assert respawn.successor_frame.self_features[5] == 1.0

    catalog = session.evaluation_context.static_mechanics_catalog
    authority = "real-lifecycle-authority"
    live_summaries = tuple(
        build_live_no_shared_obs_incoming_summary_v1(
            carrier,
            public_catalog=catalog,
            authority_session_id=authority,
        )
        for carrier in (death, respawn)
    )
    death_payload = death.model_dump(mode="python")
    death_successor = dict(cast(dict[str, object], death_payload["successor_frame"]))
    death_ally_rows = list(
        cast(tuple[tuple[float, ...], ...], death_successor["ally_unit_features"])
    )
    death_ally_rows[death.selected_team_local_slot] = tuple(
        float(500 + index) for index in range(len(death.start_frame.self_features))
    )
    death_successor["ally_unit_features"] = tuple(death_ally_rows)
    death_payload["successor_frame"] = death_successor
    masked_diagonal_poison = ActorPovAdjacentTransitionSliceV1.model_validate(
        death_payload
    )
    assert _canonical_bytes(
        build_live_no_shared_obs_incoming_summary_v1(
            masked_diagonal_poison,
            public_catalog=catalog,
            authority_session_id=authority,
        )
    ) == _canonical_bytes(live_summaries[0])

    masked_diagonal_equal_to_visible_start = _with_live_masked_relation_payload(
        death,
        endpoint_name="successor_frame",
        relation="ally",
        observation_row=death.selected_team_local_slot,
        replacement=death.start_frame.ally_unit_features[
            death.selected_team_local_slot
        ],
    )
    assert _canonical_bytes(
        build_live_no_shared_obs_incoming_summary_v1(
            masked_diagonal_equal_to_visible_start,
            public_catalog=catalog,
            authority_session_id=authority,
        )
    ) == _canonical_bytes(live_summaries[0])

    def cue_public_id(
        carrier: ActorPovAdjacentTransitionSliceV1,
        cue: ActorPovVisibleBodyObservationChangedCueV1,
    ) -> str:
        public_axis = (
            carrier.axis_mapping.ally_observation_row_public_agent_id_by_id
            if cue.relation == "ally"
            else carrier.axis_mapping.enemy_observation_row_public_agent_id_by_id
        )
        return public_axis[cue.observation_row]

    disappearance = next(
        cue
        for cue in death.transition.cues
        if type(cue) is ActorPovVisibleBodyObservationChangedCueV1
        and cue.start_visible
        and not cue.successor_visible
        and cue_public_id(death, cue) != death.public_agent_id
    )
    assert disappearance.observed_payload_changed
    disappearance_start_rows = getattr(
        death.start_frame,
        f"{disappearance.relation}_unit_features",
    )
    poisoned_disappearance = _with_live_masked_relation_payload(
        death,
        endpoint_name="successor_frame",
        relation=disappearance.relation,
        observation_row=disappearance.observation_row,
        replacement=disappearance_start_rows[disappearance.observation_row],
    )
    assert _canonical_bytes(
        build_live_no_shared_obs_incoming_summary_v1(
            poisoned_disappearance,
            public_catalog=catalog,
            authority_session_id=authority,
        )
    ) == _canonical_bytes(live_summaries[0])

    appearance = next(
        cue
        for cue in respawn.transition.cues
        if type(cue) is ActorPovVisibleBodyObservationChangedCueV1
        and not cue.start_visible
        and cue.successor_visible
        and cue_public_id(respawn, cue) != respawn.public_agent_id
    )
    assert appearance.observed_payload_changed
    appearance_successor_rows = getattr(
        respawn.successor_frame,
        f"{appearance.relation}_unit_features",
    )
    poisoned_appearance = _with_live_masked_relation_payload(
        respawn,
        endpoint_name="start_frame",
        relation=appearance.relation,
        observation_row=appearance.observation_row,
        replacement=appearance_successor_rows[appearance.observation_row],
    )
    assert _canonical_bytes(
        build_live_no_shared_obs_incoming_summary_v1(
            poisoned_appearance,
            public_catalog=catalog,
            authority_session_id=authority,
        )
    ) == _canonical_bytes(live_summaries[1])

    replay_index = _replay_index(
        CapturedEvaluationTrajectory(
            context=session.evaluation_context,
            frames=tuple(frames),
            transitions=tuple(transitions),
        ),
        global_slot=5,
    )
    replay_summaries = tuple(
        build_replay_no_shared_obs_incoming_summary_v1(
            replay_index,
            successor_frame_index=frame_index,
            public_catalog=catalog,
            authority_session_id=authority,
        )
        for frame_index in (1, 3)
    )
    assert all(summary is not None for summary in replay_summaries)

    for carrier, live, replay in zip(
        (death, respawn),
        live_summaries,
        replay_summaries,
        strict=True,
    ):
        assert replay is not None
        assert _canonical_bytes(live) == _canonical_bytes(replay)
        recipient_cues = tuple(
            cue
            for cue in live.cues
            if type(cue) is NoSharedObsVisibleBodyChangedIncomingCueV1
            and cue.agent_public_agent_id == carrier.public_agent_id
        )
        assert len(recipient_cues) == 1
        cue = recipient_cues[0]
        assert cue.observation_change_kind == "observed_values_change"
        assert cue.start_observation is not None
        assert cue.successor_observation is not None
        assert (
            cue.start_observation.life_state,
            cue.successor_observation.life_state,
        ) in {
            ("alive", "corpse"),
            ("corpse", "alive"),
        }


def test_no_shared_hidden_rows_public_ids_and_reward_are_byte_inert(
    no_shared_case: tuple[
        CapturedEvaluationTrajectory,
        ActorPovProjectionIndexV1,
    ],
) -> None:
    trajectory, source = no_shared_case
    catalog = trajectory.context.static_mechanics_catalog

    def build(index: ActorPovProjectionIndexV1) -> bytes:
        summary = build_replay_no_shared_obs_incoming_summary_v1(
            index,
            successor_frame_index=1,
            public_catalog=catalog,
            authority_session_id="no-shared-noninterference-authority",
        )
        assert summary is not None
        return _canonical_bytes(summary)

    baseline = build(source)
    assert build(_with_no_shared_reward_mutation(source)) == baseline
    assert build(_with_no_shared_source_only_evidence_mutation(source)) == baseline
    assert build(_with_hidden_no_shared_mutations(source)) == baseline

    static_poisons = (
        (_with_no_shared_successor_map_drift(source), "static map"),
        (
            _with_no_shared_lifecycle_static_drift(
                source,
                field="spawn_shield_configured_duration",
            ),
            "spawn-shield mechanics",
        ),
        (
            _with_no_shared_lifecycle_static_drift(
                source,
                field="spawn_shield_speed",
            ),
            "spawn-shield mechanics",
        ),
        (
            _with_no_shared_lifecycle_static_drift(
                source,
                field="spawn_pad_positions_by_team",
            ),
            "static spawn-pad",
        ),
        (
            _with_no_shared_lifecycle_static_drift(
                source,
                field="respawn_wave_period_step_count_by_team",
            ),
            "respawn-wave profile",
        ),
    )
    for poisoned, expected in static_poisons:
        with pytest.raises(ValueError, match=expected):
            build_replay_no_shared_obs_incoming_summary_v1(
                poisoned,
                successor_frame_index=1,
                public_catalog=catalog,
                authority_session_id="no-shared-noninterference-authority",
            )


def test_no_shared_health_sign_and_status_decrement_remain_generic(
    no_shared_case: tuple[
        CapturedEvaluationTrajectory,
        ActorPovProjectionIndexV1,
    ],
) -> None:
    trajectory, source = no_shared_case
    exhaustive = _exhaustive_no_shared_index(source)
    maximum_health = exhaustive.content.frames[0].self_features[13]
    cases = (
        (maximum_health - 2.0, maximum_health - 1.0),
        (maximum_health - 1.0, maximum_health - 2.0),
    )
    for start_health, successor_health in cases:
        index = _with_no_shared_health_and_status_values(
            exhaustive,
            start_health=start_health,
            successor_health=successor_health,
            start_status_remaining=2.0,
            successor_status_remaining=1.0,
        )
        summary = build_replay_no_shared_obs_incoming_summary_v1(
            index,
            successor_frame_index=1,
            public_catalog=trajectory.context.static_mechanics_catalog,
            authority_session_id="generic-values-authority",
        )
        assert summary is not None
        payload = cast(
            dict[str, object],
            json.loads(TypeAdapter(NoSharedObsIncomingSummaryV1).dump_json(summary)),
        )
        cue_rows = cast(list[dict[str, object]], payload["cues"])
        health = next(
            row for row in cue_rows if row["cue_type"] == "own_health_changed"
        )
        status = next(
            row for row in cue_rows if row["cue_type"] == "own_status_changed"
        )
        assert health["start_health"] == start_health
        assert health["successor_health"] == successor_health
        start_statuses = cast(list[dict[str, object]], status["start_statuses"])
        successor_statuses = cast(list[dict[str, object]], status["successor_statuses"])
        hunter_start = next(row for row in start_statuses if row["status_channel"] == 4)
        hunter_successor = next(
            row for row in successor_statuses if row["status_channel"] == 4
        )
        assert hunter_start["remaining_duration"] == 2
        assert hunter_successor["remaining_duration"] == 1
        assert set(row["cue_type"] for row in cue_rows).isdisjoint(
            {
                "damage",
                "healing",
                "regeneration",
                "status_application",
                "status_expiry",
                "respawn_wave",
            }
        )
        poisoned = cast(dict[str, object], _json_clone(payload))
        poisoned_status_cue = next(
            row
            for row in cast(list[dict[str, object]], poisoned["cues"])
            if row["cue_type"] == "own_status_changed"
        )
        poisoned_successor_statuses = cast(
            list[dict[str, object]],
            poisoned_status_cue["successor_statuses"],
        )
        poisoned_hunter = next(
            row for row in poisoned_successor_statuses if row["status_channel"] == 4
        )
        poisoned_hunter["configured_duration_steps"] = (
            cast(int, poisoned_hunter["configured_duration_steps"]) + 1
        )
        with pytest.raises(ValidationError):
            TypeAdapter(NoSharedObsIncomingSummaryV1).validate_json(
                json.dumps(poisoned)
            )


def test_shared_frame_zero_empty_delta_and_dynamic_change(
    shared_start: SharedObsAuthorizedScenePartsV1,
) -> None:
    assert build_shared_obs_incoming_summary_v1(None, shared_start) is None
    unchanged = build_shared_obs_incoming_summary_v1(
        shared_start,
        _as_successor(shared_start),
    )
    assert unchanged is not None
    assert unchanged.deltas == ()
    assert unchanged.delta_count == 0

    agent = shared_start.scene.agents[-1]
    changed = replace(agent, current_health=agent.current_health - 1.0)
    summary = build_shared_obs_incoming_summary_v1(
        shared_start,
        _replace_agent(shared_start, changed),
    )
    assert summary is not None
    assert tuple(delta.delta_kind for delta in summary.deltas) == (
        "observed_values_change",
    )
    delta = summary.deltas[0]
    assert type(delta) is SharedObsObservedValuesIncomingDeltaV1
    assert delta.agent_public_agent_id == agent.public_agent_id
    assert delta.changed_dynamic_fields == ("current_health",)
    assert summary.incoming_recipient_transition_id == (
        f"{shared_start.source_episode_id}:shared-obs-visual-union:"
        f"{shared_start.recipient_public_agent_id}:transition:0"
    )


def test_shared_provenance_entry_exit_and_static_conflict(
    shared_start: SharedObsAuthorizedScenePartsV1,
) -> None:
    public_id = shared_start.scene.agents[-1].public_agent_id
    provenance_rows = list(shared_start.agent_observation_provenance)
    row_index = next(
        index
        for index, row in enumerate(provenance_rows)
        if row.agent_public_agent_id == public_id
    )
    row = provenance_rows[row_index]
    provenance_rows[row_index] = replace(
        row,
        observation_sources=(row.observation_sources[0], row.observation_sources[-1]),
    )
    provenance_successor = _as_successor(
        shared_start,
        agent_observation_provenance=tuple(provenance_rows),
    )
    provenance_summary = build_shared_obs_incoming_summary_v1(
        shared_start,
        provenance_successor,
    )
    assert provenance_summary is not None
    assert tuple(delta.delta_kind for delta in provenance_summary.deltas) == (
        "observation_provenance_change",
    )

    without_agent = _without_agent(shared_start, public_agent_id=public_id)
    disappearance = build_shared_obs_incoming_summary_v1(
        shared_start,
        without_agent,
    )
    assert disappearance is not None
    disappeared = tuple(
        delta
        for delta in disappearance.deltas
        if delta.agent_public_agent_id == public_id
    )
    assert tuple(delta.delta_kind for delta in disappeared) == ("disappearance",)

    start_without_agent = replace(
        without_agent,
        source_frame_index=0,
        source_recipient_frame_id=(
            f"{without_agent.source_episode_id}:shared-obs-visual-union:"
            f"{without_agent.recipient_public_agent_id}:frame:0"
        ),
        source_simulator_step_count=shared_start.source_simulator_step_count,
    )
    appearance = build_shared_obs_incoming_summary_v1(
        start_without_agent,
        _as_successor(shared_start),
    )
    assert appearance is not None
    appeared = tuple(
        delta for delta in appearance.deltas if delta.agent_public_agent_id == public_id
    )
    assert tuple(delta.delta_kind for delta in appeared) == ("appearance",)

    agent = shared_start.scene.agents[-1]
    static_drift = replace(
        agent,
        base_movement_speed=agent.base_movement_speed + 0.25,
    )
    with pytest.raises(ValueError, match="static profile"):
        build_shared_obs_incoming_summary_v1(
            shared_start,
            _replace_agent(shared_start, static_drift),
        )


def test_shared_real_endpoint_availability_flip_and_unavailable_source_inertness(
    shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    start, _ = _shared_parts_from_frame(
        shared_trajectory,
        shared_trajectory.frames[0],
    )
    successor_frame = _with_shared_availability(
        shared_trajectory.frames[1],
        recipient_slot=0,
        source_slot=2,
        available=False,
    )
    successor, projections = _shared_parts_from_frame(
        shared_trajectory,
        successor_frame,
    )
    summary = build_shared_obs_incoming_summary_v1(start, successor)
    assert summary is not None
    assert "agent-slot-2" not in tuple(
        source.source_public_agent_id for source in successor.authorized_sensor_sources
    )
    provenance_deltas = tuple(
        delta
        for delta in summary.deltas
        if delta.delta_kind == "observation_provenance_change"
    )
    assert provenance_deltas
    assert all(
        "agent-slot-2"
        in tuple(
            source.source_public_agent_id for source in delta.start_observation_sources
        )
        and "agent-slot-2"
        not in tuple(
            source.source_public_agent_id
            for source in delta.successor_observation_sources
        )
        for delta in provenance_deltas
    )

    changed_unavailable = _with_inert_previous_action_mutation(projections[2])
    rebuilt_successor, _ = _shared_parts_from_frame(
        shared_trajectory,
        successor_frame,
        projection_replacements={2: changed_unavailable},
    )
    rebuilt_summary = build_shared_obs_incoming_summary_v1(
        start,
        rebuilt_successor,
    )
    assert _canonical_bytes(rebuilt_summary) == _canonical_bytes(summary)


def test_shared_static_scene_poison_rejects_and_dynamic_controls_remain_allowed(
    shared_start: SharedObsAuthorizedScenePartsV1,
) -> None:
    scene = shared_start.scene
    changed_map = replace(scene, map=replace(scene.map, width=scene.map.width + 1.0))
    shield_mechanics = scene.spawn_shield_mechanics
    assert type(shield_mechanics) is AuthorizedSpawnShieldMechanicsAvailableV2
    changed_shield = replace(
        scene,
        spawn_shield_mechanics=replace(
            shield_mechanics,
            movement_speed=shield_mechanics.movement_speed + 0.25,
        ),
    )
    first_pad = scene.spawn_pads[0]
    changed_pad = replace(
        scene,
        spawn_pads=(
            replace(
                first_pad,
                position=(first_pad.position[0] + 0.125, first_pad.position[1]),
            ),
            *scene.spawn_pads[1:],
        ),
    )
    first_wave = scene.respawn_waves[0]
    changed_wave_profile = replace(
        scene,
        respawn_waves=(
            replace(first_wave, period_steps=first_wave.period_steps + 1),
            scene.respawn_waves[1],
        ),
    )
    first_mechanics = scene.class_mechanics[0]
    changed_mechanics = replace(
        scene,
        class_mechanics=(
            replace(
                first_mechanics,
                basic_raw_damage=first_mechanics.basic_raw_damage + 0.5,
            ),
            *scene.class_mechanics[1:],
        ),
    )
    for poisoned_scene, expected in (
        (changed_map, "static map"),
        (changed_shield, "spawn-shield mechanics"),
        (changed_pad, "static spawn-pad"),
        (changed_wave_profile, "respawn-wave profile"),
        (changed_mechanics, "class mechanics"),
    ):
        successor = _as_successor(shared_start, scene=poisoned_scene)
        with pytest.raises(ValueError, match=expected):
            build_shared_obs_incoming_summary_v1(shared_start, successor)

    next_countdown = (first_wave.countdown_steps + 1) % first_wave.period_steps
    dynamic_waves = replace(
        scene,
        respawn_waves=(
            replace(first_wave, countdown_steps=next_countdown),
            scene.respawn_waves[1],
        ),
    )
    mask = shared_start.next_decision_action_mask
    mask_payload = mask.model_dump(mode="python")
    moves = list(mask.move)
    moves[-1] = not moves[-1]
    mask_payload["move"] = tuple(moves)
    changed_mask = type(mask).model_validate(mask_payload)
    allowed_successor = _as_successor(
        shared_start,
        scene=dynamic_waves,
        next_decision_action_mask=changed_mask,
    )
    allowed = build_shared_obs_incoming_summary_v1(
        shared_start,
        allowed_successor,
    )
    assert allowed is not None
    assert allowed.deltas == ()


def test_nonempty_agent_variants_exclude_oracle_axes_and_causal_event_claims(
    no_shared_case: tuple[
        CapturedEvaluationTrajectory,
        ActorPovProjectionIndexV1,
    ],
    shared_start: SharedObsAuthorizedScenePartsV1,
    shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    trajectory, source = no_shared_case
    exhaustive = _exhaustive_no_shared_index(source)
    no_shared = build_replay_no_shared_obs_incoming_summary_v1(
        exhaustive,
        successor_frame_index=1,
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id="variant-scan-no-shared",
    )
    assert no_shared is not None

    changed_agent = replace(
        shared_start.scene.agents[-1],
        current_health=shared_start.scene.agents[-1].current_health - 1.0,
    )
    values = build_shared_obs_incoming_summary_v1(
        shared_start,
        _replace_agent(shared_start, changed_agent),
    )
    removed_public = shared_start.scene.agents[-1].public_agent_id
    removed = _without_agent(shared_start, public_agent_id=removed_public)
    disappearance = build_shared_obs_incoming_summary_v1(shared_start, removed)
    removed_as_start = replace(
        removed,
        source_frame_index=0,
        source_recipient_frame_id=(
            f"{removed.source_episode_id}:shared-obs-visual-union:"
            f"{removed.recipient_public_agent_id}:frame:0"
        ),
        source_simulator_step_count=shared_start.source_simulator_step_count,
    )
    appearance = build_shared_obs_incoming_summary_v1(
        removed_as_start,
        _as_successor(shared_start),
    )
    real_start, _ = _shared_parts_from_frame(
        shared_trajectory,
        shared_trajectory.frames[0],
    )
    unavailable_frame = _with_shared_availability(
        shared_trajectory.frames[1],
        recipient_slot=0,
        source_slot=2,
        available=False,
    )
    real_successor, _ = _shared_parts_from_frame(
        shared_trajectory,
        unavailable_frame,
    )
    provenance = build_shared_obs_incoming_summary_v1(real_start, real_successor)
    shared_summaries = tuple(
        summary
        for summary in (values, appearance, disappearance, provenance)
        if summary is not None
    )
    assert {
        delta.delta_kind for summary in shared_summaries for delta in summary.deltas
    } == {
        "appearance",
        "disappearance",
        "observed_values_change",
        "observation_provenance_change",
    }

    forbidden_keys = {
        "global_slot",
        "observation_row",
        "source_frame_id",
        "incoming_transition_id",
        "event_id",
        "event_type",
        "canonical_reward",
        "raw_observation",
        "status_source_evidence",
        "processing",
        "respawn_wave",
    }
    roots: tuple[object, ...] = (no_shared, *shared_summaries)
    for root in roots:
        payload = json.loads(TypeAdapter(type(root)).dump_json(root))
        assert set(_recursive_keys(payload)).isdisjoint(forbidden_keys)
        strings = set(_recursive_strings(payload))
        episode_id = cast(dict[str, object], payload)["source_episode_id"]
        assert f"{episode_id}:frame:0" not in strings
        assert f"{episode_id}:frame:1" not in strings
        assert f"{episode_id}:transition:0" not in strings


def test_agent_summaries_are_strict_stable_and_noncausal(
    no_shared_case: tuple[
        CapturedEvaluationTrajectory,
        ActorPovProjectionIndexV1,
    ],
    shared_start: SharedObsAuthorizedScenePartsV1,
) -> None:
    trajectory, index = no_shared_case
    no_shared = build_replay_no_shared_obs_incoming_summary_v1(
        index,
        successor_frame_index=1,
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id="strict-authority",
    )
    assert no_shared is not None
    shared = build_shared_obs_incoming_summary_v1(
        shared_start,
        _as_successor(shared_start),
    )
    assert shared is not None

    for root_type, value in (
        (NoSharedObsIncomingSummaryV1, no_shared),
        (SharedObsIncomingSummaryV1, shared),
    ):
        adapter = TypeAdapter(root_type)
        encoded = adapter.dump_json(value)
        assert encoded == adapter.dump_json(value)
        assert adapter.validate_json(encoded) == value
        payload = json.loads(encoded)
        keys = set(_recursive_keys(payload))
        assert keys.isdisjoint(
            {
                "global_slot",
                "observation_row",
                "source_frame_id",
                "event_id",
                "canonical_reward",
                "incoming_transition_id",
            }
        )
        text = encoded.decode()
        assert "oracle" not in text.lower()
        assert ":event:" not in text

        missing = dict(payload)
        missing.pop("source_episode_id")
        with pytest.raises(ValidationError):
            adapter.validate_json(json.dumps(missing))
        extra = dict(payload)
        extra["global_slot"] = 0
        with pytest.raises(ValidationError):
            adapter.validate_json(json.dumps(extra))
        coerced = dict(payload)
        coerced["incoming_transition_index"] = "0"
        with pytest.raises(ValidationError):
            adapter.validate_json(json.dumps(coerced))


def test_no_shared_nested_json_and_inventory_poison_rejects(
    no_shared_case: tuple[
        CapturedEvaluationTrajectory,
        ActorPovProjectionIndexV1,
    ],
) -> None:
    trajectory, source = no_shared_case
    index = _exhaustive_no_shared_index(source)
    summary = build_replay_no_shared_obs_incoming_summary_v1(
        index,
        successor_frame_index=1,
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id="nested-no-shared-authority",
    )
    assert summary is not None
    adapter = TypeAdapter(NoSharedObsIncomingSummaryV1)
    original = cast(dict[str, object], json.loads(adapter.dump_json(summary)))

    def invalid(mutator: Callable[[dict[str, object]], None]) -> None:
        payload = cast(dict[str, object], _json_clone(original))
        mutator(payload)
        with pytest.raises(ValidationError):
            adapter.validate_json(json.dumps(payload))

    def cues(payload: dict[str, object]) -> list[dict[str, object]]:
        return cast(list[dict[str, object]], payload["cues"])

    def cue(payload: dict[str, object], cue_type: str) -> dict[str, object]:
        return next(row for row in cues(payload) if row["cue_type"] == cue_type)

    def reindex(payload: dict[str, object]) -> None:
        transition_id = cast(str, payload["incoming_recipient_transition_id"])
        for ordinal, row in enumerate(cues(payload)):
            row["ordinal"] = ordinal
            row["pov_transition_id"] = transition_id
            row["cue_id"] = f"{transition_id}:cue:{ordinal}"
        payload["cue_count"] = len(cues(payload))

    invalid(lambda payload: cues(payload)[0].__setitem__("unexpected", True))
    invalid(lambda payload: cues(payload)[0].__setitem__("cue_type", "damage"))

    def nested_observation_extra(payload: dict[str, object]) -> None:
        body = cue(payload, "visible_body_observation_changed")
        successor = cast(dict[str, object], body["successor_observation"])
        successor["unexpected"] = True

    invalid(nested_observation_extra)

    def nested_observation_missing(payload: dict[str, object]) -> None:
        body = cue(payload, "visible_body_observation_changed")
        successor = cast(dict[str, object], body["successor_observation"])
        successor.pop("class_name")

    invalid(nested_observation_missing)

    def nested_observation_coercion(payload: dict[str, object]) -> None:
        body = cue(payload, "visible_body_observation_changed")
        successor = cast(dict[str, object], body["successor_observation"])
        successor["current_health"] = str(successor["current_health"])

    invalid(nested_observation_coercion)

    def class_name_poison(payload: dict[str, object]) -> None:
        body = cue(payload, "visible_body_observation_changed")
        successor = cast(dict[str, object], body["successor_observation"])
        successor["class_name"] = "Priest"

    invalid(class_name_poison)

    def regen_poison(payload: dict[str, object]) -> None:
        body = cue(payload, "visible_body_observation_changed")
        successor = cast(dict[str, object], body["successor_observation"])
        successor["out_of_combat_health_regeneration_fraction_per_step"] = 1.5

    invalid(regen_poison)

    def neutral_aura_poison(payload: dict[str, object]) -> None:
        body = cue(payload, "visible_body_observation_changed")
        successor = cast(dict[str, object], body["successor_observation"])
        successor["aura_modifiers"] = [
            {"aura_id": "mage_damage_amplification", "multiplier": 1.0}
        ]

    invalid(neutral_aura_poison)

    def reordered_aura_poison(payload: dict[str, object]) -> None:
        body = cue(payload, "visible_body_observation_changed")
        successor = cast(dict[str, object], body["successor_observation"])
        successor["aura_modifiers"] = [
            {"aura_id": "warrior_damage_mitigation", "multiplier": 0.85},
            {"aura_id": "mage_damage_amplification", "multiplier": 1.15},
        ]

    invalid(reordered_aura_poison)

    def status_id_poison(payload: dict[str, object]) -> None:
        status_cue = cue(payload, "own_status_changed")
        statuses = cast(list[dict[str, object]], status_cue["successor_statuses"])
        statuses[0]["status_id"] = "not-the-channel-status"

    invalid(status_id_poison)

    def duplicate_status_poison(payload: dict[str, object]) -> None:
        status_cue = cue(payload, "own_status_changed")
        statuses = cast(list[dict[str, object]], status_cue["successor_statuses"])
        statuses.append(cast(dict[str, object], _json_clone(statuses[0])))

    invalid(duplicate_status_poison)

    def reordered_status_poison(payload: dict[str, object]) -> None:
        status_cue = cue(payload, "own_status_changed")
        statuses = cast(list[dict[str, object]], status_cue["successor_statuses"])
        extra = cast(dict[str, object], _json_clone(statuses[0]))
        extra["status_channel"] = 4
        extra["status_id"] = "hunter_trap_stun"
        statuses.insert(0, extra)

    invalid(reordered_status_poison)

    def duplicate_outcome(payload: dict[str, object]) -> None:
        rows = cues(payload)
        rows.insert(1, cast(dict[str, object], _json_clone(rows[0])))
        reindex(payload)

    invalid(duplicate_outcome)

    def family_reorder(payload: dict[str, object]) -> None:
        rows = cues(payload)
        position_index = next(
            index
            for index, row in enumerate(rows)
            if row["cue_type"] == "own_position_changed"
        )
        health_index = next(
            index
            for index, row in enumerate(rows)
            if row["cue_type"] == "own_health_changed"
        )
        rows[position_index], rows[health_index] = (
            rows[health_index],
            rows[position_index],
        )
        reindex(payload)

    invalid(family_reorder)

    def inactive_recipient(payload: dict[str, object]) -> None:
        lifecycle = cue(payload, "own_lifecycle_changed")
        lifecycle["successor_active"] = False

    invalid(inactive_recipient)

    def recipient_relation_poison(payload: dict[str, object]) -> None:
        body = cue(payload, "visible_body_observation_changed")
        cast(dict[str, object], body["start_observation"])["relation"] = "ally"
        cast(dict[str, object], body["successor_observation"])["relation"] = "ally"

    invalid(recipient_relation_poison)

    def recipient_key_bijection_poison(payload: dict[str, object]) -> None:
        body = cue(payload, "visible_body_observation_changed")
        forged_key = f"pov_{'f' * 64}"
        body["agent_presentation_key"] = forged_key
        cast(dict[str, object], body["start_observation"])["presentation_key"] = (
            forged_key
        )
        cast(dict[str, object], body["successor_observation"])["presentation_key"] = (
            forged_key
        )

    invalid(recipient_key_bijection_poison)

    def nonrecipient_key_alias_poison(payload: dict[str, object]) -> None:
        body = cue(payload, "visible_body_observation_changed")
        forged_public = "forged-nonrecipient"
        body["agent_public_agent_id"] = forged_public
        cast(dict[str, object], body["start_observation"])["public_agent_id"] = (
            forged_public
        )
        cast(dict[str, object], body["successor_observation"])["public_agent_id"] = (
            forged_public
        )

    invalid(nonrecipient_key_alias_poison)

    def retained_static_poison(payload: dict[str, object]) -> None:
        body = cue(payload, "visible_body_observation_changed")
        successor = cast(dict[str, object], body["successor_observation"])
        successor["base_movement_speed"] = (
            cast(float, successor["base_movement_speed"]) + 0.25
        )

    invalid(retained_static_poison)


def test_shared_nested_json_group_and_source_poison_rejects(
    shared_start: SharedObsAuthorizedScenePartsV1,
    shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    changed_agent = replace(
        shared_start.scene.agents[-1],
        current_health=shared_start.scene.agents[-1].current_health - 1.0,
    )
    values_summary = build_shared_obs_incoming_summary_v1(
        shared_start,
        _replace_agent(shared_start, changed_agent),
    )
    assert values_summary is not None and values_summary.deltas
    adapter = TypeAdapter(SharedObsIncomingSummaryV1)
    original = cast(dict[str, object], json.loads(adapter.dump_json(values_summary)))

    def invalid_values(mutator: Callable[[dict[str, object]], None]) -> None:
        payload = cast(dict[str, object], _json_clone(original))
        mutator(payload)
        with pytest.raises(ValidationError):
            adapter.validate_json(json.dumps(payload))

    def deltas(payload: dict[str, object]) -> list[dict[str, object]]:
        return cast(list[dict[str, object]], payload["deltas"])

    def reindex(payload: dict[str, object]) -> None:
        transition_id = cast(str, payload["incoming_recipient_transition_id"])
        for ordinal, row in enumerate(deltas(payload)):
            row["ordinal"] = ordinal
            row["recipient_transition_id"] = transition_id
            row["cue_id"] = f"{transition_id}:cue:{ordinal}"
        payload["delta_count"] = len(deltas(payload))

    invalid_values(lambda payload: deltas(payload)[0].__setitem__("unexpected", True))
    invalid_values(
        lambda payload: deltas(payload)[0].__setitem__("delta_kind", "damage")
    )

    def nested_value_coercion(payload: dict[str, object]) -> None:
        successor = cast(dict[str, object], deltas(payload)[0]["successor_observation"])
        successor["current_health"] = str(successor["current_health"])

    invalid_values(nested_value_coercion)

    def retained_static_poison(payload: dict[str, object]) -> None:
        successor = cast(dict[str, object], deltas(payload)[0]["successor_observation"])
        successor["base_movement_speed"] = (
            cast(float, successor["base_movement_speed"]) + 0.25
        )

    invalid_values(retained_static_poison)

    def duplicate_values(payload: dict[str, object]) -> None:
        rows = deltas(payload)
        rows.append(cast(dict[str, object], _json_clone(rows[0])))
        reindex(payload)

    invalid_values(duplicate_values)

    start, _ = _shared_parts_from_frame(
        shared_trajectory,
        shared_trajectory.frames[0],
    )
    successor_frame = _with_shared_availability(
        shared_trajectory.frames[1],
        recipient_slot=0,
        source_slot=2,
        available=False,
    )
    successor, _ = _shared_parts_from_frame(shared_trajectory, successor_frame)
    provenance_summary = build_shared_obs_incoming_summary_v1(start, successor)
    assert provenance_summary is not None
    provenance_payload = cast(
        dict[str, object],
        json.loads(adapter.dump_json(provenance_summary)),
    )
    provenance_row = next(
        row
        for row in cast(list[dict[str, object]], provenance_payload["deltas"])
        if row["delta_kind"] == "observation_provenance_change"
    )

    def invalid_provenance(mutator: Callable[[dict[str, object]], None]) -> None:
        payload = cast(dict[str, object], _json_clone(provenance_payload))
        row = next(
            item
            for item in cast(list[dict[str, object]], payload["deltas"])
            if item["cue_id"] == provenance_row["cue_id"]
        )
        mutator(row)
        with pytest.raises(ValidationError):
            adapter.validate_json(json.dumps(payload))

    def source_extra(row: dict[str, object]) -> None:
        sources = cast(list[dict[str, object]], row["start_observation_sources"])
        sources[0]["unexpected"] = True

    invalid_provenance(source_extra)

    def source_missing(row: dict[str, object]) -> None:
        sources = cast(list[dict[str, object]], row["start_observation_sources"])
        sources[0].pop("source_public_agent_id")

    invalid_provenance(source_missing)

    def source_discriminator(row: dict[str, object]) -> None:
        sources = cast(list[dict[str, object]], row["start_observation_sources"])
        sources[0]["source_kind"] = "effect_source"

    invalid_provenance(source_discriminator)

    def duplicate_source(row: dict[str, object]) -> None:
        sources = cast(list[dict[str, object]], row["start_observation_sources"])
        sources.append(cast(dict[str, object], _json_clone(sources[-1])))

    invalid_provenance(duplicate_source)

    def reorder_sources(row: dict[str, object]) -> None:
        sources = cast(list[dict[str, object]], row["start_observation_sources"])
        sources.reverse()

    invalid_provenance(reorder_sources)

    def recipient_base_key_poison(row: dict[str, object]) -> None:
        sources = cast(list[dict[str, object]], row["start_observation_sources"])
        recipient = next(
            source for source in sources if source["source_kind"] == "recipient_base"
        )
        recipient["source_presentation_key"] = f"pov_{'e' * 64}"

    invalid_provenance(recipient_base_key_poison)

    def cross_endpoint_source_bijection_poison(row: dict[str, object]) -> None:
        start_sources = cast(list[dict[str, object]], row["start_observation_sources"])
        successor_sources = cast(
            list[dict[str, object]], row["successor_observation_sources"]
        )
        common_public = next(
            cast(str, source["source_public_agent_id"])
            for source in start_sources
            if any(
                candidate["source_public_agent_id"] == source["source_public_agent_id"]
                for candidate in successor_sources
            )
        )
        successor_source = next(
            source
            for source in successor_sources
            if source["source_public_agent_id"] == common_public
        )
        successor_source["source_presentation_key"] = f"pov_{'d' * 64}"

    invalid_provenance(cross_endpoint_source_bijection_poison)

    appearing_public_id = shared_start.scene.agents[-1].public_agent_id
    start_without_agent = _without_agent(
        shared_start,
        public_agent_id=appearing_public_id,
    )
    start_without_agent = replace(
        start_without_agent,
        source_frame_index=0,
        source_recipient_frame_id=(
            f"{start_without_agent.source_episode_id}:shared-obs-visual-union:"
            f"{start_without_agent.recipient_public_agent_id}:frame:0"
        ),
        source_simulator_step_count=shared_start.source_simulator_step_count,
    )
    appearance_summary = build_shared_obs_incoming_summary_v1(
        start_without_agent,
        _as_successor(shared_start),
    )
    assert appearance_summary is not None
    appearance_payload = cast(
        dict[str, object],
        json.loads(adapter.dump_json(appearance_summary)),
    )
    appearance_row = next(
        row
        for row in cast(list[dict[str, object]], appearance_payload["deltas"])
        if row["delta_kind"] == "appearance"
    )
    appearance_sources = cast(
        list[dict[str, object]],
        appearance_row["successor_observation_sources"],
    )
    assert any(
        source["source_kind"] == "shared_sensor_source"
        and source["source_public_agent_id"] == appearing_public_id
        for source in appearance_sources
    )
    successor_observation = cast(
        dict[str, object],
        appearance_row["successor_observation"],
    )
    successor_observation["relation"] = "opponent"
    with pytest.raises(ValidationError, match="source kind conflicts"):
        adapter.validate_json(json.dumps(appearance_payload))


def test_shared_rejects_nonadjacent_endpoint_epochs(
    shared_start: SharedObsAuthorizedScenePartsV1,
) -> None:
    successor = _as_successor(shared_start)
    bad_tick = replace(
        successor,
        source_simulator_step_count=successor.source_simulator_step_count + 1,
    )
    with pytest.raises(ValueError, match="ticks are not adjacent"):
        build_shared_obs_incoming_summary_v1(shared_start, bad_tick)
    with pytest.raises(ValueError, match="requires its prior endpoint"):
        build_shared_obs_incoming_summary_v1(None, successor)
    with pytest.raises(ValueError, match="frame zero"):
        build_shared_obs_incoming_summary_v1(shared_start, shared_start)


def test_public_signatures_exclude_oracle_and_standalone_transition_inputs() -> None:
    live = inspect.signature(build_live_no_shared_obs_incoming_summary_v1).parameters
    assert tuple(live) == (
        "source",
        "public_catalog",
        "authority_session_id",
    )
    assert "transition" not in live
    no_shared = inspect.signature(
        build_replay_no_shared_obs_incoming_summary_v1
    ).parameters
    assert tuple(no_shared) == (
        "source",
        "successor_frame_index",
        "public_catalog",
        "authority_session_id",
    )
    assert "transition" not in no_shared
    shared = inspect.signature(build_shared_obs_incoming_summary_v1).parameters
    assert tuple(shared) == ("start", "successor")
    source = _MODULE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    public_functions = {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    assert public_functions == {
        "build_live_no_shared_obs_incoming_summary_v1",
        "build_replay_no_shared_obs_incoming_summary_v1",
        "build_shared_obs_incoming_summary_v1",
    }
    assert all(
        forbidden not in source
        for forbidden in (
            "VisualEventBatchV2",
            "canonical_reward",
            "status_source_evidence",
            "processing",
        )
    )
    assert "_compose_no_shared_obs_incoming_summary_v1" not in cast(
        list[str],
        ast.literal_eval(
            next(
                node.value
                for node in module.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                )
            )
        ),
    )


def test_authorized_incoming_import_is_core_jax_numpy_isolated() -> None:
    script = """
import sys
import marl_battlegrounds.rendering.authorized_incoming  # noqa: F401
blocked = sorted(
    name for name in sys.modules
    if name == "jax" or name.startswith("jax.")
    or name == "jaxlib" or name.startswith("jaxlib.")
    or name == "numpy" or name.startswith("numpy.")
    or name == "marl_battlegrounds.core"
    or name.startswith("marl_battlegrounds.core.")
)
if blocked:
    raise SystemExit("unexpected imports: " + ",".join(blocked))
"""
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cuda"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_summary_build_does_not_mutate_source_roots(
    no_shared_case: tuple[
        CapturedEvaluationTrajectory,
        ActorPovProjectionIndexV1,
    ],
    shared_start: SharedObsAuthorizedScenePartsV1,
) -> None:
    trajectory, index = no_shared_case
    content_before = index.content.model_dump_json()
    shared_adapter = TypeAdapter(SharedObsAuthorizedScenePartsV1)
    shared_before = shared_adapter.dump_json(shared_start)
    build_replay_no_shared_obs_incoming_summary_v1(
        index,
        successor_frame_index=1,
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id="nonmutation-authority",
    )
    build_shared_obs_incoming_summary_v1(
        shared_start,
        _as_successor(shared_start),
    )
    assert index.content.model_dump_json() == content_before
    assert shared_adapter.dump_json(shared_start) == shared_before
