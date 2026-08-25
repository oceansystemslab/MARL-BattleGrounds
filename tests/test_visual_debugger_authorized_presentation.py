"""Focused CP2.0 Oracle presentation-model and pure-builder proofs."""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter, ValidationError
from scripts.dev.visual_debugger.presentation import (
    build_replay_oracle_authorized_presentation_v1,
)
from scripts.dev.visual_debugger.presentation_protocol import (
    PRESENTATION_AUDIENCE_UNAVAILABLE_MESSAGE,
    PresentationApiErrorV1,
    ReplayOracleAuthorizedPresentationFrameV1,
)
from scripts.dev.visual_debugger.replay_protocol import (
    ReplayArtifactSummaryV1,
    ReplayCompletionBadgeV1,
    ReplayCursorV1,
    ReplayProcessingBadgeV1,
    ResearcherReplayViewerFrameV1,
)
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    neutral_action,
)

import marl_battlegrounds.rendering as rendering_package
from marl_battlegrounds.core.types import Action
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    EvaluationTransitionV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.replay import (
    ReplayArtifactReferenceV1,
    build_replay_artifact_reference_v1,
)
from marl_battlegrounds.evaluation.replay_io import load_replay_bundle_v1
from marl_battlegrounds.rendering import (
    authorized_presentation as authorized_presentation_module,
)
from marl_battlegrounds.rendering import evaluation_adapter as evaluation_adapter_module
from marl_battlegrounds.rendering.authorized_inspection import (
    AuthorizedNoTargetActionV1,
    AuthorizedVisibleTargetActionV1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    pov_presentation_key_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AuthorizedBattlefieldSceneV1,
    AuthorizedClassDocumentationProfileAvailableV1,
    AuthorizedClassDocumentationProfileUnavailableV1,
    AuthorizedClassMechanicsV1,
    AuthorizedClassMechanicsV2,
    AuthorizedSpawnShieldMechanicsAvailableV1,
    AuthorizedSpawnShieldMechanicsAvailableV2,
    ReplayIncomingActionRejectedEventV1,
    ReplayIncomingAuthorizedAgentIdentityV1,
    ReplayOraclePresentationPartsV1,
    build_oracle_authorized_scene_v1,
    build_replay_oracle_presentation_parts_v1,
    oracle_presentation_key_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    EvaluationScenePresentationStateV1,
    build_researcher_analyzer_projection_v2,
    build_status_source_evidence_index_v2,
    validate_oracle_scene_static_authority_v1,
)
from marl_battlegrounds.rendering.scene import (
    ActionRejectedEventV2,
    BattlefieldSceneV2,
    VisualEventBatchV2,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


@dataclass(frozen=True, slots=True)
class _OracleCases:
    trajectory: CapturedEvaluationTrajectory
    raw_frames: tuple[ResearcherReplayViewerFrameV1, ...]


def _reference(context: EvaluationEpisodeContextV1) -> ReplayArtifactReferenceV1:
    episode_id = context.identity.episode_id
    return ReplayArtifactReferenceV1(
        artifact_id=f"{episode_id}:replay",
        episode_id=episode_id,
        context_digest_sha256=canonical_digest_sha256(context),
        trajectory_content_digest_sha256=_DIGEST_B,
        canonical_digest_sha256=_DIGEST_C,
        canonical_byte_length=2048,
    )


def _build_raw_frames(
    trajectory: CapturedEvaluationTrajectory,
    *,
    viewer_session_id: str,
    replay_reference: ReplayArtifactReferenceV1 | None = None,
) -> tuple[ResearcherReplayViewerFrameV1, ...]:
    context = trajectory.context
    transition_count = len(trajectory.transitions)
    reference = _reference(context) if replay_reference is None else replay_reference
    summary = ReplayArtifactSummaryV1(
        replay_reference=reference,
        expected_transition_count=transition_count,
        recorded_transition_count=transition_count,
        recorded_frame_count=transition_count + 1,
        metric_report_availability="available",
    )
    completion = ReplayCompletionBadgeV1(
        episode_id=reference.episode_id,
        completion_state="complete",
        expected_transition_count=transition_count,
        validated_transition_count=transition_count,
        last_valid_frame_index=transition_count,
        last_valid_frame_id=f"{reference.episode_id}:frame:{transition_count}",
        terminated=False,
        truncated=False,
        completion_bases=("declared_horizon",),
    )
    processing = ReplayProcessingBadgeV1(
        status="succeeded",
        processed_transition_count=transition_count,
    )
    status_index = build_status_source_evidence_index_v2(
        context,
        trajectory.frames,
        trajectory.transitions,
    )
    rows: list[ResearcherReplayViewerFrameV1] = []
    for index, frame in enumerate(trajectory.frames):
        incoming = (
            None
            if index == 0
            else EvaluationTransitionViewV1(
                context=context,
                start_frame=trajectory.frames[index - 1],
                transition=trajectory.transitions[index - 1],
                successor_frame=frame,
            )
        )
        projection = build_researcher_analyzer_projection_v2(
            context,
            frame,
            transition_view=incoming,
            presentation=EvaluationScenePresentationStateV1(
                controlled_global_slot=0,
                # Deliberately differs from the explicit service-owned slot in
                # most tests; the authorized builder must never consult this.
                selected_global_slot=0,
                show_ranges=True,
            ),
            status_source_evidence_state=status_index.state_for_frame(index),
        )
        rows.append(
            ResearcherReplayViewerFrameV1(
                viewer_session_id=viewer_session_id,
                revision=10 + index,
                artifact_summary=summary,
                timeline_id=f"{reference.artifact_id}:timeline:researcher",
                cursor=ReplayCursorV1(
                    frame_index=index,
                    final_frame_index=transition_count,
                    cursor_generation=index,
                    choreography_generation=index,
                ),
                preset="analysis",
                frame_id=frame.frame_id,
                simulator_step_count=frame.simulator_step_count,
                incoming_transition_index=None if incoming is None else index - 1,
                incoming_transition_id=(
                    None if incoming is None else incoming.transition.transition_id
                ),
                completion=completion,
                processing=processing,
                show_ranges=True,
                recorded_ordinary_movement_distance_scale=(
                    context.resolved_env_config.ordinary_movement_distance_scale
                ),
                projection=projection,
            )
        )
    return tuple(rows)


@pytest.fixture(scope="module")
def oracle_cases() -> _OracleCases:
    neutral = neutral_action()
    invalid_submitted = Action(
        move=neutral.move.at[1].set(99),
        select_target=neutral.select_target,
        use_ultimate=neutral.use_ultimate,
    )
    moving = Action(
        move=neutral.move.at[1].set(1),
        select_target=neutral.select_target,
        use_ultimate=neutral.use_ultimate,
    )
    trajectory = captured_evaluation_trajectory(
        transition_count=5,
        expected_horizon=5,
        actions=(neutral, neutral, neutral, invalid_submitted, moving),
    )
    return _OracleCases(
        trajectory=trajectory,
        raw_frames=_build_raw_frames(
            trajectory,
            viewer_session_id="viewer-session",
        ),
    )


def _presentation(
    cases: _OracleCases,
    index: int,
    *,
    selected_internal_slot: int | None = 1,
    viewer_session_id: str | None = None,
) -> ReplayOracleAuthorizedPresentationFrameV1:
    raw = (
        cases.raw_frames[index]
        if viewer_session_id is None
        else _build_raw_frames(
            cases.trajectory,
            viewer_session_id=viewer_session_id,
        )[index]
    )
    outgoing = (
        None
        if index == len(cases.trajectory.transitions)
        else cases.trajectory.transitions[index]
    )
    return build_replay_oracle_authorized_presentation_v1(
        cases.trajectory.context,
        cases.trajectory.frames[index],
        raw,
        source_authority_epoch=raw.revision,
        selected_internal_slot=selected_internal_slot,
        incoming_transition=(
            None if index == 0 else cases.trajectory.transitions[index - 1]
        ),
        outgoing_transition=outgoing,
    )


def _all_mapping_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            mapping = cast(Mapping[object, object], item)
            for key, nested in mapping.items():
                keys.append(str(key))
                visit(nested)
        elif isinstance(item, list):
            sequence = cast(list[object], item)
            for nested in sequence:
                visit(nested)

    visit(value)
    return tuple(keys)


def _valid_authorized_status_payload(
    frame: ReplayOracleAuthorizedPresentationFrameV1,
) -> dict[str, object]:
    mechanics = next(
        row for row in frame.current_scene.class_mechanics if row.status_mechanics
    )
    status = mechanics.status_mechanics[0]
    return {
        "status_channel": status.status_channel,
        "status_id": status.status_id,
        "family": status.family,
        "configured_duration_steps": status.duration_steps,
        "remaining_duration": 1,
        "source_class_id": mechanics.class_id,
        "source_class_name": mechanics.class_name,
        "source_action_component": status.source_action_component,
        "magnitude_kind": status.magnitude_kind,
        "magnitude": status.magnitude,
        "breaks_on_positive_damage": status.breaks_on_positive_damage,
        "direct_sources": [],
    }


def _install_authorized_target(payload: dict[str, object]) -> dict[str, object]:
    current_endpoint = cast(dict[str, object], payload["current_endpoint"])
    current_scene = cast(dict[str, object], current_endpoint["scene"])
    agents = cast(list[dict[str, object]], current_scene["agents"])
    del agents
    outgoing = cast(dict[str, object], payload["replay_inspection"])
    accepted_action = cast(dict[str, object], outgoing["accepted_action"])
    accepted_action["target_action"] = 1
    accepted_action["use_ultimate_action"] = 0
    outgoing["combat_lane"] = "basic"
    decision_mask = cast(dict[str, object], outgoing["decision_mask"])
    target_actions = cast(list[dict[str, object]], decision_mask["target_actions"])
    outgoing["accepted_target"] = target_actions[1]
    return outgoing["accepted_target"]


def test_replay_epochs_are_structurally_separate_at_zero_middle_and_final(
    oracle_cases: _OracleCases,
) -> None:
    initial = _presentation(oracle_cases, 0)
    middle = _presentation(oracle_cases, 3)
    final = _presentation(oracle_cases, 5)

    assert initial.incoming_summary is None
    assert initial.outgoing_inspection is not None
    assert initial.outgoing_inspection.outgoing_transition_index == 0

    assert middle.incoming_summary is not None
    assert middle.incoming_summary.incoming_transition_index == 2
    assert middle.incoming_summary.incoming_transition_id.endswith(":transition:2")
    assert middle.outgoing_inspection is not None
    assert middle.outgoing_inspection.outgoing_transition_index == 3
    assert middle.outgoing_inspection.transition_reference.transition_id.endswith(
        ":transition:3"
    )

    assert final.incoming_summary is not None
    assert final.incoming_summary.incoming_transition_index == 4
    assert final.outgoing_inspection is None


def test_no_selected_actor_has_no_outgoing_inspection(
    oracle_cases: _OracleCases,
) -> None:
    frame = _presentation(oracle_cases, 3, selected_internal_slot=None)
    assert frame.incoming_summary is not None
    assert frame.outgoing_inspection is None
    assert frame.upcoming_transition is not None
    assert frame.upcoming_transition.outgoing_transition_index == 3


def test_service_owned_selection_ignores_legacy_scene_selection(
    oracle_cases: _OracleCases,
) -> None:
    raw = oracle_cases.raw_frames[3]
    assert raw.projection.scene.selection is not None
    assert raw.projection.scene.selection.selected_global_slot == 0

    frame = _presentation(oracle_cases, 3, selected_internal_slot=1)
    outgoing = frame.outgoing_inspection
    assert outgoing is not None
    assert outgoing.actor_public_agent_id == (
        oracle_cases.trajectory.context.roster[1].public_agent_id
    )


def test_submitted_and_accepted_action_tuples_remain_distinct(
    oracle_cases: _OracleCases,
) -> None:
    outgoing = _presentation(oracle_cases, 3).outgoing_inspection
    assert outgoing is not None
    assert outgoing.submitted_action.move_action == 99
    assert outgoing.accepted_action.move_action == 0
    assert type(outgoing.accepted_target) is AuthorizedNoTargetActionV1


def test_outgoing_anchor_is_owned_by_current_scene_without_successor_input(
    oracle_cases: _OracleCases,
) -> None:
    frame = _presentation(oracle_cases, 4)
    outgoing = frame.outgoing_inspection
    assert outgoing is not None
    current_agent = next(
        row
        for row in oracle_cases.raw_frames[4].projection.scene.agents
        if row.global_slot == 1
    )
    successor_position = oracle_cases.trajectory.frames[5].snapshot.agent_positions[1]

    assert outgoing.actor_anchor == current_agent.position
    assert outgoing.actor_anchor != successor_position
    parameters = inspect.signature(
        build_replay_oracle_authorized_presentation_v1
    ).parameters
    assert "successor_frame" not in parameters
    assert "transition_view" not in parameters


def test_checked_mirrored_sample_uses_context_target_axis_and_current_anchors() -> None:
    sample_path = (
        _REPOSITORY_ROOT
        / "examples/replays/v1/mirrored-five-class-ultimates.marlbg-replay.json"
    )
    replay = load_replay_bundle_v1(
        sample_path,
        require_metric_report=True,
    ).replay
    trajectory = CapturedEvaluationTrajectory(
        context=replay.header.context,
        frames=replay.frames,
        transitions=replay.transitions,
    )
    raw = _build_raw_frames(
        trajectory,
        viewer_session_id="checked-sample-viewer",
        replay_reference=build_replay_artifact_reference_v1(replay),
    )[1]
    frame = build_replay_oracle_authorized_presentation_v1(
        trajectory.context,
        trajectory.frames[1],
        raw,
        source_authority_epoch=raw.revision,
        selected_internal_slot=1,
        incoming_transition=trajectory.transitions[0],
        outgoing_transition=trajectory.transitions[1],
    )
    incoming = frame.incoming_summary
    outgoing = frame.outgoing_inspection
    assert incoming is not None
    assert incoming.incoming_transition_index == 0
    assert outgoing is not None
    assert (
        outgoing.accepted_action.move_action,
        outgoing.accepted_action.target_action,
        outgoing.accepted_action.use_ultimate_action,
    ) == (0, 7, 1)
    assert (
        outgoing.submitted_action.move_action,
        outgoing.submitted_action.target_action,
        outgoing.submitted_action.use_ultimate_action,
    ) == (0, 7, 1)
    assert outgoing.actor_public_agent_id == "1"
    assert outgoing.actor_anchor == (6.0, 4.0)
    assert type(outgoing.accepted_target) is AuthorizedVisibleTargetActionV1
    assert outgoing.accepted_target.target_public_agent_id == "6"
    assert outgoing.accepted_target.target_anchor == (10.0, 4.0)
    assert trajectory.frames[2].snapshot.agent_positions[1] == (9.0, 4.0)
    assert trajectory.frames[2].snapshot.agent_positions[6] == (7.0, 4.0)


def test_checked_recovery_sample_projects_status_durations() -> None:
    sample_path = (
        _REPOSITORY_ROOT
        / "examples/replays/v1/recovery-status-lifecycle.marlbg-replay.json"
    )
    replay = load_replay_bundle_v1(
        sample_path,
        require_metric_report=True,
    ).replay
    trajectory = CapturedEvaluationTrajectory(
        context=replay.header.context,
        frames=replay.frames,
        transitions=replay.transitions,
    )
    raw = _build_raw_frames(
        trajectory,
        viewer_session_id="checked-status-viewer",
        replay_reference=build_replay_artifact_reference_v1(replay),
    )[1]
    frame = build_replay_oracle_authorized_presentation_v1(
        trajectory.context,
        trajectory.frames[1],
        raw,
        source_authority_epoch=raw.revision,
        selected_internal_slot=None,
        incoming_transition=trajectory.transitions[0],
        outgoing_transition=trajectory.transitions[1],
    )
    mechanics_by_channel = {
        status.status_channel: status
        for mechanics in frame.current_scene.class_mechanics
        for status in mechanics.status_mechanics
    }
    durable_statuses = tuple(
        status for agent in frame.current_scene.agents for status in agent.statuses
    )
    assert durable_statuses
    for status in durable_statuses:
        mechanic = mechanics_by_channel[status.status_channel]
        assert status.configured_duration_steps == mechanic.duration_steps
        assert 1 <= status.remaining_duration <= status.configured_duration_steps


def test_outgoing_transition_tick_must_join_the_current_scene(
    oracle_cases: _OracleCases,
) -> None:
    raw = oracle_cases.raw_frames[3]
    transition = oracle_cases.trajectory.transitions[3]
    stale_facts = transition.facts.model_copy(
        update={"transition_start_step_count": raw.simulator_step_count + 10}
    )
    stale_transition = transition.model_copy(update={"facts": stale_facts})
    with pytest.raises(ValueError, match=r"start at (the )?displayed"):
        build_replay_oracle_authorized_presentation_v1(
            oracle_cases.trajectory.context,
            oracle_cases.trajectory.frames[3],
            raw,
            source_authority_epoch=raw.revision,
            selected_internal_slot=1,
            incoming_transition=oracle_cases.trajectory.transitions[2],
            outgoing_transition=stale_transition,
        )


def test_neutral_scene_uses_stable_opaque_keys_and_omits_legacy_roots(
    oracle_cases: _OracleCases,
) -> None:
    earlier = _presentation(oracle_cases, 3)
    later = _presentation(oracle_cases, 4)
    other_authority_session = _presentation(
        oracle_cases,
        3,
        viewer_session_id="viewer-session-other",
    )
    earlier_keys = tuple(row.presentation_key for row in earlier.current_scene.agents)
    later_keys = tuple(row.presentation_key for row in later.current_scene.agents)
    other_keys = tuple(
        row.presentation_key for row in other_authority_session.current_scene.agents
    )

    assert earlier_keys == later_keys
    assert earlier_keys != other_keys
    assert all(key.startswith("oracle_") for key in earlier_keys)
    payload = earlier.model_dump(mode="json")
    serialized_keys = tuple(key.lower() for key in _all_mapping_keys(payload))
    assert not any("global_slot" in key for key in serialized_keys)
    assert not any("researcher" in key for key in serialized_keys)
    assert not any("privileged" in key for key in serialized_keys)
    assert set(payload["current_endpoint"]["scene"]) == {
        "schema_version",
        "map",
        "agents",
        "aura_fields",
        "class_mechanics",
        "spawn_shield_mechanics",
        "spawn_pads",
        "respawn_waves",
    }
    assert "selection" not in json.dumps(payload["current_endpoint"]["scene"])
    assert "observer_visibility" not in json.dumps(payload["current_endpoint"]["scene"])
    assert "ranges" not in json.dumps(payload["current_endpoint"]["scene"])
    shield = earlier.current_scene.spawn_shield_mechanics
    assert type(shield) is AuthorizedSpawnShieldMechanicsAvailableV2
    assert shield.availability_kind == "available_v2"
    assert shield.configured_duration_steps == (
        oracle_cases.trajectory.context.resolved_env_config.spawn_shield_duration_steps
    )
    assert shield.movement_speed == (
        oracle_cases.trajectory.context.resolved_env_config.spawn_shield_movement_speed
    )
    assert (
        shield.protection_effect,
        shield.visibility_effect,
        shield.targetability_effect,
        shield.action_scope,
        shield.aura_effect,
        shield.agent_collision_effect,
        shield.ordinary_application_mechanism,
    ) == (
        "invulnerable",
        "concealed_from_opponents",
        "untargetable",
        "movement_only",
        "excluded_as_emitter_and_beneficiary",
        "phased_until_expiring_endpoint_rejoin",
        "end_of_transition_respawn_lifecycle",
    )
    assert all(
        type(row) is AuthorizedClassMechanicsV2
        and row.mechanics_version == 2
        and type(row.documentation_profile)
        is AuthorizedClassDocumentationProfileAvailableV1
        for row in earlier.current_scene.class_mechanics
    )


def test_scene_accepts_all_legacy_v1_rows_but_rejects_mixed_or_discordant_v2(
    oracle_cases: _OracleCases,
) -> None:
    scene = _presentation(oracle_cases, 3).current_scene
    shield = scene.spawn_shield_mechanics
    assert type(shield) is AuthorizedSpawnShieldMechanicsAvailableV2
    legacy_rows = tuple(
        AuthorizedClassMechanicsV1(
            **{
                field.name: getattr(row, field.name)
                for field in fields(AuthorizedClassMechanicsV1)
            }
        )
        for row in scene.class_mechanics
    )
    legacy = replace(
        scene,
        class_mechanics=legacy_rows,
        spawn_shield_mechanics=AuthorizedSpawnShieldMechanicsAvailableV1(
            availability_kind="available",
            configured_duration_steps=shield.configured_duration_steps,
            movement_speed=shield.movement_speed,
        ),
    )
    adapter = TypeAdapter(AuthorizedBattlefieldSceneV1)
    parsed = adapter.validate_json(adapter.dump_json(legacy))
    assert all(
        type(row) is AuthorizedClassMechanicsV1 for row in parsed.class_mechanics
    )
    assert type(parsed.spawn_shield_mechanics) is (
        AuthorizedSpawnShieldMechanicsAvailableV1
    )

    with pytest.raises(ValueError, match="cannot mix V1 and V2"):
        replace(
            scene,
            class_mechanics=(legacy_rows[0], *scene.class_mechanics[1:]),
        )

    first = cast(AuthorizedClassMechanicsV2, scene.class_mechanics[0])
    unavailable = AuthorizedClassDocumentationProfileUnavailableV1(
        availability_kind="unavailable"
    )
    with pytest.raises(ValueError, match="share one documentation profile"):
        replace(
            scene,
            class_mechanics=(
                replace(first, documentation_profile=unavailable),
                *scene.class_mechanics[1:],
            ),
        )


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (
            ("spawn_shield_mechanics", "protection_effect"),
            "damage_reduction",
            "literal_error",
        ),
        (
            ("spawn_shield_mechanics", "unexpected"),
            True,
            "unexpected_keyword_argument",
        ),
        (
            ("spawn_shield_mechanics", "configured_duration_steps"),
            "3",
            "int_type",
        ),
        (
            ("class_mechanics", 0, "mechanics_version"),
            1,
            "literal_error",
        ),
        (
            ("class_mechanics", 0, "unexpected"),
            True,
            "unexpected_keyword_argument",
        ),
        (
            ("class_mechanics", 0, "documentation_profile", "profile_id"),
            "unknown.profile",
            "literal_error",
        ),
        (
            ("class_mechanics", 0, "documentation_profile", "unexpected"),
            True,
            "unexpected_keyword_argument",
        ),
    ),
)
def test_v2_nested_contracts_fail_closed_on_wrong_literals_and_extras(
    oracle_cases: _OracleCases,
    path: tuple[str | int, ...],
    replacement: object,
    message: str,
) -> None:
    payload = _presentation(oracle_cases, 3).current_scene
    adapter = TypeAdapter(AuthorizedBattlefieldSceneV1)
    mutable = adapter.dump_python(payload, mode="json")
    target: object = mutable
    for segment in path[:-1]:
        if isinstance(segment, str):
            target = cast(dict[str, object], target)[segment]
        else:
            target = cast(list[object], target)[segment]
    cast(dict[str, object], target)[cast(str, path[-1])] = replacement
    with pytest.raises(ValidationError, match=message):
        adapter.validate_json(json.dumps(mutable))


@pytest.mark.parametrize(
    ("container_path", "required_field"),
    (
        (("spawn_shield_mechanics",), "action_scope"),
        (("class_mechanics", 0), "mechanics_version"),
        (("class_mechanics", 0, "documentation_profile"), "profile_id"),
    ),
)
def test_v2_nested_contracts_have_no_implicit_required_defaults(
    oracle_cases: _OracleCases,
    container_path: tuple[str | int, ...],
    required_field: str,
) -> None:
    adapter = TypeAdapter(AuthorizedBattlefieldSceneV1)
    mutable = adapter.dump_python(
        _presentation(oracle_cases, 3).current_scene,
        mode="json",
    )
    target: object = mutable
    for segment in container_path:
        if isinstance(segment, str):
            target = cast(dict[str, object], target)[segment]
        else:
            target = cast(list[object], target)[segment]
    cast(dict[str, object], target).pop(required_field)
    with pytest.raises(ValidationError, match="missing"):
        adapter.validate_json(json.dumps(mutable))


@pytest.mark.parametrize(
    ("authority_session_id", "public_agent_id", "expected"),
    (
        (
            "viewer-session",
            "0",
            "oracle_5e248798a83a3e37d94222d46f8a7bdf4eb969fad4893a6d74e9b84746f536c8",
        ),
        (
            "session",
            "agent",
            "oracle_02542050fb1a7befa3c1be8f3ed02a86c43a8c7fb4ce9f44def96be5bd4e6e30",
        ),
        (
            "",
            "",
            "oracle_9c37ea7bd324f96c6bcff37a73db5aca13185aa9d8ef73f2c42ed01d49e8701b",
        ),
    ),
)
def test_oracle_presentation_key_preserves_frozen_vectors(
    authority_session_id: str,
    public_agent_id: str,
    expected: str,
) -> None:
    assert (
        oracle_presentation_key_v1(
            authority_session_id=authority_session_id,
            public_agent_id=public_agent_id,
        )
        == expected
    )


def test_oracle_key_is_session_and_public_sensitive_and_pov_distinct() -> None:
    oracle = oracle_presentation_key_v1(
        authority_session_id="stable-session",
        public_agent_id="agent-0",
    )
    assert oracle != oracle_presentation_key_v1(
        authority_session_id="other-session",
        public_agent_id="agent-0",
    )
    assert oracle != oracle_presentation_key_v1(
        authority_session_id="stable-session",
        public_agent_id="agent-1",
    )
    pov = pov_presentation_key_v1(
        authority_session_id="stable-session",
        recipient_public_agent_id="agent-0",
        public_agent_id="agent-0",
    )
    assert oracle.startswith("oracle_")
    assert pov.startswith("pov_")
    assert oracle != pov


def test_incoming_inventory_exactly_matches_raw_projection(
    oracle_cases: _OracleCases,
) -> None:
    raw_batch = oracle_cases.raw_frames[3].projection.incoming_events
    incoming = _presentation(oracle_cases, 3).incoming_summary
    assert raw_batch is not None
    assert incoming is not None
    assert incoming.ordered_event_ids == tuple(
        event.event_id for event in raw_batch.events
    )
    assert incoming.ordered_event_kinds == tuple(
        event.event_type for event in raw_batch.events
    )
    assert incoming.event_count == len(raw_batch.events)
    assert tuple(event.event_id for event in incoming.events) == (
        incoming.ordered_event_ids
    )
    assert tuple(event.event_kind for event in incoming.events) == (
        incoming.ordered_event_kinds
    )
    assert tuple(
        (row.agent_public_agent_id, row.successor.position)
        for row in incoming.agent_phase_trajectories
    ) == tuple(
        (row.public_agent_id, row.position)
        for row in _presentation(oracle_cases, 3).current_scene.agents
    )


def _build_parts_with_incoming_batch(
    oracle_cases: _OracleCases,
    *,
    incoming_batch: VisualEventBatchV2,
    scene: BattlefieldSceneV2 | None = None,
) -> ReplayOraclePresentationPartsV1:
    raw = oracle_cases.raw_frames[1]
    return rendering_package.build_replay_oracle_presentation_parts_v1(
        oracle_cases.trajectory.context,
        raw.projection.scene if scene is None else scene,
        incoming_batch,
        authority_session_id=raw.viewer_session_id,
        final_frame_index=raw.cursor.final_frame_index,
        selected_internal_slot=None,
        outgoing_transition=None,
    )


def test_incoming_batch_inactive_feed_only_identity_must_join_context_roster(
    oracle_cases: _OracleCases,
) -> None:
    raw = oracle_cases.raw_frames[1]
    batch = raw.projection.incoming_events
    assert batch is not None
    forged_public_id = "forged-inactive-agent"
    public_ids = list(batch.public_agent_id_by_global_slot)
    public_ids[3] = forged_public_id
    rejection = ActionRejectedEventV2(
        event_id=f"{batch.transition_id}:event:0000",
        transition_id=batch.transition_id,
        ordinal=0,
        actor_global_slot=3,
        actor_public_agent_id=forged_public_id,
        actor_configured_active=False,
        rejection_component="domain",
        submitted_move_action=99,
        submitted_select_target_action=0,
        submitted_use_ultimate_action=0,
        actor_anchor=None,
    )
    forged_batch = replace(
        batch,
        public_agent_id_by_global_slot=tuple(public_ids),
        events=(rejection,),
    )
    forged_scene = replace(
        raw.projection.scene,
        incoming_event_ids=(rejection.event_id,),
    )

    with pytest.raises(ValueError, match="must equal the context roster"):
        _build_parts_with_incoming_batch(
            oracle_cases,
            incoming_batch=forged_batch,
            scene=forged_scene,
        )


def test_incoming_batch_active_identity_must_join_context_roster(
    oracle_cases: _OracleCases,
) -> None:
    batch = oracle_cases.raw_frames[1].projection.incoming_events
    assert batch is not None
    original = batch.agent_phase_trajectories[0]
    forged_public_id = "forged-active-agent"
    forged_trajectory = replace(
        original,
        public_agent_id=forged_public_id,
        transition_start=replace(
            original.transition_start,
            public_agent_id=forged_public_id,
        ),
        post_charge=replace(
            original.post_charge,
            public_agent_id=forged_public_id,
        ),
        successor=replace(
            original.successor,
            public_agent_id=forged_public_id,
        ),
    )
    public_ids = list(batch.public_agent_id_by_global_slot)
    public_ids[original.global_slot] = forged_public_id
    forged_batch = replace(
        batch,
        public_agent_id_by_global_slot=tuple(public_ids),
        agent_phase_trajectories=(
            forged_trajectory,
            *batch.agent_phase_trajectories[1:],
        ),
    )

    with pytest.raises(ValueError, match="must equal the context roster"):
        _build_parts_with_incoming_batch(
            oracle_cases,
            incoming_batch=forged_batch,
        )


def test_incoming_batch_active_flags_must_join_context_roster(
    oracle_cases: _OracleCases,
) -> None:
    batch = oracle_cases.raw_frames[1].projection.incoming_events
    assert batch is not None
    removed_slot = batch.agent_phase_trajectories[2].global_slot
    active_flags = list(batch.configured_active_by_global_slot)
    active_flags[removed_slot] = False
    forged_batch = replace(
        batch,
        configured_active_by_global_slot=tuple(active_flags),
        agent_phase_trajectories=tuple(
            row
            for row in batch.agent_phase_trajectories
            if row.global_slot != removed_slot
        ),
    )

    with pytest.raises(ValueError, match="must equal the context roster"):
        _build_parts_with_incoming_batch(
            oracle_cases,
            incoming_batch=forged_batch,
        )


def test_active_rejection_uses_authorized_identity_and_start_anchor(
    oracle_cases: _OracleCases,
) -> None:
    incoming = _presentation(oracle_cases, 4).incoming_summary
    assert incoming is not None
    rejection = next(
        event
        for event in incoming.events
        if type(event) is ReplayIncomingActionRejectedEventV1
    )
    assert type(rejection.actor_identity) is (ReplayIncomingAuthorizedAgentIdentityV1)
    assert rejection.actor_configured_active
    assert rejection.actor_anchor is not None
    assert rejection.actor_anchor.phase == "transition_start"
    assert rejection.actor_identity.presentation_key == (
        rejection.actor_anchor.presentation_key
    )


def test_protocol_rejects_incoming_successor_or_feed_only_scene_forgery(
    oracle_cases: _OracleCases,
) -> None:
    successor_payload = _presentation(oracle_cases, 3).model_dump(mode="json")
    successor_payload["latest_events"]["agent_phase_trajectories"][0]["successor"][
        "position"
    ][0] += 0.5
    with pytest.raises(ValidationError, match="current scene"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(successor_payload)
        )

    feed_only_payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    rejection = next(
        event
        for event in feed_only_payload["latest_events"]["events"]
        if event["event_kind"] == "action_rejected"
    )
    public_id = rejection["actor_identity"]["public_agent_id"]
    rejection["actor_configured_active"] = False
    rejection["actor_identity"] = {
        "identity_kind": "inactive_feed_only",
        "public_agent_id": public_id,
    }
    rejection["actor_anchor"] = None
    with pytest.raises(ValidationError, match="feed-only rejection"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(feed_only_payload)
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("source_authority_epoch", 999),
        ("source_timeline_id", "episode-001:replay:timeline:wrong"),
        ("source_frame_id", "episode-001:frame:0"),
    ),
)
def test_source_identity_mismatch_rejects(
    oracle_cases: _OracleCases,
    field_name: str,
    replacement: object,
) -> None:
    payload = _presentation(oracle_cases, 3).model_dump(mode="json")
    payload["source"][field_name] = replacement
    with pytest.raises(ValidationError):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


def test_source_identity_carries_all_artifact_digests_and_context_join(
    oracle_cases: _OracleCases,
) -> None:
    raw = oracle_cases.raw_frames[3]
    reference = raw.artifact_summary.replay_reference
    raw_before = raw.model_dump_json()
    frame = _presentation(oracle_cases, 3)

    assert frame.source.source_context_digest_sha256 == (
        reference.context_digest_sha256
    )
    assert frame.source.source_trajectory_content_digest_sha256 == (
        reference.trajectory_content_digest_sha256
    )
    assert frame.source.source_artifact_digest_sha256 == (
        reference.canonical_digest_sha256
    )
    assert raw.model_dump_json() == raw_before

    wrong_reference = reference.model_copy(update={"context_digest_sha256": "f" * 64})
    wrong_summary = raw.artifact_summary.model_copy(
        update={"replay_reference": wrong_reference}
    )
    wrong_raw = raw.model_copy(update={"artifact_summary": wrong_summary})
    with pytest.raises(ValueError, match=r"context.*digest"):
        build_replay_oracle_authorized_presentation_v1(
            oracle_cases.trajectory.context,
            oracle_cases.trajectory.frames[3],
            wrong_raw,
            source_authority_epoch=wrong_raw.revision,
            selected_internal_slot=1,
            incoming_transition=oracle_cases.trajectory.transitions[2],
            outgoing_transition=oracle_cases.trajectory.transitions[3],
        )


def test_strict_protocol_rejects_extra_and_epoch_swapped_fields(
    oracle_cases: _OracleCases,
) -> None:
    payload = _presentation(oracle_cases, 3).model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )

    payload = _presentation(oracle_cases, 3).model_dump(mode="json")
    payload["latest_events"]["incoming_transition_index"] = 3
    with pytest.raises(
        ValidationError,
        match=r"incoming summary|incoming transition ID",
    ):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


def test_every_nested_wire_object_forbids_additional_properties() -> None:
    schema = ReplayOracleAuthorizedPresentationFrameV1.model_json_schema()
    assert schema["additionalProperties"] is False
    definitions = cast(dict[str, dict[str, object]], schema["$defs"])
    strict_definitions = {
        name: definition
        for name, definition in definitions.items()
        if "properties" in definition
    }
    assert strict_definitions
    for name, definition in strict_definitions.items():
        assert definition["additionalProperties"] is False, name
        assert set(cast(dict[str, object], definition["properties"])) == set(
            cast(list[str], definition["required"])
        ), name
    assert set(cast(dict[str, object], schema["properties"])) == set(
        cast(list[str], schema["required"])
    )


def test_nested_wire_rows_reject_extra_keys_at_every_epoch_level(
    oracle_cases: _OracleCases,
) -> None:
    mutations: list[tuple[str, dict[str, object]]] = []

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    agent = payload["current_endpoint"]["scene"]["agents"][0]
    agent["poison"] = 1
    mutations.append(("agent", payload))

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    status = _valid_authorized_status_payload(_presentation(oracle_cases, 4))
    status["poison"] = 1
    payload["current_endpoint"]["scene"]["agents"][0]["statuses"] = [status]
    mutations.append(("status", payload))

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    payload["replay_inspection"]["submitted_action"]["poison"] = 1
    mutations.append(("action", payload))

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    target = _install_authorized_target(payload)
    target["poison"] = 1
    mutations.append(("target", payload))

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    payload["latest_events"]["poison"] = 1
    mutations.append(("incoming", payload))

    for level, poisoned in mutations:
        with pytest.raises(
            ValidationError,
            match=r"extra_forbidden|unexpected_keyword_argument",
        ) as error:
            ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
                json.dumps(poisoned)
            )
        assert "poison" in str(error.value), level


def test_success_and_error_roots_reject_every_missing_identity_field(
    oracle_cases: _OracleCases,
) -> None:
    deletions = (
        (None, "schema_version"),
        (None, "presentation_kind"),
        (None, "product_kind"),
        (None, "analysis_mode"),
        ("source", "source_kind"),
        ("source", "source_replay_schema_version"),
        ("authority", "authority_kind"),
        ("authority", "projection_basis"),
        ("current_endpoint", "endpoint_kind"),
        ("latest_events", "summary_kind"),
        ("replay_inspection", "inspection_kind"),
    )
    for section, field_name in deletions:
        payload = _presentation(oracle_cases, 4).model_dump(mode="json")
        owner = payload if section is None else payload[section]
        del owner[field_name]
        with pytest.raises(ValidationError, match="Field required"):
            ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
                json.dumps(payload)
            )

    error_payload = {
        "schema_version": 1,
        "error_code": "audience_unavailable",
        "message": PRESENTATION_AUDIENCE_UNAVAILABLE_MESSAGE,
    }
    for field_name in tuple(error_payload):
        incomplete = dict(error_payload)
        del incomplete[field_name]
        with pytest.raises(ValidationError, match="Field required"):
            PresentationApiErrorV1.model_validate_json(json.dumps(incomplete))


def test_nested_required_fields_are_never_repaired_from_defaults(
    oracle_cases: _OracleCases,
) -> None:
    incomplete_payloads: list[dict[str, object]] = []

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    del payload["current_endpoint"]["scene"]["map"]["obstacles"]
    incomplete_payloads.append(payload)

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    del payload["current_endpoint"]["scene"]["agents"][0]["statuses"]
    incomplete_payloads.append(payload)

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    status = _valid_authorized_status_payload(_presentation(oracle_cases, 4))
    del status["direct_sources"]
    payload["current_endpoint"]["scene"]["agents"][0]["statuses"] = [status]
    incomplete_payloads.append(payload)

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    del payload["current_endpoint"]["scene"]["spawn_shield_mechanics"][
        "availability_kind"
    ]
    incomplete_payloads.append(payload)

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    del payload["replay_inspection"]["accepted_target"]["target_kind"]
    incomplete_payloads.append(payload)

    for payload in incomplete_payloads:
        with pytest.raises(
            ValidationError,
            match=r"Field required|union_tag_not_found",
        ):
            ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
                json.dumps(payload)
            )


def test_nested_wire_rows_reject_numeric_strings_and_bool_as_int(
    oracle_cases: _OracleCases,
) -> None:
    mutations: list[tuple[str, dict[str, object]]] = []

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    payload["current_endpoint"]["scene"]["agents"][0]["team_id"] = "1"
    mutations.append(("agent numeric string", payload))

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    payload["current_endpoint"]["scene"]["agents"][0]["team_id"] = True
    mutations.append(("agent bool", payload))

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    status = _valid_authorized_status_payload(_presentation(oracle_cases, 4))
    status["remaining_duration"] = "1"
    payload["current_endpoint"]["scene"]["agents"][0]["statuses"] = [status]
    mutations.append(("status numeric string", payload))

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    payload["replay_inspection"]["accepted_action"]["move_action"] = "1"
    mutations.append(("action numeric string", payload))

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    payload["replay_inspection"]["accepted_action"]["move_action"] = True
    mutations.append(("action bool", payload))

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    target = _install_authorized_target(payload)
    target["target_anchor"] = ["1.0", 2.0]
    mutations.append(("target numeric string", payload))

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    payload["latest_events"]["event_count"] = "1"
    mutations.append(("incoming numeric string", payload))

    for level, poisoned in mutations:
        with pytest.raises(ValidationError) as error:
            ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
                json.dumps(poisoned)
            )
        assert "int_type" in str(error.value) or "float_type" in str(error.value), level


def test_target_variant_requires_the_published_discriminator(
    oracle_cases: _OracleCases,
) -> None:
    schema = ReplayOracleAuthorizedPresentationFrameV1.model_json_schema()
    outgoing = cast(
        dict[str, object],
        schema["$defs"]["ReplayInspectionPresentationV1"],
    )
    accepted_target_reference = cast(dict[str, object], outgoing["properties"])[
        "accepted_target"
    ]
    reference = cast(dict[str, str], accepted_target_reference)["$ref"]
    accepted_target = cast(dict[str, object], schema["$defs"])[
        reference.removeprefix("#/$defs/")
    ]
    assert cast(dict[str, object], accepted_target)["discriminator"] == {
        "mapping": {
            "axis_only_authorized_agent": ("#/$defs/AuthorizedAxisOnlyTargetActionV1"),
            "no_target": "#/$defs/AuthorizedNoTargetActionV1",
            "visible_authorized_agent": ("#/$defs/AuthorizedVisibleTargetActionV1"),
        },
        "propertyName": "target_kind",
    }

    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    payload["replay_inspection"]["accepted_target"]["target_kind"] = "unknown"
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


def test_represented_class_mechanics_cannot_drop_a_status_row(
    oracle_cases: _OracleCases,
) -> None:
    payload = _presentation(oracle_cases, 4).model_dump(mode="json")
    mechanics = next(
        row
        for row in payload["current_endpoint"]["scene"]["class_mechanics"]
        if row["status_mechanics"]
    )
    mechanics["status_mechanics"].pop()
    with pytest.raises(ValidationError, match="represented V1 status axis"):
        ReplayOracleAuthorizedPresentationFrameV1.model_validate_json(
            json.dumps(payload)
        )


def test_presentation_error_is_exact_fixed_and_non_disclosing() -> None:
    error = PresentationApiErrorV1(
        schema_version=1,
        error_code="audience_unavailable",
        message=PRESENTATION_AUDIENCE_UNAVAILABLE_MESSAGE,
    )
    assert error.model_dump(mode="json") == {
        "schema_version": 1,
        "error_code": "audience_unavailable",
        "message": PRESENTATION_AUDIENCE_UNAVAILABLE_MESSAGE,
    }
    schema = PresentationApiErrorV1.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(cast(dict[str, object], schema["properties"])) == set(
        cast(list[str], schema["required"])
    )
    with pytest.raises(ValidationError):
        PresentationApiErrorV1.model_validate_json(
            json.dumps(
                {
                    "schema_version": 1,
                    "error_code": "audience_unavailable",
                    "message": "different",
                }
            )
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PresentationApiErrorV1.model_validate_json(
            json.dumps(
                {
                    **error.model_dump(mode="json"),
                    "latest_frame": {"forbidden": True},
                }
            )
        )


def test_pov_or_unknown_raw_root_never_falls_back_to_oracle(
    oracle_cases: _OracleCases,
) -> None:
    with pytest.raises(TypeError, match="exact ResearcherReplayViewerFrameV1"):
        build_replay_oracle_authorized_presentation_v1(
            oracle_cases.trajectory.context,
            oracle_cases.trajectory.frames[0],
            cast(ResearcherReplayViewerFrameV1, object()),
            source_authority_epoch=0,
            selected_internal_slot=None,
            incoming_transition=None,
            outgoing_transition=None,
        )


def test_neutral_scene_root_is_exact_dataclass(
    oracle_cases: _OracleCases,
) -> None:
    scene = _presentation(oracle_cases, 3).current_scene
    assert type(scene) is AuthorizedBattlefieldSceneV1
    assert not hasattr(scene, "audience")
    assert not hasattr(scene, "incoming_transition_id")
    assert not hasattr(scene, "selection")


@pytest.mark.parametrize("frame_index", (0, 3))
def test_oracle_scene_wrapper_matches_existing_parts_without_mutation(
    oracle_cases: _OracleCases,
    frame_index: int,
) -> None:
    context = oracle_cases.trajectory.context
    raw = oracle_cases.raw_frames[frame_index]
    source_scene = raw.projection.scene
    context_before = TypeAdapter(EvaluationEpisodeContextV1).dump_json(context)
    scene_before = TypeAdapter(BattlefieldSceneV2).dump_json(source_scene)
    final_frame_index = len(oracle_cases.trajectory.transitions)
    outgoing = (
        None
        if frame_index == final_frame_index
        else oracle_cases.trajectory.transitions[frame_index]
    )
    old_before = build_replay_oracle_presentation_parts_v1(
        context,
        source_scene,
        raw.projection.incoming_events,
        authority_session_id=raw.viewer_session_id,
        final_frame_index=final_frame_index,
        selected_internal_slot=1,
        outgoing_transition=outgoing,
    )
    old_before_bytes = TypeAdapter(ReplayOraclePresentationPartsV1).dump_json(
        old_before
    )

    wrapped = build_oracle_authorized_scene_v1(
        context,
        source_scene,
        authority_session_id=raw.viewer_session_id,
    )
    assert wrapped == old_before.current_scene
    alternate = build_oracle_authorized_scene_v1(
        context,
        source_scene,
        authority_session_id="alternate-wrapper-session",
    )
    assert tuple(row.public_agent_id for row in alternate.agents) == tuple(
        row.public_agent_id for row in wrapped.agents
    )
    assert tuple(row.presentation_key for row in alternate.agents) != tuple(
        row.presentation_key for row in wrapped.agents
    )

    old_after = build_replay_oracle_presentation_parts_v1(
        context,
        source_scene,
        raw.projection.incoming_events,
        authority_session_id=raw.viewer_session_id,
        final_frame_index=final_frame_index,
        selected_internal_slot=1,
        outgoing_transition=outgoing,
    )
    assert (
        TypeAdapter(ReplayOraclePresentationPartsV1).dump_json(old_after)
        == old_before_bytes
    )
    assert TypeAdapter(EvaluationEpisodeContextV1).dump_json(context) == context_before
    assert TypeAdapter(BattlefieldSceneV2).dump_json(source_scene) == scene_before

    with pytest.raises(TypeError, match="exact EvaluationEpisodeContextV1"):
        build_oracle_authorized_scene_v1(
            cast(EvaluationEpisodeContextV1, object()),
            source_scene,
            authority_session_id=raw.viewer_session_id,
        )
    with pytest.raises(TypeError, match="exact BattlefieldSceneV2"):
        build_oracle_authorized_scene_v1(
            context,
            cast(BattlefieldSceneV2, object()),
            authority_session_id=raw.viewer_session_id,
        )


def test_rendering_package_exports_authorized_presentation_identity() -> None:
    assert (
        rendering_package.AuthorizedBattlefieldSceneV1 is AuthorizedBattlefieldSceneV1
    )
    assert rendering_package.build_replay_oracle_presentation_parts_v1 is not None
    assert rendering_package.oracle_presentation_key_v1 is oracle_presentation_key_v1
    assert (
        authorized_presentation_module.oracle_presentation_key_v1
        is oracle_presentation_key_v1
    )
    assert (
        rendering_package.build_oracle_authorized_scene_v1
        is build_oracle_authorized_scene_v1
    )
    assert (
        authorized_presentation_module.build_oracle_authorized_scene_v1
        is build_oracle_authorized_scene_v1
    )
    assert (
        rendering_package.validate_oracle_scene_static_authority_v1
        is validate_oracle_scene_static_authority_v1
    )
    assert (
        evaluation_adapter_module.validate_oracle_scene_static_authority_v1
        is validate_oracle_scene_static_authority_v1
    )
    for export_name in (
        "build_oracle_authorized_scene_v1",
        "oracle_presentation_key_v1",
    ):
        assert export_name in authorized_presentation_module.__all__
        assert export_name in rendering_package.__all__
    assert (
        "validate_oracle_scene_static_authority_v1" in evaluation_adapter_module.__all__
    )
    assert "validate_oracle_scene_static_authority_v1" in rendering_package.__all__


def test_rendering_projection_import_is_core_jax_and_numpy_free() -> None:
    script = """
import json
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
from marl_battlegrounds.rendering.authorized_presentation import (
    build_oracle_authorized_scene_v1,
    oracle_presentation_key_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    validate_oracle_scene_static_authority_v1,
)
from scripts.dev.visual_debugger.presentation_protocol import (
    AuthorizedPresentationFrameV1,
    PresentationResourceResultV1,
    build_oracle_authorized_current_endpoint_v1,
)
assert oracle_presentation_key_v1(
    authority_session_id='viewer-session',
    public_agent_id='0',
) == 'oracle_5e248798a83a3e37d94222d46f8a7bdf4eb969fad4893a6d74e9b84746f536c8'
assert build_oracle_authorized_scene_v1 is not None
assert validate_oracle_scene_static_authority_v1 is not None
assert build_oracle_authorized_current_endpoint_v1 is not None
assert AuthorizedPresentationFrameV1 is not None
assert PresentationResourceResultV1 is not None
forbidden = sorted(
    name for name in sys.modules
    if name == 'jax' or name.startswith('jax.')
    or name == 'jaxlib' or name.startswith('jaxlib.')
    or name == 'numpy' or name.startswith('numpy.')
    or name == 'marl_battlegrounds.core'
    or name.startswith('marl_battlegrounds.core.')
    or name == 'scripts.dev.visual_debugger.protocol'
    or name == 'scripts.dev.visual_debugger.replay_protocol'
    or name == 'scripts.dev.visual_debugger.service'
    or name == 'scripts.dev.visual_debugger.replay_service'
)
print(json.dumps(forbidden))
"""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cuda"
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=_REPOSITORY_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_rendering_projection_source_has_no_forbidden_ownership_imports() -> None:
    module_path = (
        _REPOSITORY_ROOT / "src/marl_battlegrounds/rendering/authorized_presentation.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.append(node.module)
    forbidden = (
        "jax",
        "jaxlib",
        "numpy",
        "marl_battlegrounds.core",
        "marl_battlegrounds.evaluation.replay",
        "marl_battlegrounds.evaluation.replay_io",
        "scripts.dev",
    )
    assert not {
        imported
        for imported in imported_roots
        if any(
            imported == prefix or imported.startswith(f"{prefix}.")
            for prefix in forbidden
        )
    }


def test_outgoing_builder_accepts_transition_not_transition_view() -> None:
    annotations = (
        inspect.signature(build_replay_oracle_authorized_presentation_v1)
        .parameters["outgoing_transition"]
        .annotation
    )
    assert "EvaluationTransitionV1" in str(annotations)
    assert "EvaluationTransitionViewV1" not in str(annotations)
    assert EvaluationTransitionV1 is not EvaluationTransitionViewV1
