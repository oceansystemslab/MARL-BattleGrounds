"""Strict, audience-separated replay-viewer wire-contract proofs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError
from scripts.dev.visual_debugger.replay_protocol import (
    ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1,
    ACTOR_POV_PROCESSING_DISCLOSURE_V1,
    ActorPovProcessingDisclosureV1,
    ActorPovReplayCompletionBadgeV1,
    ActorPovReplayTimelineRowV1,
    ActorPovReplayTimelineV1,
    ActorPovReplayViewerFrameV1,
    ReplayAbsoluteSeekCommandV1,
    ReplayApiErrorV1,
    ReplayArtifactSummaryV1,
    ReplayCommandRequestV1,
    ReplayCommandResponseV1,
    ReplayCommandV1,
    ReplayCompletionBadgeV1,
    ReplayCursorV1,
    ReplayExitCommandV1,
    ReplayFirstFrameCommandV1,
    ReplayLastFrameCommandV1,
    ReplayNextFrameCommandV1,
    ReplayPreviousFrameCommandV1,
    ReplayProcessingBadgeV1,
    ReplaySelectAgentCommandV1,
    ReplaySetPovActorCommandV1,
    ReplaySetPresetCommandV1,
    ReplaySetRangesCommandV1,
    ReplaySetVerbosityCommandV1,
    ReplaySetViewCommandV1,
    ReplayTimelineV1,
    ReplayViewerFrameV1,
    ResearcherReplayTimelineRowV1,
    ResearcherReplayTimelineV1,
    ResearcherReplayViewerFrameV1,
    SharedObsAgentPovReplayArtifactSummaryV1,
    SharedObsAgentPovReplayTimelineRowV1,
    SharedObsAgentPovReplayTimelineV1,
    SharedObsAgentPovReplayViewerFrameV1,
    SharedObsSourceMaterialReplayTimelineRowV1,
    SharedObsSourceMaterialReplayTimelineV1,
    SharedObsSourceMaterialReplayViewerFrameV1,
)
from tests.evaluation_fixtures import (
    captured_evaluation_trajectory,
    mage_target_none_ultimate_action,
)

from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.pov import build_actor_pov_current_slice_v1
from marl_battlegrounds.evaluation.replay import ReplayArtifactReferenceV1
from marl_battlegrounds.rendering.evaluation_adapter import (
    EvaluationScenePresentationStateV1,
    SharedObsSourceMaterialProjectionV1,
    build_researcher_analyzer_projection_v2,
    build_shared_obs_source_material_projection_v1,
    build_status_source_evidence_index_v2,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovAnalyzerProjectionV1,
    build_actor_pov_analyzer_projection_v1,
)
from marl_battlegrounds.rendering.scene import ResearcherAnalyzerProjectionV2

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64
_RESEARCHER_FRAME_V1_SHA256 = (
    "3e7896f916ea88882786eaf7b965cc1d76bf5df373b248aa9acd90c03e716c0a"
)
_ACTOR_POV_FRAME_V1_SHA256 = (
    "187841dd953ccb43c53eb80fe2b1a446641b866d53ee202f3f4309f736bdd8d5"
)
_RESEARCHER_TIMELINE_V1_SHA256 = (
    "24da90cf1e6c4f9fe28436ddc6ae1461d3378bcfb981feb315d9c57b09fa923a"
)
_ACTOR_POV_TIMELINE_V1_SHA256 = (
    "ccc8851c92d8d23e568b3568909bfdd0bfd546aa4bd3e20422d74feaa5fdd591"
)


@dataclass(frozen=True, slots=True)
class _ProjectionCases:
    researcher: ResearcherAnalyzerProjectionV2
    researcher_initial: ResearcherAnalyzerProjectionV2
    pov: ActorPovAnalyzerProjectionV1
    source_material: SharedObsSourceMaterialProjectionV1


@pytest.fixture(scope="module")
def projection_cases() -> _ProjectionCases:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        actions=(mage_target_none_ultimate_action(),),
    )
    incoming = EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[0],
        transition=trajectory.transitions[0],
        successor_frame=trajectory.frames[1],
    )
    status_index = build_status_source_evidence_index_v2(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
    )
    presentation = EvaluationScenePresentationStateV1(
        controlled_global_slot=0,
        selected_global_slot=5,
        show_ranges=True,
    )
    researcher = build_researcher_analyzer_projection_v2(
        trajectory.context,
        trajectory.frames[1],
        transition_view=incoming,
        presentation=presentation,
        status_source_evidence_state=status_index.state_for_frame(1),
    )
    researcher_initial = build_researcher_analyzer_projection_v2(
        trajectory.context,
        trajectory.frames[0],
        presentation=presentation,
        status_source_evidence_state=status_index.state_for_frame(0),
    )
    pov = build_actor_pov_analyzer_projection_v1(
        build_actor_pov_current_slice_v1(
            trajectory.context,
            trajectory.frames[1],
            global_slot=0,
            incoming_transition_view=incoming,
        )
    )
    shared_trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        execution_information_mode="shared_obs",
    )
    shared_incoming = EvaluationTransitionViewV1(
        context=shared_trajectory.context,
        start_frame=shared_trajectory.frames[0],
        transition=shared_trajectory.transitions[0],
        successor_frame=shared_trajectory.frames[1],
    )
    source_material = build_shared_obs_source_material_projection_v1(
        shared_trajectory.context,
        shared_trajectory.frames[1],
        selected_global_slot=5,
        transition_view=shared_incoming,
    )
    return _ProjectionCases(
        researcher=researcher,
        researcher_initial=researcher_initial,
        pov=pov,
        source_material=source_material,
    )


def _reference(episode_id: str) -> ReplayArtifactReferenceV1:
    return ReplayArtifactReferenceV1(
        artifact_id=f"{episode_id}:replay",
        episode_id=episode_id,
        context_digest_sha256=_DIGEST_A,
        trajectory_content_digest_sha256=_DIGEST_B,
        canonical_digest_sha256=_DIGEST_C,
        canonical_byte_length=1234,
    )


def _summary(
    episode_id: str,
    *,
    expected: int = 1,
    recorded: int = 1,
    metric_report_availability: Literal[
        "available",
        "missing",
        "not_available_in_actor_pov",
    ] = "available",
) -> ReplayArtifactSummaryV1:
    return ReplayArtifactSummaryV1(
        replay_reference=_reference(episode_id),
        expected_transition_count=expected,
        recorded_transition_count=recorded,
        recorded_frame_count=recorded + 1,
        metric_report_availability=metric_report_availability,
    )


def _completion(
    episode_id: str,
    *,
    expected: int = 1,
    validated: int = 1,
    state: Literal["complete", "partial", "interrupted", "failed"] = "complete",
) -> ReplayCompletionBadgeV1:
    complete = state == "complete"
    return ReplayCompletionBadgeV1(
        episode_id=episode_id,
        completion_state=state,
        expected_transition_count=expected,
        validated_transition_count=validated,
        last_valid_frame_index=validated,
        last_valid_frame_id=f"{episode_id}:frame:{validated}",
        terminated=False,
        truncated=False,
        completion_bases=("declared_horizon",) if complete else (),
        end_or_failure_reason=None if complete else "captured_prefix",
        failure_origin="capture" if state == "failed" else None,
    )


def _pov_completion(
    episode_id: str,
    *,
    expected: int = 1,
    captured: int = 1,
    state: Literal["complete", "partial", "interrupted", "failed"] = "complete",
) -> ActorPovReplayCompletionBadgeV1:
    complete = state == "complete"
    return ActorPovReplayCompletionBadgeV1(
        episode_id=episode_id,
        completion_state=state,
        expected_transition_count=expected,
        captured_transition_count=captured,
        terminated=False,
        truncated=False,
        completion_bases=("declared_horizon",) if complete else (),
        public_end_or_failure_reason=None if complete else "captured_prefix",
    )


def _processing(*, failed: bool = False) -> ReplayProcessingBadgeV1:
    return ReplayProcessingBadgeV1(
        status="failed" if failed else "succeeded",
        processed_transition_count=0 if failed else 1,
        failure_stage="reducer_advance" if failed else None,
        failure_code="test.reducer_failed" if failed else None,
        attempted_transition_index=0 if failed else None,
    )


def _cursor(
    *,
    frame_index: int = 1,
    cursor_generation: int = 1,
    choreography_generation: int = 1,
) -> ReplayCursorV1:
    return ReplayCursorV1(
        frame_index=frame_index,
        final_frame_index=1,
        cursor_generation=cursor_generation,
        choreography_generation=choreography_generation,
    )


def _researcher_frame(
    projection: ResearcherAnalyzerProjectionV2,
) -> ResearcherReplayViewerFrameV1:
    scene = projection.scene
    episode_id = scene.episode_id
    return ResearcherReplayViewerFrameV1(
        viewer_session_id="viewer-session",
        revision=4,
        artifact_summary=_summary(episode_id),
        timeline_id=f"{episode_id}:replay:timeline:researcher",
        cursor=_cursor(),
        preset="analysis",
        verbose=False,
        frame_id=scene.frame_id,
        simulator_step_count=scene.simulator_step_count,
        incoming_transition_index=0,
        incoming_transition_id=f"{episode_id}:transition:0",
        completion=_completion(episode_id),
        processing=_processing(),
        show_ranges=True,
        recorded_ordinary_movement_distance_scale=1.0,
        projection=projection,
    )


def _researcher_initial_frame(
    projection: ResearcherAnalyzerProjectionV2,
) -> ResearcherReplayViewerFrameV1:
    scene = projection.scene
    episode_id = scene.episode_id
    return ResearcherReplayViewerFrameV1(
        viewer_session_id="viewer-session",
        revision=0,
        artifact_summary=_summary(episode_id),
        timeline_id=f"{episode_id}:replay:timeline:researcher",
        cursor=_cursor(
            frame_index=0,
            cursor_generation=0,
            choreography_generation=0,
        ),
        preset="analysis",
        verbose=False,
        frame_id=scene.frame_id,
        simulator_step_count=scene.simulator_step_count,
        incoming_transition_index=None,
        incoming_transition_id=None,
        completion=_completion(episode_id),
        processing=_processing(),
        show_ranges=True,
        recorded_ordinary_movement_distance_scale=1.0,
        projection=projection,
    )


def _actor_pov_frame(
    projection: ActorPovAnalyzerProjectionV1,
) -> ActorPovReplayViewerFrameV1:
    scene = projection.scene
    episode_id = scene.episode_id
    public_agent_id = scene.self_actor.public_agent_id
    return ActorPovReplayViewerFrameV1(
        viewer_session_id="viewer-session",
        revision=5,
        artifact_summary=_summary(
            episode_id,
            metric_report_availability=ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1,
        ),
        timeline_id=(f"{episode_id}:replay:timeline:actor-pov:{public_agent_id}"),
        cursor=_cursor(),
        preset="analysis",
        verbose=False,
        pov_global_slot=scene.self_actor.global_slot,
        public_agent_id=public_agent_id,
        pov_frame_id=scene.pov_frame_id,
        simulator_step_count=scene.simulator_step_count,
        incoming_pov_transition_id=projection.incoming_transition_id,
        completion=_pov_completion(episode_id),
        processing_disclosure=ActorPovProcessingDisclosureV1(),
        projection=projection,
    )


def _source_material_frame(
    projection: SharedObsSourceMaterialProjectionV1,
) -> SharedObsSourceMaterialReplayViewerFrameV1:
    base = projection.base_sensor_frame
    episode_id = base.episode_id
    return SharedObsSourceMaterialReplayViewerFrameV1(
        viewer_session_id="viewer-session",
        revision=6,
        artifact_summary=_summary(episode_id),
        timeline_id=(
            f"{episode_id}:replay:timeline:shared-obs-source-material:"
            f"{base.public_agent_id}"
        ),
        cursor=_cursor(),
        preset="presentation",
        verbose=False,
        selected_global_slot=projection.base_sensor_scene.self_actor.global_slot,
        public_agent_id=base.public_agent_id,
        source_material_frame_id=base.source_material_frame_id,
        source_frame_id=base.source_frame_id,
        simulator_step_count=base.simulator_step_count,
        incoming_transition_id=projection.incoming_transition_id,
        completion=_completion(episode_id),
        processing=_processing(),
        projection=projection,
    )


def _shared_agent_summary(
    episode_id: str = "episode-timeline",
    public_agent_id: str = "agent-5",
    *,
    expected: int = 1,
    captured: int = 1,
) -> SharedObsAgentPovReplayArtifactSummaryV1:
    return SharedObsAgentPovReplayArtifactSummaryV1(
        schema_version=1,
        recipient_replay_id=(
            f"{episode_id}:shared-obs-visual-union:{public_agent_id}:replay"
        ),
        episode_id=episode_id,
        public_agent_id=public_agent_id,
        expected_transition_count=expected,
        captured_transition_count=captured,
        captured_frame_count=captured + 1,
    )


def _shared_agent_frame() -> SharedObsAgentPovReplayViewerFrameV1:
    episode_id = "episode-timeline"
    public_agent_id = "agent-5"
    return SharedObsAgentPovReplayViewerFrameV1(
        schema_version=1,
        frame_kind="shared_obs_agent_pov_replay_viewer",
        viewer_session_id="viewer-session",
        revision=7,
        artifact_summary=_shared_agent_summary(episode_id, public_agent_id),
        timeline_id=(
            f"{episode_id}:shared-obs-visual-union:{public_agent_id}:timeline"
        ),
        cursor=_cursor(),
        preset="analysis",
        verbose=False,
        view_mode="pov",
        public_agent_id=public_agent_id,
        recipient_frame_id=(
            f"{episode_id}:shared-obs-visual-union:{public_agent_id}:frame:1"
        ),
        simulator_step_count=42,
        incoming_recipient_transition_id=(
            f"{episode_id}:shared-obs-visual-union:{public_agent_id}:transition:0"
        ),
        completion=_pov_completion(episode_id),
    )


def _researcher_timeline() -> ResearcherReplayTimelineV1:
    episode_id = "episode-timeline"
    return ResearcherReplayTimelineV1(
        timeline_id=f"{episode_id}:replay:timeline:researcher",
        artifact_summary=_summary(episode_id),
        final_frame_index=1,
        completion=_completion(episode_id),
        rows=(
            ResearcherReplayTimelineRowV1(
                frame_index=0,
                frame_id=f"{episode_id}:frame:0",
                simulator_step_count=41,
                incoming_transition_id=None,
                incoming_event_count=0,
            ),
            ResearcherReplayTimelineRowV1(
                frame_index=1,
                frame_id=f"{episode_id}:frame:1",
                simulator_step_count=42,
                incoming_transition_id=f"{episode_id}:transition:0",
                incoming_event_count=3,
                endpoint_kind="declared_horizon",
            ),
        ),
    )


def _pov_timeline() -> ActorPovReplayTimelineV1:
    episode_id = "episode-timeline"
    public_agent_id = "agent-5"
    return ActorPovReplayTimelineV1(
        timeline_id=(f"{episode_id}:replay:timeline:actor-pov:{public_agent_id}"),
        artifact_summary=_summary(
            episode_id,
            metric_report_availability=ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1,
        ),
        final_frame_index=1,
        pov_global_slot=5,
        public_agent_id=public_agent_id,
        completion=_pov_completion(episode_id),
        rows=(
            ActorPovReplayTimelineRowV1(
                frame_index=0,
                pov_frame_id=f"{episode_id}:actor-pov:{public_agent_id}:frame:0",
                simulator_step_count=41,
                incoming_pov_transition_id=None,
                incoming_cue_count=0,
            ),
            ActorPovReplayTimelineRowV1(
                frame_index=1,
                pov_frame_id=f"{episode_id}:actor-pov:{public_agent_id}:frame:1",
                simulator_step_count=42,
                incoming_pov_transition_id=(
                    f"{episode_id}:actor-pov:{public_agent_id}:transition:0"
                ),
                incoming_cue_count=2,
                endpoint_kind="declared_horizon",
            ),
        ),
    )


def _source_timeline() -> SharedObsSourceMaterialReplayTimelineV1:
    episode_id = "episode-timeline"
    public_agent_id = "agent-5"
    return SharedObsSourceMaterialReplayTimelineV1(
        timeline_id=(
            f"{episode_id}:replay:timeline:shared-obs-source-material:{public_agent_id}"
        ),
        artifact_summary=_summary(episode_id),
        final_frame_index=1,
        selected_global_slot=5,
        public_agent_id=public_agent_id,
        completion=_completion(episode_id),
        rows=(
            SharedObsSourceMaterialReplayTimelineRowV1(
                frame_index=0,
                source_material_frame_id=(
                    f"{episode_id}:shared-obs-source-material:{public_agent_id}:frame:0"
                ),
                simulator_step_count=41,
                incoming_transition_id=None,
            ),
            SharedObsSourceMaterialReplayTimelineRowV1(
                frame_index=1,
                source_material_frame_id=(
                    f"{episode_id}:shared-obs-source-material:{public_agent_id}:frame:1"
                ),
                simulator_step_count=42,
                incoming_transition_id=f"{episode_id}:transition:0",
                endpoint_kind="declared_horizon",
            ),
        ),
    )


def _shared_agent_timeline() -> SharedObsAgentPovReplayTimelineV1:
    episode_id = "episode-timeline"
    public_agent_id = "agent-5"
    return SharedObsAgentPovReplayTimelineV1(
        schema_version=1,
        timeline_kind="shared_obs_agent_pov",
        timeline_id=(
            f"{episode_id}:shared-obs-visual-union:{public_agent_id}:timeline"
        ),
        artifact_summary=_shared_agent_summary(episode_id, public_agent_id),
        final_frame_index=1,
        completion=_pov_completion(episode_id),
        rows=(
            SharedObsAgentPovReplayTimelineRowV1(
                frame_index=0,
                recipient_frame_id=(
                    f"{episode_id}:shared-obs-visual-union:{public_agent_id}:frame:0"
                ),
                simulator_step_count=41,
                incoming_recipient_transition_id=None,
                endpoint_kind="none",
            ),
            SharedObsAgentPovReplayTimelineRowV1(
                frame_index=1,
                recipient_frame_id=(
                    f"{episode_id}:shared-obs-visual-union:{public_agent_id}:frame:1"
                ),
                simulator_step_count=42,
                incoming_recipient_transition_id=(
                    f"{episode_id}:shared-obs-visual-union:"
                    f"{public_agent_id}:transition:0"
                ),
                endpoint_kind="declared_horizon",
            ),
        ),
    )


def _json_roundtrip[ModelT: BaseModel](model: ModelT) -> ModelT:
    return type(model).model_validate_json(model.model_dump_json())


def _exact_model_input(model: BaseModel) -> dict[str, object]:
    return {
        field_name: getattr(model, field_name)
        for field_name in type(model).model_fields
    }


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        record = cast(dict[str, object], value)
        return set(record) | {
            key for child in record.values() for key in _recursive_keys(child)
        }
    if isinstance(value, list):
        sequence = cast(list[object], value)
        return {key for child in sequence for key in _recursive_keys(child)}
    return set()


def _recursive_strings(value: object) -> set[str]:
    if isinstance(value, dict):
        record = cast(dict[str, object], value)
        return {item for child in record.values() for item in _recursive_strings(child)}
    if isinstance(value, list):
        sequence = cast(list[object], value)
        return {item for child in sequence for item in _recursive_strings(child)}
    return {value} if isinstance(value, str) else set()


@pytest.mark.parametrize(
    "command",
    (
        ReplayAbsoluteSeekCommandV1(frame_index=7),
        ReplayFirstFrameCommandV1(),
        ReplayPreviousFrameCommandV1(),
        ReplayNextFrameCommandV1(),
        ReplayLastFrameCommandV1(),
        ReplaySelectAgentCommandV1(selected_global_slot=None),
        ReplaySelectAgentCommandV1(selected_global_slot=7),
        ReplaySetViewCommandV1(view_mode="pov"),
        ReplaySetPovActorCommandV1(global_slot=5),
        ReplaySetPresetCommandV1.model_validate({"preset": "debug"}),
        ReplaySetRangesCommandV1(show_ranges=False),
        ReplaySetVerbosityCommandV1(verbose=True),
        ReplayExitCommandV1(),
    ),
)
def test_every_replay_command_strictly_roundtrips(command: ReplayCommandV1) -> None:
    request = ReplayCommandRequestV1(
        client_id="client-1",
        command_id="command-1",
        base_revision=3,
        command=command,
    )

    assert _json_roundtrip(request) == request
    assert request.command is not command or request.command == command


@pytest.mark.parametrize(
    "legacy_preset",
    ("presentation", "analysis", "technical", "debug"),
)
def test_legacy_replay_preset_canonicalizes_to_analysis(
    legacy_preset: str,
) -> None:
    command = ReplaySetPresetCommandV1.model_validate({"preset": legacy_preset})

    assert command.preset == "analysis"


@pytest.mark.parametrize(
    "mutation",
    (
        {"schema_version": 2},
        {"base_revision": True},
        {"extra": "forbidden"},
        {"command": {"command_type": "unknown"}},
        {"command": {"command_type": "absolute_seek", "frame_index": True}},
        {"command": {"command_type": "set_ranges", "show_ranges": 1}},
    ),
)
def test_replay_request_rejects_future_loose_and_unknown_input(
    mutation: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "client_id": "client-1",
        "command_id": "command-1",
        "base_revision": 0,
        "command": {"command_type": "next_frame"},
    }
    payload.update(mutation)

    with pytest.raises(ValidationError):
        ReplayCommandRequestV1.model_validate_json(json.dumps(payload))


def test_scalar_roots_are_frozen_strict_and_roundtrip() -> None:
    summary = _summary("episode-scalars")
    completion = _completion("episode-scalars")
    pov_completion = _pov_completion("episode-scalars")
    processing = _processing()
    disclosure = ActorPovProcessingDisclosureV1()
    cursor = _cursor()

    for model in (
        summary,
        completion,
        pov_completion,
        processing,
        disclosure,
        cursor,
    ):
        assert _json_roundtrip(model) == model
    with pytest.raises(ValidationError):
        cursor.frame_index = 0  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValidationError):
        ReplayCursorV1(
            frame_index=True,  # type: ignore[arg-type]
            final_frame_index=1,
            cursor_generation=0,
            choreography_generation=0,
        )


def test_private_shared_models_and_product_unions_strictly_roundtrip() -> None:
    summary = _shared_agent_summary()
    timeline = _shared_agent_timeline()
    row = timeline.rows[1]
    frame = _shared_agent_frame()

    for model in (summary, row, timeline, frame):
        assert _json_roundtrip(model) == model

    frame_adapter: TypeAdapter[ReplayViewerFrameV1] = TypeAdapter(ReplayViewerFrameV1)
    timeline_adapter: TypeAdapter[ReplayTimelineV1] = TypeAdapter(ReplayTimelineV1)
    assert frame_adapter.validate_json(frame_adapter.dump_json(frame)) == frame
    assert timeline_adapter.validate_json(timeline_adapter.dump_json(timeline)) == (
        timeline
    )


def test_private_shared_completion_disclosure_is_fixed_and_generic() -> None:
    frame_payload = _shared_agent_frame().model_dump(mode="python")
    frame_payload["artifact_summary"]["expected_transition_count"] = 4
    frame_payload["completion"].update(
        {
            "completion_state": "interrupted",
            "expected_transition_count": 4,
            "completion_bases": (),
            "public_end_or_failure_reason": "captured_prefix",
        }
    )
    partial_frame = SharedObsAgentPovReplayViewerFrameV1.model_validate(frame_payload)

    timeline_payload = _shared_agent_timeline().model_dump(mode="python")
    timeline_payload["artifact_summary"]["expected_transition_count"] = 4
    timeline_payload["completion"].update(
        {
            "completion_state": "interrupted",
            "expected_transition_count": 4,
            "completion_bases": (),
            "public_end_or_failure_reason": "captured_prefix",
        }
    )
    timeline_payload["rows"][-1]["endpoint_kind"] = "captured_prefix"
    partial_timeline = SharedObsAgentPovReplayTimelineV1.model_validate(
        timeline_payload
    )

    for model, model_type in (
        (partial_frame, SharedObsAgentPovReplayViewerFrameV1),
        (partial_timeline, SharedObsAgentPovReplayTimelineV1),
    ):
        leaked = model.model_dump(mode="python")
        leaked["completion"]["public_end_or_failure_reason"] = (
            "/home/user/policy.exception.json"
        )
        with pytest.raises(ValidationError, match="captured_prefix"):
            model_type.model_validate(leaked)

    for model, model_type in (
        (_shared_agent_frame(), SharedObsAgentPovReplayViewerFrameV1),
        (_shared_agent_timeline(), SharedObsAgentPovReplayTimelineV1),
    ):
        leaked = model.model_dump(mode="python")
        leaked["completion"]["public_end_or_failure_reason"] = "captured_prefix"
        with pytest.raises(ValidationError, match="forbids a public reason"):
            model_type.model_validate(leaked)

    response_payload: dict[str, Any] = {
        "schema_version": 1,
        "result": "no_op",
        "frame": partial_frame.model_dump(mode="python"),
        "notice": None,
        "animate_incoming": False,
    }
    response_payload["frame"]["completion"]["public_end_or_failure_reason"] = (
        "HOST TRACEBACK"
    )
    with pytest.raises(ValidationError, match="captured_prefix"):
        ReplayCommandResponseV1.model_validate(response_payload)


@pytest.mark.parametrize(
    ("field_name", "poison"),
    (
        ("schema_version", True),
        ("schema_version", 1.0),
        ("verbose", 0),
        ("verbose", 0.0),
    ),
)
def test_private_shared_frame_rejects_literal_scalar_coercion(
    field_name: str,
    poison: object,
) -> None:
    payload = _shared_agent_frame().model_dump(mode="python")
    payload[field_name] = poison

    with pytest.raises(ValidationError, match="exact Python"):
        SharedObsAgentPovReplayViewerFrameV1.model_validate(payload)


def test_private_shared_roots_revalidate_instances_and_forbid_subclasses() -> None:
    frame = _shared_agent_frame()
    summary_values: dict[str, Any] = {
        **frame.artifact_summary.__dict__,
        "recipient_replay_id": "episode-timeline:replay",
    }
    forged_summary = SharedObsAgentPovReplayArtifactSummaryV1.model_construct(
        **summary_values
    )
    frame_values: dict[str, Any] = {
        **frame.__dict__,
        "artifact_summary": forged_summary,
    }
    forged_frame = SharedObsAgentPovReplayViewerFrameV1.model_construct(**frame_values)
    with pytest.raises(ValidationError, match="recipient-local"):
        TypeAdapter(ReplayViewerFrameV1).validate_python(forged_frame)
    with pytest.raises(ValidationError, match="recipient-local"):
        ReplayCommandResponseV1(result="no_op", frame=forged_frame)

    forged_cursor = ReplayCursorV1.model_construct(
        schema_version=True,
        frame_index=True,
        final_frame_index=1,
        cursor_generation=0,
        choreography_generation=0,
    )
    frame_values = {
        **frame.__dict__,
        "cursor": forged_cursor,
        "recipient_frame_id": (
            "episode-timeline:shared-obs-visual-union:agent-5:frame:True"
        ),
    }
    forged_frame = SharedObsAgentPovReplayViewerFrameV1.model_construct(**frame_values)
    with pytest.raises(ValidationError, match="exact Python ints"):
        TypeAdapter(ReplayViewerFrameV1).validate_python(forged_frame)

    completion_values: dict[str, Any] = {
        **frame.completion.__dict__,
        "schema_version": True,
    }
    forged_completion = ActorPovReplayCompletionBadgeV1.model_construct(
        **completion_values
    )
    frame_values = {**frame.__dict__, "completion": forged_completion}
    forged_frame = SharedObsAgentPovReplayViewerFrameV1.model_construct(**frame_values)
    with pytest.raises(ValidationError, match="exact Python ints"):
        TypeAdapter(ReplayViewerFrameV1).validate_python(forged_frame)

    timeline = _shared_agent_timeline()
    row_values: dict[str, Any] = {
        **timeline.rows[0].__dict__,
        "frame_index": True,
        "recipient_frame_id": (
            "episode-timeline:shared-obs-visual-union:agent-5:frame:True"
        ),
    }
    forged_row = SharedObsAgentPovReplayTimelineRowV1.model_construct(**row_values)
    timeline_values: dict[str, Any] = {
        **timeline.__dict__,
        "rows": (forged_row, timeline.rows[1]),
    }
    forged_timeline = SharedObsAgentPovReplayTimelineV1.model_construct(
        **timeline_values
    )
    with pytest.raises(ValidationError, match="exact Python ints"):
        TypeAdapter(ReplayTimelineV1).validate_python(forged_timeline)

    with pytest.raises(TypeError, match="cannot be subclassed"):
        type(
            "ForbiddenPrivateFrameSubclass",
            (SharedObsAgentPovReplayViewerFrameV1,),
            {},
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"recipient_replay_id": "episode-timeline:replay"}, "recipient-local"),
        ({"captured_frame_count": 1}, "T\\+1/T"),
        (
            {
                "captured_transition_count": 2,
                "captured_frame_count": 3,
            },
            "expected horizon",
        ),
        ({"schema_version": "1"}, None),
        ({"expected_transition_count": True}, None),
        ({"extra": "forbidden"}, None),
    ),
)
def test_private_shared_summary_rejects_identity_count_and_strictness_poison(
    mutation: dict[str, object],
    match: str | None,
) -> None:
    payload = _shared_agent_summary().model_dump(mode="python")
    payload.update(mutation)

    with pytest.raises(ValidationError, match=match):
        SharedObsAgentPovReplayArtifactSummaryV1.model_validate(payload)


def test_private_shared_summary_requires_every_declared_field() -> None:
    model = _shared_agent_summary()

    for field_name in type(model).model_fields:
        payload = model.model_dump(mode="python")
        payload.pop(field_name)
        with pytest.raises(ValidationError, match="Field required"):
            SharedObsAgentPovReplayArtifactSummaryV1.model_validate(payload)


def test_private_shared_timeline_rejects_identity_count_tick_and_endpoint_poison() -> (
    None
):
    timeline = _shared_agent_timeline()

    poisoned_payloads: list[tuple[dict[str, object], str | None]] = []

    wrong_timeline_id = timeline.model_dump(mode="python")
    wrong_timeline_id["timeline_id"] = "episode-timeline:replay:timeline:researcher"
    poisoned_payloads.append((wrong_timeline_id, "recipient-local"))

    wrong_final = timeline.model_dump(mode="python")
    wrong_final["final_frame_index"] = 0
    poisoned_payloads.append((wrong_final, "captured prefix"))

    wrong_count = timeline.model_dump(mode="python")
    wrong_count["rows"] = wrong_count["rows"][:1]
    poisoned_payloads.append((wrong_count, "T\\+1"))

    wrong_frame = timeline.model_dump(mode="python")
    wrong_frame["rows"][1]["recipient_frame_id"] = "episode-timeline:frame:1"
    poisoned_payloads.append((wrong_frame, "frame identity"))

    wrong_zero_transition = timeline.model_dump(mode="python")
    wrong_zero_transition["rows"][0]["incoming_recipient_transition_id"] = (
        "episode-timeline:transition:0"
    )
    poisoned_payloads.append((wrong_zero_transition, "transition identity"))

    wrong_transition = timeline.model_dump(mode="python")
    wrong_transition["rows"][1]["incoming_recipient_transition_id"] = (
        "episode-timeline:transition:0"
    )
    poisoned_payloads.append((wrong_transition, "transition identity"))

    nonadjacent_tick = timeline.model_dump(mode="python")
    nonadjacent_tick["rows"][1]["simulator_step_count"] = 43
    poisoned_payloads.append((nonadjacent_tick, "epochs"))

    early_endpoint = timeline.model_dump(mode="python")
    early_endpoint["rows"][0]["endpoint_kind"] = "declared_horizon"
    poisoned_payloads.append((early_endpoint, "endpoint"))

    wrong_endpoint = timeline.model_dump(mode="python")
    wrong_endpoint["rows"][1]["endpoint_kind"] = "captured_prefix"
    poisoned_payloads.append((wrong_endpoint, "endpoint"))

    wrong_completion_episode = timeline.model_dump(mode="python")
    wrong_completion_episode["completion"]["episode_id"] = "other-episode"
    poisoned_payloads.append((wrong_completion_episode, "completion"))

    wrong_completion_horizon = timeline.model_dump(mode="python")
    wrong_completion_horizon["completion"]["expected_transition_count"] = 2
    wrong_completion_horizon["completion"]["completion_state"] = "partial"
    wrong_completion_horizon["completion"]["completion_bases"] = ()
    wrong_completion_horizon["completion"]["public_end_or_failure_reason"] = (
        "captured_prefix"
    )
    poisoned_payloads.append((wrong_completion_horizon, "horizon"))

    wrong_completion_count = timeline.model_dump(mode="python")
    wrong_completion_count["completion"]["captured_transition_count"] = 0
    wrong_completion_count["completion"]["completion_state"] = "partial"
    wrong_completion_count["completion"]["completion_bases"] = ()
    wrong_completion_count["completion"]["public_end_or_failure_reason"] = (
        "captured_prefix"
    )
    poisoned_payloads.append((wrong_completion_count, "captured replay prefix"))

    wrong_kind = timeline.model_dump(mode="python")
    wrong_kind["timeline_kind"] = "shared_obs_source_material"
    poisoned_payloads.append((wrong_kind, None))

    coercion = timeline.model_dump(mode="python")
    coercion["rows"][0]["frame_index"] = "0"
    poisoned_payloads.append((coercion, None))

    extra = timeline.model_dump(mode="python")
    extra["selected_global_slot"] = 5
    poisoned_payloads.append((extra, None))

    for payload, match in poisoned_payloads:
        with pytest.raises(ValidationError, match=match):
            SharedObsAgentPovReplayTimelineV1.model_validate(payload)


def test_private_shared_timeline_requires_every_root_and_row_field() -> None:
    timeline = _shared_agent_timeline()

    for field_name in type(timeline).model_fields:
        payload = timeline.model_dump(mode="python")
        payload.pop(field_name)
        with pytest.raises(ValidationError, match="Field required"):
            SharedObsAgentPovReplayTimelineV1.model_validate(payload)

    for field_name in SharedObsAgentPovReplayTimelineRowV1.model_fields:
        payload = timeline.model_dump(mode="python")
        payload["rows"][0].pop(field_name)
        with pytest.raises(ValidationError, match="Field required"):
            SharedObsAgentPovReplayTimelineV1.model_validate(payload)


@pytest.mark.parametrize(
    "factory",
    (_researcher_timeline, _pov_timeline, _shared_agent_timeline),
)
def test_each_audience_timeline_strictly_roundtrips_and_discriminates(
    factory: Callable[[], ReplayTimelineV1],
) -> None:
    timeline = factory()
    adapter: TypeAdapter[ReplayTimelineV1] = TypeAdapter(ReplayTimelineV1)
    encoded = adapter.dump_json(timeline)

    assert adapter.validate_json(encoded) == timeline
    payload = json.loads(encoded)
    assert payload["timeline_kind"] in {
        "researcher",
        "actor_pov",
        "shared_obs_agent_pov",
    }


def test_diagnostic_shared_timeline_remains_directly_parseable_but_is_not_product() -> (
    None
):
    diagnostic = _source_timeline()
    encoded = diagnostic.model_dump_json()

    assert SharedObsSourceMaterialReplayTimelineV1.model_validate_json(encoded) == (
        diagnostic
    )
    with pytest.raises(ValidationError):
        TypeAdapter(ReplayTimelineV1).validate_json(encoded)


def test_private_shared_frame_rejects_identity_epoch_and_strictness_poison() -> None:
    frame = _shared_agent_frame()
    poisoned_payloads: list[tuple[dict[str, object], str | None]] = []

    wrong_recipient = frame.model_dump(mode="python")
    wrong_recipient["public_agent_id"] = "agent-7"
    poisoned_payloads.append((wrong_recipient, "recipient"))

    wrong_timeline = frame.model_dump(mode="python")
    wrong_timeline["timeline_id"] = "episode-timeline:replay:timeline:researcher"
    poisoned_payloads.append((wrong_timeline, "recipient-local"))

    wrong_frame = frame.model_dump(mode="python")
    wrong_frame["recipient_frame_id"] = "episode-timeline:frame:1"
    poisoned_payloads.append((wrong_frame, "frame ID"))

    wrong_transition = frame.model_dump(mode="python")
    wrong_transition["incoming_recipient_transition_id"] = (
        "episode-timeline:transition:0"
    )
    poisoned_payloads.append((wrong_transition, "transition"))

    wrong_cursor = frame.model_dump(mode="python")
    wrong_cursor["cursor"]["final_frame_index"] = 2
    poisoned_payloads.append((wrong_cursor, "cursor endpoint"))

    wrong_completion_episode = frame.model_dump(mode="python")
    wrong_completion_episode["completion"]["episode_id"] = "other-episode"
    poisoned_payloads.append((wrong_completion_episode, "completion"))

    wrong_completion_horizon = frame.model_dump(mode="python")
    wrong_completion_horizon["completion"]["expected_transition_count"] = 2
    wrong_completion_horizon["completion"]["completion_state"] = "partial"
    wrong_completion_horizon["completion"]["completion_bases"] = ()
    wrong_completion_horizon["completion"]["public_end_or_failure_reason"] = (
        "captured_prefix"
    )
    poisoned_payloads.append((wrong_completion_horizon, "horizon"))

    wrong_completion_count = frame.model_dump(mode="python")
    wrong_completion_count["completion"]["captured_transition_count"] = 0
    wrong_completion_count["completion"]["completion_state"] = "partial"
    wrong_completion_count["completion"]["completion_bases"] = ()
    wrong_completion_count["completion"]["public_end_or_failure_reason"] = (
        "captured_prefix"
    )
    poisoned_payloads.append((wrong_completion_count, "captured replay prefix"))

    wrong_kind = frame.model_dump(mode="python")
    wrong_kind["frame_kind"] = "shared_obs_source_material_replay_viewer"
    poisoned_payloads.append((wrong_kind, None))

    wrong_preset = frame.model_dump(mode="python")
    wrong_preset["preset"] = "presentation"
    poisoned_payloads.append((wrong_preset, None))

    wrong_view = frame.model_dump(mode="python")
    wrong_view["view_mode"] = "researcher"
    poisoned_payloads.append((wrong_view, None))

    wrong_verbose = frame.model_dump(mode="python")
    wrong_verbose["verbose"] = True
    poisoned_payloads.append((wrong_verbose, None))

    coercion = frame.model_dump(mode="python")
    coercion["revision"] = "7"
    poisoned_payloads.append((coercion, None))

    extra = frame.model_dump(mode="python")
    extra["selected_global_slot"] = 5
    poisoned_payloads.append((extra, None))

    for payload, match in poisoned_payloads:
        with pytest.raises(ValidationError, match=match):
            SharedObsAgentPovReplayViewerFrameV1.model_validate(payload)


def test_private_shared_frame_requires_every_declared_field() -> None:
    frame = _shared_agent_frame()

    for field_name in type(frame).model_fields:
        payload = frame.model_dump(mode="python")
        payload.pop(field_name)
        with pytest.raises(ValidationError, match="Field required"):
            SharedObsAgentPovReplayViewerFrameV1.model_validate(payload)


def test_private_shared_frame_zero_has_no_incoming_recipient_transition() -> None:
    frame = _shared_agent_frame()
    payload = frame.model_dump(mode="python")
    payload["artifact_summary"] = _shared_agent_summary(
        expected=4,
        captured=0,
    ).model_dump(mode="python")
    payload["cursor"] = ReplayCursorV1(
        frame_index=0,
        final_frame_index=0,
        cursor_generation=0,
        choreography_generation=0,
    ).model_dump(mode="python")
    payload["recipient_frame_id"] = (
        "episode-timeline:shared-obs-visual-union:agent-5:frame:0"
    )
    payload["simulator_step_count"] = 41
    payload["incoming_recipient_transition_id"] = None
    payload["completion"] = _pov_completion(
        "episode-timeline",
        expected=4,
        captured=0,
        state="interrupted",
    ).model_dump(mode="python")

    initial = SharedObsAgentPovReplayViewerFrameV1.model_validate(payload)

    assert initial.cursor.frame_index == 0
    assert initial.incoming_recipient_transition_id is None


def test_private_shared_timeline_has_exact_prefix_and_dual_done_endpoint() -> None:
    partial_payload = _shared_agent_timeline().model_dump(mode="python")
    partial_payload["artifact_summary"] = _shared_agent_summary(
        expected=4,
        captured=0,
    ).model_dump(mode="python")
    partial_payload["final_frame_index"] = 0
    partial_payload["completion"] = _pov_completion(
        "episode-timeline",
        expected=4,
        captured=0,
        state="interrupted",
    ).model_dump(mode="python")
    partial_payload["rows"] = partial_payload["rows"][:1]
    partial_payload["rows"][0]["endpoint_kind"] = "captured_prefix"
    partial = SharedObsAgentPovReplayTimelineV1.model_validate(partial_payload)

    dual_payload = _shared_agent_timeline().model_dump(mode="python")
    dual_payload["completion"] = ActorPovReplayCompletionBadgeV1(
        episode_id="episode-timeline",
        completion_state="complete",
        expected_transition_count=1,
        captured_transition_count=1,
        terminated=True,
        truncated=True,
        completion_bases=("task_terminal", "declared_horizon"),
    ).model_dump(mode="python")
    dual_payload["rows"][1]["endpoint_kind"] = "task_terminal_and_declared_horizon"
    dual = SharedObsAgentPovReplayTimelineV1.model_validate(dual_payload)

    assert partial.rows[-1].endpoint_kind == "captured_prefix"
    assert dual.rows[-1].endpoint_kind == "task_terminal_and_declared_horizon"


def test_timelines_reject_identity_epoch_endpoint_and_privacy_drift() -> None:
    researcher = _researcher_timeline()
    payload = researcher.model_dump(mode="python")
    payload["rows"][1]["simulator_step_count"] = 44
    with pytest.raises(ValidationError, match="epochs"):
        ResearcherReplayTimelineV1.model_validate(payload)

    payload = _pov_timeline().model_dump(mode="python")
    payload["rows"][1]["incoming_pov_transition_id"] = "episode-timeline:transition:0"
    with pytest.raises(ValidationError, match="POV timeline transition"):
        ActorPovReplayTimelineV1.model_validate(payload)

    payload = _source_timeline().model_dump(mode="python")
    payload["rows"][1]["endpoint_kind"] = "captured_prefix"
    with pytest.raises(ValidationError, match="endpoint"):
        SharedObsSourceMaterialReplayTimelineV1.model_validate(payload)

    unknown = _researcher_timeline().model_dump(mode="json")
    unknown["timeline_kind"] = "future"
    with pytest.raises(ValidationError):
        TypeAdapter(ReplayTimelineV1).validate_python(unknown)

    strict_tuple = _exact_model_input(_researcher_timeline())
    strict_tuple["rows"] = list(_researcher_timeline().rows)
    with pytest.raises(ValidationError):
        ResearcherReplayTimelineV1.model_validate(strict_tuple)


def test_partial_and_simultaneous_completion_endpoint_labels_are_exact() -> None:
    episode_id = "episode-partial"
    partial = ResearcherReplayTimelineV1(
        timeline_id=f"{episode_id}:replay:timeline:researcher",
        artifact_summary=_summary(episode_id, expected=4, recorded=0),
        final_frame_index=0,
        completion=_completion(
            episode_id,
            expected=4,
            validated=0,
            state="interrupted",
        ),
        rows=(
            ResearcherReplayTimelineRowV1(
                frame_index=0,
                frame_id=f"{episode_id}:frame:0",
                simulator_step_count=12,
                incoming_transition_id=None,
                incoming_event_count=0,
                endpoint_kind="captured_prefix",
            ),
        ),
    )
    simultaneous_completion = ReplayCompletionBadgeV1(
        episode_id="episode-both",
        completion_state="complete",
        expected_transition_count=1,
        validated_transition_count=1,
        last_valid_frame_index=1,
        last_valid_frame_id="episode-both:frame:1",
        terminated=True,
        truncated=True,
        completion_bases=("task_terminal", "declared_horizon"),
    )

    assert partial.rows[-1].endpoint_kind == "captured_prefix"
    assert simultaneous_completion.terminated
    assert simultaneous_completion.truncated


def test_processing_failure_is_minimal_and_separate_from_completion() -> None:
    failed = _processing(failed=True)
    lifecycle = ReplayProcessingBadgeV1(
        status="failed",
        processed_transition_count=1,
        failure_stage="lifecycle",
        failure_code="observer_already_finalized",
    )
    payload = json.loads(failed.model_dump_json())

    assert payload == {
        "schema_version": 1,
        "status": "failed",
        "processed_transition_count": 0,
        "failure_stage": "reducer_advance",
        "failure_code": "test.reducer_failed",
        "attempted_transition_index": 0,
    }
    assert "detail" not in payload
    assert "reducer_id" not in payload
    assert _json_roundtrip(lifecycle) == lifecycle
    with pytest.raises(ValidationError):
        ReplayProcessingBadgeV1(
            status="succeeded",
            processed_transition_count=1,
            failure_stage="reducer_finalize",
            failure_code="invalid",
        )


def test_output_frames_serialize_canonical_audience_specific_envelopes(
    projection_cases: _ProjectionCases,
) -> None:
    frames = (
        _researcher_frame(projection_cases.researcher),
        _actor_pov_frame(projection_cases.pov),
        _source_material_frame(projection_cases.source_material),
        _shared_agent_frame(),
    )
    expected_kinds = (
        "researcher_replay_viewer",
        "actor_pov_replay_viewer",
        "shared_obs_source_material_replay_viewer",
        "shared_obs_agent_pov_replay_viewer",
    )

    for frame, expected_kind in zip(frames, expected_kinds, strict=True):
        first = frame.model_dump_json()
        assert first == frame.model_dump_json()
        payload = json.loads(first)
        assert payload["frame_kind"] == expected_kind
        assert "hud" not in payload
        assert "scenario" not in payload
        assert "available_scenarios" not in payload
        assert "animate_incoming" not in payload
        if expected_kind == "researcher_replay_viewer":
            assert payload["recorded_ordinary_movement_distance_scale"] == 1.0
        else:
            assert "recorded_ordinary_movement_distance_scale" not in payload


def test_recorded_movement_scale_is_strict_researcher_only_replay_truth(
    projection_cases: _ProjectionCases,
) -> None:
    researcher = _researcher_frame(projection_cases.researcher)
    payload = _exact_model_input(researcher)
    payload["recorded_ordinary_movement_distance_scale"] = 0.375
    experimental = ResearcherReplayViewerFrameV1.model_validate(payload)

    assert experimental.recorded_ordinary_movement_distance_scale == 0.375

    for invalid in (0.0, -0.1, 1.01, float("inf"), float("nan")):
        invalid_payload = _exact_model_input(researcher)
        invalid_payload["recorded_ordinary_movement_distance_scale"] = invalid
        with pytest.raises(ValidationError):
            ResearcherReplayViewerFrameV1.model_validate(invalid_payload)

    missing_payload = _exact_model_input(researcher)
    missing_payload.pop("recorded_ordinary_movement_distance_scale")
    with pytest.raises(ValidationError, match="Field required"):
        ResearcherReplayViewerFrameV1.model_validate(missing_payload)

    for hidden_frame in (
        _actor_pov_frame(projection_cases.pov),
        _source_material_frame(projection_cases.source_material),
        _shared_agent_frame(),
    ):
        hidden_payload = _exact_model_input(hidden_frame)
        hidden_payload["recorded_ordinary_movement_distance_scale"] = 0.375
        with pytest.raises(ValidationError, match="Extra inputs"):
            type(hidden_frame).model_validate(hidden_payload)


def test_exact_pov_frame_cannot_expose_processing_or_researcher_truth(
    projection_cases: _ProjectionCases,
) -> None:
    frame = _actor_pov_frame(projection_cases.pov)
    payload = json.loads(frame.model_dump_json())
    all_keys = _recursive_keys(payload)

    assert payload["processing_disclosure"] == {
        "schema_version": 1,
        "disclosure": ACTOR_POV_PROCESSING_DISCLOSURE_V1,
    }
    assert (
        payload["artifact_summary"]["metric_report_availability"]
        == ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1
    )
    for forbidden in (
        "processing",
        "incoming_event_count",
        "incoming_event_ids",
        "incoming_events",
        "status_source_evidence",
        "class_mechanics",
        "aura_fields",
        "sensor_source_availability",
        "failure_stage",
        "failure_code",
        "attempted_transition_index",
        "reducer_id",
        "detail",
    ):
        assert forbidden not in all_keys
    forged = _exact_model_input(frame)
    forged["processing"] = _processing(failed=True)
    with pytest.raises(ValidationError, match="Extra inputs"):
        ActorPovReplayViewerFrameV1.model_validate(forged)


def test_each_audience_enforces_metric_report_availability_disclosure(
    projection_cases: _ProjectionCases,
) -> None:
    researcher = _researcher_frame(projection_cases.researcher)
    researcher_payload = _exact_model_input(researcher)
    researcher_payload["artifact_summary"] = researcher.artifact_summary.model_copy(
        update={"metric_report_availability": (ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1)}
    )
    with pytest.raises(ValidationError, match="researcher/source-material"):
        ResearcherReplayViewerFrameV1.model_validate(researcher_payload)

    pov = _actor_pov_frame(projection_cases.pov)
    for disclosure in ("available", "missing"):
        pov_payload = _exact_model_input(pov)
        pov_payload["artifact_summary"] = pov.artifact_summary.model_copy(
            update={"metric_report_availability": disclosure}
        )
        with pytest.raises(ValidationError, match="must not disclose"):
            ActorPovReplayViewerFrameV1.model_validate(pov_payload)

    source = _source_material_frame(projection_cases.source_material)
    source_payload = _exact_model_input(source)
    source_payload["artifact_summary"] = source.artifact_summary.model_copy(
        update={"metric_report_availability": (ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1)}
    )
    with pytest.raises(ValidationError, match="researcher/source-material"):
        SharedObsSourceMaterialReplayViewerFrameV1.model_validate(source_payload)

    pov_timeline = _pov_timeline()
    pov_timeline_payload = _exact_model_input(pov_timeline)
    pov_timeline_payload["artifact_summary"] = pov_timeline.artifact_summary.model_copy(
        update={"metric_report_availability": "available"}
    )
    with pytest.raises(ValidationError, match="must not disclose"):
        ActorPovReplayTimelineV1.model_validate(pov_timeline_payload)

    for timeline in (_researcher_timeline(), _source_timeline()):
        timeline_payload = _exact_model_input(timeline)
        timeline_payload["artifact_summary"] = timeline.artifact_summary.model_copy(
            update={
                "metric_report_availability": (ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1)
            }
        )
        with pytest.raises(ValidationError, match="researcher/source-material"):
            type(timeline).model_validate(timeline_payload)


def test_hidden_processing_differences_cannot_change_exact_pov_frame_bytes(
    projection_cases: _ProjectionCases,
) -> None:
    frame = _actor_pov_frame(projection_cases.pov)
    hidden_source_processing = (_processing(), _processing(failed=True))

    serialized = tuple(frame.model_dump_json() for _ in hidden_source_processing)

    assert serialized[0] == serialized[1]
    assert "reducer_advance" not in serialized[0]
    assert "test.reducer_failed" not in serialized[0]


def test_output_frame_validators_reject_cross_audience_and_epoch_joins(
    projection_cases: _ProjectionCases,
) -> None:
    researcher = _researcher_frame(projection_cases.researcher)
    payload = _exact_model_input(researcher)
    payload["incoming_transition_id"] = None
    with pytest.raises(ValidationError, match="incoming researcher"):
        ResearcherReplayViewerFrameV1.model_validate(payload)

    pov = _actor_pov_frame(projection_cases.pov)
    payload = _exact_model_input(pov)
    payload["public_agent_id"] = "different-agent"
    with pytest.raises(ValidationError):
        ActorPovReplayViewerFrameV1.model_validate(payload)

    source = _source_material_frame(projection_cases.source_material)
    payload = _exact_model_input(source)
    payload["observation_materialization"] = "exact_actor_input"
    with pytest.raises(ValidationError):
        SharedObsSourceMaterialReplayViewerFrameV1.model_validate(payload)


def test_response_owns_transient_animation_intent_only(
    projection_cases: _ProjectionCases,
) -> None:
    frame = _researcher_frame(projection_cases.researcher)
    animated = ReplayCommandResponseV1(
        result="applied",
        frame=frame,
        animate_incoming=True,
    )
    duplicate = ReplayCommandResponseV1(result="duplicate", frame=frame)
    error = ReplayApiErrorV1(
        error_code="stale_revision",
        message="The replay cursor changed.",
        latest_frame=frame,
    )

    assert json.loads(animated.model_dump_json())["animate_incoming"] is True
    assert json.loads(duplicate.model_dump_json())["animate_incoming"] is False
    assert "animate_incoming" not in json.loads(frame.model_dump_json())
    assert "animate_incoming" not in json.loads(error.model_dump_json())
    with pytest.raises(ValidationError, match="applied"):
        ReplayCommandResponseV1(
            result="duplicate",
            frame=frame,
            animate_incoming=True,
        )


def test_response_and_error_accept_private_frame_and_reject_diagnostic_frame(
    projection_cases: _ProjectionCases,
) -> None:
    private = _shared_agent_frame()
    response = ReplayCommandResponseV1(result="duplicate", frame=private)
    error = ReplayApiErrorV1(
        error_code="stale_revision",
        message="The replay cursor changed.",
        latest_frame=private,
    )

    assert _json_roundtrip(response) == response
    assert _json_roundtrip(error) == error
    assert response.frame is private or response.frame == private
    assert error.latest_frame is private or error.latest_frame == private

    diagnostic = _source_material_frame(projection_cases.source_material)
    with pytest.raises(ValidationError):
        ReplayCommandResponseV1.model_validate(
            {"result": "duplicate", "frame": diagnostic.model_dump(mode="python")}
        )
    with pytest.raises(ValidationError):
        ReplayApiErrorV1.model_validate(
            {
                "error_code": "stale_revision",
                "message": "The replay cursor changed.",
                "latest_frame": diagnostic.model_dump(mode="python"),
            }
        )


def test_frame_zero_cannot_request_incoming_animation(
    projection_cases: _ProjectionCases,
) -> None:
    frame = _researcher_initial_frame(projection_cases.researcher_initial)

    with pytest.raises(ValidationError, match="frame zero"):
        ReplayCommandResponseV1(
            result="applied",
            frame=frame,
            animate_incoming=True,
        )


def test_discriminated_frame_alias_serializes_without_live_protocol_import(
    projection_cases: _ProjectionCases,
) -> None:
    frames = (
        _researcher_frame(projection_cases.researcher),
        _actor_pov_frame(projection_cases.pov),
        _shared_agent_frame(),
    )
    adapter: TypeAdapter[ReplayViewerFrameV1] = TypeAdapter(ReplayViewerFrameV1)

    for frame in frames:
        payload = adapter.dump_python(frame, mode="json")
        assert payload["schema_version"] == 1
        assert payload["frame_kind"] == frame.frame_kind


def test_diagnostic_shared_frame_remains_directly_parseable_but_is_not_product(
    projection_cases: _ProjectionCases,
) -> None:
    diagnostic = _source_material_frame(projection_cases.source_material)
    encoded = diagnostic.model_dump_json()

    assert SharedObsSourceMaterialReplayViewerFrameV1.model_validate_json(encoded) == (
        diagnostic
    )
    with pytest.raises(ValidationError):
        TypeAdapter(ReplayViewerFrameV1).validate_json(encoded)


def test_private_shared_roots_have_exact_identity_only_keys_and_values() -> None:
    frame_payload = json.loads(_shared_agent_frame().model_dump_json())
    timeline_payload = json.loads(_shared_agent_timeline().model_dump_json())
    common_nested_keys = {
        "schema_version",
        "recipient_replay_id",
        "episode_id",
        "public_agent_id",
        "expected_transition_count",
        "captured_transition_count",
        "captured_frame_count",
        "completion_state",
        "terminated",
        "truncated",
        "completion_bases",
        "public_end_or_failure_reason",
    }
    expected_frame_keys = common_nested_keys | {
        "frame_kind",
        "viewer_session_id",
        "revision",
        "artifact_summary",
        "timeline_id",
        "cursor",
        "frame_index",
        "final_frame_index",
        "cursor_generation",
        "choreography_generation",
        "preset",
        "verbose",
        "view_mode",
        "recipient_frame_id",
        "simulator_step_count",
        "incoming_recipient_transition_id",
        "completion",
    }
    expected_timeline_keys = common_nested_keys | {
        "timeline_kind",
        "timeline_id",
        "artifact_summary",
        "final_frame_index",
        "completion",
        "rows",
        "frame_index",
        "recipient_frame_id",
        "simulator_step_count",
        "incoming_recipient_transition_id",
        "endpoint_kind",
    }

    assert _recursive_keys(frame_payload) == expected_frame_keys
    assert _recursive_keys(timeline_payload) == expected_timeline_keys

    all_strings = _recursive_strings(frame_payload) | _recursive_strings(
        timeline_payload
    )
    forbidden_exact_values = {
        "episode-timeline:replay",
        "episode-timeline:frame:0",
        "episode-timeline:frame:1",
        "episode-timeline:transition:0",
        _DIGEST_A,
        _DIGEST_B,
        _DIGEST_C,
    }
    assert all_strings.isdisjoint(forbidden_exact_values)
    assert all(":shared-obs-source-material:" not in value for value in all_strings)
    assert all(len(value) != 64 for value in all_strings)
    assert all("/" not in value and "\\" not in value for value in all_strings)
    assert all(
        not any(
            token in value.lower()
            for token in ("exception", "traceback", "reducer", "policy", ".json")
        )
        for value in all_strings
    )


def test_researcher_and_no_shared_model_dump_bytes_are_frozen(
    projection_cases: _ProjectionCases,
) -> None:
    cases = projection_cases
    models_and_hashes = (
        (
            _researcher_frame(cases.researcher),
            _RESEARCHER_FRAME_V1_SHA256,
        ),
        (_actor_pov_frame(cases.pov), _ACTOR_POV_FRAME_V1_SHA256),
        (_researcher_timeline(), _RESEARCHER_TIMELINE_V1_SHA256),
        (_pov_timeline(), _ACTOR_POV_TIMELINE_V1_SHA256),
    )

    for model, expected_hash in models_and_hashes:
        actual_hash = hashlib.sha256(model.model_dump_json().encode()).hexdigest()
        assert actual_hash == expected_hash


def test_private_shared_validation_is_frozen_repeatable_and_source_nonmutating() -> (
    None
):
    models_and_adapters: tuple[tuple[BaseModel, TypeAdapter[object]], ...] = (
        (
            _shared_agent_summary(),
            TypeAdapter(SharedObsAgentPovReplayArtifactSummaryV1),
        ),
        (
            _shared_agent_timeline().rows[1],
            TypeAdapter(SharedObsAgentPovReplayTimelineRowV1),
        ),
        (
            _shared_agent_timeline(),
            TypeAdapter(SharedObsAgentPovReplayTimelineV1),
        ),
        (
            _shared_agent_frame(),
            TypeAdapter(SharedObsAgentPovReplayViewerFrameV1),
        ),
    )

    for model, adapter in models_and_adapters:
        payload = model.model_dump(mode="python")
        snapshot = deepcopy(payload)
        first = adapter.validate_python(payload)
        second = adapter.validate_python(payload)
        assert payload == snapshot
        assert first == second == model
        assert model.model_dump_json() == model.model_dump_json()
        with pytest.raises(ValidationError):
            setattr(model, next(iter(type(model).model_fields)), "poison")


def test_replay_protocol_import_is_core_jax_numpy_and_live_seam_free() -> None:
    code = """
import sys
import scripts.dev.visual_debugger.replay_protocol
banned = (
    'jax',
    'jaxlib',
    'numpy',
    'marl_battlegrounds.core',
    'scripts.dev.visual_debugger.protocol',
    'scripts.dev.visual_debugger.service',
    'scripts.dev.visual_debugger.control',
    'scripts.dev.visual_debugger.scenarios',
)
loaded = [
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in banned)
]
assert loaded == [], loaded
print('isolated')
"""
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cuda"
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated"
