"""Strict, audience-separated replay-viewer wire-contract proofs."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError
from scripts.dev.visual_debugger.renderer_fixtures import get_renderer_fixture
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
    SharedObsSourceMaterialReplayTimelineRowV1,
    SharedObsSourceMaterialReplayTimelineV1,
    SharedObsSourceMaterialReplayViewerFrameV1,
)
from tests.evaluation_fixtures import captured_evaluation_trajectory

from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.replay import ReplayArtifactReferenceV1
from marl_battlegrounds.rendering.evaluation_adapter import (
    SharedObsSourceMaterialProjectionV1,
    build_shared_obs_source_material_projection_v1,
)
from marl_battlegrounds.rendering.pov_scene import ActorPovAnalyzerProjectionV1
from marl_battlegrounds.rendering.scene import ResearcherAnalyzerProjectionV2

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


@dataclass(frozen=True, slots=True)
class _ProjectionCases:
    researcher: ResearcherAnalyzerProjectionV2
    researcher_initial: ResearcherAnalyzerProjectionV2
    pov: ActorPovAnalyzerProjectionV1
    source_material: SharedObsSourceMaterialProjectionV1


@pytest.fixture(scope="module")
def projection_cases() -> _ProjectionCases:
    researcher = cast(
        ResearcherAnalyzerProjectionV2,
        get_renderer_fixture("canonical_event_vocabulary").live_frame.projection,
    )
    researcher_initial = cast(
        ResearcherAnalyzerProjectionV2,
        get_renderer_fixture("durable_controls").live_frame.projection,
    )
    pov = cast(
        ActorPovAnalyzerProjectionV1,
        get_renderer_fixture("pov_redaction").live_frame.projection,
    )
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        execution_information_mode="shared_obs",
    )
    incoming = EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[0],
        transition=trajectory.transitions[0],
        successor_frame=trajectory.frames[1],
    )
    source_material = build_shared_obs_source_material_projection_v1(
        trajectory.context,
        trajectory.frames[1],
        selected_global_slot=5,
        transition_view=incoming,
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


@pytest.mark.parametrize(
    "factory",
    (_researcher_timeline, _pov_timeline, _source_timeline),
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
        "shared_obs_source_material",
    }


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
    )
    expected_kinds = (
        "researcher_replay_viewer",
        "actor_pov_replay_viewer",
        "shared_obs_source_material_replay_viewer",
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
        _source_material_frame(projection_cases.source_material),
    )
    adapter: TypeAdapter[ReplayViewerFrameV1] = TypeAdapter(ReplayViewerFrameV1)

    for frame in frames:
        payload = adapter.dump_python(frame, mode="json")
        assert payload["schema_version"] == 1
        assert payload["frame_kind"] == frame.frame_kind


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
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated"
