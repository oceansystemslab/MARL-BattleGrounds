"""Read-only replay-service authority, privacy, and cursor proofs."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from typing import cast

import pytest
import scripts.dev.visual_debugger.replay_service as replay_service_module
from scripts.dev.visual_debugger.replay_protocol import (
    ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1,
    ACTOR_POV_PROCESSING_DISCLOSURE_V1,
    ActorPovReplayTimelineV1,
    ActorPovReplayViewerFrameV1,
    ReplayAbsoluteSeekCommandV1,
    ReplayApiErrorV1,
    ReplayCommandRequestV1,
    ReplayCommandResponseV1,
    ReplayCommandV1,
    ReplayExitCommandV1,
    ReplayFirstFrameCommandV1,
    ReplayLastFrameCommandV1,
    ReplayNextFrameCommandV1,
    ReplayPreviousFrameCommandV1,
    ReplaySelectAgentCommandV1,
    ReplaySetPovActorCommandV1,
    ReplaySetPresetCommandV1,
    ReplaySetRangesCommandV1,
    ReplaySetVerbosityCommandV1,
    ReplaySetViewCommandV1,
    ResearcherReplayTimelineV1,
    ResearcherReplayViewerFrameV1,
    SharedObsSourceMaterialReplayTimelineV1,
    SharedObsSourceMaterialReplayViewerFrameV1,
)
from scripts.dev.visual_debugger.replay_service import (
    ReplayServiceCommandResultV1,
    ReplayViewerService,
)
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
)

from marl_battlegrounds.evaluation.metrics import (
    CompletionState,
    EvaluationEpisodeCompletionV1,
    EvaluationMetricReducerStateV1,
    EvaluationMetricReducerV1,
    EvaluationMetricReportV1,
    EvaluationProcessingStatusV1,
    EvaluationTransitionViewV1,
    RolloutFailureOrigin,
    SufficientStatisticDraftV1,
    build_evaluation_observer_v1,
)
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV1,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationTransitionV1,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.replay import (
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundleV1
from marl_battlegrounds.rendering.scene import StatusSourceEvidenceIndexV2

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _FinalizeFailureState(EvaluationMetricReducerStateV1):
    """Valid immutable state used before an intentional finalization failure."""


@dataclass(slots=True)
class _FinalizeFailureReducer:
    """Fail only while materializing metrics, after every unit was processed."""

    reducer_id: str = "test.replay_service.finalize_failure"
    reducer_version: int = 1

    def initialize(
        self,
        context: EvaluationEpisodeContextV1,
        initial_frame: EvaluationFrameV1,
    ) -> EvaluationMetricReducerStateV1:
        del context, initial_frame
        return _FinalizeFailureState(
            reducer_id=self.reducer_id,
            reducer_version=self.reducer_version,
        )

    def advance(
        self,
        previous_state: EvaluationMetricReducerStateV1,
        view: EvaluationTransitionViewV1,
    ) -> EvaluationMetricReducerStateV1:
        del view
        return previous_state

    def finalize(
        self,
        state: EvaluationMetricReducerStateV1,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        del state, completion, processing_status
        raise RuntimeError("intentional replay-service finalization failure")


@dataclass(frozen=True, slots=True)
class _ServiceCase:
    """One canonical loaded bundle plus its source report."""

    bundle: LoadedReplayBundleV1
    report: EvaluationMetricReportV1


@dataclass(frozen=True, slots=True)
class _ServiceCases:
    complete: _ServiceCase
    partial: _ServiceCase
    interrupted: _ServiceCase
    failed: _ServiceCase
    zero_partial: _ServiceCase
    metric_missing: _ServiceCase
    shared: _ServiceCase
    processing_healthy: _ServiceCase
    processing_failed: _ServiceCase
    long: _ServiceCase
    nonzero_focal: _ServiceCase


def _runtime_provenance() -> RuntimeProvenanceV1:
    return RuntimeProvenanceV1(
        python_version="3.13.0",
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


def _loaded_case(
    trajectory: CapturedEvaluationTrajectory,
    *,
    completion_state: CompletionState,
    reducers: tuple[EvaluationMetricReducerV1, ...] = (),
    metric_report_available: bool = True,
) -> _ServiceCase:
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=reducers,
    )
    observer.start(trajectory.frames[0])
    for transition, successor_frame in zip(
        trajectory.transitions,
        trajectory.frames[1:],
        strict=True,
    ):
        observer.append(transition, successor_frame)
    reason = None if completion_state == "complete" else f"test_{completion_state}"
    origin: RolloutFailureOrigin | None = (
        "capture" if completion_state == "failed" else None
    )
    report = observer.finalize(
        completion_state=completion_state,
        end_or_failure_reason=reason,
        failure_origin=origin,
    )
    replay_bundle = build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=_runtime_provenance(),
    )
    loaded = LoadedReplayBundleV1(
        replay=replay_bundle.replay,
        metric_report_artifact=(
            replay_bundle.metric_report_artifact if metric_report_available else None
        ),
        status="complete" if metric_report_available else "metric_report_missing",
    )
    return _ServiceCase(bundle=loaded, report=report)


def _with_nonzero_focal(
    trajectory: CapturedEvaluationTrajectory,
) -> CapturedEvaluationTrajectory:
    assignments = tuple(
        row.model_copy(
            update={
                "evaluation_role": (
                    "focal"
                    if row.global_slot == 5
                    else "cooperative_partner"
                    if row.global_slot == 0
                    else row.evaluation_role
                )
            }
        )
        if isinstance(row, AssignedPolicySlotV1)
        else row
        for row in trajectory.context.policy_assignments
    )
    context_payload = trajectory.context.model_dump(mode="python")
    context_payload["policy_assignments"] = assignments
    context = EvaluationEpisodeContextV1.model_validate(context_payload)
    return CapturedEvaluationTrajectory(
        context=context,
        frames=trajectory.frames,
        transitions=trajectory.transitions,
    )


@pytest.fixture(scope="module")
def service_cases() -> _ServiceCases:
    complete_trajectory = captured_evaluation_trajectory(
        transition_count=2,
        expected_horizon=2,
        episode_id="service-complete",
    )
    prefix_trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=3,
        episode_id="service-prefix",
    )
    zero_trajectory = captured_evaluation_trajectory(
        transition_count=0,
        expected_horizon=3,
        episode_id="service-zero",
    )
    shared_trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        execution_information_mode="shared_obs",
        episode_id="service-shared",
    )
    processing_trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        episode_id="service-processing",
    )
    long_trajectory = captured_evaluation_trajectory(
        transition_count=6,
        expected_horizon=6,
        episode_id="service-long",
    )
    focal_trajectory = _with_nonzero_focal(
        captured_evaluation_trajectory(
            transition_count=1,
            expected_horizon=1,
            episode_id="service-focal",
        )
    )

    complete = _loaded_case(complete_trajectory, completion_state="complete")
    processing_healthy = _loaded_case(
        processing_trajectory,
        completion_state="complete",
    )
    return _ServiceCases(
        complete=complete,
        partial=_loaded_case(prefix_trajectory, completion_state="partial"),
        interrupted=_loaded_case(prefix_trajectory, completion_state="interrupted"),
        failed=_loaded_case(prefix_trajectory, completion_state="failed"),
        zero_partial=_loaded_case(zero_trajectory, completion_state="partial"),
        metric_missing=_ServiceCase(
            bundle=LoadedReplayBundleV1(
                replay=complete.bundle.replay,
                metric_report_artifact=None,
                status="metric_report_missing",
            ),
            report=complete.report,
        ),
        shared=_loaded_case(shared_trajectory, completion_state="complete"),
        processing_healthy=processing_healthy,
        processing_failed=_loaded_case(
            processing_trajectory,
            completion_state="complete",
            reducers=(cast(EvaluationMetricReducerV1, _FinalizeFailureReducer()),),
        ),
        long=_loaded_case(long_trajectory, completion_state="complete"),
        nonzero_focal=_loaded_case(focal_trajectory, completion_state="complete"),
    )


def _request(
    service: ReplayViewerService,
    command: ReplayCommandV1,
    *,
    command_id: str,
    client_id: str = "client-a",
    base_revision: int | None = None,
) -> ReplayCommandRequestV1:
    return ReplayCommandRequestV1(
        client_id=client_id,
        command_id=command_id,
        base_revision=service.revision if base_revision is None else base_revision,
        command=command,
    )


def _apply(
    service: ReplayViewerService,
    command: ReplayCommandV1,
    *,
    command_id: str,
    client_id: str = "client-a",
    base_revision: int | None = None,
) -> ReplayServiceCommandResultV1:
    return service.apply_command(
        _request(
            service,
            command,
            command_id=command_id,
            client_id=client_id,
            base_revision=base_revision,
        )
    )


def _response(result: ReplayServiceCommandResultV1) -> ReplayCommandResponseV1:
    assert result.outcome == "response"
    assert isinstance(result.payload, ReplayCommandResponseV1)
    return result.payload


def _error(result: ReplayServiceCommandResultV1) -> ReplayApiErrorV1:
    assert result.outcome != "response"
    assert isinstance(result.payload, ReplayApiErrorV1)
    return result.payload


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


@pytest.mark.parametrize("invalid_session_id", ("", "   ", 7))
def test_viewer_session_id_rejects_explicit_invalid_values(
    service_cases: _ServiceCases,
    invalid_session_id: object,
) -> None:
    with pytest.raises(ValueError, match="viewer_session_id"):
        ReplayViewerService(
            service_cases.complete.bundle,
            viewer_session_id=cast(str, invalid_session_id),
        )


@pytest.mark.parametrize(
    ("case_name", "state", "endpoint"),
    (
        ("complete", "complete", "declared_horizon"),
        ("partial", "partial", "captured_prefix"),
        ("interrupted", "interrupted", "captured_prefix"),
        ("failed", "failed", "captured_prefix"),
        ("zero_partial", "partial", "captured_prefix"),
    ),
)
def test_completion_states_and_captured_prefix_endpoints_are_exact(
    service_cases: _ServiceCases,
    case_name: str,
    state: str,
    endpoint: str,
) -> None:
    case = cast(_ServiceCase, getattr(service_cases, case_name))
    final_index = len(case.bundle.replay.transitions)
    service = ReplayViewerService(
        case.bundle,
        initial_frame_index=final_index,
        viewer_session_id=f"viewer-{case_name}",
    )

    frame = service.current_frame()
    timeline = service.current_timeline()

    assert isinstance(frame, ResearcherReplayViewerFrameV1)
    assert isinstance(timeline, ResearcherReplayTimelineV1)
    assert frame.completion.completion_state == state
    assert frame.cursor.frame_index == final_index
    assert frame.cursor.final_frame_index == final_index
    assert timeline.rows[-1].endpoint_kind == endpoint
    assert frame.completion.terminated is False
    if state == "complete":
        assert frame.completion.end_or_failure_reason is None
        assert frame.completion.completion_bases == ("declared_horizon",)
    else:
        assert frame.completion.end_or_failure_reason == f"test_{state}"
        assert frame.completion.completion_bases == ()
    assert frame.completion.failure_origin == ("capture" if state == "failed" else None)
    if case_name == "zero_partial":
        assert final_index == 0
        assert frame.incoming_transition_id is None
        assert frame.incoming_transition_index is None
        assert len(timeline.rows) == 1


def test_missing_metric_sidecar_is_researcher_visible_but_pov_hidden(
    service_cases: _ServiceCases,
) -> None:
    researcher = ReplayViewerService(
        service_cases.metric_missing.bundle,
        viewer_session_id="missing-metric-researcher",
    )
    exact_pov = ReplayViewerService(
        service_cases.metric_missing.bundle,
        view_mode="pov",
        viewer_session_id="missing-metric-pov",
    )

    researcher_frame = researcher.current_frame()
    pov_frame = exact_pov.current_frame()
    pov_timeline = exact_pov.current_timeline()

    assert isinstance(researcher_frame, ResearcherReplayViewerFrameV1)
    assert isinstance(pov_frame, ActorPovReplayViewerFrameV1)
    assert isinstance(pov_timeline, ActorPovReplayTimelineV1)
    assert researcher_frame.artifact_summary.metric_report_availability == "missing"
    assert (
        pov_frame.artifact_summary.metric_report_availability
        == ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1
    )
    assert (
        pov_timeline.artifact_summary.metric_report_availability
        == ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1
    )


def test_loaded_bundle_status_must_match_metric_sidecar_presence(
    service_cases: _ServiceCases,
) -> None:
    available = service_cases.complete.bundle
    contradictory_missing = LoadedReplayBundleV1(
        replay=available.replay,
        metric_report_artifact=available.metric_report_artifact,
        status="metric_report_missing",
    )
    contradictory_complete = LoadedReplayBundleV1(
        replay=available.replay,
        metric_report_artifact=None,
        status="complete",
    )

    for bundle in (contradictory_missing, contradictory_complete):
        with pytest.raises(ValueError, match="sidecar availability"):
            ReplayViewerService(bundle)


def test_exact_pov_is_byte_identical_with_or_without_metric_sidecar(
    service_cases: _ServiceCases,
) -> None:
    available = ReplayViewerService(
        service_cases.complete.bundle,
        view_mode="pov",
        viewer_session_id="same-pov-session",
    )
    missing = ReplayViewerService(
        service_cases.metric_missing.bundle,
        view_mode="pov",
        viewer_session_id="same-pov-session",
    )

    assert available.current_frame().model_dump_json() == (
        missing.current_frame().model_dump_json()
    )
    assert available.current_timeline().model_dump_json() == (
        missing.current_timeline().model_dump_json()
    )


def test_exact_pov_processing_failure_disclosure_is_constant_and_content_stable(
    service_cases: _ServiceCases,
) -> None:
    healthy = ReplayViewerService(
        service_cases.processing_healthy.bundle,
        view_mode="pov",
        initial_frame_index=1,
        viewer_session_id="processing-pov",
    )
    failed = ReplayViewerService(
        service_cases.processing_failed.bundle,
        view_mode="pov",
        initial_frame_index=1,
        viewer_session_id="processing-pov",
    )
    assert service_cases.processing_failed.report.processing_status.status == "failed"

    healthy_frame = cast(
        ActorPovReplayViewerFrameV1,
        healthy.current_frame(),
    )
    failed_frame = cast(
        ActorPovReplayViewerFrameV1,
        failed.current_frame(),
    )
    healthy_payload = json.loads(healthy_frame.model_dump_json())
    failed_payload = json.loads(failed_frame.model_dump_json())
    healthy_reference = healthy_payload["artifact_summary"].pop("replay_reference")
    failed_reference = failed_payload["artifact_summary"].pop("replay_reference")

    assert healthy_reference != failed_reference
    assert healthy_payload == failed_payload
    assert healthy_frame.processing_disclosure.disclosure == (
        ACTOR_POV_PROCESSING_DISCLOSURE_V1
    )
    assert failed_frame.processing_disclosure.disclosure == (
        ACTOR_POV_PROCESSING_DISCLOSURE_V1
    )
    for payload in (healthy_payload, failed_payload):
        keys = _recursive_keys(payload)
        assert "processing" not in keys
        assert "failure_stage" not in keys
        assert "failure_code" not in keys
        assert "attempted_transition_index" not in keys
        assert "reducer_id" not in keys

    healthy_timeline = json.loads(healthy.current_timeline().model_dump_json())
    failed_timeline = json.loads(failed.current_timeline().model_dump_json())
    healthy_timeline["artifact_summary"].pop("replay_reference")
    failed_timeline["artifact_summary"].pop("replay_reference")
    assert healthy_timeline == failed_timeline


def test_shared_obs_view_is_labelled_source_material_and_never_exports_exact_pov(
    service_cases: _ServiceCases,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_exact_pov(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("SharedObs source view attempted exact POV export")

    monkeypatch.setattr(
        replay_service_module,
        "export_actor_pov_replay_v1",
        reject_exact_pov,
    )
    service = ReplayViewerService(
        service_cases.shared.bundle,
        view_mode="pov",
        pov_global_slot=5,
        initial_frame_index=1,
        viewer_session_id="shared-source",
    )

    frame = service.current_frame()
    timeline = service.current_timeline()
    payload = json.loads(frame.model_dump_json())

    assert isinstance(frame, SharedObsSourceMaterialReplayViewerFrameV1)
    assert isinstance(timeline, SharedObsSourceMaterialReplayTimelineV1)
    assert frame.observation_materialization == "source_material_only"
    assert timeline.observation_materialization == "source_material_only"
    assert (
        frame.projection.base_sensor_frame.observation_materialization
        == "source_material_only"
    )
    assert frame.projection.base_sensor_frame.public_agent_id == frame.public_agent_id
    assert "pov_frame_id" not in _recursive_keys(payload)
    assert "processing_disclosure" not in _recursive_keys(payload)
    assert service.shared_timeline_build_count == 1


def test_nonzero_focal_actor_is_stable_reference_and_selection_is_independent(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(
        service_cases.nonzero_focal.bundle,
        viewer_session_id="nonzero-focal",
    )
    initial = cast(ResearcherReplayViewerFrameV1, service.current_frame())

    assert initial.projection.scene.selection is not None
    assert initial.projection.scene.selection.controlled_global_slot == 5
    assert initial.projection.scene.selection.selected_global_slot is None
    assert {row.global_slot for row in initial.projection.scene.ranges} == {5}

    selected = _response(
        _apply(
            service,
            ReplaySelectAgentCommandV1(selected_global_slot=0),
            command_id="select-zero",
        )
    ).frame
    assert isinstance(selected, ResearcherReplayViewerFrameV1)
    assert selected.projection.scene.selection is not None
    assert selected.projection.scene.selection.controlled_global_slot == 5
    assert selected.projection.scene.selection.selected_global_slot == 0

    cleared = _response(
        _apply(
            service,
            ReplaySelectAgentCommandV1(selected_global_slot=None),
            command_id="clear-selection",
        )
    ).frame
    assert isinstance(cleared, ResearcherReplayViewerFrameV1)
    assert cleared.projection.scene.selection is not None
    assert cleared.projection.scene.selection.controlled_global_slot == 5
    assert cleared.projection.scene.selection.selected_global_slot is None

    pov = ReplayViewerService(
        service_cases.nonzero_focal.bundle,
        view_mode="pov",
        viewer_session_id="nonzero-focal-pov",
    ).current_frame()
    assert isinstance(pov, ActorPovReplayViewerFrameV1)
    assert pov.pov_global_slot == 5


def test_inactive_actor_requests_fail_closed_without_state_change(
    service_cases: _ServiceCases,
) -> None:
    with pytest.raises(ValueError, match="configured-active"):
        ReplayViewerService(
            service_cases.complete.bundle,
            pov_global_slot=3,
        )
    with pytest.raises(ValueError, match="configured-active"):
        ReplayViewerService(
            service_cases.complete.bundle,
            selected_global_slot=3,
        )

    service = ReplayViewerService(service_cases.complete.bundle)
    before = service.current_frame()
    invalid_pov = _apply(
        service,
        ReplaySetPovActorCommandV1(global_slot=3),
        command_id="inactive-pov",
    )
    invalid_selection = _apply(
        service,
        ReplaySelectAgentCommandV1(selected_global_slot=3),
        command_id="inactive-selection",
    )

    assert invalid_pov.outcome == "audience_unavailable"
    assert invalid_selection.outcome == "audience_unavailable"
    assert _error(invalid_pov).latest_frame == before
    assert _error(invalid_selection).latest_frame == before
    assert service.revision == 0
    assert service.current_frame() == before


def test_cursor_generations_and_forward_choreography_are_exact(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(
        service_cases.complete.bundle,
        viewer_session_id="cursor-rules",
    )

    same = _response(
        _apply(
            service,
            ReplayAbsoluteSeekCommandV1(frame_index=0),
            command_id="same-zero",
        )
    )
    assert same.frame.cursor.frame_index == 0
    assert same.frame.cursor.cursor_generation == 1
    assert same.frame.cursor.choreography_generation == 0
    assert same.animate_incoming is False

    forward = _response(
        _apply(service, ReplayNextFrameCommandV1(), command_id="next-one")
    )
    assert forward.frame.cursor.frame_index == 1
    assert forward.frame.cursor.cursor_generation == 2
    assert forward.frame.cursor.choreography_generation == 1
    assert forward.animate_incoming is True
    assert "animate_incoming" not in json.loads(
        service.current_frame().model_dump_json()
    )

    backward = _response(
        _apply(service, ReplayPreviousFrameCommandV1(), command_id="previous-zero")
    )
    assert backward.frame.cursor.frame_index == 0
    assert backward.frame.cursor.cursor_generation == 3
    assert backward.frame.cursor.choreography_generation == 1
    assert backward.animate_incoming is False

    last = _response(_apply(service, ReplayLastFrameCommandV1(), command_id="last"))
    assert last.frame.cursor.frame_index == 2
    assert last.frame.cursor.cursor_generation == 4
    assert last.frame.cursor.choreography_generation == 1

    repeated_last = _response(
        _apply(service, ReplayLastFrameCommandV1(), command_id="last-again")
    )
    assert repeated_last.frame.cursor.frame_index == 2
    assert repeated_last.frame.cursor.cursor_generation == 5
    assert repeated_last.frame.cursor.choreography_generation == 1

    first = _response(_apply(service, ReplayFirstFrameCommandV1(), command_id="first"))
    assert first.frame.cursor.frame_index == 0
    assert first.frame.cursor.cursor_generation == 6
    assert first.frame.cursor.choreography_generation == 1


def test_cursor_bounds_reject_without_advancing_any_epoch(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(service_cases.complete.bundle)
    initial = service.current_frame()

    before_zero = _apply(
        service,
        ReplayPreviousFrameCommandV1(),
        command_id="before-zero",
    )
    past_prefix = _apply(
        service,
        ReplayAbsoluteSeekCommandV1(frame_index=3),
        command_id="past-prefix",
    )
    _response(_apply(service, ReplayLastFrameCommandV1(), command_id="to-last"))
    final = service.current_frame()
    after_final = _apply(
        service,
        ReplayNextFrameCommandV1(),
        command_id="after-final",
    )

    assert before_zero.outcome == "invalid_cursor"
    assert past_prefix.outcome == "invalid_cursor"
    assert initial.cursor.cursor_generation == 0
    assert final.cursor.cursor_generation == 1
    assert after_final.outcome == "invalid_cursor"
    assert service.current_frame() == final
    assert service.revision == 1


def test_presentation_and_audience_changes_do_not_replay_choreography(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(service_cases.complete.bundle)
    baseline = service.current_frame().cursor
    commands: tuple[ReplayCommandV1, ...] = (
        ReplaySetPresetCommandV1(preset="debug"),
        ReplaySetRangesCommandV1(show_ranges=False),
        ReplaySetVerbosityCommandV1(verbose=True),
        ReplaySetPovActorCommandV1(global_slot=1),
        ReplaySetViewCommandV1(view_mode="pov"),
        ReplaySetViewCommandV1(view_mode="researcher"),
    )

    for index, command in enumerate(commands):
        response = _response(
            _apply(service, command, command_id=f"presentation-{index}")
        )
        assert response.animate_incoming is False
        assert response.frame.cursor.frame_index == baseline.frame_index
        assert response.frame.cursor.cursor_generation == baseline.cursor_generation
        assert (
            response.frame.cursor.choreography_generation
            == baseline.choreography_generation
        )

    no_op = _response(
        _apply(
            service,
            ReplaySetVerbosityCommandV1(verbose=True),
            command_id="same-verbosity",
        )
    )
    assert no_op.result == "no_op"
    assert no_op.frame.revision == service.revision


def test_researcher_only_commands_reject_in_pov_without_active_slot_oracle(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(
        service_cases.complete.bundle,
        view_mode="pov",
        viewer_session_id="pov-command-authority",
    )
    before = service.current_frame()

    active_selection = _apply(
        service,
        ReplaySelectAgentCommandV1(selected_global_slot=0),
        command_id="pov-select-active",
    )
    inactive_selection = _apply(
        service,
        ReplaySelectAgentCommandV1(selected_global_slot=3),
        command_id="pov-select-inactive",
    )
    ranges = _apply(
        service,
        ReplaySetRangesCommandV1(show_ranges=False),
        command_id="pov-ranges",
    )

    for result in (active_selection, inactive_selection, ranges):
        assert result.outcome == "audience_unavailable"
        assert _error(result).latest_frame is before
    assert _error(active_selection).message == _error(inactive_selection).message
    assert service.revision == 0
    assert service.current_frame() is before


def test_lazy_pov_indices_and_timelines_build_once_per_actor(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(service_cases.long.bundle)
    assert service.pov_index_build_count == 0

    _response(
        _apply(
            service,
            ReplaySetPovActorCommandV1(global_slot=1),
            command_id="prepare-pov-one",
        )
    )
    assert service.pov_index_build_count == 0
    _response(
        _apply(
            service,
            ReplaySetViewCommandV1(view_mode="pov"),
            command_id="open-pov-one",
        )
    )
    first_timeline = service.current_timeline()
    assert service.pov_index_build_count == 1
    assert service.current_timeline() is first_timeline
    assert service.pov_index_build_count == 1

    _response(
        _apply(
            service,
            ReplayAbsoluteSeekCommandV1(frame_index=6),
            command_id="seek-pov-one",
        )
    )
    assert service.pov_index_build_count == 1
    _response(
        _apply(
            service,
            ReplaySetPovActorCommandV1(global_slot=5),
            command_id="open-pov-five",
        )
    )
    assert service.pov_index_build_count == 2
    _response(
        _apply(
            service,
            ReplaySetPovActorCommandV1(global_slot=1),
            command_id="return-pov-one",
        )
    )
    assert service.pov_index_build_count == 2


def test_researcher_index_builds_once_and_gets_are_cached(
    service_cases: _ServiceCases,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_calls = 0
    original_status_builder = (
        replay_service_module.build_status_source_evidence_index_v2
    )

    def counted_status_builder(
        context: EvaluationEpisodeContextV1,
        frames: tuple[EvaluationFrameV1, ...],
        transitions: tuple[EvaluationTransitionV1, ...],
    ) -> StatusSourceEvidenceIndexV2:
        nonlocal status_calls
        status_calls += 1
        return original_status_builder(context, frames, transitions)

    monkeypatch.setattr(
        replay_service_module,
        "build_status_source_evidence_index_v2",
        counted_status_builder,
    )
    service = ReplayViewerService(service_cases.long.bundle)
    first_frame = service.current_frame()
    first_timeline = service.current_timeline()

    assert status_calls == 1
    for _ in range(20):
        assert service.current_frame() is first_frame
        assert service.current_timeline() is first_timeline
    assert status_calls == 1
    assert isinstance(first_timeline, ResearcherReplayTimelineV1)
    assert len(first_timeline.rows) == 7


def test_each_seek_constructs_only_its_single_adjacent_view(
    service_cases: _ServiceCases,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_view = replay_service_module.EvaluationTransitionViewV1
    calls: list[int] = []

    def counted_view(
        *,
        context: EvaluationEpisodeContextV1,
        start_frame: EvaluationFrameV1,
        transition: EvaluationTransitionV1,
        successor_frame: EvaluationFrameV1,
    ) -> EvaluationTransitionViewV1:
        calls.append(transition.transition_index)
        return original_view(
            context=context,
            start_frame=start_frame,
            transition=transition,
            successor_frame=successor_frame,
        )

    service = ReplayViewerService(service_cases.long.bundle)
    monkeypatch.setattr(
        replay_service_module,
        "EvaluationTransitionViewV1",
        counted_view,
    )

    _response(
        _apply(
            service,
            ReplayAbsoluteSeekCommandV1(frame_index=6),
            command_id="seek-six",
        )
    )
    _response(
        _apply(
            service,
            ReplayAbsoluteSeekCommandV1(frame_index=3),
            command_id="seek-three",
        )
    )
    _response(
        _apply(
            service,
            ReplayAbsoluteSeekCommandV1(frame_index=0),
            command_id="seek-zero",
        )
    )

    assert calls == [5, 2]


def test_candidate_projection_failure_commits_no_lazy_cache_or_cursor(
    service_cases: _ServiceCases,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ReplayViewerService(
        service_cases.complete.bundle,
        viewer_session_id="atomic-candidate",
    )
    before = service.current_frame()

    def fail_after_pov_cache_build(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("injected actor POV frame failure")

    monkeypatch.setattr(
        replay_service_module,
        "ActorPovReplayViewerFrameV1",
        fail_after_pov_cache_build,
    )
    with pytest.raises(RuntimeError, match="injected actor POV"):
        _apply(
            service,
            ReplaySetViewCommandV1(view_mode="pov"),
            command_id="failing-pov",
        )

    assert service.faulted
    assert service.revision == 0
    assert service.current_frame() is before
    assert service.pov_index_build_count == 0
    faulted = _apply(
        service,
        ReplaySetRangesCommandV1(show_ranges=False),
        command_id="after-fault",
    )
    assert faulted.outcome == "service_faulted"
    assert _error(faulted).latest_frame is before


def test_failed_duplicate_response_does_not_refresh_lru_order(
    service_cases: _ServiceCases,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ReplayViewerService(service_cases.complete.bundle)
    requests = tuple(
        _request(
            service,
            ReplaySetVerbosityCommandV1(verbose=False),
            command_id=f"lru-{index}",
        )
        for index in range(256)
    )
    for request in requests:
        assert _response(service.apply_command(request)).result == "no_op"
    assert service.command_cache_size == 256

    original_response = replay_service_module.ReplayCommandResponseV1

    def fail_duplicate_response(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("injected duplicate response failure")

    monkeypatch.setattr(
        replay_service_module,
        "ReplayCommandResponseV1",
        fail_duplicate_response,
    )
    with pytest.raises(RuntimeError, match="duplicate response"):
        service.apply_command(requests[0])
    monkeypatch.setattr(
        replay_service_module,
        "ReplayCommandResponseV1",
        original_response,
    )

    _response(
        _apply(
            service,
            ReplaySetVerbosityCommandV1(verbose=False),
            command_id="lru-newest",
        )
    )
    replayed_oldest = _response(service.apply_command(requests[0]))

    assert replayed_oldest.result == "no_op"
    assert service.revision == 0


def test_failed_error_response_does_not_remember_command(
    service_cases: _ServiceCases,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ReplayViewerService(service_cases.complete.bundle)
    request = _request(
        service,
        ReplayPreviousFrameCommandV1(),
        command_id="error-materialization",
    )
    original_error = replay_service_module.ReplayApiErrorV1

    def fail_error_response(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("injected error response failure")

    monkeypatch.setattr(
        replay_service_module,
        "ReplayApiErrorV1",
        fail_error_response,
    )
    with pytest.raises(RuntimeError, match="error response"):
        service.apply_command(request)
    assert service.command_cache_size == 0
    assert service.revision == 0
    assert service.faulted is False

    monkeypatch.setattr(
        replay_service_module,
        "ReplayApiErrorV1",
        original_error,
    )
    result = service.apply_command(request)

    assert result.outcome == "invalid_cursor"
    assert service.command_cache_size == 1


def test_duplicate_stale_conflict_exit_and_bounded_idempotency_cache(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(service_cases.complete.bundle)
    request = _request(
        service,
        ReplayNextFrameCommandV1(),
        command_id="next-once",
    )
    applied = _response(service.apply_command(request))
    duplicate = _response(service.apply_command(request))

    assert applied.result == "applied"
    assert duplicate.result == "duplicate"
    assert duplicate.animate_incoming is False
    assert service.revision == 1
    assert service.current_frame().cursor.cursor_generation == 1
    assert service.current_frame().cursor.choreography_generation == 1

    conflict = service.apply_command(
        request.model_copy(update={"command": ReplayFirstFrameCommandV1()})
    )
    assert conflict.outcome == "command_id_conflict"
    stale = _apply(
        service,
        ReplayFirstFrameCommandV1(),
        command_id="stale",
        base_revision=0,
    )
    assert stale.outcome == "stale_revision"
    assert service.revision == 1

    for index in range(300):
        no_op = _apply(
            service,
            ReplaySetVerbosityCommandV1(verbose=False),
            command_id=f"cache-{index}",
        )
        assert _response(no_op).result == "no_op"
    assert service.command_cache_size == 256

    exit_result = _apply(
        service,
        ReplayExitCommandV1(),
        command_id="exit",
    )
    assert _response(exit_result).result == "shutdown_scheduled"
    assert exit_result.shutdown_requested
    assert service.shutting_down
    rejected = _apply(
        service,
        ReplaySetRangesCommandV1(show_ranges=False),
        command_id="after-exit",
    )
    assert rejected.outcome == "server_shutting_down"


def test_identical_concurrent_command_applies_exactly_once(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(service_cases.complete.bundle)
    request = ReplayCommandRequestV1(
        client_id="concurrent-client",
        command_id="same-command",
        base_revision=0,
        command=ReplayNextFrameCommandV1(),
    )
    barrier = Barrier(8)

    def submit() -> ReplayServiceCommandResultV1:
        barrier.wait()
        return service.apply_command(request)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = tuple(executor.submit(submit) for _ in range(8))
        results = tuple(future.result() for future in futures)

    responses = tuple(_response(result) for result in results)
    assert sum(response.result == "applied" for response in responses) == 1
    assert sum(response.result == "duplicate" for response in responses) == 7
    assert service.revision == 1
    assert service.current_frame().cursor.frame_index == 1
    assert service.current_frame().cursor.cursor_generation == 1
    assert service.current_frame().cursor.choreography_generation == 1


def test_simultaneous_clients_serialize_revision_authority(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(service_cases.complete.bundle)
    requests = (
        ReplayCommandRequestV1(
            client_id="client-one",
            command_id="command-one",
            base_revision=0,
            command=ReplayNextFrameCommandV1(),
        ),
        ReplayCommandRequestV1(
            client_id="client-two",
            command_id="command-two",
            base_revision=0,
            command=ReplayLastFrameCommandV1(),
        ),
    )
    barrier = Barrier(2)

    def submit(request: ReplayCommandRequestV1) -> ReplayServiceCommandResultV1:
        barrier.wait()
        return service.apply_command(request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(submit, request) for request in requests)
        results = tuple(future.result() for future in futures)

    assert sorted(result.outcome for result in results) == [
        "response",
        "stale_revision",
    ]
    assert service.revision == 1
    assert service.current_frame().cursor.cursor_generation == 1


def test_outbound_current_frame_is_bounded_and_never_contains_replay_payload(
    service_cases: _ServiceCases,
) -> None:
    short = ReplayViewerService(
        service_cases.complete.bundle,
        viewer_session_id="bounded-short",
    ).current_frame()
    long = ReplayViewerService(
        service_cases.long.bundle,
        viewer_session_id="bounded-long",
    ).current_frame()

    for frame in (short, long):
        payload = json.loads(frame.model_dump_json())
        assert "frames" not in payload
        assert "transitions" not in payload
        assert "metric_report_artifact" not in payload
        assert "context" not in payload
        assert "rows" not in payload
        assert frame.model_dump_json() == frame.model_dump_json()
    assert len(long.model_dump_json()) < 2 * len(short.model_dump_json())
    assert len(long.model_dump_json()) < len(
        canonical_json_bytes(service_cases.long.bundle.replay)
    )


def test_replay_service_import_is_core_jax_policy_and_live_seam_free() -> None:
    code = """
import sys
import scripts.dev.visual_debugger.replay_service
banned = (
    'jax',
    'jaxlib',
    'numpy',
    'marl_battlegrounds.core',
    'marl_battlegrounds.evaluation.capture',
    'marl_battlegrounds.policy',
    'marl_battlegrounds.policies',
    'marl_battlegrounds.runner',
    'marl_battlegrounds.runners',
    'marl_battlegrounds.simulator',
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
