"""Read-only replay-service authority, privacy, and cursor proofs."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any, Literal, cast

import jax
import pytest
import scripts.dev.visual_debugger.control as live_control_module
import scripts.dev.visual_debugger.replay_service as replay_service_module
import scripts.dev.visual_debugger.static_renderer as static_renderer_module
from scripts.dev.visual_debugger.presentation import (
    build_replay_no_shared_obs_authorized_presentation_v1,
    build_replay_shared_obs_authorized_presentation_v1,
)
from scripts.dev.visual_debugger.presentation_protocol import (
    PRESENTATION_AUDIENCE_UNAVAILABLE_MESSAGE,
    PresentationApiErrorV1,
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    ReplayOracleAuthorizedPresentationFrameV1,
    ReplaySharedObsAuthorizedPresentationFrameV1,
)
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
    ReplayCursorV1,
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
    SharedObsAgentPovReplayTimelineV1,
    SharedObsAgentPovReplayViewerFrameV1,
)
from scripts.dev.visual_debugger.replay_service import (
    PresentationResourceResultV1,
    ReplayServiceCommandResultV1,
    ReplayViewerService,
)
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    evaluation_env_config,
)

import marl_battlegrounds.core.env as core_env_module
import marl_battlegrounds.core.geometry as core_geometry_module
import marl_battlegrounds.evaluation.capture as evaluation_capture_module
import marl_battlegrounds.evaluation.events as evaluation_events_module
import marl_battlegrounds.rendering.evaluation_adapter as evaluation_adapter_module
from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v1
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
    JointActionV1,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.pov import export_actor_pov_replay_v1
from marl_battlegrounds.evaluation.replay import (
    ReplayBundleV1,
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.replay_io import (
    LoadedReplayBundleV1,
    load_replay_bundle_v1,
    save_replay_bundle_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    SharedObsSourceMaterialProjectionV1,
    build_shared_obs_authority_source_material_projection_v1,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovAnalyzerProjectionV1,
    build_actor_pov_projection_index_v1,
)
from marl_battlegrounds.rendering.scene import (
    BattlefieldSceneV2,
    StatusSourceEvidenceIndexV2,
    VisualEventBatchV2,
)

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
    noncanonical_movement_scale: _ServiceCase


type _ReplayPresentationFrame = (
    ReplayOracleAuthorizedPresentationFrameV1
    | ReplayNoSharedObsAuthorizedPresentationFrameV1
    | ReplaySharedObsAuthorizedPresentationFrameV1
)


class _PoisonActorPovAnalyzerProjectionV1(ActorPovAnalyzerProjectionV1):
    """Exact-field subtype used to prove the selective projection fence."""


class _PoisonReplayCursorV1(ReplayCursorV1):
    """No-extra subtype used to prove exact cursor ownership."""


class _PoisonActorPovReplayViewerFrameV1(ActorPovReplayViewerFrameV1):
    """No-extra subtype used to prove the exact raw-frame boundary."""


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


def _with_movement_scale(
    trajectory: CapturedEvaluationTrajectory,
    *,
    movement_scale: float,
) -> CapturedEvaluationTrajectory:
    """Replace only recorded experimental movement-scale provenance."""
    experimental_config = evaluation_env_config()._replace(
        ordinary_movement_distance_scale=movement_scale
    )
    context_payload = trajectory.context.model_dump(mode="python")
    context_payload["resolved_env_config"] = build_resolved_env_config_v1(
        experimental_config
    )
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
        transition_count=2,
        expected_horizon=2,
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
    noncanonical_movement_trajectory = _with_movement_scale(
        captured_evaluation_trajectory(
            transition_count=1,
            expected_horizon=1,
            episode_id="service-recorded-movement-scale",
        ),
        movement_scale=0.375,
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
        noncanonical_movement_scale=_loaded_case(
            noncanonical_movement_trajectory,
            completion_state="complete",
        ),
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


def _presentation_response(
    result: PresentationResourceResultV1,
) -> _ReplayPresentationFrame:
    assert result.outcome == "response"
    assert type(result.payload) in {
        ReplayOracleAuthorizedPresentationFrameV1,
        ReplayNoSharedObsAuthorizedPresentationFrameV1,
        ReplaySharedObsAuthorizedPresentationFrameV1,
    }
    return cast(_ReplayPresentationFrame, result.payload)


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


def _recursive_string_values(value: object) -> set[str]:
    if type(value) is str:
        return {value}
    if isinstance(value, dict):
        record = cast(dict[str, object], value)
        return {
            string
            for child in record.values()
            for string in _recursive_string_values(child)
        }
    if isinstance(value, list):
        sequence = cast(list[object], value)
        return {
            string for child in sequence for string in _recursive_string_values(child)
        }
    return set()


def _shared_authority_sources(
    case: _ServiceCase,
    *,
    frame_index: int,
    recipient_global_slot: int,
) -> tuple[
    SharedObsSourceMaterialProjectionV1,
    tuple[SharedObsSourceMaterialProjectionV1, ...],
]:
    context = case.bundle.replay.header.context
    frame = case.bundle.replay.frames[frame_index]

    def build(global_slot: int) -> SharedObsSourceMaterialProjectionV1:
        return build_shared_obs_authority_source_material_projection_v1(
            context,
            frame,
            selected_global_slot=global_slot,
        )

    return (
        build(recipient_global_slot),
        tuple(
            build(row.global_slot)
            for row in context.roster
            if row.configured_active and row.global_slot != recipient_global_slot
        ),
    )


def _shared_presentation_for_raw(
    case: _ServiceCase,
    raw: SharedObsAgentPovReplayViewerFrameV1,
    *,
    recipient_global_slot: int,
) -> ReplaySharedObsAuthorizedPresentationFrameV1:
    frame_index = raw.cursor.frame_index
    current_recipient, current_nonrecipient = _shared_authority_sources(
        case,
        frame_index=frame_index,
        recipient_global_slot=recipient_global_slot,
    )
    if frame_index == 0:
        previous_recipient = None
        previous_nonrecipient: tuple[SharedObsSourceMaterialProjectionV1, ...] = ()
        incoming_transition = None
    else:
        previous_recipient, previous_nonrecipient = _shared_authority_sources(
            case,
            frame_index=frame_index - 1,
            recipient_global_slot=recipient_global_slot,
        )
        incoming_transition = case.bundle.replay.transitions[frame_index - 1]
    outgoing_transition = (
        None
        if frame_index == len(case.bundle.replay.transitions)
        else case.bundle.replay.transitions[frame_index]
    )
    return build_replay_shared_obs_authorized_presentation_v1(
        raw,
        public_catalog=case.bundle.replay.header.context.static_mechanics_catalog,
        source_authority_epoch=raw.revision,
        authorized_recipient_global_slot=recipient_global_slot,
        current_recipient_source_material=current_recipient,
        current_active_nonrecipient_source_material=current_nonrecipient,
        previous_recipient_source_material=previous_recipient,
        previous_active_nonrecipient_source_material=previous_nonrecipient,
        incoming_transition=incoming_transition,
        outgoing_transition=outgoing_transition,
    )


def _presentation_service(
    cases: _ServiceCases,
    audience: Literal["oracle", "no_shared_obs", "shared_obs"],
    *,
    viewer_session_id: str,
    frame_index: int = 1,
) -> ReplayViewerService:
    case = cases.shared if audience == "shared_obs" else cases.complete
    return ReplayViewerService(
        case.bundle,
        initial_frame_index=frame_index,
        view_mode="researcher" if audience == "oracle" else "pov",
        pov_global_slot=0,
        viewer_session_id=viewer_session_id,
    )


def _service_read_only_snapshot(service: ReplayViewerService) -> tuple[object, ...]:
    raw = service.current_frame()
    timeline = cast(SharedObsAgentPovReplayTimelineV1, service.current_timeline())
    command_records = cast(
        dict[object, object],
        object.__getattribute__(service, "_command_records"),
    )
    pov_cache = cast(
        dict[int, object],
        object.__getattribute__(service, "_pov_cache"),
    )
    shared_cache = cast(
        dict[int, object],
        object.__getattribute__(service, "_shared_timeline_cache"),
    )
    return (
        id(raw),
        raw.model_dump_json(warnings=False),
        id(timeline),
        timeline.model_dump_json(),
        service.revision,
        object.__getattribute__(service, "_frame_index"),
        object.__getattribute__(service, "_cursor_generation"),
        object.__getattribute__(service, "_choreography_generation"),
        object.__getattribute__(service, "_view_mode"),
        object.__getattribute__(service, "_inspection_global_slot"),
        object.__getattribute__(service, "_pov_global_slot"),
        object.__getattribute__(service, "_preset"),
        object.__getattribute__(service, "_show_ranges"),
        object.__getattribute__(service, "_verbose"),
        service.shutting_down,
        service.faulted,
        id(command_records),
        tuple(command_records.items()),
        id(pov_cache),
        tuple((key, id(value), value) for key, value in pov_cache.items()),
        id(shared_cache),
        tuple((key, id(value), value) for key, value in shared_cache.items()),
    )


@pytest.mark.parametrize(
    (
        "frame_index",
        "selected_global_slot",
        "expected_incoming_index",
        "expected_outgoing_index",
    ),
    (
        (0, 1, None, 0),
        (1, 1, 0, 1),
        (2, 1, 1, None),
        (1, None, 0, None),
    ),
)
def test_current_presentation_uses_committed_oracle_epochs_only(
    service_cases: _ServiceCases,
    frame_index: int,
    selected_global_slot: int | None,
    expected_incoming_index: int | None,
    expected_outgoing_index: int | None,
) -> None:
    service = ReplayViewerService(
        service_cases.complete.bundle,
        initial_frame_index=frame_index,
        selected_global_slot=selected_global_slot,
        viewer_session_id=f"presentation-epoch-{frame_index}-{selected_global_slot}",
    )
    raw = cast(ResearcherReplayViewerFrameV1, service.current_frame())

    result = service.current_presentation()

    assert result.outcome == "response"
    assert type(result.payload) is ReplayOracleAuthorizedPresentationFrameV1
    presentation = result.payload
    source = presentation.source
    assert source.source_session_id == raw.viewer_session_id
    assert source.source_revision == raw.revision == service.revision
    assert source.source_authority_epoch == raw.revision
    assert source.source_frame_index == raw.cursor.frame_index == frame_index
    assert source.source_final_frame_index == raw.cursor.final_frame_index
    assert source.source_frame_id == raw.frame_id
    assert source.source_simulator_step_count == raw.simulator_step_count
    assert source.source_cursor_generation == raw.cursor.cursor_generation
    assert source.source_choreography_generation == raw.cursor.choreography_generation
    if expected_incoming_index is None:
        assert presentation.incoming_summary is None
    else:
        assert presentation.incoming_summary is not None
        assert (
            presentation.incoming_summary.incoming_transition_index
            == expected_incoming_index
        )
    if expected_outgoing_index is None:
        assert presentation.outgoing_inspection is None
    else:
        assert presentation.outgoing_inspection is not None
        assert (
            presentation.outgoing_inspection.outgoing_transition_index
            == expected_outgoing_index
        )


@pytest.mark.parametrize("audience", ("oracle", "no_shared_obs", "shared_obs"))
@pytest.mark.parametrize("drift", ("session", "revision", "frame", "final"))
def test_presentation_rejects_stale_committed_raw_snapshot_without_mutation(
    service_cases: _ServiceCases,
    audience: Literal["oracle", "no_shared_obs", "shared_obs"],
    drift: Literal["session", "revision", "frame", "final"],
) -> None:
    session_id = f"committed-snapshot-{audience}"
    service = _presentation_service(
        service_cases,
        audience,
        viewer_session_id=session_id,
    )
    raw = service.current_frame()
    if drift == "session":
        donor = _presentation_service(
            service_cases,
            audience,
            viewer_session_id=f"{session_id}-other",
        )
        poisoned = donor.current_frame()
    elif drift == "revision":
        donor = _presentation_service(
            service_cases,
            audience,
            viewer_session_id=session_id,
        )
        if audience == "oracle":
            _response(
                _apply(
                    donor,
                    ReplaySelectAgentCommandV1(selected_global_slot=1),
                    command_id="advance-donor-revision",
                )
            )
        else:
            _response(
                _apply(
                    donor,
                    ReplaySetPovActorCommandV1(global_slot=1),
                    command_id="advance-donor-revision",
                )
            )
        assert donor.revision == 1
        poisoned = donor.current_frame()
    elif drift == "frame":
        donor = _presentation_service(
            service_cases,
            audience,
            viewer_session_id=session_id,
            frame_index=0,
        )
        poisoned = donor.current_frame()
    else:
        poisoned_cursor = raw.cursor.model_copy(
            update={"final_frame_index": raw.cursor.final_frame_index + 1}
        )
        poisoned = raw.model_copy(update={"cursor": poisoned_cursor})
    object.__setattr__(service, "_frame", poisoned)
    before = _service_read_only_snapshot(service)

    with pytest.raises(RuntimeError, match="does not join service state"):
        service.current_presentation()

    assert _service_read_only_snapshot(service) == before


@pytest.mark.parametrize("generation", ("cursor_generation", "choreography_generation"))
def test_oracle_presentation_rejects_stale_committed_generations_without_mutation(
    service_cases: _ServiceCases,
    generation: Literal["cursor_generation", "choreography_generation"],
) -> None:
    service = _presentation_service(
        service_cases,
        "oracle",
        viewer_session_id=f"oracle-stale-{generation}",
    )
    raw = cast(ResearcherReplayViewerFrameV1, service.current_frame())
    cursor = raw.cursor.model_copy(
        update={generation: getattr(raw.cursor, generation) + 1}
    )
    object.__setattr__(service, "_frame", raw.model_copy(update={"cursor": cursor}))
    before = _service_read_only_snapshot(service)

    with pytest.raises(RuntimeError, match="cursor generations are stale"):
        service.current_presentation()

    assert _service_read_only_snapshot(service) == before


@pytest.mark.parametrize(
    "provenance_drift",
    (
        "artifact_summary",
        "timeline_id",
        "movement_value",
        "movement_type",
        "ranges_value",
        "ranges_type",
    ),
)
def test_oracle_presentation_rejects_forged_provenance_without_mutation(
    service_cases: _ServiceCases,
    provenance_drift: Literal[
        "artifact_summary",
        "timeline_id",
        "movement_value",
        "movement_type",
        "ranges_value",
        "ranges_type",
    ],
) -> None:
    service = _presentation_service(
        service_cases,
        "oracle",
        viewer_session_id=f"oracle-provenance-{provenance_drift}",
    )
    raw = cast(ResearcherReplayViewerFrameV1, service.current_frame())
    if provenance_drift == "artifact_summary":
        reference = raw.artifact_summary.replay_reference.model_copy(
            update={
                "trajectory_content_digest_sha256": "f" * 64,
                "canonical_digest_sha256": "e" * 64,
            }
        )
        poison = raw.model_copy(
            update={
                "artifact_summary": raw.artifact_summary.model_copy(
                    update={"replay_reference": reference}
                )
            }
        )
    elif provenance_drift == "timeline_id":
        poison = raw.model_copy(update={"timeline_id": "forged-timeline"})
    elif provenance_drift == "movement_value":
        poison = raw.model_copy(
            update={"recorded_ordinary_movement_distance_scale": 0.777}
        )
    elif provenance_drift == "movement_type":
        poison = raw.model_copy(update={"recorded_ordinary_movement_distance_scale": 1})
    elif provenance_drift == "ranges_value":
        assert raw.show_ranges
        poison = raw.model_copy(update={"show_ranges": False})
    else:
        poison = raw.model_copy(update={"show_ranges": 1})
    object.__setattr__(service, "_frame", poison)
    before = _service_read_only_snapshot(service)

    with pytest.raises(RuntimeError, match="provenance does not join"):
        service.current_presentation()

    assert _service_read_only_snapshot(service) == before


@pytest.mark.parametrize(
    "projection_drift",
    ("selection", "ranges", "reference", "armed_lane"),
)
def test_oracle_presentation_rejects_projection_state_drift_without_mutation(
    service_cases: _ServiceCases,
    projection_drift: Literal["selection", "ranges", "reference", "armed_lane"],
) -> None:
    session_id = f"oracle-projection-{projection_drift}"
    service = ReplayViewerService(
        service_cases.complete.bundle,
        initial_frame_index=1,
        viewer_session_id=session_id,
    )
    if projection_drift == "selection":
        donor = ReplayViewerService(
            service_cases.complete.bundle,
            initial_frame_index=1,
            selected_global_slot=1,
            viewer_session_id=session_id,
        )
    elif projection_drift == "ranges":
        donor = ReplayViewerService(
            service_cases.complete.bundle,
            initial_frame_index=1,
            show_ranges=False,
            viewer_session_id=session_id,
        )
    elif projection_drift == "reference":
        donor = ReplayViewerService(
            service_cases.complete.bundle,
            initial_frame_index=1,
            reference_global_slot=1,
            viewer_session_id=session_id,
        )
    else:
        donor = ReplayViewerService(
            service_cases.complete.bundle,
            initial_frame_index=1,
            selected_global_slot=1,
            armed_lane=1,
            viewer_session_id=session_id,
        )
    raw = cast(ResearcherReplayViewerFrameV1, service.current_frame())
    donor_raw = cast(ResearcherReplayViewerFrameV1, donor.current_frame())
    assert donor_raw.projection != raw.projection
    object.__setattr__(
        service,
        "_frame",
        raw.model_copy(update={"projection": donor_raw.projection}),
    )
    before = _service_read_only_snapshot(service)

    with pytest.raises(RuntimeError, match="provenance does not join"):
        service.current_presentation()

    assert _service_read_only_snapshot(service) == before


@pytest.mark.parametrize("audience", ("no_shared_obs", "shared_obs"))
def test_agent_presentation_ignores_committed_cursor_generations(
    service_cases: _ServiceCases,
    audience: Literal["no_shared_obs", "shared_obs"],
) -> None:
    service = _presentation_service(
        service_cases,
        audience,
        viewer_session_id=f"agent-diagnostic-generations-{audience}",
    )
    baseline = service.current_presentation()
    raw = service.current_frame()
    cursor = raw.cursor.model_copy(
        update={
            "cursor_generation": raw.cursor.cursor_generation + 17,
            "choreography_generation": raw.cursor.choreography_generation + 9,
        }
    )
    object.__setattr__(service, "_frame", raw.model_copy(update={"cursor": cursor}))
    before = _service_read_only_snapshot(service)

    mutated = service.current_presentation()

    assert mutated == baseline
    assert mutated.payload.model_dump_json() == baseline.payload.model_dump_json()
    assert _service_read_only_snapshot(service) == before


def test_current_presentation_is_read_only_and_does_not_stale_a_command(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(
        service_cases.complete.bundle,
        selected_global_slot=1,
        viewer_session_id="presentation-read-only",
    )
    raw = service.current_frame()
    timeline = service.current_timeline()
    raw_bytes = raw.model_dump_json()
    timeline_bytes = timeline.model_dump_json()
    revision = service.revision
    prepared_command = _request(
        service,
        ReplayNextFrameCommandV1(),
        command_id="prepared-before-presentation-get",
    )

    first = service.current_presentation()
    second = service.current_presentation()

    assert first == second
    assert service.current_frame() is raw
    assert service.current_frame().model_dump_json() == raw_bytes
    assert service.current_timeline() is timeline
    assert service.current_timeline().model_dump_json() == timeline_bytes
    assert service.revision == revision == 0
    assert service.command_cache_size == 0
    assert service.pov_index_build_count == 0
    assert service.shared_timeline_build_count == 0

    response = _response(service.apply_command(prepared_command))
    assert response.result == "applied"
    assert response.frame.revision == 1
    assert response.frame.cursor.frame_index == 1
    assert service.command_cache_size == 1


@pytest.mark.parametrize("audience", ("oracle", "no_shared_obs", "shared_obs"))
def test_all_presentation_getters_are_fully_read_only_and_keep_prepared_command(
    service_cases: _ServiceCases,
    audience: Literal["oracle", "no_shared_obs", "shared_obs"],
) -> None:
    service = _presentation_service(
        service_cases,
        audience,
        viewer_session_id=f"read-only-all-{audience}",
        frame_index=0,
    )
    before = _service_read_only_snapshot(service)
    prepared = _request(
        service,
        ReplayNextFrameCommandV1(),
        command_id=f"prepared-{audience}",
    )

    first = service.current_presentation()
    middle = _service_read_only_snapshot(service)
    second = service.current_presentation()

    assert first == second
    assert first.payload.model_dump_json() == second.payload.model_dump_json()
    assert before == middle == _service_read_only_snapshot(service)
    response = _response(service.apply_command(prepared))
    assert response.result == "applied"
    assert response.frame.revision == 1
    assert response.frame.cursor.frame_index == 1


@pytest.mark.parametrize("audience", ("oracle", "no_shared_obs", "shared_obs"))
def test_command_between_presentation_gets_yields_two_coherent_roots(
    service_cases: _ServiceCases,
    audience: Literal["oracle", "no_shared_obs", "shared_obs"],
) -> None:
    service = _presentation_service(
        service_cases,
        audience,
        viewer_session_id=f"between-get-{audience}",
        frame_index=0,
    )

    before = service.current_presentation()
    response = _response(
        _apply(
            service,
            ReplayNextFrameCommandV1(),
            command_id=f"between-get-next-{audience}",
        )
    )
    after = service.current_presentation()

    assert before.outcome == after.outcome == "response"
    before_frame = _presentation_response(before)
    after_frame = _presentation_response(after)
    assert before_frame.source.source_revision == 0
    assert before_frame.source.source_frame_index == 0
    assert after_frame.source.source_revision == response.frame.revision == 1
    assert (
        after_frame.source.source_frame_index == response.frame.cursor.frame_index == 1
    )
    assert before_frame.latest_events is None
    assert before_frame.latest_transition is None
    assert after_frame.latest_events is not None
    assert after_frame.latest_transition is not None
    assert after_frame.latest_transition.incoming_transition_index == 0
    if audience == "oracle":
        assert type(after_frame) is ReplayOracleAuthorizedPresentationFrameV1
    elif audience == "no_shared_obs":
        assert type(after_frame) is ReplayNoSharedObsAuthorizedPresentationFrameV1
    else:
        assert type(after_frame) is ReplaySharedObsAuthorizedPresentationFrameV1


@pytest.mark.parametrize("audience", ("oracle", "no_shared_obs", "shared_obs"))
def test_concurrent_presentation_get_and_command_never_mix_epochs(
    service_cases: _ServiceCases,
    audience: Literal["oracle", "no_shared_obs", "shared_obs"],
) -> None:
    service = _presentation_service(
        service_cases,
        audience,
        viewer_session_id=f"concurrent-get-{audience}",
        frame_index=0,
    )
    request = _request(
        service,
        ReplayNextFrameCommandV1(),
        command_id=f"concurrent-next-{audience}",
    )
    barrier = Barrier(2)

    def get_presentation() -> PresentationResourceResultV1:
        barrier.wait()
        return service.current_presentation()

    def apply_next() -> ReplayServiceCommandResultV1:
        barrier.wait()
        return service.apply_command(request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        get_future = executor.submit(get_presentation)
        command_future = executor.submit(apply_next)
        raced = get_future.result()
        command = command_future.result()

    assert command.outcome == "response"
    command_frame = _response(command).frame
    assert command_frame.revision == 1
    assert command_frame.cursor.frame_index == 1
    assert raced.outcome == "response"
    raced_frame = _presentation_response(raced)
    raced_pair = (
        raced_frame.source.source_revision,
        raced_frame.source.source_frame_index,
    )
    assert raced_pair in {(0, 0), (1, 1)}
    current = _presentation_response(service.current_presentation())
    assert (
        current.source.source_revision,
        current.source.source_frame_index,
    ) == (1, 1)


@pytest.mark.parametrize(
    ("audience", "expected_builder"),
    (
        ("oracle", "build_replay_oracle_authorized_presentation_v1"),
        ("no_shared_obs", "build_replay_no_shared_obs_authorized_presentation_v1"),
        ("shared_obs", "build_replay_shared_obs_authorized_presentation_v1"),
    ),
)
def test_presentation_dispatch_calls_only_the_exact_authority_packager(
    service_cases: _ServiceCases,
    monkeypatch: pytest.MonkeyPatch,
    audience: Literal["oracle", "no_shared_obs", "shared_obs"],
    expected_builder: str,
) -> None:
    service = _presentation_service(
        service_cases,
        audience,
        viewer_session_id=f"exact-dispatch-{audience}",
    )
    calls: list[str] = []

    class DispatchProbeError(RuntimeError):
        pass

    def probe(name: str) -> Callable[..., object]:
        def fail(*args: object, **kwargs: object) -> object:
            del args, kwargs
            calls.append(name)
            raise DispatchProbeError(name)

        return fail

    for name in (
        "build_replay_oracle_authorized_presentation_v1",
        "build_replay_no_shared_obs_authorized_presentation_v1",
        "build_replay_shared_obs_authorized_presentation_v1",
    ):
        monkeypatch.setattr(replay_service_module, name, probe(name))

    with pytest.raises(DispatchProbeError, match=expected_builder):
        service.current_presentation()

    assert calls == [expected_builder]


@pytest.mark.parametrize("audience", ("oracle", "no_shared_obs", "shared_obs"))
@pytest.mark.parametrize("state_field", ("_shutting_down", "_faulted"))
def test_presentation_get_preserves_existing_terminal_service_state(
    service_cases: _ServiceCases,
    audience: Literal["oracle", "no_shared_obs", "shared_obs"],
    state_field: Literal["_shutting_down", "_faulted"],
) -> None:
    service = _presentation_service(
        service_cases,
        audience,
        viewer_session_id=f"terminal-state-{audience}-{state_field}",
    )
    object.__setattr__(service, state_field, True)
    before = _service_read_only_snapshot(service)

    result = service.current_presentation()

    assert result.outcome == "response"
    assert _service_read_only_snapshot(service) == before


@pytest.mark.parametrize(
    ("audience", "donor_audience", "error_fragment"),
    (
        ("oracle", "no_shared_obs", "Oracle replay view"),
        ("no_shared_obs", "shared_obs", "NoSharedObs replay view"),
        ("shared_obs", "oracle", "SharedObs replay view"),
    ),
)
def test_presentation_rejects_wrong_committed_product_root_without_fallback(
    service_cases: _ServiceCases,
    audience: Literal["oracle", "no_shared_obs", "shared_obs"],
    donor_audience: Literal["oracle", "no_shared_obs", "shared_obs"],
    error_fragment: str,
) -> None:
    session_id = f"wrong-product-{audience}"
    service = _presentation_service(
        service_cases,
        audience,
        viewer_session_id=session_id,
    )
    donor = _presentation_service(
        service_cases,
        donor_audience,
        viewer_session_id=session_id,
    )
    object.__setattr__(service, "_frame", donor.current_frame())
    before = _service_read_only_snapshot(service)

    with pytest.raises(RuntimeError, match=error_fragment):
        service.current_presentation()

    assert _service_read_only_snapshot(service) == before


@pytest.mark.parametrize("frame_index", (0, 1, 2))
@pytest.mark.parametrize("pov_global_slot", (0, 5))
def test_current_presentation_no_shared_uses_fixed_recipient_epochs(
    service_cases: _ServiceCases,
    frame_index: int,
    pov_global_slot: int,
) -> None:
    service = ReplayViewerService(
        service_cases.complete.bundle,
        initial_frame_index=frame_index,
        view_mode="pov",
        pov_global_slot=pov_global_slot,
        selected_global_slot=(1 if pov_global_slot == 0 else 6),
        viewer_session_id=f"presentation-no-shared-{frame_index}-{pov_global_slot}",
    )
    raw = cast(ActorPovReplayViewerFrameV1, service.current_frame())
    raw_bytes = raw.model_dump_json()
    cache_count = service.pov_index_build_count

    first = service.current_presentation()
    second = service.current_presentation()

    assert first.outcome == second.outcome == "response"
    assert type(first.payload) is ReplayNoSharedObsAuthorizedPresentationFrameV1
    assert first.payload == second.payload
    assert first.payload.model_dump_json() == second.payload.model_dump_json()
    presentation = first.payload
    assert presentation.source.source_revision == raw.revision == service.revision
    assert presentation.source.source_authority_epoch == raw.revision
    assert presentation.source.source_frame_index == frame_index
    assert presentation.source.source_final_frame_index == 2
    assert presentation.source.source_recipient_public_agent_id == raw.public_agent_id
    assert presentation.current_endpoint.parts.recipient_public_agent_id == (
        raw.public_agent_id
    )
    assert presentation.current_endpoint.action_axis.owner_public_agent_id == (
        raw.public_agent_id
    )
    mask = presentation.current_endpoint.parts.next_decision_action_mask
    assert len(mask.move) == 9
    assert len(mask.select_target) == 11
    assert len(mask.use_ultimate) == 2
    assert len(mask.select_target_use_ultimate_joint) == 11
    assert all(len(row) == 2 for row in mask.select_target_use_ultimate_joint)
    if frame_index == 0:
        assert presentation.latest_events is None
        assert presentation.latest_transition is None
    else:
        assert presentation.latest_events is not None
        assert presentation.latest_transition is not None
        assert presentation.latest_events.incoming_transition_index == frame_index - 1
        assert (
            presentation.latest_transition.incoming_transition_index == frame_index - 1
        )
    if frame_index == 2:
        assert presentation.replay_inspection is None
    else:
        assert presentation.replay_inspection is not None
        assert presentation.replay_inspection.outgoing_transition_index == frame_index
    assert service.current_frame() is raw
    assert service.current_frame().model_dump_json() == raw_bytes
    assert service.pov_index_build_count == cache_count == 1
    assert service.command_cache_size == 0


def test_no_shared_packager_ignores_every_forbidden_raw_diagnostic_branch(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(
        service_cases.complete.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="no-shared-selective-raw",
    )
    raw = cast(ActorPovReplayViewerFrameV1, service.current_frame())
    content = export_actor_pov_replay_v1(
        service_cases.complete.bundle.replay,
        global_slot=0,
    ).content
    index = build_actor_pov_projection_index_v1(content)
    baseline = build_replay_no_shared_obs_authorized_presentation_v1(
        index,
        raw,
        public_catalog=(
            service_cases.complete.bundle.replay.header.context.static_mechanics_catalog
        ),
        source_authority_epoch=raw.revision,
    )
    diagnostic_cursor = raw.cursor.model_copy(
        update={
            "cursor_generation": raw.cursor.cursor_generation + 17,
            "choreography_generation": raw.cursor.choreography_generation + 9,
        }
    )
    diagnostic_raw = raw.model_copy(
        update={
            "artifact_summary": SimpleNamespace(forbidden="artifact"),
            "timeline_id": "forbidden-diagnostic-timeline",
            "cursor": diagnostic_cursor,
            "preset": "technical",
            "verbose": True,
            "pov_global_slot": 9,
            "completion": SimpleNamespace(forbidden="completion"),
            "processing_disclosure": SimpleNamespace(forbidden="processing"),
        }
    )

    mutated = build_replay_no_shared_obs_authorized_presentation_v1(
        index,
        diagnostic_raw,
        public_catalog=(
            service_cases.complete.bundle.replay.header.context.static_mechanics_catalog
        ),
        source_authority_epoch=raw.revision,
    )

    assert mutated == baseline
    assert mutated.model_dump_json() == baseline.model_dump_json()


@pytest.mark.parametrize(
    ("field_name", "poison"),
    (
        ("schema_version", True),
        ("frame_kind", "researcher_replay_viewer"),
        ("view_mode", "researcher"),
        ("viewer_session_id", ""),
        ("revision", True),
        ("public_agent_id", ""),
        ("pov_frame_id", ""),
        ("pov_frame_id", "wrong-local-frame"),
        ("simulator_step_count", True),
        ("simulator_step_count", 999),
    ),
)
def test_no_shared_packager_rejects_poisoned_used_raw_headers(
    service_cases: _ServiceCases,
    field_name: str,
    poison: object,
) -> None:
    service = ReplayViewerService(
        service_cases.complete.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="no-shared-used-header",
    )
    raw = cast(ActorPovReplayViewerFrameV1, service.current_frame())
    index = build_actor_pov_projection_index_v1(
        export_actor_pov_replay_v1(
            service_cases.complete.bundle.replay,
            global_slot=0,
        ).content
    )
    poisoned = raw.model_copy(update={field_name: poison})

    with pytest.raises((TypeError, ValueError)):
        build_replay_no_shared_obs_authorized_presentation_v1(
            index,
            poisoned,
            public_catalog=(
                service_cases.complete.bundle.replay.header.context.static_mechanics_catalog
            ),
            source_authority_epoch=raw.revision,
        )


def test_no_shared_packager_rejects_cursor_and_projection_runtime_poisons(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(
        service_cases.complete.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="no-shared-runtime-poison",
    )
    raw = cast(ActorPovReplayViewerFrameV1, service.current_frame())
    index = build_actor_pov_projection_index_v1(
        export_actor_pov_replay_v1(
            service_cases.complete.bundle.replay,
            global_slot=0,
        ).content
    )
    cursor_bool = cast(Any, raw.cursor).model_construct(
        **{
            **raw.cursor.model_dump(mode="python"),
            "frame_index": True,
        }
    )
    cursor_final_bool = cast(Any, raw.cursor).model_construct(
        **{
            **raw.cursor.model_dump(mode="python"),
            "final_frame_index": True,
        }
    )
    cursor_subclass = _PoisonReplayCursorV1.model_validate_json(
        raw.cursor.model_dump_json()
    )
    projection = raw.projection
    projection_subclass = _PoisonActorPovAnalyzerProjectionV1(
        scene=projection.scene,
        next_decision_action_mask=projection.next_decision_action_mask,
        incoming_transition_id=projection.incoming_transition_id,
        incoming_cues=projection.incoming_cues,
    )
    poisoned_roots = (
        raw.model_copy(update={"cursor": cursor_bool}),
        raw.model_copy(update={"cursor": cursor_final_bool}),
        raw.model_copy(update={"cursor": cursor_subclass}),
        raw.model_copy(update={"projection": projection_subclass}),
        raw.model_copy(update={"projection": SimpleNamespace()}),
        _PoisonActorPovReplayViewerFrameV1.model_validate_json(raw.model_dump_json()),
    )

    for poisoned in poisoned_roots:
        with pytest.raises((TypeError, ValueError)):
            build_replay_no_shared_obs_authorized_presentation_v1(
                index,
                poisoned,
                public_catalog=(
                    service_cases.complete.bundle.replay.header.context.static_mechanics_catalog
                ),
                source_authority_epoch=raw.revision,
            )


def test_no_shared_presentation_cache_miss_fails_closed_without_get_mutation(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(
        service_cases.complete.bundle,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="no-shared-cache-miss",
    )
    raw = service.current_frame()
    raw_bytes = raw.model_dump_json()
    cache = cast(
        dict[int, object],
        object.__getattribute__(service, "_pov_cache"),
    )
    cache.clear()

    with pytest.raises(RuntimeError, match="no POV cache entry"):
        service.current_presentation()

    assert service.current_frame() is raw
    assert service.current_frame().model_dump_json() == raw_bytes
    assert service.pov_index_build_count == 0
    assert service.revision == 0
    assert service.command_cache_size == 0


def test_no_shared_presentation_rejects_cross_recipient_cache_entry(
    service_cases: _ServiceCases,
) -> None:
    recipient_zero = ReplayViewerService(
        service_cases.complete.bundle,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="no-shared-cache-recipient-zero",
    )
    recipient_five = ReplayViewerService(
        service_cases.complete.bundle,
        view_mode="pov",
        pov_global_slot=5,
        viewer_session_id="no-shared-cache-recipient-five",
    )
    cache_zero = cast(
        dict[int, object],
        object.__getattribute__(recipient_zero, "_pov_cache"),
    )
    cache_five = cast(
        dict[int, object],
        object.__getattribute__(recipient_five, "_pov_cache"),
    )
    cache_zero[0] = cache_five[5]

    with pytest.raises(RuntimeError, match="does not join the fixed POV recipient"):
        recipient_zero.current_presentation()

    assert cache_zero[0] is cache_five[5]
    assert recipient_zero.pov_index_build_count == 1
    assert recipient_zero.revision == 0


def test_agent_presentation_is_independent_of_researcher_inspection_state(
    service_cases: _ServiceCases,
) -> None:
    services = tuple(
        ReplayViewerService(
            service_cases.complete.bundle,
            view_mode="pov",
            selected_global_slot=inspection_slot,
            pov_global_slot=0,
            viewer_session_id="fixed-pov-inspection-independence",
        )
        for inspection_slot in (1, 6)
    )

    raw_frames = tuple(service.current_frame() for service in services)
    presentations = tuple(service.current_presentation() for service in services)

    assert raw_frames[0].model_dump_json() == raw_frames[1].model_dump_json()
    assert presentations[0].outcome == presentations[1].outcome == "response"
    assert presentations[0].payload.model_dump_json() == (
        presentations[1].payload.model_dump_json()
    )


@pytest.mark.parametrize("frame_index", (0, 1, 2))
@pytest.mark.parametrize("pov_global_slot", (0, 5))
def test_current_presentation_shared_uses_fixed_recipient_epochs(
    service_cases: _ServiceCases,
    frame_index: int,
    pov_global_slot: int,
) -> None:
    service = ReplayViewerService(
        service_cases.shared.bundle,
        initial_frame_index=frame_index,
        view_mode="pov",
        pov_global_slot=pov_global_slot,
        selected_global_slot=(1 if pov_global_slot == 0 else 6),
        viewer_session_id=f"presentation-shared-{frame_index}-{pov_global_slot}",
    )
    raw = cast(SharedObsAgentPovReplayViewerFrameV1, service.current_frame())
    raw_bytes = raw.model_dump_json()
    timeline = cast(SharedObsAgentPovReplayTimelineV1, service.current_timeline())
    timeline_bytes = timeline.model_dump_json()
    cache_counts = (
        service.pov_index_build_count,
        service.shared_timeline_build_count,
    )

    first = service.current_presentation()
    second = service.current_presentation()

    assert first.outcome == second.outcome == "response"
    assert type(first.payload) is ReplaySharedObsAuthorizedPresentationFrameV1
    assert first.payload == second.payload
    assert first.payload.model_dump_json() == second.payload.model_dump_json()
    presentation = first.payload
    prefix = (
        f"{raw.artifact_summary.episode_id}:shared-obs-visual-union:"
        f"{raw.public_agent_id}"
    )
    expected_incoming_id = (
        None if frame_index == 0 else f"{prefix}:transition:{frame_index - 1}"
    )
    assert raw.frame_kind == "shared_obs_agent_pov_replay_viewer"
    assert raw.artifact_summary.recipient_replay_id == f"{prefix}:replay"
    assert raw.timeline_id == timeline.timeline_id == f"{prefix}:timeline"
    assert raw.recipient_frame_id == f"{prefix}:frame:{frame_index}"
    assert raw.incoming_recipient_transition_id == expected_incoming_id
    assert timeline.timeline_kind == "shared_obs_agent_pov"
    assert timeline.artifact_summary == raw.artifact_summary
    assert timeline.completion == raw.completion
    assert timeline.rows[frame_index].recipient_frame_id == raw.recipient_frame_id
    assert (
        timeline.rows[frame_index].incoming_recipient_transition_id
        == expected_incoming_id
    )
    assert presentation.source.source_revision == raw.revision == service.revision
    assert presentation.source.source_authority_epoch == raw.revision
    assert presentation.source.source_frame_index == frame_index
    assert presentation.source.source_final_frame_index == 2
    assert presentation.source.source_recipient_public_agent_id == raw.public_agent_id
    assert presentation.current_endpoint.parts.recipient_public_agent_id == (
        raw.public_agent_id
    )
    assert presentation.current_endpoint.action_axis.owner_public_agent_id == (
        raw.public_agent_id
    )
    assert presentation.authority.exact_actor_input_export_available is False
    mask = presentation.current_endpoint.parts.next_decision_action_mask
    assert len(mask.move) == 9
    assert len(mask.select_target) == 11
    assert len(mask.use_ultimate) == 2
    assert len(mask.select_target_use_ultimate_joint) == 11
    assert all(len(row) == 2 for row in mask.select_target_use_ultimate_joint)
    if frame_index == 0:
        assert presentation.latest_events is None
        assert presentation.latest_transition is None
    else:
        assert presentation.latest_events is not None
        assert presentation.latest_transition is not None
        assert presentation.latest_events.incoming_transition_index == frame_index - 1
        assert (
            presentation.latest_transition.incoming_transition_index == frame_index - 1
        )
    if frame_index == 2:
        assert presentation.replay_inspection is None
    else:
        assert presentation.replay_inspection is not None
        assert presentation.replay_inspection.outgoing_transition_index == frame_index
        assert presentation.replay_inspection.actor_public_agent_id == (
            raw.public_agent_id
        )
    assert service.current_frame() is raw
    assert service.current_frame().model_dump_json() == raw_bytes
    assert service.current_timeline() is timeline
    assert service.current_timeline().model_dump_json() == timeline_bytes
    assert (
        (
            service.pov_index_build_count,
            service.shared_timeline_build_count,
        )
        == cache_counts
        == (0, 1)
    )
    assert service.command_cache_size == 0


def test_shared_raw_product_contains_only_private_transport_identity(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(
        service_cases.shared.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="shared-selective-raw",
    )
    raw = cast(SharedObsAgentPovReplayViewerFrameV1, service.current_frame())
    timeline = cast(SharedObsAgentPovReplayTimelineV1, service.current_timeline())
    frame_payload = json.loads(raw.model_dump_json())
    timeline_payload = json.loads(timeline.model_dump_json())
    forbidden = {
        "replay_reference",
        "artifact_id",
        "metric_report_availability",
        "processing",
        "processing_disclosure",
        "selected_global_slot",
        "pov_global_slot",
        "global_slot",
        "source_material_frame_id",
        "source_frame_id",
        "observation_materialization",
        "projection",
        "scene",
        "action_mask",
        "events",
        "reward",
    }

    assert _recursive_keys(frame_payload).isdisjoint(forbidden)
    assert _recursive_keys(timeline_payload).isdisjoint(forbidden)
    all_keys = _recursive_keys(frame_payload) | _recursive_keys(timeline_payload)
    for key in all_keys:
        assert "global_slot" not in key
        assert "digest" not in key
        assert "metric" not in key
        assert "processing" not in key
        assert "reward" not in key
        assert "event" not in key
    forbidden_values = {
        raw.artifact_summary.episode_id + ":replay",
        *(frame.frame_id for frame in service_cases.shared.bundle.replay.frames),
        *(
            transition.transition_id
            for transition in service_cases.shared.bundle.replay.transitions
        ),
        service_cases.shared.bundle.replay.header.context_digest_sha256,
        service_cases.shared.bundle.replay.trajectory_content_digest_sha256,
        service_cases.shared.bundle.replay.canonical_digest_sha256,
    }
    all_strings = _recursive_string_values(frame_payload) | _recursive_string_values(
        timeline_payload
    )
    assert not (all_strings & forbidden_values)
    assert not any(":shared-obs-source-material:" in value for value in all_strings)
    assert raw.frame_kind == "shared_obs_agent_pov_replay_viewer"
    assert timeline.timeline_kind == "shared_obs_agent_pov"
    assert raw.artifact_summary == timeline.artifact_summary
    assert raw.public_agent_id == raw.artifact_summary.public_agent_id
    assert raw.recipient_frame_id.endswith(":frame:1")
    assert raw.incoming_recipient_transition_id is not None
    assert raw.completion.public_end_or_failure_reason is None
    assert service.current_presentation().outcome == "response"


def test_shared_command_and_error_envelopes_return_only_the_private_root(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(
        service_cases.shared.bundle,
        initial_frame_index=0,
        view_mode="pov",
        pov_global_slot=5,
        viewer_session_id="shared-private-command-envelope",
    )
    request = _request(
        service,
        ReplayNextFrameCommandV1(),
        command_id="shared-private-next",
    )

    applied = _response(service.apply_command(request))
    duplicate = _response(service.apply_command(request))
    stale = _error(
        _apply(
            service,
            ReplayNextFrameCommandV1(),
            command_id="shared-private-stale",
            base_revision=0,
        )
    )

    assert type(applied.frame) is SharedObsAgentPovReplayViewerFrameV1
    assert type(duplicate.frame) is SharedObsAgentPovReplayViewerFrameV1
    assert type(stale.latest_frame) is SharedObsAgentPovReplayViewerFrameV1
    assert duplicate.frame == applied.frame
    assert stale.latest_frame == applied.frame
    for envelope in (applied, duplicate, stale):
        payload = json.loads(envelope.model_dump_json())
        keys = _recursive_keys(payload)
        assert "projection" not in keys
        assert "replay_reference" not in keys
        assert "selected_global_slot" not in keys
        assert "global_slot" not in keys
        assert "processing" not in keys


def test_shared_raw_bytes_ignore_metric_and_processing_truth(
    service_cases: _ServiceCases,
) -> None:
    case = service_cases.shared
    missing_metric = LoadedReplayBundleV1(
        replay=case.bundle.replay,
        metric_report_artifact=None,
        status="metric_report_missing",
    )
    failure = service_cases.processing_failed.report.processing_status.failure
    assert failure is not None
    failed_processing = EvaluationProcessingStatusV1(
        status="failed",
        processed_transition_count=len(case.bundle.replay.transitions),
        failure=failure,
    )
    failed_bundle = LoadedReplayBundleV1(
        replay=case.bundle.replay.model_copy(
            update={"processing_status": failed_processing}
        ),
        metric_report_artifact=case.bundle.metric_report_artifact,
        status=case.bundle.status,
    )
    services = tuple(
        ReplayViewerService(
            bundle,
            initial_frame_index=1,
            view_mode="pov",
            pov_global_slot=0,
            viewer_session_id="shared-hidden-metadata-noninterference",
        )
        for bundle in (case.bundle, missing_metric, failed_bundle)
    )

    assert len({service.current_frame().model_dump_json() for service in services}) == 1
    assert (
        len({service.current_timeline().model_dump_json() for service in services}) == 1
    )
    assert (
        len(
            {
                service.current_presentation().payload.model_dump_json()
                for service in services
            }
        )
        == 1
    )


def test_shared_packager_binds_private_raw_identity_to_authority_sources(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(
        service_cases.shared.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="shared-shadow-join",
    )
    raw = cast(SharedObsAgentPovReplayViewerFrameV1, service.current_frame())
    other_recipient = ReplayViewerService(
        service_cases.shared.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=5,
        viewer_session_id="shared-shadow-join",
    )
    other_raw = cast(
        SharedObsAgentPovReplayViewerFrameV1,
        other_recipient.current_frame(),
    )
    with pytest.raises(ValueError, match="does not join authorized"):
        _shared_presentation_for_raw(
            service_cases.shared,
            other_raw,
            recipient_global_slot=0,
        )

    current_zero, nonrecipient_zero = _shared_authority_sources(
        service_cases.shared,
        frame_index=0,
        recipient_global_slot=0,
    )
    with pytest.raises(ValueError, match="does not join authorized"):
        build_replay_shared_obs_authorized_presentation_v1(
            raw,
            public_catalog=(
                service_cases.shared.bundle.replay.header.context.static_mechanics_catalog
            ),
            source_authority_epoch=raw.revision,
            authorized_recipient_global_slot=0,
            current_recipient_source_material=current_zero,
            current_active_nonrecipient_source_material=nonrecipient_zero,
            previous_recipient_source_material=current_zero,
            previous_active_nonrecipient_source_material=nonrecipient_zero,
            incoming_transition=service_cases.shared.bundle.replay.transitions[0],
            outgoing_transition=service_cases.shared.bundle.replay.transitions[1],
        )

    object.__setattr__(service, "_frame", other_raw)
    before = _service_read_only_snapshot(service)
    with pytest.raises(RuntimeError, match="transport identity"):
        service.current_presentation()
    assert _service_read_only_snapshot(service) == before


@pytest.mark.parametrize(
    ("field_name", "poison"),
    (
        ("schema_version", True),
        ("frame_kind", "actor_pov_replay_viewer"),
        ("view_mode", "researcher"),
        ("preset", "technical"),
        ("verbose", True),
        ("viewer_session_id", ""),
        ("revision", True),
        ("public_agent_id", ""),
        ("recipient_frame_id", ""),
        ("recipient_frame_id", "wrong-local-frame"),
        ("timeline_id", "wrong-local-timeline"),
        ("incoming_recipient_transition_id", "wrong-local-transition"),
        ("artifact_summary", SimpleNamespace(forbidden="artifact")),
        ("completion", SimpleNamespace(forbidden="completion")),
        ("simulator_step_count", True),
        ("simulator_step_count", 999),
    ),
)
def test_shared_packager_rejects_poisoned_used_raw_headers(
    service_cases: _ServiceCases,
    field_name: str,
    poison: object,
) -> None:
    service = ReplayViewerService(
        service_cases.shared.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="shared-used-header",
    )
    raw = cast(SharedObsAgentPovReplayViewerFrameV1, service.current_frame())
    poisoned = raw.model_copy(update={field_name: poison})

    with pytest.raises((TypeError, ValueError)):
        _shared_presentation_for_raw(
            service_cases.shared,
            poisoned,
            recipient_global_slot=0,
        )


@pytest.mark.parametrize("schema_version", (True, 2))
def test_agent_packagers_reject_poisoned_cursor_schema_version(
    service_cases: _ServiceCases,
    schema_version: object,
) -> None:
    no_shared_service = ReplayViewerService(
        service_cases.complete.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="agent-cursor-version-no-shared",
    )
    no_shared_raw = cast(
        ActorPovReplayViewerFrameV1,
        no_shared_service.current_frame(),
    )
    no_shared_index = build_actor_pov_projection_index_v1(
        export_actor_pov_replay_v1(
            service_cases.complete.bundle.replay,
            global_slot=0,
        ).content
    )
    no_shared_cursor = no_shared_raw.cursor.model_copy(
        update={"schema_version": schema_version}
    )
    with pytest.raises(ValueError, match="wire identity"):
        build_replay_no_shared_obs_authorized_presentation_v1(
            no_shared_index,
            no_shared_raw.model_copy(update={"cursor": no_shared_cursor}),
            public_catalog=(
                service_cases.complete.bundle.replay.header.context.static_mechanics_catalog
            ),
            source_authority_epoch=no_shared_raw.revision,
        )

    shared_service = ReplayViewerService(
        service_cases.shared.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="agent-cursor-version-shared",
    )
    shared_raw = cast(
        SharedObsAgentPovReplayViewerFrameV1,
        shared_service.current_frame(),
    )
    shared_cursor = shared_raw.cursor.model_copy(
        update={"schema_version": schema_version}
    )
    with pytest.raises(ValueError, match="wire identity"):
        _shared_presentation_for_raw(
            service_cases.shared,
            shared_raw.model_copy(update={"cursor": shared_cursor}),
            recipient_global_slot=0,
        )


def test_shared_getter_uses_only_transition_free_active_source_factory(
    service_cases: _ServiceCases,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_diagnostic_factory(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("raw Shared product called the diagnostic factory")

    monkeypatch.setattr(
        evaluation_adapter_module,
        "build_shared_obs_source_material_projection_v1",
        reject_diagnostic_factory,
    )
    service = ReplayViewerService(
        service_cases.shared.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=5,
        viewer_session_id="shared-transition-free-getter",
    )
    raw = service.current_frame()
    raw_bytes = raw.model_dump_json()
    original = (
        replay_service_module.build_shared_obs_authority_source_material_projection_v1
    )
    calls: list[tuple[int, int]] = []

    def capture_authority_source(
        context: EvaluationEpisodeContextV1,
        frame: EvaluationFrameV1,
        *,
        selected_global_slot: int,
    ) -> SharedObsSourceMaterialProjectionV1:
        calls.append((frame.frame_index, selected_global_slot))
        return original(
            context,
            frame,
            selected_global_slot=selected_global_slot,
        )

    monkeypatch.setattr(
        replay_service_module,
        "build_shared_obs_authority_source_material_projection_v1",
        capture_authority_source,
    )

    result = service.current_presentation()

    assert result.outcome == "response"
    assert type(result.payload) is ReplaySharedObsAuthorizedPresentationFrameV1
    active_slots = tuple(
        row.global_slot
        for row in service_cases.shared.bundle.replay.header.context.roster
        if row.configured_active
    )
    per_epoch = (5, *(slot for slot in active_slots if slot != 5))
    assert calls == [
        *((1, slot) for slot in per_epoch),
        *((0, slot) for slot in per_epoch),
    ]
    assert service.current_frame() is raw
    assert service.current_frame().model_dump_json() == raw_bytes
    assert service.pov_index_build_count == 0
    assert service.shared_timeline_build_count == 1


def test_shared_presentation_is_independent_of_researcher_inspection_state(
    service_cases: _ServiceCases,
) -> None:
    services = tuple(
        ReplayViewerService(
            service_cases.shared.bundle,
            initial_frame_index=1,
            view_mode="pov",
            selected_global_slot=inspection_slot,
            pov_global_slot=0,
            viewer_session_id="shared-fixed-pov-inspection-independence",
        )
        for inspection_slot in (1, 6)
    )

    raw_frames = tuple(service.current_frame() for service in services)
    presentations = tuple(service.current_presentation() for service in services)

    assert raw_frames[0].model_dump_json() == raw_frames[1].model_dump_json()
    assert presentations[0].outcome == presentations[1].outcome == "response"
    assert presentations[0].payload.model_dump_json() == (
        presentations[1].payload.model_dump_json()
    )


def test_shared_packager_ignores_forbidden_transition_branches(
    service_cases: _ServiceCases,
) -> None:
    case = service_cases.shared
    service = ReplayViewerService(
        case.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="shared-selective-transitions",
    )
    raw = cast(SharedObsAgentPovReplayViewerFrameV1, service.current_frame())
    current_recipient, current_nonrecipient = _shared_authority_sources(
        case,
        frame_index=1,
        recipient_global_slot=0,
    )
    previous_recipient, previous_nonrecipient = _shared_authority_sources(
        case,
        frame_index=0,
        recipient_global_slot=0,
    )
    incoming = case.bundle.replay.transitions[0]
    outgoing = case.bundle.replay.transitions[1]

    def poison_hidden_rows(
        transition: EvaluationTransitionV1,
    ) -> EvaluationTransitionV1:
        acceptance = transition.facts.action_acceptance_facts

        def poison_joint(joint: JointActionV1) -> JointActionV1:
            move = cast(list[object], list(joint.move))
            target = cast(list[object], list(joint.select_target))
            ultimate = cast(list[object], list(joint.use_ultimate))
            move[1] = "forbidden-other-move"
            target[1] = "forbidden-other-target"
            ultimate[1] = "forbidden-other-ultimate"
            return joint.model_copy(
                update={
                    "move": tuple(move),
                    "select_target": tuple(target),
                    "use_ultimate": tuple(ultimate),
                }
            )

        poisoned_acceptance = acceptance.model_copy(
            update={
                "submitted_joint_action": poison_joint(
                    acceptance.submitted_joint_action
                ),
                "accepted_joint_action": poison_joint(acceptance.accepted_joint_action),
                "submitted_action_tuple_is_out_of_domain_by_actor": ["forbidden"],
                "in_domain_move_action_is_rejected_by_actor": ["forbidden"],
                "in_domain_combat_action_pair_is_rejected_by_actor": ["forbidden"],
            }
        )
        poisoned_facts = transition.facts.model_copy(
            update={
                "action_acceptance_facts": poisoned_acceptance,
                "combat_transition_facts": SimpleNamespace(forbidden="combat"),
                "death_facts": SimpleNamespace(forbidden="death"),
                "spawn_shield_facts": SimpleNamespace(forbidden="shield"),
                "respawn_facts": SimpleNamespace(forbidden="respawn"),
                "regeneration_facts": SimpleNamespace(forbidden="regeneration"),
                "physical_facts": SimpleNamespace(forbidden="physical"),
                "aura_facts": SimpleNamespace(forbidden="aura"),
                "status_lifecycle_facts": SimpleNamespace(forbidden="status"),
            }
        )
        return transition.model_copy(
            update={
                "facts": poisoned_facts,
                "events": ["forbidden-event"],
                "canonical_reward_by_agent": ["forbidden-reward"],
                "canonical_reward_by_team": ["forbidden-team-reward"],
                "terminated": "forbidden-termination",
                "truncated": "forbidden-truncation",
                "owning_task_end_reason": SimpleNamespace(forbidden="history"),
            }
        )

    baseline = build_replay_shared_obs_authorized_presentation_v1(
        raw,
        public_catalog=case.bundle.replay.header.context.static_mechanics_catalog,
        source_authority_epoch=raw.revision,
        authorized_recipient_global_slot=0,
        current_recipient_source_material=current_recipient,
        current_active_nonrecipient_source_material=current_nonrecipient,
        previous_recipient_source_material=previous_recipient,
        previous_active_nonrecipient_source_material=previous_nonrecipient,
        incoming_transition=incoming,
        outgoing_transition=outgoing,
    )
    mutated = build_replay_shared_obs_authorized_presentation_v1(
        raw,
        public_catalog=case.bundle.replay.header.context.static_mechanics_catalog,
        source_authority_epoch=raw.revision,
        authorized_recipient_global_slot=0,
        current_recipient_source_material=current_recipient,
        current_active_nonrecipient_source_material=current_nonrecipient,
        previous_recipient_source_material=previous_recipient,
        previous_active_nonrecipient_source_material=previous_nonrecipient,
        incoming_transition=poison_hidden_rows(incoming),
        outgoing_transition=poison_hidden_rows(outgoing),
    )

    assert mutated == baseline
    assert mutated.model_dump_json() == baseline.model_dump_json()


def test_shared_packager_ignores_contributor_non_authority_and_unavailable_payloads(
    service_cases: _ServiceCases,
) -> None:
    case = service_cases.shared
    service = ReplayViewerService(
        case.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="shared-selective-contributors",
    )
    raw = cast(SharedObsAgentPovReplayViewerFrameV1, service.current_frame())
    current_recipient, current_nonrecipient = _shared_authority_sources(
        case,
        frame_index=1,
        recipient_global_slot=0,
    )
    previous_recipient, previous_nonrecipient = _shared_authority_sources(
        case,
        frame_index=0,
        recipient_global_slot=0,
    )

    unavailable_ids = {
        row.sensor_source_public_agent_id
        for row in current_recipient.sensor_source_availability
        if not row.recorded_available
    }

    def poison_contributors(
        contributors: tuple[SharedObsSourceMaterialProjectionV1, ...],
    ) -> tuple[SharedObsSourceMaterialProjectionV1, ...]:
        poisoned_rows: list[SharedObsSourceMaterialProjectionV1] = []
        for contributor in contributors:
            poisoned = copy.deepcopy(contributor)
            object.__setattr__(poisoned, "incoming_transition_id", "forbidden")
            object.__setattr__(
                poisoned.base_sensor_frame,
                "action_mask",
                SimpleNamespace(forbidden="mask"),
            )
            object.__setattr__(
                poisoned.base_sensor_frame,
                "spawn_lifecycle",
                SimpleNamespace(forbidden="lifecycle"),
            )
            object.__setattr__(
                poisoned.base_sensor_frame,
                "previous_timestep_actions",
                SimpleNamespace(forbidden="history"),
            )
            object.__setattr__(
                poisoned.base_sensor_frame,
                "objective_features",
                [],
            )
            object.__setattr__(
                poisoned.base_sensor_scene,
                "map",
                SimpleNamespace(forbidden="map"),
            )
            if poisoned.base_sensor_frame.public_agent_id in unavailable_ids:
                object.__setattr__(
                    poisoned.base_sensor_frame,
                    "self_features",
                    ("forbidden-unavailable-payload",),
                )
            poisoned_rows.append(poisoned)
        return tuple(poisoned_rows)

    baseline = build_replay_shared_obs_authorized_presentation_v1(
        raw,
        public_catalog=case.bundle.replay.header.context.static_mechanics_catalog,
        source_authority_epoch=raw.revision,
        authorized_recipient_global_slot=0,
        current_recipient_source_material=current_recipient,
        current_active_nonrecipient_source_material=current_nonrecipient,
        previous_recipient_source_material=previous_recipient,
        previous_active_nonrecipient_source_material=previous_nonrecipient,
        incoming_transition=case.bundle.replay.transitions[0],
        outgoing_transition=case.bundle.replay.transitions[1],
    )
    mutated = build_replay_shared_obs_authorized_presentation_v1(
        raw,
        public_catalog=case.bundle.replay.header.context.static_mechanics_catalog,
        source_authority_epoch=raw.revision,
        authorized_recipient_global_slot=0,
        current_recipient_source_material=current_recipient,
        current_active_nonrecipient_source_material=poison_contributors(
            current_nonrecipient
        ),
        previous_recipient_source_material=previous_recipient,
        previous_active_nonrecipient_source_material=poison_contributors(
            previous_nonrecipient
        ),
        incoming_transition=case.bundle.replay.transitions[0],
        outgoing_transition=case.bundle.replay.transitions[1],
    )

    assert unavailable_ids
    assert mutated == baseline
    assert mutated.model_dump_json() == baseline.model_dump_json()


@pytest.mark.parametrize("wrong_credential", (True, -1, 10, 1, 5))
def test_shared_packager_rejects_nonfixed_recipient_credentials(
    service_cases: _ServiceCases,
    wrong_credential: object,
) -> None:
    case = service_cases.shared
    service = ReplayViewerService(
        case.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="shared-fixed-credential",
    )
    raw = cast(SharedObsAgentPovReplayViewerFrameV1, service.current_frame())
    current_recipient, current_nonrecipient = _shared_authority_sources(
        case,
        frame_index=1,
        recipient_global_slot=0,
    )
    previous_recipient, previous_nonrecipient = _shared_authority_sources(
        case,
        frame_index=0,
        recipient_global_slot=0,
    )

    with pytest.raises(ValueError):
        build_replay_shared_obs_authorized_presentation_v1(
            raw,
            public_catalog=case.bundle.replay.header.context.static_mechanics_catalog,
            source_authority_epoch=raw.revision,
            authorized_recipient_global_slot=cast(int, wrong_credential),
            current_recipient_source_material=current_recipient,
            current_active_nonrecipient_source_material=current_nonrecipient,
            previous_recipient_source_material=previous_recipient,
            previous_active_nonrecipient_source_material=previous_nonrecipient,
            incoming_transition=case.bundle.replay.transitions[0],
            outgoing_transition=case.bundle.replay.transitions[1],
        )


@pytest.mark.parametrize(
    ("audience", "pov_global_slot"),
    (
        ("no_shared_obs", 0),
        ("no_shared_obs", 5),
        ("shared_obs", 0),
        ("shared_obs", 5),
    ),
)
def test_agent_presentations_exclude_privileged_fields_and_canonical_values(
    service_cases: _ServiceCases,
    audience: Literal["no_shared_obs", "shared_obs"],
    pov_global_slot: int,
) -> None:
    case = (
        service_cases.complete if audience == "no_shared_obs" else service_cases.shared
    )
    service = ReplayViewerService(
        case.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=pov_global_slot,
        viewer_session_id=f"agent-privacy-{audience}-{pov_global_slot}",
    )
    raw = service.current_frame()
    payload = service.current_presentation().payload.model_dump(mode="json")
    keys = _recursive_keys(payload)
    strings = _recursive_string_values(payload)
    replay = case.bundle.replay
    metric = replay.metric_report_reference
    forbidden_values = {
        replay.artifact_id,
        replay.canonical_digest_sha256,
        replay.trajectory_content_digest_sha256,
        replay.header.header_id,
        replay.header.context_digest_sha256,
        replay.header.first_frame_id,
        replay.header.last_frame_id,
        raw.timeline_id,
        metric.report_artifact_id,
        metric.metric_report_id,
        metric.trajectory_content_digest_sha256,
        metric.canonical_digest_sha256,
        *(frame.frame_id for frame in replay.frames),
        *(transition.transition_id for transition in replay.transitions),
        *(transition.start_frame_id for transition in replay.transitions),
        *(transition.successor_frame_id for transition in replay.transitions),
        *(
            event.event_id
            for transition in replay.transitions
            for event in transition.events
        ),
    }

    assert not (strings & forbidden_values)
    for key in keys:
        assert "global_slot" not in key
        assert "artifact" not in key
        assert "timeline" not in key
        assert "metric" not in key
        assert "completion" not in key
        assert "processing" not in key
        assert "canonical_" not in key
        assert key not in {"cursor_generation", "choreography_generation"}
        if "digest" in key:
            assert key in {
                "authorized_endpoint_digest_sha256",
                "source_authorized_endpoint_digest_sha256",
            }


def test_shared_packager_rejects_source_tuple_and_epoch_swaps(
    service_cases: _ServiceCases,
) -> None:
    case = service_cases.shared
    service = ReplayViewerService(
        case.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="shared-source-swap",
    )
    raw = cast(SharedObsAgentPovReplayViewerFrameV1, service.current_frame())
    current_recipient, current_nonrecipient = _shared_authority_sources(
        case,
        frame_index=1,
        recipient_global_slot=0,
    )
    previous_recipient, previous_nonrecipient = _shared_authority_sources(
        case,
        frame_index=0,
        recipient_global_slot=0,
    )

    def build(
        *,
        source_authority_epoch: int = 0,
        current_recipient_source_material: SharedObsSourceMaterialProjectionV1 = (
            current_recipient
        ),
        current_active_nonrecipient_source_material: tuple[
            SharedObsSourceMaterialProjectionV1, ...
        ] = current_nonrecipient,
        previous_recipient_source_material: SharedObsSourceMaterialProjectionV1 = (
            previous_recipient
        ),
        previous_active_nonrecipient_source_material: tuple[
            SharedObsSourceMaterialProjectionV1, ...
        ] = previous_nonrecipient,
        incoming_transition: EvaluationTransitionV1 = case.bundle.replay.transitions[0],
        outgoing_transition: EvaluationTransitionV1 = case.bundle.replay.transitions[1],
    ) -> ReplaySharedObsAuthorizedPresentationFrameV1:
        return build_replay_shared_obs_authorized_presentation_v1(
            raw,
            public_catalog=case.bundle.replay.header.context.static_mechanics_catalog,
            source_authority_epoch=source_authority_epoch,
            authorized_recipient_global_slot=0,
            current_recipient_source_material=current_recipient_source_material,
            current_active_nonrecipient_source_material=(
                current_active_nonrecipient_source_material
            ),
            previous_recipient_source_material=previous_recipient_source_material,
            previous_active_nonrecipient_source_material=(
                previous_active_nonrecipient_source_material
            ),
            incoming_transition=incoming_transition,
            outgoing_transition=outgoing_transition,
        )

    baseline = build()
    assert type(baseline) is ReplaySharedObsAuthorizedPresentationFrameV1
    with pytest.raises(ValueError):
        build(current_active_nonrecipient_source_material=current_nonrecipient[:-1])
    with pytest.raises(ValueError):
        build(
            current_active_nonrecipient_source_material=(
                *current_nonrecipient,
                current_nonrecipient[0],
            )
        )
    with pytest.raises(ValueError):
        build(
            previous_recipient_source_material=current_recipient,
            previous_active_nonrecipient_source_material=current_nonrecipient,
        )
    with pytest.raises(ValueError):
        build(incoming_transition=case.bundle.replay.transitions[1])
    with pytest.raises(ValueError):
        build(outgoing_transition=case.bundle.replay.transitions[0])
    with pytest.raises(ValueError):
        build(source_authority_epoch=1)

    wrong_recipient, wrong_nonrecipient = _shared_authority_sources(
        case,
        frame_index=1,
        recipient_global_slot=1,
    )
    with pytest.raises(ValueError):
        build(
            current_recipient_source_material=wrong_recipient,
            current_active_nonrecipient_source_material=wrong_nonrecipient,
        )


def test_shared_packager_changes_for_valid_recipient_authorized_mask_mutation(
    service_cases: _ServiceCases,
) -> None:
    case = service_cases.shared
    service = ReplayViewerService(
        case.bundle,
        initial_frame_index=1,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="shared-recipient-authorized-change",
    )
    raw = cast(SharedObsAgentPovReplayViewerFrameV1, service.current_frame())
    current_recipient, current_nonrecipient = _shared_authority_sources(
        case,
        frame_index=1,
        recipient_global_slot=0,
    )
    previous_recipient, previous_nonrecipient = _shared_authority_sources(
        case,
        frame_index=0,
        recipient_global_slot=0,
    )
    baseline = _shared_presentation_for_raw(
        case,
        raw,
        recipient_global_slot=0,
    )
    accepted_move = case.bundle.replay.transitions[
        1
    ].facts.action_acceptance_facts.accepted_joint_action.move[0]
    changed_move = (accepted_move + 1) % 9
    assert changed_move != accepted_move

    changed_recipient = copy.deepcopy(current_recipient)
    original_mask = current_recipient.base_sensor_frame.action_mask
    toggled_move = list(original_mask.move)
    toggled_move[changed_move] = not toggled_move[changed_move]
    changed_mask = original_mask.model_copy(update={"move": tuple(toggled_move)})
    object.__setattr__(
        changed_recipient.base_sensor_frame,
        "action_mask",
        changed_mask,
    )
    mutated = build_replay_shared_obs_authorized_presentation_v1(
        raw,
        public_catalog=case.bundle.replay.header.context.static_mechanics_catalog,
        source_authority_epoch=raw.revision,
        authorized_recipient_global_slot=0,
        current_recipient_source_material=changed_recipient,
        current_active_nonrecipient_source_material=current_nonrecipient,
        previous_recipient_source_material=previous_recipient,
        previous_active_nonrecipient_source_material=previous_nonrecipient,
        incoming_transition=case.bundle.replay.transitions[0],
        outgoing_transition=case.bundle.replay.transitions[1],
    )

    assert mutated != baseline
    assert mutated.model_dump_json() != baseline.model_dump_json()
    assert (
        mutated.current_endpoint.parts.next_decision_action_mask.move[changed_move]
        is toggled_move[changed_move]
    )


def test_presentation_resource_result_requires_exact_outcome_payload_pair(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(
        service_cases.complete.bundle,
        selected_global_slot=1,
        viewer_session_id="presentation-result-contract",
    )
    frame = service.current_presentation().payload
    assert type(frame) is ReplayOracleAuthorizedPresentationFrameV1
    error = PresentationApiErrorV1(
        schema_version=1,
        error_code="audience_unavailable",
        message=PRESENTATION_AUDIENCE_UNAVAILABLE_MESSAGE,
    )

    with pytest.raises(TypeError, match="authorized frame root"):
        PresentationResourceResultV1(outcome="response", payload=error)
    with pytest.raises(TypeError, match="API error root"):
        PresentationResourceResultV1(
            outcome="audience_unavailable",
            payload=frame,
        )
    with pytest.raises(ValueError, match="unknown presentation resource outcome"):
        PresentationResourceResultV1(
            outcome=cast(Literal["response", "audience_unavailable"], "invalid"),
            payload=error,
        )


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


def test_shared_obs_raw_view_is_private_identity_only_and_never_exports_exact_pov(
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

    assert isinstance(frame, SharedObsAgentPovReplayViewerFrameV1)
    assert isinstance(timeline, SharedObsAgentPovReplayTimelineV1)
    assert frame.frame_kind == "shared_obs_agent_pov_replay_viewer"
    assert timeline.timeline_kind == "shared_obs_agent_pov"
    assert frame.artifact_summary == timeline.artifact_summary
    assert frame.artifact_summary.public_agent_id == frame.public_agent_id
    assert frame.recipient_frame_id.endswith(":frame:1")
    assert frame.incoming_recipient_transition_id is not None
    keys = _recursive_keys(payload)
    assert "pov_frame_id" not in keys
    assert "processing_disclosure" not in keys
    assert "projection" not in keys
    assert "selected_global_slot" not in keys
    assert "source_material_frame_id" not in keys
    assert "replay_reference" not in keys
    assert service.current_presentation().outcome == "response"
    assert service.shared_timeline_build_count == 1


def test_researcher_frame_preserves_recorded_experimental_movement_scale_only(
    service_cases: _ServiceCases,
) -> None:
    case = service_cases.noncanonical_movement_scale
    researcher_service = ReplayViewerService(
        case.bundle,
        viewer_session_id="recorded-scale-researcher",
    )
    pov_service = ReplayViewerService(
        case.bundle,
        view_mode="pov",
        viewer_session_id="recorded-scale-pov",
    )

    researcher_payload = json.loads(
        researcher_service.current_frame().model_dump_json()
    )
    pov_payload = json.loads(pov_service.current_frame().model_dump_json())
    recorded_config = case.bundle.replay.header.context.resolved_env_config
    recorded_scale = recorded_config.ordinary_movement_distance_scale

    assert recorded_scale == 0.375
    assert (
        researcher_payload["recorded_ordinary_movement_distance_scale"]
        == recorded_scale
    )
    assert "recorded_ordinary_movement_distance_scale" not in _recursive_keys(
        pov_payload
    )


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
            reference_global_slot=3,
        )
    with pytest.raises(ValueError, match="configured-active"):
        ReplayViewerService(
            service_cases.complete.bundle,
            reference_global_slot=cast(int, True),
        )
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


@pytest.mark.parametrize("invalid_lane", (True, -1, 2, "1"))
def test_initial_researcher_lane_requires_exact_domain_and_selection(
    service_cases: _ServiceCases,
    invalid_lane: object,
) -> None:
    with pytest.raises(ValueError, match="armed_lane"):
        ReplayViewerService(
            service_cases.complete.bundle,
            reference_global_slot=0,
            selected_global_slot=0,
            armed_lane=cast(Literal[0, 1], invalid_lane),
        )

    with pytest.raises(ValueError, match="selected_global_slot"):
        ReplayViewerService(
            service_cases.complete.bundle,
            reference_global_slot=0,
            armed_lane=0,
        )


def test_explicit_researcher_handoff_lane_clears_on_new_selection(
    service_cases: _ServiceCases,
) -> None:
    service = ReplayViewerService(
        service_cases.complete.bundle,
        reference_global_slot=0,
        selected_global_slot=0,
        armed_lane=1,
        viewer_session_id="explicit-researcher-presentation",
    )
    initial = cast(ResearcherReplayViewerFrameV1, service.current_frame())
    initial_legality = initial.projection.scene.next_decision_selected_legality
    assert initial.projection.scene.selection is not None
    assert initial.projection.scene.selection.controlled_global_slot == 0
    assert initial.projection.scene.selection.selected_global_slot == 0
    assert initial_legality is not None
    assert initial_legality.armed_lane == 1

    selected = _response(
        _apply(
            service,
            ReplaySelectAgentCommandV1(selected_global_slot=5),
            command_id="replace-handoff-selection",
        )
    ).frame
    assert isinstance(selected, ResearcherReplayViewerFrameV1)
    selected_legality = selected.projection.scene.next_decision_selected_legality
    assert selected.projection.scene.selection is not None
    assert selected.projection.scene.selection.controlled_global_slot == 0
    assert selected.projection.scene.selection.selected_global_slot == 5
    assert selected_legality is not None
    assert selected_legality.armed_lane is None
    assert selected_legality.armed_pair_legal is False


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
        ReplaySetPresetCommandV1.model_validate({"preset": "debug"}),
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


@pytest.mark.parametrize(
    "legacy_preset",
    ("presentation", "analysis", "technical", "debug"),
)
def test_replay_preset_requests_are_fixed_analysis_no_ops(
    service_cases: _ServiceCases,
    legacy_preset: str,
) -> None:
    service = ReplayViewerService(service_cases.complete.bundle)

    response = _response(
        _apply(
            service,
            ReplaySetPresetCommandV1.model_validate({"preset": legacy_preset}),
            command_id=f"legacy-preset-{legacy_preset}",
        )
    )

    assert response.result == "no_op"
    assert response.frame.preset == "analysis"
    assert response.frame.revision == 0
    assert service.revision == 0


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


def test_loaded_commands_and_static_render_never_enter_scientific_factories(
    service_cases: _ServiceCases,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loaded replay consumers cannot become a second simulator or event source."""
    source = service_cases.complete.bundle
    assert source.metric_report_artifact is not None
    replay_path = tmp_path / "scientific-authority-sentinel.marlbg-replay.json"
    save_replay_bundle_v1(
        ReplayBundleV1(
            replay=source.replay,
            metric_report_artifact=source.metric_report_artifact,
        ),
        replay_path,
    )

    forbidden_calls: list[str] = []

    def forbidden_factory(label: str) -> Callable[..., object]:
        def fail(*_args: object, **_kwargs: object) -> object:
            forbidden_calls.append(label)
            raise AssertionError(f"loaded replay entered forbidden authority: {label}")

        return fail

    forbidden_seams = (
        (core_env_module, ("initialize_scenario_state", "reset", "step")),
        (
            core_geometry_module,
            ("project_movement_with_geometry", "has_clear_line_of_sight"),
        ),
        (
            evaluation_capture_module,
            (
                "capture_initial_evaluation_frame_v1",
                "capture_evaluation_transition_unit_v1",
                "_normalize_reward_v1",
                "_reconstruct_transition_facts",
            ),
        ),
        (evaluation_events_module, ("decode_evaluation_events_v1",)),
        (
            live_control_module,
            (
                "create_session",
                "reset_session",
                "submit_joint_action",
                "build_interactive_joint_action",
                "build_scripted_joint_action",
                "step",
                "capture_initial_evaluation_frame_v1",
                "capture_evaluation_transition_unit_v1",
            ),
        ),
        (jax.random, ("key", "split")),
    )
    for module, names in forbidden_seams:
        for name in names:
            monkeypatch.setattr(
                module,
                name,
                forbidden_factory(f"{module.__name__}.{name}"),
            )

    loaded = load_replay_bundle_v1(replay_path, require_metric_report=True)
    service = ReplayViewerService(
        loaded,
        viewer_session_id="cp75-no-second-simulator",
    )
    assert isinstance(service.current_frame(), ResearcherReplayViewerFrameV1)
    assert isinstance(service.current_timeline(), ResearcherReplayTimelineV1)

    advanced = _response(
        _apply(service, ReplayNextFrameCommandV1(), command_id="sentinel-next")
    )
    assert advanced.frame.cursor.frame_index == 1
    _response(
        _apply(
            service,
            ReplaySetPovActorCommandV1(global_slot=1),
            command_id="sentinel-choose-pov-actor",
        )
    )
    pov = _response(
        _apply(
            service,
            ReplaySetViewCommandV1(view_mode="pov"),
            command_id="sentinel-enter-pov",
        )
    ).frame
    assert isinstance(pov, ActorPovReplayViewerFrameV1)
    assert pov.pov_global_slot == 1
    assert isinstance(service.current_timeline(), ActorPovReplayTimelineV1)

    switched_actor = _response(
        _apply(
            service,
            ReplaySetPovActorCommandV1(global_slot=0),
            command_id="sentinel-switch-pov-actor",
        )
    ).frame
    assert isinstance(switched_actor, ActorPovReplayViewerFrameV1)
    assert switched_actor.pov_global_slot == 0
    restored = _response(
        _apply(
            service,
            ReplaySetViewCommandV1(view_mode="researcher"),
            command_id="sentinel-return-researcher",
        )
    ).frame
    assert isinstance(restored, ResearcherReplayViewerFrameV1)
    selected = _response(
        _apply(
            service,
            ReplaySelectAgentCommandV1(selected_global_slot=1),
            command_id="sentinel-select-reference",
        )
    ).frame
    assert isinstance(selected, ResearcherReplayViewerFrameV1)
    assert selected.projection.scene.selection is not None
    assert selected.projection.scene.selection.selected_global_slot == 1

    rendered: list[tuple[BattlefieldSceneV2, VisualEventBatchV2 | None]] = []
    show_calls: list[None] = []

    def capture_render(
        scene: BattlefieldSceneV2,
        *,
        event_batch: VisualEventBatchV2 | None = None,
    ) -> object:
        rendered.append((scene, event_batch))
        return object()

    monkeypatch.setattr(
        static_renderer_module,
        "_load_pyplot",
        lambda: SimpleNamespace(show=lambda: show_calls.append(None)),
    )
    monkeypatch.setattr(
        static_renderer_module,
        "render_scene_geometry",
        capture_render,
    )
    assert (
        static_renderer_module.run_static_replay_renderer(
            replay_path=replay_path,
            frame_index=1,
            show_ranges=True,
        )
        == 0
    )
    assert show_calls == [None]
    assert len(rendered) == 1
    assert rendered[0][0].frame_index == 1
    assert rendered[0][1] is not None
    assert forbidden_calls == []


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
