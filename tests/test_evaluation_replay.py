"""Standard replay artifact construction and semantic validation proofs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ConfigDict, PrivateAttr, ValidationError
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    captured_resumed_team_deathmatch_horizon_trajectory,
    captured_team_deathmatch_threshold_trajectory,
    evaluation_env_config,
    mage_target_none_ultimate_action,
)

from marl_battlegrounds.core.types import CONTEXT_FEATURE_CURRENT_TIMESTEP
from marl_battlegrounds.evaluation.events import decode_evaluation_events_v1
from marl_battlegrounds.evaluation.metrics import (
    CompletionState,
    EvaluationEpisodeCompletionV1,
    EvaluationEpisodeObserverV1,
    EvaluationMetricReducerStateV1,
    EvaluationMetricReducerV1,
    EvaluationMetricReportV1,
    EvaluationProcessingFailureV1,
    EvaluationProcessingStatusV1,
    EvaluationTransitionViewV1,
    RolloutFailureOrigin,
    SufficientStatisticDraftV1,
    build_evaluation_observer_v1,
)
from marl_battlegrounds.evaluation.models import (
    REQUIRED_SCHEMA_BINDINGS_V1,
    AggregationKeyV1,
    CaptureProfile,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationModel,
    EvaluationTransitionV1,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.replay import (
    REQUIRED_REPLAY_ENVELOPE_SCHEMA_BINDINGS_V1,
    EvaluationMetricReportArtifactV1,
    MetricReportReferenceV1,
    ReplayArtifactHeaderV1,
    ReplayArtifactReferenceV1,
    ReplayArtifactV1,
    ReplayBundleV1,
    ReplayTrajectoryContentReferenceV1,
    ReplayWrapperMetadataV1,
    RuntimeProvenanceV1,
    build_replay_artifact_reference_v1,
    build_replay_artifact_v1,
    build_replay_bundle_v1,
    iter_replay_transition_views_v1,
    validate_metric_report_artifact_against_replay_v1,
    validate_replay_artifact_v1,
)


@pytest.fixture(scope="module")
def runtime_provenance() -> RuntimeProvenanceV1:
    """Return deterministic, path-free host provenance for replay tests."""
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


class _FailingAdvanceState(EvaluationMetricReducerStateV1):
    """Valid immutable reducer state used before an intentional advance failure."""


@dataclass(slots=True)
class _FailingAdvanceReducer:
    """Fail after CP2 accepts one unit, separating physical and metric truth."""

    reducer_id: str = "test.replay.failing_advance"
    reducer_version: int = 1

    def initialize(
        self,
        context: EvaluationEpisodeContextV1,
        initial_frame: EvaluationFrameV1,
    ) -> EvaluationMetricReducerStateV1:
        del context, initial_frame
        return _FailingAdvanceState(
            reducer_id=self.reducer_id,
            reducer_version=self.reducer_version,
        )

    def advance(
        self,
        previous_state: EvaluationMetricReducerStateV1,
        view: EvaluationTransitionViewV1,
    ) -> EvaluationMetricReducerStateV1:
        del previous_state, view
        raise RuntimeError("intentional replay-test reducer failure")

    def finalize(
        self,
        state: EvaluationMetricReducerStateV1,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        del state, completion, processing_status
        return ()


@dataclass(slots=True)
class _FailingFinalizeReducer:
    """Process every unit, then fail while materializing the final report."""

    reducer_id: str = "test.replay.failing_finalize"
    reducer_version: int = 1

    def initialize(
        self,
        context: EvaluationEpisodeContextV1,
        initial_frame: EvaluationFrameV1,
    ) -> EvaluationMetricReducerStateV1:
        del context, initial_frame
        return _FailingAdvanceState(
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
        raise RuntimeError("intentional replay-test finalization failure")


class _ReplayArtifactSubtype(ReplayArtifactV1):
    """Undeclared wire-root subtype for exact-type rejection."""


class _PrivateRuntimeProvenance(RuntimeProvenanceV1):
    """Nested subtype carrying private storage outside its wire fields."""

    _hidden: list[int] = PrivateAttr(default_factory=lambda: [1])


class _UnfrozenFrame(EvaluationFrameV1):
    """Nested frame subtype weakening the immutable wire contract."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=False,
        strict=True,
    )


def _feed_observer(
    observer: EvaluationEpisodeObserverV1,
    trajectory: CapturedEvaluationTrajectory,
) -> None:
    observer.start(trajectory.frames[0])
    for transition, successor_frame in zip(
        trajectory.transitions,
        trajectory.frames[1:],
        strict=True,
    ):
        observer.append(transition, successor_frame)


def _finalize_observer_and_report(
    trajectory: CapturedEvaluationTrajectory,
    *,
    completion_state: CompletionState,
    end_or_failure_reason: str | None = None,
    failure_origin: RolloutFailureOrigin | None = None,
    reducers: tuple[EvaluationMetricReducerV1, ...] = (),
) -> tuple[EvaluationEpisodeObserverV1, EvaluationMetricReportV1]:
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=reducers,
    )
    _feed_observer(observer, trajectory)
    report = observer.finalize(
        completion_state=completion_state,
        end_or_failure_reason=end_or_failure_reason,
        failure_origin=failure_origin,
    )
    return observer, report


def _build_bundle(
    trajectory: CapturedEvaluationTrajectory,
    runtime_provenance: RuntimeProvenanceV1,
    *,
    completion_state: CompletionState,
    end_or_failure_reason: str | None = None,
    failure_origin: RolloutFailureOrigin | None = None,
    reducers: tuple[EvaluationMetricReducerV1, ...] = (),
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
) -> ReplayBundleV1:
    observer, report = _finalize_observer_and_report(
        trajectory,
        completion_state=completion_state,
        end_or_failure_reason=end_or_failure_reason,
        failure_origin=failure_origin,
        reducers=reducers,
    )
    return build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=runtime_provenance,
        wrapper_stack=wrapper_stack,
    )


def _wrapper(
    position: int,
    *,
    wrapper_id: str = "test.wrapper",
    digest: str = "a" * 64,
) -> ReplayWrapperMetadataV1:
    return ReplayWrapperMetadataV1(
        position=position,
        wrapper_id=wrapper_id,
        wrapper_version=1,
        configuration_digest_sha256=digest,
    )


def _rebuild_header(
    header: ReplayArtifactHeaderV1,
    **updates: object,
) -> ReplayArtifactHeaderV1:
    payload = header.model_dump(mode="python")
    payload.update(updates)
    return ReplayArtifactHeaderV1.model_validate(payload)


def _rebuild_report_artifact(
    artifact: EvaluationMetricReportArtifactV1,
    *,
    source_trajectory: ReplayTrajectoryContentReferenceV1 | None = None,
    report: EvaluationMetricReportV1 | None = None,
) -> EvaluationMetricReportArtifactV1:
    payload: dict[str, object] = {
        "schema_id": artifact.schema_id,
        "schema_version": artifact.schema_version,
        "report_artifact_id": artifact.report_artifact_id,
        "source_trajectory": (
            artifact.source_trajectory
            if source_trajectory is None
            else source_trajectory
        ),
        "report": artifact.report if report is None else report,
    }
    return EvaluationMetricReportArtifactV1.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )


def _with_recomputed_digests(
    artifact: ReplayArtifactV1,
    *,
    header: ReplayArtifactHeaderV1 | None = None,
    completion: EvaluationEpisodeCompletionV1 | None = None,
    processing_status: EvaluationProcessingStatusV1 | None = None,
    frames: tuple[EvaluationFrameV1, ...] | None = None,
    transitions: tuple[EvaluationTransitionV1, ...] | None = None,
) -> ReplayArtifactV1:
    """Re-address a locally valid envelope so O(T) semantic checks are exercised."""
    selected_header = artifact.header if header is None else header
    selected_completion = artifact.completion if completion is None else completion
    selected_processing = (
        artifact.processing_status if processing_status is None else processing_status
    )
    selected_frames = artifact.frames if frames is None else frames
    selected_transitions = artifact.transitions if transitions is None else transitions
    trajectory_payload: dict[str, object] = {
        "header": selected_header,
        "completion": selected_completion,
        "processing_status": selected_processing,
        "frames": selected_frames,
        "transitions": selected_transitions,
    }
    trajectory_digest = canonical_digest_sha256(trajectory_payload)
    metric_reference = artifact.metric_report_reference.model_copy(
        update={"trajectory_content_digest_sha256": trajectory_digest}
    )
    artifact_payload: dict[str, object] = {
        "schema_id": artifact.schema_id,
        "schema_version": artifact.schema_version,
        "artifact_id": artifact.artifact_id,
        "trajectory_content_digest_sha256": trajectory_digest,
        "header": selected_header,
        "completion": selected_completion,
        "processing_status": selected_processing,
        "metric_report_reference": metric_reference,
        "frames": selected_frames,
        "transitions": selected_transitions,
    }
    return ReplayArtifactV1.model_validate(
        {
            **artifact_payload,
            "canonical_digest_sha256": canonical_digest_sha256(artifact_payload),
        }
    )


def _with_simulator_epoch(
    frame: EvaluationFrameV1,
    context: EvaluationEpisodeContextV1,
    simulator_step_count: int,
) -> EvaluationFrameV1:
    """Shift one frame while preserving its public timestep projection."""
    context_rows = [list(row) for row in frame.base_observation.context_features]
    for global_slot, roster_row in enumerate(context.roster):
        if roster_row.configured_active:
            context_rows[global_slot][CONTEXT_FEATURE_CURRENT_TIMESTEP] = float(
                simulator_step_count
            )
    base_observation = frame.base_observation.model_copy(
        update={"context_features": tuple(tuple(row) for row in context_rows)}
    )
    return frame.model_copy(
        update={
            "simulator_step_count": simulator_step_count,
            "base_observation": base_observation,
        }
    )


def _with_readdressed_metric_reference(
    artifact: ReplayArtifactV1,
    reference: MetricReportReferenceV1,
) -> ReplayArtifactV1:
    """Change only the sidecar reference and recompute the outer artifact digest."""
    payload: dict[str, object] = {
        "schema_id": artifact.schema_id,
        "schema_version": artifact.schema_version,
        "artifact_id": artifact.artifact_id,
        "trajectory_content_digest_sha256": (artifact.trajectory_content_digest_sha256),
        "header": artifact.header,
        "completion": artifact.completion,
        "processing_status": artifact.processing_status,
        "metric_report_reference": reference,
        "frames": artifact.frames,
        "transitions": artifact.transitions,
    }
    return ReplayArtifactV1.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )


@pytest.fixture(scope="module")
def complete_bundle(runtime_provenance: RuntimeProvenanceV1) -> ReplayBundleV1:
    trajectory = captured_evaluation_trajectory(
        transition_count=2,
        expected_horizon=2,
    )
    return _build_bundle(
        trajectory,
        runtime_provenance,
        completion_state="complete",
    )


@pytest.mark.parametrize(
    ("completion_state", "failure_origin"),
    (
        ("complete", None),
        ("partial", None),
        ("interrupted", None),
        ("failed", "simulation"),
    ),
)
def test_replay_builds_every_rollout_outcome(
    runtime_provenance: RuntimeProvenanceV1,
    completion_state: CompletionState,
    failure_origin: RolloutFailureOrigin | None,
) -> None:
    """Completion classification survives without becoming processing truth."""
    transition_count = 2 if completion_state == "complete" else 1
    trajectory = captured_evaluation_trajectory(
        transition_count=transition_count,
        expected_horizon=2,
    )
    reason = None if completion_state == "complete" else f"{completion_state} prefix"

    bundle = _build_bundle(
        trajectory,
        runtime_provenance,
        completion_state=completion_state,
        end_or_failure_reason=reason,
        failure_origin=failure_origin,
    )
    replay = bundle.replay

    validate_replay_artifact_v1(replay)
    validate_metric_report_artifact_against_replay_v1(
        bundle.metric_report_artifact,
        replay,
    )
    assert replay.completion.completion_state == completion_state
    assert replay.completion.failure_origin == failure_origin
    assert replay.processing_status.status == "succeeded"
    assert replay.header.expected_transition_count == 2
    assert replay.header.recorded_transition_count == transition_count
    assert replay.header.recorded_frame_count == transition_count + 1
    assert len(replay.frames) == transition_count + 1
    assert len(replay.transitions) == transition_count


@pytest.mark.parametrize(
    ("completion_state", "failure_origin"),
    (
        ("partial", None),
        ("interrupted", None),
        ("failed", "capture"),
    ),
)
def test_replay_preserves_valid_t_zero_prefixes(
    runtime_provenance: RuntimeProvenanceV1,
    completion_state: CompletionState,
    failure_origin: RolloutFailureOrigin | None,
) -> None:
    """A validated initial frame is a persistable noncomplete artifact."""
    trajectory = captured_evaluation_trajectory(
        transition_count=0,
        expected_horizon=2,
    )
    bundle = _build_bundle(
        trajectory,
        runtime_provenance,
        completion_state=completion_state,
        end_or_failure_reason=f"{completion_state} before transition zero",
        failure_origin=failure_origin,
    )

    assert bundle.replay.header.recorded_transition_count == 0
    assert bundle.replay.header.recorded_frame_count == 1
    assert bundle.replay.completion.last_valid_frame_index == 0
    assert bundle.replay.frames == trajectory.frames
    assert bundle.replay.transitions == ()
    assert tuple(iter_replay_transition_views_v1(bundle.replay)) == ()


def test_replay_separates_complete_rollout_from_failed_processing(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    """Reducer failure cannot erase a fully validated horizon-complete trajectory."""
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
    )
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_FailingAdvanceReducer(),),
    )
    observer.start(trajectory.frames[0])
    with pytest.raises(RuntimeError, match="advance failed"):
        observer.append(trajectory.transitions[0], trajectory.frames[1])
    report = observer.finalize(completion_state="complete")

    bundle = build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=runtime_provenance,
    )

    assert bundle.replay.completion.completion_state == "complete"
    assert bundle.replay.completion.validated_transition_count == 1
    assert bundle.replay.processing_status.status == "failed"
    assert bundle.replay.processing_status.processed_transition_count == 0
    assert bundle.replay.processing_status.failure is not None
    assert bundle.replay.processing_status.failure.stage == "reducer_advance"
    assert len(bundle.replay.frames) == 2
    assert len(bundle.replay.transitions) == 1


def test_replay_preserves_terminal_and_truncated_tail_truth(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    """Tail done flags, authoritative end reason, and completion stay aligned."""
    terminal_trajectory = captured_team_deathmatch_threshold_trajectory()
    horizon_trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        config=evaluation_env_config(max_steps=1),
    )
    terminal = _build_bundle(
        terminal_trajectory,
        runtime_provenance,
        completion_state="complete",
    ).replay
    horizon = _build_bundle(
        horizon_trajectory,
        runtime_provenance,
        completion_state="complete",
    ).replay

    assert terminal.completion.terminated is True
    assert terminal.completion.truncated is False
    assert terminal.completion.completion_bases == ("task_terminal",)
    assert (
        terminal.completion.end_or_failure_reason == "team_deathmatch_score_threshold"
    )
    assert horizon.completion.terminated is False
    assert horizon.completion.truncated is True
    assert horizon.completion.completion_bases == ("declared_horizon",)
    assert horizon.completion.end_or_failure_reason is None


def test_resumed_tdm_replay_completes_after_its_remaining_horizon(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    trajectory = captured_resumed_team_deathmatch_horizon_trajectory()

    bundle = _build_bundle(
        trajectory,
        runtime_provenance,
        completion_state="complete",
    )

    replay = bundle.replay
    assert replay.frames[0].simulator_step_count == 1
    assert replay.frames[0].snapshot.team_deathmatch_scores == (2, 1)
    assert len(replay.transitions) == replay.header.context.expected_horizon == 2
    assert replay.frames[-1].simulator_step_count == 3
    assert replay.transitions[-1].facts.team_deathmatch_facts.outcome == 3
    assert replay.transitions[-1].canonical_reward_by_team == (0.0, 0.0)
    assert replay.completion.terminated is False
    assert replay.completion.truncated is True
    assert replay.completion.completion_bases == ("declared_horizon",)
    assert replay.completion.end_or_failure_reason == "team_deathmatch_horizon"


def test_replay_indices_epochs_counts_and_identifiers_are_canonical(
    complete_bundle: ReplayBundleV1,
) -> None:
    """The serialized normal form is exactly T+1/T with adjacent epochs."""
    replay = complete_bundle.replay
    episode_id = replay.header.context.identity.episode_id

    assert tuple(frame.frame_index for frame in replay.frames) == (0, 1, 2)
    assert tuple(transition.transition_index for transition in replay.transitions) == (
        0,
        1,
    )
    assert tuple(frame.frame_id for frame in replay.frames) == tuple(
        f"{episode_id}:frame:{index}" for index in range(3)
    )
    assert tuple(transition.transition_id for transition in replay.transitions) == (
        f"{episode_id}:transition:0",
        f"{episode_id}:transition:1",
    )
    assert all(
        successor.simulator_step_count == start.simulator_step_count + 1
        for start, successor in zip(
            replay.frames[:-1],
            replay.frames[1:],
            strict=True,
        )
    )
    assert all(
        transition.facts.transition_start_step_count
        == replay.frames[index].simulator_step_count
        for index, transition in enumerate(replay.transitions)
    )
    assert replay.header.first_frame_id == replay.frames[0].frame_id
    assert replay.header.last_frame_id == replay.frames[-1].frame_id
    assert replay.completion.expected_transition_count == 2
    assert replay.completion.validated_transition_count == 2
    assert replay.completion.last_valid_frame_index == 2
    assert replay.completion.last_valid_frame_id == replay.frames[-1].frame_id
    assert replay.completion.completion_bases == ("declared_horizon",)


def test_replay_allows_artifact_epoch_zero_to_start_later_in_simulator_time(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    """Artifact indexing remains independent from an initial simulator epoch."""
    trajectory = captured_evaluation_trajectory(
        transition_count=0,
        expected_horizon=2,
    )
    shifted_initial = _with_simulator_epoch(
        trajectory.frames[0],
        trajectory.context,
        17,
    )
    shifted = CapturedEvaluationTrajectory(
        context=trajectory.context,
        frames=(shifted_initial,),
        transitions=(),
    )

    replay = _build_bundle(
        shifted,
        runtime_provenance,
        completion_state="interrupted",
        end_or_failure_reason="resumed capture stopped",
    ).replay

    assert replay.frames[0].frame_index == 0
    assert replay.frames[0].simulator_step_count == 17


def test_replay_schema_maps_preserve_exact_cp2_and_envelope_bindings(
    complete_bundle: ReplayBundleV1,
) -> None:
    replay = complete_bundle.replay
    source_bindings = tuple(
        (row.schema_id, row.schema_version)
        for row in replay.header.source_schema_versions
    )
    envelope_bindings = tuple(
        (row.schema_id, row.schema_version)
        for row in replay.header.envelope_schema_versions
    )

    assert source_bindings == REQUIRED_SCHEMA_BINDINGS_V1
    assert replay.header.source_schema_versions == replay.header.context.schema_versions
    assert envelope_bindings == REQUIRED_REPLAY_ENVELOPE_SCHEMA_BINDINGS_V1


def test_replay_and_metric_sidecar_round_trip_through_json(
    complete_bundle: ReplayBundleV1,
) -> None:
    """Strict JSON revalidation preserves both separately addressed artifacts."""
    replay_json = complete_bundle.replay.model_dump_json()
    report_json = complete_bundle.metric_report_artifact.model_dump_json()

    loaded_replay = ReplayArtifactV1.model_validate_json(replay_json)
    loaded_report = EvaluationMetricReportArtifactV1.model_validate_json(report_json)

    assert loaded_replay == complete_bundle.replay
    assert loaded_report == complete_bundle.metric_report_artifact
    assert canonical_json_bytes(loaded_replay) == canonical_json_bytes(
        complete_bundle.replay
    )
    validate_replay_artifact_v1(loaded_replay)
    validate_metric_report_artifact_against_replay_v1(
        loaded_report,
        loaded_replay,
    )
    reference = build_replay_artifact_reference_v1(loaded_replay)
    assert reference.artifact_id == loaded_replay.artifact_id
    assert reference.canonical_digest_sha256 == (loaded_replay.canonical_digest_sha256)
    assert reference.canonical_byte_length == len(canonical_json_bytes(loaded_replay))


def test_replay_iterator_equals_original_validated_transition_views(
    complete_bundle: ReplayBundleV1,
) -> None:
    replay = complete_bundle.replay
    loaded = ReplayArtifactV1.model_validate_json(replay.model_dump_json())
    expected = tuple(
        EvaluationTransitionViewV1(
            context=loaded.header.context,
            start_frame=loaded.frames[index],
            transition=transition,
            successor_frame=loaded.frames[index + 1],
        )
        for index, transition in enumerate(loaded.transitions)
    )

    assert tuple(iter_replay_transition_views_v1(loaded)) == expected
    assert tuple(iter_replay_transition_views_v1(loaded)) == tuple(
        iter_replay_transition_views_v1(replay)
    )


@pytest.mark.parametrize("builder_name", ("artifact", "bundle"))
@pytest.mark.parametrize("lifecycle", ("awaiting_initial", "open"))
def test_replay_builders_reject_unfinalized_observers(
    runtime_provenance: RuntimeProvenanceV1,
    builder_name: str,
    lifecycle: str,
) -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
    )
    finalized = _build_bundle(
        trajectory,
        runtime_provenance,
        completion_state="complete",
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    if lifecycle == "open":
        observer.start(trajectory.frames[0])
    builder = (
        build_replay_artifact_v1
        if builder_name == "artifact"
        else build_replay_bundle_v1
    )

    with pytest.raises(ValueError, match="finalized observer"):
        builder(
            observer,
            finalized.metric_report_artifact.report,
            runtime_provenance=runtime_provenance,
        )


@pytest.mark.parametrize(
    "capture_profile",
    ("training_light", "debug"),
)
def test_replay_builders_reject_nonretaining_capture_profiles(
    runtime_provenance: RuntimeProvenanceV1,
    capture_profile: CaptureProfile,
) -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        capture_profile=capture_profile,
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    _feed_observer(observer, trajectory)
    report = observer.finalize(completion_state="complete")
    assert observer.retained_frames is None
    assert observer.retained_transitions is None

    for builder in (build_replay_artifact_v1, build_replay_bundle_v1):
        with pytest.raises(ValueError, match="metric-complete retaining profile"):
            builder(
                observer,
                report,
                runtime_provenance=runtime_provenance,
            )


@pytest.mark.parametrize("missing_record", ("frame", "transition"))
def test_replay_rejects_broken_t_plus_one_over_t_cardinality(
    complete_bundle: ReplayBundleV1,
    missing_record: str,
) -> None:
    replay = complete_bundle.replay
    mutated = (
        replay.model_copy(update={"frames": replay.frames[:-1]})
        if missing_record == "frame"
        else replay.model_copy(update={"transitions": replay.transitions[:-1]})
    )

    with pytest.raises(ValueError, match="structural revalidation"):
        validate_replay_artifact_v1(mutated)


@pytest.mark.parametrize(
    "corruption",
    (
        "reordered_frames",
        "gapped_frames",
        "duplicate_frames",
        "reordered_transitions",
        "gapped_transitions",
        "duplicate_transitions",
    ),
)
def test_replay_rejects_reordered_gapped_and_duplicate_records(
    complete_bundle: ReplayBundleV1,
    corruption: str,
) -> None:
    replay = complete_bundle.replay
    frames = replay.frames
    transitions = replay.transitions

    if corruption == "reordered_frames":
        mutated = _with_recomputed_digests(
            replay,
            frames=(frames[0], frames[2], frames[1]),
        )
    elif corruption == "gapped_frames":
        gapped = frames[1].model_copy(
            update={
                "frame_index": 3,
                "frame_id": f"{frames[1].episode_id}:frame:3",
            }
        )
        mutated = _with_recomputed_digests(
            replay,
            frames=(frames[0], gapped, frames[2]),
        )
    elif corruption == "duplicate_frames":
        mutated = _with_recomputed_digests(
            replay,
            frames=(frames[0], frames[1], frames[1]),
        )
    elif corruption == "reordered_transitions":
        mutated = _with_recomputed_digests(
            replay,
            transitions=(transitions[1], transitions[0]),
        )
    elif corruption == "gapped_transitions":
        gapped = transitions[1].model_copy(
            update={
                "transition_index": 2,
                "transition_id": f"{transitions[1].episode_id}:transition:2",
                "start_frame_id": f"{transitions[1].episode_id}:frame:2",
                "successor_frame_id": f"{transitions[1].episode_id}:frame:3",
            }
        )
        mutated = _with_recomputed_digests(
            replay,
            transitions=(transitions[0], gapped),
        )
    else:
        mutated = _with_recomputed_digests(
            replay,
            transitions=(transitions[0], transitions[0]),
        )

    with pytest.raises(ValueError, match=r"positions|header reference"):
        validate_replay_artifact_v1(mutated)


def test_replay_rejects_nonadjacent_simulator_epochs(
    complete_bundle: ReplayBundleV1,
) -> None:
    replay = complete_bundle.replay
    gapped_successor = _with_simulator_epoch(
        replay.frames[1],
        replay.header.context,
        replay.frames[0].simulator_step_count + 2,
    )
    mutated = _with_recomputed_digests(
        replay,
        frames=(replay.frames[0], gapped_successor, replay.frames[2]),
    )

    with pytest.raises(ValueError, match="successor simulator step"):
        validate_replay_artifact_v1(mutated)


@pytest.mark.parametrize("foreign_record", ("frame", "transition"))
def test_replay_rejects_cross_episode_records(
    complete_bundle: ReplayBundleV1,
    foreign_record: str,
) -> None:
    replay = complete_bundle.replay
    foreign = captured_evaluation_trajectory(
        transition_count=2,
        expected_horizon=2,
        episode_id="episode-foreign",
    )
    if foreign_record == "frame":
        mutated = _with_recomputed_digests(
            replay,
            frames=(replay.frames[0], foreign.frames[1], replay.frames[2]),
        )
    else:
        mutated = _with_recomputed_digests(
            replay,
            transitions=(foreign.transitions[0], replay.transitions[1]),
        )

    with pytest.raises(ValueError, match=r"context episode|context-joined"):
        validate_replay_artifact_v1(mutated)


def test_replay_rejects_continuation_after_done_transition(
    complete_bundle: ReplayBundleV1,
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    terminal = _build_bundle(
        captured_team_deathmatch_threshold_trajectory(),
        runtime_provenance,
        completion_state="complete",
    ).replay
    header = terminal.header.model_copy(
        update={
            "recorded_transition_count": 2,
            "recorded_frame_count": 3,
            "last_frame_id": f"{terminal.header.context.identity.episode_id}:frame:2",
        }
    )
    mutated = _with_recomputed_digests(
        terminal,
        header=header,
        frames=(*terminal.frames, complete_bundle.replay.frames[2]),
        transitions=(
            terminal.transitions[0],
            complete_bundle.replay.transitions[1],
        ),
    )

    with pytest.raises(ValueError, match="cannot continue after a done transition"):
        validate_replay_artifact_v1(mutated)


def test_replay_rejects_completion_horizon_mismatch(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    base = _build_bundle(
        captured_evaluation_trajectory(
            transition_count=1,
            expected_horizon=2,
        ),
        runtime_provenance,
        completion_state="partial",
        end_or_failure_reason="prefix",
    ).replay
    other_horizon = _build_bundle(
        captured_evaluation_trajectory(
            transition_count=1,
            expected_horizon=3,
        ),
        runtime_provenance,
        completion_state="partial",
        end_or_failure_reason="prefix",
    ).replay
    mutated = _with_recomputed_digests(
        base,
        completion=other_horizon.completion,
    )

    with pytest.raises(ValueError, match="context horizon"):
        validate_replay_artifact_v1(mutated)


def test_replay_rejects_completion_done_flag_mismatch(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    base_trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=100,
    )
    base = _build_bundle(
        base_trajectory,
        runtime_provenance,
        completion_state="partial",
        end_or_failure_reason="prefix",
    ).replay
    terminal = _build_bundle(
        captured_team_deathmatch_threshold_trajectory(),
        runtime_provenance,
        completion_state="complete",
    ).replay
    mutated = _with_recomputed_digests(
        base,
        completion=terminal.completion,
    )

    with pytest.raises(ValueError, match="done flags"):
        validate_replay_artifact_v1(mutated)


def test_replay_rejects_completion_reason_mismatch(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    terminal = _build_bundle(
        captured_team_deathmatch_threshold_trajectory(),
        runtime_provenance,
        completion_state="complete",
    ).replay
    mismatched_completion = EvaluationEpisodeCompletionV1.model_validate(
        {
            **terminal.completion.model_dump(mode="python"),
            "end_or_failure_reason": "different_reason",
        }
    )
    mutated = _with_recomputed_digests(
        terminal,
        completion=mismatched_completion,
    )

    with pytest.raises(ValueError, match="authoritative tail truth"):
        validate_replay_artifact_v1(mutated)


def test_replay_preserves_partial_label_for_pre_horizon_prefix(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=2,
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    _feed_observer(observer, trajectory)
    report = observer.finalize(
        completion_state="partial",
        end_or_failure_reason="external_time_limit",
    )

    replay = build_replay_artifact_v1(
        observer,
        report,
        runtime_provenance=runtime_provenance,
    )

    assert replay.completion.completion_state == "partial"
    assert replay.completion.truncated is False
    assert replay.completion.end_or_failure_reason == "external_time_limit"


@pytest.mark.parametrize(
    "digest_field",
    ("canonical_digest_sha256", "trajectory_content_digest_sha256"),
)
def test_replay_rejects_wrong_digests(
    complete_bundle: ReplayBundleV1,
    digest_field: str,
) -> None:
    replay = complete_bundle.replay
    wrong_digest = "f" * 64
    updates: dict[str, object] = {digest_field: wrong_digest}
    if digest_field == "trajectory_content_digest_sha256":
        updates["metric_report_reference"] = replay.metric_report_reference.model_copy(
            update={"trajectory_content_digest_sha256": wrong_digest}
        )
    mutated = replay.model_copy(update=updates)

    with pytest.raises(ValueError, match="structural revalidation"):
        validate_replay_artifact_v1(mutated)


@pytest.mark.parametrize("schema_target", ("artifact", "header"))
def test_replay_rejects_unknown_schema_versions(
    complete_bundle: ReplayBundleV1,
    schema_target: str,
) -> None:
    replay = complete_bundle.replay
    mutated = (
        replay.model_copy(update={"schema_version": 2})
        if schema_target == "artifact"
        else replay.model_copy(
            update={"header": replay.header.model_copy(update={"schema_version": 2})}
        )
    )

    with pytest.raises((ValueError, ValidationError), match="structural revalidation"):
        validate_replay_artifact_v1(mutated)


def test_replay_rejects_nonexact_cp2_schema_bindings(
    complete_bundle: ReplayBundleV1,
) -> None:
    replay = complete_bundle.replay
    shortened = replay.header.source_schema_versions[:-1]
    mutated_header = replay.header.model_copy(
        update={"source_schema_versions": shortened}
    )
    mutated = replay.model_copy(update={"header": mutated_header})

    with pytest.raises(ValueError, match="structural revalidation"):
        validate_replay_artifact_v1(mutated)


def test_replay_rejects_tampered_canonical_events(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        actions=(mage_target_none_ultimate_action(),),
    )
    replay = _build_bundle(
        trajectory,
        runtime_provenance,
        completion_state="complete",
    ).replay
    assert replay.transitions[0].events
    tampered_transition = replay.transitions[0].model_copy(update={"events": ()})
    mutated = _with_recomputed_digests(
        replay,
        transitions=(tampered_transition,),
    )

    with pytest.raises(ValueError, match="canonical fact decoding"):
        validate_replay_artifact_v1(mutated)


def test_metric_sidecar_rejects_tampered_digest(
    complete_bundle: ReplayBundleV1,
) -> None:
    tampered = complete_bundle.metric_report_artifact.model_copy(
        update={"canonical_digest_sha256": "f" * 64}
    )

    with pytest.raises(ValueError, match="structural revalidation"):
        validate_metric_report_artifact_against_replay_v1(
            tampered,
            complete_bundle.replay,
        )


@pytest.mark.parametrize("alias", (True, 1.0))
def test_replay_wire_roots_reject_schema_version_aliases(
    complete_bundle: ReplayBundleV1,
    runtime_provenance: RuntimeProvenanceV1,
    alias: bool | float,
) -> None:
    replay_reference: ReplayArtifactReferenceV1 = build_replay_artifact_reference_v1(
        complete_bundle.replay
    )
    roots: tuple[EvaluationModel, ...] = (
        complete_bundle.replay,
        complete_bundle.replay.header,
        runtime_provenance,
        complete_bundle.metric_report_artifact,
        complete_bundle.metric_report_artifact.source_trajectory,
        complete_bundle.replay.metric_report_reference,
        replay_reference,
        _wrapper(0),
    )

    for root in roots:
        payload = root.model_dump(mode="python")
        payload["schema_version"] = alias
        with pytest.raises(ValidationError):
            type(root).model_validate(payload)


def test_replay_wire_roots_reject_python_lists_for_tuple_fields(
    complete_bundle: ReplayBundleV1,
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    cases: tuple[tuple[EvaluationModel, str], ...] = (
        (complete_bundle.replay, "frames"),
        (complete_bundle.replay, "transitions"),
        (complete_bundle.replay.header, "source_schema_versions"),
        (complete_bundle.replay.header, "wrapper_stack"),
        (runtime_provenance, "batch_shape"),
    )

    for root, field_name in cases:
        payload = root.model_dump(mode="python")
        payload[field_name] = list(payload[field_name])  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            type(root).model_validate(payload)


def test_runtime_provenance_requires_batch_product_to_equal_environment_count(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    payload = runtime_provenance.model_dump(mode="python")
    payload.update(environment_count=3, batch_shape=(2, 2))

    with pytest.raises(ValidationError):
        RuntimeProvenanceV1.model_validate(payload)


def test_runtime_package_version_must_join_context_code_revision(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
    )
    observer, report = _finalize_observer_and_report(
        trajectory,
        completion_state="complete",
    )
    mismatched_runtime = RuntimeProvenanceV1.model_validate(
        {
            **runtime_provenance.model_dump(mode="python"),
            "package_version": "9.9.9",
        }
    )

    with pytest.raises(ValueError):
        build_replay_artifact_v1(
            observer,
            report,
            runtime_provenance=mismatched_runtime,
        )


@pytest.mark.parametrize("positions", ((0, 2), (1, 0)))
def test_replay_rejects_gapped_or_reordered_wrapper_positions(
    runtime_provenance: RuntimeProvenanceV1,
    positions: tuple[int, int],
) -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
    )
    observer, report = _finalize_observer_and_report(
        trajectory,
        completion_state="complete",
    )
    wrappers = tuple(
        _wrapper(position, wrapper_id=f"test.wrapper.{ordinal}")
        for ordinal, position in enumerate(positions)
    )

    with pytest.raises(ValueError):
        build_replay_artifact_v1(
            observer,
            report,
            runtime_provenance=runtime_provenance,
            wrapper_stack=wrappers,
        )


def test_replay_allows_repeated_wrapper_ids_at_distinct_stack_positions(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
    )
    observer, report = _finalize_observer_and_report(
        trajectory,
        completion_state="complete",
    )
    wrappers = (
        _wrapper(0, wrapper_id="test.repeated", digest="a" * 64),
        _wrapper(1, wrapper_id="test.repeated", digest="b" * 64),
    )

    replay = build_replay_artifact_v1(
        observer,
        report,
        runtime_provenance=runtime_provenance,
        wrapper_stack=wrappers,
    )

    assert replay.header.wrapper_stack == wrappers


def test_replay_builder_rejects_an_alternate_finalized_report(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=2,
    )
    observer, committed_report = _finalize_observer_and_report(
        trajectory,
        completion_state="partial",
        end_or_failure_reason="prefix",
    )
    _, context_mismatch = _finalize_observer_and_report(
        captured_evaluation_trajectory(
            transition_count=1,
            expected_horizon=2,
            episode_id="episode-report-mismatch",
        ),
        completion_state="partial",
        end_or_failure_reason="prefix",
    )
    _, completion_mismatch = _finalize_observer_and_report(
        trajectory,
        completion_state="interrupted",
        end_or_failure_reason="prefix",
    )
    _, processing_mismatch = _finalize_observer_and_report(
        trajectory,
        completion_state="partial",
        end_or_failure_reason="prefix",
        reducers=(_FailingFinalizeReducer(),),
    )
    assert committed_report != context_mismatch
    assert committed_report.context == completion_mismatch.context
    assert committed_report.completion != completion_mismatch.completion
    assert committed_report.processing_status != processing_mismatch.processing_status

    for mismatched_report in (
        context_mismatch,
        completion_mismatch,
        processing_mismatch,
    ):
        with pytest.raises(ValueError):
            build_replay_artifact_v1(
                observer,
                mismatched_report,
                runtime_provenance=runtime_provenance,
            )

    cached_report = observer.finalized_report
    assert cached_report is not None
    assert cached_report is not committed_report
    canonical_report_snapshot = observer.finalized_report
    assert canonical_report_snapshot == cached_report
    assert canonical_report_snapshot is not cached_report
    exposed_context = observer.context
    exposed_frames = observer.retained_frames
    exposed_transitions = observer.retained_transitions
    assert exposed_frames is not None
    assert exposed_transitions is not None
    object.__setattr__(exposed_context, "expected_horizon", 99)
    object.__setattr__(exposed_frames[0], "frame_index", 99)
    object.__setattr__(exposed_transitions[0], "transition_index", 99)
    assert observer.context.expected_horizon == 2
    fresh_frames = observer.retained_frames
    fresh_transitions = observer.retained_transitions
    assert fresh_frames is not None
    assert fresh_transitions is not None
    assert fresh_frames[0].frame_index == 0
    assert fresh_transitions[0].transition_index == 0
    object.__setattr__(
        cached_report,
        "processing_status",
        processing_mismatch.processing_status,
    )
    assert observer.finalized_report == canonical_report_snapshot
    object.__setattr__(
        committed_report,
        "processing_status",
        processing_mismatch.processing_status,
    )
    assert observer.finalized_report == canonical_report_snapshot
    with pytest.raises(ValueError, match="must equal the observer report"):
        build_replay_artifact_v1(
            observer,
            committed_report,
            runtime_provenance=runtime_provenance,
        )


def test_complete_rollout_preserves_runner_closeout_reason_without_task_outcome(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    _feed_observer(observer, trajectory)
    report = observer.finalize(
        completion_state="complete",
        end_or_failure_reason="runner_horizon_closeout",
    )
    replay = build_replay_artifact_v1(
        observer,
        report,
        runtime_provenance=runtime_provenance,
    )

    assert replay.completion.end_or_failure_reason == "runner_horizon_closeout"
    assert replay.completion.terminated is False
    assert replay.completion.truncated is False
    assert replay.completion.completion_bases == ("declared_horizon",)
    assert replay.transitions[-1].owning_task_end_reason is None
    validate_replay_artifact_v1(replay)


def test_metric_sidecar_cross_validator_rejects_wrong_source_trajectory(
    complete_bundle: ReplayBundleV1,
) -> None:
    report_artifact = complete_bundle.metric_report_artifact
    wrong_source = ReplayTrajectoryContentReferenceV1.model_validate(
        {
            **report_artifact.source_trajectory.model_dump(mode="python"),
            "trajectory_content_digest_sha256": "e" * 64,
        }
    )
    mutated = _rebuild_report_artifact(
        report_artifact,
        source_trajectory=wrong_source,
    )

    with pytest.raises(ValueError):
        validate_metric_report_artifact_against_replay_v1(
            mutated,
            complete_bundle.replay,
        )


def test_metric_sidecar_cross_validator_rejects_wrong_report_context(
    complete_bundle: ReplayBundleV1,
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    divergent = _build_bundle(
        captured_evaluation_trajectory(
            transition_count=2,
            expected_horizon=2,
            aggregation_keys=(
                AggregationKeyV1(name="information_regime", value="no_shared_obs"),
                AggregationKeyV1(name="side", value="team_b"),
            ),
        ),
        runtime_provenance,
        completion_state="complete",
    )

    with pytest.raises(ValueError):
        validate_metric_report_artifact_against_replay_v1(
            divergent.metric_report_artifact,
            complete_bundle.replay,
        )


def test_metric_sidecar_cross_validator_rejects_wrong_report_completion(
    complete_bundle: ReplayBundleV1,
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    partial_prefix = _build_bundle(
        captured_evaluation_trajectory(
            transition_count=1,
            expected_horizon=2,
        ),
        runtime_provenance,
        completion_state="partial",
        end_or_failure_reason="short_prefix",
    )
    base_artifact = complete_bundle.metric_report_artifact
    mutated = _rebuild_report_artifact(
        base_artifact,
        report=partial_prefix.metric_report_artifact.report,
    )

    with pytest.raises(ValueError):
        validate_metric_report_artifact_against_replay_v1(
            mutated,
            complete_bundle.replay,
        )


def test_metric_sidecar_cross_validator_rejects_wrong_report_processing(
    complete_bundle: ReplayBundleV1,
) -> None:
    base_artifact = complete_bundle.metric_report_artifact
    failed_processing = EvaluationProcessingStatusV1(
        status="failed",
        processed_transition_count=2,
        failure=EvaluationProcessingFailureV1(
            stage="reducer_finalize",
            code="intentional_failure",
            reducer_id="test.reducer",
            reducer_version=1,
            detail="intentional failure",
        ),
    )
    changed_report = EvaluationMetricReportV1.model_validate(
        {
            **base_artifact.report.model_dump(mode="python"),
            "processing_status": failed_processing,
        }
    )
    mutated = _rebuild_report_artifact(
        base_artifact,
        report=changed_report,
    )

    with pytest.raises(ValueError):
        validate_metric_report_artifact_against_replay_v1(
            mutated,
            complete_bundle.replay,
        )


@pytest.mark.parametrize("reference_field", ("digest", "byte_length", "id"))
def test_metric_sidecar_cross_validator_rejects_wrong_report_reference(
    complete_bundle: ReplayBundleV1,
    reference_field: str,
) -> None:
    replay = complete_bundle.replay
    reference = replay.metric_report_reference
    if reference_field == "digest":
        changed_reference = MetricReportReferenceV1.model_validate(
            {
                **reference.model_dump(mode="python"),
                "canonical_digest_sha256": "e" * 64,
            }
        )
        mutated = _with_readdressed_metric_reference(replay, changed_reference)
    elif reference_field == "byte_length":
        changed_reference = MetricReportReferenceV1.model_validate(
            {
                **reference.model_dump(mode="python"),
                "canonical_byte_length": reference.canonical_byte_length + 1,
            }
        )
        mutated = _with_readdressed_metric_reference(replay, changed_reference)
    else:
        changed_reference = reference.model_copy(
            update={"report_artifact_id": "episode-001:wrong-report-artifact"}
        )
        mutated = replay.model_copy(
            update={"metric_report_reference": changed_reference}
        )

    with pytest.raises(ValueError):
        validate_metric_report_artifact_against_replay_v1(
            complete_bundle.metric_report_artifact,
            mutated,
        )


def test_metric_reference_changes_only_outer_replay_identity(
    complete_bundle: ReplayBundleV1,
) -> None:
    replay = complete_bundle.replay
    changed_reference = MetricReportReferenceV1.model_validate(
        {
            **replay.metric_report_reference.model_dump(mode="python"),
            "canonical_byte_length": (
                replay.metric_report_reference.canonical_byte_length + 1
            ),
        }
    )
    stale_outer = replay.model_copy(
        update={"metric_report_reference": changed_reference}
    )
    with pytest.raises(ValueError):
        validate_replay_artifact_v1(stale_outer)

    readdressed = _with_readdressed_metric_reference(replay, changed_reference)

    assert readdressed.trajectory_content_digest_sha256 == (
        replay.trajectory_content_digest_sha256
    )
    assert readdressed.canonical_digest_sha256 != replay.canonical_digest_sha256
    validate_replay_artifact_v1(readdressed)


def test_trajectory_digest_covers_every_owned_semantic_domain(
    complete_bundle: ReplayBundleV1,
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    base = complete_bundle.replay
    changed_runtime = RuntimeProvenanceV1.model_validate(
        {
            **runtime_provenance.model_dump(mode="python"),
            "device": "different-cpu",
        }
    )
    runtime_variant = _with_recomputed_digests(
        base,
        header=_rebuild_header(
            base.header,
            runtime_provenance=changed_runtime,
        ),
    )
    wrapper_variant = _with_recomputed_digests(
        base,
        header=_rebuild_header(
            base.header,
            wrapper_stack=(_wrapper(0),),
        ),
    )
    changed_context = EvaluationEpisodeContextV1.model_validate(
        {
            **base.header.context.model_dump(mode="python"),
            "aggregation_keys": (
                *base.header.context.aggregation_keys,
                AggregationKeyV1(name="variant", value="changed"),
            ),
        }
    )
    context_variant = _with_recomputed_digests(
        base,
        header=_rebuild_header(
            base.header,
            context=changed_context,
            context_digest_sha256=canonical_digest_sha256(changed_context),
        ),
    )
    changed_processing = EvaluationProcessingStatusV1(
        status="failed",
        processed_transition_count=2,
        failure=EvaluationProcessingFailureV1(
            stage="reducer_finalize",
            code="intentional_failure",
            reducer_id="test.reducer",
            reducer_version=1,
            detail="intentional failure",
        ),
    )
    processing_variant = _with_recomputed_digests(
        base,
        processing_status=changed_processing,
    )
    shifted_frames = tuple(
        _with_simulator_epoch(
            frame,
            base.header.context,
            frame.simulator_step_count + 10,
        )
        for frame in base.frames
    )
    shifted_transitions = tuple(
        EvaluationTransitionV1.model_validate(
            {
                **transition.model_dump(mode="python"),
                "facts": transition.facts.model_copy(
                    update={
                        "transition_start_step_count": (
                            transition.facts.transition_start_step_count + 10
                        )
                    }
                ),
            }
        )
        for transition in base.transitions
    )
    frame_variant = _with_recomputed_digests(
        base,
        frames=shifted_frames,
        transitions=shifted_transitions,
    )
    base_transition = base.transitions[0]
    base_acceptance = base_transition.facts.action_acceptance_facts
    changed_submitted_action = base_acceptance.submitted_joint_action.model_copy(
        update={"move": (-1, *base_acceptance.submitted_joint_action.move[1:])}
    )
    changed_acceptance = base_acceptance.model_copy(
        update={
            "submitted_joint_action": changed_submitted_action,
            "submitted_action_tuple_is_out_of_domain_by_actor": (
                True,
                *base_acceptance.submitted_action_tuple_is_out_of_domain_by_actor[1:],
            ),
        }
    )
    changed_facts = base_transition.facts.model_copy(
        update={"action_acceptance_facts": changed_acceptance}
    )
    changed_transition = EvaluationTransitionV1.model_validate(
        {
            **base_transition.model_dump(mode="python"),
            "facts": changed_facts,
            "events": decode_evaluation_events_v1(
                base.header.context,
                base.frames[0],
                changed_facts,
                base.frames[1],
            ),
        }
    )
    transition_variant = _with_recomputed_digests(
        base,
        transitions=(changed_transition, base.transitions[1]),
    )
    partial = _build_bundle(
        captured_evaluation_trajectory(
            transition_count=1,
            expected_horizon=2,
        ),
        runtime_provenance,
        completion_state="partial",
        end_or_failure_reason="prefix",
    ).replay
    changed_completion = EvaluationEpisodeCompletionV1.model_validate(
        {
            **partial.completion.model_dump(mode="python"),
            "completion_state": "interrupted",
        }
    )
    completion_variant = _with_recomputed_digests(
        partial,
        completion=changed_completion,
    )

    variants = {
        "runtime": (base, runtime_variant),
        "wrapper": (base, wrapper_variant),
        "context": (base, context_variant),
        "completion": (partial, completion_variant),
        "processing": (base, processing_variant),
        "frame": (base, frame_variant),
        "transition": (base, transition_variant),
    }
    for original, changed in variants.values():
        validate_replay_artifact_v1(changed)
        assert changed.trajectory_content_digest_sha256 != (
            original.trajectory_content_digest_sha256
        )


@pytest.mark.parametrize("subtype_location", ("root", "private", "nested"))
def test_replay_validation_rejects_undeclared_model_subtypes(
    complete_bundle: ReplayBundleV1,
    subtype_location: str,
) -> None:
    replay = complete_bundle.replay
    if subtype_location == "root":
        mutated = _ReplayArtifactSubtype.model_validate(
            replay.model_dump(mode="python")
        )
    elif subtype_location == "private":
        private_runtime = _PrivateRuntimeProvenance.model_validate(
            replay.header.runtime_provenance.model_dump(mode="python")
        )
        mutated = replay.model_copy(
            update={
                "header": replay.header.model_copy(
                    update={"runtime_provenance": private_runtime}
                )
            }
        )
    else:
        unfrozen_frame = _UnfrozenFrame.model_validate(
            replay.frames[0].model_dump(mode="python")
        )
        mutated = replay.model_copy(
            update={"frames": (unfrozen_frame, *replay.frames[1:])}
        )

    with pytest.raises(ValueError):
        validate_replay_artifact_v1(mutated)


def test_replay_root_rejects_invalid_processing_progress_combinations(
    complete_bundle: ReplayBundleV1,
) -> None:
    replay = complete_bundle.replay
    statuses = (
        EvaluationProcessingStatusV1(
            status="succeeded",
            processed_transition_count=1,
        ),
        EvaluationProcessingStatusV1(
            status="failed",
            processed_transition_count=3,
            failure=EvaluationProcessingFailureV1(
                stage="reducer_finalize",
                code="invalid_progress",
                reducer_id="test.reducer",
                reducer_version=1,
                detail="invalid progress",
            ),
        ),
        EvaluationProcessingStatusV1(
            status="failed",
            processed_transition_count=2,
            failure=EvaluationProcessingFailureV1(
                stage="reducer_advance",
                code="invalid_progress",
                reducer_id="test.reducer",
                reducer_version=1,
                attempted_transition_index=2,
                detail="invalid progress",
            ),
        ),
        EvaluationProcessingStatusV1(
            status="failed",
            processed_transition_count=0,
            failure=EvaluationProcessingFailureV1(
                stage="reducer_initialize",
                code="invalid_progress",
                reducer_id="test.reducer",
                reducer_version=1,
                detail="invalid progress",
            ),
        ),
    )

    for status in statuses:
        mutated = _with_recomputed_digests(
            replay,
            processing_status=status,
        )
        with pytest.raises(ValueError):
            validate_replay_artifact_v1(mutated)
