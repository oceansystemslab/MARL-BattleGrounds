"""Lifecycle, atomic-byte, and path-boundary proofs for debugger recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from scripts.dev.visual_debugger import recording
from scripts.dev.visual_debugger.evaluation_bridge import (
    DebuggerActionSourceKindV1,
)
from scripts.dev.visual_debugger.recording import (
    DebuggerRecordingCloseCauseV1,
    DebuggerRecordingSpecificationV1,
    DebuggerReplayRecorderV1,
    build_debugger_recording_specification_v1,
)
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    captured_team_deathmatch_threshold_trajectory,
    evaluation_env_config,
)

from marl_battlegrounds.evaluation.metrics import (
    EvaluationEpisodeCompletionV1,
    EvaluationEpisodeObserverV1,
    EvaluationMetricReducerStateV1,
    EvaluationMetricReportV1,
    EvaluationProcessingStatusV1,
    EvaluationTransitionViewV1,
    SufficientStatisticDraftV1,
)
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
)
from marl_battlegrounds.evaluation.replay import (
    ReplayBundleV1,
    ReplayWrapperMetadataV1,
    RuntimeProvenanceV1,
)
from marl_battlegrounds.evaluation.replay_io import (
    LoadedReplayBundleV1,
    PreparedReplayBundleV1,
    ReplayBundleDestinationV1,
    ReplayLoadError,
    ReplaySaveError,
    SavedReplayBundleV1,
    preflight_replay_bundle_destination_v1,
)

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


class _FailingReducerState(EvaluationMetricReducerStateV1):
    """Minimal frozen state for the injected nonterminal processing failure."""


@dataclass(slots=True)
class _FailingAdvanceReducer:
    """Pure test reducer that fails after CP2 validation commits the unit."""

    reducer_id: str = "test.recording-failure"
    reducer_version: int = 1

    def initialize(
        self,
        context: EvaluationEpisodeContextV1,
        initial_frame: EvaluationFrameV1,
    ) -> EvaluationMetricReducerStateV1:
        del context, initial_frame
        return _FailingReducerState(
            reducer_id=self.reducer_id,
            reducer_version=self.reducer_version,
        )

    def advance(
        self,
        previous_state: EvaluationMetricReducerStateV1,
        view: EvaluationTransitionViewV1,
    ) -> EvaluationMetricReducerStateV1:
        del previous_state, view
        raise RuntimeError("injected reducer advance failure")

    def finalize(
        self,
        state: EvaluationMetricReducerStateV1,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        del state, completion, processing_status
        return ()


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


def _specification(
    *,
    action_source_kind: DebuggerActionSourceKindV1 = "manual",
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
) -> DebuggerRecordingSpecificationV1:
    return build_debugger_recording_specification_v1(
        action_source_kind=action_source_kind,
        runtime_provenance=_runtime_provenance(),
        wrapper_stack=wrapper_stack,
    )


def _trajectory(
    *,
    transition_count: int,
    expected_horizon: int,
    episode_id: str = "recording-episode",
    action_source_kind: DebuggerActionSourceKindV1 = "manual",
    maximum_episode_steps: int = 100,
) -> CapturedEvaluationTrajectory:
    return captured_evaluation_trajectory(
        transition_count=transition_count,
        expected_horizon=expected_horizon,
        episode_id=episode_id,
        capture_profile="evaluation_metric_complete",
        aggregation_keys=(
            AggregationKeyV1(name="action_source", value=action_source_kind),
        ),
        config=evaluation_env_config(max_steps=maximum_episode_steps),
    )


def _destination(tmp_path: Path, *, stem: str = "episode") -> ReplayBundleDestinationV1:
    return preflight_replay_bundle_destination_v1(
        tmp_path / f"{stem}.marlbg-replay.json"
    )


def _recorder(
    tmp_path: Path,
    trajectory: CapturedEvaluationTrajectory,
    *,
    action_source_kind: DebuggerActionSourceKindV1 = "manual",
    stem: str = "episode",
) -> DebuggerReplayRecorderV1:
    return DebuggerReplayRecorderV1(
        specification=_specification(action_source_kind=action_source_kind),
        destination=_destination(tmp_path, stem=stem),
        context=trajectory.context,
        initial_frame=trajectory.frames[0],
    )


def _append_all(
    recorder: DebuggerReplayRecorderV1,
    trajectory: CapturedEvaluationTrajectory,
) -> None:
    for transition, frame in zip(
        trajectory.transitions,
        trajectory.frames[1:],
        strict=True,
    ):
        recorder.append(transition, frame)


def test_recording_specification_is_frozen_content_addressed_and_path_free(
    tmp_path: Path,
) -> None:
    """Scientific identity changes with source/wrappers, never destinations."""
    wrapper = ReplayWrapperMetadataV1(
        position=0,
        wrapper_id="debugger-manual-adapter",
        wrapper_version=1,
        configuration_digest_sha256=_DIGEST_A,
    )
    manual = _specification(wrapper_stack=(wrapper,))
    scripted = _specification(action_source_kind="scripted", wrapper_stack=(wrapper,))

    assert manual.capture_profile == "evaluation_metric_complete"
    assert manual.recording_content_digest_sha256 != (
        scripted.recording_content_digest_sha256
    )
    assert manual == DebuggerRecordingSpecificationV1.model_validate_json(
        manual.model_dump_json()
    )
    payload = manual.model_dump(mode="json")
    assert tuple(payload) == (
        "schema_id",
        "schema_version",
        "specification_id",
        "recording_content_digest_sha256",
        "canonical_digest_sha256",
        "action_source_kind",
        "capture_profile",
        "runtime_provenance",
        "wrapper_stack",
    )
    assert str(tmp_path) not in manual.model_dump_json()
    assert "capability_token" not in manual.model_dump_json()
    with pytest.raises(ValueError, match="gap-free"):
        _specification(
            wrapper_stack=(wrapper.model_copy(update={"position": 1}),),
        )


def test_constructor_requires_retaining_profile_action_source_and_package_join(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(transition_count=0, expected_horizon=2)
    destination = _destination(tmp_path)

    wrong_profile = trajectory.context.model_copy(update={"capture_profile": "debug"})
    with pytest.raises(ValueError, match="metric-complete"):
        DebuggerReplayRecorderV1(
            specification=_specification(),
            destination=destination,
            context=wrong_profile,
            initial_frame=trajectory.frames[0],
        )
    with pytest.raises(ValueError, match="action source"):
        DebuggerReplayRecorderV1(
            specification=_specification(action_source_kind="scripted"),
            destination=destination,
            context=trajectory.context,
            initial_frame=trajectory.frames[0],
        )
    wrong_runtime = _runtime_provenance().model_copy(update={"package_version": "9"})
    with pytest.raises(ValueError, match="package version"):
        DebuggerReplayRecorderV1(
            specification=build_debugger_recording_specification_v1(
                action_source_kind="manual",
                runtime_provenance=wrong_runtime,
            ),
            destination=destination,
            context=trajectory.context,
            initial_frame=trajectory.frames[0],
        )


def test_status_preview_is_pure_and_covers_active_closeout_and_failure_codes(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(transition_count=1, expected_horizon=2)
    recorder = _recorder(tmp_path, trajectory)
    initial = recorder.status

    active = recorder.preview_status_after_append_v1(trajectory.transitions[0])
    assert active.lifecycle == "recording"
    assert active.captured_transition_count == 1
    assert active.restart_fenced is True
    for lifecycle in ("saved", "reviewing"):
        status = recorder.preview_status_v1(
            captured_transition_count=1,
            lifecycle=lifecycle,
            close_cause="finish_and_review",
        )
        assert status.completion_state == "partial"
        assert status.completion_reason == "user_finish_and_review"
    for error_code in (
        "target_unavailable",
        "publication_failed",
        "verification_failed",
    ):
        status = recorder.preview_status_v1(
            captured_transition_count=1,
            lifecycle="persistence_failed",
            close_cause="user_exit",
            persistence_error_code=error_code,
        )
        assert status.retry_available is True
        assert status.persistence_error_code == error_code

    assert recorder.status == initial
    assert recorder.validated_transition_count == 0


@pytest.mark.parametrize(
    ("transition_count", "expected_horizon"),
    ((0, 3), (2, 3)),
)
def test_finish_and_review_saves_exact_zero_or_nonzero_prefix(
    tmp_path: Path,
    transition_count: int,
    expected_horizon: int,
) -> None:
    trajectory = _trajectory(
        transition_count=transition_count,
        expected_horizon=expected_horizon,
    )
    recorder = _recorder(tmp_path, trajectory)
    _append_all(recorder, trajectory)

    assert recorder.finalize_and_save("finish_and_review") == "saved"
    bundle = recorder.bundle
    loaded = recorder.verified_loaded_bundle
    assert bundle is not None
    assert loaded is not None
    assert bundle.replay.completion.completion_state == "partial"
    assert bundle.replay.completion.end_or_failure_reason == ("user_finish_and_review")
    assert len(bundle.replay.frames) == transition_count + 1
    assert len(bundle.replay.transitions) == transition_count
    assert loaded.replay == bundle.replay
    assert recorder.status.review_available is True


def test_terminal_and_horizon_endpoints_finalize_complete(tmp_path: Path) -> None:
    terminal_trajectory = captured_team_deathmatch_threshold_trajectory(
        episode_id="terminal-recording",
        aggregation_keys=(AggregationKeyV1(name="action_source", value="manual"),),
    )
    terminal_recorder = _recorder(tmp_path, terminal_trajectory, stem="terminal")
    terminal = terminal_trajectory.transitions[0]
    preview = terminal_recorder.preview_status_after_append_v1(terminal)
    assert preview.lifecycle == "sealed"
    assert preview.completion_state == "complete"
    terminal_recorder.append(terminal, terminal_trajectory.frames[1])
    assert terminal_recorder.lifecycle == "sealed"
    assert terminal_recorder.finalize_and_save("endpoint") == "saved"
    terminal_bundle = terminal_recorder.bundle
    assert terminal_bundle is not None
    assert terminal_bundle.replay.completion.completion_bases == ("task_terminal",)
    assert terminal_bundle.replay.completion.end_or_failure_reason == (
        "team_deathmatch_score_threshold"
    )

    horizon_trajectory = _trajectory(
        transition_count=2,
        expected_horizon=2,
        episode_id="horizon-recording",
        maximum_episode_steps=2,
    )
    horizon_recorder = _recorder(tmp_path, horizon_trajectory, stem="horizon")
    _append_all(horizon_recorder, horizon_trajectory)
    assert horizon_recorder.lifecycle == "sealed"
    assert horizon_recorder.finalize_and_save("endpoint") == "saved"
    horizon_bundle = horizon_recorder.bundle
    assert horizon_bundle is not None
    assert horizon_bundle.replay.completion.completion_bases == ("declared_horizon",)
    assert horizon_bundle.replay.completion.truncated is True


def test_truncation_at_exact_horizon_remains_complete_with_declared_basis(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(
        transition_count=1,
        expected_horizon=1,
        episode_id="horizon-truncation-recording",
        maximum_episode_steps=1,
    )
    recorder = _recorder(tmp_path, trajectory, stem="horizon-truncation")
    truncated = trajectory.transitions[0]
    assert truncated.truncated is True
    assert truncated.owning_task_end_reason is None

    preview = recorder.preview_status_after_append_v1(truncated)
    assert preview.completion_state == "complete"
    assert preview.completion_reason is None
    recorder.append(truncated, trajectory.frames[1])
    assert recorder.finalize_and_save("endpoint") == "saved"
    bundle = recorder.bundle
    assert bundle is not None
    completion = bundle.replay.completion
    assert completion.completion_state == "complete"
    assert completion.truncated is True
    assert completion.completion_bases == ("declared_horizon",)
    assert completion.end_or_failure_reason is None


@pytest.mark.parametrize(
    ("close_cause", "expected_origin"),
    (
        ("simulation_failure", "simulation"),
        ("policy_failure", "policy"),
        ("capture_failure", "capture"),
        ("validation_failure", "validation"),
    ),
)
def test_failure_closeouts_preserve_origin_and_stable_reason(
    tmp_path: Path,
    close_cause: DebuggerRecordingCloseCauseV1,
    expected_origin: str,
) -> None:
    trajectory = _trajectory(
        transition_count=0,
        expected_horizon=2,
        episode_id=f"{expected_origin}-failure",
    )
    recorder = _recorder(tmp_path, trajectory, stem=expected_origin)

    assert recorder.finalize_and_save(close_cause) == "saved"
    bundle = recorder.bundle
    assert bundle is not None
    completion = bundle.replay.completion
    assert completion.completion_state == "failed"
    assert completion.failure_origin == expected_origin
    assert completion.end_or_failure_reason == close_cause


@pytest.mark.parametrize(
    ("close_cause", "expected_reason"),
    (("user_exit", "user_exit"), ("keyboard_interrupt", "keyboard_interrupt")),
)
def test_graceful_shutdown_causes_save_interrupted_prefix(
    tmp_path: Path,
    close_cause: DebuggerRecordingCloseCauseV1,
    expected_reason: str,
) -> None:
    trajectory = _trajectory(
        transition_count=1,
        expected_horizon=3,
        episode_id=f"{expected_reason}-episode",
    )
    recorder = _recorder(tmp_path, trajectory, stem=expected_reason)
    _append_all(recorder, trajectory)

    assert recorder.finalize_and_save(close_cause) == "saved"
    bundle = recorder.bundle
    assert bundle is not None
    assert bundle.replay.completion.completion_state == "interrupted"
    assert bundle.replay.completion.end_or_failure_reason == expected_reason


def test_finalize_builds_and_prepares_once_and_review_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory(transition_count=0, expected_horizon=2)
    recorder = _recorder(tmp_path, trajectory)
    actual_build = recording.build_replay_bundle_v1
    actual_prepare = recording.prepare_replay_bundle_v1
    counts = {"build": 0, "prepare": 0}

    def tracked_build(
        observer: EvaluationEpisodeObserverV1,
        report: EvaluationMetricReportV1,
        *,
        runtime_provenance: RuntimeProvenanceV1,
        wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
    ) -> ReplayBundleV1:
        counts["build"] += 1
        return actual_build(
            observer,
            report,
            runtime_provenance=runtime_provenance,
            wrapper_stack=wrapper_stack,
        )

    def tracked_prepare(bundle: ReplayBundleV1) -> PreparedReplayBundleV1:
        counts["prepare"] += 1
        return actual_prepare(bundle)

    monkeypatch.setattr(recording, "build_replay_bundle_v1", tracked_build)
    monkeypatch.setattr(recording, "prepare_replay_bundle_v1", tracked_prepare)

    assert recorder.finalize_and_save("finish_and_review") == "saved"
    prepared = recorder.prepared_bundle
    assert recorder.finalize_and_save("finish_and_review") == "saved"
    assert recorder.prepared_bundle is prepared
    assert counts == {"build": 1, "prepare": 1}
    loaded = recorder.begin_review()
    assert recorder.begin_review() is loaded
    assert recorder.lifecycle == "reviewing"


def test_one_shot_bundle_build_failure_resumes_without_refinalizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory(transition_count=1, expected_horizon=3)
    recorder = _recorder(tmp_path, trajectory)
    _append_all(recorder, trajectory)
    actual_finalize = EvaluationEpisodeObserverV1.finalize
    actual_build = recording.build_replay_bundle_v1
    calls = {"finalize": 0, "build": 0}

    def tracked_finalize(
        observer: EvaluationEpisodeObserverV1,
        *,
        completion_state: str,
        end_or_failure_reason: str | None = None,
        failure_origin: str | None = None,
    ) -> EvaluationMetricReportV1:
        calls["finalize"] += 1
        return actual_finalize(
            observer,
            completion_state=completion_state,  # type: ignore[arg-type]
            end_or_failure_reason=end_or_failure_reason,
            failure_origin=failure_origin,  # type: ignore[arg-type]
        )

    def flaky_build(
        observer: EvaluationEpisodeObserverV1,
        report: EvaluationMetricReportV1,
        *,
        runtime_provenance: RuntimeProvenanceV1,
        wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
    ) -> ReplayBundleV1:
        calls["build"] += 1
        if calls["build"] == 1:
            raise ValueError("injected bundle construction failure")
        return actual_build(
            observer,
            report,
            runtime_provenance=runtime_provenance,
            wrapper_stack=wrapper_stack,
        )

    monkeypatch.setattr(EvaluationEpisodeObserverV1, "finalize", tracked_finalize)
    monkeypatch.setattr(recording, "build_replay_bundle_v1", flaky_build)

    assert recorder.finalize_and_save("finish_and_review") == "persistence_failed"
    assert recorder.lifecycle == "persistence_failed"
    assert recorder.persistence_error_code == "publication_failed"
    assert recorder.observer_lifecycle_state == "finalized"
    assert recorder.bundle is None
    assert recorder.prepared_bundle is None
    assert recorder.validated_transition_count == 1

    assert recorder.retry_save() == "saved"
    assert calls == {"finalize": 1, "build": 2}
    bundle = cast(ReplayBundleV1, recorder.bundle)
    assert len(bundle.replay.frames) == 2
    assert len(bundle.replay.transitions) == 1


def test_one_shot_byte_preparation_failure_reuses_cached_bundle_and_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory(transition_count=1, expected_horizon=3)
    recorder = _recorder(tmp_path, trajectory)
    _append_all(recorder, trajectory)
    actual_finalize = EvaluationEpisodeObserverV1.finalize
    actual_build = recording.build_replay_bundle_v1
    actual_prepare = recording.prepare_replay_bundle_v1
    calls = {"finalize": 0, "build": 0, "prepare": 0}

    def tracked_finalize(
        observer: EvaluationEpisodeObserverV1,
        *,
        completion_state: str,
        end_or_failure_reason: str | None = None,
        failure_origin: str | None = None,
    ) -> EvaluationMetricReportV1:
        calls["finalize"] += 1
        return actual_finalize(
            observer,
            completion_state=completion_state,  # type: ignore[arg-type]
            end_or_failure_reason=end_or_failure_reason,
            failure_origin=failure_origin,  # type: ignore[arg-type]
        )

    def tracked_build(
        observer: EvaluationEpisodeObserverV1,
        report: EvaluationMetricReportV1,
        *,
        runtime_provenance: RuntimeProvenanceV1,
        wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
    ) -> ReplayBundleV1:
        calls["build"] += 1
        return actual_build(
            observer,
            report,
            runtime_provenance=runtime_provenance,
            wrapper_stack=wrapper_stack,
        )

    def flaky_prepare(bundle: ReplayBundleV1) -> PreparedReplayBundleV1:
        calls["prepare"] += 1
        if calls["prepare"] == 1:
            raise ReplaySaveError(
                "temporary_write_failed",
                path=None,
                detail="injected byte preparation failure",
            )
        return actual_prepare(bundle)

    monkeypatch.setattr(EvaluationEpisodeObserverV1, "finalize", tracked_finalize)
    monkeypatch.setattr(recording, "build_replay_bundle_v1", tracked_build)
    monkeypatch.setattr(recording, "prepare_replay_bundle_v1", flaky_prepare)

    assert recorder.finalize_and_save("finish_and_review") == "persistence_failed"
    cached_bundle = recorder.bundle
    assert cached_bundle is not None
    assert recorder.prepared_bundle is None
    assert recorder.status.retry_available is True

    assert recorder.retry_save() == "saved"
    assert calls == {"finalize": 1, "build": 1, "prepare": 2}
    assert recorder.bundle is cached_bundle
    prepared = cast(PreparedReplayBundleV1, recorder.prepared_bundle)
    assert prepared.bundle is cached_bundle
    assert len(prepared.bundle.replay.frames) == 2
    assert len(prepared.bundle.replay.transitions) == 1


def test_materialization_failure_can_resume_through_save_as(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory(transition_count=1, expected_horizon=3)
    recorder = _recorder(tmp_path, trajectory)
    _append_all(recorder, trajectory)
    actual_build = recording.build_replay_bundle_v1
    calls = 0

    def flaky_build(
        observer: EvaluationEpisodeObserverV1,
        report: EvaluationMetricReportV1,
        *,
        runtime_provenance: RuntimeProvenanceV1,
        wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
    ) -> ReplayBundleV1:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("injected one-shot build failure")
        return actual_build(
            observer,
            report,
            runtime_provenance=runtime_provenance,
            wrapper_stack=wrapper_stack,
        )

    monkeypatch.setattr(recording, "build_replay_bundle_v1", flaky_build)
    assert recorder.finalize_and_save("finish_and_review") == "persistence_failed"
    assert recorder.prepared_bundle is None

    assert recorder.save_as("alternate.marlbg-replay.json") == "saved"
    saved = cast(SavedReplayBundleV1, recorder.saved_bundle)
    prepared = cast(PreparedReplayBundleV1, recorder.prepared_bundle)
    assert calls == 2
    assert saved.replay_path.name == "alternate.marlbg-replay.json"
    assert saved.replay_path.read_bytes() == prepared.replay_json_bytes


def test_materialization_failure_can_resume_through_recovery_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory(transition_count=1, expected_horizon=3)
    recorder = _recorder(tmp_path, trajectory)
    _append_all(recorder, trajectory)
    actual_prepare = recording.prepare_replay_bundle_v1
    calls = 0

    def flaky_prepare(bundle: ReplayBundleV1) -> PreparedReplayBundleV1:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ReplaySaveError(
                "invalid_argument",
                path=None,
                detail="injected one-shot preparation failure",
            )
        return actual_prepare(bundle)

    monkeypatch.setattr(recording, "prepare_replay_bundle_v1", flaky_prepare)
    assert recorder.finalize_and_save("keyboard_interrupt") == "persistence_failed"
    cached_bundle = recorder.bundle
    assert cached_bundle is not None
    assert recorder.prepared_bundle is None

    assert recorder.save_recovery_copy() == "saved"
    prepared = cast(PreparedReplayBundleV1, recorder.prepared_bundle)
    saved = cast(SavedReplayBundleV1, recorder.saved_bundle)
    assert calls == 2
    assert prepared.bundle is cached_bundle
    assert f".recovery-{prepared.replay_payload_sha256[:16]}" in saved.replay_path.name
    assert saved.replay_path.read_bytes() == prepared.replay_json_bytes


def test_publication_failure_retries_the_exact_cached_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory(transition_count=1, expected_horizon=3)
    recorder = _recorder(tmp_path, trajectory)
    _append_all(recorder, trajectory)
    actual_publish = recording.publish_prepared_replay_bundle_v1
    attempts = 0

    def flaky_publish(
        prepared: PreparedReplayBundleV1,
        destination: ReplayBundleDestinationV1,
        *,
        verify_existing_replay: bool = False,
    ) -> SavedReplayBundleV1:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ReplaySaveError(
                "atomic_publish_failed",
                path=destination.replay_path,
                detail="injected prepublication failure",
            )
        return actual_publish(
            prepared,
            destination,
            verify_existing_replay=verify_existing_replay,
        )

    monkeypatch.setattr(recording, "publish_prepared_replay_bundle_v1", flaky_publish)
    assert recorder.finalize_and_save("finish_and_review") == "persistence_failed"
    prepared = recorder.prepared_bundle
    assert prepared is not None
    replay_bytes = prepared.replay_json_bytes
    report_bytes = prepared.metric_report_json_bytes
    assert recorder.persistence_error_code == "publication_failed"

    assert recorder.retry_save() == "saved"
    assert recorder.prepared_bundle is prepared
    assert prepared.replay_json_bytes is replay_bytes
    assert prepared.metric_report_json_bytes is report_bytes
    assert attempts == 2


def test_uncertain_publication_retry_verifies_existing_without_republishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory(transition_count=0, expected_horizon=2)
    recorder = _recorder(tmp_path, trajectory)
    actual_publish = recording.publish_prepared_replay_bundle_v1
    verification_flags: list[bool] = []

    def uncertain_publish(
        prepared: PreparedReplayBundleV1,
        destination: ReplayBundleDestinationV1,
        *,
        verify_existing_replay: bool = False,
    ) -> SavedReplayBundleV1:
        verification_flags.append(verify_existing_replay)
        saved = actual_publish(
            prepared,
            destination,
            verify_existing_replay=verify_existing_replay,
        )
        if len(verification_flags) == 1:
            raise ReplaySaveError(
                "replay_publication_verification_failed",
                path=destination.replay_path,
                detail="injected uncertain publication result",
            )
        return saved

    monkeypatch.setattr(
        recording,
        "publish_prepared_replay_bundle_v1",
        uncertain_publish,
    )
    assert recorder.finalize_and_save("finish_and_review") == "persistence_failed"
    assert recorder.persistence_error_code == "verification_failed"
    assert recorder.retry_save() == "saved"
    assert verification_flags == [False, True]


def test_failed_save_as_preserves_uncertain_original_verification_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory(transition_count=0, expected_horizon=2)
    recorder = _recorder(tmp_path, trajectory)
    actual_publish = recording.publish_prepared_replay_bundle_v1
    verification_flags: list[bool] = []

    def uncertain_publish(
        prepared: PreparedReplayBundleV1,
        destination: ReplayBundleDestinationV1,
        *,
        verify_existing_replay: bool = False,
    ) -> SavedReplayBundleV1:
        verification_flags.append(verify_existing_replay)
        saved = actual_publish(
            prepared,
            destination,
            verify_existing_replay=verify_existing_replay,
        )
        if len(verification_flags) == 1:
            raise ReplaySaveError(
                "replay_publication_verification_failed",
                path=destination.replay_path,
                detail="injected uncertain original publication",
            )
        return saved

    monkeypatch.setattr(
        recording,
        "publish_prepared_replay_bundle_v1",
        uncertain_publish,
    )
    assert recorder.finalize_and_save("finish_and_review") == "persistence_failed"
    alternate_path = tmp_path / "occupied.marlbg-replay.json"
    alternate_path.write_bytes(b"occupied")

    assert recorder.save_as(alternate_path.name) == "persistence_failed"
    assert recorder.persistence_error_code == "target_unavailable"
    assert recorder.retry_save() == "saved"
    assert verification_flags == [False, True]
    assert recorder.saved_bundle is not None
    assert recorder.saved_bundle.replay_path == tmp_path / "episode.marlbg-replay.json"
    assert alternate_path.read_bytes() == b"occupied"


def test_failed_recovery_preflight_preserves_uncertain_original_verification_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory(transition_count=0, expected_horizon=2)
    recorder = _recorder(tmp_path, trajectory)
    actual_publish = recording.publish_prepared_replay_bundle_v1
    verification_flags: list[bool] = []

    def uncertain_publish(
        prepared: PreparedReplayBundleV1,
        destination: ReplayBundleDestinationV1,
        *,
        verify_existing_replay: bool = False,
    ) -> SavedReplayBundleV1:
        verification_flags.append(verify_existing_replay)
        saved = actual_publish(
            prepared,
            destination,
            verify_existing_replay=verify_existing_replay,
        )
        if len(verification_flags) == 1:
            raise ReplaySaveError(
                "replay_publication_verification_failed",
                path=destination.replay_path,
                detail="injected uncertain original publication",
            )
        return saved

    monkeypatch.setattr(
        recording,
        "publish_prepared_replay_bundle_v1",
        uncertain_publish,
    )
    assert recorder.finalize_and_save("keyboard_interrupt") == "persistence_failed"
    prepared = cast(PreparedReplayBundleV1, recorder.prepared_bundle)
    recovery_path = tmp_path / (
        f"episode.recovery-{prepared.replay_payload_sha256[:16]}.marlbg-replay.json"
    )
    recovery_path.write_bytes(b"occupied-recovery")

    assert recorder.save_recovery_copy() == "persistence_failed"
    assert recorder.persistence_error_code == "target_unavailable"
    assert recorder.retry_save() == "saved"
    assert verification_flags == [False, True]
    assert recorder.saved_bundle is not None
    assert recorder.saved_bundle.replay_path == tmp_path / "episode.marlbg-replay.json"
    assert recovery_path.read_bytes() == b"occupied-recovery"


def test_target_conflict_requires_basename_only_save_as_with_identical_bytes(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(transition_count=0, expected_horizon=2)
    recorder = _recorder(tmp_path, trajectory)
    original_path = tmp_path / "episode.marlbg-replay.json"
    original_path.write_bytes(b"do-not-overwrite")

    assert recorder.finalize_and_save("finish_and_review") == "persistence_failed"
    assert recorder.persistence_error_code == "target_unavailable"
    assert recorder.retry_save() == "persistence_failed"
    prepared = recorder.prepared_bundle
    assert prepared is not None
    with pytest.raises(ValueError, match="basename"):
        recorder.save_as("../escape.marlbg-replay.json")

    assert recorder.save_as("recovered.marlbg-replay.json") == "saved"
    saved = recorder.saved_bundle
    assert saved is not None
    assert saved.replay_path.parent == tmp_path
    assert saved.replay_path.read_bytes() == prepared.replay_json_bytes
    assert saved.metric_report_path.read_bytes() == prepared.metric_report_json_bytes
    assert original_path.read_bytes() == b"do-not-overwrite"


def test_replacement_is_nonmutating_and_discard_never_deletes_or_publishes(
    tmp_path: Path,
) -> None:
    original_trajectory = _trajectory(
        transition_count=1,
        expected_horizon=3,
        episode_id="original-recording",
    )
    replacement_trajectory = _trajectory(
        transition_count=0,
        expected_horizon=3,
        episode_id="replacement-recording",
    )
    recorder = _recorder(tmp_path, original_trajectory)
    _append_all(recorder, original_trajectory)

    replacement = recorder.replacement_for(
        replacement_trajectory.context,
        replacement_trajectory.frames[0],
    )
    assert recorder.validated_transition_count == 1
    assert replacement.validated_transition_count == 0
    assert replacement.retained_frame_count == 1
    assert replacement.specification == recorder.specification
    assert replacement.context == replacement_trajectory.context
    assert replacement.current_frame == replacement_trajectory.frames[0]
    recorder.discard()
    assert recorder.lifecycle == "discarded"
    assert not tuple(tmp_path.iterdir())


def test_append_and_replacement_perform_no_persistence_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory(transition_count=1, expected_horizon=3)
    replacement_trajectory = _trajectory(
        transition_count=0,
        expected_horizon=3,
        episode_id="replacement-no-io",
    )
    recorder = _recorder(tmp_path, trajectory)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("per-transition persistence is forbidden")

    monkeypatch.setattr(recording, "prepare_replay_bundle_v1", forbidden)
    monkeypatch.setattr(recording, "publish_prepared_replay_bundle_v1", forbidden)
    monkeypatch.setattr(recording, "load_replay_bundle_v1", forbidden)
    monkeypatch.setattr(
        recording,
        "preflight_replay_bundle_destination_v1",
        forbidden,
    )
    recorder.append(trajectory.transitions[0], trajectory.frames[1])
    replacement = recorder.replacement_for(
        replacement_trajectory.context,
        replacement_trajectory.frames[0],
    )
    assert recorder.validated_transition_count == 1
    assert replacement.validated_transition_count == 0


def test_replacement_readdresses_only_action_source_identity(
    tmp_path: Path,
) -> None:
    manual_trajectory = _trajectory(
        transition_count=0,
        expected_horizon=3,
        episode_id="manual-draft",
        action_source_kind="manual",
    )
    scripted_trajectory = _trajectory(
        transition_count=0,
        expected_horizon=3,
        episode_id="scripted-draft",
        action_source_kind="scripted",
    )
    recorder = _recorder(tmp_path, manual_trajectory)

    replacement = recorder.replacement_for(
        scripted_trajectory.context,
        scripted_trajectory.frames[0],
    )
    assert recorder.specification.action_source_kind == "manual"
    assert replacement.specification.action_source_kind == "scripted"
    assert replacement.specification.runtime_provenance == (
        recorder.specification.runtime_provenance
    )
    assert replacement.specification.wrapper_stack == (
        recorder.specification.wrapper_stack
    )
    assert replacement.specification.recording_content_digest_sha256 != (
        recorder.specification.recording_content_digest_sha256
    )
    assert replacement.validated_transition_count == 0


def test_invalid_append_preserves_prefix_and_exposes_poisoned_observer(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(transition_count=1, expected_horizon=3)
    recorder = _recorder(tmp_path, trajectory)

    with pytest.raises(ValueError):
        recorder.append(trajectory.transitions[0], trajectory.frames[0])

    assert recorder.validated_transition_count == 0
    assert recorder.retained_frame_count == 1
    assert recorder.current_frame == trajectory.frames[0]
    assert recorder.observer_lifecycle_state == "poisoned"
    assert recorder.finalize_and_save("validation_failure") == "saved"
    bundle = recorder.bundle
    assert bundle is not None
    assert bundle.replay.completion.failure_origin == "validation"


def test_nonterminal_reducer_failure_interrupts_without_rewriting_rollout_failed(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(transition_count=1, expected_horizon=3)
    recorder = DebuggerReplayRecorderV1(
        specification=_specification(),
        destination=_destination(tmp_path),
        context=trajectory.context,
        initial_frame=trajectory.frames[0],
        reducers=(_FailingAdvanceReducer(),),
    )

    with pytest.raises(RuntimeError, match="advance failed"):
        recorder.append(trajectory.transitions[0], trajectory.frames[1])

    assert recorder.validated_transition_count == 1
    assert recorder.current_frame == trajectory.frames[1]
    assert recorder.observer_lifecycle_state == "poisoned"
    preview = recorder.preview_status_v1(
        captured_transition_count=1,
        lifecycle="finalized_unsaved",
        close_cause="processing_failure",
    )
    assert preview.completion_state == "interrupted"
    assert preview.completion_reason == "evaluation_processing_failure"

    assert recorder.finalize_and_save("processing_failure") == "saved"
    bundle = recorder.bundle
    assert bundle is not None
    assert bundle.replay.completion.completion_state == "interrupted"
    assert bundle.replay.completion.failure_origin is None
    assert bundle.replay.completion.end_or_failure_reason == (
        "evaluation_processing_failure"
    )
    assert bundle.replay.processing_status.status == "failed"
    assert bundle.replay.processing_status.processed_transition_count == 0


def test_missing_parent_retry_and_save_as_recheck_the_original_parent(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "recordings"
    parent.mkdir()
    trajectory = _trajectory(transition_count=0, expected_horizon=2)
    recorder = DebuggerReplayRecorderV1(
        specification=_specification(),
        destination=_destination(parent),
        context=trajectory.context,
        initial_frame=trajectory.frames[0],
    )
    parent.rmdir()

    assert recorder.finalize_and_save("finish_and_review") == "persistence_failed"
    assert recorder.persistence_error_code == "target_unavailable"
    assert recorder.retry_save() == "persistence_failed"
    assert recorder.persistence_error_code == "target_unavailable"
    assert recorder.save_as("alternate.marlbg-replay.json") == "persistence_failed"

    parent.mkdir()
    assert recorder.retry_save() == "saved"
    saved = recorder.saved_bundle
    assert saved is not None
    assert saved.replay_path.parent == parent


def test_recovery_copy_is_digest_named_exact_and_strictly_no_clobber(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(transition_count=0, expected_horizon=2)
    recorder = _recorder(tmp_path, trajectory)
    original_path = tmp_path / "episode.marlbg-replay.json"
    original_path.write_bytes(b"occupied")
    assert recorder.finalize_and_save("keyboard_interrupt") == "persistence_failed"
    prepared = cast(PreparedReplayBundleV1, recorder.prepared_bundle)
    recovery_path = tmp_path / (
        f"episode.recovery-{prepared.replay_payload_sha256[:16]}.marlbg-replay.json"
    )
    recovery_path.write_bytes(b"existing-recovery")

    assert recorder.save_recovery_copy() == "persistence_failed"
    assert recorder.persistence_error_code == "target_unavailable"
    assert recovery_path.read_bytes() == b"existing-recovery"
    recovery_path.unlink()

    assert recorder.save_recovery_copy() == "saved"
    saved = recorder.saved_bundle
    assert saved is not None
    assert saved.replay_path == recovery_path
    assert saved.replay_path.read_bytes() == prepared.replay_json_bytes
    assert saved.metric_report_path.read_bytes() == prepared.metric_report_json_bytes


def test_public_reload_failure_preserves_prepared_bytes_for_verification_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _trajectory(transition_count=0, expected_horizon=2)
    recorder = _recorder(tmp_path, trajectory)
    actual_load = recording.load_replay_bundle_v1
    actual_publish = recording.publish_prepared_replay_bundle_v1
    loads = 0
    verification_flags: list[bool] = []

    def tracked_publish(
        prepared: PreparedReplayBundleV1,
        destination: ReplayBundleDestinationV1,
        *,
        verify_existing_replay: bool = False,
    ) -> SavedReplayBundleV1:
        verification_flags.append(verify_existing_replay)
        return actual_publish(
            prepared,
            destination,
            verify_existing_replay=verify_existing_replay,
        )

    def flaky_load(
        path: str | Path,
        *,
        require_metric_report: bool = False,
        max_file_size_bytes: int = 1024**3,
        max_json_depth: int = 128,
    ) -> LoadedReplayBundleV1:
        nonlocal loads
        loads += 1
        if loads == 1:
            raise ReplayLoadError(
                "file_read_failed",
                path=Path(path),
                detail="injected public reload failure",
            )
        return actual_load(
            path,
            require_metric_report=require_metric_report,
            max_file_size_bytes=max_file_size_bytes,
            max_json_depth=max_json_depth,
        )

    monkeypatch.setattr(recording, "publish_prepared_replay_bundle_v1", tracked_publish)
    monkeypatch.setattr(recording, "load_replay_bundle_v1", flaky_load)
    assert recorder.finalize_and_save("finish_and_review") == "persistence_failed"
    assert recorder.persistence_error_code == "verification_failed"
    prepared = cast(PreparedReplayBundleV1, recorder.prepared_bundle)
    exact_digest = prepared.replay_payload_sha256

    assert recorder.retry_save() == "saved"
    assert verification_flags == [False, True]
    assert recorder.prepared_bundle is prepared
    assert prepared.replay_payload_sha256 == exact_digest
