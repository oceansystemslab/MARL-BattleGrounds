"""Self-contained replay V2 preserves captured facts and truthful provenance."""

import json
from pathlib import Path

import pytest
from scripts.dev.visual_debugger.recording import (
    build_debugger_recording_specification_v1,
)
from scripts.dev.visual_debugger.replay_protocol import (
    ReplayCommandRequestV1,
    ReplaySetViewCommandV1,
)
from scripts.dev.visual_debugger.replay_recorder import DebuggerReplayRecorder
from scripts.dev.visual_debugger.replay_service import ReplayViewerService
from tests.evaluation_fixtures import (
    captured_team_deathmatch_threshold_trajectory,
    current_captured_evaluation_trajectory,
)
from tests.test_evaluation_replay import runtime_provenance

from marl_battlegrounds.evaluation.metrics import EvaluationEpisodeObserverV1
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    AssignedPolicySlotV2,
    CodeRevisionV2,
    EvaluationFrameV1,
    EvaluationSeedProtocolV2,
    VersionedIdentityV1,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_io import (
    PreparedReplay,
    ReplayLoadError,
    ReplaySaveError,
    load_replay,
    load_replay_artifact_v1,
    preflight_replay_destination,
    save_replay,
)
from marl_battlegrounds.evaluation.replay_v2 import build_replay_v2, context_v2


def test_v2_replay_roundtrip_needs_neither_metrics_nor_legacy_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    trajectory = captured_team_deathmatch_threshold_trajectory()
    legacy_bytes = canonical_json_bytes(trajectory.context)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("V2 capture must not compute or validate metrics")

    monkeypatch.setattr(EvaluationEpisodeObserverV1, "finalize", forbidden)
    monkeypatch.setattr(
        "marl_battlegrounds.evaluation.replay.validate_replay_artifact_v1", forbidden
    )
    replay = build_replay_v2(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
        runtime_provenance=runtime_provenance,
    )
    path = tmp_path / "episode.marlbg-replay.json"
    saved = save_replay(replay, path)
    loaded = load_replay(path)
    assert loaded.replay == replay
    assert loaded.status == "not_recorded"
    assert loaded.metric_report_artifact is None
    assert tuple(tmp_path.iterdir()) == (path,)
    assert saved.replay_byte_length == path.stat().st_size
    assert loaded.replay.frames == trajectory.frames
    assert loaded.replay.transitions == trajectory.transitions
    assert (
        loaded.replay.completion.end_or_failure_reason
        == trajectory.transitions[-1].owning_task_end_reason
    )
    assert canonical_json_bytes(trajectory.context) == legacy_bytes
    assert b"metric_report_reference" not in path.read_bytes()
    with pytest.raises(ReplayLoadError) as failure:
        load_replay_artifact_v1(path)
    assert failure.value.code == "unsupported_schema_version"
    with pytest.raises(ReplaySaveError) as conflict:
        save_replay(replay, path)
    assert conflict.value.code == "replay_target_exists"


def test_capture_preparation_reuses_validation_but_external_models_remain_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    trajectory = captured_team_deathmatch_threshold_trajectory()
    replay = build_replay_v2(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
        runtime_provenance=runtime_provenance,
    )
    external = PreparedReplay(replay)

    def repeated_validation(*args: object, **kwargs: object) -> None:
        raise AssertionError("fresh capture must not repeat tree validation")

    with monkeypatch.context() as patch:
        patch.setattr(
            "marl_battlegrounds.evaluation.replay_io.validate_declared_model_tree",
            repeated_validation,
        )
        captured = PreparedReplay._from_capture(replay)  # pyright: ignore[reportPrivateUsage]
    assert captured.replay is replay
    assert captured.replay_json_bytes == external.replay_json_bytes
    assert captured.replay_payload_sha256 == external.replay_payload_sha256

    class UndeclaredFrame(EvaluationFrameV1):
        pass

    # Its bytes and digests remain valid: only the undeclared nested model type
    # differs. Public publication must still reject that unchecked escape hatch.
    frame = UndeclaredFrame.model_validate(replay.frames[0].model_dump())
    altered = replay.model_copy(update={"frames": (frame, *replay.frames[1:])})
    assert canonical_json_bytes(altered) == external.replay_json_bytes
    with pytest.raises(ValueError, match="undeclared nested model"):
        PreparedReplay(altered)
    path = tmp_path / "invalid.marlbg-replay.json"
    with pytest.raises(ReplaySaveError) as rejected:
        save_replay(altered, path)
    assert rejected.value.code == "invalid_argument"
    assert not path.exists()


def test_external_policy_provenance_can_be_unknown_without_dummy_training_values() -> (
    None
):
    context = context_v2(captured_team_deathmatch_threshold_trajectory().context)
    row = AssignedPolicySlotV2(
        global_slot=0,
        evaluation_role="focal",
        policy_kind="external",
        policy_id="researcher-policy",
        lifecycle="evolving",
        callable_name="researcher.module:act",
    )
    updated = type(context).model_validate(
        {
            **context.model_dump(),
            "policy_assignments": (row, *context.policy_assignments[1:]),
        }
    )
    assert updated.policy_assignments[0] == row
    assert row.training_run_id is None
    assert row.training_step is None
    assert row.policy_content_digest is None
    assert row.checkpoint_digest is None


def test_installed_package_and_unknown_rng_streams_have_no_dummy_provenance() -> None:
    context = context_v2(captured_team_deathmatch_threshold_trajectory().context)
    updated = type(context).model_validate(
        {
            **context.model_dump(),
            "code_revision": CodeRevisionV2(package_version="0.1.0"),
            "seed_protocol": EvaluationSeedProtocolV2(
                seed_protocol=VersionedIdentityV1(
                    identifier="researcher-owned-rng", version=1
                ),
                root_seed=4,
            ),
        }
    )
    assert updated.code_revision.commit_sha is None
    assert updated.code_revision.is_dirty is None
    assert updated.seed_protocol.root_seed == 4
    assert updated.seed_protocol.episode_seed is None


def test_fully_known_v2_provenance_retains_its_version_on_save(
    tmp_path: Path,
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    trajectory = captured_team_deathmatch_threshold_trajectory()
    context = context_v2(trajectory.context)
    context = type(context).model_validate(
        {
            **context.model_dump(),
            "code_revision": CodeRevisionV2(
                **trajectory.context.code_revision.model_dump()
            ),
            "seed_protocol": EvaluationSeedProtocolV2(
                **trajectory.context.seed_protocol.model_dump()
            ),
        }
    )
    replay = build_replay_v2(
        context,
        trajectory.frames,
        trajectory.transitions,
        runtime_provenance=runtime_provenance,
    )
    path = tmp_path / "known.marlbg-replay.json"
    save_replay(replay, path)
    loaded = load_replay(path)
    assert type(loaded.replay.header.context.code_revision) is CodeRevisionV2
    assert type(loaded.replay.header.context.seed_protocol) is EvaluationSeedProtocolV2


@pytest.mark.parametrize("information_mode", ["no_shared_obs", "shared_obs"])
def test_v2_replay_viewer_switches_pov_without_metrics_sidecar(
    tmp_path: Path,
    runtime_provenance: RuntimeProvenanceV1,
    information_mode: str,
) -> None:
    from typing import Literal, cast

    trajectory = captured_team_deathmatch_threshold_trajectory(
        execution_information_mode=cast(
            Literal["no_shared_obs", "shared_obs"], information_mode
        )
    )
    replay = build_replay_v2(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
        runtime_provenance=runtime_provenance,
    )
    path = tmp_path / "viewer.marlbg-replay.json"
    save_replay(replay, path)
    viewer = ReplayViewerService(load_replay(path), initial_frame_index=1)
    details_bytes, filename = viewer.episode_details()
    details = json.loads(details_bytes)
    assert filename == "episode-details.json"
    assert details["context"] == replay.header.context.model_dump(mode="json")
    assert details["completion"] == replay.completion.model_dump(mode="json")
    assert "frames" not in details and "statistics" not in details
    initial = viewer.current_presentation().payload
    assert initial is not None
    assert initial.model_dump()["source"]["source_replay_schema_version"] == 2
    for index, view in enumerate(("pov", "researcher")):
        result = viewer.apply_command(
            ReplayCommandRequestV1(
                client_id="v2-viewer",
                command_id=f"switch-{index}",
                base_revision=viewer.revision,
                command=ReplaySetViewCommandV1(view_mode=view),
            )
        )
        assert result.outcome == "response"
        assert viewer.current_presentation().payload is not None
        assert viewer.episode_details()[0] == details_bytes
    assert tuple(tmp_path.iterdir()) == (path,)


def test_current_recorder_saves_captured_facts_without_metric_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    trajectory = current_captured_evaluation_trajectory(
        aggregation_keys=(AggregationKeyV1(name="action_source", value="manual"),)
    )
    path = tmp_path / "recorded.marlbg-replay.json"
    runtime = runtime_provenance.model_copy(update={"policy_execution_included": False})

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("new recordings must not construct a metric observer")

    monkeypatch.setattr(EvaluationEpisodeObserverV1, "__init__", forbidden)
    recorder = DebuggerReplayRecorder(
        specification=build_debugger_recording_specification_v1(
            action_source_kind="manual", runtime_provenance=runtime
        ),
        destination=preflight_replay_destination(path),
        context=trajectory.context,
        initial_frame=trajectory.frames[0],
    )
    recorder.append(trajectory.transitions[0], trajectory.frames[1])
    assert recorder.lifecycle == "sealed"
    assert recorder.finalize_and_save("endpoint") == "saved"
    assert recorder.begin_review().replay.schema_version == 3
    assert recorder.begin_review().replay.frames == trajectory.frames
    assert recorder.begin_review().replay.transitions == trajectory.transitions
    assert tuple(tmp_path.iterdir()) == (path,)


@pytest.mark.parametrize("conflicting_bytes", [False, True])
def test_current_recorder_retries_same_bytes_after_uncertain_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_provenance: RuntimeProvenanceV1,
    conflicting_bytes: bool,
) -> None:
    import scripts.dev.visual_debugger.replay_recorder as module

    trajectory = current_captured_evaluation_trajectory(
        aggregation_keys=(AggregationKeyV1(name="action_source", value="manual"),)
    )
    path = tmp_path / "retry.marlbg-replay.json"
    recorder = DebuggerReplayRecorder(
        specification=build_debugger_recording_specification_v1(
            action_source_kind="manual",
            runtime_provenance=runtime_provenance.model_copy(
                update={"policy_execution_included": False}
            ),
        ),
        destination=preflight_replay_destination(path),
        context=trajectory.context,
        initial_frame=trajectory.frames[0],
    )
    recorder.append(trajectory.transitions[0], trajectory.frames[1])
    actual_load = module.load_replay

    def fail_reload(*args: object, **kwargs: object) -> None:
        raise ReplayLoadError(
            "file_read_failed", path=path, detail="injected reload failure"
        )

    monkeypatch.setattr(module, "load_replay", fail_reload)
    assert recorder.finalize_and_save("endpoint") == "persistence_failed"
    prepared = recorder.prepared_replay
    published_bytes = path.read_bytes()
    monkeypatch.setattr(module, "load_replay", actual_load)
    if conflicting_bytes:
        path.write_bytes(b"another publication")
        assert recorder.retry_save() == "persistence_failed"
        assert path.read_bytes() == b"another publication"
        assert recorder.save_as("recovered.marlbg-replay.json") == "saved"
        assert (
            tmp_path / "recovered.marlbg-replay.json"
        ).read_bytes() == published_bytes
    else:
        assert recorder.retry_save() == "saved"
        assert path.read_bytes() == published_bytes
    assert recorder.prepared_replay is prepared
    assert recorder.begin_review().status == "not_recorded"


def test_linked_replay_left_after_failed_rollback_is_loud_and_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    trajectory = captured_team_deathmatch_threshold_trajectory()
    replay = build_replay_v2(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
        runtime_provenance=runtime_provenance,
    )
    path = tmp_path / "uncertain.marlbg-replay.json"

    def fail_fsync(*args: object, **kwargs: object) -> None:
        raise OSError("injected post-link durability failure")

    def leave_published_link(*args: object) -> None:
        pass

    monkeypatch.setattr(
        "marl_battlegrounds.evaluation.replay_io._fsync_directory", fail_fsync
    )
    monkeypatch.setattr(
        "marl_battlegrounds.evaluation.replay_io._rollback_published_link",
        leave_published_link,
    )
    with pytest.raises(ReplaySaveError) as failed:
        save_replay(replay, path)
    assert failed.value.code == "atomic_publish_failed"
    assert path.read_bytes() == canonical_json_bytes(replay)
    with pytest.raises(ReplaySaveError) as retry:
        save_replay(replay, path)
    assert retry.value.code == "replay_target_exists"
    assert path.read_bytes() == canonical_json_bytes(replay)


@pytest.mark.parametrize("information_mode", ["no_shared_obs", "shared_obs"])
def test_current_recording_live_command_hands_off_to_v2_viewer(
    tmp_path: Path,
    information_mode: str,
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    from typing import Literal, cast

    from scripts.dev.visual_debugger.control import create_session
    from scripts.dev.visual_debugger.evaluation_bridge import (
        build_debugger_evaluation_launch_specification_v1,
    )
    from scripts.dev.visual_debugger.protocol import (
        CommandRequestV1,
        FinishAndReviewCommandV1,
        KeyboardCommandV1,
    )
    from scripts.dev.visual_debugger.recording_coordinator import (
        RecordingDebuggerCoordinator,
    )
    from scripts.dev.visual_debugger.scenarios import get_scenario
    from scripts.dev.visual_debugger.service import DebuggerService
    from tests.visual_debugger_fixtures import debugger_test_launch_specification

    launch = debugger_test_launch_specification()
    launch = build_debugger_evaluation_launch_specification_v1(
        root_seed=launch.root_seed,
        code_revision=launch.code_revision,
        capture_profile="evaluation_metric_complete",
    )
    session = create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=launch,
        execution_information_mode=cast(
            Literal["no_shared_obs", "shared_obs"], information_mode
        ),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    path = tmp_path / "live.marlbg-replay.json"
    recorder = DebuggerReplayRecorder(
        specification=build_debugger_recording_specification_v1(
            action_source_kind="manual",
            runtime_provenance=runtime_provenance.model_copy(
                update={
                    "package_version": (
                        session.evaluation_context.code_revision.package_version
                    ),
                    "policy_execution_included": False,
                }
            ),
        ),
        destination=preflight_replay_destination(path),
        context=session.evaluation_context,
        initial_frame=session.current_evaluation_frame,
    )
    coordinator = RecordingDebuggerCoordinator(
        DebuggerService(
            session,
            view_mode="researcher",
            preset="analysis",
            include_stress=False,
            recorder=recorder,
        )
    )
    coordinator.apply_command(
        CommandRequestV1(
            client_id="v2-live",
            command_id="tick",
            base_revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )
    handoff = coordinator.apply_command(
        CommandRequestV1(
            client_id="v2-live",
            command_id="finish",
            base_revision=1,
            command=FinishAndReviewCommandV1(),
        )
    )
    assert handoff.replay_handoff is not None
    assert recorder.saved_bundle is not None
    assert recorder.begin_review().replay.schema_version == 3
    viewer = handoff.replay_handoff
    result = viewer.apply_command(
        ReplayCommandRequestV1(
            client_id="v2-live",
            command_id="pov",
            base_revision=viewer.revision,
            command=ReplaySetViewCommandV1(view_mode="pov"),
        )
    )
    assert result.outcome == "response"
    assert viewer.current_presentation().outcome == "response"
    assert tuple(tmp_path.iterdir()) == (path,)


__all__ = ["runtime_provenance"]
