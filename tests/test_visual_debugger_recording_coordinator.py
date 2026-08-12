"""Focused same-process recording-to-replay coordinator proofs."""

from pathlib import Path
from typing import cast

from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
)
from scripts.dev.visual_debugger.protocol import (
    CommandRequestV1,
    CommandResponseV2,
    FinishAndReviewCommandV1,
    SetPresetCommandV1,
)
from scripts.dev.visual_debugger.recording import (
    DebuggerReplayRecorderV1,
    build_debugger_recording_specification_v1,
)
from scripts.dev.visual_debugger.recording_coordinator import (
    RecordingDebuggerCoordinator,
)
from scripts.dev.visual_debugger.replay_protocol import (
    ReplayViewerFrameV1,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.service import DebuggerService
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_io import (
    preflight_replay_bundle_destination_v1,
)


def _runtime_provenance() -> RuntimeProvenanceV1:
    return RuntimeProvenanceV1(
        python_version="3.14.0",
        package_version="0.1.0",
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


def _coordinator(tmp_path: Path) -> RecordingDebuggerCoordinator:
    debug_launch = debugger_test_launch_specification()
    launch = build_debugger_evaluation_launch_specification_v1(
        root_seed=debug_launch.root_seed,
        code_revision=debug_launch.code_revision,
        capture_profile="evaluation_metric_complete",
    )
    session = create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=launch,
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    recorder = DebuggerReplayRecorderV1(
        specification=build_debugger_recording_specification_v1(
            action_source_kind="manual",
            runtime_provenance=_runtime_provenance(),
        ),
        destination=preflight_replay_bundle_destination_v1(
            tmp_path / "coordinator.marlbg-replay.json"
        ),
        context=session.evaluation_context,
        initial_frame=session.current_evaluation_frame,
    )
    service = DebuggerService(
        session,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        session_id="recording-coordinator-test",
        recorder=recorder,
    )
    return RecordingDebuggerCoordinator(service)


def _live_request(
    command_id: str,
    *,
    base_revision: int,
    command: object,
) -> CommandRequestV1:
    return CommandRequestV1(
        client_id="client-a",
        command_id=command_id,
        base_revision=base_revision,
        command=command,  # pyright: ignore[reportArgumentType]
    )


def test_presentation_command_keeps_the_exact_live_binding(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    initial = coordinator.router.snapshot()

    result = coordinator.apply_command(
        _live_request(
            "preset",
            base_revision=0,
            command=SetPresetCommandV1(preset="debug"),
        )
    )

    assert result.replay_handoff is None
    assert isinstance(result.payload, CommandResponseV2)
    assert result.payload.frame.preset == "debug"
    assert coordinator.router.snapshot().generation == 0
    assert coordinator.router.snapshot().service is initial.service
    assert coordinator.router.snapshot().binding is initial.binding


def test_finish_installs_replay_before_return_and_starts_settled_at_zero(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)

    result = coordinator.apply_command(
        _live_request(
            "finish",
            base_revision=0,
            command=FinishAndReviewCommandV1(),
        )
    )
    replay = coordinator.router.snapshot()

    assert result.replay_handoff is replay.service
    assert isinstance(result.payload, CommandResponseV2)
    assert result.payload.frame.recording is not None
    assert result.payload.frame.recording.lifecycle == "reviewing"
    assert replay.generation == 1
    assert replay.binding.mode == "replay"
    frame = replay.binding.current_frame()
    typed_frame = cast(ReplayViewerFrameV1, frame)
    timeline = replay.binding.current_timeline
    assert timeline is not None
    assert typed_frame.cursor.frame_index == 0
    assert timeline().timeline_id == typed_frame.timeline_id  # pyright: ignore[reportAttributeAccessIssue]

    assert replay.binding.apply_command is not None


def test_duplicate_finish_does_not_reinstall_or_advance_router(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    request = _live_request(
        "finish",
        base_revision=0,
        command=FinishAndReviewCommandV1(),
    )
    first = coordinator.apply_command(request)
    installed = coordinator.router.snapshot()

    duplicate = coordinator.service.apply_command(request)

    assert first.replay_handoff is installed.service
    assert duplicate.replay_handoff is None
    assert duplicate.outcome == "response"
    assert isinstance(duplicate.payload, CommandResponseV2)
    assert duplicate.payload.result == "duplicate"
    assert coordinator.router.snapshot() == installed


def test_graceful_close_maps_host_only_save_outcome(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)

    result = coordinator.graceful_close()

    assert result.exit_code == 0
    assert result.message is not None
    assert "saved at" in result.message
    assert coordinator.service.recording_status is not None
    assert coordinator.service.recording_status.completion_state == "interrupted"
