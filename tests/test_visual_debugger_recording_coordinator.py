"""Focused same-process recording-to-replay coordinator proofs."""

import json
import stat
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from pathlib import Path
from threading import Thread
from typing import cast
from unittest.mock import Mock

import pytest
import scripts.dev.visual_debugger.service as debugger_service_module
from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
)
from scripts.dev.visual_debugger.presentation_protocol import (
    LiveOracleAuthorizedPresentationFrameV1,
    ReplayOracleAuthorizedPresentationFrameV1,
)
from scripts.dev.visual_debugger.protocol import (
    CommandRequestV1,
    CommandResponseV2,
    FinishAndReviewCommandV1,
    KeyboardCommandV1,
    ResearcherLiveDebuggerFrameV2,
    ReviewReplayCommandV1,
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
    ReplayCommandRequestV1,
    ReplayCommandResponseV1,
    ReplayCommandV1,
    ReplayLastFrameCommandV1,
    ReplayNextFrameCommandV1,
    ReplaySelectAgentCommandV1,
    ResearcherReplayTimelineV1,
    ResearcherReplayViewerFrameV1,
    SharedObsAgentPovReplayTimelineV1,
    SharedObsAgentPovReplayViewerFrameV1,
)
from scripts.dev.visual_debugger.replay_service import ReplayViewerService
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.server import DebuggerHTTPServer, create_server
from scripts.dev.visual_debugger.service import DebuggerService
from tests.export_visual_debugger_replay_artifacts import export_artifacts
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.evaluation.models import (
    EvaluationFrameV1,
    EvaluationTransitionV1,
)
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_io import (
    load_replay_bundle_v1,
    preflight_replay_bundle_destination_v1,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ASSET_ROOT = _REPOSITORY_ROOT / "web" / "visual_debugger"
_TOKEN_HEADER = "X-MARL-Debugger-Token"
_TOKEN = "recording-handoff-capability"


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


def _coordinator_and_recorder(
    tmp_path: Path,
    *,
    scenario_name: str = "arena_5v5",
    controlled_global_slot: int | None = None,
) -> tuple[RecordingDebuggerCoordinator, DebuggerReplayRecorderV1]:
    debug_launch = debugger_test_launch_specification()
    launch = build_debugger_evaluation_launch_specification_v1(
        root_seed=debug_launch.root_seed,
        code_revision=debug_launch.code_revision,
        capture_profile="evaluation_metric_complete",
    )
    scenario = get_scenario(scenario_name)
    session = create_session(
        scenario,
        seed=0,
        evaluation_launch_specification=launch,
        controlled_global_slot=controlled_global_slot,
        show_ranges=True,
        verbose_logging=False,
    )
    recorder = DebuggerReplayRecorderV1(
        specification=build_debugger_recording_specification_v1(
            action_source_kind=(
                "scripted" if scenario.mode == "scripted" else "manual"
            ),
            runtime_provenance=_runtime_provenance(),
        ),
        destination=preflight_replay_bundle_destination_v1(
            tmp_path / f"coordinator-{scenario_name}.marlbg-replay.json"
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
    return RecordingDebuggerCoordinator(service), recorder


def _coordinator(tmp_path: Path) -> RecordingDebuggerCoordinator:
    return _coordinator_and_recorder(tmp_path)[0]


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


def _replay_request(
    service: ReplayViewerService,
    command_id: str,
    command: ReplayCommandV1,
) -> ReplayCommandRequestV1:
    return ReplayCommandRequestV1(
        client_id="replay-client-a",
        command_id=command_id,
        base_revision=service.revision,
        command=command,
    )


def _authorized_get(
    server: DebuggerHTTPServer,
    path: str,
) -> tuple[HTTPResponse, bytes]:
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request("GET", path, headers={_TOKEN_HEADER: _TOKEN})
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response, payload


def test_presentation_command_keeps_the_exact_live_binding(tmp_path: Path) -> None:
    coordinator = _coordinator(tmp_path)
    initial = coordinator.router.snapshot()

    result = coordinator.apply_command(
        _live_request(
            "preset",
            base_revision=0,
            command=SetPresetCommandV1.model_validate({"preset": "debug"}),
        )
    )

    assert result.replay_handoff is None
    assert isinstance(result.payload, CommandResponseV2)
    assert result.payload.frame.preset == "analysis"
    assert coordinator.router.snapshot().generation == 0
    assert coordinator.router.snapshot().service is initial.service
    assert coordinator.router.snapshot().binding is initial.binding


def test_finish_installs_replay_before_return_and_starts_settled_at_zero(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path)
    live = coordinator.router.snapshot()
    assert live.binding.current_metric_report is None
    live_presentation_result = live.binding.current_presentation()
    assert live_presentation_result.outcome == "response"
    live_presentation = cast(
        LiveOracleAuthorizedPresentationFrameV1,
        live_presentation_result.payload,
    )
    assert live_presentation.presentation_kind == "live_oracle"
    assert live_presentation.source.source_frame_index == 0

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
    assert replay.binding.product_kind == "replay_viewer"
    frame = replay.binding.current_frame()
    typed_frame = cast(ResearcherReplayViewerFrameV1, frame)
    timeline = replay.binding.current_timeline
    presentation = replay.binding.current_presentation
    metric_report = replay.binding.current_metric_report
    assert timeline is not None
    assert presentation is not None
    assert metric_report is not None
    assert metric_report == replay.service.current_metric_report  # pyright: ignore[reportAttributeAccessIssue]
    assert typed_frame.cursor.frame_index == 0
    assert timeline().timeline_id == typed_frame.timeline_id  # pyright: ignore[reportAttributeAccessIssue]
    presentation_result = presentation()
    assert presentation_result.outcome == "response"
    presentation_frame = cast(
        ReplayOracleAuthorizedPresentationFrameV1,
        presentation_result.payload,
    )
    assert presentation_frame.source.source_revision == typed_frame.revision
    assert presentation_frame.source.source_frame_id == typed_frame.frame_id
    metric_result = metric_report()
    assert metric_result.outcome == "available"
    assert metric_result.payload
    assert metric_result.filename is not None
    assert metric_result.filename.endswith(".marlbg-metrics.json")

    assert replay.binding.apply_command is not None


def test_recording_handoff_http_gets_use_actual_private_shared_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = _coordinator(tmp_path)
    artifacts = export_artifacts(tmp_path / "handoff-shared-artifacts")
    shared_viewer = ReplayViewerService(
        load_replay_bundle_v1(
            Path(artifacts["shared"]),
            require_metric_report=True,
        ),
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="recording-handoff-shared-private",
    )

    def build_shared_handoff(service: DebuggerService) -> ReplayViewerService:
        assert service is coordinator.service
        return shared_viewer

    monkeypatch.setattr(
        DebuggerService,
        "_build_replay_handoff",
        build_shared_handoff,
    )
    server = create_server(
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator_router=coordinator.router,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = coordinator.apply_command(
            _live_request(
                "finish-to-shared-private",
                base_revision=0,
                command=FinishAndReviewCommandV1(),
            )
        )
        frame_response, frame_body = _authorized_get(server, "/api/frame")
        timeline_response, timeline_body = _authorized_get(
            server,
            "/api/replay/timeline",
        )
        metric_response, metric_body = _authorized_get(
            server,
            "/api/replay/metric-report",
        )
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    installed = coordinator.router.snapshot()
    assert result.replay_handoff is shared_viewer
    assert installed.service is shared_viewer
    assert installed.binding.current_frame == shared_viewer.current_frame
    assert installed.binding.current_timeline == shared_viewer.current_timeline
    assert (
        installed.binding.current_metric_report == shared_viewer.current_metric_report
    )
    assert frame_response.status == timeline_response.status == HTTPStatus.OK
    metric_result = shared_viewer.current_metric_report()
    assert metric_result.outcome == "available"
    assert metric_result.payload is not None
    assert metric_result.filename is not None
    assert metric_response.status == HTTPStatus.OK
    assert metric_body == metric_result.payload
    assert metric_response.getheader("Content-Disposition") == (
        f'attachment; filename="{metric_result.filename}"'
    )
    assert frame_response.getheader("Cache-Control") == "no-store"
    assert timeline_response.getheader("Cache-Control") == "no-store"
    assert metric_response.getheader("Cache-Control") == "no-store"

    frame = SharedObsAgentPovReplayViewerFrameV1.model_validate_json(frame_body)
    timeline = SharedObsAgentPovReplayTimelineV1.model_validate_json(timeline_body)
    assert frame.frame_kind == "shared_obs_agent_pov_replay_viewer"
    assert timeline.timeline_kind == "shared_obs_agent_pov"
    assert timeline.artifact_summary == frame.artifact_summary
    assert timeline.rows[0].recipient_frame_id == frame.recipient_frame_id
    assert frame.cursor.frame_index == 0
    assert json.loads(frame_body)["public_agent_id"] == frame.public_agent_id

    serialized = frame_body + timeline_body
    for forbidden_key in (
        b'"global_slot"',
        b'"observation_materialization"',
        b'"projection"',
        b'"selected_global_slot"',
        b'"source_material_frame_id"',
    ):
        assert forbidden_key not in serialized
    assert b"shared_obs_source_material" not in serialized


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


def test_registered_capture_round_trip_preserves_exact_researcher_presentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove one real capture survives persistence and same-process review."""
    append_units: list[tuple[str, str]] = []
    actual_append = DebuggerReplayRecorderV1.append

    def tracked_append(
        recorder: DebuggerReplayRecorderV1,
        transition: EvaluationTransitionV1,
        successor_frame: EvaluationFrameV1,
    ) -> None:
        append_units.append((transition.transition_id, successor_frame.frame_id))
        actual_append(recorder, transition, successor_frame)

    def reject_parallel_debug_observer(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("recording constructed a parallel debug observer")

    monkeypatch.setattr(DebuggerReplayRecorderV1, "append", tracked_append)
    monkeypatch.setattr(
        DebuggerService,
        "_new_evaluation_observer",
        reject_parallel_debug_observer,
    )
    replay_viewer_factory = Mock(wraps=ReplayViewerService)
    monkeypatch.setattr(
        debugger_service_module,
        "ReplayViewerService",
        replay_viewer_factory,
    )
    coordinator, recorder = _coordinator_and_recorder(
        tmp_path,
        scenario_name="basic_support",
        controlled_global_slot=2,
    )
    controlled_live = cast(
        ResearcherLiveDebuggerFrameV2,
        coordinator.service.current_frame(),
    )
    assert controlled_live.frame_index == 0
    assert coordinator.service.session.controlled_global_slot == 2
    assert (
        coordinator.service.session.pending_action.selected_global_target_slot is None
    )
    assert coordinator.service.session.pending_action.armed_lane == 0

    first_result = coordinator.apply_command(
        _live_request(
            "capture-basic-support-0",
            base_revision=0,
            command=KeyboardCommandV1(key="n"),
        )
    )
    endpoint_result = coordinator.apply_command(
        _live_request(
            "capture-basic-support-1",
            base_revision=1,
            command=KeyboardCommandV1(key="n"),
        )
    )
    assert isinstance(first_result.payload, CommandResponseV2)
    assert isinstance(endpoint_result.payload, CommandResponseV2)
    first_live = cast(ResearcherLiveDebuggerFrameV2, first_result.payload.frame)
    final_live = cast(ResearcherLiveDebuggerFrameV2, endpoint_result.payload.frame)
    live_frames = (controlled_live, first_live, final_live)
    assert tuple(frame.frame_index for frame in live_frames) == (0, 1, 2)
    assert endpoint_result.replay_handoff is None
    assert recorder.lifecycle == "saved"
    assert recorder.validated_transition_count == 2
    assert recorder.retained_transition_count == 2
    assert recorder.retained_frame_count == 3
    assert tuple(transition_id for transition_id, _ in append_units) == (
        first_live.incoming_transition_id,
        final_live.incoming_transition_id,
    )
    assert tuple(frame_id for _, frame_id in append_units) == (
        first_live.frame_id,
        final_live.frame_id,
    )

    prepared = recorder.prepared_bundle
    saved = recorder.saved_bundle
    verified = recorder.verified_loaded_bundle
    assert prepared is not None
    assert saved is not None
    assert verified is not None
    assert stat.S_ISREG(saved.replay_path.lstat().st_mode)
    assert stat.S_ISREG(saved.metric_report_path.lstat().st_mode)
    assert saved.replay_path.parent == tmp_path
    assert saved.metric_report_path.parent == tmp_path
    assert saved.replay_path.read_bytes() == prepared.replay_json_bytes
    assert saved.metric_report_path.read_bytes() == prepared.metric_report_json_bytes
    assert prepared.replay_byte_length <= prepared.max_file_size_bytes
    assert prepared.metric_report_byte_length <= prepared.max_file_size_bytes

    public_loaded = load_replay_bundle_v1(
        saved.replay_path,
        require_metric_report=True,
        max_file_size_bytes=prepared.max_file_size_bytes,
    )
    assert public_loaded == verified
    assert public_loaded.replay == prepared.bundle.replay
    assert (
        public_loaded.metric_report_artifact == prepared.bundle.metric_report_artifact
    )

    artifact = public_loaded.replay
    episode_id = artifact.header.context.identity.episode_id
    expected_frame_ids = tuple(f"{episode_id}:frame:{index}" for index in range(3))
    expected_transition_ids = tuple(
        f"{episode_id}:transition:{index}" for index in range(2)
    )
    assert artifact.header.recorded_frame_count == 3
    assert artifact.header.recorded_transition_count == 2
    assert len(artifact.frames) == len(artifact.transitions) + 1 == 3
    assert tuple(frame.frame_id for frame in artifact.frames) == expected_frame_ids
    assert (
        tuple(transition.transition_id for transition in artifact.transitions)
        == expected_transition_ids
    )
    assert tuple(unit[0] for unit in append_units) == expected_transition_ids
    assert tuple(unit[1] for unit in append_units) == expected_frame_ids[1:]

    reviewed_result = coordinator.apply_command(
        _live_request(
            "review-complete-basic-support",
            base_revision=2,
            command=ReviewReplayCommandV1(),
        )
    )
    installed = coordinator.router.snapshot()
    viewer = cast(ReplayViewerService, installed.service)
    assert reviewed_result.replay_handoff is viewer
    assert replay_viewer_factory.call_count == 1
    assert replay_viewer_factory.call_args.args[0] is recorder.verified_loaded_bundle
    assert installed.generation == 1
    assert installed.binding.mode == "replay"
    assert installed.binding.apply_command == viewer.apply_command
    assert installed.binding.current_presentation == viewer.current_presentation
    assert installed.binding.current_metric_report == viewer.current_metric_report

    replay_zero = cast(ResearcherReplayViewerFrameV1, viewer.current_frame())
    next_result = viewer.apply_command(
        _replay_request(
            viewer,
            "next-to-nonzero",
            ReplayNextFrameCommandV1(),
        )
    )
    assert isinstance(next_result.payload, ReplayCommandResponseV1)
    replay_nonzero = cast(ResearcherReplayViewerFrameV1, next_result.payload.frame)
    last_result = viewer.apply_command(
        _replay_request(
            viewer,
            "seek-final",
            ReplayLastFrameCommandV1(),
        )
    )
    assert isinstance(last_result.payload, ReplayCommandResponseV1)
    replay_final = cast(ResearcherReplayViewerFrameV1, last_result.payload.frame)
    replay_frames = (replay_zero, replay_nonzero, replay_final)
    assert tuple(frame.cursor.frame_index for frame in replay_frames) == (0, 1, 2)

    for live_frame, replay_frame in zip(live_frames, replay_frames, strict=True):
        live_scene = live_frame.projection.scene
        replay_scene = replay_frame.projection.scene
        assert replay_frame.frame_id == live_frame.frame_id
        assert replay_frame.incoming_transition_id == live_frame.incoming_transition_id
        assert replay_scene == live_scene
        assert (
            replay_frame.projection.incoming_events
            == live_frame.projection.incoming_events
        )
        assert replay_frame.projection == live_frame.projection
        assert replay_scene.selection is not None
        assert replay_scene.selection.controlled_global_slot == 2
        assert replay_scene.selection.selected_global_slot == 2
        assert replay_scene.next_decision_selected_legality is not None
        assert replay_scene.next_decision_selected_legality.armed_lane == 0

    timeline = cast(ResearcherReplayTimelineV1, viewer.current_timeline())
    assert len(timeline.rows) == 3
    assert tuple(row.frame_id for row in timeline.rows) == expected_frame_ids
    assert tuple(row.incoming_transition_id for row in timeline.rows) == (
        None,
        *expected_transition_ids,
    )
    assert tuple(row.frame_index for row in timeline.rows) == (0, 1, 2)
    assert replay_final.completion.completion_state == "complete"
    assert replay_final.completion.validated_transition_count == 2
    assert replay_final.completion.last_valid_frame_id == expected_frame_ids[-1]
    assert (
        replay_final.completion.completion_bases == artifact.completion.completion_bases
    )
    assert replay_final.processing.status == artifact.processing_status.status
    assert (
        replay_final.processing.processed_transition_count
        == artifact.processing_status.processed_transition_count
        == 2
    )

    report_artifact = public_loaded.metric_report_artifact
    assert report_artifact is not None
    report_reference = artifact.metric_report_reference
    assert report_reference.report_artifact_id == report_artifact.report_artifact_id
    assert report_reference.metric_report_id == report_artifact.report.report_id
    assert (
        report_reference.canonical_digest_sha256
        == report_artifact.canonical_digest_sha256
    )
    assert report_reference.canonical_byte_length == prepared.metric_report_byte_length
    assert (
        replay_final.artifact_summary.replay_reference.canonical_byte_length
        == prepared.replay_byte_length
    )
    assert replay_final.artifact_summary.metric_report_availability == "available"

    generic_frame = cast(
        ResearcherReplayViewerFrameV1,
        ReplayViewerService(
            public_loaded,
            viewer_session_id="generic-round-trip",
        ).current_frame(),
    )
    assert generic_frame.projection.scene.selection is not None
    assert generic_frame.projection.scene.selection.controlled_global_slot == 0
    assert generic_frame.projection.scene.selection.selected_global_slot is None
    assert generic_frame.projection.scene.next_decision_selected_legality is None

    selection_result = viewer.apply_command(
        _replay_request(
            viewer,
            "select-different-agent",
            ReplaySelectAgentCommandV1(selected_global_slot=6),
        )
    )
    assert selection_result.outcome == "response"
    assert isinstance(selection_result.payload, ReplayCommandResponseV1)
    selected_frame = cast(
        ResearcherReplayViewerFrameV1,
        selection_result.payload.frame,
    )
    selected_scene = selected_frame.projection.scene
    assert selected_scene.selection is not None
    assert selected_scene.selection.controlled_global_slot == 2
    assert selected_scene.selection.selected_global_slot == 6
    assert selected_scene.next_decision_selected_legality is not None
    assert selected_scene.next_decision_selected_legality.armed_lane is None
    assert selected_scene.next_decision_selected_legality.armed_pair_legal is False
    assert selected_frame.cursor.frame_index == 2
    assert len(append_units) == 2

    public_text = "\n".join(
        (
            controlled_live.model_dump_json(),
            endpoint_result.payload.model_dump_json(),
            reviewed_result.payload.model_dump_json(),
            replay_zero.model_dump_json(),
            timeline.model_dump_json(),
            selected_frame.model_dump_json(),
        )
    )
    for forbidden_path in (
        str(tmp_path),
        saved.replay_path.name,
        saved.metric_report_path.name,
        "replay_path",
        "metric_report_path",
    ):
        assert forbidden_path not in public_text
        assert forbidden_path.encode() not in prepared.replay_json_bytes
        assert forbidden_path.encode() not in prepared.metric_report_json_bytes
