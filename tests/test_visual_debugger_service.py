"""Focused revision, idempotency, and concurrency proofs for DebuggerService."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from unittest.mock import Mock

import jax.numpy as jnp
import pytest
import scripts.dev.visual_debugger.control as control_module
import scripts.dev.visual_debugger.input as input_module
import scripts.dev.visual_debugger.recording as recording_module
import scripts.dev.visual_debugger.service as service_module
from scripts.dev.visual_debugger.control import create_session, select_clicked_target
from scripts.dev.visual_debugger.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
)
from scripts.dev.visual_debugger.frame import build_debugger_frame
from scripts.dev.visual_debugger.input import InputDispatchResult
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.protocol import (
    ActorPovLiveDebuggerFrameV2,
    ActorPovTargetActionCommandV1,
    ApiErrorV2,
    BattlefieldPointerCommandV1,
    CommandRequestV1,
    CommandResponseV2,
    ConfirmDiscardAndReplaceCommandV1,
    ExitCommandV1,
    FinishAndReviewCommandV1,
    KeyboardCommandV1,
    ResearcherLiveDebuggerFrameV2,
    ResetCommandV1,
    RetrySaveCommandV1,
    ReviewReplayCommandV1,
    RosterSelectionCommandV1,
    SaveAsCommandV1,
    ScenarioSwitchCommandV1,
    SetMovementScaleCommandV1,
    SetPresetCommandV1,
    SetViewCommandV1,
    ViewMode,
)
from scripts.dev.visual_debugger.recording import (
    DebuggerReplayRecorderV1,
    build_debugger_recording_specification_v1,
)
from scripts.dev.visual_debugger.replay_protocol import ResearcherReplayViewerFrameV1
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.service import DebuggerService
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.types import EnvConfig, EnvState
from marl_battlegrounds.evaluation.metrics import (
    EvaluationEpisodeCompletionV1,
    EvaluationMetricReducerStateV1,
    EvaluationMetricReducerV1,
    EvaluationProcessingStatusV1,
    EvaluationTransitionViewV1,
    SufficientStatisticDraftV1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
)
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_io import (
    ReplaySaveError,
    preflight_replay_bundle_destination_v1,
)


class _FailingReducerState(EvaluationMetricReducerStateV1):
    """Frozen test reducer state for a post-validation processing failure."""


class _HorizonFailingReducerState(EvaluationMetricReducerStateV1):
    """Frozen reducer progress before a failure on the exact horizon unit."""

    processed_transition_count: int = 0


@dataclass(slots=True)
class _FailingAdvanceReducer:
    """Fail only after the recorder observer commits a valid transition unit."""

    reducer_id: str = "test.service-processing-failure"
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
        raise RuntimeError("private reducer failure detail")

    def finalize(
        self,
        state: EvaluationMetricReducerStateV1,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        del state, completion, processing_status
        return ()


@dataclass(slots=True)
class _HorizonFailingAdvanceReducer:
    """Advance one unit, then fail after CP2 validates the horizon unit."""

    reducer_id: str = "test.service-horizon-processing-failure"
    reducer_version: int = 1

    def initialize(
        self,
        context: EvaluationEpisodeContextV1,
        initial_frame: EvaluationFrameV1,
    ) -> EvaluationMetricReducerStateV1:
        del context, initial_frame
        return _HorizonFailingReducerState(
            reducer_id=self.reducer_id,
            reducer_version=self.reducer_version,
        )

    def advance(
        self,
        previous_state: EvaluationMetricReducerStateV1,
        view: EvaluationTransitionViewV1,
    ) -> EvaluationMetricReducerStateV1:
        if not isinstance(previous_state, _HorizonFailingReducerState):
            raise TypeError("unexpected horizon reducer state")
        if view.successor_frame.frame_index == view.context.expected_horizon:
            raise RuntimeError("private horizon reducer failure detail")
        return _HorizonFailingReducerState(
            reducer_id=self.reducer_id,
            reducer_version=self.reducer_version,
            processed_transition_count=previous_state.processed_transition_count + 1,
        )

    def finalize(
        self,
        state: EvaluationMetricReducerStateV1,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        del state, completion, processing_status
        return ()


def _service(
    scenario_name: str = "arena_5v5",
    *,
    include_stress: bool = False,
) -> DebuggerService:
    session = create_session(
        get_scenario(scenario_name),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    return DebuggerService(
        session,
        view_mode="researcher",
        preset="analysis",
        include_stress=include_stress,
        session_id="test-session",
    )


def _runtime_provenance() -> RuntimeProvenanceV1:
    return RuntimeProvenanceV1(
        python_version="3.13.0",
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


def _recording_service(
    tmp_path: Path,
    scenario_name: str = "arena_5v5",
    *,
    reducers: tuple[EvaluationMetricReducerV1, ...] = (),
    view_mode: ViewMode = "researcher",
    maximum_episode_steps: int | None = None,
) -> tuple[DebuggerService, DebuggerReplayRecorderV1]:
    debug_launch = debugger_test_launch_specification()
    launch = build_debugger_evaluation_launch_specification_v1(
        root_seed=debug_launch.root_seed,
        code_revision=debug_launch.code_revision,
        capture_profile="evaluation_metric_complete",
    )
    scenario = get_scenario(scenario_name)
    if maximum_episode_steps is not None:
        build_registered_scenario = scenario.build_scenario

        def build_scenario_with_horizon() -> tuple[EnvConfig, EnvState]:
            config, state = build_registered_scenario()
            return config._replace(max_steps=maximum_episode_steps), state

        scenario = replace(scenario, build_scenario=build_scenario_with_horizon)
    session = create_session(
        scenario,
        seed=0,
        evaluation_launch_specification=launch,
        controlled_global_slot=None,
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
            tmp_path / "episode.marlbg-replay.json"
        ),
        context=session.evaluation_context,
        initial_frame=session.current_evaluation_frame,
        reducers=reducers,
    )
    return (
        DebuggerService(
            session,
            view_mode=view_mode,
            preset="analysis",
            include_stress=False,
            session_id="recording-test-session",
            recorder=recorder,
        ),
        recorder,
    )


def _request(
    command_id: str,
    *,
    base_revision: int,
    command: object,
    client_id: str = "client-a",
) -> CommandRequestV1:
    return CommandRequestV1(
        client_id=client_id,
        command_id=command_id,
        base_revision=base_revision,
        command=command,  # pyright: ignore[reportArgumentType]
    )


def _close_recording_in_lifecycle(
    service: DebuggerService,
    recorder: DebuggerReplayRecorderV1,
    lifecycle: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if lifecycle == "persistence_failed":

        def fail_publish(*_args: object, **_kwargs: object) -> object:
            raise ReplaySaveError(
                "temporary_write_failed",
                path=None,
                detail="private closed-lifecycle setup detail",
            )

        monkeypatch.setattr(
            recording_module,
            "publish_prepared_replay_bundle_v1",
            fail_publish,
        )
    elif lifecycle == "saved":

        def fail_review(_self: DebuggerReplayRecorderV1) -> object:
            raise RuntimeError("private closed-lifecycle review setup detail")

        monkeypatch.setattr(DebuggerReplayRecorderV1, "begin_review", fail_review)
    elif lifecycle != "reviewing":
        raise AssertionError(f"unsupported closed lifecycle {lifecycle!r}")

    closed = service.apply_command(
        _request(
            f"close-for-{lifecycle}",
            base_revision=0,
            command=FinishAndReviewCommandV1(),
        )
    )
    assert isinstance(closed.payload, CommandResponseV2)
    assert recorder.lifecycle == lifecycle


def _authorized_pov_slots(debugger_session: DebuggerSession) -> set[int]:
    frame = build_debugger_frame(
        debugger_session,
        session_id="test-pov-slots",
        revision=0,
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )
    assert isinstance(frame, ActorPovLiveDebuggerFrameV2)
    public_ids = {
        frame.projection.scene.self_actor.public_agent_id,
        *(body.public_agent_id for body in frame.projection.scene.visible_bodies),
    }
    return {
        row.global_slot
        for row in debugger_session.evaluation_context.roster
        if row.public_agent_id in public_ids
    }


def test_initial_and_current_frame_reads_are_coherent_and_non_mutating() -> None:
    service = _service()
    initial_session = service.session
    initial_key = initial_session.key

    first = service.current_frame()
    second = service.current_frame()

    assert first is second
    assert isinstance(first, ResearcherLiveDebuggerFrameV2)
    assert first.revision == 0
    assert first.simulator_step_count == 0
    assert first.incoming_transition_id is None
    assert service.session is initial_session
    assert bool(jnp.array_equal(service.session.key, initial_key))


def test_ui_edit_advances_only_frame_revision() -> None:
    service = _service()
    initial = service.session
    request = _request(
        "move-east",
        base_revision=0,
        command=KeyboardCommandV1(key="d"),
    )

    result = service.apply_command(request)

    assert result.outcome == "response"
    assert isinstance(result.payload, CommandResponseV2)
    assert result.payload.result == "applied"
    assert result.payload.frame.revision == 1
    assert result.payload.frame.simulator_step_count == 0
    assert service.session.state is initial.state
    assert bool(jnp.array_equal(service.session.key, initial.key))


def test_live_presentation_commands_publish_authoritative_audience_fields() -> None:
    service = _service()

    ranges = service.apply_command(
        _request(
            "toggle-ranges",
            base_revision=0,
            command=KeyboardCommandV1(key="g"),
        )
    )
    verbose = service.apply_command(
        _request(
            "toggle-verbose",
            base_revision=1,
            command=KeyboardCommandV1(key="v"),
        )
    )
    pov = service.apply_command(
        _request(
            "switch-pov",
            base_revision=2,
            command=SetViewCommandV1(view_mode="pov"),
        )
    )

    assert isinstance(ranges.payload, CommandResponseV2)
    assert isinstance(ranges.payload.frame, ResearcherLiveDebuggerFrameV2)
    assert ranges.payload.frame.show_ranges is False
    assert ranges.payload.frame.verbose is False
    assert isinstance(verbose.payload, CommandResponseV2)
    assert isinstance(verbose.payload.frame, ResearcherLiveDebuggerFrameV2)
    assert verbose.payload.frame.show_ranges is False
    assert verbose.payload.frame.verbose is True
    assert isinstance(pov.payload, CommandResponseV2)
    assert isinstance(pov.payload.frame, ActorPovLiveDebuggerFrameV2)
    assert pov.payload.frame.verbose is True
    assert "show_ranges" not in pov.payload.frame.model_dump(mode="json")


def test_shift_r_is_a_host_no_op_and_ordinary_r_remains_a_restart() -> None:
    service = _service()
    initial_session = service.session
    initial_frame = service.current_frame()
    initial_count = service.evaluation_validated_transition_count
    shifted_request = _request(
        "removed-shift-r",
        base_revision=0,
        command=KeyboardCommandV1(key="R", shift_key=True),
    )

    shifted = service.apply_command(shifted_request)
    duplicate = service.apply_command(shifted_request)

    assert isinstance(shifted.payload, CommandResponseV2)
    assert shifted.payload.result == "no_op"
    assert shifted.payload.frame is initial_frame
    assert service.session is initial_session
    assert service.revision == 0
    assert service.evaluation_validated_transition_count == initial_count
    assert isinstance(duplicate.payload, CommandResponseV2)
    assert duplicate.payload.result == "duplicate"
    assert duplicate.payload.frame is initial_frame

    ordinary = service.apply_command(
        _request(
            "ordinary-r",
            base_revision=0,
            command=KeyboardCommandV1(key="r"),
        )
    )
    assert isinstance(ordinary.payload, CommandResponseV2)
    assert ordinary.payload.result == "applied"
    assert ordinary.payload.frame.run_generation == initial_frame.run_generation + 1
    assert service.revision == 1


def test_submit_duplicate_conflict_and_stale_requests_cannot_restep() -> None:
    service = _service()
    submit = _request(
        "submit-once",
        base_revision=0,
        command=KeyboardCommandV1(key=" "),
    )

    applied = service.apply_command(submit)
    duplicate = service.apply_command(submit)
    conflicting = service.apply_command(
        _request(
            "submit-once",
            base_revision=1,
            command=KeyboardCommandV1(key="r"),
        )
    )
    stale = service.apply_command(
        _request(
            "stale-submit",
            base_revision=0,
            command=KeyboardCommandV1(key="enter"),
        )
    )
    stale_duplicate = service.apply_command(
        _request(
            "stale-submit",
            base_revision=0,
            command=KeyboardCommandV1(key="enter"),
        )
    )
    stale_id_reuse = service.apply_command(
        _request(
            "stale-submit",
            base_revision=1,
            command=KeyboardCommandV1(key="enter"),
        )
    )

    assert isinstance(applied.payload, CommandResponseV2)
    assert isinstance(applied.payload.frame, ResearcherLiveDebuggerFrameV2)
    assert applied.payload.result == "applied"
    assert applied.payload.frame.simulator_step_count == 1
    assert applied.payload.frame.incoming_transition_index == 0
    assert isinstance(duplicate.payload, CommandResponseV2)
    assert duplicate.payload.result == "duplicate"
    assert duplicate.payload.frame.simulator_step_count == 1
    assert conflicting.outcome == "command_id_conflict"
    assert stale.outcome == "stale_revision"
    assert isinstance(stale_duplicate.payload, CommandResponseV2)
    assert stale_duplicate.payload.result == "duplicate"
    assert stale_id_reuse.outcome == "command_id_conflict"
    assert int(service.session.state.step_count) == 1
    assert service.revision == 1


def test_repeat_submit_is_consumed_without_revision_or_step() -> None:
    service = _service()
    result = service.apply_command(
        _request(
            "held-submit",
            base_revision=0,
            command=KeyboardCommandV1(key="Enter", repeat=True),
        )
    )

    assert isinstance(result.payload, CommandResponseV2)
    assert result.payload.result == "no_op"
    assert result.payload.frame.revision == 0
    assert result.payload.frame.simulator_step_count == 0
    assert service.command_cache_size == 1


def test_unavailable_browser_draft_is_a_revision_preserving_no_op() -> None:
    service = _service()
    initial = service.session

    result = service.apply_command(
        _request(
            "masked-basic",
            base_revision=0,
            command=KeyboardCommandV1(key="1"),
        )
    )

    assert isinstance(result.payload, CommandResponseV2)
    assert result.payload.result == "no_op"
    assert result.payload.frame.revision == 0
    assert result.payload.frame.simulator_step_count == 0
    assert service.session is initial
    assert result.payload.notice is not None
    assert "canonical no-combat tuple" in result.payload.notice


def test_movement_scale_reset_is_idempotent_revisioned_and_never_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()

    def forbidden_step(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("movement-scale reset must not call step")

    monkeypatch.setattr(control_module, "step", forbidden_step)
    request = _request(
        "scale-once",
        base_revision=0,
        command=SetMovementScaleCommandV1(movement_scale=0.1),
    )

    applied = service.apply_command(request)
    duplicate = service.apply_command(request)
    stale = service.apply_command(
        _request(
            "stale-scale",
            base_revision=0,
            command=SetMovementScaleCommandV1(movement_scale=0.25),
        )
    )
    same_effective = service.apply_command(
        _request(
            "same-scale",
            base_revision=1,
            command=SetMovementScaleCommandV1(movement_scale=0.1),
        )
    )

    assert isinstance(applied.payload, CommandResponseV2)
    assert isinstance(applied.payload.frame, ResearcherLiveDebuggerFrameV2)
    assert applied.payload.result == "applied"
    assert applied.payload.frame.revision == 1
    assert applied.payload.frame.run_generation == 1
    assert applied.payload.frame.simulator_step_count == 0
    assert applied.payload.frame.scenario.ordinary_movement_distance_scale == 0.1
    assert isinstance(duplicate.payload, CommandResponseV2)
    assert duplicate.payload.result == "duplicate"
    assert stale.outcome == "stale_revision"
    assert isinstance(same_effective.payload, CommandResponseV2)
    assert same_effective.payload.result == "no_op"
    assert same_effective.payload.frame.revision == 1
    assert service.session.run_generation == 1
    assert int(service.session.state.step_count) == 0


def test_entering_pov_clears_a_hidden_pending_target_without_stepping() -> None:
    service = _service()
    researcher_slots = {
        row.global_slot
        for row in service.session.evaluation_context.roster
        if row.configured_active
    }
    hidden_slots = researcher_slots - _authorized_pov_slots(service.session)
    assert hidden_slots
    hidden_target = min(hidden_slots)

    selected = service.apply_command(
        _request(
            "select-hidden",
            base_revision=0,
            command=RosterSelectionCommandV1(
                role="target",
                global_slot=hidden_target,
            ),
        )
    )
    assert isinstance(selected.payload, CommandResponseV2)
    assert service.session.pending_action.selected_global_target_slot == hidden_target
    before_view = service.session

    changed_view = service.apply_command(
        _request(
            "enter-pov",
            base_revision=1,
            command=SetViewCommandV1(view_mode="pov"),
        )
    )

    assert isinstance(changed_view.payload, CommandResponseV2)
    assert changed_view.payload.frame.view_mode == "pov"
    assert service.session.pending_action.selected_global_target_slot is None
    assert service.session.state is before_view.state
    assert bool(jnp.array_equal(service.session.key, before_view.key))
    assert int(service.session.state.step_count) == 0


def test_terminal_pov_sanitizer_change_does_not_append_stale_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered = get_scenario("arena_5v5")
    build_registered_scenario = registered.build_scenario

    def build_one_step_scenario() -> tuple[EnvConfig, EnvState]:
        config, state = build_registered_scenario()
        return config._replace(max_steps=1), state

    session = create_session(
        replace(registered, build_scenario=build_one_step_scenario),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=0,
        show_ranges=True,
        verbose_logging=False,
    )
    service = DebuggerService(
        session,
        view_mode="pov",
        preset="analysis",
        include_stress=False,
        session_id="terminal-pov-sanitizer",
    )
    terminal = service.apply_command(
        _request(
            "reach-pov-horizon",
            base_revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )
    assert isinstance(terminal.payload, CommandResponseV2)
    assert service.session.reached_declared_horizon
    target = min(_authorized_pov_slots(service.session))
    selected = service.apply_command(
        _request(
            "stage-terminal-pov-target",
            base_revision=1,
            command=RosterSelectionCommandV1(role="target", global_slot=target),
        )
    )
    assert isinstance(selected.payload, CommandResponseV2)
    assert service.session.pending_action.selected_global_target_slot == target
    previous_incoming = service.session.incoming_evaluation_view
    previous_state = service.session.state
    previous_key = service.session.key
    previous_observer_count = service.evaluation_validated_transition_count

    def clear_terminal_target(session: DebuggerSession) -> DebuggerSession:
        return control_module.clear_pending_target(session)

    step_spy = Mock(side_effect=AssertionError("terminal submit must not step"))
    capture_spy = Mock(side_effect=AssertionError("terminal submit must not capture"))
    monkeypatch.setattr(
        input_module, "sanitize_pov_pending_target", clear_terminal_target
    )
    monkeypatch.setattr(control_module, "step", step_spy)
    monkeypatch.setattr(
        control_module,
        "capture_evaluation_transition_unit_v1",
        capture_spy,
    )

    sanitized = service.apply_command(
        _request(
            "sanitize-terminal-pov-target",
            base_revision=2,
            command=KeyboardCommandV1(key="Enter"),
        )
    )

    assert isinstance(sanitized.payload, CommandResponseV2)
    assert sanitized.payload.result == "applied"
    assert service.revision == 3
    assert service.session.pending_action.selected_global_target_slot is None
    assert service.session.incoming_evaluation_view is previous_incoming
    assert service.session.state is previous_state
    assert service.session.key is previous_key
    assert service.evaluation_validated_transition_count == previous_observer_count == 1
    assert step_spy.call_count == 0
    assert capture_spy.call_count == 0
    assert not service.faulted


def test_initial_pov_service_clears_hidden_pending_target() -> None:
    session = create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    pov_slots = _authorized_pov_slots(session)
    hidden_target = min(
        row.global_slot
        for row in session.evaluation_context.roster
        if row.configured_active and row.global_slot not in pov_slots
    )
    selected = select_clicked_target(session, hidden_target)

    service = DebuggerService(
        selected,
        view_mode="pov",
        preset="analysis",
        include_stress=False,
        session_id="initial-pov",
    )

    assert service.session.pending_action.selected_global_target_slot is None
    frame = service.current_frame()
    assert isinstance(frame, ActorPovLiveDebuggerFrameV2)
    assert not hasattr(frame.hud, "selected_global_slot")
    assert service.session.state is session.state
    assert service.session.key is session.key
    assert int(service.session.state.step_count) == 0


def test_same_base_concurrent_commands_dispatch_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    real_dispatch = service_module.dispatch_command
    call_count = 0
    count_lock = Lock()

    def counting_dispatch(*args: object, **kwargs: object) -> InputDispatchResult:
        nonlocal call_count
        with count_lock:
            call_count += 1
        return real_dispatch(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service_module, "dispatch_command", counting_dispatch)
    requests = (
        _request(
            "preset-a",
            base_revision=0,
            client_id="client-a",
            command=SetPresetCommandV1(preset="presentation"),
        ),
        _request(
            "preset-b",
            base_revision=0,
            client_id="client-b",
            command=SetPresetCommandV1(preset="debug"),
        ),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(service.apply_command, requests))

    assert call_count == 1
    assert sorted(result.outcome for result in results) == [
        "response",
        "stale_revision",
    ]
    assert service.revision == 1


def test_same_base_concurrent_submits_call_authoritative_step_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    real_step = control_module.step
    step_calls = 0
    count_lock = Lock()

    def counting_step(*args: object, **kwargs: object) -> object:
        nonlocal step_calls
        with count_lock:
            step_calls += 1
        return real_step(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(control_module, "step", counting_step)
    requests = tuple(
        _request(
            f"concurrent-submit-{index}",
            base_revision=0,
            client_id=f"client-{index}",
            command=KeyboardCommandV1(key="Enter"),
        )
        for index in range(2)
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(service.apply_command, requests))

    assert step_calls == 1
    assert sorted(result.outcome for result in results) == [
        "response",
        "stale_revision",
    ]
    assert int(service.session.state.step_count) == 1
    assert service.revision == 1


def test_frame_build_failure_keeps_epoch_coherent_and_consumes_command_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    initial_session = service.session
    initial_frame = service.current_frame()
    real_step = control_module.step
    step_calls = 0

    def counting_step(*args: object, **kwargs: object) -> object:
        nonlocal step_calls
        step_calls += 1
        return real_step(*args, **kwargs)  # type: ignore[arg-type]

    def fail_frame(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("synthetic frame failure")

    monkeypatch.setattr(control_module, "step", counting_step)
    monkeypatch.setattr(service_module, "build_debugger_frame", fail_frame)
    submit = _request(
        "failed-frame-submit",
        base_revision=0,
        command=KeyboardCommandV1(key="Enter"),
    )

    with pytest.raises(RuntimeError, match="synthetic frame failure"):
        service.apply_command(submit)
    duplicate = service.apply_command(submit)

    assert service.session is initial_session
    assert service.current_frame() is initial_frame
    assert service.revision == 0
    assert service.faulted
    assert step_calls == 1
    assert isinstance(duplicate.payload, CommandResponseV2)
    assert duplicate.payload.result == "duplicate"
    assert duplicate.payload.frame is initial_frame

    for index in range(257):
        fenced = service.apply_command(
            _request(
                f"faulted-command-{index}",
                base_revision=0,
                command=KeyboardCommandV1(key="F13"),
            )
        )
        assert fenced.outcome == "service_faulted"
    retry_after_eviction = service.apply_command(submit)

    assert retry_after_eviction.outcome == "service_faulted"
    assert step_calls == 1


def test_accepted_exit_fences_concurrent_submissions_without_stepping() -> None:
    service = _service()
    exit_request = _request(
        "exit",
        base_revision=0,
        command=ExitCommandV1(),
    )

    accepted_exit = service.apply_command(exit_request)
    submissions = tuple(
        _request(
            f"submit-after-exit-{index}",
            base_revision=0,
            client_id=f"client-{index}",
            command=KeyboardCommandV1(key="Enter"),
        )
        for index in range(8)
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(service.apply_command, submissions))
    duplicate_exit = service.apply_command(exit_request)

    assert isinstance(accepted_exit.payload, CommandResponseV2)
    assert accepted_exit.payload.result == "shutdown_scheduled"
    assert accepted_exit.shutdown_requested
    assert service.shutting_down
    assert {result.outcome for result in results} == {"server_shutting_down"}
    for result in results:
        assert isinstance(result.payload, ApiErrorV2)
        assert result.payload.error_code == "server_shutting_down"
    assert isinstance(duplicate_exit.payload, CommandResponseV2)
    assert duplicate_exit.payload.result == "duplicate"
    assert int(service.session.state.step_count) == 0
    assert service.revision == 0


def test_command_record_cache_is_bounded() -> None:
    service = _service()
    for index in range(257):
        result = service.apply_command(
            _request(
                f"ignored-{index}",
                base_revision=0,
                command=KeyboardCommandV1(key="F13"),
            )
        )
        assert result.outcome == "response"

    assert service.command_cache_size == 256
    assert service.revision == 0


def test_evicted_applied_submit_is_stale_and_cannot_restep() -> None:
    service = _service()
    submit = _request(
        "submit-before-eviction",
        base_revision=0,
        command=KeyboardCommandV1(key="Enter"),
    )

    applied = service.apply_command(submit)
    assert isinstance(applied.payload, CommandResponseV2)
    assert applied.payload.result == "applied"
    assert applied.payload.frame.revision == 1
    assert applied.payload.frame.simulator_step_count == 1

    for index in range(256):
        no_op = service.apply_command(
            _request(
                f"post-submit-no-op-{index}",
                base_revision=1,
                command=KeyboardCommandV1(key="F13"),
            )
        )
        assert isinstance(no_op.payload, CommandResponseV2)
        assert no_op.payload.result == "no_op"

    assert service.command_cache_size == 256
    replayed = service.apply_command(submit)

    assert replayed.outcome == "stale_revision"
    assert isinstance(replayed.payload, ApiErrorV2)
    assert replayed.payload.error_code == "stale_revision"
    assert replayed.payload.latest_frame is not None
    assert replayed.payload.latest_frame.revision == 1
    assert replayed.payload.latest_frame.simulator_step_count == 1
    assert service.revision == 1
    assert int(service.session.state.step_count) == 1


def test_stress_scenario_requires_explicit_service_authorization() -> None:
    session = create_session(
        get_scenario("charge_convergence"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    with pytest.raises(ValueError, match="include_stress=True"):
        DebuggerService(
            session,
            view_mode="researcher",
            preset="analysis",
            include_stress=False,
        )


def test_recording_append_finish_handoff_and_duplicate_are_exactly_once(
    tmp_path: Path,
) -> None:
    service, recorder = _recording_service(tmp_path)
    initial_frame = service.current_frame()
    assert initial_frame.recording == recorder.status
    assert initial_frame.recording is not None
    assert initial_frame.recording.lifecycle == "recording"
    assert recorder.retained_frame_count == 1
    assert recorder.retained_transition_count == 0

    submitted = service.apply_command(
        _request(
            "record-submit",
            base_revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )
    assert isinstance(submitted.payload, CommandResponseV2)
    assert submitted.payload.frame.recording == recorder.status
    assert submitted.payload.frame.recording is not None
    assert submitted.payload.frame.recording.captured_transition_count == 1
    assert recorder.retained_frame_count == 2
    assert recorder.retained_transition_count == 1
    assert not (tmp_path / "episode.marlbg-replay.json").exists()

    finish_request = _request(
        "finish-recording",
        base_revision=1,
        command=FinishAndReviewCommandV1(),
    )
    finished = service.apply_command(finish_request)
    duplicate = service.apply_command(finish_request)

    assert isinstance(finished.payload, CommandResponseV2)
    assert finished.payload.frame.recording == recorder.status
    assert finished.payload.frame.recording is not None
    assert finished.payload.frame.recording.lifecycle == "reviewing"
    assert finished.replay_handoff is not None
    replay_frame = finished.replay_handoff.current_frame()
    assert isinstance(replay_frame, ResearcherReplayViewerFrameV1)
    assert replay_frame.viewer_session_id != "recording-test-session"
    assert replay_frame.cursor.frame_index == 0
    assert replay_frame.cursor.final_frame_index == 1
    assert replay_frame.cursor.cursor_generation == 0
    assert replay_frame.cursor.choreography_generation == 0
    assert replay_frame.preset == "analysis"
    assert replay_frame.show_ranges is True
    assert replay_frame.verbose is False
    assert (tmp_path / "episode.marlbg-replay.json").is_file()
    assert isinstance(duplicate.payload, CommandResponseV2)
    assert duplicate.payload.result == "duplicate"
    assert duplicate.replay_handoff is None
    assert recorder.retained_transition_count == 1


def test_recording_service_never_constructs_parallel_debug_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_debug_observer(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("recording service constructed a parallel observer")

    monkeypatch.setattr(
        DebuggerService,
        "_new_evaluation_observer",
        forbidden_debug_observer,
    )
    service, recorder = _recording_service(tmp_path)

    assert service.evaluation_validated_transition_count == 0
    assert recorder.retained_frame_count == 1
    assert recorder.retained_transition_count == 0


def test_endpoint_auto_save_installs_saved_status_and_review_handoff(
    tmp_path: Path,
) -> None:
    service, recorder = _recording_service(tmp_path, "basic_support")

    first = service.apply_command(
        _request("script-0", base_revision=0, command=KeyboardCommandV1(key="n"))
    )
    endpoint = service.apply_command(
        _request("script-1", base_revision=1, command=KeyboardCommandV1(key="n"))
    )

    assert isinstance(first.payload, CommandResponseV2)
    assert first.payload.frame.recording is not None
    assert first.payload.frame.recording.lifecycle == "recording"
    assert isinstance(endpoint.payload, CommandResponseV2)
    assert endpoint.replay_handoff is None
    assert endpoint.payload.frame.recording == recorder.status
    assert endpoint.payload.frame.recording is not None
    assert endpoint.payload.frame.recording.lifecycle == "saved"
    assert endpoint.payload.frame.recording.completion_state == "complete"
    assert endpoint.payload.frame.recording.review_available

    fenced = service.apply_command(
        _request(
            "script-after-end", base_revision=2, command=KeyboardCommandV1(key="n")
        )
    )
    reviewed = service.apply_command(
        _request(
            "review-endpoint",
            base_revision=2,
            command=ReviewReplayCommandV1(),
        )
    )

    assert isinstance(fenced.payload, CommandResponseV2)
    assert fenced.payload.result == "no_op"
    assert fenced.payload.frame.revision == 2
    assert isinstance(reviewed.payload, CommandResponseV2)
    assert reviewed.replay_handoff is not None
    assert reviewed.replay_handoff.current_frame().cursor.frame_index == 0
    assert recorder.lifecycle == "reviewing"


def test_exact_horizon_truncation_auto_saves_as_complete_declared_horizon(
    tmp_path: Path,
) -> None:
    service, recorder = _recording_service(
        tmp_path,
        "basic_support",
        maximum_episode_steps=2,
    )
    first = service.apply_command(
        _request(
            "horizon-truncate-0", base_revision=0, command=KeyboardCommandV1(key="n")
        )
    )
    endpoint = service.apply_command(
        _request(
            "horizon-truncate-1", base_revision=1, command=KeyboardCommandV1(key="n")
        )
    )

    assert isinstance(first.payload, CommandResponseV2)
    assert isinstance(endpoint.payload, CommandResponseV2)
    assert recorder.lifecycle == "saved"
    assert endpoint.payload.frame.recording == recorder.status
    assert endpoint.payload.frame.recording is not None
    assert endpoint.payload.frame.recording.completion_state == "complete"
    assert endpoint.payload.frame.recording.completion_reason is None
    bundle = recorder.bundle
    assert bundle is not None
    assert bundle.replay.completion.truncated is True
    assert bundle.replay.completion.completion_bases == ("declared_horizon",)
    assert recorder.publication_outcome == "saved"


def test_recording_restart_fence_and_confirmed_discard_replace_atomically(
    tmp_path: Path,
) -> None:
    service, old_recorder = _recording_service(tmp_path)
    submitted = service.apply_command(
        _request(
            "prefix-before-reset",
            base_revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )
    assert isinstance(submitted.payload, CommandResponseV2)
    old_episode_id = service.session.evaluation_context.identity.episode_id

    fenced = service.apply_command(
        _request(
            "fenced-reset",
            base_revision=1,
            command=ResetCommandV1(),
        )
    )
    confirmed = service.apply_command(
        _request(
            "confirmed-reset",
            base_revision=1,
            command=ConfirmDiscardAndReplaceCommandV1(replacement=ResetCommandV1()),
        )
    )

    assert isinstance(fenced.payload, CommandResponseV2)
    assert fenced.payload.result == "no_op"
    assert old_recorder.lifecycle == "discarded"
    assert isinstance(confirmed.payload, CommandResponseV2)
    assert confirmed.payload.result == "applied"
    assert confirmed.payload.frame.recording is not None
    assert confirmed.payload.frame.recording.lifecycle == "recording"
    assert confirmed.payload.frame.recording.captured_transition_count == 0
    assert service.session.evaluation_context.identity.episode_id != old_episode_id
    assert service.evaluation_validated_transition_count == 0


@pytest.mark.parametrize(
    "command",
    (
        ResetCommandV1(),
        KeyboardCommandV1(key="r"),
        KeyboardCommandV1(key="]"),
        ScenarioSwitchCommandV1(scenario_name="basic_support"),
        SetMovementScaleCommandV1(movement_scale=0.1),
    ),
)
def test_captured_recording_prefix_fences_every_restart_entry_point(
    tmp_path: Path,
    command: object,
) -> None:
    service, recorder = _recording_service(tmp_path)
    submitted = service.apply_command(
        _request(
            "restart-fence-prefix",
            base_revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )
    assert isinstance(submitted.payload, CommandResponseV2)
    prefix_session = service.session

    fenced = service.apply_command(
        _request(
            f"restart-fence-{type(command).__name__}",
            base_revision=1,
            command=command,
        )
    )

    assert isinstance(fenced.payload, CommandResponseV2)
    assert fenced.payload.result == "no_op"
    assert service.session is prefix_session
    assert service.revision == 1
    assert recorder.lifecycle == "recording"
    assert recorder.validated_transition_count == 1


def test_t0_restart_uses_fresh_recorder_without_mutating_old_draft(
    tmp_path: Path,
) -> None:
    service, old_recorder = _recording_service(tmp_path)
    old_episode_id = service.session.evaluation_context.identity.episode_id

    restarted = service.apply_command(
        _request("t0-reset", base_revision=0, command=ResetCommandV1())
    )

    assert isinstance(restarted.payload, CommandResponseV2)
    assert restarted.payload.result == "applied"
    assert restarted.payload.frame.recording is not None
    assert restarted.payload.frame.recording.lifecycle == "recording"
    assert restarted.payload.frame.recording.captured_transition_count == 0
    assert service.session.evaluation_context.identity.episode_id != old_episode_id
    assert service.evaluation_validated_transition_count == 0
    assert old_recorder.lifecycle == "recording"
    assert old_recorder.validated_transition_count == 0


def test_recording_candidate_frame_failure_precedes_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path)
    initial_session = service.session
    initial_frame = service.current_frame()

    def fail_frame(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected recording frame failure")

    monkeypatch.setattr(service_module, "build_debugger_frame", fail_frame)
    with pytest.raises(RuntimeError, match="recording frame failure"):
        service.apply_command(
            _request(
                "preappend-frame-failure",
                base_revision=0,
                command=KeyboardCommandV1(key="Enter"),
            )
        )

    assert service.session is initial_session
    assert service.current_frame() is initial_frame
    assert service.revision == 0
    assert recorder.lifecycle == "recording"
    assert recorder.validated_transition_count == 0


def test_finish_response_prebuild_failure_leaves_recorder_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path)
    initial_frame = service.current_frame()

    def fail_frame(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected finish frame failure")

    monkeypatch.setattr(service_module, "build_debugger_frame", fail_frame)
    with pytest.raises(RuntimeError, match="finish frame failure"):
        service.apply_command(
            _request(
                "finish-frame-failure",
                base_revision=0,
                command=FinishAndReviewCommandV1(),
            )
        )

    assert recorder.lifecycle == "recording"
    assert recorder.validated_transition_count == 0
    assert service.current_frame() is initial_frame
    assert service.revision == 0


def test_endpoint_outcome_frame_failure_precedes_recorder_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path, "basic_support")
    first = service.apply_command(
        _request(
            "endpoint-build-0", base_revision=0, command=KeyboardCommandV1(key="n")
        )
    )
    assert isinstance(first.payload, CommandResponseV2)
    initial_session = service.session
    initial_frame = service.current_frame()
    actual_build = service_module.build_debugger_frame
    calls = 0

    def fail_endpoint_candidate(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected endpoint outcome frame failure")
        return actual_build(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        service_module,
        "build_debugger_frame",
        fail_endpoint_candidate,
    )
    with pytest.raises(RuntimeError, match="endpoint outcome frame failure"):
        service.apply_command(
            _request(
                "endpoint-build-1",
                base_revision=1,
                command=KeyboardCommandV1(key="n"),
            )
        )

    assert service.session is initial_session
    assert service.current_frame() is initial_frame
    assert service.revision == 1
    assert recorder.lifecycle == "recording"
    assert recorder.validated_transition_count == 1


def test_begin_review_failure_retains_saved_reviewable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path)

    def fail_review(_self: DebuggerReplayRecorderV1) -> object:
        raise RuntimeError("injected review failure")

    monkeypatch.setattr(DebuggerReplayRecorderV1, "begin_review", fail_review)
    result = service.apply_command(
        _request(
            "finish-with-review-failure",
            base_revision=0,
            command=FinishAndReviewCommandV1(),
        )
    )

    assert isinstance(result.payload, CommandResponseV2)
    assert result.replay_handoff is None
    assert recorder.lifecycle == "saved"
    assert result.payload.frame.recording == recorder.status
    assert result.payload.frame.recording is not None
    assert result.payload.frame.recording.review_available
    assert not service.shutting_down


def test_endpoint_persistence_failure_preserves_committed_prefix_then_save_as(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path, "basic_support")
    first = service.apply_command(
        _request(
            "failure-script-0", base_revision=0, command=KeyboardCommandV1(key="n")
        )
    )
    assert isinstance(first.payload, CommandResponseV2)

    def fail_publish(*_args: object, **_kwargs: object) -> object:
        raise ReplaySaveError(
            "temporary_write_failed",
            path=None,
            detail="injected endpoint publication failure",
        )

    actual_publish = recording_module.publish_prepared_replay_bundle_v1
    monkeypatch.setattr(
        recording_module,
        "publish_prepared_replay_bundle_v1",
        fail_publish,
    )
    failed = service.apply_command(
        _request(
            "failure-script-1", base_revision=1, command=KeyboardCommandV1(key="n")
        )
    )

    assert isinstance(failed.payload, CommandResponseV2)
    assert service.revision == 2
    assert service.session.current_evaluation_frame.frame_index == 2
    assert recorder.validated_transition_count == 2
    assert recorder.lifecycle == "persistence_failed"
    assert failed.payload.frame.recording == recorder.status
    assert not service.faulted

    monkeypatch.setattr(
        recording_module,
        "publish_prepared_replay_bundle_v1",
        actual_publish,
    )
    recovered = service.apply_command(
        _request(
            "save-as-after-endpoint",
            base_revision=2,
            command=SaveAsCommandV1(file_name="endpoint-recovered.marlbg-replay.json"),
        )
    )

    assert isinstance(recovered.payload, CommandResponseV2)
    assert recovered.replay_handoff is not None
    assert recorder.lifecycle == "reviewing"
    assert recovered.payload.frame.recording == recorder.status
    assert (tmp_path / "endpoint-recovered.marlbg-replay.json").is_file()


def test_exit_waits_for_durable_save_and_retry_before_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path)
    actual_publish = recording_module.publish_prepared_replay_bundle_v1

    def fail_publish(*_args: object, **_kwargs: object) -> object:
        raise ReplaySaveError(
            "temporary_write_failed",
            path=None,
            detail="injected exit publication failure",
        )

    monkeypatch.setattr(
        recording_module,
        "publish_prepared_replay_bundle_v1",
        fail_publish,
    )
    failed_exit = service.apply_command(
        _request("exit-save-failure", base_revision=0, command=ExitCommandV1())
    )

    assert isinstance(failed_exit.payload, CommandResponseV2)
    assert failed_exit.payload.result == "applied"
    assert not failed_exit.shutdown_requested
    assert not service.shutting_down
    assert recorder.lifecycle == "persistence_failed"
    assert failed_exit.payload.frame.recording == recorder.status

    monkeypatch.setattr(
        recording_module,
        "publish_prepared_replay_bundle_v1",
        actual_publish,
    )
    retried_exit = service.apply_command(
        _request("exit-save-retry", base_revision=1, command=ExitCommandV1())
    )

    assert isinstance(retried_exit.payload, CommandResponseV2)
    assert retried_exit.payload.result == "shutdown_scheduled"
    assert retried_exit.shutdown_requested
    assert service.shutting_down
    assert recorder.lifecycle == "saved"
    assert retried_exit.payload.frame.recording == recorder.status


def test_retry_save_opens_review_without_replaying_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path)
    actual_publish = recording_module.publish_prepared_replay_bundle_v1
    calls = 0

    def fail_once(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ReplaySaveError(
                "temporary_write_failed",
                path=None,
                detail="injected first publication failure",
            )
        return actual_publish(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        recording_module,
        "publish_prepared_replay_bundle_v1",
        fail_once,
    )
    failed = service.apply_command(
        _request(
            "finish-before-retry",
            base_revision=0,
            command=FinishAndReviewCommandV1(),
        )
    )
    retried = service.apply_command(
        _request(
            "retry-save",
            base_revision=1,
            command=RetrySaveCommandV1(),
        )
    )

    assert isinstance(failed.payload, CommandResponseV2)
    assert failed.payload.frame.recording is not None
    assert failed.payload.frame.recording.lifecycle == "persistence_failed"
    assert isinstance(retried.payload, CommandResponseV2)
    assert retried.replay_handoff is not None
    assert retried.replay_handoff.current_frame().cursor.frame_index == 0
    assert recorder.lifecycle == "reviewing"
    assert calls == 2


def test_keyboard_interrupt_closeout_retries_then_uses_recovery_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path)
    submitted = service.apply_command(
        _request(
            "prefix-before-interrupt",
            base_revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )
    assert isinstance(submitted.payload, CommandResponseV2)
    actual_publish = recording_module.publish_prepared_replay_bundle_v1
    calls = 0

    def fail_twice(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise ReplaySaveError(
                "temporary_write_failed",
                path=None,
                detail="injected interrupt publication failure",
            )
        return actual_publish(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        recording_module,
        "publish_prepared_replay_bundle_v1",
        fail_twice,
    )
    closed = service.close_recording_for_keyboard_interrupt()

    assert closed.saved
    assert calls == 3
    assert ".recovery-" in closed.message
    assert recorder.lifecycle == "saved"
    recording_status = service.current_frame().recording
    assert recording_status == recorder.status
    assert recording_status is not None
    assert recording_status.completion_state == "interrupted"
    assert recording_status.completion_reason == "keyboard_interrupt"


def test_keyboard_interrupt_ordinary_retry_success_skips_recovery_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path)
    actual_publish = recording_module.publish_prepared_replay_bundle_v1
    calls = 0

    def fail_once(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ReplaySaveError(
                "temporary_write_failed",
                path=None,
                detail="injected one-shot closeout failure",
            )
        return actual_publish(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        recording_module,
        "publish_prepared_replay_bundle_v1",
        fail_once,
    )
    closed = service.close_recording_for_keyboard_interrupt()

    assert closed.saved
    assert calls == 2
    assert ".recovery-" not in closed.message
    assert recorder.lifecycle == "saved"
    assert service.current_frame().recording == recorder.status
    assert not tuple(tmp_path.glob("*.recovery-*.marlbg-replay.json"))


def test_keyboard_interrupt_closeout_failure_stays_recoverable_and_coherent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path)

    def fail_publish(*_args: object, **_kwargs: object) -> object:
        raise ReplaySaveError(
            "temporary_write_failed",
            path=None,
            detail="injected persistent closeout failure",
        )

    monkeypatch.setattr(
        recording_module,
        "publish_prepared_replay_bundle_v1",
        fail_publish,
    )
    closed = service.close_recording_for_keyboard_interrupt()

    assert not closed.saved
    assert recorder.lifecycle == "persistence_failed"
    recording_status = service.current_frame().recording
    assert recording_status == recorder.status
    assert recording_status is not None
    assert recording_status.retry_available
    assert not service.shutting_down


def test_concurrent_finish_publishes_and_hands_off_once(tmp_path: Path) -> None:
    service, recorder = _recording_service(tmp_path)
    requests = tuple(
        _request(
            f"concurrent-finish-{index}",
            base_revision=0,
            client_id=f"finish-client-{index}",
            command=FinishAndReviewCommandV1(),
        )
        for index in range(2)
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(service.apply_command, requests))

    assert sorted(result.outcome for result in results) == [
        "response",
        "stale_revision",
    ]
    assert sum(result.replay_handoff is not None for result in results) == 1
    assert recorder.lifecycle == "reviewing"
    assert service.revision == 1


@pytest.mark.parametrize(
    "lifecycle",
    ("persistence_failed", "saved", "reviewing"),
)
@pytest.mark.parametrize(
    ("command_name", "command"),
    (
        ("keyboard-movement", KeyboardCommandV1(key="w")),
        (
            "battlefield-pointer",
            BattlefieldPointerCommandV1(
                world_x=3.0,
                world_y=10.0,
                button="primary",
            ),
        ),
        (
            "roster-target",
            RosterSelectionCommandV1(role="target", global_slot=1),
        ),
        (
            "roster-control",
            RosterSelectionCommandV1(role="control", global_slot=1),
        ),
        ("actor-pov-target", ActorPovTargetActionCommandV1(target_action=0)),
    ),
)
def test_closed_recording_fences_every_scientific_command_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: str,
    command_name: str,
    command: object,
) -> None:
    service, recorder = _recording_service(tmp_path, view_mode="pov")
    _close_recording_in_lifecycle(service, recorder, lifecycle, monkeypatch)
    before_session = service.session
    before_frame = service.current_frame()
    before_revision = service.revision
    before_observer_count = service.evaluation_validated_transition_count
    before_cache_size = service.command_cache_size

    def forbidden_dispatch(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"closed recording dispatched {command_name}")

    monkeypatch.setattr(service_module, "dispatch_command", forbidden_dispatch)
    request = _request(
        f"closed-{lifecycle}-{command_name}",
        base_revision=before_revision,
        command=command,
    )
    fenced = service.apply_command(request)
    duplicate = service.apply_command(request)

    assert isinstance(fenced.payload, CommandResponseV2)
    assert fenced.payload.result == "no_op"
    assert fenced.payload.notice == (
        "Scientific controls are fenced because this recording is no longer "
        "capturing transitions."
    )
    assert fenced.payload.frame is before_frame
    assert isinstance(duplicate.payload, CommandResponseV2)
    assert duplicate.payload.result == "duplicate"
    assert service.session is before_session
    assert service.current_frame() is before_frame
    assert service.revision == before_revision
    assert service.evaluation_validated_transition_count == before_observer_count
    assert service.command_cache_size == before_cache_size + 1
    assert recorder.lifecycle == lifecycle
    assert not service.faulted


def test_closed_recording_retains_only_exact_presentation_command_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path)
    _close_recording_in_lifecycle(service, recorder, "saved", monkeypatch)

    commands = (
        SetViewCommandV1(view_mode="pov"),
        SetPresetCommandV1(preset="debug"),
        KeyboardCommandV1(key="g"),
        KeyboardCommandV1(key="v"),
        KeyboardCommandV1(key="p"),
        KeyboardCommandV1(key="?"),
    )
    results: list[CommandResponseV2] = []
    for index, command in enumerate(commands):
        outcome = service.apply_command(
            _request(
                f"closed-presentation-{index}",
                base_revision=service.revision,
                command=command,
            )
        )
        assert isinstance(outcome.payload, CommandResponseV2)
        results.append(outcome.payload)

    assert [result.result for result in results] == [
        "applied",
        "applied",
        "applied",
        "applied",
        "no_op",
        "no_op",
    ]
    assert service.current_frame().view_mode == "pov"
    assert service.current_frame().preset == "debug"
    assert service.session.show_ranges is False
    assert service.session.verbose_logging is True
    assert service.revision == 5
    assert service.evaluation_validated_transition_count == 0
    assert recorder.lifecycle == "saved"
    assert service.current_frame().recording == recorder.status
    assert all(
        result.notice
        != "Scientific controls are fenced because this recording is no longer "
        "capturing transitions."
        for result in results
    )


@pytest.mark.parametrize(
    ("boundary", "expected_origin", "expected_reason"),
    (
        ("action_build", "policy", "policy_failure"),
        ("simulation", "simulation", "simulation_failure"),
        ("capture", "capture", "capture_failure"),
        ("validation", "validation", "validation_failure"),
    ),
)
def test_typed_transition_failure_saves_last_prefix_with_exact_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    expected_origin: str,
    expected_reason: str,
) -> None:
    service, recorder = _recording_service(tmp_path)
    prefix = service.apply_command(
        _request(
            "failure-prefix",
            base_revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )
    assert isinstance(prefix.payload, CommandResponseV2)
    prefix_session = service.session

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("private typed failure detail")

    if boundary == "action_build":
        monkeypatch.setattr(control_module, "build_interactive_joint_action", fail)
    elif boundary == "simulation":
        monkeypatch.setattr(control_module, "step", fail)
    elif boundary == "capture":
        monkeypatch.setattr(
            control_module,
            "capture_evaluation_transition_unit_v1",
            fail,
        )
    else:
        monkeypatch.setattr(control_module, "EvaluationTransitionViewV1", fail)

    request = _request(
        f"{boundary}-failure",
        base_revision=1,
        command=KeyboardCommandV1(key="Enter"),
    )
    failed = service.apply_command(request)
    duplicate = service.apply_command(request)

    assert isinstance(failed.payload, CommandResponseV2)
    assert failed.payload.result == "applied"
    assert failed.payload.notice is not None
    assert "private typed failure detail" not in failed.payload.notice
    assert service.session is prefix_session
    assert service.session.current_evaluation_frame.frame_index == 1
    assert recorder.validated_transition_count == 1
    assert recorder.lifecycle == "saved"
    assert failed.payload.frame.recording == recorder.status
    assert not service.faulted
    bundle = recorder.bundle
    assert bundle is not None
    assert bundle.replay.completion.completion_state == "failed"
    assert bundle.replay.completion.failure_origin == expected_origin
    assert bundle.replay.completion.end_or_failure_reason == expected_reason
    assert isinstance(duplicate.payload, CommandResponseV2)
    assert duplicate.payload.result == "duplicate"
    assert duplicate.replay_handoff is None


def test_typed_transition_failure_with_save_failure_stays_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path)
    actual_publish = recording_module.publish_prepared_replay_bundle_v1

    def fail_step(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("private simulator failure detail")

    def fail_publish(*_args: object, **_kwargs: object) -> object:
        raise ReplaySaveError(
            "temporary_write_failed",
            path=None,
            detail="private persistence detail",
        )

    monkeypatch.setattr(control_module, "step", fail_step)
    monkeypatch.setattr(
        recording_module,
        "publish_prepared_replay_bundle_v1",
        fail_publish,
    )
    failed = service.apply_command(
        _request(
            "simulation-and-save-failure",
            base_revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )

    assert isinstance(failed.payload, CommandResponseV2)
    assert recorder.lifecycle == "persistence_failed"
    assert recorder.validated_transition_count == 0
    assert failed.payload.frame.recording == recorder.status
    assert failed.payload.notice is not None
    assert "private" not in failed.payload.notice
    assert not service.faulted

    monkeypatch.setattr(
        recording_module,
        "publish_prepared_replay_bundle_v1",
        actual_publish,
    )
    recovered = service.apply_command(
        _request(
            "recover-failed-transition",
            base_revision=1,
            command=RetrySaveCommandV1(),
        )
    )
    assert isinstance(recovered.payload, CommandResponseV2)
    assert recovered.replay_handoff is not None
    assert recorder.lifecycle == "reviewing"


@pytest.mark.parametrize("failure_site", ("pov_sanitizer", "result_envelope"))
def test_transition_result_packaging_failure_saves_uncommitted_candidate_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    service, recorder = _recording_service(
        tmp_path,
        view_mode="pov" if failure_site == "pov_sanitizer" else "researcher",
    )
    initial_session = service.session
    tracked_step = Mock(wraps=control_module.step)
    tracked_capture = Mock(wraps=control_module.capture_evaluation_transition_unit_v1)

    def fail_packaging(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("private transition packaging detail")

    monkeypatch.setattr(control_module, "step", tracked_step)
    monkeypatch.setattr(
        control_module,
        "capture_evaluation_transition_unit_v1",
        tracked_capture,
    )
    if failure_site == "pov_sanitizer":
        monkeypatch.setattr(input_module, "sanitize_pov_pending_target", fail_packaging)
    else:
        monkeypatch.setattr(input_module, "_result", fail_packaging)

    request = _request(
        f"{failure_site}-failure",
        base_revision=0,
        command=KeyboardCommandV1(key="Enter"),
    )
    failed = service.apply_command(request)
    duplicate = service.apply_command(request)

    assert tracked_step.call_count == 1
    assert tracked_capture.call_count == 1
    assert isinstance(failed.payload, CommandResponseV2)
    assert failed.payload.result == "applied"
    assert failed.payload.notice is not None
    assert "private transition packaging detail" not in failed.payload.notice
    assert service.session is initial_session
    assert service.session.current_evaluation_frame.frame_index == 0
    assert recorder.validated_transition_count == 0
    assert recorder.lifecycle == "saved"
    assert failed.payload.frame.recording == recorder.status
    assert not service.faulted
    bundle = recorder.bundle
    assert bundle is not None
    assert bundle.replay.completion.validated_transition_count == 0
    assert bundle.replay.completion.completion_state == "failed"
    assert bundle.replay.completion.failure_origin == "validation"
    assert bundle.replay.completion.end_or_failure_reason == "validation_failure"
    assert isinstance(duplicate.payload, CommandResponseV2)
    assert duplicate.payload.result == "duplicate"
    assert tracked_step.call_count == 1
    assert tracked_capture.call_count == 1


def test_recorder_precommit_append_failure_closes_old_prefix_as_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path)
    initial_session = service.session

    def fail_append(
        _self: DebuggerReplayRecorderV1,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        raise ValueError("private invalid transition unit")

    monkeypatch.setattr(DebuggerReplayRecorderV1, "append", fail_append)
    failed = service.apply_command(
        _request(
            "recorder-validation-failure",
            base_revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )

    assert isinstance(failed.payload, CommandResponseV2)
    assert service.session is initial_session
    assert recorder.validated_transition_count == 0
    assert recorder.lifecycle == "saved"
    bundle = recorder.bundle
    assert bundle is not None
    assert bundle.replay.completion.failure_origin == "validation"
    assert not service.faulted


def test_recorder_processing_failure_commits_validated_unit_then_closes_prefix(
    tmp_path: Path,
) -> None:
    service, recorder = _recording_service(
        tmp_path,
        reducers=(_FailingAdvanceReducer(),),
    )
    result = service.apply_command(
        _request(
            "reducer-processing-failure",
            base_revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )

    assert isinstance(result.payload, CommandResponseV2)
    assert service.session.current_evaluation_frame.frame_index == 1
    assert service.revision == 1
    assert recorder.validated_transition_count == 1
    assert recorder.observer_lifecycle_state == "finalized"
    assert recorder.lifecycle == "saved"
    assert result.payload.frame.recording == recorder.status
    assert result.payload.frame.recording is not None
    assert result.payload.frame.recording.completion_state == "interrupted"
    assert result.payload.frame.recording.completion_reason == (
        "evaluation_processing_failure"
    )
    bundle = recorder.bundle
    assert bundle is not None
    assert bundle.replay.processing_status.status == "failed"
    assert bundle.replay.processing_status.processed_transition_count == 0
    assert bundle.replay.completion.validated_transition_count == 1
    assert not service.faulted


def test_horizon_reducer_failure_preserves_complete_rollout_and_failed_processing(
    tmp_path: Path,
) -> None:
    service, recorder = _recording_service(
        tmp_path,
        "basic_support",
        reducers=(_HorizonFailingAdvanceReducer(),),
    )
    first = service.apply_command(
        _request(
            "horizon-reducer-first",
            base_revision=0,
            command=KeyboardCommandV1(key="n"),
        )
    )
    endpoint = service.apply_command(
        _request(
            "horizon-reducer-failure",
            base_revision=1,
            command=KeyboardCommandV1(key="n"),
        )
    )

    assert isinstance(first.payload, CommandResponseV2)
    assert isinstance(endpoint.payload, CommandResponseV2)
    assert service.session.current_evaluation_frame.frame_index == 2
    assert service.session.reached_declared_horizon
    assert service.revision == 2
    assert recorder.validated_transition_count == 2
    assert recorder.observer_lifecycle_state == "finalized"
    assert recorder.lifecycle == "saved"
    assert endpoint.payload.frame.recording == recorder.status
    assert endpoint.payload.frame.recording is not None
    assert endpoint.payload.frame.recording.completion_state == "complete"
    assert endpoint.payload.frame.recording.completion_reason is None
    bundle = recorder.bundle
    assert bundle is not None
    completion = bundle.replay.completion
    assert completion.completion_state == "complete"
    assert completion.validated_transition_count == 2
    assert completion.completion_bases == ("declared_horizon",)
    assert completion.terminated is False
    assert completion.truncated is False
    processing = bundle.replay.processing_status
    assert processing.status == "failed"
    assert processing.processed_transition_count == 1
    assert processing.failure is not None
    assert processing.failure.stage == "reducer_advance"
    assert processing.failure.attempted_transition_index == 1
    assert not service.faulted
