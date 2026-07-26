"""Focused revision, idempotency, and concurrency proofs for DebuggerService."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import jax.numpy as jnp
import pytest
import scripts.dev.visual_debugger.control as control_module
import scripts.dev.visual_debugger.service as service_module
from scripts.dev.visual_debugger.control import create_session, select_clicked_target
from scripts.dev.visual_debugger.input import InputDispatchResult
from scripts.dev.visual_debugger.protocol import (
    ApiErrorV1,
    CommandRequestV1,
    CommandResponseV1,
    ExitCommandV1,
    KeyboardCommandV1,
    RosterSelectionCommandV1,
    SetPresetCommandV1,
    SetViewCommandV1,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.scene_adapter import build_battlefield_scene
from scripts.dev.visual_debugger.service import DebuggerService


def _service(
    scenario_name: str = "arena_5v5",
    *,
    include_stress: bool = False,
) -> DebuggerService:
    session = create_session(
        get_scenario(scenario_name),
        seed=0,
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


def test_initial_and_current_frame_reads_are_coherent_and_non_mutating() -> None:
    service = _service()
    initial_session = service.session
    initial_key = initial_session.key

    first = service.current_frame()
    second = service.current_frame()

    assert first is second
    assert first.revision == 0
    assert first.simulator_step == 0
    assert first.transition_id is None
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
    assert isinstance(result.payload, CommandResponseV1)
    assert result.payload.result == "applied"
    assert result.payload.frame.revision == 1
    assert result.payload.frame.simulator_step == 0
    assert service.session.state is initial.state
    assert bool(jnp.array_equal(service.session.key, initial.key))


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

    assert isinstance(applied.payload, CommandResponseV1)
    assert applied.payload.result == "applied"
    assert applied.payload.frame.simulator_step == 1
    assert applied.payload.frame.transition_id == 1
    assert isinstance(duplicate.payload, CommandResponseV1)
    assert duplicate.payload.result == "duplicate"
    assert duplicate.payload.frame.simulator_step == 1
    assert conflicting.outcome == "command_id_conflict"
    assert stale.outcome == "stale_revision"
    assert isinstance(stale_duplicate.payload, CommandResponseV1)
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

    assert isinstance(result.payload, CommandResponseV1)
    assert result.payload.result == "no_op"
    assert result.payload.frame.revision == 0
    assert result.payload.frame.simulator_step == 0
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

    assert isinstance(result.payload, CommandResponseV1)
    assert result.payload.result == "no_op"
    assert result.payload.frame.revision == 0
    assert result.payload.frame.simulator_step == 0
    assert service.session is initial
    assert result.payload.notice is not None
    assert "canonical no-combat tuple" in result.payload.notice


def test_entering_pov_clears_a_hidden_pending_target_without_stepping() -> None:
    service = _service()
    researcher = build_battlefield_scene(service.session, audience="researcher")
    pov = build_battlefield_scene(service.session, audience="agent_pov")
    hidden_slots = {agent.global_slot for agent in researcher.agents} - {
        agent.global_slot for agent in pov.agents
    }
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
    assert isinstance(selected.payload, CommandResponseV1)
    assert service.session.pending_action.selected_global_target_slot == hidden_target
    before_view = service.session

    changed_view = service.apply_command(
        _request(
            "enter-pov",
            base_revision=1,
            command=SetViewCommandV1(view_mode="pov"),
        )
    )

    assert isinstance(changed_view.payload, CommandResponseV1)
    assert changed_view.payload.frame.view_mode == "pov"
    assert service.session.pending_action.selected_global_target_slot is None
    assert service.session.state is before_view.state
    assert bool(jnp.array_equal(service.session.key, before_view.key))
    assert int(service.session.state.step_count) == 0


def test_initial_pov_service_clears_hidden_pending_target() -> None:
    session = create_session(
        get_scenario("arena_5v5"),
        seed=0,
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    pov_slots = {
        agent.global_slot
        for agent in build_battlefield_scene(session, audience="agent_pov").agents
    }
    hidden_target = min(
        agent.global_slot
        for agent in build_battlefield_scene(session, audience="researcher").agents
        if agent.global_slot not in pov_slots
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
    assert service.current_frame().hud.selected_global_slot is None
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
    assert isinstance(duplicate.payload, CommandResponseV1)
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

    assert isinstance(accepted_exit.payload, CommandResponseV1)
    assert accepted_exit.payload.result == "shutdown_scheduled"
    assert accepted_exit.shutdown_requested
    assert service.shutting_down
    assert {result.outcome for result in results} == {"server_shutting_down"}
    for result in results:
        assert isinstance(result.payload, ApiErrorV1)
        assert result.payload.error_code == "server_shutting_down"
    assert isinstance(duplicate_exit.payload, CommandResponseV1)
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
    assert isinstance(applied.payload, CommandResponseV1)
    assert applied.payload.result == "applied"
    assert applied.payload.frame.revision == 1
    assert applied.payload.frame.simulator_step == 1

    for index in range(256):
        no_op = service.apply_command(
            _request(
                f"post-submit-no-op-{index}",
                base_revision=1,
                command=KeyboardCommandV1(key="F13"),
            )
        )
        assert isinstance(no_op.payload, CommandResponseV1)
        assert no_op.payload.result == "no_op"

    assert service.command_cache_size == 256
    replayed = service.apply_command(submit)

    assert replayed.outcome == "stale_revision"
    assert isinstance(replayed.payload, ApiErrorV1)
    assert replayed.payload.error_code == "stale_revision"
    assert replayed.payload.latest_frame is not None
    assert replayed.payload.latest_frame.revision == 1
    assert replayed.payload.latest_frame.simulator_step == 1
    assert service.revision == 1
    assert int(service.session.state.step_count) == 1


def test_stress_scenario_requires_explicit_service_authorization() -> None:
    session = create_session(
        get_scenario("charge_convergence"),
        seed=0,
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
