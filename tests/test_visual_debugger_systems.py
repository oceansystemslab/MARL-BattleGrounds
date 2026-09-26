"""Check persistent Systems through the existing DevClient step and identity owners.

CPU-only trajectories compare the public System route with accepted DevClient
steps. Actor history stays private, survives accepted turns and clears on restart.
A failed joint decision installs neither a transition nor proposed memory. Legacy
controller identity bytes are checked by the existing bridge regression files.
"""

from collections.abc import Generator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger import control
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.protocol import DebuggerCommandV1
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.service import DebuggerService, ServiceCommandResult
from tests.llm_fixtures import FakeModel, server
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds import llm
from marl_battlegrounds.environment import make
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV2,
    EvaluationFrameV3,
    EvaluationTransitionV1,
)
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
    apply_systems,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations


def _create(
    first: System | None, second: System | None, *, other: str = "manual"
) -> DebuggerSession:
    return control.create_session(
        get_scenario("arena_5v5"),
        seed=7,
        evaluation_launch_specification=debugger_test_launch_specification(7),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="system:first" if first is not None else other,
        team_b_controller="system:second" if second is not None else other,
        systems=(first, second),
        execution_information_mode="shared_obs",
    )


def _init(variables: PolicyTree, inputs: SystemInput, key: jax.Array) -> list[int]:
    return [0 for _ in inputs.valid]


def _stay(
    variables: PolicyTree,
    memory: PolicyTree,
    inputs: SystemInput,
    key: jax.Array,
) -> SystemOutput:
    zeros = jnp.zeros(inputs.active_mask.shape, dtype=jnp.int32)
    return SystemOutput(
        ActorAction(zeros, zeros, zeros),
        [
            int(value) + int(valid)
            for value, valid in zip(memory, inputs.valid, strict=True)
        ],
    )


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("other", ["manual", "random_valid"])
def test_system_memory_matches_public_apply_and_single_step(
    team: int, other: str
) -> None:
    method = System("Counter", _stay, init=_init, execution="host")
    session = _create(
        method if team == 0 else None, method if team == 1 else None, other=other
    )
    assert session.system_memory is not None
    assert session.environment_state is not None
    assert session.systems is not None
    initial_root = session.evaluation_context.seed_protocol.focal_policy_seed
    for turn in range(2):
        assert session.systems is not None
        wrapped, memory = session.environment_state, session.system_memory
        assert wrapped is not None and memory is not None
        action, expected_memory, _ = apply_systems(
            session.systems[0],
            session.systems[1],
            memory,
            Observations(wrapped.observation, wrapped.source_availability),
            wrapped,
            jax.random.fold_in(jax.random.key(cast(int, initial_root)), turn),
        )
        _, expected, *_ = make(
            "tdm", metrics="none", balance_spawn_locations=False
        ).step(
            jax.random.split(session.key)[1],
            wrapped,
            action,
        )
        session = control.submit_interactive(session)
        assert session.system_memory is not None
        np.testing.assert_array_equal(
            session.system_memory.team_a, expected_memory.team_a
        )
        np.testing.assert_array_equal(
            session.system_memory.team_b, expected_memory.team_b
        )
        assert session.environment_state is not None
        assert session.environment_state.core_state is session.state
        for actual, wanted in zip(
            jax.tree.leaves((session.state, session.observation, session.action_mask)),
            jax.tree.leaves(
                (expected.core_state, expected.observation, expected.action_mask)
            ),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, wanted)
        assert session.current_evaluation_frame.frame_index == turn + 1
        assert int(session.environment_state.cumulative_transition_count) == turn + 1
    restarted = control.reset_session(session)
    assert restarted.system_memory is not None
    assert (
        restarted.system_memory.team_a if team == 0 else restarted.system_memory.team_b
    ) == [0]
    assert restarted.evaluation_context.seed_protocol.focal_policy_seed == initial_root
    assert (
        restarted.evaluation_context.identity.episode_id
        != session.evaluation_context.identity.episode_id
    )


@pytest.mark.parametrize("both", [False, True])
def test_llm_history_and_unknown_identity_survive_accepted_devclient_turns(
    both: bool,
) -> None:
    model = FakeModel()
    with server(model.serve) as url, llm.Client(url) as client:
        method = llm.make_system("fake", client=client, history_turns=2)
        session = _create(method, method if both else None)
        memory = session.system_memory
        assert memory is not None
        for turn in range(2):
            session = control.submit_interactive(session)
            assert session.system_memory is not None
            for history in session.system_memory.team_a[0]:
                assert len(history) == turn + 1
        assert len(model.generations()) == (20 if both else 10)
        for slot in range(10 if both else 5):
            assignment = session.evaluation_context.policy_assignments[slot]
            assert isinstance(assignment, AssignedPolicySlotV2)
            assert assignment.policy_kind == "system"
            assert assignment.policy_content_digest is None
            assert assignment.execution_mode is None
            assert assignment.algorithm_id is None
        assert memory.team_a == [tuple(() for _ in range(5))]
        restarted = control.reset_session(session)
        assert restarted.system_memory is not None
        assert restarted.system_memory.team_a == [tuple(() for _ in range(5))]
        assert (
            client.request("/tokenize", {"messages": [{"content": "Still open"}]})[
                "count"
            ]
            > 0
        )


def test_second_system_failure_does_not_advance_first_memory_or_game() -> None:
    def fail(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, key: jax.Array
    ) -> SystemOutput:
        raise RuntimeError("Declared test failure")

    first = System("Counter", _stay, init=_init, execution="host")
    second = System("Failure", fail, init=_init, execution="host")
    session = _create(first, second)
    with pytest.raises(
        control.DebuggerTransitionFailureV1, match="action_build"
    ) as failure:
        control.submit_interactive(session)
    assert failure.value.stable_code == "policy_action_build_failed"
    assert session.current_evaluation_frame.frame_index == 0
    assert session.system_memory is not None
    assert session.system_memory.team_a == session.system_memory.team_b == [0]
    assert session.environment_state is not None
    assert int(session.environment_state.cumulative_transition_count) == 0


def test_changed_system_content_changes_match_identity_with_same_alias() -> None:
    first = System("Counter", _stay, init=_init, execution="host", variables=1)
    original = _create(first, None)
    changed = control.set_combat_configuration(
        original,
        team_a_controller="system:first",
        team_b_controller="manual",
        execution_information_mode="shared_obs",
        systems=(replace(first, variables=2), None),
    )
    assert changed.system_ids != original.system_ids
    assert (
        changed.evaluation_context.identity.matchup_id
        != original.evaluation_context.identity.matchup_id
    )
    with pytest.raises(ValueError, match="SharedObs"):
        control.set_combat_configuration(
            original,
            team_a_controller="system:first",
            team_b_controller="manual",
            execution_information_mode="no_shared_obs",
        )


def test_system_continuation_rejects_old_memory_or_changed_method() -> None:
    method = System("Counter", _stay, init=_init, execution="host")
    initial = _create(method, None)
    advanced = control.submit_interactive(initial)
    assert advanced.systems is not None
    with pytest.raises(ValueError, match="continuation identity"):
        replace(advanced, system_memory=initial.system_memory)
    with pytest.raises(ValueError, match="continuation identity"):
        replace(
            advanced, systems=(replace(method, name="Different"), advanced.systems[1])
        )
    assert advanced.environment_state is not None
    with pytest.raises(ValueError, match="source permissions"):
        replace(
            advanced,
            raw_continuation_identity=None,
            environment_state=advanced.environment_state._replace(
                source_availability=jnp.zeros((10, 10), jnp.bool_)
            ),
        )


def test_shared_system_setup_copies_and_registers_once_with_separate_memories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copies: list[System] = []
    registrations: list[object] = []
    freeze = control.freeze_evaluation_method
    register = control.normalize_system_registration

    def copied(method: System) -> System:
        copies.append(method)
        return cast(System, freeze(method))

    def registered(
        value: object, *, phase: str, frozen: bool = False
    ) -> tuple[str, dict[str, object]]:
        registrations.append(value)
        return register(value, phase=phase, frozen=frozen)

    monkeypatch.setattr(control, "freeze_evaluation_method", copied)
    monkeypatch.setattr(control, "normalize_system_registration", registered)
    method = System(
        "Counter",
        _stay,
        init=_init,
        execution="host",
        variables=np.zeros(128, np.float32),
    )
    session = _create(method, method)
    assert len(copies) == len(registrations) == 1
    assert session.systems is not None and session.systems[0] is session.systems[1]
    assert session.system_memory is not None
    assert session.system_memory.team_a is not session.system_memory.team_b
    control.submit_interactive(session)
    assert len(copies) == len(registrations) == 1


@pytest.mark.parametrize("manual_team", [0, 1])
def test_manual_draft_reaches_only_its_own_system_adapter(manual_team: int) -> None:
    method = System("Counter", _stay, init=_init, execution="host")
    session = _create(
        None if manual_team == 0 else method, None if manual_team == 1 else method
    )
    slot = manual_team * 5
    session = control.select_controlled_actor(session, slot)
    session = control.set_pending_movement(session, 3)
    result = control.submit_interactive(session)
    assert result.incoming_evaluation_view is not None
    facts = result.incoming_evaluation_view.transition.facts.action_acceptance_facts
    submitted = facts.submitted_joint_action
    assert submitted.move[slot] == 3
    assert sum(value != 0 for value in submitted.move) == 1
    assert result.pending_actions[slot].move_action == 0


def _send(service: DebuggerService, command: DebuggerCommandV1) -> ServiceCommandResult:
    from uuid import uuid4

    from scripts.dev.visual_debugger.protocol import CommandRequestV1

    return service.apply_command(
        CommandRequestV1(
            client_id="system-test",
            command_id=uuid4().hex,
            base_revision=service.revision,
            command=command,
        )
    )


def _finish(service: DebuggerService) -> None:
    import time

    from scripts.dev.visual_debugger.protocol import FinishSystemCommandV1

    while service.current_frame().system_operation is not None:
        operation = service.current_frame().system_operation
        assert operation is not None
        assert operation.state != "failed"
        _send(service, FinishSystemCommandV1(operation_id=operation.operation_id))
        assert not service.faulted
        time.sleep(0.01)


def _choose(service: DebuggerService, name: str = "counter") -> ServiceCommandResult:
    from scripts.dev.visual_debugger.protocol import SetCombatConfigurationCommandV1

    return _send(
        service,
        SetCombatConfigurationCommandV1(
            team_a_controller=f"system:{name}",
            team_b_controller="manual",
            execution_information_mode="shared_obs",
        ),
    )


def test_declared_system_service_loads_lazily_and_accepts_each_turn_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.dev.visual_debugger import system_worker
    from scripts.dev.visual_debugger.protocol import KeyboardCommandV1, SetViewCommandV1
    from scripts.dev.visual_debugger.service import DebuggerService

    calls: list[str] = []

    def factory(reference: str) -> System:
        calls.append(reference)
        return System("Counter", _stay, init=_init, execution="host")

    monkeypatch.setattr(system_worker, "load_factory", factory)
    service = DebuggerService(
        _create(None, None),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        offered_systems={"counter": "test:counter"},
    )
    try:
        assert calls == []
        _choose(service, "undeclared")
        assert calls == []
        _choose(service)
        assert service.session.current_evaluation_frame.frame_index == 0
        _finish(service)
        assert calls == ["test:counter"]
        assert service.session.system_memory is not None
        assert service.session.system_memory.team_a == [0]
        for turn in range(2):
            _send(service, KeyboardCommandV1(key="Enter"))
            _send(service, SetViewCommandV1(view_mode="pov"))
            _finish(service)
            assert service.session.current_evaluation_frame.frame_index == turn + 1
            assert service.session.system_memory is not None
            assert service.session.system_memory.team_a == [turn + 1]
        assert len(calls) == 1
    finally:
        service.close()


def test_cancelled_factory_is_not_installed_and_late_scope_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager
    from threading import Event

    from scripts.dev.visual_debugger import system_worker
    from scripts.dev.visual_debugger.protocol import CancelSystemCommandV1
    from scripts.dev.visual_debugger.service import DebuggerService

    started, release = Event(), Event()
    events: list[str] = []

    @contextmanager
    def scope(recording: bool) -> Generator[None]:
        events.append("opened")
        try:
            yield
        finally:
            events.append("closed")

    def factory(reference: str) -> System:
        started.set()
        assert release.wait(20)
        return System(
            "Counter", _stay, init=_init, execution="host", resource_scope=scope
        )

    monkeypatch.setattr(system_worker, "load_factory", factory)
    service = DebuggerService(
        _create(None, None),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        offered_systems={"counter": "test:counter"},
    )
    before = service.session
    try:
        _choose(service)
        assert started.wait(5)
        operation = service.current_frame().system_operation
        assert operation is not None
        _send(service, CancelSystemCommandV1(operation_id=operation.operation_id))
        assert service.session is before
        operation = service.current_frame().system_operation
        assert operation is not None and operation.state == "cancelling"
        release.set()
        _finish(service)
        assert service.session is before
        assert events == ["opened", "closed"]
    finally:
        release.set()
        service.close()


@pytest.mark.parametrize("custom", [False, True])
def test_declared_llm_records_verified_replay_and_actual_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    custom: bool,
) -> None:
    import json
    import time

    from examples.llm import parse_words, prompt_for_words
    from scripts.dev.visual_debugger import system_worker
    from scripts.dev.visual_debugger.protocol import (
        FinishAndReviewCommandV1,
        KeyboardCommandV1,
    )
    from tests.test_visual_debugger_service import (
        _recording_service,  # pyright: ignore[reportPrivateUsage]
    )

    model = FakeModel(
        "stay no_combat" if custom else '{"move":"stay","combat":"no_combat"}'
    )
    with server(model.serve) as url:

        def factory(reference: str) -> System:
            return llm.make_system(
                "fake",
                server_url=url,
                history_turns=2,
                records="full",
                prompt_builder=prompt_for_words if custom else None,
                reply_parser=parse_words if custom else None,
                custom_version="words-v1" if custom else None,
            )

        monkeypatch.setattr(system_worker, "load_factory", factory)
        previous, recorder = _recording_service(tmp_path)
        service = DebuggerService(
            previous.session,
            view_mode="researcher",
            preset="analysis",
            include_stress=False,
            recorder=recorder,
            offered_systems={"counter": "test:counter"},
        )
        try:
            _choose(service)
            _finish(service)
            for _ in range(2):
                _send(service, KeyboardCommandV1(key="Enter"))
                _finish(service)
            assert service.session.current_evaluation_frame.frame_index == 2
            directory = service.system_evidence_directory
            assert directory is not None
            rows = list(llm.read_calls(directory))
            assert len(rows) == 10
            assert {row["outcome"] for row in rows} == {"played"}
            _send(service, FinishAndReviewCommandV1())
            until = time.monotonic() + 10
            while not json.loads((directory / "session.json").read_text())[
                "saved_prefixes"
            ]:
                assert time.monotonic() < until
                time.sleep(0.01)
            metadata = json.loads((directory / "session.json").read_text())
            assert metadata["accepted_turns"] == 2
            assert metadata["saved_prefixes"][0]["transitions"] == 2
            assert len(model.generations()) == 10
        finally:
            service.close()


def test_cancelled_llm_reply_cannot_advance_or_commit_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import Event

    from scripts.dev.visual_debugger import system_worker
    from scripts.dev.visual_debugger.protocol import (
        CancelSystemCommandV1,
        KeyboardCommandV1,
        ResetCommandV1,
    )

    started, release = Event(), Event()
    model = FakeModel()

    def reply(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        if handler.path.endswith("completions"):
            started.set()
            assert release.wait(20)
        model.serve(handler, payload)

    with server(reply) as url:

        def factory(reference: str) -> System:
            return llm.make_system("fake", server_url=url, history_turns=2)

        monkeypatch.setattr(system_worker, "load_factory", factory)
        service = DebuggerService(
            _create(None, None),
            view_mode="researcher",
            preset="analysis",
            include_stress=False,
            offered_systems={"counter": "test:counter"},
        )
        try:
            _choose(service)
            _finish(service)
            before = service.session
            _send(service, KeyboardCommandV1(key="Enter"))
            assert started.wait(10)
            operation = service.current_frame().system_operation
            assert operation is not None
            _send(service, CancelSystemCommandV1(operation_id=operation.operation_id))
            assert service.session is before
            assert service.current_presentation().outcome == "response"
            release.set()
            _finish(service)
            assert service.session is before
            _send(service, ResetCommandV1())
            _finish(service)
            _send(service, KeyboardCommandV1(key="Enter"))
            _finish(service)
            assert service.session.current_evaluation_frame.frame_index == 1
            assert len(model.generations()) == 10
        finally:
            release.set()
            service.close()


def test_authored_system_replacement_publishes_only_after_adoption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.dev.visual_debugger import system_worker
    from scripts.dev.visual_debugger.authoring_service import (
        DevCurrentBufferSourceV1,
        DevScenarioLoadService,
        LoadedDevScenarioSnapshotV1,
        debugger_scenario_from_snapshot,
    )
    from scripts.dev.visual_debugger.authoring_store import DevAssetStore
    from scripts.dev.visual_debugger.system_worker import PendingSystemOperation
    from tests.test_dev_client_authoring import new_scenario_draft

    def factory(reference: str) -> System:
        return System("Counter", _stay, init=_init, execution="host")

    monkeypatch.setattr(system_worker, "load_factory", factory)
    service = DebuggerService(
        _create(None, None),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        offered_systems={"counter": "test:counter"},
    )

    def install(snapshot: LoadedDevScenarioSnapshotV1) -> str | None:
        result = service.load_scenario(
            debugger_scenario_from_snapshot(snapshot),
            on_installed=lambda: loader.accept_installed_snapshot(snapshot),
        )
        return result.identifier if isinstance(result, PendingSystemOperation) else None

    loader = DevScenarioLoadService(DevAssetStore(tmp_path), install_snapshot=install)
    try:
        _choose(service)
        _finish(service)
        before = service.session
        draft = new_scenario_draft("system_authored", red_zone_depth=0)
        result = loader.load(
            DevCurrentBufferSourceV1(asset_kind="scenario", draft=draft)
        )
        assert not result.ok and result.pending_operation_id is not None
        assert result.summary is None and result.problems == ()
        assert loader.current_snapshot is None and service.session is before
        _finish(service)
        assert loader.current_snapshot is not None
        assert loader.current_snapshot.source.draft is draft
        assert service.session is not before
        assert service.session.system_memory is not None
        assert service.session.system_memory.team_a == [0]
        assert service.session.current_evaluation_frame.frame_index == 0
        assert service.session.evaluation_context.identity.scenario is not None
    finally:
        service.close()


@pytest.mark.parametrize("after_accept", [False, True])
def test_model_failure_and_post_accept_record_error_keep_true_call_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    after_accept: bool,
) -> None:
    from scripts.dev.visual_debugger import system_worker
    from scripts.dev.visual_debugger.protocol import KeyboardCommandV1
    from scripts.dev.visual_debugger.replay_recorder import DebuggerReplayRecorder
    from tests.test_visual_debugger_service import (
        _recording_service,  # pyright: ignore[reportPrivateUsage]
    )

    model = FakeModel() if after_accept else FakeModel("not an action")
    with server(model.serve) as url:

        def factory(reference: str) -> System:
            return llm.make_system("fake", server_url=url)

        monkeypatch.setattr(system_worker, "load_factory", factory)
        previous, recorder = _recording_service(tmp_path)
        service = DebuggerService(
            previous.session,
            view_mode="researcher",
            preset="analysis",
            include_stress=False,
            recorder=recorder,
            offered_systems={"counter": "test:counter"},
        )
        try:
            _choose(service)
            _finish(service)
            directory = service.system_evidence_directory
            assert directory is not None
            if after_accept:
                original = DebuggerReplayRecorder.append

                def fail_after_append(
                    self: DebuggerReplayRecorder,
                    transition: EvaluationTransitionV1,
                    successor_frame: EvaluationFrameV3,
                ) -> None:
                    original(self, transition, successor_frame)
                    raise RuntimeError("Injected processing failure after acceptance")

                monkeypatch.setattr(DebuggerReplayRecorder, "append", fail_after_append)
            _send(service, KeyboardCommandV1(key="Enter"))
            _finish(service)
            assert not service.faulted
            expected = 1 if after_accept else 0
            assert service.session.current_evaluation_frame.frame_index == expected
            rows = list(llm.read_calls(directory))
            assert len(rows) == 5
            assert {row["outcome"] for row in rows} == {
                "played" if after_accept else "abandoned"
            }
            status = service.recording_status
            assert status is not None and status.lifecycle == "saved"
            assert status.captured_transition_count == expected
        finally:
            service.close()
