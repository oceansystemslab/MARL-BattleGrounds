"""Transactional CP2/CP3 integration proofs for the live debugger service."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import pytest
import scripts.dev.visual_debugger.control as control_module
import scripts.dev.visual_debugger.service as service_module
from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.evaluation_bridge import (
    DebuggerEvaluationLaunchSpecificationV1,
    build_debugger_evaluation_launch_specification_v1,
)
from scripts.dev.visual_debugger.input import InputDispatchResult, dispatch_command
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.protocol import (
    CommandRequestV1,
    KeyboardCommandV1,
    ResetCommandV1,
    SetPresetCommandV1,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.service import DebuggerService

import marl_battlegrounds.evaluation.capture as capture_module
from marl_battlegrounds.evaluation.metrics import EvaluationEpisodeObserverV1
from marl_battlegrounds.evaluation.models import (
    CodeRevisionV1,
    EvaluationEpisodeContextV1,
)


def _launch(root_seed: int = 19) -> DebuggerEvaluationLaunchSpecificationV1:
    return build_debugger_evaluation_launch_specification_v1(
        root_seed=root_seed,
        code_revision=CodeRevisionV1(
            package_version="0.0.0",
            commit_sha="a" * 40,
            source_tree_digest="b" * 64,
            is_dirty=False,
            dirty_patch_digest=None,
        ),
    )


def _session(
    scenario_name: str = "arena_5v5",
    *,
    seed: int = 19,
) -> DebuggerSession:
    return create_session(
        get_scenario(scenario_name),
        seed=seed,
        evaluation_launch_specification=_launch(seed),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )


def _service(scenario_name: str = "arena_5v5") -> DebuggerService:
    return DebuggerService(
        _session(scenario_name),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        session_id="live-evaluation-test",
    )


def _request(
    command_id: str,
    *,
    revision: int,
    command: object,
) -> CommandRequestV1:
    return CommandRequestV1(
        client_id="live-evaluation-client",
        command_id=command_id,
        base_revision=revision,
        command=command,  # pyright: ignore[reportArgumentType]
    )


def _scientific_epoch(session: DebuggerSession) -> tuple[object, ...]:
    return (
        session.evaluation_context,
        session.current_evaluation_frame,
        session.incoming_evaluation_view,
        session.status_source_evidence_state,
    )


def _raw_continuation(session: DebuggerSession) -> tuple[object, ...]:
    return (
        session.config,
        session.key,
        session.state,
        session.observation,
        session.action_mask,
        session.raw_continuation_identity,
    )


def _assert_observer_evidence(
    service: DebuggerService,
    *,
    expected_count: int,
    expected_lifecycle: str = "open",
) -> None:
    assert service.evaluation_validated_transition_count == expected_count
    assert service.evaluation_observer_lifecycle_state == expected_lifecycle
    assert service.session.current_evaluation_frame.frame_index == expected_count


def test_create_session_joins_launch_context_frame_zero_and_environment_seed() -> None:
    launch = _launch(73)
    session = create_session(
        get_scenario("arena_5v5"),
        seed=73,
        evaluation_launch_specification=launch,
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )

    context = session.evaluation_context
    frame = session.current_evaluation_frame
    assert context.identity.run_id == (
        f"debugger-run:{launch.launch_content_digest_sha256}"
    )
    assert context.code_revision == launch.code_revision
    assert context.seed_protocol.root_seed == launch.root_seed
    assert context.capture_profile == "debug"
    assert context.execution_information_mode == "no_shared_obs"
    assert frame.episode_id == context.identity.episode_id
    assert frame.frame_index == 0
    assert frame.simulator_step_count == int(session.state.step_count)
    assert session.incoming_evaluation_view is None
    assert bool(
        jnp.array_equal(
            session.key,
            jax.random.key(context.seed_protocol.environment_seed),
        )
    )


def test_create_session_rejects_a_seed_that_disagrees_with_launch_provenance() -> None:
    with pytest.raises(ValueError, match=r"seed.*launch|launch.*seed"):
        create_session(
            get_scenario("arena_5v5"),
            seed=74,
            evaluation_launch_specification=_launch(73),
            controlled_global_slot=None,
            show_ranges=True,
            verbose_logging=False,
        )


def test_one_core_step_has_one_cp2_capture_and_ui_only_input_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    real_step = control_module.step
    real_capture = control_module.capture_evaluation_transition_unit_v1
    calls = {"step": 0, "capture": 0}

    def counting_step(*args: object, **kwargs: object) -> object:
        calls["step"] += 1
        return real_step(*args, **kwargs)  # type: ignore[arg-type]

    def counting_capture(*args: object, **kwargs: object) -> object:
        calls["capture"] += 1
        return real_capture(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(control_module, "step", counting_step)
    monkeypatch.setattr(
        control_module,
        "capture_evaluation_transition_unit_v1",
        counting_capture,
    )

    ui_only = dispatch_command(
        session,
        KeyboardCommandV1(key="g"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    assert ui_only.changed
    assert ui_only.transition_applied is None
    assert not ui_only.episode_restarted
    assert calls == {"step": 0, "capture": 0}
    assert _scientific_epoch(ui_only.session) == _scientific_epoch(session)
    assert all(
        candidate is previous
        for candidate, previous in zip(
            _raw_continuation(ui_only.session),
            _raw_continuation(session),
            strict=True,
        )
    )

    submitted = dispatch_command(
        ui_only.session,
        KeyboardCommandV1(key="Enter"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    assert submitted.changed
    assert submitted.transition_applied is submitted.session.incoming_evaluation_view
    assert not submitted.episode_restarted
    assert calls == {"step": 1, "capture": 1}
    assert submitted.session.current_evaluation_frame.frame_index == 1


def test_one_core_step_has_one_bundled_host_transfer_and_ui_only_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    real_device_get = capture_module.jax.device_get
    transferred_bundles: list[object] = []

    def counting_device_get(bundle: object) -> object:
        transferred_bundles.append(bundle)
        return real_device_get(bundle)

    monkeypatch.setattr(capture_module.jax, "device_get", counting_device_get)
    ui_only = dispatch_command(
        session,
        KeyboardCommandV1(key="g"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    assert ui_only.changed
    assert transferred_bundles == []

    submitted = dispatch_command(
        ui_only.session,
        KeyboardCommandV1(key="Enter"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    assert submitted.transition_applied is not None
    assert len(transferred_bundles) == 1
    assert isinstance(transferred_bundles[0], tuple)
    assert len(cast(tuple[object, ...], transferred_bundles[0])) == 8


def test_interactive_submit_passes_context_to_action_builder_and_config_to_core_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    real_action_builder = control_module.build_interactive_joint_action
    real_step = control_module.step
    observed = {"context": False, "config": False}

    def checking_action_builder(
        context: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        assert context is session.evaluation_context
        assert type(context) is EvaluationEpisodeContextV1
        observed["context"] = True
        return real_action_builder(context, *args, **kwargs)  # type: ignore[arg-type]

    def checking_step(
        config: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        assert config is session.config
        observed["config"] = True
        return real_step(config, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        control_module,
        "build_interactive_joint_action",
        checking_action_builder,
    )
    monkeypatch.setattr(control_module, "step", checking_step)

    submitted = control_module.submit_interactive(session)

    assert observed == {"context": True, "config": True}
    assert submitted.current_evaluation_frame.frame_index == 1


def test_restart_builds_pending_rows_from_the_new_evaluation_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    real_builder = control_module._default_pending_actions  # pyright: ignore[reportPrivateUsage]
    observed_contexts: list[EvaluationEpisodeContextV1] = []

    def checking_builder(
        context: EvaluationEpisodeContextV1,
        *args: object,
        **kwargs: object,
    ) -> object:
        assert type(context) is EvaluationEpisodeContextV1
        observed_contexts.append(context)
        return real_builder(context, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(control_module, "_default_pending_actions", checking_builder)

    restarted = control_module.reset_session(session)

    assert observed_contexts == [restarted.evaluation_context]
    assert restarted.evaluation_context is not session.evaluation_context
    assert restarted.current_evaluation_frame.frame_index == 0


def test_declared_script_horizon_blocks_an_extra_step_without_core_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session("basic_support")
    first = dispatch_command(
        session,
        KeyboardCommandV1(key="n"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    second = dispatch_command(
        first.session,
        KeyboardCommandV1(key="n"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert second.session.current_evaluation_frame.frame_index == 2
    assert second.session.reached_declared_horizon
    assert not second.session.terminated
    assert not second.session.truncated

    def forbidden_step(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("declared-horizon input must not call the core step")

    monkeypatch.setattr(control_module, "step", forbidden_step)
    blocked = dispatch_command(
        second.session,
        KeyboardCommandV1(key="n"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert not blocked.changed
    assert blocked.transition_applied is None
    assert blocked.session is second.session
    assert blocked.notice is not None
    assert "declared horizon" in blocked.notice


def test_input_results_distinguish_transition_restart_and_ui_only_changes() -> None:
    initial = _session()

    ui_only = dispatch_command(
        initial,
        SetPresetCommandV1(preset="debug"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    assert ui_only.changed
    assert ui_only.transition_applied is None
    assert not ui_only.episode_restarted

    transitioned = dispatch_command(
        initial,
        KeyboardCommandV1(key="Enter"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    assert transitioned.changed
    assert (
        transitioned.transition_applied is transitioned.session.incoming_evaluation_view
    )
    assert not transitioned.episode_restarted

    restarted = dispatch_command(
        transitioned.session,
        ResetCommandV1(),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    assert restarted.changed
    assert restarted.transition_applied is None
    assert restarted.episode_restarted
    assert restarted.session.current_evaluation_frame.frame_index == 0
    assert restarted.session.incoming_evaluation_view is None


def test_ui_only_service_command_cannot_mutate_scientific_epoch_or_observer() -> None:
    service = _service()
    initial_session = service.session
    initial_epoch = _scientific_epoch(initial_session)
    initial_continuation = _raw_continuation(initial_session)

    result = service.apply_command(
        _request(
            "toggle-ranges",
            revision=0,
            command=KeyboardCommandV1(key="g"),
        )
    )

    assert result.outcome == "response"
    assert service.revision == 1
    assert service.session is not initial_session
    assert _scientific_epoch(service.session) == initial_epoch
    assert all(
        candidate is previous
        for candidate, previous in zip(
            _raw_continuation(service.session),
            initial_continuation,
            strict=True,
        )
    )
    _assert_observer_evidence(service, expected_count=0)


def test_session_rejects_unmarked_raw_replacement_before_dispatch() -> None:
    service = _service()
    initial_session = service.session
    initial_frame = service.current_frame()

    with pytest.raises(ValueError, match="raw continuation identity must bind"):
        replace(initial_session, key=jax.random.key(987654321))

    assert not service.faulted
    assert service.session is initial_session
    assert service.current_frame() is initial_frame
    assert service.revision == 0
    _assert_observer_evidence(service, expected_count=0)


def test_marked_transition_rejects_rebound_raw_continuation_before_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    initial_session = service.session
    initial_frame = service.current_frame()
    real_result = dispatch_command(
        initial_session,
        KeyboardCommandV1(key="Enter"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    assert real_result.transition_applied is not None
    candidate = replace(
        real_result.session,
        key=jax.random.key(987654321),
        raw_continuation_identity=None,
    )

    with pytest.raises(ValueError, match="bind the candidate raw continuation"):
        replace(real_result, session=candidate)

    malicious_result = copy(real_result)
    object.__setattr__(malicious_result, "session", candidate)
    assert malicious_result.raw_continuation_identity is (
        real_result.session.raw_continuation_identity
    )
    assert malicious_result.raw_continuation_identity is not (
        candidate.raw_continuation_identity
    )

    def malicious_dispatch(
        *_args: object,
        **_kwargs: object,
    ) -> InputDispatchResult:
        return malicious_result

    monkeypatch.setattr(
        service_module,
        "dispatch_command",
        malicious_dispatch,
    )
    with pytest.raises(RuntimeError, match="transition marker"):
        service.apply_command(
            _request(
                "rebound-marked-transition",
                revision=0,
                command=KeyboardCommandV1(key="Enter"),
            )
        )

    assert service.faulted
    assert service.session is initial_session
    assert service.current_frame() is initial_frame
    assert service.revision == 0
    _assert_observer_evidence(service, expected_count=0)


def test_unmarked_script_cursor_replacement_faults_before_any_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service("basic_support")
    initial_session = service.session
    initial_frame = service.current_frame()
    candidate = replace(initial_session, next_script_frame_index=1)

    def malicious_dispatch(*_args: object, **_kwargs: object) -> InputDispatchResult:
        return InputDispatchResult(
            session=candidate,
            view_mode="researcher",
            preset="analysis",
            handled=True,
            changed=True,
        )

    monkeypatch.setattr(service_module, "dispatch_command", malicious_dispatch)
    with pytest.raises(RuntimeError, match="scientific session state changed"):
        service.apply_command(
            _request(
                "unmarked-script-cursor",
                revision=0,
                command=KeyboardCommandV1(key="g"),
            )
        )

    assert service.faulted
    assert service.session is initial_session
    assert service.current_frame() is initial_frame
    assert service.revision == 0
    _assert_observer_evidence(service, expected_count=0)


def test_unmarked_submission_metadata_replacement_faults_before_any_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    advanced = service.apply_command(
        _request(
            "real-submit",
            revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )
    assert advanced.outcome == "response"
    initial_session = service.session
    initial_frame = service.current_frame()
    assert initial_session.last_submission_kind == "interactive"
    candidate = replace(initial_session, last_submission_kind="scripted")

    def malicious_dispatch(*_args: object, **_kwargs: object) -> InputDispatchResult:
        return InputDispatchResult(
            session=candidate,
            view_mode="researcher",
            preset="analysis",
            handled=True,
            changed=True,
        )

    monkeypatch.setattr(service_module, "dispatch_command", malicious_dispatch)
    with pytest.raises(RuntimeError, match="scientific session state changed"):
        service.apply_command(
            _request(
                "unmarked-submission-kind",
                revision=1,
                command=KeyboardCommandV1(key="g"),
            )
        )

    assert service.faulted
    assert service.session is initial_session
    assert service.current_frame() is initial_frame
    assert service.revision == 1
    _assert_observer_evidence(service, expected_count=1)


def test_service_builds_candidate_frame_before_appending_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    real_build = service_module.build_debugger_frame
    real_append = EvaluationEpisodeObserverV1.append
    candidate_frame_built = False
    append_observations: list[tuple[bool, int]] = []

    def observing_build(*args: object, **kwargs: object) -> object:
        nonlocal candidate_frame_built
        result = real_build(*args, **kwargs)  # type: ignore[arg-type]
        candidate_frame_built = True
        return result

    def observing_append(
        observer: EvaluationEpisodeObserverV1,
        *args: object,
        **kwargs: object,
    ) -> None:
        append_observations.append(
            (candidate_frame_built, service.evaluation_validated_transition_count)
        )
        real_append(observer, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service_module, "build_debugger_frame", observing_build)
    monkeypatch.setattr(EvaluationEpisodeObserverV1, "append", observing_append)
    result = service.apply_command(
        _request(
            "submit",
            revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )

    assert result.outcome == "response"
    assert append_observations == [(True, 0)]
    _assert_observer_evidence(service, expected_count=1)


def test_restart_starts_a_separate_observer_before_swapping_service_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    advanced = service.apply_command(
        _request(
            "pre-reset-submit",
            revision=0,
            command=KeyboardCommandV1(key="Enter"),
        )
    )
    assert advanced.outcome == "response"
    _assert_observer_evidence(service, expected_count=1)
    initial_session = service.session
    real_factory = service._new_evaluation_observer  # pyright: ignore[reportPrivateUsage]
    candidate_frame_built = False
    created_evidence: list[tuple[str, int, str]] = []
    real_build = service_module.build_debugger_frame

    def observing_build(*args: object, **kwargs: object) -> object:
        nonlocal candidate_frame_built
        result = real_build(*args, **kwargs)  # type: ignore[arg-type]
        candidate_frame_built = True
        return result

    def observing_factory(session: DebuggerSession) -> EvaluationEpisodeObserverV1:
        assert candidate_frame_built
        assert service.session is initial_session
        observer = real_factory(session)
        created_evidence.append(
            (
                observer.lifecycle_state,
                observer.validated_transition_count,
                observer.context.identity.episode_id,
            )
        )
        return observer

    monkeypatch.setattr(service_module, "build_debugger_frame", observing_build)
    monkeypatch.setattr(service, "_new_evaluation_observer", observing_factory)
    result = service.apply_command(
        _request(
            "reset",
            revision=1,
            command=ResetCommandV1(),
        )
    )

    assert result.outcome == "response"
    assert created_evidence == [
        (
            "open",
            0,
            service.session.evaluation_context.identity.episode_id,
        )
    ]
    assert service.session is not initial_session
    assert service.session.run_generation == initial_session.run_generation + 1
    _assert_observer_evidence(service, expected_count=0)


def test_service_observer_seals_at_the_exact_declared_horizon() -> None:
    service = _service("basic_support")

    first = service.apply_command(
        _request(
            "script-0",
            revision=0,
            command=KeyboardCommandV1(key="n"),
        )
    )
    second = service.apply_command(
        _request(
            "script-1",
            revision=1,
            command=KeyboardCommandV1(key="n"),
        )
    )

    assert first.outcome == "response"
    assert second.outcome == "response"
    assert service.session.reached_declared_horizon
    assert not service.session.terminated
    assert not service.session.truncated
    _assert_observer_evidence(
        service,
        expected_count=2,
        expected_lifecycle="sealed",
    )


def test_candidate_frame_failure_leaves_observer_session_frame_and_revision_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    initial_session = service.session
    initial_frame = service.current_frame()

    def fail_frame(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic candidate-frame failure")

    monkeypatch.setattr(service_module, "build_debugger_frame", fail_frame)
    with pytest.raises(RuntimeError, match="candidate-frame failure"):
        service.apply_command(
            _request(
                "failed-frame",
                revision=0,
                command=KeyboardCommandV1(key="Enter"),
            )
        )

    assert service.faulted
    assert service.session is initial_session
    assert service.current_frame() is initial_frame
    assert service.revision == 0
    _assert_observer_evidence(service, expected_count=0)


def test_command_record_failure_precedes_observer_append_and_keeps_epoch_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    initial_session = service.session
    initial_frame = service.current_frame()
    real_record = service_module._CommandRecord  # pyright: ignore[reportPrivateUsage]
    construction_count = 0

    def fail_first_record(**kwargs: object) -> object:
        nonlocal construction_count
        construction_count += 1
        if construction_count == 1:
            raise RuntimeError("synthetic command-record failure")
        return real_record(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service_module, "_CommandRecord", fail_first_record)
    with pytest.raises(RuntimeError, match="command-record failure"):
        service.apply_command(
            _request(
                "failed-record",
                revision=0,
                command=KeyboardCommandV1(key="Enter"),
            )
        )

    assert construction_count == 2
    assert service.faulted
    assert service.session is initial_session
    assert service.current_frame() is initial_frame
    assert service.revision == 0
    _assert_observer_evidence(service, expected_count=0)


def test_response_failure_leaves_observer_session_frame_and_revision_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    initial_session = service.session
    initial_frame = service.current_frame()

    def fail_response(**_kwargs: object) -> object:
        raise RuntimeError("synthetic candidate-response failure")

    monkeypatch.setattr(service_module, "CommandResponseV2", fail_response)
    with pytest.raises(RuntimeError, match="candidate-response failure"):
        service.apply_command(
            _request(
                "failed-response",
                revision=0,
                command=KeyboardCommandV1(key="Enter"),
            )
        )

    assert service.faulted
    assert service.session is initial_session
    assert service.current_frame() is initial_frame
    assert service.revision == 0
    _assert_observer_evidence(service, expected_count=0)
