"""Scenario 1 host eligibility, coherent execution, and diagnostic provenance."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import pytest
import scripts.dev.visual_debugger.control as control
import scripts.dev.visual_debugger.evaluation_bridge as evaluation_bridge
from pydantic import ValidationError
from scripts.dev.visual_debugger.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
)
from scripts.dev.visual_debugger.model import (
    DebuggerScenario,
    DebuggerSession,
    TeamController,
)
from scripts.dev.visual_debugger.protocol import CombatConfigurationV1
from scripts.dev.visual_debugger.recording import (
    DebuggerReplayRecorderV1,
    build_debugger_recording_specification_v1,
)
from tests.scenario_controller_fixtures import load_scenario_1
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.types import EnvConfig, EnvState
from marl_battlegrounds.evaluation.models import AssignedPolicySlotV1
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_io import (
    load_replay_bundle_v1,
    preflight_replay_bundle_destination_v1,
)


def _scenario(
    *,
    remaining: int = 5,
    countdown: int = 4,
    unsupported_alive: bool = False,
) -> DebuggerScenario:
    compiled = load_scenario_1()
    config = compiled.config._replace(
        max_steps=int(compiled.initial_state.step_count) + remaining,
    )
    state = compiled.initial_state._replace(
        team_respawn_wave_countdowns=compiled.initial_state.team_respawn_wave_countdowns.at[
            1
        ].set(countdown),
    )
    if unsupported_alive:
        state = state._replace(
            alive_mask=state.alive_mask.at[6].set(True),
            current_health=state.current_health.at[6].set(
                config.agent_profile.max_health[6]
            ),
        )

    def build() -> tuple[EnvConfig, EnvState]:
        return config, state

    return DebuggerScenario(
        name="copied_and_renamed_scenario",
        title="A copied scenario without a special asset ID",
        description="Test-only physical fixture",
        mode="interactive",
        build_scenario=build,
        frames=(),
        default_controlled_slot=2,
    )


def _session(
    *,
    scenario: DebuggerScenario | None = None,
    team_a: TeamController = "manual",
    recording: bool = False,
) -> DebuggerSession:
    launch = debugger_test_launch_specification(7)
    if recording:
        launch = build_debugger_evaluation_launch_specification_v1(
            root_seed=7,
            code_revision=launch.code_revision,
            capture_profile="evaluation_metric_complete",
        )
    return control.create_session(
        _scenario() if scenario is None else scenario,
        seed=7,
        evaluation_launch_specification=launch,
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
        team_a_controller=team_a,
        team_b_controller="scenario_1",
        execution_information_mode="shared_obs",
    )


def _tree_equal(left: object, right: object) -> bool:
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


@pytest.mark.parametrize("team_a", ("manual", "scripted_tdm", "random_valid"))
def test_scenario_controller_uses_one_epoch_bank_assembler_and_step(
    monkeypatch: pytest.MonkeyPatch,
    team_a: TeamController,
) -> None:
    session = _session(team_a=team_a)
    calls = {"bank": 0, "assembler": 0, "step": 0}
    real_bank = control.build_shared_obs_sensor_source_bank
    real_assembler = control.build_joint_action_from_actor_actions
    real_step = control.step
    real_executor = control.execute_shared_obs_team_policy
    input_epochs: list[tuple[object, object, object]] = []

    def bank(observation: object) -> object:
        calls["bank"] += 1
        assert observation is session.observation
        return real_bank(observation)  # type: ignore[arg-type]

    def assembler(*args: object) -> object:
        calls["assembler"] += 1
        return real_assembler(*args)  # type: ignore[arg-type]

    def step(*args: object) -> object:
        calls["step"] += 1
        return real_step(*args)  # type: ignore[arg-type]

    def executor(*args: object, **kwargs: object) -> object:
        input_epochs.append((args[0], args[1], args[3]))
        return real_executor(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(control, "build_shared_obs_sensor_source_bank", bank)
    monkeypatch.setattr(control, "build_joint_action_from_actor_actions", assembler)
    monkeypatch.setattr(control, "step", step)
    monkeypatch.setattr(control, "execute_shared_obs_team_policy", executor)
    advanced = control.submit_interactive(session)
    assert calls == {"bank": 1, "assembler": 1, "step": 1}
    assert len(input_epochs) == (1 if team_a == "manual" else 2)
    for observation, mask, source_bank in input_epochs:
        assert observation is session.observation
        assert mask is session.action_mask
        assert source_bank is input_epochs[0][2]
    assert int(advanced.state.step_count) == int(session.state.step_count) + 1
    assert advanced.incoming_evaluation_view is not None
    context = advanced.evaluation_context
    aggregation = {row.name: row.value for row in context.aggregation_keys}
    assert aggregation["action_source"] == ("mixed" if team_a == "manual" else "policy")
    assert aggregation["pressure_protocol"] == "scenario-1-pressure-controller@1"
    for row in context.policy_assignments[5:]:
        assert isinstance(row, AssignedPolicySlotV1)
        assert row.policy_kind == "scenario_1"
        assert row.algorithm_id == "scenario-1-pressure-controller"
        assert row.execution_mode == "deterministic"
        assert row.checkpoint_digest is None
        assert row.training_run_id == "not_applicable"
        assert row.policy_content_digest == aggregation["pressure_protocol_digest"]


@pytest.mark.parametrize("remaining", (1, 5))
def test_scenario_controller_accepts_supported_horizon_and_final_wave(
    remaining: int,
) -> None:
    session = _session(scenario=_scenario(remaining=remaining, countdown=remaining - 1))
    assert session.evaluation_context.expected_horizon == remaining
    assert session.scenario.name == "copied_and_renamed_scenario"


def test_scenario_controller_uses_team_b_clock_not_team_a_clock() -> None:
    scenario = _scenario()
    config, state = scenario.build_scenario()
    state = state._replace(
        team_respawn_wave_countdowns=jnp.asarray([0, 4], dtype=jnp.int32),
    )
    asymmetric = replace(scenario, build_scenario=lambda: (config, state))
    assert _session(scenario=asymmetric).team_b_controller == "scenario_1"
    state = state._replace(
        team_respawn_wave_countdowns=jnp.asarray([4, 0], dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="final transition"):
        _session(scenario=asymmetric)


@pytest.mark.parametrize(
    ("remaining", "countdown", "alive", "message"),
    (
        (6, 4, False, "one to five"),
        (5, 3, False, "start dead"),
        (5, 4, True, "start dead"),
    ),
)
def test_incompatible_load_is_atomic(
    remaining: int,
    countdown: int,
    alive: bool,
    message: str,
) -> None:
    session = _session()
    before = (
        session.state,
        session.pending_actions,
        session.key,
        session.evaluation_context,
    )
    with pytest.raises(ValueError, match=message):
        control.switch_scenario(
            session,
            _scenario(
                remaining=remaining, countdown=countdown, unsupported_alive=alive
            ),
        )
    assert session.run_generation == 0
    assert before == (
        session.state,
        session.pending_actions,
        session.key,
        session.evaluation_context,
    )


def test_scenario_controller_protocol_rejects_team_a_and_nosharedobs() -> None:
    configuration = {
        "team_a_controller": "manual",
        "team_b_controller": "scenario_1",
        "execution_information_mode": "shared_obs",
    }
    assert (
        CombatConfigurationV1.model_validate(configuration).team_b_controller
        == "scenario_1"
    )
    with pytest.raises(ValidationError, match="team_a_controller"):
        CombatConfigurationV1.model_validate(
            {**configuration, "team_a_controller": "scenario_1"}
        )
    with pytest.raises(ValidationError, match="requires SharedObs"):
        CombatConfigurationV1.model_validate(
            {**configuration, "execution_information_mode": "no_shared_obs"}
        )
    session = _session()
    with pytest.raises(ValueError, match="requires SharedObs"):
        control.set_combat_configuration(
            session,
            team_a_controller="manual",
            team_b_controller="scenario_1",
            execution_information_mode="no_shared_obs",
        )
    with pytest.raises(ValueError, match="team_a_controller"):
        control.set_combat_configuration(
            session,
            team_a_controller=cast(TeamController, "scenario_1"),
            team_b_controller="scenario_1",
            execution_information_mode="shared_obs",
        )
    assert session.run_generation == 0


def test_scenario_controller_cannot_replace_registered_scripted_frames() -> None:
    with pytest.raises(ValueError, match="interactive TDM"):
        _session(scenario=replace(_scenario(), mode="scripted"))


def test_scenario_controller_failure_is_atomic_and_policy_labelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()

    def failed_policy(*args: object) -> object:
        del args
        raise RuntimeError("injected scenario controller failure")

    monkeypatch.setattr(control, "scenario_1_policy", failed_policy)
    with pytest.raises(control.DebuggerTransitionFailureV1) as raised:
        control.submit_interactive(session)
    assert raised.value.stable_code == "policy_action_build_failed"
    assert raised.value.stage == "action_build"
    assert session.current_evaluation_frame.frame_index == 0
    assert session.incoming_evaluation_view is None


def test_scenario_controller_reset_noop_and_readonly_actions() -> None:
    session = _session()
    assert (
        control.set_combat_configuration(
            session,
            team_a_controller="manual",
            team_b_controller="scenario_1",
            execution_information_mode="shared_obs",
        )
        is session
    )
    inspected = control.select_controlled_actor(session, 5)
    assert control.set_pending_movement(inspected, 1) is inspected
    first = control.submit_interactive(session)
    restarted = control.reset_session(first)
    assert restarted.run_generation == 1
    assert restarted.team_b_controller == "scenario_1"
    assert restarted.team_a_controller == "manual"
    assert _tree_equal(restarted.state, session.state)
    assert _tree_equal(restarted.key, session.key)
    repeated = control.submit_interactive(restarted)
    assert _tree_equal(repeated.state, first.state)
    assert first.incoming_evaluation_view is not None
    assert repeated.incoming_evaluation_view is not None
    assert (
        first.incoming_evaluation_view.transition.facts
        == repeated.incoming_evaluation_view.transition.facts
    )


def test_pressure_identity_is_independent_of_scenario_name_and_generation() -> None:
    original = _session()
    renamed = control.switch_scenario(
        original, replace(original.scenario, name="another_copy", title="Renamed")
    )
    original_row = cast(
        AssignedPolicySlotV1, original.evaluation_context.policy_assignments[5]
    )
    renamed_row = cast(
        AssignedPolicySlotV1, renamed.evaluation_context.policy_assignments[5]
    )
    assert original_row.policy_content_digest == renamed_row.policy_content_digest
    assert (
        original.evaluation_context.identity.scenario
        != renamed.evaluation_context.identity.scenario
    )


def test_pressure_identity_binds_descriptor_version_and_launch_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    original_row = cast(
        AssignedPolicySlotV1,
        session.evaluation_context.policy_assignments[5],
    )
    descriptor = evaluation_bridge.scenario_1_controller_descriptor()
    descriptor["version"] = 2
    monkeypatch.setattr(
        evaluation_bridge,
        "scenario_1_controller_descriptor",
        lambda: descriptor,
    )
    changed = control.reset_session(session)
    changed_row = cast(
        AssignedPolicySlotV1,
        changed.evaluation_context.policy_assignments[5],
    )
    assert changed_row.policy_content_digest != original_row.policy_content_digest
    assert {row.name: row.value for row in changed.evaluation_context.aggregation_keys}[
        "pressure_protocol"
    ] == "scenario-1-pressure-controller@2"
    descriptor["version"] = 1
    launch = debugger_test_launch_specification(7)
    changed_code = launch.code_revision.model_copy(update={"commit_sha": "b" * 40})
    changed_launch = build_debugger_evaluation_launch_specification_v1(
        root_seed=7,
        code_revision=changed_code,
    )
    changed = control.create_session(
        session.scenario,
        seed=7,
        evaluation_launch_specification=changed_launch,
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
        team_a_controller="manual",
        team_b_controller="scenario_1",
        execution_information_mode="shared_obs",
    )
    changed_row = cast(
        AssignedPolicySlotV1,
        changed.evaluation_context.policy_assignments[5],
    )
    assert changed_row.policy_content_digest != original_row.policy_content_digest


def test_scenario_controller_recording_reopens_without_replay_changes(
    tmp_path: Path,
) -> None:
    session = _session(recording=True)
    runtime = RuntimeProvenanceV1(
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
        policy_execution_included=True,
    )
    destination = preflight_replay_bundle_destination_v1(
        tmp_path / "scenario-1.marlbg-replay.json"
    )
    recorder = DebuggerReplayRecorderV1(
        specification=build_debugger_recording_specification_v1(
            action_source_kind="mixed",
            runtime_provenance=runtime,
        ),
        destination=destination,
        context=session.evaluation_context,
        initial_frame=session.current_evaluation_frame,
    )
    advanced = control.submit_interactive(session)
    assert advanced.incoming_evaluation_view is not None
    recorder.append(
        advanced.incoming_evaluation_view.transition,
        advanced.incoming_evaluation_view.successor_frame,
    )
    assert recorder.finalize_and_save("finish_and_review") == "saved"
    loaded = recorder.begin_review()
    assert loaded.replay.header.context == session.evaluation_context
    assert loaded.replay.header.runtime_provenance.policy_execution_included
    assert recorder.saved_bundle is not None
    reopened = load_replay_bundle_v1(recorder.saved_bundle.replay_path)
    assert reopened.replay.header.context == loaded.replay.header.context
