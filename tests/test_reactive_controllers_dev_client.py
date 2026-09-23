"""Check DevClient controller selection, execution and recorded identities."""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import pytest
import scripts.dev.visual_debugger.control as control
import scripts.dev.visual_debugger.evaluation_bridge as evaluation_bridge
from pydantic import ValidationError
from scripts.dev.visual_debugger.authoring_models import (
    new_map_draft,
    new_scenario_draft,
)
from scripts.dev.visual_debugger.authoring_service import (
    DevAuthoringCommandRequestV1,
    DevClientAuthoringBinding,
    DevScenarioLoadService,
    LoadedDevScenarioSnapshotV1,
    debugger_scenario_from_snapshot,
)
from scripts.dev.visual_debugger.authoring_store import DevAssetStore
from scripts.dev.visual_debugger.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
)
from scripts.dev.visual_debugger.model import (
    DebuggerScenario,
    DebuggerSession,
    TeamBController,
    TeamController,
)
from scripts.dev.visual_debugger.protocol import CombatConfigurationV1
from scripts.dev.visual_debugger.recording import (
    build_debugger_recording_specification_v1,
)
from scripts.dev.visual_debugger.replay_recorder import DebuggerReplayRecorder
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.service import DebuggerService
from tests.scenario_controller_fixtures import load_scenario_1
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.types import TEAM_B_ID, EnvConfig, EnvState
from marl_battlegrounds.evaluation import policy_execution
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV2,
    EvaluationEpisodeContextV3,
    EvaluationFrameV2,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.policy_execution import Policy, policy
from marl_battlegrounds.evaluation.replay import (
    RuntimeProvenanceV1,
)
from marl_battlegrounds.evaluation.replay_io import (
    load_replay,
    preflight_replay_destination,
    save_replay,
)
from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3, build_replay_v3
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput, Observations
from marl_battlegrounds.policies.reactive_tdm_alpha import reactive_tdm_alpha_policy
from marl_battlegrounds.policies.shared_obs import (
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)


def _scenario(
    *,
    remaining: int = 5,
    countdown: int = 4,
    warrior_hunter_alive: bool = False,
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
    if warrior_hunter_alive:
        state = state._replace(
            alive_mask=state.alive_mask.at[6:8].set(True),
            current_health=state.current_health.at[6:8].set(
                config.agent_profile.max_health[6:8]
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
    team_b: TeamBController = "scenario_5",
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
        team_b_controller=team_b,
        execution_information_mode="shared_obs",
    )


def _tree_equal(left: object, right: object) -> bool:
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


@pytest.fixture
def scenario_controller() -> TeamBController:
    return "scenario_5"


@pytest.mark.parametrize("team_a", ("manual", "reactive_tdm", "random_valid"))
def test_scenario_controller_uses_one_epoch_bank_assembler_and_step(
    monkeypatch: pytest.MonkeyPatch,
    team_a: TeamController,
    scenario_controller: TeamBController,
) -> None:
    session = _session(team_a=team_a, team_b=scenario_controller)
    calls = {"bank": 0, "assembler": 0, "step": 0}
    real_bank = policy_execution.build_team_actor_input
    real_assembler = policy_execution.build_joint_action_from_actor_actions
    real_step = control.step
    real_executor = control.apply_policies
    input_epochs: list[tuple[object, object, object]] = []

    def bank(observations: Observations, team: int) -> ActorInput:
        calls["bank"] += 1
        assert observations.observation is session.observation
        return real_bank(observations, team)

    def assembler(*args: object) -> object:
        calls["assembler"] += 1
        return real_assembler(*args)  # type: ignore[arg-type]

    def step(*args: object) -> object:
        calls["step"] += 1
        return real_step(*args)  # type: ignore[arg-type]

    def executor(*args: object, **kwargs: object) -> object:
        input_epochs.append((args[6], args[7], args[8]))
        assert args[1] is policy("tdm-beta").apply
        if team_a != "manual":
            expected = "tdm-alpha" if team_a == "reactive_tdm" else "random"
            assert args[0] is policy(expected).apply
        return real_executor(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(policy_execution, "build_team_actor_input", bank)
    monkeypatch.setattr(
        policy_execution, "build_joint_action_from_actor_actions", assembler
    )
    monkeypatch.setattr(control, "step", step)
    monkeypatch.setattr(control, "apply_policies", executor)
    advanced = control.submit_interactive(session)
    assert calls == {"bank": 2, "assembler": 1, "step": 1}
    assert len(input_epochs) == 1
    observations, mask, keys = input_epochs[0]
    assert isinstance(observations, Observations)
    assert observations.observation is session.observation
    assert mask is session.action_mask
    assert isinstance(keys, jax.Array) and keys.shape == (10,)
    assert int(advanced.state.step_count) == int(session.state.step_count) + 1
    assert advanced.incoming_evaluation_view is not None
    acceptance = (
        advanced.incoming_evaluation_view.transition.facts.action_acceptance_facts
    )
    assert acceptance.submitted_joint_action == acceptance.accepted_joint_action
    context = advanced.evaluation_context
    aggregation = {row.name: row.value for row in context.aggregation_keys}
    assert aggregation["action_source"] == ("mixed" if team_a == "manual" else "policy")
    algorithm = f"{scenario_controller.replace('_', '-')}-pressure-controller"
    assert aggregation["pressure_protocol"] == f"{algorithm}@5"
    if team_a == "reactive_tdm":
        assert (
            aggregation["reactive_tdm_controller"]
            == "reactive-team-deathmatch-controller@3"
        )
        assert (
            aggregation["reactive_tdm_controller_digest"]
            != aggregation["pressure_protocol_digest"]
        )
        for row in context.policy_assignments[:5]:
            assert isinstance(row, AssignedPolicySlotV2)
            assert row.algorithm_id == "reactive-team-deathmatch-controller"
            assert row.execution_mode == "deterministic"
            assert (
                row.policy_content_digest
                == aggregation["reactive_tdm_controller_digest"]
            )
    for row in context.policy_assignments[5:]:
        assert isinstance(row, AssignedPolicySlotV2)
        assert row.policy_kind == scenario_controller
        assert row.algorithm_id == algorithm
        assert row.execution_mode == "deterministic"
        assert row.checkpoint_digest is None
        assert row.training_run_id == "not_applicable"
        assert row.policy_content_digest == aggregation["pressure_protocol_digest"]


@pytest.mark.parametrize("remaining", (1, 5, 6, 300))
def test_scenario_controller_accepts_any_valid_horizon(
    remaining: int,
    scenario_controller: TeamBController,
) -> None:
    session = _session(
        scenario=_scenario(remaining=remaining), team_b=scenario_controller
    )
    assert session.evaluation_context.expected_horizon == remaining
    assert session.scenario.name == "copied_and_renamed_scenario"


@pytest.mark.parametrize("warrior_hunter_alive", (False, True))
def test_scenario_controller_keeps_live_and_reviving_classes_reactive(
    warrior_hunter_alive: bool,
) -> None:
    session = _session(
        scenario=_scenario(countdown=0, warrior_hunter_alive=warrior_hunter_alive),
    )
    for _ in range(2):
        baseline = cast(
            ActorAction,
            execute_shared_obs_team_policy(
                session.observation,
                session.action_mask,
                control._policy_keys(session),  # pyright: ignore[reportPrivateUsage]
                build_shared_obs_sensor_source_bank(session.observation),
                control.build_default_shared_obs_information_availability(
                    session.config.agent_profile.active_mask,
                    session.config.agent_profile.team_ids,
                ),
                policy=reactive_tdm_alpha_policy,
                team_identity=TEAM_B_ID,
            ),
        )
        advanced = control.submit_interactive(session)
        incoming = advanced.incoming_evaluation_view
        assert incoming is not None
        acceptance = incoming.transition.facts.action_acceptance_facts
        for action in (
            acceptance.submitted_joint_action,
            acceptance.accepted_joint_action,
        ):
            for slot in (6, 7):
                assert (
                    action.move[slot],
                    action.select_target[slot],
                    action.use_ultimate[slot],
                ) == (
                    int(baseline.move[slot - 5]),
                    int(baseline.select_target[slot - 5]),
                    int(baseline.use_ultimate[slot - 5]),
                )
        assert bool(jnp.all(advanced.state.alive_mask[6:8]))
        session = advanced


def test_scenario_controller_accepts_either_respawn_clock_order() -> None:
    scenario = _scenario()
    config, state = scenario.build_scenario()
    state = state._replace(
        team_respawn_wave_countdowns=jnp.asarray([0, 4], dtype=jnp.int32),
    )
    asymmetric = replace(scenario, build_scenario=lambda: (config, state))
    assert _session(scenario=asymmetric).team_b_controller == "scenario_5"
    state = state._replace(
        team_respawn_wave_countdowns=jnp.asarray([4, 0], dtype=jnp.int32),
    )
    assert _session(scenario=asymmetric).team_b_controller == "scenario_5"


@pytest.mark.parametrize(
    ("remaining", "countdown", "alive"),
    (
        (300, 4, False),
        (5, 0, False),
        (5, 4, True),
    ),
)
def test_scenario_controller_survives_loading_long_or_reviving_rosters(
    remaining: int,
    countdown: int,
    alive: bool,
    scenario_controller: TeamBController,
) -> None:
    session = _session(team_b=scenario_controller)
    loaded = control.switch_scenario(
        session,
        _scenario(remaining=remaining, countdown=countdown, warrior_hunter_alive=alive),
    )
    assert loaded.run_generation == session.run_generation + 1
    assert loaded.seed == session.seed
    assert loaded.team_b_controller == scenario_controller
    assert loaded.team_a_controller == session.team_a_controller
    assert loaded.evaluation_context.expected_horizon == remaining
    assert loaded.evaluation_context.identity.task.identifier == "team_deathmatch"
    restarted = control.reset_session(control.submit_interactive(loaded))
    assert _tree_equal(restarted.state, loaded.state)
    assert _tree_equal(restarted.key, loaded.key)


def test_scenario_controller_protocol_rejects_team_a_and_nosharedobs(
    scenario_controller: TeamBController,
) -> None:
    configuration = {
        "team_a_controller": "manual",
        "team_b_controller": scenario_controller,
        "execution_information_mode": "shared_obs",
    }
    assert (
        CombatConfigurationV1.model_validate(configuration).team_b_controller
        == scenario_controller
    )
    with pytest.raises(ValidationError, match="team_a_controller"):
        CombatConfigurationV1.model_validate(
            {**configuration, "team_a_controller": scenario_controller}
        )
    with pytest.raises(ValidationError, match="require SharedObs"):
        CombatConfigurationV1.model_validate(
            {**configuration, "execution_information_mode": "no_shared_obs"}
        )
    session = _session(team_b=scenario_controller)
    with pytest.raises(ValueError, match="require SharedObs"):
        control.set_combat_configuration(
            session,
            team_a_controller="manual",
            team_b_controller=scenario_controller,
            execution_information_mode="no_shared_obs",
        )
    with pytest.raises(ValueError, match="team_a_controller"):
        control.set_combat_configuration(
            session,
            team_a_controller=cast(TeamController, scenario_controller),
            team_b_controller=scenario_controller,
            execution_information_mode="shared_obs",
        )
    assert session.run_generation == 0


def test_scenario_controller_cannot_replace_registered_scripted_frames(
    scenario_controller: TeamBController,
) -> None:
    with pytest.raises(ValueError, match="interactive"):
        _session(
            scenario=replace(_scenario(), mode="scripted"),
            team_b=scenario_controller,
        )


@pytest.mark.parametrize("asset_kind", ("map", "scenario"))
def test_controller_selected_in_default_arena_survives_current_and_saved_loads(
    tmp_path: Path,
    asset_kind: str,
    scenario_controller: TeamBController,
) -> None:
    initial = _session(scenario=get_scenario("arena_5v5"), team_b=scenario_controller)
    assert initial.evaluation_context.identity.task.identifier == (
        "visual-debugger-analysis-task"
    )
    service = DebuggerService(
        initial,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    def install(snapshot: LoadedDevScenarioSnapshotV1) -> None:
        service.load_scenario(debugger_scenario_from_snapshot(snapshot))

    store = DevAssetStore(tmp_path)
    binding = DevClientAuthoringBinding(
        store,
        scenario_loader=DevScenarioLoadService(store, install_snapshot=install),
    )
    draft = (
        new_map_draft("renamed_arena")
        if asset_kind == "map"
        else new_scenario_draft("renamed_scenario")
    )
    saved = store.save_draft(draft, expected_revision=0)
    for source in (
        {
            "source_kind": "current_buffer",
            "asset_kind": asset_kind,
            "draft": draft.model_dump(mode="json", by_alias=True),
        },
        {
            "source_kind": "saved_draft",
            "asset_kind": asset_kind,
            "asset_id": saved.asset_id,
            "revision": saved.revision,
        },
    ):
        response = binding.apply_command(
            DevAuthoringCommandRequestV1.model_validate_json(
                json.dumps({"command_type": "open_in_debug", "source": source})
            ),
        )
        assert response.ok
        loaded = service.session
        assert loaded.team_b_controller == scenario_controller
        assert loaded.team_a_controller == "manual"
        assert loaded.seed == initial.seed
        assert loaded.evaluation_context.expected_horizon == 300
        assert loaded.evaluation_context.identity.task.identifier == "team_deathmatch"
        if asset_kind == "map":
            assert loaded.scenario.title == "Default TDM map preview"
        assert not service.faulted


def test_scenario_controller_can_be_selected_after_terminal_and_reset(
    scenario_controller: TeamBController,
) -> None:
    scenario = _scenario(remaining=1)
    initial = _session(scenario=scenario, team_b=scenario_controller)
    manual = control.set_combat_configuration(
        initial,
        team_a_controller="manual",
        team_b_controller="manual",
        execution_information_mode="shared_obs",
    )
    terminal = control.submit_interactive(manual)
    assert terminal.reached_declared_horizon
    installed = control.set_combat_configuration(
        terminal,
        team_a_controller="manual",
        team_b_controller=scenario_controller,
        execution_information_mode="shared_obs",
    )
    assert installed.run_generation == terminal.run_generation + 1
    assert _tree_equal(installed.state, initial.state)
    assert _tree_equal(installed.key, initial.key)
    assert not installed.reached_declared_horizon
    assert (
        control.set_combat_configuration(
            installed,
            team_a_controller="manual",
            team_b_controller=scenario_controller,
            execution_information_mode="shared_obs",
        )
        is installed
    )


def test_scenario_controller_failure_is_atomic_and_policy_labelled(
    monkeypatch: pytest.MonkeyPatch,
    scenario_controller: TeamBController,
) -> None:
    session = _session(team_b=scenario_controller)

    def failed_policy(*args: object) -> object:
        del args
        raise RuntimeError("injected scenario controller failure")

    def injected_policy(name: str) -> Policy:
        original = policy(name)
        return (
            replace(original, apply=failed_policy) if name == "tdm-beta" else original
        )

    monkeypatch.setattr(control, "policy", injected_policy)
    with pytest.raises(control.DebuggerTransitionFailureV1) as raised:
        control.submit_interactive(session)
    assert raised.value.stable_code == "policy_action_build_failed"
    assert raised.value.stage == "action_build"
    assert session.current_evaluation_frame.frame_index == 0
    assert session.incoming_evaluation_view is None


def test_scenario_controller_reset_noop_and_readonly_actions(
    scenario_controller: TeamBController,
) -> None:
    session = _session(team_b=scenario_controller)
    assert (
        control.set_combat_configuration(
            session,
            team_a_controller="manual",
            team_b_controller=scenario_controller,
            execution_information_mode="shared_obs",
        )
        is session
    )
    inspected = control.select_controlled_actor(session, 5)
    assert control.set_pending_movement(inspected, 1) is inspected
    first = control.submit_interactive(session)
    restarted = control.reset_session(first)
    assert restarted.run_generation == 1
    assert restarted.team_b_controller == scenario_controller
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


def test_pressure_identity_is_independent_of_scenario_name_and_generation(
    scenario_controller: TeamBController,
) -> None:
    original = _session(team_b=scenario_controller)
    renamed = control.switch_scenario(
        original, replace(original.scenario, name="another_copy", title="Renamed")
    )
    original_row = cast(
        AssignedPolicySlotV2, original.evaluation_context.policy_assignments[5]
    )
    renamed_row = cast(
        AssignedPolicySlotV2, renamed.evaluation_context.policy_assignments[5]
    )
    assert original_row.policy_content_digest == renamed_row.policy_content_digest
    assert (
        original.evaluation_context.identity.scenario
        != renamed.evaluation_context.identity.scenario
    )


def test_pressure_identity_binds_descriptor_version_and_launch_revision(
    monkeypatch: pytest.MonkeyPatch,
    scenario_controller: TeamBController,
) -> None:
    session = _session(team_b=scenario_controller)
    original_row = cast(
        AssignedPolicySlotV2,
        session.evaluation_context.policy_assignments[5],
    )
    descriptor_name = "reactive_tdm_beta_controller_descriptor"
    descriptor = getattr(evaluation_bridge, descriptor_name)()
    original_version = 5
    assert descriptor["version"] == original_version
    original_descriptor = descriptor.copy()
    algorithm = f"{scenario_controller.replace('_', '-')}-pressure-controller"
    assert {row.name: row.value for row in session.evaluation_context.aggregation_keys}[
        "pressure_protocol"
    ] == f"{algorithm}@{original_version}"
    descriptor["version"] = original_version + 1
    monkeypatch.setattr(
        evaluation_bridge,
        descriptor_name,
        lambda: descriptor,
    )
    changed = control.reset_session(session)
    changed_row = cast(
        AssignedPolicySlotV2,
        changed.evaluation_context.policy_assignments[5],
    )
    assert changed_row.policy_content_digest != original_row.policy_content_digest
    assert {row.name: row.value for row in changed.evaluation_context.aggregation_keys}[
        "pressure_protocol"
    ] == f"{algorithm}@{original_version + 1}"
    descriptor["version"] = original_version
    descriptor["pursuit"] = "injected alternative pursuit rule"
    changed = control.reset_session(session)
    changed_row = cast(
        AssignedPolicySlotV2,
        changed.evaluation_context.policy_assignments[5],
    )
    assert changed_row.policy_content_digest != original_row.policy_content_digest
    assert {row.name: row.value for row in changed.evaluation_context.aggregation_keys}[
        "pressure_protocol"
    ] == f"{algorithm}@{original_version}"
    descriptor.clear()
    descriptor.update(original_descriptor)
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
        team_b_controller=scenario_controller,
        execution_information_mode="shared_obs",
    )
    changed_row = cast(
        AssignedPolicySlotV2,
        changed.evaluation_context.policy_assignments[5],
    )
    assert changed_row.policy_content_digest != original_row.policy_content_digest


@pytest.mark.parametrize(
    "team_b,historical",
    (("reactive_tdm", False), ("scenario_5", False), ("scenario_5", True)),
)
def test_reactive_controller_recording_reopens_without_replay_changes(
    tmp_path: Path,
    team_b: TeamBController,
    historical: bool,
) -> None:
    session = _session(recording=True, team_b=team_b)
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
    destination = preflight_replay_destination(
        tmp_path / "reactive-controller.marlbg-replay.json"
    )
    recorder = DebuggerReplayRecorder(
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
    assert isinstance(
        advanced.incoming_evaluation_view.successor_frame, EvaluationFrameV2
    )
    recorder.append(
        advanced.incoming_evaluation_view.transition,
        advanced.incoming_evaluation_view.successor_frame,
    )
    assert recorder.finalize_and_save("finish_and_review") == "saved"
    loaded = recorder.begin_review()
    assert loaded.replay.header.context == session.evaluation_context
    assert loaded.replay.header.runtime_provenance.policy_execution_included
    assert recorder.saved_bundle is not None
    reopened = load_replay(recorder.saved_bundle.replay_path)
    assert isinstance(reopened.replay, ReplayArtifactV3)
    assert reopened.replay.header.context == loaded.replay.header.context
    context = reopened.replay.header.context
    descriptor = (
        evaluation_bridge.reactive_tdm_beta_controller_descriptor()
        if team_b == "scenario_5"
        else evaluation_bridge.reactive_tdm_alpha_controller_descriptor()
    )
    expected_version = 5 if team_b == "scenario_5" else 3
    assert descriptor["version"] == expected_version
    expected_digest = canonical_digest_sha256(
        {"behavior": descriptor, "code_revision": context.code_revision}
    )
    aggregation = {row.name: row.value for row in context.aggregation_keys}
    identity_key = (
        "reactive_tdm_controller" if team_b == "reactive_tdm" else "pressure_protocol"
    )
    assert aggregation[identity_key] == (
        f"{descriptor['policy_id']}@{expected_version}"
    )
    assert aggregation[f"{identity_key}_digest"] == expected_digest
    for row in context.policy_assignments[5:]:
        assert isinstance(row, AssignedPolicySlotV2)
        assert row.policy_kind == team_b
        assert row.algorithm_id == descriptor["policy_id"]
        assert row.policy_content_digest == expected_digest

    if historical:
        # A retired controller name remains valid provenance in a current record.
        # This does not claim that the retired controller generated these actions.
        legacy_payload = context.model_dump(mode="json")
        for row in legacy_payload["policy_assignments"]:
            if row.get("policy_kind") == "scenario_5":
                row.update(
                    policy_kind="scenario_3",
                    policy_id=row["policy_id"].replace("scenario_5", "scenario_3"),
                    algorithm_id="scenario-3-pressure-controller",
                    policy_content_digest="3" * 64,
                    parameter_sharing_group_id=row[
                        "parameter_sharing_group_id"
                    ].replace("scenario_5", "scenario_3"),
                )
        for row in legacy_payload["aggregation_keys"]:
            if row["name"] == "team_b_controller":
                row["value"] = "scenario_3"
            elif row["name"] == "pressure_protocol":
                row["value"] = "scenario-3-pressure-controller@1"
            elif row["name"] == "pressure_protocol_digest":
                row["value"] = "3" * 64
        legacy_context = EvaluationEpisodeContextV3.model_validate_json(
            json.dumps(legacy_payload)
        )
        replay = build_replay_v3(
            legacy_context,
            reopened.replay.frames,
            reopened.replay.transitions,
            runtime_provenance=runtime,
            completion_state="partial",
            end_or_failure_reason="user_finish_and_review",
        )
        saved = save_replay(replay, tmp_path / "retired-controller.marlbg-replay.json")
        legacy_reopened = load_replay(saved.replay_path)
        assert legacy_reopened.replay.header.context == legacy_context
        for row in legacy_reopened.replay.header.context.policy_assignments[5:]:
            assert isinstance(row, AssignedPolicySlotV2)
            assert row.policy_kind == "scenario_3"
            assert row.algorithm_id == "scenario-3-pressure-controller"
