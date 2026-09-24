"""Check TDM-GAMMA in the DevClient: selection, execution, records and replays.

GAMMA's private live controller ID is ``tdm_gamma``, its public policy name is
``tdm-gamma`` and its label is "Reactive TDM GAMMA". Like ALPHA
(``reactive_tdm``) and BETA (``scenario_5``) it can play either team, and it
needs SharedObs on either team. These tests check that:

- Team B GAMMA is admitted with SharedObs by the protocol models,
  ``create_session`` and ``set_combat_configuration``.
- Team A GAMMA is admitted with SharedObs by the protocol, the control layer,
  ``DebuggerSession`` and ``build_debugger_evaluation_context_v1``.
- GAMMA on either team without SharedObs is rejected by those same layers,
  and the old session stays unchanged.
- ``_configured_policy`` gives either team the registered
  ``policy("tdm-gamma")`` apply, whose ``controller_identity`` is GAMMA
  version 2; the configured joint action passes it on that team's own side.
- A real step submits legal, accepted Team B actions; reset, reselection and a
  scenario switch keep GAMMA selected.
- The saved context gives every Team B row, the Priest included, policy kind
  ``tdm_gamma``, GAMMA's algorithm ID, deterministic execution and the
  ``pressure_protocol_digest``; ``pressure_protocol`` reads
  ``reactive-team-deathmatch-gamma-controller@2``; GAMMA rows count as policy
  execution.
- With GAMMA on Team A and BETA on Team B, each row carries its own team's
  controller: Team A rows name GAMMA and ``team_a_pressure_protocol_digest``
  (``team_a_pressure_protocol`` reads
  ``reactive-team-deathmatch-gamma-controller@2``), and Team B rows name BETA
  and ``pressure_protocol_digest`` (``pressure_protocol`` reads
  ``scenario-5-pressure-controller@5``).
- The action-source payload is exactly V6 for Team B GAMMA and needs GAMMA's
  identity. It is exactly V7 when Team A is GAMMA or BETA, and needs Team A's
  identity; the saved context hashes that V7 payload. V1, V4, V5 and V6
  payload bytes are unchanged for the same inputs, and none of them gains
  ``team_a_scenario_controller``.
- The launcher's recording metadata accepts GAMMA on either team, beside every
  supported controller on the other team.
- The match summary names GAMMA "Reactive TDM GAMMA", even in a scenario whose
  name selects a scenario-pressure label for BETA.
- Recording epochs (causal audit): each recorded transition starts at the frame
  of its decision tick, that frame holds the pre-action Hunter Trap ticks, and
  the recorded Team B action equals GAMMA's choice from that same observation.
- Trap hold in the recording: while the trapped actor has 2 or more Trap
  ticks, no Team B non-Priest targets it. This is GAMMA's rule, not luck:
  BETA, given one of those same observations, does target it.
- A saved GAMMA replay reopens with GAMMA rows and the GAMMA label.

The session scenario is the compiled Scenario 1 fixture with Team B's Warrior
and Hunter alive and a 3-tick Hunter Trap already on Team A slot 2, so Team B
sees Trap ticks 3, 2, 1 and 0 over the recorded decisions.
"""

from collections.abc import Sequence
from dataclasses import replace
from functools import cache
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scripts.dev.visual_debugger.control as control
import scripts.dev.visual_debugger.evaluation_bridge as evaluation_bridge
from pydantic import ValidationError
from scripts.dev.debug_renderer import (
    _recording_policy_execution_included,  # pyright: ignore[reportPrivateUsage]
)
from scripts.dev.visual_debugger.evaluation_bridge import (
    build_debugger_evaluation_context_v1,
    build_debugger_evaluation_launch_specification_v1,
)
from scripts.dev.visual_debugger.model import (
    DebuggerScenario,
    DebuggerScenarioProvenance,
    DebuggerSession,
    TeamBController,
    TeamController,
    team_controller_action_source,
)
from scripts.dev.visual_debugger.protocol import (
    CombatConfigurationV1,
    CommandRequestV1,
    SetCombatConfigurationCommandV1,
)
from scripts.dev.visual_debugger.recording import (
    build_debugger_recording_specification_v1,
    recording_policy_execution_included,
)
from scripts.dev.visual_debugger.replay_recorder import DebuggerReplayRecorder
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.scenario_controller_fixtures import load_scenario_1
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION,
    PRIEST_CLASS_ID,
    STUN_CHANNEL_HUNTER_TRAP,
    TEAM_A_ID,
    TEAM_B_ID,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV2,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContextV3,
    EvaluationFrameV2,
    ExecutionInformationMode,
    JointActionV1,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.policy_execution import controller_identity, policy
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_io import (
    load_replay,
    preflight_replay_destination,
)
from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.reactive_tdm_alpha import (
    reactive_tdm_alpha_controller_descriptor,
)
from marl_battlegrounds.policies.reactive_tdm_beta import (
    reactive_tdm_beta_controller_descriptor,
    reactive_tdm_beta_policy,
)
from marl_battlegrounds.policies.reactive_tdm_gamma import (
    reactive_tdm_gamma_controller_descriptor,
    reactive_tdm_gamma_policy,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsPolicy,
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)
from marl_battlegrounds.viewer.match_summary import build_match_summary_v1

_GAMMA_ALGORITHM = "reactive-team-deathmatch-gamma-controller"
_GAMMA_VERSION = 2
_GAMMA_PROTOCOL = f"{_GAMMA_ALGORITHM}@{_GAMMA_VERSION}"
_BETA_ALGORITHM = "scenario-5-pressure-controller"
_GAMMA_LABEL = "Reactive TDM GAMMA"
# Team A controllers that keep the V6 payload beside Team B GAMMA. BETA and
# GAMMA on Team A make the payload V7 and have their own tests.
_V6_TEAM_A_CONTROLLERS: tuple[TeamController, ...] = (
    "manual",
    "reactive_tdm",
    "random_valid",
)
_ALL_CONTROLLERS: tuple[TeamController, ...] = (
    *_V6_TEAM_A_CONTROLLERS,
    "scenario_5",
    "tdm_gamma",
)
_GAMMA_ON_ONE_TEAM: tuple[tuple[TeamController, TeamBController], ...] = (
    ("manual", "tdm_gamma"),
    ("tdm_gamma", "manual"),
)
_TEAM_A_SLOTS = tuple(range(5))
_TEAM_B_SLOTS = tuple(range(5, 10))
_TRAPPED_TEAM_A_SLOT = 2
_TRAPPED_TARGET = 6 + _TRAPPED_TEAM_A_SLOT
_START_TRAP_TICKS = 3
_RECORDED_DECISIONS = 4
_ACTION_SOURCE_SCHEMA = "marl_battlegrounds.visual_debugger.action_source_contract"
_RUNTIME = RuntimeProvenanceV1(
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


@cache
def _scenario() -> DebuggerScenario:
    compiled = load_scenario_1()
    config = compiled.config._replace(
        max_steps=int(compiled.initial_state.step_count) + _RECORDED_DECISIONS + 1,
    )
    state = compiled.initial_state
    state = state._replace(
        alive_mask=state.alive_mask.at[6:8].set(True),
        current_health=state.current_health.at[6:8].set(
            config.agent_profile.max_health[6:8]
        ),
        stun_durations=state.stun_durations.at[
            _TRAPPED_TEAM_A_SLOT, STUN_CHANNEL_HUNTER_TRAP
        ].set(_START_TRAP_TICKS),
    )

    def build() -> tuple[EnvConfig, EnvState]:
        return config, state

    return DebuggerScenario(
        name="gamma_dev_client_scenario",
        title="Scenario 1 with a trapped Team A actor",
        description="Test-only physical fixture",
        mode="interactive",
        build_scenario=build,
        frames=(),
        default_controlled_slot=2,
    )


def _session(
    *,
    team_a: TeamController = "manual",
    team_b: TeamBController = "tdm_gamma",
    mode: ExecutionInformationMode = "shared_obs",
    scenario: DebuggerScenario | None = None,
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
        execution_information_mode=mode,
    )


def _context(
    *,
    team_a: TeamController = "manual",
    team_b: TeamBController = "tdm_gamma",
    mode: ExecutionInformationMode = "shared_obs",
    scenario: DebuggerScenario | None = None,
) -> EvaluationEpisodeContextV3:
    source = get_scenario("arena_5v5") if scenario is None else scenario
    config, _state = source.build_scenario()
    return build_debugger_evaluation_context_v1(
        debugger_test_launch_specification(7),
        scenario=source,
        config=config,
        run_generation=0,
        action_source_kind=team_controller_action_source(team_a, team_b),
        team_a_controller=team_a,
        team_b_controller=team_b,
        execution_information_mode=mode,
    )


def _aggregation(context: EvaluationEpisodeContextV3) -> dict[str, str]:
    return {row.name: row.value for row in context.aggregation_keys}


def _identity(
    descriptor: dict[str, object], context: EvaluationEpisodeContextV3
) -> ContentAddressedIdentityV1:
    return ContentAddressedIdentityV1.model_validate(
        {
            "identifier": descriptor["policy_id"],
            "version": descriptor["version"],
            "canonical_digest": canonical_digest_sha256(
                {"behavior": descriptor, "code_revision": context.code_revision}
            ),
        }
    )


def _action_payload(
    team_a: TeamController,
    team_b: TeamBController,
    *,
    scenario_digest: str = "a" * 64,
    reactive_identity: ContentAddressedIdentityV1 | None = None,
    scenario_identity: ContentAddressedIdentityV1 | None = None,
    team_a_scenario_identity: ContentAddressedIdentityV1 | None = None,
) -> dict[str, object]:
    return evaluation_bridge._action_source_contract_payload(  # pyright: ignore[reportPrivateUsage]
        action_source_kind=team_controller_action_source(team_a, team_b),
        scenario_mode="interactive",
        team_a_controller=team_a,
        team_b_controller=team_b,
        execution_information_mode="shared_obs",
        scenario_contract_digest=scenario_digest,
        reactive_tdm_identity=reactive_identity,
        scenario_controller_identity=scenario_identity,
        team_a_scenario_controller_identity=team_a_scenario_identity,
    )


def _tree_equal(left: object, right: object) -> bool:
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _team_b_choice(
    session: DebuggerSession, controller: SharedObsPolicy = reactive_tdm_gamma_policy
) -> ActorAction:
    return cast(
        ActorAction,
        execute_shared_obs_team_policy(
            session.observation,
            session.action_mask,
            control._policy_keys(session),  # pyright: ignore[reportPrivateUsage]
            build_shared_obs_sensor_source_bank(session.observation),
            build_default_shared_obs_information_availability(
                session.config.agent_profile.active_mask,
                session.config.agent_profile.team_ids,
            ),
            policy=controller,
            team_identity=TEAM_B_ID,
        ),
    )


def _actor_rows(action: ActorAction) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(move), int(target), int(ultimate))
        for move, target, ultimate in zip(
            np.asarray(action.move).tolist(),
            np.asarray(action.select_target).tolist(),
            np.asarray(action.use_ultimate).tolist(),
            strict=True,
        )
    )


def _joint_rows(
    action: JointActionV1, slots: Sequence[int]
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (action.move[slot], action.select_target[slot], action.use_ultimate[slot])
        for slot in slots
    )


def _assert_gamma_team_b_rows(context: EvaluationEpisodeContextV3) -> None:
    aggregation = _aggregation(context)
    digest = _identity(
        reactive_tdm_gamma_controller_descriptor(), context
    ).canonical_digest
    assert aggregation["team_b_controller"] == "tdm_gamma"
    assert aggregation["pressure_protocol"] == _GAMMA_PROTOCOL
    assert aggregation["pressure_protocol_digest"] == digest
    team_b_slots: list[int] = []
    priest_slots: list[int] = []
    for roster_row, row in zip(context.roster, context.policy_assignments, strict=True):
        if not roster_row.configured_active or roster_row.configured_team_id != (
            TEAM_B_ID
        ):
            continue
        assert isinstance(row, AssignedPolicySlotV2)
        assert row.policy_kind == "tdm_gamma"
        assert (
            row.policy_id == f"debugger-action-source:tdm_gamma:slot:{row.global_slot}"
        )
        assert row.algorithm_id == _GAMMA_ALGORITHM
        assert row.execution_mode == "deterministic"
        assert row.policy_content_digest == aggregation["pressure_protocol_digest"]
        assert row.checkpoint_digest is None
        assert row.training_run_id == "not_applicable"
        team_b_slots.append(row.global_slot)
        if roster_row.class_id == PRIEST_CLASS_ID:
            priest_slots.append(row.global_slot)
    assert tuple(team_b_slots) == _TEAM_B_SLOTS
    assert len(priest_slots) == 1


def _assert_unchanged_gamma_session(session: DebuggerSession) -> None:
    assert session.run_generation == 0
    assert (session.team_a_controller, session.team_b_controller) == (
        "manual",
        "tdm_gamma",
    )
    assert session.evaluation_context.execution_information_mode == "shared_obs"
    assert session.current_evaluation_frame.frame_index == 0
    assert session.incoming_evaluation_view is None
    assert session.evaluation_context == _session().evaluation_context
    assert (
        control.set_combat_configuration(
            session,
            team_a_controller="manual",
            team_b_controller="tdm_gamma",
            execution_information_mode="shared_obs",
        )
        is session
    )


class _RecordedGamma(NamedTuple):
    sessions: tuple[DebuggerSession, ...]
    replay_path: Path


class _StopBeforeExecutionError(Exception):
    pass


@pytest.fixture(scope="module")
def recorded_gamma(tmp_path_factory: pytest.TempPathFactory) -> _RecordedGamma:
    session = _session(recording=True)
    recorder = DebuggerReplayRecorder(
        specification=build_debugger_recording_specification_v1(
            action_source_kind="mixed",
            runtime_provenance=_RUNTIME,
        ),
        destination=preflight_replay_destination(
            tmp_path_factory.mktemp("gamma") / "gamma-controller.marlbg-replay.json"
        ),
        context=session.evaluation_context,
        initial_frame=session.current_evaluation_frame,
    )
    sessions = [session]
    for _ in range(_RECORDED_DECISIONS):
        advanced = control.submit_interactive(sessions[-1])
        incoming = advanced.incoming_evaluation_view
        assert incoming is not None
        assert isinstance(incoming.successor_frame, EvaluationFrameV2)
        recorder.append(incoming.transition, incoming.successor_frame)
        sessions.append(advanced)
        if advanced.episode_sealed:
            break
    # An early game end seals the recording with its own close cause; the
    # tests then report the missing decisions instead of a save error.
    close_cause = recorder.close_cause or "finish_and_review"
    assert recorder.finalize_and_save(close_cause) == "saved"
    assert recorder.saved_bundle is not None
    return _RecordedGamma(tuple(sessions), recorder.saved_bundle.replay_path)


def test_gamma_is_admitted_on_team_b_with_shared_obs() -> None:
    for team_a in _V6_TEAM_A_CONTROLLERS:
        configuration = {
            "team_a_controller": team_a,
            "team_b_controller": "tdm_gamma",
            "execution_information_mode": "shared_obs",
        }
        assert (
            CombatConfigurationV1.model_validate(configuration).team_b_controller
            == "tdm_gamma"
        )
        request = CommandRequestV1.model_validate(
            {
                "client_id": "client-1",
                "command_id": "command-1",
                "base_revision": 0,
                "command": {
                    "command_type": "set_combat_configuration",
                    **configuration,
                },
            }
        )
        assert isinstance(request.command, SetCombatConfigurationCommandV1)
        assert request.command.team_b_controller == "tdm_gamma"
        session = _session(team_a=team_a)
        assert (session.team_a_controller, session.team_b_controller) == (
            team_a,
            "tdm_gamma",
        )
        assert session.evaluation_context.execution_information_mode == "shared_obs"
        aggregation = _aggregation(session.evaluation_context)
        assert aggregation["team_a_controller"] == team_a
        assert aggregation["team_b_controller"] == "tdm_gamma"
        assert aggregation["action_source"] == (
            "mixed" if team_a == "manual" else "policy"
        )
    manual = _session(team_b="manual", mode="no_shared_obs")
    installed = control.set_combat_configuration(
        manual,
        team_a_controller="manual",
        team_b_controller="tdm_gamma",
        execution_information_mode="shared_obs",
    )
    assert installed.run_generation == manual.run_generation + 1
    assert installed.team_b_controller == "tdm_gamma"
    assert installed.evaluation_context.execution_information_mode == "shared_obs"
    assert _tree_equal(installed.state, manual.state)
    assert (
        control.set_combat_configuration(
            installed,
            team_a_controller="manual",
            team_b_controller="tdm_gamma",
            execution_information_mode="shared_obs",
        )
        is installed
    )


def test_protocol_accepts_team_a_gamma_and_needs_shared_obs() -> None:
    accepted = {
        "team_a_controller": "manual",
        "team_b_controller": "tdm_gamma",
        "execution_information_mode": "shared_obs",
    }
    on_team_a = {
        **accepted,
        "team_a_controller": "tdm_gamma",
        "team_b_controller": "manual",
    }
    on_both_teams = {**accepted, "team_a_controller": "tdm_gamma"}
    without_shared_obs = {**accepted, "execution_information_mode": "no_shared_obs"}
    team_a_without_shared_obs = {
        **on_team_a,
        "execution_information_mode": "no_shared_obs",
    }
    for model in (CombatConfigurationV1, SetCombatConfigurationCommandV1):
        for configuration in (on_team_a, on_both_teams):
            validated = model.model_validate(configuration)
            assert (validated.team_a_controller, validated.team_b_controller) == (
                configuration["team_a_controller"],
                configuration["team_b_controller"],
            )
        for rejected in (without_shared_obs, team_a_without_shared_obs):
            with pytest.raises(ValidationError, match="require SharedObs"):
                model.model_validate(rejected)
    for configuration in (on_team_a, on_both_teams):
        request = CommandRequestV1.model_validate(
            {
                "client_id": "client-1",
                "command_id": "command-1",
                "base_revision": 0,
                "command": {
                    "command_type": "set_combat_configuration",
                    **configuration,
                },
            }
        )
        assert isinstance(request.command, SetCombatConfigurationCommandV1)
        assert request.command.team_a_controller == "tdm_gamma"
    for rejected in (without_shared_obs, team_a_without_shared_obs):
        for command in (
            {"command_type": "set_combat_configuration", **rejected},
            {
                "command_type": "confirm_discard_and_replace",
                "replacement": {"command_type": "set_combat_configuration", **rejected},
            },
        ):
            with pytest.raises(ValidationError, match="require SharedObs"):
                CommandRequestV1.model_validate(
                    {
                        "client_id": "client-1",
                        "command_id": "command-1",
                        "base_revision": 0,
                        "command": command,
                    }
                )


def test_control_accepts_gamma_on_team_a_and_keeps_the_old_session() -> None:
    session = _session()
    for team_b in ("manual", "tdm_gamma"):
        installed = control.set_combat_configuration(
            session,
            team_a_controller="tdm_gamma",
            team_b_controller=team_b,
            execution_information_mode="shared_obs",
        )
        assert installed.run_generation == session.run_generation + 1
        assert (installed.team_a_controller, installed.team_b_controller) == (
            "tdm_gamma",
            team_b,
        )
        assert installed.evaluation_context.execution_information_mode == "shared_obs"
        aggregation = _aggregation(installed.evaluation_context)
        assert aggregation["team_a_controller"] == "tdm_gamma"
        assert aggregation["team_b_controller"] == team_b
        assert _tree_equal(installed.state, session.state)
    created = _session(team_a="tdm_gamma", team_b="manual")
    assert (created.team_a_controller, created.team_b_controller) == (
        "tdm_gamma",
        "manual",
    )
    assert _aggregation(created.evaluation_context)["action_source"] == "mixed"
    # DebuggerSession accepts the controller itself; this copy fails only
    # because its saved context still describes the old manual Team A.
    with pytest.raises(ValueError, match="must join its evaluation context"):
        replace(session, team_a_controller="tdm_gamma")
    _assert_unchanged_gamma_session(session)


def test_control_rejects_gamma_without_shared_obs_on_either_team() -> None:
    session = _session()
    manual = _session(team_b="manual", mode="no_shared_obs")
    for current in (session, manual):
        for team_a, team_b in _GAMMA_ON_ONE_TEAM:
            with pytest.raises(
                control.CombatConfigurationRejectedError, match="require SharedObs"
            ):
                control.set_combat_configuration(
                    current,
                    team_a_controller=team_a,
                    team_b_controller=team_b,
                    execution_information_mode="no_shared_obs",
                )
    for team_a, team_b in _GAMMA_ON_ONE_TEAM:
        with pytest.raises(
            control.CombatConfigurationRejectedError, match="require SharedObs"
        ):
            _session(team_a=team_a, team_b=team_b, mode="no_shared_obs")
    with pytest.raises(ValueError, match="require SharedObs"):
        replace(manual, team_b_controller="tdm_gamma")
    with pytest.raises(ValueError, match="require SharedObs"):
        replace(manual, team_a_controller="tdm_gamma")
    _assert_unchanged_gamma_session(session)
    assert manual.run_generation == 0
    assert (manual.team_a_controller, manual.team_b_controller) == (
        "manual",
        "manual",
    )
    assert manual.evaluation_context.execution_information_mode == "no_shared_obs"


def test_evaluation_context_needs_shared_obs_for_gamma_on_either_team() -> None:
    scripted = replace(get_scenario("arena_5v5"), mode="scripted")
    for team_a, team_b in _GAMMA_ON_ONE_TEAM:
        with pytest.raises(ValueError, match="require interactive SharedObs"):
            _context(team_a=team_a, team_b=team_b, mode="no_shared_obs")
        with pytest.raises(ValueError, match="require interactive SharedObs"):
            _context(team_a=team_a, team_b=team_b, scenario=scripted)
    aggregation = _aggregation(_context(team_a="tdm_gamma", team_b="manual"))
    assert aggregation["team_a_controller"] == "tdm_gamma"
    assert aggregation["team_b_controller"] == "manual"
    assert aggregation["action_source"] == "mixed"
    assert aggregation["team_a_pressure_protocol"] == _GAMMA_PROTOCOL
    assert "pressure_protocol" not in aggregation


def test_configured_policy_is_the_registered_gamma_apply_on_either_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    registered = policy("tdm-gamma")
    for team_identity in (TEAM_B_ID, TEAM_A_ID):
        configured = control._configured_policy(session, team_identity, "tdm_gamma")  # pyright: ignore[reportPrivateUsage]
        assert configured.apply is registered.apply
        assert configured.apply is not policy("tdm-beta").apply
        assert (
            configured.name,
            configured.variables,
            configured.initial_carry,
            configured.checkpoint,
            configured.execution,
        ) == ("tdm-gamma", (), (), None, "jax")
        assert controller_identity(configured) == {
            "identifier": _GAMMA_ALGORITHM,
            "version": _GAMMA_VERSION,
            "canonical_digest": canonical_digest_sha256(
                reactive_tdm_gamma_controller_descriptor()
            ),
        }
    applied: list[tuple[object, object]] = []
    real_apply_policies = control.apply_policies

    def spy(*args: object, **kwargs: object) -> object:
        applied.append((args[0], args[1]))
        return real_apply_policies(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(control, "apply_policies", spy)
    control._build_configured_joint_action(session)  # pyright: ignore[reportPrivateUsage]
    assert applied == [
        (control._manual_policy, registered.apply)  # pyright: ignore[reportPrivateUsage]
    ]

    # Team A GAMMA: only the argument order matters, so stop before GAMMA runs.
    def stop(*args: object, **_kwargs: object) -> object:
        applied.append((args[0], args[1]))
        raise _StopBeforeExecutionError

    monkeypatch.setattr(control, "apply_policies", stop)
    with pytest.raises(_StopBeforeExecutionError):
        control._build_configured_joint_action(  # pyright: ignore[reportPrivateUsage]
            _session(team_a="tdm_gamma", team_b="manual")
        )
    assert applied[1:] == [
        (registered.apply, control._manual_policy)  # pyright: ignore[reportPrivateUsage]
    ]


def test_gamma_step_submits_legal_accepted_team_b_actions(
    recorded_gamma: _RecordedGamma,
) -> None:
    sessions = recorded_gamma.sessions
    configured = control._build_configured_joint_action(sessions[0])  # pyright: ignore[reportPrivateUsage]
    first = sessions[1].incoming_evaluation_view
    assert first is not None
    assert _joint_rows(
        first.transition.facts.action_acceptance_facts.submitted_joint_action,
        range(10),
    ) == _actor_rows(ActorAction(*configured))
    targeted = False
    for before, after in pairwise(sessions):
        incoming = after.incoming_evaluation_view
        assert incoming is not None
        acceptance = incoming.transition.facts.action_acceptance_facts
        move_mask = np.asarray(before.action_mask.move_mask)
        combat_mask = np.asarray(
            before.action_mask.select_target_use_ultimate_joint_mask
        )
        submitted = _joint_rows(acceptance.submitted_joint_action, _TEAM_B_SLOTS)
        assert submitted == _joint_rows(acceptance.accepted_joint_action, _TEAM_B_SLOTS)
        for slot, (move, target, ultimate) in zip(
            _TEAM_B_SLOTS, submitted, strict=True
        ):
            assert bool(move_mask[slot, move])
            assert bool(combat_mask[slot, target, ultimate])
            assert not acceptance.submitted_action_tuple_is_out_of_domain_by_actor[slot]
            assert not acceptance.in_domain_move_action_is_rejected_by_actor[slot]
            assert not acceptance.in_domain_combat_action_pair_is_rejected_by_actor[
                slot
            ]
            targeted = targeted or target != 0
        assert int(after.state.step_count) == int(before.state.step_count) + 1
    assert targeted


def test_reset_and_reselection_keep_gamma_selected(
    recorded_gamma: _RecordedGamma,
) -> None:
    initial, advanced = recorded_gamma.sessions[0], recorded_gamma.sessions[-1]
    restarted = control.reset_session(advanced)
    assert restarted.run_generation == advanced.run_generation + 1 == 1
    assert (restarted.team_a_controller, restarted.team_b_controller) == (
        "manual",
        "tdm_gamma",
    )
    assert _tree_equal(restarted.state, initial.state)
    assert _tree_equal(restarted.key, initial.key)
    assert restarted.current_evaluation_frame.frame_index == 0
    assert restarted.incoming_evaluation_view is None
    _assert_gamma_team_b_rows(restarted.evaluation_context)
    assert (
        restarted.evaluation_context.identity.episode_id
        != initial.evaluation_context.identity.episode_id
    )
    assert (
        control._configured_policy(  # pyright: ignore[reportPrivateUsage]
            restarted, TEAM_B_ID, restarted.team_b_controller
        ).apply
        is policy("tdm-gamma").apply
    )
    assert (
        control.set_combat_configuration(
            restarted,
            team_a_controller="manual",
            team_b_controller="tdm_gamma",
            execution_information_mode="shared_obs",
        )
        is restarted
    )
    manual = control.set_combat_configuration(
        restarted,
        team_a_controller="manual",
        team_b_controller="manual",
        execution_information_mode="shared_obs",
    )
    reselected = control.set_combat_configuration(
        manual,
        team_a_controller="manual",
        team_b_controller="tdm_gamma",
        execution_information_mode="shared_obs",
    )
    assert manual.team_b_controller == "manual"
    assert reselected.team_b_controller == "tdm_gamma"
    assert reselected.run_generation == restarted.run_generation + 2
    _assert_gamma_team_b_rows(reselected.evaluation_context)
    switched = control.switch_scenario(
        reselected,
        replace(reselected.scenario, name="renamed_gamma_copy", title="Renamed"),
    )
    assert switched.team_b_controller == "tdm_gamma"
    assert switched.scenario.name == "renamed_gamma_copy"
    _assert_gamma_team_b_rows(switched.evaluation_context)


@pytest.mark.parametrize("team_a", _V6_TEAM_A_CONTROLLERS)
def test_saved_context_records_gamma_rows_and_pressure_keys(
    team_a: TeamController,
) -> None:
    context = _context(team_a=team_a)
    _assert_gamma_team_b_rows(context)
    aggregation = _aggregation(context)
    assert aggregation["team_a_controller"] == team_a
    assert aggregation["action_source"] == ("mixed" if team_a == "manual" else "policy")
    assert (
        tuple(
            row.policy_kind
            for row in context.policy_assignments[:5]
            if isinstance(row, AssignedPolicySlotV2)
        )
        == (team_a,) * 5
    )
    if team_a == "reactive_tdm":
        assert (
            aggregation["reactive_tdm_controller_digest"]
            != aggregation["pressure_protocol_digest"]
        )
    else:
        assert "reactive_tdm_controller" not in aggregation
    beta = _context(team_a=team_a, team_b="scenario_5")
    assert (
        _aggregation(beta)["pressure_protocol_digest"]
        != aggregation["pressure_protocol_digest"]
    )
    assert beta.identity.evaluation_id != context.identity.evaluation_id
    # With a manual Team A, only the GAMMA rows can make this true.
    assert recording_policy_execution_included(context)


def test_saved_context_keeps_each_teams_gamma_or_beta_identity() -> None:
    context = _context(team_a="tdm_gamma", team_b="scenario_5")
    aggregation = _aggregation(context)
    gamma_digest = _identity(
        reactive_tdm_gamma_controller_descriptor(), context
    ).canonical_digest
    beta_digest = _identity(
        reactive_tdm_beta_controller_descriptor(), context
    ).canonical_digest
    assert gamma_digest != beta_digest
    assert aggregation["team_a_controller"] == "tdm_gamma"
    assert aggregation["team_b_controller"] == "scenario_5"
    assert aggregation["action_source"] == "policy"
    assert aggregation["team_a_pressure_protocol"] == _GAMMA_PROTOCOL
    assert aggregation["team_a_pressure_protocol_digest"] == gamma_digest
    assert aggregation["pressure_protocol"] == f"{_BETA_ALGORITHM}@5"
    assert aggregation["pressure_protocol_digest"] == beta_digest
    assert "reactive_tdm_controller" not in aggregation
    slots_by_team: dict[int, list[int]] = {TEAM_A_ID: [], TEAM_B_ID: []}
    for roster_row, row in zip(context.roster, context.policy_assignments, strict=True):
        if not roster_row.configured_active:
            continue
        assert isinstance(row, AssignedPolicySlotV2)
        team_id = roster_row.configured_team_id
        policy_kind, algorithm_id, digest = (
            (
                "tdm_gamma",
                _GAMMA_ALGORITHM,
                aggregation["team_a_pressure_protocol_digest"],
            )
            if team_id == TEAM_A_ID
            else (
                "scenario_5",
                _BETA_ALGORITHM,
                aggregation["pressure_protocol_digest"],
            )
        )
        assert row.policy_kind == policy_kind
        assert row.policy_id == (
            f"debugger-action-source:{policy_kind}:slot:{row.global_slot}"
        )
        assert row.algorithm_id == algorithm_id
        assert row.execution_mode == "deterministic"
        assert row.policy_content_digest == digest
        assert row.checkpoint_digest is None
        assert row.training_run_id == "not_applicable"
        slots_by_team[team_id].append(row.global_slot)
    assert tuple(slots_by_team[TEAM_A_ID]) == _TEAM_A_SLOTS
    assert tuple(slots_by_team[TEAM_B_ID]) == _TEAM_B_SLOTS
    assert recording_policy_execution_included(context)


@pytest.mark.parametrize("team_a", _V6_TEAM_A_CONTROLLERS)
def test_v6_action_source_payload_is_exact_and_needs_gamma_identity(
    team_a: TeamController,
) -> None:
    scenario = get_scenario("arena_5v5")
    context = _context(team_a=team_a, scenario=scenario)
    gamma_identity = _identity(reactive_tdm_gamma_controller_descriptor(), context)
    reactive_identity = (
        _identity(reactive_tdm_alpha_controller_descriptor(), context)
        if team_a == "reactive_tdm"
        else None
    )
    scenario_digest = canonical_digest_sha256(
        evaluation_bridge._scenario_contract_payload(scenario)  # pyright: ignore[reportPrivateUsage]
    )
    payload = _action_payload(
        team_a,
        "tdm_gamma",
        scenario_digest=scenario_digest,
        reactive_identity=reactive_identity,
        scenario_identity=gamma_identity,
    )
    expected: dict[str, object] = {
        "schema_id": _ACTION_SOURCE_SCHEMA,
        "schema_version": 6,
        "action_source_kind": "mixed" if team_a == "manual" else "policy",
        "team_a_controller": team_a,
        "team_b_controller": "tdm_gamma",
        "execution_information_mode": "shared_obs",
        "manual_submission_included": team_a == "manual",
        "reactive_tdm_execution_included": team_a == "reactive_tdm",
        "random_policy_execution_included": team_a == "random_valid",
        "scenario_3_execution_included": False,
        "reactive_tdm_controller": reactive_identity,
        "scenario_controller": gamma_identity,
        "scenario_contract_digest_sha256": scenario_digest,
        "policy_execution_included": True,
        "tdm_gamma_execution_included": True,
    }
    assert payload == expected
    assert (
        gamma_identity.canonical_digest
        == _aggregation(context)["pressure_protocol_digest"]
    )
    if team_a != "reactive_tdm":
        # Manual and Random rows carry the action-source digest itself.
        for row in context.policy_assignments[:5]:
            assert isinstance(row, AssignedPolicySlotV2)
            assert row.policy_content_digest == canonical_digest_sha256(expected)
    with pytest.raises(
        ValueError, match="Scenario action source requires its controller identity"
    ):
        _action_payload(team_a, "tdm_gamma", reactive_identity=reactive_identity)


def test_v7_action_source_payload_is_exact_and_needs_team_a_identity() -> None:
    gamma = ContentAddressedIdentityV1(
        identifier=_GAMMA_ALGORITHM,
        version=_GAMMA_VERSION,
        canonical_digest="6" * 64,
    )
    beta = ContentAddressedIdentityV1(
        identifier=_BETA_ALGORITHM,
        version=5,
        canonical_digest="5" * 64,
    )
    cases: tuple[tuple[dict[str, object], dict[str, object]], ...] = (
        (
            _action_payload("tdm_gamma", "manual", team_a_scenario_identity=gamma),
            {
                "schema_id": _ACTION_SOURCE_SCHEMA,
                "schema_version": 7,
                "action_source_kind": "mixed",
                "team_a_controller": "tdm_gamma",
                "team_b_controller": "manual",
                "execution_information_mode": "shared_obs",
                "manual_submission_included": True,
                "reactive_tdm_execution_included": False,
                "random_policy_execution_included": False,
                "scenario_3_execution_included": False,
                "reactive_tdm_controller": None,
                "scenario_controller": None,
                "scenario_contract_digest_sha256": "a" * 64,
                "policy_execution_included": True,
                "team_a_scenario_controller": gamma,
                "scenario_5_execution_included": False,
                "tdm_gamma_execution_included": True,
            },
        ),
        (
            _action_payload(
                "scenario_5",
                "tdm_gamma",
                scenario_identity=gamma,
                team_a_scenario_identity=beta,
            ),
            {
                "schema_id": _ACTION_SOURCE_SCHEMA,
                "schema_version": 7,
                "action_source_kind": "policy",
                "team_a_controller": "scenario_5",
                "team_b_controller": "tdm_gamma",
                "execution_information_mode": "shared_obs",
                "manual_submission_included": False,
                "reactive_tdm_execution_included": False,
                "random_policy_execution_included": False,
                "scenario_3_execution_included": False,
                "reactive_tdm_controller": None,
                "scenario_controller": gamma,
                "scenario_contract_digest_sha256": "a" * 64,
                "policy_execution_included": True,
                "team_a_scenario_controller": beta,
                "scenario_5_execution_included": True,
                "tdm_gamma_execution_included": True,
            },
        ),
    )
    for payload, expected in cases:
        assert payload == expected
        assert canonical_json_bytes(payload) == canonical_json_bytes(expected)
    with pytest.raises(ValueError, match="Team A scenario action source requires"):
        _action_payload("tdm_gamma", "manual")
    with pytest.raises(ValueError, match="Team A scenario action source requires"):
        _action_payload("scenario_5", "tdm_gamma", scenario_identity=gamma)
    with pytest.raises(
        ValueError, match="Scenario action source requires its controller identity"
    ):
        _action_payload("scenario_5", "tdm_gamma", team_a_scenario_identity=beta)
    # The saved context hashes the V7 payload: Team B's Manual rows carry the
    # action-source digest itself.
    scenario = get_scenario("arena_5v5")
    context = _context(team_a="tdm_gamma", team_b="manual", scenario=scenario)
    saved = _action_payload(
        "tdm_gamma",
        "manual",
        scenario_digest=canonical_digest_sha256(
            evaluation_bridge._scenario_contract_payload(scenario)  # pyright: ignore[reportPrivateUsage]
        ),
        team_a_scenario_identity=_identity(
            reactive_tdm_gamma_controller_descriptor(), context
        ),
    )
    assert saved["schema_version"] == 7
    for row in context.policy_assignments[5:]:
        assert isinstance(row, AssignedPolicySlotV2)
        assert row.policy_kind == "manual"
        assert row.policy_content_digest == canonical_digest_sha256(saved)


def test_v1_v4_v5_v6_action_source_payload_bytes_are_unchanged() -> None:
    reactive = ContentAddressedIdentityV1(
        identifier="reactive-team-deathmatch-controller",
        version=3,
        canonical_digest="4" * 64,
    )
    beta = ContentAddressedIdentityV1(
        identifier="scenario-5-pressure-controller",
        version=5,
        canonical_digest="5" * 64,
    )
    gamma = ContentAddressedIdentityV1(
        identifier=_GAMMA_ALGORITHM,
        version=_GAMMA_VERSION,
        canonical_digest="6" * 64,
    )
    cases: tuple[tuple[dict[str, object], dict[str, object]], ...] = (
        (
            _action_payload("reactive_tdm", "random_valid", reactive_identity=reactive),
            {
                "schema_id": _ACTION_SOURCE_SCHEMA,
                "schema_version": 4,
                "action_source_kind": "policy",
                "team_a_controller": "reactive_tdm",
                "team_b_controller": "random_valid",
                "execution_information_mode": "shared_obs",
                "manual_submission_included": False,
                "reactive_tdm_execution_included": True,
                "random_policy_execution_included": True,
                "scenario_3_execution_included": False,
                "reactive_tdm_controller": reactive,
                "scenario_controller": None,
                "scenario_contract_digest_sha256": "a" * 64,
                "policy_execution_included": True,
            },
        ),
        (
            _action_payload("random_valid", "manual"),
            {
                "schema_id": _ACTION_SOURCE_SCHEMA,
                "schema_version": 4,
                "action_source_kind": "mixed",
                "team_a_controller": "random_valid",
                "team_b_controller": "manual",
                "execution_information_mode": "shared_obs",
                "manual_submission_included": True,
                "reactive_tdm_execution_included": False,
                "random_policy_execution_included": True,
                "scenario_3_execution_included": False,
                "reactive_tdm_controller": None,
                "scenario_controller": None,
                "scenario_contract_digest_sha256": "a" * 64,
                "policy_execution_included": True,
            },
        ),
        (
            _action_payload("manual", "scenario_5", scenario_identity=beta),
            {
                "schema_id": _ACTION_SOURCE_SCHEMA,
                "schema_version": 5,
                "action_source_kind": "mixed",
                "team_a_controller": "manual",
                "team_b_controller": "scenario_5",
                "execution_information_mode": "shared_obs",
                "manual_submission_included": True,
                "reactive_tdm_execution_included": False,
                "random_policy_execution_included": False,
                "scenario_3_execution_included": False,
                "scenario_5_execution_included": True,
                "reactive_tdm_controller": None,
                "scenario_controller": beta,
                "scenario_contract_digest_sha256": "a" * 64,
                "policy_execution_included": True,
            },
        ),
        (
            _action_payload("random_valid", "tdm_gamma", scenario_identity=gamma),
            {
                "schema_id": _ACTION_SOURCE_SCHEMA,
                "schema_version": 6,
                "action_source_kind": "policy",
                "team_a_controller": "random_valid",
                "team_b_controller": "tdm_gamma",
                "execution_information_mode": "shared_obs",
                "manual_submission_included": False,
                "reactive_tdm_execution_included": False,
                "random_policy_execution_included": True,
                "scenario_3_execution_included": False,
                "reactive_tdm_controller": None,
                "scenario_controller": gamma,
                "scenario_contract_digest_sha256": "a" * 64,
                "policy_execution_included": True,
                "tdm_gamma_execution_included": True,
            },
        ),
        (
            evaluation_bridge._action_source_contract_payload(  # pyright: ignore[reportPrivateUsage]
                action_source_kind="scripted",
                scenario_mode="scripted",
                team_a_controller="manual",
                team_b_controller="manual",
                execution_information_mode="shared_obs",
                scenario_contract_digest="a" * 64,
            ),
            {
                "schema_id": _ACTION_SOURCE_SCHEMA,
                "schema_version": 1,
                "action_source_kind": "scripted",
                "team_b_controller": "manual",
                "execution_information_mode": "shared_obs",
                "manual_submission_included": False,
                "scripted_submission_included": True,
                "scenario_contract_digest_sha256": "a" * 64,
                "policy_execution_included": False,
            },
        ),
    )
    for payload, expected in cases:
        assert canonical_json_bytes(payload) == canonical_json_bytes(expected)
        assert canonical_digest_sha256(payload) == canonical_digest_sha256(expected)
        assert "team_a_scenario_controller" not in payload
        assert ("tdm_gamma_execution_included" in payload) == (
            payload["schema_version"] == 6
        )


@pytest.mark.parametrize("other", _ALL_CONTROLLERS)
def test_launcher_recording_metadata_accepts_gamma_on_either_team(
    other: TeamController,
) -> None:
    for team_a, team_b in (("tdm_gamma", other), (other, "tdm_gamma")):
        session = SimpleNamespace(team_a_controller=team_a, team_b_controller=team_b)
        assert _recording_policy_execution_included(session) is True


def test_match_summary_names_gamma_team_b() -> None:
    source = get_scenario("arena_5v5")
    config, state = source.build_scenario()
    config = config._replace(task_mode=1, team_deathmatch_score_threshold=20)
    scenario = replace(
        source,
        name="scenario_5",
        build_scenario=lambda: (config, state),
        provenance=DebuggerScenarioProvenance(
            source_kind="saved_draft",
            source_identity="saved_draft:scenario:scenario_5:r1",
            scenario_semantic_digest="b" * 64,
            map_semantic_digest="c" * 64,
            resolved_configuration_digest="d" * 64,
            resolved_initial_state_digest="e" * 64,
        ),
    )
    names: dict[str, str] = {}
    for team_b in ("tdm_gamma", "scenario_5"):
        session = _session(team_b=team_b, scenario=scenario)
        summary = build_match_summary_v1(
            session.evaluation_context, session.current_evaluation_frame
        )
        names[team_b] = summary.teams[1].display_name
        assert summary.teams[0].display_name == "Manual"
        assert summary.teams[1].policy_ids == tuple(
            f"debugger-action-source:{team_b}:slot:{slot}" for slot in _TEAM_B_SLOTS
        )
    # The BETA label proves the scenario-pressure naming path is active here.
    assert names == {
        "tdm_gamma": _GAMMA_LABEL,
        "scenario_5": "tdm-scenario-5-controller-beta",
    }


def test_recorded_gamma_transitions_pair_each_decision_frame_with_its_choice(
    recorded_gamma: _RecordedGamma,
) -> None:
    sessions = recorded_gamma.sessions
    replay = load_replay(recorded_gamma.replay_path).replay
    assert isinstance(replay, ReplayArtifactV3)
    assert len(replay.frames) == len(replay.transitions) + 1 == len(sessions)
    config = sessions[0].config
    team_b_priests = tuple(
        slot
        for slot in _TEAM_B_SLOTS
        if int(config.agent_profile.class_ids[slot]) == PRIEST_CLASS_ID
    )
    assert len(team_b_priests) == 1
    trap_ticks_by_frame: list[int] = []
    successor_choice_differs = False
    beta_hits_trapped = False
    for index, transition in enumerate(replay.transitions):
        before, after = sessions[index], sessions[index + 1]
        frame = replay.frames[index]
        assert transition.transition_index == frame.frame_index == index
        assert transition.start_frame_id == frame.frame_id
        assert transition.successor_frame_id == replay.frames[index + 1].frame_id
        assert (
            transition.facts.transition_start_step_count
            == frame.simulator_step_count
            == int(before.state.step_count)
        )
        assert frame == before.current_evaluation_frame
        assert after.incoming_evaluation_view is not None
        assert transition == after.incoming_evaluation_view.transition

        # The decision frame carries the pre-action Trap ticks GAMMA saw.
        recorded_trap = np.asarray(frame.base_observation.enemy_unit_features)[
            5:, :, AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION
        ]
        np.testing.assert_array_equal(
            recorded_trap,
            np.asarray(before.observation.enemy_unit_features)[
                5:, :, AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION
            ],
        )
        visible = np.asarray(frame.base_observation.enemy_visibility_mask)[
            5:, _TRAPPED_TEAM_A_SLOT
        ]
        assert bool(visible.any())
        seen = recorded_trap[:, _TRAPPED_TEAM_A_SLOT]
        assert set(seen[visible].tolist()) == {float(seen.max())}
        assert not bool(seen[~visible].any())
        trap_ticks = int(seen.max())
        trap_ticks_by_frame.append(trap_ticks)

        recorded = _joint_rows(
            transition.facts.action_acceptance_facts.submitted_joint_action,
            _TEAM_B_SLOTS,
        )
        assert recorded == _actor_rows(_team_b_choice(before))
        successor_choice_differs = successor_choice_differs or recorded != (
            _actor_rows(_team_b_choice(after))
        )
        if trap_ticks >= 2:
            beta = _actor_rows(_team_b_choice(before, reactive_tdm_beta_policy))
            for slot, (_move, target, _ultimate), (_, beta_target, _) in zip(
                _TEAM_B_SLOTS, recorded, beta, strict=True
            ):
                if slot not in team_b_priests:
                    assert target != _TRAPPED_TARGET
                    beta_hits_trapped = beta_hits_trapped or (
                        beta_target == _TRAPPED_TARGET
                    )
    assert len(replay.transitions) == _RECORDED_DECISIONS
    assert trap_ticks_by_frame == list(
        range(_START_TRAP_TICKS, _START_TRAP_TICKS - _RECORDED_DECISIONS, -1)
    )
    # A one-tick epoch shift would change at least one recorded Team B action.
    assert successor_choice_differs
    # BETA would hit the trapped actor here, so the Trap hold above is GAMMA's.
    assert beta_hits_trapped


def test_gamma_replay_reopens_with_gamma_rows_and_label(
    recorded_gamma: _RecordedGamma,
) -> None:
    initial = recorded_gamma.sessions[0]
    reopened = load_replay(recorded_gamma.replay_path)
    assert isinstance(reopened.replay, ReplayArtifactV3)
    header = reopened.replay.header
    assert header.context == initial.evaluation_context
    assert header.runtime_provenance.policy_execution_included
    context = header.context
    assert isinstance(context, EvaluationEpisodeContextV3)
    _assert_gamma_team_b_rows(context)
    assert recording_policy_execution_included(context)
    summary = build_match_summary_v1(context, reopened.replay.frames[0])
    assert summary.teams[1].display_name == _GAMMA_LABEL
    assert summary.teams[0].display_name == "Manual"
