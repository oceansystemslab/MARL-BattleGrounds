"""Write deterministic replay files for browser integration tests.

Playwright runs this module in a fresh temporary directory, then opens the real
Replay Viewer with those files. It does not use the user's saved recordings.
The default export writes historical version-1 replay bundles. The
--c7-acceptance export writes current version-4 replays, which record the Red
Zone depth: the eight installed scenarios through the shared evaluator and one
dense-death DevClient capture. The --red-zone export writes one current
version-4 DevClient capture at Red Zone depth 5.0 on the 20 x 12 fixture map:
in one 5v5 all-Mage tick Team A kills agents 5 and 6 inside Team B's Red Zone
(2 points each) and Team B kills agent 3 outside Team A's Red Zone (1 point).
It fails unless the deaths are (3, 5, 6) and the scores are (4, 1). Both
DevClient captures share one one-tick capture helper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
from tests.evaluation_fixtures import (
    captured_evaluation_trajectory,
    captured_team_deathmatch_threshold_trajectory,
    evaluation_env_config,
    mage_target_none_ultimate_action,
    neutral_action,
    valid_shared_availability,
)
from tests.evaluation_fixtures import (
    historical_initialize_scenario_state as initialize_scenario_state,
)
from tests.evaluation_fixtures import (
    historical_reset as reset,
)
from tests.evaluation_fixtures import (
    historical_step as step,
)

from marl_battlegrounds.evaluation.capture import (
    capture_evaluation_transition_unit_v1,
    capture_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.metrics import build_evaluation_observer_v1
from marl_battlegrounds.evaluation.replay import (
    ReplayBundleV1,
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.replay_io import (
    REPLAY_FILE_SUFFIX_V1,
    canonical_replay_json_bytes_v1,
    save_replay_bundle_v1,
)

if TYPE_CHECKING:
    from scripts.dev.visual_debugger.model import ActorCommand

    from marl_battlegrounds.core.types import EnvConfig, EnvState
    from marl_battlegrounds.evaluation.replay_v4 import ReplayArtifactV4

type _CompletionState = Literal["complete", "partial"]
type _InformationMode = Literal["no_shared_obs", "shared_obs"]


def _runtime_provenance() -> RuntimeProvenanceV1:
    return RuntimeProvenanceV1(
        python_version="3.14.0",
        package_version="0.0.0",
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


def _build_bundle(
    *,
    episode_id: str,
    transition_count: int,
    expected_horizon: int,
    completion_state: _CompletionState,
    execution_information_mode: _InformationMode = "no_shared_obs",
) -> ReplayBundleV1:
    trajectory = captured_evaluation_trajectory(
        transition_count=transition_count,
        expected_horizon=expected_horizon,
        execution_information_mode=execution_information_mode,
        episode_id=episode_id,
        actions=(
            (
                mage_target_none_ultimate_action(),
                *(neutral_action() for _ in range(transition_count - 1)),
            )
            if transition_count > 0
            else ()
        ),
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    observer.start(trajectory.frames[0])
    for transition, successor in zip(
        trajectory.transitions,
        trajectory.frames[1:],
        strict=True,
    ):
        observer.append(transition, successor)
    report = observer.finalize(
        completion_state=completion_state,
        end_or_failure_reason=(
            None if completion_state == "complete" else "browser_test_capture_stopped"
        ),
    )
    return build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=_runtime_provenance(),
    )


def build_corpse_overlay_bundle(
    *, execution_information_mode: _InformationMode
) -> ReplayBundleV1:
    episode_id = f"browser-corpse-overlay-{execution_information_mode}"
    trajectory = captured_evaluation_trajectory(
        transition_count=0,
        expected_horizon=2,
        execution_information_mode=execution_information_mode,
        episode_id=episode_id,
    )
    frame = trajectory.frames[0]
    config = evaluation_env_config()
    state, observation, action_mask, _ = reset(config, jax.random.PRNGKey(0))
    availability = (
        valid_shared_availability(trajectory.context)
        if execution_information_mode == "shared_obs"
        else None
    )
    baseline = capture_initial_evaluation_frame_v1(
        trajectory.context,
        state,
        observation,
        action_mask,
        availability,
    )
    if baseline != frame:
        raise RuntimeError("corpse-overlay fixture lost its reset authority.")
    authored_state = state._replace(
        agent_positions=(
            state.agent_positions.at[5].set((4.0, 1.5)).at[6].set((15.0, 10.0))
        ),
        alive_mask=state.alive_mask.at[5].set(False).at[6].set(False),
        current_health=state.current_health.at[5].set(0.0).at[6].set(0.0),
    )
    coherent_state, coherent_observation, coherent_mask, _ = initialize_scenario_state(
        authored_state,
        config,
    )
    corpse_frame = capture_initial_evaluation_frame_v1(
        trajectory.context,
        coherent_state,
        coherent_observation,
        coherent_mask,
        availability,
    )
    if corpse_frame.base_observation == frame.base_observation:
        raise RuntimeError("corpse-overlay fixture did not rebuild policy input.")
    observer = build_evaluation_observer_v1(trajectory.context)
    observer.start(corpse_frame)
    (
        successor_state,
        successor_observation,
        canonical_reward,
        done_flags,
        successor_mask,
        transition_info,
    ) = step(
        config,
        coherent_state,
        coherent_mask,
        neutral_action(),
        jax.random.PRNGKey(1),
    )
    transition, successor_frame = capture_evaluation_transition_unit_v1(
        trajectory.context,
        corpse_frame,
        successor_state,
        successor_observation,
        successor_mask,
        transition_info.transition_facts,
        canonical_reward,
        done_flags,
        successor_shared_obs_information_availability_by_recipient_and_sensor_source=(
            availability
        ),
    )
    observer.append(transition, successor_frame)
    report = observer.finalize(
        completion_state="partial",
        end_or_failure_reason="corpse_overlay_browser_fixture",
    )
    return build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=_runtime_provenance(),
    )


def export_artifacts(output_directory: Path) -> dict[str, str]:
    output_directory.mkdir(parents=True, exist_ok=True)
    complete = _build_bundle(
        episode_id="browser-replay-complete",
        transition_count=5,
        expected_horizon=5,
        completion_state="complete",
    )
    partial = _build_bundle(
        episode_id="browser-replay-partial",
        transition_count=2,
        expected_horizon=5,
        completion_state="partial",
    )
    shared = _build_bundle(
        episode_id="browser-replay-shared-source-material",
        transition_count=2,
        expected_horizon=2,
        completion_state="complete",
        execution_information_mode="shared_obs",
    )
    corpse_no_shared = build_corpse_overlay_bundle(
        execution_information_mode="no_shared_obs"
    )
    corpse_shared = build_corpse_overlay_bundle(execution_information_mode="shared_obs")
    tdm_bundles: dict[_InformationMode, ReplayBundleV1] = {}
    for mode in ("no_shared_obs", "shared_obs"):
        tdm = captured_team_deathmatch_threshold_trajectory(
            execution_information_mode=mode
        )
        tdm_observer = build_evaluation_observer_v1(tdm.context)
        tdm_observer.start(tdm.frames[0])
        tdm_observer.append(tdm.transitions[0], tdm.frames[1])
        tdm_bundles[mode] = build_replay_bundle_v1(
            tdm_observer,
            tdm_observer.finalize(completion_state="complete"),
            runtime_provenance=_runtime_provenance(),
        )

    complete_path = output_directory / f"complete{REPLAY_FILE_SUFFIX_V1}"
    partial_path = output_directory / f"partial{REPLAY_FILE_SUFFIX_V1}"
    shared_path = output_directory / f"shared{REPLAY_FILE_SUFFIX_V1}"
    tdm_path = output_directory / f"tdm{REPLAY_FILE_SUFFIX_V1}"
    tdm_shared_path = output_directory / f"tdm-shared{REPLAY_FILE_SUFFIX_V1}"
    corpse_no_shared_path = (
        output_directory / f"corpse-no-shared{REPLAY_FILE_SUFFIX_V1}"
    )
    corpse_shared_path = output_directory / f"corpse-shared{REPLAY_FILE_SUFFIX_V1}"
    save_replay_bundle_v1(complete, complete_path)
    save_replay_bundle_v1(partial, partial_path)
    save_replay_bundle_v1(shared, shared_path)
    save_replay_bundle_v1(tdm_bundles["no_shared_obs"], tdm_path)
    save_replay_bundle_v1(tdm_bundles["shared_obs"], tdm_shared_path)
    save_replay_bundle_v1(corpse_no_shared, corpse_no_shared_path)
    save_replay_bundle_v1(corpse_shared, corpse_shared_path)

    missing_directory = output_directory / "missing-sidecar"
    missing_directory.mkdir()
    missing_metric_path = missing_directory / f"complete{REPLAY_FILE_SUFFIX_V1}"
    missing_metric_path.write_bytes(canonical_replay_json_bytes_v1(complete.replay))

    return {
        "complete": str(complete_path.resolve()),
        "partial": str(partial_path.resolve()),
        "shared": str(shared_path.resolve()),
        "tdm": str(tdm_path.resolve()),
        "tdm_shared": str(tdm_shared_path.resolve()),
        "corpse_no_shared": str(corpse_no_shared_path.resolve()),
        "corpse_shared": str(corpse_shared_path.resolve()),
        "missing_metric": str(missing_metric_path.resolve()),
    }


def export_acceptance_artifacts(output_directory: Path) -> dict[str, object]:
    from scripts.dev.qualify_tdm_scenarios import (
        _planned_episode,  # pyright: ignore[reportPrivateUsage]
    )
    from scripts.dev.visual_debugger.model import ActorCommand

    from marl_battlegrounds.core.types import MOVE_STAY
    from marl_battlegrounds.evaluation.evaluate import evaluate_episodes
    from marl_battlegrounds.evaluation.policy_execution import policy
    from marl_battlegrounds.evaluation.replay_io import save_replay
    from marl_battlegrounds.evaluation.tdm_scenarios import TDM_BETA_SCENARIO_IDS

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = [""] * 8
    planned = {
        scenario_id: _planned_episode(scenario_id, 0)[1] for scenario_id in range(1, 9)
    }
    # Two shared evaluator calls amortize compilation over the eight real inputs.
    # This identity/render proof needs captures, not full numerical diagnostics.
    for pressure in ("tdm-alpha", "tdm-beta"):
        ids = tuple(
            scenario_id
            for scenario_id in planned
            if (scenario_id in TDM_BETA_SCENARIO_IDS) == (pressure == "tdm-beta")
        )
        episodes = tuple(planned[scenario_id] for scenario_id in ids)
        result = evaluate_episodes(
            policy("tdm-alpha"),
            policy(pressure),
            episodes,
            seed=0,
            num_envs=1,
            metrics="none",
            replay_episodes=tuple(episode.episode_id for episode in episodes),
            phase="qualification",
            pass_id="browser-acceptance",
            run_id="browser-c7",
            chunk_size=5,
        )
        for scenario_id, replay in zip(ids, result.replays, strict=True):
            path = output_directory / f"scenario-{scenario_id}{REPLAY_FILE_SUFFIX_V1}"
            save_replay(replay, path)
            paths[scenario_id - 1] = str(path.resolve())

    config = _all_mage_config(red_zone_depth=0.0)
    state, _, _, _ = reset(config, jax.random.key(0))
    positions = jnp.asarray(
        [(x, 1.5 + slot * 2.25) for x in (8.0, 10.0) for slot in range(5)],
        dtype=jnp.float32,
    )
    initial = state._replace(
        agent_positions=positions,
        current_health=jnp.ones((10,), dtype=jnp.float32),
    )
    replay, deaths, scores = _one_step_devclient_replay(
        config,
        initial,
        tuple(ActorCommand(slot, MOVE_STAY, (slot + 5) % 10, 0) for slot in range(10)),
        name="browser_dense_death_acceptance",
        title="Simultaneous reciprocal Mage deaths",
        description="Test-only HUD capacity witness, not an official scenario.",
        frame_description=(
            "All ten living Mages target their opposite peer in one epoch."
        ),
    )
    if deaths != tuple(range(10)) or scores != (5, 5):
        raise RuntimeError("dense witness must resolve all ten deaths through Core")
    dense = output_directory / f"dense-deaths{REPLAY_FILE_SUFFIX_V1}"
    save_replay(replay, dense)
    return {"scenarios": paths, "dense": str(dense.resolve())}


def _all_mage_config(*, red_zone_depth: float) -> EnvConfig:
    from marl_battlegrounds.core.config import resolve_agent_profile
    from marl_battlegrounds.core.types import MAGE_CLASS_ID

    return evaluation_env_config(
        team_sizes=(5, 5),
        task_mode=1,
        team_deathmatch_score_threshold=20,
        team_deathmatch_red_zone_depth=red_zone_depth,
        max_steps=1,
    )._replace(
        agent_profile=resolve_agent_profile(
            jnp.full((10,), MAGE_CLASS_ID, dtype=jnp.int32),
            jnp.asarray((5, 5), dtype=jnp.int32),
        )
    )


def _one_step_devclient_replay(
    config: EnvConfig,
    initial: EnvState,
    commands: tuple[ActorCommand, ...],
    *,
    name: str,
    title: str,
    description: str,
    frame_description: str,
) -> tuple[ReplayArtifactV4, tuple[int, ...], tuple[int, int]]:
    from scripts.dev.visual_debugger.control import (
        create_session,
        submit_next_script_frame,
    )
    from scripts.dev.visual_debugger.evaluation_bridge import (
        build_debugger_evaluation_launch_specification_v1,
    )
    from scripts.dev.visual_debugger.model import DebuggerScenario, ScenarioFrame
    from tests.visual_debugger_fixtures import debugger_test_launch_specification

    from marl_battlegrounds.evaluation.replay_v4 import build_replay_v4

    scenario = DebuggerScenario(
        name=name,
        title=title,
        description=description,
        mode="scripted",
        build_scenario=lambda: (config, initial),
        frames=(ScenarioFrame("one-tick", frame_description, commands),),
        default_controlled_slot=0,
    )
    revision = debugger_test_launch_specification().code_revision
    session = create_session(
        scenario,
        seed=0,
        evaluation_launch_specification=build_debugger_evaluation_launch_specification_v1(
            root_seed=0,
            code_revision=revision,
            capture_profile="evaluation_metric_complete",
        ),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        execution_information_mode="shared_obs",
    )
    advanced = submit_next_script_frame(session)
    view = advanced.incoming_evaluation_view
    if view is None:
        raise RuntimeError(f"{name} produced no authoritative transition")
    deaths = tuple(
        event.recipient_global_slot
        for event in view.transition.events
        if event.event_type == "agent_died"
    )
    scores = cast(
        tuple[int, int],
        tuple(int(value) for value in advanced.state.team_deathmatch_scores),
    )
    replay = build_replay_v4(
        session.evaluation_context,
        (session.current_evaluation_frame, advanced.current_evaluation_frame),
        (view.transition,),
        runtime_provenance=_runtime_provenance().model_copy(
            update={"package_version": revision.package_version}
        ),
    )
    return replay, deaths, scores


def export_red_zone_artifacts(output_directory: Path) -> dict[str, str]:
    from scripts.dev.visual_debugger.model import ActorCommand

    from marl_battlegrounds.core.types import MOVE_STAY
    from marl_battlegrounds.evaluation.replay_io import save_replay

    output_directory.mkdir(parents=True, exist_ok=True)
    config = _all_mage_config(red_zone_depth=5.0)
    state, _, _, _ = reset(config, jax.random.key(0))
    # Depth 5 on a 20-wide map: Team A's Red Zone is x <= 5 and Team B's is
    # x >= 15. Agents 5 and 6 die inside Team B's; agent 3 dies at x = 9.
    positions = {
        0: (13.0, 1.5),
        5: (16.0, 1.5),
        1: (13.0, 4.0),
        6: (16.0, 4.0),
        8: (12.0, 8.0),
        3: (9.0, 8.0),
        2: (6.0, 10.5),
        4: (2.0, 10.5),
        7: (18.0, 10.5),
        9: (18.5, 6.5),
    }
    placed = state.agent_positions
    for slot, xy in positions.items():
        placed = placed.at[slot].set(xy)
    health = state.current_health
    for slot in (3, 5, 6):
        health = health.at[slot].set(1.0)
    initial = state._replace(agent_positions=placed, current_health=health)
    replay, deaths, scores = _one_step_devclient_replay(
        config,
        initial,
        (
            ActorCommand(0, MOVE_STAY, 5, 0),
            ActorCommand(1, MOVE_STAY, 6, 0),
            ActorCommand(8, MOVE_STAY, 3, 0),
        ),
        name="browser_red_zone_metrics",
        title="Red Zone kills and one ordinary kill",
        description="Test-only Red Zone metric witness, not an official scenario.",
        frame_description=(
            "Two Team A Mages kill inside Team B's Red Zone; a Team B Mage kills "
            "outside Team A's Red Zone."
        ),
    )
    if deaths != (3, 5, 6) or scores != (4, 1):
        raise RuntimeError(
            f"Red Zone witness needs deaths (3, 5, 6) and scores (4, 1); got "
            f"{deaths} and {scores}"
        )
    path = output_directory / f"red-zone{REPLAY_FILE_SUFFIX_V1}"
    save_replay(replay, path)
    return {"red_zone": str(path.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--c7-acceptance", action="store_true")
    choice.add_argument("--red-zone", action="store_true")
    args = parser.parse_args()
    exporter = (
        export_acceptance_artifacts
        if args.c7_acceptance
        else export_red_zone_artifacts
        if args.red_zone
        else export_artifacts
    )
    print(json.dumps(exporter(args.output_directory), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
