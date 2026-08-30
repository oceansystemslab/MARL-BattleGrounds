"""Export deterministic canonical replay artifacts for browser integration tests.

This module is test infrastructure rather than a production launcher.  The
Playwright suite invokes it in a fresh temporary directory, then launches the
real replay viewer against the emitted canonical artifact files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import jax
from tests.evaluation_fixtures import (
    captured_evaluation_trajectory,
    evaluation_env_config,
    mage_target_none_ultimate_action,
    neutral_action,
    valid_shared_availability,
)

from marl_battlegrounds.core.env import initialize_scenario_state, reset, step
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

type _CompletionState = Literal["complete", "partial"]
type _InformationMode = Literal["no_shared_obs", "shared_obs"]


def _runtime_provenance() -> RuntimeProvenanceV1:
    """Return stable provenance for the deterministic browser-test artifacts."""
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
    """Build one artifact only through public capture, observer, and replay APIs."""
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
    """Build one partial replay with local and out-of-range corpse candidates."""
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
    """Write complete, partial, and missing-sidecar variants and return paths."""
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

    complete_path = output_directory / f"complete{REPLAY_FILE_SUFFIX_V1}"
    partial_path = output_directory / f"partial{REPLAY_FILE_SUFFIX_V1}"
    shared_path = output_directory / f"shared{REPLAY_FILE_SUFFIX_V1}"
    corpse_no_shared_path = (
        output_directory / f"corpse-no-shared{REPLAY_FILE_SUFFIX_V1}"
    )
    corpse_shared_path = output_directory / f"corpse-shared{REPLAY_FILE_SUFFIX_V1}"
    save_replay_bundle_v1(complete, complete_path)
    save_replay_bundle_v1(partial, partial_path)
    save_replay_bundle_v1(shared, shared_path)
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
        "corpse_no_shared": str(corpse_no_shared_path.resolve()),
        "corpse_shared": str(corpse_shared_path.resolve()),
        "missing_metric": str(missing_metric_path.resolve()),
    }


def main() -> int:
    """CLI used by Playwright support to create isolated test inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_artifacts(args.output_directory), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
