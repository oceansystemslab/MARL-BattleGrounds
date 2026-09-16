"""Check required episode summaries and failures before optional recording.

These tests cover local action epochs, authored scores, terminal padding,
chunk error retention and schedule checks before saved files can be recovered.
"""

import csv
import json
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds import make
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import Action, DoneFlags, EnvConfig, Reward
from marl_battlegrounds.environment import EnvironmentState, EpisodeInfo
from marl_battlegrounds.evaluation.evaluate import (
    EpisodeSpec,
    _Carry,  # pyright: ignore[reportPrivateUsage]
    evaluate_episodes,
)
from marl_battlegrounds.evaluation.policy_execution import policy
from marl_battlegrounds.evaluation.recording_context import capture_recording_provenance
from marl_battlegrounds.evaluation.recording_types import validate_recording_errors
from marl_battlegrounds.evaluation.replay_capture import ReplayPackets
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

type StepResult = tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]
type Step = Callable[[Array, EnvironmentState, Action], StepResult]


def _config(max_steps: int = 7) -> EnvConfig:
    return make_standard_team_deathmatch_config(
        map_id=12,
        team_a_roster=("priest",),
        team_b_roster=("priest",),
        max_steps=max_steps,
    )


@pytest.mark.parametrize("mode", ("none", "priority", "full"))
def test_required_summaries_track_real_decisions_from_authored_start(
    mode: Literal["none", "priority", "full"],
) -> None:
    config = _config()
    initial, _, _, _ = core.reset(config, jax.random.key(0))
    initial = initial._replace(
        step_count=jnp.int32(5), team_deathmatch_scores=jnp.asarray([2, 3], jnp.int32)
    )
    env = make("tdm", env_config=config, metrics=mode)
    initial_state, observation, mask, _ = core.initialize_scenario_state(
        initial, config
    )
    _, state = env.reset(jax.random.key(1), initial=(initial_state, observation, mask))
    action = Action(*(jnp.zeros(10, jnp.int32) for _ in range(3)))
    step = cast(Step, jax.jit(env.step))
    for index in range(3):
        _, state, _, _, info = step(jax.random.key(index + 2), state, action)
        assert int(info.decision_step) == (index if index < 2 else -1)
        assert int(info.episode_length) == min(index + 1, 2)
        np.testing.assert_array_equal(info.team_scores, [2, 3])
        assert bool(info.completed) == (index == 1)
        assert not bool(info.lifecycle_error)
        assert info.episode_start_records is info.episode_tracking_error is None
        assert (info.priority is None) == (mode == "none")


def test_unfinished_and_padded_rounds_keep_lifecycle_failures() -> None:
    execution = import_module("marl_battlegrounds.evaluation.evaluate")
    env = make("tdm", num_envs=2, metrics="none")
    _, state = env.reset(jax.random.key(0), _config(), episode_id=jnp.asarray([1, 2]))
    state = state._replace(lifecycle_error=jnp.asarray([True, False]))
    empty = execution._empty_completed(state)
    np.testing.assert_array_equal(empty.info.lifecycle_error, [True, False])
    zeros = jnp.zeros(2, jnp.int32)
    info = EpisodeInfo(
        state.episode_id,
        jnp.asarray([False, True]),
        jnp.asarray([0, 3], jnp.int32),
        state.config,
        None,
        None,
        None,
        jnp.asarray([0, 0], jnp.int32),
        jnp.ones(2, jnp.int32),
        jnp.zeros((2, 2), jnp.int32),
        jnp.asarray([False, True]),
    )
    retained = execution._retain_completion(empty, state, info)
    padded = info._replace(
        completed=jnp.zeros(2, bool),
        decision_step=jnp.full(2, -1, jnp.int32),
        episode_length=zeros,
        lifecycle_error=jnp.zeros(2, bool),
    )
    retained = execution._retain_completion(retained, state, padded)
    np.testing.assert_array_equal(retained.info.lifecycle_error, [True, True])
    np.testing.assert_array_equal(retained.info.decision_step, [-1, 0])
    np.testing.assert_array_equal(retained.info.episode_length, [0, 1])
    with pytest.raises(ValueError, match="lifecycle_error"):
        validate_recording_errors(retained.info)


def test_counter_overflow_reaches_info_without_waiting_for_completion() -> None:
    env = make("tdm", env_config=_config(), metrics="none")
    _, state = env.reset(jax.random.key(0))
    state = state._replace(cumulative_transition_count=jnp.int32(2**31 - 1))
    action = Action(*(jnp.zeros(10, jnp.int32) for _ in range(3)))
    _, successor, _, _, info = cast(Step, jax.jit(env.step))(
        jax.random.key(1), state, action
    )
    assert bool(info.lifecycle_error) and bool(successor.lifecycle_error)
    assert not bool(info.completed)
    assert int(info.decision_step) == 0
    assert int(info.episode_length) == 1
    with pytest.raises(ValueError, match="lifecycle_error"):
        validate_recording_errors(info)


@pytest.mark.parametrize("saved", (False, True))
def test_lifecycle_failure_blocks_evaluator_replay_consumption(
    saved: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = import_module("marl_battlegrounds.evaluation.evaluate")
    chunk = cast(
        Callable[..., tuple[_Carry, ReplayPackets | None]], execution._jax_chunk
    )

    def fail_chunk(*args: object, **kwargs: object) -> object:
        carry, packets = chunk(*args, **kwargs)
        completed = carry.completed._replace(lifecycle_error=jnp.ones(1, bool))
        return carry._replace(completed=completed), packets

    def consume(*args: object, **kwargs: object) -> None:
        pytest.fail("replay consumption happened before lifecycle validation")

    monkeypatch.setattr(execution, "_jax_chunk", fail_chunk)
    monkeypatch.setattr(execution.ReplayCollector, "write", consume)
    with pytest.raises(ValueError, match="lifecycle_error"):
        evaluate_episodes(
            policy("random"),
            policy("random"),
            [EpisodeSpec(1, _config(1))],
            metrics="none",
            replay_episodes=[1],
            output_dir=tmp_path if saved else None,
            chunk_size=1,
        )
    assert not list(tmp_path.rglob("*.csv"))
    assert not list(tmp_path.rglob("*.marlbg-replay.json"))


def test_saved_none_summaries_and_capture_coverage_are_separate(tmp_path: Path) -> None:
    result = evaluate_episodes(
        policy("random"),
        policy("random"),
        [EpisodeSpec(1, _config(1)), EpisodeSpec(2, _config(2))],
        num_envs=2,
        metrics="none",
        full_metrics_episodes=[2],
        output_dir=tmp_path,
        chunk_size=3,
    )
    assert result.paths is not None
    with result.paths["episodes"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {int(row["episode_id"]): row for row in rows}
    for episode in result.episodes:
        row = by_id[episode.episode_id]
        assert int(row["episode_length"]) == episode.episode_length
        assert int(row["outcome"]) == episode.outcome
        assert int(row["team_a_score"]) == episode.team_a_score
        assert int(row["team_b_score"]) == episode.team_b_score
    manifest = json.loads(result.paths["run_details"].read_text())
    entry = next(iter(manifest["passes"].values()))
    assert entry["details"]["metrics"] == "none"
    assert entry["recorded_metrics_by_episode"] == {"1": "none", "2": "full"}


@pytest.mark.parametrize("change", ("config", "start", "order", "run_id"))
def test_resume_mismatch_does_not_recover_an_interrupted_suffix(
    change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance()

    def same_source(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(execution, "capture_recording_provenance", same_source)
    specs = [EpisodeSpec(1, _config(1)), EpisodeSpec(2, _config(1))]
    result = evaluate_episodes(
        policy("random"), policy("random"), specs, output_dir=tmp_path, metrics="none"
    )
    assert result.paths is not None
    run = result.paths["run_details"].parent
    with result.paths["episodes"].open("ab") as stream:
        stream.write(b"interrupted suffix\n")
    if change == "config":
        specs[0] = EpisodeSpec(1, _config(2))
    elif change == "start":
        config = _config(1)
        initial, _, _, _ = core.reset(config, jax.random.key(0))
        specs[0] = EpisodeSpec(1, config, initial_state=initial)
    elif change == "order":
        specs.reverse()
    before = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match=r"identity|schedule|run_id"):
        evaluate_episodes(
            policy("random"),
            policy("random"),
            specs,
            resume_from=run,
            metrics="none",
            run_id="wrong-run" if change == "run_id" else None,
        )
    assert {
        path.name: path.read_bytes() for path in run.iterdir() if path.is_file()
    } == before
