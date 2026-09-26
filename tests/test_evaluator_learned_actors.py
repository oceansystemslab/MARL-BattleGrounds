"""Check learned actor memory, keys and replays through ordinary evaluation.

The six built-in learners use their real actor hooks with distinct game seeds
and different numerical weights. Batched runs and one-lane references preserve
recurrent memory within 1e-6 and captured actions/frames exactly on CPU. Writer
results agree with in-memory results. These checks make no learning or GPU
speed claim and do not use tournament workers, retries or grouped kernels.
"""

# pyright: reportPrivateUsage=false
import json
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds.baselines.ppo import make_ppo_system
from marl_battlegrounds.baselines.pqn import make_pqn_system
from marl_battlegrounds.baselines.qmix import make_qmix_system
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.policy_execution import System
from marl_battlegrounds.evaluation.replay_io import load_replay
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config
from marl_battlegrounds.training.checkpoints import _actor_template, _pqn_network

type Tree = Any
runner = import_module("marl_battlegrounds.evaluation.evaluate")


def _system(method: str, variables: Tree) -> System:
    if method == "qmix":
        return make_qmix_system(variables, actor_input_schema=2, spawn_frame="left")
    if method == "pqn_vdn":
        return make_pqn_system(variables, actor_input_schema=2, spawn_frame="left")
    return make_ppo_system(
        variables, method=method, actor_input_schema=2, spawn_frame="left"
    )


@lru_cache(maxsize=6)
def _weights(method: str) -> Tree:
    template = _actor_template(method)

    # Positive nonzero BatchNorm variances and weights change recurrent state.
    def fill(x: jax.ShapeDtypeStruct) -> jax.Array:
        return jnp.full(x.shape, 0.001, x.dtype)

    return jax.tree.map(fill, template)


@pytest.mark.parametrize(
    "method", ["mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"]
)
def test_builtin_batched_memory_keys_actions_and_saved_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str
) -> None:
    observed: dict[tuple[int, int], tuple[np.ndarray[Any, Any], ...]] = {}
    reference: dict[tuple[int, int], tuple[np.ndarray[Any, Any], ...]] = {}
    target = observed
    original = runner._jax_system_chunk

    def remember(*args: Tree, **kwargs: Tree) -> Tree:
        result = original(*args, **kwargs)
        carry = result[0]
        ids = np.asarray(carry.state.episode_id)
        steps = np.asarray(carry.state.core_state.step_count)
        memories = [
            np.asarray(x)
            for x in jax.tree.leaves((carry.memory.team_a, carry.memory.team_b))
        ]
        for lane, identifier in enumerate(ids):
            target[int(identifier), int(steps[lane])] = tuple(
                memory[lane].copy() for memory in memories
            )
        return result

    monkeypatch.setattr(runner, "_jax_system_chunk", remember)
    config = make_standard_team_deathmatch_config(
        map_id=12,
        team_a_roster=("priest",),
        team_b_roster=("mage",),
        max_steps=4,
    )
    specs = tuple(EpisodeSpec(11 + i, config, seed_id=1000 + i) for i in range(4))
    first = _weights(method)

    def scaled(value: jax.Array) -> jax.Array:
        return value * 1.1

    second = jax.tree.map(scaled, first)
    if method == "pqn_vdn":
        first, second = _pqn_network(first), _pqn_network(second)
    a, b = _system(method, first), _system(method, second)
    options: dict[str, Any] = {
        "seed": 1729,
        "chunk_size": 1,
        "metrics": "priority",
        "replay_episodes": tuple(spec.episode_id for spec in specs),
    }
    result = evaluate_episodes(a, b, specs, num_envs=2, output_dir=tmp_path, **options)
    target = reference
    expected = evaluate_episodes(
        a, b, specs, num_envs=1, run_id=result.metadata["run_id"], **options
    )
    assert result.episodes == expected.episodes
    assert result.completed_episode_ids == (11, 12, 13, 14)
    assert observed.keys() == reference.keys()
    assert len(observed) == 16
    maximum = 0.0
    for key, memories in observed.items():
        for actual, wanted in zip(memories, reference[key], strict=True):
            np.testing.assert_allclose(actual, wanted, rtol=1e-6, atol=1e-6)
            maximum = max(maximum, float(np.max(np.abs(actual - wanted))))
    recurrent = method not in {"ff_mappo", "ff_ippo"}
    if recurrent:
        assert any(np.any(x != 0) for row in observed.values() for x in row)
    else:
        assert all(not row for row in observed.values())
    assert result.paths is not None
    assert result.priority_metrics == {} and result.replays == ()
    wanted_replays = {
        replay.header.context.identity.episode_id: replay for replay in expected.replays
    }
    assert len(result.replay_paths) == len(wanted_replays) == 4
    for path in result.replay_paths:
        replay = load_replay(path).replay
        before = wanted_replays[replay.header.context.identity.episode_id]
        assert replay.frames == before.frames
        assert replay.transitions == before.transitions
    (tmp_path / "memory-comparison.json").write_text(
        json.dumps(
            {
                "method": method,
                "game_step_pairs": len(observed),
                "maximum_absolute_difference": maximum,
                "relative_tolerance": 1e-6,
                "absolute_tolerance": 1e-6,
                "memory_nonzero": recurrent,
            },
            indent=2,
        )
        + "\n"
    )
