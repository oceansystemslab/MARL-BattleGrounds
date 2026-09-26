"""Check fixed-batch evaluator placement, lane keys and publication.

These CPU checks retain the useful invariants from the retired capacity and
batching tests. Kernels keep committed inputs and reuse compiled programs when
only weights change. Refilling lanes preserves per-game actions and replays.
Backend errors propagate without retry, including errors after earlier games
were saved. Researchers own model memory and provider processes.
"""

# pyright: reportPrivateUsage=false
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.policy_execution import Policy, System, shared_policy
from marl_battlegrounds.evaluation.replay_io import load_replay
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

type Tree = Any
runner = import_module("marl_battlegrounds.evaluation.evaluate")


def _act(
    variables: jax.Array,
    carry: jax.Array,
    actor: ActorInput,
    mask: ActionMask,
    key: jax.Array,
) -> tuple[ActorAction, jax.Array]:
    key = jax.random.fold_in(key, carry + variables)
    return random_policy(actor.observation, mask, key), carry + 1


def _policy() -> Policy:
    return Policy("remember", _act, jnp.int32(7), jnp.int32(0))


def _specs() -> tuple[EpisodeSpec, ...]:
    return tuple(
        EpisodeSpec(
            11 + i,
            make_standard_team_deathmatch_config(
                map_id=12,
                team_a_roster=("priest",),
                team_b_roster=("mage",),
                max_steps=horizon,
            ),
            seed_id=300 + i,
        )
        for i, horizon in enumerate((1, 3, 2, 1, 2, 3))
    )


def _run(method: Policy | System, **options: Tree) -> Tree:
    return evaluate_episodes(
        method,
        method,
        _specs(),
        seed=1729,
        chunk_size=1,
        metrics="priority",
        replay_episodes=(11, 12, 13, 14, 15, 16),
        **options,
    )


def test_setup_and_refill_keep_committed_inputs_and_reuse_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runner._jax_chunk
    signatures: list[tuple[tuple[object, ...], ...]] = []
    seed_buffers: dict[tuple[int, ...], list[jax.Array]] = {}
    device = cast(Tree, jax.local_devices()[0])

    def observe(*args: Tree, **kwargs: Tree) -> Tree:
        arrays = jax.tree.leaves((args[0], args[3:8]))
        seed_buffers.setdefault(
            tuple(cast(list[int], np.asarray(args[7], dtype=np.uint32).tolist())), []
        ).append(args[7])
        signatures.append(
            tuple((x.shape, x.dtype, x.committed, str(x.sharding)) for x in arrays)
        )
        assert all(x.committed and x.devices() == {device} for x in arrays)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_jax_chunk", observe)
    _run(_policy(), num_envs=2)
    assert len(set(signatures)) == 1
    assert any(len(values) > 1 for values in seed_buffers.values())
    assert all(
        all(value is values[0] for value in values) for values in seed_buffers.values()
    )
    compiled = original._cache_size()
    _run(Policy("changed", _act, jnp.int32(8), jnp.int32(0)), num_envs=2)
    assert original._cache_size() == compiled


@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.parametrize("batch", [2, 3])
def test_refill_and_saved_replays_match_single_lane(
    tmp_path: Path, adapter: bool, batch: int
) -> None:
    method = shared_policy(_policy()) if adapter else _policy()
    result = _run(method, num_envs=batch, output_dir=tmp_path)
    reference = _run(method, num_envs=1, run_id=result.metadata["run_id"])
    assert result.episodes == reference.episodes
    assert result.completed_episode_ids == (11, 12, 13, 14, 15, 16)
    assert result.priority_metrics == {} and result.replays == ()
    expected = {
        replay.header.context.identity.episode_id: replay
        for replay in reference.replays
    }
    assert len(result.replay_paths) == len(expected) == 6
    for path in result.replay_paths:
        actual = load_replay(path).replay
        before = expected[actual.header.context.identity.episode_id]
        assert actual.frames == before.frames
        assert actual.transitions == before.transitions


@pytest.mark.parametrize("adapter", [False, True])
def test_backend_allocation_errors_propagate_without_retry(
    monkeypatch: pytest.MonkeyPatch, adapter: bool
) -> None:
    calls: list[int] = []
    error_type = cast(type[RuntimeError], jax.errors.JaxRuntimeError)
    error = error_type("RESOURCE_EXHAUSTED: allocation failed")

    def fail(*args: Tree, **kwargs: Tree) -> Tree:
        calls.append(args[0].num_envs)
        raise error

    monkeypatch.setattr(runner, "_jax_system_chunk" if adapter else "_jax_chunk", fail)
    method = shared_policy(_policy()) if adapter else _policy()
    with pytest.raises(error_type) as caught:
        _run(method, num_envs=2)
    assert caught.value is error
    assert calls == [2]


def test_existing_device_placement_and_shards_are_preserved(tmp_path: Path) -> None:
    script = tmp_path / "devices.py"
    script.write_text(
        """
from dataclasses import replace
from importlib import import_module
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.policy_execution import policy
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config
E = import_module('marl_battlegrounds.evaluation.evaluate')
devices = jax.local_devices()
assert len(devices) == 2
mesh = Mesh(np.array(devices), ('d',))
weights = jax.device_put(jnp.zeros(2), NamedSharding(mesh, PartitionSpec('d')))
method = replace(policy('random'), variables=weights)
config = make_standard_team_deathmatch_config(
    map_id=12, max_steps=1, team_a_roster=('mage',), team_b_roster=('mage',),
)
specs = (EpisodeSpec(1, config), EpisodeSpec(2, config))
original = E._jax_chunk
seen = []
def chunk(*args, **kwargs):
    seen.append(args[3].devices())
    return original(*args, **kwargs)
E._jax_chunk = chunk
local = replace(method, variables=jax.device_put(jnp.zeros(2), devices[1]))
result = evaluate_episodes(
    local, local, specs, num_envs=2, chunk_size=1, replay_episodes=(1,),
)
assert seen == [{devices[1]}]
assert result.metadata['execution_layout']['device'] == str(devices[1])
assert result.metadata['runtime_provenance']['backend'] == devices[1].platform
assert result.metadata['runtime_provenance']['device'] == devices[1].device_kind
assert weights.devices() == set(devices)
seen.clear()
result = evaluate_episodes(
    method, method, specs, num_envs=2, chunk_size=1, metrics='none',
)
assert seen == [set(devices)]
assert result.metadata['execution_layout']['device'] == 'Caller context'
assert result.metadata['execution_layout']['observed_numerical_devices'] == sorted(
    str(d) for d in devices
)
"""
    )
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_completed_saved_pass_resume_skips_numerical_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _run(_policy(), num_envs=2, output_dir=tmp_path)

    def forbidden(*args: Tree, **kwargs: Tree) -> Tree:
        raise AssertionError("A completed saved pass must not run another game")

    monkeypatch.setattr(runner, "_reset", forbidden)
    resumed = _run(_policy(), num_envs=2, resume_from=first.run_dir)
    assert resumed.episodes == ()
    assert resumed.completed_episode_ids == first.completed_episode_ids
    assert resumed.priority_metrics == {} and resumed.replays == ()
    for name in ("episodes", "priority_metrics"):
        actual, expected = resumed.table(name), first.table(name)
        assert actual.keys() == expected.keys()
        for column in actual:
            np.testing.assert_array_equal(actual[column], expected[column])


@pytest.mark.parametrize("failure_site", ["writer", "host"])
def test_writer_and_host_failures_do_not_repeat_policy_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_site: str
) -> None:
    calls: list[int] = []
    error_type = cast(type[RuntimeError], jax.errors.JaxRuntimeError)
    error = error_type("RESOURCE_EXHAUSTED: external allocation failed")

    def fail(*args: Tree, **kwargs: Tree) -> Tree:
        raise error

    if failure_site == "writer":
        original = runner._jax_system_chunk

        def observe(*args: Tree, **kwargs: Tree) -> Tree:
            calls.append(args[0].num_envs)
            return original(*args, **kwargs)

        monkeypatch.setattr(runner, "_jax_system_chunk", observe)
        monkeypatch.setattr(runner.RunWriter, "_write_collected", fail)
        method = shared_policy(_policy())
    else:

        def fail_host(*args: Tree, **kwargs: Tree) -> Tree:
            calls.append(args[0].num_envs)
            raise error

        monkeypatch.setattr(runner, "_host_chunk", fail_host)
        method = Policy("host", _act, jnp.int32(7), jnp.int32(0), execution="host")
    with pytest.raises(error_type) as caught:
        _run(method, num_envs=2, output_dir=tmp_path)
    assert caught.value is error
    assert calls == [2]
