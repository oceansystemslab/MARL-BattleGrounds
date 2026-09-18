"""Trace complete System evaluations through refill, records and scoped results.

CPU cases compare actual replay trajectories, fresh recurrent memory and action
randomness. They cover scalar/odd worker counts, mixed host/JAX execution, one
provider call per decision, provider failure before advancement, and compilation
reuse for same-shaped changed parameters. No learner or Core rule changes here.
"""

# Public workflow proofs also inspect their shared compiled execution boundary.
# pyright: reportPrivateUsage=false

from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.evaluation.evaluate import (
    EpisodeSpec,
    evaluate,
    evaluate_episodes,
)
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
    policy,
    shared_policy,
)
from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3
from marl_battlegrounds.evaluation.results import EvaluationResult
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

runner = import_module("marl_battlegrounds.evaluation.evaluate")


def _specs() -> tuple[EpisodeSpec, ...]:
    return tuple(
        EpisodeSpec(
            i + 1,
            make_standard_team_deathmatch_config(
                map_id=12,
                team_a_roster=("priest",),
                team_b_roster=("mage", "mage"),
                max_steps=horizon,
            ),
            seed_id=100 + i,
        )
        for i, horizon in enumerate((1, 3, 2, 1))
    )


def _init(variables: PolicyTree, inputs: SystemInput, keys: Array) -> Array:
    del variables, keys
    return jnp.zeros(inputs.valid.shape, jnp.int32)


def _act(
    variables: PolicyTree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    def actors(key: Array) -> Array:
        return jax.random.split(key, 5)

    action = jax.vmap(jax.vmap(random_policy))(
        inputs.actors.observation,
        inputs.action_mask,
        jax.vmap(actors)(keys),
    )
    prefer_idle = (memory[:, None] % 2 == 0) & (variables > 0)
    action = action._replace(
        move=jnp.where(
            prefer_idle & inputs.action_mask.move_mask[..., 0], 0, action.move
        )
    )
    return SystemOutput(
        action,
        memory + inputs.valid.astype(jnp.int32),
        learning_outputs=jnp.zeros((*inputs.active_mask.shape, 64)),
    )


def test_explicit_shared_adapter_retains_bare_policy_trajectories() -> None:
    method = policy("random")
    options: dict[str, Any] = dict(
        num_envs=2,
        chunk_size=2,
        metrics="none",
        replay_episodes=(1, 2, 3, 4),
        run_id="parity",
    )
    bare = evaluate_episodes(method, method, _specs(), **options)
    adapted = evaluate_episodes(
        shared_policy(method), shared_policy(method), _specs(), **options
    )
    first = {r.header.context.identity.episode_id: r for r in bare.replays}
    second = {r.header.context.identity.episode_id: r for r in adapted.replays}
    # Run identities differ; per-game replay content has the same ID suffix.
    for a, b in zip(first.values(), second.values(), strict=True):
        assert a.frames == b.frames
        assert a.transitions == b.transitions
    assert bare.episodes == adapted.episodes


def test_recurrent_system_refill_and_chunking_keep_games_identical() -> None:
    method = System("count", _act, variables=jnp.int32(1), init=_init)
    first = evaluate_episodes(
        method,
        method,
        _specs(),
        num_envs=1,
        chunk_size=1,
        replay_episodes=(1, 2, 3, 4),
        metrics="none",
        run_id="refill",
    )
    second = evaluate_episodes(
        method,
        method,
        tuple(reversed(_specs())),
        num_envs=3,
        chunk_size=2,
        replay_episodes=(1, 2, 3, 4),
        metrics="none",
        run_id="refill",
    )

    def indexed(result: EvaluationResult) -> dict[int, ReplayArtifactV3]:
        return {
            int(r.header.context.identity.episode_id.split("-")[-1]): r
            for r in result.replays
        }

    left, right = indexed(first), indexed(second)
    for i in left:
        assert left[i].frames == right[i].frames
        assert left[i].transitions == right[i].transitions
    assert first.episodes == second.episodes


@pytest.mark.parametrize("host_team", [0, 1])
def test_host_system_opaque_memory_and_batch_calls_survive_refill(
    host_team: int,
) -> None:
    observed: dict[tuple[int, ...], int] = {}
    batch_shapes: list[int] = []
    opaque = object()

    def initialize(
        variables: PolicyTree, inputs: SystemInput, keys: Array
    ) -> list[dict[str, object]]:
        del variables, keys
        return [{"count": 0, "session": opaque} for _ in inputs.valid]

    def host(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        del variables
        batch_shapes.append(len(inputs.valid))
        result = list(memory)
        for lane in np.flatnonzero(inputs.valid):
            assert memory[lane]["session"] is opaque
            identity = tuple(int(x) for x in jax.random.key_data(keys[lane]))
            assert identity not in observed
            observed[identity] = memory[lane]["count"]
            result[lane] = {"count": memory[lane]["count"] + 1, "session": opaque}
        zero = cast(Array, np.zeros(inputs.active_mask.shape, np.int32))
        return SystemOutput(
            ActorAction(zero, zero, zero), result, learning_outputs=opaque
        )

    provider = System("provider", host, init=initialize, execution="host")
    numerical = System("count", _act, variables=jnp.int32(0), init=_init)
    methods = (provider, numerical) if host_team == 0 else (numerical, provider)
    evaluate_episodes(*methods, _specs(), num_envs=1, chunk_size=2, metrics="none")
    reference = dict(observed)
    assert set(batch_shapes) == {1}
    observed.clear()
    batch_shapes.clear()
    evaluate_episodes(
        *methods, tuple(reversed(_specs())), num_envs=3, chunk_size=2, metrics="none"
    )
    assert observed == reference
    assert set(batch_shapes) == {3}
    assert len(batch_shapes) == 3


def test_host_failure_precedes_environment_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fail(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        del variables, memory, inputs, keys
        raise RuntimeError("provider stopped")

    def forbidden(*args: object, **kwargs: object) -> object:
        calls.append(1)
        raise AssertionError("environment advanced after provider failure")

    monkeypatch.setattr(runner, "_step_environment", forbidden)
    with pytest.raises(RuntimeError, match="provider stopped"):
        evaluate_episodes(
            System("failing", fail, execution="host"),
            "random",
            _specs(),
            num_envs=2,
            metrics="none",
        )
    assert not calls


def test_same_shape_checkpoint_values_reuse_system_chunk_program() -> None:
    method = System("counter", _act, variables=jnp.int32(0), init=_init)
    evaluate_episodes(
        method, method, _specs(), num_envs=2, metrics="none", chunk_size=1
    )
    count = runner._jax_system_chunk._cache_size()
    changed = replace(method, variables=jnp.int32(1))
    result = evaluate_episodes(
        changed, changed, _specs(), num_envs=2, metrics="none", chunk_size=1
    )
    assert runner._jax_system_chunk._cache_size() == count
    assert result.status == "complete"


def test_saved_system_pair_routes_and_replays_join_actual_episodes(
    tmp_path: Path,
) -> None:
    method = System("counter", _act, variables=jnp.int32(0), init=_init)
    result = evaluate(
        method,
        method,
        num_episodes=2,
        maps=[12],
        max_steps=2,
        num_envs=2,
        chunk_size=3,
        metrics="none",
        full_metrics_episodes=[1],
        save_replays=2,
        output_dir=tmp_path,
    )
    assert result.status == "complete"
    assert len(result.replay_paths) == 2
    assert len(result.table("episodes")["episode_id"]) == 2
    assert list(result.table("full_metrics")["episode_id"]) == [1]
    assert result.metadata["spawn_balance"]["paired_complete"] is True
