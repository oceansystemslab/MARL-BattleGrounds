"""Check changed evaluation work without making CPU speed claims.

Public CPU workflows count setup identities, occupied full-metric transfers and
route capture. They prevent repeated setup hashing and a second dense completion
transfer, while keeping in-memory routing records out of the rollout buffer.
They also compare compiled row selection with the eager projection and check
that changing selected indices reuses the same compiled program.
"""

# These proofs inspect the shared host boundary without changing numerical work.
# pyright: reportPrivateUsage=false

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.core.types import TASK_MODE_TDM, EnvConfig
from marl_battlegrounds.evaluation import evaluation_conditions
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.evaluation_capture import _rows
from marl_battlegrounds.evaluation.metric_catalog import FULL_METRIC_NAMES
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    policy,
    shared_policy,
)

runner = import_module("marl_battlegrounds.evaluation.evaluate")


@pytest.mark.parametrize("leading", (1, 2))
@pytest.mark.parametrize("selection", ("some", "all", "none"))
def test_recording_rows_preserve_values_and_reuse_changed_indices(
    leading: int, selection: str
) -> None:
    tree = {
        "values": jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
        "mask": jnp.arange(6).reshape(2, 3) % 2 == 0,
        "missing": None,
    }
    count = 2 if leading == 1 else 6
    selected = (
        [count - 1, 0]
        if selection == "some"
        else list(range(count))
        if selection == "all"
        else []
    )
    indices = np.asarray(selected, dtype=np.int32)

    def expected(value: jax.Array) -> jax.Array:
        return value.reshape((-1, *value.shape[leading:]))[indices]

    actual = cast(dict[str, jax.Array | None], _rows(tree, indices, leading=leading))
    wanted = jax.tree.map(expected, tree)
    assert jax.tree.structure(actual) == jax.tree.structure(wanted)
    for first, second in zip(
        jax.tree.leaves(actual), jax.tree.leaves(wanted), strict=True
    ):
        assert first.dtype == second.dtype and first.shape == second.shape
        np.testing.assert_array_equal(first, second)
    cache_size = cast(Callable[[], int], _rows._cache_size)  # pyright: ignore[reportAttributeAccessIssue]
    compilations = cache_size()
    changed = cast(
        dict[str, jax.Array | None],
        _rows(tree, indices[::-1].copy(), leading=leading),
    )
    assert cache_size() == compilations
    for first, second in zip(
        jax.tree.leaves(changed), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(first, second[::-1])


def test_one_identity_per_shared_live_setup_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = evaluation_env_config(
        team_sizes=(1, 1),
        max_steps=1,
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=20,
    )
    seen: list[int] = []
    original = evaluation_conditions.config_record

    def record(value: EnvConfig) -> tuple[str, dict[str, object]]:
        seen.append(id(value))
        return original(value)

    def duplicate(*args: object, **kwargs: object) -> None:
        raise AssertionError("new contract repeated the legacy config hashing path")

    monkeypatch.setattr(evaluation_conditions, "config_record", record)
    monkeypatch.setattr(runner, "configuration_identity", duplicate)
    result = evaluate_episodes(
        "random",
        "random",
        [EpisodeSpec(1, configuration), EpisodeSpec(2, configuration)],
        num_envs=2,
        metrics="none",
        chunk_size=1,
    )
    assert result.status == "complete"
    assert seen == [id(configuration)]


def test_selected_full_transfers_once_and_memory_result_skips_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = evaluation_env_config(
        team_sizes=(1, 1),
        max_steps=1,
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=20,
    )
    specs = [EpisodeSpec(1, configuration), EpisodeSpec(2, configuration)]
    method = shared_policy(policy("random"))
    transfers: list[tuple[int, ...]] = []
    routes: list[bool] = []
    device_get = jax.device_get
    chunk = runner._jax_system_chunk

    def transferred[T](tree: T) -> T:
        for leaf in jax.tree.leaves(tree):
            if (
                isinstance(leaf, jax.Array)
                and leaf.dtype == np.float32
                and leaf.ndim
                and leaf.shape[-1] == len(FULL_METRIC_NAMES)
            ):
                transfers.append(leaf.shape)
        return cast(T, device_get(tree))

    def called(*args: PolicyTree, **kwargs: PolicyTree) -> PolicyTree:
        routes.append(kwargs["capture_routes"])
        return chunk(*args, **kwargs)

    monkeypatch.setattr(jax, "device_get", transferred)
    monkeypatch.setattr(runner, "_jax_system_chunk", called)
    result = evaluate_episodes(
        method,
        method,
        specs,
        num_envs=2,
        metrics="none",
        chunk_size=1,
        full_metrics_episodes=[1],
        replay_episodes=[1],
        output_dir=tmp_path,
    )
    assert result.status == "complete"
    assert result.table("full_metrics")["episode_id"].tolist() == [1]
    assert transfers == [(1, len(FULL_METRIC_NAMES))]
    assert routes == [True]
    routes.clear()
    evaluate_episodes(
        method,
        method,
        specs,
        num_envs=2,
        metrics="none",
        chunk_size=1,
        replay_episodes=[1],
    )
    assert routes == [False]
