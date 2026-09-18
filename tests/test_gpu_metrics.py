"""Check selected metric capture through 32-lane compiled GPU workflows.

The same tests run on CPU in the complete suite. Every environment reset and
step uses 32 lanes, including the full-capture reference. These checks cover
chunk continuation, exact metric values, partial reset and capture reassignment;
they do not measure throughput or learning.
"""

from collections.abc import Callable
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds import EnvironmentState, make
from marl_battlegrounds.core.types import Action, DoneFlags, Reward
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation.metric_catalog import PRIORITY_METRIC_NAMES
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

type StepResult = tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]
type Step = Callable[[Array, EnvironmentState, Action], StepResult]

_BATCH = 32


def _assert_tree_exact(
    actual: object, expected: object, *, rows: int | slice = slice(None)
) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        a, b = np.asarray(left)[rows], np.asarray(right)[rows]
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


def _first[T](tree: T) -> T:
    def first(leaf: Array) -> Array:
        return leaf[0]

    return jax.tree.map(first, tree)


@pytest.mark.parametrize("mode", ("none", "priority"))
def test_selected_metrics_match_full_across_chunks_and_resets(
    mode: Literal["none", "priority"],
) -> None:
    config = make_standard_team_deathmatch_config(
        map_id=12,
        team_a_roster=("mage", "priest"),
        team_b_roster=("mage",),
        max_steps=2,
    )
    replacement = make_standard_team_deathmatch_config(
        map_id=13,
        team_a_roster=("priest",),
        team_b_roster=("hunter", "mage", "mage"),
        max_steps=1,
    )
    ids = jnp.arange(1, _BATCH + 1, dtype=jnp.int32)
    reset_key = jax.random.key(19)
    step_keys = jax.random.split(jax.random.key(20), 2)
    full = make("tdm", num_envs=_BATCH, metrics="full")
    selected = make("tdm", num_envs=_BATCH, metrics=mode, full_metrics_episodes=(2, 33))
    _, full_state = full.reset(reset_key, config, episode_id=ids)
    _, selected_state = selected.reset(reset_key, config, episode_id=ids)
    assert selected_state.core_state.step_count.shape == (_BATCH,)
    assert selected_state.full is not None
    initial_full = selected_state.full
    zeros = jnp.zeros((_BATCH, 10), jnp.int32)
    actions = Action(jnp.ones_like(zeros), zeros, zeros)
    full_step = cast(Step, jax.jit(full.step))

    def advance(
        state: EnvironmentState, key: Array
    ) -> tuple[EnvironmentState, EpisodeInfo]:
        _, successor, _, _, info = selected.step(key, state, actions)
        return successor, info

    def scan_chunk(
        state: EnvironmentState, keys: Array
    ) -> tuple[EnvironmentState, EpisodeInfo]:
        return jax.lax.scan(advance, state, keys)

    scan = cast(
        Callable[[EnvironmentState, Array], tuple[EnvironmentState, EpisodeInfo]],
        jax.jit(scan_chunk),
    )
    length_column = PRIORITY_METRIC_NAMES.index("episode_length")
    selected_lanes = np.arange(_BATCH) == 1
    for tick in range(2):
        selected_state, history = scan(selected_state, step_keys[tick : tick + 1])
        info = _first(history)
        _, full_state, _, _, full_info = full_step(step_keys[tick], full_state, actions)
        np.testing.assert_array_equal(info.completed, np.full(_BATCH, tick == 1))
        np.testing.assert_array_equal(info.episode_id, np.arange(1, _BATCH + 1))
        _assert_tree_exact(selected_state.core_state, full_state.core_state)
        assert info.full is not None and info.priority is not None
        assert full_info.full is not None and full_info.priority is not None
        _assert_tree_exact(info.full, full_info.full, rows=1)
        _assert_tree_exact(info.priority, full_info.priority, rows=1)
        assert int(info.priority.values[1, length_column]) == tick + 1
        np.testing.assert_array_equal(
            info.full.valid.any(axis=-1), selected_lanes & (tick == 1)
        )
        np.testing.assert_array_equal(
            info.priority.valid.any(axis=-1),
            np.ones(_BATCH, bool) if mode == "priority" else selected_lanes,
        )
        assert selected_state.full is not None
        _assert_tree_exact(selected_state.full, initial_full, rows=0)
        _assert_tree_exact(selected_state.full, initial_full, rows=slice(2, None))

    # Select ID33 in lane 0, then remove selected ID2 by resetting lane 1 to ID34.
    for lane, identifier in ((0, 33), (1, 34)):
        before = selected_state
        ids = ids.at[lane].set(identifier)
        reset_mask = jnp.arange(_BATCH) == lane
        _, selected_state = selected.reset(
            reset_key,
            replacement,
            episode_id=ids,
            state=selected_state,
            reset_mask=reset_mask,
        )
        _, full_state = full.reset(
            reset_key,
            replacement,
            episode_id=ids,
            state=full_state,
            reset_mask=reset_mask,
        )
        _assert_tree_exact(selected_state, before, rows=1 - lane)
        _assert_tree_exact(selected_state, before, rows=slice(2, None))
        _assert_tree_exact(selected_state.core_state, full_state.core_state)

    np.testing.assert_array_equal(
        selected_state.collect_full_metrics, np.arange(_BATCH) == 0
    )
    assert selected_state.full is not None
    deselected_full = selected_state.full
    selected_state, history = scan(selected_state, step_keys[:1])
    info = _first(history)
    _, full_state, _, _, full_info = full_step(step_keys[0], full_state, actions)
    assert info.full is not None and info.priority is not None
    assert full_info.full is not None and full_info.priority is not None
    np.testing.assert_array_equal(info.completed, np.arange(_BATCH) < 2)
    np.testing.assert_array_equal(info.episode_id, np.asarray(ids))
    _assert_tree_exact(selected_state.core_state, full_state.core_state)
    _assert_tree_exact(info.full, full_info.full, rows=0)
    _assert_tree_exact(info.priority, full_info.priority, rows=0)
    assert int(info.priority.values[0, length_column]) == 1
    np.testing.assert_array_equal(info.full.valid.any(axis=-1), np.arange(_BATCH) == 0)
    assert selected_state.full is not None
    _assert_tree_exact(selected_state.full, deselected_full, rows=slice(1, None))
    assert not bool(selected_state.lifecycle_error.any())
