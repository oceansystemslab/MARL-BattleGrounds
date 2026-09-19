"""Supply compact CPU helpers for MAPPO integration and continuation proofs.

Synthetic banks change only declared test horizons after real content admission;
they are never represented as approved training content or passed to restore
validation. Comparisons retain every numerical leaf, including typed key bits.
"""

from collections.abc import Callable
from functools import lru_cache, partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.episode_tracking import init_episode_tracking
from marl_battlegrounds.evaluation.policy_execution import init_systems
from marl_battlegrounds.training.collection import (
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    scan_training_rollout,
)
from marl_battlegrounds.training.distributions import (
    sample_training_configs,
    training_keys,
)
from marl_battlegrounds.training.learner import (
    LearnerState,
    UpdateResult,
    update_learner,
)

type Tree = Any


def equal(left: Tree, right: Tree, *, close: bool = False) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        if close and jnp.issubdtype(a.dtype, jnp.inexact):
            np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6)
        else:
            np.testing.assert_array_equal(a, b)


def finite(tree: Tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert np.isfinite(np.asarray(leaf)).all()


@lru_cache
def scanner(
    collection: TrainingCollection, length: int
) -> Callable[[TrainingCarry], tuple[TrainingCarry, TrainingRollout]]:
    return cast(
        Callable[[TrainingCarry], tuple[TrainingCarry, TrainingRollout]],
        jax.jit(partial(scan_training_rollout, collection, length=length)),
    )


@lru_cache
def updater(
    ppo: PPOConfig,
) -> Callable[
    [LearnerState, TrainingCarry, TrainingRollout], tuple[LearnerState, UpdateResult]
]:
    return cast(
        Callable[
            [LearnerState, TrainingCarry, TrainingRollout],
            tuple[LearnerState, UpdateResult],
        ],
        jax.jit(partial(update_learner, ppo=ppo)),
    )


def synthetic_horizons(
    collection: TrainingCollection,
    learner: LearnerState,
    *,
    horizon: int = 3,
    first_horizon: int | None = None,
) -> LearnerState:
    carry = learner.carry
    count = collection.schedule.num_envs
    assert carry.tracking.source_configs is not None
    limits = jnp.full(42, horizon, jnp.int32)
    if first_horizon is not None:
        assert np.unique(np.asarray(carry.source_indices)).size > 1
        limits = limits.at[carry.source_indices[0]].set(first_horizon)
    bank = carry.tracking.source_configs._replace(max_steps=limits)
    generations = jnp.zeros(count, jnp.int32)
    sampled = sample_training_configs(
        bank,
        carry.root_key,
        generations,
        eligible_maps=carry.schedule.eligible_maps[0],
        team_size=carry.schedule.team_sizes[0],
    )
    env = marl_bgs.make(
        "tdm",
        env_config=sampled.config,
        num_envs=count,
        metrics=cast(Any, collection.metrics),
    )
    observations, state = env.reset(
        training_keys(carry.root_key, generations, stream="reset")
    )
    memory = init_systems(
        collection.actor,
        collection.opponent,
        observations,
        state,
        training_keys(carry.root_key, generations, stream="initialization"),
        variables_a=carry.history.current_variables,
        variables_b=carry.history,
    )
    tracking = init_episode_tracking(
        env,
        state,
        source_configs=bank,
        source_indices=sampled.source_indices,
        source_class_ids=sampled.source_class_ids,
        record_starts=collection.recording,
    ).begin_stage(state, total_env_steps=int(carry.schedule.round_budgets[0]) * count)
    return learner._replace(
        carry=carry._replace(
            env=env,
            observations=observations,
            state=state,
            memory=memory,
            tracking=tracking,
            source_indices=sampled.source_indices,
            source_class_ids=sampled.source_class_ids,
        )
    )


def stack(*values: Array) -> Array:
    return jnp.stack(values)
