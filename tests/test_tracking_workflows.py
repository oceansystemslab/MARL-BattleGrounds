"""Check composed tracking, independent seed carries and validation isolation.

The compiled public workflow retains exact action data, resets recurrent memory,
keeps every seed's stage separate and restores complete numerical state. All
metric modes and selected captures retain their existing episode ownership.
"""

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.autoreset import AutoReset
from marl_battlegrounds.environment import EnvironmentState, MetricMode
from marl_battlegrounds.episode_tracking import EpisodeTrackingState
from marl_battlegrounds.evaluation.policy_execution import (
    SystemInput,
    SystemOutput,
    SystemState,
)
from marl_battlegrounds.types import ActorAction

type Carry = tuple[EnvironmentState, EpisodeTrackingState, SystemState, Array]
type Tree = Any


def _initialize(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    del variables, keys
    return jnp.zeros(inputs.active_mask.shape, jnp.int32)


def _act(
    variables: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    del variables, keys
    zeros = jnp.zeros(inputs.active_mask.shape, jnp.int32)
    return SystemOutput(
        ActorAction(zeros, zeros, zeros),
        memory + inputs.valid[:, None],
        learning_outputs=memory,
    )


def _equal(left: Tree, right: Tree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):  # pyright: ignore[reportPrivateImportUsage]
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)


def _setup(
    mode: MetricMode, seed: int, *, selected: bool = False
) -> tuple[AutoReset, Carry, Callable[[AutoReset, Carry], Carry]]:
    base = marl_bgs.make(
        "tdm",
        map_id=0,
        num_envs=2,
        max_steps=2,
        metrics=mode,
        full_metrics_episodes=(1,) if selected else (),
        replay_episodes=(1,) if selected else (),
    )
    env = AutoReset(base)
    key, reset, init = jax.random.split(jax.random.key(seed), 3)
    obs, state = env.reset(reset)
    system = marl_bgs.System("Per-Actor Counter", _act, init=_initialize)
    memory = marl_bgs.init_systems(system, system, obs, state, init)
    tracker = marl_bgs.init_episode_tracking(env, state).begin_stage(
        state, total_env_steps=8
    )

    def advance(handle: AutoReset, carry: Carry) -> Carry:
        state, tracking, methods, rng = carry
        rng, action_key, step_key = jax.random.split(rng, 3)
        actions, methods, learning = marl_bgs.apply_systems(
            system,
            system,
            methods,
            handle.get_observations(state),
            state,
            action_key,
        )
        result = handle.step(step_key, state, actions)
        tracking, result = marl_bgs.track_episode_step(tracking, state, result)
        data = marl_bgs.system_step_data(state, actions, result)
        # Make learning/final inputs part of checked execution without keeping
        # their expanded source banks in the recurrent carry.
        final = handle.final_policy_inputs(result[4])
        assert learning[0].shape == (2, 5)
        assert data.advanced.shape == (2,)
        assert final.valid.shape == (2,)
        return result[1], tracking, methods, rng

    return env, (state, tracker, memory, key), advance


@pytest.mark.parametrize("mode", ["none", "priority", "full"])
def test_compiled_restore_and_validation_preserve_training(mode: MetricMode) -> None:
    env, initial, advance = _setup(mode, 42, selected=mode == "none")

    def two_steps(handle: AutoReset, carry: Carry) -> Carry:
        def body(value: Carry, unused: None) -> tuple[Carry, None]:
            del unused
            return advance(handle, value), None

        return jax.lax.scan(body, carry, None, length=2)[0]

    compiled = cast(Callable[[AutoReset, Carry], Carry], jax.jit(two_steps))
    saved = compiled(env, initial)

    def restore(value: Array) -> Array:
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):  # pyright: ignore[reportPrivateImportUsage]
            return cast(Array, jax.random.wrap_key_data(jax.random.key_data(value)))
        return jax.device_put(np.asarray(value))

    restored = jax.tree.map(restore, saved)
    validation_env, validation, validate = _setup("none", 99)
    jax.jit(validate)(validation_env, validation)
    uninterrupted = compiled(env, saved)
    resumed = compiled(env, restored)
    _equal(uninterrupted, resumed)
    state, tracking, _, _ = resumed
    assert tracking.stage_summary(state) == {
        "requested_env_steps": 8,
        "env_steps": 8,
        "default_spawn_steps": 4,
        "swapped_spawn_steps": 4,
        "unreported_spawn_steps": 0,
        "status": "complete",
        "reason": None,
    }


def test_batched_seed_axis_matches_separate_independent_contexts() -> None:
    env, first, advance = _setup("none", 42)
    _, second, _ = _setup("none", 43)
    compiled = cast(Callable[[AutoReset, Carry], Carry], jax.jit(advance))
    separately = [compiled(env, carry) for carry in (first, second)]

    def stack(a: Array, b: Array) -> Array:
        return jnp.stack((a, b))

    batched = jax.tree.map(stack, first, second)
    multi = cast(
        Callable[[AutoReset, Carry], Carry],
        jax.jit(jax.vmap(advance, in_axes=(None, 0))),
    )
    combined = multi(env, batched)
    for index, expected in enumerate(separately):

        def row(values: Array, index: int = index) -> Array:
            return values[index]

        actual = jax.tree.map(row, combined)
        _equal(expected, actual)
        state, tracking, _, _ = actual
        assert tracking.stage_summary(state)["env_steps"] == 2
        assert tracking.stage_summary(state)["status"] == "incomplete"
