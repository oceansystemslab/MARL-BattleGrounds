"""Check separate actor memory and changing Policy values in 32 environments.

The same test runs on CPU and CUDA. It keeps the legacy actor-key route and
checks compiled Policy calls, distinct lane/team/actor memory, and selected
memory resets. Environment setup always uses 32 lanes; this is not a speed test.
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import Action, ActionMask
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    apply_policies,
    initial_policy_carry,
    select_policy_carry,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

_BATCH = 32


def test_policy_actor_memory_and_dynamic_values_remain_separate() -> None:
    config = make_standard_team_deathmatch_config(
        map_id=12,
        team_a_roster=("mage", "priest", "warrior"),
        team_b_roster=("rogue", "priest"),
        max_steps=3,
    )
    env = marl_bgs.make("tdm", num_envs=_BATCH, metrics="none")
    observations, state = env.reset(jax.random.key(0), config)
    assert state.core_state.step_count.shape == (_BATCH,)
    traces: list[int] = []

    def apply(
        variables: PolicyTree,
        memory: PolicyTree,
        actor: ActorInput,
        action_mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        del action_mask, key
        traces.append(1)
        zero = jnp.asarray(0, jnp.int32)
        return (
            ActorAction(variables, zero, zero),
            memory + actor.observation.self_ally_index + 1,
        )

    run = cast(
        Callable[..., tuple[Action, Array, Array]],
        jax.jit(
            jax.vmap(
                apply_policies,
                in_axes=(None, None, None, None, 0, 0, 0, 0, 0),
            ),
            static_argnums=(0, 1),
        ),
    )
    template = cast(Array, initial_policy_carry(jnp.int32(0), _BATCH))
    offsets = jnp.arange(_BATCH, dtype=jnp.int32)[:, None] * 100
    actors = jnp.arange(5, dtype=jnp.int32)[None, :]
    initial_a = template + offsets + actors
    initial_b = template + 10_000 + offsets + actors
    first_keys = jax.random.split(jax.random.key(1), (_BATCH, 10))
    first, carry_a, carry_b = run(
        apply,
        apply,
        jnp.int32(0),
        jnp.int32(0),
        initial_a,
        initial_b,
        observations,
        state.action_mask,
        first_keys,
    )
    trace_count = len(traces)
    assert trace_count > 0
    second, next_a, next_b = run(
        apply,
        apply,
        jnp.int32(1),
        jnp.int32(2),
        carry_a,
        carry_b,
        observations,
        state.action_mask,
        jax.random.split(jax.random.key(2), (_BATCH, 10)),
    )
    assert len(traces) == trace_count
    np.testing.assert_array_equal(first.move, np.zeros((_BATCH, 10), np.int32))
    np.testing.assert_array_equal(
        second.move, np.broadcast_to(np.array([1] * 5 + [2] * 5), (_BATCH, 10))
    )
    increments_a = np.array([1, 2, 3, 1, 1], np.int32)
    increments_b = np.array([1, 2, 1, 1, 1], np.int32)
    np.testing.assert_array_equal(carry_a, np.asarray(initial_a) + increments_a)
    np.testing.assert_array_equal(carry_b, np.asarray(initial_b) + increments_b)
    np.testing.assert_array_equal(next_a, np.asarray(initial_a) + 2 * increments_a)
    np.testing.assert_array_equal(next_b, np.asarray(initial_b) + 2 * increments_b)

    reset_mask = jnp.arange(_BATCH) % 2 == 0
    reset_a = cast(Array, select_policy_carry(reset_mask, initial_a, next_a))
    reset_b = cast(Array, select_policy_carry(reset_mask, initial_b, next_b))
    np.testing.assert_array_equal(reset_a[::2], initial_a[::2])
    np.testing.assert_array_equal(reset_b[::2], initial_b[::2])
    np.testing.assert_array_equal(reset_a[1::2], next_a[1::2])
    np.testing.assert_array_equal(reset_b[1::2], next_b[1::2])
    np.testing.assert_array_equal(template, np.zeros((_BATCH, 5), np.int32))
