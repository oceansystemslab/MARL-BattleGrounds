"""Supply a slow host System for the real DevClient browser workflow proof.

The launcher loads this factory only when its declared menu choice is selected.
A two-second decision lets the test exercise responsive views and cancellation.
No model, network call or GPU is used. Memory advances only on accepted turns.
"""

import time

import jax
import jax.numpy as jnp

from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.policies.actor import ActorAction


def _init(variables: PolicyTree, inputs: SystemInput, key: jax.Array) -> list[int]:
    return [0 for _ in inputs.valid]


def _apply(
    variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, key: jax.Array
) -> SystemOutput:
    time.sleep(2)
    zeros = jnp.zeros(inputs.active_mask.shape, dtype=jnp.int32)
    return SystemOutput(
        ActorAction(zeros, zeros, zeros), [value + 1 for value in memory]
    )


def make_system() -> System:
    return System("Browser Counter", _apply, init=_init, execution="host")
