"""Selected per-step replay packets, independent of numerical metric selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)

if TYPE_CHECKING:
    from marl_battlegrounds.environment import EnvironmentState


class ReplayPackets(NamedTuple):
    """A fixed-capacity packet batch; invalid rows contain no episode evidence.

    Initial fields are present only on the first valid transition. Later packets
    carry the successor and authoritative facts; the writer joins them to the
    preceding frame. No episode-length buffer lives inside the environment.
    """

    valid: Array
    episode_id: Array
    transition_index: Array
    initial: Array
    config: EnvConfig
    initial_state: EnvState
    initial_observation: Observation
    initial_action_mask: ActionMask
    state: EnvState
    observation: Observation
    action_mask: ActionMask
    reward: Reward
    done: DoneFlags
    info: Info


def capture_packets(
    before: EnvironmentState,
    after: EnvironmentState,
    reward: Reward,
    info: Info,
    *,
    capacity: int,
) -> ReplayPackets:
    """Gather only requested real transitions, with bounded static packet capacity."""
    scalar = before.episode_id.ndim == 0
    valid = before.collect_replay & info.transition_facts.has_transition
    valid = jnp.atleast_1d(valid)
    indices = jnp.nonzero(valid, size=capacity, fill_value=0)[0]
    packet_valid = jnp.arange(capacity) < valid.sum(dtype=jnp.int32)

    def gather(value: Array) -> Array:
        batch = value[None] if scalar else value
        return batch[indices]

    def admitted(value: Array) -> Array:
        gathered = gather(value)
        mask = packet_valid.reshape((capacity, *((1,) * (gathered.ndim - 1))))
        return jnp.where(mask, gathered, jnp.zeros_like(gathered))

    episode_id = admitted(before.episode_id)
    transition_index = admitted(
        before.core_state.step_count - before.initial_step_count
    )
    initial = packet_valid & (transition_index == 0)

    def first(value: Array) -> Array:
        gathered = gather(value)
        mask = initial.reshape((capacity, *((1,) * (gathered.ndim - 1))))
        return jnp.where(mask, gathered, jnp.zeros_like(gathered))

    return ReplayPackets(
        packet_valid,
        episode_id,
        transition_index,
        initial,
        jax.tree.map(first, before.config),
        jax.tree.map(first, before.core_state),
        jax.tree.map(first, before.observation),
        jax.tree.map(first, before.action_mask),
        jax.tree.map(admitted, after.core_state),
        jax.tree.map(admitted, after.observation),
        jax.tree.map(admitted, after.action_mask),
        jax.tree.map(admitted, reward),
        jax.tree.map(admitted, after.done),
        jax.tree.map(admitted, info),
    )
