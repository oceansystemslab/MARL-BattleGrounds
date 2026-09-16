"""Build fixed-capacity replay packets from selected real transitions.

This numerical gather sits after a wrapper step. It preserves the acting
source permissions, successor state and Core facts, with initial data only
on each episode's first valid transition. Invalid capacity rows are zeros.
Collection into a complete replay and file writing belong to host modules.
"""

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
    preceding frame. Source availability belongs to the acting frame.
    No episode-length buffer lives inside the environment.

    Attributes
    ----------
    valid : Array
        bool (C,) admission mask for fixed packet capacity C.
    episode_id : Array
        int32 (C,) identity from the acting state.
    transition_index : Array
        int32 (C,) step offset from that episode's initial step.
    initial : Array
        bool (C,); True only for a first valid transition.
    config : EnvConfig
        Config tree with C leading rows; populated only for initial rows.
    initial_state : EnvState
        Acting Core state, populated only for initial rows.
    initial_observation : Observation
        Acting observation, populated only for initial rows.
    initial_action_mask : ActionMask
        Acting masks, populated only for initial rows.
    state : EnvState
        Successor Core state for each admitted transition.
    observation : Observation
        Successor observation for each admitted transition.
    action_mask : ActionMask
        Successor masks for each admitted transition.
    reward : Reward
        Core reward from each admitted transition.
    done : DoneFlags
        Successor termination and truncation flags.
    info : Info
        Authoritative Core facts for the admitted transition.
    source_availability : Array
        Acting epoch's (C, 10, 10) source permissions.

    Invalid rows are zero padding. A scan adds its time axis before C. This
    privileged recording payload is not a policy input or a host artifact.
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
    source_availability: Array


def capture_packets(
    before: EnvironmentState,
    after: EnvironmentState,
    reward: Reward,
    info: Info,
    *,
    capacity: int,
) -> ReplayPackets:
    """Gather selected real transitions into a fixed-capacity recording payload.

    Parameters
    ----------
    before : EnvironmentState
        Acting EnvironmentState, scalar or native batch (B,).
    after : EnvironmentState
        Matching successor with the same lane layout.
    reward : Reward
        Core reward from that transition, with the same lane layout.
    info : Info
        Core Info from that transition; has_transition decides admission.
    capacity : int
        Static packet count C. The caller must make C large enough for
        all simultaneously selected lanes; excess rows would be omitted.

    Returns
    -------
    ReplayPackets
        ReplayPackets with C leading rows. Valid selected lanes retain lane order;
        unused rows are zero. Initial fields are populated only at transition
        index zero, while each valid row retains the successor and Core facts.

    This pure numerical gather supports jit/vmap/scan and has no file I/O or
    episode-length buffer. Inputs are unchanged. Selection comes from before;
    no metrics option is needed to capture replay.
    """
    scalar = before.episode_id.ndim == 0
    valid = before.collect_replay & info.transition_facts.has_transition
    valid = jnp.atleast_1d(valid)
    indices = jnp.nonzero(valid, size=capacity, fill_value=0)[0]
    packet_valid = jnp.arange(capacity) < valid.sum(dtype=jnp.int32)

    def gather(value: Array) -> Array:
        """Gather selected rows after adding the scalar lane axis if needed."""
        batch = value[None] if scalar else value
        return batch[indices]

    def admitted(value: Array) -> Array:
        """Keep valid gathered rows and zero padding without changing static shapes."""
        gathered = gather(value)
        mask = packet_valid.reshape((capacity, *((1,) * (gathered.ndim - 1))))
        return jnp.where(mask, gathered, jnp.zeros_like(gathered))

    episode_id = admitted(before.episode_id)
    transition_index = admitted(
        before.core_state.step_count - before.initial_step_count
    )
    initial = packet_valid & (transition_index == 0)

    def first(value: Array) -> Array:
        """Keep initial data only on each replay's first valid transition."""
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
        admitted(before.source_availability),
    )
