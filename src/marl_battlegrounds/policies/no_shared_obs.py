"""NoSharedObs execution over fixed recipient-aligned policy inputs."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.axis_mappings import (
    TEAM_A_START,
    TEAM_B_START,
)
from marl_battlegrounds.core.types import (
    MAX_AGENTS_PER_TEAM,
    TEAM_A_ID,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction

NoSharedObsPolicy = Callable[[Observation, ActionMask, Array], ActorAction]


@jax.jit(static_argnums=3)
def execute_no_shared_obs_team_policy(
    observation: Observation,
    action_mask: ActionMask,
    key: Array,
    policy: NoSharedObsPolicy,
    team_identity: int | Array,
) -> ActorAction:
    """Map one scalar NoSharedObs policy over a fixed five-slot team block.

    ``observation``, ``action_mask``, and ``key`` retain global-slot order on
    entry. ``team_identity`` is expected to be either ``TEAM_A_ID`` or
    ``TEAM_B_ID``; host validation of that internal precondition stays outside
    this traced execution seam.
    """

    start_index = jnp.where(team_identity == TEAM_A_ID, TEAM_A_START, TEAM_B_START)

    def _prune_tree(leaf: Array) -> Array:
        return jax.lax.dynamic_slice_in_dim(leaf, start_index, MAX_AGENTS_PER_TEAM)

    team_observation = jax.tree.map(_prune_tree, observation)
    team_action_mask = jax.tree.map(_prune_tree, action_mask)
    team_keys = jax.tree.map(_prune_tree, key)

    policy_vmap = jax.vmap(
        fun=policy,
        in_axes=(0, 0, 0),
        out_axes=0,
    )

    return policy_vmap(team_observation, team_action_mask, team_keys)
