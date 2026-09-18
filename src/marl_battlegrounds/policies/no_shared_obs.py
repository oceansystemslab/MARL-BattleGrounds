"""Call a policy separately for each actor without sharing private observations.

execute_no_shared_obs_team_policy selects one team's five routing rows, then
maps the same scalar policy over them. Global team identity is used only to
select rows. Each policy call receives its actor's own observation, mask and
key; the policy does not receive the team ID or other actors' private rows.
"""

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
    """Apply one policy to the five actor slots of the selected team.

    Parameters
    ----------
    observation : Observation
        Current observations for one game. Every leaf starts with the ten global
        actor slots. The contents of each actor's row are already actor-relative.
    action_mask : ActionMask
        Current masks in the same ten-slot order as observation.
    key : Array
        Ten per-actor JAX keys in global-slot order. Use typed keys of shape (10,)
        or legacy uint32 keys of shape (10, 2); a single root key is not accepted.
    policy : callable
        A scalar callable taking (observation, action_mask, key) and returning
        ActorAction. It must work with JAX transformations. The callable is static
        for compilation; numerical observations, masks, keys and team ID are not.
    team_identity : int or Array
        TEAM_A_ID or TEAM_B_ID. This is an internal routing input. The caller must
        validate it before compiled execution; this function does not reject other
        values.

    Returns
    -------
    ActorAction
        Five action rows in that team's existing slot order, including inactive
        slots. Each field has shape (5,).

    Notes
    -----
    The policy sees neither team_identity nor the global routing axis. No masks
    or observations are rebuilt here. The policy owns legal action selection.
    Inputs are unchanged and no recording or host callback is performed.
    """

    start_index = jnp.where(team_identity == TEAM_A_ID, TEAM_A_START, TEAM_B_START)

    def _prune_tree(leaf: Array) -> Array:
        """Take the selected team's five consecutive rows from one numerical leaf.

        The enclosing call owns the valid start index and ten-row input shape. The
        slice is dynamic in JAX and retains all dimensions after the actor axis.
        """
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
