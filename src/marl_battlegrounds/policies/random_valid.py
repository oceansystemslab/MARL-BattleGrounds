"""Sample legal random actions for one actor.

random_policy uses the actor's movement mask and joint target/Ultimate mask.
It needs no shared observations, memory or model weights. Environment sampling
routes this same policy over actors; legality remains owned by Core's masks.
"""

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction


def random_policy(
    observation: Observation,
    action_mask: ActionMask,
    key: Array,
) -> ActorAction:
    """Choose a random legal movement and a random legal combat pair.

    Parameters
    ----------
    observation : Observation
        One actor's current observation, without a leading actor or game axis.
        Accepted to match the policy interface; this policy does not read it.
    action_mask : ActionMask
        That actor's mask for the same decision. Movement has shape (9,);
        the joint target/Ultimate mask has shape (11, 2). Values are Boolean.
        Each sampled mask must contain at least one True entry.
    key : Array
        One JAX random key. It is split into separate movement and combat keys.

    Returns
    -------
    ActorAction
        Three scalar int32 choices. Movement is uniform over allowed movements.
        The target/Ultimate pair is uniform over allowed pairs, not over each
        combat head independently.

    Notes
    -----
    Use the exact current mask supplied by Core. This function does not check
    whether a mask is stale or empty. Core gives inactive and dead actors their
    allowed no-op choices. Inputs are unchanged. The function supports jit and
    vmap; provide distinct keys when independent draws are wanted.
    """

    del observation

    move_logits = jnp.where(action_mask.move_mask, 1.0, -jnp.inf).astype(jnp.float32)
    select_target_use_ultimate_logits = jnp.where(
        action_mask.select_target_use_ultimate_joint_mask, 1.0, -jnp.inf
    ).astype(jnp.float32)

    move_key, select_target_use_ultimate_key = jax.random.split(key, 2)

    random_movement_action = jax.random.categorical(move_key, move_logits)

    random_select_target_use_ultimate_action_pair_flattened = jax.random.categorical(
        select_target_use_ultimate_key,
        jnp.ravel(select_target_use_ultimate_logits),
    )

    random_select_target_use_ultimate_action_pair = jnp.unravel_index(
        random_select_target_use_ultimate_action_pair_flattened,
        (NUM_TARGET_ACTIONS, NUM_ULTIMATE_ACTIONS),
    )

    return ActorAction(
        move=random_movement_action.astype(jnp.int32),
        select_target=random_select_target_use_ultimate_action_pair[0].astype(
            jnp.int32
        ),
        use_ultimate=random_select_target_use_ultimate_action_pair[1].astype(jnp.int32),
    )
