"""Uniform random sampling from one actor's exact local action support."""

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
    """Sample legal movement and combat choices for one NoSharedObs actor.

    Movement is sampled from its categorical mask. Target and Ultimate are
    sampled together from the exact joint mask so legal marginals cannot form
    an illegal pair. The observation remains in the callable contract even
    though this policy deliberately uses no observation features.
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
