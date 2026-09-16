"""Describe actor actions and join two teams' actions for Core.

ActorAction stores one actor's three action choices. A batch of five ActorAction
rows represents a team in supplied roster order. The assembler places Team A
before Team B for Core. It does not grant actors access to global identities,
check legality or change submitted choices.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import Action


class ActorAction(NamedTuple):
    """Store one actor's movement, target and Ultimate choices.

    Attributes
    ----------
    move : Array
        Scalar int32 movement category from Core's action vocabulary.
    select_target : Array
        Scalar int32 target category: 0 for no target, 1 through 5 for ally
        rows, and 6 through 10 for enemy rows in the actor's observation.
    use_ultimate : Array
        Scalar int32 choice: 0 without Ultimate, 1 with Ultimate.

    Notes
    -----
    A team batch adds a leading axis of size 5 to each field. The record is an
    immutable JAX-compatible tuple. It does not check values. Select the target
    and Ultimate together from the actor's joint mask; legal values in each
    separate head can still form an illegal pair.
    """

    move: Array  # scalar JAX array int32
    select_target: Array  # scalar JAX array int32
    use_ultimate: Array  # scalar JAX array int32


def build_joint_action_from_actor_actions(
    team_a_joint_action: ActorAction, team_b_joint_action: ActorAction
) -> Action:
    """Join Team A and Team B actions in Core's fixed slot order.

    Parameters
    ----------
    team_a_joint_action : ActorAction
        Team A's five actions. Each field must be an int32 array of shape (5,),
        including the no-op choices for inactive slots.
    team_b_joint_action : ActorAction
        Team B's five actions with the same field shapes and dtypes.

    Returns
    -------
    Action
        Three arrays of shape (10,). Team A occupies global slots 0 through 4;
        Team B occupies slots 5 through 9. Within-team order is unchanged.

    Notes
    -----
    This numerical helper works inside jit and can be mapped over games with vmap.
    It joins arrays along their first axis, so pass one game at a time or vmap the
    call. Inputs are unchanged. The caller supplies compatible shapes and legal
    choices; this function performs no mask checks, clipping or action repair.
    """

    joint_move_action = jnp.concatenate(
        (team_a_joint_action.move, team_b_joint_action.move)
    )
    joint_select_target_action = jnp.concatenate(
        (team_a_joint_action.select_target, team_b_joint_action.select_target)
    )
    joint_use_ultimate_action = jnp.concatenate(
        (team_a_joint_action.use_ultimate, team_b_joint_action.use_ultimate)
    )

    return Action(
        joint_move_action, joint_select_target_action, joint_use_ultimate_action
    )
