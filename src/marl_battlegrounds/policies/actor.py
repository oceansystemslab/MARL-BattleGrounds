"""Mode-neutral scalar and fixed-team policy action contracts."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import Action


class ActorAction(NamedTuple):
    """Factored scalar action selected for one fixed actor slot."""

    move: Array  # scalar JAX array int32
    select_target: Array  # scalar JAX array int32
    use_ultimate: Array  # scalar JAX array int32


def build_joint_action_from_actor_actions(
    team_a_joint_action: ActorAction, team_b_joint_action: ActorAction
) -> Action:
    """Concatenate two fixed five-slot actor batches into one core action.

    Team A remains in global slots ``0:5`` and Team B in ``5:10``. The
    assembler deliberately performs no validation, masking, or semantic repair.
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
