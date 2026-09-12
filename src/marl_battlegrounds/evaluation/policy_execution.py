"""One authorized actor-application boundary for learned and existing policies."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core.types import Action, ActionMask, Observation
from marl_battlegrounds.evaluation.models import canonical_digest_sha256
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.input import ActorInput, Observations
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.policies.reactive_tdm_alpha import (
    reactive_tdm_alpha_controller_descriptor,
    reactive_tdm_alpha_policy,
)
from marl_battlegrounds.policies.reactive_tdm_beta import (
    reactive_tdm_beta_controller_descriptor,
    reactive_tdm_beta_policy,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsPolicy,
    build_shared_obs_sensor_source_bank,
    mask_source_bank_for_recipient,
)

# Policies may use any ordinary array PyTree, including an empty tuple for a
# stateless controller. The tree structure and leaf shapes remain fixed in scan.
type PolicyTree = Any
type PolicyApply = Callable[
    [PolicyTree, PolicyTree, ActorInput, ActionMask, Array],
    tuple[ActorAction, PolicyTree],
]
type PolicyExecution = Literal["jax", "host"]


@dataclass(frozen=True, eq=False)
class Policy:
    """A scalar actor function with explicit variables and episode-local memory.

    ``apply(variables, carry, actor_input, action_mask, key)`` returns an
    ``ActorAction`` and next carry. It must be pure and honor the supplied RNG.
    Variables are dynamic inputs, never static JIT arguments. Evaluation freezes
    their current value; ``checkpoint`` records an identity supplied by the caller
    and stays absent when no checkpoint identity is known.

    Use ``execution='host'`` for a non-JAX provider. Its synchronous calls have the
    same decision barrier and propagate errors, including provider timeouts.
    Such providers incur per-step host/device transfers; the default JAX path
    applies actors and environments in compiled batches.
    """

    name: str
    apply: PolicyApply
    variables: PolicyTree = ()
    initial_carry: PolicyTree = ()
    checkpoint: str | None = None
    execution: PolicyExecution = "jax"

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.name), str) or not self.name.strip():
            raise ValueError("policy name must be a nonempty string")
        if not callable(self.apply):
            raise TypeError("policy apply must be callable")
        if self.execution not in ("jax", "host"):
            raise ValueError("policy execution must be 'jax' or 'host'")


def _random_apply(
    variables: PolicyTree,
    carry: PolicyTree,
    actor: ActorInput,
    mask: ActionMask,
    key: Array,
) -> tuple[ActorAction, PolicyTree]:
    del variables
    # Preserve the original NoSharedObs callable's three-argument input boundary.
    return random_policy(actor.observation, mask, key), carry


def _shared_apply(function: SharedObsPolicy) -> PolicyApply:
    def apply(
        variables: PolicyTree,
        carry: PolicyTree,
        actor: ActorInput,
        mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        del variables
        return function(
            actor.observation,
            mask,
            key,
            actor.source_bank,
            actor.source_availability,
        ), carry

    return apply


_CONTROLLERS: dict[str, PolicyApply] = {
    "random": _random_apply,
    "tdm-alpha": _shared_apply(reactive_tdm_alpha_policy),
    "tdm-beta": _shared_apply(reactive_tdm_beta_policy),
}


def policy(name: str) -> Policy:
    """Adapt ``random``, ``tdm-alpha`` or diagnostic ``tdm-beta`` without new rules."""
    if name not in _CONTROLLERS:
        raise ValueError(f"unknown policy {name!r}; choose {', '.join(_CONTROLLERS)}")
    return Policy(name=name, apply=_CONTROLLERS[name])


def controller_identity(team: Policy) -> dict[str, object] | None:
    """Identify registered controller rules by their actual callable, not a label."""
    for name, describe in (
        ("tdm-alpha", reactive_tdm_alpha_controller_descriptor),
        ("tdm-beta", reactive_tdm_beta_controller_descriptor),
    ):
        if team.apply is _CONTROLLERS[name]:
            descriptor = describe()
            return {
                "identifier": descriptor["policy_id"],
                "version": descriptor["version"],
                "canonical_digest": canonical_digest_sha256(descriptor),
            }
    return None


def freeze_variables(variables: PolicyTree) -> PolicyTree:
    """Snapshot mutable NumPy inputs once; immutable JAX arrays need no copy."""

    def freeze(value: object) -> Array:
        if isinstance(value, np.ndarray):
            return jnp.array(value, copy=True)
        return jnp.asarray(value)

    return jax.tree.map(freeze, variables)


def initial_policy_carry(initial_carry: PolicyTree, num_envs: int) -> PolicyTree:
    """Give each environment and each of five team actors independent memory."""

    def broadcast(value: object) -> Array:
        array = jnp.asarray(value)
        return jnp.broadcast_to(array, (num_envs, 5, *array.shape))

    return jax.tree.map(broadcast, freeze_variables(initial_carry))


def select_policy_carry(
    mask: Array, selected: PolicyTree, other: PolicyTree
) -> PolicyTree:
    """Replace whole environment memory rows without resetting other actors."""

    def choose(new: Array, old: Array) -> Array:
        shape = (*mask.shape, *((1,) * (new.ndim - mask.ndim)))
        return jnp.where(mask.reshape(shape), new, old)

    return jax.tree.map(choose, selected, other)


def _checked_action(action: object) -> ActorAction:
    if not isinstance(action, ActorAction):
        raise TypeError("policy apply must return (ActorAction, next_carry)")
    arrays = tuple(jnp.asarray(value) for value in action)
    if any(value.shape != () or value.dtype != jnp.int32 for value in arrays):
        raise TypeError("each policy action component must be an int32 scalar")
    # Out-of-domain integer choices remain Core's responsibility and can be
    # diagnosed through action-acceptance metrics; do not silently repair them.
    return ActorAction(*arrays)


def apply_policies(
    apply_a: PolicyApply,
    apply_b: PolicyApply,
    variables_a: PolicyTree,
    variables_b: PolicyTree,
    carry_a: PolicyTree,
    carry_b: PolicyTree,
    observations: Observations,
    action_mask: ActionMask,
    actor_keys: Array,
    *,
    execution_a: PolicyExecution = "jax",
    execution_b: PolicyExecution = "jax",
) -> tuple[Action, PolicyTree, PolicyTree]:
    """Apply both teams from one epoch, returning one complete joint action.

    This function accepts one environment with ten actor rows. External ``vmap``
    adds the environment axis. It derives transient source banks once per team, then
    redacts immediately before each policy receives its scalar ``ActorInput``.
    """
    bank = build_shared_obs_sensor_source_bank(observations.observation)

    def team(
        start: int,
        apply: PolicyApply,
        variables: PolicyTree,
        carry: PolicyTree,
        execution: PolicyExecution,
    ) -> tuple[ActorAction, PolicyTree]:
        def take_bank(leaf: Array) -> Array:
            return leaf[start // 5]

        team_bank = jax.tree.map(take_bank, bank)

        def take(value: Array) -> Array:
            return value[start : start + 5]

        def actor_apply(
            memory: PolicyTree,
            observation: Observation,
            mask: ActionMask,
            availability: Array,
            key: Array,
        ) -> tuple[ActorAction, PolicyTree]:
            actor = ActorInput(
                observation,
                mask_source_bank_for_recipient(team_bank, availability),
                availability,
            )
            if execution == "host":
                actor, mask = jax.device_get((actor, mask))
            action, next_carry = apply(variables, memory, actor, mask, key)
            return _checked_action(action), next_carry

        inputs = (
            carry,
            jax.tree.map(take, observations.observation),
            jax.tree.map(take, action_mask),
            observations.source_availability[start : start + 5, start : start + 5],
            actor_keys[start : start + 5],
        )
        if execution == "jax":
            return jax.vmap(actor_apply)(*inputs)
        results = []
        for index in range(5):

            def row(value: Array, index: int = index) -> Array:
                return value[index]

            results.append(actor_apply(*jax.tree.map(row, inputs)))

        def stack(*values: Array) -> Array:
            return jnp.stack(values)

        return cast(tuple[ActorAction, PolicyTree], jax.tree.map(stack, *results))

    action_a, next_a = team(0, apply_a, variables_a, carry_a, execution_a)
    action_b, next_b = team(5, apply_b, variables_b, carry_b, execution_b)
    return build_joint_action_from_actor_actions(action_a, action_b), next_a, next_b
