"""Apply researcher policies through each actor's permitted view.

Policy stores a method and its explicit parameters and memory. apply_policies
redacts shared sources, calls both teams from one decision epoch and assembles
their joint action. It does not change game state or choose a learning method.
The JAX path batches actors; the host path makes synchronous Python calls and
copies permitted inputs to the host. Recording identities use real callables,
not policy names alone.
"""

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

    Parameters
    ----------
    name : str
        Nonempty policy label used in tables and recorded passes.
    apply : PolicyApply
        Callable taking variables, one actor's memory, ActorInput, its
        ActionMask and a JAX key. Return (ActorAction, next_memory); each action
        head is a scalar int32. The actor input is already redacted.
    variables : PolicyTree
        Numerical parameter tree, default (). The evaluator snapshots
        mutable NumPy leaves once. This class itself stores the supplied value.
    initial_carry : PolicyTree
        One actor's initial memory tree, default (). Evaluation
        broadcasts it over environments and the five actors on this team.
    checkpoint : str | None
        Optional caller-supplied checkpoint label; default None.
    execution : PolicyExecution
        "jax" by default, or "host" for Python/external methods. JAX
        callables must support tracing and fixed memory shapes.

    Raises
    ------
    TypeError
        apply is not callable.
    ValueError
        name is blank or execution is not "jax" or "host".

    The descriptor is frozen; referenced mutable objects are not made immutable
    by constructing it. The callable owns its method and must respect each
    actor's information limits, including any memory or predictions it uses.
    """

    name: str
    apply: PolicyApply
    variables: PolicyTree = ()
    initial_carry: PolicyTree = ()
    checkpoint: str | None = None
    execution: PolicyExecution = "jax"

    def __post_init__(self) -> None:
        """Reject an empty label, a non-callable method or an unknown execution mode."""
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
    """Apply the existing random controller and return the actor memory unchanged."""
    del variables
    # Preserve the original NoSharedObs callable's three-argument input boundary.
    return random_policy(actor.observation, mask, key), carry


def _shared_apply(function: SharedObsPolicy) -> PolicyApply:
    """Adapt an existing shared-observation controller to the Policy call signature."""

    def apply(
        variables: PolicyTree,
        carry: PolicyTree,
        actor: ActorInput,
        mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        """Pass this actor's observation and permitted source bank to the controller."""
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
    """Return a stateless adapter for one built-in controller.

    Parameters
    ----------
    name : str
        Exactly "random", "tdm-alpha" or "tdm-beta".

    Returns
    -------
    Policy
        A Policy with the supplied name, empty variables/memory and JAX execution.
        The adapter keeps the controller's existing observation and action rules.

    Raises
    ------
    ValueError
        name does not identify a registered controller.

    This host lookup does not run an environment or load a checkpoint.
    """
    if name not in _CONTROLLERS:
        raise ValueError(f"unknown policy {name!r}; choose {', '.join(_CONTROLLERS)}")
    return Policy(name=name, apply=_CONTROLLERS[name])


def controller_identity(team: Policy) -> dict[str, object] | None:
    """Return verified rule metadata for a registered scripted controller.

    Parameters
    ----------
    team : Policy
        Policy whose callable is checked by object identity.

    Returns
    -------
    dict[str, object] | None
        A fresh dict with identifier, version and canonical_digest for the exact
        registered ALPHA or BETA callable; otherwise None. A matching policy name
        alone does not establish identity.

    This host metadata lookup does not execute the policy or hash its variables.
    """
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
    """Convert a numerical tree to JAX arrays, copying mutable NumPy leaves.

    Parameters
    ----------
    variables : PolicyTree
        Parameter or memory PyTree. Empty trees are allowed; every
        leaf must be accepted by jnp.asarray.

    Returns
    -------
    PolicyTree
        The same tree structure with JAX array leaves. NumPy arrays are copied so
        later caller edits cannot change the snapshot. Existing JAX leaves need
        no forced copy; Python scalars follow JAX's current dtype settings.

    Raises
    ------
    TypeError
        A leaf cannot be converted to a numerical JAX array.

    Call once at the host setup boundary. Transfers may occur here; no input
    container is edited and this does not freeze arbitrary external state.
    """

    def freeze(value: object) -> Array:
        """Copy mutable NumPy storage; otherwise use JAX's normal array conversion."""
        if isinstance(value, np.ndarray):
            return jnp.array(value, copy=True)
        return jnp.asarray(value)

    return jax.tree.map(freeze, variables)


def initial_policy_carry(initial_carry: PolicyTree, num_envs: int) -> PolicyTree:
    """Give every lane and team actor its own view of initial memory.

    Parameters
    ----------
    initial_carry : PolicyTree
        One actor's numerical memory tree; () means no memory.
    num_envs : int
        Positive environment count B, checked by the caller.

    Returns
    -------
    PolicyTree
        The same tree with each original leaf shape S expanded to (B, 5, *S).
        Mutable NumPy inputs are snapshotted first. Updating returned carry later
        does not change this template or another actor's carry.

    Use during host setup; broadcasting is numerical and can also be traced.
    """

    def broadcast(value: object) -> Array:
        """Add environment and five-actor axes while preserving each memory leaf's
        shape.
        """
        array = jnp.asarray(value)
        return jnp.broadcast_to(array, (num_envs, 5, *array.shape))

    return jax.tree.map(broadcast, freeze_variables(initial_carry))


def select_policy_carry(
    mask: Array, selected: PolicyTree, other: PolicyTree
) -> PolicyTree:
    """Choose complete memory rows without changing the input trees.

    Parameters
    ----------
    mask : Array
        Boolean leading-axis mask, normally (B,) or scalar for one lane.
    selected : PolicyTree
        Memory tree to use where mask is True.
    other : PolicyTree
        Same structure, shapes and dtypes as selected; used elsewhere.

    Returns
    -------
    PolicyTree
        A new tree selected with jnp.where. The mask broadcasts across remaining
        actor and memory axes, so all actors in a selected lane reset together.

    This pure numerical helper supports jit/vmap/scan. Callers must supply
    compatible shapes; it performs no host validation.
    """

    def choose(new: Array, old: Array) -> Array:
        """Broadcast the lane mask across the remaining memory axes before selecting."""
        shape = (*mask.shape, *((1,) * (new.ndim - mask.ndim)))
        return jnp.where(mask.reshape(shape), new, old)

    return jax.tree.map(choose, selected, other)


def _checked_action(action: object) -> ActorAction:
    """Require three scalar int32 action heads; leave value-domain decisions to Core."""
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

    Parameters
    ----------
    apply_a : PolicyApply
        Team A scalar PolicyApply callable.
    apply_b : PolicyApply
        Team B scalar PolicyApply callable.
    variables_a : PolicyTree
        Team A parameter tree, shared by its five calls.
    variables_b : PolicyTree
        Team B parameter tree, shared by its five calls.
    carry_a : PolicyTree
        Team A memory tree with leading actor axis (5,).
    carry_b : PolicyTree
        Team B memory tree with leading actor axis (5,).
    observations : Observations
        One environment's compact Observations, with ten actor rows
        and a (10, 10) source-availability matrix for this decision epoch.
    action_mask : ActionMask
        Joint masks with ten actor rows from the same epoch.
    actor_keys : Array
        Ten typed JAX keys, or legacy uint32 keys shaped (10, 2).
    execution_a : PolicyExecution
        "jax" by default; "host" uses five Python actor calls.
    execution_b : PolicyExecution
        Same mode choice for Team B.

    Returns
    -------
    tuple[Action, PolicyTree, PolicyTree]
        (joint_action, next_carry_a, next_carry_b). Each Action head is int32 (10,)
        in global Team A then Team B order. Memory keeps its actor axis and tree
        layout; JAX/scan callers must keep leaf shapes and dtypes fixed.

    Raises
    ------
    TypeError
        A policy result is not ActorAction, or an action head is not
        a scalar int32. Policy exceptions propagate unchanged.

    Both teams choose from the supplied pre-step epoch. Calls do not mutate the
    inputs or advance the game. JAX mode supports outer jit/vmap; host mode copies
    each actor's permitted input and mask to the host and must run outside jit.
    Variables, memory and keys retain their supplied representation at that host
    call; providers may need their own conversion. Values outside legal action
    domains are not silently repaired here; Core owns accepted-action behavior.
    """
    bank = build_shared_obs_sensor_source_bank(observations.observation)

    def team(
        start: int,
        apply: PolicyApply,
        variables: PolicyTree,
        carry: PolicyTree,
        execution: PolicyExecution,
    ) -> tuple[ActorAction, PolicyTree]:
        """Apply this team's five actors at the chosen execution boundary."""

        def take_bank(leaf: Array) -> Array:
            """Read this team's bank; never include the opposing team's bank."""
            return leaf[start // 5]

        team_bank = jax.tree.map(take_bank, bank)

        def take(value: Array) -> Array:
            """Select five contiguous global actor rows for the current team."""
            return value[start : start + 5]

        def actor_apply(
            memory: PolicyTree,
            observation: Observation,
            mask: ActionMask,
            availability: Array,
            key: Array,
        ) -> tuple[ActorAction, PolicyTree]:
            """Redact one actor's sources, call its policy and check its action."""
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
                """Read one actor's input and memory from the team's leading axis."""
                return value[index]

            results.append(actor_apply(*jax.tree.map(row, inputs)))

        def stack(*values: Array) -> Array:
            """Restore the five-actor axis after separate Python policy calls."""
            return jnp.stack(values)

        return cast(tuple[ActorAction, PolicyTree], jax.tree.map(stack, *results))

    action_a, next_a = team(0, apply_a, variables_a, carry_a, execution_a)
    action_b, next_b = team(5, apply_b, variables_b, carry_b, execution_b)
    return build_joint_action_from_actor_actions(action_a, action_b), next_a, next_b
