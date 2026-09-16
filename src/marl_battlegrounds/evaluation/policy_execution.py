"""Apply Policies and Systems through each actor's permitted view.

Policy stores a method and its explicit parameters and memory. apply_policies
redacts shared sources, calls both teams from one decision epoch and assembles
their joint action. It does not change game state or choose a learning method.
The JAX path batches actors; the host path makes synchronous Python calls and
copies permitted inputs to the host. Recording identities use real callables,
not policy names alone.

System adds batched method calls, explicit recurrent memory and optional learning
outputs above the same input/action authorities. init_systems prepares memory;
apply_systems handles reset generations and returns submitted actions without
advancing the game. Generic host Systems keep opaque memory outside JAX, while
their JAX opponents remain batched. Privileged training state stays separate.
"""

# Private System adapter fields are shared only by their owning module's helpers.
# pyright: reportPrivateUsage=false

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

from marl_battlegrounds.core.types import Action, ActionMask
from marl_battlegrounds.evaluation.models import canonical_digest_sha256
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.input import (
    ActorInput,
    Observations,
    build_team_actor_input,
)
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.policies.reactive_tdm_alpha import (
    reactive_tdm_alpha_controller_descriptor,
    reactive_tdm_alpha_policy,
)
from marl_battlegrounds.policies.reactive_tdm_beta import (
    reactive_tdm_beta_controller_descriptor,
    reactive_tdm_beta_policy,
)
from marl_battlegrounds.policies.shared_obs import SharedObsPolicy

if TYPE_CHECKING:
    from marl_battlegrounds.environment import EnvironmentState

# Policies may use any ordinary array PyTree, including an empty tuple for a
# stateless controller. The tree structure and leaf shapes remain fixed in scan.
type PolicyTree = Any
type PolicyApply = Callable[
    [PolicyTree, PolicyTree, ActorInput, ActionMask, Array],
    tuple[ActorAction, PolicyTree],
]
type PolicyExecution = Literal["jax", "host"]


def _array_row(value: Array, *, index: int) -> Array:
    """Select one numerical leading-axis row without converting or copying it."""
    return value[index]


def _actor_column(value: Array, *, index: int) -> Array:
    """Select one actor from each environment; preserve later feature axes."""
    return value[:, index]


def _stack_arrays(*values: Array) -> Array:
    """Restore a leading batch axis after ordered numerical callback results."""
    return jnp.stack(values)


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

    def team(
        start: int,
        apply: PolicyApply,
        variables: PolicyTree,
        carry: PolicyTree,
        execution: PolicyExecution,
    ) -> tuple[ActorAction, PolicyTree]:
        """Prepare one team's permitted views and preserve scalar Policy calls."""
        actors = build_team_actor_input(observations, start // 5)

        def take_team(value: Array) -> Array:
            """Read this team's five current mask rows in simulator slot order."""
            return value[start : start + 5]

        masks = jax.tree.map(take_team, action_mask)
        return _apply_actor_rows(
            apply,
            variables,
            carry,
            actors,
            masks,
            actor_keys[start : start + 5],
            execution=execution,
        )

    action_a, next_a = team(0, apply_a, variables_a, carry_a, execution_a)
    action_b, next_b = team(5, apply_b, variables_b, carry_b, execution_b)
    return build_joint_action_from_actor_actions(action_a, action_b), next_a, next_b


def _apply_actor_rows(
    apply: PolicyApply,
    variables: PolicyTree,
    memory: PolicyTree,
    actors: ActorInput,
    masks: ActionMask,
    keys: Array,
    *,
    execution: PolicyExecution,
) -> tuple[ActorAction, PolicyTree]:
    """Apply scalar Policy callbacks to one leading row axis.

    Inputs are already redacted. JAX vmaps rows; host mode visits them in order
    and converts only actor inputs and masks to NumPy. Variables, memory and
    keys keep their representation. Return checked actions and stacked memory.
    Empty memory trees are supported. Policy callbacks own fixed tree shapes.
    """

    def one(
        carry: PolicyTree, actor: ActorInput, mask: ActionMask, key: Array
    ) -> tuple[ActorAction, PolicyTree]:
        """Call one scalar policy once, then check structural action validity."""
        if execution == "host":
            actor, mask = jax.device_get((actor, mask))
        action, updated = apply(variables, carry, actor, mask, key)
        return _checked_action(action), updated

    if execution == "jax":
        return jax.vmap(one)(memory, actors, masks, keys)
    rows = [
        one(*jax.tree.map(partial(_array_row, index=i), (memory, actors, masks, keys)))
        for i in range(keys.shape[0])
    ]
    return cast(
        tuple[ActorAction, PolicyTree],
        jax.tree.map(_stack_arrays, *rows),
    )


class SystemInput(NamedTuple):
    """One team's separate actor views at a single decision.

    Attributes
    ----------
    actors : ActorInput
        Permitted inputs with leading axes (B, 5). Each recipient retains its
        own information rights; the team method must not pool private rows.
    action_mask : ActionMask
        Boolean move, target, Ultimate and joint masks shaped (B,5,9),
        (B,5,11), (B,5,2) and (B,5,11,2).
    active_mask : Array
        Boolean (B,5) roster membership, including currently dead members.
    episode_start : Array
        Boolean (B,) first-decision markers. Death does not start a new episode.
    valid : Array
        Boolean (B,) live decisions, or selected lanes during initialization.
        False lanes are padding and must not update method-owned memory.

    Notes
    -----
    B is present even for scalar environments. No team ID, global slot, config
    or privileged state is delivered. Arrays are JAX values for JAX methods and
    NumPy values for host methods. Masks describe this same pre-step epoch.
    """

    actors: ActorInput
    action_mask: ActionMask
    active_mask: Array
    episode_start: Array
    valid: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SystemOutput:
    """Return an action, next memory and optional learning values from one call.

    Parameters
    ----------
    actions : ActorAction
        Three int32 arrays shaped (B,5), describing submitted actions.
    next_memory : Any
        Updated method-owned memory. Keep invalid lanes unchanged.
    learning_outputs : Any, keyword-only
        Already-computed learner values; default (). JAX trees, shapes and
        dtypes stay fixed per callable. Teams may return different structures.
    policy_ids : Array | None, keyword-only
        Optional int32 (B,5) indices into this System's components; -1 means
        unreported. Default None supplies all -1 values.

    Notes
    -----
    This frozen record is a JAX PyTree. Values pass through without copies,
    gradient changes or automatic recording. Host methods own the lifetime of
    returned objects: later calls must not overwrite retained learning values.
    An action's probability describes its submitted action, even if Core later
    rejects that action. A bare third tuple result always means policy IDs.
    """

    actions: ActorAction
    next_memory: PolicyTree
    learning_outputs: PolicyTree = field(default=(), kw_only=True)
    policy_ids: Array | None = field(default=None, kw_only=True)


class PolicyTrace(NamedTuple):
    """Identify the decision that supplied a joint action, without learner data.

    episode_id and decision_step are int32 (B,); valid is Boolean (B,).
    policy_ids is int32 (B,10), with fixed Team A then Team B ownership and
    per-team component indices. -1 means unreported. Initial and padding traces
    are invalid. This numerical record is not an automatic writer interface.
    """

    episode_id: Array
    decision_step: Array
    valid: Array
    policy_ids: Array


class SystemState(NamedTuple):
    """Carry both methods' memories and their environment reset bindings.

    team_a and team_b are independent method-owned memory trees or host values.
    episode_id and reset_generation are int32 (B,) bindings. init_key holds the
    original root or aligned lane roots supplied to init_systems. policy_trace
    describes the last submitted action, initially invalid. adapter_templates
    holds two small numerical initial-memory templates; keeping these dynamic
    avoids capturing arrays in static adapter functions. Generic methods use ().

    Obtain this record through init_systems and carry each returned successor.
    JAX methods use fixed-shape PyTrees. Host objects must stay outside jit.
    A new unrelated environment context requires new initialization; this record
    cannot detect a discarded execution branch. No learning outputs are stored.
    """

    team_a: PolicyTree
    team_b: PolicyTree
    episode_id: Array
    reset_generation: Array
    init_key: Array
    policy_trace: PolicyTrace
    adapter_templates: tuple[PolicyTree, PolicyTree] = ((), ())


class SystemStepData(NamedTuple):
    """Expose one team's submitted action and reward for an explicit step.

    actions contains int32 (B,5) heads. rewards is float32 (B,5), retaining
    Core's per-agent reward vector. active_mask is Boolean (B,5) roster activity;
    episode_id is int32 (B,) from before the step; advanced is Boolean (B,).
    Terminal steps advance, later padding does not. Scalar games retain B=1.
    Rewards are not aggregated and actions are not replaced by accepted actions.
    """

    actions: ActorAction
    rewards: Array
    active_mask: Array
    episode_id: Array
    advanced: Array


type SystemApply = Callable[[PolicyTree, PolicyTree, SystemInput, Array], Any]
type SystemInit = Callable[[PolicyTree, SystemInput, Array], PolicyTree]
type SystemReset = Callable[[PolicyTree, PolicyTree, Array], PolicyTree]


@dataclass(frozen=True)
class _SystemExecution:
    """Hash only stable callables and adapter layout, never numerical values.

    This private descriptor keys mixed-path JAX compilation. Names, checkpoint
    labels, parameter values and initial-memory templates are intentionally absent.
    """

    apply: SystemApply
    init: SystemInit | None
    reset: SystemReset | None
    execution: PolicyExecution
    policies: tuple[PolicyApply, ...] = ()
    shared: bool = False
    component_count: int = 0


@dataclass(frozen=True, eq=False)
class System:
    """Describe a method that acts for one team from permitted actor inputs.

    Parameters
    ----------
    name : str
        Nonempty display name. It does not establish a controller's identity.
    apply : callable
        apply(variables, memory, inputs, keys) returns (actions, next_memory),
        (actions, next_memory, policy_ids), or SystemOutput. Call once per
        decision; actions are ActorAction int32 (B,5). Keys are JAX (B,) typed
        keys or legacy (B,2), including at a host boundary.
    variables : Any
        Method parameters, default (). Pass changing arrays through explicit
        variables_a/b arguments inside compiled training functions.
    init : callable | None
        Optional init(variables, inputs, keys) returning fresh memory without
        choosing an action. None means (). inputs.valid selects new lanes.
    reset_memory : callable | None
        Optional reset_memory(memory, fresh_memory, reset_mask). Preserve all
        unselected lanes. Defaults select leading-B JAX leaves or host list
        entries. Empty memory is supported. Other layouts require this hook.
    execution : {"jax", "host"}
        Default "jax" requires pure fixed-shape functions. "host" receives
        NumPy inputs and may keep opaque Python memory outside compiled loops.
    components : tuple[dict, ...] | None
        Optional ordered JSON-compatible identities with nonempty name and
        optional version, code_digest, checkpoint and parameters_digest.
        IDs returned by apply index this table; None permits only -1 IDs.
    checkpoint : str | None
        Optional caller-supplied checkpoint identity; default None.

    Notes
    -----
    Every component influencing an action must honor that actor's information
    rights. A team function is not a security sandbox. Predicting hidden facts
    from allowed inputs is permitted; reading privileged state is separate.
    Memory for invalid lanes must remain unchanged, including custom layouts.
    This descriptor is frozen; referenced parameters and host objects are not.
    It is not a static container for changing weights and not a learner.

    Raises
    ------
    TypeError
        A method/hook is not callable or a component table is malformed.
    ValueError
        The name, execution mode or component identity is invalid.
    """

    name: str
    apply: SystemApply
    variables: PolicyTree = ()
    init: SystemInit | None = None
    reset_memory: SystemReset | None = None
    execution: PolicyExecution = "jax"
    components: tuple[dict[str, Any], ...] | None = None
    checkpoint: str | None = None
    _policies: tuple[Policy, ...] = field(default=(), init=False, repr=False)
    _shared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate host descriptor structure without executing a method."""
        if not isinstance(cast(object, self.name), str) or not self.name.strip():
            raise ValueError("system name must be a nonempty string")
        if not callable(self.apply):
            raise TypeError("system apply must be callable")
        if any(
            hook is not None and not callable(hook)
            for hook in (self.init, self.reset_memory)
        ):
            raise TypeError("system init and reset_memory must be callable or None")
        if self.execution not in ("jax", "host"):
            raise ValueError("system execution must be 'jax' or 'host'")
        if self.components is not None:
            if not isinstance(cast(object, self.components), tuple):
                raise TypeError("system components must be an ordered tuple")
            allowed = {
                "name",
                "version",
                "code_digest",
                "checkpoint",
                "parameters_digest",
            }
            for component in self.components:
                if (
                    not isinstance(cast(object, component), dict)
                    or not component.keys() <= allowed
                    or not isinstance(component.get("name"), str)
                    or not component["name"].strip()
                ):
                    raise ValueError(
                        "each component needs a nonempty name and known identity fields"
                    )
                canonical_digest_sha256(component)


def _execution(system: System) -> _SystemExecution:
    """Extract static callable structure, leaving all numerical values dynamic."""
    return _SystemExecution(
        system.apply,
        system.init,
        system.reset_memory,
        system.execution,
        tuple(policy.apply for policy in system._policies),
        system._shared,
        len(system.components or ()),
    )


def _adapter_marker(
    variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
) -> SystemOutput:
    """Reject direct adapter callbacks; init_systems/apply_systems own their layout."""
    del variables, memory, inputs, keys
    raise TypeError("use apply_systems to execute a Policy adapter")


def shared_policy(policy: Policy) -> System:
    """Adapt one scalar Policy across a team's five capacity slots.

    Parameters
    ----------
    policy : Policy
        Existing scalar policy with one parameter and initial-memory tree.

    Returns
    -------
    System
        Shared parameters, independent memory shaped (B,5,*leaf_shape), and
        component ID zero for active actors. JAX batches both lane and actor
        axes; host compatibility uses scalar calls. No action is chosen here.
        The initial memory template is copied during init_systems, not captured
        in a new callable. Apply through apply_systems, not the marker callback.

    Raises
    ------
    TypeError
        policy is not a Policy, or its copied component identity cannot be
        represented as JSON-compatible metadata.

    Notes
    -----
    Construction keeps the original variables and Policy reference; it does
    not copy weights or change the scalar callback. Each actor receives only
    its own permitted inputs and memory. The adapter applies to either fixed
    team and supports all valid roster sizes without a per-agent Policy list.
    """
    if not isinstance(cast(object, policy), Policy):
        raise TypeError("shared_policy requires a Policy")
    result = System(
        policy.name,
        _adapter_marker,
        variables=policy.variables,
        execution=policy.execution,
        components=(
            {
                "name": policy.name,
                **({"checkpoint": policy.checkpoint} if policy.checkpoint else {}),
            },
        ),
        checkpoint=policy.checkpoint,
    )
    object.__setattr__(result, "_policies", (policy,))
    object.__setattr__(result, "_shared", True)
    return result


def independent_policies(policies: Sequence[Policy]) -> System:
    """Bind 1-5 scalar Policies to active team slots in supplied order.

    Parameters
    ----------
    policies : sequence[Policy]
        One Policy for each active roster slot. Repeated classes remain distinct
        actors. All entries must use the same execution mode; parameter and
        memory trees may differ. Inputs are retained as an immutable tuple.

    Returns
    -------
    System
        A method with tuple parameters and tuple memory entries whose leaves
        have shape (B,*actor_memory_shape). Its roster must match in every
        participating lane. No class-based sorting or agent movement occurs.

    Raises
    ------
    ValueError
        Count is outside 1-5 or execution modes differ.
    TypeError
        An entry is not a Policy. Concrete incompatible rosters fail at setup;
        compiled callers must keep that roster precondition true.
    """
    entries = tuple(policies)
    if not 1 <= len(entries) <= 5:
        raise ValueError("independent_policies requires between one and five Policies")
    if any(not isinstance(cast(object, item), Policy) for item in entries):
        raise TypeError("independent_policies entries must be Policies")
    if len({item.execution for item in entries}) != 1:
        raise ValueError("independent Policies must use the same execution mode")
    result = System(
        " / ".join(item.name for item in entries),
        _adapter_marker,
        variables=tuple(item.variables for item in entries),
        execution=entries[0].execution,
        components=tuple(
            {
                "name": item.name,
                **({"checkpoint": item.checkpoint} if item.checkpoint else {}),
            }
            for item in entries
        ),
    )
    object.__setattr__(result, "_policies", entries)
    return result


def _team_index(team: int) -> int:
    """Require a static integer role 0/1; booleans are not routing identifiers."""
    if isinstance(team, bool) or not isinstance(team, Integral) or team not in (0, 1):
        raise ValueError("team must be the integer 0 or 1")
    return int(team)


def _batched(value: PolicyTree, native: bool) -> PolicyTree:
    """Add one lane axis to scalar numerical values; native values are unchanged."""

    def expand(leaf: Array) -> Array:
        """Give a scalar game's numerical leaf one leading environment axis."""
        return jnp.expand_dims(leaf, 0)

    return value if native else jax.tree.map(expand, value)


def system_inputs(
    observations: Observations, state: EnvironmentState, *, team: int = 0
) -> SystemInput:
    """Prepare one team's exact permitted inputs for env.policy_inputs.

    observations and state must describe the same pre-step epoch. team is a
    static 0/1 routing choice. Return a SystemInput with leading B even for a
    scalar game. Supplied source restrictions and Core masks remain unchanged.
    No opposite-team bank, privileged state, host transfer or history is built.
    """
    team = _team_index(team)
    native = state.episode_id.ndim != 0
    return cast(
        SystemInput,
        _prepare_system_inputs(
            _batched(observations, native),
            _batched(state.action_mask, native),
            _batched(state.config.agent_profile.active_mask, native),
            _batched(state.episode_start, native),
            _batched(~state.done.done, native),
            team=team,
        ),
    )


@jax.jit(static_argnames=("team",))
def _prepare_system_inputs(
    observations: Observations,
    masks: ActionMask,
    active: Array,
    starts: Array,
    valid: Array,
    *,
    team: int,
) -> SystemInput:
    """Fuse numerical input preparation for compiled and mixed host loops.

    All arguments already include B. Only team is static. Keep the needed
    compact fields dynamic without capturing the full environment or weights.
    This prevents separate eager kernels for every redaction/assembly operation.
    """
    actors = jax.vmap(build_team_actor_input, in_axes=(0, None))(observations, team)
    start = team * 5

    def take_team(value: Array) -> Array:
        """Select only this team's actor columns from a native mask."""
        return value[:, start : start + 5]

    return SystemInput(
        actors,
        jax.tree.map(take_team, masks),
        active[:, start : start + 5],
        starts,
        valid,
    )


def _require_int32(values: tuple[object, ...], name: str) -> None:
    """Reject non-int32 source arrays before JAX can narrow their values.

    NumPy int64/uint64 inputs must not become apparently valid actions or IDs
    when JAX x64 is disabled. Traced arrays expose dtype without a host read.
    """
    if any(getattr(value, "dtype", None) != np.dtype("int32") for value in values):
        raise TypeError(f"{name} must contain int32 arrays")


def join_system_actions(a: ActorAction, b: ActorAction, *, batched: bool) -> Action:
    """Join checked team heads without changing their submitted values.

    For native execution, a/b each contain int32 (B,5) arrays of equal shape.
    Scalar execution also accepts (5,) or (1,5), returning (10,). Native output
    remains (B,10), including B=1. Core alone decides whether values are legal.
    TypeError reports wrong records, shapes or dtypes. No input is changed.
    """
    normalized: list[ActorAction] = []
    for item in (a, b):
        if not isinstance(cast(object, item), ActorAction):
            raise TypeError("team actions must be ActorAction")
        _require_int32(tuple(item), "team action heads")
        arrays = tuple(jnp.asarray(value) for value in item)
        if not batched:
            arrays = tuple(
                value[0] if value.shape == (1, 5) else value for value in arrays
            )
        shape = arrays[0].shape
        expected = len(shape) == 2 and shape[1] == 5 if batched else shape == (5,)
        if not expected or any(
            value.shape != shape or value.dtype != jnp.int32 for value in arrays
        ):
            raise TypeError(
                "team action heads must be matching int32 (B,5) or scalar (5,) arrays"
            )
        normalized.append(ActorAction(*arrays))
    if normalized[0].move.shape != normalized[1].move.shape:
        raise TypeError("both teams must have the same action batch shape")
    if batched:
        return jax.vmap(build_joint_action_from_actor_actions)(*normalized)
    return build_joint_action_from_actor_actions(*normalized)


def system_step_data(
    before_state: EnvironmentState,
    actions: Action,
    step_result: tuple[Any, ...],
    *,
    system: int = 0,
) -> SystemStepData:
    """Project an explicit environment step onto one team's learning data.

    Parameters
    ----------
    before_state : EnvironmentState
        Exact state supplied to step, before any later reset.
    actions : Action
        Joint action submitted to that step; values are not replaced by Core's
        accepted action. Probabilities from SystemOutput refer to these values.
    step_result : tuple
        Complete (observations, state, reward, done, info) returned by that step.
        The caller must keep this result paired with its before_state/actions.
    system : int
        Team 0 (default) or 1. This selects reward ownership, not a model ID.

    Returns
    -------
    SystemStepData
        Batched action, reward, roster activity, pre-step ID and advancement.
        A live pre-step lane advances, including its terminal transition;
        padding does not. Rewards retain all five per-agent entries.

    Raises
    ------
    ValueError
        system is not 0/1 or the result is not the complete five-part tuple.
    """
    team = _team_index(system)
    if not isinstance(cast(object, step_result), tuple) or len(step_result) != 5:
        raise ValueError(
            "step_result must be the complete five-part environment result"
        )
    native = before_state.episode_id.ndim != 0
    start = team * 5
    submitted = _batched(actions, native)
    reward = _batched(step_result[2].rewards, native)
    return SystemStepData(
        ActorAction(*(value[:, start : start + 5] for value in submitted)),
        reward[:, start : start + 5],
        _batched(before_state.config.agent_profile.active_mask, native)[
            :, start : start + 5
        ],
        _batched(before_state.episode_id, native),
        _batched(~before_state.done.done, native),
    )


def _lane_roots(key: Array, episode_ids: Array, *, fold_explicit: bool) -> Array:
    """Normalize supported typed/legacy roots without losing their PRNG type.

    A scalar root folds episode IDs. B supplied roots retain lane order; init
    additionally folds current IDs so later resets can use saved lane roots.
    Shape checks use key representation metadata, never transfer numerical bits.
    """
    # Keep the existing JAX dtype predicate; its module omits a typing re-export.
    typed = jax.dtypes.issubdtype(  # pyright: ignore[reportPrivateImportUsage]
        key.dtype, jax.dtypes.prng_key
    )
    batch_shape = key.shape if typed else key.shape[:-1]
    if not typed and (key.dtype != jnp.uint32 or key.shape[-1:] != (2,)):
        raise ValueError("key must be a supported typed JAX key or uint32 legacy key")
    if batch_shape == ():
        return jax.vmap(jax.random.fold_in, in_axes=(None, 0))(key, episode_ids)
    if batch_shape != episode_ids.shape:
        raise ValueError("key must be one root or one key per environment")
    return jax.vmap(jax.random.fold_in)(key, episode_ids) if fold_explicit else key


def _tag_team(keys: Array, tag: int, team: int) -> Array:
    """Separate action/initialization domains and fixed team roles numerically."""

    def fold(key: Array) -> Array:
        """Fold the fixed domain first, then the assigned team, into one key."""
        return jax.random.fold_in(jax.random.fold_in(key, tag), team)

    return jax.vmap(fold)(keys)


def _action_keys(key: Array, episode_ids: Array, team: int) -> Array:
    """Use fresh caller action roots, episode normalization and a fixed team domain."""
    return _tag_team(
        _lane_roots(key, episode_ids, fold_explicit=False), 0x53414354, team
    )


def _initialization_keys(key: Array, episode_ids: Array, team: int) -> Array:
    """Derive fresh-memory keys from saved roots and IDs, not reset generations."""
    return _tag_team(
        _lane_roots(key, episode_ids, fold_explicit=True), 0x53494E49, team
    )


# Host orchestration reuses these small numerical programs instead of launching
# each fold operation separately. An outer compiled rollout can compose them.
_action_keys = cast(Callable[..., Array], jax.jit(_action_keys, static_argnums=2))
_initialization_keys = cast(
    Callable[..., Array],
    jax.jit(_initialization_keys, static_argnums=2),
)


def _adapter_template(system: System) -> PolicyTree:
    """Snapshot scalar Policy memory templates as dynamic numerical setup data."""
    if not system._policies:
        return ()
    templates = tuple(
        freeze_variables(policy.initial_carry) for policy in system._policies
    )
    return templates[0] if system._shared else templates


def _initial_memory(
    execution: _SystemExecution,
    variables: PolicyTree,
    template: PolicyTree,
    inputs: SystemInput,
    keys: Array,
) -> PolicyTree:
    """Prepare selected episode memory without sampling actions.

    Generic init receives a full stable batch and valid marks selected lanes.
    Adapter templates broadcast numerically; no per-step snapshot or hashing is
    involved. Host initializers must avoid sessions for unselected lanes.
    """
    count = inputs.valid.shape[0]
    if execution.policies:
        if execution.shared:

            def shared(leaf: Array) -> Array:
                """Give every lane and actor an independent scalar memory row."""
                return jnp.broadcast_to(leaf, (count, 5, *leaf.shape))

            return jax.tree.map(shared, template)

        def independent(leaf: Array) -> Array:
            """Give each lane a row from this actor's distinct memory template."""
            return jnp.broadcast_to(leaf, (count, *leaf.shape))

        return tuple(jax.tree.map(independent, tree) for tree in template)
    if execution.init is None:
        return ()
    return execution.init(variables, inputs, keys)


def _validate_roster(system: System, inputs: SystemInput) -> None:
    """Validate concrete independent rosters; compiled values remain preconditions."""
    _validate_adapter_roster(_execution(system), inputs.active_mask)


def _validate_adapter_roster(execution: _SystemExecution, active: Array) -> None:
    """Reject incompatible concrete active prefixes before choosing any action.

    Traced masks obey the same precondition without callbacks. Host validation
    reuses already delivered NumPy masks. Generic Systems and shared Policies
    accept all valid roster sizes and need no numerical check here.
    """
    if execution.policies and not execution.shared and not isinstance(active, Tracer):
        expected = np.arange(5) < len(execution.policies)
        if np.any(np.asarray(active) != expected):
            raise ValueError(
                "independent_policies length must match the active prefix roster"
            )


def _empty_trace(ids: Array) -> PolicyTrace:
    """Make the fixed trace shape invalid before any method chooses an action."""
    return PolicyTrace(
        ids,
        jnp.zeros_like(ids),
        jnp.zeros(ids.shape, jnp.bool_),
        jnp.full((ids.shape[0], 10), -1, jnp.int32),
    )


def init_systems(
    a: System,
    b: System,
    observations: Observations,
    state: EnvironmentState,
    key: Array,
    *,
    variables_a: PolicyTree = None,
    variables_b: PolicyTree = None,
) -> SystemState:
    """Initialize two methods without choosing an action.

    Parameters
    ----------
    a, b : System
        Fixed Team A and Team B methods. Each actor keeps its own information
        rights even when the method returns all five team actions together.
    observations : Observations
        Permitted compact inputs from the same epoch as state.
    state : EnvironmentState
        Current scalar or native state. A wholly new context needs this call.
    key : Array
        One typed JAX root shaped (), legacy uint32 root shaped (2,), or B
        lane roots shaped (B,) typed or (B, 2) legacy. Scalar games use B=1.
        Retained unchanged; memory keys fold episode ID, a fixed initialization
        tag and team, including for explicit lane roots. Reused IDs may repeat
        randomness while still creating fresh memory after a later reset.
    variables_a, variables_b : Any | None
        Explicit dynamic parameters; default None uses the matching descriptor's
        variables. Pass changing weights explicitly inside compiled functions.

    Returns
    -------
    SystemState
        Both memories, lifecycle bindings, saved roots and an invalid trace.
        JAX values can enter scan; opaque host memory stays in host loops.
        No apply function, action sampler or recording writer is called.

    Raises
    ------
    ValueError
        Key representation or lane count is unsupported, or a concrete
        independent-adapter roster does not match its Policy list.
    TypeError
        JAX rejects a malformed key or a non-numerical adapter memory template.
        Exceptions raised by a method's initializer propagate unchanged.

    Notes
    -----
    Generic host inputs become NumPy, while keys retain their JAX representation.
    Initializers receive valid=False for padding and must not create sessions
    there. JAX initializers keep fixed array structures; custom memory layouts
    need a matching reset_memory hook for later selected resets. This call does
    not invoke that reset hook. Adapter memory templates are snapshotted here;
    generic method parameters and host objects are not automatically copied.
    Save the returned state and pass each successor to apply_systems. A new
    unrelated environment context requires a new call instead of reusing it.
    """
    ids = _batched(state.episode_id, state.episode_id.ndim != 0)
    memories: list[PolicyTree] = []
    templates = (_adapter_template(a), _adapter_template(b))
    for team, (system, override) in enumerate(((a, variables_a), (b, variables_b))):
        inputs = system_inputs(observations, state, team=team)
        _validate_roster(system, inputs)
        execution = _execution(system)
        if system.execution == "host":
            inputs = jax.device_get(inputs)
        memories.append(
            _initial_memory(
                execution,
                system.variables if override is None else override,
                templates[team],
                inputs,
                _initialization_keys(key, ids, team),
            )
        )
    return SystemState(
        memories[0],
        memories[1],
        ids,
        _batched(state.reset_generation, state.episode_id.ndim != 0),
        key,
        _empty_trace(ids),
        templates,
    )


def _default_reset(
    memory: PolicyTree, fresh: PolicyTree, mask: Array, *, host: bool
) -> PolicyTree:
    """Select default memory layouts; never treat opaque host objects as arrays."""
    if isinstance(memory, tuple) and len(cast(tuple[PolicyTree, ...], memory)) == 0:
        if jax.tree.structure(memory) != jax.tree.structure(fresh):
            raise TypeError("memory must keep the structure established by init")
        return cast(PolicyTree, memory)
    if host:
        if (
            not isinstance(memory, list)
            or not isinstance(fresh, list)
            or len(cast(list[PolicyTree], memory)) != len(mask)
            or len(cast(list[PolicyTree], fresh)) != len(mask)
        ):
            raise TypeError("host memory needs a length-B list or custom reset_memory")
        return [
            new if bool(selected) else old
            for new, old, selected in zip(
                cast(list[PolicyTree], fresh),
                cast(list[PolicyTree], memory),
                mask,
                strict=True,
            )
        ]
    if jax.tree.structure(memory) != jax.tree.structure(fresh):
        raise TypeError("fresh memory must keep the current memory tree structure")
    for old, new in zip(jax.tree.leaves(memory), jax.tree.leaves(fresh), strict=True):
        if (
            not hasattr(old, "shape")
            or not old.shape
            or old.shape[0] != mask.shape[0]
            or old.shape != new.shape
            or old.dtype != new.dtype
        ):
            raise TypeError(
                "default JAX reset requires matching leading-B memory leaves"
            )
    return select_policy_carry(mask, fresh, memory)


def _reset_for_decision(
    execution: _SystemExecution,
    variables: PolicyTree,
    template: PolicyTree,
    memory: PolicyTree,
    inputs: SystemInput,
    keys: Array,
    reset_mask: Array,
) -> PolicyTree:
    """Reset only selected generations, skipping initialization when none changed."""

    def reset(old: PolicyTree) -> PolicyTree:
        """Create selected fresh states and merge them with continuing memories."""
        selected = inputs._replace(valid=reset_mask)
        fresh = _initial_memory(execution, variables, template, selected, keys)
        if execution.reset is not None:
            return execution.reset(old, fresh, reset_mask)
        return _default_reset(
            old,
            fresh,
            reset_mask,
            host=execution.execution == "host" and not execution.policies,
        )

    if execution.execution == "host":
        return reset(memory) if np.any(reset_mask) else memory

    def unchanged(old: PolicyTree) -> PolicyTree:
        """Leave the existing memory untouched when no reset was selected."""
        return old

    return cast(PolicyTree, jax.lax.cond(jnp.any(reset_mask), reset, unchanged, memory))


def _adapter_actions(
    execution: _SystemExecution,
    variables: PolicyTree,
    memory: PolicyTree,
    inputs: SystemInput,
    keys: Array,
) -> SystemOutput:
    """Batch scalar Policies while keeping their original parameter trees separate."""
    count = inputs.valid.shape[0]

    def actor_streams(key: Array) -> Array:
        """Split a generic team's key by local actor index; legacy keys bypass this."""
        return jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
            key, jnp.arange(5, dtype=jnp.uint32)
        )

    actor_keys = jax.vmap(actor_streams)(keys)
    if execution.shared:

        def lane(
            carry: PolicyTree, actors: ActorInput, masks: ActionMask, lane_keys: Array
        ) -> tuple[ActorAction, PolicyTree]:
            """Use the scalar Policy authority across one team's actor rows."""
            return _apply_actor_rows(
                execution.policies[0],
                variables,
                carry,
                actors,
                masks,
                lane_keys,
                execution=execution.execution,
            )

        if execution.execution == "jax":
            actions, updated = jax.vmap(lane)(
                memory, inputs.actors, inputs.action_mask, actor_keys
            )
        else:
            rows = []
            for index in range(count):
                row = jax.tree.map(
                    partial(_array_row, index=index),
                    (memory, inputs.actors, inputs.action_mask, actor_keys),
                )
                if inputs.valid[index]:
                    rows.append(lane(*row))
                else:
                    rows.append(
                        (
                            ActorAction(*(jnp.zeros(5, jnp.int32) for _ in range(3))),
                            row[0],
                        )
                    )
            actions, updated = jax.tree.map(_stack_arrays, *rows)
        ids = jnp.where(inputs.active_mask, 0, -1).astype(jnp.int32)
    else:
        heads: list[ActorAction] = []
        carries: list[PolicyTree] = []
        for index, apply in enumerate(execution.policies):
            actor = jax.tree.map(partial(_actor_column, index=index), inputs.actors)
            mask = jax.tree.map(partial(_actor_column, index=index), inputs.action_mask)
            if execution.execution == "host":
                rows = []
                for lane_index in range(count):
                    row = jax.tree.map(
                        partial(_array_row, index=lane_index),
                        (memory[index], actor, mask, actor_keys[:, index]),
                    )
                    if inputs.valid[lane_index]:
                        result, carry = apply(variables[index], *row)
                        rows.append((_checked_action(result), carry))
                    else:
                        rows.append(
                            (
                                ActorAction(
                                    *(jnp.array(0, jnp.int32) for _ in range(3))
                                ),
                                row[0],
                            )
                        )
                action, updated_one = jax.tree.map(_stack_arrays, *rows)
            else:
                action, updated_one = _apply_actor_rows(
                    apply,
                    variables[index],
                    memory[index],
                    actor,
                    mask,
                    actor_keys[:, index],
                    execution="jax",
                )
            heads.append(action)
            carries.append(updated_one)
        actions = ActorAction(
            *(
                jnp.pad(
                    jnp.stack([head[i] for head in heads], axis=1),
                    ((0, 0), (0, 5 - len(heads))),
                )
                for i in range(3)
            )
        )
        updated = tuple(carries)
        ids = jnp.where(inputs.active_mask, jnp.arange(5), -1).astype(jnp.int32)
    actions = ActorAction(
        *(jnp.where(inputs.active_mask, value, 0) for value in actions)
    )
    return SystemOutput(actions, updated, policy_ids=ids)


def _normal_output(
    result: PolicyTree, count: int, *, component_count: int, check_ids: bool
) -> SystemOutput:
    """Normalize accepted method results without guessing a bare third value."""
    if isinstance(result, SystemOutput):
        output = result
    elif isinstance(result, tuple) and len(cast(tuple[PolicyTree, ...], result)) in (
        2,
        3,
    ):
        parts = cast(tuple[PolicyTree, ...], result)
        output = SystemOutput(
            parts[0], parts[1], policy_ids=parts[2] if len(parts) == 3 else None
        )
    else:
        raise TypeError(
            "system apply must return two values, three values with policy IDs, "
            "or SystemOutput"
        )
    if not isinstance(cast(object, output.actions), ActorAction):
        raise TypeError("system actions must be ActorAction")
    _require_int32(tuple(output.actions), "system action heads")
    heads = tuple(jnp.asarray(value) for value in output.actions)
    if any(value.shape != (count, 5) or value.dtype != jnp.int32 for value in heads):
        raise TypeError("system action heads must be int32 (B,5)")
    if output.policy_ids is not None:
        _require_int32((output.policy_ids,), "policy_ids")
    if (
        check_ids
        and output.policy_ids is not None
        and not isinstance(output.policy_ids, Tracer)
    ):
        choices = np.asarray(output.policy_ids)
        if np.any((choices < -1) | (choices >= component_count)):
            raise ValueError("policy_ids must be -1 or index a declared component")
    ids = (
        jnp.full((count, 5), -1, jnp.int32)
        if output.policy_ids is None
        else jnp.asarray(output.policy_ids)
    )
    if ids.shape != (count, 5) or ids.dtype != jnp.int32:
        raise TypeError("policy_ids must be int32 (B,5)")
    return SystemOutput(
        ActorAction(*heads),
        output.next_memory,
        learning_outputs=output.learning_outputs,
        policy_ids=ids,
    )


def _preserve_padding(
    execution: _SystemExecution, old: PolicyTree, new: PolicyTree, valid: Array
) -> PolicyTree:
    """Protect known memory layouts without invoking a custom reset hook.

    Custom-layout methods own invalid-lane preservation. This function cannot
    undo mutation inside an opaque object and does not claim to be a sandbox.
    """
    if execution.reset is not None:
        return new
    return _default_reset(
        old, new, valid, host=execution.execution == "host" and not execution.policies
    )


def _jax_apply(
    execution: _SystemExecution,
    variables: PolicyTree,
    template: PolicyTree,
    memory: PolicyTree,
    inputs: SystemInput,
    keys: Array,
    init_keys: Array,
    reset_mask: Array,
) -> SystemOutput:
    """Apply one numerical team, with selected reset and no host callbacks."""
    memory = _reset_for_decision(
        execution, variables, template, memory, inputs, init_keys, reset_mask
    )
    result = (
        _adapter_actions(execution, variables, memory, inputs, keys)
        if execution.policies
        else execution.apply(variables, memory, inputs, keys)
    )
    output = _normal_output(
        result,
        inputs.valid.shape[0],
        component_count=execution.component_count,
        check_ids=not execution.policies,
    )
    return SystemOutput(
        output.actions,
        _preserve_padding(execution, memory, output.next_memory, inputs.valid),
        learning_outputs=output.learning_outputs,
        policy_ids=output.policy_ids,
    )


@partial(jax.jit, static_argnums=0, static_argnames=("team",))
def _mixed_jax_apply(
    execution: _SystemExecution,
    variables: PolicyTree,
    template: PolicyTree,
    memory: PolicyTree,
    observations: Observations,
    masks: ActionMask,
    active: Array,
    starts: Array,
    valid: Array,
    episode_ids: Array,
    key: Array,
    init_key: Array,
    reset_mask: Array,
    *,
    team: int,
) -> SystemOutput:
    """Prepare and apply the JAX team in one mixed-loop device call.

    execution and team are fixed call structure. All other values stay dynamic.
    observations/masks/active describe the same full B-lane decision; starts,
    valid, episode_ids and reset_mask have shape (B,). key and init_key retain
    the public root-or-B-key representation. variables, template and memory are
    the team's numerical trees, with their existing fixed shapes and dtypes.

    Reuse the input, RNG and method authorities inside one compiled boundary.
    Derive initialization keys only when an initializer exists and a live lane
    reset. Return one SystemOutput without reading device data on the host.
    Concrete independent-roster validation belongs to the caller before this
    helper; compiled numerical ID ranges retain their existing precondition.
    """
    inputs = cast(
        SystemInput,
        _prepare_system_inputs(observations, masks, active, starts, valid, team=team),
    )
    action_keys = _action_keys(key, episode_ids, team)
    init_keys = action_keys
    if execution.init is not None:

        def fresh_keys(_: None) -> Array:
            """Fold the saved initialization root only for selected new episodes."""
            return _initialization_keys(init_key, episode_ids, team)

        def unused_keys(_: None) -> Array:
            """Keep saved-key type and shape without deriving unused randomness."""
            suffix = () if jnp.issubdtype(init_key.dtype, jax.dtypes.prng_key) else (2,)
            return jnp.broadcast_to(init_key, (*episode_ids.shape, *suffix))

        init_keys = cast(
            Array,
            jax.lax.cond(jnp.any(reset_mask), fresh_keys, unused_keys, operand=None),
        )
    return _jax_apply(
        execution,
        variables,
        template,
        memory,
        inputs,
        action_keys,
        init_keys,
        reset_mask,
    )


def _padding_output(memory: PolicyTree, count: int) -> SystemOutput:
    """Return a host no-call result with unchanged memory and invalid component IDs."""
    zero = jnp.zeros((count, 5), jnp.int32)
    return SystemOutput(
        ActorAction(zero, zero, zero),
        memory,
        policy_ids=jnp.full((count, 5), -1, jnp.int32),
    )


def _host_apply(
    execution: _SystemExecution,
    variables: PolicyTree,
    template: PolicyTree,
    memory: PolicyTree,
    inputs: SystemInput,
    keys: Array,
    init_keys: Array,
    reset_mask: Array,
) -> SystemOutput:
    """Transfer only the host team's inputs and keep its full lane order."""
    inputs, selected = jax.device_get((inputs, reset_mask))
    _validate_adapter_roster(execution, inputs.active_mask)
    memory = _reset_for_decision(
        execution, variables, template, memory, inputs, init_keys, selected
    )
    if not np.any(inputs.valid):
        return _padding_output(memory, len(inputs.valid))
    result = (
        _adapter_actions(execution, variables, memory, inputs, keys)
        if execution.policies
        else execution.apply(variables, memory, inputs, keys)
    )
    output = _normal_output(
        result,
        inputs.valid.shape[0],
        component_count=execution.component_count,
        check_ids=not execution.policies,
    )
    return SystemOutput(
        output.actions,
        _preserve_padding(execution, memory, output.next_memory, inputs.valid),
        learning_outputs=output.learning_outputs,
        policy_ids=output.policy_ids,
    )


def apply_systems(
    a: System,
    b: System,
    memory: SystemState,
    observations: Observations,
    state: EnvironmentState,
    key: Array,
    *,
    variables_a: PolicyTree = None,
    variables_b: PolicyTree = None,
) -> tuple[Action, SystemState, tuple[PolicyTree, PolicyTree]]:
    """Choose both teams' actions once and pass learning values to the caller.

    Parameters
    ----------
    a, b : System
        Fixed Team A/B descriptors used to initialize this memory.
    memory : SystemState
        Latest returned memory from the same environment context.
    observations : Observations
        Compact permitted inputs matching state at the current decision.
    state : EnvironmentState
        Scalar/native pre-step state. Reset generations trigger selected fresh
        memory; deaths, respawns and recording ID changes alone do not.
    key : Array
        Fresh typed/legacy root or B lane action keys for this decision. Roots
        fold episode IDs; explicit lane keys retain their mapping. Both then
        fold a fixed action tag and team. No local step is additionally folded.
        Initialization uses the separately saved roots, never these action keys.
    variables_a, variables_b : Any | None
        Explicit parameters override descriptor values. Pass changing weights
        here inside jit; callables and numerical tree structures stay stable.

    Returns
    -------
    tuple[Action, SystemState, tuple[Any, Any]]
        Joint submitted action, successor memory/trace, and separate Team A/B
        learning outputs. Absent outputs are (). Outputs are not copied, saved,
        placed in recurrent memory or obtained with another method call.
        Scalar actions have (10,) heads; native actions have (B,10).

    Raises
    ------
    TypeError
        Action/result structure or memory reset layout is invalid.
    ValueError
        Concrete component IDs or independent roster layouts are invalid.
        Numeric ranges inside compiled execution remain caller preconditions;
        invalid IDs are preserved for later publication checks, never repaired.

    Notes
    -----
    JAX systems compose with jit/scan without host callbacks. Mixed execution
    runs outside jit: the host gets one stable NumPy batch and JAX keys, while
    its opponent stays compiled and batched. All-invalid host calls are skipped.
    Exceptions propagate before the caller can step the environment. Opaque
    external mutations cannot be rolled back by this helper. A returned action
    does not itself advance the environment.
    """
    native = state.episode_id.ndim != 0
    ids = _batched(state.episode_id, native)
    generations = _batched(state.reset_generation, native)
    if memory.episode_id.shape != ids.shape:
        raise ValueError("SystemState and environment must have the same lane count")
    valid = _batched(~state.done.done, native)
    reset = (generations != memory.reset_generation) & valid
    outputs: list[SystemOutput] = []
    mixed = a.execution == "host" or b.execution == "host"
    # One small host check can skip every expanded input/ID transfer on padding.
    if mixed:
        host_valid, host_reset = jax.device_get((valid, reset))
        host_has_work = bool(np.any(host_valid))
        needs_initialization = bool(np.any(host_reset))
    else:
        host_has_work = True
        needs_initialization = True
    for team, (system, old, override) in enumerate(
        ((a, memory.team_a, variables_a), (b, memory.team_b, variables_b))
    ):
        if system.execution == "host" and not host_has_work:
            outputs.append(_padding_output(old, ids.shape[0]))
            continue
        execution = _execution(system)
        variables = system.variables if override is None else override
        if mixed and system.execution == "jax":
            active = _batched(state.config.agent_profile.active_mask, native)
            if needs_initialization:
                _validate_adapter_roster(execution, active[:, team * 5 : team * 5 + 5])
            outputs.append(
                cast(
                    SystemOutput,
                    _mixed_jax_apply(
                        execution,
                        variables,
                        memory.adapter_templates[team],
                        old,
                        _batched(observations, native),
                        _batched(state.action_mask, native),
                        active,
                        _batched(state.episode_start, native),
                        valid,
                        ids,
                        key,
                        memory.init_key,
                        reset,
                        team=team,
                    ),
                )
            )
            continue
        inputs = system_inputs(observations, state, team=team)
        if system.execution == "jax":
            _validate_adapter_roster(execution, inputs.active_mask)
        function = _host_apply if system.execution == "host" else _jax_apply
        action_keys = _action_keys(key, ids, team)
        # Adapters broadcast stored templates; they never consume init RNG.
        # On mixed steps without resets, the unused argument can reuse keys.
        init_keys = (
            _initialization_keys(memory.init_key, ids, team)
            if execution.init is not None and needs_initialization
            else action_keys
        )
        output = function(
            execution,
            variables,
            memory.adapter_templates[team],
            old,
            inputs,
            action_keys,
            init_keys,
            reset,
        )
        outputs.append(output)
    valid = _batched(~state.done.done, native)
    trace = PolicyTrace(
        ids,
        _batched(state.core_state.step_count - state.initial_step_count, native),
        valid,
        jnp.concatenate(
            (cast(Array, outputs[0].policy_ids), cast(Array, outputs[1].policy_ids)),
            axis=1,
        ),
    )
    updated = SystemState(
        outputs[0].next_memory,
        outputs[1].next_memory,
        ids,
        generations,
        memory.init_key,
        trace,
        memory.adapter_templates,
    )
    return (
        join_system_actions(outputs[0].actions, outputs[1].actions, batched=native),
        updated,
        (outputs[0].learning_outputs, outputs[1].learning_outputs),
    )


def apply_policy_batch(
    apply_a: PolicyApply,
    apply_b: PolicyApply,
    variables_a: PolicyTree,
    variables_b: PolicyTree,
    carry_a: PolicyTree,
    carry_b: PolicyTree,
    observations: Observations,
    action_mask: ActionMask,
    actor_keys: Array,
    valid: Array,
    *,
    execution_a: PolicyExecution = "jax",
    execution_b: PolicyExecution = "jax",
) -> tuple[Action, PolicyTree, PolicyTree]:
    """Apply existing Policies over B lanes with their explicit actor keys.

    Parameters
    ----------
    apply_a, apply_b : PolicyApply
        Team A and Team B scalar callbacks. Each receives variables, one actor's
        memory, permitted ActorInput, ActionMask and its supplied random key.
        It returns (ActorAction, next_memory).
    variables_a, variables_b : Any
        Separate team parameter trees, shared across their actor calls. These
        are dynamic numerical inputs on the compiled path, without a B axis
        added by this helper. Callers own any evaluation snapshot.
    carry_a, carry_b : Any
        Team memory trees with leading axes (B, 5). Remaining leaf shapes and
        dtypes stay fixed for a compiled callback. Empty tuples are supported.
    observations : Observations
        Compact inputs for B games at one decision. Observation leaves lead
        with (B, 10); source permissions have shape (B, 10, 10).
    action_mask : ActionMask
        Core's masks from that same decision, with leading axes (B, 10).
    actor_keys : Array
        Supplied typed keys shaped (B, 10), or legacy uint32 keys (B, 10, 2).
        Global slot order is Team A then Team B. No keys are split or remapped.
    valid : Array
        Boolean (B,) live-game flags. False lanes keep their original memory.
    execution_a, execution_b : {"jax", "host"}
        Defaults are "jax". Two JAX Policies run in compiled-compatible
        batches. Either "host" selects the legacy ordered scalar callback path,
        which must stay outside jit and transfers permitted inputs to the host.

    Returns
    -------
    tuple[Action, Any, Any]
        Joint submitted Action with int32 (B, 10) fields, then updated Team A
        and Team B memory trees. Memory layouts remain unchanged. On the host
        path, finished lanes have zero actions and make no callback requests.
        The compiled path may compute padded actions; its caller must ignore
        them and retain the live-game mask.

    Raises
    ------
    TypeError
        A callback returns malformed scalar ActorAction fields. Callback errors
        propagate unchanged; no environment transition is performed here.

    Notes
    -----
    Both teams receive the same pre-step epoch. The legacy host path visits
    ascending live lanes, then Team A actors 0-4 and Team B actors 5-9. Keep
    this order for existing providers with observable call side effects.

    This compatibility route keeps existing callback counts, keys and result
    shapes. It creates no SystemState, traces or initialization keys. Generic
    host Systems use their separate one-full-batch method contract.
    """
    if execution_a == execution_b == "jax":
        action, next_a, next_b = jax.vmap(
            apply_policies, in_axes=(None, None, None, None, 0, 0, 0, 0, 0)
        )(
            apply_a,
            apply_b,
            variables_a,
            variables_b,
            carry_a,
            carry_b,
            observations,
            action_mask,
            actor_keys,
        )
        return (
            action,
            select_policy_carry(valid, next_a, carry_a),
            select_policy_carry(valid, next_b, carry_b),
        )
    rows: list[tuple[Action, PolicyTree, PolicyTree]] = []
    for index, live in enumerate(np.asarray(valid)):

        def row(value: Array, index: int = index) -> Array:
            """Read one numerical Policy lane; opaque Systems never use this."""
            return value[index]

        memory_a, memory_b = jax.tree.map(row, (carry_a, carry_b))
        if live:
            rows.append(
                apply_policies(
                    apply_a,
                    apply_b,
                    variables_a,
                    variables_b,
                    memory_a,
                    memory_b,
                    jax.tree.map(row, observations),
                    jax.tree.map(row, action_mask),
                    actor_keys[index],
                    execution_a=execution_a,
                    execution_b=execution_b,
                )
            )
        else:
            rows.append(
                (
                    Action(*(jnp.zeros(10, jnp.int32) for _ in range(3))),
                    memory_a,
                    memory_b,
                )
            )

    def stack(*values: Array) -> Array:
        """Restore numerical environment axes after ordered scalar host calls."""
        return jnp.stack(values)

    return cast(tuple[Action, PolicyTree, PolicyTree], jax.tree.map(stack, *rows))
