"""Prepare frozen method values and memory roots for evaluation runners.

These host setup helpers preserve original Policy descriptions while exposing
stable callables and dynamic values to the shared System execution authority.
They never choose an action, write a recording, or define random stream numbers.
The evaluator owns scheduled keys and configuration validation. Opaque provider
objects remain caller-owned; numerical snapshots do not freeze external services.
"""

# Runner helpers deliberately share private execution descriptors and adapters.
# pyright: reportPrivateUsage=false

from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyTree,
    System,
    SystemState,
    _adapter_template,
    _check_weights,
    _composition,
    _CompositionVariables,
    _execution,
    _SystemExecution,
    _validate_adapter_roster,
    freeze_variables,
    policy,
    shared_policy,
)


def _freeze_host_variables(variables: PolicyTree) -> PolicyTree:
    """Copy numerical host storage while retaining opaque leaves by reference.

    variables is a method-owned PyTree. NumPy arrays become independent read-only
    arrays; JAX arrays and immutable scalars need no copy. Other leaves, including
    provider sessions, stay untouched. New tree containers isolate ordinary
    dict/list replacement, but no guarantee is made about opaque object mutation.
    This host setup operation performs no device transfer or method call.
    """

    def snapshot(value: object) -> object:
        """Copy only mutable NumPy leaves; never ask an opaque leaf for an array."""
        if isinstance(value, np.ndarray):
            result = cast(np.ndarray[Any, Any], value).copy()
            result.flags.writeable = False
            return result
        return value

    return jax.tree.map(snapshot, variables)


def freeze_evaluation_method(value: System | Policy | str) -> System | Policy:
    """Snapshot evaluation parameters without losing the original descriptor kind.

    Parameters
    ----------
    value : System | Policy | str
        A researcher method, scalar Policy, or supported built-in Policy name.
        Policy parameters and templates must remain numerical. Generic host
        Systems may also contain opaque provider leaves.

    Returns
    -------
    System | Policy
        A fresh descriptor with frozen numerical values. A bare Policy remains a
        Policy so the runner can preserve its existing saved description/digests.
        A System keeps its hooks, adapter kind/order and copied component metadata.
        Adapter Policy descriptions refer to the same parameter snapshot actually
        used by the System. The original descriptor and values are not edited.

    Raises
    ------
    TypeError
        value has another type or a numerical method/template contains an
        unsupported leaf. Component metadata must keep its valid JSON structure.
    ValueError
        A built-in name or reconstructed descriptor is invalid.

    Notes
    -----
    Host setup only. NumPy numerical values are copied; existing immutable JAX
    arrays are reused. Generic host NumPy parameters stay on the host and become
    read-only copies. Opaque leaves are never array-converted, deep-copied or
    serialized. The caller must keep their external behavior fixed. Initializers
    and action methods are not invoked; recorded identity is the writer's job.
    """
    if isinstance(value, str):
        value = policy(value)
    if not isinstance(cast(object, value), (System, Policy)):
        raise TypeError(
            "evaluation methods must be Systems, Policies or built-in names"
        )
    transform = (
        _freeze_host_variables
        if isinstance(value, System)
        and value.execution == "host"
        and not value._policies
        else freeze_variables
    )
    if isinstance(value, Policy):
        return replace(
            value,
            variables=transform(value.variables),
            initial_carry=transform(value.initial_carry),
        )
    variables = transform(value.variables)
    result = replace(value, variables=variables, components=deepcopy(value.components))
    if value._policies:
        entries = tuple(
            replace(
                entry,
                variables=variables if value._shared else variables[index],
                initial_carry=transform(entry.initial_carry),
            )
            for index, entry in enumerate(value._policies)
        )
        object.__setattr__(result, "_policies", entries)
        object.__setattr__(result, "_shared", value._shared)
    return result


def prepare_evaluation_system(
    value: System | Policy,
) -> tuple[_SystemExecution, PolicyTree, PolicyTree]:
    """Separate one frozen method's stable call structure and dynamic values.

    Parameters
    ----------
    value : System | Policy
        Method already snapshotted by freeze_evaluation_method. A bare Policy is
        adapted only for execution; retain value for its original saved identity.

    Returns
    -------
    tuple[_SystemExecution, Any, Any]
        Stable execution descriptor, explicit parameters, and numerical adapter
        memory template. Generic methods use () for the template. Only the first
        item belongs in static JAX arguments. Values and templates stay dynamic.

    Raises
    ------
    TypeError
        value is not a System/Policy or an adapter template is not numerical.

    Notes
    -----
    Call once at runner setup. No initializer or action method runs, and no
    closure captures changing weights. Equivalent descriptors with unchanged
    callables/layout compare equally across separate parameter snapshots.
    """
    if isinstance(value, Policy):
        system = shared_policy(value)
    elif isinstance(cast(object, value), System):
        system = value
    else:
        raise TypeError("a prepared evaluation method must be a System or Policy")
    return _execution(system), system.variables, _adapter_template(system)


def replace_initialization_roots(
    memory: SystemState, roots: Array, reset_mask: Array
) -> SystemState:
    """Replace only refill lanes' saved roots before their next decision.

    Parameters
    ----------
    memory : SystemState
        Evaluation memory initialized with B lane roots, not one scalar root.
        Existing memories and reset bindings describe the previous decision.
    roots : Array
        New scheduled stream-3 roots with exactly the saved key shape and dtype:
        typed (B,) or legacy uint32 (B,2). Values in unselected lanes are ignored.
    reset_mask : Array
        Boolean (B,) selecting lanes the environment actually reset.

    Returns
    -------
    SystemState
        New roots for selected lanes; every other field is unchanged. The shared
        application helper resets memory on the next generation-changing decision.
        No method, initializer, reset hook or host transfer occurs here.

    Raises
    ------
    ValueError
        Mask, root shape or root dtype differs from this numerical context.

    Notes
    -----
    This helper composes with jit and scan. It does not allocate episode IDs or
    validate key values. Call it only for a genuine same-context partial refill;
    unrelated execution contexts need fresh initialization.
    """
    old = memory.init_key
    typed = jax.dtypes.issubdtype(  # pyright: ignore[reportPrivateImportUsage]
        old.dtype, jax.dtypes.prng_key
    )
    expected = memory.episode_id.shape + (() if typed else (2,))
    if (
        old.shape != expected
        or roots.shape != expected
        or roots.dtype != old.dtype
        or (not typed and old.dtype != jnp.uint32)
        or reset_mask.shape != memory.episode_id.shape
        or reset_mask.dtype != jnp.bool_
    ):
        raise ValueError("refill roots and mask must match the evaluation lane keys")
    mask = reset_mask if typed else reset_mask[:, None]
    return memory._replace(init_key=jnp.where(mask, roots, old))


def validate_evaluation_rosters(
    first: _SystemExecution,
    second: _SystemExecution,
    config: EnvConfig,
    *,
    variables_a: PolicyTree = None,
    variables_b: PolicyTree = None,
) -> None:
    """Check every possible member's roster before the runner opens a writer.

    first and second describe Team A/B. config is one exact scalar episode
    configuration already checked by the setup authority. variables_a and
    variables_b are the corresponding frozen parameters; compositions require
    them, while direct adapters and generic Systems may omit them.

    Independent adapters must match the active prefix when used alone. Nested
    adapters must contain every physical slot they control. Team selectors must
    cover each incoming active slot exactly once, including repeated classes.
    Every positive-share pool member is checked, regardless of a future draw.
    Generic Systems and shared Policies have no extra roster-size rule.

    Host setup only: copy the small roster/selector/share arrays, never method
    parameters or memory. No initializer, action method or resource scope runs.
    Invalid shapes, coverage or adapter lengths raise ValueError before any
    recording mutation. Call once for each scheduled configuration, not per step.
    """
    active = np.asarray(config.agent_profile.active_mask)
    if active.shape != (10,) or active.dtype != np.bool_:
        raise ValueError("evaluation roster checks require a scalar ten-agent config")
    classes = np.asarray(config.agent_profile.class_ids)

    def validate(
        execution: _SystemExecution,
        variables: PolicyTree,
        activity: np.ndarray[Any, Any],
        class_ids: np.ndarray[Any, Any],
        controlled: np.ndarray[Any, Any] | None = None,
    ) -> None:
        """Check one member and recurse only through slots it may control.

        activity and class_ids retain five physical slots. controlled=None means
        the top-level method; nested methods receive their exact owned subset.
        variables is the member's frozen tree, never inspected for generic code.
        """
        composition = _composition(execution)
        if composition is None:
            _validate_adapter_roster(
                execution,
                cast(Array, activity),
                controlled=cast(Array | None, controlled),
            )
            return
        if not isinstance(variables, _CompositionVariables):
            raise ValueError("composition roster checks require composition variables")
        owned = activity if controlled is None else activity & controlled
        if composition.pooled:
            weights = _check_weights(variables.weights)
            masks = tuple(owned & (weight > 0) for weight in weights)
        else:
            slots = np.asarray(variables.slots)
            masks = tuple(
                owned
                & (
                    row[:5]
                    | np.any(row[5:] & (class_ids[:, None] == np.arange(1, 6)), axis=-1)
                )
                for row in slots
            )
            coverage = np.sum(np.stack(masks), axis=0)
            if np.any(coverage[owned] != 1):
                raise ValueError(
                    "team slots must cover each controlled active slot exactly once"
                )
        for member, values, mask in zip(
            composition.members, variables.members, masks, strict=True
        ):
            if np.any(mask):
                validate(member, values, activity, class_ids, mask)

    validate(first, variables_a, active[:5], classes[:5])
    validate(second, variables_b, active[5:], classes[5:])
