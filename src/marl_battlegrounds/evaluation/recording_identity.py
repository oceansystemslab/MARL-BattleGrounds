"""Describe recorded methods once at the shared host registration boundary.

Policy evaluation, raw System recording and replay metadata use these helpers.
Descriptions contain labels and available numerical/code evidence, never live
memory, sessions or learning outputs. A content hash identifies these recorded
facts; missing code or external-state evidence does not become a verified
controller merely because its description has a hash. No method is executed.
"""

import json
import marshal
from collections.abc import Mapping
from hashlib import sha256
from numbers import Number
from types import CodeType, FunctionType
from typing import TYPE_CHECKING, cast

import jax
import numpy as np

from marl_battlegrounds.evaluation.models import canonical_digest_sha256

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.policy_execution import Policy


def tree_digest(tree: object) -> str:
    """Hash numerical tree structure, leaf shapes, dtypes and bytes on the host.

    Parameters
    ----------
    tree : Any
        A numerical JAX tree, including an empty tree. Object/string leaves are
        not supported. Callers supply the actual snapshot they wish to identify.

    Returns
    -------
    str
        Full lowercase SHA256, preserving the existing frozen Policy digest.

    Raises
    ------
    TypeError
        A leaf is not a numerical array or scalar.

    Notes
    -----
    This reads device leaves and may synchronize. It neither freezes future
    execution nor changes the input. Do not call it on each action decision.
    """
    leaves = jax.tree.leaves(tree)
    if any(
        not isinstance(value, (jax.Array, np.ndarray, np.generic, Number))
        for value in leaves
    ):
        raise TypeError("recording digests require numerical tree leaves")
    structure = cast(object, jax.tree.structure(tree))
    digest = sha256(str(structure).encode())
    for value in jax.device_get(leaves):
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise TypeError("recording digests require numerical tree leaves")
        header = f"{array.dtype.str}:{array.shape}".encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(np.ascontiguousarray(array).data)
    return digest.hexdigest()


def callable_name(function: object) -> str:
    """Return a callable's module and qualified name, or its type's name.

    This is a display reference, not proof of code or provider identity. The
    callable is not executed and no state attached to it is serialized.
    """
    owner = type(function)
    module = getattr(function, "__module__", owner.__module__)
    name = getattr(function, "__qualname__", owner.__qualname__)
    return f"{module}.{name}"


def policy_description(
    team: Policy,
    variables: object,
    initial_carry: object,
    *,
    include_digests: bool,
) -> dict[str, object]:
    """Describe the explicitly frozen Policy snapshot used by an evaluator.

    Parameters
    ----------
    team : Policy
        Policy providing its name, callable, execution mode and checkpoint.
    variables : Any
        Actual numerical variables already frozen by the caller for this pass.
    initial_carry : Any
        Actual numerical actor-memory template already frozen by the caller.
    include_digests : bool
        True computes content/controller identities. False skips hashing and
        stores None in those fields.

    Returns
    -------
    dict[str, object]
        The existing Policy description, with variables_frozen=True and unchanged
        digest meanings. Inputs and the Policy are not changed.

    Raises
    ------
    TypeError
        Enabled numerical hashing encounters an unsupported leaf.

    Notes
    -----
    Host-only. This function records the caller's explicit freezing contract;
    it does not freeze values, verify checkpoint names or execute a method.
    """
    from marl_battlegrounds.evaluation.policy_execution import controller_identity

    return {
        "name": team.name,
        "checkpoint": team.checkpoint,
        "execution": team.execution,
        "variables_frozen": True,
        "callable_name": callable_name(team.apply),
        "controller_identity": controller_identity(team) if include_digests else None,
        "variables_digest": tree_digest(variables) if include_digests else None,
        "initial_carry_digest": tree_digest(initial_carry) if include_digests else None,
    }


def _numerical_evidence(value: object) -> str | None:
    """Hash supported numerical leaves, leaving opaque provider content unknown."""
    try:
        return tree_digest(value)
    except TypeError, ValueError:
        return None


def _portable_code(code: CodeType) -> CodeType:
    """Remove file locations from Python code evidence, including nested functions.

    File paths and line positions do not identify behavior. Keep bytecode,
    constants, names and exception tables. The result is only used for hashing;
    it is never executed or substituted for a method's real code.
    """
    return code.replace(
        co_filename="",
        co_firstlineno=0,
        co_linetable=b"",
        co_consts=tuple(
            _portable_code(value) if isinstance(value, CodeType) else value
            for value in code.co_consts
        ),
    )


def _callable_evidence(function: object) -> dict[str, object] | None:
    """Describe a hook without calling it or traversing sessions and closures.

    A Python bytecode digest covers that function's code only. Closure values,
    mutable globals and provider state remain unverified. Non-Python callables
    keep their display name with unknown code content.
    """
    if function is None:
        return None
    python_function = isinstance(function, FunctionType)
    return {
        "callable_name": callable_name(function),
        "code_digest": sha256(
            marshal.dumps(_portable_code(function.__code__))
        ).hexdigest()
        if python_function
        else None,
        "code_scope": "python_function_bytecode" if python_function else "unknown",
        "closure_content": "unknown"
        if not python_function or function.__closure__
        else "none",
        "defaults_digest": _numerical_evidence(
            (function.__defaults__, function.__kwdefaults__)
        )
        if python_function
        else None,
        "external_state": "unknown",
    }


def _json_description(value: Mapping[str, object]) -> dict[str, object]:
    """Copy JSON-only registration facts; reject arbitrary objects and NaN."""
    return cast(dict[str, object], json.loads(json.dumps(dict(value), allow_nan=False)))


def normalize_system_registration(
    value: object, *, phase: str, frozen: bool = False
) -> tuple[str, dict[str, object]]:
    """Return a stable registration ID and truthful Policy/System metadata.

    Parameters
    ----------
    value : System, Policy, mapping or str
        Live method descriptor, existing serialized descriptor, or legacy label.
        Mapping values must be JSON-compatible. Labels carry no executable proof.
    phase : str
        Recording phase. Exactly "training" records parameters as evolving,
        even when a checkpoint label is present.
    frozen : bool
        False by default. True is the runner's explicit guarantee that supplied
        numerical parameters are the frozen execution snapshot. A serialized
        Policy description may carry that existing guarantee itself. Raw
        registration never freezes the method.

    Returns
    -------
    tuple[str, dict[str, object]]
        Full canonical SHA256 and a fresh JSON-ready description. Ordered
        components preserve team-local IDs. parameter_status is frozen,
        evolving or unknown. Unknown evidence stays None or explicitly unknown.

    Raises
    ------
    TypeError
        value is unsupported or serialized metadata contains non-JSON objects.
    ValueError
        Serialized metadata contains invalid numerical values.

    Notes
    -----
    Host-only and intended once per pass registration. Hashing may transfer
    numerical arrays. Callbacks are never invoked. Live recurrent memory,
    provider sessions and learning outputs are never inspected or serialized.
    Adapter initial-memory templates are numerical setup evidence, not live
    memory. An ID identifies available facts, not unknown external behavior.
    """
    from marl_battlegrounds.evaluation.policy_execution import Policy, System

    if isinstance(value, System):
        parameters = _numerical_evidence(value.variables)
        registration: dict[str, object] = {
            "kind": "system",
            "name": value.name,
            "checkpoint": value.checkpoint,
            "execution": value.execution,
            "components": _json_description({"items": value.components or ()})["items"],
            "variables_digest": parameters,
            "parameter_evidence": "registration_snapshot" if parameters else "unknown",
            "hooks": {
                "apply": _callable_evidence(value.apply),
                "init": _callable_evidence(value.init),
                "reset_memory": _callable_evidence(value.reset_memory),
            },
            "adapter_kind": ("shared" if value._shared else "independent")  # pyright: ignore[reportPrivateUsage]
            if value._policies  # pyright: ignore[reportPrivateUsage]
            else None,
            "adapter_policies": [],
        }
        adapter_policies: list[dict[str, object]] = []
        for entry in value._policies:  # pyright: ignore[reportPrivateUsage]
            descriptor = policy_description(
                entry, entry.variables, entry.initial_carry, include_digests=True
            )
            descriptor["variables_frozen"] = bool(frozen and phase != "training")
            descriptor["apply"] = _callable_evidence(entry.apply)
            adapter_policies.append(descriptor)
        registration["adapter_policies"] = adapter_policies
        frozen = frozen and parameters is not None
    elif isinstance(value, Policy):
        parameters = _numerical_evidence(value.variables)
        template = _numerical_evidence(value.initial_carry)
        from marl_battlegrounds.evaluation.policy_execution import controller_identity

        registration = {
            "kind": "policy",
            "name": value.name,
            "checkpoint": value.checkpoint,
            "execution": value.execution,
            "callable_name": callable_name(value.apply),
            "controller_identity": controller_identity(value),
            "variables_digest": parameters,
            "initial_carry_digest": template,
            "components": [{"name": value.name}],
            "hooks": {"apply": _callable_evidence(value.apply)},
            "parameter_evidence": "registration_snapshot" if parameters else "unknown",
        }
        frozen = frozen and parameters is not None
    elif isinstance(value, Mapping):
        registration = _json_description(cast(Mapping[str, object], value))
        registration.setdefault("kind", "policy")
        registration.setdefault(
            "components", [{"name": registration.get("name", "unknown")}]
        )
        frozen = frozen or registration.get("variables_frozen") is True
    elif isinstance(value, str):
        registration = {"kind": "unknown", "name": value, "components": []}
        frozen = False
    else:
        raise TypeError(
            "recorded policies must be Systems, Policies, descriptions or names"
        )
    components = registration.get("components")
    if not isinstance(components, list) or any(
        not isinstance(component, dict)
        or not isinstance(component.get("name"), str)
        or not component["name"].strip()
        for component in cast(list[object], components)
    ):
        raise ValueError(
            "recorded components must be an ordered list with nonempty names"
        )
    status = "evolving" if phase == "training" else "frozen" if frozen else "unknown"
    registration["parameter_status"] = status
    registration["variables_frozen"] = status == "frozen"
    return canonical_digest_sha256(registration), registration
