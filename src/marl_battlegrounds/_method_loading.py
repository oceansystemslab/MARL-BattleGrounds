"""Resolve method references: built-in names, exported actors and factories.

Command factories and tournament loaders share the ``module:function`` grammar
through installed_callable. load_method is the one owner of the wider
reference grammar used for training opponents: a built-in Policy name, an
exported actor directory, or a ``module:function`` factory. Importing this
module uses only the standard library; load_method imports the evaluation and
training code it needs only when called. Calling a selected function remains
the caller's responsibility; imported researcher code may have its own side
effects.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.policy_execution import Policy, System


def installed_callable(reference: str) -> Callable[..., object]:
    """Import a trusted ``module:function`` and return it without calling it.

    The attribute must be one module-level name, not a dotted traversal.
    Invalid names and noncallable attributes raise ValueError. Import and
    attribute errors retain their original cause. No files are downloaded and
    no path is added to Python's import search path.
    """
    if not isinstance(cast(object, reference), str) or reference.count(":") != 1:
        raise ValueError("Controller factory/loader must use module:function")
    module_name, attribute = reference.split(":")
    if not module_name or not attribute or "." in attribute:
        raise ValueError(
            "Controller factory/loader must name one module-level callable"
        )
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise ValueError("Controller factory/loader is not callable")
    return value


def load_factory(reference: str) -> Policy | System:
    """Import a ``module:function`` factory, call it once, and check what it returns.

    Parameters
    ----------
    reference : str
        ``module:function`` naming one module-level callable that takes no
        arguments.

    Returns
    -------
    Policy or System
        The factory's result, unfrozen.

    Raises
    ------
    ValueError
        The text is not a valid ``module:function`` reference or names a
        noncallable attribute.
    TypeError
        The factory returned something other than a Policy or System.
    ImportError, AttributeError
        The module or attribute cannot be imported, with the original cause.
        Any error raised by the factory itself propagates with its own type.

    Notes
    -----
    Host-only. Runs researcher code once. The command line and load_method
    share this one rule.
    """
    from marl_battlegrounds.evaluation.policy_execution import Policy, System

    value = installed_callable(reference)()
    if not isinstance(value, (Policy, System)):
        raise TypeError(f"Factory {reference!r} must return a System or Policy")
    return value


def load_method(reference: str) -> Policy | System:
    """Turn one method reference string into a Policy or System, unfrozen.

    Parameters
    ----------
    reference : str
        Checked in this order. A built-in Policy name ("random", "tdm-alpha" or
        "tdm-beta", as ``policy`` defines them) gives that Policy. Otherwise an
        existing directory is loaded as an exported PPO actor with
        ``marl_battlegrounds.training.load_system``; it must be an actor
        export, not a learner checkpoint, and needs the training extra.
        Otherwise the text must be a ``module:function`` factory (see
        load_factory). A relative path resolves against the current directory,
        so saved configurations must use absolute paths.

    Returns
    -------
    Policy or System
        The method exactly as its source produced it. Callers freeze it with
        ``freeze_evaluation_method`` before use.

    Raises
    ------
    TypeError
        reference is not a string, or a factory returned something other than
        a Policy or System.
    ValueError
        The text is empty, names no built-in, directory or valid factory, the
        directory is a learner checkpoint, or its files fail their checks.
    OSError
        An actor directory cannot be read.
    ImportError
        A directory was given without the training extra installed, or a
        factory's module cannot be imported. Errors from a factory's call
        propagate with their original type.

    Notes
    -----
    Host-only. A directory is read and its file hashes are checked; a factory
    runs researcher code once. Nothing is written.
    """
    if not isinstance(cast(object, reference), str):
        raise TypeError("A method reference must be a string")
    if not reference.strip():
        raise ValueError("A method reference must not be empty")
    from marl_battlegrounds.evaluation.policy_execution import policy

    try:
        return policy(reference)
    except ValueError:
        pass
    path = Path(reference)
    if path.is_dir():
        if (path / "checkpoint_details.json").exists():
            raise ValueError(
                "A learner checkpoint is not a method; export the actor first "
                "and pass the actor directory"
            )
        from marl_battlegrounds.training.checkpoints import load_system

        return load_system(path)
    if reference.count(":") != 1:
        raise ValueError(
            f"Method reference {reference!r} is not a built-in name, an existing "
            "actor directory or a module:function factory"
        )
    return load_factory(reference)


def validate_saved_method_reference(reference: str) -> None:
    """Check a durable method reference without opening files or calling factories.

    reference is a nonempty built-in Policy name, an absolute export path, or
    module:function. Relative paths, including bare directory names, are refused
    because their meaning changes with the working directory. Built-in names use
    the existing Policy authority. Missing imports and export files are checked
    later by load_method. Invalid types or grammar raise TypeError or ValueError.
    """
    if not isinstance(cast(object, reference), str):
        raise TypeError("A saved method reference must be a string")
    if not reference.strip():
        raise ValueError("A saved method reference must not be empty")
    from marl_battlegrounds.evaluation.policy_execution import policy

    try:
        policy(reference)
        return
    except ValueError:
        pass
    if Path(reference).is_absolute():
        return
    if reference.count(":") == 1:
        module, attribute = reference.split(":")
        if module and attribute and "." not in attribute and "/" not in reference:
            return
    raise ValueError(
        "Method paths must be absolute; otherwise use a built-in name "
        "or module:function"
    )
