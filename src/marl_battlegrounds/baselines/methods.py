"""Name the built-in training methods for code that can see any of them.

The PPO module keeps its own method check (``ppo.validate_ppo_method``) for
PPO-only code. Checkpoint, configuration and runner entry points that accept
any built-in method use ``validate_training_method`` here instead, and
``method_settings_field`` says which config field holds a method's settings.
This module imports only the standard library, so reading a method name never
loads JAX, Flax or Flashbax. It is internal: names and validation only, no
registry.
"""

from typing import Literal

type TrainingMethod = Literal["mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"]
"""A built-in training method name."""

type SettingsField = Literal["ppo", "qmix", "pqn"]
"""The TrainConfig field that holds one method's settings."""

PPO_METHODS = ("mappo", "ippo", "ff_mappo", "ff_ippo")
"""The four PPO method names, in their historical order."""
TRAINING_METHODS = (*PPO_METHODS, "qmix", "pqn_vdn")
"""Every built-in training method name."""


def validate_training_method(method: object) -> TrainingMethod:
    """Return a supported built-in method name or raise ValueError.

    Parameters
    ----------
    method : object
        Candidate name. Only the exact strings in TRAINING_METHODS pass.

    Returns
    -------
    TrainingMethod
        The same string, typed as a method name.

    Raises
    ------
    ValueError
        method is not a string or not a built-in method name.

    Examples
    --------
    >>> validate_training_method("qmix")
    'qmix'
    """
    if not isinstance(method, str) or method not in TRAINING_METHODS:
        raise ValueError(
            "Training method must be mappo, ippo, ff_mappo, ff_ippo, qmix or pqn_vdn"
        )
    return method


def is_ppo_method(method: object) -> bool:
    """Return True for the four PPO names; raise ValueError for unknown names.

    Parameters
    ----------
    method : object
        Candidate name, checked with validate_training_method.

    Returns
    -------
    bool
        True for mappo, ippo, ff_mappo and ff_ippo; False for qmix and
        pqn_vdn.

    Raises
    ------
    ValueError
        method is not one of the six training method names.
    """
    return validate_training_method(method) in PPO_METHODS


def method_settings_field(method: object) -> SettingsField:
    """Return the name of the TrainConfig field that holds a method's settings.

    Parameters
    ----------
    method : object
        Candidate name, checked with validate_training_method.

    Returns
    -------
    SettingsField
        "ppo" for the four PPO methods, "qmix" for qmix and "pqn" for
        pqn_vdn. Saved configs keep each method's settings under this key.

    Raises
    ------
    ValueError
        method is not one of the six training method names.

    Examples
    --------
    >>> method_settings_field("pqn_vdn")
    'pqn'
    """
    name = validate_training_method(method)
    if name == "qmix":
        return "qmix"
    return "pqn" if name == "pqn_vdn" else "ppo"
