"""Load optional baseline components without adding cost to environment imports.

The inputs and actions modules use base JAX dependencies. The ppo and qmix
modules need the training extra. Importing this package alone loads neither JAX
nor Flax. Use ppo.make_ppo_system or qmix.make_qmix_system for an initialized,
untrained M8 System; these components do not start training or load a
deployment checkpoint.
"""

# Exports are loaded by __getattr__ so a base import stays light.
# pyright: reportUnsupportedDunderAll=false
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType

__all__ = ["actions", "inputs", "ppo", "qmix"]


def __getattr__(name: str) -> ModuleType:
    """Load a named baseline module on its first request.

    Parameters
    ----------
    name : str
        One of actions, inputs, ppo or qmix. Other names raise AttributeError.

    Returns
    -------
    ModuleType
        The owning module, cached for later access.

    Raises
    ------
    ImportError
        PPO or QMIX is requested without the optional Flax and Optax
        dependencies.
    AttributeError
        The name is not part of the package interface.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in ("ppo", "qmix") and any(
        find_spec(package) is None for package in ("flax", "optax")
    ):
        raise ImportError(
            f"{name.upper()} baselines need the training extra. "
            "Install marl-battlegrounds[training]."
        )
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
