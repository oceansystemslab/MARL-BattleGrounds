"""Expose local debugger discovery helpers through lazy imports.

``iter_scenario_summaries`` uses metadata only. ``get_scenario`` and
``list_scenarios`` load live scenario definitions when requested. Importing this
package alone does not initialize the simulator or an array backend. Launch the
product through its development shell scripts.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.dev.visual_debugger.scenario_catalog import iter_scenario_summaries
    from scripts.dev.visual_debugger.scenarios import (
        get_scenario,
        list_scenarios,
    )

__all__ = [
    "get_scenario",
    "iter_scenario_summaries",
    "list_scenarios",
]


def __getattr__(name: str) -> object:
    """Resolve one advertised helper without eagerly loading live scenario modules.

    Parameters
    ----------
    name : str
        Exported helper name requested by Python attribute access.

    Returns
    -------
    object
        The requested callable from the metadata or live scenario module.

    Raises
    ------
    AttributeError
        If the name is not in the public export list.
    """
    if name not in __all__:
        raise AttributeError(name)
    module_name = (
        "scripts.dev.visual_debugger.scenario_catalog"
        if name == "iter_scenario_summaries"
        else "scripts.dev.visual_debugger.scenarios"
    )
    module = import_module(module_name)
    return getattr(module, name)
