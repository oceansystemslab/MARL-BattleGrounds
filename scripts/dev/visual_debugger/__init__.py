"""Lazy public entry surface for the repository-local visual debugger."""

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
    """Load live-simulator scenario helpers only when explicitly requested."""
    if name not in __all__:
        raise AttributeError(name)
    module_name = (
        "scripts.dev.visual_debugger.scenario_catalog"
        if name == "iter_scenario_summaries"
        else "scripts.dev.visual_debugger.scenarios"
    )
    module = import_module(module_name)
    return getattr(module, name)
