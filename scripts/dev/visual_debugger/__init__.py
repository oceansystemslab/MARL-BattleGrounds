"""Public entry surface for the repository-local visual debugger."""

from scripts.dev.visual_debugger.scenarios import (
    get_scenario,
    iter_scenario_summaries,
    list_scenarios,
)

__all__ = [
    "get_scenario",
    "iter_scenario_summaries",
    "list_scenarios",
]
