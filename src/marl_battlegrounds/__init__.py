"""Expose the researcher API without loading every subsystem at import time.

Use import marl_battlegrounds as marl_bgs, then call make for the environment,
evaluate for frozen-policy games, or run_tournament for cross-play. Setup helpers
and the Policy adapter are available from the same package.

Exports load their owning module when first requested and are then cached here.
Importing this package alone does not import the environment or start JAX.
Each callable's own documentation defines its inputs, defaults and outputs.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marl_battlegrounds.environment import Environment, EnvironmentState, make
    from marl_battlegrounds.evaluation.evaluate import EvaluationResult, evaluate
    from marl_battlegrounds.evaluation.policy_execution import Policy, policy
    from marl_battlegrounds.evaluation.run_writer import RunWriter
    from marl_battlegrounds.evaluation.tournament import (
        TournamentResult,
        run_tournament,
    )
    from marl_battlegrounds.tasks import (
        balanced_spawn_configs,
        canonical_tournament_rosters,
        list_tdm_maps,
        list_tdm_scenarios,
        load_tdm_scenario,
    )

__all__ = [
    "Environment",
    "EnvironmentState",
    "EvaluationResult",
    "Policy",
    "RunWriter",
    "TournamentResult",
    "balanced_spawn_configs",
    "canonical_tournament_rosters",
    "evaluate",
    "list_tdm_maps",
    "list_tdm_scenarios",
    "load_tdm_scenario",
    "make",
    "policy",
    "run_tournament",
]

_MODULES = {
    "Environment": "environment",
    "EnvironmentState": "environment",
    "make": "environment",
    "EvaluationResult": "evaluation.evaluate",
    "evaluate": "evaluation.evaluate",
    "Policy": "evaluation.policy_execution",
    "policy": "evaluation.policy_execution",
    "RunWriter": "evaluation.run_writer",
    "TournamentResult": "evaluation.tournament",
    "run_tournament": "evaluation.tournament",
    "balanced_spawn_configs": "tasks",
    "canonical_tournament_rosters": "tasks",
    "list_tdm_maps": "tasks",
    "list_tdm_scenarios": "tasks",
    "load_tdm_scenario": "tasks",
}


def __getattr__(name: str) -> object:
    """Load and cache a supported public name on first access.

    Parameters
    ----------
    name : str
        Exact name of an export in _MODULES.

    Returns
    -------
    object
        The object from its owning module. Later access reuses the cached object.

    Raises
    ------
    AttributeError
        The name is not a supported lazy export.

    Notes
    -----
    Import errors from the owning module are allowed to reach the caller. Loading
    an environment or policy export may load JAX; reading a name does not promise
    that every exported subsystem is free of device dependencies.
    """
    if name in _MODULES:
        value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
