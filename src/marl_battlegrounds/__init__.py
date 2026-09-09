"""MARL-BattleGrounds package."""

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

__all__ = [
    "Environment",
    "EnvironmentState",
    "EvaluationResult",
    "Policy",
    "RunWriter",
    "TournamentResult",
    "evaluate",
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
}


def __getattr__(name: str) -> object:
    """Keep read-only artifact imports independent of the JAX runtime."""
    if name in _MODULES:
        value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
