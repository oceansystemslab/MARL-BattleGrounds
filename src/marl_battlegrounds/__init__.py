"""Expose the researcher API without loading every subsystem at import time.

Use import marl_battlegrounds as marl_bgs, then call make for the environment,
evaluate for frozen-System games, or run_tournament for cross-play. System,
init_systems and apply_systems support researcher-owned methods in raw loops.
Setup helpers and the existing Policy adapter use the same package entry point.
run_canonical_tournament uses a released tournament snapshot and verified records.
It requires a separately installed, qualified bundle; none is fabricated here.
select_initial_population freezes a declared choice from saved tournament games.

Exports load their owning module when first requested and are then cached here.
Importing this package alone does not import the environment or start JAX.
Each callable's own documentation defines its inputs, defaults and outputs.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marl_battlegrounds.autoreset import AutoReset
    from marl_battlegrounds.collection import collect_rollout
    from marl_battlegrounds.environment import Environment, EnvironmentState, make
    from marl_battlegrounds.episode_tracking import (
        EpisodeTrackingState,
        init_episode_tracking,
        track_episode_step,
    )
    from marl_battlegrounds.evaluation.canonical import run_canonical_tournament
    from marl_battlegrounds.evaluation.evaluate import (
        EpisodeSpec,
        evaluate,
        evaluate_episodes,
    )
    from marl_battlegrounds.evaluation.policy_execution import (
        Policy,
        System,
        SystemInput,
        SystemOutput,
        SystemState,
        SystemStepData,
        apply_systems,
        independent_policies,
        init_systems,
        policy,
        shared_policy,
        system_step_data,
    )
    from marl_battlegrounds.evaluation.population_selection import (
        select_initial_population,
    )
    from marl_battlegrounds.evaluation.results import (
        CanonicalTournamentResult,
        EvaluationResult,
        load_results,
    )
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
    "AutoReset",
    "CanonicalTournamentResult",
    "Environment",
    "EnvironmentState",
    "EpisodeSpec",
    "EpisodeTrackingState",
    "EvaluationResult",
    "Policy",
    "RunWriter",
    "System",
    "SystemInput",
    "SystemOutput",
    "SystemState",
    "SystemStepData",
    "TournamentResult",
    "apply_systems",
    "balanced_spawn_configs",
    "canonical_tournament_rosters",
    "collect_rollout",
    "evaluate",
    "evaluate_episodes",
    "independent_policies",
    "init_episode_tracking",
    "init_systems",
    "list_tdm_maps",
    "list_tdm_scenarios",
    "load_results",
    "load_tdm_scenario",
    "make",
    "policy",
    "run_canonical_tournament",
    "run_tournament",
    "select_initial_population",
    "shared_policy",
    "system_step_data",
    "track_episode_step",
]

_MODULES = {
    "AutoReset": "autoreset",
    "collect_rollout": "collection",
    "EpisodeTrackingState": "episode_tracking",
    "init_episode_tracking": "episode_tracking",
    "track_episode_step": "episode_tracking",
    "Environment": "environment",
    "EnvironmentState": "environment",
    "make": "environment",
    "EvaluationResult": "evaluation.results",
    "CanonicalTournamentResult": "evaluation.results",
    "EpisodeSpec": "evaluation.evaluate",
    "evaluate_episodes": "evaluation.evaluate",
    "load_results": "evaluation.results",
    "evaluate": "evaluation.evaluate",
    "Policy": "evaluation.policy_execution",
    "policy": "evaluation.policy_execution",
    "System": "evaluation.policy_execution",
    "SystemInput": "evaluation.policy_execution",
    "SystemOutput": "evaluation.policy_execution",
    "SystemState": "evaluation.policy_execution",
    "SystemStepData": "evaluation.policy_execution",
    "apply_systems": "evaluation.policy_execution",
    "independent_policies": "evaluation.policy_execution",
    "init_systems": "evaluation.policy_execution",
    "shared_policy": "evaluation.policy_execution",
    "system_step_data": "evaluation.policy_execution",
    "RunWriter": "evaluation.run_writer",
    "TournamentResult": "evaluation.tournament",
    "run_tournament": "evaluation.tournament",
    "run_canonical_tournament": "evaluation.canonical",
    "select_initial_population": "evaluation.population_selection",
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
