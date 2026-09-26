"""Run prepared tournament matchups in order through the shared evaluator.

The caller prepares game identities and loads each pair. This module runs one
pair at a time, batching its games in the evaluator and publishing through the
caller's single RunWriter. Historical runs retain their original evaluator route.
It creates no workers, memory manager or provider lifecycle.
"""

# Historical runs retain their existing evaluator contract.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from marl_battlegrounds.environment import MetricMode
    from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, EvaluationResult
    from marl_battlegrounds.evaluation.policy_execution import Policy, System
    from marl_battlegrounds.evaluation.run_writer import RunWriter


@dataclass(frozen=True)
class TournamentJob:
    """Keep one prepared pair's exact execution and capture settings.

    Attributes
    ----------
    job_id : int
        Caller-owned job identifier, returned unchanged with its result.
    first, second : str
        Exact participant identities supplied to the caller's pair loader.
    episodes : tuple of EpisodeSpec
        Ordered saved or prepared games. Their global IDs, random keys, maps,
        spawn choices and memory rules must already have passed plan checks.
    seed : int
        Root random seed for this execution group, unchanged by scheduling.
    phase, pass_id : str
        Exact existing writer coordinates. Historical pass IDs stay unchanged.
    full_metrics_episodes, replay_episodes : tuple of int
        Execution episode IDs to capture, empty by default. Whole-pass full
        mode is still resolved by the evaluator.
    legacy_contract : bool, default=False
        True only for historical reversed-team runs that never had an explicit
        evaluation contract. It does not enable new historical-format runs.

    Notes
    -----
    The record is immutable; nested episode contents remain caller-owned.
    This record does not validate a plan or copy model parameters.
    """

    job_id: int
    first: str
    second: str
    episodes: tuple[EpisodeSpec, ...]
    seed: int
    phase: str
    pass_id: str
    full_metrics_episodes: tuple[int, ...] = ()
    replay_episodes: tuple[int, ...] = ()
    legacy_contract: bool = False


def execute_tournament_jobs(
    jobs: Iterable[TournamentJob],
    load_pair: Callable[
        [TournamentJob], AbstractContextManager[tuple[System | Policy, System | Policy]]
    ],
    *,
    writer: RunWriter | None,
    run_id: str,
    num_envs: int,
    metrics: MetricMode,
    chunk_size: int,
    registered_maps: Mapping[int, Mapping[str, Any]] | None = None,
) -> Generator[tuple[TournamentJob, EvaluationResult]]:
    """Run one matchup at a time with unchanged game identities and random keys.

    Parameters
    ----------
    jobs : iterable of TournamentJob
        Checked jobs in execution order. The iterable may skip completed jobs
        using the latest writer state before yielding the next one.
    load_pair : callable
        Accept a job and return a context manager yielding its ordered pair.
        The caller owns loading and keeps models available across matchups.
    writer : RunWriter or None
        Existing writer shared by all jobs, or None for in-memory results.
        This function never opens, replaces or closes it.
    run_id : str
        Stable run identity passed unchanged to every evaluator call.
    num_envs : int
        Positive maximum number of games evaluated at once within each matchup.
        Public tournament calls default to 128.
    metrics : MetricMode
        Caller-selected priority, full or none mode, unchanged for each job.
    chunk_size : int
        Positive number of ticks per evaluator chunk, unchanged for each job.
    registered_maps : mapping or None, default=None
        Checked source-map registrations for current fixed-team games. None
        uses the evaluator's ordinary lookup. Historical reversed-team jobs
        retain their old route without a new registration assertion.

    Yields
    ------
    tuple of TournamentJob and EvaluationResult
        Original job and shared evaluator result, after the pair context exits.
        The result contains no loaded model references.

    Raises
    ------
    Exception
        Loading, execution and writer errors propagate without automatic retry.
        Games already written remain available to ordinary resume. No failed
        execution is turned into a game result.

    Notes
    -----
    All loading and host calls stay on the calling thread. The evaluator owns
    action execution, episode memory, random streams, captures and recovery.
    This loop does not change model placement or scientific settings.
    """
    from marl_battlegrounds.evaluation.evaluate import (
        _evaluate_tournament_episodes,
        _run_evaluation,
    )

    for job in jobs:
        with load_pair(job) as (first, second):
            execute = (
                _run_evaluation
                if job.legacy_contract
                else _evaluate_tournament_episodes
            )
            result = execute(
                first,
                second,
                job.episodes,
                seed=job.seed,
                num_envs=num_envs,
                metrics=metrics,
                full_metrics_episodes=job.full_metrics_episodes,
                replay_episodes=job.replay_episodes,
                writer=writer,
                phase=job.phase,
                pass_id=job.pass_id,
                chunk_size=chunk_size,
                run_id=run_id,
                registered_maps=None if job.legacy_contract else registered_maps,
            )
        del first, second
        yield job, result
