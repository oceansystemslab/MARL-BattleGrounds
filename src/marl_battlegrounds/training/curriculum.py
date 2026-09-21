"""Resolve exact training budgets and count the distributions actually played.

make_training_schedule is host setup and reads no content files. Its arrays
travel through compiled collection without changing shape. The private progress
helpers count already tracked real transitions and finish accounting stages;
they never reset games, apply policies or change live episode configurations.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.environment import (
    EnvironmentState,
    EpisodeInfo,
    _checked_increment,  # pyright: ignore[reportPrivateUsage]
    episode_advanced,
)
from marl_battlegrounds.episode_tracking import (
    EpisodeTrackingState,
    _begin_stage_numerical,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.evaluation.recording_types import (
    TRACKING_ERROR_ACCOUNTING,
    TRACKING_ERROR_OVERFLOW,
)
from marl_battlegrounds.training.distributions import validate_training_distribution

_STAGES = 17
_MAPS = 42
_MAX_COUNT = int(np.iinfo(np.int32).max)
_SCORE_THRESHOLDS = (*range(1, 11), 12, 15, 20)


class ScheduleArrays(NamedTuple):
    """Carry fixed-size, dynamic controls for one validated training schedule.

    Attributes
    ----------
    stage_count : Array
        Int32 scalar: 1 without curriculum, 17 for team/map curriculum, or
        13 for score-threshold curriculum.
    round_budgets, round_ends : Array
        Int32 (17,) stage round budgets and cumulative ends. Active budgets are
        positive. Unused budgets are zero and ends repeat total_rounds.
    team_sizes : Array
        Int32 (17,) equal team sizes in 1..5; unused rows contain 5.
    eligible_maps : Array
        Bool (17,42) masks in map order within each source block. Unused rows enable
        all maps. Each active pool contains at least one map.
    total_rounds : Array
        Int32 scalar; one round is one real transition from every lane.
    history_threshold_rounds : Array
        Int32 (20,) 5%, 10%, through 100% thresholds rounded upward to whole
        rounds, or round 1 followed by 5% through 95% when early history capture
        is requested. Repeated thresholds are allowed for small schedules.
    score_thresholds : Array
        Int32 (17,) winning scores requested at future resets. Unused rows
        contain 20. Continuing games keep their previous threshold.

    Notes
    -----
    All leaves are dynamic JAX values. Only make_training_schedule validates
    schedule values; replacing leaves requires the same contract. This record
    is immutable and does not identify the distribution of a continuing game.
    """

    stage_count: Array
    round_budgets: Array
    round_ends: Array
    team_sizes: Array
    eligible_maps: Array
    total_rounds: Array
    history_threshold_rounds: Array
    score_thresholds: Array


@dataclass(frozen=True)
class TrainingSchedule:
    """Keep a checked host schedule and its numerical collection controls.

    Attributes
    ----------
    total_env_steps : int
        Exact whole-batch experience requested; may exceed int32.
    num_envs : int
        Positive even number of fixed environment lanes.
    curriculum : bool
        Whether to use the 17-stage starting recipe instead of one full stage.
    arrays : ScheduleArrays
        Dynamic numerical controls; pass these into compiled collection.
    rounding_report : Mapping[str, object]
        Read-only, JSON-ready report after conversion with dict. Contains the
        requested total and batch, stage shares and ideal rounds as exact
        (numerator, denominator) pairs, assigned round counts, cumulative ends
        and the rounding rule. Threshold curricula also include the ordered
        score_thresholds. Nested values are immutable tuples or scalars.
    score_threshold_curriculum : bool, default=False
        Use the separate 13-stage score schedule with canonical 5v5 and all
        training maps. Cannot be combined with the team/map curriculum.
    early_history_capture : bool, default=False
        Capture the first self-play snapshot after the first completed update
        instead of at 5%, keeping the 5% through 95% captures and dropping the
        never-played 100% capture. Required by a positive pinned opponent share.

    Notes
    -----
    Construct with make_training_schedule. This frozen host descriptor writes
    no file and does not belong in a compiled numerical carry. Positive stages
    establish executable budgets, not useful exposure or learned behavior.
    """

    total_env_steps: int
    num_envs: int
    curriculum: bool
    arrays: ScheduleArrays
    rounding_report: Mapping[str, object]
    score_threshold_curriculum: bool = False
    early_history_capture: bool = False

    @property
    def score_thresholds(self) -> tuple[int, ...]:
        """Return the ordered source-bank thresholds required by this schedule."""
        return _SCORE_THRESHOLDS if self.score_threshold_curriculum else (20,)


class TrainingProgress(NamedTuple):
    """Count real training experience with fixed storage and no host transfers.

    Attributes
    ----------
    rounds : Array
        Int32 scalar counting successful full real rounds.
    episode_stage : Array
        Int32 (B,) distribution selected at each lane's latest reset. Collection
        changes selected rows only when those games reset; initial rows are 0.
    starts, exposure : Array
        Int32 (17,B) first real decisions and real transitions, grouped by each
        producing game's reset-time distribution. Resets alone count neither.
    actor_decisions : Array
        Int32 (17,B,5) Team A decisions by active, living pre-action actor.
    completed_stage_counts : Array
        Int32 (17,B,3) frozen default/swapped/unknown tracker counts, grouped by
        the accounting stage in which experience occurred.
    stage_complete : Array
        Bool (17,) successful accounting-stage completions.
    map_steps : Array
        Int32 (42,B) real transitions by producing source map. Collection owns
        these increments; the curriculum helper leaves them unchanged.
    opponent_starts, opponent_steps : Array
        Int32 (21,B) first decisions and real transitions by opponent source:
        current policy at row 0, then historical slots 0..19 at rows 1..20.
        Collection owns these increments; the curriculum helper preserves them.

    Notes
    -----
    All fields are dynamic JAX leaves in an immutable tuple. Each lane counter
    is bounded by its cumulative real experience. Sum across lanes, stages or
    actors on the host using Python integers to avoid whole-batch int32 overflow.
    EpisodeTrackingState remains the authority for source and transition truth.
    """

    rounds: Array
    episode_stage: Array
    starts: Array
    exposure: Array
    actor_decisions: Array
    completed_stage_counts: Array
    stage_complete: Array
    map_steps: Array
    opponent_starts: Array
    opponent_steps: Array


def make_training_schedule(
    *,
    total_env_steps: int,
    num_envs: int,
    curriculum: bool = False,
    score_threshold_curriculum: bool = False,
    early_history_capture: bool = False,
) -> TrainingSchedule:
    """Allocate exact whole-batch experience to the approved training stages.

    Parameters
    ----------
    total_env_steps : int
        Positive Python integer counting real environment transitions across
        all lanes. Must divide by num_envs exactly; bool is rejected.
    num_envs : int
        Positive even Python integer; bool is rejected. Fixed lanes keep their
        default or exchanged spawn arrangement throughout training.
    curriculum : bool, default=False
        False requests one 5v5 stage over all 42 maps. True requests sizes 1..5
        on map 0 at 4% each, then 5v5 on pools 0..1 through 0..11 at 20/11%
        each, then 5v5 on all maps for the remaining 60%.
    score_threshold_curriculum : bool, default=False
        Request K1 for 10% of transitions, K2..10 for 1/30 each, K12 and K15
        for 5% each, and K20 for 50%. All stages use canonical 5v5 and all 42
        training maps. Cannot be combined with curriculum=True. Changes apply
        at reset, so actual played shares can lag these requested shares.
    early_history_capture : bool, default=False
        Make the first self-play history threshold round 1 and keep the 5%
        through 95% thresholds, dropping the 100% capture that no later game can
        play. False keeps the 5% through 100% thresholds unchanged. Use True
        only together with a positive pinned opponent share in collection.

    Returns
    -------
    TrainingSchedule
        Immutable host report plus fixed-capacity numerical arrays. All stage
        shares are allocated together by largest remainder; earlier stages win
        exact ties. No total is rounded up. Only future resets use new controls.

    Raises
    ------
    TypeError
        Counts are not Python integers, are bool, or a switch is not bool.
    ValueError
        Counts are nonpositive, the batch is odd, the total is not divisible,
        per-lane rounds exceed int32, or any active stage receives zero rounds.

    Notes
    -----
    Host-only. Uses exact rational arithmetic and the existing distribution
    validator; reads no maps, writes no files and consumes no random keys.
    Keep whole-batch totals as Python integers even when they exceed int32.
    A short valid schedule does not prove meaningful learning exposure.
    """
    for name, value in (("total_env_steps", total_env_steps), ("num_envs", num_envs)):
        if isinstance(value, bool) or not isinstance(cast(object, value), int):
            raise TypeError(f"{name} must be a Python integer, not bool")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        ("curriculum", curriculum),
        ("score_threshold_curriculum", score_threshold_curriculum),
        ("early_history_capture", early_history_capture),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be bool")
    if curriculum and score_threshold_curriculum:
        raise ValueError("Choose team/map curriculum or score-threshold curriculum")
    if num_envs % 2:
        raise ValueError("num_envs must be even")
    if total_env_steps % num_envs:
        raise ValueError("total_env_steps must be divisible by num_envs")
    rounds = total_env_steps // num_envs
    if rounds > _MAX_COUNT:
        raise ValueError("per-lane total rounds must fit int32")
    shares = (
        (Fraction(1, 10),)
        + (Fraction(1, 30),) * 9
        + (Fraction(1, 20),) * 2
        + (Fraction(1, 2),)
        if score_threshold_curriculum
        else (Fraction(1, 25),) * 5 + (Fraction(1, 55),) * 11 + (Fraction(3, 5),)
        if curriculum
        else (Fraction(1),)
    )
    ideals = tuple(share * rounds for share in shares)
    counts = [ideal.numerator // ideal.denominator for ideal in ideals]
    order = sorted(range(len(shares)), key=lambda i: (-(ideals[i] - counts[i]), i))
    for index in order[: rounds - sum(counts)]:
        counts[index] += 1
    if min(counts) < 1:
        raise ValueError("total budget must give every active stage at least one round")
    budgets = np.zeros(_STAGES, np.int32)
    budgets[: len(counts)] = counts
    ends = np.cumsum(budgets, dtype=np.int64).astype(np.int32)
    sizes = np.full(_STAGES, 5, np.int32)
    pools = np.ones((_STAGES, _MAPS), np.bool_)
    thresholds = np.full(_STAGES, 20, np.int32)
    if score_threshold_curriculum:
        thresholds[: len(_SCORE_THRESHOLDS)] = _SCORE_THRESHOLDS
    if curriculum:
        sizes[:5] = np.arange(1, 6, dtype=np.int32)
        pools[:16] = False
        pools[:5, 0] = True
        for index in range(5, 16):
            pools[index, : index - 3] = True
    for index in range(len(counts)):
        validate_training_distribution(
            eligible_maps=pools[index], team_size=sizes[index]
        )
    history_thresholds = [(rounds * i + 19) // 20 for i in range(1, 21)]
    if early_history_capture:
        history_thresholds = [1, *history_thresholds[:-1]]
    arrays = ScheduleArrays(
        jnp.asarray(len(counts), jnp.int32),
        jnp.asarray(budgets),
        jnp.asarray(ends),
        jnp.asarray(sizes),
        jnp.asarray(pools),
        jnp.asarray(rounds, jnp.int32),
        jnp.asarray(history_thresholds, jnp.int32),
        jnp.asarray(thresholds),
    )
    report: Mapping[str, object] = MappingProxyType(
        {
            "total_env_steps": total_env_steps,
            "num_envs": num_envs,
            "stage_shares": tuple((v.numerator, v.denominator) for v in shares),
            "ideal_round_counts": tuple((v.numerator, v.denominator) for v in ideals),
            "assigned_round_counts": tuple(counts),
            "cumulative_round_ends": tuple(int(v) for v in ends[: len(counts)]),
            "rounding_rule": "Largest remainders; earlier stages win exact ties",
        }
    )
    if score_threshold_curriculum:
        report = MappingProxyType(
            {
                **report,
                "score_thresholds": _SCORE_THRESHOLDS,
            }
        )
    if early_history_capture:
        report = MappingProxyType(
            {
                **report,
                "history_thresholds": "first update, then 5% through 95%",
            }
        )
    return TrainingSchedule(
        total_env_steps,
        num_envs,
        curriculum,
        arrays,
        report,
        score_threshold_curriculum,
        early_history_capture,
    )


def _init_training_progress(  # pyright: ignore[reportUnusedFunction]
    *, num_envs: int
) -> TrainingProgress:
    """Create zeroed dynamic counters for fresh stage-zero games in B lanes.

    num_envs is a positive even Python integer already checked by setup. Return
    TrainingProgress with fixed stage/map/history capacity; no content is read,
    no game is reset and no input is changed. Call once on the host, then carry
    the returned arrays through collection. Wrong batch values raise ValueError.
    """
    if isinstance(num_envs, bool) or not isinstance(cast(object, num_envs), int):
        raise ValueError("num_envs must be a positive even Python integer")
    if num_envs <= 0 or num_envs % 2:
        raise ValueError("num_envs must be a positive even Python integer")
    return TrainingProgress(
        rounds=jnp.zeros((), jnp.int32),
        episode_stage=jnp.zeros(num_envs, jnp.int32),
        starts=jnp.zeros((_STAGES, num_envs), jnp.int32),
        exposure=jnp.zeros((_STAGES, num_envs), jnp.int32),
        actor_decisions=jnp.zeros((_STAGES, num_envs, 5), jnp.int32),
        completed_stage_counts=jnp.zeros((_STAGES, num_envs, 3), jnp.int32),
        stage_complete=jnp.zeros(_STAGES, jnp.bool_),
        map_steps=jnp.zeros((_MAPS, num_envs), jnp.int32),
        opponent_starts=jnp.zeros((21, num_envs), jnp.int32),
        opponent_steps=jnp.zeros((21, num_envs), jnp.int32),
    )


def _advance_training_schedule(  # pyright: ignore[reportUnusedFunction]
    progress: TrainingProgress,
    tracking: EpisodeTrackingState,
    before_state: EnvironmentState,
    after_state: EnvironmentState,
    info: EpisodeInfo,
    *,
    schedule: ScheduleArrays,
) -> tuple[TrainingProgress, EpisodeTrackingState, EpisodeInfo]:
    """Count one tracked real round and safely finish its accounting stage.

    Parameters
    ----------
    progress : TrainingProgress
        Counters before this round, with reset-time episode_stage already set.
    tracking : EpisodeTrackingState
        Tracker returned by track_episode_step for this producing transition.
        Its positive even native B fixes first-half default/second-half swapped
        spawn lanes. Stage zero must already have begun through checked setup.
    before_state, after_state : EnvironmentState
        Exact decision state and producing successor, before any later reset.
        Each lane must have made one real transition.
    info : EpisodeInfo
        Already tracked producing result. Returned flags include new failures.
    schedule : ScheduleArrays
        Host-validated dynamic controls for this same run and batch.

    Returns
    -------
    tuple
        Updated progress, tracker and producing info. Valid real experience is
        grouped by reset-time distribution. A finished stage freezes its proof
        then begins the next request; the final stage keeps its complete tracker.
        No episode, configuration or System memory is changed.

    Raises
    ------
    ValueError
        The static tracker batch is missing, odd or differs from progress.

    Notes
    -----
    Pure JAX; supports jit and scan without host reads. Boundary accounting
    failures add sticky bit 2 and overflow adds bit 4. Failed guards preserve
    the supplied stage counters, never begin a new stage and never mark it
    complete. Collection must stop before any further real action when flags
    are nonzero. This helper makes no policy, reset or tracking call. It checks
    fixed-lane completion without a whole-batch int32 sum. Padding is an invalid
    round and cannot count as experience or complete a stage.
    """
    batch = tracking.num_envs
    if batch is None or batch <= 0 or batch % 2:
        raise ValueError("training stages require a positive even native batch")
    if progress.episode_stage.shape != (batch,):
        raise ValueError("progress must match the tracker batch")
    ordinal = tracking.stage_ordinal
    safe_stage = jnp.clip(ordinal, 0, _STAGES - 1)
    previous_end = jnp.where(safe_stage > 0, schedule.round_ends[safe_stage - 1], 0)
    full_round = jnp.all(episode_advanced(info))
    category = (jnp.arange(batch) >= batch // 2).astype(jnp.int32)
    expected = jax.nn.one_hot(category, 3, dtype=jnp.int32) * tracking.stage_rounds
    accounting = (
        ~full_round
        | ~tracking.full_batch_rounds_valid
        | (ordinal < 0)
        | (ordinal >= schedule.stage_count)
        | (schedule.stage_count < 1)
        | (schedule.stage_count > _STAGES)
        | (tracking.stage_round_budget != schedule.round_budgets[safe_stage])
        | (tracking.stage_round_budget <= 0)
        | (tracking.stage_rounds <= 0)
        | (tracking.stage_rounds > tracking.stage_round_budget)
        | (progress.rounds < 0)
        | (progress.rounds >= schedule.total_rounds)
        | (progress.rounds - previous_end != tracking.stage_rounds - 1)
        | jnp.any(tracking.stage_counts != expected)
        | jnp.any(
            tracking.accounted_transition_count
            != after_state.cumulative_transition_count
        )
        | jnp.any(before_state.cumulative_transition_count != progress.rounds)
        | jnp.any(
            after_state.cumulative_transition_count
            - before_state.cumulative_transition_count
            != 1
        )
        | jnp.any(
            before_state.lifecycle_error
            | after_state.lifecycle_error
            | info.lifecycle_error
        )
        | jnp.any(
            (progress.episode_stage < 0)
            | (progress.episode_stage >= schedule.stage_count)
        )
        | progress.stage_complete[safe_stage]
    )
    flags = tracking.error_flags
    if info.episode_tracking_error is not None:
        flags |= info.episode_tracking_error
    pending = jnp.any(flags != 0)
    rounds, round_overflow = _checked_increment(progress.rounds, 1)
    lanes = jnp.arange(batch)
    distribution = jnp.clip(progress.episode_stage, 0, _STAGES - 1)
    exposure, exposure_overflow = _checked_increment(
        progress.exposure[distribution, lanes], 1
    )
    starts, starts_overflow = _checked_increment(
        progress.starts[distribution, lanes],
        before_state.episode_start.astype(jnp.int32),
    )
    actors, actor_overflow = _checked_increment(
        progress.actor_decisions[distribution, lanes],
        (
            before_state.config.agent_profile.active_mask[:, :5]
            & before_state.core_state.alive_mask[:, :5]
        ).astype(jnp.int32),
    )
    completed = tracking.stage_rounds == tracking.stage_round_budget
    has_next = ordinal < schedule.stage_count - 1
    next_budget = schedule.round_budgets[jnp.minimum(safe_stage + 1, _STAGES - 1)]
    boundary_bad = completed & (
        (rounds != schedule.round_ends[safe_stage]) | (has_next & (next_budget <= 0))
    )
    boundary_overflow = (
        completed
        & has_next
        & (
            (ordinal == _MAX_COUNT)
            | jnp.any(next_budget > _MAX_COUNT - tracking.accounted_transition_count)
        )
    )
    overflow = (
        round_overflow
        | jnp.any(exposure_overflow)
        | jnp.any(starts_overflow)
        | jnp.any(actor_overflow)
        | boundary_overflow
        | (ordinal == _MAX_COUNT)
    )
    flags |= jnp.where(accounting | boundary_bad, TRACKING_ERROR_ACCOUNTING, 0)
    flags |= jnp.where(overflow, TRACKING_ERROR_OVERFLOW, 0)
    tracking = replace(tracking, error_flags=flags.astype(jnp.int32))
    valid = ~pending & ~accounting & ~boundary_bad & ~overflow

    def accept() -> tuple[TrainingProgress, EpisodeTrackingState]:
        """Keep the real-round counts and install a next stage only when due."""
        updated = progress._replace(
            rounds=rounds,
            starts=progress.starts.at[distribution, lanes].set(starts),
            exposure=progress.exposure.at[distribution, lanes].set(exposure),
            actor_decisions=progress.actor_decisions.at[distribution, lanes].set(
                actors
            ),
        )

        def finish() -> tuple[TrainingProgress, EpisodeTrackingState]:
            """Freeze this exact proof while leaving the producing games intact."""
            finished = updated._replace(
                completed_stage_counts=updated.completed_stage_counts.at[
                    safe_stage
                ].set(tracking.stage_counts),
                stage_complete=updated.stage_complete.at[safe_stage].set(True),
            )
            next_tracking = cast(
                EpisodeTrackingState,
                jax.lax.cond(
                    has_next,
                    lambda: _begin_stage_numerical(tracking, round_budget=next_budget),
                    lambda: tracking,
                ),
            )
            return finished, next_tracking

        return cast(
            tuple[TrainingProgress, EpisodeTrackingState],
            jax.lax.cond(completed, finish, lambda: (updated, tracking)),
        )

    next_progress, next_tracking = cast(
        tuple[TrainingProgress, EpisodeTrackingState],
        jax.lax.cond(valid, accept, lambda: (progress, tracking)),
    )
    return next_progress, next_tracking, info._replace(episode_tracking_error=flags)
