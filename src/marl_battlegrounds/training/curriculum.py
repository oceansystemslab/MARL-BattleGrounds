"""Resolve exact training budgets and count the distributions actually played.

make_training_schedule is host setup and reads no content files. Its arrays
travel through compiled collection without changing shape. The private progress
helpers count already tracked real transitions and finish accounting stages;
they never reset games, apply policies or change live episode configurations.
Private continuation helpers give added work a fresh bounded stage proof while
retaining original distributions, old proofs and cumulative experience.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from types import MappingProxyType
from typing import Any, NamedTuple, cast

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
from marl_battlegrounds.tasks import (
    AgentClassName,
    _roster_ids,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.training.distributions import validate_training_distribution

_STAGES = 17
_MAPS = 42
_MAX_COUNT = int(np.iinfo(np.int32).max)
_SCORE_THRESHOLDS = (*range(1, 11), 12, 15, 20)
type Curriculum = bool | Sequence[Mapping[str, object]]


class ScheduleArrays(NamedTuple):
    """Carry fixed-size, dynamic controls for one validated training schedule.

    Attributes
    ----------
    stage_count : Array
        Int32 scalar active accounting stages: 1 without curriculum, 17 for
        fresh team/map curriculum, 13 for fresh score curriculum, or 1..17
        custom stages. A child holds only its remaining accounting stages.
    round_budgets, round_ends : Array
        Int32 (17,) local stage round budgets and absolute cumulative ends.
        Active budgets are positive. Unused budgets are zero and ends repeat
        total_rounds. The first child budget starts at round_offset.
    team_sizes : Array
        Int32 (17,) equal sampled team sizes in 1..5; unused rows contain 5.
        A nonzero roster_class_ids row supplies its own sizes instead.
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
    round_offset, distribution_offset, distribution_count : Array or None
        Child segments use int32 scalars: cumulative rounds before this segment,
        the first distribution it requests, and the total number of preserved
        distributions across the lineage. Stage budgets/proofs use local slots;
        distribution tables and episode exposure keep their original slots.
        Ends and total_rounds stay cumulative. Fresh schedules use None,
        preserving their array layout.
    history_threshold_count : Array or None
        Optional int32 scalar length of the active prefix of the 20 history
        thresholds. Inactive rows are zero and never trigger a capture. None
        keeps the original all-20 contract and its checkpoint array layout.

    roster_class_ids : Array or None
        Custom schedules use int32 (17,10), in Team A then Team B slot order.
        Each explicit roster is compact, may repeat classes and may have unequal
        team sizes. An all-zero row uses team_sizes and the usual random roster.
        Boolean schedules keep None and their original numerical layout.

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
    round_offset: Array | None = None
    distribution_offset: Array | None = None
    distribution_count: Array | None = None
    history_threshold_count: Array | None = None
    roster_class_ids: Array | None = None


@dataclass(frozen=True)
class TrainingSchedule:
    """Keep a checked host schedule and its numerical collection controls.

    Attributes
    ----------
    total_env_steps : int
        Exact whole-batch target; may exceed int32. For a child this includes
        its parent experience. continuation records the added budget separately.
    num_envs : int
        Positive even number of fixed environment lanes.
    curriculum : bool or sequence of mappings
        False keeps one full stage and True keeps the built-in 17-stage recipe.
        Custom stages are copied into immutable mappings, preserving their order.
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
    continuation : Mapping[str, object] or None, default=None
        Read-only child declaration, including the original schedule, cumulative
        offset, parent stage proof and new budget. Nested records are also frozen.
        Use _continuation_details for a JSON-ready copy. None means a fresh run.

    history_capture_rounds : tuple[int, ...] or None
        Optional explicit fresh-run capture targets. Empty disables explicit
        captures; None keeps the original defaults. Use the checked helper
        _with_history_capture_rounds rather than replacing numerical leaves.

    Notes
    -----
    Construct with make_training_schedule. This frozen host descriptor writes
    no file and does not belong in a compiled numerical carry. Positive stages
    establish executable budgets, not useful exposure or learned behavior.
    """

    total_env_steps: int
    num_envs: int
    curriculum: Curriculum
    arrays: ScheduleArrays
    rounding_report: Mapping[str, object]
    score_threshold_curriculum: bool = False
    early_history_capture: bool = False
    continuation: Mapping[str, object] | None = None
    history_capture_rounds: tuple[int, ...] | None = None

    @property
    def score_thresholds(self) -> tuple[int, ...]:
        """Return the ordered source-bank thresholds required by this schedule."""
        count = int(
            self.arrays.stage_count
            if self.arrays.distribution_count is None
            else self.arrays.distribution_count
        )
        values = np.asarray(self.arrays.score_thresholds)[:count]
        return tuple(dict.fromkeys(int(v) for v in values))


class TrainingProgress(NamedTuple):
    """Count real training experience with fixed storage and no host transfers.

    Attributes
    ----------
    rounds : Array
        Int32 scalar counting cumulative successful full real rounds, including
        parent segments. A child accounting boundary does not reset this count.
    episode_stage : Array
        Int32 (B,) original distribution slot selected at each lane's latest
        reset. Child segments retain these IDs for inherited games. Collection
        changes selected rows only when those games reset; initial rows are 0.
    starts, exposure : Array
        Int32 (17,B) first real decisions and real transitions, grouped by each
        producing game's reset-time distribution. Resets alone count neither.
    actor_decisions : Array
        Int32 (17,B,5) Team A decisions by active, living pre-action actor.
    completed_stage_counts : Array
        Int32 (17,B,3) frozen default/swapped/unknown tracker counts, grouped by
        the current segment's accounting stage in which experience occurred.
        A checked child boundary archives the old proof and clears only these
        counts and stage_complete; all exposure counters remain cumulative.
    stage_complete : Array
        Bool (17,) successful accounting-stage completions.
    map_steps : Array
        Int32 (42,B) real transitions by producing source map. Collection owns
        these increments; the curriculum helper leaves them unchanged.
    opponent_starts, opponent_steps : Array
        Int32 (C+2,B) first decisions and real transitions by stable identity.
        C bounds total captures, not resident slots: row 0 current, row 1
        permanent pin, and row capture_id+2 for historical copies.
        Collection owns these increments; the curriculum helper preserves them.
    partner_used : Array or None
        Bool (M,) for declared fixed partners that supplied at least one active,
        living owned action on either team. None when partner training is off.
        A declared but unused member does not become training exposure.

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
    partner_used: Array | None = None


def make_training_schedule(
    *,
    total_env_steps: int,
    num_envs: int,
    curriculum: Curriculum = False,
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
    curriculum : bool or sequence of mappings, default=False
        False requests one 5v5 stage over all 42 maps. True requests sizes 1..5
        on map 0 at 4% each, then 5v5 on pools 0..1 through 0..11 at 20/11%
        each, then 5v5 on all maps for the remaining 60%. A custom list has
        1..17 stages. Each stage gives a positive share, nonempty distinct maps
        from 0..41, and either team_size (1..5) or rosters with system/opponent
        class-name lists. Explicit rosters permit repeated classes and unequal
        sizes. Shares must sum to one. Optional score_threshold defaults to 20
        and means the winning score, not a test for promotion. Custom stages
        cannot be combined with score_threshold_curriculum.
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
    custom = None if type(curriculum) is bool else _custom_stages(curriculum)
    for name, value in (
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
        tuple(Fraction(str(stage["share"])) for stage in custom)
        if custom is not None
        else (Fraction(1, 10),)
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
    roster_ids = None
    if custom is not None:
        roster_ids = np.zeros((_STAGES, 10), np.int32)
        for index, stage in enumerate(custom):
            pools[index] = np.isin(
                np.arange(_MAPS), cast(tuple[int, ...], stage["maps"])
            )
            thresholds[index] = cast(int, stage.get("score_threshold", 20))
            if "team_size" in stage:
                sizes[index] = cast(int, stage["team_size"])
            else:
                rosters = cast(Mapping[str, Sequence[AgentClassName]], stage["rosters"])
                roster_ids[index] = (
                    *_roster_ids(rosters["system"], name="system roster"),
                    *_roster_ids(rosters["opponent"], name="opponent roster"),
                )
    elif curriculum:
        sizes[:5] = np.arange(1, 6, dtype=np.int32)
        pools[:16] = False
        pools[:5, 0] = True
        for index in range(5, 16):
            pools[index, : index - 3] = True
    for index in range(len(counts)):
        validate_training_distribution(
            eligible_maps=pools[index],
            team_size=sizes[index],
            roster_class_ids=None if roster_ids is None else roster_ids[index],
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
        roster_class_ids=None if roster_ids is None else jnp.asarray(roster_ids),
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
    if custom is not None:
        assert roster_ids is not None
        report = MappingProxyType(
            {
                **report,
                "stage_maps": tuple(
                    cast(tuple[int, ...], stage["maps"]) for stage in custom
                ),
                "team_sizes": tuple(int(v) for v in sizes[: len(custom)]),
                "roster_class_ids": tuple(
                    tuple(int(v) for v in row) for row in roster_ids[: len(custom)]
                ),
                "score_thresholds": tuple(int(v) for v in thresholds[: len(custom)]),
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
        curriculum if custom is None else custom,
        arrays,
        report,
        score_threshold_curriculum,
        early_history_capture,
    )


def _custom_stages(value: object) -> tuple[Mapping[str, object], ...]:
    """Validate a stage list and copy its JSON values into immutable mappings.

    Stages use positive finite shares summing exactly to one as decimal
    fractions. Maps are training IDs 0..41. Roster names use the task owner's
    validator; thresholds receive their depth-specific upper check when the
    source bank is prepared. This helper reads no assets and changes no inputs.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("curriculum must be bool or a sequence of stage mappings")
    stages = cast(Sequence[object], value)
    if not 1 <= len(stages) <= _STAGES:
        raise ValueError("A custom curriculum needs 1..17 stages")
    total = Fraction(0)
    for index, raw in enumerate(stages):
        if not isinstance(raw, Mapping):
            raise TypeError(f"Curriculum stage {index} must be a mapping")
        stage = cast(Mapping[str, Any], raw)
        keys = set(stage)
        if (
            not {"share", "maps"} <= keys
            or keys - {"share", "maps", "team_size", "rosters", "score_threshold"}
            or ("team_size" in keys) == ("rosters" in keys)
        ):
            raise ValueError(
                f"Curriculum stage {index} needs share, maps and exactly one of "
                "team_size or rosters"
            )
        share = stage["share"]
        if type(share) not in (int, float) or not 0 < share <= 1:
            raise ValueError("Stage shares must be positive finite numbers")
        total += Fraction(str(share))
        raw_maps = stage["maps"]
        if not isinstance(raw_maps, (list, tuple)):
            raise ValueError("Stage maps must be a list of training IDs from 0..41")
        maps = cast(Sequence[object], raw_maps)
        if (
            not maps
            or any(type(item) is not int or not 0 <= item < _MAPS for item in maps)
            or len(set(maps)) != len(maps)
        ):
            raise ValueError("Stage maps must be distinct training IDs from 0..41")
        if "team_size" in stage:
            size = stage["team_size"]
            if type(size) is not int or not 1 <= size <= 5:
                raise ValueError("Stage team_size must be a Python integer in 1..5")
        else:
            rosters = cast(Mapping[str, object], stage["rosters"])
            if not isinstance(cast(object, rosters), Mapping) or set(rosters) != {
                "system",
                "opponent",
            }:
                raise ValueError("Stage rosters must name system and opponent")
            for team in ("system", "opponent"):
                _roster_ids(cast(Sequence[AgentClassName], rosters[team]), name=team)
        threshold = stage.get("score_threshold", 20)
        if type(threshold) is not int or not 1 <= threshold <= 2**24 - 4:
            raise ValueError(
                "Stage score_threshold is outside the supported score range"
            )
    if total != 1:
        raise ValueError("Curriculum stage shares must sum to one")
    frozen = _freeze_continuation({"stages": stages})
    return cast(tuple[Mapping[str, object], ...], frozen["stages"])


def _freeze_continuation(value: Mapping[str, object]) -> Mapping[str, object]:
    """Copy a finite JSON-like declaration into read-only nested records.

    Mapping keys must be strings; leaves must be None, strings, booleans, finite
    numbers or sequences of these. Raises ValueError for other values. No input
    is retained as mutable storage. Use _continuation_details to encode a schedule.
    """

    def freeze(item: object) -> object:
        """Freeze one mapping or sequence, checking its scalar leaves."""
        if isinstance(item, Mapping):
            mapping = cast(Mapping[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ValueError("Continuation keys must be strings")
            return MappingProxyType(
                {cast(str, key): freeze(val) for key, val in mapping.items()}
            )
        if isinstance(item, (tuple, list)):
            return tuple(
                freeze(val) for val in cast(tuple[object, ...] | list[object], item)
            )
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float and np.isfinite(item):
            return item
        raise ValueError("Continuation must contain finite JSON values")

    if not isinstance(cast(object, value), Mapping):
        raise ValueError("Continuation declaration must be a mapping")
    return cast(Mapping[str, object], freeze(value))


def _continuation_details(schedule: TrainingSchedule) -> dict[str, Any] | None:
    """Return a separate JSON-ready copy of a schedule's child declaration.

    None means a fresh schedule. Nested read-only mappings become dictionaries
    and tuples become lists. This reads host metadata only and changes no input.
    """

    def plain(item: object) -> object:
        """Copy one frozen JSON value into ordinary JSON containers."""
        if isinstance(item, Mapping):
            return {
                key: plain(val) for key, val in cast(Mapping[str, object], item).items()
            }
        if isinstance(item, (tuple, list)):
            return [plain(val) for val in cast(tuple[object, ...] | list[object], item)]
        return item

    return cast(dict[str, Any] | None, plain(schedule.continuation))


def _distribution_schedule(
    root: TrainingSchedule, changes: Sequence[Mapping[str, object]]
) -> TrainingSchedule:
    """Append declared child distributions without reusing historical stage IDs.

    Each change names its segment, cumulative starting round, added whole-batch
    budget and custom curriculum. At most 17 distributions fit across the whole
    lineage. The latest change supplies the active accounting boundaries; old
    map, roster and threshold rows remain available to unfinished games and
    stored experience. This is host setup with no game reset or file writes.
    """
    current = root
    count = int(root.arrays.stage_count)
    previous_index = 0
    previous_round = -1
    for change in changes:
        if not isinstance(cast(object, change), Mapping) or set(change) != {
            "segment_index",
            "round_offset",
            "additional_env_steps",
            "curriculum",
        }:
            raise ValueError("Changed curriculum declaration fields differ")
        index, start, added = (
            change["segment_index"],
            change["round_offset"],
            change["additional_env_steps"],
        )
        if (
            type(index) is not int
            or index <= previous_index
            or type(start) is not int
            or start < 0
            or start < previous_round
            or type(added) is not int
            or added <= 0
            or added % root.num_envs
            or start + added // root.num_envs > _MAX_COUNT
        ):
            raise ValueError(
                "Changed curriculum needs ordered valid segment boundaries"
            )
        stages = _custom_stages(change["curriculum"])
        new = make_training_schedule(
            total_env_steps=added, num_envs=root.num_envs, curriculum=stages
        )
        added_count = int(new.arrays.stage_count)
        if count + added_count > _STAGES:
            raise ValueError(
                "Changed curriculum exceeds 17 distribution IDs across this lineage; "
                "start a fresh run to use another stage table"
            )

        def append(
            old: Array, values: Array, first: int = count, length: int = added_count
        ) -> Array:
            """Keep existing rows and place new controls in unused fixed slots."""
            return old.at[first : first + length].set(values[:length])

        previous_rosters = current.arrays.roster_class_ids
        if previous_rosters is None:
            previous_rosters = jnp.zeros((_STAGES, 10), jnp.int32)
        assert new.arrays.roster_class_ids is not None
        arrays = new.arrays._replace(
            team_sizes=append(current.arrays.team_sizes, new.arrays.team_sizes),
            eligible_maps=append(
                current.arrays.eligible_maps, new.arrays.eligible_maps
            ),
            score_thresholds=append(
                current.arrays.score_thresholds, new.arrays.score_thresholds
            ),
            roster_class_ids=append(previous_rosters, new.arrays.roster_class_ids),
            round_ends=new.arrays.round_ends + jnp.int32(start),
            total_rounds=new.arrays.total_rounds + jnp.int32(start),
            distribution_offset=jnp.int32(count),
            distribution_count=jnp.int32(count + added_count),
        )
        current = replace(root, arrays=arrays)
        count += added_count
        previous_index, previous_round = index, start
    return current


def _segment_arrays(
    root: TrainingSchedule,
    *,
    start: int,
    stop: int,
    thresholds: tuple[int, ...],
    threshold_count: int | None,
    curriculum_changes: Sequence[Mapping[str, object]] = (),
) -> tuple[ScheduleArrays, Mapping[str, object]]:
    """Build one bounded suffix while keeping original distribution tables.

    start/stop are checked cumulative rounds with 0 <= start < stop <= int32.
    thresholds is the complete 20-row absolute history table; threshold_count
    is its active prefix or None for all rows. The final distribution keeps
    extra work in its own slot. Return dynamic arrays and an immutable report;
    no original array is changed and no random key is consumed. Optional child
    curricula append source distributions, retaining their original IDs.
    """
    root = _distribution_schedule(root, curriculum_changes)
    original_count = int(root.arrays.stage_count)
    distribution_base = (
        0
        if root.arrays.distribution_offset is None
        else int(root.arrays.distribution_offset)
    )
    distribution_count = (
        original_count
        if root.arrays.distribution_count is None
        else int(root.arrays.distribution_count)
    )
    original_ends = tuple(
        int(value) for value in np.asarray(root.arrays.round_ends)[:original_count]
    )
    local_first = min(sum(end <= start for end in original_ends), original_count - 1)
    first = distribution_base + local_first
    ends = tuple(end for end in original_ends[local_first:-1] if start < end < stop)
    ends = (*ends, stop)
    budgets = tuple(
        end - previous for previous, end in zip((start, *ends[:-1]), ends, strict=True)
    )
    arrays = root.arrays._replace(
        stage_count=jnp.int32(len(ends)),
        round_budgets=jnp.asarray(
            (*budgets, *((0,) * (_STAGES - len(ends)))), jnp.int32
        ),
        round_ends=jnp.asarray((*ends, *((stop,) * (_STAGES - len(ends)))), jnp.int32),
        total_rounds=jnp.int32(stop),
        history_threshold_rounds=jnp.asarray(thresholds, jnp.int32),
        round_offset=jnp.int32(start),
        distribution_offset=jnp.int32(first),
        distribution_count=jnp.int32(distribution_count),
        history_threshold_count=None
        if threshold_count is None
        else jnp.int32(threshold_count),
    )
    report = MappingProxyType(
        {
            "total_env_steps": stop * root.num_envs,
            "num_envs": root.num_envs,
            "assigned_round_counts": budgets,
            "cumulative_round_ends": ends,
            "distribution_indices": tuple(range(first, first + len(ends))),
            "score_thresholds": tuple(
                int(root.arrays.score_thresholds[i])
                for i in range(first, first + len(ends))
            ),
            "rounding_rule": "Whole rounds; retain original distribution boundaries",
        }
    )
    return arrays, report


def _make_continuation_schedule(  # pyright: ignore[reportUnusedFunction]
    parent_schedule: TrainingSchedule,
    *,
    completed_rounds: int,
    additional_env_steps: int,
    history_threshold_rounds: tuple[int, ...] | None = None,
    curriculum: Sequence[Mapping[str, object]] | None = None,
) -> TrainingSchedule:
    """Declare added whole-batch work after a parent's actual saved boundary.

    Parameters
    ----------
    parent_schedule : TrainingSchedule
        Original or checked child schedule. Its root distribution tables and
        old boundaries remain the authority across repeated extensions.
    completed_rounds : int
        Cumulative real rounds at the parent checkpoint, within its segment.
    additional_env_steps : int
        Positive new transitions, divisible by the parent's num_envs. No silent
        rounding is performed; the cumulative per-lane total must fit int32.
    history_threshold_rounds : tuple[int, ...] or None, default=None
        Explicit replacement active history thresholds in cumulative rounds,
        sorted and positive, with at most 20 entries. Unused rows are zero.
        None preserves the parent's exact thresholds and active count. The
        checked collection boundary also proves that old snapshots remain bound.
    curriculum : sequence of mappings or None, default=None
        Optional replacement stage list for this child's additional budget.
        It uses make_training_schedule's stage fields and appends distribution
        IDs without changing old games. Old and new distributions together must
        fit 17 rows. None continues the parent's existing stage boundaries.

    Returns
    -------
    TrainingSchedule
        A pending child declaration with fixed 17-row arrays. Its parent proof
        is None until collection._begin_training_segment binds the actual carry.
        The learner context, if any, is added by the learner/runner owner.

    Raises
    ------
    ValueError
        The parent declaration, boundary, budget or history table is invalid.

    Notes
    -----
    Host-only; reads small schedule arrays, consumes no keys, writes no files
    and never initializes or resets a game. Completed parent proofs stay in
    their parent segment. Extra work in the final distribution shares its last
    child slot, retaining the original final endpoint in the root declaration.
    """
    _check_training_schedule(parent_schedule)
    start = (
        0
        if parent_schedule.arrays.round_offset is None
        else int(parent_schedule.arrays.round_offset)
    )
    if type(completed_rounds) is not int or not start <= completed_rounds <= int(
        parent_schedule.arrays.total_rounds
    ):
        raise ValueError("Parent rounds must lie within its declared segment")
    if (
        type(additional_env_steps) is not int
        or additional_env_steps <= 0
        or additional_env_steps % parent_schedule.num_envs
    ):
        raise ValueError(
            "Additional experience must be positive whole-batch transitions"
        )
    stop = completed_rounds + additional_env_steps // parent_schedule.num_envs
    if stop > _MAX_COUNT:
        raise ValueError("Cumulative continuation rounds must fit int32")
    previous = _continuation_details(parent_schedule)
    root = (
        {
            "total_env_steps": parent_schedule.total_env_steps,
            "num_envs": parent_schedule.num_envs,
            "curriculum": parent_schedule.curriculum,
            "score_threshold_curriculum": parent_schedule.score_threshold_curriculum,
            "early_history_capture": parent_schedule.early_history_capture,
            **(
                {}
                if parent_schedule.history_capture_rounds is None
                else {"history_capture_rounds": parent_schedule.history_capture_rounds}
            ),
        }
        if previous is None
        else previous["root_schedule"]
    )
    old_thresholds = tuple(
        int(value)
        for value in np.asarray(parent_schedule.arrays.history_threshold_rounds)
    )
    old_count = parent_schedule.arrays.history_threshold_count
    if history_threshold_rounds is not None and not isinstance(
        cast(object, history_threshold_rounds), tuple
    ):
        raise ValueError("History thresholds must be a tuple of absolute rounds")
    thresholds = (
        old_thresholds if history_threshold_rounds is None else history_threshold_rounds
    )
    count = (
        (None if old_count is None else int(old_count))
        if history_threshold_rounds is None
        else len(thresholds)
    )
    if (
        not isinstance(cast(object, thresholds), tuple)
        or len(thresholds) > 20
        or any(
            type(value) is not int or not 0 < value <= _MAX_COUNT
            for value in thresholds[: 20 if count is None else count]
        )
        or tuple(sorted(thresholds[: 20 if count is None else count]))
        != thresholds[: 20 if count is None else count]
    ):
        raise ValueError(
            "History thresholds must be at most 20 ordered positive rounds"
        )
    if history_threshold_rounds is not None:
        thresholds = (*thresholds, *((0,) * (20 - len(thresholds))))
    changes: list[Mapping[str, object]] = (
        [] if previous is None else list(previous.get("curriculum_changes", ()))
    )
    segment_index = 1 if previous is None else previous["segment_index"] + 1
    if curriculum is not None:
        changes.append(
            {
                "segment_index": segment_index,
                "round_offset": completed_rounds,
                "additional_env_steps": additional_env_steps,
                "curriculum": _custom_stages(curriculum),
            }
        )
    declaration = {
        "schema_version": 2 if changes else 1,
        "root_schedule": root,
        "root_rounding_report": dict(parent_schedule.rounding_report)
        if previous is None
        else previous["root_rounding_report"],
        "segment_index": segment_index,
        "round_offset": completed_rounds,
        "additional_env_steps": additional_env_steps,
        "history_threshold_rounds": thresholds,
        "history_threshold_count": count,
        "parent_schedule": {
            "round_offset": start,
            "total_rounds": int(parent_schedule.arrays.total_rounds),
            "stage_count": int(parent_schedule.arrays.stage_count),
            "round_budgets": tuple(
                int(value) for value in np.asarray(parent_schedule.arrays.round_budgets)
            ),
            "round_ends": tuple(
                int(value) for value in np.asarray(parent_schedule.arrays.round_ends)
            ),
            "history_threshold_rounds": old_thresholds,
            "history_threshold_count": None if old_count is None else int(old_count),
        },
        "parent_stage_proof": None,
    }
    if changes:
        declaration["curriculum_changes"] = changes
    return _build_continuation_schedule(declaration, require_parent_proof=False)


def _build_continuation_schedule(
    declaration: Mapping[str, object], *, require_parent_proof: bool
) -> TrainingSchedule:
    """Check host segment facts and reconstruct its numerical schedule.

    declaration uses the private schema emitted by _make_continuation_schedule.
    require_parent_proof=False permits only the pending in-memory builder result;
    durable restoration requires its exact parent proof. The optional learner
    mapping is retained for its own owner's validation. Raises ValueError on
    malformed or self-inconsistent data; no files or live state are changed.
    """
    frozen = _freeze_continuation(declaration)
    expected = {
        "schema_version",
        "root_schedule",
        "root_rounding_report",
        "segment_index",
        "round_offset",
        "additional_env_steps",
        "history_threshold_rounds",
        "history_threshold_count",
        "parent_schedule",
        "parent_stage_proof",
    }
    if (
        set(frozen) - {"learner", "curriculum_changes"} != expected
        or type(frozen["schema_version"]) is not int
        or frozen["schema_version"] not in (1, 2)
        or (frozen["schema_version"] == 2) != ("curriculum_changes" in frozen)
    ):
        raise ValueError("Unsupported continuation schedule declaration")
    root_record = frozen["root_schedule"]
    if not isinstance(root_record, Mapping):
        raise ValueError("Continuation root schedule must be a mapping")
    root_record = cast(Mapping[str, Any], root_record)
    root_keys = {
        "total_env_steps",
        "num_envs",
        "curriculum",
        "score_threshold_curriculum",
        "early_history_capture",
    }
    if set(root_record) - {"history_capture_rounds"} != root_keys:
        raise ValueError("Continuation root schedule fields differ")
    root_arguments = dict(root_record)
    capture_rounds = root_arguments.pop("history_capture_rounds", None)
    root = make_training_schedule(**cast(Any, root_arguments))
    if capture_rounds is not None:
        root = _with_history_capture_rounds(root, capture_rounds)
    if frozen["root_rounding_report"] != _freeze_continuation(root.rounding_report):
        raise ValueError(
            "Continuation original boundaries differ from the root schedule"
        )
    start, added, index = (
        frozen["round_offset"],
        frozen["additional_env_steps"],
        frozen["segment_index"],
    )
    if (
        type(start) is not int
        or start < 0
        or type(added) is not int
        or added <= 0
        or added % root.num_envs
        or type(index) is not int
        or index < 1
    ):
        raise ValueError("Continuation budget and index must be valid whole counts")
    stop = start + added // root.num_envs
    if stop > _MAX_COUNT:
        raise ValueError("Cumulative continuation rounds must fit int32")
    changes = frozen.get("curriculum_changes", ())
    if not isinstance(changes, tuple) or (
        frozen["schema_version"] == 2 and not changes
    ):
        raise ValueError("Changed curricula must be a nonempty ordered sequence")
    changes = cast(tuple[Mapping[str, Any], ...], changes)
    _distribution_schedule(root, changes)
    for change in changes:
        if change["segment_index"] > index or change["round_offset"] > start:
            raise ValueError("A curriculum change is beyond this continuation boundary")
        if change["segment_index"] == index and (
            change["round_offset"] != start or change["additional_env_steps"] != added
        ):
            raise ValueError("Current curriculum change must match its child budget")
    thresholds = cast(tuple[int, ...], frozen["history_threshold_rounds"])
    count = frozen["history_threshold_count"]
    active = 20 if count is None else count
    if (
        type(active) is not int
        or not 0 <= active <= 20
        or not isinstance(cast(object, thresholds), tuple)
        or len(thresholds) != 20
        or any(type(value) is not int for value in thresholds)
        or any(not 0 < value <= _MAX_COUNT for value in thresholds[:active])
        or tuple(sorted(thresholds[:active])) != thresholds[:active]
        or any(value != 0 for value in thresholds[active:])
    ):
        raise ValueError("Continuation history table is malformed")
    parent = cast(Mapping[str, Any], frozen["parent_schedule"])
    if not isinstance(cast(object, parent), Mapping) or set(parent) != {
        "round_offset",
        "total_rounds",
        "stage_count",
        "round_budgets",
        "round_ends",
        "history_threshold_rounds",
        "history_threshold_count",
    }:
        raise ValueError("Continuation parent schedule is malformed")
    parent_start, parent_stop = parent["round_offset"], parent["total_rounds"]
    if (
        type(parent_start) is not int
        or type(parent_stop) is not int
        or not 0 <= parent_start <= start <= parent_stop <= _MAX_COUNT
        or parent_start == parent_stop
    ):
        raise ValueError("Continuation parent boundary is invalid")
    if (
        type(parent["stage_count"]) is not int
        or not 1 <= parent["stage_count"] <= _STAGES
    ):
        raise ValueError("Continuation parent stage count is invalid")
    for name in ("round_budgets", "round_ends"):
        values = cast(tuple[int, ...], parent[name])
        if (
            not isinstance(cast(object, values), tuple)
            or len(values) != _STAGES
            or any(type(value) is not int for value in values)
        ):
            raise ValueError("Continuation parent stage counts must be integer rows")
    parent_thresholds = cast(tuple[int, ...], parent["history_threshold_rounds"])
    parent_active = (
        20
        if parent["history_threshold_count"] is None
        else parent["history_threshold_count"]
    )
    if (
        type(parent_active) is not int
        or not 0 <= parent_active <= 20
        or not isinstance(cast(object, parent_thresholds), tuple)
        or len(parent_thresholds) != 20
        or any(type(value) is not int for value in parent_thresholds)
        or any(
            not 0 < value <= _MAX_COUNT for value in parent_thresholds[:parent_active]
        )
        or tuple(sorted(parent_thresholds[:parent_active]))
        != parent_thresholds[:parent_active]
        or any(parent_thresholds[parent_active:])
    ):
        raise ValueError("Continuation parent history table is malformed")
    parent_arrays, _ = _segment_arrays(
        root,
        start=parent_start,
        stop=parent_stop,
        thresholds=cast(tuple[int, ...], thresholds),
        threshold_count=cast(int | None, count),
        curriculum_changes=tuple(c for c in changes if c["segment_index"] < index),
    )
    if (
        int(parent_arrays.stage_count) != parent["stage_count"]
        or tuple(np.asarray(parent_arrays.round_budgets).tolist())
        != parent["round_budgets"]
        or tuple(np.asarray(parent_arrays.round_ends).tolist()) != parent["round_ends"]
    ):
        raise ValueError("Continuation parent stages differ from original boundaries")
    proof = frozen["parent_stage_proof"]
    if proof is None:
        if require_parent_proof:
            raise ValueError("Continuation requires the checked parent stage proof")
    else:
        _check_parent_stage_proof(parent, proof, rounds=start, num_envs=root.num_envs)
    arrays, report = _segment_arrays(
        root,
        start=start,
        stop=stop,
        thresholds=cast(tuple[int, ...], thresholds),
        threshold_count=cast(int | None, count),
        curriculum_changes=changes,
    )
    return replace(
        root,
        total_env_steps=stop * root.num_envs,
        arrays=arrays,
        rounding_report=report,
        continuation=frozen,
    )


def _check_parent_stage_proof(
    parent: Mapping[str, object], proof: object, *, rounds: int, num_envs: int
) -> None:
    """Check an archived stage proof against its exact parent segment budget.

    The proof is the small host snapshot made by collection at a checked fork.
    Counts must show the original completed stages and the current partial or
    complete stage. Snapshot/history identities remain tied to the parent
    checkpoint by the runner. Raises ValueError; never changes the proof.
    """
    checked_proof = cast(Mapping[str, Any], proof)
    keys = {
        "rounds",
        "stage_ordinal",
        "stage_round_budget",
        "stage_rounds",
        "stage_counts",
        "full_batch_rounds_valid",
        "completed_stage_counts",
        "stage_complete",
        "history_threshold_to_snapshot",
    }
    if (
        not isinstance(proof, Mapping)
        or set(checked_proof) != keys
        or checked_proof["rounds"] != rounds
        or checked_proof["full_batch_rounds_valid"] is not True
    ):
        raise ValueError("Continuation parent stage proof is malformed")
    if any(
        type(checked_proof[key]) is not int
        for key in ("rounds", "stage_ordinal", "stage_round_budget", "stage_rounds")
    ):
        raise ValueError("Continuation parent proof counts must be integers")
    for name, shape, kind in (
        ("stage_counts", (num_envs, 3), "iu"),
        ("completed_stage_counts", (_STAGES, num_envs, 3), "iu"),
        ("stage_complete", (_STAGES,), "b"),
        ("history_threshold_to_snapshot", (20,), "iu"),
    ):
        values = np.asarray(checked_proof[name])
        if values.shape != shape or values.dtype.kind not in kind:
            raise ValueError(
                "Continuation parent proof rows have invalid types or shapes"
            )
    history_mapping = np.asarray(checked_proof["history_threshold_to_snapshot"])
    if np.any(history_mapping < -1) or np.any(
        history_mapping >= np.iinfo(np.int32).max
    ):
        raise ValueError(
            "Continuation parent history mapping has invalid stable capture IDs"
        )
    count = cast(int, parent["stage_count"])
    ends = np.asarray(parent["round_ends"])
    budgets = np.asarray(parent["round_budgets"])
    stage = min(int(np.searchsorted(ends[:count], rounds, side="right")), count - 1)
    current_rounds = rounds - (
        int(ends[stage - 1]) if stage else cast(int, parent["round_offset"])
    )
    complete = (np.arange(_STAGES) < count) & (ends <= rounds)
    expected = np.zeros((_STAGES, num_envs, 3), np.int64)
    expected[complete, : num_envs // 2, 0] = budgets[complete, None]
    expected[complete, num_envs // 2 :, 1] = budgets[complete, None]
    current = np.zeros((num_envs, 3), np.int64)
    current[: num_envs // 2, 0] = current_rounds
    current[num_envs // 2 :, 1] = current_rounds
    if (
        checked_proof["stage_ordinal"] != stage
        or checked_proof["stage_round_budget"] != int(budgets[stage])
        or checked_proof["stage_rounds"] != current_rounds
        or not np.array_equal(checked_proof["stage_counts"], current)
        or not np.array_equal(checked_proof["stage_complete"], complete)
        or not np.array_equal(checked_proof["completed_stage_counts"], expected)
    ):
        raise ValueError(
            "Continuation parent stage proof differs from its exact budget"
        )


def _restore_continuation_schedule(
    declaration: Mapping[str, object],
) -> TrainingSchedule:
    """Restore a checked child schedule from saved host metadata only.

    declaration is a JSON mapping previously returned by _continuation_details.
    It must include the frozen parent stage proof; malformed budgets, mappings
    or proofs raise ValueError. The optional learner mapping is retained for
    its learner owner to validate. No arrays from a live game, files or random
    keys are read. Use this schedule only with its restored full learner state.
    """
    return _build_continuation_schedule(declaration, require_parent_proof=True)


def _with_history_capture_rounds(
    schedule: TrainingSchedule, rounds: tuple[int, ...]
) -> TrainingSchedule:
    """Declare a fresh schedule's explicit captures, including an empty table.

    rounds contains at most 20 strictly increasing positive full-round targets
    within this schedule. The runner converts transition targets and validates
    game-length spacing. None on the descriptor means the unchanged 20-point
    default. Children already declare their pending absolute table through the
    continuation owner and cannot be changed by this helper. No game runs.
    """
    if schedule.continuation is not None:
        raise ValueError("Change child captures through its continuation declaration")
    if (
        not isinstance(cast(object, rounds), tuple)
        or len(rounds) > 20
        or any(
            type(value) is not int or not 0 < value <= int(schedule.arrays.total_rounds)
            for value in rounds
        )
        or any(a >= b for a, b in pairwise(rounds))
    ):
        raise ValueError(
            "Capture rounds must be at most 20 increasing in-budget integers"
        )
    padding = (0,) * (20 - len(rounds))
    return replace(
        schedule,
        history_capture_rounds=rounds,
        rounding_report=MappingProxyType(
            {
                **schedule.rounding_report,
                "history_thresholds": "Explicit capture targets"
                if rounds
                else "No explicit capture targets",
                "history_capture_rounds": rounds,
            }
        ),
        arrays=schedule.arrays._replace(
            history_threshold_rounds=jnp.asarray((*rounds, *padding), jnp.int32),
            history_threshold_count=jnp.int32(len(rounds)),
        ),
    )


def _check_training_schedule(schedule: TrainingSchedule) -> None:
    """Reject a schedule whose arrays or report differ from its declaration.

    Fresh schedules reconstruct through make_training_schedule; children use
    their saved root, offset and parent proof. Host-only, with small array reads.
    Raises ValueError before a malformed schedule can run or fork.
    """
    if not isinstance(cast(object, schedule), TrainingSchedule):
        raise ValueError("Schedule must be a TrainingSchedule")
    checked = (
        make_training_schedule(
            total_env_steps=schedule.total_env_steps,
            num_envs=schedule.num_envs,
            curriculum=schedule.curriculum,
            score_threshold_curriculum=schedule.score_threshold_curriculum,
            early_history_capture=schedule.early_history_capture,
        )
        if schedule.continuation is None
        else _restore_continuation_schedule(schedule.continuation)
    )
    if schedule.continuation is None and schedule.history_capture_rounds is not None:
        checked = _with_history_capture_rounds(checked, schedule.history_capture_rounds)
    if (
        (
            schedule.total_env_steps,
            schedule.num_envs,
            schedule.curriculum,
            schedule.score_threshold_curriculum,
            schedule.early_history_capture,
        )
        != (
            checked.total_env_steps,
            checked.num_envs,
            checked.curriculum,
            checked.score_threshold_curriculum,
            checked.early_history_capture,
        )
        or jax.tree.structure(schedule.arrays) != jax.tree.structure(checked.arrays)
        or any(
            left.dtype != right.dtype or not np.array_equal(left, right)
            for left, right in zip(
                jax.tree.leaves(schedule.arrays),
                jax.tree.leaves(checked.arrays),
                strict=True,
            )
        )
        or dict(schedule.rounding_report) != dict(checked.rounding_report)
    ):
        raise ValueError(
            "Training schedule arrays and report must match its declared settings"
        )


def _init_training_progress(  # pyright: ignore[reportUnusedFunction]
    *,
    num_envs: int,
    history_capture_capacity: int = 20,
    opponent_members: int = 0,
    partner_members: int = 0,
) -> TrainingProgress:
    """Create zeroed dynamic counters for fresh stage-zero games in B lanes.

    num_envs is a positive even Python integer already checked by setup. Return
    history_capture_capacity is the nonnegative maximum total capture count.
    Return TrainingProgress with fixed stage/map/identity capacity; no content is read,
    no game is reset and no input is changed. Call once on the host, then carry
    the returned arrays through collection. Wrong batch values raise ValueError.
    """
    if isinstance(num_envs, bool) or not isinstance(cast(object, num_envs), int):
        raise ValueError("num_envs must be a positive even Python integer")
    if num_envs <= 0 or num_envs % 2:
        raise ValueError("num_envs must be a positive even Python integer")
    if type(history_capture_capacity) is not int or history_capture_capacity < 0:
        raise ValueError("history_capture_capacity must be a nonnegative integer")
    if type(opponent_members) is not int or opponent_members < 0:
        raise ValueError("opponent_members must be a nonnegative integer")
    if type(partner_members) is not int or partner_members < 0:
        raise ValueError("partner_members must be a nonnegative integer")
    opponent_rows = history_capture_capacity + 2 + opponent_members
    return TrainingProgress(
        rounds=jnp.zeros((), jnp.int32),
        episode_stage=jnp.zeros(num_envs, jnp.int32),
        starts=jnp.zeros((_STAGES, num_envs), jnp.int32),
        exposure=jnp.zeros((_STAGES, num_envs), jnp.int32),
        actor_decisions=jnp.zeros((_STAGES, num_envs, 5), jnp.int32),
        completed_stage_counts=jnp.zeros((_STAGES, num_envs, 3), jnp.int32),
        stage_complete=jnp.zeros(_STAGES, jnp.bool_),
        map_steps=jnp.zeros((_MAPS, num_envs), jnp.int32),
        opponent_starts=jnp.zeros((opponent_rows, num_envs), jnp.int32),
        opponent_steps=jnp.zeros((opponent_rows, num_envs), jnp.int32),
        partner_used=None
        if not partner_members
        else jnp.zeros(partner_members, jnp.bool_),
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
        Host-validated dynamic controls for this segment and batch. Stage
        proofs use local slots; distribution exposure keeps original slots and
        real experience stays cumulative across the segment offset.

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
    offset = 0 if schedule.round_offset is None else schedule.round_offset
    distribution_count = (
        schedule.stage_count
        if schedule.distribution_count is None
        else schedule.distribution_count
    )
    previous_end = jnp.where(
        safe_stage > 0, schedule.round_ends[safe_stage - 1], offset
    )
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
        | (progress.rounds < offset)
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
            | (progress.episode_stage >= distribution_count)
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
