"""Collect JAX rollouts while keeping optional recording storage bounded.

Researchers supply their ordinary scan step and numerical carry. The host
collector runs reusable compiled chunks, stores only requested learner outputs,
and drains compact records through RunWriter. It never owns the optimizer or
changes game rules. Use direct scan when no files or outer differentiation are
needed; this optional host entry point is not itself jittable.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache, partial
from numbers import Integral
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import DTypeLike

from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation.collection_types import (
    CollectedAssignments,
    CollectedBatch,
    CollectedBuffers,
    CollectedCompletions,
    CollectedCounts,
    CollectedStarts,
    CollectionErrors,
    ConfigEvidence,
)
from marl_battlegrounds.evaluation.episode_metrics import MetricValues
from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
    PRIORITY_METRIC_NAMES,
)
from marl_battlegrounds.evaluation.policy_execution import PolicyTrace
from marl_battlegrounds.evaluation.recording_types import EpisodeStartRecords
from marl_battlegrounds.evaluation.replay_capture import ReplayPackets

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.run_writer import RunWriter

type StepFunction = Callable[
    [Any, None], tuple[Any, tuple[Any, EpisodeInfo, PolicyTrace | None]]
]
type DrainFunction = Callable[[CollectedBatch], None]
type CompilerOptions = tuple[tuple[str, bool | int | float | str], ...]

_MAX_COUNT = int(np.iinfo(np.int32).max)


@dataclass(frozen=True, eq=False)
class _StepIdentity:
    """Cache a callable by identity, including callable objects without a hash.

    function is the original pure scan callback. The bounded cache keeps at most
    sixteen callable/capacity combinations alive, never numerical input carries.
    """

    function: StepFunction

    def __hash__(self) -> int:
        """Return the live callable's identity for the bounded factory cache."""
        return id(self.function)

    def __eq__(self, other: object) -> bool:
        """Compare callable identity, never callable-defined numerical equality."""
        return isinstance(other, _StepIdentity) and self.function is other.function


class _Prepared(NamedTuple):
    """Keep host-only abstract output structure and checked buffer dimensions."""

    transition: Any
    info: EpisodeInfo
    trace: PolicyTrace | None
    scalar: bool
    batch_size: int
    capacity: int
    replay_capacity: int


def _integer(value: object, name: str, minimum: int = 0) -> int:
    """Check a host integer before narrowing it to an int32 index.

    name labels errors; minimum defaults to zero. Booleans, tracers, negative
    values and values above the int32 limit fail with ValueError.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{minimum}, {_MAX_COUNT}]")
    integer = int(value)
    if not minimum <= integer <= _MAX_COUNT:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_MAX_COUNT}]")
    return integer


def _compiler_options(
    value: Mapping[str, bool | int | float | str] | None,
) -> CompilerOptions:
    """Copy optional scalar JAX compiler settings into an immutable cache key.

    None preserves ordinary JAX defaults. Names must be nonempty strings and
    values must be Python bool, int, finite float or str. Reject malformed
    scalar settings before tracing a callback or changing a writer. JAX checks
    supported names and values later, when the kernel is compiled.
    """
    if value is None:
        return ()
    if not isinstance(cast(object, value), Mapping):
        raise TypeError("compiler_options must be a mapping or None")
    for name, option in value.items():
        if not isinstance(cast(object, name), str) or not name:
            raise ValueError("Compiler option names must be nonempty strings")
        if type(option) not in (bool, int, float, str):
            raise TypeError("Compiler option values must be Python scalar values")
        if isinstance(option, float) and not np.isfinite(option):
            raise ValueError("Compiler option floats must be finite")
    return tuple(sorted(value.items()))


def _array_spec(
    value: object, shape: tuple[int, ...], dtype: DTypeLike, name: str
) -> None:
    """Reject a malformed abstract numerical field before executing the callback."""
    if not isinstance(value, jax.ShapeDtypeStruct) or (
        value.shape != shape or value.dtype != np.dtype(dtype)
    ):
        raise ValueError(f"{name} must have shape {shape} and dtype {np.dtype(dtype)}")


def _prepare(
    step_fn: StepFunction, carry: object, record_capacity: int | None
) -> _Prepared:
    """Trace pure output shapes, validate record structure and resolve capacities.

    This executes no numerical action. The callback is traced using the same
    (carry, None) arguments as direct scan. Source values and parameters remain
    inputs; only returned array shapes determine allocation. Invalid public
    records or capacities fail before collection advances.
    """
    result: object = jax.eval_shape(step_fn, carry, None)
    if not isinstance(result, tuple) or len(cast(tuple[object, ...], result)) != 2:
        raise ValueError("step_fn must return (next_carry, (transition, info, trace))")
    output = cast(tuple[object, ...], result)[1]
    if not isinstance(output, tuple) or len(cast(tuple[object, ...], output)) != 3:
        raise ValueError("step_fn output must contain transition, info and trace")
    transition, info, trace = cast(tuple[object, object, object], output)
    if not isinstance(info, EpisodeInfo):
        raise TypeError("step_fn info must be EpisodeInfo")
    leading = info.episode_id.shape
    if len(leading) > 1 or (leading and leading[0] <= 0):
        raise ValueError("collector accepts one scalar or native environment batch")
    scalar = not leading
    batch = 1 if scalar else leading[0]
    capacity = _integer(
        4 * batch if record_capacity is None else record_capacity,
        "record_capacity",
        batch,
    )
    _integer(4 * batch, "assignment capacity", 1)
    for name in ("episode_id", "outcome", "decision_step", "episode_length"):
        _array_spec(getattr(info, name), leading, np.int32, f"info.{name}")
    for name in ("completed", "lifecycle_error"):
        _array_spec(getattr(info, name), leading, np.bool_, f"info.{name}")
    _array_spec(info.team_scores, (*leading, 2), np.int32, "info.team_scores")
    if not isinstance(cast(object, info.config), EnvConfig):
        raise TypeError("info.config must contain the actual EnvConfig")
    for value in jax.tree.leaves(info.config):
        if value.shape[: len(leading)] != leading:
            raise ValueError("info.config leaves must share the info batch axes")
    _array_spec(
        info.config.agent_profile.active_mask,
        (*leading, 10),
        np.bool_,
        "info.config.agent_profile.active_mask",
    )
    _array_spec(
        info.config.agent_profile.class_ids,
        (*leading, 10),
        np.int32,
        "info.config.agent_profile.class_ids",
    )
    for name, width in (
        ("priority", len(PRIORITY_METRIC_NAMES)),
        ("full", len(FULL_METRIC_NAMES)),
    ):
        metric = getattr(info, name)
        if metric is not None:
            if not isinstance(metric, MetricValues):
                raise TypeError(f"info.{name} must be MetricValues or None")
            _array_spec(metric.values, (*leading, width), np.float32, f"{name}.values")
            _array_spec(metric.valid, (*leading, width), np.bool_, f"{name}.valid")
    if info.episode_tracking_error is not None:
        _array_spec(
            info.episode_tracking_error,
            leading,
            np.int32,
            "info.episode_tracking_error",
        )
    starts = info.episode_start_records
    if starts is not None:
        if not isinstance(cast(object, starts), EpisodeStartRecords):
            raise TypeError("episode_start_records must be EpisodeStartRecords")
        for name in EpisodeStartRecords._fields:
            if name == "source_class_ids" and starts.source_class_ids is None:
                continue
            dtype = (
                np.bool_
                if name in {"valid", "source_known", "authored_start"}
                else np.int32
            )
            shape = leading
            if name == "source_table_id":
                dtype, shape = np.uint32, (*leading, 8)
            elif name == "source_class_ids":
                shape = (*leading, 10)
            _array_spec(getattr(starts, name), shape, dtype, f"starts.{name}")
    if trace is not None:
        if not isinstance(trace, PolicyTrace):
            raise TypeError("trace must be PolicyTrace or None")
        for name in PolicyTrace._fields:
            shape = (batch, 10) if name == "policy_ids" else (batch,)
            _array_spec(
                getattr(trace, name),
                shape,
                np.bool_ if name == "valid" else np.int32,
                f"trace.{name}",
            )
    replay_capacity = 0
    if info.replay is not None:
        if (
            not isinstance(cast(object, info.replay), ReplayPackets)
            or info.replay.valid.ndim != 1
        ):
            raise ValueError("replay must be one per-step ReplayPackets batch")
        replay_capacity = info.replay.valid.shape[0]
        _integer(4 * replay_capacity, "replay capacity")
        for value in jax.tree.leaves(info.replay):
            if not value.shape or value.shape[0] != replay_capacity:
                raise ValueError("replay leaves must share the packet capacity axis")
        _array_spec(info.replay.valid, (replay_capacity,), np.bool_, "replay.valid")
    return _Prepared(transition, info, trace, scalar, batch, capacity, replay_capacity)


def _zeros_from_shape[Tree](tree: Tree, length: int, leading_axes: int) -> Tree:
    """Allocate owned zero arrays, replacing a record's leading axes by length."""

    def allocate(value: jax.ShapeDtypeStruct) -> Array:
        """Allocate one leaf from its abstract shape without reading input data."""
        return jnp.zeros((length, *value.shape[leading_axes:]), value.dtype)

    return jax.tree.map(allocate, tree)


def _empty_counts() -> CollectedCounts:
    """Create one zero int32 occupied count for each recording family."""
    return CollectedCounts(
        *(jnp.asarray(0, jnp.int32) for _ in CollectedCounts._fields)
    )


def _empty_errors(batch: int) -> CollectionErrors:
    """Create clear lane flags and absent first-failure diagnostics for B lanes."""
    return CollectionErrors(
        jnp.zeros(batch, bool),
        jnp.zeros(batch, jnp.int32),
        jnp.zeros(batch, jnp.int32),
        *(jnp.asarray(-1, jnp.int32) for _ in range(5)),
    )


def _empty_batch(prepared: _Prepared) -> CollectedBatch:
    """Allocate only present recording families, with the reviewed fixed capacities."""
    info, trace = prepared.info, prepared.trace
    size, batch = prepared.capacity, prepared.batch_size
    axes = 0 if prepared.scalar else 1

    def integers(length: int = size) -> Array:
        """Allocate a fresh owned int32 vector for one private record field."""
        return jnp.zeros(length, jnp.int32)

    evidence = ConfigEvidence(
        integers(), integers(), _zeros_from_shape(info.config, size, axes)
    )
    starts = (
        None
        if info.episode_start_records is None
        else CollectedStarts(
            _zeros_from_shape(info.episode_start_records, size, axes),
            integers(),
            integers(),
            integers(),
        )
    )
    completions = CollectedCompletions(
        integers(),
        integers(),
        integers(),
        integers(),
        jnp.zeros((size, 2), jnp.int32),
        integers(),
        integers(),
        integers(),
    )
    assignments = (
        None
        if trace is None
        else CollectedAssignments(
            _zeros_from_shape(trace, 4 * batch, 1),
            *(integers(4 * batch) for _ in range(4)),
        )
    )
    buffers = CollectedBuffers(
        evidence,
        starts,
        completions,
        None if info.priority is None else _zeros_from_shape(info.priority, size, axes),
        None if info.full is None else _zeros_from_shape(info.full, size, axes),
        assignments,
        None
        if not prepared.replay_capacity
        else _zeros_from_shape(info.replay, 4 * prepared.replay_capacity, 1),
    )
    return CollectedBatch(buffers, _empty_counts(), _empty_errors(batch))


def _append[Tree](
    tree: Tree, values: Tree, valid: Array, count: Array
) -> tuple[Tree, Array, Array]:
    """Append admitted rows stably and return per-input row references.

    Every leaf has a leading capacity axis. The caller reserves len(valid) slots
    before the step. Invalid packed rows scatter out of bounds with drop mode;
    they cannot overwrite a valid row or turn -1 into a last-row reference.
    """
    capacity = jax.tree.leaves(tree)[0].shape[0]
    number = jnp.sum(valid, dtype=jnp.int32)
    selected = jnp.nonzero(valid, size=valid.shape[0], fill_value=0)[0]
    offset = jnp.arange(valid.shape[0], dtype=jnp.int32)
    destination = jnp.where(offset < number, count + offset, capacity)

    def write(buffer: Array, value: Array) -> Array:
        """Append one leaf, dropping the invalid packed tail."""
        return buffer.at[destination].set(value[selected], mode="drop")

    updated = jax.tree.map(write, tree, values)
    indices = jnp.where(valid, count + jnp.cumsum(valid, dtype=jnp.int32) - 1, -1)
    return updated, count + number, indices


def _equal_config_rows(first: EnvConfig, second: EnvConfig) -> Array:
    """Compare corresponding B configurations without changing serialized identity.

    Float leaves compare their bit patterns, including signed zero. Reduce only
    payload axes, leaving one Boolean per lane. No configuration cross-product,
    hashing, source reconstruction or host transfer is performed.
    """
    equal = jnp.ones(first.agent_profile.active_mask.shape[0], bool)
    for left, right in zip(
        jax.tree.leaves(first), jax.tree.leaves(second), strict=True
    ):
        if jnp.issubdtype(left.dtype, jnp.floating):
            left = jax.lax.bitcast_convert_type(left, jnp.uint8)
            right = jax.lax.bitcast_convert_type(right, jnp.uint8)
        matches = left == right
        equal &= jnp.all(matches, axis=tuple(range(1, matches.ndim)))
    return equal


def _append_evidence(
    evidence: ConfigEvidence, count: Array, info: EpisodeInfo
) -> tuple[ConfigEvidence, Array, Array, Array]:
    """Retain all producing configurations, including no-output and padding rows.

    Match only occupied table IDs, gather matching rows, and compare exact bits.
    New episodes append once. Duplicate IDs within one native step are invalid
    because recording identities name individual games, not interchangeable lanes.
    Return table, count, per-lane references and private per-lane error bits.
    """
    ids = info.episode_id
    occupied = jnp.arange(evidence.episode_id.shape[0]) < count
    matches = (ids[:, None] == evidence.episode_id[None, :]) & occupied[None, :]
    known = jnp.any(matches, axis=1)
    index = jnp.argmax(matches, axis=1).astype(jnp.int32)

    def gather(value: Array) -> Array:
        """Read at most one previous configuration per lane."""
        return value[index]

    previous = jax.tree.map(gather, evidence.config)
    changed = known & ~_equal_config_rows(previous, info.config)
    duplicate = jnp.sum(ids[:, None] == ids[None, :], axis=1) > 1
    codes = jnp.where(changed, 4, 0) | jnp.where((ids <= 0) | duplicate, 16, 0)
    table, total, added = _append(
        evidence, ConfigEvidence(ids, info.decision_step, info.config), ~known, count
    )
    return table, total, jnp.where(known, index, added), codes.astype(jnp.int32)


def _accumulate_errors(
    errors: CollectionErrors,
    info: EpisodeInfo,
    code: Array,
    cursor: Array,
    expected: Array,
    observed: Array,
) -> CollectionErrors:
    """Keep failures and the first bad step, lane and available integer evidence.

    expected/observed are per-lane ID, action-index or validity evidence; -1 means
    no exact expected value is available. They explain an error without fixing it.
    """
    tracking = (
        jnp.zeros_like(code)
        if info.episode_tracking_error is None
        else info.episode_tracking_error
    )
    bad = info.lifecycle_error | (tracking != 0) | (code != 0)
    first = (errors.step < 0) & jnp.any(bad)
    lane = jnp.argmax(bad).astype(jnp.int32)
    return CollectionErrors(
        errors.lifecycle_error | info.lifecycle_error,
        errors.episode_tracking_error | tracking,
        errors.code | code,
        jnp.where(first, cursor, errors.step),
        jnp.where(first, lane, errors.lane),
        jnp.where(first, info.episode_id[lane], errors.episode_id),
        jnp.where(first, expected[lane], errors.expected),
        jnp.where(first, observed[lane], errors.observed),
    )


def _consume(
    batch: CollectedBatch,
    raw_info: EpisodeInfo,
    trace: PolicyTrace | None,
    cursor: Array,
    scalar: bool,
) -> CollectedBatch:
    """Append one admitted step without retaining a dense info or final-data tree."""
    replay = raw_info.replay
    info = raw_info._replace(replay=None, final=None)
    if scalar:

        def add_batch(value: Array) -> Array:
            """Normalize a scalar info leaf to one internal lane."""
            return jnp.expand_dims(value, 0)

        info = jax.tree.map(add_batch, info)
    buffers, counts, errors = batch
    evidence, evidence_count, config_index, codes = _append_evidence(
        buffers.evidence, counts.evidence, info
    )
    expected = jnp.where(info.episode_id <= 0, 1, -1)
    observed = jnp.where(info.episode_id <= 0, info.episode_id, -1)
    starts, starts_count = buffers.starts, counts.starts
    if starts is not None:
        claims = info.episode_start_records
        assert claims is not None
        wrong_id = claims.episode_id != info.episode_id
        mismatch = claims.valid & (wrong_id | (info.decision_step != 0))
        report = mismatch & (expected < 0)
        expected = jnp.where(report, jnp.where(wrong_id, info.episode_id, 0), expected)
        observed = jnp.where(
            report,
            jnp.where(wrong_id, claims.episode_id, info.decision_step),
            observed,
        )
        codes |= jnp.where(mismatch, 2, 0)
        starts, starts_count, _ = _append(
            starts,
            CollectedStarts(claims, info.episode_id, info.decision_step, config_index),
            claims.valid,
            starts_count,
        )
    assignments, assignment_count = buffers.assignments, counts.assignments
    if assignments is not None:
        assert trace is not None
        wrong_id = trace.episode_id != info.episode_id
        mismatch = trace.valid & (
            wrong_id
            | (trace.decision_step != info.decision_step)
            | (info.decision_step < 0)
        )
        report = mismatch & (expected < 0)
        expected = jnp.where(
            report,
            jnp.where(wrong_id, info.episode_id, jnp.maximum(info.decision_step, 0)),
            expected,
        )
        observed = jnp.where(
            report,
            jnp.where(
                wrong_id,
                trace.episode_id,
                jnp.where(info.decision_step < 0, 1, trace.decision_step),
            ),
            observed,
        )
        codes |= jnp.where(mismatch, 2, 0)
        assignments, assignment_count, _ = _append(
            assignments,
            CollectedAssignments(
                trace,
                info.episode_id,
                info.decision_step,
                info.episode_length,
                config_index,
            ),
            trace.valid,
            assignment_count,
        )
    available_priority = jnp.zeros_like(info.completed)
    available_full = jnp.zeros_like(info.completed)
    if info.priority is not None:
        available_priority = info.completed & jnp.any(info.priority.valid, axis=-1)
    if info.full is not None:
        available_full = info.completed & jnp.any(info.full.valid, axis=-1)
    if info.priority is not None and info.full is not None:
        width = len(PRIORITY_METRIC_NAMES)
        mask_equal = jnp.all(info.priority.valid == info.full.valid[:, :width], axis=-1)
        values_equal = jnp.all(
            ~info.priority.valid
            | (info.priority.values == info.full.values[:, :width]),
            axis=-1,
        )
        codes |= jnp.where(
            available_priority & available_full & ~(mask_equal & values_equal), 8, 0
        )
    priority, full = buffers.priority, buffers.full
    priority_count, full_count = counts.priority, counts.full
    priority_index = jnp.full_like(info.episode_id, -1)
    full_index = jnp.full_like(info.episode_id, -1)
    if priority is not None:
        priority, priority_count, priority_index = _append(
            priority,
            info.priority,
            available_priority & ~available_full,
            priority_count,
        )
    if full is not None:
        full, full_count, full_index = _append(
            full, info.full, available_full, full_count
        )
    completions, completion_count, _ = _append(
        buffers.completions,
        CollectedCompletions(
            info.episode_id,
            info.outcome,
            info.decision_step,
            info.episode_length,
            info.team_scores,
            config_index,
            priority_index,
            full_index,
        ),
        info.completed,
        counts.completions,
    )
    replay_buffer, replay_count = buffers.replay, counts.replay
    if replay_buffer is not None:
        assert replay is not None
        replay_buffer, replay_count, _ = _append(
            replay_buffer, replay, replay.valid, replay_count
        )
    return CollectedBatch(
        CollectedBuffers(
            evidence, starts, completions, priority, full, assignments, replay_buffer
        ),
        CollectedCounts(
            evidence_count,
            starts_count,
            completion_count,
            priority_count,
            full_count,
            assignment_count,
            replay_count,
        ),
        _accumulate_errors(errors, info, codes, cursor, expected, observed),
    )


def _has_errors(errors: CollectionErrors) -> Array:
    """Return whether any lane has a lifecycle, tracking or collector failure."""
    return jnp.any(
        errors.lifecycle_error
        | (errors.episode_tracking_error != 0)
        | (errors.code != 0)
    )


@lru_cache(maxsize=16)
def _compiled_chunk(
    identity: _StepIdentity,
    output_steps: int,
    batch_size: int,
    capacity: int,
    replay_capacity: int,
    scalar: bool,
    compiler_options: CompilerOptions = (),
) -> Any:  # noqa: ANN401 - JAX returns a callable with lowering/cache methods.
    """Build a reusable chunk with dynamic carry, outputs, buffers and counters.

    Only the callback, structural sizes and compiler settings enter this cache.
    Numerical inputs are arguments. JAX owns any additional shape/dtype specializations.
    Only collector-owned learner/output buffers are donated; user carry is not.
    """

    def run(
        carry: object,
        output: object,
        buffers: CollectedBuffers,
        counts: CollectedCounts,
        errors: CollectionErrors,
        cursor: Array,
        total: Array,
    ) -> tuple[Any, Any, CollectedBuffers, CollectedCounts, CollectionErrors, Array]:
        """Advance until the next complete step would exceed a buffer or T."""

        def condition(
            current: tuple[
                Any, Any, CollectedBuffers, CollectedCounts, CollectionErrors, Array
            ],
        ) -> Array:
            """Reserve the worst possible whole-step emission before acting."""
            _, _, current_buffers, occupied, failures, offset = current
            room = (offset < total) & ~_has_errors(failures)
            for name in ("evidence", "completions", "starts", "priority", "full"):
                if getattr(current_buffers, name) is not None:
                    room &= getattr(occupied, name) <= capacity - batch_size
            if current_buffers.assignments is not None:
                room &= occupied.assignments <= 3 * batch_size
            if current_buffers.replay is not None:
                room &= occupied.replay <= 3 * replay_capacity
            return room

        def body(
            current: tuple[
                Any, Any, CollectedBuffers, CollectedCounts, CollectionErrors, Array
            ],
        ) -> tuple[
            Any, Any, CollectedBuffers, CollectedCounts, CollectionErrors, Array
        ]:
            """Execute exactly one callback and consume its same-decision output."""
            old_carry, learner, current_buffers, occupied, failures, offset = current
            next_carry, (transition, info, trace) = identity.function(old_carry, None)

            def retain(buffer: Array, row: Array) -> Array:
                """Write this decision's learner output at its logical index."""
                return buffer.at[offset].set(row)

            learner = jax.tree.map(retain, learner, transition)
            next_batch = _consume(
                CollectedBatch(current_buffers, occupied, failures),
                info,
                trace,
                offset,
                scalar,
            )
            return (
                next_carry,
                learner,
                next_batch.buffers,
                next_batch.counts,
                next_batch.errors,
                offset + jnp.int32(1),
            )

        return jax.lax.while_loop(
            condition, body, (carry, output, buffers, counts, errors, cursor)
        )

    return jax.jit(
        run, donate_argnums=(1, 2), compiler_options=dict(compiler_options) or None
    )


def _raise_errors(errors: CollectionErrors) -> None:
    """Reject host-visible flags before transferring or publishing larger records."""
    for name in ("lifecycle_error", "episode_tracking_error", "code"):
        values = np.asarray(getattr(errors, name))
        if np.any(values):
            evidence = (
                f"; expected {int(errors.expected)}, observed {int(errors.observed)}"
                if int(errors.expected) >= 0
                else ""
            )
            raise ValueError(
                f"collection rejected {name} at lanes "
                f"{np.flatnonzero(values).tolist()}; "
                f"first logical step {int(errors.step)}, lane {int(errors.lane)}, "
                f"episode {int(errors.episode_id)}{evidence}"
            )


def _prefixes(batch: CollectedBatch, counts: CollectedCounts) -> CollectedBatch:
    """Slice only occupied device prefixes; never transfer learner or invalid rows."""

    def take(row: Array, *, size: int) -> Array:
        """Return an occupied prefix while leaving payload data on its device."""
        return row[:size]

    parts: list[Any] = []
    for name in CollectedBuffers._fields:
        value = getattr(batch.buffers, name)
        size = int(getattr(counts, name))
        parts.append(
            None if value is None else jax.tree.map(partial(take, size=size), value)
        )
    return CollectedBatch(CollectedBuffers(*parts), counts, batch.errors)


def _collect(
    step_fn: StepFunction,
    carry: object,
    *,
    num_steps: int,
    drain: DrainFunction,
    record_capacity: int | None = None,
    output_steps: int | None = None,
    compiler_options: CompilerOptions = (),
) -> tuple[Any, Any]:
    """Run the bounded numerical engine through a supplied internal drain.

    drain receives occupied device prefixes after small errors are checked. This
    internal seam also lets tests compare numerical collection without inventing
    a public no-op writer. Exceptions propagate; no successful partial rollout
    is returned. Exactly num_steps callback executions occur on success.
    output_steps optionally reserves a larger returned tree; its suffix is zero
    and never reaches the drain. The real stopping cursor stays dynamic.
    compiler_options is the already checked immutable outer-JIT option list.
    """
    total = _integer(num_steps, "num_steps")
    capacity = (
        total if output_steps is None else _integer(output_steps, "output_steps", total)
    )
    prepared = _prepare(step_fn, carry, record_capacity)
    output = _zeros_from_shape(prepared.transition, capacity, 0)
    if not total:
        return carry, output
    batch = _empty_batch(prepared)
    arguments = (
        _StepIdentity(step_fn),
        capacity,
        prepared.batch_size,
        prepared.capacity,
        prepared.replay_capacity,
        prepared.scalar,
    )
    chunk = (
        _compiled_chunk(*arguments, compiler_options=compiler_options)
        if compiler_options
        else _compiled_chunk(*arguments)
    )
    cursor = jnp.asarray(0, jnp.int32)
    offset = 0
    while offset < total:
        carry, output, buffers, counts, errors, cursor = chunk(
            carry,
            output,
            batch.buffers,
            batch.counts,
            batch.errors,
            cursor,
            jnp.asarray(total, jnp.int32),
        )
        host_cursor, host_counts, host_errors = jax.device_get((cursor, counts, errors))
        _raise_errors(host_errors)
        next_offset = int(host_cursor)
        if next_offset <= offset:
            raise RuntimeError(
                "collector made no progress despite an empty recording buffer"
            )
        batch = CollectedBatch(buffers, counts, errors)
        drain(_prefixes(batch, host_counts))
        offset = next_offset
        batch = CollectedBatch(
            buffers, _empty_counts(), _empty_errors(prepared.batch_size)
        )
    return carry, output


def _check_collection_writer(writer: RunWriter, *, has_steps: bool) -> None:
    """Reject unusable recording ownership before any policy or provider call.

    writer must be an open, healthy RunWriter outside another collection.
    has_steps says whether work will advance; only advancing work refuses an
    unfinished manual start handoff. No state is changed. TypeError and
    RuntimeError retain the existing collector's meanings. Both numerical and
    host-method collectors use this preflight.
    """
    from marl_battlegrounds.evaluation.run_writer import RunWriter

    if not isinstance(cast(object, writer), RunWriter):
        raise TypeError("writer must be RunWriter")
    writer._check_open()  # pyright: ignore[reportPrivateUsage]
    if getattr(writer, "_collection_active", False):
        raise RuntimeError("writer is already collecting")
    if has_steps and writer.has_pending_numerical_starts:
        raise RuntimeError(
            "finish the pending manual register/write handoff before collection"
        )


def collect_rollout(
    step_fn: StepFunction,
    carry: object,
    *,
    num_steps: int,
    writer: RunWriter,
    record_capacity: int | None = None,
    source_configs: EnvConfig | None = None,
    output_steps: int | None = None,
    compiler_options: Mapping[str, bool | int | float | str] | None = None,
) -> tuple[Any, Any]:
    """Run an exact JAX rollout and write optional records using bounded buffers.

    Parameters
    ----------
    step_fn : callable
        Pure scan callback ``step_fn(carry, None)`` returning
        ``(next_carry, (transition, EpisodeInfo, PolicyTrace_or_None))``.
        Keep changing parameters and source values in the numerical carry.
    carry : numerical PyTree
        Caller-owned state. Return and reuse the latest carry; it is never
        silently donated or transferred to the host.
    num_steps : int
        Number of callback executions, from zero through the int32 limit.
        Padding is not real experience and is not replaced with extra steps.
    writer : RunWriter
        Open, healthy writer. Finish any earlier manual start-registration
        handoff before a nonempty collection. Existing dictionary schedules work.
    record_capacity : int or None, default None
        Start/completion/evidence capacity. None uses four environment batches;
        an explicit value must fit at least one batch. Booleans are rejected.
    source_configs : EnvConfig or None, default None
        Immutable source bank referenced by start records. Required for a new
        known source; the writer owns verified reuse. Never mutate or donate it.
    output_steps : int or None, default None
        Returned transition capacity, at least num_steps and at most the int32
        limit. None uses num_steps. A larger capacity adds zero rows without
        calling the callback or writing records. Booleans are rejected. Reusing
        a capacity and callback allows different real prefixes to share a
        compiled chunk. This option does not label learner-specific padding.
    compiler_options : mapping or None, default None
        Optional settings passed to the outer jax.jit compiler. None preserves
        its defaults. Names omit the leading "--" and map to Python bool, int,
        finite float or str values. JAX checks supported names and values at
        compilation. Settings enter the compiled-chunk cache key; changing
        them creates a separate kernel. Callbacks stay pure and composable.
        Built-in training uses this route to keep recovery compilation stable.

    Returns
    -------
    tuple
        Latest carry and the transition tree with leading output_steps (or
        num_steps when omitted). Drains do
        not change actions, random keys, resets, memory or learning updates.
        No wide info history is returned. Zero steps makes no writer changes.

    Raises
    ------
    TypeError, ValueError
        Invalid shapes, arguments, declarations, ownership or recording errors.
    RuntimeError
        Writer is closed/failed, has an unfinished manual start handoff, or is
        already collecting. A failed collection does not return partial success.
    OSError
        Recording fails. Close and restore through the writer's recovery route.

    Notes
    -----
    Host-only orchestration around compiled chunks; do not wrap this call in
    jit/vmap/grad. Use direct scan for no-file or differentiable rollouts. Only
    recording prefixes cross the host boundary. The writer owns publication;
    flush/close establishes durability. This does not save a learner checkpoint.
    """
    options = _compiler_options(compiler_options)
    total = _integer(num_steps, "num_steps")
    capacity = (
        total if output_steps is None else _integer(output_steps, "output_steps", total)
    )
    _check_collection_writer(writer, has_steps=bool(total))
    if not total:
        prepared = _prepare(step_fn, carry, record_capacity)
        return carry, _zeros_from_shape(prepared.transition, capacity, 0)
    writer._collection_active = True  # pyright: ignore[reportPrivateUsage]
    try:
        return _collect(
            step_fn,
            carry,
            num_steps=total,
            record_capacity=record_capacity,
            output_steps=capacity,
            compiler_options=options,
            drain=partial(writer._write_collected, source_configs=source_configs),  # pyright: ignore[reportPrivateUsage]
        )
    except BaseException as error:
        writer.record_failure(error)
        raise
    finally:
        writer._collection_active = False  # pyright: ignore[reportPrivateUsage]
