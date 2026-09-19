"""Check exact numerical rollout collection and bounded record ownership.

Synthetic numerical callbacks isolate collector behavior from simulator cost.
Direct scan is the independent trajectory reference. Tests check scalar/native
shapes, dynamic inputs, sparse/full records, first starts, padding, errors and
capacity pressure. Fixed output capacity must keep real prefixes exact, suffixes
zero and writer records unchanged while reusing the same compiled chunk. A
test-only drain consumes every admitted record; it does not stand in for writer
durability, which has separate integration tests.
"""

import csv
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, Never, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.collection import (
    _collect,  # pyright: ignore[reportPrivateUsage]
    _compiled_chunk,  # pyright: ignore[reportPrivateUsage]
    _compiler_options,  # pyright: ignore[reportPrivateUsage]
    _empty_batch,  # pyright: ignore[reportPrivateUsage]
    _prepare,  # pyright: ignore[reportPrivateUsage]
    _StepIdentity,  # pyright: ignore[reportPrivateUsage]
    collect_rollout,
)
from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation.collection_types import CollectedBatch
from marl_battlegrounds.evaluation.episode_metrics import MetricValues
from marl_battlegrounds.evaluation.metric_catalog import FULL_METRIC_NAMES
from marl_battlegrounds.evaluation.policy_execution import PolicyTrace
from marl_battlegrounds.evaluation.recording_types import EpisodeStartRecords
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


class _Carry(NamedTuple):
    config: EnvConfig
    key: Array
    episode_id: Array
    tick: Array
    lengths: Array
    memory: Array
    weight: Array
    rounds: Array


@pytest.fixture(scope="module")
def source() -> EnvConfig:
    return make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=("mage",), team_b_roster=("priest",)
    )


def _initial(source: EnvConfig, batch: int | None = 2) -> _Carry:
    size = batch or 1
    shape = () if batch is None else (size,)

    def repeat(value: Array) -> Array:
        return jnp.stack([value] * size)

    return _Carry(
        source if batch is None else jax.tree.map(repeat, source),
        jax.random.key(91),
        jnp.arange(1, size + 1, dtype=jnp.int32).reshape(shape),
        jnp.zeros(shape, jnp.int32),
        (jnp.arange(size, dtype=jnp.int32) + 1).reshape(shape),
        jnp.zeros(shape, jnp.float32),
        jnp.asarray(2, jnp.float32),
        jnp.asarray(0, jnp.int32),
    )


def _step(
    carry: _Carry,
    unused: None,
    *,
    metrics: str = "priority",
    starts: bool = True,
    traces: bool = True,
    padding: bool = False,
    fault: str = "",
    empty_transition: bool = False,
) -> tuple[_Carry, tuple[object, EpisodeInfo, PolicyTrace | None]]:
    del unused
    shape = carry.episode_id.shape
    batch = carry.episode_id.size
    key, draw = jax.random.split(carry.key)
    value = carry.memory + carry.weight * jax.random.uniform(draw, shape)
    advanced = ~(padding & (carry.tick >= carry.lengths))
    length = carry.tick + advanced.astype(jnp.int32)
    completed = advanced & (length >= carry.lengths)
    decision = jnp.where(advanced, carry.tick, -1)
    priority = None
    full = None
    if metrics != "none":
        values = jnp.broadcast_to(length[..., None].astype(jnp.float32), (*shape, 16))
        priority = MetricValues(values, jnp.ones((*shape, 16), bool))
    if metrics in {"full", "selected"}:
        width = len(FULL_METRIC_NAMES)
        values = jnp.broadcast_to(
            length[..., None].astype(jnp.float32), (*shape, width)
        )
        selected = completed & ((metrics == "full") | (carry.episode_id % 2 == 0))
        full = MetricValues(values, jnp.broadcast_to(selected[..., None], values.shape))
    claims = None
    if starts:
        valid = advanced & (carry.tick == 0)
        claims = EpisodeStartRecords(
            carry.episode_id + (1 if fault == "start_id" else 0),
            carry.rounds + jnp.zeros(shape, jnp.int32),
            jnp.zeros((*shape, 8), jnp.uint32),
            jnp.full(shape, -1, jnp.int32),
            jnp.full(shape, -1, jnp.int32),
            jnp.zeros(shape, jnp.int32),
            jnp.zeros(shape, bool),
            jnp.zeros(shape, bool),
            valid,
        )
    config = carry.config
    if fault == "config":
        config = config._replace(
            max_steps=config.max_steps + (carry.rounds == 1).astype(jnp.int32)
        )
    if fault == "signed_zero":
        config = config._replace(
            ordinary_movement_distance_scale=jnp.broadcast_to(
                jnp.where(carry.rounds == 1, jnp.float32(-0.0), jnp.float32(0.0)),
                shape,
            )
        )
    info = EpisodeInfo(
        carry.episode_id,
        completed,
        jnp.where(completed, 3, 0).astype(jnp.int32),
        config,
        priority,
        full,
        None,
        decision,
        length,
        jnp.zeros((*shape, 2), jnp.int32),
        jnp.full(shape, fault == "lifecycle", bool),
        claims,
        jnp.full(shape, 2 if fault == "tracking" else 0, jnp.int32),
    )
    trace = None
    if traces:
        trace = PolicyTrace(
            jnp.reshape(carry.episode_id, (batch,)),
            jnp.reshape(decision, (batch,)) + (1 if fault == "trace" else 0),
            jnp.ones((batch,), bool)
            if fault == "padded_trace"
            else jnp.reshape(advanced, (batch,)),
            jnp.zeros((batch, 10), jnp.int32),
        )
    next_carry = carry._replace(
        key=key,
        memory=value,
        tick=length if padding else jnp.where(completed, 0, length),
        episode_id=carry.episode_id
        if padding
        else carry.episode_id + completed.astype(jnp.int32) * batch,
        rounds=carry.rounds + 1,
    )
    transition = () if empty_transition else (value, carry.episode_id, decision)
    return next_carry, (transition, info, trace)


def _assert_tree(first: object, second: object) -> None:
    assert jax.tree.structure(first) == jax.tree.structure(second)
    for left, right in zip(
        jax.tree.leaves(first), jax.tree.leaves(second), strict=True
    ):
        if jnp.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        np.testing.assert_array_equal(left, right)


def _sink(records: list[CollectedBatch], batch: CollectedBatch) -> None:
    records.append(cast(CollectedBatch, jax.device_get(batch)))


@pytest.mark.parametrize(
    "batch,capacity,steps", [(None, 1, 5), (1, 4, 7), (2, 3, 7), (3, 12, 9)]
)
def test_collection_matches_direct_scan_under_repeated_drains(
    source: EnvConfig, batch: int | None, capacity: int, steps: int
) -> None:
    initial = _initial(source, batch)
    expected, (transitions, _, _) = jax.lax.scan(_step, initial, None, length=steps)
    records: list[CollectedBatch] = []
    actual, output = _collect(
        _step,
        initial,
        num_steps=steps,
        record_capacity=capacity,
        drain=partial(_sink, records),
    )
    _assert_tree(actual, expected)
    _assert_tree(output, transitions)
    assert int(actual.rounds) == steps
    assert sum(int(record.counts.assignments) for record in records) == steps * (
        batch or 1
    )
    assert all(int(record.counts.evidence) <= capacity for record in records)
    for record in records:
        for name in record.buffers._fields:
            family = getattr(record.buffers, name)
            if family is not None:
                assert all(
                    leaf.shape[0] == int(getattr(record.counts, name))
                    for leaf in jax.tree.leaves(family)
                )


@pytest.mark.parametrize("steps", [0, 1, 5])
def test_zero_and_empty_learner_output(source: EnvConfig, steps: int) -> None:
    initial = _initial(source)
    function = partial(_step, empty_transition=True)
    records: list[CollectedBatch] = []
    final, output = _collect(
        function, initial, num_steps=steps, drain=partial(_sink, records)
    )
    assert output == ()
    assert int(final.rounds) == steps
    if not steps:
        assert final is initial and records == []


@pytest.mark.parametrize("metrics", ["none", "priority", "full", "selected"])
def test_metric_rows_are_compact_and_full_prefix_is_shared(
    source: EnvConfig, metrics: str
) -> None:
    initial = _initial(source)
    records: list[CollectedBatch] = []
    _collect(
        partial(_step, metrics=metrics),
        initial,
        num_steps=5,
        drain=partial(_sink, records),
    )
    completions = sum(int(record.counts.completions) for record in records)
    priorities = sum(int(record.counts.priority) for record in records)
    full = sum(int(record.counts.full) for record in records)
    assert completions == 7
    if metrics == "none":
        assert priorities == full == 0
        assert all(
            record.buffers.priority is None and record.buffers.full is None
            for record in records
        )
    elif metrics == "priority":
        assert priorities == completions and full == 0
    else:
        assert priorities + full == completions
        assert full == (completions if metrics == "full" else 2)
        for record in records:
            headers = record.buffers.completions
            assert np.all((headers.priority_index >= 0) ^ (headers.full_index >= 0))


def test_padding_is_preserved_and_not_replaced_with_extra_steps(
    source: EnvConfig,
) -> None:
    initial = _initial(source)
    function = partial(_step, padding=True)
    records: list[CollectedBatch] = []
    expected, (transitions, _, _) = jax.lax.scan(function, initial, None, length=7)
    actual, output = _collect(
        function, initial, num_steps=7, drain=partial(_sink, records)
    )
    _assert_tree(actual, expected)
    _assert_tree(output, transitions)
    assert sum(int(record.counts.completions) for record in records) == 2
    assert sum(int(record.counts.assignments) for record in records) == 3


@pytest.mark.parametrize(
    "fault", ["tracking", "lifecycle", "trace", "config", "signed_zero"]
)
def test_bad_step_blocks_the_entire_current_drain(
    source: EnvConfig, fault: str
) -> None:
    initial = _initial(source)._replace(lengths=jnp.full(2, 100, jnp.int32))
    records: list[CollectedBatch] = []
    function = partial(
        _step, fault=fault, starts=False, traces=fault == "trace", metrics="none"
    )
    with pytest.raises(ValueError, match="collection rejected"):
        _collect(function, initial, num_steps=3, drain=partial(_sink, records))
    assert records == []


def test_no_output_evidence_uses_one_row_per_episode(source: EnvConfig) -> None:
    initial = _initial(source)._replace(lengths=jnp.full(2, 100, jnp.int32))
    records: list[CollectedBatch] = []
    _collect(
        partial(_step, starts=False, traces=False, metrics="none"),
        initial,
        num_steps=17,
        drain=partial(_sink, records),
    )
    assert len(records) == 1
    assert int(records[0].counts.evidence) == 2
    assert int(records[0].counts.completions) == 0


@pytest.mark.parametrize(
    "fault,expected,observed",
    [("trace", 0, 1), ("start_id", 1, 2), ("padded_trace", 0, 1)],
)
def test_ownership_errors_include_expected_and_observed_epoch_or_id(
    source: EnvConfig, fault: str, expected: int, observed: int
) -> None:
    records: list[CollectedBatch] = []
    with pytest.raises(ValueError, match=f"expected {expected}, observed {observed}"):
        _collect(
            partial(_step, fault=fault, padding=fault == "padded_trace"),
            _initial(source),
            num_steps=2,
            drain=partial(_sink, records),
        )
    assert records == []


def test_changed_values_reuse_the_numerical_kernel(source: EnvConfig) -> None:
    initial = _initial(source)
    prepared = _prepare(_step, initial, None)
    function = _compiled_chunk(_StepIdentity(_step), 5, 2, 8, 0, False)
    records: list[CollectedBatch] = []
    first, _ = _collect(_step, initial, num_steps=5, drain=partial(_sink, records))
    before = function._cache_size()
    changed = initial._replace(
        weight=jnp.float32(7),
        key=jax.random.key(111),
        config=initial.config._replace(max_steps=initial.config.max_steps + 1),
    )
    second, _ = _collect(_step, changed, num_steps=5, drain=partial(_sink, records))
    assert function._cache_size() == before
    assert not np.array_equal(first.memory, second.memory)
    buffers = _empty_batch(prepared).buffers
    assert buffers.evidence.episode_id.shape == (8,)
    assert (
        buffers.assignments is not None
        and buffers.assignments.trace.valid.shape == (8,)
    )


@pytest.mark.parametrize("output_steps", [None, 7, 11])
@pytest.mark.parametrize("compiler_options", [None, {}, {"xla_gpu_autotune_level": 0}])
def test_public_collector_saves_each_required_outcome_once(
    source: EnvConfig,
    tmp_path: Path,
    output_steps: int | None,
    compiler_options: dict[str, int] | None,
) -> None:
    initial = _initial(source)
    function = partial(_step, starts=False, traces=False, metrics="none")
    expected, (transitions, _, _) = jax.lax.scan(function, initial, None, length=7)
    with RunWriter(tmp_path, buffer_size=2) as writer:
        actual, output = collect_rollout(
            function,
            initial,
            num_steps=7,
            writer=writer,
            record_capacity=2,
            output_steps=output_steps,
            compiler_options=compiler_options,
        )
        _assert_tree(actual, expected)
        for full, real in zip(
            jax.tree.leaves(output), jax.tree.leaves(transitions), strict=True
        ):
            assert full.shape == (output_steps or 7, *real.shape[1:])
            np.testing.assert_array_equal(full[:7], real)
            np.testing.assert_array_equal(full[7:], jnp.zeros_like(full[7:]))
        run_dir = writer.run_dir
    with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert len({row["episode_id"] for row in rows}) == 10
    assert sorted(int(row["episode_length"]) for row in rows) == [1] * 7 + [2] * 3
    assert not (run_dir / "metrics.csv").exists()


@pytest.mark.parametrize("output_steps", [None, 0, 5])
@pytest.mark.parametrize("pending_starts", [False, True])
def test_zero_steps_leave_writer_files_and_pending_buffers_unchanged(
    source: EnvConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_steps: int | None,
    pending_starts: bool,
) -> None:
    initial = _initial(source)

    def unexpected_drain(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("zero steps must never drain")

    with RunWriter(tmp_path) as writer:
        if pending_starts:
            _, (_, info, _) = _step(initial, None)
            assert info.episode_start_records is not None
            writer.register_episodes(info.episode_start_records)
        assert writer.has_pending_numerical_starts == pending_starts
        before = {
            path.relative_to(writer.run_dir): path.read_bytes()
            for path in writer.run_dir.rglob("*")
            if path.is_file()
        }
        monkeypatch.setattr(writer, "_write_collected", unexpected_drain)
        actual, output = collect_rollout(
            _step, initial, num_steps=0, output_steps=output_steps, writer=writer
        )
        assert actual is initial
        for row in jax.tree.leaves(output):
            assert row.shape[0] == (output_steps or 0)
            np.testing.assert_array_equal(row, jnp.zeros_like(row))
        assert writer.has_pending_numerical_starts == pending_starts
        assert writer.completed_episode_ids == frozenset()
        after = {
            path.relative_to(writer.run_dir): path.read_bytes()
            for path in writer.run_dir.rglob("*")
            if path.is_file()
        }
        assert before == after


@pytest.mark.parametrize("batch", [None, 2])
def test_fixed_output_suffix_adds_no_start_completion_or_assignment_records(
    source: EnvConfig, batch: int | None
) -> None:
    initial = _initial(source, batch)
    records: list[CollectedBatch] = []
    expected, real = _collect(
        _step, initial, num_steps=3, drain=partial(_sink, records)
    )
    padded_records: list[CollectedBatch] = []
    actual, output = _collect(
        _step,
        initial,
        num_steps=3,
        output_steps=8,
        drain=partial(_sink, padded_records),
    )
    _assert_tree(actual, expected)
    assert int(actual.rounds) == 3
    for padded, prefix in zip(
        jax.tree.leaves(output), jax.tree.leaves(real), strict=True
    ):
        assert padded.shape == (8, *prefix.shape[1:])
        np.testing.assert_array_equal(padded[:3], prefix)
        np.testing.assert_array_equal(padded[3:], jnp.zeros_like(padded[3:]))
    assert len(padded_records) == len(records)
    for padded, prefix in zip(padded_records, records, strict=True):
        _assert_tree(padded, prefix)


def test_public_fixed_capacity_reuses_compilation_for_different_real_prefixes(
    source: EnvConfig, tmp_path: Path
) -> None:
    initial = _initial(source)
    function = partial(_step, starts=False, traces=False, metrics="none")
    kernel = _compiled_chunk(_StepIdentity(function), 8, 2, 8, 0, False)
    cache_size = 0
    for steps in (3, 5, 1):
        with RunWriter(tmp_path / str(steps)) as writer:
            actual, output = collect_rollout(
                function, initial, num_steps=steps, output_steps=8, writer=writer
            )
        assert int(actual.rounds) == steps
        assert kernel._cache_size() > 0
        if cache_size:
            assert kernel._cache_size() == cache_size
        cache_size = kernel._cache_size()
        for row in jax.tree.leaves(output):
            assert row.shape[0] == 8
            np.testing.assert_array_equal(row[steps:], jnp.zeros_like(row[steps:]))


@pytest.mark.parametrize("output_steps", [True, np.bool_(False), -1, 0, 1.5, 2**31])
def test_invalid_output_capacity_fails_before_callback_and_writer_changes(
    source: EnvConfig, tmp_path: Path, output_steps: object
) -> None:
    def unexpected_step(_carry: object, _unused: None) -> Never:
        raise AssertionError("invalid output capacity must reject before tracing")

    with RunWriter(tmp_path) as writer:
        before = (writer.run_dir / "run_details.json").read_bytes()
        with pytest.raises(ValueError, match="output_steps"):
            collect_rollout(
                unexpected_step,
                _initial(source),
                num_steps=1,
                output_steps=cast(int, output_steps),
                writer=writer,
            )
        assert (writer.run_dir / "run_details.json").read_bytes() == before
        assert writer.completed_episode_ids == frozenset()
        assert not list(writer.run_dir.glob("*.csv"))


def test_repeated_preparation_reuses_abstract_trace_for_changed_values(
    source: EnvConfig,
) -> None:
    initial = _initial(source)
    traced: list[None] = []

    def function(
        carry: _Carry, unused: None
    ) -> tuple[_Carry, tuple[object, EpisodeInfo, PolicyTrace | None]]:
        traced.append(None)
        return _step(carry, unused)

    first = _prepare(function, initial, None)
    changed = initial._replace(
        key=jax.random.key(991),
        weight=jnp.float32(9),
        config=initial.config._replace(max_steps=initial.config.max_steps + 4),
    )
    second = _prepare(function, changed, None)
    assert len(traced) == 1
    assert first == second


def test_pending_manual_start_rejects_before_any_callback(
    source: EnvConfig, tmp_path: Path
) -> None:
    initial = _initial(source)
    _, (_, info, _) = _step(initial, None)
    assert info.episode_start_records is not None

    def unexpected_step(_carry: object, _unused: None) -> Never:
        raise AssertionError("pending starts must reject before tracing or executing")

    with RunWriter(tmp_path) as writer:
        writer.register_episodes(info.episode_start_records)
        assert writer.has_pending_numerical_starts
        with pytest.raises(RuntimeError, match="pending manual"):
            collect_rollout(unexpected_step, initial, num_steps=1, writer=writer)


@pytest.mark.parametrize("steps", [True, -1, 1.5, 2**31])
def test_invalid_logical_length_fails_before_callback(
    source: EnvConfig, steps: object
) -> None:
    with pytest.raises(ValueError, match="num_steps"):
        _collect(
            _step, _initial(source), num_steps=cast(int, steps), drain=lambda _: None
        )


@pytest.mark.parametrize("capacity", [True, 0, 1, 2**31])
def test_invalid_capacity_fails_before_callback(
    source: EnvConfig, capacity: object
) -> None:
    with pytest.raises(ValueError, match="record_capacity"):
        _collect(
            _step,
            _initial(source),
            num_steps=1,
            record_capacity=cast(int, capacity),
            drain=lambda _: None,
        )


@pytest.mark.parametrize(
    "options",
    [[], {"": 0}, {3: 0}, {"xla_gpu_autotune_level": []}, {"x": float("nan")}],
)
def test_invalid_compiler_settings_reject_before_callback_and_writer(
    source: EnvConfig, tmp_path: Path, options: object
) -> None:
    def unexpected_step(_carry: object, _unused: None) -> Never:
        raise AssertionError("Invalid compiler settings must reject before tracing")

    with RunWriter(tmp_path) as writer:
        before = (writer.run_dir / "run_details.json").read_bytes()
        with pytest.raises((TypeError, ValueError), match=r"[Cc]ompiler"):
            collect_rollout(
                unexpected_step,
                _initial(source),
                num_steps=1,
                writer=writer,
                compiler_options=cast(Any, options),
            )
        assert (writer.run_dir / "run_details.json").read_bytes() == before
        assert writer.completed_episode_ids == frozenset()
        assert not list(writer.run_dir.glob("*.csv"))


def test_compiler_options_copy_values_and_separate_cached_kernels() -> None:
    options = {"xla_gpu_autotune_level": 0}
    frozen = _compiler_options(options)
    first = _compiled_chunk(
        _StepIdentity(_step), 5, 2, 8, 0, False, compiler_options=frozen
    )
    options["xla_gpu_autotune_level"] = 4
    assert frozen == (("xla_gpu_autotune_level", 0),)
    assert first is _compiled_chunk(
        _StepIdentity(_step),
        5,
        2,
        8,
        0,
        False,
        compiler_options=_compiler_options({"xla_gpu_autotune_level": 0}),
    )
    assert first is not _compiled_chunk(
        _StepIdentity(_step),
        5,
        2,
        8,
        0,
        False,
        compiler_options=_compiler_options(options),
    )
