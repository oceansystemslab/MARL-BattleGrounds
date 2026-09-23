"""Check the MAPPO cost tool's evidence guards and real reporting files.

These small CPU checks cover full-tree comparison, minimum timing samples,
explicit hardware selection, compact update summaries and actual quiet/verbose
writes. Failed comparisons retain every leaf path and unchanged tolerance, so
the next diagnostic can identify the first numerical boundary that differs.
These checks do not simulate games or establish GPU speed. The separate tool's
CPU smoke checks its real collection, reference update and compilation wiring.
"""

# The developer tool deliberately exposes small private measurement owners.
# pyright: reportPrivateUsage=false
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev import benchmark_mappo_training as bench

from marl_battlegrounds.baselines.ppo import PPOMetrics
from marl_battlegrounds.training._compilation import training_compiler_options
from marl_battlegrounds.training.learner import UpdateResult, UpdateSummary
from marl_battlegrounds.training.opponents import SnapshotEvent

type Tree = Any


def test_compile_uses_shared_policy_without_global_changes_or_value_retracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = bench._compile_measured
    observed: list[object] = []

    def measured(function: Tree, value: Tree, **options: Tree) -> Tree:
        observed.append(options["compiler_options"])
        return original(function, value, **options)

    monkeypatch.setattr(bench, "_compile_measured", measured)
    before = bench.execution_identity()

    def affine(value: jax.Array) -> jax.Array:
        return value * 2 + 1

    function, traces, costs = bench._compile(affine, jnp.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(function(jnp.asarray([1.0, 2.0])), [3.0, 5.0])
    np.testing.assert_array_equal(function(jnp.asarray([3.0, 4.0])), [7.0, 9.0])
    assert observed == [training_compiler_options()]
    assert len(traces) == 1
    assert not costs["host_callbacks"]
    assert bench.execution_identity() == before


def _result() -> UpdateResult:
    zero = cast(Tree, np.asarray(0, np.int32))
    true = cast(Tree, np.asarray(True))
    false = cast(Tree, np.asarray(False))
    scalar = cast(Tree, np.asarray(1.0, np.float32))
    metrics = PPOMetrics(
        *(cast(Tree, np.ones((1, 2), np.float32)) for _ in PPOMetrics._fields)
    )
    summary = UpdateSummary(
        scalar,
        scalar,
        cast(Tree, np.int32(8)),
        cast(Tree, np.int32(4)),
        cast(Tree, np.int32(8)),
        cast(Tree, np.zeros((17, 3), np.int32)),
        cast(Tree, np.zeros((17, 2), np.int32)),
        cast(Tree, np.zeros(17, np.int32)),
        cast(Tree, np.zeros(17, np.int32)),
    )
    snapshot = SnapshotEvent(
        false, zero - 1, zero - 1, zero - 1, cast(Tree, np.zeros(20, bool))
    )
    return UpdateResult(true, metrics, false, zero, snapshot, summary)


def test_full_tree_comparison_checks_keys_shapes_dtypes_and_every_leaf() -> None:
    a = (jax.random.key(1), jnp.asarray((1.0, 2.0)), jnp.asarray((2, 3)))
    bench._equal(a, a)
    bench._equal(a, (a[0], a[1] + 1e-6, a[2]))
    for changed in (
        (jax.random.key(2), a[1], a[2]),
        (a[0], a[1] + 1, a[2]),
        (a[0], a[1], a[2] + 1),
        (a[0], a[1][None], a[2]),
        (a[0], a[1], a[2].astype(jnp.float32)),
    ):
        with pytest.raises(AssertionError):
            bench._equal(a, changed)


def test_compiler_cost_pairs_keep_fixed_inputs_and_save_numerical_comparison(
    tmp_path: Path,
) -> None:
    class State(NamedTuple):
        carry: jax.Array

    def collect(carry: jax.Array) -> tuple[jax.Array, jax.Array]:
        return carry + 1, carry * 2

    def update(value: Tree) -> tuple[jax.Array, jax.Array]:
        state, carry, rows = value
        return state.carry + carry, rows * 3

    state = State(jnp.asarray([1.0, 2.0]))
    collector, _, _ = bench._compile(collect, state.carry)
    inputs = (state, *collector(state.carry))
    updater, _, _ = bench._compile(update, inputs)
    result = bench._compiler_comparison(
        tmp_path,
        collect,
        update,
        collector,
        updater,
        cast(Tree, inputs),
        updater(inputs),
    )
    assert result["execution_order"] == [
        ["policy", "default"],
        ["default", "policy"],
        ["policy", "default"],
    ]
    assert result["default_trace_counts"] == [1, 1]
    assert result["policy_options"] == training_compiler_options()
    assert result["comparison_options"] is None
    for name in ("policy", "default"):
        for values in result["warm_seconds"][name].values():
            assert len(values) == 3 and all(value > 0 for value in values)
    comparisons = json.loads(
        (tmp_path / "compiler_output_comparisons.json").read_text()
    )
    assert all(value["passed"] for value in comparisons.values())
    np.testing.assert_array_equal(state.carry, [1.0, 2.0])


def test_cost_requires_five_positive_samples_and_reports_real_steps() -> None:
    for values in ([1.0] * 4, [0.0] * 5, [-1.0] * 5, [float("nan")] * 5):
        with pytest.raises(ValueError):
            bench._cost(values, 8)
    result = bench._cost([1.0, 2.0, 3.0, 4.0, 5.0], 12)
    assert result["median_seconds"] == 3.0
    assert result["spread_seconds"] == 4.0
    assert result["real_transitions_per_second"] == 4.0


def test_comparison_saves_all_failures_and_keeps_nonfinite_values_json_safe(
    tmp_path: Path,
) -> None:
    expected = {
        "actor": np.asarray([0.0, 0.5], np.float32),
        "critic": np.asarray([1.0, 2.0], np.float32),
        "key": jax.random.key(42),
        "nonfinite": np.asarray([0.0], np.float32),
    }
    actual = {
        "actor": np.asarray([6e-6, 0.5], np.float32),
        "critic": np.asarray([1.0, 2.01], np.float32),
        "key": jax.random.key(43),
        "nonfinite": np.asarray([float("nan")], np.float32),
    }
    path = tmp_path / "comparison.json"
    with pytest.raises(AssertionError, match=r"actor.*critic.*key.*nonfinite"):
        bench._equal(actual, expected, report_path=path)
    report = json.loads(path.read_text())
    assert not report["passed"]
    assert len(report["failed_paths"]) == 4
    leaves = {leaf["path"]: leaf for leaf in report["leaves"]}
    actor = leaves["['actor']"]
    assert actor["tolerance_mismatches"] == 1
    assert actor["max_absolute_error"] == pytest.approx(6e-6)
    assert actor["max_relative_error"] > 1e30
    assert leaves["['nonfinite']"]["max_absolute_error"] is None
    assert report["float_rtol"] == 1e-5 and report["float_atol"] == 2e-6
    json.dumps(report, allow_nan=False)


def test_device_guards_require_exact_uuid_and_one_declared_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(jax, "devices", lambda: [object()])
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert bench._gpu_identity(None, cpu_smoke=True) == ("cpu", None)
    with pytest.raises(ValueError):
        bench._gpu_identity("GPU-declared", cpu_smoke=True)
    with pytest.raises(ValueError):
        bench._gpu_identity("GPU-declared", cpu_smoke=False)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError):
        bench._gpu_identity("GPU-declared", cpu_smoke=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-declared")
    calls: list[list[str]] = []

    def query(command: list[str], **_options: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="RTX 5090, GPU-declared, 01:00.0, 32768, test")

    monkeypatch.setattr(bench.subprocess, "run", query)
    assert bench._gpu_identity("GPU-declared", cpu_smoke=False)[0] == "gpu"
    assert "--id=GPU-declared" in calls[0]
    monkeypatch.setattr(jax, "devices", lambda: [object(), object()])
    with pytest.raises(ValueError):
        bench._gpu_identity("GPU-declared", cpu_smoke=False)


def test_device_host_checks_happen_before_any_backend_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def backend() -> str:
        events.append("backend")
        return "gpu"

    def devices() -> list[object]:
        events.append("devices")
        return [object()]

    def query(*_arguments: object, **_options: object) -> SimpleNamespace:
        events.append("query")
        return SimpleNamespace(stdout="RTX 5090, GPU-declared, 01:00.0, 32768, test")

    monkeypatch.setattr(jax, "default_backend", backend)
    monkeypatch.setattr(jax, "devices", devices)
    monkeypatch.setattr(bench.subprocess, "run", query)
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError):
        bench._gpu_identity("GPU-declared", cpu_smoke=False)
    with pytest.raises(ValueError):
        bench._gpu_identity(None, cpu_smoke=True)
    assert events == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-declared")
    bench._gpu_identity("GPU-declared", cpu_smoke=False)
    assert events == ["query", "backend", "devices"]
    events.clear()

    def wrong_card(*_arguments: object, **_options: object) -> SimpleNamespace:
        events.append("query")
        return SimpleNamespace(stdout="RTX 5090, GPU-other, 01:00.0, 32768, test")

    monkeypatch.setattr(bench.subprocess, "run", wrong_card)
    with pytest.raises(ValueError):
        bench._gpu_identity("GPU-declared", cpu_smoke=False)
    assert events == ["query"]


@pytest.mark.parametrize("updates", [5, 12])
def test_reporting_pass_writes_real_records_and_throttled_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, updates: int
) -> None:
    class State(NamedTuple):
        carry: Tree

    initial = State(np.int32(0))
    result = _result()
    inputs: list[int] = []
    estimates: list[int] = []
    original_estimate = bench.TrainingSpeedEstimate

    def estimate() -> Tree:
        estimates.append(1)
        return original_estimate()

    monkeypatch.setattr(bench, "TrainingSpeedEstimate", estimate)

    def collect(carry: Tree) -> tuple[Tree, tuple[()]]:
        inputs.append(int(carry))
        return carry + 1, ()

    def update(value: Tree) -> tuple[State, UpdateResult]:
        return State(value[1]), result

    report = bench._reporting_pairs(
        tmp_path, cast(Tree, initial), collect, update, transitions=8, updates=updates
    )
    assert inputs == list(range(updates)) * 4
    assert len(estimates) == 2
    assert report["identical_final_states"]
    assert [pair["execution_order"] for pair in report["pairs"]] == [
        ["quiet", "verbose"],
        ["verbose_again", "quiet_again"],
    ]
    assert all(
        len(pair["verbose_minus_quiet_update_seconds"]) == updates
        for pair in report["pairs"]
    )
    for name, verbose in (
        ("quiet", False),
        ("verbose", True),
        ("verbose_again", True),
        ("quiet_again", False),
    ):
        directory = tmp_path / name
        cost = report[name]
        records = [
            json.loads(line)
            for line in (directory / "training.jsonl").read_text().splitlines()
        ]
        assert [row["env_steps"] for row in records] == [
            8 * index for index in range(1, updates + 1)
        ]
        assert records[-1]["total_env_steps"] == updates * 8
        assert (records[-1]["estimated_transitions_per_second"] is not None) == verbose
        assert (
            json.loads((directory / "status.json").read_text())["completed_updates"]
            == updates
        )
        assert (
            json.loads((directory / "status.json").read_text())["phase"] == "finished"
        )
        assert len(cost["update_seconds"]) == updates
        assert cost["compact_transferred_bytes"] == updates * bench._bytes(result)
        assert cost["updates"] == updates and cost["cadence_seconds"] == 10
        assert cost["stdout_lines"] >= 3 if verbose else cost["stdout_lines"] == 0
        assert records[-1]["task_reward_mean"] == 0.125


def test_unperformed_or_failed_update_never_becomes_reporting_evidence() -> None:
    result = _result()
    for changed in (
        result._replace(performed=cast(Tree, np.bool_(False))),
        result._replace(failed=cast(Tree, np.bool_(True))),
    ):
        with pytest.raises(AssertionError):
            bench._status(changed, index=1, transitions=8, elapsed=1.0)
