"""Measure the complete MAPPO collection/update boundary and reporting costs.

Run ``python -m scripts.dev.benchmark_mappo_training --gpu-uuid GPU-...
--output DIRECTORY`` after verifying the internal RTX 5090's physical identity.
CUDA_VISIBLE_DEVICES must equal that UUID. GPU work is fixed at B32/T128 with
the donor PPO settings. ``JAX_PLATFORMS=cpu ... --cpu-smoke`` uses B4/T2 and one
PPO epoch only to check this tool. Both modes take five synchronized samples.

The reference independently joins the existing critic, PPO and history calls.
All outer compiled calls use the shared training compiler policy, including
the independent reference. No global compiler flag is set by this tool.
It shares collection, equations and reporting reductions with production. It
also skips the final critic call when all games ended, preserving the same
bootstrap branch boundary. Full states and outputs must agree before timings
are accepted. Integer and key
leaves agree exactly; float32 leaves use rtol=1e-5 and atol=2e-6. It measures a
fresh full block, ordinary changed values, and two reporting pairs in reversed
order. Each GPU mode performs twelve real successive updates per pair so the
normal ten-second progress line is exercised; CPU smoke uses five. Recording is
disabled. Existing directories are preserved. A separate three-pair comparison
measures default compiler settings on the same immutable inputs and reports any
numerical differences; it is not the production/reference acceptance check.

The default tool writes source identity, cost JSON, real update/status files and
captured stdout files. That lower-level route does not save checkpoints, run
validation, plot learning curves or establish useful learning.

Use ``timeout 600s python -m scripts.dev.benchmark_mappo_training
--full-training-config CONFIG.json --gpu-uuid GPU-... --output NEW_DIRECTORY``
for one full public train() call with the exact JSON settings. This mode requires
MAPPO with 512 environments (four with --cpu-smoke) and at most 3,932,160 real
transitions. It includes declared checkpoints, validation, captures and reports.
There are at most three approved matched GPU cases; a timeout authorizes no retry.
First-block time combines compilation and execution; later blocks and inclusive
output timers come from the normal saved records. Untimed work stays a named
residual, never an assumed host bottleneck. Compilation alone and transfer cost
are not measured here. Existing evidence is preserved; each case needs a new
output. No case establishes learning quality or sample efficiency.
GPU memory polling is a sampled lower bound, not a guaranteed peak.
The learner uses the baseline's default spawn frame, "left" since 22 September
2026. Measurements saved before that date used "world"; a rerun measures
the left program and is not a like-for-like repeat of them.

Use --diagnose-update for one block only when numerical agreement fails. It
writes every differing leaf's path and error, then compares separately compiled
PPO inputs and updates with the complete update calls. This mode produces no
speed qualification and never relaxes the comparison tolerance. It also checks
the unconditional bootstrap and an extra compiler boundary before PPO. These
diagnostic alternatives change no production code or donor equations/settings.
"""

from __future__ import annotations

# Reuse existing measurement and compact summary owners in this developer tool.
# pyright: reportPrivateUsage=false
import argparse
import importlib.metadata
import json
import math
import os
import resource
import statistics
import subprocess
import time
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from marl_battlegrounds.baselines.ppo import (
    PPOBatch,
    PPOConfig,
    PPOTrainState,
    _denormalize_values,
    critic_values,
    update_recurrent_ppo,
)
from marl_battlegrounds.evaluation.recording_identity import (
    ordered_source_bank_identity,
)
from marl_battlegrounds.training._compilation import (
    execution_identity,
    training_compiler_options,
)
from marl_battlegrounds.training._run_io import (
    ProgressReporter,
    TrainingSpeedEstimate,
    append_jsonl,
    atomic_json,
    read_jsonl,
)
from marl_battlegrounds.training.checkpoints import checkpoint_dependencies
from marl_battlegrounds.training.collection import (
    TrainingCarry,
    TrainingRollout,
    scan_training_rollout,
)
from marl_battlegrounds.training.curriculum import make_training_schedule
from marl_battlegrounds.training.learner import (
    LearnerState,
    UpdateResult,
    _summary,
    build_ppo_batch,
    init_learner,
    update_learner,
)
from marl_battlegrounds.training.opponents import refresh_opponents
from marl_battlegrounds.training.runner import (
    TrainConfig,
    config_to_dict,
    read_config,
    train,
)
from scripts.dev.benchmark_training_collection import (
    _array,
    _bytes,
    _MemorySampler,
    _prepare_content,
    _timed,
)
from scripts.dev.benchmark_training_collection import (
    _compile as _compile_measured,
)

type Tree = Any
type UpdateInput = tuple[LearnerState, TrainingCarry, TrainingRollout]


def _compile(
    function: Callable[..., Tree], value: Tree
) -> tuple[Tree, list[int], dict[str, Tree]]:
    """Measure one outer call using the same compiler policy as public training.

    Apply the shared policy to production, reference and diagnostic calls alike.
    The shared measurement helper returns the callable, trace list and costs.
    Neither helper changes JAX's global settings or the supplied numerical inputs.
    """
    return _compile_measured(
        function, value, compiler_options=training_compiler_options()
    )


def _reference_batch(
    state: LearnerState,
    rollout: TrainingRollout,
    *,
    ppo: PPOConfig,
    bootstrap_branch: bool = True,
) -> tuple[PPOBatch, jax.Array]:
    """Build valid full-block PPO inputs independently of the learner adapter.

    state precedes rollout. ppo supplies its input scale and critic setting.
    Both follow the admitted collector contract. Return
    the batch and critic memory after the sequence, discarding bootstrap memory.
    This pure reference assumes a nonempty block and retains initial memories;
    the recurrent networks reset them on episode-start rows themselves.
    bootstrap_branch=True skips the successor call when every final lane ended,
    using a device branch. False always evaluates the masked successor and is
    retained for diagnosing compiler rounding differences. Both variants preserve
    the same admitted-input equations; the flag is static when compiled.
    """
    rows = rollout.transitions
    assert rows.training_state is not None
    assert rollout.final_training_state is not None
    final_features = rollout.final_training_state
    memory, values = critic_values(
        state.critic_params,
        state.critic_memory,
        rows.training_state,
        rows.episode_start,
        rows.valid,
        input_scale=ppo.input_scale,
    )

    def bootstrap(_: None) -> jax.Array:
        """Value continuing successors and discard the bootstrap-only memory."""
        _unused_memory, final = critic_values(
            state.critic_params,
            memory,
            final_features[None],
            jnp.zeros_like(rollout.final_ended)[None],
            (~rollout.final_ended)[None],
            input_scale=ppo.input_scale,
        )
        return jnp.where(
            (~rollout.final_ended)[:, None] & rollout.final_active, final[0], 0.0
        )

    def no_bootstrap(_: None) -> jax.Array:
        """Return zero successor values when every final lane ended."""
        return jnp.zeros_like(values[0])

    final = (
        cast(
            jax.Array,
            jax.lax.cond(jnp.any(~rollout.final_ended), bootstrap, no_bootstrap, None),
        )
        if bootstrap_branch
        else bootstrap(None)
    )
    old_values = jnp.where(rows.valid[..., None], values, 0.0)
    old_normalized_values = None
    if state.value_norm is not None:
        # Preserve exact network outputs for clipping; GAE consumes raw units.
        old_normalized_values = old_values
        old_values = jnp.where(
            rows.valid[..., None],
            _denormalize_values(old_values, state.value_norm),
            0.0,
        )
        final = jnp.where(
            (~rollout.final_ended)[:, None] & rollout.final_active,
            _denormalize_values(final, state.value_norm),
            0.0,
        )
    batch = PPOBatch(
        rows.observations,
        rows.training_state,
        rows.action_mask,
        rows.learning_outputs.action_indices,
        rows.learning_outputs.log_prob,
        old_values,
        rows.task_rewards + jnp.where(rows.active, rows.shaping_reward[..., None], 0.0),
        rows.ended,
        rows.episode_start,
        rows.valid,
        rows.active,
        rows.alive,
        final,
        rollout.initial_memory,
        state.critic_memory,
        old_normalized_values,
    )
    return batch, memory


def _reference_update(
    value: UpdateInput,
    ppo: PPOConfig,
    *,
    bootstrap_branch: bool = True,
    batch_barrier: bool = False,
) -> tuple[LearnerState, UpdateResult]:
    """Arrange existing numerical calls for one valid nonempty full block.

    value contains the accepted state, collected successor and its compact rows;
    ppo supplies static donor settings. This pure reference deliberately assumes
    valid finite input. Production's admission/failure guards are measured as
    integration overhead. It never calls build_ppo_batch or update_learner.
    bootstrap_branch=True selects the independent successor branch above.
    batch_barrier=False permits normal fusion; True puts a compiler boundary
    around the complete batch before PPO. These static diagnostic options change
    no equations, array values, donor settings or host/device placement.
    """
    state, collected, rollout = value
    batch, memory = _reference_batch(
        state, rollout, ppo=ppo, bootstrap_branch=bootstrap_branch
    )
    if batch_barrier:
        batch = jax.lax.optimization_barrier(batch)
    index = state.completed_updates + jnp.int32(1)
    network, metrics = update_recurrent_ppo(
        PPOTrainState(
            state.carry.history.current_variables,
            state.critic_params,
            state.actor_opt_state,
            state.critic_opt_state,
            state.value_norm,
        ),
        batch,
        jax.random.fold_in(state.shuffle_root, index),
        ppo,
    )
    history, snapshot = refresh_opponents(
        collected.history,
        network.actor_params,
        completed_rounds=collected.progress.rounds,
        update_index=index,
        schedule=collected.schedule,
    )
    successor = LearnerState(
        collected._replace(history=history),
        network.critic_params,
        network.actor_opt_state,
        network.critic_opt_state,
        memory,
        state.shuffle_root,
        index,
        jnp.bool_(False),
        jnp.int32(0),
        network.value_norm,
    )
    return successor, UpdateResult(
        jnp.bool_(True),
        metrics,
        successor.failed,
        successor.failure_reason,
        snapshot,
        _summary(rollout),
    )


def _comparison(actual: Tree, expected: Tree) -> dict[str, Tree]:
    """Describe every leaf under the unchanged numerical agreement rules.

    Transfer the two trees outside warmed measurements. Return JSON-safe shapes,
    dtypes, mismatch counts, paths and float errors, including leaves after the
    first failure. Relative error divides by the expected absolute value with
    a float32 tiny floor. Nonfinite errors are null and always fail agreement.
    Integer and key leaves require exact equality; floats use 1e-5 relative and
    2e-6 absolute tolerance. Different tree structures are reported as failures.
    """
    flatten = cast(
        Callable[[Tree], tuple[list[tuple[Tree, Tree]], Tree]],
        jax.tree_util.tree_flatten_with_path,
    )
    actual_leaves, actual_structure = flatten(actual)
    expected_leaves, expected_structure = flatten(expected)
    report: dict[str, Tree] = {
        "passed": actual_structure == expected_structure,
        "same_structure": actual_structure == expected_structure,
        "float_rtol": 1e-5,
        "float_atol": 2e-6,
        "relative_denominator_floor": float(np.finfo(np.float32).tiny),
        "leaves": [],
    }
    if not report["same_structure"]:
        report["actual_paths"] = [
            jax.tree_util.keystr(path) for path, _ in actual_leaves
        ]
        report["expected_paths"] = [
            jax.tree_util.keystr(path) for path, _ in expected_leaves
        ]
        return report
    for (path, left), (_, right) in zip(actual_leaves, expected_leaves, strict=True):
        a, b = _array(left), _array(right)
        leaf: dict[str, Tree] = {
            "path": jax.tree_util.keystr(path),
            "actual_shape": list(a.shape),
            "expected_shape": list(b.shape),
            "actual_dtype": str(a.dtype),
            "expected_dtype": str(b.dtype),
            "passed": False,
        }
        if a.shape == b.shape and a.dtype == b.dtype:
            leaf["exact_mismatches"] = int(np.count_nonzero(a != b))
            if np.issubdtype(a.dtype, np.inexact):
                finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
                leaf["finite"] = finite
                leaf["tolerance_mismatches"] = int(
                    np.count_nonzero(~np.isclose(a, b, rtol=1e-5, atol=2e-6))
                )
                leaf["passed"] = finite and leaf["tolerance_mismatches"] == 0
                if finite:
                    absolute = np.abs(a.astype(np.float64) - b.astype(np.float64))
                    relative = absolute / np.maximum(
                        np.abs(b.astype(np.float64)), np.finfo(np.float32).tiny
                    )
                    leaf["max_absolute_error"] = float(np.max(absolute, initial=0))
                    leaf["max_relative_error"] = float(np.max(relative, initial=0))
                    leaf["expected_max_absolute_value"] = float(
                        np.max(np.abs(b), initial=0)
                    )
                else:
                    leaf["max_absolute_error"] = None
                    leaf["max_relative_error"] = None
            else:
                leaf["passed"] = leaf["exact_mismatches"] == 0
        report["leaves"].append(leaf)
        report["passed"] = report["passed"] and leaf["passed"]
    report["failed_paths"] = [
        leaf["path"] for leaf in report["leaves"] if not leaf["passed"]
    ]
    return report


def _equal(actual: Tree, expected: Tree, *, report_path: Path | None = None) -> None:
    """Check complete trees and optionally save all errors before raising.

    actual and expected obey _comparison's exact/inexact rules. report_path=None
    writes nothing; a supplied path receives the full JSON comparison even on
    failure. All host comparisons and writes happen outside warmed measurements.
    """
    report = _comparison(actual, expected)
    if report_path is not None:
        atomic_json(report_path, report)
    if not report["passed"]:
        raise AssertionError(
            f"Reference and learner differ: {report.get('failed_paths', 'structure')}"
        )


def _diagnose_update(
    directory: Path,
    inputs: UpdateInput,
    actual: tuple[LearnerState, UpdateResult],
    expected: tuple[LearnerState, UpdateResult],
    ppo: PPOConfig,
) -> None:
    """Locate input versus optimizer drift using one already collected block.

    directory receives diagnostic JSON, never accepted cost evidence. inputs,
    actual and expected come from the same initial boundary and full block.
    Compile production and independent batch builders, then one shared PPO
    executable with the fixed ppo settings. Compare both standalone updates with
    each other and with the two whole-update results. No new rollout is collected
    and no tolerance changes. Compilation, transfers and arrays here are solely
    diagnostic work; they are excluded from every performance claim.

    Also compare the earlier unconditional bootstrap reference and the default
    branch with an extra compiler boundary around the PPO batch. Their batch
    values and whole updates are checked separately. These optional comparisons
    do not replace the default reference's required agreement check.
    """
    state, _, rollout = inputs

    def production_batch(value: tuple[LearnerState, TrainingRollout]) -> Tree:
        """Build the adapter's PPO batch at the supplied pre-collection state."""
        return build_ppo_batch(*value, ppo=ppo)

    def reference_batch(value: tuple[LearnerState, TrainingRollout]) -> Tree:
        """Build the independent batch using the same state and compact rows."""
        return _reference_batch(*value, ppo=ppo)

    batch_input = (state, rollout)
    production_fn, _, production_cost = _compile(production_batch, batch_input)
    reference_fn, _, reference_cost = _compile(reference_batch, batch_input)
    production, production_cost["first_seconds"] = _timed(production_fn, batch_input)
    reference, reference_cost["first_seconds"] = _timed(reference_fn, batch_input)
    atomic_json(directory / "batch_comparison.json", _comparison(production, reference))

    def network(value: LearnerState) -> PPOTrainState:
        """Extract the network and optimizer owners without copying device arrays."""
        return PPOTrainState(
            value.carry.history.current_variables,
            value.critic_params,
            value.actor_opt_state,
            value.critic_opt_state,
            value.value_norm,
        )

    key = jax.random.fold_in(state.shuffle_root, state.completed_updates + jnp.int32(1))

    def numerical(value: tuple[PPOTrainState, PPOBatch, jax.Array]) -> Tree:
        """Apply the unchanged PPO owner to dynamic state, batch and shuffle key."""
        return update_recurrent_ppo(*value, config=ppo)

    production_input = (network(state), production[0], key)
    numerical_fn, traces, numerical_cost = _compile(numerical, production_input)
    separate_actual, numerical_cost["production_seconds"] = _timed(
        numerical_fn, production_input
    )
    separate_expected, numerical_cost["reference_seconds"] = _timed(
        numerical_fn, (network(state), reference[0], key)
    )
    comparisons = {
        "standalone_ppo": (separate_actual, separate_expected),
        "production_context": (
            (network(actual[0]), actual[1].metrics),
            separate_actual,
        ),
        "reference_context": (
            (network(expected[0]), expected[1].metrics),
            separate_expected,
        ),
    }
    for name, pair in comparisons.items():
        atomic_json(directory / f"{name}_comparison.json", _comparison(*pair))

    def unconditional_batch(value: tuple[LearnerState, TrainingRollout]) -> Tree:
        """Build the earlier reference without a successor branch for diagnosis."""
        return _reference_batch(*value, ppo=ppo, bootstrap_branch=False)

    def unconditional_update(value: UpdateInput) -> Tree:
        """Run the earlier unconditional reference to isolate compiler rounding."""
        return _reference_update(value, ppo, bootstrap_branch=False)

    def barrier_update(value: UpdateInput) -> Tree:
        """Keep the successor branch and prevent fusion across the PPO batch."""
        return _reference_update(value, ppo, bootstrap_branch=True, batch_barrier=True)

    unconditional_fn, _, unconditional_cost = _compile(unconditional_batch, batch_input)
    unconditional, unconditional_cost["first_seconds"] = _timed(
        unconditional_fn, batch_input
    )
    atomic_json(
        directory / "unconditional_batch_comparison.json",
        _comparison(production, unconditional),
    )
    unconditional_numerical, numerical_cost["unconditional_seconds"] = _timed(
        numerical_fn, (network(state), unconditional[0], key)
    )
    atomic_json(
        directory / "unconditional_standalone_ppo_comparison.json",
        _comparison(separate_actual, unconditional_numerical),
    )
    variants: dict[str, Tree] = {}
    variant_metrics: dict[str, Tree] = {}
    for name, function, standalone in (
        ("unconditional", unconditional_update, unconditional_numerical),
        ("branch_barrier", barrier_update, separate_expected),
    ):
        function_fn, _, cost = _compile(function, inputs)
        result, cost["first_seconds"] = _timed(function_fn, inputs)
        atomic_json(
            directory / f"{name}_full_comparison.json", _comparison(actual, result)
        )
        atomic_json(
            directory / f"{name}_context_comparison.json",
            _comparison((network(result[0]), result[1].metrics), standalone),
        )
        variants[name] = cost
        variant_metrics[name] = {
            field: _array(value).tolist()
            for field, value in result[1].metrics._asdict().items()
        }
    atomic_json(
        directory / "update_diagnostic.json",
        {
            "scope": "One fixed block; diagnostic comparisons only; no speed evidence",
            "production_batch": production_cost,
            "reference_batch": reference_cost,
            "standalone_ppo": numerical_cost,
            "standalone_trace_count": len(traces),
            "unconditional_batch": unconditional_cost,
            "reference_variants": variants,
            "reference_variant_metrics": variant_metrics,
            "metrics": {
                name: {
                    field: _array(value).tolist()
                    for field, value in metrics._asdict().items()
                }
                for name, metrics in (
                    ("whole_production", actual[1].metrics),
                    ("whole_reference", expected[1].metrics),
                    ("standalone_production_batch", separate_actual[1]),
                    ("standalone_reference_batch", separate_expected[1]),
                )
            },
        },
    )


def _cost(samples: list[float], transitions: int) -> dict[str, Tree]:
    """Summarize synchronized seconds and real transition throughput.

    samples must contain at least five positive timings. transitions counts
    real game transitions in each sample, without multiplying PPO epochs.
    """
    if len(samples) < 5 or any(
        not np.isfinite(value) or value <= 0 for value in samples
    ):
        raise ValueError("At least five positive synchronized samples are required")
    median = statistics.median(samples)
    return {
        "warm_seconds": samples,
        "median_seconds": median,
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "spread_seconds": max(samples) - min(samples),
        "real_transitions_per_second": transitions / median,
    }


def _compiler_comparison(
    directory: Path,
    collect: Callable[..., Tree],
    update: Callable[..., Tree],
    compiled_collect: Callable[..., Tree],
    compiled_update: Callable[..., Tree],
    inputs: UpdateInput,
    production_output: Tree,
) -> dict[str, Tree]:
    """Compare shared-policy and default compiler costs on fixed identical inputs.

    Compile the same pure functions once with default options, then run three
    synchronized pairs in alternating order. Collection receives the original
    carry; both updater versions receive the exact same already collected PPO
    inputs. Save full output comparisons without requiring two different compiler
    choices to produce equal floats or actions. This cost probe changes no state
    in the benchmark's accepted continuation. All copies precede warm timings.
    """
    default_collect, collect_traces, collection_cost = _compile_measured(
        collect, inputs[0].carry
    )
    default_update, update_traces, update_cost = _compile_measured(update, inputs)
    default_rows, collection_cost["first_seconds"] = _timed(
        default_collect, inputs[0].carry
    )
    default_result, update_cost["first_seconds"] = _timed(default_update, inputs)
    comparisons = {
        "collection": _comparison(inputs[1:], default_rows),
        "update": _comparison(production_output, default_result),
    }
    atomic_json(directory / "compiler_output_comparisons.json", comparisons)
    samples: dict[str, dict[str, list[float]]] = {
        name: {"collection": [], "update": []} for name in ("policy", "default")
    }
    orders = []
    with jax.transfer_guard("disallow_explicit"):
        for index in range(3):
            order = ["policy", "default"] if index % 2 == 0 else ["default", "policy"]
            orders.append(order)
            for name in order:
                collector = compiled_collect if name == "policy" else default_collect
                updater = compiled_update if name == "policy" else default_update
                _, seconds = _timed(collector, inputs[0].carry)
                samples[name]["collection"].append(seconds)
                _, seconds = _timed(updater, inputs)
                samples[name]["update"].append(seconds)
    for values in samples.values():
        values["composed"] = [
            a + b for a, b in zip(values["collection"], values["update"], strict=True)
        ]
    traces = [len(collect_traces), len(update_traces)]
    if traces != [1, 1]:
        raise AssertionError(f"Fixed-input compiler comparison retraced: {traces}")
    return {
        "policy_options": training_compiler_options(),
        "comparison_options": None,
        "execution_order": orders,
        "warm_seconds": samples,
        "median_seconds": {
            name: {stage: statistics.median(times) for stage, times in values.items()}
            for name, values in samples.items()
        },
        "default_collection": collection_cost,
        "default_update": update_cost,
        "default_trace_counts": traces,
        "output_comparisons": {
            name: {"passed": result["passed"], "failed_paths": result["failed_paths"]}
            for name, result in comparisons.items()
        },
        "scope": (
            "Three reversed-order synchronized pairs on identical fixed inputs; "
            "descriptive compiler costs, not scientific trajectory equivalence"
        ),
    }


def _status(
    result: UpdateResult,
    *,
    index: int,
    transitions: int,
    elapsed: float,
    estimated_rate: float | None = None,
    updates: int = 5,
) -> dict[str, Tree]:
    """Turn already transferred compact update results into ordinary host fields.

    index is the completed update count in this reporting pass. transitions is
    the fixed number of game transitions per update; elapsed is pass wall time.
    estimated_rate is the existing warmed host estimate, or None until ready.
    updates is the declared number of real blocks in this pass, default five.
    No rollout or learner state is read here.
    """
    if not bool(result.performed) or bool(result.failed):
        raise AssertionError("Benchmark update was not accepted")
    summary = result.summary
    return {
        "phase": "training",
        "env_steps": index * transitions,
        "total_env_steps": updates * transitions,
        "completed_updates": index,
        "elapsed_seconds": elapsed,
        "transitions_per_second": index * transitions / max(elapsed, 1e-9),
        "estimated_transitions_per_second": estimated_rate,
        "policy_loss": float(np.mean(result.metrics.actor_loss)),
        "value_loss": float(np.mean(result.metrics.value_loss)),
        "entropy": float(np.mean(result.metrics.entropy)),
        "task_reward_mean": float(summary.task_reward_sum)
        / max(int(summary.active_samples), 1),
        "shaping_mean": float(summary.shaping_reward_sum)
        / max(int(summary.real_transitions), 1),
        "live_actor_decisions": int(summary.live_actor_decisions),
    }


def _reporting_pass(
    directory: Path,
    initial: LearnerState,
    collect: Callable[..., Tree],
    update: Callable[..., Tree],
    *,
    verbose: bool,
    transitions: int,
    updates: int = 5,
) -> tuple[LearnerState, dict[str, Tree]]:
    """Measure the declared real updates with mandatory files and optional stdout.

    directory must not exist. Both compiled callables accept dynamic arrays;
    collect takes a TrainingCarry and update takes UpdateInput. Retain initial
    and return the final state plus separate compute, transfer and output costs.
    A real file captures stdout. ProgressReporter uses its normal ten-second
    throttle, plus real starting/training/finished phase changes; no fake clock
    or repeated forced progress lines are used. Disk failures propagate.
    updates defaults to five for CPU smoke; GPU callers use twelve to exercise
    a periodic line with the warmed ETA. The result counts actual periodic lines.
    """
    directory.mkdir()
    state = initial
    compute: list[float] = []
    transfers: list[float] = []
    writes: list[float] = []
    complete: list[float] = []
    compact_bytes = 0
    status: dict[str, Tree] = {}
    started = time.perf_counter()
    with (directory / "stdout.log").open("x", encoding="utf-8") as stream:
        reporter = ProgressReporter(enabled=verbose, stream=stream)
        speed_estimate = TrainingSpeedEstimate() if verbose else None
        reporter.report({"phase": "starting", "total_env_steps": updates * transitions})
        for index in range(1, updates + 1):
            step_started = time.perf_counter()
            (carry, rollout), collection_time = _timed(collect, state.carry)
            (state, result), update_time = _timed(update, (state, carry, rollout))
            host, transfer_time = _timed(jax.device_get, result)
            if speed_estimate is not None:
                speed_estimate.observe(
                    transitions, collection_time + update_time + transfer_time
                )
            compact_bytes += _bytes(result)
            status = _status(
                host,
                index=index,
                transitions=transitions,
                elapsed=time.perf_counter() - started,
                estimated_rate=None if speed_estimate is None else speed_estimate.rate,
                updates=updates,
            )
            output_started = time.perf_counter()
            append_jsonl(directory / "training.jsonl", status)
            atomic_json(directory / "status.json", status)
            reporter.report(status)
            writes.append(time.perf_counter() - output_started)
            transfers.append(transfer_time)
            compute.append(collection_time + update_time)
            complete.append(time.perf_counter() - step_started)
        status = {**status, "phase": "finished"}
        atomic_json(directory / "status.json", status)
        reporter.report(status, force=True)
    lines = (directory / "stdout.log").read_text().splitlines()
    training_lines = [line for line in lines if line.startswith("Training:")]
    return state, {
        "verbose": verbose,
        "elapsed_seconds": time.perf_counter() - started,
        "update_seconds": complete,
        "compute_seconds": compute,
        "compact_transfer_seconds": transfers,
        "reporting_seconds": writes,
        "compact_transferred_bytes": compact_bytes,
        "files": {path.name: path.stat().st_size for path in directory.iterdir()},
        "stdout_lines": len(lines),
        "periodic_progress_lines": max(0, len(training_lines) - 1),
        "periodic_warmed_eta_lines": sum(
            "Training ETA Estimating" not in line for line in training_lines[1:]
        ),
        "updates": updates,
        "cadence_seconds": 10,
        "cadence": "Actual ten-second throttle plus real phase changes",
    }


def _reporting_pairs(
    directory: Path,
    initial: LearnerState,
    collect: Callable[..., Tree],
    update: Callable[..., Tree],
    *,
    transitions: int,
    updates: int = 5,
) -> dict[str, Tree]:
    """Run quiet/verbose then verbose/quiet from the same numerical boundary.

    Arguments follow _reporting_pass, except directory already exists and holds
    four new child directories. Compare every final state and retain all raw
    timings. Per-update differences match the same update index across modes;
    positive differences mean verbose took longer. These two short pairs expose
    order effects, but do not provide a statistical no-slowdown guarantee.
    """
    measured: dict[str, Tree] = {}
    reference = None
    for name, enabled in (
        ("quiet", False),
        ("verbose", True),
        ("verbose_again", True),
        ("quiet_again", False),
    ):
        state, cost = _reporting_pass(
            directory / name,
            initial,
            collect,
            update,
            verbose=enabled,
            transitions=transitions,
            updates=updates,
        )
        if reference is None:
            reference = state
        else:
            _equal(reference, state)
        measured[name] = cost
    pairs = []
    for quiet, verbose, order in (
        ("quiet", "verbose", ["quiet", "verbose"]),
        ("quiet_again", "verbose_again", ["verbose_again", "quiet_again"]),
    ):
        q, v = measured[quiet], measured[verbose]
        pairs.append(
            {
                "execution_order": order,
                "verbose_minus_quiet_update_seconds": [
                    slow - fast
                    for slow, fast in zip(
                        v["update_seconds"], q["update_seconds"], strict=True
                    )
                ],
                "verbose_minus_quiet_elapsed_seconds": v["elapsed_seconds"]
                - q["elapsed_seconds"],
            }
        )
    return {
        **measured,
        "pairs": pairs,
        "identical_final_states": True,
        "periodic_reporting_exercised": all(
            measured[name]["periodic_progress_lines"] > 0
            and measured[name]["periodic_warmed_eta_lines"] > 0
            for name in ("verbose", "verbose_again")
        ),
        "scope": "Two reversed-order pairs with real files; descriptive timings only",
    }


def _gpu_identity(uuid: str | None, *, cpu_smoke: bool) -> tuple[str, str | None]:
    """Check backend and explicit device selection before benchmark setup.

    uuid is the caller's physically verified internal GPU UUID; this check
    verifies the exposed device's name/UUID, not where the card is installed.
    cpu_smoke requires a single CPU device. GPU mode requires one RTX 5090 and
    CUDA_VISIBLE_DEVICES equal to uuid. Raise ValueError for any mismatch.
    """
    if cpu_smoke:
        if os.environ.get("JAX_PLATFORMS") != "cpu" or uuid is not None:
            raise ValueError("CPU smoke needs JAX_PLATFORMS=cpu and no GPU UUID")
        identity = None
    else:
        platforms = os.environ.get("JAX_PLATFORMS")
        if (
            not uuid
            or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid
            or (
                platforms is not None
                and platforms not in ("cuda", "gpu", "cuda,cpu", "gpu,cpu")
            )
        ):
            raise ValueError(
                "Select the verified internal GPU UUID in CUDA_VISIBLE_DEVICES "
                "and --gpu-uuid with a GPU JAX_PLATFORMS setting"
            )
        identity = subprocess.run(
            [
                "nvidia-smi",
                f"--id={uuid}",
                "--query-gpu=name,uuid,pci.bus_id,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        columns = [column.strip() for column in identity.split(",")]
        if len(columns) != 5 or "RTX 5090" not in columns[0] or columns[1] != uuid:
            raise ValueError("Selected hardware does not match the declared RTX 5090")
    # Backend discovery may initialize accelerators. Do it only after the host
    # checks have proved which device the process is allowed to expose.
    backend = jax.default_backend()
    if backend != ("cpu" if cpu_smoke else "gpu"):
        raise ValueError("JAX did not select the declared CPU/GPU backend")
    if len(cast(list[Tree], jax.devices())) != 1:
        raise ValueError("Expose exactly one device to this benchmark")
    return backend, identity


def _full_training_config(path: Path, *, cpu_smoke: bool) -> TrainConfig:
    """Read one unchanged MAPPO config inside the declared P9 measurement bounds.

    GPU cases use 512 environments; CPU tool checks use four. Both retain the
    exact JSON budget and settings, at most 3,932,160 real transitions. This
    reader does not adjust capture spacing, outputs or learner settings. Public
    train performs its ordinary setup validation before creating run files.
    The caller enforces the 600-second wall limit; a timeout permits no retry.
    """
    config = read_config(path)
    expected = 4 if cpu_smoke else 512
    if config.method != "mappo" or config.num_envs != expected:
        raise ValueError(
            f"Full training measurement requires MAPPO with {expected} environments"
        )
    if config.total_env_steps > 3_932_160:
        raise ValueError(
            "Full training measurement allows at most 3,932,160 transitions"
        )
    return config


def _full_training_cost(
    run_dir: Path,
    config: TrainConfig,
    *,
    seconds: float,
    complete: bool,
) -> dict[str, Tree]:
    """Read real block counts and nonoverlapping inclusive phase timers from files.

    seconds is the complete public train call, including setup and final outputs.
    Update rows already synchronize their result. First-block time includes
    compilation and execution; later rows are reported separately without a
    compile-only claim. Saved output-event timers include that operation's host,
    device, transfer and file work. Residual time includes logging, status,
    bookkeeping and any unmeasured work; it is not a Python-overhead estimate.
    Invalid counts, negative/nonfinite times or missing completed work raise.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Full training wall time must be finite and positive")
    rows = read_jsonl(run_dir / "training_updates.jsonl")
    blocks: list[dict[str, Tree]] = []
    previous = 0
    for row in rows:
        steps = row.get("env_steps")
        duration = row.get("collection_update_seconds")
        if (
            type(steps) is not int
            or steps <= previous
            or steps > config.total_env_steps
            or (steps - previous) % config.num_envs
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError(
                "Saved training blocks need increasing real counts "
                "and positive finite times"
            )
        blocks.append(
            {
                "env_steps": steps,
                "real_transitions": steps - previous,
                "seconds": float(duration),
            }
        )
        previous = steps
    if complete and previous != config.total_env_steps:
        raise ValueError(
            "Completed training count differs from the declared experience budget"
        )
    events = read_jsonl(run_dir / "run_events.jsonl")
    event_names = {
        "checkpoint": ("checkpoint_saved",),
        "actor_export": ("actor_exported",),
        "past_capture_export": ("past_copy_exported",),
        "validation": ("validation_complete", "random_validation_complete"),
        "report": ("reports_written",),
    }
    phases: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, names in event_names.items():
        matching = [row for row in events if row.get("event") in names]
        durations = [row.get("seconds") for row in matching]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in durations
        ):
            raise ValueError(f"Saved {name} events need finite nonnegative seconds")
        phases[name] = sum(float(cast(int | float, value)) for value in durations)
        counts[name] = len(matching)
    details_path = run_dir / "run_details.json"
    details: object = (
        json.loads(details_path.read_text()) if details_path.exists() else {}
    )
    if not isinstance(details, dict):
        raise ValueError("Saved run details must be an object")
    setup = cast(dict[str, Tree], details).get("initial_setup_seconds", 0.0)
    if (
        isinstance(setup, bool)
        or not isinstance(setup, (int, float))
        or not math.isfinite(setup)
        or setup < 0
    ):
        raise ValueError("Saved setup time must be finite and nonnegative")
    training = sum(row["seconds"] for row in blocks)
    accounted = setup + training + sum(phases.values())
    if accounted > seconds + max(0.01, seconds * 1e-6):
        raise ValueError("Saved phase times overlap or exceed the full train wall time")
    warm = blocks[1:]
    warm_seconds = sum(row["seconds"] for row in warm)
    warm_steps = sum(row["real_transitions"] for row in warm)
    disk: dict[str, int] = {}
    for path in run_dir.rglob("*"):
        if path.is_file():
            group = path.relative_to(run_dir).parts[0]
            disk[group] = disk.get(group, 0) + path.stat().st_size
    return {
        "complete": complete,
        "scope": (
            "One unchanged public train() call, including setup, "
            "all declared outputs and cleanup"
        ),
        "declared_env_steps": config.total_env_steps,
        "real_transitions": previous,
        "total_seconds": seconds,
        "end_to_end_transitions_per_second": previous / seconds,
        "initial_setup_seconds": setup,
        "first_block_compile_and_execute": blocks[0] if blocks else None,
        "later_blocks": {
            "count": len(warm),
            "seconds": warm_seconds,
            "real_transitions": warm_steps,
            "transitions_per_second": warm_steps / warm_seconds
            if warm_seconds
            else None,
            "samples": warm,
        },
        "collection_update_seconds": training,
        "inclusive_output_seconds": phases,
        "output_event_counts": counts,
        "residual_seconds": max(0.0, seconds - accounted),
        "residual_scope": (
            "Logging, status, bookkeeping, untimed cleanup "
            "and any other unmeasured work"
        ),
        "disk_bytes_by_entry": disk,
        "disk_bytes": sum(disk.values()),
        "unmeasured": [
            "Compilation alone",
            "Separate host/device transfer cost",
            "Learned behavior or sample efficiency",
        ],
    }


def _full_training(
    output: Path,
    config: TrainConfig,
    *,
    gpu_uuid: str | None,
    tool_started: float,
) -> None:
    """Measure one complete ordinary train call and retain its raw normal outputs.

    output is the new benchmark directory already holding source identity. The
    training child must not exist. No settings change or retry occurs. The caller
    enforces the declared 600-second timeout; forced termination can leave only
    partial source/workload/training files. Normal failures get a partial cost
    record and propagate. The extra tool clock ends after the sampler stops and
    excludes writing this cost report. GPU sampling is a lower bound, not a
    guaranteed peak.
    """
    run_dir = output / "training"
    if run_dir.exists():
        raise ValueError("Existing full-training evidence must be preserved")
    atomic_json(
        output / "workload.json",
        {
            "mode": "Full public training",
            "config": config_to_dict(config),
            "maximum_real_transitions": 3_932_160,
            "caller_wall_limit_seconds": 600,
            "case_limit": (
                "At most three cases in the approved investigation; no automatic retry"
            ),
            "timing": (
                "First block includes compilation and execution; "
                "later blocks remain separate"
            ),
            "timeout": (
                "Enforce with the caller's timeout 600s; "
                "termination keeps existing files"
            ),
        },
    )
    complete = False
    error: str | None = None
    sampler = _MemorySampler(gpu_uuid)
    seconds = 0.0
    try:
        with sampler:
            started = time.perf_counter()
            try:
                result = train(config, output_dir=run_dir)
                if result.completed_env_steps != config.total_env_steps:
                    raise ValueError(
                        "Public train result differs from its declared transition count"
                    )
                complete = True
            finally:
                seconds = time.perf_counter() - started
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        tool_seconds = time.perf_counter() - tool_started
        cost: dict[str, Tree]
        try:
            cost = _full_training_cost(
                run_dir, config, seconds=seconds, complete=complete
            )
        except (ValueError, OSError) as measurement_error:
            if complete:
                raise
            # Keep the original training failure if its partial records are invalid.
            cost = {
                "complete": False,
                "total_seconds": seconds,
                "measurement_error": str(measurement_error),
            }
        cost.update(
            error=error,
            tool_seconds_through_sampling=tool_seconds,
            tool_setup_and_sampler_cleanup_seconds=max(0.0, tool_seconds - seconds),
            process_peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            gpu_process_memory=sampler.report(),
            device_memory_stats=cast(Tree, jax.devices()[0].memory_stats()),
            memory_scope=(
                "Whole process; GPU polling covers public train, "
                "including setup and outputs"
            ),
        )
        atomic_json(output / "full_training_cost.json", cost)
    print(
        json.dumps(
            {
                "output": str(output),
                "real_transitions": config.total_env_steps,
                "seconds": seconds,
            }
        )
    )


def main() -> None:
    """Parse the fixed workload, run independent checks, and save cost evidence.

    --full-training-config reads an unchanged bounded public MAPPO run instead
    of the lower-level comparison. Its JSON owns the seed; --seed applies only
    to the default route. Accept --output (new directory), --seed (default
    19041900), --cpu-smoke and
    --gpu-uuid. --diagnose-update stops after one block and writes stage comparisons.
    Invalid device selection or existing output raises a CLI error.
    Numerical disagreement aborts without a success cost report. No backend,
    physical GPU identity or learning competence is inferred from timing.
    """
    tool_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=19_041_900)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--diagnose-update", action="store_true")
    parser.add_argument("--full-training-config", type=Path)
    options = parser.parse_args()
    full_config = None
    if options.full_training_config is not None:
        if options.diagnose_update:
            parser.error(
                "--full-training-config cannot be combined with --diagnose-update"
            )
        try:
            full_config = _full_training_config(
                options.full_training_config, cpu_smoke=options.cpu_smoke
            )
        except (ValueError, TypeError) as error:
            parser.error(str(error))
    if options.output.exists():
        parser.error("Existing evidence directories must be preserved")
    jax.config.update("jax_enable_compilation_cache", False)
    try:
        backend, gpu = _gpu_identity(options.gpu_uuid, cpu_smoke=options.cpu_smoke)
    except ValueError as error:
        parser.error(str(error))
    options.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    paths = [
        *sorted((root / "src/marl_battlegrounds").rglob("*.py")),
        Path(__file__).resolve(),
        root / "scripts/dev/benchmark_training_collection.py",
        root / "uv.lock",
    ]
    atomic_json(
        options.output / "source_identity.json",
        {
            "source_sha256": {
                str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "git_status": subprocess.run(
                ["git", "status", "--short"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout,
            "gpu": gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "allocator": {
                name: os.environ.get(name)
                for name in (
                    "XLA_PYTHON_CLIENT_PREALLOCATE",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION",
                    "XLA_PYTHON_CLIENT_ALLOCATOR",
                )
            },
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "numpy", "flax", "marl-battlegrounds")
            },
            "numerical_dependencies": checkpoint_dependencies(),
            "execution": execution_identity(),
        },
    )
    if full_config is not None:
        _full_training(
            options.output,
            full_config,
            gpu_uuid=options.gpu_uuid,
            tool_started=tool_started,
        )
        return
    batch, length = (4, 2) if options.cpu_smoke else (32, 128)
    ppo = PPOConfig(rollout_length=length, epochs=1 if options.cpu_smoke else 4)
    prepared, prepare_seconds = _timed(_prepare_content)
    reporting_updates = 5 if options.cpu_smoke else 12
    schedule = make_training_schedule(
        total_env_steps=batch * length * max(8, reporting_updates), num_envs=batch
    )
    started = time.perf_counter()
    collection, initial = init_learner(
        schedule=schedule, seed=options.seed, ppo=ppo, prepared=prepared
    )
    jax.block_until_ready(initial)
    initialization_seconds = time.perf_counter() - started
    initial, placement_seconds = _timed(
        cast(Callable[..., Tree], jax.device_put), initial
    )
    atomic_json(
        options.output / "workload.json",
        {
            "batch": batch,
            "steps": length,
            "epochs": ppo.epochs,
            "seed": options.seed,
            "diagnostic_only": options.diagnose_update,
            "reporting_updates_per_pass": reporting_updates,
            "verified_content": prepared.binding.model_dump(mode="json"),
            "actual_source_bank_digest": ordered_source_bank_identity(
                initial.carry.tracking.source_configs
            )[0],
        },
    )

    def collect(carry: TrainingCarry) -> tuple[TrainingCarry, TrainingRollout]:
        """Collect the declared fixed capacity from the supplied dynamic boundary."""
        return scan_training_rollout(collection, carry, length=length)

    def update(value: UpdateInput) -> tuple[LearnerState, UpdateResult]:
        """Apply production admission, PPO, history and compact summaries."""
        return update_learner(*value, ppo=ppo)

    def reference(value: UpdateInput) -> tuple[LearnerState, UpdateResult]:
        """Use the independent valid-input arrangement with the same settings."""
        return _reference_update(value, ppo)

    collect_fn, collect_traces, collection_cost = _compile(collect, initial.carry)
    (carry, rollout), collection_cost["first_seconds"] = _timed(
        collect_fn, initial.carry
    )
    inputs = (initial, carry, rollout)
    update_fn, update_traces, update_cost = _compile(update, inputs)
    reference_fn, reference_traces, reference_cost = _compile(reference, inputs)
    with _MemorySampler(options.gpu_uuid) as sampler:
        actual, update_cost["first_seconds"] = _timed(update_fn, inputs)
        expected, reference_cost["first_seconds"] = _timed(reference_fn, inputs)
        initial_comparison = _comparison(actual, expected)
        atomic_json(options.output / "initial_comparison.json", initial_comparison)
        if options.diagnose_update:
            _diagnose_update(options.output, inputs, actual, expected, ppo)
        if not initial_comparison["passed"]:
            raise AssertionError(
                "Initial full update differs: "
                f"{initial_comparison.get('failed_paths', 'structure')}; "
                f"see {options.output / 'initial_comparison.json'}"
            )
        if options.diagnose_update:
            print(json.dumps({"diagnostic": str(options.output)}))
            return
        if not bool(actual[1].performed) or bool(actual[1].failed):
            raise AssertionError("Initial full block update was not accepted")
        if int(rollout.real_steps) != length:
            raise AssertionError(
                "Workload did not contain the declared real transitions"
            )
        samples: dict[str, list[float]] = {
            "collection": [],
            "adapter": [],
            "reference": [],
        }
        with jax.transfer_guard("disallow_explicit"):
            for repeat in range(5):
                _, seconds = _timed(collect_fn, initial.carry)
                samples["collection"].append(seconds)
                order = (
                    ("adapter", "reference")
                    if repeat % 2 == 0
                    else ("reference", "adapter")
                )
                for name in order:
                    _, seconds = _timed(
                        update_fn if name == "adapter" else reference_fn, inputs
                    )
                    samples[name].append(seconds)
        # New continuation memory and changed parameters stay dynamic.
        next_carry, next_rows = collect_fn(actual[0].carry)
        next_inputs = (actual[0], next_carry, next_rows)
        _equal(
            update_fn(next_inputs),
            reference_fn(next_inputs),
            report_path=options.output / "continued_comparison.json",
        )
        _, changed = init_learner(
            schedule=schedule, seed=options.seed + 1, ppo=ppo, prepared=prepared
        )
        changed_carry, changed_rows = collect_fn(changed.carry)
        changed_inputs = (changed, changed_carry, changed_rows)
        _equal(
            update_fn(changed_inputs),
            reference_fn(changed_inputs),
            report_path=options.output / "changed_seed_comparison.json",
        )
        reporting = _reporting_pairs(
            options.output,
            initial,
            collect_fn,
            update_fn,
            transitions=batch * length,
            updates=reporting_updates,
        )
        compiler_comparison = _compiler_comparison(
            options.output, collect, update, collect_fn, update_fn, inputs, actual
        )
    counts = [len(collect_traces), len(update_traces), len(reference_traces)]
    if counts != [1, 1, 1]:
        raise AssertionError(f"Ordinary changed values retraced a callable: {counts}")
    for name, report in (
        ("collection", collection_cost),
        ("adapter", update_cost),
        ("reference", reference_cost),
    ):
        report.update(_cost(samples[name], batch * length))
    # Fresh executions avoid NumPy equality checks populating a transfer cache.
    fresh, _ = _timed(update_fn, inputs)
    _, compact_transfer_seconds = _timed(jax.device_get, fresh[1])
    device = cast(Tree, jax.devices()[0])
    atomic_json(
        options.output / "costs.json",
        {
            "cpu_smoke_only": options.cpu_smoke,
            "backend": backend,
            "batch": batch,
            "steps": length,
            "seed": options.seed,
            "epochs": ppo.epochs,
            "metrics": "priority",
            "recording": False,
            "verified_content": prepared.binding.model_dump(mode="json"),
            "actual_source_bank_digest": ordered_source_bank_identity(
                initial.carry.tracking.source_configs
            )[0],
            "prepare_content_seconds": prepare_seconds,
            "initialization_seconds": initialization_seconds,
            "initial_placement_seconds": placement_seconds,
            "whole_tool_seconds_before_final_report": time.perf_counter()
            - tool_started,
            "collection": collection_cost,
            "adapter_update": update_cost,
            "reference_update": reference_cost,
            "composed_adapter": _cost(
                [
                    a + b
                    for a, b in zip(
                        samples["collection"], samples["adapter"], strict=True
                    )
                ],
                batch * length,
            ),
            "composed_reference": _cost(
                [
                    a + b
                    for a, b in zip(
                        samples["collection"], samples["reference"], strict=True
                    )
                ],
                batch * length,
            ),
            "full_state_output_agreement": {
                "integer_and_key": "Exact",
                "float_rtol": 1e-5,
                "float_atol": 2e-6,
            },
            "trace_counts": counts,
            "changed_values": (
                "New weights, keys, maps, observations and continued recurrent memory"
            ),
            "transfer": {
                "compact_result_bytes": _bytes(fresh[1]),
                "compact_result_seconds": compact_transfer_seconds,
                "hot_numerics": "Explicit transfers disallowed during all warm samples",
            },
            "logical_bytes": {
                "learner": _bytes(initial),
                "rollout": _bytes(rollout),
                "update_output": _bytes(actual),
                "physical_training_features": _bytes(
                    rollout.transitions.training_state
                ),
            },
            "process_peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "gpu_process_memory": sampler.report(),
            "device_memory_stats": device.memory_stats(),
            "memory_scope": (
                "Whole process includes production/reference and default-policy "
                "comparison executables and states. GPU polling starts after "
                "production/reference compilation and includes later default "
                "compilation; its 0.2-second samples may miss short peaks."
            ),
            "retained_rollout_fields": list(rollout.transitions._fields),
            "reporting": reporting,
            "compiler_comparison": compiler_comparison,
            "generated_collection_transitions": (8 + 4 * reporting_updates + 7)
            * batch
            * length,
            "generated_collection_scope": (
                "Eight primary calls, four reporting passes and seven compiler "
                "comparison calls; repeated fixed-boundary samples are counted "
                "as generated work, not additional unique learning experience"
            ),
            "reporting_qualification": (
                "CPU smoke only"
                if options.cpu_smoke
                else "Periodic warmed ETA exercised in both verbose passes"
                if reporting["periodic_reporting_exercised"]
                else "Gap: normal periodic warmed ETA was not exercised"
            ),
            "unmeasured": [
                "Checkpoint save/restore/hashing",
                "Loaded validation",
                "Plotting",
                "Useful learning",
                "Guaranteed instantaneous memory peak",
            ],
            "scope": (
                "Valid full-block numerical integration and reporting; compact "
                "host transfers only in measured reporting; logical bytes may "
                "include aliases; composed timing sums separately synchronized stages"
            ),
        },
    )
    print(
        json.dumps(
            {
                "costs": str(options.output / "costs.json"),
                "cpu_smoke_only": options.cpu_smoke,
            }
        )
    )


if __name__ == "__main__":
    main()
