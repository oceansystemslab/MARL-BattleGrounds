"""Measure the sampled public loop against identical preselected inputs on B32.

Run ``python -m scripts.dev.benchmark_training_distributions --case no-reset
--output DIRECTORY`` with the internal GPU UUID selected. Repeat for
partial-reset and all-reset. The latter two use explicitly synthetic short
horizons; they are reset stress, not approved training distributions. Each case
uses T128, changing observations, Random Systems and priority metrics. Setup,
compilation, five synchronized warm samples and optional host writing are
reported separately. No optimizer, learner checkpoint or learning run exists.

--cpu-smoke uses B2/T4 only to check this tool. Its timings are not GPU evidence.
The reference tape is made outside timing and its extra storage is reported.
"""

from __future__ import annotations

# Benchmark instrumentation inspects existing private writer drains only.
# pyright: reportPrivateUsage=false
import argparse
import cProfile
import importlib.metadata
import json
import os
import pstats
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
from examples import training_distributions as workflow

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.evaluation.collection_types import CollectedBatch
from marl_battlegrounds.evaluation.recording_identity import (
    ordered_source_bank_identity,
)
from marl_battlegrounds.training import prepare_training_content

type Tree = Any


def _array(value: Tree) -> np.ndarray:
    """Transfer one numerical leaf, exposing typed keys as uint32 words."""
    if jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
        value = jax.random.key_data(value)
    return np.asarray(value)


def _bytes(value: Tree) -> int:
    """Count numerical leaf bytes without transferring device values."""
    return sum(int(leaf.size * leaf.dtype.itemsize) for leaf in jax.tree.leaves(value))


def _equal(actual: Tree, expected: Tree) -> None:
    """Require exact integer and float output equality outside measurement."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(_array(left), _array(right))


def _timed(function: Callable[..., Tree], *args: Tree) -> tuple[Tree, float]:
    """Synchronize a complete call and return its result and elapsed seconds."""
    started = time.perf_counter()
    result = cast(Tree, jax.block_until_ready(function(*args)))
    return result, time.perf_counter() - started


def _compile(
    function: Callable[..., Tree], *args: Tree
) -> tuple[Tree, list[int], dict[str, Tree]]:
    """Compile dynamic arguments and report time, temporary memory and callbacks."""
    traces: list[int] = []

    def counted(*values: Tree) -> Tree:
        """Count traces without placing a callback in compiled execution."""
        traces.append(1)
        return function(*values)

    compiled = jax.jit(counted)
    started = time.perf_counter()
    lowered = compiled.lower(*args)
    executable = lowered.compile()
    elapsed = time.perf_counter() - started
    program = lowered.as_text()
    for marker in ("python_callback", "host_callback", "io_callback", "xla_ffi_python"):
        if marker in program:
            raise AssertionError(f"Unexpected callback: {marker}")
    memory = executable.memory_analysis()
    report = {
        "compile_seconds": elapsed,
        "lowered_sha256": sha256(program.encode()).hexdigest(),
        "host_callbacks": False,
        "memory_bytes": None
        if memory is None
        else {
            name: int(getattr(memory, name))
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
        },
    }
    return compiled, traces, report


class _TimedWriter(marl_bgs.RunWriter):
    """Measure writer drains separately from numerical collection and compilation.

    Inherited arguments and durability rules remain unchanged. drain_seconds
    includes source verification, serialization and publication. Host buffers
    have already been collected; verification can still dispatch small checks.
    """

    drain_seconds: float = 0.0
    drain_bytes: int = 0
    drain_count: int = 0

    def _write_collected(
        self, batch: object, *, source_configs: EnvConfig | None = None
    ) -> None:
        """Time the normal owned writer drain and count its incoming host bytes."""
        started = time.perf_counter()
        super()._write_collected(batch, source_configs=source_configs)
        self.drain_seconds += time.perf_counter() - started
        self.drain_bytes += _bytes(cast(CollectedBatch, batch))
        self.drain_count += 1


def main() -> None:
    """Save one case's evidence, rejecting stale destinations or wrong devices.

    Requires --case and --output. --cpu-smoke reduces only the correctness
    workload. --skip-writing omits the separately reported bounded writer run.
    Refuse existing measurement.json rather than replacing evidence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("no-reset", "partial-reset", "all-reset"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--skip-writing", action="store_true")
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    target = options.output / "measurement.json"
    if target.exists():
        parser.error("Existing measurement evidence must be preserved")
    jax.config.update("jax_enable_compilation_cache", False)
    backend = jax.default_backend()
    if backend != "gpu" and not options.cpu_smoke:
        parser.error("Select the internal CUDA GPU or explicitly request --cpu-smoke")
    devices = cast(list[Tree], jax.devices())
    if len(devices) != 1:
        parser.error("Select exactly one GPU with CUDA_VISIBLE_DEVICES")
    device = devices[0]
    batch, steps = (2, 4) if options.cpu_smoke else (32, 128)
    profiler = cProfile.Profile()
    started = time.perf_counter()
    prepared = profiler.runcall(prepare_training_content)
    jax.block_until_ready(prepared.source_configs)
    preparation_seconds = time.perf_counter() - started
    profiler.dump_stats(str(options.output / "preparation.prof"))
    profile = pstats.Stats(profiler)
    hash_self_seconds = sum(
        entry[2]
        for key, entry in cast(
            dict[tuple[str, int, str], tuple[int, int, float, float, Tree]],
            cast(Tree, profile).stats,
        ).items()
        if "hashlib" in key[2] or "sha256" in key[2] or "hexdigest" in key[2]
    )
    bank = prepared.source_configs
    synthetic = options.case != "no-reset"
    if synthetic:
        horizons = (
            jnp.ones(42, jnp.int32)
            if options.case == "all-reset"
            else 3 + 2 * (jnp.arange(42, dtype=jnp.int32) % 2)
        )
        bank = bank._replace(max_steps=horizons)
    started = time.perf_counter()
    context = workflow.make_context(
        prepared=prepared,
        num_envs=batch,
        recording=True,
        metrics="priority",
        source_configs=bank if synthetic else None,
    )
    initial = context.carry
    jax.block_until_ready(initial)
    setup_seconds = time.perf_counter() - started
    step = workflow.make_step(context.actor, context.opponent)

    def generate(
        carry: workflow.Carry,
    ) -> tuple[workflow.Carry, workflow.DecisionInputs]:
        """Make a reference tape from the same accepted public trajectory."""

        def take(
            current: workflow.Carry, unused: None
        ) -> tuple[workflow.Carry, workflow.DecisionInputs]:
            """Save matching draw and key inputs; no tape enters production carry."""
            del unused
            inputs = workflow.decision_inputs(current)
            following, _ = step(current, inputs)
            return following, inputs

        return jax.lax.scan(take, carry, None, length=steps)

    tape_runner = jax.jit(generate)
    (_, tape), tape_seconds = _timed(tape_runner, initial)

    def advance(
        current: workflow.Carry, selected: workflow.DecisionInputs | None
    ) -> tuple[workflow.Carry, Tree]:
        """Retain identical compact transitions and source-start rows for both paths."""
        following, (transition, info, _) = step(current, selected)
        return following, (transition, info.episode_start_records)

    def sampled(carry: workflow.Carry) -> Tree:
        """Run the public loop with in-loop keys and conditional reset sampling."""
        return jax.lax.scan(advance, carry, None, length=steps)

    def reference(carry: workflow.Carry, selections: workflow.DecisionInputs) -> Tree:
        """Run the identical public loop with preselected configurations and keys."""
        return jax.lax.scan(advance, carry, selections)

    host_initial, to_host_seconds = _timed(jax.device_get, initial)

    def place(value: Tree) -> Tree:
        """Place the prepared host carry on the selected single device."""
        return jax.device_put(value, device)

    initial, placement_seconds = _timed(place, host_initial)
    del host_initial
    compiled, traces, sampled_report = _compile(sampled, initial)
    direct, direct_traces, direct_report = _compile(reference, initial, tape)
    actual, sampled_report["first_seconds"] = _timed(compiled, initial)
    expected, direct_report["first_seconds"] = _timed(direct, initial, tape)
    _equal(actual, expected)
    real_transitions = int(
        np.asarray(
            jnp.sum(
                actual[0].tracking.accounted_transition_count
                - initial.tracking.accounted_transition_count
            )
        )
    )
    resets = int(
        np.asarray(
            jnp.sum(actual[0].state.reset_generation - initial.state.reset_generation)
        )
    )
    assert real_transitions == batch * steps
    if options.case == "no-reset":
        assert resets == 0
    elif options.case == "all-reset":
        assert resets == batch * steps
    else:
        assert 0 < resets < batch * steps
    samples: dict[str, list[float]] = {"sampled": [], "preselected": []}
    with jax.transfer_guard("disallow_explicit"):
        for index in range(5):
            order = (
                ("sampled", "preselected")
                if index % 2 == 0
                else ("preselected", "sampled")
            )
            for name in order:
                _, duration = (
                    _timed(compiled, initial)
                    if name == "sampled"
                    else _timed(direct, initial, tape)
                )
                samples[name].append(duration)
    for name, report in (("sampled", sampled_report), ("preselected", direct_report)):
        report["warm_seconds"] = samples[name]
        report["median_seconds"] = statistics.median(samples[name])
        report["real_transitions_per_second"] = (
            real_transitions / report["median_seconds"]
        )
    changed_context = workflow.make_context(
        prepared=prepared,
        seed=71,
        num_envs=batch,
        team_size=2,
        eligible_maps=jnp.arange(42) < 17,
        recording=True,
        metrics="priority",
        source_configs=bank if synthetic else None,
    )
    changed = changed_context.carry
    (_, changed_tape), _ = _timed(tape_runner, changed)
    changed_actual, _ = _timed(compiled, changed)
    changed_expected, _ = _timed(direct, changed, changed_tape)
    _equal(changed_actual, changed_expected)
    assert traces == [1] and direct_traces == [1]
    output_host, output_transfer_seconds = _timed(jax.device_get, actual)
    writing: dict[str, Tree] | None = None
    if not options.skip_writing:
        with _TimedWriter(
            options.output / "recording",
            phase="training",
            policies={"team_a": context.actor, "team_b": context.opponent},
        ) as writer:
            started = time.perf_counter()
            final, transitions = marl_bgs.collect_rollout(
                step, initial, num_steps=steps, writer=writer, source_configs=bank
            )
            jax.block_until_ready((final, transitions))
            collection_seconds = time.perf_counter() - started
            _equal(final, actual[0])
            _equal(transitions, actual[1][0])
            started = time.perf_counter()
            writer.flush()
            flush_seconds = time.perf_counter() - started
            writing = {
                "collection_including_compilation_seconds": collection_seconds,
                "writer_drain_seconds": cast(_TimedWriter, writer).drain_seconds,
                "writer_drain_bytes": cast(_TimedWriter, writer).drain_bytes,
                "writer_drain_count": cast(_TimedWriter, writer).drain_count,
                "final_flush_seconds": flush_seconds,
                "durable_bytes": sum(
                    path.stat().st_size
                    for path in writer.run_dir.rglob("*")
                    if path.is_file()
                ),
            }
    root = Path(__file__).resolve().parents[2]
    paths = [
        "src/marl_battlegrounds/training/_content.py",
        "src/marl_battlegrounds/training/distributions.py",
        "src/marl_battlegrounds/tasks.py",
        "src/marl_battlegrounds/episode_tracking.py",
        "src/marl_battlegrounds/collection.py",
        "src/marl_battlegrounds/evaluation/run_writer.py",
        "examples/training_distributions.py",
        "scripts/dev/benchmark_training_distributions.py",
        "uv.lock",
    ]
    gpu = None
    if backend == "gpu":
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,pci.bus_id,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    report = {
        "case": options.case,
        "synthetic_reset_stress": synthetic,
        "backend": backend,
        "cpu_smoke_only": options.cpu_smoke,
        "batch": batch,
        "steps": steps,
        "metrics": "priority",
        "method": "Random",
        "real_transitions": real_transitions,
        "episode_resets": resets,
        "preparation_profiled_seconds": preparation_seconds,
        "hashing_profile_self_seconds": hash_self_seconds,
        "context_setup_seconds": setup_seconds,
        "setup_device_to_host_seconds": to_host_seconds,
        "placement_seconds": placement_seconds,
        "placed_bytes": _bytes(initial),
        "reference_tape_prepare_including_compile_seconds": tape_seconds,
        "reference_tape_bytes": _bytes(tape),
        "sampled": sampled_report,
        "preselected": direct_report,
        "exact_output_agreement": True,
        "traces_after_changed_inputs": [len(traces), len(direct_traces)],
        "warm_transfers": "Explicit and implicit transfers disallowed",
        "retained_output_bytes": _bytes(actual),
        "output_device_to_host_seconds": output_transfer_seconds,
        "writer": writing,
        "peak_process_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "device_allocator_statistics": device.memory_stats(),
        "gpu_query_after_run": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "verified_base_binding": prepared.binding.canonical_digest,
        "actual_source_bank_identity": ordered_source_bank_identity(bank)[0],
        "source_sha256": {
            path: sha256((root / path).read_bytes()).hexdigest() for path in paths
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "numpy", "marl-battlegrounds")
        },
        "limits": (
            "One GPU, Random Systems, fixed B32/T128. Reset stress uses changed "
            "horizons. No learner or learning throughput. RAM/allocator peaks "
            "cover the whole process including reference tapes, compilers and "
            "writing. Setup profiling adds overhead; hashing self time excludes "
            "serialization. Preselected tape storage is reference-only and "
            "never a production rollout buffer."
        ),
    }
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez(
        options.output / "final_counts.npz",
        counts=_array(output_host[0].tracking.accounted_transition_count),
        generations=_array(output_host[0].state.reset_generation),
    )
    print(
        json.dumps(
            {
                name: report[name]
                for name in (
                    "case",
                    "real_transitions",
                    "episode_resets",
                    "sampled",
                    "preselected",
                    "writer",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
