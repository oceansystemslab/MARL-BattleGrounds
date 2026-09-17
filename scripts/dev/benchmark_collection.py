"""Measure optional bounded collection through the existing evaluation benchmark.

Run ``python -m scripts.dev.benchmark_evaluation --collection --output PATH``.
Use --collection-case to isolate direct scan, immediate writer, collected outcomes,
traces, or selected replays. The fixed workload is 32 games and 16 decisions with
unequal short games. At least five synchronized warm runs are required. This
measures recording and restart overhead, not learned behavior or maximum speed.
The manual case can use the committed package through PYTHONPATH and --package-root.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.evaluation.policy_execution import PolicyTrace
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

type Tree = Any
CASES = ("disabled", "manual", "outcomes", "traces", "replay")


class Carry(NamedTuple):
    """Keep the dynamic handle, live state, tracker and next random key together."""

    env: Tree
    state: Tree
    tracking: Tree
    rng: Tree


def _step(carry: Carry, unused: None) -> tuple[Carry, Tree]:
    """Choose one legal batch, advance it and return compact learner evidence.

    carry is dynamic. unused is the scan placeholder. The trace describes the
    same random-valid decision; no second action call occurs. Return latest carry
    and (learner outputs, old episode info, trace). Start production is selected
    by the tracker structure created during setup.
    """
    del unused
    env, before, tracker, rng = carry
    rng, action_key, step_key = jax.random.split(rng, 3)
    actions = env.sample_actions(action_key, before)
    tracker, result = marl_bgs.track_episode_step(
        tracker, before, env.step(step_key, before, actions)
    )
    _, after, reward, done, info = result
    valid = info.decision_step >= 0
    trace = PolicyTrace(
        info.episode_id,
        info.decision_step,
        valid,
        jnp.where(info.active_mask, jnp.int32(0), jnp.int32(-1)),
    )
    transition = (
        actions,
        reward,
        done,
        info.episode_id,
        info.decision_step,
        info.episode_length,
        info.team_scores,
    )
    return Carry(env, after, tracker, rng), (transition, info, trace)


def _without_trace(carry: Carry, unused: None) -> tuple[Carry, Tree]:
    """Use the same numerical decision while omitting optional assignment records."""
    latest, (transition, info, _) = _step(carry, unused)
    return latest, (transition, info, None)


def _scan(carry: Carry) -> tuple[Carry, Tree]:
    """Run the exact sixteen-decision no-file numerical reference."""

    def advance(current: Carry, unused: None) -> tuple[Carry, Tree]:
        """Retain only caller-selected learner outputs, without recording history."""
        latest, (transition, _, _) = _step(current, unused)
        return latest, transition

    return jax.lax.scan(advance, carry, None, length=16)


def _context(case: str, map_id: int) -> Carry:
    """Prepare immutable matching assets and dynamic inputs for one measured case.

    case selects only optional starts/traces/replay work. map_id chooses the same
    frozen map for both source rows. Source horizons 3 and 7 force unequal resets;
    initial source indices preserve equal spawn choices in both batch halves.
    """
    roster_a, roster_b = marl_bgs.canonical_tournament_rosters()
    sources = [
        make_standard_team_deathmatch_config(
            map_id=map_id,
            max_steps=length,
            team_a_roster=roster_a,
            team_b_roster=roster_b,
        )
        for length in (3, 7)
    ]
    bank = jax.tree.map(lambda *values: jnp.stack(values), *sources)
    indices = jnp.arange(32, dtype=jnp.int32) % 2
    config = marl_bgs.balanced_spawn_configs(
        jax.tree.map(lambda x: x[indices], bank), num_envs=32
    )
    env = marl_bgs.AutoReset(
        marl_bgs.make(
            "tdm",
            num_envs=32,
            env_config=config,
            metrics="priority",
            replay_episodes=(1, 225) if case == "replay" else (),
        )
    )
    _, state = env.reset(jax.random.key(42))
    tracking = marl_bgs.init_episode_tracking(
        env,
        state,
        source_configs=bank,
        source_indices=indices,
        record_starts=case != "disabled",
    ).begin_stage(state, total_env_steps=512)
    return Carry(env, state, tracking, jax.random.PRNGKey(43))


def _write_manual(carry: Carry, writer: Tree, compiled: Tree) -> tuple[Carry, Tree]:
    """Record sixteen compiled decisions through the existing immediate writer.

    writer is a fresh run; compiled is the reusable one-step function. Transfer
    only info through the normal writer API. Stack learner outputs once at the
    end as a simple correct reference, not a production collection strategy.
    """
    outputs = []
    for _ in range(16):
        carry, (transition, info, trace) = compiled(carry, None)
        writer.register_episodes(
            info.episode_start_records, source_configs=carry.tracking.source_configs
        )
        writer.write(info, policy_trace=trace)
        outputs.append(transition)
    return carry, jax.tree.map(lambda *rows: jnp.stack(rows), *outputs)


def _synchronized(call: Tree) -> tuple[Tree, float]:
    """Call a zero-argument measurement and wait for all returned JAX arrays."""
    start = time.perf_counter()
    result = jax.block_until_ready(call())
    return result, (time.perf_counter() - start) * 1000


def _prepare_collection_measurement(
    carry: Carry, step_fn: Tree
) -> tuple[Tree, dict[str, Any]]:
    """Compile the exact private chunk separately from host recording work.

    carry and step_fn are the same values passed to collect_rollout. Inspect
    its existing preparation authority rather than build a second collector.
    Return the cached kernel and structural bytes/compiler memory. Preparation
    executes no action and does not create a writer or registration.
    """
    from marl_battlegrounds import collection
    from scripts.dev.benchmark_evaluation import _bytes

    started = time.perf_counter()
    prepared = collection._prepare(step_fn, carry, None)
    batch = collection._empty_batch(prepared)
    output = collection._zeros_from_shape(prepared.transition, 16, 0)
    jax.block_until_ready((batch, output))
    setup_ms = (time.perf_counter() - started) * 1000
    kernel = collection._compiled_chunk(
        collection._StepIdentity(step_fn),
        16,
        prepared.batch_size,
        prepared.capacity,
        prepared.replay_capacity,
        prepared.scalar,
    )
    started = time.perf_counter()
    executable = kernel.lower(
        carry, output, batch.buffers, batch.counts, batch.errors, jnp.int32(0)
    ).compile()
    compile_seconds = time.perf_counter() - started
    memory = executable.memory_analysis()
    return kernel, {
        "abstract_preparation_allocation_ms": setup_ms,
        "compilation_seconds": compile_seconds,
        "recording_buffer_bytes": _bytes(batch),
        "learner_output_bytes": _bytes(output),
        "compiler_memory": {
            name: getattr(memory, name)
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
        }
        if memory is not None
        else None,
    }


def run(args: argparse.Namespace) -> int:
    """Run a bounded collection case and retain raw timings and source identities.

    args is the shared CLI namespace. Backend defaults to GPU; CPU is correctness
    diagnostics only. sizes/lengths must be absent or 32/16. Existing evidence is
    never overwritten. Manual and disabled cases work against the committed
    reference package. Checkpoint measurements require the candidate writer.
    """
    from scripts.dev.benchmark_evaluation import (
        _assert_equal,
        _bytes,
        _foundation_assets,
        _foundation_digest,
        _foundation_identity,
        _foundation_measure,
    )

    if (
        args.repeats < 5
        or args.sizes not in (None, [32])
        or args.lengths not in (None, [16])
    ):
        raise ValueError(
            "Collection uses batch 32, rollout 16 and at least five warm samples"
        )
    if any(
        (
            args.foundations,
            args.systems,
            args.recording_contracts,
            args.tracking,
            args.metrics_only,
            args.include_scalar,
        )
    ):
        raise ValueError("Collection cannot be combined with another benchmark mode")
    package = (args.package_root or Path(__file__).resolve().parents[2]).resolve()
    assets = (args.assets_root or package / "src/marl_battlegrounds/data/tdm").resolve()
    identity = _foundation_identity(package, assets)
    identity["collection_harness_sha256"] = sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    _foundation_assets(assets)
    device = jax.devices(args.backend or "gpu")[0]
    args.output.mkdir(parents=True, exist_ok=True)
    cases = (args.collection_case,) if args.collection_case else CASES
    method = marl_bgs.shared_policy(marl_bgs.policy("random"))
    for case in cases:
        path = args.output / f"collection-{case}.json"
        if path.exists():
            raise FileExistsError(f"Choose a new evidence directory: {path}")
        record: dict[str, Any] = {
            "status": "running",
            "case": case,
            "identity": identity,
            "device": str(device),
            "jax": jax.__version__,
            "batch": 32,
            "rollout_length": 16,
            "scope": "Unequal short-game recording stress",
        }
        path.write_text(json.dumps(record, indent=2) + "\n")
        try:
            with jax.default_device(device):
                initial, setup_ms = _synchronized(
                    partial(
                        _context, case, args.map_id if args.map_id is not None else 12
                    )
                )
                record["setup_ms"] = setup_ms
                record["initial_bytes"] = _bytes(initial)
                reference, scan_timing, _ = _foundation_measure(
                    _scan, (initial,), args.repeats
                )
                record["direct_scan"] = scan_timing
                record["learner_output_bytes"] = _bytes(reference[1])
                record["reference_transition_digest"] = _foundation_digest(reference[1])
                if case == "disabled":
                    samples = scan_timing["warm_samples_ms"]
                    record["recording_work"] = "No writer or collector constructed"
                else:
                    kernel = None
                    step_fn = _step if case in ("traces", "replay") else _without_trace
                    if case != "manual":
                        kernel, preparation = _prepare_collection_measurement(
                            initial, step_fn
                        )
                        record["collection_preparation"] = preparation
                    compiled = jax.jit(_step)
                    started = time.perf_counter()
                    compiled.lower(initial, None).compile()
                    record["manual_step_compile_seconds"] = (
                        time.perf_counter() - started
                    )
                    samples = []
                    flush_samples = []
                    drain_samples = []
                    transfer_samples = []
                    cache_samples = []
                    for repeat in range(args.repeats + 1):
                        writer = marl_bgs.RunWriter(
                            args.output / case / f"sample-{repeat}",
                            phase="training",
                            policies={"team_a": method, "team_b": method},
                        )
                        drains: list[dict[str, int]] = []
                        if case != "manual":
                            original = writer._write_collected

                            def drain(
                                batch: Tree,
                                *,
                                source_configs: Tree = None,
                                _drains: list[dict[str, int]] = drains,
                                _original: Tree = original,
                            ) -> None:
                                """Count bounded payloads outside numerical work."""
                                _drains.append({"allocated_bytes": _bytes(batch)})
                                _original(batch, source_configs=source_configs)

                            writer._write_collected = drain
                        try:
                            if case == "manual":
                                result, elapsed = _synchronized(
                                    partial(_write_manual, initial, writer, compiled)
                                )
                            else:
                                result, elapsed = _synchronized(
                                    partial(
                                        marl_bgs.collect_rollout,
                                        step_fn,
                                        initial,
                                        num_steps=16,
                                        writer=writer,
                                        source_configs=initial.tracking.source_configs,
                                    )
                                )
                            _assert_equal(result, reference)
                            if kernel is not None:
                                cache_samples.append(kernel._cache_size())
                            _, flush_ms = _synchronized(writer.flush)
                            if repeat == 0:
                                record["first_collection_including_compile_ms"] = (
                                    elapsed
                                )
                                record["first_drain_count"] = len(drains)
                            else:
                                samples.append(elapsed)
                                flush_samples.append(flush_ms)
                                drain_samples.append(len(drains))
                                transfer_samples.append(
                                    sum(row["allocated_bytes"] for row in drains)
                                )
                            if repeat == args.repeats and case in (
                                "outcomes",
                                "replay",
                            ):
                                token, checkpoint_ms = _synchronized(
                                    writer.checkpoint_recording
                                )
                                run_dir = writer.run_dir
                                record["checkpoint_ms"] = checkpoint_ms
                                bundle = (
                                    run_dir
                                    / "recording_checkpoints"
                                    / token["checkpoint_id"]
                                )
                                record["checkpoint_bytes"] = sum(
                                    p.stat().st_size
                                    for p in bundle.rglob("*")
                                    if p.is_file()
                                )
                                record["output_file_bytes"] = sum(
                                    p.stat().st_size
                                    for p in run_dir.rglob("*")
                                    if p.is_file()
                                )
                                writer.close()
                                started = time.perf_counter()
                                restored = marl_bgs.RunWriter(
                                    resume_from=run_dir,
                                    recording_checkpoint=token,
                                    phase="training",
                                    policies={"team_a": method, "team_b": method},
                                )
                                record["restore_ms"] = (
                                    time.perf_counter() - started
                                ) * 1000
                                restored.close()
                        finally:
                            writer.close()
                    record["final_flush_samples_ms"] = flush_samples
                    record["drain_counts"] = drain_samples
                    record["drain_allocated_bytes_upper_bound"] = transfer_samples
                    record["kernel_cache_entries_after_each_run"] = cache_samples
                record["warm_samples_ms"] = samples
                record["warm_median_ms"] = statistics.median(samples)
                record["real_transitions"] = 512
                record["real_transitions_per_second"] = 512_000 / statistics.median(
                    samples
                )
                record["device_memory_stats"] = device.memory_stats()
                record["process_peak_ram_bytes"] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                )
                record["measurement_limits"] = [
                    "Allocator high water includes compilation and reference work "
                    "in this process",
                    "Drain allocated bytes are an upper bound; occupied host slices "
                    "may transfer less",
                    "Warm collection excludes separately reported final flush "
                    "and checkpoint costs",
                ]
                # A benchmark cannot qualify a source tree that changed during its run.
                final_identity = _foundation_identity(package, assets)
                if final_identity != {
                    k: v
                    for k, v in identity.items()
                    if k != "collection_harness_sha256"
                }:
                    raise RuntimeError("Source or assets changed during measurement")
                record["status"] = "complete"
        except BaseException as error:
            record["status"] = "failed"
            record["error"] = repr(error)
            raise
        finally:
            path.write_text(json.dumps(record, indent=2) + "\n")
    return 0
