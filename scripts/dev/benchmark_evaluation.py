"""Qualify complete JAX metric collection and native rollout cost on CPU/GPU.

Run with the CUDA environment, for example:
  python -m scripts.dev.benchmark_evaluation --output artifacts/m8-performance
Each requested size uses a fresh process. No result silently substitutes a smaller
batch. Full authoritative facts are retained only for this comparison, not by the
production evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core.types import (
    Action,
    ActionMask,
    EnvConfig,
    EnvState,
    Info,
    Reward,
)
from marl_battlegrounds.environment import (
    Environment,
    EnvironmentState,
    EpisodeInfo,
    make,
)
from marl_battlegrounds.evaluation.episode_metrics import (
    MetricValues,
    PriorityTotals,
    initialize_priority,
    priority_values,
    update_priority,
)
from marl_battlegrounds.evaluation.evaluate import (  # pyright: ignore[reportPrivateUsage]
    _actor_keys,  # pyright: ignore[reportPrivateUsage]
    _empty_completed,  # pyright: ignore[reportPrivateUsage]
    _retain_completion,  # pyright: ignore[reportPrivateUsage]
    episode_keys,
)
from marl_battlegrounds.evaluation.full_metrics import (
    FullTotals,
    full_values,
    initialize_full,
    update_full,
)
from marl_battlegrounds.evaluation.metric_catalog import FULL_METRIC_NAMES
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyApply,
    apply_policies,
    policy,
)
from marl_battlegrounds.policies.input import ActorInput, Observations
from marl_battlegrounds.tasks import (
    CANONICAL_TDM_EVALUATION_MAP_IDS,
    make_standard_team_deathmatch_config,
)

type Tree = Any

_POLICIES: dict[str, object] = {
    "team_a": "tdm-alpha-exploration-10pct",
    "team_b": "tdm-beta-exploration-10pct",
}


def _exploratory_apply(name: str) -> PolicyApply:
    controller, random = policy(name).apply, policy("random").apply

    def apply(
        variables: Tree, carry: Tree, actor: ActorInput, mask: ActionMask, key: Array
    ) -> tuple[Tree, Tree]:
        selection_key, controller_key, random_key = jax.random.split(key, 3)
        action, next_carry = controller(variables, carry, actor, mask, controller_key)
        exploration, _ = random((), (), actor, mask, random_key)
        explore = jax.random.bernoulli(selection_key, 0.1)

        def choose(chosen: Array, usual: Array) -> Array:
            return jnp.where(explore, chosen, usual)

        return jax.tree.map(choose, exploration, action), next_carry

    return apply


class Facts(NamedTuple):
    state: EnvState
    mask: ActionMask
    info: Info
    reward: Reward


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src" / "marl_battlegrounds").rglob("*.py"))
    files.append(Path(__file__).resolve())
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _schedule(count: int) -> list[dict[str, object]]:
    maps = CANONICAL_TDM_EVALUATION_MAP_IDS
    return [
        {"episode_id": lane + 1, "seed_id": lane + 1, "map_id": maps[lane % len(maps)]}
        for lane in range(count)
    ]


def _bytes(tree: object) -> int:
    return sum(
        math.prod(value.shape) * np.dtype(value.dtype).itemsize
        for value in jax.tree.leaves(tree)
    )


def _platforms(tree: object) -> list[str]:
    return sorted(
        {
            device.platform
            for value in jax.tree.leaves(tree)
            for device in value.devices()
        }
    )


def _configs(count: int) -> EnvConfig:
    canonical = ("mage", "warrior", "hunter", "rogue", "priest")
    maps = CANONICAL_TDM_EVALUATION_MAP_IDS
    configurations = [
        make_standard_team_deathmatch_config(
            map_id=maps[index % len(maps)],
            team_a_roster=("mage", "mage", "priest") if index % 16 == 15 else canonical,
            team_b_roster=("warrior", "priest") if index % 16 == 15 else canonical,
            max_steps=300,
        )
        for index in range(count)
    ]
    return jax.tree.map(lambda *rows: jnp.stack(rows), *configurations)


def _actions(state: EnvironmentState, env: Environment, tick: Array) -> Action:
    first, second = _exploratory_apply("tdm-alpha"), _exploratory_apply("tdm-beta")
    actor_keys = _actor_keys(
        jax.random.key(42), state.episode_id, jnp.full_like(state.episode_id, tick)
    )

    def choose(observation: Observations, mask: ActionMask, keys: Array) -> Action:
        return apply_policies(first, second, (), (), (), (), observation, mask, keys)[0]

    return jax.vmap(choose)(env._observations(state), state.action_mask, actor_keys)  # pyright: ignore[reportPrivateUsage]


def _rollout(env: Environment, *, frozen_facts: bool = False) -> Callable[..., Any]:
    def run(initial: EnvironmentState) -> Tree:
        def transition(carry: Tree, tick: Array) -> tuple[Tree, Tree]:
            state, completion = carry
            actions = _actions(state, env, tick)
            keys = episode_keys(
                jax.random.key(42),
                state.episode_id,
                jnp.full_like(state.episode_id, tick),
                1,
            )
            if frozen_facts:
                successor, reward, _, info = jax.vmap(env._step_one)(  # pyright: ignore[reportPrivateUsage]
                    keys, state, actions
                )  # pyright: ignore[reportPrivateUsage]
                return (successor, None), Facts(
                    state.core_state, state.action_mask, info, reward
                )
            _, successor, _, _, info = env.step(keys, state, actions)
            # Returning capture packets is intentional logging-chunk storage;
            # ordinary throughput retains no per-step metric table.
            completion = _retain_completion(completion, successor, info)
            return (successor, completion), info.replay

        carry = (initial, None if frozen_facts else _empty_completed(initial))
        (final, completed), records = jax.lax.scan(
            transition, carry, jnp.arange(300, dtype=jnp.int32)
        )
        return (final, records) if frozen_facts else (final, (completed, records))

    return run


def _metrics(
    initial: EnvironmentState, final: EnvironmentState, facts: Facts
) -> tuple[FullTotals, MetricValues, MetricValues]:
    totals = jax.vmap(initialize_full)(initial.config, initial.core_state)

    def empty(_: Array) -> PriorityTotals:
        return initialize_priority()

    priority = jax.vmap(empty)(initial.episode_id)

    def collect(
        carry: tuple[FullTotals, PriorityTotals], step: Facts
    ) -> tuple[tuple[FullTotals, PriorityTotals], None]:
        full, priority = carry
        full = jax.vmap(update_full)(
            full, initial.config, step.state, step.mask, step.info
        )
        priority = jax.vmap(update_priority)(priority, step.reward, step.info)
        return (full, priority), None

    (totals, priority), _ = jax.lax.scan(collect, (totals, priority), facts)
    outcome = facts.info.transition_facts.team_deathmatch_facts.outcome.max(axis=0)
    critical = jax.vmap(priority_values)(
        final.config, final.core_state, initial.initial_step_count, priority, outcome
    )
    return totals, critical, jax.vmap(full_values)(totals, final.config, critical)


def _measure(
    function: Callable[..., Any], args: tuple[Any, ...], repeats: int
) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    executable = cast(Any, jax.jit(function).lower(*args).compile())
    compilation = time.perf_counter() - started
    started = time.perf_counter()
    result = cast(Any, jax.block_until_ready(executable(*args)))
    first = time.perf_counter() - started
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = cast(Any, jax.block_until_ready(executable(*args)))
        samples.append((time.perf_counter() - started) * 1000)
    memory = executable.memory_analysis()
    return result, {
        "compilation_seconds": compilation,
        "first_execution_ms": first * 1000,
        "warm_median_ms": statistics.median(samples),
        "warm_min_ms": min(samples),
        "warm_max_ms": max(samples),
        "warm_samples_ms": samples,
        "compiler_memory_bytes": None
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


def _assert_equal(first: Tree, second: Tree) -> None:
    for left, right in zip(
        jax.tree.leaves(first), jax.tree.leaves(second), strict=True
    ):
        if np.issubdtype(left.dtype, np.integer) or left.dtype == np.bool_:
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, rtol=3e-5, atol=2e-3)


def _notify(message: str) -> None:
    print(message, flush=True)


def run_size(count: int, output: Path, repeats: int) -> dict[str, object]:
    gpu, cpu = cast(Any, jax.devices("gpu")[0]), cast(Any, jax.devices("cpu")[0])
    cpu_info = Path("/proc/cpuinfo")
    cpu_model = (
        next(
            (
                line.partition(":")[2].strip()
                for line in cpu_info.read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
        if cpu_info.exists()
        else platform.processor()
    )
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    result: dict[str, Any] = {
        "num_envs": count,
        "worker_pid": os.getpid(),
        "horizon": 300,
        "full_columns": len(FULL_METRIC_NAMES),
        "device": str(gpu),
        "gpu_model": gpu.device_kind,
        "cpu_model": cpu_model,
        "host_platform": platform.platform(),
        "gpu_driver_inventory": driver,
        "gpu_runtime": str(gpu.client.platform_version),
        "precision": {
            "floating_accumulators": "float32",
            "integer_accumulators": "int32",
            "controlled_effect_matmul": "HIGHEST",
            "jax_enable_x64": cast(bool, jax.config.read("jax_enable_x64")),
            "float_comparison_rtol": 3e-5,
            "float_comparison_atol": 0.002,
        },
        "source_file_sha256": _source_hashes(),
        "jax": jax.__version__,
        "repeats": repeats,
        "policies": _POLICIES,
        "exploration_probability": 0.1,
        "rosters": {"canonical_5v5": count - count // 16, "repeated_3v2": count // 16},
    }
    output.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        (output / f"batch-{count}.json").write_text(json.dumps(result, indent=2) + "\n")

    result["status"] = "running"
    save()
    with jax.default_device(gpu):
        configs = _configs(count)
        ids = jnp.arange(1, count + 1, dtype=jnp.int32)
        env = make("tdm", num_envs=count, metrics="none")
        reset_keys = episode_keys(jax.random.key(42), ids, jnp.zeros_like(ids), 0)

        def reset(config: EnvConfig) -> tuple[Observations, EnvironmentState]:
            return env.reset(reset_keys, config, episode_id=ids)

        observations, initial = cast(
            tuple[Observations, EnvironmentState],
            jax.block_until_ready(
                cast(tuple[Observations, EnvironmentState], jax.jit(reset)(configs))
            ),
        )
        shapes = jax.eval_shape(_rollout(env, frozen_facts=True), initial)
        result["frozen_output_bytes"] = _bytes(shapes)
        result["environment_state_bytes"] = _bytes(initial)
        result["observations_bytes"] = _bytes(observations)
        result["observation_rollout_128_bytes"] = _bytes(observations) * 128
        result["mask_rollout_128_bytes"] = _bytes(initial.action_mask) * 128
        stats: dict[str, Any] = gpu.memory_stats() or {}
        limit = int(stats.get("bytes_limit", 0))
        if limit and _bytes(shapes) * 2 + _bytes(initial) > limit * 0.8:
            raise MemoryError(
                "frozen-fact comparison exceeds preflight VRAM budget; "
                "native batch was not reduced"
            )
        _notify(
            f"B{count}: compiling/generating 300-step exploratory ALPHA/BETA episodes"
        )
        (final, facts), generation = _measure(
            _rollout(env, frozen_facts=True), (initial,), 1
        )
        result["fact_generation"] = generation
        result["action_rollout_128_bytes"] = (
            _bytes(
                facts.info.transition_facts.action_acceptance_facts.accepted_joint_action
            )
            // 300
            * 128
        )
        result["reward_rollout_128_bytes"] = _bytes(facts.reward) // 300 * 128
        lengths = np.asarray(jax.device_get(final.core_state.step_count))
        if not np.asarray(jax.device_get(final.done.done)).all():
            raise AssertionError("benchmark did not finish every episode")
        result["completed_episodes"] = int(count)
        result["real_transitions"] = int(lengths.sum())
        result["padding_transitions"] = int(300 * count - lengths.sum())
        result["episode_lengths"] = lengths.tolist()
        semantic_traces = jax.device_get(
            (
                facts.state.agent_positions,
                facts.state.current_health,
                facts.info.transition_facts.action_acceptance_facts.accepted_joint_action,
            )
        )
        trace_hashes: list[str] = []
        for lane, length in enumerate(lengths):
            digest = sha256()
            for value in jax.tree.leaves(semantic_traces):
                digest.update(np.asarray(value[: int(length), lane]).tobytes())
            trace_hashes.append(digest.hexdigest())
        result["unique_semantic_trajectories"] = len(set(trace_hashes))
        result["trajectory_sha256"] = trace_hashes
        if len(set(trace_hashes)) != count:
            raise AssertionError("benchmark contains repeated semantic trajectories")
        del semantic_traces
        _notify(f"B{count}: complete frozen-fact metrics on GPU")
        gpu_values, gpu_metrics = _measure(_metrics, (initial, final, facts), repeats)
        gpu_metrics["result_platforms"] = _platforms(gpu_values)
        if gpu_metrics["result_platforms"] != ["gpu"]:
            raise AssertionError("GPU metric results did not execute on GPU")
        gpu_metrics["amortized_ms_per_episode"] = (
            float(gpu_metrics["warm_median_ms"]) / count
        )
        result["gpu_metrics"] = gpu_metrics
        started = time.perf_counter()
        host_values = jax.device_get(gpu_values)
        result["metric_host_transfer_ms"] = (time.perf_counter() - started) * 1000
        _notify(f"B{count}: identical frozen-fact metrics on CPU")
        cpu_args = jax.device_put((initial, final, facts), cpu)
        jax.block_until_ready(cpu_args)
        cpu_values, cpu_metrics = _measure(_metrics, cpu_args, repeats)
        cpu_metrics["result_platforms"] = _platforms(cpu_values)
        if cpu_metrics["result_platforms"] != ["cpu"]:
            raise AssertionError("CPU metric results did not execute on CPU")
        cpu_metrics["amortized_ms_per_episode"] = (
            float(cpu_metrics["warm_median_ms"]) / count
        )
        result["cpu_metrics"] = cpu_metrics
        save()
        _assert_equal(host_values, jax.device_get(cpu_values))
        result["cpu_gpu_agreement"] = (
            "PASS: exact integers/validity; float rtol=3e-5, atol=0.002"
        )
        save()
        reference_final = jax.device_get(final.core_state)
        reference_priority, reference_full = host_values[1:]

        from marl_battlegrounds.evaluation.run_writer import RunWriter

        _, priority, full = gpu_values
        outcomes = facts.info.transition_facts.team_deathmatch_facts.outcome.max(axis=0)
        completed = EpisodeInfo(
            ids, jnp.ones(count, bool), outcomes, configs, priority, full, None
        )
        started = time.perf_counter()
        with RunWriter(
            output / "csv",
            policies=_POLICIES,
            details={
                "seed": 42,
                "rng_protocol": "episode-fold-in-v1",
                "num_envs": count,
            },
        ) as writer:
            writer.register_episodes(_schedule(count))
            writer.write(completed)
            writer.flush()
        result["full_csv_write_and_fsync_ms"] = (time.perf_counter() - started) * 1000
        result["csv_run_dir"] = str(writer.run_dir)
        # Release the benchmark-only history before measuring streaming execution.
        del facts, cpu_args, cpu_values, gpu_values, host_values, final
        import gc

        gc.collect()
        rollouts: dict[str, object] = {}
        baseline: float | None = None
        for mode in ("none", "priority", "full", "sparse_full", "replay"):
            _notify(f"B{count}: native {mode} rollout")
            current = make(
                "tdm",
                num_envs=count,
                metrics="full"
                if mode == "full"
                else "none"
                if mode == "none"
                else "priority",
                full_metrics_episodes=range(1, count + 1, 16)
                if mode == "sparse_full"
                else (),
                replay_episodes=range(1, min(4, count) + 1) if mode == "replay" else (),
            )

            def reset_current(
                config: EnvConfig, current: Environment = current
            ) -> tuple[Observations, EnvironmentState]:
                return current.reset(reset_keys, config, episode_id=ids)

            _, state = cast(
                tuple[Observations, EnvironmentState], jax.jit(reset_current)(configs)
            )
            (end, (terminal, packets)), timing = _measure(
                _rollout(current), (state,), repeats
            )
            if not np.asarray(jax.device_get(end.done.done)).all():
                raise AssertionError(f"{mode} did not complete every episode")
            for expected, actual in zip(
                jax.tree.leaves(reference_final),
                jax.tree.leaves(jax.device_get(end.core_state)),
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            timing["same_final_state_as_frozen_facts"] = True
            if terminal.priority is not None:
                _assert_equal(reference_priority, jax.device_get(terminal.priority))
            if terminal.full is not None:
                host_full = jax.device_get(terminal.full)
                selected = (
                    np.arange(count) % 16 == 0
                    if mode == "sparse_full"
                    else np.ones(count, bool)
                )
                _assert_equal(
                    (reference_full.values[selected], reference_full.valid[selected]),
                    (host_full.values[selected], host_full.valid[selected]),
                )
                if host_full.valid[~selected].any():
                    raise AssertionError("unselected native full metrics were exposed")
            timing["same_requested_metrics_as_frozen_facts"] = True
            median = float(timing["warm_median_ms"])
            if baseline is None:
                baseline = median
            timing["added_ms_vs_none"] = median - baseline
            timing["transitions_per_second"] = int(lengths.sum()) * 1000 / median
            timing["state_bytes"] = _bytes(state)
            timing["capture_output_bytes"] = _bytes(packets)
            if packets is not None:
                started = time.perf_counter()
                host_packets = jax.device_get(packets)
                timing["capture_host_transfer_ms"] = (
                    time.perf_counter() - started
                ) * 1000
                started = time.perf_counter()
                with RunWriter(
                    output / "replays",
                    policies=_POLICIES,
                    details={
                        "seed": 42,
                        "rng_protocol": "episode-fold-in-v1",
                        "num_envs": count,
                    },
                ) as writer:
                    writer.register_episodes(_schedule(min(4, count)))
                    writer.write_replay(host_packets)
                    writer.flush()
                timing["replay_build_write_and_fsync_ms"] = (
                    time.perf_counter() - started
                ) * 1000
                timing["replay_run_dir"] = str(writer.run_dir)
            rollouts[mode] = timing
            result["native_gpu_rollouts"] = rollouts
            save()
            del state, end, terminal, packets
            gc.collect()
        result["jax_gpu_memory_stats"] = gpu.memory_stats()
        current_hashes = _source_hashes()
        result["source_changes_during_measurement"] = {
            name: current_hashes.get(name)
            for name in sorted(
                result["source_file_sha256"].keys() | current_hashes.keys()
            )
            if result["source_file_sha256"].get(name) != current_hashes.get(name)
        }
        result["process_peak_ram_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        result["status"] = (
            "source_changed"
            if result["source_changes_during_measurement"]
            else "complete"
        )
        save()
        if result["source_changes_during_measurement"]:
            raise RuntimeError(
                "source changed during measurement; "
                "retain diagnostics and rerun qualification"
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 2 or any(size < 1 for size in args.sizes):
        parser.error("positive batch sizes and at least two warm repeats are required")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        run_size(args.worker, args.output, args.repeats)
        return 0
    failed = False
    for size in args.sizes:
        path = args.output / f"batch-{size}.json"
        path.write_text(json.dumps({"num_envs": size, "status": "starting"}) + "\n")
        command = [
            sys.executable,
            "-m",
            "scripts.dev.benchmark_evaluation",
            "--worker",
            str(size),
            "--repeats",
            str(args.repeats),
            "--output",
            str(args.output),
        ]
        completed = subprocess.run(
            command,
            check=False,
            env={
                **os.environ,
                "JAX_PLATFORMS": "cuda,cpu",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            },
        )
        if completed.returncode:
            failed = True
            result = (
                json.loads(path.read_text()) if path.exists() else {"num_envs": size}
            )
            result.update(status="failed", exit_code=completed.returncode)
            path.write_text(json.dumps(result, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
