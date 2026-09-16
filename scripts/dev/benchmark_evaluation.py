"""Qualify complete JAX metric collection and native rollout cost on CPU/GPU.

Run with the CUDA environment, for example:
  python -m scripts.dev.benchmark_evaluation --output artifacts/m8-performance
Add --metrics-only --map-id 48 to simulate each fixed-map ALPHA/BETA batch once,
then compare CPU/GPU metric passes over those same facts.
Each requested size uses a fresh process. No result silently substitutes a smaller
batch. Full authoritative facts are retained only for this comparison, not by the
production evaluator.

The separate --foundations mode qualifies GPU setup and explicit-reset workflows.
CPU mode is available for diagnostics; CPU timings are not acceptance evidence.
Use --setup-only for preparation checks, or --action-workload sample to include
legal action sampling. GPU environment sizes are 32, 64, 128, 512 and 1024.

Use --systems --package-root CANDIDATE --baseline-root BASELINE --output RESULTS
for the bounded Packet 2 comparisons: raw B32/L16, raw B128/L128, mixed B32/L16,
and the old/new Policy evaluator at B128/chunk128. --systems-case selects one.
The same frozen --assets-root and this harness serve both package revisions.
Systems workers save matching retained payloads, five or more warmed samples,
compilation/load evidence and memory limits. The fake host method uses no network.
Use --recording-contracts for only Packet 3: batch 32, rollout 16 and five warm
samples of none/priority/full plus separate host save costs. Set PYTHONPATH to
the selected --package-root/src when comparing committed and current sources.
No mode establishes learning efficiency or proves theoretical optimality.
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
from jax.core import Tracer

from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MOVE_STAY,
    TEAM_A_ID,
    TEAM_B_ID,
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
    MetricMode,
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
    AgentClassName,
    make_standard_team_deathmatch_config,
)

type Tree = Any

_POLICIES: dict[str, object] = {
    "team_a": "tdm-alpha-exploration-10pct",
    "team_b": "tdm-beta-exploration-10pct",
}
_ROLLOUT_MODES = ("none", "priority", "full", "sparse_full", "replay")


def _exploratory_apply(name: str) -> PolicyApply:
    """Build a policy callable that selects a random legal action 10% of the time.

    The named controller always advances its carry; exploration changes only the
    selected action. Controller, exploration, and selection use separate keys.

    Parameters
    ----------
    name : str
        Registered base controller name. The exploration source is the existing
        random policy.

    Returns
    -------
    PolicyApply
        Policy callable with the standard variables, carry, actor input, mask and
        key interface.
    """
    controller, random = policy(name).apply, policy("random").apply

    def apply(
        variables: Tree, carry: Tree, actor: ActorInput, mask: ActionMask, key: Array
    ) -> tuple[Tree, Tree]:
        """Apply both action sources with separate keys and keep the controller
        carry.
        """
        selection_key, controller_key, random_key = jax.random.split(key, 3)
        action, next_carry = controller(variables, carry, actor, mask, controller_key)
        exploration, _ = random((), (), actor, mask, random_key)
        explore = jax.random.bernoulli(selection_key, 0.1)

        def choose(chosen: Array, usual: Array) -> Array:
            """Use the same exploration choice for every leaf of one actor action."""
            return jnp.where(explore, chosen, usual)

        return jax.tree.map(choose, exploration, action), next_carry

    return apply


class Facts(NamedTuple):
    """Retain pre-step state and mask with that step's info and reward.

    These full arrays exist only for the legacy metric comparison. A scan adds a
    leading time axis; native batches retain their environment axis beneath it.
    """

    state: EnvState
    mask: ActionMask
    info: Info
    reward: Reward


def _source_hashes() -> dict[str, str]:
    """Hash package Python files and this harness for the legacy local comparison."""
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src" / "marl_battlegrounds").rglob("*.py"))
    files.append(Path(__file__).resolve())
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _schedule(count: int, map_id: int | None = None) -> list[dict[str, object]]:
    """Assign positive episode and seed IDs while cycling the requested maps.

    Parameters
    ----------
    count : int
        Number of lanes; callers supply a positive batch size.
    map_id : int | None
        Fixed map ID, or None to cycle the canonical evaluation map IDs.

    Returns
    -------
    list[dict[str, object]]
        One episode/seed/map row per lane, in lane order.
    """
    maps = CANONICAL_TDM_EVALUATION_MAP_IDS if map_id is None else (map_id,)
    return [
        {"episode_id": lane + 1, "seed_id": lane + 1, "map_id": maps[lane % len(maps)]}
        for lane in range(count)
    ]


def _bytes(tree: object) -> int:
    """Count logical bytes in array leaves, without deduplicating shared buffers.

    Parameters
    ----------
    tree : object
        Tree whose leaves expose array shape and dtype, including shape-only
        objects.

    Returns
    -------
    int
        Logical byte sum; shared or aliased buffers are counted for each leaf.
    """
    return sum(
        math.prod(value.shape) * np.dtype(value.dtype).itemsize
        for value in jax.tree.leaves(tree)
    )


def _platforms(tree: object) -> list[str]:
    """List the device platform names that hold the supplied array leaves.

    Parameters
    ----------
    tree : object
        Tree of device arrays whose current placements should be reported.

    Returns
    -------
    list[str]
        Sorted distinct device platform names.
    """
    return sorted(
        {
            device.platform
            for value in jax.tree.leaves(tree)
            for device in value.devices()
        }
    )


def _benchmark_rosters() -> tuple[
    tuple[AgentClassName, ...], tuple[AgentClassName, ...]
]:
    """Use the task authority, including its old name for baseline comparisons."""
    from marl_battlegrounds import tasks

    discover = getattr(tasks, "canonical_tournament_rosters", None)
    if discover is not None:
        return discover()
    canonical = tasks._CANONICAL_ROSTER  # pyright: ignore[reportPrivateUsage]
    return canonical, canonical


def _configs(count: int, map_id: int | None = None) -> EnvConfig:
    """Build a native config batch for the legacy 300-step controller workload.

    A fixed map uses canonical rosters in every lane. The default mixed schedule
    cycles canonical maps and inserts one smaller roster pair every 16 lanes.

    Parameters
    ----------
    count : int
        Positive native environment count.
    map_id : int | None
        Fixed map ID, or None for the declared mixed-map and roster schedule.

    Returns
    -------
    EnvConfig
        Configuration tree with a leading environment axis on every leaf.
    """
    canonical_a, canonical_b = _benchmark_rosters()
    if map_id is not None:
        config = make_standard_team_deathmatch_config(
            map_id=map_id,
            team_a_roster=canonical_a,
            team_b_roster=canonical_b,
            max_steps=300,
        )

        def batch(value: Array | int | float | bool) -> Array:
            """Broadcast one config leaf across the requested number of environments."""
            return jnp.broadcast_to(value, (count, *jnp.shape(value)))

        return jax.tree.map(batch, config)
    maps = CANONICAL_TDM_EVALUATION_MAP_IDS
    configurations = [
        make_standard_team_deathmatch_config(
            map_id=maps[index % len(maps)],
            team_a_roster=("mage", "mage", "priest")
            if index % 16 == 15
            else canonical_a,
            team_b_roster=("warrior", "priest") if index % 16 == 15 else canonical_b,
            max_steps=300,
        )
        for index in range(count)
    ]
    return jax.tree.map(lambda *rows: jnp.stack(rows), *configurations)


def _actions(
    state: EnvironmentState, env: Environment, tick: Array, *, exploration: bool = True
) -> Action:
    """Apply both team controllers to current public inputs with reproducible keys.

    Parameters
    ----------
    state : EnvironmentState
        Current native state, including episode IDs and action masks.
    env : Environment
        Environment used to reconstruct current public observations.
    tick : Array
        Scalar integer decision index used in the deterministic action-key stream.
    exploration : bool
        Whether to select the random action source with probability 0.1; default
        True.

    Returns
    -------
    Action
        Joint action with leading environment axis, ready for the current masks.
    """
    first, second = (
        _exploratory_apply(name) if exploration else policy(name).apply
        for name in ("tdm-alpha", "tdm-beta")
    )
    actor_keys = _actor_keys(
        jax.random.key(42), state.episode_id, jnp.full_like(state.episode_id, tick)
    )

    def choose(observation: Observations, mask: ActionMask, keys: Array) -> Action:
        """Choose one environment's joint action using the existing policy adapter."""
        return apply_policies(first, second, (), (), (), (), observation, mask, keys)[0]

    return jax.vmap(choose)(env.get_observations(state), state.action_mask, actor_keys)


def _rollout(
    env: Environment, *, frozen_facts: bool = False, exploration: bool = True
) -> Callable[..., Any]:
    """Build the legacy 300-step scan, optionally retaining full transition facts.

    Ordinary mode saves completion data and requested replay packets. Fact mode
    uses the numerical step authority to collect the same inputs for later metric
    comparisons. This helper does not reset completed episodes.

    Parameters
    ----------
    env : Environment
        Execution handle whose static choices define this legacy scan.
    frozen_facts : bool
        False retains completion/capture outputs; True retains full per-step facts.
    exploration : bool
        Whether controller action selection includes the fixed 10% exploration
        rate.

    Returns
    -------
    Callable[..., Any]
        Callable from initial EnvironmentState to final state and selected retained
        outputs.
    """

    def run(initial: EnvironmentState) -> Tree:
        """Scan the fixed horizon from one native initial state and return saved
        outputs.
        """

        def transition(carry: Tree, tick: Array) -> tuple[Tree, Tree]:
            """Advance one tick and retain either its raw facts or its requested
            capture.
            """
            state, completion = carry
            actions = _actions(state, env, tick, exploration=exploration)
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
    """Recompute full and priority totals from the retained transition history.

    Initial and final states must belong to the supplied facts. The leading fact
    axes are time then environment. Return full totals and both final tables.

    Parameters
    ----------
    initial : EnvironmentState
        Initial native state for this exact retained trajectory.
    final : EnvironmentState
        Final native state from the same trajectory.
    facts : Facts
        Retained pre-step state, mask, info and reward with leading time and batch
        axes.

    Returns
    -------
    tuple[FullTotals, MetricValues, MetricValues]
        Full totals, priority values, and full metric values for each environment.
    """
    totals = jax.vmap(initialize_full)(initial.config, initial.core_state)

    def empty(_: Array) -> PriorityTotals:
        """Make one empty priority accumulator for each environment lane."""
        return initialize_priority()

    priority = jax.vmap(empty)(initial.episode_id)

    def collect(
        carry: tuple[FullTotals, PriorityTotals], step: Facts
    ) -> tuple[tuple[FullTotals, PriorityTotals], None]:
        """Add one batched transition to the two metric accumulators."""
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
    """Compile once, time a first call, then time synchronized warmed calls.

    Return the final output and timings in milliseconds, with compilation in
    seconds. Compiler memory describes this executable, not total process memory.

    Parameters
    ----------
    function : Callable[..., Any]
        Pure numerical callable to compile for the given arguments.
    args : tuple[Any, ...]
        Dynamic positional input trees for every measured call.
    repeats : int
        Number of warmed calls after the separately timed first execution.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        Final ready output and compilation, first-call, warmed-call and compiler-
        memory measurements.
    """
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
        "execution_count": 1 + repeats,
        "warm_median_ms": statistics.median(samples) if samples else None,
        "warm_min_ms": min(samples) if samples else None,
        "warm_max_ms": max(samples) if samples else None,
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
    """Require equal integer leaves and close floating leaves in matching trees.

    Floating comparisons use relative tolerance 3e-5 and absolute tolerance 0.002.
    A different leaf count or failed comparison raises an error.

    Parameters
    ----------
    first : Tree
        Expected array tree in the agreed comparison schema.
    second : Tree
        Observed tree with the same ordered leaves.

    Returns
    -------
    None
        None; a failed comparison raises an assertion or leaf-count error.
    """
    for left, right in zip(
        jax.tree.leaves(first), jax.tree.leaves(second), strict=True
    ):
        if np.issubdtype(left.dtype, np.integer) or left.dtype == np.bool_:
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, rtol=3e-5, atol=2e-3)


def _notify(message: str) -> None:
    """Print one progress message immediately, including when output is redirected.

    Parameters
    ----------
    message : str
        One human-readable progress line to flush immediately.

    Returns
    -------
    None
        None; writes the message to standard output.
    """
    print(message, flush=True)


def _cpu_model() -> str:
    """Read the Linux CPU model when available, otherwise use the platform fallback."""
    cpu_info = Path("/proc/cpuinfo")
    return (
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


def run_size(
    count: int,
    output: Path,
    repeats: int,
    rollout_modes: tuple[str, ...] = _ROLLOUT_MODES,
    *,
    metrics_only: bool = False,
    map_id: int | None = None,
) -> dict[str, object]:
    """Measure one legacy batch and save its rollout and metric comparisons.

    Parameters
    ----------
    count : int
        Number of native environment lanes; no smaller substitute is used.
    output : Path
        Directory for this batch's JSON record and requested replay files.
    repeats : int
        Number of warmed calls after the separately measured first call.
    rollout_modes : tuple[str, ...]
        Optional output modes to compare on the GPU.
    metrics_only : bool
        Reuse one fixed fact history for the CPU/GPU metric comparison.
    map_id : int | None
        Fixed canonical map for metrics-only mode; defaults to 48 there.

    Returns
    -------
    dict[str, object]
        Measurements, input identities, device facts, and comparison outcomes.

    Raises
    ------
    ValueError
        If a fixed map is requested outside metrics-only mode.
    MemoryError
        If the retained fact history exceeds the preflight GPU budget.
    AssertionError
        If a rollout or metric comparison changes required results.
    """
    if map_id is not None and not metrics_only:
        raise ValueError("--map-id requires --metrics-only")
    if metrics_only:
        map_id = 48 if map_id is None else map_id
        rollout_modes = ()
    policies: dict[str, object] = (
        {"team_a": "tdm-alpha", "team_b": "tdm-beta"} if metrics_only else _POLICIES
    )
    gpu, cpu = cast(Any, jax.devices("gpu")[0]), cast(Any, jax.devices("cpu")[0])
    cpu_model = _cpu_model()
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
            "controlled_effect_arithmetic": "float32 pairwise products and reductions",
            "jax_enable_x64": cast(bool, jax.config.read("jax_enable_x64")),
            "float_comparison_rtol": 3e-5,
            "float_comparison_atol": 0.002,
        },
        "source_file_sha256": _source_hashes(),
        "jax": jax.__version__,
        "repeats": repeats,
        "benchmark_mode": "metrics_only" if metrics_only else "rollouts_and_metrics",
        "rollout_modes": rollout_modes,
        "native_gpu_rollouts": {},
        "policies": policies,
        "map_ids": list(CANONICAL_TDM_EVALUATION_MAP_IDS)
        if map_id is None
        else [map_id],
        "exploration_probability": 0.0 if metrics_only else 0.1,
        "rosters": {
            "canonical_5v5": count if metrics_only else count - count // 16,
            "repeated_3v2": 0 if metrics_only else count // 16,
        },
    }
    output.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        """Write the latest legacy batch record so completed evidence survives a
        failure.
        """
        (output / f"batch-{count}.json").write_text(json.dumps(result, indent=2) + "\n")

    result["status"] = "running"
    save()
    with jax.default_device(gpu):
        configs = _configs(count, map_id)
        ids = jnp.arange(1, count + 1, dtype=jnp.int32)
        env = make("tdm", num_envs=count, metrics="none")
        reset_keys = episode_keys(jax.random.key(42), ids, jnp.zeros_like(ids), 0)

        def reset(config: EnvConfig) -> tuple[Observations, EnvironmentState]:
            """Reset the fixed legacy batch with its saved IDs and reset keys."""
            return env.reset(reset_keys, config, episode_id=ids)

        observations, initial = cast(
            tuple[Observations, EnvironmentState],
            jax.block_until_ready(
                cast(tuple[Observations, EnvironmentState], jax.jit(reset)(configs))
            ),
        )
        generate_facts = _rollout(env, frozen_facts=True, exploration=not metrics_only)
        shapes = jax.eval_shape(generate_facts, initial)
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
        _notify(f"B{count}: generating ALPHA/BETA episodes (up to 300 steps)")
        (final, facts), generation = _measure(
            generate_facts, (initial,), 0 if metrics_only else 1
        )
        result["fact_generation"] = generation
        result["episode_executions"] = count * int(generation["execution_count"])
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
        if not metrics_only and len(set(trace_hashes)) != count:
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
        result["full_accumulator_bytes"] = _bytes(gpu_values[0])
        result["full_result_bytes"] = _bytes(gpu_values[2])
        result["full_accumulator_bytes_per_episode"] = _bytes(gpu_values[0]) // count
        result["full_result_bytes_per_episode"] = _bytes(gpu_values[2]) // count
        # Replay inspection retains values and validity at every boundary; this
        # is a storage calculation, not memory allocated by this benchmark.
        result["estimated_full_prefix_bytes"] = (
            sum(int(length) + 1 for length in lengths) * _bytes(gpu_values[2]) // count
        )
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
            ids,
            jnp.ones(count, bool),
            outcomes,
            configs,
            priority,
            full,
            None,
            final.core_state.step_count - final.initial_step_count - 1,
            final.core_state.step_count - final.initial_step_count,
            final.core_state.team_deathmatch_scores,
            final.lifecycle_error,
        )
        started = time.perf_counter()
        with RunWriter(
            output / "csv",
            policies=policies,
            details={
                "seed": 42,
                "rng_protocol": "episode-fold-in-v1",
                "num_envs": count,
            },
        ) as writer:
            writer.register_episodes(_schedule(count, map_id))
            writer.write(completed)
            writer.flush()
        result["full_csv_write_and_fsync_ms"] = (time.perf_counter() - started) * 1000
        result["csv_run_dir"] = str(writer.run_dir)
        result["full_csv_bytes"] = writer.paths["full_metrics"].stat().st_size
        result["priority_csv_bytes"] = writer.paths["priority_metrics"].stat().st_size
        result["run_files_bytes"] = sum(
            path.stat().st_size for path in writer.run_dir.rglob("*") if path.is_file()
        )
        # Release the benchmark-only history before measuring streaming execution.
        del facts, cpu_args, cpu_values, gpu_values, host_values, final
        import gc

        gc.collect()
        rollouts: dict[str, object] = {}
        baseline: float | None = None
        for mode in rollout_modes:
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
                """Reset the selected output mode from the same configs, IDs, and
                keys.
                """
                return current.reset(reset_keys, config, episode_id=ids)

            _, state = cast(
                tuple[Observations, EnvironmentState], jax.jit(reset_current)(configs)
            )
            (end, (terminal, packets)), timing = _measure(
                _rollout(current), (state,), repeats
            )
            result["episode_executions"] += count * int(timing["execution_count"])
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
            if mode == "none":
                baseline = median
            timing["added_ms_vs_none"] = None if baseline is None else median - baseline
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
                    policies=policies,
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


class _FoundationCarry(NamedTuple):
    """Keep one final state and small totals while the benchmark scans its steps.

    Advances, completions and resets are scalar or [B]. Rewards are [10] or
    [B, 10]. The two metric totals consume completed priority/full values so that
    collection cannot disappear from the compiled program. Only the manual ID
    reference needs a scalar high-water mark. Replay history is returned separately.
    """

    state: EnvironmentState
    high_water: Array | None
    advances: Array
    completions: Array
    resets: Array
    rewards: Array
    metric_sums: Array
    metric_counts: Array


def _foundation_identity(package_root: Path, assets_root: Path) -> dict[str, Any]:
    """Return source/asset hashes and reject an unintended imported package.

    Both paths are resolved by the dispatcher. The source root must contain
    src/marl_battlegrounds; the asset root contains the TDM JSON resource tree.
    Hashing happens outside measured work and is repeated before accepting a case.

    Parameters
    ----------
    package_root : Path
        Resolved source checkout/export containing src/marl_battlegrounds.
    assets_root : Path
        Resolved frozen TDM resource directory containing the manifest and JSON
        assets.

    Returns
    -------
    dict[str, Any]
        Exact imported path and source, asset and harness SHA-256 records.

    Raises
    ------
    RuntimeError
        If Python imported a package outside the requested frozen source root.
    """
    import marl_battlegrounds

    actual = Path(marl_battlegrounds.__file__).resolve().parent
    expected = (package_root / "src" / "marl_battlegrounds").resolve()
    if actual != expected:
        raise RuntimeError(
            f"Imported {actual}; expected the requested package {expected}"
        )
    return {
        "imported_package_path": str(actual),
        "asset_path": str(assets_root),
        "package_sha256": {
            str(path.relative_to(actual)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(actual.rglob("*.py"))
        },
        "asset_sha256": {
            str(path.relative_to(assets_root)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(assets_root.rglob("*.json"))
        },
        "harness_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _foundation_assets(assets_root: Path) -> None:
    """Use one frozen resource tree in this disposable worker only.

    Factories and digest validation still run through the existing package loader.
    This changes no files and is never installed in the parent process.

    Parameters
    ----------
    assets_root : Path
        Frozen TDM resource directory to use for this disposable worker.

    Returns
    -------
    None
        None; replaces only this worker's resource reader and clears its resource
        caches.

    Raises
    ------
    ValueError
        If a requested resource escapes the frozen asset directory.
    """
    from marl_battlegrounds import _tdm_assets

    def read(relative_path: str) -> bytes:
        """Read one loader-requested resource without leaving the frozen directory."""
        path = (assets_root / relative_path).resolve()
        if not path.is_relative_to(assets_root):
            raise ValueError("Asset paths must stay inside the fixed input directory")
        return path.read_bytes()

    _tdm_assets._resource_bytes = read  # pyright: ignore[reportPrivateUsage]
    _tdm_assets.asset_manifest.cache_clear()
    _tdm_assets.map_id_aliases.cache_clear()


def _foundation_digest(tree: Tree) -> str:
    """Hash ordered leaf shapes, dtypes and bytes outside measured calls.

    Accept host or device arrays. Callers fix the tree schema; this digest compares
    its numerical payload. Device inputs are copied to the host for this check.

    Parameters
    ----------
    tree : Tree
        Host or device array tree whose ordered numerical payload should be hashed.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest of leaf shapes, dtypes and bytes.
    """
    digest = sha256()
    for value in jax.tree.leaves(jax.device_get(tree)):
        array = np.asarray(value)
        digest.update(json.dumps([array.shape, array.dtype.str]).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _foundation_array(value: Tree) -> Array:
    """Convert one benchmark input leaf with JAX's normal dtype rules.

    Parameters
    ----------
    value : Tree
        Scalar or array-like benchmark leaf to convert; normal JAX dtype rules
        apply.

    Returns
    -------
    Array
        JAX array corresponding to the supplied leaf.
    """
    return jnp.asarray(value)


def _foundation_balanced(
    source: EnvConfig, count: int | None, *, balance: bool = True
) -> EnvConfig:
    """Return the manual scalar/native preparation used for matched inputs.

    Source is one config or a full native source batch. None means exact scalar
    input; a native count broadcasts when needed and exchanges complete spawn
    banks in the second half. This numerical reference assumes validated inputs.
    Balance False keeps every source bank and only broadcasts the input.
    Only positive even counts claim balanced setup; raw odd batches remain usable.

    Parameters
    ----------
    source : EnvConfig
        One validated config or a complete config batch on the native leading axis.
    count : int | None
        Native batch count, or None for an exact scalar config. Balanced proof uses
        even counts.
    balance : bool
        True exchanges both spawn banks in the second half; False keeps source
        placements.

    Returns
    -------
    EnvConfig
        Prepared configuration tree, preserving scalar or requested native shape.
    """
    if count is None:
        return jax.tree.map(_foundation_array, source)

    def broadcast(value: Tree) -> Array:
        """Add the requested native axis to one unbatched config leaf."""
        array = jnp.asarray(value)
        return jnp.broadcast_to(array, (count, *array.shape))

    batch = (
        source
        if source.agent_profile.active_mask.ndim == 2
        else jax.tree.map(broadcast, source)
    )
    if not balance:
        return batch
    pads = batch.team_spawn_pad_positions
    swapped = jnp.flip(pads, axis=1)
    choice = (jnp.arange(count) >= count // 2).reshape(count, 1, 1, 1)
    return batch._replace(
        team_spawn_pad_positions=jnp.where(choice, swapped, pads),
    )


def _foundation_configs(source: EnvConfig, count: int | None) -> EnvConfig:
    """Prepare fixed benchmark configs with 4/7-step native episode horizons.

    Scalar episodes use 4 steps. Native horizons alternate by lane so that a scan
    contains both partial resets and continuing games. This is a declared timing
    input, not a training or tournament configuration recommendation.

    Parameters
    ----------
    source : EnvConfig
        Validated map and roster configuration shared by the compared routes.
    count : int | None
        Native batch size, or None for scalar execution.

    Returns
    -------
    EnvConfig
        Prepared config with the declared short episode horizons that force
        repeated partial resets.
    """
    prepared = _foundation_balanced(source, count)
    return prepared._replace(
        max_steps=jnp.asarray(4, jnp.int32)
        if count is None
        else jnp.where(jnp.arange(count) % 2 == 0, 4, 7).astype(jnp.int32)
    )


def _foundation_keys(root: Array, tick: Array, stream: int, count: int | None) -> Array:
    """Return one scalar key or [B] explicit lane keys from stable coordinates.

    Fold in stream, then round, then lane. Streams 0/1/2/3 cover initial reset,
    continuing reset, transition and action sampling. Explicit lane keys keep old
    and new APIs comparable without depending on their root-key convenience rules.

    Parameters
    ----------
    root : Array
        Typed root PRNG key shared by both compared routes.
    tick : Array
        Scalar integer round coordinate folded into the stream key.
    stream : int
        Fixed stream number: 0 initial reset, 1 partial reset, 2 step, 3 action
        sampling.
    count : int | None
        Native lane count for an array of keys, or None for one scalar key.

    Returns
    -------
    Array
        Typed scalar key or a typed key array shaped [count].
    """
    key = jax.random.fold_in(jax.random.fold_in(root, stream), tick)
    if count is None:
        return jax.random.fold_in(key, 0)
    return jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
        key, jnp.arange(count, dtype=jnp.uint32)
    )


def _foundation_environment(
    config: EnvConfig, count: int | None, api: str, mode: str
) -> Environment:
    """Build the selected raw/default-config route with fixed output capacities.

    Reference uses the old explicit-config reset route. Automatic stores the same
    config as constructor defaults. Sparse full metrics and replay select initial
    episode IDs only, so subsequent episodes also exercise disabled collection.

    Parameters
    ----------
    config : EnvConfig
        Exact resolved scalar or native config used by this case.
    count : int | None
        Native batch size, or None for a scalar environment.
    api : str
        reference for explicit configuration/IDs, or automatic for constructor
        defaults.
    mode : str
        Output case: none, priority, full, sparse_full, or replay.

    Returns
    -------
    Environment
        Execution handle for the exact API and output case.
    """
    options: dict[str, Any] = {
        "num_envs": count,
        "metrics": "full"
        if mode == "full"
        else "none"
        if mode == "none"
        else "priority",
        "full_metrics_episodes": tuple(range(1, (count or 1) + 1, 16))
        if mode == "sparse_full"
        else (),
        "replay_episodes": tuple(range(1, min(4, count or 1) + 1))
        if mode == "replay"
        else (),
    }
    if api == "automatic":
        options["env_config"] = config
    return make("tdm", **options)


def _foundation_initialize(api: str, count: int | None) -> Callable[..., Tree]:
    """Build a reusable reset callable taking env, resolved config and root key.

    The manual route supplies initial IDs 1..B, or scalar 1. The automatic route
    asks the public wrapper to assign those same IDs. Count fixes the batch shape.

    Parameters
    ----------
    api : str
        reference supplies explicit IDs/config; automatic uses public reset
        defaults.
    count : int | None
        Native batch size, or None for the scalar route.

    Returns
    -------
    Callable[..., Tree]
        Reset callable accepting an environment, resolved configuration and root
        key.
    """

    def initialize(env: Environment, config: EnvConfig, root: Array) -> Tree:
        """Reset matched inputs through the requested public API route."""
        keys = _foundation_keys(root, jnp.asarray(0, jnp.int32), 0, count)
        if api == "automatic":
            return env.reset(keys)
        ids = (
            jnp.asarray(1, jnp.int32)
            if count is None
            else jnp.arange(1, count + 1, dtype=jnp.int32)
        )
        return env.reset(keys, config, episode_id=ids)

    return initialize


def _foundation_rollout(
    api: str, count: int | None, length: int, action_workload: str = "fixed"
) -> Callable[..., Tree]:
    """Build a compiled explicit-reset loop with matched action/reset key streams.

    Return the final carry and optional selected replay packets for each round.
    Keep no full state history. Each round resets finished lanes before choosing
    its action and stepping; the manual ID ledger reserves a complete native block.
    Fixed and random-valid methods do not use observation tensors, so unused
    intermediate observation work may be removed. This is not learner timing.

    Parameters
    ----------
    api : str
        reference uses the manual ID/reset path; automatic uses public reset_done.
    count : int | None
        Native environment count, or None for scalar execution.
    length : int
        Fixed positive scan length; CLI foundations cases require at least eight
        rounds.
    action_workload : str
        fixed uses stay/no-target/no-Ultimate; sample uses the existing legal
        sampler.

    Returns
    -------
    Callable[..., Tree]
        Callable accepting an environment, initialized carry and root key; returns
        final carry and capture.
    """

    def run(env: Environment, initial: EnvironmentState, root: Array) -> Tree:
        """Run all requested rounds and retain the outputs needed for comparison."""
        shape = initial.episode_id.shape
        actions = Action(
            jnp.full((*shape, MAX_AGENT_SLOTS), MOVE_STAY, jnp.int32),
            jnp.zeros((*shape, MAX_AGENT_SLOTS), jnp.int32),
            jnp.zeros((*shape, MAX_AGENT_SLOTS), jnp.int32),
        )
        carry = _FoundationCarry(
            initial,
            jnp.asarray(count or 1, jnp.int32) if api == "reference" else None,
            jnp.zeros(shape, jnp.int32),
            jnp.zeros(shape, jnp.int32),
            jnp.zeros(shape, jnp.int32),
            jnp.zeros_like(actions.move, dtype=jnp.float32),
            jnp.zeros(2, jnp.float32),
            jnp.zeros(2, jnp.int32),
        )

        def advance(
            current: _FoundationCarry, tick: Array
        ) -> tuple[_FoundationCarry, Tree]:
            """Reset, act and step once while consuming completion/metric results."""
            state = current.state
            selected = state.done.done
            reset_keys = _foundation_keys(root, tick, 1, count)
            high_water = current.high_water
            if api == "automatic":
                _, state = env.reset_done(reset_keys, state)
            else:
                assert high_water is not None
                offsets = (
                    jnp.asarray(1, jnp.int32)
                    if count is None
                    else jnp.arange(1, count + 1, dtype=jnp.int32)
                )
                ids = jnp.where(selected, high_water + offsets, state.episode_id)
                _, state = env.reset(
                    reset_keys,
                    state.config,
                    episode_id=ids,
                    state=state,
                    reset_mask=selected,
                )
                high_water = high_water + jnp.where(jnp.any(selected), count or 1, 0)
            active = ~state.done.done
            submitted = (
                _foundation_sample(
                    env, _foundation_keys(root, tick, 3, count), state, count
                )
                if action_workload == "sample"
                else actions
            )
            _, state, reward, _, info = env.step(
                _foundation_keys(root, tick, 2, count),
                state,
                submitted,
            )
            sums: list[Array] = []
            counts: list[Array] = []
            for values in (info.priority, info.full):
                if values is None:
                    sums.append(jnp.asarray(0, jnp.float32))
                    counts.append(jnp.asarray(0, jnp.int32))
                else:
                    valid = values.valid & info.completed[..., None]
                    sums.append(jnp.sum(jnp.where(valid, values.values, 0)))
                    counts.append(jnp.sum(valid, dtype=jnp.int32))
            return _FoundationCarry(
                state,
                high_water,
                current.advances + active.astype(jnp.int32),
                current.completions + info.completed.astype(jnp.int32),
                current.resets + selected.astype(jnp.int32),
                current.rewards + reward.rewards,
                current.metric_sums + jnp.stack(sums),
                current.metric_counts + jnp.stack(counts),
            ), info.replay

        return jax.lax.scan(advance, carry, jnp.arange(length, dtype=jnp.int32))

    return run


def _foundation_sample(
    env: Environment, key: Array, state: EnvironmentState, count: int | None
) -> Action:
    """Return a legal joint Action through the public or archived adapter route.

    Keys are already scalar/per-lane. Both paths fold in global actor slots and
    call the existing joint-head random policy and team assembler. The fallback
    supports the frozen old package; it defines no new legality or routing rules.

    Parameters
    ----------
    env : Environment
        Execution handle whose public sampler is used when available.
    key : Array
        One scalar key or explicit per-lane typed keys, already derived for this
        round.
    state : EnvironmentState
        Current state and exact action masks after any selected reset.
    count : int | None
        Native lane count, or None to use the scalar actor adapter.

    Returns
    -------
    Action
        Legal joint Action with scalar or native batch shape.
    """
    sample = getattr(env, "sample_actions", None)
    if sample is not None:
        return sample(key, state)

    from marl_battlegrounds.policies.actor import (
        ActorAction,
        build_joint_action_from_actor_actions,
    )
    from marl_battlegrounds.policies.no_shared_obs import (
        execute_no_shared_obs_team_policy,
    )
    from marl_battlegrounds.policies.random_valid import random_policy

    def one(lane_key: Array, observation: Tree, mask: ActionMask) -> Action:
        """Apply the archived scalar actor adapters using the declared key order."""
        actor_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
            lane_key, jnp.arange(MAX_AGENT_SLOTS, dtype=jnp.uint32)
        )
        first = cast(
            ActorAction,
            execute_no_shared_obs_team_policy(
                observation, mask, actor_keys, random_policy, TEAM_A_ID
            ),
        )
        second = cast(
            ActorAction,
            execute_no_shared_obs_team_policy(
                observation, mask, actor_keys, random_policy, TEAM_B_ID
            ),
        )
        return build_joint_action_from_actor_actions(first, second)

    if count is None:
        return one(key, state.observation, state.action_mask)
    return jax.vmap(one)(key, state.observation, state.action_mask)


def _foundation_common(result: Tree) -> Tree:
    """Return fields shared by both wrapper schemas for trajectory comparison.

    Include final simulator/input/metric data, IDs, totals and captured packets.
    Exclude newly added lifecycle leaves and the benchmark's manual ID ledger;
    their public semantic tests are separate from this before/after comparison.

    Parameters
    ----------
    result : Tree
        Final benchmark carry and replay packets from either API route.

    Returns
    -------
    Tree
        Shared comparison fields, excluding new lifecycle leaves and the manual ID
        counter.
    """
    carry, replays = result
    state = carry.state
    return (
        state.config,
        state.core_state,
        state.observation,
        state.action_mask,
        state.done,
        state.episode_id,
        state.initial_step_count,
        state.collect_metrics,
        state.collect_full_metrics,
        state.collect_replay,
        state.priority,
        state.full,
        state.source_availability,
        carry.advances,
        carry.completions,
        carry.resets,
        carry.rewards,
        carry.metric_sums,
        carry.metric_counts,
        replays,
    )


def _foundation_measure(
    function: Callable[..., Tree],
    args: tuple[Tree, ...],
    repeats: int,
    probes: tuple[tuple[Tree, ...], ...] = (),
    *,
    execute_probes: bool = True,
) -> tuple[Tree, dict[str, Any], list[Tree]]:
    """Return the ready result, timing/memory report and executed probe results.

    Args are dynamic trees except an archived unregistered Environment. Compile
    once, time the first dispatch separately, then synchronize each warmed call.
    Guard warmed device-to-host transfers and inspect lowered code for callbacks.
    Large rollout probes only lower changed values; smaller cases execute them.
    Trace/cache counts describe this callable, not every nested backend program.

    Parameters
    ----------
    function : Callable[..., Tree]
        Numerical workflow callable to lower and compile once.
    args : tuple[Tree, ...]
        Initial positional array trees, plus an environment handle where required.
    repeats : int
        Number of warmed calls after first dispatch; each result is synchronized.
    probes : tuple[tuple[Tree, ...], ...]
        Same-schema changed-value argument tuples; default empty.
    execute_probes : bool
        True executes changed-value probes; False only lowers them to check trace
        reuse.

    Returns
    -------
    tuple[Tree, dict[str, Any], list[Tree]]
        Ready final output, timing/memory/guard report, and results of executed
        probes.

    Raises
    ------
    AssertionError
        If lowered numerical code contains a host callback. The JAX transfer guard also
        rejects device-to-host reads during warmed calls.
    """
    traces = 0

    def observed(*inputs: Tree) -> Tree:
        """Count Python tracing while keeping numerical inputs unchanged."""
        nonlocal traces
        traces += 1
        return function(*inputs)

    # The old Environment is an unregistered static dataclass. The candidate is
    # a PyTree whose changing defaults are ordinary numerical input leaves.
    static_env = any(leaf is args[0] for leaf in jax.tree.leaves(args[0]))
    compiled = cast(Any, jax.jit(observed, static_argnums=(0,) if static_env else ()))
    started = time.perf_counter()
    lowered = compiled.lower(*args)
    executable = lowered.compile()
    compilation = time.perf_counter() - started
    lowered_text = lowered.as_text()
    callback_markers = (
        "xla_ffi_python",
        "python_callback",
        "host_callback",
        "io_callback",
        "debug_callback",
    )
    if any(marker in lowered_text for marker in callback_markers):
        raise AssertionError("The numerical program contains a host callback")
    started = time.perf_counter()
    result = cast(Tree, jax.block_until_ready(compiled(*args)))
    first_ms = (time.perf_counter() - started) * 1000
    samples: list[float] = []
    with jax.transfer_guard_device_to_host("disallow_explicit"):
        for _ in range(repeats):
            started = time.perf_counter()
            result = cast(Tree, jax.block_until_ready(compiled(*args)))
            samples.append((time.perf_counter() - started) * 1000)
    before_probes = traces
    probe_results: list[Tree] = []
    for inputs in probes:
        if execute_probes:
            probe_results.append(cast(Tree, jax.block_until_ready(compiled(*inputs))))
        else:
            compiled.lower(*inputs)
    memory = executable.memory_analysis()
    cache_size = getattr(compiled, "_cache_size", None)
    return (
        result,
        {
            "compilation_seconds": compilation,
            "first_dispatch_ms": first_ms,
            "warm_samples_ms": samples,
            "warm_median_ms": statistics.median(samples),
            "warm_device_to_host_guard": "PASS: disallow_explicit",
            "host_callback_inspection": "PASS: no callback marker in lowered program",
            "lowered_program_sha256": sha256(lowered_text.encode()).hexdigest(),
            "outer_trace_count": traces,
            "new_traces_for_changed_values": traces - before_probes,
            "outer_dispatch_cache_entries": cache_size()
            if callable(cache_size)
            else None,
            "changed_value_probe": "executed"
            if execute_probes
            else "lowered only; numerical agreement is checked in small cases",
            "environment_argument": "static legacy descriptor"
            if static_env
            else "dynamic PyTree",
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
            "compile_count_limit": (
                "Trace/cache counts describe the outer workflow, "
                "not every nested backend executable"
            ),
        },
        probe_results,
    )


def _foundation_host_measure(
    function: Callable[[], Tree], repeats: int
) -> tuple[Tree, dict[str, Any]]:
    """Return the last setup result and first/warmed synchronized call times.

    Function takes no arguments and returns an array tree or execution handle.
    Host validation and any necessary transfers belong to this setup timing.
    These calls are outside the numerical transfer guard and contain no game loop.

    Parameters
    ----------
    function : Callable[[], Tree]
        No-argument setup callable returning arrays or an execution handle.
    repeats : int
        Number of warmed setup calls after the separately measured first call.

    Returns
    -------
    tuple[Tree, dict[str, Any]]
        Last setup result and first/warmed wall times in milliseconds.
    """
    started = time.perf_counter()
    result = cast(Tree, jax.block_until_ready(function()))
    first = (time.perf_counter() - started) * 1000
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = cast(Tree, jax.block_until_ready(function()))
        samples.append((time.perf_counter() - started) * 1000)
    return result, {
        "first_call_ms": first,
        "warm_samples_ms": samples,
        "warm_median_ms": statistics.median(samples),
    }


def _foundation_reference_setup(
    source: EnvConfig, count: int, *, balance: bool = True
) -> EnvConfig:
    """Validate distinct source rows once, then use the explicit array reference.

    This benchmark-only fallback gives the archived API a fair setup comparison.
    Snapshot the input once, deduplicate exact rows and validate source banks with
    Core. Balance True also validates the exchanged banks. Use CPU vectors when
    available. Traced numerical preparation assumes that this host validation already
    succeeded.

    Parameters
    ----------
    source : EnvConfig
        Scalar config or native source batch; concrete rows receive host
        validation.
    count : int
        Positive native batch count. Balanced cases use an even count.
    balance : bool
        Whether exchanged spawn banks must also be validated and prepared; default
        True.

    Returns
    -------
    EnvConfig
        Prepared batch after concrete host validation, or numerical preparation for
        traced inputs.

    Raises
    ------
    TypeError
        If validation changes a source vector dtype. The shared Core validator reports
        invalid concrete configurations.
    """
    from marl_battlegrounds.core.config import validate_env_config

    if not any(isinstance(value, Tracer) for value in jax.tree.leaves(source)):

        def host_array(value: Tree) -> Tree:
            """Keep a transferred leaf's shape, dtype and value on the host."""
            return np.asarray(value)

        host = jax.tree.map(host_array, jax.device_get(source))
        try:
            validation_device = cast(Any, jax.local_devices(backend="cpu")[0])
        except RuntimeError:
            validation_device = cast(Any, jax.local_devices()[0])

        def validation_value(value: Tree) -> Tree:
            """Adapt a host leaf to Core's existing scalar/vector type contract."""
            if value.ndim == 0:
                return value.item()
            array = jax.device_put(value, validation_device)
            if array.dtype != value.dtype:
                raise TypeError("Reference config dtype changed during validation")
            return array

        native = source.agent_profile.active_mask.ndim == 2
        checked: set[Tree] = set()
        for index in range(count if native else 1):

            def take(value: Tree, index: int = index) -> Tree:
                """Select one full source row from the single host snapshot."""
                return value[index]

            row = jax.tree.map(take, host) if native else host
            identity = tuple(
                (value.dtype.str, value.shape, value.tobytes())
                for value in jax.tree.leaves(row)
            )
            if identity in checked:
                continue
            checked.add(identity)
            with jax.ensure_compile_time_eval(), jax.default_device(validation_device):
                row = jax.tree.map(validation_value, row)
                validate_env_config(row)
                if balance:
                    validate_env_config(
                        row._replace(
                            team_spawn_pad_positions=row.team_spawn_pad_positions[::-1]
                        )
                    )
    return _foundation_balanced(source, count, balance=balance)


def _foundation_setup(
    args: argparse.Namespace, source: EnvConfig, count: int
) -> dict[str, Any]:
    """Measure preparation and balanced/exact map constructors without stepping.

    Count is a positive even native size. Compare one scalar source and a native
    batch with two distinct source rows. Check every prepared leaf, changed-value
    execution/reuse and the constructor's public reset config. Return separate
    host setup and numerical preparation reports, with logical array byte counts.

    Parameters
    ----------
    args : argparse.Namespace
        Validated worker options, including map ID, repeat count and output
        choices.
    source : EnvConfig
        Scalar validated source configuration used for both preparation input
        forms.
    count : int
        Positive even native size used by this setup-only case.

    Returns
    -------
    dict[str, Any]
        Separate preparation and map-construction measurements, input/output
        identities and limits.

    Raises
    ------
    AssertionError
        If prepared values differ from the manual reference or changed source values
        cause another trace.
    """
    from marl_battlegrounds import tasks

    helper = getattr(tasks, "balanced_spawn_configs", None)

    def prepare(config: EnvConfig) -> EnvConfig:
        """Use the candidate's public helper or the archived manual reference."""
        if helper is not None:
            return helper(config, num_envs=count)
        return _foundation_reference_setup(config, count)

    def broadcast(value: Tree) -> Array:
        """Create a full selected-source input batch without a swapped source bank."""
        array = jnp.asarray(value)
        return jnp.broadcast_to(array, (count, *array.shape))

    native = jax.tree.map(broadcast, source)
    native = native._replace(
        max_steps=jnp.asarray(source.max_steps, jnp.int32)
        + (jnp.arange(count, dtype=jnp.int32) % 2)
    )
    measurements = {}
    for name, inputs in (("scalar_source", source), ("native_two_sources", native)):
        expected = _foundation_balanced(inputs, count)
        prepared, host = _foundation_host_measure(
            lambda inputs=inputs: prepare(inputs), args.repeats
        )
        _assert_equal(expected, prepared)
        changed = inputs._replace(max_steps=jnp.asarray(inputs.max_steps) + 1)
        numerical, compiled, probes = _foundation_measure(
            prepare, (inputs,), args.repeats, ((changed,),)
        )
        _assert_equal(expected, numerical)
        _assert_equal(_foundation_balanced(changed, count), probes[0])
        if compiled["new_traces_for_changed_values"]:
            raise AssertionError(
                "Changed source values caused another preparation trace"
            )
        compiled.pop("environment_argument")
        measurements[name] = {
            "input_sha256": _foundation_digest(inputs),
            "output_sha256": _foundation_digest(prepared),
            "input_bytes": _bytes(jax.tree.map(_foundation_array, inputs)),
            "output_bytes": _bytes(prepared),
            "host_setup": host,
            "compiled_preparation": compiled,
        }

    def construct(balance: bool) -> tuple[Environment, EnvConfig | None]:
        """Build ready defaults or the archived equivalent handle/config pair."""
        if helper is not None:
            return make(
                "tdm",
                map_id=args.map_id,
                num_envs=count,
                balance_spawn_locations=balance,
                metrics="none",
                max_steps=7,
            ), None
        rosters = _benchmark_rosters()
        manual_source = make_standard_team_deathmatch_config(
            map_id=args.map_id,
            team_a_roster=rosters[0],
            team_b_roster=rosters[1],
            max_steps=7,
        )
        return make("tdm", num_envs=count, metrics="none"), _foundation_reference_setup(
            manual_source, count, balance=balance
        )

    constructors = {}
    for name, balance in (("balanced", True), ("exact_source", False)):
        (env, manual_config), constructor = _foundation_host_measure(
            lambda balance=balance: construct(balance), args.repeats
        )
        keys = _foundation_keys(jax.random.key(42), jnp.asarray(0, jnp.int32), 0, count)
        _, state = (
            env.reset(keys)
            if manual_config is None
            else env.reset(
                keys,
                manual_config,
                episode_id=jnp.arange(1, count + 1, dtype=jnp.int32),
            )
        )
        _assert_equal(
            _foundation_balanced(source, count, balance=balance), state.config
        )
        constructors[name] = constructor
    return {
        "setup_route": "public balanced_spawn_configs and map constructor"
        if helper is not None
        else "archived API: manual Core validation, bank exchange and raw constructor",
        "setup_comparison": measurements,
        "map_constructors": constructors,
        "setup_limits": (
            "Full rows are checked; compiled preparation uses a validated source pool. "
            "Only one source bank and the required prepared batch are inputs/outputs. "
            "Compiler memory reports temporary arrays; no rollout history is retained. "
            "The manual reference snapshots the input once, deduplicates exact rows, "
            "and validates both requested choices once per distinct source."
        ),
    }


def _run_foundations_worker(args: argparse.Namespace, output_path: Path) -> None:
    """Measure one declared workload and save source-bound results or its failure.

    The dispatcher supplies one batch, length and output mode per fresh process.
    Freeze resource loading, compare common results, measure ready output transfer
    separately and record memory before the automatic-only manual check. CPU runs
    are diagnostics; GPU qualification requires a quiet, separately scheduled run.

    Parameters
    ----------
    args : argparse.Namespace
        Worker options naming one API, backend, shape, length, mode and frozen
        roots.
    output_path : Path
        JSON evidence path updated as the case runs, including on failure.

    Returns
    -------
    None
        None; saves case evidence and raises on a failed check.

    Raises
    ------
    RuntimeError
        If source, harness or asset bytes change while the case is running.
    AssertionError
        If trajectory equality, reset coverage, or compilation-reuse checks fail.
    """
    count = None if args.worker == 0 else args.worker
    length = args.lengths[0]
    mode = args.rollout_modes[0]
    package_root = args.package_root.resolve()
    assets_root = args.assets_root.resolve()
    before = _foundation_identity(package_root, assets_root)
    _foundation_assets(assets_root)
    result: dict[str, Any] = {
        "status": "running",
        "benchmark_mode": "foundations",
        "qualification_scope": "GPU efficiency"
        if args.backend == "gpu"
        else "CPU diagnostic only; not acceptance evidence",
        "api": args.api,
        "backend": args.backend,
        "num_envs": count,
        "rollout_length": length,
        "metrics_mode": mode,
        "map_id": args.map_id,
        "source_revision": args.source_revision,
        "identity": before,
        "jax": jax.__version__,
        "python": sys.version,
        "host_platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "worker_pid": os.getpid(),
        "repeats": args.repeats,
        "action_workload": (
            "Public legal sampler; archived API uses existing actor adapters"
        )
        if args.action_workload == "sample"
        else (
            "Fixed stay/no-target/no-Ultimate joint action; "
            "no policy or action-sampling cost"
        ),
        "random_streams": (
            "fold_in(root, stream), then round, then lane; "
            "streams 0 initial reset, 1 reset_done, 2 step, 3 action sampling; "
            "sampling then folds in each global actor slot"
        ),
        "scope": (
            "Raw setup/reset/step and selected metric/capture production; "
            "no writer, provider, learner or geometry qualification"
        ),
    }

    def save() -> None:
        """Preserve the current result, including failures, in this case's JSON file."""
        output_path.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        device = cast(Any, jax.devices(args.backend)[0])
        result.update(
            device=str(device),
            device_kind=device.device_kind,
            runtime=str(device.client.platform_version),
        )
        with jax.default_device(device):
            roster_a, roster_b = _benchmark_rosters()
            result["rosters"] = {"team_a": roster_a, "team_b": roster_b}
            started = time.perf_counter()
            source = make_standard_team_deathmatch_config(
                map_id=args.map_id,
                team_a_roster=roster_a,
                team_b_roster=roster_b,
                max_steps=7,
            )
            jax.block_until_ready(source)
            result["source_factory_ms"] = (time.perf_counter() - started) * 1000
            if args.setup_only:
                assert count is not None
                result.update(_foundation_setup(args, source, count))
                result["device_memory_stats"] = device.memory_stats()
                result["process_peak_ram_bytes"] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                )
                if before != _foundation_identity(package_root, assets_root):
                    raise RuntimeError("Setup source, harness or assets changed")
                result["status"] = "complete"
                return
            started = time.perf_counter()
            config = cast(
                EnvConfig, jax.block_until_ready(_foundation_configs(source, count))
            )
            result["manual_preparation_ms"] = (time.perf_counter() - started) * 1000
            result["resolved_config_sha256"] = _foundation_digest(config)
            result["resolved_config_bytes"] = _bytes(config)
            started = time.perf_counter()
            env = _foundation_environment(config, count, args.api, mode)
            jax.block_until_ready(env)
            result["environment_construction_ms"] = (
                time.perf_counter() - started
            ) * 1000
            changed = config._replace(max_steps=jnp.asarray(config.max_steps) + 1)
            other_env = _foundation_environment(changed, count, args.api, mode)
            root = jax.random.key(42)
            other_root = jax.random.key(43)
            (_, state), initialization, initialized = _foundation_measure(
                _foundation_initialize(args.api, count),
                (env, config, root),
                args.repeats,
                ((other_env, changed, other_root),),
            )
            _assert_equal(
                jax.tree.map(_foundation_array, changed), initialized[0][1].config
            )
            result["initialization"] = initialization
            (carry, replays), timing, _ = _foundation_measure(
                _foundation_rollout(args.api, count, length, args.action_workload),
                (env, state, root),
                args.repeats,
                ((other_env, initialized[0][1], other_root),),
                execute_probes=(count or 1) <= 2,
            )
            if (
                initialization["new_traces_for_changed_values"]
                or timing["new_traces_for_changed_values"]
            ):
                raise AssertionError(
                    "Same-shaped changed values caused a new workflow trace"
                )
            observed = (carry, replays)
            result["rollout"] = timing
            started = time.perf_counter()
            host_observed = jax.device_get(observed)
            result["final_output_transfer"] = {
                "device_get_ms": (time.perf_counter() - started) * 1000,
                "logical_bytes": _bytes(observed),
                "capture_logical_bytes": _bytes(replays),
                "scope": (
                    "Already-ready benchmark result: final carry and selected capture. "
                    "Outside warmed execution; not a per-step library transfer. "
                    "Logical bytes may include buffers JAX can reuse on the host."
                ),
            }
            result["common_result_sha256"] = _foundation_digest(
                _foundation_common(host_observed)
            )
            result["initial_state_bytes"] = _bytes(state)
            result["output_bytes"] = _bytes(observed)
            result["capture_output_bytes"] = _bytes(replays)
            np.testing.assert_array_equal(
                np.asarray(carry.advances), np.full(carry.advances.shape, length)
            )
            if not bool(np.all(np.asarray(carry.resets) > 0)):
                raise AssertionError("The workload did not exercise repeated resets")
            result["advances"] = np.asarray(carry.advances).tolist()
            result["completed_episodes"] = int(np.asarray(carry.completions).sum())
            result["lane_resets"] = int(np.asarray(carry.resets).sum())
            result["transitions_per_second"] = (
                (count or 1) * length * 1000 / timing["warm_median_ms"]
            )
            result["device_memory_stats"] = device.memory_stats()
            result["process_peak_ram_bytes"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
            if args.api == "automatic":
                manual = _foundation_environment(config, count, "reference", mode)
                _, manual_state = _foundation_initialize("reference", count)(
                    manual, config, root
                )
                expected, reference_timing, _ = _foundation_measure(
                    _foundation_rollout(
                        "reference", count, length, args.action_workload
                    ),
                    (manual, manual_state, root),
                    2,
                )
                result["manual_reference_timing"] = reference_timing
                result["manual_reference_repeats"] = 2
                _assert_equal(
                    _foundation_common(expected), _foundation_common(observed)
                )
                result["manual_reference_agreement"] = (
                    "PASS: all common final state, accounting and captured arrays"
                )
                result["after_reference_check_memory"] = {
                    "device_memory_stats": device.memory_stats(),
                    "process_peak_ram_bytes": (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                    ),
                    "scope": "Includes the extra correctness executable and output",
                }
        result["memory_limit"] = (
            "Process/device peaks include setup, compilation, reuse probes and the "
            "final host copy, before the optional manual check. "
            "Compiler memory is per executable."
        )
        after = _foundation_identity(package_root, assets_root)
        if before != after:
            result["identity_after"] = after
            raise RuntimeError(
                "Package, harness or fixed assets changed during this measurement"
            )
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


def _run_foundations(args: argparse.Namespace) -> int:
    """Run isolated foundations workers with allowed GPU batch defaults.

    GPU is the foundations default; cuda,cpu permits efficient host setup checks.
    An explicit cuda-only platform list supports a separate compatibility check.
    Each worker verifies the imported package and fixed assets, and failures never
    substitute a smaller batch. Setup probes use positive even native sizes only.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed foundations options; this function validates combinations and
        dispatches workers.

    Returns
    -------
    int
        Zero when all workers pass, otherwise one. Each worker uses a fresh
        process.

    Raises
    ------
    ValueError
        If requested sizes, lengths, repeats or mode combinations are unsupported.
    """
    package_root = (args.package_root or Path(__file__).resolve().parents[2]).resolve()
    assets_root = (
        args.assets_root or package_root / "src/marl_battlegrounds/data/tdm"
    ).resolve()
    args.package_root, args.assets_root = package_root, assets_root
    args.backend = args.backend or "gpu"
    args.api = args.api or "reference"
    args.action_workload = args.action_workload or "fixed"
    args.lengths = args.lengths or [16, 128]
    args.sizes = args.sizes or [32, 128, 1024]
    if args.backend == "gpu" and (
        args.include_scalar
        or any(size not in _GPU_BATCHES for size in args.sizes)
        or (args.worker is not None and args.worker not in _GPU_BATCHES)
    ):
        raise ValueError(
            "GPU environment batches must be 32, 64, 128, 512 or 1024; "
            "scalar and odd checks belong on CPU"
        )
    args.rollout_modes = args.rollout_modes or ["none", "priority"]
    args.map_id = 12 if args.map_id is None else args.map_id
    if (
        args.metrics_only
        or args.repeats < 2
        or any(size < 1 for size in args.sizes)
        or (not args.setup_only and any(length < 8 for length in args.lengths))
    ):
        raise ValueError(
            "Foundations needs positive sizes, lengths of at least 8, "
            "two repeats, and no --metrics-only"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    workloads = (
        [args.worker]
        if args.worker is not None
        else ([0] if args.include_scalar else []) + args.sizes
    )
    if args.setup_only:
        if args.action_workload != "fixed" or any(
            count <= 0 or count % 2 for count in workloads
        ):
            raise ValueError(
                "Setup probes need positive even native sizes and no sampler"
            )
        args.lengths, args.rollout_modes = [0], ["none"]
    gpu_platforms = os.environ.get("JAX_PLATFORMS", "cuda,cpu")
    if not {"cuda", "gpu"}.intersection(gpu_platforms.split(",")):
        gpu_platforms = "cuda,cpu"
    failed = False
    for count in workloads:
        for length in args.lengths:
            for mode in args.rollout_modes:
                label = "scalar" if count == 0 else f"b{count}"
                name = f"foundations-{args.api}-{args.backend}-{label}-l{length}-{mode}"
                if args.setup_only:
                    name = f"foundations-setup-{args.backend}-{label}"
                elif args.action_workload == "sample":
                    name += "-sample"
                path = args.output / f"{name}.json"
                if args.worker is not None:
                    _run_foundations_worker(args, path)
                    return 0
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--foundations",
                    "--worker",
                    str(count),
                    "--api",
                    args.api,
                    "--backend",
                    args.backend,
                    "--lengths",
                    str(length),
                    "--rollout-modes",
                    mode,
                    "--action-workload",
                    args.action_workload,
                    "--repeats",
                    str(args.repeats),
                    "--package-root",
                    str(package_root),
                    "--assets-root",
                    str(assets_root),
                    "--map-id",
                    str(args.map_id),
                    "--output",
                    str(args.output.resolve()),
                ]
                if args.setup_only:
                    command.append("--setup-only")
                if args.source_revision:
                    command.extend(("--source-revision", args.source_revision))
                path.write_text(
                    json.dumps(
                        {
                            "status": "starting",
                            "api": args.api,
                            "backend": args.backend,
                            "num_envs": None if count == 0 else count,
                            "rollout_length": length,
                            "metrics_mode": mode,
                        }
                    )
                    + "\n"
                )
                _notify(
                    f"Foundations {args.api}/{args.backend}/{label}/{length}/{mode}"
                )
                process = subprocess.run(
                    command,
                    check=False,
                    cwd=package_root,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(package_root / "src"),
                        "JAX_PLATFORMS": "cpu"
                        if args.backend == "cpu"
                        else gpu_platforms,
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    },
                )
                if process.returncode:
                    record = json.loads(path.read_text())
                    record.update(status="failed", exit_code=process.returncode)
                    path.write_text(json.dumps(record, indent=2) + "\n")
                failed |= process.returncode != 0
    return int(failed)


_SYSTEM_CASES = {
    "raw-short": (32, 16),
    "raw-long": (128, 128),
    "mixed": (32, 16),
    "evaluator": (128, 128),
}
_GPU_BATCHES = (32, 64, 128, 512, 1024)


class _SystemsCarry(NamedTuple):
    """Retain final raw state, method memory and real-transition totals.

    Observations and state have native leading B axes. Memory is SystemState;
    advances/completions are int32 [B], and rewards are float32 [B,10]. Full
    per-decision actions and learning values are returned separately by the scan.
    """

    observations: Tree
    state: Tree
    memory: Tree
    advances: Array
    completions: Array
    rewards: Array


def _systems_numeric_tree(tree: Tree) -> Tree:
    """Replace typed PRNG-key leaves by uint32 data for size and digest reports.

    tree is any retained System/array tree. Return the same structure with typed
    key leaves replaced by key_data; other leaves are untouched. This helper does
    not transfer arrays and runs outside measured work. It keeps saved NPZ files
    independent of JAX's typed-key object representation.
    """

    def numeric(value: Tree) -> Tree:
        """Expose key bits only when this leaf has JAX's typed PRNG-key dtype."""
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
            return jax.random.key_data(value)
        return value

    return jax.tree.map(numeric, tree)


def _systems_method(
    variables: Tree, memory: Tree, inputs: Tree, keys: Tree, *, host: bool = False
) -> Tree:
    """Apply the same small recurrent method with JAX or NumPy array operations.

    Parameters
    ----------
    variables : tuple of arrays
        Projection weights [58,8] and action weights [8,31], both float32.
    memory : array
        Float32 [B,5,8] previous actor memories. Actors never share memories.
    inputs : SystemInput
        Recipient-authorized native inputs, masks and validity from the public API.
    keys : array
        Typed JAX keys [B], or already-read host uint32 key data [B,2]. The
        fake provider owns the required key read. A deterministic key
        offset influences each actor's numerical work; this is not a trained model.
    host : bool, default=False
        Use NumPy for the fake provider when True. No network or credentials.

    Returns
    -------
    SystemOutput
        Legal ActorAction int32 [B,5], fresh [B,5,8] memory, and float32 value
        [B,5] learning outputs. No actor may read another recipient's inputs.

    Notes
    -----
    This fixed method consumes self features and every permitted source-bank
    field. It chooses movement and the coupled target/Ultimate category. It is
    a reproducible workload, not a learning or tactical-competence claim. Host
    returns own their numerical buffers; later calls cannot overwrite them.
    """
    from marl_battlegrounds.evaluation.policy_execution import SystemOutput
    from marl_battlegrounds.policies.actor import ActorAction

    xp: Any = np if host else jnp
    data = keys if host else jax.random.key_data(keys)
    features = inputs.actors.observation.self_features
    summary = xp.zeros(features.shape[:2], dtype=xp.float32)
    for leaf in jax.tree.leaves(inputs.actors.source_bank):
        summary = summary + xp.mean(
            xp.asarray(leaf, dtype=xp.float32), axis=tuple(range(2, leaf.ndim))
        )
    noise = xp.asarray(data[:, 0] % xp.uint32(97), dtype=xp.float32)[
        :, None, None
    ] * xp.float32(0.0001)
    proposed = xp.tanh(
        features @ variables[0] * xp.float32(0.01)
        + memory * xp.float32(0.5)
        + summary[..., None] * xp.float32(0.001)
        + noise
    )
    valid = inputs.valid[:, None] & inputs.active_mask
    following = xp.where(valid[..., None], proposed, memory)
    logits = following @ variables[1]
    move = xp.argmax(
        xp.where(inputs.action_mask.move_mask, logits[..., :9], -xp.inf), axis=-1
    ).astype(xp.int32)
    joint_mask = inputs.action_mask.select_target_use_ultimate_joint_mask.reshape(
        (*valid.shape, 22)
    )
    joint = xp.argmax(xp.where(joint_mask, logits[..., 9:], -xp.inf), axis=-1).astype(
        xp.int32
    )
    actions = ActorAction(
        xp.where(valid, move, xp.int32(0)),
        xp.where(valid, joint // 2, xp.int32(0)),
        xp.where(valid, joint % 2, xp.int32(0)),
    )
    return SystemOutput(
        actions, following, learning_outputs={"value": xp.mean(following, axis=-1)}
    )


def _systems_initial_memory(variables: Tree, inputs: Tree, keys: Tree) -> Tree:
    """Create float32 [B,5,8] zero memory without choosing an action.

    variables and keys follow System.init but are unused by this fixed workload.
    inputs supplies the native batch shape. Return device memory; the host
    factory supplies its own NumPy initializer and reset hook instead.
    """
    del variables, keys
    return jnp.zeros((*inputs.active_mask.shape, 8), jnp.float32)


def _systems_pair(*, mixed: bool = False) -> tuple[Tree, Tree, dict[str, Any]]:
    """Build stable JAX methods, optionally replacing Team A with a fake provider.

    Parameters
    ----------
    mixed : bool, default=False
        True returns one NumPy host method and one JAX method. False returns two
        JAX methods. Both use identical immutable numerical weight values.

    Returns
    -------
    tuple
        Team A System, Team B System, and mutable benchmark-only host counters.
        Counters measure delivered logical payload, provider compute and calls;
        they do not claim to count every runtime transfer or allocation.
    """
    from marl_battlegrounds.evaluation.policy_execution import System

    weights = (
        jnp.sin(jnp.arange(58 * 8, dtype=jnp.float32)).reshape(58, 8) * 0.1,
        jnp.cos(jnp.arange(8 * 31, dtype=jnp.float32)).reshape(8, 31) * 0.1,
    )
    second = System(
        "benchmark-jax", _systems_method, weights, init=_systems_initial_memory
    )
    counts: dict[str, Any] = {
        "apply_calls": 0,
        "input_logical_bytes": 0,
        "output_logical_bytes": 0,
        "provider_seconds": 0.0,
        "key_read_seconds": 0.0,
    }
    if not mixed:
        return second, second, counts
    host_weights = jax.device_get(weights)

    def initialize(variables: Tree, inputs: Tree, keys: Tree) -> Tree:
        """Return new NumPy [B,5,8] zeros; no action or provider request is made."""
        del variables, keys
        return cast(Tree, np.zeros((*inputs.active_mask.shape, 8), np.float32))

    def reset(memory: Tree, fresh: Tree, selected: Tree) -> Tree:
        """Replace only selected [B] rows of the host [B,5,8] memory array."""
        return cast(Tree, np.where(np.asarray(selected)[:, None, None], fresh, memory))

    def provider(variables: Tree, memory: Tree, inputs: Tree, keys: Tree) -> Tree:
        """Count one batch request and return fresh NumPy action and learning arrays."""
        started = time.perf_counter()
        numerical_keys = np.asarray(jax.random.key_data(keys))
        counts["key_read_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        output = _systems_method(variables, memory, inputs, numerical_keys, host=True)
        counts["provider_seconds"] += time.perf_counter() - started
        counts["apply_calls"] += 1
        counts["input_logical_bytes"] += _bytes((inputs, _systems_numeric_tree(keys)))
        counts["output_logical_bytes"] += _bytes(
            (output.actions, output.learning_outputs)
        )
        return output

    return (
        System(
            "benchmark-host",
            provider,
            host_weights,
            init=initialize,
            reset_memory=reset,
            execution="host",
        ),
        second,
        counts,
    )


def _systems_run(
    first: Tree, second: Tree, *, length: int, manual: bool, mixed: bool
) -> Callable[..., Tree]:
    """Build a raw public or manual loop retaining identical decision data.

    Parameters
    ----------
    first, second : System
        Fixed method structures. Ordinary weights are explicit dynamic arguments
        to the returned callable; no checkpoint values are captured by a new jit.
    length : int
        Number of real reset/act/step rounds, 16 or 128 in the agreed cases.
    manual : bool
        Use public permitted-input/action tools plus explicit method/memory
        composition instead of apply_systems. RNG primitives and initial
        SystemState are shared so this comparison isolates application overhead.
    mixed : bool
        True uses a host loop with one batched JAX teammate and compiled step.
        False uses lax.scan and supports an enclosing jit.

    Returns
    -------
    Callable
        Accepts env, initial carry, root key and both weight trees. Returns final
        carry plus complete action, separate learning-output, reward, terminal
        valid and priority-metric histories. Full simulator-state history is never
        retained.
    """
    from marl_battlegrounds.evaluation import policy_execution as execution

    def reset_call(env: Environment, key: Array, state: EnvironmentState) -> Tree:
        """Reset finished lanes through the public wrapper with explicit keys."""
        return env.reset_done(key, state)

    def step_call(
        env: Environment, key: Array, state: EnvironmentState, actions: Action
    ) -> Tree:
        """Advance the native environment once using the already-chosen actions."""
        return env.step(key, state, actions)

    def prepare_call(
        env: Environment,
        observations: Observations,
        state: EnvironmentState,
        team: int,
        key: Array,
    ) -> Tree:
        """Build one team's permitted inputs and keys from the same pre-step epoch."""
        return env.policy_inputs(
            observations, state, team=team
        ), execution._action_keys(key, state.episode_id, team)  # pyright: ignore[reportPrivateUsage]

    def second_call(
        env: Environment,
        observations: Observations,
        state: EnvironmentState,
        variables: Tree,
        memory: Tree,
        generations: Array,
        key: Array,
    ) -> Tree:
        """Fuse the manual JAX teammate's input, zero-reset and method application.

        Inputs describe one native batch. The benchmark initializer is exactly
        zero memory, so selected generations choose zeros with no action call.
        Return the same complete SystemOutput as the unfused method. Ordinary
        variables, memory and keys stay dynamic in this stable compiled function.
        """
        inputs, keys = prepare_call(env, observations, state, 1, key)
        selected = state.reset_generation != generations
        memory = jnp.where(selected[:, None, None], jnp.zeros_like(memory), memory)
        return second.apply(variables, memory, inputs, keys)

    numerical_second = cast(Callable[..., Tree], jax.jit(second_call))

    reset_environment = cast(
        Callable[..., Tree], jax.jit(reset_call) if mixed else reset_call
    )
    step_environment = cast(
        Callable[..., Tree], jax.jit(step_call) if mixed else step_call
    )
    prepare = cast(
        Callable[..., Tree],
        jax.jit(prepare_call, static_argnums=3) if mixed else prepare_call,
    )

    def run(
        env: Tree,
        initial: _SystemsCarry,
        root: Array,
        variables_a: Tree,
        variables_b: Tree,
    ) -> Tree:
        """Run from the same initial arrays and retain every method result."""

        def advance(current: _SystemsCarry, tick: Array) -> tuple[_SystemsCarry, Tree]:
            """Reset completed lanes, apply both teams once, then advance Core once."""
            observations, state = reset_environment(
                env, jax.random.fold_in(root, tick * 3), current.state
            )
            decision_key = jax.random.fold_in(root, tick * 3 + 1)
            if manual:
                memory = current.memory
                selected = state.reset_generation != memory.reset_generation
                outputs: list[Tree] = []
                memories: list[Tree] = []
                for team, method, variables, old in (
                    (0, first, variables_a, memory.team_a),
                    (1, second, variables_b, memory.team_b),
                ):
                    if mixed and team == 1:
                        output = numerical_second(
                            env,
                            observations,
                            state,
                            variables,
                            old,
                            memory.reset_generation,
                            decision_key,
                        )
                        memories.append(output.next_memory)
                        outputs.append(output)
                        continue
                    inputs, keys = prepare(env, observations, state, team, decision_key)
                    if mixed and team == 0:
                        host_inputs, host_selected = jax.device_get((inputs, selected))
                        if np.any(host_selected):
                            fresh = first.init(variables, host_inputs, keys)
                            old = first.reset_memory(old, fresh, host_selected)
                        output = first.apply(variables, old, host_inputs, keys)
                    else:
                        fresh = _systems_initial_memory(
                            variables,
                            inputs,
                            execution._initialization_keys(  # pyright: ignore[reportPrivateUsage]
                                memory.init_key, state.episode_id, team
                            ),
                        )
                        old = jnp.where(selected[:, None, None], fresh, old)
                        output = method.apply(variables, old, inputs, keys)
                    memories.append(output.next_memory)
                    outputs.append(output)
                actions = env.join_actions(outputs[0].actions, outputs[1].actions)
                valid = ~state.done.done
                trace = memory.policy_trace._replace(
                    episode_id=state.episode_id,
                    decision_step=state.core_state.step_count
                    - state.initial_step_count,
                    valid=valid,
                    policy_ids=jnp.full((*state.episode_id.shape, 10), -1, jnp.int32),
                )
                memory = memory._replace(
                    team_a=memories[0],
                    team_b=memories[1],
                    episode_id=state.episode_id,
                    reset_generation=state.reset_generation,
                    policy_trace=trace,
                )
                learning = (outputs[0].learning_outputs, outputs[1].learning_outputs)
            else:
                actions, memory, learning = execution.apply_systems(
                    first,
                    second,
                    current.memory,
                    observations,
                    state,
                    decision_key,
                    variables_a=variables_a,
                    variables_b=variables_b,
                )
            valid = ~state.done.done
            observations, following, reward, done, info = step_environment(
                env, jax.random.fold_in(root, tick * 3 + 2), state, actions
            )
            updated = _SystemsCarry(
                observations,
                following,
                memory,
                current.advances + valid.astype(jnp.int32),
                current.completions + info.completed.astype(jnp.int32),
                current.rewards + reward.rewards,
            )
            return updated, (
                actions,
                learning,
                reward.rewards,
                done.done,
                valid,
                info.priority,
            )

        if not mixed:
            return jax.lax.scan(advance, initial, jnp.arange(length, dtype=jnp.int32))
        current = initial
        history = []
        for tick in range(length):
            current, row = advance(current, jnp.asarray(tick, jnp.int32))
            history.append(row)
        return current, jax.tree.map(lambda *rows: jnp.stack(rows), *history)

    return run


def _systems_host_measure(
    function: Callable[..., Tree],
    args: tuple[Tree, ...],
    repeats: int,
    probes: tuple[tuple[Tree, ...], ...] = (),
) -> tuple[Tree, dict[str, Any], list[Tree]]:
    """Time a host-controlled workflow and observe JAX compile/load events.

    Parameters
    ----------
    function : Callable
        Complete synchronous host/mixed or evaluator operation. Returned numerical
        leaves are synchronized; EvaluationResult already owns ready host arrays.
    args : tuple
        First call arguments. Each warm repeat starts from these same inputs.
    repeats : int
        At least five warm samples for qualification.
    probes : tuple, default=()
        Extra argument sets, used for changed values and repeated matchup checks.

    Returns
    -------
    tuple
        Last warm result, first/warm/event reports, and separate probe results.
        Backend events include persistent-cache loads, not only new compilations.
        Their durations are not subtracted to invent an isolated execution time.
    """
    events: list[dict[str, Any]] = []
    cache_hits: list[str] = []

    def duration(event: str, duration_secs: float, **metadata: str | int) -> None:
        """Retain only JAX's backend compilation-or-cache-load duration events."""
        if event == "/jax/core/compile/backend_compile_duration":
            events.append({"seconds": duration_secs, **metadata})

    def cache_event(event: str, **metadata: str | int) -> None:
        """Count persistent-cache hits separately from backend compile/load events."""
        del metadata
        if event == "/jax/compilation_cache/cache_hits":
            cache_hits.append(event)

    def call(values: tuple[Tree, ...]) -> tuple[Tree, dict[str, Any]]:
        """Synchronize one operation and associate only its new monitor events."""
        before, hits = len(events), len(cache_hits)
        started = time.perf_counter()
        output = function(*values)
        if not hasattr(output, "episodes"):
            jax.block_until_ready(output)
        report = {
            "wall_ms": (time.perf_counter() - started) * 1000,
            "backend_compile_or_load_events": events[before:],
            "persistent_cache_hits": len(cache_hits) - hits,
        }
        return output, report

    jax.monitoring.register_event_duration_secs_listener(duration)
    jax.monitoring.register_event_listener(cache_event)
    try:
        result, initial = call(args)
        warmed: list[dict[str, Any]] = []
        for _ in range(repeats):
            result, report = call(args)
            warmed.append(report)
        extra: list[Tree] = []
        probe_reports: list[dict[str, Any]] = []
        for values in probes:
            output, report = call(values)
            extra.append(output)
            probe_reports.append(report)
    finally:
        jax.monitoring.unregister_event_duration_listener(duration)
        jax.monitoring.unregister_event_listener(cache_event)
    samples = [row["wall_ms"] for row in warmed]
    return (
        result,
        {
            "first_call": initial,
            "warm_calls": warmed,
            "warm_samples_ms": samples,
            "warm_median_ms": statistics.median(samples),
            "warm_spread_ms": max(samples) - min(samples),
            "probe_calls": probe_reports,
            "compile_count_limit": (
                "JAX backend compile/load monitoring, with cache hits separate; "
                "first wall time also includes host setup and first execution."
            ),
        },
        extra,
    )


def _systems_transfer_probe(
    function: Callable[..., Tree], args: tuple[Tree, ...]
) -> dict[str, Any]:
    """Measure explicit device_get boundaries during one untimed mixed repeat.

    Parameters
    ----------
    function : Callable
        The already-warmed public or manual mixed loop. No concurrent work may
        use this worker while its temporary device_get observer is installed.
    args : tuple
        Same initial environment, memories, root and weights used by warm calls.

    Returns
    -------
    dict
        Each explicit get's logical device-array bytes and wall time, including
        waits for producing work. Implicit NumPy reads and automatic host-to-device
        placement are not counted. The fake provider separately times key reads.
        This is a boundary diagnostic, not an exhaustive transfer profiler.
    """
    original = jax.device_get
    calls: list[dict[str, Any]] = []

    def observed(value: Tree) -> Tree:
        """Time one explicit get without changing its returned structure or values."""
        size = sum(
            _bytes(_systems_numeric_tree(leaf))
            for leaf in jax.tree.leaves(value)
            if isinstance(leaf, Array)
        )
        started = time.perf_counter()
        output = original(value)
        calls.append(
            {
                "logical_device_input_bytes": size,
                "wall_ms": (time.perf_counter() - started) * 1000,
            }
        )
        return output

    cast(Any, jax).device_get = observed
    try:
        jax.block_until_ready(function(*args))
    finally:
        cast(Any, jax).device_get = original
    return {
        "calls": calls,
        "count": len(calls),
        "logical_device_input_bytes": sum(
            row["logical_device_input_bytes"] for row in calls
        ),
        "wall_ms": sum(row["wall_ms"] for row in calls),
        "scope": "One separate warmed repeat. Explicit device_get only; bytes count "
        "logical leaves, and wall time includes required synchronization. "
        "Provider key reads are separate; no exhaustive H2D/D2H event claim.",
    }


def _systems_evaluator(args: argparse.Namespace) -> tuple[Tree, dict[str, Any]]:
    """Measure the existing Policy evaluator against the same frozen schedule.

    args supplies map_id and repeats. Run 128 ordinary 300-tick-limit episodes,
    batch 128, chunk 128, priority metrics and no writer/replay. Retain all public
    episode rows and metric columns. Repeated same-shaped values and compatible
    matchup probes are outside warmed timing. Return numerical/identity results
    and complete-call timing; source-dependent runtime provenance is excluded.
    """
    from dataclasses import asdict, replace

    from marl_battlegrounds.evaluation.evaluate import (
        EpisodeSpec,
        evaluate_episodes,
        policy_description,
    )
    from marl_battlegrounds.evaluation.policy_execution import Policy

    roster_a, roster_b = _benchmark_rosters()
    config = make_standard_team_deathmatch_config(
        map_id=args.map_id, team_a_roster=roster_a, team_b_roster=roster_b
    )
    schedule = tuple(
        EpisodeSpec(index + 1, config, args.map_id, index + 1) for index in range(128)
    )
    first = policy("tdm-alpha")
    second = policy("tdm-beta")

    def run(a: Policy, b: Policy, seed: int) -> Tree:
        """Complete one fixed pass; methods and numerical variables are arguments."""
        return evaluate_episodes(
            a,
            b,
            schedule,
            seed=seed,
            num_envs=128,
            chunk_size=128,
            metrics="priority",
            run_id="systems-benchmark",
            pass_id="1",
        )

    changed = replace(first, variables=jnp.asarray(1.0, jnp.float32))
    base = replace(first, variables=jnp.asarray(0.0, jnp.float32))
    output, timing, probes = _systems_host_measure(
        run,
        (base, second, 42),
        args.repeats,
        ((changed, second, 43), (second, base, 42), (second, changed, 43)),
    )
    timing["compilation_reuse_pass"] = not any(
        call["backend_compile_or_load_events"]
        for call in (
            *timing["warm_calls"],
            timing["probe_calls"][0],
            timing["probe_calls"][2],
        )
    )
    descriptions = [
        policy_description(
            team, team.variables, team.initial_carry, include_digests=True
        )
        for team in (base, second)
    ]
    retained = {
        "episodes": [asdict(episode) for episode in output.episodes],
        "priority_metrics": {
            name: values.tolist() for name, values in output.priority_metrics.items()
        },
        "full_metrics": {
            name: values.tolist() for name, values in output.full_metrics.items()
        },
        "completed_episode_ids": list(output.completed_episode_ids),
        "policies": descriptions,
        "probe_completed_counts": [len(item.episodes) for item in probes],
    }
    timing["real_transitions"] = sum(
        episode.episode_length for episode in output.episodes
    )
    timing["transitions_per_second"] = (
        timing["real_transitions"] * 1000 / timing["warm_median_ms"]
    )
    timing["reuse_probe_limit"] = (
        "Reactive Policies ignore numerical variables; probes test "
        "wrapper recompilation on same-shaped values and repeated "
        "compatible callable pairings, not neural-model parameter use. A "
        "reversed callable pairing may require its first distinct "
        "compilation."
    )
    timing["retained_output_bytes"] = len(json.dumps(retained, sort_keys=True).encode())
    timing["output_size_limit"] = (
        "JSON encoding size of retained comparison payload; encoding "
        "occurs outside timing. No writer/storage throughput is measured."
    )
    return retained, timing


def _run_systems_worker(args: argparse.Namespace, output_path: Path) -> None:
    """Save one isolated Systems comparison with frozen source and asset identities.

    args selects one named case and public/manual or baseline/current route.
    output_path is updated on success or failure. GPU is qualification; CPU is a
    diagnostic only. Results include retained numerical payload for direct paired
    comparison, plus process/device peaks with their whole-worker limitations.
    """
    count, length = _SYSTEM_CASES[args.systems_case]
    before = _foundation_identity(args.package_root, args.assets_root)
    _foundation_assets(args.assets_root)
    result: dict[str, Any] = {
        "status": "running",
        "benchmark_mode": "systems",
        "case": args.systems_case,
        "route": args.systems_route,
        "identity": before,
        "source_revision": args.source_revision,
        "num_envs": count,
        "rollout_length": length,
        "metrics_mode": "priority",
        "map_id": args.map_id,
        "backend": args.backend,
        "repeats": args.repeats,
        "jax": jax.__version__,
        "python": sys.version,
        "worker_pid": os.getpid(),
        "qualification_scope": "GPU workflow efficiency"
        if args.backend == "gpu"
        else "CPU diagnostic only",
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    try:
        device = cast(Any, jax.devices(args.backend)[0])
        result["device"] = str(device)
        result["device_kind"] = device.device_kind
        with jax.default_device(device):
            if args.systems_case == "evaluator":
                retained, timing = _systems_evaluator(args)
                result.update(
                    timing=timing,
                    retained=retained,
                    common_result_sha256=sha256(
                        json.dumps(retained, sort_keys=True).encode()
                    ).hexdigest(),
                )
                if not timing["compilation_reuse_pass"]:
                    raise AssertionError(
                        "Repeated evaluator/matchup calls recompiled on ordinary values"
                    )
            else:
                from marl_battlegrounds.evaluation.policy_execution import init_systems

                mixed = args.systems_case == "mixed"
                first, second, counters = _systems_pair(mixed=mixed)
                roster_a, roster_b = _benchmark_rosters()
                started = time.perf_counter()
                config = make_standard_team_deathmatch_config(
                    map_id=args.map_id, team_a_roster=roster_a, team_b_roster=roster_b
                )
                env = make(
                    "tdm",
                    num_envs=count,
                    env_config=config,
                    metrics="priority",
                    balance_spawn_locations=False,
                )
                root = jax.random.key(42)
                observations, state = env.reset(root)
                memory = init_systems(
                    first, second, observations, state, jax.random.key(43)
                )
                initial = _SystemsCarry(
                    observations,
                    state,
                    memory,
                    jnp.zeros(count, jnp.int32),
                    jnp.zeros(count, jnp.int32),
                    jnp.zeros((count, 10), jnp.float32),
                )
                jax.block_until_ready(initial)
                result["setup_including_first_initialization_ms"] = (
                    time.perf_counter() - started
                ) * 1000
                function = _systems_run(
                    first,
                    second,
                    length=length,
                    manual=args.systems_route == "manual",
                    mixed=mixed,
                )
                values = (env, initial, root, first.variables, second.variables)

                def changed_value(leaf: Tree) -> Tree:
                    """Change a numerical value while keeping its shape and dtype."""
                    return leaf + jnp.asarray(0.0001, dtype=leaf.dtype)

                changed_a = (
                    jax.tree.map(changed_value, first.variables)
                    if not mixed
                    else first.variables
                )
                changed_b = jax.tree.map(changed_value, second.variables)
                changed_memory = memory._replace(
                    team_a=memory.team_a
                    if mixed
                    else jax.tree.map(changed_value, memory.team_a),
                    team_b=jax.tree.map(changed_value, memory.team_b),
                )
                probes = (
                    (
                        env,
                        initial._replace(memory=changed_memory),
                        jax.random.key(44),
                        changed_a,
                        changed_b,
                    ),
                )
                if mixed:
                    observed, timing, _ = _systems_host_measure(
                        function, values, args.repeats, probes
                    )
                    result["timing"] = timing
                    if any(
                        call["backend_compile_or_load_events"]
                        for call in (*timing["warm_calls"], *timing["probe_calls"])
                    ):
                        raise AssertionError(
                            "Repeated mixed calls or same-shaped values triggered "
                            "backend compile/load work"
                        )
                    result["host_method_counters"] = dict(counters)
                    expected_calls = length * (args.repeats + 2)
                    if counters["apply_calls"] != expected_calls:
                        raise AssertionError(
                            "The live mixed loop did not call its host method "
                            "once per decision"
                        )
                    result["explicit_transfer_probe"] = _systems_transfer_probe(
                        function, values
                    )
                    result["host_counter_scope"] = (
                        "Includes first, warm and changed-value calls; delivered "
                        "logical payload and provider compute, not all "
                        "device transfer events."
                    )
                else:
                    observed, timing, _ = _foundation_measure(
                        function, values, args.repeats, probes
                    )
                    result["timing"] = timing
                    if timing["new_traces_for_changed_values"]:
                        raise AssertionError(
                            "Same-shaped raw System values triggered a new trace"
                        )
                carry, history = observed
                result["timing"] = timing
                result["real_transitions"] = int(
                    np.asarray(jax.device_get(carry.advances)).sum()
                )
                result["transitions_per_second"] = (
                    result["real_transitions"] * 1000 / timing["warm_median_ms"]
                )
                result["completed_episodes"] = int(
                    np.asarray(jax.device_get(carry.completions)).sum()
                )
                numeric_output = _systems_numeric_tree(observed)
                result["retained_tree_structure"] = str(
                    cast(object, jax.tree.structure(numeric_output))
                )
                result["retained_output_bytes"] = _bytes(numeric_output)
                result["retained_history_bytes"] = _bytes(history)
                started = time.perf_counter()
                host_output = jax.device_get(numeric_output)
                result["ready_output_transfer_ms"] = (
                    time.perf_counter() - started
                ) * 1000
                result["common_result_sha256"] = _foundation_digest(host_output)
                payload_path = output_path.with_suffix(".npz")
                payload: dict[str, Any] = {
                    f"leaf_{index:04d}": np.asarray(leaf)
                    for index, leaf in enumerate(jax.tree.leaves(host_output))
                }
                np.savez(payload_path, allow_pickle=False, **payload)
                result["comparison_payload"] = str(payload_path)
                result["output_file_bytes"] = payload_path.stat().st_size
                result["reference_dependencies"] = (
                    "Same initial SystemState, public permitted-input/action/reset "
                    "helpers, and private action/init key primitives. Manual route "
                    "replaces System application, reset-memory selection and trace "
                    "packaging."
                )
                result["method_scope"] = (
                    "Fixed recurrent numerical method consuming recipient-authorized "
                    "self/source data and returning values; not training, sample "
                    "efficiency or learned tactics. Full final carry and all "
                    "action/learning/reward/terminal/valid histories retained "
                    "equally."
                )
            result["device_memory_stats"] = device.memory_stats()
            result["process_peak_ram_bytes"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
            result["memory_limit"] = (
                "Whole-worker peaks include setup, compile/load, warmed calls, "
                "probes and retained outputs. Compiler estimates describe "
                "individual executables; process peaks are not pure execution "
                "allocations."
            )
        if before != _foundation_identity(args.package_root, args.assets_root):
            raise RuntimeError(
                "Package, harness or fixed assets changed during measurement"
            )
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        output_path.write_text(json.dumps(result, indent=2) + "\n")


def _run_systems(args: argparse.Namespace) -> int:
    """Dispatch only the three agreed shapes and compare their isolated workers.

    args supplies candidate/baseline source roots, one fixed asset root and output
    directory. Named cases avoid an accidental Cartesian workload matrix. Both
    evaluator workers run this exact harness, importing their requested package.
    Return nonzero for worker failure or mismatched retained outputs. Never run
    concurrent GPU workers, reduce a batch, or overwrite an existing result.
    """
    root = Path(__file__).resolve().parents[2]
    args.package_root = (args.package_root or root).resolve()
    args.assets_root = (
        args.assets_root or args.package_root / "src/marl_battlegrounds/data/tdm"
    ).resolve()
    args.backend = args.backend or "gpu"
    args.map_id = 12 if args.map_id is None else args.map_id
    if args.repeats < 5 or any(
        (
            args.foundations,
            args.setup_only,
            args.metrics_only,
            args.sizes,
            args.lengths,
            args.include_scalar,
            args.rollout_modes,
            args.api,
            args.action_workload,
        )
    ):
        raise ValueError(
            "Systems uses named cases, priority metrics and at least five warm repeats"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.systems_route is not None:
        if args.systems_case is None:
            raise ValueError("A Systems worker needs one named case")
        allowed_routes = (
            ("baseline", "current")
            if args.systems_case == "evaluator"
            else ("manual", "public")
        )
        if args.systems_route not in allowed_routes:
            raise ValueError("The route does not belong to this Systems case")
        output = (
            args.output
            / f"systems-{args.systems_case}-{args.systems_route}-{args.backend}.json"
        )
        _run_systems_worker(args, output)
        return 0
    cases = (args.systems_case,) if args.systems_case else tuple(_SYSTEM_CASES)
    if "evaluator" in cases and args.baseline_root is None:
        raise ValueError(
            "The evaluator case needs --baseline-root for committed 91415c3"
        )
    failed = False
    for case in cases:
        paths: list[Path] = []
        for route in (
            ("baseline", "current") if case == "evaluator" else ("manual", "public")
        ):
            package = (
                args.baseline_root.resolve()
                if route == "baseline"
                else args.package_root
            )
            path = args.output / f"systems-{case}-{route}-{args.backend}.json"
            if path.exists() or path.with_suffix(".npz").exists():
                raise FileExistsError(
                    f"Preserve prior evidence; choose a new output directory: {path}"
                )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--systems",
                "--systems-case",
                case,
                "--systems-route",
                route,
                "--backend",
                args.backend,
                "--package-root",
                str(package),
                "--assets-root",
                str(args.assets_root),
                "--map-id",
                str(args.map_id),
                "--repeats",
                str(args.repeats),
                "--output",
                str(args.output.resolve()),
            ]
            revision = (
                "91415c33193402aa7a5166ee8a53763fe2d24156"
                if route == "baseline"
                else args.source_revision
            )
            if revision:
                command.extend(("--source-revision", revision))
            _notify(f"Systems {case}/{route}/{args.backend}")
            process = subprocess.run(
                command,
                cwd=package,
                env={
                    **os.environ,
                    "PYTHONPATH": str(package / "src"),
                    "JAX_PLATFORMS": "cpu" if args.backend == "cpu" else "cuda,cpu",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                check=False,
            )
            failed |= process.returncode != 0
            paths.append(path)
        if any(not path.exists() for path in paths):
            failed = True
            continue
        records = [json.loads(path.read_text()) for path in paths]
        if any(record["status"] != "complete" for record in records):
            failed = True
            continue
        if (
            records[0]["identity"]["asset_sha256"]
            != records[1]["identity"]["asset_sha256"]
            or records[0]["identity"]["harness_sha256"]
            != records[1]["identity"]["harness_sha256"]
        ):
            raise AssertionError(
                "Paired workers used different assets or harness bytes"
            )
        if (
            case != "evaluator"
            and records[0]["identity"]["package_sha256"]
            != records[1]["identity"]["package_sha256"]
        ):
            raise AssertionError("Public/manual workers used different package bytes")
        if case == "evaluator":
            if records[0]["common_result_sha256"] != records[1]["common_result_sha256"]:
                raise AssertionError(
                    "Legacy evaluator retained outputs or identities changed"
                )
        else:
            if (
                records[0]["retained_tree_structure"]
                != records[1]["retained_tree_structure"]
            ):
                raise AssertionError("Retained result structures differ")
            with (
                np.load(paths[0].with_suffix(".npz")) as expected,
                np.load(paths[1].with_suffix(".npz")) as actual,
            ):
                if expected.files != actual.files:
                    raise AssertionError("Retained result leaf inventories differ")
                for name in expected.files:
                    _assert_equal(np.asarray(expected[name]), np.asarray(actual[name]))
        comparison = {
            "status": "complete",
            "case": case,
            "workers": [str(path) for path in paths],
            "warm_median_ms": {
                record["route"]: record["timing"]["warm_median_ms"]
                for record in records
            },
            "second_over_first_median_ratio": records[1]["timing"]["warm_median_ms"]
            / records[0]["timing"]["warm_median_ms"],
            "retained_output_agreement": (
                "Exact evaluator JSON; integer/bool equality and rtol=3e-5, "
                "atol=0.002 for raw floating leaves"
            ),
            "memory_limit": (
                "Separate worker peaks avoid simultaneous public/reference "
                "executables; allocator/process measurements retain their "
                "declared limits."
            ),
        }
        (args.output / f"systems-{case}-comparison-{args.backend}.json").write_text(
            json.dumps(comparison, indent=2) + "\n"
        )
    return int(failed)


def _recording_rollout(
    env: Environment, initial: EnvironmentState, root: Array
) -> Tree:
    """Advance 32 games for 16 decisions and retain terminal records only.

    The environment and initial state are dynamic numerical inputs. Both old and
    new packages use the same ALPHA/BETA actions, keys and complete final state.
    No writer, trace history or host callback belongs to this compiled workload.
    """

    def advance(carry: Tree, tick: Array) -> tuple[Tree, None]:
        """Choose and apply one action per game, retaining first terminal summaries."""
        state, completed = carry
        actions = _actions(state, env, tick, exploration=False)
        keys = episode_keys(
            root, state.episode_id, jnp.full_like(state.episode_id, tick), 1
        )
        _, successor, _, _, info = env.step(keys, state, actions)
        return (successor, _retain_completion(completed, successor, info)), None

    final, _ = jax.lax.scan(
        advance, (initial, _empty_completed(initial)), jnp.arange(16, dtype=jnp.int32)
    )
    return final[0], final[1].info


def _recording_save_costs(
    output: Path, host_state: Tree, host_info: Tree, mode: str, repeats: int
) -> dict[str, Any]:
    """Measure current writer saves and a simple exact-summary CSV reference.

    Inputs are already on the host. Every repeat writes the same 32 completed
    games to a fresh run and flushes it. The manual reference writes and fsyncs
    only required summary cells; it excludes identities, metrics and recovery.
    A separate current-schema probe measures trace compression without games.
    These host timings explain recording costs; they are not GPU simulation rates.
    """
    import csv

    from marl_battlegrounds.evaluation.evaluate import policy_description
    from marl_battlegrounds.evaluation.run_writer import RunWriter

    samples: list[float] = []
    sizes: list[int] = []
    manual_samples: list[float] = []
    identity_samples: list[float] = []
    descriptors = (policy("tdm-alpha"), policy("tdm-beta"))
    for _ in range(repeats):
        started = time.perf_counter()
        for descriptor in descriptors:
            policy_description(
                descriptor,
                descriptor.variables,
                descriptor.initial_carry,
                include_digests=True,
            )
        identity_samples.append((time.perf_counter() - started) * 1000)
    summaries = np.column_stack(
        (
            np.asarray(host_state.episode_id),
            np.asarray(host_info.outcome),
            np.asarray(
                host_state.core_state.step_count - host_state.initial_step_count
            ),
            np.asarray(host_state.core_state.team_deathmatch_scores),
        )
    ).tolist()
    for repeat in range(repeats):
        started = time.perf_counter()
        with RunWriter(
            output / f"saved-{repeat}",
            policies={"team_a": policy("tdm-alpha"), "team_b": policy("tdm-beta")}
            if hasattr(host_info, "decision_step")
            else {"team_a": "tdm-alpha", "team_b": "tdm-beta"},
            details={"metrics": mode},
        ) as writer:
            writer.write(host_info)
            writer.flush()
        samples.append((time.perf_counter() - started) * 1000)
        sizes.append(
            sum(
                path.stat().st_size
                for path in writer.run_dir.rglob("*")
                if path.is_file()
            )
        )
        started = time.perf_counter()
        with (output / f"manual-outcomes-{repeat}.csv").open("w", newline="") as stream:
            csv_writer = csv.writer(stream)
            csv_writer.writerow(
                (
                    "episode_id",
                    "outcome",
                    "episode_length",
                    "team_a_score",
                    "team_b_score",
                )
            )
            csv_writer.writerows(summaries)
            stream.flush()
            os.fsync(stream.fileno())
        manual_samples.append((time.perf_counter() - started) * 1000)
    result: dict[str, Any] = {
        "complete_save_samples_ms": samples,
        "complete_save_median_ms": statistics.median(samples),
        "complete_run_bytes": sizes,
        "two_policy_identity_samples_ms": identity_samples,
        "two_policy_identity_median_ms": statistics.median(identity_samples),
        "manual_summary_samples_ms": manual_samples,
        "manual_summary_median_ms": statistics.median(manual_samples),
        "manual_scope": (
            "32 exact integer outcome rows, CSV encoding and file fsync only; "
            "excludes policy/config identity, metrics, manifest and recovery."
        ),
        "prior_schema_none_limit": (
            "Schema 1 has no ordinary none-mode outcome CSV; the manual reference "
            "records those required summaries on both revisions."
        ),
    }
    if hasattr(host_info, "decision_step"):
        from marl_battlegrounds.evaluation.policy_execution import PolicyTrace

        started = time.perf_counter()
        with RunWriter(
            output / "assignments",
            buffer_size=32,
            policies={"team_a": policy("tdm-alpha"), "team_b": policy("tdm-beta")},
        ) as writer:
            for tick in range(16):
                local = np.full(32, tick, np.int32)
                info = host_info._replace(
                    completed=np.zeros(32, bool),
                    priority=None,
                    full=None,
                    decision_step=local,
                    episode_length=local + 1,
                )
                trace = PolicyTrace(
                    host_info.episode_id,
                    cast(Array, local),
                    cast(Array, np.ones(32, bool)),
                    cast(Array, np.full((32, 10), -1 if tick % 2 else 0, np.int32)),
                )
                writer.write(info, policy_trace=trace)
            writer.flush()
        result["assignment_probe_ms"] = (time.perf_counter() - started) * 1000
        with writer.paths["policy_assignments"].open(newline="") as stream:
            result["assignment_rows"] = sum(1 for _ in csv.DictReader(stream))
        result["assignment_csv_bytes"] = (
            writer.paths["policy_assignments"].stat().st_size
        )
        result["assignment_probe_scope"] = (
            "Host-only synthetic 16-decision epochs over the actual roster: "
            "alternating known/unknown choices, no completions. This measures "
            "bounded flushing, not action generation."
        )
    return result


def _run_recording_contracts(args: argparse.Namespace) -> int:
    """Measure only the Packet 3 batch-32, length-16 recording contract.

    Run this harness in separate processes with PYTHONPATH selecting the declared
    package root. Output stores each mode's exact source/assets, final common
    payload, measurements and writer files. This mode never launches the older
    benchmark campaign. GPU qualifies performance; CPU is a smoke check only.
    """
    from marl_battlegrounds.evaluation.metric_catalog import (
        METRIC_SCHEMA_VERSION,
        PRIORITY_METRIC_NAMES,
    )

    root = Path(__file__).resolve().parents[2]
    args.package_root = (args.package_root or root).resolve()
    args.assets_root = (
        args.assets_root or args.package_root / "src/marl_battlegrounds/data/tdm"
    ).resolve()
    args.backend = args.backend or "gpu"
    args.map_id = 12 if args.map_id is None else args.map_id
    if (
        args.repeats < 5
        or args.sizes not in (None, [32])
        or args.lengths not in (None, [16])
    ):
        raise ValueError(
            "Recording contracts use batch 32, length 16 and at least five warm samples"
        )
    if args.systems or args.foundations or args.metrics_only or args.include_scalar:
        raise ValueError(
            "Recording contracts cannot be combined with another benchmark mode"
        )
    modes = args.rollout_modes or ["none", "priority", "full"]
    if any(mode not in ("none", "priority", "full") for mode in modes):
        raise ValueError("Recording contracts support none, priority and full only")
    identity = _foundation_identity(args.package_root, args.assets_root)
    _foundation_assets(args.assets_root)
    args.output.mkdir(parents=True, exist_ok=True)
    device = cast(Any, jax.devices(args.backend)[0])
    for mode in modes:
        path = args.output / f"recording-{mode}-{args.backend}.json"
        if path.exists():
            raise FileExistsError(
                f"Preserve prior measurements; choose another output folder: {path}"
            )
        record: dict[str, Any] = {
            "status": "running",
            "identity": identity,
            "source_revision": args.source_revision,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "mode": mode,
            "batch": 32,
            "rollout_length": 16,
            "backend": args.backend,
            "device": str(device),
            "device_kind": device.device_kind,
            "jax": jax.__version__,
            "qualification_scope": "GPU efficiency"
            if args.backend == "gpu"
            else "CPU smoke only",
        }
        path.write_text(json.dumps(record, indent=2) + "\n")
        try:
            with jax.default_device(device):
                started = time.perf_counter()
                roster_a, roster_b = _benchmark_rosters()
                config = make_standard_team_deathmatch_config(
                    map_id=args.map_id,
                    team_a_roster=roster_a,
                    team_b_roster=roster_b,
                    max_steps=16,
                )
                env = make(
                    "tdm",
                    num_envs=32,
                    env_config=config,
                    metrics=cast(MetricMode, mode),
                )
                _, initial = env.reset(jax.random.key(42))
                jax.block_until_ready(initial)
                record["setup_including_reset_ms"] = (
                    time.perf_counter() - started
                ) * 1000
                (final, info), timing, _ = _foundation_measure(
                    _recording_rollout,
                    (env, initial, jax.random.key(43)),
                    args.repeats,
                    ((env, initial, jax.random.key(44)),),
                )
                if timing["new_traces_for_changed_values"]:
                    raise AssertionError(
                        "Same-shaped changed keys caused recompilation"
                    )
                record["timing"] = timing
                started = time.perf_counter()
                host_state, host_info = jax.device_get((final, info))
                record["retained_host_transfer_ms"] = (
                    time.perf_counter() - started
                ) * 1000
                record["retained_logical_bytes"] = _bytes((final, info))
                record["state_bytes"] = _bytes(final)
                record["info_bytes"] = _bytes(info)
                record["priority_value_validity_bytes"] = (
                    _bytes(info.priority) if info.priority is not None else 0
                )
                record["full_value_validity_bytes"] = (
                    _bytes(info.full) if info.full is not None else 0
                )
                record["real_transitions"] = int(
                    np.asarray(host_state.cumulative_transition_count).sum()
                )
                record["transitions_per_second"] = (
                    record["real_transitions"] * 1000 / timing["warm_median_ms"]
                )
                common = (
                    host_state,
                    host_info.episode_id,
                    host_info.completed,
                    host_info.outcome,
                )
                record["common_trajectory_sha256"] = _foundation_digest(common)
                record["metrics_by_name"] = {
                    group: {
                        name: {
                            "values": np.asarray(values.values[:, index]).tolist(),
                            "valid": np.asarray(values.valid[:, index]).tolist(),
                        }
                        for index, name in enumerate(names)
                        if not (name.startswith("agent_") and name.endswith("_return"))
                    }
                    for group, names, values in (
                        ("priority", PRIORITY_METRIC_NAMES, host_info.priority),
                        ("full", FULL_METRIC_NAMES, host_info.full),
                    )
                    if values is not None
                }
                folder = args.output / f"recording-{mode}-{args.backend}-files"
                folder.mkdir()
                record["saving"] = _recording_save_costs(
                    folder, host_state, host_info, mode, args.repeats
                )
                record["device_memory_stats"] = device.memory_stats()
                record["process_peak_ram_bytes"] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                )
                record["memory_scope"] = (
                    "Process/device allocator peaks include prior modes, setup, "
                    "compilation and save probes. Per-executable memory is "
                    "reported separately."
                )
            if identity != _foundation_identity(args.package_root, args.assets_root):
                raise RuntimeError(
                    "Package, harness or assets changed during measurement"
                )
            record["status"] = "complete"
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            path.write_text(json.dumps(record, indent=2) + "\n")
    return 0


def main() -> int:
    """Parse options and run fresh workers for the chosen benchmark mode.

    Legacy workloads keep their behavior with allowed GPU batch sizes. Foundations
    mode uses frozen package and asset paths, records each case separately,
    and treats GPU timings
    as performance evidence. CPU foundations runs are diagnostics only.

    Returns
    -------
    int
        Zero when all workers succeed, or a nonzero status if any worker fails.

    Raises
    ------
    SystemExit
        If arguments are invalid or help was requested.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="native batch sizes (mode-specific defaults)",
    )
    parser.add_argument(
        "--foundations",
        action="store_true",
        help="qualify GPU setup and raw reset/step; CPU mode is diagnostic only",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        help="foundations/Systems backend (default: gpu; cpu is diagnostic only)",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="foundations: measure validated preparation and map construction only",
    )
    parser.add_argument(
        "--action-workload",
        choices=("fixed", "sample"),
        help="foundations: fixed neutral actions or legal sampling (default: fixed)",
    )
    parser.add_argument(
        "--api",
        choices=("reference", "automatic"),
        help="foundations reset route (default: reference)",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        help="foundations rollout lengths (default: 16 128)",
    )
    parser.add_argument(
        "--include-scalar",
        action="store_true",
        help="include a separate scalar foundations worker",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        help="source checkout/export to import in foundations/Systems workers",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        help="fixed TDM resources shared by foundations/Systems workers",
    )
    parser.add_argument(
        "--source-revision",
        help="declared package revision; file hashes identify the actual bytes",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="simulate each batch once with canonical 5v5 ALPHA/BETA, "
        "then compare metrics only",
    )
    parser.add_argument(
        "--map-id",
        type=int,
        choices=range(52),
        metavar="ID",
        help="fixed map for --metrics-only (default: 48) or foundations (default: 12)",
    )
    parser.add_argument(
        "--rollout-modes",
        nargs="+",
        choices=_ROLLOUT_MODES,
        help="modes to measure; legacy mode also compares full CPU/GPU metrics",
    )
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--systems",
        action="store_true",
        help="run the bounded Packet 2 System and legacy Policy comparisons",
    )
    parser.add_argument(
        "--systems-case",
        choices=tuple(_SYSTEM_CASES),
        help="one named case; default runs all four comparisons at three agreed shapes",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="source export of committed 91415c3 for the Systems evaluator comparison",
    )
    parser.add_argument(
        "--systems-route",
        choices=("manual", "public", "baseline", "current"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--recording-contracts",
        action="store_true",
        help="Packet 3 only: batch 32, rollout 16, priority/full/none and save costs",
    )
    args = parser.parse_args()
    if args.recording_contracts:
        return _run_recording_contracts(args)
    if args.systems:
        return _run_systems(args)
    if any((args.systems_case, args.systems_route, args.baseline_root)):
        parser.error("--systems-case/--systems-route/--baseline-root require --systems")
    if args.foundations:
        return _run_foundations(args)
    if any(
        (
            args.backend,
            args.api,
            args.lengths,
            args.include_scalar,
            args.package_root,
            args.assets_root,
            args.source_revision,
            args.setup_only,
            args.action_workload,
        )
    ):
        parser.error(
            "--backend/--api/--lengths/--include-scalar/--package-root/"
            "--assets-root/--source-revision/--setup-only/--action-workload "
            "require --foundations"
        )
    args.sizes = args.sizes or [32, 64, 128, 512, 1024]
    if any(size not in _GPU_BATCHES for size in args.sizes) or (
        args.worker is not None and args.worker not in _GPU_BATCHES
    ):
        parser.error("GPU environment batches must be 32, 64, 128, 512 or 1024")
    args.rollout_modes = args.rollout_modes or list(_ROLLOUT_MODES)
    if args.repeats < 2 or any(size < 1 for size in args.sizes):
        parser.error("positive batch sizes and at least two warm repeats are required")
    if args.map_id is not None and not args.metrics_only:
        parser.error("--map-id requires --metrics-only")
    # Preserve the reference-first ordering even when CLI selections differ.
    rollout_modes = tuple(mode for mode in _ROLLOUT_MODES if mode in args.rollout_modes)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        run_size(
            args.worker,
            args.output,
            args.repeats,
            rollout_modes,
            metrics_only=args.metrics_only,
            map_id=args.map_id,
        )
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
            "--rollout-modes",
            *rollout_modes,
        ]
        if args.metrics_only:
            command.append("--metrics-only")
        if args.map_id is not None:
            command.extend(("--map-id", str(args.map_id)))
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
