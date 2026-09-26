"""Compare fixed-input recurrent actor calls with their System adapter on B32.

Run ``python -m scripts.dev.benchmark_baseline --steps 16 --output DIRECTORY``
with the training extra installed and the intended CUDA UUID selected. Repeat
with --steps 128. Both paths use the same permitted inputs, weights, memory and
keys, retain identical outputs, and sample at least five warmed measurements.
Separate public cases include apply_systems, input preparation, both teams and
action/memory packaging, using the same random opponent in both comparisons.
--cpu-smoke permits a CPU correctness check; its timings are not GPU evidence.

The tool saves JSON measurements and small output arrays. It stores one input
snapshot, never T expanded actor-input copies. Fixed inputs may let the compiler
reuse encoding across decisions. This measures actor calls, not simulator steps,
varying-observation rollouts, learner updates or learned behavior.
The actor uses the baseline's default spawn frame, "left" since 22 September
2026. Measurements saved before that date used "world"; a rerun measures
the left program and is not a like-for-like repeat of them.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import resource
import statistics
import subprocess
import time
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines import ppo
from marl_battlegrounds.baselines.actions import (
    action_log_prob,
    categorical_action_mask,
    decode_actions,
    mirror_action_indices,
    sample_actions,
)
from marl_battlegrounds.baselines.inputs import (
    encode_actor_inputs,
    spawn_frame_flag,
    team_obstacle_partners,
)
from marl_battlegrounds.core.types import CONTEXT_FEATURE_CURRENT_TIMESTEP
from marl_battlegrounds.policies.input import mirror_team_view
from marl_battlegrounds.types import ActorAction, System, SystemInput, SystemOutput

type Tree = Any
type ActorCall = Callable[[Tree, Array, SystemInput, Array], SystemOutput]
_BATCH = 32


class ActorResult(NamedTuple):
    """Keep native actions and the same-call learner values for each decision.

    actions has three int32 (B,5) heads. learning holds int32 action indices
    and float32 log probabilities with that shape. A scan adds the time axis.
    Final recurrent memory is returned separately, without a memory history.
    """

    actions: ActorAction
    learning: ppo.PPOLearningOutputs


def _direct_actor(
    parameters: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    """Compose public numerical helpers for the same work as the System actor.

    parameters is the actor variable tree, memory is float32 (32,5,128),
    inputs is the same permitted SystemInput, and keys contains 32 lane keys.
    Return the same native actions, carry and learning values. This comparison
    reuses algorithm authorities; it is not a separate correctness reference.
    Match the declared left spawn frame: reflect permitted inputs and move masks,
    then return world-frame actions and learning indices.
    """
    flag = spawn_frame_flag(inputs.actors, "left")
    actors, frame_mask = mirror_team_view(
        inputs.actors,
        inputs.action_mask,
        flag,
        obstacle_partners=team_obstacle_partners(inputs.actors),
    )
    features = encode_actor_inputs(actors)
    valid = jnp.broadcast_to(inputs.valid[:, None], inputs.active_mask.shape)
    starts = jnp.broadcast_to(inputs.episode_start[:, None], inputs.active_mask.shape)
    memory, logits = cast(
        tuple[Array, Array],
        ppo.RecurrentActor().apply(
            parameters, memory, features[None], starts[None], valid[None]
        ),
    )
    logits = logits[0]
    mask = categorical_action_mask(frame_mask)

    def split(key: Array) -> Array:
        """Derive the same five actor streams as the public System adapter."""
        return jax.random.split(key, 5)

    actor_keys = jax.vmap(split)(keys)
    indices = sample_actions(logits, mask, actor_keys)
    world = mirror_action_indices(indices, flag)
    return SystemOutput(
        decode_actions(world),
        memory,
        learning_outputs=ppo.PPOLearningOutputs(
            world, action_log_prob(logits, mask, indices)
        ),
    )


def _scan(actor: ActorCall, steps: int) -> Callable[..., Tree]:
    """Build one fixed-length scan without storing expanded inputs over time.

    actor is either the public System callable or its direct equivalent.
    steps is 16 or 128. The returned function takes dynamic parameters, initial
    memory, one input snapshot and lane keys. Only the first decision uses the
    supplied episode-start flags. Return final memory and action/learning rows.
    """

    def run(parameters: Tree, memory: Array, inputs: SystemInput, keys: Array) -> Tree:
        """Apply a dynamic snapshot repeatedly with fresh per-decision keys."""

        def step(carry: Array, tick: Array) -> tuple[Array, ActorResult]:
            """Keep one actor carry and emit only each decision's used outputs."""
            current = inputs._replace(episode_start=inputs.episode_start & (tick == 0))

            def choose_key(key: Array) -> Array:
                """Tag each lane's key with this decision's sequence index."""
                return jax.random.fold_in(key, tick)

            action_keys = jax.vmap(choose_key)(keys)
            result = actor(parameters, carry, current, action_keys)
            return result.next_memory, ActorResult(
                result.actions, result.learning_outputs
            )

        return jax.lax.scan(step, memory, jnp.arange(steps, dtype=jnp.int32))

    return run


def _bytes(tree: Tree) -> int:
    """Sum numerical leaf sizes in bytes; shared storage is counted per leaf."""
    return sum(int(leaf.size * leaf.dtype.itemsize) for leaf in jax.tree.leaves(tree))


def _public_scan(system: System, opponent: System, steps: int) -> Callable[..., Tree]:
    """Include public input preparation, both teams and action/memory packaging.

    system supplies Team A's actor; opponent is the existing random System.
    steps fixes scan length. The returned function accepts dynamic actor weights,
    compact observations, environment snapshot, System memory and lane keys.
    It retains joint actions, Team A learning rows and final System memory.
    These fixed-input calls perform no simulator transition or episode reset.
    """

    def run(
        parameters: Tree,
        observations: Tree,
        state: Tree,
        memory: Tree,
        keys: Array,
    ) -> Tree:
        """Repeat the public call while retaining one compact input snapshot."""

        def step(carry: Tree, tick: Array) -> tuple[Tree, Tree]:
            """Prepare permitted actor rows inside each public decision call."""
            before = state._replace(episode_start=state.episode_start & (tick == 0))

            def choose_key(key: Array) -> Array:
                """Supply one newly tagged public action root for each lane."""
                return jax.random.fold_in(key, tick)

            actions, following, learning = marl_bgs.apply_systems(
                system,
                opponent,
                carry,
                observations,
                before,
                jax.vmap(choose_key)(keys),
                variables_a=parameters,
            )
            return following, (actions, learning[0])

        return jax.lax.scan(step, memory, jnp.arange(steps, dtype=jnp.int32))

    return run


def _compile(function: Callable[..., Tree], arguments: tuple[Tree, ...]) -> Tree:
    """Compile one dynamic workload and capture callback and cache evidence.

    function consumes arguments, which are arrays or numerical PyTrees.
    Return its callable, trace counter and compile report. Compilation timing
    includes lowering. Host callback markers in the lowered program fail the run.
    """
    traces = [0]

    def counted(*values: Tree) -> Tree:
        """Count outer traces while passing every numerical value dynamically."""
        traces[0] += 1
        return function(*values)

    compiled = jax.jit(counted)
    started = time.perf_counter()
    lowered = compiled.lower(*arguments)
    executable = lowered.compile()
    elapsed = time.perf_counter() - started
    lowered_text = lowered.as_text()
    for marker in ("python_callback", "host_callback", "io_callback", "xla_ffi_python"):
        if marker in lowered_text:
            raise AssertionError(f"Unexpected host callback: {marker}")
    memory = executable.memory_analysis()
    report = {
        "compile_seconds": elapsed,
        "lowered_program_sha256": sha256(lowered_text.encode()).hexdigest(),
        "host_callback_markers": False,
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
    return compiled, traces, report


def _timed(
    function: Callable[..., Tree], arguments: tuple[Tree, ...]
) -> tuple[Tree, float]:
    """Run one call, wait for all outputs and return outputs plus milliseconds."""
    started = time.perf_counter()
    result = cast(Tree, jax.block_until_ready(function(*arguments)))
    return result, (time.perf_counter() - started) * 1000


def _compare(first: Tree, second: Tree) -> float:
    """Check matching outputs and return maximum absolute float error.

    Integer actions must match exactly. Float memory/log probabilities use
    atol=rtol=0.00001 to allow equivalent compiled operation ordering. This host
    check happens outside timing. Mismatches raise AssertionError.
    """
    assert jax.tree.structure(first) == jax.tree.structure(second)
    maximum = 0.0
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        left, right = _portable_array(a), _portable_array(b)
        if np.issubdtype(left.dtype, np.floating):
            np.testing.assert_allclose(left, right, atol=1e-5, rtol=1e-5)
            maximum = max(maximum, float(np.max(np.abs(left - right))))
        else:
            np.testing.assert_array_equal(left, right)
    return maximum


def _portable_array(value: Array) -> np.ndarray:
    """Transfer one array, preserving typed random keys as their uint32 words."""
    if jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
        value = jax.random.key_data(value)
    return np.asarray(value)


def main() -> None:
    """Measure the declared actor workload and save a reproducible evidence record.

    Parse --steps, --repeats, --output and optional --cpu-smoke. The output
    directory is created if absent; existing evidence files are rejected.
    Require CUDA unless CPU smoke was explicitly requested. No environment step,
    optimizer update, source edit or external communication takes place.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=(16, 128), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-smoke", action="store_true")
    options = parser.parse_args()
    if options.repeats < 5:
        parser.error("At least five warmed samples are required.")
    options.output.mkdir(parents=True, exist_ok=True)
    destination = options.output / "measurement.json"
    if destination.exists() or (options.output / "outputs.npz").exists():
        parser.error("The output directory already contains measurement evidence.")
    jax.config.update("jax_enable_compilation_cache", False)
    backend = jax.default_backend()
    if backend != "gpu" and not options.cpu_smoke:
        parser.error("CUDA is required; use --cpu-smoke only for CPU correctness.")
    devices = cast(list[Tree], jax.devices())
    device = devices[0]
    if len(devices) != 1:
        parser.error("Select exactly one device with CUDA_VISIBLE_DEVICES.")
    root = Path(__file__).resolve().parents[2]
    source_paths = (
        "src/marl_battlegrounds/baselines/__init__.py",
        "src/marl_battlegrounds/baselines/inputs.py",
        "src/marl_battlegrounds/baselines/actions.py",
        "src/marl_battlegrounds/baselines/ppo.py",
        "scripts/dev/benchmark_baseline.py",
        "uv.lock",
    )
    hashes = {
        path: sha256((root / path).read_bytes()).hexdigest() for path in source_paths
    }
    setup_started = time.perf_counter()
    weights = ppo.initialize_ppo(jax.random.key(42)).actor_params
    system = ppo.make_recurrent_mappo_system(weights, spawn_frame="left")
    env = marl_bgs.make("tdm", map_id=0, num_envs=_BATCH, metrics="none")
    observations, state = env.reset(jax.random.key(43))
    inputs = env.policy_inputs(observations, state)
    keys = jax.random.split(jax.random.key(44), _BATCH)
    assert system.init is not None
    memory = system.init(weights, inputs, keys)
    arguments = (weights, memory, inputs, keys)
    jax.block_until_ready(arguments)
    setup_seconds = time.perf_counter() - setup_started
    transfer_started = time.perf_counter()
    host_arguments = jax.device_get(arguments)
    host_transfer_ms = (time.perf_counter() - transfer_started) * 1000
    transfer_started = time.perf_counter()
    arguments = cast(
        tuple[Tree, ...],
        jax.block_until_ready(jax.device_put(host_arguments, device)),
    )
    device_transfer_ms = (time.perf_counter() - transfer_started) * 1000
    del host_arguments
    changed_observation = inputs.actors.observation._replace(
        context_features=inputs.actors.observation.context_features.at[
            ..., CONTEXT_FEATURE_CURRENT_TIMESTEP
        ].add(1.0)
    )
    changed_inputs = inputs._replace(
        actors=inputs.actors._replace(observation=changed_observation),
        valid=inputs.valid.at[0].set(False),
        episode_start=~inputs.episode_start,
    )

    def change_weight(value: Array) -> Array:
        """Change ordinary parameter values while preserving their shapes."""
        return value + jnp.float32(0.00001)

    def change_key(key: Array) -> Array:
        """Select another same-shaped lane stream for the reuse proof."""
        return jax.random.fold_in(key, 99)

    changed = (
        jax.tree.map(change_weight, weights),
        memory + jnp.float32(0.1),
        changed_inputs,
        jax.vmap(change_key)(keys),
    )
    jax.block_until_ready(changed)
    opponent = marl_bgs.shared_policy(marl_bgs.policy("random"))
    methods = marl_bgs.init_systems(system, opponent, observations, state, keys)
    direct_system = System("Direct Actor Reference", _direct_actor, init=system.init)
    public_arguments = (weights, observations, state, methods, keys)
    changed_compact = observations._replace(
        observation=observations.observation._replace(
            context_features=observations.observation.context_features.at[
                ..., CONTEXT_FEATURE_CURRENT_TIMESTEP
            ].add(1.0)
        )
    )
    public_changed = (
        changed[0],
        changed_compact,
        state,
        methods._replace(team_a=memory + jnp.float32(0.1)),
        changed[3],
    )
    jax.block_until_ready((public_arguments, public_changed))
    paths = {
        "system": _scan(cast(ActorCall, system.apply), options.steps),
        "direct": _scan(_direct_actor, options.steps),
        "public_system": _public_scan(system, opponent, options.steps),
        "public_direct": _public_scan(direct_system, opponent, options.steps),
    }
    arguments_by_path = {
        name: public_arguments if name.startswith("public_") else arguments
        for name in paths
    }
    compiled = {
        name: _compile(function, arguments_by_path[name])
        for name, function in paths.items()
    }
    reports: dict[str, Tree] = {}
    results: dict[str, Tree] = {}
    for name, (function, _, report) in compiled.items():
        results[name], first_ms = _timed(function, arguments_by_path[name])
        reports[name] = dict(report, first_execution_ms=first_ms, warm_samples_ms=[])
    maximum = _compare(results["system"], results["direct"])
    public_maximum = _compare(results["public_system"], results["public_direct"])
    with jax.transfer_guard("disallow_explicit"):
        for repeat in range(options.repeats):
            order = tuple(paths) if repeat % 2 == 0 else tuple(reversed(paths))
            for name in order:
                results[name], elapsed = _timed(
                    compiled[name][0], arguments_by_path[name]
                )
                reports[name]["warm_samples_ms"].append(elapsed)
    changed_results: dict[str, Tree] = {}
    for name, (function, traces, _) in compiled.items():
        before = traces[0]
        probe = public_changed if name.startswith("public_") else changed
        changed_results[name] = jax.block_until_ready(function(*probe))
        assert traces[0] == before == 1
        samples = reports[name]["warm_samples_ms"]
        median = statistics.median(samples)
        reports[name].update(
            warm_median_ms=median,
            warm_min_ms=min(samples),
            warm_max_ms=max(samples),
            learner_actor_decisions_per_second=(
                _BATCH * 5 * options.steps / (median / 1000)
            ),
            actors_per_game=10 if name.startswith("public_") else 5,
            input_bytes=_bytes(arguments_by_path[name]),
            retained_output_bytes=_bytes(results[name]),
            outer_traces=traces[0],
            changed_value_new_traces=traces[0] - before,
        )
    changed_maximum = _compare(changed_results["system"], changed_results["direct"])
    public_changed_maximum = _compare(
        changed_results["public_system"], changed_results["public_direct"]
    )
    assert hashes == {
        path: sha256((root / path).read_bytes()).hexdigest() for path in source_paths
    }, "Measured source changed during this run."
    memory_stats = device.memory_stats()
    retained = jax.device_get(results["system"])
    np.savez(options.output / "outputs.npz", *jax.tree.leaves(retained))
    public_retained = jax.device_get(results["public_system"])
    np.savez(
        options.output / "public-outputs.npz",
        *(_portable_array(value) for value in jax.tree.leaves(public_retained)),
    )
    result: dict[str, Tree] = {
        "backend": backend,
        "device": str(device),
        "selected_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_version": jax.__version__,
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("jaxlib", "flax", "optax", "numpy")
        },
        "persistent_compilation_cache": False,
        "matmul_precision": cast(Tree, jax.config).jax_default_matmul_precision,
        "gpu_preallocation": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": hashes,
        "batch": _BATCH,
        "steps": options.steps,
        "actors_per_game": 5,
        "spawn_frame": "left",
        "seed": 42,
        "setup_seconds": setup_seconds,
        "setup_includes": (
            "Network/optimizer initialization and B32 environment reset compilation"
        ),
        "setup_snapshot_device_to_host_ms": host_transfer_ms,
        "setup_snapshot_host_to_device_ms": device_transfer_ms,
        "input_bytes": _bytes(arguments),
        "actor_parameter_bytes": _bytes(weights),
        "expanded_input_snapshot_bytes": _bytes(inputs),
        "final_carry_bytes": _bytes(retained[0]),
        "retained_output_bytes": _bytes(retained),
        "saved_output_bytes": (options.output / "outputs.npz").stat().st_size,
        "max_absolute_output_error": maximum,
        "max_absolute_changed_value_output_error": changed_maximum,
        "max_absolute_public_output_error": public_maximum,
        "max_absolute_public_changed_value_output_error": public_changed_maximum,
        "warm_transfer_guard": (
            "Passed: no explicit host/device transfers in timed calls"
        ),
        "paths": reports,
        "system_minus_direct_median_ms": reports["system"]["warm_median_ms"]
        - reports["direct"]["warm_median_ms"],
        "public_system_minus_direct_median_ms": reports["public_system"][
            "warm_median_ms"
        ]
        - reports["public_direct"]["warm_median_ms"],
        "device_allocator_memory": memory_stats,
        "process_peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "limits": [
            "Fixed inputs may let compilation move encoding outside the scan.",
            "Counts are actor decisions, not simulator steps or learner updates.",
            "Public cases include the random opponent and M8 input/action handling.",
            "No full-time expanded model-input trajectory is retained.",
            "Allocator high water includes setup, both comparisons and compilation.",
            "Outer trace count is not a count of every nested backend program.",
            "CPU smoke timings do not establish GPU speed.",
        ],
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
