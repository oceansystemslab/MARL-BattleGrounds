"""Measure untrained MAPPO collection and a separately arranged public-call loop.

Run ``python -m scripts.dev.benchmark_training_collection --case no-reset
--opponents current --gpu-uuid GPU-... --output DIRECTORY`` on the verified
internal RTX 5090. GPU work is fixed at B32/T128 and five synchronized warm
samples. --cpu-smoke uses B2/T4 only to check this tool. Use --case continuation
for real native-horizon endings, or reset-stage-stress for explicitly synthetic
short horizons and curriculum boundaries. --opponents mixed publishes frozen
untrained variables after real untimed experience, then uses reset-time draws.

--recording adds one fresh bounded writer case to current/no-reset only. It
reports drains separately, including their existing physical validation cost.
The reference uses public choose/step/track calls, existing reset and schedule
components, and its own row packing. It never calls the training step or scan.
Private helpers are used for numerical guard/padding contracts and instrumentation.
No optimizer update, learner checkpoint, competence claim or GPU selection is
performed by this script. Existing output directories are never overwritten.
"""

from __future__ import annotations

# This evidence tool checks owned private guards, padding and writer drains.
# pyright: reportPrivateUsage=false
import argparse
import importlib.metadata
import json
import os
import resource
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines.inputs import encode_training_state
from marl_battlegrounds.baselines.ppo import initialize_ppo, make_recurrent_mappo_system
from marl_battlegrounds.core.types import ActionMask, EnvConfig
from marl_battlegrounds.episode_tracking import (
    init_episode_tracking,
    track_episode_step,
)
from marl_battlegrounds.evaluation.collection_types import CollectedBatch
from marl_battlegrounds.evaluation.episode_metrics import MetricValues
from marl_battlegrounds.evaluation.policy_execution import (
    apply_systems,
    init_systems,
    system_inputs,
)
from marl_battlegrounds.evaluation.recording_identity import (
    ordered_source_bank_identity,
)
from marl_battlegrounds.training._content import (
    PreparedTrainingContent,
    prepare_training_content,
)
from marl_battlegrounds.training._execution import _reset_finished
from marl_battlegrounds.training.collection import (
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    TrainingTransition,
    _failed,
    _padding,
    advance_training_step,
    collect_training_rollout,
    init_training_collection,
    scan_training_rollout,
    training_summary,
)
from marl_battlegrounds.training.curriculum import (
    _advance_training_schedule,
    make_training_schedule,
)
from marl_battlegrounds.training.distributions import (
    sample_training_configs,
    training_keys,
)
from marl_battlegrounds.training.opponents import (
    init_opponent_history,
    refresh_opponents,
)
from marl_battlegrounds.training.shaping import team_potential_shaping

type Tree = Any


def _bytes(tree: Tree) -> int:
    """Count logical numerical bytes; shared leaves can be counted more than once."""
    return sum(int(x.size * x.dtype.itemsize) for x in jax.tree.leaves(tree))


def _array(value: Tree) -> np.ndarray:
    """Copy one leaf to the host, exposing typed random keys as uint32 words."""
    if jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
        value = jax.random.key_data(value)
    return np.asarray(value)


def _equal(actual: Tree, expected: Tree) -> None:
    """Require equal numerical structures and exact values outside measured calls."""
    if jax.tree.structure(actual) != jax.tree.structure(expected):
        raise AssertionError("Reference and collector structures differ")
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(_array(left), _array(right))


def _timed(function: Callable[..., Tree], *values: Tree) -> tuple[Tree, float]:
    """Wait for the whole result and return it with elapsed wall-clock seconds."""
    started = time.perf_counter()
    result = cast(Tree, jax.block_until_ready(function(*values)))
    return result, time.perf_counter() - started


def _prepare_content() -> PreparedTrainingContent:
    """Wait for all prepared source arrays before ending the setup measurement."""
    prepared = prepare_training_content()
    jax.block_until_ready(prepared.source_configs)
    return prepared


def _initialize_actor(seed: int) -> Tree:
    """Wait for the shared untrained initializer, then retain only actor variables.

    The shared baseline initializer also creates critic/optimizer state. Waiting
    for it here keeps that setup work outside every collection measurement.
    """
    networks = initialize_ppo(jax.random.key(seed))
    jax.block_until_ready(networks)
    return networks.actor_params


def _compile(
    function: Callable[..., Tree],
    value: Tree,
    *,
    compiler_options: dict[str, Tree] | None = None,
) -> tuple[Tree, list[int], dict[str, Tree]]:
    """Separate lowering/compilation and inspect temporary bytes and callbacks.

    The returned callable counts Python traces without adding device callbacks.
    memory_analysis describes this executable, not live process-wide allocation.
    compiler_options passes optional settings to this outer JIT. None preserves
    JAX's current defaults; this helper does not change global compiler settings.
    """
    traces: list[int] = []

    def counted(argument: Tree) -> Tree:
        """Count only retracing; all changing arrays remain explicit arguments."""
        traces.append(1)
        return function(argument)

    compiled = jax.jit(counted, compiler_options=compiler_options)
    started = time.perf_counter()
    lowered = compiled.lower(value)
    lowering = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compilation = time.perf_counter() - started
    program = lowered.as_text()
    for marker in ("python_callback", "host_callback", "io_callback", "xla_ffi_python"):
        if marker in program:
            raise AssertionError(f"Unexpected host callback: {marker}")
    analysis = executable.memory_analysis()
    return (
        compiled,
        traces,
        {
            "lowering_seconds": lowering,
            "compilation_seconds": compilation,
            "lowered_sha256": sha256(program.encode()).hexdigest(),
            "host_callbacks": False,
            "compiler_memory_bytes": None
            if analysis is None
            else {
                field: int(getattr(analysis, field))
                for field in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "alias_size_in_bytes",
                )
            },
        },
    )


def _team_mask(state: Tree) -> ActionMask:
    """Retain Team A's five same-epoch native action masks in the reference."""

    def select(value: Array) -> Array:
        """Keep the Team A actor prefix."""
        return value[:, :5]

    return cast(ActionMask, jax.tree.map(select, state.action_mask))


def _manual_reset(carry: TrainingCarry) -> TrainingCarry:
    """Reset pending lanes through the shared reset component before choosing."""
    from marl_battlegrounds.training.opponents import assign_opponents

    mask = carry.state.done.done
    stage = carry.tracking.stage_ordinal
    assert carry.tracking.source_configs is not None
    observations, state, indices, classes = _reset_finished(
        carry.env,
        carry.observations,
        carry.state,
        carry.source_indices,
        carry.source_class_ids,
        source_configs=carry.tracking.source_configs,
        root_key=carry.root_key,
        eligible_maps=carry.schedule.eligible_maps[stage],
        team_size=carry.schedule.team_sizes[stage],
    )
    return carry._replace(
        observations=observations,
        state=state,
        source_indices=indices,
        source_class_ids=classes,
        progress=carry.progress._replace(
            episode_stage=jnp.where(mask, stage, carry.progress.episode_stage)
        ),
        history=assign_opponents(
            carry.history,
            mask,
            training_keys(carry.root_key, state.reset_generation, stream="opponent"),
        ),
    )


def _manual_step(
    collection: TrainingCollection, carry: TrainingCarry
) -> tuple[TrainingCarry, TrainingTransition]:
    """Independently arrange public action, simulator and tracking calls once.

    This benchmark reference admits only valid, unexhausted inputs. It shares
    existing numerical reset, stage and shaping rules, but neither training's
    action/step helper nor its step/scan or row-building helper is called.
    """

    def retain(value: TrainingCarry) -> TrainingCarry:
        """Preserve the exact pending-free boundary without reset work."""
        return value

    carry = cast(
        TrainingCarry,
        jax.lax.cond(jnp.any(carry.state.done.done), _manual_reset, retain, carry),
    )
    before = carry.state
    local_step = before.core_state.step_count - before.initial_step_count
    actions, memory, learning = apply_systems(
        collection.actor,
        collection.opponent,
        carry.memory,
        carry.observations,
        before,
        training_keys(
            carry.root_key,
            before.reset_generation,
            stream="action",
            decision_step=local_step,
        ),
        variables_a=carry.history.current_variables,
        variables_b=carry.history,
    )
    tracking, result = track_episode_step(
        carry.tracking,
        before,
        carry.env.step(
            training_keys(
                carry.root_key,
                before.reset_generation,
                stream="step",
                decision_step=local_step,
            ),
            before,
            actions,
        ),
        source_indices=carry.source_indices,
        source_class_ids=carry.source_class_ids,
    )
    observations, state, rewards, _, info = result
    progress, tracking, info = _advance_training_schedule(
        carry.progress, tracking, before, state, info, schedule=carry.schedule
    )
    valid = info.decision_step >= 0
    ended = info.completed & valid
    lanes = jnp.arange(before.episode_id.shape[0])
    slots = carry.history.lane_snapshot + 1
    progress = progress._replace(
        map_steps=progress.map_steps.at[carry.source_indices, lanes].add(
            valid.astype(jnp.int32)
        ),
        opponent_starts=progress.opponent_starts.at[slots, lanes].add(
            (valid & before.episode_start).astype(jnp.int32)
        ),
        opponent_steps=progress.opponent_steps.at[slots, lanes].add(
            valid.astype(jnp.int32)
        ),
    )
    priority = info.priority
    if priority is not None:
        available = priority.valid & ended[:, None]
        priority = MetricValues(jnp.where(available, priority.values, 0), available)
    opponent_update = jnp.where(
        carry.history.lane_snapshot == -1,
        carry.history.current_update,
        carry.history.captured_updates[jnp.maximum(carry.history.lane_snapshot, 0)],
    )
    row = TrainingTransition(
        observations=carry.observations,
        action_mask=_team_mask(before),
        actions=actions,
        learning_outputs=learning[0],
        task_rewards=rewards.rewards[:, :5],
        shaping_reward=(
            team_potential_shaping(
                before.core_state.team_deathmatch_scores,
                info,
                discount=carry.discount,
                coefficient=carry.coefficient,
            )[:, 0]
            if collection.shaping
            else jnp.zeros_like(rewards.rewards[:, 0])
        ),
        active=before.config.agent_profile.active_mask[:, :5],
        alive=before.core_state.alive_mask[:, :5],
        episode_start=before.episode_start,
        ended=ended,
        valid=valid,
        outcome=jnp.where(ended, info.outcome, 0),
        episode_length=jnp.where(ended, info.episode_length, 0),
        final_scores=jnp.where(ended[:, None], info.team_scores, 0),
        priority=priority,
        episode_id=info.episode_id,
        decision_step=info.decision_step,
        requested_stage=jnp.full_like(info.episode_id, carry.tracking.stage_ordinal),
        episode_stage=carry.progress.episode_stage,
        source_index=carry.source_indices,
        learner_update=jnp.full_like(info.episode_id, carry.history.current_update),
        opponent_update=opponent_update,
        opponent_snapshot=carry.history.lane_snapshot,
        training_state=(
            encode_training_state(before.core_state, before.config)
            if collection.collect_training_state
            else None
        ),
    )
    memory = memory._replace(
        policy_trace=memory.policy_trace._replace(
            policy_ids=jnp.full_like(memory.policy_trace.policy_ids, -1)
        )
    )
    return carry._replace(
        observations=observations,
        state=state,
        memory=memory,
        progress=progress,
        tracking=tracking,
    ), row


def _reference(
    collection: TrainingCollection, initial: TrainingCarry, length: int
) -> tuple[TrainingCarry, TrainingRollout]:
    """Scan independently arranged decisions and retain exactly the same outputs.

    Padding and guard values reuse owned contracts; the real decision body and
    final epoch packing remain separate from production training orchestration.
    """

    def step(
        carry: TrainingCarry, _unused: None
    ) -> tuple[TrainingCarry, TrainingTransition]:
        """Choose either a complete real decision or an untouched padding row."""

        def real(value: TrainingCarry) -> tuple[TrainingCarry, TrainingTransition]:
            """Enter only the independent public-call reference body."""
            return _manual_step(collection, value)

        def pad(value: TrainingCarry) -> tuple[TrainingCarry, TrainingTransition]:
            """Reuse the declared neutral computational padding structure."""
            return value, _padding(collection, value)

        return cast(
            tuple[TrainingCarry, TrainingTransition],
            jax.lax.cond(
                (carry.progress.rounds < carry.schedule.total_rounds) & ~_failed(carry),
                real,
                pad,
                carry,
            ),
        )

    carry, rows = jax.lax.scan(step, initial, None, length=length)
    return carry, TrainingRollout(
        transitions=rows,
        initial_memory=initial.memory.team_a,
        final_observations=carry.observations,
        final_action_mask=_team_mask(carry.state),
        final_active=carry.state.config.agent_profile.active_mask[:, :5],
        final_alive=carry.state.core_state.alive_mask[:, :5],
        final_ended=carry.state.done.done,
        final_training_state=(
            encode_training_state(carry.state.core_state, carry.state.config)
            if collection.collect_training_state
            else None
        ),
        real_steps=carry.progress.rounds - initial.progress.rounds,
    )


def _synthetic_stress(
    collection: TrainingCollection, carry: TrainingCarry
) -> TrainingCarry:
    """Build declared short-horizon benchmark data outside public training setup.

    This test-only mutation does not change the verified content binding. The
    report labels it synthetic and saves the actual changed source-bank digest.
    It rebuilds fresh state/tracking/memory before any benchmark decision.
    """
    assert carry.tracking.source_configs is not None
    bank = carry.tracking.source_configs._replace(
        max_steps=2 + jnp.arange(42, dtype=jnp.int32) % 3
    )
    generation = jnp.zeros_like(carry.state.reset_generation)
    sampled = sample_training_configs(
        bank,
        carry.root_key,
        generation,
        eligible_maps=carry.schedule.eligible_maps[0],
        team_size=carry.schedule.team_sizes[0],
    )
    observations, state = carry.env.reset(
        training_keys(carry.root_key, generation, stream="reset"), sampled.config
    )
    tracking = init_episode_tracking(
        carry.env,
        state,
        source_configs=bank,
        source_indices=sampled.source_indices,
        source_class_ids=sampled.source_class_ids,
        record_starts=False,
    ).begin_stage(
        state,
        total_env_steps=int(carry.schedule.round_budgets[0])
        * collection.schedule.num_envs,
    )
    memory = init_systems(
        collection.actor,
        collection.opponent,
        observations,
        state,
        training_keys(carry.root_key, generation, stream="initialization"),
        variables_a=carry.history.current_variables,
        variables_b=carry.history,
    )
    return carry._replace(
        observations=observations,
        state=state,
        tracking=tracking,
        memory=memory,
        source_indices=sampled.source_indices,
        source_class_ids=sampled.source_class_ids,
    )


def _burnin(
    collection: TrainingCollection, carry: TrainingCarry, rounds: int
) -> TrainingCarry:
    """Advance genuine untimed experience without retaining a time-axis output tree."""

    def run(value: TrainingCarry, count: Array) -> TrainingCarry:
        """Use a compiled loop with a dynamic round count for boundary preparation."""

        def step(_index: Array, current: TrainingCarry) -> TrainingCarry:
            """Discard the valid row while retaining its actual next state."""
            return advance_training_step(collection, current)[0]

        return cast(TrainingCarry, jax.lax.fori_loop(0, count, step, value))

    return cast(TrainingCarry, jax.jit(run)(carry, jnp.int32(rounds)))


def _change_weights(leaf: Array) -> Array:
    """Make a small declared synthetic inference update without an optimizer."""
    return leaf + jnp.asarray(0.00001, leaf.dtype)


def _opponent_application_cost(
    collection: TrainingCollection, carry: TrainingCarry
) -> dict[str, Tree]:
    """Compare current and mapped Team B inference with identical inputs/weights.

    This bounded probe uses B32 once per call, not an environment rollout. Copy
    current variables into occupied history solely for this comparison, retaining
    the real mixed assignment. Hold observations, memory and keys fixed. Compare
    actions exactly and record floating memory differences. A highest-precision
    matrix-multiply diagnostic checks whether the two layouts agree within 1e-5.
    It does not change production precision or the timed default calls. Five
    synchronized samples per branch use one compiled callable and forbid transfer.
    Temporary compiler bytes cover both branches; they are not a measured peak
    of only the mixed branch. No new snapshot, game or update is produced.
    """

    def repeat(leaf: Array) -> Array:
        """Give every probe snapshot the same current inference values."""
        return jnp.broadcast_to(leaf, (20, *leaf.shape))

    history = carry.history._replace(
        historical_variables=jax.tree.map(repeat, carry.history.current_variables)
    )
    if not bool(jnp.any(history.lane_snapshot >= 0)):
        raise AssertionError("The inference probe requires a real mixed assignment")
    inputs = system_inputs(carry.observations, carry.state, team=1)
    keys = training_keys(
        carry.root_key,
        carry.state.reset_generation,
        stream="action",
        decision_step=carry.state.core_state.step_count
        - carry.state.initial_step_count,
    )
    mixed = (history, carry.memory.team_b, inputs, keys)
    current = (
        history._replace(lane_snapshot=jnp.full_like(history.lane_snapshot, -1)),
        *mixed[1:],
    )

    def apply(values: Tree) -> Tree:
        """Keep parameters, inputs, memory, keys and assignment dynamic."""
        return collection.opponent.apply(*values)

    compiled, traces, report = _compile(apply, current)
    left, report["current_first_seconds"] = _timed(compiled, current)
    right, report["mixed_first_seconds"] = _timed(compiled, mixed)
    memory_differences: list[float] = []
    memory_rms_differences: list[float] = []
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jnp.inexact):
            host_a, host_b = np.asarray(a), np.asarray(b)
            if not np.all(np.isfinite(host_a)) or not np.all(np.isfinite(host_b)):
                raise AssertionError("Opponent inference produced nonfinite memory")
            memory_differences.append(float(np.max(np.abs(host_a - host_b))))
            memory_rms_differences.append(
                float(np.sqrt(np.mean((host_a.astype(np.float64) - host_b) ** 2)))
            )
        else:
            np.testing.assert_array_equal(a, b)
    samples: dict[str, list[float]] = {"current": [], "mixed": []}
    with jax.transfer_guard("disallow_explicit"):
        for _ in range(5):
            for name, value in (("current", current), ("mixed", mixed)):
                _, seconds = _timed(compiled, value)
                samples[name].append(seconds)
    if traces != [1]:
        raise AssertionError("Opponent assignment changed the compiled program")
    with jax.default_matmul_precision("highest"):
        accurate, _, precision_report = _compile(apply, current)
        accurate_a, _ = _timed(accurate, current)
        accurate_b, _ = _timed(accurate, mixed)
    accurate_differences: list[float] = []
    for a, b in zip(
        jax.tree.leaves(accurate_a), jax.tree.leaves(accurate_b), strict=True
    ):
        if jnp.issubdtype(a.dtype, jnp.inexact):
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
            accurate_differences.append(
                float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            )
        else:
            np.testing.assert_array_equal(a, b)
    precision_report.update(
        maximum_memory_difference=max(accurate_differences, default=0.0),
        comparison_atol=1e-5,
        comparison_rtol=1e-5,
        scope="Diagnostic only; production and timed calls keep default precision",
    )
    report.update(
        warm_seconds=samples,
        median_seconds={name: statistics.median(row) for name, row in samples.items()},
        matched_inputs_memory_keys_and_weights=True,
        maximum_default_memory_difference=max(memory_differences, default=0.0),
        maximum_leaf_rms_memory_difference=max(memory_rms_differences, default=0.0),
        highest_precision_diagnostic=precision_report,
        exact_actions=True,
        selected_lane_parameter_logical_upper_bound_bytes=(
            collection.schedule.num_envs * _bytes(history.current_variables)
        ),
        temporary_limit=(
            "Compiler may fuse selected leaves; logical bound is not live allocation"
        ),
    )
    return report


def _prepare_case(
    collection: TrainingCollection, carry: TrainingCarry, *, case: str, opponents: str
) -> tuple[TrainingCarry, dict[str, Tree]]:
    """Prepare real episode/update boundaries and report their untimed cost.

    Mixed history is created only after actual collected rounds. Continuing
    games retain their assignments; only natural resets draw frozen versions.
    No fake future-round history or learner update is introduced.
    """
    if case == "reset-stage-stress":
        carry = _synthetic_stress(collection, carry)
    before = int(carry.progress.rounds)
    started = time.perf_counter()
    if opponents == "mixed":
        horizon = int(np.max(np.asarray(carry.state.config.max_steps)))
        first = (
            int(carry.schedule.history_threshold_rounds[0])
            if case == "reset-stage-stress"
            else horizon - 1
        )
        carry = _burnin(collection, carry, first)
        history, event = refresh_opponents(
            carry.history,
            carry.history.current_variables,
            completed_rounds=carry.progress.rounds,
            update_index=jnp.int32(1),
            schedule=carry.schedule,
        )
        if not bool(event.created):
            raise AssertionError("Preparation did not cross a snapshot threshold")
        carry = _burnin(collection, carry._replace(history=history), 1)
        history, _ = refresh_opponents(
            carry.history,
            jax.tree.map(_change_weights, carry.history.current_variables),
            completed_rounds=carry.progress.rounds,
            update_index=jnp.int32(2),
            schedule=carry.schedule,
        )
        carry = carry._replace(history=history)
        if case == "no-reset":
            carry = _burnin(collection, carry, 1)
    elif case == "continuation":
        carry = _burnin(
            collection, carry, int(np.max(np.asarray(carry.state.config.max_steps)))
        )
    jax.block_until_ready(carry)
    training_summary(collection, carry)
    return carry, {
        "seconds_including_compilation": time.perf_counter() - started,
        "real_rounds": int(carry.progress.rounds) - before,
        "snapshot_count": int(carry.history.count),
        "current_update": int(carry.history.current_update),
        "pending_finished_lanes": int(np.sum(np.asarray(carry.state.done.done))),
        "historical_lanes_before_measurement": int(
            np.sum(np.asarray(carry.history.lane_snapshot) >= 0)
        ),
        "update_meaning": (
            "Synthetic inference changes after real experience; "
            "no optimizer or learning"
        ),
    }


class _MemorySampler:
    """Sample this process's selected-GPU usage while measured work runs.

    Polling uses nvidia-smi every 0.2 seconds. The maximum is a sampled lower
    bound on the live peak, and polling can add host overhead. CPU mode records
    no GPU samples. Executable memory analysis and allocator peaks are separate.
    """

    def __init__(self, uuid: str | None) -> None:
        """Keep the explicitly selected GPU identity; allocate no device memory."""
        self.uuid = uuid
        self.samples: list[int] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        """Record only this PID's resident GPU bytes until collection is done."""
        while not self._stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.uuid}",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                self.errors.append(result.stderr.strip())
                return
            try:
                for line in result.stdout.splitlines():
                    pid, megabytes = (item.strip() for item in line.split(","))
                    if int(pid) == os.getpid():
                        self.samples.append(int(megabytes) * 1024 * 1024)
            except ValueError as error:
                self.errors.append(str(error))
                return
            self._stop.wait(0.2)

    def __enter__(self) -> _MemorySampler:
        """Start optional sampling after compilation, before first execution."""
        if self.uuid is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_error: object) -> None:
        """Stop polling and join its thread before writing measurement evidence."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def report(self) -> dict[str, Tree]:
        """Return explicitly scoped live-memory observations and polling limits."""
        return {
            "source": "nvidia-smi per-process polling every 0.2 seconds",
            "samples": len(self.samples),
            "sampled_peak_bytes": max(self.samples, default=None),
            "last_sample_bytes": self.samples[-1] if self.samples else None,
            "errors": self.errors,
        }


class _TimedWriter(marl_bgs.RunWriter):
    """Measure normal bounded drains without changing their validation or storage."""

    drain_seconds: float = 0.0
    drain_bytes: int = 0
    drain_count: int = 0

    def _write_collected(
        self, batch: object, *, source_configs: EnvConfig | None = None
    ) -> None:
        """Time occupied-record transfer, validation and writes without bypasses."""
        started = time.perf_counter()
        super()._write_collected(batch, source_configs=source_configs)
        self.drain_seconds += time.perf_counter() - started
        self.drain_bytes += _bytes(cast(CollectedBatch, batch))
        self.drain_count += 1


def _writing(
    directory: Path,
    prepared: PreparedTrainingContent,
    variables: Tree,
    *,
    schedule: Tree,
    seed: int,
    steps: int,
    expected: TrainingRollout,
) -> dict[str, Tree]:
    """Run one separately initialized recorded block, attaching its writer first."""
    actor = make_recurrent_mappo_system(variables)
    collection, carry = init_training_collection(
        actor,
        variables,
        prepared=prepared,
        schedule=schedule,
        seed=seed,
        recording=True,
    )
    with _TimedWriter(
        directory,
        phase="training",
        policies={
            "team_a": collection.actor,
            "team_b": collection.opponent,
        },
    ) as writer:

        def collect(value: TrainingCarry) -> tuple[TrainingCarry, TrainingRollout]:
            """Use the public bounded writer route with its ordinary capacity."""
            return collect_training_rollout(
                collection, value, length=steps, writer=writer
            )

        (final, output), seconds = _timed(collect, carry)
        _equal(output, expected)
        training_summary(collection, final)
        _, flush_seconds = _timed(writer.flush)
        return {
            "seconds_including_compilation_and_transfers": seconds,
            "drain_seconds_including_transfers_and_physical_validation": cast(
                _TimedWriter, writer
            ).drain_seconds,
            "drain_bytes": cast(_TimedWriter, writer).drain_bytes,
            "drain_count": cast(_TimedWriter, writer).drain_count,
            "flush_seconds": flush_seconds,
            "durable_bytes": sum(
                path.stat().st_size
                for path in writer.run_dir.rglob("*")
                if path.is_file()
            ),
            "exact_rollout_agreement": True,
        }


def main() -> None:
    """Save one fixed-size evidence case, refusing wrong devices or reused paths.

    --gpu-uuid must identify the verified internal GPU for actual measurement;
    the caller chooses CUDA_VISIBLE_DEVICES. --recording is deliberately limited
    to one fresh current/no-reset case. --cpu-smoke requires the CPU backend.
    Runtime/versions, source hashes, exact output checks and limits are saved.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("no-reset", "continuation", "reset-stage-stress"),
        required=True,
    )
    parser.add_argument("--opponents", choices=("current", "mixed"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--recording", action="store_true")
    options = parser.parse_args()
    if options.recording and (
        options.case != "no-reset" or options.opponents != "current"
    ):
        parser.error("--recording measures only the fresh current/no-reset case")
    if options.output.exists():
        parser.error("Existing evidence directories must be preserved")
    jax.config.update("jax_enable_compilation_cache", False)
    backend = jax.default_backend()
    if options.cpu_smoke and backend != "cpu":
        parser.error("--cpu-smoke requires JAX_PLATFORMS=cpu")
    if not options.cpu_smoke and (backend != "gpu" or not options.gpu_uuid):
        parser.error("Select the verified internal GPU and supply --gpu-uuid")
    devices = cast(list[Tree], jax.devices())
    if len(devices) != 1:
        parser.error("Expose exactly one device to this benchmark")
    device = devices[0]
    gpu_identity = None
    if backend == "gpu":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != options.gpu_uuid:
            parser.error(
                "CUDA_VISIBLE_DEVICES must equal the supplied verified GPU UUID"
            )
        gpu_identity = subprocess.run(
            [
                "nvidia-smi",
                f"--id={options.gpu_uuid}",
                "--query-gpu=name,uuid,pci.bus_id,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if "RTX 5090" not in gpu_identity or options.gpu_uuid not in gpu_identity:
            parser.error("The selected device is not the declared RTX 5090")
    options.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    paths = [
        *sorted((root / "src/marl_battlegrounds/training").glob("*.py")),
        *sorted((root / "src/marl_battlegrounds/baselines").glob("*.py")),
        root / "src/marl_battlegrounds/environment.py",
        root / "src/marl_battlegrounds/episode_tracking.py",
        root / "src/marl_battlegrounds/evaluation/policy_execution.py",
        root / "src/marl_battlegrounds/collection.py",
        root / "src/marl_battlegrounds/core/env.py",
        root / "uv.lock",
        Path(__file__).resolve(),
    ]
    identity = {
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
        "gpu": gpu_identity,
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
    }
    (options.output / "source_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    batch, steps = (2, 4) if options.cpu_smoke else (32, 128)
    prepared, preparation_seconds = _timed(_prepare_content)
    variables, model_seconds = _timed(_initialize_actor, options.seed)
    temporary_history, history_setup_seconds = _timed(
        lambda: init_opponent_history(variables, num_envs=batch)
    )
    del temporary_history
    actor = make_recurrent_mappo_system(variables)
    stress = options.case == "reset-stage-stress"
    total_rounds = (
        (40 if options.cpu_smoke else 256)
        if stress
        else 2 * int(np.max(np.asarray(prepared.source_configs.max_steps))) + steps
    )
    schedule = make_training_schedule(
        total_env_steps=batch * total_rounds, num_envs=batch, curriculum=stress
    )
    started = time.perf_counter()
    collection, carry = init_training_collection(
        actor, variables, schedule=schedule, prepared=prepared, seed=options.seed
    )
    jax.block_until_ready(carry)
    setup_seconds = time.perf_counter() - started
    carry, boundary_report = _prepare_case(
        collection, carry, case=options.case, opponents=options.opponents
    )
    host, to_host_seconds = _timed(jax.device_get, carry)

    def place(value: TrainingCarry) -> TrainingCarry:
        """Place the complete host carry back on the selected device."""
        return cast(TrainingCarry, jax.device_put(value, device))

    carry, placement_seconds = _timed(place, host)
    del host

    def production(value: TrainingCarry) -> tuple[TrainingCarry, TrainingRollout]:
        """Time the public pure collector, retaining its full declared result."""
        return scan_training_rollout(collection, value, length=steps)

    def reference(value: TrainingCarry) -> tuple[TrainingCarry, TrainingRollout]:
        """Time the independent public-call composition with identical outputs."""
        return _reference(collection, value, steps)

    compiled, traces, measured = _compile(production, carry)
    direct, direct_traces, reference_report = _compile(reference, carry)
    with _MemorySampler(options.gpu_uuid if backend == "gpu" else None) as sampler:
        actual, measured["first_seconds"] = _timed(compiled, carry)
        expected, reference_report["first_seconds"] = _timed(direct, carry)
        _equal(actual, expected)
        training_summary(collection, actual[0])
        samples: dict[str, list[float]] = {"collector": [], "reference": []}
        with jax.transfer_guard("disallow_explicit"):
            for repeat in range(5):
                order = (
                    ("collector", "reference")
                    if repeat % 2 == 0
                    else ("reference", "collector")
                )
                for name in order:
                    _, elapsed = _timed(
                        compiled if name == "collector" else direct, carry
                    )
                    samples[name].append(elapsed)
    real_transitions = int(actual[1].real_steps) * batch
    resets = int(
        np.sum(
            np.asarray(actual[0].state.reset_generation - carry.state.reset_generation)
        )
    )
    historical_rows = int(
        np.sum(np.asarray(actual[1].transitions.opponent_snapshot) >= 0)
    )
    if real_transitions != batch * steps:
        raise AssertionError(
            "Measured block did not contain the declared real experience"
        )
    if options.case == "no-reset" and resets:
        raise AssertionError("The no-reset workload unexpectedly reset a game")
    if options.case == "continuation" and not resets:
        raise AssertionError(
            "Continuation did not reset a completed native-horizon game"
        )
    if stress and (
        not resets
        or int(actual[0].tracking.stage_ordinal) <= int(carry.tracking.stage_ordinal)
    ):
        raise AssertionError("Stress did not exercise both reset and stage boundaries")
    if options.opponents == "mixed" and not 0 < historical_rows < batch * steps:
        raise AssertionError(
            "The selected seed did not produce a mixed measured workload"
        )
    for name, result in (("collector", measured), ("reference", reference_report)):
        values = samples[name]
        result.update(
            warm_seconds=values,
            median_seconds=statistics.median(values),
            minimum_seconds=min(values),
            maximum_seconds=max(values),
            spread_seconds=max(values) - min(values),
            real_transitions_per_second=real_transitions / statistics.median(values),
        )

    _, changed = init_training_collection(
        actor, variables, schedule=schedule, prepared=prepared, seed=options.seed + 1
    )
    if stress:
        changed = _synthetic_stress(collection, changed)
    changed = changed._replace(
        history=changed.history._replace(
            current_variables=jax.tree.map(
                _change_weights, changed.history.current_variables
            )
        )
    )
    changed_actual, _ = _timed(compiled, changed)
    changed_expected, _ = _timed(direct, changed)
    _equal(changed_actual, changed_expected)
    if traces != [1] or direct_traces != [1]:
        raise AssertionError(
            "Same-shaped changing values retraced a benchmark callable"
        )
    # Equality checks above populate host caches. Use a fresh execution whose
    # result has never been read on the host to measure a real first transfer.
    fresh_output, _ = _timed(compiled, carry)
    host_output, output_transfer_seconds = _timed(jax.device_get, fresh_output[1])
    del host_output
    del fresh_output
    writing = (
        _writing(
            options.output / "recording",
            prepared,
            variables,
            schedule=schedule,
            seed=options.seed,
            steps=steps,
            expected=actual[1],
        )
        if options.recording
        else None
    )
    needs_opponent_probe = (
        options.opponents == "mixed" and options.case == "continuation"
    )
    assert carry.tracking.source_configs is not None
    report = {
        "case": options.case,
        "opponents": options.opponents,
        "cpu_smoke_only": options.cpu_smoke,
        "backend": backend,
        "batch": batch,
        "steps": steps,
        "seed": options.seed,
        "synthetic_reset_stage_stress": stress,
        "metrics": "priority",
        "actor": "Untrained recurrent MAPPO; actor-only inference, no optimizer",
        "source_identity_file": "source_identity.json",
        "preparation_seconds": preparation_seconds,
        "model_initialization_seconds": model_seconds,
        "history_initialization_seconds": history_setup_seconds,
        "collection_setup_seconds": setup_seconds,
        "boundary_preparation": boundary_report,
        "setup_device_to_host_seconds": to_host_seconds,
        "placement_seconds": placement_seconds,
        "placed_logical_bytes": _bytes(carry),
        "current_actor_bytes": _bytes(carry.history.current_variables),
        "history_bank_bytes": _bytes(carry.history.historical_variables),
        "history_record_bytes": _bytes(carry.history),
        "real_transitions": real_transitions,
        "episode_resets": resets,
        "historical_transition_rows": historical_rows,
        "collector": measured,
        "reference": reference_report,
        "exact_full_output_agreement": True,
        "trace_counts_after_changed_values": [len(traces), len(direct_traces)],
        "changed_value_check": (
            "New keys, initial maps, observations and weights; initial memory "
            "templates are reused. Mixed mode also changes history count and "
            "assignments. Stress can change reset rosters inside the block."
        ),
        "optional_work": {
            "recording_disabled_in_timed_paths": True,
            "physical_state_output_absent": actual[1].transitions.training_state
            is None,
            "shaping_disabled": not collection.shaping,
            "host_callbacks_absent_in_both_lowered_programs": True,
            "actor_outputs_have_no_critic": True,
        },
        "warm_transfers": (
            "Implicit and explicit transfers disallowed around all ten timed warm calls"
        ),
        "rollout_output_bytes": _bytes(actual[1]),
        "returned_carry_bytes": _bytes(actual[0]),
        "output_device_to_host_seconds": output_transfer_seconds,
        "output_transfer_scope": "First host read of a fresh completed rollout",
        "writer": writing,
        "matched_opponent_application": None,
        "matched_opponent_probe_status": "pending"
        if needs_opponent_probe
        else "not_requested",
        "sampled_gpu_memory": sampler.report(),
        "device_allocator_statistics": device.memory_stats(),
        "peak_process_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "verified_content_digest": prepared.binding.canonical_digest,
        "actual_source_bank_digest": ordered_source_bank_identity(
            carry.tracking.source_configs
        )[0],
        "limits": (
            "One fixed batch/length, no training or sample-efficiency claim. "
            "Model setup also initializes then discards critic/optimizer arrays. "
            "Logical tree bytes may double-count shared leaves. Allocator/RAM "
            "peaks cover setup, burn-in, both compiled paths and writing; "
            "nvidia-smi is a sampled lower bound. Reference shares reset/schedule/"
            "shaping rules but independently arranges public choose-step-track "
            "and row epochs. No optimizer runs. Synthetic stress is not an "
            "approved training distribution."
        ),
    }
    (options.output / "measurement.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if needs_opponent_probe:
        report["matched_opponent_application"] = _opponent_application_cost(
            collection, actual[0]
        )
        report["matched_opponent_probe_status"] = "passed"
        (options.output / "measurement.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {
                name: report[name]
                for name in (
                    "case",
                    "opponents",
                    "real_transitions",
                    "episode_resets",
                    "collector",
                    "reference",
                    "writer",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
