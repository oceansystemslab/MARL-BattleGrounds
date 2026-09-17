"""Measure the bounded Packet 6 evaluator and result-access workflow.

Run through ``python -m scripts.dev.benchmark_evaluation --evaluation-access
--evaluation-case CASE --output DIRECTORY``. Cases are legacy, manual, systems,
mixed and capture. Each uses 64 complete 5v5 games, 32 lanes and 16-step chunks.
Alternating three/seven-step horizons expose refill and padding costs. Five warm
samples are mandatory. Use the same source-independent harness and frozen assets
for the committed legacy reference and the candidate. CPU is smoke evidence only.

The manual reference uses public reset/apply/step and the evaluator's declared
random-stream formula. It retains matching outcomes and priority measurements,
without a writer or a dense trajectory. The capture case measures optional saved
replays/full rows separately. This tool writes measurements and selected run files;
it does not change source, train a method or claim a maximum achievable speed.
"""

from __future__ import annotations

# Benchmark modes share private measurement helpers in their owning dispatcher.
# pyright: reportPrivateUsage=false
import argparse
import csv
import json
import resource
import statistics
import time
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.environment import Environment, EnvironmentState, EpisodeInfo
from marl_battlegrounds.evaluation.episode_metrics import MetricValues
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.metric_catalog import PRIORITY_METRIC_NAMES
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
    SystemState,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config
from scripts.dev.benchmark_evaluation import (
    _foundation_assets,
    _foundation_digest,
    _foundation_identity,
    _systems_host_measure,
)

type Tree = Any
CASES = ("legacy", "manual", "systems", "mixed", "capture")
_BATCH = 32
_CHUNK = 16
_GAMES = 64


class _Rows(NamedTuple):
    """Retain one game's exact outcome and priority measurements per lane.

    outcome and length are int32 (32,); scores is int32 (32,2) in team order.
    values is float32 (32,16) in catalog order and valid is its Boolean mask.
    The outer two-wave scan adds one leading axis. No per-step history survives.
    """

    outcome: Array
    length: Array
    scores: Array
    values: Array
    valid: Array


class _Carry(NamedTuple):
    """Keep one wave's observations, state, recurrent memory and terminal rows."""

    observations: Observations
    state: EnvironmentState
    memory: SystemState
    rows: _Rows


def _keys(seeds: Array, decisions: Array, stream: int) -> Array:
    """Apply the declared seed-42 stream/seed/local-decision formula independently."""
    root = jax.random.fold_in(jax.random.key(42), stream)

    def lane(seed: Array, decision: Array) -> Array:
        """Keep a scheduled episode's key independent of its lane and chunk."""
        return jax.random.fold_in(jax.random.fold_in(root, seed), decision)

    return jax.vmap(lane)(seeds, decisions)


def _actor(
    weights: PolicyTree,
    memory: PolicyTree,
    inputs: PolicyTree,
    masks: PolicyTree,
    key: Array,
) -> tuple[ActorAction, PolicyTree]:
    """Choose a masked random move; hold Basic/Ultimate at their idle category.

    weights is one float32 scalar and memory one int32 counter. The move depends
    on the supplied key, legal move mask and weights. Inputs are permitted but not
    needed by this fixed workload. Return the action and incremented memory.
    """
    del inputs
    logits = jnp.where(masks.move_mask, weights * jnp.arange(9), -jnp.inf)
    move = jax.random.categorical(key, logits).astype(jnp.int32)
    zero = jnp.asarray(0, jnp.int32)
    return ActorAction(move, zero, zero), memory + 1


def _init(weights: PolicyTree, inputs: SystemInput, keys: Array) -> Array:
    """Initialize B-by-5 recurrent counters without choosing an action."""
    del weights, keys
    return jnp.zeros(inputs.active_mask.shape, jnp.int32)


def _method(
    weights: PolicyTree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    """Produce each actor's move once and expose an unused learning array.

    Inputs remain separate by recipient. Fold actor-local slots into each team's
    B decision keys. The memory update depends on valid lanes. Evaluation must
    discard the already produced learning array inside its compiled boundary.
    """

    def lane(key: Array, masks: Tree, old: Array) -> tuple[ActorAction, Array]:
        """Apply the fixed scalar workload to five separately permitted actors."""
        actor_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
            key, jnp.arange(5, dtype=jnp.uint32)
        )
        return jax.vmap(_actor, in_axes=(None, 0, None, 0, 0))(
            weights, old, (), masks, actor_keys
        )

    actions, updated = jax.vmap(lane)(keys, inputs.action_mask, memory)
    return SystemOutput(
        actions,
        jnp.where(inputs.valid[:, None], updated, memory),
        learning_outputs=jnp.sin(jnp.arange(4096, dtype=jnp.float32) + weights),
        policy_ids=jnp.where(inputs.active_mask, 0, -1).astype(jnp.int32),
    )


_SYSTEM = System(
    "Benchmark Recurrent",
    _method,
    jnp.asarray(0.1, jnp.float32),
    init=_init,
    components=({"name": "fixed-move-method"},),
)


def _rows(info: EpisodeInfo, previous: _Rows) -> _Rows:
    """Retain only first terminal summaries; repeated padding cannot replace them."""
    metrics = cast(MetricValues, info.priority)
    return _Rows(
        jnp.where(info.completed, info.outcome, previous.outcome),
        jnp.where(info.completed, info.episode_length, previous.length),
        jnp.where(info.completed[:, None], info.team_scores, previous.scores),
        jnp.where(info.completed[:, None], metrics.values, previous.values),
        jnp.where(info.completed[:, None], metrics.valid, previous.valid),
    )


@jax.jit
def _manual(env: Environment, configs: Tree, weights: Array) -> _Rows:
    """Run two independently reset waves through the raw public numerical API.

    env, the (2,32,...) config bank and weights are dynamic. Return the same
    terminal outcome/priority payload as the evaluator. This reference has no
    recorded traces, writer, config hashing or result-table implementation. Its
    prepared-input numerical timing is reported separately from evaluator setup.
    """

    def wave(unused: None, item: Tree) -> tuple[None, _Rows]:
        """Reset 32 scheduled games, apply sixteen decisions and retain completions."""
        del unused
        config, seeds = item
        zeros = jnp.zeros(_BATCH, jnp.int32)
        obs, state = env.reset(
            _keys(seeds, zeros, 0), env_config=config, episode_id=seeds
        )
        memory = marl_bgs.init_systems(
            _SYSTEM,
            _SYSTEM,
            obs,
            state,
            _keys(seeds, zeros, 3),
            variables_a=weights,
            variables_b=weights,
        )
        empty = _Rows(
            zeros,
            zeros,
            jnp.zeros((_BATCH, 2), jnp.int32),
            jnp.zeros((_BATCH, len(PRIORITY_METRIC_NAMES)), jnp.float32),
            jnp.zeros((_BATCH, len(PRIORITY_METRIC_NAMES)), jnp.bool_),
        )

        def step(carry: _Carry, unused: None) -> tuple[_Carry, None]:
            """Choose one joint action and retain only its game's terminal values."""
            del unused
            local = carry.state.core_state.step_count - carry.state.initial_step_count
            actions, memory, _ = marl_bgs.apply_systems(
                _SYSTEM,
                _SYSTEM,
                carry.memory,
                carry.observations,
                carry.state,
                _keys(seeds, local, 2),
                variables_a=weights,
                variables_b=weights,
            )
            obs, state, _, _, info = env.step(
                _keys(seeds, local, 1), carry.state, actions
            )
            return _Carry(obs, state, memory, _rows(info, carry.rows)), None

        latest, _ = jax.lax.scan(
            step, _Carry(obs, state, memory, empty), None, length=_CHUNK
        )
        return None, latest.rows

    seeds = jnp.arange(1, _GAMES + 1, dtype=jnp.int32).reshape((2, _BATCH))
    return jax.lax.scan(wave, None, (configs, seeds))[1]


def _summary(result: Tree) -> dict[str, object]:
    """Project public result rows into the same ordered exact numerical payload."""
    episodes = sorted(result.episodes, key=lambda row: row.episode_id)
    table = result.priority_metrics
    values = (
        np.stack([table[name] for name in PRIORITY_METRIC_NAMES], axis=1)
        if table
        else None
    )
    return {
        "episode_ids": [row.episode_id for row in episodes],
        "outcome": [row.outcome for row in episodes],
        "length": [row.episode_length for row in episodes],
        "scores": [[row.team_a_score, row.team_b_score] for row in episodes],
        "priority": None if values is None else values.tolist(),
    }


def _manual_summary(rows: _Rows) -> dict[str, object]:
    """Flatten two waves after synchronization into public result ordering."""
    rows = cast(_Rows, jax.device_get(rows))
    return {
        "episode_ids": list(range(1, _GAMES + 1)),
        "outcome": rows.outcome.reshape(-1).tolist(),
        "length": rows.length.reshape(-1).tolist(),
        "scores": rows.scores.reshape((-1, 2)).tolist(),
        "priority": np.where(rows.valid, rows.values, np.nan)
        .reshape((_GAMES, -1))
        .tolist(),
    }


def _host_pair(calls: list[int]) -> tuple[System, System]:
    """Create one fake full-batch host provider with opaque lane memory and JAX foe.

    calls receives the actual batch size once per live decision. The fake host
    chooses idle actions and returns a private learning object that must not be
    serialized. This isolates host-boundary cost, without a network provider.
    """

    def initialize(weights: Tree, inputs: SystemInput, keys: Array) -> list[object]:
        """Create one opaque object per valid lane and no actions."""
        del weights, keys
        return [object() if valid else None for valid in inputs.valid]

    def apply(
        weights: Tree, memory: Tree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        """Keep the full stable batch and return one idle action for each actor."""
        del weights, keys
        calls.append(len(inputs.valid))
        zeros = np.zeros(inputs.active_mask.shape, np.int32)
        return SystemOutput(
            ActorAction(cast(Array, zeros), cast(Array, zeros), cast(Array, zeros)),
            memory,
            learning_outputs=object(),
            policy_ids=cast(
                Array, np.where(inputs.active_mask, 0, -1).astype(np.int32)
            ),
        )

    return System(
        "Benchmark Host",
        apply,
        (),
        init=initialize,
        execution="host",
        components=({"name": "fake-host"},),
    ), _SYSTEM


def _array(value: Tree) -> Array:
    """Normalize one config leaf with the same JAX dtype rules used by reset."""
    return jnp.asarray(value)


def _file_bytes(path: Path) -> int:
    """Count actual saved bytes recursively after the measured writer closes."""
    return sum(row.stat().st_size for row in path.rglob("*") if row.is_file())


def run(args: argparse.Namespace) -> int:
    """Run one explicitly chosen small case and save its full measurement record.

    args comes from benchmark_evaluation. Require one case, at least five warm
    repeats, an allowed backend, fixed B32 and chunk16, and matching package/assets
    paths. Hashing and post-run serialization are outside complete-call timings.
    Failed validation, backend access or comparisons raise instead of reducing the
    workload. The output JSON records first-call compilation separately from warm
    complete calls, source identity, payload size and measurement limitations.
    """
    case = args.evaluation_case
    if case not in CASES or args.repeats < 5:
        raise ValueError("select one evaluation case and at least five warm repeats")
    if any(
        (
            args.foundations,
            args.systems,
            args.recording_contracts,
            args.tracking,
            args.collection,
            args.metrics_only,
        )
    ):
        raise ValueError("evaluation access is separate from other benchmark modes")
    if args.sizes not in (None, [_BATCH]) or args.lengths not in (None, [_CHUNK]):
        raise ValueError("evaluation access uses only batch 32 and chunk 16")
    package = (args.package_root or Path.cwd()).resolve()
    assets = (args.assets_root or package / "src/marl_battlegrounds/data/tdm").resolve()
    _foundation_assets(assets)
    identity = _foundation_identity(package, assets)
    identity["evaluation_harness_sha256"] = sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    backend = args.backend or "gpu"
    device = cast(Any, jax.devices(backend)[0])
    path = output / f"evaluation-{case}-{backend}.json"
    record: dict[str, Tree] = {
        "case": case,
        "source_revision": args.source_revision,
        "identity": identity,
        "backend": backend,
        "device": str(device),
        "device_kind": device.device_kind,
        "requested_batch": _BATCH,
        "actual_batch": _BATCH,
        "chunk_size": _CHUNK,
        "games": _GAMES,
        "attempted_lane_rounds": 2 * _BATCH * _CHUNK,
    }
    with jax.default_device(device):
        started = time.perf_counter()
        a, b = marl_bgs.canonical_tournament_rosters()
        configs = tuple(
            make_standard_team_deathmatch_config(
                map_id=args.map_id if args.map_id is not None else 0,
                team_a_roster=a,
                team_b_roster=b,
                max_steps=length,
            )
            for length in (3, 7)
        )
        specs = tuple(
            EpisodeSpec(
                i + 1,
                configs[i % 2],
                args.map_id if args.map_id is not None else 0,
                i + 1,
            )
            for i in range(_GAMES)
        )
        weights = jnp.asarray(0.1, jnp.float32)
        changed = jnp.asarray(0.7, jnp.float32)
        jax.block_until_ready(configs)
        record["source_setup_ms"] = (time.perf_counter() - started) * 1000
        record["config_digest"] = _foundation_digest(configs)
        host_calls: list[int] = []
        first, second = (
            _host_pair(host_calls) if case == "mixed" else (_SYSTEM, _SYSTEM)
        )
        if case == "legacy":
            first = second = Policy(
                "Benchmark Policy", _actor, weights, jnp.asarray(0, jnp.int32)
            )
        invocations = 0
        saved: list[Path] = []

        def evaluate(parameters: Array) -> Tree:
            """Evaluate all exact scheduled games once, with fresh per-game memory."""
            nonlocal invocations
            invocations += 1
            selected = case == "capture"
            folder = output / "saved" if selected else None
            kwargs: dict[str, Tree] = {}
            if selected:
                kwargs.update(
                    output_dir=folder,
                    full_metrics_episodes=(1, 33),
                    replay_episodes=(1, 33),
                )
            else:
                kwargs["run_id"] = f"packet6-{case}-{invocations}"
            left = first if case == "mixed" else replace(first, variables=parameters)
            right = replace(second, variables=parameters)
            result = evaluate_episodes(
                left,
                right,
                specs,
                seed=42,
                num_envs=_BATCH,
                chunk_size=_CHUNK,
                metrics="priority",
                **kwargs,
            )
            if selected:
                saved.append(cast(dict[str, Path], result.paths)["run_details"].parent)
            return result

        function: Callable[[Array], Tree] = evaluate
        if case == "manual":
            env = marl_bgs.make("tdm", num_envs=_BATCH, metrics="priority")
            bank = jax.tree.map(
                lambda *values: jnp.stack(values).reshape(
                    (2, _BATCH, *values[0].shape)
                ),
                *(jax.tree.map(_array, spec.env_config) for spec in specs),
            )
            record["prepared_config_bytes"] = sum(
                leaf.nbytes for leaf in jax.tree.leaves(bank)
            )

            def manual(parameters: Array) -> _Rows:
                """Use prepared inputs; the timing wrapper synchronizes the result."""
                return cast(_Rows, _manual(env, bank, parameters))

            function = manual
        result, timing, probes = _systems_host_measure(
            function, (weights,), args.repeats, ((changed,),)
        )
        summary = _manual_summary(result) if case == "manual" else _summary(result)
        lengths = cast(list[int], summary["length"])
        if len(lengths) != _GAMES or sum(lengths) != 320:
            raise AssertionError(
                "the full fixed schedule did not complete its 320 real transitions"
            )
        record.update(timing)
        record["summary"] = summary
        record["real_transitions"] = sum(lengths)
        record["padded_lane_rounds"] = record["attempted_lane_rounds"] - sum(lengths)
        record["real_transitions_per_second"] = (
            sum(lengths) * 1000 / timing["warm_median_ms"]
        )
        record["warm_compilation_reused"] = not any(
            call["backend_compile_or_load_events"]
            for call in (*timing["warm_calls"], *timing["probe_calls"])
        )
        record["changed_weights_probe_completed_games"] = (
            len(cast(list[int], _manual_summary(probes[0])["length"]))
            if case == "manual"
            else len(probes[0].episodes)
        )
        record["host_call_batch_sizes"] = host_calls
        record["host_calls_per_pass"] = 14 if case == "mixed" else None
        if case == "mixed" and (
            set(host_calls) != {_BATCH} or len(host_calls) != 14 * (args.repeats + 2)
        ):
            raise AssertionError(
                "host execution lost its full batch or called a padded decision"
            )
        if saved:
            last = saved[-2]
            started = time.perf_counter()
            loaded = marl_bgs.load_results(last)
            priority = loaded.table("priority_metrics")
            record["load_and_priority_table_ms"] = (
                time.perf_counter() - started
            ) * 1000
            record["priority_table_rows"] = len(next(iter(priority.values())))
            record["saved_output_bytes"] = _file_bytes(last)
            record["saved_replay_count"] = len(loaded.replay_paths)
            with (last / "priority_metrics.csv").open(newline="") as handle:
                rows = sorted(
                    csv.DictReader(handle), key=lambda row: int(row["episode_id"])
                )
            record["summary"]["priority"] = [
                [
                    float(row[name]) if row[name] else None
                    for name in PRIORITY_METRIC_NAMES
                ]
                for row in rows
            ]
        record["retained_summary_json_bytes"] = len(
            json.dumps(summary, sort_keys=True).encode()
        )
        record["device_memory_stats"] = device.memory_stats()
        record["process_peak_ram_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        record["measurement_limits"] = [
            "CPU runs are correctness smoke checks, not a simulation-speed target.",
            "First call includes setup, compilation/cache loads and first "
            "execution; event durations are reported without inventing an "
            "isolated compile subtraction.",
            "The manual reference starts from prepared conditions and retains "
            "the same terminal rows. Public evaluator timings include "
            "condition identity, scheduling and result construction; this "
            "difference is intentional and visible.",
            "Allocator/RSS peaks include setup and compilation. They are "
            "process high-water marks, not isolated kernel live-memory peaks.",
            "Host transfers are included in complete-call latency; this run "
            "does not measure bus bandwidth or claim exact transfer bytes. "
            "No-file routes retain no replay/full history.",
            "Outcome/priority equality is a numerical comparison, not a "
            "complete trajectory proof; dedicated key/action and public "
            "replay tests supply that correctness evidence.",
            "Very short games emphasize scheduling/refill and padding costs; "
            "these numbers do not establish sustained long-combat throughput.",
            "The fake host provider has no network cost. No learning, maximum "
            "speed or whole-project optimality claim follows.",
        ]
    after = _foundation_identity(package, assets)
    after["evaluation_harness_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    record["identity_after"] = after
    record["source_assets_unchanged"] = after == identity
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if after != identity:
        raise RuntimeError("source/assets changed during this measured case")
    print(
        json.dumps(
            {
                "case": case,
                "output": str(path),
                "warm_ms": statistics.median(record["warm_samples_ms"]),
                "real_transitions_per_second": record["real_transitions_per_second"],
            }
        ),
        flush=True,
    )
    return 0
