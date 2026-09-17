"""Measure one explicit Packet 7 tournament workload without changing defaults.

Use benchmark_evaluation --canonical-contracts --canonical-case CASE
--canonical-workload WORKLOAD.json --output DIRECTORY. Cases are shared,
canonical, capture and reuse. The workload supplies immutable custom configs and
the same resolved jobs in a baseline-readable format. The shared case also runs
on the committed Packet 6 package; it deliberately excludes tournament analysis.
Fresh cases use three models, batch 32, chunk 16 and 64 games per matchup. Reuse
uses artificial twelve-entry records and never presents them as played results.

This tool times five or more complete warm calls. One separate instrumented call
counts explicit device_get payloads and attributes inclusive helper costs. Those
counts are not physical bus traffic. It changes no project source, training state
or official snapshot. Saved captures and measurement JSON stay under --output.
"""

from __future__ import annotations

# These measurements inspect existing private owners without adding public knobs.
# pyright: reportPrivateUsage=false
import argparse
import importlib
import json
import resource
import sys
import time
import weakref
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.environment import MetricMode
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.evaluation_conditions import restore_config
from marl_battlegrounds.evaluation.metric_catalog import PRIORITY_METRIC_NAMES
from marl_battlegrounds.evaluation.policy_execution import Policy
from scripts.dev.benchmark_evaluation import (
    _foundation_assets,
    _foundation_identity,
    _systems_host_measure,
)
from scripts.dev.benchmark_evaluation_access import _actor

type Tree = Any
CASES = ("shared", "canonical", "capture", "reuse")
_BATCH, _CHUNK = 32, 16


def load_bundle(paths: Mapping[str, Path]) -> Policy:
    """Load one small measured controller from a verified JSON parameter bundle.

    paths contains exactly one file with name and weight. Return the same actor
    callable for every entrant, one dynamic float32 weight and an int32 memory
    template. No action or initializer runs here. The caller owns asset checking.
    """
    if len(paths) != 1:
        raise ValueError("benchmark controller needs exactly one parameter bundle")
    value = json.loads(next(iter(paths.values())).read_text())
    return Policy(
        value["name"],
        _actor,
        jnp.asarray(value["weight"], jnp.float32),
        jnp.asarray(0, jnp.int32),
    )


def _read_workload(path: Path) -> dict[str, Any]:
    """Read explicit benchmark inputs and reject a silently changed workload size.

    The version-1 JSON has case-independent config/reference inputs, optional
    probe_config/probe_reference, and expected_games/expected_transitions. Each
    reference holds model bundles, configuration records and exact execution jobs.
    Paths are absolute. This trusted developer input is not a tournament config.
    """
    value = json.loads(path.read_text())
    if value.get("format") != "marlbg-canonical-benchmark" or value.get("version") != 1:
        raise ValueError("expected canonical benchmark workload version 1")
    if value.get("batch") != _BATCH or value.get("chunk") != _CHUNK:
        raise ValueError("canonical benchmark uses batch 32 and chunk 16")
    return value


def _workload_assets(workload: Mapping[str, Any]) -> dict[str, str]:
    """Hash the exact external fixture files outside all measured calls.

    Include both config files and all locally declared assets, even if a measured
    mode does not need their optional payloads. Hashing here is only the immutable
    input guard; timed asset verification remains the runner's own work.
    """
    paths: set[Path] = set()
    for key in ("config", "probe_config"):
        if key not in workload:
            continue
        config_path = Path(workload[key])
        paths.add(config_path)
        config = json.loads(config_path.read_text())
        for asset in config["assets"].values():
            if asset.get("path") is not None:
                paths.add(Path(asset["path"]))
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep exact sorted game outcomes and priority cells, excluding run labels."""
    fields = (
        "episode_id",
        "outcome",
        "episode_length",
        "team_a_score",
        "team_b_score",
        *PRIORITY_METRIC_NAMES,
    )
    return [
        {field: _json_cell(row.get(field)) for field in dict.fromkeys(fields)}
        for row in sorted(rows, key=lambda item: item["episode_id"])
    ]


def _json_cell(value: object) -> object:
    """Convert one numerical cell to JSON while preserving unavailable values."""
    if isinstance(value, np.generic):
        value = cast(Any, value).item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _shared(reference: Mapping[str, Any], metrics: MetricMode) -> list[dict[str, Any]]:
    """Run exact ready jobs on the existing evaluator and retain matching game rows.

    Models load only for the current pair. Configurations are restored once per
    call, using the same serialized numerical contents as the config runner. This
    simple reference includes evaluator setup and outcomes, but excludes asset
    verification, tournament fits, headline calculations and optional saving.
    """
    configs = {
        name: restore_config(value) for name, value in reference["configs"].items()
    }
    rows: list[dict[str, Any]] = []
    for job in reference["jobs"]:
        first = load_bundle({"bundle": Path(reference["models"][job["team_a"]])})
        second = load_bundle({"bundle": Path(reference["models"][job["team_b"]])})
        specs = tuple(
            EpisodeSpec(
                row["episode_id"],
                configs[row["config_id"]],
                row["map_id"],
                row["seed_id"],
                metadata=row.get("metadata"),
                source_config=configs[row["source_config_id"]],
                spawn_locations=row["spawn_locations"],
                paired_comparison_key=row["paired_comparison_key"],
            )
            for row in job["episodes"]
        )
        result = evaluate_episodes(
            first,
            second,
            specs,
            seed=job["seed"],
            num_envs=_BATCH,
            chunk_size=_CHUNK,
            metrics=metrics,
        )
        logical = {row["episode_id"]: row["logical_game_id"] for row in job["episodes"]}
        outcomes = {row.episode_id: row for row in result.episodes}
        values = result.priority_metrics
        for index, identifier in enumerate(sorted(outcomes)):
            game = outcomes[identifier]
            rows.append(
                {
                    "episode_id": logical[identifier],
                    "outcome": game.outcome,
                    "episode_length": game.episode_length,
                    "team_a_score": game.team_a_score,
                    "team_b_score": game.team_b_score,
                    **{
                        name: column[index]
                        for name, column in values.items()
                        if name in PRIORITY_METRIC_NAMES
                    },
                }
            )
        del first, second, result
    return _summary(rows)


def _result_rows(result: Tree) -> list[dict[str, Any]]:
    """Read the canonical match view once and normalize its logical game order."""
    records = result._record_access
    owners = {}
    for game in records.games:
        origin = records.origin(game)
        owners[
            tuple(
                origin[field] for field in ("run_id", "phase", "pass_id", "episode_id")
            )
        ] = game["logical_game_id"]
    rows: list[dict[str, Any]] = []
    for columns in result.iter_table("matches", rows=128):
        if not columns:
            continue
        names = tuple(columns)
        for index in range(len(columns[names[0]])):
            row = {name: columns[name][index] for name in names}
            key = tuple(
                row[field] for field in ("run_id", "phase", "pass_id", "episode_id")
            )
            row["episode_id"] = owners[key]
            rows.append(row)
    return _summary(rows)


def _deep_bytes(value: object, seen: set[int] | None = None) -> int:
    """Estimate retained Python index storage once per object, without arrays."""
    visited: set[int] = set() if seen is None else seen
    if id(value) in visited:
        return 0
    visited.add(id(value))
    count = sys.getsizeof(value)
    if isinstance(value, dict):
        count += sum(
            _deep_bytes(k, visited) + _deep_bytes(v, visited)
            for k, v in cast(dict[object, object], value).items()
        )
    elif isinstance(value, (tuple, list, set, frozenset)):
        count += sum(
            _deep_bytes(item, visited) for item in cast(tuple[object, ...], value)
        )
    return count


def _attribute(
    function: Callable[[bool], Tree], *, shared: bool = False
) -> tuple[Tree, dict[str, Any]]:
    """Inspect one warmed call with temporary wrappers, restoring every owner.

    Timed regions are inclusive and can overlap. Chunk wrappers synchronize the
    returned device tree and count attempted lane rounds. Explicit device_get
    counts logical input array bytes, including repeated requests for ready data;
    implicit NumPy conversions and host-to-device transfers are outside that count.
    Index bytes estimate retained Python offset dictionaries, not peak allocations.
    """
    evaluation = importlib.import_module("marl_battlegrounds.evaluation.evaluate")
    report: dict[str, Any] = {
        "regions": {},
        "device_get_calls": 0,
        "logical_device_get_bytes": 0,
        "model_loads": 0,
        "max_live_loaded_models": 0,
        "actual_batches": [],
        "attempted_lane_rounds": 0,
        "indexes": [],
    }
    references: list[weakref.ReferenceType[object]] = []
    original_get = jax.device_get

    def transfer(value: Tree) -> Tree:
        """Count only explicit JAX array input payloads at this host boundary."""
        report["device_get_calls"] += 1
        report["logical_device_get_bytes"] += sum(
            leaf.nbytes
            for leaf in jax.tree.leaves(value)
            if isinstance(leaf, jax.Array)
        )
        return original_get(value)

    def wrap(
        owner: Tree,
        name: str,
        label: str,
        *,
        chunk: bool = False,
        loading: bool = False,
        index: bool = False,
    ) -> Callable[..., Tree]:
        """Build one timed wrapper without altering its arguments or return value."""
        original = getattr(owner, name)

        def measured(*args: Tree, **kwargs: Tree) -> Tree:
            """Time an existing owner and retain only narrow measurement facts."""
            start = time.perf_counter()
            value = original(*args, **kwargs)
            if chunk:
                jax.block_until_ready(value)
                batch = int(args[0].num_envs)
                report["actual_batches"].append(batch)
                report["attempted_lane_rounds"] += batch * _CHUNK
            if loading:
                references.append(weakref.ref(value))
                report["model_loads"] += 1
                report["max_live_loaded_models"] = max(
                    report["max_live_loaded_models"],
                    sum(reference() is not None for reference in references),
                )
            if index:
                report["indexes"].append(
                    {
                        "records": len(args[0]._positions),
                        "retained_python_bytes": _deep_bytes(args[0]._positions),
                        "committed_source_bytes": args[0].size,
                    }
                )
            elapsed = (time.perf_counter() - start) * 1000
            row = report["regions"].setdefault(label, {"calls": 0, "wall_ms": 0.0})
            row["calls"] += 1
            row["wall_ms"] += elapsed
            return value

        return measured

    with ExitStack() as stack:
        stack.enter_context(patch.object(jax, "device_get", transfer))
        if shared:
            module = sys.modules[__name__]
            stack.enter_context(
                patch.object(
                    module,
                    "load_bundle",
                    wrap(module, "load_bundle", "reference_model_load", loading=True),
                )
            )
        for name, label, chunk in (
            ("_stack_configs", "config_stacking", False),
            ("_jax_chunk", "synchronized_evaluator_chunks", True),
            ("_run_evaluation", "complete_evaluator_jobs", False),
        ):
            if hasattr(evaluation, name):
                stack.enter_context(
                    patch.object(
                        evaluation, name, wrap(evaluation, name, label, chunk=chunk)
                    )
                )
        for module_name, name, label, loading, indexed in (
            (
                "canonical",
                "_source_configs",
                "source_configuration_preparation",
                False,
                False,
            ),
            (
                "canonical",
                "_specs",
                "exact_job_preparation",
                False,
                False,
            ),
            (
                "canonical",
                "_result",
                "result_assembly",
                False,
                False,
            ),
            (
                "tournament_evidence",
                "prepare_reuse_evidence",
                "physical_population_verification",
                False,
                False,
            ),
            (
                "tournament_assets",
                "load_tournament_controller",
                "model_load_and_verification",
                True,
                False,
            ),
            (
                "tournament_config",
                "load_tournament_config",
                "config_resolution",
                False,
                False,
            ),
            (
                "tournament_statistics",
                "summarize_tournament",
                "statistics",
                False,
                False,
            ),
            ("tournament_headlines", "summarize_headlines", "headlines", False, False),
        ):
            try:
                module = importlib.import_module(
                    f"marl_battlegrounds.evaluation.{module_name}"
                )
            except ModuleNotFoundError:
                continue
            stack.enter_context(
                patch.object(
                    module,
                    name,
                    wrap(module, name, label, loading=loading, index=indexed),
                )
            )
        try:
            assets = importlib.import_module(
                "marl_battlegrounds.evaluation.tournament_assets"
            )
        except ModuleNotFoundError:
            assets = None
        if assets is not None:
            cls = assets.AssetVerifier
            stack.enter_context(
                patch.object(cls, "verify", wrap(cls, "verify", "asset_verification"))
            )
        scalar = importlib.import_module("marl_battlegrounds.evaluation.scalar_reports")
        if hasattr(scalar, "IndexedScalarTable"):
            cls = scalar.IndexedScalarTable
            stack.enter_context(
                patch.object(
                    cls,
                    "__init__",
                    wrap(cls, "__init__", "source_row_index", index=True),
                )
            )
        writer = importlib.import_module(
            "marl_battlegrounds.evaluation.run_writer"
        ).RunWriter
        stack.enter_context(
            patch.object(writer, "flush", wrap(writer, "flush", "writer_flush"))
        )
        start = time.perf_counter()
        result = function(False)
        report["complete_instrumented_ms"] = (time.perf_counter() - start) * 1000
    report["scope"] = (
        "Separate warmed attribution call. Inclusive timings overlap; "
        "explicit logical get bytes are not bus traffic."
    )
    return result, report


def run(args: argparse.Namespace) -> int:
    """Execute one selected fixed workload and write its measurements under output.

    Require five warm repeats, batch32/chunk16 and an explicit workload file.
    Fresh execution requires a GPU for performance qualification; CPU is useful
    only for smoke checks. Reuse may run on CPU because it plays no games. Config
    and package identities are recorded before and after all measured calls.
    """
    case = args.canonical_case
    if case not in CASES or args.repeats < 5 or args.canonical_workload is None:
        raise ValueError("select one canonical case, workload and five warm repeats")
    if any(
        getattr(args, name)
        for name in (
            "foundations",
            "systems",
            "recording_contracts",
            "tracking",
            "collection",
            "evaluation_access",
            "metrics_only",
        )
    ):
        raise ValueError("canonical contracts is separate from other benchmark modes")
    if args.sizes not in (None, [_BATCH]) or args.lengths not in (None, [_CHUNK]):
        raise ValueError("canonical contracts uses only batch32 and chunk16")
    workload_path = args.canonical_workload.resolve()
    workload = _read_workload(workload_path)
    package = (args.package_root or Path.cwd()).resolve()
    assets = (args.assets_root or package / "src/marl_battlegrounds/data/tdm").resolve()
    _foundation_assets(assets)
    identity = _foundation_identity(package, assets)
    identity["canonical_harness_sha256"] = sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    identity["workload_sha256"] = sha256(workload_path.read_bytes()).hexdigest()
    identity["workload_assets"] = _workload_assets(workload)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    backend = args.backend or "gpu"
    device = cast(Any, jax.devices(backend)[0])
    metrics = workload.get("metrics", "priority")
    saved: list[Path] = []

    def execute(probe: bool) -> Tree:
        """Run the exact unchanged or changed-weight workload through one route."""
        if case == "shared":
            key = "probe_reference" if probe else "reference"
            return _shared(workload[key], metrics)
        config = workload["probe_config" if probe else "config"]
        kwargs: dict[str, Any] = {}
        if case == "capture":
            kwargs.update(
                output_dir=output / "saved",
                full_metrics_episodes=tuple(workload["capture_ids"]),
                replay_episodes=tuple(workload["capture_ids"]),
            )
        result = cast(Any, marl_bgs.run_tournament)(
            config=config, metrics=metrics, num_envs=_BATCH, chunk_size=_CHUNK, **kwargs
        )
        if result.paths is not None:
            saved.append(result.paths["run_details"].parent)
        return _result_rows(result)

    record: dict[str, Any] = {
        "case": case,
        "source_revision": args.source_revision,
        "identity": identity,
        "backend": backend,
        "device": str(device),
        "device_kind": device.device_kind,
        "requested_batch": _BATCH,
        "chunk_size": _CHUNK,
        "metrics": metrics,
        "artificial_reuse_records": case == "reuse",
    }
    with jax.default_device(device):
        probe_key = "probe_reference" if case == "shared" else "probe_config"
        probe = ((True,),) if probe_key in workload else ()
        result, timing, probes = _systems_host_measure(
            execute, (False,), args.repeats, probe
        )
        record.update(timing)
        record["summary"] = result
        record["probe_summary"] = probes[0] if probes else None
        if len(result) != workload["expected_games"]:
            raise AssertionError("benchmark did not retain every expected game")
        transitions = (
            0 if case == "reuse" else sum(row["episode_length"] for row in result)
        )
        if transitions != workload["expected_transitions"]:
            raise AssertionError("benchmark transition count differs from workload")
        observed, attribution = _attribute(execute, shared=case == "shared")
        if observed != result:
            raise AssertionError("instrumentation changed the final game rows")
        record["attribution"] = attribution
        record["real_transitions"] = transitions
        record["real_agent_decisions"] = transitions * 10
        record["padding_lane_rounds"] = (
            attribution["attempted_lane_rounds"] - transitions
        )
        record["real_transitions_per_second"] = (
            transitions * 1000 / timing["warm_median_ms"]
        )
        record["warm_compilation_reused"] = not any(
            row["backend_compile_or_load_events"]
            for row in (*timing["warm_calls"], *timing["probe_calls"])
        )
        record["device_memory_stats"] = device.memory_stats()
        record["process_peak_ram_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        if saved:
            record["saved_output_bytes"] = sum(
                path.stat().st_size for path in saved[-1].rglob("*") if path.is_file()
            )
            record["last_saved_run"] = str(saved[-1])
    after = _foundation_identity(package, assets)
    after["canonical_harness_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    after["workload_sha256"] = sha256(workload_path.read_bytes()).hexdigest()
    after["workload_assets"] = _workload_assets(workload)
    record["identity_after"] = after
    record["source_assets_unchanged"] = identity == after
    record["measurement_limits"] = [
        "Shared calls retain matching outcomes/priority but exclude new asset "
        "verification and tournament summaries.",
        "First call includes setup, compilation/cache loads and execution; "
        "backend events do not isolate all compiler cost.",
        "Five warm calls are complete workflow latency. One instrumented call "
        "attributes inclusive overlapping regions.",
        "Explicit logical device_get bytes exclude implicit NumPy conversions "
        "and automatic host-to-device placement.",
        "Allocator/RSS peaks cover the entire worker, including setup and "
        "compilation; they are not kernel-only peaks.",
        "Short games emphasize scheduling, padding and host setup. No long-combat, "
        "learning or theoretical-optimality claim follows.",
        "Artificial twelve-entry reuse rows measure joins and analysis only; "
        "they are not qualified official results.",
    ]
    path = output / f"canonical-{case}-{metrics}-{backend}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if identity != after:
        raise RuntimeError("source or fixed assets changed during this case")
    print(
        json.dumps(
            {"case": case, "output": str(path), "warm_ms": timing["warm_median_ms"]}
        ),
        flush=True,
    )
    return 0
