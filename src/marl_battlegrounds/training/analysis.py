"""Summarize saved training and frozen validation without running a learner.

The runner, CLI and researchers share selection, paired uncertainty and report
generation here. M8 remains the owner of recorded game scores and metrics.
Reports read committed trainer records and never rewrite original evidence.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

import numpy as np

Record = dict[str, Any]


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    """Require a plain integer at least minimum; Boolean values raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _score(value: object) -> float:
    """Read one authoritative W/D/L score; reject missing or invented values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("A completed game needs a numerical system_game_score")
    result = float(value)
    if result not in (0.0, 0.5, 1.0):
        raise ValueError("system_game_score must be 0, 0.5 or 1")
    return result


def _bootstrap(
    blocks: Sequence[np.ndarray[Any, np.dtype[np.float64]]],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """Resample whole independent blocks within maps, then weight maps equally.

    Each one-dimensional float64 array contains one map's block means. The caller
    must keep dependent games together inside each block. draws is positive and
    seed is a nonnegative integer. Return the 2.5/97.5 percentile score bounds.
    This host calculation uses its own NumPy generator and changes no input.
    """
    _integer(draws, "bootstrap draws", minimum=1)
    _integer(seed, "bootstrap seed")
    random = np.random.default_rng(seed)
    means = np.zeros(draws, np.float64)
    for values in blocks:
        indices = random.integers(len(values), size=(draws, len(values)))
        means += values[indices].mean(axis=1) / len(blocks)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def summarize_validation(
    rows: Sequence[Mapping[str, Any]],
    *,
    maps: Sequence[int],
    opponents: Sequence[str],
    seed_pairs: int,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 19_044_001,
) -> Record:
    """Reduce complete frozen-panel games with their shared random seeds intact.

    Parameters
    ----------
    rows : sequence of mappings
        Saved M8 rows augmented with opponent and spawn_locations. Required fields
        are map_id, seed_id, opponent, spawn_locations and system_game_score. Each
        map/seed/opponent has exactly two ends. Optional scores and lengths are
        reported only when every game in that cell supplies them.
    maps, opponents : sequences
        Nonempty distinct map IDs and frozen opponent labels in declared order.
    seed_pairs : int
        Positive independent paired seeds per map/opponent. Opponents share those
        seeds; different maps must use distinct seed IDs.
    bootstrap_draws, bootstrap_seed : int
        Positive replicate count and independent NumPy seed, default 2000/19044001.

    Returns
    -------
    dict
        Equal-weight score, descriptive paired interval, W/D/L cells and complete
        counts. The interval concerns these fixed actors, not training-seed variation.

    Raises
    ------
    ValueError
        Coverage, score values, dimensions or shared seed pairing are incompatible.

    Notes
    -----
        Host-only. All opponents and both ends stay together in a bootstrap block.
        No original metrics are recomputed and no file or global RNG is changed.
    """
    _integer(seed_pairs, "seed_pairs", minimum=1)
    map_ids, names = tuple(maps), tuple(opponents)
    if not map_ids or not names or len(set(map_ids)) != len(map_ids):
        raise ValueError("Validation needs nonempty distinct maps and opponents")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("Opponent labels must be nonempty and distinct")
    expected_count = len(map_ids) * len(names) * seed_pairs * 2
    if len(rows) != expected_count:
        raise ValueError("Validation has missing or extra games")
    indexed: dict[tuple[int, int, str, int], Mapping[str, Any]] = {}
    seeds: dict[int, set[int]] = {map_id: set() for map_id in map_ids}
    for row in rows:
        map_id = _integer(row.get("map_id"), "map_id")
        seed_id = _integer(row.get("seed_id"), "seed_id")
        end = _integer(row.get("spawn_locations"), "spawn_locations")
        name = row.get("opponent")
        if map_id not in seeds or name not in names or end not in (0, 1):
            raise ValueError("Validation game is outside its declared conditions")
        key = (map_id, seed_id, cast(str, name), end)
        if key in indexed:
            raise ValueError("Validation repeats a map/seed/opponent/spawn game")
        _score(row.get("system_game_score"))
        indexed[key] = row
        seeds[map_id].add(seed_id)
    seen_seeds: set[int] = set()
    blocks: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    cells: list[Record] = []
    for map_id in map_ids:
        selected_seeds = sorted(seeds[map_id])
        if len(selected_seeds) != seed_pairs or seen_seeds.intersection(selected_seeds):
            raise ValueError("Each map needs its own exact independent seed blocks")
        seen_seeds.update(selected_seeds)
        values = np.empty((seed_pairs, len(names), 2), np.float64)
        for opponent_index, name in enumerate(names):
            cell_rows: list[Mapping[str, Any]] = []
            for seed_index, seed_id in enumerate(selected_seeds):
                for end in (0, 1):
                    try:
                        row = indexed[map_id, seed_id, name, end]
                    except KeyError as error:
                        raise ValueError(
                            "A dependent validation block is incomplete"
                        ) from error
                    values[seed_index, opponent_index, end] = _score(
                        row["system_game_score"]
                    )
                    cell_rows.append(row)
            scores = values[:, opponent_index].reshape(-1)
            cell: Record = {
                "map_id": map_id,
                "opponent": name,
                "games": len(scores),
                "seed_pairs": seed_pairs,
                "score": float(scores.mean()),
                "wins": int(np.count_nonzero(scores == 1)),
                "draws": int(np.count_nonzero(scores == 0.5)),
                "losses": int(np.count_nonzero(scores == 0)),
            }
            for column in ("episode_length", "team_a_score", "team_b_score"):
                present = [row.get(column) for row in cell_rows]
                cell[f"mean_{column}"] = (
                    float(np.mean(np.asarray(present, np.float64)))
                    if all(x is not None for x in present)
                    else None
                )
            cells.append(cell)
        blocks.append(values.mean(axis=(1, 2)))
    low, high = _bootstrap(blocks, draws=bootstrap_draws, seed=bootstrap_seed)
    return {
        "complete": True,
        "score": float(np.mean([values.mean() for values in blocks])),
        "ci_low": low,
        "ci_high": high,
        "games": expected_count,
        "independent_blocks": len(map_ids) * seed_pairs,
        "cells": cells,
        "uncertainty": (
            "Conditional game-sampling interval; shared opponents and spawn ends "
            "stay together"
        ),
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
    }


def _candidates(results: Sequence[Mapping[str, Any]]) -> dict[str, Record]:
    """Validate complete unique noninitial checkpoint summaries for selection."""
    selected: dict[str, Record] = {}
    for result in results:
        identifier = result.get("checkpoint_id")
        if not isinstance(identifier, str) or not identifier or identifier in selected:
            raise ValueError("Selection needs distinct nonempty checkpoint identities")
        if result.get("complete") is not True:
            raise ValueError("Incomplete validation cannot select a checkpoint")
        steps = _integer(result.get("env_steps"), "env_steps")
        value = result.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Selection score is missing")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Selection score must be finite and between zero and one")
        if steps:
            selected[identifier] = dict(result)
    if not selected:
        raise ValueError("No eligible trained checkpoint is available")
    panels = {row.get("panel_digest") for row in selected.values()}
    if len(panels) != 1:
        raise ValueError("Selection cannot mix frozen panels")
    return selected


def confirmation_candidates(
    routine_results: Sequence[Mapping[str, Any]], *, final_checkpoint_id: str
) -> tuple[str, ...]:
    """Return the best two routine checkpoint IDs, then final when distinct.

    routine_results must be complete unique fixed-panel summaries containing
    checkpoint_id, env_steps, score and complete. Initialization is ignored. Ties
    prefer earlier experience, then identity for stable output. The final checkpoint
    must be present. Incomplete/mismatched evidence raises ValueError. No file changes.
    """
    candidates = _candidates(routine_results)
    if final_checkpoint_id not in candidates:
        raise ValueError("The final checkpoint lacks complete routine validation")
    ordered = sorted(
        candidates,
        key=lambda key: (-candidates[key]["score"], candidates[key]["env_steps"], key),
    )[:2]
    if final_checkpoint_id not in ordered:
        ordered.append(final_checkpoint_id)
    return tuple(ordered)


def select_checkpoint(confirmation_results: Sequence[Mapping[str, Any]]) -> Record:
    """Choose a complete fresh-confirmation winner, with earlier steps breaking ties.

    Each input uses the same summary contract as confirmation_candidates and must
    have purpose='confirmation'. Return a new copy of the winning summary. The
    caller owns checking that every scheduled candidate was confirmed. Empty,
    incomplete, mixed-panel or wrong-purpose records raise ValueError. No I/O occurs.
    """
    if any(row.get("purpose") != "confirmation" for row in confirmation_results):
        raise ValueError("Selection requires fresh confirmation results")
    candidates = _candidates(confirmation_results)
    return dict(
        min(
            candidates.values(),
            key=lambda row: (-row["score"], row["env_steps"], row["checkpoint_id"]),
        )
    )


def slot_planning_blocks(
    *, effect: float = 0.1, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Return the normal-approximation block count using the variance bound one.

    effect is a positive expected-score difference; power and alpha lie in (0,1).
    The default gives 785 blocks before balancing across maps. This is a planning
    approximation, not a distribution-free power guarantee. Invalid settings raise
    ValueError. The host calculation has no randomness or file effects.
    """
    if not 0 < effect <= 1 or not 0 < power < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid slot diagnostic planning settings")
    normal = NormalDist()
    return math.ceil(
        ((normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)) / effect) ** 2
    )


def _effect_summary(
    blocks: Sequence[np.ndarray[Any, np.dtype[np.float64]]], *, seed: int, draws: int
) -> Record:
    """Compute an equal-map mean effect and its stratified independent-block error.

    blocks contains one float64 vector per map, with at least two independent
    blocks per vector. A zero estimated variance leaves inference unavailable.
    The normal p-value is approximate and the bootstrap interval is descriptive.
    """
    effect = float(np.mean([value.mean() for value in blocks]))
    variance = (
        sum(float(value.var(ddof=1)) / len(value) for value in blocks)
        / len(blocks) ** 2
    )
    error = math.sqrt(variance)
    if error == 0:
        return {
            "effect": effect,
            "standard_error": 0.0,
            "p_value": None,
            "ci_low": None,
            "ci_high": None,
            "inference": "Unavailable: zero estimated variance",
        }
    low, high = _bootstrap(blocks, draws=draws, seed=seed)
    return {
        "effect": effect,
        "standard_error": error,
        "p_value": math.erfc(abs(effect / error) / math.sqrt(2)),
        "ci_low": low,
        "ci_high": high,
        "inference": (
            "Approximate normal mean test; independent blocks; "
            "descriptive bootstrap interval"
        ),
    }


def summarize_slot_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    *,
    maps: Sequence[int],
    seed_blocks: int = 160,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 19_044_002,
) -> Record:
    """Compare a fixed focal model across slot blocks while keeping its side fixed.

    Rows require map_id, seed_id, focal_team (0/1), spawn_locations (0/1) and M8's
    system_game_score. Four games form each independent map/seed block. Team B's
    focal score is one minus M8's Team A score. maps are distinct and equally
    weighted; seed_blocks is at least two. Return primary side-balanced and both
    physical-side effects, normal p-values, Holm-adjusted side p-values and counts.
    The independent seeds, unchanged physical conditions and numerical equivalence
    of checkpoints are caller evidence obligations. Missing, duplicate or malformed
    conditions raise ValueError. This host-only helper changes no files or inputs.
    """
    _integer(seed_blocks, "seed_blocks", minimum=2)
    map_ids = tuple(maps)
    if not map_ids or len(set(map_ids)) != len(map_ids):
        raise ValueError("Slot diagnostic maps must be nonempty and distinct")
    if len(rows) != len(map_ids) * seed_blocks * 4:
        raise ValueError("Slot diagnostic has missing or extra games")
    indexed: dict[tuple[int, int, int, int], float] = {}
    seeds: dict[int, set[int]] = {map_id: set() for map_id in map_ids}
    for row in rows:
        map_id = _integer(row.get("map_id"), "map_id")
        seed = _integer(row.get("seed_id"), "seed_id")
        team = _integer(row.get("focal_team"), "focal_team")
        end = _integer(row.get("spawn_locations"), "spawn_locations")
        if map_id not in seeds or team not in (0, 1) or end not in (0, 1):
            raise ValueError("Invalid slot diagnostic condition")
        key = (map_id, seed, team, end)
        if key in indexed:
            raise ValueError("Slot diagnostic repeats a condition")
        score = _score(row.get("system_game_score"))
        indexed[key] = score if team == 0 else 1 - score
        seeds[map_id].add(seed)
    seen: set[int] = set()
    side_blocks: tuple[list[np.ndarray[Any, np.dtype[np.float64]]], ...] = ([], [])
    balanced: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for map_id in map_ids:
        values = np.empty((seed_blocks, 2), np.float64)
        selected = sorted(seeds[map_id])
        if len(selected) != seed_blocks or seen.intersection(selected):
            raise ValueError("Slot seed blocks must be complete and map-distinct")
        seen.update(selected)
        for i, seed in enumerate(selected):
            for side in (0, 1):
                try:
                    values[i, side] = (
                        indexed[map_id, seed, 0, side]
                        - indexed[map_id, seed, 1, 1 - side]
                    )
                except KeyError as error:
                    raise ValueError("Slot diagnostic block is incomplete") from error
        for side in (0, 1):
            side_blocks[side].append(values[:, side])
        balanced.append(values.mean(axis=1))
    primary = _effect_summary(balanced, seed=bootstrap_seed, draws=bootstrap_draws)
    sides = [
        _effect_summary(blocks, seed=bootstrap_seed + side + 1, draws=bootstrap_draws)
        for side, blocks in enumerate(side_blocks)
    ]
    available = sorted(
        (float(row["p_value"]), i)
        for i, row in enumerate(sides)
        if row["p_value"] is not None
    )
    previous = 0.0
    for rank, (value, index) in enumerate(available):
        previous = max(previous, min(1.0, (2 - rank) * value))
        sides[index]["holm_p_value"] = previous
    for side, row in enumerate(sides):
        row["physical_side"] = side
        row.setdefault("holm_p_value", None)
    return {
        "complete": True,
        "games": len(rows),
        "independent_blocks": len(map_ids) * seed_blocks,
        "primary": primary,
        "sides": sides,
        "alpha": 0.05,
        "systematic_effect_detected": bool(
            (primary["p_value"] is not None and primary["p_value"] < 0.05)
            or any(
                row["holm_p_value"] is not None and row["holm_p_value"] < 0.05
                for row in sides
            )
        ),
        "scope": (
            "These frozen models and validation maps only; no universal fairness claim"
        ),
    }


def _read_json(path: Path, default: object) -> Any:  # noqa: ANN401
    """Read one JSON file or return default when absent; malformed JSON raises."""
    return json.loads(path.read_text()) if path.exists() else default


def _read_rows(path: Path) -> list[Record]:
    """Read committed newline-delimited JSON objects; reject a partial final row."""
    if not path.exists():
        return []
    text = path.read_text()
    if text and not text.endswith("\n"):
        raise ValueError(f"Incomplete committed log row: {path.name}")
    rows = [json.loads(line) for line in text.splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected object rows in {path.name}")
    return cast(list[Record], rows)


def _atomic_text(path: Path, content: str) -> None:
    """Replace one report atomically on its filesystem; fsync before publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_events(path: Path) -> list[Record]:
    """Read append-only events, marking torn or malformed lines as unknown work.

    Unlike update logs, event history can preserve an interrupted fragment before
    a resumed attempt. Every invalid line, including a final line without its
    newline, becomes an incomplete_event marker. No success or score is inferred
    from that fragment. Original bytes stay untouched; absent files give [].
    """
    if not path.exists():
        return []
    result: list[Record] = []
    for index, line in enumerate(path.read_text().splitlines(keepends=True), start=1):
        try:
            value: object = json.loads(line)
            if not line.endswith("\n") or not isinstance(value, dict):
                raise ValueError("Incomplete event")
            result.append(cast(Record, value))
        except ValueError:
            result.append({"event": "incomplete_event", "line": index})
    return result


def _csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize scalar report rows with a stable first-seen union of columns."""
    columns = tuple(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _treatment(config: Mapping[str, Any]) -> str:
    """Name the enabled curriculum/shaping options; absent options mean Plain."""
    return (
        "+".join(
            name
            for key, name in (("curriculum", "Curriculum"), ("shaping", "Shaping"))
            if config.get(key)
        )
        or "Plain"
    )


def _phase_costs(events: Sequence[Record], details: Record) -> Record:
    """Sum saved finite phase seconds across all attempts without new timing.

    Count only the phase owner's completed event, not nested evaluation segments.
    Missing or invalid times remain unavailable. Initial setup in run_details is
    a fallback only when no attempt setup event has a usable time; it is never
    added to that same event. Interrupted work may have no completed measurement.
    """
    costs: Record = {}
    for phase, event, field in (
        ("Setup", "attempt_started", "setup_seconds"),
        ("Validation", "validation_complete", "seconds"),
        ("Checkpoint", "checkpoint_saved", "seconds"),
        ("Report", "reports_written", "seconds"),
        ("Slot Diagnostic", "slot_diagnostic_finished", "seconds"),
    ):
        values = [
            float(row[field])
            for row in events
            if row.get("event") == event
            and type(row.get(field)) in (float, int)
            and math.isfinite(row[field])
            and row[field] >= 0
        ]
        source = "run_events.jsonl"
        initial = details.get("initial_setup_seconds")
        if (
            phase == "Setup"
            and not values
            and isinstance(initial, (float, int))
            and not isinstance(initial, bool)
            and math.isfinite(initial)
            and initial >= 0
        ):
            values = [float(initial)]
            source = "run_details.json (initial setup only)"
        costs[phase] = {
            "seconds": math.fsum(values) if values else None,
            "records": len(values),
            "source": source,
        }
    return costs


def _summary(directory: Path) -> Record:
    """Read run facts, withholding active results while checkpoint recovery is open.

    An unfinished recovery can leave an older complete status and output files.
    Keep their paths as evidence, but do not label them as current results or load
    the possibly partly recovered update log. This reader changes no source file.
    """
    details = _read_json(directory / "run_details.json", None)
    if not isinstance(details, dict):
        raise ValueError(f"Missing run_details.json in {directory}")
    details = cast(Record, details)
    config = details.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("run_details.json config must contain an object")
    config = cast(Record, config)
    recovering = (directory / "checkpoint_recovery.json").exists()
    recovery = (
        _read_json(directory / "checkpoint_recovery.json", None) if recovering else None
    )
    updates = [] if recovering else _read_rows(directory / "training_updates.jsonl")
    events = _read_events(directory / "run_events.jsonl")
    results: object = (
        [] if recovering else _read_json(directory / "validation_results.json", [])
    )
    if isinstance(results, dict):
        results = cast(Record, results).get("results", [])
    if not isinstance(results, list):
        raise ValueError("validation_results.json must contain result objects")
    status = _read_json(directory / "status.json", {})
    if not isinstance(status, dict):
        raise ValueError("status.json must contain an object")
    status = cast(Record, status)
    return {
        "run_id": str(details.get("run_id", directory.name)),
        "run_dir": str(directory),
        "method": config.get("method", "Unavailable"),
        "treatment": _treatment(config),
        "seed": config.get("seed"),
        "declared_env_steps": config.get("total_env_steps"),
        "details_path": str(directory / "run_details.json"),
        "schemas": details.get("schemas"),
        "initial_runtime": details.get("runtime"),
        "attempt_runtimes": [
            {
                "attempt_id": event.get("attempt_id"),
                "time_utc": event.get("time_utc"),
                "runtime": event.get("runtime"),
            }
            for event in events
            if event.get("event") == "attempt_started"
        ],
        "memory": None if recovering else status.get("memory"),
        "terminal_memory_history": [
            {
                "attempt_id": event.get("attempt_id"),
                "time_utc": event.get("time_utc"),
                "event": event["event"],
                "memory": event.get("memory"),
            }
            for event in events
            if event.get("event") in ("attempt_complete", "attempt_failed")
        ],
        "panel_digest": details.get("panel_digest"),
        "status": "recovering" if recovering else status.get("status", "incomplete"),
        "saved_status": status.get("status", "incomplete"),
        "recovery": recovery,
        "elapsed_seconds": None if recovering else status.get("elapsed_seconds"),
        "wall_seconds": None if recovering else status.get("wall_seconds"),
        "env_steps": None
        if recovering
        else updates[-1].get("env_steps", 0)
        if updates
        else 0,
        "validation_checkpoints": len(cast(list[object], results)),
        "recorded_failures": sum(
            row.get("event") in ("failure", "attempt_failed") for row in events
        ),
        "incomplete_events": sum(
            row.get("event") == "incomplete_event" for row in events
        ),
        "phase_costs": _phase_costs(events, details),
        "selection": None
        if recovering
        else _read_json(directory / "selection.json", None),
        "final_actor": None if recovering else status.get("final_actor"),
        "selected_actor": None if recovering else status.get("selected_actor"),
        "latest_update": updates[-1] if updates else None,
        "exposure": None
        if recovering
        else _read_json(directory / "exposure.json", None),
        "slot_diagnostic": None
        if recovering
        else _read_json(directory / "slot_diagnostic.json", None),
    }


def _runtime_text(runtime: object) -> str:
    """Describe saved M8 runtime facts; missing historical fields stay unknown."""
    if not isinstance(runtime, dict):
        return "Unknown; no runtime record was saved."
    record = cast(Record, runtime)
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        return "Unknown; no M8 runtime provenance was saved."
    facts = cast(Record, provenance)
    return (
        f"Backend: {facts.get('backend', 'Unknown')}; "
        f"Device: {facts.get('device', 'Unknown')}; "
        f"Precision: {facts.get('precision', 'Unknown')}; "
        f"Batch Shape: {facts.get('batch_shape', 'Unknown')}."
    )


def _memory_lines(memory: object, *, label: str) -> list[str]:
    """Format one saved terminal memory record with its original limits and errors.

    label says which recorded status or attempt owns this snapshot. RAM uses bytes;
    allocator counters keep their saved names and values. Missing records/counters
    remain unknown. This formats existing facts and queries no process or device.
    """
    if not isinstance(memory, dict):
        return [f"{label}: Unknown; no memory snapshot was saved.", ""]
    record = cast(Record, memory)
    ram = record.get("process_peak_ram_bytes")
    allocator = record.get("jax_allocator")
    return [
        f"{label}:",
        f"Process Lifetime Peak RAM: {ram} bytes."
        if ram is not None
        else "Process Lifetime Peak RAM: Unknown.",
        f"JAX Allocator Counters: `{json.dumps(allocator, sort_keys=True)}`."
        if allocator is not None
        else "JAX Allocator Counters: Unknown; no counters were available.",
        f"Measurement Scope: {record.get('scope', 'Unknown')}",
        f"Measurement Read Errors: `{json.dumps(record.get('errors'))}`.",
        "",
    ]


def _write_summary(summaries: Sequence[Record], destination: Path) -> Path:
    """Atomically write factual Markdown from already-read per-run summaries."""
    lines = [
        "# Training Run Summary",
        "",
        "This report contains measured records. Useful learning and broader "
        "competence still need an evidence review.",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"## {summary['run_id']}",
                "",
                f"Method: {summary['method']}. Treatment: {summary['treatment']}. "
                f"Training seed: {summary['seed']}.",
                "Declared budget: "
                f"{summary['declared_env_steps']} environment transitions.",
                "Exact settings, source identity (`source`), content identity "
                "(`content_binding`), schema versions (`schemas`), initial runtime "
                "(`runtime`), dependency versions and frozen panel identity "
                f"(`panel_digest`): [run_details.json](<{summary['details_path']}>).",
                f"Frozen panel identity: `{summary['panel_digest']}`. "
                "None means no panel was declared.",
                "",
                f"Status: {summary['status']}. Recorded environment transitions: "
                f"{summary['env_steps']}.",
                f"Validation records: {summary['validation_checkpoints']}. "
                f"Recorded failures: {summary['recorded_failures']}.",
                f"Incomplete event records: {summary['incomplete_events']}. "
                "Their work is unknown.",
                f"Original evidence: `{summary['run_dir']}`.",
                f"Recorded Active Time: {summary['elapsed_seconds']} seconds. "
                f"Wall duration including pauses: {summary['wall_seconds']} seconds. "
                "None means the measurement is unavailable.",
                "Recorded active time carries forward the larger saved checkpoint "
                "or same-run status value, then adds this attempt. Interrupted "
                "work without a durable timing record may be missing. Wall duration "
                "includes the gaps between attempts.",
                "",
            ]
        )
        lines.extend(
            [
                "Schema Versions: "
                + (
                    f"`{json.dumps(summary['schemas'], sort_keys=True)}`."
                    if summary["schemas"] is not None
                    else "Unknown; no schema record was saved."
                ),
                "Initial Runtime: " + _runtime_text(summary["initial_runtime"]),
                "Declared device environment values remain in `runtime` in "
                "run_details.json. These selectors do not verify a physical GPU's "
                "UUID or PCI bus; the frozen launch package owns that check.",
                "",
            ]
        )
        if summary["attempt_runtimes"]:
            lines.append("Recorded Runtime By Attempt:")
            lines.append("")
            for attempt in summary["attempt_runtimes"]:
                lines.append(
                    f"- `{attempt['attempt_id']}`: " + _runtime_text(attempt["runtime"])
                )
            lines.append("")
        lines.extend(
            [
                "Latest Status Memory: Withheld while checkpoint recovery "
                "is unfinished.",
                "",
            ]
            if summary["status"] == "recovering"
            else _memory_lines(summary["memory"], label="Latest Status Memory")
        )
        prior_memory = [
            record
            for record in summary["terminal_memory_history"]
            if record["memory"] != summary["memory"]
        ]
        if prior_memory:
            lines.extend(
                [
                    "Earlier terminal snapshots are historical measurements, "
                    "including failed or abandoned attempts. They are not current "
                    "memory use and their peaks must not be added together.",
                    "",
                ]
            )
            for record in prior_memory:
                lines.extend(
                    _memory_lines(
                        record["memory"],
                        label=f"Recorded {record['event']} / {record['attempt_id']}",
                    )
                )
        if summary["status"] == "recovering":
            lines.extend(
                [
                    "Checkpoint recovery is unfinished. The saved status before "
                    f"recovery was {summary['saved_status']}; it does not describe "
                    "the active attempt. Prior update, validation, actor, selection "
                    "and exposure files are preserved but withheld from this report.",
                    "Resume the checkpoint named in "
                    f"[checkpoint_recovery.json](<{summary['run_dir']}"
                    "/checkpoint_recovery.json>).",
                    "",
                ]
            )
        selected = summary["selection"]
        if isinstance(selected, dict):
            selection = cast(Record, selected)
            lines.extend(
                [
                    "Selected checkpoint: "
                    f"`{selection.get('checkpoint_id', 'Unavailable')}`. "
                    f"Confirmation score: {selection.get('score', 'Unavailable')}.",
                    "",
                ]
            )
        for key, label in (
            ("final_actor", "Final Actor"),
            ("selected_actor", "Selected Actor"),
        ):
            actor = summary[key]
            lines.extend(
                [
                    f"{label}: [{actor}](<{actor}>)."
                    if isinstance(actor, str)
                    else f"{label}: Unavailable in the saved active status.",
                    "",
                ]
            )
        load_path = summary["selected_actor"] or summary["final_actor"]
        if isinstance(load_path, str):
            lines.extend(
                [
                    "Load this saved actor for evaluation:",
                    "",
                    "```python",
                    "from marl_battlegrounds import training",
                    f"system = training.load_system({load_path!r})",
                    "```",
                    "",
                ]
            )
        update = summary["latest_update"]
        if isinstance(update, dict):
            latest = cast(Record, update)
            lines.extend(
                [
                    "Recorded Active Time At The Last Update: "
                    f"{latest.get('elapsed_seconds', 'Unavailable')} seconds. "
                    "Collection/update time: "
                    f"{latest.get('training_seconds', 'Unavailable')} seconds.",
                    "Used policy samples: "
                    f"{latest.get('used_policy_samples', 'Unavailable')}. "
                    "Used value samples: "
                    f"{latest.get('used_value_samples', 'Unavailable')}.",
                    "Losses, entropy, rewards and per-update costs remain in "
                    "learning_curve.csv.",
                    "",
                ]
            )
        lines.extend(
            [
                "Measured phase costs include completed records from all attempts, "
                "including retried work. Interrupted phases may have unmeasured work. "
                "These totals are not a breakdown of only the current branch.",
                "",
                "| Phase | Seconds | Recorded Measurements | Source |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for phase, cost in summary["phase_costs"].items():
            source_file = str(cost["source"]).split(" ", 1)[0]
            seconds = cost["seconds"] if cost["seconds"] is not None else "Unavailable"
            lines.append(
                f"| {phase} | {seconds} "
                f"| {cost['records']} | [{cost['source']}]"
                f"(<{summary['run_dir']}/{source_file}>) |"
            )
        lines.append("")
        if summary["exposure"] is not None:
            lines.extend(
                [
                    "Actual stage, map and opponent exposure, completed games and "
                    "unfinished games: "
                    f"[exposure.json]({summary['run_dir']}/exposure.json).",
                    "",
                ]
            )
        slot = summary["slot_diagnostic"]
        if isinstance(slot, dict):
            diagnostic = cast(Record, slot)
            detected = diagnostic.get("systematic_effect_detected")
            lines.extend(
                [
                    "Final slot diagnostic: "
                    + (
                        "A systematic effect was detected; a fairness claim is blocked."
                        if detected is True
                        else "No systematic effect was detected. "
                        "This is not proof of fairness."
                        if detected is False
                        else "Incomplete; no inference is available."
                    ),
                    "Exact scope and uncertainty: "
                    f"[slot_diagnostic.json]({summary['run_dir']}/slot_diagnostic.json).",
                    "",
                ]
            )
    lines.extend(
        [
            "Intervals describe game sampling for fixed actors. They do not "
            "measure training-seed variation.",
            "Task reward and shaping remain separate. Confirmation scores are "
            "used for selection and are not held-out test results.",
            "Task reward is divided by active-agent samples; shaping is divided "
            "by real environment transitions.",
            "",
            "Exact settings, source/content identities and seeds remain in each "
            "run's run_details.json. Attempts and execution segments remain in "
            "run_events.jsonl.",
            "",
        ]
    )
    target = destination / "run_summary.md"
    _atomic_text(target, "\n".join(lines))
    return target


def refresh_summary(run_dirs: Sequence[str | Path], *, output_dir: str | Path) -> Path:
    """Refresh only factual Markdown after the runner publishes its final status.

    Parameters
    ----------
    run_dirs : sequence of paths
        Nonempty list of exact saved run directories, as accepted by analyze.
    output_dir : path
        Report directory. Only run_summary.md is atomically replaced. Original
        records and existing plots/CSVs are unchanged; no plotting library loads.

    Returns
    -------
    pathlib.Path
        Written run_summary.md. Completion uses saved status.json and requires
        no unfinished checkpoint_recovery.json marker.

    Raises
    ------
    ValueError, OSError
        Required records are missing, malformed or cannot be read or written.
    """
    if not run_dirs:
        raise ValueError("refresh_summary needs at least one exact run directory")
    return _write_summary(
        [_summary(Path(value).resolve()) for value in run_dirs], Path(output_dir)
    )


def analyze(run_dirs: Sequence[str | Path], *, output_dir: str | Path) -> Record:
    """Write factual training curves and summaries from saved trainer records.

    Parameters
    ----------
    run_dirs : sequence of paths
        Exact run directories containing run_details.json, committed
        training_updates.jsonl and optional validation_results.json/selection.json.
        Validation results are a list, or an object with a results list. Rolled-back
        update logs and validation results must already describe the active attempt.
        If checkpoint_recovery.json exists, active plots/results are withheld until
        recovery finishes; preserved old files are not treated as current results.
    output_dir : path
        Report directory. Only derived CSV, PNG, SVG and run_summary.md are replaced.

    Returns
    -------
    dict
        Artifact paths, per-run status and a provisional qualification description.
        A successful report does not certify useful learning or general competence.
        Summaries retain saved schemas, initial/per-attempt runtime and terminal
        memory snapshots. Missing historical facts stay unknown; no device query
        or new memory measurement occurs.

    Raises
    ------
    ValueError, OSError
        Required identity files, record structures or committed log rows are invalid.
    ImportError
        The viz extra is unavailable. Input evidence is never changed.

    Notes
    -----
        Host-only and headless. Plotting uses original task/shaping units and splits
        different frozen panels. Missing results stay missing. No learner is loaded.
    """
    if not run_dirs:
        raise ValueError("analyze needs at least one exact run directory")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    curves: list[Record] = []
    cells: list[Record] = []
    summaries: list[Record] = []
    figure, axes = plt.subplots(
        2, 2, figsize=(12, 8), constrained_layout=True, sharex="col"
    )
    for value in run_dirs:
        directory = Path(value).resolve()
        raw_details = _read_json(directory / "run_details.json", None)
        if not isinstance(raw_details, dict):
            raise ValueError(f"Missing run_details.json in {directory}")
        details = cast(Record, raw_details)
        run_id = str(details.get("run_id", directory.name))
        config = details.get("config", {})
        treatment = _treatment(config)
        run_label = f"{run_id[:8]} / Seed {config.get('seed', '?')} / {treatment}"
        summary = _summary(directory)
        recovering = summary["status"] == "recovering"
        updates = [] if recovering else _read_rows(directory / "training_updates.jsonl")
        raw_results: object = (
            [] if recovering else _read_json(directory / "validation_results.json", [])
        )
        if isinstance(raw_results, dict):
            raw_results = cast(Record, raw_results).get("results", [])
        if not isinstance(raw_results, list) or any(
            not isinstance(row, dict) for row in cast(list[object], raw_results)
        ):
            raise ValueError("validation_results.json must contain result objects")
        results = cast(list[Record], raw_results)
        for row in updates:
            curves.append(
                {
                    "run_id": run_id,
                    "kind": "update",
                    **{
                        key: value
                        for key, value in row.items()
                        if not isinstance(value, (dict, list))
                    },
                }
            )
        for result in results:
            curves.append(
                {
                    "run_id": run_id,
                    "kind": str(result.get("purpose", "validation")),
                    **{
                        key: result.get(key)
                        for key in (
                            "env_steps",
                            "elapsed_seconds",
                            "wall_seconds",
                            "checkpoint_id",
                            "actor_digest",
                            "panel_digest",
                            "complete",
                            "score",
                            "ci_low",
                            "ci_high",
                        )
                    },
                }
            )
            for cell in result.get("cells", []):
                cells.append(
                    {
                        "run_id": run_id,
                        "checkpoint_id": result.get("checkpoint_id"),
                        "purpose": result.get("purpose"),
                        "panel_digest": result.get("panel_digest"),
                        **cell,
                    }
                )
        for x_index, x_name in enumerate(("env_steps", "elapsed_seconds")):
            for name, label in (
                ("task_reward_mean", "Task Reward / Active Agent"),
                ("shaping_mean", "Shaping / Environment"),
            ):
                axes[1, x_index].plot(
                    [row.get(x_name, np.nan) for row in updates],
                    [row.get(name, np.nan) for row in updates],
                    label=f"{run_label}: {label}",
                )
            panels = sorted(
                {str(row.get("panel_digest", "Unknown Panel")) for row in results}
            )
            for panel in panels:
                selected = sorted(
                    (
                        row
                        for row in results
                        if str(row.get("panel_digest", "Unknown Panel")) == panel
                        and row.get("purpose") in ("routine", "initialization")
                    ),
                    key=lambda row: row.get("env_steps", 0),
                )
                if selected:
                    x = [
                        row.get(x_name) if row.get(x_name) is not None else np.nan
                        for row in selected
                    ]
                    scores = [
                        row.get("score") if row.get("complete") else np.nan
                        for row in selected
                    ]
                    plotted = axes[0, x_index].plot(
                        x, scores, marker="o", label=f"{run_label} / Panel {panel[:8]}"
                    )
                    lower = [
                        row.get("ci_low")
                        if row.get("complete") and row.get("ci_low") is not None
                        else np.nan
                        for row in selected
                    ]
                    upper = [
                        row.get("ci_high")
                        if row.get("complete") and row.get("ci_high") is not None
                        else np.nan
                        for row in selected
                    ]
                    axes[0, x_index].vlines(
                        x, lower, upper, colors=plotted[0].get_color(), alpha=0.4
                    )
                    axes[0, x_index].fill_between(
                        x,
                        lower,
                        upper,
                        alpha=0.15,
                    )
        summaries.append(summary)
    for x_index, label in enumerate(
        ("Environment Transitions", "Recorded Active Time (Seconds)")
    ):
        for row_index in range(2):
            axis = axes[row_index, x_index]
            axis.set_xlabel(label)
            axis.set_ylabel(
                "Fixed-Panel Expected Score"
                if row_index == 0
                else "Mean Reward (Separate Denominators)"
            )
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend(fontsize="small")
        axes[0, x_index].set_ylim(-0.02, 1.02)
        if not any(
            np.isfinite(np.asarray(line.get_ydata(), dtype=float)).any()
            for line in axes[0, x_index].lines
        ):
            axes[0, x_index].text(
                0.5,
                0.5,
                "No Validation Results",
                ha="center",
                va="center",
                transform=axes[0, x_index].transAxes,
                color="0.4",
            )
    paths: dict[str, str] = {}
    for suffix in ("png", "svg"):
        target = destination / f"learning_curves.{suffix}"
        temporary = destination / f".learning_curves.{suffix}.tmp"
        figure.savefig(temporary, format=suffix, dpi=150)
        os.replace(temporary, target)
        paths[suffix] = str(target)
    plt.close(figure)
    for key, name, rows in (
        ("curve", "learning_curve.csv", curves),
        ("cells", "validation_cells.csv", cells),
    ):
        target = destination / name
        _atomic_text(target, _csv(rows))
        paths[key] = str(target)
    paths["summary"] = str(_write_summary(summaries, destination))
    return {
        "artifacts": paths,
        "runs": summaries,
        "complete": all(row["status"] == "complete" for row in summaries),
        "qualification": (
            "Provisional; inspect saved evidence before making learning claims"
        ),
    }
