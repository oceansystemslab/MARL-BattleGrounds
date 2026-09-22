"""Summarize saved training and frozen validation without running a learner.

The runner, CLI and researchers share selection, paired uncertainty and report
generation here. M8 remains the owner of recorded game scores and metrics.
Reports read committed trainer records and never rewrite original evidence.
analyze_screen also reports an unfinished configuration screen, retaining shared
Random checks as one statistical task while plotting each case's own timing.
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
    shaping = (
        "Score Delta" if config.get("shaping_mode") == "score_delta" else "Shaping"
    )
    return (
        "+".join(
            name
            for key, name in (("curriculum", "Curriculum"), ("shaping", shaping))
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
        ("Random Validation", "random_validation_complete", "seconds"),
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
        "shaping_mode": config.get("shaping_mode", "potential"),
        "input_scale": config.get("ppo", {}).get("input_scale", 1.0),
        "spawn_frame": config.get("ppo", {}).get("spawn_frame", "world"),
        "pinned_opponent": config.get("pinned_opponent"),
        "pinned_opponent_share": config.get("pinned_opponent_share", 0.0),
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
                f"Shaping Mode: {summary['shaping_mode']}. "
                f"Input Scale: {summary['input_scale']}. "
                f"Spawn Frame: {summary['spawn_frame']}.",
                *(
                    [
                        f"Pinned Opponent: {summary['pinned_opponent']} in "
                        f"{summary['pinned_opponent_share']} of new games, a "
                        "training opponent; its evidence is in exposure.json."
                    ]
                    if summary["pinned_opponent"] is not None
                    else []
                ),
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


def _screen_object(path: Path) -> Record:
    """Read an optional screen object; reject non-object JSON without changing it."""
    value = _read_json(path, {})
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return cast(Record, value)


def _screen_seconds(value: object) -> float | None:
    """Return finite nonnegative saved seconds, or None for an unavailable value."""
    if type(value) not in (float, int):
        return None
    number = float(cast(float, value))
    return number if math.isfinite(number) and number >= 0 else None


def _screen_duration(value: object) -> str:
    """Show saved nonnegative seconds as hours, minutes and seconds, or Unknown."""
    from marl_battlegrounds.training._run_io import duration

    seconds = _screen_seconds(value)
    return "Unknown" if seconds is None else duration(seconds)


def _screen_warm_seconds(updates: Sequence[Record], steps: int) -> float | None:
    """Sum recorded blocks through steps, excluding each attempt's first block.

    updates is the committed active training prefix. A missing block time makes
    the sum unknown. The excluded block contains both cold setup and useful
    numerical work; its duration is not a measurement of compilation alone.
    """
    seen: set[object] = set()
    seconds: list[float] = []
    for row in updates:
        if row.get("env_steps", 0) > steps:
            continue
        value = _screen_seconds(row.get("collection_update_seconds"))
        if value is None:
            return None
        attempt = row.get("attempt_id")
        if attempt in seen:
            seconds.append(value)
        seen.add(attempt)
    return math.fsum(seconds)


def _screen_ancestry(directory: Path, details: Record) -> dict[str, Record]:
    """Read the active checkpoint chain through its existing host integrity owner.

    directory is one run and details is its immutable run_details.json. The
    latest pointer must stay inside checkpoints and match the description. Join
    the run ID and complete config before accepting its active actor boundaries.
    No numerical arrays are loaded, and historical source bytes are not required.
    """
    from marl_battlegrounds.training import checkpoints
    from marl_battlegrounds.training._run_io import checkpoint_ancestry

    pointer = _screen_object(directory / "latest_checkpoint.json")
    identifier = pointer.get("checkpoint_id")
    relative = pointer.get("relative_path")
    if (
        not isinstance(identifier, str)
        or not isinstance(relative, str)
        or relative != f"checkpoints/{identifier}"
        or Path(identifier).name != identifier
    ):
        raise ValueError("Screen run has no valid active checkpoint pointer")
    checkpoint = checkpoints.read_checkpoint_description(directory / relative)
    if checkpoint["checkpoint_id"] != identifier or any(
        checkpoint["metadata"].get(key) != details.get(key)
        for key in ("run_id", "config", "source", "dependencies", "execution")
    ):
        raise ValueError("Screen checkpoint differs from its recorded experiment")
    return checkpoint_ancestry(directory, checkpoint)


def _screen_evidence(
    result: Mapping[str, Any], *, run_id: str | None, seed: int | None
) -> tuple[Record, list[Record]]:
    """Verify one saved Random task and return its summary and original M8 rows.

    result is a runner random_diagnostics entry, including original task and
    artifact identities. run_id and seed bind a fresh capture to its own run;
    both are None only for the explicitly shared original initialization.
    Existing validation owns task and score verification;
    M8 owns CSV loading. No actor is applied and no game or file is changed.
    """
    from marl_battlegrounds.training.validation import (
        _verified_random_result,  # pyright: ignore[reportPrivateUsage]
    )

    return _verified_random_result(
        result,
        actor_digest=result["actor_digest"],
        seed_pairs=result["seed_pairs"],
        checkpoint_id=result["checkpoint_id"],
        env_steps=result["env_steps"],
        run_id=run_id,
        seed=seed,
    )


def _screen_statistics(
    rows: Sequence[Record], result: Record, *, summary: Record | None = None
) -> Record:
    """Keep both spawn ends in each map/seed block for score and kill margin.

    Complete coverage and native scores are checked by summarize_validation.
    A supplied summary must be its verified result for these same rows; reuse it
    to avoid reducing scores twice after the saved-evidence integrity check.
    All points use the same private bootstrap seed, so matching blocks receive
    matching resamples across cases. Optional kill scores must be finite and
    nonnegative when present; absent scores leave that diagnostic unavailable.
    """
    if summary is None:
        summary = summarize_validation(
            rows,
            maps=result["maps"],
            opponents=("Random",),
            seed_pairs=result["seed_pairs"],
        )
    if summary["score"] != result.get("score") or summary["cells"] != result.get(
        "cells"
    ):
        raise ValueError("Random summary differs from its saved game rows")
    blocks: dict[int, dict[int, tuple[float, float | None]]] = {}
    for map_id in result["maps"]:
        grouped: dict[int, list[Record]] = {}
        for row in rows:
            if row["map_id"] == map_id:
                grouped.setdefault(row["seed_id"], []).append(row)
        mapped: dict[int, tuple[float, float | None]] = {}
        for seed, pair in sorted(grouped.items()):
            margins: list[float] = []
            for row in pair:
                first, second = row.get("team_a_score"), row.get("team_b_score")
                if any(
                    value is not None and _screen_seconds(value) is None
                    for value in (first, second)
                ):
                    raise ValueError(
                        "Random kill scores must be finite and nonnegative"
                    )
                if first is not None and second is not None:
                    margins.append(float(first) - float(second))
            mapped[seed] = (
                math.fsum(_score(row["system_game_score"]) for row in pair) / 2,
                math.fsum(margins) / 2 if len(margins) == 2 else None,
            )
        blocks[map_id] = mapped
    margin_blocks = [
        np.asarray([pair[1] for pair in mapped.values()], dtype=np.float64)
        for mapped in blocks.values()
    ]
    if all(np.isfinite(values).all() for values in margin_blocks):
        low, high = _bootstrap(margin_blocks, draws=2000, seed=19_044_001)
        margin: float | None = float(
            np.mean([values.mean() for values in margin_blocks])
        )
    else:
        margin, low, high = None, None, None
    return {
        **summary,
        **{
            name: (
                math.fsum(cell[column] for cell in summary["cells"])
                / len(summary["cells"])
                if all(cell.get(column) is not None for cell in summary["cells"])
                else None
            )
            for name, column in (
                ("mean_kills_for", "mean_team_a_score"),
                ("mean_kills_against", "mean_team_b_score"),
            )
        },
        **{
            name: sum(cell[name] for cell in summary["cells"])
            for name in ("wins", "draws", "losses")
        },
        "kill_margin": margin,
        "kill_margin_ci_low": low,
        "kill_margin_ci_high": high,
        "blocks": blocks,
    }


def _screen_change(current: Record, initial: Record) -> Record:
    """Compare one checkpoint to shared initialization using matching whole blocks.

    Both inputs are _screen_statistics results. Missing or changed map/seed
    coverage raises ValueError; no unpaired fallback is used. Return descriptive
    mean changes and intervals for fixed actors, never training-seed uncertainty.
    """
    before, after = initial["blocks"], current["blocks"]
    if before.keys() != after.keys() or any(
        before[map_id].keys() != after[map_id].keys() for map_id in before
    ):
        raise ValueError("Random checks do not share the same map/seed blocks")
    result: Record = {}
    for index, name in enumerate(("score", "kill_margin")):
        values = [
            np.asarray(
                [
                    after[map_id][seed][index] - before[map_id][seed][index]
                    if after[map_id][seed][index] is not None
                    and before[map_id][seed][index] is not None
                    else np.nan
                    for seed in sorted(before[map_id])
                ],
                dtype=np.float64,
            )
            for map_id in before
        ]
        mean, low, high = None, None, None
        if all(np.isfinite(block).all() for block in values):
            mean = float(np.mean([block.mean() for block in values]))
            low, high = _bootstrap(values, draws=2000, seed=19_044_001)
        result[f"{name}_change_from_initial"] = mean
        result[f"{name}_change_ci_low"] = low
        result[f"{name}_change_ci_high"] = high
    return result


def _screen_task_identity(result: Mapping[str, Any]) -> str:
    """Describe immutable scientific fields, excluding case timing and reuse links."""
    excluded = {
        "elapsed_seconds",
        "training_seconds",
        "wall_seconds",
        "reference_path",
        "reused_initialization",
    }
    return json.dumps(
        {key: value for key, value in result.items() if key not in excluded},
        sort_keys=True,
    )


def _screen_reference(result: Record, package: Path) -> None:
    """Check that a reused initialization keeps its original task and actor IDs.

    reference_path names the copied original summary, absolute or package-relative.
    Its timing belongs to the first case and is deliberately excluded from this
    comparison. Reuse without a complete zero-step reference is rejected.
    """
    reused = result.get("reused_initialization")
    if type(reused) is not bool or reused != (result.get("reference_path") is not None):
        raise ValueError("Random initialization reuse needs its exact reference")
    if not reused:
        return
    value = result.get("reference_path")
    if not isinstance(value, str) or not value:
        raise ValueError("Reused initialization needs its original reference_path")
    path = Path(value)
    original = _screen_object(path if path.is_absolute() else package / path)
    if (
        result.get("env_steps") != 0
        or original.get("env_steps") != 0
        or original.get("complete") is not True
        or original.get("reused_initialization") is not False
        or original.get("reference_path") is not None
        or _screen_task_identity(original) != _screen_task_identity(result)
    ):
        raise ValueError(
            "Shared initialization changed its original scientific identity"
        )


def _screen_plot(points: Sequence[Record], destination: Path) -> dict[str, str]:
    """Plot combat improvement and task score against elapsed minutes and steps.

    Missing x values are not invented. Each case references its shared initial
    task, but intervals were computed only once for each unique task. Write two
    derived images atomically; original evidence and numerical state stay intact.
    The first row uses paired kill-margin change since initialization, which can
    distinguish short-run combat progress even when every game ends in a draw.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    names = tuple(dict.fromkeys(row["case"] for row in points))
    batches = sorted({row["num_envs"] for row in points})
    lengths = sorted({row["rollout_length"] for row in points})
    colors = plt.get_cmap("tab10")
    styles = ("-", "--", "-.", ":")
    handles: list[Any] = []
    labels: list[str] = []
    for name in names:
        rows = sorted(
            (row for row in points if row["case"] == name),
            key=lambda row: row["env_steps"],
        )
        color = colors(batches.index(rows[0]["num_envs"]) % 10)
        style = styles[lengths.index(rows[0]["rollout_length"]) % len(styles)]
        for column, x_name in enumerate(
            ("wall_seconds", "warm_training_seconds", "env_steps")
        ):
            for index, metric in enumerate(
                ("kill_margin_change_from_initial", "score")
            ):
                selected = [
                    row
                    for row in rows
                    if row.get(x_name) is not None and row.get(metric) is not None
                ]
                if not selected:
                    continue
                divisor = 1 if x_name == "env_steps" else 60
                x = [row[x_name] / divisor for row in selected]
                low_key = "ci_low" if metric == "score" else "kill_margin_change_ci_low"
                high_key = (
                    "ci_high" if metric == "score" else "kill_margin_change_ci_high"
                )
                line = axes[index, column].plot(
                    x,
                    [row[metric] for row in selected],
                    marker="o",
                    markersize=4,
                    linewidth=1.4,
                    color=color,
                    linestyle=style,
                    label=name,
                )[0]
                axes[index, column].vlines(
                    x,
                    [row[low_key] for row in selected],
                    [row[high_key] for row in selected],
                    colors=line.get_color(),
                    alpha=0.3,
                )
                if name not in labels:
                    handles.append(line)
                    labels.append(name)
    for column, label in enumerate(
        (
            "Elapsed Run Time (Minutes, Including Overhead)",
            "Training Time After Each Attempt's First Block (Minutes)",
            "Environment Transitions",
        )
    ):
        for index, metric in enumerate(
            (
                "Change In Kills Minus Deaths Per Game\n"
                "Since Before Training (Higher Is Better)",
                "Game Result Score Against Random\nWin = 1, Draw = 0.5, Loss = 0",
            )
        ):
            axis = axes[index, column]
            axis.set_xlabel(label)
            axis.set_ylabel(metric)
            axis.grid(alpha=0.2)
            if not axis.lines:
                axis.text(
                    0.5,
                    0.5,
                    "No Completed Games Against Random",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="0.4",
                )
        axes[1, column].set_ylim(-0.02, 1.02)
        axes[0, column].axhline(0, color="0.5", linewidth=0.7, linestyle=":")
    if handles:
        figure.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    figure.suptitle(
        "Short MAPPO Screen: Combat Improvement For Time Spent\n"
        "Games Against Random; Error Bars Show Game Variation, "
        "Not Training-Seed Variation",
        fontsize=13,
    )
    paths: dict[str, str] = {}
    try:
        for suffix in ("png", "svg"):
            path = destination / f"learning_curves.{suffix}"
            temporary = destination / f".learning_curves.{suffix}.tmp"
            figure.savefig(temporary, format=suffix, dpi=150)
            os.replace(temporary, path)
            paths[suffix] = str(path)
    finally:
        plt.close(figure)
    return paths


def analyze_screen(
    package_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    render_plots: bool = True,
) -> Record:
    """Report every declared configuration from a complete or unfinished screen.

    Parameters
    ----------
    package_dir : path
        Frozen screen package with declaration.json and its ordered cases list.
        Optional study.json and budgets.json hold lifecycle facts and fixed
        timing-based budgets. Trials use jobs/<name>/run and ordinary trainer
        records, including random_diagnostics.json. Referenced shared initial
        results retain their original task/checkpoint/actor IDs and raw M8 paths.
        Initial actors are shared only within one training seed. Started cases
        keep their frozen config, latest checkpoint pointer and
        ancestor descriptions. Original actor and M8 evidence remain readable.
    output_dir : path or None, default=None
        Destination for derived reports, default package_dir/reports. Replace only
        run_summary.md, baseline_trials.csv, validation_cells.csv,
        learning_curve.csv and learning_curves.png/.svg.
    render_plots : bool, default=True
        Draw the ordinary screen plots. False skips plotting and its imports;
        callers making a larger study report can reuse the verified tables.

    Returns
    -------
    dict
        artifacts maps summary/trials/cells/curve/png/svg to paths. cases includes
        failed and unstarted trials. unique_validation_tasks counts shared checks
        once. evidence_errors names unreadable or inconsistent case evidence.
        complete requires every raw run and expected Random capture to be complete;
        it does not depend on the driver's terminal status or select a winner.

    Raises
    ------
    ValueError, OSError
        The declaration is missing, malformed or repeats case names, or reports
        cannot be written. Invalid individual runs remain visible as evidence
        errors, allowing valid cases and unfinished studies to be reported.
    ImportError
        Plotting dependencies from the viz extra are unavailable.

    Notes
    -----
    Host-only, headless analysis; no learner or game runs. Shared map/seed blocks
    keep both spawn ends together. Changes from initialization use matching blocks.
    Intervals describe game sampling for fixed actors, not training-seed variation.
    Combat change against Random over wall time is the short-screen comparison;
    native outcomes remain separate. All draws do not establish absent learning.
    The secondary training axis excludes each attempt's
    first complete block, including its useful work; that block is not pure compile
    time. Missing timings remain missing. Original evidence is never changed.
    """
    package = Path(package_dir).resolve()
    if type(render_plots) is not bool:
        raise TypeError("render_plots must be bool")
    declaration = _screen_object(package / "declaration.json")
    raw_cases = declaration.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Screen declaration needs a nonempty ordered cases list")
    declared: list[Record] = []
    names: set[str] = set()
    for raw in cast(list[object], raw_cases):
        if not isinstance(raw, dict):
            raise ValueError("Each declared screen case must be an object")
        row = cast(Record, raw)
        name = row.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or Path(name).name != name
            or name in (".", "..")
        ):
            raise ValueError("Screen case names must be distinct directory names")
        names.add(name)
        declared.append(row)
    study = _screen_object(package / "study.json")
    budgets = _screen_object(package / "budgets.json")
    budget_rows = budgets.get("cases", [])
    if not isinstance(budget_rows, list):
        raise ValueError("Screen budgets need a cases list")
    by_budget: dict[str, Record] = {}
    for value in cast(list[object], budget_rows):
        if not isinstance(value, dict):
            raise ValueError("Each screen budget must be an object")
        row = cast(Record, value)
        if row.get("name") not in names or row["name"] in by_budget:
            raise ValueError("Screen budgets have an unknown or repeated case")
        by_budget[row["name"]] = row
    destination = (
        Path(output_dir).resolve() if output_dir is not None else package / "reports"
    )
    destination.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, Record] = {}
    shared = _screen_object(package / "shared" / "initialization.json")
    initial_identities: dict[int, tuple[object, object]] = {}
    cases: list[Record] = []
    curves: list[Record] = []
    points: list[Record] = []
    errors: list[Record] = []
    for definition in declared:
        name = definition["name"]
        budget = by_budget.get(name, {})
        lifecycle = study.get("cases", {}).get(name, {})
        directory = package / "jobs" / name / "run"
        case: Record = {
            "case": name,
            "num_envs": definition.get("num_envs"),
            "rollout_length": definition.get("rollout_length"),
            "status": lifecycle.get("status", "unstarted"),
            "phase": lifecycle.get("phase"),
            "error": lifecycle.get("error"),
            "run_dir": str(directory),
            "declared_env_steps": budget.get("total_env_steps"),
            "declared_updates": budget.get("updates"),
            "calibration_median_update_seconds": budget.get("median_update_seconds"),
            "complete": False,
        }
        cases.append(case)
        if not (directory / "run_details.json").exists():
            if case["status"] == "complete":
                errors.append(
                    {"case": name, "error": "Complete case has no run_details.json"}
                )
            continue
        try:
            summary = _summary(directory)
            case.update(
                {
                    key: summary.get(key)
                    for key in (
                        "run_id",
                        "status",
                        "env_steps",
                        "wall_seconds",
                        "elapsed_seconds",
                        "phase_costs",
                        "memory",
                        "exposure",
                        "recorded_failures",
                        "incomplete_events",
                        "final_actor",
                        "seed",
                        "details_path",
                    )
                }
            )
            if summary["status"] == "recovering":
                continue
            details = _screen_object(directory / "run_details.json")
            config = cast(Record, details.get("config", {}))
            training_seed = _integer(config.get("seed"), "training seed")
            if definition.get("seed", training_seed) != training_seed:
                raise ValueError("Saved training seed differs from its declaration")
            declared_config = budget.get("config")
            if declared_config is None and budget.get("config_path"):
                declared_config = _screen_object(package / budget["config_path"])
            if declared_config is not None and config != declared_config:
                raise ValueError("Saved config differs from its frozen case config")
            if budget and declared_config is None:
                raise ValueError("Screen budget has no frozen case config")
            if (
                config.get("num_envs") != definition.get("num_envs")
                or config.get("ppo", {}).get("rollout_length")
                != definition.get("rollout_length")
                or (
                    budget.get("total_env_steps") is not None
                    and config.get("total_env_steps") != budget["total_env_steps"]
                )
            ):
                raise ValueError("Saved run differs from its declared shape or budget")
            ancestry = _screen_ancestry(directory, details)
            exposure = summary.get("exposure")
            if isinstance(exposure, dict):
                case["exposure"] = {
                    key: exposure.get(key)
                    for key in (
                        "starts_by_episode_stage",
                        "steps_by_episode_stage",
                        "actor_decisions_by_episode_stage",
                        "steps_by_map",
                        "starts_by_opponent",
                        "steps_by_opponent",
                        "stage_completed",
                        "stage_k20_count",
                        "incomplete_games",
                    )
                }
            updates = _read_rows(directory / "training_updates.jsonl")
            latest = updates[-1] if updates else {}
            case.update(
                {
                    key: latest.get(key)
                    for key in (
                        "training_seconds",
                        "actor_decisions",
                        "used_policy_samples",
                        "used_value_samples",
                    )
                }
            )
            case["first_block_seconds"] = (
                updates[0].get("collection_update_seconds") if updates else None
            )
            case["warm_training_seconds"] = _screen_warm_seconds(
                updates, latest.get("env_steps", 0)
            )
            for row in updates:
                curves.append(
                    {
                        "case": name,
                        "kind": "update",
                        **{
                            key: value
                            for key, value in row.items()
                            if not isinstance(value, (dict, list))
                        },
                    }
                )
            raw_results = _read_json(directory / "random_diagnostics.json", [])
            if not isinstance(raw_results, list):
                raise ValueError("random_diagnostics.json must contain a list")
            seen_steps: set[int] = set()
            local: list[Record] = []
            for raw in cast(list[object], raw_results):
                if not isinstance(raw, dict):
                    raise ValueError("Random diagnostic entries must be objects")
                result = cast(Record, raw)
                if result.get("seed_pairs") != config.get(
                    "random_diagnostic_seed_pairs"
                ):
                    raise ValueError(
                        "Random seed-pair count differs from the run config"
                    )
                steps = _integer(result.get("env_steps"), "diagnostic env_steps")
                if steps in seen_steps:
                    raise ValueError("A case repeats a Random diagnostic step")
                seen_steps.add(steps)
                if result.get("complete") is not True:
                    continue
                _screen_reference(result, package)
                from marl_battlegrounds.training.checkpoints import (
                    _inference_digest,  # pyright: ignore[reportPrivateUsage]
                )

                reused = result["reused_initialization"]
                reference = config.get("random_initialization_result")
                expected_reuse = steps == 0 and reference is not None
                expected_reference = (
                    str(Path(cast(str, reference)).absolute())
                    if expected_reuse
                    else None
                )
                if (
                    reused != expected_reuse
                    or result.get("reference_path") != expected_reference
                ):
                    raise ValueError(
                        "Random initialization reuse differs from its declared config"
                    )
                origin = result.get("checkpoint_id")
                if not isinstance(origin, str):
                    raise ValueError("Random result needs its originating checkpoint")
                boundary = (
                    next(
                        (
                            item
                            for item in ancestry.values()
                            if item["counters"]["env_steps"] == 0
                        ),
                        None,
                    )
                    if reused
                    else ancestry.get(origin)
                )
                if (
                    boundary is None
                    or boundary["counters"]["env_steps"] != steps
                    or _inference_digest(boundary) != result.get("actor_digest")
                ):
                    raise ValueError(
                        "Random actor is outside this run's active checkpoint chain"
                    )
                task_id = result.get("task_id")
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError("Random diagnostic needs its original task_id")
                if steps == 0:
                    identity_pair = (task_id, result.get("actor_digest"))
                    expected_initial = initial_identities.setdefault(
                        training_seed,
                        (shared["task_id"], shared.get("actor_digest"))
                        if shared.get("task_id") is not None
                        else identity_pair,
                    )
                    if expected_initial != identity_pair:
                        raise ValueError(
                            "Each training seed must share one original initial "
                            "task and actor"
                        )
                identity = _screen_task_identity(result)
                if task_id in tasks:
                    if tasks[task_id]["identity"] != identity:
                        raise ValueError(
                            "A shared task ID has conflicting scientific evidence"
                        )
                else:
                    verified, rows = _screen_evidence(
                        result,
                        run_id=None if reused else details["run_id"],
                        seed=None if reused else config["seed"],
                    )
                    statistics = _screen_statistics(rows, result, summary=verified)
                    tasks[task_id] = {
                        "identity": identity,
                        "result": result,
                        "statistics": statistics,
                        "cases": [],
                    }
                task = tasks[task_id]
                if name not in task["cases"]:
                    task["cases"].append(name)
                point: Record = {
                    "case": name,
                    "kind": "random",
                    "num_envs": definition["num_envs"],
                    "rollout_length": definition["rollout_length"],
                    "env_steps": steps,
                    "task_id": task_id,
                    "checkpoint_id": result.get("checkpoint_id"),
                    "actor_digest": result.get("actor_digest"),
                    "actor_path": result.get("actor_path"),
                    "summary_path": result.get("summary_path"),
                    "reference_path": result.get("reference_path"),
                    "reused_initialization": bool(result.get("reused_initialization")),
                    "wall_seconds": _screen_seconds(result.get("wall_seconds")),
                    "elapsed_seconds": _screen_seconds(result.get("elapsed_seconds")),
                    "training_seconds": _screen_seconds(result.get("training_seconds")),
                    "warm_training_seconds": _screen_warm_seconds(updates, steps),
                    **{
                        key: value
                        for key, value in task["statistics"].items()
                        if key not in ("blocks", "cells")
                    },
                }
                local.append(point)
            initial = next((row for row in local if row["env_steps"] == 0), None)
            if initial is not None:
                reference = tasks[initial["task_id"]]["statistics"]
                for point in local:
                    point.update(
                        _screen_change(tasks[point["task_id"]]["statistics"], reference)
                    )
                    point["initial_task_id"] = initial["task_id"]
            points.extend(local)
            curves.extend(local)
            total = _integer(
                budget.get("total_env_steps", config.get("total_env_steps")),
                "declared total_env_steps",
                minimum=1,
            )
            expected = {0, total}
            expected.update(
                _integer(value, "diagnostic checkpoint_env_steps", minimum=1)
                for value in config.get(
                    "checkpoint_env_steps", budget.get("checkpoint_env_steps", [])
                )
            )
            completed = {row["env_steps"] for row in local}
            case["expected_diagnostic_steps"] = sorted(expected)
            case["completed_diagnostic_steps"] = sorted(completed)
            final = next((row for row in local if row["env_steps"] == total), None)
            if final is not None:
                case.update(
                    {
                        f"final_{key}": final.get(key)
                        for key in (
                            "score",
                            "ci_low",
                            "ci_high",
                            "kill_margin",
                            "kill_margin_ci_low",
                            "kill_margin_ci_high",
                            "mean_kills_for",
                            "mean_kills_against",
                            "wins",
                            "draws",
                            "losses",
                            "score_change_from_initial",
                            "score_change_ci_low",
                            "score_change_ci_high",
                            "kill_margin_change_from_initial",
                            "kill_margin_change_ci_low",
                            "kill_margin_change_ci_high",
                            "task_id",
                        )
                    }
                )
            case["complete"] = (
                case["status"] == "complete"
                and case["env_steps"] == total
                and expected == completed
            )
        except (ValueError, OSError, KeyError, TypeError) as error:
            case["evidence_error"] = str(error)
            case["complete"] = False
            errors.append({"case": name, "error": str(error)})
    cells: list[Record] = []
    for task_id, task in tasks.items():
        result = task["result"]
        for cell in task["statistics"]["cells"]:
            cells.append(
                {
                    "task_id": task_id,
                    "cases": json.dumps(task["cases"]),
                    "checkpoint_id": result.get("checkpoint_id"),
                    "actor_digest": result.get("actor_digest"),
                    "env_steps": result["env_steps"],
                    "summary_path": result.get("summary_path"),
                    **cell,
                    "mean_kill_margin": (
                        cell["mean_team_a_score"] - cell["mean_team_b_score"]
                        if cell.get("mean_team_a_score") is not None
                        and cell.get("mean_team_b_score") is not None
                        else None
                    ),
                }
            )
    artifacts = _screen_plot(points, destination) if render_plots else {}
    for key, filename, rows in (
        ("trials", "baseline_trials.csv", cases),
        ("cells", "validation_cells.csv", cells),
        ("curve", "learning_curve.csv", curves),
    ):
        path = destination / filename
        scalar_rows = [
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            for row in rows
        ]
        empty_columns = {
            "trials": "case,status,complete\n",
            "cells": "task_id,cases,map_id,score,mean_kill_margin\n",
            "curve": "case,kind,env_steps,wall_seconds,score,kill_margin\n",
        }
        _atomic_text(path, _csv(scalar_rows) if scalar_rows else empty_columns[key])
        artifacts[key] = str(path)
    complete = (
        len(by_budget) == len(cases)
        and all(row["complete"] for row in cases)
        and not errors
    )
    lines = [
        "# Configuration Screen Summary",
        "",
        f"Study Status: {study.get('status', 'Not Started')}. "
        f"Phase: {study.get('phase', 'Unavailable')}.",
        f"Complete Case Evidence: {sum(row['complete'] for row in cases)}"
        f"/{len(cases)}.",
        f"Total Study Time: {_screen_duration(study.get('elapsed_seconds'))}.",
        f"Unique Random Tasks: {len(tasks)}. Unique Evaluated Games: "
        f"{sum(task['statistics']['games'] for task in tasks.values())}.",
        "",
        "Frozen recipe, seed, order and grid: "
        f"[declaration.json]({package / 'declaration.json'}). "
        "Timing-based budgets and forecast: "
        f"[budgets.json]({package / 'budgets.json'}). "
        f"Original lifecycle: [study.json]({package / 'study.json'}).",
        "",
        "This short screen compares combat improvement against Random for the "
        "time spent. Wins are not required. We compare the change in average "
        "kills minus deaths per game since before training; higher is better. "
        "Kills and deaths are also shown separately. Winning the original task "
        "remains a separate outcome (win 1, draw 0.5, loss 0). "
        "No winner or next trial is selected automatically.",
        "Elapsed run time is the primary plot axis, in minutes. It includes "
        "setup, previous checks, saving and any pauses. Training time and environment "
        "transitions are secondary. The training axis excludes each attempt's "
        "first block, including its useful work; the excluded duration is not "
        "pure compilation time. Missing measurements stay unknown.",
        "Each capture's wall clock is saved just before its Random check. Final "
        "run and study totals also include that check and report generation.",
        "Intervals are descriptive game-sampling intervals for fixed actors. "
        "Whole map/seed blocks keep both spawn ends together, using matching "
        "resamples across cases. Changes from initialization use those same "
        "blocks. A shared initialization is counted once; case curves only "
        "reference it. These intervals do not measure training variation. "
        "Compare separate training seeds in the study's configuration summary; "
        "a short development screen alone does not establish a final best setting.",
        "The first case pays for the shared initialization evaluation in its wall "
        "time. Later cases reuse that evidence; their evaluation costs are lower "
        "by that shared work. The separate study clock includes the full cost.",
        "Different budgets change self-play snapshot timing. Equal-experience "
        "checkpoints compare these complete procedures. All draws mean that "
        "win/draw/loss outcomes do not distinguish these actors. They do not "
        "show that nothing was learned: kills and deaths can still improve. "
        "An interval collapsed to one number does not prove certainty beyond "
        "the games played. Combat progress against Random does not establish "
        "competence against stronger opponents.",
        "",
        "## Cases",
        "",
        "| Configuration | State | Steps Completed | Elapsed Run Time | "
        "Kills / Deaths Per Game | Change In Kills Minus Deaths | "
        "Wins / Draws / Losses |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for case in cases:

        def shown(key: str, record: Record = case) -> str:
            """Format one optional saved case value; missing facts stay unavailable."""
            value = record.get(key)
            if value is None:
                return "Unknown"
            return f"{value:.2f}" if isinstance(value, float) else str(value)

        lines.append(
            f"| {case['case']} | {shown('status')} | {shown('env_steps')} | "
            f"{_screen_duration(case.get('wall_seconds'))} | "
            f"{shown('final_mean_kills_for')} / {shown('final_mean_kills_against')} | "
            f"{shown('final_kill_margin_change_from_initial')} | "
            f"{shown('final_wins')} / {shown('final_draws')} / "
            f"{shown('final_losses')} |"
        )
    for case in cases:
        lines.extend(
            [
                "",
                f"### {case['case']}",
                "",
                f"Original run: [{case['run_dir']}]({case['run_dir']}).",
            ]
        )
        change = case.get("final_kill_margin_change_from_initial")
        if change is not None:
            lines.append(
                f"Change In Kills Minus Deaths: {change:+.2f} per game since "
                "before training. Paired Game Interval: "
                f"{case['final_kill_margin_change_ci_low']:+.2f} to "
                f"{case['final_kill_margin_change_ci_high']:+.2f}. "
                "This range reflects the sampled games, not different training seeds."
            )
        if case.get("error") or case.get("evidence_error"):
            lines.append(
                "Saved Failure Or Evidence Error: "
                f"{case.get('evidence_error') or case.get('error')}."
            )
        if "phase_costs" in case:
            lines.append(
                "Measured phase-owner costs across attempts (seconds): "
                + ", ".join(
                    f"{name}: {value['seconds']}"
                    if value["seconds"] is not None
                    else f"{name}: Unavailable"
                    for name, value in case["phase_costs"].items()
                )
                + ". Interrupted work can be missing from these measurements."
            )
        if case.get("memory") is not None:
            lines.extend(_memory_lines(case["memory"], label="Latest Saved Memory"))
        if case.get("exposure") is not None:
            lines.append(
                "Actual map, opponent and actor exposure: "
                f"[exposure.json]({Path(case['run_dir']) / 'exposure.json'})."
            )
        if case.get("final_actor"):
            lines.append(
                f"Final actor: [{case['final_actor']}]({case['final_actor']})."
            )
    if errors:
        lines.extend(["", "## Evidence Errors", ""])
        lines.extend(f"- {row['case']}: {row['error']}" for row in errors)
    path = destination / "run_summary.md"
    _atomic_text(path, "\n".join(lines) + "\n")
    artifacts["summary"] = str(path)
    return {
        "artifacts": artifacts,
        "cases": cases,
        "complete": complete,
        "unique_validation_tasks": len(tasks),
        "evidence_errors": errors,
        "qualification": (
            "Descriptive one-seed configuration screen; "
            "no automatic winner or learning qualification"
        ),
    }
