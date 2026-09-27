"""Check fixed-panel scores, dependent uncertainty, selection and saved reports.

Known outcome tables establish expected scores without the production reducer.
Reports keep training/validation Red Zone labels and separate depth curves.
Missing depths stay unknown, distinct from zero. They keep reward sources and
panels separate, preserve incomplete states
and leave original records untouched. New choices use only mean point margins;
historical choices keep their saved rules. No learner or environment runs here.
Reports name saved artifacts and sum only phase-owner timing records. An open
checkpoint recovery must hide stale completion and results until recovery ends.
Selection keeps compatible PPO and QMIX rows together, drops QMIX warmup actors
with zero optimizer steps and rejects missing or malformed QMIX counts; the
same holds for PQN-VDN rows, whose initial-collection actors have zero
optimizer steps. A PQN-VDN run's report reads its "pqn" settings block, adds
its summary line (optimizer steps, learning blocks, used and kept-row pairs,
initial transitions never learned, capped by the transitions generated so far)
and draws pqn_curves from scalar columns only, keeping the used_exposure
dictionary out of the CSV.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from marl_battlegrounds.training.analysis import (
    analyze,
    confirmation_candidates,
    refresh_summary,
    select_checkpoint,
    summarize_validation,
)


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "map_id": 42,
            "seed_id": seed + 1,
            "opponent": opponent,
            "spawn_locations": end,
            "system_game_score": float(seed),
            "team_a_score": seed * 20,
            "team_b_score": (1 - seed) * 20,
            "episode_length": 12,
        }
        for seed in (0, 1)
        for opponent in ("Halfway", "Final")
        for end in (0, 1)
    ]


def _result(identifier: str, steps: int, score: float, **values: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "checkpoint_id": identifier,
        "env_steps": steps,
        "score": score,
        "complete": True,
        "panel_digest": "panel",
        "purpose": "routine",
        "cells": [{"mean_team_a_score": 10 * score, "mean_team_b_score": 0.0}],
        **values,
    }


def test_shared_opponent_and_spawn_noise_stays_one_independent_block() -> None:
    result = summarize_validation(
        _rows(), maps=(42,), opponents=("Halfway", "Final"), seed_pairs=2
    )
    assert result["games"] == 8
    assert result["independent_blocks"] == 2
    assert result["score"] == 0.5
    assert (result["ci_low"], result["ci_high"]) == (0.0, 1.0)
    assert all(cell["wins"] == cell["losses"] == 2 for cell in result["cells"])
    assert all(cell["mean_team_a_score"] == 10 for cell in result["cells"])
    shuffled = list(reversed(_rows()))
    assert (
        summarize_validation(
            shuffled, maps=(42,), opponents=("Halfway", "Final"), seed_pairs=2
        )
        == result
    )


def test_equal_map_and_opponent_weight_with_exact_wdl() -> None:
    rows = _rows()
    rows.extend(
        {**row, "map_id": 43, "seed_id": row["seed_id"] + 2, "system_game_score": 1.0}
        for row in _rows()
    )
    result = summarize_validation(
        rows, maps=(42, 43), opponents=("Halfway", "Final"), seed_pairs=2
    )
    assert result["score"] == 0.75
    assert sum(row["wins"] for row in result["cells"]) == 12


@pytest.mark.parametrize("change", ("missing", "duplicate", "seed", "score", "map"))
def test_incomplete_or_conflicting_cells_are_rejected(change: str) -> None:
    rows = _rows()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "seed":
        rows[-1]["seed_id"] = 99
    elif change == "score":
        rows[-1]["system_game_score"] = float("nan")
    else:
        rows[-1]["map_id"] = 43
    with pytest.raises(ValueError):
        summarize_validation(
            rows, maps=(42,), opponents=("Halfway", "Final"), seed_pairs=2
        )


def test_selection_ignores_initialization_preserves_final_and_earlier_tie() -> None:
    rows = [
        _result("init", 0, 1.0),
        _result("a", 4, 0.8),
        _result("b", 8, 0.8),
        _result("final", 12, 0.5),
    ]
    assert confirmation_candidates(rows, final_checkpoint_id="final") == (
        "a",
        "b",
        "final",
    )
    confirmed = [{**row, "purpose": "confirmation", "score": 0.5} for row in rows[1:]]
    assert select_checkpoint(confirmed)["checkpoint_id"] == "a"
    assert confirmation_candidates(rows[:3], final_checkpoint_id="b") == ("a", "b")


@pytest.mark.parametrize(
    "change", ("incomplete", "panel", "duplicate", "missing_final", "nan")
)
def test_selection_rejects_bad_evidence(change: str) -> None:
    rows = [_result("a", 4, 0.8), _result("final", 8, 0.4)]
    final = "final"
    if change == "incomplete":
        rows[0]["complete"] = False
    elif change == "panel":
        rows[0]["panel_digest"] = "other"
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "nan":
        rows[0]["score"] = float("nan")
    else:
        final = "missing"
    with pytest.raises(ValueError):
        confirmation_candidates(rows, final_checkpoint_id=final)
    with pytest.raises(ValueError, match="fresh confirmation"):
        select_checkpoint([_result("a", 4, 0.8)])


def test_reports_use_saved_records_keep_missing_values_and_do_not_edit_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_details.json").write_text(
        json.dumps({"run_id": "Example", "config": {"seed": 7}})
    )
    updates = [
        {
            "env_steps": 4,
            "elapsed_seconds": 1.0,
            "task_reward_mean": 0.25,
            "shaping_mean": 0.01,
        },
        {
            "env_steps": 8,
            "elapsed_seconds": 2.0,
            "task_reward_mean": 0.5,
            "shaping_mean": 0.02,
        },
    ]
    (root / "training_updates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in updates)
    )
    (root / "run_events.jsonl").write_text(json.dumps({"event": "failure"}) + "\n")
    results = [
        _result("a", 4, 0.5, elapsed_seconds=1.0, ci_low=0.2, ci_high=0.8),
        _result("b", 8, 0.0, complete=False, elapsed_seconds=2.0),
    ]
    (root / "validation_results.json").write_text(json.dumps(results))
    (root / "status.json").write_text(
        json.dumps({"status": "failed", "elapsed_seconds": 3.0, "wall_seconds": 8.0})
    )
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    report = analyze([root], output_dir=tmp_path / "report")
    assert not report["complete"]
    assert report["runs"][0]["recorded_failures"] == 1
    for path in report["artifacts"].values():
        assert Path(path).is_file()
    assert Path(report["artifacts"]["png"]).read_bytes().startswith(b"\x89PNG")
    assert "Fixed-Panel Expected Score" in Path(report["artifacts"]["svg"]).read_text()
    assert "failed" in Path(report["artifacts"]["summary"]).read_text()
    assert (
        "Recorded Active Time: 3.0 seconds"
        in Path(report["artifacts"]["summary"]).read_text()
    )
    assert (
        "Wall duration including pauses: 8.0 seconds"
        in Path(report["artifacts"]["summary"]).read_text()
    )
    curve = Path(report["artifacts"]["curve"]).read_text()
    assert "task_reward_mean" in curve and "shaping_mean" in curve
    again = analyze([root], output_dir=tmp_path / "report")
    assert again == report
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    old_outputs = {
        name: Path(path).read_bytes()
        for name, path in report["artifacts"].items()
        if name != "summary"
    }
    (root / "status.json").write_text(json.dumps({"status": "complete"}))
    summary = refresh_summary([root], output_dir=tmp_path / "report")
    assert "Status: complete" in summary.read_text()
    assert {
        name: Path(report["artifacts"][name]).read_bytes() for name in old_outputs
    } == old_outputs
    (root / "training_updates.jsonl").write_text('{"env_steps":')
    with pytest.raises(ValueError, match="Incomplete committed"):
        analyze([root], output_dir=tmp_path / "report")


def test_missing_measurements_are_not_zero() -> None:
    rows = _rows()
    rows[0]["episode_length"] = None
    result = summarize_validation(
        rows, maps=(42,), opponents=("Halfway", "Final"), seed_pairs=2
    )
    assert result["cells"][0]["mean_episode_length"] is None
    assert np.isfinite(result["score"])


def test_torn_event_history_stays_visible_without_inventing_success(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.training.analysis import (
        _read_events,  # pyright: ignore[reportPrivateUsage]
    )

    path = tmp_path / "run_events.jsonl"
    fragment = '{"event":"attempt_complete"'
    path.write_text(fragment)
    assert _read_events(path) == [{"event": "incomplete_event", "line": 1}]
    path.write_text(fragment + '\n{"event":"attempt_started"}\n')
    before = path.read_bytes()
    assert _read_events(path) == [
        {"event": "incomplete_event", "line": 1},
        {"event": "attempt_started"},
    ]
    assert path.read_bytes() == before


def test_empty_validation_is_labelled_and_slot_qualification_stays_visible(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_details.json").write_text(
        json.dumps({"run_id": "a" * 64, "config": {"seed": 42}})
    )
    (run / "slot_diagnostic.json").write_text(
        json.dumps({"systematic_effect_detected": True})
    )
    report = analyze([run], output_dir=tmp_path / "report")
    assert "No Validation Results" in Path(report["artifacts"]["svg"]).read_text()
    assert (
        "fairness claim is blocked" in Path(report["artifacts"]["summary"]).read_text()
    )


def test_summary_names_exact_run_and_actors_and_counts_phase_costs_once(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.training.analysis import (
        _summary,  # pyright: ignore[reportPrivateUsage]
    )

    run = tmp_path / "run with spaces"
    run.mkdir()
    final = str(run / "actors" / "final")
    selected = str(run / "actors" / "selected")
    (run / "run_details.json").write_text(
        json.dumps(
            {
                "run_id": "named",
                "config": {
                    "method": "mappo",
                    "seed": 19,
                    "total_env_steps": 1000,
                    "curriculum": True,
                    "shaping": True,
                },
                "panel_digest": "exact-panel",
                "source": {"source_tree_digest": "exact-source"},
                "content_binding": {"content": "exact-content"},
                "initial_setup_seconds": 99.0,
            }
        )
    )
    (run / "status.json").write_text(
        json.dumps(
            {"status": "complete", "final_actor": final, "selected_actor": selected}
        )
    )
    events = [
        {"event": "attempt_started", "setup_seconds": 2.0},
        {"event": "attempt_started", "setup_seconds": 3.0},
        {"event": "evaluation_segment_complete", "seconds": 7.0},
        {"event": "validation_complete", "result": {"score": 0.7}},
        {"event": "validation_complete", "seconds": 11.0},
        {"event": "checkpoint_saved", "seconds": 4.0},
        {"event": "checkpoint_saved", "seconds": 6.0},
        {"event": "reports_written", "seconds": 0.0},
        {"event": "slot_diagnostic_finished", "seconds": 13.0},
    ]
    (run / "run_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    facts = _summary(run)
    assert facts["phase_costs"]["Setup"]["seconds"] == 5.0
    assert facts["phase_costs"]["Validation"]["seconds"] == 11.0
    assert facts["phase_costs"]["Validation"]["records"] == 1
    assert facts["phase_costs"]["Checkpoint"]["seconds"] == 10.0
    assert facts["phase_costs"]["Report"]["seconds"] == 0.0
    assert facts["phase_costs"]["Slot Diagnostic"]["seconds"] == 13.0
    output = refresh_summary([run], output_dir=tmp_path / "report").read_text()
    assert "Method: mappo. Treatment: Curriculum+Shaping. Training seed: 19" in output
    assert "Declared budget: 1000 environment transitions" in output
    assert f"[run_details.json](<{run}/run_details.json>)" in output
    assert "`source`" in output and "`content_binding`" in output
    assert "exact-panel" in output
    assert f"Final Actor: [{final}](<{final}>)" in output
    assert f"Selected Actor: [{selected}](<{selected}>)" in output
    assert f"system = training.load_system({selected!r})" in output
    assert "including retried work" in output and "unmeasured work" in output
    assert {path.name: path.read_bytes() for path in run.iterdir()} == before


def test_setup_fallback_and_unmeasured_phases_are_explicit(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_details.json").write_text(json.dumps({"initial_setup_seconds": 2.0}))
    output = refresh_summary([run], output_dir=tmp_path / "report").read_text()
    assert "| Setup | 2.0 | 1 |" in output
    assert "run_details.json (initial setup only)" in output
    assert "| Validation | Unavailable | 0 |" in output
    assert "Final Actor: Unavailable" in output
    assert "training.load_system(" not in output
    assert "Schema Versions: Unknown" in output
    assert "Initial Runtime: Unknown" in output
    assert "Latest Status Memory: Unknown" in output


def test_reports_keep_saved_runtime_and_terminal_memory_scope(tmp_path: Path) -> None:
    from marl_battlegrounds.training.analysis import (
        _summary,  # pyright: ignore[reportPrivateUsage]
    )

    run = tmp_path / "run"
    run.mkdir()
    schemas = {"actor_input": 1, "training_state": 1, "model": "recorded-model"}
    initial_runtime = {
        "provenance": {
            "backend": "cpu",
            "device": "Test CPU",
            "precision": "float32",
            "batch_shape": [4],
        },
        "declared_environment": {"JAX_PLATFORMS": "cpu"},
    }
    resumed_runtime = {
        "provenance": {
            "backend": "gpu",
            "device": "Test GPU",
            "precision": "float32",
            "batch_shape": [4],
        },
        "declared_environment": {"CUDA_VISIBLE_DEVICES": "declared-only"},
    }
    earlier_memory = {
        "process_peak_ram_bytes": 1000,
        "jax_allocator": None,
        "errors": ["Example unavailable allocator"],
        "scope": "Earlier process lifetime only.",
    }
    latest_memory: dict[str, Any] = {
        "process_peak_ram_bytes": 2000,
        "jax_allocator": {"peak_bytes_in_use": 3000},
        "errors": [],
        "scope": "JAX allocator only; driver and workers excluded.",
    }
    (run / "run_details.json").write_text(
        json.dumps({"schemas": schemas, "runtime": initial_runtime})
    )
    (run / "status.json").write_text(
        json.dumps({"status": "complete", "memory": latest_memory})
    )
    events = [
        {"event": "attempt_started", "attempt_id": "first", "runtime": initial_runtime},
        {"event": "attempt_failed", "attempt_id": "first", "memory": earlier_memory},
        {
            "event": "attempt_started",
            "attempt_id": "second",
            "runtime": resumed_runtime,
        },
        {"event": "attempt_complete", "attempt_id": "second", "memory": latest_memory},
    ]
    (run / "run_events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    summary = _summary(run)
    assert summary["schemas"] == schemas
    assert summary["initial_runtime"] == initial_runtime
    assert [row["runtime"] for row in summary["attempt_runtimes"]] == [
        initial_runtime,
        resumed_runtime,
    ]
    assert summary["memory"] == latest_memory
    assert [row["memory"] for row in summary["terminal_memory_history"]] == [
        earlier_memory,
        latest_memory,
    ]
    output = refresh_summary([run], output_dir=tmp_path / "report").read_text()
    assert "Initial Runtime: Backend: cpu; Device: Test CPU" in output
    assert "`second`: Backend: gpu; Device: Test GPU" in output
    assert "Process Lifetime Peak RAM: 2000 bytes" in output
    assert '"peak_bytes_in_use": 3000' in output
    assert "JAX allocator only; driver and workers excluded." in output
    assert "Example unavailable allocator" in output
    assert "their peaks must not be added together" in output
    assert {path.name: path.read_bytes() for path in run.iterdir()} == before


def test_open_recovery_withholds_stale_complete_results_and_artifacts(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_details.json").write_text(json.dumps({"run_id": "recovering-run"}))
    (run / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "final_actor": "stale-final",
                "selected_actor": "stale-selected",
            }
        )
    )
    (run / "selection.json").write_text(json.dumps({"checkpoint_id": "stale-choice"}))
    (run / "exposure.json").write_text(json.dumps({"completed": 10}))
    (run / "slot_diagnostic.json").write_text(
        json.dumps({"systematic_effect_detected": False})
    )
    (run / "checkpoint_recovery.json").write_text(
        json.dumps({"checkpoint_id": "retry-this-checkpoint"})
    )
    # Recovery can fail before the trainer log has been restored to its prefix.
    (run / "training_updates.jsonl").write_text('{"env_steps":')
    (run / "validation_results.json").write_text(
        json.dumps([_result("stale-choice", 1000, 1.0)])
    )
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    report = analyze([run], output_dir=tmp_path / "report")
    facts = report["runs"][0]
    assert not report["complete"] and facts["status"] == "recovering"
    assert facts["saved_status"] == "complete"
    assert facts["recovery"]["checkpoint_id"] == "retry-this-checkpoint"
    assert facts["env_steps"] is None
    assert facts["validation_checkpoints"] == 0
    for name in (
        "selection",
        "final_actor",
        "selected_actor",
        "exposure",
        "slot_diagnostic",
    ):
        assert facts[name] is None
    output = Path(report["artifacts"]["summary"]).read_text()
    assert "Checkpoint recovery is unfinished" in output
    assert "stale-choice" not in output and "stale-selected" not in output
    assert "Final slot diagnostic:" not in output
    assert "stale-choice" not in Path(report["artifacts"]["curve"]).read_text()
    assert "No Validation Results" in Path(report["artifacts"]["svg"]).read_text()
    assert {path.name: path.read_bytes() for path in run.iterdir()} == before
    assert not report["complete"]


def test_selection_keeps_compatible_qmix_rows_and_drops_warmup_actors() -> None:
    rows = [
        _result("warm", 96, 0.9, method="qmix", optimizer_steps=0),
        _result("q", 128, 0.6, method="qmix", optimizer_steps=1),
        _result("p", 128, 0.7),
        _result("final", 192, 0.5, method="qmix", optimizer_steps=3),
    ]
    assert confirmation_candidates(rows, final_checkpoint_id="final") == (
        "p",
        "q",
        "final",
    )
    confirmed = [{**row, "purpose": "confirmation"} for row in rows[1:]]
    assert select_checkpoint(confirmed)["checkpoint_id"] == "p"
    for bad in (None, -1, True, 1.0):
        malformed = _result("q", 128, 0.6, method="qmix", optimizer_steps=bad)
        with pytest.raises(ValueError, match="optimizer_steps"):
            confirmation_candidates([malformed, rows[3]], final_checkpoint_id="final")


def test_selection_keeps_compatible_pqn_rows_and_drops_initial_actors() -> None:
    rows = [
        _result("initial", 32, 0.9, method="pqn_vdn", optimizer_steps=0),
        _result("pqn", 48, 0.6, method="pqn_vdn", optimizer_steps=2),
        _result("q", 64, 0.7, method="qmix", optimizer_steps=1),
        _result("final", 80, 0.5, method="pqn_vdn", optimizer_steps=6),
    ]
    assert confirmation_candidates(rows, final_checkpoint_id="final") == (
        "q",
        "pqn",
        "final",
    )
    for bad in (None, -1, True, 1.0):
        malformed = _result("pqn", 48, 0.6, method="pqn_vdn", optimizer_steps=bad)
        with pytest.raises(ValueError, match="optimizer_steps"):
            confirmation_candidates([malformed, rows[3]], final_checkpoint_id="final")


def test_a_pqn_report_uses_its_settings_summary_line_and_figure(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.training.analysis import (
        _settings,  # pyright: ignore[reportPrivateUsage]
    )

    config = {
        "method": "pqn_vdn",
        "seed": 7,
        "num_envs": 4,
        "pqn": {"rollout_length": 4, "input_scale": 1.0, "spawn_frame": "left"},
        "ppo": {"input_scale": 9.0},
    }
    assert _settings(config)["rollout_length"] == 4
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_details.json").write_text(
        json.dumps({"run_id": "Example", "config": config})
    )
    exposure = {"by_stage": [4] + [0] * 16, "by_source": [4], "by_opponent": [4]}
    updates = [
        {
            "env_steps": 16,
            "completed_updates": 0,
            "learning_blocks": 0,
            "used_td_pairs": 0,
            "used_prefix_td_pairs": 0,
            "used_exposure": {**exposure, "by_stage": [0] * 17},
            "loss": None,
            "epsilon": 1.0,
        },
        {
            "env_steps": 40,
            "completed_updates": 2,
            "learning_blocks": 1,
            "used_td_pairs": 20,
            "used_prefix_td_pairs": 8,
            "used_exposure": exposure,
            "loss": 1.5,
            "epsilon": 0.01,
        },
    ]
    (root / "training_updates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in updates)
    )
    (root / "status.json").write_text(json.dumps({"status": "complete"}))
    report = analyze([root], output_dir=tmp_path / "report")
    assert report["runs"][0]["input_scale"] == 1.0
    assert report["runs"][0]["spawn_frame"] == "left"
    assert Path(report["artifacts"]["pqn_png"]).read_bytes().startswith(b"\x89PNG")
    assert "Share Of Used Pairs" in Path(report["artifacts"]["pqn_svg"]).read_text()
    header = Path(report["artifacts"]["curve"]).read_text().splitlines()[0]
    assert "used_prefix_td_pairs" in header and "used_exposure" not in header
    summary = Path(report["artifacts"]["summary"]).read_text()
    assert "Optimizer steps: 2. Learning blocks: 1." in summary
    assert "of which pairs starting on kept rows: 8." in summary
    assert "Initial random transitions generated but never learned: 16." in summary
    # A run saved before round T has generated fewer initial transitions.
    early = {**updates[0], "env_steps": 8}
    (root / "training_updates.jsonl").write_text(json.dumps(early) + "\n")
    (root / "status.json").write_text(json.dumps({"status": "incomplete"}))
    report = analyze([root], output_dir=tmp_path / "early-report")
    summary = Path(report["artifacts"]["summary"]).read_text()
    assert "Initial random transitions generated but never learned: 8." in summary


def test_reports_keep_training_and_validation_red_zone_depths_separate(
    tmp_path: Path,
) -> None:
    import csv

    runs: list[Path] = []
    for index, depth in enumerate((None, 0.0, 5.0)):
        root = tmp_path / f"run-{index}"
        root.mkdir()
        config: dict[str, Any] = {"method": "mappo", "seed": index}
        if depth is not None:
            config["red_zone_depth"] = depth
        (root / "run_details.json").write_text(
            json.dumps({"run_id": root.name, "config": config})
        )
        (root / "training_updates.jsonl").write_text(
            json.dumps({"env_steps": 4, "elapsed_seconds": 1.0}) + "\n"
        )
        records = [
            _result(
                f"actor-{index}-{step}",
                step,
                0.5,
                **({"red_zone_depth": value} if value is not None else {}),
                cells=[{"map_id": 42, "score": 0.5}],
            )
            for step, value in ((4, depth), (8, 6.0))
        ]
        (root / "validation_results.json").write_text(json.dumps(records))
        (root / "selection.json").write_text(
            json.dumps(
                {
                    "checkpoint_id": records[-1]["checkpoint_id"],
                    "score": 0.5,
                    **({"red_zone_depth": 6.0} if index else {}),
                }
            )
        )
        runs.append(root)
    before = {path: path.read_bytes() for root in runs for path in root.iterdir()}
    report = analyze(runs, output_dir=tmp_path / "report")
    assert [row["red_zone_depth"] for row in report["runs"]] == [None, 0.0, 5.0]
    summary = Path(report["artifacts"]["summary"]).read_text()
    svg = Path(report["artifacts"]["svg"]).read_text()
    for label in ("Unknown (not recorded)", "0.0 map units", "5.0 map units"):
        assert label in summary and label in svg
    assert "Validation Red Zone: 6.0 map units" in svg
    assert "Training Red Zone Depth:" in summary
    assert "Recorded Validation Red Zone Depths:" in summary
    assert "Confirmation Red Zone Depth: 6.0 map units" in summary
    assert "Confirmation Red Zone Depth: Unknown (not recorded)" in summary
    for artifact in ("curve", "cells"):
        with Path(report["artifacts"][artifact]).open() as stream:
            rows = list(csv.DictReader(stream))
        assert {row["red_zone_depth"] for row in rows} == {"", "0.0", "5.0", "6.0"}
        assert {row["training_red_zone_depth"] for row in rows} == {"", "0.0", "5.0"}
    assert {path: path.read_bytes() for path in before} == before


def test_point_margin_alone_selects_and_keeps_stable_step_and_identity_ties() -> None:
    point_winner = _result(
        "points",
        20,
        0.1,
        selection_schema_version=2,
        mean_kill_difference=-20,
        cells=[
            {"mean_team_a_score": 7, "mean_team_b_score": 3},
            {"mean_team_a_score": 11, "mean_team_b_score": 3},
        ],
    )
    more_wins = _result(
        "wins",
        4,
        1.0,
        selection_schema_version=2,
        mean_kill_difference=100,
        cells=[{"mean_team_a_score": 8, "mean_team_b_score": 3}],
    )
    final = _result(
        "final",
        24,
        0.9,
        selection_schema_version=2,
        mean_kill_difference=50,
        cells=[{"mean_team_a_score": 1, "mean_team_b_score": 3}],
    )
    rows = [final, more_wins, point_winner]
    assert confirmation_candidates(rows, final_checkpoint_id="final") == (
        "points",
        "wins",
        "final",
    )
    confirmed = [{**row, "purpose": "confirmation"} for row in rows]
    assert select_checkpoint(confirmed)["checkpoint_id"] == "points"
    assert select_checkpoint(confirmed, rule="saved")["checkpoint_id"] == "wins"
    early = {**confirmed[-1], "checkpoint_id": "early", "env_steps": 8}
    other_id = {
        **early,
        "checkpoint_id": "aaa",
        "score": 0.0,
        "mean_kill_difference": -1000,
    }
    for ordered in ([early, other_id, confirmed[-1]], [confirmed[-1], other_id, early]):
        assert select_checkpoint(ordered)["checkpoint_id"] == "aaa"


@pytest.mark.parametrize("points", [None, True, "3", float("nan"), float("inf")])
def test_point_selection_requires_finite_saved_points(points: object) -> None:
    row = _result(
        "actor",
        4,
        1.0,
        purpose="confirmation",
        cells=[{"mean_team_a_score": points, "mean_team_b_score": 0}],
    )
    with pytest.raises(ValueError, match="finite saved points"):
        select_checkpoint([row])
    assert select_checkpoint([row], rule="saved")["checkpoint_id"] == "actor"


@pytest.mark.parametrize("cells", [None, [], [{}], [None]])
def test_point_selection_does_not_substitute_wins_or_kills(cells: object) -> None:
    row = _result(
        "actor", 4, 1.0, purpose="confirmation", cells=cells, mean_kill_difference=100
    )
    with pytest.raises(ValueError, match="saved points"):
        select_checkpoint([row])


def test_point_selection_can_use_historical_point_cells_without_kill_fields() -> None:
    row = _result("actor", 4, 1.0, purpose="confirmation", selection_schema_version=2)
    assert select_checkpoint([row]) == row
    with pytest.raises(ValueError, match="kill difference"):
        select_checkpoint([row], rule="saved")
    with pytest.raises(ValueError, match="Selection rule"):
        select_checkpoint([row], rule="other")


@pytest.mark.parametrize(
    "changed",
    (
        {"maps": [43]},
        {"system_roster": ["mage", "mage"]},
        {"opponent_roster": ["warrior"]},
    ),
)
def test_selection_refuses_changed_validation_maps_and_rosters(
    changed: dict[str, Any],
) -> None:
    first = _result("first", 4, 0.5, maps=[42])
    second = _result("second", 8, 0.5, maps=[42])
    second.update(changed)
    with pytest.raises(ValueError, match="maps or team rosters"):
        confirmation_candidates([first, second], final_checkpoint_id="second")
    first["purpose"] = second["purpose"] = "confirmation"
    with pytest.raises(ValueError, match="maps or team rosters"):
        select_checkpoint([first, second])


def test_selection_accepts_explicit_canonical_rosters_with_historical_defaults() -> (
    None
):
    from marl_battlegrounds.tasks import canonical_tournament_rosters

    first = _result("first", 4, 0.5, maps=[42])
    second = _result("second", 8, 0.5, maps=[42])
    a, b = canonical_tournament_rosters()
    second.update(system_roster=list(a), opponent_roster=list(b))
    first["purpose"] = second["purpose"] = "confirmation"
    assert select_checkpoint([first, second])["checkpoint_id"] == "first"
