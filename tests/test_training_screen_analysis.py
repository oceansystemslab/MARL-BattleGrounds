"""Check saved screen reports without running policies, games or optimizers.

Known tables prove native scores, kill margins and whole-block paired changes.
Synthetic run records exercise shared initialization identity, missing evidence,
unfinished cases, timing scopes and immutable inputs. One report renders real
headless plots; other cases replace plotting only. M8 loading has its own tests.
"""

# pyright: reportPrivateUsage=false
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.training import analysis

Record = dict[str, Any]
MAPS = tuple(range(42, 47))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _rows(*, improved: bool = False, tied: bool = False) -> list[Record]:
    return [
        {
            "map_id": map_id,
            "seed_id": 100 * map_id + seed,
            "opponent": "Random",
            "spawn_locations": end,
            "system_game_score": 0.5 if tied else 0.5 * (seed % 2) + 0.5 * improved,
            "team_a_score": 4 * seed + end + 2 * improved,
            "team_b_score": 2,
            "episode_length": 300,
        }
        for map_id in MAPS
        for seed in range(4)
        for end in (0, 1)
    ]


def _result(task: str, steps: int, rows: list[Record]) -> Record:
    return {
        "schema_version": 1,
        "task_id": task,
        "checkpoint_id": f"checkpoint-{task}",
        "actor_digest": f"actor-{task}",
        "actor_path": f"/original/actors/{task}",
        "env_steps": steps,
        "maps": list(MAPS),
        "root": 19_043_001,
        "seed_pairs": 4,
        "purpose": "random",
        "panel_digest": "random-diagnostic-v1",
        "members": [{"name": "Random", "actor_digest": "builtin-random"}],
        "pass_paths": [f"/original/games/{task}"],
        "summary_path": f"/original/{task}/validation_summary.json",
        "reference_path": None,
        "reused_initialization": False,
        "wall_seconds": 5 + steps / 100,
        "elapsed_seconds": 4 + steps / 100,
        "training_seconds": 0 if not steps else 10 + 2 * (steps // 512 - 1),
        **analysis.summarize_validation(
            rows, maps=MAPS, opponents=("Random",), seed_pairs=4
        ),
    }


def _package(tmp_path: Path) -> tuple[Path, dict[str, list[Record]]]:
    package = tmp_path / "package"
    cases: list[Record] = [
        {"name": "b32-t16", "num_envs": 32, "rollout_length": 16},
        {"name": "b32-t32", "num_envs": 32, "rollout_length": 32},
    ]
    _write(package / "declaration.json", {"cases": cases, "seed": 7})
    _write(
        package / "study.json",
        {
            "status": "running",
            "phase": "reports",
            "cases": {case["name"]: {"status": "complete"} for case in cases},
        },
    )
    _write(
        package / "budgets.json",
        {
            "cases": [
                {
                    **case,
                    "updates": 4,
                    "total_env_steps": 2048,
                    "median_update_seconds": 2,
                    "checkpoint_env_steps": [1024],
                }
                for case in cases
            ]
        },
    )
    initial = _result("initial", 0, _rows())
    _write(package / "shared" / "initialization.json", initial)
    sources = {"initial": _rows()}
    for case_index, case in enumerate(cases):
        run = package / "jobs" / case["name"] / "run"
        _write(
            run / "run_details.json",
            {
                "run_id": case["name"],
                "config": {
                    "seed": 7,
                    "num_envs": case["num_envs"],
                    "total_env_steps": 2048,
                    "ppo": {"rollout_length": case["rollout_length"]},
                    "checkpoint_env_steps": [1024],
                    "random_diagnostic_seed_pairs": 4,
                    "random_initialization_result": str(
                        package / "shared" / "initialization.json"
                    )
                    if case_index
                    else None,
                },
            },
        )
        updates = [
            {
                "attempt_id": "first",
                "env_steps": step,
                "collection_update_seconds": 10 if index == 0 else 2,
                "training_seconds": 10 + index * 2,
                "wall_seconds": 12 + index * 3,
            }
            for index, step in enumerate((512, 1024, 1536, 2048))
        ]
        (run / "training_updates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in updates)
        )
        _write(
            run / "status.json",
            {"status": "complete", "wall_seconds": 30, "elapsed_seconds": 28},
        )
        (run / "run_events.jsonl").write_text(
            json.dumps({"event": "random_validation_complete", "seconds": 3}) + "\n"
        )
        records = [{**initial}]
        if case_index:
            records[0].update(
                reference_path=str(package / "shared" / "initialization.json"),
                reused_initialization=True,
                wall_seconds=7,
                elapsed_seconds=6,
            )
        for steps in (1024, 2048):
            task = f"{case['name']}-{steps}"
            sources[task] = _rows(improved=True)
            records.append(_result(task, steps, sources[task]))
        _write(run / "random_diagnostics.json", records)
    budget_path = package / "budgets.json"
    budgets = json.loads(budget_path.read_text())
    for row in budgets["cases"]:
        details = json.loads(
            (package / "jobs" / row["name"] / "run" / "run_details.json").read_text()
        )
        row["config"] = details["config"]
    _write(budget_path, budgets)
    return package, sources


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _skip_plots(points: Sequence[Record], destination: Path) -> dict[str, str]:
    return {}


def _ancestry(directory: Path, details: Record) -> dict[str, Record]:
    name = directory.parent.name
    return {
        f"checkpoint-{task}": {
            "kind": "learner",
            "metadata": {"config": details["config"]},
            "counters": {"env_steps": steps},
            "actor_digest": f"actor-{task if steps else 'initial'}",
        }
        for task, steps in (
            ("initial" if name == "b32-t16" else "local-initial", 0),
            (f"{name}-1024", 1024),
            (f"{name}-2048", 2048),
        )
    }


def _saved_rows(
    monkeypatch: pytest.MonkeyPatch, sources: dict[str, list[Record]]
) -> None:
    def read(
        result: Record, *, run_id: str | None, seed: int | None
    ) -> tuple[Record, list[Record]]:
        rows = sources[result["task_id"]]
        return analysis.summarize_validation(
            rows,
            maps=result["maps"],
            opponents=("Random",),
            seed_pairs=result["seed_pairs"],
        ), rows

    monkeypatch.setattr(analysis, "_screen_evidence", read)
    monkeypatch.setattr(analysis, "_screen_ancestry", _ancestry)


def test_paired_changes_keep_spawn_pairs_and_correlated_initial_noise() -> None:
    original = _rows()
    improved = _rows(improved=True)
    first = analysis._screen_statistics(original, _result("initial", 0, original))
    last = analysis._screen_statistics(improved, _result("last", 64, improved))
    result = analysis._screen_change(last, first)
    assert first["independent_blocks"] == 20
    assert first["games"] == 40
    assert (first["score"], last["score"]) == (0.25, 0.75)
    assert result["score_change_from_initial"] == 0.5
    assert result["score_change_ci_low"] == result["score_change_ci_high"] == 0.5
    assert result["kill_margin_change_from_initial"] == 2
    assert (
        result["kill_margin_change_ci_low"] == result["kill_margin_change_ci_high"] == 2
    )
    shuffled = analysis._screen_statistics(
        list(reversed(improved)), _result("last", 64, improved)
    )
    assert analysis._screen_change(shuffled, first) == result


def test_ties_remain_ties_and_missing_kill_scores_stay_unknown() -> None:
    rows = _rows(tied=True)
    for row in rows:
        row.pop("team_a_score")
    stats = analysis._screen_statistics(rows, _result("tied", 0, rows))
    assert stats["score"] == stats["ci_low"] == stats["ci_high"] == 0.5
    assert stats["kill_margin"] is None
    assert stats["kill_margin_ci_low"] is None
    assert stats["mean_kills_for"] is None
    assert stats["mean_kills_against"] == 2
    assert sum(cell["draws"] for cell in stats["cells"]) == 40


def test_all_draws_can_contain_paired_combat_improvement() -> None:
    original = _rows(tied=True)
    improved = _rows(tied=True, improved=True)
    first = analysis._screen_statistics(original, _result("initial", 0, original))
    last = analysis._screen_statistics(improved, _result("last", 64, improved))
    change = analysis._screen_change(last, first)
    assert (first["score"], last["score"]) == (0.5, 0.5)
    assert (last["wins"], last["draws"], last["losses"]) == (0, 40, 0)
    assert last["mean_kills_for"] - first["mean_kills_for"] == 2
    assert last["mean_kills_against"] == first["mean_kills_against"] == 2
    assert change["score_change_from_initial"] == 0
    assert change["kill_margin_change_from_initial"] == 2
    assert change["kill_margin_change_ci_low"] == 2


def test_one_missing_team_score_does_not_hide_invalid_other_score() -> None:
    rows = _rows(tied=True)
    rows[0].pop("team_a_score")
    rows[0]["team_b_score"] = -1
    with pytest.raises(ValueError, match="finite and nonnegative"):
        analysis._screen_statistics(rows, _result("invalid", 0, rows))


@pytest.mark.parametrize(
    ("seconds", "expected"),
    ((None, "Unknown"), (-1, "Unknown"), (976.5, "16m 16s"), (7200, "2h 00m 00s")),
)
def test_screen_duration_has_clear_units(seconds: object, expected: str) -> None:
    assert analysis._screen_duration(seconds) == expected


def test_complete_screen_keeps_shared_task_once_and_renders_actual_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, sources = _package(tmp_path)
    calls: list[str] = []

    def read(
        result: Record, *, run_id: str | None, seed: int | None
    ) -> tuple[Record, list[Record]]:
        calls.append(result["task_id"])
        rows = sources[result["task_id"]]
        return analysis.summarize_validation(
            rows,
            maps=result["maps"],
            opponents=("Random",),
            seed_pairs=result["seed_pairs"],
        ), rows

    monkeypatch.setattr(analysis, "_screen_evidence", read)
    monkeypatch.setattr(analysis, "_screen_ancestry", _ancestry)
    before = {
        str(path.relative_to(package)): path.read_bytes()
        for path in package.rglob("*")
        if path.is_file()
    }
    result = analysis.analyze_screen(package)
    assert result["complete"]
    assert result["evidence_errors"] == []
    assert result["unique_validation_tasks"] == 5
    assert calls.count("initial") == 1
    assert len(calls) == 5
    cells = _csv(Path(result["artifacts"]["cells"]))
    assert len(cells) == 25
    assert len([row for row in cells if row["task_id"] == "initial"]) == 5
    initial_cells = [row for row in cells if row["task_id"] == "initial"]
    assert all(
        json.loads(row["cases"]) == ["b32-t16", "b32-t32"] for row in initial_cells
    )
    curves = _csv(Path(result["artifacts"]["curve"]))
    initial_points = [
        row for row in curves if row["kind"] == "random" and row["env_steps"] == "0"
    ]
    assert [float(row["wall_seconds"]) for row in initial_points] == [5, 7]
    assert {row["checkpoint_id"] for row in initial_points} == {"checkpoint-initial"}
    assert all(row["initial_task_id"] == "initial" for row in initial_points)
    trials = _csv(Path(result["artifacts"]["trials"]))
    assert all(float(row["warm_training_seconds"]) == 6 for row in trials)
    assert all(float(row["final_score_change_from_initial"]) == 0.5 for row in trials)
    assert all(Path(path).is_file() for path in result["artifacts"].values())
    svg = Path(result["artifacts"]["svg"]).read_text()
    assert "Elapsed Run Time (Minutes, Including Overhead)" in svg
    assert "Environment Transitions" in svg
    assert "Change In Kills Minus Deaths Per Game" in svg
    assert "Win = 1, Draw = 0.5, Loss = 0" in svg
    summary = Path(result["artifacts"]["summary"]).read_text()
    assert "Unique Evaluated Games: 200" in summary
    assert "not pure compilation time" in summary
    assert "No winner" in summary
    assert "training variation" in summary
    assert "Random Validation: 3.0" in summary
    assert "first case pays" in summary
    assert "Wins are not required" in summary
    assert "They do not show that nothing was learned" in summary
    assert "Kills / Deaths Per Game" in summary
    assert "Paired Game Interval: +2.00 to +2.00" in summary
    assert "30s" in summary
    assert all((package / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize(
    "damage",
    (
        "missing_cell",
        "reference_id",
        "different_seeds",
        "duplicate_initial",
        "wrong_shape",
        "wrong_seed",
        "wrong_scale",
        "wrong_reward",
        "other_run",
        "wrong_actor",
        "undeclared_reference",
    ),
)
def test_invalid_evidence_is_visible_without_certifying_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    package, sources = _package(tmp_path)
    _saved_rows(monkeypatch, sources)
    monkeypatch.setattr(analysis, "_screen_plot", _skip_plots)
    path = package / "jobs" / "b32-t32" / "run" / "random_diagnostics.json"
    records = json.loads(path.read_text())
    if damage == "missing_cell":
        sources[records[-1]["task_id"]].pop()
    elif damage == "reference_id":
        records[0]["checkpoint_id"] = "not-the-original"
    elif damage == "different_seeds":
        for row in sources[records[-1]["task_id"]]:
            row["seed_id"] += 10000
    elif damage == "duplicate_initial":
        records[0].update(task_id="another-initial", reused_initialization=False)
    elif damage == "other_run":
        records[-1] = json.loads(
            (
                package / "jobs" / "b32-t16" / "run" / "random_diagnostics.json"
            ).read_text()
        )[-1]
    elif damage == "wrong_actor":
        records[-1]["actor_digest"] = "actor-from-another-boundary"
    elif damage == "undeclared_reference":
        alternate = package / "another-initialization.json"
        _write(
            alternate,
            json.loads((package / "shared" / "initialization.json").read_text()),
        )
        records[0]["reference_path"] = str(alternate)
    else:
        detail_path = path.parent / "run_details.json"
        details = json.loads(detail_path.read_text())
        if damage == "wrong_shape":
            details["config"]["num_envs"] = 1024
        elif damage == "wrong_seed":
            details["config"]["seed"] += 1
        elif damage == "wrong_scale":
            details["config"]["ppo"]["input_scale"] = 0.01
        else:
            details["config"]["shaping_mode"] = "score_delta"
        _write(detail_path, details)
    _write(path, records)
    result = analysis.analyze_screen(package)
    assert not result["complete"]
    assert result["evidence_errors"][0]["case"] == "b32-t32"
    assert result["cases"][0]["complete"]
    assert not result["cases"][1]["complete"]
    assert "Evidence Errors" in Path(result["artifacts"]["summary"]).read_text()


def test_no_budgets_failed_and_unstarted_cases_stay_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    _write(
        package / "declaration.json",
        {
            "cases": [
                {"name": "first", "num_envs": 32, "rollout_length": 16},
                {"name": "second", "num_envs": 1024, "rollout_length": 128},
            ]
        },
    )
    _write(
        package / "study.json",
        {
            "status": "failed",
            "cases": {
                "first": {"status": "failed", "error": "Calibration deadline"},
                "second": {"status": "unstarted"},
            },
        },
    )
    monkeypatch.setattr(analysis, "_screen_plot", _skip_plots)
    result = analysis.analyze_screen(package)
    assert not result["complete"]
    assert result["unique_validation_tasks"] == 0
    trials = _csv(Path(result["artifacts"]["trials"]))
    assert [row["status"] for row in trials] == ["failed", "unstarted"]
    assert all("final_score" not in row for row in trials)
    assert _csv(Path(result["artifacts"]["cells"])) == []
    assert "Calibration deadline" in Path(result["artifacts"]["summary"]).read_text()


def test_warm_seconds_excludes_each_attempt_first_block_without_relabelling_it() -> (
    None
):
    rows = [
        {"attempt_id": attempt, "env_steps": step, "collection_update_seconds": cost}
        for attempt, step, cost in (
            ("one", 32, 10),
            ("one", 64, 2),
            ("two", 96, 8),
            ("two", 128, 3),
        )
    ]
    assert analysis._screen_warm_seconds(rows, 0) == 0
    assert analysis._screen_warm_seconds(rows, 64) == 2
    assert analysis._screen_warm_seconds(rows, 128) == 5
    rows[-1].pop("collection_update_seconds")
    assert analysis._screen_warm_seconds(rows, 128) is None


def test_pending_recovery_withholds_stale_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, sources = _package(tmp_path)
    run = package / "jobs" / "b32-t32" / "run"
    _write(run / "checkpoint_recovery.json", {"checkpoint_id": "earlier"})
    (run / "training_updates.jsonl").write_text("torn")
    _saved_rows(monkeypatch, sources)
    monkeypatch.setattr(analysis, "_screen_plot", _skip_plots)
    result = analysis.analyze_screen(package)
    assert not result["complete"]
    assert result["cases"][1]["status"] == "recovering"
    assert result["unique_validation_tasks"] == 3
    curves = _csv(Path(result["artifacts"]["curve"]))
    assert {row["case"] for row in curves} == {"b32-t16"}


def test_each_training_seed_can_have_its_own_shared_initial_actor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, sources = _package(tmp_path)
    _saved_rows(monkeypatch, sources)
    monkeypatch.setattr(analysis, "_screen_plot", _skip_plots)
    (package / "shared" / "initialization.json").unlink()
    name = "b32-t32"
    run = package / "jobs" / name / "run"
    details = json.loads((run / "run_details.json").read_text())
    details["config"].update(seed=8, random_initialization_result=None)
    _write(run / "run_details.json", details)
    budgets = json.loads((package / "budgets.json").read_text())
    budgets["cases"][1]["config"] = details["config"]
    _write(package / "budgets.json", budgets)
    records = json.loads((run / "random_diagnostics.json").read_text())
    records[0] = _result("second-seed-initial", 0, _rows())
    sources["second-seed-initial"] = _rows()
    _write(run / "random_diagnostics.json", records)

    def ancestry(directory: Path, saved: Record) -> dict[str, Record]:
        result = _ancestry(directory, saved)
        if saved["config"]["seed"] == 8:
            initial = result.pop("checkpoint-local-initial")
            initial["actor_digest"] = "actor-second-seed-initial"
            result["checkpoint-second-seed-initial"] = initial
        return result

    monkeypatch.setattr(analysis, "_screen_ancestry", ancestry)
    result = analysis.analyze_screen(package, render_plots=False)
    assert result["complete"], result["evidence_errors"]
    assert result["unique_validation_tasks"] == 6
    assert "png" not in result["artifacts"]
