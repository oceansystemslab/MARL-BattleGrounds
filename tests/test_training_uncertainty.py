"""Check declared training variation without training or evaluating a policy.

Whole run vectors preserve cross-cell dependence. Paired comparisons preserve
run matching, and missing runs/cells or related continuations never create extra
independent seeds. Public reports keep selected identities and original files.
New fixed-System sampling evidence cannot disguise an unsupported interval;
legacy records retain their original conditional numerical interpretation.
"""

# pyright: reportPrivateUsage=false

import json
import pickle
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytest

from marl_battlegrounds.evaluation.sampling_evidence import summarize_sampling_evidence
from marl_battlegrounds.training import analysis

Record = dict[str, Any]


def _run(
    path: Path,
    scores: tuple[float, ...],
    *,
    status: str = "complete",
    lineage: str | None = None,
    parent: bool = False,
) -> Path:
    path.mkdir()
    details: Record = {
        "run_id": path.name,
        "config": {
            "seed": int.from_bytes(path.name.encode(), "little"),
            "red_zone_depth": 5.0,
        },
    }
    if lineage is not None:
        details["training_lineage"] = {"root_run_id": lineage}
    if parent:
        details["continuation"] = {"parent_run_id": "parent"}
    selection = {
        "checkpoint_id": f"actor-{path.name}",
        "actor_digest": f"digest-{path.name}",
        "panel_digest": "frozen-panel",
        "purpose": "confirmation",
        "complete": True,
        "red_zone_depth": 5.0,
        "env_steps": 8,
        "score": float(np.mean(scores)),
        "ci_low": 0.0,
        "ci_high": 1.0,
        "cells": [
            {"map_id": map_id, "opponent": opponent, "score": score}
            for (map_id, opponent), score in zip(
                ((42, "Alpha"), (42, "Beta"), (43, "Alpha"), (43, "Beta")),
                scores,
                strict=True,
            )
        ],
    }
    for name, value in (
        ("run_details.json", details),
        ("status.json", {"status": status}),
        ("selection.json", selection),
        ("validation_results.json", [selection]),
    ):
        (path / name).write_text(json.dumps(value))
    return path


def _grouping(entries: list[tuple[Path, str, str]]) -> Record:
    return {
        "schema_version": 1,
        "maps": [42, 43],
        "opponents": ["Alpha", "Beta"],
        "panel_digest": "frozen-panel",
        "red_zone_depth": 5.0,
        "runs": [
            {"run_dir": str(path), "group": group, "unit": unit}
            for path, group, unit in entries
        ],
        "bootstrap_draws": 2000,
        "bootstrap_seed": 819,
    }


def _reduce(grouping: Record) -> Record:
    paths = [Path(row["run_dir"]) for row in grouping["runs"]]
    summaries = [analysis._summary(path) for path in paths if path.exists()]
    details = {
        str(path): json.loads((path / "run_details.json").read_text())
        for path in paths
        if path.exists()
    }
    return analysis._training_variation(
        analysis._training_grouping(grouping, paths), summaries, details
    )


def test_public_report_resamples_whole_run_vectors_and_preserves_inputs(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path / "first", (0.0, 1.0, 0.0, 1.0))
    second = _run(tmp_path / "second", (1.0, 0.0, 1.0, 0.0))
    grouping = _grouping([(first, "Method", "seed-a"), (second, "Method", "seed-b")])
    before = {
        path: path.read_bytes() for root in (first, second) for path in root.iterdir()
    }
    random_state = pickle.dumps(np.random.get_state())
    report = analysis.analyze(
        [first, second], output_dir=tmp_path / "report", grouping=grouping
    )
    variation = report["training_variation"]
    group = variation["groups"][0]
    # Each actual training run scores exactly one half across the full field.
    # Resampling cells independently would wrongly make that overall score vary.
    assert (group["score"], group["ci_low"], group["ci_high"]) == (0.5, 0.5, 0.5)
    assert [(cell["ci_low"], cell["ci_high"]) for cell in group["cells"]] == [
        (0.0, 1.0)
    ] * 4
    assert group["independent_units"] == 2
    assert group["confidence"] == 0.95
    assert group["sampling_unit"] == "Whole independent training run"
    assert [row["checkpoint_id"] for row in variation["runs"]] == [
        "actor-first",
        "actor-second",
    ]
    saved = json.loads(Path(report["artifacts"]["training_variation"]).read_text())
    assert saved == variation
    summary = Path(report["artifacts"]["summary"]).read_text()
    assert "Declared Training Variation" in summary
    assert "fixed-actor game-sampling intervals" in summary
    assert pickle.dumps(np.random.get_state()) == random_state
    assert all(path.read_bytes() == content for path, content in before.items())
    assert _reduce(grouping) == variation


def test_declared_pairs_preserve_matching_and_unpaired_groups_resample_separately(
    tmp_path: Path,
) -> None:
    entries: list[tuple[Path, str, str]] = []
    for index, score in enumerate((0.25, 0.5, 0.75)):
        entries.append(
            (_run(tmp_path / f"a{index}", (score,) * 4), "A", f"pair{index}")
        )
    for index, score in reversed(tuple(enumerate((0.125, 0.375, 0.625)))):
        entries.append(
            (_run(tmp_path / f"b{index}", (score,) * 4), "B", f"pair{index}")
        )
    grouping = _grouping(entries)
    grouping["comparisons"] = [{"left": "A", "right": "B", "paired": True}]
    paired = _reduce(grouping)["comparisons"][0]
    assert paired["score"] == paired["ci_low"] == paired["ci_high"] == 0.125
    assert paired["paired_units"] == 3
    grouping["comparisons"][0]["paired"] = False
    blocked = _reduce(grouping)["comparisons"][0]
    assert blocked["ci_low"] is None
    assert "Related training units" in " ".join(blocked["unavailable_reasons"])
    for row in grouping["runs"]:
        if row["group"] == "B":
            row["unit"] = f"independent-{row['unit']}"
    independent = _reduce(grouping)["comparisons"][0]
    assert independent["score"] == 0.125
    assert independent["ci_low"] < 0 < independent["ci_high"]
    assert independent["paired_units"] is None


@pytest.mark.parametrize(
    "failure",
    ("missing", "failed", "running", "cell", "selection", "depth", "panel", "purpose"),
)
def test_incomplete_declared_runs_never_become_a_smaller_comparison(
    tmp_path: Path, failure: str
) -> None:
    first = _run(tmp_path / "first", (0.5,) * 4)
    second = tmp_path / "second"
    if failure != "missing":
        _run(
            second,
            (1.0,) * 4,
            status=failure if failure in ("failed", "running") else "complete",
        )
        selection_path = second / "selection.json"
        selected = json.loads(selection_path.read_text())
        if failure == "cell":
            selected["cells"].pop()
            selection_path.write_text(json.dumps(selected))
        elif failure == "selection":
            selection_path.unlink()
            status = {"status": "complete", "final_actor": "unused-final"}
            (second / "status.json").write_text(json.dumps(status))
        elif failure == "purpose":
            selected["purpose"] = "routine"
            selection_path.write_text(json.dumps(selected))
        elif failure in ("depth", "panel"):
            selected["red_zone_depth" if failure == "depth" else "panel_digest"] = (
                0 if failure == "depth" else "other"
            )
            selection_path.write_text(json.dumps(selected))
    report = _reduce(_grouping([(first, "M", "a"), (second, "M", "b")]))
    assert len(report["runs"]) == 2
    assert len(report["runs"][1]["cells"]) == 4
    assert not report["runs"][1]["available"]
    group = report["groups"][0]
    assert group["declared_runs"] == 2
    assert group["available_runs"] == 1
    assert group["independent_units"] == (None if failure == "missing" else 2)
    assert group["score"] is group["ci_low"] is group["ci_high"] is None
    assert len(group["cells"]) == 4
    assert group["unavailable_reasons"]


def test_missing_run_is_visible_in_public_report_and_is_not_complete(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path / "first", (0.5,) * 4)
    missing = tmp_path / "missing"
    report = analysis.analyze(
        [first, missing],
        output_dir=tmp_path / "report",
        grouping=_grouping([(first, "Method", "a"), (missing, "Method", "b")]),
    )
    assert report["complete"] is False
    assert report["training_variation"]["runs"][1]["status"] == "missing"
    assert "Run status is missing" in Path(report["artifacts"]["summary"]).read_text()
    assert not missing.exists()


@pytest.mark.parametrize(
    "case", ("one", "siblings", "unknown-parent", "duplicate-unit", "duplicate-seed")
)
def test_too_few_or_related_runs_do_not_add_independent_training_seeds(
    tmp_path: Path, case: str
) -> None:
    first = _run(
        tmp_path / "first", (0.25,) * 4, lineage="root" if case == "siblings" else None
    )
    entries = [(first, "M", "a")]
    if case != "one":
        second = _run(
            tmp_path / "second",
            (0.75,) * 4,
            lineage="root" if case == "siblings" else None,
            parent=case == "unknown-parent",
        )
        entries.append((second, "M", "a" if case == "duplicate-unit" else "b"))
        if case == "duplicate-seed":
            details = json.loads((second / "run_details.json").read_text())
            details["config"]["seed"] = json.loads(
                (first / "run_details.json").read_text()
            )["config"]["seed"]
            (second / "run_details.json").write_text(json.dumps(details))
    group = _reduce(_grouping(entries))["groups"][0]
    assert group["ci_low"] is group["ci_high"] is None
    assert group["unavailable_reasons"]
    if case == "one":
        assert group["score"] == 0.25
        assert group["independent_units"] == 1
    else:
        assert group["score"] is None
        assert group["independent_units"] is None


def test_paired_related_runs_cannot_cross_the_declared_pair_boundaries(
    tmp_path: Path,
) -> None:
    entries: list[tuple[Path, str, str]] = []
    for group, roots in (("A", ("root0", "root1")), ("B", ("root1", "root0"))):
        for index, root in enumerate(roots):
            entries.append(
                (
                    _run(tmp_path / f"{group}{index}", (0.5,) * 4, lineage=root),
                    group,
                    f"pair{index}",
                )
            )
    grouping = _grouping(entries)
    grouping["comparisons"] = [{"left": "A", "right": "B", "paired": True}]
    estimate = _reduce(grouping)["comparisons"][0]
    assert estimate["ci_low"] is None
    assert "lineage crosses" in " ".join(estimate["unavailable_reasons"])


@pytest.mark.parametrize(
    ("right_seeds", "paired", "reason"),
    (
        ((10, 11), False, "Shared saved training seeds"),
        ((11, 12), False, "Shared saved training seeds"),
        ((11, 10), True, "Saved training seeds cross"),
        ((10, 11), True, None),
        ((12, 13), False, None),
        ((None, None), False, None),
    ),
)
def test_saved_seeds_preserve_dependence_across_declared_groups(
    tmp_path: Path,
    right_seeds: tuple[int | None, int | None],
    paired: bool,
    reason: str | None,
) -> None:
    entries: list[tuple[Path, str, str]] = []
    for group, seeds in (("A", (10, 11)), ("B", right_seeds)):
        for index, seed in enumerate(seeds):
            score = float(seed % 2) if seed is not None else float(index)
            path = _run(tmp_path / f"{group}{index}", (score,) * 4)
            details_path = path / "run_details.json"
            details = json.loads(details_path.read_text())
            details["config"]["seed"] = seed
            details_path.write_text(json.dumps(details))
            unit = f"pair{index}" if paired else f"{group}-unit{index}"
            entries.append((path, group, unit))
    grouping = _grouping(entries)
    grouping["comparisons"] = [{"left": "A", "right": "B", "paired": paired}]
    report = _reduce(grouping)
    assert len(report["runs"]) == 4
    assert all(row["available"] for row in report["runs"])
    assert all(row["independent_units"] == 2 for row in report["groups"])
    estimate = report["comparisons"][0]
    if reason is not None:
        assert estimate["score"] is estimate["ci_low"] is estimate["ci_high"] is None
        assert reason in " ".join(estimate["unavailable_reasons"])
        assert all(
            cell["ci_low"] is cell["ci_high"] is None for cell in estimate["cells"]
        )
    else:
        assert estimate["unavailable_reasons"] == []
        assert estimate["score"] == 0.0
        if paired:
            # Equal saved seeds belong to the same pair, whose score difference is zero.
            assert estimate["ci_low"] == estimate["ci_high"] == 0.0
        else:
            # Separate or unknown seed values do not invent a shared-seed match.
            assert estimate["ci_low"] < 0 < estimate["ci_high"]


def test_missing_declared_pair_is_visible_without_dropping_the_other_run(
    tmp_path: Path,
) -> None:
    entries: list[tuple[Path, str, str]] = []
    for group, count in (("A", 3), ("B", 2)):
        for index in range(count):
            entries.append(
                (_run(tmp_path / f"{group}{index}", (0.5,) * 4), group, f"pair{index}")
            )
    grouping = _grouping(entries)
    grouping["comparisons"] = [{"left": "A", "right": "B", "paired": True}]
    report = _reduce(grouping)
    assert len(report["runs"]) == 5
    comparison = report["comparisons"][0]
    assert comparison["left_declared_runs"] == 3
    assert comparison["right_declared_runs"] == 2
    assert comparison["paired_units"] is None
    assert comparison["score"] is comparison["ci_low"] is None
    assert "same declared training units" in " ".join(comparison["unavailable_reasons"])


def test_path_grouping_resolves_relative_paths_and_rejects_undeclared_inputs(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path / "first", (0.5,) * 4)
    second = _run(tmp_path / "second", (0.5,) * 4)
    declaration = _grouping([(Path("first"), "M", "a")])
    path = tmp_path / "grouping.json"
    path.write_text(json.dumps(declaration))
    resolved = analysis._training_grouping(path, [first])
    assert resolved["runs"][0]["run_dir"] == str(first)
    with pytest.raises(ValueError, match="Available analysis inputs"):
        analysis._training_grouping(path, [first, second])
    with pytest.raises(ValueError, match="Available analysis inputs"):
        analysis._training_grouping(
            _grouping([(first, "M", "a"), (second, "M", "b")]), [first]
        )
    with pytest.raises(ValueError, match="repeat a run"):
        analysis._training_grouping(path, [first, first])


@pytest.mark.parametrize(
    "change", ("unknown", "schema", "map", "opponent", "unit", "paired", "draws")
)
def test_invalid_grouping_declarations_fail_before_reporting(
    tmp_path: Path, change: str
) -> None:
    root = _run(tmp_path / "run", (0.5,) * 4)
    declaration = _grouping([(root, "M", "a")])
    if change == "unknown":
        declaration["drop_missing"] = True
    elif change == "schema":
        declaration["schema_version"] = True
    elif change == "map":
        declaration["maps"] = [42, 42]
    elif change == "opponent":
        declaration["opponents"] = ["Alpha", "Alpha"]
    elif change == "unit":
        declaration["runs"][0]["unit"] = ""
    elif change == "paired":
        declaration["comparisons"] = [{"left": "M", "right": "absent", "paired": "yes"}]
    else:
        declaration["bootstrap_draws"] = 0
    with pytest.raises(ValueError):
        analysis.analyze([root], output_dir=tmp_path / "report", grouping=declaration)
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("mechanism", ("deterministic", "unknown", "stochastic"))
def test_fixed_system_bounds_use_supported_sampling_not_repeated_scores(
    mechanism: Literal["deterministic", "unknown", "stochastic"],
) -> None:
    rows = [
        {
            "map_id": 42,
            "opponent": "Alpha",
            "seed_id": seed,
            "spawn_locations": end,
            "system_game_score": 0.5,
        }
        for seed in (1, 2)
        for end in (0, 1)
    ]
    evidence = summarize_sampling_evidence(
        range(4),
        scheduled_games=4,
        sampling_units={index: index // 2 for index in range(4)},
        determinism=mechanism,
        basis=None if mechanism == "unknown" else "Declared fixture mechanism",
    )
    legacy = analysis.summarize_validation(
        rows, maps=[42], opponents=["Alpha"], seed_pairs=2
    )
    result = analysis.summarize_validation(
        rows, maps=[42], opponents=["Alpha"], seed_pairs=2, sampling_evidence=evidence
    )
    assert legacy["ci_low"] == legacy["ci_high"] == 0.5
    assert "sampling_evidence" not in legacy
    assert "declared_blocks" not in legacy
    assert legacy["independent_blocks"] == 2
    assert result["sampling_evidence"] == evidence
    assert result["declared_blocks"] == result["independent_blocks"] == 2
    assert (
        result["supported_independent_sampling_units"]
        == {
            "deterministic": 0,
            "unknown": None,
            "stochastic": 2,
        }[mechanism]
    )
    # Screen reporting can receive an already qualified summary from validation.
    assert analysis._validation_sampling_interval(result, evidence, 2) == result
    if mechanism == "stochastic":
        assert result["ci_low"] == result["ci_high"] == 0.5
    else:
        assert result["ci_low"] is result["ci_high"] is None
        assert result["conditional_ci_low"] == result["conditional_ci_high"] == 0.5
        assert "unsupported" in result["conditional_interval_assumption"]


def test_stratified_game_intervals_need_replicates_inside_each_stratum() -> None:
    rows = [
        {
            "map_id": map_id,
            "opponent": "Alpha",
            "seed_id": map_id,
            "spawn_locations": end,
            "system_game_score": 0.5,
        }
        for map_id in (42, 43)
        for end in (0, 1)
    ]
    evidence = summarize_sampling_evidence(
        range(4),
        scheduled_games=4,
        sampling_units={index: index // 2 for index in range(4)},
        determinism="stochastic",
        basis="Two maps with one paired draw each",
    )
    assert evidence["interval_status"] == "Available"
    result = analysis.summarize_validation(
        rows,
        maps=[42, 43],
        opponents=["Alpha"],
        seed_pairs=1,
        sampling_evidence=evidence,
    )
    assert result["ci_low"] is result["ci_high"] is None
    assert "stratum" in result["interval_status"]
    with pytest.raises(ValueError, match="game counts differ"):
        analysis.summarize_validation(
            rows[:2],
            maps=[42],
            opponents=["Alpha"],
            seed_pairs=1,
            sampling_evidence=evidence,
        )


def test_screen_does_not_restore_unsupported_score_kill_or_change_intervals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_training_screen_analysis import _package, _saved_rows

    package, sources = _package(tmp_path)
    _saved_rows(monkeypatch, sources)
    for path in (
        package / "shared" / "initialization.json",
        *package.glob("jobs/*/run/random_diagnostics.json"),
    ):
        saved = json.loads(path.read_text())
        records = cast(list[Record], saved) if isinstance(saved, list) else [saved]
        for row in records:
            count = len(sources[row["task_id"]])
            row["sampling_evidence"] = summarize_sampling_evidence(
                range(count), scheduled_games=count
            )
        path.write_text(json.dumps(saved))
    report = analysis.analyze_screen(package, render_plots=False)
    assert report["complete"]
    for case in report["cases"]:
        assert case["final_score"] == 0.75
        assert case["final_score_change_from_initial"] == 0.5
        assert case["final_ci_low"] is case["final_ci_high"] is None
        assert case["final_kill_margin_ci_low"] is None
        assert case["final_kill_margin_change_ci_low"] is None
    text = Path(report["artifacts"]["summary"]).read_text()
    assert "Unavailable: Sampling mechanism unknown" in text
