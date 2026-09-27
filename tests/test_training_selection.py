"""Check immutable re-selection, new shortlists, exact identities and follow-ups.

Synthetic verified evidence isolates orchestration and rule choices. The separate
selection-evidence tests exercise real artifact and saved-game verification.
New choices use saved point margins alone; old choices keep their saved rules.
No learning, provider call or GPU work occurs. Missing confirmation tasks must
produce executable public calls, never a provisional winner presented as final.
"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.training import selection

# pyright: reportPrivateUsage=false
from marl_battlegrounds.training.validation import (
    FrozenPanel,
    PanelMember,
    panel_task_description,
)


def _evidence(root: Path, seed: int = 1) -> dict[str, Any]:
    panel = FrozenPanel(
        path=root / "panel.json",
        digest="panel",
        members=(
            PanelMember(
                name="Opponent", registration_id="opponent", actor_digest="other"
            ),
        ),
        qualified=True,
        schema_version=2,
        roots={"routine": 11, "initialization": 12, "confirmation": 13},
    )
    records: list[dict[str, Any]] = []
    actors = {}
    for key, steps, score, kills in (
        ("a", 4, 0.8, 1.0),
        ("b", 8, 0.8, 2.0),
        ("final", 12, 0.2, 0.0),
    ):
        actors[key] = {
            "actor_path": str(root / "actors" / key),
            "checkpoint_id": key,
            "artifact_id": f"export-{key}",
            "actor_digest": f"inference-{key}",
        }
        for purpose in ("routine", "confirmation"):
            task = panel_task_description(
                checkpoint_id=key,
                actor_digest=f"inference-{key}",
                env_steps=steps,
                panel=panel,
                purpose=purpose,
                seed_pairs=2,
                red_zone_depth=5.0,
            )
            records.append(
                {
                    **task,
                    "complete": True,
                    "score": score,
                    "mean_kill_difference": kills,
                    "cells": [
                        {
                            "map_id": map_id,
                            "opponent": "Opponent",
                            "mean_team_a_score": 10 + kills,
                            "mean_team_b_score": 10,
                        }
                        for map_id in task["maps"]
                    ],
                }
            )
    return {
        "run_dir": str(root),
        "run_id": f"run-{seed}",
        "seed": seed,
        "status": "complete",
        "panel": panel,
        "panel_path": panel.path,
        "final_checkpoint_id": "final",
        "records": records,
        "actors": actors,
        "config": {"confirmation_seed_pairs": 2},
        "source_files": [],
    }


def _reader(monkeypatch: pytest.MonkeyPatch, items: dict[Path, dict[str, Any]]) -> None:
    from marl_battlegrounds.training import _selection_evidence

    def read(root: Path, **_: object) -> dict[str, Any]:
        return copy.deepcopy(items[root])

    monkeypatch.setattr(_selection_evidence, "read_run_evidence", read)


def test_separate_decision_preserves_exact_three_identities_and_tie_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    original = root / "selection.json"
    original.write_text('{"checkpoint_id":"original"}\n')
    _reader(monkeypatch, {root: _evidence(root)})
    result = selection.reselect_checkpoint(
        [root], declaration={"name": "New Choice"}, output_dir=tmp_path / "choice"
    )
    winner = result["runs"][0]["winner"]
    assert result["status"] == "complete"
    assert winner["checkpoint_id"] == "b"
    assert winner["artifact_id"] == "export-b"
    assert winner["actor_digest"] == "inference-b"
    assert [row["checkpoint_id"] for row in result["runs"][0]["shortlist"]] == [
        "b",
        "a",
        "final",
    ]
    assert original.read_text() == '{"checkpoint_id":"original"}\n'
    assert (
        selection.read_selection_decision(tmp_path / "choice/selection_decision.json")
        == result
    )
    with pytest.raises(ValueError, match="new or empty"):
        selection.reselect_checkpoint(
            [root], declaration={"name": "Another"}, output_dir=tmp_path / "choice"
        )


def test_new_rule_builds_own_shortlist_and_missing_confirmation_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    item = _evidence(root)
    for row in item["records"]:
        if row["checkpoint_id"] == "b":
            for cell in row["cells"]:
                cell["mean_team_a_score"] = 11
    item["records"] = [
        row
        for row in item["records"]
        if not (row["checkpoint_id"] == "a" and row["purpose"] == "confirmation")
    ]
    _reader(monkeypatch, {root: item})
    output = tmp_path / "new-rule"
    result = selection.reselect_checkpoint(
        [root],
        declaration={
            "name": "Earlier Ties",
            "rule": "point_margin",
            "shortlist_size": 1,
        },
        output_dir=output,
    )
    run = result["runs"][0]
    assert result["status"] == "needs_confirmation" and run["winner"] is None
    assert [row["checkpoint_id"] for row in run["shortlist"]] == ["a", "final"]
    call = run["needs_confirmation"][0]
    assert call["arguments"]["checkpoint"] == str(root / "actors/a")
    assert call["arguments"]["purpose"] == "confirmation"
    assert call["arguments"]["seed_pairs"] == 2
    assert call["arguments"]["red_zone_depth"] == 5.0
    compile(call["python"], "confirmation-example", "exec")
    assert call["expected_task"]["root"] == 13
    assert not Path(call["arguments"]["output_dir"]).exists()


def test_completed_follow_up_keeps_pending_decision_and_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    item = _evidence(root)
    for row in item["records"]:
        if row["checkpoint_id"] == "b":
            for cell in row["cells"]:
                cell["mean_team_a_score"] = 11
    missing = item["records"].pop(1)
    _reader(monkeypatch, {root: item})
    declaration = {
        "name": "Earlier Ties",
        "rule": "point_margin",
        "shortlist_size": 1,
    }
    pending = selection.reselect_checkpoint(
        [root], declaration=declaration, output_dir=tmp_path / "pending"
    )
    path = tmp_path / "pending/selection_decision.json"
    before = path.read_bytes()
    item["records"].append(missing)
    _reader(monkeypatch, {root: item})
    completed = selection.reselect_checkpoint(
        [root],
        declaration={**declaration, "previous_decision": str(path)},
        output_dir=tmp_path / "done",
    )
    assert completed["status"] == "complete"
    assert completed["previous_decision_id"] == pending["decision_id"]
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="original declared"):
        selection.reselect_checkpoint(
            [root],
            declaration={
                **declaration,
                "previous_decision": str(path),
                "shortlist_size": 2,
            },
            output_dir=tmp_path / "bad",
        )


@pytest.mark.parametrize(("first_steps", "expected_seed"), [(6, 11), (8, 22)])
def test_earlier_checkpoint_step_breaks_point_ties_before_seed_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_steps: int,
    expected_seed: int,
) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    items = {
        root: _evidence(root, seed) for root, seed in zip(roots, (11, 22), strict=True)
    }
    for row in items[roots[0]]["records"]:
        if row["checkpoint_id"] == "b":
            row["env_steps"] = first_steps
    _reader(monkeypatch, items)
    result = selection.reselect_checkpoint(
        roots,
        declaration={
            "name": "Across Runs",
            "across_runs": True,
            "seed_order": [22, 11],
        },
        output_dir=tmp_path / "choice",
    )
    assert result["across_run_selection"]["seed"] == expected_seed
    assert len(result["runs"]) == 2
    with pytest.raises(ValueError, match="exactly once"):
        selection.reselect_checkpoint(
            roots,
            declaration={"name": "Bad Seeds", "across_runs": True, "seed_order": [11]},
            output_dir=tmp_path / "bad",
        )


def test_failed_run_stays_visible_and_prevents_across_run_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = [tmp_path / "complete", tmp_path / "failed"]
    items = {
        root: _evidence(root, seed) for root, seed in zip(roots, (11, 22), strict=True)
    }
    items[roots[1]]["final_checkpoint_id"] = None
    items[roots[1]]["status"] = "failed"
    _reader(monkeypatch, items)
    result = selection.reselect_checkpoint(
        roots,
        declaration={
            "name": "Keep Failures",
            "across_runs": True,
            "seed_order": [11, 22],
        },
        output_dir=tmp_path / "choice",
    )
    assert result["status"] == "incomplete"
    assert result["across_run_selection"] is None
    assert result["runs"][1]["status"] == "incomplete"
    assert "final actor" in result["runs"][1]["reason"]


def test_changed_roots_require_explicit_unpaired_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    item = _evidence(root)
    item["records"][0]["root"] = 14
    _reader(monkeypatch, {root: item})
    with pytest.raises(ValueError, match="allow_different_roots"):
        selection.reselect_checkpoint(
            [root], declaration={"name": "Mixed Roots"}, output_dir=tmp_path / "bad"
        )
    result = selection.reselect_checkpoint(
        [root],
        declaration={"name": "Mixed Roots", "allow_different_roots": True},
        output_dir=tmp_path / "choice",
    )
    assert (
        result["runs"][0]["routine_comparison"] == "Unpaired: Different Declared Roots"
    )
    item["records"][0]["root"] = 13
    _reader(monkeypatch, {root: item})
    with pytest.raises(ValueError, match="every routine"):
        selection.reselect_checkpoint(
            [root],
            declaration={"name": "Not Fresh", "allow_different_roots": True},
            output_dir=tmp_path / "reused",
        )


def test_file_relative_paths_and_changed_decision_hash(tmp_path: Path) -> None:
    directory = tmp_path / "declarations"
    directory.mkdir()
    path = directory / "choice.json"
    path.write_text(
        json.dumps(
            {
                "name": "Follow Up",
                "previous_decision": "../previous/selection_decision.json",
                "confirmation_results": ["../games/validation_summary.json"],
            }
        )
    )
    resolved = selection._declaration(path)
    assert resolved["previous_decision"] == str(
        tmp_path / "previous/selection_decision.json"
    )
    assert resolved["confirmation_results"] == [
        str(tmp_path / "games/validation_summary.json")
    ]
    path.write_text(json.dumps({"schema_version": 1, "decision_id": "changed"}))
    with pytest.raises(ValueError, match="content differs"):
        selection.read_selection_decision(path)


@pytest.mark.parametrize(
    "change",
    [
        {"shortlist_size": True},
        {"seed_order": [1, 1]},
        {"schema_version": True},
        {"confirmation_root": 2**32},
        {"across_runs": True},
        {"extra": 1},
        {"rule": "unscored"},
    ],
)
def test_invalid_declaration_fails_before_writing(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        selection._declaration({"name": "Invalid", **change})
    assert not list(tmp_path.iterdir())


def test_saved_interval_reader_requires_the_exact_availability_contract() -> None:
    from marl_battlegrounds.evaluation.sampling_evidence import (
        summarize_sampling_evidence,
    )
    from marl_battlegrounds.training._run_io import _check_validation_interval
    from marl_battlegrounds.training.analysis import _validation_sampling_interval

    old = {
        "score": 0.5,
        "ci_low": 0.25,
        "ci_high": 0.75,
        "games": 4,
        "seed_pairs": 2,
        "independent_blocks": 2,
    }
    _check_validation_interval(old)
    evidence = summarize_sampling_evidence(list(range(4)), scheduled_games=4)
    new = _validation_sampling_interval(old, evidence, 2)
    assert new["ci_low"] is None and new["ci_high"] is None
    _check_validation_interval(new)
    assert new["independent_blocks"] == 2
    assert new["supported_independent_sampling_units"] is None
    historical = dict(new)
    historical.pop("supported_independent_sampling_units")
    historical["independent_blocks"] = None
    _check_validation_interval(historical)
    for change in (
        {"sampling_evidence": None},
        {"interval_status": "Available"},
        {"ci_high": 0.75},
        {"score": float("nan")},
        {"conditional_ci_low": 0.9},
        {"confidence": 0.9},
        {"independent_blocks": 99},
        {"supported_independent_sampling_units": 2},
    ):
        with pytest.raises(ValueError):
            _check_validation_interval({**new, **change})


def test_report_verification_rebuilds_the_decision_instead_of_trusting_its_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    _reader(monkeypatch, {root: _evidence(root)})
    result = selection.reselect_checkpoint(
        [root], declaration={"name": "Verify Me"}, output_dir=tmp_path / "choice"
    )
    path = tmp_path / "choice/selection_decision.json"
    assert selection.read_selection_decision(path, verify_evidence=True) == result
    forged = copy.deepcopy(result)
    forged["runs"][0]["winner"]["score"] = 1.0
    forged.pop("decision_id")
    forged["decision_id"] = selection._digest(forged)
    path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="verified source evidence"):
        selection.read_selection_decision(path, verify_evidence=True)


def test_incomplete_decision_remains_readable_but_is_not_selection_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    item = _evidence(root)
    item["final_checkpoint_id"] = None
    _reader(monkeypatch, {root: item})
    result = selection.reselect_checkpoint(
        [root], declaration={"name": "Unfinished"}, output_dir=tmp_path / "choice"
    )
    path = tmp_path / "choice/selection_decision.json"
    assert selection.read_selection_decision(path) == result
    with pytest.raises(ValueError, match="ordinary progress"):
        selection.read_selection_decision(path, verify_evidence=True)


def test_failed_publication_leaves_no_partial_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.training import _run_io

    root = tmp_path / "run"
    _reader(monkeypatch, {root: _evidence(root)})

    def fail(path: Path, value: object) -> None:
        del path, value
        raise OSError("Injected storage failure")

    monkeypatch.setattr(_run_io, "atomic_json", fail)
    with pytest.raises(OSError, match="Injected storage"):
        selection.reselect_checkpoint(
            [root],
            declaration={"name": "No Partial File"},
            output_dir=tmp_path / "choice",
        )
    assert not (tmp_path / "choice/selection_decision.json").exists()
    assert not (tmp_path / "choice/.selection.lock").exists()


def test_missing_run_keeps_across_run_decision_incomplete(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    output = tmp_path / "choice"
    result = selection.reselect_checkpoint(
        [missing],
        declaration={"name": "Missing Seed", "across_runs": True, "seed_order": [123]},
        output_dir=output,
    )
    assert result["status"] == "incomplete"
    assert result["across_run_selection"] is None
    assert result["runs"][0]["seed"] is None
    assert result["runs"][0]["winner"] is None
    assert not missing.exists()
    saved = selection.read_selection_decision(output / "selection_decision.json")
    assert saved == result


@pytest.mark.parametrize("unknown_seed", [True, False])
def test_known_seed_order_stays_checked_with_unfinished_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unknown_seed: bool
) -> None:
    roots = [tmp_path / "complete", tmp_path / "unfinished"]
    items = {
        root: _evidence(root, seed) for root, seed in zip(roots, (11, 22), strict=True)
    }
    items[roots[1]]["final_checkpoint_id"] = None
    items[roots[1]]["status"] = "unfinished"
    if unknown_seed:
        items[roots[1]]["seed"] = None
    _reader(monkeypatch, items)
    result = selection.reselect_checkpoint(
        roots,
        declaration={"name": "Unfinished", "across_runs": True, "seed_order": [22, 11]},
        output_dir=tmp_path / "choice",
    )
    assert result["status"] == "incomplete"
    assert result["across_run_selection"] is None
    for order in ([22, 99], [11], [11, 22, 33]):
        with pytest.raises(ValueError, match="every distinct run seed"):
            selection.reselect_checkpoint(
                roots,
                declaration={
                    "name": "Bad Order",
                    "across_runs": True,
                    "seed_order": order,
                },
                output_dir=tmp_path / "bad",
            )
        assert not (tmp_path / "bad").exists()


def _child_evidence(root: Path, *, confirmation_root: int = 19) -> dict[str, Any]:
    item = _evidence(root)
    item["validation_declaration"] = {
        "roots": {
            "routine": 11,
            "initialization": 12,
            "confirmation": confirmation_root,
        },
        "confirmation_seed_pairs": 3,
    }
    item["used_roots"] = [11, 12, 15]
    item["actors"]["b"]["actor_path"] = str(root.parent / "parent/actors/b")
    item["records"] = [row for row in item["records"] if row["purpose"] == "routine"]
    return item


def test_child_reselection_uses_effective_defaults_and_original_parent_actor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "child"
    item = _child_evidence(root)
    _reader(monkeypatch, {root: item})
    pending = selection.reselect_checkpoint(
        [root], declaration={"name": "Child Defaults"}, output_dir=tmp_path / "pending"
    )
    assert pending["status"] == "needs_confirmation"
    calls = pending["runs"][0]["needs_confirmation"]
    assert {call["arguments"]["root_seed"] for call in calls} == {19}
    assert {call["arguments"]["seed_pairs"] for call in calls} == {3}
    assert calls[0]["arguments"]["checkpoint"] == str(tmp_path / "parent/actors/b")
    routines = {row["checkpoint_id"]: row for row in item["records"]}
    for call in calls:
        prior = routines[call["checkpoint_id"]]
        item["records"].append(
            {
                **call["expected_task"],
                "complete": True,
                "score": prior["score"],
                "mean_kill_difference": prior["mean_kill_difference"],
                "cells": prior["cells"],
            }
        )
    _reader(monkeypatch, {root: item})
    result = selection.reselect_checkpoint(
        [root],
        declaration={
            "name": "Child Defaults",
            "previous_decision": str(tmp_path / "pending/selection_decision.json"),
        },
        output_dir=tmp_path / "complete",
    )
    assert result["status"] == "complete"
    assert result["runs"][0]["winner"]["checkpoint_id"] == "b"
    assert result["runs"][0]["winner"]["actor_path"] == str(
        tmp_path / "parent/actors/b"
    )


def test_child_reselection_rejects_root_used_only_by_excluded_parent_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "child"
    item = _child_evidence(root, confirmation_root=15)
    _reader(monkeypatch, {root: item})
    with pytest.raises(ValueError, match="every routine"):
        selection.reselect_checkpoint(
            [root], declaration={"name": "Reused Root"}, output_dir=tmp_path / "bad"
        )
    assert not (tmp_path / "bad").exists()


def test_across_run_confirmation_is_fresh_against_every_compared_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    items = {roots[0]: _evidence(roots[0], 1), roots[1]: _evidence(roots[1], 2)}
    items[roots[1]]["used_roots"] = [13]
    items[roots[1]]["validation_declaration"] = {
        "roots": {"routine": 11, "initialization": 12, "confirmation": 14},
        "confirmation_seed_pairs": 2,
    }
    _reader(monkeypatch, items)
    separate = selection.reselect_checkpoint(
        roots, declaration={"name": "Separate Runs"}, output_dir=tmp_path / "separate"
    )
    assert separate["runs"][0]["status"] == "complete"
    assert separate["runs"][1]["status"] == "needs_confirmation"
    with pytest.raises(ValueError, match="every routine"):
        selection.reselect_checkpoint(
            roots,
            declaration={
                "name": "Compared Runs",
                "across_runs": True,
                "seed_order": [1, 2],
            },
            output_dir=tmp_path / "bad",
        )
    assert not (tmp_path / "bad").exists()


def test_child_reselection_reports_no_eligible_trained_actor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "child"
    item = _child_evidence(root)
    for row in item["records"]:
        row.update(method="pqn_vdn", optimizer_steps=0)
    _reader(monkeypatch, {root: item})
    result = selection.reselect_checkpoint(
        [root], declaration={"name": "Before Learning"}, output_dir=tmp_path / "choice"
    )
    assert result["status"] == "incomplete"
    assert result["runs"][0]["reason"] == "No eligible trained checkpoint"
    assert result["runs"][0]["winner"] is None


def test_across_run_routine_roots_require_permission_and_keep_unpaired_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    items = {roots[0]: _evidence(roots[0], 1), roots[1]: _evidence(roots[1], 2)}
    for row in items[roots[1]]["records"]:
        if row["purpose"] == "routine":
            row["root"] = 14
    _reader(monkeypatch, items)
    declaration = {"name": "Across Roots", "across_runs": True, "seed_order": [1, 2]}
    with pytest.raises(ValueError, match="allow_different_roots"):
        selection.reselect_checkpoint(
            roots, declaration=declaration, output_dir=tmp_path / "bad"
        )
    result = selection.reselect_checkpoint(
        roots,
        declaration={**declaration, "allow_different_roots": True},
        output_dir=tmp_path / "choice",
    )
    assert result["across_run_selection"]["routine_comparison"] == (
        "Unpaired: Different Declared Roots"
    )
    assert (
        result["across_run_selection"]["confirmation_comparison"]
        == "Common Declared Root"
    )


@pytest.mark.parametrize("rule", ["saved", "score_then_kills", "score_then_step"])
def test_historical_rules_stay_readable_but_cannot_create_new_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rule: str
) -> None:
    root = tmp_path / "run"
    evidence = _evidence(root)
    for row in evidence["records"]:
        row.pop("cells")
    _reader(monkeypatch, {root: evidence})
    declaration = {"name": "Historical Choice", "rule": rule}
    historical = selection._build_selection(
        [root],
        selection._declaration(declaration, allow_historical=True),
        tmp_path / "historical",
        None,
    )
    path = tmp_path / "saved_decision.json"
    path.write_text(json.dumps(historical))
    before = path.read_bytes()
    assert selection.read_selection_decision(path, verify_evidence=True) == historical
    expected = "a" if rule == "score_then_step" else "b"
    assert historical["runs"][0]["winner"]["checkpoint_id"] == expected
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="New selection decisions require"):
        selection.reselect_checkpoint(
            [root], declaration=declaration, output_dir=tmp_path / "new"
        )
    assert not (tmp_path / "new").exists()


def test_new_saved_decision_uses_points_not_wins_kills_or_unverified_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    evidence = _evidence(root)
    for row in evidence["records"]:
        if row["checkpoint_id"] == "b":
            row["score"] = 0.0
            row["mean_kill_difference"] = -100
        else:
            row["mean_point_margin"] = 10000
    _reader(monkeypatch, {root: evidence})
    result = selection.reselect_checkpoint(
        [root], declaration={"name": "Points Only"}, output_dir=tmp_path / "new"
    )
    assert result["declaration"]["rule"] == "point_margin"
    assert result["runs"][0]["winner"]["checkpoint_id"] == "b"
    for row in evidence["records"]:
        row.pop("cells")
    _reader(monkeypatch, {root: evidence})
    with pytest.raises(ValueError, match="saved points"):
        selection.reselect_checkpoint(
            [root], declaration={"name": "Missing Points"}, output_dir=tmp_path / "bad"
        )
    assert not (tmp_path / "bad").exists()
