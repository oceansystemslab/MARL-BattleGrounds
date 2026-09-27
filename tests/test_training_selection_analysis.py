"""Check reports that use a separate verified checkpoint-selection decision.

The shared selection reader owns actor/game verification and ranking. Synthetic
verified records isolate report wiring: incomplete or mismatched decisions fail
before output, selected identities remain exact, original choices are preserved,
final actors stay separate, and declared training groups use the chosen winner.
"""

# pyright: reportPrivateUsage=false

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.training import analysis, selection
from test_training_selection import _evidence, _reader

Record = dict[str, Any]


def _run(root: Path, seed: int) -> Record:
    root.mkdir()
    evidence = _evidence(root, seed)
    for row in evidence["records"]:
        if row["checkpoint_id"] == "a":
            row["score"] = 0.2
        row["cells"] = [
            {
                "map_id": map_id,
                "opponent": "Opponent",
                "score": row["score"],
                "mean_team_a_score": 10 + row["mean_kill_difference"],
                "mean_team_b_score": 10,
            }
            for map_id in row["maps"]
        ]
    original = next(
        row
        for row in evidence["records"]
        if row["checkpoint_id"] == "a" and row["purpose"] == "confirmation"
    )
    for actor in evidence["actors"].values():
        Path(actor["actor_path"]).mkdir(parents=True)
    for name, value in (
        (
            "run_details.json",
            {
                "run_id": evidence["run_id"],
                "config": {"seed": seed, "red_zone_depth": 5.0},
            },
        ),
        (
            "status.json",
            {
                "status": "complete",
                "selected_actor": evidence["actors"]["a"]["actor_path"],
                "final_actor": evidence["actors"]["final"]["actor_path"],
            },
        ),
        ("selection.json", original),
        ("validation_results.json", evidence["records"]),
    ):
        (root / name).write_text(json.dumps(value))
    return evidence


def test_verified_selection_controls_report_and_groups_without_rewriting_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    evidence = {root: _run(root, seed) for seed, root in enumerate(roots)}
    _reader(monkeypatch, evidence)
    decision_dir = tmp_path / "decision"
    decision = selection.reselect_checkpoint(
        roots, declaration={"name": "Declared New Choice"}, output_dir=decision_dir
    )
    decision_path = decision_dir / "selection_decision.json"
    old = analysis.analyze(roots, output_dir=tmp_path / "ordinary")
    assert all(row["selection"]["checkpoint_id"] == "a" for row in old["runs"])
    before = {
        path: path.read_bytes()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }
    grouping = {
        "schema_version": 1,
        "maps": decision["runs"][0]["winner"]["maps"],
        "opponents": ["Opponent"],
        "panel_digest": "panel",
        "red_zone_depth": 5.0,
        "runs": [
            {"run_dir": str(root), "group": "Method", "unit": f"seed-{index}"}
            for index, root in enumerate(roots)
        ],
    }
    observed: list[bool] = []
    reader = selection.read_selection_decision

    def checked(path: str | Path, *, verify_evidence: bool = False) -> Record:
        observed.append(verify_evidence)
        return reader(path, verify_evidence=verify_evidence)

    monkeypatch.setattr(selection, "read_selection_decision", checked)
    report = analysis.analyze(
        roots,
        output_dir=tmp_path / "new-report",
        selection=decision_path,
        grouping=grouping,
    )
    assert observed == [True]
    assert report["selection_decision"] == {
        "path": str(decision_path),
        "decision_id": decision["decision_id"],
        "name": "Declared New Choice",
    }
    for root, row in zip(roots, report["runs"], strict=True):
        assert row["selection"]["checkpoint_id"] == "b"
        assert row["selection"]["artifact_id"] == "export-b"
        assert row["selection"]["actor_digest"] == "inference-b"
        assert row["selected_actor"] == str(root / "actors/b")
        assert row["original_selection"]["checkpoint_id"] == "a"
        assert row["original_selection_path"] == str(root / "selection.json")
        assert row["original_selected_actor"] == str(root / "actors/a")
        assert row["final_actor"] == str(root / "actors/final")
    variation = report["training_variation"]
    assert [row["checkpoint_id"] for row in variation["runs"]] == ["b", "b"]
    assert variation["groups"][0]["score"] == pytest.approx(0.8)
    text = Path(report["artifacts"]["summary"]).read_text()
    assert "Separate declared decision" in text
    assert "Original Saved Selection" in text
    assert "Final Actor (Extra Result)" in text
    assert all(path.read_bytes() == value for path, value in before.items())


def test_incomplete_decision_fails_before_outputs_but_progress_report_still_works(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    evidence = _run(root, 1)
    evidence["records"] = [
        row
        for row in evidence["records"]
        if not (row["checkpoint_id"] == "b" and row["purpose"] == "confirmation")
    ]
    _reader(monkeypatch, {root: evidence})
    choice = tmp_path / "pending"
    decision = selection.reselect_checkpoint(
        [root], declaration={"name": "Pending"}, output_dir=choice
    )
    assert decision["status"] == "needs_confirmation"
    output = tmp_path / "blocked-report"
    with pytest.raises(ValueError, match=r"(?i)complete|confirmation"):
        analysis.analyze(
            [root], output_dir=output, selection=choice / "selection_decision.json"
        )
    assert not output.exists()
    (root / "status.json").write_text('{"status":"failed"}')
    report = analysis.analyze([root], output_dir=tmp_path / "progress")
    assert report["runs"][0]["status"] == "failed"
    assert report["complete"] is False
    assert "selection_decision" not in report


@pytest.mark.parametrize(
    "change",
    (
        "extra-run",
        "duplicate-run",
        "run-id",
        "winner",
        "artifact",
        "inference",
        "actor-path",
    ),
)
def test_mismatched_decision_cannot_write_a_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    root = tmp_path / "run"
    evidence = _run(root, 1)
    _reader(monkeypatch, {root: evidence})
    choice = tmp_path / "choice"
    selection.reselect_checkpoint(
        [root], declaration={"name": "Choice"}, output_dir=choice
    )
    path = choice / "selection_decision.json"
    runs = [root]
    if change == "extra-run":
        other = tmp_path / "other"
        _run(other, 2)
        runs.append(other)
    elif change == "duplicate-run":
        runs.append(root)
    elif change == "run-id":
        details = json.loads((root / "run_details.json").read_text())
        details["run_id"] = "different"
        (root / "run_details.json").write_text(json.dumps(details))
    else:
        value = json.loads(path.read_text())
        field = {
            "winner": "checkpoint_id",
            "artifact": "artifact_id",
            "inference": "actor_digest",
            "actor-path": "actor_path",
        }[change]
        value["runs"][0]["winner"][field] = "forged"
        value.pop("decision_id")
        value["decision_id"] = selection._digest(value)
        path.write_text(json.dumps(value))
    output = tmp_path / "report"
    with pytest.raises(ValueError):
        analysis.analyze(runs, output_dir=output, selection=path)
    assert not output.exists()


def test_shared_verifier_error_reaches_caller_without_partial_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    evidence = _run(root, 1)
    _reader(monkeypatch, {root: evidence})
    choice = tmp_path / "choice"
    selection.reselect_checkpoint(
        [root], declaration={"name": "Choice"}, output_dir=choice
    )
    # A different verified source summary changes the rebuilt decision even if a
    # caller recomputes the outer decision hash. No actor is restored in analysis.
    changed = copy.deepcopy(evidence)
    for row in changed["records"]:
        if row["checkpoint_id"] == "b":
            row["score"] = 0.0
    _reader(monkeypatch, {root: changed})
    with pytest.raises(ValueError):
        analysis.analyze(
            [root],
            output_dir=tmp_path / "report",
            selection=choice / "selection_decision.json",
        )
    assert not (tmp_path / "report").exists()
