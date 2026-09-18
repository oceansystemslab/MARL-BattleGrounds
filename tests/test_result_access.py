"""Check scoped host results, exact cells, bounded reads and truthful coverage.

Saved fixtures exercise current and historical schemas without running games.
The tests reject ambiguous scopes, incomplete summaries and destructive restore,
while preserving recorded row order, optional metrics and legacy result fields.
"""

import csv
import io
import json
import subprocess
import sys
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from marl_battlegrounds.evaluation.metric_catalog import PRIORITY_METRIC_NAMES
from marl_battlegrounds.evaluation.results import (
    EpisodeResult,
    EvaluationResult,
    load_results,
)
from marl_battlegrounds.evaluation.run_writer import (
    EPISODE_COLUMNS,
    IDENTITY_COLUMNS,
    MATCH_COLUMNS,
    RunWriter,
)


def _fixture(
    tmp_path: Path, *, modes: tuple[str, ...] = ("priority",), historical: bool = False
) -> tuple[Path, dict[str, Any]]:
    directory = tmp_path / "run"
    directory.mkdir()
    manifest: dict[str, Any] = {
        "schema_version": 1 if historical else 2,
        "metric_schema_id": "marlbg.tdm.scalar",
        "metric_schema_version": 13 if historical else 14,
        "run_id": "run",
        "systems": {},
        "configurations": {"cfg": {}},
        "tables": {},
        "details": {},
        "passes": {},
    }
    for index, mode in enumerate(modes):
        phase = "evaluation" if index == 0 else "validation"
        manifest["passes"][str(index)] = {
            "phase": phase,
            "pass_id": "same",
            "policies": {},
            "details": {
                "metrics": mode,
                "evaluation_contract": {
                    "version": 1,
                    "spawn_mode": "paired",
                    "pairing_protocol": "fixed-team-spawn-v1",
                },
            },
            "episodes": {
                str(value): {
                    "seed_id": 0,
                    "map_id": 0,
                    "config_id": "cfg",
                    "spawn_locations": value - 1,
                    "comparison_kind": "verified_spawn_pair",
                    "paired_comparison_key": "pair",
                }
                for value in (1, 2)
            },
            "completed_episode_ids": [2, 1],
            "replays": {},
            "recorded_metrics_by_episode": {"1": mode, "2": mode},
            "result_state": {
                "version": 1,
                "status": "complete",
                "schedule_digest": "digest",
                "reason": None,
            },
            "system_ids": {"team_a": "a", "team_b": "b"},
        }
    return directory, manifest


def _row(
    *, episode_id: int = 1, phase: str = "evaluation", length: int = 2**24 + 1
) -> dict[str, Any]:
    row = dict.fromkeys(IDENTITY_COLUMNS)
    row.update(
        run_id="run",
        phase=phase,
        pass_id="same",
        episode_id=episode_id,
        seed_id=0,
        map_id=0,
        config_id="cfg",
        team_a_policy="001",
        team_b_policy="B",
        checkpoint_id=None,
    )
    row.update(
        {
            f"agent_{slot}_{field}": 0
            for slot in range(10)
            for field in ("active", "class_id")
        }
    )
    row.update(outcome=1, episode_length=length, team_a_score=2**24 + 3, team_b_score=1)
    return row


def _table(
    directory: Path,
    manifest: dict[str, Any],
    filename: str,
    header: tuple[str, ...],
    rows: list[dict[str, Any]],
    *,
    suffix: bytes = b"",
) -> None:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(tuple(row.get(name) for name in header) for row in rows)
    payload = stream.getvalue().encode()
    (directory / filename).write_bytes(payload + suffix)
    manifest["tables"][filename] = {"durable_bytes": len(payload), "rows": len(rows)}


def _save(directory: Path, manifest: dict[str, Any]) -> None:
    (directory / "run_details.json").write_text(json.dumps(manifest))


def test_required_outcomes_exact_order_scope_and_game_score(tmp_path: Path) -> None:
    directory, manifest = _fixture(tmp_path, modes=("none", "none"))
    rows = [_row(episode_id=2), _row(episode_id=1), _row(phase="validation")]
    _table(
        directory,
        manifest,
        "episodes.csv",
        EPISODE_COLUMNS,
        rows,
        suffix=b"uncommitted",
    )
    _save(directory, manifest)
    result = load_results(directory, phase="evaluation", pass_id="same")
    table = result.table("episodes")
    assert table["episode_id"].tolist() == [2, 1]
    assert table["episode_length"].tolist() == [2**24 + 1] * 2
    assert table["episode_length"].dtype == np.int64
    assert table["team_a_policy"].tolist() == ["001", "001"]
    assert table["system_game_score"].tolist() == [1.0, 1.0]
    assert result.status == "complete"
    assert result.table("priority_metrics") == {}
    assert result.metadata["tables"]["priority_metrics"]["availability"] == "disabled"
    coverage = result.metadata["spawn_balance"]
    assert coverage["completed_games"] == {"default": 1, "swapped": 1, "unreported": 0}
    assert coverage["completed_steps"]["default"] == 2**24 + 1
    assert coverage["paired_complete"] is True
    with pytest.raises(ValueError, match="several phases"):
        load_results(directory, pass_id="same")
    with pytest.raises(ValueError, match="no recorded passes"):
        load_results(directory, phase="absent")
    with pytest.raises(ValueError, match="one saved run"):
        load_results(tmp_path)
    with pytest.raises(ValueError, match="tournament"):
        result.table("matches")


def test_bounded_iteration_detects_restore_and_ignores_append_suffix(
    tmp_path: Path,
) -> None:
    directory, manifest = _fixture(tmp_path)
    _table(
        directory,
        manifest,
        "episodes.csv",
        EPISODE_COLUMNS,
        [_row(episode_id=2), _row()],
    )
    _save(directory, manifest)
    result = load_results(directory)
    iterator = result.iter_table("episodes", rows=1)
    assert next(iterator)["episode_id"].tolist() == [2]
    manifest["tables"]["episodes.csv"]["durable_bytes"] -= 1
    _save(directory, manifest)
    with pytest.raises(ValueError, match=r"restored|truncated"):
        next(iterator)
    for size in (True, 0, -1):
        with pytest.raises(ValueError, match="positive integer"):
            list(result.iter_table("episodes", rows=size))
    with pytest.raises(ValueError, match="unknown table"):
        result.table("nope")


def test_none_tournament_outcomes_do_not_invent_priority(tmp_path: Path) -> None:
    directory, manifest = _fixture(tmp_path, modes=("none",))
    entry = manifest["passes"]["0"]
    entry["phase"] = "tournament"
    entry["recorded_metrics_by_episode"]["2"] = "full"
    rows: list[dict[str, Any]] = []
    for identifier in (1, 2):
        row = _row(episode_id=identifier, phase="tournament")
        row.update(block_id=1, bootstrap_group=None)
        if identifier == 2:
            row.update({name: 7.0 for name in PRIORITY_METRIC_NAMES if name not in row})
        rows.append(row)
    _table(directory, manifest, "match_results.csv", MATCH_COLUMNS, rows)
    _save(directory, manifest)
    result = load_results(directory)
    assert len(result.table("episodes")["episode_id"]) == 2
    assert result.table("priority_metrics")["episode_id"].tolist() == [2]
    assert result.table("priority_metrics")["team_a_return"].dtype == np.float32
    assert result.status == "incomplete"
    with pytest.raises(ValueError, match="finalized"):
        result.table("tournament_headline_metrics")


def test_historical_precision_and_missing_focal_binding(tmp_path: Path) -> None:
    directory, manifest = _fixture(tmp_path, historical=True)
    entry = manifest["passes"]["0"]
    entry["details"].pop("evaluation_contract")
    header = (*IDENTITY_COLUMNS, "team_a_return", "agent_0_return")
    row = _row()
    row.update(team_a_return=1.000000000000001, agent_0_return=5.25)
    _table(directory, manifest, "priority_metrics.csv", header, [row])
    _table(directory, manifest, "episodes.csv", EPISODE_COLUMNS, [row])
    _save(directory, manifest)
    result = load_results(directory)
    table = result.table("priority_metrics")
    assert table["team_a_return"].dtype == np.float64
    assert table["team_a_return"][0] == row["team_a_return"]
    assert table["agent_0_return"].tolist() == [5.25]
    assert result.table("episodes")["system_game_score"].tolist() == [None]


def test_memory_legacy_fields_and_new_order_do_not_copy_wide_arrays() -> None:
    header = (*IDENTITY_COLUMNS, *PRIORITY_METRIC_NAMES)
    rows = [_row(), _row(episode_id=2)]
    for row in rows:
        row.update({name: 1.0 for name in PRIORITY_METRIC_NAMES if name not in row})
    columns = {name: np.asarray([row[name] for row in rows]) for name in header}
    episodes = tuple(EpisodeResult(index, 0, 0, 1, 3, 1, 0, "cfg") for index in (1, 2))
    result = EvaluationResult(
        columns,
        {},
        episodes,
        {
            "run_id": "run",
            "phase": "evaluation",
            "pass_id": "same",
            "metrics": "priority",
            "completion_order": [2, 1],
            "evaluation_contract": {"version": 1},
        },
        (1, 2),
    )
    assert result.priority_metrics is columns
    assert result.episodes == episodes
    assert result.table("priority_metrics")["episode_id"].tolist() == [2, 1]
    assert result.table("episodes")["episode_id"].tolist() == [2, 1]
    assert result.run_dir is None and result.replay_paths == ()
    json.dumps(result.metadata, default=str)


def test_status_failed_then_resumed_success_and_restore_read_rejection(
    tmp_path: Path,
) -> None:
    directory, manifest = _fixture(tmp_path, modes=("none",))
    _table(
        directory,
        manifest,
        "episodes.csv",
        EPISODE_COLUMNS,
        [_row(), _row(episode_id=2)],
    )
    manifest["passes"]["0"]["result_state"]["status"] = "failed"
    _save(directory, manifest)
    assert load_results(directory).status == "failed"
    manifest["passes"]["0"]["result_state"]["status"] = "complete"
    _save(directory, manifest)
    (directory / "failures.jsonl").write_text('{"message":"old failure"}\n')
    assert load_results(directory).status == "complete"
    manifest["recording_restore"] = {}
    _save(directory, manifest)
    with pytest.raises(ValueError, match="restore is pending"):
        load_results(directory)


def test_load_module_does_not_import_jax_or_provider_modules() -> None:
    command = (
        "import sys; from marl_battlegrounds.evaluation.results import load_results; "
        "assert 'jax' not in sys.modules; "
        "assert not any('policy_execution' in name for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_writer_status_does_not_complete_an_open_ended_pass(tmp_path: Path) -> None:
    with RunWriter(tmp_path, details={"metrics": "none"}) as writer:
        writer.mark_pass_result("incomplete", schedule_digest="schedule")
        with pytest.raises(ValueError, match="every scheduled episode"):
            writer.mark_pass_result("complete", schedule_digest="schedule")
        run_dir = writer.run_dir
    assert load_results(run_dir).status == "incomplete"


def test_summary_text_exact_ties_and_whole_population_scope(tmp_path: Path) -> None:
    directory, manifest = _fixture(tmp_path, modes=("priority",))
    entry = manifest["passes"]["0"]
    entry["phase"] = "tournament"
    entry["result_state"]["schedule_digest"] = "population"
    manifest["passes"]["coordinator"] = {
        "phase": "tournament",
        "pass_id": "schedule",
        "pass_role": "tournament_coordinator",
        "episodes": {},
        "completed_episode_ids": [],
        "details": {},
        "replays": {},
        "result_state": {
            "version": 1,
            "status": "complete",
            "schedule_digest": "population",
            "reason": None,
        },
    }
    manifest["details"]["participants"] = {
        "First": "id-z",
        "Second": "id-a",
        "Third": "id-c",
    }
    manifest["tournament_summary"] = {
        "digest": "summary",
        "qualification": {"status": "complete"},
    }
    header = ("policy", "elo", "elo_interval_status", "matches")
    summary = [
        {
            "policy": "First",
            "elo": 1200.0,
            "elo_interval_status": "Insufficient blocks",
            "matches": 2,
        },
        {
            "policy": "Second",
            "elo": 1200.0,
            "elo_interval_status": "Unavailable",
            "matches": 2,
        },
        {
            "policy": "Third",
            "elo": 1199.9999999999,
            "elo_interval_status": "Unavailable",
            "matches": 2,
        },
    ]
    _table(directory, manifest, "tournament_results.csv", header, summary)
    _save(directory, manifest)
    result = load_results(directory)
    assert result.status == "complete"
    rankings = result.table("tournament_rankings")
    assert rankings["policy"].tolist() == ["Second", "First", "Third"]
    assert rankings["rank"].tolist() == [1, 1, 3]
    assert rankings["pass_id"].tolist() == [None] * 3
    assert rankings["elo_interval_status"].tolist() == [
        "Unavailable",
        "Insufficient blocks",
        "Unavailable",
    ]
    assert load_results(directory, phase="tournament").status == "complete"
    with pytest.raises(ValueError, match="complete tournament population"):
        load_results(directory, phase="tournament", pass_id="same").table(
            "tournament_results"
        )


def test_missing_historical_lengths_are_not_zero(tmp_path: Path) -> None:
    directory, manifest = _fixture(tmp_path, historical=True)
    _save(directory, manifest)
    result = load_results(directory)
    coverage = result.metadata["spawn_balance"]
    assert coverage["completed_steps"] == {
        "default": None,
        "swapped": None,
        "unreported": 0,
    }
    assert coverage["missing_episode_lengths"] == (1, 2)
    with pytest.raises(ValueError, match="not recorded"):
        result.table("episodes")


def test_failure_marker_is_durable_and_clears_only_on_explicit_resume(
    tmp_path: Path,
) -> None:
    writer = RunWriter(tmp_path, details={"metrics": "none"})
    writer.mark_pass_result("incomplete", schedule_digest="schedule")
    error = RuntimeError("provider failed")
    writer.record_failure(error)
    directory = writer.run_dir
    writer.close()
    assert load_results(directory).status == "failed"
    with RunWriter(resume_from=directory, details={"metrics": "none"}) as resumed:
        resumed.mark_pass_result("incomplete", schedule_digest="schedule")
    assert load_results(directory).status == "incomplete"
    assert (directory / "failures.jsonl").exists()


def test_current_metric_blanks_use_nan_and_historical_headers_stay_exact(
    tmp_path: Path,
) -> None:
    directory, manifest = _fixture(tmp_path)
    row = _row()
    header = (*IDENTITY_COLUMNS, *PRIORITY_METRIC_NAMES)
    row.update({name: None for name in PRIORITY_METRIC_NAMES})
    _table(directory, manifest, "priority_metrics.csv", header, [row])
    _save(directory, manifest)
    result = load_results(directory)
    assert tuple(result.table("priority_metrics")) == header
    assert np.isnan(result.table("priority_metrics")["team_a_return"][0])
    assert result.table("priority_metrics")["team_a_return"].dtype == np.float32


def test_wide_reader_stops_after_requested_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation import scalar_reports
    from marl_battlegrounds.evaluation.metric_catalog import FULL_METRIC_NAMES

    directory, manifest = _fixture(tmp_path, modes=("full",))
    header = (*IDENTITY_COLUMNS, *FULL_METRIC_NAMES)
    rows: list[dict[str, Any]] = []
    for identifier in (2, 1):
        row = _row(episode_id=identifier)
        row.update({name: 1.0 for name in FULL_METRIC_NAMES})
        rows.append(row)
    _table(directory, manifest, "full_metrics.csv", header, rows)
    _save(directory, manifest)
    original = scalar_reports._csv_rows  # pyright: ignore[reportPrivateUsage]
    seen: list[int] = []

    def counted(stream: io.TextIOBase) -> Iterator[list[str]]:
        for row in original(stream):
            seen.append(len(row))
            yield row

    monkeypatch.setattr(scalar_reports, "_csv_rows", counted)
    result = load_results(directory)
    assert seen == []
    iterator = result.iter_table("full_metrics", rows=1)
    first = next(iterator)
    assert len(first) == len(header)
    assert all(value.shape == (1,) for value in first.values())
    assert len(seen) == 2
    assert isinstance(iterator, Generator)
    iterator.close()


def test_truncation_without_manifest_change_is_detected(tmp_path: Path) -> None:
    directory, manifest = _fixture(tmp_path)
    _table(
        directory,
        manifest,
        "episodes.csv",
        EPISODE_COLUMNS,
        [_row(), _row(episode_id=2)],
    )
    _save(directory, manifest)
    result = load_results(directory)
    iterator = result.iter_table("episodes", rows=1)
    next(iterator)
    path = directory / "episodes.csv"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        next(iterator)


@pytest.mark.parametrize("tournament", (False, True))
def test_historical_none_selected_full_preserves_stored_priority(
    tmp_path: Path, tournament: bool
) -> None:
    directory, manifest = _fixture(tmp_path, modes=("none",), historical=True)
    entry = manifest["passes"]["0"]
    entry.pop("recorded_metrics_by_episode")
    entry.pop("result_state")
    entry["details"].pop("evaluation_contract")
    entry["details"]["full_metrics_episodes"] = [1]
    phase = "tournament" if tournament else "evaluation"
    entry["phase"] = phase
    row = _row(phase=phase)
    row.update(team_a_return=2.5, agent_0_return=2.5)
    if tournament:
        header = (
            *IDENTITY_COLUMNS,
            "block_id",
            "outcome",
            "episode_length",
            "team_a_score",
            "team_b_score",
            "team_a_return",
            "agent_0_return",
        )
        rows = [row, _row(phase=phase, episode_id=2)]
        filename = "match_results.csv"
    else:
        header = (*IDENTITY_COLUMNS, "team_a_return", "agent_0_return")
        rows = [row]
        filename = "priority_metrics.csv"
    _table(directory, manifest, filename, header, rows)
    _save(directory, manifest)
    result = load_results(directory)
    assert result.metadata["tables"]["priority_metrics"]["availability"] == "available"
    values = result.table("priority_metrics")
    assert values["episode_id"].tolist() == [1]
    assert values["team_a_return"].tolist() == [2.5]
    assert values["agent_0_return"].tolist() == [2.5]
    assert values["team_a_return"].dtype == np.float64


def test_same_pass_label_in_two_phases_keeps_each_row_owner(tmp_path: Path) -> None:
    directory, manifest = _fixture(tmp_path, modes=("none", "none"))
    validation = manifest["passes"]["1"]["details"]
    validation.pop("evaluation_contract")
    validation["first_system_team"] = "team_b"
    _table(
        directory,
        manifest,
        "episodes.csv",
        EPISODE_COLUMNS,
        [_row(), _row(phase="validation")],
    )
    _save(directory, manifest)
    result = load_results(directory)
    values = result.table("episodes")
    assert values["phase"].tolist() == ["evaluation", "validation"]
    assert values["system_game_score"].tolist() == [1.0, 0.0]


def test_declared_zero_game_sources_and_repeats_keep_honest_coverage(
    tmp_path: Path,
) -> None:
    directory, manifest = _fixture(tmp_path, modes=("none",))
    entry = manifest["passes"]["0"]
    played = {
        "map_id": 0,
        "source_config_id": "source-a",
        "map_metadata": [{"split": "validation"}],
    }
    unused = {
        "map_id": 1,
        "source_config_id": "source-b",
        "map_metadata": [{"split": "test"}],
    }
    entry["details"]["evaluation_contract"]["source_choices"] = [played, unused, played]
    for declaration in entry["episodes"].values():
        declaration["source_config_id"] = "source-a"
    _table(
        directory,
        manifest,
        "episodes.csv",
        EPISODE_COLUMNS,
        [_row(length=2), _row(episode_id=2, length=3)],
    )
    _save(directory, manifest)
    coverage = load_results(directory).metadata["spawn_balance"]["source_coverage"]
    assert len(coverage) == 2
    assert coverage[0]["source_choice_indices"] == (0, 2)
    assert coverage[0]["scheduled_games"] == coverage[0]["completed_games"] == 2
    assert coverage[0]["completed_steps"] == 5
    assert coverage[1] == {
        **unused,
        "source_choice_indices": (1,),
        "scheduled_games": 0,
        "completed_games": 0,
        "completed_steps": 0,
    }


def test_selected_pass_source_contents_survive_global_played_config_table(
    tmp_path: Path,
) -> None:
    directory, manifest = _fixture(tmp_path, modes=("none", "none"))
    manifest["passes"]["0"]["details"]["configurations"] = {
        "source-a": {"max_steps": 7}
    }
    manifest["passes"]["1"]["details"]["configurations"] = {
        "source-b": {"max_steps": 9}
    }
    _save(directory, manifest)
    result = load_results(directory, phase="evaluation", pass_id="same")
    assert result.metadata["configurations"] == {
        "cfg": {},
        "source-a": {"max_steps": 7},
    }
    assert load_results(directory).metadata["configurations"] == {
        "cfg": {},
        "source-a": {"max_steps": 7},
        "source-b": {"max_steps": 9},
    }
    manifest["passes"]["0"]["details"]["configurations"]["cfg"] = {"changed": True}
    _save(directory, manifest)
    with pytest.raises(ValueError, match="conflicting recorded configuration"):
        load_results(directory, phase="evaluation", pass_id="same")


@pytest.mark.parametrize("kind", ("exact", "tournament", "historical"))
def test_replay_paths_follow_declared_order_not_json_object_keys(
    tmp_path: Path,
    kind: str,
) -> None:
    directory, manifest = _fixture(tmp_path, modes=("none",))
    entry = manifest["passes"].pop("0")
    entry["episodes"] = {str(value): {"seed_id": value} for value in (11, 2, 9)}
    entry["completed_episode_ids"] = [2, 11, 9]
    entry["replays"] = {
        str(value): {"path": f"replays/{value}.json", "bytes": 3}
        for value in (11, 2, 9)
    }
    (directory / "replays").mkdir()
    for value in (11, 2, 9):
        (directory / "replays" / f"{value}.json").write_bytes(b"{}\n")
    if kind == "exact":
        entry["details"]["evaluation_contract"]["episode_ids"] = [9, 2, 11]
        manifest["passes"]["one"] = entry
        expected = [9, 2, 11]
    elif kind == "historical":
        entry["details"].pop("evaluation_contract")
        manifest["passes"]["one"] = entry
        expected = [2, 9, 11]
    else:
        entry["phase"] = "tournament"
        manifest["details"]["schedule"] = [
            {"episode_id": value} for value in (9, 2, 11)
        ]
        for key, values in (("a", (11,)), ("z", (2, 9))):
            manifest["passes"][key] = {
                **entry,
                "pass_id": key,
                "episodes": {str(i): entry["episodes"][str(i)] for i in values},
                "completed_episode_ids": list(values),
                "replays": {str(i): entry["replays"][str(i)] for i in values},
            }
        expected = [9, 2, 11]
    (directory / "run_details.json").write_text(json.dumps(manifest, sort_keys=True))
    assert [path.stem for path in load_results(directory).replay_paths] == list(
        map(str, expected)
    )


def test_original_coordinator_qualifies_only_complete_recorded_population(
    tmp_path: Path,
) -> None:
    directory, manifest = _fixture(tmp_path, modes=("none",))
    games = manifest["passes"]["0"]
    games["phase"] = "tournament"
    games.pop("result_state")
    games["details"].pop("evaluation_contract")
    manifest["passes"]["coordinator"] = {
        "phase": "tournament",
        "pass_id": "schedule",
        "episodes": {},
        "completed_episode_ids": [],
        "details": {},
    }
    manifest["details"] = {
        "schedule_digest": "old",
        "policies": ["a", "b"],
        "map_ids": [0],
        "num_matches": 2,
        "configuration_ids_by_map": {"0": "cfg"},
    }
    manifest["tournament_summary"] = {"digest": "old-summary"}
    _table(
        directory,
        manifest,
        "match_results.csv",
        MATCH_COLUMNS,
        [_row(phase="tournament"), _row(phase="tournament", episode_id=2)],
    )
    for filename in (
        "tournament_results.csv",
        "matchup_results.csv",
        "map_results.csv",
    ):
        _table(
            directory,
            manifest,
            filename,
            ("policy", "elo"),
            [{"policy": "a", "elo": 1000}],
        )
    _save(directory, manifest)
    result = load_results(directory)
    assert result.status == "complete"
    assert result.table("tournament_results")["policy"].tolist() == ["a"]
    assert ("tournament", "schedule") not in result.metadata["spawn_balance"]
    with pytest.raises(ValueError, match="not recorded"):
        result.table("tournament_headline_metrics")
    games["completed_episode_ids"] = [1]
    _table(
        directory,
        manifest,
        "match_results.csv",
        MATCH_COLUMNS,
        [_row(phase="tournament")],
    )
    _save(directory, manifest)
    assert load_results(directory).status == "incomplete"
