"""Check canonical table scope, original ownership and read-only bounded access.

Synthetic complete records exercise the public saved reader and its memory view.
No ratings are fitted, no games are played and no official bundle is installed.
A reuse run of a snapshot saved before the Red Zone rule (pins 14, 2, 3) reads
its full reports with the schema-14 header and its metadata says
metric_schema_version 14, on the saved and the in-memory route alike; a current
snapshot uses schema 15 for both. Head-to-head views join reused game origins
to their recorded entrants and agree between saved and in-memory readers.
"""

# pyright: reportPrivateUsage=false

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from canonical_record_fixtures import build_record_bundle
from marl_battlegrounds.evaluation.canonical_results import CanonicalView
from marl_battlegrounds.evaluation.results import load_results
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.evaluation.tournament_assets import AssetVerifier
from marl_battlegrounds.evaluation.tournament_records import TournamentRecords
from marl_battlegrounds.evaluation.tournament_reuse import resolve_reuse_plan


def _saved(
    tmp_path: Path,
    *,
    mode: str = "priority",
    full: bool = False,
    historical: bool = False,
) -> tuple[Path, dict[str, Any], TournamentRecords]:
    bundle = build_record_bundle(
        tmp_path / "source", entrants=3, maps=1, full=full, historical=historical
    )
    config = bundle["config"]
    plan = resolve_reuse_plan(config, bundle["paths"])
    details: dict[str, object] = {
        "metrics": mode,
        "full_metrics_episodes": [1] if full else [],
        "replay_episodes": [],
        "snapshot_id": config["snapshot_id"],
    }
    with RunWriter(
        output_dir=tmp_path / "runs",
        phase="tournament",
        pass_id="schedule",
        details=details,
    ) as writer:
        writer.set_tournament_coordinator("schedule")
        reuse = {
            "version": 1,
            "source_descriptors": list(plan.source_descriptors),
            "execution_plan": list(plan.jobs),
            "budget": plan.budget,
            "challenger_id": None,
            "state": "incomplete",
        }
        writer._install_tournament_plan(config, plan.games, reuse)
        path = writer.run_dir
    manifest = json.loads((path / "run_details.json").read_bytes())
    records = TournamentRecords(
        config,
        plan.games,
        plan.jobs,
        AssetVerifier(config),
        manifest=manifest,
        run_dir=path,
    )
    return path, manifest, records


def test_saved_and_memory_tables_keep_original_rows_and_scope(tmp_path: Path) -> None:
    path, manifest, records = _saved(tmp_path)
    loaded = load_results(path, phase="tournament", pass_id="schedule")
    memory = CanonicalView(manifest, run_dir=None, memory={"_record_access": records})
    rows = list(memory.iter_rows("matches", 2))
    table = loaded.table("matches")
    assert len(rows) == len(table["episode_id"]) == 6
    assert table["run_id"].tolist() == [row["run_id"] for row in rows]
    assert all(row["run_id"] != manifest["run_id"] for row in rows)
    assert loaded.status == "incomplete"
    assert loaded.metadata["origin_join_version"] == 1
    assert loaded.metadata["tournament_owner"]["run_id"] == manifest["run_id"]
    assert loaded.table("episodes")["system_game_score"].tolist() == [None] * 6
    assert [
        len(batch["episode_id"]) for batch in loaded.iter_table("matches", rows=2)
    ] == [2, 2, 2]
    assert not (path / "match_results.csv").exists()


def test_none_keeps_outcomes_and_only_selected_full_rows(tmp_path: Path) -> None:
    path, _, _ = _saved(tmp_path, mode="none", full=True)
    loaded = load_results(path)
    assert len(loaded.table("matches")["outcome"]) == 6
    assert loaded.table("full_metrics")["episode_id"].tolist() == [1001]
    assert loaded.table("priority_metrics")["episode_id"].tolist() == [1001]
    assert loaded.table("tournament_headline_metrics") == {}
    assert (
        loaded.metadata["tables"]["tournament_headline_metrics"]["availability"]
        == "disabled"
    )
    assert len(list(loaded.iter_table("full_metrics", rows=1))) == 1


def test_full_reports_read_with_the_snapshot_pinned_schema_header(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.evaluation.metric_catalog import (
        FULL_METRIC_NAMES_BY_SCHEMA_VERSION,
    )
    from marl_battlegrounds.evaluation.run_writer import IDENTITY_COLUMNS

    for historical, version in ((True, 14), (False, 15)):
        root = tmp_path / str(version)
        root.mkdir()
        path, manifest, records = _saved(
            root, mode="full", full=True, historical=historical
        )
        assert records.scalar_schema == version
        header = (*IDENTITY_COLUMNS, *FULL_METRIC_NAMES_BY_SCHEMA_VERSION[version])
        loaded = load_results(path)
        # The run's own writer records 15; the result reports the snapshot pin.
        assert manifest["metric_schema_version"] == 15
        assert loaded.metadata["metric_schema_version"] == version
        memory = CanonicalView(
            manifest, run_dir=None, memory={"_record_access": records}
        )
        assert memory.metadata["metric_schema_version"] == version
        assert memory.metadata["tables"]["full_metrics"]["columns"] == header
        assert loaded.metadata["tables"]["full_metrics"]["columns"] == header
        table = loaded.table("full_metrics")
        assert tuple(table) == header
        assert len(table["episode_id"]) == 6
        assert any("red_zone" in name for name in table) is (not historical)


def test_missing_source_is_an_error_not_a_partial_table(tmp_path: Path) -> None:
    path, _, records = _saved(tmp_path)
    table_id = records.config["record_sources"][0]["tables"]["match_results"][
        "asset_id"
    ]
    Path(records.config["assets"][table_id]["path"]).unlink()
    loaded = load_results(path)
    with pytest.raises(
        (ValueError, FileNotFoundError), match=r"[Mm]issing|[Pp]repare|[Aa]sset"
    ):
        loaded.table("matches")


def test_loading_and_iteration_leave_run_bytes_unchanged(tmp_path: Path) -> None:
    path, _, _ = _saved(tmp_path)
    before = {
        str(file.relative_to(path)): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }
    loaded = load_results(path)
    list(loaded.iter_table("matches", rows=1))
    after = {
        str(file.relative_to(path)): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }
    assert before == after


def test_saved_reader_has_no_jax_provider_or_executor_imports(tmp_path: Path) -> None:
    path, _, _ = _saved(tmp_path)
    script = """
import sys
import marl_battlegrounds as marl_bgs
result = marl_bgs.load_results(sys.argv[1])
assert len(result.table('matches')['episode_id']) == 6
assert 'jax' not in sys.modules
assert 'marl_battlegrounds.evaluation.evaluate' not in sys.modules
assert 'marl_battlegrounds.evaluation.policy_execution' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "filename", ["tournament_config.json", "tournament_games.jsonl"]
)
def test_changed_immutable_plan_rejected_without_recovery(
    tmp_path: Path, filename: str
) -> None:
    path, _, _ = _saved(tmp_path)
    (path / filename).write_bytes((path / filename).read_bytes() + b" ")
    before = {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in path.iterdir()
        if file.is_file()
    }
    with pytest.raises(ValueError, match="missing or changed"):
        load_results(path)
    assert before == {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in path.iterdir()
        if file.is_file()
    }


def test_pending_restore_is_rejected_before_source_access(tmp_path: Path) -> None:
    path, manifest, records = _saved(tmp_path)
    manifest["recording_restore"] = {"version": 1}
    (path / "run_details.json").write_text(json.dumps(manifest))
    for asset in records.config["assets"].values():
        Path(asset["path"]).unlink(missing_ok=True)
    with pytest.raises(ValueError, match="restore is pending"):
        load_results(path)


def test_open_view_rejects_changed_logical_plan(tmp_path: Path) -> None:
    path, _, _ = _saved(tmp_path)
    loaded = load_results(path)
    (path / "tournament_games.jsonl").write_bytes(
        (path / "tournament_games.jsonl").read_bytes() + b"\n"
    )
    with pytest.raises(ValueError, match="missing or changed"):
        loaded.table("matches")


def test_open_view_rejects_changed_capture_claims(tmp_path: Path) -> None:
    path, manifest, _ = _saved(tmp_path)
    loaded = load_results(path)
    manifest["details"]["metrics"] = "none"
    (path / "run_details.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="capture settings changed"):
        loaded.table("matches")


def test_spawn_coverage_has_actual_owners_and_a_single_challenger_scope(
    tmp_path: Path,
) -> None:
    path, manifest, records = _saved(tmp_path)
    loaded = load_results(path)
    by_pair = loaded.metadata["spawn_balance"]
    assert len(by_pair) == 3
    assert all(
        row["completed_games"] == {"default": 1, "swapped": 1, "unreported": 0}
        for row in by_pair.values()
    )
    assert all(row["paired_complete"] is True for row in by_pair.values())
    challenger = records.config["participants"][0]["entrant_id"]
    manifest["tournament_reuse"]["challenger_id"] = challenger
    view = CanonicalView(manifest, run_dir=None, memory={"_record_access": records})
    focal = view.metadata["spawn_balance"]
    assert focal["system_id"] == challenger
    assert focal["opponent_id"] is None
    assert focal["completed_games"] == {"default": 2, "swapped": 2, "unreported": 0}
    assert focal["completed_pairs"] == focal["expected_pairs"] == 2
    assert len(focal["matchups"]) == 2


def test_example_prepares_saved_snapshot_and_only_needed_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from argparse import Namespace
    from importlib import import_module
    from importlib.util import module_from_spec, spec_from_file_location
    from types import SimpleNamespace

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation import tournament_assets, tournament_config

    spec = spec_from_file_location(
        "canonical_example",
        Path(__file__).parents[1] / "examples" / "canonical_tournament.py",
    )
    assert spec is not None and spec.loader is not None
    example = module_from_spec(spec)
    spec.loader.exec_module(example)
    previous = {
        "canonical_config": {"snapshot_id": "saved"},
        "metrics": "none",
        "rerun_existing": False,
    }

    def loaded(path: object) -> SimpleNamespace:
        return SimpleNamespace(metadata=previous)

    monkeypatch.setattr(marl_bgs, "load_results", loaded)
    calls = []

    def resolve(config: object, *, official: bool, saved: object) -> object:
        assert config is None and official and saved == previous["canonical_config"]
        return saved

    def prepare(config: object, **kwargs: object) -> dict[str, object]:
        calls.append((config, kwargs))
        return {"missing": [], "config": config}

    def stop(**kwargs: object) -> None:
        assert kwargs["config"] == previous["canonical_config"]
        raise RuntimeError("ready to execute")

    # Bind the real runner imports before replacing the example's dependencies.
    import_module("marl_battlegrounds.evaluation.canonical")
    monkeypatch.setattr(tournament_config, "resolve_tournament_config", resolve)
    monkeypatch.setattr(tournament_assets, "prepare_tournament_assets", prepare)
    monkeypatch.setattr(marl_bgs, "run_canonical_tournament", stop)
    args = Namespace(
        config=None,
        system=None,
        games_per_opponent=None,
        metrics=None,
        save_replays=None,
        rerun_existing=None,
        prepare=True,
        cache_dir=None,
        output_dir=None,
        resume_from=tmp_path,
        num_envs=32,
        chunk_size=16,
    )
    with pytest.raises(RuntimeError, match="ready to execute"):
        example.compare(args)
    assert calls[0][1]["roles"] == ("outcomes_priority",)


@pytest.mark.parametrize("challenger", [None, "tdm-beta"])
def test_generic_metadata_matches_direct_saved_and_completed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, challenger: str | None
) -> None:
    from importlib import import_module

    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )
    from marl_battlegrounds.evaluation.tournament import run_tournament

    provenance = capture_recording_provenance()

    def stable_source(**_: object) -> dict[str, object]:
        return provenance

    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    monkeypatch.setattr(evaluator, "capture_recording_provenance", stable_source)
    direct = run_tournament(
        config={
            "entrants": ["random", "tdm-alpha"],
            "maps": [47],
            "games_per_opponent": 2,
            "max_steps": 1,
            "metrics": "none",
        },
        challenger=challenger,
        num_envs=2,
        chunk_size=1,
        output_dir=tmp_path,
    )
    assert direct.paths is not None
    directory = direct.paths["run_details"].parent
    before = {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }
    saved = load_results(directory)
    assert before == {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }

    def no_new_games(*_: object, **__: object) -> None:
        pytest.fail("A complete resume must not play more games")

    monkeypatch.setattr(evaluator, "_evaluate_tournament_episodes", no_new_games)
    resumed = run_tournament(resume_from=directory, challenger=challenger)
    names = {"random", "tdm-alpha"}
    if challenger is not None:
        names.add(challenger)
    expected_games = 6 if challenger else 2
    for field in (
        "red_zone_depth",
        "num_matches",
        "schedule",
        "schedule_digest",
        "policies",
        "participants",
        "method_sampling",
        "configuration_ids_by_map",
        "input_metadata",
    ):
        assert (
            direct.metadata[field] == saved.metadata[field] == resumed.metadata[field]
        )
    assert (
        direct.metadata["num_matches"]
        == len(saved.metadata["schedule"])
        == expected_games
    )
    assert set(saved.metadata["participants"]) == names
    assert set(saved.metadata["method_sampling"]) == names
    assert {row["name"] for row in saved.metadata["policies"]} == names
    assert set(saved.metadata["input_metadata"]["participants"]) == {
        "random",
        "tdm-alpha",
    }
    assert saved.metadata["participants"] == {
        row["name"]: identifier for identifier, row in saved.metadata["systems"].items()
    }
    assert (
        saved.metadata["method_sampling"]
        == json.loads(before["run_details.json"])["tournament_summary"]["metadata"][
            "method_sampling"
        ]
    )
    script = """
import sys
import marl_battlegrounds as marl_bgs
result = marl_bgs.load_results(sys.argv[1])
assert result.metadata['num_matches'] == int(sys.argv[2])
assert 'jax' not in sys.modules
assert 'marl_battlegrounds.evaluation.policy_execution' not in sys.modules
assert 'marl_battlegrounds.evaluation.evaluate' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script, str(directory), str(expected_games)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_head_to_head_joins_reused_origins_and_keeps_saved_memory_parity(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.evaluation.results import SavedResults

    path, manifest, records = _saved(tmp_path, mode="none")
    saved = load_results(path)
    view = CanonicalView(manifest, run_dir=None, memory={"_record_access": records})
    memory = SavedResults(view.metadata, view)
    for by in ("overall", "map", "spawn"):
        left, right = saved.head_to_head(by), memory.head_to_head(by)
        for name in left:
            np.testing.assert_equal(left[name], right[name])
        assert set(left["system_id"]) == {"fixture-00", "fixture-01", "fixture-02"}
    overall = saved.head_to_head()
    assert overall["games"].tolist() == [2] * 6
    assert saved.head_to_head("spawn")["games"].tolist() == [1] * 12
    matrix = saved.opponent_matrix("games")["values"]
    np.testing.assert_equal(matrix, [[np.nan, 2, 2], [2, np.nan, 2], [2, 2, np.nan]])
    assert not (path / "match_results.csv").exists()
