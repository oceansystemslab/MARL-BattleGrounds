"""Check variable released fields and unchanged historical custom populations.

Artificial record bundles exercise reuse, complete ranking membership, original
provenance, and saved-first resume without loading or calling entrants. Separate
one-tick CPU games check public challenger and explicit fresh-execution counts.
Fixture catalogs are isolated test inputs, not an official release or research
results. Version-1 bytes and custom populations retain their original meaning.
"""

# Population tests inspect the immutable plan and isolate release lookup.
# pyright: reportPrivateUsage=false

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.canonical_fixtures import config_descriptor
from tests.canonical_record_fixtures import build_record_bundle, fixture_policy

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.evaluation import canonical, tournament_statistics
from marl_battlegrounds.evaluation import tournament_config as configs
from marl_battlegrounds.evaluation.results import CanonicalTournamentResult
from marl_battlegrounds.evaluation.tournament_assets import AssetVerifier
from marl_battlegrounds.evaluation.tournament_reuse import resolve_reuse_plan
from marl_battlegrounds.tasks import list_tdm_maps


def _released_v2(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> dict[str, Any]:
    released = copy.deepcopy(config)
    released["version"] = 2
    released["release"] = config_descriptor(entrants=12, official=True)["release"]
    maps = {item.map_id: item for item in list_tdm_maps()}
    for source in released["conditions"]["map_sources"]:
        source["registered_map"] = maps[source["map_id"]].model_dump(mode="json")
        source["split"] = maps[source["map_id"]].split
    for index, participant in enumerate(released["participants"]):
        participant["elo"] = 1212.0 - index
        participant["result_ref"] = {
            "source_id": released["record_sources"][0]["source_id"],
            "table": "tournament_results",
            "policy": participant["name"],
        }
    released["snapshot_id"] = configs.snapshot_identity(released)
    monkeypatch.setattr(
        configs,
        "_installed_catalog",
        lambda: {
            "default": released["snapshot_id"],
            "snapshots": {
                released["snapshot_id"]: {
                    "config": released,
                    "release": released["release"],
                }
            },
        },
    )
    return released


def _unexpected(*args: object, **kwargs: object) -> None:
    raise AssertionError("Complete saved reuse must not load, execute, refit or repin")


def _bytes(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("entrants,maps", [(2, 1), (5, 2), (12, 5), (16, 1)])
def test_released_population_reuses_every_game_and_resumes_its_saved_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrants: int,
    maps: int,
) -> None:
    bundle = build_record_bundle(tmp_path / "inputs", entrants=entrants, maps=maps)
    released = _released_v2(monkeypatch, bundle["config"])
    frozen_descriptor = configs.canonical_json(released)
    budget = released["conditions"]["games_per_opponent"]
    names = {row["name"] for row in released["participants"]}
    ids = {row["entrant_id"] for row in released["participants"]}
    expected_games = entrants * (entrants - 1) // 2 * budget

    verifier = AssetVerifier(released)
    plan = resolve_reuse_plan(
        released,
        canonical._plan_paths(released, verifier, True),
        challenger_id="population-challenger",
        official_verified=True,
    )
    challenger_games = [row for row in plan.games if row["origin"] is None]
    assert len(plan.games) == expected_games + entrants * budget
    assert len(challenger_games) == entrants * budget
    assert len(plan.jobs) == entrants
    assert {row["team_a"] for row in challenger_games} == {"population-challenger"}
    assert Counter(row["team_b"] for row in challenger_games) == dict.fromkeys(
        ids, budget
    )
    for opponent in ids:
        pair = [row for row in challenger_games if row["team_b"] == opponent]
        assert Counter((row["map_id"], row["spawn_locations"]) for row in pair) == {
            (source["map_id"], spawn): budget // (2 * maps)
            for source in released["conditions"]["map_sources"]
            for spawn in (0, 1)
        }
    assert [row["origin"] for row in plan.games[:expected_games]] == [
        row["origin"] for row in bundle["games"]
    ]

    monkeypatch.setattr(canonical, "_active_pair", _unexpected)
    result = marl_bgs.run_canonical_tournament(output_dir=tmp_path / "out")
    assert result.status == "complete"
    assert result.planned_games == result.reused_games == expected_games
    assert result.executed_games == result.metadata["executed_this_call"] == 0
    assert result.protocol_compliant is True
    assert result.metadata["official_snapshot_verified"] is True
    assert result.snapshot_id == result.metadata["big_12_id"] == released["snapshot_id"]
    assert set(result.table("tournament_rankings")["policy"]) == names
    assert len(result.table("tournament_rankings")["rank"]) == entrants
    assert len(result.matches) == expected_games
    assert result.matches == tuple(bundle["matches"])
    assert result.metadata["canonical_plan"]["games"] == tuple(bundle["games"])
    assert configs.canonical_json(released) == frozen_descriptor
    assert result.run_dir is not None
    before = _bytes(result.run_dir)

    monkeypatch.setattr(configs, "_installed_catalog", _unexpected)
    monkeypatch.setattr(tournament_statistics, "summarize_tournament", _unexpected)
    resumed = marl_bgs.run_canonical_tournament(resume_from=result.run_dir)
    loaded = marl_bgs.load_results(result.run_dir)
    assert resumed.matches == result.matches
    assert resumed.tournament_results == result.tournament_results
    assert resumed.metadata["executed_this_call"] == 0
    assert set(loaded.table("tournament_rankings")["policy"]) == names
    assert loaded.metadata["num_matches"] == expected_games
    assert (
        loaded.metadata["snapshot_id"]
        == loaded.metadata["big_12_id"]
        == released["snapshot_id"]
    )
    assert loaded.metadata["protocol_compliant"] is True
    assert _bytes(result.run_dir) == before


@pytest.mark.parametrize("entrants", [2, 5, 12])
def test_v1_custom_population_keeps_descriptor_bytes_and_saved_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrants: int
) -> None:
    bundle = build_record_bundle(tmp_path / "inputs", entrants=entrants, maps=1)
    original = bundle["config"]
    encoded = configs.canonical_json(original)
    decoded = configs.load_tournament_config(json.loads(encoded), official=False)
    assert decoded["version"] == 1
    assert configs.canonical_json(decoded) == encoded
    assert configs.snapshot_identity(decoded) == original["snapshot_id"]
    monkeypatch.setattr(canonical, "_active_pair", _unexpected)
    result = marl_bgs.run_tournament(config=decoded, output_dir=tmp_path / "out")
    assert isinstance(result, CanonicalTournamentResult)
    assert result.planned_games == result.reused_games == entrants * (entrants - 1)
    assert result.executed_games == 0
    assert result.protocol_compliant is False
    assert len(result.table("tournament_rankings")["rank"]) == entrants
    assert result.matches == tuple(bundle["matches"])
    assert result.run_dir is not None
    before = _bytes(result.run_dir)
    monkeypatch.setattr(configs, "_installed_catalog", _unexpected)
    monkeypatch.setattr(tournament_statistics, "summarize_tournament", _unexpected)
    resumed = marl_bgs.run_tournament(resume_from=result.run_dir)
    assert resumed.matches == result.matches
    assert resumed.tournament_results == result.tournament_results
    assert resumed.metadata["canonical_config"]["version"] == 1
    assert (
        resumed.metadata["canonical_config"]["snapshot_id"] == original["snapshot_id"]
    )
    assert resumed.metadata["big_12_id"] == original["snapshot_id"]
    assert _bytes(result.run_dir) == before
    assert configs.canonical_json(original) == encoded


@pytest.mark.parametrize(
    "roster",
    [
        ("mage", "warrior", "hunter", "rogue", "priest"),
        ("mage",),
        ("hunter", "warrior", "priest"),
        ("mage",) * 5,
    ],
)
def test_public_v2_challenger_and_explicit_fresh_run_use_all_required_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, roster: tuple[str, ...]
) -> None:
    # Build real source configurations and artificial rows for this exact roster.
    # The shared builder can have two import aliases during full collection.
    # Patch its own globals so another alias cannot redirect this roster change.
    monkeypatch.setitem(
        build_record_bundle.__globals__,
        "canonical_tournament_rosters",
        lambda: (roster, roster),
    )
    bundle = build_record_bundle(tmp_path / "inputs", entrants=2, maps=1, max_steps=1)
    bundle["config"]["conditions"]["rosters"] = {
        "team_a": list(roster),
        "team_b": list(roster),
    }
    released = _released_v2(monkeypatch, bundle["config"])
    from marl_battlegrounds.evaluation import admission

    # Exercise the real-release roster guard rather than its fixture-only bypass.
    admission._check_release_population(
        released, {"fixture_only": False}, AssetVerifier(released)
    )
    wrong_identity = copy.deepcopy(released)
    wrong_identity["participants"][0]["controller_id"] = "0" * 64
    wrong_identity["snapshot_id"] = configs.snapshot_identity(wrong_identity)
    with pytest.raises(ValueError, match="complete verified controller identities"):
        admission._check_release_population(
            wrong_identity, {"fixture_only": False}, AssetVerifier(wrong_identity)
        )
    challenger = fixture_policy(31)
    reused = marl_bgs.run_canonical_tournament(
        challenger,
        config=released,
        metrics="none",
        num_envs=2,
        chunk_size=1,
        output_dir=tmp_path / "comparison",
    )
    assert reused.status == "complete"
    assert reused.planned_games == 6
    assert reused.reused_games == 2
    assert reused.executed_games == reused.metadata["executed_this_call"] == 4
    games = reused.metadata["canonical_plan"]["games"]
    assert [row["origin"] for row in games[:2]] == [
        row["origin"] for row in bundle["games"]
    ]
    assert all(row["team_a"] == reused.challenger_id for row in games[2:])
    assert {row["team_b"] for row in games[2:]} == {
        row["entrant_id"] for row in released["participants"]
    }
    assert all(row["episode_length"] == 1 for row in reused.matches[2:])
    assert len(reused.table("tournament_rankings")["rank"]) == 3
    assert reused.run_dir is not None
    saved_bytes = _bytes(reused.run_dir)
    saved = marl_bgs.load_results(reused.run_dir)
    resumed = marl_bgs.run_canonical_tournament(challenger, resume_from=reused.run_dir)
    expected_columns = reused.table("matches")
    for result in (saved, resumed):
        actual_columns = result.table("matches")
        assert actual_columns.keys() == expected_columns.keys()
        for column in expected_columns:
            np.testing.assert_array_equal(
                actual_columns[column], expected_columns[column]
            )
    assert resumed.metadata["canonical_config"]["conditions"]["rosters"] == {
        "team_a": list(roster),
        "team_b": list(roster),
    }
    assert _bytes(reused.run_dir) == saved_bytes

    fresh = marl_bgs.run_canonical_tournament(
        challenger,
        config=released,
        rerun_existing=True,
        metrics="none",
        num_envs=2,
        chunk_size=1,
    )
    assert fresh.status == "complete"
    assert fresh.planned_games == fresh.executed_games == 6
    assert fresh.metadata["executed_this_call"] == 6
    assert fresh.reused_games == 0
    assert all(row["episode_length"] == 1 for row in fresh.matches)
    fresh_games = fresh.metadata["canonical_plan"]["games"]
    assert all(row["origin"] is None for row in fresh_games)
    assert [row["prior_origin"] for row in fresh_games[:2]] == [
        row["origin"] for row in bundle["games"]
    ]
    for original, rerun in zip(games, fresh_games, strict=True):
        for field in ("execution", "team_a", "team_b", "map_id", "spawn_locations"):
            assert rerun[field] == original[field]


def test_released_v2_still_checks_the_actual_source_roster_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = build_record_bundle(tmp_path / "inputs", entrants=2, maps=1)
    bundle["config"]["conditions"]["rosters"] = {"team_a": ["mage"], "team_b": ["mage"]}
    released = _released_v2(monkeypatch, bundle["config"])
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="Source roster differs"):
        marl_bgs.run_canonical_tournament(config=released, output_dir=output)
    assert not output.exists()
