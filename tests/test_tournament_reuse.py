"""Check immutable tournament schedules, balanced reuse and original RNG ownership.

These host-only fixtures contain declared records rather than simulated games.
They prove schedule selection, origins, extensions and job/capture projections;
the shared reader separately proves report contents and physical configurations.
"""

import copy
import json
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.evaluation.tournament_config import canonical_json
from marl_battlegrounds.evaluation.tournament_reuse import (
    CHALLENGER_PLACEHOLDER,
    ReusePlan,
    analysis_schedule,
    rebuild_companion,
    record_source_identity,
    resolve_reuse_plan,
    selection_unit_id,
)


def _asset(data: bytes, role: str = "schedule") -> dict[str, object]:
    return {
        "sha256": sha256(data).hexdigest(),
        "size_bytes": len(data),
        "role": role,
        "path": None,
        "url": None,
    }


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))


def _fixture(
    tmp_path: Path,
    *,
    entrants: int = 12,
    maps: int = 5,
    pairs: int = 2,
    saved: bool = True,
) -> tuple[
    dict[str, Any],
    dict[str, Path],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    names = [f"entrant-{index:02}" for index in range(entrants)]
    assets: dict[str, Any] = {
        "source-manifest": _asset(b"{}", "run_manifest"),
        "source-rows": _asset(b"episode_id\n", "outcomes_priority"),
    }
    source: dict[str, Any] = {
        "run_id": "original-run",
        "manifest_asset": "source-manifest",
        "tables": {
            "match_results": {
                "asset_id": "source-rows",
                "committed_bytes": len(b"episode_id\n"),
                "rows": entrants * (entrants - 1) // 2 * maps * pairs * 2,
                "header_sha256": sha256(b"episode_id\n").hexdigest(),
            }
        },
        "replays": [],
    }
    source["source_id"] = record_source_identity(source, assets)
    games: list[dict[str, Any]] = []
    companion: list[dict[str, Any]] = []
    logical_id, pair_id = 1, 1
    for a, b in [
        *combinations(names, 2),
        *((CHALLENGER_PLACEHOLDER, name) for name in names),
    ]:
        target = companion if a == CHALLENGER_PLACEHOLDER else games
        for map_id in range(maps):
            for order in range(pairs):
                for spawn in (0, 1):
                    execution_id = 1000 + logical_id
                    target.append(
                        {
                            "logical_game_id": logical_id,
                            "matchup_id": f"matchup:{a}:{b}",
                            "pair_id": f"pair:{pair_id}",
                            "pair_order": order,
                            "team_a": a,
                            "team_b": b,
                            "map_id": map_id,
                            "source_config_id": f"config:{map_id}",
                            "resolved_config_id": f"config:{map_id}:spawn:{spawn}",
                            "spawn_locations": spawn,
                            "execution": {
                                "group_id": "first",
                                "root_seed": 47,
                                "seed_id": pair_id,
                                "episode_id": execution_id,
                                "action_stream_version": "evaluation-systems-v1",
                                "initialization_stream_version": (
                                    "evaluation-systems-v1"
                                ),
                                "bootstrap_group": None,
                            },
                            "origin": {
                                "source_id": source["source_id"],
                                "run_id": "original-run",
                                "phase": "tournament",
                                "pass_id": "old-pair",
                                "episode_id": execution_id,
                            }
                            if saved and target is games
                            else None,
                            "prior_origin": None,
                        }
                    )
                    logical_id += 1
                pair_id += 1
    manifest = {
        "format": "marlbg-tournament-schedule",
        "version": 1,
        "games_asset": "games",
        "challenger_games_asset": "companion",
        "execution_groups": [
            {
                "group_id": "first",
                "root_seed": 47,
                "action_stream_version": "evaluation-systems-v1",
                "initialization_stream_version": "evaluation-systems-v1",
                "next_episode_id": 1000 + logical_id,
                "next_seed_id": pair_id,
                "extension": "independent-pairs-v1",
            }
        ],
        "selection_blocks": None,
        "allocation_order": names,
    }
    paths = {
        name: tmp_path / f"{name}.json" for name in ("schedule", "games", "companion")
    }
    _write(paths["schedule"], manifest)
    _write_rows(paths["games"], games)
    _write_rows(paths["companion"], companion)
    for name, path in paths.items():
        assets[name] = _asset(path.read_bytes())
    config: dict[str, Any] = {
        "participants": [{"entrant_id": name, "name": name} for name in names],
        "conditions": {
            "map_sources": [
                {"map_id": index, "source_config_id": f"config:{index}"}
                for index in range(maps)
            ],
            "games_per_opponent": maps * pairs * 2,
            "schedule_asset": "schedule",
        },
        "compatibility": {
            "action_stream_version": "evaluation-systems-v1",
            "initialization_stream_version": "evaluation-systems-v1",
        },
        "assets": assets,
        "record_sources": [source],
        "release": {"rules_id": "fixture-only"} if entrants == 12 else None,
    }
    return config, paths, manifest, games, companion


def test_twelve_only_and_challenger_keep_incumbent_ids_streams_and_sources(
    tmp_path: Path,
) -> None:
    config, paths, _, _, _ = _fixture(tmp_path)
    before = copy.deepcopy(config)
    twelve = resolve_reuse_plan(config, paths, official_verified=True)
    thirteen = resolve_reuse_plan(
        config, paths, challenger_id="00-challenger", official_verified=True
    )
    assert len(twelve.games) == 66 * 20
    assert len(thirteen.games) == 78 * 20
    assert thirteen.games[: len(twelve.games)] == twelve.games
    assert len(twelve.jobs) == 0
    assert len(thirteen.jobs) == 12
    assert {row["team_a"] for row in thirteen.games[len(twelve.games) :]} == {
        "00-challenger"
    }
    assert all(row["origin"] is not None for row in twelve.games)
    assert config == before
    assert thirteen.budget == {
        "official_games_per_opponent": 20,
        "resolved_games_per_opponent": 20,
        "budget_override": False,
        "protocol_compliant": True,
    }
    unverified = resolve_reuse_plan(config, paths)
    assert unverified.games == twelve.games
    assert unverified.budget["protocol_compliant"] is False


def test_map_major_source_selects_equal_prefixes_for_every_cell(tmp_path: Path) -> None:
    config, paths, _, _, _ = _fixture(tmp_path, pairs=5)
    plan = resolve_reuse_plan(
        config, paths, challenger_id="challenger", games_per_opponent=10
    )
    assert len(plan.games) == 78 * 10
    assert set(
        Counter((row["matchup_id"], row["map_id"]) for row in plan.games).values()
    ) == {2}
    assert {row["pair_order"] for row in plan.games} == {0}
    assert plan.budget["budget_override"] is True
    assert plan.budget["protocol_compliant"] is False
    assert all(len(job["logical_game_ids"]) == 10 for job in plan.jobs)


def test_100_game_request_never_keeps_500_incumbent_games(tmp_path: Path) -> None:
    config, paths, _, _, _ = _fixture(tmp_path, entrants=3, pairs=50)
    plan = resolve_reuse_plan(
        config, paths, challenger_id="challenger", games_per_opponent=100
    )
    counts = Counter(row["matchup_id"] for row in plan.games)
    assert len(counts) == 6
    assert set(counts.values()) == {100}
    assert {row["pair_order"] for row in plan.games} == set(range(10))


@pytest.mark.parametrize("budget", [True, False, 0, -1, 10.0, 11, 2**31])
def test_bad_budgets_fail_without_execution(tmp_path: Path, budget: object) -> None:
    config, paths, _, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError):
        resolve_reuse_plan(config, paths, games_per_opponent=budget)  # type: ignore[arg-type]


def test_fresh_execution_preserves_ancestry_and_uniform_budget(tmp_path: Path) -> None:
    config, paths, _, _, _ = _fixture(tmp_path)
    reused = resolve_reuse_plan(
        config, paths, challenger_id="challenger", games_per_opponent=10
    )
    fresh = resolve_reuse_plan(
        config,
        paths,
        challenger_id="challenger",
        games_per_opponent=10,
        rerun_existing=True,
    )
    assert len(fresh.jobs) == 78
    assert all(row["origin"] is None for row in fresh.games)
    assert sum(row["prior_origin"] is not None for row in fresh.games) == 66 * 10
    for old, new in zip(reused.games, fresh.games, strict=True):
        assert old["execution"] == new["execution"]
        assert old["logical_game_id"] == new["logical_game_id"]
    assert analysis_schedule(reused) == analysis_schedule(fresh)


def test_insufficient_reuse_requires_explicit_complete_fresh_execution(
    tmp_path: Path,
) -> None:
    config, paths, _, _, _ = _fixture(tmp_path, pairs=1)
    with pytest.raises(ValueError, match="rerun_existing=True"):
        resolve_reuse_plan(
            config, paths, challenger_id="challenger", games_per_opponent=20
        )
    plan = resolve_reuse_plan(
        config,
        paths,
        challenger_id="challenger",
        games_per_opponent=20,
        rerun_existing=True,
    )
    assert len(plan.games) == 78 * 20
    assert len(plan.jobs) == 78
    assert {row["pair_order"] for row in plan.games} == {0, 1}
    assert all(row["origin"] is None for row in plan.games)
    group = plan.schedule_manifest["execution_groups"][0]
    assert group["next_episode_id"] > max(
        row["execution"]["episode_id"] for row in plan.games
    )
    assert group["next_seed_id"] > max(
        row["execution"]["seed_id"] for row in plan.games
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("extension", "none", "cannot extend"),
        ("next_episode_id", 2**31 - 1, "limits"),
        ("next_seed_id", 2**32, "limits"),
        ("next_episode_id", 1, "overlap"),
        ("next_seed_id", 0, "overlap"),
    ],
)
def test_fresh_extension_checks_rules_and_integer_exhaustion(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    config, paths, manifest, _, _ = _fixture(tmp_path, entrants=2, maps=1, pairs=1)
    manifest["execution_groups"][0][field] = value
    _write(paths["schedule"], manifest)
    with pytest.raises(ValueError, match=message):
        resolve_reuse_plan(config, paths, games_per_opponent=4, rerun_existing=True)


@pytest.mark.parametrize("field", ["next_episode_id", "next_seed_id"])
def test_twelve_only_extension_reserves_unused_companion_coordinates(
    tmp_path: Path, field: str
) -> None:
    config, paths, manifest, games, companion = _fixture(
        tmp_path, entrants=3, maps=1, pairs=1
    )
    plan = resolve_reuse_plan(config, paths, games_per_opponent=4, rerun_existing=True)
    added = [row for row in plan.games if row["pair_order"] == 1]
    assert min(row["logical_game_id"] for row in added) > max(
        row["logical_game_id"] for row in companion
    )
    coordinate = "episode_id" if field == "next_episode_id" else "seed_id"
    manifest["execution_groups"][0][field] = (
        max(row["execution"][coordinate] for row in games) + 1
    )
    _write(paths["schedule"], manifest)
    with pytest.raises(ValueError, match="overlap"):
        resolve_reuse_plan(config, paths, games_per_opponent=4, rerun_existing=True)


def test_custom_fresh_route_and_provisional_required_reuse_are_distinct(
    tmp_path: Path,
) -> None:
    config, paths, _, _, _ = _fixture(tmp_path, entrants=3, maps=1, saved=False)
    assert len(resolve_reuse_plan(config, paths).jobs) == 3
    with pytest.raises(ValueError, match="Incumbent records are missing"):
        resolve_reuse_plan(config, paths, require_reuse=True)
    assert (
        len(
            resolve_reuse_plan(
                config, paths, require_reuse=True, rerun_existing=True
            ).jobs
        )
        == 3
    )


def test_capture_projection_uses_original_episode_ids_and_skips_reuse(
    tmp_path: Path,
) -> None:
    config, paths, _, _, _ = _fixture(tmp_path, entrants=3, maps=1, pairs=1)
    plan = resolve_reuse_plan(config, paths, challenger_id="challenger")
    reused, new = plan.games[0], plan.games[-1]
    captures = plan.capture_ids([reused["logical_game_id"], new["logical_game_id"]])
    assert sum(len(value) for value in captures.values()) == 1
    assert (new["execution"]["episode_id"],) in captures.values()
    with pytest.raises(ValueError, match="distinct"):
        plan.capture_ids([1, 1])
    with pytest.raises(ValueError, match="resolved schedule"):
        plan.capture_ids([2**31 - 1])


def test_analysis_projection_keeps_equal_seed_numbers_under_different_roots_distinct(
    tmp_path: Path,
) -> None:
    config, paths, manifest, games, _ = _fixture(tmp_path, entrants=2, maps=1, pairs=2)
    group = copy.deepcopy(manifest["execution_groups"][0])
    group.update(group_id="second", root_seed=48)
    manifest["execution_groups"].append(group)
    for row in games[2:]:
        row["execution"].update(
            group_id="second", root_seed=48, seed_id=games[0]["execution"]["seed_id"]
        )
    _write(paths["schedule"], manifest)
    _write_rows(paths["games"], games)
    schedule = analysis_schedule(resolve_reuse_plan(config, paths))
    assert schedule[0].seed_id == schedule[1].seed_id
    assert schedule[2].seed_id == schedule[3].seed_id
    assert schedule[0].seed_id != schedule[2].seed_id
    assert [row.episode_id for row in schedule] == [1, 2, 3, 4]


def test_colliding_execution_ids_split_passes_without_renumbering(
    tmp_path: Path,
) -> None:
    config, paths, _, games, _ = _fixture(tmp_path, entrants=2, maps=1, pairs=2)
    for old, new in zip(games[:2], games[2:], strict=True):
        new["execution"]["episode_id"] = old["execution"]["episode_id"]
    _write_rows(paths["games"], games)
    plan = resolve_reuse_plan(config, paths, rerun_existing=True)
    assert len(plan.jobs) == 2
    assert plan.capture_ids([row["logical_game_id"] for row in plan.games]) == {
        0: (1001, 1002),
        1: (1001, 1002),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("logical_game_id", 1, "Duplicate logical"),
        ("spawn_locations", 0, "both spawn"),
        ("team_a", "entrant-01", "two different"),
        ("source_config_id", "wrong", "source/map"),
    ],
)
def test_invalid_game_declarations_fail_before_a_plan(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    config, paths, _, games, _ = _fixture(tmp_path, entrants=2, maps=1, pairs=1)
    games[1][field] = value
    _write_rows(paths["games"], games)
    with pytest.raises(ValueError, match=message):
        resolve_reuse_plan(config, paths)


def test_duplicate_origin_and_unreported_rng_coupling_fail(tmp_path: Path) -> None:
    config, paths, _, games, _ = _fixture(tmp_path, entrants=2, maps=1, pairs=2)
    original = copy.deepcopy(games)
    games[2]["origin"] = copy.deepcopy(games[0]["origin"])
    _write_rows(paths["games"], games)
    with pytest.raises(ValueError, match="counted twice"):
        resolve_reuse_plan(config, paths)
    games = original
    for row in games[2:]:
        row["execution"]["seed_id"] = games[0]["execution"]["seed_id"]
    _write_rows(paths["games"], games)
    with pytest.raises(ValueError, match="bootstrap_group"):
        resolve_reuse_plan(config, paths)


def _couple(
    config: dict[str, Any],
    paths: dict[str, Path],
    manifest: dict[str, Any],
    games: list[dict[str, Any]],
) -> None:
    for row in games:
        row["execution"]["bootstrap_group"] = f"round-{row['pair_order']}"
    companion = [
        json.loads(line) for line in paths["companion"].read_text().splitlines()
    ]
    manifest["selection_blocks"] = [
        [
            canonical_json(["group", f"round-{index}"]).decode(),
            *[
                selection_unit_id(row)
                for row in companion
                if row["pair_order"] == index and row["spawn_locations"] == 0
            ],
        ]
        for index in sorted({row["pair_order"] for row in games})
    ]
    manifest["execution_groups"][0]["extension"] = "none"
    _write(paths["schedule"], manifest)
    _write_rows(paths["games"], games)


def test_coupled_prefix_keeps_complete_groups_and_equal_map_exposure(
    tmp_path: Path,
) -> None:
    config, paths, manifest, games, _ = _fixture(tmp_path, entrants=3, maps=2, pairs=3)
    _couple(config, paths, manifest, games)
    plan = resolve_reuse_plan(config, paths, games_per_opponent=8)
    assert len(plan.games) == 3 * 8
    assert {row["execution"]["bootstrap_group"] for row in plan.games} == {
        "round-0",
        "round-1",
    }
    assert set(
        Counter((row["matchup_id"], row["map_id"]) for row in plan.games).values()
    ) == {4}


def test_coupled_prefix_cannot_split_a_larger_group_to_reach_budget(
    tmp_path: Path,
) -> None:
    config, paths, manifest, games, _ = _fixture(tmp_path, entrants=3, maps=2, pairs=2)
    _couple(config, paths, manifest, games)
    manifest["selection_blocks"] = [
        [identifier for block in manifest["selection_blocks"] for identifier in block]
    ]
    _write(paths["schedule"], manifest)
    with pytest.raises(ValueError, match="No whole"):
        resolve_reuse_plan(config, paths, games_per_opponent=4)
    assert len(resolve_reuse_plan(config, paths, games_per_opponent=8).games) == 24


def test_coupled_selection_rejects_unbalanced_missing_and_repeated_groups(
    tmp_path: Path,
) -> None:
    config, paths, manifest, games, _ = _fixture(tmp_path, entrants=3, maps=1, pairs=2)
    _couple(config, paths, manifest, games)
    original = copy.deepcopy(manifest["selection_blocks"])
    manifest["selection_blocks"] = [original[0], original[0]]
    _write(paths["schedule"], manifest)
    with pytest.raises(ValueError, match="occur once"):
        resolve_reuse_plan(config, paths)
    manifest["selection_blocks"] = [original[0]]
    _write(paths["schedule"], manifest)
    with pytest.raises(ValueError, match="cover every"):
        resolve_reuse_plan(config, paths)
    games[0]["execution"]["bootstrap_group"] = "other"
    games[1]["execution"]["bootstrap_group"] = "other"
    manifest["selection_blocks"] = [
        [canonical_json(["group", "other"]).decode()],
        *original,
    ]
    _write(paths["schedule"], manifest)
    _write_rows(paths["games"], games)
    with pytest.raises(ValueError, match="equal positive"):
        resolve_reuse_plan(config, paths)


def test_source_identity_ignores_asset_aliases_and_locations_but_not_content(
    tmp_path: Path,
) -> None:
    config, _, _, _, _ = _fixture(tmp_path, entrants=2, maps=1)
    source = config["record_sources"][0]
    assets = config["assets"]
    original = record_source_identity(source, assets)
    moved = copy.deepcopy(source)
    moved["manifest_asset"] = "another-name"
    assets["another-name"] = copy.deepcopy(assets["source-manifest"])
    assets["another-name"]["path"] = "/different/path"
    assert record_source_identity(moved, assets) == original
    assets["another-name"]["sha256"] = "0" * 64
    assert record_source_identity(moved, assets) != original


def test_source_manifest_identity_tamper_is_not_accepted(tmp_path: Path) -> None:
    config, paths, _, _, _ = _fixture(tmp_path, entrants=2, maps=1)
    config["record_sources"][0]["tables"]["match_results"]["rows"] += 1
    with pytest.raises(ValueError, match="record source identity"):
        resolve_reuse_plan(config, paths)


def test_jsonl_rejects_duplicate_keys_and_nonfinite_values(tmp_path: Path) -> None:
    config, paths, _, _, _ = _fixture(tmp_path, entrants=2, maps=1)
    for data in (
        '{"logical_game_id":1,"logical_game_id":2}\n',
        '{"value":NaN}\n',
        "\n",
    ):
        paths["games"].write_text(data)
        with pytest.raises(ValueError, match="schedule row"):
            resolve_reuse_plan(config, paths)


def test_missing_companion_does_not_silently_create_new_conditions(
    tmp_path: Path,
) -> None:
    config, paths, manifest, _, _ = _fixture(tmp_path, entrants=2, maps=1)
    manifest["challenger_games_asset"] = None
    _write(paths["schedule"], manifest)
    with pytest.raises(ValueError, match="asset ID"):
        resolve_reuse_plan(config, paths, challenger_id="challenger")


def test_schedule_source_files_are_not_mutated(tmp_path: Path) -> None:
    config, paths, _, _, _ = _fixture(tmp_path, entrants=2, maps=1, pairs=1)
    before = {key: path.read_bytes() for key, path in paths.items()}
    resolve_reuse_plan(config, paths, games_per_opponent=4, rerun_existing=True)
    assert {key: path.read_bytes() for key, path in paths.items()} == before
    assert set(tmp_path.iterdir()) == set(paths.values())


def _completed_retained(
    plan: ReusePlan, config: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    challenger = plan.participant_ids[-1]
    survivors = [*plan.participant_ids[1:-1], challenger]
    new_source = copy.deepcopy(config["record_sources"][0])
    new_source["run_id"] = "admission-run"
    new_source["source_id"] = record_source_identity(new_source, config["assets"])
    retained = [
        copy.deepcopy(row)
        for row in plan.games
        if row["team_a"] in survivors and row["team_b"] in survivors
    ]
    for row in retained:
        if row["origin"] is None:
            row["origin"] = {
                "source_id": new_source["source_id"],
                "run_id": new_source["run_id"],
                "phase": "tournament",
                "pass_id": f"versus-{row['team_b']}",
                "episode_id": row["execution"]["episode_id"],
            }
    return survivors, retained, new_source


def test_promotion_rebuilds_next_companion_without_changing_66_retained_matchups(
    tmp_path: Path,
) -> None:
    config, paths, _, _, _ = _fixture(tmp_path, pairs=1)
    plan = resolve_reuse_plan(config, paths, challenger_id="promoted")
    survivors, retained, new_source = _completed_retained(plan, config)
    before = copy.deepcopy(retained)
    manifest, retained_rows, companion = rebuild_companion(
        plan,
        retained,
        survivors[::-1],
        games_asset="retained",
        challenger_games_asset="next-companion",
        map_ids=range(5),
    )
    assert retained == before == list(retained_rows)
    assert len(retained_rows) == 66 * 10
    assert len(companion) == 12 * 10
    assert manifest["allocation_order"] == survivors
    assert {row["team_a"] for row in companion} == {CHALLENGER_PLACEHOLDER}
    assert {row["team_b"] for row in companion} == set(survivors)
    old_seed_ids = {row["execution"]["seed_id"] for row in plan.games}
    assert old_seed_ids.isdisjoint({row["execution"]["seed_id"] for row in companion})
    assert min(row["logical_game_id"] for row in companion) > max(
        row["logical_game_id"] for row in plan.games
    )
    config["participants"] = [
        {"entrant_id": value, "name": value} for value in survivors[::-1]
    ]
    config["record_sources"].append(new_source)
    paths["retained"], paths["next-companion"] = (
        tmp_path / "retained.jsonl",
        tmp_path / "next.jsonl",
    )
    _write(paths["schedule"], manifest)
    _write_rows(paths["retained"], list(retained_rows))
    _write_rows(paths["next-companion"], list(companion))
    next_plan = resolve_reuse_plan(config, paths, challenger_id="next")
    assert next_plan.games[: len(retained_rows)] == retained_rows
    assert len(next_plan.games) == 78 * 10
    assert {row["team_a"] for row in next_plan.games[len(retained_rows) :]} == {"next"}


@pytest.mark.parametrize("merged_blocks", [False, True])
def test_coupled_promotion_keeps_groups_and_adds_independent_companion_units(
    tmp_path: Path,
    merged_blocks: bool,
) -> None:
    config, paths, manifest, games, _ = _fixture(tmp_path, pairs=2)
    _couple(config, paths, manifest, games)
    if merged_blocks:
        manifest["selection_blocks"] = [
            [unit for block in manifest["selection_blocks"] for unit in block]
        ]
    manifest["execution_groups"][0]["extension"] = "independent-pairs-v1"
    _write(paths["schedule"], manifest)
    plan = resolve_reuse_plan(config, paths, challenger_id="promoted")
    survivors, retained, source = _completed_retained(plan, config)
    original = copy.deepcopy(retained)
    next_manifest, retained_rows, companion = rebuild_companion(
        plan,
        retained,
        survivors,
        games_asset="retained",
        challenger_games_asset="next-companion",
        map_ids=range(5),
    )
    assert retained == original == list(retained_rows)
    assert len(retained_rows) == 66 * 20
    assert len(companion) == 12 * 20
    assert all(row["execution"]["bootstrap_group"] is None for row in companion)
    assert {row["execution"]["seed_id"] for row in plan.games}.isdisjoint(
        {row["execution"]["seed_id"] for row in companion}
    )
    for index, block in enumerate(next_manifest["selection_blocks"]):
        assert canonical_json(["group", f"round-{index}"]).decode() in block
        expected = {
            selection_unit_id(row)
            for row in [*retained_rows, *companion]
            if merged_blocks or row["pair_order"] == index
        }
        assert set(block) == expected
    config["participants"] = [
        {"entrant_id": identifier, "name": identifier} for identifier in survivors
    ]
    config["record_sources"].append(source)
    paths["retained"] = tmp_path / "retained.jsonl"
    paths["next-companion"] = tmp_path / "next-companion.jsonl"
    _write(paths["schedule"], next_manifest)
    _write_rows(paths["retained"], list(retained_rows))
    _write_rows(paths["next-companion"], list(companion))
    for challenger, expected_matchups in ((None, 66), ("next", 78)):
        budget = 20 if merged_blocks else 10
        selected = resolve_reuse_plan(
            config, paths, challenger_id=challenger, games_per_opponent=budget
        )
        assert len(selected.games) == expected_matchups * budget
        assert {row["pair_order"] for row in selected.games} == (
            {0, 1} if merged_blocks else {0}
        )
        selected_groups = {
            row["execution"]["bootstrap_group"]
            for row in selected.games
            if row["execution"]["bootstrap_group"] is not None
        }
        assert selected_groups == (
            {"round-0", "round-1"} if merged_blocks else {"round-0"}
        )
        if merged_blocks:
            with pytest.raises(ValueError, match="No whole"):
                resolve_reuse_plan(
                    config, paths, challenger_id=challenger, games_per_opponent=10
                )


def test_selection_unit_references_are_tagged_and_reject_bare_group_names(
    tmp_path: Path,
) -> None:
    config, paths, manifest, games, companion = _fixture(
        tmp_path, entrants=3, maps=1, pairs=1
    )
    _couple(config, paths, manifest, games)
    independent = copy.deepcopy(companion[0])
    independent["pair_id"] = "round-0"
    assert selection_unit_id(independent) != selection_unit_id(games[0])
    manifest["selection_blocks"][0][0] = "round-0"
    _write(paths["schedule"], manifest)
    with pytest.raises(ValueError, match="known and occur once"):
        resolve_reuse_plan(config, paths)


@pytest.mark.parametrize(
    "change",
    [
        "missing_origin",
        "changed_rng",
        "removed_game",
        "stale_counter",
        "unapproved_extension",
    ],
)
def test_promotion_does_not_publish_missing_or_changed_evidence(
    tmp_path: Path, change: str
) -> None:
    config, paths, _, _, _ = _fixture(tmp_path, entrants=3, maps=1, pairs=1)
    plan = resolve_reuse_plan(config, paths, challenger_id="promoted")
    survivors, retained, _ = _completed_retained(plan, config)
    if change == "missing_origin":
        retained[-1]["origin"] = None
    elif change == "changed_rng":
        retained[-1]["execution"]["episode_id"] += 1
    elif change == "removed_game":
        retained.pop()
    elif change == "stale_counter":
        plan.schedule_manifest["execution_groups"][0]["next_episode_id"] = 1
    else:
        plan.schedule_manifest["execution_groups"][0]["extension"] = "none"
    with pytest.raises(ValueError):
        rebuild_companion(
            plan,
            retained,
            survivors,
            games_asset="retained",
            challenger_games_asset="next",
            map_ids=(0,),
        )


@pytest.mark.parametrize(
    ("location", "value"), [("version", 1.0), ("root_seed", 47.0), ("root_seed", True)]
)
def test_integer_coordinates_never_accept_float_or_boolean_aliases(
    tmp_path: Path, location: str, value: object
) -> None:
    config, paths, manifest, games, _ = _fixture(tmp_path, entrants=2, maps=1, pairs=1)
    if location == "version":
        manifest[location] = value
        _write(paths["schedule"], manifest)
    else:
        games[0]["execution"][location] = value
        _write_rows(paths["games"], games)
    with pytest.raises(ValueError):
        resolve_reuse_plan(config, paths)


def test_each_group_preserves_its_supported_original_rng_protocol(
    tmp_path: Path,
) -> None:
    config, paths, manifest, games, _ = _fixture(tmp_path, entrants=3, maps=1, pairs=1)
    legacy = copy.deepcopy(manifest["execution_groups"][0])
    legacy.update(
        group_id="legacy",
        action_stream_version="episode-fold-in-v1",
        initialization_stream_version="episode-fold-in-v1",
    )
    manifest["execution_groups"].append(legacy)
    for row in games:
        row["execution"].update(
            group_id="legacy",
            action_stream_version="episode-fold-in-v1",
            initialization_stream_version="episode-fold-in-v1",
        )
    _write(paths["schedule"], manifest)
    _write_rows(paths["games"], games)
    plan = resolve_reuse_plan(config, paths, challenger_id="new-system")
    assert {row["execution"]["action_stream_version"] for row in plan.games} == {
        "episode-fold-in-v1",
        "evaluation-systems-v1",
    }
    for invalid in ("future-protocol", "evaluation-systems-v1"):
        legacy["initialization_stream_version"] = invalid
        _write(paths["schedule"], manifest)
        with pytest.raises(ValueError, match="supported action/init"):
            resolve_reuse_plan(config, paths)


@pytest.mark.parametrize(
    "inline_names",
    [("schedule",), ("games", "companion"), ("schedule", "games", "companion")],
)
def test_v2_inline_schedule_matches_file_plan_and_keeps_challenger_keys(
    tmp_path: Path, inline_names: tuple[str, ...]
) -> None:
    from marl_battlegrounds.evaluation.tournament_assets import AssetVerifier

    config, paths, manifest, games, companion = _fixture(
        tmp_path, entrants=3, maps=2, pairs=2, saved=False
    )
    expected = resolve_reuse_plan(config, paths, challenger_id="challenger")
    config["version"] = 2
    values = {"schedule": manifest, "games": games, "companion": companion}
    for name in inline_names:
        value = values[name]
        encoded = canonical_json(value)
        config["assets"][name] = {
            "role": "schedule",
            "sha256": sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
            "inline": value,
        }
        paths[name].unlink()
        paths.pop(name)
    verifier = AssetVerifier(config, cache_dir=tmp_path / "cache")
    actual = resolve_reuse_plan(
        config, paths, asset_reader=verifier.read_json, challenger_id="challenger"
    )
    assert actual == expected
    assert not (tmp_path / "cache").exists()
    with pytest.raises(ValueError, match="checked reader"):
        resolve_reuse_plan(config, paths, challenger_id="challenger")
    games_asset = config["assets"]["games"]
    if "inline" in games_asset:
        games_asset["inline"][0]["execution"]["seed_id"] += 1
        with pytest.raises(ValueError, match="size or digest"):
            AssetVerifier(config)


def _deferred_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    config, paths, manifest, games, companion = _fixture(
        tmp_path, entrants=3, maps=2, pairs=1
    )
    config["version"] = 2
    manifest["companion_stream_rule"] = "participant-method-v1"
    kinds = {
        row["entrant_id"]: "system" if index == 2 else "policy"
        for index, row in enumerate(config["participants"])
    }
    groups: dict[str, dict[str, Any]] = {}
    template = manifest["execution_groups"][0]
    for row in [*games, *companion]:
        first, second = row["team_a"], row["team_b"]
        group_id = (
            f"deferred:{second}"
            if first == CHALLENGER_PLACEHOLDER
            else f"incumbent:{first}:{second}"
        )
        stream = (
            "episode-fold-in-v1"
            if kinds.get(first, "policy") == kinds[second] == "policy"
            else "evaluation-systems-v1"
        )
        row["execution"].update(
            group_id=group_id,
            action_stream_version=stream,
            initialization_stream_version=stream,
        )
        groups[group_id] = {
            **template,
            "group_id": group_id,
            "action_stream_version": stream,
            "initialization_stream_version": stream,
        }
    manifest["execution_groups"] = list(groups.values())
    _write(paths["schedule"], manifest)
    _write_rows(paths["games"], games)
    _write_rows(paths["companion"], companion)
    return config, paths, kinds


def test_marked_v2_repeated_promotions_keep_old_games_and_bind_new_method_streams(
    tmp_path: Path,
) -> None:
    config, paths, kinds = _deferred_fixture(tmp_path)
    seen_seed_coordinates: set[tuple[int, int]] = set()
    seen_episode_coordinates: set[tuple[int, int]] = set()
    expected_retained = None
    for generation, (challenger, kind) in enumerate(
        (("policy-one", "policy"), ("system-two", "system"), ("policy-three", "policy"))
    ):
        kinds[challenger] = kind
        plan = resolve_reuse_plan(
            config, paths, challenger_id=challenger, participant_kinds=kinds
        )
        if expected_retained is not None:
            assert plan.games[: len(expected_retained)] == expected_retained
        for row in plan.games:
            if row["team_a"] == challenger:
                stream = (
                    "episode-fold-in-v1"
                    if kinds[challenger] == kinds[row["team_b"]] == "policy"
                    else "evaluation-systems-v1"
                )
                assert row["execution"]["action_stream_version"] == stream
                assert row["execution"]["initialization_stream_version"] == stream
            execution = row["execution"]
            seen_seed_coordinates.add((execution["root_seed"], execution["seed_id"]))
            seen_episode_coordinates.add(
                (execution["root_seed"], execution["episode_id"])
            )
        original_plan = copy.deepcopy(plan)
        survivors, retained, source = _completed_retained(plan, config)
        source["run_id"] = f"promotion-{generation}"
        source["source_id"] = record_source_identity(source, config["assets"])
        original_by_id = {row["logical_game_id"]: row for row in plan.games}
        for row in retained:
            if original_by_id[row["logical_game_id"]]["origin"] is None:
                row["origin"].update(
                    source_id=source["source_id"], run_id=source["run_id"]
                )
        original_retained = copy.deepcopy(retained)
        manifest, retained_rows, companion = rebuild_companion(
            plan,
            retained,
            survivors,
            games_asset=f"retained-{generation}",
            challenger_games_asset=f"companion-{generation}",
            map_ids=(0, 1),
        )
        assert plan == original_plan
        assert retained == original_retained == list(retained_rows)
        assert manifest["companion_stream_rule"] == "participant-method-v1"
        old_groups = {
            row["group_id"] for row in plan.schedule_manifest["execution_groups"]
        }
        companion_groups = {row["execution"]["group_id"] for row in companion}
        assert old_groups.isdisjoint(companion_groups)
        assert len(companion_groups) == len(survivors)
        for entrant in survivors:
            assert (
                len(
                    {
                        row["execution"]["group_id"]
                        for row in companion
                        if row["team_b"] == entrant
                    }
                )
                == 1
            )
        assert seen_seed_coordinates.isdisjoint(
            (row["execution"]["root_seed"], row["execution"]["seed_id"])
            for row in companion
        )
        assert seen_episode_coordinates.isdisjoint(
            (row["execution"]["root_seed"], row["execution"]["episode_id"])
            for row in companion
        )
        assert {row["execution"]["root_seed"] for row in companion} == {47}
        assert len({row["execution"]["episode_id"] for row in companion}) == len(
            companion
        )
        assert (
            len({row["execution"]["seed_id"] for row in companion})
            == len(companion) // 2
        )
        for group in manifest["execution_groups"]:
            assert group["next_episode_id"] > max(
                row["execution"]["episode_id"] for row in companion
            )
            assert group["next_seed_id"] > max(
                row["execution"]["seed_id"] for row in companion
            )
        config["participants"] = [
            {"entrant_id": identifier, "name": identifier} for identifier in survivors
        ]
        config["record_sources"].append(source)
        for name, rows in (
            (manifest["games_asset"], retained_rows),
            (manifest["challenger_games_asset"], companion),
        ):
            paths[name] = tmp_path / f"{name}.jsonl"
            _write_rows(paths[name], list(rows))
            config["assets"][name] = _asset(paths[name].read_bytes())
        _write(paths["schedule"], manifest)
        expected_retained = retained_rows


@pytest.mark.parametrize("challenger", [None, "next"])
def test_marked_v2_fresh_extension_uses_one_counter_per_root(
    tmp_path: Path, challenger: str | None
) -> None:
    config, paths, kinds = _deferred_fixture(tmp_path)
    if challenger is not None:
        kinds[challenger] = "policy"
    originals = {name: path.read_bytes() for name, path in paths.items()}
    before = resolve_reuse_plan(
        config, paths, challenger_id=challenger, participant_kinds=kinds
    )
    plan = resolve_reuse_plan(
        config,
        paths,
        challenger_id=challenger,
        participant_kinds=kinds,
        games_per_opponent=8,
        rerun_existing=True,
    )
    assert {name: path.read_bytes() for name, path in paths.items()} == originals
    assert len(plan.games) == (3 if challenger is None else 6) * 8
    by_id = {row["logical_game_id"]: row for row in plan.games}
    for original in before.games:
        fresh = by_id[original["logical_game_id"]]
        assert fresh["execution"] == original["execution"]
        assert fresh["origin"] is None
        assert fresh["prior_origin"] == original["origin"]
    added = [row for row in plan.games if row["pair_order"] == 1]
    assert len({row["execution"]["episode_id"] for row in added}) == len(added)
    assert len({row["execution"]["seed_id"] for row in added}) == len(added) // 2
    prior_companion = [
        json.loads(line) for line in paths["companion"].read_text().splitlines()
    ]
    assert min(row["execution"]["episode_id"] for row in added) > max(
        row["execution"]["episode_id"] for row in prior_companion
    )
    assert min(row["execution"]["seed_id"] for row in added) > max(
        row["execution"]["seed_id"] for row in prior_companion
    )
    for group in plan.schedule_manifest["execution_groups"]:
        assert group["next_episode_id"] > max(
            row["execution"]["episode_id"] for row in plan.games
        )
        assert group["next_seed_id"] > max(
            row["execution"]["seed_id"] for row in plan.games
        )
