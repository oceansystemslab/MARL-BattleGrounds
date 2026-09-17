"""Build complete artificial tournament records without playing any games.

build_record_bundle writes a format-1 custom config, actual map configurations,
complete spawn-pair schedules and schema-14 outcome/measurement CSV files. It
uses real frozen Policy registrations but invents outcomes and measurements.
These files test recording joins and admission machinery; they are not research
results, qualified controllers or an official release. Full reports are optional
and streamed a row at a time. No action method or statistical fit is called.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np

from marl_battlegrounds.evaluation.evaluation_conditions import config_record
from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
)
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
    policy,
    shared_policy,
)
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.evaluation.run_writer import IDENTITY_COLUMNS, MATCH_COLUMNS
from marl_battlegrounds.evaluation.tournament_assets import (
    AssetVerifier,
    controller_content_identity,
    environment_source_manifest,
    inference_dependency_lock,
)
from marl_battlegrounds.evaluation.tournament_config import (
    canonical_json,
    load_tournament_config,
    snapshot_identity,
)
from marl_battlegrounds.evaluation.tournament_reuse import (
    CHALLENGER_PLACEHOLDER,
    record_source_identity,
)
from marl_battlegrounds.tasks import (
    balanced_spawn_configs,
    canonical_tournament_rosters,
    make_standard_team_deathmatch_config,
)


def config_descriptor(
    *, entrants: int = 2, official: bool = False, root: Path | None = None
) -> dict[str, Any]:
    """Build an incomplete artificial descriptor; this function writes no files.

    entrants selects the field size. official=True supplies structural release
    fields only and never installs or qualifies a bundle. root names the future
    asset directory; None uses an intentionally unavailable fixture path. The
    complete builder replaces every dummy asset before returning a usable config.
    """
    base = Path("/fixture") if root is None else root.absolute()
    digest = sha256(b"fixture only").hexdigest()
    names = ("source", "config", "schedule", "registration", "dependencies", "manifest")
    roles = (
        "qualification",
        "configuration",
        "schedule",
        "registration",
        "dependencies",
        "run_manifest",
    )
    assets = {
        name: {
            "sha256": digest,
            "size_bytes": 0,
            "role": role,
            "path": str(base / name),
            "url": None,
        }
        for name, role in zip(names, roles, strict=True)
    }
    source_id = sha256(b"fixture source declaration").hexdigest()
    participants = []
    for index in range(entrants):
        name = f"fixture-{index}"
        participants.append(
            {
                "entrant_id": name,
                "name": name,
                "controller_id": sha256(name.encode()).hexdigest(),
                "controller": {
                    "kind": "builtin",
                    "name": "random",
                    "content": {
                        "code": None,
                        "parameters": None,
                        "memory_template": None,
                        "input_preparation": None,
                        "decision_settings": None,
                        "adapter_bindings": [],
                        "external_state": None,
                    },
                },
                "registration_asset": "registration",
                "elo": 1200.0 if official else None,
                "result_ref": {
                    "source_id": source_id,
                    "table": "tournament_results",
                    "policy": name,
                }
                if official
                else None,
            }
        )
    maps = []
    for map_id in range(47, 52) if official else (0,):
        maps.append(
            {
                "map_id": map_id,
                "split": "test" if official else None,
                "source_config_asset": "config",
                "source_config_id": digest,
                "registered_map": {
                    "map_id": map_id,
                    "name": f"Fixture map {map_id}",
                    "split": "test",
                    "curriculum": False,
                    "source": {
                        "asset_id": f"fixture-map-{map_id}",
                        "revision": 1,
                        "source_path": "fixture",
                        "source_sha256": digest,
                        "semantic_digest": digest,
                    },
                    "resource_sha256": digest,
                }
                if official
                else None,
            }
        )
    result: dict[str, Any] = {
        "format": "marlbg-tournament-config",
        "version": 1,
        "snapshot_id": digest,
        "release": {
            "release_at_utc": "2026-11-01T00:00:00+00:00",
            "release_at_local": "2026-11-01T00:00:00+00:00",
            "cutoff_at_utc": "2026-10-29T00:00:00+00:00",
            "cutoff_at_local": "2026-10-29T00:00:00+00:00",
            "rules_id": "fixture-rules",
            "approval_id": "fixture-only",
        }
        if official
        else None,
        "participants": participants,
        "conditions": {
            "task": "tdm",
            "map_sources": maps,
            "rosters": {
                "team_a": ["mage", "warrior", "hunter", "rogue", "priest"],
                "team_b": ["mage", "warrior", "hunter", "rogue", "priest"],
            },
            "information_mode": "SharedObs",
            "games_per_opponent": 10,
            "score_threshold": 20,
            "max_steps": 300,
            "memory_rule": "fresh-per-game",
            "pairing_protocol": "fixed-team-spawn-v1",
            "schedule_asset": "schedule",
        },
        "analysis": {
            "method_id": "centered-davidson-v1",
            "bootstrap_replicates": 5000,
            "bootstrap_seed": 0,
            "opponent_weights": None,
        },
        "compatibility": {
            "package_version": "0.0.0",
            "code_revision": {"package_version": "0.0.0"},
            "environment_id": digest,
            "source_manifest_asset": "source",
            "scalar_schema": 14,
            "run_schema": 2,
            "replay_schema": 3,
            "action_stream_version": "evaluation-systems-v1",
            "initialization_stream_version": "evaluation-systems-v1",
            "dependency_lock_asset": "dependencies",
        },
        "assets": assets,
        "record_sources": [
            {
                "source_id": source_id,
                "run_id": "fixture-source",
                "manifest_asset": "manifest",
                "tables": {},
                "replays": [],
            }
        ]
        if official
        else [],
    }
    result["snapshot_id"] = snapshot_identity(result)
    return result


def fixture_policy(index: int) -> Policy:
    """Return a random Policy with a distinct scalar int32 example version.

    index is an illustrative version, not a learned parameter or result. It
    changes numerical identity without making any action or loading a model.
    """
    return replace(
        policy("random"),
        name=f"fixture-{index:02d}",
        variables={"fixture_version": np.asarray(index, dtype=np.int32)},
    )


def _factory(index: int) -> Callable[[], System]:
    """Bind an example version to one installed module-level factory name."""

    def create() -> System:
        """Return a fresh shared Policy adapter for this artificial version."""
        return shared_policy(fixture_policy(index))

    return create


for _index in range(32):
    globals()[f"load_fixture_{_index}"] = _factory(_index)


def _asset(
    directory: Path,
    assets: dict[str, Any],
    identifier: str,
    value: object,
    role: str,
    *,
    raw: bool = False,
) -> Path:
    """Write one local fixture asset and add its verified byte declaration.

    directory and identifier select the file. value is finite JSON unless raw
    is True, when it is already bytes. assets is updated in place. These files
    are explicit example outputs, never installed official assets.
    """
    payload = value if raw else canonical_json(value)
    assert isinstance(payload, bytes)
    path = directory / identifier
    path.write_bytes(payload)
    assets[identifier] = {
        "sha256": sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "role": role,
        "path": str(path),
        "url": None,
    }
    return path


def _participants(
    directory: Path, assets: dict[str, Any], count: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Create count numerical versions and their name-independent evidence.

    Return ordered participant descriptors, registrations by entrant, and the
    original writer system table. Every asset and qualification claim is clearly
    labelled as an artificial fixture. No policy action runs.
    """
    participants: list[dict[str, Any]] = []
    registrations: dict[str, dict[str, Any]] = {}
    systems: dict[str, Any] = {}
    for index in range(count):
        method = shared_policy(fixture_policy(index))
        registration_id, registration = normalize_system_registration(
            method, phase="tournament", frozen=True
        )
        name = method.name
        registrations[name] = dict(registration)
        systems[registration_id] = registration
        registration_asset = f"registration-{index}.json"
        _asset(directory, assets, registration_asset, registration, "registration")
        hooks = cast(dict[str, dict[str, Any]], registration["hooks"])
        evidence: dict[str, Any] = {
            "code": {
                "code_digest": hooks["apply"]["code_digest"],
                "fixture_only": True,
            },
            "parameters": {"digest": registration["variables_digest"]},
            "memory_template": {
                "digest": cast(list[dict[str, Any]], registration["adapter_policies"])[
                    0
                ]["initial_carry_digest"]
            },
            "input_preparation": {"mode": "SharedObs", "fixture_only": True},
            "decision_settings": {"execution": "jax", "fixture_only": True},
            "external_state": {"state": "none", "fixture_only": True},
        }
        content: dict[str, Any] = {"adapter_bindings": []}
        for field, value in evidence.items():
            identifier = f"controller-{index}-{field}.json"
            _asset(directory, assets, identifier, value, "registration")
            content[field] = {
                "asset_id": identifier,
                "sha256": assets[identifier]["sha256"],
            }
        controller = {
            "kind": "factory",
            "factory": f"canonical_fixture:load_fixture_{index}",
            "content": content,
        }
        identifier, known = controller_content_identity(
            controller, AssetVerifier({"assets": assets})
        )
        assert known
        participants.append(
            {
                "entrant_id": name,
                "name": name,
                "controller_id": identifier,
                "controller": controller,
                "registration_asset": registration_asset,
                "elo": None,
                "result_ref": None,
            }
        )
    return participants, registrations, systems


def _configs(
    directory: Path, assets: dict[str, Any], count: int, max_steps: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, tuple[str, str]]]:
    """Write count real source configs and resolve both complete spawn choices.

    max_steps is the positive example horizon. Return map declarations, exact
    serialized source/swapped configs and their IDs. Assets use packaged maps;
    synthetic outcomes do not become measurements of those maps.
    """
    rosters = canonical_tournament_rosters()
    maps: list[dict[str, Any]] = []
    configurations: dict[str, Any] = {}
    choices: dict[int, tuple[str, str]] = {}
    for map_id in range(47, 47 + count):
        source = make_standard_team_deathmatch_config(
            map_id=map_id,
            team_a_roster=rosters[0],
            team_b_roster=rosters[1],
            max_steps=max_steps,
        )
        batch = balanced_spawn_configs(source, num_envs=2)
        source_id, source_content = config_record(source)

        def swapped_row(value: jax.Array) -> jax.Array:
            """Select the swapped scalar leaf from the shared two-choice batch."""
            return value[1]

        swapped_id, swapped_content = config_record(jax.tree.map(swapped_row, batch))
        configurations[source_id] = source_content
        configurations[swapped_id] = swapped_content
        choices[map_id] = source_id, swapped_id
        identifier = f"config-{map_id}.json"
        _asset(directory, assets, identifier, source_content, "configuration")
        maps.append(
            {
                "map_id": map_id,
                "split": "test",
                "source_config_asset": identifier,
                "source_config_id": source_id,
                "registered_map": None,
            }
        )
    return maps, configurations, choices


def _write_table(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Stream one artificial CSV and return its content/header/size evidence.

    columns gives exact order. rows holds narrow outcomes; absent full-only cells
    receive the fixture value zero. No wide rows are retained after writing.
    """
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row.get(name, 0) for name in columns)
    with path.open("rb") as stream:
        header = stream.readline()
        digest = sha256(header)
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
        "header_sha256": sha256(header).hexdigest(),
        "rows": len(rows),
    }


def build_record_bundle(
    directory: Path,
    *,
    entrants: int = 12,
    full: bool = False,
    maps: int = 5,
    pairs_per_map: int = 1,
    max_steps: int = 16,
) -> dict[str, Any]:
    """Create a complete artificial local tournament bundle without playing games.

    Parameters
    ----------
    directory : Path
        Explicit fixture output directory, created when absent. Use a temporary
        directory for tests. Existing fixture filenames are replaced.
    entrants : int
        Population size from 2 to 32; default 12. Versions are random adapters
        with distinct scalar numerical evidence, not official controllers.
    full : bool
        False writes outcomes and priority values. True also streams schema-14
        full rows. All measurements are invented fixture values.
    maps : int
        Number of packaged test maps from 1 to 5; default 5.
    pairs_per_map : int
        Positive complete spawn pairs per matchup/map; default 1.
    max_steps : int
        Positive source horizon, default 16. No game is run by this helper.

    Returns
    -------
    dict
        config, config_path, manifest, matches, registrations, games, companion
        and verified local paths. Pass config to the custom tournament route.
        Only an isolated test catalog may pin it as a canonical fixture.

    Notes
    -----
    This writes explicit artificial evidence, never a scientifically qualified
    release. It is an example support tool, not part of the researcher API.
    """
    if (
        not 2 <= entrants <= 32
        or not 1 <= maps <= 5
        or pairs_per_map < 1
        or type(max_steps) is not int
        or max_steps <= 0
    ):
        raise ValueError("Fixture dimensions exceed supported small host examples")
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()
    assets: dict[str, Any] = {}
    participants, registrations, systems = _participants(directory, assets, entrants)
    sources, configurations, choices = _configs(directory, assets, maps, max_steps)
    names = [row["entrant_id"] for row in participants]
    registration_ids = {value["name"]: key for key, value in systems.items()}
    games: list[dict[str, Any]] = []
    companion: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    passes: dict[str, Any] = {}
    run_id, phase, root_seed = "fixture-original-run", "tournament", 471
    stream_version = "evaluation-systems-v1"
    logical_id, pair_id = 1, 1
    for team_a, team_b in (
        *combinations(names, 2),
        *((CHALLENGER_PLACEHOLDER, name) for name in names),
    ):
        template = team_a == CHALLENGER_PLACEHOLDER
        target = companion if template else games
        pass_id = f"matchup-{team_a}-{team_b}"
        entry: dict[str, Any] = {
            "phase": phase,
            "pass_id": pass_id,
            "metrics": "full" if full else "priority",
            "system_ids": {}
            if template
            else {
                "team_a": registration_ids[team_a],
                "team_b": registration_ids[team_b],
            },
            "details": {"seed": root_seed, "rng_protocol": stream_version},
            "episodes": {},
            "completed_episode_ids": [],
            "recorded_metrics_by_episode": {},
            "replays": {},
        }
        for map_id, (source_id, swapped_id) in choices.items():
            for order in range(pairs_per_map):
                for spawn in (0, 1):
                    episode_id = logical_id + 1000
                    resolved_id = source_id if spawn == 0 else swapped_id
                    game = {
                        "logical_game_id": logical_id,
                        "matchup_id": pass_id,
                        "pair_id": f"pair-{pair_id}",
                        "pair_order": order,
                        "team_a": team_a,
                        "team_b": team_b,
                        "map_id": map_id,
                        "source_config_id": source_id,
                        "resolved_config_id": resolved_id,
                        "spawn_locations": spawn,
                        "execution": {
                            "group_id": "fixture-group",
                            "root_seed": root_seed,
                            "seed_id": pair_id,
                            "episode_id": episode_id,
                            "action_stream_version": stream_version,
                            "initialization_stream_version": stream_version,
                            "bootstrap_group": None,
                        },
                        "origin": None,
                        "prior_origin": None,
                    }
                    target.append(game)
                    if not template:
                        game["origin"] = {
                            "source_id": "pending",
                            "run_id": run_id,
                            "phase": phase,
                            "pass_id": pass_id,
                            "episode_id": episode_id,
                        }
                        outcome = 1 + pair_id % 3
                        a_score, b_score = ((3, 1), (1, 3), (2, 2))[outcome - 1]
                        row = {
                            "run_id": run_id,
                            "phase": phase,
                            "pass_id": pass_id,
                            "episode_id": episode_id,
                            "seed_id": pair_id,
                            "map_id": map_id,
                            "config_id": resolved_id,
                            "team_a_policy": team_a,
                            "team_b_policy": team_b,
                            "checkpoint_id": None,
                            "block_id": pair_id,
                            "bootstrap_group": None,
                            "outcome": outcome,
                            "episode_length": 16,
                            "team_a_win": int(outcome == 1),
                            "team_a_draw": int(outcome == 3),
                            "team_a_loss": int(outcome == 2),
                            "team_b_win": int(outcome == 2),
                            "team_b_draw": int(outcome == 3),
                            "team_b_loss": int(outcome == 1),
                            "team_a_return": float(a_score - b_score),
                            "team_b_return": float(b_score - a_score),
                            "team_a_score": a_score,
                            "team_b_score": b_score,
                            "score_difference": a_score - b_score,
                            "team_a_kills": a_score,
                            "team_b_kills": b_score,
                            "team_a_deaths": b_score,
                            "team_b_deaths": a_score,
                        }
                        profile = configurations[resolved_id]["agent_profile"]
                        for slot in range(10):
                            row[f"agent_{slot}_class_id"] = profile["class_ids"][slot]
                            row[f"agent_{slot}_active"] = int(
                                profile["active_mask"][slot]
                            )
                        matches.append(row)
                        entry["episodes"][str(episode_id)] = {
                            "episode_id": episode_id,
                            "configuration_digest": resolved_id,
                            "source_config_id": source_id,
                            "spawn_locations": spawn,
                            "comparison_kind": "verified_spawn_pair",
                            "block_id": pair_id,
                            "seed_id": pair_id,
                            "map_id": map_id,
                            "initial_state_digest": None,
                        }
                        entry["completed_episode_ids"].append(episode_id)
                        entry["recorded_metrics_by_episode"][str(episode_id)] = (
                            "full" if full else "priority"
                        )
                    logical_id += 1
                pair_id += 1
        if not template:
            passes[json.dumps((phase, pass_id), separators=(",", ":"))] = entry
    tables: dict[str, Any] = {}
    manifest_tables: dict[str, Any] = {}
    selected_tables = [("match_results", MATCH_COLUMNS, "outcomes_priority")]
    if full:
        selected_tables.append(
            ("full_metrics", (*IDENTITY_COLUMNS, *FULL_METRIC_NAMES), "full_report")
        )
    for name, columns, role in selected_tables:
        identifier = name + ".csv"
        path = directory / identifier
        info = _write_table(path, columns, matches)
        assets[identifier] = {
            "sha256": info["sha256"],
            "size_bytes": info["size_bytes"],
            "role": role,
            "path": str(path),
            "url": None,
        }
        tables[name] = {
            "asset_id": identifier,
            "committed_bytes": info["size_bytes"],
            "rows": info["rows"],
            "header_sha256": info["header_sha256"],
        }
        manifest_tables[identifier] = {
            "durable_bytes": info["size_bytes"],
            "rows": info["rows"],
        }
    # These point values are artificial ordering fixtures, not fitted ratings.
    # Public analysis always fits the complete underlying game records afresh.
    rating_path = directory / "tournament_results.csv"
    rating_info = _write_table(
        rating_path,
        ("policy", "elo"),
        [{"policy": name, "elo": 1212.0 - index} for index, name in enumerate(names)],
    )
    assets["tournament_results.csv"] = {
        "sha256": rating_info["sha256"],
        "size_bytes": rating_info["size_bytes"],
        "role": "outcomes_priority",
        "path": str(rating_path),
        "url": None,
    }
    tables["tournament_results"] = {
        "asset_id": "tournament_results.csv",
        "committed_bytes": rating_info["size_bytes"],
        "rows": rating_info["rows"],
        "header_sha256": rating_info["header_sha256"],
    }
    manifest_tables["tournament_results.csv"] = {
        "durable_bytes": rating_info["size_bytes"],
        "rows": rating_info["rows"],
    }
    manifest = {
        "run_id": run_id,
        "schema_version": 2,
        "metric_schema_version": 14,
        "metric_schema_id": "marlbg.tdm.scalar",
        "systems": systems,
        "configurations": configurations,
        "tables": manifest_tables,
        "passes": passes,
    }
    _asset(directory, assets, "run_details.json", manifest, "run_manifest")
    source: dict[str, Any] = {
        "run_id": run_id,
        "manifest_asset": "run_details.json",
        "tables": tables,
        "replays": [],
    }
    source["source_id"] = record_source_identity(source, assets)
    for game in games:
        game["origin"]["source_id"] = source["source_id"]
    for identifier, selected in (
        ("games.jsonl", games),
        ("companion.jsonl", companion),
    ):
        _asset(
            directory,
            assets,
            identifier,
            b"".join(canonical_json(game) + b"\n" for game in selected),
            "schedule",
            raw=True,
        )
    schedule = {
        "format": "marlbg-tournament-schedule",
        "version": 1,
        "games_asset": "games.jsonl",
        "challenger_games_asset": "companion.jsonl",
        "execution_groups": [
            {
                "group_id": "fixture-group",
                "root_seed": root_seed,
                "action_stream_version": stream_version,
                "initialization_stream_version": stream_version,
                "next_episode_id": 1000 + logical_id,
                "next_seed_id": pair_id,
                "extension": "independent-pairs-v1",
            }
        ],
        "selection_blocks": None,
        "allocation_order": names,
    }
    _asset(directory, assets, "schedule.json", schedule, "schedule")
    source_manifest = environment_source_manifest()
    _asset(directory, assets, "fixture-source.json", source_manifest, "qualification")
    _asset(
        directory,
        assets,
        "fixture-dependencies.json",
        inference_dependency_lock(),
        "dependencies",
    )
    config = config_descriptor(entrants=entrants, root=directory)
    config.update(assets=assets, participants=participants, record_sources=[source])
    config["conditions"].update(
        map_sources=sources,
        games_per_opponent=2 * maps * pairs_per_map,
        max_steps=max_steps,
        schedule_asset="schedule.json",
    )
    config["compatibility"].update(
        environment_id=sha256(canonical_json(source_manifest)).hexdigest(),
        source_manifest_asset="fixture-source.json",
        dependency_lock_asset="fixture-dependencies.json",
        action_stream_version=stream_version,
        initialization_stream_version=stream_version,
    )
    config["snapshot_id"] = snapshot_identity(config)
    config = load_tournament_config(config, official=False)
    config_path = directory / "fixture-tournament.json"
    config_path.write_bytes(canonical_json(config))
    return {
        "config": config,
        "config_path": config_path,
        "manifest": manifest,
        "matches": matches,
        "registrations": registrations,
        "games": games,
        "companion": companion,
        "paths": {key: Path(value["path"]) for key, value in assets.items()},
    }
