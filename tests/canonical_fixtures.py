"""Build small, clearly artificial tournament descriptions for host tests.

These declarations contain no qualified official controllers or real game records.
Tests that exercise assets, schedules or execution must replace the dummy content
with their own checked evidence. Merely constructing a fixture writes no files.
"""

from hashlib import sha256
from pathlib import Path
from typing import Any

from marl_battlegrounds.evaluation.tournament_config import snapshot_identity


def config_descriptor(
    *, entrants: int = 2, official: bool = False, root: Path | None = None
) -> dict[str, Any]:
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
