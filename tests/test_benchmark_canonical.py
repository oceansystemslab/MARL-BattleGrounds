"""Check the narrow canonical cost harness without running simulated games.

The fixtures prove explicit workload limits, dynamic model parameters, original
to logical row joins, and instrumentation that restores wrapped owners. They do
not establish tournament correctness or GPU speed; packet evidence owns those.
"""

# The developer harness deliberately inspects existing private measurement owners.
# pyright: reportPrivateUsage=false
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest
from scripts.dev import benchmark_canonical as bench


def test_workload_rejects_silent_batch_or_version_changes(tmp_path: Path) -> None:
    path = tmp_path / "workload.json"
    value = {
        "format": "marlbg-canonical-benchmark",
        "version": 1,
        "batch": 32,
        "chunk": 16,
    }
    path.write_text(json.dumps(value))
    assert bench._read_workload(path) == value
    for key, changed in (("version", 2), ("batch", 2), ("chunk", 128)):
        path.write_text(json.dumps({**value, key: changed}))
        with pytest.raises(ValueError):
            bench._read_workload(path)


def test_bundle_keeps_one_actor_and_dynamic_same_shaped_weights(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"name": "First", "weight": 0.1}))
    first = bench.load_bundle({"first": path})
    path.write_text(json.dumps({"name": "Second", "weight": 0.7}))
    second = bench.load_bundle({"second": path})
    assert first.apply is second.apply
    assert first.variables.shape == second.variables.shape == ()
    assert first.variables.dtype == second.variables.dtype == np.float32
    assert float(first.variables) != float(second.variables)
    with pytest.raises(ValueError, match="exactly one"):
        bench.load_bundle({})


def test_result_join_uses_complete_origin_not_repeated_episode_id() -> None:
    games = [
        {
            "logical_game_id": 8,
            "origin": {
                "run_id": "a",
                "phase": "tournament",
                "pass_id": "p",
                "episode_id": 1,
            },
        },
        {
            "logical_game_id": 4,
            "origin": {
                "run_id": "b",
                "phase": "tournament",
                "pass_id": "p",
                "episode_id": 1,
            },
        },
    ]

    def origin(game: dict[str, Any]) -> dict[str, Any]:
        return game["origin"]

    records = SimpleNamespace(games=games, origin=origin)
    columns = {
        "run_id": ("a", "b"),
        "phase": ("tournament", "tournament"),
        "pass_id": ("p", "p"),
        "episode_id": (1, 1),
        "outcome": (1, 2),
        "episode_length": (4, 7),
        "team_a_score": (2, 0),
        "team_b_score": (0, 1),
    }

    def iter_table(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], ...]:
        return (columns,)

    result = SimpleNamespace(_record_access=records, iter_table=iter_table)
    rows = bench._result_rows(result)
    assert [row["episode_id"] for row in rows] == [4, 8]
    assert [row["outcome"] for row in rows] == [2, 1]
    assert columns["episode_id"] == (1, 1)


def test_attribution_restores_get_and_tracks_only_explicit_payload() -> None:
    original = jax.device_get
    data = jax.device_put(np.arange(5, dtype=np.int32))

    def run(_probe: bool) -> int:
        return int(jax.device_get(data).sum())

    value, report = bench._attribute(run)
    assert value == 10
    assert jax.device_get is original
    assert report["device_get_calls"] == 1
    assert report["logical_device_get_bytes"] == 20
    assert report["model_loads"] == 0


def test_index_size_estimate_counts_shared_objects_once() -> None:
    shared = {("run", "phase", "pass", 1): (20, 80)}
    assert bench._deep_bytes([shared, shared]) < bench._deep_bytes(
        [shared, {("other", "phase", "pass", 1): (20, 80)}]
    )
