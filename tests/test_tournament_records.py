"""Check one logical view over local and immutable foreign tournament game rows.

Full original identities prevent reused episode collisions. Required outcomes,
optional coverage and exact source files are checked without model imports or
copying foreign reports into the new run. The snapshot's pinned scalar schema
(current when the config has no compatibility section) picks the full-report
header and the scalar version every record source must have.
"""

import csv
import hashlib
import io
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES_BY_SCHEMA_VERSION,
    METRIC_SCHEMA_VERSION,
    PRIORITY_METRIC_NAMES,
)
from marl_battlegrounds.evaluation.run_writer import MATCH_COLUMNS
from marl_battlegrounds.evaluation.tournament_assets import AssetVerifier
from marl_battlegrounds.evaluation.tournament_records import TournamentRecords


def _asset(path: Path, role: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "role": role,
        "path": str(path),
        "url": None,
    }


def _records(
    tmp_path: Path, *, scalar: int = METRIC_SCHEMA_VERSION, pin: int | None = None
) -> tuple[TournamentRecords, dict[str, Any]]:
    assets: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    for i in range(2):
        run_id = f"source-{i}"
        row = dict.fromkeys(MATCH_COLUMNS, None)
        row.update(
            run_id=run_id,
            phase="tournament",
            pass_id="same",
            episode_id=1,
            seed_id=2,
            map_id=47,
            config_id="config",
            team_a_policy="A",
            team_b_policy="B",
            block_id=0,
            outcome=3,
        )
        row.update(dict.fromkeys(PRIORITY_METRIC_NAMES, 0))
        row.update(episode_length=2**24 + 1, team_a_score=4, team_b_score=4)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=MATCH_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
        payload = stream.getvalue().encode()
        table = tmp_path / f"table-{i}"
        table.write_bytes(payload)
        entry: dict[str, Any] = {
            "phase": "tournament",
            "pass_id": "same",
            "episodes": {"1": {}},
            "completed_episode_ids": [1],
            "recorded_metrics_by_episode": {"1": "priority"},
        }
        manifest: dict[str, Any] = {
            "run_id": run_id,
            "schema_version": 2,
            "metric_schema_version": scalar,
            "metric_schema_id": "marlbg.tdm.scalar",
            "passes": {"p": entry},
            "configurations": {"config": {}},
            "tables": {"match_results.csv": {"durable_bytes": len(payload), "rows": 1}},
        }
        path = tmp_path / f"manifest-{i}"
        path.write_text(json.dumps(manifest))
        assets[f"table-{i}"] = _asset(table, "outcomes_priority")
        assets[f"manifest-{i}"] = _asset(path, "run_manifest")
        source: dict[str, Any] = {
            "source_id": run_id,
            "run_id": run_id,
            "manifest_asset": f"manifest-{i}",
            "tables": {
                "match_results": {
                    "asset_id": f"table-{i}",
                    "committed_bytes": len(payload),
                    "rows": 1,
                    "header_sha256": hashlib.sha256(
                        payload.splitlines(keepends=True)[0]
                    ).hexdigest(),
                }
            },
            "replays": [],
        }
        sources.append(source)
        games.append(
            {
                "logical_game_id": i + 1,
                "origin": {
                    "source_id": run_id,
                    "run_id": run_id,
                    "phase": "tournament",
                    "pass_id": "same",
                    "episode_id": 1,
                },
            }
        )
    config: dict[str, Any] = {"assets": assets, "record_sources": sources}
    if pin is not None:
        config["compatibility"] = {"scalar_schema": pin}
    records = TournamentRecords(
        config,
        games,
        (),
        AssetVerifier(config),
        manifest={"run_id": "new", "passes": {}},
    )
    return records, config


def test_foreign_rows_keep_original_ownership_and_requested_order(
    tmp_path: Path,
) -> None:
    records, _ = _records(tmp_path)
    records.require_coverage(metrics="priority")
    batches = list(records.iter_rows("match_results.csv", game_ids=[2, 1], rows=1))
    assert [row["run_id"] for batch in batches for row in batch] == [
        "source-1",
        "source-0",
    ]
    assert all(row["episode_id"] == 1 for batch in batches for row in batch)
    assert batches[0][0]["episode_length"] == 2**24 + 1
    assert list(records.iter_rows("full_metrics.csv")) == []
    with pytest.raises(ValueError, match="full measurements"):
        records.require_coverage(metrics="none", full_ids=[1])
    with pytest.raises(ValueError, match="requested replay"):
        records.require_coverage(metrics="none", replay_ids=[2])


def test_source_content_change_fails_truthfully(tmp_path: Path) -> None:
    records, config = _records(tmp_path)
    assert len(next(records.iter_rows("match_results.csv"))) == 2
    path = Path(config["assets"]["table-1"]["path"])
    path.write_bytes(path.read_bytes().replace(b",4,4,", b",5,4,"))
    # Alter a byte even if the optional pattern is absent in this schema order.
    payload = bytearray(path.read_bytes())
    payload[-2] = ord("9") if payload[-2] != ord("9") else ord("8")
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="changed or is corrupt"):
        list(records.iter_rows("match_results.csv"))


def test_logical_duplicate_does_not_double_count_original_game(tmp_path: Path) -> None:
    records, config = _records(tmp_path)
    duplicate = {**records.games[0], "logical_game_id": 3}
    with pytest.raises(ValueError, match="more than once"):
        TournamentRecords(
            config,
            [*records.games, duplicate],
            (),
            AssetVerifier(config),
            manifest={"run_id": "new", "passes": {}},
        )


def test_none_outcome_read_does_not_convert_optional_measurements(
    tmp_path: Path,
) -> None:
    records, config = _records(tmp_path)
    path = Path(config["assets"]["table-0"]["path"])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["team_a_return"] = "not-a-measurement"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MATCH_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = path.read_bytes()
    config["assets"]["table-0"] = _asset(path, "outcomes_priority")
    source = config["record_sources"][0]
    source["tables"]["match_results"]["committed_bytes"] = len(payload)
    manifest_path = Path(config["assets"]["manifest-0"]["path"])
    manifest = json.loads(manifest_path.read_bytes())
    manifest["tables"]["match_results.csv"]["durable_bytes"] = len(payload)
    manifest["passes"]["p"]["recorded_metrics_by_episode"]["1"] = "none"
    manifest_path.write_text(json.dumps(manifest))
    config["assets"]["manifest-0"] = _asset(manifest_path, "run_manifest")
    fresh = TournamentRecords(
        config,
        records.games,
        (),
        AssetVerifier(config),
        manifest={"run_id": "new", "passes": {}},
    )
    fresh.require_coverage(metrics="none")
    values = list(fresh.iter_rows("match_results.csv", columns=["outcome"]))
    assert values == [({"outcome": 3}, {"outcome": 3})]
    with pytest.raises(ValueError, match="priority measurements"):
        fresh.require_coverage(metrics="priority")


@pytest.mark.parametrize(
    "field", ("outcome", "episode_length", "team_a_score", "team_b_return")
)
def test_missing_required_cell_rejected_in_preflight(
    tmp_path: Path, field: str
) -> None:
    records, config = _records(tmp_path)
    path = Path(config["assets"]["table-0"]["path"])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0][field] = ""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MATCH_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    size = path.stat().st_size
    config["assets"]["table-0"] = _asset(path, "outcomes_priority")
    config["record_sources"][0]["tables"]["match_results"]["committed_bytes"] = size
    manifest_path = Path(config["assets"]["manifest-0"]["path"])
    manifest = json.loads(manifest_path.read_bytes())
    manifest["tables"]["match_results.csv"]["durable_bytes"] = size
    manifest_path.write_text(json.dumps(manifest))
    config["assets"]["manifest-0"] = _asset(manifest_path, "run_manifest")
    fresh = TournamentRecords(
        config,
        records.games,
        (),
        AssetVerifier(config),
        manifest={"run_id": "new", "passes": {}},
    )
    with pytest.raises(ValueError, match=r"requires|missing a required"):
        fresh.require_coverage(metrics="priority")


def test_claimed_full_without_actual_row_fails_before_execution(tmp_path: Path) -> None:
    from marl_battlegrounds.evaluation.metric_catalog import FULL_METRIC_NAMES
    from marl_battlegrounds.evaluation.run_writer import IDENTITY_COLUMNS

    records, config = _records(tmp_path)
    full = tmp_path / "full-table"
    with full.open("w", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerow(
            (*IDENTITY_COLUMNS, *FULL_METRIC_NAMES)
        )
    config["assets"]["full-0"] = _asset(full, "full_report")
    config["record_sources"][0]["tables"]["full_metrics"] = {
        "asset_id": "full-0",
        "rows": 0,
        "committed_bytes": full.stat().st_size,
        "header_sha256": hashlib.sha256(full.read_bytes()).hexdigest(),
    }
    manifest_path = Path(config["assets"]["manifest-0"]["path"])
    manifest = json.loads(manifest_path.read_bytes())
    manifest["tables"]["full_metrics.csv"] = {
        "durable_bytes": full.stat().st_size,
        "rows": 0,
    }
    manifest["passes"]["p"]["recorded_metrics_by_episode"]["1"] = "full"
    manifest_path.write_text(json.dumps(manifest))
    config["assets"]["manifest-0"] = _asset(manifest_path, "run_manifest")
    fresh = TournamentRecords(
        config,
        records.games,
        (),
        AssetVerifier(config),
        manifest={"run_id": "new", "passes": {}},
    )
    with pytest.raises(ValueError, match="missing selected origins"):
        fresh.require_coverage(metrics="none", full_ids=[1])


def test_bounded_read_rechecks_sources_before_yield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation.scalar_reports import (
        IndexedScalarTable,
        OriginKey,
        ScalarCell,
    )

    records, config = _records(tmp_path)
    original = IndexedScalarTable.iter_rows
    changed = False

    def mutate_after_read(
        self: IndexedScalarTable,
        origins: Sequence[OriginKey],
        *,
        batch_size: int = 128,
        columns: Sequence[str] | None = None,
    ) -> Iterator[tuple[dict[str, ScalarCell], ...]]:
        nonlocal changed
        for batch in original(self, origins, batch_size=batch_size, columns=columns):
            if not changed:
                path = Path(config["assets"]["manifest-0"]["path"])
                path.write_bytes(path.read_bytes() + b" ")
                changed = True
            yield batch

    monkeypatch.setattr(IndexedScalarTable, "iter_rows", mutate_after_read)
    with pytest.raises(ValueError, match="changed during verification"):
        next(records.iter_rows("match_results.csv", rows=2))


def test_preflight_verifies_each_asset_at_both_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records, _ = _records(tmp_path)
    calls: dict[str, int] = {}
    original = records.verifier._candidate  # pyright: ignore[reportPrivateUsage]

    def count(identifier: str) -> Path | None:
        calls[identifier] = calls.get(identifier, 0) + 1
        return original(identifier)

    monkeypatch.setattr(records.verifier, "_candidate", count)
    records.require_coverage(metrics="priority")
    assert calls == {
        "manifest-0": 2,
        "manifest-1": 2,
        "table-0": 2,
        "table-1": 2,
    }


def test_snapshot_pin_picks_full_header_and_required_source_version(
    tmp_path: Path,
) -> None:
    for name in ("old", "mixed"):
        (tmp_path / name).mkdir()
    records, config = _records(tmp_path / "old", scalar=14, pin=14)
    assert records.scalar_schema == 14
    header = records.headers["full_metrics.csv"]
    assert header[len(header) - 11148 :] == FULL_METRIC_NAMES_BY_SCHEMA_VERSION[14]
    assert not any("red_zone" in column for column in header)
    assert len(next(records.iter_rows("match_results.csv"))) == 2
    current, _ = _records(tmp_path / "mixed", scalar=14, pin=METRIC_SCHEMA_VERSION)
    assert (
        current.headers["full_metrics.csv"][-11192:]
        == (FULL_METRIC_NAMES_BY_SCHEMA_VERSION[METRIC_SCHEMA_VERSION])
    )
    with pytest.raises(ValueError, match="schema"):
        next(current.iter_rows("match_results.csv"))
    with pytest.raises(ValueError, match="scalar_schema"):
        TournamentRecords(
            {**config, "compatibility": {"scalar_schema": 16}},
            records.games,
            (),
            AssetVerifier(config),
            manifest={"run_id": "new", "passes": {}},
        )
