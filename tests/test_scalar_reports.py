"""Check versioned scalar reads, durable prefixes, ownership and missing values.

These host-only checks preserve historical headers without rewriting files and
reject current schema, row, identity and commit-boundary mismatches.
"""

import copy
import csv
import io
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from marl_battlegrounds.evaluation.scalar_reports import (
    ScalarCell,
    iter_policy_assignments,
    iter_scalar_rows,
)

_HEADER = (
    "run_id",
    "phase",
    "pass_id",
    "episode_id",
    "seed_id",
    "map_id",
    "config_id",
    "team_a_return",
    "episode_length",
)


def _write(
    tmp_path: Path,
    *,
    host: int = 2,
    scalar: int = 14,
    header: tuple[str, ...] = _HEADER,
    rows: Sequence[tuple[object, ...]] | None = None,
    suffix: bytes = b"not,committed",
) -> tuple[Path, dict[str, object]]:
    if rows is None:
        rows = [("run", "evaluation", "first", 17, 3, 1, "config", "", 2)]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    payload = stream.getvalue().encode()
    path = tmp_path / "episodes.csv"
    path.write_bytes(payload + suffix)
    manifest: dict[str, object] = {
        "schema_version": host,
        "metric_schema_id": "marlbg.tdm.scalar",
        "metric_schema_version": scalar,
        "run_id": "run",
        "configurations": {"config": {}},
        "passes": {
            "key": {
                "phase": "evaluation",
                "pass_id": "first",
                "completed_episode_ids": [17],
                "episodes": {"17": {"seed_id": 3, "map_id": 1, "config_id": "config"}},
            }
        },
        "tables": {path.name: {"durable_bytes": len(payload), "rows": len(rows)}},
    }
    return path, manifest


def _rows(
    path: Path, manifest: dict[str, object], *, batch_size: int = 256
) -> list[dict[str, ScalarCell]]:
    return [
        row
        for batch in iter_scalar_rows(
            path, manifest=manifest, expected_header=_HEADER, batch_size=batch_size
        )
        for row in batch
    ]


def test_current_read_preserves_exact_integer_summary_and_ignores_suffix(
    tmp_path: Path,
) -> None:
    path, manifest = _write(
        tmp_path,
        rows=[("run", "evaluation", "first", 17, 3, 1, "config", "", 2**24 + 1)],
    )
    before = path.read_bytes()
    row = _rows(path, manifest)[0]
    assert row["episode_length"] == 2**24 + 1
    assert type(row["episode_length"]) is int
    assert row["team_a_return"] is None
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", [1, 7, 13])
def test_historical_header_owns_removed_measurements_and_blank_values(
    tmp_path: Path, version: int
) -> None:
    header = (*_HEADER, "agent_0_return")
    path, manifest = _write(
        tmp_path,
        host=1,
        scalar=version,
        header=header,
        rows=[("run", "evaluation", "first", 17, 3, 1, "config", "", "2.0", "1.25")],
    )
    row = _rows(path, manifest)[0]
    assert tuple(row) == header
    assert row["agent_0_return"] == 1.25
    assert row["team_a_return"] is None
    assert row["episode_length"] == 2.0


@pytest.mark.parametrize(
    "host,scalar", [(1, 14), (2, 13), (2, 15), (3, 14), (True, 13), (1, True)]
)
def test_unknown_version_pairs_fail_without_changing_files(
    tmp_path: Path, host: int, scalar: int
) -> None:
    path, manifest = _write(tmp_path, host=host, scalar=scalar)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="schema"):
        _rows(path, manifest)
    assert path.read_bytes() == before


def test_current_header_is_required_and_exact(tmp_path: Path) -> None:
    path, manifest = _write(tmp_path)
    with pytest.raises(ValueError, match="expected header"):
        list(iter_scalar_rows(path, manifest=manifest))
    with pytest.raises(ValueError, match="current header"):
        list(
            iter_scalar_rows(
                path, manifest=manifest, expected_header=(*_HEADER, "extra")
            )
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "wrong"),
        ("phase", "training"),
        ("pass_id", "other"),
        ("episode_id", 18),
        ("seed_id", 4),
        ("map_id", 2),
        ("config_id", "missing"),
    ],
)
def test_recorded_ownership_must_match(
    tmp_path: Path, field: str, value: ScalarCell
) -> None:
    row: dict[str, ScalarCell] = dict(
        zip(
            _HEADER,
            ("run", "evaluation", "first", 17, 3, 1, "config", 0, 2),
            strict=True,
        )
    )
    row[field] = value
    path, manifest = _write(tmp_path, rows=[tuple(row.values())])
    with pytest.raises(ValueError):
        _rows(path, manifest)


@pytest.mark.parametrize("cell", ["nan", "inf", "text"])
def test_invalid_available_numeric_cells_fail(tmp_path: Path, cell: str) -> None:
    path, manifest = _write(
        tmp_path, rows=[("run", "evaluation", "first", 17, 3, 1, "config", cell, 2)]
    )
    with pytest.raises(ValueError, match="scalar value"):
        _rows(path, manifest)


@pytest.mark.parametrize(
    "change",
    [
        "short",
        "long",
        "rows",
        "row_width",
        "duplicate_header",
        "missing_header",
        "negative_bytes",
    ],
)
def test_invalid_header_width_or_durable_boundary_fails(
    tmp_path: Path, change: str
) -> None:
    header = (*_HEADER, "run_id") if change == "duplicate_header" else _HEADER
    if change == "missing_header":
        header = ("wrong", *_HEADER[1:])
    rows: list[tuple[object, ...]] = [
        ("run", "evaluation", "first", 17, 3, 1, "config", 0, 2)
    ]
    if change == "row_width":
        rows[0] = (*rows[0], 3)
    path, manifest = _write(tmp_path, header=header, rows=rows)
    if change == "short":
        path.write_bytes(path.read_bytes()[:10])
    if change == "long":
        cast(dict[str, dict[str, int]], manifest["tables"])[path.name][
            "durable_bytes"
        ] -= 1
    if change == "rows":
        cast(dict[str, dict[str, int]], manifest["tables"])[path.name]["rows"] += 1
    if change == "negative_bytes":
        cast(dict[str, dict[str, int]], manifest["tables"])[path.name][
            "durable_bytes"
        ] = -1
    with pytest.raises(ValueError):
        _rows(path, manifest)


def test_batches_are_bounded_and_missing_optional_table_is_empty(
    tmp_path: Path,
) -> None:
    row = ("run", "evaluation", "first", 17, 3, 1, "config", 0, 2)
    path, manifest = _write(tmp_path, rows=[row] * 7)
    batches = list(
        iter_scalar_rows(path, manifest=manifest, expected_header=_HEADER, batch_size=3)
    )
    assert [len(batch) for batch in batches] == [3, 3, 1]
    assert _rows(tmp_path / "full_metrics.csv", manifest) == []
    bad = copy.deepcopy(manifest)
    cast(dict[str, dict[str, int]], bad["tables"])[path.name]["rows"] = 6
    with pytest.raises(ValueError, match="row count"):
        _rows(path, bad)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_bad_batch_size_is_rejected(tmp_path: Path, batch_size: int) -> None:
    path, manifest = _write(tmp_path)
    with pytest.raises(ValueError, match="batch_size"):
        _rows(path, manifest, batch_size=batch_size)


def test_scalar_reader_import_does_not_load_jax() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import marl_battlegrounds.evaluation.scalar_reports; "
                "print(json.dumps([name for name in sys.modules "
                "if name == 'jax' or name.startswith('jax.')]))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_archived_schema_two_fixture_keeps_original_bytes_and_agent_returns() -> None:
    # Exact original 2026-09-09 writer-parity files; their schema is not relabelled.
    source = Path(__file__).parent / "fixtures" / "scalar_schema_2"
    path = source / "priority_metrics.csv"
    manifest = json.loads((source / "run_details.json").read_bytes())
    before = path.read_bytes()
    rows = [row for batch in iter_scalar_rows(path, manifest=manifest) for row in batch]
    assert len(rows) == 4
    assert tuple(rows[0]) == tuple(next(csv.reader(io.StringIO(before.decode()))))
    assert all(f"agent_{slot}_return" in rows[0] for slot in range(10))
    assert rows[0]["agent_3_return"] == -1.0
    assert rows[0]["seed_id"] is None
    assert path.read_bytes() == before


_ASSIGNMENT_HEADER = (
    "run_id",
    "phase",
    "pass_id",
    "episode_id",
    "team",
    "global_slot",
    "system_id",
    "policy_id",
    "decision_start",
    "decision_stop",
)


def _assignment_fixture(
    tmp_path: Path,
    rows: Sequence[tuple[object, ...]],
) -> tuple[Path, dict[str, object]]:
    path, manifest = _write(tmp_path, header=_ASSIGNMENT_HEADER, rows=rows)
    old = path
    path = tmp_path / "policy_assignments.csv"
    old.rename(path)
    tables = cast(dict[str, object], manifest["tables"])
    tables[path.name] = tables.pop(old.name)
    entry = cast(dict[str, dict[str, object]], manifest["passes"])["key"]
    entry["completed_episode_ids"] = []
    entry["trace_epochs"] = {"17": 8}
    entry["system_ids"] = {"team_a": "a", "team_b": "b"}
    entry["trace_config_ids"] = {"17": "config"}
    manifest["systems"] = {
        "a": {"components": [{"name": "first"}, {"name": "second"}]},
        "b": {"components": [{"name": "only"}]},
    }
    manifest["configurations"] = {
        "config": {
            "agent_profile": {
                "active_mask": [
                    True,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ]
            }
        }
    }
    return path, manifest


def test_assignment_reader_coalesces_across_flushes_and_preserves_unknown_gaps(
    tmp_path: Path,
) -> None:
    rows = [
        ("run", "evaluation", "first", 17, "team_a", 0, "a", 1, 0, 2),
        ("run", "evaluation", "first", 17, "team_b", 5, "b", 0, 0, 2),
        ("run", "evaluation", "first", 17, "team_a", 0, "a", 1, 2, 4),
        ("run", "evaluation", "first", 17, "team_b", 5, "b", 0, 2, 4),
        ("run", "evaluation", "first", 17, "team_a", 0, "a", -1, 5, 6),
        ("run", "evaluation", "first", 17, "team_a", 0, "a", -1, 6, 7),
        ("run", "evaluation", "first", 17, "team_a", 0, "a", -1, 8, 9),
    ]
    path, manifest = _assignment_fixture(tmp_path, rows)
    before = path.read_bytes()
    batches = list(
        iter_policy_assignments(
            path, manifest=manifest, expected_header=_ASSIGNMENT_HEADER, batch_size=2
        )
    )
    assert all(len(batch) <= 2 for batch in batches)
    actual = sorted(
        (
            row["global_slot"],
            row["policy_id"],
            row["decision_start"],
            row["decision_stop"],
        )
        for batch in batches
        for row in batch
    )
    assert actual == [(0, -1, 5, 7), (0, -1, 8, 9), (0, 1, 0, 4), (5, 0, 0, 4)]
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("global_slot", 10),
        ("global_slot", 1),
        ("team", "team_b"),
        ("system_id", "b"),
        ("policy_id", 2),
        ("policy_id", -2),
        ("decision_start", -1),
        ("decision_stop", 10),
        ("decision_stop", 0),
    ],
)
def test_assignment_reader_rejects_wrong_slots_owners_choices_and_epochs(
    tmp_path: Path, field: str, value: ScalarCell
) -> None:
    row: dict[str, ScalarCell] = dict(
        zip(
            _ASSIGNMENT_HEADER,
            ("run", "evaluation", "first", 17, "team_a", 0, "a", 1, 0, 2),
            strict=True,
        )
    )
    row[field] = value
    path, manifest = _assignment_fixture(tmp_path, [tuple(row.values())])
    with pytest.raises(ValueError):
        list(
            iter_policy_assignments(
                path, manifest=manifest, expected_header=_ASSIGNMENT_HEADER
            )
        )


def test_assignment_reader_rejects_overlaps_and_schedule_config_mismatch(
    tmp_path: Path,
) -> None:
    row = ("run", "evaluation", "first", 17, "team_a", 0, "a", 1, 0, 2)
    path, manifest = _assignment_fixture(tmp_path, [row, row])
    with pytest.raises(ValueError, match="overlap"):
        list(
            iter_policy_assignments(
                path, manifest=manifest, expected_header=_ASSIGNMENT_HEADER
            )
        )
    path, manifest = _write(tmp_path)
    entry = cast(dict[str, dict[str, object]], manifest["passes"])["key"]
    episodes = cast(dict[str, dict[str, object]], entry["episodes"])
    episodes["17"]["configuration_digest"] = "wrong"
    with pytest.raises(ValueError, match="schedule digest"):
        _rows(path, manifest)
