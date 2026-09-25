"""Check bounded indexed reports, complete origin keys and true CSV record offsets.

Indexes use the ordinary scalar ownership/cell rules, keep UTF-8 and quoted
newlines intact, skip unused metric conversion, and reject changed sources.
Host schema 2 indexes read scalar schemas 14 and 15 and reject 16.
"""

import csv
import io
from pathlib import Path
from typing import Any, cast

import pytest

from marl_battlegrounds.evaluation.scalar_reports import (
    IndexedScalarTable,
    OriginKey,
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
    "team_a_policy",
    "team_a_return",
    "episode_length",
)


def _fixture(
    tmp_path: Path, scalar: int = 14
) -> tuple[Path, dict[str, Any], tuple[OriginKey, ...]]:
    names = ('Mâge "one"\nSecond line', "Other, player", "Last")
    rows = [
        ("run", "tournament", f"pass-{i}", 1, i, 47, "config", name, 2.5, 2**24 + 1)
        for i, name in enumerate(names)
    ]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(_HEADER)
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    path = tmp_path / "match_results.csv"
    path.write_bytes(payload + b"uncommitted,broken\n")
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "metric_schema_version": scalar,
        "metric_schema_id": "marlbg.tdm.scalar",
        "run_id": "run",
        "configurations": {"config": {}},
        "passes": {
            str(i): {
                "phase": "tournament",
                "pass_id": f"pass-{i}",
                "completed_episode_ids": [1],
                "episodes": {"1": {"seed_id": i}},
            }
            for i in range(3)
        },
        "tables": {"match_results.csv": {"durable_bytes": len(payload), "rows": 3}},
    }
    keys = tuple(("run", "tournament", f"pass-{i}", 1) for i in range(3))
    return path, manifest, keys


def test_index_seeks_actual_multiline_utf8_records_and_keeps_full_origin(
    tmp_path: Path,
) -> None:
    path, manifest, keys = _fixture(tmp_path)
    expected = [
        row
        for batch in iter_scalar_rows(path, manifest=manifest, expected_header=_HEADER)
        for row in batch
    ]
    renamed = path.with_name("content-sha256")
    path.rename(renamed)
    index = IndexedScalarTable(
        renamed,
        table_name="match_results.csv",
        manifest=manifest,
        expected_header=_HEADER,
        origins=keys,
    )
    actual = list(index.iter_rows(keys[::-1], batch_size=2))
    assert [len(batch) for batch in actual] == [2, 1]
    assert [row for batch in actual for row in batch] == expected[::-1]
    assert actual[-1][0]["episode_length"] == 2**24 + 1
    assert len(index._positions) == 3  # pyright: ignore[reportPrivateUsage]


def test_index_subset_parses_only_requested_cells(tmp_path: Path) -> None:
    path, manifest, keys = _fixture(tmp_path)
    path.write_bytes(path.read_bytes().replace(b",2.5,", b",bad,"))
    index = IndexedScalarTable(
        path,
        table_name="match_results.csv",
        manifest=manifest,
        expected_header=_HEADER,
        origins=(keys[0], keys[2]),
    )
    assert list(index.iter_rows((keys[2],), columns=("episode_length",))) == [
        ({"episode_length": 2**24 + 1},)
    ]
    with pytest.raises(ValueError, match="invalid scalar value"):
        list(index.iter_rows((keys[2],)))
    with pytest.raises(ValueError, match="indexed game"):
        list(index.iter_rows((keys[1],)))
    with pytest.raises(ValueError, match="not stored"):
        list(index.iter_rows((keys[2],), columns=("not_here",)))


@pytest.mark.parametrize("kind", ("same_size", "replacement", "truncate", "append"))
def test_index_rejects_changed_content(tmp_path: Path, kind: str) -> None:
    path, manifest, keys = _fixture(tmp_path)
    index = IndexedScalarTable(
        path,
        table_name="match_results.csv",
        manifest=manifest,
        expected_header=_HEADER,
        origins=keys,
    )
    before = path.read_bytes()
    if kind == "same_size":
        path.write_bytes(before.replace(b",2.5,", b",3.5,"))
    elif kind == "replacement":
        path.unlink()
        path.write_bytes(before)
    elif kind == "truncate":
        path.write_bytes(before[:20])
    else:
        path.write_bytes(before + b"suffix")
    with pytest.raises(ValueError, match="changed"):
        list(index.iter_rows(keys))


def test_index_rejects_missing_duplicate_and_incomplete_origins(tmp_path: Path) -> None:
    path, manifest, keys = _fixture(tmp_path)
    for selection in ((*keys, keys[0]), ((*keys[0][:3], 9),)):
        with pytest.raises(ValueError, match=r"repeats|missing"):
            IndexedScalarTable(
                path,
                table_name="match_results.csv",
                manifest=manifest,
                expected_header=_HEADER,
                origins=selection,
            )
    manifest["passes"]["0"]["completed_episode_ids"] = []
    with pytest.raises(ValueError, match="durable completion"):
        IndexedScalarTable(
            path,
            table_name="match_results.csv",
            manifest=manifest,
            expected_header=_HEADER,
            origins=keys,
        )


@pytest.mark.parametrize("size", (True, 0, -1, 1.5))
def test_index_rejects_invalid_read_batch_size(tmp_path: Path, size: object) -> None:
    path, manifest, keys = _fixture(tmp_path)
    index = IndexedScalarTable(
        path,
        table_name="match_results.csv",
        manifest=manifest,
        expected_header=_HEADER,
        origins=keys,
    )
    with pytest.raises(ValueError, match="positive integer"):
        list(index.iter_rows(keys, batch_size=cast(int, size)))


def test_index_reads_both_current_scalar_schemas_and_rejects_a_later_one(
    tmp_path: Path,
) -> None:
    for scalar in (14, 15, 16):
        directory = tmp_path / str(scalar)
        directory.mkdir()
        path, manifest, keys = _fixture(directory, scalar)
        if scalar == 16:
            with pytest.raises(ValueError, match="schema"):
                IndexedScalarTable(
                    path,
                    table_name="match_results.csv",
                    manifest=manifest,
                    expected_header=_HEADER,
                    origins=keys,
                )
            continue
        index = IndexedScalarTable(
            path,
            table_name="match_results.csv",
            manifest=manifest,
            expected_header=_HEADER,
            origins=keys,
        )
        rows = [row for batch in index.iter_rows(keys[::-1]) for row in batch]
        assert [row["pass_id"] for row in rows] == ["pass-2", "pass-1", "pass-0"]
