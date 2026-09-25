"""Read scoped evaluation tables without running games or changing saved files.

Results keep the older convenience fields and add bounded NumPy table access.
Saved views use one committed manifest snapshot. They never recover a writer,
load JAX, recompute old measurements or fit tournament ratings. Whole-table
allocation is explicit through ``table``; ``iter_table`` bounds raw row batches.
Each saved run reads with the full-report header of its own recorded scalar
schema (14 before the Red Zone columns, 15 now).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
    FULL_METRIC_NAMES_BY_SCHEMA_VERSION,
    METRIC_SCHEMA_ID,
    METRIC_SCHEMA_VERSION,
    PRIORITY_METRIC_NAMES,
)
from marl_battlegrounds.evaluation.run_writer import (
    EPISODE_COLUMNS,
    IDENTITY_COLUMNS,
    MATCH_COLUMNS,
)
from marl_battlegrounds.evaluation.scalar_reports import (
    iter_scalar_rows,
    iter_summary_rows,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.replay_v4 import ReplayArtifactV4
    from marl_battlegrounds.evaluation.tournament_records import TournamentRecords

Columns = dict[str, NDArray[np.generic]]
Row = dict[str, Any]
_TABLES = (
    "episodes",
    "priority_metrics",
    "full_metrics",
    "matches",
    "tournament_results",
    "tournament_rankings",
    "matchup_results",
    "map_results",
    "tournament_headline_metrics",
)
_SUMMARIES = frozenset(_TABLES[4:])
_FILES = {name: f"{name}.csv" for name in _TABLES if name != "tournament_rankings"}
_FILES["matches"] = "match_results.csv"
_TEXT = frozenset(
    {
        "run_id",
        "phase",
        "pass_id",
        "config_id",
        "team_a_policy",
        "team_b_policy",
        "checkpoint_id",
        "bootstrap_group",
        "system_id",
        "system_name",
        "policy",
        "opponent",
    }
)
_INTS = frozenset(
    {
        "episode_id",
        "seed_id",
        "map_id",
        "block_id",
        "outcome",
        "rank",
        "matches",
        "independent_blocks",
        "games_played",
        "completed_pairs",
        "total_steps_played",
        "episode_length_min",
        "episode_length_max",
        "score_difference_min",
        "score_difference_max",
        "total_score_difference",
        "wins",
        "draws",
        "losses",
        *(
            f"agent_{slot}_{suffix}"
            for slot in range(10)
            for suffix in ("class_id", "active")
        ),
    }
)
_EXACT = frozenset({"episode_length", "team_a_score", "team_b_score"})
_NULLABLE = frozenset({"seed_id", "map_id", "pass_id", "block_id", "rank"})
_HEADERS = {
    "episodes.csv": EPISODE_COLUMNS,
    "priority_metrics.csv": (*IDENTITY_COLUMNS, *PRIORITY_METRIC_NAMES),
    "full_metrics.csv": (*IDENTITY_COLUMNS, *FULL_METRIC_NAMES),
    "match_results.csv": MATCH_COLUMNS,
}
_CELL_TYPES = {
    "text": ("text", "None when absent"),
    "integer": ("integer", "None when absent; never rounded through float"),
    "float32": ("float32", "NaN when a measurement is unavailable"),
    "float64": ("float64", "NaN when a measurement is unavailable"),
    "nullable_float64": ("float64", "None when the derived value is unavailable"),
}


def _cell_type(column: str, table: str, current: bool) -> tuple[str, str]:
    """Share immutable type/null descriptions instead of thousands of tiny dicts."""
    if column in _TEXT or column.endswith("_status"):
        return _CELL_TYPES["text"]
    if column in _INTS or (
        column in _EXACT and table in {"episodes", "matches"} and current
    ):
        return _CELL_TYPES["integer"]
    if column in {"kd_ratio", "system_game_score"}:
        return _CELL_TYPES["nullable_float64"]
    return _CELL_TYPES[
        "float32"
        if current and table in {"priority_metrics", "full_metrics", "matches"}
        else "float64"
    ]


def _mapping(value: object) -> dict[str, Any]:
    """Return a plain host mapping or an empty mapping for absent metadata."""
    return dict(cast(Mapping[str, Any], value)) if isinstance(value, Mapping) else {}


def _stamp(path: Path) -> tuple[int, int, int, int]:
    """Keep nanosecond file identity; stat-result equality loses subsecond edits."""
    value = path.stat()
    return value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _schema(manifest: Mapping[str, Any]) -> None:
    """Reject unsupported host/scalar pairs before opening any table.

    Host schema 1 accepts scalar schemas 1 to 13. Host schema 2 accepts each
    scalar schema that FULL_METRIC_NAMES_BY_SCHEMA_VERSION lists (14 and 15).
    Raises ValueError for any other pair, a different metric schema ID or a
    pending recording restore.
    """
    host, scalar = manifest.get("schema_version"), manifest.get("metric_schema_version")
    if manifest.get("metric_schema_id") != METRIC_SCHEMA_ID or not (
        type(host) is int
        and type(scalar) is int
        and (
            (host == 2 and scalar in FULL_METRIC_NAMES_BY_SCHEMA_VERSION)
            or (host == 1 and 1 <= scalar <= 13)
        )
    ):
        raise ValueError("unsupported result host/scalar schema pair")
    if "recording_restore" in manifest:
        raise ValueError(
            "recording restore is pending; complete the explicit restore first"
        )


def _headers(manifest: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return the raw table headers that a checked manifest's version expects.

    A host-schema-2 manifest gets the full-report header of its own scalar
    schema from FULL_METRIC_NAMES_BY_SCHEMA_VERSION, so a saved schema-14 table
    reads with its 11,148 original columns. Host schema 1 keeps the current
    headers; its tables are always read by their stored headers. Call only
    after _schema has accepted the manifest.
    """
    if manifest["schema_version"] != 2:
        return _HEADERS
    names = FULL_METRIC_NAMES_BY_SCHEMA_VERSION[manifest["metric_schema_version"]]
    return {**_HEADERS, "full_metrics.csv": (*IDENTITY_COLUMNS, *names)}


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    """Read one exact run, refusing parent-directory guessing and pending restore."""
    path = run_dir / "run_details.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("run_dir must name one saved run containing run_details.json")
    try:
        manifest = json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as error:
        raise ValueError("run_details.json is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("run_details.json must contain a mapping")
    _schema(cast(dict[str, Any], manifest))
    return cast(dict[str, Any], manifest)


def _select_passes(
    manifest: Mapping[str, Any], phase: str | None, pass_id: str | None
) -> dict[str, Row]:
    """Resolve exact recorded scope, rejecting ambiguous or empty selections."""
    for value in (phase, pass_id):
        if value is not None and (
            not isinstance(cast(object, value), str) or not value
        ):
            raise ValueError("phase and pass_id must be nonempty strings when supplied")
    selected = {
        key: _mapping(entry)
        for key, entry in _mapping(manifest.get("passes")).items()
        if (phase is None or _mapping(entry).get("phase") == phase)
        and (pass_id is None or _mapping(entry).get("pass_id") == pass_id)
    }
    if not selected:
        raise ValueError("result selection contains no recorded passes")
    if phase is None and pass_id is not None and len(selected) != 1:
        raise ValueError("pass_id appears in several phases; supply phase too")
    identities = [
        (entry.get("phase"), entry.get("pass_id")) for entry in selected.values()
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("run manifest repeats a pass identity")
    return selected


def _scope(selected: Mapping[str, Row], whole_tournament: bool) -> Row:
    """Describe selected recorded passes without inventing a combined pass."""
    return {
        "passes": tuple(
            (entry["phase"], entry["pass_id"]) for entry in selected.values()
        ),
        "pass_keys": tuple(selected),
        "whole_tournament": whole_tournament,
    }


def _historical_coordinator(entry: Row, manifest: Mapping[str, Any]) -> bool:
    """Recognize the actual old zero-game coordinator from its complete contract."""
    if (
        entry.get("phase") != "tournament"
        or entry.get("pass_id") != "schedule"
        or entry.get("episodes")
    ):
        return False
    details = _mapping(manifest.get("details"))
    expected = details.get("num_matches")
    tables = _mapping(manifest.get("tables"))
    passes = [
        value
        for value in _mapping(manifest.get("passes")).values()
        if value.get("phase") == "tournament" and value.get("pass_id") != "schedule"
    ]
    identifiers = [
        value for item in passes for value in item.get("completed_episode_ids", ())
    ]
    return bool(
        _mapping(manifest.get("tournament_summary")).get("digest")
        and details.get("schedule_digest")
        and details.get("policies")
        and details.get("map_ids")
        and details.get("configuration_ids_by_map")
        and type(expected) is int
        and expected > 0
        and len(identifiers) == len(set(identifiers)) == expected
        and all(
            set(map(str, item.get("completed_episode_ids", ())))
            == set(item.get("episodes", {}))
            for item in passes
        )
        and tables.get("match_results.csv", {}).get("rows") == expected
        and all(
            tables.get(name, {}).get("rows", 0) > 0
            for name in (
                "tournament_results.csv",
                "matchup_results.csv",
                "map_results.csv",
            )
        )
    )


def _state(entry: Row, manifest: Mapping[str, Any]) -> tuple[str, str | None]:
    """Read explicit status or prove only the older recorded completion contract."""
    marker = _mapping(entry.get("result_state"))
    if marker:
        if marker.get("version") != 1 or marker.get("status") not in {
            "complete",
            "incomplete",
            "failed",
        }:
            raise ValueError("invalid pass result_state")
        return str(marker["status"]), marker.get("reason")
    if entry.get("pass_role") == "tournament_coordinator":
        summary = _mapping(manifest.get("tournament_summary"))
        qualification = _mapping(summary.get("qualification"))
        return (
            ("complete", None)
            if qualification.get("status") == "complete"
            else ("incomplete", "Tournament summary is not complete")
        )
    schedule = _mapping(entry.get("episodes"))
    completed = entry.get("completed_episode_ids", ())
    if schedule and set(map(str, completed)) == set(schedule):
        details = _mapping(entry.get("details"))
        replays = _mapping(entry.get("replays"))
        if all(str(value) in replays for value in details.get("replay_episodes", ())):
            return "complete", None
    # Older tournaments had a zero-game coordinator with a documented summary.
    if _historical_coordinator(entry, manifest):
        return "complete", None
    return "incomplete", "No complete recorded schedule and coverage proof"


def _source_coverage(
    entry: Mapping[str, Any],
    lengths: Mapping[tuple[str, str, str], int],
) -> tuple[Row, ...]:
    """Count saved source choices, including choices with no scheduled game.

    Repeated choices share one count and retain all their original positions.
    Counts join each recorded game by its declared map and source identity, never
    today's catalog. Zero played games means zero steps; a missing played length
    means None. Older or exact schedules without a source-choice list return ().
    """
    contract = _mapping(_mapping(entry.get("details")).get("evaluation_contract"))
    groups: dict[tuple[object, object], Row] = {}
    for index, choice in enumerate(contract.get("source_choices", ())):
        identity = choice.get("map_id"), choice.get("source_config_id")
        if identity not in groups:
            groups[identity] = {
                "map_id": identity[0],
                "source_config_id": identity[1],
                "source_choice_indices": [],
                "map_metadata": choice.get("map_metadata"),
                "scheduled_games": 0,
                "completed_games": 0,
                "completed_steps": 0,
            }
        groups[identity]["source_choice_indices"].append(index)
    completed = set(map(str, entry.get("completed_episode_ids", ())))
    for identifier, declaration in _mapping(entry.get("episodes")).items():
        group = groups.get(
            (declaration.get("map_id"), declaration.get("source_config_id"))
        )
        if group is None:
            continue
        group["scheduled_games"] += 1
        if identifier in completed:
            group["completed_games"] += 1
            length = lengths.get((entry["phase"], entry["pass_id"], identifier))
            if length is None:
                group["completed_steps"] = None
            elif group["completed_steps"] is not None:
                group["completed_steps"] += length
    for group in groups.values():
        group["source_choice_indices"] = tuple(group["source_choice_indices"])
    return tuple(groups.values())


def _columns(rows: Sequence[Row], *, table: str, current: bool) -> Columns:
    """Build one bounded NumPy batch with exact integers and honest missing cells."""
    if not rows:
        return {}
    result: Columns = {}
    for name in rows[0]:
        values = [row.get(name) for row in rows]
        text = name in _TEXT or name.endswith("_status")
        integer = name in _INTS or (
            name in _EXACT and table in {"episodes", "matches"} and current
        )
        if (
            text
            or name in _NULLABLE
            or (integer and any(value is None for value in values))
            or (
                name in {"kd_ratio", "system_game_score"}
                and any(value is None for value in values)
            )
        ):
            result[name] = np.asarray(values, dtype=object)
        elif integer:
            result[name] = np.asarray(values, dtype=np.int64)
        else:
            dtype = (
                np.float32
                if current and table in {"priority_metrics", "full_metrics", "matches"}
                else np.float64
            )
            result[name] = np.asarray(
                [np.nan if value is None else value for value in values], dtype=dtype
            )
    return result


def _row_arrays(columns: Columns, order: Sequence[int] | None = None) -> Iterator[Row]:
    """Yield host scalar rows without duplicating an existing wide column table."""
    if not columns:
        return
    size = len(next(iter(columns.values())))
    if any(len(value) != size for value in columns.values()):
        raise ValueError("in-memory table columns have different lengths")
    for index in range(size) if order is None else order:
        yield {
            name: value[index].item()
            if isinstance(value[index], np.generic)
            else value[index]
            for name, value in columns.items()
        }


class _View:
    """Keep one manifest scope and bounded row sources for a result object.

    Saved views own no open streams until iteration. Memory views borrow existing
    column arrays and keep only narrow episode/order metadata alongside them.
    """

    def __init__(
        self,
        manifest: dict[str, Any],
        *,
        run_dir: Path | None,
        phase: str | None = None,
        pass_id: str | None = None,
        memory: Mapping[str, object] | None = None,
    ) -> None:
        """Resolve one immutable manifest snapshot and describe available tables.

        The raw table headers follow the manifest's own scalar schema version.
        """
        _schema(manifest)
        self._headers = _headers(manifest)
        self.manifest = manifest
        self.run_dir = run_dir
        self._file_inodes: dict[str, int] = {}
        self._manifest_stamp = (
            None if run_dir is None else _stamp(run_dir / "run_details.json")
        )
        self.memory = {} if memory is None else dict(memory)
        self.selected = _select_passes(manifest, phase, pass_id)
        self.identities = {
            (entry["phase"], entry["pass_id"]) for entry in self.selected.values()
        }
        self._entries_by_identity = {
            (entry["phase"], entry["pass_id"]): entry
            for entry in self.selected.values()
        }
        tournament_keys = {
            key
            for key, entry in manifest["passes"].items()
            if entry.get("phase") == "tournament"
        }
        self.whole_tournament = bool(tournament_keys) and (
            tournament_keys.issubset(self.selected)
            or (
                manifest.get("tournament_reuse") is not None
                and any(
                    entry.get("pass_role") == "tournament_coordinator"
                    for entry in self.selected.values()
                )
            )
        )
        self.scope = _scope(self.selected, self.whole_tournament)
        states = {key: _state(entry, manifest) for key, entry in self.selected.items()}
        self.status = (
            "failed"
            if any(value[0] == "failed" for value in states.values())
            else "incomplete"
            if any(value[0] != "complete" for value in states.values())
            else "complete"
        )
        if self.whole_tournament and not manifest.get("tournament_summary"):
            self.status = "incomplete" if self.status != "failed" else "failed"
        inherited = _mapping(manifest.get("details"))
        if len(self.selected) == 1:
            inherited.update(
                _mapping(next(iter(self.selected.values())).get("details"))
            )
        configurations = _mapping(manifest.get("configurations"))
        for entry in self.selected.values():
            for identifier, content in _mapping(
                _mapping(entry.get("details")).get("configurations")
            ).items():
                if (
                    identifier in configurations
                    and configurations[identifier] != content
                ):
                    raise ValueError("conflicting recorded configuration content")
                configurations[identifier] = content
        self.metadata: Row = {
            **inherited,
            "run_id": manifest.get("run_id"),
            "scope": self.scope,
            "metric_schema_id": manifest["metric_schema_id"],
            "metric_schema_version": manifest["metric_schema_version"],
            "status": self.status,
            "pass_status": {
                key: {"status": value[0], "reason": value[1]}
                for key, value in states.items()
            },
            "systems": manifest.get("systems", {}),
            "configurations": configurations,
            "passes": self.selected,
        }
        self.metadata["tables"] = {name: self._describe(name) for name in _TABLES}
        self.metadata["spawn_balance"] = self._spawn_balance()

    def _header(self, filename: str) -> tuple[str, ...]:
        """Read only a stored header, or use the manifest's schema for an empty table.

        A current (host schema 2) stored header must equal the header of the
        manifest's own scalar schema; otherwise ValueError is raised.
        """
        if self.run_dir is None:
            value = self.memory.get(filename.removesuffix(".csv"))
            if isinstance(value, dict):
                return tuple(cast(dict[str, Any], value))
            if isinstance(value, (tuple, list)) and value:
                return tuple(cast(Sequence[Row], value)[0])
            return self._headers.get(filename, ())
        boundary = _mapping(self.manifest.get("tables")).get(filename)
        if boundary is None or not boundary.get("durable_bytes"):
            return self._headers.get(filename, ())
        path = self.run_dir / filename
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size < boundary["durable_bytes"]
        ):
            raise ValueError(
                f"committed result table is missing or truncated: {filename}"
            )
        self._file_inodes.setdefault(filename, path.stat().st_ino)
        with path.open("rb") as stream:
            header = stream.readline(boundary["durable_bytes"])
        if not header.endswith(b"\n"):
            raise ValueError(f"result table has an incomplete header: {filename}")
        try:
            names = tuple(next(csv.reader([header.decode("utf-8")], strict=True)))
        except (ValueError, UnicodeError, csv.Error) as error:
            raise ValueError(f"invalid result table header: {filename}") from error
        if (
            not names
            or len(set(names)) != len(names)
            or any(not name for name in names)
        ):
            raise ValueError(f"invalid result table header: {filename}")
        if (
            self.manifest["schema_version"] == 2
            and filename in self._headers
            and names != self._headers[filename]
        ):
            raise ValueError(f"incompatible current result table header: {filename}")
        return names

    def _present(self, filename: str) -> bool:
        """Test recorded table presence without treating a zero value as missing."""
        if self.run_dir is None:
            return filename.removesuffix(".csv") in self.memory
        return filename in self.manifest.get("tables", {})

    def _describe(self, name: str) -> Row:
        """Declare table columns, availability and scope without reading wide rows."""
        filename = (
            "tournament_results.csv" if name == "tournament_rankings" else _FILES[name]
        )
        summary = name in _SUMMARIES
        availability, reason = "available", None
        current = self.manifest["schema_version"] == 2
        selected_modes = [
            _mapping(entry.get("details")).get("metrics")
            for entry in self.selected.values()
            if entry.get("pass_role") != "tournament_coordinator"
        ]
        measured = any(
            value in {"priority", "full"}
            for entry in self.selected.values()
            for value in _mapping(entry.get("recorded_metrics_by_episode")).values()
        )
        has_full = any(
            value == "full"
            for entry in self.selected.values()
            for value in _mapping(entry.get("recorded_metrics_by_episode")).values()
        )
        requested_full = any(
            _mapping(entry.get("details")).get("full_metrics_episodes")
            for entry in self.selected.values()
        )
        present = self._present(filename)
        if name in {"episodes", "priority_metrics"}:
            present |= self._present("match_results.csv") and any(
                entry["phase"] == "tournament" for entry in self.selected.values()
            )
        if summary and not self.whole_tournament:
            availability, reason = (
                "unavailable",
                "This scope does not contain the complete tournament population",
            )
        elif summary and self.status != "complete":
            availability, reason = (
                "unavailable",
                "The complete tournament has not been finalized",
            )
        elif (
            name == "tournament_headline_metrics"
            and selected_modes
            and all(mode == "none" for mode in selected_modes)
        ):
            availability, reason = "disabled", "Tournament metrics were disabled"
        elif (
            name == "priority_metrics"
            and selected_modes
            and all(mode == "none" for mode in selected_modes)
            and not measured
            and (
                current
                or (not requested_full and not self._present("priority_metrics.csv"))
            )
        ):
            availability, reason = "disabled", "Priority metrics were disabled"
        elif (
            name == "full_metrics"
            and not has_full
            and not requested_full
            and selected_modes
            and all(mode != "full" for mode in selected_modes)
            and (current or not present)
        ):
            availability, reason = "disabled", "No full metric episodes were selected"
        elif not present:
            if (
                current
                and not summary
                and name in {"episodes", "priority_metrics", "full_metrics"}
                and not any(
                    entry.get("completed_episode_ids")
                    for entry in self.selected.values()
                )
            ):
                availability = "available"
            else:
                availability, reason = (
                    "unavailable",
                    "This table was not recorded in the selected experiment",
                )
        if name == "matches" and not any(
            entry["phase"] == "tournament" for entry in self.selected.values()
        ):
            availability, reason = (
                "unavailable",
                "This scope does not contain tournament games",
            )
        columns = self._header(filename)
        if name == "episodes":
            columns = (*EPISODE_COLUMNS, "system_game_score")
        elif (
            name == "priority_metrics"
            and not self._present(filename)
            and self._present("match_results.csv")
        ):
            columns = (
                *IDENTITY_COLUMNS,
                *(
                    value
                    for value in self._header("match_results.csv")
                    if value in PRIORITY_METRIC_NAMES
                    or (
                        not current
                        and value in {f"agent_{slot}_return" for slot in range(10)}
                    )
                ),
            )
        if summary:
            if name == "tournament_rankings":
                columns = (
                    *columns,
                    *(
                        value
                        for value in ("system_id", "system_name", "rank")
                        if value not in columns
                    ),
                )
            columns = (
                *columns,
                *(
                    value
                    for value in ("run_id", "phase", "pass_id")
                    if value not in columns
                ),
            )
        missing_fields = [
            column
            for column in EPISODE_COLUMNS
            if name == "episodes" and column not in self._header(filename)
        ]
        if name == "episodes" and any(
            _mapping(_mapping(entry.get("details")).get("evaluation_contract")).get(
                "version"
            )
            != 1
            and _mapping(entry.get("details")).get("first_system_team")
            not in {"team_a", "team_b"}
            for entry in self.selected.values()
        ):
            missing_fields.append("system_game_score")
        if name == "tournament_rankings" and not self.manifest.get("systems"):
            missing_fields.append("system_id")
        return {
            "availability": availability,
            "reason": reason,
            "columns": columns,
            "column_types": {
                column: _cell_type(column, name, current) for column in columns
            },
            "scope": self.scope,
            "missing_historical_fields": tuple(missing_fields),
            "unavailable_columns": {
                column: "The original recorded identity or value is unavailable"
                for column in missing_fields
            },
            "missing_passes": tuple(
                (entry["phase"], entry["pass_id"])
                for entry in self.selected.values()
                if self.manifest["schema_version"] == 1
                and name in {"priority_metrics", "full_metrics"}
                and not entry.get("recorded_metrics_by_episode")
                and _mapping(entry.get("details")).get("metrics") is None
            ),
        }

    def _check_snapshot(self, filename: str) -> None:
        """Reject destructive recovery while allowing append-only later commits."""
        if self.run_dir is None:
            return
        before = _mapping(self.manifest.get("tables")).get(
            filename, {"durable_bytes": 0, "rows": 0}
        )
        if before["durable_bytes"]:
            path = self.run_dir / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("result snapshot table is missing or unsafe")
            table_stat = path.stat()
            if table_stat.st_size < before[
                "durable_bytes"
            ] or table_stat.st_ino != self._file_inodes.setdefault(
                filename, table_stat.st_ino
            ):
                raise ValueError("result snapshot table was truncated or replaced")
        stamp = _stamp(self.run_dir / "run_details.json")
        if stamp == self._manifest_stamp:
            return
        current = _read_manifest(self.run_dir)
        if current.get("run_id") != self.manifest.get("run_id"):
            raise ValueError("saved result identity changed during reading")
        after = _mapping(current.get("tables")).get(
            filename, {"durable_bytes": 0, "rows": 0}
        )
        if after.get("durable_bytes", 0) < before.get("durable_bytes", 0) or after.get(
            "rows", 0
        ) < before.get("rows", 0):
            raise ValueError("result snapshot was truncated or restored while reading")
        for key, entry in self.selected.items():
            now = _mapping(current.get("passes")).get(key)
            if now is None or not set(entry.get("completed_episode_ids", ())).issubset(
                now.get("completed_episode_ids", ())
            ):
                raise ValueError("selected result pass was restored while reading")
            for identifier, declaration in _mapping(entry.get("episodes")).items():
                if _mapping(now.get("episodes")).get(identifier) != declaration:
                    raise ValueError("selected result conditions changed while reading")
        if (
            self.manifest.get("tournament_summary") != current.get("tournament_summary")
            and self.manifest.get("tournament_summary") is not None
        ):
            raise ValueError("selected tournament summary changed while reading")
        self._manifest_stamp = stamp

    def _raw(self, filename: str, rows: int) -> Iterator[Row]:
        """Stream scoped rows from an existing memory table or durable CSV prefix."""
        if not self._present(filename):
            return
        if self.run_dir is None:
            data = self.memory[filename.removesuffix(".csv")]
            if isinstance(data, dict):
                order = None
                columns = cast(Columns, data)
                if "episode_id" in columns and "completion_order" in self.memory:
                    indexes = {
                        int(value): index
                        for index, value in enumerate(columns["episode_id"])
                    }
                    order = tuple(
                        indexes[value]
                        for value in cast(
                            Sequence[int], self.memory["completion_order"]
                        )
                        if value in indexes
                    )
                iterator = _row_arrays(cast(Columns, data), order)
            else:
                sequence = cast(Sequence[Row], data)
                if filename in self._headers and "completion_order" in self.memory:
                    indexed = {int(row["episode_id"]): row for row in sequence}
                    iterator = (
                        indexed[value]
                        for value in cast(
                            Sequence[int], self.memory["completion_order"]
                        )
                        if value in indexed
                    )
                else:
                    iterator = iter(sequence)
            for row in iterator:
                if (
                    filename not in self._headers
                    or (row.get("phase"), row.get("pass_id")) in self.identities
                ):
                    yield dict(row)
            return
        summary = filename not in self._headers
        reader = iter_summary_rows if summary else iter_scalar_rows
        for batch in reader(
            self.run_dir / filename,
            manifest=self.manifest,
            expected_header=self._headers.get(filename),
            batch_size=rows,
        ):
            self._check_snapshot(filename)
            for row in batch:
                if summary or (row.get("phase"), row.get("pass_id")) in self.identities:
                    yield dict(row)

    def _entry(self, row: Mapping[str, Any]) -> Row:
        """Find the exact phase/pass owner of one recorded row."""
        entry = self._entries_by_identity.get((row.get("phase"), row.get("pass_id")))
        if entry is None:
            raise ValueError("result row has no selected pass owner")
        return entry

    def _episodes(self, rows: int) -> Iterator[Row]:
        """Project exact required outcomes while keeping each raw authority's order."""
        for filename in ("episodes.csv", "match_results.csv"):
            for row in self._raw(filename, rows):
                result = {name: row.get(name) for name in EPISODE_COLUMNS}
                entry = self._entry(row)
                details = _mapping(entry.get("details"))
                new_contract = (
                    _mapping(details.get("evaluation_contract")).get("version") == 1
                )
                binding = details.get("first_system_team")
                # Only new fixed-team contracts establish Team A as the focal
                # submitted System. Old records must carry their actual binding.
                team = "team_a" if new_contract else binding
                outcome = row.get("outcome")
                result["system_game_score"] = (
                    None
                    if team not in {"team_a", "team_b"} or outcome not in {1, 2, 3}
                    else 0.5
                    if outcome == 3
                    else float(outcome == (1 if team == "team_a" else 2))
                )
                yield result

    def _rankings(self, rows: int) -> Iterator[Row]:
        """Add exact competition ranks from saved unrounded Elo, without fitting."""
        values = list(self._raw("tournament_results.csv", rows))
        participants = _mapping(
            _mapping(self.manifest.get("details")).get("participants")
        )
        if not participants:
            for entry in self.selected.values():
                ids = _mapping(entry.get("system_ids"))
                for team, descriptor in _mapping(entry.get("policies")).items():
                    label = _mapping(descriptor).get("name")
                    if label is not None and ids.get(team) is not None:
                        previous = participants.get(str(label), ids[team])
                        participants[str(label)] = (
                            ids[team] if previous == ids[team] else None
                        )
        for row in values:
            label = row.get("policy")
            row["system_id"] = participants.get(str(label))
            row["system_name"] = label
        values.sort(
            key=lambda row: (
                row.get("elo") is None,
                -float(row.get("elo") or 0),
                str(row.get("system_id") or row.get("policy") or ""),
            )
        )
        previous: object = object()
        rank = 0
        for index, row in enumerate(values, start=1):
            rating = row.get("elo")
            if rating is not None and rating != previous:
                rank = index
            row["rank"] = None if rating is None else rank
            previous = rating
            yield row

    def iter_rows(self, name: str, rows: int) -> Iterator[Row]:
        """Apply one small projection over raw rows; never refit or reduce metrics."""
        description = self.metadata["tables"][name]
        availability = description["availability"]
        if availability == "disabled":
            return
        if availability != "available":
            raise ValueError(description["reason"])
        if name == "episodes":
            yield from self._episodes(rows)
            return
        if name == "priority_metrics":
            yield from self._raw("priority_metrics.csv", rows)
            for row in self._raw("match_results.csv", rows):
                entry = self._entry(row)
                coverage = _mapping(entry.get("recorded_metrics_by_episode"))
                measured = coverage.get(str(row["episode_id"])) in {"priority", "full"}
                if not coverage and self.manifest["schema_version"] == 1:
                    details = _mapping(entry.get("details"))
                    measured = details.get("metrics") in {
                        "priority",
                        "full",
                    } or row["episode_id"] in details.get("full_metrics_episodes", ())
                if measured:
                    yield {name: row.get(name) for name in description["columns"]}
            return
        iterator = (
            self._rankings(rows)
            if name == "tournament_rankings"
            else self._raw(_FILES[name], rows)
        )
        for row in iterator:
            if name in _SUMMARIES:
                row.setdefault("run_id", self.manifest.get("run_id"))
                row.setdefault("phase", "tournament")
                row.setdefault("pass_id", None)
            yield row

    def _spawn_balance(self) -> object:
        """Count only completed recorded games and steps under their own bindings."""
        records: dict[tuple[str, str], Row] = {}
        lengths: dict[tuple[str, str, str], int] = {}
        for filename in ("episodes.csv", "match_results.csv"):
            for row in self._raw(filename, 128):
                if row.get("episode_length") is not None:
                    lengths[
                        (str(row["phase"]), str(row["pass_id"]), str(row["episode_id"]))
                    ] = int(row["episode_length"])
        for entry in self.selected.values():
            if entry.get(
                "pass_role"
            ) == "tournament_coordinator" or _historical_coordinator(
                entry, self.manifest
            ):
                continue
            details = _mapping(entry.get("details"))
            contract = _mapping(details.get("evaluation_contract"))
            schedule = _mapping(entry.get("episodes"))
            completed = set(map(str, entry.get("completed_episode_ids", ())))
            games = dict.fromkeys(("default", "swapped", "unreported"), 0)
            steps: dict[str, int | None] = {name: 0 for name in games}
            missing_lengths: list[int] = []
            pairs: dict[str, list[tuple[str, object]]] = {}
            for identifier, declaration in schedule.items():
                declaration = _mapping(declaration)
                choice = declaration.get("spawn_locations")
                key = declaration.get(
                    "paired_comparison_key", declaration.get("comparison_key")
                )
                kind = declaration.get("comparison_kind")
                if key is not None and kind == "verified_spawn_pair":
                    pairs.setdefault(str(key), []).append((identifier, choice))
                if identifier in completed:
                    label = (
                        "default"
                        if choice == 0
                        else "swapped"
                        if choice == 1
                        else "unreported"
                    )
                    games[label] += 1
                    length = lengths.get((entry["phase"], entry["pass_id"], identifier))
                    if length is None:
                        steps[label] = None
                        missing_lengths.append(int(identifier))
                    elif steps[label] is not None:
                        steps[label] = cast(int, steps[label]) + length
            for identifier in completed - set(schedule):
                games["unreported"] += 1
                length = lengths.get((entry["phase"], entry["pass_id"], identifier))
                if length is None:
                    steps["unreported"] = None
                    missing_lengths.append(int(identifier))
                elif steps["unreported"] is not None:
                    steps["unreported"] += length
            complete_pairs = sum(
                len(group) == 2
                and {value for _, value in group} == {0, 1}
                and all(identifier in completed for identifier, _ in group)
                for group in pairs.values()
            )
            ids = _mapping(entry.get("system_ids"))
            mode = contract.get(
                "spawn_mode",
                contract.get(
                    "requested_spawn_locations",
                    details.get("spawn_locations", "explicit"),
                ),
            )
            declared = (
                bool(pairs)
                or mode == "paired"
                or contract.get("pairing_protocol") is not None
            )
            paired = (
                None
                if not declared
                else bool(schedule)
                and complete_pairs * 2 == len(schedule)
                and len(completed) == len(schedule)
            )
            records[(entry["phase"], entry["pass_id"])] = {
                "mode": mode,
                "system_id": ids.get("team_a"),
                "opponent_id": ids.get("team_b"),
                "completed_games": games,
                "completed_steps": steps,
                "expected_pairs": len(schedule) // 2
                if mode == "paired"
                else len(pairs),
                "completed_pairs": complete_pairs,
                "missing_episode_ids": tuple(
                    sorted(map(int, set(schedule) - completed))
                ),
                "paired_complete": paired,
                "missing_episode_lengths": tuple(sorted(missing_lengths)),
                "source_coverage": _source_coverage(entry, lengths),
            }
        return next(iter(records.values())) if len(records) == 1 else records

    @property
    def replay_paths(self) -> tuple[Path, ...]:
        """Return existing committed replays in the recorded episode order.

        Prefer each pass's explicit contract order, then its older declared ID
        list. Whole-tournament views use the shared global schedule. Historical
        files without an ordered declaration fall back to ascending episode IDs;
        JSON object key order never claims an execution or completion order.
        """
        if self.run_dir is None:
            return ()
        result: list[tuple[str, str, Path]] = []
        for entry in self.selected.values():
            records = _mapping(entry.get("replays"))
            schedule = _mapping(entry.get("episodes"))
            details = _mapping(entry.get("details"))
            contract = _mapping(details.get("evaluation_contract"))
            declared = tuple(
                map(str, contract.get("episode_ids", details.get("episode_ids", ())))
            )
            if len(set(declared)) != len(declared):
                raise ValueError("recorded replay schedule repeats an episode ID")
            remaining = sorted((set(schedule) | set(records)) - set(declared), key=int)
            for episode_id in (*declared, *remaining):
                record = records.get(episode_id)
                if record is None:
                    continue
                relative = Path(record["path"])
                if len(relative.parts) != 2 or relative.parts[0] != "replays":
                    raise ValueError("committed replay path leaves the run")
                path = self.run_dir / relative
                if (
                    path.is_symlink()
                    or path.parent.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != record["bytes"]
                ):
                    raise ValueError("committed replay is missing, changed or unsafe")
                result.append((entry["phase"], episode_id, path))
        if self.whole_tournament:
            schedule = _mapping(self.manifest.get("details")).get("schedule", ())
            order = {
                str(row["episode_id"]): index for index, row in enumerate(schedule)
            }
            tournament = iter(
                sorted(
                    (row for row in result if row[0] == "tournament"),
                    key=lambda row: (order.get(row[1], len(order)), int(row[1])),
                )
            )
            return tuple(
                next(tournament)[2] if phase == "tournament" else path
                for phase, _, path in result
            )
        return tuple(path for _, _, path in result)


def _result_view(
    manifest: Row,
    *,
    run_dir: Path | None,
    phase: str | None = None,
    pass_id: str | None = None,
    memory: Mapping[str, object] | None = None,
) -> _View:
    """Dispatch additive canonical references without importing execution code."""
    if "tournament_reuse" in manifest:
        from marl_battlegrounds.evaluation.canonical_results import CanonicalView

        return CanonicalView(
            manifest, run_dir=run_dir, phase=phase, pass_id=pass_id, memory=memory
        )
    return _View(manifest, run_dir=run_dir, phase=phase, pass_id=pass_id, memory=memory)


class _ResultAccess:
    """Share bounded table methods across saved, evaluation and tournament results."""

    _view: _View

    @property
    def status(self) -> str:
        """Return complete, incomplete or failed for the exact selected scope."""
        return self._view.status

    @property
    def run_dir(self) -> Path | None:
        """Return the exact saved run directory, or None for in-memory results."""
        return self._view.run_dir

    @property
    def replay_paths(self) -> tuple[Path, ...]:
        """Return existing committed replay paths, or () for memory-only replays.

        Paths follow the recorded schedule, including global tournament order.
        Older files without an ordered declaration use ascending episode IDs;
        that fallback does not claim the original execution or completion order.
        Missing, changed or unsafe committed replay files raise ValueError.
        """
        return self._view.replay_paths

    def iter_table(self, name: str, rows: int = 128) -> Iterator[Columns]:
        """Yield at most ``rows`` recorded rows as NumPy columns per batch.

        Parameters
        ----------
        name : str
            Name listed in ``metadata['tables']``. Raw views retain completion
            order. Summary views require the whole completed tournament scope.
        rows : int, optional
            Positive batch size, default 128. Booleans are not valid sizes.

        Yields
        ------
        dict[str, numpy.ndarray]
            One bounded host batch. Missing metric cells are NaN; nullable
            identity/count cells use None. No file is written or result fitted.

        Raises
        ------
        ValueError
            Name, size, availability, schema or a saved boundary is invalid.
        OSError
            A saved file cannot be read. Disabled/available-empty tables yield
            nothing; their metadata distinguishes these cases.
        """
        if name not in _TABLES:
            raise ValueError(
                f"unknown table {name!r}; choose from {', '.join(_TABLES)}"
            )
        if type(rows) is not int or rows <= 0:
            raise ValueError("rows must be a positive integer, excluding booleans")
        batch: list[Row] = []
        for row in self._view.iter_rows(name, rows):
            batch.append(row)
            if len(batch) == rows:
                yield _columns(
                    batch,
                    table=name,
                    current=self._view.manifest["schema_version"] == 2,
                )
                batch.clear()
        if batch:
            yield _columns(
                batch, table=name, current=self._view.manifest["schema_version"] == 2
            )

    def table(self, name: str) -> Columns:
        """Allocate the whole selected table as NumPy columns.

        ``name`` uses the same names, scope, errors and missing-value rules as
        ``iter_table``. This explicit convenience can use substantial RAM for
        full reports. Disabled or available-empty tables return an empty dict.
        Existing saved files and legacy fields are never changed.
        """
        batches = list(self.iter_table(name))
        if not batches:
            return {}
        if len(batches) == 1:
            return batches[0]
        return {
            name: np.concatenate([batch[name] for batch in batches])
            for name in batches[0]
        }


@dataclass(frozen=True)
class EpisodeResult:
    """Keep one exact required game outcome, independent of optional metrics.

    IDs and scores are Python integers; map_id can be None for a custom map.
    ``episode_length`` counts real transitions from this episode's start, even
    when the authored simulator tick starts above zero. ``config_id`` names the
    exact resolved configuration. Outcome is 1/2 for Team A/B wins or 3 for draw.
    """

    episode_id: int
    seed_id: int
    map_id: int | None
    outcome: int
    episode_length: int
    team_a_score: int
    team_b_score: int
    config_id: str


def _memory_manifest(
    metadata: Row, *, tournament: bool, completed: Sequence[int]
) -> Row:
    """Project already resolved host metadata into the common read-only schema."""
    passes: Any = metadata.get("passes") if tournament else None
    if not isinstance(passes, dict):
        phase, pass_id = (
            str(metadata.get("phase", "evaluation")),
            str(metadata.get("pass_id", "1")),
        )
        schedule = metadata.get("schedule", {})
        if isinstance(schedule, (tuple, list)):
            schedule = {
                str(row["episode_id"]): row for row in cast(Sequence[Row], schedule)
            }
        else:
            schedule = {str(key): value for key, value in _mapping(schedule).items()}
        passes = {
            f"{phase}:{pass_id}": {
                "phase": phase,
                "pass_id": pass_id,
                "details": dict(metadata),
                "episodes": schedule,
                "completed_episode_ids": list(completed),
                "policies": metadata.get("policies", {}),
                "system_ids": metadata.get("system_ids", {}),
                "replays": {},
                "recorded_metrics_by_episode": {
                    str(value): "full"
                    if metadata.get("metrics") == "full"
                    or value in metadata.get("full_metrics_episodes", ())
                    else metadata.get("metrics", "priority")
                    for value in completed
                },
                "result_state": {
                    "version": 1,
                    "status": metadata.get("status", "complete"),
                    "schedule_digest": metadata.get("schedule_digest", "memory"),
                    "reason": None,
                },
            }
        }
    result: Row = {
        "schema_version": 2,
        "metric_schema_id": METRIC_SCHEMA_ID,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "run_id": metadata.get("run_id"),
        "passes": passes,
        "systems": metadata.get("systems", {}),
        "configurations": metadata.get("configurations", {}),
        "tables": {},
        "details": dict(metadata),
    }
    if tournament and "tournament_reuse" in metadata:
        result["tournament_reuse"] = metadata["tournament_reuse"]
    if tournament:
        result["tournament_summary"] = metadata.get(
            "tournament_summary", {"digest": "in-memory-qualified"}
        )
    return result


@dataclass(frozen=True)
class EvaluationResult(_ResultAccess):
    """Expose evaluation output and scoped host tables without owning a writer.

    Legacy fields keep their existing meanings: saved metric dicts are empty,
    and ``episodes`` contains only newly completed games. Uniform table methods
    include the whole selected saved pass, including earlier resumed games.
    ``metadata`` records table availability, status, scope and spawn coverage.
    Memory arrays are borrowed, not copied into a second wide report.

    Attributes
    ----------
    priority_metrics, full_metrics : dict of str to numpy.ndarray
        Without a writer: column name to one-dimensional NumPy array, one row
        per selected game, in episode ID order. priority_metrics has every
        game, except with metrics="none", when it has only the
        full_metrics_episodes games. full_metrics has only the
        full_metrics_episodes games, except with metrics="full", when it has
        every game. Each dict starts with the IDENTITY_COLUMNS: run_id, phase,
        pass_id, episode_id (int32), seed_id (uint32), map_id, config_id,
        team_a_policy, team_b_policy, checkpoint_id, and each slot's
        agent_<slot>_class_id and agent_<slot>_active. The float32 metric
        columns follow, where NaN means unavailable. A table with no selected
        game is an empty dict. With a writer both are empty; the values stay
        in its files.
    episodes : tuple of EpisodeResult
        Compact required outcomes of the games completed by this call, in
        episode ID order.
    metadata : dict
        The actual schedule, systems, settings and run identity. Creating the
        result adds table availability, status, scope and spawn coverage to
        this same dictionary. A generated pass records its resolved Red Zone
        depth, in map units, as "red_zone_depth" in the options of
        ``metadata["evaluation_contract"]`` (0.0 means the rule is off).
    completed_episode_ids : tuple of int, default=()
        Every durably completed episode ID in this pass, earlier resumed
        games included, in ascending order. Empty means this call's
        ``episodes`` are the whole pass.
    paths : dict of str to Path or None, default=None
        The writer's files, by name. None for in-memory results.
    replays : tuple of ReplayArtifactV4, default=()
        Current (version 4) replays captured in memory by this call. Empty
        with a writer, which saves them as files instead.
    """

    priority_metrics: Columns
    full_metrics: Columns
    episodes: tuple[EpisodeResult, ...]
    metadata: Row
    completed_episode_ids: tuple[int, ...] = ()
    paths: dict[str, Path] | None = None
    replays: tuple[ReplayArtifactV4, ...] = ()
    _view: _View = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Bind saved scope or borrow memory columns and exact required outcomes."""
        if self.paths and "run_details" in self.paths:
            directory = Path(self.paths["run_details"]).parent
            view = _View(
                _read_manifest(directory),
                run_dir=directory,
                phase=self.metadata.get("phase"),
                pass_id=self.metadata.get("pass_id"),
            )
        else:
            completed = self.completed_episode_ids or tuple(
                value.episode_id for value in self.episodes
            )
            manifest = _memory_manifest(
                self.metadata, tournament=False, completed=completed
            )
            entry = next(iter(manifest["passes"].values()))
            by_id = {value.episode_id: value for value in self.episodes}
            order = self.metadata.get("completion_order", tuple(by_id))
            rows: list[Row] = []
            for identifier in order:
                value = by_id[identifier]
                declaration = _mapping(entry["episodes"].get(str(identifier)))
                row = {name: declaration.get(name) for name in IDENTITY_COLUMNS}
                row.update(
                    run_id=manifest["run_id"],
                    phase=entry["phase"],
                    pass_id=entry["pass_id"],
                    episode_id=value.episode_id,
                    seed_id=value.seed_id,
                    map_id=value.map_id,
                    config_id=value.config_id,
                    outcome=value.outcome,
                    episode_length=value.episode_length,
                    team_a_score=value.team_a_score,
                    team_b_score=value.team_b_score,
                )
                policies = self.metadata.get("policies", ())
                if (
                    isinstance(policies, (tuple, list))
                    and len(cast(Sequence[object], policies)) == 2
                ):
                    descriptions = cast(Sequence[object], policies)
                    row["team_a_policy"] = _mapping(descriptions[0]).get("name")
                    row["team_b_policy"] = _mapping(descriptions[1]).get("name")
                row["checkpoint_id"] = self.metadata.get("checkpoint_id")
                config = _mapping(
                    _mapping(manifest.get("configurations")).get(value.config_id)
                )
                profile = _mapping(config.get("agent_profile"))
                for slot in range(10):
                    for suffix, source in (
                        ("class_id", "class_ids"),
                        ("active", "active_mask"),
                    ):
                        if source in profile:
                            row[f"agent_{slot}_{suffix}"] = int(profile[source][slot])
                rows.append(row)
            memory: Row = {"episodes": tuple(rows), "completion_order": order}
            for name, columns in (
                ("priority_metrics", self.priority_metrics),
                ("full_metrics", self.full_metrics),
            ):
                if columns:
                    memory[name] = columns
            view = _View(manifest, run_dir=None, memory=memory)
        object.__setattr__(self, "_view", view)
        self.metadata.update(view.metadata)


@dataclass(frozen=True)
class TournamentResult(_ResultAccess):
    """Keep existing tournament fields plus uniform complete-population views.

    Row tuples are the already computed summary outputs. ``headline_metrics`` is
    an optional keyword-only table computed by the runner once. No accessor fits
    Elo or reconstructs missing measurements. Saved views use committed files;
    in-memory views borrow the supplied rows and wide full-metric arrays.

    Attributes
    ----------
    matches : tuple of dict
        One row per completed game: the match_results table with its outcome,
        scores and priority values.
    tournament_results : tuple of dict
        One row per entrant: centered rating, rates, interval availability and
        weighted worst-20-percent expected score.
    matchup_results : tuple of dict
        Directed policy/opponent rows with rates and score intervals.
    map_results : tuple of dict
        Policy/map rows with opponent-weighted rates and match counts.
    full_metrics : dict of str to numpy.ndarray
        Selected full metric columns captured in memory by this call. Empty
        with file output or when full metrics are off.
    replays : tuple of ReplayArtifactV4
        Current (version 4) replays captured in memory by this call. Empty with
        file output, which saves them as files instead.
    metadata : dict
        Method details, systems, participants, passes, configurations and the
        run's settings. Creating the result adds table availability, status,
        scope and spawn coverage to this same dictionary. A new run from
        run_tournament's policies route records "red_zone_depth" here (the Red
        Zone depth in map units; 0.0 means the rule is off); such a run saved
        before the rule has no such key and played at 0.0. Configured and
        canonical results (CanonicalTournamentResult) never carry this key, so
        a missing key says nothing about their depth. For every result, each
        game's depth is team_deathmatch_red_zone_depth in its configuration
        under "configurations"; content saved before the rule lacks that field
        and played at 0.0.
    paths : dict of str to Path or None
        Produced files by name, or None for in-memory results.
    headline_metrics : tuple of dict, default=()
        Optional keyword-only headline table, computed by the runner once.
    """

    matches: tuple[Row, ...]
    tournament_results: tuple[Row, ...]
    matchup_results: tuple[Row, ...]
    map_results: tuple[Row, ...]
    full_metrics: Columns
    replays: tuple[ReplayArtifactV4, ...]
    metadata: Row
    paths: dict[str, Path] | None
    headline_metrics: tuple[Row, ...] = field(default=(), kw_only=True)
    _view: _View = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Bind complete tournament rows to their actual pass and system owners."""
        if self.paths and "run_details" in self.paths:
            directory = Path(self.paths["run_details"]).parent
            view = _result_view(
                _read_manifest(directory), run_dir=directory, phase="tournament"
            )
        else:
            manifest = _memory_manifest(self.metadata, tournament=True, completed=())
            memory: Row = {
                "match_results": self.matches,
                "tournament_results": self.tournament_results,
                "matchup_results": self.matchup_results,
                "map_results": self.map_results,
            }
            if "completion_order" in self.metadata:
                memory["completion_order"] = self.metadata["completion_order"]
            if self.full_metrics:
                memory["full_metrics"] = self.full_metrics
            if self.headline_metrics:
                memory["tournament_headline_metrics"] = self.headline_metrics
            if isinstance(self, CanonicalTournamentResult):
                memory["_record_access"] = self._record_access  # pyright: ignore[reportPrivateUsage]
            view = _result_view(
                manifest, run_dir=None, phase="tournament", memory=memory
            )
        object.__setattr__(self, "_view", view)
        self.metadata.update(view.metadata)


@dataclass(frozen=True)
class CanonicalTournamentResult(TournamentResult):
    """Expose a complete selected snapshot through ordinary tournament tables.

    Attributes
    ----------
    snapshot_id : str
        Immutable selected configuration identity; also ``big_12_id``
        in metadata. Fixture snapshots do not establish official qualification.
    challenger_id : str or None
        Logical entrant ID of the optional canonical challenger. None means
        twelve only for a canonical call. Custom configuration results describe
        their selected population, of any supported size, with no separate challenger.
    planned_games, reused_games, executed_games : int
        Whole logical-run counts, including durable games from earlier attempts.
        Metadata ``executed_this_call`` counts only new work in this call.
    games_per_opponent : int
        Uniform resolved number of completed games per unordered participant pair.
    protocol_compliant : bool
        Whether the resolved scientific conditions match the snapshot protocol.
        This is not admission approval or proof of scientific eligibility.

    Notes
    -----
    Original row and replay identities remain unchanged. Uniform tables cover
    reused and newly executed games; inherited full_metrics/replays retain only
    new in-memory captures. No-file results borrow immutable source files for
    lazy access. Keep those files available for the lifetime of the result.
    """

    snapshot_id: str = field(kw_only=True)
    challenger_id: str | None = field(kw_only=True)
    planned_games: int = field(kw_only=True)
    reused_games: int = field(kw_only=True)
    executed_games: int = field(kw_only=True)
    games_per_opponent: int = field(kw_only=True)
    protocol_compliant: bool = field(kw_only=True)
    _record_access: TournamentRecords | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )


@dataclass(frozen=True)
class SavedResults(_ResultAccess):
    """Hold a read-only saved scope with metadata and bounded table methods.

    Instances come from ``load_results``. They own no writer or open file and do
    not promise live updates. A destructive restore invalidates active iteration.
    """

    metadata: Row
    _view: _View = field(repr=False, compare=False)


def load_results(
    run_dir: str | Path, *, phase: str | None = None, pass_id: str | None = None
) -> SavedResults:
    """Read one saved run or an explicitly selected pass without changing it.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Exact directory containing run_details.json. Parent output directories
        are not searched and no newest child is selected.
    phase : str or None, optional
        Select all passes in this phase, default None for the whole run.
    pass_id : str or None, optional
        Select a named pass, default None. Without phase it must be unique.

    Returns
    -------
    SavedResults
        Read-only status, metadata, replay paths and NumPy table access. Missing
        historical facts stay unavailable; no current-schema values are rebuilt.
        Each table's column_types maps names to immutable (type, null-rule)
        tuples; repeated descriptions share storage.

    Raises
    ------
    ValueError
        Path, schema, scope or committed content is invalid, or coordinated
        recording restore is pending.
    OSError
        Files cannot be read. No files, locks, downloads or JAX work are created.
    """
    directory = Path(run_dir)
    view = _result_view(
        _read_manifest(directory), run_dir=directory, phase=phase, pass_id=pass_id
    )
    return SavedResults(view.metadata, view)
