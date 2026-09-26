"""Read durable scalar CSV rows without changing saved files or loading JAX.

``iter_scalar_rows`` reads bounded batches from a run manifest's committed byte
range. Historical headers keep their recorded order and values. Current headers
must match the caller's shared schema. Host schema 2 accepts each scalar schema
listed by metric_catalog.FULL_METRIC_NAMES_BY_SCHEMA_VERSION (14 and 15); the
catalog is standard-library only, so this reader still loads no JAX. This is a
low-level host reader; it does not resume a writer, recompute metrics or
qualify a tournament population.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Buffer, Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO, cast

from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES_BY_SCHEMA_VERSION,
)

ScalarCell = str | int | float | None
type _Passes = Mapping[tuple[str, str], tuple[Mapping[str, object], frozenset[int]]]

_TEXT_COLUMNS = frozenset(
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
        "team",
        "policy",
        "opponent",
        "determinism",
        "conditional_interval_assumption",
        "system_name",
        "elo_interval_status",
        "expected_score_interval_status",
        "sampling_interval_status",
    }
)
_INTEGER_COLUMNS = frozenset(
    {
        "episode_id",
        "seed_id",
        "map_id",
        "block_id",
        "outcome",
        "global_slot",
        "policy_id",
        "decision_start",
        "decision_stop",
        *(
            f"agent_{slot}_{field}"
            for slot in range(10)
            for field in ("class_id", "active")
        ),
    }
)
_SUMMARY_COLUMNS = frozenset({"episode_length", "team_a_score", "team_b_score"})
_AGGREGATE_INTEGERS = frozenset(
    {
        "matches",
        "independent_blocks",
        "declared_blocks",
        "scheduled_games",
        "completed_games",
        "supported_independent_sampling_units",
        "wins",
        "draws",
        "losses",
        "games_played",
        "completed_pairs",
        "episode_length_min",
        "episode_length_max",
        "total_steps_played",
        "score_difference_min",
        "score_difference_max",
        "total_score_difference",
        "rank",
    }
)


class _DurablePrefix(io.RawIOBase):
    """Expose at most a saved byte count from an already opened binary file.

    The caller owns the underlying stream. Closing this wrapper does not close
    that stream. Buffered text readers cannot read an uncommitted suffix through
    this wrapper, even when their internal read-ahead is larger than the prefix.
    """

    def __init__(self, stream: BinaryIO, size: int) -> None:
        """Keep the stream and remaining nonnegative byte count; do not seek it."""
        super().__init__()
        self._stream: BinaryIO = stream
        self._remaining: int = size

    def readable(self) -> bool:
        """Return True so BufferedReader can use this read-only stream."""
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        """Copy available committed bytes into buffer and return their count."""
        view = memoryview(buffer).cast("B")
        payload = self._stream.read(min(len(view), self._remaining))
        count = len(payload)
        view[:count] = payload
        self._remaining -= count
        return count


def _integer(value: object, *, name: str) -> int:
    """Require a nonnegative manifest integer, rejecting booleans and fractions."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _parse_cell(
    name: str, value: str, *, current: bool, summary: bool = False
) -> ScalarCell:
    """Restore one CSV cell, preserving blanks and historical scalar precision."""
    if value == "":
        return None
    if name in _TEXT_COLUMNS:
        return value
    try:
        if (
            name in _INTEGER_COLUMNS
            or (current and name in _SUMMARY_COLUMNS)
            or (summary and name in _AGGREGATE_INTEGERS)
        ):
            return int(value)
        number = float(value)
    except ValueError as error:
        raise ValueError(f"invalid scalar value for {name}: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"nonfinite scalar value for {name}")
    return number


def _check_ownership(
    row: Mapping[str, ScalarCell],
    manifest: Mapping[str, object],
    passes: _Passes,
    *,
    require_completion: bool = True,
) -> None:
    """Check a row's run, named pass and completed episode against its manifest."""
    if row.get("run_id") != manifest.get("run_id"):
        raise ValueError("scalar row belongs to a different run")
    selected = passes.get((str(row.get("phase")), str(row.get("pass_id"))))
    if selected is None:
        raise ValueError("scalar row does not identify one recorded pass")
    entry, completed = selected
    episode_id = row.get("episode_id")
    if not isinstance(episode_id, int) or episode_id < 0:
        raise ValueError("scalar row has no valid episode ID")
    if require_completion and episode_id not in completed:
        raise ValueError("scalar row episode is not a durable completion")
    schedule = entry.get("episodes", {})
    if not isinstance(schedule, Mapping):
        raise ValueError("run manifest has an invalid episode table")
    declaration = cast(Mapping[str, object], schedule).get(str(episode_id), {})
    if not isinstance(declaration, Mapping):
        raise ValueError("run manifest has an invalid episode declaration")
    declaration = cast(Mapping[str, object], declaration)
    for name in ("seed_id", "map_id", "config_id"):
        if name in row and name in declaration and declaration[name] != row[name]:
            raise ValueError(f"scalar row {name} differs from its episode declaration")
    if (
        "config_id" in row
        and "configuration_digest" in declaration
        and declaration["configuration_digest"] != row["config_id"]
    ):
        raise ValueError("scalar row configuration differs from its schedule digest")
    config_id = row.get("config_id")
    configurations = manifest.get("configurations", {})
    if config_id is not None and (
        not isinstance(configurations, Mapping) or config_id not in configurations
    ):
        raise ValueError("scalar row configuration is not recorded")


def _pass_lookup(
    manifest: Mapping[str, object],
) -> dict[tuple[str, str], tuple[Mapping[str, object], frozenset[int]]]:
    """Index existing pass and completion declarations once for constant-time joins."""
    entries = manifest.get("passes")
    if not isinstance(entries, Mapping):
        raise ValueError("run manifest has no valid pass table")
    result: dict[tuple[str, str], tuple[Mapping[str, object], frozenset[int]]] = {}
    for entry in cast(Mapping[str, object], entries).values():
        if not isinstance(entry, Mapping):
            raise ValueError("run manifest has an invalid pass entry")
        entry = cast(Mapping[str, object], entry)
        phase, pass_id = entry.get("phase"), entry.get("pass_id")
        completed = entry.get("completed_episode_ids")
        if not isinstance(phase, str) or not isinstance(pass_id, str):
            raise ValueError("run manifest has invalid pass labels")
        if not isinstance(completed, (list, tuple)) or any(
            type(value) is not int or value < 0
            for value in cast(Sequence[object], completed)
        ):
            raise ValueError("run manifest has invalid completed episode IDs")
        key = (phase, pass_id)
        if key in result:
            raise ValueError("run manifest repeats a named pass")
        result[key] = entry, frozenset(cast(Sequence[int], completed))
    return result


def _csv_rows(stream: io.TextIOBase) -> Iterator[list[str]]:
    """Parse CSV rows and report malformed CSV or UTF-8 as a clear value error."""
    try:
        yield from csv.reader(stream, strict=True)
    except (csv.Error, UnicodeError) as error:
        raise ValueError("invalid scalar CSV encoding or quoting") from error


class _ByteLines:
    """Yield UTF-8 physical lines while retaining exact byte positions.

    ``csv.reader`` asks for additional lines when a quoted field spans a newline.
    Reading one physical line at a time keeps the byte offset at the end of the
    consumed CSV record. The supplied binary stream stays caller-owned. ``stop``
    is an exclusive durable byte boundary; bytes beyond it are never read.
    """

    def __init__(self, stream: BinaryIO, stop: int) -> None:
        """Borrow a positioned binary stream and its checked exclusive boundary."""
        self.stream = stream
        self.stop = stop
        self.position = stream.tell()

    def __iter__(self) -> _ByteLines:
        """Return this single-pass physical-line iterator."""
        return self

    def __next__(self) -> str:
        """Read one physical line, failing if the durable prefix became shorter."""
        if self.position >= self.stop:
            raise StopIteration
        payload = self.stream.readline(self.stop - self.position)
        if not payload:
            raise ValueError("durable scalar table was truncated while reading")
        self.position += len(payload)
        return payload.decode("utf-8")


def _csv_rows_from_lines(lines: _ByteLines) -> Iterator[list[str]]:
    """Use the shared CSV quoting rules with exact binary record positions."""
    try:
        yield from csv.reader(lines, strict=True)
    except (csv.Error, UnicodeError) as error:
        raise ValueError("invalid scalar CSV encoding or quoting") from error


def _parse_report_row(
    header: Sequence[str],
    cells: Sequence[str],
    *,
    current: bool,
    summary: bool = False,
    columns: frozenset[str] | None = None,
) -> dict[str, ScalarCell]:
    """Check width and parse requested cells with the shared historical rules.

    ``columns=None`` parses every cell. A narrow internal request avoids turning
    unused full-report values into Python floats. The CSV record is still parsed
    correctly, including quoting. Unknown requested columns are not invented.
    """
    if len(cells) != len(header):
        raise ValueError("scalar row width differs from its header")
    return {
        name: _parse_cell(name, value, current=current, summary=summary)
        for name, value in zip(header, cells, strict=True)
        if columns is None or name in columns
    }


type OriginKey = tuple[str, str, str, int]
_OWNERSHIP_COLUMNS = frozenset(
    {"run_id", "phase", "pass_id", "episode_id", "seed_id", "map_id", "config_id"}
)


def _origin_key(row: Mapping[str, ScalarCell]) -> OriginKey:
    """Keep full recorded ownership; episode numbers alone are not unique."""
    return (
        str(row["run_id"]),
        str(row["phase"]),
        str(row["pass_id"]),
        cast(int, row["episode_id"]),
    )


def _file_stamp(path: Path) -> tuple[int, int, int, int, int]:
    """Detect file replacement and same-size edits while an index is borrowed."""
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class IndexedScalarTable:
    """Read selected immutable report rows in requested origin order.

    Parameters
    ----------
    path : str or Path
        Verified local content file. Its basename need not be the table role.
    table_name : str
        Original CSV role, such as ``full_metrics.csv``, in the source manifest.
    manifest : mapping
        Immutable source manifest with durable byte/row counts and completions.
    expected_header : sequence of str or None
        Exact current header; historical files use their stored header.
    origins : sequence of (str, str, str, int)
        Unique full run/phase/pass/episode keys that this index must contain.

    Notes
    -----
    Host-only. Construction scans the committed prefix once through the shared
    reader and retains byte offsets only for selected games. It validates every
    row's ownership but does not convert optional measurements. Each later read
    seeks directly to the selected records, with memory bounded by its batch and
    one CSV record. A changed file invalidates the index. Asset content hashing
    belongs to the caller's asset authority, not to every indexed row read.
    No files are created, repaired or changed.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        table_name: str,
        manifest: Mapping[str, object],
        expected_header: Sequence[str] | None,
        origins: Sequence[OriginKey],
    ) -> None:
        """Validate one immutable prefix and index all requested game origins.

        Raises ValueError for duplicate/missing origins, unsafe content, malformed
        CSV, ownership errors or changing files. File access can raise OSError.
        """
        self.path = Path(path)
        self.table_name = table_name
        self.manifest = manifest
        self.origins = frozenset(origins)
        if len(self.origins) != len(origins):
            raise ValueError("indexed scalar selection repeats an origin")
        self._positions: dict[OriginKey, tuple[int, int]] = {}
        self._stamp = _file_stamp(self.path)
        for _ in _iter_rows(
            self.path,
            manifest=manifest,
            expected_header=expected_header,
            table_name=table_name,
            columns=_OWNERSHIP_COLUMNS,
            index_record=self._remember,
        ):
            pass
        if self.origins != self._positions.keys():
            raise ValueError("indexed scalar table is missing selected origins")
        self._check_unchanged()
        tables = cast(Mapping[str, Mapping[str, int]], manifest["tables"])
        self.size = tables[table_name]["durable_bytes"]
        with self.path.open("rb") as stream:
            self.header = tuple(
                next(_csv_rows_from_lines(_ByteLines(stream, self.size)))
            )
        self._passes = _pass_lookup(manifest)
        self._exact = manifest["schema_version"] == 2 and table_name in {
            "episodes.csv",
            "match_results.csv",
        }

    def _remember(self, row: dict[str, ScalarCell], start: int, stop: int) -> None:
        """Retain one selected record span; reject duplicate selected ownership."""
        key = _origin_key(row)
        if key not in self.origins:
            return
        if key in self._positions:
            raise ValueError("indexed scalar table repeats a selected origin")
        self._positions[key] = start, stop

    def _check_unchanged(self) -> None:
        """Reject symlinks and changed bytes before using saved record offsets."""
        if self.path.is_symlink() or _file_stamp(self.path) != self._stamp:
            raise ValueError("indexed scalar source changed after verification")

    def iter_rows(
        self,
        origins: Sequence[OriginKey],
        *,
        batch_size: int = 128,
        columns: Sequence[str] | None = None,
    ) -> Iterator[tuple[dict[str, ScalarCell], ...]]:
        """Yield selected rows in caller order without scanning unrelated rows.

        ``origins`` must be unique indexed keys. ``batch_size`` is a positive
        integer, excluding booleans. ``columns=None`` returns the entire stored
        row; a sequence selects existing columns without recomputing values.
        Missing keys/columns or changed content raise ValueError. Returned dicts
        are fresh caller-owned values. Errors may follow earlier yielded batches.
        """
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if len(set(origins)) != len(origins) or any(
            key not in self._positions for key in origins
        ):
            raise ValueError("requested origins must be unique indexed game keys")
        selected = None if columns is None else frozenset(columns)
        if selected is not None and not selected.issubset(self.header):
            raise ValueError("requested scalar columns are not stored in this table")
        parsed = None if selected is None else selected | _OWNERSHIP_COLUMNS
        self._check_unchanged()
        batch: list[dict[str, ScalarCell]] = []
        with self.path.open("rb") as stream:
            for key in origins:
                start, stop = self._positions[key]
                stream.seek(start)
                reader = _csv_rows_from_lines(_ByteLines(stream, stop))
                row = _parse_report_row(
                    self.header,
                    next(reader),
                    current=self._exact,
                    columns=parsed,
                )
                _check_ownership(row, self.manifest, self._passes)
                if _origin_key(row) != key or next(reader, None) is not None:
                    raise ValueError(
                        "indexed scalar record no longer matches its origin"
                    )
                batch.append(
                    row
                    if selected is None
                    else {
                        name: value for name, value in row.items() if name in selected
                    }
                )
                if len(batch) == batch_size:
                    self._check_unchanged()
                    yield tuple(batch)
                    batch.clear()
            self._check_unchanged()
            if batch:
                yield tuple(batch)


def _iter_rows(
    path: str | Path,
    *,
    manifest: Mapping[str, object],
    expected_header: Sequence[str] | None = None,
    batch_size: int = 256,
    assignments: bool = False,
    passes: _Passes | None = None,
    summary: bool = False,
    table_name: str | None = None,
    columns: frozenset[str] | None = None,
    index_record: Callable[[dict[str, ScalarCell], int, int], None] | None = None,
) -> Iterator[tuple[dict[str, ScalarCell], ...]]:
    """Parse shared durable rows, admitting unfinished games only for assignments.

    Public wrappers own argument/result contracts. ``assignments=True`` requires
    host schema 2 (scalar 14 or 15) and trace ownership instead of a completion.
    Host schema 1 accepts scalar 1 to 13; host schema 2 accepts each key of
    FULL_METRIC_NAMES_BY_SCHEMA_VERSION. A supplied pass
    lookup reuses the caller's manifest index; otherwise build it once. No rows
    outside the saved byte boundary are parsed and no file is changed.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    host = manifest.get("schema_version")
    scalar = manifest.get("metric_schema_version")
    if (
        type(host) is not int
        or type(scalar) is not int
        or not (
            (host == 1 and 1 <= scalar <= 13)
            or (host == 2 and scalar in FULL_METRIC_NAMES_BY_SCHEMA_VERSION)
        )
        or manifest.get("metric_schema_id") != "marlbg.tdm.scalar"
    ):
        raise ValueError("unsupported host/scalar report schema combination")
    current = host == 2
    if assignments and not current:
        raise ValueError(
            "policy assignments require host schema 2 and scalar schema 14 or 15"
        )
    if current and expected_header is None and not summary:
        raise ValueError("current scalar reports require an exact expected header")
    selected = Path(path)
    filename = selected.name if table_name is None else table_name
    tables = manifest.get("tables")
    if not isinstance(tables, Mapping):
        raise ValueError("run manifest has no valid table boundaries")
    boundary = cast(Mapping[str, object], tables).get(
        filename, {"durable_bytes": 0, "rows": 0}
    )
    if not isinstance(boundary, Mapping):
        raise ValueError("run table has an invalid durable boundary")
    boundary = cast(Mapping[str, object], boundary)
    size = _integer(boundary.get("durable_bytes"), name="durable_bytes")
    expected_rows = _integer(boundary.get("rows"), name="rows")
    if size == 0:
        if expected_rows:
            raise ValueError("nonempty scalar table has a zero durable byte boundary")
        return
    if (
        selected.is_symlink()
        or not selected.is_file()
        or selected.stat().st_size < size
    ):
        raise ValueError(
            "durable scalar table is missing, truncated or a symbolic link"
        )
    if passes is None and not summary:
        passes = _pass_lookup(manifest)
    exact_summaries = current and filename in {"episodes.csv", "match_results.csv"}
    with selected.open("rb") as binary:
        binary.seek(size - 1)
        if binary.read(1) != b"\n":
            raise ValueError("durable scalar boundary does not end at a complete row")
        binary.seek(0)
        with io.TextIOWrapper(
            io.BufferedReader(_DurablePrefix(binary, size)),
            encoding="utf-8",
            newline="",
        ) as text:
            lines = _ByteLines(binary, size) if index_record is not None else None
            reader = _csv_rows(text) if lines is None else _csv_rows_from_lines(lines)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError("durable scalar table has no header") from error
            if (
                not header
                or any(not name for name in header)
                or len(header) != len(set(header))
            ):
                raise ValueError(
                    "scalar table header names must be nonempty and unique"
                )
            if not summary and not {
                "run_id",
                "phase",
                "pass_id",
                "episode_id",
            }.issubset(header):
                raise ValueError(
                    "scalar table header is missing required ownership columns"
                )
            if (
                current
                and expected_header is not None
                and tuple(header) != tuple(expected_header)
            ):
                raise ValueError("scalar table has an incompatible current header")
            batch: list[dict[str, ScalarCell]] = []
            count = 0
            while True:
                start = 0 if lines is None else lines.position
                try:
                    cells = next(reader)
                except StopIteration:
                    break
                row = _parse_report_row(
                    header,
                    cells,
                    current=exact_summaries,
                    summary=summary,
                    columns=columns,
                )
                if not summary:
                    assert passes is not None
                    _check_ownership(
                        row, manifest, passes, require_completion=not assignments
                    )
                if index_record is not None and lines is not None:
                    index_record(row, start, lines.position)
                count += 1
                if count > expected_rows:
                    raise ValueError("scalar table exceeds its durable row count")
                batch.append(row)
                if len(batch) == batch_size:
                    yield tuple(batch)
                    batch.clear()
            if count != expected_rows:
                raise ValueError("scalar table differs from its durable row count")
            if batch:
                yield tuple(batch)


def iter_scalar_rows(
    path: str | Path,
    *,
    manifest: Mapping[str, object],
    expected_header: Sequence[str] | None = None,
    batch_size: int = 256,
) -> Iterator[tuple[dict[str, ScalarCell], ...]]:
    """Read committed scalar CSV rows with their original names and availability.

    Parameters
    ----------
    path : str or pathlib.Path
        Table path. Its filename selects the manifest's durable byte/row boundary.
    manifest : Mapping[str, object]
        Loaded run_details.json. Supports host schema 1/scalar 1 through 13, and
        host schema 2/scalar 14 or 15. Rows must name a run, pass and completion.
    expected_header : sequence of str or None, optional
        Required exact header for a current table. Ignored for historical tables,
        whose recorded header owns the column names and order. Defaults to None.
    batch_size : int, optional
        Maximum rows per returned tuple. Defaults to 256; must be positive and
        cannot be a boolean.

    Yields
    ------
    tuple of dict[str, str or int or float or None]
        File-order rows in batches no larger than batch_size. Blank cells are
        None. Identity strings remain strings. Integer identities and current
        episodes.csv/match_results.csv summaries stay integers; metrics are
        floats. Missing tables with no durable rows yield nothing.

    Raises
    ------
    ValueError
        Unsupported versions, invalid header/cells/ownership or inconsistent
        committed byte/row boundaries. Errors in later rows can follow earlier
        yielded batches; consume the iterator fully to validate its final count.
    OSError
        The table cannot be read.

    Notes
    -----
    Host-only, with bounded row buffering and no JAX imports. The reader ignores
    uncommitted suffixes and never writes or truncates files. It neither computes
    missing historical values nor authorizes append/resume. The caller owns
    manifest freshness. Completion lookup memory scales with recorded episodes.
    """
    yield from _iter_rows(
        path, manifest=manifest, expected_header=expected_header, batch_size=batch_size
    )


def iter_summary_rows(
    path: str | Path,
    *,
    manifest: Mapping[str, object],
    expected_header: Sequence[str] | None = None,
    batch_size: int = 128,
) -> Iterator[tuple[dict[str, ScalarCell], ...]]:
    """Read a durable tournament summary without requiring episode columns.

    Parameters
    ----------
    path : str or pathlib.Path
        Saved summary CSV inside the run. Its filename selects the committed
        byte and row boundary from ``manifest``.
    manifest : Mapping[str, object]
        One committed run manifest, with a supported host/scalar schema pair.
    expected_header : sequence of str or None, optional
        Exact current header when the caller knows it. Historical headers retain
        their stored order. None accepts the saved summary's unique header.
    batch_size : int, optional
        Positive maximum rows per yielded tuple, default 128. Booleans fail.

    Yields
    ------
    tuple of dict
        Stored rows. Identity/status text stays text, exact aggregate counts
        remain integers and empty cells become None. No ratings are fitted.

    Raises
    ------
    ValueError
        Versions, headers, values or durable boundaries are invalid.
    OSError
        Saved content cannot be read.

    Notes
    -----
    The result layer owns whole-population availability. This shared parser
    reads bounded committed prefixes without JAX, writes or recovery.
    """
    yield from _iter_rows(
        path,
        manifest=manifest,
        expected_header=expected_header,
        batch_size=batch_size,
        summary=True,
    )


def _assignment_values(
    row: Mapping[str, ScalarCell],
    manifest: Mapping[str, object],
    passes: _Passes,
) -> tuple[tuple[str, str, int, int], int, int]:
    """Validate one assignment's slot, interval, saved epoch and registered owner."""
    phase, pass_id = str(row["phase"]), str(row["pass_id"])
    episode_id = row["episode_id"]
    slot, choice = row.get("global_slot"), row.get("policy_id")
    start, stop = row.get("decision_start"), row.get("decision_stop")
    if (
        not isinstance(episode_id, int)
        or episode_id < 1
        or not isinstance(slot, int)
        or not 0 <= slot < 10
        or not isinstance(choice, int)
        or choice < -1
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= 2**31
    ):
        raise ValueError(
            "policy assignment has invalid slot, choice or action interval"
        )
    entry = passes[(phase, pass_id)][0]
    epochs = entry.get("trace_epochs")
    if not isinstance(epochs, Mapping):
        raise ValueError("policy assignment has no durable trace epochs")
    epoch = cast(Mapping[str, object], epochs).get(str(episode_id))
    if type(epoch) is not int or not 0 <= epoch < 2**31 or stop > epoch + 1:
        raise ValueError("policy assignment exceeds its durable action epoch")
    team = "team_a" if slot < 5 else "team_b"
    if row.get("team") != team:
        raise ValueError("policy assignment team does not own its global slot")
    episodes = cast(Mapping[str, Mapping[str, object]], entry.get("episodes", {}))
    episode = episodes.get(str(episode_id), {})
    system_ids = episode.get("system_ids", entry.get("system_ids"))
    if not isinstance(system_ids, Mapping) or system_ids.get(team) != row.get(
        "system_id"
    ):
        raise ValueError(
            "policy assignment System differs from its recorded team owner"
        )
    systems = manifest.get("systems")
    if not isinstance(systems, Mapping):
        raise ValueError("policy assignment has no recorded System table")
    system = cast(Mapping[str, object], systems).get(str(row.get("system_id")))
    if not isinstance(system, Mapping):
        raise ValueError("policy assignment System is not registered")
    components = cast(Mapping[str, object], system).get("components")
    if not isinstance(components, list) or choice >= len(
        cast(list[object], components)
    ):
        raise ValueError("policy assignment ID is outside its component table")
    config_ids = entry.get("trace_config_ids", {})
    if not isinstance(config_ids, Mapping):
        raise ValueError("policy assignment has an invalid configuration binding")
    config_id = cast(Mapping[str, object], config_ids).get(str(episode_id))
    configurations = manifest.get("configurations", {})
    if not isinstance(configurations, Mapping) or config_id not in configurations:
        raise ValueError("policy assignment has no recorded episode configuration")
    if not isinstance(config_id, str):
        raise ValueError("policy assignment configuration ID must be text")
    config = cast(Mapping[str, object], configurations)[config_id]
    profile = (
        cast(Mapping[str, object], config).get("agent_profile")
        if isinstance(config, Mapping)
        else None
    )
    active = (
        cast(Mapping[str, object], profile).get("active_mask")
        if isinstance(profile, Mapping)
        else None
    )
    if (
        not isinstance(active, list)
        or len(cast(list[object], active)) != 10
        or not active[slot]
    ):
        raise ValueError("policy assignment slot is not active in the recorded roster")
    return (phase, pass_id, episode_id, slot), start, stop


def iter_policy_assignments(
    path: str | Path,
    *,
    manifest: Mapping[str, object],
    expected_header: Sequence[str],
    batch_size: int = 256,
) -> Iterator[tuple[dict[str, ScalarCell], ...]]:
    """Read verified assignment intervals and join equal adjacent flush segments.

    Parameters
    ----------
    path : str or pathlib.Path
        policy_assignments.csv path. Only its saved durable prefix is read.
    manifest : Mapping[str, object]
        Host schema 2/scalar 14 or 15 run metadata, including Systems, per-pass trace
        epochs and trace_config_ids. Incomplete games are valid trace owners.
    expected_header : sequence of str
        Exact shared assignment header supplied by the writer/table authority.
    batch_size : int, optional
        Positive maximum output rows per tuple, default 256; booleans fail.

    Yields
    ------
    tuple of dict[str, str or int or float or None]
        Validated intervals. Stops are exclusive; policy_id -1 remains unknown.
        Adjacent rows for one pass/episode/slot and the same System and choice
        merge, including across writer flushes. Gaps stay gaps. Output follows
        interval closure order, not a global sort by episode or decision.

    Raises
    ------
    ValueError
        Schema, durable boundaries, ownership, roster, component IDs or epochs
        disagree, or one slot's intervals overlap or move backward.
    OSError
        A recorded file cannot be read.

    Notes
    -----
    Host-only and read-only. Holds at most one pending segment per recorded
    pass/episode/slot, plus bounded input/output batches; it does not retain the
    whole decision history. Consume fully to validate all rows. No interval is
    invented for a missing decision, absent trace or inactive actor. This reader
    does not prove that a declared component actually produced an action.
    """
    passes = _pass_lookup(manifest)
    pending: dict[tuple[str, str, int, int], dict[str, ScalarCell]] = {}
    output: list[dict[str, ScalarCell]] = []
    for batch in _iter_rows(
        path,
        manifest=manifest,
        expected_header=expected_header,
        batch_size=batch_size,
        assignments=True,
        passes=passes,
    ):
        for row in batch:
            key, start, stop = _assignment_values(row, manifest, passes)
            previous = pending.get(key)
            if previous is not None:
                previous_stop = cast(int, previous["decision_stop"])
                if start < previous_stop:
                    raise ValueError(
                        "policy assignment intervals overlap or move backward"
                    )
                if (
                    start == previous_stop
                    and previous["system_id"] == row["system_id"]
                    and previous["policy_id"] == row["policy_id"]
                ):
                    previous["decision_stop"] = stop
                    continue
                output.append(previous)
            pending[key] = row
            if len(output) == batch_size:
                yield tuple(output)
                output.clear()
    for row in pending.values():
        output.append(row)
        if len(output) == batch_size:
            yield tuple(output)
            output.clear()
    if output:
        yield tuple(output)
