"""Read durable scalar CSV rows without changing saved files or loading JAX.

``iter_scalar_rows`` reads bounded batches from a run manifest's committed byte
range. Historical headers keep their recorded order and values. Current headers
must match the caller's shared schema. This is a low-level host reader; it does
not resume a writer, recompute metrics or qualify a tournament population.
"""

import csv
import io
import math
from collections.abc import Buffer, Iterator, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO, cast

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


def _parse_cell(name: str, value: str, *, current: bool) -> ScalarCell:
    """Restore one CSV cell, preserving blanks and historical scalar precision."""
    if value == "":
        return None
    if name in _TEXT_COLUMNS:
        return value
    try:
        if name in _INTEGER_COLUMNS or (current and name in _SUMMARY_COLUMNS):
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


def _iter_rows(
    path: str | Path,
    *,
    manifest: Mapping[str, object],
    expected_header: Sequence[str] | None = None,
    batch_size: int = 256,
    assignments: bool = False,
    passes: _Passes | None = None,
) -> Iterator[tuple[dict[str, ScalarCell], ...]]:
    """Parse shared durable rows, admitting unfinished games only for assignments.

    Public wrappers own argument/result contracts. ``assignments=True`` requires
    schema 2/14 and trace ownership instead of a completion. A supplied pass
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
        or not ((host == 1 and 1 <= scalar <= 13) or (host == 2 and scalar == 14))
        or manifest.get("metric_schema_id") != "marlbg.tdm.scalar"
    ):
        raise ValueError("unsupported host/scalar report schema combination")
    current = host == 2
    if assignments and not current:
        raise ValueError("policy assignments require host schema 2 and scalar 14")
    if current and expected_header is None:
        raise ValueError("current scalar reports require an exact expected header")
    selected = Path(path)
    tables = manifest.get("tables")
    if not isinstance(tables, Mapping):
        raise ValueError("run manifest has no valid table boundaries")
    boundary = cast(Mapping[str, object], tables).get(
        selected.name, {"durable_bytes": 0, "rows": 0}
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
    if passes is None:
        passes = _pass_lookup(manifest)
    exact_summaries = current and selected.name in {"episodes.csv", "match_results.csv"}
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
            reader = _csv_rows(text)
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
            if not {"run_id", "phase", "pass_id", "episode_id"}.issubset(header):
                raise ValueError(
                    "scalar table header is missing required ownership columns"
                )
            if current and tuple(header) != tuple(expected_header or ()):
                raise ValueError("scalar table has an incompatible current header")
            batch: list[dict[str, ScalarCell]] = []
            count = 0
            for cells in reader:
                if len(cells) != len(header):
                    raise ValueError("scalar row width differs from its header")
                row = {
                    name: _parse_cell(name, value, current=exact_summaries)
                    for name, value in zip(header, cells, strict=True)
                }
                _check_ownership(
                    row, manifest, passes, require_completion=not assignments
                )
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
        host schema 2/scalar 14. Rows must name a run, pass and completion.
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
        Host schema 2/scalar 14 run metadata, including Systems, per-pass trace
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
