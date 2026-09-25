"""Read a logical tournament through its exact local or immutable source rows.

TournamentRecords joins full original run/pass/episode keys to a selected logical
schedule. The runner, physical checks, writer and result views share this host
accessor. Reused wide reports stay in their original files and use the shared
indexed scalar reader. The snapshot's pinned scalar schema picks the full-report
header and the scalar version every record source must have. This module
imports no simulator, policy or JAX runtime.
"""

# Shared private scalar helpers own all original pass checks.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
    FULL_METRIC_NAMES_BY_SCHEMA_VERSION,
    METRIC_SCHEMA_VERSION,
    PRIORITY_METRIC_NAMES,
)
from marl_battlegrounds.evaluation.run_writer import IDENTITY_COLUMNS, MATCH_COLUMNS
from marl_battlegrounds.evaluation.scalar_reports import (
    IndexedScalarTable,
    OriginKey,
    _iter_rows,
    _pass_lookup,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.tournament_assets import AssetVerifier

Row = dict[str, Any]
_HEADERS = {
    "match_results.csv": MATCH_COLUMNS,
    "full_metrics.csv": (*IDENTITY_COLUMNS, *FULL_METRIC_NAMES),
    "priority_metrics.csv": (*IDENTITY_COLUMNS, *PRIORITY_METRIC_NAMES),
}


def _pinned_scalar_schema(config: Mapping[str, Any]) -> int:
    """Return the scalar schema a tournament snapshot pins for its records.

    config is the resolved format-1 tournament configuration. A config with a
    compatibility section returns its scalar_schema, which must be a key of
    FULL_METRIC_NAMES_BY_SCHEMA_VERSION (14 before the Red Zone columns, 15
    now). A config without that section, such as a narrow internal accessor,
    uses the current METRIC_SCHEMA_VERSION. Raises ValueError for an unknown
    pinned version.
    """
    compatibility = config.get("compatibility")
    if not isinstance(compatibility, Mapping):
        return METRIC_SCHEMA_VERSION
    pinned = cast(Mapping[str, Any], compatibility).get("scalar_schema")
    if type(pinned) is not int or pinned not in FULL_METRIC_NAMES_BY_SCHEMA_VERSION:
        raise ValueError("Unsupported tournament scalar_schema")
    return pinned


def origin_key(origin: Mapping[str, Any]) -> OriginKey:
    """Read the full original identity, rejecting lossy or ambiguous IDs.

    Phase, pass and run labels must be nonempty strings. Episode IDs are exact
    positive int32 values. The optional source_id is not part of physical row
    ownership; it identifies the immutable bundle that contains that row.
    """
    labels = tuple(origin.get(field) for field in ("run_id", "phase", "pass_id"))
    identifier = origin.get("episode_id")
    if any(not isinstance(value, str) or not value for value in labels) or (
        type(identifier) is not int or not 0 < identifier <= 0x7FFFFFFF
    ):
        raise ValueError(
            "tournament origin requires full labels and a positive int32 episode ID"
        )
    return cast(OriginKey, (*labels, identifier))


def origin_text(origin: Mapping[str, Any]) -> str:
    """Encode full ownership without delimiter collisions in labels."""
    return json.dumps(origin_key(origin), separators=(",", ":"), ensure_ascii=False)


class TournamentRecords:
    """Borrow one resolved schedule and read its complete original game records.

    Parameters
    ----------
    config, games, jobs : mappings and sequences
        Resolved format-1 configuration, selected logical games and local jobs.
        These must remain unchanged while the accessor is used.
    verifier : AssetVerifier
        The shared content verifier for source manifests, tables and replays.
    manifest : mapping
        Current local manifest snapshot, including its run_id and durable passes.
    run_dir : Path or None, optional
        Current saved run directory. None uses the supplied memory tables.
    memory : mapping or None, optional
        Local original table-name to row sequence. Defaults to no local rows.
        Reused reports are never copied into this mapping.

    Attributes
    ----------
    scalar_schema : int
        The snapshot's pinned scalar schema (see ``_pinned_scalar_schema``).
        Every foreign record source manifest must record this version.
    headers : dict of str to tuple of str
        Raw table headers for this snapshot. ``full_metrics.csv`` uses the
        pinned schema's saved column order, so schema-14 reports keep their
        original 11,148 metric columns.

    Notes
    -----
    The accessor caches verified manifests and narrow byte indexes. It makes no
    writer, downloads, fits or model calls. Missing required assets and changed
    content raise errors, never a partial-success result. Read methods preserve
    declared logical order while returning each row's original identity. Creating
    the accessor alone does not open or hash full-report or model payloads.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        games: Sequence[Mapping[str, Any]],
        jobs: Sequence[Mapping[str, Any]],
        verifier: AssetVerifier,
        *,
        manifest: Mapping[str, Any],
        run_dir: Path | None = None,
        memory: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        """Index narrow logical ownership, rejecting duplicate or missing jobs.

        Also reads the snapshot's pinned scalar schema; an unknown pinned
        version raises ValueError.
        """
        self.config = config
        self.scalar_schema = _pinned_scalar_schema(config)
        self.headers: dict[str, tuple[str, ...]] = {
            **_HEADERS,
            "full_metrics.csv": (
                *IDENTITY_COLUMNS,
                *FULL_METRIC_NAMES_BY_SCHEMA_VERSION[self.scalar_schema],
            ),
        }
        self.games = tuple(games)
        self.verifier = verifier
        self.manifest = manifest
        self.run_dir = run_dir
        self.memory: Mapping[str, Sequence[Mapping[str, Any]]] = (
            {} if memory is None else memory
        )
        self._sources = {
            source["source_id"]: source for source in config["record_sources"]
        }
        self._manifests: dict[str, Row] = {}
        self._pass_indexes: dict[str | None, dict[tuple[str, str], Row]] = {}
        self._completed: dict[tuple[str | None, str, str], frozenset[int]] = {}
        self._replay_checks: dict[
            tuple[str, str, str, int], tuple[int, int, int, int]
        ] = {}
        self._indexes: dict[tuple[str | None, str], IndexedScalarTable] = {}
        self._source_games: dict[str | None, list[Mapping[str, Any]]] = {}
        self._table_checks: dict[tuple[str, str], tuple[int, int, int, int]] = {}
        self._checked_full: set[int] = set()
        self._memory_indexes: dict[str, dict[OriginKey, Mapping[str, Any]]] = {}
        self._games = {game["logical_game_id"]: game for game in games}
        if len(self._games) != len(games):
            raise ValueError("logical tournament games repeat an ID")
        self._jobs: dict[int, Mapping[str, Any]] = {}
        for job in jobs:
            for identifier in job["logical_game_ids"]:
                if identifier in self._jobs or identifier not in self._games:
                    raise ValueError("execution job repeats or invents a logical game")
                self._jobs[identifier] = job
        origins: set[OriginKey] = set()
        for game in games:
            has_origin = game.get("origin") is not None
            if has_origin == (game["logical_game_id"] in self._jobs):
                raise ValueError(
                    "each logical game needs one original record or execution job"
                )
            key = origin_key(self.origin(game))
            if key in origins:
                raise ValueError(
                    "logical tournament reuses one original game more than once"
                )
            origins.add(key)
            self._source_games.setdefault(
                self.origin(game).get("source_id"), []
            ).append(game)

    def origin(self, game: Mapping[str, Any]) -> Row:
        """Return original row ownership, deriving local ownership from its job."""
        if game.get("origin") is not None:
            return dict(game["origin"])
        job = self._jobs[game["logical_game_id"]]
        return {
            "run_id": self.manifest["run_id"],
            "phase": job["phase"],
            "pass_id": job["pass_id"],
            "episode_id": game["execution"]["episode_id"],
        }

    def source_manifest(self, game: Mapping[str, Any]) -> Mapping[str, Any]:
        """Verify the originating manifest and retain its original schema/pass data.

        Parameters
        ----------
        game : Mapping[str, Any]
            One selected logical game from this accessor's schedule.

        Returns
        -------
        Mapping[str, Any]
            The run manifest that recorded the game. A local game (its origin
            has no source_id) gets the supplied current snapshot. A reused game
            gets its record source's manifest, parsed once and then cached.
            Callers must not change the returned mapping.

        Raises
        ------
        ValueError
            The game names an undeclared record source; the source's manifest
            asset is missing, changed or not a JSON object; or the manifest is
            not host schema 2 with the snapshot's pinned scalar schema
            (scalar_schema), the "marlbg.tdm.scalar" metric schema ID, no
            pending coordinated restore and the exact declared run identity.

        Notes
        -----
        Local games use the supplied current snapshot. Foreign manifests must be
        host schema 2 with the snapshot's pinned scalar schema, no pending
        coordinated restore and the exact declared run identity. A snapshot
        saved before the Red Zone rule pins scalar schema 14, so its own sources
        still pass. Historical files remain readable elsewhere but cannot
        certify new canonical fixed-team reuse. Host-only. Outside
        AssetVerifier.verification_scope, every call rechecks the manifest
        file's stamps through the verifier (and rehashes the file when they
        changed), even when its parsed content is cached. Inside a scope,
        repeated calls reuse the first check, and the file is checked for
        changes once, when the outermost scope exits.
        """
        origin = self.origin(game)
        source_id = origin.get("source_id")
        if source_id is None:
            return self.manifest
        if source_id not in self._sources:
            raise ValueError("game references an undeclared record source")
        source = self._sources[source_id]
        # verify rechecks changed file stamps even when parsed content is cached.
        self.verifier.require((source["manifest_asset"],))
        if source_id not in self._manifests:
            value = self.verifier.read_json(source["manifest_asset"])
            if not isinstance(value, dict):
                raise ValueError("record source manifest must be a JSON object")
            if (
                value.get("run_id") != source["run_id"]
                or value.get("run_id") != origin["run_id"]
                or value.get("schema_version") != 2
                or value.get("metric_schema_version") != self.scalar_schema
                or value.get("metric_schema_id") != "marlbg.tdm.scalar"
                or "recording_restore" in value
            ):
                raise ValueError(
                    "record source identity, schema or recovery state is incompatible"
                )
            self._manifests[source_id] = value
        return self._manifests[source_id]

    def entry(self, game: Mapping[str, Any]) -> Row | None:
        """Return the exact original pass, or None for a not-yet-started local job."""
        origin = self.origin(game)
        source_id = origin.get("source_id")
        manifest = self.source_manifest(game)
        if source_id not in self._pass_indexes:
            indexed = _pass_lookup(manifest)
            self._pass_indexes[source_id] = {
                key: cast(Row, value[0]) for key, value in indexed.items()
            }
            for (phase, pass_id), (_, completed) in indexed.items():
                self._completed[source_id, phase, pass_id] = completed
        return self._pass_indexes[source_id].get((origin["phase"], origin["pass_id"]))

    def completed(self, game: Mapping[str, Any]) -> bool:
        """Check actual durable completion; declarations alone are not outcomes."""
        entry = self.entry(game)
        origin = self.origin(game)
        complete = (
            entry is not None
            and origin["episode_id"]
            in self._completed[
                origin.get("source_id"), origin["phase"], origin["pass_id"]
            ]
        )
        if game.get("origin") is not None and not complete:
            raise ValueError("reused tournament game is not a durable completion")
        return complete

    def coverage(self, game: Mapping[str, Any]) -> str | None:
        """Return actual optional measurement coverage without upgrading none."""
        entry = self.entry(game)
        if entry is None:
            return None
        value = entry.get("recorded_metrics_by_episode", {}).get(
            str(self.origin(game)["episode_id"])
        )
        if value not in {None, "none", "priority", "full"}:
            raise ValueError("recorded game has an unknown metric coverage value")
        return value

    def _table_path(self, game: Mapping[str, Any], filename: str) -> Path:
        """Verify one selected immutable table prefix or return the local CSV path."""
        origin = self.origin(game)
        source_id = origin.get("source_id")
        if source_id is None:
            if self.run_dir is None:
                raise ValueError("in-memory games have no table file")
            return self.run_dir / filename
        source = self._sources[source_id]
        declaration = source["tables"].get(filename.removesuffix(".csv"))
        if declaration is None:
            raise ValueError(f"record source does not contain required {filename}")
        identifier = declaration["asset_id"]
        path = self.verifier.require((identifier,))[identifier]
        boundary = self.source_manifest(game).get("tables", {}).get(filename)
        if boundary is None or (
            boundary.get("durable_bytes"),
            boundary.get("rows"),
        ) != (declaration["committed_bytes"], declaration["rows"]):
            raise ValueError("record source table boundaries differ from its manifest")
        if path.stat().st_size != declaration["committed_bytes"]:
            raise ValueError(
                "published table asset must contain exactly its committed prefix"
            )
        stamp = path.stat()
        counters = stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns
        if self._table_checks.get((source_id, filename)) != counters:
            with path.open("rb") as stream:
                header = stream.readline(declaration["committed_bytes"])
            if hashlib.sha256(header).hexdigest() != declaration["header_sha256"]:
                raise ValueError(
                    "record source table header differs from its declaration"
                )
            self._table_checks[source_id, filename] = counters
        return path

    def _eligible(self, game: Mapping[str, Any], filename: str) -> bool:
        """Select only real completions with the requested optional table coverage."""
        return self.completed(game) and (
            filename == "match_results.csv"
            or (filename == "full_metrics.csv" and self.coverage(game) == "full")
            or (
                filename == "priority_metrics.csv"
                and self.coverage(game) in {"priority", "full"}
            )
        )

    def _index(self, game: Mapping[str, Any], filename: str) -> IndexedScalarTable:
        """Build at most one narrow selected-origin index per source and table."""
        source_id = self.origin(game).get("source_id")
        key = source_id, filename
        path = self._table_path(game, filename)
        if key not in self._indexes:
            origins = tuple(
                origin_key(self.origin(item))
                for item in self._source_games[source_id]
                if self._eligible(item, filename)
            )
            self._indexes[key] = IndexedScalarTable(
                path,
                table_name=filename,
                manifest=self.source_manifest(game),
                expected_header=self.headers[filename],
                origins=origins,
            )
        return self._indexes[key]

    def iter_rows(
        self,
        filename: str,
        *,
        game_ids: Sequence[int] | None = None,
        rows: int = 128,
        columns: Sequence[str] | None = None,
    ) -> Iterator[tuple[Row, ...]]:
        """Read completed selected games in logical order with bounded row batches.

        ``filename`` is an existing raw table role. None ``game_ids`` selects the
        entire logical schedule; explicit IDs must be unique and known. Missing
        full reports are omitted only where full capture was not recorded. Use
        ``require_coverage`` before execution when capture is required. A columns
        sequence avoids conversion of unused values; None reads the whole row.
        """
        if filename not in _HEADERS or type(rows) is not int or rows <= 0:
            raise ValueError(
                "record table and positive integer batch size are required"
            )
        ids = tuple(self._games) if game_ids is None else tuple(game_ids)
        if len(set(ids)) != len(ids) or any(
            identifier not in self._games for identifier in ids
        ):
            raise ValueError("selected logical game IDs must be unique and known")
        for start in range(0, len(ids), rows):
            with self.verifier.verification_scope():
                games = [
                    self._games[identifier]
                    for identifier in ids[start : start + rows]
                    if self._eligible(self._games[identifier], filename)
                ]
                groups: dict[str | None, list[Mapping[str, Any]]] = {}
                for game in games:
                    groups.setdefault(self.origin(game).get("source_id"), []).append(
                        game
                    )
                selected_rows: dict[OriginKey, Row] = {}
                for source_id, selected in groups.items():
                    keys = tuple(origin_key(self.origin(game)) for game in selected)
                    if source_id is None and self.run_dir is None:
                        if filename not in self._memory_indexes:
                            values = self.memory.get(filename, ())
                            indexed = {origin_key(row): row for row in values}
                            if len(indexed) != len(values):
                                raise ValueError(
                                    "local memory records repeat an origin"
                                )
                            self._memory_indexes[filename] = indexed
                        for key in keys:
                            value = self._memory_indexes[filename].get(key)
                            if value is None:
                                raise ValueError(
                                    "completed local game is missing its required row"
                                )
                            selected_rows[key] = (
                                dict(value)
                                if columns is None
                                else {name: value[name] for name in columns}
                            )
                    else:
                        index = self._index(selected[0], filename)
                        batch = next(
                            index.iter_rows(keys, batch_size=rows, columns=columns)
                        )
                        selected_rows.update(zip(keys, batch, strict=True))
            if games:
                yield tuple(
                    selected_rows[origin_key(self.origin(game))] for game in games
                )

    def require_coverage(
        self,
        *,
        metrics: str,
        full_ids: Sequence[int] = (),
        replay_ids: Sequence[int] = (),
    ) -> None:
        """Preflight durable outcomes and requested captures before new actions.

        Reused games must be complete. Already saved local games get the same
        checks; unstarted local jobs remain pending. Full coverage
        satisfies priority; ratings and headlines cannot replace raw measurements.
        Selected replay evidence is verified separately from metric availability.
        """
        if metrics not in {"priority", "full", "none"}:
            raise ValueError("metrics must be priority, full or none")
        with self.verifier.verification_scope():
            selected: list[int] = []
            for game in self.games:
                if not self.completed(game):
                    continue
                selected.append(game["logical_game_id"])
                coverage = self.coverage(game)
                if (
                    metrics == "full" or game["logical_game_id"] in full_ids
                ) and coverage != "full":
                    raise ValueError(
                        "reused game lacks required full measurements; "
                        "prepare assets or use rerun_existing=True"
                    )
                if metrics == "priority" and coverage not in {"priority", "full"}:
                    raise ValueError(
                        "reused game lacks priority measurements; "
                        "use rerun_existing=True"
                    )
                if game.get("origin") is not None or self.run_dir is not None:
                    self._table_path(game, "match_results.csv")
                    if metrics == "full" or game["logical_game_id"] in full_ids:
                        self._index(game, "full_metrics.csv")
                if game["logical_game_id"] in replay_ids:
                    path = self.replay_path(game)
                    if path is None and (
                        game.get("origin") is not None or self.run_dir is not None
                    ):
                        raise ValueError("completed game lacks its requested replay")
            required = (
                *IDENTITY_COLUMNS,
                "outcome",
                "episode_length",
                "team_a_score",
                "team_b_score",
            )
            columns = (
                tuple(dict.fromkeys((*required, *PRIORITY_METRIC_NAMES)))
                if metrics != "none"
                else required
            )
            for batch in self.iter_rows(
                "match_results.csv", game_ids=selected, columns=columns
            ):
                for row in batch:
                    for field in (
                        "outcome",
                        "episode_length",
                        "team_a_score",
                        "team_b_score",
                    ):
                        value = row.get(field)
                        if type(value) is not int or value < 0:
                            raise ValueError(
                                f"reused game requires an exact nonnegative {field}"
                            )
                    if row["outcome"] not in (1, 2, 3) or row["episode_length"] <= 0:
                        raise ValueError(
                            "reused game needs a completed outcome and positive length"
                        )
                    if metrics != "none" and any(
                        row.get(field) is None for field in PRIORITY_METRIC_NAMES
                    ):
                        raise ValueError(
                            "reused game is missing a required priority measurement"
                        )
            selected_full = [
                identifier
                for identifier in selected
                if (metrics == "full" or identifier in full_ids)
                and identifier not in self._checked_full
            ]
            # One wide row at a time checks requested stored values with the same
            # parser. Legitimately unavailable full-only cells may remain blank.
            for batch in self.iter_rows(
                "full_metrics.csv", game_ids=selected_full, rows=1
            ):
                if any(
                    row.get(field) is None
                    for row in batch
                    for field in PRIORITY_METRIC_NAMES
                ):
                    raise ValueError(
                        "full report is missing a required priority measurement"
                    )
        # Cache successful conversion only after the source boundary check passes.
        # Unchanged bytes need no second conversion of every full-report cell.
        self._checked_full.update(selected_full)

    def verify_stored_ratings(self, participants: Sequence[Mapping[str, Any]]) -> None:
        """Check each declared old Elo against its exact immutable summary row.

        Participants with null result_ref make no stored-rating claim. Required
        references are read once per source through the shared durable summary
        parser. A missing/duplicate policy row or different Elo raises ValueError
        before new actions. Old ratings remain ordering evidence, never fit credit.
        Fresh reruns need not call this method for reports being regenerated.
        """
        wanted: dict[str, list[Mapping[str, Any]]] = {}
        for participant in participants:
            reference = participant.get("result_ref")
            if reference is not None:
                if reference.get("table") != "tournament_results":
                    raise ValueError("stored rating must reference tournament_results")
                wanted.setdefault(reference["source_id"], []).append(participant)
        for source_id, entries in wanted.items():
            source = self._sources.get(source_id)
            if source is None:
                raise ValueError("stored rating references an undeclared source")
            game = {
                "origin": {
                    "source_id": source_id,
                    "run_id": source["run_id"],
                    "phase": "tournament",
                    "pass_id": "summary",
                    "episode_id": 1,
                }
            }
            path = self._table_path(game, "tournament_results.csv")
            values: dict[str, object] = {}
            for batch in _iter_rows(
                path,
                manifest=self.source_manifest(game),
                summary=True,
                table_name="tournament_results.csv",
                columns=frozenset({"policy", "elo"}),
            ):
                for row in batch:
                    name = row.get("policy")
                    if not isinstance(name, str) or name in values:
                        raise ValueError(
                            "stored ratings require unique participant rows"
                        )
                    values[name] = row.get("elo")
            for participant in entries:
                name = participant["result_ref"]["policy"]
                if (
                    participant.get("elo") is None
                    or name not in values
                    or values[name] != participant["elo"]
                ):
                    raise ValueError("snapshot Elo differs from its stored result row")

    def replay_path(self, game: Mapping[str, Any]) -> Path | None:
        """Verify a selected original replay without rewriting its identity.

        Parameters
        ----------
        game : Mapping[str, Any]
            One selected logical game from this accessor's schedule.

        Returns
        -------
        Path or None
            The verified replay file's path. For a local game, None means no
            durable replay record exists yet: none was requested, the local
            pass has not started, or the replay is not saved yet. None also
            comes back when a local replay record exists but this accessor has
            no run_dir (memory-only use). A reused game never returns None.

        Raises
        ------
        ValueError
            A reused game lacks its requested replay ("prepare assets or use
            rerun_existing=True"); a reused replay lacks exactly one original
            identity in its record source, or its asset is missing; a local
            replay path leaves its run; the file is missing, a symbolic link,
            the wrong size or changed during verification; or its content does
            not parse under the pinned replay model, is not canonical JSON, or
            differs from the game's original identity (digest, run, phase, pass,
            episode or config ID). source_manifest errors also pass through.

        Notes
        -----
        The snapshot's replay pin picks the replay model: replay V3 for
        snapshots saved before the Red Zone rule (pins 14, 2, 3) and replay V4
        for current ones. Records are never read under another version.
        Host-only: the whole file is read and parsed on the first check, and
        again only when its file stamps (inode, size, change times) differ. No
        file is written.
        """
        origin = self.origin(game)
        entry = self.entry(game)
        record = (
            None
            if entry is None
            else entry.get("replays", {}).get(str(origin["episode_id"]))
        )
        if record is None:
            if game.get("origin") is not None:
                raise ValueError(
                    "reused game lacks its requested replay; "
                    "prepare assets or use rerun_existing=True"
                )
            return None
        if game.get("origin") is not None:
            source = self._sources[origin["source_id"]]
            refs = [
                item
                for item in source["replays"]
                if origin_key(item) == origin_key(origin)
            ]
            if len(refs) != 1:
                raise ValueError("reused replay requires one exact original identity")
            identifier = refs[0]["asset_id"]
            path = self.verifier.require((identifier,))[identifier]
        else:
            if self.run_dir is None:
                return None
            relative = Path(record["path"])
            if len(relative.parts) != 2 or relative.parts[0] != "replays":
                raise ValueError("recorded replay path leaves its run")
            path = self.run_dir / relative
        if (
            path.is_symlink()
            or path.parent.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError("recorded replay is missing, changed or unsafe")
        stamp = path.stat()
        identity = origin_key(origin)
        counters = (stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns)
        if self._replay_checks.get(identity) != counters:
            from marl_battlegrounds.evaluation.models import canonical_json_bytes
            from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3
            from marl_battlegrounds.evaluation.replay_v4 import ReplayArtifactV4

            payload = path.read_bytes()
            # The snapshot's own replay pin picks the model: 3 before Red Zone,
            # 4 after. Records are never read under another version.
            model = (
                ReplayArtifactV4
                if self.config["compatibility"]["replay_schema"] == 4
                else ReplayArtifactV3
            )
            replay = model.model_validate_json(payload)
            context = replay.header.context
            labels = {item.name: item.value for item in context.aggregation_keys}
            if (
                canonical_json_bytes(replay) != payload
                or replay.canonical_digest_sha256 != record["canonical_digest_sha256"]
                or context.identity.run_id != origin["run_id"]
                or labels.get("phase") != origin["phase"]
                or labels.get("pass_id") != origin["pass_id"]
                or not context.identity.episode_id.endswith(
                    f":episode-{origin['episode_id']}"
                )
                or record.get("config_id") != game.get("resolved_config_id")
            ):
                raise ValueError(
                    "selected replay content differs from its original game identity"
                )
            after = path.stat()
            if (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != counters:
                raise ValueError("selected replay changed during verification")
            self._replay_checks[identity] = counters
        return path
