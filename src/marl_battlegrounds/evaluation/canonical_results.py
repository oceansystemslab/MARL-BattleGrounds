"""Read canonical result references through the existing host result interface.

CanonicalView extends the shared table view with full original game ownership.
It reads immutable configuration/schedule references and borrows TournamentRecords
for raw rows. The full-report columns follow the snapshot's pinned scalar schema,
the same header TournamentRecords reads with, and the result's metadata reports
that pinned version as metric_schema_version. Loading summaries does not verify
unrelated model/report payloads, run a method, fit ratings, create locks or
repair files.
"""

from __future__ import annotations

# Private result methods are extended here; this is not another table API.
# pyright: reportPrivateUsage=false
import json
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from marl_battlegrounds.evaluation.metric_catalog import PRIORITY_METRIC_NAMES
from marl_battlegrounds.evaluation.models import canonical_digest_sha256
from marl_battlegrounds.evaluation.results import (
    Row,
    _cell_type,
    _mapping,
    _schema,
    _stamp,
    _View,
)
from marl_battlegrounds.evaluation.run_writer import (
    EPISODE_COLUMNS,
    IDENTITY_COLUMNS,
    MATCH_COLUMNS,
)
from marl_battlegrounds.evaluation.tournament_assets import (
    AssetVerifier,
    _hash_file,
    asset_location_config,
    policy_recording_registration,
)
from marl_battlegrounds.evaluation.tournament_config import (
    canonical_json,
    load_tournament_config,
    read_config_json,
    tournament_qualification,
)
from marl_battlegrounds.evaluation.tournament_records import (
    TournamentRecords,
    origin_key,
)
from marl_battlegrounds.evaluation.tournament_reuse import analysis_schedule

# Raw roles whose headers never depend on the scalar schema. full_metrics.csv
# is also raw; its header comes from the snapshot pin (CanonicalView._raw_header).
_RAW_HEADERS = {
    "match_results.csv": MATCH_COLUMNS,
    "priority_metrics.csv": (*IDENTITY_COLUMNS, *PRIORITY_METRIC_NAMES),
    "episodes.csv": EPISODE_COLUMNS,
}
_RAW_ROLES = frozenset({*_RAW_HEADERS, "full_metrics.csv"})


def _saved_qualification(
    manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Check every saved qualification copy against its immutable descriptor.

    manifest is the in-memory run record and config is its hash-checked saved
    descriptor (including optional challenger assets). Older narrow views may
    omit qualification metadata; they remain unverified. Conflicting release,
    budget or flag copies raise ValueError before recovery or model loading.
    Asset location hints may differ; asset content and all scientific fields may
    not. Return the shared qualification projection without editing either input.
    """
    settings = _mapping(manifest.get("details"))
    copies = [settings]
    copies.extend(
        _mapping(entry.get("details"))
        for entry in _mapping(manifest.get("passes")).values()
        if entry.get("phase") == "tournament" and entry.get("pass_id") == "schedule"
    )
    verified = settings.get("official_snapshot_verified", False)
    budget = _mapping(manifest["tournament_reuse"].get("budget"))
    decision = tournament_qualification(config, budget, official_verified=verified)
    if (
        type(budget.get("protocol_compliant")) is not bool
        or budget["protocol_compliant"] != decision["protocol_compliant"]
    ):
        raise ValueError("Saved tournament qualification differs from its budget")
    for details in copies:
        if (
            "official_snapshot_verified" in details
            and (
                type(details["official_snapshot_verified"]) is not bool
                or details["official_snapshot_verified"] != verified
            )
        ) or ("budget" in details and details["budget"] != budget):
            raise ValueError("Saved tournament qualification copies disagree")
        declared = details.get("canonical_config")
        if declared is None:
            continue
        try:
            declared = load_tournament_config(declared, official=False)
            scientific = {
                key: value
                for key, value in declared.items()
                if key not in {"assets", "source_location"}
            }
            recorded = {
                key: value
                for key, value in config.items()
                if key not in {"assets", "source_location"}
            }
            if canonical_json(scientific) != canonical_json(recorded):
                raise ValueError("Saved descriptor differs")
            for key, asset in declared["assets"].items():
                saved_asset = config["assets"].get(key, {})
                if {
                    field: value
                    for field, value in asset.items()
                    if field not in {"path", "url"}
                } != {
                    field: value
                    for field, value in saved_asset.items()
                    if field not in {"path", "url"}
                }:
                    raise ValueError("Saved asset identity differs")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Saved tournament qualification differs from its saved release record"
            ) from error
    return decision


def read_canonical_files(
    directory: Path, manifest: Mapping[str, Any]
) -> tuple[Row, tuple[Row, ...]]:
    """Read exactly the immutable config and logical plan selected by a manifest.

    Both content hashes are checked before parsing. Unknown versions, pending
    coordinated restores, unsafe paths, changed files and malformed records fail
    without changing the directory. The small descriptor and narrow schedule are
    materialized; wide reports and model assets are not opened.
    """
    _schema(manifest)
    reuse = manifest.get("tournament_reuse")
    if not isinstance(reuse, dict) or reuse.get("version") != 1:
        raise ValueError("unsupported canonical result reference version")
    for filename, key in (
        ("tournament_config.json", "config_sha256"),
        ("tournament_games.jsonl", "games_sha256"),
    ):
        path = directory / filename
        if (
            path.is_symlink()
            or not path.is_file()
            or _hash_file(path)[0] != reuse.get(key)
        ):
            raise ValueError(
                f"canonical result reference is missing or changed: {filename}"
            )
    config = read_config_json(directory / "tournament_config.json")
    games: list[Row] = []
    with (directory / "tournament_games.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("canonical logical game must be a JSON object")
            games.append(cast(Row, value))
    _saved_qualification(manifest, config)
    return config, tuple(games)


def _population_metadata(records: TournamentRecords) -> Row:
    """Project one population for direct, resumed and read-only results.

    records supplies checked run settings and immutable logical games. Keep the
    caller's original field under input_metadata. Population aliases use the
    resolved registrations and shared schedule projection, including the
    challenger. Participant IDs retain the generic writer-registration meaning;
    participant_descriptors separately hold tournament entrant IDs. Sampling
    facts come only from the saved summary, otherwise they stay unknown. This
    host-only projection loads no models or reports and changes no saved bytes.
    """
    manifest = records.manifest
    settings = _mapping(manifest.get("details"))
    aliases = deepcopy(
        {
            name: value
            for name, value in _mapping(settings.get("input_metadata")).items()
            if name not in settings
        }
    )
    for name in ("policies", "participants", "method_sampling", "schedule"):
        aliases.pop(name, None)
    aliases["num_matches"] = len(records.games)
    registrations = _mapping(settings.get("participant_registrations"))
    if registrations:
        descriptions: list[Row] = []
        participant_ids: dict[str, str] = {}
        for registration in sorted(registrations.values(), key=lambda row: row["name"]):
            description = policy_recording_registration(registration)
            participant_ids[description["name"]] = canonical_digest_sha256(description)
            if registration["kind"] == "policy":
                for field in ("kind", "components", "parameter_status"):
                    description.pop(field, None)
            descriptions.append(description)
        aliases["policies"] = descriptions
        aliases["participants"] = participant_ids
        aliases["schedule"] = [
            asdict(row)
            for row in analysis_schedule(
                records.games,
                {identifier: row["name"] for identifier, row in registrations.items()},
            )
        ]
    summary = _mapping(manifest.get("tournament_summary"))
    sampling = _mapping(summary.get("metadata")).get("method_sampling")
    if sampling is not None:
        aliases["method_sampling"] = deepcopy(sampling)
    elif registrations:
        aliases["method_sampling"] = {
            row["name"]: {
                "determinism": "unknown",
                "basis": "This result has no saved complete sampling evidence",
            }
            for row in registrations.values()
        }
    return aliases


class CanonicalView(_View):
    """Keep the normal result scope while resolving reused original game rows.

    ``memory['_record_access']`` supplies an already prepared accessor for a
    no-file result. Saved views read their two immutable reference files and
    narrow outcomes for spawn coverage at construction. Wide data stays lazy.
    Summaries use the existing durable reader and shared ranking projection.

    ``metadata['metric_schema_version']`` is the snapshot's pinned scalar
    schema, not the version in the run's own ``run_details.json``. The full
    report follows that pin, so a reader can pick the matching header with
    ``FULL_METRIC_NAMES_BY_SCHEMA_VERSION[metadata['metric_schema_version']]``.
    Example: a run that reuses a snapshot saved before the Red Zone rule
    (pins 14, 2, 3) says 14 and has the 11,148 schema-14 metric columns,
    even though its own writer recorded 15.

    Attributes
    ----------
    records : TournamentRecords
        The accessor that joins each logical game to its original rows. A
        no-file result uses the supplied one; a saved view builds one from the
        two reference files and the run's recorded asset locations.
    metadata : dict
        The shared view's metadata, with metric_schema_version replaced by the
        snapshot's pinned scalar schema (see above), plus tournament_owner (the
        run's tournament "schedule" pass), origin_join_version (1) and
        schedule_reference (the recorded SHA-256 of tournament_games.jsonl).
        List and short-config convenience settings stay available here. Complete
        population aliases describe every entrant, including the challenger;
        input_metadata retains the original field declaration. Participants map
        names to writer-registration IDs; participant_descriptors carry the
        separate tournament entrant IDs. The immutable plan supplies the schedule.
        Sampling facts come from the saved summary and otherwise stay unknown.
        Other attributes come from the shared view unchanged.
    """

    def __init__(
        self,
        manifest: Row,
        *,
        run_dir: Path | None,
        phase: str | None = None,
        pass_id: str | None = None,
        memory: Mapping[str, object] | None = None,
    ) -> None:
        """Resolve current logical scope without reading full reports or models.

        Parameters
        ----------
        manifest : Row
            The run's checked manifest. It must hold a version-1
            ``tournament_reuse`` reference.
        run_dir : pathlib.Path or None
            The saved run directory, or None for a no-file result.
        phase, pass_id : str or None, optional
            The selected scope, as for the shared view. Default None.
        memory : mapping or None, optional
            In-memory tables. A no-file result must supply
            ``memory['_record_access']``, its prepared TournamentRecords.

        Raises
        ------
        ValueError
            The reference version is unsupported, a no-file result has no
            record accessor, or a saved reference file or pin is invalid.

        Notes
        -----
        After the shared view is built, ``metadata['metric_schema_version']``
        is replaced by the snapshot's pinned scalar schema
        (``self.records.scalar_schema``). This covers saved runs,
        ``load_results`` and in-memory results alike.
        """
        reuse = _mapping(manifest.get("tournament_reuse"))
        if reuse.get("version") != 1:
            raise ValueError("unsupported canonical result reference version")
        self._reuse = reuse
        self._settings = _mapping(manifest.get("details"))
        self._plan_stamps: dict[str, tuple[int, int, int, int]] = {}
        if run_dir is None:
            record_access = None if memory is None else memory.get("_record_access")
            if not isinstance(record_access, TournamentRecords):
                raise ValueError(
                    "in-memory canonical results need their original record accessor"
                )
            self.records = record_access
        else:
            config, games = read_canonical_files(run_dir, manifest)
            self._plan_stamps = {
                name: _stamp(run_dir / name)
                for name in ("tournament_config.json", "tournament_games.jsonl")
            }
            self.records = TournamentRecords(
                config,
                games,
                reuse["execution_plan"],
                AssetVerifier(
                    asset_location_config(config, reuse.get("asset_locations", {}))
                ),
                manifest=manifest,
                run_dir=run_dir,
            )
        self._game_by_origin = {
            origin_key(self.records.origin(game)): game for game in self.records.games
        }
        super().__init__(
            manifest, run_dir=run_dir, phase=phase, pass_id=pass_id, memory=memory
        )
        self.metadata.update(
            {
                **_population_metadata(self.records),
                # The full report follows the snapshot pin, so the version a
                # researcher reads must be the pin too (14 for an old snapshot).
                **_saved_qualification(manifest, self.records.config),
                "snapshot_id": self.records.config["snapshot_id"],
                "big_12_id": self.records.config["snapshot_id"],
                "metric_schema_version": self.records.scalar_schema,
                "tournament_owner": {
                    "run_id": manifest["run_id"],
                    "phase": "tournament",
                    "pass_id": "schedule",
                },
                "origin_join_version": 1,
                "schedule_reference": reuse["games_sha256"],
            }
        )

    def _selected_game_ids(self) -> tuple[int, ...]:
        """Return logical game scope; the coordinator owns the whole population."""
        if self.whole_tournament:
            return tuple(game["logical_game_id"] for game in self.records.games)
        return tuple(
            game["logical_game_id"]
            for game in self.records.games
            if game.get("origin") is None
            and (
                self.records.origin(game)["phase"],
                self.records.origin(game)["pass_id"],
            )
            in self.identities
        )

    def _raw_header(self, filename: str) -> tuple[str, ...]:
        """Return the header of one raw role in the snapshot's pinned schema.

        filename is a key of _RAW_ROLES. full_metrics.csv uses the header that
        TournamentRecords reads with, so a snapshot pinned to scalar schema 14
        describes its original 11,148 metric columns.
        """
        if filename == "full_metrics.csv":
            return self.records.headers[filename]
        return _RAW_HEADERS[filename]

    def _header(self, filename: str) -> tuple[str, ...]:
        """Use pinned raw schemas without opening unrelated source reports."""
        return (
            self._raw_header(filename)
            if filename in _RAW_ROLES
            else super()._header(filename)
        )

    def _present(self, filename: str) -> bool:
        """Describe logical raw table roles without claiming optional capture."""
        return filename in _RAW_ROLES or super()._present(filename)

    def _describe(self, name: str) -> Row:
        """Keep table availability separate from borrowed files and required scores.

        Full-report columns follow the snapshot's pinned scalar schema.
        """
        if name not in {"episodes", "matches", "priority_metrics", "full_metrics"}:
            result = super()._describe(name)
            if (
                name == "tournament_headline_metrics"
                and self._settings["metrics"] == "none"
            ):
                result.update(
                    availability="disabled", reason="Tournament metrics were disabled"
                )
            return result
        mode = self._settings["metrics"]
        full_ids = set(self._settings["full_metrics_episodes"])
        selected = set(self._selected_game_ids())
        enabled = (
            name in {"episodes", "matches"}
            or (
                name == "priority_metrics"
                and (mode != "none" or bool(full_ids & selected))
            )
            or (
                name == "full_metrics" and (mode == "full" or bool(full_ids & selected))
            )
        )
        columns = (
            (*EPISODE_COLUMNS, "system_game_score")
            if name == "episodes"
            else MATCH_COLUMNS
            if name == "matches"
            else self._raw_header(f"{name}.csv")
        )
        return {
            "availability": "available" if enabled else "disabled",
            "reason": None if enabled else "This optional table was not selected",
            "columns": columns,
            "column_types": {
                column: _cell_type(column, name, True) for column in columns
            },
            "scope": self.scope,
            "missing_historical_fields": (),
            "unavailable_columns": {},
            "missing_passes": (),
        }

    def _raw(self, filename: str, rows: int) -> Iterator[Row]:
        """Stream original selected rows while leaving summary reading unchanged."""
        if filename in {"episodes.csv", "priority_metrics.csv"}:
            return
        if filename not in {"match_results.csv", "full_metrics.csv"}:
            yield from super()._raw(filename, rows)
            return
        ids = self._selected_game_ids()
        if filename == "full_metrics.csv" and self._settings["metrics"] != "full":
            wanted = set(self._settings["full_metrics_episodes"])
            ids = tuple(identifier for identifier in ids if identifier in wanted)
        if filename == "full_metrics.csv":
            selected = set(ids)
            for game in self.records.games:
                if (
                    game["logical_game_id"] in selected
                    and self.records.completed(game)
                    and self.records.coverage(game) != "full"
                ):
                    raise ValueError(
                        "selected completed game lacks its required full report"
                    )
        columns = None
        if filename == "match_results.csv" and self._settings["metrics"] == "none":
            columns = (
                *IDENTITY_COLUMNS,
                "block_id",
                "bootstrap_group",
                "outcome",
                "episode_length",
                "team_a_score",
                "team_b_score",
            )
        for batch in self.records.iter_rows(
            filename, game_ids=ids, rows=rows, columns=columns
        ):
            self._check_snapshot(filename)
            for row in batch:
                if filename == "match_results.csv" and columns is not None:
                    row = {name: row.get(name) for name in MATCH_COLUMNS}
                yield row

    def _check_snapshot(self, filename: str) -> None:
        """Reject a changed logical plan while allowing normal later game appends."""
        stamp = self._manifest_stamp
        super()._check_snapshot(filename)
        if self.run_dir is None:
            return
        if stamp != self._manifest_stamp:
            current = read_config_json(self.run_dir / "run_details.json")
            reuse = _mapping(current.get("tournament_reuse"))
            if {
                key: value
                for key, value in reuse.items()
                if key not in {"state", "asset_locations"}
            } != {
                key: value
                for key, value in self._reuse.items()
                if key not in {"state", "asset_locations"}
            }:
                raise ValueError("canonical result plan changed during reading")
            details = _mapping(current.get("details"))
            if any(
                details.get(key) != self._settings.get(key)
                for key in ("metrics", "full_metrics_episodes", "replay_episodes")
            ):
                raise ValueError("canonical capture settings changed during reading")
        if any(
            (self.run_dir / name).is_symlink()
            or not (self.run_dir / name).is_file()
            or _stamp(self.run_dir / name) != old
            for name, old in self._plan_stamps.items()
        ):
            read_canonical_files(self.run_dir, self.manifest)
            self._plan_stamps = {
                name: _stamp(self.run_dir / name) for name in self._plan_stamps
            }

    def _entry(self, row: Mapping[str, Any]) -> Row:
        """Find a row's complete foreign or local pass owner, including run ID."""
        game = self._game_by_origin.get(origin_key(row))
        entry = None if game is None else self.records.entry(game)
        if entry is None:
            raise ValueError("canonical row has no exact original pass owner")
        return entry

    def _episodes(self, rows: int) -> Iterator[Row]:
        """Project exact outcomes with focal score only for challenger-owned games."""
        challenger = self._reuse["challenger_id"]
        for row in self._raw("match_results.csv", rows):
            game = self._game_by_origin[origin_key(row)]
            team = (
                1
                if game["team_a"] == challenger
                else 2
                if game["team_b"] == challenger
                else None
            )
            result = {name: row.get(name) for name in EPISODE_COLUMNS}
            result["system_game_score"] = (
                None
                if team is None
                else 0.5
                if row["outcome"] == 3
                else float(row["outcome"] == team)
            )
            yield result

    def iter_rows(self, name: str, rows: int) -> Iterator[Row]:
        """Use shared projections, selecting priority from actual requested games."""
        if name != "priority_metrics":
            yield from super().iter_rows(name, rows)
            return
        if self.metadata["tables"][name]["availability"] == "disabled":
            return
        ids = self._selected_game_ids()
        if self._settings["metrics"] == "none":
            wanted = set(self._settings["full_metrics_episodes"])
            ids = tuple(identifier for identifier in ids if identifier in wanted)
        columns = (*IDENTITY_COLUMNS, *PRIORITY_METRIC_NAMES)
        for batch in self.records.iter_rows(
            "match_results.csv", game_ids=ids, rows=rows, columns=columns
        ):
            self._check_snapshot("match_results.csv")
            yield from batch

    def _spawn_balance(self) -> object:
        """Count scoped physical pairs from narrow original completed outcomes.

        A challenger is the only focal entrant; field-only views report actual
        team bindings separately. Missing referenced evidence makes coverage
        unavailable without blocking unrelated stored summaries. This reads no
        full report, model or optional metric cell and never fits statistics.
        """
        selected = set(self._selected_game_ids())
        challenger = self._reuse.get("challenger_id")
        games = [
            game for game in self.records.games if game["logical_game_id"] in selected
        ]
        if challenger is not None:
            games = [
                game for game in games if challenger in (game["team_a"], game["team_b"])
            ]
        try:
            lengths = {
                origin_key(row): row["episode_length"]
                for batch in self.records.iter_rows(
                    "match_results.csv",
                    game_ids=[game["logical_game_id"] for game in games],
                    columns=(
                        "run_id",
                        "phase",
                        "pass_id",
                        "episode_id",
                        "episode_length",
                    ),
                )
                for row in batch
            }
        except (OSError, ValueError) as error:
            return {
                "availability": "unavailable",
                "reason": str(error),
                "system_id": challenger,
                "paired_complete": None,
            }

        def summarize(
            items: list[Mapping[str, Any]],
            system_id: str | None,
            opponent_id: str | None,
        ) -> Row:
            """Summarize one exact focal or team-bound subset using logical IDs."""
            counts = dict.fromkeys(("default", "swapped", "unreported"), 0)
            steps: dict[str, int | None] = {str(key): 0 for key in counts}
            pairs: dict[object, list[Mapping[str, Any]]] = {}
            sources: dict[str, Row] = {}
            missing: list[int] = []
            missing_lengths: list[int] = []
            for game in items:
                pairs.setdefault(game["pair_id"], []).append(game)
                source = sources.setdefault(
                    game["source_config_id"],
                    {
                        "source_config_id": game["source_config_id"],
                        "map_id": game["map_id"],
                        "scheduled_games": 0,
                        "completed_games": 0,
                        "completed_steps": 0,
                    },
                )
                source["scheduled_games"] += 1
                key = origin_key(self.records.origin(game))
                if key not in lengths:
                    missing.append(game["logical_game_id"])
                    continue
                label = (
                    "default"
                    if game["spawn_locations"] == 0
                    else "swapped"
                    if game["spawn_locations"] == 1
                    else "unreported"
                )
                counts[label] += 1
                source["completed_games"] += 1
                length = lengths[key]
                if length is None:
                    steps[label] = None
                    source["completed_steps"] = None
                    missing_lengths.append(game["logical_game_id"])
                else:
                    if steps[label] is not None:
                        steps[label] = cast(int, steps[label]) + int(length)
                    if source["completed_steps"] is not None:
                        source["completed_steps"] += int(length)
            complete_pairs = sum(
                len(pair) == 2
                and {game["spawn_locations"] for game in pair} == {0, 1}
                and all(
                    origin_key(self.records.origin(game)) in lengths for game in pair
                )
                for pair in pairs.values()
            )
            return {
                "mode": "paired",
                "system_id": system_id,
                "opponent_id": opponent_id,
                "completed_games": counts,
                "completed_steps": steps,
                "expected_pairs": len(pairs),
                "completed_pairs": complete_pairs,
                "missing_episode_ids": tuple(missing),
                "missing_episode_lengths": tuple(missing_lengths),
                "paired_complete": bool(items)
                and not missing
                and complete_pairs * 2 == len(items),
                "source_coverage": tuple(sources.values()),
            }

        bindings: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for game in games:
            bindings.setdefault((game["team_a"], game["team_b"]), []).append(game)
        matchups = {
            owners: summarize(items, *owners) for owners, items in bindings.items()
        }
        if challenger is None:
            return matchups
        return {**summarize(games, challenger, None), "matchups": matchups}

    @property
    def replay_paths(self) -> tuple[Path, ...]:
        """Verify selected original replay files in logical schedule order."""
        selected = set(self._selected_game_ids()) & set(
            self._settings["replay_episodes"]
        )
        result: list[Path] = []
        for game in self.records.games:
            if game["logical_game_id"] in selected:
                path = self.records.replay_path(game)
                if path is not None:
                    result.append(path)
        return tuple(result)
