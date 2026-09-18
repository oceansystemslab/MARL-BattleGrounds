"""Resolve immutable tournament game plans without loading models or report rows.

The canonical runner uses ``resolve_reuse_plan`` after asset verification. This
module checks declared pairs, chooses balanced records, and keeps original RNG
coordinates separate from logical result IDs. It reads small JSON/JSONL schedule
assets only. The shared recording reader separately verifies actual game rows,
configurations and metric coverage before execution or publication.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from marl_battlegrounds.evaluation.tournament_config import (
    canonical_json,
    read_config_json,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.tournament_schedule import TournamentMatch

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
CHALLENGER_PLACEHOLDER = "$challenger"


def _integer(
    value: object, label: str, *, lower: int = 0, upper: int = _INT32_MAX
) -> int:
    """Return a bounded host integer, rejecting booleans and fractional values."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not lower <= int(value) <= upper
    ):
        raise ValueError(f"{label} must be an integer from {lower} to {upper}")
    return int(value)


def _text(value: object, label: str) -> str:
    """Return a nonblank string unchanged, or identify its invalid field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _object(value: object, label: str, keys: set[str] | None = None) -> dict[str, Any]:
    """Copy a JSON object and, when supplied, enforce its exact field names."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    raw = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{label} must have string keys")
    result = dict(cast(Mapping[str, Any], raw))
    if keys is not None and set(result) != keys:
        raise ValueError(f"{label} fields must be {sorted(keys)}")
    return result


def _sequence(value: object, label: str) -> list[Any]:
    """Copy a JSON array while rejecting strings, mappings and absent values."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    return list(cast(Sequence[Any], value))


def _identity(value: object, label: str) -> str | int:
    """Validate a declared pair/matchup label without coercing its JSON type."""
    if isinstance(value, str):
        return _text(value, label)
    return _integer(value, label, lower=1, upper=_UINT32_MAX)


def _asset_evidence(
    asset_id: object, assets: Mapping[str, object]
) -> dict[str, object]:
    """Resolve an asset alias to its declared digest and size, without file I/O."""
    identifier = _text(asset_id, "asset_id")
    if identifier not in assets:
        raise ValueError(f"Missing asset declaration: {identifier}")
    asset = _object(assets[identifier], f"asset {identifier}")
    digest = _text(asset.get("sha256"), f"asset {identifier} sha256")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"asset {identifier} sha256 must be lowercase SHA256")
    return {
        "sha256": digest,
        "size_bytes": _integer(
            asset.get("size_bytes"), "asset size_bytes", upper=2**63 - 1
        ),
    }


def record_source_identity(
    source: Mapping[str, object], assets: Mapping[str, object]
) -> str:
    """Hash a source-run descriptor using content references instead of paths.

    Parameters
    ----------
    source : Mapping[str, object]
        One format-1 record source. ``source_id`` is ignored while calculating
        its identity, so callers may compute the value before adding that field.
    assets : Mapping[str, object]
        Config asset declarations containing lowercase SHA256 and byte sizes.

    Returns
    -------
    str
        Lowercase SHA256 of the normalized manifest/table/replay references.
        Asset aliases and file locations do not affect this identity.

    Raises
    ------
    ValueError
        A declaration is missing, malformed or repeated. This verifies declared
        metadata only; the asset authority must separately verify file bytes.
    """
    raw = _object(source, "record source")
    if set(raw) - {"source_id", "run_id", "manifest_asset", "tables", "replays"}:
        raise ValueError("Unknown record source field")
    run_id = _text(raw.get("run_id"), "source run_id")
    tables = _object(raw.get("tables"), "source tables")
    normalized_tables: dict[str, object] = {}
    for role, value in tables.items():
        table = _object(
            value,
            f"source table {role}",
            {"asset_id", "committed_bytes", "rows", "header_sha256"},
        )
        asset = _asset_evidence(table["asset_id"], assets)
        committed = _integer(
            table["committed_bytes"], "committed_bytes", upper=2**63 - 1
        )
        if committed != asset["size_bytes"]:
            raise ValueError(
                "A source table asset must contain exactly its committed prefix"
            )
        normalized_tables[role] = {
            "asset": asset,
            "committed_bytes": committed,
            "rows": _integer(table["rows"], "table rows", upper=2**63 - 1),
            "header_sha256": _text(table["header_sha256"], "header_sha256"),
        }
    replays: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for item in _sequence(raw.get("replays"), "source replays"):
        replay = _object(
            item,
            "source replay",
            {"run_id", "phase", "pass_id", "episode_id", "asset_id"},
        )
        key = _origin_key(replay)
        if key[0] != run_id or key in seen:
            raise ValueError(
                "Replay origins must be unique and belong to their source run"
            )
        seen.add(key)
        replays.append(
            {
                "run_id": key[0],
                "phase": key[1],
                "pass_id": key[2],
                "episode_id": key[3],
                "asset": _asset_evidence(replay["asset_id"], assets),
            }
        )
    payload = {
        "run_id": run_id,
        "manifest_asset": _asset_evidence(raw.get("manifest_asset"), assets),
        "tables": normalized_tables,
        "replays": replays,
    }
    return sha256(canonical_json(payload)).hexdigest()


def _origin_key(origin: Mapping[str, object]) -> tuple[str, str, str, int]:
    """Return the complete physical row key; source aliases never replace it."""
    return (
        _text(origin.get("run_id"), "origin run_id"),
        _text(origin.get("phase"), "origin phase"),
        _text(origin.get("pass_id"), "origin pass_id"),
        _integer(origin.get("episode_id"), "origin episode_id", lower=1),
    )


def _origin(
    value: object, sources: Mapping[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """Validate an optional saved-game origin against declared source runs."""
    if value is None:
        return None
    result = _object(
        value, "origin", {"source_id", "run_id", "phase", "pass_id", "episode_id"}
    )
    source_id = _text(result["source_id"], "origin source_id")
    if source_id not in sources:
        raise ValueError(f"Unknown origin source_id: {source_id}")
    key = _origin_key(result)
    if key[0] != sources[source_id]["run_id"]:
        raise ValueError("Origin run_id differs from its source descriptor")
    return result


def _path(asset_id: object, paths: Mapping[str, Path]) -> Path:
    """Require a caller-verified asset path; never download missing schedule data."""
    identifier = _text(asset_id, "schedule asset ID")
    if identifier not in paths:
        raise ValueError(
            f"Missing verified schedule asset {identifier}; "
            "prepare tournament assets explicitly"
        )
    return Path(paths[identifier])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read strict object JSON lines with bounded line parsing and useful errors."""
    import json

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Reject duplicate object keys instead of silently retaining the last."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        """Reject JSON extensions for nonfinite numbers."""
        raise ValueError(f"Nonfinite JSON value: {value}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank schedule row at {path}:{number}")
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=object_pairs,
                    parse_constant=invalid_constant,
                )
                rows.append(_object(value, "schedule row"))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid schedule row at {path}:{number}: {error}"
                ) from error
    return rows


def _groups(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate declared execution groups and the supported extension counters."""
    result: dict[str, dict[str, Any]] = {}
    keys = {
        "group_id",
        "root_seed",
        "action_stream_version",
        "initialization_stream_version",
        "next_episode_id",
        "next_seed_id",
        "extension",
    }
    for value in _sequence(manifest.get("execution_groups"), "execution_groups"):
        group = _object(value, "execution group", keys)
        identifier = _text(group["group_id"], "group_id")
        if identifier in result:
            raise ValueError("Duplicate execution group_id")
        group["root_seed"] = _integer(
            group["root_seed"], "root_seed", upper=_UINT32_MAX
        )
        group["next_episode_id"] = _integer(
            group["next_episode_id"], "next_episode_id", lower=1, upper=_INT32_MAX + 1
        )
        group["next_seed_id"] = _integer(
            group["next_seed_id"], "next_seed_id", upper=_UINT32_MAX + 1
        )
        if group["extension"] not in ("none", "independent-pairs-v1"):
            raise ValueError("Unknown schedule extension rule")
        action_version = _text(group["action_stream_version"], "action_stream_version")
        initial_version = _text(
            group["initialization_stream_version"], "initialization_stream_version"
        )
        if action_version not in {"episode-fold-in-v1", "evaluation-systems-v1"} or (
            initial_version != action_version
        ):
            raise ValueError("Execution group needs one supported action/init protocol")
        result[identifier] = group
    if not result:
        raise ValueError("The schedule needs at least one execution group")
    return result


def _games(
    rows: Sequence[Mapping[str, object]],
    groups: Mapping[str, dict[str, Any]],
    sources: Mapping[str, dict[str, Any]],
    entrants: set[str],
    map_sources: Mapping[int, str],
    *,
    challenger_id: str | None,
    templates: bool,
) -> list[dict[str, Any]]:
    """Validate game declarations and replace only the companion's Team A token."""
    result: list[dict[str, Any]] = []
    keys = {
        "logical_game_id",
        "matchup_id",
        "pair_id",
        "pair_order",
        "team_a",
        "team_b",
        "map_id",
        "source_config_id",
        "resolved_config_id",
        "spawn_locations",
        "execution",
        "origin",
        "prior_origin",
    }
    execution_keys = {
        "group_id",
        "root_seed",
        "seed_id",
        "episode_id",
        "action_stream_version",
        "initialization_stream_version",
        "bootstrap_group",
    }
    for row in rows:
        game = _object(row, "game", keys)
        game["logical_game_id"] = _integer(
            game["logical_game_id"], "logical_game_id", lower=1
        )
        game["matchup_id"] = _identity(game["matchup_id"], "matchup_id")
        game["pair_id"] = _identity(game["pair_id"], "pair_id")
        game["pair_order"] = _integer(game["pair_order"], "pair_order")
        game["map_id"] = _integer(game["map_id"], "map_id")
        game["spawn_locations"] = _integer(
            game["spawn_locations"], "spawn_locations", upper=1
        )
        if templates:
            if (
                game["team_a"] != CHALLENGER_PLACEHOLDER
                or game["origin"] is not None
                or game["prior_origin"] is not None
            ):
                raise ValueError(
                    "Companion games need an unplayed $challenger in Team A"
                )
            game["team_a"] = challenger_id
        a, b = _text(game["team_a"], "team_a"), _text(game["team_b"], "team_b")
        if a not in entrants or b not in entrants or a == b:
            raise ValueError("Each game needs two different declared entrants")
        if (
            game["map_id"] not in map_sources
            or game["source_config_id"] != map_sources[game["map_id"]]
        ):
            raise ValueError("Game source/map differs from the configured source")
        _text(game["resolved_config_id"], "resolved_config_id")
        execution = _object(game["execution"], "game execution", execution_keys)
        identifier = _text(execution["group_id"], "execution group_id")
        if identifier not in groups:
            raise ValueError("Unknown game execution group")
        group = groups[identifier]
        execution["root_seed"] = _integer(
            execution["root_seed"], "execution root_seed", upper=_UINT32_MAX
        )
        for field in (
            "root_seed",
            "action_stream_version",
            "initialization_stream_version",
        ):
            if execution[field] != group[field] or isinstance(execution[field], bool):
                raise ValueError(f"Game {field} differs from its execution group")
        execution["seed_id"] = _integer(
            execution["seed_id"], "seed_id", upper=_UINT32_MAX
        )
        execution["episode_id"] = _integer(
            execution["episode_id"], "execution episode_id", lower=1
        )
        if execution["bootstrap_group"] is not None:
            _text(execution["bootstrap_group"], "bootstrap_group")
        game["execution"] = execution
        game["origin"] = _origin(game["origin"], sources)
        game["prior_origin"] = _origin(game["prior_origin"], sources)
        result.append(game)
    return result


def _pairs(games: Sequence[dict[str, Any]]) -> dict[str | int, list[dict[str, Any]]]:
    """Check complete fixed-team spawn pairs, unique origins and RNG dependence."""
    pairs: dict[str | int, list[dict[str, Any]]] = defaultdict(list)
    logical: set[int] = set()
    origins: set[tuple[str, str, str, int]] = set()
    matchups: dict[str | int, tuple[str, str]] = {}
    reverse_matchups: dict[tuple[str, str], str | int] = {}
    for game in games:
        if game["logical_game_id"] in logical:
            raise ValueError("Duplicate logical_game_id across schedule assets")
        logical.add(game["logical_game_id"])
        if game["origin"] is not None:
            key = _origin_key(game["origin"])
            if key in origins:
                raise ValueError("An original game cannot be counted twice")
            origins.add(key)
        team_pair = (game["team_a"], game["team_b"])
        if matchups.setdefault(game["matchup_id"], team_pair) != team_pair:
            raise ValueError("A matchup must keep fixed Team A/B ownership")
        unordered = tuple(sorted(team_pair))
        if (
            reverse_matchups.setdefault(unordered, game["matchup_id"])
            != game["matchup_id"]
        ):
            raise ValueError("One unordered matchup cannot use competing matchup IDs")
        pairs[game["pair_id"]].append(game)
    orders: set[tuple[str | int, int, int]] = set()
    seed_groups: dict[tuple[object, ...], tuple[str, object]] = {}
    for identifier, pair in pairs.items():
        if len(pair) != 2 or {row["spawn_locations"] for row in pair} != {0, 1}:
            raise ValueError("Every pair must contain exactly both spawn choices")
        first, second = pair
        for field in (
            "matchup_id",
            "pair_order",
            "team_a",
            "team_b",
            "map_id",
            "source_config_id",
        ):
            if first[field] != second[field]:
                raise ValueError(f"Pair {identifier} has different {field}")
        for field in (
            "group_id",
            "root_seed",
            "seed_id",
            "action_stream_version",
            "initialization_stream_version",
            "bootstrap_group",
        ):
            if first["execution"][field] != second["execution"][field]:
                raise ValueError(f"Pair {identifier} has different execution {field}")
        if first["execution"]["episode_id"] == second["execution"]["episode_id"]:
            raise ValueError("Paired games need distinct execution episode IDs")
        order = (first["matchup_id"], first["map_id"], first["pair_order"])
        if order in orders:
            raise ValueError("Duplicate pair_order within a matchup and map")
        orders.add(order)
        execution = first["execution"]
        coordinate = tuple(
            execution[field]
            for field in (
                "root_seed",
                "seed_id",
                "action_stream_version",
                "initialization_stream_version",
            )
        )
        unit = (
            ("pair", identifier)
            if execution["bootstrap_group"] is None
            else ("declared", execution["bootstrap_group"])
        )
        if seed_groups.setdefault(coordinate, unit) != unit:
            raise ValueError(
                "Repeated original RNG coordinates require one declared bootstrap_group"
            )
    return dict(pairs)


def _cells(
    pairs: Mapping[str | int, list[dict[str, Any]]],
) -> dict[tuple[str, str, int], list[list[dict[str, Any]]]]:
    """Group complete pairs by unordered matchup/map and sort declared pair order."""
    cells: dict[tuple[str, str, int], list[list[dict[str, Any]]]] = defaultdict(list)
    for pair in pairs.values():
        first = pair[0]
        a, b = sorted((first["team_a"], first["team_b"]))
        cells[(a, b, first["map_id"])].append(pair)
    for values in cells.values():
        values.sort(key=lambda pair: pair[0]["pair_order"])
    return dict(cells)


def selection_unit_id(game: Mapping[str, Any]) -> str:
    """Name one whole selection unit without changing statistical dependence.

    A non-null ``game['execution']['bootstrap_group']`` names a declared coupled
    unit. Otherwise the complete spawn pair is one independent unit. The returned
    compact JSON string tags the namespace, so a group and pair with equal labels
    cannot collide. These references are used only in ``selection_blocks``;
    original bootstrap groups, pair IDs and execution coordinates stay unchanged.
    """
    group = game["execution"]["bootstrap_group"]
    return canonical_json(
        ["pair", game["pair_id"]] if group is None else ["group", group]
    ).decode("utf-8")


def _extend(
    games: list[dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    cells: Mapping[tuple[str, str, int], list[list[dict[str, Any]]]],
    target: int,
    allocation_order: Sequence[str],
    map_order: Sequence[int],
    *,
    reserved_games: Sequence[dict[str, Any]] = (),
) -> None:
    """Append whole independent rounds using declared counters and known configs."""
    rounds = max(target - len(pairs) for pairs in cells.values())
    if rounds <= 0:
        return
    if any(game["execution"]["bootstrap_group"] is not None for game in games):
        raise ValueError("A coupled schedule cannot use independent-pairs-v1 extension")
    declared = [*games, *reserved_games]
    next_logical = max(game["logical_game_id"] for game in declared) + 1
    if next_logical + 2 * rounds * len(cells) - 1 > _INT32_MAX:
        raise ValueError("Extended logical game IDs exceed int32")
    for group_id, group in groups.items():
        owned = [
            game["execution"]
            for game in declared
            if game["execution"]["group_id"] == group_id
        ]
        if owned and (
            group["next_episode_id"] <= max(item["episode_id"] for item in owned)
            or group["next_seed_id"] <= max(item["seed_id"] for item in owned)
        ):
            raise ValueError(
                "Extension counters overlap declared execution coordinates"
            )
    for round_index in range(rounds):
        for a, b in combinations(allocation_order, 2):
            for map_id in map_order:
                first, second = sorted((a, b))
                cell = cells[(first, second, map_id)]
                template = cell[-1]
                group = groups[template[0]["execution"]["group_id"]]
                if group["extension"] != "independent-pairs-v1":
                    raise ValueError(
                        "Requested budget exceeds coverage; this schedule cannot extend"
                    )
                episode_id, seed_id = group["next_episode_id"], group["next_seed_id"]
                if episode_id + 1 > _INT32_MAX or seed_id > _UINT32_MAX:
                    raise ValueError(
                        "Extended execution coordinates exceed int32/uint32 limits"
                    )
                order = template[0]["pair_order"] + round_index + 1
                if order > _INT32_MAX:
                    raise ValueError("Extended pair_order exceeds int32")
                pair_id = (
                    "extended:"
                    + sha256(
                        canonical_json([template[0]["matchup_id"], map_id, order])
                    ).hexdigest()
                )
                for side, row in enumerate(
                    sorted(template, key=lambda item: item["spawn_locations"])
                ):
                    new = deepcopy(row)
                    new.update(
                        logical_game_id=next_logical,
                        pair_id=pair_id,
                        pair_order=order,
                        origin=None,
                        prior_origin=None,
                    )
                    new["execution"].update(
                        episode_id=episode_id + side, seed_id=seed_id
                    )
                    games.append(new)
                    next_logical += 1
                group["next_episode_id"] += 2
                group["next_seed_id"] += 1


def _select(
    pairs: Mapping[str | int, list[dict[str, Any]]],
    cells: Mapping[tuple[str, str, int], list[list[dict[str, Any]]]],
    target: int,
    selection_blocks: object,
    known_units: set[str] | None = None,
) -> set[str | int]:
    """Select outcome-independent per-cell prefixes or complete balanced blocks."""
    if selection_blocks is None:
        if any(
            pair[0]["execution"]["bootstrap_group"] is not None
            for pair in pairs.values()
        ):
            raise ValueError(
                "Coupled schedules need declared balanced selection_blocks"
            )
        if any(len(value) < target for value in cells.values()):
            raise ValueError(
                "Requested coverage is unavailable; use rerun_existing=True "
                "for a declared fresh schedule"
            )
        return {
            pair[0]["pair_id"] for values in cells.values() for pair in values[:target]
        }
    groups: dict[str, list[str | int]] = defaultdict(list)
    for pair_id, pair in pairs.items():
        groups[selection_unit_id(pair[0])].append(pair_id)
    declared_units = set(groups) if known_units is None else known_units
    selected: set[str | int] = set()
    used: set[str] = set()
    count = 0
    reached: set[str | int] | None = None
    for raw in _sequence(selection_blocks, "selection_blocks"):
        block = _sequence(raw, "selection block")
        if not block or any(not isinstance(value, str) for value in block):
            raise ValueError("A selection block must list complete group IDs")
        additions: dict[tuple[str, str, int], int] = defaultdict(int)
        for identifier in block:
            if identifier in used or identifier not in declared_units:
                raise ValueError("Selection groups must be known and occur once")
            used.add(identifier)
            for pair_id in groups.get(identifier, ()):
                selected.add(pair_id)
                first = pairs[pair_id][0]
                a, b = sorted((first["team_a"], first["team_b"]))
                additions[(a, b, first["map_id"])] += 1
        counts = {additions.get(cell, 0) for cell in cells}
        if len(counts) != 1 or 0 in counts:
            raise ValueError(
                "Each selection block must add equal positive matchup/map coverage"
            )
        count += counts.pop()
        if count == target:
            reached = set(selected)
    if used != declared_units:
        raise ValueError("selection_blocks must cover every declared coupled group")
    if reached is None:
        raise ValueError("No whole selection-block prefix reaches the requested budget")
    return reached


def _jobs(games: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Group fresh games without changing roots, team ownership or episode IDs."""
    jobs: list[dict[str, Any]] = []
    active: dict[tuple[object, ...], tuple[dict[str, Any], set[int]]] = {}
    for game in games:
        if game["origin"] is not None:
            continue
        execution = game["execution"]
        key = (
            execution["group_id"],
            execution["root_seed"],
            execution["action_stream_version"],
            execution["initialization_stream_version"],
            game["team_a"],
            game["team_b"],
        )
        current = active.get(key)
        if current is None or execution["episode_id"] in current[1]:
            identifier = len(jobs)
            job: dict[str, Any] = {
                "job_id": identifier,
                "group_id": execution["group_id"],
                "root_seed": execution["root_seed"],
                "team_a": game["team_a"],
                "team_b": game["team_b"],
                "phase": "tournament",
                "pass_id": f"canonical-job-{identifier}",
                "logical_game_ids": [],
            }
            current = (job, set[int]())
            active[key] = current
            jobs.append(job)
        current[0]["logical_game_ids"].append(game["logical_game_id"])
        current[1].add(execution["episode_id"])
    return tuple(jobs)


@dataclass(frozen=True)
class ReusePlan:
    """One checked host schedule, containing copied JSON-friendly records.

    Attributes
    ----------
    games : tuple of dict
        Selected games in declared logical order. Original IDs and RNG values
        stay in each execution/origin record. Treat nested mappings as read-only.
    jobs : tuple of dict
        Fresh execution passes. Reused games never appear in a job.
    source_descriptors : tuple of dict
        Declared immutable source runs. Actual report contents need a separate
        shared-reader preflight before a caller may act on this plan.
    budget : dict
        Official/resolved per-opponent counts, override and protocol flags.
    schedule_manifest : dict
        A copied manifest, including advanced counters after explicit extension.
    participant_ids : tuple of str
        Incumbent order, followed by the challenger when present.

    Notes
    -----
    This object neither writes files nor owns models. Its records are JSON-ready
    copies, not a certificate that source CSV measurements exist or are correct.
    """

    games: tuple[dict[str, Any], ...]
    jobs: tuple[dict[str, Any], ...]
    source_descriptors: tuple[dict[str, Any], ...]
    budget: dict[str, Any]
    schedule_manifest: dict[str, Any]
    participant_ids: tuple[str, ...]

    def capture_ids(self, logical_ids: Sequence[int]) -> dict[int, tuple[int, ...]]:
        """Map selected logical IDs to fresh jobs' original episode IDs.

        ``logical_ids`` must contain distinct scheduled positive integers. Reused
        selections are valid but absent from the return: their assets are checked
        through source origins. Every job is present with an empty tuple when it
        has no selection. This host method never changes the plan or records.
        """
        selected = [
            _integer(value, "capture logical ID", lower=1) for value in logical_ids
        ]
        game_by_id = {game["logical_game_id"]: game for game in self.games}
        if (
            len(set(selected)) != len(selected)
            or not set(selected) <= game_by_id.keys()
        ):
            raise ValueError(
                "Capture IDs must be distinct games in the resolved schedule"
            )
        wanted = set(selected)
        return {
            job["job_id"]: tuple(
                game_by_id[identifier]["execution"]["episode_id"]
                for identifier in job["logical_game_ids"]
                if identifier in wanted
            )
            for job in self.jobs
        }


def resolve_reuse_plan(
    config: Mapping[str, object],
    asset_paths: Mapping[str, Path],
    *,
    challenger_id: str | None = None,
    games_per_opponent: int | None = None,
    rerun_existing: bool = False,
    require_reuse: bool | None = None,
    official_verified: bool = False,
) -> ReusePlan:
    """Resolve balanced saved/fresh games using an already checked config.

    Parameters
    ----------
    config : Mapping[str, object]
        Format-1 descriptor checked by the shared configuration authority.
    asset_paths : Mapping[str, pathlib.Path]
        Paths whose bytes were checked by the asset authority. This function
        reads only the schedule manifest and game JSONL assets.
    challenger_id : str | None
        Distinct entrant ID to place in every companion game's Team A. None
        selects the configured population alone.
    games_per_opponent : int | None
        Equal completed-game count for every unordered matchup. None inherits
        the config. Values must be positive, nonboolean integers divisible by
        twice the map count.
    rerun_existing : bool
        False reuses declared incumbent origins. True makes every selected game
        fresh, retains ancestry, and permits the declared independent extension.
    require_reuse : bool | None
        Private orchestration assertion. True requires every incumbent game to
        have an origin unless fresh execution was requested. None applies that
        rule to released configs; admission uses True for provisional configs.
        False permits fresh custom populations. It does not alter public APIs.
    official_verified : bool, default=False
        Private resolver evidence that the selected release passed its installed
        approval pin, or inherited that verified fact from a saved run. A release
        label alone cannot establish protocol compliance. Custom and provisional
        admission callers leave this False; no public trust override is added.

    Returns
    -------
    ReusePlan
        Copied selected records, fresh jobs, sources and budget metadata. This
        does not load models, inspect CSV reports, create files or draw randomness.

    Raises
    ------
    ValueError
        Schedule structure, origins, pairs, budgets, stream declarations or
        extension limits are invalid. Official missing reuse coverage requires
        an explicit fresh run; there is no automatic execution fallback.
    OSError
        A caller-provided schedule asset cannot be read.
    """
    if not isinstance(cast(object, rerun_existing), bool):
        raise ValueError("rerun_existing must be a boolean")
    if require_reuse is not None and not isinstance(cast(object, require_reuse), bool):
        raise ValueError("require_reuse must be a boolean or None")
    if not isinstance(cast(object, official_verified), bool):
        raise ValueError("official_verified must be a boolean")
    raw = deepcopy(dict(config))
    conditions = _object(raw.get("conditions"), "conditions")
    _object(raw.get("compatibility"), "compatibility")
    assets = _object(raw.get("assets"), "assets")
    participants = [
        _object(value, "participant")
        for value in _sequence(raw.get("participants"), "participants")
    ]
    entrant_ids = [
        _text(value.get("entrant_id"), "entrant_id") for value in participants
    ]
    if (
        len(entrant_ids) < 2
        or len(set(entrant_ids)) != len(entrant_ids)
        or CHALLENGER_PLACEHOLDER in entrant_ids
    ):
        raise ValueError("A tournament needs at least two distinct entrant IDs")
    if challenger_id is not None:
        _text(challenger_id, "challenger_id")
        if challenger_id in entrant_ids or challenger_id == CHALLENGER_PLACEHOLDER:
            raise ValueError("The challenger must have a distinct real entrant ID")
        entrant_ids.append(challenger_id)
    map_sources: dict[int, str] = {}
    for value in _sequence(conditions.get("map_sources"), "map_sources"):
        source = _object(value, "map source")
        map_id = _integer(source.get("map_id"), "map_id")
        if map_id in map_sources:
            raise ValueError("Repeated source map ID")
        map_sources[map_id] = _text(source.get("source_config_id"), "source_config_id")
    if not map_sources:
        raise ValueError("The tournament needs at least one map")
    original_budget = _integer(
        conditions.get("games_per_opponent"), "configured games_per_opponent", lower=1
    )
    budget = (
        original_budget
        if games_per_opponent is None
        else _integer(games_per_opponent, "games_per_opponent", lower=1)
    )
    if original_budget % (2 * len(map_sources)) or budget % (2 * len(map_sources)):
        raise ValueError("games_per_opponent must be divisible by twice the map count")
    if len(entrant_ids) * (len(entrant_ids) - 1) // 2 * budget > _INT32_MAX:
        raise ValueError("Resolved tournament exceeds int32 logical game capacity")
    sources: dict[str, dict[str, Any]] = {}
    for value in _sequence(raw.get("record_sources"), "record_sources"):
        source = _object(value, "record source")
        identifier = _text(source.get("source_id"), "source_id")
        if (
            identifier in sources
            or record_source_identity(source, assets) != identifier
        ):
            raise ValueError("Duplicate or mismatching record source identity")
        sources[identifier] = source
    manifest = deepcopy(
        read_config_json(_path(conditions.get("schedule_asset"), asset_paths))
    )
    allowed = {
        "format",
        "version",
        "games_asset",
        "challenger_games_asset",
        "execution_groups",
        "selection_blocks",
        "allocation_order",
    }
    if (
        set(manifest) != allowed
        or manifest.get("format") != "marlbg-tournament-schedule"
    ):
        raise ValueError("Expected exact tournament schedule format 1")
    _integer(manifest.get("version"), "schedule version", lower=1, upper=1)
    groups = _groups(manifest)
    allocation = [
        _text(value, "allocation_order entrant")
        for value in _sequence(manifest["allocation_order"], "allocation_order")
    ]
    incumbents = entrant_ids if challenger_id is None else entrant_ids[:-1]
    if len(allocation) != len(set(allocation)) or set(allocation) != set(incumbents):
        raise ValueError("allocation_order must list every configured entrant once")
    if challenger_id is not None:
        allocation.append(challenger_id)
    games = _games(
        _read_jsonl(_path(manifest["games_asset"], asset_paths)),
        groups,
        sources,
        set(incumbents),
        map_sources,
        challenger_id=None,
        templates=False,
    )
    companion: list[dict[str, Any]] = []
    if challenger_id is not None or (
        (manifest["selection_blocks"] is not None or rerun_existing)
        and manifest["challenger_games_asset"] is not None
    ):
        template_id = CHALLENGER_PLACEHOLDER if challenger_id is None else challenger_id
        companion = _games(
            _read_jsonl(_path(manifest["challenger_games_asset"], asset_paths)),
            groups,
            sources,
            {*incumbents, template_id},
            map_sources,
            challenger_id=template_id,
            templates=True,
        )
    all_pairs = _pairs([*games, *companion])
    known_units = {selection_unit_id(pair[0]) for pair in all_pairs.values()}
    if challenger_id is not None:
        games.extend(companion)
    pairs = _pairs(games)
    cells = _cells(pairs)
    expected = {
        (*sorted((a, b)), map_id)
        for a, b in combinations(entrant_ids, 2)
        for map_id in map_sources
    }
    if set(cells) != expected:
        raise ValueError(
            "Schedule must cover every unordered matchup and configured map"
        )
    target = budget // (2 * len(map_sources))
    if any(len(value) < target for value in cells.values()) and rerun_existing:
        if manifest["selection_blocks"] is not None:
            raise ValueError(
                "This coupled schedule cannot extend; "
                "supply compatible declared coverage"
            )
        _extend(
            games,
            groups,
            cells,
            target,
            allocation,
            tuple(map_sources),
            reserved_games=companion if challenger_id is None else (),
        )
        pairs = _pairs(games)
        cells = _cells(pairs)
    selected_pairs = _select(
        pairs, cells, target, manifest["selection_blocks"], known_units
    )
    selected = [game for game in games if game["pair_id"] in selected_pairs]
    official = raw.get("release") is not None
    reuse_required = official if require_reuse is None else require_reuse
    for game in selected:
        if rerun_existing:
            if game["origin"] is not None:
                game["prior_origin"] = game["origin"]
            game["origin"] = None
        elif (
            reuse_required
            and game["team_a"] != challenger_id
            and game["team_b"] != challenger_id
            and game["origin"] is None
        ):
            raise ValueError(
                "Incumbent records are missing; explicitly use "
                "rerun_existing=True for fresh execution"
            )
    manifest["execution_groups"] = list(groups.values())
    return ReusePlan(
        tuple(selected),
        _jobs(selected),
        tuple(sources.values()),
        {
            "official_games_per_opponent": original_budget,
            "resolved_games_per_opponent": budget,
            "budget_override": budget != original_budget,
            "protocol_compliant": official_verified
            and official
            and budget == original_budget,
        },
        manifest,
        tuple(entrant_ids),
    )


def analysis_schedule(
    plan: ReusePlan, participant_names: Mapping[str, str] | None = None
) -> tuple[TournamentMatch, ...]:
    """Project verified logical records into the existing statistical schedule.

    Parameters
    ----------
    plan : ReusePlan
        Declared plan. The runner must verify physical/report evidence before
        fitting; this projection alone does not certify a completed population.
    participant_names : Mapping[str, str] | None
        Optional complete entrant-ID to distinct output-label mapping. None uses
        entrant IDs directly, which avoids confusing labels from foreign runs.

    Returns
    -------
    tuple of TournamentMatch
        Same game order with logical row/block/seed IDs. Seed interning includes
        the original root and stream versions; it never changes execution RNG.

    Raises
    ------
    ValueError
        Labels are incomplete/nonunique or logical block/seed limits are exceeded.
    """
    from marl_battlegrounds.evaluation.tournament_schedule import TournamentMatch

    labels = (
        {identifier: identifier for identifier in plan.participant_ids}
        if participant_names is None
        else dict(participant_names)
    )
    if set(labels) != set(plan.participant_ids) or len(set(labels.values())) != len(
        labels
    ):
        raise ValueError("Analysis labels must cover every entrant distinctly")
    for value in labels.values():
        _text(value, "analysis label")
    blocks: dict[str | int, int] = {}
    seeds: dict[tuple[object, ...], int] = {}
    result: list[TournamentMatch] = []
    for game in plan.games:
        execution = game["execution"]
        block = blocks.setdefault(game["pair_id"], len(blocks) + 1)
        coordinate = tuple(
            execution[field]
            for field in (
                "root_seed",
                "seed_id",
                "action_stream_version",
                "initialization_stream_version",
            )
        )
        seed = seeds.setdefault(coordinate, len(seeds))
        if block > _UINT32_MAX or seed > _UINT32_MAX:
            raise ValueError("Logical statistics coordinates exceed uint32")
        result.append(
            TournamentMatch(
                game["logical_game_id"],
                block,
                seed,
                game["map_id"],
                labels[game["team_a"]],
                labels[game["team_b"]],
                bootstrap_group=execution["bootstrap_group"],
                pairing_protocol="fixed-team-spawn-v1",
                spawn_locations=game["spawn_locations"],
                source_config_id=game["source_config_id"],
                resolved_config_id=game["resolved_config_id"],
            )
        )
    return tuple(result)


def rebuild_companion(
    plan: ReusePlan,
    retained_games: Sequence[Mapping[str, object]],
    participant_ids: Sequence[str],
    *,
    games_asset: str,
    challenger_games_asset: str,
    map_ids: Sequence[int],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Prepare a promoted population's next companion without rewriting history.

    Parameters
    ----------
    plan : ReusePlan
        Completed challenger comparison, with its original manifest counters.
        Physical evidence must already have passed the runner's shared checks.
    retained_games : Sequence[Mapping[str, object]]
        Exactly the surviving population's complete games. The maintainer must
        attach actual durable origins to newly executed games before this call.
        Old and new origins and execution coordinates remain unchanged.
    participant_ids : Sequence[str]
        Complete new population. It must remove exactly one prior incumbent and
        retain the previous challenger. Presentation/Elo order does not control
        RNG allocation; survivors keep their prior allocation order.
    games_asset, challenger_games_asset : str
        Distinct asset aliases the caller will use for the returned JSONL data.
    map_ids : Sequence[int]
        Exact configured map order, used for deterministic template allocation.

    Returns
    -------
    tuple[dict, tuple[dict, ...], tuple[dict, ...]]
        New schedule manifest, unchanged copied retained rows, and unplayed
        companion rows with ``$challenger`` in Team A. The caller owns immutable
        asset writing and the atomic membership commit; this function does no I/O.

    Raises
    ------
    ValueError
        The population/retained coverage is wrong, original rows changed, origins
        are missing, or declared extension rules/counters cannot allocate the
        next comparison. No new seed root or coupling rule is invented.

    Notes
    -----
    Call only after promotion. A failed or nonpromoted challenger leaves the
    population and its existing companion unchanged. Retrying that comparison
    keeps its saved schedule. This helper creates no global per-attempt allocator.
    """
    ids = [_text(value, "retained entrant ID") for value in participant_ids]
    old_order = list(plan.schedule_manifest["allocation_order"])
    previous_challengers = set(plan.participant_ids) - set(old_order)
    if (
        len(previous_challengers) != 1
        or len(ids) != len(set(ids))
        or len(ids) != len(old_order)
        or not set(ids) <= set(plan.participant_ids)
        or not previous_challengers <= set(ids)
        or len(set(old_order) - set(ids)) != 1
    ):
        raise ValueError(
            "Promotion must retain the challenger and all but one incumbent"
        )
    challenger = next(iter(previous_challengers))
    maps = [_integer(value, "retained map ID") for value in map_ids]
    if not maps or len(maps) != len(set(maps)):
        raise ValueError("Configured map order must be nonempty and distinct")
    _text(games_asset, "games_asset")
    _text(challenger_games_asset, "challenger_games_asset")
    if games_asset == challenger_games_asset:
        raise ValueError("Retained games and companion need different asset aliases")
    retained = [deepcopy(dict(row)) for row in retained_games]
    source_rows = {row["logical_game_id"]: row for row in plan.games}
    expected_ids = {
        row["logical_game_id"]
        for row in plan.games
        if row["team_a"] in ids and row["team_b"] in ids
    }
    if {row.get("logical_game_id") for row in retained} != expected_ids:
        raise ValueError("Retained games must cover exactly the surviving population")
    for row in retained:
        original = source_rows[row["logical_game_id"]]
        if row.get("origin") is None:
            raise ValueError("Every retained game needs its actual durable origin")
        _origin_key(_object(row["origin"], "retained origin"))
        for key, value in original.items():
            if key != "origin" and row.get(key) != value:
                raise ValueError(f"A retained game changed its original {key}")
        if original["origin"] is not None and row["origin"] != original["origin"]:
            raise ValueError("A reused game's original origin cannot change")
    pairs = _pairs(retained)
    cells = _cells(pairs)
    expected_cells = {
        (*sorted((a, b)), map_id) for a, b in combinations(ids, 2) for map_id in maps
    }
    count = plan.budget["resolved_games_per_opponent"] // (2 * len(maps))
    if set(cells) != expected_cells or {len(value) for value in cells.values()} != {
        count
    }:
        raise ValueError("Retained games must have complete equal matchup/map coverage")
    manifest = deepcopy(plan.schedule_manifest)
    groups = {row["group_id"]: row for row in manifest["execution_groups"]}
    for group_id, group in groups.items():
        executions = [
            row["execution"]
            for row in plan.games
            if row["execution"]["group_id"] == group_id
        ]
        if executions and (
            group["next_episode_id"] <= max(row["episode_id"] for row in executions)
            or group["next_seed_id"] <= max(row["seed_id"] for row in executions)
        ):
            raise ValueError("Companion allocation counters overlap previous games")
    order = [identifier for identifier in old_order if identifier in ids]
    order.append(challenger)
    next_logical = max(row["logical_game_id"] for row in plan.games) + 1
    required_rows = len(ids) * len(maps) * count * 2
    if next_logical + required_rows - 1 > _INT32_MAX:
        raise ValueError("New companion logical IDs exceed int32")
    templates: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in plan.games:
        if row["team_a"] == challenger:
            templates[(row["team_b"], row["map_id"])].setdefault(
                row["spawn_locations"], row
            )
    companion: list[dict[str, Any]] = []
    generation = next_logical
    for entrant in order:
        matchup = (
            "companion:"
            + sha256(canonical_json([order, entrant, generation])).hexdigest()
        )
        for map_id in maps:
            template = templates.get((entrant, map_id))
            if entrant == challenger:
                template = next(
                    (
                        value
                        for (_, source_map), value in templates.items()
                        if source_map == map_id
                    ),
                    None,
                )
            if template is None or set(template) != {0, 1}:
                raise ValueError(
                    "The prior companion lacks a required source/map template"
                )
            if template[0]["execution"]["bootstrap_group"] is not None:
                raise ValueError(
                    "A coupled challenger template needs an approved extension rule"
                )
            for pair_order in range(count):
                group = groups[template[0]["execution"]["group_id"]]
                if group["extension"] != "independent-pairs-v1":
                    raise ValueError(
                        "The declared group cannot allocate a new companion"
                    )
                episode_id, seed_id = group["next_episode_id"], group["next_seed_id"]
                if episode_id + 1 > _INT32_MAX or seed_id > _UINT32_MAX:
                    raise ValueError("New companion execution IDs exceed int32/uint32")
                pair_id = (
                    "companion:"
                    + sha256(
                        canonical_json(
                            [
                                entrant,
                                map_id,
                                pair_order,
                                group["group_id"],
                                episode_id,
                                seed_id,
                            ]
                        )
                    ).hexdigest()
                )
                for spawn in (0, 1):
                    row = deepcopy(template[spawn])
                    row.update(
                        logical_game_id=next_logical,
                        matchup_id=matchup,
                        pair_id=pair_id,
                        pair_order=pair_order,
                        team_a=CHALLENGER_PLACEHOLDER,
                        team_b=entrant,
                        origin=None,
                        prior_origin=None,
                    )
                    row["execution"].update(
                        episode_id=episode_id + spawn, seed_id=seed_id
                    )
                    companion.append(row)
                    next_logical += 1
                group["next_episode_id"] += 2
                group["next_seed_id"] += 1
    manifest.update(
        games_asset=games_asset,
        challenger_games_asset=challenger_games_asset,
        allocation_order=order,
    )
    if manifest["selection_blocks"] is not None:
        manifest["selection_blocks"] = _promoted_selection_blocks(
            manifest["selection_blocks"], retained, companion, maps, order, count
        )
    _pairs([*retained, *companion])
    return manifest, tuple(retained), tuple(companion)


def _promoted_selection_blocks(
    blocks: Sequence[Sequence[str]],
    retained: Sequence[dict[str, Any]],
    companion: Sequence[dict[str, Any]],
    maps: Sequence[int],
    entrants: Sequence[str],
    expected_pairs: int,
) -> list[list[str]]:
    """Keep retained units and add equal independent companion coverage per block.

    Population removal can shrink a declared coupled unit but never changes its
    name or splits its retained members between blocks. New pair references are
    added in entrant/map/order sequence. This only builds selection references;
    existing statistical dependence and original execution remain untouched.
    """
    retained_pairs = _pairs(retained)
    units: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for pair in retained_pairs.values():
        units[selection_unit_id(pair[0])].append(pair)
    companion_pairs = _pairs(companion)
    next_by_cell: dict[tuple[str, int], dict[int, str]] = defaultdict(dict)
    for pair in companion_pairs.values():
        first = pair[0]
        next_by_cell[(first["team_b"], first["map_id"])][first["pair_order"]] = (
            selection_unit_id(first)
        )
    expected_cells = set(_cells(retained_pairs))
    result: list[list[str]] = []
    completed = 0
    used: set[str] = set()
    for block in blocks:
        surviving = [identifier for identifier in block if identifier in units]
        additions: dict[tuple[str, str, int], int] = defaultdict(int)
        for identifier in surviving:
            if identifier in used:
                raise ValueError("A retained selection unit occurs in multiple blocks")
            used.add(identifier)
            for pair in units[identifier]:
                first = pair[0]
                a, b = sorted((first["team_a"], first["team_b"]))
                additions[(a, b, first["map_id"])] += 1
        if not surviving:
            continue
        increments = {additions.get(cell, 0) for cell in expected_cells}
        if len(increments) != 1 or 0 in increments:
            raise ValueError("Retained coupled blocks no longer have balanced coverage")
        increment = increments.pop()
        stop = completed + increment
        if stop > expected_pairs:
            raise ValueError("Retained selection blocks exceed the completed budget")
        extended = list(surviving)
        for entrant in entrants:
            for map_id in maps:
                available = next_by_cell[(entrant, map_id)]
                if any(index not in available for index in range(completed, stop)):
                    raise ValueError("New companion cannot cover a retained block")
                extended.extend(available[index] for index in range(completed, stop))
        result.append(extended)
        completed = stop
    if completed != expected_pairs or used != set(units):
        raise ValueError("Retained selection blocks do not cover every retained pair")
    return result
