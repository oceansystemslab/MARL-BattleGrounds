"""Resolve immutable tournament descriptions before loading models or opening runs.

The configuration, asset and canonical runners share these host-only checks.
``load_tournament_config`` reads a JSON path or copies a parsed description;
``resolve_tournament_config`` applies saved-first resume. Neither function reads
payload assets, imports JAX, executes factories, opens a writer or changes files.
Configuration integrity is separate from physical game and controller evidence.
Version 1 preserves twelve-entry releases and variable-size custom fields.
Version 2 also permits releases with any frozen population of at least two.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from hashlib import sha256
from importlib.resources import files
from math import isfinite
from pathlib import Path
from typing import Any, cast

CONFIG_FORMAT = "marlbg-tournament-config"
CONFIG_VERSION = 1
INLINE_METADATA_ROLES = frozenset(
    {"registration", "configuration", "schedule", "dependencies", "qualification"}
)
ASSET_ROLES = frozenset(
    {
        "model",
        "registration",
        "configuration",
        "schedule",
        "run_manifest",
        "outcomes_priority",
        "full_report",
        "replay",
        "dependencies",
        "qualification",
    }
)
# Scalar metric, run and replay schema pins of snapshots saved before Red Zone.
_PRE_RED_ZONE_PINS = (14, 2, 3)
_CONFIG_FIELDS = frozenset(
    {
        "format",
        "version",
        "snapshot_id",
        "release",
        "participants",
        "conditions",
        "analysis",
        "compatibility",
        "assets",
        "record_sources",
    }
)
_CONTENT_FIELDS = frozenset(
    {
        "code",
        "parameters",
        "memory_template",
        "input_preparation",
        "decision_settings",
        "adapter_bindings",
        "external_state",
    }
)


def canonical_json(value: object) -> bytes:
    """Encode finite JSON with sorted object keys and preserved array order.

    Return UTF-8 bytes without extra whitespace. Reject nonstring mapping keys,
    nonfinite numbers, cycles and non-JSON values; do not silently turn an integer
    key into a string. No file or device is accessed. All tournament content
    identities use this encoding; location exclusion belongs to their caller.
    """
    pending = [value]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, (Mapping, list, tuple)):
            identity = id(cast(object, current))
            if identity in visited:
                continue
            visited.add(identity)
            if isinstance(current, Mapping):
                mapping = cast(Mapping[object, object], current)
                if any(not isinstance(key, str) for key in mapping):
                    raise ValueError("Tournament JSON object keys must be strings")
                pending.extend(mapping.values())
            else:
                pending.extend(cast(Sequence[object], current))
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated JSON fields before a parser could silently discard one."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Tournament JSON repeats field {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    """Reject JSON's nonstandard NaN and infinity spellings."""
    raise ValueError(f"Tournament JSON cannot contain {value}")


def _finite_float(value: str) -> float:
    """Reject exponent spellings that overflow Python's finite float domain."""
    result = float(value)
    if not isfinite(result):
        raise ValueError("Tournament JSON numbers must be finite")
    return result


def read_config_json(path: str | Path) -> dict[str, Any]:
    """Read one strict JSON object without validating a particular schema.

    ``path`` names an existing UTF-8 file. Duplicate keys, nonfinite values and
    nonobject roots raise ValueError. File and decoding errors propagate. The
    configuration, schedule and asset readers share this read-only parser.
    """
    value: object = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )
    if not isinstance(value, dict):
        raise ValueError("Tournament JSON must contain one object")
    return cast(dict[str, Any], value)


def snapshot_identity(config: Mapping[str, object]) -> str:
    """Hash declared scientific content without its ID or asset location hints.

    ``config`` is a JSON-compatible description. This helper copies it, removes
    top-level ``snapshot_id``/``source_location`` and each asset's ``path``/``url``,
    then returns lowercase SHA256. Order in arrays remains meaningful. This
    computes identity only; it does not validate release approval or asset bytes.
    """
    value: dict[str, Any] = json.loads(canonical_json(dict(config)))
    value.pop("snapshot_id", None)
    value.pop("source_location", None)
    for asset in value.get("assets", {}).values():
        if isinstance(asset, dict):
            asset.pop("path", None)
            asset.pop("url", None)
    return sha256(canonical_json(value)).hexdigest()


def _object(
    value: object,
    name: str,
    fields: set[str] | frozenset[str],
    *,
    optional: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Read one exact JSON object, naming missing and unexpected fields."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    row = cast(dict[str, Any], value)
    missing, extra = fields - row.keys(), row.keys() - fields - optional
    if missing or extra:
        raise ValueError(
            f"{name} fields differ: missing {sorted(missing)}, extra {sorted(extra)}"
        )
    return row


def _text(value: object, name: str) -> str:
    """Return a nonblank string without changing its spelling or whitespace."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(
    value: object, name: str, minimum: int = 0, maximum: int | None = None
) -> int:
    """Accept exact Python integers in range; booleans are never counts or IDs."""
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be an integer in the supported range")
    return value


def _digest(value: object, name: str) -> str:
    """Require one full lowercase SHA256 value."""
    result = _text(value, name)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return result


def _array(value: object, name: str) -> list[Any]:
    """Require a JSON array; its order is retained throughout resolution."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return cast(list[Any], value)


def _asset_ref(value: object, assets: Mapping[str, Any], name: str) -> str:
    """Check an asset ID against declarations without opening its file."""
    identifier = _text(value, name)
    if identifier not in assets:
        raise ValueError(f"{name} references undeclared asset {identifier!r}")
    return identifier


def validate_inline_asset(descriptor: Mapping[str, Any], *, version: int) -> None:
    """Check optional v2 metadata against its canonical JSON size and digest.

    Inline metadata is an object or array in ``inline``. It has no file or URL,
    and only the roles in INLINE_METADATA_ROLES may use it. Version 1 forbids
    this field. This host-only check never writes files or imports controllers.
    Invalid roles, content, byte counts and digests raise ValueError.
    """
    if "inline" not in descriptor:
        return
    if version != 2 or descriptor.get("role") not in INLINE_METADATA_ROLES:
        raise ValueError("Only version-2 metadata assets may contain inline JSON")
    if descriptor.get("path") is not None or descriptor.get("url") is not None:
        raise ValueError("Inline metadata cannot also have a file path or URL")
    content = descriptor["inline"]
    if not isinstance(content, (dict, list)):
        raise ValueError("Inline metadata must contain a JSON object or array")
    encoded = canonical_json(cast(object, content))
    if (
        descriptor.get("sha256") != sha256(encoded).hexdigest()
        or type(descriptor.get("size_bytes")) is not int
        or descriptor["size_bytes"] != len(encoded)
    ):
        raise ValueError("Inline metadata differs from its declared size or digest")


def _assets(value: object, *, version: int = 1) -> dict[str, Any]:
    """Validate file declarations and verify embedded version-2 metadata."""
    if not isinstance(value, dict):
        raise ValueError("assets must be an object keyed by asset ID")
    result = cast(dict[str, Any], value)
    for identifier, raw in result.items():
        _text(identifier, "asset ID")
        row = _object(
            raw,
            f"asset {identifier}",
            {"sha256", "size_bytes", "role"},
            optional={"path", "url", "inline"} if version == 2 else {"path", "url"},
        )
        _digest(row["sha256"], "asset sha256")
        _integer(row["size_bytes"], "asset size_bytes")
        if _text(row["role"], "asset role") not in ASSET_ROLES:
            raise ValueError(f"Unsupported asset role {row['role']!r}")
        for key in ("path", "url"):
            if row.get(key) is not None:
                _text(row[key], f"asset {key}")
        validate_inline_asset(row, version=version)
    return result


def _controller(value: object, assets: Mapping[str, Any], *, version: int = 1) -> None:
    """Check controller loader and evidence declarations without importing code."""
    if not isinstance(value, dict):
        raise ValueError("controller must be an object")
    variants = {
        "builtin": {"name"},
        "factory": {"factory"},
        "bundle": {"loader", "asset_ids"},
    }
    if version == 2:
        variants["reference"] = {"reference"}
    raw = cast(dict[str, Any], value)
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in variants:
        raise ValueError("Unsupported controller kind for this descriptor version")
    row = _object(raw, "controller", {"kind", "content", *variants[kind]})
    if kind == "builtin":
        _text(row["name"], "builtin name")
    elif kind == "reference":
        reference = _text(row["reference"], "controller reference")
        if not Path(reference).is_absolute():
            if "/" in reference or "\\" in reference:
                raise ValueError("Controller folder references must be absolute")
            if ":" in reference:
                pieces = reference.split(":")
                if len(pieces) != 2 or not all(pieces) or "." in pieces[1]:
                    raise ValueError("Controller factory must use module:function")
    else:
        reference = _text(
            row["factory" if kind == "factory" else "loader"], "controller loader"
        )
        if reference.count(":") != 1 or not all(
            part.strip() for part in reference.split(":")
        ):
            raise ValueError("Controller loader must use module:callable")
    if kind == "bundle":
        names = _array(row["asset_ids"], "controller asset_ids")
        resolved = [_asset_ref(item, assets, "controller asset") for item in names]
        if len(set(resolved)) != len(resolved):
            raise ValueError("Controller asset_ids repeat an asset")
    content = _object(row["content"], "controller content", _CONTENT_FIELDS)
    for key in _CONTENT_FIELDS - {"adapter_bindings"}:
        if content[key] is not None:
            evidence = _object(content[key], key, {"asset_id", "sha256"})
            asset = _asset_ref(evidence["asset_id"], assets, key)
            if _digest(evidence["sha256"], key) != assets[asset]["sha256"]:
                raise ValueError(f"{key} evidence differs from its asset digest")
    if content["adapter_bindings"] is not None:
        slots: set[int] = set()
        for raw in _array(content["adapter_bindings"], "adapter_bindings"):
            binding = _object(
                raw, "adapter binding", {"team_slot", "component_id", "controller_id"}
            )
            slot = _integer(binding["team_slot"], "team_slot", 0, 4)
            if slot in slots:
                raise ValueError("Adapter bindings repeat a team slot")
            slots.add(slot)
            _integer(binding["component_id"], "component_id")
            if binding["controller_id"] is not None:
                _digest(binding["controller_id"], "component controller_id")


def _participants(
    value: object, assets: Mapping[str, Any], *, official: bool, version: int = 1
) -> list[dict[str, Any]]:
    """Check ordered participants under the declared descriptor version.

    Every field needs at least two entrants. Official version-1 releases retain
    exactly twelve. Official releases of either version require distinct
    controller identities, saved ratings and rating order. Custom populations
    may give different labels to the same controller for declared comparisons.
    """
    rows = _array(value, "participants")
    if len(rows) < 2:
        raise ValueError("Tournament fields need at least two participants")
    if official and version == 1 and len(rows) != 12:
        raise ValueError(
            "An official version-1 snapshot needs exactly twelve participants"
        )
    result: list[dict[str, Any]] = []
    for raw in rows:
        row = _object(
            raw,
            "participant",
            {
                "entrant_id",
                "name",
                "controller_id",
                "controller",
                "registration_asset",
                "elo",
                "result_ref",
            },
        )
        for key in ("entrant_id", "name"):
            _text(row[key], f"participant {key}")
        _digest(row["controller_id"], "controller_id")
        _asset_ref(row["registration_asset"], assets, "registration_asset")
        _controller(row["controller"], assets, version=version)
        rating = row["elo"]
        if rating is not None and (
            type(rating) not in (int, float) or not isfinite(rating)
        ):
            raise ValueError("Participant Elo must be finite or null")
        if official and (rating is None or row["result_ref"] is None):
            raise ValueError(
                "Released participants require stored Elo and its result reference"
            )
        if row["result_ref"] is not None:
            reference = _object(
                row["result_ref"], "result_ref", {"source_id", "table", "policy"}
            )
            _digest(reference["source_id"], "result source_id")
            if reference["table"] != "tournament_results":
                raise ValueError(
                    "Participant result_ref must select tournament_results"
                )
            _text(reference["policy"], "result policy")
        result.append(row)
    for key in ("entrant_id", "name"):
        if len({row[key] for row in result}) != len(result):
            raise ValueError(
                f"Participant {key} collision; supply distinct declared labels and IDs"
            )
    if official and len({row["controller_id"] for row in result}) != len(result):
        raise ValueError("Official participants repeat a controller version")
    if official and any(
        result[i]["elo"] < result[i + 1]["elo"] for i in range(len(result) - 1)
    ):
        raise ValueError("Official participants must follow their stored Elo order")
    return result


def _conditions(
    value: object, assets: Mapping[str, Any], *, official: bool, version: int = 1
) -> None:
    """Check game conditions without constructing simulator configs.

    Version-1 releases retain five maps and five agents per team. Version-2
    releases use their nonempty map list and one through five agents per team.
    Both retain registered test maps, mirrored rosters and equal coverage of
    maps and spawn choices. Custom fields keep their supported choices.
    """
    row = _object(
        value,
        "conditions",
        {
            "task",
            "map_sources",
            "rosters",
            "information_mode",
            "games_per_opponent",
            "score_threshold",
            "max_steps",
            "memory_rule",
            "pairing_protocol",
            "schedule_asset",
        },
    )
    if row["task"] != "tdm" or row["pairing_protocol"] != "fixed-team-spawn-v1":
        raise ValueError("Tournament configs require TDM and fixed-team-spawn-v1")
    for key in ("information_mode", "memory_rule"):
        _text(row[key], key)
    for key in ("games_per_opponent", "score_threshold", "max_steps"):
        _integer(row[key], key, 1)
    _asset_ref(row["schedule_asset"], assets, "schedule_asset")
    maps = _array(row["map_sources"], "map_sources")
    if not maps:
        raise ValueError("Tournament maps must be nonempty")
    if official and version == 1 and len(maps) != 5:
        raise ValueError("Official version-1 snapshots require five test maps")
    map_ids: set[int] = set()
    for raw in maps:
        item = _object(
            raw,
            "map source",
            {
                "map_id",
                "split",
                "source_config_asset",
                "source_config_id",
                "registered_map",
            },
        )
        identifier = _integer(item["map_id"], "map_id", 0, 2**31 - 1)
        if identifier in map_ids:
            raise ValueError("Tournament map sources repeat a map ID")
        map_ids.add(identifier)
        if item["split"] not in (None, "training", "validation", "test"):
            raise ValueError("Map split must be training, validation, test or null")
        _asset_ref(item["source_config_asset"], assets, "source_config_asset")
        _digest(item["source_config_id"], "source_config_id")
        if item["registered_map"] is not None:
            from marl_battlegrounds._tdm_assets import TDMMapInfo

            registered = TDMMapInfo.model_validate(item["registered_map"])
            if registered.map_id != identifier or registered.split != item["split"]:
                raise ValueError(
                    "Registered map identity differs from its declared ID or split"
                )
        if official and (item["split"] != "test" or item["registered_map"] is None):
            raise ValueError(
                "Official maps require saved registered test-map identities"
            )
    if row["games_per_opponent"] % (2 * len(maps)):
        raise ValueError(
            "games_per_opponent must divide equally across maps and both spawn choices"
        )
    rosters = _object(row["rosters"], "rosters", {"team_a", "team_b"})
    for value in rosters.values():
        entries = _array(value, "team roster")
        if not 1 <= len(entries) <= 5:
            raise ValueError("Each roster needs one through five agent classes")
        for entry in entries:
            _text(entry, "roster class")
    if official and rosters["team_a"] != rosters["team_b"]:
        raise ValueError("Official rosters must be mirrored")
    if official and version == 1 and len(rosters["team_a"]) != 5:
        raise ValueError("Official version-1 rosters must have five agents")


def _records(value: object, assets: Mapping[str, Any]) -> set[str]:
    """Check source-reference structure; the reuse authority checks raw evidence."""
    identifiers: set[str] = set()
    for raw in _array(value, "record_sources"):
        row = _object(
            raw,
            "record source",
            {"source_id", "run_id", "manifest_asset", "tables", "replays"},
        )
        identifier = _digest(row["source_id"], "source_id")
        if identifier in identifiers:
            raise ValueError("Record sources repeat an identity")
        identifiers.add(identifier)
        _text(row["run_id"], "source run_id")
        _asset_ref(row["manifest_asset"], assets, "manifest_asset")
        if not isinstance(row["tables"], dict):
            raise ValueError("Source tables must be a mapping")
        for name, raw_table in cast(dict[str, Any], row["tables"]).items():
            _text(name, "table role")
            table = _object(
                raw_table,
                "source table",
                {"asset_id", "committed_bytes", "rows", "header_sha256"},
            )
            asset = _asset_ref(table["asset_id"], assets, "source table asset")
            length = _integer(table["committed_bytes"], "committed_bytes")
            if length != assets[asset]["size_bytes"]:
                raise ValueError(
                    "Source table asset must contain exactly its committed prefix"
                )
            _integer(table["rows"], "source table rows")
            _digest(table["header_sha256"], "header_sha256")
        replay_keys: set[tuple[str, str, str, int]] = set()
        for raw_replay in _array(row["replays"], "source replays"):
            replay = _object(
                raw_replay,
                "source replay",
                {"run_id", "phase", "pass_id", "episode_id", "asset_id"},
            )
            key = (
                _text(replay["run_id"], "run_id"),
                _text(replay["phase"], "phase"),
                _text(replay["pass_id"], "pass_id"),
                _integer(replay["episode_id"], "episode_id", 1, 2**31 - 1),
            )
            if key in replay_keys or key[0] != row["run_id"]:
                raise ValueError("Source replays repeat an origin or use another run")
            replay_keys.add(key)
            _asset_ref(replay["asset_id"], assets, "replay asset")
    return identifiers


def _validate(config: dict[str, Any], *, official: bool) -> None:
    """Check complete descriptor structure without executing its referenced work.

    The compatibility pins (scalar_schema, run_schema, replay_schema) must be
    exactly (14, 2, 3), a snapshot saved before the Red Zone rule that can be
    reused but not extended, or (METRIC_SCHEMA_VERSION, 2, 4), a current one. A
    value no set allows raises ValueError "Unsupported tournament <key>"; valid
    values in a mixed combination raise "Unsupported tournament schema
    combination".
    """
    _object(
        config,
        "tournament config",
        _CONFIG_FIELDS,
        optional={"source_location", "selection"},
    )
    if (
        config["format"] != CONFIG_FORMAT
        or type(config["version"]) is not int
        or config["version"] not in (1, 2)
    ):
        raise ValueError("Unsupported tournament configuration format/version")
    _digest(config["snapshot_id"], "snapshot_id")
    assets = _assets(config["assets"], version=config["version"])
    participants = _participants(
        config["participants"], assets, official=official, version=config["version"]
    )
    _conditions(
        config["conditions"], assets, official=official, version=config["version"]
    )
    sources = _records(config["record_sources"], assets)
    for participant in participants:
        if (
            participant["result_ref"] is not None
            and participant["result_ref"]["source_id"] not in sources
        ):
            raise ValueError("Participant result_ref names an undeclared source")
    analysis = _object(
        config["analysis"],
        "analysis",
        {"method_id", "bootstrap_replicates", "bootstrap_seed", "opponent_weights"},
    )
    if (
        analysis["method_id"] != "centered-davidson-v1"
        or type(analysis["bootstrap_replicates"]) is not int
        or analysis["bootstrap_replicates"] != 5000
    ):
        raise ValueError(
            "Tournament analysis must retain the existing estimator and 5000 replicates"
        )
    _integer(analysis["bootstrap_seed"], "bootstrap_seed", 0, 2**32 - 1)
    from marl_battlegrounds.evaluation.tournament_statistics import (
        validate_opponent_weights,
    )

    validate_opponent_weights(
        [row["entrant_id"] for row in participants], analysis["opponent_weights"]
    )
    if official and analysis["opponent_weights"] is not None:
        raise ValueError("Released snapshots use equal opponent weights")
    compatibility = _object(
        config["compatibility"],
        "compatibility",
        {
            "package_version",
            "code_revision",
            "environment_id",
            "source_manifest_asset",
            "scalar_schema",
            "run_schema",
            "replay_schema",
            "action_stream_version",
            "initialization_stream_version",
            "dependency_lock_asset",
        },
    )
    from marl_battlegrounds.evaluation.models import CodeRevisionV2

    revision = CodeRevisionV2.model_validate(compatibility["code_revision"])
    if revision.package_version != _text(
        compatibility["package_version"], "package_version"
    ):
        raise ValueError("Package version differs from code provenance")
    _digest(compatibility["environment_id"], "environment_id")
    for key in ("scalar_schema", "run_schema", "replay_schema"):
        if type(compatibility[key]) is not int:
            raise ValueError(f"Unsupported tournament {key}")
    pins = (
        compatibility["scalar_schema"],
        compatibility["run_schema"],
        compatibility["replay_schema"],
    )
    # Exactly two pin sets: snapshots saved before the Red Zone rule (scalar
    # schema 14, replay 3; readable and reusable, not extendable) and current
    # ones (today's scalar schema, replay 4).
    from marl_battlegrounds.evaluation.metric_catalog import METRIC_SCHEMA_VERSION

    allowed = (_PRE_RED_ZONE_PINS, (METRIC_SCHEMA_VERSION, 2, 4))
    if pins not in allowed:
        for index, key in enumerate(("scalar_schema", "run_schema", "replay_schema")):
            if all(pins[index] != row[index] for row in allowed):
                raise ValueError(f"Unsupported tournament {key}")
        raise ValueError("Unsupported tournament schema combination")
    for key in ("action_stream_version", "initialization_stream_version"):
        _text(compatibility[key], key)
    for key in ("source_manifest_asset", "dependency_lock_asset"):
        _asset_ref(compatibility[key], assets, key)
    release = config["release"]
    if official and release is None:
        raise ValueError("Official snapshots require an approved release record")
    if release is not None:
        row = _object(
            release,
            "release",
            {
                "release_at_utc",
                "release_at_local",
                "cutoff_at_utc",
                "cutoff_at_local",
                "rules_id",
                "approval_id",
            },
        )
        for key in ("rules_id", "approval_id"):
            _text(row[key], key)
        for key in (
            "release_at_utc",
            "release_at_local",
            "cutoff_at_utc",
            "cutoff_at_local",
        ):
            moment = datetime.fromisoformat(_text(row[key], key))
            if moment.tzinfo is None or moment.utcoffset() is None:
                raise ValueError(f"{key} must include its UTC offset")
    if "selection" in config:
        from marl_battlegrounds.evaluation.population_selection import (
            validate_bound_selection,
        )

        validate_bound_selection(config)
    if config["snapshot_id"] != snapshot_identity(config):
        raise ValueError("Tournament snapshot content differs from snapshot_id")


def _installed_catalog() -> dict[str, Any]:
    """Read the installed release pins, or an empty catalog before first release.

    Tests replace this private provider with an isolated fixture catalog. No
    public option bypasses the release check. Catalog entries store ``config``
    paths/mappings and the pinned ``release`` record under their snapshot digest.
    """
    resource = files("marl_battlegrounds").joinpath("data/tournaments/catalog.json")
    if not resource.is_file():
        return {"default": None, "snapshots": {}}
    value: object = json.loads(
        resource.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )
    catalog = _object(value, "official catalog", {"default", "snapshots"})
    if not isinstance(catalog["snapshots"], dict):
        raise ValueError("Official catalog snapshots must be a mapping")
    return catalog


def _locations(config: dict[str, Any], parent: Path | None) -> None:
    """Resolve declared local paths once and reject relative bundle escapes."""
    supplied = config.get("source_location")
    if supplied is not None:
        source = Path(_text(supplied, "source_location"))
        if not source.is_absolute():
            raise ValueError("source_location must be an absolute directory")
        if parent is None:
            parent = source.resolve()
    if parent is not None:
        config["source_location"] = str(parent)
    for row in config["assets"].values():
        if row.get("path") is None:
            continue
        path = Path(row["path"])
        if not path.is_absolute():
            if parent is None:
                raise ValueError(
                    "Relative asset paths require source_location or a JSON file path"
                )
            resolved = (parent / path).resolve()
            if not resolved.is_relative_to(parent):
                raise ValueError(
                    "Relative asset path escapes the configuration directory"
                )
            path = resolved
        row["path"] = str(path)


def load_tournament_config(
    config: str | Path | Mapping[str, object] | None = None,
    *,
    official: bool | None = None,
) -> dict[str, Any]:
    """Read, copy and validate one tournament description without side effects.

    Parameters
    ----------
    config : path, mapping or None
        JSON path or parsed version-1 or version-2 descriptor. Version 1 keeps
        twelve-entry releases; version 2 permits any frozen release population
        of at least two. Both allow custom fields of at least two. Version 2
        also supports inline metadata and shared method references. None picks
        the installed official default and fails before a bundle is released.
        File-relative asset paths use the file's parent. Parsed mappings need
        absolute asset paths or an absolute ``source_location``.
    official : bool or None
        None checks release pins for descriptors with a release record. True
        always requires an unchanged pinned official snapshot, using its
        descriptor version's population and game-condition checks.
        False performs structural checks for custom or private provisional
        descriptors; public canonical calls must never use this bypass.

    Returns
    -------
    dict
        A fresh JSON-only copy with verified descriptor identity and absolute
        local asset hints. Payload existence, hashes, executable behavior and
        physical game evidence are checked by their owning later authorities.

    Raises
    ------
    ValueError
        JSON, fields, references, paths, identity or required release pins fail.
        The compatibility pins must be exactly (14, 2, 3), a snapshot saved
        before the Red Zone rule (reusable, not extendable), or (current scalar
        schema, 2, 4); any other value or combination raises "Unsupported
        tournament <key>" or "Unsupported tournament schema combination".
    TypeError
        config is not a supported path/mapping or contains non-JSON values.
    OSError
        A requested configuration file cannot be read.

    Notes
    -----
    Host-only. Does not import JAX, open payload assets, execute a controller,
    construct a writer, download anything or change the supplied mapping.
    """
    catalog: dict[str, Any] | None = None
    selected: str | None = None
    if config is None:
        catalog = _installed_catalog()
        selected = catalog.get("default")
        if selected is None or selected not in catalog.get("snapshots", {}):
            raise ValueError(
                "No released official tournament snapshot is installed; prepare an "
                "approved bundle or use run_tournament with a custom config"
            )
        config = catalog["snapshots"][selected]["config"]
        official = True
    parent: Path | None = None
    if isinstance(config, (str, Path)):
        path = Path(config).absolute()
        descriptor = read_config_json(path)
        parent = path.parent.resolve()
    elif isinstance(config, Mapping):
        descriptor = cast(dict[str, Any], json.loads(canonical_json(dict(config))))
    else:
        raise TypeError("Tournament config must be a JSON path or parsed mapping")
    require_official = official is True or (
        official is None and descriptor.get("release") is not None
    )
    _validate(descriptor, official=require_official)
    if selected is not None and descriptor["snapshot_id"] != selected:
        raise ValueError("Installed default content differs from its pinned identity")
    _locations(descriptor, parent)
    if require_official:
        catalog = _installed_catalog() if catalog is None else catalog
        entry = catalog.get("snapshots", {}).get(descriptor["snapshot_id"])
        if not isinstance(entry, dict) or entry.get("release") != descriptor["release"]:
            raise ValueError(
                "Configuration is not an unchanged pinned official snapshot; "
                "edited populations or rules require run_tournament(config=...)"
            )
    return descriptor


def resolve_tournament_config(
    config: str | Path | Mapping[str, object] | None,
    *,
    official: bool | None = None,
    saved: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Resolve a saved description before consulting a new-run default.

    ``saved`` is the exact resolved descriptor already read from the selected
    run. None for ``config`` inherits it; a supplied description must identify
    the same scientific content. The caller owns omitted-versus-explicit None
    assertions on other public arguments. Saved release approval comes from
    its recorded pins, not today's installed alias; full compatibility remains
    the runner's responsibility. No file is created, recovered or changed.
    """
    if saved is None:
        return load_tournament_config(config, official=official)
    previous = load_tournament_config(saved, official=False)
    if config is None:
        return previous
    # The saved descriptor already owns the release identity. Even an explicit
    # equal file is an assertion against it, not a request for today's catalog.
    current = load_tournament_config(config, official=False)
    if current["snapshot_id"] != previous["snapshot_id"]:
        raise ValueError("Tournament config differs from the saved resolved snapshot")
    return current


def tournament_qualification(
    config: Mapping[str, Any],
    budget: Mapping[str, Any],
    *,
    official_verified: bool,
) -> dict[str, Any]:
    """Describe release verification and budget compliance without catalog access.

    config is the checked saved or freshly resolved descriptor. budget holds its
    original and requested games-per-opponent counts. official_verified records
    the resolver's approval check, not the presence of self-declared release text.
    Return the two distinct flags and a readable reason. Invalid count/flag
    combinations raise ValueError; no files, models or current catalog are read.
    """
    original = config["conditions"]["games_per_opponent"]
    resolved = budget["resolved_games_per_opponent"]
    if (
        type(official_verified) is not bool
        or type(original) is not int
        or type(resolved) is not int
        or resolved <= 0
        or budget["official_games_per_opponent"] != original
        or type(budget["budget_override"]) is not bool
        or budget["budget_override"] != (resolved != original)
        or (official_verified and config.get("release") is None)
    ):
        raise ValueError("Tournament qualification has inconsistent release or budget")
    compliant = official_verified and resolved == original
    reason = (
        "Released snapshot verified; original game budget retained"
        if compliant
        else "Released snapshot verified; game budget changed"
        if official_verified
        else "Snapshot has no recorded official verification"
    )
    return {
        "official_snapshot_verified": official_verified,
        "protocol_compliant": compliant,
        "qualification_reason": reason,
    }
