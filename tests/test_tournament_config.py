"""Check tournament descriptors before any model, writer or game is touched.

Tests cover strict JSON, immutable content identity, declared asset locations,
official release pins, field/reference validation and saved-first resume.
Version-1 hashes and twelve-entry releases stay unchanged. Version-2 releases
use their frozen population, registered test maps and supported mirrored rosters.
The fixture catalog is isolated and never establishes real official qualification.
Exactly two schema pin sets load: (14, 2, 3) for snapshots saved before the Red
Zone rule, such as the shared fixture descriptor, and today's scalar schema
with run schema 2 and replay schema 4. Any other scalar, run or replay pin is
refused and names the field. A mix of the two sets, (14, 2, 4) or today's
scalar schema with replay schema 3, is refused as an unsupported schema
combination.
"""

import copy
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from tests.canonical_fixtures import config_descriptor

from marl_battlegrounds.evaluation import tournament_config as configs
from marl_battlegrounds.evaluation.metric_catalog import METRIC_SCHEMA_VERSION


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    value["snapshot_id"] = configs.snapshot_identity(value)
    return value


def _pin(monkeypatch: pytest.MonkeyPatch, value: dict[str, Any]) -> None:
    monkeypatch.setattr(
        configs,
        "_installed_catalog",
        lambda: {
            "default": value["snapshot_id"],
            "snapshots": {
                value["snapshot_id"]: {"config": value, "release": value["release"]}
            },
        },
    )


def test_custom_description_is_copied_and_keeps_supplied_order() -> None:
    original = config_descriptor(entrants=3)
    before = copy.deepcopy(original)
    result = configs.load_tournament_config(original)
    assert result == before
    result["participants"][0]["name"] = "Changed in returned copy"
    assert original == before


def test_location_hints_do_not_change_science_but_array_order_does() -> None:
    original = config_descriptor()
    moved = copy.deepcopy(original)
    moved["source_location"] = "/new-location"
    for asset in moved["assets"].values():
        asset["path"] = "/new-location/file"
        asset["url"] = "https://example.invalid/file"
    assert configs.snapshot_identity(moved) == original["snapshot_id"]
    moved["participants"].reverse()
    assert configs.snapshot_identity(moved) != original["snapshot_id"]


@pytest.mark.parametrize(
    "content",
    [
        '{"a":1,"a":2}',
        '{"x":{"a":1,"a":2}}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":1e999}',
        "[]",
    ],
)
def test_strict_json_rejects_ambiguous_or_nonfinite_values(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "config.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        configs.read_config_json(path)


@pytest.mark.parametrize(
    "value", [{1: "wrong key"}, {"v": float("nan")}, {"v": float("inf")}]
)
def test_parsed_json_rejects_implicit_key_conversion_and_nonfinite_values(
    value: object,
) -> None:
    with pytest.raises(ValueError):
        configs.canonical_json(value)


def test_relative_file_assets_use_file_parent_without_reading_payloads(
    tmp_path: Path,
) -> None:
    value = config_descriptor()
    for name, asset in value["assets"].items():
        asset["path"] = name
    path = tmp_path / "config.json"
    path.write_bytes(configs.canonical_json(value))
    result = configs.load_tournament_config(path)
    assert result["snapshot_id"] == value["snapshot_id"]
    assert result["source_location"] == str(tmp_path)
    assert all(Path(a["path"]).parent == tmp_path for a in result["assets"].values())
    assert sorted(item.name for item in tmp_path.iterdir()) == ["config.json"]


def test_mapping_relative_assets_need_a_declared_source_location(
    tmp_path: Path,
) -> None:
    value = config_descriptor()
    value["assets"]["config"]["path"] = "config"
    with pytest.raises(ValueError, match="Relative asset paths"):
        configs.load_tournament_config(value)
    value["source_location"] = str(tmp_path)
    result = configs.load_tournament_config(value)
    assert result["assets"]["config"]["path"] == str(tmp_path / "config")


@pytest.mark.parametrize("escape", ["../outside", "linked/file"])
def test_relative_bundle_escape_is_rejected(tmp_path: Path, escape: str) -> None:
    (tmp_path / "linked").symlink_to(tmp_path.parent, target_is_directory=True)
    value = config_descriptor(root=tmp_path)
    value["source_location"] = str(tmp_path)
    value["assets"]["config"]["path"] = escape
    with pytest.raises(ValueError, match="escapes"):
        configs.load_tournament_config(value)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("version",), True),
        (("version",), 3),
        (("format",), "other"),
        (("conditions", "games_per_opponent"), True),
        (("conditions", "games_per_opponent"), 0),
        (("conditions", "games_per_opponent"), 3),
        (("conditions", "max_steps"), 1.5),
        (("conditions", "schedule_asset"), "missing"),
        (("compatibility", "scalar_schema"), 13),
        (("compatibility", "run_schema"), True),
        (("analysis", "bootstrap_replicates"), 50),
        (("analysis", "opponent_weights"), {"fixture-0": 1.0}),
        (("assets", "config", "size_bytes"), False),
        (("assets", "config", "sha256"), "bad"),
    ],
)
def test_invalid_nested_conditions_fail_before_any_file_work(
    path: tuple[str, ...], value: object
) -> None:
    descriptor = config_descriptor()
    target = descriptor
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    with pytest.raises((ValueError, TypeError)):
        configs.load_tournament_config(_seal(descriptor))


@pytest.mark.parametrize(
    ("pins", "refused"),
    [
        ((14, 2, 3), None),
        ((METRIC_SCHEMA_VERSION, 2, 4), None),
        ((13, 2, 3), "scalar_schema"),
        ((METRIC_SCHEMA_VERSION + 1, 2, 4), "scalar_schema"),
        ((14, 1, 3), "run_schema"),
        ((METRIC_SCHEMA_VERSION, 3, 4), "run_schema"),
        ((14, 2, 2), "replay_schema"),
        ((METRIC_SCHEMA_VERSION, 2, 5), "replay_schema"),
        ((14, 2, 4), "schema combination"),
        ((METRIC_SCHEMA_VERSION, 2, 3), "schema combination"),
    ],
)
def test_only_the_pre_red_zone_and_current_schema_pins_load(
    pins: tuple[int, int, int], refused: str | None
) -> None:
    descriptor = config_descriptor()
    compatibility = descriptor["compatibility"]
    # The shared fixture descriptor is a snapshot saved before the Red Zone rule.
    assert (
        compatibility["scalar_schema"],
        compatibility["run_schema"],
        compatibility["replay_schema"],
    ) == (14, 2, 3)
    compatibility.update(
        scalar_schema=pins[0], run_schema=pins[1], replay_schema=pins[2]
    )
    descriptor = _seal(descriptor)
    if refused is None:
        loaded = configs.load_tournament_config(descriptor)
        assert loaded["compatibility"]["replay_schema"] == pins[2]
        return
    with pytest.raises(ValueError, match=f"^Unsupported tournament {refused}$"):
        configs.load_tournament_config(descriptor)


def test_unknown_fields_missing_fields_and_tampering_fail() -> None:
    value = config_descriptor()
    value["conditions"]["new_rule"] = "unsupported"
    with pytest.raises(ValueError, match="extra"):
        configs.load_tournament_config(_seal(value))
    value = config_descriptor()
    del value["conditions"]["max_steps"]
    with pytest.raises(ValueError, match="missing"):
        configs.load_tournament_config(_seal(value))
    value = config_descriptor()
    value["conditions"]["max_steps"] = 16
    with pytest.raises(ValueError, match="snapshot_id"):
        configs.load_tournament_config(value)


def test_official_defaults_need_release_pins_not_twelve_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = config_descriptor(entrants=12, official=True)
    with pytest.raises(ValueError, match="pinned"):
        configs.load_tournament_config(value)
    _pin(monkeypatch, value)
    result = configs.load_tournament_config()
    assert len(result["participants"]) == 12
    assert result["snapshot_id"] == value["snapshot_id"]


def test_no_initial_bundle_fails_without_substituting_builtins() -> None:
    with pytest.raises(ValueError, match="No released official tournament"):
        configs.load_tournament_config()


@pytest.mark.parametrize("count", [2, 11, 13])
def test_official_population_must_be_twelve(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    value = config_descriptor(entrants=count, official=True)
    _pin(monkeypatch, value)
    with pytest.raises(ValueError, match="exactly twelve"):
        configs.load_tournament_config()


def test_names_and_exact_official_controller_versions_are_separate_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = config_descriptor(entrants=12, official=True)
    value["participants"][1]["name"] = value["participants"][0]["name"]
    _pin(monkeypatch, _seal(value))
    with pytest.raises(ValueError, match="name collision"):
        configs.load_tournament_config()
    value = config_descriptor(entrants=12, official=True)
    value["participants"][1]["controller_id"] = value["participants"][0][
        "controller_id"
    ]
    _pin(monkeypatch, _seal(value))
    with pytest.raises(ValueError, match="controller version"):
        configs.load_tournament_config()


def test_saved_first_resolution_does_not_consult_current_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = config_descriptor(entrants=12, official=True)

    def unavailable() -> dict[str, Any]:
        raise AssertionError("Saved resolution consulted today's catalog")

    monkeypatch.setattr(configs, "_installed_catalog", unavailable)
    assert configs.resolve_tournament_config(None, official=True, saved=saved) == saved
    assert configs.resolve_tournament_config(saved, official=True, saved=saved) == saved
    conflict = config_descriptor()
    with pytest.raises(ValueError, match="saved resolved snapshot"):
        configs.resolve_tournament_config(conflict, official=False, saved=saved)


def test_explicit_equal_saved_custom_config_can_move_assets() -> None:
    saved = config_descriptor()
    moved = copy.deepcopy(saved)
    moved["assets"]["config"]["path"] = "/another-location/config"
    result = configs.resolve_tournament_config(moved, saved=saved)
    assert result["snapshot_id"] == saved["snapshot_id"]
    assert result["assets"]["config"]["path"] == "/another-location/config"


@pytest.mark.parametrize("version", [1, 2])
def test_loading_config_does_not_import_simulation_or_create_files(
    tmp_path: Path,
    version: int,
) -> None:
    path = tmp_path / "config.json"
    descriptor = config_descriptor(root=tmp_path)
    descriptor["version"] = version
    if version == 2:
        content = {"name": "fixture-0"}
        encoded = configs.canonical_json(content)
        descriptor["assets"]["registration"] = {
            "role": "registration",
            "sha256": sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
            "inline": content,
        }
        row = descriptor["participants"][0]
        row["controller"] = {
            "kind": "reference",
            "reference": "not_imported:create",
            "content": row["controller"]["content"],
        }
    path.write_bytes(configs.canonical_json(_seal(descriptor)))
    code = (
        "import sys\n"
        "from marl_battlegrounds.evaluation import tournament_config as c\n"
        "c.load_tournament_config(sys.argv[1])\n"
        "assert 'jax' not in sys.modules\n"
        "assert 'marl_battlegrounds.environment' not in sys.modules\n"
        "assert 'marl_battlegrounds.evaluation.run_writer' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code, str(path)], check=True, cwd=tmp_path)
    assert [item.name for item in tmp_path.iterdir()] == ["config.json"]


@pytest.mark.parametrize("verified", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_release_verification_and_budget_compliance_are_separate(
    verified: bool, override: bool
) -> None:
    config = config_descriptor(entrants=12, official=True)
    original = config["conditions"]["games_per_opponent"]
    budget = {
        "official_games_per_opponent": original,
        "resolved_games_per_opponent": original * (2 if override else 1),
        "budget_override": override,
    }
    result = configs.tournament_qualification(
        config, budget, official_verified=verified
    )
    assert result["official_snapshot_verified"] is verified
    assert result["protocol_compliant"] is (verified and not override)
    assert result["qualification_reason"]


@pytest.mark.parametrize(
    "reference", ["random", "examples.method:create", "/saved/actor"]
)
def test_v2_references_and_inline_metadata_need_no_runtime(reference: str) -> None:
    value = config_descriptor()
    value["version"] = 2
    content = {"name": "fixture-0"}
    encoded = configs.canonical_json(content)
    value["assets"]["registration"] = {
        "role": "registration",
        "sha256": sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        "inline": content,
    }
    for row in value["participants"]:
        row["controller"] = {
            "kind": "reference",
            "reference": reference,
            "content": row["controller"]["content"],
        }
    result = configs.load_tournament_config(_seal(value))
    assert result["assets"]["registration"]["inline"] == content
    assert result["participants"][0]["controller"]["reference"] == reference


@pytest.mark.parametrize(
    "mutation", ["v1", "payload", "path", "url", "hash", "size", "scalar"]
)
def test_inline_metadata_rejects_wrong_version_location_and_bytes(
    mutation: str,
) -> None:
    value = config_descriptor()
    value["version"] = 2
    content = {"name": "fixture-0"}
    encoded = configs.canonical_json(content)
    asset = {
        "role": "registration",
        "sha256": sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        "inline": content,
    }
    value["assets"]["registration"] = asset
    if mutation == "v1":
        value["version"] = 1
    elif mutation == "payload":
        asset["role"] = "model"
    elif mutation in {"path", "url"}:
        asset[mutation] = (
            "/somewhere" if mutation == "path" else "https://example.invalid"
        )
    elif mutation == "hash":
        asset["sha256"] = "0" * 64
    elif mutation == "size":
        asset["size_bytes"] = len(encoded) + 1
    else:
        asset["inline"] = "not metadata"
    with pytest.raises(ValueError):
        configs.load_tournament_config(_seal(value))


@pytest.mark.parametrize(
    "version,reference", [(1, "random"), (2, "./actor"), (2, "module:object.call")]
)
def test_reference_controller_rejects_old_schema_and_unresolved_paths(
    version: int, reference: str
) -> None:
    value = config_descriptor()
    value["version"] = version
    row = value["participants"][0]
    row["controller"] = {
        "kind": "reference",
        "reference": reference,
        "content": row["controller"]["content"],
    }
    with pytest.raises(ValueError):
        configs.load_tournament_config(_seal(value))


@pytest.mark.parametrize("count", [2, 7, 12, 13])
@pytest.mark.parametrize("version", [1, 2])
def test_custom_population_roundtrip_has_no_twelve_entry_limit(
    tmp_path: Path, count: int, version: int
) -> None:
    value = config_descriptor(entrants=count)
    value["version"] = version
    value = _seal(value)
    path = tmp_path / "field.json"
    path.write_bytes(configs.canonical_json(value))
    loaded = configs.load_tournament_config(path)
    assert loaded["participants"] == value["participants"]
    assert loaded["snapshot_id"] == value["snapshot_id"]
    assert configs.snapshot_identity(loaded) == value["snapshot_id"]
    assert configs.load_tournament_config(loaded) == loaded


@pytest.mark.parametrize("count", [2, 7, 12, 13])
def test_v2_official_population_comes_from_the_frozen_list(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    value = config_descriptor(entrants=count, official=True)
    value["version"] = 2
    value = _seal(value)
    with pytest.raises(ValueError, match="pinned official snapshot"):
        configs.load_tournament_config(value)
    _pin(monkeypatch, value)
    result = configs.load_tournament_config()
    assert result == value
    assert len(result["participants"]) == count
    changed = copy.deepcopy(value)
    changed["participants"].pop()
    changed = _seal(changed)
    with pytest.raises(ValueError, match=r"at least two|pinned official snapshot"):
        configs.load_tournament_config(changed)


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("official", [False, True])
@pytest.mark.parametrize("version", [1, 2])
def test_all_fields_require_at_least_two_participants(
    monkeypatch: pytest.MonkeyPatch, count: int, official: bool, version: int
) -> None:
    value = config_descriptor(entrants=count, official=official)
    value["version"] = version
    value = _seal(value)
    if official:
        _pin(monkeypatch, value)
    with pytest.raises(ValueError, match="at least two"):
        configs.load_tournament_config(value)


@pytest.mark.parametrize("field", ["entrant_id", "name", "controller_id"])
def test_v2_official_duplicate_check_includes_entries_after_twelve(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    value = config_descriptor(entrants=14, official=True)
    value["version"] = 2
    value["participants"][-1][field] = value["participants"][-2][field]
    _pin(monkeypatch, _seal(value))
    with pytest.raises(ValueError, match=r"collision|repeat a controller version"):
        configs.load_tournament_config()


@pytest.mark.parametrize("field", ["elo", "result_ref"])
def test_v2_official_participants_still_need_saved_ratings(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    value = config_descriptor(entrants=2, official=True)
    value["version"] = 2
    value["participants"][-1][field] = None
    _pin(monkeypatch, _seal(value))
    with pytest.raises(ValueError, match="stored Elo and its result reference"):
        configs.load_tournament_config()


def test_v2_official_rating_order_includes_entries_after_twelve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = config_descriptor(entrants=14, official=True)
    value["version"] = 2
    value["participants"][-1]["elo"] = 1201.0
    _pin(monkeypatch, _seal(value))
    with pytest.raises(ValueError, match="stored Elo order"):
        configs.load_tournament_config()


@pytest.mark.parametrize("map_count,team_size", [(1, 1), (3, 3), (5, 5)])
def test_v2_release_uses_its_declared_maps_and_mirrored_rosters(
    monkeypatch: pytest.MonkeyPatch, map_count: int, team_size: int
) -> None:
    value = config_descriptor(entrants=3, official=True)
    value["version"] = 2
    conditions = value["conditions"]
    conditions["map_sources"] = conditions["map_sources"][:map_count]
    conditions["games_per_opponent"] = 2 * map_count
    for team in ("team_a", "team_b"):
        conditions["rosters"][team] = conditions["rosters"][team][:team_size]
    _pin(monkeypatch, _seal(value))
    loaded = configs.load_tournament_config()
    assert loaded["conditions"] == conditions
    assert len(loaded["conditions"]["map_sources"]) == map_count
    assert len(loaded["conditions"]["rosters"]["team_a"]) == team_size


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("mutation", ["split", "identity", "roster", "weights"])
def test_official_protocol_limits_survive_descriptor_v2(
    monkeypatch: pytest.MonkeyPatch, version: int, mutation: str
) -> None:
    value = config_descriptor(entrants=12 if version == 1 else 3, official=True)
    value["version"] = version
    if mutation == "split":
        item = value["conditions"]["map_sources"][0]
        item["split"] = "validation"
        item["registered_map"]["split"] = "validation"
        match = "registered test-map identities"
    elif mutation == "identity":
        value["conditions"]["map_sources"][0]["registered_map"] = None
        match = "registered test-map identities"
    elif mutation == "roster":
        value["conditions"]["rosters"]["team_b"].reverse()
        match = "mirrored"
    else:
        value["analysis"]["opponent_weights"] = {
            row["entrant_id"]: 1.0 for row in value["participants"]
        }
        match = "equal opponent weights"
    _pin(monkeypatch, _seal(value))
    with pytest.raises(ValueError, match=match):
        configs.load_tournament_config()


@pytest.mark.parametrize("mutation", ["maps", "rosters"])
def test_v1_official_map_and_roster_counts_are_not_reinterpreted(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    value = config_descriptor(entrants=12, official=True)
    if mutation == "maps":
        value["conditions"]["map_sources"] = value["conditions"]["map_sources"][:1]
        match = "five test maps"
    else:
        for team in ("team_a", "team_b"):
            value["conditions"]["rosters"][team] = ["mage"]
        match = "five agents"
    _pin(monkeypatch, _seal(value))
    with pytest.raises(ValueError, match=match):
        configs.load_tournament_config()


@pytest.mark.parametrize(
    "count,official,identity,encoded_digest",
    [
        (
            2,
            False,
            "a532d8b2e927fe67e70df6739b5e11239cb0ec80b5d3f5faa94942ecd0273b8b",
            "e13fe0d6456d0dfb264ce9c2e19d5fd986b70bf056cbf89fb07801ceb24b5a6c",
        ),
        (
            13,
            False,
            "8de89d0acc9dd519455082ad9cce679638dd86baea6c146b9179b28ab5cb60df",
            "a517353b49020c266076437878937390d6c3e4885b120f472335fb16e995d9e3",
        ),
        (
            12,
            True,
            "6e2efc7b5094a6e1ecac188e35b822ac080155660bcf65b6e2d24291590dcd49",
            "c86ccaa5d7f1f8fc68b70a6ee3fc399cd121d7fc8186c0d51d0988416fea8b9b",
        ),
    ],
)
def test_v1_fixture_hashes_and_serialized_bytes_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    official: bool,
    identity: str,
    encoded_digest: str,
) -> None:
    value = config_descriptor(entrants=count, official=official)
    if official:
        _pin(monkeypatch, value)
    loaded = configs.load_tournament_config(value)
    assert loaded == value
    assert loaded["snapshot_id"] == identity
    assert sha256(configs.canonical_json(loaded)).hexdigest() == encoded_digest


def test_v2_saved_first_resume_does_not_read_the_installed_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = config_descriptor(entrants=7, official=True)
    saved["version"] = 2
    saved = _seal(saved)

    def unavailable() -> dict[str, Any]:
        raise AssertionError("Saved resolution consulted today's catalog")

    monkeypatch.setattr(configs, "_installed_catalog", unavailable)
    assert configs.resolve_tournament_config(None, official=True, saved=saved) == saved
    assert configs.resolve_tournament_config(saved, official=True, saved=saved) == saved
    conflict = copy.deepcopy(saved)
    conflict["participants"].pop()
    with pytest.raises(ValueError, match="saved resolved snapshot"):
        configs.resolve_tournament_config(_seal(conflict), official=True, saved=saved)
