"""Check the built-in training content gate, frozen bindings and exact map bank.

Tests cover scientific layout identity, protected-content rejection, finite
closure, resource verification, compatibility after harmless repackaging and
failure before setup/resume effects. Content binding version 3 records the Red
Zone depth: bindings prepared at depths 0.0, 5.0 (the default) and 6.0 differ
in every map configuration identity, source row and scientific projection, all
hash REQUIRED_SCHEMA_BINDINGS_V4, and a resume uses the saved depth. Version 3
requires score_thresholds and red_zone_depth and applies Core's scalar depth
rules; versions 1 and 2 reject a depth field. The version 1 binding captured
before Red Zone scoring (tests/fixtures/training_content_binding_v1.json)
still reproduces its exact bytes, and it and a version 2 binding are refused
by prepare_training_content with the "saved before Red Zone scoring" message
before any preparation starts. A pinned export whose learner saved a version 1
or 2 binding under a different protected scenario closure (as after the
scenario republish) links as a declared export with unknown exposure and all
eight scenarios familiar, without an error; this holds against today's real
republished binding as well as a binding whose protected roots alone differ.
No learner or training run is created.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
from collections.abc import Callable
from hashlib import sha256
from operator import itemgetter
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import pytest
from pydantic import ValidationError

from marl_battlegrounds import _tdm_assets
from marl_battlegrounds._tdm_assets import MapGeometry
from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v2
from marl_battlegrounds.evaluation.models import (
    REQUIRED_SCHEMA_BINDINGS_V3,
    REQUIRED_SCHEMA_BINDINGS_V4,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.evaluation.recording_identity import (
    ordered_source_bank_identity,
)
from marl_battlegrounds.tasks import (
    balanced_spawn_configs,
    canonical_tournament_rosters,
    load_tdm_scenario,
    make_standard_team_deathmatch_config,
)
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    TrainingContentBinding,
    _content,
    checkpoints,
    prepare_training_content,
)

_V1_FIXTURE = Path(__file__).parent / "fixtures" / "training_content_binding_v1.json"
_BEFORE_RED_ZONE = (
    "This training content was saved before Red Zone scoring (content binding "
    "version {}, one point per kill). This version cannot rebuild it. Resuming "
    "that experiment needs the original source environment that created it."
)


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    payload["canonical_digest"] = canonical_digest_sha256(
        payload, exclude={"canonical_digest"}
    )
    return payload


def _validate(payload: dict[str, Any]) -> TrainingContentBinding:
    # Plain json.dumps keeps -0.0, which canonical JSON would turn into 0.0.
    return TrainingContentBinding.model_validate_json(json.dumps(payload))


def _saved_before_red_zone(version: int) -> dict[str, Any]:
    payload = json.loads(_V1_FIXTURE.read_bytes())
    if version == 2:
        # A valid version 2 record: two winning scores, one 42-row block each.
        payload["schema_version"] = 2
        payload["score_thresholds"] = [19, 20]
        payload["source_configurations"] = payload["source_configurations"] * 2
        raw = (
            json.dumps(
                payload["source_configurations"], sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode()
        payload["source_bank"]["canonical_digest"] = sha256(raw).hexdigest()
        _rehash(payload)
    return payload


def _unused_apply(
    variables: object, memory: object, inputs: SystemInput, keys: object
) -> SystemOutput:
    raise AssertionError("pinned_opponent_evidence never runs the method")


def test_prepared_bank_matches_verified_maps_and_both_spawn_choices(
    prepared: PreparedTrainingContent,
) -> None:
    assert len(prepared.binding.maps) == 52
    assert len(prepared.binding.protected_scenarios) == 8
    assert all(leaf.shape[0] == 42 for leaf in jax.tree.leaves(prepared.source_configs))
    team_a, team_b = canonical_tournament_rosters()
    for index in range(42):
        actual = jax.tree.map(itemgetter(index), prepared.source_configs)
        expected = make_standard_team_deathmatch_config(
            map_id=index, team_a_roster=team_a, team_b_roster=team_b
        )
        for got, wanted in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_array_equal(got, wanted)
        choices = balanced_spawn_configs(actual, num_envs=2)
        np.testing.assert_array_equal(
            choices.team_spawn_pad_positions[0], actual.team_spawn_pad_positions
        )
        np.testing.assert_array_equal(
            choices.team_spawn_pad_positions[1], actual.team_spawn_pad_positions[::-1]
        )
    identifier, references, _ = ordered_source_bank_identity(prepared.source_configs)
    assert prepared.binding.source_bank.canonical_digest == identifier
    assert prepared.binding.source_configurations == tuple(references)


def test_layout_identity_uses_float32_values_and_order_not_wrappers() -> None:
    geometry = _tdm_assets.map_geometry(_tdm_assets.asset_manifest().maps[20])
    payload = geometry.model_dump(mode="json")
    payload["map_width"] = float(np.nextafter(np.float64(geometry.map_width), np.inf))
    slightly_different = MapGeometry.model_validate_json(canonical_json_bytes(payload))
    assert _content._layout_identity(geometry) == _content._layout_identity(
        slightly_different
    )  # pyright: ignore[reportPrivateUsage]
    payload["map_width"] += 1
    changed = MapGeometry.model_validate_json(canonical_json_bytes(payload))
    assert _content._layout_identity(geometry) != _content._layout_identity(changed)  # pyright: ignore[reportPrivateUsage]
    reversed_pads = geometry.model_copy(
        update={"team_spawn_pad_positions": geometry.team_spawn_pad_positions[::-1]}
    )
    assert _content._layout_identity(geometry) != _content._layout_identity(
        reversed_pads
    )  # pyright: ignore[reportPrivateUsage]
    reversed_obstacles = geometry.model_copy(
        update={"obstacles": geometry.obstacles[::-1]}
    )
    assert _content._layout_identity(geometry) != _content._layout_identity(
        reversed_obstacles
    )  # pyright: ignore[reportPrivateUsage]


def test_binding_is_frozen_and_round_trips_through_json(
    prepared: PreparedTrainingContent,
) -> None:
    restored = TrainingContentBinding.model_validate_json(
        prepared.binding.model_dump_json()
    )
    assert restored == prepared.binding
    with pytest.raises(ValidationError, match="frozen"):
        restored.canonical_digest = "a" * 64
    resumed = prepare_training_content(expected=json.loads(restored.model_dump_json()))
    assert resumed.binding == prepared.binding


def test_resume_ignores_audit_locations_labels_and_resource_repackaging(
    prepared: PreparedTrainingContent,
) -> None:
    payload = prepared.binding.model_dump(mode="json")
    for row in payload["maps"]:
        row["info"]["name"] = "Relocated Map"
        row["info"]["source"]["source_path"] = "/different/location.json"
        row["info"]["source"]["source_sha256"] = "a" * 64
        row["info"]["resource_sha256"] = "b" * 64
    for row in payload["protected_scenarios"]:
        row["info"]["name"] = "Renamed Scenario"
        row["info"]["description"] = "New explanatory wording."
        row["info"]["source"]["source_path"] = "/different/scenario.json"
    resumed = prepare_training_content(expected=_rehash(payload))
    assert resumed.binding == prepared.binding
    assert resumed.binding.canonical_digest != payload["canonical_digest"]


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "eligibility",
        "kind",
        "split",
        "missing_map",
        "missing_scenario",
        "missing_controller",
        "missing_shared",
        "unknown_dependency",
        "cyclic_reference",
        "protected_alias",
        "partition_alias",
        "source_order",
        "false_digest",
    ],
)
def test_invalid_binding_rejected_before_collection_update_or_writer(
    prepared: PreparedTrainingContent,
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = prepared.binding.model_dump(mode="json")
    if mutation == "schema":
        payload["schema_version"] = 4
    elif mutation == "eligibility":
        payload["eligibility_version"] = 2
    elif mutation == "kind":
        payload["protected_scenarios"][0]["kind"] = "shared-definition"
    elif mutation == "split":
        payload["maps"][0]["info"]["split"] = "validation"
    elif mutation == "missing_map":
        payload["maps"].pop()
    elif mutation == "missing_scenario":
        payload["protected_scenarios"].pop()
    elif mutation == "missing_controller":
        payload["protected_scenarios"][0].pop("pressure")
    elif mutation == "missing_shared":
        payload["shared_definitions"].pop()
    elif mutation == "unknown_dependency":
        payload["dependencies"] = [{"kind": "external-opponent", "path": "weights"}]
    elif mutation == "cyclic_reference":
        payload["protected_scenarios"][0]["dependencies"] = [{"root": "self"}]
    elif mutation == "protected_alias":
        payload["maps"][0]["layout"] = payload["protected_scenarios"][0]["layout"]
        payload["maps"][0]["info"]["name"] = "Ordinary Training Map"
    elif mutation == "partition_alias":
        payload["maps"][0]["layout"] = payload["maps"][47]["layout"]
    elif mutation == "source_order":
        payload["source_configurations"].reverse()
    _rehash(payload)
    if mutation == "false_digest":
        payload["canonical_digest"] = "0" * 64
    counters = {"collection": 0, "update": 0, "writer": 0}
    recording = tmp_path / "episodes.jsonl"
    recording.write_bytes(b"Existing recording bytes\n")
    before = recording.read_bytes()
    with pytest.raises(ValueError):
        prepare_training_content(expected=payload)
        for name in counters:
            counters[name] += 1
        recording.write_bytes(b"Wrongly recovered")
    assert counters == {"collection": 0, "update": 0, "writer": 0}
    assert recording.read_bytes() == before


@pytest.mark.parametrize(
    "change", ["geometry", "rules", "bank_order", "pressure", "hypothesis"]
)
def test_resume_rejects_consistent_but_changed_scientific_content(
    prepared: PreparedTrainingContent,
    change: str,
) -> None:
    payload = prepared.binding.model_dump(mode="json")
    if change == "geometry":
        payload["maps"][0]["layout"]["canonical_digest"] = "c" * 64
    elif change == "rules":
        payload["shared_definitions"][2]["canonical_digest"] = "c" * 64
    elif change in {"pressure", "hypothesis"}:
        row = payload["protected_scenarios"][0]
        spec = row["qualification_specification"]
        if change == "pressure":
            row["pressure"]["canonical_digest"] = "c" * 64
            spec["pressure_protocol"] = row["pressure"]
        else:
            spec["hypothesis"] = "A changed scientific hypothesis."
        spec["canonical_digest_sha256"] = canonical_digest_sha256(
            spec, exclude={"canonical_digest_sha256"}
        )
    else:
        payload["source_configurations"].reverse()
        raw = (
            json.dumps(
                payload["source_configurations"], sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode()
        payload["source_bank"]["canonical_digest"] = sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="incompatible"):
        prepare_training_content(expected=_rehash(payload))


@pytest.mark.parametrize("failure", ["missing", "wrong_bytes"])
def test_installed_resources_are_reverified(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _tdm_assets._resource_bytes  # pyright: ignore[reportPrivateUsage]

    def changed(path: str) -> bytes:
        if path == "maps/0.json":
            if failure == "missing":
                raise FileNotFoundError(path)
            return original(path) + b" "
        return original(path)

    monkeypatch.setattr(_tdm_assets, "_resource_bytes", changed)
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_training_content()


def test_protected_layout_cannot_be_renamed_as_installed_training_map(
    prepared: PreparedTrainingContent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_factory = _content.make_standard_team_deathmatch_config
    scenario = load_tdm_scenario(1)
    ordinary = original_factory(
        map_id=0,
        team_a_roster=canonical_tournament_rosters()[0],
        team_b_roster=canonical_tournament_rosters()[1],
    )
    renamed = scenario.config._replace(agent_profile=ordinary.agent_profile)

    def factory(**kwargs: object) -> object:
        return (
            renamed
            if kwargs["map_id"] == 0
            else cast(Callable[..., object], original_factory)(**kwargs)
        )

    monkeypatch.setattr(_content, "make_standard_team_deathmatch_config", factory)
    with pytest.raises(ValueError, match="protected scenario layout"):
        prepare_training_content()
    assert (
        prepared.binding.maps[0].layout
        != prepared.binding.protected_scenarios[0].layout
    )


def test_protected_instruments_and_shared_rules_have_separate_roles(
    prepared: PreparedTrainingContent,
) -> None:
    binding = prepared.binding
    assert {row.identifier for row in binding.shared_definitions} == {
        "static-mechanics",
        "evaluation-schemas",
        "simulator-rules",
    }
    schedule = binding.protected_scenarios[0].qualification_specification.seed_schedule
    assert tuple(row.episode_seed for row in schedule.realized_seed_protocols) == (0, 1)
    assert all(
        row.root.identifier == "protected-scenario-root"
        for row in binding.protected_scenarios
    )
    assert all(
        row.qualification_specification.seed_schedule == schedule
        for row in binding.protected_scenarios
    )
    assert (
        len({row.pressure.canonical_digest for row in binding.protected_scenarios}) == 2
    )
    assert all(row.info.class_ids for row in binding.protected_scenarios)


def test_bad_binding_cannot_recover_a_real_writer_directory(
    prepared: PreparedTrainingContent,
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.evaluation.run_writer import RunWriter

    with RunWriter(tmp_path) as writer:
        run_dir = writer.run_dir
    before = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert Path("run_details.json") in before
    bad = prepared.binding.model_copy(update={"canonical_digest": "0" * 64})
    counters = {"writer": 0, "collection": 0, "update": 0}

    def setup() -> None:
        prepare_training_content(expected=bad)
        counters["writer"] += 1
        with RunWriter(resume_from=run_dir):
            counters["collection"] += 1
            counters["update"] += 1

    with pytest.raises(ValueError, match="binding digest mismatch"):
        setup()
    after = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert counters == {"writer": 0, "collection": 0, "update": 0}


def test_preparation_reads_each_resource_and_validates_each_scenario_once(
    prepared: PreparedTrainingContent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections import Counter

    from marl_battlegrounds import tasks

    reads: Counter[str] = Counter()
    validations = 0
    read = _tdm_assets._resource_bytes
    validate = tasks.validate_scenario_initial_state

    def read_once(path: str) -> bytes:
        reads[path] += 1
        return read(path)

    def validate_once(*args: object) -> None:
        nonlocal validations
        validations += 1
        cast(Callable[..., None], validate)(*args)

    monkeypatch.setattr(_tdm_assets, "_resource_bytes", read_once)
    monkeypatch.setattr(tasks, "validate_scenario_initial_state", validate_once)
    actual = prepare_training_content()
    assert reads == Counter(
        {
            **{f"maps/{i}.json": 1 for i in range(52)},
            **{f"scenarios/{i}.json": 1 for i in range(1, 9)},
        }
    )
    assert validations == 8
    assert actual.binding == prepared.binding


def test_version_3_bindings_record_their_depth_and_current_schemas(
    prepared: PreparedTrainingContent,
) -> None:
    team_a, team_b = canonical_tournament_rosters()
    bindings = {5.0: prepared.binding}
    for depth in (0.0, 6.0):
        other = prepare_training_content(red_zone_depth=depth)
        np.testing.assert_array_equal(
            other.source_configs.team_deathmatch_red_zone_depth, np.float32(depth)
        )
        bindings[depth] = other.binding
    np.testing.assert_array_equal(
        prepared.source_configs.team_deathmatch_red_zone_depth, np.float32(5.0)
    )
    schemas = canonical_digest_sha256({"bindings": REQUIRED_SCHEMA_BINDINGS_V4})
    assert schemas != canonical_digest_sha256({"bindings": REQUIRED_SCHEMA_BINDINGS_V3})
    for depth, binding in bindings.items():
        assert binding.schema_version == 3
        assert binding.red_zone_depth == depth
        assert binding.score_thresholds == (20,)
        assert binding.shared_definitions[1].identifier == "evaluation-schemas"
        assert binding.shared_definitions[1].canonical_digest == schemas
        assert {
            (row.configuration.identifier, row.configuration.version)
            for row in binding.maps
        } == {("resolved-env-config", 2)}
        config = make_standard_team_deathmatch_config(
            map_id=47, team_a_roster=team_a, team_b_roster=team_b, red_zone_depth=depth
        )
        assert (
            binding.maps[47].configuration.canonical_digest
            == build_resolved_env_config_v2(config).canonical_digest_sha256
        )
        dumped = binding.model_dump(mode="json")
        assert dumped["red_zone_depth"] == depth and dumped["score_thresholds"] == [20]
        assert _validate(dumped) == binding
    for left, right in ((0.0, 5.0), (0.0, 6.0), (5.0, 6.0)):
        first, second = bindings[left], bindings[right]
        assert first.canonical_digest != second.canonical_digest
        assert first.source_bank != second.source_bank
        assert all(
            a.configuration != b.configuration
            for a, b in zip(first.maps, second.maps, strict=True)
        )
        assert not set(first.source_configurations) & set(second.source_configurations)
        assert first.scientific_projection() != second.scientific_projection()
        assert [row.layout for row in first.maps] == [row.layout for row in second.maps]
        assert first.protected_scenarios == second.protected_scenarios
    assert prepare_training_content(expected=bindings[0.0]).binding == bindings[0.0]


def test_version_3_requires_a_valid_depth_and_old_versions_reject_one(
    prepared: PreparedTrainingContent,
) -> None:
    current = prepared.binding.model_dump(mode="json")
    for name in ("red_zone_depth", "score_thresholds"):
        payload = dict(current)
        payload.pop(name)
        with pytest.raises(ValidationError, match="requires score_thresholds"):
            _validate(_rehash(payload))
    for depth in (None, -0.0, -1.0, 1e-40, 1e39):
        with pytest.raises(ValidationError):
            _validate(_rehash({**current, "red_zone_depth": depth}))
    for depth in (True, "5.0", float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            _validate({**current, "red_zone_depth": depth})
    for version in (1, 2):
        old = _saved_before_red_zone(version)
        assert _validate(old).red_zone_depth is None
        for depth in (0.0, 5.0, None):
            with pytest.raises(ValidationError, match="has no red_zone_depth"):
                _validate(_rehash({**old, "red_zone_depth": depth}))
    for depth, error in (
        (5, TypeError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
    ):
        with pytest.raises(error):
            prepare_training_content(red_zone_depth=depth)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="map_width"):
        prepare_training_content(red_zone_depth=20.5)


def test_bindings_saved_before_red_zone_keep_their_bytes_and_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _V1_FIXTURE.read_bytes()
    saved = TrainingContentBinding.model_validate_json(raw)
    assert (saved.schema_version, saved.score_thresholds) == (1, (20,))
    assert saved.red_zone_depth is None
    assert (
        json.dumps(saved.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode() == raw
    assert saved.shared_definitions[1].canonical_digest == canonical_digest_sha256(
        {"bindings": REQUIRED_SCHEMA_BINDINGS_V3}
    )
    assert {row.configuration.version for row in saved.maps} == {1}
    version_2 = _validate(_saved_before_red_zone(2))
    assert "red_zone_depth" not in version_2.model_dump(mode="json")
    assert version_2.model_dump(mode="json")["score_thresholds"] == [19, 20]

    def no_preparation() -> object:
        pytest.fail("A binding saved before Red Zone scoring reached preparation")

    monkeypatch.setattr(_content, "asset_manifest", no_preparation)
    for version, expected in (
        (1, saved),
        (1, json.loads(raw)),
        (2, version_2),
        (2, _saved_before_red_zone(2)),
    ):
        with pytest.raises(ValueError) as error:
            prepare_training_content(expected=expected)
        assert str(error.value) == _BEFORE_RED_ZONE.format(version)


@pytest.mark.parametrize("version", [1, 2])
def test_an_export_trained_before_the_republish_links_as_declared(
    prepared: PreparedTrainingContent,
    version: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _saved_before_red_zone(version)
    identifier = "c" * 64
    export = tmp_path / "run" / "actors" / "final"
    learner = tmp_path / "run" / "checkpoints" / identifier
    descriptions: dict[Path, dict[str, object]] = {
        export.resolve(): {
            "kind": "actor",
            "actor_digest": "d" * 64,
            "metadata": {
                "checkpoint_id": identifier,
                "run_id": "old-run",
                "seed": 3,
                "env_steps": 40,
                "pinned_opponent": None,
            },
        },
        learner.resolve(): {
            "kind": "learner",
            "checkpoint_id": identifier,
            "actor_digest": "d" * 64,
            "metadata": {"run_id": "old-run", "config": {"seed": 3}},
            "counters": {"env_steps": 40},
            "collection": {"pinned_opponent": None, "content_binding": source},
        },
    }

    def read(path: Path) -> dict[str, object]:
        return descriptions[Path(path).resolve()]

    monkeypatch.setattr(checkpoints, "read_checkpoint_description", read)
    method = System("Old Export", _unused_apply)
    # Under the source binding's own protected closure the old record links.
    same = _validate(source)
    assert _content._export_origin(same, export) == ("verified", None)
    linked = _content.pinned_opponent_evidence(same, method, export=export)
    assert (linked["source"], linked["exposure"]) == ("verified export", "none")
    declared = {
        "source": "declared export",
        "exposure": "unknown",
        "controllers": [],
        "familiar_scenarios": list(range(1, 9)),
    }
    # Today's real binding (the installed scenarios republished at depth 5.0)
    # no longer matches the old record's protected closure.
    assert _content._export_origin(prepared.binding, export) == ("declared", None)
    assert (
        _content.pinned_opponent_evidence(prepared.binding, method, export=export)
        == declared
    )
    # The same holds when only the protected roots differ from the source.
    republished = prepared.binding.model_copy(
        update={
            "protected_scenarios": tuple(
                row.model_copy(
                    update={
                        "root": row.root.model_copy(
                            update={"canonical_digest": "e" * 64}
                        )
                    }
                )
                for row in prepared.binding.protected_scenarios
            )
        }
    )
    assert _content._export_origin(republished, export) == ("declared", None)
    record = _content.pinned_opponent_evidence(republished, method, export=export)
    assert record == declared
