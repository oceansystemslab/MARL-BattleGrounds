"""Check the built-in training content gate, frozen bindings and exact map bank.

Tests cover scientific layout identity, protected-content rejection, finite
closure, resource verification, compatibility after harmless repackaging and
failure before setup/resume effects. No learner or training run is created.
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
from marl_battlegrounds.evaluation.models import (
    canonical_digest_sha256,
    canonical_json_bytes,
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
    prepare_training_content,
)


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    payload["canonical_digest"] = canonical_digest_sha256(
        payload, exclude={"canonical_digest"}
    )
    return payload


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
    if mutation in {"schema", "eligibility"}:
        payload["schema_version" if mutation == "schema" else "eligibility_version"] = 2
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
