"""Check export provenance and file handling without compiling simulator data.

Source changes after the CLI's initial checks must fail before compilation or writes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scripts.dev import export_tdm_assets as exporter
from scripts.dev.visual_debugger.authoring_models import (
    DevMapDraftV1,
    DevScenarioDraftV1,
    new_map_draft,
    new_scenario_draft,
)

from marl_battlegrounds._tdm_assets import (
    MapGeometry,
    ScenarioContent,
    TDMAssetManifest,
    TDMAssetSource,
    TDMMapInfo,
)
from marl_battlegrounds.evaluation.models import (
    GlobalAnalysisSnapshotV1,
    ResolvedEnvConfigV1,
)


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()


class _State:
    def __init__(self, snapshot: GlobalAnalysisSnapshotV1) -> None:
        values = snapshot.model_dump(exclude={"schema_id", "schema_version"})
        self._fields = (*values, "step_count")
        self._values = (*values.values(), 0)
        self.step_count = 0

    def __iter__(self) -> Iterator[object]:
        return iter(self._values)


@dataclass
class _Inputs:
    records: tuple[TDMMapInfo, ...]
    scenarios: tuple[Path, ...]
    geometry: MapGeometry
    package: Path
    calls: list[str]

    def compile_map(self, draft: DevMapDraftV1) -> SimpleNamespace:
        self.calls.append(draft.asset_id)
        return SimpleNamespace(
            content=draft.content,
            semantic_digest="a" * 64,
            obstacles=np.asarray(self.geometry.obstacles),
            team_spawn_pad_positions=np.asarray(self.geometry.team_spawn_pad_positions),
        )

    def export(self, destination: Path) -> TDMAssetManifest:
        return exporter.export_tdm_assets(
            map_records=self.records,
            scenario_sources=self.scenarios,
            destination=destination,
        )


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Inputs:
    installed = exporter.files("marl_battlegrounds").joinpath("data", "tdm")
    content = ScenarioContent.model_validate_json(
        installed.joinpath("scenarios", "1.json").read_bytes()
    )
    package = tmp_path / "package" / "data" / "tdm"
    package.mkdir(parents=True)
    for name in ("map_history.json", "map_id_aliases.json"):
        (package / name).write_bytes(b"\n" + installed.joinpath(name).read_bytes())

    def package_files(_: str) -> Path:
        return tmp_path / "package"

    monkeypatch.setattr(exporter, "files", package_files)
    blank = new_map_draft()
    pads = tuple((pad.position.x, pad.position.y) for pad in blank.content.spawn_pads)
    geometry = MapGeometry.model_validate(
        {
            "map_width": 20.0,
            "map_height": 10.0,
            "obstacles": ((0.0,) * 8,) * 32,
            "team_spawn_pad_positions": (pads[:5], pads[5:]),
        }
    )
    resource_sha = hashlib.sha256(
        _json_bytes(geometry.model_dump(mode="json"))
    ).hexdigest()
    records: list[TDMMapInfo] = []
    for map_id in range(52):
        draft = new_map_draft(f"map_{map_id}").model_copy(update={"revision": 1})
        path = tmp_path / f"map_{map_id}.json"
        raw = _json_bytes(draft.model_dump(mode="json"))
        path.write_bytes(raw)
        records.append(
            TDMMapInfo(
                map_id=map_id,
                name=draft.content.name,
                split="training",
                curriculum=map_id < 12,
                source=TDMAssetSource(
                    asset_id=draft.asset_id,
                    revision=1,
                    source_path=str(path),
                    source_sha256=hashlib.sha256(raw).hexdigest(),
                    semantic_digest="a" * 64,
                ),
                resource_sha256=resource_sha,
            )
        )
    sources: list[Path] = []
    for scenario_id in range(1, 9):
        draft = new_scenario_draft(f"scenario_{scenario_id}").model_copy(
            update={"revision": 25}
        )
        path = tmp_path / f"scenario_{scenario_id}" / "r25.json"
        path.parent.mkdir()
        path.write_bytes(_json_bytes(draft.model_dump(mode="json")))
        if scenario_id == 3:
            approval = draft.model_copy(update={"revision": 24})
            path.with_name("r24.json").write_bytes(
                _json_bytes(approval.model_dump(mode="json"))
            )
        sources.append(path)
    data = _Inputs(tuple(records), tuple(sources), geometry, package, [])

    def compile_scenario(draft: DevScenarioDraftV1) -> SimpleNamespace:
        data.calls.append(draft.asset_id)
        return SimpleNamespace(
            content=draft.content,
            semantic_digest="b" * 64,
            map_semantic_digest="c" * 64,
            resolved_configuration_digest="d" * 64,
            resolved_initial_state_digest="e" * 64,
            config=SimpleNamespace(
                max_steps=300,
                agent_profile=SimpleNamespace(class_ids=np.zeros(10, dtype=np.int32)),
            ),
            initial_state=_State(content.initial_snapshot),
        )

    monkeypatch.setattr(exporter, "compile_dev_map", data.compile_map)
    monkeypatch.setattr(exporter, "compile_dev_scenario", compile_scenario)

    def resolved_config(_: object) -> ResolvedEnvConfigV1:
        return content.configuration

    monkeypatch.setattr(exporter, "build_resolved_env_config_v1", resolved_config)
    return data


def test_all_approved_source_bytes_are_checked_before_writes(
    inputs: _Inputs, tmp_path: Path
) -> None:
    last_source = Path(inputs.records[-1].source.source_path)
    last_source.write_bytes(last_source.read_bytes() + b"\n")
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="approved source bytes changed"):
        inputs.export(destination)
    assert not destination.exists()
    assert inputs.calls == []


@pytest.mark.parametrize("field", ("asset_id", "revision", "semantic_digest", "name"))
def test_map_identity_mismatch_is_rejected(
    inputs: _Inputs, tmp_path: Path, field: str
) -> None:
    first = inputs.records[0]
    if field == "name":
        first = first.model_copy(update={"name": "Wrong name"})
    else:
        value = {"asset_id": "wrong_map", "revision": 2, "semantic_digest": "f" * 64}[
            field
        ]
        first = first.model_copy(
            update={"source": first.source.model_copy(update={field: value})}
        )
    inputs.records = (first, *inputs.records[1:])
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="map source differs from approved map_id 0"):
        inputs.export(destination)
    assert not destination.exists()


def test_compiled_resource_mismatch_is_rejected_before_it_is_written(
    inputs: _Inputs, tmp_path: Path
) -> None:
    inputs.records = (
        inputs.records[0].model_copy(update={"resource_sha256": "f" * 64}),
        *inputs.records[1:],
    )
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="compiled geometry differs"):
        inputs.export(destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "changed", ("map", "scenario", "approval", "history", "aliases")
)
def test_export_detects_mid_export_changes(
    inputs: _Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    path = {
        "map": Path(inputs.records[-1].source.source_path),
        "scenario": inputs.scenarios[-1],
        "approval": inputs.scenarios[2].with_name("r24.json"),
        "history": inputs.package / "map_history.json",
        "aliases": inputs.package / "map_id_aliases.json",
    }[changed]
    original = path.read_bytes()

    def change_source(draft: DevMapDraftV1) -> SimpleNamespace:
        if not inputs.calls:
            path.write_bytes(original + b"\n")
        return inputs.compile_map(draft)

    monkeypatch.setattr(exporter, "compile_dev_map", change_source)
    with pytest.raises(RuntimeError, match="changed while exporting"):
        inputs.export(tmp_path / "output")
    assert path.read_bytes() == original + b"\n"


def test_repeated_exports_are_exact_and_keep_history_alias_bytes(
    inputs: _Inputs, tmp_path: Path
) -> None:
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = inputs.export(first_dir)
    second = inputs.export(second_dir)
    assert first == second
    assert first.maps == inputs.records
    assert len(first.scenarios) == 8
    assert first.scenarios[2].source.revision == 25
    assert first.scenarios[2].approved_source.revision == 24
    first_files = {
        path.relative_to(first_dir): path.read_bytes()
        for path in first_dir.rglob("*.json")
    }
    second_files = {
        path.relative_to(second_dir): path.read_bytes()
        for path in second_dir.rglob("*.json")
    }
    assert first_files == second_files
    assert len(first_files) == 63
    for name in ("map_history.json", "map_id_aliases.json"):
        assert first_files[Path(name)] == (inputs.package / name).read_bytes()
    assert inputs.calls.count("scenario_8") == 2


def test_cli_exports_to_a_fresh_directory_from_the_installed_manifest(
    inputs: _Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = inputs.export(tmp_path / "seed")
    destination = tmp_path / "fresh"
    monkeypatch.setattr(exporter, "asset_manifest", lambda: manifest)
    monkeypatch.setattr(
        sys, "argv", ["export_tdm_assets", "--destination", str(destination)]
    )
    exporter.main()
    assert (
        TDMAssetManifest.model_validate_json(
            (destination / "manifest.json").read_bytes()
        )
        == manifest
    )


@pytest.mark.parametrize("changed", ("scenario", "approval"))
def test_cli_rejects_source_changes_between_preflight_and_snapshot(
    inputs: _Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    manifest = inputs.export(tmp_path / "seed")
    inputs.calls.clear()
    source = (
        inputs.scenarios[-1]
        if changed == "scenario"
        else inputs.scenarios[2].with_name("r24.json")
    )
    original = source.read_bytes()
    destination = tmp_path / "fresh"
    export = exporter.export_tdm_assets

    def change_before_snapshot(
        *,
        map_records: tuple[TDMMapInfo, ...],
        scenario_sources: tuple[Path, ...],
        destination: Path,
        expected_scenario_source_sha256: Mapping[Path, str] | None = None,
    ) -> TDMAssetManifest:
        source.write_bytes(original + b"\n")
        return export(
            map_records=map_records,
            scenario_sources=scenario_sources,
            destination=destination,
            expected_scenario_source_sha256=expected_scenario_source_sha256,
        )

    monkeypatch.setattr(exporter, "asset_manifest", lambda: manifest)
    monkeypatch.setattr(exporter, "export_tdm_assets", change_before_snapshot)
    monkeypatch.setattr(
        sys, "argv", ["export_tdm_assets", "--destination", str(destination)]
    )
    with pytest.raises(ValueError, match="approved source bytes changed"):
        exporter.main()
    assert inputs.calls == []
    assert not destination.exists()
    assert source.read_bytes() == original + b"\n"
