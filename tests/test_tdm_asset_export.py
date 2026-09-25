"""Check export provenance and file handling without compiling simulator data.

Source changes after the CLI's initial checks must fail before compilation or writes.
Scenario sources and approvals must be version-2 drafts, which declare a Red Zone
depth; a version-1 source or approval is refused. Each scenario's approval is the
approved source that its manifest record names, the same for every scenario: with
none named, each scenario is approved at its own source. A named approval is
recorded with its own path, revision and bytes, must share the source's asset ID
and equal its source in everything except notes, and comes one per scenario. The
CLI passes the manifest's approved sources, so an untouched export re-exports byte
for byte.
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
from pydantic import ValidationError
from scripts.dev import export_tdm_assets as exporter
from scripts.dev.visual_debugger.authoring_models import (
    DevMapDraftV1,
    DevScenarioDraftV2,
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
    ResolvedEnvConfigV2,
)
from marl_battlegrounds.tasks import DEFAULT_TDM_RED_ZONE_DEPTH


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


_FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class _Inputs:
    records: tuple[TDMMapInfo, ...]
    scenarios: tuple[Path, ...]
    geometry: MapGeometry
    package: Path
    calls: list[str]
    approvals: tuple[Path, ...] | None = None

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
            approved_scenario_sources=self.approvals,
        )

    def approve_separately(self, scenario_id: int, revision: int) -> Path:
        # Approve one scenario at its own earlier revision file.
        source = self.scenarios[scenario_id - 1]
        draft = DevScenarioDraftV2.model_validate_json(source.read_bytes())
        approval = source.with_name(f"r{revision}.json")
        approval.write_bytes(
            _json_bytes(
                draft.model_copy(update={"revision": revision}).model_dump(mode="json")
            )
        )
        approvals = list(self.scenarios if self.approvals is None else self.approvals)
        approvals[scenario_id - 1] = approval
        self.approvals = tuple(approvals)
        return approval


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
        draft = new_scenario_draft(
            f"scenario_{scenario_id}", red_zone_depth=DEFAULT_TDM_RED_ZONE_DEPTH
        ).model_copy(update={"revision": 25})
        path = tmp_path / f"scenario_{scenario_id}" / "r25.json"
        path.parent.mkdir()
        path.write_bytes(_json_bytes(draft.model_dump(mode="json")))
        sources.append(path)
    data = _Inputs(tuple(records), tuple(sources), geometry, package, [])

    def compile_scenario(draft: DevScenarioDraftV2) -> SimpleNamespace:
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

    def resolved_config(_: object) -> ResolvedEnvConfigV2:
        return content.configuration

    monkeypatch.setattr(exporter, "build_resolved_env_config_v2", resolved_config)
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
        "approval": inputs.approve_separately(6, 24),
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
    # With no approvals named, each scenario is approved at its own source.
    for row in first.scenarios:
        assert row.source.revision == 25
        assert row.approved_source == row.source
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
    # A separate approval is reproduced only if the CLI passes the manifest's own.
    inputs.approve_separately(3, 24)
    seed = tmp_path / "seed"
    manifest = inputs.export(seed)
    assert manifest.scenarios[2].approved_source.revision == 24
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
    seed_files = {
        path.relative_to(seed): path.read_bytes() for path in seed.rglob("*.json")
    }
    fresh_files = {
        path.relative_to(destination): path.read_bytes()
        for path in destination.rglob("*.json")
    }
    assert len(seed_files) == 63
    assert fresh_files == seed_files


@pytest.mark.parametrize("changed", ("scenario", "approval"))
def test_cli_rejects_source_changes_between_preflight_and_snapshot(
    inputs: _Inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    approval = inputs.approve_separately(3, 24)
    manifest = inputs.export(tmp_path / "seed")
    inputs.calls.clear()
    source = inputs.scenarios[-1] if changed == "scenario" else approval
    original = source.read_bytes()
    destination = tmp_path / "fresh"
    export = exporter.export_tdm_assets

    def change_before_snapshot(
        *,
        map_records: tuple[TDMMapInfo, ...],
        scenario_sources: tuple[Path, ...],
        destination: Path,
        expected_scenario_source_sha256: Mapping[Path, str] | None = None,
        approved_scenario_sources: tuple[Path, ...] | None = None,
    ) -> TDMAssetManifest:
        source.write_bytes(original + b"\n")
        return export(
            map_records=map_records,
            scenario_sources=scenario_sources,
            destination=destination,
            expected_scenario_source_sha256=expected_scenario_source_sha256,
            approved_scenario_sources=approved_scenario_sources,
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


def test_each_scenario_is_approved_by_the_source_its_manifest_names(
    inputs: _Inputs, tmp_path: Path
) -> None:
    approval = inputs.approve_separately(6, 24)
    manifest = inputs.export(tmp_path / "output")
    for scenario_id, row in enumerate(manifest.scenarios, start=1):
        if scenario_id != 6:
            assert row.approved_source == row.source
            continue
        raw = approval.read_bytes()
        assert row.source.revision == 25
        assert row.approved_source == row.source.model_copy(
            update={
                "revision": 24,
                "source_path": approval.as_posix(),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )


@pytest.mark.parametrize("fault", ("asset_id", "count", "physics"))
def test_mismatched_approvals_are_refused(
    inputs: _Inputs, tmp_path: Path, fault: str
) -> None:
    destination = tmp_path / "output"
    if fault == "physics":
        approval = inputs.approve_separately(6, 24)
        draft = DevScenarioDraftV2.model_validate_json(approval.read_bytes())
        notes_only = draft.content.model_copy(update={"notes": "Approved setup."})
        approval.write_bytes(
            _json_bytes(
                draft.model_copy(update={"content": notes_only}).model_dump(mode="json")
            )
        )
        inputs.export(tmp_path / "notes_only")
        task = draft.content.task.model_copy(update={"score_threshold": 7})
        changed = draft.content.model_copy(update={"task": task})
        approval.write_bytes(
            _json_bytes(
                draft.model_copy(update={"content": changed}).model_dump(mode="json")
            )
        )
        with pytest.raises(
            ValueError, match="scenario 6 source must keep its approved"
        ):
            inputs.export(destination)
        assert not (destination / "manifest.json").exists()
        return
    if fault == "count":
        inputs.approvals = inputs.scenarios[:7]
        with pytest.raises(ValueError, match="exactly one approved source"):
            inputs.export(destination)
        assert not destination.exists()
        assert inputs.calls == []
        return
    # Scenario 6 named as approved by scenario 5's draft.
    inputs.approvals = (
        *inputs.scenarios[:5],
        inputs.scenarios[4],
        *inputs.scenarios[6:],
    )
    with pytest.raises(ValueError, match="scenario approval disagrees with source 6"):
        inputs.export(destination)
    assert not (destination / "manifest.json").exists()


@pytest.mark.parametrize("role", ("source", "approval"))
def test_version_1_scenario_drafts_are_refused(
    inputs: _Inputs, tmp_path: Path, role: str
) -> None:
    # The committed Scenario 1 r34 fixture is a version-1 draft with no depth.
    old = (_FIXTURES / "scenario_1_r34.json").read_bytes()
    assert json.loads(old)["schema"] == "dev-scenario-draft@1"
    path = inputs.scenarios[0]
    if role == "source":
        path.write_bytes(old)
    else:
        path = inputs.approve_separately(1, 24)
        path.write_bytes(old)
    destination = tmp_path / "output"
    with pytest.raises(ValidationError, match="dev-scenario-draft@2"):
        inputs.export(destination)
    assert not (destination / "manifest.json").exists()
    assert not (destination / "scenarios").exists()
    assert ("scenario_1" in inputs.calls) == (role == "approval")
