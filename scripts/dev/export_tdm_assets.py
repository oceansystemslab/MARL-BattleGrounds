"""Mechanically package exact approved authoring revisions for installed users.

Normal regeneration reads the existing package manifest's explicit source paths.
No source is edited and no runtime API resolves a mutable latest revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np

from marl_battlegrounds._tdm_assets import (
    MapGeometry,
    ScenarioContent,
    TDMAssetManifest,
    TDMAssetSource,
    TDMMapInfo,
    TDMScenarioInfo,
)
from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v1
from marl_battlegrounds.evaluation.models import GlobalAnalysisSnapshotV1
from scripts.dev.visual_debugger.authoring_compiler import (
    compile_dev_map,
    compile_dev_scenario,
)
from scripts.dev.visual_debugger.authoring_models import (
    DevMapDraftV1,
    DevScenarioDraftV1,
)


def _write_json(path: Path, payload: object) -> str:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _source(
    path: Path, *, asset_id: str, revision: int, semantic_digest: str
) -> TDMAssetSource:
    return TDMAssetSource(
        asset_id=asset_id,
        revision=revision,
        source_path=path.as_posix(),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        semantic_digest=semantic_digest,
    )


def export_tdm_assets(
    *,
    map_sources: tuple[Path, ...],
    scenario_sources: tuple[Path, ...],
    destination: Path,
) -> TDMAssetManifest:
    """Export one complete, explicitly selected approved source set."""
    if len(map_sources) != 52 or len(scenario_sources) != 8:
        raise ValueError("export requires exactly 52 maps and eight scenarios")
    original_bytes = {
        path: path.read_bytes() for path in (*map_sources, *scenario_sources)
    }
    maps: list[TDMMapInfo] = []
    scenarios: list[TDMScenarioInfo] = []
    for map_id, path in enumerate(map_sources):
        draft = DevMapDraftV1.model_validate_json(original_bytes[path])
        if not draft.asset_id.startswith(f"tdm_map_id_{map_id}_"):
            raise ValueError(f"map source order disagrees with map_id {map_id}")
        compiled = compile_dev_map(draft)
        geometry = MapGeometry.model_validate(
            {
                "map_width": compiled.content.width,
                "map_height": compiled.content.height,
                "obstacles": tuple(
                    tuple(float(value) for value in row)
                    for row in np.asarray(compiled.obstacles)
                ),
                "team_spawn_pad_positions": tuple(
                    tuple(tuple(float(value) for value in point) for point in team)
                    for team in np.asarray(compiled.team_spawn_pad_positions)
                ),
            }
        )
        split = draft.asset_id.rsplit("_", 1)[1]
        info = TDMMapInfo.model_validate(
            {
                "map_id": map_id,
                "name": draft.content.name,
                "split": split,
                "curriculum": map_id >= 40,
                "source": _source(
                    path,
                    asset_id=draft.asset_id,
                    revision=draft.revision,
                    semantic_digest=compiled.semantic_digest,
                ),
                "resource_sha256": _write_json(
                    destination / "maps" / f"{map_id}.json",
                    geometry.model_dump(mode="json"),
                ),
            }
        )
        maps.append(info)
    for scenario_id, path in enumerate(scenario_sources, start=1):
        draft = DevScenarioDraftV1.model_validate_json(original_bytes[path])
        if draft.asset_id != f"scenario_{scenario_id}":
            raise ValueError(f"scenario source order disagrees with {scenario_id}")
        compiled = compile_dev_scenario(draft)
        state = compiled.initial_state
        snapshot = GlobalAnalysisSnapshotV1.model_validate_json(
            json.dumps(
                {
                    field_name: np.asarray(value).tolist()
                    for field_name, value in zip(state._fields, state, strict=True)
                    if field_name != "step_count"
                }
            )
        )
        content = ScenarioContent(
            configuration=build_resolved_env_config_v1(compiled.config),
            initial_snapshot=snapshot,
            step_count=int(state.step_count),
            notes=compiled.content.notes,
        )
        approval_path = path.with_name("r24.json") if scenario_id == 3 else path
        approval = DevScenarioDraftV1.model_validate_json(approval_path.read_bytes())
        if scenario_id == 3:
            approved_content = approval.content.model_dump(exclude={"notes"})
            if draft.content.model_dump(exclude={"notes"}) != approved_content:
                raise ValueError("Scenario 3 source must preserve approved r24 physics")
        info = TDMScenarioInfo(
            approved_source=_source(
                approval_path,
                asset_id=approval.asset_id,
                revision=approval.revision,
                semantic_digest=compiled.semantic_digest,
            ),
            scenario_id=scenario_id,
            name=compiled.content.name,
            description=compiled.content.description,
            source=_source(
                path,
                asset_id=draft.asset_id,
                revision=draft.revision,
                semantic_digest=compiled.semantic_digest,
            ),
            map_semantic_digest=compiled.map_semantic_digest,
            resolved_configuration_digest=compiled.resolved_configuration_digest,
            resolved_initial_state_digest=compiled.resolved_initial_state_digest,
            class_ids=tuple(
                int(value)
                for value in np.asarray(compiled.config.agent_profile.class_ids)
            ),
            team_sizes=(compiled.content.team_a_size, compiled.content.team_b_size),
            horizon=compiled.config.max_steps - int(state.step_count),
            resource_sha256=_write_json(
                destination / "scenarios" / f"{scenario_id}.json",
                content.model_dump(mode="json"),
            ),
        )
        scenarios.append(info)
    manifest = TDMAssetManifest(maps=tuple(maps), scenarios=tuple(scenarios))
    _write_json(destination / "manifest.json", manifest.model_dump(mode="json"))
    for path, original in original_bytes.items():
        if path.read_bytes() != original:
            raise RuntimeError(f"source changed while exporting: {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=Path("src/marl_battlegrounds/data/tdm")
    )
    args = parser.parse_args()
    destination = cast(Path, args.destination)
    manifest = TDMAssetManifest.model_validate_json(
        (destination / "manifest.json").read_bytes()
    )
    for info in (*manifest.maps, *manifest.scenarios):
        source = Path(info.source.source_path)
        if hashlib.sha256(source.read_bytes()).hexdigest() != info.source.source_sha256:
            raise ValueError(f"approved source bytes changed: {source}")
    export_tdm_assets(
        map_sources=tuple(Path(row.source.source_path) for row in manifest.maps),
        scenario_sources=tuple(
            Path(row.source.source_path) for row in manifest.scenarios
        ),
        destination=destination,
    )


if __name__ == "__main__":
    main()
