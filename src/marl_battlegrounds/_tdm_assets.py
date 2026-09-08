"""Frozen package content for the approved Team Deathmatch benchmark."""

from __future__ import annotations

import hashlib
from functools import cache
from importlib.resources import files
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from marl_battlegrounds.evaluation.models import (
    EvaluationModel,
    GlobalAnalysisSnapshotV1,
    ResolvedEnvConfigV1,
)

type _Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
type _Point = tuple[float, float]
type _ObstacleRow = tuple[float, float, float, float, float, float, float, float]


class TDMAssetSource(EvaluationModel):
    """Exact authored source identity; paths are provenance, never runtime inputs."""

    asset_id: str
    revision: Annotated[int, Field(gt=0)]
    source_path: str
    source_sha256: _Digest
    semantic_digest: _Digest


class TDMMapInfo(EvaluationModel):
    """One approved map and its immutable package/source identities."""

    map_id: Annotated[int, Field(ge=0, le=51)]
    name: str
    split: Literal["training", "validation", "test"]
    curriculum: bool
    source: TDMAssetSource
    resource_sha256: _Digest


class TDMScenarioInfo(EvaluationModel):
    """One approved scenario, preserving source and compiled content identities."""

    scenario_id: Annotated[int, Field(ge=1, le=8)]
    approved_source: TDMAssetSource
    name: str
    description: str
    source: TDMAssetSource
    map_semantic_digest: _Digest
    resolved_configuration_digest: _Digest
    resolved_initial_state_digest: _Digest
    class_ids: tuple[int, ...]
    team_sizes: tuple[int, int]
    horizon: Annotated[int, Field(gt=0)]
    resource_sha256: _Digest


class TDMAssetManifest(EvaluationModel):
    schema_id: Literal["marl_battlegrounds.tdm_assets"] = (
        "marl_battlegrounds.tdm_assets"
    )
    schema_version: Literal[1] = 1
    maps: tuple[TDMMapInfo, ...]
    scenarios: tuple[TDMScenarioInfo, ...]


class MapGeometry(EvaluationModel):
    """Authoring-compiled geometry; runtime loading performs no angle conversion."""

    map_width: float
    map_height: float
    obstacles: tuple[_ObstacleRow, ...]
    team_spawn_pad_positions: tuple[tuple[_Point, ...], ...]


class ScenarioContent(EvaluationModel):
    configuration: ResolvedEnvConfigV1
    initial_snapshot: GlobalAnalysisSnapshotV1
    step_count: int
    notes: str


def _resource_bytes(relative_path: str) -> bytes:
    return (
        files("marl_battlegrounds").joinpath("data", "tdm", relative_path).read_bytes()
    )


@cache
def asset_manifest() -> TDMAssetManifest:
    manifest = TDMAssetManifest.model_validate_json(_resource_bytes("manifest.json"))
    if tuple(row.map_id for row in manifest.maps) != tuple(range(52)):
        raise ValueError("TDM package must contain ordered maps 0 through 51")
    if tuple(row.scenario_id for row in manifest.scenarios) != tuple(range(1, 9)):
        raise ValueError("TDM package must contain ordered scenarios 1 through 8")
    return manifest


def _verified_resource(relative_path: str, digest: str) -> bytes:
    payload = _resource_bytes(relative_path)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"TDM package content digest mismatch: {relative_path}")
    return payload


def map_geometry(info: TDMMapInfo) -> MapGeometry:
    return MapGeometry.model_validate_json(
        _verified_resource(f"maps/{info.map_id}.json", info.resource_sha256)
    )


def scenario_content(info: TDMScenarioInfo) -> ScenarioContent:
    return ScenarioContent.model_validate_json(
        _verified_resource(f"scenarios/{info.scenario_id}.json", info.resource_sha256)
    )
