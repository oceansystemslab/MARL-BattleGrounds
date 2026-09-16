"""Read the installed TDM catalog and verify its packaged content.

This module owns package resource reads, catalog identities and historical map
aliases. It does not read authored source paths or mutable DevClient drafts.
Those paths are recorded evidence of where an asset came from.

The catalog and alias table are cached after successful loading. Map and
scenario content is read with its expected SHA-256 byte digest. Task factories
then check resolved configuration and initial-state meaning through Core.
No game, JAX device or training distribution is created here.
"""

from __future__ import annotations

import hashlib
import json
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
    """Identify the exact authored revision that supplied an asset.

    Attributes
    ----------
    asset_id : str
        Stable authored asset name.
    revision : int
        Positive saved revision number.
    source_path : str
        Recorded location of the authored file. Runtime loaders do not open it.
    source_sha256 : str
        SHA-256 hex digest of the authored file's bytes.
    semantic_digest : str
        Digest of the authored content under its declared semantic format.

    Notes
    -----
    This is a frozen strict record inherited from EvaluationModel. Digest fields
    require 64 lowercase hexadecimal characters. A valid record alone does not
    prove that its referenced source file exists.
    """

    asset_id: str
    revision: Annotated[int, Field(gt=0)]
    source_path: str
    source_sha256: _Digest
    semantic_digest: _Digest


class TDMMapInfo(EvaluationModel):
    """Describe one installed map and its declared scientific split.

    Attributes
    ----------
    map_id : int
        Current public ID, from 0 through 51.
    name : str
        Display name from the installed catalog.
    split : str
        Declared training, validation or test split.
    curriculum : bool
        Whether the map belongs to the curriculum subset.
    source : TDMAssetSource
        Exact authored revision and source-content identities.
    resource_sha256 : str
        Expected SHA-256 byte digest of the packaged map resource.

    Notes
    -----
    Catalog identity includes every field, not only the numeric map ID. Re-select
    a current entry rather than assuming an old numeric ID has the same meaning.
    """

    map_id: Annotated[int, Field(ge=0, le=51)]
    name: str
    split: Literal["training", "validation", "test"]
    curriculum: bool
    source: TDMAssetSource
    resource_sha256: _Digest


class TDMScenarioInfo(EvaluationModel):
    """Describe an installed scenario and its expected resolved start.

    Attributes
    ----------
    scenario_id : int
        Public scenario number, from 1 through 8.
    approved_source : TDMAssetSource
        Authored source revision recorded at approval.
    name, description : str
        Catalog name and explanation of the scenario.
    source : TDMAssetSource
        Source revision used for the packaged content.
    map_semantic_digest : str
        Identity of the embedded map's semantic content.
    resolved_configuration_digest : str
        Expected identity of the resolved simulator configuration.
    resolved_initial_state_digest : str
        Expected identity of the authored initial-state values and storage layout.
    class_ids : tuple[int, ...]
        Declared class IDs in simulator slot order, including inactive padding.
    team_sizes : tuple[int, int]
        Declared active sizes for Team A and Team B.
    horizon : int
        Positive number of transitions remaining from the authored starting step.
    resource_sha256 : str
        Expected SHA-256 byte digest of the packaged scenario resource.

    Notes
    -----
    Task loading checks the declared resolved identities against current mechanics.
    Record shape/type validation alone does not play or qualify a scenario.
    """

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
    """Store the versioned installed map and scenario catalogs.

    Attributes
    ----------
    schema_id : str, default="marl_battlegrounds.tdm_assets"
        Fixed format identity.
    schema_version : int, default=1
        Supported format version.
    maps : tuple[TDMMapInfo, ...]
        Map entries; asset_manifest checks ordered IDs 0 through 51.
    scenarios : tuple[TDMScenarioInfo, ...]
        Scenario entries; asset_manifest checks ordered IDs 1 through 8.

    Notes
    -----
    Constructing the model validates its declared fields. The loader adds catalog
    coverage and ordering checks before caching the manifest.
    """

    schema_id: Literal["marl_battlegrounds.tdm_assets"] = (
        "marl_battlegrounds.tdm_assets"
    )
    schema_version: Literal[1] = 1
    maps: tuple[TDMMapInfo, ...]
    scenarios: tuple[TDMScenarioInfo, ...]


class TDMMapAlias(EvaluationModel):
    """Map one historical map identity to its current public ID.

    Attributes
    ----------
    map_id : int
        Historical numeric ID, from 0 through 51.
    current_map_id : int
        Current catalog ID, from 0 through 51.
    name, asset_id : str
        Historical display name and authored asset ID.
    revision : int
        Positive historical authored revision.
    semantic_digest : str
        Map geometry identity that must agree with the retained old catalog.

    Notes
    -----
    Aliases support historical reads. They do not rewrite old result files or
    authorize treating an arbitrary old number as a current map selection.
    """

    map_id: Annotated[int, Field(ge=0, le=51)]
    current_map_id: Annotated[int, Field(ge=0, le=51)]
    name: str
    asset_id: str
    revision: Annotated[int, Field(gt=0)]
    semantic_digest: _Digest


class TDMMapAliases(EvaluationModel):
    """Hold the versioned table used to recognize historical map identities.

    Attributes
    ----------
    schema_id : str, default="marl_battlegrounds.tdm_map_id_aliases"
        Fixed alias-table format identity.
    schema_version : int, default=1
        Supported alias-table version.
    maps : tuple[TDMMapAlias, ...]
        Historical-to-current map records. The loader checks full coverage,
        distinct historical names and matching geometry identities.
    """

    schema_id: Literal["marl_battlegrounds.tdm_map_id_aliases"] = (
        "marl_battlegrounds.tdm_map_id_aliases"
    )
    schema_version: Literal[1] = 1
    maps: tuple[TDMMapAlias, ...]


class MapGeometry(EvaluationModel):
    """Store map geometry already converted to the simulator's coordinate format.

    Attributes
    ----------
    map_width, map_height : float
        World-space map dimensions.
    obstacles : tuple
        Ordered obstacle rows. Each has type, x, y, radius, width, height, angle
        and activity values. Angles are already in radians.
    team_spawn_pad_positions : tuple
        Team A then Team B's ordered (x, y) spawn banks in world units.

    Notes
    -----
    Runtime loading does not convert angles or rearrange pads. This storage model
    does not replace Core's configuration checks, including fixed array capacities
    and physical clearance.
    """

    map_width: float
    map_height: float
    obstacles: tuple[_ObstacleRow, ...]
    team_spawn_pad_positions: tuple[tuple[_Point, ...], ...]


class _HistoricalMap(EvaluationModel):
    """Keep one frozen catalog entry beside its original runtime geometry.

    info is the previous TDMMapInfo; geometry is its MapGeometry. Both are
    immutable. The history loader checks their original resource-byte digest.
    """

    info: TDMMapInfo
    geometry: MapGeometry


class _MapHistory(EvaluationModel):
    """Read the private package format for the retained 52-map catalog.

    schema_id and schema_version select this storage format. maps contains
    ordered entries 0 through 51; the loader checks order and resource hashes.
    This record does not make old maps available for new evaluations.
    """

    schema_id: Literal["marl_battlegrounds.tdm_map_history"]
    schema_version: Literal[1]
    maps: tuple[_HistoricalMap, ...]


class ScenarioContent(EvaluationModel):
    """Store a scenario's packaged configuration, exact snapshot and notes.

    Attributes
    ----------
    configuration : ResolvedEnvConfigV1
        Full resolved configuration recorded for the scenario.
    initial_snapshot : GlobalAnalysisSnapshotV1
        Authored state fields under the snapshot's versioned storage format.
    step_count : int
        Simulator step at which the authored start begins.
    notes : str
        Authored explanation of the start and intended use.

    Notes
    -----
    tasks.load_tdm_scenario pairs these fields and checks the expected digests and
    remaining horizon. Deserializing this record alone does not run that check.
    """

    configuration: ResolvedEnvConfigV1
    initial_snapshot: GlobalAnalysisSnapshotV1
    step_count: int
    notes: str


def _resource_bytes(relative_path: str) -> bytes:
    """Read one known resource below the installed package's data/tdm directory.

    relative_path is an internal package-relative name. Return raw bytes; ordinary
    resource and file errors propagate. This helper does not sanitize arbitrary
    external paths and must only be called with owned resource names.
    """
    return (
        files("marl_battlegrounds").joinpath("data", "tdm", relative_path).read_bytes()
    )


@cache
def asset_manifest() -> TDMAssetManifest:
    """Load, validate and cache the installed TDM manifest.

    Returns
    -------
    TDMAssetManifest
        Ordered catalog with maps 0 through 51 and scenarios 1 through 8.

    Raises
    ------
    ValueError
        Ordered map or scenario IDs do not cover the required catalog.

    Notes
    -----
    Package read and Pydantic schema errors also propagate. The first successful
    call reads manifest.json; later calls reuse the cached frozen model. Content
    resource hashes are checked when each map or scenario is loaded.
    """
    manifest = TDMAssetManifest.model_validate_json(_resource_bytes("manifest.json"))
    if tuple(row.map_id for row in manifest.maps) != tuple(range(52)):
        raise ValueError("TDM package must contain ordered maps 0 through 51")
    if tuple(row.scenario_id for row in manifest.scenarios) != tuple(range(1, 9)):
        raise ValueError("TDM package must contain ordered scenarios 1 through 8")
    return manifest


@cache
def map_history() -> tuple[_HistoricalMap, ...]:
    """Load and cache the immutable geometry used by older recordings.

    Read map_history.json only on demand. Return 52 ordered frozen entries.
    Reject missing or repeated IDs, duplicate names or source IDs, and geometry
    whose original exporter JSON bytes do not match its catalog digest. File
    and model-validation errors propagate; no source draft is read or changed.
    """
    history = _MapHistory.model_validate_json(_resource_bytes("map_history.json"))
    if tuple(row.info.map_id for row in history.maps) != tuple(range(52)):
        raise ValueError("map history must contain ordered maps 0 through 51")
    if (
        len({row.info.name for row in history.maps}) != 52
        or len({row.info.source.asset_id for row in history.maps}) != 52
    ):
        raise ValueError("map history must have distinct names and source IDs")
    for row in history.maps:
        original = (
            json.dumps(
                row.geometry.model_dump(mode="json"), indent=2, ensure_ascii=False
            )
            + "\n"
        ).encode()
        if hashlib.sha256(original).hexdigest() != row.info.resource_sha256:
            raise ValueError(f"map history geometry digest mismatch: {row.info.map_id}")
    return history.maps


@cache
def map_id_aliases() -> TDMMapAliases:
    """Load and cache the checked historical-to-current map table.

    Returns
    -------
    TDMMapAliases
        Full alias coverage for all 52 historical and current IDs.

    Raises
    ------
    ValueError
        Coverage is incomplete or duplicated, old names/source IDs are not
        distinct, or an alias disagrees with the retained map's geometry identity.

    Notes
    -----
    The first successful call reads the alias table and retained map history.
    File and schema errors propagate. It does not change recorded historical IDs.
    """
    aliases = TDMMapAliases.model_validate_json(_resource_bytes("map_id_aliases.json"))
    if tuple(row.map_id for row in aliases.maps) != tuple(range(52)) or tuple(
        sorted(row.current_map_id for row in aliases.maps)
    ) != tuple(range(52)):
        raise ValueError("map aliases must cover all 52 old and current IDs once")
    if (
        len({row.name for row in aliases.maps}) != 52
        or len({row.asset_id for row in aliases.maps}) != 52
    ):
        raise ValueError("map aliases must have distinct old names and source IDs")
    historical = map_history()
    if any(
        row.semantic_digest
        != historical[row.current_map_id].info.source.semantic_digest
        for row in aliases.maps
    ):
        raise ValueError("map aliases must preserve each map's original geometry")
    return aliases


def current_map_id(info: TDMMapInfo) -> int:
    """Return a map ID only when all supplied catalog details are current.

    Parameters
    ----------
    info : TDMMapInfo
        Map entry selected from the installed catalog.

    Returns
    -------
    int
        The entry's current public ID.

    Raises
    ------
    ValueError
        Its ID is invalid or any field differs from the installed catalog entry.

    Notes
    -----
    This does not translate aliases. Callers with stale entries must select again
    through current discovery; historical readers use the alias table explicitly.
    """
    entries = asset_manifest().maps
    if (
        type(info.map_id) is not int
        or not 0 <= info.map_id < len(entries)
        or info != entries[info.map_id]
    ):
        raise ValueError(
            "map details do not match the installed catalogue; "
            "choose the map again with list_tdm_maps() from marl_battlegrounds.tasks"
        )
    return info.map_id


def _verified_resource(relative_path: str, digest: str) -> bytes:
    """Read a known package resource and check its exact SHA-256 digest.

    relative_path is an owned package-relative name. digest is the expected
    lowercase hex digest from the catalog. Return the original bytes on equality;
    raise ValueError on mismatch. File errors propagate without substitution.
    """
    payload = _resource_bytes(relative_path)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"TDM package content digest mismatch: {relative_path}")
    return payload


def map_geometry(info: TDMMapInfo) -> MapGeometry:
    """Read and verify the packaged geometry for a selected map.

    Parameters
    ----------
    info : TDMMapInfo
        Catalog entry giving the map ID and expected resource-byte digest.

    Returns
    -------
    MapGeometry
        Parsed geometry with the original ordered obstacle rows and spawn banks.

    Raises
    ------
    ValueError
        Resource bytes do not match the expected digest. Schema validation can
        also reject malformed content.

    Notes
    -----
    File errors propagate. This verifies content bytes, not physical validity;
    task construction runs Core's checks. Use a current catalog entry for new
    experiments. The function does not open the recorded authored source path.
    """
    return MapGeometry.model_validate_json(
        _verified_resource(f"maps/{info.map_id}.json", info.resource_sha256)
    )


def scenario_content(info: TDMScenarioInfo) -> ScenarioContent:
    """Read and verify the packaged content for a selected scenario.

    Parameters
    ----------
    info : TDMScenarioInfo
        Catalog entry with the scenario ID and expected resource-byte digest.

    Returns
    -------
    ScenarioContent
        Parsed resolved configuration, initial snapshot, starting step and notes.

    Raises
    ------
    ValueError
        Resource bytes do not match their recorded digest. Schema validation can
        also reject malformed content.

    Notes
    -----
    File errors propagate. tasks.load_tdm_scenario performs the further checks of
    live mechanics, state identity and horizon. This read creates no episode.
    """
    return ScenarioContent.model_validate_json(
        _verified_resource(f"scenarios/{info.scenario_id}.json", info.resource_sha256)
    )
