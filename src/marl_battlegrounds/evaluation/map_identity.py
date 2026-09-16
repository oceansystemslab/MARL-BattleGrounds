"""Read and verify recorded map names without changing simulator conditions.

Host-only helpers match explicit asset identity or registered metadata against
the current catalog, retained map history and approved aliases. Geometry alone
never grants an old recording a map ID or split. Custom layouts remain unnamed
unless the record provides truthful descriptive text. No JAX runtime is imported here.
"""

from __future__ import annotations

import re
from array import array
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field

from marl_battlegrounds._tdm_assets import (
    MapGeometry,
    asset_manifest,
    map_geometry,
    map_history,
    map_id_aliases,
)
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    EvaluationEpisodeContext,
    EvaluationModel,
    ResolvedEnvConfigV1,
)


class RecordedMap(EvaluationModel):
    """Validated display identity for one recorded layout.

    Attributes
    ----------
    map_id : Annotated[int, Field(ge=0, le=51)] | None
        Registered integer ID 0..51, or None when no approved identity is known.
    technical_name : Annotated[str, Field(min_length=1)]
        Nonempty stable source name or recorded layout identifier.
    display_name : Annotated[str, Field(min_length=1)]
        Nonempty readable label; custom maps use "Custom Map".
    split : Literal['training', 'validation', 'test'] | None
        "training", "validation", "test", or None when not established.

    This frozen host model is descriptive metadata, not a simulator config.
    """

    map_id: Annotated[int, Field(ge=0, le=51)] | None
    technical_name: Annotated[str, Field(min_length=1)]
    display_name: Annotated[str, Field(min_length=1)]
    split: Literal["training", "validation", "test"] | None


def _float32(values: tuple[float, ...]) -> tuple[float, ...]:
    """Round Python values to the float32 values stored in recorded geometry."""
    return tuple(array("f", values))


def approved_map_id(asset_id: str | None, semantic_digest: str) -> int | None:
    """Find a current map whose explicit authored-source identity matches.

    Parameters
    ----------
    asset_id : str | None
        Declared source asset name, or None.
    semantic_digest : str
        Declared digest of that source's semantic content.

    Returns
    -------
    int | None
        Current integer map ID, or None when no source name/digest pair matches.

    Notes
    -----
        This cached-catalog host lookup does not infer identity from geometry or
        validate a source revision number. It does not edit map metadata.
    """
    return next(
        (
            row.map_id
            for row in asset_manifest().maps
            if row.source.asset_id == asset_id
            and row.source.semantic_digest == semantic_digest
        ),
        None,
    )


def _map_identity(
    map_id: int, technical_name: str, split: Literal["training", "validation", "test"]
) -> RecordedMap:
    """Build a readable label from a verified ID, technical name and split."""
    prefix = r"^tdm[-_]map[-_]id[-_]\d+[-_]"
    name = re.sub(prefix, "", technical_name)
    name = re.sub(rf"[-_]{split}$", "", name)
    display = re.sub(r"[-_]+", " ", name).title()
    return RecordedMap(
        map_id=map_id,
        technical_name=technical_name,
        display_name=display,
        split=split,
    )


@lru_cache(maxsize=52)
def _registered_map(map_id: int) -> RecordedMap:
    """Look up a current map ID and reject IDs outside the approved catalog."""
    entries = asset_manifest().maps
    if type(map_id) is not int or not 0 <= map_id < len(entries):
        raise ValueError("recorded map_id must identify an approved TDM map")
    info = entries[map_id]
    return _map_identity(map_id, info.name, info.split)


@lru_cache(maxsize=104)
def _registered_map_by_name(name: str) -> tuple[RecordedMap, int]:
    """Resolve an exact current or approved historical technical name.

    Return the recorded identity and its catalog-family ID separately. Historical
    map numbers and labels remain unchanged; unknown names raise ValueError.
    """
    for info in asset_manifest().maps:
        if info.name == name:
            return _registered_map(info.map_id), info.map_id
    for row in map_history():
        if row.info.name == name:
            info = row.info
            return _map_identity(info.map_id, info.name, info.split), info.map_id
    for alias in map_id_aliases().maps:
        if alias.name == name:
            historical = map_history()[alias.current_map_id].info
            return _map_identity(
                alias.map_id, alias.name, historical.split
            ), alias.current_map_id
    raise ValueError("recorded map name conflicts with the approved catalogue")


@lru_cache(maxsize=156)
def _authored_map(
    asset_id: str, revision: int, digest: str
) -> tuple[RecordedMap, MapGeometry] | None:
    """Match an explicit asset name, revision and digest to current or approved old
    sources. Return its recorded name and that exact revision's geometry, or None.
    Check the current catalog first; load history only when it does not match.
    """
    map_id = approved_map_id(asset_id, digest)
    if map_id is not None and revision == asset_manifest().maps[map_id].source.revision:
        return _registered_map(map_id), map_geometry(asset_manifest().maps[map_id])
    for row in map_history():
        source = row.info.source
        if (
            source.asset_id == asset_id
            and source.revision == revision
            and source.semantic_digest == digest
        ):
            info = row.info
            return _map_identity(info.map_id, info.name, info.split), row.geometry
    for alias in map_id_aliases().maps:
        if (
            alias.asset_id == asset_id
            and alias.revision == revision
            and alias.semantic_digest == digest
        ):
            row = map_history()[alias.current_map_id]
            return (
                _map_identity(alias.map_id, alias.name, row.info.split),
                row.geometry,
            )
    return None


def _geometry_matches(config: ResolvedEnvConfigV1, geometry: MapGeometry) -> bool:
    """Compare a recorded config with one exact packaged geometry on the host.

    Check TDM mode, dimensions, every ordered float32 obstacle row and both
    ordered spawn banks. Roster and other rules do not change map identity.
    Neither immutable input is changed; this does not simulate or read files.
    """
    actual_obstacles = tuple(
        (
            float(row.obstacle_type_id),
            row.x,
            row.y,
            row.radius,
            row.width,
            row.height,
            row.theta,
            float(row.is_active),
        )
        for row in config.obstacle_slots
    )
    expected_obstacles = tuple(_float32(row) for row in geometry.obstacles)
    expected_pads = tuple(
        tuple(_float32(point) for point in team)
        for team in geometry.team_spawn_pad_positions
    )
    return (
        config.task_mode == 1
        and config.map_width == geometry.map_width
        and config.map_height == geometry.map_height
        and actual_obstacles == expected_obstacles
        and config.team_spawn_pad_positions == expected_pads
    )


@lru_cache(maxsize=128)
def registered_map_metadata(
    map_id: int, config: ResolvedEnvConfigV1
) -> tuple[AggregationKeyV1, ...]:
    """Verify a declared map's geometry and return stable recording keys.

    Parameters
    ----------
    map_id : int
        Current approved integer map ID.
    config : ResolvedEnvConfigV1
        Frozen ResolvedEnvConfigV1 for the actual episode.

    Returns
    -------
    tuple[AggregationKeyV1, ...]
        Four AggregationKeyV1 values naming map_id, map_name, map_origin and
        map_split. Successful results are cached for this immutable config.

    Raises
    ------
    ValueError
        The map ID is invalid, the task is not TDM, or dimensions,
        ordered obstacle slots or ordered spawn banks differ from the asset.

    Notes
    -----
        Host-only. This is exact registered-geometry verification, including spawn
        order; it does not accept an exchanged bank by guessing a source relation.
        Roster and other non-geometry rules do not establish or change map identity.
    """
    identity = _registered_map(map_id)
    geometry = map_geometry(asset_manifest().maps[map_id])
    if not _geometry_matches(config, geometry):
        raise ValueError("recorded map_id does not match the episode geometry")
    return (
        AggregationKeyV1(name="map_id", value=str(map_id)),
        AggregationKeyV1(name="map_name", value=identity.technical_name),
        AggregationKeyV1(name="map_origin", value="registered"),
        AggregationKeyV1(name="map_split", value=str(identity.split)),
    )


@lru_cache(maxsize=128)
def _recorded_map(bindings: tuple[tuple[str, str], ...], layout: str) -> RecordedMap:
    """Interpret immutable recorded keys without inventing historical identity.

    Explicit registered keys must contain a coherent ID, name and split. Custom
    keys cannot also claim registered metadata. Older free-form names remain
    descriptive only. Invalid explicit claims raise ValueError.
    """
    metadata = dict(bindings)
    origin = metadata.get("map_origin")
    if origin == "custom":
        if metadata != {"map_origin": "custom"}:
            raise ValueError("custom map metadata cannot claim a registered identity")
        return RecordedMap(
            map_id=None,
            technical_name=layout,
            display_name="Custom Map",
            split=None,
        )
    if origin == "registered":
        if set(metadata) != {"map_origin", "map_id", "map_name", "map_split"}:
            raise ValueError("recorded map identity requires ID, name and split")
        try:
            map_id = int(metadata["map_id"])
        except ValueError as error:
            raise ValueError("recorded map_id must be an integer") from error
        result, _ = _registered_map_by_name(metadata["map_name"])
        if map_id != result.map_id or metadata["map_split"] != result.split:
            raise ValueError("recorded map name or split conflicts with its map_id")
        return result
    if origin is not None:
        raise ValueError("recorded map_origin must be registered or custom")
    # Historical aggregation keys were free-form. A recorded name is useful,
    # but neither its text nor arbitrary old IDs establish an approved split.
    name = metadata.get("map_name")
    return RecordedMap(
        map_id=None,
        technical_name=name or layout,
        display_name=name or "Map name unavailable",
        split=None,
    )


def recorded_map(context: EvaluationEpisodeContext) -> RecordedMap:
    """Return a verified display identity for an episode's recorded layout.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Immutable evaluation context containing layout identity,
        aggregation keys and the exact resolved episode config.

    Returns
    -------
    RecordedMap
        RecordedMap preserving a verified current or approved historical name.
        Without such evidence, return a custom/descriptive identity with no
        registered map ID or split.

    Raises
    ------
    ValueError
        Explicit map keys conflict, a declared name is unknown, or
        the episode geometry disagrees with its map revision, or an explicit
        authored source conflicts with the registered metadata.

    Notes
    -----
        Host-only and read-only. An exact authored name/revision/digest may identify
        an older recording without keys. Matching geometry by itself is never used
        to invent a historical map assignment.
    """
    bindings = tuple(
        (row.name, row.value)
        for row in context.aggregation_keys
        if row.name in {"map_id", "map_name", "map_split", "map_origin"}
    )
    layout = context.identity.layout
    result = _recorded_map(bindings, layout.identifier)
    if not bindings:
        # Older authored recordings already name and fingerprint their source
        # layout. Recognize that explicit identity, never geometry alone.
        authored = _authored_map(
            layout.identifier, layout.version, layout.canonical_digest
        )
        if authored is not None:
            result, geometry = authored
            if not _geometry_matches(context.resolved_env_config, geometry):
                raise ValueError("recorded source does not match the episode geometry")
        return result
    if result.map_id is None:
        return result
    if layout.identifier != "resolved-layout":
        # An explicit source selects one version. A conflicting source cannot
        # borrow another version merely because its geometry would match.
        authored = _authored_map(
            layout.identifier, layout.version, layout.canonical_digest
        )
        if authored is None or authored[0] != result:
            raise ValueError("recorded source identity conflicts with its map metadata")
        if not _geometry_matches(context.resolved_env_config, authored[1]):
            raise ValueError("recorded source does not match the episode geometry")
        return result
    _, map_id = _registered_map_by_name(result.technical_name)
    current = asset_manifest().maps[map_id]
    if result.technical_name == current.name and _geometry_matches(
        context.resolved_env_config, map_geometry(current)
    ):
        return result
    historical = map_history()[map_id]
    if _geometry_matches(context.resolved_env_config, historical.geometry):
        return result
    raise ValueError("recorded map_id does not match the episode geometry")
