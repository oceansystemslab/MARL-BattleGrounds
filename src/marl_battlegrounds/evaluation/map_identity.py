"""Recorded map names, independent of simulator execution and policy observations."""

from __future__ import annotations

import re
from array import array
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field

from marl_battlegrounds._tdm_assets import (
    asset_manifest,
    map_geometry,
    map_id_aliases,
)
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    EvaluationEpisodeContext,
    EvaluationModel,
    ResolvedEnvConfigV1,
)


class RecordedMap(EvaluationModel):
    """Verified registered identity, or an honestly unnamed recorded layout."""

    map_id: Annotated[int, Field(ge=0, le=51)] | None
    technical_name: Annotated[str, Field(min_length=1)]
    display_name: Annotated[str, Field(min_length=1)]
    split: Literal["training", "validation", "test"] | None


def _float32(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(array("f", values))


def approved_map_id(asset_id: str | None, semantic_digest: str) -> int | None:
    """Recognize an explicitly named authored source only while it is unchanged."""
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
    entries = asset_manifest().maps
    if type(map_id) is not int or not 0 <= map_id < len(entries):
        raise ValueError("recorded map_id must identify an approved TDM map")
    info = entries[map_id]
    return _map_identity(map_id, info.name, info.split)


@lru_cache(maxsize=104)
def _registered_map_by_name(name: str) -> tuple[RecordedMap, int]:
    """Keep the saved name and number; find its unchanged current geometry."""
    for info in asset_manifest().maps:
        if info.name == name:
            return _registered_map(info.map_id), info.map_id
    for alias in map_id_aliases().maps:
        if alias.name == name:
            current = asset_manifest().maps[alias.current_map_id]
            return _map_identity(
                alias.map_id, alias.name, current.split
            ), current.map_id
    raise ValueError("recorded map name conflicts with the approved catalogue")


@lru_cache(maxsize=104)
def _authored_map(asset_id: str, revision: int, digest: str) -> RecordedMap | None:
    """Recognize only an exact current or historical authored source."""
    map_id = approved_map_id(asset_id, digest)
    if map_id is not None and revision == asset_manifest().maps[map_id].source.revision:
        return _registered_map(map_id)
    for alias in map_id_aliases().maps:
        if (
            alias.asset_id == asset_id
            and alias.revision == revision
            and alias.semantic_digest == digest
        ):
            return _registered_map_by_name(alias.name)[0]
    return None


@lru_cache(maxsize=128)
def registered_map_metadata(
    map_id: int, config: ResolvedEnvConfigV1
) -> tuple[AggregationKeyV1, ...]:
    """Verify declared geometry once at recording, then preserve its split/name."""
    identity = _registered_map(map_id)
    geometry = map_geometry(asset_manifest().maps[map_id])
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
    if (
        config.task_mode != 1
        or config.map_width != geometry.map_width
        or config.map_height != geometry.map_height
        or actual_obstacles != expected_obstacles
        or config.team_spawn_pad_positions != expected_pads
    ):
        raise ValueError("recorded map_id does not match the episode geometry")
    return (
        AggregationKeyV1(name="map_id", value=str(map_id)),
        AggregationKeyV1(name="map_name", value=identity.technical_name),
        AggregationKeyV1(name="map_origin", value="registered"),
        AggregationKeyV1(name="map_split", value=str(identity.split)),
    )


@lru_cache(maxsize=128)
def _recorded_map(bindings: tuple[tuple[str, str], ...], layout: str) -> RecordedMap:
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
    """Read explicit metadata; matching geometry never invents a historical split."""
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
            result = authored
    if result.map_id is not None:
        # EvaluationModel is frozen and Pydantic supplies its runtime hash.
        _, current_id = _registered_map_by_name(result.technical_name)
        registered_map_metadata(current_id, context.resolved_env_config)  # pyright: ignore[reportArgumentType]
    return result
