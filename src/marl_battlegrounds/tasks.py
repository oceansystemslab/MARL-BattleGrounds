"""Construct approved TDM matches and load fixed diagnostic scenarios.

Factories run on the host before ``reset`` or a compiled rollout. They return
ordinary ``EnvConfig`` values and never select training distributions or seeds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds._tdm_assets import (
    MapGeometry,
    TDMAssetSource,
    TDMMapInfo,
    TDMScenarioInfo,
    asset_manifest,
    map_geometry,
    scenario_content,
)
from marl_battlegrounds.core.config import (
    CANONICAL_PRODUCT_MOVEMENT_SCALE,
    resolve_agent_profile,
    validate_product_env_config,
    validate_scenario_initial_state,
)
from marl_battlegrounds.core.types import (
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENTS_PER_TEAM,
    NEUTRAL_CLASS_ID,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TASK_MODE_TDM,
    WARRIOR_CLASS_ID,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v1
from marl_battlegrounds.evaluation.models import canonical_digest_sha256

type AgentClassName = Literal["mage", "warrior", "hunter", "rogue", "priest"]

_CLASS_IDS: dict[AgentClassName, int] = {
    "mage": MAGE_CLASS_ID,
    "warrior": WARRIOR_CLASS_ID,
    "hunter": HUNTER_CLASS_ID,
    "rogue": ROGUE_CLASS_ID,
    "priest": PRIEST_CLASS_ID,
}
_CANONICAL_ROSTER: tuple[AgentClassName, ...] = tuple(_CLASS_IDS)
CANONICAL_TDM_EVALUATION_MAP_IDS = (17, 20, 25, 35, 39)


@dataclass(frozen=True)
class TDMScenario:
    """Approved config/state plus provenance; initialize with Core's public API.

    ``initialize_scenario_state(initial_state, config)`` returns the ordinary
    state, observation, action mask and info. Loading performs no transition.
    """

    info: TDMScenarioInfo
    config: EnvConfig
    initial_state: EnvState
    notes: str


def list_tdm_maps() -> tuple[TDMMapInfo, ...]:
    """Return all 40 base and 12 curriculum maps in stable numeric order."""
    return asset_manifest().maps


def list_tdm_scenarios() -> tuple[TDMScenarioInfo, ...]:
    """Return the eight approved scenarios in stable numeric order."""
    return asset_manifest().scenarios


def _map_info(map_id: int) -> TDMMapInfo:
    if type(map_id) is not int or not 0 <= map_id < len(list_tdm_maps()):
        raise ValueError("map_id must be an approved integer from 0 through 51")
    return list_tdm_maps()[map_id]


def _roster_ids(roster: Sequence[AgentClassName], *, name: str) -> tuple[int, ...]:
    if isinstance(roster, (str, bytes)) or not isinstance(
        cast(object, roster), Sequence
    ):
        raise TypeError(f"{name} must be an ordered sequence of class names")
    if not 1 <= len(roster) <= MAX_AGENTS_PER_TEAM:
        raise ValueError(f"{name} must contain one through five agents")
    for value in roster:
        if type(value) is not str or value not in _CLASS_IDS:
            raise ValueError(f"{name} has unsupported class {value!r}")
    return tuple(_CLASS_IDS[value] for value in roster) + (NEUTRAL_CLASS_ID,) * (
        MAX_AGENTS_PER_TEAM - len(roster)
    )


def _make_config(
    geometry: MapGeometry,
    class_ids: tuple[int, ...],
    team_sizes: tuple[int, int],
    *,
    score_threshold: int,
    max_steps: int,
    movement_scale: float = float(CANONICAL_PRODUCT_MOVEMENT_SCALE),
    shield_duration: int = 3,
    shield_speed: float = 2.0,
    wave_periods: tuple[int, int] = (5, 5),
) -> EnvConfig:
    config = EnvConfig(
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=score_threshold,
        max_steps=max_steps,
        map_width=geometry.map_width,
        map_height=geometry.map_height,
        obstacles=jnp.asarray(geometry.obstacles, dtype=jnp.float32),
        agent_profile=resolve_agent_profile(
            jnp.asarray(class_ids, dtype=jnp.int32),
            jnp.asarray(team_sizes, dtype=jnp.int32),
        ),
        ordinary_movement_distance_scale=movement_scale,
        team_spawn_pad_positions=jnp.asarray(
            geometry.team_spawn_pad_positions, dtype=jnp.float32
        ),
        spawn_shield_duration_steps=shield_duration,
        spawn_shield_movement_speed=shield_speed,
        team_respawn_wave_period_step_count=jnp.asarray(wave_periods, dtype=jnp.int32),
    )
    validate_product_env_config(config)
    return config


def make_standard_team_deathmatch_config(
    *,
    map_id: int,
    team_a_roster: Sequence[AgentClassName],
    team_b_roster: Sequence[AgentClassName],
    score_threshold: int = 20,
    max_steps: int = 300,
) -> EnvConfig:
    """Build TDM on an approved map with independently ordered 1-5-agent teams.

    Duplicate classes, asymmetric teams and Priest-only teams are supported.
    Each team's roster order selects its corresponding team-local spawn pads.
    Canonical movement, five-step waves and three-step spawn shields apply.
    """
    ids = _roster_ids(team_a_roster, name="team_a_roster") + _roster_ids(
        team_b_roster, name="team_b_roster"
    )
    return _make_config(
        map_geometry(_map_info(map_id)),
        ids,
        (len(team_a_roster), len(team_b_roster)),
        score_threshold=score_threshold,
        max_steps=max_steps,
    )


def make_canonical_team_deathmatch_evaluation_config(*, map_id: int) -> EnvConfig:
    """Build mirrored Mage/Warrior/Hunter/Rogue/Priest 5v5, K=20 and H=300.

    Only the five held-out evaluation maps are eligible. Corresponding classes
    occupy reflected spawn pads with the same y coordinate on opposite teams.
    """
    _map_info(map_id)
    if map_id not in CANONICAL_TDM_EVALUATION_MAP_IDS:
        raise ValueError(
            "canonical evaluation map_id must be one of "
            f"{CANONICAL_TDM_EVALUATION_MAP_IDS}"
        )
    return make_standard_team_deathmatch_config(
        map_id=map_id,
        team_a_roster=_CANONICAL_ROSTER,
        team_b_roster=_CANONICAL_ROSTER,
    )


def _initial_state_digest(state: EnvState) -> str:
    payload: dict[str, object] = {"schema": "dev-resolved-initial-state@1"}
    for field_name, value in zip(state._fields, state, strict=True):
        host = np.asarray(value)
        payload[field_name] = {
            "dtype": str(host.dtype),
            "shape": list(host.shape),
            "values": host.tolist(),
        }
    return canonical_digest_sha256(payload)


def load_tdm_scenario(scenario_id: int) -> TDMScenario:
    """Load one exact packaged scenario without development files or mutable saves.

    Config and state must match the approved compiled digests. A changed mechanics
    catalog fails explicitly rather than silently changing an approved scenario.
    """
    if type(scenario_id) is not int or not 1 <= scenario_id <= 8:
        raise ValueError("scenario_id must be an approved integer from 1 through 8")
    info = list_tdm_scenarios()[scenario_id - 1]
    content = scenario_content(info)
    resolved = content.configuration
    geometry = MapGeometry(
        map_width=resolved.map_width,
        map_height=resolved.map_height,
        obstacles=tuple(
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
            for row in resolved.obstacle_slots
        ),
        team_spawn_pad_positions=resolved.team_spawn_pad_positions,
    )
    config = _make_config(
        geometry,
        info.class_ids,
        info.team_sizes,
        score_threshold=resolved.team_deathmatch_score_threshold,
        max_steps=resolved.maximum_episode_steps,
        movement_scale=resolved.ordinary_movement_distance_scale,
        shield_duration=resolved.spawn_shield_duration_steps,
        shield_speed=resolved.spawn_shield_movement_speed,
        wave_periods=cast(tuple[int, int], resolved.team_respawn_wave_period_steps),
    )
    if (
        build_resolved_env_config_v1(config) != resolved
        or resolved.canonical_digest_sha256 != info.resolved_configuration_digest
    ):
        raise ValueError(
            "packaged scenario configuration disagrees with live mechanics"
        )
    snapshot = content.initial_snapshot.model_dump()
    snapshot.pop("schema_id")
    snapshot.pop("schema_version")
    snapshot["step_count"] = content.step_count
    float_fields = {"agent_positions", "current_health"}
    bool_fields = {"alive_mask", "has_previous_timestep_joint_action"}
    state_values: dict[str, Array] = {}
    for field_name in EnvState._fields:
        dtype = (
            jnp.float32
            if field_name in float_fields
            else jnp.bool_
            if field_name in bool_fields
            else jnp.int32
        )
        state_values[field_name] = jnp.asarray(snapshot[field_name], dtype=dtype)
    state = EnvState(**state_values)
    validate_scenario_initial_state(config, state)
    if _initial_state_digest(state) != info.resolved_initial_state_digest:
        raise ValueError("packaged scenario initial-state digest mismatch")
    if config.max_steps - int(state.step_count) != info.horizon:
        raise ValueError("packaged scenario horizon disagrees with its manifest")
    return TDMScenario(
        info=info, config=config, initial_state=state, notes=content.notes
    )


__all__ = [
    "CANONICAL_TDM_EVALUATION_MAP_IDS",
    "AgentClassName",
    "TDMAssetSource",
    "TDMMapInfo",
    "TDMScenario",
    "TDMScenarioInfo",
    "list_tdm_maps",
    "list_tdm_scenarios",
    "load_tdm_scenario",
    "make_canonical_team_deathmatch_evaluation_config",
    "make_standard_team_deathmatch_config",
]
