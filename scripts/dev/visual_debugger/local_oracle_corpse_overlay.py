"""Presentation-only local-Oracle corpse authorization.

The actor-input endpoint deliberately excludes non-self corpses because the
published policy observation does.  This module projects only locally visible
corpse bodies from the same authoritative epoch for debugger/replay painting;
it never changes policy observations, masks, targeting, or simulator state.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    ResolvedObstacleV1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    pov_presentation_key_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AuthorizedAgentV1,
    AuthorizedAuraIdV1,
    AuthorizedAuraModifierV1,
    AuthorizedBattlefieldSceneV1,
    AuthorizedClassMechanics,
    AuthorizedStatusV1,
)
from marl_battlegrounds.rendering.evaluation_wire_features import (
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1,
)
from marl_battlegrounds.rendering.vocabulary import (
    status_sort_key,
    status_token_id_from_catalog_status_id,
)
from scripts.dev.visual_debugger.presentation_protocol import (
    LocalOracleCorpseObservationV1,
    LocalOracleCorpseOverlayV1,
    LocalOracleCorpsePublicFactsV1,
    seal_local_oracle_corpse_overlay_v1,
)

_Float32Array = NDArray[np.float32]
_GEOMETRY_EPSILON_F32 = np.float32(1e-6)
_GEOMETRY_TOLERANCE_F32 = np.float32(1e-5)
_OBSTACLE_TYPE_PILLAR = 1
_OBSTACLE_TYPE_WALL = 2


def _float32_point(point: tuple[float, float]) -> _Float32Array:
    """Materialize one wire point with the simulator's float32 semantics."""
    return np.asarray(point, dtype=np.float32)


def _host_within_observation_radius_v1(
    observer_position: tuple[float, float],
    candidate_position: tuple[float, float],
    observation_radius: float,
) -> bool:
    """Mirror the simulator's float32 Euclidean visibility-radius gate."""
    observer = _float32_point(observer_position)
    candidate = _float32_point(candidate_position)
    distance = np.asarray(np.linalg.norm(candidate - observer), dtype=np.float32)
    return bool(distance <= np.float32(observation_radius))


def _host_segment_intersects_circle_v1(
    segment_start: _Float32Array,
    segment_end: _Float32Array,
    circle_center: _Float32Array,
    circle_radius: np.float32,
) -> bool:
    """Host equivalent of the canonical float32 pillar LOS kernel."""
    vector = segment_end - segment_start
    center_delta = circle_center - segment_start
    denominator = np.maximum(
        np.dot(vector, vector),
        _GEOMETRY_EPSILON_F32,
    )
    alpha = np.asarray(
        np.dot(center_delta, vector) / denominator,
        dtype=np.float32,
    )
    alpha_clipped = np.asarray(
        np.clip(alpha, np.float32(0.0), np.float32(1.0)),
        dtype=np.float32,
    )
    closest_point = segment_start + alpha_clipped * vector
    difference = closest_point - circle_center
    distance_squared = np.dot(difference, difference)
    padded_radius = np.float32(circle_radius + _GEOMETRY_TOLERANCE_F32)
    return bool(distance_squared <= np.float32(padded_radius * padded_radius))


def _host_segment_intersects_rotated_rect_v1(
    segment_start: _Float32Array,
    segment_end: _Float32Array,
    rectangle_center: _Float32Array,
    width: np.float32,
    height: np.float32,
    theta: np.float32,
) -> bool:
    """Host equivalent of the canonical float32 rotated-wall LOS kernel."""
    cos_theta = np.asarray(np.cos(np.float32(-theta)), dtype=np.float32)
    sin_theta = np.asarray(np.sin(np.float32(-theta)), dtype=np.float32)
    world_to_local = np.asarray(
        ((cos_theta, -sin_theta), (sin_theta, cos_theta)),
        dtype=np.float32,
    )
    segment_start_local = world_to_local @ (segment_start - rectangle_center)
    segment_end_local = world_to_local @ (segment_end - rectangle_center)

    half_width = np.float32(width / np.float32(2.0))
    half_height = np.float32(height / np.float32(2.0))
    minimum_x, maximum_x = -half_width, half_width
    minimum_y, maximum_y = -half_height, half_height
    vector_x = np.float32(segment_end_local[0] - segment_start_local[0])
    vector_y = np.float32(segment_end_local[1] - segment_start_local[1])
    vertical = bool(np.abs(vector_x) <= _GEOMETRY_EPSILON_F32)
    horizontal = bool(np.abs(vector_y) <= _GEOMETRY_EPSILON_F32)
    x_inside = bool(minimum_x <= segment_start_local[0] <= maximum_x)
    y_inside = bool(minimum_y <= segment_start_local[1] <= maximum_y)

    safe_vector_x = np.float32(1.0) if vertical else vector_x
    safe_vector_y = np.float32(1.0) if horizontal else vector_y
    alpha_x_1 = np.float32((minimum_x - segment_start_local[0]) / safe_vector_x)
    alpha_x_2 = np.float32((maximum_x - segment_start_local[0]) / safe_vector_x)
    alpha_y_1 = np.float32((minimum_y - segment_start_local[1]) / safe_vector_y)
    alpha_y_2 = np.float32((maximum_y - segment_start_local[1]) / safe_vector_y)
    entry_x = np.minimum(alpha_x_1, alpha_x_2)
    exit_x = np.maximum(alpha_x_1, alpha_x_2)
    entry_y = np.minimum(alpha_y_1, alpha_y_2)
    exit_y = np.maximum(alpha_y_1, alpha_y_2)
    if vertical:
        entry_x = np.float32(-np.inf if x_inside else np.inf)
        exit_x = np.float32(np.inf if x_inside else -np.inf)
    if horizontal:
        entry_y = np.float32(-np.inf if y_inside else np.inf)
        exit_y = np.float32(np.inf if y_inside else -np.inf)
    return bool(
        np.maximum(np.float32(0.0), np.maximum(entry_x, entry_y))
        <= np.minimum(np.float32(1.0), np.minimum(exit_x, exit_y))
    )


def _host_has_clear_line_of_sight_v1(
    observer_position: tuple[float, float],
    candidate_position: tuple[float, float],
    obstacles: tuple[ResolvedObstacleV1, ...],
) -> bool:
    """Evaluate canonical V1 static LOS without initializing a JAX backend.

    This development presentation mirror intentionally uses explicit float32
    host arithmetic. Differential tests against ``core.geometry`` are the
    authority guard: core remains the simulator contract while replay/debugger
    presentation remains independent of CUDA device discovery.
    """
    segment_start = _float32_point(observer_position)
    segment_end = _float32_point(candidate_position)
    for obstacle in obstacles:
        if not obstacle.is_active:
            continue
        center = _float32_point((obstacle.x, obstacle.y))
        if obstacle.obstacle_type_id == _OBSTACLE_TYPE_PILLAR:
            blocked = _host_segment_intersects_circle_v1(
                segment_start,
                segment_end,
                center,
                np.float32(obstacle.radius),
            )
        elif obstacle.obstacle_type_id == _OBSTACLE_TYPE_WALL:
            blocked = _host_segment_intersects_rotated_rect_v1(
                segment_start,
                segment_end,
                center,
                np.float32(obstacle.width),
                np.float32(obstacle.height),
                np.float32(obstacle.theta),
            )
        else:
            blocked = False
        if blocked:
            return False
    return True


def _status_durations(frame: EvaluationFrameV1, global_slot: int) -> tuple[int, ...]:
    snapshot = frame.snapshot
    return (
        *snapshot.slow_durations[global_slot],
        *snapshot.stun_durations[global_slot],
        snapshot.rogue_poison_anti_heal_durations[global_slot],
        snapshot.mage_burst_damage_amplification_durations[global_slot],
        snapshot.priest_blessing_of_freedom_slow_floor_durations[global_slot],
    )


def _corpse_statuses(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    *,
    global_slot: int,
) -> tuple[AuthorizedStatusV1, ...]:
    rows: list[AuthorizedStatusV1] = []
    for channel, (remaining, mechanic) in enumerate(
        zip(
            _status_durations(frame, global_slot),
            context.static_mechanics_catalog.status_channels,
            strict=True,
        )
    ):
        if remaining <= 0:
            continue
        source_class = context.static_mechanics_catalog.class_mechanics[
            mechanic.source_class_id
        ]
        rows.append(
            AuthorizedStatusV1(
                status_channel=channel,
                status_id=mechanic.status_id,
                family=mechanic.family,
                configured_duration_steps=mechanic.duration_steps,
                remaining_duration=remaining,
                source_class_id=mechanic.source_class_id,
                source_class_name=source_class.class_name,
                source_action_component=mechanic.source_action_component,
                magnitude_kind=mechanic.magnitude_kind,
                magnitude=mechanic.magnitude,
                breaks_on_positive_damage=mechanic.breaks_on_positive_damage,
                direct_sources=(),
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: status_sort_key(
                status_token_id_from_catalog_status_id(row.status_id)
            ),
        )
    )


def _corpse_agent(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    *,
    global_slot: int,
    authority_session_id: str,
    recipient_public_agent_id: str,
    recipient_team_id: int,
) -> AuthorizedAgentV1:
    roster = context.roster[global_slot]
    mechanics = context.resolved_env_config.slot_mechanics[global_slot]
    class_mechanics = context.static_mechanics_catalog.class_mechanics[roster.class_id]
    snapshot = frame.snapshot
    self_features = frame.base_observation.self_features[global_slot]
    relation: Literal["ally", "opponent"] = (
        "ally" if roster.configured_team_id == recipient_team_id else "opponent"
    )
    aura_rows: tuple[tuple[AuthorizedAuraIdV1, float], ...] = (
        (
            "mage_damage_amplification",
            self_features[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1],
        ),
        (
            "warrior_damage_mitigation",
            self_features[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1],
        ),
    )
    aura_modifiers = tuple(
        AuthorizedAuraModifierV1(aura_id=aura_id, multiplier=multiplier)
        for aura_id, multiplier in aura_rows
        if multiplier != 1.0
    )
    return AuthorizedAgentV1(
        presentation_key=pov_presentation_key_v1(
            authority_session_id=authority_session_id,
            recipient_public_agent_id=recipient_public_agent_id,
            public_agent_id=roster.public_agent_id,
        ),
        public_agent_id=roster.public_agent_id,
        relation=relation,
        team_id=roster.configured_team_id,
        class_id=roster.class_id,
        class_name=class_mechanics.class_name,
        position=cast(
            tuple[float, float],
            snapshot.agent_positions[global_slot],
        ),
        radius=mechanics.body_radius,
        life_state="corpse",
        current_health=snapshot.current_health[global_slot],
        maximum_health=mechanics.maximum_health,
        base_movement_speed=mechanics.base_movement_speed,
        effective_movement_speed=self_features[
            AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1
        ],
        observation_radius=mechanics.observation_radius,
        basic_interaction_radius=mechanics.basic_interaction_radius,
        ultimate_interaction_radius=mechanics.ultimate_interaction_radius,
        ultimate_cooldown_remaining=snapshot.ultimate_cooldowns[global_slot],
        spawn_shield_remaining=snapshot.spawn_shield_durations[global_slot],
        steps_until_out_of_combat=snapshot.steps_until_out_of_combat[global_slot],
        out_of_combat_delay_steps=mechanics.out_of_combat_delay_steps,
        out_of_combat_health_regeneration_fraction_per_step=(
            mechanics.out_of_combat_health_regeneration_fraction_per_step
        ),
        statuses=_corpse_statuses(
            context,
            frame,
            global_slot=global_slot,
        ),
        aura_modifiers=aura_modifiers,
    )


def build_local_oracle_corpse_overlay_v1(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    base_scene: AuthorizedBattlefieldSceneV1,
    *,
    authority_session_id: str,
    source_authority_epoch: int,
    recipient_public_agent_id: str,
    living_sensor_public_agent_ids: tuple[str, ...],
) -> LocalOracleCorpseOverlayV1:
    """Authorize dead bodies visible to one or more living local sensors."""
    if type(context) is not EvaluationEpisodeContextV1:
        raise TypeError("context must use the exact EvaluationEpisodeContextV1 root.")
    if type(frame) is not EvaluationFrameV1:
        raise TypeError("frame must use the exact EvaluationFrameV1 root.")
    if type(base_scene) is not AuthorizedBattlefieldSceneV1:
        raise TypeError("base_scene must use the exact authorized scene root.")
    if frame.episode_id != context.identity.episode_id:
        raise ValueError("corpse overlay frame and context must join one episode.")
    roster_by_id = {row.public_agent_id: row for row in context.roster}
    recipient = roster_by_id.get(recipient_public_agent_id)
    if recipient is None or not recipient.configured_active:
        raise ValueError("corpse overlay recipient must be configured active.")
    if type(living_sensor_public_agent_ids) is not tuple or len(
        living_sensor_public_agent_ids
    ) != len(set(living_sensor_public_agent_ids)):
        raise ValueError("corpse overlay sensors must be an ordered unique tuple.")
    sensor_slots: list[int] = []
    for public_id in living_sensor_public_agent_ids:
        sensor = roster_by_id.get(public_id)
        if (
            sensor is None
            or not sensor.configured_active
            or not frame.snapshot.alive_mask[sensor.global_slot]
            or sensor.configured_team_id != recipient.configured_team_id
        ):
            raise ValueError(
                "corpse overlay sensors must be living configured recipient allies."
            )
        sensor_slots.append(sensor.global_slot)
    base_ids = {row.public_agent_id for row in base_scene.agents}
    obstacles = context.resolved_env_config.obstacle_slots
    candidate_order = (
        *(
            row
            for row in context.roster
            if row.configured_team_id == recipient.configured_team_id
        ),
        *(
            row
            for row in context.roster
            if row.configured_team_id not in (0, recipient.configured_team_id)
        ),
    )
    observations: list[LocalOracleCorpseObservationV1] = []
    for candidate in candidate_order:
        if (
            not candidate.configured_active
            or frame.snapshot.alive_mask[candidate.global_slot]
            or candidate.public_agent_id in base_ids
        ):
            continue
        candidate_position = cast(
            tuple[float, float],
            frame.snapshot.agent_positions[candidate.global_slot],
        )
        observing_ids: list[str] = []
        for sensor_id, sensor_slot in zip(
            living_sensor_public_agent_ids,
            sensor_slots,
            strict=True,
        ):
            sensor_position = cast(
                tuple[float, float],
                frame.snapshot.agent_positions[sensor_slot],
            )
            radius = context.resolved_env_config.slot_mechanics[
                sensor_slot
            ].observation_radius
            if not _host_within_observation_radius_v1(
                sensor_position,
                candidate_position,
                radius,
            ):
                continue
            if _host_has_clear_line_of_sight_v1(
                sensor_position,
                candidate_position,
                obstacles,
            ):
                observing_ids.append(sensor_id)
        if not observing_ids:
            continue
        corpse = _corpse_agent(
            context,
            frame,
            global_slot=candidate.global_slot,
            authority_session_id=authority_session_id,
            recipient_public_agent_id=recipient_public_agent_id,
            recipient_team_id=recipient.configured_team_id,
        )
        observations.append(
            LocalOracleCorpseObservationV1(
                corpse=corpse,
                oracle_public_facts=LocalOracleCorpsePublicFactsV1(
                    public_agent_id=corpse.public_agent_id,
                    team_id=corpse.team_id,
                    class_id=corpse.class_id,
                    class_name=corpse.class_name,
                    position=corpse.position,
                    radius=corpse.radius,
                    life_state="corpse",
                    current_health=corpse.current_health,
                    maximum_health=corpse.maximum_health,
                    base_movement_speed=corpse.base_movement_speed,
                    effective_movement_speed=corpse.effective_movement_speed,
                    observation_radius=corpse.observation_radius,
                    basic_interaction_radius=corpse.basic_interaction_radius,
                    ultimate_interaction_radius=corpse.ultimate_interaction_radius,
                    ultimate_cooldown_remaining=(corpse.ultimate_cooldown_remaining),
                    spawn_shield_remaining=corpse.spawn_shield_remaining,
                    steps_until_out_of_combat=corpse.steps_until_out_of_combat,
                    out_of_combat_delay_steps=corpse.out_of_combat_delay_steps,
                    out_of_combat_health_regeneration_fraction_per_step=(
                        corpse.out_of_combat_health_regeneration_fraction_per_step
                    ),
                    statuses=corpse.statuses,
                    aura_modifiers=corpse.aura_modifiers,
                ),
                observing_sensor_public_agent_ids=tuple(observing_ids),
            )
        )
    recipient_key = pov_presentation_key_v1(
        authority_session_id=authority_session_id,
        recipient_public_agent_id=recipient_public_agent_id,
        public_agent_id=recipient_public_agent_id,
    )
    return seal_local_oracle_corpse_overlay_v1(
        source_episode_id=frame.episode_id,
        source_frame_index=frame.frame_index,
        source_simulator_step_count=frame.simulator_step_count,
        source_authority_epoch=source_authority_epoch,
        recipient_public_agent_id=recipient_public_agent_id,
        recipient_presentation_key=recipient_key,
        living_sensor_public_agent_ids=living_sensor_public_agent_ids,
        corpse_observations=tuple(observations),
    )


def validate_local_oracle_corpse_overlay_against_source_v1(
    overlay: LocalOracleCorpseOverlayV1,
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    base_scene: AuthorizedBattlefieldSceneV1,
    *,
    authority_session_id: str,
    source_authority_epoch: int,
    recipient_public_agent_id: str,
    living_sensor_public_agent_ids: tuple[str, ...],
) -> None:
    """Bind serialized corpse facts to the trusted same-epoch source frame.

    The browser can verify structural joins and content digests, but only this
    same-origin Python producer owns the hidden global snapshot needed to prove
    position and visibility. Rebuilding here makes a coordinated, re-sealed
    corpse/public-facts mutation fail before any presentation is emitted.
    """
    if type(overlay) is not LocalOracleCorpseOverlayV1:
        raise TypeError("overlay must use the exact local-Oracle root.")
    expected = build_local_oracle_corpse_overlay_v1(
        context,
        frame,
        base_scene,
        authority_session_id=authority_session_id,
        source_authority_epoch=source_authority_epoch,
        recipient_public_agent_id=recipient_public_agent_id,
        living_sensor_public_agent_ids=living_sensor_public_agent_ids,
    )
    if overlay != expected:
        raise ValueError(
            "local-Oracle corpse overlay changed from its authoritative "
            "same-epoch source."
        )


def compose_local_oracle_corpse_scene_v1(
    base_scene: AuthorizedBattlefieldSceneV1,
    overlay: LocalOracleCorpseOverlayV1,
    *,
    researcher_class_mechanics: tuple[AuthorizedClassMechanics, ...],
) -> AuthorizedBattlefieldSceneV1:
    """Compose authorized corpse bodies into a paint-only scene."""
    if type(base_scene) is not AuthorizedBattlefieldSceneV1:
        raise TypeError("base_scene must use the exact authorized scene root.")
    if type(overlay) is not LocalOracleCorpseOverlayV1:
        raise TypeError("overlay must use the exact local-Oracle root.")
    agents = (
        *base_scene.agents,
        *(row.corpse for row in overlay.corpse_observations),
    )
    represented = {row.class_id for row in agents}
    mechanics_by_id = {row.class_id: row for row in researcher_class_mechanics}
    if not represented <= set(mechanics_by_id):
        raise ValueError("corpse overlay lacks researcher class mechanics.")
    return replace(
        base_scene,
        agents=agents,
        class_mechanics=tuple(
            mechanics_by_id[class_id] for class_id in sorted(represented)
        ),
    )


__all__ = [
    "build_local_oracle_corpse_overlay_v1",
    "compose_local_oracle_corpse_scene_v1",
]
