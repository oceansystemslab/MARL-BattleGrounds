"""Focused CP2.1 frozen-decoder and NoSharedObs authority proofs."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from struct import pack, unpack
from typing import cast

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
)

from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    StaticMechanicsCatalogV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovActionMaskV1,
    ActorPovAxisMappingV1,
    ActorPovCurrentSliceV1,
    ActorPovFrameV1,
    ActorPovSpawnLifecycleV1,
    build_actor_pov_adjacent_transition_slice_v1,
    build_actor_pov_current_slice_v1,
)
from marl_battlegrounds.rendering import evaluation_wire_features as wire
from marl_battlegrounds.rendering.authorized_pov_scene import (
    NoSharedObsAuthorizedScenePartsV1,
    build_no_shared_obs_authorized_scene_v1,
    pov_presentation_key_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AUTHORIZED_CLASS_DOCUMENTATION_CATALOG_FINGERPRINT_V1,
    AuthorizedBattlefieldSceneV1,
    AuthorizedClassDocumentationProfileAvailableV1,
    AuthorizedClassDocumentationProfileUnavailableV1,
    AuthorizedClassMechanicsV2,
    AuthorizedSpawnShieldMechanicsAvailableV2,
    authorized_class_documentation_profile_v1,
)
from marl_battlegrounds.rendering.pov_scene import (
    build_actor_pov_analyzer_projection_v1,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def no_shared_trajectory() -> CapturedEvaluationTrajectory:
    return captured_evaluation_trajectory(transition_count=1)


def _current_slice(
    trajectory: CapturedEvaluationTrajectory,
    *,
    global_slot: int,
    frame_index: int = 0,
) -> ActorPovCurrentSliceV1:
    incoming = None
    if frame_index > 0:
        incoming = EvaluationTransitionViewV1(
            context=trajectory.context,
            start_frame=trajectory.frames[frame_index - 1],
            transition=trajectory.transitions[frame_index - 1],
            successor_frame=trajectory.frames[frame_index],
        )
    return build_actor_pov_current_slice_v1(
        trajectory.context,
        trajectory.frames[frame_index],
        global_slot=global_slot,
        incoming_transition_view=incoming,
    )


def _replace_tuple_item[T](
    values: tuple[T, ...], index: int, value: T
) -> tuple[T, ...]:
    rows = list(values)
    rows[index] = value
    return tuple(rows)


def _replace_current_slice(
    source: ActorPovCurrentSliceV1,
    *,
    frame: ActorPovFrameV1 | None = None,
    axis_mapping: ActorPovAxisMappingV1 | None = None,
) -> ActorPovCurrentSliceV1:
    payload = source.model_dump(mode="python")
    if frame is not None:
        payload["frame"] = frame.model_dump(mode="python")
    if axis_mapping is not None:
        payload["axis_mapping"] = axis_mapping.model_dump(mode="python")
    return ActorPovCurrentSliceV1.model_validate(payload)


def _replace_frame_rows(
    source: ActorPovCurrentSliceV1,
    *,
    self_features: tuple[float, ...] | None = None,
    ally_rows: tuple[tuple[float, ...], ...] | None = None,
    enemy_rows: tuple[tuple[float, ...], ...] | None = None,
) -> ActorPovCurrentSliceV1:
    payload = source.frame.model_dump(mode="python")
    if self_features is not None:
        payload["self_features"] = self_features
        if ally_rows is None:
            ally_rows = _replace_tuple_item(
                source.frame.ally_unit_features,
                source.selected_team_local_slot,
                self_features,
            )
    if ally_rows is not None:
        payload["ally_unit_features"] = ally_rows
    if enemy_rows is not None:
        payload["enemy_unit_features"] = enemy_rows
    frame = ActorPovFrameV1.model_validate(payload)
    return _replace_current_slice(source, frame=frame)


def _catalog_from_payload(payload: dict[str, object]) -> StaticMechanicsCatalogV1:
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )
    return StaticMechanicsCatalogV1.model_validate(payload)


def _catalog_with_status_duration(
    catalog: StaticMechanicsCatalogV1,
    *,
    channel: int,
    duration: int,
) -> StaticMechanicsCatalogV1:
    payload = catalog.model_dump(mode="python")
    status_rows = list(cast(tuple[dict[str, object], ...], payload["status_channels"]))
    status_rows[channel] = {**status_rows[channel], "duration_steps": duration}
    payload["status_channels"] = tuple(status_rows)
    return _catalog_from_payload(payload)


def _catalog_with_aura_values(
    catalog: StaticMechanicsCatalogV1,
    *,
    per_emitter_multiplier: float | None = None,
    clamp_value: float | None = None,
) -> StaticMechanicsCatalogV1:
    payload = catalog.model_dump(mode="python")
    aura_rows = list(cast(tuple[dict[str, object], ...], payload["aura_mechanics"]))
    changed = dict(aura_rows[0])
    if per_emitter_multiplier is not None:
        changed["per_emitter_multiplier"] = per_emitter_multiplier
    if clamp_value is not None:
        changed["clamp_value"] = clamp_value
    aura_rows[0] = changed
    payload["aura_mechanics"] = tuple(aura_rows)
    return _catalog_from_payload(payload)


def _catalog_with_unrepresented_hunter_damage(
    catalog: StaticMechanicsCatalogV1,
    value: float,
) -> StaticMechanicsCatalogV1:
    payload = catalog.model_dump(mode="python")
    class_rows = list(cast(tuple[dict[str, object], ...], payload["class_mechanics"]))
    class_rows[3] = {**class_rows[3], "basic_raw_damage": value}
    payload["class_mechanics"] = tuple(class_rows)
    return _catalog_from_payload(payload)


def _valid_documentation_catalog_leaf_mutations(
    catalog: StaticMechanicsCatalogV1,
) -> Iterator[tuple[str, StaticMechanicsCatalogV1]]:
    """Yield every valid historical mutation of a documented mutable leaf."""
    payload = catalog.model_dump(mode="python")
    payload["global_slow_floor"] = catalog.global_slow_floor + 0.01
    yield "global_slow_floor", _catalog_from_payload(payload)

    class_float_fields = (
        "maximum_health",
        "body_radius",
        "base_movement_speed",
        "observation_radius",
        "basic_interaction_radius",
        "basic_raw_damage",
        "basic_raw_healing",
        "ultimate_interaction_radius",
        "ultimate_raw_damage",
        "ultimate_raw_healing",
        "out_of_combat_health_regeneration_fraction_per_step",
    )
    for index, row in enumerate(catalog.class_mechanics):
        for field_name in (
            "class_name",
            *class_float_fields,
            "basic_target_mode",
            "ultimate_target_mode",
            "ultimate_cooldown_steps",
            "out_of_combat_delay_steps",
        ):
            payload = catalog.model_dump(mode="python")
            rows = list(cast(tuple[dict[str, object], ...], payload["class_mechanics"]))
            changed = dict(rows[index])
            if field_name == "class_name":
                historical_name = f"{row.class_name} Historical"
                value: object = historical_name
                class_names = list(cast(tuple[str, ...], payload["class_name_by_id"]))
                class_names[index] = historical_name
                payload["class_name_by_id"] = tuple(class_names)
            elif field_name == "basic_target_mode":
                value = "ally" if row.basic_target_mode != "ally" else "enemy"
            elif field_name == "ultimate_target_mode":
                value = "ally" if row.ultimate_target_mode != "ally" else "enemy"
            elif field_name in (
                "ultimate_cooldown_steps",
                "out_of_combat_delay_steps",
            ):
                value = cast(int, getattr(row, field_name)) + 1
            else:
                value = cast(float, getattr(row, field_name)) + 0.01
            changed[field_name] = value
            rows[index] = changed
            payload["class_mechanics"] = tuple(rows)
            yield (
                f"class_mechanics[{index}].{field_name}",
                _catalog_from_payload(payload),
            )

    for index, row in enumerate(catalog.status_channels):
        for field_name in (
            "status_id",
            "family",
            "source_class_id",
            "source_action_component",
            "duration_steps",
            "magnitude_kind",
            "magnitude",
            "breaks_on_positive_damage",
        ):
            payload = catalog.model_dump(mode="python")
            rows = list(cast(tuple[dict[str, object], ...], payload["status_channels"]))
            changed = dict(rows[index])
            if field_name == "status_id":
                changed[field_name] = f"{row.status_id}_historical"
            elif field_name == "family":
                changed[field_name] = (
                    "anti_heal" if row.family != "anti_heal" else "slow"
                )
            elif field_name == "source_class_id":
                changed[field_name] = row.source_class_id % 5 + 1
            elif field_name == "source_action_component":
                changed[field_name] = (
                    "basic" if row.source_action_component == "ultimate" else "ultimate"
                )
            elif field_name == "duration_steps":
                changed[field_name] = row.duration_steps + 1
            elif field_name == "magnitude_kind":
                if row.magnitude_kind == "none":
                    changed[field_name] = "movement_multiplier"
                    changed["magnitude"] = 0.73
                else:
                    changed[field_name] = "none"
                    changed["magnitude"] = None
            elif field_name == "magnitude":
                if row.magnitude is None:
                    # A null magnitude is coupled to the literal `none` kind, so
                    # the smallest valid historical mutation changes both leaves.
                    changed["magnitude_kind"] = "movement_multiplier"
                    changed[field_name] = 0.74
                else:
                    changed[field_name] = row.magnitude + 0.01
            else:
                changed[field_name] = not row.breaks_on_positive_damage
            rows[index] = changed
            payload["status_channels"] = tuple(rows)
            yield (
                f"status_channels[{index}].{field_name}",
                _catalog_from_payload(payload),
            )

    for index, row in enumerate(catalog.aura_mechanics):
        for field_name in (
            "emitter_class_id",
            "radius",
            "per_emitter_multiplier",
            "clamp_kind",
            "clamp_value",
        ):
            payload = catalog.model_dump(mode="python")
            rows = list(cast(tuple[dict[str, object], ...], payload["aura_mechanics"]))
            changed = dict(rows[index])
            if field_name == "emitter_class_id":
                changed[field_name] = row.emitter_class_id % 5 + 1
            elif field_name == "clamp_kind":
                changed[field_name] = (
                    "floor" if row.clamp_kind == "ceiling" else "ceiling"
                )
            else:
                changed[field_name] = cast(float, getattr(row, field_name)) + 0.01
            rows[index] = changed
            payload["aura_mechanics"] = tuple(rows)
            yield (
                f"aura_mechanics[{index}].{field_name}",
                _catalog_from_payload(payload),
            )


def _invalid_immutable_documentation_catalog_leaf_payloads(
    catalog: StaticMechanicsCatalogV1,
) -> Iterator[tuple[str, dict[str, object]]]:
    """Yield literal-only or fixed-axis facts that have no valid historical peer."""
    for field_name in ("health_unit", "spatial_unit", "duration_unit"):
        payload = catalog.model_dump(mode="python")
        payload[field_name] = "historical_unit"
        yield field_name, payload

    for section, index, field_name, value in (
        ("class_mechanics", 1, "class_id", 2),
        ("status_channels", 0, "status_channel_id", 1),
        ("status_channels", 0, "application_update", "replace_duration"),
        ("aura_mechanics", 0, "aura_id", "historical_aura"),
        ("aura_mechanics", 0, "beneficiary_relation", "opponents"),
        ("aura_mechanics", 0, "stacking_rule", "add_then_clamp"),
    ):
        payload = catalog.model_dump(mode="python")
        rows = list(cast(tuple[dict[str, object], ...], payload[section]))
        changed = dict(rows[index])
        changed[field_name] = value
        rows[index] = changed
        payload[section] = tuple(rows)
        yield f"{section}[{index}].{field_name}", payload


def _scene_bytes(parts: NoSharedObsAuthorizedScenePartsV1) -> bytes:
    return json.dumps(
        asdict(parts.scene),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def test_frozen_v1_names_cover_all_58_columns_and_decode_sentinels() -> None:
    names = (
        "AGENT_FEATURE_X_V1",
        "AGENT_FEATURE_Y_V1",
        "AGENT_FEATURE_RADIUS_V1",
        "AGENT_FEATURE_TEAM_ID_V1",
        "AGENT_FEATURE_ACTIVE_V1",
        "AGENT_FEATURE_ALIVE_V1",
        "AGENT_FEATURE_CLASS_ID_V1",
        "AGENT_FEATURE_BASE_MOVEMENT_SPEED_V1",
        "AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1",
        "AGENT_FEATURE_OBSERVATION_RADIUS_V1",
        "AGENT_FEATURE_BASIC_INTERACTION_RADIUS_V1",
        "AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS_V1",
        "AGENT_FEATURE_CURRENT_HEALTH_V1",
        "AGENT_FEATURE_MAX_HEALTH_V1",
        "AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1",
        "AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION_V1",
        "AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION_V1",
        "AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION_V1",
        "AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1",
        "AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER_V1",
        "AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER_V1",
        "AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1",
        "AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION_V1",
        "AGENT_FEATURE_STUN_ROGUE_POISON_DURATION_V1",
        "AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION_V1",
        "AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1",
        "AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1",
        "AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1",
        "AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1",
        "AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT_V1",
        "AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1",
        "AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1",
        "AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE_V1",
        "AGENT_FEATURE_CAPABILITY_BASIC_HEALING_V1",
        "AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION_V1",
        "AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION_V1",
        "AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION_V1",
        "AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION_V1",
        "AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1",
        "AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER_V1",
        "AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER_V1",
        "AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION_V1",
        "AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION_V1",
        "AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION_V1",
        "AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_DURATION_V1",
        "AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1",
        "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1",
        "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER_V1",
        "AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1",
        "AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1",
        "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS_V1",
        "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1",
        "AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS_V1",
        "AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1",
        "AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING_V1",
        "AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE_V1",
        "AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS_V1",
        "AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP_V1",
    )
    assert tuple(getattr(wire, name) for name in names) == tuple(range(58))

    row = [0.0] * 58
    scalar_values = {
        0: -3.25,
        1: 4.5,
        2: 0.75,
        3: 2.0,
        4: 1.0,
        5: 0.0,
        6: 4.0,
        7: 1.3,
        8: 0.65,
        9: 8.5,
        10: 1.5,
        11: 4.0,
        12: 40.0,
        13: 100.0,
        14: 6.0,
        29: 10.0,
        30: 1.23,
        31: 0.77,
        32: 12.5,
        33: 2.5,
        34: 13.0,
        50: 2.25,
        51: 1.15,
        52: 2.75,
        53: 0.85,
        54: 25.5,
        55: 17.5,
        56: 14.0,
        57: 0.125,
    }
    for index, value in scalar_values.items():
        row[index] = value
    for channel, column in enumerate(
        wire.AGENT_STATUS_REMAINING_DURATION_COLUMN_BY_CHANNEL_V1
    ):
        row[column] = float(channel + 1)
    for channel, column in enumerate(
        wire.AGENT_STATUS_ACTIVE_MAGNITUDE_COLUMN_BY_CHANNEL_V1
    ):
        if column is not None:
            row[column] = 0.51 + channel * 0.01
    for channel, column in enumerate(
        wire.AGENT_STATUS_CAPABILITY_DURATION_COLUMN_BY_CHANNEL_V1
    ):
        row[column] = float(channel + 11)
    for channel, column in enumerate(
        wire.AGENT_STATUS_CAPABILITY_MAGNITUDE_COLUMN_BY_CHANNEL_V1
    ):
        if column is not None:
            row[column] = 0.61 + channel * 0.01

    decoded = wire.decode_agent_feature_row_v1(tuple(row))
    assert decoded.position == (-3.25, 4.5)
    assert decoded.radius == 0.75
    assert (decoded.team_id, decoded.configured_active, decoded.alive) == (
        2,
        True,
        False,
    )
    assert decoded.class_id == 4
    assert (
        decoded.base_movement_speed,
        decoded.effective_movement_speed,
        decoded.observation_radius,
        decoded.basic_interaction_radius,
        decoded.ultimate_interaction_radius,
    ) == (1.3, 0.65, 8.5, 1.5, 4.0)
    assert (decoded.current_health, decoded.maximum_health) == (40.0, 100.0)
    assert decoded.ultimate_cooldown_remaining == 6
    assert decoded.status_remaining_duration_by_channel == tuple(range(1, 10))
    assert decoded.status_capability_duration_by_channel == tuple(range(11, 20))
    assert decoded.status_active_magnitude_by_channel == tuple(
        None if column is None else 0.51 + channel * 0.01
        for channel, column in enumerate(
            wire.AGENT_STATUS_ACTIVE_MAGNITUDE_COLUMN_BY_CHANNEL_V1
        )
    )
    assert decoded.status_capability_magnitude_by_channel == tuple(
        None if column is None else 0.61 + channel * 0.01
        for channel, column in enumerate(
            wire.AGENT_STATUS_CAPABILITY_MAGNITUDE_COLUMN_BY_CHANNEL_V1
        )
    )
    assert decoded.steps_until_out_of_combat == 10
    assert (
        decoded.mage_aura_damage_multiplier,
        decoded.warrior_aura_damage_multiplier,
    ) == (1.23, 0.77)
    assert (
        decoded.basic_raw_damage,
        decoded.basic_raw_healing,
        decoded.ultimate_cooldown_steps,
        decoded.ultimate_raw_healing,
        decoded.ultimate_raw_damage,
        decoded.out_of_combat_delay_steps,
        decoded.out_of_combat_health_regeneration_fraction_per_step,
    ) == (12.5, 2.5, 13, 25.5, 17.5, 14, 0.125)


@pytest.mark.parametrize("global_slot", (0, 5))
def test_no_shared_scene_matches_legacy_authorized_rows_without_self_duplicate(
    no_shared_trajectory: CapturedEvaluationTrajectory,
    global_slot: int,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=global_slot)
    legacy = build_actor_pov_analyzer_projection_v1(source)
    parts = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="authority-session",
    )

    by_public_id = {row.public_agent_id: row for row in parts.scene.agents}
    assert len(by_public_id) == len(parts.scene.agents)
    assert tuple(
        row.public_agent_id for row in parts.scene.agents if row.relation == "self"
    ) == (source.public_agent_id,)
    expected_visible_ids = {
        body.public_agent_id
        for body in legacy.scene.visible_bodies
        if body.public_agent_id != source.public_agent_id
    }
    assert set(by_public_id) == {source.public_agent_id, *expected_visible_ids}
    assert parts.next_decision_action_mask == source.frame.action_mask
    assert (parts.scene.map.width, parts.scene.map.height) == (
        legacy.scene.map.width,
        legacy.scene.map.height,
    )
    assert tuple(row.class_id for row in parts.scene.class_mechanics) == tuple(
        sorted({row.class_id for row in parts.scene.agents})
    )

    legacy_self = legacy.scene.self_actor
    new_self = by_public_id[source.public_agent_id]
    assert (
        new_self.position,
        new_self.radius,
        new_self.current_health,
        new_self.maximum_health,
        new_self.effective_movement_speed,
        new_self.ultimate_cooldown_remaining,
        new_self.steps_until_out_of_combat,
        new_self.spawn_shield_remaining,
    ) == (
        legacy_self.position,
        legacy_self.radius,
        legacy_self.current_health,
        legacy_self.max_health,
        legacy_self.effective_movement_speed,
        legacy_self.ultimate_cooldown_remaining,
        legacy_self.steps_until_out_of_combat,
        legacy_self.spawn_shield_remaining,
    )
    for body in legacy.scene.visible_bodies:
        if body.public_agent_id == source.public_agent_id:
            continue
        new = by_public_id[body.public_agent_id]
        assert (
            new.position,
            new.radius,
            new.team_id,
            new.class_id,
            new.current_health,
            new.maximum_health,
            new.effective_movement_speed,
            new.ultimate_cooldown_remaining,
            new.steps_until_out_of_combat,
        ) == (
            body.position,
            body.radius,
            body.team_id,
            body.class_id,
            body.current_health,
            body.max_health,
            body.effective_movement_speed,
            body.ultimate_cooldown_remaining,
            body.steps_until_out_of_combat,
        )


def test_team_b_lifecycle_is_absolute_sorted_and_hidden_assignees_are_absent(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=5)
    lifecycle_payload = source.frame.spawn_lifecycle.model_dump(mode="python")
    own_positions = tuple((15.0 + slot * 0.5, 1.0 + slot * 2.0) for slot in range(5))
    opponent_positions = tuple(
        (2.0 + slot * 0.5, 1.5 + slot * 2.0) for slot in range(5)
    )
    own_shields = (3, 2, 0, 0, 0)
    opponent_shields = (1, 0, 2, 0, 0)
    own_active = (True, True, False, False, False)
    opponent_active = (True, True, True, False, False)
    own_alive = (True, True, False, False, False)
    opponent_alive = (True, False, True, False, False)
    lifecycle_payload["spawn_pad_positions_by_team"] = (
        own_positions,
        opponent_positions,
    )
    lifecycle_payload["spawn_shield_actual_durations_by_team"] = (
        own_shields,
        opponent_shields,
    )
    lifecycle_payload["respawn_wave_period_step_count_by_team"] = (11, 13)
    lifecycle_payload["respawn_wave_countdowns_by_team"] = (4, 9)
    lifecycle_payload["active_mask_by_team"] = (own_active, opponent_active)
    lifecycle_payload["alive_mask_by_team"] = (own_alive, opponent_alive)
    lifecycle = ActorPovSpawnLifecycleV1.model_validate(lifecycle_payload)
    frame_payload = source.frame.model_dump(mode="python")
    frame_payload["spawn_lifecycle"] = lifecycle.model_dump(mode="python")
    source = _replace_current_slice(
        source,
        frame=ActorPovFrameV1.model_validate(frame_payload),
    )
    parts = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="team-b-authority",
    )

    assert tuple(
        (row.team_index, row.team_id) for row in parts.scene.respawn_waves
    ) == (
        (0, 1),
        (1, 2),
    )
    assert tuple(
        (row.team_id, row.team_local_slot) for row in parts.scene.spawn_pads
    ) == tuple((team_id, slot) for team_id in (1, 2) for slot in range(5))
    pads_by_key = {
        (row.team_id, row.team_local_slot): row for row in parts.scene.spawn_pads
    }
    for slot in range(5):
        team_a = pads_by_key[(1, slot)]
        assert (
            team_a.position,
            team_a.spawn_shield_remaining,
            team_a.configured_active,
            team_a.currently_alive,
        ) == (
            opponent_positions[slot],
            opponent_shields[slot],
            opponent_active[slot],
            opponent_alive[slot],
        )
        team_b = pads_by_key[(2, slot)]
        assert (
            team_b.position,
            team_b.spawn_shield_remaining,
            team_b.configured_active,
            team_b.currently_alive,
        ) == (
            own_positions[slot],
            own_shields[slot],
            own_active[slot],
            own_alive[slot],
        )
    waves_by_team = {row.team_id: row for row in parts.scene.respawn_waves}
    assert (
        waves_by_team[1].period_steps,
        waves_by_team[1].countdown_steps,
    ) == (13, 9)
    assert (
        waves_by_team[2].period_steps,
        waves_by_team[2].countdown_steps,
    ) == (11, 4)
    authorized_ids = {row.public_agent_id for row in parts.scene.agents}
    for pad in parts.scene.spawn_pads:
        if pad.assigned_public_agent_id is None:
            assert pad.assigned_presentation_key is None
        else:
            assert pad.assigned_public_agent_id in authorized_ids
    assert all(
        row.assigned_public_agent_id is None
        for row in parts.scene.spawn_pads
        if row.team_id == 1
    )


def test_pov_keys_are_stable_recipient_scoped_opaque_and_oracle_distinct(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    first = _current_slice(no_shared_trajectory, global_slot=0, frame_index=0)
    second = _current_slice(no_shared_trajectory, global_slot=0, frame_index=1)
    first_parts = build_no_shared_obs_authorized_scene_v1(
        first,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="stable-session",
    )
    second_parts = build_no_shared_obs_authorized_scene_v1(
        second,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="stable-session",
    )
    second_keys = {
        row.public_agent_id: row.presentation_key for row in second_parts.scene.agents
    }
    for row in first_parts.scene.agents:
        if row.public_agent_id in second_keys:
            assert second_keys[row.public_agent_id] == row.presentation_key
        assert row.presentation_key.startswith("pov_")
        assert not row.presentation_key.startswith("oracle_")
        assert row.public_agent_id not in row.presentation_key
        assert "slot" not in row.presentation_key
        assert len(row.presentation_key.removeprefix("pov_")) == 64
        assert row.presentation_key == pov_presentation_key_v1(
            authority_session_id="stable-session",
            recipient_public_agent_id=first.public_agent_id,
            public_agent_id=row.public_agent_id,
        )
    other_recipient_key = pov_presentation_key_v1(
        authority_session_id="stable-session",
        recipient_public_agent_id="agent-slot-1",
        public_agent_id="agent-slot-1",
    )
    assert other_recipient_key != first_parts.recipient_presentation_key


def test_hidden_row_payload_and_public_id_are_byte_inert(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    assert not source.frame.enemy_visibility_mask[0]
    hidden_payload = tuple(float(100 + index) for index in range(58))
    changed_enemy = _replace_tuple_item(
        source.frame.enemy_unit_features,
        0,
        hidden_payload,
    )
    changed = _replace_frame_rows(source, enemy_rows=changed_enemy)

    axis_payload = changed.axis_mapping.model_dump(mode="python")
    enemy_ids = list(
        cast(
            tuple[str, ...], axis_payload["enemy_observation_row_public_agent_id_by_id"]
        )
    )
    enemy_ids[0] = "hidden-agent-renamed"
    target_ids = list(
        cast(
            tuple[str | None, ...],
            axis_payload["target_action_recipient_public_agent_id_by_id"],
        )
    )
    target_ids[6] = "hidden-agent-renamed"
    axis_payload["enemy_observation_row_public_agent_id_by_id"] = tuple(enemy_ids)
    axis_payload["target_action_recipient_public_agent_id_by_id"] = tuple(target_ids)
    changed_axis = ActorPovAxisMappingV1.model_validate(axis_payload)
    changed = _replace_current_slice(changed, axis_mapping=changed_axis)

    original_parts = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="privacy-session",
    )
    changed_parts = build_no_shared_obs_authorized_scene_v1(
        changed,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="privacy-session",
    )
    assert _scene_bytes(original_parts) == _scene_bytes(changed_parts)


def test_authorized_row_mutation_changes_canonical_scene_bytes(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    self_row = list(source.frame.self_features)
    self_row[wire.AGENT_FEATURE_X_V1] += 0.125
    changed = _replace_frame_rows(source, self_features=tuple(self_row))
    original_parts = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="authorized-mutation-session",
    )
    changed_parts = build_no_shared_obs_authorized_scene_v1(
        changed,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="authorized-mutation-session",
    )
    assert _scene_bytes(original_parts) != _scene_bytes(changed_parts)
    assert (
        changed_parts.scene.agents[0].position[0] == self_row[wire.AGENT_FEATURE_X_V1]
    )


def test_previous_action_material_is_scene_byte_inert(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0, frame_index=1)
    frame_payload = source.frame.model_dump(mode="python")
    previous = cast(dict[str, object], frame_payload["previous_timestep_actions"])
    ally_move = list(
        cast(tuple[tuple[float, ...], ...], previous["ally_move_actions_one_hot"])
    )
    first_row = list(ally_move[0])
    first_row[0] = 9876.5
    ally_move[0] = tuple(first_row)
    previous["ally_move_actions_one_hot"] = tuple(ally_move)
    changed = _replace_current_slice(
        source,
        frame=ActorPovFrameV1.model_validate(frame_payload),
    )
    original_parts = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="previous-action-inert-session",
    )
    changed_parts = build_no_shared_obs_authorized_scene_v1(
        changed,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="previous-action-inert-session",
    )
    assert _scene_bytes(original_parts) == _scene_bytes(changed_parts)


def test_no_shared_scene_is_strict_json_and_excludes_privileged_material(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0, frame_index=1)
    parts = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="strict-scene-session",
    )
    adapter = TypeAdapter(AuthorizedBattlefieldSceneV1)
    encoded = adapter.dump_json(parts.scene)
    assert adapter.validate_json(encoded) == parts.scene

    payload = json.loads(encoded)
    extra = json.loads(encoded)
    extra["agents"][0]["global_slot"] = 0
    missing = json.loads(encoded)
    del missing["agents"][0]["relation"]
    coerced = json.loads(encoded)
    coerced["agents"][0]["team_id"] = "1"
    for poisoned in (extra, missing, coerced):
        with pytest.raises(ValidationError):
            adapter.validate_json(json.dumps(poisoned))

    oracle_poison = json.loads(encoded)
    oracle_poison["agents"][0]["relation"] = "oracle"
    oracle_poison["agents"][0]["radius"] += 0.125
    with pytest.raises(ValidationError, match="Oracle agent static facts"):
        adapter.validate_json(json.dumps(oracle_poison))

    cooldown_poison = json.loads(encoded)
    class_id = cooldown_poison["agents"][0]["class_id"]
    mechanics = next(
        row for row in cooldown_poison["class_mechanics"] if row["class_id"] == class_id
    )
    cooldown_poison["agents"][0]["ultimate_cooldown_remaining"] = (
        mechanics["ultimate_cooldown_steps"] + 1
    )
    with pytest.raises(ValidationError, match="cooldown remaining"):
        adapter.validate_json(json.dumps(cooldown_poison))

    keys: list[str] = []
    string_values: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, dict):
            mapping = cast(dict[object, object], value)
            for key, nested in mapping.items():
                keys.append(str(key).lower())
                collect(nested)
        elif isinstance(value, list):
            sequence = cast(list[object], value)
            for nested in sequence:
                collect(nested)
        elif isinstance(value, str):
            string_values.append(value.lower())

    collect(payload)
    forbidden_key_fragments = (
        "global_slot",
        "researcher",
        "oracle",
        "event",
        "metric",
        "reward",
        "previous_action",
    )
    assert not {
        key
        for key in keys
        if any(fragment in key for fragment in forbidden_key_fragments)
    }
    assert not any(value == "researcher" for value in string_values)
    assert not any(value.startswith("oracle_") for value in string_values)
    assert source.frame.source_frame_id.lower() not in string_values
    assert source.incoming_transition is not None
    assert source.incoming_transition.pov_transition_id.lower() not in string_values


def test_visible_self_diagonal_conflict_fails_closed(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    assert source.frame.ally_visibility_mask[source.selected_team_local_slot]
    diagonal = list(source.frame.ally_unit_features[source.selected_team_local_slot])
    diagonal[wire.AGENT_FEATURE_X_V1] += 0.25
    ally_rows = _replace_tuple_item(
        source.frame.ally_unit_features,
        source.selected_team_local_slot,
        tuple(diagonal),
    )
    changed = _replace_frame_rows(source, ally_rows=ally_rows)
    with pytest.raises(ValueError, match="self diagonal conflicts"):
        build_no_shared_obs_authorized_scene_v1(
            changed,
            public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
            authority_session_id="conflict-session",
        )


def test_false_self_diagonal_is_not_read_and_self_remains_authorized(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    frame_payload = source.frame.model_dump(mode="python")
    visibility = _replace_tuple_item(
        source.frame.ally_visibility_mask,
        source.selected_team_local_slot,
        False,
    )
    poisoned_diagonal = tuple(float(500 + index) for index in range(58))
    ally_rows = _replace_tuple_item(
        source.frame.ally_unit_features,
        source.selected_team_local_slot,
        poisoned_diagonal,
    )
    frame_payload["ally_visibility_mask"] = visibility
    frame_payload["ally_unit_features"] = ally_rows
    changed = _replace_current_slice(
        source,
        frame=ActorPovFrameV1.model_validate(frame_payload),
    )
    parts = build_no_shared_obs_authorized_scene_v1(
        changed,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="false-diagonal-session",
    )
    assert parts.scene.agents[0].public_agent_id == source.public_agent_id
    assert parts.scene.agents[0].relation == "self"
    assert (
        sum(row.public_agent_id == source.public_agent_id for row in parts.scene.agents)
        == 1
    )


def test_visibility_true_non_self_corpse_fails_closed(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    assert source.frame.ally_visibility_mask[1]
    corpse_row = list(source.frame.ally_unit_features[1])
    corpse_row[wire.AGENT_FEATURE_ALIVE_V1] = 0.0
    corpse_row[wire.AGENT_FEATURE_CURRENT_HEALTH_V1] = 0.0
    ally_rows = _replace_tuple_item(
        source.frame.ally_unit_features,
        1,
        tuple(corpse_row),
    )
    lifecycle_payload = source.frame.spawn_lifecycle.model_dump(mode="python")
    alive_rows = list(
        cast(tuple[tuple[bool, ...], ...], lifecycle_payload["alive_mask_by_team"])
    )
    alive_rows[0] = _replace_tuple_item(alive_rows[0], 1, False)
    lifecycle_payload["alive_mask_by_team"] = tuple(alive_rows)
    shield_rows = list(
        cast(
            tuple[tuple[int, ...], ...],
            lifecycle_payload["spawn_shield_actual_durations_by_team"],
        )
    )
    shield_rows[0] = _replace_tuple_item(shield_rows[0], 1, 0)
    lifecycle_payload["spawn_shield_actual_durations_by_team"] = tuple(shield_rows)
    lifecycle = ActorPovSpawnLifecycleV1.model_validate(lifecycle_payload)
    frame_payload = source.frame.model_dump(mode="python")
    frame_payload["ally_unit_features"] = ally_rows
    frame_payload["spawn_lifecycle"] = lifecycle.model_dump(mode="python")
    changed = _replace_current_slice(
        source,
        frame=ActorPovFrameV1.model_validate(frame_payload),
    )

    with pytest.raises(ValueError, match="configured active and alive"):
        build_no_shared_obs_authorized_scene_v1(
            changed,
            public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
            authority_session_id="forged-visible-corpse-session",
        )


def test_valid_visible_body_joins_authorized_pad_assignment(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    parts = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="visible-alive-session",
    )
    visible = next(
        row for row in parts.scene.agents if row.public_agent_id == "agent-slot-1"
    )
    pad = next(
        row
        for row in parts.scene.spawn_pads
        if (row.team_id, row.team_local_slot) == (1, 1)
    )
    assert visible.life_state == "alive"
    assert pad.assigned_public_agent_id == visible.public_agent_id
    assert pad.assigned_presentation_key == visible.presentation_key
    assert pad.currently_alive
    assert pad.spawn_shield_remaining == visible.spawn_shield_remaining


def test_recipient_self_corpse_remains_authorized_without_hidden_bodies(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    self_row = list(source.frame.self_features)
    self_row[wire.AGENT_FEATURE_ALIVE_V1] = 0.0
    self_row[wire.AGENT_FEATURE_CURRENT_HEALTH_V1] = 0.0

    lifecycle_payload = source.frame.spawn_lifecycle.model_dump(mode="python")
    alive_rows = list(
        cast(tuple[tuple[bool, ...], ...], lifecycle_payload["alive_mask_by_team"])
    )
    alive_rows[0] = _replace_tuple_item(
        alive_rows[0],
        source.selected_team_local_slot,
        False,
    )
    lifecycle_payload["alive_mask_by_team"] = tuple(alive_rows)
    shield_rows = list(
        cast(
            tuple[tuple[int, ...], ...],
            lifecycle_payload["spawn_shield_actual_durations_by_team"],
        )
    )
    shield_rows[0] = _replace_tuple_item(
        shield_rows[0],
        source.selected_team_local_slot,
        0,
    )
    lifecycle_payload["spawn_shield_actual_durations_by_team"] = tuple(shield_rows)

    hidden_row = tuple(0.0 for _ in range(58))
    frame_payload = source.frame.model_dump(mode="python")
    frame_payload["self_features"] = tuple(self_row)
    frame_payload["ally_visibility_mask"] = (False,) * 5
    frame_payload["enemy_visibility_mask"] = (False,) * 5
    frame_payload["ally_unit_features"] = (hidden_row,) * 5
    frame_payload["enemy_unit_features"] = (hidden_row,) * 5
    frame_payload["spawn_lifecycle"] = ActorPovSpawnLifecycleV1.model_validate(
        lifecycle_payload
    ).model_dump(mode="python")
    changed = _replace_current_slice(
        source,
        frame=ActorPovFrameV1.model_validate(frame_payload),
    )

    parts = build_no_shared_obs_authorized_scene_v1(
        changed,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="self-corpse-session",
    )
    assert len(parts.scene.agents) == 1
    recipient = parts.scene.agents[0]
    assert recipient.public_agent_id == source.public_agent_id
    assert recipient.relation == "self"
    assert recipient.life_state == "corpse"
    self_pad = next(
        row
        for row in parts.scene.spawn_pads
        if (row.team_id, row.team_local_slot)
        == (source.configured_team_id, source.selected_team_local_slot)
    )
    assert self_pad.assigned_public_agent_id == source.public_agent_id
    assert not self_pad.currently_alive
    assert all(
        pad.assigned_public_agent_id is None
        for pad in parts.scene.spawn_pads
        if pad is not self_pad
    )


def test_cross_class_status_uses_dynamic_row_and_public_catalog_semantics(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    hunter_slow = no_shared_trajectory.context.static_mechanics_catalog.status_channels[
        1
    ]
    assert hunter_slow.source_class_id == 3
    self_row = list(source.frame.self_features)
    self_row[wire.AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION_V1] = 1.0
    assert hunter_slow.magnitude is not None
    self_row[wire.AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER_V1] = unpack(
        ">f", pack(">f", hunter_slow.magnitude)
    )[0]
    changed = _replace_frame_rows(source, self_features=tuple(self_row))
    mutated_catalog = _catalog_with_status_duration(
        no_shared_trajectory.context.static_mechanics_catalog,
        channel=1,
        duration=9,
    )

    parts = build_no_shared_obs_authorized_scene_v1(
        changed,
        public_catalog=mutated_catalog,
        authority_session_id="status-session",
    )
    represented_classes = {row.class_id for row in parts.scene.class_mechanics}
    assert 3 not in represented_classes
    status = next(
        row for row in parts.scene.agents[0].statuses if row.status_channel == 1
    )
    assert status.configured_duration_steps == 9
    assert status.remaining_duration == 1
    assert (
        status.magnitude == self_row[wire.AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER_V1]
    )
    assert (
        status.status_id,
        status.family,
        status.source_class_id,
        status.source_action_component,
        status.breaks_on_positive_damage,
    ) == (
        hunter_slow.status_id,
        hunter_slow.family,
        hunter_slow.source_class_id,
        hunter_slow.source_action_component,
        hunter_slow.breaks_on_positive_damage,
    )
    assert status.direct_sources == ()

    poisoned_row = list(self_row)
    poisoned_row[wire.AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER_V1] = 0.25
    poisoned = _replace_frame_rows(source, self_features=tuple(poisoned_row))
    with pytest.raises(ValueError, match="active magnitude does not join"):
        build_no_shared_obs_authorized_scene_v1(
            poisoned,
            public_catalog=mutated_catalog,
            authority_session_id="status-session",
        )


def test_aura_clamp_comes_from_catalog_while_field_capabilities_remain_wire_exact(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    row_multiplier = source.frame.self_features[
        wire.AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1
    ]
    widened_catalog = _catalog_with_aura_values(
        no_shared_trajectory.context.static_mechanics_catalog,
        per_emitter_multiplier=1.15,
        clamp_value=1.4,
    )
    parts = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=widened_catalog,
        authority_session_id="aura-session",
    )
    mage_field = next(
        field
        for field in parts.scene.aura_fields
        if field.source_public_agent_id == source.public_agent_id
    )
    assert mage_field.per_emitter_multiplier == row_multiplier
    assert mage_field.clamp_value == 1.4
    assert mage_field.stacking_rule == "multiply_then_clamp"
    assert tuple(
        (row.aura_id, row.multiplier) for row in parts.scene.agents[0].aura_modifiers
    ) == (
        (
            "mage_damage_amplification",
            source.frame.self_features[
                wire.AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1
            ],
        ),
    )

    mismatched_catalog = _catalog_with_aura_values(
        no_shared_trajectory.context.static_mechanics_catalog,
        per_emitter_multiplier=1.16,
    )
    with pytest.raises(ValueError, match="does not join"):
        build_no_shared_obs_authorized_scene_v1(
            source,
            public_catalog=mismatched_catalog,
            authority_session_id="aura-session",
        )


def test_hidden_emitter_aggregate_has_no_source_and_exact_neutral_is_omitted(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=5)
    neutral = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="hidden-emitter-session",
    )
    assert neutral.scene.aura_fields == ()
    assert all(row.aura_modifiers == () for row in neutral.scene.agents)

    self_row = list(source.frame.self_features)
    self_row[wire.AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1] = unpack(
        ">f",
        pack(
            ">f",
            no_shared_trajectory.context.static_mechanics_catalog.aura_mechanics[
                0
            ].per_emitter_multiplier,
        ),
    )[0]
    changed = _replace_frame_rows(source, self_features=tuple(self_row))
    parts = build_no_shared_obs_authorized_scene_v1(
        changed,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="hidden-emitter-session",
    )
    assert parts.scene.aura_fields == ()
    modifier = parts.scene.agents[0].aura_modifiers[0]
    assert modifier.aura_id == "mage_damage_amplification"
    assert (
        modifier.multiplier
        == self_row[wire.AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1]
    )
    assert set(asdict(modifier)) == {"aura_id", "multiplier"}


def test_recipient_shield_lifecycle_and_action_mask_flow_without_substitution(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    lifecycle_payload = source.frame.spawn_lifecycle.model_dump(mode="python")
    lifecycle_payload["spawn_shield_configured_duration"] = 7
    lifecycle_payload["spawn_shield_speed"] = 2.75
    shield_rows = list(
        cast(
            tuple[tuple[int, ...], ...],
            lifecycle_payload["spawn_shield_actual_durations_by_team"],
        )
    )
    shield_rows[0] = _replace_tuple_item(
        shield_rows[0],
        source.selected_team_local_slot,
        5,
    )
    lifecycle_payload["spawn_shield_actual_durations_by_team"] = tuple(shield_rows)
    countdowns = cast(
        tuple[int, int],
        lifecycle_payload["respawn_wave_countdowns_by_team"],
    )
    lifecycle_payload["respawn_wave_countdowns_by_team"] = (
        (countdowns[0] + 1) % 5,
        (countdowns[1] + 1) % 7,
    )
    lifecycle = ActorPovSpawnLifecycleV1.model_validate(lifecycle_payload)

    action_payload = source.frame.action_mask.model_dump(mode="python")
    move = cast(tuple[bool, ...], action_payload["move"])
    action_payload["move"] = _replace_tuple_item(move, 0, not move[0])
    action_mask = ActorPovActionMaskV1.model_validate(action_payload)
    frame_payload = source.frame.model_dump(mode="python")
    frame_payload["spawn_lifecycle"] = lifecycle.model_dump(mode="python")
    frame_payload["action_mask"] = action_mask.model_dump(mode="python")
    changed = _replace_current_slice(
        source,
        frame=ActorPovFrameV1.model_validate(frame_payload),
    )

    parts = build_no_shared_obs_authorized_scene_v1(
        changed,
        public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
        authority_session_id="recipient-owned-session",
    )
    assert parts.next_decision_action_mask == action_mask
    shield = parts.scene.spawn_shield_mechanics
    assert type(shield) is AuthorizedSpawnShieldMechanicsAvailableV2
    assert shield.availability_kind == "available_v2"
    assert shield.configured_duration_steps == 7
    assert shield.movement_speed == 2.75
    assert parts.scene.agents[0].spawn_shield_remaining == 5
    own_wave = next(row for row in parts.scene.respawn_waves if row.team_id == 1)
    opponent_wave = next(row for row in parts.scene.respawn_waves if row.team_id == 2)
    assert own_wave.countdown_steps == lifecycle.respawn_wave_countdowns_by_team[0]
    assert opponent_wave.countdown_steps == lifecycle.respawn_wave_countdowns_by_team[1]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        (
            wire.AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1,
            1.5,
            "Mage aggregate aura multiplier",
        ),
        (
            wire.AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1,
            0.5,
            "Warrior aggregate aura multiplier",
        ),
    ),
)
def test_poisoned_aggregate_aura_multiplier_fails_catalog_bounds(
    no_shared_trajectory: CapturedEvaluationTrajectory,
    column: int,
    value: float,
    message: str,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    self_row = list(source.frame.self_features)
    self_row[column] = value
    changed = _replace_frame_rows(source, self_features=tuple(self_row))
    with pytest.raises(ValueError, match=message):
        build_no_shared_obs_authorized_scene_v1(
            changed,
            public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
            authority_session_id="poisoned-aura-session",
        )


@pytest.mark.parametrize(
    ("remaining_column", "configured_column", "message"),
    (
        (
            wire.AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1,
            wire.AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION_V1,
            "ultimate cooldown remaining",
        ),
        (
            wire.AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT_V1,
            wire.AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS_V1,
            "out-of-combat countdown",
        ),
    ),
)
def test_dynamic_counter_cannot_exceed_recorded_configured_capability(
    no_shared_trajectory: CapturedEvaluationTrajectory,
    remaining_column: int,
    configured_column: int,
    message: str,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    self_row = list(source.frame.self_features)
    self_row[remaining_column] = self_row[configured_column] + 1.0
    changed = _replace_frame_rows(source, self_features=tuple(self_row))
    with pytest.raises(ValueError, match=message):
        build_no_shared_obs_authorized_scene_v1(
            changed,
            public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
            authority_session_id="counter-session",
        )


@pytest.mark.parametrize("class_id", (0.0, 6.0))
def test_visible_class_outside_v1_domain_raises_value_error_not_index_error(
    no_shared_trajectory: CapturedEvaluationTrajectory,
    class_id: float,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    visible_row = list(source.frame.ally_unit_features[1])
    visible_row[wire.AGENT_FEATURE_CLASS_ID_V1] = class_id
    changed = _replace_frame_rows(
        source,
        ally_rows=_replace_tuple_item(
            source.frame.ally_unit_features,
            1,
            tuple(visible_row),
        ),
    )
    with pytest.raises(ValueError, match="class"):
        build_no_shared_obs_authorized_scene_v1(
            changed,
            public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
            authority_session_id="class-domain-session",
        )


def test_catalog_profile_certification_fails_closed_for_resealed_mutations(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    catalog = no_shared_trajectory.context.static_mechanics_catalog
    assert catalog.canonical_digest_sha256 == (
        AUTHORIZED_CLASS_DOCUMENTATION_CATALOG_FINGERPRINT_V1
    )
    available = authorized_class_documentation_profile_v1(catalog)
    assert type(available) is AuthorizedClassDocumentationProfileAvailableV1

    mutation_labels: set[str] = set()
    for label, changed_catalog in _valid_documentation_catalog_leaf_mutations(catalog):
        mutation_labels.add(label)
        assert changed_catalog.canonical_digest_sha256 != (
            AUTHORIZED_CLASS_DOCUMENTATION_CATALOG_FINGERPRINT_V1
        ), label
        assert type(authorized_class_documentation_profile_v1(changed_catalog)) is (
            AuthorizedClassDocumentationProfileUnavailableV1
        ), label
    assert len(mutation_labels) == 1 + 6 * 16 + 9 * 8 + 2 * 5

    # These remaining documented leaves are exact Literals or fixed ordered
    # axes. They cannot form a valid historical catalog, so revalidation—not
    # the profile selector—must reject even a correctly resealed payload.
    immutable_labels: set[str] = set()
    for (
        label,
        invalid_payload,
    ) in _invalid_immutable_documentation_catalog_leaf_payloads(catalog):
        immutable_labels.add(label)
        invalid_payload["canonical_digest_sha256"] = canonical_digest_sha256(
            invalid_payload,
            exclude={"canonical_digest_sha256"},
        )
        with pytest.raises(ValidationError):
            StaticMechanicsCatalogV1.model_validate(invalid_payload)
    assert len(immutable_labels) == 9

    source = _current_slice(no_shared_trajectory, global_slot=0)
    original = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=catalog,
        authority_session_id="unrepresented-session",
    )
    assert 3 not in {row.class_id for row in original.scene.class_mechanics}
    changed_catalog = _catalog_with_unrepresented_hunter_damage(catalog, 123.25)
    changed = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=changed_catalog,
        authority_session_id="unrepresented-session",
    )
    assert all(
        type(row) is AuthorizedClassMechanicsV2
        and type(row.documentation_profile)
        is AuthorizedClassDocumentationProfileAvailableV1
        for row in original.scene.class_mechanics
    )
    assert all(
        type(row) is AuthorizedClassMechanicsV2
        and type(row.documentation_profile)
        is AuthorizedClassDocumentationProfileUnavailableV1
        for row in changed.scene.class_mechanics
    )
    assert type(original.scene.spawn_shield_mechanics) is (
        AuthorizedSpawnShieldMechanicsAvailableV2
    )


def test_per_slot_profile_overrides_remain_agent_facts_not_class_documentation(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    assert source.frame.ally_visibility_mask[1]
    catalog = no_shared_trajectory.context.static_mechanics_catalog
    mage_catalog = catalog.class_mechanics[1]
    ooc_regen_column = (
        wire.AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP_V1
    )

    first_mage = list(source.frame.self_features)
    first_overrides = {
        wire.AGENT_FEATURE_RADIUS_V1: 0.61,
        wire.AGENT_FEATURE_BASE_MOVEMENT_SPEED_V1: 1.75,
        wire.AGENT_FEATURE_OBSERVATION_RADIUS_V1: 8.25,
        wire.AGENT_FEATURE_BASIC_INTERACTION_RADIUS_V1: 1.25,
        wire.AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS_V1: 4.25,
        wire.AGENT_FEATURE_CURRENT_HEALTH_V1: 101.0,
        wire.AGENT_FEATURE_MAX_HEALTH_V1: 135.0,
        wire.AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS_V1: 11.0,
        ooc_regen_column: 0.07,
    }
    for column, value in first_overrides.items():
        first_mage[column] = value

    second_mage = list(first_mage)
    second_overrides = {
        wire.AGENT_FEATURE_X_V1: source.frame.ally_unit_features[1][
            wire.AGENT_FEATURE_X_V1
        ],
        wire.AGENT_FEATURE_Y_V1: source.frame.ally_unit_features[1][
            wire.AGENT_FEATURE_Y_V1
        ],
        wire.AGENT_FEATURE_RADIUS_V1: 0.72,
        wire.AGENT_FEATURE_BASE_MOVEMENT_SPEED_V1: 1.50,
        wire.AGENT_FEATURE_OBSERVATION_RADIUS_V1: 7.75,
        wire.AGENT_FEATURE_BASIC_INTERACTION_RADIUS_V1: 1.40,
        wire.AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS_V1: 3.75,
        wire.AGENT_FEATURE_CURRENT_HEALTH_V1: 88.0,
        wire.AGENT_FEATURE_MAX_HEALTH_V1: 145.0,
        wire.AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS_V1: 13.0,
        ooc_regen_column: 0.09,
    }
    for column, value in second_overrides.items():
        second_mage[column] = value

    ally_rows = _replace_tuple_item(
        source.frame.ally_unit_features,
        source.selected_team_local_slot,
        tuple(first_mage),
    )
    ally_rows = _replace_tuple_item(
        ally_rows,
        1,
        tuple(second_mage),
    )
    changed = _replace_frame_rows(
        source,
        self_features=tuple(first_mage),
        ally_rows=ally_rows,
    )
    parts = build_no_shared_obs_authorized_scene_v1(
        changed,
        public_catalog=catalog,
        authority_session_id="per-slot-profile-session",
    )

    mage_agents = tuple(row for row in parts.scene.agents if row.class_id == 1)
    assert len(mage_agents) == 2
    assert tuple(
        (
            row.radius,
            row.base_movement_speed,
            row.observation_radius,
            row.basic_interaction_radius,
            row.ultimate_interaction_radius,
            row.maximum_health,
            row.out_of_combat_delay_steps,
            row.out_of_combat_health_regeneration_fraction_per_step,
        )
        for row in mage_agents
    ) == (
        (0.61, 1.75, 8.25, 1.25, 4.25, 135.0, 11, 0.07),
        (0.72, 1.50, 7.75, 1.40, 3.75, 145.0, 13, 0.09),
    )
    mage_mechanics = next(
        row for row in parts.scene.class_mechanics if row.class_id == 1
    )
    assert (
        mage_mechanics.maximum_health,
        mage_mechanics.body_radius,
        mage_mechanics.base_movement_speed,
        mage_mechanics.observation_radius,
        mage_mechanics.basic_interaction_radius,
        mage_mechanics.ultimate_interaction_radius,
        mage_mechanics.out_of_combat_delay_steps,
        mage_mechanics.out_of_combat_health_regeneration_fraction_per_step,
    ) == (
        mage_catalog.maximum_health,
        mage_catalog.body_radius,
        mage_catalog.base_movement_speed,
        mage_catalog.observation_radius,
        mage_catalog.basic_interaction_radius,
        mage_catalog.ultimate_interaction_radius,
        mage_catalog.out_of_combat_delay_steps,
        mage_catalog.out_of_combat_health_regeneration_fraction_per_step,
    )


def test_same_class_fixed_capability_conflict_fails_closed(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    second_mage = list(source.frame.self_features)
    second_mage[wire.AGENT_FEATURE_X_V1] = source.frame.ally_unit_features[1][
        wire.AGENT_FEATURE_X_V1
    ]
    second_mage[wire.AGENT_FEATURE_Y_V1] = source.frame.ally_unit_features[1][
        wire.AGENT_FEATURE_Y_V1
    ]
    second_mage[wire.AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE_V1] += 1.0
    changed = _replace_frame_rows(
        source,
        ally_rows=_replace_tuple_item(
            source.frame.ally_unit_features,
            1,
            tuple(second_mage),
        ),
    )
    with pytest.raises(ValueError, match="basic raw damage does not join"):
        build_no_shared_obs_authorized_scene_v1(
            changed,
            public_catalog=no_shared_trajectory.context.static_mechanics_catalog,
            authority_session_id="same-class-fixed-conflict-session",
        )


def test_forged_catalog_is_revalidated_before_any_scene_decode(
    no_shared_trajectory: CapturedEvaluationTrajectory,
) -> None:
    source = _current_slice(no_shared_trajectory, global_slot=0)
    forged = no_shared_trajectory.context.static_mechanics_catalog.model_copy(
        update={"canonical_digest_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="canonical digest mismatch"):
        build_no_shared_obs_authorized_scene_v1(
            source,
            public_catalog=forged,
            authority_session_id="forged-catalog-session",
        )


def test_decoder_and_no_shared_builder_import_without_jax_numpy_or_core() -> None:
    code = """
import sys
import marl_battlegrounds.rendering.evaluation_wire_features
import marl_battlegrounds.rendering.authorized_pov_scene
forbidden = ('jax', 'jaxlib', 'numpy', 'marl_battlegrounds.core')
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden)
)
assert loaded == [], loaded
print('forbidden', loaded)
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cuda"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "forbidden []"


def test_no_shared_builder_source_has_no_oracle_builder_call() -> None:
    module_path = (
        _REPOSITORY_ROOT / "src/marl_battlegrounds/rendering/authorized_pov_scene.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    call_names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            call_names.append(node.func.attr)
    assert not {name for name in call_names if "oracle" in name.lower()}


def test_adjacent_carrier_selects_only_exact_nonzero_endpoints_with_scene_parity() -> (
    None
):
    trajectory = captured_evaluation_trajectory(transition_count=2)
    view_zero = EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[0],
        transition=trajectory.transitions[0],
        successor_frame=trajectory.frames[1],
    )
    view_one = EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[1],
        transition=trajectory.transitions[1],
        successor_frame=trajectory.frames[2],
    )
    carrier = build_actor_pov_adjacent_transition_slice_v1(view_one, global_slot=0)
    authority = "adjacent-scene-parity"
    catalog = trajectory.context.static_mechanics_catalog

    carrier_start = build_no_shared_obs_authorized_scene_v1(
        carrier,
        public_catalog=catalog,
        authority_session_id=authority,
        frame_index=1,
    )
    current_start = build_no_shared_obs_authorized_scene_v1(
        build_actor_pov_current_slice_v1(
            trajectory.context,
            trajectory.frames[1],
            global_slot=0,
            incoming_transition_view=view_zero,
        ),
        public_catalog=catalog,
        authority_session_id=authority,
    )
    carrier_successor = build_no_shared_obs_authorized_scene_v1(
        carrier,
        public_catalog=catalog,
        authority_session_id=authority,
        frame_index=2,
    )
    current_successor = build_no_shared_obs_authorized_scene_v1(
        build_actor_pov_current_slice_v1(
            trajectory.context,
            trajectory.frames[2],
            global_slot=0,
            incoming_transition_view=view_one,
        ),
        public_catalog=catalog,
        authority_session_id=authority,
    )

    assert carrier_start == current_start
    assert carrier_successor == current_successor
    for invalid_index in (None, 0, 3, True):
        with pytest.raises(ValueError, match="exact endpoint indexes"):
            build_no_shared_obs_authorized_scene_v1(
                carrier,
                public_catalog=catalog,
                authority_session_id=authority,
                frame_index=invalid_index,
            )


def test_adjacent_scene_revalidates_forged_carrier_before_endpoint_decode() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=1)
    view = EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[0],
        transition=trajectory.transitions[0],
        successor_frame=trajectory.frames[1],
    )
    carrier = build_actor_pov_adjacent_transition_slice_v1(view, global_slot=0)
    forged = carrier.model_copy(update={"selected_team_local_slot": 1})

    with pytest.raises(ValidationError, match="team-local slot"):
        build_no_shared_obs_authorized_scene_v1(
            forged,
            public_catalog=trajectory.context.static_mechanics_catalog,
            authority_session_id="forged-adjacent-scene",
            frame_index=0,
        )
