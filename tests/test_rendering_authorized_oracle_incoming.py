"""Focused CP2.3 lossless Oracle incoming-event projection proofs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.test_rendering_evaluation_events_v2 import (
    _all_event_batch,  # pyright: ignore[reportPrivateUsage]
)

from marl_battlegrounds.rendering.authorized_presentation import (
    ReplayIncomingActionRejectedEventV1,
    ReplayIncomingAgentAnchorV1,
    ReplayIncomingAuthorizedAgentIdentityV1,
    ReplayIncomingFeedOnlyAgentIdentityV1,
    ReplayIncomingSummaryV1,
    _project_replay_incoming_summary_v1,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.rendering.scene import (
    ActionRejectedEventV2,
    VisualAgentAnchorV2,
    VisualAgentPhaseTrajectoryV2,
    VisualEventBatchV2,
)

_ALL_EVENT_KINDS_V2 = (
    "action_rejected",
    "ability_activated",
    "source_damage_output",
    "source_healing_output",
    "recipient_health_resolution",
    "combat_countdown_reset",
    "health_regenerated",
    "cooldown_started",
    "cooldown_ready",
    "charge_phase_displacement",
    "ordinary_movement_phase_displacement",
    "agent_died",
    "lethal_damage_contribution",
    "status_aged_to_zero",
    "status_broken_by_damage",
    "status_applied",
    "status_refreshed_or_extended",
    "status_cleared_by_new_death",
    "spawn_shield_expired",
    "respawn_wave_occurred",
    "agent_respawned",
)


def _key_by_slot(
    trajectories: tuple[VisualAgentPhaseTrajectoryV2, ...],
) -> dict[int, str]:
    return {row.global_slot: f"oracle_{row.global_slot:064x}" for row in trajectories}


def _summary() -> tuple[
    VisualEventBatchV2,
    ReplayIncomingSummaryV1,
    dict[int, str],
]:
    batch, _ = _all_event_batch()
    keys = _key_by_slot(batch.agent_phase_trajectories)
    return (
        batch,
        _project_replay_incoming_summary_v1(
            batch,
            key_by_internal_slot=keys,
        ),
        keys,
    )


def _assert_anchor(
    neutral: ReplayIncomingAgentAnchorV1,
    raw: VisualAgentAnchorV2,
    keys: dict[int, str],
) -> None:
    assert neutral.phase == raw.phase
    assert neutral.presentation_key == keys[raw.global_slot]
    assert neutral.public_agent_id == raw.public_agent_id
    assert neutral.position == raw.position
    assert set(asdict(neutral)) == {
        "phase",
        "presentation_key",
        "public_agent_id",
        "position",
    }


def _assert_optional_anchor(
    neutral: ReplayIncomingAgentAnchorV1 | None,
    raw: VisualAgentAnchorV2 | None,
    keys: dict[int, str],
) -> None:
    assert (neutral is None) == (raw is None)
    if neutral is not None and raw is not None:
        _assert_anchor(neutral, raw, keys)


def _direct_fields(kind: str) -> tuple[str, ...]:
    return {
        "action_rejected": (
            "actor_configured_active",
            "rejection_component",
        ),
        "ability_activated": ("ability_component",),
        "source_damage_output": (
            "raw_damage_output",
            "source_modified_damage_output",
            "recipient_damage_modifier",
        ),
        "source_healing_output": (
            "raw_healing_output",
            "source_modified_healing_output",
            "recipient_healing_modifier",
        ),
        "recipient_health_resolution": (
            "transition_start_health",
            "total_effective_damage",
            "total_effective_healing",
            "health_after_combat_resolution",
            "realized_net_health_change",
        ),
        "combat_countdown_reset": (),
        "health_regenerated": ("actual_health_regenerated",),
        "cooldown_started": (),
        "cooldown_ready": (),
        "charge_phase_displacement": ("realized_displacement",),
        "ordinary_movement_phase_displacement": ("realized_displacement",),
        "agent_died": (),
        "lethal_damage_contribution": ("attributed_death_damage",),
        "status_aged_to_zero": ("status_channel", "status_id"),
        "status_broken_by_damage": ("status_channel", "status_id"),
        "status_applied": ("status_channel", "status_id"),
        "status_refreshed_or_extended": ("status_channel", "status_id"),
        "status_cleared_by_new_death": ("status_channel", "status_id"),
        "spawn_shield_expired": (),
        "respawn_wave_occurred": (),
        "agent_respawned": ("team_id", "realized_successor_position"),
    }[kind]


def test_all_21_events_project_one_to_one_without_internal_slot_axes() -> None:
    batch, _ = _all_event_batch()
    raw_before = json.dumps(asdict(batch), sort_keys=True, separators=(",", ":"))
    keys = _key_by_slot(batch.agent_phase_trajectories)
    summary = _project_replay_incoming_summary_v1(
        batch,
        key_by_internal_slot=keys,
    )

    assert tuple(event.event_type for event in batch.events) == _ALL_EVENT_KINDS_V2
    assert summary.ordered_event_kinds == _ALL_EVENT_KINDS_V2
    assert summary.event_count == 21
    assert len({type(event) for event in summary.events}) == 21
    assert summary.ordered_event_ids == tuple(event.event_id for event in batch.events)
    assert summary.ordered_event_kinds == tuple(
        event.event_type for event in batch.events
    )
    assert tuple(event.ordinal for event in summary.events) == tuple(range(21))
    assert tuple(event.phase_rank for event in summary.events) == tuple(
        event.phase_rank for event in batch.events
    )

    assert len(summary.agent_phase_trajectories) == len(batch.agent_phase_trajectories)
    for raw, neutral in zip(
        batch.agent_phase_trajectories,
        summary.agent_phase_trajectories,
        strict=True,
    ):
        assert neutral.agent_presentation_key == keys[raw.global_slot]
        assert neutral.agent_public_agent_id == raw.public_agent_id
        _assert_anchor(neutral.transition_start, raw.transition_start, keys)
        _assert_anchor(neutral.post_charge, raw.post_charge, keys)
        _assert_anchor(neutral.successor, raw.successor, keys)

    trajectory_by_slot = {
        row.global_slot: row for row in batch.agent_phase_trajectories
    }
    for raw_row, neutral_row in zip(batch.events, summary.events, strict=True):
        raw = cast(Any, raw_row)
        neutral = cast(Any, neutral_row)
        kind = raw.event_type
        assert neutral.event_id == raw.event_id
        assert neutral.ordinal == raw.ordinal
        assert neutral.phase_rank == raw.phase_rank
        assert neutral.event_kind == kind
        for field_name in _direct_fields(kind):
            assert getattr(neutral, field_name) == getattr(raw, field_name)

        handled = {
            "event_id",
            "transition_id",
            "ordinal",
            "event_type",
            "phase_rank",
            *_direct_fields(kind),
        }
        if kind == "action_rejected":
            handled.update(
                {
                    "actor_global_slot",
                    "actor_public_agent_id",
                    "submitted_move_action",
                    "submitted_select_target_action",
                    "submitted_use_ultimate_action",
                    "actor_anchor",
                }
            )
            identity = neutral.actor_identity
            if raw.actor_configured_active:
                assert type(identity) is ReplayIncomingAuthorizedAgentIdentityV1
                assert identity.presentation_key == keys[raw.actor_global_slot]
            else:
                assert type(identity) is ReplayIncomingFeedOnlyAgentIdentityV1
            assert identity.public_agent_id == raw.actor_public_agent_id
            assert (
                neutral.submitted_action.move_action,
                neutral.submitted_action.target_action,
                neutral.submitted_action.use_ultimate_action,
            ) == (
                raw.submitted_move_action,
                raw.submitted_select_target_action,
                raw.submitted_use_ultimate_action,
            )
            _assert_optional_anchor(neutral.actor_anchor, raw.actor_anchor, keys)
        elif kind in {
            "ability_activated",
            "source_damage_output",
            "source_healing_output",
        }:
            handled.update(
                {
                    "source_global_slot",
                    "recipient_global_slot",
                    "source_anchor",
                    "recipient_anchor",
                }
            )
            _assert_anchor(neutral.source_anchor, raw.source_anchor, keys)
            _assert_optional_anchor(
                neutral.recipient_anchor,
                raw.recipient_anchor,
                keys,
            )
            if kind == "source_damage_output":
                handled.update(
                    {
                        "mage_damage_aura_covering_emitter_global_slots",
                        "warrior_mitigation_aura_covering_emitter_global_slots",
                    }
                )
                for slots, emitters in (
                    (
                        raw.mage_damage_aura_covering_emitter_global_slots,
                        neutral.mage_damage_aura_covering_emitters,
                    ),
                    (
                        raw.warrior_mitigation_aura_covering_emitter_global_slots,
                        neutral.warrior_mitigation_aura_covering_emitters,
                    ),
                ):
                    assert len(slots) == len(emitters)
                    for slot, emitter in zip(slots, emitters, strict=True):
                        _assert_anchor(
                            emitter,
                            trajectory_by_slot[slot].transition_start,
                            keys,
                        )
        elif kind == "recipient_health_resolution":
            handled.update({"recipient_global_slot", "recipient_anchor"})
            _assert_anchor(neutral.recipient_anchor, raw.recipient_anchor, keys)
        elif kind in {
            "combat_countdown_reset",
            "health_regenerated",
            "cooldown_started",
            "cooldown_ready",
            "spawn_shield_expired",
        }:
            handled.update({"agent_global_slot", "agent_anchor"})
            _assert_anchor(neutral.agent_anchor, raw.agent_anchor, keys)
        elif kind in {
            "charge_phase_displacement",
            "ordinary_movement_phase_displacement",
        }:
            handled.update({"agent_global_slot", "start_anchor", "end_anchor"})
            _assert_anchor(neutral.start_anchor, raw.start_anchor, keys)
            _assert_anchor(neutral.end_anchor, raw.end_anchor, keys)
        elif kind == "agent_died":
            handled.update({"recipient_global_slot", "recipient_anchor"})
            _assert_anchor(neutral.recipient_anchor, raw.recipient_anchor, keys)
        elif kind == "lethal_damage_contribution":
            handled.update(
                {
                    "source_global_slot",
                    "recipient_global_slot",
                    "source_anchor",
                    "recipient_anchor",
                }
            )
            _assert_anchor(neutral.source_anchor, raw.source_anchor, keys)
            _assert_anchor(neutral.recipient_anchor, raw.recipient_anchor, keys)
        elif kind.startswith("status_"):
            handled.update({"recipient_global_slot", "recipient_anchor"})
            _assert_anchor(neutral.recipient_anchor, raw.recipient_anchor, keys)
            if kind == "status_applied":
                handled.update({"source_global_slot", "source_anchor"})
                _assert_anchor(neutral.source_anchor, raw.source_anchor, keys)
        elif kind == "respawn_wave_occurred":
            handled.update({"team_index", "team_id", "team_anchor"})
            assert neutral.team_anchor.phase == raw.team_anchor.phase
            assert neutral.team_anchor.team_index == raw.team_index
            assert neutral.team_anchor.team_id == raw.team_id
        elif kind == "agent_respawned":
            handled.update({"agent_global_slot", "agent_anchor"})
            _assert_anchor(neutral.agent_anchor, raw.agent_anchor, keys)
        else:  # pragma: no cover - the exhaustive kind assertion guards this.
            raise AssertionError(f"unhandled incoming kind: {kind}")
        assert set(asdict(raw)) == handled, kind

    assert json.dumps(asdict(batch), sort_keys=True, separators=(",", ":")) == (
        raw_before
    )


def test_active_and_inactive_rejections_have_distinct_identity_variants() -> None:
    batch, _ = _all_event_batch()
    original = cast(ActionRejectedEventV2, batch.events[0])
    trajectory = batch.agent_phase_trajectories[0]
    active = ActionRejectedEventV2(
        event_id=original.event_id,
        transition_id=original.transition_id,
        ordinal=original.ordinal,
        actor_global_slot=trajectory.global_slot,
        actor_public_agent_id=trajectory.public_agent_id,
        actor_configured_active=True,
        rejection_component=original.rejection_component,
        submitted_move_action=original.submitted_move_action,
        submitted_select_target_action=original.submitted_select_target_action,
        submitted_use_ultimate_action=original.submitted_use_ultimate_action,
        actor_anchor=trajectory.transition_start,
    )
    changed = replace(batch, events=(active, *batch.events[1:]))
    keys = _key_by_slot(changed.agent_phase_trajectories)

    projected = _project_replay_incoming_summary_v1(
        changed,
        key_by_internal_slot=keys,
    )
    active_projection = projected.events[0]

    assert type(active_projection) is ReplayIncomingActionRejectedEventV1
    assert type(active_projection.actor_identity) is (
        ReplayIncomingAuthorizedAgentIdentityV1
    )
    assert active_projection.actor_identity.presentation_key == keys[0]
    assert active_projection.actor_anchor is not None


def _mapping_keys(value: object) -> tuple[str, ...]:
    found: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, nested in cast(Mapping[object, object], item).items():
                found.append(str(key))
                visit(nested)
        elif isinstance(item, list):
            for nested in cast(list[object], item):
                visit(nested)

    visit(value)
    return tuple(found)


def test_neutral_summary_json_is_strict_and_contains_no_internal_slot_axis() -> None:
    _, summary, _ = _summary()
    adapter = TypeAdapter(ReplayIncomingSummaryV1)
    payload = json.loads(adapter.dump_json(summary))

    keys = _mapping_keys(payload)
    assert not {
        key
        for key in keys
        if "global_slot" in key
        or key
        in {
            "public_agent_id_by_global_slot",
            "configured_active_by_global_slot",
        }
    }
    assert "team_index" in keys

    extra = json.loads(adapter.dump_json(summary))
    extra["events"][2]["poison"] = True
    with pytest.raises(
        ValidationError,
        match=r"extra_forbidden|unexpected_keyword_argument",
    ):
        adapter.validate_json(json.dumps(extra))

    missing = json.loads(adapter.dump_json(summary))
    del missing["events"][2]["source_anchor"]
    with pytest.raises(ValidationError, match="Field required"):
        adapter.validate_json(json.dumps(missing))

    coerced = json.loads(adapter.dump_json(summary))
    coerced["events"][2]["phase_rank"] = "30"
    with pytest.raises(ValidationError, match="int_type"):
        adapter.validate_json(json.dumps(coerced))

    wrong_discriminator = json.loads(adapter.dump_json(summary))
    wrong_discriminator["events"][2]["event_kind"] = "unknown"
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        adapter.validate_json(json.dumps(wrong_discriminator))

    schema = adapter.json_schema()
    event_union = cast(dict[str, object], schema["$defs"])["ReplayIncomingEventV1"]
    discriminator = cast(dict[str, object], event_union)["discriminator"]
    assert cast(dict[str, object], discriminator)["propertyName"] == "event_kind"
    assert (
        len(
            cast(
                dict[str, object],
                cast(dict[str, object], discriminator)["mapping"],
            )
        )
        == 21
    )
    identity_union = cast(dict[str, object], schema["$defs"])[
        "ReplayIncomingAgentIdentityV1"
    ]
    identity_discriminator = cast(dict[str, object], identity_union)["discriminator"]
    assert cast(dict[str, object], identity_discriminator)["propertyName"] == (
        "identity_kind"
    )


def test_strict_summary_fences_inventory_and_every_event_anchor_to_trajectory() -> None:
    _, summary, _ = _summary()
    adapter = TypeAdapter(ReplayIncomingSummaryV1)

    inventory = json.loads(adapter.dump_json(summary))
    inventory["ordered_event_kinds"][2] = "source_healing_output"
    with pytest.raises(ValidationError, match="identity/kind inventory"):
        adapter.validate_json(json.dumps(inventory))

    anchor = json.loads(adapter.dump_json(summary))
    anchor["events"][2]["source_anchor"]["position"][0] += 0.5
    with pytest.raises(ValidationError, match="ordered trajectories"):
        adapter.validate_json(json.dumps(anchor))

    emitter = json.loads(adapter.dump_json(summary))
    emitter["events"][2]["mage_damage_aura_covering_emitters"][0]["phase"] = "successor"
    with pytest.raises(ValidationError, match="transition_start"):
        adapter.validate_json(json.dumps(emitter))

    trajectory = json.loads(adapter.dump_json(summary))
    trajectory["agent_phase_trajectories"][0]["successor"]["public_agent_id"] = (
        "forged-public-id"
    )
    with pytest.raises(ValidationError, match="retain one identity"):
        adapter.validate_json(json.dumps(trajectory))


def test_standalone_summary_rejects_mismatched_epoch_ids_and_displacements() -> None:
    _, summary, _ = _summary()
    adapter = TypeAdapter(ReplayIncomingSummaryV1)

    transition = json.loads(adapter.dump_json(summary))
    transition["incoming_transition_id"] = "unrelated:transition:0"
    with pytest.raises(ValidationError, match="frame IDs must join"):
        adapter.validate_json(json.dumps(transition))

    start_frame = json.loads(adapter.dump_json(summary))
    start_frame["incoming_start_frame_id"] = "episode-001:frame:99"
    with pytest.raises(ValidationError, match="frame IDs must join"):
        adapter.validate_json(json.dumps(start_frame))

    for event_kind, message in (
        ("charge_phase_displacement", "Charge end anchor"),
        ("ordinary_movement_phase_displacement", "movement end anchor"),
    ):
        displacement = json.loads(adapter.dump_json(summary))
        event = next(
            row for row in displacement["events"] if row["event_kind"] == event_kind
        )
        event["end_anchor"]["position"][0] += 1.0
        with pytest.raises(ValidationError, match=message):
            adapter.validate_json(json.dumps(displacement))


def test_aura_emitter_tuples_preserve_trajectory_order_and_uniqueness() -> None:
    _, summary, _ = _summary()
    adapter = TypeAdapter(ReplayIncomingSummaryV1)
    ordered = json.loads(adapter.dump_json(summary))
    damage = next(
        row for row in ordered["events"] if row["event_kind"] == "source_damage_output"
    )
    start_anchors = [
        row["transition_start"] for row in ordered["agent_phase_trajectories"][:2]
    ]
    damage["mage_damage_aura_covering_emitters"] = start_anchors
    adapter.validate_json(json.dumps(ordered))

    reordered = json.loads(json.dumps(ordered))
    reordered_damage = next(
        row
        for row in reordered["events"]
        if row["event_kind"] == "source_damage_output"
    )
    reordered_damage["mage_damage_aura_covering_emitters"].reverse()
    with pytest.raises(ValidationError, match="preserve trajectory order"):
        adapter.validate_json(json.dumps(reordered))

    duplicate = json.loads(json.dumps(ordered))
    duplicate_damage = next(
        row
        for row in duplicate["events"]
        if row["event_kind"] == "source_damage_output"
    )
    duplicate_damage["mage_damage_aura_covering_emitters"] = [
        start_anchors[0],
        start_anchors[0],
    ]
    with pytest.raises(ValidationError, match="unique authorized emitters"):
        adapter.validate_json(json.dumps(duplicate))


def test_respawn_wave_team_index_and_id_are_both_strictly_preserved() -> None:
    _, summary, _ = _summary()
    adapter = TypeAdapter(ReplayIncomingSummaryV1)
    wave_index = summary.ordered_event_kinds.index("respawn_wave_occurred")
    wave = cast(Any, summary.events[wave_index])
    assert wave.team_anchor.team_id == wave.team_anchor.team_index + 1

    payload = json.loads(adapter.dump_json(summary))
    payload["events"][wave_index]["team_anchor"]["team_index"] = 1
    payload["events"][wave_index]["team_anchor"]["team_id"] = 1
    with pytest.raises(ValidationError, match="team_id must equal"):
        adapter.validate_json(json.dumps(payload))
