"""Structural tests for canonical synthetic browser renderer fixtures."""

import inspect
import json
import re
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from math import dist

import pytest
import scripts.dev.visual_debugger.renderer_fixtures as fixture_module
from scripts.dev.visual_debugger.protocol import (
    ActorPovLiveDebuggerFrameV2,
    ResearcherLiveDebuggerFrameV2,
)
from scripts.dev.visual_debugger.renderer_fixtures import (
    CATALOG_STATUS_ORDER,
    RENDERER_FIXTURES,
    RendererFixtureV2,
    fixture_pov_target_reference_v1,
    get_renderer_fixture,
    list_renderer_fixtures,
    renderer_fixture_to_jsonable,
)

from marl_battlegrounds.core.types import (
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovAnalyzerProjectionV1,
    ActorPovBattlefieldSceneV1,
)
from marl_battlegrounds.rendering.scene import (
    AbilityActivatedEventV2,
    BattlefieldSceneV2,
    ChargePhaseDisplacementEventV2,
    RecipientHealthResolutionEventV2,
    StatusAgedToZeroEventV2,
    StatusAppliedEventV2,
    StatusBrokenByDamageEventV2,
    StatusClearedByNewDeathEventV2,
    StatusRefreshedOrExtendedEventV2,
)

CANONICAL_EVENT_TYPES = (
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


def test_renderer_fixture_registry_is_exact_synthetic_and_v2() -> None:
    assert tuple(RENDERER_FIXTURES) == (
        "visual_vocabulary",
        "durable_controls",
        "crowded_teamfight",
        "route_collision",
        "mixed_net_zero",
        "viewport_matrix",
        "canonical_event_vocabulary",
        "pov_redaction",
    )
    assert tuple(fixture.name for fixture in list_renderer_fixtures()) == tuple(
        RENDERER_FIXTURES
    )
    assert all(
        fixture.description.startswith("SYNTHETIC:")
        for fixture in list_renderer_fixtures()
    )
    with pytest.raises(ValueError, match="unknown renderer fixture"):
        get_renderer_fixture("not-a-fixture")
    with pytest.raises(FrozenInstanceError):
        get_renderer_fixture("route_collision").description = "changed"  # type: ignore[misc]

    source = inspect.getsource(fixture_module)
    for forbidden in (
        "VisualEventBatchV1",
        "AcceptedActivationEventV1",
        "NetHealthEventV1",
        "StatusLifecycleEventV1",
        "DebuggerScenario",
        "create_session",
        "submit_joint_action",
        "EnvState",
    ):
        assert forbidden not in source
    assert re.search(r"(?<!ActorPov)BattlefieldSceneV1", source) is None


def test_every_fixture_carries_an_exact_validated_live_envelope() -> None:
    for fixture in list_renderer_fixtures():
        if fixture.audience == "researcher":
            assert type(fixture.scene) is BattlefieldSceneV2
            assert type(fixture.live_frame) is ResearcherLiveDebuggerFrameV2
            assert fixture.live_frame.projection.scene == fixture.scene
            assert fixture.live_frame.projection.incoming_events == fixture.event_batch
            assert fixture.scene.selection is not None
            assert tuple(
                row.observer_global_slot for row in fixture.scene.observer_visibility
            ) == (fixture.scene.selection.controlled_global_slot,) * len(
                fixture.scene.agents
            )
            assert tuple(
                row.candidate_global_slot for row in fixture.scene.observer_visibility
            ) == tuple(agent.global_slot for agent in fixture.scene.agents)
            payload = fixture.live_frame.model_dump(mode="json")
            assert payload["schema_version"] == 2
            assert payload["frame_kind"] == "researcher_live_debugger"
        else:
            assert type(fixture.scene) is ActorPovBattlefieldSceneV1
            assert type(fixture.live_frame) is ActorPovLiveDebuggerFrameV2
            assert fixture.live_frame.projection == fixture.pov_projection
            payload = fixture.live_frame.model_dump(mode="json")
            assert payload["schema_version"] == 2
            assert payload["frame_kind"] == "actor_pov_live_debugger"


def test_every_renderer_fixture_is_recursively_json_serializable() -> None:
    all_event_ids: list[str] = []
    for fixture in list_renderer_fixtures():
        payload = renderer_fixture_to_jsonable(fixture)
        assert json.loads(json.dumps(payload)) == payload
        if fixture.event_batch is not None:
            all_event_ids.extend(event.event_id for event in fixture.event_batch.events)
        if fixture.privileged_source_event_batch is not None:
            all_event_ids.extend(
                event.event_id for event in fixture.privileged_source_event_batch.events
            )
    assert len(all_event_ids) == len(set(all_event_ids))


def test_durable_fixture_preserves_canonical_status_channels_and_sources() -> None:
    fixture = get_renderer_fixture("durable_controls")
    assert type(fixture.scene) is BattlefieldSceneV2
    assert fixture.event_batch is None
    assert tuple(agent.global_slot for agent in fixture.scene.agents) == (0, 5)
    statuses = tuple(
        status for agent in fixture.scene.agents for status in agent.statuses
    )
    assert tuple(status.status_id for status in statuses) == (
        "warrior_charge_stun",
        "hunter_trap_stun",
        "rogue_poison_stun",
        "warrior_charge_slow",
        "hunter_basic_slow",
        "rogue_poison_slow",
    )
    assert tuple(status.status_channel for status in statuses) == (3, 4, 5, 0, 1, 2)
    assert tuple(status.source_class_id for status in statuses) == (
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
    )


def test_status_roots_reject_scientific_identity_and_presentation_order_drift() -> None:
    durable = get_renderer_fixture("durable_controls")
    assert type(durable.scene) is BattlefieldSceneV2
    agent = durable.scene.agents[0]
    assert len(agent.statuses) >= 2
    with pytest.raises(ValueError, match="canonical presentation order"):
        replace(agent, statuses=tuple(reversed(agent.statuses)))
    with pytest.raises(ValueError, match="retain V1 identity"):
        replace(
            agent.statuses[0],
            status_channel=(agent.statuses[0].status_channel + 1) % 9,
        )

    canonical = get_renderer_fixture("canonical_event_vocabulary")
    assert canonical.event_batch is not None
    status_event = next(
        event
        for event in canonical.event_batch.events
        if type(event) is StatusAgedToZeroEventV2
    )
    with pytest.raises(ValueError, match="retain V1 identity"):
        replace(status_event, status_channel=0)

    crowded = get_renderer_fixture("crowded_teamfight")
    assert type(crowded.live_frame) is ResearcherLiveDebuggerFrameV2
    evidence = crowded.live_frame.projection.status_source_evidence.active_statuses
    assert evidence
    mismatched_id = (
        "hunter_trap_stun"
        if evidence[0].status_id != "hunter_trap_stun"
        else "warrior_charge_slow"
    )
    with pytest.raises(ValueError, match="retain V1 identity"):
        replace(evidence[0], status_id=mismatched_id)


def test_visual_vocabulary_covers_current_class_and_event_grammar() -> None:
    fixture = get_renderer_fixture("visual_vocabulary")
    assert type(fixture.scene) is BattlefieldSceneV2
    batch = fixture.event_batch
    assert batch is not None
    assert tuple(agent.class_id for agent in fixture.scene.agents[:5]) == (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
    )
    assert tuple(
        agent.ultimate_cooldown_remaining for agent in fixture.scene.agents[:5]
    ) == (1, 2, 3, 4, 5)
    assert tuple(
        (field.source_global_slot, field.aura_id) for field in fixture.scene.aura_fields
    ) == (
        (0, "mage_damage_amplification"),
        (1, "warrior_damage_mitigation"),
    )
    activations = tuple(
        event for event in batch.events if type(event) is AbilityActivatedEventV2
    )
    assert tuple(event.ability_component for event in activations) == (
        ("basic",) * 5 + ("ultimate",) * 5
    )
    assert all(not hasattr(event, "amount") for event in activations)
    outcomes = tuple(
        event
        for event in batch.events
        if type(event) is RecipientHealthResolutionEventV2
    )
    assert tuple(event.realized_net_health_change > 0.0 for event in outcomes) == (
        False,
        True,
    )


def test_crowded_teamfight_retains_all_visual_pressure_in_v2() -> None:
    fixture = get_renderer_fixture("crowded_teamfight")
    assert type(fixture.scene) is BattlefieldSceneV2
    batch = fixture.event_batch
    assert batch is not None
    scene = fixture.scene
    assert (len(scene.agents), len(scene.map.obstacles), len(scene.aura_fields)) == (
        10,
        2,
        4,
    )
    assert len(scene.ranges) == 3
    assert scene.selection is not None
    assert scene.next_decision_selected_legality is not None
    assert all(
        tuple(status.status_id for status in agent.statuses) == CATALOG_STATUS_ORDER
        for agent in scene.agents
    )
    assert sum(len(agent.statuses) for agent in scene.agents) == 90
    assert sum(len(agent.aura_modifiers) for agent in scene.agents) == 20
    assert Counter(type(event) for event in batch.events) == {
        AbilityActivatedEventV2: 10,
        RecipientHealthResolutionEventV2: 8,
        ChargePhaseDisplacementEventV2: 2,
        StatusAppliedEventV2: 12,
    }
    assert len(batch.events) == 32


def test_route_and_mixed_fixtures_preserve_exact_anchor_and_health_truth() -> None:
    route = get_renderer_fixture("route_collision")
    assert route.event_batch is not None
    activations = tuple(
        event
        for event in route.event_batch.events
        if type(event) is AbilityActivatedEventV2
    )
    endpoint_pairs = Counter(
        (event.source_global_slot, event.recipient_global_slot) for event in activations
    )
    assert endpoint_pairs[(0, 5)] == endpoint_pairs[(5, 0)] == 1
    assert endpoint_pairs[(1, 6)] == 2
    local = next(
        event
        for event in activations
        if (event.source_global_slot, event.recipient_global_slot) == (4, 9)
    )
    assert local.recipient_anchor is not None
    assert dist(local.source_anchor.position, local.recipient_anchor.position) < 0.1

    mixed = get_renderer_fixture("mixed_net_zero")
    assert mixed.event_batch is not None
    outcome = next(
        event
        for event in mixed.event_batch.events
        if type(event) is RecipientHealthResolutionEventV2
    )
    assert outcome.recipient_global_slot == 5
    assert outcome.realized_net_health_change == 0.0
    assert outcome.health_after_combat_resolution == outcome.transition_start_health


def test_canonical_event_fixture_covers_exact_union_order_and_ids() -> None:
    fixture = get_renderer_fixture("canonical_event_vocabulary")
    assert type(fixture.scene) is BattlefieldSceneV2
    batch = fixture.event_batch
    assert batch is not None
    assert tuple(event.event_type for event in batch.events) == CANONICAL_EVENT_TYPES
    assert tuple(event.ordinal for event in batch.events) == tuple(range(21))
    assert tuple(event.event_id for event in batch.events) == tuple(
        f"{batch.transition_id}:event:{ordinal:04d}" for ordinal in range(21)
    )
    assert fixture.scene.incoming_event_ids == tuple(
        event.event_id for event in batch.events
    )
    aged = batch.events[13]
    broken = batch.events[14]
    applied = batch.events[15]
    refreshed = batch.events[16]
    cleared = batch.events[17]
    assert type(aged) is StatusAgedToZeroEventV2
    assert type(broken) is StatusBrokenByDamageEventV2
    assert type(applied) is StatusAppliedEventV2
    assert type(refreshed) is StatusRefreshedOrExtendedEventV2
    assert type(cleared) is StatusClearedByNewDeathEventV2
    assert (
        aged.status_channel,
        broken.status_channel,
        applied.status_channel,
        refreshed.status_channel,
        cleared.status_channel,
    ) == (3, 4, 1, 2, 6)
    assert (
        aged.status_id,
        broken.status_id,
        applied.status_id,
        refreshed.status_id,
        cleared.status_id,
    ) == (
        "warrior_charge_stun",
        "hunter_trap_stun",
        "hunter_basic_slow",
        "rogue_poison_slow",
        "rogue_poison_anti_heal",
    )


def test_pov_fixture_is_independent_recipient_safe_and_axis_bound() -> None:
    fixture = get_renderer_fixture("pov_redaction")
    assert type(fixture.scene) is ActorPovBattlefieldSceneV1
    assert type(fixture.pov_projection) is ActorPovAnalyzerProjectionV1
    assert type(fixture.live_frame) is ActorPovLiveDebuggerFrameV2
    assert fixture.event_batch is None
    assert fixture.privileged_source_scene is not None
    assert fixture.privileged_source_event_batch is not None
    assert fixture.scene.self_actor.public_agent_id == "0"
    assert tuple(body.public_agent_id for body in fixture.scene.visible_bodies) == (
        "1",
    )
    assert all(
        not hasattr(body, "global_slot") for body in fixture.scene.visible_bodies
    )
    assert tuple(cue.ordinal for cue in fixture.pov_projection.incoming_cues) == tuple(
        range(6)
    )
    assert tuple(cue.cue_type for cue in fixture.pov_projection.incoming_cues) == (
        "own_action_outcome",
        "own_position_changed",
        "own_health_changed",
        "own_status_changed",
        "own_status_changed",
        "own_status_changed",
    )
    safe_json = json.dumps(renderer_fixture_to_jsonable(fixture.live_frame))
    assert fixture.privileged_source_event_batch.events[1].event_id not in safe_json
    assert fixture.pov_target_public_agent_ids is not None
    assert (
        tuple(
            row.target.public_agent_id
            for row in fixture.live_frame.hud.candidate_legalities
        )
        == fixture.pov_target_public_agent_ids
    )


def test_pov_target_references_use_explicit_team_b_axis_not_slot_arithmetic() -> None:
    team_b_axis: tuple[str | None, ...] = (
        None,
        "5",
        "6",
        "7",
        "8",
        "9",
        "0",
        "1",
        "2",
        "3",
        "4",
    )
    assert fixture_pov_target_reference_v1(1, team_b_axis).public_agent_id == "5"
    assert fixture_pov_target_reference_v1(6, team_b_axis).public_agent_id == "0"
    with pytest.raises(ValueError, match="outside"):
        fixture_pov_target_reference_v1(11, team_b_axis)


def test_fixture_envelope_rejects_unlabelled_or_cross_audience_payloads() -> None:
    researcher = get_renderer_fixture("mixed_net_zero")
    with pytest.raises(ValueError, match="SYNTHETIC"):
        replace(researcher, description="ordinary fixture")

    pov = get_renderer_fixture("pov_redaction")
    with pytest.raises(ValueError, match="target axis"):
        replace(pov, pov_target_public_agent_ids=None)
    with pytest.raises(ValueError, match="validated V2 frame"):
        RendererFixtureV2(
            name="mixed_net_zero",
            description="SYNTHETIC: mismatched frame.",
            audience="researcher",
            scene=researcher.scene,
            live_frame=pov.live_frame,
            event_batch=researcher.event_batch,
        )
