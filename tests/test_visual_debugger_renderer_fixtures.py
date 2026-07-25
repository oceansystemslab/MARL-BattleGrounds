"""Structural tests for explicitly synthetic browser renderer fixtures."""

import inspect
import json
from collections import Counter
from dataclasses import FrozenInstanceError
from math import dist

import pytest
import scripts.dev.visual_debugger.renderer_fixtures as fixture_module
from scripts.dev.visual_debugger.renderer_fixtures import (
    RENDERER_FIXTURES,
    RendererFixtureV1,
    get_renderer_fixture,
    list_renderer_fixtures,
)

from marl_battlegrounds.rendering.scene import (
    AcceptedActivationEventV1,
    ChargeDisplacementEventV1,
    NetHealthEventV1,
    StatusLifecycleEventV1,
    to_jsonable,
)
from marl_battlegrounds.rendering.vocabulary import CANONICAL_STATUS_ORDER


def test_renderer_fixture_registry_is_exact_synthetic_and_separate() -> None:
    assert tuple(RENDERER_FIXTURES) == (
        "crowded_teamfight",
        "route_collision",
        "mixed_net_zero",
        "viewport_matrix",
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
        "DebuggerScenario",
        "create_session",
        "submit_joint_action",
        "EnvState",
        "ActionMask",
    ):
        assert forbidden not in source


def test_every_renderer_fixture_is_recursively_json_serializable() -> None:
    all_event_ids: list[str] = []
    for fixture in list_renderer_fixtures():
        payload = to_jsonable(fixture)
        assert json.loads(json.dumps(payload)) == payload
        if fixture.event_batch is not None:
            all_event_ids.extend(event.event_id for event in fixture.event_batch.events)
        if fixture.privileged_source_event_batch is not None:
            all_event_ids.extend(
                event.event_id for event in fixture.privileged_source_event_batch.events
            )
    assert len(all_event_ids) == len(set(all_event_ids))


def test_crowded_teamfight_contains_every_durable_overlay_pressure() -> None:
    fixture = get_renderer_fixture("crowded_teamfight")
    scene = fixture.scene
    batch = fixture.event_batch
    assert batch is not None
    assert len(scene.agents) == 10
    assert len(scene.map.obstacles) == 2
    assert len(scene.aura_fields) == 4
    assert len(scene.ranges) == 3
    assert scene.selection is not None
    assert scene.selected_legality is not None
    assert scene.pending_route is not None
    assert len(scene.observer_visibility) == 10
    assert all(
        tuple(status.token_id for status in agent.statuses) == CANONICAL_STATUS_ORDER
        for agent in scene.agents
    )
    assert sum(len(agent.statuses) for agent in scene.agents) == 90
    assert all(len(agent.modifiers) == 2 for agent in scene.agents)
    assert sum(len(agent.modifiers) for agent in scene.agents) == 20
    activations = tuple(
        event for event in batch.events if isinstance(event, AcceptedActivationEventV1)
    )
    assert len(activations) == 10
    assert len({event.source_global_slot for event in activations}) == 10
    assert all(not hasattr(event, "amount") for event in activations)

    outcomes = tuple(
        event for event in batch.events if isinstance(event, NetHealthEventV1)
    )
    assert len(outcomes) == 8
    health_by_slot = {agent.global_slot: agent.current_health for agent in scene.agents}
    assert all(
        health_by_slot[event.recipient_global_slot] == event.health_after
        for event in outcomes
    )

    charge_events = tuple(
        event for event in batch.events if isinstance(event, ChargeDisplacementEventV1)
    )
    assert len(charge_events) == 2
    assert {event.source_global_slot for event in charge_events} == {1, 6}

    lifecycle_events = tuple(
        event for event in batch.events if isinstance(event, StatusLifecycleEventV1)
    )
    assert len(lifecycle_events) == 12
    activation_ids = {event.event_id for event in activations}
    assert all(
        set(event.application_event_ids) <= activation_ids
        and len(event.application_event_ids) == 1
        for event in lifecycle_events
    )
    assert len(batch.events) == 32


def test_route_collision_retains_reciprocal_crossing_and_local_routes() -> None:
    fixture = get_renderer_fixture("route_collision")
    batch = fixture.event_batch
    assert batch is not None
    routes = tuple(
        event for event in batch.events if isinstance(event, AcceptedActivationEventV1)
    )
    assert len(routes) == 9
    endpoint_pairs = Counter(
        (event.source_global_slot, event.target_global_slot) for event in routes
    )
    assert (0, 5) in endpoint_pairs and (5, 0) in endpoint_pairs
    assert endpoint_pairs[(1, 6)] == 2
    assert endpoint_pairs[(2, 7)] == 1
    assert endpoint_pairs[(3, 8)] == 1
    assert endpoint_pairs[(0, 8)] == 1
    assert endpoint_pairs[(3, 5)] == 1
    local = next(
        event
        for event in routes
        if (event.source_global_slot, event.target_global_slot) == (4, 9)
    )
    assert local.source_anchor is not None
    assert local.target_anchor is not None
    assert dist(local.source_anchor, local.target_anchor) < 0.1


def test_mixed_net_zero_separates_intent_from_recipient_outcome() -> None:
    fixture = get_renderer_fixture("mixed_net_zero")
    batch = fixture.event_batch
    assert batch is not None
    activations = tuple(
        event for event in batch.events if isinstance(event, AcceptedActivationEventV1)
    )
    assert tuple(event.token_id for event in activations) == (
        "basic_damage",
        "basic_heal",
    )
    assert all(not hasattr(event, "amount") for event in activations)
    outcomes = tuple(
        event for event in batch.events if isinstance(event, NetHealthEventV1)
    )
    assert len(outcomes) == 1
    assert outcomes[0].recipient_global_slot == 5
    assert outcomes[0].net_delta == 0.0
    assert outcomes[0].outcome == "unchanged"


def test_viewport_matrix_has_four_layouts_and_explicit_reduced_motion() -> None:
    viewport = get_renderer_fixture("viewport_matrix")
    assert tuple(
        (case.label, case.width, case.height, case.expected_layout)
        for case in viewport.viewports
    ) == (
        ("desktop", 1440, 900, "split"),
        ("compact", 1024, 768, "split"),
        ("minimum", 960, 600, "split"),
        ("stacked", 800, 900, "stacked"),
    )
    assert viewport.exercise_reduced_motion is True
    batch = viewport.event_batch
    assert batch is not None
    assert len(batch.events) == 3
    assert all(
        event.event_id.startswith("synthetic:viewport_matrix:")
        for event in batch.events
    )
    crowded = get_renderer_fixture("crowded_teamfight")
    assert crowded.event_batch is not None
    assert {event.event_id for event in batch.events}.isdisjoint(
        event.event_id for event in crowded.event_batch.events
    )


def test_pov_redaction_pairs_privileged_and_safe_scene_event_payloads() -> None:
    pov = get_renderer_fixture("pov_redaction")
    assert pov.privileged_source_scene is not None
    assert pov.privileged_source_event_batch is not None
    assert pov.privileged_source_scene.audience == "researcher"
    assert pov.scene.audience == "agent_pov"
    assert 5 in {agent.global_slot for agent in pov.privileged_source_scene.agents}
    assert 5 not in {agent.global_slot for agent in pov.scene.agents}
    assert pov.scene.observer_visibility == ()
    batch = pov.event_batch
    assert batch is not None
    source_batch = pov.privileged_source_event_batch
    assert len(source_batch.events) == 7
    assert len(batch.events) == 5

    source_activations = tuple(
        event
        for event in source_batch.events
        if isinstance(event, AcceptedActivationEventV1)
    )
    assert len(source_activations) == 2
    hidden_activation = next(
        event for event in source_activations if event.source_global_slot == 5
    )
    safe_activations = tuple(
        event for event in batch.events if isinstance(event, AcceptedActivationEventV1)
    )
    assert len(safe_activations) == 1
    activation = safe_activations[0]
    assert activation.target_disclosure == "redacted"
    assert activation.target_global_slot is None
    assert activation.target_anchor is None

    safe_outcomes = tuple(
        event for event in batch.events if isinstance(event, NetHealthEventV1)
    )
    assert tuple(event.recipient_global_slot for event in safe_outcomes) == (0,)

    source_lifecycles = tuple(
        event
        for event in source_batch.events
        if isinstance(event, StatusLifecycleEventV1)
    )
    assert len(source_lifecycles) == 3
    assert all(
        event.application_event_ids == (hidden_activation.event_id,)
        for event in source_lifecycles
    )
    safe_lifecycles = tuple(
        event for event in batch.events if isinstance(event, StatusLifecycleEventV1)
    )
    assert len(safe_lifecycles) == 3
    assert all(event.application_event_ids == () for event in safe_lifecycles)
    safe_event_ids = {event.event_id for event in batch.events}
    assert hidden_activation.event_id not in safe_event_ids


def test_fixture_envelope_rejects_unlabelled_payloads() -> None:
    fixture = get_renderer_fixture("mixed_net_zero")
    with pytest.raises(ValueError, match="SYNTHETIC"):
        RendererFixtureV1(
            name="mixed_net_zero",
            description="ordinary fixture",
            scene=fixture.scene,
            event_batch=fixture.event_batch,
        )

    with pytest.raises(ValueError, match="supplied together"):
        RendererFixtureV1(
            name="pov_redaction",
            description="SYNTHETIC: incomplete privileged comparison.",
            scene=get_renderer_fixture("pov_redaction").scene,
            privileged_source_scene=get_renderer_fixture(
                "pov_redaction"
            ).privileged_source_scene,
        )
