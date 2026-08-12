"""Pure tests for renderer-neutral scene, event, and vocabulary contracts."""

import json
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

from marl_battlegrounds.rendering.scene import (
    EVENT_SCHEMA_VERSION,
    SCENE_SCHEMA_VERSION,
    AcceptedActivationEventV1,
    AgentSceneV1,
    BattlefieldSceneV1,
    MapSceneV1,
    NetHealthEventV1,
    ObserverVisibilitySceneV1,
    ObstacleSceneV1,
    PendingRouteSceneV1,
    RejectedActionEventV1,
    RespawnWaveSceneV2,
    SelectedLegalitySceneV1,
    SelectionSceneV1,
    StatusLifecycleEventV1,
    StatusSceneV1,
    VisualEventBatchV1,
    to_jsonable,
)
from marl_battlegrounds.rendering.vocabulary import (
    CANONICAL_STATUS_ORDER,
    CATALOG_STATUS_ID_BY_CHANNEL,
    CATALOG_STATUS_TOKEN_ID_BY_STATUS_ID,
    STATUS_TOKENS,
    class_token_from_id,
    lookup_status_token,
    status_sort_key,
    status_token_id_from_catalog_status_id,
)


def _status(token_id: str, priority: int) -> StatusSceneV1:
    definition = lookup_status_token(token_id)
    assert definition.source_class_id is not None
    return StatusSceneV1(
        token_id=token_id,
        duration=2,
        source_class_id=definition.source_class_id,
        label=definition.label,
        short_label=definition.short_label,
        accessible_name=definition.accessible_name,
        priority=priority,
    )


def test_researcher_respawn_wave_countdown_stays_below_period() -> None:
    wave = RespawnWaveSceneV2(
        team_index=0,
        team_id=1,
        period_steps=10,
        countdown_steps=9,
    )
    assert wave.countdown_steps == 9
    with pytest.raises(ValueError, match="less than period_steps"):
        replace(wave, countdown_steps=10)


def _agent() -> AgentSceneV1:
    return AgentSceneV1(
        global_slot=0,
        team_id=1,
        class_id=1,
        position=(2.0, 3.0),
        radius=0.5,
        active=True,
        alive=True,
        current_health=80.0,
        max_health=100.0,
        ultimate_cooldown=0,
        effective_speed=1.0,
        statuses=(
            _status("stun_warrior_charge", 0),
            _status("mage_burst", 8),
        ),
    )


def _activation(
    event_id: str = "transition-1:activation-0",
) -> AcceptedActivationEventV1:
    return AcceptedActivationEventV1(
        event_id=event_id,
        transition_id=1,
        token_id="basic_damage",
        source_global_slot=0,
        target_global_slot=5,
        source_anchor=(2.0, 3.0),
        target_anchor=(6.0, 3.0),
        target_disclosure="public",
        lane=0,
        source_class_id=1,
    )


def _status_lifecycle(
    application_event_ids: tuple[str, ...] = (),
) -> StatusLifecycleEventV1:
    return StatusLifecycleEventV1(
        event_id="transition-1:status-0",
        transition_id=1,
        recipient_global_slot=5,
        recipient_anchor=(6.0, 3.0),
        token_id="stun_warrior_charge",
        change="applied",
        duration_before=0,
        duration_after=2,
        source_class_id=2,
        application_event_ids=application_event_ids,
    )


def test_canonical_status_registry_is_complete_ordered_and_stable() -> None:
    assert tuple(definition.token_id for definition in STATUS_TOKENS) == (
        CANONICAL_STATUS_ORDER
    )
    assert len(CANONICAL_STATUS_ORDER) == 9
    assert {definition.glyph for definition in STATUS_TOKENS[:3]} == {"⬢"}
    assert {definition.glyph for definition in STATUS_TOKENS[3:6]} == {"↻"}
    assert len({definition.accessible_name for definition in STATUS_TOKENS[:3]}) == 3
    assert len({definition.accessible_name for definition in STATUS_TOKENS[3:6]}) == 3
    assert len({definition.source_class_id for definition in STATUS_TOKENS[:3]}) == 3
    assert len({definition.source_class_id for definition in STATUS_TOKENS[3:6]}) == 3
    assert tuple(sorted(CANONICAL_STATUS_ORDER, key=status_sort_key)) == (
        CANONICAL_STATUS_ORDER
    )
    assert tuple(CATALOG_STATUS_TOKEN_ID_BY_STATUS_ID.values()) == (
        CANONICAL_STATUS_ORDER
    )
    assert tuple(
        CATALOG_STATUS_ID_BY_CHANNEL.index(status_id)
        for status_id in CATALOG_STATUS_TOKEN_ID_BY_STATUS_ID
    ) == (3, 4, 5, 0, 1, 2, 6, 8, 7)
    assert (
        tuple(
            status_token_id_from_catalog_status_id(status_id)
            for status_id in CATALOG_STATUS_TOKEN_ID_BY_STATUS_ID
        )
        == CANONICAL_STATUS_ORDER
    )
    assert lookup_status_token("future_status").family == "unknown"
    assert lookup_status_token("future_status").token_id == "future_status"
    assert class_token_from_id(999).family == "unknown"


def test_scene_records_are_frozen_slotted_and_scalar_only() -> None:
    scene = BattlefieldSceneV1(
        schema_version=SCENE_SCHEMA_VERSION,
        audience="researcher",
        audience_badge="PRIVILEGED RESEARCHER VIEW",
        map=MapSceneV1(
            width=12.0,
            height=8.0,
            obstacles=(
                ObstacleSceneV1(
                    obstacle_id="pillar-0",
                    kind="pillar",
                    center=(4.0, 4.0),
                    radius=0.75,
                ),
            ),
        ),
        agents=(_agent(),),
        selection=SelectionSceneV1(
            controlled_global_slot=0,
            selected_global_slot=None,
        ),
    )

    payload = to_jsonable(scene)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["schema_version"] == SCENE_SCHEMA_VERSION  # type: ignore[index]
    assert hasattr(BattlefieldSceneV1, "__slots__")
    with pytest.raises(FrozenInstanceError):
        scene.audience = "agent_pov"  # type: ignore[misc]
    with pytest.raises(TypeError):
        to_jsonable(np.asarray((1.0, 2.0), dtype=np.float32))


def test_scene_schema_rejects_invalid_geometry_order_and_audience_marking() -> None:
    with pytest.raises(ValueError, match="radius"):
        ObstacleSceneV1(
            obstacle_id="bad",
            kind="pillar",
            center=(1.0, 1.0),
            radius=None,
        )
    with pytest.raises(ValueError, match="width and height"):
        ObstacleSceneV1(
            obstacle_id="bad",
            kind="wall",
            center=(1.0, 1.0),
            width=1.0,
        )
    with pytest.raises(ValueError, match="canonical"):
        AgentSceneV1(
            **{
                field.name: getattr(_agent(), field.name)
                for field in fields(AgentSceneV1)
                if field.name != "statuses"
            },
            statuses=(
                _status("mage_burst", 8),
                _status("stun_warrior_charge", 0),
            ),
        )
    with pytest.raises(ValueError, match="PRIVILEGED"):
        BattlefieldSceneV1(
            schema_version=SCENE_SCHEMA_VERSION,
            audience="researcher",
            audience_badge="Researcher",
            map=MapSceneV1(width=1.0, height=1.0),
            agents=(),
        )


def test_scene_schema_requires_exact_python_bool_fields() -> None:
    with pytest.raises(ValueError, match="active must be a Python bool"):
        replace(_agent(), active=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alive must be a Python bool"):
        replace(_agent(), alive=np.bool_(True))  # type: ignore[arg-type]

    legality = SelectedLegalitySceneV1(
        controlled_global_slot=0,
        target_global_slot=5,
        target_action=6,
        lane_0_available=True,
        lane_1_available=False,
        armed_lane=0,
        armed_pair_legal=True,
    )
    with pytest.raises(ValueError, match="lane_0_available must be a Python bool"):
        replace(legality, lane_0_available=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lane_1_available must be a Python bool"):
        replace(legality, lane_1_available=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="armed_pair_legal must be a Python bool"):
        replace(legality, armed_pair_legal=1)  # type: ignore[arg-type]

    pending = PendingRouteSceneV1(
        source_global_slot=0,
        target_global_slot=5,
        source_anchor=(2.0, 3.0),
        target_anchor=(6.0, 3.0),
        lane=0,
        legal=True,
    )
    with pytest.raises(ValueError, match="legal must be a Python bool"):
        replace(pending, legal=1)  # type: ignore[arg-type]

    visibility = ObserverVisibilitySceneV1(
        observer_global_slot=0,
        candidate_global_slot=5,
        visible=True,
    )
    with pytest.raises(ValueError, match="visible must be a Python bool"):
        replace(visibility, visible=1)  # type: ignore[arg-type]

    rejection = RejectedActionEventV1(
        event_id="transition-1:rejection-0",
        transition_id=1,
        actor_global_slot=0,
        component="movement",
        actor_anchor=(2.0, 3.0),
        target_global_slot=None,
        target_anchor=None,
        target_disclosure="target_none",
        lane=None,
        movement_mask_value=False,
        pair_mask_value=True,
    )
    with pytest.raises(ValueError, match="movement_mask_value must be a Python bool"):
        replace(rejection, movement_mask_value=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pair_mask_value must be a Python bool"):
        replace(rejection, pair_mask_value=1)  # type: ignore[arg-type]


def test_scene_schema_rejects_bool_lanes_versions_and_non_allowlisted_records() -> None:
    scene = BattlefieldSceneV1(
        schema_version=SCENE_SCHEMA_VERSION,
        audience="researcher",
        audience_badge="PRIVILEGED RESEARCHER VIEW",
        map=MapSceneV1(width=12.0, height=8.0),
        agents=(_agent(),),
    )
    with pytest.raises(ValueError, match="schema_version must be a Python int"):
        replace(scene, schema_version=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="map must be MapSceneV1"):
        replace(scene, map=np.asarray((1.0, 2.0)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="selection must be SelectionSceneV1"):
        replace(scene, selection=np.asarray((1.0, 2.0)))  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match="selected_legality must be SelectedLegalitySceneV1",
    ):
        replace(scene, selected_legality="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pending_route must be PendingRouteSceneV1"):
        replace(scene, pending_route=False)  # type: ignore[arg-type]

    legality = SelectedLegalitySceneV1(
        controlled_global_slot=0,
        target_global_slot=5,
        target_action=6,
        lane_0_available=True,
        lane_1_available=False,
        armed_lane=0,
        armed_pair_legal=True,
    )
    with pytest.raises(ValueError, match="armed_lane must be the Python int"):
        replace(legality, armed_lane=False)  # type: ignore[arg-type]

    pending = PendingRouteSceneV1(
        source_global_slot=0,
        target_global_slot=5,
        source_anchor=(2.0, 3.0),
        target_anchor=(6.0, 3.0),
        lane=0,
        legal=True,
    )
    with pytest.raises(ValueError, match="lane must be the Python int"):
        replace(pending, lane=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="lane must be the Python int"):
        replace(_activation(), lane=False)  # type: ignore[arg-type]

    rejection = RejectedActionEventV1(
        event_id="transition-1:rejection-0",
        transition_id=1,
        actor_global_slot=0,
        component="movement",
        actor_anchor=(2.0, 3.0),
        target_global_slot=None,
        target_anchor=None,
        target_disclosure="target_none",
        lane=None,
        movement_mask_value=False,
        pair_mask_value=True,
    )
    with pytest.raises(ValueError, match="lane must be the Python int"):
        replace(rejection, lane=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="schema_version must be a Python int"):
        VisualEventBatchV1(
            schema_version=True,  # type: ignore[arg-type]
            transition_id=1,
            simulator_step=1,
            events=(),
        )


def test_scene_schema_requires_exact_tuple_containers() -> None:
    with pytest.raises(ValueError, match="obstacles must be a Python tuple"):
        MapSceneV1(width=12.0, height=8.0, obstacles=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="statuses must be a Python tuple"):
        replace(_agent(), statuses=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="modifiers must be a Python tuple"):
        replace(_agent(), modifiers=[])  # type: ignore[arg-type]

    scene = BattlefieldSceneV1(
        schema_version=SCENE_SCHEMA_VERSION,
        audience="researcher",
        audience_badge="PRIVILEGED RESEARCHER VIEW",
        map=MapSceneV1(width=12.0, height=8.0),
        agents=(_agent(),),
    )
    with pytest.raises(ValueError, match="agents must be a Python tuple"):
        replace(scene, agents=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="aura_fields must be a Python tuple"):
        replace(scene, aura_fields=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ranges must be a Python tuple"):
        replace(scene, ranges=[])  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match="observer_visibility must be a Python tuple",
    ):
        replace(scene, observer_visibility=[])  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="application_event_ids must be a Python tuple",
    ):
        _status_lifecycle(["transition-1:activation-0"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="events must be a Python tuple"):
        VisualEventBatchV1(
            schema_version=EVENT_SCHEMA_VERSION,
            transition_id=1,
            simulator_step=1,
            events=[],  # type: ignore[arg-type]
        )


def test_scene_schema_requires_exact_tuple_element_types() -> None:
    with pytest.raises(ValueError, match=r"obstacles\[0\].*ObstacleSceneV1"):
        MapSceneV1(
            width=12.0,
            height=8.0,
            obstacles=("not-an-obstacle",),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r"statuses\[0\].*StatusSceneV1"):
        replace(_agent(), statuses=("not-a-status",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"modifiers\[0\].*ModifierSceneV1"):
        replace(_agent(), modifiers=("not-a-modifier",))  # type: ignore[arg-type]

    scene = BattlefieldSceneV1(
        schema_version=SCENE_SCHEMA_VERSION,
        audience="researcher",
        audience_badge="PRIVILEGED RESEARCHER VIEW",
        map=MapSceneV1(width=12.0, height=8.0),
        agents=(_agent(),),
    )
    with pytest.raises(ValueError, match=r"agents\[0\].*AgentSceneV1"):
        replace(scene, agents=("not-an-agent",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"aura_fields\[0\].*AuraFieldSceneV1"):
        replace(scene, aura_fields=("not-an-aura",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"ranges\[0\].*RangeSceneV1"):
        replace(scene, ranges=("not-a-range",))  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=r"observer_visibility\[0\].*ObserverVisibilitySceneV1",
    ):
        replace(scene, observer_visibility=("not-visibility",))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"application_event_ids\[0\].*str"):
        _status_lifecycle((1,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"events\[0\].*AcceptedActivationEventV1"):
        VisualEventBatchV1(
            schema_version=EVENT_SCHEMA_VERSION,
            transition_id=1,
            simulator_step=1,
            events=("not-an-event",),  # type: ignore[arg-type]
        )


def test_status_lifecycle_application_ids_are_nonempty_unique_and_local() -> None:
    with pytest.raises(ValueError, match="non-empty Python string"):
        _status_lifecycle(("",))
    with pytest.raises(ValueError, match="unique"):
        _status_lifecycle(
            (
                "transition-1:activation-0",
                "transition-1:activation-0",
            )
        )

    activation = _activation()
    lifecycle = _status_lifecycle((activation.event_id,))
    batch = VisualEventBatchV1(
        schema_version=EVENT_SCHEMA_VERSION,
        transition_id=1,
        simulator_step=1,
        events=(activation, lifecycle),
    )
    assert batch.events == (activation, lifecycle)

    with pytest.raises(ValueError, match="AcceptedActivationEventV1"):
        VisualEventBatchV1(
            schema_version=EVENT_SCHEMA_VERSION,
            transition_id=1,
            simulator_step=1,
            events=(_status_lifecycle(("transition-1:activation-missing",)),),
        )

    health = NetHealthEventV1(
        event_id="transition-1:net-health-0",
        transition_id=1,
        recipient_global_slot=5,
        recipient_anchor=(6.0, 3.0),
        health_before=10.0,
        health_after=10.0,
        net_delta=0.0,
        outcome="unchanged",
    )
    with pytest.raises(ValueError, match="AcceptedActivationEventV1"):
        VisualEventBatchV1(
            schema_version=EVENT_SCHEMA_VERSION,
            transition_id=1,
            simulator_step=1,
            events=(
                health,
                _status_lifecycle((health.event_id,)),
            ),
        )


def test_activation_event_preserves_prestate_anchors_without_health_amount() -> None:
    event = AcceptedActivationEventV1(
        event_id="transition-1:activation-0",
        transition_id=1,
        token_id="basic_damage",
        source_global_slot=0,
        target_global_slot=5,
        source_anchor=(2.0, 3.0),
        target_anchor=(6.0, 3.0),
        target_disclosure="public",
        lane=0,
        source_class_id=1,
    )
    assert not hasattr(event, "amount")
    assert not hasattr(event, "damage")
    assert event.source_anchor == (2.0, 3.0)
    assert event.target_anchor == (6.0, 3.0)

    with pytest.raises(ValueError, match="public targets"):
        AcceptedActivationEventV1(
            event_id="transition-1:activation-1",
            transition_id=1,
            token_id="basic_damage",
            source_global_slot=0,
            target_global_slot=None,
            source_anchor=(2.0, 3.0),
            target_anchor=None,
            target_disclosure="public",
            lane=0,
            source_class_id=1,
        )


def test_target_none_and_redacted_are_structurally_distinct() -> None:
    target_none = AcceptedActivationEventV1(
        event_id="transition-1:activation-0",
        transition_id=1,
        token_id="mage_burst",
        source_global_slot=0,
        target_global_slot=None,
        source_anchor=(2.0, 3.0),
        target_anchor=None,
        target_disclosure="target_none",
        lane=1,
        source_class_id=1,
    )
    redacted = AcceptedActivationEventV1(
        event_id="transition-1:activation-1",
        transition_id=1,
        token_id="basic_damage",
        source_global_slot=0,
        target_global_slot=None,
        source_anchor=(2.0, 3.0),
        target_anchor=None,
        target_disclosure="redacted",
        lane=0,
        source_class_id=1,
    )
    assert target_none.target_disclosure != redacted.target_disclosure


@pytest.mark.parametrize(
    ("health_before", "health_after", "outcome"),
    ((10.0, 4.0, "damage"), (4.0, 10.0, "healing"), (10.0, 10.0, "unchanged")),
)
def test_net_health_event_is_exact_recipient_level_truth(
    health_before: float,
    health_after: float,
    outcome: str,
) -> None:
    event = NetHealthEventV1(
        event_id="transition-1:net-health-0",
        transition_id=1,
        recipient_global_slot=5,
        recipient_anchor=(6.0, 3.0),
        health_before=health_before,
        health_after=health_after,
        net_delta=health_after - health_before,
        outcome=outcome,  # type: ignore[arg-type]
    )
    assert not hasattr(event, "source_global_slot")
    assert event.net_delta == health_after - health_before

    with pytest.raises(ValueError, match="net_delta"):
        NetHealthEventV1(
            event_id="transition-1:net-health-1",
            transition_id=1,
            recipient_global_slot=5,
            recipient_anchor=(6.0, 3.0),
            health_before=10.0,
            health_after=4.0,
            net_delta=-5.0,
            outcome="damage",
        )


def test_event_batch_rejects_duplicate_or_cross_transition_ids() -> None:
    event = NetHealthEventV1(
        event_id="transition-1:net-health-0",
        transition_id=1,
        recipient_global_slot=5,
        recipient_anchor=(6.0, 3.0),
        health_before=10.0,
        health_after=10.0,
        net_delta=0.0,
        outcome="unchanged",
    )
    with pytest.raises(ValueError, match="unique"):
        VisualEventBatchV1(
            schema_version=EVENT_SCHEMA_VERSION,
            transition_id=1,
            simulator_step=1,
            events=(event, event),
        )
    with pytest.raises(ValueError, match="transition_id"):
        VisualEventBatchV1(
            schema_version=EVENT_SCHEMA_VERSION,
            transition_id=2,
            simulator_step=2,
            events=(event,),
        )
