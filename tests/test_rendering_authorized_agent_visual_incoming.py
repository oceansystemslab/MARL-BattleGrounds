"""Focused Agent POV visual-incoming authority and noninterference proofs."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import pytest
from pydantic import TypeAdapter
from tests.evaluation_fixtures import captured_evaluation_trajectory

from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.pov import (
    build_actor_pov_adjacent_transition_slice_v1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    build_no_shared_obs_authorized_scene_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AgentPovVisualIncomingAgentPhaseTrajectoryV1,
    AgentPovVisualIncomingAgentRespawnedEventV1,
    AgentPovVisualIncomingRecipientHealthResolutionEventV1,
    AgentPovVisualIncomingSummaryV1,
    AuthorizedBattlefieldSceneV1,
    ReplayIncomingAbilityActivatedEventV1,
    ReplayIncomingActionRejectedEventV1,
    ReplayIncomingStatusAppliedEventV1,
    build_agent_pov_visual_incoming_summary_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    build_visual_event_batch_v2,
)
from marl_battlegrounds.rendering.scene import (
    AbilityActivatedEventV2,
    ActionRejectedEventV2,
    AgentRespawnedEventV2,
    ChargePhaseDisplacementEventV2,
    CombatCountdownResetEventV2,
    LethalDamageContributionEventV2,
    OrdinaryMovementPhaseDisplacementEventV2,
    RecipientHealthResolutionEventV2,
    RespawnWaveOccurredEventV2,
    SourceDamageOutputEventV2,
    SourceHealingOutputEventV2,
    StatusAppliedEventV2,
    VisualAgentPhaseTrajectoryV2,
    VisualEventBatchV2,
    VisualEventV2,
    VisualTeamAnchorV2,
)


@dataclass(frozen=True)
class _AgentVisualCase:
    batch: VisualEventBatchV2
    transition_start_scene: AuthorizedBattlefieldSceneV1
    successor_scene: AuthorizedBattlefieldSceneV1
    recipient_public_agent_id: str


@pytest.fixture(scope="module")
def agent_visual_case() -> _AgentVisualCase:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
    )
    view = EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[0],
        transition=trajectory.transitions[0],
        successor_frame=trajectory.frames[1],
    )
    recipient_public_agent_id = "agent-slot-0"
    source = build_actor_pov_adjacent_transition_slice_v1(view, global_slot=0)
    start = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id="agent-visual-authority",
        frame_index=0,
    )
    successor = build_no_shared_obs_authorized_scene_v1(
        source,
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id="agent-visual-authority",
        frame_index=1,
    )
    return _AgentVisualCase(
        batch=build_visual_event_batch_v2(view),
        transition_start_scene=start.scene,
        successor_scene=successor.scene,
        recipient_public_agent_id=recipient_public_agent_id,
    )


def _trajectory(
    batch: VisualEventBatchV2,
    global_slot: int,
) -> VisualAgentPhaseTrajectoryV2:
    return next(
        row for row in batch.agent_phase_trajectories if row.global_slot == global_slot
    )


def _event_id(batch: VisualEventBatchV2, ordinal: int) -> str:
    return f"{batch.transition_id}:event:{ordinal:04d}"


def _recipient_health_event(
    batch: VisualEventBatchV2,
    *,
    ordinal: int,
    total_effective_damage: float = 3.0,
    total_effective_healing: float = 0.0,
    health_after_combat_resolution: float = 97.0,
    realized_net_health_change: float = -3.0,
) -> RecipientHealthResolutionEventV2:
    recipient = _trajectory(batch, 2).transition_start
    return RecipientHealthResolutionEventV2(
        event_id=_event_id(batch, ordinal),
        transition_id=batch.transition_id,
        ordinal=ordinal,
        recipient_global_slot=2,
        transition_start_health=100.0,
        total_effective_damage=total_effective_damage,
        total_effective_healing=total_effective_healing,
        health_after_combat_resolution=health_after_combat_resolution,
        realized_net_health_change=realized_net_health_change,
        recipient_anchor=recipient,
    )


def _with_events(
    batch: VisualEventBatchV2,
    events: tuple[VisualEventV2, ...],
) -> VisualEventBatchV2:
    return replace(batch, events=events)


def _without_agent(
    scene: AuthorizedBattlefieldSceneV1,
    *,
    public_agent_id: str,
) -> AuthorizedBattlefieldSceneV1:
    agents = tuple(
        row for row in scene.agents if row.public_agent_id != public_agent_id
    )
    represented_classes = {row.class_id for row in agents}
    return replace(
        scene,
        agents=agents,
        aura_fields=tuple(
            row
            for row in scene.aura_fields
            if row.source_public_agent_id != public_agent_id
        ),
        class_mechanics=tuple(
            row for row in scene.class_mechanics if row.class_id in represented_classes
        ),
        spawn_pads=tuple(
            replace(
                row,
                assigned_presentation_key=None,
                assigned_public_agent_id=None,
            )
            if row.assigned_public_agent_id == public_agent_id
            else row
            for row in scene.spawn_pads
        ),
    )


def _build(
    case: _AgentVisualCase,
    batch: VisualEventBatchV2,
    *,
    namespace: str = "actor-pov",
    transition_start_scene: AuthorizedBattlefieldSceneV1 | None = None,
    successor_scene: AuthorizedBattlefieldSceneV1 | None = None,
) -> AgentPovVisualIncomingSummaryV1:
    prefix = f"{batch.episode_id}:{namespace}:{case.recipient_public_agent_id}"
    return build_agent_pov_visual_incoming_summary_v1(
        batch,
        transition_start_scene=(
            case.transition_start_scene
            if transition_start_scene is None
            else transition_start_scene
        ),
        successor_scene=(
            case.successor_scene if successor_scene is None else successor_scene
        ),
        recipient_public_agent_id=case.recipient_public_agent_id,
        incoming_recipient_transition_id=(
            f"{prefix}:transition:{batch.transition_index}"
        ),
        incoming_start_recipient_frame_id=(f"{prefix}:frame:{batch.transition_index}"),
        incoming_successor_recipient_frame_id=(
            f"{prefix}:frame:{batch.transition_index + 1}"
        ),
    )


def test_agent_visual_trajectory_has_optional_start_and_no_charge_phase(
    agent_visual_case: _AgentVisualCase,
) -> None:
    summary = _build(agent_visual_case, agent_visual_case.batch)

    assert summary.source_episode_id == agent_visual_case.batch.episode_id
    assert tuple(
        row.agent_public_agent_id for row in summary.agent_phase_trajectories
    ) == ("agent-slot-0", "agent-slot-1", "agent-slot-2")
    assert all(
        row.transition_start is not None for row in summary.agent_phase_trajectories
    )
    assert {
        field.name for field in fields(AgentPovVisualIncomingAgentPhaseTrajectoryV1)
    } == {
        "agent_presentation_key",
        "agent_public_agent_id",
        "agent_class_id",
        "transition_start",
        "successor",
    }
    payload = TypeAdapter(AgentPovVisualIncomingSummaryV1).dump_python(
        summary,
        mode="json",
    )
    assert "post_charge" not in payload["agent_phase_trajectories"][0]


def test_agent_visual_filters_hidden_and_global_events_but_keeps_target_health(
    agent_visual_case: _AgentVisualCase,
) -> None:
    batch = agent_visual_case.batch
    visible_source = _trajectory(batch, 1)
    visible_recipient = _trajectory(batch, 2)
    hidden_source = _trajectory(batch, 5)
    hidden_recipient = _trajectory(batch, 6)
    events: tuple[VisualEventV2, ...] = (
        ActionRejectedEventV2(
            event_id=_event_id(batch, 0),
            transition_id=batch.transition_id,
            ordinal=0,
            actor_global_slot=3,
            actor_public_agent_id=batch.public_agent_id_by_global_slot[3],
            actor_configured_active=False,
            rejection_component="domain",
            submitted_move_action=99,
            submitted_select_target_action=0,
            submitted_use_ultimate_action=0,
            actor_anchor=None,
        ),
        AbilityActivatedEventV2(
            event_id=_event_id(batch, 1),
            transition_id=batch.transition_id,
            ordinal=1,
            source_global_slot=5,
            ability_component="basic",
            recipient_global_slot=6,
            source_anchor=hidden_source.transition_start,
            recipient_anchor=hidden_recipient.transition_start,
        ),
        AbilityActivatedEventV2(
            event_id=_event_id(batch, 2),
            transition_id=batch.transition_id,
            ordinal=2,
            source_global_slot=1,
            ability_component="basic",
            recipient_global_slot=2,
            source_anchor=visible_source.transition_start,
            recipient_anchor=visible_recipient.transition_start,
        ),
        SourceDamageOutputEventV2(
            event_id=_event_id(batch, 3),
            transition_id=batch.transition_id,
            ordinal=3,
            source_global_slot=1,
            recipient_global_slot=2,
            raw_damage_output=3.0,
            source_modified_damage_output=3.0,
            recipient_damage_modifier=1.0,
            mage_damage_aura_covering_emitter_global_slots=(5,),
            warrior_mitigation_aura_covering_emitter_global_slots=(),
            source_anchor=visible_source.transition_start,
            recipient_anchor=visible_recipient.transition_start,
        ),
        _recipient_health_event(batch, ordinal=4),
        ChargePhaseDisplacementEventV2(
            event_id=_event_id(batch, 5),
            transition_id=batch.transition_id,
            ordinal=5,
            agent_global_slot=1,
            realized_displacement=(
                visible_source.post_charge.position[0]
                - visible_source.transition_start.position[0],
                visible_source.post_charge.position[1]
                - visible_source.transition_start.position[1],
            ),
            start_anchor=visible_source.transition_start,
            end_anchor=visible_source.post_charge,
        ),
        OrdinaryMovementPhaseDisplacementEventV2(
            event_id=_event_id(batch, 6),
            transition_id=batch.transition_id,
            ordinal=6,
            agent_global_slot=1,
            realized_displacement=(
                visible_source.successor.position[0]
                - visible_source.post_charge.position[0],
                visible_source.successor.position[1]
                - visible_source.post_charge.position[1],
            ),
            start_anchor=visible_source.post_charge,
            end_anchor=visible_source.successor,
        ),
        RespawnWaveOccurredEventV2(
            event_id=_event_id(batch, 7),
            transition_id=batch.transition_id,
            ordinal=7,
            team_index=0,
            team_id=1,
            team_anchor=VisualTeamAnchorV2(
                phase="successor",
                team_index=0,
                team_id=1,
            ),
        ),
    )

    summary = _build(agent_visual_case, _with_events(batch, events))

    assert tuple(type(event) for event in summary.events) == (
        ReplayIncomingAbilityActivatedEventV1,
        AgentPovVisualIncomingRecipientHealthResolutionEventV1,
    )
    assert summary.ordered_event_kinds == (
        "ability_activated",
        "recipient_health_resolution",
    )
    assert summary.ordered_event_ids == tuple(
        f"{summary.incoming_recipient_transition_id}:visual-event:{ordinal:04d}"
        for ordinal in range(2)
    )
    assert tuple(event.ordinal for event in summary.events) == (0, 1)
    serialized_health = TypeAdapter(AgentPovVisualIncomingSummaryV1).dump_json(summary)
    assert b"total_effective_damage" not in serialized_health
    assert b"total_effective_healing" not in serialized_health


def test_agent_visual_rejection_discloses_only_the_fixed_recipient_action(
    agent_visual_case: _AgentVisualCase,
) -> None:
    batch = agent_visual_case.batch
    recipient = _trajectory(batch, 0)
    visible_nonrecipient = _trajectory(batch, 1)
    events: tuple[VisualEventV2, ...] = (
        ActionRejectedEventV2(
            event_id=_event_id(batch, 0),
            transition_id=batch.transition_id,
            ordinal=0,
            actor_global_slot=1,
            actor_public_agent_id=visible_nonrecipient.public_agent_id,
            actor_configured_active=True,
            rejection_component="domain",
            submitted_move_action=0,
            submitted_select_target_action=99,
            submitted_use_ultimate_action=0,
            actor_anchor=visible_nonrecipient.transition_start,
        ),
        ActionRejectedEventV2(
            event_id=_event_id(batch, 1),
            transition_id=batch.transition_id,
            ordinal=1,
            actor_global_slot=0,
            actor_public_agent_id=recipient.public_agent_id,
            actor_configured_active=True,
            rejection_component="domain",
            submitted_move_action=0,
            submitted_select_target_action=98,
            submitted_use_ultimate_action=0,
            actor_anchor=recipient.transition_start,
        ),
    )

    summary = _build(agent_visual_case, _with_events(batch, events))

    assert len(summary.events) == 1
    event = summary.events[0]
    assert type(event) is ReplayIncomingActionRejectedEventV1
    assert event.actor_identity.public_agent_id == (
        agent_visual_case.recipient_public_agent_id
    )
    assert event.submitted_action.target_action == 98
    serialized = TypeAdapter(AgentPovVisualIncomingSummaryV1).dump_json(summary)
    assert b'"target_action":99' not in serialized


def test_agent_visual_health_bytes_ignore_hidden_cancelling_gross_causes(
    agent_visual_case: _AgentVisualCase,
) -> None:
    batch = agent_visual_case.batch
    zero_gross = _with_events(
        batch,
        (
            _recipient_health_event(
                batch,
                ordinal=0,
                total_effective_damage=0.0,
                total_effective_healing=0.0,
                health_after_combat_resolution=100.0,
                realized_net_health_change=0.0,
            ),
        ),
    )
    hidden_cancelling_gross = _with_events(
        batch,
        (
            _recipient_health_event(
                batch,
                ordinal=0,
                total_effective_damage=7.0,
                total_effective_healing=7.0,
                health_after_combat_resolution=100.0,
                realized_net_health_change=0.0,
            ),
        ),
    )
    adapter = TypeAdapter(AgentPovVisualIncomingSummaryV1)

    zero_bytes = adapter.dump_json(_build(agent_visual_case, zero_gross))
    hidden_bytes = adapter.dump_json(_build(agent_visual_case, hidden_cancelling_gross))

    assert hidden_bytes == zero_bytes
    assert b"total_effective_damage" not in hidden_bytes
    assert b"total_effective_healing" not in hidden_bytes


def test_agent_visual_excludes_every_feed_only_canonical_event_kind(
    agent_visual_case: _AgentVisualCase,
) -> None:
    batch = agent_visual_case.batch
    source = _trajectory(batch, 1)
    recipient = _trajectory(batch, 2)
    events: tuple[VisualEventV2, ...] = (
        SourceDamageOutputEventV2(
            event_id=_event_id(batch, 0),
            transition_id=batch.transition_id,
            ordinal=0,
            source_global_slot=source.global_slot,
            recipient_global_slot=recipient.global_slot,
            raw_damage_output=5.0,
            source_modified_damage_output=5.0,
            recipient_damage_modifier=1.0,
            mage_damage_aura_covering_emitter_global_slots=(),
            warrior_mitigation_aura_covering_emitter_global_slots=(),
            source_anchor=source.transition_start,
            recipient_anchor=recipient.transition_start,
        ),
        SourceHealingOutputEventV2(
            event_id=_event_id(batch, 1),
            transition_id=batch.transition_id,
            ordinal=1,
            source_global_slot=source.global_slot,
            recipient_global_slot=recipient.global_slot,
            raw_healing_output=4.0,
            source_modified_healing_output=4.0,
            recipient_healing_modifier=1.0,
            source_anchor=source.transition_start,
            recipient_anchor=recipient.transition_start,
        ),
        CombatCountdownResetEventV2(
            event_id=_event_id(batch, 2),
            transition_id=batch.transition_id,
            ordinal=2,
            agent_global_slot=source.global_slot,
            agent_anchor=source.transition_start,
        ),
        LethalDamageContributionEventV2(
            event_id=_event_id(batch, 3),
            transition_id=batch.transition_id,
            ordinal=3,
            source_global_slot=source.global_slot,
            recipient_global_slot=recipient.global_slot,
            attributed_death_damage=5.0,
            source_anchor=source.successor,
            recipient_anchor=recipient.successor,
        ),
    )

    summary = _build(agent_visual_case, _with_events(batch, events))

    assert summary.events == ()
    assert summary.ordered_event_kinds == ()
    assert summary.event_count == 0
    schema = TypeAdapter(AgentPovVisualIncomingSummaryV1).json_schema()
    definitions = schema["$defs"]
    excluded = {
        "source_damage_output",
        "source_healing_output",
        "combat_countdown_reset",
        "lethal_damage_contribution",
    }
    assert excluded.isdisjoint(definitions["AgentPovVisualIncomingEventKindV1"]["enum"])
    assert excluded.isdisjoint(
        definitions["AgentPovVisualIncomingEventV1"]["discriminator"]["mapping"]
    )


def test_agent_visual_bytes_ignore_hidden_feed_only_lethal_scalar(
    agent_visual_case: _AgentVisualCase,
) -> None:
    batch = agent_visual_case.batch
    appearing_source = _trajectory(batch, 1)
    recipient = _trajectory(batch, 2)
    start_without_source = _without_agent(
        agent_visual_case.transition_start_scene,
        public_agent_id=appearing_source.public_agent_id,
    )

    def batch_with_attribution(value: float) -> VisualEventBatchV2:
        event = LethalDamageContributionEventV2(
            event_id=_event_id(batch, 0),
            transition_id=batch.transition_id,
            ordinal=0,
            source_global_slot=appearing_source.global_slot,
            recipient_global_slot=recipient.global_slot,
            attributed_death_damage=value,
            source_anchor=appearing_source.successor,
            recipient_anchor=recipient.successor,
        )
        return _with_events(batch, (event,))

    adapter = TypeAdapter(AgentPovVisualIncomingSummaryV1)
    low = _build(
        agent_visual_case,
        batch_with_attribution(1.0),
        transition_start_scene=start_without_source,
    )
    high = _build(
        agent_visual_case,
        batch_with_attribution(999.0),
        transition_start_scene=start_without_source,
    )

    assert low.events == ()
    assert adapter.dump_json(high) == adapter.dump_json(low)


def test_agent_visual_hidden_only_mutations_do_not_change_output(
    agent_visual_case: _AgentVisualCase,
) -> None:
    batch = agent_visual_case.batch
    visible_only = _with_events(
        batch,
        (_recipient_health_event(batch, ordinal=0),),
    )
    hidden_source = _trajectory(batch, 5)
    hidden_recipient = _trajectory(batch, 6)
    with_hidden_event = _with_events(
        batch,
        (
            AbilityActivatedEventV2(
                event_id=_event_id(batch, 0),
                transition_id=batch.transition_id,
                ordinal=0,
                source_global_slot=5,
                ability_component="ultimate",
                recipient_global_slot=6,
                source_anchor=hidden_source.transition_start,
                recipient_anchor=hidden_recipient.transition_start,
            ),
            _recipient_health_event(batch, ordinal=1),
        ),
    )
    moved_hidden = replace(
        hidden_source,
        transition_start=replace(
            hidden_source.transition_start,
            position=(17.0, 1.0),
        ),
        post_charge=replace(
            hidden_source.post_charge,
            position=(17.0, 1.0),
        ),
        successor=replace(
            hidden_source.successor,
            position=(17.0, 1.0),
        ),
    )
    moved_rows = tuple(
        moved_hidden if row.global_slot == 5 else row
        for row in visible_only.agent_phase_trajectories
    )
    with_hidden_trajectory_mutation = replace(
        visible_only,
        agent_phase_trajectories=moved_rows,
    )

    expected = _build(agent_visual_case, visible_only)
    assert _build(agent_visual_case, with_hidden_event) == expected
    assert _build(agent_visual_case, with_hidden_trajectory_mutation) == expected


def test_agent_visual_start_visible_ability_survives_successor_disappearance(
    agent_visual_case: _AgentVisualCase,
) -> None:
    batch = agent_visual_case.batch
    disappearing = _trajectory(batch, 1)
    recipient = _trajectory(batch, 0)
    successor_without_source = _without_agent(
        agent_visual_case.successor_scene,
        public_agent_id=disappearing.public_agent_id,
    )
    event = AbilityActivatedEventV2(
        event_id=_event_id(batch, 0),
        transition_id=batch.transition_id,
        ordinal=0,
        source_global_slot=disappearing.global_slot,
        ability_component="basic",
        recipient_global_slot=recipient.global_slot,
        source_anchor=disappearing.transition_start,
        recipient_anchor=recipient.transition_start,
    )

    summary = _build(
        agent_visual_case,
        _with_events(batch, (event,)),
        successor_scene=successor_without_source,
    )

    disappearing_row = next(
        row
        for row in summary.agent_phase_trajectories
        if row.agent_public_agent_id == disappearing.public_agent_id
    )
    assert disappearing_row.transition_start is not None
    assert disappearing_row.successor is None
    assert disappearing_row.agent_class_id == 2
    assert tuple(type(row) for row in summary.events) == (
        ReplayIncomingAbilityActivatedEventV1,
    )


def test_agent_visual_successor_only_status_and_respawn_survive_appearance(
    agent_visual_case: _AgentVisualCase,
) -> None:
    batch = agent_visual_case.batch
    source = _trajectory(batch, 0)
    appearing = _trajectory(batch, 1)
    start_without_recipient = _without_agent(
        agent_visual_case.transition_start_scene,
        public_agent_id=appearing.public_agent_id,
    )
    events: tuple[VisualEventV2, ...] = (
        StatusAppliedEventV2(
            event_id=_event_id(batch, 0),
            transition_id=batch.transition_id,
            ordinal=0,
            recipient_global_slot=appearing.global_slot,
            status_channel=7,
            status_id="mage_burst_damage_amplification",
            recipient_anchor=appearing.successor,
            source_global_slot=source.global_slot,
            source_anchor=source.successor,
        ),
        AgentRespawnedEventV2(
            event_id=_event_id(batch, 1),
            transition_id=batch.transition_id,
            ordinal=1,
            agent_global_slot=appearing.global_slot,
            team_id=1,
            realized_successor_position=appearing.successor.position,
            agent_anchor=appearing.successor,
        ),
    )

    summary = _build(
        agent_visual_case,
        _with_events(batch, events),
        transition_start_scene=start_without_recipient,
    )

    assert tuple(
        row.agent_public_agent_id for row in summary.agent_phase_trajectories
    ) == ("agent-slot-0", "agent-slot-2", "agent-slot-1")
    appearing_row = summary.agent_phase_trajectories[-1]
    assert appearing_row.transition_start is None
    assert appearing_row.successor is not None
    assert tuple(type(row) for row in summary.events) == (
        ReplayIncomingStatusAppliedEventV1,
        AgentPovVisualIncomingAgentRespawnedEventV1,
    )
    assert tuple(row.ordinal for row in summary.events) == (0, 1)


def test_agent_visual_respawn_strips_unjoined_team_and_redundant_position(
    agent_visual_case: _AgentVisualCase,
) -> None:
    batch = agent_visual_case.batch
    appearing = _trajectory(batch, 1)
    start_without_agent = _without_agent(
        agent_visual_case.transition_start_scene,
        public_agent_id=appearing.public_agent_id,
    )

    def batch_with_team(team_id: int) -> VisualEventBatchV2:
        return _with_events(
            batch,
            (
                AgentRespawnedEventV2(
                    event_id=_event_id(batch, 0),
                    transition_id=batch.transition_id,
                    ordinal=0,
                    agent_global_slot=appearing.global_slot,
                    team_id=team_id,
                    realized_successor_position=appearing.successor.position,
                    agent_anchor=appearing.successor,
                ),
            ),
        )

    adapter = TypeAdapter(AgentPovVisualIncomingSummaryV1)
    team_one = _build(
        agent_visual_case,
        batch_with_team(1),
        transition_start_scene=start_without_agent,
    )
    team_two = _build(
        agent_visual_case,
        batch_with_team(2),
        transition_start_scene=start_without_agent,
    )

    assert type(team_one.events[0]) is AgentPovVisualIncomingAgentRespawnedEventV1
    assert adapter.dump_json(team_two) == adapter.dump_json(team_one)
    respawn_schema = adapter.json_schema()["$defs"][
        "AgentPovVisualIncomingAgentRespawnedEventV1"
    ]["properties"]
    assert set(respawn_schema) == {
        "event_id",
        "ordinal",
        "phase_rank",
        "event_kind",
        "agent_anchor",
    }
    assert "team_id" not in respawn_schema
    assert "realized_successor_position" not in respawn_schema


def test_agent_visual_health_result_requires_authorized_successor(
    agent_visual_case: _AgentVisualCase,
) -> None:
    batch = agent_visual_case.batch
    recipient = _trajectory(batch, 2)
    successor_without_recipient = _without_agent(
        agent_visual_case.successor_scene,
        public_agent_id=recipient.public_agent_id,
    )

    summary = _build(
        agent_visual_case,
        _with_events(batch, (_recipient_health_event(batch, ordinal=0),)),
        successor_scene=successor_without_recipient,
    )

    recipient_row = next(
        row
        for row in summary.agent_phase_trajectories
        if row.agent_public_agent_id == recipient.public_agent_id
    )
    assert recipient_row.transition_start is not None
    assert recipient_row.successor is None
    assert summary.events == ()
    assert summary.event_count == 0


def test_agent_visual_rejects_visible_scene_trajectory_mismatch(
    agent_visual_case: _AgentVisualCase,
) -> None:
    batch = agent_visual_case.batch
    visible = _trajectory(batch, 1)
    mismatched = replace(
        visible,
        successor=replace(
            visible.successor,
            position=(visible.successor.position[0] + 0.25, 3.75),
        ),
    )
    rows = tuple(
        mismatched if row.global_slot == 1 else row
        for row in batch.agent_phase_trajectories
    )
    invalid_for_scene = replace(batch, agent_phase_trajectories=rows)

    with pytest.raises(
        ValueError,
        match="successor_scene positions must join canonical visual trajectories",
    ):
        _build(agent_visual_case, invalid_for_scene)


def test_agent_visual_rejects_common_identity_class_drift(
    agent_visual_case: _AgentVisualCase,
) -> None:
    scene = agent_visual_case.successor_scene
    original = next(
        row for row in scene.agents if row.public_agent_id == "agent-slot-1"
    )
    changed = replace(
        original,
        class_id=5,
        class_name="Priest",
        aura_modifiers=(),
    )
    changed_scene = replace(
        scene,
        agents=tuple(
            changed if row.public_agent_id == original.public_agent_id else row
            for row in scene.agents
        ),
        aura_fields=tuple(
            row
            for row in scene.aura_fields
            if row.source_public_agent_id != original.public_agent_id
        ),
        class_mechanics=tuple(
            row for row in scene.class_mechanics if row.class_id != original.class_id
        ),
    )

    with pytest.raises(
        ValueError,
        match="visual class identity changed within one transition",
    ):
        _build(
            agent_visual_case,
            agent_visual_case.batch,
            successor_scene=changed_scene,
        )


@pytest.mark.parametrize(
    "namespace",
    ("actor-pov", "shared-obs-visual-union"),
)
def test_agent_visual_accepts_existing_recipient_local_epoch_namespaces(
    agent_visual_case: _AgentVisualCase,
    namespace: str,
) -> None:
    summary = _build(
        agent_visual_case,
        agent_visual_case.batch,
        namespace=namespace,
    )
    assert f":{namespace}:" in summary.incoming_recipient_transition_id
    assert agent_visual_case.batch.transition_id not in summary.ordered_event_ids
