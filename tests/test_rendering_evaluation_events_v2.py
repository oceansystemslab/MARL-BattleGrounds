"""Focused proof for canonical V2 researcher events and status evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from importlib import import_module
from typing import Protocol, cast

import pytest
from scripts.dev.visual_debugger.control import create_session, submit_next_script_frame
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    mage_target_none_ultimate_action,
    neutral_action,
)
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.types import EnvConfig, EnvState
from marl_battlegrounds.evaluation.events import decode_evaluation_events_v1
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    AbilityActivatedEventV1,
    ActionRejectedEventV1,
    AgentDiedEventV1,
    AgentRespawnedEventV1,
    ChargePhaseDisplacementEventV1,
    CombatCountdownResetEventV1,
    CooldownReadyEventV1,
    CooldownStartedEventV1,
    EvaluationEventBaseV1,
    EvaluationEventV1,
    HealthRegeneratedEventV1,
    LethalDamageContributionEventV1,
    OrdinaryMovementPhaseDisplacementEventV1,
    RecipientHealthResolutionEventV1,
    RespawnWaveOccurredEventV1,
    SourceDamageOutputEventV1,
    SourceHealingOutputEventV1,
    SpawnShieldExpiredEventV1,
    StatusAgedToZeroEventV1,
    StatusAppliedEventV1,
    StatusBrokenByDamageEventV1,
    StatusClearedByNewDeathEventV1,
    StatusRefreshedOrExtendedEventV1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    _project_visual_event_v2,  # pyright: ignore[reportPrivateUsage]
    advance_status_source_evidence_v2,
    build_researcher_analyzer_projection_v2,
    build_status_source_evidence_index_v2,
    build_visual_event_batch_v2,
    initialize_status_source_evidence_v2,
)
from marl_battlegrounds.rendering.scene import (
    EVENT_V2_SCHEMA_VERSION,
    ActionRejectedEventV2,
    OrdinaryMovementPhaseDisplacementEventV2,
    VisualAgentAnchorV2,
    VisualAgentPhaseTrajectoryV2,
    VisualEventBatchV2,
)
from marl_battlegrounds.rendering.scene_geometry import render_scene_geometry

_ALL_EVENT_TYPES = (
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


class _ArtistLike(Protocol):
    def get_gid(self) -> str | None: ...


class _AxesLike(Protocol):
    patches: list[_ArtistLike]
    lines: list[_ArtistLike]
    texts: list[_ArtistLike]


def _replace_item[T](values: tuple[T, ...], index: int, value: T) -> tuple[T, ...]:
    return (*values[:index], value, *values[index + 1 :])


def _view_for_transition(
    trajectory: CapturedEvaluationTrajectory,
    transition_index: int,
) -> EvaluationTransitionViewV1:
    return EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[transition_index],
        transition=trajectory.transitions[transition_index],
        successor_frame=trajectory.frames[transition_index + 1],
    )


def _phase_trajectory(
    global_slot: int,
    public_agent_id: str,
    *,
    transition_start: tuple[float, float],
    post_charge: tuple[float, float],
    successor: tuple[float, float],
) -> VisualAgentPhaseTrajectoryV2:
    return VisualAgentPhaseTrajectoryV2(
        global_slot=global_slot,
        public_agent_id=public_agent_id,
        transition_start=VisualAgentAnchorV2(
            phase="transition_start",
            global_slot=global_slot,
            public_agent_id=public_agent_id,
            position=transition_start,
        ),
        post_charge=VisualAgentAnchorV2(
            phase="post_charge",
            global_slot=global_slot,
            public_agent_id=public_agent_id,
            position=post_charge,
        ),
        successor=VisualAgentAnchorV2(
            phase="successor",
            global_slot=global_slot,
            public_agent_id=public_agent_id,
            position=successor,
        ),
    )


def _canonical_event(
    event_type: type[EvaluationEventBaseV1],
    ordinal: int,
    **payload: object,
) -> EvaluationEventV1:
    transition_id = "episode-001:transition:0"
    return cast(
        EvaluationEventV1,
        event_type.model_validate(
            {
                "transition_id": transition_id,
                "ordinal": ordinal,
                "event_id": f"{transition_id}:event:{ordinal:04d}",
                **payload,
            }
        ),
    )


def _all_canonical_events() -> tuple[EvaluationEventV1, ...]:
    specs: tuple[tuple[type[EvaluationEventBaseV1], dict[str, object]], ...] = (
        (
            ActionRejectedEventV1,
            {
                "actor_global_slot": 3,
                "rejection_component": "domain",
                "submitted_move_action": 99,
                "submitted_select_target_action": -1,
                "submitted_use_ultimate_action": 2,
            },
        ),
        (
            AbilityActivatedEventV1,
            {
                "source_global_slot": 0,
                "ability_component": "ultimate",
                "recipient_global_slot": None,
            },
        ),
        (
            SourceDamageOutputEventV1,
            {
                "source_global_slot": 1,
                "recipient_global_slot": 5,
                "raw_damage_output": 12.5,
                "source_modified_damage_output": 13.0,
                "recipient_damage_modifier": 0.8,
                "mage_damage_aura_covering_emitter_global_slots": (0,),
                "warrior_mitigation_aura_covering_emitter_global_slots": (1,),
            },
        ),
        (
            SourceHealingOutputEventV1,
            {
                "source_global_slot": 2,
                "recipient_global_slot": 1,
                "raw_healing_output": 8.0,
                "source_modified_healing_output": 9.0,
                "recipient_healing_modifier": 1.0,
            },
        ),
        (
            RecipientHealthResolutionEventV1,
            {
                "recipient_global_slot": 5,
                "transition_start_health": 20.0,
                "total_effective_damage": 12.0,
                "total_effective_healing": 2.0,
                "health_after_combat_resolution": 10.0,
                "realized_net_health_change": -10.0,
            },
        ),
        (CombatCountdownResetEventV1, {"agent_global_slot": 1}),
        (
            HealthRegeneratedEventV1,
            {"agent_global_slot": 2, "actual_health_regenerated": 1.25},
        ),
        (CooldownStartedEventV1, {"agent_global_slot": 0}),
        (CooldownReadyEventV1, {"agent_global_slot": 1}),
        (
            ChargePhaseDisplacementEventV1,
            {"agent_global_slot": 1, "realized_displacement": (0.5, 0.0)},
        ),
        (
            OrdinaryMovementPhaseDisplacementEventV1,
            {"agent_global_slot": 0, "realized_displacement": (0.25, 0.0)},
        ),
        (AgentDiedEventV1, {"recipient_global_slot": 5}),
        (
            LethalDamageContributionEventV1,
            {
                "source_global_slot": 1,
                "recipient_global_slot": 5,
                "attributed_death_damage": 12.0,
            },
        ),
        (
            StatusAgedToZeroEventV1,
            {
                "recipient_global_slot": 5,
                "status_channel": 0,
                "status_id": "warrior_charge_slow",
            },
        ),
        (
            StatusBrokenByDamageEventV1,
            {
                "recipient_global_slot": 5,
                "status_channel": 3,
                "status_id": "warrior_charge_stun",
            },
        ),
        (
            StatusAppliedEventV1,
            {
                "recipient_global_slot": 5,
                "status_channel": 6,
                "status_id": "rogue_poison_anti_heal",
                "source_global_slot": 2,
            },
        ),
        (
            StatusRefreshedOrExtendedEventV1,
            {
                "recipient_global_slot": 5,
                "status_channel": 7,
                "status_id": "mage_burst_damage_amplification",
            },
        ),
        (
            StatusClearedByNewDeathEventV1,
            {
                "recipient_global_slot": 5,
                "status_channel": 8,
                "status_id": "priest_blessing_of_freedom_movement_floor",
            },
        ),
        (SpawnShieldExpiredEventV1, {"agent_global_slot": 2}),
        (RespawnWaveOccurredEventV1, {"team_index": 0, "team_id": 1}),
        (
            AgentRespawnedEventV1,
            {
                "agent_global_slot": 6,
                "team_id": 2,
                "realized_successor_position": (6.0, 1.0),
            },
        ),
    )
    return tuple(
        _canonical_event(event_type, ordinal, **payload)
        for ordinal, (event_type, payload) in enumerate(specs)
    )


def _all_event_batch() -> tuple[VisualEventBatchV2, tuple[EvaluationEventV1, ...]]:
    public_ids = tuple(f"agent-slot-{slot}" for slot in range(10))
    active = (True, True, True, False, False, True, True, False, False, False)
    trajectories = (
        _phase_trajectory(
            0,
            public_ids[0],
            transition_start=(0.0, 0.0),
            post_charge=(0.0, 0.0),
            successor=(0.25, 0.0),
        ),
        _phase_trajectory(
            1,
            public_ids[1],
            transition_start=(1.0, 0.0),
            post_charge=(1.5, 0.0),
            successor=(1.5, 0.0),
        ),
        _phase_trajectory(
            2,
            public_ids[2],
            transition_start=(2.0, 0.0),
            post_charge=(2.0, 0.0),
            successor=(2.0, 0.0),
        ),
        _phase_trajectory(
            5,
            public_ids[5],
            transition_start=(5.0, 0.0),
            post_charge=(5.0, 0.0),
            successor=(5.0, 0.0),
        ),
        _phase_trajectory(
            6,
            public_ids[6],
            transition_start=(6.0, 0.0),
            post_charge=(6.0, 0.0),
            successor=(6.0, 1.0),
        ),
    )
    trajectory_by_slot = {row.global_slot: row for row in trajectories}
    source_events = _all_canonical_events()
    projected = tuple(
        _project_visual_event_v2(
            event,
            trajectory_by_slot=trajectory_by_slot,
            public_agent_id_by_global_slot=public_ids,
            configured_active_by_global_slot=active,
        )
        for event in source_events
    )
    return (
        VisualEventBatchV2(
            schema_version=EVENT_V2_SCHEMA_VERSION,
            episode_id="episode-001",
            transition_index=0,
            transition_id="episode-001:transition:0",
            start_frame_id="episode-001:frame:0",
            successor_frame_id="episode-001:frame:1",
            start_simulator_step_count=7,
            successor_simulator_step_count=8,
            public_agent_id_by_global_slot=public_ids,
            configured_active_by_global_slot=active,
            agent_phase_trajectories=trajectories,
            events=projected,
        ),
        source_events,
    )


def test_all_21_canonical_events_project_independently_with_direct_payloads() -> None:
    batch, source_events = _all_event_batch()

    assert tuple(event.event_type for event in batch.events) == _ALL_EVENT_TYPES
    assert tuple(event.event_id for event in batch.events) == tuple(
        event.event_id for event in source_events
    )
    assert len({type(event) for event in batch.events}) == 21
    for source, projected in zip(source_events, batch.events, strict=True):
        for field_name in type(source).model_fields:
            if field_name in {"schema_id", "schema_version"}:
                continue
            assert getattr(projected, field_name) == getattr(source, field_name)

    inactive_rejection = cast(ActionRejectedEventV2, batch.events[0])
    assert inactive_rejection.actor_global_slot == 3
    assert inactive_rejection.actor_public_agent_id == "agent-slot-3"
    assert not inactive_rejection.actor_configured_active
    assert inactive_rejection.actor_anchor is None


def test_phase_displacement_anchors_accept_float32_rounding_only() -> None:
    start = VisualAgentAnchorV2(
        phase="post_charge",
        global_slot=0,
        public_agent_id="agent-slot-0",
        position=(0.10000000149011612, -0.20000000298023224),
    )
    authoritative_successor = VisualAgentAnchorV2(
        phase="successor",
        global_slot=0,
        public_agent_id="agent-slot-0",
        position=(0.30000001192092896, -0.10000000149011612),
    )
    event = OrdinaryMovementPhaseDisplacementEventV2(
        event_id="episode-001:transition:0:event:0000",
        transition_id="episode-001:transition:0",
        ordinal=0,
        agent_global_slot=0,
        realized_displacement=(0.20000000298023224, 0.10000000149011612),
        start_anchor=start,
        end_anchor=authoritative_successor,
    )
    assert event.end_anchor is authoritative_successor

    wrong_end = replace(authoritative_successor, position=(0.31, -0.1))
    with pytest.raises(ValueError, match="recorded displacement"):
        replace(event, end_anchor=wrong_end)


def test_valid_inactive_rejection_is_feed_only_and_retains_exact_identity() -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
    )
    original = trajectory.transitions[0]
    acceptance = original.facts.action_acceptance_facts
    inactive_rejections = _replace_item(
        acceptance.submitted_action_tuple_is_out_of_domain_by_actor,
        3,
        True,
    )
    facts = original.facts.model_copy(
        update={
            "action_acceptance_facts": acceptance.model_copy(
                update={
                    "submitted_action_tuple_is_out_of_domain_by_actor": (
                        inactive_rejections
                    )
                }
            )
        }
    )
    events = decode_evaluation_events_v1(
        trajectory.context,
        trajectory.frames[0],
        facts,
        trajectory.frames[1],
    )
    transition = original.model_copy(update={"facts": facts, "events": events})
    view = EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[0],
        transition=transition,
        successor_frame=trajectory.frames[1],
    )

    batch = build_visual_event_batch_v2(view)
    state = advance_status_source_evidence_v2(
        initialize_status_source_evidence_v2(
            trajectory.context,
            trajectory.frames[0],
        ),
        view,
    )
    projection = build_researcher_analyzer_projection_v2(
        trajectory.context,
        trajectory.frames[1],
        transition_view=view,
        status_source_evidence_state=state,
    )

    assert projection.incoming_events == batch
    assert tuple(agent.global_slot for agent in projection.scene.agents) == (
        0,
        1,
        2,
        5,
        6,
    )
    assert len(batch.events) == 1
    event = cast(ActionRejectedEventV2, batch.events[0])
    assert event.event_id == transition.events[0].event_id
    assert event.actor_public_agent_id == "agent-slot-3"
    assert event.actor_anchor is None


def test_event_batch_requires_adjacent_exact_simulator_epochs() -> None:
    batch, _ = _all_event_batch()
    with pytest.raises(ValueError, match="adjacent simulator step"):
        replace(batch, successor_simulator_step_count=9)
    with pytest.raises(ValueError, match="Python int"):
        replace(batch, start_simulator_step_count=True)


def test_status_source_evidence_live_and_replay_reduction_are_identical() -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=2,
        expected_horizon=2,
        actions=(mage_target_none_ultimate_action(), neutral_action()),
    )
    initial = initialize_status_source_evidence_v2(
        trajectory.context,
        trajectory.frames[0],
    )
    first = advance_status_source_evidence_v2(
        initial, _view_for_transition(trajectory, 0)
    )
    second = advance_status_source_evidence_v2(
        first, _view_for_transition(trajectory, 1)
    )
    index = build_status_source_evidence_index_v2(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
    )

    assert index.frame_states == (initial, first, second)
    direct_rows = tuple(
        row for row in first.active_statuses if row.direct_source_evidence
    )
    assert direct_rows
    assert all(
        evidence.source_public_agent_id == "agent-slot-0"
        for row in direct_rows
        for evidence in row.direct_source_evidence
    )
    assert tuple(row.direct_source_evidence for row in second.active_statuses) == tuple(
        row.direct_source_evidence for row in first.active_statuses
    )
    forged_evidence = replace(
        direct_rows[0].direct_source_evidence[0],
        event_id="episode-001:transition:1:event:0000",
    )
    forged_row = replace(
        direct_rows[0],
        direct_source_evidence=(forged_evidence,),
    )
    forged_rows = tuple(
        forged_row if row is direct_rows[0] else row for row in first.active_statuses
    )
    with pytest.raises(ValueError, match="before the bound frame"):
        replace(first, active_statuses=forged_rows)

    projection = build_researcher_analyzer_projection_v2(
        trajectory.context,
        trajectory.frames[2],
        transition_view=_view_for_transition(trajectory, 1),
        status_source_evidence_state=index.state_for_frame(2),
    )
    assert projection.status_source_evidence == second
    with pytest.raises(ValueError, match="require frame-bound"):
        build_researcher_analyzer_projection_v2(
            trajectory.context,
            trajectory.frames[2],
            transition_view=_view_for_transition(trajectory, 1),
        )


def test_source_less_status_refresh_clears_durable_source_agent_evidence() -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=2,
        expected_horizon=2,
        actions=(mage_target_none_ultimate_action(), neutral_action()),
    )
    initial = initialize_status_source_evidence_v2(
        trajectory.context,
        trajectory.frames[0],
    )
    first = advance_status_source_evidence_v2(
        initial,
        _view_for_transition(trajectory, 0),
    )
    directly_sourced = next(
        row for row in first.active_statuses if row.direct_source_evidence
    )
    original = trajectory.transitions[1]
    lifecycle = original.facts.status_lifecycle_facts
    refreshed = _replace_item(
        lifecycle.refreshed_or_extended_by_recipient_and_status_channel,
        directly_sourced.recipient_global_slot,
        _replace_item(
            lifecycle.refreshed_or_extended_by_recipient_and_status_channel[
                directly_sourced.recipient_global_slot
            ],
            directly_sourced.status_channel,
            True,
        ),
    )
    facts = original.facts.model_copy(
        update={
            "status_lifecycle_facts": lifecycle.model_copy(
                update={
                    "refreshed_or_extended_by_recipient_and_status_channel": (refreshed)
                }
            )
        }
    )
    events = decode_evaluation_events_v1(
        trajectory.context,
        trajectory.frames[1],
        facts,
        trajectory.frames[2],
    )
    transition = original.model_copy(update={"facts": facts, "events": events})
    refresh_view = EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[1],
        transition=transition,
        successor_frame=trajectory.frames[2],
    )

    refreshed_state = advance_status_source_evidence_v2(first, refresh_view)
    refreshed_row = next(
        row
        for row in refreshed_state.active_statuses
        if (
            row.recipient_global_slot == directly_sourced.recipient_global_slot
            and row.status_channel == directly_sourced.status_channel
        )
    )
    assert tuple(event.event_type for event in events) == (
        "status_refreshed_or_extended",
    )
    assert refreshed_row.direct_source_evidence == ()
    assert directly_sourced.direct_source_evidence


def test_initial_status_is_unknown_and_death_clear_removes_source_evidence() -> None:
    base_scenario = get_scenario("status_stack")

    def build_low_health_status_scenario() -> tuple[EnvConfig, EnvState]:
        config, state = base_scenario.build_scenario()
        return config, state._replace(
            current_health=state.current_health.at[5].set(1.0),
            slow_durations=state.slow_durations.at[5, 0].set(3),
        )

    session = create_session(
        replace(base_scenario, build_scenario=build_low_health_status_scenario),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(0),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    initial = session.status_source_evidence_state
    initial_status = next(
        row
        for row in initial.active_statuses
        if (row.recipient_global_slot, row.status_channel) == (5, 0)
    )
    assert initial_status.direct_source_evidence == ()

    session = submit_next_script_frame(session)
    incoming = session.incoming_evaluation_view
    assert incoming is not None
    cleared_keys = {
        (event.recipient_global_slot, event.status_channel)
        for event in incoming.transition.events
        if event.event_type == "status_cleared_by_new_death"
    }
    assert (5, 0) in cleared_keys
    assert all(
        (row.recipient_global_slot, row.status_channel) not in cleared_keys
        for row in session.status_source_evidence_state.active_statuses
    )


def test_status_expiry_and_break_remove_prior_direct_source_evidence() -> None:
    expiring = captured_evaluation_trajectory(
        transition_count=6,
        expected_horizon=6,
        actions=(
            mage_target_none_ultimate_action(),
            *(neutral_action() for _ in range(5)),
        ),
    )
    expiry_index = build_status_source_evidence_index_v2(
        expiring.context,
        expiring.frames,
        expiring.transitions,
    )
    assert any(
        event.event_type == "status_aged_to_zero"
        for event in expiring.transitions[-1].events
    )
    assert expiry_index.state_for_frame(5).active_statuses
    assert expiry_index.state_for_frame(6).active_statuses == ()

    session = create_session(
        get_scenario("status_stack"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(0),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    session = submit_next_script_frame(session)
    before = session.status_source_evidence_state
    session = submit_next_script_frame(session)
    incoming = session.incoming_evaluation_view
    assert incoming is not None
    applied_keys = {
        (applied.recipient_global_slot, applied.status_channel)
        for event in incoming.transition.events
        if type(event) is StatusAppliedEventV1
        for applied in (event,)
    }
    removed_keys = {
        (removed.recipient_global_slot, removed.status_channel)
        for event in incoming.transition.events
        if type(event) in (StatusAgedToZeroEventV1, StatusBrokenByDamageEventV1)
        for removed in (
            cast(StatusAgedToZeroEventV1 | StatusBrokenByDamageEventV1, event),
        )
    } - applied_keys
    assert removed_keys
    assert any(
        event.event_type == "status_broken_by_damage"
        for event in incoming.transition.events
    )
    assert any(
        (row.recipient_global_slot, row.status_channel) in removed_keys
        and row.direct_source_evidence
        for row in before.active_statuses
    )
    assert all(
        (row.recipient_global_slot, row.status_channel) not in removed_keys
        for row in session.status_source_evidence_state.active_statuses
    )


def test_empty_respawn_wave_remains_an_independent_visual_event() -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=5,
        expected_horizon=5,
    )
    view = _view_for_transition(trajectory, 4)
    assert tuple(event.event_type for event in view.transition.events) == (
        "respawn_wave_occurred",
    )

    batch = build_visual_event_batch_v2(view)
    assert tuple(event.event_type for event in batch.events) == (
        "respawn_wave_occurred",
    )
    assert not any(event.event_type == "agent_respawned" for event in batch.events)


def test_static_v2_renderer_draws_each_canonical_event_exactly_once() -> None:
    pytest.importorskip("matplotlib.pyplot")
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        actions=(mage_target_none_ultimate_action(),),
    )
    view = _view_for_transition(trajectory, 0)
    state = advance_status_source_evidence_v2(
        initialize_status_source_evidence_v2(
            trajectory.context,
            trajectory.frames[0],
        ),
        view,
    )
    projection = build_researcher_analyzer_projection_v2(
        trajectory.context,
        trajectory.frames[1],
        transition_view=view,
        status_source_evidence_state=state,
    )
    assert projection.incoming_events is not None

    result = render_scene_geometry(
        projection.scene,
        event_batch=projection.incoming_events,
    )
    try:
        axes = cast(_AxesLike, result.axes)
        artists = (*axes.patches, *axes.lines, *axes.texts)
        event_gids = tuple(
            artist.get_gid()
            for artist in artists
            if (artist.get_gid() or "").startswith("scene:v2:event:")
        )
        assert event_gids == tuple(
            f"scene:v2:event:{event.event_id}"
            for event in projection.incoming_events.events
        )
    finally:
        pyplot = import_module("matplotlib.pyplot")
        cast(Callable[[object], object], pyplot.close)(result.figure)
