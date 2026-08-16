"""Focused contract tests for deterministic evaluation-event decoding."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.evaluation_fixtures import evaluation_context, evaluation_env_config

from marl_battlegrounds.evaluation.events import decode_evaluation_events_v1
from marl_battlegrounds.evaluation.models import (
    ActionAcceptanceFactsV1,
    ActionMaskV1,
    AgentRespawnedEventV1,
    AuraTransitionFactsV1,
    BaseObservationV1,
    CombatTransitionFactsV1,
    CooldownReadyEventV1,
    DeathTransitionFactsV1,
    EvaluationEpisodeContextV1,
    EvaluationEventV1,
    EvaluationFrameV1,
    EvaluationTransitionV1,
    GlobalAnalysisSnapshotV1,
    JointActionV1,
    PhysicalTransitionFactsV1,
    RecipientHealthResolutionEventV1,
    RegenerationTransitionFactsV1,
    RespawnTransitionFactsV1,
    SourceDamageOutputEventV1,
    SpawnShieldTransitionFactsV1,
    StatusAppliedEventV1,
    StatusLifecycleTransitionFactsV1,
    TeamDeathmatchCompletedEventV1,
    TeamDeathmatchScoreChangedEventV1,
    TeamDeathmatchTransitionFactsV1,
    TransitionFactsV1,
)

_FALSE_10 = (False,) * 10
_FALSE_2 = (False,) * 2
_ZERO_INT_10 = (0,) * 10
_ZERO_FLOAT_10 = (0.0,) * 10
_HEALTH_100 = (100.0,) * 10
_NONE_10: tuple[int | None, ...] = (None,) * 10
_FALSE_MATRIX_10 = (_FALSE_10,) * 10
_FALSE_STATUS_MATRIX = ((False,) * 9,) * 10
_FALSE_THREE_CHANNEL_MATRIX = ((False,) * 3,) * 10
_ZERO_DISPLACEMENT_MATRIX = ((0.0, 0.0),) * 10
_EVENT_ADAPTER: TypeAdapter[EvaluationEventV1] = TypeAdapter(EvaluationEventV1)

_ALL_EVENT_TYPES = {
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
}


def _replace_item[T](values: tuple[T, ...], index: int, value: T) -> tuple[T, ...]:
    """Return one tuple with a single test value replaced."""
    return (*values[:index], value, *values[index + 1 :])


def _replace_matrix_item[T](
    values: tuple[tuple[T, ...], ...],
    row: int,
    column: int,
    value: T,
) -> tuple[tuple[T, ...], ...]:
    """Return one fixed matrix with a single test value replaced."""
    return _replace_item(
        values,
        row,
        _replace_item(values[row], column, value),
    )


def _snapshot(
    *,
    alive_mask: tuple[bool, ...] = (True,) * 10,
    current_health: tuple[float, ...] = _HEALTH_100,
    ultimate_cooldowns: tuple[int, ...] = _ZERO_INT_10,
    agent_positions: tuple[tuple[float, float], ...] | None = None,
    team_deathmatch_scores: tuple[int, int] = (0, 0),
) -> GlobalAnalysisSnapshotV1:
    """Build the validated dynamic subset consumed by the event decoder."""
    if agent_positions is None:
        agent_positions = tuple((float(slot), 0.0) for slot in range(10))
    return GlobalAnalysisSnapshotV1(
        team_deathmatch_scores=team_deathmatch_scores,
        alive_mask=alive_mask,
        agent_positions=agent_positions,
        current_health=current_health,
        ultimate_cooldowns=ultimate_cooldowns,
        slow_durations=((0, 0, 0),) * 10,
        stun_durations=((0, 0, 0),) * 10,
        rogue_poison_anti_heal_durations=_ZERO_INT_10,
        mage_burst_damage_amplification_durations=_ZERO_INT_10,
        priest_blessing_of_freedom_slow_floor_durations=_ZERO_INT_10,
        team_respawn_wave_countdowns=(1, 1),
        spawn_shield_durations=_ZERO_INT_10,
        steps_until_out_of_combat=_ZERO_INT_10,
        previous_timestep_move_actions=_ZERO_INT_10,
        previous_timestep_select_target_actions=_ZERO_INT_10,
        previous_timestep_use_ultimate_actions=_ZERO_INT_10,
        has_previous_timestep_joint_action=False,
    )


def _frame(
    frame_index: int,
    simulator_step_count: int,
    snapshot: GlobalAnalysisSnapshotV1,
) -> EvaluationFrameV1:
    """Build a decoder frame while omitting irrelevant validated tensor payloads."""
    return EvaluationFrameV1.model_construct(
        episode_id="episode-001",
        frame_index=frame_index,
        frame_id=f"episode-001:frame:{frame_index}",
        simulator_step_count=simulator_step_count,
        snapshot=snapshot,
        base_observation=BaseObservationV1.model_construct(),
        action_mask=ActionMaskV1.model_construct(),
        shared_obs_information_availability_by_recipient_and_sensor_source=None,
    )


def _joint_action(
    *,
    move: tuple[int, ...] = _ZERO_INT_10,
    select_target: tuple[int, ...] = _ZERO_INT_10,
    use_ultimate: tuple[int, ...] = _ZERO_INT_10,
) -> JointActionV1:
    """Build one fixed-width host joint action."""
    return JointActionV1(
        move=move,
        select_target=select_target,
        use_ultimate=use_ultimate,
    )


def _neutral_facts(*, transition_start_step_count: int = 7) -> TransitionFactsV1:
    """Build real-transition facts whose event view is empty."""
    neutral_action = _joint_action()
    return TransitionFactsV1(
        has_transition=True,
        transition_start_step_count=transition_start_step_count,
        action_acceptance_facts=ActionAcceptanceFactsV1(
            submitted_joint_action=neutral_action,
            accepted_joint_action=neutral_action,
            submitted_action_tuple_is_out_of_domain_by_actor=_FALSE_10,
            in_domain_move_action_is_rejected_by_actor=_FALSE_10,
            in_domain_combat_action_pair_is_rejected_by_actor=_FALSE_10,
        ),
        combat_transition_facts=CombatTransitionFactsV1(
            basic_effect_is_activated_by_source=_FALSE_10,
            ultimate_effect_is_activated_by_source=_FALSE_10,
            combat_effect_has_recipient_by_source=_FALSE_10,
            combat_effect_recipient_global_slot_by_source=_NONE_10,
            raw_damage_output_by_source=_ZERO_FLOAT_10,
            source_modified_damage_output_by_source=_ZERO_FLOAT_10,
            recipient_damage_modifier_by_source=_ZERO_FLOAT_10,
            total_effective_damage_by_recipient=_ZERO_FLOAT_10,
            raw_healing_output_by_source=_ZERO_FLOAT_10,
            source_modified_healing_output_by_source=_ZERO_FLOAT_10,
            recipient_healing_modifier_by_source=_ZERO_FLOAT_10,
            total_effective_healing_by_recipient=_ZERO_FLOAT_10,
            health_after_combat_resolution_by_recipient=_HEALTH_100,
            slow_is_applied_by_source_and_channel=_FALSE_THREE_CHANNEL_MATRIX,
            stun_is_applied_by_source_and_channel=_FALSE_THREE_CHANNEL_MATRIX,
            rogue_poison_anti_heal_is_applied_by_source=_FALSE_10,
            mage_burst_damage_amplification_is_applied_by_source=_FALSE_10,
            priest_blessing_of_freedom_is_applied_by_source=_FALSE_10,
        ),
        death_facts=DeathTransitionFactsV1(
            is_newly_dead_by_recipient=_FALSE_10,
            contributed_to_new_death_by_source=_FALSE_10,
            attributed_death_damage_by_source=_ZERO_FLOAT_10,
        ),
        spawn_shield_facts=SpawnShieldTransitionFactsV1(
            was_active_at_transition_start_by_agent=_FALSE_10,
            expired_at_transition_end_by_agent=_FALSE_10,
        ),
        respawn_facts=RespawnTransitionFactsV1(
            respawn_wave_occurred_this_transition_by_team=_FALSE_2,
            was_respawned_this_transition_by_agent=_FALSE_10,
        ),
        regeneration_facts=RegenerationTransitionFactsV1(
            combat_countdown_was_reset_by_agent=_FALSE_10,
            actual_health_regenerated_this_step_by_agent=_ZERO_FLOAT_10,
        ),
        physical_facts=PhysicalTransitionFactsV1(
            charge_phase_displacement_by_agent=_ZERO_DISPLACEMENT_MATRIX,
            ordinary_movement_phase_displacement_by_agent=(_ZERO_DISPLACEMENT_MATRIX),
        ),
        aura_facts=AuraTransitionFactsV1(
            is_covered_by_mage_damage_aura_by_emitter_and_beneficiary=(
                _FALSE_MATRIX_10
            ),
            is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary=(
                _FALSE_MATRIX_10
            ),
        ),
        status_lifecycle_facts=StatusLifecycleTransitionFactsV1(
            aged_to_zero_by_recipient_and_status_channel=_FALSE_STATUS_MATRIX,
            refreshed_or_extended_by_recipient_and_status_channel=(
                _FALSE_STATUS_MATRIX
            ),
            broken_by_damage_by_recipient_and_status_channel=(_FALSE_STATUS_MATRIX),
            cleared_by_new_death_by_recipient_and_status_channel=(_FALSE_STATUS_MATRIX),
        ),
        team_deathmatch_facts=TeamDeathmatchTransitionFactsV1(outcome=0),
    )


def _decode(
    facts: TransitionFactsV1,
    *,
    context: EvaluationEpisodeContextV1 | None = None,
    start_snapshot: GlobalAnalysisSnapshotV1 | None = None,
    successor_snapshot: GlobalAnalysisSnapshotV1 | None = None,
) -> tuple[EvaluationEventV1, ...]:
    """Decode one standard adjacent test transition."""
    if context is None:
        context = evaluation_context()
    if start_snapshot is None:
        start_snapshot = _snapshot()
    if successor_snapshot is None:
        successor_snapshot = _snapshot()
    return decode_evaluation_events_v1(
        context,
        _frame(3, 7, start_snapshot),
        facts,
        _frame(4, 8, successor_snapshot),
    )


def _event_types(events: Iterable[EvaluationEventV1]) -> tuple[str, ...]:
    """Return event discriminators in decoded order."""
    return tuple(event.event_type for event in events)


def _representative_multi_event_transition() -> EvaluationTransitionV1:
    """Build a transition whose discriminated event tuple has multiple variants."""
    facts = _neutral_facts()
    facts = facts.model_copy(
        update={
            "action_acceptance_facts": facts.action_acceptance_facts.model_copy(
                update={
                    "submitted_action_tuple_is_out_of_domain_by_actor": _replace_item(
                        _FALSE_10, 2, True
                    )
                }
            ),
            "physical_facts": facts.physical_facts.model_copy(
                update={
                    "charge_phase_displacement_by_agent": _replace_item(
                        _ZERO_DISPLACEMENT_MATRIX, 1, (0.5, -0.25)
                    )
                }
            ),
        }
    )
    events = _decode(facts)
    return EvaluationTransitionV1(
        episode_id="episode-001",
        transition_index=3,
        transition_id="episode-001:transition:3",
        start_frame_id="episode-001:frame:3",
        successor_frame_id="episode-001:frame:4",
        facts=facts,
        events=events,
        canonical_reward_by_agent=_ZERO_FLOAT_10,
        canonical_reward_by_team=None,
        terminated=False,
        truncated=False,
        owning_task_end_reason=None,
    )


def test_neutral_real_transition_decodes_to_no_events() -> None:
    assert _decode(_neutral_facts()) == ()


def test_event_union_contains_exactly_23_strict_variants() -> None:
    schema = _EVENT_ADAPTER.json_schema()

    assert len(schema["oneOf"]) == 23


def test_team_deathmatch_decodes_bilateral_score_edges_and_threshold_result() -> None:
    context = evaluation_context(
        config=evaluation_env_config(
            task_mode=1,
            team_deathmatch_score_threshold=3,
        )
    )
    newly_dead = _replace_item(_FALSE_10, 0, True)
    newly_dead = _replace_item(newly_dead, 5, True)
    newly_dead = _replace_item(newly_dead, 6, True)
    facts = _neutral_facts().model_copy(
        update={
            "death_facts": _neutral_facts().death_facts.model_copy(
                update={"is_newly_dead_by_recipient": newly_dead}
            ),
            "team_deathmatch_facts": TeamDeathmatchTransitionFactsV1(outcome=1),
        }
    )

    events = _decode(
        facts,
        context=context,
        start_snapshot=_snapshot(team_deathmatch_scores=(2, 2)),
        successor_snapshot=_snapshot(team_deathmatch_scores=(4, 3)),
    )

    task_events = tuple(
        event
        for event in events
        if isinstance(
            event,
            (TeamDeathmatchScoreChangedEventV1, TeamDeathmatchCompletedEventV1),
        )
    )
    assert task_events == (
        TeamDeathmatchScoreChangedEventV1(
            transition_id="episode-001:transition:3",
            ordinal=3,
            event_id="episode-001:transition:3:event:0003",
            team_index=0,
            team_id=1,
            score_increment=2,
            previous_score=2,
            successor_score=4,
        ),
        TeamDeathmatchScoreChangedEventV1(
            transition_id="episode-001:transition:3",
            ordinal=4,
            event_id="episode-001:transition:3:event:0004",
            team_index=1,
            team_id=2,
            score_increment=1,
            previous_score=2,
            successor_score=3,
        ),
        TeamDeathmatchCompletedEventV1(
            transition_id="episode-001:transition:3",
            ordinal=5,
            event_id="episode-001:transition:3:event:0005",
            outcome="team_a_win",
            completion_basis="score_threshold",
        ),
    )


@pytest.mark.parametrize(
    ("start_scores", "successor_scores", "outcome", "completion_basis"),
    (
        ((2, 1), (2, 1), 3, "horizon"),
        ((2, 2), (3, 3), 3, "score_threshold_at_horizon"),
    ),
)
def test_team_deathmatch_horizon_and_threshold_at_horizon_are_explicit(
    start_scores: tuple[int, int],
    successor_scores: tuple[int, int],
    outcome: int,
    completion_basis: str,
) -> None:
    threshold = 3 if completion_basis == "score_threshold_at_horizon" else 10
    context = evaluation_context(
        config=evaluation_env_config(
            task_mode=1,
            team_deathmatch_score_threshold=threshold,
            max_steps=8,
        ),
        expected_horizon=8,
    )
    newly_dead = _FALSE_10
    if successor_scores != start_scores:
        newly_dead = _replace_item(newly_dead, 0, True)
        newly_dead = _replace_item(newly_dead, 5, True)
    base_facts = _neutral_facts()
    facts = base_facts.model_copy(
        update={
            "death_facts": base_facts.death_facts.model_copy(
                update={"is_newly_dead_by_recipient": newly_dead}
            ),
            "team_deathmatch_facts": TeamDeathmatchTransitionFactsV1(outcome=outcome),
        }
    )

    events = _decode(
        facts,
        context=context,
        start_snapshot=_snapshot(team_deathmatch_scores=start_scores),
        successor_snapshot=_snapshot(team_deathmatch_scores=successor_scores),
    )

    completion = next(
        event for event in events if isinstance(event, TeamDeathmatchCompletedEventV1)
    )
    assert completion.outcome == "draw"
    assert completion.completion_basis == completion_basis


def test_team_deathmatch_decoder_rejects_score_edges_not_authored_by_deaths() -> None:
    context = evaluation_context(
        config=evaluation_env_config(
            task_mode=1,
            team_deathmatch_score_threshold=5,
        )
    )

    with pytest.raises(ValueError, match="score edges"):
        _decode(
            _neutral_facts(),
            context=context,
            start_snapshot=_snapshot(team_deathmatch_scores=(0, 0)),
            successor_snapshot=_snapshot(team_deathmatch_scores=(1, 0)),
        )


def test_team_deathmatch_score_event_rejects_incoherent_team_or_score_join() -> None:
    payload: dict[str, object] = {
        "transition_id": "episode-001:transition:0",
        "ordinal": 0,
        "event_id": "episode-001:transition:0:event:0000",
        "team_index": 0,
        "team_id": 1,
        "score_increment": 2,
        "previous_score": 3,
        "successor_score": 5,
    }

    assert (
        TeamDeathmatchScoreChangedEventV1.model_validate(payload).successor_score == 5
    )
    with pytest.raises(ValidationError, match="team_id"):
        TeamDeathmatchScoreChangedEventV1.model_validate({**payload, "team_id": 2})
    with pytest.raises(ValidationError, match="successor_score"):
        TeamDeathmatchScoreChangedEventV1.model_validate(
            {**payload, "successor_score": 4}
        )


def test_representative_multi_event_transition_round_trips_json() -> None:
    transition = _representative_multi_event_transition()

    assert _event_types(transition.events) == (
        "action_rejected",
        "charge_phase_displacement",
    )
    assert (
        EvaluationTransitionV1.model_validate_json(transition.model_dump_json())
        == transition
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (("event_type", "unknown_event"), ("schema_version", 2)),
)
def test_transition_json_rejects_unknown_event_contract_values(
    field_name: str,
    invalid_value: object,
) -> None:
    payload = json.loads(_representative_multi_event_transition().model_dump_json())
    payload["events"][0][field_name] = invalid_value

    with pytest.raises(ValidationError, match=field_name):
        EvaluationTransitionV1.model_validate_json(json.dumps(payload))


def test_decoder_emits_all_atomic_variants_with_canonical_ids_and_order() -> None:
    facts = _neutral_facts()

    submitted = _joint_action(
        move=_replace_item(_ZERO_INT_10, 2, -(2**31)),
        select_target=_replace_item(_ZERO_INT_10, 2, 2**31 - 1),
        use_ultimate=_replace_item(_ZERO_INT_10, 2, -1),
    )
    accepted_use_ultimate = _replace_item(_ZERO_INT_10, 0, 1)
    accepted_use_ultimate = _replace_item(accepted_use_ultimate, 1, 1)
    accepted_use_ultimate = _replace_item(accepted_use_ultimate, 5, 1)
    accepted = _joint_action(use_ultimate=accepted_use_ultimate)
    acceptance = facts.action_acceptance_facts.model_copy(
        update={
            "submitted_joint_action": submitted,
            "accepted_joint_action": accepted,
            "submitted_action_tuple_is_out_of_domain_by_actor": _replace_item(
                _FALSE_10, 2, True
            ),
            "in_domain_move_action_is_rejected_by_actor": _replace_item(
                _FALSE_10, 3, True
            ),
            "in_domain_combat_action_pair_is_rejected_by_actor": _replace_item(
                _FALSE_10, 4, True
            ),
        }
    )

    recipient_by_source = _replace_item(_NONE_10, 1, 5)
    recipient_by_source = _replace_item(recipient_by_source, 2, 1)
    recipient_by_source = _replace_item(recipient_by_source, 3, 5)
    recipient_by_source = _replace_item(recipient_by_source, 5, 0)
    has_recipient = _replace_item(_FALSE_10, 1, True)
    has_recipient = _replace_item(has_recipient, 2, True)
    has_recipient = _replace_item(has_recipient, 3, True)
    has_recipient = _replace_item(has_recipient, 5, True)
    ultimate_activated = _replace_item(_FALSE_10, 0, True)
    ultimate_activated = _replace_item(ultimate_activated, 1, True)
    ultimate_activated = _replace_item(ultimate_activated, 5, True)
    combat = facts.combat_transition_facts.model_copy(
        update={
            "basic_effect_is_activated_by_source": _replace_item(_FALSE_10, 2, True),
            "ultimate_effect_is_activated_by_source": ultimate_activated,
            "combat_effect_has_recipient_by_source": has_recipient,
            "combat_effect_recipient_global_slot_by_source": recipient_by_source,
            "raw_damage_output_by_source": _replace_item(
                _replace_item(_ZERO_FLOAT_10, 1, 40.0), 3, 10.0
            ),
            "source_modified_damage_output_by_source": _replace_item(
                _replace_item(_ZERO_FLOAT_10, 1, 50.0), 3, 10.0
            ),
            "recipient_damage_modifier_by_source": _replace_item(
                _replace_item(_ZERO_FLOAT_10, 1, 0.8), 3, 0.8
            ),
            "raw_healing_output_by_source": _replace_item(_ZERO_FLOAT_10, 2, 10.0),
            "source_modified_healing_output_by_source": _replace_item(
                _ZERO_FLOAT_10, 2, 10.0
            ),
            "recipient_healing_modifier_by_source": _replace_item(
                _ZERO_FLOAT_10, 2, 1.0
            ),
            "total_effective_damage_by_recipient": _replace_item(
                _ZERO_FLOAT_10, 5, 40.0
            ),
            "total_effective_healing_by_recipient": _replace_item(
                _ZERO_FLOAT_10, 5, 5.0
            ),
            "health_after_combat_resolution_by_recipient": _replace_item(
                _HEALTH_100, 5, 0.0
            ),
            "slow_is_applied_by_source_and_channel": _replace_matrix_item(
                _FALSE_THREE_CHANNEL_MATRIX, 1, 0, True
            ),
            "stun_is_applied_by_source_and_channel": _replace_matrix_item(
                _FALSE_THREE_CHANNEL_MATRIX, 5, 1, True
            ),
            "mage_burst_damage_amplification_is_applied_by_source": (
                _replace_item(_FALSE_10, 0, True)
            ),
        }
    )

    mage_aura = _replace_matrix_item(_FALSE_MATRIX_10, 0, 1, True)
    mage_aura = _replace_matrix_item(mage_aura, 2, 1, True)
    warrior_aura = _replace_matrix_item(_FALSE_MATRIX_10, 1, 5, True)
    warrior_aura = _replace_matrix_item(warrior_aura, 2, 5, True)
    aura = facts.aura_facts.model_copy(
        update={
            "is_covered_by_mage_damage_aura_by_emitter_and_beneficiary": (mage_aura),
            "is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary": (
                warrior_aura
            ),
        }
    )

    death = facts.death_facts.model_copy(
        update={
            "is_newly_dead_by_recipient": _replace_item(_FALSE_10, 5, True),
            "contributed_to_new_death_by_source": _replace_item(
                _replace_item(_FALSE_10, 1, True), 3, True
            ),
            "attributed_death_damage_by_source": _replace_item(
                _replace_item(_ZERO_FLOAT_10, 1, 32.0), 3, 8.0
            ),
        }
    )

    aged = _replace_matrix_item(_FALSE_STATUS_MATRIX, 5, 0, True)
    broken = _replace_matrix_item(_FALSE_STATUS_MATRIX, 5, 4, True)
    refreshed = _replace_matrix_item(_FALSE_STATUS_MATRIX, 0, 7, True)
    cleared = _replace_matrix_item(_FALSE_STATUS_MATRIX, 0, 7, True)
    cleared = _replace_matrix_item(cleared, 5, 0, True)
    cleared = _replace_matrix_item(cleared, 5, 4, True)
    lifecycle = facts.status_lifecycle_facts.model_copy(
        update={
            "aged_to_zero_by_recipient_and_status_channel": aged,
            "broken_by_damage_by_recipient_and_status_channel": broken,
            "refreshed_or_extended_by_recipient_and_status_channel": refreshed,
            "cleared_by_new_death_by_recipient_and_status_channel": cleared,
        }
    )

    facts = facts.model_copy(
        update={
            "action_acceptance_facts": acceptance,
            "combat_transition_facts": combat,
            "death_facts": death,
            "aura_facts": aura,
            "regeneration_facts": facts.regeneration_facts.model_copy(
                update={
                    "combat_countdown_was_reset_by_agent": _replace_item(
                        _FALSE_10, 1, True
                    ),
                    "actual_health_regenerated_this_step_by_agent": _replace_item(
                        _ZERO_FLOAT_10, 2, 4.0
                    ),
                }
            ),
            "physical_facts": facts.physical_facts.model_copy(
                update={
                    "charge_phase_displacement_by_agent": _replace_item(
                        _ZERO_DISPLACEMENT_MATRIX, 1, (1.5, -0.5)
                    ),
                    "ordinary_movement_phase_displacement_by_agent": _replace_item(
                        _ZERO_DISPLACEMENT_MATRIX, 0, (0.0, 1.0)
                    ),
                }
            ),
            "status_lifecycle_facts": lifecycle,
            "spawn_shield_facts": facts.spawn_shield_facts.model_copy(
                update={
                    "expired_at_transition_end_by_agent": _replace_item(
                        _FALSE_10, 2, True
                    )
                }
            ),
            "respawn_facts": facts.respawn_facts.model_copy(
                update={
                    "respawn_wave_occurred_this_transition_by_team": (
                        True,
                        True,
                    ),
                    "was_respawned_this_transition_by_agent": _replace_item(
                        _FALSE_10, 6, True
                    ),
                }
            ),
        }
    )

    start_health = _replace_item(_HEALTH_100, 5, 30.0)
    start_cooldowns = _replace_item(_ZERO_INT_10, 2, 1)
    successor_cooldowns = _replace_item(_ZERO_INT_10, 0, 5)
    successor_cooldowns = _replace_item(successor_cooldowns, 1, 5)
    successor_cooldowns = _replace_item(successor_cooldowns, 5, 5)
    successor_positions = tuple((float(slot), 0.0) for slot in range(10))
    successor_positions = _replace_item(successor_positions, 6, (8.0, 9.0))
    events = _decode(
        facts,
        start_snapshot=_snapshot(
            current_health=start_health,
            ultimate_cooldowns=start_cooldowns,
        ),
        successor_snapshot=_snapshot(
            current_health=_replace_item(start_health, 5, 0.0),
            ultimate_cooldowns=successor_cooldowns,
            agent_positions=successor_positions,
        ),
    )

    assert _event_types(events) == (
        "action_rejected",
        "action_rejected",
        "action_rejected",
        "ability_activated",
        "ability_activated",
        "ability_activated",
        "ability_activated",
        "source_damage_output",
        "source_healing_output",
        "source_damage_output",
        "recipient_health_resolution",
        "combat_countdown_reset",
        "health_regenerated",
        "cooldown_started",
        "cooldown_started",
        "cooldown_ready",
        "cooldown_started",
        "charge_phase_displacement",
        "ordinary_movement_phase_displacement",
        "agent_died",
        "lethal_damage_contribution",
        "lethal_damage_contribution",
        "status_applied",
        "status_applied",
        "status_refreshed_or_extended",
        "status_cleared_by_new_death",
        "status_aged_to_zero",
        "status_applied",
        "status_cleared_by_new_death",
        "status_broken_by_damage",
        "status_cleared_by_new_death",
        "spawn_shield_expired",
        "respawn_wave_occurred",
        "respawn_wave_occurred",
        "agent_respawned",
    )
    assert set(_event_types(events)) == _ALL_EVENT_TYPES
    assert Counter(_event_types(events)) == Counter(
        {
            "action_rejected": 3,
            "ability_activated": 4,
            "source_damage_output": 2,
            "source_healing_output": 1,
            "recipient_health_resolution": 1,
            "combat_countdown_reset": 1,
            "health_regenerated": 1,
            "cooldown_started": 3,
            "cooldown_ready": 1,
            "charge_phase_displacement": 1,
            "ordinary_movement_phase_displacement": 1,
            "agent_died": 1,
            "lethal_damage_contribution": 2,
            "status_aged_to_zero": 1,
            "status_broken_by_damage": 1,
            "status_applied": 3,
            "status_refreshed_or_extended": 1,
            "status_cleared_by_new_death": 3,
            "spawn_shield_expired": 1,
            "respawn_wave_occurred": 2,
            "agent_respawned": 1,
        }
    )
    assert tuple(event.phase_rank for event in events) == tuple(
        sorted(event.phase_rank for event in events)
    )
    assert tuple(event.ordinal for event in events) == tuple(range(len(events)))
    assert tuple(event.event_id for event in events) == tuple(
        f"episode-001:transition:3:event:{ordinal:04d}"
        for ordinal in range(len(events))
    )
    assert events == _decode(
        facts,
        start_snapshot=_snapshot(
            current_health=start_health,
            ultimate_cooldowns=start_cooldowns,
        ),
        successor_snapshot=_snapshot(
            current_health=_replace_item(start_health, 5, 0.0),
            ultimate_cooldowns=successor_cooldowns,
            agent_positions=successor_positions,
        ),
    )
    assert (
        tuple(_EVENT_ADAPTER.validate_json(event.model_dump_json()) for event in events)
        == events
    )

    payloads_by_phase = {
        phase_rank: tuple(
            event.model_dump() for event in events if event.phase_rank == phase_rank
        )
        for phase_rank in (10, 20, 30, 50, 60, 70, 80, 90, 100, 110, 120)
    }
    assert tuple(
        (payload["actor_global_slot"], payload["rejection_component"])
        for payload in payloads_by_phase[10]
    ) == ((2, "domain"), (3, "movement"), (4, "combat_pair"))
    assert tuple(
        (
            payload["source_global_slot"],
            payload["ability_component"],
            payload["recipient_global_slot"],
        )
        for payload in payloads_by_phase[20]
    ) == (
        (0, "ultimate", None),
        (1, "ultimate", 5),
        (2, "basic", 1),
        (5, "ultimate", 0),
    )
    assert tuple(
        (
            payload["event_type"],
            payload["source_global_slot"],
            payload["recipient_global_slot"],
        )
        for payload in payloads_by_phase[30]
    ) == (
        ("source_damage_output", 1, 5),
        ("source_healing_output", 2, 1),
        ("source_damage_output", 3, 5),
    )
    assert tuple(
        (
            payload["event_type"],
            payload["agent_global_slot"],
            payload.get("actual_health_regenerated"),
        )
        for payload in payloads_by_phase[50]
    ) == (
        ("combat_countdown_reset", 1, None),
        ("health_regenerated", 2, 4.0),
    )
    assert tuple(
        (payload["event_type"], payload["agent_global_slot"])
        for payload in payloads_by_phase[60]
    ) == (
        ("cooldown_started", 0),
        ("cooldown_started", 1),
        ("cooldown_ready", 2),
        ("cooldown_started", 5),
    )
    assert tuple(
        (payload["agent_global_slot"], payload["realized_displacement"])
        for payload in payloads_by_phase[70]
    ) == ((1, (1.5, -0.5)),)
    assert tuple(
        (payload["agent_global_slot"], payload["realized_displacement"])
        for payload in payloads_by_phase[80]
    ) == ((0, (0.0, 1.0)),)
    assert tuple(
        (
            payload["event_type"],
            payload["recipient_global_slot"],
            payload.get("source_global_slot"),
        )
        for payload in payloads_by_phase[90]
    ) == (
        ("agent_died", 5, None),
        ("lethal_damage_contribution", 5, 1),
        ("lethal_damage_contribution", 5, 3),
    )
    assert tuple(
        (
            payload["event_type"],
            payload["recipient_global_slot"],
            payload["status_channel"],
            payload.get("source_global_slot"),
        )
        for payload in payloads_by_phase[100]
    ) == (
        ("status_applied", 0, 4, 5),
        ("status_applied", 0, 7, 0),
        ("status_refreshed_or_extended", 0, 7, None),
        ("status_cleared_by_new_death", 0, 7, None),
        ("status_aged_to_zero", 5, 0, None),
        ("status_applied", 5, 0, 1),
        ("status_cleared_by_new_death", 5, 0, None),
        ("status_broken_by_damage", 5, 4, None),
        ("status_cleared_by_new_death", 5, 4, None),
    )
    assert tuple(
        (payload["event_type"], payload["agent_global_slot"])
        for payload in payloads_by_phase[110]
    ) == (("spawn_shield_expired", 2),)
    assert tuple(
        (
            payload["event_type"],
            payload.get("team_index", payload.get("agent_global_slot")),
            payload["team_id"],
        )
        for payload in payloads_by_phase[120]
    ) == (
        ("respawn_wave_occurred", 0, 1),
        ("respawn_wave_occurred", 1, 2),
        ("agent_respawned", 6, 2),
    )

    rejected = next(
        event
        for event in events
        if event.event_type == "action_rejected"
        and event.model_dump().get("actor_global_slot") == 2
    )
    assert rejected.model_dump(
        include={
            "rejection_component",
            "submitted_move_action",
            "submitted_select_target_action",
            "submitted_use_ultimate_action",
        }
    ) == {
        "rejection_component": "domain",
        "submitted_move_action": -(2**31),
        "submitted_select_target_action": 2**31 - 1,
        "submitted_use_ultimate_action": -1,
    }

    damage = next(
        event
        for event in events
        if isinstance(event, SourceDamageOutputEventV1)
        and event.source_global_slot == 1
    )
    assert damage.recipient_global_slot == 5
    assert damage.raw_damage_output == 40.0
    assert damage.source_modified_damage_output == 50.0
    assert damage.recipient_damage_modifier == 0.8
    assert damage.mage_damage_aura_covering_emitter_global_slots == (0, 2)
    assert damage.warrior_mitigation_aura_covering_emitter_global_slots == (1, 2)

    healing = next(
        event for event in events if event.event_type == "source_healing_output"
    )
    assert healing.model_dump(
        include={
            "source_global_slot",
            "recipient_global_slot",
            "raw_healing_output",
            "source_modified_healing_output",
            "recipient_healing_modifier",
        }
    ) == {
        "source_global_slot": 2,
        "recipient_global_slot": 1,
        "raw_healing_output": 10.0,
        "source_modified_healing_output": 10.0,
        "recipient_healing_modifier": 1.0,
    }

    health = next(
        event for event in events if isinstance(event, RecipientHealthResolutionEventV1)
    )
    assert health.transition_start_health == 30.0
    assert health.health_after_combat_resolution == 0.0
    assert health.realized_net_health_change == -30.0

    respawn = next(
        event for event in events if isinstance(event, AgentRespawnedEventV1)
    )
    assert respawn.team_id == 2
    assert respawn.realized_successor_position == (8.0, 9.0)


def test_mage_target_none_application_is_source_local() -> None:
    facts = _neutral_facts()
    facts = facts.model_copy(
        update={
            "action_acceptance_facts": facts.action_acceptance_facts.model_copy(
                update={
                    "accepted_joint_action": _joint_action(
                        use_ultimate=_replace_item(_ZERO_INT_10, 0, 1)
                    )
                }
            ),
            "combat_transition_facts": facts.combat_transition_facts.model_copy(
                update={
                    "ultimate_effect_is_activated_by_source": _replace_item(
                        _FALSE_10, 0, True
                    ),
                    "mage_burst_damage_amplification_is_applied_by_source": (
                        _replace_item(_FALSE_10, 0, True)
                    ),
                }
            ),
        }
    )
    events = _decode(
        facts,
        successor_snapshot=_snapshot(
            ultimate_cooldowns=_replace_item(_ZERO_INT_10, 0, 5)
        ),
    )

    activation = next(
        event for event in events if event.event_type == "ability_activated"
    )
    application = next(
        event for event in events if isinstance(event, StatusAppliedEventV1)
    )
    assert activation.recipient_global_slot is None
    assert application.source_global_slot == 0
    assert application.recipient_global_slot == 0
    assert application.status_channel == 7
    assert application.status_id == "mage_burst_damage_amplification"


def test_cooldown_ready_is_alive_independent_and_zero_to_zero_is_silent() -> None:
    start_alive = _replace_item((True,) * 10, 2, False)
    successor_alive = _replace_item((True,) * 10, 2, False)
    events = _decode(
        _neutral_facts(),
        start_snapshot=_snapshot(
            alive_mask=start_alive,
            ultimate_cooldowns=_replace_item(_ZERO_INT_10, 2, 1),
        ),
        successor_snapshot=_snapshot(
            alive_mask=successor_alive,
            ultimate_cooldowns=_ZERO_INT_10,
        ),
    )

    assert _event_types(events) == ("cooldown_ready",)
    ready = events[0]
    assert isinstance(ready, CooldownReadyEventV1)
    assert ready.agent_global_slot == 2


@pytest.mark.parametrize("amount_field", ("damage", "healing"))
def test_positive_source_output_requires_authoritative_recipient(
    amount_field: str,
) -> None:
    facts = _neutral_facts()
    update: dict[str, tuple[float, ...]]
    if amount_field == "damage":
        update = {"raw_damage_output_by_source": _replace_item(_ZERO_FLOAT_10, 0, 1.0)}
    else:
        update = {"raw_healing_output_by_source": _replace_item(_ZERO_FLOAT_10, 0, 1.0)}
    facts = facts.model_copy(
        update={
            "combat_transition_facts": facts.combat_transition_facts.model_copy(
                update=update
            )
        }
    )

    with pytest.raises(ValueError, match="authoritative recipient route"):
        _decode(facts)


def test_source_and_recipient_events_use_only_their_direct_positive_triggers() -> None:
    facts = _neutral_facts()
    combat = facts.combat_transition_facts.model_copy(
        update={
            "source_modified_damage_output_by_source": _replace_item(
                _ZERO_FLOAT_10, 0, 9.0
            ),
            "recipient_damage_modifier_by_source": _replace_item(
                _ZERO_FLOAT_10, 0, 1.0
            ),
            "total_effective_healing_by_recipient": _replace_item(
                _ZERO_FLOAT_10, 4, 2.0
            ),
            "health_after_combat_resolution_by_recipient": _replace_item(
                _HEALTH_100, 4, 100.0
            ),
        }
    )
    events = _decode(facts.model_copy(update={"combat_transition_facts": combat}))

    assert _event_types(events) == ("recipient_health_resolution",)
    resolution = events[0]
    assert isinstance(resolution, RecipientHealthResolutionEventV1)
    assert resolution.recipient_global_slot == 4
    assert resolution.total_effective_healing == 2.0


def test_broken_cooldown_and_contribution_joins_raise() -> None:
    facts = _neutral_facts()
    accepted_without_activation = facts.model_copy(
        update={
            "action_acceptance_facts": facts.action_acceptance_facts.model_copy(
                update={
                    "accepted_joint_action": _joint_action(
                        use_ultimate=_replace_item(_ZERO_INT_10, 0, 1)
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="Ultimate action and activation"):
        _decode(accepted_without_activation)

    activation_without_accepted = facts.model_copy(
        update={
            "combat_transition_facts": facts.combat_transition_facts.model_copy(
                update={
                    "ultimate_effect_is_activated_by_source": _replace_item(
                        _FALSE_10, 0, True
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="Ultimate action and activation"):
        _decode(activation_without_accepted)

    contribution_without_death = facts.model_copy(
        update={
            "combat_transition_facts": facts.combat_transition_facts.model_copy(
                update={
                    "combat_effect_has_recipient_by_source": _replace_item(
                        _FALSE_10, 1, True
                    ),
                    "combat_effect_recipient_global_slot_by_source": _replace_item(
                        _NONE_10, 1, 5
                    ),
                }
            ),
            "death_facts": facts.death_facts.model_copy(
                update={
                    "contributed_to_new_death_by_source": _replace_item(
                        _FALSE_10, 1, True
                    ),
                    "attributed_death_damage_by_source": _replace_item(
                        _ZERO_FLOAT_10, 1, 3.0
                    ),
                }
            ),
        }
    )
    with pytest.raises(ValueError, match="newly dead recipient"):
        _decode(contribution_without_death)


@pytest.mark.parametrize(
    ("contributor_flag", "attributed_damage", "newly_dead", "error_match"),
    (
        (True, 0.0, False, "flag and positive attributed damage"),
        (False, 1.0, False, "flag and positive attributed damage"),
        (True, 1.0, True, "authoritative recipient route"),
    ),
)
def test_contribution_requires_consistent_flag_amount_and_route(
    contributor_flag: bool,
    attributed_damage: float,
    newly_dead: bool,
    error_match: str,
) -> None:
    facts = _neutral_facts()
    facts = facts.model_copy(
        update={
            "death_facts": facts.death_facts.model_copy(
                update={
                    "is_newly_dead_by_recipient": _replace_item(
                        _FALSE_10, 5, newly_dead
                    ),
                    "contributed_to_new_death_by_source": _replace_item(
                        _FALSE_10, 1, contributor_flag
                    ),
                    "attributed_death_damage_by_source": _replace_item(
                        _ZERO_FLOAT_10, 1, attributed_damage
                    ),
                }
            )
        }
    )

    with pytest.raises(ValueError, match=error_match):
        _decode(facts)


def test_non_mage_activation_and_status_application_require_routed_recipient() -> None:
    facts = _neutral_facts()
    facts = facts.model_copy(
        update={
            "action_acceptance_facts": facts.action_acceptance_facts.model_copy(
                update={
                    "accepted_joint_action": _joint_action(
                        use_ultimate=_replace_item(_ZERO_INT_10, 1, 1)
                    )
                }
            ),
            "combat_transition_facts": facts.combat_transition_facts.model_copy(
                update={
                    "ultimate_effect_is_activated_by_source": _replace_item(
                        _FALSE_10, 1, True
                    ),
                    "slow_is_applied_by_source_and_channel": _replace_matrix_item(
                        _FALSE_THREE_CHANNEL_MATRIX, 1, 0, True
                    ),
                }
            ),
        }
    )

    with pytest.raises(ValueError, match="authoritative recipient route"):
        _decode(facts)


def test_basic_activation_requires_routed_recipient() -> None:
    facts = _neutral_facts()
    facts = facts.model_copy(
        update={
            "combat_transition_facts": facts.combat_transition_facts.model_copy(
                update={
                    "basic_effect_is_activated_by_source": _replace_item(
                        _FALSE_10, 1, True
                    )
                }
            )
        }
    )

    with pytest.raises(ValueError, match="Basic activation"):
        _decode(facts)


def test_unavailable_basic_activation_is_rejected() -> None:
    facts = _neutral_facts()
    facts = facts.model_copy(
        update={
            "combat_transition_facts": facts.combat_transition_facts.model_copy(
                update={
                    "basic_effect_is_activated_by_source": _replace_item(
                        _FALSE_10, 3, True
                    ),
                    "combat_effect_has_recipient_by_source": _replace_item(
                        _FALSE_10, 3, True
                    ),
                    "combat_effect_recipient_global_slot_by_source": _replace_item(
                        _NONE_10, 3, 5
                    ),
                }
            )
        }
    )

    with pytest.raises(ValueError, match="unavailable Basic"):
        _decode(facts)


@pytest.mark.parametrize(
    ("source_global_slot", "recipient_global_slot", "error_match"),
    (
        (1, None, "enemy Ultimate activation"),
        (0, 5, "target-none Ultimate activation"),
        (3, None, "unavailable Ultimate"),
    ),
)
def test_ultimate_activation_route_must_match_catalog_target_mode(
    source_global_slot: int,
    recipient_global_slot: int | None,
    error_match: str,
) -> None:
    facts = _neutral_facts()
    has_recipient = _FALSE_10
    recipient_by_source = _NONE_10
    if recipient_global_slot is not None:
        has_recipient = _replace_item(has_recipient, source_global_slot, True)
        recipient_by_source = _replace_item(
            recipient_by_source,
            source_global_slot,
            recipient_global_slot,
        )
    facts = facts.model_copy(
        update={
            "action_acceptance_facts": facts.action_acceptance_facts.model_copy(
                update={
                    "accepted_joint_action": _joint_action(
                        use_ultimate=_replace_item(_ZERO_INT_10, source_global_slot, 1)
                    )
                }
            ),
            "combat_transition_facts": facts.combat_transition_facts.model_copy(
                update={
                    "ultimate_effect_is_activated_by_source": _replace_item(
                        _FALSE_10, source_global_slot, True
                    ),
                    "combat_effect_has_recipient_by_source": has_recipient,
                    "combat_effect_recipient_global_slot_by_source": (
                        recipient_by_source
                    ),
                }
            ),
        }
    )

    with pytest.raises(ValueError, match=error_match):
        _decode(facts)


def test_status_application_requires_catalog_source_class_and_activation() -> None:
    facts = _neutral_facts()
    routed_combat = facts.combat_transition_facts.model_copy(
        update={
            "combat_effect_has_recipient_by_source": _replace_item(_FALSE_10, 1, True),
            "combat_effect_recipient_global_slot_by_source": _replace_item(
                _NONE_10, 1, 5
            ),
        }
    )

    wrong_class = facts.model_copy(
        update={
            "combat_transition_facts": routed_combat.model_copy(
                update={
                    "basic_effect_is_activated_by_source": _replace_item(
                        _FALSE_10, 1, True
                    ),
                    "slow_is_applied_by_source_and_channel": _replace_matrix_item(
                        _FALSE_THREE_CHANNEL_MATRIX, 1, 1, True
                    ),
                }
            )
        }
    )
    with pytest.raises(ValueError, match="source class"):
        _decode(wrong_class)

    missing_activation = facts.model_copy(
        update={
            "combat_transition_facts": routed_combat.model_copy(
                update={
                    "slow_is_applied_by_source_and_channel": _replace_matrix_item(
                        _FALSE_THREE_CHANNEL_MATRIX, 1, 0, True
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="ability activation"):
        _decode(missing_activation)


def test_respawn_order_groups_each_team_wave_before_its_agents() -> None:
    facts = _neutral_facts()
    respawned_by_agent = _replace_item(_FALSE_10, 1, True)
    respawned_by_agent = _replace_item(respawned_by_agent, 5, True)
    facts = facts.model_copy(
        update={
            "respawn_facts": facts.respawn_facts.model_copy(
                update={
                    "respawn_wave_occurred_this_transition_by_team": (True, True),
                    "was_respawned_this_transition_by_agent": respawned_by_agent,
                }
            )
        }
    )

    events = _decode(facts)

    assert _event_types(events) == (
        "respawn_wave_occurred",
        "agent_respawned",
        "respawn_wave_occurred",
        "agent_respawned",
    )
    assert tuple(
        (
            payload["event_type"],
            payload.get("team_index", payload.get("agent_global_slot")),
            payload["team_id"],
        )
        for payload in (event.model_dump() for event in events)
    ) == (
        ("respawn_wave_occurred", 0, 1),
        ("agent_respawned", 1, 1),
        ("respawn_wave_occurred", 1, 2),
        ("agent_respawned", 5, 2),
    )


@pytest.mark.parametrize(
    ("wave_by_team", "successor_alive", "error_match"),
    (
        ((False, True), True, "configured team's wave"),
        ((True, False), False, "alive in the successor"),
    ),
)
def test_respawn_requires_team_wave_and_successor_alive(
    wave_by_team: tuple[bool, bool],
    successor_alive: bool,
    error_match: str,
) -> None:
    facts = _neutral_facts()
    facts = facts.model_copy(
        update={
            "respawn_facts": facts.respawn_facts.model_copy(
                update={
                    "respawn_wave_occurred_this_transition_by_team": wave_by_team,
                    "was_respawned_this_transition_by_agent": _replace_item(
                        _FALSE_10, 0, True
                    ),
                }
            )
        }
    )
    successor_alive_mask = _replace_item((True,) * 10, 0, successor_alive)

    with pytest.raises(ValueError, match=error_match):
        _decode(
            facts,
            successor_snapshot=_snapshot(alive_mask=successor_alive_mask),
        )


def test_decoder_rejects_nonadjacent_record_joins() -> None:
    context = evaluation_context()
    snapshot = _snapshot()
    facts = _neutral_facts()

    with pytest.raises(ValueError, match="join the context episode"):
        decode_evaluation_events_v1(
            context,
            _frame(3, 7, snapshot).model_copy(update={"episode_id": "other-episode"}),
            facts,
            _frame(4, 8, snapshot),
        )
    with pytest.raises(ValueError, match="directly adjacent"):
        decode_evaluation_events_v1(
            context,
            _frame(3, 7, snapshot),
            facts,
            _frame(5, 8, snapshot),
        )
    with pytest.raises(ValueError, match="adjacent simulator steps"):
        decode_evaluation_events_v1(
            context,
            _frame(3, 7, snapshot),
            facts,
            _frame(4, 9, snapshot),
        )
    with pytest.raises(ValueError, match="start frame simulator step"):
        decode_evaluation_events_v1(
            context,
            _frame(3, 6, snapshot),
            facts,
            _frame(4, 7, snapshot),
        )
    with pytest.raises(ValueError, match="real-transition facts"):
        decode_evaluation_events_v1(
            context,
            _frame(3, 7, snapshot),
            facts.model_copy(update={"has_transition": False}),
            _frame(4, 8, snapshot),
        )
