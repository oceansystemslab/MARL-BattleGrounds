"""Deterministic host decoding of authoritative transition facts into events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from marl_battlegrounds.evaluation.models import (
    AbilityActivatedEventV1,
    ActionRejectedEventV1,
    AgentDiedEventV1,
    AgentRespawnedEventV1,
    ChargePhaseDisplacementEventV1,
    CombatCountdownResetEventV1,
    CooldownReadyEventV1,
    CooldownStartedEventV1,
    EvaluationEpisodeContextV1,
    EvaluationEventBaseV1,
    EvaluationEventV1,
    EvaluationFrameV1,
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
    TeamDeathmatchCompletedEventV1,
    TeamDeathmatchScoreChangedEventV1,
    TransitionFactsV1,
)

_NULL_RECIPIENT_SORT_INDEX = 10
_MAGE_BURST_STATUS_CHANNEL = 7
_TEAM_DEATHMATCH_TASK_MODE = 1
_OUTCOME_ONGOING = 0
_OUTCOME_TEAM_A_WIN = 1
_OUTCOME_TEAM_B_WIN = 2
_OUTCOME_DRAW = 3


@dataclass(frozen=True, slots=True)
class _TeamDeathmatchAuthorityV1:
    """Validated task authority joined across one adjacent transition."""

    score_increments: tuple[int, int]
    outcome: int
    threshold_reached: bool
    horizon_reached: bool
    completion_basis: str | None


@dataclass(frozen=True, slots=True)
class _EventCandidate:
    """One event payload plus its stable pre-ordinal ordering coordinates."""

    phase_rank: int
    primary_slot_or_team_index: int
    secondary_slot_or_status_channel: int
    subtype_rank: int
    source_slot: int
    model_type: type[EvaluationEventBaseV1]
    payload: dict[str, object]

    @property
    def sort_key(self) -> tuple[int, int, int, int, int]:
        """Return the complete canonical candidate ordering key."""
        return (
            self.phase_rank,
            self.primary_slot_or_team_index,
            self.secondary_slot_or_status_channel,
            self.subtype_rank,
            self.source_slot,
        )


def _append_candidate(
    candidates: list[_EventCandidate],
    *,
    phase_rank: int,
    primary_slot_or_team_index: int,
    secondary_slot_or_status_channel: int = 0,
    subtype_rank: int = 0,
    source_slot: int = 0,
    model_type: type[EvaluationEventBaseV1],
    payload: dict[str, object],
) -> None:
    """Append one fully keyed event candidate without assigning identity yet."""
    candidates.append(
        _EventCandidate(
            phase_rank=phase_rank,
            primary_slot_or_team_index=primary_slot_or_team_index,
            secondary_slot_or_status_channel=(secondary_slot_or_status_channel),
            subtype_rank=subtype_rank,
            source_slot=source_slot,
            model_type=model_type,
            payload=payload,
        )
    )


def _recipient_sort_index(recipient_global_slot: int | None) -> int:
    """Place a nullable recipient after every concrete fixed global slot."""
    if recipient_global_slot is None:
        return _NULL_RECIPIENT_SORT_INDEX
    return recipient_global_slot


def _require_routed_recipient(
    recipient_global_slot: int | None,
    *,
    relation: str,
) -> int:
    """Return a required authoritative route or reject a broken direct join."""
    if recipient_global_slot is None:
        raise ValueError(f"{relation} requires an authoritative recipient route")
    return recipient_global_slot


def _ultimate_activation_recipient(
    context: EvaluationEpisodeContextV1,
    source_global_slot: int,
    recipient_global_slot: int | None,
) -> int | None:
    """Validate an Ultimate route against its serialized class target mode."""
    class_id = context.roster[source_global_slot].class_id
    target_mode = context.static_mechanics_catalog.class_mechanics[
        class_id
    ].ultimate_target_mode
    if target_mode == "unavailable":
        raise ValueError("unavailable Ultimate cannot have an activation fact")
    if target_mode == "target_none":
        if recipient_global_slot is not None:
            raise ValueError(
                "target-none Ultimate activation must not have a recipient"
            )
        return None
    return _require_routed_recipient(
        recipient_global_slot,
        relation=f"{target_mode} Ultimate activation",
    )


def _basic_activation_recipient(
    context: EvaluationEpisodeContextV1,
    source_global_slot: int,
    recipient_global_slot: int | None,
) -> int:
    """Validate a Basic route against its serialized class target mode."""
    class_id = context.roster[source_global_slot].class_id
    target_mode = context.static_mechanics_catalog.class_mechanics[
        class_id
    ].basic_target_mode
    if target_mode == "unavailable":
        raise ValueError("unavailable Basic cannot have an activation fact")
    return _require_routed_recipient(
        recipient_global_slot,
        relation=f"{target_mode} Basic activation",
    )


def _validate_decoder_inputs(
    context: EvaluationEpisodeContextV1,
    start_frame: EvaluationFrameV1,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrameV1,
) -> None:
    """Reject records that cannot represent one directly adjacent transition."""
    episode_id = context.identity.episode_id
    if start_frame.episode_id != episode_id or successor_frame.episode_id != episode_id:
        raise ValueError("event decoder frames must join the context episode")
    if successor_frame.frame_index != start_frame.frame_index + 1:
        raise ValueError("event decoder frames must be directly adjacent")
    if successor_frame.simulator_step_count != start_frame.simulator_step_count + 1:
        raise ValueError("event decoder frames must represent adjacent simulator steps")
    if not facts.has_transition:
        raise ValueError("event decoding requires real-transition facts")
    if facts.transition_start_step_count != start_frame.simulator_step_count:
        raise ValueError(
            "transition facts must identify the start frame simulator step"
        )


def _append_action_candidates(
    candidates: list[_EventCandidate],
    context: EvaluationEpisodeContextV1,
    facts: TransitionFactsV1,
) -> None:
    """Copy every independently rejected component and accepted activation."""
    acceptance = facts.action_acceptance_facts
    submitted = acceptance.submitted_joint_action
    rejection_families = (
        (
            acceptance.submitted_action_tuple_is_out_of_domain_by_actor,
            "domain",
            0,
        ),
        (
            acceptance.in_domain_move_action_is_rejected_by_actor,
            "movement",
            1,
        ),
        (
            acceptance.in_domain_combat_action_pair_is_rejected_by_actor,
            "combat_pair",
            2,
        ),
    )
    for actor_global_slot in range(len(submitted.move)):
        for flags, rejection_component, subtype_rank in rejection_families:
            if not flags[actor_global_slot]:
                continue
            _append_candidate(
                candidates,
                phase_rank=10,
                primary_slot_or_team_index=actor_global_slot,
                subtype_rank=subtype_rank,
                model_type=ActionRejectedEventV1,
                payload={
                    "actor_global_slot": actor_global_slot,
                    "rejection_component": rejection_component,
                    "submitted_move_action": submitted.move[actor_global_slot],
                    "submitted_select_target_action": (
                        submitted.select_target[actor_global_slot]
                    ),
                    "submitted_use_ultimate_action": (
                        submitted.use_ultimate[actor_global_slot]
                    ),
                },
            )

    combat = facts.combat_transition_facts
    activation_families = (
        (combat.basic_effect_is_activated_by_source, "basic", 0),
        (combat.ultimate_effect_is_activated_by_source, "ultimate", 1),
    )
    for source_global_slot, recipient_global_slot in enumerate(
        combat.combat_effect_recipient_global_slot_by_source
    ):
        for flags, ability_component, subtype_rank in activation_families:
            if not flags[source_global_slot]:
                continue
            if ability_component == "basic":
                event_recipient_global_slot = _basic_activation_recipient(
                    context,
                    source_global_slot,
                    recipient_global_slot,
                )
            else:
                event_recipient_global_slot = _ultimate_activation_recipient(
                    context,
                    source_global_slot,
                    recipient_global_slot,
                )
            _append_candidate(
                candidates,
                phase_rank=20,
                primary_slot_or_team_index=source_global_slot,
                secondary_slot_or_status_channel=_recipient_sort_index(
                    event_recipient_global_slot
                ),
                subtype_rank=subtype_rank,
                source_slot=source_global_slot,
                model_type=AbilityActivatedEventV1,
                payload={
                    "source_global_slot": source_global_slot,
                    "ability_component": ability_component,
                    "recipient_global_slot": event_recipient_global_slot,
                },
            )


def _covering_emitters(
    emitter_by_beneficiary: tuple[tuple[bool, ...], ...],
    beneficiary_global_slot: int,
) -> tuple[int, ...]:
    """Join one beneficiary to every direct covering emitter in slot order."""
    return tuple(
        emitter_global_slot
        for emitter_global_slot, beneficiary_row in enumerate(emitter_by_beneficiary)
        if beneficiary_row[beneficiary_global_slot]
    )


def _append_health_output_candidates(
    candidates: list[_EventCandidate],
    facts: TransitionFactsV1,
) -> None:
    """Emit positive raw source outputs and affected recipient resolutions."""
    combat = facts.combat_transition_facts
    aura = facts.aura_facts
    for source_global_slot, recipient_global_slot in enumerate(
        combat.combat_effect_recipient_global_slot_by_source
    ):
        raw_damage_output = combat.raw_damage_output_by_source[source_global_slot]
        if raw_damage_output > 0.0:
            damage_recipient_global_slot = _require_routed_recipient(
                recipient_global_slot,
                relation="positive raw damage output",
            )
            mage_emitters = _covering_emitters(
                aura.is_covered_by_mage_damage_aura_by_emitter_and_beneficiary,
                source_global_slot,
            )
            warrior_emitters = _covering_emitters(
                aura.is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary,
                damage_recipient_global_slot,
            )
            _append_candidate(
                candidates,
                phase_rank=30,
                primary_slot_or_team_index=source_global_slot,
                secondary_slot_or_status_channel=_recipient_sort_index(
                    damage_recipient_global_slot
                ),
                subtype_rank=0,
                source_slot=source_global_slot,
                model_type=SourceDamageOutputEventV1,
                payload={
                    "source_global_slot": source_global_slot,
                    "recipient_global_slot": damage_recipient_global_slot,
                    "raw_damage_output": raw_damage_output,
                    "source_modified_damage_output": (
                        combat.source_modified_damage_output_by_source[
                            source_global_slot
                        ]
                    ),
                    "recipient_damage_modifier": (
                        combat.recipient_damage_modifier_by_source[source_global_slot]
                    ),
                    "mage_damage_aura_covering_emitter_global_slots": (mage_emitters),
                    "warrior_mitigation_aura_covering_emitter_global_slots": (
                        warrior_emitters
                    ),
                },
            )

        raw_healing_output = combat.raw_healing_output_by_source[source_global_slot]
        if raw_healing_output > 0.0:
            healing_recipient_global_slot = _require_routed_recipient(
                recipient_global_slot,
                relation="positive raw healing output",
            )
            _append_candidate(
                candidates,
                phase_rank=30,
                primary_slot_or_team_index=source_global_slot,
                secondary_slot_or_status_channel=_recipient_sort_index(
                    healing_recipient_global_slot
                ),
                subtype_rank=1,
                source_slot=source_global_slot,
                model_type=SourceHealingOutputEventV1,
                payload={
                    "source_global_slot": source_global_slot,
                    "recipient_global_slot": healing_recipient_global_slot,
                    "raw_healing_output": raw_healing_output,
                    "source_modified_healing_output": (
                        combat.source_modified_healing_output_by_source[
                            source_global_slot
                        ]
                    ),
                    "recipient_healing_modifier": (
                        combat.recipient_healing_modifier_by_source[source_global_slot]
                    ),
                },
            )


def _append_health_resolution_candidates(
    candidates: list[_EventCandidate],
    start_frame: EvaluationFrameV1,
    facts: TransitionFactsV1,
) -> None:
    """Join affected recipient totals to transition-start health once."""
    combat = facts.combat_transition_facts
    for recipient_global_slot, (
        total_effective_damage,
        total_effective_healing,
        health_after_combat_resolution,
        transition_start_health,
    ) in enumerate(
        zip(
            combat.total_effective_damage_by_recipient,
            combat.total_effective_healing_by_recipient,
            combat.health_after_combat_resolution_by_recipient,
            start_frame.snapshot.current_health,
            strict=True,
        )
    ):
        if total_effective_damage <= 0.0 and total_effective_healing <= 0.0:
            continue
        _append_candidate(
            candidates,
            phase_rank=40,
            primary_slot_or_team_index=recipient_global_slot,
            model_type=RecipientHealthResolutionEventV1,
            payload={
                "recipient_global_slot": recipient_global_slot,
                "transition_start_health": transition_start_health,
                "total_effective_damage": total_effective_damage,
                "total_effective_healing": total_effective_healing,
                "health_after_combat_resolution": health_after_combat_resolution,
                "realized_net_health_change": (
                    health_after_combat_resolution - transition_start_health
                ),
            },
        )


def _append_regeneration_candidates(
    candidates: list[_EventCandidate],
    facts: TransitionFactsV1,
) -> None:
    """Emit direct combat-countdown reset and realized regeneration facts."""
    regeneration = facts.regeneration_facts
    for agent_global_slot, (
        combat_countdown_was_reset,
        actual_health_regenerated,
    ) in enumerate(
        zip(
            regeneration.combat_countdown_was_reset_by_agent,
            regeneration.actual_health_regenerated_this_step_by_agent,
            strict=True,
        )
    ):
        if combat_countdown_was_reset:
            _append_candidate(
                candidates,
                phase_rank=50,
                primary_slot_or_team_index=agent_global_slot,
                subtype_rank=0,
                model_type=CombatCountdownResetEventV1,
                payload={"agent_global_slot": agent_global_slot},
            )
        if actual_health_regenerated > 0.0:
            _append_candidate(
                candidates,
                phase_rank=50,
                primary_slot_or_team_index=agent_global_slot,
                subtype_rank=1,
                model_type=HealthRegeneratedEventV1,
                payload={
                    "agent_global_slot": agent_global_slot,
                    "actual_health_regenerated": actual_health_regenerated,
                },
            )


def _append_cooldown_candidates(
    candidates: list[_EventCandidate],
    start_frame: EvaluationFrameV1,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrameV1,
) -> None:
    """Emit accepted starts and direct adjacent positive-to-zero readiness."""
    accepted_use_ultimate = (
        facts.action_acceptance_facts.accepted_joint_action.use_ultimate
    )
    ultimate_activated = (
        facts.combat_transition_facts.ultimate_effect_is_activated_by_source
    )
    for agent_global_slot, (
        accepted_ultimate_action,
        has_ultimate_activation,
        start_cooldown,
        successor_cooldown,
    ) in enumerate(
        zip(
            accepted_use_ultimate,
            ultimate_activated,
            start_frame.snapshot.ultimate_cooldowns,
            successor_frame.snapshot.ultimate_cooldowns,
            strict=True,
        )
    ):
        cooldown_started = accepted_ultimate_action == 1
        if cooldown_started != has_ultimate_activation:
            raise ValueError("accepted Ultimate action and activation fact must agree")
        if cooldown_started:
            _append_candidate(
                candidates,
                phase_rank=60,
                primary_slot_or_team_index=agent_global_slot,
                subtype_rank=0,
                model_type=CooldownStartedEventV1,
                payload={"agent_global_slot": agent_global_slot},
            )
        if start_cooldown > 0 and successor_cooldown == 0:
            _append_candidate(
                candidates,
                phase_rank=60,
                primary_slot_or_team_index=agent_global_slot,
                subtype_rank=1,
                model_type=CooldownReadyEventV1,
                payload={"agent_global_slot": agent_global_slot},
            )


def _append_displacement_candidates(
    candidates: list[_EventCandidate],
    facts: TransitionFactsV1,
) -> None:
    """Emit each nonzero phase-authored displacement without reprojection."""
    displacement_families = (
        (
            facts.physical_facts.charge_phase_displacement_by_agent,
            70,
            ChargePhaseDisplacementEventV1,
        ),
        (
            facts.physical_facts.ordinary_movement_phase_displacement_by_agent,
            80,
            OrdinaryMovementPhaseDisplacementEventV1,
        ),
    )
    for displacement_by_agent, phase_rank, model_type in displacement_families:
        for agent_global_slot, realized_displacement in enumerate(
            displacement_by_agent
        ):
            if realized_displacement == (0.0, 0.0):
                continue
            _append_candidate(
                candidates,
                phase_rank=phase_rank,
                primary_slot_or_team_index=agent_global_slot,
                model_type=model_type,
                payload={
                    "agent_global_slot": agent_global_slot,
                    "realized_displacement": realized_displacement,
                },
            )


def _append_death_candidates(
    candidates: list[_EventCandidate],
    facts: TransitionFactsV1,
) -> None:
    """Emit recipient death before one atomic event per direct contributor."""
    death = facts.death_facts
    combat = facts.combat_transition_facts
    for recipient_global_slot, is_newly_dead in enumerate(
        death.is_newly_dead_by_recipient
    ):
        if is_newly_dead:
            _append_candidate(
                candidates,
                phase_rank=90,
                primary_slot_or_team_index=recipient_global_slot,
                subtype_rank=0,
                model_type=AgentDiedEventV1,
                payload={"recipient_global_slot": recipient_global_slot},
            )

    for source_global_slot, (
        contributed_to_new_death,
        attributed_death_damage,
        recipient_global_slot,
    ) in enumerate(
        zip(
            death.contributed_to_new_death_by_source,
            death.attributed_death_damage_by_source,
            combat.combat_effect_recipient_global_slot_by_source,
            strict=True,
        )
    ):
        if contributed_to_new_death != (attributed_death_damage > 0.0):
            raise ValueError(
                "death contributor flag and positive attributed damage must agree"
            )
        if not contributed_to_new_death:
            continue
        routed_recipient_global_slot = _require_routed_recipient(
            recipient_global_slot,
            relation="lethal damage contribution",
        )
        if not death.is_newly_dead_by_recipient[routed_recipient_global_slot]:
            raise ValueError(
                "lethal damage contribution must route to a newly dead recipient"
            )
        _append_candidate(
            candidates,
            phase_rank=90,
            primary_slot_or_team_index=routed_recipient_global_slot,
            subtype_rank=1,
            source_slot=source_global_slot,
            model_type=LethalDamageContributionEventV1,
            payload={
                "source_global_slot": source_global_slot,
                "recipient_global_slot": routed_recipient_global_slot,
                "attributed_death_damage": attributed_death_damage,
            },
        )


def _status_application_flags_by_source(
    facts: TransitionFactsV1,
    source_global_slot: int,
) -> tuple[bool, ...]:
    """Project distinct core application leaves into their catalog channel axis."""
    combat = facts.combat_transition_facts
    return (
        *combat.slow_is_applied_by_source_and_channel[source_global_slot],
        *combat.stun_is_applied_by_source_and_channel[source_global_slot],
        combat.rogue_poison_anti_heal_is_applied_by_source[source_global_slot],
        combat.mage_burst_damage_amplification_is_applied_by_source[source_global_slot],
        combat.priest_blessing_of_freedom_is_applied_by_source[source_global_slot],
    )


def _validate_status_application_source(
    context: EvaluationEpisodeContextV1,
    facts: TransitionFactsV1,
    source_global_slot: int,
    status_channel: int,
) -> None:
    """Require one application fact to join its catalog class and activation."""
    status_mechanic = context.static_mechanics_catalog.status_channels[status_channel]
    roster_class_id = context.roster[source_global_slot].class_id
    if roster_class_id != status_mechanic.source_class_id:
        raise ValueError("status application source class must match its catalog row")
    combat = facts.combat_transition_facts
    activation_by_source = (
        combat.basic_effect_is_activated_by_source
        if status_mechanic.source_action_component == "basic"
        else combat.ultimate_effect_is_activated_by_source
    )
    if not activation_by_source[source_global_slot]:
        raise ValueError(
            "status application requires its catalog-named ability activation"
        )


def _append_status_candidates(
    candidates: list[_EventCandidate],
    context: EvaluationEpisodeContextV1,
    facts: TransitionFactsV1,
) -> None:
    """Emit independent lifecycle causes and direct source applications."""
    lifecycle = facts.status_lifecycle_facts
    lifecycle_families = (
        (
            lifecycle.aged_to_zero_by_recipient_and_status_channel,
            0,
            StatusAgedToZeroEventV1,
        ),
        (
            lifecycle.broken_by_damage_by_recipient_and_status_channel,
            1,
            StatusBrokenByDamageEventV1,
        ),
        (
            lifecycle.refreshed_or_extended_by_recipient_and_status_channel,
            3,
            StatusRefreshedOrExtendedEventV1,
        ),
        (
            lifecycle.cleared_by_new_death_by_recipient_and_status_channel,
            4,
            StatusClearedByNewDeathEventV1,
        ),
    )
    status_catalog = context.static_mechanics_catalog.status_channels
    for cause_by_recipient_and_channel, subtype_rank, model_type in lifecycle_families:
        for recipient_global_slot, cause_by_status_channel in enumerate(
            cause_by_recipient_and_channel
        ):
            for status_channel, has_cause in enumerate(cause_by_status_channel):
                if not has_cause:
                    continue
                _append_candidate(
                    candidates,
                    phase_rank=100,
                    primary_slot_or_team_index=recipient_global_slot,
                    secondary_slot_or_status_channel=status_channel,
                    subtype_rank=subtype_rank,
                    model_type=model_type,
                    payload={
                        "recipient_global_slot": recipient_global_slot,
                        "status_channel": status_channel,
                        "status_id": status_catalog[status_channel].status_id,
                    },
                )

    routed_recipient_by_source = (
        facts.combat_transition_facts.combat_effect_recipient_global_slot_by_source
    )
    for source_global_slot, routed_recipient_global_slot in enumerate(
        routed_recipient_by_source
    ):
        for status_channel, is_applied in enumerate(
            _status_application_flags_by_source(facts, source_global_slot)
        ):
            if not is_applied:
                continue
            _validate_status_application_source(
                context,
                facts,
                source_global_slot,
                status_channel,
            )
            recipient_global_slot = (
                source_global_slot
                if status_channel == _MAGE_BURST_STATUS_CHANNEL
                else _require_routed_recipient(
                    routed_recipient_global_slot,
                    relation="status application",
                )
            )
            _append_candidate(
                candidates,
                phase_rank=100,
                primary_slot_or_team_index=recipient_global_slot,
                secondary_slot_or_status_channel=status_channel,
                subtype_rank=2,
                source_slot=source_global_slot,
                model_type=StatusAppliedEventV1,
                payload={
                    "source_global_slot": source_global_slot,
                    "recipient_global_slot": recipient_global_slot,
                    "status_channel": status_channel,
                    "status_id": status_catalog[status_channel].status_id,
                },
            )


def _append_respawn_candidates(
    candidates: list[_EventCandidate],
    context: EvaluationEpisodeContextV1,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrameV1,
) -> None:
    """Emit shield expiry, due waves, and realized successor respawns."""
    for agent_global_slot, expired in enumerate(
        facts.spawn_shield_facts.expired_at_transition_end_by_agent
    ):
        if expired:
            _append_candidate(
                candidates,
                phase_rank=110,
                primary_slot_or_team_index=agent_global_slot,
                model_type=SpawnShieldExpiredEventV1,
                payload={"agent_global_slot": agent_global_slot},
            )

    for team_index, wave_occurred in enumerate(
        facts.respawn_facts.respawn_wave_occurred_this_transition_by_team
    ):
        if wave_occurred:
            _append_candidate(
                candidates,
                phase_rank=120,
                primary_slot_or_team_index=team_index,
                secondary_slot_or_status_channel=-1,
                subtype_rank=0,
                model_type=RespawnWaveOccurredEventV1,
                payload={"team_index": team_index, "team_id": team_index + 1},
            )

    for agent_global_slot, was_respawned in enumerate(
        facts.respawn_facts.was_respawned_this_transition_by_agent
    ):
        if not was_respawned:
            continue
        team_id = context.roster[agent_global_slot].configured_team_id
        if team_id not in (1, 2):
            raise ValueError("respawned agent must join an active roster team")
        if not successor_frame.snapshot.alive_mask[agent_global_slot]:
            raise ValueError("respawned agent must be alive in the successor frame")
        if not facts.respawn_facts.respawn_wave_occurred_this_transition_by_team[
            team_id - 1
        ]:
            raise ValueError("respawned agent must join its configured team's wave")
        _append_candidate(
            candidates,
            phase_rank=120,
            primary_slot_or_team_index=team_id - 1,
            secondary_slot_or_status_channel=agent_global_slot,
            subtype_rank=1,
            model_type=AgentRespawnedEventV1,
            payload={
                "agent_global_slot": agent_global_slot,
                "team_id": team_id,
                "realized_successor_position": (
                    successor_frame.snapshot.agent_positions[agent_global_slot]
                ),
            },
        )


def _derive_team_deathmatch_authority_v1(
    context: EvaluationEpisodeContextV1,
    start_frame: EvaluationFrameV1,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrameV1,
) -> _TeamDeathmatchAuthorityV1:
    """Join TDM score, death, outcome, topology, and horizon authority."""
    task_mode = context.resolved_env_config.task_mode
    start_scores = start_frame.snapshot.team_deathmatch_scores
    successor_scores = successor_frame.snapshot.team_deathmatch_scores
    outcome = facts.team_deathmatch_facts.outcome

    if task_mode != _TEAM_DEATHMATCH_TASK_MODE:
        if start_scores != (0, 0) or successor_scores != (0, 0):
            raise ValueError("non-Team-Deathmatch snapshots require zero team scores")
        if outcome != _OUTCOME_ONGOING:
            raise ValueError("non-Team-Deathmatch facts require an ongoing TDM outcome")
        if (
            start_frame.simulator_step_count
            >= context.resolved_env_config.maximum_episode_steps
        ):
            raise ValueError("task-neutral transitions cannot start after completion")
        return _TeamDeathmatchAuthorityV1(
            score_increments=(0, 0),
            outcome=_OUTCOME_ONGOING,
            threshold_reached=False,
            horizon_reached=(
                successor_frame.simulator_step_count
                >= context.resolved_env_config.maximum_episode_steps
            ),
            completion_basis=None,
        )

    score_threshold = context.resolved_env_config.team_deathmatch_score_threshold
    if (
        start_frame.simulator_step_count
        >= context.resolved_env_config.maximum_episode_steps
        or any(score >= score_threshold for score in start_scores)
    ):
        raise ValueError("Team Deathmatch transitions cannot start after completion")

    expected_score_increments = [0, 0]
    for is_newly_dead, roster_row in zip(
        facts.death_facts.is_newly_dead_by_recipient,
        context.roster,
        strict=True,
    ):
        if not is_newly_dead or not roster_row.configured_active:
            continue
        if roster_row.configured_team_id == 1:
            expected_score_increments[1] += 1
        elif roster_row.configured_team_id == 2:
            expected_score_increments[0] += 1
        else:
            raise ValueError("configured active TDM roster slots require team 1 or 2")

    actual_score_increments = tuple(
        successor_score - start_score
        for start_score, successor_score in zip(
            start_scores,
            successor_scores,
            strict=True,
        )
    )
    if actual_score_increments != tuple(expected_score_increments):
        raise ValueError(
            "Team Deathmatch score edges must equal newly dead configured opponents"
        )

    threshold_reached = any(score >= score_threshold for score in successor_scores)
    horizon_reached = (
        successor_frame.simulator_step_count
        >= context.resolved_env_config.maximum_episode_steps
    )
    if threshold_reached:
        if successor_scores[0] > successor_scores[1]:
            expected_outcome = _OUTCOME_TEAM_A_WIN
        elif successor_scores[1] > successor_scores[0]:
            expected_outcome = _OUTCOME_TEAM_B_WIN
        else:
            expected_outcome = _OUTCOME_DRAW
    elif horizon_reached:
        expected_outcome = _OUTCOME_DRAW
    else:
        expected_outcome = _OUTCOME_ONGOING
    if outcome != expected_outcome:
        raise ValueError(
            "Team Deathmatch outcome fact conflicts with successor score authority"
        )

    completion_basis: str | None = None
    if threshold_reached and horizon_reached:
        completion_basis = "score_threshold_at_horizon"
    elif threshold_reached:
        completion_basis = "score_threshold"
    elif horizon_reached:
        completion_basis = "horizon"
    return _TeamDeathmatchAuthorityV1(
        score_increments=cast(tuple[int, int], actual_score_increments),
        outcome=outcome,
        threshold_reached=threshold_reached,
        horizon_reached=horizon_reached,
        completion_basis=completion_basis,
    )


def _append_team_deathmatch_candidates(
    candidates: list[_EventCandidate],
    context: EvaluationEpisodeContextV1,
    start_frame: EvaluationFrameV1,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrameV1,
) -> None:
    """Emit authoritative score edges followed by the sole completion event."""
    authority = _derive_team_deathmatch_authority_v1(
        context,
        start_frame,
        facts,
        successor_frame,
    )
    for team_index, score_increment in enumerate(authority.score_increments):
        if score_increment <= 0:
            continue
        _append_candidate(
            candidates,
            phase_rank=130,
            primary_slot_or_team_index=team_index,
            model_type=TeamDeathmatchScoreChangedEventV1,
            payload={
                "team_index": team_index,
                "team_id": team_index + 1,
                "score_increment": score_increment,
                "previous_score": (
                    start_frame.snapshot.team_deathmatch_scores[team_index]
                ),
                "successor_score": (
                    successor_frame.snapshot.team_deathmatch_scores[team_index]
                ),
            },
        )

    if authority.outcome == _OUTCOME_ONGOING:
        return
    outcome_name_by_value = {
        _OUTCOME_TEAM_A_WIN: "team_a_win",
        _OUTCOME_TEAM_B_WIN: "team_b_win",
        _OUTCOME_DRAW: "draw",
    }
    if authority.completion_basis is None:
        raise ValueError("terminal Team Deathmatch outcome requires a completion basis")
    _append_candidate(
        candidates,
        phase_rank=140,
        primary_slot_or_team_index=0,
        model_type=TeamDeathmatchCompletedEventV1,
        payload={
            "outcome": outcome_name_by_value[authority.outcome],
            "completion_basis": authority.completion_basis,
        },
    )


def decode_evaluation_events_v1(
    context: EvaluationEpisodeContextV1,
    start_frame: EvaluationFrameV1,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrameV1,
) -> tuple[EvaluationEventV1, ...]:
    """Decode one fact record into a deterministic tuple of atomic V1 events."""
    _validate_decoder_inputs(context, start_frame, facts, successor_frame)
    candidates: list[_EventCandidate] = []
    _append_action_candidates(candidates, context, facts)
    _append_health_output_candidates(candidates, facts)
    _append_health_resolution_candidates(candidates, start_frame, facts)
    _append_regeneration_candidates(candidates, facts)
    _append_cooldown_candidates(candidates, start_frame, facts, successor_frame)
    _append_displacement_candidates(candidates, facts)
    _append_death_candidates(candidates, facts)
    _append_status_candidates(candidates, context, facts)
    _append_respawn_candidates(candidates, context, facts, successor_frame)
    _append_team_deathmatch_candidates(
        candidates,
        context,
        start_frame,
        facts,
        successor_frame,
    )

    candidates.sort(key=lambda candidate: candidate.sort_key)
    transition_id = (
        f"{context.identity.episode_id}:transition:{start_frame.frame_index}"
    )
    events: list[EvaluationEventV1] = []
    for ordinal, candidate in enumerate(candidates):
        event = candidate.model_type.model_validate(
            {
                "transition_id": transition_id,
                "ordinal": ordinal,
                "event_id": f"{transition_id}:event:{ordinal:04d}",
                "phase_rank": candidate.phase_rank,
                **candidate.payload,
            }
        )
        events.append(cast(EvaluationEventV1, event))
    return tuple(events)


__all__ = ["decode_evaluation_events_v1"]
