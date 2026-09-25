"""Decode recorded transition facts into stable, ordered host events.

The decoder joins adjacent frames with Core-authored facts and recorded class
rules. It does not step the simulator or use fresh physics calculations. Event
order is a serialization convention: ranks group related events and stable slot
coordinates break ties. The order does not create extra policy decision epochs.
Team Deathmatch points are re-derived from the recorded start positions and the
recorded configuration (the Red Zone rule, when the record holds a depth): a
rule check, not physics. Renderers read recorded facts and never re-run it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from marl_battlegrounds.evaluation.models import (
    TEAM_DEATHMATCH_POINTS_PER_DEATH,
    TEAM_DEATHMATCH_POINTS_PER_RED_ZONE_DEATH,
    AbilityActivatedEventV1,
    ActionRejectedEventV1,
    AgentDiedEventV1,
    AgentLeftCombatEventV1,
    AgentRespawnedEventV1,
    ChargePhaseDisplacementEventV1,
    CombatCountdownResetEventV1,
    CooldownReadyEventV1,
    CooldownStartedEventV1,
    EvaluationEpisodeContext,
    EvaluationEventBaseV1,
    EvaluationEventV1,
    EvaluationFrame,
    HealthRegeneratedEventV1,
    LethalDamageContributionEventV1,
    OrdinaryMovementPhaseDisplacementEventV1,
    RecipientHealthResolutionEventV1,
    ResolvedEnvConfigV2,
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
    red_zone_team_on_right,
    red_zone_x_range,
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
    """Hold checked task results for one adjacent pair of frames.

    Attributes
    ----------
    score_increments : tuple[int, int]
        Team A and Team B point gains during this transition.
    outcome : int
        0 ongoing, 1 Team A win, 2 Team B win, or 3 draw.
    threshold_reached : bool
        Whether a successor score reaches the configured threshold.
    horizon_reached : bool
        Whether the successor tick reaches the configured horizon.
    completion_basis : str | None
        score_threshold, horizon, score_threshold_at_horizon, or None.

    Notes
    -----
    Frozen internal record. Non-TDM transitions have zero gains, ongoing outcome,
    and no completion basis, but still report horizon_reached.
    """

    score_increments: tuple[int, int]
    outcome: int
    threshold_reached: bool
    horizon_reached: bool
    completion_basis: str | None


@dataclass(frozen=True, slots=True)
class _EventCandidate:
    """Keep an event payload with coordinates used before assigning its final ID.

    Attributes
    ----------
    phase_rank : int
        Primary event-family ordering rank.
    primary_slot_or_team_index : int
        First slot/team ordering coordinate within the rank.
    secondary_slot_or_status_channel : int
        Second coordinate, such as recipient or status.
    subtype_rank : int
        Tie-breaker among related event kinds.
    source_slot : int
        Final source-slot tie-breaker.
    model_type : type[EvaluationEventBaseV1]
        Strict event model used after sorting.
    payload : dict[str, object]
        Event-specific fields; identity, ordinal, and rank are added later.

    Notes
    -----
    The dataclass is frozen, but payload is still a dictionary owned by the decoder.
    """

    phase_rank: int
    primary_slot_or_team_index: int
    secondary_slot_or_status_channel: int
    subtype_rank: int
    source_slot: int
    model_type: type[EvaluationEventBaseV1]
    payload: dict[str, object]

    @property
    def sort_key(self) -> tuple[int, int, int, int, int]:
        """Return the five ordering coordinates from phase rank through source slot.

        The key excludes payload and model type, so canonical event order depends only
        on the declared numeric coordinates.
        """
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
    """Append one event candidate to the caller-owned list without assigning an ID.

    The optional secondary coordinate, subtype, and source slot default to zero.
    Payload is retained by reference until decoding builds the strict event model.
    """
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
    """Return a recipient slot, or 10 for None so missing recipients sort last.

    Concrete global slots are expected to be validated integers 0 through 9.
    """
    if recipient_global_slot is None:
        return _NULL_RECIPIENT_SORT_INDEX
    return recipient_global_slot


def _require_routed_recipient(
    recipient_global_slot: int | None,
    *,
    relation: str,
) -> int:
    """Return a concrete recipient slot or raise ValueError using the relation label.

    This only checks that the route is present; typed fact models own slot bounds.
    """
    if recipient_global_slot is None:
        raise ValueError(f"{relation} requires an authoritative recipient route")
    return recipient_global_slot


def _ultimate_activation_recipient(
    context: EvaluationEpisodeContext,
    source_global_slot: int,
    recipient_global_slot: int | None,
) -> int | None:
    """Check route presence against the source class's recorded Ultimate target mode.

    Target-none activation requires None; targeted activation requires a concrete
    route. Unavailable abilities and contradictory routes raise ValueError. This
    does not independently test distance, team relation, or action-mask legality.
    """
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
    context: EvaluationEpisodeContext,
    source_global_slot: int,
    recipient_global_slot: int | None,
) -> int:
    """Require an available Basic and concrete route using the recorded class catalog.

    Raise ValueError for an unavailable Basic or absent route. Distance, team
    relation, and action-mask legality are supplied by Core facts, not recomputed.
    """
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
    context: EvaluationEpisodeContext,
    start_frame: EvaluationFrame,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrame,
) -> None:
    """Check episode joins, adjacent indices/ticks, and real transition-start facts.

    Raise ValueError when the frames do not describe one direct transition. This
    is a narrow join check; full strict-tree validation belongs to validation.py.
    """
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
    context: EvaluationEpisodeContext,
    facts: TransitionFactsV1,
) -> None:
    """Append every flagged rejection and activated Basic/Ultimate to the event list.

    Keep domain, movement, and combat-pair rejection causes distinct, including the
    submitted three-head action. Activation routes must fit recorded class modes.
    Mutate candidates only; input frames and facts remain unchanged.
    """
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
    """Return source slots whose recorded aura row covers one beneficiary.

    Read the fixed emitter-by-beneficiary boolean matrix in ascending emitter order.
    The caller supplies a validated beneficiary slot; no distance is recomputed.
    """
    return tuple(
        emitter_global_slot
        for emitter_global_slot, beneficiary_row in enumerate(emitter_by_beneficiary)
        if beneficiary_row[beneficiary_global_slot]
    )


def _append_health_output_candidates(
    candidates: list[_EventCandidate],
    facts: TransitionFactsV1,
) -> None:
    """Append positive raw damage/healing outputs and their recorded modifiers.

    Require a recipient for each positive source output. Damage also records direct
    Mage emitters covering the source and Warrior emitters covering the recipient.
    These are gross output stages, not effective recipient health changes.
    """
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
    start_frame: EvaluationFrame,
    facts: TransitionFactsV1,
) -> None:
    """Append one health-resolution event per recipient with damage or healing.

    Join pre-transition health with recorded combat totals and post-combat health.
    The net change stops at combat resolution; later regeneration is separate.
    Recipients with neither positive damage nor positive healing produce no event.
    """
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
    start_frame: EvaluationFrame,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrame,
) -> None:
    """Append combat-countdown reset, exit, and positive regeneration events.

    An exit requires a living agent at both frames, no reset, and a countdown change
    from one to zero. Regeneration reports the recorded actual health restored, not
    a class maximum or an inferred source of healing.
    """
    regeneration = facts.regeneration_facts
    for agent_global_slot, (
        combat_countdown_was_reset,
        actual_health_regenerated,
        start_countdown,
        successor_countdown,
        start_alive,
        successor_alive,
    ) in enumerate(
        zip(
            regeneration.combat_countdown_was_reset_by_agent,
            regeneration.actual_health_regenerated_this_step_by_agent,
            start_frame.snapshot.steps_until_out_of_combat,
            successor_frame.snapshot.steps_until_out_of_combat,
            start_frame.snapshot.alive_mask,
            successor_frame.snapshot.alive_mask,
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
        if (
            not combat_countdown_was_reset
            and start_alive
            and successor_alive
            and start_countdown == 1
            and successor_countdown == 0
        ):
            _append_candidate(
                candidates,
                phase_rank=50,
                primary_slot_or_team_index=agent_global_slot,
                subtype_rank=1,
                model_type=AgentLeftCombatEventV1,
                payload={"agent_global_slot": agent_global_slot},
            )
        if actual_health_regenerated > 0.0:
            _append_candidate(
                candidates,
                phase_rank=50,
                primary_slot_or_team_index=agent_global_slot,
                subtype_rank=2,
                model_type=HealthRegeneratedEventV1,
                payload={
                    "agent_global_slot": agent_global_slot,
                    "actual_health_regenerated": actual_health_regenerated,
                },
            )


def _append_cooldown_candidates(
    candidates: list[_EventCandidate],
    start_frame: EvaluationFrame,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrame,
) -> None:
    """Append accepted Ultimate starts and positive-to-zero readiness changes.

    Require accepted Ultimate use to agree with the activation fact or raise
    ValueError. Start and readiness checks are independent; the decoder retains
    whatever valid adjacent facts report without recomputing cooldown rules.
    """
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
    """Append nonzero Charge and ordinary-movement displacement vectors separately.

    Use each phase's recorded world-unit vector in global-slot order. Do not infer
    the path from final positions or repeat collision projection.
    """
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
    """Append new deaths and one event for each direct positive damage contributor.

    Check that contributor flags match positive attributed damage and route to a
    newly dead recipient. Raise ValueError on a broken join. For each recipient,
    death sorts before its contributors; this is event order, not a new action phase.
    """
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
    """Pack one source's application flags into the catalog's nine-channel order.

    Combine three slow channels, three stun channels, Poison anti-heal, Burst
    amplification, and Freedom movement floor. The source slot is already validated.
    """
    combat = facts.combat_transition_facts
    return (
        *combat.slow_is_applied_by_source_and_channel[source_global_slot],
        *combat.stun_is_applied_by_source_and_channel[source_global_slot],
        combat.rogue_poison_anti_heal_is_applied_by_source[source_global_slot],
        combat.mage_burst_damage_amplification_is_applied_by_source[source_global_slot],
        combat.priest_blessing_of_freedom_is_applied_by_source[source_global_slot],
    )


def _validate_status_application_source(
    context: EvaluationEpisodeContext,
    facts: TransitionFactsV1,
    source_global_slot: int,
    status_channel: int,
) -> None:
    """Require a status application to match its catalog class and Basic/Ultimate.

    Raise ValueError when source class or the required activation flag disagrees.
    This checks recorded attribution, without applying the status again.
    """
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
    context: EvaluationEpisodeContext,
    facts: TransitionFactsV1,
) -> None:
    """Append independent status lifecycle causes and attributed applications.

    Sort by recipient/channel, then aging, damage break, application, refresh, and
    death clearing. Multiple causes can be present in one transition. Burst applies
    to its source; other applications require a routed recipient. Do not invent a
    caster for source-free lifecycle events.
    """
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
    context: EvaluationEpisodeContext,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrame,
) -> None:
    """Append shield expiry, team-wave events, and realized agent respawns.

    A respawn must join Team A or B, its team's wave, and a living successor slot;
    otherwise raise ValueError. Record the successor position in world units.
    A wave event can exist even when no agent respawns.
    """
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
    context: EvaluationEpisodeContext,
    start_frame: EvaluationFrame,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrame,
) -> _TeamDeathmatchAuthorityV1:
    """Check recorded scores and outcome against deaths, roster, and horizon.

    Each newly dead configured recipient gives the other team points: 2
    (TEAM_DEATHMATCH_POINTS_PER_RED_ZONE_DEATH) when its recorded start position
    is inside its own team's Red Zone, else 1. The depth is the recorded float32
    depth when the context holds ResolvedEnvConfigV2 and 0.0 for older records
    (one point per death, their original rule). The side comes from
    red_zone_team_on_right on the recorded pads and the strip from
    red_zone_x_range, both bounds inclusive. Score changes must equal these
    points exactly. At a threshold, higher score wins and equal scores draw; a
    horizon-only finish draws. Both completion conditions may hold. Reject
    post-terminal starts or inconsistent scores/outcomes with ValueError.
    Task-neutral records keep zero TDM scores and ongoing outcome while still
    tracking horizon completion.
    """
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

    config = context.resolved_env_config
    red_zone_ranges: tuple[tuple[float, float], ...] | None = None
    if (
        isinstance(config, ResolvedEnvConfigV2)
        and config.team_deathmatch_red_zone_depth > 0.0
    ):
        red_zone_ranges = tuple(
            red_zone_x_range(
                config.map_width,
                config.team_deathmatch_red_zone_depth,
                red_zone_team_on_right(config.map_width, [pad[0] for pad in team_pads]),
            )
            for team_pads in config.team_spawn_pad_positions
        )
    start_positions = start_frame.snapshot.agent_positions
    expected_score_increments = [0, 0]
    for slot, (is_newly_dead, roster_row) in enumerate(
        zip(
            facts.death_facts.is_newly_dead_by_recipient,
            context.roster,
            strict=True,
        )
    ):
        if not is_newly_dead or not roster_row.configured_active:
            continue
        team_id = roster_row.configured_team_id
        if team_id not in (1, 2):
            raise ValueError("configured active TDM roster slots require team 1 or 2")
        points = TEAM_DEATHMATCH_POINTS_PER_DEATH
        if red_zone_ranges is not None:
            low, high = red_zone_ranges[team_id - 1]
            if low <= start_positions[slot][0] <= high:
                points = TEAM_DEATHMATCH_POINTS_PER_RED_ZONE_DEATH
        # A Team A victim scores for Team B (index 1), and the reverse.
        expected_score_increments[2 - team_id] += points

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
            "Team Deathmatch score edges must equal points from newly dead "
            "configured opponents"
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
    context: EvaluationEpisodeContext,
    start_frame: EvaluationFrame,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrame,
) -> None:
    """Append positive team-score changes and at most one TDM completion event.

    Use the shared joined authority so score and outcome validation stay in one
    place. Task-neutral transitions emit neither kind of TDM event.
    """
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
    context: EvaluationEpisodeContext,
    start_frame: EvaluationFrame,
    facts: TransitionFactsV1,
    successor_frame: EvaluationFrame,
) -> tuple[EvaluationEventV1, ...]:
    """Decode one captured transition into deterministic atomic events.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Supported episode context with roster and recorded mechanics catalog.
    start_frame : EvaluationFrame
        Frame immediately before the transition.
    facts : TransitionFactsV1
        Real V1 transition facts whose start tick matches start_frame.
    successor_frame : EvaluationFrame
        Frame from the same episode, exactly one artifact index
        and simulator tick after start_frame.

    Returns
    -------
    tuple[EvaluationEventV1, ...]
        Tuple of strict V1 events sorted by family rank and stable slot coordinates.
        Ordinals start at zero; IDs use the episode, start-frame index, and a
        zero-padded event ordinal. No-event transitions return an empty tuple.

    Raises
    ------
    ValueError
        Episode/epoch joins, ability routes, status sources, cooldowns,
        deaths, respawns, task authority, or constructed event fields are inconsistent.

    Notes
    -----
    Runs entirely on the host and leaves inputs unchanged. Inputs are expected
    to be structurally validated wire models. Use validate_evaluation_transition_unit
    for complete strict-tree and frame/context checks. Event order organizes
    recorded facts; it does not allow actions between events or rerun physics.
    """
    _validate_decoder_inputs(context, start_frame, facts, successor_frame)
    candidates: list[_EventCandidate] = []
    _append_action_candidates(candidates, context, facts)
    _append_health_output_candidates(candidates, facts)
    _append_health_resolution_candidates(candidates, start_frame, facts)
    _append_regeneration_candidates(candidates, start_frame, facts, successor_frame)
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
