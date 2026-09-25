"""Build and advance one fixed-slot game with pure JAX transition code.

reset creates an ordinary pad-based start. step accepts actions using the
paired current mask, resolves simultaneous effects and movement phases, then
returns the successor state with matching decision inputs and transition facts.
initialize_scenario_state is the separate host-validated authored-start path.

Core owns action acceptance, observations, health/status/death/respawn rules,
task rewards and event facts. Geometry and combat catalogs remain shared
authorities in their own modules. Policies, native environment batching,
episode scheduling, logging, metrics collection and persistence live outside
this module. Numerical entry points do not auto-reset terminal states."""

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.axis_mappings import (
    GLOBAL_RECIPIENT_SLOT_INDEX_BY_ACTOR_AND_TARGET_ACTION as _ACTOR_RELATIVE_SELECT_TARGET_ACTION_TO_GLOBAL_AGENT_SLOT_LOOKUP_TABLE,  # noqa: E501
)
from marl_battlegrounds.core.axis_mappings import (
    TEAM_A_END,
    TEAM_A_START,
    TEAM_B_END,
    TEAM_B_START,
    spawn_bank_on_right,
)
from marl_battlegrounds.core.axis_mappings import (
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY as _JOINT_ACTION_MOVE_TO_DISPLACEMENT_LOOKUP_TABLE,  # noqa: E501
)
from marl_battlegrounds.core.combat import (
    BASIC_DAMAGE_BY_CLASS,
    BASIC_HEALING_BY_CLASS,
    HUNTER_BASIC_SLOW_DURATION_TICKS,
    HUNTER_BASIC_SLOW_MULTIPLIER,
    HUNTER_TRAP_STUN_DURATION_TICKS,
    MAGE_BURST_DAMAGE_DURATION_TICKS,
    MAGE_BURST_DAMAGE_MULTIPLIER,
    MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER,
    MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER_CEILING,
    MAGE_DAMAGE_AMPLIFICATION_AURA_RADIUS,
    ONLY_ALLY_TARGET_ULTIMATE_MODE,
    ONLY_ENEMY_TARGET_ULTIMATE_MODE,
    ONLY_NONE_TARGET_ULTIMATE_MODE,
    PRIEST_HEAL_SPEED_FLOOR,
    PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS,
    ROGUE_POISON_ANTI_HEAL_DURATION_TICKS,
    ROGUE_POISON_ANTI_HEAL_MULTIPLIER,
    ROGUE_POISON_SLOW_DURATION_TICKS,
    ROGUE_POISON_SLOW_MULTIPLIER,
    ROGUE_POISON_STUN_DURATION_TICKS,
    ULTIMATE_COOLDOWN_BY_CLASS,
    ULTIMATE_DAMAGE_BY_CLASS,
    ULTIMATE_HEALING_BY_CLASS,
    WARRIOR_CHARGE_SLOW_DURATION_TICKS,
    WARRIOR_CHARGE_SLOW_MULTIPLIER,
    WARRIOR_CHARGE_STUN_DURATION_TICKS,
    WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER,
    WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER_FLOOR,
    WARRIOR_DAMAGE_MITIGATION_AURA_RADIUS,
    build_rogue_poison_anti_heal_multipliers,
    derive_effective_movement_speeds,
    derive_status_magnitudes,
    get_basic_damage_by_class_ids,
    get_basic_healing_by_class_ids,
    get_ultimate_cooldown_by_class_ids,
    get_ultimate_damage_by_class_ids,
    get_ultimate_healing_by_class_ids,
    get_ultimate_target_mode_by_class_ids,
)
from marl_battlegrounds.core.config import (
    validate_env_config,
    validate_scenario_initial_state,
)
from marl_battlegrounds.core.geometry import (
    has_clear_line_of_sight,
    project_charge_endpoints_with_geometry,
    project_movement_with_geometry,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_IS_ENEMY,
    CONTEXT_FEATURE_ALLY_TEAM_SIZE,
    CONTEXT_FEATURE_CURRENT_TIMESTEP,
    CONTEXT_FEATURE_ENEMY_TEAM_SIZE,
    CONTEXT_FEATURE_IS_CTF,
    CONTEXT_FEATURE_IS_TDM,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_TDM_ALLY_SCORE,
    CONTEXT_FEATURE_TDM_ENEMY_SCORE,
    CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH,
    CONTEXT_FEATURE_TDM_SCORE_THRESHOLD,
    CONTEXT_FEATURES,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBJECTIVE_SLOTS,
    MAX_OBSTACLE_SLOTS,
    MOVE_STAY,
    NEUTRAL_CLASS_ID,
    NO_TEAM_ID,
    NUM_MOVE_ACTIONS,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    NUM_TARGET_ACTIONS,
    NUM_TASKS,
    NUM_TEAMS,
    NUM_ULTIMATE_ACTIONS,
    OBJECTIVE_FEATURES,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    REWARD_FOR_LOSING,
    REWARD_FOR_WINNING,
    ROGUE_CLASS_ID,
    SLOW_CHANNEL_HUNTER_BASIC,
    SLOW_CHANNEL_ROGUE_POISON,
    SLOW_CHANNEL_WARRIOR_CHARGE,
    STUN_CHANNEL_HUNTER_TRAP,
    STUN_CHANNEL_ROGUE_POISON,
    STUN_CHANNEL_WARRIOR_CHARGE,
    TASK_MODE_CTF,
    TASK_MODE_KOTH,
    TASK_MODE_OUTCOME_DRAW,
    TASK_MODE_OUTCOME_ONGOING,
    TASK_MODE_OUTCOME_TEAM_A_WIN,
    TASK_MODE_OUTCOME_TEAM_B_WIN,
    TASK_MODE_TDM,
    TEAM_A_ID,
    TEAM_B_ID,
    TEAM_DEATHMATCH_POINTS_PER_DEATH,
    TEAM_DEATHMATCH_POINTS_PER_RED_ZONE_DEATH,
    UNIT_FEATURES,
    WARRIOR_CLASS_ID,
    Action,
    ActionAcceptanceFacts,
    ActionMask,
    AuraTransitionFacts,
    CombatTransitionFacts,
    DeathTransitionFacts,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    PhysicalTransitionFacts,
    PreviousTimestepActionObservation,
    RegenerationTransitionFacts,
    RespawnTransitionFacts,
    Reward,
    SpawnLifecycleObservation,
    SpawnShieldTransitionFacts,
    StatusLifecycleTransitionFacts,
    TeamDeathmatchTransitionFacts,
    TransitionFacts,
)

# Private Helpers ---

_GLOBAL_AGENT_SLOT_INDICES = jnp.arange(MAX_AGENT_SLOTS, dtype=jnp.int32)


class _CombatEffectAggregationResult(NamedTuple):
    """Share accepted effect totals between health updates and transition facts.

    Fields describe one current transition in fixed global-slot order. Boolean
    source/recipient vectors have shape (10,); damage, healing and modifier
    vectors are float32 (10,). Source fields retain who applied an effect;
    recipient totals retain the sums used for simultaneous health resolution.
    The record is immutable, requires every field and performs no validation.
    """

    hunter_basic_slow_applied_this_tick_by_global_recipient_slot: Array
    """Accepted Hunter Basic slow recipients.

    Bool (10,) in global recipient order; used to merge successor durations.
    """
    priest_freedom_applied_this_tick_by_global_recipient_slot: Array
    """Accepted Priest Basic movement-floor recipients.

    Bool (10,) in global recipient order; used to merge successor durations.
    """
    accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot: Array
    """Recipients of accepted positive raw damage.

    Bool (10,). This breaks the aged remainder of an old Hunter Trap, independently
    of net health loss.
    """
    hunter_basic_slow_applied_this_tick_by_global_actor_slot: Array
    """Hunter sources that applied an accepted Basic slow.

    Bool (10,) retained by source for application facts.
    """
    basic_effect_is_activated_by_source: Array
    """Whether each source applies an accepted targeted Basic.

    Bool (10,) in global source order. Target None or accepted Ultimate use makes
    this False.
    """
    ultimate_effect_is_activated_by_source: Array
    """Whether each source uses an accepted Ultimate.

    Bool (10,), including no-target Mage Burst. Activation does not imply a routed
    health recipient.
    """
    raw_damage_output_by_source: Array
    """Accepted catalog damage before any source or recipient modifier.

    Float32 (10,) nonnegative health units. Sources without a damage payload are
    zero.
    """
    source_modified_damage_output_by_source: Array
    """Damage after source Burst and Mage aura modifiers.

    Float32 (10,) nonnegative health units, before recipient mitigation or health
    clipping.
    """
    recipient_damage_modifier_by_source: Array
    """The chosen recipient's damage factor for each damage source.

    Float32 (10,) dimensionless Warrior mitigation factors. Sources without positive
    raw routed damage use zero, not an identity factor.
    """
    total_effective_damage_by_recipient: Array
    """Gross incoming damage after source and recipient modifiers.

    Float32 (10,) nonnegative health units in global recipient order. Totals precede
    net healing and health clipping; they are not realized health loss.
    """
    raw_healing_output_by_source: Array
    """Accepted catalog healing before recipient modification.

    Float32 (10,) nonnegative health units. Sources without a healing payload are
    zero.
    """
    source_modified_healing_output_by_source: Array
    """Healing after source modification and before recipient modification.

    Float32 (10,) nonnegative health units. Current mechanics have no healing
    amplifier, so this equals raw healing.
    """
    recipient_healing_modifier_by_source: Array
    """The chosen recipient's healing factor for each healing source.

    Float32 (10,) dimensionless current Poison factors. Sources without positive raw
    routed healing use zero.
    """
    total_effective_healing_by_recipient: Array
    """Gross incoming healing after recipient modifiers.

    Float32 (10,) nonnegative health units in global recipient order, before net
    damage and maximum-health clipping.
    """
    priest_blessing_of_freedom_is_applied_by_source: Array
    """Accepted Priest Basic movement-floor applications.

    Bool (10,) in global source order, routed to the source's accepted recipient.
    """
    is_combat_participant_this_tick_by_source: Array
    """Actors whose accepted combat participation resets the delay.

    Bool (10,) covering damage sources/recipients and healing routes whose recipient
    was already in combat at transition start.
    """


class _CombatStatusAggregationResult(NamedTuple):
    """Carry successor status durations and the causes recorded during this step.

    Next slow and stun durations are int32 (10, 3); other next durations are
    int32 (10,). Source/channel application masks are bool (10, 3), and source
    flags are bool (10,). status_lifecycle_facts keeps the separate recipient
    causes. Applications remain recorded even when death clears successor
    durations. This immutable record requires every field and validates nothing.
    """

    next_slow_durations: Array
    """Successor slow memory after ageing, applications and death clearing.

    Int32 (10, 3), in Warrior Charge, Hunter Basic, Rogue Poison order.
    """
    next_stun_durations: Array
    """Successor stun memory after ageing, break, applications and death.

    Int32 (10, 3), in Warrior Charge, Hunter Trap, Rogue Poison order.
    """
    next_rogue_anti_heal_durations: Array
    """Successor Rogue Poison healing-reduction memory.

    Nonnegative int32 (10,) after ageing, applications and death clearing.
    """
    next_mage_burst_durations: Array
    """Successor Mage Burst memory.

    Nonnegative int32 (10,) after ageing, source-local applications and death
    clearing.
    """
    next_priest_freedom_slow_floor_durations: Array
    """Successor Priest movement-floor memory.

    Nonnegative int32 (10,) after ageing, applications and death clearing.
    """
    next_spawn_shield_durations: Array
    """Aged shield memory before the end-of-transition respawn override.

    Nonnegative int32 (10,). Dead and unused rows are zero.
    """
    slow_is_applied_by_source_and_channel: Array
    """Accepted slow applications by source and mechanic channel.

    Bool (10, 3), with Warrior Charge, Hunter Basic and Rogue Poison columns. These
    record applications, even if successor death clears the duration.
    """
    stun_is_applied_by_source_and_channel: Array
    """Accepted stun applications by source and mechanic channel.

    Bool (10, 3), with Warrior Charge, Hunter Trap and Rogue Poison columns. The
    chosen global recipient is stored separately.
    """
    rogue_poison_anti_heal_is_applied_by_source: Array
    """Accepted Rogue Poison anti-heal applications.

    Bool (10,) in global source order, routed to the source's accepted recipient.
    """
    mage_burst_damage_amplification_is_applied_by_source: Array
    """Accepted Mage Burst applications to the source itself.

    Bool (10,) in global source order. This self-buff has no routed recipient
    target.
    """
    status_lifecycle_facts: StatusLifecycleTransitionFacts
    """Separate recipient causes retained before final duration packaging."""


class _CombatAuraAggregationResult(NamedTuple):
    """Carry current aura strengths and the coverage that produced them.

    Mage and Warrior multipliers are float32 (10,) in global beneficiary order.
    aura_facts contains bool (10, 10) emitter-to-beneficiary relations from the
    same snapshot. Empty coverage yields multiplier one. The immutable record
    packages these required fields without validation or further reduction.
    """

    mage_damage_amplification_aura_multipliers: Array
    """Current bounded outgoing-damage factors by beneficiary.

    Float32 (10,) from one shared snapshot; no eligible aura yields 1.0.
    """
    warrior_damage_mitigation_aura_multipliers: Array
    """Current bounded incoming-damage factors by beneficiary.

    Float32 (10,) from one shared snapshot; no eligible aura yields 1.0.
    """
    aura_facts: AuraTransitionFacts
    """Coverage relations that produced the two bounded factor vectors."""


def _compute_global_pairwise_distances_from_agent_positions(
    agent_positions: Array,
) -> Array:
    """Return world distances for every pair of fixed global slots.

    Float32 agent_positions (10, 2) produce a float32 (10, 10) matrix with
    observer/source rows and candidate columns. Padding is not masked here;
    downstream visibility and interaction rules decide which pairs participate.
    """
    return cast(
        Array,
        jnp.linalg.norm(
            (agent_positions[None, :, :] - agent_positions[:, None, :]), axis=-1
        ),
    )


def _build_global_visibility_mask_and_distances(
    state: EnvState, config: EnvConfig
) -> tuple[Array, Array]:
    """Build current directed visibility and reuse its pairwise distances.

    state and matching config describe one ten-slot decision. Return bool
    visibility (10, 10), then float32 world distances (10, 10). Entry [i, j]
    describes observer i and candidate j. Both must be configured and alive,
    the candidate must be within i's observation radius, and static sight must
    be clear. Shielded opponents are hidden; self and allies still use ordinary
    range and sight rules. Returning distances avoids rebuilding them for masks.
    """
    # Pairwise observer-candidate validity from active/alive state.
    alive_active_mask = jnp.logical_and(
        config.agent_profile.active_mask, state.alive_mask
    )
    global_pairwise_validity_mask = jnp.logical_and(
        alive_active_mask[:, None],
        alive_active_mask[None, :],
    )

    global_pairwise_distances = _compute_global_pairwise_distances_from_agent_positions(
        state.agent_positions
    )

    # Observer-specific observation-radius check.
    observer_radii_bc = config.agent_profile.observation_radii[:, None]
    observation_radii_mask = global_pairwise_distances <= observer_radii_bc

    def _build_los_row(
        observer_center: Array,
        candidate_centers: Array,
        obstacles: Array,
    ) -> Array:
        """Return one observer's clear-sight flags for all ten candidate centers.

        observer_center is float32 (2,), candidate_centers is float32 (10, 2),
        and obstacles is the shared float32 (32, 8) table. Map the authoritative
        sight helper over candidates to return bool (10,); no radius gate is added.
        """
        candidate_los_vmap = jax.vmap(
            has_clear_line_of_sight,
            in_axes=(None, 0, None),
            out_axes=0,
        )
        return candidate_los_vmap(observer_center, candidate_centers, obstacles)

    observer_los_vmap = jax.vmap(
        _build_los_row,
        in_axes=(0, None, None),
        out_axes=0,
    )
    los_mask = observer_los_vmap(
        state.agent_positions,
        state.agent_positions,
        config.obstacles,
    )

    pre_spawn_shield_pairwise_global_visibility_mask = jnp.logical_and(
        global_pairwise_validity_mask,
        jnp.logical_and(observation_radii_mask, los_mask),
    )

    # Preserve ordinary self/ally visibility while hiding shielded opponents.
    canvas = jnp.ones_like(pre_spawn_shield_pairwise_global_visibility_mask)
    shielded_agents_by_global_slot = (
        state.spawn_shield_durations > 0
    )  # (MAX_AGENT_SLOTS)
    team_a_shielded_agents_by_global_slot = shielded_agents_by_global_slot[
        TEAM_A_START:TEAM_A_END
    ]
    team_b_shielded_agents_by_global_slot = shielded_agents_by_global_slot[
        TEAM_B_START:TEAM_B_END
    ]

    team_a_shielded_opponent_mask = jnp.logical_not(
        jnp.repeat(
            team_b_shielded_agents_by_global_slot[None, :], MAX_AGENTS_PER_TEAM, axis=0
        )
    )
    team_b_shielded_opponent_mask = jnp.logical_not(
        jnp.repeat(
            team_a_shielded_agents_by_global_slot[None, :], MAX_AGENTS_PER_TEAM, axis=0
        )
    )

    # Replace only the two opposing-team blocks of the directed visibility mask.
    shielded_opponent_mask = canvas.at[
        TEAM_A_START:TEAM_A_END, TEAM_B_START:TEAM_B_END
    ].set(team_a_shielded_opponent_mask)
    shielded_opponent_mask = shielded_opponent_mask.at[
        TEAM_B_START:TEAM_B_END, TEAM_A_START:TEAM_A_END
    ].set(team_b_shielded_opponent_mask)

    final_global_pairwise_visibility_mask = jnp.logical_and(
        pre_spawn_shield_pairwise_global_visibility_mask, shielded_opponent_mask
    )

    return (
        final_global_pairwise_visibility_mask,
        global_pairwise_distances,
    )


def _build_ally_enemy_masks(global_mask: Array) -> tuple[Array, Array]:
    """Reorder a global pair matrix into each observer's ally and enemy rows.

    global_mask is bool (10, 10), with Team A in slots 0..4 and Team B in 5..9.
    Return ally then enemy masks, both bool (10, 5). Each relation retains its
    stable team-roster order so masks align with unit features and target IDs.
    The input is not changed and no visibility rule is recomputed.
    """
    ally_mask_team_a = global_mask[
        TEAM_A_START:TEAM_A_END,
        TEAM_A_START:TEAM_A_END,
    ]
    enemy_mask_team_a = global_mask[
        TEAM_A_START:TEAM_A_END,
        TEAM_B_START:TEAM_B_END,
    ]

    ally_mask_team_b = global_mask[
        TEAM_B_START:TEAM_B_END,
        TEAM_B_START:TEAM_B_END,
    ]
    enemy_mask_team_b = global_mask[
        TEAM_B_START:TEAM_B_END,
        TEAM_A_START:TEAM_A_END,
    ]

    ally_mask = jnp.vstack((ally_mask_team_a, ally_mask_team_b))
    enemy_mask = jnp.vstack((enemy_mask_team_a, enemy_mask_team_b))

    return (ally_mask, enemy_mask)


def _build_global_pairwise_team_masks(team_ids: Array) -> tuple[Array, Array]:
    """Return same-team and opposing-team relations for configured team IDs.

    team_ids is int32 (10,), with 0 unused, 1 Team A and 2 Team B. Return two
    bool (10, 10) matrices in ally then enemy order. Pairs involving ID zero
    are False. These relations do not include alive, sight or distance gates.
    """
    has_real_team = team_ids != NO_TEAM_ID
    both_slots_have_real_teams = jnp.logical_and(
        has_real_team[:, None], has_real_team[None, :]
    )

    global_pairwise_ally_mask = jnp.logical_and(
        team_ids[None, :] == team_ids[:, None],
        both_slots_have_real_teams,
    )

    global_pairwise_enemy_mask = jnp.logical_and(
        team_ids[None, :] != team_ids[:, None],
        both_slots_have_real_teams,
    )

    return global_pairwise_ally_mask, global_pairwise_enemy_mask


def _build_select_target_use_ultimate_joint_mask(
    state: EnvState,
    config: EnvConfig,
    global_visibility_mask: Array,
    global_pairwise_distances: Array,
) -> Array:
    """Build the current authority for every target and Ultimate pair.

    state and config match the same decision. global_visibility_mask is bool
    (10, 10) and global_pairwise_distances is float32 (10, 10) in world units.
    Return bool (10, 11, 2), ordered by actor, actor-relative target, Ultimate.
    Lane zero checks Basic relation/range/control rules; lane one also checks
    Ultimate mode and cooldown. Both reuse the supplied spatial truth.
    Dead and unused actors admit only (Target None, no Ultimate); shield and
    stun prevent nonneutral combat. Marginal head masks are derived separately.
    """
    class_ids = config.agent_profile.class_ids
    active_and_alive_mask = jnp.logical_and(
        config.agent_profile.active_mask, state.alive_mask
    )

    basic_interaction_radii = config.agent_profile.basic_interaction_radii[:, None]
    basic_interaction_radius_mask = global_pairwise_distances <= basic_interaction_radii

    # Stun is actor-side control: any active channel removes non-empty targets.
    is_not_stunned = jnp.all(state.stun_durations == 0, axis=-1)

    # Stun and spawn shield independently make a source combat-ineligible.
    is_not_under_spawn_shield = state.spawn_shield_durations == 0
    is_combat_capable = jnp.logical_and(is_not_stunned, is_not_under_spawn_shield)

    # Fixed catalog payloads describe whether each actor owns the interaction.
    does_basic_damage = get_basic_damage_by_class_ids(class_ids) > 0
    does_basic_healing = get_basic_healing_by_class_ids(class_ids) > 0

    # Actor-owned facts broadcast across candidate columns.
    actor_can_damage = jnp.logical_and(is_combat_capable, does_basic_damage)[:, None]
    actor_can_heal = jnp.logical_and(is_combat_capable, does_basic_healing)[:, None]

    global_pairwise_ally_mask, global_pairwise_enemy_mask = (
        _build_global_pairwise_team_masks(config.agent_profile.team_ids)
    )

    # Opponent concealment already excludes shielded enemy candidates; apply
    # the same interaction rule to visible ally candidates.
    global_pairwise_ally_mask = jnp.logical_and(
        global_pairwise_ally_mask,
        is_not_under_spawn_shield[None, :],
    )

    global_basic_relation_mask = jnp.logical_or(
        jnp.logical_and(actor_can_heal, global_pairwise_ally_mask),
        jnp.logical_and(actor_can_damage, global_pairwise_enemy_mask),
    )

    # Class/control legality only narrows the established spatial relation.
    global_basic_unit_mask = jnp.logical_and(
        global_visibility_mask,
        jnp.logical_and(basic_interaction_radius_mask, global_basic_relation_mask),
    )

    ally_basic_select_target_mask, enemy_basic_select_target_mask = (
        _build_ally_enemy_masks(global_basic_unit_mask)
    )

    relative_basic_mask = jnp.concatenate(
        (ally_basic_select_target_mask, enemy_basic_select_target_mask), axis=-1
    )

    basic_target_action_mask = jnp.concatenate(
        (active_and_alive_mask[:, None], relative_basic_mask), axis=-1
    )

    # Build the ultimate conditioned mask.
    ultimate_interaction_radii = config.agent_profile.ultimate_interaction_radii[
        :, None
    ]
    ultimate_interaction_radius_mask = (
        global_pairwise_distances <= ultimate_interaction_radii
    )

    ultimate_is_off_cooldown = state.ultimate_cooldowns == 0

    ultimate_target_modes = get_ultimate_target_mode_by_class_ids(class_ids)

    has_available_enemy_targeted_ultimate = jnp.logical_and(
        ultimate_target_modes == ONLY_ENEMY_TARGET_ULTIMATE_MODE,
        ultimate_is_off_cooldown,
    )
    has_available_ally_targeted_ultimate = jnp.logical_and(
        ultimate_target_modes == ONLY_ALLY_TARGET_ULTIMATE_MODE,
        ultimate_is_off_cooldown,
    )
    has_available_no_target_ultimate = jnp.logical_and(
        ultimate_target_modes == ONLY_NONE_TARGET_ULTIMATE_MODE,
        ultimate_is_off_cooldown,
    )

    actor_can_use_enemy_targeted_ultimate = jnp.logical_and(
        is_combat_capable, has_available_enemy_targeted_ultimate
    )[:, None]
    actor_can_use_ally_targeted_ultimate = jnp.logical_and(
        is_combat_capable, has_available_ally_targeted_ultimate
    )[:, None]
    actor_can_use_no_target_ultimate = jnp.logical_and(
        is_combat_capable, has_available_no_target_ultimate
    )
    actor_can_use_no_target_ultimate = jnp.logical_and(
        active_and_alive_mask, actor_can_use_no_target_ultimate
    )

    global_targeted_ultimate_relation_mask = jnp.logical_or(
        jnp.logical_and(
            actor_can_use_ally_targeted_ultimate, global_pairwise_ally_mask
        ),
        jnp.logical_and(
            actor_can_use_enemy_targeted_ultimate, global_pairwise_enemy_mask
        ),
    )

    global_targeted_ultimate_mask = jnp.logical_and(
        global_visibility_mask,
        jnp.logical_and(
            ultimate_interaction_radius_mask,
            global_targeted_ultimate_relation_mask,
        ),
    )

    ally_ultimate_select_target_mask, enemy_ultimate_select_target_mask = (
        _build_ally_enemy_masks(global_targeted_ultimate_mask)
    )

    relative_ultimate_mask = jnp.concatenate(
        (ally_ultimate_select_target_mask, enemy_ultimate_select_target_mask), axis=-1
    )

    ultimate_target_action_mask = jnp.concatenate(
        (actor_can_use_no_target_ultimate[:, None], relative_ultimate_mask), axis=-1
    )

    canonical_nonacting_target_mask = jnp.arange(NUM_TARGET_ACTIONS)[None, :] == 0

    basic_target_action_mask = jnp.where(
        active_and_alive_mask[:, None],
        basic_target_action_mask,
        canonical_nonacting_target_mask,
    )

    return jnp.stack((basic_target_action_mask, ultimate_target_action_mask), axis=-1)


def _build_marginal_action_masks(
    select_target_use_ultimate_joint_mask: Array,
) -> tuple[Array, Array]:
    """Derive target and Ultimate head masks from exact combat-pair legality.

    The bool joint mask is (10, 11, 2). Return target mask (10, 11), then
    Ultimate mask (10, 2), using whether any compatible partner exists.
    Two True marginal entries need not form an allowed joint pair.
    """
    select_target_mask = jnp.any(select_target_use_ultimate_joint_mask, axis=-1)
    use_ultimate_mask = jnp.any(select_target_use_ultimate_joint_mask, axis=1)

    return select_target_mask, use_ultimate_mask


def _build_move_mask(state: EnvState, config: EnvConfig) -> Array:
    """Return current admitted movement categories for every fixed slot.

    state and config describe one decision. Return bool (10, 9): active,
    living and unstunned actors may submit every category. Stunned, dead or
    unused actors may submit only Stay. This checks submission, not whether
    geometry will allow the requested displacement.
    """
    active_and_alive_mask = jnp.logical_and(
        config.agent_profile.active_mask, state.alive_mask
    )
    active_alive_not_stunned_mask = jnp.logical_and(
        active_and_alive_mask,
        jnp.all(state.stun_durations == 0, axis=-1),
    )

    canonical_stay_mask = jnp.arange(NUM_MOVE_ACTIONS) == MOVE_STAY

    return jnp.logical_or(
        active_alive_not_stunned_mask[:, None], canonical_stay_mask[None, :]
    )


def _build_context_features(state: EnvState, config: EnvConfig) -> Array:
    """Pack public current-step and episode context in each observer's team order.

    Matching state and config produce raw float32 (10, 20) values following
    CONTEXT_FEATURE_* columns. Scores and sizes use own team before opponent;
    map coordinates are not reflected. Column 19 is the configured Team
    Deathmatch Red Zone depth, the same for every configured row (0 when the
    rule is off or the task is neutral). Reserved task/objective fields are
    zero, as are unused observer rows. Configured dead observers retain context.
    """

    team_a_ally_team_size = jnp.sum(
        jnp.logical_and(
            config.agent_profile.team_ids[TEAM_A_START:TEAM_A_END] == TEAM_A_ID,
            config.agent_profile.active_mask[TEAM_A_START:TEAM_A_END],
        )
    )

    team_b_ally_team_size = jnp.sum(
        jnp.logical_and(
            config.agent_profile.team_ids[TEAM_B_START:TEAM_B_END] == TEAM_B_ID,
            config.agent_profile.active_mask[TEAM_B_START:TEAM_B_END],
        )
    )

    team_a_enemy_team_size = team_b_ally_team_size
    team_b_enemy_team_size = team_a_ally_team_size

    # Reserved mode, objective, score, and threshold columns start neutral.
    context_features = jnp.zeros(
        shape=(MAX_AGENT_SLOTS, CONTEXT_FEATURES), dtype=jnp.float32
    )

    context_features = context_features.at[
        :, CONTEXT_FEATURE_CURRENT_TIMESTEP : CONTEXT_FEATURE_MAP_HEIGHT + 1
    ].set(
        jnp.asarray(
            [
                state.step_count,
                config.max_steps,
                config.map_width,
                config.map_height,
            ]
        )
    )

    # Team A rows see Team A as allies and Team B as enemies.
    context_features = context_features.at[
        TEAM_A_START:TEAM_A_END,
        CONTEXT_FEATURE_ALLY_TEAM_SIZE : CONTEXT_FEATURE_ENEMY_TEAM_SIZE + 1,
    ].set(jnp.asarray([team_a_ally_team_size, team_a_enemy_team_size]))

    # Team B rows receive the actor-relative inverse of those counts.
    context_features = context_features.at[
        TEAM_B_START:TEAM_B_END,
        CONTEXT_FEATURE_ALLY_TEAM_SIZE : CONTEXT_FEATURE_ENEMY_TEAM_SIZE + 1,
    ].set(jnp.asarray([team_b_ally_team_size, team_b_enemy_team_size]))

    # Expose the selected fixed task mode without introducing a string payload.
    context_features = context_features.at[
        :,
        CONTEXT_FEATURE_IS_TDM : CONTEXT_FEATURE_IS_CTF + 1,
    ].set(
        jnp.asarray(
            (
                config.task_mode == TASK_MODE_TDM,
                config.task_mode == TASK_MODE_KOTH,
                config.task_mode == TASK_MODE_CTF,
            ),
            dtype=jnp.bool_,
        )
    )

    # Team Deathmatch scores are actor-relative: allies precede enemies.
    context_features = context_features.at[
        TEAM_A_START:TEAM_A_END,
        CONTEXT_FEATURE_TDM_ALLY_SCORE : CONTEXT_FEATURE_TDM_ENEMY_SCORE + 1,
    ].set(state.team_deathmatch_scores)

    context_features = context_features.at[
        TEAM_B_START:TEAM_B_END,
        CONTEXT_FEATURE_TDM_ALLY_SCORE : CONTEXT_FEATURE_TDM_ENEMY_SCORE + 1,
    ].set(jnp.flip(state.team_deathmatch_scores))

    # The configured Team Deathmatch threshold is globally public.
    context_features = context_features.at[:, CONTEXT_FEATURE_TDM_SCORE_THRESHOLD].set(
        config.team_deathmatch_score_threshold
    )

    # The Red Zone depth is globally public, like the threshold.
    context_features = context_features.at[:, CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH].set(
        config.team_deathmatch_red_zone_depth
    )

    # Global episode facts are policy inputs only for configured actor slots.
    context_features = jnp.where(
        config.agent_profile.active_mask[:, None],
        context_features,
        jnp.zeros_like(context_features),
    )

    return context_features.astype(jnp.float32)


def _build_ally_enemy_one_hot_action_tensors(
    previous_joint_action_head_one_hot: Array, num_actions: int
) -> tuple[Array, Array]:
    """Place one action-head history into stable ally and enemy observation rows.

    previous_joint_action_head_one_hot is float32 (10, num_actions), already
    encoded in the desired category convention. Static num_actions is the last
    axis size. Return ally then enemy tensors (10, 5, num_actions). This only
    reorders rows; it does not remap target categories or apply visibility.
    """
    ally_actions_one_hot_team_a = jnp.broadcast_to(
        previous_joint_action_head_one_hot[TEAM_A_START:TEAM_A_END, :],
        (MAX_AGENTS_PER_TEAM, MAX_AGENTS_PER_TEAM, num_actions),
    )
    ally_actions_one_hot_team_b = jnp.broadcast_to(
        previous_joint_action_head_one_hot[TEAM_B_START:TEAM_B_END, :],
        (MAX_AGENTS_PER_TEAM, MAX_AGENTS_PER_TEAM, num_actions),
    )
    ally_actions_one_hot = jnp.concatenate(
        (ally_actions_one_hot_team_a, ally_actions_one_hot_team_b), axis=0
    )

    enemy_actions_one_hot_team_a = jnp.broadcast_to(
        previous_joint_action_head_one_hot[TEAM_B_START:TEAM_B_END, :],
        (MAX_AGENTS_PER_TEAM, MAX_AGENTS_PER_TEAM, num_actions),
    )
    enemy_actions_one_hot_team_b = jnp.broadcast_to(
        previous_joint_action_head_one_hot[TEAM_A_START:TEAM_A_END, :],
        (MAX_AGENTS_PER_TEAM, MAX_AGENTS_PER_TEAM, num_actions),
    )
    enemy_actions_one_hot = jnp.concatenate(
        (enemy_actions_one_hot_team_a, enemy_actions_one_hot_team_b), axis=0
    )

    return ally_actions_one_hot, enemy_actions_one_hot


def _build_visibility_masked_previous_timestep_action_observation(
    state: EnvState, ally_visibility_mask: Array, enemy_visibility_mask: Array
) -> PreviousTimestepActionObservation:
    """Expose accepted action history using the current observed-actor visibility.

    state holds compact accepted int32 heads (10,) and a scalar history-valid
    flag. ally_visibility_mask and enemy_visibility_mask are bool (10, 5) from
    that state's observation. Return PreviousTimestepActionObservation with
    float32 (10, 5, 9), (10, 5, 11) and (10, 5, 2) one-hot families.

    Reset and hidden-actor rows are zero. A visible accepted category zero has
    a one in category zero. Movement and Ultimate only need row projection;
    opposing observers also swap target relation blocks to retain physical
    target identity. Target visibility does not hide a visible actor's target.
    """
    previous_joint_move_actions = state.previous_timestep_move_actions
    previous_joint_use_ultimate_actions = state.previous_timestep_use_ultimate_actions

    previous_joint_move_actions_one_hot = jax.nn.one_hot(
        previous_joint_move_actions, NUM_MOVE_ACTIONS, dtype=jnp.float32
    )

    # Target categories are actor-relative. Opposing observers preserve target
    # identity by exchanging the ally and enemy category blocks.
    previous_joint_select_target_actions = state.previous_timestep_select_target_actions

    actor_relative_select_target_matrix = jax.nn.one_hot(
        previous_joint_select_target_actions, NUM_TARGET_ACTIONS, dtype=jnp.float32
    )

    team_a_none_target_column = actor_relative_select_target_matrix[
        TEAM_A_START:TEAM_A_END, 0:1
    ]
    team_b_none_target_column = actor_relative_select_target_matrix[
        TEAM_B_START:TEAM_B_END, 0:1
    ]

    team_a_actor_relative_select_target_matrix = actor_relative_select_target_matrix[
        TEAM_A_START:TEAM_A_END, 1:
    ]
    team_b_actor_relative_select_target_matrix = actor_relative_select_target_matrix[
        TEAM_B_START:TEAM_B_END, 1:
    ]

    team_a_block_for_team_b_observer = jnp.concatenate(
        (
            team_a_none_target_column,
            team_a_actor_relative_select_target_matrix[:, TEAM_B_START:TEAM_B_END],
            team_a_actor_relative_select_target_matrix[:, TEAM_A_START:TEAM_A_END],
        ),
        axis=-1,
        dtype=jnp.float32,
    )

    team_a_block_for_team_a_observer = actor_relative_select_target_matrix[
        TEAM_A_START:TEAM_A_END, :
    ]

    team_b_block_for_team_a_observer = jnp.concatenate(
        (
            team_b_none_target_column,
            team_b_actor_relative_select_target_matrix[:, TEAM_B_START:TEAM_B_END],
            team_b_actor_relative_select_target_matrix[:, TEAM_A_START:TEAM_A_END],
        ),
        axis=-1,
        dtype=jnp.float32,
    )

    team_b_block_for_team_b_observer = actor_relative_select_target_matrix[
        TEAM_B_START:TEAM_B_END, :
    ]

    unmasked_ally_previous_timestep_select_target_actions_one_hot = jnp.concatenate(
        (
            jnp.broadcast_to(
                team_a_block_for_team_a_observer,
                (MAX_AGENTS_PER_TEAM, MAX_AGENTS_PER_TEAM, NUM_TARGET_ACTIONS),
            ),
            jnp.broadcast_to(
                team_b_block_for_team_b_observer,
                (MAX_AGENTS_PER_TEAM, MAX_AGENTS_PER_TEAM, NUM_TARGET_ACTIONS),
            ),
        ),
        axis=0,
        dtype=jnp.float32,
    )

    unmasked_enemy_previous_timestep_select_target_actions_one_hot = jnp.concatenate(
        (
            jnp.broadcast_to(
                team_b_block_for_team_a_observer,
                (MAX_AGENTS_PER_TEAM, MAX_AGENTS_PER_TEAM, NUM_TARGET_ACTIONS),
            ),
            jnp.broadcast_to(
                team_a_block_for_team_b_observer,
                (MAX_AGENTS_PER_TEAM, MAX_AGENTS_PER_TEAM, NUM_TARGET_ACTIONS),
            ),
        ),
        axis=0,
        dtype=jnp.float32,
    )

    ally_previous_timestep_select_target_actions_one_hot = (
        ally_visibility_mask[:, :, None]
        * state.has_previous_timestep_joint_action
        * unmasked_ally_previous_timestep_select_target_actions_one_hot
    )

    enemy_previous_timestep_select_target_actions_one_hot = (
        enemy_visibility_mask[:, :, None]
        * state.has_previous_timestep_joint_action
        * unmasked_enemy_previous_timestep_select_target_actions_one_hot
    )

    previous_joint_use_ultimate_actions_one_hot = jax.nn.one_hot(
        previous_joint_use_ultimate_actions, NUM_ULTIMATE_ACTIONS, dtype=jnp.float32
    )

    (
        unmasked_ally_previous_timestep_move_actions_one_hot,
        unmasked_enemy_previous_timestep_move_actions_one_hot,
    ) = _build_ally_enemy_one_hot_action_tensors(
        previous_joint_move_actions_one_hot, NUM_MOVE_ACTIONS
    )

    (
        unmasked_ally_previous_timestep_use_ultimate_actions_one_hot,
        unmasked_enemy_previous_timestep_use_ultimate_actions_one_hot,
    ) = _build_ally_enemy_one_hot_action_tensors(
        previous_joint_use_ultimate_actions_one_hot, NUM_ULTIMATE_ACTIONS
    )

    ally_previous_timestep_move_actions_one_hot = (
        ally_visibility_mask[:, :, None]
        * state.has_previous_timestep_joint_action
        * unmasked_ally_previous_timestep_move_actions_one_hot
    )

    enemy_previous_timestep_move_actions_one_hot = (
        enemy_visibility_mask[:, :, None]
        * state.has_previous_timestep_joint_action
        * unmasked_enemy_previous_timestep_move_actions_one_hot
    )

    ally_previous_timestep_use_ultimate_actions_one_hot = (
        ally_visibility_mask[:, :, None]
        * state.has_previous_timestep_joint_action
        * unmasked_ally_previous_timestep_use_ultimate_actions_one_hot
    )

    enemy_previous_timestep_use_ultimate_actions_one_hot = (
        enemy_visibility_mask[:, :, None]
        * state.has_previous_timestep_joint_action
        * unmasked_enemy_previous_timestep_use_ultimate_actions_one_hot
    )

    return PreviousTimestepActionObservation(
        ally_previous_timestep_move_actions_one_hot,
        enemy_previous_timestep_move_actions_one_hot,
        ally_previous_timestep_select_target_actions_one_hot,
        enemy_previous_timestep_select_target_actions_one_hot,
        ally_previous_timestep_use_ultimate_actions_one_hot,
        enemy_previous_timestep_use_ultimate_actions_one_hot,
    )


def _build_spawn_lifecycle_observation(
    state: EnvState, config: EnvConfig
) -> SpawnLifecycleObservation:
    """Expose public spawn and roster truth for the current decision.

    Matching state and config produce SpawnLifecycleObservation. Every leaf
    starts with ten observers; its team axis is [own team, opponent], with
    five roster rows where present. Pads and speed are float32, durations,
    clocks and classes int32, and membership/alive flags bool.
    Configured living and dead observers see the same public roster/rules and
    current clocks. Unused observer rows are zero. Coordinates remain in the
    world frame; no visibility filtering or simulator time advance occurs.
    """
    spawn_pad_positions_team_a_view = jnp.concatenate(
        (
            config.team_spawn_pad_positions[TEAM_A_ID - 1, :, :][None, :, :],
            config.team_spawn_pad_positions[TEAM_B_ID - 1, :, :][None, :, :],
        ),
        axis=0,
    )

    spawn_pad_positions_team_b_view = jnp.concatenate(
        (
            config.team_spawn_pad_positions[TEAM_B_ID - 1, :, :][None, :, :],
            config.team_spawn_pad_positions[TEAM_A_ID - 1, :, :][None, :, :],
        ),
        axis=0,
    )

    unmasked_spawn_pad_positions = jnp.concatenate(
        (
            jnp.repeat(
                spawn_pad_positions_team_a_view[None, :, :, :],
                MAX_AGENTS_PER_TEAM,
                axis=0,
            ),
            jnp.repeat(
                spawn_pad_positions_team_b_view[None, :, :, :],
                MAX_AGENTS_PER_TEAM,
                axis=0,
            ),
        ),
        axis=0,
    )

    spawn_pad_positions = (
        config.agent_profile.active_mask[:, None, None, None]
        * unmasked_spawn_pad_positions
    )

    active_mask_team_a_view = jnp.concatenate(
        (
            config.agent_profile.active_mask[TEAM_A_START:TEAM_A_END][None, :],
            config.agent_profile.active_mask[TEAM_B_START:TEAM_B_END][None, :],
        ),
        axis=0,
    )

    active_mask_team_b_view = jnp.concatenate(
        (
            config.agent_profile.active_mask[TEAM_B_START:TEAM_B_END][None, :],
            config.agent_profile.active_mask[TEAM_A_START:TEAM_A_END][None, :],
        ),
        axis=0,
    )

    unmasked_active_mask = jnp.concatenate(
        (
            jnp.repeat(
                active_mask_team_a_view[None, :, :], MAX_AGENTS_PER_TEAM, axis=0
            ),
            jnp.repeat(
                active_mask_team_b_view[None, :, :], MAX_AGENTS_PER_TEAM, axis=0
            ),
        ),
        axis=0,
    )

    active_mask = jnp.logical_and(
        unmasked_active_mask, config.agent_profile.active_mask[:, None, None]
    )

    # Configured classes are public roster metadata, so only inactive observer
    # rows are hidden; death, shielding, and visibility do not alter them.
    class_ids_team_a_view = jnp.concatenate(
        (
            config.agent_profile.class_ids[TEAM_A_START:TEAM_A_END][None, :],
            config.agent_profile.class_ids[TEAM_B_START:TEAM_B_END][None, :],
        ),
        axis=0,
    )

    class_ids_team_b_view = jnp.concatenate(
        (
            config.agent_profile.class_ids[TEAM_B_START:TEAM_B_END][None, :],
            config.agent_profile.class_ids[TEAM_A_START:TEAM_A_END][None, :],
        ),
        axis=0,
    )

    unmasked_class_ids = jnp.concatenate(
        (
            jnp.repeat(class_ids_team_a_view[None, :, :], MAX_AGENTS_PER_TEAM, axis=0),
            jnp.repeat(class_ids_team_b_view[None, :, :], MAX_AGENTS_PER_TEAM, axis=0),
        ),
        axis=0,
    )

    class_ids = unmasked_class_ids * config.agent_profile.active_mask[:, None, None]

    alive_mask_team_a_view = jnp.concatenate(
        (
            state.alive_mask[TEAM_A_START:TEAM_A_END][None, :],
            state.alive_mask[TEAM_B_START:TEAM_B_END][None, :],
        ),
        axis=0,
    )

    alive_mask_team_b_view = jnp.concatenate(
        (
            state.alive_mask[TEAM_B_START:TEAM_B_END][None, :],
            state.alive_mask[TEAM_A_START:TEAM_A_END][None, :],
        ),
        axis=0,
    )

    unmasked_alive_mask = jnp.concatenate(
        (
            jnp.repeat(alive_mask_team_a_view[None, :, :], MAX_AGENTS_PER_TEAM, axis=0),
            jnp.repeat(alive_mask_team_b_view[None, :, :], MAX_AGENTS_PER_TEAM, axis=0),
        ),
        axis=0,
    )

    alive_mask = jnp.logical_and(
        unmasked_alive_mask, config.agent_profile.active_mask[:, None, None]
    )

    spawn_shield_actual_durations_team_a_view = jnp.concatenate(
        (
            state.spawn_shield_durations[TEAM_A_START:TEAM_A_END][None, :],
            state.spawn_shield_durations[TEAM_B_START:TEAM_B_END][None, :],
        ),
        axis=0,
    )

    spawn_shield_actual_durations_team_b_view = jnp.concatenate(
        (
            state.spawn_shield_durations[TEAM_B_START:TEAM_B_END][None, :],
            state.spawn_shield_durations[TEAM_A_START:TEAM_A_END][None, :],
        ),
        axis=0,
    )

    unmasked_spawn_shield_actual_durations = jnp.concatenate(
        (
            jnp.repeat(
                spawn_shield_actual_durations_team_a_view[None, :, :],
                MAX_AGENTS_PER_TEAM,
                axis=0,
            ),
            jnp.repeat(
                spawn_shield_actual_durations_team_b_view[None, :, :],
                MAX_AGENTS_PER_TEAM,
                axis=0,
            ),
        ),
        axis=0,
    )

    spawn_shield_actual_durations = (
        unmasked_spawn_shield_actual_durations
        * config.agent_profile.active_mask[:, None, None]
    ).astype(jnp.int32)

    spawn_shield_configured_duration_by_agent = (
        config.spawn_shield_duration_steps * config.agent_profile.active_mask
    )
    spawn_shield_speed_by_agent = (
        config.spawn_shield_movement_speed * config.agent_profile.active_mask
    )

    # Actor-relative respawn-wave periods (MAX_AGENT_SLOTS, NUM_TEAMS), int32
    team_respawn_wave_period_step_count_team_a_view = jnp.repeat(
        config.team_respawn_wave_period_step_count[None, :], MAX_AGENTS_PER_TEAM, axis=0
    )

    team_respawn_wave_period_step_count_team_b_view = jnp.repeat(
        jnp.asarray(
            (
                config.team_respawn_wave_period_step_count[TEAM_B_ID - 1],
                config.team_respawn_wave_period_step_count[TEAM_A_ID - 1],
            )
        )[None, :],
        MAX_AGENTS_PER_TEAM,
        axis=0,
    )

    respawn_wave_period_step_count_by_agent_by_team = (
        jnp.concatenate(
            (
                team_respawn_wave_period_step_count_team_a_view,
                team_respawn_wave_period_step_count_team_b_view,
            ),
            axis=0,
            dtype=jnp.int32,
        )
        * config.agent_profile.active_mask[:, None]
    )

    # Actor-relative respawn-wave countdowns (MAX_AGENT_SLOTS, NUM_TEAMS), int32
    team_respawn_wave_countdowns_team_a_view = jnp.repeat(
        state.team_respawn_wave_countdowns[None, :], MAX_AGENTS_PER_TEAM, axis=0
    )

    team_respawn_wave_countdowns_team_b_view = jnp.repeat(
        jnp.asarray(
            (
                state.team_respawn_wave_countdowns[TEAM_B_ID - 1],
                state.team_respawn_wave_countdowns[TEAM_A_ID - 1],
            )
        )[None, :],
        MAX_AGENTS_PER_TEAM,
        axis=0,
    )

    respawn_wave_countdowns_by_agent_by_team = (
        jnp.concatenate(
            (
                team_respawn_wave_countdowns_team_a_view,
                team_respawn_wave_countdowns_team_b_view,
            ),
            axis=0,
            dtype=jnp.int32,
        )
        * config.agent_profile.active_mask[:, None]
    )

    return SpawnLifecycleObservation(
        spawn_pad_positions_by_agent_by_team=spawn_pad_positions,
        spawn_shield_actual_durations_by_agent_by_team=spawn_shield_actual_durations,
        spawn_shield_configured_duration_by_agent=spawn_shield_configured_duration_by_agent.astype(
            jnp.int32
        ),
        spawn_shield_speed_by_agent=spawn_shield_speed_by_agent.astype(jnp.float32),
        respawn_wave_period_step_count_by_agent_by_team=respawn_wave_period_step_count_by_agent_by_team,
        respawn_wave_countdowns_by_agent_by_team=respawn_wave_countdowns_by_agent_by_team,
        active_mask_by_agent_by_team=active_mask,
        alive_mask_by_agent_by_team=alive_mask,
        class_ids_by_agent_by_team=class_ids,
    )


def _build_observation_and_action_mask(
    state: EnvState, config: EnvConfig
) -> tuple[Observation, ActionMask]:
    """Build an observation and action mask from the same current snapshot.

    state and config describe one scalar ten-slot game. Return Observation,
    then ActionMask, with the schemas documented on those types. Stable ally
    and enemy feature rows are zeroed by current visibility; public geometry
    is broadcast to every observer. Previous actions remain a separate family
    and use current actor visibility. Reuse pair distances and aura values
    within this construction so the returned input and mask agree in time.
    """
    global_visibility_mask, global_pairwise_distances = (
        _build_global_visibility_mask_and_distances(state, config)
    )

    combat_aura_aggregation_result = _derive_aura_damage_multipliers(
        config,
        global_pairwise_distances,
        state.alive_mask,
        state.spawn_shield_durations == 0,
    )

    self_features = _build_self_features(
        state,
        config,
        combat_aura_aggregation_result.mage_damage_amplification_aura_multipliers,
        combat_aura_aggregation_result.warrior_damage_mitigation_aura_multipliers,
    )
    ally_features = _build_ally_features(self_features)
    enemy_features = _build_enemy_features(self_features)

    ally_visibility_mask, enemy_visibility_mask = _build_ally_enemy_masks(
        global_visibility_mask
    )

    ally_features = _mask_unit_features(ally_features, ally_visibility_mask)
    enemy_features = _mask_unit_features(enemy_features, enemy_visibility_mask)

    select_target_use_ultimate_joint_mask = (
        _build_select_target_use_ultimate_joint_mask(
            state, config, global_visibility_mask, global_pairwise_distances
        )
    )
    select_target_mask, use_ultimate_mask = _build_marginal_action_masks(
        select_target_use_ultimate_joint_mask
    )
    move_mask = _build_move_mask(state, config)

    context_features = _build_context_features(state, config)

    map_obstacle_features = jnp.broadcast_to(
        config.obstacles[None, :, :],
        (MAX_AGENT_SLOTS, MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
    )

    current_action_mask = ActionMask(
        move_mask=move_mask,
        select_target_mask=select_target_mask,
        use_ultimate_mask=use_ultimate_mask,
        select_target_use_ultimate_joint_mask=select_target_use_ultimate_joint_mask,
    )

    visibility_masked_previous_timestep_action_observation = (
        _build_visibility_masked_previous_timestep_action_observation(
            state, ally_visibility_mask, enemy_visibility_mask
        )
    )

    spawn_lifecycle_observation = _build_spawn_lifecycle_observation(state, config)

    current_observation = Observation(
        self_features=self_features,
        ally_unit_features=ally_features,
        enemy_unit_features=enemy_features,
        map_obstacle_features=map_obstacle_features,
        objective_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_OBJECTIVE_SLOTS, OBJECTIVE_FEATURES),
            dtype=jnp.float32,
        ),
        context_features=context_features,
        ally_visibility_mask=ally_visibility_mask,
        enemy_visibility_mask=enemy_visibility_mask,
        previous_timestep_actions=visibility_masked_previous_timestep_action_observation,
        spawn_lifecycle=spawn_lifecycle_observation,
        self_ally_index=jnp.where(
            config.agent_profile.active_mask,
            jnp.arange(MAX_AGENT_SLOTS, dtype=jnp.int32) % MAX_AGENTS_PER_TEAM,
            0,
        ),
    )

    return current_observation, current_action_mask


def _build_intended_movement_deltas(
    current_state: EnvState,
    config: EnvConfig,
    accepted_joint_action: Action,
) -> Array:
    """Convert accepted moves and current visible speed into requested travel.

    current_state and config describe the action's decision. The accepted
    Action has int32 heads (10,). Return float32 world deltas (10, 2), using
    the shared effective-speed authority and unchanged compass directions.
    Charge relocation and geometry are separate phases; new statuses from this
    transition do not retroactively alter this precommitted movement.
    """
    intended_movement_deltas_unscaled = _JOINT_ACTION_MOVE_TO_DISPLACEMENT_LOOKUP_TABLE[
        accepted_joint_action.move
    ]
    # Control-adjusted speed is shared with the observation contract.
    effective_movement_speeds = derive_effective_movement_speeds(
        current_state.slow_durations,
        current_state.priest_blessing_of_freedom_slow_floor_durations,
        current_state.stun_durations,
        current_state.spawn_shield_durations,
        config.agent_profile.base_movement_speeds,
        config.spawn_shield_movement_speed,
        jnp.logical_and(config.agent_profile.active_mask, current_state.alive_mask),
        config.ordinary_movement_distance_scale,
    )

    intended_movement_deltas = (
        effective_movement_speeds[:, None] * intended_movement_deltas_unscaled
    )

    return intended_movement_deltas


def _active_mage_class_mask(config: EnvConfig) -> Array:
    """Return configured Mage slots from the resolved profile.

    config supplies int32 classes and bool membership (10,). Return bool (10,).
    This identifies roster membership, not current alive, control or shield status;
    callers add the eligibility checks needed by their mechanic.
    """
    return jnp.logical_and(
        config.agent_profile.class_ids == MAGE_CLASS_ID,
        config.agent_profile.active_mask,
    )


def _active_warrior_class_mask(config: EnvConfig) -> Array:
    """Return configured Warrior slots from the resolved profile.

    config supplies int32 classes and bool membership (10,). Return bool (10,).
    This identifies roster membership, not current alive, control or shield status;
    callers add the eligibility checks needed by their mechanic.
    """
    return jnp.logical_and(
        config.agent_profile.class_ids == WARRIOR_CLASS_ID,
        config.agent_profile.active_mask,
    )


def _active_hunter_class_mask(config: EnvConfig) -> Array:
    """Return configured Hunter slots from the resolved profile.

    config supplies int32 classes and bool membership (10,). Return bool (10,).
    This identifies roster membership, not current alive, control or shield status;
    callers add the eligibility checks needed by their mechanic.
    """
    return jnp.logical_and(
        config.agent_profile.class_ids == HUNTER_CLASS_ID,
        config.agent_profile.active_mask,
    )


def _active_rogue_class_mask(config: EnvConfig) -> Array:
    """Return configured Rogue slots from the resolved profile.

    config supplies int32 classes and bool membership (10,). Return bool (10,).
    This identifies roster membership, not current alive, control or shield status;
    callers add the eligibility checks needed by their mechanic.
    """
    return jnp.logical_and(
        config.agent_profile.class_ids == ROGUE_CLASS_ID,
        config.agent_profile.active_mask,
    )


def _active_priest_class_mask(config: EnvConfig) -> Array:
    """Return configured Priest slots from the resolved profile.

    config supplies int32 classes and bool membership (10,). Return bool (10,).
    This identifies roster membership, not current alive, control or shield status;
    callers add the eligibility checks needed by their mechanic.
    """
    return jnp.logical_and(
        config.agent_profile.class_ids == PRIEST_CLASS_ID,
        config.agent_profile.active_mask,
    )


def _build_self_features(
    state: EnvState,
    config: EnvConfig,
    mage_damage_amplification_aura_multipliers: Array,
    warrior_damage_mitigation_aura_multipliers: Array,
) -> Array:
    """Pack each observer's own current state and class capabilities.

    Matching state/config and the two current float32 aura multiplier vectors
    (10,) produce raw float32 (10, 58) rows following AGENT_FEATURE_* columns.
    mage_damage_amplification_aura_multipliers modify outgoing damage;
    warrior_damage_mitigation_aura_multipliers modify incoming damage.
    Current speed uses the same authority as movement. Derived capabilities
    are included without adding duplicate fields to EnvState. No unit-visibility
    filtering occurs here; relation builders apply it to candidate rows later.
    """
    class_ids = config.agent_profile.class_ids

    (
        slow_multipliers,
        rogue_poison_anti_heal_multipliers,
        priest_blessing_of_freedom_slow_floor_fraction,
    ) = derive_status_magnitudes(
        state.slow_durations,
        state.rogue_poison_anti_heal_durations,
        state.priest_blessing_of_freedom_slow_floor_durations,
    )

    effective_movement_speeds = derive_effective_movement_speeds(
        state.slow_durations,
        state.priest_blessing_of_freedom_slow_floor_durations,
        state.stun_durations,
        state.spawn_shield_durations,
        config.agent_profile.base_movement_speeds,
        config.spawn_shield_movement_speed,
        jnp.logical_and(config.agent_profile.active_mask, state.alive_mask),
        config.ordinary_movement_distance_scale,
    )

    features_0_to_14 = jnp.concatenate(
        (
            state.agent_positions,
            config.agent_profile.agent_radii[:, None],
            jnp.zeros((MAX_AGENT_SLOTS, 1), dtype=jnp.float32),
            config.agent_profile.active_mask[:, None],
            state.alive_mask[:, None],
            class_ids[:, None],
            config.agent_profile.base_movement_speeds[:, None],
            effective_movement_speeds[:, None],
            config.agent_profile.observation_radii[:, None],
            config.agent_profile.basic_interaction_radii[:, None],
            config.agent_profile.ultimate_interaction_radii[:, None],
            state.current_health[:, None],
            config.agent_profile.max_health[:, None],
            state.ultimate_cooldowns[:, None],
        ),
        axis=-1,
        dtype=jnp.float32,
    )

    features_15_to_31 = jnp.concatenate(
        (
            state.slow_durations,
            slow_multipliers,
            state.stun_durations,
            state.rogue_poison_anti_heal_durations[:, None],
            rogue_poison_anti_heal_multipliers[:, None],
            state.mage_burst_damage_amplification_durations[:, None],
            state.priest_blessing_of_freedom_slow_floor_durations[:, None],
            priest_blessing_of_freedom_slow_floor_fraction[:, None],
            state.steps_until_out_of_combat[:, None],
            mage_damage_amplification_aura_multipliers[:, None],
            warrior_damage_mitigation_aura_multipliers[:, None],
        ),
        axis=-1,
        dtype=jnp.float32,
    )

    warrior_mask = _active_warrior_class_mask(config)
    mage_mask = _active_mage_class_mask(config)
    hunter_mask = _active_hunter_class_mask(config)
    rogue_mask = _active_rogue_class_mask(config)
    priest_mask = _active_priest_class_mask(config)
    active_mask_bc = config.agent_profile.active_mask[:, None]
    # Non-state derived payload descriptors.
    # These tell us about an agent's inherent properties, not what's happening to it.

    basic_health_and_ultimate_cooldown_capability_features = jnp.where(
        active_mask_bc,
        jnp.concatenate(
            (
                BASIC_DAMAGE_BY_CLASS[class_ids][:, None],
                BASIC_HEALING_BY_CLASS[class_ids][:, None],
                ULTIMATE_COOLDOWN_BY_CLASS[class_ids][:, None],
            ),
            axis=-1,
        ),
        0.0,
    ).astype(jnp.float32)

    warrior_hunter_rogue_mask = jnp.tile(
        jnp.concatenate(
            (warrior_mask[:, None], hunter_mask[:, None], rogue_mask[:, None]), axis=-1
        ),
        3,
    )

    slow_stun_durations_multipliers = jnp.where(
        warrior_hunter_rogue_mask,
        jnp.asarray(
            [
                WARRIOR_CHARGE_SLOW_DURATION_TICKS,
                HUNTER_BASIC_SLOW_DURATION_TICKS,
                ROGUE_POISON_SLOW_DURATION_TICKS,
                WARRIOR_CHARGE_SLOW_MULTIPLIER,
                HUNTER_BASIC_SLOW_MULTIPLIER,
                ROGUE_POISON_SLOW_MULTIPLIER,
                WARRIOR_CHARGE_STUN_DURATION_TICKS,
                HUNTER_TRAP_STUN_DURATION_TICKS,
                ROGUE_POISON_STUN_DURATION_TICKS,
            ]
        )[None, :],
        0.0,
    ).astype(jnp.float32)

    rogue_anti_heal_capability_features = jnp.where(
        rogue_mask[:, None],
        jnp.asarray(
            [ROGUE_POISON_ANTI_HEAL_DURATION_TICKS, ROGUE_POISON_ANTI_HEAL_MULTIPLIER]
        )[None, :],
        0.0,
    ).astype(jnp.float32)

    mage_burst_capability_features = jnp.where(
        mage_mask[:, None],
        jnp.asarray([MAGE_BURST_DAMAGE_DURATION_TICKS, MAGE_BURST_DAMAGE_MULTIPLIER])[
            None, :
        ],
        0.0,
    ).astype(jnp.float32)

    priest_blessing_of_freedom_capability_features = jnp.where(
        priest_mask[:, None],
        jnp.asarray([PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS, PRIEST_HEAL_SPEED_FLOOR])[
            None, :
        ],
        0.0,
    ).astype(jnp.float32)

    mage_warrior_aura_mask = jnp.concatenate(
        (
            jnp.tile(mage_mask[:, None], 2),
            jnp.tile(warrior_mask[:, None], 2),
        ),
        axis=-1,
    )

    mage_and_warrior_aura_capability_features = jnp.where(
        mage_warrior_aura_mask,
        jnp.asarray(
            [
                MAGE_DAMAGE_AMPLIFICATION_AURA_RADIUS,
                MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER,
                WARRIOR_DAMAGE_MITIGATION_AURA_RADIUS,
                WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER,
            ]
        )[None, :],
        0.0,
    ).astype(jnp.float32)

    # Capability payloads remain zero for inactive rows even if a malformed
    # profile assigns those rows non-neutral class IDs.
    ultimate_healing_capability_features = jnp.where(
        config.agent_profile.active_mask,
        get_ultimate_healing_by_class_ids(class_ids),
        0.0,
    )[:, None]

    ultimate_damage_capability_features = jnp.where(
        config.agent_profile.active_mask,
        get_ultimate_damage_by_class_ids(class_ids),
        0.0,
    )[:, None]

    ooc_delay_steps_capability_features = jnp.where(
        config.agent_profile.active_mask,
        config.agent_profile.out_of_combat_delay_steps,
        0,
    )[:, None].astype(jnp.float32)

    ooc_health_regen_fraction_per_step_capability_features = jnp.where(
        config.agent_profile.active_mask,
        config.agent_profile.out_of_combat_health_regen_fraction_per_step,
        0,
    )[:, None]

    feature_32_to_57 = jnp.concatenate(
        (
            basic_health_and_ultimate_cooldown_capability_features,
            slow_stun_durations_multipliers,
            rogue_anti_heal_capability_features,
            mage_burst_capability_features,
            priest_blessing_of_freedom_capability_features,
            mage_and_warrior_aura_capability_features,
            ultimate_healing_capability_features,
            ultimate_damage_capability_features,
            ooc_delay_steps_capability_features,
            ooc_health_regen_fraction_per_step_capability_features,
        ),
        axis=-1,
        dtype=jnp.float32,
    )

    return jnp.concatenate(
        (
            features_0_to_14,
            features_15_to_31,
            feature_32_to_57,
        ),
        axis=-1,
        dtype=jnp.float32,
    )


def _build_ally_features(self_features: Array) -> Array:
    """Repeat each team's agent features in its own observers' stable roster order.

    self_features is float32 (10, 58), Team A before Team B. Return float32
    (10, 5, 58) ally rows, including self. Visibility is applied afterward.
    """
    ally_features = jnp.zeros(
        (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES), dtype=jnp.float32
    )

    ally_features = ally_features.at[TEAM_A_START:TEAM_A_END, :, :].set(
        self_features[TEAM_A_START:TEAM_A_END, :]
    )
    ally_features = ally_features.at[TEAM_B_START:TEAM_B_END, :, :].set(
        self_features[TEAM_B_START:TEAM_B_END, :]
    )

    return ally_features


def _build_enemy_features(self_features: Array) -> Array:
    """Repeat opposing-team features in each observer's stable enemy roster order.

    self_features is float32 (10, 58), Team A before Team B. Return float32
    (10, 5, 58) rows with the enemy indicator set. Visibility is applied later;
    coordinates retain the world frame.
    """
    enemy_features = jnp.zeros(
        (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES), dtype=jnp.float32
    )

    enemy_features = enemy_features.at[TEAM_A_START:TEAM_A_END, :, :].set(
        self_features[TEAM_B_START:TEAM_B_END, :]
    )
    enemy_features = enemy_features.at[TEAM_B_START:TEAM_B_END, :, :].set(
        self_features[TEAM_A_START:TEAM_A_END, :]
    )

    return enemy_features.at[:, :, AGENT_FEATURE_IS_ENEMY].set(1.0)


def _mask_unit_features(unit_features: Array, visibility_mask: Array) -> Array:
    """Zero hidden candidate rows without changing visible feature values.

    unit_features is float32 (10, 5, 58) and visibility_mask is bool (10, 5)
    for the same decision and relation order. Return the same feature shape;
    no rows are removed, so target and observation indices stay stable.
    """
    return jnp.where(
        visibility_mask[:, :, None],
        unit_features,
        jnp.zeros_like(unit_features),
    ).astype(jnp.float32)


def _build_accepted_joint_action_from_submitted_joint_action(
    current_action_mask: ActionMask, submitted_joint_action: Action
) -> tuple[Action, ActionAcceptanceFacts]:
    """Accept or replace submitted categories using the current authoritative mask.

    Parameters
    ----------
    current_action_mask : ActionMask
        Bool mask arrays from the same pre-state as the submission. The joint
        target/Ultimate mask, not its marginals, owns combat acceptance.
    submitted_joint_action : Action
        Three int32 arrays (10,). Normal domains are movement 0..8, target
        0..10 and Ultimate 0..1. Out-of-range IDs are handled as rejection.

    Returns
    -------
    tuple of Action and ActionAcceptanceFacts
        Accepted heads (10,), then submitted/accepted values and bool rejection
        flags (10,). Out-of-domain input clears that actor's whole tuple to
        (Stay, Target None, no Ultimate), without affecting other actors.
        For in-domain input, movement is accepted separately from the combat
        pair. A rejected Ultimate pair never falls back to a Basic action.

    Notes
    -----
    Replace unsafe indices before gathering any mask entry; JAX indexing alone
    does not reject negative or excessive categories. This pure numerical
    helper does not validate storage shapes/dtypes or rebuild mask provenance.
    Out-of-domain facts point to a policy or sampler defect for host inspection.
    """
    # Domain containment must precede every indexed mask access because negative
    # and upper-out-of-domain JAX indices are not semantic rejection.
    move_action_is_out_of_domain = jnp.logical_not(
        jnp.logical_and(
            submitted_joint_action.move < NUM_MOVE_ACTIONS,
            submitted_joint_action.move >= 0,
        )
    )
    select_target_action_is_out_of_domain = jnp.logical_not(
        jnp.logical_and(
            submitted_joint_action.select_target < NUM_TARGET_ACTIONS,
            submitted_joint_action.select_target >= 0,
        )
    )
    use_ultimate_action_is_out_of_domain = jnp.logical_not(
        jnp.logical_and(
            submitted_joint_action.use_ultimate < NUM_ULTIMATE_ACTIONS,
            submitted_joint_action.use_ultimate >= 0,
        )
    )

    combat_pair_is_out_of_domain = jnp.logical_or(
        select_target_action_is_out_of_domain, use_ultimate_action_is_out_of_domain
    )

    submitted_action_tuple_is_out_of_domain = jnp.logical_or(
        move_action_is_out_of_domain, combat_pair_is_out_of_domain
    )

    # A malformed head canonicalizes that actor's complete tuple to no-op.
    domain_safe_move_action = jnp.where(
        submitted_action_tuple_is_out_of_domain, MOVE_STAY, submitted_joint_action.move
    )
    domain_safe_select_target_action = jnp.where(
        submitted_action_tuple_is_out_of_domain, 0, submitted_joint_action.select_target
    )
    domain_safe_use_ultimate_action = jnp.where(
        submitted_action_tuple_is_out_of_domain, 0, submitted_joint_action.use_ultimate
    )

    submitted_move_action_is_valid_by_actor_slot = current_action_mask.move_mask[
        _GLOBAL_AGENT_SLOT_INDICES, domain_safe_move_action
    ]
    accepted_move_joint_action = jnp.where(
        submitted_move_action_is_valid_by_actor_slot,
        domain_safe_move_action,
        MOVE_STAY,
    )

    submitted_select_target_and_use_ultimate_pair_is_valid_by_actor_slot = (
        current_action_mask.select_target_use_ultimate_joint_mask[
            _GLOBAL_AGENT_SLOT_INDICES,
            domain_safe_select_target_action,
            domain_safe_use_ultimate_action,
        ]
    )

    accepted_select_target_joint_action = jnp.where(
        submitted_select_target_and_use_ultimate_pair_is_valid_by_actor_slot,
        domain_safe_select_target_action,
        0,  # Target-none action
    )
    accepted_use_ultimate_joint_action = jnp.where(
        submitted_select_target_and_use_ultimate_pair_is_valid_by_actor_slot,
        domain_safe_use_ultimate_action,
        0,  # No-ultimate action
    )

    in_domain_move_action_is_rejected_by_actor = jnp.logical_and(
        jnp.logical_not(submitted_action_tuple_is_out_of_domain),
        jnp.logical_not(submitted_move_action_is_valid_by_actor_slot),
    )

    in_domain_combat_action_pair_is_rejected_by_actor = jnp.logical_and(
        jnp.logical_not(submitted_action_tuple_is_out_of_domain),
        jnp.logical_not(
            submitted_select_target_and_use_ultimate_pair_is_valid_by_actor_slot
        ),
    )

    accepted_joint_action = Action(
        accepted_move_joint_action,
        accepted_select_target_joint_action,
        accepted_use_ultimate_joint_action,
    )

    action_acceptance_facts = ActionAcceptanceFacts(
        submitted_joint_action=submitted_joint_action,
        accepted_joint_action=accepted_joint_action,
        submitted_action_tuple_is_out_of_domain_by_actor=(
            submitted_action_tuple_is_out_of_domain
        ),
        in_domain_move_action_is_rejected_by_actor=(
            in_domain_move_action_is_rejected_by_actor
        ),
        in_domain_combat_action_pair_is_rejected_by_actor=(
            in_domain_combat_action_pair_is_rejected_by_actor
        ),
    )

    return accepted_joint_action, action_acceptance_facts


def _build_global_pairwise_actor_and_recipient_target_one_hot_matrix(
    accepted_select_target_joint_action: Array,
) -> tuple[Array, Array, Array]:
    """Route accepted actor-relative targets to fixed global recipients.

    accepted_select_target_joint_action is int32 (10,) in 0..10. Return the
    float32 source/recipient one-hot matrix (10, 10), bool has-recipient flags
    (10,), then int32 recipient IDs (10,). Target None has an all-zero route,
    False flag and ID -1. Each other source has exactly one recipient.
    """
    # Translate actor-relative selections once for every accepted effect lane.
    accepted_global_target_slot_by_actor_slot = (
        _ACTOR_RELATIVE_SELECT_TARGET_ACTION_TO_GLOBAL_AGENT_SLOT_LOOKUP_TABLE[
            _GLOBAL_AGENT_SLOT_INDICES, accepted_select_target_joint_action
        ]
    )

    has_recipient_by_source = accepted_global_target_slot_by_actor_slot > -1

    recipient_global_slot_by_source = jnp.where(
        has_recipient_by_source, accepted_global_target_slot_by_actor_slot, -1
    )

    return (
        jax.nn.one_hot(
            accepted_global_target_slot_by_actor_slot,
            num_classes=MAX_AGENT_SLOTS,
            dtype=jnp.float32,
        ),
        has_recipient_by_source,
        recipient_global_slot_by_source,
    )


def _aggregate_health_effects_and_basic_passives_by_global_slot(
    current_state: EnvState,
    config: EnvConfig,
    accepted_joint_action: Action,
    accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix: Array,
    mage_damage_amplification_aura_multipliers: Array,
    warrior_damage_mitigation_aura_multipliers: Array,
) -> _CombatEffectAggregationResult:
    """Sum simultaneous health effects and retain their accepted-action causes.

    Parameters
    ----------
    current_state : EnvState
        Transition-start health, status and combat countdowns for ten slots.
    config : EnvConfig
        Matching resolved roster and class capabilities.
    accepted_joint_action : Action
        Current accepted int32 action heads (10,).
    accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix : jax.Array
        Float32 (10, 10) source/recipient routes; Target None rows are zero.
    mage_damage_amplification_aura_multipliers : jax.Array
        Current float32 (10,) outgoing-damage factors by source.
    warrior_damage_mitigation_aura_multipliers : jax.Array
        Current float32 (10,) incoming-damage factors by recipient.

    Returns
    -------
    _CombatEffectAggregationResult
        Source values, recipient health totals and bool application/participation
        causes. All vectors have shape (10,); magnitudes are float32.

    Notes
    -----
    Basic and Ultimate routes are mutually exclusive after acceptance. Reduce
    all source contributions before changing health so agents resolve together.
    Current Burst and aura values affect damage; current Poison affects healing.
    Trap break uses accepted positive raw damage, not realized health loss.
    Healing resets combat only when its recipient was already in combat at
    transition start, preventing same-transition propagation through heal chains.
    Inputs are not mutated and successor status applications are handled later.
    """
    # Pre-state source and recipient modifiers affect this transition's payloads.
    mage_burst_damage_amplification_multipliers = jnp.where(
        current_state.mage_burst_damage_amplification_durations > 0,
        MAGE_BURST_DAMAGE_MULTIPLIER,
        1.0,
    )

    rogue_poison_anti_heal_multipliers_by_global_recipient_slot = (
        build_rogue_poison_anti_heal_multipliers(
            current_state.rogue_poison_anti_heal_durations
        )
    )

    # Basic and ultimate lanes are mutually exclusive after action acceptance.
    actor_applies_accepted_basic_effect = jnp.logical_and(
        accepted_joint_action.use_ultimate == 0,
        accepted_joint_action.select_target > 0,
    )

    basic_effect_source_class_ids_by_actor_slot = jnp.where(
        actor_applies_accepted_basic_effect,
        config.agent_profile.class_ids,
        NEUTRAL_CLASS_ID,
    )
    raw_basic_damage_by_actor_slot = BASIC_DAMAGE_BY_CLASS[
        basic_effect_source_class_ids_by_actor_slot
    ]
    raw_basic_healing_by_actor_slot = BASIC_HEALING_BY_CLASS[
        basic_effect_source_class_ids_by_actor_slot
    ]

    amplified_basic_damage_by_actor_slot = (
        raw_basic_damage_by_actor_slot
        * mage_burst_damage_amplification_multipliers
        * mage_damage_amplification_aura_multipliers
    )

    basic_damage_contribution_by_actor_and_global_recipient_slot = (
        accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix
        * amplified_basic_damage_by_actor_slot[:, None]
    )

    basic_healing_contribution_by_actor_and_global_recipient_slot = (
        accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix
        * raw_basic_healing_by_actor_slot[:, None]
    )

    # No-target ultimates are accepted actions but have no routed health payload.
    actor_applies_accepted_targeted_ultimate_effect = jnp.logical_and(
        accepted_joint_action.use_ultimate == 1,
        accepted_joint_action.select_target > 0,
    )
    targeted_ultimate_source_class_ids_by_actor_slot = jnp.where(
        actor_applies_accepted_targeted_ultimate_effect,
        config.agent_profile.class_ids,
        NEUTRAL_CLASS_ID,
    )
    raw_ultimate_damage_by_actor_slot = ULTIMATE_DAMAGE_BY_CLASS[
        targeted_ultimate_source_class_ids_by_actor_slot
    ]
    raw_ultimate_healing_by_actor_slot = ULTIMATE_HEALING_BY_CLASS[
        targeted_ultimate_source_class_ids_by_actor_slot
    ]

    amplified_ultimate_damage_by_actor_slot = (
        raw_ultimate_damage_by_actor_slot
        * mage_damage_amplification_aura_multipliers
        * mage_burst_damage_amplification_multipliers
    )

    ultimate_damage_contribution_by_actor_and_global_recipient_slot = (
        accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix
        * amplified_ultimate_damage_by_actor_slot[:, None]
    )

    ultimate_healing_contribution_by_actor_and_global_recipient_slot = (
        accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix
        * raw_ultimate_healing_by_actor_slot[:, None]
    )

    # Trap break follows accepted positive raw damage, not effective health loss.
    accepted_positive_raw_basic_damage_received_this_tick = (
        jnp.sum(
            accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix
            * raw_basic_damage_by_actor_slot[:, None],
            axis=0,
        )
        > 0
    )

    accepted_positive_raw_ultimate_damage_received_this_tick = (
        jnp.sum(
            accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix
            * raw_ultimate_damage_by_actor_slot[:, None],
            axis=0,
        )
        > 0
    )

    # Both damage lanes share one recipient-level Trap-break predicate.
    accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot = (
        jnp.logical_or(
            accepted_positive_raw_basic_damage_received_this_tick,
            accepted_positive_raw_ultimate_damage_received_this_tick,
        )
    )

    # Reuse accepted recipient routing for source-specific basic passives.
    hunter_basic_slow_applied_this_tick_by_global_actor_slot = jnp.logical_and(
        _active_hunter_class_mask(config),
        actor_applies_accepted_basic_effect,
    )

    hunter_basic_slow_applied_this_tick_by_global_slot_mask = jnp.logical_and(
        accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
        hunter_basic_slow_applied_this_tick_by_global_actor_slot[:, None],
    )

    hunter_basic_slow_applied_this_tick_by_global_recipient_slot = jnp.any(
        hunter_basic_slow_applied_this_tick_by_global_slot_mask, axis=0
    )

    priest_freedom_applied_this_tick_by_global_actor_slot = jnp.logical_and(
        _active_priest_class_mask(config),
        actor_applies_accepted_basic_effect,
    )

    priest_freedom_applied_this_tick_by_global_slot_mask = jnp.logical_and(
        accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
        priest_freedom_applied_this_tick_by_global_actor_slot[:, None],
    )

    priest_freedom_applied_this_tick_by_global_recipient_slot = jnp.any(
        priest_freedom_applied_this_tick_by_global_slot_mask, axis=0
    )

    # Recipient modifiers apply after source contributions aggregate.
    total_damage_received_by_global_recipient_slot = (
        jnp.sum(
            basic_damage_contribution_by_actor_and_global_recipient_slot
            + ultimate_damage_contribution_by_actor_and_global_recipient_slot,
            axis=0,
        )
        * warrior_damage_mitigation_aura_multipliers
    )

    total_healing_received_by_global_recipient_slot = (
        jnp.sum(
            basic_healing_contribution_by_actor_and_global_recipient_slot
            + ultimate_healing_contribution_by_actor_and_global_recipient_slot,
            axis=0,
        )
        * rogue_poison_anti_heal_multipliers_by_global_recipient_slot
    )

    basic_effect_is_activated_by_source = actor_applies_accepted_basic_effect
    ultimate_effect_is_activated_by_source = accepted_joint_action.use_ultimate.astype(
        jnp.bool_
    )

    raw_healing_output_by_source = (
        raw_basic_healing_by_actor_slot + raw_ultimate_healing_by_actor_slot
    )
    source_modified_damage_output_by_source = (
        amplified_basic_damage_by_actor_slot + amplified_ultimate_damage_by_actor_slot
    )

    # Route each recipient's mitigation factor back to contributing sources.
    raw_damage_output_by_source = (
        raw_basic_damage_by_actor_slot + raw_ultimate_damage_by_actor_slot
    )
    accepted_damage_global_pairwise_actor_and_recipient_target_one_hot_matrix = (
        jnp.logical_and(
            accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
            (raw_damage_output_by_source > 0)[:, None],
        )
    )
    recipient_damage_modifier_by_source = jnp.sum(
        accepted_damage_global_pairwise_actor_and_recipient_target_one_hot_matrix
        * warrior_damage_mitigation_aura_multipliers[None, :],
        axis=-1,
    )

    total_effective_damage_by_recipient = total_damage_received_by_global_recipient_slot

    # There are currently no healing amplifiers in the game.
    source_modified_healing_output_by_source = raw_healing_output_by_source

    accepted_healing_global_pairwise_actor_and_recipient_target_one_hot_matrix = (
        jnp.logical_and(
            accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
            (raw_healing_output_by_source > 0)[:, None],
        )
    )
    recipient_healing_modifier_by_source = jnp.sum(
        accepted_healing_global_pairwise_actor_and_recipient_target_one_hot_matrix
        * rogue_poison_anti_heal_multipliers_by_global_recipient_slot[None, :],
        axis=-1,
    )

    total_effective_healing_by_recipient = (
        total_healing_received_by_global_recipient_slot
    )

    # Healing qualification reads only transition-start combat truth. Combine
    # healing and damage participants only after that snapshot decision so a
    # same-transition reset cannot propagate through another healing route.
    is_currently_in_combat = current_state.steps_until_out_of_combat > 0
    combat_healing_source_this_tick_by_agent = jnp.any(
        jnp.logical_and(
            accepted_healing_global_pairwise_actor_and_recipient_target_one_hot_matrix,
            is_currently_in_combat[None, :],
        ),
        axis=1,
    )
    combat_healing_recipient_this_tick_by_agent = jnp.logical_and(
        is_currently_in_combat,
        jnp.any(
            accepted_healing_global_pairwise_actor_and_recipient_target_one_hot_matrix,
            axis=0,
        ),
    )
    combat_healing_participation_this_tick_by_agent = jnp.logical_or(
        combat_healing_source_this_tick_by_agent,
        combat_healing_recipient_this_tick_by_agent,
    )

    damage_source_this_tick_by_agent = jnp.any(
        accepted_damage_global_pairwise_actor_and_recipient_target_one_hot_matrix,
        axis=1,
    )
    damage_recipient_this_tick_by_agent = jnp.any(
        accepted_damage_global_pairwise_actor_and_recipient_target_one_hot_matrix,
        axis=0,
    )
    combat_damage_participation_this_tick_by_agent = jnp.logical_or(
        damage_source_this_tick_by_agent, damage_recipient_this_tick_by_agent
    )

    is_combat_participant_this_tick_by_source = jnp.logical_or(
        combat_healing_participation_this_tick_by_agent,
        combat_damage_participation_this_tick_by_agent,
    )

    return _CombatEffectAggregationResult(
        hunter_basic_slow_applied_this_tick_by_global_recipient_slot=(
            hunter_basic_slow_applied_this_tick_by_global_recipient_slot
        ),
        priest_freedom_applied_this_tick_by_global_recipient_slot=(
            priest_freedom_applied_this_tick_by_global_recipient_slot
        ),
        accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot=(
            accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot
        ),
        hunter_basic_slow_applied_this_tick_by_global_actor_slot=(
            hunter_basic_slow_applied_this_tick_by_global_actor_slot
        ),
        basic_effect_is_activated_by_source=basic_effect_is_activated_by_source,
        ultimate_effect_is_activated_by_source=ultimate_effect_is_activated_by_source,
        raw_damage_output_by_source=raw_damage_output_by_source,
        source_modified_damage_output_by_source=(
            source_modified_damage_output_by_source
        ),
        recipient_damage_modifier_by_source=recipient_damage_modifier_by_source,
        total_effective_damage_by_recipient=total_effective_damage_by_recipient,
        raw_healing_output_by_source=raw_healing_output_by_source,
        source_modified_healing_output_by_source=(
            source_modified_healing_output_by_source
        ),
        recipient_healing_modifier_by_source=recipient_healing_modifier_by_source,
        total_effective_healing_by_recipient=total_effective_healing_by_recipient,
        priest_blessing_of_freedom_is_applied_by_source=(
            priest_freedom_applied_this_tick_by_global_actor_slot
        ),
        is_combat_participant_this_tick_by_source=is_combat_participant_this_tick_by_source,
    )


def _compute_health_after_simultaneous_damage_and_healing(
    total_damage_received_by_global_slot: Array,
    total_healing_received_by_global_slot: Array,
    current_state: EnvState,
    config: EnvConfig,
) -> Array:
    """Apply net simultaneous health effects and clip to each class maximum.

    Damage and healing totals are float32 (10,) in global recipient order.
    current_state supplies starting health; config supplies maxima. Return
    float32 (10,) health after their net change, clipped to [0, max_health].
    This precedes regeneration, death resolution and respawn; no source is
    chosen as a killer and no individual effect is clipped before summation.
    """
    health_delta_by_slot = (
        total_healing_received_by_global_slot - total_damage_received_by_global_slot
    )

    net_health_by_slot = current_state.current_health + health_delta_by_slot
    return jnp.clip(
        net_health_by_slot,
        min=0,
        max=config.agent_profile.max_health,
    )


def _derive_aura_damage_multipliers(
    config: EnvConfig,
    global_pairwise_distances: Array,
    alive_mask: Array,
    is_not_under_spawn_shield: Array,
) -> _CombatAuraAggregationResult:
    """Compute current Mage outgoing and Warrior incoming aura multipliers.

    config provides classes and teams; global_pairwise_distances is float32
    (10, 10). alive_mask and is_not_under_spawn_shield are bool (10,) from the
    same snapshot. Return two float32 multiplier vectors (10,) plus bool
    emitter/beneficiary coverage matrices (10, 10) in _CombatAuraAggregationResult.
    Only configured living unshielded allies participate. Eligible emitters
    benefit from their own aura. Inclusive range gates precede multiplication
    and catalog caps; no sight gate is applied to auras.
    """
    global_pairwise_ally_mask, _ = _build_global_pairwise_team_masks(
        config.agent_profile.team_ids
    )

    active_and_alive = jnp.logical_and(alive_mask, config.agent_profile.active_mask)
    active_and_alive_pairs = jnp.logical_and(
        active_and_alive[None, :], active_and_alive[:, None]
    )
    global_pairwise_active_and_alive_ally_mask = jnp.logical_and(
        jnp.logical_and(active_and_alive_pairs, global_pairwise_ally_mask),
        jnp.logical_and(
            is_not_under_spawn_shield[:, None],
            is_not_under_spawn_shield[None, :],
        ),
    )

    mage_masked_global_pairwise_distances = jnp.where(
        jnp.logical_and(
            _active_mage_class_mask(config)[:, None],
            global_pairwise_active_and_alive_ally_mask,
        ),
        global_pairwise_distances,
        jnp.inf,
    )

    warrior_masked_global_pairwise_distances = jnp.where(
        jnp.logical_and(
            _active_warrior_class_mask(config)[:, None],
            global_pairwise_active_and_alive_ally_mask,
        ),
        global_pairwise_distances,
        jnp.inf,
    )

    is_covered_by_mage_damage_aura_by_emitter_and_beneficiary = (
        mage_masked_global_pairwise_distances <= MAGE_DAMAGE_AMPLIFICATION_AURA_RADIUS
    )
    mage_actor_benefits_recipient_with_aura = jnp.where(
        is_covered_by_mage_damage_aura_by_emitter_and_beneficiary,
        MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER,
        1.0,
    )

    is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary = (
        warrior_masked_global_pairwise_distances
        <= WARRIOR_DAMAGE_MITIGATION_AURA_RADIUS
    )
    warrior_actor_benefits_recipient_with_aura = jnp.where(
        is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary,
        WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER,
        1.0,
    )

    mage_aura_outgoing_damage_multiplier_by_actor_slot = jnp.prod(
        mage_actor_benefits_recipient_with_aura, axis=0
    )
    warrior_aura_incoming_damage_multiplier_by_global_recipient_slot = jnp.prod(
        warrior_actor_benefits_recipient_with_aura, axis=0
    )

    return _CombatAuraAggregationResult(
        jnp.clip(
            mage_aura_outgoing_damage_multiplier_by_actor_slot,
            min=1.0,
            max=MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER_CEILING,
        ),
        jnp.clip(
            warrior_aura_incoming_damage_multiplier_by_global_recipient_slot,
            min=WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER_FLOOR,
            max=1.0,
        ),
        AuraTransitionFacts(
            is_covered_by_mage_damage_aura_by_emitter_and_beneficiary,
            is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary,
        ),
    )


def _resolve_status_duration_lifecycle(
    current_state: EnvState,
    config: EnvConfig,
    accepted_use_ultimate_by_actor_slot: Array,
    accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix: Array,
    hunter_basic_slow_applied_this_tick_by_global_recipient_slot: Array,
    priest_freedom_applied_this_tick_by_global_recipient_slot: Array,
    accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot: Array,
    hunter_basic_slow_applied_this_tick_by_global_actor_slot: Array,
    next_alive_mask: Array,
    is_newly_dead: Array,
) -> _CombatStatusAggregationResult:
    """Build successor durations and separate status-application causes.

    Parameters
    ----------
    current_state : EnvState
        Current durations and shields for the decision that just acted.
    config : EnvConfig
        Matching configured classes and participation mask.
    accepted_use_ultimate_by_actor_slot : jax.Array
        Int32 (10,) accepted Ultimate categories, zero or one.
    accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix : jax.Array
        Float32 (10, 10) accepted source/recipient routes.
    hunter_basic_slow_applied_this_tick_by_global_recipient_slot : jax.Array
        Bool (10,) recipients of an accepted Hunter Basic slow.
    priest_freedom_applied_this_tick_by_global_recipient_slot : jax.Array
        Bool (10,) recipients of an accepted Priest Basic movement floor.
    accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot : jax.Array
        Bool (10,) recipients of positive raw damage, used for old Trap break.
    hunter_basic_slow_applied_this_tick_by_global_actor_slot : jax.Array
        Bool (10,) Hunter sources, retained separately for application facts.
    next_alive_mask : jax.Array
        Bool (10,) alive status after health/death resolution, before respawn.
    is_newly_dead : jax.Array
        Bool (10,) actors that died in this transition.

    Returns
    -------
    _CombatStatusAggregationResult
        Int32 successor durations, bool source application masks, and the
        bool (10, 9) recipient lifecycle causes in their shared channel order.

    Notes
    -----
    Age old durations once. Positive raw damage then clears the aged remainder
    of an old Hunter Trap. Merge fresh applications at full duration without
    shortening a longer remainder. They first govern the next policy decision.
    Death clears successor durations but retains application facts. Shields age
    after movement; a later respawn creates its new shield without ageing it.
    Expiry, damage break, refresh and death clearing remain independent facts.
    """
    current_status_durations = jnp.concatenate(
        (
            current_state.slow_durations,
            current_state.stun_durations,
            current_state.rogue_poison_anti_heal_durations[:, None],
            current_state.mage_burst_damage_amplification_durations[:, None],
            current_state.priest_blessing_of_freedom_slow_floor_durations[:, None],
        ),
        axis=-1,
        dtype=jnp.int32,
    )

    # Derive fresh applications once from the action accepted for this transition.
    (
        mage_uses_accepted_ultimate_this_tick_by_actor_slot,
        warrior_uses_accepted_ultimate_this_tick_by_actor_slot,
        hunter_uses_accepted_ultimate_this_tick_by_actor_slot,
        rogue_uses_accepted_ultimate_this_tick_by_actor_slot,
        warrior_charge_applied_this_tick_by_recipient_slot,
        hunter_trap_applied_this_tick_by_recipient_slot,
        rogue_poison_applied_this_tick_by_recipient_slot,
    ) = _derive_accepted_ultimate_status_applications(
        config,
        accepted_use_ultimate_by_actor_slot,
        accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
    )

    # Age every current duration exactly once for successor-state memory.
    decremented_slow_durations = jnp.maximum(0, current_state.slow_durations - 1)
    decremented_stun_durations = jnp.maximum(0, current_state.stun_durations - 1)
    decremented_rogue_anti_heal_durations = jnp.maximum(
        0, current_state.rogue_poison_anti_heal_durations - 1
    )
    decremented_mage_burst_durations = jnp.maximum(
        0, current_state.mage_burst_damage_amplification_durations - 1
    )
    decremented_priest_freedom_slow_floor_durations = jnp.maximum(
        0, current_state.priest_blessing_of_freedom_slow_floor_durations - 1
    )

    # Pack every duration family into the public nine-channel order.
    aged_next_status_durations = jnp.concatenate(
        (
            decremented_slow_durations,
            decremented_stun_durations,
            decremented_rogue_anti_heal_durations[:, None],
            decremented_mage_burst_durations[:, None],
            decremented_priest_freedom_slow_floor_durations[:, None],
        ),
        axis=-1,
        dtype=jnp.int32,
    )

    # Natural expiry is authored before Trap break or fresh application.
    aged_to_zero_by_recipient_and_status_channel = jnp.logical_and(
        current_status_durations == 1, aged_next_status_durations == 0
    )

    # Raw damage breaks only the pre-existing Trap successor. A fresh Trap is
    # merged later, so damage cannot retroactively break the new application.
    decremented_stun_durations_after_trap_break = decremented_stun_durations.at[
        :, STUN_CHANNEL_HUNTER_TRAP
    ].set(
        jnp.where(
            accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot,
            0,
            decremented_stun_durations[:, STUN_CHANNEL_HUNTER_TRAP],
        )
    )

    # Hunter Trap is the only damage-breakable status in the current catalog.
    hunter_trap_broken_by_damage_by_recipient = jnp.logical_and(
        accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot,
        decremented_stun_durations[:, STUN_CHANNEL_HUNTER_TRAP] >= 1,
    )
    broken_by_damage_by_recipient_and_status_channel = (
        jnp.zeros((MAX_AGENT_SLOTS, 9), dtype=jnp.bool_)
        # Column 4 is Hunter Trap in the canonical combined status order.
        .at[:, NUM_SLOW_CHANNELS + STUN_CHANNEL_HUNTER_TRAP]
        .set(hunter_trap_broken_by_damage_by_recipient)
    )

    # Mage Burst is source-local rather than recipient-routed.
    next_mage_burst_durations = jnp.where(
        mage_uses_accepted_ultimate_this_tick_by_actor_slot,
        jnp.maximum(
            MAGE_BURST_DAMAGE_DURATION_TICKS,
            decremented_mage_burst_durations,
        ),
        decremented_mage_burst_durations,
    )

    # Rogue anti-heal shares Poison's accepted recipient.
    next_rogue_anti_heal_durations = jnp.where(
        rogue_poison_applied_this_tick_by_recipient_slot,
        jnp.maximum(
            ROGUE_POISON_ANTI_HEAL_DURATION_TICKS,
            decremented_rogue_anti_heal_durations,
        ),
        decremented_rogue_anti_heal_durations,
    )

    # Preserve source identity by refreshing each stun channel independently.
    decremented_warrior_charge_stun_durations = (
        decremented_stun_durations_after_trap_break[:, STUN_CHANNEL_WARRIOR_CHARGE]
    )
    decremented_hunter_trap_stun_durations = (
        decremented_stun_durations_after_trap_break[:, STUN_CHANNEL_HUNTER_TRAP]
    )
    decremented_rogue_poison_stun_durations = (
        decremented_stun_durations_after_trap_break[:, STUN_CHANNEL_ROGUE_POISON]
    )

    next_warrior_charge_stun_durations = jnp.where(
        warrior_charge_applied_this_tick_by_recipient_slot,
        jnp.maximum(
            WARRIOR_CHARGE_STUN_DURATION_TICKS,
            decremented_warrior_charge_stun_durations,
        ),
        decremented_warrior_charge_stun_durations,
    )
    next_stun_durations = decremented_stun_durations_after_trap_break.at[
        :, STUN_CHANNEL_WARRIOR_CHARGE
    ].set(next_warrior_charge_stun_durations)

    next_hunter_trap_stun_durations = jnp.where(
        hunter_trap_applied_this_tick_by_recipient_slot,
        jnp.maximum(
            HUNTER_TRAP_STUN_DURATION_TICKS, decremented_hunter_trap_stun_durations
        ),
        decremented_hunter_trap_stun_durations,
    )
    next_stun_durations = next_stun_durations.at[:, STUN_CHANNEL_HUNTER_TRAP].set(
        next_hunter_trap_stun_durations
    )

    next_rogue_poison_stun_durations = jnp.where(
        rogue_poison_applied_this_tick_by_recipient_slot,
        jnp.maximum(
            ROGUE_POISON_STUN_DURATION_TICKS, decremented_rogue_poison_stun_durations
        ),
        decremented_rogue_poison_stun_durations,
    )
    next_stun_durations = next_stun_durations.at[:, STUN_CHANNEL_ROGUE_POISON].set(
        next_rogue_poison_stun_durations
    )

    # Refresh each slow source independently without disturbing other channels.
    decremented_warrior_charge_slow_durations = decremented_slow_durations[
        :, SLOW_CHANNEL_WARRIOR_CHARGE
    ]
    decremented_hunter_basic_slow_durations = decremented_slow_durations[
        :, SLOW_CHANNEL_HUNTER_BASIC
    ]
    decremented_rogue_poison_slow_durations = decremented_slow_durations[
        :, SLOW_CHANNEL_ROGUE_POISON
    ]

    next_warrior_charge_slow_durations = jnp.where(
        warrior_charge_applied_this_tick_by_recipient_slot,
        jnp.maximum(
            WARRIOR_CHARGE_SLOW_DURATION_TICKS,
            decremented_warrior_charge_slow_durations,
        ),
        decremented_warrior_charge_slow_durations,
    )
    next_slow_durations = decremented_slow_durations.at[
        :, SLOW_CHANNEL_WARRIOR_CHARGE
    ].set(next_warrior_charge_slow_durations)

    next_hunter_basic_slow_durations = jnp.where(
        hunter_basic_slow_applied_this_tick_by_global_recipient_slot,
        jnp.maximum(
            HUNTER_BASIC_SLOW_DURATION_TICKS, decremented_hunter_basic_slow_durations
        ),
        decremented_hunter_basic_slow_durations,
    )
    next_slow_durations = next_slow_durations.at[:, SLOW_CHANNEL_HUNTER_BASIC].set(
        next_hunter_basic_slow_durations
    )

    next_rogue_poison_slow_durations = jnp.where(
        rogue_poison_applied_this_tick_by_recipient_slot,
        jnp.maximum(
            ROGUE_POISON_SLOW_DURATION_TICKS, decremented_rogue_poison_slow_durations
        ),
        decremented_rogue_poison_slow_durations,
    )
    next_slow_durations = next_slow_durations.at[:, SLOW_CHANNEL_ROGUE_POISON].set(
        next_rogue_poison_slow_durations
    )

    # Freedom raises the movement floor without clearing any slow channel.
    next_priest_freedom_slow_floor_durations = jnp.where(
        priest_freedom_applied_this_tick_by_global_recipient_slot,
        jnp.maximum(
            PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS,
            decremented_priest_freedom_slow_floor_durations,
        ),
        decremented_priest_freedom_slow_floor_durations,
    ).astype(jnp.int32)

    # Preserve the source and mechanic channel of each accepted application.
    warrior_slow_applied_by_source_and_channel = (
        warrior_uses_accepted_ultimate_this_tick_by_actor_slot[:, None]
    )
    hunter_slow_applied_by_source_and_channel = (
        hunter_basic_slow_applied_this_tick_by_global_actor_slot[:, None]
    )
    rogue_slow_applied_by_source_and_channel = (
        rogue_uses_accepted_ultimate_this_tick_by_actor_slot[:, None]
    )

    slow_is_applied_by_source_and_channel = jnp.concatenate(
        (
            warrior_slow_applied_by_source_and_channel,
            hunter_slow_applied_by_source_and_channel,
            rogue_slow_applied_by_source_and_channel,
        ),
        axis=-1,
    )

    warrior_stun_applied_by_source_and_channel = (
        warrior_uses_accepted_ultimate_this_tick_by_actor_slot[:, None]
    )
    hunter_stun_applied_by_source_and_channel = (
        hunter_uses_accepted_ultimate_this_tick_by_actor_slot[:, None]
    )
    rogue_stun_applied_by_source_and_channel = (
        rogue_uses_accepted_ultimate_this_tick_by_actor_slot[:, None]
    )

    stun_is_applied_by_source_and_channel = jnp.concatenate(
        (
            warrior_stun_applied_by_source_and_channel,
            hunter_stun_applied_by_source_and_channel,
            rogue_stun_applied_by_source_and_channel,
        ),
        axis=-1,
    )

    rogue_poison_anti_heal_is_applied_by_source = (
        rogue_uses_accepted_ultimate_this_tick_by_actor_slot
    )

    mage_burst_damage_amplification_is_applied_by_source = (
        mage_uses_accepted_ultimate_this_tick_by_actor_slot
    )

    # A refresh restores or extends a still-positive status after ordinary age.
    # Natural expiry followed by application and Trap break followed by
    # reapplication remain separate causal edges rather than refreshes.
    pre_death_next_status_durations = jnp.concatenate(
        (
            next_slow_durations,
            next_stun_durations,
            next_rogue_anti_heal_durations[:, None],
            next_mage_burst_durations[:, None],
            next_priest_freedom_slow_floor_durations[:, None],
        ),
        axis=-1,
        dtype=jnp.int32,
    )

    refreshed_or_extended_by_recipient_and_status_channel = jnp.logical_and(
        jnp.logical_and(
            pre_death_next_status_durations >= current_status_durations,
            jnp.logical_and(
                current_status_durations > 0,
                jnp.logical_not(aged_to_zero_by_recipient_and_status_channel),
            ),
        ),
        jnp.logical_not(broken_by_damage_by_recipient_and_status_channel),
    )

    # Transient status memory cannot survive into a dead successor slot.
    death_masked_next_slow_durations = next_slow_durations * next_alive_mask[:, None]

    death_masked_next_stun_durations = next_stun_durations * next_alive_mask[:, None]

    death_masked_next_mage_burst_durations = next_mage_burst_durations * next_alive_mask

    death_masked_next_rogue_anti_heal_durations = (
        next_rogue_anti_heal_durations * next_alive_mask
    )

    death_masked_next_priest_freedom_slow_floor_durations = (
        next_priest_freedom_slow_floor_durations * next_alive_mask
    )

    # Spawn shielding ages after movement and cannot survive death or padding.
    death_masked_next_spawn_shield_durations = (
        jnp.maximum(current_state.spawn_shield_durations - 1, 0).astype(jnp.int32)
        * next_alive_mask
        * config.agent_profile.active_mask
    )

    # New death clears only statuses that survived through the application phase.
    cleared_by_new_death_by_recipient_and_status_channel = jnp.logical_and(
        pre_death_next_status_durations > 0, is_newly_dead[:, None]
    )

    status_lifecycle_facts = StatusLifecycleTransitionFacts(
        aged_to_zero_by_recipient_and_status_channel,
        refreshed_or_extended_by_recipient_and_status_channel,
        broken_by_damage_by_recipient_and_status_channel,
        cleared_by_new_death_by_recipient_and_status_channel,
    )

    return _CombatStatusAggregationResult(
        death_masked_next_slow_durations,
        death_masked_next_stun_durations,
        death_masked_next_rogue_anti_heal_durations,
        death_masked_next_mage_burst_durations,
        death_masked_next_priest_freedom_slow_floor_durations,
        death_masked_next_spawn_shield_durations,
        slow_is_applied_by_source_and_channel,
        stun_is_applied_by_source_and_channel,
        rogue_poison_anti_heal_is_applied_by_source,
        mage_burst_damage_amplification_is_applied_by_source,
        status_lifecycle_facts,
    )


def _derive_accepted_ultimate_status_applications(
    config: EnvConfig,
    accepted_use_ultimate_by_actor_slot: Array,
    accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Identify accepted Ultimate sources and their status recipients.

    config supplies configured classes, accepted_use_ultimate_by_actor_slot is
    int32 (10,), and the float32 (10, 10) matrix routes accepted targets.
    Return seven bool (10,) arrays in order: Mage, Warrior, Hunter and Rogue
    source flags; then Warrior Charge, Hunter Trap and Rogue Poison recipient
    flags. Mage Burst stays on its source. No durations are changed here.
    """
    uses_ultimate_this_tick = accepted_use_ultimate_by_actor_slot == 1

    # Mage Burst applies to its source; targeted ultimates reduce by recipient.
    mage_uses_accepted_ultimate_this_tick_by_actor_slot = jnp.logical_and(
        _active_mage_class_mask(config), uses_ultimate_this_tick
    )

    warrior_uses_accepted_ultimate_this_tick_by_actor_slot = jnp.logical_and(
        _active_warrior_class_mask(config), uses_ultimate_this_tick
    )
    warrior_charge_applied_this_tick_by_recipient_slot = jnp.any(
        jnp.logical_and(
            warrior_uses_accepted_ultimate_this_tick_by_actor_slot[:, None],
            accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
        ),
        axis=0,
    )

    hunter_uses_accepted_ultimate_this_tick_by_actor_slot = jnp.logical_and(
        _active_hunter_class_mask(config), uses_ultimate_this_tick
    )
    hunter_trap_applied_this_tick_by_recipient_slot = jnp.any(
        jnp.logical_and(
            hunter_uses_accepted_ultimate_this_tick_by_actor_slot[:, None],
            accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
        ),
        axis=0,
    )

    rogue_uses_accepted_ultimate_this_tick_by_actor_slot = jnp.logical_and(
        _active_rogue_class_mask(config), uses_ultimate_this_tick
    )
    rogue_poison_applied_this_tick_by_recipient_slot = jnp.any(
        jnp.logical_and(
            rogue_uses_accepted_ultimate_this_tick_by_actor_slot[:, None],
            accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
        ),
        axis=0,
    )

    return (
        mage_uses_accepted_ultimate_this_tick_by_actor_slot,
        warrior_uses_accepted_ultimate_this_tick_by_actor_slot,
        hunter_uses_accepted_ultimate_this_tick_by_actor_slot,
        rogue_uses_accepted_ultimate_this_tick_by_actor_slot,
        warrior_charge_applied_this_tick_by_recipient_slot,
        hunter_trap_applied_this_tick_by_recipient_slot,
        rogue_poison_applied_this_tick_by_recipient_slot,
    )


def _return_unchanged_agent_positions(
    agent_positions: Array,
    agent_radii: Array,
    intended_movement_deltas: Array,
    active_mask: Array,
    alive_mask: Array,
    map_width: Array | float,
    map_height: Array | float,
    obstacles: Array,
    always_participates_in_agent_agent_collision: Array,
    participates_in_agent_agent_collision_at_final_position: Array,
) -> Array:
    """Return agent_positions unchanged when no accepted Charge needs projection.

    Positions are float32 (10, 2). All remaining operands match the Charge
    geometry call but are unused: radii/deltas, active/alive flags, map bounds,
    obstacle rows and traversal/final-contact masks. Matching operands lets
    lax.cond skip that numerical branch without changing its return structure.
    """
    del (
        agent_radii,
        intended_movement_deltas,
        active_mask,
        alive_mask,
        map_width,
        map_height,
        obstacles,
        always_participates_in_agent_agent_collision,
        participates_in_agent_agent_collision_at_final_position,
    )

    return agent_positions


def _resolve_post_charge_agent_positions(
    current_state: EnvState,
    config: EnvConfig,
    accepted_joint_action: Action,
    accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix: Array,
    always_participates_in_agent_agent_collision: Array,
) -> Array:
    """Resolve all accepted Warrior Charge arrivals from one transition start.

    current_state and config supply positions, bodies and map geometry.
    accepted_joint_action has int32 heads (10,); its float32 (10, 10) routing
    matrix gives each source at most one recipient. The bool (10,) collision
    mask controls both Charge contact stages. Return float32 positions (10, 2).
    Aim at the source-facing tangent around each accepted recipient, then use
    shared endpoint geometry. No accepted Charge returns the input positions.
    Already chosen ordinary movement resolves from these arrivals afterward.
    """

    # Accepted ultimate use is already validated against the current mask.
    accepted_warrior_charge_by_actor = jnp.logical_and(
        accepted_joint_action.use_ultimate == 1, _active_warrior_class_mask(config)
    )

    # Retain only accepted Warrior source-recipient pairs.
    charge_actor_and_recipient_pairs = jnp.logical_and(
        accepted_warrior_charge_by_actor[:, None],
        accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
    )

    pairwise_displacement_vectors_to_recipient_from_actor = (
        current_state.agent_positions[None, :, :]
        - current_state.agent_positions[:, None, :]
    )

    norm = cast(
        Array,
        jnp.linalg.norm(pairwise_displacement_vectors_to_recipient_from_actor, axis=-1)[
            :, :, None
        ],
    )
    safe_norm = jnp.where(norm > 0, norm, jnp.ones_like(norm))
    pairwise_direction_vectors_to_recipient_from_actor = (
        pairwise_displacement_vectors_to_recipient_from_actor / safe_norm
    )

    radii_scaled_pairwise_direction_vectors_to_recipient_from_actor = (
        config.agent_profile.agent_radii[:, None, None]
        + config.agent_profile.agent_radii[None, :, None]
    ) * pairwise_direction_vectors_to_recipient_from_actor

    adjusted_agent_position_deltas = (
        pairwise_displacement_vectors_to_recipient_from_actor
        - radii_scaled_pairwise_direction_vectors_to_recipient_from_actor
    )

    post_charge_current_agent_position_deltas = jnp.where(
        charge_actor_and_recipient_pairs[:, :, None],
        adjusted_agent_position_deltas,
        jnp.zeros_like(adjusted_agent_position_deltas),
    )

    # Each actor has at most one accepted recipient, so reduce recipient rows
    # into one slot-aligned forced displacement.
    intended_movement_deltas = jnp.sum(
        post_charge_current_agent_position_deltas, axis=1
    )

    there_is_a_charging_warrior = jnp.any(accepted_warrior_charge_by_actor)

    return cast(
        Array,
        jax.lax.cond(
            there_is_a_charging_warrior,
            project_charge_endpoints_with_geometry,
            _return_unchanged_agent_positions,
            current_state.agent_positions,
            config.agent_profile.agent_radii,
            intended_movement_deltas,
            config.agent_profile.active_mask,
            current_state.alive_mask,
            config.map_width,
            config.map_height,
            config.obstacles,
            always_participates_in_agent_agent_collision,
            always_participates_in_agent_agent_collision,
        ),
    )


def _build_combat_transition_facts(
    combat_effect_aggregation_result: _CombatEffectAggregationResult,
    combat_effect_has_recipient_by_source: Array,
    combat_effect_recipient_global_slot_by_source: Array,
    slow_is_applied_by_source_and_channel: Array,
    stun_is_applied_by_source_and_channel: Array,
    rogue_poison_anti_heal_is_applied_by_source: Array,
    mage_burst_damage_amplification_is_applied_by_source: Array,
    next_health_after_effective_damage_and_healing: Array,
) -> CombatTransitionFacts:
    """Package existing combat results without recomputing effect semantics.

    combat_effect_aggregation_result supplies source values and recipient totals.
    has-recipient flags are bool (10,) and recipient IDs int32 (10,), with -1
    for none. Slow/stun application masks are bool (10, 3); Poison and Burst
    flags are bool (10,). The final float32 (10,) health argument is the result
    after combat and before regeneration. Return CombatTransitionFacts carrying
    these same arrays; this packaging does not advance state or filter policies.
    """
    return CombatTransitionFacts(
        basic_effect_is_activated_by_source=(
            combat_effect_aggregation_result.basic_effect_is_activated_by_source
        ),
        ultimate_effect_is_activated_by_source=(
            combat_effect_aggregation_result.ultimate_effect_is_activated_by_source
        ),
        combat_effect_has_recipient_by_source=combat_effect_has_recipient_by_source,
        combat_effect_recipient_global_slot_by_source=(
            combat_effect_recipient_global_slot_by_source
        ),
        raw_damage_output_by_source=(
            combat_effect_aggregation_result.raw_damage_output_by_source
        ),
        source_modified_damage_output_by_source=(
            combat_effect_aggregation_result.source_modified_damage_output_by_source
        ),
        recipient_damage_modifier_by_source=(
            combat_effect_aggregation_result.recipient_damage_modifier_by_source
        ),
        total_effective_damage_by_recipient=(
            combat_effect_aggregation_result.total_effective_damage_by_recipient
        ),
        raw_healing_output_by_source=(
            combat_effect_aggregation_result.raw_healing_output_by_source
        ),
        source_modified_healing_output_by_source=(
            combat_effect_aggregation_result.source_modified_healing_output_by_source
        ),
        recipient_healing_modifier_by_source=(
            combat_effect_aggregation_result.recipient_healing_modifier_by_source
        ),
        total_effective_healing_by_recipient=(
            combat_effect_aggregation_result.total_effective_healing_by_recipient
        ),
        health_after_combat_resolution_by_recipient=next_health_after_effective_damage_and_healing,
        slow_is_applied_by_source_and_channel=slow_is_applied_by_source_and_channel,
        stun_is_applied_by_source_and_channel=stun_is_applied_by_source_and_channel,
        rogue_poison_anti_heal_is_applied_by_source=(
            rogue_poison_anti_heal_is_applied_by_source
        ),
        mage_burst_damage_amplification_is_applied_by_source=(
            mage_burst_damage_amplification_is_applied_by_source
        ),
        priest_blessing_of_freedom_is_applied_by_source=(
            combat_effect_aggregation_result.priest_blessing_of_freedom_is_applied_by_source
        ),
    )


def _build_death_transition_facts(
    current_state: EnvState,
    config: EnvConfig,
    next_health_after_effective_damage_and_healing: Array,
    combat_effect_recipient_global_slot_by_source: Array,
    source_modified_damage_output_by_source: Array,
    recipient_damage_modifier_by_source: Array,
) -> DeathTransitionFacts:
    """Identify new deaths and preserve each source's gross damage contribution.

    current_state/config determine who was configured and alive at the start.
    The float32 health vector (10,) is the successor value after combat and
    regeneration but before respawn. Recipient IDs are int32 (10,), with -1
    for none. Source-modified damage and recipient factors are float32 (10,).
    Return bool new-death and contributor vectors, then float32 attributed
    damage in DeathTransitionFacts. Contributors retain positive effective
    damage to newly dead recipients; healing and health clipping do not select
    a killer or apportion realized health loss between sources.
    """

    was_active_and_alive = jnp.logical_and(
        current_state.alive_mask,
        config.agent_profile.active_mask,
    )

    is_newly_dead_by_recipient = jnp.logical_and(
        was_active_and_alive,
        next_health_after_effective_damage_and_healing == 0,
    )

    combat_effect_has_recipient_by_source_global_one_hot_routing_matrix = (
        jax.nn.one_hot(
            combat_effect_recipient_global_slot_by_source,
            num_classes=MAX_AGENT_SLOTS,
            dtype=jnp.bool_,
        )
    )

    attributed_death_damage_by_source = jnp.sum(
        combat_effect_has_recipient_by_source_global_one_hot_routing_matrix
        * is_newly_dead_by_recipient[None, :]
        * source_modified_damage_output_by_source[:, None]
        * recipient_damage_modifier_by_source[:, None],
        axis=-1,
    )

    contributed_to_new_death_by_source = attributed_death_damage_by_source > 0

    return DeathTransitionFacts(
        is_newly_dead_by_recipient=is_newly_dead_by_recipient,
        contributed_to_new_death_by_source=contributed_to_new_death_by_source,
        attributed_death_damage_by_source=attributed_death_damage_by_source,
    )


def build_canonical_no_transition_info_object(initial_state: EnvState) -> Info:
    """Build neutral diagnostic facts for initialization or fixed-storage padding.

    Parameters
    ----------
    initial_state : EnvState
        A ten-slot state supplying slow/stun and position array shapes.
        Its values, including its current step, are not copied into event facts.

    Returns
    -------
    Info
        Fixed-shape neutral facts: has_transition is False, start step is -1,
        recipient IDs are -1, actions are neutral, and event values are zero
        or False. The actual initial step remains available on the state.

    Notes
    -----
    This pure JAX builder does not validate or advance the state. It creates
    diagnostics, not a policy observation or evidence that an event occurred.
    A zero measurement here is padding for an absent transition, not an observed
    zero event. Consumers must check has_transition.
    """

    all_false_vector = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.bool_)
    all_zeroes_vector = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32)
    canonical_no_op_action_vector = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)

    reset_combat_transition_facts = CombatTransitionFacts(
        basic_effect_is_activated_by_source=all_false_vector,
        ultimate_effect_is_activated_by_source=all_false_vector,
        combat_effect_has_recipient_by_source=all_false_vector,
        combat_effect_recipient_global_slot_by_source=jnp.full(
            (MAX_AGENT_SLOTS,), -1, dtype=jnp.int32
        ),
        raw_damage_output_by_source=all_zeroes_vector,
        source_modified_damage_output_by_source=all_zeroes_vector,
        recipient_damage_modifier_by_source=all_zeroes_vector,
        total_effective_damage_by_recipient=all_zeroes_vector,
        raw_healing_output_by_source=all_zeroes_vector,
        source_modified_healing_output_by_source=all_zeroes_vector,
        recipient_healing_modifier_by_source=all_zeroes_vector,
        total_effective_healing_by_recipient=all_zeroes_vector,
        health_after_combat_resolution_by_recipient=all_zeroes_vector,
        slow_is_applied_by_source_and_channel=jnp.zeros_like(
            initial_state.slow_durations, dtype=jnp.bool_
        ),
        stun_is_applied_by_source_and_channel=jnp.zeros_like(
            initial_state.stun_durations, dtype=jnp.bool_
        ),
        rogue_poison_anti_heal_is_applied_by_source=all_false_vector,
        mage_burst_damage_amplification_is_applied_by_source=all_false_vector,
        priest_blessing_of_freedom_is_applied_by_source=all_false_vector,
    )

    canonical_no_op_action = Action(
        move=canonical_no_op_action_vector,
        select_target=canonical_no_op_action_vector,
        use_ultimate=canonical_no_op_action_vector,
    )

    reset_action_acceptance_facts = ActionAcceptanceFacts(
        submitted_joint_action=canonical_no_op_action,
        accepted_joint_action=canonical_no_op_action,
        submitted_action_tuple_is_out_of_domain_by_actor=all_false_vector,
        in_domain_move_action_is_rejected_by_actor=all_false_vector,
        in_domain_combat_action_pair_is_rejected_by_actor=all_false_vector,
    )

    reset_death_facts = DeathTransitionFacts(
        is_newly_dead_by_recipient=all_false_vector,
        contributed_to_new_death_by_source=all_false_vector,
        attributed_death_damage_by_source=all_zeroes_vector,
    )

    reset_spawn_shield_facts = SpawnShieldTransitionFacts(
        was_active_at_transition_start_by_agent=all_false_vector,
        expired_at_transition_end_by_agent=all_false_vector,
    )

    reset_respawn_facts = RespawnTransitionFacts(
        respawn_wave_occurred_this_transition_by_team=jnp.full(
            (NUM_TEAMS,), False, dtype=jnp.bool_
        ),
        was_respawned_this_transition_by_agent=all_false_vector,
    )

    reset_regeneration_facts = RegenerationTransitionFacts(
        combat_countdown_was_reset_by_agent=all_false_vector,
        actual_health_regenerated_this_step_by_agent=all_zeroes_vector,
    )

    reset_physical_facts = PhysicalTransitionFacts(
        charge_phase_displacement_by_agent=jnp.zeros_like(
            initial_state.agent_positions, dtype=jnp.float32
        ),
        ordinary_movement_phase_displacement_by_agent=jnp.zeros_like(
            initial_state.agent_positions, dtype=jnp.float32
        ),
    )

    reset_aura_facts = AuraTransitionFacts(
        is_covered_by_mage_damage_aura_by_emitter_and_beneficiary=jnp.zeros(
            (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS), dtype=jnp.bool_
        ),
        is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary=jnp.zeros(
            (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS), dtype=jnp.bool_
        ),
    )

    reset_status_lifecycle_facts = StatusLifecycleTransitionFacts(
        aged_to_zero_by_recipient_and_status_channel=jnp.zeros(
            (MAX_AGENT_SLOTS, 9), dtype=jnp.bool_
        ),
        refreshed_or_extended_by_recipient_and_status_channel=jnp.zeros(
            (MAX_AGENT_SLOTS, 9), dtype=jnp.bool_
        ),
        broken_by_damage_by_recipient_and_status_channel=jnp.zeros(
            (MAX_AGENT_SLOTS, 9), dtype=jnp.bool_
        ),
        cleared_by_new_death_by_recipient_and_status_channel=jnp.zeros(
            (MAX_AGENT_SLOTS, 9), dtype=jnp.bool_
        ),
    )

    reset_team_deathmatch_facts = TeamDeathmatchTransitionFacts(
        outcome=jnp.asarray(TASK_MODE_OUTCOME_ONGOING, dtype=jnp.int32)
    )

    reset_transition_facts = TransitionFacts(
        has_transition=jnp.asarray(False),
        transition_start_step_count=jnp.asarray(-1, dtype=jnp.int32),
        action_acceptance_facts=reset_action_acceptance_facts,
        combat_transition_facts=reset_combat_transition_facts,
        death_facts=reset_death_facts,
        spawn_shield_facts=reset_spawn_shield_facts,
        respawn_facts=reset_respawn_facts,
        regeneration_facts=reset_regeneration_facts,
        physical_facts=reset_physical_facts,
        aura_facts=reset_aura_facts,
        status_lifecycle_facts=reset_status_lifecycle_facts,
        team_deathmatch_facts=reset_team_deathmatch_facts,
    )

    return Info(transition_facts=reset_transition_facts)


def _build_was_respawned_this_transition_by_agent_array(
    current_state: EnvState, config: EnvConfig
) -> Array:
    """Select configured start-dead actors whose team wave is due now.

    current_state and config describe the transition start. Return bool (10,)
    in global slot order, using each team's zero countdown. Actors newly killed
    during this transition are absent, so they wait for a later wave.
    """
    is_active_but_dead = jnp.logical_and(
        config.agent_profile.active_mask, jnp.logical_not(current_state.alive_mask)
    )

    team_a_respawned_this_transition_array = jnp.logical_and(
        is_active_but_dead[TEAM_A_START:TEAM_A_END],
        current_state.team_respawn_wave_countdowns[TEAM_A_ID - 1] == 0,
    )

    team_b_respawned_this_transition_array = jnp.logical_and(
        is_active_but_dead[TEAM_B_START:TEAM_B_END],
        current_state.team_respawn_wave_countdowns[TEAM_B_ID - 1] == 0,
    )

    return jnp.concatenate(
        (
            team_a_respawned_this_transition_array,
            team_b_respawned_this_transition_array,
        ),
        dtype=jnp.bool_,
    )


def _handle_end_of_transition_respawn_wave_event(
    config: EnvConfig,
    was_respawned_this_transition_by_agent: Array,
    next_alive_mask: Array,
    next_health_after_effective_damage_and_healing: Array,
    next_spawn_shield_durations: Array,
    next_agent_positions: Array,
) -> tuple[Array, Array, Array, Array]:
    """Override eligible successor rows with simultaneous respawn values.

    config supplies class health, shield duration and fixed ordered pads.
    was_respawned_this_transition_by_agent is bool (10,), selected from the
    start-dead due-wave rule. The remaining next arrays are alive bool (10,),
    health float32 (10,), shield int32 (10,) and positions float32 (10, 2).
    Return those four arrays in the same order. Respawn sets full health and
    full shield duration at the slot's pad; occupancy does not select a new pad.
    The new shield is not decremented by the already completed ageing phase.
    """
    updated_next_alive_mask = jnp.where(
        was_respawned_this_transition_by_agent,
        jnp.ones_like(next_alive_mask),
        next_alive_mask,
    )

    # Health and shield are successor values, so the newly created shield does
    # not participate in the ordinary decrement that already occurred.
    updated_next_health = jnp.where(
        was_respawned_this_transition_by_agent,
        config.agent_profile.max_health,
        next_health_after_effective_damage_and_healing,
    )

    updated_next_spawn_shield_durations = jnp.where(
        was_respawned_this_transition_by_agent,
        config.spawn_shield_duration_steps,
        next_spawn_shield_durations,
    )

    # Global slot identity determines the immutable team-local pad; occupancy
    # deliberately does not participate in this selection.
    updated_team_a_next_agent_positions = jnp.where(
        was_respawned_this_transition_by_agent[TEAM_A_START:TEAM_A_END, None],
        config.team_spawn_pad_positions[TEAM_A_ID - 1, :, :],
        next_agent_positions[TEAM_A_START:TEAM_A_END, :],
    )

    updated_team_b_next_agent_positions = jnp.where(
        was_respawned_this_transition_by_agent[TEAM_B_START:TEAM_B_END, None],
        config.team_spawn_pad_positions[TEAM_B_ID - 1, :, :],
        next_agent_positions[TEAM_B_START:TEAM_B_END, :],
    )

    updated_next_agent_positions = jnp.concatenate(
        (updated_team_a_next_agent_positions, updated_team_b_next_agent_positions),
        axis=0,
        dtype=jnp.float32,
    )

    return (
        updated_next_alive_mask,
        updated_next_health,
        updated_next_spawn_shield_durations,
        updated_next_agent_positions,
    )


def _return_original_next_state_items(
    config: EnvConfig,
    was_respawned_this_transition_by_agent: Array,
    next_alive_mask: Array,
    next_health_after_effective_damage_and_healing: Array,
    next_spawn_shield_durations: Array,
    next_agent_positions: Array,
) -> tuple[Array, Array, Array, Array]:
    """Return unchanged alive, health, shield and position arrays when no wave is due.

    The returned order is next_alive_mask bool (10,), health float32 (10,),
    next_spawn_shield_durations int32 (10,), next_agent_positions float32
    (10, 2). config and was_respawned_this_transition_by_agent are unused operands
    retained to match the active lax.cond branch.
    """
    del config, was_respawned_this_transition_by_agent
    return (
        next_alive_mask,
        next_health_after_effective_damage_and_healing,
        next_spawn_shield_durations,
        next_agent_positions,
    )


def _compute_next_steps_until_out_of_combat(
    current_state: EnvState,
    config: EnvConfig,
    is_combat_participant_this_tick_by_agent: Array,
    next_alive_mask: Array,
) -> Array:
    """Reset, age or clear each successor combat countdown.

    current_state holds the old int32 (10,) countdown; config holds each
    class delay. Bool (10,) participation resets the delay, otherwise it ages
    once toward zero. Bool next_alive_mask is checked before respawn; dead or
    unused rows return zero. Return int32 (10,) without changing the inputs.
    """
    next_steps_until_ooc_active_masked = jnp.where(
        is_combat_participant_this_tick_by_agent,
        config.agent_profile.out_of_combat_delay_steps,
        jnp.maximum(current_state.steps_until_out_of_combat - 1, 0),
    )

    return jnp.where(
        jnp.logical_and(next_alive_mask, config.agent_profile.active_mask),
        next_steps_until_ooc_active_masked,
        0,
    ).astype(jnp.int32)


def _compute_health_after_out_of_combat_health_regeneration(
    current_state: EnvState,
    config: EnvConfig,
    next_health_after_effective_damage_and_healing: Array,
    is_combat_participant_this_tick: Array,
) -> tuple[Array, Array]:
    """Apply eligible recovery after combat and report actual health gained.

    current_state supplies start-alive flags, recovery countdowns and Poison;
    config supplies maximum health and recovery fractions. Post-combat health
    is float32 (10,) and current combat participation is bool (10,). Return
    float32 (10,) health, then actual recovery amounts (10,), clipped by maximum
    health. Eligibility requires start-alive, an already-zero countdown and no
    participation this transition. Current Poison reduces the recovery amount.
    A countdown that reaches zero only in the successor does not grant recovery
    early; respawn health restoration is not counted here.
    """
    raw_health_regen_deltas = (
        config.agent_profile.max_health
        * config.agent_profile.out_of_combat_health_regen_fraction_per_step
    )

    is_afflicted_with_rogue_poison = current_state.rogue_poison_anti_heal_durations > 0
    health_regen_deltas = jnp.where(
        is_afflicted_with_rogue_poison,
        raw_health_regen_deltas * ROGUE_POISON_ANTI_HEAL_MULTIPLIER,
        raw_health_regen_deltas,
    )

    regenerated_health_bars = jnp.minimum(
        next_health_after_effective_damage_and_healing + health_regen_deltas,
        config.agent_profile.max_health,
    )

    regenerates_health_this_tick = jnp.logical_and(
        jnp.logical_not(is_combat_participant_this_tick),
        jnp.logical_and(
            current_state.steps_until_out_of_combat == 0, current_state.alive_mask
        ),
    )

    health_after_out_of_combat_regeneration = jnp.where(
        regenerates_health_this_tick,
        regenerated_health_bars,
        next_health_after_effective_damage_and_healing,
    )

    actual_health_regenerated_this_tick_by_agent = (
        regenerates_health_this_tick
        * (
            health_after_out_of_combat_regeneration
            - next_health_after_effective_damage_and_healing
        )
    ).astype(jnp.float32)

    return (
        health_after_out_of_combat_regeneration,
        actual_health_regenerated_this_tick_by_agent,
    )


def _handle_team_deathmatch_outcome_and_rewards(
    next_state: EnvState, config: EnvConfig
) -> tuple[Reward, DoneFlags, Array]:
    """Resolve Team Deathmatch completion from successor scores and step count.

    next_state and config describe one game after death scoring. Return Reward
    with float32 (10,) values, DoneFlags with bool scalars, then int32 outcomes
    (4,) with the TDM entry set. A higher score at or above threshold wins;
    simultaneous threshold ties and horizon expiry without a threshold winner
    draw. Termination and truncation may both be True. Configured winner/loser
    slots receive +1/-1 even when dead; ongoing, draw and unused slots get zero.
    The caller must stop at done; this helper does not suppress repeated calls
    on an already-complete state or latch a one-time reward.
    """

    team_a_score = next_state.team_deathmatch_scores[TEAM_A_ID - 1]
    team_b_score = next_state.team_deathmatch_scores[TEAM_B_ID - 1]
    points_needed_to_win = config.team_deathmatch_score_threshold

    episode_has_truncated = next_state.step_count >= config.max_steps
    episode_has_terminated = jnp.logical_or(
        team_a_score >= points_needed_to_win, team_b_score >= points_needed_to_win
    )

    team_a_wins = jnp.logical_and(
        team_a_score >= points_needed_to_win, team_a_score > team_b_score
    )

    team_b_wins = jnp.logical_and(
        team_b_score >= points_needed_to_win, team_b_score > team_a_score
    )

    neither_team_has_won = jnp.logical_and(~team_a_wins, ~team_b_wins)

    episode_is_complete = jnp.logical_or(episode_has_truncated, episode_has_terminated)

    # Horizon expiry without a threshold winner is a draw. A simultaneous
    # threshold tie is also a draw, including when threshold and horizon coincide.
    teams_draw = jnp.logical_or(
        jnp.logical_and(neither_team_has_won, episode_has_truncated),
        jnp.logical_and(
            jnp.logical_and(
                team_b_score >= points_needed_to_win, team_b_score == team_a_score
            ),
            episode_is_complete,
        ),
    )

    # Accumulate one categorical result from mutually exclusive predicates.
    outcome = jnp.where(
        team_a_wins, TASK_MODE_OUTCOME_TEAM_A_WIN, TASK_MODE_OUTCOME_ONGOING
    )

    outcome = jnp.where(team_b_wins, TASK_MODE_OUTCOME_TEAM_B_WIN, outcome)
    outcome = jnp.where(teams_draw, TASK_MODE_OUTCOME_DRAW, outcome)
    # If no predicate is true, the task is still ongoing.

    task_outcomes = jnp.zeros((NUM_TASKS + 1,), dtype=jnp.int32)
    task_outcomes = task_outcomes.at[TASK_MODE_TDM].set(outcome)

    done_flags = DoneFlags(
        terminated=episode_has_terminated,
        truncated=episode_has_truncated,
    )

    outcome_branches = [
        _handle_task_rewards_ongoing,
        _handle_task_rewards_team_a_win,
        _handle_task_rewards_team_b_win,
        _handle_task_rewards_draw,
    ]

    rewards = cast(
        Reward,
        jax.lax.switch(outcome, outcome_branches, config),
    )

    return rewards, done_flags, task_outcomes


def _canonical_no_outcome_and_rewards(
    next_state: EnvState, config: EnvConfig
) -> tuple[Reward, DoneFlags, Array]:
    """Return neutral task values while preserving the successor horizon check.

    next_state supplies the step and config the horizon. Return zero float32
    Reward (10,), DoneFlags with terminated False and the horizon bool, then
    zero int32 outcomes (4,). No winner or draw is inferred for neutral mode.
    """
    reward = Reward(rewards=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32))
    done_flags = DoneFlags(
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(
            next_state.step_count >= config.max_steps, dtype=jnp.bool_
        ),
    )
    outcome = jnp.zeros((NUM_TASKS + 1,), dtype=jnp.int32)
    return reward, done_flags, outcome


def _handle_no_task_outcome_and_rewards(
    next_state: EnvState, config: EnvConfig
) -> tuple[Reward, DoneFlags, Array]:
    """Route neutral-mode successor state and config to the shared neutral output.

    Return zero Reward (10,), bool DoneFlags and zero outcomes (4,), retaining
    only the horizon truncation check. Inputs are not changed.
    """
    return _canonical_no_outcome_and_rewards(next_state, config)


def _handle_capture_the_flag_outcome_and_rewards(
    next_state: EnvState, config: EnvConfig
) -> tuple[Reward, DoneFlags, Array]:
    """Keep the reserved Capture the Flag branch structurally compatible.

    next_state and config are passed to the shared neutral-output helper.
    Return its zero Reward, horizon-only DoneFlags and zero outcomes (4,).
    Host validation rejects this mode; this branch is not implemented CTF play.
    """
    return _canonical_no_outcome_and_rewards(next_state, config)


def _handle_king_of_the_hill_outcome_and_rewards(
    next_state: EnvState, config: EnvConfig
) -> tuple[Reward, DoneFlags, Array]:
    """Keep the reserved King of the Hill branch structurally compatible.

    next_state and config are passed to the shared neutral-output helper.
    Return its zero Reward, horizon-only DoneFlags and zero outcomes (4,).
    Host validation rejects this mode; this branch is not implemented KOTH play.
    """
    return _canonical_no_outcome_and_rewards(next_state, config)


def _handle_task_rewards_ongoing(config: EnvConfig) -> Reward:
    """Return float32 zero rewards (10,) for an ongoing task transition.

    config is unused but keeps the same operand structure as winner branches
    in lax.switch. The Reward record covers all fixed slots, including padding.
    """
    del config
    return Reward(rewards=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32))


def _handle_task_rewards_draw(config: EnvConfig) -> Reward:
    """Return float32 zero rewards (10,) for a drawn task.

    config is unused but keeps the same operand structure as winner branches
    in lax.switch. A draw has no positive or negative terminal reward.
    """
    del config
    return Reward(rewards=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32))


def _handle_task_rewards_team_a_win(config: EnvConfig) -> Reward:
    """Return Team A's terminal reward for every configured roster slot.

    config supplies bool membership (10,). Return Reward with float32 (10,):
    +1 for Team A, -1 for Team B and zero for unused slots. Alive status is
    deliberately not an input, so configured dead agents share the team result.
    """
    zero_vector = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32)
    incomplete_reward_vector = zero_vector.at[TEAM_A_START:TEAM_A_END].set(
        REWARD_FOR_WINNING
    )
    complete_reward_vector = incomplete_reward_vector.at[TEAM_B_START:TEAM_B_END].set(
        REWARD_FOR_LOSING
    )
    return Reward(rewards=config.agent_profile.active_mask * complete_reward_vector)


def _handle_task_rewards_team_b_win(config: EnvConfig) -> Reward:
    """Return Team B's terminal reward for every configured roster slot.

    config supplies bool membership (10,). Return Reward with float32 (10,):
    +1 for Team B, -1 for Team A and zero for unused slots. Alive status is
    deliberately not an input, so configured dead agents share the team result.
    """
    zero_vector = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32)
    incomplete_reward_vector = zero_vector.at[TEAM_B_START:TEAM_B_END].set(
        REWARD_FOR_WINNING
    )
    complete_reward_vector = incomplete_reward_vector.at[TEAM_A_START:TEAM_A_END].set(
        REWARD_FOR_LOSING
    )
    return Reward(rewards=config.agent_profile.active_mask * complete_reward_vector)


def _compute_next_team_deathmatch_scores(
    current_scores: Array,
    new_deaths: Array,
    acting_agent_positions: Array,
    config: EnvConfig,
) -> Array:
    """Add each newly dead recipient's points to the opposing team.

    Parameters
    ----------
    current_scores : Array
        int32 (2,) scores in Team A, Team B order.
    new_deaths : Array
        bool (10,) in global slot order, already restricted to new configured
        deaths.
    acting_agent_positions : Array
        float32 (10, 2) body centres at the start of the transition, when combat
        resolves (before Charge and ordinary movement).
    config : EnvConfig
        Supplies the Red Zone depth, map width and spawn pads.

    Returns
    -------
    Array
        int32 (2,) updated scores. A victim inside its own team's Red Zone
        (red_zone_death_mask) gives TEAM_DEATHMATCH_POINTS_PER_RED_ZONE_DEATH;
        every other new death gives TEAM_DEATHMATCH_POINTS_PER_DEATH. Both teams
        can score in the same step; source attribution and individual killer
        selection do not affect totals.
    """

    red_zone_deaths = red_zone_death_mask(config, acting_agent_positions, new_deaths)
    points = new_deaths.astype(jnp.int32) * jnp.where(
        red_zone_deaths,
        jnp.int32(TEAM_DEATHMATCH_POINTS_PER_RED_ZONE_DEATH),
        jnp.int32(TEAM_DEATHMATCH_POINTS_PER_DEATH),
    )
    team_a_points = jnp.sum(points[TEAM_A_START:TEAM_A_END], dtype=jnp.int32)
    team_b_points = jnp.sum(points[TEAM_B_START:TEAM_B_END], dtype=jnp.int32)

    return jnp.asarray(
        (
            current_scores[TEAM_A_ID - 1] + team_b_points,
            current_scores[TEAM_B_ID - 1] + team_a_points,
        ),
        dtype=jnp.int32,
    )


def _not_team_deathmatch(
    current_scores: Array,
    new_deaths: Array,
    acting_agent_positions: Array,
    config: EnvConfig,
) -> Array:
    """Return zero int32 Team Deathmatch scores (2,) for another task mode.

    The four operands match the active scoring branch and are unused. Neutral
    play does not accumulate Team Deathmatch scores.
    """
    del current_scores, new_deaths, acting_agent_positions, config
    return jnp.zeros((NUM_TEAMS,), dtype=jnp.int32)


# Public ---


def red_zone_death_mask(
    config: EnvConfig,
    acting_agent_positions: Array,
    is_newly_dead_by_recipient: Array,
) -> Array:
    """Return which new deaths happened inside the victim's own team's Red Zone.

    Each team's Red Zone is a full-height strip on its own spawn side,
    config.team_deathmatch_red_zone_depth map units deep. The side comes from
    axis_mappings.spawn_bank_on_right on the team's five pads. Left strip:
    0 <= x <= depth. Right strip: float32(width - depth) <= x <= width, with the
    float32 map width. Both bounds are included. Each victim is checked only
    against its own team's strip, so overlapping strips are allowed. There is
    no y test.

    Parameters
    ----------
    config : EnvConfig
        Supplies map_width, team_spawn_pad_positions (2, 5, 2) and the depth.
    acting_agent_positions : Array
        float32 (10, 2) body centres at the start of the transition, when
        combat resolves. Later movement never changes a death's value.
    is_newly_dead_by_recipient : Array
        bool (10,) new configured deaths in global slot order.

    Returns
    -------
    Array
        bool (10,): True where a new death is inside its own team's strip.
        All False when the depth is 0.0 (rule off). A respawn is not a death.

    Notes
    -----
    Pure JAX with no Python branch on the depth, so it works under jit, vmap
    and lax.scan, and a changed depth does not recompile. Scoring and the full
    metrics both call this one owner.
    """
    width = jnp.asarray(config.map_width, dtype=jnp.float32)
    depth = jnp.asarray(config.team_deathmatch_red_zone_depth, dtype=jnp.float32)
    on_right = jnp.repeat(
        spawn_bank_on_right(config.team_spawn_pad_positions, width),
        MAX_AGENTS_PER_TEAM,
    )
    x = acting_agent_positions[:, 0]
    inside = jnp.where(on_right, x >= width - depth, x <= depth)
    return is_newly_dead_by_recipient & inside & (depth > 0.0)


def initialize_scenario_state(
    initial_state: EnvState, config: EnvConfig
) -> tuple[EnvState, Observation, ActionMask, Info]:
    """Validate and expose an authored state without advancing the simulator.

    Parameters
    ----------
    initial_state : EnvState
        One authored ten-slot state in the EnvState schema. Living bodies must
        not overlap, and accepted history must satisfy scenario shield rules.
    config : EnvConfig
        Matching scalar configuration with concrete host settings and JAX arrays.

    Returns
    -------
    tuple of EnvState, Observation, ActionMask and Info
        The same initial_state, its current observation and mask, then neutral
        no-transition diagnostics. Position, health, counters and step count
        are preserved, including an authored nonzero starting step.

    Raises
    ------
    TypeError
        Configuration or scenario storage/type checks fail.
    ValueError
        Configuration, state, geometry, start-limit or history checks fail.
    AssertionError
        A validated nonzero history target violates the internal mapping rule.

    Notes
    -----
    This is a host entry point: it validates configuration and authored state,
    may read device values and must stay outside jit/vmap/scan. It draws no
    randomness and does not treat the authored start as a simulator transition.
    Ordinary pad-based starts use reset instead.
    """
    validate_env_config(config)
    validate_scenario_initial_state(config, initial_state)
    obs, action_mask = _build_observation_and_action_mask(initial_state, config)
    info = build_canonical_no_transition_info_object(initial_state)
    return (initial_state, obs, action_mask, info)


def reset(
    config: EnvConfig, key: Array
) -> tuple[EnvState, Observation, ActionMask, Info]:
    """Build the initial state and decision inputs from fixed ordered spawn pads.

    Parameters
    ----------
    config : EnvConfig
        One episode configuration already checked on the host. It supplies
        ten-slot class facts, ordered pads, map geometry and episode rules.
        Under JAX, its array and scalar values are dynamic inputs.
    key : jax.Array
        Explicit PRNG key retained by the Core reset interface. This ordinary
        reset currently ignores it: randomized task builders choose the resolved
        setup before reset, and reset does not resample positions or classes.

    Returns
    -------
    tuple of EnvState, Observation, ActionMask and Info
        New state at step zero; its initial observation; its matching action
        mask; and neutral no-transition facts. Configured agents start alive
        at their ordered pads with full health. Cooldowns, statuses, shields,
        combat delays and history are zero. Each team clock starts at period
        minus one. Unused state rows are zero and inactive.

    Notes
    -----
    This pure numerical function handles one game and performs no host
    validation, I/O or mutation. It supports jit and an outer vmap for games.
    Carry the returned state and mask together into step. Configured sizes
    below five retain all ten array slots; active_mask marks participation.
    Authored nonstandard starts use initialize_scenario_state instead.
    """
    # Reset keeps all arrays at MAX_AGENT_SLOTS length. Smaller tasks use the
    # resolved profile's active mask to distinguish agents from padded slots.
    # Ordinary reset starts all active agents alive. Scenario loaders may later
    # create active-but-dead agents from curated states.
    # Randomized task builders consume keys while constructing resolved episode
    # configurations. Ordinary reset intentionally does not resample starts.
    # Curated starts use ``initialize_scenario_state`` so ordinary reset keeps a
    # single deterministic pad-based position authority.
    del key

    team_spawn_pad_positions = jnp.concatenate(
        (
            config.team_spawn_pad_positions[TEAM_A_ID - 1, :, :],
            config.team_spawn_pad_positions[TEAM_B_ID - 1, :, :],
        ),
        axis=0,
    )

    active_team_spawn_pad_positions = (
        team_spawn_pad_positions * config.agent_profile.active_mask[:, None]
    )

    initial_state = EnvState(
        team_deathmatch_scores=jnp.zeros((NUM_TEAMS,), dtype=jnp.int32),
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=active_team_spawn_pad_positions.astype(jnp.float32),
        alive_mask=config.agent_profile.active_mask,
        current_health=config.agent_profile.max_health,
        ultimate_cooldowns=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        slow_durations=jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32),
        stun_durations=jnp.zeros((MAX_AGENT_SLOTS, NUM_STUN_CHANNELS), dtype=jnp.int32),
        rogue_poison_anti_heal_durations=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        mage_burst_damage_amplification_durations=jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        priest_blessing_of_freedom_slow_floor_durations=jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        team_respawn_wave_countdowns=config.team_respawn_wave_period_step_count - 1,
        spawn_shield_durations=jnp.zeros((MAX_AGENT_SLOTS), dtype=jnp.int32),
        steps_until_out_of_combat=jnp.zeros((MAX_AGENT_SLOTS), dtype=jnp.int32),
        previous_timestep_move_actions=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        previous_timestep_select_target_actions=jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        previous_timestep_use_ultimate_actions=jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        has_previous_timestep_joint_action=jnp.asarray(0, dtype=bool),
    )

    initial_observation, initial_action_mask = _build_observation_and_action_mask(
        initial_state, config
    )

    info = build_canonical_no_transition_info_object(initial_state)

    return (initial_state, initial_observation, initial_action_mask, info)


def step(
    config: EnvConfig,
    current_state: EnvState,
    current_action_mask: ActionMask,
    joint_action: Action,
    key: Array,
) -> tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]:
    """Advance one paired state and action-mask snapshot by one simulator step.

    All actions are chosen from the current observation/mask before this call.
    Combat reads current positions and current status strengths. Charge places
    its arrivals first; the already chosen voluntary moves then resolve from
    those positions. Fresh statuses enter the successor and first govern the
    next decision. Due respawns happen at the end for actors already dead at
    transition start. New deaths wait for a later wave.

    Parameters
    ----------
    config : EnvConfig
        Matching validated one-game configuration. Keep episode rules fixed
        within the game; their JAX values remain dynamic inputs.
    current_state : EnvState
        Current ten-slot state returned by reset or the preceding step.
    current_action_mask : ActionMask
        The mask produced with current_state. It is the sole authority for
        accepting this submission; Core does not rebuild or validate its origin.
    joint_action : Action
        Int32 movement, target and Ultimate arrays (10,). Target IDs are
        actor-relative. In-domain masked movement becomes Stay; a masked combat
        pair becomes (Target None, no Ultimate). Any out-of-domain head clears
        that actor's entire tuple. Rejections are recorded in Info.
    key : jax.Array
        Explicit PRNG key retained by the interface. Current transitions are
        deterministic from the other arguments and ignore this key.

    Returns
    -------
    tuple of EnvState, Observation, Reward, DoneFlags, ActionMask and Info
        Successor state; its observation; float32 rewards (10,); scalar bool
        termination/truncation flags; its matching mask; and privileged facts
        about the transition just completed. State step_count increases once.
        Observation history contains the accepted action, not the rejected
        submission. Carry successor state and mask into the next step.

    Notes
    -----
    This pure JAX function handles one game. It performs no host validation,
    policy call, I/O, random draw or input mutation. Use jit/scan and an outer
    vmap for games. Stop or reset when DoneFlags.done becomes True: Core does
    not auto-reset, absorb terminal states or suppress rewards if called again.
    Team Deathmatch can terminate and truncate together. Threshold ties draw;
    horizon expiry without a threshold winner also draws. Each new configured
    death gives the other team 1 point, or 2 when the victim's centre at the
    start of the tick (when combat resolves, before Charge and ordinary
    movement) is inside its own team's Red Zone (red_zone_death_mask). Both
    teams' points apply before the outcome is chosen; a lethally hit agent that
    then moves across the boundary keeps its value, and a respawn is not a
    death. The next observation shows the new scores and the same depth.
    Configured agents
    share their team's +1/-1 result even when dead; unused slots receive zero.
    Ongoing transitions and draws give every slot zero reward.
    Info is global simulator truth and does not grant policy information rights.
    """
    del key

    accepted_joint_action, action_acceptance_facts = (
        _build_accepted_joint_action_from_submitted_joint_action(
            current_action_mask=current_action_mask, submitted_joint_action=joint_action
        )
    )

    (
        accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
        combat_effect_has_recipient_by_source,
        combat_effect_recipient_global_slot_by_source,
    ) = _build_global_pairwise_actor_and_recipient_target_one_hot_matrix(
        accepted_joint_action.select_target
    )

    current_global_pairwise_distances = (
        _compute_global_pairwise_distances_from_agent_positions(
            current_state.agent_positions
        )
    )

    combat_aura_aggregation_result = _derive_aura_damage_multipliers(
        config,
        current_global_pairwise_distances,
        current_state.alive_mask,
        current_state.spawn_shield_durations == 0,
    )

    combat_effect_aggregation_result = (
        _aggregate_health_effects_and_basic_passives_by_global_slot(
            current_state,
            config,
            accepted_joint_action,
            accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
            combat_aura_aggregation_result.mage_damage_amplification_aura_multipliers,
            combat_aura_aggregation_result.warrior_damage_mitigation_aura_multipliers,
        )
    )

    next_health_after_effective_damage_and_healing = (
        _compute_health_after_simultaneous_damage_and_healing(
            combat_effect_aggregation_result.total_effective_damage_by_recipient,
            combat_effect_aggregation_result.total_effective_healing_by_recipient,
            current_state,
            config,
        )
    )

    (
        next_health_after_out_of_combat_regeneration,
        actual_health_regenerated_this_tick_by_agent,
    ) = _compute_health_after_out_of_combat_health_regeneration(
        current_state,
        config,
        next_health_after_effective_damage_and_healing,
        combat_effect_aggregation_result.is_combat_participant_this_tick_by_source,
    )

    # Accepted use starts a full cooldown; every unreplaced cooldown ticks once.
    next_ultimate_cooldowns = jnp.where(
        accepted_joint_action.use_ultimate == 1,
        get_ultimate_cooldown_by_class_ids(config.agent_profile.class_ids),
        jnp.maximum(0, current_state.ultimate_cooldowns - 1),
    )

    # Current public status truth governs the current movement decision.
    intended_movement_deltas = _build_intended_movement_deltas(
        current_state,
        config,
        accepted_joint_action,
    )

    # The current counter governs both traversal and final-endpoint collision.
    # Geometry independently intersects these lifecycle masks with active/alive.
    always_participates_in_agent_agent_collision = (
        current_state.spawn_shield_durations == 0
    )
    participates_in_agent_agent_collision_at_final_position = jnp.logical_or(
        always_participates_in_agent_agent_collision,
        current_state.spawn_shield_durations == 1,
    )

    post_charge_current_agent_positions = _resolve_post_charge_agent_positions(
        current_state,
        config,
        accepted_joint_action,
        accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
        always_participates_in_agent_agent_collision,
    )

    # Resolve every precommitted voluntary move from the realized Charge phase.
    next_agent_positions = project_movement_with_geometry(
        post_charge_current_agent_positions,
        config.agent_profile.agent_radii,
        intended_movement_deltas,
        config.agent_profile.active_mask,
        current_state.alive_mask,
        config.map_width,
        config.map_height,
        config.obstacles,
        always_participates_in_agent_agent_collision,
        participates_in_agent_agent_collision_at_final_position,
    )

    physical_facts = PhysicalTransitionFacts(
        post_charge_current_agent_positions - current_state.agent_positions,
        next_agent_positions - post_charge_current_agent_positions,
    )

    death_facts = _build_death_transition_facts(
        current_state,
        config,
        next_health_after_out_of_combat_regeneration,
        combat_effect_recipient_global_slot_by_source,
        combat_effect_aggregation_result.source_modified_damage_output_by_source,
        combat_effect_aggregation_result.recipient_damage_modifier_by_source,
    )

    next_alive_mask = jnp.logical_and(
        jnp.logical_not(death_facts.is_newly_dead_by_recipient),
        current_state.alive_mask,
    )

    next_steps_until_ooc = _compute_next_steps_until_out_of_combat(
        current_state,
        config,
        combat_effect_aggregation_result.is_combat_participant_this_tick_by_source,
        next_alive_mask,
    )

    regeneration_facts = RegenerationTransitionFacts(
        combat_countdown_was_reset_by_agent=combat_effect_aggregation_result.is_combat_participant_this_tick_by_source,
        actual_health_regenerated_this_step_by_agent=actual_health_regenerated_this_tick_by_agent,
    )

    combat_status_aggregation_result = _resolve_status_duration_lifecycle(
        current_state,
        config,
        accepted_joint_action.use_ultimate,
        accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
        combat_effect_aggregation_result.hunter_basic_slow_applied_this_tick_by_global_recipient_slot,
        combat_effect_aggregation_result.priest_freedom_applied_this_tick_by_global_recipient_slot,
        combat_effect_aggregation_result.accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot,
        combat_effect_aggregation_result.hunter_basic_slow_applied_this_tick_by_global_actor_slot,
        next_alive_mask,
        death_facts.is_newly_dead_by_recipient,
    )

    combat_transition_facts = _build_combat_transition_facts(
        combat_effect_aggregation_result,
        combat_effect_has_recipient_by_source,
        combat_effect_recipient_global_slot_by_source,
        combat_status_aggregation_result.slow_is_applied_by_source_and_channel,
        combat_status_aggregation_result.stun_is_applied_by_source_and_channel,
        combat_status_aggregation_result.rogue_poison_anti_heal_is_applied_by_source,
        combat_status_aggregation_result.mage_burst_damage_amplification_is_applied_by_source,
        next_health_after_effective_damage_and_healing,
    )

    respawn_facts = RespawnTransitionFacts(
        respawn_wave_occurred_this_transition_by_team=current_state.team_respawn_wave_countdowns
        == 0,
        was_respawned_this_transition_by_agent=_build_was_respawned_this_transition_by_agent_array(
            current_state, config
        ),
    )

    # Every team clock advances independently, including an empty due wave.
    next_team_respawn_wave_countdowns = jnp.where(
        current_state.team_respawn_wave_countdowns == 0,
        config.team_respawn_wave_period_step_count - 1,
        current_state.team_respawn_wave_countdowns - 1,
    )

    (
        next_alive_mask,
        next_health_bars,
        next_spawn_shield_durations,
        next_agent_positions,
    ) = cast(
        tuple[Array, Array, Array, Array],
        jax.lax.cond(
            jnp.any(
                respawn_facts.respawn_wave_occurred_this_transition_by_team, axis=-1
            ),
            _handle_end_of_transition_respawn_wave_event,
            _return_original_next_state_items,
            config,
            respawn_facts.was_respawned_this_transition_by_agent,
            next_alive_mask,
            next_health_after_out_of_combat_regeneration,
            combat_status_aggregation_result.next_spawn_shield_durations,
            next_agent_positions,
        ),
    )

    spawn_shield_facts = SpawnShieldTransitionFacts(
        was_active_at_transition_start_by_agent=current_state.spawn_shield_durations
        > 0,
        expired_at_transition_end_by_agent=jnp.logical_and(
            current_state.spawn_shield_durations == 1, next_alive_mask
        ),
    )
    next_team_deathmatch_scores = cast(
        Array,
        jax.lax.cond(
            config.task_mode == TASK_MODE_TDM,
            _compute_next_team_deathmatch_scores,
            _not_team_deathmatch,
            current_state.team_deathmatch_scores,
            death_facts.is_newly_dead_by_recipient,
            current_state.agent_positions,
            config,
        ),
    )

    next_state = EnvState(
        team_deathmatch_scores=next_team_deathmatch_scores,
        step_count=current_state.step_count + 1,
        agent_positions=next_agent_positions,
        alive_mask=next_alive_mask,
        current_health=next_health_bars,
        # NOTE: Ultimate CD carries over into death to prevent abuse.
        ultimate_cooldowns=next_ultimate_cooldowns,
        slow_durations=combat_status_aggregation_result.next_slow_durations,
        stun_durations=combat_status_aggregation_result.next_stun_durations,
        rogue_poison_anti_heal_durations=combat_status_aggregation_result.next_rogue_anti_heal_durations,
        mage_burst_damage_amplification_durations=combat_status_aggregation_result.next_mage_burst_durations,
        priest_blessing_of_freedom_slow_floor_durations=combat_status_aggregation_result.next_priest_freedom_slow_floor_durations,
        team_respawn_wave_countdowns=next_team_respawn_wave_countdowns,
        spawn_shield_durations=next_spawn_shield_durations,
        steps_until_out_of_combat=next_steps_until_ooc,
        previous_timestep_move_actions=accepted_joint_action.move,
        previous_timestep_select_target_actions=accepted_joint_action.select_target,
        previous_timestep_use_ultimate_actions=accepted_joint_action.use_ultimate,
        has_previous_timestep_joint_action=jnp.asarray(1, dtype=bool),
    )

    next_observation, next_action_mask = _build_observation_and_action_mask(
        next_state, config
    )

    task_branches = [
        _handle_no_task_outcome_and_rewards,
        _handle_team_deathmatch_outcome_and_rewards,
        _handle_king_of_the_hill_outcome_and_rewards,
        _handle_capture_the_flag_outcome_and_rewards,
    ]

    rewards, done_flags, task_outcomes = cast(
        tuple[Reward, DoneFlags, Array],
        jax.lax.switch(
            config.task_mode,
            task_branches,
            next_state,
            config,
        ),
    )

    transition_facts = TransitionFacts(
        has_transition=jnp.asarray(True),
        transition_start_step_count=current_state.step_count,
        action_acceptance_facts=action_acceptance_facts,
        combat_transition_facts=combat_transition_facts,
        death_facts=death_facts,
        spawn_shield_facts=spawn_shield_facts,
        respawn_facts=respawn_facts,
        regeneration_facts=regeneration_facts,
        physical_facts=physical_facts,
        aura_facts=combat_aura_aggregation_result.aura_facts,
        status_lifecycle_facts=combat_status_aggregation_result.status_lifecycle_facts,
        team_deathmatch_facts=TeamDeathmatchTransitionFacts(
            outcome=task_outcomes[TASK_MODE_TDM]
        ),
    )

    info = Info(
        transition_facts=transition_facts,
    )

    return (next_state, next_observation, rewards, done_flags, next_action_mask, info)
