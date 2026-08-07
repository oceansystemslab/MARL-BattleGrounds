"""Functional reset and step entry points for the core JAX simulator."""

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.combat import (
    BASIC_DAMAGE_BY_CLASS,
    BASIC_HEALING_BY_CLASS,
    HUNTER_BASIC_SLOW_DURATION_TICKS,
    HUNTER_BASIC_SLOW_MULTIPLIER,
    HUNTER_TRAP_STUN_DURATION_TICKS,
    MAGE_BURST_DAMAGE_DURATION_TICKS,
    MAGE_BURST_DAMAGE_MULTIPLIER,
    MAGE_DAMAGE_AURA_MULTIPLIER,
    MAGE_DAMAGE_AURA_MULTIPLIER_CEILING,
    MAGE_DAMAGE_AURA_RADIUS,
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
    DEFAULT_AGENT_PROJECTION_PASSES,
    has_clear_line_of_sight,
    project_movement_with_geometry,
)
from marl_battlegrounds.core.types import (
    CONTEXT_FEATURE_ALLY_TEAM_SIZE,
    CONTEXT_FEATURE_CURRENT_TIMESTEP,
    CONTEXT_FEATURE_ENEMY_TEAM_SIZE,
    CONTEXT_FEATURE_MAP_HEIGHT,
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
    NUM_TEAMS,
    NUM_ULTIMATE_ACTIONS,
    OBJECTIVE_FEATURES,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    SLOW_CHANNEL_HUNTER_BASIC,
    SLOW_CHANNEL_ROGUE_POISON,
    SLOW_CHANNEL_WARRIOR_CHARGE,
    STUN_CHANNEL_HUNTER_TRAP,
    STUN_CHANNEL_ROGUE_POISON,
    STUN_CHANNEL_WARRIOR_CHARGE,
    TEAM_A_ID,
    TEAM_B_ID,
    UNIT_FEATURES,
    WARRIOR_CLASS_ID,
    Action,
    ActionAcceptanceFacts,
    ActionMask,
    CombatTransitionFacts,
    DeathTransitionFacts,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    PreviousTimestepActionObservation,
    RegenerationTransitionFacts,
    RespawnTransitionFacts,
    Reward,
    SpawnLifecycleObservation,
    SpawnShieldTransitionFacts,
    TransitionFacts,
)

# Private Helpers ---

TEAM_A_START = 0
TEAM_A_END = MAX_AGENTS_PER_TEAM
TEAM_B_START = MAX_AGENTS_PER_TEAM
TEAM_B_END = MAX_AGENT_SLOTS

# Direction rows are unit-length and ordered to match the MOVE_* constants.
_INV_SQRT_2 = 1 / jnp.sqrt(2.0)

_GLOBAL_AGENT_SLOT_INDICES = jnp.arange(MAX_AGENT_SLOTS, dtype=jnp.int32)

_JOINT_ACTION_MOVE_TO_DISPLACEMENT_LOOKUP_TABLE = jnp.array(
    [
        jnp.array((0, 0), dtype=jnp.float32),  # MOVE_STAY = 0
        jnp.array((0, 1), dtype=jnp.float32),  # MOVE_NORTH = 1
        jnp.array((0, -1), dtype=jnp.float32),  # MOVE_SOUTH = 2
        jnp.array((1, 0), dtype=jnp.float32),  # MOVE_EAST = 3
        jnp.array((-1, 0), dtype=jnp.float32),  # MOVE_WEST = 4
        jnp.array((_INV_SQRT_2, _INV_SQRT_2), dtype=jnp.float32),  # MOVE_NORTHEAST = 5
        jnp.array((-_INV_SQRT_2, _INV_SQRT_2), dtype=jnp.float32),  # MOVE_NORTHWEST = 6
        jnp.array((_INV_SQRT_2, -_INV_SQRT_2), dtype=jnp.float32),  # MOVE_SOUTHEAST = 7
        jnp.array(
            (-_INV_SQRT_2, -_INV_SQRT_2), dtype=jnp.float32
        ),  # MOVE_SOUTHWEST = 8
    ]
)


# Each fixed team block shares one actor-relative candidate order. Mapping
# target-none to -1 makes that category an all-zero row under ``jax.nn.one_hot``.
_ACTOR_RELATIVE_SELECT_TARGET_ACTION_TO_GLOBAL_AGENT_SLOT_LOOKUP_TABLE = jnp.asarray(
    [
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # agent 0
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # agent 1
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # agent 2
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # agent 3
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # agent 4
        [-1, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],  # agent 5
        [-1, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],  # agent 6
        [-1, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],  # agent 7
        [-1, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],  # agent 8
        [-1, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],  # agent 9
    ],
    dtype=jnp.int32,
)


class _CombatEffectAggregationResult(NamedTuple):
    """Internal effect arrays shared by successor state and transition facts."""

    hunter_basic_slow_applied_this_tick_by_global_recipient_slot: Array
    priest_freedom_applied_this_tick_by_global_recipient_slot: Array
    accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot: Array
    hunter_basic_slow_applied_this_tick_by_global_actor_slot: Array
    basic_effect_is_activated_by_source: Array
    ultimate_effect_is_activated_by_source: Array
    raw_damage_output_by_source: Array
    source_modified_damage_output_by_source: Array
    recipient_damage_modifier_by_source: Array
    total_effective_damage_by_recipient: Array
    raw_healing_output_by_source: Array
    source_modified_healing_output_by_source: Array
    recipient_healing_modifier_by_source: Array
    total_effective_healing_by_recipient: Array
    priest_blessing_of_freedom_is_applied_by_source: Array
    is_combat_participant_this_tick_by_source: Array


def _compute_global_pairwise_distances_from_agent_positions(
    agent_positions: Array,
) -> Array:
    """Return the dense Euclidean distance matrix for fixed-slot positions."""
    return cast(
        Array,
        jnp.linalg.norm(
            (agent_positions[None, :, :] - agent_positions[:, None, :]), axis=-1
        ),
    )


def _build_global_visibility_mask_and_distances(
    state: EnvState, config: EnvConfig
) -> tuple[Array, Array]:
    """Build LOS-gated visibility and distances between all global slots.

    Entry ``[i, j]`` is true when observer slot ``i`` can currently observe
    candidate slot ``j``. The mask is directed because each observer owns its
    own effective observation radius. Pairwise distances are returned alongside
    the mask so targetability can reuse the same observer-candidate geometry.

    Visibility requires:
    1. observer is active and alive;
    2. candidate is active and alive;
    3. candidate center is within observer i's observation radius;
    4. static line of sight from observer to candidate is clear.
    5. an opposing candidate does not have an active spawn shield.

    Spawn shield concealment is directional: self and allied visibility retain
    the ordinary range and line-of-sight rules.
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
        """Build one observer's LOS row against all candidate centers."""
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
    """Project a global observer-candidate matrix into relation slots.

    ``global_mask[i, j]`` stores a directed boolean relation from observer
    global slot ``i`` to candidate global slot ``j``. The returned masks keep
    candidate slots relation-local so they align with unit-feature and
    target-selection heads.

    Slot layout is fixed:
    - Team A occupies global slots ``0..MAX_AGENTS_PER_TEAM - 1``;
    - Team B occupies global slots ``MAX_AGENTS_PER_TEAM..MAX_AGENT_SLOTS - 1``.
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
    """Return dense same-team and opposing-team masks for real team IDs."""
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
    """Build authoritative legality for every target/ultimate action pair.

    The returned axes are actor slot, actor-relative target action, and
    ultimate-use choice. Lane zero preserves class-aware basic legality; lane
    one applies class-specific ultimate relation, range, cooldown, and control
    gates. Both lanes reuse the supplied visibility and distance truth, so this
    helper performs no geometry or LOS work itself.
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
    """Derive per-head masks from authoritative target/ultimate pair legality."""
    select_target_mask = jnp.any(select_target_use_ultimate_joint_mask, axis=-1)
    use_ultimate_mask = jnp.any(select_target_use_ultimate_joint_mask, axis=1)

    return select_target_mask, use_ultimate_mask


def _build_move_mask(state: EnvState, config: EnvConfig) -> Array:
    """Build protocol-admissible movement choices for every fixed slot.

    Active, living, unstunned actors may submit every movement category.
    Currently stunned, dead, or inactive slots expose only ``MOVE_STAY``, the
    effect-inert canonical submission required for direct categorical sampling.
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
    """Build raw actor-relative episode context for configured policy slots.

    The simulator exposes semantic facts without learner-specific scaling.
    Mode-owned columns remain zero until their battleground contracts exist,
    and configured-but-dead actors retain context while padded rows stay zero.
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
    """Project global actor rows into fixed ally and enemy relation blocks.

    ``previous_joint_action_head_one_hot`` has one row per global actor slot.
    The returned tensors add the observer axis while preserving the same stable
    relation-row convention used by ally and enemy unit features. Visibility is
    applied later by the complete previous-action observation builder.
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
    """Build policy-facing history using current actor visibility.

    State stores compact accepted categories in each actor's own action
    convention. Movement and ultimate-use categories need only relation-row
    projection. Target categories additionally swap ally/enemy category blocks
    when actor and observer belong to opposing teams so every observer decodes
    the same stable target identity.

    History validity and visibility of the observed actor gate complete rows.
    Target visibility does not gate the accepted target identity.
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
    """Build actor-relative public spawn pads, shield rules, and lifecycle truth.

    Team row zero is the observer's team and row one is its opponent. Configured
    living and dead observers receive the same public roster, spawn-shield
    counters, and configured rules; padded observer rows remain zero.
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
    )


def _build_observation_and_action_mask(
    state: EnvState, config: EnvConfig
) -> tuple[Observation, ActionMask]:
    """Build the current observation contract from one slot-aligned state.

    Self rows are canonical fixed-slot agent rows. Ally and enemy unit rows use
    the same agent-feature schema in relation-local candidate order, with
    nonvisible candidate rows zeroed by the visibility masks.
    """
    global_visibility_mask, global_pairwise_distances = (
        _build_global_visibility_mask_and_distances(state, config)
    )

    (
        mage_damage_amplification_aura_multipliers,
        warrior_damage_mitigation_aura_multipliers,
    ) = _derive_aura_damage_multipliers(
        config,
        global_pairwise_distances,
        state.alive_mask,
        state.spawn_shield_durations == 0,
    )

    self_features = _build_self_features(
        state,
        config,
        mage_damage_amplification_aura_multipliers,
        warrior_damage_mitigation_aura_multipliers,
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
    )

    return current_observation, current_action_mask


def _build_intended_movement_deltas(
    current_state: EnvState,
    config: EnvConfig,
    accepted_joint_action: Action,
) -> Array:
    """Build voluntary movement intent from current public control truth.

    Status durations in ``current_state`` must be the values visible when the
    policy selects the current action. Statuses accepted during this transition
    are packaged for the next action and must not retroactively change this
    movement intent. For a host-valid unshielded actor, current stun yields zero
    intent; spawn shield and stun cannot coexist at the host boundary. Death or
    inactivity also yields zero intent as defense in depth. Geometry remains
    free to displace a zero-intent body during collision resolution.
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
    """Return the fixed-slot mask of active Mage agents."""
    return jnp.logical_and(
        config.agent_profile.class_ids == MAGE_CLASS_ID,
        config.agent_profile.active_mask,
    )


def _active_warrior_class_mask(config: EnvConfig) -> Array:
    """Return the fixed-slot mask of active Warrior agents."""
    return jnp.logical_and(
        config.agent_profile.class_ids == WARRIOR_CLASS_ID,
        config.agent_profile.active_mask,
    )


def _active_hunter_class_mask(config: EnvConfig) -> Array:
    """Return the fixed-slot mask of active Hunter agents."""
    return jnp.logical_and(
        config.agent_profile.class_ids == HUNTER_CLASS_ID,
        config.agent_profile.active_mask,
    )


def _active_rogue_class_mask(config: EnvConfig) -> Array:
    """Return the fixed-slot mask of active Rogue agents."""
    return jnp.logical_and(
        config.agent_profile.class_ids == ROGUE_CLASS_ID,
        config.agent_profile.active_mask,
    )


def _active_priest_class_mask(config: EnvConfig) -> Array:
    """Return the fixed-slot mask of active Priest agents."""
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
    """Build slot-aligned self rows from the shared agent-feature schema."""
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
            config.agent_profile.team_ids[:, None],
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
                MAGE_DAMAGE_AURA_RADIUS,
                MAGE_DAMAGE_AURA_MULTIPLIER,
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
    """Project global self rows into relation-local ally candidate rows."""
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
    """Project global self rows into relation-local enemy candidate rows."""
    enemy_features = jnp.zeros(
        (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES), dtype=jnp.float32
    )

    enemy_features = enemy_features.at[TEAM_A_START:TEAM_A_END, :, :].set(
        self_features[TEAM_B_START:TEAM_B_END, :]
    )
    enemy_features = enemy_features.at[TEAM_B_START:TEAM_B_END, :, :].set(
        self_features[TEAM_A_START:TEAM_A_END, :]
    )

    return enemy_features


def _mask_unit_features(unit_features: Array, visibility_mask: Array) -> Array:
    """Zero relation-local candidate rows that are hidden from each observer."""
    return jnp.where(
        visibility_mask[:, :, None],
        unit_features,
        jnp.zeros_like(unit_features),
    ).astype(jnp.float32)


def _build_accepted_joint_action_from_submitted_joint_action(
    current_action_mask: ActionMask, submitted_joint_action: Action
) -> tuple[Action, ActionAcceptanceFacts]:
    """Canonicalize one submitted action from authoritative pre-state masks.

    Each submitted category is replaced with a safe neutral gather index before
    any mask access. If any head is outside its categorical domain, that actor's
    complete tuple becomes the canonical no-op without affecting other actors.
    Out-of-domain IDs indicate an upstream policy or sampler defect; categorical
    domain validation is the first debugging step.

    For wholly in-domain tuples, movement acceptance is independent of combat
    acceptance. Target and ultimate-use heads are accepted together from the
    authoritative joint mask so an invalid ultimate attempt cannot fall back to
    a valid basic action.
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
    """Map actor-relative accepted targets to dense global recipient rows."""
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
    """Aggregate accepted health effects and basic-passive applications.

    The accepted target categories are actor-relative. After fixed-slot
    translation, dense one-hot rows route every actor's catalog payload to its
    recipient. Basic and ultimate contributions share this routing boundary.
    Reducing the complete matrices before health mutation preserves simultaneous,
    actor-order-independent resolution. The result retains the exact recipient
    totals consumed by health resolution alongside the source-aligned values
    needed for public facts. Its boolean vectors preserve passive and Trap-break
    causes without inferring them from successor state.
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
    """Net simultaneous health effects and clamp each fixed-slot health value."""
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
) -> tuple[Array, Array]:
    """Derive bounded Mage outgoing and Warrior incoming aura modifiers.

    Rows represent aura emitters and columns represent beneficiary slots.
    Only interaction-eligible active, living allies with real team IDs
    participate as emitters or beneficiaries. Auras include eligible emitters,
    use inclusive radius boundaries, and stack multiplicatively before the
    completed vectors are bounded.
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

    mage_actor_benefits_recipient_with_aura = jnp.where(
        mage_masked_global_pairwise_distances <= MAGE_DAMAGE_AURA_RADIUS,
        MAGE_DAMAGE_AURA_MULTIPLIER,
        1.0,
    )
    warrior_actor_benefits_recipient_with_aura = jnp.where(
        warrior_masked_global_pairwise_distances
        <= WARRIOR_DAMAGE_MITIGATION_AURA_RADIUS,
        WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER,
        1.0,
    )

    mage_aura_outgoing_damage_multiplier_by_actor_slot = jnp.prod(
        mage_actor_benefits_recipient_with_aura, axis=0
    )
    warrior_aura_incoming_damage_multiplier_by_global_recipient_slot = jnp.prod(
        warrior_actor_benefits_recipient_with_aura, axis=0
    )

    return (
        jnp.clip(
            mage_aura_outgoing_damage_multiplier_by_actor_slot,
            min=1.0,
            max=MAGE_DAMAGE_AURA_MULTIPLIER_CEILING,
        ),
        jnp.clip(
            warrior_aura_incoming_damage_multiplier_by_global_recipient_slot,
            min=WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER_FLOOR,
            max=1.0,
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
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Resolve successor transient durations and status-application facts.

    Current durations age once toward zero. Accepted positive raw damage then
    clears only the aged successor of a pre-existing Hunter Trap, after which
    fresh source-local applications merge at full configured duration. A fresh
    application never shortens a longer aged remainder and first governs the
    next policy action. The spawn-shield counter also ages once after movement.
    Dead successor slots retain status-application facts but carry no transient
    duration into the next state.
    """

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

    # Transient status memory cannot survive into a dead successor slot.
    next_slow_durations = next_slow_durations * next_alive_mask[:, None]

    next_stun_durations = next_stun_durations * next_alive_mask[:, None]

    next_mage_burst_durations = next_mage_burst_durations * next_alive_mask

    next_rogue_anti_heal_durations = next_rogue_anti_heal_durations * next_alive_mask

    next_priest_freedom_slow_floor_durations = (
        next_priest_freedom_slow_floor_durations * next_alive_mask
    )

    # Spawn shielding ages after movement and cannot survive death or padding.
    next_spawn_shield_durations = (
        jnp.maximum(current_state.spawn_shield_durations - 1, 0).astype(jnp.int32)
        * next_alive_mask
        * config.agent_profile.active_mask
    )

    return (
        next_slow_durations,
        next_stun_durations,
        next_rogue_anti_heal_durations,
        next_mage_burst_durations,
        next_priest_freedom_slow_floor_durations,
        next_spawn_shield_durations,
        slow_is_applied_by_source_and_channel,
        stun_is_applied_by_source_and_channel,
        rogue_poison_anti_heal_is_applied_by_source,
        mage_burst_damage_amplification_is_applied_by_source,
    )


def _derive_accepted_ultimate_status_applications(
    config: EnvConfig,
    accepted_use_ultimate_by_actor_slot: Array,
    accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Derive source-local and recipient-routed accepted ultimate statuses."""
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
    agent_agent_overlap_projection_passes: int,
    collision_projection_passes: int,
    movement_substeps: int,
) -> Array:
    """Return unchanged positions when the conditional Charge phase is inactive."""
    del (
        agent_radii,
        intended_movement_deltas,
        active_mask,
        alive_mask,
        map_width,
        map_height,
        obstacles,
        agent_agent_overlap_projection_passes,
        collision_projection_passes,
        movement_substeps,
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
    """Resolve all accepted Warrior Charge relocations from one pre-state.

    Every desired endpoint is the source-facing tangent point around the
    accepted recipient. The fixed-shape batch then receives one endpoint
    placement pass through the shared geometry boundary. Ordinary voluntary
    movement is deliberately excluded and resolves afterward.
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

    agent_agent_overlap_projection_passes = 1
    collision_projection_passes = DEFAULT_AGENT_PROJECTION_PASSES
    movement_substeps = 1
    there_is_a_charging_warrior = jnp.any(accepted_warrior_charge_by_actor)

    return cast(
        Array,
        jax.lax.cond(
            there_is_a_charging_warrior,
            project_movement_with_geometry,
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
            agent_agent_overlap_projection_passes,
            collision_projection_passes,
            movement_substeps,
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
) -> CombatTransitionFacts:
    """Package existing combat intermediates as authoritative public facts."""
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
    """Derive new successor deaths and source-aligned damage attribution.

    A recipient dies only when it was active and alive at transition start
    and its final clamped successor health is zero. Each contributing source
    retains its gross post-source, post-recipient effective damage; simultaneous
    healing and health clamping do not redistribute or select a killer.
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


def _build_canonical_no_transition_info_object(initial_state: EnvState) -> Info:
    """Return neutral facts for reset or curated initialization.

    Initialization exposes the supplied state's step count only through the
    state itself. Transition facts use their canonical sentinel because no
    action was accepted and no simulator transition occurred.
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

    reset_transition_facts = TransitionFacts(
        has_transition=jnp.asarray(False),
        transition_start_step_count=jnp.asarray(-1, dtype=jnp.int32),
        action_acceptance_facts=reset_action_acceptance_facts,
        combat_transition_facts=reset_combat_transition_facts,
        death_facts=reset_death_facts,
        spawn_shield_facts=reset_spawn_shield_facts,
        respawn_facts=reset_respawn_facts,
        regeneration_facts=reset_regeneration_facts,
    )

    return Info(transition_facts=reset_transition_facts)


def _build_was_respawned_this_transition_by_agent_array(
    current_state: EnvState, config: EnvConfig
) -> Array:
    """Return transition-start dead slots whose team wave is currently due."""
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
    """Apply the simultaneous end-of-transition respawn state override."""
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
    """Return unchanged successor leaves when neither team wave is due."""
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
    """Reset, decrement, or clear each successor combat countdown."""
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
    """Apply eligible post-combat regeneration and return its actual amount."""
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


# Public ---


def initialize_scenario_state(
    initial_state: EnvState, config: EnvConfig
) -> tuple[EnvState, Observation, ActionMask, Info]:
    """Validate and expose one authored state without advancing the simulator."""
    validate_env_config(config)
    validate_scenario_initial_state(config, initial_state)
    obs, action_mask = _build_observation_and_action_mask(initial_state, config)
    info = _build_canonical_no_transition_info_object(initial_state)
    return (initial_state, obs, action_mask, info)


def reset(
    config: EnvConfig, key: Array
) -> tuple[EnvState, Observation, ActionMask, Info]:
    """Create initial fixed-slot state from a host-validated configuration."""
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

    info = _build_canonical_no_transition_info_object(initial_state)

    return (initial_state, initial_observation, initial_action_mask, info)


def step(
    config: EnvConfig,
    current_state: EnvState,
    current_action_mask: ActionMask,
    joint_action: Action,
    key: Array,
) -> tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]:
    """Advance from one paired state/mask snapshot and build the next snapshot.

    ``current_action_mask`` is the mask produced with ``current_state`` by
    reset or the preceding step. It is the sole source of submitted-action
    acceptance; the returned observation and mask describe ``next_state``.
    """
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

    (
        current_mage_damage_amplification_aura_multipliers,
        current_warrior_damage_mitigation_aura_multipliers,
    ) = _derive_aura_damage_multipliers(
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
            current_mage_damage_amplification_aura_multipliers,
            current_warrior_damage_mitigation_aura_multipliers,
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

    (
        next_slow_durations,
        next_stun_durations,
        next_rogue_anti_heal_durations,
        next_mage_burst_durations,
        next_priest_freedom_slow_floor_durations,
        next_spawn_shield_durations,
        slow_is_applied_by_source_and_channel,
        stun_is_applied_by_source_and_channel,
        rogue_poison_anti_heal_is_applied_by_source,
        mage_burst_damage_amplification_is_applied_by_source,
    ) = _resolve_status_duration_lifecycle(
        current_state,
        config,
        accepted_joint_action.use_ultimate,
        accepted_global_pairwise_actor_and_recipient_target_one_hot_matrix,
        combat_effect_aggregation_result.hunter_basic_slow_applied_this_tick_by_global_recipient_slot,
        combat_effect_aggregation_result.priest_freedom_applied_this_tick_by_global_recipient_slot,
        combat_effect_aggregation_result.accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot,
        combat_effect_aggregation_result.hunter_basic_slow_applied_this_tick_by_global_actor_slot,
        next_alive_mask,
    )

    combat_transition_facts = _build_combat_transition_facts(
        combat_effect_aggregation_result,
        combat_effect_has_recipient_by_source,
        combat_effect_recipient_global_slot_by_source,
        slow_is_applied_by_source_and_channel,
        stun_is_applied_by_source_and_channel,
        rogue_poison_anti_heal_is_applied_by_source,
        mage_burst_damage_amplification_is_applied_by_source,
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
            next_spawn_shield_durations,
            next_agent_positions,
        ),
    )

    next_state = EnvState(
        step_count=current_state.step_count + 1,
        agent_positions=next_agent_positions,
        alive_mask=next_alive_mask,
        current_health=next_health_bars,
        # NOTE: Ultimate CD carries over into death to prevent abuse.
        ultimate_cooldowns=next_ultimate_cooldowns,
        slow_durations=next_slow_durations,
        stun_durations=next_stun_durations,
        rogue_poison_anti_heal_durations=next_rogue_anti_heal_durations,
        mage_burst_damage_amplification_durations=next_mage_burst_durations,
        priest_blessing_of_freedom_slow_floor_durations=next_priest_freedom_slow_floor_durations,
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

    rewards = Reward(rewards=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32))

    done_flags = DoneFlags(
        terminated=jnp.array(False),
        truncated=jnp.array(next_state.step_count >= config.max_steps),
    )

    spawn_shield_facts = SpawnShieldTransitionFacts(
        was_active_at_transition_start_by_agent=current_state.spawn_shield_durations
        > 0,
        expired_at_transition_end_by_agent=jnp.logical_and(
            current_state.spawn_shield_durations == 1, next_alive_mask
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
    )

    info = Info(
        transition_facts=transition_facts,
    )

    return (next_state, next_observation, rewards, done_flags, next_action_mask, info)
