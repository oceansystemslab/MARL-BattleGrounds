"""Functional reset and step entry points for the core JAX simulator."""

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.combat import (
    BASIC_DAMAGE_BY_CLASS,
    BASIC_HEALING_BY_CLASS,
    GLOBAL_SLOW_FLOOR,
    HUNTER_BASIC_SLOW_DURATION_TICKS,
    HUNTER_BASIC_SLOW_MULTIPLIER,
    HUNTER_TRAP_STUN_DURATION_TICKS,
    MAGE_DAMAGE_AURA_MULTIPLIER,
    MAGE_DAMAGE_AURA_RADIUS,
    MAGE_ULT_DAMAGE_DURATION_TICKS,
    MAGE_ULT_DAMAGE_MULTIPLIER,
    ONLY_ALLY_TARGET_ULTIMATE_MODE,
    ONLY_ENEMY_TARGET_ULTIMATE_MODE,
    ONLY_NONE_TARGET_ULTIMATE_MODE,
    PRIEST_HEAL_SPEED_FLOOR,
    PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS,
    PRIEST_ULT_HEAL_AMOUNT,
    ROGUE_POISON_ANTI_HEAL_DURATION_TICKS,
    ROGUE_POISON_ANTI_HEAL_MULTIPLIER,
    ROGUE_POISON_SLOW_DURATION_TICKS,
    ROGUE_POISON_SLOW_MULTIPLIER,
    ROGUE_POISON_STUN_DURATION_TICKS,
    ULTIMATE_COOLDOWN_BY_CLASS,
    WARRIOR_CHARGE_SLOW_DURATION_TICKS,
    WARRIOR_CHARGE_SLOW_MULTIPLIER,
    WARRIOR_CHARGE_STUN_DURATION_TICKS,
    WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER,
    WARRIOR_DAMAGE_MITIGATION_AURA_RADIUS,
    derive_effective_movement_speeds_from_durations,
    derive_status_magnitudes,
    get_basic_damage_by_class_ids,
    get_basic_healing_by_class_ids,
    get_ultimate_target_mode_by_class_ids,
)
from marl_battlegrounds.core.geometry import (
    has_clear_line_of_sight,
    project_movement_with_geometry,
)
from marl_battlegrounds.core.types import (
    CONTEXT_FEATURES,
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBJECTIVE_SLOTS,
    MAX_OBSTACLE_SLOTS,
    MOVE_STAY,
    NO_TEAM_ID,
    NUM_MOVE_ACTIONS,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    NUM_TARGET_ACTIONS,
    OBJECTIVE_FEATURES,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    UNIT_FEATURES,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)

TEAM_A_START = 0
TEAM_A_END = MAX_AGENTS_PER_TEAM
TEAM_B_START = MAX_AGENTS_PER_TEAM
TEAM_B_END = MAX_AGENT_SLOTS

# Private Helpers ---

# Direction rows are unit-length and ordered to match the MOVE_* constants.
_INV_SQRT_2 = 1 / jnp.sqrt(2.0)
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

    Self-visibility falls out naturally from the same predicate.
    """
    # Pairwise observer-candidate validity from active/alive state.
    alive_active_mask = jnp.logical_and(
        config.agent_profile.active_mask, state.alive_mask
    )
    global_pairwise_validity_mask = jnp.logical_and(
        alive_active_mask[:, None],
        alive_active_mask[None, :],
    )

    # Pairwise center-to-center distances.
    observer_positions_bc = state.agent_positions[:, None, :]
    candidate_positions_bc = state.agent_positions[None, :, :]
    pairwise_displacement_vectors = observer_positions_bc - candidate_positions_bc
    global_pairwise_distances = cast(
        Array,
        jnp.linalg.norm(pairwise_displacement_vectors, axis=-1),
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

    return (
        jnp.logical_and(
            global_pairwise_validity_mask,
            jnp.logical_and(observation_radii_mask, los_mask),
        ),
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
    team_ids = config.agent_profile.team_ids
    active_and_alive_mask = jnp.logical_and(
        config.agent_profile.active_mask, state.alive_mask
    )

    basic_interaction_radii = config.agent_profile.basic_interaction_radii[:, None]
    basic_interaction_radius_mask = global_pairwise_distances <= basic_interaction_radii

    # Stun is actor-side control: any active channel removes non-empty targets.
    is_not_stunned = jnp.all(state.stun_durations == 0, axis=-1)

    # Fixed catalog payloads describe whether each actor owns the interaction.
    does_basic_damage = get_basic_damage_by_class_ids(class_ids) > 0
    does_basic_healing = get_basic_healing_by_class_ids(class_ids) > 0

    # Actor-owned facts broadcast across candidate columns.
    actor_can_damage = jnp.logical_and(is_not_stunned, does_basic_damage)[:, None]
    actor_can_heal = jnp.logical_and(is_not_stunned, does_basic_healing)[:, None]

    # Equal padding sentinels never constitute a real ally relation.
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
        is_not_stunned, has_available_enemy_targeted_ultimate
    )[:, None]
    actor_can_use_ally_targeted_ultimate = jnp.logical_and(
        is_not_stunned, has_available_ally_targeted_ultimate
    )[:, None]
    actor_can_use_no_target_ultimate = jnp.logical_and(
        is_not_stunned, has_available_no_target_ultimate
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

    Active, living actors may submit every movement category. Dead or inactive
    slots expose only ``MOVE_STAY``, the effect-inert canonical submission.
    """
    active_and_alive_mask = jnp.logical_and(
        config.agent_profile.active_mask, state.alive_mask
    )
    canonical_stay_mask = jnp.arange(NUM_MOVE_ACTIONS)[None, :] == MOVE_STAY

    return jnp.logical_or(active_and_alive_mask[:, None], canonical_stay_mask)


def _build_observation_and_action_mask(
    state: EnvState, config: EnvConfig
) -> tuple[Observation, ActionMask]:
    """Build the current observation contract from one slot-aligned state.

    Self rows are canonical fixed-slot agent rows. Ally and enemy unit rows use
    the same agent-feature schema in relation-local candidate order, with
    nonvisible candidate rows zeroed by the visibility masks.
    """

    self_features = _build_self_features(state, config)
    ally_features = _build_ally_features(self_features)
    enemy_features = _build_enemy_features(self_features)
    global_visibility_mask, global_pairwise_distances = (
        _build_global_visibility_mask_and_distances(state, config)
    )
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

    map_obstacle_features = jnp.broadcast_to(
        config.obstacles[None, :, :],
        (MAX_AGENT_SLOTS, MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
    )

    action_mask = ActionMask(
        move_mask=move_mask,
        select_target_mask=select_target_mask,
        use_ultimate_mask=use_ultimate_mask,
        select_target_use_ultimate_joint_mask=select_target_use_ultimate_joint_mask,
    )

    obs = Observation(
        self_features=self_features,
        ally_unit_features=ally_features,
        enemy_unit_features=enemy_features,
        map_obstacle_features=map_obstacle_features,
        objective_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_OBJECTIVE_SLOTS, OBJECTIVE_FEATURES),
            dtype=jnp.float32,
        ),
        context_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, CONTEXT_FEATURES), dtype=jnp.float32
        ),
        ally_visibility_mask=ally_visibility_mask,
        enemy_visibility_mask=enemy_visibility_mask,
    )

    return obs, action_mask


def _build_intended_movement_deltas(
    joint_action: Action, state: EnvState, config: EnvConfig
) -> Array:
    """Convert per-slot movement action IDs into scaled displacement vectors."""
    intended_movement_deltas_unscaled = _JOINT_ACTION_MOVE_TO_DISPLACEMENT_LOOKUP_TABLE[
        joint_action.move
    ]
    # Status-adjusted speed is shared with the observation contract.
    effective_movement_speeds = derive_effective_movement_speeds_from_durations(
        config.agent_profile.base_movement_speeds,
        state.slow_durations,
        state.priest_blessing_of_freedom_slow_floor_durations,
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


def _derive_effective_movement_speeds_from_multipliers(
    base_movement_speeds: Array,
    slow_multipliers: Array,
    priest_blessing_of_freedom_slow_floor_fraction: Array,
) -> Array:
    """Apply stacked slows, the Priest floor, and the global floor to speeds."""

    effective_movement_multipliers = jnp.maximum(
        jnp.prod(slow_multipliers, axis=-1),
        priest_blessing_of_freedom_slow_floor_fraction,
    )

    return base_movement_speeds * jnp.maximum(
        effective_movement_multipliers, GLOBAL_SLOW_FLOOR
    )


def _build_self_features(state: EnvState, config: EnvConfig) -> Array:
    """Build slot-aligned self rows from the shared agent-feature schema."""

    (
        slow_multipliers,
        rogue_poison_anti_heal_multipliers,
        priest_blessing_of_freedom_slow_floor_fraction,
    ) = derive_status_magnitudes(
        state.slow_durations,
        state.rogue_poison_anti_heal_durations,
        state.priest_blessing_of_freedom_slow_floor_durations,
    )

    effective_movement_speeds = _derive_effective_movement_speeds_from_multipliers(
        config.agent_profile.base_movement_speeds,
        slow_multipliers,
        priest_blessing_of_freedom_slow_floor_fraction,
    )

    features_0_to_14 = jnp.concatenate(
        (
            state.agent_positions,
            config.agent_profile.agent_radii[:, None],
            config.agent_profile.team_ids[:, None],
            config.agent_profile.active_mask[:, None],
            state.alive_mask[:, None],
            config.agent_profile.class_ids[:, None],
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

    # TODO: Inert placeholders
    mage_damage_amplification_aura_multipliers = jnp.ones((MAX_AGENT_SLOTS,))
    warrior_damage_mitigation_aura_multipliers = jnp.ones((MAX_AGENT_SLOTS,))

    features_15_to_30 = jnp.concatenate(
        (
            state.slow_durations,
            slow_multipliers,
            state.stun_durations,
            state.rogue_poison_anti_heal_durations[:, None],
            rogue_poison_anti_heal_multipliers[:, None],
            state.mage_burst_damage_amplification_durations[:, None],
            state.priest_blessing_of_freedom_slow_floor_durations[:, None],
            priest_blessing_of_freedom_slow_floor_fraction[:, None],
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

    basic_dmg_heal_ult_cds = jnp.where(
        active_mask_bc,
        jnp.concatenate(
            (
                BASIC_DAMAGE_BY_CLASS[config.agent_profile.class_ids][:, None],
                BASIC_HEALING_BY_CLASS[config.agent_profile.class_ids][:, None],
                ULTIMATE_COOLDOWN_BY_CLASS[config.agent_profile.class_ids][:, None],
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
        jnp.asarray([MAGE_ULT_DAMAGE_DURATION_TICKS, MAGE_ULT_DAMAGE_MULTIPLIER])[
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

    mage_warrior_priest_mask = jnp.concatenate(
        (
            jnp.tile(mage_mask[:, None], 2),
            jnp.tile(warrior_mask[:, None], 2),
            priest_mask[:, None],
        ),
        axis=-1,
    )

    mage_war_aura_priest_ult_features = jnp.where(
        mage_warrior_priest_mask,
        jnp.asarray(
            [
                MAGE_DAMAGE_AURA_RADIUS,
                MAGE_DAMAGE_AURA_MULTIPLIER,
                WARRIOR_DAMAGE_MITIGATION_AURA_RADIUS,
                WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER,
                PRIEST_ULT_HEAL_AMOUNT,
            ]
        )[None, :],
        0.0,
    ).astype(jnp.float32)

    feature_31_to_53 = jnp.concatenate(
        (
            basic_dmg_heal_ult_cds,
            slow_stun_durations_multipliers,
            rogue_anti_heal_capability_features,
            mage_burst_capability_features,
            priest_blessing_of_freedom_capability_features,
            mage_war_aura_priest_ult_features,
        ),
        axis=-1,
        dtype=jnp.float32,
    )

    return jnp.concatenate(
        (
            features_0_to_14,
            features_15_to_30,
            feature_31_to_53,
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


# Public ---


def reset(
    config: EnvConfig, key: Array
) -> tuple[EnvState, Observation, ActionMask, Info]:
    """Create the initial fixed-slot simulator state and placeholders."""

    # Reset keeps all arrays at MAX_AGENT_SLOTS length. Smaller tasks use the
    # resolved profile's active mask to distinguish agents from padded slots.
    # Ordinary reset starts all active agents alive. Scenario loaders may later
    # create active-but-dead agents from curated states.
    # TODO(M4+): Use key when reset begins sampling spawn positions or randomized
    # layouts. Deterministic dummy reset may accept the key without consuming it.
    # TODO(Scenario): Keep curated scenario starts out of ordinary reset. A future
    # scenario loader should validate and return EnvState values that reuse the
    # same transition, observation, and mask machinery.
    # TODO(Config Validation): Eventually implement a non-JAX validator in pipeline.

    deterministic_key = jax.random.key(42)
    max_val = jnp.min(jnp.array([config.map_width, config.map_height]))
    # Reset emits geometry-valid placeholder centers so MOVE_STAY does not
    # trigger corrective projection before scenario loading exists.
    default_agent_positions = jax.random.uniform(
        deterministic_key,
        shape=(MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        dtype=jnp.float32,
        minval=0 + 0.5,
        maxval=max_val - 0.5,  # Placeholder default agent positions
    )

    state = EnvState(
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=default_agent_positions,  # Placeholder
        alive_mask=config.agent_profile.active_mask,
        current_health=config.agent_profile.max_health,
        # At restart, current health should be max health.
        ultimate_cooldowns=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        # Everything below is a placeholder
        slow_durations=jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32),
        stun_durations=jnp.zeros((MAX_AGENT_SLOTS, NUM_STUN_CHANNELS), dtype=jnp.int32),
        rogue_poison_anti_heal_durations=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        mage_burst_damage_amplification_durations=jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        priest_blessing_of_freedom_slow_floor_durations=jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
    )

    obs, action_mask = _build_observation_and_action_mask(state, config)

    info = Info()

    return (state, obs, action_mask, info)


def step(
    config: EnvConfig, state: EnvState, joint_action: Action, key: Array
) -> tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]:
    """Advance movement and rebuild current fixed-shape observations and masks."""

    intended_movement_deltas = _build_intended_movement_deltas(
        joint_action, state, config
    )

    next_agent_positions = project_movement_with_geometry(
        state.agent_positions,
        config.agent_profile.agent_radii,
        intended_movement_deltas,
        config.agent_profile.active_mask,
        state.alive_mask,
        config.map_width,
        config.map_height,
        config.obstacles,
    )

    # TODO: mutation of health, cooldowns, statuses, etc.

    next_state = EnvState(
        step_count=state.step_count + 1,
        agent_positions=next_agent_positions,
        alive_mask=state.alive_mask,
        current_health=state.current_health,
        ultimate_cooldowns=state.ultimate_cooldowns,
        slow_durations=state.slow_durations,
        stun_durations=state.stun_durations,
        rogue_poison_anti_heal_durations=state.rogue_poison_anti_heal_durations,
        mage_burst_damage_amplification_durations=state.mage_burst_damage_amplification_durations,
        priest_blessing_of_freedom_slow_floor_durations=state.priest_blessing_of_freedom_slow_floor_durations,
    )

    obs, action_mask = _build_observation_and_action_mask(next_state, config)

    rewards = Reward(rewards=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32))

    done_flags = DoneFlags(
        terminated=jnp.array(False),
        truncated=jnp.array(next_state.step_count >= config.max_steps),
    )

    info = Info()

    return (next_state, obs, rewards, done_flags, action_mask, info)
