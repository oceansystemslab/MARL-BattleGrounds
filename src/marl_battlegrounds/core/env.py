"""Functional reset and step entry points for the core JAX simulator."""

from typing import cast

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
from marl_battlegrounds.core.geometry import (
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
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
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

    # Fixed catalog payloads describe whether each actor owns the interaction.
    does_basic_damage = get_basic_damage_by_class_ids(class_ids) > 0
    does_basic_healing = get_basic_healing_by_class_ids(class_ids) > 0

    # Actor-owned facts broadcast across candidate columns.
    actor_can_damage = jnp.logical_and(is_not_stunned, does_basic_damage)[:, None]
    actor_can_heal = jnp.logical_and(is_not_stunned, does_basic_healing)[:, None]

    global_pairwise_ally_mask, global_pairwise_enemy_mask = (
        _build_global_pairwise_team_masks(config.agent_profile.team_ids)
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
        config, global_pairwise_distances, state.alive_mask
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
    )

    return current_observation, current_action_mask


def _build_intended_movement_deltas(
    current_state: EnvState,
    config: EnvConfig,
    accepted_joint_action: Action,
) -> Array:
    """Build voluntary movement intent from current public control truth.

    Status durations in ``current_state`` must be the values visible at the
    current decision epoch. Statuses accepted during this transition are
    packaged for the next decision and must not retroactively change this
    movement intent. Current stun, death, or inactivity yields zero intent here
    as defense in depth; geometry remains free to displace a zero-intent body
    during collision resolution.
    """
    intended_movement_deltas_unscaled = _JOINT_ACTION_MOVE_TO_DISPLACEMENT_LOOKUP_TABLE[
        accepted_joint_action.move
    ]
    # Control-adjusted speed is shared with the observation contract.
    effective_movement_speeds = derive_effective_movement_speeds(
        current_state.slow_durations,
        current_state.priest_blessing_of_freedom_slow_floor_durations,
        current_state.stun_durations,
        config.agent_profile.base_movement_speeds,
        jnp.logical_and(config.agent_profile.active_mask, current_state.alive_mask),
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
        config.agent_profile.base_movement_speeds,
        jnp.logical_and(config.agent_profile.active_mask, state.alive_mask),
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

    feature_31_to_54 = jnp.concatenate(
        (
            basic_health_and_ultimate_cooldown_capability_features,
            slow_stun_durations_multipliers,
            rogue_anti_heal_capability_features,
            mage_burst_capability_features,
            priest_blessing_of_freedom_capability_features,
            mage_and_warrior_aura_capability_features,
            ultimate_healing_capability_features,
            ultimate_damage_capability_features,
        ),
        axis=-1,
        dtype=jnp.float32,
    )

    return jnp.concatenate(
        (
            features_0_to_14,
            features_15_to_30,
            feature_31_to_54,
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
) -> Action:
    """Canonicalize one submitted action from authoritative pre-state masks.

    Movement acceptance is independent of combat acceptance. The target and
    ultimate-use heads are accepted together from the authoritative joint mask
    so an invalid ultimate attempt cannot fall back to a valid basic action.
    """
    submitted_move_action_is_valid_by_actor_slot = current_action_mask.move_mask[
        _GLOBAL_AGENT_SLOT_INDICES, submitted_joint_action.move
    ]
    accepted_move_joint_action = jnp.where(
        submitted_move_action_is_valid_by_actor_slot,
        submitted_joint_action.move,
        MOVE_STAY,
    )

    submitted_select_target_and_use_ultimate_pair_is_valid_by_actor_slot = (
        current_action_mask.select_target_use_ultimate_joint_mask[
            _GLOBAL_AGENT_SLOT_INDICES,
            submitted_joint_action.select_target,
            submitted_joint_action.use_ultimate,
        ]
    )

    accepted_select_target_joint_action = jnp.where(
        submitted_select_target_and_use_ultimate_pair_is_valid_by_actor_slot,
        submitted_joint_action.select_target,
        0,  # Target-none action
    )
    accepted_use_ultimate_joint_action = jnp.where(
        submitted_select_target_and_use_ultimate_pair_is_valid_by_actor_slot,
        submitted_joint_action.use_ultimate,
        0,  # No-ultimate action
    )

    return Action(
        move=accepted_move_joint_action,
        select_target=accepted_select_target_joint_action,
        use_ultimate=accepted_use_ultimate_joint_action,
    )


def _build_global_pairwise_actor_and_recipient_target_one_hot_matrix(
    accepted_select_target_joint_action: Array,
) -> Array:
    """Map actor-relative accepted targets to dense global recipient rows."""
    # Translate actor-relative selections once for every accepted effect lane.
    accepted_global_target_slot_by_actor_slot = (
        _ACTOR_RELATIVE_SELECT_TARGET_ACTION_TO_GLOBAL_AGENT_SLOT_LOOKUP_TABLE[
            _GLOBAL_AGENT_SLOT_INDICES, accepted_select_target_joint_action
        ]
    )

    return jax.nn.one_hot(
        accepted_global_target_slot_by_actor_slot,
        num_classes=MAX_AGENT_SLOTS,
        dtype=jnp.float32,
    )


def _aggregate_health_effects_and_basic_passives_by_global_slot(
    current_state: EnvState,
    config: EnvConfig,
    accepted_joint_action: Action,
    accepted_recipient_one_hot_by_actor_and_global_slot: Array,
    mage_damage_amplification_aura_multipliers: Array,
    warrior_damage_mitigation_aura_multipliers: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Aggregate accepted health effects and basic-passive applications.

    The accepted target categories are actor-relative. After fixed-slot
    translation, dense one-hot rows route every actor's catalog payload to its
    recipient. Basic and ultimate contributions share this routing boundary.
    Reducing the complete matrices before health mutation preserves simultaneous,
    actor-order-independent resolution. Returned boolean vectors identify
    recipients of accepted Hunter and Priest basics without inferring passive
    triggers from effective health changes, plus recipients of accepted positive
    raw damage for Hunter Trap breaking.
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
        accepted_recipient_one_hot_by_actor_and_global_slot
        * amplified_basic_damage_by_actor_slot[:, None]
    )

    basic_healing_contribution_by_actor_and_global_recipient_slot = (
        accepted_recipient_one_hot_by_actor_and_global_slot
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
        accepted_recipient_one_hot_by_actor_and_global_slot
        * amplified_ultimate_damage_by_actor_slot[:, None]
    )

    ultimate_healing_contribution_by_actor_and_global_recipient_slot = (
        accepted_recipient_one_hot_by_actor_and_global_slot
        * raw_ultimate_healing_by_actor_slot[:, None]
    )

    # Trap break follows accepted positive raw damage, not effective health loss.
    accepted_positive_raw_basic_damage_received_this_tick = (
        jnp.sum(
            accepted_recipient_one_hot_by_actor_and_global_slot
            * raw_basic_damage_by_actor_slot[:, None],
            axis=0,
        )
        > 0
    )

    accepted_positive_raw_ultimate_damage_received_this_tick = (
        jnp.sum(
            accepted_recipient_one_hot_by_actor_and_global_slot
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
    hunter_basic_slow_applied_this_tick_by_global_recipient_slot_mask = jnp.logical_and(
        accepted_recipient_one_hot_by_actor_and_global_slot,
        jnp.logical_and(
            _active_hunter_class_mask(config)[:, None],
            actor_applies_accepted_basic_effect[:, None],
        ),
    )

    hunter_basic_slow_applied_this_tick_by_global_recipient_slot = jnp.any(
        hunter_basic_slow_applied_this_tick_by_global_recipient_slot_mask, axis=0
    )

    priest_freedom_applied_this_tick_by_global_recipient_slot_mask = jnp.logical_and(
        accepted_recipient_one_hot_by_actor_and_global_slot,
        jnp.logical_and(
            _active_priest_class_mask(config)[:, None],
            actor_applies_accepted_basic_effect[:, None],
        ),
    )

    priest_freedom_applied_this_tick_by_global_recipient_slot = jnp.any(
        priest_freedom_applied_this_tick_by_global_recipient_slot_mask, axis=0
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

    return (
        total_damage_received_by_global_recipient_slot,
        total_healing_received_by_global_recipient_slot,
        hunter_basic_slow_applied_this_tick_by_global_recipient_slot,
        priest_freedom_applied_this_tick_by_global_recipient_slot,
        accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot,
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
) -> tuple[Array, Array]:
    """Derive bounded Mage outgoing and Warrior incoming aura modifiers.

    Rows represent aura emitters and columns represent beneficiary slots.
    Only active, living allies with real team IDs participate. Auras include
    their emitter, use inclusive radius boundaries, and stack multiplicatively
    before the completed vectors are bounded.
    """
    global_pairwise_ally_mask, _ = _build_global_pairwise_team_masks(
        config.agent_profile.team_ids
    )

    active_and_alive = jnp.logical_and(alive_mask, config.agent_profile.active_mask)
    active_and_alive_pairs = jnp.logical_and(
        active_and_alive[None, :], active_and_alive[:, None]
    )
    global_pairwise_active_and_alive_ally_mask = jnp.logical_and(
        active_and_alive_pairs, global_pairwise_ally_mask
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
    accepted_recipient_one_hot_by_actor_and_global_slot: Array,
    hunter_basic_slow_applied_this_tick_by_global_recipient_slot: Array,
    priest_freedom_applied_this_tick_by_global_recipient_slot: Array,
    accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return the five successor-state status-duration arrays.

    Current durations age once toward zero. Accepted positive raw damage then
    clears only the aged successor of a pre-existing Hunter Trap, after which
    fresh source-local applications merge at full configured duration. A fresh
    application never shortens a longer aged remainder and first governs the
    next observable decision epoch.
    """

    # Derive fresh applications once from the accepted action at this epoch.
    (
        mage_burst_activated_this_tick_by_actor_slot,
        warrior_charge_applied_this_tick_by_recipient_slot,
        hunter_trap_applied_this_tick_by_recipient_slot,
        rogue_poison_applied_this_tick_by_recipient_slot,
    ) = _derive_accepted_ultimate_status_applications(
        config,
        accepted_use_ultimate_by_actor_slot,
        accepted_recipient_one_hot_by_actor_and_global_slot,
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
        mage_burst_activated_this_tick_by_actor_slot,
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

    return (
        next_slow_durations,
        next_stun_durations,
        next_rogue_anti_heal_durations,
        next_mage_burst_durations,
        next_priest_freedom_slow_floor_durations,
    )


def _derive_accepted_ultimate_status_applications(
    config: EnvConfig,
    accepted_use_ultimate_by_actor_slot: Array,
    accepted_recipient_one_hot_by_actor_and_global_slot: Array,
) -> tuple[Array, Array, Array, Array]:
    """Derive source-local and recipient-routed accepted ultimate statuses."""
    uses_ultimate_this_tick = accepted_use_ultimate_by_actor_slot == 1

    # Mage Burst applies to its source; targeted ultimates reduce by recipient.
    mage_burst_activated_this_tick_by_actor_slot = jnp.logical_and(
        _active_mage_class_mask(config), uses_ultimate_this_tick
    )

    warrior_uses_accepted_ultimate_this_tick_by_actor_slot = jnp.logical_and(
        _active_warrior_class_mask(config), uses_ultimate_this_tick
    )
    warrior_charge_applied_this_tick_by_recipient_slot = jnp.any(
        jnp.logical_and(
            warrior_uses_accepted_ultimate_this_tick_by_actor_slot[:, None],
            accepted_recipient_one_hot_by_actor_and_global_slot,
        ),
        axis=0,
    )

    hunter_uses_accepted_ultimate_this_tick_by_actor_slot = jnp.logical_and(
        _active_hunter_class_mask(config), uses_ultimate_this_tick
    )
    hunter_trap_applied_this_tick_by_recipient_slot = jnp.any(
        jnp.logical_and(
            hunter_uses_accepted_ultimate_this_tick_by_actor_slot[:, None],
            accepted_recipient_one_hot_by_actor_and_global_slot,
        ),
        axis=0,
    )

    rogue_uses_accepted_ultimate_this_tick_by_actor_slot = jnp.logical_and(
        _active_rogue_class_mask(config), uses_ultimate_this_tick
    )
    rogue_poison_applied_this_tick_by_recipient_slot = jnp.any(
        jnp.logical_and(
            rogue_uses_accepted_ultimate_this_tick_by_actor_slot[:, None],
            accepted_recipient_one_hot_by_actor_and_global_slot,
        ),
        axis=0,
    )

    return (
        mage_burst_activated_this_tick_by_actor_slot,
        warrior_charge_applied_this_tick_by_recipient_slot,
        hunter_trap_applied_this_tick_by_recipient_slot,
        rogue_poison_applied_this_tick_by_recipient_slot,
    )


# Public ---


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
    # TODO(Scenario): Keep curated scenario starts out of ordinary reset. A future
    # scenario loader should validate and return EnvState values that reuse the
    # same transition, observation, and mask machinery.
    del key
    initial_state = EnvState(
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=config.initial_agent_positions,
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

    initial_observation, initial_action_mask = _build_observation_and_action_mask(
        initial_state, config
    )

    info = Info()

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
    accepted_joint_action = _build_accepted_joint_action_from_submitted_joint_action(
        current_action_mask=current_action_mask, submitted_joint_action=joint_action
    )

    accepted_recipient_one_hot_by_actor_and_global_slot = (
        _build_global_pairwise_actor_and_recipient_target_one_hot_matrix(
            accepted_joint_action.select_target
        )
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
        config, current_global_pairwise_distances, current_state.alive_mask
    )

    (
        total_effective_damage_received_by_global_slot,
        total_effective_healing_received_by_global_slot,
        hunter_basic_slow_applied_this_tick_by_global_recipient_slot,
        priest_freedom_applied_this_tick_by_global_recipient_slot,
        accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot,
    ) = _aggregate_health_effects_and_basic_passives_by_global_slot(
        current_state,
        config,
        accepted_joint_action,
        accepted_recipient_one_hot_by_actor_and_global_slot,
        current_mage_damage_amplification_aura_multipliers,
        current_warrior_damage_mitigation_aura_multipliers,
    )

    next_health_after_effective_damage_and_healing = (
        _compute_health_after_simultaneous_damage_and_healing(
            total_effective_damage_received_by_global_slot,
            total_effective_healing_received_by_global_slot,
            current_state,
            config,
        )
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

    next_agent_positions = project_movement_with_geometry(
        current_state.agent_positions,
        config.agent_profile.agent_radii,
        intended_movement_deltas,
        config.agent_profile.active_mask,
        current_state.alive_mask,
        config.map_width,
        config.map_height,
        config.obstacles,
    )

    (
        next_slow_durations,
        next_stun_durations,
        next_rogue_anti_heal_durations,
        next_mage_burst_durations,
        next_priest_freedom_slow_floor_durations,
    ) = _resolve_status_duration_lifecycle(
        current_state,
        config,
        accepted_joint_action.use_ultimate,
        accepted_recipient_one_hot_by_actor_and_global_slot,
        hunter_basic_slow_applied_this_tick_by_global_recipient_slot,
        priest_freedom_applied_this_tick_by_global_recipient_slot,
        accepted_positive_raw_damage_received_this_tick_by_global_recipient_slot,
    )

    next_state = EnvState(
        step_count=current_state.step_count + 1,
        agent_positions=next_agent_positions,
        alive_mask=current_state.alive_mask,
        current_health=next_health_after_effective_damage_and_healing,
        ultimate_cooldowns=next_ultimate_cooldowns,
        slow_durations=next_slow_durations,
        stun_durations=next_stun_durations,
        rogue_poison_anti_heal_durations=next_rogue_anti_heal_durations,
        mage_burst_damage_amplification_durations=next_mage_burst_durations,
        priest_blessing_of_freedom_slow_floor_durations=next_priest_freedom_slow_floor_durations,
    )

    next_observation, next_action_mask = _build_observation_and_action_mask(
        next_state, config
    )

    rewards = Reward(rewards=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32))

    done_flags = DoneFlags(
        terminated=jnp.array(False),
        truncated=jnp.array(next_state.step_count >= config.max_steps),
    )

    info = Info()

    return (next_state, next_observation, rewards, done_flags, next_action_mask, info)
