"""Functional reset and step entry points for the core JAX simulator."""

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.combat import (
    get_basic_interaction_radius_by_class_ids,
    get_body_radius_by_class_ids,
    get_max_health_by_class_ids,
    get_movement_speed_by_class_ids,
    get_observation_radius_by_class_ids,
    get_ultimate_interaction_radius_by_class_ids,
)
from marl_battlegrounds.core.geometry import (
    has_clear_line_of_sight,
    project_movement_with_geometry,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_MOVEMENT_SPEED,
    AGENT_FEATURE_OBSERVATION_RADIUS,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_TEAM_ID,
    AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    CONTEXT_FEATURES,
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBJECTIVE_SLOTS,
    MAX_OBSTACLE_SLOTS,
    NEUTRAL_CLASS_ID,
    NUM_MOVE_ACTIONS,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    OBJECTIVE_FEATURES,
    OBSTACLE_FEATURES,
    SELF_FEATURES,
    UNIT_FEATURES,
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
    alive_active_mask = jnp.logical_and(state.active_mask, state.alive_mask)
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
    observer_radii_bc = state.observation_radii[:, None]
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
    - team 0 occupies global slots ``0..MAX_AGENTS_PER_TEAM - 1``;
    - team 1 occupies global slots ``MAX_AGENTS_PER_TEAM..MAX_AGENT_SLOTS - 1``.
    """
    team_0_start = 0
    team_0_end = MAX_AGENTS_PER_TEAM
    team_1_start = MAX_AGENTS_PER_TEAM
    team_1_end = MAX_AGENT_SLOTS

    ally_mask_team_0 = global_mask[
        team_0_start:team_0_end,
        team_0_start:team_0_end,
    ]
    enemy_mask_team_0 = global_mask[
        team_0_start:team_0_end,
        team_1_start:team_1_end,
    ]

    ally_mask_team_1 = global_mask[
        team_1_start:team_1_end,
        team_1_start:team_1_end,
    ]
    enemy_mask_team_1 = global_mask[
        team_1_start:team_1_end,
        team_0_start:team_0_end,
    ]

    ally_mask = jnp.vstack((ally_mask_team_0, ally_mask_team_1))
    enemy_mask = jnp.vstack((enemy_mask_team_0, enemy_mask_team_1))

    return (ally_mask, enemy_mask)


def _build_global_targetability_mask(
    state: EnvState, global_visibility_mask: Array, global_pairwise_distances: Array
) -> Array:
    """Build Milestone 4 basic-interaction targetability between global slots.

    Milestone 4 targetability is deliberately basic-only: visible, active/alive
    candidates within the observer's current basic interaction radius are valid
    under neutral placeholder class legality. Ultimate-specific targetability
    starts in Milestone 5.
    """
    basic_interaction_radii = state.basic_interaction_radii[:, None]
    basic_interaction_radius_mask = global_pairwise_distances <= basic_interaction_radii

    # Neutral placeholder legality: any visible in-range unit candidate is legal
    # until Milestone 5 replaces this with class-specific ally/enemy rules.
    class_legality_mask = jnp.ones_like(global_visibility_mask, dtype=bool)

    return jnp.logical_and(
        global_visibility_mask,
        jnp.logical_and(basic_interaction_radius_mask, class_legality_mask),
    )


def _build_observation(state: EnvState, config: EnvConfig) -> Observation:
    """Build the current observation contract from one slot-aligned state.

    Self rows are canonical fixed-slot agent rows. Ally and enemy unit rows use
    the same agent-feature schema in relation-local candidate order, with
    nonvisible candidate rows zeroed by the visibility masks.
    """

    self_features = _build_self_features(state)

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

    map_obstacle_features = jnp.broadcast_to(
        config.obstacles[None, :, :],
        (MAX_AGENT_SLOTS, MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
    )

    ally_targetability_mask, enemy_targetability_mask = _build_ally_enemy_masks(
        _build_global_targetability_mask(
            state, global_visibility_mask, global_pairwise_distances
        )
    )

    return Observation(
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
        ally_targetability_mask=ally_targetability_mask,
        enemy_targetability_mask=enemy_targetability_mask,
    )


def _build_action_mask(state: EnvState, observation: Observation) -> ActionMask:
    """Build action masks from observer liveness and observation masks."""
    ones_column_vector = jnp.ones(shape=(MAX_AGENT_SLOTS, 1), dtype=bool)
    zeros_column_vector = jnp.zeros(shape=(MAX_AGENT_SLOTS, 1), dtype=bool)

    target_mask = jnp.concatenate(
        (
            ones_column_vector,
            observation.ally_targetability_mask,
            observation.enemy_targetability_mask,
        ),
        axis=1,
    )

    ult_mask = jnp.concatenate((ones_column_vector, zeros_column_vector), axis=1)

    alive_active_mask_bc = jnp.logical_and(state.active_mask, state.alive_mask)[:, None]

    return ActionMask(
        move=jnp.logical_and(
            jnp.ones(shape=(MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS), dtype=bool),
            alive_active_mask_bc,
        ),
        target=jnp.logical_and(target_mask, alive_active_mask_bc),
        use_ultimate=jnp.logical_and(
            ult_mask,
            alive_active_mask_bc,
        ),
    )


def _build_intended_movement_deltas(joint_action: Action, state: EnvState) -> Array:
    """Convert per-slot movement action IDs into scaled displacement vectors."""
    intended_movement_deltas_unscaled = _JOINT_ACTION_MOVE_TO_DISPLACEMENT_LOOKUP_TABLE[
        joint_action.move
    ]

    intended_movement_deltas = (
        state.movement_speeds[:, None] * intended_movement_deltas_unscaled
    )

    return intended_movement_deltas


def _build_self_features(state: EnvState) -> Array:
    """Build slot-aligned self rows from the shared agent-feature schema."""
    self_features = jnp.zeros(shape=(MAX_AGENT_SLOTS, SELF_FEATURES), dtype=jnp.float32)
    self_features = self_features.at[:, AGENT_FEATURE_X : AGENT_FEATURE_Y + 1].set(
        state.agent_positions
    )
    self_features = self_features.at[:, AGENT_FEATURE_RADIUS].set(state.agent_radii)
    self_features = self_features.at[:, AGENT_FEATURE_TEAM_ID].set(
        state.team_ids.astype(jnp.float32)
    )
    self_features = self_features.at[:, AGENT_FEATURE_ACTIVE].set(
        state.active_mask.astype(jnp.float32)
    )
    self_features = self_features.at[:, AGENT_FEATURE_ALIVE].set(
        state.alive_mask.astype(jnp.float32)
    )
    self_features = self_features.at[:, AGENT_FEATURE_CLASS_ID].set(
        state.class_ids.astype(jnp.float32)
    )
    self_features = self_features.at[:, AGENT_FEATURE_MOVEMENT_SPEED].set(
        state.movement_speeds
    )
    self_features = self_features.at[:, AGENT_FEATURE_OBSERVATION_RADIUS].set(
        state.observation_radii
    )
    self_features = self_features.at[:, AGENT_FEATURE_BASIC_INTERACTION_RADIUS].set(
        state.basic_interaction_radii
    )
    self_features = self_features.at[:, AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS].set(
        state.ultimate_interaction_radii
    )

    return self_features


def _build_ally_features(self_features: Array) -> Array:
    """Project global self rows into relation-local ally candidate rows."""
    ally_features = jnp.zeros(
        (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES), dtype=jnp.float32
    )

    team_0_start = 0
    team_0_end = MAX_AGENTS_PER_TEAM
    team_1_start = MAX_AGENTS_PER_TEAM
    team_1_end = MAX_AGENT_SLOTS

    ally_features = ally_features.at[team_0_start:team_0_end, :, :].set(
        self_features[team_0_start:team_0_end, :]
    )
    ally_features = ally_features.at[team_1_start:team_1_end, :, :].set(
        self_features[team_1_start:team_1_end, :]
    )

    return ally_features


def _build_enemy_features(self_features: Array) -> Array:
    """Project global self rows into relation-local enemy candidate rows."""
    enemy_features = jnp.zeros(
        (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES), dtype=jnp.float32
    )

    team_0_start = 0
    team_0_end = MAX_AGENTS_PER_TEAM
    team_1_start = MAX_AGENTS_PER_TEAM
    team_1_end = MAX_AGENT_SLOTS

    enemy_features = enemy_features.at[team_0_start:team_0_end, :, :].set(
        self_features[team_1_start:team_1_end, :]
    )
    enemy_features = enemy_features.at[team_1_start:team_1_end, :, :].set(
        self_features[team_0_start:team_0_end, :]
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

    # Reset keeps all arrays at MAX_AGENT_SLOTS length. Smaller tasks use
    # active_mask to distinguish real agents from padded slots.
    # Ordinary reset starts all active agents alive. Scenario loaders may later
    # create active-but-dead agents from curated states.
    # TODO(M4+): Use key when reset begins sampling spawn positions or randomized
    # layouts. Deterministic dummy reset may accept the key without consuming it.
    # TODO(Scenario): Keep curated scenario starts out of ordinary reset. A future
    # scenario loader should validate and return EnvState values that reuse the
    # same transition, observation, and mask machinery.
    # TODO(Config Validation): Eventually implement a non-JAX validator in pipeline.

    team_0_ids = jnp.zeros((MAX_AGENTS_PER_TEAM,), dtype=jnp.int32)
    team_1_ids = jnp.ones((MAX_AGENTS_PER_TEAM,), dtype=jnp.int32)

    indices = jnp.arange(MAX_AGENT_SLOTS)

    team_0_active_mask = indices < config.team_size

    team_1_active_mask = (indices >= MAX_AGENTS_PER_TEAM) & (
        indices < MAX_AGENTS_PER_TEAM + config.team_size
    )

    active_mask = jnp.logical_or(team_0_active_mask, team_1_active_mask)

    initial_class_ids = jnp.where(
        active_mask, config.initial_class_ids, NEUTRAL_CLASS_ID
    )

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

    max_health = get_max_health_by_class_ids(initial_class_ids)

    state = EnvState(
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=default_agent_positions,  # Placeholder
        agent_radii=get_body_radius_by_class_ids(initial_class_ids),
        team_ids=jnp.concat([team_0_ids, team_1_ids]),
        class_ids=initial_class_ids,  # (MAX_AGENT_SLOTS,), int32
        movement_speeds=get_movement_speed_by_class_ids(initial_class_ids),
        observation_radii=get_observation_radius_by_class_ids(initial_class_ids),
        basic_interaction_radii=get_basic_interaction_radius_by_class_ids(
            initial_class_ids
        ),
        ultimate_interaction_radii=get_ultimate_interaction_radius_by_class_ids(
            initial_class_ids
        ),
        active_mask=active_mask,
        alive_mask=active_mask,  # Placeholder
        current_health=max_health,
        # At restart, current health should be max health.
        max_health=max_health,
        ultimate_cooldowns=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        # Everything below is a placeholder
        slow_durations=jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32),
        slow_multipliers=jnp.ones(
            (MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.float32
        ),
        stun_durations=jnp.zeros((MAX_AGENT_SLOTS, NUM_STUN_CHANNELS), dtype=jnp.int32),
        anti_heal_durations=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        anti_heal_multipliers=jnp.ones((MAX_AGENT_SLOTS,), dtype=jnp.float32),
        damage_amplification_durations=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        damage_amplification_multipliers=jnp.ones(
            (MAX_AGENT_SLOTS,), dtype=jnp.float32
        ),
        blessing_of_freedom_durations=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )

    obs = _build_observation(state, config)

    action_mask = _build_action_mask(state, obs)

    info = Info()

    return (state, obs, action_mask, info)


def step(
    config: EnvConfig, state: EnvState, joint_action: Action, key: Array
) -> tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]:
    """Advance movement while preserving current Milestone 4 placeholders."""

    intended_movement_deltas = _build_intended_movement_deltas(joint_action, state)

    next_agent_positions = project_movement_with_geometry(
        state.agent_positions,
        state.agent_radii,
        intended_movement_deltas,
        state.active_mask,
        state.alive_mask,
        config.map_width,
        config.map_height,
        config.obstacles,
    )

    next_state = EnvState(
        step_count=state.step_count + 1,
        agent_positions=next_agent_positions,
        agent_radii=state.agent_radii,
        team_ids=state.team_ids,
        class_ids=state.class_ids,
        movement_speeds=state.movement_speeds,
        observation_radii=state.observation_radii,
        basic_interaction_radii=state.basic_interaction_radii,
        ultimate_interaction_radii=state.ultimate_interaction_radii,
        active_mask=state.active_mask,
        alive_mask=state.alive_mask,
        current_health=state.current_health,
        max_health=state.max_health,
        ultimate_cooldowns=state.ultimate_cooldowns,
        slow_multipliers=state.slow_multipliers,
        slow_durations=state.slow_durations,
        stun_durations=state.stun_durations,
        anti_heal_multipliers=state.anti_heal_multipliers,
        anti_heal_durations=state.anti_heal_durations,
        damage_amplification_multipliers=state.damage_amplification_multipliers,
        damage_amplification_durations=state.damage_amplification_durations,
        blessing_of_freedom_durations=state.blessing_of_freedom_durations,
    )

    obs = _build_observation(next_state, config)

    rewards = Reward(rewards=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32))

    done_flags = DoneFlags(
        terminated=jnp.array(False),
        truncated=jnp.array(next_state.step_count >= config.max_steps),
    )

    action_mask = _build_action_mask(next_state, obs)

    info = Info()

    return (next_state, obs, rewards, done_flags, action_mask, info)
