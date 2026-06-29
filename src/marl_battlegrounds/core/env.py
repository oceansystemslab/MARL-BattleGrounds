"""Functional reset and step entry points for the core JAX simulator."""

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.geometry import project_movement_with_geometry
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_TEAM_ID,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    CONTEXT_FEATURES,
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBJECTIVE_SLOTS,
    MAX_OBSTACLE_SLOTS,
    NUM_MOVE_ACTIONS,
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


def _build_observation(state: EnvState, config: EnvConfig) -> Observation:
    """Build the current observation contract from one slot-aligned state."""
    self_features = _build_self_features(state)
    ally_features = _build_ally_features(self_features)
    enemy_features = _build_enemy_features(self_features)

    base_self_visibility = jnp.identity(MAX_AGENTS_PER_TEAM, dtype=bool)
    alive_active_mask_bc = jnp.logical_and(state.active_mask, state.alive_mask)[:, None]

    map_obstacle_features = jnp.broadcast_to(
        config.obstacles[None, :, :],
        (MAX_AGENT_SLOTS, MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
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
        ally_visibility_mask=jnp.logical_and(
            jnp.vstack((base_self_visibility, base_self_visibility)),
            alive_active_mask_bc,
        ),
        enemy_visibility_mask=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM), dtype=bool
        ),
        ally_targetability_mask=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM), dtype=bool
        ),
        enemy_targetability_mask=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM), dtype=bool
        ),
    )


def _build_action_mask(state: EnvState, observation: Observation) -> ActionMask:
    """Build action masks from observer liveness and observation targetability."""
    ones_column_vector = jnp.ones(shape=(MAX_AGENT_SLOTS, 1), dtype=bool)
    zeros_column_vector = jnp.zeros(shape=(MAX_AGENT_SLOTS, 1), dtype=bool)

    target_mask = jnp.concat(
        (
            ones_column_vector,
            observation.ally_targetability_mask,
            observation.enemy_targetability_mask,
        ),
        axis=1,
    )

    ult_mask = jnp.concat((ones_column_vector, zeros_column_vector), axis=1)

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


def _build_intended_movement_deltas(joint_action: Action, config: EnvConfig) -> Array:
    """Convert per-slot movement action IDs into scaled displacement vectors."""
    intended_movement_deltas_unscaled = _JOINT_ACTION_MOVE_TO_DISPLACEMENT_LOOKUP_TABLE[
        joint_action.move
    ]

    movement_speed_scaling_factor = jnp.asarray(
        config.movement_speed, dtype=jnp.float32
    )

    intended_movement_deltas = (
        movement_speed_scaling_factor * intended_movement_deltas_unscaled
    )

    return intended_movement_deltas


def _build_self_features(state: EnvState) -> Array:
    """Build slot-aligned self rows from the shared base agent schema."""
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

    return self_features


def _build_ally_features(self_features: Array) -> Array:
    """Return Checkpoint 2 placeholder ally candidate rows."""
    del self_features
    return jnp.zeros(
        (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES), dtype=jnp.float32
    )


def _build_enemy_features(self_features: Array) -> Array:
    """Return Checkpoint 2 placeholder enemy candidate rows."""
    del self_features
    return jnp.zeros(
        (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES), dtype=jnp.float32
    )


# Public ---


def reset(
    config: EnvConfig, key: Array
) -> tuple[EnvState, Observation, ActionMask, Info]:
    """Create the initial Milestone 4 contract state and placeholders."""

    # Reset keeps all arrays at MAX_AGENT_SLOTS length. Smaller tasks use
    # active_mask to distinguish real agents from padded slots.
    # Ordinary reset starts all active agents alive. Scenario loaders may later
    # create active-but-dead agents from curated states.
    # TODO(M4+): Use key when reset begins sampling spawn positions or randomized
    # layouts. Deterministic dummy reset may accept the key without consuming it.
    # TODO(Scenario): Keep curated scenario starts out of ordinary reset. A future
    # scenario loader should validate and return EnvState values that reuse the
    # same transition, observation, and mask machinery.

    team_0_ids = jnp.zeros((MAX_AGENTS_PER_TEAM,), dtype=jnp.int32)
    team_1_ids = jnp.ones((MAX_AGENTS_PER_TEAM,), dtype=jnp.int32)

    indices = jnp.arange(MAX_AGENT_SLOTS)

    team_0_active = indices < config.team_size

    team_1_active = (indices >= MAX_AGENTS_PER_TEAM) & (
        indices < MAX_AGENTS_PER_TEAM + config.team_size
    )

    initial_mask = jnp.logical_or(team_0_active, team_1_active)

    deterministic_key = jax.random.key(42)
    max_val = jnp.min(jnp.array([config.map_width, config.map_height]))
    # Reset emits geometry-valid placeholder centers so MOVE_STAY does not
    # trigger corrective projection before scenario loading exists.
    default_agent_positions = jax.random.uniform(
        deterministic_key,
        shape=(MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        dtype=jnp.float32,
        minval=0 + config.default_agent_radius,
        maxval=max_val - config.default_agent_radius,
    )

    state = EnvState(
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=default_agent_positions,
        agent_radii=jnp.full(
            shape=(MAX_AGENT_SLOTS,),
            fill_value=config.default_agent_radius,
            dtype=jnp.float32,
        ),
        team_ids=jnp.concat([team_0_ids, team_1_ids]),
        active_mask=initial_mask,
        alive_mask=initial_mask,
    )

    obs = _build_observation(state, config)

    action_mask = _build_action_mask(state, obs)

    info = Info()

    return (state, obs, action_mask, info)


def step(
    config: EnvConfig, state: EnvState, joint_action: Action, key: Array
) -> tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]:
    """Advance movement while preserving current Milestone 4 placeholders."""

    intended_movement_deltas = _build_intended_movement_deltas(joint_action, config)

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
        active_mask=state.active_mask,
        alive_mask=state.alive_mask,
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
