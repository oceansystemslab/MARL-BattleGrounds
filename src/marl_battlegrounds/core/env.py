"""Functional reset and step entry points for the core JAX simulator."""

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
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
    default_agent_positions = jax.random.uniform(
        deterministic_key,
        shape=(MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        dtype=jnp.float32,
        minval=0,
        maxval=max_val,
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

    base_self_visibility = jnp.identity(MAX_AGENTS_PER_TEAM, dtype=bool)
    alive_active_mask_bc = jnp.logical_and(state.active_mask, state.alive_mask)[:, None]

    map_obstacle_features = jnp.broadcast_to(
        config.obstacles[None, :, :],
        (MAX_AGENT_SLOTS, MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
    )

    obs = Observation(
        self_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, SELF_FEATURES), dtype=jnp.float32
        ),
        ally_unit_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES),
            dtype=jnp.float32,
        ),
        enemy_unit_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES),
            dtype=jnp.float32,
        ),
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

    ones_column_vector = jnp.ones((MAX_AGENT_SLOTS, 1), dtype=bool)
    zeros_column_vector = jnp.zeros((MAX_AGENT_SLOTS, 1), dtype=bool)

    target_mask = jnp.concat(
        (
            ones_column_vector,
            obs.ally_targetability_mask,
            obs.enemy_targetability_mask,
        ),
        axis=1,
    )

    ult_mask = jnp.concat((ones_column_vector, zeros_column_vector), axis=1)

    action_mask = ActionMask(
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

    info = Info()

    return (state, obs, action_mask, info)


def step(
    config: EnvConfig, state: EnvState, joint_action: Action, key: Array
) -> tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]:
    """Advance the placeholder transition while preserving M4 contracts."""

    next_state = EnvState(
        step_count=state.step_count + 1,
        agent_positions=state.agent_positions,
        agent_radii=state.agent_radii,
        team_ids=state.team_ids,
        active_mask=state.active_mask,
        alive_mask=state.alive_mask,
    )

    base_self_visibility = jnp.identity(MAX_AGENTS_PER_TEAM, dtype=bool)
    alive_active_mask_bc = jnp.logical_and(
        next_state.active_mask, next_state.alive_mask
    )[:, None]

    map_obstacle_features = jnp.broadcast_to(
        config.obstacles[None, :, :],
        (MAX_AGENT_SLOTS, MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
    )

    obs = Observation(
        self_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, SELF_FEATURES), dtype=jnp.float32
        ),
        ally_unit_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES),
            dtype=jnp.float32,
        ),
        enemy_unit_features=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES),
            dtype=jnp.float32,
        ),
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

    rewards = Reward(rewards=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32))

    done_flags = DoneFlags(
        terminated=jnp.array(False),
        truncated=jnp.array(next_state.step_count >= config.max_steps),
    )

    ones_column_vector = jnp.ones(shape=(MAX_AGENT_SLOTS, 1), dtype=bool)
    zeros_column_vector = jnp.zeros(shape=(MAX_AGENT_SLOTS, 1), dtype=bool)

    target_mask = jnp.concat(
        (ones_column_vector, obs.ally_targetability_mask, obs.enemy_targetability_mask),
        axis=1,
    )

    ult_mask = jnp.concat((ones_column_vector, zeros_column_vector), axis=1)

    action_mask = ActionMask(
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

    info = Info()

    return (next_state, obs, rewards, done_flags, action_mask, info)
