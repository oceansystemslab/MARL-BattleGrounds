"""Functional environment entry points for the core JAX simulator spine."""

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    NUM_MOVE_ACTIONS,
    NUM_OBSERVATION_FEATURES,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
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
    """Create the initial core state, observations, masks, and info."""

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

    state = EnvState(
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32
        ),
        team_ids=jnp.concat([team_0_ids, team_1_ids]),
        active_mask=initial_mask,
        alive_mask=initial_mask,
    )

    obs = Observation(
        observation_vectors=jnp.zeros(
            shape=(MAX_AGENT_SLOTS, NUM_OBSERVATION_FEATURES), dtype=jnp.float32
        )
    )

    # Reshape to broadcast across rows.
    mask = state.active_mask[:, None]

    action_mask = ActionMask(
        move=jnp.logical_and(
            jnp.ones(shape=(MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS), dtype=bool), mask
        ),
        target=jnp.logical_and(
            jnp.ones(shape=(MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS), dtype=bool), mask
        ),
        use_ultimate=jnp.logical_and(
            jnp.ones(
                shape=(MAX_AGENT_SLOTS, NUM_ULTIMATE_ACTIONS),
                dtype=bool,
            ),
            mask,
        ),
    )

    info = Info()

    return (state, obs, action_mask, info)


def step(
    config: EnvConfig, state: EnvState, joint_action: Action, key: Array
) -> tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]:
    """Advance the placeholder core transition by one timestep."""

    # Next state
    next_state_step_count = state.step_count + 1
    next_agent_positions = state.agent_positions
    next_active_mask = state.active_mask
    next_alive_mask = state.alive_mask

    next_state = EnvState(
        next_state_step_count,
        next_agent_positions,
        team_ids=state.team_ids,
        active_mask=next_active_mask,
        alive_mask=next_alive_mask,
    )

    observations = Observation(
        observation_vectors=jnp.zeros(
            (MAX_AGENT_SLOTS, NUM_OBSERVATION_FEATURES), dtype=jnp.float32
        )
    )

    rewards = Reward(rewards=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32))

    done_flags = DoneFlags(
        terminated=jnp.array(False),
        truncated=jnp.array(next_state.step_count >= config.max_steps),
    )

    next_active_mask_bc = next_state.active_mask.reshape(-1, 1)

    action_mask = ActionMask(
        move=jnp.logical_and(
            jnp.ones((MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS), dtype=bool),
            next_active_mask_bc,
        ),
        target=jnp.logical_and(
            jnp.ones((MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS), dtype=bool),
            next_active_mask_bc,
        ),
        use_ultimate=jnp.logical_and(
            jnp.ones((MAX_AGENT_SLOTS, NUM_ULTIMATE_ACTIONS), dtype=bool),
            next_active_mask_bc,
        ),
    )

    info = Info()

    return (next_state, observations, rewards, done_flags, action_mask, info)
