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
    ActionMask,
    EnvConfig,
    EnvState,
    Info,
    Observation,
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
