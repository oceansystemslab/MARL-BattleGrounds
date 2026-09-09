"""Small JAX episode accumulators over authoritative Core outputs."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import EnvConfig, EnvState, Info, Reward


class PriorityTotals(NamedTuple):
    """Sufficient statistics not already retained in the Core state."""

    agent_returns: Array
    team_deaths: Array


class MetricValues(NamedTuple):
    """Scalar values and availability in the order declared by the catalog."""

    values: Array
    valid: Array


def initialize_priority() -> PriorityTotals:
    """Create constant-size counters; no trajectory or host report is retained."""
    return PriorityTotals(jnp.zeros(10, jnp.float32), jnp.zeros(2, jnp.int32))


def update_priority(
    totals: PriorityTotals, reward: Reward, info: Info
) -> PriorityTotals:
    """Accumulate actual returns and observed deaths, including repeated lives."""
    deaths = info.transition_facts.death_facts.is_newly_dead_by_recipient
    return PriorityTotals(
        totals.agent_returns + reward.rewards,
        totals.team_deaths + deaths.reshape(2, 5).sum(axis=1, dtype=jnp.int32),
    )


def priority_values(
    config: EnvConfig,
    state: EnvState,
    initial_step_count: Array,
    totals: PriorityTotals,
    outcome: Array,
) -> MetricValues:
    """Read Core scores/outcomes and de-broadcast its canonical team rewards.

    A score may include scenario initialization; observed kills do not. Team
    reward is repeated per active teammate by Core, so summing those copies
    would incorrectly make return depend on roster size.
    """
    active = config.agent_profile.active_mask
    first_active = jnp.argmax(active.reshape(2, 5), axis=1)
    team_returns = totals.agent_returns.reshape(2, 5)[jnp.arange(2), first_active]
    scores = state.team_deathmatch_scores
    outcomes = jnp.stack(
        (
            outcome == 1,
            outcome == 3,
            outcome == 2,
            outcome == 2,
            outcome == 3,
            outcome == 1,
        )
    )
    values = jnp.concatenate(
        (
            (state.step_count - initial_step_count)[None],
            outcomes,
            team_returns,
            totals.agent_returns,
            scores,
            (scores[0] - scores[1])[None],
            totals.team_deaths[::-1],
            totals.team_deaths,
        )
    ).astype(jnp.float32)
    valid = jnp.concatenate(
        (
            jnp.ones(1, bool),
            jnp.repeat(outcome != 0, 6),
            jnp.ones(2, bool),
            active,
            jnp.ones(7, bool),
        )
    )
    return MetricValues(values, valid)
