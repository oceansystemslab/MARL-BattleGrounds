"""Accumulate compact priority metrics from Core rewards and transition facts.

The numerical trees cover one episode and support outer JAX batching. They
hold returns and observed deaths; final values also read authoritative Core
scores, elapsed steps and outcome. Availability is a separate boolean mask.
This module neither retains trajectories nor writes host reports.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import EnvConfig, EnvState, Info, Reward


class PriorityTotals(NamedTuple):
    """Constant-size running values for one episode.

    Attributes
    ----------
    agent_returns : Array
        float32 (10,) cumulative Core rewards in global slot order.
    team_deaths : Array
        int32 (2,) observed deaths in Team A, Team B order, including
        repeated lives. Authored initial scores do not count as observed deaths.

    The tree contains no host metadata or trajectory history. An outer batch
    operation may add environment axes.
    """

    agent_returns: Array
    team_deaths: Array


class MetricValues(NamedTuple):
    """Metric values and availability masks in the metric catalog's order.

    Attributes
    ----------
    values : Array
        float32 (..., M) scalar values for M named metrics.
    valid : Array
        bool (..., M); False means unavailable, not a measured zero.

    Leading axes belong to the caller's batch. Readers must apply valid before
    publishing values; a zero in an invalid position has no scientific meaning.
    """

    values: Array
    valid: Array


def initialize_priority() -> PriorityTotals:
    """Create empty running priority metrics for one episode.

    Returns
    -------
    PriorityTotals
        PriorityTotals with float32 zeros (10,) for agent returns and int32 zeros
        (2,) for observed team deaths.

    Takes no arguments, performs no host work and supports jit/vmap. No trajectory
    or completed report is retained; callers create a new tree on reset.
    """
    return PriorityTotals(jnp.zeros(10, jnp.float32), jnp.zeros(2, jnp.int32))


def update_priority(
    totals: PriorityTotals, reward: Reward, info: Info
) -> PriorityTotals:
    """Add one step's actual rewards and newly observed deaths.

    Parameters
    ----------
    totals : PriorityTotals
        PriorityTotals for one episode before this transition.
    reward : Reward
        Core Reward with ten slot-aligned values.
    info : Info
        Core Info from the same transition. Its death facts identify newly
        dead recipients, including repeated deaths across respawned lives.

    Returns
    -------
    PriorityTotals
        New PriorityTotals with the same shapes and dtypes; inputs are unchanged.

    This pure numerical helper supports jit and outer vmap. The caller owns
    transition admission; supply zero reward/facts when no transition occurred.
    """
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

    Parameters
    ----------
    config : EnvConfig
        Exact scalar episode config, including ten active-slot flags.
    state : EnvState
        Latest scalar Core state supplying scores and elapsed steps.
    initial_step_count : Array
        Scalar starting step, so authored starts count only
        transitions in the recorded episode.
    totals : PriorityTotals
        Running PriorityTotals from those same transitions.
    outcome : Array
        Scalar Core outcome code: 0 before an outcome, 1 Team A win,
        2 Team B win, or 3 draw.

    Returns
    -------
    MetricValues
        MetricValues with float32 values and bool validity, each shape (26,),
        ordered by PRIORITY_METRIC_NAMES. Inactive actor returns and unresolved
        outcome flags are unavailable. Other valid neutral values can be zero.

    This pure numerical read supports jit and outer vmap. It does not reset
    counters, validate configs, write tables or decide when an episode completes.
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
