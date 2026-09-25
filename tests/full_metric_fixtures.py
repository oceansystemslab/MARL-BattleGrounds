"""Provide shared game steps and metric lookups for full-metric tests.

The helpers drive public transitions and keep the game state paired with its
metric counters. Tests choose the setup and expected result. start_run builds a
Team Deathmatch game on the 20 x 12 fixture map with spawn pads at x = 1.5
(Team A) and x = 18.5 (Team B); its red_zone_depth keyword (default 0.0, the
rule off) sets the Red Zone depth, so depth 5 gives strips x <= 5 and x >= 15.
"""

from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config, neutral_action

from marl_battlegrounds.core.axis_mappings import global_slot_to_target_action
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import (
    initialize_scenario_state,
    reset,
    step,
)
from marl_battlegrounds.core.types import (
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.evaluation.episode_metrics import (
    MetricValues,
    PriorityTotals,
    initialize_priority,
    priority_values,
    update_priority,
)
from marl_battlegrounds.evaluation.full_metrics import (
    FullTotals,
    full_values,
    initialize_full,
    update_full,
)
from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
)

_step = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(step),
)
update_metrics = cast(
    Callable[[FullTotals, EnvConfig, EnvState, ActionMask, Info], FullTotals],
    jax.jit(update_full),
)
_final = cast(
    Callable[[FullTotals, EnvConfig, MetricValues], MetricValues], jax.jit(full_values)
)
_COLUMN_INDEX = {name: index for index, name in enumerate(FULL_METRIC_NAMES)}


class MetricRun(NamedTuple):
    config: EnvConfig
    state: EnvState
    mask: ActionMask
    full: FullTotals
    priority: PriorityTotals
    initial_step: Array
    outcome: Array


def start_run(
    *,
    team_sizes: tuple[int, int] = (3, 2),
    classes: tuple[tuple[int, int], ...] = (),
    arrange: Callable[[EnvState], EnvState] | None = None,
    red_zone_depth: float = 0.0,
) -> MetricRun:
    config = evaluation_env_config(
        team_sizes=team_sizes,
        task_mode=1,
        team_deathmatch_score_threshold=20,
        max_steps=16,
    )._replace(team_deathmatch_red_zone_depth=red_zone_depth)
    class_ids = config.agent_profile.class_ids
    for slot, class_id in classes:
        class_ids = class_ids.at[slot].set(class_id)
    config = config._replace(
        agent_profile=resolve_agent_profile(
            class_ids, jnp.asarray(team_sizes, jnp.int32)
        )
    )
    state, _, _, _ = reset(config, jax.random.key(0))
    positions = jnp.asarray(
        [(6.0, 1.5 + 2 * slot) for slot in range(5)]
        + [(9.0, 1.5 + 2 * slot) for slot in range(5)],
        jnp.float32,
    )
    state = state._replace(
        agent_positions=jnp.where(
            config.agent_profile.active_mask[:, None], positions, 0
        )
    )
    if arrange is not None:
        state = arrange(state)
    state, _, mask, _ = initialize_scenario_state(state, config)
    return MetricRun(
        config,
        state,
        mask,
        initialize_full(config, state),
        initialize_priority(),
        state.step_count,
        jnp.asarray(0, jnp.int32),
    )


def actions(*choices: tuple[int, int, bool]) -> Action:
    actions = neutral_action()
    for source, recipient, ultimate in choices:
        actions = actions._replace(
            select_target=actions.select_target.at[source].set(
                global_slot_to_target_action(source, recipient)
            ),
            use_ultimate=actions.use_ultimate.at[source].set(int(ultimate)),
        )
    return actions


def advance(run: MetricRun, actions: Action) -> tuple[MetricRun, Info]:
    successor, _, reward, _, mask, info = _step(
        run.config, run.state, run.mask, actions, jax.random.key(1)
    )
    return (
        run._replace(
            state=successor,
            mask=mask,
            full=update_metrics(run.full, run.config, run.state, run.mask, info),
            priority=update_priority(run.priority, reward, info),
            outcome=info.transition_facts.team_deathmatch_facts.outcome,
        ),
        info,
    )


def priority(run: MetricRun) -> MetricValues:
    return priority_values(
        run.config, run.state, run.initial_step, run.priority, run.outcome
    )


def values(run: MetricRun) -> MetricValues:
    return _final(run.full, run.config, priority(run))


def value(values: MetricValues, name: str) -> float:
    index = _COLUMN_INDEX[name]
    assert bool(values.valid[index]), name
    return float(values.values[index])


def assert_missing(values: MetricValues, name: str) -> None:
    assert not bool(values.valid[_COLUMN_INDEX[name]]), name
