"""Share public action, tracking and reset calls between training examples.

These private numerical helpers own no learner or schedule. Both the ordinary
distribution example and the training collector pass explicit Systems, weights,
keys and declarations. The caller decides when a reset belongs in its loop.
"""

# These helpers are shared only by the two owned collection paths.
# pyright: reportUnusedFunction=false, reportPrivateUsage=false
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import Action, EnvConfig
from marl_battlegrounds.environment import (
    Environment,
    EnvironmentState,
    _checked_increment,
)
from marl_battlegrounds.episode_tracking import (
    EpisodeTrackingState,
    StepResult,
    track_episode_step,
)
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemState,
    apply_systems,
)
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.training.distributions import (
    SampledTrainingConfigs,
    sample_training_configs,
    training_keys,
)

type Tree = Any


def _apply_and_track(
    env: Environment,
    observations: Observations,
    state: EnvironmentState,
    memory: SystemState,
    tracking: EpisodeTrackingState,
    *,
    actor: System,
    opponent: System,
    variables_a: Tree,
    variables_b: Tree,
    action_keys: Array,
    step_keys: Array,
    source_indices: Array,
    source_class_ids: Array,
) -> tuple[Action, SystemState, tuple[Any, Any], EpisodeTrackingState, StepResult]:
    """Choose once, step once and certify that exact producing transition.

    observations/state/memory/tracking must describe one native B-lane decision.
    The two Systems are stable descriptors; variables and B lane keys are
    dynamic. Source indices (B,) and class rows (B,10) describe those games.
    Return submitted actions, successor System memory, separate same-call
    learning trees, successor tracker and the enriched five-part step result.
    There is no reset, host transfer, shaping, storage or learner update here.
    Public System/environment/tracker shape errors propagate unchanged. The
    caller must reject lifecycle errors before calling this helper.
    """
    actions, memory, learning = apply_systems(
        actor,
        opponent,
        memory,
        observations,
        state,
        action_keys,
        variables_a=variables_a,
        variables_b=variables_b,
    )
    tracking, result = track_episode_step(
        tracking,
        state,
        env.step(step_keys, state, actions),
        source_indices=source_indices,
        source_class_ids=source_class_ids,
    )
    return actions, memory, learning, tracking, result


def _reset_finished(
    env: Environment,
    observations: Observations,
    state: EnvironmentState,
    source_indices: Array,
    source_class_ids: Array,
    *,
    source_configs: EnvConfig,
    root_key: Array,
    eligible_maps: Array,
    team_size: Array,
    score_threshold: Array | None = None,
    roster_class_ids: Array | None = None,
    sampled: SampledTrainingConfigs | None = None,
    reset_keys: Array | None = None,
) -> tuple[Observations, EnvironmentState, Array, Array]:
    """Reset finished lanes and replace only their source/roster declarations.

    Inputs describe native B-lane games and the immutable source bank. Controls
    have the sampler's bool (42,) and scalar int32 shapes. The root is Threefry.
    roster_class_ids is optional int32 (10,) for the sampler's explicit roster;
    None or ten zeros keeps its existing team-size route. score_threshold is
    scalar int32, or None for K20, and must name a prepared
    source block. Only reset lanes adopt it; continuing games retain their K.
    Optional sampled choices and lane reset keys supply an exact benchmark
    reference; ordinary callers omit both. Return observations, state, source
    indices (B,) and class rows (B,10). Continuing lanes stay unchanged. No reset
    lanes means no sampling or reset work. Memory, history and stage labels are
    caller-owned. Preserve environment lifecycle failures for the caller's
    post-reset guard; never initialize or apply a System here.
    """

    def reset(_operand: None) -> tuple[Observations, EnvironmentState, Array, Array]:
        """Draw next-generation conditions and merge selected lanes only."""
        generations, _ = _checked_increment(state.reset_generation, 1)
        chosen = (
            sampled
            if sampled is not None
            else sample_training_configs(
                source_configs,
                root_key,
                generations,
                eligible_maps=eligible_maps,
                team_size=team_size,
                score_threshold=score_threshold,
                roster_class_ids=roster_class_ids,
            )
        )
        keys = (
            reset_keys
            if reset_keys is not None
            else training_keys(root_key, generations, stream="reset")
        )
        mask = state.done.done
        next_observations, next_state = env.reset_done(keys, state, chosen.config)
        return (
            next_observations,
            next_state,
            jnp.where(mask, chosen.source_indices, source_indices),
            jnp.where(mask[:, None], chosen.source_class_ids, source_class_ids),
        )

    def retain(_operand: None) -> tuple[Observations, EnvironmentState, Array, Array]:
        """Return all original values without sampling when no game ended."""
        return observations, state, source_indices, source_class_ids

    return cast(
        tuple[Observations, EnvironmentState, Array, Array],
        jax.lax.cond(
            jnp.any(state.done.done),
            reset,
            retain,
            None,
        ),
    )
