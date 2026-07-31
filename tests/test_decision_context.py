"""Decision-context observation contract tests."""
# pyright: reportPrivateUsage=false

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

import marl_battlegrounds.core.types as core_types
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import _build_observation_and_action_mask, reset, step
from marl_battlegrounds.core.types import (
    CLASS_NEUTRAL,
    CONTEXT_FEATURES,
    MAX_AGENT_SLOTS,
    MAX_OBSTACLE_SLOTS,
    MOVE_STAY,
    OBSTACLE_FEATURES,
    Action,
    ActionMask,
    EnvConfig,
    EnvState,
    Info,
    Observation,
)


def _config(
    team_sizes: tuple[int, int] = (3, 2),
    *,
    max_steps: int = 37,
    map_width: float = 24.0,
    map_height: float = 16.0,
) -> EnvConfig:
    """Build a deterministic config with an explicitly asymmetric roster."""
    profile = resolve_agent_profile(
        jnp.full((MAX_AGENT_SLOTS,), CLASS_NEUTRAL, dtype=jnp.int32),
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )
    positions = jnp.asarray(
        (
            (0.5, 0.5),
            (2.5, 0.5),
            (4.5, 0.5),
            (6.5, 0.5),
            (8.5, 0.5),
            (0.5, 7.5),
            (2.5, 7.5),
            (4.5, 7.5),
            (6.5, 7.5),
            (8.5, 7.5),
        ),
        dtype=jnp.float32,
    )
    return EnvConfig(
        max_steps=max_steps,
        map_width=map_width,
        map_height=map_height,
        obstacles=jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32),
        agent_profile=profile,
        initial_agent_positions=jnp.where(profile.active_mask[:, None], positions, 0.0),
        ordinary_movement_distance_scale=1.0,
    )


def _stay_action() -> Action:
    """Return an effect-inert action for every fixed agent slot."""
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _assert_context_row(
    context_features: Array,
    slot: int,
    *,
    expected_timestep: int,
    expected_horizon: int,
    expected_map_width: float,
    expected_map_height: float,
    expected_ally_team_size: int,
    expected_enemy_team_size: int,
) -> None:
    """Assert one active row's populated fields and reserved zero suffix."""
    expected_populated_features = jnp.asarray(
        (
            expected_timestep,
            expected_horizon,
            expected_map_width,
            expected_map_height,
            expected_ally_team_size,
            expected_enemy_team_size,
        ),
        dtype=jnp.float32,
    )
    assert bool(
        jnp.array_equal(context_features[slot, :6], expected_populated_features)
    )
    assert bool(jnp.all(context_features[slot, 6:] == 0.0))


def test_context_feature_indices_are_contiguous_and_complete() -> None:
    context_feature_indices = sorted(
        value
        for name, value in vars(core_types).items()
        if name.startswith("CONTEXT_FEATURE_") and isinstance(value, int)
    )

    assert CONTEXT_FEATURES == 19
    assert context_feature_indices == list(range(CONTEXT_FEATURES))


def test_reset_exposes_raw_actor_relative_asymmetric_context() -> None:
    config = _config(team_sizes=(3, 2))
    _, observation, _, _ = reset(config, jax.random.key(0))
    context_features = observation.context_features

    assert context_features.shape == (MAX_AGENT_SLOTS, CONTEXT_FEATURES)
    assert context_features.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(context_features)))
    for team_a_slot in (0, 1, 2):
        _assert_context_row(
            context_features,
            team_a_slot,
            expected_timestep=0,
            expected_horizon=37,
            expected_map_width=24.0,
            expected_map_height=16.0,
            expected_ally_team_size=3,
            expected_enemy_team_size=2,
        )
    for team_b_slot in (5, 6):
        _assert_context_row(
            context_features,
            team_b_slot,
            expected_timestep=0,
            expected_horizon=37,
            expected_map_width=24.0,
            expected_map_height=16.0,
            expected_ally_team_size=2,
            expected_enemy_team_size=3,
        )

    inactive_slots = jnp.asarray((3, 4, 7, 8, 9), dtype=jnp.int32)
    assert bool(jnp.all(context_features[inactive_slots] == 0.0))


def test_active_dead_actor_retains_decision_context() -> None:
    config = _config(team_sizes=(3, 2))
    state, _, _, _ = reset(config, jax.random.key(1))
    state_with_dead_actor = state._replace(
        alive_mask=state.alive_mask.at[1].set(False),
        current_health=state.current_health.at[1].set(0.0),
    )

    observation, _ = _build_observation_and_action_mask(state_with_dead_actor, config)

    _assert_context_row(
        observation.context_features,
        1,
        expected_timestep=0,
        expected_horizon=37,
        expected_map_width=24.0,
        expected_map_height=16.0,
        expected_ally_team_size=3,
        expected_enemy_team_size=2,
    )


def test_context_distinguishes_policy_relevant_static_configurations() -> None:
    baseline_config = _config(team_sizes=(3, 2))
    alternate_config = _config(
        team_sizes=(3, 4), max_steps=91, map_width=30.0, map_height=18.0
    )

    _, baseline_observation, _, _ = reset(baseline_config, jax.random.key(2))
    _, alternate_observation, _, _ = reset(alternate_config, jax.random.key(2))

    assert bool(
        jnp.array_equal(
            baseline_observation.context_features[0, :6],
            jnp.asarray((0, 37, 24, 16, 3, 2), dtype=jnp.float32),
        )
    )
    assert bool(
        jnp.array_equal(
            alternate_observation.context_features[0, :6],
            jnp.asarray((0, 91, 30, 18, 3, 4), dtype=jnp.float32),
        )
    )
    assert not bool(
        jnp.array_equal(
            baseline_observation.context_features[0],
            alternate_observation.context_features[0],
        )
    )


def test_step_exposes_successor_timestep_without_normalization_or_clipping() -> None:
    config = _config(team_sizes=(1, 1), max_steps=2)
    state, observation, action_mask, _ = reset(config, jax.random.key(3))
    action = _stay_action()

    assert (
        observation.context_features[0, core_types.CONTEXT_FEATURE_CURRENT_TIMESTEP]
        == 0
    )
    observed_timesteps: list[float] = []
    observed_truncations: list[bool] = []
    for step_index in range(1, 4):
        state, observation, _, done_flags, action_mask, _ = step(
            config,
            state,
            action_mask,
            action,
            jax.random.key(step_index + 3),
        )
        observed_timesteps.append(
            float(
                observation.context_features[
                    0, core_types.CONTEXT_FEATURE_CURRENT_TIMESTEP
                ]
            )
        )
        observed_truncations.append(bool(done_flags.truncated))

    assert observed_timesteps == [1.0, 2.0, 3.0]
    assert observed_truncations == [False, True, True]
    assert state.step_count == 3


def test_context_is_stable_under_jit_and_scanned_rollout() -> None:
    config = _config(team_sizes=(2, 4), max_steps=5)
    initial_state, initial_observation, initial_mask, _ = reset(
        config, jax.random.key(7)
    )
    compiled_reset = cast(
        tuple[EnvState, Observation, ActionMask, Info],
        jax.jit(reset)(config, jax.random.key(7)),
    )
    assert bool(
        jnp.array_equal(
            compiled_reset[1].context_features, initial_observation.context_features
        )
    )

    action = _stay_action()

    def _scan_step(
        carry: tuple[EnvState, ActionMask], key: Array
    ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array]]:
        current_state, current_mask = carry
        next_state, observation, _, done_flags, next_mask, _ = step(
            config, current_state, current_mask, action, key
        )
        return (next_state, next_mask), (
            observation.context_features,
            done_flags.truncated,
        )

    def _rollout(
        state: EnvState, action_mask: ActionMask, keys: Array
    ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array]]:
        return jax.lax.scan(_scan_step, (state, action_mask), keys)

    keys = jax.random.split(jax.random.key(8), 6)
    eager_rollout = _rollout(initial_state, initial_mask, keys)
    compiled_rollout = cast(
        tuple[tuple[EnvState, ActionMask], tuple[Array, Array]],
        jax.jit(_rollout)(initial_state, initial_mask, keys),
    )
    eager_context_history, eager_truncation_history = eager_rollout[1]
    compiled_context_history, compiled_truncation_history = compiled_rollout[1]

    assert eager_context_history.shape == (6, MAX_AGENT_SLOTS, CONTEXT_FEATURES)
    assert bool(jnp.array_equal(eager_context_history, compiled_context_history))
    assert bool(jnp.array_equal(eager_truncation_history, compiled_truncation_history))
    assert bool(
        jnp.array_equal(
            eager_context_history[:, 0, core_types.CONTEXT_FEATURE_CURRENT_TIMESTEP],
            jnp.arange(1, 7, dtype=jnp.float32),
        )
    )
    assert bool(
        jnp.all(
            eager_context_history[:, 0, core_types.CONTEXT_FEATURE_EPISODE_HORIZON]
            == 5.0
        )
    )
    assert bool(
        jnp.array_equal(
            eager_truncation_history,
            jnp.asarray((False, False, False, False, True, True)),
        )
    )
