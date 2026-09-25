"""Check the public context features available when an actor chooses an action.

Context column 19 is the Team Deathmatch Red Zone depth. Every configured row
shows it after reset, after step and at an authored start, dead rows included;
unused rows and neutral mode show 0. Two configs that differ only in depth give
observations that differ only in that column. Changing the depth reuses one
compiled reset, and a vmapped reset gives each game its own depth.
"""
# pyright: reportPrivateUsage=false

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.test_team_deathmatch_semantics import _scenario, _task_config

import marl_battlegrounds.core.types as core_types
from marl_battlegrounds.core.config import resolve_agent_profile, validate_env_config
from marl_battlegrounds.core.env import _build_observation_and_action_mask, reset, step
from marl_battlegrounds.core.types import (
    CLASS_NEUTRAL,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH,
    CONTEXT_FEATURES,
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
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
        task_mode=0,
        team_deathmatch_score_threshold=0,
        team_deathmatch_red_zone_depth=0.0,
        max_steps=max_steps,
        map_width=map_width,
        map_height=map_height,
        obstacles=jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32),
        agent_profile=profile,
        ordinary_movement_distance_scale=1.0,
        team_spawn_pad_positions=positions.reshape(
            (2, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS)
        ),
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray((5, 5), dtype=jnp.int32),
    )


def _stay_action() -> Action:
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

    assert CONTEXT_FEATURES == 20
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
    for step_index in range(1, 3):
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

    assert observed_timesteps == [1.0, 2.0]
    assert observed_truncations == [False, True]
    assert state.step_count == 2


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

    keys = jax.random.split(jax.random.key(8), 5)
    eager_rollout = _rollout(initial_state, initial_mask, keys)
    compiled_rollout = cast(
        tuple[tuple[EnvState, ActionMask], tuple[Array, Array]],
        jax.jit(_rollout)(initial_state, initial_mask, keys),
    )
    eager_context_history, eager_truncation_history = eager_rollout[1]
    compiled_context_history, compiled_truncation_history = compiled_rollout[1]

    assert eager_context_history.shape == (5, MAX_AGENT_SLOTS, CONTEXT_FEATURES)
    assert bool(jnp.array_equal(eager_context_history, compiled_context_history))
    assert bool(jnp.array_equal(eager_truncation_history, compiled_truncation_history))
    assert bool(
        jnp.array_equal(
            eager_context_history[:, 0, core_types.CONTEXT_FEATURE_CURRENT_TIMESTEP],
            jnp.arange(1, 6, dtype=jnp.float32),
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
            jnp.asarray((False, False, False, False, True)),
        )
    )


type _ResetResult = tuple[EnvState, Observation, ActionMask, Info]
# One compiled reset serves every Team Deathmatch config below: they share
# shapes and dtypes, so changing values does not compile again.
_compiled_reset = cast(Callable[[EnvConfig, Array], _ResetResult], jax.jit(reset))


def _red_zone_config(depth: float) -> EnvConfig:
    # Width-12 Team Deathmatch with Team A slots 0-1 and Team B slot 5 configured.
    return _task_config(team_sizes=(2, 1), red_zone_depth=depth)


def _configured_rows(config: EnvConfig) -> np.ndarray:
    return np.asarray(config.agent_profile.active_mask)


def _assert_only_red_zone_column_differs(
    observation: Observation,
    zero_depth_observation: Observation,
    *,
    depth: float,
    configured_rows: np.ndarray,
) -> None:
    for leaf, zero_depth_leaf in zip(
        jax.tree.leaves(observation._replace(context_features=jnp.zeros(()))),
        jax.tree.leaves(
            zero_depth_observation._replace(context_features=jnp.zeros(()))
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(zero_depth_leaf))
    context = np.asarray(observation.context_features)
    zero_depth_context = np.asarray(zero_depth_observation.context_features)
    other_columns = np.arange(CONTEXT_FEATURES) != CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH
    np.testing.assert_array_equal(
        context[:, other_columns], zero_depth_context[:, other_columns]
    )
    np.testing.assert_array_equal(
        context[:, CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH],
        np.where(configured_rows, np.float32(depth), np.float32(0.0)),
    )
    np.testing.assert_array_equal(
        zero_depth_context[:, CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH], 0.0
    )
    assert bool(np.all(context[~configured_rows] == 0.0))


def test_red_zone_depth_column_follows_reset_step_and_an_authored_dead_start() -> None:
    config = _red_zone_config(5.0)
    zero_depth_config = _red_zone_config(0.0)
    configured_rows = _configured_rows(config)
    assert tuple(np.flatnonzero(configured_rows)) == (0, 1, 5)
    key = jax.random.key(4)

    state, observation, action_mask, _ = _compiled_reset(config, key)
    zero_state, zero_observation, zero_action_mask, _ = _compiled_reset(
        zero_depth_config, key
    )
    _assert_only_red_zone_column_differs(
        observation, zero_observation, depth=5.0, configured_rows=configured_rows
    )

    step_key = jax.random.key(5)
    _, next_observation, _, _, _, _ = step(
        config, state, action_mask, _stay_action(), step_key
    )
    _, zero_next_observation, _, _, _, _ = step(
        zero_depth_config, zero_state, zero_action_mask, _stay_action(), step_key
    )
    assert (
        float(
            next_observation.context_features[
                0, core_types.CONTEXT_FEATURE_CURRENT_TIMESTEP
            ]
        )
        == 1.0
    )
    _assert_only_red_zone_column_differs(
        next_observation,
        zero_next_observation,
        depth=5.0,
        configured_rows=configured_rows,
    )

    # Slot 1 is configured but dead at the authored start; it keeps the depth.
    authored_state, authored_observation, _, _ = _scenario(config, dead_slots=(1,))
    _, zero_authored_observation, _, _ = _scenario(zero_depth_config, dead_slots=(1,))
    assert not bool(authored_state.alive_mask[1])
    _assert_only_red_zone_column_differs(
        authored_observation,
        zero_authored_observation,
        depth=5.0,
        configured_rows=configured_rows,
    )


def _with_map_height(config: EnvConfig) -> EnvConfig:
    return config._replace(map_height=17.5)


def _with_swapped_banks(config: EnvConfig) -> EnvConfig:
    return config._replace(
        team_spawn_pad_positions=config.team_spawn_pad_positions[::-1]
    )


def _unchanged(config: EnvConfig) -> EnvConfig:
    return config


@pytest.mark.parametrize(
    ("depth", "change"),
    (
        (12.0, _unchanged),
        (5.0, _with_map_height),
        (5.0, _with_swapped_banks),
    ),
    ids=("depth-equal-to-width", "custom-map-height", "swapped-banks"),
)
def test_red_zone_depth_column_holds_on_edge_and_custom_layouts(
    depth: float, change: Callable[[EnvConfig], EnvConfig]
) -> None:
    config = change(_red_zone_config(depth))
    zero_depth_config = change(_red_zone_config(0.0))
    validate_env_config(config)
    key = jax.random.key(6)
    _, observation, _, _ = _compiled_reset(config, key)
    _, zero_observation, _, _ = _compiled_reset(zero_depth_config, key)
    assert float(observation.context_features[0, CONTEXT_FEATURE_MAP_HEIGHT]) == (
        config.map_height
    )
    _assert_only_red_zone_column_differs(
        observation,
        zero_observation,
        depth=depth,
        configured_rows=_configured_rows(config),
    )


def test_neutral_context_keeps_red_zone_depth_column_zero() -> None:
    _, observation, _, _ = reset(_config(team_sizes=(3, 2)), jax.random.key(0))
    assert bool(
        jnp.all(
            observation.context_features[:, CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH] == 0
        )
    )


def test_changing_red_zone_depth_reuses_one_compiled_reset() -> None:
    trace_count = 0

    def counted_reset(config: EnvConfig, key: Array) -> _ResetResult:
        nonlocal trace_count
        trace_count += 1
        return reset(config, key)

    compiled = cast(Callable[[EnvConfig, Array], _ResetResult], jax.jit(counted_reset))
    for depth in (5.0, 6.0):
        _, observation, _, _ = compiled(_red_zone_config(depth), jax.random.key(7))
        np.testing.assert_array_equal(
            observation.context_features[:, CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH],
            np.where(_configured_rows(_red_zone_config(depth)), depth, 0.0),
        )
    assert trace_count == 1


def test_vmapped_reset_gives_each_game_its_own_red_zone_depth() -> None:
    depths = (0.0, 5.0)
    configs = tuple(_red_zone_config(depth) for depth in depths)

    def stack(*leaves: object) -> Array:
        return jnp.stack(tuple(jnp.asarray(leaf) for leaf in leaves))

    stacked = cast(EnvConfig, jax.tree.map(stack, *configs))
    batched_reset = cast(
        Callable[[EnvConfig, Array], _ResetResult], jax.jit(jax.vmap(reset))
    )
    _, observation, _, _ = batched_reset(
        stacked, jax.random.split(jax.random.key(8), len(depths))
    )
    assert observation.context_features.shape == (2, MAX_AGENT_SLOTS, CONTEXT_FEATURES)
    for lane, depth in enumerate(depths):
        np.testing.assert_array_equal(
            observation.context_features[lane, :, CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH],
            np.where(_configured_rows(configs[lane]), depth, 0.0),
        )
