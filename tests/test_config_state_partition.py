"""Focused config/state ownership proofs for Milestone 5 Step 2 CP3C."""
# pyright: reportPrivateUsage=false

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import (
    _build_observation_and_action_mask,
    reset,
    step,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_BASE_MOVEMENT_SPEED,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_MAX_HEALTH,
    AGENT_FEATURE_OBSERVATION_RADIUS,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_TEAM_ID,
    AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS,
    ENVIRONMENT_DIMENSIONS,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_STAY,
    MOVE_WEST,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    SLOW_CHANNEL_HUNTER_BASIC,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)

_FINAL_STATE_FIELDS = (
    "step_count",
    "agent_positions",
    "alive_mask",
    "current_health",
    "ultimate_cooldowns",
    "slow_durations",
    "stun_durations",
    "rogue_poison_anti_heal_durations",
    "mage_burst_damage_amplification_durations",
    "priest_blessing_of_freedom_slow_floor_durations",
)


def _requested_roster() -> Array:
    return jnp.asarray(
        (
            MAGE_CLASS_ID,
            PRIEST_CLASS_ID,
            MAGE_CLASS_ID,
            PRIEST_CLASS_ID,
            MAGE_CLASS_ID,
            PRIEST_CLASS_ID,
            MAGE_CLASS_ID,
            PRIEST_CLASS_ID,
            MAGE_CLASS_ID,
            PRIEST_CLASS_ID,
        ),
        dtype=jnp.int32,
    )


def _config(team_sizes: tuple[int, int] = (1, 1)) -> EnvConfig:
    return EnvConfig(
        team_size=max(team_sizes),
        max_steps=1000,
        map_width=20.0,
        map_height=12.0,
        obstacles=jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32),
        agent_profile=resolve_agent_profile(
            _requested_roster(), jnp.asarray(team_sizes, dtype=jnp.int32)
        ),
    )


def _zero_action() -> Action:
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _current_action_mask(config: EnvConfig, state: EnvState) -> ActionMask:
    """Return the action mask paired with an explicitly built test state."""
    _, action_mask = _build_observation_and_action_mask(state, config)
    return action_mask


def _assert_tree_equal(left: EnvState, right: EnvState) -> None:
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        assert bool(jnp.array_equal(left_leaf, right_leaf))


def test_env_state_contains_only_dynamic_transition_memory() -> None:
    assert EnvState._fields == _FINAL_STATE_FIELDS


def test_reset_initializes_dynamic_state_from_resolved_profile() -> None:
    config = _config((1, 2))
    state, observation, action_mask, _ = reset(config, jax.random.key(7))
    profile = config.agent_profile

    assert state.alive_mask.dtype == jnp.bool_
    assert bool(jnp.array_equal(state.alive_mask, profile.active_mask))
    assert bool(jnp.array_equal(state.current_health, profile.max_health))
    inactive_mask = jnp.logical_not(profile.active_mask)
    assert bool(jnp.all(action_mask.move_mask[inactive_mask, MOVE_STAY]))
    assert bool(jnp.all(jnp.sum(action_mask.move_mask[inactive_mask], axis=-1) == 1))
    assert bool(jnp.all(action_mask.select_target_mask[inactive_mask, 0]))
    assert bool(
        jnp.all(jnp.sum(action_mask.select_target_mask[inactive_mask], axis=-1) == 1)
    )
    assert bool(jnp.all(action_mask.use_ultimate_mask[inactive_mask, 0]))
    assert bool(
        jnp.all(jnp.sum(action_mask.use_ultimate_mask[inactive_mask], axis=-1) == 1)
    )

    static_columns = (
        (AGENT_FEATURE_RADIUS, profile.agent_radii),
        (AGENT_FEATURE_TEAM_ID, profile.team_ids.astype(jnp.float32)),
        (AGENT_FEATURE_ACTIVE, profile.active_mask.astype(jnp.float32)),
        (AGENT_FEATURE_CLASS_ID, profile.class_ids.astype(jnp.float32)),
        (AGENT_FEATURE_BASE_MOVEMENT_SPEED, profile.base_movement_speeds),
        (AGENT_FEATURE_OBSERVATION_RADIUS, profile.observation_radii),
        (AGENT_FEATURE_BASIC_INTERACTION_RADIUS, profile.basic_interaction_radii),
        (
            AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS,
            profile.ultimate_interaction_radii,
        ),
        (AGENT_FEATURE_MAX_HEALTH, profile.max_health),
    )
    for column, expected in static_columns:
        assert bool(jnp.array_equal(observation.self_features[:, column], expected))


def test_profile_base_speed_changes_movement_without_replacing_state() -> None:
    config = _config((1, 0))
    state, *_ = reset(config, jax.random.key(1))
    state = state._replace(
        agent_positions=state.agent_positions.at[0].set(
            jnp.asarray((10.0, 6.0), dtype=jnp.float32)
        )
    )
    action = _zero_action()._replace(move=_zero_action().move.at[0].set(MOVE_EAST))

    faster_profile = config.agent_profile._replace(
        base_movement_speeds=config.agent_profile.base_movement_speeds.at[0].set(1.75)
    )
    faster_config = config._replace(agent_profile=faster_profile)

    ordinary_state, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        action,
        jax.random.key(2),
    )
    faster_state, *_ = step(
        faster_config,
        state,
        _current_action_mask(faster_config, state),
        action,
        jax.random.key(2),
    )

    assert bool(
        faster_state.agent_positions[0, 0] > ordinary_state.agent_positions[0, 0]
    )


def test_profile_radius_changes_geometry_projection_without_replacing_state() -> None:
    config = _config((1, 0))
    state, *_ = reset(config, jax.random.key(3))
    state = state._replace(
        agent_positions=state.agent_positions.at[0].set(
            jnp.asarray((0.6, 6.0), dtype=jnp.float32)
        )
    )
    action = _zero_action()._replace(move=_zero_action().move.at[0].set(MOVE_WEST))

    small_config = config._replace(
        agent_profile=config.agent_profile._replace(
            agent_radii=config.agent_profile.agent_radii.at[0].set(0.25)
        )
    )
    large_config = config._replace(
        agent_profile=config.agent_profile._replace(
            agent_radii=config.agent_profile.agent_radii.at[0].set(0.75)
        )
    )

    small_state, *_ = step(
        small_config,
        state,
        _current_action_mask(small_config, state),
        action,
        jax.random.key(4),
    )
    large_state, *_ = step(
        large_config,
        state,
        _current_action_mask(large_config, state),
        action,
        jax.random.key(4),
    )

    assert bool(small_state.agent_positions[0, 0] < large_state.agent_positions[0, 0])


def test_profile_radii_control_visibility_and_targetability() -> None:
    config = _config((1, 1))
    state, *_ = reset(config, jax.random.key(11))
    positions = state.agent_positions
    positions = positions.at[0].set(jnp.asarray((2.0, 2.0), dtype=jnp.float32))
    positions = positions.at[5].set(jnp.asarray((8.0, 2.0), dtype=jnp.float32))
    state = state._replace(agent_positions=positions)

    short_profile = config.agent_profile._replace(
        observation_radii=config.agent_profile.observation_radii.at[0].set(3.0),
        basic_interaction_radii=(
            config.agent_profile.basic_interaction_radii.at[0].set(3.0)
        ),
    )
    long_observation_profile = short_profile._replace(
        observation_radii=short_profile.observation_radii.at[0].set(10.0)
    )
    long_interaction_profile = long_observation_profile._replace(
        basic_interaction_radii=(
            long_observation_profile.basic_interaction_radii.at[0].set(10.0)
        )
    )

    short_config = config._replace(agent_profile=short_profile)
    visible_config = config._replace(agent_profile=long_observation_profile)
    targetable_config = config._replace(agent_profile=long_interaction_profile)

    _, short_observation, _, _, short_action_mask, _ = step(
        short_config,
        state,
        _current_action_mask(short_config, state),
        _zero_action(),
        jax.random.key(12),
    )
    _, visible_observation, _, _, visible_action_mask, _ = step(
        visible_config,
        state,
        _current_action_mask(visible_config, state),
        _zero_action(),
        jax.random.key(12),
    )
    _, _, _, _, targetable_action_mask, _ = step(
        targetable_config,
        state,
        _current_action_mask(targetable_config, state),
        _zero_action(),
        jax.random.key(12),
    )

    assert not bool(short_observation.enemy_visibility_mask[0, 0])
    assert bool(visible_observation.enemy_visibility_mask[0, 0])
    enemy_target = 1 + MAX_AGENTS_PER_TEAM
    assert not bool(
        short_action_mask.select_target_use_ultimate_joint_mask[0, enemy_target, 0]
    )
    assert not bool(
        visible_action_mask.select_target_use_ultimate_joint_mask[0, enemy_target, 0]
    )
    assert bool(
        targetable_action_mask.select_target_use_ultimate_joint_mask[0, enemy_target, 0]
    )


def test_step_preserves_non_inert_dynamic_memory() -> None:
    config = _config()
    state, *_ = reset(config, jax.random.key(5))
    state = state._replace(
        current_health=state.current_health.at[0].set(12.5),
        ultimate_cooldowns=state.ultimate_cooldowns.at[0].set(7),
        slow_durations=state.slow_durations.at[0, 1].set(3),
        stun_durations=state.stun_durations.at[5, 2].set(2),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[5].set(4)
        ),
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[0].set(5)
        ),
        priest_blessing_of_freedom_slow_floor_durations=(
            state.priest_blessing_of_freedom_slow_floor_durations.at[5].set(1)
        ),
    )

    next_state, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        _zero_action(),
        jax.random.key(6),
    )

    assert next_state.step_count == state.step_count + 1
    assert bool(jnp.array_equal(next_state.current_health, state.current_health))
    assert bool(
        jnp.array_equal(
            next_state.ultimate_cooldowns,
            jnp.maximum(state.ultimate_cooldowns - 1, 0),
        )
    )
    expected_slow_durations = state.slow_durations.at[0, SLOW_CHANNEL_HUNTER_BASIC].set(
        2
    )
    assert bool(jnp.array_equal(next_state.slow_durations, expected_slow_durations))
    assert bool(jnp.array_equal(next_state.stun_durations, state.stun_durations))
    assert bool(
        jnp.array_equal(
            next_state.rogue_poison_anti_heal_durations,
            state.rogue_poison_anti_heal_durations,
        )
    )
    assert bool(
        jnp.array_equal(
            next_state.mage_burst_damage_amplification_durations,
            state.mage_burst_damage_amplification_durations,
        )
    )
    expected_freedom_durations = (
        state.priest_blessing_of_freedom_slow_floor_durations.at[5].set(0)
    )
    assert bool(
        jnp.array_equal(
            next_state.priest_blessing_of_freedom_slow_floor_durations,
            expected_freedom_durations,
        )
    )


def test_step_matches_jit_and_keeps_scan_carry_structure_stable() -> None:
    config = _config()
    state, *_ = reset(config, jax.random.key(8))
    action = _zero_action()
    current_action_mask = _current_action_mask(config, state)
    eager_state, *_ = step(
        config, state, current_action_mask, action, jax.random.key(9)
    )
    jitted_output = cast(
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        jax.jit(step)(
            config,
            state,
            current_action_mask,
            action,
            jax.random.key(9),
        ),
    )
    jitted_state = jitted_output[0]
    _assert_tree_equal(eager_state, jitted_state)

    def _scan_step(
        carry: tuple[EnvState, ActionMask], key: Array
    ) -> tuple[tuple[EnvState, ActionMask], Array]:
        current_state, action_mask = carry
        next_state, _, _, _, next_action_mask, _ = step(
            config, current_state, action_mask, action, key
        )
        return (next_state, next_action_mask), next_state.step_count

    (scanned_state, scanned_action_mask), history = jax.lax.scan(
        _scan_step,
        (state, current_action_mask),
        jax.random.split(jax.random.key(10), 3),
    )
    assert jax.tree_util.tree_structure(scanned_state) == jax.tree_util.tree_structure(
        state
    )
    assert history.shape == (3,)
    assert scanned_state.step_count == 3
    assert jax.tree_util.tree_structure(
        scanned_action_mask
    ) == jax.tree_util.tree_structure(current_action_mask)


def test_action_shapes_remain_independent_of_profile_storage() -> None:
    action = _zero_action()
    assert action.move.shape == (MAX_AGENT_SLOTS,)
    assert action.select_target.shape == (MAX_AGENT_SLOTS,)
    assert action.use_ultimate.shape == (MAX_AGENT_SLOTS,)
    assert NUM_TARGET_ACTIONS == MAX_AGENT_SLOTS + 1
    assert NUM_ULTIMATE_ACTIONS == 2
    assert ENVIRONMENT_DIMENSIONS == 2
    assert NUM_SLOW_CHANNELS == 3
    assert NUM_STUN_CHANNELS == 3
