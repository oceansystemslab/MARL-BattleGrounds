"""Check explicit resets, episode IDs and cumulative real-transition counts.

Cases include partial resets, retained configuration overrides and counter limits.
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds import Environment, EnvironmentState, make
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import TASK_MODE_TDM, Action, EnvConfig
from marl_battlegrounds.environment import InitialSnapshot
from marl_battlegrounds.tasks import (
    balanced_spawn_configs,
    load_tdm_scenario,
    make_standard_team_deathmatch_config,
)

_INT32_MAX = int(np.iinfo(np.int32).max)
_DEFAULTS = 0
_REUSE = 1
_EXPLICIT = 2
_AUTHORED = 3


def _config(*, max_steps: int = 3, alternate: bool = False) -> EnvConfig:
    return make_standard_team_deathmatch_config(
        map_id=13 if alternate else 12,
        team_a_roster=("priest",) if alternate else ("mage", "priest"),
        team_b_roster=("hunter", "mage", "mage") if alternate else ("mage",),
        max_steps=max_steps,
    )


def _idle(batch: int | None = None) -> Action:
    shape = (10,) if batch is None else (batch, 10)
    zeros = jnp.zeros(shape, jnp.int32)
    return Action(zeros, zeros, zeros)


def _assert_tree_exact(actual: object, expected: object) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        a, b = np.asarray(left), np.asarray(right)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


def _row[T](tree: T, index: int) -> T:
    def select(value: Array) -> Array:
        return value[index]

    return jax.tree.map(select, tree)


def _reset_selected(
    env: Environment,
    state: EnvironmentState,
    *,
    mask: Array,
    env_config: EnvConfig | None = None,
    episode_id: int | Array | None = None,
) -> EnvironmentState:
    return env.reset(
        jax.random.key(12),
        env_config,
        state=state,
        reset_mask=mask,
        episode_id=episode_id,
    )[1]


def test_scalar_reset_keeps_override_history_without_an_intervening_step() -> None:
    original, replacement = _config(), _config(max_steps=5, alternate=True)
    env = make("tdm", env_config=original, metrics="none")
    _, initial = env.reset(jax.random.key(12))
    assert int(initial.episode_id) == 1
    assert int(initial.last_reserved_episode_id) == 1
    assert int(initial.reset_generation) == 0
    assert int(initial.reset_origin) == _DEFAULTS
    assert int(initial.config_origin_generation) == -1
    assert int(initial.cumulative_transition_count) == 0
    assert bool(initial.episode_start)
    assert not bool(initial.authored_start)
    assert not bool(initial.lifecycle_error)

    explicit = _reset_selected(
        env,
        initial,
        mask=jnp.asarray(True),
        env_config=replacement,
        episode_id=1,
    )
    assert int(explicit.episode_id) == 1
    assert int(explicit.reset_generation) == 1
    assert int(explicit.reset_origin) == _EXPLICIT
    assert int(explicit.config_origin_generation) == 1
    reused = _reset_selected(env, explicit, mask=jnp.asarray(True))
    assert int(reused.episode_id) == 2
    assert int(reused.reset_generation) == 2
    assert int(reused.reset_origin) == _REUSE
    assert int(reused.config_origin_generation) == 1
    assert int(reused.cumulative_transition_count) == 0
    _assert_tree_exact(reused.config, explicit.config)
    np.testing.assert_array_equal(
        reused.config.team_spawn_pad_positions, replacement.team_spawn_pad_positions
    )
    assert int(reused.config.max_steps) == 5
    for field in (
        reused.episode_id,
        reused.reset_generation,
        reused.config_origin_generation,
        reused.cumulative_transition_count,
        reused.last_reserved_episode_id,
    ):
        assert field.shape == () and field.dtype == jnp.int32


def test_native_reservations_preserve_every_unselected_row() -> None:
    env = make("tdm", env_config=_config(), num_envs=2, metrics="priority")
    key = jax.random.key(12)
    _, initial = env.reset(key)
    np.testing.assert_array_equal(initial.episode_id, (1, 2))
    initial = initial._replace(source_availability=jnp.zeros((2, 10, 10), bool))
    _, old, *_ = env.step(key, initial, _idle(2))
    replacement = _config(max_steps=5, alternate=True)
    first = _reset_selected(
        env, old, mask=jnp.asarray((True, False)), env_config=replacement
    )
    _assert_tree_exact(_row(first, 1), _row(old, 1))
    np.testing.assert_array_equal(first.episode_id, (3, 2))
    np.testing.assert_array_equal(first.last_reserved_episode_id, (4, 2))
    np.testing.assert_array_equal(first.reset_generation, (1, 0))
    np.testing.assert_array_equal(first.episode_start, (True, False))
    np.testing.assert_array_equal(first.cumulative_transition_count, (1, 1))
    np.testing.assert_array_equal(first.config.max_steps, (5, 3))

    # The unused explicit ID still reserves a gap when another lane resets.
    explicit = _reset_selected(
        env,
        first,
        mask=jnp.asarray((False, True)),
        episode_id=jnp.asarray((99, 7), jnp.int32),
    )
    _assert_tree_exact(_row(explicit, 0), _row(first, 0))
    np.testing.assert_array_equal(explicit.episode_id, (3, 7))
    np.testing.assert_array_equal(explicit.last_reserved_episode_id, (4, 99))
    untouched = _reset_selected(
        env,
        explicit,
        mask=jnp.asarray((False, False)),
        env_config=_config(),
        episode_id=jnp.asarray((_INT32_MAX, _INT32_MAX - 1), jnp.int32),
    )
    _assert_tree_exact(untouched, explicit)

    automatic = _reset_selected(env, untouched, mask=jnp.asarray((True, False)))
    _assert_tree_exact(_row(automatic, 1), _row(untouched, 1))
    np.testing.assert_array_equal(automatic.episode_id, (100, 7))
    np.testing.assert_array_equal(automatic.last_reserved_episode_id, (101, 99))
    lower = _reset_selected(
        env,
        automatic,
        mask=jnp.asarray((True, True)),
        episode_id=jnp.asarray((1, 2), jnp.int32),
    )
    np.testing.assert_array_equal(lower.last_reserved_episode_id, (101, 101))
    final = _reset_selected(env, lower, mask=jnp.asarray((True, True)))
    np.testing.assert_array_equal(final.episode_id, (102, 103))
    np.testing.assert_array_equal(final.last_reserved_episode_id, (103, 103))
    assert not bool(jnp.any(final.lifecycle_error))


@pytest.mark.parametrize(
    "episode_id",
    (
        0,
        -1,
        _INT32_MAX + 1,
        np.asarray(_INT32_MAX + 1, np.uint32),
        np.asarray(2**32 + 1, np.uint64),
        np.asarray(2**32 + 1, np.int64),
        True,
        1.0,
    ),
    ids=("zero", "negative", "large-int", "uint32", "uint64", "int64", "bool", "float"),
)
def test_host_reset_rejects_ids_before_narrowing(episode_id: object) -> None:
    env = make("tdm", env_config=_config(), metrics="none")
    with pytest.raises(ValueError, match=r"episode|int32"):
        env.reset(jax.random.key(12), episode_id=cast(int | Array, episode_id))


def test_compiled_unsigned_ids_keep_explicit_failure_evidence() -> None:
    env = make("tdm", env_config=_config(), metrics="none")

    def reset(episode_id: Array) -> EnvironmentState:
        return env.reset(jax.random.key(12), episode_id=episode_id)[1]

    compiled = cast(Callable[[Array], EnvironmentState], jax.jit(reset))
    valid = compiled(jnp.asarray(_INT32_MAX, jnp.uint32))
    assert int(valid.episode_id) == _INT32_MAX
    assert int(valid.last_reserved_episode_id) == _INT32_MAX
    assert not bool(valid.lifecycle_error)
    for value in (0, _INT32_MAX + 1, 2**32 - 1):
        invalid = compiled(jnp.asarray(value, jnp.uint32))
        assert bool(invalid.lifecycle_error)
        assert int(invalid.episode_id) > 0
        sticky = _reset_selected(env, invalid, mask=jnp.asarray(True), episode_id=1)
        assert bool(sticky.lifecycle_error)


def test_native_id_exhaustion_checks_the_whole_reserved_block() -> None:
    env = make("tdm", env_config=_config(), num_envs=2, metrics="none")
    key = jax.random.key(12)

    def reset(state: EnvironmentState, mask: Array) -> EnvironmentState:
        return env.reset(key, state=state, reset_mask=mask)[1]

    compiled = cast(
        Callable[[EnvironmentState, Array], EnvironmentState], jax.jit(reset)
    )
    _, near_limit = env.reset(
        key, episode_id=jnp.asarray((_INT32_MAX - 3, _INT32_MAX - 2), jnp.int32)
    )
    maximum = compiled(near_limit, jnp.asarray((True, True)))
    np.testing.assert_array_equal(maximum.episode_id, (_INT32_MAX - 1, _INT32_MAX))
    np.testing.assert_array_equal(
        maximum.last_reserved_episode_id, (_INT32_MAX, _INT32_MAX)
    )
    assert not bool(jnp.any(maximum.lifecycle_error))
    _assert_tree_exact(compiled(maximum, jnp.asarray((False, False))), maximum)

    _, one_id_left = env.reset(
        key, episode_id=jnp.asarray((_INT32_MAX - 2, _INT32_MAX - 1), jnp.int32)
    )
    exhausted = compiled(one_id_left, jnp.asarray((True, False)))
    assert bool(exhausted.lifecycle_error[0])
    _assert_tree_exact(_row(exhausted, 1), _row(one_id_left, 1))
    assert int(exhausted.last_reserved_episode_id[0]) >= _INT32_MAX - 1
    assert int(exhausted.episode_id[0]) > 0
    sticky = _reset_selected(
        env,
        exhausted,
        mask=jnp.asarray((True, False)),
        episode_id=jnp.asarray((1, 2), jnp.int32),
    )
    assert bool(sticky.lifecycle_error[0])
    _assert_tree_exact(_row(sticky, 1), _row(exhausted, 1))


def test_generation_and_transition_limits_preserve_failure_across_resets() -> None:
    env = make("tdm", env_config=_config(max_steps=2), metrics="none")
    key = jax.random.key(12)
    _, state = env.reset(key)

    def reset(state: EnvironmentState) -> EnvironmentState:
        return env.reset(key, state=state, reset_mask=jnp.asarray(True), episode_id=1)[
            1
        ]

    compiled_reset = cast(
        Callable[[EnvironmentState], EnvironmentState], jax.jit(reset)
    )
    near_generation = state._replace(
        reset_generation=jnp.asarray(_INT32_MAX - 1, jnp.int32)
    )
    maximum_generation = compiled_reset(near_generation)
    assert int(maximum_generation.reset_generation) == _INT32_MAX
    assert not bool(maximum_generation.lifecycle_error)
    exhausted_generation = compiled_reset(maximum_generation)
    assert bool(exhausted_generation.lifecycle_error)
    assert int(exhausted_generation.reset_generation) >= 0
    assert bool(compiled_reset(exhausted_generation).lifecycle_error)

    near_count = state._replace(
        cumulative_transition_count=jnp.asarray(_INT32_MAX - 1, jnp.int32)
    )
    _, maximum_count, *_ = env.step(key, near_count, _idle())
    assert int(maximum_count.cumulative_transition_count) == _INT32_MAX
    assert not bool(maximum_count.lifecycle_error)
    _, exhausted_count, _, done, _ = env.step(key, maximum_count, _idle())
    assert bool(done.done)
    assert bool(exhausted_count.lifecycle_error)
    assert int(exhausted_count.cumulative_transition_count) >= 0
    _, padded, _, _, info = env.step(key, exhausted_count, _idle())
    _assert_tree_exact(padded, exhausted_count)
    assert not bool(info.completed)
    reset_count = compiled_reset(exhausted_count)
    assert bool(reset_count.lifecycle_error)
    assert int(reset_count.cumulative_transition_count) == int(
        exhausted_count.cumulative_transition_count
    )

    # A terminal state at the maximum needs no extra count for a padding call.
    safe_terminal = exhausted_count._replace(
        cumulative_transition_count=jnp.asarray(_INT32_MAX, jnp.int32),
        lifecycle_error=jnp.asarray(False),
    )
    _, padded_at_limit, *_ = env.step(key, safe_terminal, _idle())
    _assert_tree_exact(padded_at_limit, safe_terminal)


def test_external_vmap_keeps_scalar_id_contexts_independent() -> None:
    env = make("tdm", env_config=_config(), metrics="none")

    def run(key: Array) -> tuple[EnvironmentState, EnvironmentState]:
        _, first = env.reset(key)
        _, second = env.reset(key, state=first, reset_mask=jnp.asarray(True))
        return first, second

    batched = cast(
        Callable[[Array], tuple[EnvironmentState, EnvironmentState]],
        jax.jit(jax.vmap(run)),
    )
    keys = jax.random.split(jax.random.key(12), 2)
    first, second = batched(keys)
    np.testing.assert_array_equal(first.episode_id, (1, 1))
    np.testing.assert_array_equal(second.episode_id, (2, 2))
    np.testing.assert_array_equal(second.last_reserved_episode_id, (2, 2))
    for lane in range(2):
        _assert_tree_exact((_row(first, lane), _row(second, lane)), run(keys[lane]))


def test_scan_keeps_terminal_results_before_reset_done() -> None:
    config = _config(max_steps=2)
    env = make("tdm", env_config=config, metrics="none")
    keys = jax.random.split(jax.random.key(12), 5)

    def run(automatic: bool) -> object:
        _, initial = env.reset(jax.random.key(12))

        def advance(
            state: EnvironmentState, key: Array
        ) -> tuple[EnvironmentState, object]:
            result = env.step(key, state, _idle())
            terminal = result[1]
            _, successor = (
                env.reset_done(key, terminal)
                if automatic
                else env.reset(
                    key,
                    config,
                    episode_id=terminal.episode_id + 1,
                    state=terminal,
                    reset_mask=terminal.done.done,
                )
            )
            return successor, (
                result[0],
                terminal.core_state,
                terminal.action_mask,
                result[2:],
                terminal.episode_id,
                terminal.reset_generation,
                terminal.cumulative_transition_count,
                successor.episode_id,
                successor.episode_start,
                successor.cumulative_transition_count,
            )

        return jax.lax.scan(advance, initial, keys)[1]

    automatic = cast(tuple[object, ...], jax.jit(lambda: run(True))())
    manual = cast(object, jax.jit(lambda: run(False))())
    _assert_tree_exact(automatic, manual)
    np.testing.assert_array_equal(automatic[4], (1, 1, 2, 2, 3))
    np.testing.assert_array_equal(automatic[5], (0, 0, 1, 1, 2))
    np.testing.assert_array_equal(automatic[6], (1, 2, 3, 4, 5))
    np.testing.assert_array_equal(automatic[7], (1, 2, 2, 3, 3))
    np.testing.assert_array_equal(automatic[8], (False, True, False, True, False))
    np.testing.assert_array_equal(automatic[9], (1, 2, 3, 4, 5))


def test_prepared_authored_start_keeps_core_epoch_and_separate_reset_history() -> None:
    config = _config(max_steps=4)
    key = jax.random.key(12)
    core_state, *_ = core.reset(config, key)
    authored = core_state._replace(
        step_count=jnp.asarray(2, jnp.int32),
        team_deathmatch_scores=jnp.asarray((3, 1), jnp.int32),
    )
    prepared = core.initialize_scenario_state(authored, config)
    initial: InitialSnapshot = prepared[:3]
    env = make("tdm", env_config=_config(max_steps=9, alternate=True), metrics="none")

    def reset(env_config: EnvConfig, initial: InitialSnapshot) -> EnvironmentState:
        return env.reset(key, env_config, initial=initial)[1]

    compiled = cast(
        Callable[[EnvConfig, InitialSnapshot], EnvironmentState], jax.jit(reset)
    )
    state = compiled(config, initial)
    _assert_tree_exact(
        (state.core_state, state.observation, state.action_mask), initial
    )
    assert int(state.initial_step_count) == 2
    assert int(state.cumulative_transition_count) == 0
    assert int(state.reset_generation) == 0
    assert int(state.reset_origin) == _AUTHORED
    assert int(state.config_origin_generation) == 0
    assert bool(state.authored_start) and bool(state.episode_start)
    for _ in range(2):
        _, state, *_ = env.step(key, state, _idle())
    assert bool(state.done.done)
    assert int(state.cumulative_transition_count) == 2
    _, reused = env.reset_done(key, state)
    assert int(reused.core_state.step_count) == 0
    assert int(reused.initial_step_count) == 0
    assert int(reused.cumulative_transition_count) == 2
    assert int(reused.reset_generation) == 1
    assert int(reused.reset_origin) == _REUSE
    assert int(reused.config_origin_generation) == 0
    assert not bool(reused.authored_start)
    _assert_tree_exact(reused.config, state.config)


def test_scenario_reset_uses_its_exact_start_and_rejects_conflicts() -> None:
    scenario = load_tdm_scenario(1)
    config = _config()
    env = make("tdm", env_config=config, metrics="none")
    key = jax.random.key(12)
    expected = core.initialize_scenario_state(scenario.initial_state, scenario.config)
    observation, state = env.reset(key, scenario=scenario)
    _assert_tree_exact(state.core_state, expected[0])
    _assert_tree_exact(observation.observation, expected[1])
    _assert_tree_exact(state.action_mask, expected[2])
    np.testing.assert_array_equal(
        state.config.team_spawn_pad_positions, scenario.config.team_spawn_pad_positions
    )
    assert int(state.config.max_steps) == scenario.config.max_steps
    assert int(state.reset_origin) == _AUTHORED
    assert int(state.config_origin_generation) == 0
    assert bool(state.authored_start)
    with pytest.raises(ValueError, match="scenario"):
        env.reset(key, config, scenario=scenario)
    with pytest.raises(ValueError, match="scenario"):
        env.reset(key, initial=expected[:3], scenario=scenario)

    def traced_scenario(key: Array) -> EnvironmentState:
        return env.reset(key, scenario=scenario)[1]

    with pytest.raises((TypeError, ValueError), match=r"host|scenario|trac"):
        jax.jit(traced_scenario)(key)


def test_reset_reuses_prepared_spawn_banks_without_exchanging_them_again() -> None:
    source = _config(max_steps=1)
    prepared = balanced_spawn_configs(source, num_envs=2)
    env = make("tdm", map_id=12, num_envs=2, max_steps=1, metrics="none")
    key = jax.random.key(12)
    _, state = env.reset(key, prepared)
    for _ in range(3):
        np.testing.assert_array_equal(
            state.config.team_spawn_pad_positions, prepared.team_spawn_pad_positions
        )
        np.testing.assert_array_equal(
            state.config.agent_profile.class_ids, prepared.agent_profile.class_ids
        )
        _, terminal, *_ = env.step(key, state, _idle(2))
        _, state = env.reset_done(key, terminal)
    assert not bool(jnp.any(state.authored_start))
    np.testing.assert_array_equal(state.config_origin_generation, (0, 0))
    np.testing.assert_array_equal(state.cumulative_transition_count, (3, 3))


def test_authored_death_and_respawn_keep_each_teams_resolved_spawn_bank() -> None:
    config = evaluation_env_config(
        team_sizes=(1, 1),
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=20,
        max_steps=5,
    )
    config = config._replace(
        team_spawn_pad_positions=config.team_spawn_pad_positions[::-1],
        spawn_shield_duration_steps=0,
        team_respawn_wave_period_step_count=jnp.asarray((2, 2), jnp.int32),
    )
    key = jax.random.key(12)
    initial_core, *_ = core.reset(config, key)
    authored = initial_core._replace(
        agent_positions=initial_core.agent_positions.at[0]
        .set(jnp.asarray((4.0, 4.0), jnp.float32))
        .at[5]
        .set(jnp.asarray((6.5, 4.0), jnp.float32)),
        current_health=initial_core.current_health.at[0].set(1.0).at[5].set(1.0),
        team_respawn_wave_countdowns=jnp.asarray((1, 1), jnp.int32),
    )
    prepared = core.initialize_scenario_state(authored, config)
    env = make("tdm", env_config=config, metrics="none")
    _, state = env.reset(key, config, initial=prepared[:3])
    for slot in (0, 5):
        assert bool(state.action_mask.select_target_use_ultimate_joint_mask[slot, 6, 0])
    action = _idle()._replace(
        select_target=jnp.zeros(10, jnp.int32).at[0].set(6).at[5].set(6)
    )
    _, dead, *_ = env.step(key, state, action)
    np.testing.assert_array_equal(
        dead.core_state.alive_mask[jnp.asarray((0, 5))], (False, False)
    )
    _, respawned, *_ = env.step(key, dead, _idle())
    np.testing.assert_array_equal(
        respawned.core_state.alive_mask[jnp.asarray((0, 5))], (True, True)
    )
    for team, slot in ((0, 0), (1, 5)):
        np.testing.assert_array_equal(
            respawned.core_state.agent_positions[slot],
            config.team_spawn_pad_positions[team, 0],
        )
    _assert_tree_exact(respawned.config, state.config)
    assert int(respawned.episode_id) == int(state.episode_id)
    assert int(respawned.cumulative_transition_count) == 2
