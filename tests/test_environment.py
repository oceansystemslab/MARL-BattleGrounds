"""Check the public environment against Core trajectories.

The tests cover configuration ownership, metric selection, batched execution
and episode bookkeeping.
"""

from collections.abc import Callable
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds import Environment, EnvironmentState, make
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import (
    TASK_MODE_TDM,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    Reward,
)
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation.metric_catalog import PRIORITY_METRIC_NAMES
from marl_battlegrounds.evaluation.policy_execution import apply_policies, policy
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

type StepResult = tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]
type Step = Callable[[Array, EnvironmentState, Action], StepResult]
type _PolicyTrajectory = tuple[
    tuple[Observations, EnvironmentState], tuple[Action, StepResult]
]


def _assert_tree_exact(actual: object, expected: object) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        a, b = np.asarray(left), np.asarray(right)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


def _stack(*leaves: Array) -> Array:
    return jnp.stack(leaves)


def _row[T](tree: T, index: int) -> T:
    def select(leaf: Array) -> Array:
        return leaf[index]

    return jax.tree.map(select, tree)


def _config(*, max_steps: int = 3, alternate: bool = False) -> EnvConfig:
    return make_standard_team_deathmatch_config(
        map_id=13 if alternate else 12,
        team_a_roster=("priest",) if alternate else ("mage", "priest"),
        team_b_roster=("hunter", "mage", "mage") if alternate else ("mage",),
        score_threshold=7 if alternate else 20,
        max_steps=max_steps,
    )


def _idle(batch: int | None = None) -> Action:
    shape = (10,) if batch is None else (batch, 10)
    zeros = jnp.zeros(shape, jnp.int32)
    return Action(zeros, zeros, zeros)


def _metric(info: EpisodeInfo, name: str) -> float:
    assert info.priority is not None
    column = PRIORITY_METRIC_NAMES.index(name)
    assert bool(info.priority.valid[column]), name
    return float(info.priority.values[column])


def _scenario_start(
    env: Environment,
    *,
    scores: tuple[int, int],
    max_steps: int,
    initial_step: int = 0,
    mutual: bool = False,
) -> EnvironmentState:
    config = evaluation_env_config(
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=20,
        max_steps=max_steps,
    )
    _, state = env.reset(jax.random.key(0), config, episode_id=77)
    initial = state.core_state._replace(
        step_count=jnp.asarray(initial_step, jnp.int32),
        team_deathmatch_scores=jnp.asarray(scores, jnp.int32),
        agent_positions=state.core_state.agent_positions.at[0]
        .set(jnp.asarray((4.0, 4.0), jnp.float32))
        .at[5]
        .set(jnp.asarray((6.5, 4.0), jnp.float32)),
        current_health=state.core_state.current_health.at[5]
        .set(1.0)
        .at[0]
        .set(1.0 if mutual else state.core_state.current_health[0]),
    )
    initial, observation, mask, _ = core.initialize_scenario_state(initial, config)
    return state._replace(
        core_state=initial,
        observation=observation,
        action_mask=mask,
        initial_step_count=initial.step_count,
    )


@pytest.fixture(scope="module")
def scalar_step() -> Step:
    return cast(Step, jax.jit(make("tdm").step))


@pytest.fixture(scope="module")
def native_step() -> Step:
    return cast(Step, jax.jit(make("tdm", num_envs=2).step))


def test_scalar_wrapper_matches_core_with_rejected_submitted_action(
    scalar_step: Step,
) -> None:
    env = make("tdm")
    key = jax.random.key(31)
    config = _config()
    observation, state = env.reset(key, config, episode_id=9)
    expected_state, expected_obs, expected_mask, _ = core.reset(config, key)
    _assert_tree_exact(state.core_state, expected_state)
    _assert_tree_exact(observation.observation, expected_obs)
    _assert_tree_exact(env.get_action_mask(state), expected_mask)

    action = _idle()._replace(select_target=jnp.zeros(10, jnp.int32).at[0].set(1))
    assert not bool(expected_mask.select_target_use_ultimate_joint_mask[0, 1, 0])
    expected = core.step(config, expected_state, expected_mask, action, key)
    assert bool(
        expected[
            5
        ].transition_facts.action_acceptance_facts.in_domain_combat_action_pair_is_rejected_by_actor[
            0
        ]
    )
    actual_obs, actual, reward, done, info = scalar_step(key, state, action)
    _assert_tree_exact(
        (actual.core_state, actual_obs.observation, reward, done, actual.action_mask),
        expected[:5],
    )
    assert int(info.episode_id) == 9
    assert not bool(info.completed)
    assert _metric(info, "episode_length") == 1


def test_typed_and_legacy_root_keys_preserve_native_policy_trajectories() -> None:
    env = make("tdm", num_envs=2)
    config = _config(max_steps=2)
    random_apply = policy("random").apply

    def choose(observations: Observations, mask: ActionMask, keys: Array) -> Action:
        return apply_policies(
            random_apply, random_apply, (), (), (), (), observations, mask, keys
        )[0]

    def advance(
        carry: tuple[Observations, EnvironmentState], key: Array
    ) -> tuple[tuple[Observations, EnvironmentState], tuple[Action, StepResult]]:
        observations, state = carry
        actions = jax.vmap(choose)(
            observations, state.action_mask, jax.random.split(key, (2, 10))
        )
        result = env.step(key, state, actions)
        return (result[0], result[1]), (actions, result)

    def run(key: Array) -> _PolicyTrajectory:
        initial = env.reset(key, config, episode_id=jnp.asarray((11, 12)))
        return jax.lax.scan(advance, initial, jax.random.split(key, 3))

    compiled = cast(Callable[[Array], _PolicyTrajectory], jax.jit(run))
    typed = compiled(jax.random.key(31))
    legacy = compiled(jax.random.PRNGKey(31))
    _assert_tree_exact(typed, legacy)
    np.testing.assert_array_equal(
        typed[1][1][4].completed, ((False, False), (True, True), (False, False))
    )


def test_native_batch_matches_scalar_core_under_documented_key_assignment(
    native_step: Step,
) -> None:
    env = make("tdm", num_envs=2)
    key = jax.random.key(42)
    configs = (_config(max_steps=1), _config(max_steps=3, alternate=True))
    ids = jnp.asarray((37, 105), jnp.int32)
    _, state = env.reset(key, jax.tree.map(_stack, *configs), episode_id=ids)
    obs, actual, reward, done, info = native_step(key, state, _idle(2))
    for lane, config in enumerate(configs):
        lane_key = jax.random.fold_in(key, ids[lane])
        initial, _, mask, _ = core.reset(config, lane_key)
        expected = core.step(config, initial, mask, _idle(), lane_key)
        _assert_tree_exact(
            (
                _row(actual.core_state, lane),
                _row(obs.observation, lane),
                _row(reward, lane),
                _row(done, lane),
                _row(actual.action_mask, lane),
            ),
            expected[:5],
        )
    np.testing.assert_array_equal(info.episode_id, ids)
    np.testing.assert_array_equal(info.completed, (True, False))


def test_partial_reset_replaces_only_completed_lane_and_its_configuration(
    native_step: Step,
) -> None:
    env = make("tdm", num_envs=2)
    key = jax.random.key(12)
    configs = (_config(max_steps=1), _config(max_steps=3))
    _, initial = env.reset(
        key, jax.tree.map(_stack, *configs), episode_id=jnp.asarray((1, 2))
    )
    _, old, _, done, _ = native_step(key, initial, _idle(2))
    replacement = _config(max_steps=5, alternate=True)
    obs, reset_state = env.reset(
        key,
        replacement,
        episode_id=jnp.asarray((91, 999)),
        state=old,
        reset_mask=done.done,
    )
    _assert_tree_exact(_row(reset_state, 1), _row(old, 1))
    fresh_core, fresh_obs, fresh_mask, _ = core.reset(
        replacement, jax.random.fold_in(key, 91)
    )
    _assert_tree_exact(_row(reset_state.core_state, 0), fresh_core)
    _assert_tree_exact(_row(obs.observation, 0), fresh_obs)
    _assert_tree_exact(_row(reset_state.action_mask, 0), fresh_mask)
    np.testing.assert_array_equal(reset_state.episode_id, (91, 2))
    np.testing.assert_array_equal(reset_state.config.max_steps, (5, 3))
    _, successor, _, _, info = native_step(key, reset_state, _idle(2))
    assert int(successor.core_state.step_count[0]) == 1
    assert int(successor.core_state.step_count[1]) == 2
    assert _metric(_row(info, 0), "episode_length") == 1
    assert _metric(_row(info, 1), "episode_length") == 2


def test_reset_reuses_one_executable_for_changed_config_and_roster() -> None:
    env = make("tdm")
    key = jax.random.key(0)
    episode_id = jnp.asarray(8, jnp.int32)
    first, second = _config(), _config(max_steps=5, alternate=True)
    compiled = jax.jit(env.reset).lower(key, first, episode_id=episode_id).compile()
    actual = compiled(key, second, episode_id=episode_id)
    expected = env.reset(key, second, episode_id=episode_id)
    _assert_tree_exact(actual, expected)
    assert int(actual[1].config.max_steps) == 5
    np.testing.assert_array_equal(
        actual[1].config.agent_profile.class_ids, second.agent_profile.class_ids
    )


def test_separate_handles_reuse_jit_with_changed_capture_selections() -> None:
    key, config = jax.random.key(28), _config(max_steps=1)
    ids = jnp.asarray((11, 12), jnp.int32)
    first = make(
        "tdm",
        num_envs=2,
        metrics="none",
        full_metrics_episodes=(11,),
        replay_episodes=(12,),
    )
    second = make(
        "tdm",
        num_envs=2,
        metrics="none",
        full_metrics_episodes=(12,),
        replay_episodes=(11,),
    )
    traces = 0

    def run(current: Environment, config: EnvConfig, key: Array) -> StepResult:
        nonlocal traces
        traces += 1
        _, state = current.reset(key, config, episode_id=ids)
        return current.step(key, state, _idle(2))

    compiled = cast(Callable[[Environment, EnvConfig, Array], StepResult], jax.jit(run))
    before = compiled(first, config, key)
    after = compiled(second, config, key)
    assert traces == 1
    _assert_tree_exact(before[0], after[0])
    _assert_tree_exact(before[1].core_state, after[1].core_state)
    _assert_tree_exact(before[1].action_mask, after[1].action_mask)
    _assert_tree_exact(before[2:4], after[2:4])

    for result, selected, replay_id in (
        (before, (True, False), 12),
        (after, (False, True), 11),
    ):
        _, state, _, _, info = result
        np.testing.assert_array_equal(state.collect_full_metrics, selected)
        np.testing.assert_array_equal(state.collect_replay, np.logical_not(selected))
        assert info.priority is not None and info.full is not None
        np.testing.assert_array_equal(info.priority.valid.any(axis=-1), selected)
        np.testing.assert_array_equal(info.full.valid.any(axis=-1), selected)
        assert info.replay is not None
        np.testing.assert_array_equal(info.replay.valid, (True,))
        np.testing.assert_array_equal(info.replay.episode_id, (replay_id,))
        lane = replay_id - 11
        _assert_tree_exact(_row(info.replay.state, 0), _row(state.core_state, lane))
        _assert_tree_exact(
            _row(info.replay.action_mask, 0), _row(state.action_mask, lane)
        )


def test_scalar_reset_remains_valid_under_external_vmap() -> None:
    env = make("tdm")
    config = _config()
    keys = jax.random.split(jax.random.key(7), 2)
    ids = jnp.asarray((11, 12), jnp.int32)

    def reset_lane(
        key: Array, episode_id: Array
    ) -> tuple[Observations, EnvironmentState]:
        return env.reset(key, config, episode_id=episode_id)

    actual = cast(
        tuple[Observations, EnvironmentState], jax.jit(jax.vmap(reset_lane))(keys, ids)
    )
    expected = jax.tree.map(_stack, *(reset_lane(keys[i], ids[i]) for i in range(2)))
    _assert_tree_exact(actual, expected)


def test_concrete_reset_rejects_invalid_ids_and_retained_lane_collisions() -> None:
    key, config = jax.random.key(0), _config()
    scalar = make("tdm")
    for episode_id in (0, -1):
        with pytest.raises(ValueError, match="positive int32"):
            scalar.reset(key, config, episode_id=episode_id)
    # JAX's default 32-bit conversion would otherwise alias this value to ID 1.
    wide_id = cast(Array, np.asarray(2**32 + 1, dtype=np.int64))
    with pytest.raises(ValueError, match="int32"):
        scalar.reset(key, config, episode_id=wide_id)

    native = make("tdm", num_envs=2)
    with pytest.raises(ValueError, match="unique"):
        native.reset(key, config, episode_id=jnp.asarray((3, 3), jnp.int32))

    _, state = native.reset(key, config, episode_id=jnp.asarray((3, 4), jnp.int32))
    with pytest.raises(ValueError, match="unique"):
        native.reset(
            key,
            config,
            episode_id=jnp.asarray((4, 99), jnp.int32),
            state=state,
            reset_mask=jnp.asarray((True, False)),
        )


def test_terminal_win_counts_observed_kills_separately_from_initialized_score(
    scalar_step: Step,
) -> None:
    env = make("tdm")
    state = _scenario_start(env, scores=(19, 7), max_steps=6, initial_step=4)
    action = _idle()._replace(select_target=jnp.zeros(10, jnp.int32).at[0].set(6))
    assert bool(state.action_mask.select_target_use_ultimate_joint_mask[0, 6, 0])
    key = jax.random.key(3)
    expected = core.step(state.config, state.core_state, state.action_mask, action, key)
    accepted = expected[
        5
    ].transition_facts.action_acceptance_facts.accepted_joint_action
    assert int(accepted.select_target[0]) == 6
    _, successor, reward, done, info = scalar_step(key, state, action)
    _assert_tree_exact(
        (successor.core_state, reward, done, successor.action_mask),
        (expected[0], expected[2], expected[3], expected[4]),
    )
    assert not bool(
        successor.action_mask.select_target_use_ultimate_joint_mask[0, 6, 0]
    )
    assert bool(done.terminated) and not bool(done.truncated)
    assert bool(info.completed) and int(info.episode_id) == 77
    assert not bool(successor.core_state.alive_mask[5])
    for name, expected in (
        ("episode_length", 1),
        ("team_a_score", 20),
        ("team_b_score", 7),
        ("team_a_kills", 1),
        ("team_b_kills", 0),
        ("team_a_deaths", 0),
        ("team_b_deaths", 1),
        ("team_a_return", 1),
        ("team_b_return", -1),
        ("team_a_win", 1),
        ("team_b_loss", 1),
    ):
        assert _metric(info, name) == expected
    for slot, active in enumerate(np.asarray(state.config.agent_profile.active_mask)):
        assert info.priority is not None
        column = PRIORITY_METRIC_NAMES.index(f"agent_{slot}_return")
        assert bool(info.priority.valid[column]) == bool(active)
        if active:
            assert _metric(info, f"agent_{slot}_return") == float(reward.rewards[slot])

    obs, padded, padding_reward, padding_done, padding_info = scalar_step(
        jax.random.key(4), successor, action
    )
    _assert_tree_exact(padded, successor)
    _assert_tree_exact(obs.observation, successor.observation)
    _assert_tree_exact(padding_done, done)
    np.testing.assert_array_equal(padding_reward.rewards, np.zeros(10))
    assert not bool(padding_info.completed)


def test_horizon_draw_keeps_unequal_scores_and_real_zero_measurements(
    scalar_step: Step,
) -> None:
    state = _scenario_start(make("tdm"), scores=(7, 3), max_steps=3, initial_step=2)
    _, _, _, done, info = scalar_step(jax.random.key(5), state, _idle())
    assert not bool(done.terminated) and bool(done.truncated)
    assert bool(info.completed)
    for name, expected in (
        ("episode_length", 1),
        ("score_difference", 4),
        ("team_a_draw", 1),
        ("team_b_draw", 1),
        ("team_a_win", 0),
        ("team_b_loss", 0),
        ("team_a_return", 0),
        ("team_b_return", 0),
        ("team_a_kills", 0),
        ("team_b_deaths", 0),
    ):
        assert _metric(info, name) == expected


def test_simultaneous_threshold_deaths_are_a_draw_with_both_kills(
    scalar_step: Step,
) -> None:
    state = _scenario_start(make("tdm"), scores=(19, 19), max_steps=3, mutual=True)
    action = _idle()._replace(
        select_target=jnp.zeros(10, jnp.int32).at[jnp.asarray((0, 5))].set(6)
    )
    assert bool(state.action_mask.select_target_use_ultimate_joint_mask[0, 6, 0])
    assert bool(state.action_mask.select_target_use_ultimate_joint_mask[5, 6, 0])
    _, successor, _, done, info = scalar_step(jax.random.key(2), state, action)
    assert bool(done.terminated) and not bool(done.truncated)
    np.testing.assert_array_equal(successor.core_state.team_deathmatch_scores, (20, 20))
    assert _metric(info, "team_a_draw") == _metric(info, "team_b_draw") == 1
    assert _metric(info, "team_a_kills") == _metric(info, "team_b_kills") == 1
    assert _metric(info, "team_a_return") == _metric(info, "team_b_return") == 0


def test_chunked_scan_preserves_metrics_and_emits_completion_once() -> None:
    env = make("tdm")
    _, initial = env.reset(jax.random.key(0), _config(), episode_id=6)
    keys = jax.random.split(jax.random.key(99), 4)

    def advance(
        state: EnvironmentState, key: Array
    ) -> tuple[EnvironmentState, EpisodeInfo]:
        _, successor, _, _, info = env.step(key, state, _idle())
        return successor, info

    def scan_steps(
        state: EnvironmentState, keys: Array
    ) -> tuple[EnvironmentState, EpisodeInfo]:
        return jax.lax.scan(advance, state, keys)

    scan = cast(
        Callable[[EnvironmentState, Array], tuple[EnvironmentState, EpisodeInfo]],
        jax.jit(scan_steps),
    )
    whole_state, whole_info = scan(initial, keys)
    middle, first = scan(initial, keys[:2])
    chunk_state, second = scan(middle, keys[2:])

    def concatenate(left: Array, right: Array) -> Array:
        return jnp.concatenate((left, right))

    _assert_tree_exact(chunk_state, whole_state)
    _assert_tree_exact(jax.tree.map(concatenate, first, second), whole_info)
    np.testing.assert_array_equal(whole_info.completed, (False, False, True, False))
    assert _metric(_row(whole_info, 2), "episode_length") == 3


def test_none_mode_has_no_optional_statistics_and_preserves_core_transition(
    scalar_step: Step,
) -> None:
    key = jax.random.key(4)
    config = _config(max_steps=1)
    env = make("tdm", metrics="none")
    _, disabled = env.reset(key, config, episode_id=1)
    _, enabled = make("tdm").reset(key, config, episode_id=1)
    assert disabled.priority is None
    assert disabled.full is None and enabled.full is None
    actual = cast(Step, jax.jit(env.step))(key, disabled, _idle())
    expected = scalar_step(key, enabled, _idle())
    _assert_tree_exact(actual[0], expected[0])
    _assert_tree_exact(actual[1].core_state, expected[1].core_state)
    _assert_tree_exact(actual[2:4], expected[2:4])
    assert actual[1].priority is None and actual[4].priority is None
    assert actual[1].full is None and actual[4].full is None
    assert bool(actual[4].completed)
    assert int(actual[4].outcome) == 3
    assert "core_info" not in actual[4]._fields


@pytest.mark.parametrize("mode", ("none", "priority"))
def test_native_selected_metrics_match_full_across_chunks_and_resets(
    mode: Literal["none", "priority"],
) -> None:
    key, config = jax.random.key(19), _config(max_steps=2)
    keys = jax.random.split(jax.random.key(20), 2)
    ids = jnp.asarray((11, 12), jnp.int32)
    full = make("tdm", num_envs=2, metrics="full")
    selected = make("tdm", num_envs=2, metrics=mode, full_metrics_episodes=(12, 13))
    scalar = make("tdm", metrics="full")
    _, full_state = full.reset(key, config, episode_id=ids)
    _, selected_state = selected.reset(key, config, episode_id=ids)
    _, scalar_state = scalar.reset(jax.random.fold_in(key, 12), config, episode_id=12)
    assert selected_state.full is not None
    unselected_initial = _row(selected_state.full, 0)
    full_step = cast(Step, jax.jit(full.step))
    scalar_full_step = cast(Step, jax.jit(scalar.step))

    def advance(
        state: EnvironmentState, step_key: Array
    ) -> tuple[EnvironmentState, EpisodeInfo]:
        _, successor, _, _, info = selected.step(step_key, state, _idle(2))
        return successor, info

    def scan_chunk(
        state: EnvironmentState, chunk_keys: Array
    ) -> tuple[EnvironmentState, EpisodeInfo]:
        return jax.lax.scan(advance, state, chunk_keys)

    scan = cast(
        Callable[[EnvironmentState, Array], tuple[EnvironmentState, EpisodeInfo]],
        jax.jit(scan_chunk),
    )
    for transition in range(2):
        step_key = keys[transition]
        selected_state, chunk_info = scan(
            selected_state, keys[transition : transition + 1]
        )
        info = _row(chunk_info, 0)
        _, full_state, _, _, full_info = full_step(step_key, full_state, _idle(2))
        _, scalar_state, _, _, scalar_info = scalar_full_step(
            jax.random.fold_in(step_key, 12), scalar_state, _idle()
        )
        np.testing.assert_array_equal(info.completed, (transition == 1,) * 2)
        assert info.full is not None and info.priority is not None
        assert full_info.full is not None and scalar_info.full is not None
        _assert_tree_exact(_row(info.full, 1), _row(full_info.full, 1))
        _assert_tree_exact(_row(info.full, 1), scalar_info.full)
        _assert_tree_exact(_row(selected_state.core_state, 1), scalar_state.core_state)
        assert selected_state.full is not None
        _assert_tree_exact(_row(selected_state.full, 0), unselected_initial)
        assert not bool(info.full.valid[0].any())
        assert bool(info.priority.valid[0].any()) == (mode == "priority")

    # Replace the unselected lane with selected ID13, then selected ID12 with ID14.
    replacement = _config(max_steps=1, alternate=True)
    _, reset_first = selected.reset(
        key,
        replacement,
        episode_id=jnp.asarray((13, 99), jnp.int32),
        state=selected_state,
        reset_mask=jnp.asarray((True, False)),
    )
    _assert_tree_exact(_row(reset_first, 1), _row(selected_state, 1))
    _, reset_second = selected.reset(
        key,
        replacement,
        episode_id=jnp.asarray((99, 14), jnp.int32),
        state=reset_first,
        reset_mask=jnp.asarray((False, True)),
    )
    _assert_tree_exact(_row(reset_second, 0), _row(reset_first, 0))
    np.testing.assert_array_equal(reset_second.collect_full_metrics, (True, False))
    assert reset_second.full is not None
    deselected_initial = _row(reset_second.full, 1)
    successor, chunk_info = scan(reset_second, keys[:1])
    info = _row(chunk_info, 0)
    assert info.full is not None and successor.full is not None
    np.testing.assert_array_equal(info.completed, (True, True))
    np.testing.assert_array_equal(info.episode_id, (13, 14))
    _assert_tree_exact(_row(successor.full, 1), deselected_initial)
    assert not bool(info.full.valid[1].any())
    _, fresh = scalar.reset(jax.random.fold_in(key, 13), replacement, episode_id=13)
    _, _, _, _, expected_info = scalar_full_step(
        jax.random.fold_in(keys[0], 13), fresh, _idle()
    )
    _assert_tree_exact(_row(info.full, 0), expected_info.full)


def test_partial_reset_restores_source_choices_only_in_reset_lanes() -> None:
    env = make("tdm", num_envs=2, metrics="none")
    config = _config()
    _, state = env.reset(jax.random.key(0), config, episode_id=jnp.asarray([1, 2]))
    default = state.source_availability
    state = state._replace(source_availability=jnp.zeros_like(default))
    observations = env.get_observations(state)
    np.testing.assert_array_equal(observations.source_availability, 0)
    observations, state = cast(
        tuple[Observations, EnvironmentState],
        jax.jit(env.reset)(
            jax.random.key(1),
            config,
            episode_id=jnp.asarray([3, 4]),
            state=state,
            reset_mask=jnp.asarray([True, False]),
        ),
    )
    np.testing.assert_array_equal(state.source_availability[0], default[0])
    np.testing.assert_array_equal(state.source_availability[1], 0)
    np.testing.assert_array_equal(
        observations.source_availability, state.source_availability
    )
    next_observations, successor, _, _, info = cast(Step, jax.jit(env.step))(
        jax.random.key(2), state, _idle(2)
    )
    np.testing.assert_array_equal(
        successor.source_availability, state.source_availability
    )
    np.testing.assert_array_equal(
        next_observations.source_availability, state.source_availability
    )
    assert info.replay is None and info.priority is None and info.full is None
