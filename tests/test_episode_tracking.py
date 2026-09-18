"""Check exact source bindings, real-step accounting and compact episode starts.

These tests exercise reset provenance, declarations, actual terminal/padded
transitions, compiled updates and failure preservation without changing Core.
"""

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds import make
from marl_battlegrounds.core.types import Action, EnvConfig
from marl_battlegrounds.environment import Environment, EnvironmentState
from marl_battlegrounds.episode_tracking import (
    EpisodeTrackingState,
    StepResult,
    init_episode_tracking,
    track_episode_step,
)
from marl_battlegrounds.tasks import (
    balanced_spawn_configs,
    make_standard_team_deathmatch_config,
)


def _source(max_steps: int = 2) -> EnvConfig:
    return make_standard_team_deathmatch_config(
        map_id=0,
        max_steps=max_steps,
        team_a_roster=("mage",),
        team_b_roster=("priest",),
    )


def _idle(state: EnvironmentState) -> Action:
    zeros = jnp.zeros_like(state.config.agent_profile.class_ids)
    return Action(zeros, zeros, zeros)


def _numerical_step(env: Environment, state: EnvironmentState) -> StepResult:
    return env.step(jax.random.key(12), state, _idle(state))


_compiled_step = jax.jit(_numerical_step)
_compiled_tracking = jax.jit(track_episode_step)


def _step(env: Environment, state: EnvironmentState) -> StepResult:
    return cast(StepResult, _compiled_step(env, state))


def _track_jit(
    tracker: EpisodeTrackingState,
    state: EnvironmentState,
    result: StepResult,
    *,
    source_indices: object = None,
) -> tuple[EpisodeTrackingState, StepResult]:
    return cast(
        tuple[EpisodeTrackingState, StepResult],
        _compiled_tracking(tracker, state, result, source_indices=source_indices),
    )


def _setup(
    batch: int | None = 2, *, record: bool = False
) -> tuple[Environment, EnvironmentState, EpisodeTrackingState]:
    env = make("tdm", env_config=_source(), num_envs=batch, metrics="none")
    config = (
        _source()
        if batch is None or batch % 2
        else balanced_spawn_configs(_source(), num_envs=batch)
    )
    _, state = env.reset(jax.random.key(0), config)
    tracker = init_episode_tracking(
        env, state, source_configs=_source(), source_indices=0, record_starts=record
    )
    return env, state, tracker


def _assert_tree(actual: object, expected: object) -> None:
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("batch", [None, 1, 2, 3])
def test_raw_tracking_counts_real_terminal_and_padding(batch: int | None) -> None:
    env, state, tracker = _setup(batch)
    for index in range(3):
        result = _step(env, state)
        tracker, tracked = _track_jit(tracker, state, result)
        _assert_tree(tracked[:4], result[:4])
        state = result[1]
        np.testing.assert_array_equal(
            tracker.accounted_transition_count, min(index + 1, 2)
        )
        np.testing.assert_array_equal(tracker.error_flags, 0)
    assert tracker.stage_summary(state)["env_steps"] == 2 * (batch or 1)
    assert tracker.source_configs is not None
    assert tracker.source_configs.agent_profile.active_mask.shape == (1, 10)


def test_constructor_source_eligibility_and_explicit_unknown_bank() -> None:
    env = make("tdm", map_id=0, num_envs=2, metrics="none")
    _, state = env.reset(jax.random.key(0))
    known = init_episode_tracking(env, state)
    np.testing.assert_array_equal(known.source_index, [0, 0])
    np.testing.assert_array_equal(known.spawn_locations, [0, 1])
    explicit = init_episode_tracking(env, state, source_configs=known.source_configs)
    np.testing.assert_array_equal(explicit.source_index, [-1, -1])
    _, overridden = env.reset(
        jax.random.key(1),
        state.config,
        state=state,
        reset_mask=jnp.array([True, False]),
    )
    inferred = init_episode_tracking(env, overridden)
    np.testing.assert_array_equal(inferred.source_index, [-1, 0])


def test_reuse_retains_binding_but_zero_step_override_clears_it() -> None:
    env, state, tracker = _setup()
    _, overridden = env.reset(
        jax.random.key(1),
        state.config,
        state=state,
        reset_mask=jnp.array([True, False]),
    )
    _, reused = env.reset(
        jax.random.key(2), state=overridden, reset_mask=jnp.array([True, False])
    )
    tracker, result = track_episode_step(tracker, reused, _step(env, reused))
    np.testing.assert_array_equal(tracker.source_index, [-1, 0])
    np.testing.assert_array_equal(tracker.error_flags, 0)
    assert tracker.stage_summary(result[1])["unreported_spawn_steps"] == 1


def test_new_declaration_identifies_override_and_ignores_continuing_lane_values() -> (
    None
):
    env, state, tracker = _setup()
    _, reset = env.reset(
        jax.random.key(1),
        state.config,
        state=state,
        reset_mask=jnp.array([True, False]),
    )
    tracker, result = track_episode_step(
        tracker, reset, _step(env, reset), source_indices=np.array([0, 999], np.int64)
    )
    np.testing.assert_array_equal(tracker.source_index, [0, 0])
    np.testing.assert_array_equal(tracker.error_flags, 0)
    assert tracker.stage_summary(result[1])["env_steps"] == 2


@pytest.mark.parametrize(
    "indices", [True, 0.0, [0, 0, 0], np.array([2**32, 0], np.uint64), [-2, 0], [1, 0]]
)
def test_invalid_initial_source_indices_fail_before_narrowing(indices: object) -> None:
    env, state, _ = _setup()
    with pytest.raises((TypeError, ValueError)):
        init_episode_tracking(
            env, state, source_configs=_source(), source_indices=indices
        )


def test_traced_invalid_index_stays_an_error_without_balance_credit() -> None:
    env, state, tracker = _setup()
    _, reset = env.reset(jax.random.key(1), state=state)
    tracker, result = _track_jit(
        tracker,
        reset,
        _step(env, reset),
        source_indices=jnp.array([0, 2**32 - 1], jnp.uint32),
    )
    assert int(tracker.error_flags[1]) & 1
    np.testing.assert_array_equal(result[4].episode_tracking_error, tracker.error_flags)
    with pytest.raises(ValueError, match="accounting failed"):
        tracker.stage_summary(result[1])


def test_relationship_mismatch_is_declaration_error() -> None:
    env, state, tracker = _setup()
    changed = state.config._replace(max_steps=jnp.array([3, 2], jnp.int32))
    _, reset = env.reset(
        jax.random.key(1), changed, state=state, reset_mask=jnp.array([True, False])
    )
    tracker, _ = track_episode_step(tracker, reset, _step(env, reset), source_indices=0)
    assert int(tracker.error_flags[0]) & 1
    assert int(tracker.error_flags[1]) == 0


@pytest.mark.parametrize(
    "mode", ["duplicate", "skip", "wrong_id", "wrong_epoch", "lifecycle"]
)
def test_bad_accounting_keeps_last_valid_endpoint(mode: str) -> None:
    env, state, tracker = _setup()
    result = _step(env, state)
    if mode == "duplicate":
        tracker, _ = track_episode_step(tracker, state, result)
    elif mode == "skip":
        state = result[1]
        result = _step(env, state)
    elif mode == "wrong_id":
        result = (*result[:4], result[4]._replace(episode_id=result[4].episode_id + 2))
    elif mode == "wrong_epoch":
        result = (
            *result[:4],
            result[4]._replace(decision_step=result[4].decision_step + 1),
        )
    else:
        result = (*result[:4], result[4]._replace(lifecycle_error=jnp.ones(2, bool)))
    old_count = tracker.accounted_transition_count
    tracker, tracked = track_episode_step(tracker, state, result)
    np.testing.assert_array_equal(tracker.accounted_transition_count, old_count)
    assert np.all(np.asarray(tracker.error_flags) & 2)
    assert not np.any(np.asarray(tracker.error_flags) & 4)
    np.testing.assert_array_equal(
        tracked[4].episode_tracking_error, tracker.error_flags
    )


def test_mid_episode_attachment_and_recorded_initialization_limit() -> None:
    env, state, _ = _setup()
    state = _step(env, state)[1]
    tracker = init_episode_tracking(
        env, state, source_configs=_source(), source_indices=0
    )
    tracker = tracker.begin_stage(state, total_env_steps=2)
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    np.testing.assert_array_equal(tracker.episode_start_stage, -1)
    assert tracker.stage_summary(result[1])["env_steps"] == 2
    with pytest.raises(ValueError, match="first real transition"):
        init_episode_tracking(env, state, record_starts=True)


def test_recorded_starts_are_once_per_episode_and_preserve_original_stage() -> None:
    env, state, tracker = _setup(record=True)
    tracker = tracker.begin_stage(state, total_env_steps=2)
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    starts = result[4].episode_start_records
    assert starts is not None
    np.testing.assert_array_equal(starts.valid, True)
    np.testing.assert_array_equal(starts.episode_start_stage, 0)
    assert starts.source_table_id.shape == (2, 8)
    state = result[1]
    tracker = tracker.begin_stage(state, total_env_steps=2)
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    starts = result[4].episode_start_records
    assert starts is not None
    np.testing.assert_array_equal(starts.valid, False)
    np.testing.assert_array_equal(tracker.episode_start_stage, 0)
    _, state = env.reset_done(jax.random.key(2), result[1])
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    starts = result[4].episode_start_records
    assert starts is not None
    np.testing.assert_array_equal(starts.valid, True)
    np.testing.assert_array_equal(starts.episode_start_stage, 1)


def test_unknown_starts_have_no_bank_claim_and_disabled_has_no_payload() -> None:
    env, state, tracker = _setup()
    unknown = init_episode_tracking(
        env, state, source_configs=_source(), record_starts=True
    )
    unknown, result = track_episode_step(unknown, state, _step(env, state))
    starts = result[4].episode_start_records
    assert starts is not None
    np.testing.assert_array_equal(starts.source_index, -1)
    np.testing.assert_array_equal(starts.source_table_id, 0)
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    assert result[4].episode_start_records is None
    assert tracker.source_table_id is None


def test_ordinary_steps_skip_relationship_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marl_battlegrounds.episode_tracking as module

    env, state, tracker = _setup()
    result = _step(env, state)
    calls: list[int] = []
    original = module.spawn_locations_for_source

    def check(config: EnvConfig, source: EnvConfig) -> tuple[Array, Array]:
        jax.debug.callback(lambda: calls.append(1))
        return original(config, source)

    monkeypatch.setattr(module, "spawn_locations_for_source", check)
    _compiled_tracking.clear_cache()  # pyright: ignore[reportAttributeAccessIssue]
    tracker, _ = _track_jit(tracker, state, result)
    jax.block_until_ready(tracker)
    assert calls == []
    _, reset = env.reset(jax.random.key(1), state=state)
    tracker = init_episode_tracking(
        env, state, source_configs=_source(), source_indices=0
    )
    calls.clear()
    jax.block_until_ready(_track_jit(tracker, reset, _step(env, reset)))
    assert calls == [1]


def test_counter_overflow_is_sticky_and_never_wraps() -> None:
    env, state, tracker = _setup()
    maximum = np.iinfo(np.int32).max
    tracker = replace(
        tracker, stage_counts=jnp.array([[maximum, 0, 0], [0, maximum, 0]], jnp.int32)
    )
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    np.testing.assert_array_equal(
        tracker.stage_counts, [[maximum, 0, 0], [0, maximum, 0]]
    )
    assert np.all(np.asarray(tracker.error_flags) & 4)
    with pytest.raises(ValueError):
        tracker.stage_summary(result[1])


def test_unused_invalid_spawn_exchange_does_not_block_tracking() -> None:
    from marl_battlegrounds.core.config import resolve_agent_profile

    source = _source()._replace(
        task_mode=0,
        team_deathmatch_score_threshold=0,
        agent_profile=resolve_agent_profile(
            jnp.array([5] + [0] * 9, jnp.int32), jnp.array([1, 0], jnp.int32)
        ),
        obstacles=jnp.zeros_like(_source().obstacles),
        team_spawn_pad_positions=_source().team_spawn_pad_positions.at[1, :, 0].set(0),
    )
    env = make("tdm", env_config=source, metrics="none")
    _, state = env.reset(jax.random.key(0))
    tracker = init_episode_tracking(env, state, source_configs=source, source_indices=0)
    assert bool(tracker.source_known)
    assert int(tracker.spawn_locations) == 0
    with pytest.raises(ValueError, match="swapped spawn"):
        balanced_spawn_configs(source, num_envs=2)


def test_ambiguous_source_is_known_without_spawn_balance_credit() -> None:
    from marl_battlegrounds.core.config import resolve_agent_profile

    source = _source()._replace(
        task_mode=0,
        team_deathmatch_score_threshold=0,
        agent_profile=resolve_agent_profile(
            jnp.zeros(10, jnp.int32), jnp.zeros(2, jnp.int32)
        ),
    )
    source = source._replace(
        team_spawn_pad_positions=source.team_spawn_pad_positions.at[1].set(
            source.team_spawn_pad_positions[0]
        )
    )
    env = make("tdm", env_config=source, metrics="none")
    _, state = env.reset(jax.random.key(0))
    tracker = init_episode_tracking(env, state, source_configs=source, source_indices=0)
    assert bool(tracker.source_known)
    assert int(tracker.spawn_locations) == -1
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    assert tracker.stage_summary(result[1])["unreported_spawn_steps"] == 1


def test_authored_start_uses_local_zero_and_stays_custom() -> None:
    from marl_battlegrounds.core import env as core

    source = _source(4)
    core_state, *_ = core.reset(source, jax.random.key(1))
    core_state = core_state._replace(step_count=jnp.asarray(2, jnp.int32))
    initial = core.initialize_scenario_state(core_state, source)[:3]
    env = make("tdm", env_config=source, metrics="none")
    _, state = env.reset(jax.random.key(0), initial=initial)
    tracker = init_episode_tracking(
        env, state, source_configs=source, source_indices=0, record_starts=True
    )
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    assert int(result[4].decision_step) == 0
    starts = result[4].episode_start_records
    assert starts is not None
    assert bool(starts.authored_start) and bool(starts.source_known)
    assert tracker.stage_summary(result[1])["unreported_spawn_steps"] == 1


def test_external_vmap_tracks_scalar_contexts_without_a_native_axis() -> None:
    env, state, tracker = _setup(None)
    result = _step(env, state)

    def stack(value: Array) -> Array:
        return jnp.stack([value, value])

    trackers = jax.tree.map(stack, tracker)
    states = jax.tree.map(stack, state)
    results = jax.tree.map(stack, result)
    tracked = cast(
        tuple[EpisodeTrackingState, StepResult],
        jax.jit(jax.vmap(track_episode_step))(trackers, states, results),
    )
    np.testing.assert_array_equal(tracked[0].error_flags, 0)
    np.testing.assert_array_equal(tracked[0].accounted_transition_count, [1, 1])
    assert tracked[0].num_envs is None


def test_changing_same_shaped_sources_reuses_compiled_tracking() -> None:
    traces: list[int] = []

    def track(
        t: EpisodeTrackingState, before: EnvironmentState, result: StepResult
    ) -> tuple[EpisodeTrackingState, StepResult]:
        traces.append(1)
        return track_episode_step(t, before, result)

    compiled = jax.jit(track)
    for length in (2, 3):
        source = _source(length)
        env = make("tdm", env_config=source, num_envs=2, metrics="none")
        _, state = env.reset(
            jax.random.key(length), balanced_spawn_configs(source, num_envs=2)
        )
        tracker = init_episode_tracking(
            env, state, source_configs=source, source_indices=0
        )
        tracked = cast(
            tuple[EpisodeTrackingState, StepResult],
            compiled(tracker, state, _step(env, state)),
        )
        np.testing.assert_array_equal(tracked[0].error_flags, 0)
    assert traces == [1]


@pytest.mark.parametrize(
    "field", ["decision_step", "episode_length", "completed", "lifecycle_error"]
)
def test_wrong_result_lane_shapes_fail_instead_of_broadcasting(field: str) -> None:
    env, state, tracker = _setup()
    result = _step(env, state)
    info = result[4]._replace(**{field: getattr(result[4], field)[0]})
    with pytest.raises(ValueError, match="shape"):
        track_episode_step(tracker, state, (*result[:4], info))
