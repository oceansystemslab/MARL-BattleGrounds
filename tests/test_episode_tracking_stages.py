"""Prove exact balanced stages, host boundary checks and numerical continuation.

The cases include unequal episode lengths, padding, missed final transitions,
large host totals, reset-only changes, and restored compiled rollout state.
"""

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.test_episode_tracking import (
    _assert_tree,  # pyright: ignore[reportPrivateUsage]
    _idle,  # pyright: ignore[reportPrivateUsage]
    _setup,  # pyright: ignore[reportPrivateUsage]
    _source,  # pyright: ignore[reportPrivateUsage]
    _step,  # pyright: ignore[reportPrivateUsage]
)

from marl_battlegrounds import make
from marl_battlegrounds.environment import Environment, EnvironmentState
from marl_battlegrounds.episode_tracking import (
    EpisodeTrackingState,
    init_episode_tracking,
    track_episode_step,
)
from marl_battlegrounds.tasks import balanced_spawn_configs


@pytest.mark.parametrize("budget", [True, False, 0, -2, 1, 3, 2.0, 2**33])
def test_invalid_budgets_do_not_clear_live_state(budget: object) -> None:
    _, state, tracker = _setup()
    with pytest.raises(ValueError):
        tracker.begin_stage(state, total_env_steps=cast(int, budget))
    assert int(tracker.stage_ordinal) == -1


@pytest.mark.parametrize("batch", [None, 1, 3])
def test_raw_scalar_and_odd_tracking_cannot_claim_balanced_stage(
    batch: int | None,
) -> None:
    _, state, tracker = _setup(batch)
    with pytest.raises(ValueError, match="even native batch"):
        tracker.begin_stage(state, total_env_steps=6)


def test_stage_completion_preserves_live_games_and_read_only_summary() -> None:
    env, state, tracker = _setup()
    tracker = tracker.begin_stage(state, total_env_steps=2)
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    state = result[1]
    saved = jax.tree.map(jnp.copy, tracker)
    assert tracker.stage_summary(state) == {
        "requested_env_steps": 2,
        "env_steps": 2,
        "default_spawn_steps": 1,
        "swapped_spawn_steps": 1,
        "unreported_spawn_steps": 0,
        "status": "complete",
        "reason": None,
    }
    _assert_tree(tracker, saved)
    next_tracker = tracker.begin_stage(state, total_env_steps=2)
    assert int(next_tracker.stage_ordinal) == 1
    assert int(next_tracker.stage_rounds) == 0
    _assert_tree(
        next_tracker.accounted_transition_count, tracker.accounted_transition_count
    )
    _assert_tree(next_tracker.episode_start_stage, tracker.episode_start_stage)
    np.testing.assert_array_equal(next_tracker.stage_counts, 0)


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("boundary", ["summary", "begin"])
def test_untracked_final_advance_blocks_both_boundaries(
    terminal: bool, boundary: str
) -> None:
    env, state, tracker = _setup()
    tracker = tracker.begin_stage(state, total_env_steps=2)
    if terminal:
        tracker, result = track_episode_step(tracker, state, _step(env, state))
        state = result[1]
    result = _step(env, state)
    state = result[1]
    if terminal:
        _, state = env.reset_done(jax.random.key(3), state)
    with pytest.raises(ValueError, match=r"expected.*observed"):
        if boundary == "summary":
            tracker.stage_summary(state)
        else:
            tracker.begin_stage(state, total_env_steps=2)


def test_reset_only_changes_are_legal_and_incomplete_stage_cannot_be_replaced() -> None:
    env, state, tracker = _setup()
    tracker = tracker.begin_stage(state, total_env_steps=4)
    _, state = env.reset(jax.random.key(3), state=state)
    assert tracker.stage_summary(state)["reason"] == "budget_unmet"
    with pytest.raises(ValueError, match="prior stage"):
        tracker.begin_stage(state, total_env_steps=2)
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    assert tracker.stage_summary(result[1])["env_steps"] == 2


def test_padding_cannot_be_hidden_by_equal_final_spawn_totals() -> None:
    env, state, tracker = _setup()
    tracker = tracker.begin_stage(state, total_env_steps=4)
    for _ in range(3):
        tracker, result = track_episode_step(tracker, state, _step(env, state))
        state = result[1]
    summary = tracker.stage_summary(state)
    assert summary["default_spawn_steps"] == summary["swapped_spawn_steps"] == 2
    assert summary["env_steps"] == summary["requested_env_steps"] == 4
    assert summary["reason"] == "invalid_full_batch_round"


def test_extra_full_round_is_budget_exceeded() -> None:
    env, state, tracker = _setup()
    tracker = tracker.begin_stage(state, total_env_steps=2)
    for _ in range(2):
        tracker, result = track_episode_step(tracker, state, _step(env, state))
        state = result[1]
    assert tracker.stage_summary(state)["reason"] == "budget_exceeded"


@pytest.mark.parametrize("unknown", [True, False])
def test_complete_budget_with_wrong_spawn_coverage_stays_incomplete(
    unknown: bool,
) -> None:
    env, state, _ = _setup()
    if not unknown:
        _, state = env.reset(jax.random.key(2), _source(), state=state)
    tracker = init_episode_tracking(
        env, state, source_configs=_source(), source_indices=None if unknown else 0
    )
    tracker = tracker.begin_stage(state, total_env_steps=2)
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    expected = "unreported_spawn_steps" if unknown else "unequal_spawn_steps"
    assert tracker.stage_summary(result[1])["reason"] == expected


@pytest.mark.parametrize("field", ["error_flags", "lifecycle_error"])
def test_pending_failures_are_rejected_even_at_exact_budget(field: str) -> None:
    env, state, tracker = _setup()
    tracker = tracker.begin_stage(state, total_env_steps=2)
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    state = result[1]
    if field == "error_flags":
        tracker = replace(tracker, error_flags=jnp.array([0, 4], jnp.int32))
    else:
        state = state._replace(lifecycle_error=jnp.array([False, True]))
    with pytest.raises(ValueError, match=r"lanes.*1"):
        tracker.stage_summary(state)


def test_large_host_totals_do_not_overflow_and_bad_counters_fail() -> None:
    _, state, tracker = _setup()
    amount = 2**30 + 1
    state = state._replace(cumulative_transition_count=jnp.full(2, amount, jnp.int32))
    tracker = replace(
        tracker,
        accounted_transition_count=state.cumulative_transition_count,
        stage_ordinal=jnp.array(0, jnp.int32),
        stage_round_budget=jnp.array(amount, jnp.int32),
        stage_rounds=jnp.array(amount, jnp.int32),
        stage_counts=jnp.array([[amount, 0, 0], [0, amount, 0]], jnp.int32),
    )
    summary = tracker.stage_summary(state)
    assert summary["env_steps"] == 2**31 + 2
    assert summary["status"] == "complete"
    with pytest.raises(ValueError, match="remaining int32"):
        tracker.begin_stage(state, total_env_steps=2**31)
    bad = replace(tracker, stage_rounds=jnp.array(amount - 1, jnp.int32))
    with pytest.raises(ValueError, match=r"rounds.*totals"):
        bad.stage_summary(state)


def _scan(
    env: Environment,
    state: EnvironmentState,
    tracker: EpisodeTrackingState,
    keys: Array,
) -> tuple[EnvironmentState, EpisodeTrackingState]:
    def advance(
        carry: tuple[EnvironmentState, EpisodeTrackingState], key: Array
    ) -> tuple[tuple[EnvironmentState, EpisodeTrackingState], None]:
        current, tracking = carry
        _, current = env.reset_done(jax.random.fold_in(key, 1), current)
        tracking, result = track_episode_step(
            tracking, current, env.step(key, current, _idle(current))
        )
        return (result[1], tracking), None

    carry, _ = jax.lax.scan(advance, (state, tracker), keys)
    return carry


def test_unequal_lengths_and_restored_scan_carry_match_uninterrupted_execution() -> (
    None
):
    env = make("tdm", num_envs=2, metrics="none")
    first, second = _source(1), _source(3)

    def stack(a: Array, b: Array) -> Array:
        return jnp.stack([a, b])

    bank = jax.tree.map(stack, first, second)
    selected = balanced_spawn_configs(bank, num_envs=2)
    _, state = env.reset(jax.random.key(0), selected)
    tracker = init_episode_tracking(
        env, state, source_configs=bank, source_indices=jnp.arange(2, dtype=jnp.int32)
    )
    tracker = tracker.begin_stage(state, total_env_steps=12)
    keys = jax.random.split(jax.random.key(7), 6)
    scan = jax.jit(_scan)
    whole = cast(
        tuple[EnvironmentState, EpisodeTrackingState], scan(env, state, tracker, keys)
    )
    first_half = cast(
        tuple[EnvironmentState, EpisodeTrackingState],
        scan(env, state, tracker, keys[:3]),
    )

    def restore(value: object) -> Array:
        return jnp.asarray(value)

    restored = jax.tree.map(restore, jax.device_get(first_half))
    final = cast(
        tuple[EnvironmentState, EpisodeTrackingState], scan(env, *restored, keys[3:])
    )
    _assert_tree(final, whole)
    latest_state, latest_tracker = final
    assert latest_tracker.stage_summary(latest_state)["status"] == "complete"
    assert int(latest_state.reset_generation[0]) == 5
    assert int(latest_state.reset_generation[1]) == 1


def test_alternating_padded_lanes_cannot_earn_balanced_completion() -> None:
    first, second = _source(1), _source(2)

    def stack(a: Array, b: Array) -> Array:
        return jnp.stack([a, b])

    bank = jax.tree.map(stack, first, second)
    env = make("tdm", num_envs=2, metrics="none")
    _, state = env.reset(jax.random.key(0), balanced_spawn_configs(bank, num_envs=2))
    tracker = init_episode_tracking(
        env, state, source_configs=bank, source_indices=jnp.arange(2, dtype=jnp.int32)
    )
    tracker = tracker.begin_stage(state, total_env_steps=4)
    for index in range(3):
        if index == 2:
            _, state = env.reset(
                jax.random.key(3), state=state, reset_mask=jnp.array([True, False])
            )
        tracker, result = track_episode_step(tracker, state, _step(env, state))
        state = result[1]
    summary = tracker.stage_summary(state)
    assert summary["env_steps"] == summary["requested_env_steps"] == 4
    assert summary["default_spawn_steps"] == summary["swapped_spawn_steps"] == 2
    assert summary["reason"] == "invalid_full_batch_round"


def test_exhausted_stage_ordinal_cannot_wrap_on_next_begin() -> None:
    env, state, tracker = _setup()
    tracker = tracker.begin_stage(state, total_env_steps=2)
    tracker, result = track_episode_step(tracker, state, _step(env, state))
    tracker = replace(
        tracker, stage_ordinal=jnp.array(np.iinfo(np.int32).max, jnp.int32)
    )
    with pytest.raises(ValueError, match="ordinal"):
        tracker.begin_stage(result[1], total_env_steps=2)
