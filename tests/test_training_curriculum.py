"""Prove exact schedules and guarded compiled stage accounting on CPU.

The independent allocation oracle uses rational arithmetic. Real public
environment/tracker loops prove stage crossings, reset-time labels, terminal
timing, live-game preservation, fixed-lane counts and stopped failure paths.
Early history capture moves only the first self-play threshold to round one and
drops the 100% capture; the default threshold list and report stay unchanged.
These tiny schedules test execution; they do not establish useful learning.
"""

import json
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.test_episode_tracking import (
    _assert_tree,  # pyright: ignore[reportPrivateUsage]
    _idle,  # pyright: ignore[reportPrivateUsage]
    _source,  # pyright: ignore[reportPrivateUsage]
    _step,  # pyright: ignore[reportPrivateUsage]
)

from marl_battlegrounds import make
from marl_battlegrounds.environment import Environment, EnvironmentState, EpisodeInfo
from marl_battlegrounds.episode_tracking import (
    EpisodeTrackingState,
    init_episode_tracking,
    track_episode_step,
)
from marl_battlegrounds.tasks import balanced_spawn_configs
from marl_battlegrounds.training.curriculum import (
    ScheduleArrays,
    TrainingProgress,
    TrainingSchedule,
    _advance_training_schedule,  # pyright: ignore[reportPrivateUsage]
    _init_training_progress,  # pyright: ignore[reportPrivateUsage]
    make_training_schedule,
)

type LoopCarry = tuple[EnvironmentState, EpisodeTrackingState, TrainingProgress]
type LoopResult = tuple[LoopCarry, tuple[Array, Array, Array]]


@pytest.mark.parametrize("rounds", [40, 55, 100, 550, 12347, 2**31 - 1])
def test_exact_curriculum_allocation_matches_independent_rational_oracle(
    rounds: int,
) -> None:
    schedule = make_training_schedule(
        total_env_steps=rounds * 4, num_envs=4, curriculum=True
    )
    shares = [Fraction(4, 100)] * 5 + [Fraction(20, 1100)] * 11 + [Fraction(60, 100)]
    quotas = [rounds * share for share in shares]
    expected = [int(quota) for quota in quotas]
    residuals = [quota - whole for quota, whole in zip(quotas, expected, strict=True)]
    for _ in range(rounds - sum(expected)):
        winner = residuals.index(max(residuals))
        expected[winner] += 1
        residuals[winner] = Fraction(-1)
    np.testing.assert_array_equal(schedule.arrays.round_budgets, expected)
    np.testing.assert_array_equal(schedule.arrays.round_ends, np.cumsum(expected))
    assert sum(expected) == rounds
    assert min(expected) > 0
    assert schedule.total_env_steps == rounds * 4
    assert int(schedule.arrays.total_rounds) == rounds
    assert int(schedule.arrays.stage_count) == 17
    np.testing.assert_array_equal(schedule.arrays.team_sizes[:5], np.arange(1, 6))
    np.testing.assert_array_equal(schedule.arrays.team_sizes[5:], 5)
    for index, eligible in enumerate(np.asarray(schedule.arrays.eligible_maps)):
        count = 1 if index < 5 else index - 3 if index < 16 else 42
        np.testing.assert_array_equal(eligible, np.arange(42) < count)
    np.testing.assert_array_equal(
        schedule.arrays.history_threshold_rounds,
        [(rounds * index + 19) // 20 for index in range(1, 21)],
    )
    report = dict(schedule.rounding_report)
    assert report["stage_shares"] == tuple((v.numerator, v.denominator) for v in shares)
    assert report["ideal_round_counts"] == tuple(
        (v.numerator, v.denominator) for v in quotas
    )
    assert report["assigned_round_counts"] == tuple(expected)
    assert json.loads(json.dumps(report))["total_env_steps"] == rounds * 4


def test_early_history_capture_moves_first_threshold_to_round_one() -> None:
    plain = make_training_schedule(total_env_steps=80, num_envs=4)
    early = make_training_schedule(
        total_env_steps=80, num_envs=4, early_history_capture=True
    )
    expected = [(20 * index + 19) // 20 for index in range(1, 21)]
    np.testing.assert_array_equal(plain.arrays.history_threshold_rounds, expected)
    np.testing.assert_array_equal(
        early.arrays.history_threshold_rounds, [1, *expected[:-1]]
    )
    assert not plain.early_history_capture
    assert early.early_history_capture
    assert "history_thresholds" not in plain.rounding_report
    assert early.rounding_report["history_thresholds"] == (
        "first update, then 5% through 95%"
    )
    assert dict(plain.rounding_report) == {
        key: value
        for key, value in early.rounding_report.items()
        if key != "history_thresholds"
    }
    for left, right in zip(plain.arrays[:6], early.arrays[:6], strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(
        plain.arrays.score_thresholds, early.arrays.score_thresholds
    )
    with pytest.raises(TypeError, match="early_history_capture"):
        make_training_schedule(
            total_env_steps=80, num_envs=4, early_history_capture=cast(bool, 1)
        )


def test_exact_ties_favor_earlier_stages_and_report_is_immutable() -> None:
    schedule = make_training_schedule(total_env_steps=110, num_envs=2, curriculum=True)
    np.testing.assert_array_equal(schedule.arrays.round_budgets[:5], [3, 2, 2, 2, 2])
    with pytest.raises(FrozenInstanceError):
        schedule.num_envs = 4  # type: ignore[misc]
    with pytest.raises(TypeError):
        schedule.rounding_report["num_envs"] = 4  # type: ignore[index]


@pytest.mark.parametrize("rounds", [1, 2**31 - 1])
def test_plain_schedule_keeps_fixed_capacity_and_large_host_total(rounds: int) -> None:
    schedule = make_training_schedule(total_env_steps=rounds * 4, num_envs=4)
    arrays = schedule.arrays
    assert int(arrays.stage_count) == 1
    np.testing.assert_array_equal(arrays.round_budgets, [rounds] + [0] * 16)
    np.testing.assert_array_equal(arrays.round_ends, rounds)
    np.testing.assert_array_equal(arrays.team_sizes, 5)
    np.testing.assert_array_equal(arrays.eligible_maps, True)
    assert schedule.rounding_report["stage_shares"] == ((1, 1),)


@pytest.mark.parametrize(
    ("total", "batch", "curriculum", "error"),
    [
        (True, 2, False, TypeError),
        (2, True, False, TypeError),
        (2.0, 2, False, TypeError),
        (2, 2.0, False, TypeError),
        (np.int64(2), 2, False, TypeError),
        (2, np.int32(2), False, TypeError),
        (2, 2, 1, TypeError),
        (0, 2, False, ValueError),
        (-2, 2, False, ValueError),
        (2, 0, False, ValueError),
        (2, -2, False, ValueError),
        (3, 3, False, ValueError),
        (3, 2, False, ValueError),
        (2**32, 2, False, ValueError),
        (34, 2, True, ValueError),
        (100, 2, True, ValueError),
    ],
)
def test_malformed_or_empty_stage_budgets_fail_before_collection(
    total: object, batch: object, curriculum: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        make_training_schedule(
            total_env_steps=cast(int, total),
            num_envs=cast(int, batch),
            curriculum=cast(bool, curriculum),
        )


def _setup_loop(schedule: TrainingSchedule) -> tuple[Environment, LoopCarry]:
    batch = schedule.num_envs
    sources = [_source(1 if lane % 2 == 0 else 100) for lane in range(batch)]
    bank = jax.tree.map(lambda *leaves: jnp.stack(leaves), *sources)
    env = make("tdm", num_envs=batch, metrics="none")
    _, state = env.reset(
        jax.random.key(0), balanced_spawn_configs(bank, num_envs=batch)
    )
    tracking = init_episode_tracking(
        env,
        state,
        source_configs=bank,
        source_indices=jnp.arange(batch, dtype=jnp.int32),
    )
    tracking = tracking.begin_stage(
        state, total_env_steps=int(schedule.arrays.round_budgets[0]) * batch
    )
    return env, (state, tracking, _init_training_progress(num_envs=batch))


def _public_scan(
    env: Environment, carry: LoopCarry, schedule: ScheduleArrays, root: Array
) -> tuple[LoopCarry, tuple[Array, Array, Array]]:
    def tick(
        current: LoopCarry, tick_index: Array
    ) -> tuple[LoopCarry, tuple[Array, Array, Array]]:
        state, tracking, progress = current
        active = (
            (progress.rounds < schedule.total_rounds)
            & ~jnp.any(tracking.error_flags)
            & ~jnp.any(state.lifecycle_error)
        )

        def real() -> tuple[LoopCarry, tuple[Array, Array, Array]]:
            key = jax.random.fold_in(root, tick_index)
            reset = state.done.done
            _, decision = env.reset_done(key, state)
            selected_progress = progress._replace(
                episode_stage=jnp.where(
                    reset, tracking.stage_ordinal, progress.episode_stage
                )
            )
            tracked, result = track_episode_step(
                tracking, decision, env.step(key, decision, _idle(decision))
            )
            advanced, tracked, info = _advance_training_schedule(
                selected_progress,
                tracked,
                decision,
                result[1],
                result[4],
                schedule=schedule,
            )
            assert info.episode_tracking_error is not None
            return (result[1], tracked, advanced), (
                selected_progress.episode_stage,
                tracked.stage_ordinal,
                jnp.all(info.episode_tracking_error == 0),
            )

        def padding() -> tuple[LoopCarry, tuple[Array, Array, Array]]:
            return current, (
                progress.episode_stage,
                tracking.stage_ordinal,
                jnp.asarray(False),
            )

        return cast(LoopResult, jax.lax.cond(active, real, padding))

    return jax.lax.scan(tick, carry, jnp.arange(128, dtype=jnp.int32))


_compiled_loop = cast(
    Callable[[Environment, LoopCarry, ScheduleArrays, Array], LoopResult],
    jax.jit(_public_scan),
)
_schedule_jit = jax.jit(_advance_training_schedule)


def _compiled_schedule(
    progress: TrainingProgress,
    tracking: EpisodeTrackingState,
    before: EnvironmentState,
    after: EnvironmentState,
    info: EpisodeInfo,
    *,
    schedule: ScheduleArrays,
) -> tuple[TrainingProgress, EpisodeTrackingState, EpisodeInfo]:
    return cast(
        tuple[TrainingProgress, EpisodeTrackingState, EpisodeInfo],
        _schedule_jit(progress, tracking, before, after, info, schedule=schedule),
    )


@pytest.mark.parametrize("batch", [2, 4])
def test_real_t128_loop_crosses_all_stages_and_keeps_reset_time_exposure(
    batch: int,
) -> None:
    schedule = make_training_schedule(
        total_env_steps=40 * batch, num_envs=batch, curriculum=True
    )
    env, carry = _setup_loop(schedule)
    final, (distributions, requested, valid) = _compiled_loop(
        env, carry, schedule.arrays, jax.random.key(4)
    )
    state, tracking, progress = final
    np.testing.assert_array_equal(tracking.error_flags, 0)
    np.testing.assert_array_equal(progress.stage_complete, True)
    np.testing.assert_array_equal(valid, np.arange(128) < 40)
    np.testing.assert_array_equal(progress.completed_stage_counts.sum(axis=(0, 2)), 40)
    np.testing.assert_array_equal(
        progress.exposure.sum(axis=0), tracking.accounted_transition_count
    )
    np.testing.assert_array_equal(
        progress.starts.sum(axis=0), state.reset_generation + 1
    )
    np.testing.assert_array_equal(progress.actor_decisions[:, :, 0], progress.exposure)
    np.testing.assert_array_equal(progress.actor_decisions[:, :, 1:], 0)
    np.testing.assert_array_equal(progress.map_steps, 0)
    np.testing.assert_array_equal(progress.opponent_steps, 0)
    np.testing.assert_array_equal(progress.opponent_starts, 0)
    assert int(progress.rounds) == 40
    assert int(tracking.stage_ordinal) == 16
    assert tracking.stage_summary(state)["status"] == "complete"
    # The one-tick lane ends at the boundary and resets under the new request.
    np.testing.assert_array_equal(distributions[:17, 0], np.arange(17))
    np.testing.assert_array_equal(requested[:16], np.arange(1, 17))
    # A hundred-tick game continues unchanged through all accounting stages.
    np.testing.assert_array_equal(distributions[:, 1], 0)
    assert int(progress.exposure[0, 1]) == 40
    assert int(progress.starts[0, 1]) == 1
    assert int(state.reset_generation[1]) == 0
    assert int(state.reset_generation[0]) == 39
    _assert_tree(state.config, carry[0].config)
    again, (_, _, valid_again) = _compiled_loop(
        env, final, schedule.arrays, jax.random.key(9)
    )
    _assert_tree(again, final)
    np.testing.assert_array_equal(valid_again, False)


def test_compiled_handoff_matches_checked_host_handoff() -> None:
    schedule = make_training_schedule(total_env_steps=80, num_envs=2, curriculum=True)
    env, (state, tracking, progress) = _setup_loop(schedule)
    tracked, result = track_episode_step(tracking, state, _step(env, state))
    host_next = tracked.begin_stage(result[1], total_env_steps=2)
    next_progress, compiled_next, info = _compiled_schedule(
        progress, tracked, state, result[1], result[4], schedule=schedule.arrays
    )
    _assert_tree(compiled_next, host_next)
    np.testing.assert_array_equal(
        next_progress.completed_stage_counts[0], tracked.stage_counts
    )
    assert bool(next_progress.stage_complete[0])
    np.testing.assert_array_equal(info.episode_tracking_error, 0)


@pytest.mark.parametrize(
    ("fault", "bit"),
    [
        ("counts", 2),
        ("full", 2),
        ("unknown_spawn", 2),
        ("swapped_lane", 2),
        ("budget", 2),
        ("rounds", 2),
        ("next_budget", 2),
        ("lifecycle", 2),
        ("info_lifecycle", 2),
        ("pending", 1),
        ("info_pending", 1),
        ("room", 4),
        ("ordinal", 4),
        ("exposure", 4),
        ("starts", 4),
        ("actors", 4),
        ("round_counter", 4),
    ],
)
def test_boundary_failure_preserves_evidence_games_and_stops_later_actions(
    fault: str, bit: int
) -> None:
    schedule = make_training_schedule(total_env_steps=80, num_envs=2, curriculum=True)
    env, (before, tracking, progress) = _setup_loop(schedule)
    tracking, result = track_episode_step(tracking, before, _step(env, before))
    after, info = result[1], result[4]
    arrays = schedule.arrays
    if fault == "counts":
        tracking = replace(tracking, accounted_transition_count=jnp.zeros(2, jnp.int32))
    elif fault == "full":
        tracking = replace(tracking, full_batch_rounds_valid=jnp.asarray(False))
    elif fault == "unknown_spawn":
        tracking = replace(
            tracking, stage_counts=jnp.array([[0, 0, 1], [0, 1, 0]], jnp.int32)
        )
    elif fault == "swapped_lane":
        tracking = replace(tracking, stage_counts=tracking.stage_counts[::-1])
    elif fault == "budget":
        tracking = replace(tracking, stage_round_budget=jnp.int32(2))
    elif fault == "rounds":
        tracking = replace(tracking, stage_rounds=jnp.int32(2))
    elif fault == "next_budget":
        arrays = arrays._replace(round_budgets=arrays.round_budgets.at[1].set(0))
    elif fault == "lifecycle":
        after = after._replace(lifecycle_error=jnp.array([True, False]))
    elif fault == "info_lifecycle":
        info = info._replace(lifecycle_error=jnp.array([False, True]))
    elif fault == "pending":
        tracking = replace(tracking, error_flags=jnp.array([1, 0], jnp.int32))
    elif fault == "info_pending":
        info = info._replace(episode_tracking_error=jnp.array([0, 1], jnp.int32))
    elif fault == "room":
        arrays = arrays._replace(
            round_budgets=arrays.round_budgets.at[1].set(2**31 - 1)
        )
    elif fault == "ordinal":
        tracking = replace(tracking, stage_ordinal=jnp.int32(2**31 - 1))
    elif fault == "exposure":
        progress = progress._replace(exposure=progress.exposure.at[0, 0].set(2**31 - 1))
    elif fault == "starts":
        progress = progress._replace(starts=progress.starts.at[0, 0].set(2**31 - 1))
    elif fault == "round_counter":
        progress = progress._replace(rounds=jnp.int32(2**31 - 1))
    else:
        progress = progress._replace(
            actor_decisions=progress.actor_decisions.at[0, 0, 0].set(2**31 - 1)
        )
    updated, failed, failed_info = _compiled_schedule(
        progress, tracking, before, after, info, schedule=arrays
    )
    assert np.any(np.asarray(failed.error_flags) & bit)
    np.testing.assert_array_equal(
        failed_info.episode_tracking_error, failed.error_flags
    )
    _assert_tree(updated, progress)
    _assert_tree(replace(failed, error_flags=tracking.error_flags), tracking)
    stopped, (_, _, valid) = _compiled_loop(
        env, (after, failed, updated), arrays, jax.random.key(5)
    )
    _assert_tree(stopped, (after, failed, updated))
    np.testing.assert_array_equal(valid, False)


def test_padding_cannot_add_exposure_or_complete_a_stage() -> None:
    schedule = make_training_schedule(total_env_steps=4, num_envs=2)
    env, (before, tracking, progress) = _setup_loop(schedule)
    tracking, result = track_episode_step(tracking, before, _step(env, before))
    progress, tracking, _ = _compiled_schedule(
        progress, tracking, before, result[1], result[4], schedule=schedule.arrays
    )
    before = result[1]
    tracking, result = track_episode_step(tracking, before, _step(env, before))
    updated, failed, _ = _compiled_schedule(
        progress, tracking, before, result[1], result[4], schedule=schedule.arrays
    )
    _assert_tree(updated, progress)
    np.testing.assert_array_equal(updated.stage_complete, False)
    assert np.any(np.asarray(failed.error_flags) & 2)


def test_same_shaped_schedule_keys_and_counter_values_reuse_one_trace() -> None:
    schedule = make_training_schedule(total_env_steps=80, num_envs=2, curriculum=True)
    changed = make_training_schedule(total_env_steps=110, num_envs=2, curriculum=True)
    env, carry = _setup_loop(schedule)
    changed_env, changed_carry = _setup_loop(changed)
    traces: list[None] = []

    def counted(
        env: Environment, carry: LoopCarry, arrays: ScheduleArrays, root: Array
    ) -> tuple[LoopCarry, tuple[Array, Array, Array]]:
        traces.append(None)
        return _public_scan(env, carry, arrays, root)

    compiled = cast(
        Callable[[Environment, LoopCarry, ScheduleArrays, Array], LoopResult],
        jax.jit(counted),
    )
    first = compiled(env, carry, schedule.arrays, jax.random.key(1))
    second = compiled(changed_env, changed_carry, changed.arrays, jax.random.key(2))
    third = compiled(env, first[0], schedule.arrays, jax.random.key(3))
    jax.block_until_ready((first, second, third))
    assert len(traces) == 1
    assert int(second[0][2].rounds) == 55
    _assert_tree(third[0], first[0])


def test_actor_exposure_uses_pre_action_alive_status() -> None:
    schedule = make_training_schedule(total_env_steps=4, num_envs=2)
    env, (before, tracking, progress) = _setup_loop(schedule)
    before = before._replace(
        core_state=before.core_state._replace(
            alive_mask=before.core_state.alive_mask.at[0, 0].set(False)
        )
    )
    tracking, result = track_episode_step(tracking, before, _step(env, before))
    updated, tracking, _ = _compiled_schedule(
        progress, tracking, before, result[1], result[4], schedule=schedule.arrays
    )
    np.testing.assert_array_equal(tracking.error_flags, 0)
    np.testing.assert_array_equal(updated.actor_decisions[0, :, 0], [0, 1])
    np.testing.assert_array_equal(updated.exposure[0], 1)


def test_compiled_final_stage_keeps_large_totals_without_whole_batch_overflow() -> None:
    maximum = 2**31 - 1
    schedule = make_training_schedule(total_env_steps=maximum * 4, num_envs=4)
    env, (before, tracking, progress) = _setup_loop(schedule)
    before = before._replace(
        cumulative_transition_count=jnp.full(4, maximum - 1, jnp.int32)
    )
    category = np.arange(4) >= 2
    counts = np.zeros((4, 3), np.int32)
    counts[np.arange(4), category.astype(np.int32)] = maximum - 1
    tracking = replace(
        tracking,
        accounted_transition_count=before.cumulative_transition_count,
        stage_rounds=jnp.int32(maximum - 1),
        stage_counts=jnp.asarray(counts),
    )
    progress = progress._replace(
        rounds=jnp.int32(maximum - 1),
        exposure=progress.exposure.at[0].set(maximum - 1),
        actor_decisions=progress.actor_decisions.at[0, :, 0].set(maximum - 1),
    )
    tracking, result = track_episode_step(tracking, before, _step(env, before))
    updated, tracking, info = _compiled_schedule(
        progress, tracking, before, result[1], result[4], schedule=schedule.arrays
    )
    np.testing.assert_array_equal(tracking.error_flags, 0)
    np.testing.assert_array_equal(info.episode_tracking_error, 0)
    assert int(updated.rounds) == maximum
    assert bool(updated.stage_complete[0])
    assert int(tracking.stage_ordinal) == 0
    np.testing.assert_array_equal(updated.exposure[0], maximum)
    assert (
        sum(int(v) for v in np.asarray(updated.completed_stage_counts).flat)
        == maximum * 4
    )
    assert tracking.stage_summary(result[1])["env_steps"] == maximum * 4
