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
    _source_config_with_class_ids,  # pyright: ignore[reportPrivateUsage]
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
    source_class_ids: object = None,
) -> tuple[EpisodeTrackingState, StepResult]:
    return cast(
        tuple[EpisodeTrackingState, StepResult],
        _compiled_tracking(
            tracker,
            state,
            result,
            source_indices=source_indices,
            source_class_ids=source_class_ids,
        ),
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
        team_deathmatch_red_zone_depth=0.0,
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
        team_deathmatch_red_zone_depth=0.0,
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


def _roster_setup(
    batch: int | None = 2, *, record: bool = True
) -> tuple[Environment, EnvironmentState, EpisodeTrackingState, Array]:
    source = _source()
    classes = jnp.array([1, 3, 0, 0, 0, 2, 5, 0, 0, 0], jnp.int32)
    resolved, valid = _source_config_with_class_ids(source, classes)
    assert bool(valid)
    config = (
        resolved if batch is None else balanced_spawn_configs(resolved, num_envs=batch)
    )
    env = make("tdm", env_config=source, num_envs=batch, metrics="none")
    _, state = env.reset(jax.random.key(302), config)
    tracker = init_episode_tracking(
        env,
        state,
        source_configs=source,
        source_indices=0,
        source_class_ids=classes,
        record_starts=record,
    )
    return env, state, tracker, classes


@pytest.mark.parametrize("batch", [None, 2])
def test_declared_rosters_emit_once_and_preserve_real_transition_counts(
    batch: int | None,
) -> None:
    env, state, tracker, classes = _roster_setup(batch)
    expected = np.broadcast_to(classes, (() if batch is None else (batch,)) + (10,))
    np.testing.assert_array_equal(tracker.source_class_ids, expected)
    saved_bank = tracker.source_configs
    for tick in range(3):
        tracker, result = _track_jit(tracker, state, _step(env, state))
        starts = result[4].episode_start_records
        assert starts is not None and starts.source_class_ids is not None
        np.testing.assert_array_equal(starts.valid, tick == 0)
        np.testing.assert_array_equal(
            starts.source_class_ids, expected if tick == 0 else -1
        )
        np.testing.assert_array_equal(tracker.error_flags, 0)
        state = result[1]
    _assert_tree(tracker.source_configs, saved_bank)
    assert tracker.stage_summary(state)["env_steps"] == 2 * (batch or 1)


def test_partial_reset_rosters_ignore_continuing_values_and_keep_first_ownership() -> (
    None
):
    env, state, tracker, classes = _roster_setup()
    tracker, result = _track_jit(tracker, state, _step(env, state))
    state = result[1]
    new_classes = jnp.array([4, 0, 0, 0, 0, 1, 0, 0, 0, 0], jnp.int32)
    resolved, _ = _source_config_with_class_ids(_source(), new_classes)
    _, reset = env.reset(
        jax.random.key(303),
        balanced_spawn_configs(resolved, num_envs=2),
        state=state,
        reset_mask=jnp.array([True, False]),
    )
    supplied = np.stack(
        (np.asarray(new_classes, np.int64), np.full(10, 2**40, np.int64))
    )
    tracker, tracked = track_episode_step(
        tracker,
        reset,
        _step(env, reset),
        source_indices=np.array([0, 999]),
        source_class_ids=supplied,
    )
    np.testing.assert_array_equal(
        tracker.source_class_ids, np.stack((new_classes, classes))
    )
    np.testing.assert_array_equal(tracker.error_flags, 0)
    np.testing.assert_array_equal(
        reset.core_state.agent_positions[1], state.core_state.agent_positions[1]
    )
    starts = tracked[4].episode_start_records
    assert starts is not None and starts.source_class_ids is not None
    np.testing.assert_array_equal(starts.valid, [True, False])
    np.testing.assert_array_equal(starts.source_class_ids[0], new_classes)
    np.testing.assert_array_equal(starts.source_class_ids[1], -1)
    assert tracker.stage_summary(tracked[1])["env_steps"] == 4


@pytest.mark.parametrize(
    "mode", ["reuse", "changed_origin", "explicit_original", "explicit_roster"]
)
def test_roster_rebinding_has_explicit_ownership_rules(mode: str) -> None:
    env, state, tracker, classes = _roster_setup()
    config = state.config if mode == "changed_origin" else None
    _, reset = env.reset(jax.random.key(304), config, state=state)
    kwargs: dict[str, object] = {}
    if mode == "explicit_original":
        kwargs["source_indices"] = 0
    elif mode == "explicit_roster":
        kwargs["source_class_ids"] = classes
    tracker, result = track_episode_step(tracker, reset, _step(env, reset), **kwargs)
    if mode == "explicit_original":
        np.testing.assert_array_equal(tracker.error_flags, 1)
        np.testing.assert_array_equal(tracker.stage_counts, 0)
    elif mode == "changed_origin":
        np.testing.assert_array_equal(tracker.source_known, False)
        np.testing.assert_array_equal(tracker.source_class_ids, -1)
        assert tracker.stage_summary(result[1])["unreported_spawn_steps"] == 2
    else:
        np.testing.assert_array_equal(
            tracker.source_class_ids, np.broadcast_to(classes, (2, 10))
        )
        np.testing.assert_array_equal(tracker.error_flags, 0)


@pytest.mark.parametrize(
    "classes",
    [
        np.ones(10, bool),
        np.ones(10, np.float32),
        np.ones(9, np.int32),
        np.full(10, 2**32, np.uint64),
        np.array([1, -1, 0, 0, 0, 2, 0, 0, 0, 0]),
        np.array([1, 0, 2, 0, 0, 2, 0, 0, 0, 0]),
    ],
)
def test_bad_initial_roster_declarations_fail_before_narrowing(classes: object) -> None:
    env, state, _, _ = _roster_setup()
    with pytest.raises((TypeError, ValueError)):
        init_episode_tracking(
            env,
            state,
            source_configs=_source(),
            source_indices=0,
            source_class_ids=classes,
        )


def test_traced_bad_roster_is_sticky_and_receives_no_balance_credit() -> None:
    env, state, tracker, classes = _roster_setup()
    _, reset = env.reset(jax.random.key(305), state=state)
    rows = jnp.broadcast_to(classes, (2, 10)).at[0, 1].set(-1)
    tracker, result = _track_jit(
        tracker,
        reset,
        _step(env, reset),
        source_indices=0,
        source_class_ids=rows,
    )
    np.testing.assert_array_equal(tracker.error_flags, [1, 0])
    np.testing.assert_array_equal(tracker.stage_counts[0], 0)
    assert result[4].episode_start_records is not None
    np.testing.assert_array_equal(result[4].episode_start_records.valid, [False, True])
    tracker, following = _track_jit(tracker, result[1], _step(env, result[1]))
    assert int(tracker.error_flags[0]) & 1
    np.testing.assert_array_equal(tracker.stage_counts[0], 0)
    with pytest.raises(ValueError, match="accounting failed"):
        tracker.stage_summary(following[1])


@pytest.mark.parametrize("change", ["capability", "geometry", "rule", "spawn", "class"])
def test_roster_declarations_cannot_change_other_source_facts(change: str) -> None:
    env, state, _, classes = _roster_setup()
    config = state.config
    if change == "capability":
        config = config._replace(
            agent_profile=config.agent_profile._replace(
                max_health=config.agent_profile.max_health + 1,
            )
        )
    elif change == "geometry":
        config = config._replace(map_width=config.map_width + 1)
    elif change == "rule":
        config = config._replace(max_steps=config.max_steps + 1)
    elif change == "spawn":
        config = config._replace(
            team_spawn_pad_positions=config.team_spawn_pad_positions.at[:, :, 0, 0].add(
                0.1
            )
        )
    else:
        classes = classes.at[0].set(2)
    # This test attacks the declaration directly; physical validity is tested by reset.
    altered = state._replace(config=config)
    with pytest.raises(ValueError):
        init_episode_tracking(
            env,
            altered,
            source_configs=_source(),
            source_indices=0,
            source_class_ids=classes,
        )


def test_dynamic_roster_declarations_reuse_the_tracker_compilation() -> None:
    env, state, tracker, classes = _roster_setup()
    traces: list[int] = []

    @jax.jit
    def compiled(
        t: EpisodeTrackingState, s: EnvironmentState, result: StepResult, rows: Array
    ) -> tuple[EpisodeTrackingState, StepResult]:
        traces.append(1)
        return track_episode_step(
            t, s, result, source_indices=jnp.zeros(2, jnp.int32), source_class_ids=rows
        )

    for first_class in (1, 4):
        declared = classes.at[0].set(first_class)
        resolved, _ = _source_config_with_class_ids(_source(), declared)
        _, reset = env.reset(
            jax.random.key(first_class),
            balanced_spawn_configs(resolved, num_envs=2),
            state=state,
        )
        following, _ = cast(
            tuple[EpisodeTrackingState, StepResult],
            compiled(
                tracker, reset, _step(env, reset), jnp.broadcast_to(declared, (2, 10))
            ),
        )
        np.testing.assert_array_equal(following.error_flags, 0)
    assert traces == [1]


def test_concrete_bad_reset_roster_and_unowned_initial_claim_are_rejected() -> None:
    env, state, tracker, classes = _roster_setup()
    with pytest.raises(ValueError, match="source index"):
        init_episode_tracking(
            env,
            state,
            source_configs=_source(),
            source_indices=-1,
            source_class_ids=classes,
        )
    _, reset = env.reset(jax.random.key(308), state=state)
    with pytest.raises(ValueError, match="compact"):
        track_episode_step(
            tracker,
            reset,
            _step(env, reset),
            source_indices=0,
            source_class_ids=classes.at[1].set(-1),
        )
