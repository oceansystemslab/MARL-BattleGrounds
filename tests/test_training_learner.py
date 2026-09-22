"""Check real compact collection, recurrent critic epochs and complete PPO updates.

CPU proofs compare the adapter with direct existing numerical calls, including
reset/cutoff/ending boundaries, final padding, same-call actor data, exact update
and history counts, masked samples, finite failure guards and compilation reuse.
Synthetic short horizons are test inputs after content admission, not proposed
training settings. The default and an explicit zero pinned opponent share with
no named pinned opponent give identical learner states after real updates. The
shared context pins the
"world" frame so its direct recomputation needs no reflection. With the "left"
spawn frame, at two input scales, the direct composition reflects the rebuilt
view, mask and stored index the way the actor did, the stored log probabilities
match that recomputation before any optimizer update, the update is accepted,
and a stored index left in the reflected frame is rejected by the admission
guard. No test claims useful
learning or GPU performance.
"""

from dataclasses import replace
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config
from tests.test_baseline_ppo import _forced_actor  # pyright: ignore[reportPrivateUsage]
from tests.training_learner_helpers import (
    equal,
    finite,
    scanner,
    synthetic_horizons,
    updater,
)

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines.actions import (
    action_log_prob,
    categorical_action_mask,
    mirror_action_indices,
)
from marl_battlegrounds.baselines.inputs import encode_actor_inputs, spawn_frame_flag
from marl_battlegrounds.baselines.ppo import (
    PPOBatch,
    PPOConfig,
    PPOMetrics,
    PPOTrainState,
    RecurrentActor,
    calculate_gae,
    critic_values,
    initialize_ppo,
    update_recurrent_ppo,
)
from marl_battlegrounds.core import env as core
from marl_battlegrounds.episode_tracking import init_episode_tracking
from marl_battlegrounds.evaluation.policy_execution import init_systems
from marl_battlegrounds.policies.input import build_team_actor_input, mirror_team_view
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    make_training_schedule,
    prepare_training_content,
)
from marl_battlegrounds.training.distributions import training_keys
from marl_battlegrounds.training.learner import (
    LEARNER_ERROR_ACTION,
    LEARNER_ERROR_BOUNDARY,
    LEARNER_ERROR_COLLECTION,
    LEARNER_ERROR_NONFINITE_BATCH,
    LEARNER_ERROR_NONFINITE_UPDATE,
    MODEL_INITIALIZATION_TAG,
    SHUFFLE_ROOT_TAG,
    LearnerState,
    UpdateResult,
    UpdateSummary,
    _summary,  # pyright: ignore[reportPrivateUsage]
    build_ppo_batch,
    init_learner,
    update_learner,
    validate_learner,
)

type Tree = Any
type Context = tuple[TrainingCollection, LearnerState, PPOConfig]


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


@pytest.fixture(scope="module")
def context(prepared: PreparedTrainingContent) -> Context:
    ppo = PPOConfig(rollout_length=4, epochs=1, spawn_frame="world")
    collection, state = init_learner(
        schedule=make_training_schedule(total_env_steps=64, num_envs=4),
        seed=71,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
    )
    return collection, state, ppo


@pytest.fixture(scope="module")
def collected(context: Context) -> tuple[TrainingCarry, TrainingRollout]:
    collection, state, ppo = context
    return scanner(collection, ppo.rollout_length)(state.carry)


def _numerical(state: LearnerState) -> PPOTrainState:
    return PPOTrainState(
        state.carry.history.current_variables,
        state.critic_params,
        state.actor_opt_state,
        state.critic_opt_state,
    )


def _nan(value: Array) -> Array:
    return jnp.full_like(value, jnp.nan)


def _nan_floats(value: Array) -> Array:
    return _nan(value) if jnp.issubdtype(value.dtype, jnp.inexact) else value


def _increment_integers(value: Array) -> Array:
    return value + 1 if jnp.issubdtype(value.dtype, jnp.integer) else value


def _direct_batch(
    state: LearnerState, rollout: TrainingRollout
) -> tuple[PPOBatch, Tree]:
    rows = rollout.transitions
    assert rows.training_state is not None
    assert rollout.final_training_state is not None
    memory, values = critic_values(
        state.critic_params,
        state.critic_memory,
        rows.training_state,
        rows.episode_start,
        rows.valid,
    )
    _, final = critic_values(
        state.critic_params,
        memory,
        rollout.final_training_state[None],
        jnp.zeros_like(rollout.final_ended)[None],
        (~rollout.final_ended)[None],
    )
    batch = PPOBatch(
        rows.observations,
        rows.training_state,
        rows.action_mask,
        rows.learning_outputs.action_indices,
        rows.learning_outputs.log_prob,
        jnp.where(rows.valid[..., None], values, 0.0),
        rows.task_rewards + jnp.where(rows.active, rows.shaping_reward[..., None], 0.0),
        rows.ended,
        rows.episode_start,
        rows.valid,
        rows.active,
        rows.alive,
        jnp.where(
            (~rollout.final_ended)[:, None] & rollout.final_active, final[0], 0.0
        ),
        rollout.initial_memory,
        state.critic_memory,
    )
    return batch, memory


def test_default_and_explicit_zero_pinned_share_give_identical_learners(
    prepared: PreparedTrainingContent,
) -> None:
    ppo = PPOConfig(rollout_length=4, epochs=1)
    schedule = make_training_schedule(total_env_steps=64, num_envs=4)
    default = init_learner(
        schedule=schedule, seed=71, ppo=ppo, prepared=prepared, metrics="none"
    )
    explicit = init_learner(
        schedule=schedule,
        seed=71,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
        pinned_opponent_share=0.0,
        pinned_opponent=None,
    )
    states: list[LearnerState] = []
    for collection, state in (default, explicit):
        assert collection.pinned_opponent_share == 0.0
        for _ in range(2):
            carry, rollout = scanner(collection, ppo.rollout_length)(state.carry)
            state, _ = updater(ppo)(state, carry, rollout)
        states.append(state)
    equal(states[0], states[1])


def test_initial_state_owns_one_actor_and_separate_deterministic_keys(
    context: Context,
) -> None:
    collection, state, ppo = context
    assert "actor_params" not in state._fields
    assert collection.actor.variables == ()
    assert collection.collect_training_state
    assert int(state.completed_updates) == int(state.carry.history.current_update) == 0
    assert not bool(state.failed)
    np.testing.assert_array_equal(state.critic_memory, 0)
    root = jax.random.key(71, impl="threefry2x32")
    expected = initialize_ppo(jax.random.fold_in(root, MODEL_INITIALIZATION_TAG), ppo)
    equal(_numerical(state), expected)
    equal(state.shuffle_root, jax.random.fold_in(root, SHUFFLE_ROOT_TAG))
    assert not np.array_equal(
        jax.random.key_data(state.shuffle_root), jax.random.key_data(root)
    )
    validate_learner(collection, state, ppo=ppo)


@pytest.mark.parametrize(
    ("input_scale", "spawn_frame"),
    ((1.0, "world"), (0.01, "world"), (0.01, "left"), (1.0, "left")),
)
def test_collected_values_behavior_and_update_match_direct_composition(
    context: Context,
    collected: tuple[TrainingCarry, TrainingRollout],
    prepared: PreparedTrainingContent,
    input_scale: float,
    spawn_frame: str,
) -> None:
    collection, state, ppo = context
    after, rollout = collected
    if input_scale != 1.0 or spawn_frame != "world":
        ppo = replace(ppo, input_scale=input_scale, spawn_frame=spawn_frame)
        collection, state = init_learner(
            schedule=collection.schedule,
            seed=71,
            ppo=ppo,
            prepared=prepared,
            metrics="none",
        )
        after, rollout = scanner(collection, ppo.rollout_length)(state.carry)
    batch, memory = cast(
        tuple[PPOBatch, Array],
        jax.jit(partial(build_ppo_batch, ppo=ppo))(state, rollout),
    )
    scaled_rollout = rollout._replace(
        transitions=rollout.transitions._replace(
            training_state=cast(Array, rollout.transitions.training_state) * input_scale
        ),
        final_training_state=cast(Array, rollout.final_training_state) * input_scale,
    )
    direct, direct_memory = _direct_batch(state, scaled_rollout)
    direct = direct._replace(training_state=batch.training_state)
    equal(batch, direct, close=True)
    equal(memory, direct_memory, close=True)
    inputs = jax.vmap(
        jax.vmap(build_team_actor_input, in_axes=(0, None)), in_axes=(0, None)
    )(batch.observations, 0)
    native_mask = batch.action_mask
    actions = batch.actions
    if spawn_frame != "world":
        # Recompute in the frame the actor sampled in: reflect the rebuilt view
        # and mask and map the stored world index into that frame.
        flag = spawn_frame_flag(inputs, spawn_frame)
        assert bool(flag.any()) and not bool(flag.all())
        inputs, native_mask = mirror_team_view(inputs, native_mask, flag)
        actions = mirror_action_indices(actions, flag)
    shape = batch.actions.shape
    _, logits = cast(
        tuple[Array, Array],
        RecurrentActor().apply(
            state.carry.history.current_variables,
            batch.actor_memory,
            encode_actor_inputs(inputs) * input_scale,
            jnp.broadcast_to(batch.episode_start[..., None], shape),
            jnp.broadcast_to(batch.valid[..., None], shape),
        ),
    )
    np.testing.assert_allclose(
        action_log_prob(logits, categorical_action_mask(native_mask), actions),
        batch.old_log_prob,
        atol=2e-6,
        rtol=2e-6,
    )
    expected, metrics = cast(
        tuple[PPOTrainState, PPOMetrics],
        jax.jit(partial(update_recurrent_ppo, config=ppo))(
            _numerical(state),
            direct,
            jax.random.fold_in(state.shuffle_root, jnp.int32(1)),
        ),
    )
    updated, result = updater(ppo)(state, after, rollout)
    assert bool(result.performed) and not bool(result.failed)
    equal(_numerical(updated), expected, close=True)
    equal(result.metrics, metrics, close=True)
    equal(updated.critic_memory, direct_memory, close=True)
    equal(updated.carry.memory, after.memory)
    equal(updated.carry.history.lane_snapshot, after.history.lane_snapshot)
    assert (
        int(updated.completed_updates) == int(updated.carry.history.current_update) == 1
    )
    assert (
        int(updated.carry.history.last_refresh_rounds)
        == int(after.progress.rounds)
        == 4
    )
    assert int(result.summary.real_transitions) == 16
    assert int(result.summary.active_samples) == 80
    assert int(result.metrics.critic_samples.sum()) == 80
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(expected.actor_params),
            jax.tree.leaves(state.carry.history.current_variables),
            strict=True,
        )
    )
    continued, next_rollout = scanner(collection, ppo.rollout_length)(updated.carry)
    assert int(continued.progress.rounds) == 8
    np.testing.assert_array_equal(next_rollout.transitions.learner_update, 1)
    finite((updated, result))


def test_full_t128_public_collection_updates_and_finishes_partial_budget(
    prepared: PreparedTrainingContent,
) -> None:
    ppo = PPOConfig(rollout_length=128, epochs=1)
    collection, state = init_learner(
        schedule=make_training_schedule(total_env_steps=4 * 129, num_envs=4),
        seed=72,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
    )
    collect = scanner(collection, 128)
    after, first = collect(state.carry)
    updated, result = updater(ppo)(state, after, first)
    assert bool(result.performed) and not bool(result.failed)
    assert int(result.summary.real_transitions) == 512
    after_final, final = collect(updated.carry)
    assert int(final.real_steps) == 1
    finished, second = updater(ppo)(updated, after_final, final)
    assert bool(second.performed) and not bool(second.failed)
    assert int(second.summary.real_transitions) == 4
    assert int(finished.carry.progress.rounds) == 129
    assert int(finished.completed_updates) == 2
    empty_after, empty = collect(finished.carry)
    unchanged, absent = updater(ppo)(finished, empty_after, empty)
    equal(unchanged, finished)
    assert not bool(absent.performed) and not bool(absent.failed)
    assert all(np.all(np.asarray(value) == 0) for value in absent.metrics)
    finite((finished, result, second))


@pytest.mark.parametrize("real_steps", [1, 3])
def test_short_prefix_and_empty_followup_keep_finite_values_and_exact_history(
    prepared: PreparedTrainingContent,
    real_steps: int,
) -> None:
    ppo = PPOConfig(rollout_length=4, epochs=1)
    collection, state = init_learner(
        schedule=make_training_schedule(total_env_steps=4 * real_steps, num_envs=4),
        seed=73,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
        shaping=True,
    )
    after, rollout = scanner(collection, 4)(state.carry)
    updated, result = updater(ppo)(state, after, rollout)
    assert bool(result.performed) and not bool(result.failed)
    assert int(updated.carry.progress.rounds) == real_steps
    assert int(result.metrics.critic_samples.sum()) == real_steps * 4 * 5
    assert int(updated.carry.history.count) == 1
    np.testing.assert_array_equal(updated.carry.history.threshold_to_snapshot, 0)
    np.testing.assert_array_equal(updated.carry.state.reset_generation, 0)
    batch, _ = build_ppo_batch(state, rollout)
    advantages, targets = calculate_gae(
        batch.rewards,
        batch.old_values,
        batch.ended,
        batch.final_values,
        valid=batch.valid[..., None] & batch.active,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
    )
    finite((batch, advantages, targets, updated, result))
    np.testing.assert_array_equal(targets[real_steps:], 0)
    empty_after, empty = scanner(collection, 4)(updated.carry)
    # Nonfinite critic weights prove a truly empty call does not read a network.
    poisoned = updated._replace(critic_params=jax.tree.map(_nan, updated.critic_params))
    unchanged, absent = updater(ppo)(poisoned, empty_after, empty)
    equal(unchanged, poisoned)
    assert not bool(absent.performed) and not bool(absent.failed)
    assert not bool(absent.snapshot.created)


def test_pending_reset_and_bootstrap_preserve_correct_two_block_critic_carry(
    prepared: PreparedTrainingContent,
) -> None:
    ppo = PPOConfig(rollout_length=1, epochs=1)
    collection, state = init_learner(
        schedule=make_training_schedule(total_env_steps=20, num_envs=4),
        seed=74,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
    )
    state = synthetic_horizons(collection, state, horizon=20, first_horizon=1)
    collect = scanner(collection, 1)
    after, first = collect(state.carry)
    assert np.any(np.asarray(first.final_ended)) and not np.all(
        np.asarray(first.final_ended)
    )
    updated, result = updater(ppo)(state, after, first)
    assert bool(result.performed)
    second_after, second = collect(updated.carry)
    batch, memory = build_ppo_batch(updated, second)
    raw, raw_memory = _direct_batch(updated, second)
    reset = np.asarray(second.transitions.episode_start[0])
    assert reset.any() and not reset.all()
    np.testing.assert_array_equal(batch.actor_memory[reset], 0)
    np.testing.assert_array_equal(batch.critic_memory[reset], 0)
    equal(batch.actor_memory[~reset], second.initial_memory[~reset])
    equal(batch.critic_memory[~reset], updated.critic_memory[~reset])
    equal(batch.old_values, raw.old_values, close=True)
    equal(memory, raw_memory, close=True)
    equal(batch.final_values, raw.final_values, close=True)
    assert np.any(np.asarray(updated.critic_memory[~reset]) != 0)
    assert second.final_training_state is not None
    advanced, _ = critic_values(
        updated.critic_params,
        memory,
        second.final_training_state[None],
        jnp.zeros((1, 4), bool),
        (~second.final_ended)[None],
    )
    assert not np.allclose(advanced[~reset], memory[~reset])
    final, result = updater(ppo)(updated, second_after, second)
    equal(final.critic_memory, memory, close=True)
    assert bool(result.performed) and not bool(result.failed)


def test_interior_horizon_ending_cuts_gae_and_final_value(context: Context) -> None:
    collection, state, _ = context
    state = synthetic_horizons(collection, state, horizon=2)
    _, rollout = scanner(collection, 4)(state.carry)
    batch, _ = build_ppo_batch(state, rollout)
    np.testing.assert_array_equal(batch.episode_start[:, 0], (True, False, True, False))
    np.testing.assert_array_equal(batch.ended[:, 0], (False, True, False, True))
    np.testing.assert_array_equal(batch.final_values, 0)
    _, targets = calculate_gae(
        batch.rewards,
        batch.old_values,
        batch.ended,
        jnp.full_like(batch.final_values, 12345),
        valid=batch.valid,
    )
    np.testing.assert_allclose(targets[-1], batch.rewards[-1], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    "kind",
    ["boundary", "collection", "probability", "action", "reflected_index", "optimizer"],
)
def test_bad_candidates_never_publish_or_advance_prior_boundary(
    context: Context,
    collected: tuple[TrainingCarry, TrainingRollout],
    kind: str,
) -> None:
    _, state, ppo = context
    after, rollout = collected
    expected_reason = LEARNER_ERROR_BOUNDARY
    if kind == "boundary":
        rollout = rollout._replace(real_steps=jnp.int32(3))
    elif kind == "collection":
        after = after._replace(
            state=after.state._replace(
                lifecycle_error=jnp.ones_like(after.state.lifecycle_error)
            )
        )
        expected_reason = LEARNER_ERROR_COLLECTION
    elif kind == "probability":
        outputs = rollout.transitions.learning_outputs
        rollout = rollout._replace(
            transitions=rollout.transitions._replace(
                learning_outputs=outputs._replace(
                    log_prob=outputs.log_prob.at[0, 0, 0].set(jnp.nan)
                )
            )
        )
        expected_reason = LEARNER_ERROR_NONFINITE_BATCH
    elif kind == "action":
        outputs = rollout.transitions.learning_outputs
        rollout = rollout._replace(
            transitions=rollout.transitions._replace(
                learning_outputs=outputs._replace(
                    action_indices=outputs.action_indices.at[0, 0, 0].set(198)
                )
            )
        )
        expected_reason = LEARNER_ERROR_ACTION
    elif kind == "reflected_index":
        # A stored index left in the reflected frame for a sideways move must
        # fail the guard that compares it with the submitted world action.
        outputs = rollout.transitions.learning_outputs
        indices = np.asarray(outputs.action_indices)
        moving = np.argwhere((indices // 22) >= 3)
        assert moving.size, "the collected rollout holds no sideways move"
        t, b, a = (int(value) for value in moving[0])
        reflected = mirror_action_indices(
            jnp.asarray(indices[t, b, a], jnp.int32), jnp.bool_(True)
        )
        rollout = rollout._replace(
            transitions=rollout.transitions._replace(
                learning_outputs=outputs._replace(
                    action_indices=outputs.action_indices.at[t, b, a].set(reflected)
                )
            )
        )
        expected_reason = LEARNER_ERROR_ACTION
    else:
        state = state._replace(
            critic_opt_state=jax.tree.map(_nan_floats, state.critic_opt_state)
        )
        expected_reason = LEARNER_ERROR_NONFINITE_UPDATE
    rejected, result = updater(ppo)(state, after, rollout)
    assert bool(result.failed) and not bool(result.performed)
    assert int(result.failure_reason) == expected_reason
    equal(
        rejected._replace(failed=state.failed, failure_reason=state.failure_reason),
        state,
    )
    assert not bool(result.snapshot.created)
    sticky, repeated = updater(ppo)(rejected, *collected)
    equal(sticky, rejected)
    assert int(repeated.failure_reason) == expected_reason


def test_same_shape_changed_state_and_data_reuse_one_compilation(
    context: Context,
    collected: tuple[TrainingCarry, TrainingRollout],
) -> None:
    collection, initial, ppo = context
    traces: list[int] = []

    def apply(
        state: LearnerState, after: TrainingCarry, rows: TrainingRollout
    ) -> tuple[LearnerState, UpdateResult]:
        traces.append(1)
        return update_learner(state, after, rows, ppo=ppo)

    compiled = jax.jit(apply)
    first, result = cast(
        tuple[LearnerState, UpdateResult], compiled(initial, *collected)
    )
    following = scanner(collection, 4)(first.carry)
    second, again = cast(tuple[LearnerState, UpdateResult], compiled(first, *following))
    assert bool(result.performed) and bool(again.performed)
    assert int(second.completed_updates) == 2
    assert traces == [1]


def test_static_contracts_fail_before_numerical_update(
    context: Context,
    collected: tuple[TrainingCarry, TrainingRollout],
    prepared: PreparedTrainingContent,
) -> None:
    _, state, ppo = context
    after, rollout = collected
    with pytest.raises(ValueError, match="collect_training_state"):
        build_ppo_batch(state, rollout._replace(final_training_state=None))
    with pytest.raises(TypeError, match="PPOLearningOutputs"):
        build_ppo_batch(
            state,
            rollout._replace(
                transitions=rollout.transitions._replace(learning_outputs=())
            ),
        )
    with pytest.raises(ValueError, match="T=rollout_length"):
        update_learner(state, after, rollout, ppo=replace(ppo, rollout_length=8))
    with pytest.raises(ValueError, match="divisible"):
        init_learner(
            schedule=make_training_schedule(total_env_steps=12, num_envs=6),
            ppo=ppo,
            prepared=prepared,
        )
    with pytest.raises(TypeError, match="seed"):
        init_learner(
            schedule=make_training_schedule(total_env_steps=16, num_envs=4),
            seed=True,
            ppo=ppo,
            prepared=prepared,
        )


def test_restore_guard_rejects_counter_key_and_optimizer_incoherence(
    context: Context,
) -> None:
    collection, state, ppo = context
    with pytest.raises(ValueError, match="counts disagree"):
        validate_learner(
            collection, state._replace(completed_updates=jnp.int32(1)), ppo=ppo
        )
    with pytest.raises(ValueError, match="shuffle key"):
        validate_learner(
            collection, state._replace(shuffle_root=jax.random.key(999)), ppo=ppo
        )
    corrupted = state._replace(
        actor_opt_state=jax.tree.map(_increment_integers, state.actor_opt_state)
    )
    with pytest.raises(ValueError, match="optimizer counter"):
        validate_learner(collection, corrupted, ppo=ppo)


@pytest.mark.parametrize("value", (None, 0, 1, "False", np.bool_(True)))
def test_content_recheck_requires_python_bool_before_reading_state(
    value: object,
) -> None:
    with pytest.raises(TypeError, match=r"recheck_installed_content.*Python bool"):
        validate_learner(
            cast(TrainingCollection, None),
            cast(LearnerState, None),
            recheck_installed_content=cast(bool, value),
        )


@pytest.mark.parametrize(
    "field", ("historical_actor", "physical_state", "observations")
)
def test_restore_rejects_nonfinite_complete_carry_before_any_recovery(
    context: Context,
    collected: tuple[TrainingCarry, TrainingRollout],
    field: str,
) -> None:
    collection, initial, ppo = context
    state, result = updater(ppo)(initial, *collected)
    assert bool(result.performed) and int(state.carry.history.count) > 0
    validate_learner(collection, state, ppo=ppo)
    carry = state.carry
    if field == "historical_actor":
        carry = carry._replace(
            history=carry.history._replace(
                historical_variables=jax.tree.map(
                    _nan, carry.history.historical_variables
                )
            )
        )
    elif field == "physical_state":
        carry = carry._replace(
            state=carry.state._replace(
                core_state=carry.state.core_state._replace(
                    agent_positions=carry.state.core_state.agent_positions.at[
                        0, 0, 0
                    ].set(jnp.nan)
                )
            )
        )
    else:
        carry = carry._replace(
            observations=jax.tree.map(_nan_floats, carry.observations)
        )
    with pytest.raises(ValueError, match="nonfinite"):
        validate_learner(collection, state._replace(carry=carry), ppo=ppo)


def _combat_context(prepared: PreparedTrainingContent, mode: str) -> Context:
    length = 3 if mode == "respawn" else 1
    ppo = PPOConfig(rollout_length=length, epochs=1)
    collection, learner = init_learner(
        schedule=make_training_schedule(total_env_steps=4 * length, num_envs=4),
        seed=75,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
    )
    config = evaluation_env_config(
        team_sizes=(1, 1),
        task_mode=core.TASK_MODE_TDM,
        team_deathmatch_score_threshold=1 if mode == "win-loss" else 20,
        max_steps=20,
    )._replace(
        spawn_shield_duration_steps=0,
        team_respawn_wave_period_step_count=jnp.full(
            2, 100 if mode == "dead" else 2, jnp.int32
        ),
    )
    initial, *_ = core.reset(config, jax.random.key(50))
    authored = initial._replace(
        step_count=jnp.int32(5),
        agent_positions=initial.agent_positions.at[0]
        .set(jnp.asarray((4.0, 4.0)))
        .at[5]
        .set(jnp.asarray((6.5, 4.0))),
        current_health=initial.current_health.at[0].set(1.0).at[5].set(1.0),
        team_respawn_wave_countdowns=jnp.asarray((1, 1), jnp.int32),
    )
    if mode == "dead":
        authored = authored._replace(
            current_health=jnp.zeros_like(authored.current_health),
            alive_mask=jnp.zeros_like(authored.alive_mask),
            team_respawn_wave_countdowns=jnp.full(2, 99, jnp.int32),
        )
    elif mode == "win-loss":
        authored = authored._replace(
            current_health=authored.current_health.at[0].set(
                config.agent_profile.max_health[0]
            )
        )
    snapshot = core.initialize_scenario_state(authored, config)
    other = (
        authored._replace(
            current_health=authored.current_health.at[0]
            .set(1.0)
            .at[5]
            .set(config.agent_profile.max_health[5])
        )
        if mode == "win-loss"
        else authored
    )
    exchanged = core.initialize_scenario_state(
        other,
        config._replace(team_spawn_pad_positions=config.team_spawn_pad_positions[::-1]),
    )

    def mirrored(a: Array, b: Array) -> Array:
        return jnp.stack((a, a, b, b))

    starts = jax.tree.map(mirrored, snapshot[:3], exchanged[:3])

    def four(value: Array) -> Array:
        return jnp.broadcast_to(jnp.asarray(value), (4, *jnp.shape(value)))

    def bank(value: Array) -> Array:
        return jnp.broadcast_to(jnp.asarray(value), (42, *jnp.shape(value)))

    carry = learner.carry
    env = marl_bgs.make(
        "tdm",
        env_config=marl_bgs.balanced_spawn_configs(config, num_envs=4),
        num_envs=4,
        metrics="none",
    )
    generation = jnp.zeros(4, jnp.int32)
    observations, state = env.reset(
        training_keys(carry.root_key, generation, stream="reset"),
        initial=starts,
    )
    # This admitted Core scenario is a synthetic learner fixture, not training
    # content. Use normal balanced accounting to isolate death/respawn memory.
    state = state._replace(authored_start=jnp.zeros(4, jnp.bool_))
    history = carry.history._replace(
        current_variables=_forced_actor(carry.history.current_variables, 12)
    )
    memory = init_systems(
        collection.actor,
        collection.opponent,
        observations,
        state,
        training_keys(carry.root_key, generation, stream="initialization"),
        variables_a=history.current_variables,
        variables_b=history,
    )
    sources = jnp.zeros(4, jnp.int32)
    classes = four(config.agent_profile.class_ids)
    tracking = init_episode_tracking(
        env,
        state,
        source_configs=jax.tree.map(bank, config),
        source_indices=sources,
        source_class_ids=classes,
    ).begin_stage(state, total_env_steps=4 * length)
    learner = learner._replace(
        carry=carry._replace(
            env=env,
            observations=observations,
            state=state,
            history=history,
            memory=memory,
            tracking=tracking,
            source_indices=sources,
            source_class_ids=classes,
        )
    )
    return collection, learner, ppo


def test_death_and_respawn_keep_critic_memory_and_only_remove_policy_samples(
    prepared: PreparedTrainingContent,
) -> None:
    collection, learner, ppo = _combat_context(prepared, "respawn")
    after, rollout = scanner(collection, ppo.rollout_length)(learner.carry)
    assert bool(jnp.all(rollout.transitions.valid)), (
        np.asarray(after.tracking.error_flags).tolist(),
        np.asarray(after.history.error).tolist(),
        np.asarray(after.progress.rounds).tolist(),
    )
    np.testing.assert_array_equal(
        rollout.transitions.alive[:, 0, 0], (True, False, True)
    )
    np.testing.assert_array_equal(
        rollout.transitions.episode_start[:, 0], (True, False, False)
    )
    np.testing.assert_array_equal(
        rollout.transitions.learning_outputs.action_indices[1], 0
    )
    batch, memory_after = build_ppo_batch(learner, rollout)
    direct, reference_memory = _direct_batch(learner, rollout)
    equal(batch.old_values, direct.old_values, close=True)
    equal(memory_after, reference_memory, close=True)
    updated, result = updater(ppo)(learner, after, rollout)
    assert bool(result.performed) and not bool(result.failed)
    assert int(result.metrics.actor_samples.sum()) == 8
    assert int(result.metrics.critic_samples.sum()) == 12
    assert int(result.summary.live_actor_decisions) == 8
    assert int(result.summary.active_samples) == 12
    np.testing.assert_array_equal(updated.carry.state.reset_generation, 0)
    finite((updated, result))


def test_wholly_dead_real_block_advances_critic_history_but_not_actor_optimizer(
    prepared: PreparedTrainingContent,
) -> None:
    collection, learner, ppo = _combat_context(prepared, "dead")
    after, rollout = scanner(collection, ppo.rollout_length)(learner.carry)
    np.testing.assert_array_equal(rollout.transitions.valid, True)
    np.testing.assert_array_equal(rollout.transitions.alive, False)
    np.testing.assert_array_equal(rollout.transitions.active.sum(axis=-1), 1)
    updated, result = updater(ppo)(learner, after, rollout)
    assert bool(result.performed) and not bool(result.failed)
    assert int(result.metrics.actor_samples.sum()) == 0
    assert int(result.metrics.critic_samples.sum()) == 4
    equal(
        updated.carry.history.current_variables, learner.carry.history.current_variables
    )
    equal(updated.actor_opt_state, learner.actor_opt_state)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(updated.critic_params),
            jax.tree.leaves(learner.critic_params),
            strict=True,
        )
    )
    assert int(updated.completed_updates) == 1
    assert int(updated.carry.history.current_update) == 1
    assert int(updated.carry.history.last_refresh_rounds) == 1
    assert int(result.summary.live_actor_decisions) == 0
    finite((updated, result))


@pytest.mark.parametrize("dense", (False, True))
def test_real_win_and_loss_endings_flow_into_terminal_gae_and_complete_update(
    prepared: PreparedTrainingContent,
    dense: bool,
) -> None:
    collection, learner, ppo = _combat_context(prepared, "win-loss")
    if dense:
        collection = replace(collection, shaping=True, shaping_mode="score_delta")
    after, rollout = scanner(collection, ppo.rollout_length)(learner.carry)
    np.testing.assert_array_equal(rollout.transitions.valid, True)
    np.testing.assert_array_equal(rollout.transitions.ended, True)
    np.testing.assert_array_equal(
        rollout.transitions.outcome[0],
        (core.TASK_MODE_OUTCOME_TEAM_A_WIN,) * 2
        + (core.TASK_MODE_OUTCOME_TEAM_B_WIN,) * 2,
    )
    batch, _ = build_ppo_batch(learner, rollout)
    np.testing.assert_array_equal(
        rollout.transitions.task_rewards[0, :, 0], (1, 1, -1, -1)
    )
    np.testing.assert_allclose(
        batch.rewards[0, :, 0],
        (1.01, 1.01, -1.01, -1.01) if dense else (1, 1, -1, -1),
        atol=2e-6,
    )
    np.testing.assert_array_equal(batch.final_values, 0)
    _, targets = calculate_gae(
        batch.rewards,
        batch.old_values,
        batch.ended,
        batch.final_values,
        valid=batch.valid[..., None] & batch.active,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
    )
    np.testing.assert_allclose(targets, batch.rewards, atol=2e-6)
    updated, result = updater(ppo)(learner, after, rollout)
    assert bool(result.performed) and not bool(result.failed)
    assert int(result.metrics.actor_samples.sum()) == 4
    assert int(result.metrics.critic_samples.sum()) == 4
    np.testing.assert_array_equal(result.summary.stage_completed[0], (2, 0, 2))
    np.testing.assert_array_equal(result.summary.stage_score_sums[0], (2, 2))
    assert int(updated.completed_updates) == 1
    finite((updated, result))


def test_curriculum_boundary_reports_producing_stage_and_completed_game_facts(
    prepared: PreparedTrainingContent,
) -> None:
    ppo = PPOConfig(rollout_length=4, epochs=1)
    collection, state = init_learner(
        schedule=make_training_schedule(
            total_env_steps=400, num_envs=4, curriculum=True
        ),
        seed=76,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
        shaping=True,
    )
    state = synthetic_horizons(collection, state, horizon=1)
    after, rollout = scanner(collection, 4)(state.carry)
    updated, result = updater(ppo)(state, after, rollout)
    assert bool(result.performed) and not bool(result.failed)
    assert int(after.tracking.stage_ordinal) == 1
    np.testing.assert_array_equal(rollout.transitions.episode_stage, 0)
    np.testing.assert_array_equal(rollout.transitions.active.sum(axis=-1), 1)
    np.testing.assert_array_equal(result.summary.stage_completed[0], (0, 16, 0))
    np.testing.assert_array_equal(result.summary.stage_completed[1:], 0)
    np.testing.assert_array_equal(result.summary.stage_score_sums, 0)
    np.testing.assert_array_equal(result.summary.stage_length_sum[0], 16)
    np.testing.assert_array_equal(result.summary.stage_k20_count, 0)
    assert int(result.metrics.actor_samples.sum()) == 16
    assert int(updated.carry.history.current_update) == 1
    assert int(updated.carry.history.last_refresh_rounds) == 4
    next_after, next_rows = scanner(collection, 4)(updated.carry)
    np.testing.assert_array_equal(next_rows.transitions.episode_stage, 1)
    np.testing.assert_array_equal(next_rows.transitions.active.sum(axis=-1), 2)
    final, next_result = updater(ppo)(updated, next_after, next_rows)
    assert bool(next_result.performed) and not bool(next_result.failed)
    np.testing.assert_array_equal(next_result.summary.stage_completed[1], (0, 16, 0))
    assert int(final.carry.history.current_update) == 2


def test_completion_summary_keeps_outcomes_scores_and_valid_stage_ownership(
    collected: tuple[TrainingCarry, TrainingRollout],
) -> None:
    _, rollout = collected
    rows = rollout.transitions
    valid = jnp.zeros_like(rows.valid).at[0, :3].set(True)
    rows = rows._replace(
        valid=valid,
        ended=jnp.ones_like(rows.ended),
        episode_stage=jnp.full_like(rows.episode_stage, 16).at[0, :3].set(2),
        outcome=jnp.full_like(rows.outcome, core.TASK_MODE_OUTCOME_TEAM_A_WIN)
        .at[0, 1]
        .set(core.TASK_MODE_OUTCOME_DRAW)
        .at[0, 2]
        .set(core.TASK_MODE_OUTCOME_TEAM_B_WIN),
        final_scores=jnp.full_like(rows.final_scores, 99)
        .at[0, :3]
        .set(jnp.asarray(((20, 1), (7, 7), (3, 20)), jnp.int32)),
        episode_length=jnp.full_like(rows.episode_length, 999)
        .at[0, :3]
        .set(jnp.asarray((10, 11, 12), jnp.int32)),
    )
    result = cast(UpdateSummary, jax.jit(_summary)(rollout._replace(transitions=rows)))
    np.testing.assert_array_equal(result.stage_completed[2], (1, 1, 1))
    np.testing.assert_array_equal(result.stage_score_sums[2], (30, 28))
    assert int(result.stage_length_sum[2]) == 33
    assert int(result.stage_k20_count[2]) == 2
    np.testing.assert_array_equal(result.stage_completed[16], 0)
    np.testing.assert_array_equal(result.stage_score_sums[16], 0)
    assert int(result.stage_length_sum[16]) == int(result.stage_k20_count[16]) == 0
