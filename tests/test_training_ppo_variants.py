"""Check compact PPO variant learning through real CPU collection boundaries.

Local critics cannot read physical state or another recipient's private data.
Recurrent IPPO values use the true decision rows and preserve their own carry;
bootstrap-only work does not advance it. Feedforward methods have empty memory
through collection, updates and opponent history. Real short prefixes, task
endings, death/respawn and failed updates preserve the existing sample, state
and publication rules. Synthetic combat and short horizons are test fixtures,
not approved training content. IPPO curriculum and shaping runs check reset-time
stages and separate reward records. The complete native-horizon public workflows
live in test_training_ppo_workflows.py. These tests establish no learning or
speed claim.
"""

import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.test_baseline_ppo import _forced_actor  # pyright: ignore[reportPrivateUsage]
from tests.test_training_learner import (
    _combat_context,  # pyright: ignore[reportPrivateUsage]
)
from tests.test_training_pinned_wrapper import (
    _history,  # pyright: ignore[reportPrivateUsage]
    _setup,  # pyright: ignore[reportPrivateUsage]
)
from tests.training_learner_helpers import (
    equal,
    finite,
    scanner,
    synthetic_horizons,
    updater,
)

from marl_battlegrounds.baselines.inputs import encode_actor_inputs, spawn_frame_flag
from marl_battlegrounds.baselines.ppo import (
    PPOConfig,
    RecurrentValueNet,
    initialize_ppo,
    make_ppo_system,
)
from marl_battlegrounds.core.types import AGENT_FEATURE_X
from marl_battlegrounds.evaluation.policy_execution import (
    SystemOutput,
    apply_systems,
    init_systems,
)
from marl_battlegrounds.evaluation.system_evaluation import prepare_evaluation_system
from marl_battlegrounds.policies.input import build_team_actor_input, mirror_team_view
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    make_training_schedule,
    prepare_training_content,
)
from marl_battlegrounds.training import collection as collection_module
from marl_battlegrounds.training.collection import (
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    scan_training_rollout,
)
from marl_battlegrounds.training.distributions import training_keys
from marl_battlegrounds.training.learner import (
    LEARNER_ERROR_NONFINITE_UPDATE,
    LearnerState,
    build_ppo_batch,
    init_learner,
    validate_learner,
)
from marl_battlegrounds.training.opponents import (
    init_opponent_history,
    make_opponent_system,
    refresh_opponents,
)

type Tree = Any
type Context = tuple[str, TrainingCollection, LearnerState, PPOConfig]


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


@pytest.fixture(scope="module", params=("ippo", "ff_mappo", "ff_ippo"))
def context(
    request: pytest.FixtureRequest, prepared: PreparedTrainingContent
) -> Context:
    method = cast(str, request.param)
    ppo = PPOConfig(rollout_length=4, epochs=1, input_scale=0.01)
    collection, state = init_learner(
        schedule=make_training_schedule(total_env_steps=20, num_envs=4),
        seed=19047101,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
        method=method,
    )
    return method, collection, state, ppo


@pytest.fixture(scope="module")
def collected(context: Context) -> tuple[TrainingCarry, TrainingRollout]:
    _, collection, state, ppo = context
    return scanner(collection, ppo.rollout_length)(state.carry)


def _nan_floats(value: Array) -> Array:
    return (
        jnp.full_like(value, jnp.nan)
        if jnp.issubdtype(value.dtype, jnp.inexact)
        else value
    )


def test_real_variant_update_then_partial_and_empty_rollouts_keep_exact_counts(
    context: Context,
    collected: tuple[TrainingCarry, TrainingRollout],
) -> None:
    method, collection, initial, ppo = context
    validate_learner(collection, initial, ppo=ppo, method=method)
    after, rollout = collected
    batch, memory = build_ppo_batch(initial, rollout, ppo=ppo, method=method)
    assert collection.collect_training_state == (method == "ff_mappo")
    assert (batch.training_state is None) == (method != "ff_mappo")
    update = updater(ppo, method)
    state, result = update(initial, after, rollout)
    assert bool(result.performed) and not bool(result.failed)
    assert int(result.summary.real_transitions) == 16
    assert int(result.metrics.critic_samples.sum()) == 80
    assert int(state.completed_updates) == int(state.carry.history.current_update) == 1
    equal(state.critic_memory, memory, close=True)
    equal(state.carry.memory, after.memory)
    if method.startswith("ff_"):
        assert batch.actor_memory == batch.critic_memory == memory == ()
        assert (
            rollout.initial_memory
            == state.carry.memory.team_a
            == state.carry.memory.team_b
            == ()
        )
    collect = scanner(collection, ppo.rollout_length)
    last_after, last = collect(state.carry)
    assert int(last.real_steps) == 1
    np.testing.assert_array_equal(last.transitions.valid[1:], False)
    np.testing.assert_array_equal(
        last.transitions.learning_outputs.action_indices[1:], 0
    )
    np.testing.assert_array_equal(last.transitions.learning_outputs.log_prob[1:], 0)
    finished, last_result = update(state, last_after, last)
    assert bool(last_result.performed) and not bool(last_result.failed)
    assert int(last_result.summary.real_transitions) == 4
    assert int(last_result.metrics.critic_samples.sum()) == 20
    assert int(finished.carry.progress.rounds) == 5
    assert int(finished.completed_updates) == 2
    empty_after, empty = collect(finished.carry)
    poisoned = finished._replace(
        critic_params=jax.tree.map(_nan_floats, finished.critic_params)
    )
    unchanged, absent = update(poisoned, empty_after, empty)
    equal(unchanged, poisoned)
    assert not bool(absent.performed) and not bool(absent.failed)
    assert not bool(absent.snapshot.created)
    finite((state, result, finished, last_result))


def test_variant_rejected_update_rolls_back_memory_statistics_and_history(
    context: Context,
    collected: tuple[TrainingCarry, TrainingRollout],
) -> None:
    method, _, state, ppo = context
    poisoned = state._replace(
        critic_opt_state=jax.tree.map(_nan_floats, state.critic_opt_state)
    )
    failed, result = updater(ppo, method)(poisoned, *collected)
    assert bool(result.failed) and not bool(result.performed)
    assert int(result.failure_reason) == LEARNER_ERROR_NONFINITE_UPDATE
    assert not bool(result.snapshot.created)
    equal(
        failed._replace(failed=poisoned.failed, failure_reason=poisoned.failure_reason),
        poisoned,
    )


@pytest.mark.parametrize("method", ("ippo", "ff_ippo"))
def test_ippo_collection_never_calls_the_physical_state_encoder(
    prepared: PreparedTrainingContent,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    def forbidden(*args: Tree, **kwargs: Tree) -> Tree:
        raise AssertionError("An IPPO rollout tried to encode physical state")

    monkeypatch.setattr(collection_module, "encode_training_state", forbidden)
    ppo = PPOConfig(rollout_length=1, epochs=1, input_scale=0.01)
    collection, state = init_learner(
        schedule=make_training_schedule(total_env_steps=4, num_envs=4),
        seed=19047103,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
        method=method,
    )
    _, rollout = cast(
        tuple[TrainingCarry, TrainingRollout],
        jax.jit(partial(scan_training_rollout, collection, length=1))(state.carry),
    )
    assert rollout.transitions.training_state is None
    assert rollout.final_training_state is None
    batch, _ = build_ppo_batch(state, rollout, ppo=ppo, method=method)
    assert batch.training_state is None
    finite(batch)


@pytest.mark.parametrize("method", ("ippo", "ff_ippo"))
def test_local_critic_cannot_use_unpermitted_rows_or_physical_state(
    prepared: PreparedTrainingContent,
    method: str,
) -> None:
    ppo = PPOConfig(
        rollout_length=1, epochs=1, input_scale=0.01, value_normalization=False
    )
    collection, state = init_learner(
        schedule=make_training_schedule(total_env_steps=4, num_envs=4),
        seed=19047104,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
        method=method,
    )
    _, rollout = scanner(collection, 1)(state.carry)
    rows = rollout.transitions
    # Remove sharing explicitly before varying another recipient's private row.
    observations = rows.observations._replace(
        source_availability=jnp.zeros_like(rows.observations.source_availability)
    )
    final = rollout.final_observations._replace(
        source_availability=jnp.zeros_like(
            rollout.final_observations.source_availability
        )
    )
    rollout = rollout._replace(
        transitions=rows._replace(
            observations=observations,
            episode_start=jnp.zeros_like(rows.episode_start),
        ),
        final_observations=final,
    )
    baseline, original_memory = build_ppo_batch(state, rollout, ppo=ppo, method=method)
    private = observations.observation
    changed_obs = observations._replace(
        observation=private._replace(
            self_features=private.self_features.at[..., 1, AGENT_FEATURE_X].add(17)
        )
    )
    hidden = state.carry.state._replace(
        core_state=state.carry.state.core_state._replace(
            current_health=state.carry.state.core_state.current_health + 7
        )
    )
    hidden_state = state._replace(carry=state.carry._replace(state=hidden))
    changed_rollout = rollout._replace(
        transitions=rollout.transitions._replace(observations=changed_obs)
    )
    private_changes = [(hidden_state, rollout), (state, changed_rollout)]
    if method == "ippo":
        private_changes.append(
            (
                state._replace(critic_memory=state.critic_memory.at[:, 1].set(0.4)),
                rollout,
            )
        )
    for changed_state, changed_rows in private_changes:
        changed, changed_memory = build_ppo_batch(
            changed_state, changed_rows, ppo=ppo, method=method
        )
        np.testing.assert_array_equal(
            changed.old_values[..., 0], baseline.old_values[..., 0]
        )
        if method == "ippo":
            np.testing.assert_array_equal(changed_memory[:, 0], original_memory[:, 0])
    own_obs = observations._replace(
        observation=private._replace(
            self_features=private.self_features.at[..., 0, AGENT_FEATURE_X].add(17)
        )
    )
    own, _ = build_ppo_batch(
        state,
        rollout._replace(
            transitions=rollout.transitions._replace(observations=own_obs)
        ),
        ppo=ppo,
        method=method,
    )
    assert not np.allclose(
        own.old_values[..., 0], baseline.old_values[..., 0], atol=1e-7, rtol=1e-7
    )
    if method == "ippo":
        remembered, _ = build_ppo_batch(
            state._replace(critic_memory=state.critic_memory.at[:, 0].set(0.4)),
            rollout,
            ppo=ppo,
            method=method,
        )
        assert not np.allclose(
            remembered.old_values[..., 0],
            baseline.old_values[..., 0],
            atol=1e-7,
            rtol=1e-7,
        )


def test_ippo_true_successor_bootstrap_does_not_advance_saved_critic_carry(
    prepared: PreparedTrainingContent,
) -> None:
    ppo = PPOConfig(
        rollout_length=1, epochs=1, input_scale=0.01, value_normalization=False
    )
    collection, initial = init_learner(
        schedule=make_training_schedule(total_env_steps=12, num_envs=4),
        seed=74,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
        method="ippo",
    )
    initial = synthetic_horizons(collection, initial, horizon=20, first_horizon=1)
    collect = scanner(collection, 1)
    after, first = collect(initial.carry)
    current, result = updater(ppo, "ippo")(initial, after, first)
    assert bool(result.performed) and not bool(result.failed)
    after_second, second = collect(current.carry)
    batch, saved_memory = build_ppo_batch(current, second, ppo=ppo, method="ippo")
    reset = np.asarray(second.transitions.episode_start[0])
    assert reset.any() and not reset.all()
    np.testing.assert_array_equal(batch.critic_memory[reset], 0)
    equal(batch.critic_memory[~reset], current.critic_memory[~reset])
    final_inputs = jax.vmap(build_team_actor_input, in_axes=(0, None))(
        second.final_observations, 0
    )
    flags = spawn_frame_flag(final_inputs, ppo.spawn_frame)
    final_inputs, _ = jax.vmap(mirror_team_view)(
        final_inputs, second.final_action_mask, flags
    )
    valid = jnp.broadcast_to((~second.final_ended)[:, None], (4, 5))
    advanced, values = cast(
        tuple[Array, Array],
        RecurrentValueNet(input_scale=ppo.input_scale).apply(
            current.critic_params,
            saved_memory,
            encode_actor_inputs(final_inputs)[None],
            jnp.zeros((1, 4, 5), jnp.bool_),
            valid[None],
        ),
    )
    expected = jnp.where(valid & second.final_active, values[0], 0)
    np.testing.assert_allclose(batch.final_values, expected, atol=2e-6, rtol=2e-6)
    assert not np.allclose(advanced[~reset], saved_memory[~reset], atol=1e-7, rtol=1e-7)
    updated, outcome = updater(ppo, "ippo")(current, after_second, second)
    assert bool(outcome.performed) and not bool(outcome.failed)
    equal(updated.critic_memory, saved_memory, close=True)


@pytest.mark.parametrize("method", ("ff_mappo", "ff_ippo"))
def test_feedforward_accepts_time_game_divisibility_without_recurrent_divisibility(
    prepared: PreparedTrainingContent,
    method: str,
) -> None:
    ppo = PPOConfig(rollout_length=4, epochs=1, input_scale=0.01)
    schedule = make_training_schedule(total_env_steps=24, num_envs=6)
    collection, state = init_learner(
        schedule=schedule,
        seed=19047105,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
        method=method,
    )
    validate_learner(collection, state, ppo=ppo, method=method)
    updated, result = updater(ppo, method)(state, *scanner(collection, 4)(state.carry))
    assert bool(result.performed) and not bool(result.failed)
    assert int(result.summary.real_transitions) == 24
    assert int(result.metrics.critic_samples.sum()) == 120
    assert updated.critic_memory == ()
    with pytest.raises(ValueError, match=r"divis|batch|games|B "):
        init_learner(
            schedule=schedule,
            seed=19047105,
            ppo=ppo,
            prepared=prepared,
            metrics="none",
            method="ippo",
        )


@pytest.mark.parametrize(
    ("method", "mode"),
    (
        ("ippo", "respawn"),
        ("ippo", "dead"),
        ("ippo", "win-loss"),
        ("ff_ippo", "respawn"),
        ("ff_mappo", "dead"),
        ("ff_mappo", "win-loss"),
    ),
)
def test_variant_public_combat_keeps_death_respawn_and_task_endings_distinct(
    prepared: PreparedTrainingContent,
    method: str,
    mode: str,
) -> None:
    original_collection, authored, original_ppo = _combat_context(prepared, mode)
    ppo = replace(original_ppo, input_scale=0.01)
    collection, state = init_learner(
        schedule=original_collection.schedule,
        seed=75,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
        method=method,
    )
    carry = authored.carry
    history = state.carry.history._replace(
        current_variables=_forced_actor(state.carry.history.current_variables, 12)
    )
    memory = init_systems(
        collection.actor,
        collection.opponent,
        carry.observations,
        carry.state,
        training_keys(carry.root_key, jnp.zeros(4, jnp.int32), stream="initialization"),
        variables_a=history.current_variables,
        variables_b=history,
    )
    state = state._replace(carry=carry._replace(history=history, memory=memory))
    after, rollout = scanner(collection, ppo.rollout_length)(state.carry)
    np.testing.assert_array_equal(rollout.transitions.valid, True)
    batch, critic_memory = build_ppo_batch(state, rollout, ppo=ppo, method=method)
    updated, result = updater(ppo, method)(state, after, rollout)
    assert bool(result.performed) and not bool(result.failed)
    if mode == "respawn":
        np.testing.assert_array_equal(
            rollout.transitions.alive[:, 0, 0], (True, False, True)
        )
        np.testing.assert_array_equal(
            rollout.transitions.episode_start[:, 0], (True, False, False)
        )
        np.testing.assert_array_equal(
            rollout.transitions.learning_outputs.action_indices[1], 0
        )
        assert int(result.metrics.actor_samples.sum()) == 8
        assert int(result.metrics.critic_samples.sum()) == 12
        np.testing.assert_array_equal(updated.carry.state.reset_generation, 0)
    elif mode == "dead":
        np.testing.assert_array_equal(rollout.transitions.alive, False)
        assert int(result.metrics.actor_samples.sum()) == 0
        assert int(result.metrics.critic_samples.sum()) == 4
        equal(updated.actor_opt_state, state.actor_opt_state)
        equal(
            updated.carry.history.current_variables,
            state.carry.history.current_variables,
        )
    else:
        np.testing.assert_array_equal(rollout.transitions.ended, True)
        np.testing.assert_array_equal(batch.final_values, 0)
        np.testing.assert_array_equal(batch.rewards[0, :, 0], (1, 1, -1, -1))
    equal(updated.critic_memory, critic_memory, close=True)
    if method.startswith("ff_"):
        assert (
            updated.critic_memory
            == updated.carry.memory.team_a
            == updated.carry.memory.team_b
            == ()
        )
    finite((updated, result))


def test_feedforward_current_and_historical_opponents_keep_empty_memory() -> None:
    env, observations, state = _setup(3)
    weights = initialize_ppo(jax.random.key(19047107), method="ff_ippo").actor_params
    actor = make_ppo_system(weights, method="ff_ippo", input_scale=0.01)
    wrapped = make_opponent_system(actor)
    history, event = refresh_opponents(
        init_opponent_history(weights, num_envs=4, keep_past=1),
        weights,
        completed_rounds=jnp.int32(1),
        update_index=jnp.int32(1),
        schedule=make_training_schedule(total_env_steps=4, num_envs=4).arrays,
    )
    assert bool(event.created) and not bool(history.error)
    history = history._replace(lane_snapshot=jnp.asarray((0, -1, 0, -1), jnp.int32))
    inputs = env.policy_inputs(observations, state, team=1)
    keys = jax.random.split(jax.random.key(19047108), 4)
    assert actor.init is None and wrapped.init is not None
    assert wrapped.init(history, inputs, keys) == ()
    expected = cast(SystemOutput, actor.apply(weights, (), inputs, keys))
    for slots in ((-1, -1, -1, -1), (0, -1, 0, -1)):
        actual = cast(
            SystemOutput,
            jax.jit(wrapped.apply)(
                history._replace(lane_snapshot=jnp.asarray(slots, jnp.int32)),
                (),
                inputs,
                keys,
            ),
        )
        assert actual.next_memory == () and actual.learning_outputs == ()
        equal(actual.actions, expected.actions)


@pytest.mark.parametrize(
    ("method", "pinned_method"), (("ff_ippo", "ippo"), ("ippo", "ff_ippo"))
)
def test_cross_architecture_pins_keep_separate_memory_and_matching_actions(
    method: str,
    pinned_method: str,
) -> None:
    _, observations, state = _setup(3)
    weights = initialize_ppo(jax.random.key(19047109), method=method).actor_params
    pin_weights = initialize_ppo(
        jax.random.key(19047110), method=pinned_method
    ).actor_params
    actor = make_ppo_system(weights, method=method, input_scale=0.01)
    pinned = make_ppo_system(pin_weights, method=pinned_method, input_scale=0.01)
    _, pin_variables, template = prepare_evaluation_system(pinned)
    wrapped = make_opponent_system(actor, pinned=pinned)
    variables = (_history(weights, (-2, -1, -2, -1)), pin_variables, template)
    memory = init_systems(
        actor,
        wrapped,
        observations,
        state,
        jax.random.key(19047111),
        variables_a=weights,
        variables_b=variables,
    )
    actual, next_memory, _ = apply_systems(
        actor,
        wrapped,
        memory,
        observations,
        state,
        jax.random.key(19047112),
        variables_a=weights,
        variables_b=variables,
    )
    pin_memory = init_systems(
        actor,
        pinned,
        observations,
        state,
        jax.random.key(19047111),
        variables_a=weights,
    )
    expected, expected_memory, _ = apply_systems(
        actor,
        pinned,
        pin_memory,
        observations,
        state,
        jax.random.key(19047112),
        variables_a=weights,
    )
    # The wrapper's Team B pin is active only in lanes 0 and 2.
    for action, direct in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            action[jnp.asarray((0, 2)), 5:], direct[jnp.asarray((0, 2)), 5:]
        )
    own, pin = next_memory.team_b
    if method == "ff_ippo":
        assert next_memory.team_a == own == ()
        assert pin.shape == (4, 5, 128)
        np.testing.assert_allclose(
            pin[jnp.asarray((0, 2))],
            expected_memory.team_b[jnp.asarray((0, 2))],
            atol=2e-6,
            rtol=2e-6,
        )
    else:
        assert pin == ()
        assert own.shape == (4, 5, 128)


@pytest.mark.parametrize(
    ("curriculum", "shaping"), ((True, False), (False, True), (True, True))
)
def test_public_ippo_curriculum_and_shaping_keep_reset_and_reward_ownership(
    curriculum: bool,
    shaping: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds import training
    from marl_battlegrounds.training import checkpoints

    identity = checkpoints.runtime_identity()
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: identity)
    original_collect = collection_module.collect_training_rollout
    summaries: list[
        tuple[float, float, np.ndarray[Tree, Tree], np.ndarray[Tree, Tree]]
    ] = []
    scheduled_stages: list[int] = []

    def collect(*args: Tree, **kwargs: Tree) -> tuple[TrainingCarry, TrainingRollout]:
        after, rollout = original_collect(*args, **kwargs)
        rows = rollout.transitions
        valid, active = np.asarray(rows.valid), np.asarray(rows.active)
        task = np.asarray(rows.task_rewards)
        adjustment = np.asarray(rows.shaping_reward)
        summaries.append(
            (
                float(np.where(valid[..., None] & active, task, 0).sum())
                / int((valid[..., None] & active).sum()),
                float(np.where(valid, adjustment, 0).sum()) / int(valid.sum()),
                np.asarray(rows.episode_stage),
                np.asarray(rows.episode_start),
            )
        )
        scheduled_stages.append(int(after.tracking.stage_ordinal))
        return after, rollout

    monkeypatch.setattr(collection_module, "collect_training_rollout", collect)
    config = training.TrainConfig(
        keep_past=0,
        method="ippo",
        seed=19047102,
        num_envs=4,
        total_env_steps=256,
        curriculum=curriculum,
        shaping=shaping,
        ppo=PPOConfig(rollout_length=4, groups=2, minibatches=2, epochs=1),
        metrics="priority",
        verbose=False,
    )
    result = training.train(config, output_dir=tmp_path / "ippo-treatment")
    assert result.completed_env_steps == 256 and result.completed_updates == 16
    assert result.selected_actor is None
    rows = [
        json.loads(line)
        for line in (result.run_dir / "training_updates.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(summaries) == 16
    for row, (task, adjustment, _, _) in zip(rows, summaries, strict=True):
        assert row["task_reward_mean"] == pytest.approx(task, abs=1e-7)
        assert row["shaping_mean"] == pytest.approx(adjustment, abs=1e-7)
        if not shaping:
            assert row["shaping_mean"] == 0
    stages = np.concatenate([summary[2] for summary in summaries])
    starts = np.concatenate([summary[3] for summary in summaries])
    assert np.all((stages[1:] == stages[:-1]) | starts[1:])
    if curriculum:
        assert max(scheduled_stages) > 0
    else:
        np.testing.assert_array_equal(stages, 0)
    details = json.loads((result.run_dir / "run_details.json").read_text())
    assert details["config"]["curriculum"] == curriculum
    assert details["config"]["shaping"] == shaping
    assert details["config"]["method"] == "ippo"
    assert details["config"]["validation_opponents"] is None
    assert details["config"]["pinned_opponent"] is None
    exposure = json.loads((result.run_dir / "exposure.json").read_text())
    assert sum(exposure["steps_by_episode_stage"]) == 256
