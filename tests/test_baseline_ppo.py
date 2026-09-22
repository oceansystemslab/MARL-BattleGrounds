"""Check BG recurrent MAPPO information, decision timing and masked updates.

These CPU tests use real-width actor Systems through public environment calls,
including legal action probabilities, death, respawn, reset and padding. Small
fixed donor tensors isolate group averaging and complete Adam-state preservation.
Input-scale checks pair inference and PPO with explicit feature multiplication,
preserve default parameters, and distinguish recorded inference settings.
Spawn-frame checks reject bad names including the removed "right", keep
parameter bytes, make each frame a distinct System identity with a closure-free
hook whose numerical keyword defaults are recorded for every factory System, show
that the factory's default is the unscaled "left" hook while an explicit "world"
System uses the direct actor hook, and prove that a "left" System equals the
actor run on explicitly reflected features with its world-frame actions and
stored indices agreeing while its unflagged lanes stay bit-identical to an
explicit "world" System. They do not establish GPU speed, learning
quality or full learner recovery.
"""

from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
from jax import Array
from jax.typing import ArrayLike
from tests.baseline_donor_reference import load_reference, reference_tree
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.baselines.actions import (
    action_log_prob,
    categorical_action_mask,
    encode_actions,
    mirror_action_indices,
)
from marl_battlegrounds.baselines.inputs import (
    encode_actor_inputs,
    encode_training_state,
    spawn_frame_flag,
)
from marl_battlegrounds.baselines.ppo import (
    HIDDEN_SIZE,
    PPOConfig,
    PPOLearningOutputs,
    PPOMetrics,
    PPOMinibatch,
    PPOTrainState,
    RecurrentActor,
    _apply_actor,  # pyright: ignore[reportPrivateUsage]
    calculate_gae,
    critic_values,
    initialize_ppo,
    make_recurrent_mappo_system,
    update_minibatch,
)
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import AGENT_FEATURE_X, TASK_MODE_TDM, Action
from marl_battlegrounds.environment import Environment, EnvironmentState, make
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
    SystemState,
    apply_systems,
    init_systems,
)
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations, mirror_team_view
from marl_battlegrounds.tasks import balanced_spawn_configs

type Tree = Any


def _asarray(value: ArrayLike) -> Array:
    return jnp.asarray(value)


def _assert_tree_equal(actual: Tree, expected: Tree) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if jnp.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        np.testing.assert_array_equal(left, right)


def _assert_tree_close(actual: Tree, expected: Tree) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(left, right, atol=2e-7, rtol=2e-6)


def _tree_changed(actual: Tree, expected: Tree) -> bool:
    return any(
        not np.array_equal(left, right)
        for left, right in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        )
    )


@pytest.fixture(scope="module")
def networks() -> PPOTrainState:
    state = initialize_ppo(jax.random.key(91))
    actor = jax.tree.map(_asarray, state.actor_params)
    actor["params"]["pre_torso"]["Dense_0"]["kernel"] *= 0.01
    return state._replace(actor_params=actor)


def _setup(
    *, batch: int | None = 2, max_steps: int = 8
) -> tuple[Environment, Observations, EnvironmentState]:
    env = make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(2, 2), max_steps=max_steps),
        num_envs=batch,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(10))
    return env, observations, state


def _actor_output(
    system: System, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    return cast(SystemOutput, system.apply(system.variables, memory, inputs, keys))


def _actor_forward(
    params: Tree, memory: Array, inputs: SystemInput
) -> tuple[Array, Array]:
    valid = jnp.broadcast_to(inputs.valid[:, None], inputs.active_mask.shape)
    starts = jnp.broadcast_to(inputs.episode_start[:, None], inputs.active_mask.shape)
    next_memory, logits = cast(
        tuple[Array, Array],
        RecurrentActor().apply(
            params,
            memory,
            encode_actor_inputs(inputs.actors)[None],
            starts[None],
            valid[None],
        ),
    )
    return next_memory, logits[0]


def _forced_actor(params: Tree, category: int) -> Tree:
    result = jax.tree.map(_asarray, params)
    head = result["params"]["action_head"]["Dense_0"]
    head["kernel"] = jnp.zeros_like(head["kernel"])
    head["bias"] = jnp.full_like(head["bias"], -1000).at[category].set(1000)
    return result


def test_actor_inputs_and_memory_stay_separate_for_each_recipient(
    networks: PPOTrainState,
) -> None:
    env, observations, state = _setup(batch=1)
    inputs = env.policy_inputs(observations, state)._replace(
        episode_start=jnp.zeros((1,), jnp.bool_)
    )
    system = make_recurrent_mappo_system(networks.actor_params)
    memory = jnp.zeros((1, 5, HIDDEN_SIZE), jnp.float32)
    keys = jax.random.split(jax.random.key(20), 1)
    original = _actor_output(system, memory, inputs, keys)
    observation = inputs.actors.observation
    changed = inputs._replace(
        actors=inputs.actors._replace(
            observation=observation._replace(
                self_features=observation.self_features.at[0, 1, AGENT_FEATURE_X].add(
                    17
                )
            )
        )
    )
    changed_output = _actor_output(system, memory.at[0, 1].set(0.3), changed, keys)
    np.testing.assert_array_equal(
        original.next_memory[0, 0], changed_output.next_memory[0, 0]
    )
    assert not np.array_equal(
        original.next_memory[0, 1], changed_output.next_memory[0, 1]
    )
    for left, right in zip(original.actions, changed_output.actions, strict=True):
        np.testing.assert_array_equal(left[0, 0], right[0, 0])
    np.testing.assert_array_equal(
        original.learning_outputs.log_prob[0, 0],
        changed_output.learning_outputs.log_prob[0, 0],
    )
    legacy = jax.random.key_data(keys)
    _assert_tree_equal(_actor_output(system, memory, inputs, legacy), original)


def test_privileged_state_and_critic_memory_do_not_enter_actor_system(
    networks: PPOTrainState,
) -> None:
    _, observations, state = _setup(batch=1)
    system = make_recurrent_mappo_system(networks.actor_params)
    assert system.variables is networks.actor_params
    memory = init_systems(system, system, observations, state, jax.random.key(21))
    key = jax.random.key(22)
    expected = apply_systems(system, system, memory, observations, state, key)
    hidden = state._replace(
        core_state=state.core_state._replace(
            current_health=state.core_state.current_health + 7
        )
    )
    changed = apply_systems(system, system, memory, observations, hidden, key)
    _assert_tree_equal(changed, expected)
    features = encode_training_state(state.core_state, state.config)[None]
    changed_features = encode_training_state(hidden.core_state, hidden.config)[None]
    flags = jnp.zeros((1, 1), jnp.bool_)
    valid = jnp.ones_like(flags)
    zero = jnp.zeros((1, 5, HIDDEN_SIZE), jnp.float32)
    first_critic = critic_values(networks.critic_params, zero, features, flags, valid)
    second_critic = critic_values(
        networks.critic_params, zero + 0.4, changed_features, flags, valid
    )
    assert _tree_changed(first_critic, second_critic)
    _assert_tree_equal(
        apply_systems(system, system, memory, observations, state, key), expected
    )


def test_public_decision_keeps_legal_submitted_actions_log_probabilities_and_next_epoch(
    networks: PPOTrainState,
) -> None:
    env, observations, state = _setup(batch=1)
    system = make_recurrent_mappo_system(networks.actor_params)
    memory = init_systems(system, system, observations, state, jax.random.key(25))
    for tick in range(2):
        inputs = env.policy_inputs(observations, state)
        expected_memory, logits = _actor_forward(
            networks.actor_params, memory.team_a, inputs
        )
        before_step = np.asarray(state.core_state.step_count).copy()
        actions, next_memory, outputs = apply_systems(
            system, system, memory, observations, state, jax.random.key(30 + tick)
        )
        learner = cast(PPOLearningOutputs, outputs[0])
        submitted = ActorAction(*(head[:, :5] for head in actions))
        np.testing.assert_array_equal(learner.action_indices, encode_actions(submitted))
        masks = categorical_action_mask(inputs.action_mask)
        assert np.all(
            np.take_along_axis(
                np.asarray(masks),
                np.asarray(learner.action_indices)[..., None],
                axis=-1,
            )
        )
        np.testing.assert_allclose(
            learner.log_prob,
            action_log_prob(logits, masks, learner.action_indices),
            atol=2e-6,
            rtol=2e-6,
        )
        np.testing.assert_allclose(
            next_memory.team_a, expected_memory, atol=2e-6, rtol=2e-6
        )
        observations, state, *_ = env.step(jax.random.key(40 + tick), state, actions)
        accepted = Action(
            state.core_state.previous_timestep_move_actions,
            state.core_state.previous_timestep_select_target_actions,
            state.core_state.previous_timestep_use_ultimate_actions,
        )
        _assert_tree_equal(accepted, actions)
        np.testing.assert_array_equal(state.core_state.step_count, before_step + 1)
        next_inputs = env.policy_inputs(observations, state)
        np.testing.assert_array_equal(
            next_inputs.action_mask.move_mask, state.action_mask.move_mask[:, :5]
        )
        np.testing.assert_array_equal(
            next_inputs.actors.observation.self_features,
            observations.observation.self_features[:, :5],
        )
        memory = next_memory


def test_public_death_and_respawn_keep_actor_recurrent_memory(
    networks: PPOTrainState,
) -> None:
    config = evaluation_env_config(
        team_sizes=(1, 1),
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=20,
        max_steps=20,
    )._replace(
        spawn_shield_duration_steps=0,
        team_respawn_wave_period_step_count=jnp.asarray((2, 2), jnp.int32),
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
    prepared = core.initialize_scenario_state(authored, config)
    env = make("tdm", env_config=config, metrics="none")
    observations, state = env.reset(jax.random.key(51), initial=prepared[:3])
    params = _forced_actor(networks.actor_params, 12)
    system = make_recurrent_mappo_system(params)
    memory = init_systems(system, system, observations, state, jax.random.key(52))
    for decision in range(3):
        inputs = env.policy_inputs(observations, state)
        expected, _ = _actor_forward(params, memory.team_a, inputs)
        reset_memory, _ = _actor_forward(params, jnp.zeros_like(memory.team_a), inputs)
        actions, next_memory, learning = apply_systems(
            system, system, memory, observations, state, jax.random.key(60 + decision)
        )
        np.testing.assert_allclose(next_memory.team_a, expected, atol=2e-6, rtol=2e-6)
        if decision:
            assert not np.allclose(
                next_memory.team_a[0, 0], reset_memory[0, 0], atol=1e-6, rtol=1e-6
            )
            assert not bool(inputs.episode_start[0])
        if decision == 0:
            np.testing.assert_array_equal(
                actions.select_target[jnp.asarray((0, 5))], (6, 6)
            )
        if decision == 1:
            np.testing.assert_array_equal(
                cast(PPOLearningOutputs, learning[0]).action_indices,
                jnp.zeros((1, 5), jnp.int32),
            )
            np.testing.assert_array_equal(
                cast(PPOLearningOutputs, learning[0]).log_prob,
                jnp.zeros((1, 5), jnp.float32),
            )
        observations, state, *_ = env.step(
            jax.random.key(70 + decision), state, actions
        )
        if decision < 2:
            np.testing.assert_array_equal(
                state.core_state.alive_mask[jnp.asarray((0, 5))],
                (decision == 1, decision == 1),
            )
        memory = next_memory
    assert int(state.reset_generation) == 0
    assert int(state.episode_id) == 1


def test_public_partial_reset_and_padding_preserve_only_continuing_memory(
    networks: PPOTrainState,
) -> None:
    env, _, state = _setup(batch=2)
    observations, state = env.reset(
        jax.random.key(81),
        state.config._replace(max_steps=jnp.asarray((1, 3), jnp.int32)),
    )
    system = make_recurrent_mappo_system(_forced_actor(networks.actor_params, 0))
    memory = init_systems(system, system, observations, state, jax.random.key(82))
    actions, memory, _ = apply_systems(
        system, system, memory, observations, state, jax.random.key(83)
    )
    observations, state, *_ = env.step(jax.random.key(84), state, actions)
    np.testing.assert_array_equal(state.done.done, (True, False))
    before = memory
    _, padded, _ = apply_systems(
        system, system, memory, observations, state, jax.random.key(85)
    )
    np.testing.assert_array_equal(padded.team_a[0], before.team_a[0])
    assert not np.array_equal(padded.team_a[1], before.team_a[1])
    observations, state = env.reset_done(jax.random.key(86), state)
    inputs = env.policy_inputs(observations, state)
    expected, _ = _actor_forward(system.variables, before.team_a, inputs)
    _, reset, _ = apply_systems(
        system, system, before, observations, state, jax.random.key(87)
    )
    np.testing.assert_allclose(reset.team_a, expected, atol=2e-6, rtol=2e-6)
    fresh, _ = _actor_forward(system.variables, jnp.zeros_like(before.team_a), inputs)
    np.testing.assert_allclose(reset.team_a[0], fresh[0], atol=2e-6, rtol=2e-6)
    assert not np.allclose(reset.team_a[1], fresh[1], atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(inputs.episode_start, (True, False))


def test_system_dynamic_weights_reuse_compilation(networks: PPOTrainState) -> None:
    _, observations, state = _setup(batch=1)
    system = make_recurrent_mappo_system(networks.actor_params)
    memory = init_systems(system, system, observations, state, jax.random.key(90))
    traces: list[int] = []

    @jax.jit
    def run(
        params: Tree, carry: SystemState, key: Array
    ) -> tuple[Action, SystemState, tuple[Tree, Tree]]:
        traces.append(1)
        return apply_systems(
            system,
            system,
            carry,
            observations,
            state,
            key,
            variables_a=params,
            variables_b=params,
        )

    first = cast(
        tuple[Action, SystemState, tuple[Tree, Tree]],
        run(networks.actor_params, memory, jax.random.key(92)),
    )
    changed = jax.tree.map(_asarray, networks.actor_params)
    changed["params"]["action_head"]["Dense_0"]["bias"] = (
        jnp.arange(198, dtype=jnp.float32) * 0.01
    )
    second = cast(
        tuple[Action, SystemState, tuple[Tree, Tree]],
        run(changed, memory, jax.random.key(92)),
    )
    assert traces == [1]
    assert not np.array_equal(first[2][0].log_prob, second[2][0].log_prob)
    np.testing.assert_array_equal(first[1].team_a, second[1].team_a)


def test_gae_stops_at_real_endings_bootstraps_cutoffs_and_ignores_padding() -> None:
    rewards = jnp.broadcast_to(
        jnp.asarray(((1, 1), (2, 2), (999, 3)), jnp.float32)[..., None], (3, 2, 5)
    )
    values = jnp.broadcast_to(
        jnp.asarray(((10, 10), (20, 20), (999, 30)), jnp.float32)[..., None], (3, 2, 5)
    )
    ended = jnp.asarray(((False, False), (True, False), (False, False)))
    valid = jnp.asarray(((True, True), (True, True), (False, True)))
    advantages, targets = calculate_gae(
        rewards,
        values,
        ended,
        jnp.full((2, 5), 40, jnp.float32),
        valid=valid,
        gamma=1.0,
        gae_lambda=1.0,
    )
    expected_targets = jnp.broadcast_to(
        jnp.asarray(((3, 46), (2, 45), (0, 43)), jnp.float32)[..., None], (3, 2, 5)
    )
    np.testing.assert_array_equal(targets, expected_targets)
    np.testing.assert_array_equal(
        advantages, jnp.where(valid[..., None], expected_targets - values, 0)
    )
    horizon = calculate_gae(
        rewards[:, 1:],
        values[:, 1:],
        jnp.asarray(((False,), (False,), (True,))),
        jnp.full((1, 5), 40, jnp.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )[1]
    np.testing.assert_array_equal(horizon[:, 0, 0], (6, 5, 3))
    episode_returns = calculate_gae(
        jnp.ones((4, 1, 5), jnp.float32),
        jnp.zeros((4, 1, 5), jnp.float32),
        jnp.asarray(((False,), (True,), (False,), (False,))),
        jnp.full((1, 5), 10, jnp.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )[1]
    np.testing.assert_array_equal(episode_returns[:, 0, 0], (2, 1, 12, 11))


@pytest.fixture(scope="module")
def numerical_batch() -> tuple[PPOTrainState, PPOMinibatch, PPOConfig]:
    _, arrays = load_reference()
    actor = jax.tree.map(_asarray, reference_tree(arrays, "parameters/actor"))
    critic = jax.tree.map(_asarray, reference_tree(arrays, "parameters/critic"))
    config = PPOConfig(rollout_length=4)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.actor_lr, eps=config.adam_epsilon),
    )
    state = PPOTrainState(actor, critic, optimizer.init(actor), optimizer.init(critic))
    valid = jnp.ones_like(jnp.asarray(arrays["input/episode_start"]), dtype=jnp.bool_)
    batch = PPOMinibatch(
        jnp.asarray(arrays["input/actor_features"]),
        jnp.asarray(arrays["input/critic_features"]),
        jnp.asarray(arrays["input/action_mask"]),
        jnp.asarray(arrays["input/actions"]),
        jnp.asarray(arrays["input/old_log_probabilities"]),
        jnp.asarray(arrays["input/old_values"]),
        jnp.asarray(arrays["expected/advantages"]),
        jnp.asarray(arrays["expected/targets"]),
        jnp.asarray(arrays["input/episode_start"]),
        valid,
        valid,
        valid,
        jnp.asarray(arrays["input/actor_carry"]),
        jnp.asarray(arrays["input/critic_carry"]),
    )
    return state, batch, config


_compiled_update = jax.jit(update_minibatch, static_argnames=("config",))


def _update(
    state: PPOTrainState, batch: PPOMinibatch, config: PPOConfig
) -> tuple[PPOTrainState, PPOMetrics]:
    return cast(
        tuple[PPOTrainState, PPOMetrics], _compiled_update(state, batch, config)
    )


def _group(batch: PPOMinibatch, group: int) -> PPOMinibatch:
    def take(value: Array) -> Array:
        return value[group : group + 1]

    return jax.tree.map(take, batch)


def test_empty_samples_preserve_nonzero_adam_memory_and_skip_networks_independently(
    numerical_batch: tuple[PPOTrainState, PPOMinibatch, PPOConfig],
) -> None:
    state, batch, config = numerical_batch
    warmed, _ = _update(state, batch, config)
    assert any(
        np.any(value) for value in jax.tree.leaves(warmed.actor_opt_state[1][0].mu)
    )
    assert any(
        np.any(value) for value in jax.tree.leaves(warmed.critic_opt_state[1][0].mu)
    )
    empty = jnp.zeros_like(batch.actor_samples)
    unchanged, metrics = _update(
        warmed, batch._replace(actor_samples=empty, critic_samples=empty), config
    )
    _assert_tree_equal(unchanged, warmed)
    assert all(np.all(np.asarray(value) == 0) for value in metrics)
    critic_only, _ = _update(warmed, batch._replace(actor_samples=empty), config)
    _assert_tree_equal(
        (critic_only.actor_params, critic_only.actor_opt_state),
        (warmed.actor_params, warmed.actor_opt_state),
    )
    assert _tree_changed(critic_only.critic_opt_state, warmed.critic_opt_state)
    actor_only, _ = _update(warmed, batch._replace(critic_samples=empty), config)
    _assert_tree_equal(
        (actor_only.critic_params, actor_only.critic_opt_state),
        (warmed.critic_params, warmed.critic_opt_state),
    )
    assert _tree_changed(actor_only.actor_opt_state, warmed.actor_opt_state)


def test_nonempty_groups_are_selected_separately_for_actor_and_critic(
    numerical_batch: tuple[PPOTrainState, PPOMinibatch, PPOConfig],
) -> None:
    state, batch, config = numerical_batch
    split = batch._replace(
        actor_samples=batch.actor_samples.at[1].set(False),
        critic_samples=batch.critic_samples.at[0].set(False),
    )
    actual, metrics = _update(state, split, config)
    one = replace(config, groups=1)
    actor, actor_metrics = _update(state, _group(batch, 0), one)
    critic, critic_metrics = _update(state, _group(batch, 1), one)
    _assert_tree_close(
        (actual.actor_params, actual.actor_opt_state),
        (actor.actor_params, actor.actor_opt_state),
    )
    _assert_tree_close(
        (actual.critic_params, actual.critic_opt_state),
        (critic.critic_params, critic.critic_opt_state),
    )
    np.testing.assert_allclose(
        metrics.actor_loss, actor_metrics.actor_loss, atol=2e-7, rtol=2e-6
    )
    np.testing.assert_allclose(
        metrics.value_loss, critic_metrics.value_loss, atol=2e-7, rtol=2e-6
    )
    assert int(metrics.actor_samples) == int(actor_metrics.actor_samples)
    assert int(metrics.critic_samples) == int(critic_metrics.critic_samples)


def test_excluded_samples_do_not_change_statistics_and_dead_members_still_train_values(
    numerical_batch: tuple[PPOTrainState, PPOMinibatch, PPOConfig],
) -> None:
    state, batch, config = numerical_batch
    batch = batch._replace(
        actor_samples=batch.actor_samples.at[..., 3:].set(False),
        critic_samples=batch.critic_samples.at[..., 3].set(False),
    )
    expected, expected_metrics = _update(state, batch, config)
    changed = batch._replace(
        advantages=batch.advantages.at[..., 3:].set(50_000),
        old_log_prob=batch.old_log_prob.at[..., 3:].set(2),
        targets=batch.targets.at[..., 3].set(-50_000),
        old_values=batch.old_values.at[..., 3].set(10_000),
    )
    actual, metrics = _update(state, changed, config)
    _assert_tree_equal(actual, expected)
    _assert_tree_equal(metrics, expected_metrics)
    dead_value, _ = _update(
        state, changed._replace(targets=changed.targets.at[..., 4].add(10)), config
    )
    _assert_tree_equal(dead_value.actor_params, expected.actor_params)
    assert _tree_changed(dead_value.critic_opt_state, expected.critic_opt_state)


def test_unequal_nonempty_groups_keep_equal_group_weight(
    numerical_batch: tuple[PPOTrainState, PPOMinibatch, PPOConfig],
) -> None:
    state, batch, config = numerical_batch
    config = replace(config, max_grad_norm=1_000_000.0)
    samples = batch.actor_samples.at[1, :, 1:].set(False)
    batch = batch._replace(actor_samples=samples, critic_samples=samples)
    combined, metrics = _update(state, batch, config)
    first, first_metrics = _update(state, _group(batch, 0), replace(config, groups=1))
    second, second_metrics = _update(state, _group(batch, 1), replace(config, groups=1))

    def average(left: Array, right: Array) -> Array:
        return (left + right) / 2

    for field in ("actor_opt_state", "critic_opt_state"):
        left = getattr(first, field)[1][0].mu
        right = getattr(second, field)[1][0].mu
        actual = getattr(combined, field)[1][0].mu
        _assert_tree_close(actual, jax.tree.map(average, left, right))
    np.testing.assert_allclose(
        metrics.actor_loss,
        (first_metrics.actor_loss + second_metrics.actor_loss) / 2,
        atol=2e-7,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        metrics.value_loss,
        (first_metrics.value_loss + second_metrics.value_loss) / 2,
        atol=2e-7,
        rtol=2e-6,
    )
    assert int(first_metrics.actor_samples) == 4 * int(second_metrics.actor_samples)
    weighted = (4 * first_metrics.value_loss + second_metrics.value_loss) / 5
    assert not np.isclose(metrics.value_loss, weighted, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "scale", [0.0, -0.01, float("nan"), float("inf"), -float("inf"), True, "0.01", None]
)
def test_input_scale_rejects_invalid_static_settings(scale: object) -> None:
    with pytest.raises(ValueError, match="input_scale"):
        PPOConfig(input_scale=cast(float, scale))
    with pytest.raises(ValueError, match="input_scale"):
        make_recurrent_mappo_system({}, input_scale=cast(float, scale))


def test_input_scale_keeps_initial_parameter_bytes_and_optimizer_state() -> None:
    key = jax.random.key(145)
    default = initialize_ppo(key)
    explicit = initialize_ppo(key, PPOConfig(input_scale=1.0))
    scaled = initialize_ppo(key, PPOConfig(input_scale=0.01))
    _assert_tree_equal(default, explicit)
    _assert_tree_equal(default, scaled)


def test_input_scale_is_part_of_system_identity_and_reuses_its_apply_hook() -> None:
    raw = make_recurrent_mappo_system({})
    explicit = make_recurrent_mappo_system({}, input_scale=1.0)
    scaled = make_recurrent_mappo_system({}, input_scale=0.01)
    repeated = make_recurrent_mappo_system({}, input_scale=0.01)
    other = make_recurrent_mappo_system({}, input_scale=0.02)
    assert raw.apply is explicit.apply
    assert scaled.apply is repeated.apply
    records = [
        normalize_system_registration(value, phase="evaluation", frozen=True)
        for value in (raw, explicit, scaled, repeated, other)
    ]
    assert records[0] == records[1]
    assert records[2] == records[3]
    assert len({records[index][0] for index in (0, 2, 4)}) == 3
    assert len({record[1]["variables_digest"] for record in records}) == 1
    hooks = [cast(dict[str, Any], record[1]["hooks"])["apply"] for record in records]
    assert hooks[2]["closure_content"] == "none"
    assert hooks[2]["defaults_digest"] != hooks[4]["defaults_digest"]


def test_scaled_system_and_critic_match_explicit_feature_multiplication(
    networks: PPOTrainState,
) -> None:
    env, observations, state = _setup(batch=1)
    inputs = env.policy_inputs(observations, state)
    features = encode_actor_inputs(inputs.actors)[None]
    starts = jnp.broadcast_to(inputs.episode_start[:, None], inputs.active_mask.shape)
    valid = jnp.broadcast_to(inputs.valid[:, None], inputs.active_mask.shape)
    memory = jnp.full((1, 5, HIDDEN_SIZE), 0.1, jnp.float32)
    system = make_recurrent_mappo_system(networks.actor_params, input_scale=0.01)
    output = _actor_output(
        system, memory, inputs, jax.random.split(jax.random.key(146), 1)
    )
    expected_memory, logits = cast(
        tuple[Array, Array],
        RecurrentActor().apply(
            networks.actor_params, memory, features * 0.01, starts[None], valid[None]
        ),
    )
    _assert_tree_close(output.next_memory, expected_memory)
    np.testing.assert_allclose(
        output.learning_outputs.log_prob,
        action_log_prob(
            logits[0],
            categorical_action_mask(inputs.action_mask),
            output.learning_outputs.action_indices,
        ),
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_array_equal(encode_actor_inputs(inputs.actors)[None], features)
    physical = encode_training_state(state.core_state, state.config)[None]
    flags = jnp.zeros((1, 1), jnp.bool_)
    present = jnp.ones_like(flags)
    scaled = critic_values(
        networks.critic_params, memory, physical, flags, present, input_scale=0.01
    )
    expected = critic_values(
        networks.critic_params, memory, physical * 0.01, flags, present
    )
    _assert_tree_equal(scaled, expected)
    assert _tree_changed(
        scaled, critic_values(networks.critic_params, memory, physical, flags, present)
    )


def test_scaled_ppo_losses_and_gradients_match_explicit_scaled_features(
    numerical_batch: tuple[PPOTrainState, PPOMinibatch, PPOConfig],
) -> None:
    state, batch, config = numerical_batch
    actual = _update(state, batch, replace(config, input_scale=0.01))
    expected = _update(
        state,
        batch._replace(
            actor_features=batch.actor_features * 0.01,
            critic_features=batch.critic_features * 0.01,
        ),
        config,
    )
    _assert_tree_close(actual, expected)
    assert _tree_changed(actual, _update(state, batch, config))


def _balanced_setup(
    max_steps: int = 8,
) -> tuple[Environment, Observations, EnvironmentState]:
    config = balanced_spawn_configs(
        evaluation_env_config(team_sizes=(2, 2), max_steps=max_steps), num_envs=2
    )
    env = make("tdm", env_config=config, num_envs=2, metrics="none")
    observations, state = env.reset(jax.random.key(12))
    return env, observations, state


@pytest.mark.parametrize("frame", ["World", "right", "up", "", None, 0, True, b"left"])
def test_spawn_frame_rejects_invalid_static_settings(frame: object) -> None:
    with pytest.raises(ValueError, match="spawn_frame"):
        PPOConfig(spawn_frame=cast(str, frame))
    with pytest.raises(ValueError, match="spawn_frame"):
        make_recurrent_mappo_system({}, spawn_frame=cast(str, frame))


def test_spawn_frame_keeps_initial_parameter_bytes() -> None:
    key = jax.random.key(147)
    default = initialize_ppo(key)
    for frame in ("world", "left"):
        _assert_tree_equal(default, initialize_ppo(key, PPOConfig(spawn_frame=frame)))


def test_spawn_frame_is_part_of_system_identity_and_reuses_its_apply_hook() -> None:
    default = make_recurrent_mappo_system({})
    unscaled_left = make_recurrent_mappo_system({}, spawn_frame="left")
    world = make_recurrent_mappo_system({}, spawn_frame="world")
    scaled_left = make_recurrent_mappo_system({}, input_scale=0.01)
    repeated = make_recurrent_mappo_system({}, input_scale=0.01, spawn_frame="left")
    scaled_world = make_recurrent_mappo_system(
        {}, input_scale=0.01, spawn_frame="world"
    )
    assert default.apply is unscaled_left.apply
    assert world.apply is _apply_actor
    assert scaled_left.apply is repeated.apply
    records = [
        normalize_system_registration(value, phase="evaluation", frozen=True)
        for value in (
            default,
            unscaled_left,
            world,
            scaled_left,
            repeated,
            scaled_world,
        )
    ]
    assert records[0] == records[1]
    assert records[3] == records[4]
    assert len({records[index][0] for index in (0, 2, 3, 5)}) == 4
    hooks = [cast(dict[str, Any], record[1]["hooks"])["apply"] for record in records]
    assert all(hook["closure_content"] == "none" for hook in hooks)
    digests = [hook["defaults_digest"] for hook in hooks]
    assert all(digest is not None for digest in digests)
    assert len({digests[index] for index in (0, 2, 3, 5)}) == 4
    assert default.apply.__kwdefaults__ == {"input_scale": 1.0, "spawn_frame_index": 1}
    assert world.apply.__kwdefaults__ == {"input_scale": 1.0, "spawn_frame_index": 0}


@pytest.mark.parametrize("frame", ["left"])
def test_reflected_system_matches_explicit_reflected_features_and_world_actions(
    networks: PPOTrainState, frame: str
) -> None:
    env, observations, state = _balanced_setup()
    inputs = env.policy_inputs(observations, state)
    flag = spawn_frame_flag(inputs.actors, frame)
    assert bool(flag.any()) and not bool(flag.all())
    actors, native_mask = mirror_team_view(inputs.actors, inputs.action_mask, flag)
    memory = jnp.full((2, 5, HIDDEN_SIZE), 0.1, jnp.float32)
    keys = jax.random.split(jax.random.key(148), 2)
    system = make_recurrent_mappo_system(
        networks.actor_params, input_scale=0.01, spawn_frame=frame
    )
    output = _actor_output(system, memory, inputs, keys)
    starts = jnp.broadcast_to(inputs.episode_start[:, None], inputs.active_mask.shape)
    valid = jnp.broadcast_to(inputs.valid[:, None], inputs.active_mask.shape)
    expected_memory, logits = cast(
        tuple[Array, Array],
        RecurrentActor(input_scale=0.01).apply(
            networks.actor_params,
            memory,
            encode_actor_inputs(actors)[None],
            starts[None],
            valid[None],
        ),
    )
    _assert_tree_close(output.next_memory, expected_memory)
    mask = categorical_action_mask(native_mask)
    reflected_indices = mirror_action_indices(
        output.learning_outputs.action_indices, flag
    )
    np.testing.assert_allclose(
        output.learning_outputs.log_prob,
        action_log_prob(logits[0], mask, reflected_indices),
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_array_equal(
        output.learning_outputs.action_indices, encode_actions(output.actions)
    )
    legal = categorical_action_mask(inputs.action_mask)
    assert bool(
        jnp.all(
            jnp.take_along_axis(
                legal, output.learning_outputs.action_indices[..., None], axis=-1
            )
        )
    )
    plain = _actor_output(
        make_recurrent_mappo_system(
            networks.actor_params, input_scale=0.01, spawn_frame="world"
        ),
        memory,
        inputs,
        keys,
    )
    unflagged = ~flag
    for name in ("move", "select_target", "use_ultimate"):
        np.testing.assert_array_equal(
            np.asarray(getattr(output.actions, name))[np.asarray(unflagged)],
            np.asarray(getattr(plain.actions, name))[np.asarray(unflagged)],
        )
    np.testing.assert_array_equal(
        np.asarray(output.learning_outputs.log_prob)[np.asarray(unflagged)],
        np.asarray(plain.learning_outputs.log_prob)[np.asarray(unflagged)],
    )
    np.testing.assert_array_equal(
        np.asarray(output.next_memory)[np.asarray(unflagged)],
        np.asarray(plain.next_memory)[np.asarray(unflagged)],
    )
