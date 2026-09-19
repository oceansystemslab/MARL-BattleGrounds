"""Compare compact BG rollout updates with unchanged donor epoch calculations.

This CPU test expands permitted features only for its independent source
reference. Production receives compact observations and per-game critic inputs.
The fixed synthetic rollout includes two groups, two minibatches, four epochs,
an internal reset and a continuing final bootstrap. It is no learning trial.
"""

# The extracted donor namespace has runtime types, not importable annotations.
# pyright: reportUnknownLambdaType=false
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
from jax import Array
from tests.baseline_donor_reference import build_same_stack_reference
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.baselines.actions import categorical_action_mask, decode_actions
from marl_battlegrounds.baselines.inputs import (
    encode_actor_inputs,
    encode_training_state,
)
from marl_battlegrounds.baselines.ppo import (
    PPOBatch,
    PPOConfig,
    PPOMetrics,
    PPOTrainState,
    initialize_ppo,
    update_recurrent_ppo,
)
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.environment import make
from marl_battlegrounds.policies.input import Observations


def _compare_trees(actual: Any, expected: Any) -> float:  # noqa: ANN401
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    maximum = 0.0
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert left.shape == right.shape and left.dtype == right.dtype
        if np.issubdtype(left.dtype, np.integer):
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, atol=2e-6, rtol=2e-6)
        maximum = max(maximum, float(np.max(np.abs(np.asarray(left) - right))))
    return maximum


def test_compact_whole_sequence_update_matches_original_grouped_epochs() -> None:
    length, games, groups = 4, 8, 2
    config = PPOConfig(rollout_length=length)
    learner = initialize_ppo(jax.random.key(101), config)
    env = make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(5, 5)),
        num_envs=games,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(102))

    def repeat(value: Array) -> Array:
        return jnp.broadcast_to(value, (length, *value.shape))

    # Move each lane differently before collection so the input-routing proof
    # can detect a wrong game or time row. Stored carries/rewards below remain
    # synthetic fixed learner inputs, not a training run.
    observations, state, *_ = env.step(
        jax.random.key(110), state, env.sample_actions(jax.random.key(111), state)
    )
    snapshots: list[Observations] = []
    physical_rows: list[Array] = []
    actor_rows: list[Array] = []
    mask_rows: list[ActionMask] = []
    start_rows: list[Array] = []
    action_rows: list[Array] = []
    for timestep in range(length):
        if timestep == 2:
            observations, state = env.reset(
                jax.random.key(112),
                state=state,
                reset_mask=jnp.arange(games) == 0,
            )
        inputs = env.policy_inputs(observations, state)
        legal = np.asarray(categorical_action_mask(inputs.action_mask))
        choices = np.empty((games, 5), np.int32)
        for game, actor_row in np.ndindex(games, 5):
            candidates = np.flatnonzero(legal[game, actor_row])
            choices[game, actor_row] = candidates[
                (timestep * 13 + game * 7 + actor_row * 3) % len(candidates)
            ]
        snapshots.append(observations)
        physical_rows.append(encode_training_state(state.core_state, state.config))
        actor_rows.append(encode_actor_inputs(inputs.actors))
        mask_rows.append(inputs.action_mask)
        start_rows.append(inputs.episode_start)
        action_rows.append(jnp.asarray(choices))
        opponent = env.policy_inputs(observations, state, team=1)
        zero = jnp.zeros(opponent.active_mask.shape, jnp.int32)
        native = env.join_actions(
            decode_actions(jnp.asarray(choices)), decode_actions(zero)
        )
        observations, state, *_ = env.step(
            jax.random.key(120 + timestep), state, native
        )

    def stack(*values: Array) -> Array:
        return jnp.stack(values)

    compact = jax.tree.map(stack, *snapshots)
    masks = jax.tree.map(stack, *mask_rows)
    physical = jnp.stack(physical_rows)
    expanded = jnp.stack(actor_rows)
    starts = jnp.stack(start_rows)
    # The authored reset is the one synthetic ending in this numerical case.
    ended = jnp.zeros_like(starts).at[1, 0].set(True)
    shape = (length, games, 5)
    rng = np.random.default_rng(731)
    actor_memory = jnp.asarray(rng.normal(0, 0.1, (games, 5, 128)), jnp.float32)
    critic_memory = jnp.asarray(rng.normal(0, 0.1, (games, 5, 128)), jnp.float32)
    rewards = jnp.asarray(rng.normal(0, 0.2, shape), jnp.float32)
    actions = jnp.stack(action_rows)
    assert not np.array_equal(expanded[0, 0], expanded[0, 1])
    assert not np.array_equal(expanded[0], expanded[1])
    assert not np.array_equal(physical[0, 0], physical[0, 1])
    assert not np.array_equal(physical[0], physical[1])
    assert np.unique(actions).size > 1
    final_values = jnp.asarray(rng.normal(0, 0.1, (games, 5)), jnp.float32)

    ns = build_same_stack_reference()
    torso = ns["MLPTorso"]
    actor = ns["RecurrentActor"](
        torso((128,)), torso((128,)), ns["DiscreteActionHead"](198)
    )
    critic = ns["RecurrentValueNet"](
        torso((128,)), torso((128,)), centralised_critic=True
    )
    reference_observation = ns["ObservationGlobalState"](
        expanded,
        categorical_action_mask(masks),
        jnp.broadcast_to(physical[..., None, :], (*shape, 919)),
    )
    actor_starts = jnp.broadcast_to(starts[..., None], shape)
    _, distribution = actor.apply(
        learner.actor_params, actor_memory, (reference_observation, actor_starts)
    )
    _, values = critic.apply(
        learner.critic_params, critic_memory, (reference_observation, actor_starts)
    )
    old_logs = distribution.log_prob(actions) + jnp.asarray(
        rng.normal(0, 0.25, shape), jnp.float32
    )
    old_values = values + jnp.asarray(rng.normal(0, 0.3, shape), jnp.float32)
    batch = PPOBatch(
        compact,
        physical,
        masks,
        actions,
        old_logs,
        old_values,
        rewards,
        ended,
        starts,
        jnp.ones_like(starts),
        jnp.ones(shape, bool),
        jnp.ones(shape, bool),
        final_values,
        actor_memory,
        critic_memory,
    )

    def grouped(value: Array) -> Array:
        return value.reshape(
            length, groups, games // groups, *value.shape[2:]
        ).swapaxes(0, 1)

    memories = ns["HiddenStates"](repeat(actor_memory), repeat(critic_memory))
    trajectory = ns["RNNPPOTransition"](
        actor_starts,
        actions,
        old_values,
        rewards,
        old_logs,
        reference_observation,
        memories,
    )
    trajectory = jax.tree.map(grouped, trajectory)
    bootstrap = final_values.reshape(groups, games // groups, 5)
    final_done = jnp.zeros_like(bootstrap, bool)
    advantages, targets = jax.vmap(ns["calculate_gae"], in_axes=(0, 0, 0, None, None))(
        trajectory, bootstrap, final_done, config.gamma, config.gae_lambda
    )
    ns["config"] = SimpleNamespace(
        system=SimpleNamespace(
            rollout_length=length,
            recurrent_chunk_size=length,
            num_minibatches=2,
            clip_eps=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
        ),
        arch=SimpleNamespace(num_envs=games // groups),
    )
    ns["actor_apply_fn"], ns["critic_apply_fn"] = actor.apply, critic.apply
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5), optax.adam(0.00025, eps=1e-5)
    )
    ns["actor_update_fn"] = ns["critic_update_fn"] = optimizer.update
    params = ns["Params"](learner.actor_params, learner.critic_params)
    opt_states = ns["OptStates"](learner.actor_opt_state, learner.critic_opt_state)

    def copies(value: Array) -> Array:
        return jnp.broadcast_to(value, (1, groups, *value.shape))

    def add_device(value: Array) -> Array:
        return value[None]

    def remove_device_group(value: Array) -> Array:
        return value[0, 0]

    @jax.jit
    def update(
        s: PPOTrainState, b: PPOBatch, k: Array
    ) -> tuple[PPOTrainState, PPOMetrics]:
        return update_recurrent_ppo(s, b, k, config)

    key = jax.random.key(103)
    reference_state = (
        jax.tree.map(copies, params),
        jax.tree.map(copies, opt_states),
        jax.tree.map(add_device, trajectory),
        advantages[None],
        targets[None],
        jax.random.split(key, groups)[None],
    )
    epoch = jax.vmap(
        jax.vmap(ns["_update_epoch"], in_axes=(0, None), axis_name="batch"),
        in_axes=(0, None),
        axis_name="device",
    )
    reference, losses = jax.lax.scan(epoch, reference_state, None, length=4)
    actual, metrics = cast(
        tuple[PPOTrainState, PPOMetrics], update(learner, batch, key)
    )
    expected_params = jax.tree.map(remove_device_group, reference[0])
    expected_opt = jax.tree.map(remove_device_group, reference[1])
    maximum = max(
        _compare_trees(actual.actor_params, expected_params.actor_params),
        _compare_trees(actual.critic_params, expected_params.critic_params),
        _compare_trees(actual.actor_opt_state, expected_opt.actor_opt_state),
        _compare_trees(actual.critic_opt_state, expected_opt.critic_opt_state),
    )
    for label in ("actor_loss", "entropy", "value_loss"):
        np.testing.assert_allclose(
            getattr(metrics, label), losses[label][:, 0, 0], atol=2e-6, rtol=2e-6
        )
    np.testing.assert_array_equal(metrics.actor_samples, np.full((4, 2), 80))
    print(f"Compact versus donor maximum parameter/Adam error: {maximum:.9g}")
