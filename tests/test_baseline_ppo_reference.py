"""Compare the recurrent PPO port with independently preserved donor calculations.

The fixed CPU fixture uses two groups, four time steps, four games per group,
five actors, legal 198-way choices, nonzero memory, resets and terminal flags.
These tests compare exact parameter paths and float32 shapes/dtypes, current-
stack initialization, forward outputs, GAE, losses, gradients and separate Adam
states through all four epochs. Original source bodies provide a second live
reference on this stack; no network access or historical package install is
needed. All samples are present. BG padding/death/permission adaptations have
separate semantic tests and are not used to excuse a donor difference here.

The numerical bounds are 2e-6 absolute and 2e-6 relative. The absolute bound
covers float32 reduction and historical library differences near zero; the
relative bound covers accumulated updates away from zero. Integer state and
parameter-tree identity are exact. Tests record the observed maximum errors;
these checks do not establish learning quality or GPU execution costs.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray
from tests.baseline_donor_reference import (
    _store_tree,
    build_same_stack_reference,
    load_reference,
    reference_tree,
)

from marl_battlegrounds.baselines import ppo
from marl_battlegrounds.baselines.actions import action_entropy, action_log_prob

# Private numerical boundaries expose gradients without changing the public API.
# pyright: reportPrivateUsage=false
type Tree = Any
type RecordProperty = Callable[[str, object], None]
_ATOL = 2e-6
_RTOL = 2e-6
_CONFIG = ppo.PPOConfig(rollout_length=4, value_normalization=False)


def _asarray(value: NDArray[Any]) -> Array:
    return jnp.asarray(value)


def _first_group(value: Array) -> Array:
    return value[0]


def _first_half(value: Array) -> Array:
    return value[:, :, :2]


def _group_mean(value: Array) -> Array:
    return value.mean(0)


@dataclass
class Reference:
    metadata: dict[str, Any]
    arrays: dict[str, NDArray[Any]]
    inputs: dict[str, Array]
    namespace: dict[str, Any]
    actor: Any
    critic: Any
    trajectory: Any
    state: ppo.PPOTrainState
    batch: ppo.PPOMinibatch


@pytest.fixture(scope="module")
def reference() -> Reference:
    metadata, arrays = load_reference()
    inputs = {
        key.removeprefix("input/"): jnp.asarray(value)
        for key, value in arrays.items()
        if key.startswith("input/")
    }
    ns = build_same_stack_reference()
    actor = ns["RecurrentActor"](
        ns["MLPTorso"]((128,)),
        ns["MLPTorso"]((128,)),
        ns["DiscreteActionHead"](198),
    )
    critic = ns["RecurrentValueNet"](
        ns["MLPTorso"]((128,)), ns["MLPTorso"]((128,)), centralised_critic=True
    )
    ns["actor_apply_fn"], ns["critic_apply_fn"] = actor.apply, critic.apply
    ns["config"] = SimpleNamespace(
        system=SimpleNamespace(**metadata["settings"]["system"]),
        arch=SimpleNamespace(**metadata["settings"]["arch"]),
    )
    actor_params = jax.tree.map(_asarray, reference_tree(arrays, "parameters/actor"))
    critic_params = jax.tree.map(_asarray, reference_tree(arrays, "parameters/critic"))
    actor_optim, critic_optim = ppo._optimizers(_CONFIG)
    ns["actor_update_fn"], ns["critic_update_fn"] = (
        actor_optim.update,
        critic_optim.update,
    )
    state = ppo.PPOTrainState(
        actor_params,
        critic_params,
        actor_optim.init(actor_params),
        critic_optim.init(critic_params),
    )
    present = jnp.ones_like(inputs["episode_start"])
    batch = ppo.PPOMinibatch(
        inputs["actor_features"],
        inputs["critic_features"],
        inputs["action_mask"],
        inputs["actions"],
        inputs["old_log_probabilities"],
        inputs["old_values"],
        jnp.asarray(arrays["expected/advantages"]),
        jnp.asarray(arrays["expected/targets"]),
        inputs["episode_start"],
        present,
        present,
        present,
        inputs["actor_carry"],
        inputs["critic_carry"],
    )
    shape = (*present.shape, 128)
    trajectory = ns["RNNPPOTransition"](
        inputs["episode_start"],
        inputs["actions"],
        inputs["old_values"],
        inputs["rewards"],
        inputs["old_log_probabilities"],
        ns["ObservationGlobalState"](
            inputs["actor_features"], inputs["action_mask"], inputs["critic_features"]
        ),
        ns["HiddenStates"](
            jnp.broadcast_to(inputs["actor_carry"][:, None], shape),
            jnp.broadcast_to(inputs["critic_carry"][:, None], shape),
        ),
    )
    return Reference(
        metadata, arrays, inputs, ns, actor, critic, trajectory, state, batch
    )


def _compare(
    actual: Tree,
    expected: dict[str, NDArray[Any]],
    prefix: str,
    *,
    exact: bool = False,
) -> float:
    leaves: dict[str, NDArray[Any]] = {}
    _store_tree(leaves, prefix, actual, jax)
    expected_keys = {
        key for key in expected if key == prefix or key.startswith(prefix + "/")
    }
    assert set(leaves) == expected_keys, prefix
    maximum = 0.0
    for key, value in leaves.items():
        target = expected[key]
        assert value.shape == target.shape, key
        assert value.dtype == target.dtype, key
        if exact or not np.issubdtype(value.dtype, np.floating):
            np.testing.assert_array_equal(value, target, err_msg=key)
        else:
            np.testing.assert_allclose(
                value, target, atol=_ATOL, rtol=_RTOL, err_msg=key
            )
        difference = np.abs(value.astype(np.float64) - target.astype(np.float64))
        maximum = max(maximum, float(difference.max(initial=0.0)))
    return maximum


def _same_tree(actual: Tree, expected: Tree, *, exact: bool = False) -> float:
    leaves: dict[str, NDArray[Any]] = {}
    _store_tree(leaves, "same_stack", expected, jax)
    return _compare(actual, leaves, "same_stack", exact=exact)


def _select_games(batch: ppo.PPOMinibatch, indices: Array) -> ppo.PPOMinibatch:
    def sequence_rows(value: Array, index: Array) -> Array:
        return jnp.take(value, index, axis=1)

    def memory_rows(value: Array, index: Array) -> Array:
        return jnp.take(value, index, axis=0)

    sequence = tuple(jax.vmap(sequence_rows)(leaf, indices) for leaf in batch[:12])
    memory = tuple(jax.vmap(memory_rows)(leaf, indices) for leaf in batch[12:])
    return ppo.PPOMinibatch(*sequence, *memory)


def test_defaults_match_resolved_donor_settings(reference: Reference) -> None:
    settings = reference.metadata["settings"]["system"]
    assert _CONFIG.actor_lr == settings["actor_lr"]
    assert _CONFIG.critic_lr == settings["critic_lr"]
    assert _CONFIG.epochs == settings["ppo_epochs"] == 4
    assert _CONFIG.minibatches == settings["num_minibatches"] == 2
    assert _CONFIG.groups == settings["update_batch_size"] == 2
    assert _CONFIG.rollout_length == settings["recurrent_chunk_size"] == 4
    assert ppo.DEFAULT_PPO_CONFIG.rollout_length == 128
    assert _CONFIG.gamma == settings["gamma"]
    assert _CONFIG.gae_lambda == settings["gae_lambda"]
    assert _CONFIG.clip_epsilon == settings["clip_eps"]
    assert _CONFIG.entropy_coefficient == settings["ent_coef"]
    assert _CONFIG.value_coefficient == settings["vf_coef"]
    assert _CONFIG.max_grad_norm == settings["max_grad_norm"]
    assert _CONFIG.adam_epsilon == 1e-5
    assert settings["decay_learning_rates"] is False


def test_initialization_matches_same_stack_source(reference: Reference) -> None:
    obs = jax.tree.map(_first_group, reference.trajectory.obs)
    starts = reference.inputs["episode_start"][0]
    valid = jnp.ones_like(starts)
    for name, key, donor, production in (
        ("actor", 19, reference.actor, ppo.RecurrentActor()),
        ("critic", 23, reference.critic, ppo.RecurrentValueNet()),
    ):
        memory = reference.inputs[f"{name}_carry"][0]
        features = reference.inputs[f"{name}_features"][0]
        actual = production.init(
            jax.random.PRNGKey(key), memory, features, starts, valid
        )
        expected = donor.init(jax.random.PRNGKey(key), memory, (obs, starts))
        _same_tree(actual, expected, exact=True)
        # Historical initialization may differ numerically between JAX versions,
        # but parameter ownership, paths, dimensions and types must not drift.
        actual_leaves: dict[str, NDArray[Any]] = {}
        _store_tree(actual_leaves, f"parameters/{name}", actual, jax)
        expected_keys = {
            key for key in reference.arrays if key.startswith(f"parameters/{name}/")
        }
        assert set(actual_leaves) == expected_keys
        for path, value in actual_leaves.items():
            assert value.shape == reference.arrays[path].shape
            assert value.dtype == reference.arrays[path].dtype


def test_forward_outputs_match_historical_and_live_source(
    reference: Reference, record_property: RecordProperty
) -> None:
    batch, state = reference.batch, reference.state
    actor_memory, logits = jax.vmap(
        cast(Callable[..., tuple[Array, Array]], ppo.RecurrentActor().apply),
        in_axes=(None, 0, 0, 0, 0),
    )(
        state.actor_params,
        batch.actor_memory,
        batch.actor_features,
        batch.episode_start,
        batch.valid,
    )
    critic_memory, values = jax.vmap(
        cast(Callable[..., tuple[Array, Array]], ppo.RecurrentValueNet().apply),
        in_axes=(None, 0, 0, 0, 0),
    )(
        state.critic_params,
        batch.critic_memory,
        batch.critic_features,
        batch.episode_start,
        batch.valid,
    )
    outputs = {
        "actor_carry": actor_memory,
        "critic_carry": critic_memory,
        "raw_logits": logits,
        "values": values,
        "log_probabilities": action_log_prob(logits, batch.action_mask, batch.actions),
        "entropy": action_entropy(logits, batch.action_mask),
    }
    errors = [
        _compare(value, reference.arrays, f"expected/{name}")
        for name, value in outputs.items()
    ]

    def source_forward(group: Tree) -> Tree:
        next_actor, distribution = reference.actor.apply(
            state.actor_params,
            group.hstates.policy_hidden_state[0],
            (group.obs, group.done),
        )
        next_critic, value = reference.critic.apply(
            state.critic_params,
            group.hstates.critic_hidden_state[0],
            (group.obs, group.done),
        )
        return (
            next_actor,
            next_critic,
            value,
            distribution.distribution.logits,
            distribution.log_prob(group.action),
            distribution.entropy(),
        )

    source = jax.vmap(source_forward)(reference.trajectory)
    errors.append(_compare(source[3], reference.arrays, "expected/masked_logits"))
    errors.append(
        _same_tree(
            (
                actor_memory,
                critic_memory,
                values,
                outputs["log_probabilities"],
                outputs["entropy"],
            ),
            (source[0], source[1], source[2], source[4], source[5]),
        )
    )
    record_property("maximum_absolute_error", max(errors))


def test_gae_uses_the_same_transition_endings_as_donor(
    reference: Reference, record_property: RecordProperty
) -> None:
    inputs, ns = reference.inputs, reference.namespace
    ended = jnp.concatenate(
        (inputs["episode_start"][:, 1:], inputs["final_done"][:, None]), axis=1
    )
    actual = jax.vmap(ppo.calculate_gae)(
        inputs["rewards"], inputs["old_values"], ended, inputs["final_values"]
    )
    expected = jax.vmap(ns["calculate_gae"], in_axes=(0, 0, 0, None, None))(
        reference.trajectory,
        inputs["final_values"],
        inputs["final_done"],
        _CONFIG.gamma,
        _CONFIG.gae_lambda,
    )
    errors = [_same_tree(actual, expected)]
    for value, name in zip(actual, ("advantages", "targets"), strict=True):
        errors.append(_compare(value, reference.arrays, f"expected/{name}"))
    record_property("maximum_absolute_error", max(errors))


def test_minibatch_losses_and_gradients_match_both_references(
    reference: Reference, record_property: RecordProperty
) -> None:
    first = _select_games(reference.batch, jnp.asarray([[0, 1], [0, 1]]))
    state, ns = reference.state, reference.namespace
    (actor_loss, entropy), actor_grads = jax.vmap(
        jax.value_and_grad(ppo._actor_loss, has_aux=True), in_axes=(None, 0, None)
    )(state.actor_params, first, _CONFIG)

    def critic_loss(params: Tree, group: ppo.PPOMinibatch) -> tuple[Array, Array]:
        loss = ppo._critic_loss(params, group, _CONFIG)
        return _CONFIG.value_coefficient * loss, loss

    critic_losses, critic_grads = jax.vmap(
        jax.value_and_grad(critic_loss, has_aux=True), in_axes=(None, 0)
    )(state.critic_params, first)
    actor_losses = (
        actor_loss,
        (actor_loss + _CONFIG.entropy_coefficient * entropy, entropy),
    )
    outputs = {
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "actor_group_gradients": actor_grads,
        "critic_group_gradients": critic_grads,
        "actor_gradients": jax.tree.map(_group_mean, actor_grads),
        "critic_gradients": jax.tree.map(_group_mean, critic_grads),
    }
    errors = [
        _compare(value, reference.arrays, f"first_minibatch/{name}")
        for name, value in outputs.items()
    ]
    trajectory = jax.tree.map(_first_half, reference.trajectory)
    source_actor = jax.vmap(
        jax.value_and_grad(ns["_actor_loss_fn"], has_aux=True), in_axes=(None, 0, 0, 0)
    )(state.actor_params, trajectory, first.advantages, reference.inputs["update_keys"])
    source_critic = jax.vmap(
        jax.value_and_grad(ns["_critic_loss_fn"], has_aux=True), in_axes=(None, 0, 0)
    )(state.critic_params, trajectory, first.targets)
    errors.append(_same_tree((actor_losses, actor_grads), source_actor))
    errors.append(_same_tree((critic_losses, critic_grads), source_critic))
    record_property("maximum_absolute_error", max(errors))


def test_first_optimizer_update_matches_historical_reference(
    reference: Reference, record_property: RecordProperty
) -> None:
    batch = _select_games(reference.batch, jnp.asarray([[0, 1], [0, 1]]))
    update = jax.jit(ppo.update_minibatch, static_argnames="config")
    state, metrics = cast(
        tuple[ppo.PPOTrainState, ppo.PPOMetrics],
        update(reference.state, batch, config=_CONFIG),
    )
    outputs = {
        "actor_parameters": state.actor_params,
        "critic_parameters": state.critic_params,
        "actor_optimizer": state.actor_opt_state,
        "critic_optimizer": state.critic_opt_state,
    }
    errors = [
        _compare(value, reference.arrays, f"first_minibatch/{name}")
        for name, value in outputs.items()
    ]
    expected_metrics = tuple(
        jnp.asarray(reference.arrays[name]).mean(0)
        for name in (
            "first_minibatch/actor_losses/0",
            "first_minibatch/actor_losses/1/1",
            "first_minibatch/critic_losses/1",
        )
    )
    errors.append(_same_tree(metrics[:3], expected_metrics))
    assert int(metrics.actor_samples) == int(metrics.critic_samples) == 80
    record_property("maximum_absolute_error", max(errors))


def test_four_epochs_match_grouped_donor_updates(
    reference: Reference, record_property: RecordProperty
) -> None:
    permutations = reference.inputs["epoch_permutations"]
    selections = jnp.transpose(permutations.reshape(4, 2, 2, 2), (0, 2, 1, 3))

    def epoch(
        state: ppo.PPOTrainState, minibatches: Array
    ) -> tuple[ppo.PPOTrainState, ppo.PPOMetrics]:
        def step(
            state: ppo.PPOTrainState, indices: Array
        ) -> tuple[ppo.PPOTrainState, ppo.PPOMetrics]:
            batch = _select_games(reference.batch, indices)
            return ppo.update_minibatch(state, batch, _CONFIG)

        return jax.lax.scan(step, state, minibatches)

    def four_epochs(
        start: ppo.PPOTrainState,
    ) -> tuple[ppo.PPOTrainState, ppo.PPOMetrics]:
        return jax.lax.scan(epoch, start, selections)

    state, metrics = cast(
        tuple[ppo.PPOTrainState, ppo.PPOMetrics],
        jax.jit(four_epochs)(reference.state),
    )
    ns = reference.namespace
    outputs = {
        "parameters": ns["Params"](state.actor_params, state.critic_params),
        "optimizer": ns["OptStates"](state.actor_opt_state, state.critic_opt_state),
    }
    errors = [
        _compare(value, reference.arrays, f"four_epochs/{name}")
        for name, value in outputs.items()
    ]
    for name in ("actor_loss", "entropy", "value_loss"):
        expected = reference.arrays[f"four_epochs/losses/{name}"][:, 0, 0]
        np.testing.assert_array_equal(
            reference.arrays[f"four_epochs/losses/{name}"][:, 0, 1], expected
        )
        errors.append(_same_tree(getattr(metrics, name), expected))
    np.testing.assert_array_equal(metrics.actor_samples, np.full((4, 2), 80))
    np.testing.assert_array_equal(metrics.critic_samples, np.full((4, 2), 80))
    record_property("maximum_absolute_error", max(errors))
