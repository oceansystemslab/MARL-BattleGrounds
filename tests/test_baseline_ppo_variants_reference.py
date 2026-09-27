"""Compare IPPO and feedforward PPO with independent pinned Mava calculations.

Each method uses distinct actor rows, legal masks, two gradient groups, two
minibatches and four epochs. Recurrent IPPO also has nonzero independent carries
and an interior reset. Feedforward samples retain all actors of each time/game
row. Saved arrays come from unchanged donor bodies on its locked CPU stack;
current-stack donor bodies provide a second reference. BG production functions
never generate either oracle. Normalization is disabled and inputs use the world
frame here; BG padding, frame and ValueNorm adaptations have separate tests.
A compact-versus-repeated critic check also verifies equal masked gradients and
updates while the compact path sends fewer rows through its dense layers.

Float comparisons use the established 2e-6 absolute and relative tolerances;
integer state, parameter paths, shapes and dtypes are exact. These checks prove
numerical agreement, not learning performance, GPU cost or new information rights.
"""

# Runtime donor types cannot be imported into the repository's type environment.
# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
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
    load_variants_reference,
    reference_optimizers,
    reference_tree,
)
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.baselines import ppo
from marl_battlegrounds.baselines.actions import (
    action_entropy,
    action_log_prob,
    categorical_action_mask,
    decode_actions,
)
from marl_battlegrounds.baselines.inputs import (
    encode_actor_inputs,
    encode_training_state,
)
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.environment import make
from marl_battlegrounds.policies.input import Observations

type Tree = Any
type RecordProperty = Callable[[str, object], None]
_CONFIG = ppo.PPOConfig(
    rollout_length=4, value_normalization=False, spawn_frame="world"
)
_METHODS = ("ippo", "ff_mappo", "ff_ippo")


def _compare(actual: Tree, expected: Tree, *, exact: bool = False) -> float:
    actual_leaves: dict[str, NDArray[Any]] = {}
    expected_leaves: dict[str, NDArray[Any]] = {}
    _store_tree(actual_leaves, "value", actual, jax)
    _store_tree(expected_leaves, "value", expected, jax)
    assert actual_leaves.keys() == expected_leaves.keys()
    maximum = 0.0
    for name, value in actual_leaves.items():
        target = expected_leaves[name]
        assert value.shape == target.shape, name
        assert value.dtype == target.dtype, name
        if exact or not np.issubdtype(value.dtype, np.floating):
            np.testing.assert_array_equal(value, target, err_msg=name)
        else:
            np.testing.assert_allclose(
                value, target, atol=2e-6, rtol=2e-6, err_msg=name
            )
        maximum = max(
            maximum,
            float(
                np.abs(value.astype(np.float64) - target.astype(np.float64)).max(
                    initial=0
                )
            ),
        )
    return maximum


def _stored(actual: Tree, arrays: dict[str, NDArray[Any]], prefix: str) -> float:
    actual_leaves: dict[str, NDArray[Any]] = {}
    _store_tree(actual_leaves, prefix, actual, jax)
    selected = {
        key: value
        for key, value in arrays.items()
        if key == prefix or key.startswith(prefix + "/")
    }
    return _compare(actual_leaves, selected)


def _flatten(value: Array) -> Array:
    return value.reshape(
        value.shape[0], value.shape[1] * value.shape[2], *value.shape[3:]
    )


def _asarray(value: NDArray[Any]) -> Array:
    return jnp.asarray(value)


def _first_group(value: Array) -> Array:
    return value[0]


def _group_mean(value: Array) -> Array:
    return value.mean(0)


@dataclass
class Reference:
    method: str
    metadata: dict[str, Any]
    arrays: dict[str, NDArray[Any]]
    inputs: dict[str, Array]
    namespace: dict[str, Any]
    actor: Any
    critic: Any
    trajectory: Any
    state: ppo.PPOTrainState
    batch: ppo.PPOMinibatch

    @property
    def recurrent(self) -> bool:
        return self.method == "ippo"


@pytest.fixture(scope="module", params=_METHODS)
def reference(request: pytest.FixtureRequest) -> Reference:
    method = str(request.param)
    metadata, arrays = load_variants_reference(method)
    inputs = {
        key.removeprefix("input/"): jnp.asarray(value)
        for key, value in arrays.items()
        if key.startswith("input/")
    }
    ns = build_same_stack_reference(method)
    ns["config"] = SimpleNamespace(
        system=SimpleNamespace(**metadata["settings"]["system"]),
        arch=SimpleNamespace(**metadata["settings"]["arch"]),
    )
    torso, head = ns["MLPTorso"], ns["DiscreteActionHead"](198)
    if method == "ippo":
        actor = ns["RecurrentActor"](torso((128,)), torso((128,)), head)
        critic = ns["RecurrentValueNet"](torso((128,)), torso((128,)))
    else:
        actor = ns["FeedForwardActor"](torso((128, 128)), head)
        critic = ns["FeedForwardValueNet"](
            torso((128, 128)), centralised_critic=method == "ff_mappo"
        )
    ns["actor_apply_fn"], ns["critic_apply_fn"] = actor.apply, critic.apply
    actor_opt, critic_opt = reference_optimizers(ns, method)
    actor_params = jax.tree.map(_asarray, reference_tree(arrays, "parameters/actor"))
    critic_params = jax.tree.map(_asarray, reference_tree(arrays, "parameters/critic"))
    state = ppo.PPOTrainState(
        actor_params,
        critic_params,
        actor_opt.init(actor_params),
        critic_opt.init(critic_params),
    )
    present = jnp.ones_like(inputs["episode_start"])
    fields = (
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
    )
    observation = ns["ObservationGlobalState"](
        inputs["actor_features"], inputs["action_mask"], inputs["critic_features"]
    )
    transition = (
        inputs["episode_start"],
        inputs["actions"],
        inputs["old_values"],
        inputs["rewards"],
        inputs["old_log_probabilities"],
        observation,
    )
    if method == "ippo":
        batch = ppo.PPOMinibatch(*fields, inputs["actor_carry"], inputs["critic_carry"])
        memory_shape = (*present.shape, 128)
        trajectory = ns["RNNPPOTransition"](
            *transition,
            ns["HiddenStates"](
                jnp.broadcast_to(inputs["actor_carry"][:, None], memory_shape),
                jnp.broadcast_to(inputs["critic_carry"][:, None], memory_shape),
            ),
        )
    else:
        batch = jax.tree.map(_flatten, ppo.PPOMinibatch(*fields, (), ()))
        trajectory = ns["PPOTransition"](*transition)
    return Reference(
        method, metadata, arrays, inputs, ns, actor, critic, trajectory, state, batch
    )


def _select(reference: Reference, indices: Array) -> ppo.PPOMinibatch:
    axis = 1 if reference.recurrent else 0

    def rows(value: Array, index: Array) -> Array:
        return jnp.take(value, index, axis=axis)

    def memory_rows(value: Array, index: Array) -> Array:
        return value[index]

    sequence = tuple(jax.vmap(rows)(leaf, indices) for leaf in reference.batch[:12])
    memory = (
        tuple(jax.vmap(memory_rows)(leaf, indices) for leaf in reference.batch[12:14])
        if reference.recurrent
        else ((), ())
    )
    return ppo.PPOMinibatch(*sequence, *memory)


def _first(reference: Reference) -> ppo.PPOMinibatch:
    size = 2 if reference.recurrent else 8
    return _select(reference, jnp.broadcast_to(jnp.arange(size), (2, size)))


def test_method_defaults_match_the_actual_pinned_config(reference: Reference) -> None:
    settings = reference.metadata["settings"]["system"]
    for field, name in (
        ("actor_lr", "actor_lr"),
        ("critic_lr", "critic_lr"),
        ("epochs", "ppo_epochs"),
        ("minibatches", "num_minibatches"),
        ("groups", "update_batch_size"),
        ("gamma", "gamma"),
        ("gae_lambda", "gae_lambda"),
        ("clip_epsilon", "clip_eps"),
        ("entropy_coefficient", "ent_coef"),
        ("value_coefficient", "vf_coef"),
        ("max_grad_norm", "max_grad_norm"),
    ):
        assert getattr(_CONFIG, field) == settings[name]
    assert _CONFIG.adam_epsilon == 1e-5
    assert settings["decay_learning_rates"] is False
    assert ppo.DEFAULT_PPO_CONFIG.rollout_length == 128
    if reference.recurrent:
        assert settings["recurrent_chunk_size"] == _CONFIG.rollout_length
    if reference.method != "ff_mappo":
        np.testing.assert_array_equal(
            reference.inputs["critic_features"], reference.inputs["actor_features"]
        )


def test_initialization_matches_current_stack_donor_exactly(
    reference: Reference,
) -> None:
    obs = jax.tree.map(_first_group, reference.trajectory.obs)
    starts = reference.inputs["episode_start"][0]
    valid = jnp.ones_like(starts)
    for name, seed, donor in (
        ("actor", 19, reference.actor),
        ("critic", 23, reference.critic),
    ):
        features = reference.inputs[f"{name}_features"][0]
        if reference.recurrent:
            model = ppo.RecurrentActor() if name == "actor" else ppo.RecurrentValueNet()
            memory = reference.inputs[f"{name}_carry"][0]
            actual = model.init(
                jax.random.PRNGKey(seed), memory, features, starts, valid
            )
            expected = donor.init(jax.random.PRNGKey(seed), memory, (obs, starts))
        else:
            model = (
                ppo.FeedForwardActor() if name == "actor" else ppo.FeedForwardValueNet()
            )
            actual = model.init(jax.random.PRNGKey(seed), features)
            expected = donor.init(jax.random.PRNGKey(seed), obs)
        _compare(actual, expected, exact=True)
        historical = reference_tree(reference.arrays, f"parameters/{name}")
        assert jax.tree.structure(actual) == jax.tree.structure(historical)
        for left, right in zip(
            jax.tree.leaves(actual), jax.tree.leaves(historical), strict=True
        ):
            assert left.shape == right.shape and left.dtype == right.dtype


def test_forward_matches_historical_and_current_stack_donor(
    reference: Reference, record_property: RecordProperty
) -> None:
    inputs, state = reference.inputs, reference.state
    if reference.recurrent:
        memory_a, logits = jax.vmap(
            cast(Callable[..., tuple[Array, Array]], ppo.RecurrentActor().apply),
            in_axes=(None, 0, 0, 0, 0),
        )(
            state.actor_params,
            inputs["actor_carry"],
            inputs["actor_features"],
            inputs["episode_start"],
            jnp.ones_like(inputs["episode_start"]),
        )
        memory_c, values = jax.vmap(
            cast(Callable[..., tuple[Array, Array]], ppo.RecurrentValueNet().apply),
            in_axes=(None, 0, 0, 0, 0),
        )(
            state.critic_params,
            inputs["critic_carry"],
            inputs["critic_features"],
            inputs["episode_start"],
            jnp.ones_like(inputs["episode_start"]),
        )
    else:
        logits = cast(
            Array,
            ppo.FeedForwardActor().apply(state.actor_params, inputs["actor_features"]),
        )
        values = cast(
            Array,
            ppo.FeedForwardValueNet().apply(
                state.critic_params, inputs["critic_features"]
            ),
        )
        memory_a, memory_c = (), ()
    outputs: dict[str, Tree] = {
        "raw_logits": logits,
        "values": values,
        "log_probabilities": action_log_prob(
            logits, inputs["action_mask"], inputs["actions"]
        ),
        "entropy": action_entropy(logits, inputs["action_mask"]),
    }
    if reference.recurrent:
        outputs.update(actor_carry=memory_a, critic_carry=memory_c)
    errors = [
        _stored(value, reference.arrays, f"expected/{name}")
        for name, value in outputs.items()
    ]

    def donor_forward(row: Tree) -> Tree:
        if reference.recurrent:
            next_a, distribution = reference.actor.apply(
                state.actor_params,
                row.hstates.policy_hidden_state[0],
                (row.obs, row.done),
            )
            next_c, value = reference.critic.apply(
                state.critic_params,
                row.hstates.critic_hidden_state[0],
                (row.obs, row.done),
            )
        else:
            distribution = reference.actor.apply(state.actor_params, row.obs)
            value = reference.critic.apply(state.critic_params, row.obs)
            next_a, next_c = (), ()
        return (
            next_a,
            next_c,
            value,
            distribution.log_prob(row.action),
            distribution.entropy(),
        )

    source = jax.vmap(donor_forward)(reference.trajectory)
    errors.append(
        _compare(
            (
                memory_a,
                memory_c,
                values,
                outputs["log_probabilities"],
                outputs["entropy"],
            ),
            source,
        )
    )
    record_property("maximum_absolute_error", max(errors))


def test_gae_matches_original_transition_endings(
    reference: Reference, record_property: RecordProperty
) -> None:
    inputs, ns = reference.inputs, reference.namespace
    ended = jnp.concatenate(
        (inputs["episode_start"][:, 1:], inputs["final_done"][:, None]), axis=1
    )
    actual = jax.vmap(ppo.calculate_gae)(
        inputs["rewards"], inputs["old_values"], ended, inputs["final_values"]
    )
    source = jax.vmap(ns["calculate_gae"], in_axes=(0, 0, 0, None, None))(
        reference.trajectory,
        inputs["final_values"],
        inputs["final_done"],
        _CONFIG.gamma,
        _CONFIG.gae_lambda,
    )
    errors = [_compare(actual, source)]
    for value, name in zip(actual, ("advantages", "targets"), strict=True):
        errors.append(_stored(value, reference.arrays, f"expected/{name}"))
    record_property("maximum_absolute_error", max(errors))


def test_losses_and_gradients_match_both_donor_references(
    reference: Reference, record_property: RecordProperty
) -> None:
    batch, state, ns = _first(reference), reference.state, reference.namespace
    actor_fn = partial(ppo._actor_loss, method=reference.method)
    (actor_loss, entropy), actor_grads = jax.vmap(
        jax.value_and_grad(actor_fn, has_aux=True), in_axes=(None, 0, None)
    )(state.actor_params, batch, _CONFIG)

    def critic_fn(params: Tree, group: ppo.PPOMinibatch) -> tuple[Array, Array]:
        value = ppo._critic_loss(params, group, _CONFIG, method=reference.method)
        return _CONFIG.value_coefficient * value, value

    critic_losses, critic_grads = jax.vmap(
        jax.value_and_grad(critic_fn, has_aux=True), in_axes=(None, 0)
    )(state.critic_params, batch)
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
        _stored(value, reference.arrays, f"first_minibatch/{name}")
        for name, value in outputs.items()
    ]

    def first_rows(value: Array) -> Array:
        return value[:, :, :2] if reference.recurrent else _flatten(value)[:, :8]

    trajectory = jax.tree.map(first_rows, reference.trajectory)
    source_actor = jax.vmap(
        jax.value_and_grad(ns["_actor_loss_fn"], has_aux=True), in_axes=(None, 0, 0, 0)
    )(state.actor_params, trajectory, batch.advantages, reference.inputs["update_keys"])
    source_critic = jax.vmap(
        jax.value_and_grad(ns["_critic_loss_fn"], has_aux=True), in_axes=(None, 0, 0)
    )(state.critic_params, trajectory, batch.targets)
    errors.extend(
        (
            _compare((actor_losses, actor_grads), source_actor),
            _compare((critic_losses, critic_grads), source_critic),
        )
    )
    record_property("maximum_absolute_error", max(errors))


def test_first_optimizer_step_matches_saved_donor(
    reference: Reference, record_property: RecordProperty
) -> None:
    update = jax.jit(
        partial(ppo.update_minibatch, config=_CONFIG, method=reference.method)
    )
    state, metrics = cast(
        tuple[ppo.PPOTrainState, ppo.PPOMetrics],
        update(reference.state, _first(reference)),
    )
    outputs = {
        "actor_parameters": state.actor_params,
        "critic_parameters": state.critic_params,
        "actor_optimizer": state.actor_opt_state,
        "critic_optimizer": state.critic_opt_state,
    }
    errors = [
        _stored(value, reference.arrays, f"first_minibatch/{name}")
        for name, value in outputs.items()
    ]
    expected_metrics = tuple(
        jnp.asarray(reference.arrays[f"first_minibatch/{name}"]).mean(0)
        for name in ("actor_losses/0", "actor_losses/1/1", "critic_losses/1")
    )
    errors.append(_compare(metrics[:3], expected_metrics))
    assert int(metrics.actor_samples) == int(metrics.critic_samples) == 80
    record_property("maximum_absolute_error", max(errors))


def test_four_epochs_match_original_grouped_updates(
    reference: Reference, record_property: RecordProperty
) -> None:
    width = 2 if reference.recurrent else 8
    selections = jnp.transpose(
        reference.inputs["epoch_permutations"].reshape(4, 2, 2, width), (0, 2, 1, 3)
    )

    def epoch(
        state: ppo.PPOTrainState, selected: Array
    ) -> tuple[ppo.PPOTrainState, ppo.PPOMetrics]:
        def minibatch(
            state: ppo.PPOTrainState, indices: Array
        ) -> tuple[ppo.PPOTrainState, ppo.PPOMetrics]:
            return ppo.update_minibatch(
                state, _select(reference, indices), _CONFIG, method=reference.method
            )

        return jax.lax.scan(minibatch, state, selected)

    def run(initial: ppo.PPOTrainState) -> tuple[ppo.PPOTrainState, ppo.PPOMetrics]:
        return jax.lax.scan(epoch, initial, selections)

    state, metrics = cast(
        tuple[ppo.PPOTrainState, ppo.PPOMetrics], jax.jit(run)(reference.state)
    )
    ns = reference.namespace
    outputs = {
        "parameters": ns["Params"](state.actor_params, state.critic_params),
        "optimizer": ns["OptStates"](state.actor_opt_state, state.critic_opt_state),
    }
    errors = [
        _stored(value, reference.arrays, f"four_epochs/{name}")
        for name, value in outputs.items()
    ]
    for name in ("actor_loss", "entropy", "value_loss"):
        expected = reference.arrays[f"four_epochs/losses/{name}"][:, 0, 0]
        np.testing.assert_array_equal(
            reference.arrays[f"four_epochs/losses/{name}"][:, 0, 1], expected
        )
        errors.append(_compare(getattr(metrics, name), expected))
    np.testing.assert_array_equal(metrics.actor_samples, np.full((4, 2), 80))
    np.testing.assert_array_equal(metrics.critic_samples, np.full((4, 2), 80))
    record_property("maximum_absolute_error", max(errors))


@dataclass
class CompactTrajectory:
    observations: Observations
    physical: Array
    actor_features: Array
    masks: ActionMask
    starts: Array
    actions: Array


@pytest.fixture(scope="module")
def compact_trajectory() -> CompactTrajectory:
    env = make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(5, 5)),
        num_envs=8,
        metrics="none",
    )
    _, state = env.reset(jax.random.key(102))
    observations, state, *_ = env.step(
        jax.random.key(110), state, env.sample_actions(jax.random.key(111), state)
    )
    snapshots: list[Observations] = []
    physical_rows: list[Array] = []
    actor_rows: list[Array] = []
    mask_rows: list[ActionMask] = []
    start_rows: list[Array] = []
    action_rows: list[Array] = []
    for timestep in range(4):
        if timestep == 2:
            observations, state = env.reset(
                jax.random.key(112), state=state, reset_mask=jnp.arange(8) == 0
            )
        inputs = env.policy_inputs(observations, state)
        legal = np.asarray(categorical_action_mask(inputs.action_mask))
        choices = np.empty((8, 5), np.int32)
        for game, actor_row in np.ndindex(8, 5):
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
        zero = jnp.zeros((8, 5), jnp.int32)
        native = env.join_actions(
            decode_actions(jnp.asarray(choices)), decode_actions(zero)
        )
        observations, state, *_ = env.step(
            jax.random.key(120 + timestep), state, native
        )

    def stack(*values: Array) -> Array:
        return jnp.stack(values)

    result = CompactTrajectory(
        jax.tree.map(stack, *snapshots),
        jnp.stack(physical_rows),
        jnp.stack(actor_rows),
        jax.tree.map(stack, *mask_rows),
        jnp.stack(start_rows),
        jnp.stack(action_rows),
    )
    # The independent source must see distinct rows to detect incorrect selection.
    for features in (result.actor_features, result.physical):
        assert not np.array_equal(features[0, 0], features[0, 1])
        assert not np.array_equal(features[0], features[1])
    assert np.unique(result.actions).size > 1
    return result


def test_compact_rollout_matches_original_grouped_epochs(
    reference: Reference,
    compact_trajectory: CompactTrajectory,
    record_property: RecordProperty,
) -> None:
    data, ns = compact_trajectory, reference.namespace
    initial = ppo.initialize_ppo(jax.random.key(101), _CONFIG, method=reference.method)
    shape = (4, 8, 5)
    rng = np.random.default_rng(731)
    actor_memory = jnp.asarray(rng.normal(0, 0.1, (8, 5, 128)), jnp.float32)
    critic_memory = jnp.asarray(rng.normal(0, 0.1, (8, 5, 128)), jnp.float32)
    rewards = jnp.asarray(rng.normal(0, 0.2, shape), jnp.float32)
    final_values = jnp.asarray(rng.normal(0, 0.1, (8, 5)), jnp.float32)
    physical = data.physical if reference.method == "ff_mappo" else None
    critic_features = (
        data.actor_features
        if physical is None
        else jnp.broadcast_to(physical[..., None, :], (*shape, physical.shape[-1]))
    )
    observation = ns["ObservationGlobalState"](
        data.actor_features, categorical_action_mask(data.masks), critic_features
    )
    starts = jnp.broadcast_to(data.starts[..., None], shape)
    if reference.recurrent:
        _, distribution = reference.actor.apply(
            initial.actor_params, actor_memory, (observation, starts)
        )
        _, values = reference.critic.apply(
            initial.critic_params, critic_memory, (observation, starts)
        )
    else:
        distribution = reference.actor.apply(initial.actor_params, observation)
        values = reference.critic.apply(initial.critic_params, observation)
    old_logs = distribution.log_prob(data.actions) + jnp.asarray(
        rng.normal(0, 0.25, shape), jnp.float32
    )
    old_values = values + jnp.asarray(rng.normal(0, 0.3, shape), jnp.float32)
    batch = ppo.PPOBatch(
        data.observations,
        physical,
        data.masks,
        data.actions,
        old_logs,
        old_values,
        rewards,
        jnp.zeros_like(data.starts).at[1, 0].set(True),
        data.starts,
        jnp.ones_like(data.starts),
        jnp.ones(shape, bool),
        jnp.ones(shape, bool),
        final_values,
        actor_memory if reference.recurrent else (),
        critic_memory if reference.recurrent else (),
    )
    transition = (starts, data.actions, old_values, rewards, old_logs, observation)
    if reference.recurrent:
        memory = ns["HiddenStates"](
            jnp.broadcast_to(actor_memory, (4, *actor_memory.shape)),
            jnp.broadcast_to(critic_memory, (4, *critic_memory.shape)),
        )
        trajectory = ns["RNNPPOTransition"](*transition, memory)
    else:
        trajectory = ns["PPOTransition"](*transition)

    def grouped(value: Array) -> Array:
        return value.reshape(4, 2, 4, *value.shape[2:]).swapaxes(0, 1)

    trajectory = jax.tree.map(grouped, trajectory)
    bootstrap = final_values.reshape(2, 4, 5)
    advantages, targets = jax.vmap(ns["calculate_gae"], in_axes=(0, 0, 0, None, None))(
        trajectory,
        bootstrap,
        jnp.zeros_like(bootstrap, bool),
        _CONFIG.gamma,
        _CONFIG.gae_lambda,
    )

    def copies(value: Array) -> Array:
        return jnp.broadcast_to(value, (1, 2, *value.shape))

    def add_device(value: Array) -> Array:
        return value[None]

    def remove_device_group(value: Array) -> Array:
        return value[0, 0]

    key = jax.random.key(103)
    donor_state = (
        jax.tree.map(copies, ns["Params"](initial.actor_params, initial.critic_params)),
        jax.tree.map(
            copies, ns["OptStates"](initial.actor_opt_state, initial.critic_opt_state)
        ),
        jax.tree.map(add_device, trajectory),
        advantages[None],
        targets[None],
        jax.random.split(key, 2)[None],
    )
    donor_epoch = jax.vmap(
        jax.vmap(ns["_update_epoch"], in_axes=(0, None), axis_name="batch"),
        in_axes=(0, None),
        axis_name="device",
    )
    expected, losses = jax.lax.scan(donor_epoch, donor_state, None, length=4)
    update = jax.jit(partial(ppo.update_ppo, config=_CONFIG, method=reference.method))
    actual, metrics = cast(
        tuple[ppo.PPOTrainState, ppo.PPOMetrics], update(initial, batch, key)
    )
    expected_params = jax.tree.map(remove_device_group, expected[0])
    expected_opt = jax.tree.map(remove_device_group, expected[1])
    errors = [
        _compare(actual.actor_params, expected_params.actor_params),
        _compare(actual.critic_params, expected_params.critic_params),
        _compare(actual.actor_opt_state, expected_opt.actor_opt_state),
        _compare(actual.critic_opt_state, expected_opt.critic_opt_state),
    ]
    for name in ("actor_loss", "entropy", "value_loss"):
        errors.append(_compare(getattr(metrics, name), losses[name][:, 0, 0]))
    np.testing.assert_array_equal(metrics.actor_samples, np.full((4, 2), 80))
    np.testing.assert_array_equal(metrics.critic_samples, np.full((4, 2), 80))
    record_property("maximum_absolute_error", max(errors))


@pytest.mark.parametrize("reference", ("ff_mappo",), indirect=True)
def test_compact_physical_critic_matches_repeated_rows_and_skips_duplicate_dots(
    reference: Reference, record_property: RecordProperty
) -> None:
    batch = _first(reference)
    physical = batch.critic_features[:, :, 0]
    samples = batch.critic_samples.at[0, :, 1].set(False).at[1, 1::2, 3:].set(False)
    compact = batch._replace(critic_features=physical, critic_samples=samples)
    repeated = compact._replace(
        critic_features=jnp.broadcast_to(
            physical[:, :, None], batch.critic_features.shape
        )
    )

    def loss(params: Tree, group: ppo.PPOMinibatch) -> Array:
        return ppo._critic_loss(params, group, _CONFIG, method="ff_mappo")

    gradient = jax.vmap(jax.value_and_grad(loss), in_axes=(None, 0))
    compact_loss, compact_gradient = gradient(reference.state.critic_params, compact)
    repeated_loss, repeated_gradient = gradient(reference.state.critic_params, repeated)
    errors = [
        _compare(compact_loss, repeated_loss),
        _compare(compact_gradient, repeated_gradient),
    ]
    update = jax.jit(partial(ppo.update_minibatch, config=_CONFIG, method="ff_mappo"))
    actual = cast(
        tuple[ppo.PPOTrainState, ppo.PPOMetrics], update(reference.state, compact)
    )
    expected = cast(
        tuple[ppo.PPOTrainState, ppo.PPOMetrics], update(reference.state, repeated)
    )
    errors.append(_compare(actual, expected))
    assert int(actual[1].critic_samples) == int(samples.sum())

    group = jax.tree.map(_first_group, compact)
    traced = jax.make_jaxpr(loss)(reference.state.critic_params, group)
    dot_inputs = [
        tuple(equation.invars[0].aval.shape)
        for equation in traced.jaxpr.eqns
        if equation.primitive.name == "dot_general"
    ]
    # The three dense layers see eight game rows, not forty copied actor rows.
    rows, width = physical.shape[1:]
    assert dot_inputs == [(rows, width), (rows, 128), (rows, 128)]
    record_property("maximum_absolute_error", max(errors))
