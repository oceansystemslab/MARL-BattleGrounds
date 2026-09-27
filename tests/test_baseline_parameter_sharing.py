"""Check grouped actor models against independent original-network applications.

Small encoded inputs prove routing, physical actor memory, reset-time class
changes, unused groups, differentiated contractions and PQN's eligible-only
normalization. Existing donor and registration tests protect the shared path.
CPU ragged-dot fallback proves correctness, not the GPU implementation's cost.
"""

# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
import dataclasses
import functools
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.test_baseline_inputs import _two_lane_reset

from marl_battlegrounds.baselines import ppo, pqn, qmix
from marl_battlegrounds.baselines.actions import (
    categorical_action_mask,
    encode_actions,
    sample_actions,
)
from marl_battlegrounds.core.types import AGENT_FEATURE_CLASS_ID
from marl_battlegrounds.evaluation.policy_execution import SystemOutput, system_inputs
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.policies.input import ActorInput

type Tree = Any


def _select_group(tree: Tree, group: int) -> Tree:
    def select(value: Array) -> Array:
        return value[group]

    return jax.tree.map(select, tree)


@pytest.fixture(scope="module", params=("ppo", "ff_ppo", "qmix", "pqn"))
def model(request: pytest.FixtureRequest) -> tuple[str, Any, Tree, int]:
    family = str(request.param)
    network, width = {
        "ppo": (ppo.RecurrentActor(), ppo.HIDDEN_SIZE),
        "ff_ppo": (ppo.FeedForwardActor(), 0),
        "qmix": (qmix.RecurrentQNetwork(), qmix.QMIX_HIDDEN_SIZE),
        "pqn": (pqn.PQNNetwork(), pqn.PQN_HIDDEN_SIZE),
    }[family]
    x = jnp.zeros((1, 1, 5, 4), jnp.float32)
    memory = jnp.zeros((1, 5, width), jnp.float32)
    start = jnp.zeros((1, 1, 5), jnp.bool_)
    valid = jnp.ones_like(start)

    def initialize(key: Array) -> Tree:
        if family == "ff_ppo":
            return network.init(key, x)
        if family == "pqn":
            return network.init(key, memory, x, start[..., 0], valid[..., 0], valid)
        return network.init(key, memory, x, start, valid)

    return (
        family,
        network,
        jax.vmap(initialize)(jax.random.split(jax.random.key(75), 5)),
        width,
    )


def test_grouped_models_match_selected_original_networks_and_keep_physical_memory(
    model: tuple[str, Any, Tree, int],
) -> None:
    family, network, variables, width = model
    x = jax.random.normal(jax.random.key(76), (2, 2, 5, 4))
    memory = jax.random.normal(jax.random.key(77), (2, 5, width))
    groups = jnp.asarray(
        [[[0, 0, 1, 2, 4], [2, 1, 1, 0, 0]], [[4, 1, 0, 0, 2], [2, 1, 1, 0, 0]]],
        jnp.int32,
    )
    starts = jnp.zeros(groups.shape, jnp.bool_).at[1, 0].set(True)
    valid = jnp.ones_like(starts).at[1, 1].set(False)
    active = jnp.ones_like(starts).at[:, 0, 4].set(False)
    if family == "ff_ppo":
        actual = cast(
            Array, jax.jit(lambda v: network.apply(v, x, actor_group=groups))(variables)
        )
        expected = jnp.zeros_like(actual)
        for group in range(5):
            selected = _select_group(variables, group)
            expected = jnp.where(
                (groups == group)[..., None], network.apply(selected, x), expected
            )
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
        return

    def apply(
        v: Tree,
        carry: Array,
        inputs: Array,
        reset: Array,
        real: Array,
        present: Array,
        group: Array | None,
    ) -> tuple[Array, Array]:
        if family == "pqn":
            return cast(
                tuple[Array, Array],
                network.apply(
                    v,
                    carry,
                    inputs,
                    reset[..., 0],
                    real[..., 0],
                    present,
                    actor_group=group,
                ),
            )
        return cast(
            tuple[Array, Array],
            network.apply(v, carry, inputs, reset, real, actor_group=group),
        )

    actual_memory, actual = cast(
        tuple[Array, Array],
        jax.jit(apply)(variables, memory, x, starts, valid, active, groups),
    )
    expected_memory = memory
    values: list[Array] = []
    for time in range(2):
        next_memory = jnp.zeros_like(memory)
        expected = jnp.zeros((2, 5, 198), jnp.float32)
        for group in range(5):
            selected = _select_group(variables, group)
            carry, output = apply(
                selected,
                expected_memory,
                x[time : time + 1],
                starts[time : time + 1],
                valid[time : time + 1],
                active[time : time + 1],
                None,
            )
            mask = (groups[time] == group)[..., None]
            next_memory = jnp.where(mask, carry, next_memory)
            expected = jnp.where(mask, output[0], expected)
        expected_memory = next_memory
        values.append(expected)
    np.testing.assert_allclose(actual, jnp.stack(values), atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(actual_memory, expected_memory, atol=3e-6, rtol=3e-5)
    assert not np.allclose(actual_memory[0, 2], actual_memory[0, 3])


def test_grouped_contraction_gradient_matches_selected_matrices() -> None:
    x = jax.random.normal(jax.random.key(80), (7, 3))
    groups = jnp.asarray([4, 0, 0, 1, 4, 1, 0], jnp.int32)
    params = {
        "kernel": jax.random.normal(jax.random.key(81), (5, 3, 2)),
        "bias": jnp.ones((5, 2)),
    }

    def grouped(value: Tree) -> Array:
        order, selected, sizes = ppo._group_order(groups)
        y = ppo._group_dense(x[order], value, selected, sizes)
        return jnp.square(y).sum()

    def reference(value: Tree) -> Array:
        y = jnp.einsum("nf,nfh->nh", x, value["kernel"][groups]) + value["bias"][groups]
        return jnp.square(y).sum()

    actual = cast(tuple[Array, Tree], jax.jit(jax.value_and_grad(grouped))(params))
    expected = jax.value_and_grad(reference)(params)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(left, right, atol=1e-5, rtol=2e-5)
    np.testing.assert_array_equal(actual[1]["kernel"][2:4], 0)


def test_pqn_grouped_batchnorm_uses_only_assigned_eligible_rows(
    model: tuple[str, Any, Tree, int],
) -> None:
    family, network, variables, width = model
    if family != "pqn":
        pytest.skip("PQN owns BatchNorm")
    x = jax.random.normal(jax.random.key(82), (2, 2, 5, 4))
    memory = jnp.zeros((2, 5, width), jnp.float32)
    groups = jnp.broadcast_to(
        jnp.asarray([[0, 0, 1, 2, 4], [2, 1, 1, 0, 0]], jnp.int32), (2, 2, 5)
    )
    starts = jnp.zeros((2, 2), jnp.bool_)
    valid = jnp.ones_like(starts)
    active = jnp.ones(groups.shape, jnp.bool_)
    eligible = (groups != 4).at[:, 0, 1].set(False)
    (_, actual), updates = network.apply(
        variables,
        memory,
        x,
        starts,
        valid,
        active,
        actor_group=groups,
        normalization_mask=eligible,
        train=True,
        mutable=["batch_stats"],
    )
    assert np.isfinite(actual).all()
    for group in range(5):
        selected = _select_group(variables, group)
        mask = eligible & (groups == group)
        if np.any(mask):
            _, expected = network.apply(
                selected,
                memory,
                x,
                starts,
                valid,
                mask,
                train=True,
                mutable=["batch_stats"],
            )
            target = expected["batch_stats"]
        else:
            target = selected["batch_stats"]
        got = _select_group(updates["batch_stats"], group)
        for left, right in zip(
            jax.tree.leaves(got), jax.tree.leaves(target), strict=True
        ):
            np.testing.assert_allclose(left, right, atol=2e-6, rtol=3e-5)


@pytest.mark.parametrize("sharing", ("class", "none"))
def test_grouped_factories_route_raw_class_or_physical_slot_and_reuse_compilation(
    model: tuple[str, Any, Tree, int],
    monkeypatch: pytest.MonkeyPatch,
    sharing: str,
) -> None:
    family, network, variables, width = model
    _, observations, state = _two_lane_reset(team_sizes=(3, 2))
    inputs = system_inputs(observations, state, team=0)
    classes = jnp.asarray([[1, 1, 3, 0, 0], [3, 2, 2, 0, 0]], jnp.float32)
    own = inputs.actors.observation._replace(
        self_features=inputs.actors.observation.self_features.at[
            ..., AGENT_FEATURE_CLASS_ID
        ].set(classes)
    )
    inputs = inputs._replace(actors=inputs.actors._replace(observation=own))
    original_encoder = ppo.encode_actor_inputs

    def encode(actors: ActorInput) -> Array:
        return original_encoder(actors)[..., :4]

    monkeypatch.setattr(ppo, "encode_actor_inputs", encode)
    if family == "pqn":
        method_variables = pqn.PQNInferenceVariables(
            variables["params"], variables["batch_stats"]
        )
        system = pqn.make_pqn_system(
            method_variables, parameter_sharing=sharing, spawn_frame="world"
        )
        other = pqn.make_pqn_system(
            method_variables,
            parameter_sharing="none" if sharing == "class" else "class",
            spawn_frame="world",
        )
    elif family == "qmix":
        system = qmix.make_qmix_system(
            variables, parameter_sharing=sharing, spawn_frame="world"
        )
        other = qmix.make_qmix_system(
            variables,
            parameter_sharing="none" if sharing == "class" else "class",
            spawn_frame="world",
        )
    else:
        method = "ff_ippo" if family == "ff_ppo" else "mappo"
        system = ppo.make_ppo_system(
            variables, method=method, parameter_sharing=sharing, spawn_frame="world"
        )
        other = ppo.make_ppo_system(
            variables,
            method=method,
            parameter_sharing="none" if sharing == "class" else "class",
            spawn_frame="world",
        )
    memory = jnp.zeros((2, 5, width), jnp.float32) if width else ()
    keys = jax.random.split(jax.random.key(84), 2)
    traces: list[int] = []

    def apply(values: Tree) -> SystemOutput:
        traces.append(1)
        return cast(SystemOutput, system.apply(values, memory, inputs, keys))

    compiled = jax.jit(apply)
    actual = cast(SystemOutput, compiled(system.variables))
    compiled(jax.tree.map(jnp.copy, system.variables))
    assert traces == [1]
    groups = (
        jnp.clip(classes.astype(jnp.int32) - 1, 0, 4)
        if sharing == "class"
        else jnp.broadcast_to(jnp.arange(5), classes.shape)
    )
    np.testing.assert_array_equal(ppo._actor_groups(inputs.actors, sharing), groups)
    features = encode(inputs.actors)
    starts = jnp.broadcast_to(inputs.episode_start[:, None], groups.shape)
    valid = jnp.broadcast_to(inputs.valid[:, None], groups.shape)
    if family == "ff_ppo":
        logits = network.apply(variables, features, actor_group=groups)
    elif family == "pqn":
        _, values = network.apply(
            variables,
            memory,
            features[None],
            inputs.episode_start[None],
            inputs.valid[None],
            inputs.active_mask[None],
            actor_group=groups[None],
        )
        logits = values[0]
    else:
        _, values = network.apply(
            variables,
            memory,
            features[None],
            starts[None],
            valid[None],
            actor_group=groups[None],
        )
        logits = values[0]
    mask = categorical_action_mask(inputs.action_mask)
    expected = (
        qmix.greedy_actions(logits, mask)
        if family in {"qmix", "pqn"}
        else sample_actions(
            logits, mask, jax.vmap(functools.partial(jax.random.split, num=5))(keys)
        )
    )
    np.testing.assert_array_equal(encode_actions(actual.actions), expected)
    assert (
        normalize_system_registration(system, phase="evaluation")[0]
        != normalize_system_registration(other, phase="evaluation")[0]
    )


@pytest.mark.parametrize("sharing", ("all", "class", "none"))
def test_actor_initializers_keep_shared_shapes_or_add_exactly_five_groups(
    sharing: str,
) -> None:
    for method in ("mappo", "ippo", "ff_mappo", "ff_ippo"):
        state = jax.eval_shape(
            functools.partial(
                ppo.initialize_ppo,
                jax.random.key(86),
                ppo.PPOConfig(parameter_sharing=sharing),
                method=method,
            )
        )
        leaf = state.actor_params["params"][
            "pre_torso" if "ff_" not in method else "torso"
        ]["Dense_0"]["kernel"]
        assert leaf.shape == ((5,) if sharing != "all" else ()) + (5165, 128)
    q = qmix.qmix_actor_template(qmix.QMIXConfig(parameter_sharing=sharing))
    assert q["params"]["pre_torso"]["Dense_0"]["kernel"].shape == (
        (5,) if sharing != "all" else ()
    ) + (5165, 256)
    p = pqn.pqn_actor_template(parameter_sharing=sharing)
    assert p.params["Dense_0"]["kernel"].shape == ((5,) if sharing != "all" else ()) + (
        5165,
        512,
    )
    for config in (ppo.PPOConfig, qmix.QMIXConfig, pqn.PQNConfig):
        with pytest.raises(ValueError, match="parameter_sharing"):
            config(parameter_sharing="bad")


def _actor_parts(state: Tree, family: str) -> tuple[Tree, Tree, Tree]:
    if family == "pqn":
        return state.network.params, state.opt_state, state.network.batch_stats
    if family == "qmix":
        return state.online_q, state.opt_state[0], state.target_q
    return state.actor_params, state.actor_opt_state, ()


def _equal_tree(left: Tree, right: Tree) -> None:
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_grouped_updates_skip_complete_absent_states_and_start_fresh_groups(
    model: tuple[str, Any, Tree, int],
) -> None:
    family, _, variables, width = model
    state: Tree
    batch: Tree
    update: Callable[..., tuple[Tree, Tree]]
    x = jax.random.normal(jax.random.key(88), (3, 2, 5, 4))
    groups = jnp.broadcast_to(jnp.asarray([0, 0, 1, 3, 4], jnp.int32), (3, 2, 5))
    owned = jnp.zeros(groups.shape, jnp.bool_).at[..., 0].set(True)
    active = jnp.ones_like(owned)
    mask = jnp.ones((*groups.shape, 198), jnp.bool_)
    actions = jnp.zeros(groups.shape, jnp.int32)
    starts = jnp.zeros((3, 2), jnp.bool_).at[0].set(True)
    valid = jnp.ones_like(starts)
    rewards = jnp.asarray([[0.3, -0.1], [0.2, 0.4], [0.0, 0.0]], jnp.float32)
    if family == "pqn":
        config = pqn.PQNConfig(parameter_sharing="class", epochs=1, num_minibatches=1)
        network = pqn.PQNInferenceVariables(
            variables["params"], variables["batch_stats"]
        )
        state = pqn.PQNTrainState(
            network,
            jax.vmap(pqn.pqn_optimizer(config, 10).init)(network.params),
            jnp.int32(0),
        )
        batch = pqn.PQNBatch(
            x,
            mask,
            actions,
            rewards,
            starts,
            jnp.zeros_like(starts),
            valid,
            active,
            jnp.zeros((2, 5, width)),
            groups,
            owned,
        )
        update = functools.partial(
            pqn.update_pqn, pqn=config, planned_learning_blocks=10
        )
    elif family == "qmix":
        config = qmix.QMIXConfig(parameter_sharing="class", update_period=2)
        mixer = qmix.QMixingNetwork().init(
            jax.random.key(89), jnp.zeros((1, 1, 5)), jnp.zeros((1, 1, 4))
        )
        optimizer = qmix._optimizer(config)
        state = qmix.QMIXTrainState(
            variables,
            jax.tree.map(functools.partial(jnp.multiply, 0.9), variables),
            mixer,
            mixer,
            (jax.vmap(optimizer.init)(variables), optimizer.init(mixer)),
            jnp.int32(0),
        )
        swap = functools.partial(jnp.swapaxes, axis1=0, axis2=1)
        batch = qmix.QMIXBatch(
            swap(x),
            swap(mask),
            swap(actions),
            swap(rewards),
            swap(starts),
            jnp.zeros((2, 3), jnp.bool_),
            swap(valid),
            swap(active),
            jax.random.normal(jax.random.key(92), (2, 3, 4)),
            swap(groups),
            swap(owned),
        )
        update = functools.partial(qmix.update_qmix, config=config)
    else:
        config = ppo.PPOConfig(
            parameter_sharing="class", groups=1, value_normalization=False
        )
        recurrent = family == "ppo"
        critic_model = (
            ppo.RecurrentValueNet() if recurrent else ppo.FeedForwardValueNet()
        )
        memory: Tree = jnp.zeros((2, 5, width)) if recurrent else ()
        reset = jnp.broadcast_to(starts[..., None], groups.shape)
        real = jnp.broadcast_to(valid[..., None], groups.shape)
        critic = (
            critic_model.init(jax.random.key(90), memory, x, reset, real)
            if recurrent
            else critic_model.init(jax.random.key(90), x)
        )
        actor_opt, critic_opt = ppo._optimizers(config)
        state = ppo.PPOTrainState(
            variables,
            critic,
            jax.vmap(actor_opt.init)(variables),
            critic_opt.init(critic),
        )
        batch = ppo.PPOMinibatch(
            x[None],
            x[None],
            mask[None],
            actions[None],
            jnp.full((1, 3, 2, 5), -jnp.log(198.0)),
            jnp.zeros((1, 3, 2, 5)),
            jax.random.normal(jax.random.key(91), (1, 3, 2, 5)),
            jnp.zeros((1, 3, 2, 5)),
            reset[None],
            real[None],
            owned[None],
            jnp.zeros((1, 3, 2, 5), jnp.bool_),
            memory[None] if recurrent else (),
            memory[None] if recurrent else (),
            groups[None],
        )
        update = functools.partial(
            ppo.update_minibatch,
            config=config,
            method="mappo" if recurrent else "ff_ippo",
        )
    compiled = cast(Callable[[Tree, Tree], tuple[Tree, Tree]], jax.jit(update))
    initial = state
    for _ in range(3):
        state, _ = compiled(state, batch)
    before_parts = _actor_parts(initial, family)
    after_parts = _actor_parts(state, family)
    for group in (1, 2, 3, 4):
        _equal_tree(
            _select_group(before_parts, group), _select_group(after_parts, group)
        )
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(_select_group(before_parts[0], 0)),
            jax.tree.leaves(_select_group(after_parts[0], 0)),
            strict=True,
        )
    )
    counts = [
        np.asarray(leaf)
        for leaf in jax.tree.leaves(after_parts[1])
        if np.issubdtype(leaf.dtype, np.integer)
    ]
    assert counts and all(value.tolist() == [3, 0, 0, 0, 0] for value in counts)
    field = "actor_samples" if family in {"ppo", "ff_ppo"} else "learner_active"
    new_owned = jnp.zeros_like(getattr(batch, field)).at[..., 3].set(True)
    changed = batch._replace(**{field: new_owned})
    activated, _ = compiled(state, changed)
    fresh_parts = _actor_parts(activated, family)
    counts = [
        np.asarray(leaf)
        for leaf in jax.tree.leaves(fresh_parts[1])
        if np.issubdtype(leaf.dtype, np.integer)
    ]
    assert all(value.tolist() == [3, 0, 0, 1, 0] for value in counts)
    _equal_tree(_select_group(after_parts, 0), _select_group(fresh_parts, 0))
    if family == "pqn":
        assert isinstance(config, pqn.PQNConfig)
        # A newly seen class has fresh RAdam moments but uses the run-wide rate.
        early, _ = compiled(state._replace(optimizer_steps=jnp.int32(0)), changed)
        start = _select_group(state.network.params, 3)
        early_params = _select_group(early.network.params, 3)
        late_params = _select_group(activated.network.params, 3)
        ratio = (0.7 * config.q_lr + 0.3e-10) / config.q_lr
        moving_zero_leaves = 0
        for before, first, later in zip(
            jax.tree.leaves(start),
            jax.tree.leaves(early_params),
            jax.tree.leaves(late_params),
            strict=True,
        ):
            # Zero-initialized biases avoid subtracting tiny steps from 1.0.
            if np.all(np.asarray(before) == 0):
                moving_zero_leaves += int(np.max(np.abs(first)) > 1e-7)
                np.testing.assert_allclose(later, first * ratio, rtol=2e-5, atol=1e-10)
        assert moving_zero_leaves > 0
    elif family == "qmix":
        assert isinstance(config, qmix.QMIXConfig)
        soft, _ = qmix.update_qmix(
            initial, batch, config=dataclasses.replace(config, hard_update=False)
        )
        for group in (1, 2, 3, 4):
            _equal_tree(
                _select_group(initial.target_q, group),
                _select_group(soft.target_q, group),
            )
    else:
        assert isinstance(config, ppo.PPOConfig)
        many = jax.tree.map(functools.partial(jnp.repeat, repeats=3, axis=0), batch)
        many = many._replace(
            actor_samples=many.actor_samples.at[1, ..., 3].set(True).at[2].set(False),
            advantages=many.advantages.at[1].multiply(-2.0),
        )
        many_config = dataclasses.replace(config, groups=3)
        method = "mappo" if family == "ppo" else "ff_ippo"
        gradients = []
        for index in (0, 1):
            group_batch = jax.tree.map(
                functools.partial(jnp.take, indices=index, axis=0), many
            )
            _, gradient = jax.value_and_grad(ppo._actor_loss, has_aux=True)(
                initial.actor_params, group_batch, many_config, method=method
            )
            gradients.append(gradient)

        def mean_gradient(a: Array, b: Array) -> Array:
            return (a + b) * 0.5

        averaged = jax.tree.map(mean_gradient, gradients[0], gradients[1])
        expected_params = initial.actor_params
        expected_opt = initial.actor_opt_state
        optimizer = ppo._optimizers(config)[0]
        for group in (0, 3):
            delta, opt = optimizer.update(
                _select_group(averaged, group),
                _select_group(initial.actor_opt_state, group),
                _select_group(initial.actor_params, group),
            )
            params = jax.tree.map(
                jnp.add, _select_group(initial.actor_params, group), delta
            )

            def put(original: Array, value: Array, index: int = group) -> Array:
                return original.at[index].set(value)

            expected_params = jax.tree.map(put, expected_params, params)
            expected_opt = jax.tree.map(put, expected_opt, opt)
        actual, _ = ppo.update_minibatch(initial, many, many_config, method=method)
        for observed, expected in zip(
            jax.tree.leaves((actual.actor_params, actual.actor_opt_state)),
            jax.tree.leaves((expected_params, expected_opt)),
            strict=True,
        ):
            np.testing.assert_allclose(observed, expected, rtol=3e-5, atol=3e-7)
    empty = batch._replace(**{field: jnp.zeros_like(getattr(batch, field))})
    unchanged, _ = compiled(activated, empty)
    _equal_tree(activated, unchanged)
    # Changing the learner's unused partner predictions must not change learning.
    keep = getattr(batch, field)
    poisoned = batch._replace(
        actor_features=jnp.where(keep[..., None], batch.actor_features, 1000.0)
    )
    clean_state, clean_metrics = compiled(initial, batch)
    changed_state, changed_metrics = compiled(initial, poisoned)
    _equal_tree(clean_state, changed_state)
    _equal_tree(clean_metrics, changed_metrics)
