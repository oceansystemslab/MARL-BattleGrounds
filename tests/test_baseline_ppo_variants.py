"""Check PPO variant actor rights, empty memory and shared masked updates.

Historical MAPPO registrations are compared with records captured before the
variant work, using the exact stored parameters behind those records (their
initialization rounds differently with other CPU thread counts). Real
permitted inputs exercise both spawn frames, action legality and same-call
probabilities. Feedforward actors keep no recurrent parameters,
state or network loop. Masked updates preserve unused optimizers and ignore
excluded rows while keeping the existing shared ValueNorm rules. The System
factories of all six methods take actor_input_schema: the default 2 keeps
the existing hooks (and so the recorded registrations above), 1 picks the
cached schema-1 hook once per scale and frame, and any other value (0, 3, a
Boolean, a float or a string) is refused. These CPU checks prove software
contracts, not learned skill or GPU cost.
"""

# pyright: reportPrivateUsage=false
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config
from tests.training_learner_helpers import equal, finite

from marl_battlegrounds.baselines import ppo, pqn, qmix
from marl_battlegrounds.baselines.actions import (
    action_log_prob,
    categorical_action_mask,
    encode_actions,
    mirror_action_indices,
)
from marl_battlegrounds.baselines.inputs import (
    ACTOR_FEATURE_SIZE,
    TRAINING_STATE_FEATURE_SIZE,
    encode_actor_inputs,
    spawn_frame_flag,
)
from marl_battlegrounds.baselines.ppo import (
    FeedForwardActor,
    FeedForwardValueNet,
    PPOConfig,
    PPOLearningOutputs,
    PPOMetrics,
    PPOMinibatch,
    PPOTrainState,
    initialize_ppo,
    make_ppo_system,
    make_recurrent_mappo_system,
    update_minibatch,
)
from marl_battlegrounds.core.types import AGENT_FEATURE_X
from marl_battlegrounds.environment import make
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.policies.input import mirror_team_view
from marl_battlegrounds.tasks import balanced_spawn_configs

type Tree = Any


def _zero(value: Array) -> Array:
    return jnp.zeros_like(value)


@pytest.fixture(scope="module", params=("ippo", "ff_mappo", "ff_ippo"))
def variant(request: pytest.FixtureRequest) -> tuple[str, PPOTrainState]:
    method = cast(str, request.param)
    return method, initialize_ppo(jax.random.key(19047001), method=method)


@pytest.fixture(scope="module", params=("ff_mappo", "ff_ippo"))
def ff_variant(request: pytest.FixtureRequest) -> tuple[str, PPOTrainState]:
    method = cast(str, request.param)
    return method, initialize_ppo(jax.random.key(19047001), method=method)


@pytest.fixture(scope="module")
def actor_inputs() -> SystemInput:
    config = evaluation_env_config(team_sizes=(2, 2), max_steps=8)
    env = make(
        "tdm",
        env_config=balanced_spawn_configs(config, num_envs=2),
        num_envs=2,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(19047002))
    return env.policy_inputs(observations, state)


def test_historical_mappo_system_registrations_stay_exact() -> None:
    path = (
        Path(__file__).parent / "fixtures/baseline_donor/mappo_system_registration.json"
    )
    reference = json.loads(path.read_text())
    # The captured digests come from initialize_ppo(key(seed)) with two CPU
    # threads; its orthogonal initializer rounds differently with other thread
    # counts, so the exact captured parameters are stored beside the records.
    template = jax.eval_shape(
        lambda: initialize_ppo(jax.random.key(reference["seed"])).actor_params
    )
    leaves, structure = cast(
        tuple[list[tuple[Any, Any]], Any],
        jax.tree_util.tree_flatten_with_path(template),
    )
    names = [jax.tree_util.keystr(key) for key, _ in leaves]
    with np.load(path.with_name("mappo_registration_params.npz")) as stored:
        assert set(stored.files) == set(names)
        params = jax.tree_util.tree_unflatten(
            structure, [jnp.asarray(stored[name]) for name in names]
        )
    for row in reference["registrations"]:
        for factory in (make_recurrent_mappo_system, make_ppo_system):
            identity, registration = normalize_system_registration(
                factory(
                    params,
                    input_scale=row["input_scale"],
                    spawn_frame=row["spawn_frame"],
                    name="MAPPO Compatibility Reference",
                    checkpoint="packet-5-reference",
                ),
                phase="evaluation",
                frozen=True,
            )
            assert registration == row["registration"]
            assert identity == row["identity"]


def test_variant_initialization_has_only_its_own_network_and_memory(
    variant: tuple[str, PPOTrainState],
    actor_inputs: SystemInput,
) -> None:
    method, state = variant
    system = make_ppo_system(state.actor_params, method=method)
    keys = jax.random.split(jax.random.key(19047003), 2)
    if method == "ippo":
        assert system.init is not None
        memory = system.init(system.variables, actor_inputs, keys)
        assert memory.shape == (2, 5, 128)
        np.testing.assert_array_equal(memory, 0)
        expected = initialize_ppo(jax.random.key(19047001))
        equal(state.actor_params, expected.actor_params)
        assert (
            state.critic_params["params"]["pre_torso"]["Dense_0"]["kernel"].shape[0]
            == ACTOR_FEATURE_SIZE
        )
    else:
        assert system.init is None
        keys_in_tree = str(cast(object, jax.tree.structure(state.actor_params)))
        assert "ScannedRNN" not in keys_in_tree and "GRU" not in keys_in_tree
        features = encode_actor_inputs(actor_inputs.actors)
        network = FeedForwardActor()
        logits = cast(Array, network.apply(state.actor_params, features))
        scalar = cast(Array, network.apply(state.actor_params, features[0, 0]))
        assert logits.shape == (2, 5, 198) and scalar.shape == (198,)
        np.testing.assert_allclose(logits[0, 0], scalar, atol=2e-6, rtol=2e-6)
        traced = jax.make_jaxpr(network.apply)(state.actor_params, features)
        assert "scan[" not in str(traced)
        critic_width = (
            TRAINING_STATE_FEATURE_SIZE if method == "ff_mappo" else ACTOR_FEATURE_SIZE
        )
        critic = FeedForwardValueNet()
        values = cast(
            Array,
            critic.apply(
                state.critic_params, jnp.zeros((2, 5, critic_width), jnp.float32)
            ),
        )
        assert values.shape == (2, 5)
        assert "scan[" not in str(
            jax.make_jaxpr(critic.apply)(
                state.critic_params, jnp.zeros(critic_width, jnp.float32)
            )
        )
    finite(state)


@pytest.mark.parametrize("frame", ("world", "left"))
def test_feedforward_public_actions_match_one_call_in_the_declared_frame(
    ff_variant: tuple[str, PPOTrainState],
    actor_inputs: SystemInput,
    frame: str,
) -> None:
    method, state = ff_variant
    system = make_ppo_system(
        state.actor_params, method=method, input_scale=0.01, spawn_frame=frame
    )
    keys = jax.random.split(jax.random.key(19047004), 2)
    output = cast(SystemOutput, system.apply(system.variables, (), actor_inputs, keys))
    assert output.next_memory == ()
    learning = cast(PPOLearningOutputs, output.learning_outputs)
    np.testing.assert_array_equal(
        encode_actions(output.actions), learning.action_indices
    )
    native_legal = categorical_action_mask(actor_inputs.action_mask)
    assert bool(
        jnp.all(
            jnp.take_along_axis(
                native_legal, learning.action_indices[..., None], axis=-1
            )
        )
    )
    np.testing.assert_array_equal(learning.action_indices[:, 2:], 0)
    np.testing.assert_array_equal(learning.log_prob[:, 2:], 0)
    inputs, mask = actor_inputs.actors, actor_inputs.action_mask
    indices = learning.action_indices
    if frame == "left":
        flag = spawn_frame_flag(inputs, frame)
        assert bool(flag.any()) and not bool(flag.all())
        inputs, mask = mirror_team_view(inputs, mask, flag)
        indices = mirror_action_indices(indices, flag)
    logits = cast(
        Array,
        FeedForwardActor(input_scale=0.01).apply(
            state.actor_params, encode_actor_inputs(inputs)
        ),
    )
    expected = action_log_prob(logits, categorical_action_mask(mask), indices)
    np.testing.assert_allclose(learning.log_prob, expected, atol=2e-6, rtol=2e-6)
    again = cast(
        SystemOutput,
        system.apply(system.variables, output.next_memory, actor_inputs, keys),
    )
    equal(output, again)


def test_variant_actor_cannot_read_another_recipient_or_critic(
    variant: tuple[str, PPOTrainState],
    actor_inputs: SystemInput,
) -> None:
    method, state = variant
    system = make_ppo_system(state.actor_params, method=method, input_scale=0.01)
    inputs = actor_inputs._replace(episode_start=jnp.zeros(2, jnp.bool_))
    keys = jax.random.split(jax.random.key(19047005), 2)
    memory = () if system.init is None else system.init(system.variables, inputs, keys)
    first = cast(SystemOutput, system.apply(system.variables, memory, inputs, keys))
    own = inputs.actors.observation
    changed = inputs._replace(
        actors=inputs.actors._replace(
            observation=own._replace(
                self_features=own.self_features.at[:, 1, AGENT_FEATURE_X].add(17),
            )
        )
    )
    changed_memory = (
        memory if method != "ippo" else cast(Array, memory).at[:, 1].set(0.3)
    )
    second = cast(
        SystemOutput, system.apply(system.variables, changed_memory, changed, keys)
    )
    for before, after in zip(first.actions, second.actions, strict=True):
        np.testing.assert_array_equal(before[:, 0], after[:, 0])
    np.testing.assert_array_equal(
        first.learning_outputs.log_prob[:, 0], second.learning_outputs.log_prob[:, 0]
    )
    assert system.variables is state.actor_params
    changed_critic = state._replace(
        critic_params=jax.tree.map(_zero, state.critic_params), value_norm=None
    )
    other = make_ppo_system(
        changed_critic.actor_params, method=method, input_scale=0.01
    )
    equal(first, other.apply(other.variables, memory, inputs, keys))


def test_feedforward_changing_weights_inputs_and_keys_reuse_one_compilation(
    actor_inputs: SystemInput,
) -> None:
    weights = initialize_ppo(jax.random.key(19047011), method="ff_ippo").actor_params
    system = make_ppo_system(weights, method="ff_ippo", input_scale=0.01)
    traces: list[int] = []

    def apply(params: Tree, inputs: SystemInput, keys: Array) -> SystemOutput:
        traces.append(1)
        return cast(SystemOutput, system.apply(params, (), inputs, keys))

    compiled = jax.jit(apply)
    keys = jax.random.split(jax.random.key(19047012), 2)
    original = cast(SystemOutput, compiled(weights, actor_inputs, keys))
    changed = jax.tree.map(_zero, weights)
    changed["params"]["action_head"]["Dense_0"]["bias"] = (
        jnp.arange(198, dtype=jnp.float32) * 0.01
    )
    second = cast(SystemOutput, compiled(changed, actor_inputs, keys))
    observation = actor_inputs.actors.observation
    changed_inputs = actor_inputs._replace(
        actors=actor_inputs.actors._replace(
            observation=observation._replace(
                self_features=observation.self_features.at[:, 0, AGENT_FEATURE_X].add(1)
            ),
        )
    )
    third = cast(
        SystemOutput,
        compiled(
            weights, changed_inputs, jax.random.split(jax.random.key(19047013), 2)
        ),
    )
    assert traces == [1]
    assert original.next_memory == second.next_memory == third.next_memory == ()
    assert not np.array_equal(
        original.learning_outputs.log_prob, second.learning_outputs.log_prob
    )


def test_system_factories_pick_the_schema_1_hook_and_refuse_other_schemas() -> None:
    # Cast so each factory is called through one loose signature; the loop passes
    # the same empty weights to all of them, which only the hook choice reads.
    hooks = cast(
        tuple[tuple[Callable[..., System], Callable[[float, str], object]], ...],
        (
            (partial(make_ppo_system, method="ippo"), ppo._schema_1_actor_apply),
            (make_recurrent_mappo_system, ppo._schema_1_actor_apply),
            (
                partial(make_ppo_system, method="ff_mappo"),
                ppo._schema_1_feedforward_actor_apply,
            ),
            (qmix.make_qmix_system, qmix._schema_1_q_actor_apply),
            (pqn.make_pqn_system, pqn._schema_1_pqn_actor_apply),
        ),
    )
    for factory, hook in hooks:
        for scale, frame in ((1.0, "world"), (0.01, "left")):
            legacy = factory(
                (), input_scale=scale, spawn_frame=frame, actor_input_schema=1
            )
            again = factory(
                (), input_scale=scale, spawn_frame=frame, actor_input_schema=1
            )
            current = factory((), input_scale=scale, spawn_frame=frame)
            assert legacy.apply is again.apply is hook(scale, frame)
            assert current.apply is not legacy.apply
            assert legacy.init is current.init
        for schema in cast(tuple[Any, ...], (0, 3, True, 2.0, "2")):
            with pytest.raises(ValueError, match="actor_input_schema"):
                factory((), actor_input_schema=schema)
    world = make_recurrent_mappo_system((), spawn_frame="world", actor_input_schema=2)
    assert world.apply is ppo._apply_actor


@pytest.mark.parametrize("method", ("qmix", "", "FF-IPPO"))
def test_unknown_method_is_rejected_before_network_setup(method: str) -> None:
    with pytest.raises(ValueError, match=r"method|Method|PPO"):
        initialize_ppo(jax.random.key(1), method=method)
    with pytest.raises(ValueError, match=r"method|Method|PPO"):
        make_ppo_system((), method=method)


def test_feedforward_empty_sides_preserve_warmed_optimizers_and_statistics() -> None:
    config = PPOConfig(rollout_length=2, epochs=1, input_scale=0.01)
    state = initialize_ppo(jax.random.key(19047006), config, method="ff_ippo")
    shape = (2, 2, 5)
    features = jax.random.normal(jax.random.key(19047007), (*shape, ACTOR_FEATURE_SIZE))
    actions = jnp.zeros(shape, jnp.int32)
    masks = jnp.ones((*shape, 198), jnp.bool_)
    logits = cast(
        Array,
        FeedForwardActor(input_scale=config.input_scale).apply(
            state.actor_params, features
        ),
    )
    values = cast(
        Array,
        FeedForwardValueNet(input_scale=config.input_scale).apply(
            state.critic_params, features
        ),
    )
    eligible = jnp.ones(shape, jnp.bool_)
    batch = PPOMinibatch(
        features,
        features,
        masks,
        actions,
        action_log_prob(logits, masks, actions),
        values,
        jnp.arange(20, dtype=jnp.float32).reshape(shape) / 10,
        jnp.full(shape, 2, jnp.float32),
        jnp.zeros(shape, jnp.bool_),
        eligible,
        eligible,
        eligible,
        (),
        (),
    )
    update = jax.jit(partial(update_minibatch, config=config, method="ff_ippo"))
    warmed, _ = cast(tuple[PPOTrainState, PPOMetrics], update(state, batch))
    assert any(
        np.any(a != b)
        for a, b in zip(
            jax.tree.leaves(warmed.actor_opt_state),
            jax.tree.leaves(state.actor_opt_state),
            strict=True,
        )
    )
    empty = jnp.zeros(shape, jnp.bool_)
    unchanged, metrics = cast(
        tuple[PPOTrainState, PPOMetrics],
        update(warmed, batch._replace(actor_samples=empty, critic_samples=empty)),
    )
    equal(unchanged, warmed)
    assert all(np.all(np.asarray(value) == 0) for value in metrics)
    actor_only, _ = cast(
        tuple[PPOTrainState, PPOMetrics],
        update(warmed, batch._replace(critic_samples=empty)),
    )
    equal(
        (actor_only.critic_params, actor_only.critic_opt_state, actor_only.value_norm),
        (warmed.critic_params, warmed.critic_opt_state, warmed.value_norm),
    )
    critic_only, _ = cast(
        tuple[PPOTrainState, PPOMetrics],
        update(warmed, batch._replace(actor_samples=empty)),
    )
    equal(
        (critic_only.actor_params, critic_only.actor_opt_state),
        (warmed.actor_params, warmed.actor_opt_state),
    )
    assert critic_only.value_norm is not None and warmed.value_norm is not None
    assert float(critic_only.value_norm.debiasing_term) > float(
        warmed.value_norm.debiasing_term
    )


def test_feedforward_excluded_rows_do_not_change_losses_or_value_statistics() -> None:
    config = PPOConfig(rollout_length=2, epochs=1, input_scale=0.01)
    state = initialize_ppo(jax.random.key(19047008), config, method="ff_ippo")
    shape = (2, 2, 5)
    features = jnp.ones((*shape, ACTOR_FEATURE_SIZE), jnp.float32)
    actions = jnp.zeros(shape, jnp.int32)
    masks = jnp.ones((*shape, 198), jnp.bool_)
    eligible = jnp.ones(shape, jnp.bool_)
    batch = PPOMinibatch(
        features,
        features,
        masks,
        actions,
        jnp.full(shape, -jnp.log(198)),
        jnp.zeros(shape),
        jnp.arange(20, dtype=jnp.float32).reshape(shape),
        jnp.ones(shape),
        jnp.zeros(shape, jnp.bool_),
        eligible,
        eligible.at[..., 3:].set(False),
        eligible.at[..., 3].set(False),
        (),
        (),
    )
    update = jax.jit(partial(update_minibatch, config=config, method="ff_ippo"))
    original = cast(tuple[PPOTrainState, PPOMetrics], update(state, batch))
    excluded = batch._replace(
        advantages=batch.advantages.at[..., 3:].set(50_000),
        old_log_prob=batch.old_log_prob.at[..., 3:].set(2),
        targets=batch.targets.at[..., 3].set(-50_000),
        old_values=batch.old_values.at[..., 3].set(10_000),
    )
    equal(original, update(state, excluded))
    changed, _ = cast(
        tuple[PPOTrainState, PPOMetrics],
        update(state, excluded._replace(targets=excluded.targets.at[..., 4].add(10))),
    )
    expected = original[0]
    equal(changed.actor_params, expected.actor_params)
    assert not all(
        np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(changed.critic_opt_state),
            jax.tree.leaves(expected.critic_opt_state),
            strict=True,
        )
    )
