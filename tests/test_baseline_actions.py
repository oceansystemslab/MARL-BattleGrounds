"""Check the baseline's exact native action codec and categorical calculations.

CPU cases cover every category, coupled legality, lifecycle neutral choices,
independent keyed batches, probability/entropy gradients and public action
acceptance. Scalar/odd batches are correctness checks, not GPU benchmarks.
"""

import itertools
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.baselines.actions import (
    NUM_ACTIONS,
    action_entropy,
    action_log_prob,
    categorical_action_mask,
    decode_actions,
    encode_actions,
    sample_actions,
)
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.environment import make
from marl_battlegrounds.policies.actor import ActorAction


def test_all_native_categories_round_trip_in_declared_order() -> None:
    combinations = np.asarray(list(itertools.product(range(9), range(11), range(2))))
    native = ActorAction(
        *(jnp.asarray(combinations[:, i], jnp.int32) for i in range(3))
    )
    indices = encode_actions(native)
    np.testing.assert_array_equal(indices, np.arange(198))
    decoded = cast(ActorAction, jax.jit(decode_actions)(indices.reshape(2, 3, 33)))
    for expected, actual in zip(native, decoded, strict=True):
        assert actual.dtype == jnp.int32
        np.testing.assert_array_equal(actual.reshape(-1), expected)
    assert tuple(int(x) for x in decode_actions(jnp.asarray(0, jnp.int32))) == (0, 0, 0)


def test_joint_mask_excludes_false_pair_despite_true_marginals() -> None:
    joint = jnp.zeros((11, 2), jnp.bool_).at[0, 0].set(True).at[1, 1].set(True)
    mask = ActionMask(
        jnp.ones(9, jnp.bool_), jnp.any(joint, 1), jnp.any(joint, 0), joint
    )
    flat = categorical_action_mask(mask)
    expected = np.asarray(
        [
            bool(mask.move_mask[m] and joint[t, u])
            for m, t, u in itertools.product(range(9), range(11), range(2))
        ]
    )
    np.testing.assert_array_equal(flat, expected)
    assert mask.select_target_mask[1] and mask.use_ultimate_mask[0]
    forbidden = int(
        encode_actions(ActorAction(*(jnp.asarray(x, jnp.int32) for x in (0, 1, 0))))
    )
    assert not flat[forbidden]
    logits = jnp.zeros((4096, NUM_ACTIONS))
    sampled = cast(
        Array,
        jax.jit(sample_actions)(
            logits,
            jnp.broadcast_to(flat, logits.shape),
            jax.random.split(jax.random.key(4), 4096),
        ),
    )
    assert np.all(np.asarray(flat)[np.asarray(sampled)])
    assert np.isneginf(
        action_log_prob(jnp.zeros(198), flat, jnp.asarray(forbidden, jnp.int32))
    )


@pytest.mark.parametrize("shape", [(), (3,), (2, 3, 5)])
@pytest.mark.parametrize("legacy", [False, True])
def test_sampling_keeps_per_sample_keys_and_leading_axes(
    shape: tuple[int, ...], legacy: bool
) -> None:
    count = int(np.prod(shape))
    keys = jax.random.split(jax.random.key(15), count).reshape(shape)
    if legacy:
        keys = jax.random.key_data(keys)
    logits = jnp.arange(count * 198, dtype=jnp.float32).reshape(*shape, 198) % 17 / 9
    mask = jnp.ones_like(logits, dtype=jnp.bool_)
    result = cast(Array, jax.jit(sample_actions)(logits, mask, keys))
    rows = logits.reshape(count, 198)
    flat_keys = keys.reshape((count, 2) if legacy else (count,))
    expected = jnp.stack(
        [jax.random.categorical(flat_keys[i], rows[i]) for i in range(count)]
    )
    assert result.shape == shape
    np.testing.assert_array_equal(result.reshape(-1), expected)


@pytest.mark.parametrize("implementation", ["rbg", "unsafe_rbg"])
def test_sampling_rejects_keys_that_ignore_individual_rows_under_vmap(
    implementation: str,
) -> None:
    keys = jax.random.split(jax.random.key(1, impl=implementation), 3)
    with pytest.raises(ValueError, match="Threefry"):
        jax.jit(sample_actions)(jnp.zeros((3, 198)), jnp.ones((3, 198), bool), keys)


@pytest.mark.parametrize("single", [False, True])
def test_probabilities_entropy_and_gradients_match_independent_calculation(
    single: bool,
) -> None:
    valid_indices = [7] if single else [0, 3, 7, 52]
    logits = jnp.linspace(-10, 10, 198)
    mask = jnp.zeros(198, jnp.bool_).at[jnp.asarray(valid_indices)].set(True)
    raw = np.asarray(logits, dtype=np.float64)[valid_indices]
    weights = np.exp(raw - np.max(raw))
    probability = weights / np.sum(weights)
    expected_logp = np.log(probability)
    expected_entropy = -np.sum(probability * expected_logp)
    indices = jnp.asarray(valid_indices, jnp.int32)
    actual_logp = jax.vmap(action_log_prob, in_axes=(None, None, 0))(
        logits, mask, indices
    )
    np.testing.assert_allclose(actual_logp, expected_logp, atol=2e-6, rtol=1e-6)
    np.testing.assert_allclose(
        action_entropy(logits, mask), expected_entropy, atol=1e-6
    )
    for index, selected in enumerate(valid_indices):
        grad = cast(
            Array,
            jax.grad(action_log_prob)(logits, mask, jnp.asarray(selected, jnp.int32)),
        )
        expected = np.zeros(198)
        expected[valid_indices] = -probability
        expected[selected] += 1
        np.testing.assert_allclose(grad, expected, atol=2e-6)
        assert np.isfinite(actual_logp[index])
    entropy_grad = cast(Array, jax.grad(action_entropy)(logits, mask))
    expected = np.zeros(198)
    expected[valid_indices] = -probability * (expected_logp + expected_entropy)
    np.testing.assert_allclose(entropy_grad, expected, atol=2e-6)
    assert np.all(np.isfinite(entropy_grad))


def test_extreme_logits_and_malformed_empty_mask_are_not_repaired() -> None:
    logits = jnp.linspace(-10000, 10000, 198)
    mask = jnp.ones(198, jnp.bool_)
    assert np.isfinite(action_entropy(logits, mask))
    assert np.all(np.isfinite(cast(Array, jax.grad(action_entropy)(logits, mask))))
    assert np.isfinite(action_log_prob(logits, mask, jnp.asarray(0, jnp.int32)))
    empty = jnp.zeros(198, jnp.bool_)
    assert np.isnan(action_log_prob(logits, empty, jnp.asarray(0, jnp.int32)))


def test_public_step_accepts_decoded_actions_and_neutral_lifecycle_masks() -> None:
    config = evaluation_env_config(team_sizes=(2, 3))
    env = make("tdm", env_config=config, metrics="none")
    observations, state = env.reset(jax.random.key(4))
    # A valid dead member has zero health and no transient state at reset.
    core_state = state.core_state._replace(
        alive_mask=state.core_state.alive_mask.at[0].set(False),
        current_health=state.core_state.current_health.at[0].set(0),
    )
    prepared = core.initialize_scenario_state(core_state, config)
    observations, state = env.reset(jax.random.key(4), initial=prepared[:3])
    choices: list[ActorAction] = []
    for team in (0, 1):
        inputs = env.policy_inputs(observations, state, team=team)
        flat = categorical_action_mask(inputs.action_mask)
        indices = sample_actions(
            jnp.zeros(flat.shape),
            flat,
            jax.random.split(jax.random.key(team), 5).reshape(1, 5),
        )
        choices.append(ActorAction(*(head[0] for head in decode_actions(indices))))
        for actor in range(5):
            slot = team * 5 + actor
            if not bool(core_state.alive_mask[slot]):
                np.testing.assert_array_equal(np.flatnonzero(flat[0, actor]), [0])
                assert int(indices[0, actor]) == 0
    action = env.join_actions(*choices)
    _, after, *_ = env.step(jax.random.key(8), state, action)
    accepted_heads = (
        after.core_state.previous_timestep_move_actions,
        after.core_state.previous_timestep_select_target_actions,
        after.core_state.previous_timestep_use_ultimate_actions,
    )
    for submitted, accepted in zip(action, accepted_heads, strict=True):
        np.testing.assert_array_equal(submitted, accepted)


def test_static_shapes_and_dtypes_fail_without_action_repair() -> None:
    with pytest.raises(TypeError):
        decode_actions(jnp.asarray(0.0))
    with pytest.raises(ValueError):
        sample_actions(
            jnp.zeros((2, 198)), jnp.ones((2, 198), jnp.bool_), jax.random.key(0)
        )
    with pytest.raises(TypeError):
        action_entropy(jnp.zeros(198), jnp.ones(198))
    with pytest.raises(ValueError):
        action_log_prob(
            jnp.zeros((2, 198)),
            jnp.ones((2, 198), jnp.bool_),
            jnp.asarray(0, jnp.int32),
        )


def test_same_shape_values_reuse_compilation() -> None:
    traces = 0

    @jax.jit
    def act(logits: Array, mask: Array, keys: Array) -> Array:
        nonlocal traces
        traces += 1
        return sample_actions(logits, mask, keys)

    for seed in (1, 2):
        logits = jnp.full((3, 198), float(seed))
        mask = jnp.ones_like(logits, jnp.bool_).at[:, seed].set(False)
        jax.block_until_ready(
            cast(Array, act(logits, mask, jax.random.split(jax.random.key(seed), 3)))
        )
    assert traces == 1
