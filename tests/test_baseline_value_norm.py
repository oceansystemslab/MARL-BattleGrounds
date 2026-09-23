"""Check critic normalization against independent moments and raw-loss references.

The fixed NumPy oracle covers corrected moving averages, masks, constant targets,
empty minibatches and numerical scales. One real recurrent minibatch compares the
normalized learner with an independently prepared donor loss, including clipping
anchors, both optimizers and repeated statistics updates. These are correctness
checks; they make no learning or GPU-performance claim.
"""

from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.test_baseline_ppo import numerical_batch as numerical_batch
from tests.training_learner_helpers import equal

from marl_battlegrounds.baselines import ppo

# Independent tests inspect the numerical boundary and its fixed constants.
# pyright: reportPrivateUsage=false


def _oracle(
    state: np.ndarray[Any, np.dtype[np.float64]], values: Array, mask: Array
) -> np.ndarray[Any, np.dtype[np.float64]]:
    selected = np.asarray(values, np.float64)[np.asarray(mask, bool)]
    if not selected.size:
        return state.copy()
    observation = np.asarray((selected.mean(), np.square(selected).mean(), 1.0))
    return 0.99999 * state + (1.0 - 0.99999) * observation


@pytest.mark.parametrize("bad", [0, 1, "true", None, np.bool_(True)])
def test_normalization_requires_an_explicit_boolean(bad: object) -> None:
    with pytest.raises(ValueError, match="Python bool"):
        ppo.PPOConfig(value_normalization=cast(bool, bad))


def test_masked_moments_match_independent_reference_and_empty_is_exact() -> None:
    state = ppo._initial_value_norm()
    reference = np.zeros(3, np.float64)
    apply = cast(Any, jax.jit(ppo._update_value_norm))
    values = jnp.asarray([[[[-5.0, 2.0, 7.0, 10000.0, -10000.0]]]])
    mask = jnp.asarray([[[[True, True, True, False, False]]]])
    for offset in (0.0, 3.0, -7.0, 5.0):
        batch = values + offset
        reference = _oracle(reference, batch, mask)
        state = apply(state, batch, mask)
        np.testing.assert_allclose(np.asarray(state), reference, rtol=3e-6, atol=1e-10)
        mean, variance = ppo._value_norm_moments(state)
        expected_mean = reference[0] / max(reference[2], 1e-5)
        expected_variance = max(
            reference[1] / max(reference[2], 1e-5) - expected_mean**2, 1e-2
        )
        np.testing.assert_allclose(mean, expected_mean, rtol=3e-6, atol=1e-6)
        np.testing.assert_allclose(variance, expected_variance, rtol=3e-6, atol=1e-5)
    equal(apply(state, values, jnp.zeros_like(mask)), state)
    restored = ppo._denormalize_values(ppo._normalize_values(values, state), state)
    np.testing.assert_allclose(restored, values, rtol=2e-6, atol=2e-6)
    assert apply._cache_size() == 1


@pytest.mark.parametrize("value", [0.0, -12.0, 12.0, 1e8])
def test_startup_constant_and_large_finite_targets_have_finite_scale(
    value: float,
) -> None:
    initial = ppo._initial_value_norm()
    mean, variance = ppo._value_norm_moments(initial)
    assert float(mean) == 0
    np.testing.assert_allclose(variance, 0.01)
    targets = jnp.full((1, 2, 1, 5), value, jnp.float32)
    state = ppo._update_value_norm(initial, targets, jnp.ones_like(targets, bool))
    normalized = ppo._normalize_values(targets, state)
    assert np.isfinite(np.asarray(normalized)).all()
    np.testing.assert_allclose(
        ppo._denormalize_values(normalized, state), targets, atol=2e-6, rtol=2e-6
    )


def test_normalized_update_matches_independently_scaled_donor_loss(
    numerical_batch: tuple[ppo.PPOTrainState, ppo.PPOMinibatch, ppo.PPOConfig],
) -> None:
    raw_state, batch, raw_config = numerical_batch
    config = replace(raw_config, value_normalization=True)
    state = raw_state._replace(value_norm=ppo._initial_value_norm())
    reference = np.zeros(3, np.float64)
    original_anchors = np.asarray(batch.old_values).copy()
    run = cast(Any, jax.jit(ppo.update_minibatch, static_argnames=("config",)))
    for _ in range(2):
        reference = _oracle(reference, batch.targets, batch.critic_samples)
        mean = reference[0] / max(reference[2], 1e-5)
        std = np.sqrt(max(reference[1] / max(reference[2], 1e-5) - mean**2, 1e-2))
        expected, expected_metrics = run(
            state._replace(value_norm=None),
            batch._replace(
                targets=jnp.asarray(
                    (np.asarray(batch.targets) - mean) / std, jnp.float32
                )
            ),
            raw_config,
        )
        state, metrics = run(state, batch, config)
        equal(state._replace(value_norm=None), expected, close=True)
        np.testing.assert_allclose(
            metrics.value_loss, expected_metrics.value_loss, rtol=3e-6, atol=3e-6
        )
        np.testing.assert_allclose(
            np.asarray(state.value_norm), reference, rtol=3e-6, atol=1e-10
        )
        np.testing.assert_array_equal(batch.old_values, original_anchors)
    empty = jnp.zeros_like(batch.critic_samples)
    unchanged, metrics = run(
        state, batch._replace(critic_samples=empty, actor_samples=empty), config
    )
    equal(unchanged, state)
    assert all(np.all(np.asarray(value) == 0) for value in metrics)


def test_statistics_refresh_does_not_move_unchanged_network_clipping_anchor(
    numerical_batch: tuple[ppo.PPOTrainState, ppo.PPOMinibatch, ppo.PPOConfig],
) -> None:
    state, batch, config = numerical_batch

    def first_group(value: Array) -> Array:
        return value[0]

    one = jax.tree.map(first_group, batch)
    _, prediction = ppo.RecurrentValueNet().apply(
        state.critic_params,
        one.critic_memory,
        one.critic_features,
        one.episode_start,
        one.valid,
    )
    stats = ppo._update_value_norm(
        ppo._initial_value_norm(), batch.targets + 50.0, batch.critic_samples
    )
    prepared = one._replace(
        old_values=prediction, targets=ppo._normalize_values(one.targets + 50.0, stats)
    )
    _, (_, _, error) = ppo._critic_loss_with_metrics(
        state.critic_params, prepared, config
    )
    loss, (clip_fraction, _, _) = ppo._critic_loss_with_metrics(
        state.critic_params, prepared, config
    )
    assert float(clip_fraction) == 0
    np.testing.assert_allclose(loss, 0.5 * error)


def test_normalization_state_is_required_only_when_enabled() -> None:
    with pytest.raises(ValueError, match="disagree"):
        ppo._check_value_norm(None, True)
    with pytest.raises(ValueError, match="disagree"):
        ppo._check_value_norm(ppo._initial_value_norm(), False)
    with pytest.raises(ValueError, match="float32 scalar"):
        ppo._check_value_norm(
            ppo.ValueNormState(jnp.zeros(2), jnp.float32(0), jnp.float32(0)), True
        )
