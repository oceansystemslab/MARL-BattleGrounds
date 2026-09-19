"""Map native actor actions to one exactly masked categorical choice.

PPO and local Q-networks use 198 categories in movement, target, Ultimate order,
with Ultimate changing fastest. Core owns legality and physical effects. These
pure JAX helpers preserve arbitrary leading sample axes and never repair actions.
They require base dependencies only and do not retain random state.
"""

import math

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    ActionMask,
)
from marl_battlegrounds.policies.actor import ActorAction

ACTION_SCHEMA_VERSION = 1
NUM_ACTIONS = NUM_MOVE_ACTIONS * NUM_TARGET_ACTIONS * NUM_ULTIMATE_ACTIONS


def encode_actions(actions: ActorAction) -> Array:
    """Encode native action heads as int32 categorical indices.

    Parameters
    ----------
    actions : ActorAction
        Matching int32 arrays with any leading shape. Legal head ranges are
        movement 0..8, target 0..10 and Ultimate 0..1. Range validity is a caller
        precondition; this helper does not clip or consult a mask.

    Returns
    -------
    Array
        Int32 indices with the same shape, in 0..197 for valid heads. Neutral
        (0, 0, 0) becomes zero. Numerical work supports jit and vmap.

    Raises
    ------
    TypeError
        A head is not an int32 array.
    ValueError
        The head shapes differ.
    """
    if any(head.dtype != jnp.int32 for head in actions):
        raise TypeError("Action heads must be int32 arrays.")
    if any(head.shape != actions.move.shape for head in actions):
        raise ValueError("Action heads must have matching shapes.")
    return (
        actions.move * NUM_TARGET_ACTIONS + actions.select_target
    ) * NUM_ULTIMATE_ACTIONS + actions.use_ultimate


def decode_actions(indices: Array) -> ActorAction:
    """Decode int32 indices into native ActorAction heads without repair.

    Parameters
    ----------
    indices : Array
        Int32 array with any leading shape. Values in 0..197 are a caller
        precondition; zero is the neutral action.

    Returns
    -------
    ActorAction
        Same-shaped int32 movement, target and Ultimate arrays. Join teams with
        the existing environment helper before stepping. No state is changed.

    Raises
    ------
    TypeError
        indices is not int32.
    """
    if indices.dtype != jnp.int32:
        raise TypeError("Action indices must be int32 arrays.")
    combat = indices % (NUM_TARGET_ACTIONS * NUM_ULTIMATE_ACTIONS)
    return ActorAction(
        indices // (NUM_TARGET_ACTIONS * NUM_ULTIMATE_ACTIONS),
        combat // NUM_ULTIMATE_ACTIONS,
        combat % NUM_ULTIMATE_ACTIONS,
    )


def categorical_action_mask(mask: ActionMask) -> Array:
    """Combine movement legality with the exact target/Ultimate joint mask.

    Parameters
    ----------
    mask : ActionMask
        Current Boolean masks. Movement has shape (*L, 9), and joint combat has
        shape (*L, 11, 2). Separate combat marginals do not determine pair
        legality and are not used. The mask and actor input must share an epoch.

    Returns
    -------
    Array
        Boolean (*L, 198) mask in the codec's order. Valid Core masks contain
        at least one choice; dead and inactive actors admit only index zero.

    Raises
    ------
    TypeError
        Movement or joint masks are not Boolean.
    ValueError
        Their static shapes are inconsistent.
    """
    move = mask.move_mask
    joint = mask.select_target_use_ultimate_joint_mask
    if move.dtype != jnp.bool_ or joint.dtype != jnp.bool_:
        raise TypeError("Action masks must be Boolean arrays.")
    if move.shape[-1:] != (NUM_MOVE_ACTIONS,) or joint.shape != (
        *move.shape[:-1],
        NUM_TARGET_ACTIONS,
        NUM_ULTIMATE_ACTIONS,
    ):
        raise ValueError("Movement and joint combat mask shapes do not match.")
    return (move[..., :, None, None] & joint[..., None, :, :]).reshape(
        *move.shape[:-1], NUM_ACTIONS
    )


def _masked_logits(logits: Array, mask: Array) -> Array:
    """Return float32 masked scores for matching (*L,198) arrays.

    Require finite logits and at least one legal choice per row. Values must
    permit finite float32 differences for legal log probabilities. Shape/dtype
    checks are static; numerical preconditions are not checked on the host.
    Illegal entries return negative infinity. An empty malformed row stays
    invalid rather than receiving a fabricated neutral fallback.
    """
    if logits.shape[-1:] != (NUM_ACTIONS,) or mask.shape != logits.shape:
        raise ValueError("Logits and masks must have matching final width 198.")
    if mask.dtype != jnp.bool_:
        raise TypeError("The categorical mask must be Boolean.")
    return jnp.where(mask, logits.astype(jnp.float32), -jnp.inf)


def _log_probabilities(logits: Array, mask: Array) -> Array:
    """Normalize checked (...,198) scores; keep masked log probabilities at -inf."""
    return jax.nn.log_softmax(_masked_logits(logits, mask), axis=-1)


def sample_actions(logits: Array, mask: Array, keys: Array) -> Array:
    """Sample one legal categorical index per explicitly keyed sample.

    Parameters
    ----------
    logits : Array
        Finite scores shaped (*L, 198), converted to float32.
    mask : Array
        Matching Boolean legality. Every row must have a legal category.
    keys : Array
        One typed Threefry key per sample, shape L, or uint32 legacy keys
        (*L, 2) with JAX's default Threefry implementation.
        A scalar sample uses one scalar typed key or one legacy key (2,).
        The caller supplies fresh keys; this function creates no hidden stream.

    Returns
    -------
    Array
        Int32 sampled indices with shape L. Decode using decode_actions.

    Raises
    ------
    TypeError
        The mask or key representation has the wrong dtype.
    ValueError
        Logit, mask or key shapes do not match, or keys do not use Threefry.

    Notes
    -----
    Supports jit and vmap. RBG keys are rejected because their batching can
    ignore individual keys. Inputs are unchanged. Empty masks and nonfinite logits
    violate the numerical preconditions and are not repaired. Legal logit
    differences must fit float32 for finite supported log probabilities.
    """
    scores = _masked_logits(logits, mask)
    typed = jax.dtypes.issubdtype(  # pyright: ignore[reportPrivateImportUsage]
        keys.dtype, jax.dtypes.prng_key
    )
    leading = logits.shape[:-1]
    if not typed and keys.dtype != jnp.uint32:
        raise TypeError("Keys must be typed JAX keys or uint32 legacy keys.")
    if keys.shape != (leading if typed else (*leading, 2)):
        raise ValueError("Supply exactly one random key per action sample.")
    count = math.prod(leading)
    rows = scores.reshape(count, NUM_ACTIONS)
    flat_keys = keys.reshape((count,) if typed else (count, 2))
    if str(jax.random.key_impl(keys)) != "threefry2x32":
        raise ValueError("Baseline action sampling requires Threefry random keys.")
    sampled = jax.vmap(jax.random.categorical)(flat_keys, rows)
    return sampled.astype(jnp.int32).reshape(leading)


def action_log_prob(logits: Array, mask: Array, indices: Array) -> Array:
    """Return action-time log probabilities for chosen categorical indices.

    Parameters
    ----------
    logits, mask : Array
        Matching (*L,198) finite scores and Boolean legality. Scores are
        converted to float32; every row needs a legal choice.
    indices : Array
        Int32 choices shaped L, in 0..197. Range validity is a precondition.

    Returns
    -------
    Array
        Float32 shape L. A masked choice has negative-infinite log probability;
        supported legal choices have finite values. No action is repaired.

    Raises
    ------
    TypeError
        Indices or mask have the wrong dtype.
    ValueError
        Shapes disagree. Numerical logit/key preconditions match sample_actions.
    """
    if indices.dtype != jnp.int32:
        raise TypeError("Action indices must be int32 arrays.")
    if indices.shape != logits.shape[:-1]:
        raise ValueError("Action indices must match the leading logit shape.")
    logp = _log_probabilities(logits, mask)
    return jnp.take_along_axis(logp, indices[..., None], axis=-1)[..., 0]


def action_entropy(logits: Array, mask: Array) -> Array:
    """Return the masked categorical entropy in natural-log units.

    Parameters
    ----------
    logits, mask : Array
        Matching (*L,198) finite scores and Boolean legality. Scores become
        float32. Every row must contain a legal choice.

    Returns
    -------
    Array
        Float32 shape L. Masked and underflowed zero-probability entries
        contribute zero. A one-choice row has zero entropy and finite gradients.
        Inputs are unchanged; jit and automatic differentiation are supported.

    Raises
    ------
    TypeError
        The mask is not Boolean.
    ValueError
        Shapes disagree. Other numerical preconditions match sample_actions.
    """
    logp = _log_probabilities(logits, mask)
    probability = jnp.exp(logp)
    safe_logp = jnp.where(probability > 0.0, logp, 0.0)
    return -jnp.sum(probability * safe_logp, axis=-1)
