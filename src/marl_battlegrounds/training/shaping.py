"""Add optional team score feedback without changing simulator rewards.

Validate settings once on the host with validate_shaping. Potential feedback
preserves the discounted task objective; score-delta feedback adds a separate
combat objective. Both use the producing transition's pre-action scores even
when AutoReset returns a replacement game. These pure adjustments belong to
learner feedback, never actor inputs or official benchmark scores.
"""

from numbers import Real
from typing import cast

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

from marl_battlegrounds.environment import EpisodeInfo, episode_advanced


def validate_shaping(
    *,
    discount: float | Array,
    coefficient: float | Array = 0.01,
    mode: str = "potential",
) -> None:
    """Check score-shaping settings before compiled collection starts.

    Parameters
    ----------
    discount : float or Array
        Finite real scalar from zero to one. Use the learner's reward discount
        per environment transition, including when it is zero or one.
    coefficient : float or Array, default=0.01
        Finite nonnegative real scalar. It scales the team's score difference.
        Zero produces no adjustment. Boolean and complex values are rejected.
    mode : {"potential", "score_delta"}, default="potential"
        Potential feedback preserves the discounted objective. Score-delta
        feedback rewards new kills minus new deaths, changing that objective.
        Validate the mode even when the caller disables shaping.

    Returns
    -------
    None
        All settings satisfy the host contract. Inputs stay unchanged.

    Raises
    ------
    TypeError
        A value is not real, is Boolean, or is traced inside a JAX transform.
    ValueError
        The mode is unsupported, or a numerical value is not scalar, is not
        finite, or lies outside its stated range.

    Notes
    -----
    Host-only validation may read two device scalars. Validate changing settings
    before execution, then pass them as dynamic scalar float32 arrays. This
    function reads no files and creates no reward configuration or history.
    """
    if not isinstance(cast(object, mode), str) or mode not in (
        "potential",
        "score_delta",
    ):
        raise ValueError("shaping mode must be potential or score_delta")
    for name, value in (("discount", discount), ("coefficient", coefficient)):
        if isinstance(value, Tracer):
            raise TypeError("Validate shaping on the host before compiled use")
        scalar = np.asarray(value)
        if scalar.shape != ():
            raise ValueError(f"{name} must be a scalar")
        if scalar.dtype.kind not in "iuf":
            raise TypeError(f"{name} must be a real scalar, excluding Boolean values")
        if not np.isfinite(scalar) or scalar < 0 or (name == "discount" and scalar > 1):
            limit = "in [0, 1]" if name == "discount" else "nonnegative"
            raise ValueError(f"{name} must be finite and {limit}")


def _scalar_float32(value: float | Array, *, name: str) -> Array:
    """Check one scalar setting without reading device values.

    Python and NumPy real numbers are converted to float32. Array inputs must
    already have float32 dtype. Raise ValueError for a non-scalar shape and
    TypeError for Boolean or other dtypes. Numerical bounds are preconditions.
    """
    if np.shape(value) != ():
        raise ValueError(f"{name} must be a scalar")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must have float32 dtype")
    if isinstance(value, Real):
        return jnp.asarray(value, jnp.float32)
    if getattr(value, "dtype", None) != jnp.float32:
        raise TypeError(f"{name} must have float32 dtype")
    return jnp.asarray(value)


def _score_shapes(before_scores: Array, info: EpisodeInfo) -> None:
    """Check shared score and producing-step shapes/dtypes without reading values.

    before_scores and info.team_scores must be int32 (B,2) with positive B.
    info.decision_step must be int32 (B,). Raise ValueError for a shape mismatch
    and TypeError for a dtype mismatch. Other EpisodeInfo fields are not read.
    """
    shape = np.shape(before_scores)
    if len(shape) != 2 or shape[0] == 0 or shape[1] != 2:
        raise ValueError("before_scores must have nonempty shape (B, 2)")
    for name, value, expected_shape in (
        ("before_scores", before_scores, shape),
        ("info.team_scores", info.team_scores, shape),
        ("info.decision_step", info.decision_step, shape[:1]),
    ):
        if np.shape(value) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if getattr(value, "dtype", None) != jnp.int32:
            raise TypeError(f"{name} must have int32 dtype")


def team_potential_shaping(
    before_scores: Array,
    info: EpisodeInfo,
    *,
    discount: float | Array,
    coefficient: float | Array = 0.01,
) -> Array:
    """Return each team's optional score adjustment for one real transition.

    Parameters
    ----------
    before_scores : Array
        Int32 (B,2) scores immediately before the producing action. B must be
        positive; columns are Team A and Team B. Include authored start scores.
        Read these before step, not from an AutoReset replacement game.
    info : EpisodeInfo
        Exact producing transition. team_scores must be int32 (B,2), completed
        bool (B,), and decision_step int32 (B,). Its after-action scores remain
        correct when the returned environment state already belongs to a new
        game. Other fields are not read or validated here.
    discount : float or Array
        Learner's discount per real transition, from zero to one. Python real
        scalars are converted to float32; arrays must be scalar float32.
    coefficient : float or Array, default=0.01
        Nonnegative scale for own score minus opponent score. Has the same
        scalar type rules as discount. Zero gives zero adjustments.

    Returns
    -------
    Array
        Float32 (B,2) adjustments in Team A/Team B order. Each is the discounted
        next potential minus the pre-action potential. A potential is the
        coefficient times own score minus opponent score. Real wins, losses
        and horizon draws set the next potential to zero. Non-advancing padding
        returns zero. Actor death, collection cutoffs and stage ends do not
        cancel a living game's potential. Inputs remain unchanged.

    Raises
    ------
    TypeError
        A used array or scalar has the wrong dtype.
    ValueError
        A used array or scalar has the wrong shape.

    Notes
    -----
    Pure JAX arithmetic supports jit, scan and outer vmap. Validate finite scalar
    values and their ranges with validate_shaping before compiled use; this
    helper checks only static shapes and dtypes. It performs no host transfer,
    metric calculation, reset or state update, and retains no history.

    Over a complete episode, discounted adjustments sum to minus the starting
    potential. They sum to zero for a tied start. This preserves the declared
    discounted objective for fixed starts; it proves neither faster learning
    nor an undiscounted win-rate result. Keep task rewards separate. Do not sum
    repeated actor copies to scale this team signal. A disabled collector must
    skip this helper entirely; a zero coefficient still uses the enabled path.
    """
    _score_shapes(before_scores, info)
    shape = np.shape(before_scores)[:1]
    if np.shape(info.completed) != shape:
        raise ValueError(f"info.completed must have shape {shape}")
    if getattr(info.completed, "dtype", None) != jnp.bool_:
        raise TypeError("info.completed must have bool dtype")
    gamma = _scalar_float32(discount, name="discount")
    scale = _scalar_float32(coefficient, name="coefficient")
    before = jnp.asarray(before_scores, jnp.float32)
    after = jnp.asarray(info.team_scores, jnp.float32)
    current = scale * (before[:, 0] - before[:, 1])
    following = jnp.where(info.completed, 0.0, scale * (after[:, 0] - after[:, 1]))
    adjustment = jnp.where(episode_advanced(info), gamma * following - current, 0.0)
    return jnp.stack((adjustment, -adjustment), axis=-1)


def team_score_delta_shaping(
    before_scores: Array,
    info: EpisodeInfo,
    *,
    coefficient: float | Array = 0.01,
) -> Array:
    """Reward each team's new kills minus new deaths on one transition.

    Parameters
    ----------
    before_scores : Array
        Int32 (B,2) scores immediately before the producing action, with
        positive B and Team A/Team B columns. Read them after any pending reset
        and before step. Include authored start scores.
    info : EpisodeInfo
        Producing transition with int32 team_scores (B,2) and decision_step
        (B,). Its scores belong to that transition even when AutoReset returns
        a new game. Other fields, including completion, are not read.
    coefficient : float or Array, default=0.01
        Finite nonnegative amount per net kill. Python real scalars convert to
        float32; arrays must be scalar float32. Zero gives zero adjustments.

    Returns
    -------
    Array
        Float32 (B,2) Team A/Team B adjustments. A new own kill adds the
        coefficient; a new own death subtracts it. Simultaneous changes net
        together. Real endings keep this feedback; they do not cancel it.
        Non-advancing rows return zero. No score change gives zero even when a
        team already leads. Inputs and native task rewards remain unchanged.

    Raises
    ------
    TypeError
        A used array or scalar has the wrong dtype.
    ValueError
        A used array or scalar has the wrong shape.

    Notes
    -----
    Pure JAX arithmetic supports jit, scan and outer vmap. Validate settings
    with validate_shaping(discount=..., mode="score_delta") before compiled use;
    this helper checks only static shapes and dtypes. It performs no host transfer,
    reset, metric calculation or state update. Disabled collectors skip this helper.
    Keep one team adjustment per game before adding it to active actor rewards.
    This adds a combat objective and does not preserve the original discounted
    objective. It makes no claim about learning gains or official task wins.
    """
    _score_shapes(before_scores, info)
    scale = _scalar_float32(coefficient, name="coefficient")
    changes = jnp.asarray(info.team_scores - before_scores, jnp.float32)
    adjustment = jnp.where(
        episode_advanced(info), scale * (changes[:, 0] - changes[:, 1]), 0.0
    )
    return jnp.stack((adjustment, -adjustment), axis=-1)
