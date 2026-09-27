"""Add optional training feedback without changing simulator rewards.

Validate settings once on the host with validate_shaping. Potential feedback
preserves the discounted task objective; score-delta feedback adds a separate
combat objective. Both use the producing transition's pre-action scores even
when AutoReset returns a replacement game. Scores are Team Deathmatch points,
so both follow the task's scoring: with a positive Red Zone depth a Red Zone
death moves a team's score, and this feedback, by 2 instead of 1. There is no
separate Red Zone reward. These pure adjustments belong to learner feedback,
never actor inputs or official benchmark scores. Custom reward_adjustments
uses selected Core facts and the existing training-round clock. validate_reward
checks its scalar output before collection; learners own runtime finite checks.
resolve_reward loads a declared callback once and records available code evidence.
"""

from collections.abc import Callable
from functools import partial
from numbers import Real
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

from marl_battlegrounds.core.types import EnvState
from marl_battlegrounds.environment import EpisodeInfo, TrainingFacts, episode_advanced

type RewardFunction = Callable[[EnvState, TrainingFacts, EnvState, Array], Array]


def resolve_reward(
    reference: str | None,
) -> tuple[RewardFunction | None, dict[str, object] | None]:
    """Load one declared reward and record the available code evidence.

    Parameters
    ----------
    reference : str or None
        Installed ``module:function`` with the reward_adjustments signature.
        None disables the callback and returns (None, None) without imports.
        The imported module is trusted researcher code; imports can have their
        own side effects. This function never calls the reward.

    Returns
    -------
    tuple[callable or None, dict or None]
        The callable and its JSON-ready declaration. The record includes the
        reference, existing bytecode/default evidence, the defining module's
        source hash when readable, and the completed-rounds progress clock.
        Save this record with collection settings and compare it before resume.
        Validate shapes separately with validate_reward after preparing inputs.

    Raises
    ------
    ValueError, ImportError, AttributeError
        The shared installed-callable loader rejects the reference, import or
        attribute. Its original error is retained.

    Notes
    -----
    Host setup only. Source hashing reads one file; no callback, numerical
    transition or output write occurs. Reuse the returned callable during the
    run. A source hash covers constants written in that module, but does not
    prove imported configuration, mutable globals, closures or external state.
    Keep those values fixed and record their meaning with the experiment.
    Missing evidence stays unknown, never a verified reward identity. This is
    the same evidence limit as a recorded System, not a callback serializer.
    """
    if reference is None:
        return None, None
    import inspect
    from hashlib import sha256
    from pathlib import Path

    from marl_battlegrounds._method_loading import installed_callable
    from marl_battlegrounds.evaluation.recording_identity import (
        _callable_evidence,  # pyright: ignore[reportPrivateUsage]
    )

    reward = cast(RewardFunction, installed_callable(reference))
    try:
        source = inspect.getsourcefile(reward)
        source_digest = (
            None if source is None else sha256(Path(source).read_bytes()).hexdigest()
        )
    except OSError, TypeError:
        source_digest = None
    return reward, {
        "reference": reference,
        "callable": _callable_evidence(reward),
        "source_digest": source_digest,
        "source_scope": "defining_module" if source_digest is not None else "unknown",
        "configuration_scope": "source_and_defaults_only",
        "external_state": "unknown",
        "progress_clock": "completed_training_rounds_before_step",
    }


def reward_adjustments(
    reward: RewardFunction,
    before: EnvState,
    facts: TrainingFacts,
    after: EnvState,
    progress: Array,
) -> Array:
    """Apply one custom training reward to one producing transition.

    Parameters
    ----------
    reward : callable
        Pure JAX function reward(before, facts, after, progress). Return one
        float32 (10,) adjustment in global slot order. The function owns its
        feedback rule, including how dead or inactive agents are treated.
    before, after : EnvState
        Exact public Environment.training_state views before and after the same
        action. Pass the producing successor, never an AutoReset replacement.
        With AutoReset, enable include_training_state and use final.training_state
        for completed lanes. Neither state may enter an actor or its memory.
    facts : TrainingFacts
        Scalar producing facts from make(training_facts=True). A false
        has_transition marks padding and forces the returned adjustment to zero.
    progress : Array
        Int32 scalar completed training rounds before this batched step,
        including parent lineage. One round advances each environment once.
        Multiplying by the fixed saved batch size gives real transitions.
        Resets, padding and optimizer updates do not advance this clock.

    Returns
    -------
    Array
        Float32 (10,) extra learner feedback. Native task rewards and scores
        remain separate. Nonfinite real-step values remain visible so the
        learner's existing failure check can reject them. Inputs are unchanged.

    Raises
    ------
    TypeError
        reward is not callable, facts/progress have wrong types, or the return
        does not have float32 dtype.
    ValueError
        facts or progress are batched, or the return does not have shape (10,).

    Notes
    -----
    Pure numerical code supports jit and scan. Use vmap with progress shared
    across lanes for native batches. Validate once with validate_reward before
    collection. Callback errors propagate; numerical bounds and finiteness are
    the caller's runtime checks. Masking padding does not promise to skip the
    function inside vmap. Disabled collectors must skip this helper entirely.
    Arbitrary custom feedback need not preserve the native training objective.
    """
    if not callable(reward):
        raise TypeError("reward must be a pure JAX callable")
    if not isinstance(cast(object, facts), TrainingFacts):
        raise TypeError("facts must come from make(training_facts=True)")
    for name, value, dtype in (
        ("facts.has_transition", facts.has_transition, jnp.bool_),
        ("progress", progress, jnp.int32),
    ):
        if np.shape(value) != ():
            raise ValueError(f"{name} must be a scalar")
        if getattr(value, "dtype", None) != dtype:
            raise TypeError(f"{name} must have {jnp.dtype(dtype)} dtype")
    adjustment = reward(before, facts, after, progress)
    if np.shape(adjustment) != (10,):
        raise ValueError("reward must return shape (10,)")
    if getattr(adjustment, "dtype", None) != jnp.float32:
        raise TypeError("reward must return float32 dtype")
    return jnp.where(facts.has_transition, adjustment, jnp.float32(0))


def validate_reward(
    reward: RewardFunction,
    before: EnvState,
    facts: TrainingFacts,
    after: EnvState,
    progress: Array,
) -> None:
    """Trace a custom reward's scalar contract before collection starts.

    Parameters
    ----------
    reward, before, facts, after, progress
        The callback and scalar inputs documented by reward_adjustments. Use
        one prepared game, its producing facts and an int32 round counter.
        Actual input values are not inspected; tracing checks shapes and dtypes.

    Returns
    -------
    None
        The callback traces with an exact float32 (10,) result. This checks
        JAX compatibility, not numerical finiteness or scientific validity.

    Raises
    ------
    TypeError, ValueError
        A static input/output contract is wrong. JAX tracing errors and callback
        exceptions propagate unchanged, before a game or writer is advanced.

    Notes
    -----
    Host setup only. jax.eval_shape traces Python once but executes no numerical
    transition. The callback must have no I/O or side effects. No callable is
    saved, no clock is allocated and no file or device result is written.
    """
    jax.eval_shape(partial(reward_adjustments, reward), before, facts, after, progress)


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
        feedback rewards the team's new points minus the enemy's new points
        (one point per kill, two for a Red Zone kill when the depth is
        positive), changing that objective. Validate the mode even when the
        caller disables shaping.

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
        Nonnegative scale for own score minus opponent score, in points (a Red
        Zone death is worth 2 when the depth is positive). Has the same scalar
        type rules as discount. Zero gives zero adjustments.

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
    """Reward each team's new points minus the enemy's new points on one transition.

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
        Finite nonnegative amount per net point. Python real scalars convert to
        float32; arrays must be scalar float32. Zero gives zero adjustments.

    Returns
    -------
    Array
        Float32 (B,2) Team A/Team B adjustments: coefficient times the team's
        score change minus the enemy's score change. Scores are points, so an
        ordinary kill adds the coefficient once and a Red Zone kill (enemy dies
        in its own Red Zone, depth positive) adds it twice; an own death
        subtracts the same amount. Simultaneous changes net
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
