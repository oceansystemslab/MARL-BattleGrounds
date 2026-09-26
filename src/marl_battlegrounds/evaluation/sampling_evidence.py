"""Report what saved games support about sampling, without running policies.

Summaries keep the declared game groups separate from observed repeated action
or trajectory digests. Equal records never establish a sampling mechanism.
Callers own the evidence that methods and game conditions are fixed; unknown
providers remain unknown. No numerical backend, actor or learner is imported
until ``method_sampling_fact`` inspects an already constructed method.
"""

from collections.abc import Hashable, Mapping, Sequence
from numbers import Integral
from typing import Any, Literal, cast

type Determinism = Literal["deterministic", "stochastic", "unknown"]
type Record = dict[str, Any]

_LABELS = {
    "deterministic": "Deterministic",
    "stochastic": "Stochastic",
    "unknown": "Determinism Unknown",
}
_NO_VARIATION = "Unavailable: No random sampling variation"
_UNKNOWN = "Unavailable: Sampling mechanism unknown"
_TOO_FEW = "Unavailable: Too few independent sampling units"


def _count(value: object, name: str) -> int:
    """Read one nonnegative integer count, rejecting booleans and missing data."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _identity(value: object) -> Hashable:
    """Check a stable integer, nonempty string or tuple of such identities."""
    if isinstance(value, str) and value:
        return value
    if not isinstance(value, bool) and isinstance(value, Integral):
        return int(value)
    if isinstance(value, tuple) and value:
        return tuple(_identity(item) for item in cast(tuple[object, ...], value))
    raise ValueError("Sampling identities must be integers, nonempty strings or tuples")


def _repeats(game_ids: set[Hashable], digests: Mapping[Any, str] | None) -> Record:
    """Count observed matching SHA-256 digests without claiming dependence."""
    if digests is None:
        return {"recorded_games": 0, "distinct_digests": None, "repeated_games": None}
    if not set(digests) <= game_ids:
        raise ValueError("Digest evidence names an unknown completed game")
    values = tuple(digests.values())
    if any(
        not isinstance(cast(object, value), str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values
    ):
        raise ValueError("Trajectory and action digests must be lowercase SHA-256")
    distinct = len(set(values))
    return {
        "recorded_games": len(values),
        "distinct_digests": distinct,
        "repeated_games": len(values) - distinct,
    }


def summarize_sampling_evidence(
    game_ids: Sequence[Hashable],
    *,
    scheduled_games: int,
    sampling_units: Mapping[Any, Hashable | None] | None = None,
    determinism: Determinism = "unknown",
    basis: str | None = None,
    trajectory_digests: Mapping[Any, str] | None = None,
    action_digests: Mapping[Any, str] | None = None,
) -> Record:
    """Count games, declared sampling groups and separately observed repeats.

    Parameters
    ----------
    game_ids : sequence of stable identities
        Unique completed game IDs. Integers, nonempty strings and tuples work.
    scheduled_games : int
        Nonnegative number of games planned, including incomplete games.
    sampling_units : mapping or None, default=None
        Each completed game maps to its whole independent sampling group.
        A group may contain both spawn ends and other deliberately paired work.
        None as a group value marks fixed deterministic work in a mixed field.
        The mapping must cover every completed game. An absent mapping leaves
        the independent-unit count unknown, even for a stochastic method.
    determinism : {"deterministic", "stochastic", "unknown"}, default="unknown"
        Evidence about the full pairing under fixed conditions. Deterministic
        means changing sampling keys introduces no variation. Stochastic means
        all relevant randomness obeys the supplied group declaration; it does
        not promise that sampled actions or scores differ. Any unknown provider
        or unverified sampling mechanism requires unknown.
    basis : str or None, default=None
        Nonempty source or declaration explaining a known mechanism. Required
        for deterministic or stochastic evidence. Names alone are not evidence.
    trajectory_digests, action_digests : mapping or None, default=None
        Optional completed-game SHA-256 observations. A trajectory digest must
        cover the full permitted input/action trajectory; an action digest
        covers actions only. Matching digests count observed repeats but never
        change the number of supported sampling groups. No capture is performed.

    Returns
    -------
    dict
        Versioned JSON-ready counts and an explicit interval-support status.
        Deterministic conditions have zero random sampling units; unknown
        mechanisms have None. Available sampling support does not replace the
        estimator's own coverage, variation or model checks.

    Raises
    ------
    ValueError
        IDs, group coverage, counts, digests or mechanism evidence are invalid.

    Notes
    -----
    Host-only bookkeeping, linear in completed games. No policy call, file IO,
    numerical array copy or per-step transfer occurs. Inputs remain unchanged.
    """
    scheduled = _count(scheduled_games, "scheduled_games")
    identifiers = tuple(_identity(value) for value in game_ids)
    ids = set(identifiers)
    if len(ids) != len(identifiers) or len(ids) > scheduled:
        raise ValueError("Completed games must be unique and fit the scheduled count")
    if determinism not in _LABELS:
        raise ValueError("Unknown determinism declaration")
    if basis is not None and (
        not isinstance(cast(object, basis), str) or not basis.strip()
    ):
        raise ValueError("Sampling basis must be nonempty text or None")
    if determinism != "unknown" and basis is None:
        raise ValueError("Known sampling behavior needs its source or declaration")
    declared: int | None = None
    if sampling_units is not None:
        if set(sampling_units) != ids:
            raise ValueError("Sampling units must cover every completed game exactly")
        declared = len(
            {_identity(value) for value in sampling_units.values() if value is not None}
        )
    supported = 0 if determinism == "deterministic" else declared
    if determinism == "unknown":
        supported = None
    status = (
        _NO_VARIATION
        if determinism == "deterministic"
        else _UNKNOWN
        if supported is None
        else _TOO_FEW
        if supported < 2
        else "Available"
    )
    return {
        "schema_version": 1,
        "scheduled_games": scheduled,
        "completed_games": len(ids),
        "sampling_unit": "Declared whole game groups",
        "declared_sampling_units": declared,
        "supported_independent_sampling_units": supported,
        "determinism": _LABELS[determinism],
        "basis": basis,
        "interval_status": status,
        "action_observations": _repeats(ids, action_digests),
        "trajectory_observations": _repeats(ids, trajectory_digests),
        "observation_limit": (
            "Matching scores or digests do not establish determinism or dependence"
        ),
    }


def validate_sampling_evidence(
    value: Mapping[str, Any],
    *,
    scheduled_games: int | None = None,
    completed_games: int | None = None,
) -> Record:
    """Check and copy a saved version-1 sampling report without changing its meaning.

    Optional scheduled_games and completed_games must match their report counts.
    Invalid versions, counts, mechanism/status combinations or observed-repeat
    counts raise ValueError. This validates record consistency, not the truth of
    its source declaration. It reads no external evidence and writes no files.
    """
    import copy

    if value.get("schema_version") != 1 or isinstance(
        value.get("schema_version"), bool
    ):
        raise ValueError("Unsupported sampling evidence schema")
    planned = _count(value.get("scheduled_games"), "scheduled_games")
    finished = _count(value.get("completed_games"), "completed_games")
    if finished > planned:
        raise ValueError("Completed sampling games exceed the plan")
    for expected, actual in ((scheduled_games, planned), (completed_games, finished)):
        if expected is not None and _count(expected, "expected games") != actual:
            raise ValueError("Sampling evidence game counts differ from the result")
    declared = value.get("declared_sampling_units")
    if declared is not None and _count(declared, "declared_sampling_units") > finished:
        raise ValueError("Sampling units exceed completed games")
    label = value.get("determinism")
    if label not in _LABELS.values():
        raise ValueError("Unknown sampling evidence determinism")
    basis = value.get("basis")
    if basis is not None and (not isinstance(basis, str) or not basis.strip()):
        raise ValueError("Sampling basis must be nonempty text or None")
    if label != "Determinism Unknown" and basis is None:
        raise ValueError("Known sampling behavior needs its source or declaration")
    supported = value.get("supported_independent_sampling_units")
    expected_supported = (
        0
        if label == "Deterministic"
        else None
        if label == "Determinism Unknown"
        else declared
    )
    if supported is not None:
        _count(supported, "supported_independent_sampling_units")
    if supported != expected_supported:
        raise ValueError("Supported sampling units disagree with the mechanism")
    expected_status = (
        _NO_VARIATION
        if label == "Deterministic"
        else _UNKNOWN
        if supported is None
        else _TOO_FEW
        if supported < 2
        else "Available"
    )
    if value.get("interval_status") != expected_status:
        raise ValueError("Sampling interval status disagrees with its support")
    if value.get("sampling_unit") != "Declared whole game groups":
        raise ValueError("Unsupported sampling unit")
    for field in ("action_observations", "trajectory_observations"):
        observations = value.get(field)
        if not isinstance(observations, Mapping):
            raise ValueError("Sampling observations must be a record")
        observations = cast(Mapping[str, object], observations)
        recorded = _count(observations.get("recorded_games"), "recorded_games")
        distinct, repeats = (
            observations.get("distinct_digests"),
            observations.get("repeated_games"),
        )
        if recorded > finished:
            raise ValueError("Observed digest count exceeds completed games")
        if distinct is None and repeats is None and recorded == 0:
            continue
        if (
            _count(distinct, "distinct_digests") + _count(repeats, "repeated_games")
            != recorded
        ):
            raise ValueError("Observed repeat counts disagree")
    return copy.deepcopy(dict(value))


def method_sampling_fact(method: object) -> Record:
    """Describe sampling of an already frozen native method, or leave it unknown.

    The caller must use this same frozen method and unchanged numerical values
    for the games. Exact built-in callable objects and known actor factory hooks
    establish behavior; display names and checkpoint labels never do. Greedy
    Q-value hooks also require their factory's initializer, no custom reset hook,
    and scalar epsilon zero. Positive epsilon and PPO hooks use actor keys.

    Return a small record with lower-case determinism and a plain source basis.
    Unknown or host methods return unknown. Adapter policies are checked one by
    one. This setup inspection may read the one-scalar epsilon from its device;
    it never restores weights, calls an actor or imports a training learner.
    """
    import importlib
    import math
    import sys
    from types import FunctionType

    from marl_battlegrounds.evaluation import policy_execution as execution

    unknown = {"determinism": "unknown", "basis": "Sampling behavior is not verified"}
    if not isinstance(method, (execution.Policy, execution.System)):
        return unknown
    if method.execution != "jax":
        return unknown
    if isinstance(method, execution.Policy):
        for name in ("random", "tdm-alpha", "tdm-beta", "tdm-gamma"):
            if method.apply is execution.policy(name).apply:
                return {
                    "determinism": "stochastic"
                    if name == "random"
                    else "deterministic",
                    "basis": f"Exact installed {name} policy callable",
                }
        return unknown
    if method._policies:  # pyright: ignore[reportPrivateUsage]
        if method.apply is not execution._adapter_marker:  # pyright: ignore[reportPrivateUsage]
            return unknown
        facts = [method_sampling_fact(item) for item in method._policies]  # pyright: ignore[reportPrivateUsage]
        return combine_sampling_facts(facts)
    if method.reset_memory is not None or not isinstance(method.apply, FunctionType):
        return unknown
    module_name = method.apply.__module__
    if module_name not in {
        "marl_battlegrounds.baselines.qmix",
        "marl_battlegrounds.baselines.pqn",
        "marl_battlegrounds.baselines.ppo",
    }:
        return unknown
    module = sys.modules.get(module_name) or importlib.import_module(module_name)
    defaults = method.apply.__kwdefaults__ or {}
    scale, frame = defaults.get("input_scale"), defaults.get("spawn_frame_index")
    if (
        not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not math.isfinite(scale)
    ):
        return unknown
    if isinstance(frame, bool) or frame not in (0, 1):
        return unknown
    frame_name = ("world", "left")[frame]
    family = module_name.rsplit(".", 1)[-1]
    hooks = {
        "qmix": ("_q_actor_apply", "_schema_1_q_actor_apply"),
        "pqn": ("_pqn_actor_apply", "_schema_1_pqn_actor_apply"),
        "ppo": (
            "_scaled_actor_apply",
            "_schema_1_actor_apply",
            "_feedforward_actor_apply",
            "_schema_1_feedforward_actor_apply",
        ),
    }[family]
    known = any(
        method.apply is getattr(module, name)(scale, frame_name) for name in hooks
    )
    if family == "ppo":
        recurrent = method.init is module._initial_actor_memory
        direct = method.apply is module._apply_actor
        feedforward = method.init is None and any(
            method.apply is getattr(module, name)(scale, frame_name)
            for name in (
                "_feedforward_actor_apply",
                "_schema_1_feedforward_actor_apply",
            )
        )
        if (recurrent and (known or direct)) or feedforward:
            return {
                "determinism": "stochastic",
                "basis": "Exact installed sampled PPO actor hooks",
            }
        return unknown
    initializer = "_initial_q_memory" if family == "qmix" else "_initial_pqn_memory"
    variables_type = "QMIXActorVariables" if family == "qmix" else "PQNActorVariables"
    if (
        not known
        or method.init is not getattr(module, initializer)
        or not isinstance(method.variables, getattr(module, variables_type))
    ):
        return unknown
    import numpy as np

    rate = np.asarray(method.variables.epsilon)
    if rate.shape != () or not np.isfinite(rate) or not 0 <= rate <= 1:
        return unknown
    return {
        "determinism": "deterministic" if float(rate) == 0.0 else "stochastic",
        "basis": (
            f"Exact installed {family.upper()} actor hooks; "
            f"frozen epsilon {float(rate)}"
        ),
    }


def combine_sampling_facts(facts: Sequence[Mapping[str, Any]]) -> Record:
    """Join fixed methods: unknown wins, then stochastic, then deterministic.

    Each fact needs determinism (lower-case) and nonempty source basis. No facts
    or any unknown method leaves the combined mechanism unknown. This does not
    claim that game conditions or external runtime behavior are deterministic;
    the execution owner must establish those separately.
    """
    for fact in facts:
        if (
            not isinstance(cast(object, fact), Mapping)
            or fact.get("determinism") not in _LABELS
            or not isinstance(fact.get("basis"), str)
            or not fact["basis"].strip()
        ):
            raise ValueError(
                "Method sampling facts need a known label and source basis"
            )
    kinds = {fact["determinism"] for fact in facts}
    return {
        "determinism": "unknown"
        if not facts or "unknown" in kinds
        else "stochastic"
        if "stochastic" in kinds
        else "deterministic",
        "basis": "; ".join(dict.fromkeys(fact["basis"] for fact in facts))
        or "No method sampling facts recorded",
    }
