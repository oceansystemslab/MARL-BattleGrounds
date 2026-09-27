"""Run fixed training-validation tasks through the existing M8 evaluator.

New panels retain any valid System or Policy and its normal M8 registration.
Tasks preserve frozen method identities, paired schedules and durable M8 passes.
Live candidates use those same registrations; custom map and roster conditions
are recorded before evaluation. Repeated names get distinct panel labels.
The learner never enters this module. New tasks keep fixed batches and live
clients in process; historical two-actor panels keep their original protocol.
Every new task records the Team Deathmatch Red Zone depth its games use (task
schemas 3, 4 and slot 2) and reports recorded kills beside points; a task
description built without a depth keeps its original layout and identity, so
saved tasks from before the Red Zone rule stay valid but are never reused under
the new rule. Ranked actors remain loadable under another depth; the ranking keeps its
original conditions. Child runs freeze their future validation points and roots
before execution. Their selection combines only checked compatible ancestry and
keeps every actor's original identity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from marl_battlegrounds.evaluation.sampling_evidence import (
    combine_sampling_facts,
    method_sampling_fact,
    summarize_sampling_evidence,
)
from marl_battlegrounds.tasks import DEFAULT_TDM_RED_ZONE_DEPTH, AgentClassName
from marl_battlegrounds.training.analysis import (
    _KILL_COLUMNS,  # pyright: ignore[reportPrivateUsage]
    _atomic_text,  # pyright: ignore[reportPrivateUsage]
    _integer,  # pyright: ignore[reportPrivateUsage]
    summarize_slot_diagnostic,
    summarize_validation,
)

if TYPE_CHECKING:
    from marl_battlegrounds.core.types import EnvConfig
    from marl_battlegrounds.evaluation.evaluate import EpisodeSpec
    from marl_battlegrounds.evaluation.policy_execution import Policy, System

Record = dict[str, Any]
EventCallback = Callable[[Record], None]
VALIDATION_MAPS = (42, 43, 44, 45, 46)
_ROOTS = {
    "routine": 19_043_002,
    "initialization": 19_043_002,
    "confirmation": 19_043_003,
    "random": 19_043_001,
    "slot": 19_043_004,
}


def _root_seed(value: object) -> int:
    """Require one plain uint32 root seed; reject Booleans and invalid numbers."""
    seed = _integer(value, "root_seed")
    if seed > 0xFFFFFFFF:
        raise ValueError("root_seed must fit an unsigned 32-bit integer")
    return seed


def _checked_depth(value: object) -> float:
    """Check one Red Zone depth that validation games will use.

    Parameters
    ----------
    value : object
        Red Zone depth in map units; it must be exactly a Python float.

    Returns
    -------
    float
        The same value, unchanged.

    Raises
    ------
    TypeError
        value is not exactly a Python float (None, bool, int and NumPy
        scalars included).
    ValueError
        value breaks Core's scalar rules (not finite, negative or -0.0, or a
        positive value that is not a normal float32 number), or a positive
        value is wider than a validation map (float32 width).

    Notes
    -----
    Host-only; the training content owner holds the scalar rules. A positive
    depth is also checked against every validation map (VALIDATION_MAPS) by
    building each map's config with Core's validation, so a too-wide depth
    fails before any validation file (such as task.json) is written.
    """
    from marl_battlegrounds.training._content import (
        _validate_red_zone_depth,  # pyright: ignore[reportPrivateUsage]
    )

    depth = _validate_red_zone_depth(value)
    if depth > 0.0:
        _check_depth_fits_validation_maps(depth)
    return depth


@lru_cache(maxsize=8)
def _check_depth_fits_validation_maps(depth: float) -> None:
    """Raise ValueError unless a positive depth fits every validation map.

    depth is a positive float that already passed the scalar rules. Build each
    VALIDATION_MAPS config once through the TDM factory, so Core checks the
    depth against the map's float32 width. Return None; a passing depth is
    cached, so repeated validation calls pay this host work once. A failing
    depth raises every time (errors are not cached).
    """
    from marl_battlegrounds.tasks import (
        canonical_tournament_rosters,
        make_standard_team_deathmatch_config,
    )

    team_a, team_b = canonical_tournament_rosters()
    for map_id in VALIDATION_MAPS:
        make_standard_team_deathmatch_config(
            map_id=map_id,
            team_a_roster=team_a,
            team_b_roster=team_b,
            red_zone_depth=depth,
        )


def _task_depth(value: float | None) -> float | None:
    """Check the optional Red Zone depth a task description records.

    None (a description built with the layout saved before the Red Zone rule,
    which records no depth) is returned as it is; any other value goes through
    _checked_depth and raises its TypeError or ValueError. Host-only.
    """
    return None if value is None else _checked_depth(value)


def _digest(value: object) -> str:
    """Hash canonical finite JSON, preserving list order and excluding no fields."""
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _json(path: Path) -> Record:
    """Read a JSON object or raise ValueError without recovering any writer."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return cast(Record, value)


def _publish(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably publish finite JSON on a local filesystem before task execution.

    Sync the containing directory after atomic replacement. Callers own immutable
    identity checks. No game starts before this host publication returns.
    """
    _atomic_text(
        path, json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ValidationPoint:
    """One reachable collection boundary and all requested thresholds that reached it.

    requested_steps is a tuple of exact rational-upward-rounded experience
    targets. env_steps is the actual cumulative experience at the boundary.
    update_index is the collection-block ordinal of that boundary: the number
    of accepted collection blocks when it is reached (for PQN this includes
    initial random blocks, so it is not an optimizer or learning-block count).
    The initial diagnostic has zero for each and is never selectable.
    """

    requested_steps: tuple[int, ...]
    env_steps: int
    update_index: int


def _collection_boundary(
    requested_steps: int,
    *,
    total_env_steps: int,
    num_envs: int,
    rollout_length: int,
    initial_rounds: int = 0,
) -> tuple[int, int]:
    """Return the first reachable collection boundary at or above a request.

    Parameters
    ----------
    requested_steps : int
        Nonnegative requested experience in environment transitions.
    total_env_steps : int
        Positive declared budget, divisible by num_envs; its round count is
        R = total_env_steps / num_envs.
    num_envs : int
        Positive number of game lanes (B).
    rollout_length : int
        Positive rounds in a normal collection block (T).
    initial_rounds : int, default=0
        Nonnegative rounds collected before normal blocks start (W). Zero is the
        PPO and QMIX rule. PQN passes W = memory_window + rollout_length; W must
        then be less than R.

    Returns
    -------
    tuple[int, int]
        ``(steps, ordinal)``. In rounds, reachable boundaries are: zero; before
        W, multiples of T capped at W; after W, ``W + j * T`` capped at R.
        steps is the first reachable boundary at or above the request, capped at
        the total, times B. ordinal is the number of collection blocks that
        reach it (``ceil(W / T) + ceil((r - W) / T)`` after W, ``ceil(r / T)``
        before). A request of zero gives ``(0, 0)``.

    Raises
    ------
    ValueError
        A count is not a plain integer in range, the total is not divisible by
        num_envs, or a positive initial_rounds is not below R.

    Notes
    -----
    Integer ceil division only, so results are exact for any Python integer.
    With ``initial_rounds=0`` this equals the earlier rule
    ``min(total, ceil(requested / (B * T)) * B * T)``. One helper serves
    explicit save checks, fraction resolution, learner validation and host
    recovery; no list of every block is built. Host-only; no I/O.
    """
    requested = _integer(requested_steps, "requested_steps", minimum=0)
    total = _integer(total_env_steps, "total_env_steps", minimum=1)
    batch = _integer(num_envs, "num_envs", minimum=1)
    length = _integer(rollout_length, "rollout_length", minimum=1)
    initial = _integer(initial_rounds, "initial_rounds", minimum=0)
    if total % batch:
        raise ValueError("total_env_steps must be divisible by num_envs")
    budget = total // batch
    if initial and initial >= budget:
        raise ValueError("initial_rounds must be below the total rounds")
    wanted = -(-requested // batch)
    if wanted <= 0:
        return 0, 0
    if initial and wanted <= initial:
        rounds = min(initial, -(-wanted // length) * length)
        return rounds * batch, -(-rounds // length)
    rounds = min(budget, initial + -(-(wanted - initial) // length) * length)
    ordinal = -(-initial // length) + -(-(rounds - initial) // length)
    return rounds * batch, ordinal


def resolve_validation_schedule(
    total_env_steps: int,
    num_envs: int,
    rollout_length: int = 128,
    fractions: Sequence[float] = tuple(i / 10 for i in range(1, 11)),
    *,
    initial_rounds: int = 0,
) -> tuple[ValidationPoint, ...]:
    """Resolve requested fractions to reachable collection boundaries, including final.

    total_env_steps is positive and divisible by positive num_envs. rollout_length
    is a positive Python integer. fractions must be finite, increasing and inside
    (0,1]. Decimal strings define their exact fractions. initial_rounds is the
    keyword-only count of rounds collected before normal blocks start: 0 (the
    default, PPO and QMIX) or PQN's memory_window + rollout_length. Each request
    rounds up to the first reachable boundary from :func:`_collection_boundary`.
    Initial zero and final budget are always included; coincident rounded points
    are merged. Return an ordered tuple without changing settings, splitting a
    rollout or doing I/O. Invalid values raise ValueError.
    """
    total = _integer(total_env_steps, "total_env_steps", minimum=1)
    batch = _integer(num_envs, "num_envs", minimum=1)
    length = _integer(rollout_length, "rollout_length", minimum=1)
    if total % batch:
        raise ValueError("total_env_steps must be divisible by num_envs")
    amounts: list[Fraction] = []
    for value in fractions:
        if (
            isinstance(value, bool)
            or not isinstance(cast(object, value), (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 1
        ):
            raise ValueError("Validation fractions must be finite values in (0,1]")
        fraction = Fraction(str(value))
        if amounts and fraction <= amounts[-1]:
            raise ValueError("Validation fractions must increase strictly")
        amounts.append(fraction)
    if not amounts or amounts[-1] != 1:
        amounts.append(Fraction(1))
    points: dict[int, list[int]] = {0: [0]}
    ordinals: dict[int, int] = {0: 0}
    for fraction in amounts:
        requested = math.ceil(total * fraction)
        actual, ordinal = _collection_boundary(
            requested,
            total_env_steps=total,
            num_envs=batch,
            rollout_length=length,
            initial_rounds=initial_rounds,
        )
        points.setdefault(actual, []).append(requested)
        ordinals[actual] = ordinal
    return tuple(
        ValidationPoint(tuple(dict.fromkeys(requests)), actual, ordinals[actual])
        for actual, requests in sorted(points.items())
    )


@dataclass(frozen=True)
class PanelMember:
    """One ordered opponent with its ordinary evaluation identity.

    New members use name, registration_id, registration and optional reference.
    The remaining fields describe historical MAPPO artifacts only and remain
    empty for new methods. path=None means no historical artifact path exists.
    """

    name: str
    path: Path | None = None
    actor_digest: str = ""
    checkpoint_id: str = ""
    run_id: str = ""
    seed: int = 0
    env_steps: int = 0
    registration_id: str = ""
    registration: Record = field(default_factory=lambda: {})
    reference: str | None = None


@dataclass(frozen=True)
class FrozenPanel:
    """Verified panel manifest, identity and retained execution methods.

    path locates panel.json. digest excludes reload paths. members preserve the
    declared order. New schema_version=2 panels retain purpose roots and live
    methods; qualified=True means their method contract was checked, not that
    they meet a learning threshold. Historical version 1 keeps Halfway/Final
    artifacts and its original Random qualification meaning. Live methods may
    contain opaque client state that is neither serialized nor reproducible.
    """

    path: Path
    digest: str
    members: tuple[PanelMember, ...]
    qualified: bool
    schema_version: int = 1
    roots: Record = field(default_factory=lambda: {})
    methods: tuple[Any, ...] = field(default=(), repr=False, compare=False)


def run_validation_declaration(
    config: Mapping[str, Any],
    panel: FrozenPanel | None,
    *,
    continuation: Mapping[str, Any] | None = None,
    deployment: Mapping[str, Any] | None = None,
) -> Record:
    """Resolve one run's exact validation points and expected task settings.

    config is the saved training config. panel is already verified, or None.
    continuation is the checked child context, with the actual start, the
    parent's small validation declaration, learner boundary rules and optional
    changes.validation. Parent absolute targets and effective roots are kept by
    default; child final is always included. Explicit env_steps replaces future
    targets only. deployment optionally fixes partner registrations, learner
    slots and physical conditions; None keeps historical bare-task bytes. Returned
    JSON-ready facts belong in run and checkpoint metadata.

    This host-only helper runs no games and writes nothing. It rejects unknown
    changes, invalid roots/counts, old targets and incompatible historical root
    overrides with ValueError. It never uses an observed result as an expected
    root. The caller verifies the parent checkpoint and supplied panel first.
    """
    from marl_battlegrounds.baselines.methods import method_settings_field

    method = config.get("method", "mappo")
    settings = config[method_settings_field(method)]
    total = _integer(config["total_env_steps"], "total_env_steps", minimum=1)
    batch = _integer(config["num_envs"], "num_envs", minimum=1)
    length = _integer(settings["rollout_length"], "rollout_length", minimum=1)
    initial = settings["memory_window"] + length if method == "pqn_vdn" else 0
    if total % batch:
        raise ValueError("total_env_steps must be divisible by num_envs")
    roots = (
        {}
        if panel is None
        else dict(panel.roots)
        if panel.schema_version == 2
        else {key: _ROOTS[key] for key in ("routine", "initialization", "confirmation")}
    )
    changes: Record = {}
    parent: Mapping[str, Any] | None = None
    start = 0
    if continuation is not None:
        parent = continuation.get("parent_validation_declaration")
        if not isinstance(parent, Mapping):
            raise ValueError("Continuation needs its parent's validation declaration")
        start = _integer(continuation.get("start_env_steps"), "start_env_steps")
        if start >= total or start % batch:
            raise ValueError("Continuation validation start must precede its end")
        raw = continuation.get("changes", {}).get("validation", {})
        if not isinstance(raw, Mapping):
            raise ValueError("validation changes must be an object")
        changes = dict(cast(Mapping[str, Any], raw))
        allowed = {
            "panel",
            "env_steps",
            "roots",
            "routine_seed_pairs",
            "confirmation_seed_pairs",
            "inherit_parent_candidates",
            "allow_different_roots",
            "bindings",
        }
        if set(changes) - allowed:
            raise ValueError("Unknown continuation validation changes")
        if panel is not None and parent.get("panel_digest") == panel.digest:
            roots = dict(parent["roots"])
    if "roots" in changes:
        supplied = changes["roots"]
        if panel is None or panel.schema_version != 2:
            raise ValueError("Historical or disabled panels cannot override roots")
        if not isinstance(supplied, Mapping):
            raise ValueError("Validation roots need known purpose names")
        supplied = cast(Mapping[str, Any], supplied)
        if set(supplied) - set(roots):
            raise ValueError("Validation roots need known purpose names")
        roots.update({key: _root_seed(value) for key, value in supplied.items()})
    if roots:
        roots = _panel_roots(roots)
    pairs = {}
    for name in ("routine_seed_pairs", "confirmation_seed_pairs"):
        value = changes.get(
            name, parent.get(name, config[name]) if parent else config[name]
        )
        pairs[name] = _integer(value, name, minimum=1)
    flags = {}
    for name, default in (
        ("inherit_parent_candidates", True),
        ("allow_different_roots", False),
    ):
        value = changes.get(name, parent.get(name, default) if parent else default)
        if type(value) is not bool:
            raise ValueError(f"{name} must be Boolean")
        flags[name] = value
    points: list[Record] = []
    if panel is not None and continuation is None:
        points = [
            {
                "requested_steps": list(point.requested_steps),
                "env_steps": point.env_steps,
                "update_index": point.update_index,
            }
            for point in resolve_validation_schedule(
                total,
                batch,
                length,
                config["validation_fractions"],
                initial_rounds=initial,
            )
        ]
    elif panel is not None:
        from marl_battlegrounds.training._continuation_schedules import (
            continuation_boundary,
            learner_continuation,
        )

        assert parent is not None and continuation is not None
        wanted = changes.get("env_steps")
        if wanted is None:
            wanted = [
                point["env_steps"]
                for point in parent["points"]
                if start < point["env_steps"] <= total
            ]
        if not isinstance(wanted, (list, tuple)):
            raise ValueError("Validation env_steps must be a list of absolute targets")
        requests = [
            _integer(value, "validation env_steps", minimum=1)
            for value in cast(Sequence[Any], wanted)
        ]
        if requests != sorted(set(requests)) or any(
            not start < value <= total for value in requests
        ):
            raise ValueError(
                "Future validation targets must increase after start through final"
            )
        if not requests or requests[-1] != total:
            requests.append(total)
        context = learner_continuation(continuation.get("learner"))
        if context is None:
            raise ValueError("Continuation validation needs its learner boundary rules")
        resolved: dict[int, Record] = {}
        for requested in requests:
            rounds, ordinal, _ = continuation_boundary(
                -(-requested // batch),
                total_rounds=total // batch,
                continuation=context,
                rollout_length=length,
                initial_rounds=initial,
                minimum=settings.get("min_buffer_size", 0),
            )
            actual = rounds * batch
            row = resolved.setdefault(
                actual,
                {"requested_steps": [], "env_steps": actual, "update_index": ordinal},
            )
            row["requested_steps"].append(requested)
        points = list(resolved.values())
    return {
        "schema_version": 1,
        "panel_path": None if panel is None else str(panel.path.absolute()),
        "panel_digest": None if panel is None else panel.digest,
        "panel_schema_version": None if panel is None else panel.schema_version,
        "start_env_steps": start,
        "total_env_steps": total,
        "points": points,
        "roots": roots,
        **pairs,
        "red_zone_depth": config.get("red_zone_depth"),
        **(
            {"deployment": _checked_deployment(deployment)}
            if deployment is not None
            else {}
        ),
        **flags,
    }


def saved_validation_declaration(
    run: Mapping[str, Any], panel: FrozenPanel | None
) -> Record | None:
    """Check a child run's frozen validation facts against its saved inputs.

    run is verified run or checkpoint metadata and panel is its checked panel.
    Historical runs without validation_declaration return None and keep their
    existing task rules. Child declarations must equal the result rebuilt from
    config and continuation; mismatches raise ValueError before writer recovery.
    This helper reads no files and changes no input.
    """
    saved = run.get("validation_declaration")
    if saved is None:
        if run.get("continuation") is not None:
            raise ValueError("Continuation is missing its validation declaration")
        return None
    if not isinstance(saved, Mapping):
        raise ValueError("Saved validation declaration must be an object")
    expected = run_validation_declaration(
        run["config"],
        panel,
        continuation=run.get("continuation"),
        deployment=run.get("validation_deployment"),
    )
    if _digest(dict(cast(Mapping[str, Any], saved))) != _digest(expected):
        raise ValueError("Saved validation declaration differs from its frozen inputs")
    return expected


def declared_panel_task(
    declaration: Mapping[str, Any],
    panel: FrozenPanel,
    *,
    checkpoint_id: str,
    actor_digest: str,
    env_steps: int,
    purpose: str,
) -> Record:
    """Build an expected task from saved facts rather than observed result fields.

    declaration is run_validation_declaration's saved result and panel is its
    checked frozen panel. Actor fields identify the verified originating learner.
    purpose is routine, initialization or confirmation. An optional deployment
    also fixes partners and physical conditions. Schema 1 keeps fixed
    roots and old task hashes. Return panel_task_description's complete task;
    mismatched panels, malformed roots or unsupported purposes raise ValueError.
    No file or model is read and no game runs.
    """
    if (
        type(declaration.get("schema_version")) is not int
        or declaration.get("schema_version") != 1
        or declaration.get("panel_digest") != panel.digest
        or declaration.get("panel_schema_version") != panel.schema_version
        or purpose not in ("routine", "initialization", "confirmation")
    ):
        raise ValueError(
            "Saved validation declaration differs from its panel or purpose"
        )
    roots = _panel_roots(declaration.get("roots"))
    if panel.schema_version == 1 and roots != {key: _ROOTS[key] for key in roots}:
        raise ValueError("Historical panels retain their original roots")
    return panel_task_description(
        checkpoint_id=checkpoint_id,
        actor_digest=actor_digest,
        env_steps=env_steps,
        panel=panel,
        purpose=purpose,
        seed_pairs=declaration[
            "confirmation_seed_pairs"
            if purpose == "confirmation"
            else "routine_seed_pairs"
        ],
        root_seed=roots[purpose] if panel.schema_version == 2 else None,
        red_zone_depth=declaration.get("red_zone_depth"),
        deployment=declaration.get("deployment"),
        **{
            key: declaration["deployment"][key]
            for key in ("maps", "system_roster", "opponent_roster")
            if key in declaration.get("deployment", {})
        },
    )


def selection_validation_results(
    routine_results: Sequence[Mapping[str, Any]],
    confirmation_results: Sequence[Mapping[str, Any]],
    inherited: Mapping[str, Any],
    *,
    declaration: Mapping[str, Any],
    panel: FrozenPanel,
    final_checkpoint_id: str,
    rule: str = "saved",
) -> tuple[list[Record], list[Record], tuple[str, ...]]:
    """Join checked ancestor candidates with child work for one final shortlist.

    The first two inputs are child-owned results. inherited is the checked
    ancestry reader's records/actors/used_roots result. declaration and panel are
    the child's frozen task rules. final_checkpoint_id is always the child final.
    rule selects the existing analysis ranking: new runs pass point_margin;
    saved keeps historical score-based choices when no new rule was recorded.
    Return combined routine rows, matching confirmations and required candidate
    IDs. A child with no eligible trained actor returns empty confirmations and
    candidate IDs; malformed rows still fail. Missing confirmations stay missing
    so the runner can play them. Original
    identities and paths remain unchanged. Incompatible inherited confirmations
    are ignored; incompatible local confirmations and ambiguous tasks raise.
    This helper runs no games, changes no input and delegates ranking to analysis.
    """
    from marl_battlegrounds.training.analysis import (
        _candidates,  # pyright: ignore[reportPrivateUsage]
        confirmation_candidates,
    )

    inherited_rows = inherited.get("records", [])
    routine = [dict(row) for row in inherited_rows if row["purpose"] == "routine"]
    routine.extend(dict(row) for row in routine_results)
    validation_root_comparison(
        routine, allow_different_roots=declaration["allow_different_roots"]
    )
    check_confirmation_roots(
        declaration["roots"]["confirmation"],
        routine,
        used_roots=inherited.get("used_roots", []),
    )
    try:
        _candidates(routine, rule=rule)
    except ValueError as error:
        if error.args != ("No eligible trained checkpoint is available",):
            raise
        if confirmation_results:
            raise ValueError(
                "Untrained checkpoints cannot have selection confirmations"
            ) from error
        return routine, [], ()
    candidates = confirmation_candidates(
        routine, final_checkpoint_id=final_checkpoint_id, rule=rule
    )
    confirmed: dict[str, Record] = {}
    for local, rows in ((False, inherited_rows), (True, confirmation_results)):
        for row in rows:
            if row["purpose"] != "confirmation":
                continue
            identifier = row["checkpoint_id"]
            if identifier not in candidates:
                if local:
                    raise ValueError(
                        "Saved confirmation includes an unselected candidate"
                    )
                continue
            expected = declared_panel_task(
                declaration,
                panel,
                checkpoint_id=identifier,
                actor_digest=row["actor_digest"],
                env_steps=row["env_steps"],
                purpose="confirmation",
            )
            if any(row.get(name) != value for name, value in expected.items()):
                if local:
                    raise ValueError(
                        "Saved confirmation differs from its declared task"
                    )
                continue
            if identifier in confirmed:
                raise ValueError("Selection has more than one matching confirmation")
            confirmed[identifier] = dict(row)
    return routine, list(confirmed.values()), candidates


def validation_root_comparison(
    records: Sequence[Mapping[str, Any]], *, allow_different_roots: bool
) -> str:
    """Label checked task roots and reject undeclared mixed-root comparisons.

    records contains one phase's routine or confirmation results. Return a
    plain-English label suitable for a saved decision. Different roots require
    allow_different_roots=True and remain explicitly unpaired. This host-only
    helper changes no records, performs no ranking and reads no files.
    """
    different = len({row["root"] for row in records}) > 1
    if different and not allow_different_roots:
        raise ValueError(
            "Different validation roots require allow_different_roots=True"
        )
    return "Unpaired: Different Declared Roots" if different else "Common Declared Root"


def check_confirmation_roots(
    root: int, records: Sequence[Mapping[str, Any]], *, used_roots: Sequence[int] = ()
) -> None:
    """Reject confirmation root reuse across all supplied candidate-family evidence.

    root is the declared confirmation root. records contain already verified
    routine/initialization tasks, including inherited tasks. used_roots retains
    verified roots whose incompatible score was excluded. Candidates may share
    one fresh confirmation root. Invalid roots or reuse raise ValueError; no I/O.
    """
    checked = _root_seed(root)
    previous = {_root_seed(value) for value in used_roots}
    previous.update(
        _root_seed(row["root"])
        for row in records
        if row.get("purpose") in ("routine", "initialization")
    )
    if checked in previous:
        raise ValueError(
            "Confirmation root must differ from every routine "
            "and initialization root used"
        )


def _artifact(path: str | Path) -> Record:
    """Verify an artifact and retain both its ID and originating learner checkpoint.

    An actor export has its own storage identity. Selection joins its original
    learner checkpoint, which the persistence owner verifies in export metadata.
    """
    from marl_battlegrounds.training.checkpoints import artifact_identity

    identity = artifact_identity(path)
    return {
        **identity,
        "artifact_id": identity["checkpoint_id"],
        "checkpoint_id": identity["metadata"].get(
            "checkpoint_id", identity["checkpoint_id"]
        ),
    }


def _method_fields(identity: Mapping[str, Any]) -> Record:
    """Return the method fields a validation summary adds for its actor.

    identity comes from _artifact. A QMIX or PQN-VDN actor adds its method
    (``"qmix"`` or ``"pqn_vdn"``) and its integer ``optimizer_steps``, so
    selection can drop actors saved before learning (QMIX warmup, PQN-VDN
    initial collection) and resume can check them against the learner; PPO
    summaries gain nothing, keeping their historical bytes. The task identity
    and score are unaffected.
    """
    method = identity.get("method")
    if method not in ("qmix", "pqn_vdn"):
        return {}
    return {"method": method, "optimizer_steps": identity["optimizer_steps"]}


def create_panel(
    halfway: str | Path | None = None,
    final: str | Path | None = None,
    *,
    output_dir: str | Path,
    diagnostics: Sequence[Mapping[str, Any]] = (),
    test_only: bool = False,
    opponents: Sequence[Any] | None = None,
    roots: Mapping[str, int] | None = None,
    ranking: object = None,
    size: int | None = None,
) -> FrozenPanel:
    """Freeze a validation panel before training or checkpoint comparison.

    Parameters
    ----------
    opponents : sequence of System, Policy or str, optional
        Ordered methods. Repeated display names gain numbered panel labels;
        their original names and full identities stay unchanged. Strings are
        built-in names, absolute actor paths or installed module:function factories.
        Factories run once; live methods are retained without serialization.
    output_dir : str or Path
        Folder for immutable panel.json. Conflicting contents are refused.
    roots : mapping or None, default=None
        New-panel uint32 purpose roots. Defaults: routine/initialization
        19043002 and confirmation 19043003. Initialization inherits a supplied
        routine root unless given separately. Confirmation must differ.
    ranking : TournamentResult, str, Path or None, default=None
        Optional complete existing development tournament, or its saved run.
        opponents then supplies its entire current candidate pool. Canonical
        current maps 42-46, 5v5, K20/H300 and paired ends are required. The
        tournament's Red Zone depth (read from its recorded configurations) is
        saved in the ranking evidence. Actors remain loadable at another depth;
        the new validation task records its own actual depth.
    size : int or None, default=None
        Positive selected count, required with ranking and forbidden otherwise.
        Highest saved Elo wins; registration IDs break exact ties.
    halfway, final : str, Path or None, default=None
        Historical positional route only: same-run MAPPO artifacts with
        positive increasing experience. Cannot be mixed with opponents.
    diagnostics : sequence of mappings, default=()
        Historical route's two complete 100-game Random qualification summaries.
    test_only : bool, default=False
        Historical route only. Skip qualification and mark a fixture unqualified.

    Returns
    -------
    FrozenPanel
        Checked manifest and retained methods. No game or learning update runs.

    Raises
    ------
    ValueError, TypeError
        Methods, declarations, evidence or an existing manifest conflict.
    OSError, ImportError
        A reference cannot be read or imported. Factory errors retain their type.
    """
    if opponents is not None:
        if halfway is not None or final is not None or diagnostics or test_only:
            raise ValueError(
                "Use opponents or the historical halfway/final panel, not both"
            )
        return _create_system_panel(
            opponents, output_dir=output_dir, roots=roots, ranking=ranking, size=size
        )
    if (
        halfway is None
        or final is None
        or roots is not None
        or ranking is not None
        or size is not None
    ):
        raise ValueError("Supply opponents, or both historical halfway/final artifacts")
    identities = [_artifact(path) for path in (halfway, final)]
    if any(
        identity.get("run_id") != identities[0].get("run_id")
        or identity.get("seed") != identities[0].get("seed")
        for identity in identities
    ):
        raise ValueError("Panel members must use one predeclared development run")
    if not 0 < identities[0]["env_steps"] < identities[1]["env_steps"]:
        raise ValueError("Panel needs positive increasing halfway/final experience")
    qualified = not test_only
    evidence: list[Record] = []
    if qualified:
        if len(diagnostics) != 2:
            raise ValueError("Panel usefulness requires both fixed Random diagnostics")
        for identity, result in zip(identities, diagnostics, strict=True):
            evidence.append(_qualify_random(identity, result))
    records: list[Record] = []
    for name, path, identity in zip(
        ("Halfway", "Final"), (halfway, final), identities, strict=True
    ):
        records.append(
            {
                "name": name,
                "path": str(Path(path).resolve()),
                **{
                    key: identity[key]
                    for key in (
                        "actor_digest",
                        "checkpoint_id",
                        "run_id",
                        "seed",
                        "env_steps",
                        "schemas",
                    )
                },
            }
        )
    content: Record = {
        "schema_version": 1,
        "provisional": True,
        "qualified": qualified,
        "inference": "Sampled masked categorical",
        "members": records,
        "qualification_evidence": evidence,
    }
    content["panel_digest"] = _panel_digest(content)
    target = Path(output_dir) / "panel.json"
    if target.exists() and _json(target) != content:
        raise ValueError("A different immutable panel already occupies output_dir")
    if not target.exists():
        _publish(target, content)
    return load_panel(target)


def _method_snapshot(value: object) -> tuple[Any, str, Record, str | None]:
    """Freeze one valid method and record exactly M8's normal registration.

    Strings use the shared built-in/actor-folder/factory loader. A factory is called
    once. Numerical values are frozen; opaque provider state stays declared as
    unknown. Return the live method, registration ID, description and reload
    reference. This performs no policy decision and serializes no client object.
    """
    from marl_battlegrounds._method_loading import (
        load_method,
        validate_saved_method_reference,
    )
    from marl_battlegrounds.evaluation.policy_execution import Policy
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
        policy_description,
    )
    from marl_battlegrounds.evaluation.system_evaluation import freeze_evaluation_method

    reference = value if isinstance(value, str) else None
    if reference is not None:
        validate_saved_method_reference(reference)
        value = load_method(reference)
    method = freeze_evaluation_method(cast(Any, value))
    description = (
        policy_description(
            method, method.variables, method.initial_carry, include_digests=True
        )
        if isinstance(method, Policy)
        else normalize_system_registration(method, phase="validation", frozen=True)[1]
    )
    identifier, registration = normalize_system_registration(
        description, phase="validation"
    )
    return method, identifier, cast(Record, registration), reference


def _checked_deployment(value: Mapping[str, Any]) -> Record:
    """Copy a frozen partner declaration and check its exact recorded identities.

    A declaration may fix only maps or rosters. Partner declarations also name
    learner slots and ordered registrations; those two fields appear together.
    This reads no models and creates no files. Execution and evidence reads share it.
    """
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
    )

    result: Record = json.loads(json.dumps(dict(value), allow_nan=False))
    allowed = {"learner_slots", "partners", "maps", "system_roster", "opponent_roster"}
    if set(result) - allowed or not result:
        raise ValueError("Validation deployment fields differ")
    paired_fields = {"learner_slots", "partners"} & result.keys()
    if not paired_fields:
        return result
    if paired_fields != {"learner_slots", "partners"}:
        raise ValueError("Validation partners and learner_slots must appear together")
    if not isinstance(result["learner_slots"], list):
        raise ValueError("Validation learner_slots must be a list")
    slots = cast(list[Any], result["learner_slots"])
    if (
        not slots
        or len(slots) > 4
        or any(type(slot) is not int or not 0 <= slot < 5 for slot in slots)
        or len(set(slots)) != len(slots)
    ):
        raise ValueError("Validation learner_slots needs one to four distinct slots")
    if not isinstance(result["partners"], list) or not result["partners"]:
        raise ValueError("Validation deployment needs at least one partner")
    entries = cast(list[Any], result["partners"])
    names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Validation partner must be an object")
        row = cast(Record, entry)
        if set(row) != {
            "name",
            "label",
            "registration_id",
            "registration",
        }:
            raise ValueError("Validation partner fields differ")
        name = row["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Validation partner names must be distinct and nonempty")
        names.add(name)
        if row["label"] not in {"familiar", "held_out", "unknown"}:
            raise ValueError("Partner labels must be familiar, held_out or unknown")
        identifier, registration = normalize_system_registration(
            row["registration"], phase="validation"
        )
        if identifier != row["registration_id"] or registration != row["registration"]:
            raise ValueError("Validation partner registration differs")
    return result


def freeze_validation_partners(
    partners: Mapping[str, System | Policy | str],
    *,
    learner_slots: Sequence[int],
    partner_labels: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], Record]:
    """Freeze named partners once and return bindings plus their JSON declaration.

    Names identify distinct ordered panel entries. learner_slots are physical
    Team A slots; each partner fills their complement through ordinary team().
    Labels are declared familiarity, not inferred training exposure. Missing
    labels mean unknown. Factories resolve once and opaque methods stay in this
    process. No policy decision, environment step or file write occurs here.
    """
    labels = {} if partner_labels is None else dict(partner_labels)
    if set(labels) - set(partners):
        raise ValueError("Partner labels name a missing validation partner")
    bindings: dict[str, Any] = {}
    rows: list[Record] = []
    for name, value in partners.items():
        method, identifier, registration, _ = _method_snapshot(value)
        bindings[name] = method
        rows.append(
            {
                "name": name,
                "label": labels.get(name, "unknown"),
                "registration_id": identifier,
                "registration": registration,
            }
        )
    declaration = _checked_deployment(
        {"learner_slots": list(learner_slots), "partners": rows}
    )
    return bindings, declaration


def _deployment_cell(task: Mapping[str, Any], index: int) -> Record | None:
    """Describe exact constituents for one composed pass's existing sidecar."""
    if "partners" not in task:
        return None
    member = task["members"][index]
    return {
        "learner": {
            key: task[key]
            for key in (
                "checkpoint_id",
                "actor_digest",
                "env_steps",
                "system_id",
                "system",
            )
            if key in task
        },
        "learner_slots": task["learner_slots"],
        "partner": task["partners"][member["partner_index"]],
        "system_roster": task["system_roster"],
        "opponent_roster": task["opponent_roster"],
    }


def _cell_directory(directory: Path, task: Mapping[str, Any], index: int) -> Path:
    """Return the existing opponent folder beneath its optional partner folder."""
    member = task["members"][index]
    if "partners" not in task:
        return _pass_directory(directory, index, member["name"])
    partner_index = member["partner_index"]
    parent = _pass_directory(
        directory,
        partner_index,
        task["partners"][partner_index]["name"],
        kind="partner",
    )
    return _pass_directory(parent, member["opponent_index"], member["opponent_name"])


def _panel_roots(roots: Mapping[str, int] | None) -> Record:
    """Return checked purpose roots, with initialization sharing routine by default."""
    values = {key: _ROOTS[key] for key in ("routine", "initialization", "confirmation")}
    if roots is not None:
        if not isinstance(cast(object, roots), Mapping):
            raise ValueError(
                "Panel roots must be a mapping of purpose names to integers"
            )
        if set(roots) - set(values):
            raise ValueError(
                "Panel roots support routine, initialization and confirmation"
            )
        values.update({key: _root_seed(value) for key, value in roots.items()})
        if "initialization" not in roots:
            values["initialization"] = values["routine"]
    if values["confirmation"] in (values["routine"], values["initialization"]):
        raise ValueError("Confirmation needs a fresh root, separate from routine games")
    return values


def _rank_panel(
    records: Sequence[Record], ranking: object, size: int | None
) -> tuple[list[int], Record]:
    """Select a declared number from a complete, current development tournament.

    ranking is a TournamentResult or its exact saved run directory. The shared
    tournament evidence owner checks real spawn pairs, coverage and identities.
    Canonical current configurations bind maps 42-46, mirrored 5v5, K20 and H300.
    The Red Zone depth is read from the ranking's own recorded configurations
    (0.0 for configurations recorded before the rule); all five maps must share
    it, and their identities must equal the canonical configurations at that
    depth. Elo is read from existing results, never refitted. Ties use
    registration IDs. The returned evidence has "schema_version": 2 and the
    ranked "red_zone_depth", which stays attached to that historical ranking.
    """
    from marl_battlegrounds.evaluation.evaluation_conditions import (
        config_record,
        restore_recorded_config,
    )
    from marl_battlegrounds.evaluation.results import TournamentResult, load_results
    from marl_battlegrounds.evaluation.tournament import (
        _prepare_pair_evidence,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.evaluation.tournament_schedule import TournamentMatch
    from marl_battlegrounds.tasks import (
        canonical_tournament_rosters,
        make_standard_team_deathmatch_config,
    )

    count = _integer(size, "size", minimum=1)
    if count > len(records):
        raise ValueError("Ranked panel size exceeds its declared candidate pool")
    result = (
        load_results(ranking, phase="tournament")
        if isinstance(ranking, (str, Path))
        else ranking
    )
    if (
        not isinstance(ranking, (str, Path, TournamentResult))
        or cast(Any, result).status != "complete"
    ):
        raise ValueError("Ranking must be a complete existing tournament result")
    result = cast(Any, result)
    metadata = result.metadata
    details = metadata
    if "schedule" not in details:
        candidates = [
            entry.get("details", {})
            for entry in metadata.get("passes", {}).values()
            if "participants" in entry.get("details", {})
            and "num_matches" in entry.get("details", {})
        ]
        if len(candidates) != 1:
            raise ValueError("Ranking needs one complete tournament schedule")
        details = candidates[0]
    names = {row["name"] for row in records}
    if (
        details.get("pairing_protocol") != "fixed-team-spawn-v1"
        or details.get("map_ids") != list(VALIDATION_MAPS)
        or details.get("score_threshold") != 20
        or details.get("max_steps") != 300
        or set(details.get("participants", {})) != names
    ):
        raise ValueError(
            "Ranking needs the entire declared pool on paired "
            "canonical development maps"
        )
    expected_ids = {row["name"]: row["registration_id"] for row in records}
    if details["participants"] != expected_ids:
        raise ValueError("Ranking methods differ from the current candidate pool")
    for row in records:
        if (
            metadata.get("systems", {}).get(row["registration_id"])
            != row["registration"]
        ):
            raise ValueError(
                "Ranking method descriptions differ from the current candidate pool"
            )
    roster_a, roster_b = canonical_tournament_rosters()
    saved_ids = details.get("configuration_ids_by_map")
    contents = metadata.get("configurations")
    if (
        not isinstance(saved_ids, dict)
        or not isinstance(contents, dict)
        or set(cast(dict[str, Any], saved_ids)) != {str(m) for m in VALIDATION_MAPS}
        or any(
            identifier not in cast(dict[str, Any], contents)
            for identifier in cast(dict[str, Any], saved_ids).values()
        )
    ):
        raise ValueError(
            "Ranking configurations differ from the current canonical content"
        )
    ids = cast(dict[str, str], saved_ids)
    recorded = [
        restore_recorded_config(
            cast(dict[str, Any], contents)[ids[str(map_id)]], ids[str(map_id)]
        )
        for map_id in VALIDATION_MAPS
    ]
    rules = {
        (float(config.team_deathmatch_red_zone_depth), historical)
        for config, historical in recorded
    }
    if len(rules) != 1:
        raise ValueError("Ranking maps use different Red Zone scoring rules")
    depth, historical = rules.pop()
    current_ids = {
        str(map_id): config_record(
            make_standard_team_deathmatch_config(
                map_id=map_id,
                team_a_roster=roster_a,
                team_b_roster=roster_b,
                score_threshold=20,
                max_steps=300,
                red_zone_depth=depth,
            ),
            historical=historical,
        )[0]
        for map_id in VALIDATION_MAPS
    }
    if ids != current_ids:
        raise ValueError(
            "Ranking configurations differ from the current canonical content"
        )

    def table(name: str) -> list[Record]:
        """Read shared result columns as plain host records without refitting them."""
        rows: list[Record] = []
        for batch in result.iter_table(name):
            columns = {key: values.tolist() for key, values in batch.items()}
            rows.extend(
                dict(zip(columns, values, strict=True))
                for values in zip(*columns.values(), strict=True)
            )
        return rows

    matches = table("matches")
    schedule = tuple(TournamentMatch(**row) for row in details["schedule"])
    evidence = _prepare_pair_evidence(
        schedule,
        matches,
        configurations=metadata["configurations"],
        systems=metadata["systems"],
        passes=metadata["passes"],
    )
    covered = {frozenset((match.team_a, match.team_b)) for match in schedule}
    expected_pairs = {
        frozenset((first, second))
        for first in names
        for second in names
        if first != second
    }
    if covered != expected_pairs:
        raise ValueError("Ranking is missing candidate matchups")
    ratings = table("tournament_results")
    if len(ratings) != len(records) or {row.get("policy") for row in ratings} != names:
        raise ValueError("Ranking is missing candidate ratings")
    scores: dict[str, float] = {}
    for row in ratings:
        value = row.get("elo")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Ranking Elo must be finite")
        scores[row["policy"]] = float(value)
    selected = sorted(
        range(len(records)),
        key=lambda index: (
            -scores[records[index]["name"]],
            records[index]["registration_id"],
        ),
    )[:count]
    return selected, {
        "schema_version": 2,
        "red_zone_depth": depth,
        "schedule_digest": details["schedule_digest"],
        "evidence_digest": _digest(evidence),
        "ratings": [
            {"registration_id": row["registration_id"], "elo": scores[row["name"]]}
            for row in records
        ],
        "size": count,
    }


def _create_system_panel(
    opponents: Sequence[Any],
    *,
    output_dir: str | Path,
    roots: Mapping[str, int] | None,
    ranking: object,
    size: int | None,
) -> FrozenPanel:
    """Freeze and save a System panel through the shared preparation owner."""
    panel, content = _prepare_system_panel(
        opponents, output_dir=output_dir, roots=roots, ranking=ranking, size=size
    )
    _publish_system_panel(panel, content)
    return panel


def _prepare_system_panel(
    opponents: Sequence[Any],
    *,
    output_dir: str | Path,
    roots: Mapping[str, int] | None,
    ranking: object,
    size: int | None,
) -> tuple[FrozenPanel, Record]:
    """Freeze ordered methods and return the checked panel plus its saved content.

    opponents is the ordered methods or references accepted by create_panel;
    factories resolve once. output_dir names the intended panel.json folder.
    roots sets optional purpose seeds; ranking and size select an existing
    tournament's highest-ranked members under create_panel's rules. Return the
    checked panel and matching JSON content, retaining frozen values and live
    clients without creating output. Publish only after team and roster checks.
    """
    if isinstance(opponents, (str, bytes)) or not opponents:
        raise ValueError(
            "opponents must be a nonempty sequence of methods or references"
        )
    snapshots = [_method_snapshot(value) for value in opponents]
    labels: list[str] = []
    reserved = {item[0].name for item in snapshots}
    for method, *_ in snapshots:
        label = method.name
        if label in labels:
            suffix = 2
            while f"{method.name} ({suffix})" in reserved | set(labels):
                suffix += 1
            label = f"{method.name} ({suffix})"
        labels.append(label)
    records = [
        {
            "name": label,
            "registration_id": identifier,
            "registration": registration,
            "reference": reference,
        }
        for label, (_, identifier, registration, reference) in zip(
            labels, snapshots, strict=True
        )
    ]
    evidence = None
    if ranking is not None:
        indices, evidence = _rank_panel(records, ranking, size)
        records = [records[index] for index in indices]
        snapshots = [snapshots[index] for index in indices]
    elif size is not None:
        raise ValueError("size is used only with an existing ranking")
    content: Record = {
        "schema_version": 2,
        "members": records,
        "roots": _panel_roots(roots),
        "selection_schema_version": 2,
        "ranking_evidence": evidence,
    }
    content["panel_digest"] = _panel_digest(content)
    target = Path(output_dir).resolve() / "panel.json"
    panel = _load_system_panel(target, content, tuple(item[0] for item in snapshots))
    return panel, content


def _publish_system_panel(panel: FrozenPanel, content: Record) -> None:
    """Save prepared content once, refusing a different panel at the same path.

    panel and content are the matching values from _prepare_system_panel. This
    reuses the immutable output check and atomic writer, without loading methods
    again or changing their frozen identity. Matching existing content is reused.
    """
    target = panel.path
    if target.exists() and _json(target) != content:
        raise ValueError("A different immutable panel already occupies output_dir")
    if not target.exists():
        _publish(target, content)


def _ranked_depth(evidence: object) -> float | None:
    """Return the Red Zone depth a System panel's ranking was played under.

    Parameters
    ----------
    evidence : object
        The panel's saved "ranking_evidence": None for opponents frozen
        directly, an object without "schema_version" for a ranking saved
        before the Red Zone rule, or a version 2 object with "red_zone_depth".

    Returns
    -------
    float or None
        None when the panel has no ranking; 0.0 for a ranking without a
        version (one point per death); otherwise the recorded depth.

    Raises
    ------
    ValueError
        The evidence is not an object, has another version, or a version 2
        record lacks a float red_zone_depth.

    Notes
    -----
    Host-only; reads only the given value.
    """
    if evidence is None:
        return None
    if not isinstance(evidence, dict):
        raise ValueError("Invalid System panel ranking evidence")
    record = cast(Record, evidence)
    if "schema_version" not in record:
        return 0.0
    depth = record.get("red_zone_depth")
    if record["schema_version"] != 2 or type(depth) is not float:
        raise ValueError("Invalid System panel ranking evidence")
    return depth


def _load_system_panel(
    target: Path,
    content: Record,
    bindings: Sequence[Any] | None,
    red_zone_depth: float | None = None,
) -> FrozenPanel:
    """Verify a new panel and retain its methods instead of reopening them per pass.

    Supplied bindings replace reload references only when their normal M8 identity
    matches. Missing live-only methods fail before any writer is opened. Factories
    are called once on load; later validation rechecks these live snapshots.
    red_zone_depth names the new task, not an actor-admission rule. Ranking
    evidence retains its own depth and is checked for valid saved structure.
    No saved ranking or game is relabelled or reused under different rules.
    """
    entries = content.get("members")
    if (
        content.get("panel_digest") != _panel_digest(content)
        or content.get("selection_schema_version") != 2
        or not isinstance(entries, list)
        or not entries
        or any(not isinstance(row, dict) for row in cast(list[object], entries))
    ):
        raise ValueError("Invalid System panel schema or digest")
    _ranked_depth(content.get("ranking_evidence"))
    if red_zone_depth is not None:
        _checked_depth(red_zone_depth)
    entries = cast(list[Record], entries)
    roots = _panel_roots(content.get("roots"))
    if roots != content.get("roots"):
        raise ValueError("System panel needs all frozen purpose roots")
    if bindings is not None and len(bindings) != len(entries):
        raise ValueError("Bindings must cover every frozen panel member in order")
    methods: list[Any] = []
    members: list[PanelMember] = []
    for index, row in enumerate(entries):
        value = bindings[index] if bindings is not None else row.get("reference")
        if value is None:
            raise ValueError(
                "This panel contains a live-only method; supply its original bindings"
            )
        if bindings is None and isinstance(value, str) and row.get("export_relative"):
            value = str((target.parent / value).resolve())
        method, identifier, registration, _ = _method_snapshot(value)
        if (
            identifier != row.get("registration_id")
            or registration != row.get("registration")
            or not isinstance(row.get("name"), str)
            or not row["name"].strip()
        ):
            raise ValueError("Panel method no longer matches its frozen M8 identity")
        methods.append(method)
        members.append(
            PanelMember(
                name=row["name"],
                registration_id=identifier,
                registration=registration,
                reference=row.get("reference"),
            )
        )
    if len({member.name for member in members}) != len(members):
        raise ValueError("Panel opponents need distinct names")
    return FrozenPanel(
        target,
        content["panel_digest"],
        tuple(members),
        True,
        schema_version=2,
        roots=roots,
        methods=tuple(methods),
    )


def panel_task_description(
    *,
    checkpoint_id: str | None,
    actor_digest: str | None,
    env_steps: int | None,
    panel: FrozenPanel,
    purpose: str,
    seed_pairs: int,
    root_seed: int | None = None,
    red_zone_depth: float | None = None,
    candidate: Mapping[str, Any] | None = None,
    maps: Sequence[int] | None = None,
    system_roster: Sequence[AgentClassName] | None = None,
    opponent_roster: Sequence[AgentClassName] | None = None,
    deployment: Mapping[str, Any] | None = None,
) -> Record:
    """Describe the saved panel's protocol without changing historical task hashes.

    New panels use independently derived opponent roots and exact native scores.
    Their root_seed override supports predeclared fresh assessment games. Legacy
    panels keep their original purpose, roots and members.

    Parameters
    ----------
    checkpoint_id, actor_digest : str or None
        Nonempty saved learner/actor identities. Both must be None for a live
        candidate registration.
    env_steps : int or None
        Saved training transitions, 0 or more. None for a live candidate.
    candidate : mapping or None, default=None
        A frozen M8 method registration. With this route, the task records the
        System identity without inventing a learner checkpoint or training step.
    maps : sequence of int or None, default=None
        Distinct installed map IDs. None uses development maps 42 through 46.
    system_roster, opponent_roster : sequence of class names or None, default=None
        Ordered physical slots, one to five members. None uses canonical 5v5.
        Explicit conditions use task schema 5 and are checked by the evaluator's
        shared source builder before anything is written.
    deployment : mapping or None, default=None
        Frozen partner registrations and learner slots, with optional conditions.
        Partner tasks use schema 6. They reuse each opponent's roots across
        partners so matching games stay paired in the aggregate uncertainty.
        Maps and rosters must also be passed through their named arguments.
    panel : FrozenPanel
        The loaded panel. A legacy (schema 1) panel is handed to
        validation_task_description unchanged, so its own rules apply.
    purpose : str
        For a new panel: "routine", "initialization", "confirmation" or
        "assessment". For a legacy panel: the purposes that
        validation_task_description accepts.
    seed_pairs : int
        Positive number of paired seeds per map.
    root_seed : int or None, default=None
        None uses the panel's saved root for purpose. A plain integer in
        [0, 2**32 - 1] replaces it for predeclared fresh games and changes the
        task ID. Assessment needs one, and it must differ from every saved
        panel root. A confirmation root must differ from the routine and
        initialization roots.
    red_zone_depth : float or None, default=None
        Red Zone depth in map units that the task's games use. None builds the
        layout saved before the Red Zone rule: schema 2 for new panels, schema 1
        for legacy panels, with their original bytes and task IDs. A float
        builds the current layout: schema 4 for new panels (selection schema
        stays 2) and schema 3 for legacy panels, each recording
        "red_zone_depth". So tasks under different depths (0.0 included) have
        different IDs, and a saved task is never reused under another depth.

    Returns
    -------
    dict
        A new JSON-ready description ending with task_id, the hash of every
        other field. For a new panel it lists each member's name, registration
        ID and its own opponent root, derived from the task root.

    Raises
    ------
    TypeError
        red_zone_depth is neither None nor a Python float.
    ValueError
        red_zone_depth breaks Core's scalar rules (not finite, negative, -0.0,
        or a positive value that is not a normal float32 number); env_steps or
        seed_pairs is not a valid integer; purpose is unknown; a root is
        missing, reused or outside uint32; an identity is empty; or two
        opponent roots collide.

    Notes
    -----
    Host-only. This helper opens no file and runs no game. The caller checks
    the actor and the panel. Core checks the depth against each map's width
    later, when a game's config is built.
    """
    deployment = None if deployment is None else _checked_deployment(deployment)
    paired_partners = deployment is not None and "partners" in deployment
    custom_conditions = deployment is not None or any(
        value is not None for value in (maps, system_roster, opponent_roster)
    )
    if panel.schema_version == 1:
        if candidate is not None or custom_conditions:
            raise ValueError(
                "Live candidates and custom conditions need an opponents panel"
            )
        if checkpoint_id is None or actor_digest is None or env_steps is None:
            raise ValueError("Historical panels need a saved checkpoint identity")
        return validation_task_description(
            checkpoint_id=checkpoint_id,
            actor_digest=actor_digest,
            env_steps=env_steps,
            panel_digest=panel.digest,
            purpose=purpose,
            seed_pairs=seed_pairs,
            members=tuple(
                (member.name, member.actor_digest) for member in panel.members
            ),
            root_seed=root_seed,
            red_zone_depth=red_zone_depth,
        )
    depth = _task_depth(red_zone_depth)
    if candidate is None:
        _integer(env_steps, "env_steps")
        if not checkpoint_id or not actor_digest:
            raise ValueError("Validation needs checkpoint and actor identities")
        focal: Record = {
            "checkpoint_id": checkpoint_id,
            "actor_digest": actor_digest,
            "env_steps": env_steps,
        }
    else:
        if any(value is not None for value in (checkpoint_id, actor_digest, env_steps)):
            raise ValueError(
                "Supply a candidate registration or checkpoint fields, not both"
            )
        from marl_battlegrounds.evaluation.recording_identity import (
            normalize_system_registration,
        )

        identifier, description = normalize_system_registration(
            candidate, phase="validation"
        )
        focal = {"system_id": identifier, "system": description}
    map_ids = tuple(VALIDATION_MAPS if maps is None else maps)
    conditions: Record = {}
    if custom_conditions:
        from marl_battlegrounds.evaluation.evaluate import normalize_episode_specs
        from marl_battlegrounds.tasks import canonical_tournament_rosters

        if not map_ids or len(set(map_ids)) != len(map_ids):
            raise ValueError("Validation maps must be nonempty and distinct")
        for map_id in map_ids:
            _integer(map_id, "map_id")
        default_a, default_b = canonical_tournament_rosters()
        roster_a = tuple(default_a if system_roster is None else system_roster)
        roster_b = tuple(default_b if opponent_roster is None else opponent_roster)
        normalize_episode_specs(
            map_ids,
            len(map_ids),
            roster_a,
            roster_b,
            20,
            300,
            red_zone_depth=DEFAULT_TDM_RED_ZONE_DEPTH if depth is None else depth,
        )
        conditions = {
            "system_roster": list(roster_a),
            "opponent_roster": list(roster_b),
        }
    _integer(seed_pairs, "seed_pairs", minimum=1)
    if purpose not in (*panel.roots, "assessment"):
        raise ValueError("Unknown System panel validation purpose")
    if root_seed is None and purpose == "assessment":
        raise ValueError("Assessment requires its own explicit fresh root_seed")
    root = panel.roots[purpose] if root_seed is None else _root_seed(root_seed)
    if purpose == "confirmation" and root in (
        panel.roots["routine"],
        panel.roots["initialization"],
    ):
        raise ValueError(
            "Confirmation requires a fresh root separate from routine games"
        )
    if purpose == "assessment" and root in panel.roots.values():
        raise ValueError(
            "Assessment requires a fresh root separate from selection games"
        )
    registrations = [member.registration_id for member in panel.members]
    members = [
        {
            "name": member.name,
            "registration_id": member.registration_id,
            "root": int(
                _digest(
                    {
                        "root": root,
                        "opponent": member.registration_id,
                        **(
                            {"member": member.name}
                            if registrations.count(member.registration_id) > 1
                            else {}
                        ),
                    }
                )[:8],
                16,
            ),
        }
        for member in panel.members
    ]
    if len({member["root"] for member in members}) != len(members):
        raise ValueError("Opponent roots collided; choose a different panel root")
    if paired_partners:
        assert deployment is not None
        members = [
            {
                **member,
                "name": f"{partner_index + 1}: {partner['name']} / {member['name']}",
                "opponent_name": member["name"],
                "opponent_index": index,
                "partner_index": partner_index,
            }
            for partner_index, partner in enumerate(deployment["partners"])
            for index, member in enumerate(members)
        ]
    result: Record = {
        "schema_version": 6
        if paired_partners
        else 5
        if candidate is not None or custom_conditions
        else (2 if depth is None else 4),
        "selection_schema_version": 2,
        **focal,
        **conditions,
        **(
            {
                "learner_slots": cast(Record, deployment)["learner_slots"],
                "partners": cast(Record, deployment)["partners"],
                "paired_partners": True,
            }
            if paired_partners
            else {}
        ),
        "panel_digest": panel.digest,
        "purpose": purpose,
        "seed_pairs": seed_pairs,
        "maps": list(map_ids),
        "root": root,
        "members": members,
    }
    if depth is not None:
        result["red_zone_depth"] = depth
    return {**result, "task_id": _digest(result)}


def _verify_panel_pass(
    run_dir: Path,
    actor: object,
    opponent: object,
    *,
    task: Mapping[str, Any],
    member_index: int,
    num_envs: int,
    chunk_size: int = 128,
) -> None:
    """Bind an existing pass to its exact task before reuse or writer recovery.

    actor and opponent are the already loaded frozen methods. task is the checked
    System-panel task: schema 2, schema 4 with depth, or schema 5 with a live
    candidate or custom conditions. member_index chooses its ordered opponent/root.
    num_envs and chunk_size are the intended execution settings. Recorded depth,
    maps and rosters are asserted against the saved pass. The M8 owner verifies
    generated conditions, actual registrations and saved declarations. It opens
    no writer and calls neither method. A mismatch raises ValueError.
    """
    from marl_battlegrounds.evaluation.evaluate import (
        _verify_evaluation,  # pyright: ignore[reportPrivateUsage]
    )

    member = task["members"][member_index]
    rules: Record = (
        {"red_zone_depth": task["red_zone_depth"]} if "red_zone_depth" in task else {}
    )
    rules.update(
        {
            name: task[name]
            for name in ("system_roster", "opponent_roster")
            if name in task
        }
    )
    _verify_evaluation(
        cast(Any, actor),
        cast(Any, opponent),
        num_episodes=len(task["maps"]) * task["seed_pairs"] * 2,
        maps=task["maps"],
        spawn_mode="paired",
        seed=member["root"],
        num_envs=num_envs,
        keep_batch_size=True,
        metrics="priority",
        phase="validation",
        pass_id=validation_pass_id(task["task_id"], member["name"]),
        chunk_size=chunk_size,
        resume_from=run_dir,
        **rules,
    )


def _record_pass_sampling(
    parent: Path,
    *,
    task_id: str,
    pass_id: str,
    first: object,
    second: object,
    run_dir: Path | None,
    deployment: Mapping[str, Any] | None = None,
) -> None:
    """Save checked method facts once, before a newly created validation pass.

    The compact sidecar belongs to validation, outside the RunWriter directory.
    Existing passes without it remain unknown. Existing facts must match the
    actual frozen registration IDs and their task/pass before reuse. Registration
    hashing occurs once here at setup; no policy or learner is run.
    """
    path = parent / "sampling_facts.json"
    if run_dir is not None and not path.exists():
        if deployment is not None:
            raise ValueError("Saved deployed-team pass is missing constituent evidence")
        return
    methods = [_method_snapshot(value) for value in (first, second)]
    content = {
        "schema_version": 1,
        **(
            {"deployment": deepcopy(dict(deployment))} if deployment is not None else {}
        ),
        "task_id": task_id,
        "pass_id": pass_id,
        "system_ids": {
            team: value[1]
            for team, value in zip(("team_a", "team_b"), methods, strict=True)
        },
        "methods": {
            team: method_sampling_fact(value[0])
            for team, value in zip(("team_a", "team_b"), methods, strict=True)
        },
    }
    if path.exists():
        if _json(path) != content:
            raise ValueError(
                "Saved sampling facts differ from the task or actual frozen methods"
            )
    else:
        _publish(path, content)


def _read_pass_sampling(run_dir: Path, *, task_id: str, pass_id: str) -> Record:
    """Read method facts only when bound to the completed pass's recorded identities.

    Missing historical evidence stays unknown. Mismatched task, pass or actual
    recorded registration IDs raise ValueError. This reads small metadata only;
    it does not load actors, infer behavior from labels or alter old results.
    """
    path = run_dir.parent / "sampling_facts.json"
    if not path.exists():
        return {
            "determinism": "unknown",
            "basis": "No sampling facts were recorded for this pass",
        }
    from marl_battlegrounds.evaluation.results import load_results

    saved = _json(path)
    result = load_results(run_dir, phase="validation", pass_id=pass_id)
    entries = list(result.metadata["passes"].values())
    if (
        len(entries) != 1
        or saved.get("schema_version") != 1
        or isinstance(saved.get("schema_version"), bool)
        or saved.get("task_id") != task_id
        or saved.get("pass_id") != pass_id
        or saved.get("system_ids") != entries[0]["system_ids"]
    ):
        raise ValueError(
            "Sampling facts do not match the saved task/pass registrations"
        )
    methods = saved.get("methods")
    if not isinstance(methods, dict) or set(cast(Record, methods)) != {
        "team_a",
        "team_b",
    }:
        raise ValueError("Sampling facts must cover both recorded teams")
    return combine_sampling_facts(tuple(cast(Record, methods).values()))


def _preserve_summary_sampling(
    directory: Path, evidence: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """Keep an existing historical summary's fields and interpretation unchanged.

    New summaries, including summaries of older passes, report available facts
    or unknowns. Resuming an already published summary without this extension
    keeps its old conditional calculation; later read-only analysis can qualify
    those assumptions without rewriting the saved result.
    """
    previous = directory / "validation_summary.json"
    if previous.exists() and "sampling_evidence" not in _json(previous):
        return None
    return evidence


def validation_sampling_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    facts: Mapping[str, Mapping[str, Any]],
    scheduled_games: int,
    independent_opponents: bool,
) -> Record:
    """Describe saved validation rows with their actual per-opponent sampling facts.

    facts maps each opponent to the combined method fact for its pass. Whole
    groups use map/seed and, for independent roots, opponent identity. Fixed
    deterministic pairs contribute no random unit. This pure reporting helper
    captures no actions and is also used when checking saved decisions.
    """
    overall = combine_sampling_facts(tuple(facts.values()))
    identifiers = [
        (row["map_id"], row["seed_id"], row["opponent"], row["spawn_locations"])
        for row in rows
    ]
    units = {
        identifier: (
            None
            if facts[row["opponent"]]["determinism"] == "deterministic"
            else (row["map_id"], row["opponent"], row["seed_id"])
            if independent_opponents
            else (row["map_id"], row["seed_id"])
        )
        for identifier, row in zip(identifiers, rows, strict=True)
    }
    return summarize_sampling_evidence(
        identifiers,
        scheduled_games=scheduled_games,
        sampling_units=units,
        determinism=overall["determinism"],
        basis=overall["basis"] + "; Fixed native initial conditions and declared roots",
    )


def _summarize_panel_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: Mapping[str, Any],
    sampling: Mapping[str, Any] | None,
    facts: Mapping[str, Mapping[str, Any]],
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 19_044_001,
) -> Record:
    """Reuse shared score reduction for bare opponents or paired partner cells.

    Partner tasks keep every matched map/seed and both spawn ends in the same
    uncertainty block. Separate per-partner summaries retain the panel's usual
    independent opponent roots. These intervals concern fixed Systems only.
    """
    paired = "partners" in task
    names = [member["name"] for member in task["members"]]
    result = summarize_validation(
        rows,
        maps=task["maps"],
        opponents=names,
        seed_pairs=task["seed_pairs"],
        independent_opponents=not paired,
        actual_kills="red_zone_depth" in task,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        sampling_evidence=sampling,
    )
    if not paired:
        return result
    partners: list[Record] = []
    for index, partner in enumerate(task["partners"]):
        names = [
            member["name"]
            for member in task["members"]
            if member["partner_index"] == index
        ]
        selected = [row for row in rows if row["opponent"] in names]
        evidence = (
            validation_sampling_evidence(
                selected,
                facts={name: facts[name] for name in names},
                scheduled_games=len(task["maps"]) * len(names) * task["seed_pairs"] * 2,
                independent_opponents=True,
            )
            if sampling is not None
            else None
        )
        summary = summarize_validation(
            selected,
            maps=task["maps"],
            opponents=names,
            seed_pairs=task["seed_pairs"],
            independent_opponents=True,
            actual_kills="red_zone_depth" in task,
            bootstrap_draws=bootstrap_draws,
            bootstrap_seed=bootstrap_seed,
            sampling_evidence=evidence,
        )
        partners.append({"name": partner["name"], "label": partner["label"], **summary})
    result["partner_results"] = partners
    result["mean_kill_difference"] = sum(
        row["mean_kill_difference"] for row in partners
    ) / len(partners)
    result["uncertainty"] = (
        "Conditional game-sampling interval; partner comparisons "
        "and both spawn ends stay paired"
    )
    return result


def prepare_validation_teams(
    actor: System | Policy,
    opponents: Sequence[System | Policy | str],
    *,
    partners: Mapping[str, System | Policy] | None = None,
    learner_slots: Sequence[int] | None = None,
    maps: Sequence[int] | None = None,
    system_roster: Sequence[AgentClassName] | None = None,
    opponent_roster: Sequence[AgentClassName] | None = None,
    red_zone_depth: float = DEFAULT_TDM_RED_ZONE_DEPTH,
) -> tuple[System | Policy, ...]:
    """Build and check every deployed team before creating any run output.

    actor is the frozen learner. opponents are live methods or load references.
    partners contains bindings from freeze_validation_partners; learner_slots
    must be the slots that helper checked. Each partner fills their complement.
    None keeps the bare actor. Return teams in partner order, or just the actor.
    Maps default to 42 through 46 and rosters to canonical 5v5. These checks use
    the shared evaluation owner and actual physical roster. An invalid member,
    empty learner assignment or unsupported roster raises ValueError before any
    policy call, environment step or file write. Import references may load once.
    """
    from marl_battlegrounds.evaluation.evaluate import normalize_episode_specs
    from marl_battlegrounds.evaluation.policy_execution import team
    from marl_battlegrounds.evaluation.system_evaluation import (
        prepare_evaluation_system,
        validate_evaluation_rosters,
    )
    from marl_battlegrounds.tasks import canonical_tournament_rosters

    default_a, default_b = canonical_tournament_rosters()
    roster_a = tuple(default_a if system_roster is None else system_roster)
    roster_b = tuple(default_b if opponent_roster is None else opponent_roster)
    map_ids = tuple(VALIDATION_MAPS if maps is None else maps)
    if not map_ids or len(set(map_ids)) != len(map_ids):
        raise ValueError("Validation maps must be nonempty and distinct")
    configs = normalize_episode_specs(
        map_ids,
        len(map_ids),
        roster_a,
        roster_b,
        20,
        300,
        red_zone_depth=red_zone_depth,
    )
    actors: tuple[System | Policy, ...] = (actor,)
    if partners is not None:
        if learner_slots is None or not partners:
            raise ValueError("Partner validation needs learner_slots and partners")
        if not any(slot < len(roster_a) for slot in learner_slots):
            raise ValueError("Validation roster has no active learner slot")
        remaining = [slot for slot in range(5) if slot not in learner_slots]
        actors = tuple(
            team(actor, partner, slots=[learner_slots, remaining])
            for partner in partners.values()
        )
    elif learner_slots is not None:
        raise ValueError("learner_slots needs partners")
    prepared_opponents = tuple(
        prepare_evaluation_system(_method_snapshot(value)[0]) for value in opponents
    )
    # Every map shares one roster profile; compatibility is independent of map.
    for candidate in actors:
        first, variables_a, _ = prepare_evaluation_system(candidate)
        for second, variables_b, _ in prepared_opponents:
            validate_evaluation_rosters(
                first,
                second,
                configs[0].env_config,
                variables_a=variables_a,
                variables_b=variables_b,
            )
    return actors


def _validate_system_panel(
    checkpoint: System | Policy | str | Path,
    panel: FrozenPanel,
    *,
    output_dir: str | Path,
    purpose: str,
    seed_pairs: int,
    num_envs: int,
    chunk_size: int,
    event_callback: EventCallback | None,
    root_seed: int | None,
    red_zone_depth: float,
    maps: Sequence[int] | None = None,
    system_roster: Sequence[AgentClassName] | None = None,
    opponent_roster: Sequence[AgentClassName] | None = None,
    partners: Mapping[str, System | Policy | str] | None = None,
    learner_slots: Sequence[int] | None = None,
    partner_labels: Mapping[str, str] | None = None,
) -> Record:
    """Run retained methods in stable batches using the ordinary M8 evaluator.

    Each opponent keeps independent paired game keys. Interrupted tails remain in
    this process with inactive padding, so opaque clients are never serialized.
    Only complete durable rows feed the shared summary and selection owner. A
    QMIX or PQN-VDN actor's summary also carries method and optimizer_steps
    from _method_fields; PPO summaries are unchanged. Every game uses
    red_zone_depth (map units), recorded by task schema 4 or 5. Custom maps and
    rosters remain fixed for the whole task. The summary reports native points
    and recorded kills; live candidates keep their ordinary System registration
    without invented learner metadata. Roster checks run before any output.
    """
    import jax

    from marl_battlegrounds.evaluation.evaluate import evaluate
    from marl_battlegrounds.training.checkpoints import load_system

    _integer(num_envs, "num_envs", minimum=1)
    _integer(chunk_size, "chunk_size", minimum=1)
    if jax.default_backend() != "cpu" and num_envs != 32:
        raise ValueError("Training validation on GPU requires num_envs=32")
    from marl_battlegrounds.evaluation.policy_execution import Policy, System, policy

    # Keep artifact metadata for saved learners. Built-ins, factories and live
    # objects use the same registration as ordinary evaluation instead.
    live = isinstance(checkpoint, (System, Policy))
    if isinstance(checkpoint, str):
        try:
            policy(checkpoint)
            live = True
        except ValueError:
            live = not Path(checkpoint).is_dir() and ":" in checkpoint
    identity: Record = {}
    registration = None
    if live:
        actor, _, registration, _ = _method_snapshot(checkpoint)
    else:
        saved = cast(str | Path, checkpoint)
        identity = _artifact(saved)
        actor = load_system(saved)
    deployment = None
    bindings: dict[str, Any] | None = None
    if partners is not None:
        if learner_slots is None:
            raise ValueError("Partner validation needs learner_slots")
        bindings, deployment = freeze_validation_partners(
            partners, learner_slots=learner_slots, partner_labels=partner_labels
        )
    elif learner_slots is not None or partner_labels is not None:
        raise ValueError("learner_slots and partner_labels need partners")
    directory = Path(output_dir).resolve()
    task = panel_task_description(
        checkpoint_id=identity.get("checkpoint_id"),
        actor_digest=identity.get("actor_digest"),
        env_steps=identity.get("env_steps"),
        panel=panel,
        purpose=purpose,
        seed_pairs=seed_pairs,
        root_seed=root_seed,
        red_zone_depth=red_zone_depth,
        candidate=registration,
        maps=maps,
        system_roster=system_roster,
        opponent_roster=opponent_roster,
        deployment=deployment,
    )
    actors = prepare_validation_teams(
        actor,
        panel.methods,
        partners=bindings,
        learner_slots=learner_slots,
        maps=task["maps"],
        system_roster=task.get("system_roster"),
        opponent_roster=task.get("opponent_roster"),
        red_zone_depth=red_zone_depth,
    )
    task = _task(directory, task, event_callback)
    map_ids = task["maps"]
    rows: list[Record] = []
    paths: list[str] = []
    sampling_facts: dict[str, Record] = {}
    for index, member in enumerate(task["members"]):
        actor = actors[cast(int, member.get("partner_index", 0))]
        opponent = panel.methods[member.get("opponent_index", index)]
        pass_id = validation_pass_id(task["task_id"], member["name"])
        parent = _cell_directory(directory, task, index)
        run_dir = _saved_run(parent, pass_id)
        if run_dir is not None:
            _verify_panel_pass(
                run_dir,
                actor,
                opponent,
                task=task,
                member_index=index,
                num_envs=num_envs,
                chunk_size=chunk_size,
            )
        _record_pass_sampling(
            parent,
            task_id=task["task_id"],
            pass_id=pass_id,
            first=actor,
            second=opponent,
            run_dir=run_dir,
            deployment=_deployment_cell(task, index),
        )
        pending = _pending(
            run_dir, pass_id=pass_id, total=len(map_ids) * seed_pairs * 2
        )
        if pending:
            if event_callback is not None:
                event_callback(
                    {
                        "event": "evaluation_segment",
                        "task_id": task["task_id"],
                        "pass_id": pass_id,
                        "backend": jax.default_backend(),
                        "pending_games": pending,
                        "num_envs": num_envs,
                    }
                )
            started = time.monotonic()
            options: Record = (
                {"output_dir": parent} if run_dir is None else {"resume_from": run_dir}
            )
            result = evaluate(
                actor,
                opponent,
                num_episodes=len(map_ids) * seed_pairs * 2,
                maps=map_ids,
                system_roster=task.get("system_roster"),
                opponent_roster=task.get("opponent_roster"),
                spawn_mode="paired",
                seed=task["members"][index]["root"],
                num_envs=num_envs,
                keep_batch_size=True,
                metrics="priority",
                phase="validation",
                pass_id=pass_id,
                chunk_size=chunk_size,
                red_zone_depth=red_zone_depth,
                **options,
            )
            run_dir = result.run_dir
            if event_callback is not None:
                event_callback(
                    {
                        "event": "evaluation_segment_complete",
                        "task_id": task["task_id"],
                        "pass_id": pass_id,
                        "backend": jax.default_backend(),
                        "seconds": time.monotonic() - started,
                    }
                )
        assert run_dir is not None
        paths.append(str(run_dir))
        sampling_facts[member["name"]] = _read_pass_sampling(
            run_dir,
            task_id=task["task_id"],
            pass_id=pass_id,
        )
        rows.extend(
            _rows(run_dir, pass_id=pass_id, opponent=member["name"], kills=True)
        )
    result = {
        **task,
        **_summarize_panel_rows(
            rows,
            task=task,
            facts=sampling_facts,
            sampling=_preserve_summary_sampling(
                directory,
                validation_sampling_evidence(
                    rows,
                    facts=sampling_facts,
                    scheduled_games=len(map_ids)
                    * len(task["members"])
                    * seed_pairs
                    * 2,
                    independent_opponents=deployment is None,
                ),
            ),
        ),
        "pass_paths": paths,
        **_method_fields(identity),
    }
    summary_path = directory / "validation_summary.json"
    if not summary_path.exists() or _json(summary_path) != result:
        _publish(summary_path, result)
    if event_callback is not None:
        event_callback(
            {
                "event": "validation_complete",
                "task_id": task["task_id"],
                "result": deepcopy(result),
            }
        )
    return result


def validation_task_description(
    *,
    checkpoint_id: str,
    actor_digest: str,
    env_steps: int,
    panel_digest: str,
    purpose: str,
    seed_pairs: int,
    members: Sequence[tuple[str, str]],
    root_seed: int | None = None,
    red_zone_depth: float | None = None,
) -> Record:
    """Describe one exact fixed-map validation task without files or numerical work.

    Shared constants supply the maps (42-46) and the purpose roots.

    Parameters
    ----------
    checkpoint_id : str
        Nonempty ID of the originating learner boundary.
    actor_digest : str
        Nonempty identity of the actor's frozen weights.
    env_steps : int
        Real training transitions behind the actor; a plain integer, 0 or more.
    panel_digest : str
        Nonempty identity of the frozen panel (or of the Random diagnostic).
    purpose : str
        "routine", "initialization", "confirmation" or "random".
    seed_pairs : int
        Positive number of paired seeds per map.
    members : sequence of (str, str)
        The panel's ordered (name, weight digest) pairs. It must not be empty;
        names must be distinct and every name and digest nonempty.
    root_seed : int or None, default=None
        None keeps the purpose's fixed root. A plain integer in [0, 2**32 - 1]
        is allowed for purpose "random" only; changing it changes the actual
        game keys and the task ID.
    red_zone_depth : float or None, default=None
        Red Zone depth in map units that the task's games use. None builds the
        schema 1 layout saved before the Red Zone rule, so existing calls keep
        exactly the same task ID. A float builds schema 3, which records
        "red_zone_depth"; tasks under different depths, 0.0 included, have
        different IDs.

    Returns
    -------
    dict
        A new JSON-ready description ending with its immutable task_id, the
        hash of every other field.

    Raises
    ------
    TypeError
        red_zone_depth is neither None nor a Python float.
    ValueError
        red_zone_depth breaks Core's scalar rules (not finite, negative, -0.0,
        or a positive value that is not a normal float32 number); a count is
        invalid; purpose is unknown; root_seed is given for a purpose other
        than "random" or is outside uint32; an identity is not a nonempty
        string; or members is empty or repeats a name.

    Notes
    -----
    Host-only. This helper opens nothing. The caller verifies artifact
    contents and the declared panel. Core checks the depth against each map's
    width later, when a game's config is built.
    """
    depth = _task_depth(red_zone_depth)
    _integer(env_steps, "env_steps")
    _integer(seed_pairs, "seed_pairs", minimum=1)
    if purpose not in ("routine", "initialization", "confirmation", "random"):
        raise ValueError("Unknown fixed-map validation purpose")
    if root_seed is not None and purpose != "random":
        raise ValueError("An explicit root_seed is supported for Random checks only")
    root = _ROOTS[purpose] if root_seed is None else _root_seed(root_seed)
    if any(
        not isinstance(cast(object, value), str) or not value
        for value in (checkpoint_id, actor_digest, panel_digest)
    ):
        raise ValueError("Validation task identities must be nonempty strings")
    if (
        not members
        or any(
            not isinstance(cast(object, name), str)
            or not name
            or not isinstance(cast(object, digest), str)
            or not digest
            for name, digest in members
        )
        or len({name for name, _ in members}) != len(members)
    ):
        raise ValueError("Validation members need distinct names and nonempty digests")
    result: Record = {
        "schema_version": 1 if depth is None else 3,
        "checkpoint_id": checkpoint_id,
        "actor_digest": actor_digest,
        "env_steps": env_steps,
        "panel_digest": panel_digest,
        "purpose": purpose,
        "seed_pairs": seed_pairs,
        "maps": list(VALIDATION_MAPS),
        "root": root,
        "members": [{"name": name, "actor_digest": digest} for name, digest in members],
    }
    if depth is not None:
        result["red_zone_depth"] = depth
    return {**result, "task_id": _digest(result)}


def _qualify_random(identity: Mapping[str, Any], result: Mapping[str, Any]) -> Record:
    """Check the complete declared Random diagnostic and retain portable evidence.

    identity names a checked actor and originating learner checkpoint. result is
    the summary returned by validate_random for exactly 100 games. Verify finite
    W/D/L totals, all five cells, declared seed roots and the immutable task hash.
    The task layout follows the result's own red_zone_depth (none for evidence
    saved before the Red Zone rule). Return its JSON-ready scientific fields
    without pass storage paths. Missing, conflicting, nonfinite or unsuccessful
    evidence raises ValueError.
    """
    depth = result.get("red_zone_depth")
    if depth is not None and type(depth) is not float:
        raise ValueError("Panel usefulness needs the exact complete Random task")
    expected = validation_task_description(
        checkpoint_id=identity["checkpoint_id"],
        actor_digest=identity["actor_digest"],
        env_steps=identity["env_steps"],
        panel_digest="random-diagnostic-v1",
        purpose="random",
        seed_pairs=10,
        members=(("Random", "builtin-random"),),
        red_zone_depth=depth,
    )
    if (
        any(result.get(key) != value for key, value in expected.items())
        or result.get("complete") is not True
        or result.get("games") != 100
        or result.get("independent_blocks") != 50
        or result.get("bootstrap_draws") != 2000
        or result.get("bootstrap_seed") != 19_044_001
    ):
        raise ValueError("Panel usefulness needs the exact complete Random task")

    def finite(value: object, name: str, low: float, high: float) -> float:
        """Require a finite real summary measurement inside its declared bounds."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Panel usefulness {name} is missing or nonnumerical")
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Panel usefulness {name} is outside its finite bounds")
        return float(value)

    score = finite(result.get("score"), "score", 0, 1)
    lower = finite(result.get("ci_low"), "lower interval", 0, 1)
    upper = finite(result.get("ci_high"), "upper interval", 0, 1)
    if lower > upper or score <= 0.5:
        raise ValueError("The fixed early panel failed its usefulness gate")
    raw: object = result.get("cells")
    if (
        not isinstance(raw, list)
        or len(cast(list[object], raw)) != 5
        or any(not isinstance(cell, dict) for cell in cast(list[object], raw))
    ):
        raise ValueError("Panel usefulness needs all five map cells")
    cells = cast(list[Record], raw)
    if sorted(cell.get("map_id", -1) for cell in cells) != list(VALIDATION_MAPS):
        raise ValueError("Panel usefulness cells must cover exactly maps 42 through 46")
    scores: list[float] = []
    good_maps = 0
    for cell in cells:
        if (
            cell.get("opponent") != "Random"
            or cell.get("games") != 20
            or cell.get("seed_pairs") != 10
        ):
            raise ValueError("Panel usefulness needs twenty games per Random map cell")
        wins, draws, losses = (
            _integer(cell.get(key), key) for key in ("wins", "draws", "losses")
        )
        measured = finite(cell.get("score"), "cell score", 0, 1)
        if wins + draws + losses != 20 or measured != (wins + 0.5 * draws) / 20:
            raise ValueError("Panel usefulness score disagrees with its W/D/L counts")
        scores.append(measured)
        good_maps += (
            finite(cell.get("mean_team_a_score"), "Team A score", 0, float("inf")) > 0
        )
        finite(cell.get("mean_team_b_score"), "Team B score", 0, float("inf"))
        finite(cell.get("mean_episode_length"), "game length", 1, 300)
    if good_maps < 2 or not math.isclose(score, sum(scores) / 5, abs_tol=1e-12):
        raise ValueError("The fixed early panel failed its usefulness gate")
    evidence = {key: value for key, value in result.items() if key != "pass_paths"}
    _digest(evidence)
    return evidence


def validation_pass_id(task_id: str, opponent: str) -> str:
    """Identify one fixed-panel M8 pass by its task hash and opponent member name.

    Both arguments must be nonempty strings. Return the deterministic SHA-256
    identity used by execution and saved-host validation. No files or games run.
    Invalid arguments raise ValueError; changing either field changes the input
    to the hash. Slot diagnostics use their separate four-condition task protocol.
    """
    if any(
        not isinstance(cast(object, value), str) or not value
        for value in (task_id, opponent)
    ):
        raise ValueError("Validation pass needs its task identity and opponent name")
    return _digest({"task_id": task_id, "opponent": opponent})


def _panel_digest(content: Mapping[str, Any]) -> str:
    """Hash scientific panel contents while excluding digest and storage paths."""
    return _digest(
        {
            key: [
                {
                    field: value
                    for field, value in member.items()
                    if field not in ("path", "reference", "export_relative")
                }
                for member in cast(Sequence[Mapping[str, Any]], value)
            ]
            if key == "members"
            else value
            for key, value in content.items()
            if key != "panel_digest"
        }
    )


def load_panel(
    path: str | Path,
    *,
    bindings: Sequence[Any] | None = None,
    red_zone_depth: float | None = None,
) -> FrozenPanel:
    """Load and verify a frozen panel without applying a method.

    Parameters
    ----------
    path : str or Path
        panel.json or its directory. Export paths in packaged manifests may be
        relative to the panel directory; historical paths keep their old rule.
    bindings : sequence or None, default=None
        The existing new-panel methods in saved order; each normal M8 identity
        must match. None reloads built-in/actor-folder/factory references once.
        Live-only members require bindings. Historical panels take none.
    red_zone_depth : float or None, default=None
        The Red Zone depth (map units) the caller will validate under. When
        given, it must be a valid task depth. Actors may load regardless of the
        ranking's depth. That ranking keeps its original conditions and identity;
        validation records the new task's depth separately. None declares no new
        task depth. Loading grants no permission to reuse games under new rules.

    Returns
    -------
    FrozenPanel
        Retains the checked methods for repeated validation.

    Raises
    ------
    ValueError
        Changed identities, malformed content, missing bindings or an invalid
        new task depth. System-panel checks run before any method is loaded.

    Notes
    -----
    Import, factory and file errors keep their cause. No client is serialized
    and no file is written. This is the one owner of panel admission.
    """
    target = Path(path).resolve()
    if target.is_dir():
        target /= "panel.json"
    content = _json(target)
    if content.get("schema_version") == 2:
        return _load_system_panel(target, content, bindings, red_zone_depth)
    if bindings is not None:
        raise ValueError("Historical panels use their saved actor paths")
    raw_entries: object = content.get("members", [])
    if not isinstance(raw_entries, list) or any(
        not isinstance(row, dict) for row in cast(list[object], raw_entries)
    ):
        raise ValueError("Panel members must be objects")
    if (
        content.get("schema_version") != 1
        or content.get("provisional") is not True
        or not isinstance(content.get("qualified"), bool)
        or content.get("inference") != "Sampled masked categorical"
        or content.get("panel_digest") != _panel_digest(content)
    ):
        raise ValueError("Invalid frozen-panel schema or content digest")
    entries = cast(list[Record], raw_entries)
    if [row.get("name") for row in entries] != [
        "Halfway",
        "Final",
    ]:
        raise ValueError("The early panel must contain Halfway then Final")
    members: list[PanelMember] = []
    for row in entries:
        member_path = Path(row["path"])
        if not member_path.is_absolute():
            member_path = target.parent / member_path
        identity = _artifact(member_path)
        if any(
            identity.get(key) != row.get(key)
            for key in (
                "actor_digest",
                "checkpoint_id",
                "run_id",
                "seed",
                "env_steps",
                "schemas",
            )
        ):
            raise ValueError("Panel member no longer matches its frozen identity")
        members.append(
            PanelMember(
                row["name"],
                member_path.resolve(),
                row["actor_digest"],
                row["checkpoint_id"],
                row["run_id"],
                row["seed"],
                row["env_steps"],
            )
        )
    if (
        members[0].run_id != members[1].run_id
        or members[0].seed != members[1].seed
        or not 0 < members[0].env_steps < members[1].env_steps
    ):
        raise ValueError("Panel development lineage or checkpoint order is invalid")
    if content["qualified"]:
        evidence = content.get("qualification_evidence", [])
        if not isinstance(evidence, list) or len(cast(list[object], evidence)) != 2:
            raise ValueError("Qualified panel is missing its fixed Random evidence")
        for member, result in zip(entries, cast(list[object], evidence), strict=True):
            if not isinstance(result, dict):
                raise ValueError("Panel qualification evidence must contain objects")
            _qualify_random(member, cast(Record, result))
    return FrozenPanel(
        target, content["panel_digest"], tuple(members), content["qualified"]
    )


def _task(
    directory: Path, content: Record, event_callback: EventCallback | None
) -> Record:
    """Persist exact task identity before running any pass; reject conflicting reuse."""
    scientific = {key: value for key, value in content.items() if key != "task_id"}
    result = {**scientific, "task_id": _digest(scientific)}
    if content.get("task_id", result["task_id"]) != result["task_id"]:
        raise ValueError("Validation task description has an inconsistent identity")
    path = directory / "task.json"
    if path.exists():
        if _json(path) != result:
            raise ValueError(
                "Validation task path already contains different scientific conditions"
            )
    else:
        _publish(path, result)
        if event_callback is not None:
            event_callback({"event": "validation_task", **deepcopy(result)})
    return result


def _pass_directory(
    directory: Path, index: int, name: str, *, kind: str = "opponent"
) -> Path:
    """Find an old pass folder or choose its clear new snake_case name.

    directory is the owning task; index is its stable member order and name is
    the saved panel label. Return a path without creating it. The index keeps
    equal or similarly shortened names separate. Existing historical folders
    retain their original paths. Linked, conflicting or non-directory entries
    raise ValueError instead of being followed or replaced.
    """
    label = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40].rstrip("_")
    current = directory / f"{kind}_{index:02d}_{label or 'unnamed'}"
    legacy = directory / f"{kind}-{index}"
    for path in (current, legacy):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError("Validation opponent folder must be a plain directory")
    if current.exists() and legacy.exists():
        raise ValueError("Validation has two folders for the same opponent")
    return legacy if legacy.exists() else current


def _saved_run(parent: Path, pass_id: str) -> Path | None:
    """Find the sole M8 run in a task-owned pass folder, rejecting ambiguity.

    This folder belongs to exactly one persisted task. It is never a general
    newest-run lookup. A malformed or different pass is rejected before recovery.
    """
    if not parent.exists():
        return None
    candidates = sorted(path.parent for path in parent.glob("*/run_details.json"))
    if len(candidates) > 1:
        raise ValueError("A validation pass folder contains multiple M8 runs")
    if not candidates:
        if any(parent.iterdir()):
            raise ValueError(
                "A validation pass folder contains incomplete unknown output"
            )
        return None
    from marl_battlegrounds.evaluation.results import load_results

    load_results(candidates[0], phase="validation", pass_id=pass_id)
    return candidates[0]


def _pending(run_dir: Path | None, *, pass_id: str, total: int) -> int:
    """Read exact durable completion counts without opening/recovering a writer."""
    if run_dir is None:
        return total
    from marl_battlegrounds.evaluation.results import load_results

    result = load_results(run_dir, phase="validation", pass_id=pass_id)
    completed = sum(len(batch["episode_id"]) for batch in result.iter_table("episodes"))
    if completed > total:
        raise ValueError("Saved evaluation exceeds its task budget")
    return total - completed


def evaluation_backend(pending: int, backend: str) -> str:
    """Choose read-only, CPU or existing GPU work from pending game count.

    Zero returns 'none'. A CPU process stays on CPU. Any non-CPU backend needs at
    least 32 pending games; smaller tails return 'cpu'. Counts must be nonnegative
    integers and backend must be nonempty. This pure host decision does no I/O.
    """
    _integer(pending, "pending")
    if not backend:
        raise ValueError("backend must name the current execution backend")
    return (
        "none"
        if pending == 0
        else "cpu"
        if backend == "cpu" or pending < 32
        else backend
    )


def _execute_request(request: Mapping[str, Any]) -> Path:
    """Load checked frozen actors and execute one exact M8 pass in this process.

    request comes from a persisted task or its CPU worker file. It contains actor
    paths/digests, pass/schedule fields, the task's red_zone_depth (map units)
    and a task-owned output parent. Panel and Random passes give the depth to
    evaluate; slot passes build their explicit schedule at that depth. No learner
    state is accepted. M8 owns scientific resume validation and all writer
    recovery, including refusing a saved pass recorded under another depth.
    """
    from marl_battlegrounds.evaluation.evaluate import evaluate, evaluate_episodes
    from marl_battlegrounds.training.checkpoints import load_system

    actor_identity = _artifact(request["actor"])
    if actor_identity["actor_digest"] != request["actor_digest"]:
        raise ValueError("The evaluated actor differs from the persisted task")
    actor = load_system(request["actor"])
    if request["opponent"] == "random":
        opponent: Any = "random"
    else:
        identity = _artifact(request["opponent"])
        if identity["actor_digest"] != request["opponent_digest"]:
            raise ValueError("The evaluated opponent differs from the persisted task")
        opponent = load_system(request["opponent"])
    parent = Path(request["output_parent"])
    run_dir = _saved_run(parent, request["pass_id"])
    reverse = request["kind"] == "slot" and int(request["focal_team"]) == 1
    _record_pass_sampling(
        parent,
        task_id=request["task_id"],
        pass_id=request["pass_id"],
        first=opponent if reverse else actor,
        second=actor if reverse else opponent,
        run_dir=run_dir,
    )
    options: Record = {
        "seed": request["root"],
        "num_envs": request["num_envs"],
        "metrics": "priority",
        "phase": "validation",
        "pass_id": request["pass_id"],
        "chunk_size": request["chunk_size"],
    }
    options["output_dir" if run_dir is None else "resume_from"] = (
        parent if run_dir is None else run_dir
    )
    if request["kind"] == "slot":
        schedule = make_slot_diagnostic_schedule(
            maps=request["maps"],
            seed_blocks=request["seed_pairs"],
            red_zone_depth=request["red_zone_depth"],
        )
        team = int(request["focal_team"])
        episodes = schedule[team]
        result = evaluate_episodes(
            actor if team == 0 else opponent,
            opponent if team == 0 else actor,
            episodes,
            **options,
        )
    else:
        result = evaluate(
            actor,
            opponent,
            num_episodes=request["total_games"],
            maps=request["maps"],
            spawn_mode="paired",
            red_zone_depth=request["red_zone_depth"],
            **options,
        )
    assert result.run_dir is not None
    return result.run_dir


def _run_pass(request: Record, *, event_callback: EventCallback | None) -> Path:
    """Run/resume one pass, delegating a small GPU tail to a CPU subprocess."""
    import jax

    parent = Path(request["output_parent"])
    run_dir = _saved_run(parent, request["pass_id"])
    pending = _pending(
        run_dir, pass_id=request["pass_id"], total=request["total_games"]
    )
    current = jax.default_backend()
    backend = evaluation_backend(pending, current)
    if backend == "none":
        assert run_dir is not None
        return run_dir
    if current != "cpu" and request["num_envs"] != 32:
        raise ValueError("Training validation on GPU requires num_envs=32")
    event: Record = {
        "event": "evaluation_segment",
        "task_id": request["task_id"],
        "pass_id": request["pass_id"],
        "backend": backend,
        "pending_games": pending,
        "num_envs": min(request["num_envs"], pending),
        "previous_runtime": [
            entry.get("details", {}).get("runtime_provenance")
            for entry in _json(run_dir / "run_details.json").get("passes", {}).values()
        ]
        if run_dir is not None
        else None,
    }
    if event_callback is not None:
        event_callback(event)
    started = time.monotonic()
    if backend == "cpu" and current != "cpu":
        worker_file = parent.parent / f"worker-{request['pass_id']}.json"
        _publish(worker_file, request)
        environment = dict(os.environ, JAX_PLATFORMS="cpu")
        subprocess.run(
            [sys.executable, "-m", __name__, "--worker", str(worker_file)],
            env=environment,
            check=True,
        )
        run_dir = _saved_run(parent, request["pass_id"])
        if run_dir is None:
            raise RuntimeError("CPU worker returned without its saved pass")
    else:
        run_dir = _execute_request(request)
    if event_callback is not None:
        event_callback(
            {
                "event": "evaluation_segment_complete",
                "task_id": request["task_id"],
                "pass_id": request["pass_id"],
                "backend": backend,
                "seconds": time.monotonic() - started,
                "run_dir": str(run_dir),
            }
        )
    return run_dir


def _rows(
    run_dir: Path,
    *,
    pass_id: str,
    opponent: str,
    focal_team: int | None = None,
    kills: bool = False,
) -> list[Record]:
    """Join M8's complete saved rows to exact schedule conditions after resume.

    Parameters
    ----------
    run_dir : Path
        One complete M8 validation pass.
    pass_id : str
        Its exact pass ID.
    opponent : str
        Label added to every row.
    focal_team : int or None, default=None
        Slot diagnostics add the focal team (0 or 1); None adds nothing.
    kills : bool, default=False
        True also joins each game's recorded team_a_kills and team_b_kills
        from the pass's priority metrics (a missing value becomes None). Use
        it for tasks with a Red Zone depth, where points and kills differ.

    Returns
    -------
    list of dict
        One plain row per completed game (episode columns, score columns in
        points, opponent and spawn_locations, and the optional fields above).

    Raises
    ------
    ValueError
        The pass is incomplete, or kills=True and a game has no priority
        metrics row.

    Notes
    -----
    Reads saved files only; opens no writer and runs no game.
    """
    from marl_battlegrounds.evaluation.results import load_results

    result = load_results(run_dir, phase="validation", pass_id=pass_id)
    if result.status != "complete":
        raise ValueError("Validation pass is incomplete")
    entries = result.metadata["passes"]
    entry = next(iter(entries.values()))
    declared = entry["episodes"]
    recorded: dict[int, tuple[float | None, ...]] = {}
    if kills:
        for batch in result.iter_table("priority_metrics"):
            for index in range(len(batch["episode_id"])):
                values = (float(batch[name][index]) for name in _KILL_COLUMNS)
                recorded[int(batch["episode_id"][index])] = tuple(
                    value if math.isfinite(value) else None for value in values
                )
    rows: list[Record] = []
    for batch in result.iter_table("episodes"):
        for index in range(len(batch["episode_id"])):
            row = {
                key: values[index].item()
                if hasattr(values[index], "item")
                else values[index]
                for key, values in batch.items()
            }
            schedule = declared[str(row["episode_id"])]
            row.update(opponent=opponent, spawn_locations=schedule["spawn_locations"])
            if focal_team is not None:
                row["focal_team"] = focal_team
            if kills:
                if row["episode_id"] not in recorded:
                    raise ValueError("Validation game has no recorded kills")
                row.update(zip(_KILL_COLUMNS, recorded[row["episode_id"]], strict=True))
            rows.append(row)
    return rows


def _validate(
    checkpoint: str | Path,
    members: Sequence[tuple[str, str, str]],
    *,
    panel_digest: str,
    output_dir: str | Path,
    purpose: str,
    seed_pairs: int,
    num_envs: int,
    chunk_size: int,
    event_callback: EventCallback | None,
    red_zone_depth: float,
    root_seed: int | None = None,
) -> Record:
    """Own one exact checkpoint/panel task and reduce its complete saved M8 rows.

    A QMIX or PQN-VDN actor's summary also carries method and optimizer_steps
    from _method_fields; PPO summaries are unchanged. Every game uses
    red_zone_depth (map units), which the schema 3 task records; the summary
    reports points and recorded kills.
    """
    _integer(seed_pairs, "seed_pairs", minimum=1)
    _integer(num_envs, "num_envs", minimum=1)
    _integer(chunk_size, "chunk_size", minimum=1)
    if root_seed is not None:
        _root_seed(root_seed)
    identity = _artifact(checkpoint)
    directory = Path(output_dir).resolve()
    task = _task(
        directory,
        validation_task_description(
            checkpoint_id=identity["checkpoint_id"],
            actor_digest=identity["actor_digest"],
            env_steps=identity["env_steps"],
            panel_digest=panel_digest,
            purpose=purpose,
            seed_pairs=seed_pairs,
            members=tuple((name, digest) for name, _, digest in members),
            root_seed=root_seed,
            red_zone_depth=red_zone_depth,
        ),
        event_callback,
    )
    rows: list[Record] = []
    paths: list[str] = []
    sampling_facts: dict[str, Record] = {}
    for index, (name, opponent, digest) in enumerate(members):
        pass_id = validation_pass_id(task["task_id"], name)
        request: Record = {
            "kind": "panel",
            "task_id": task["task_id"],
            "actor": str(Path(checkpoint).resolve()),
            "actor_digest": identity["actor_digest"],
            "opponent": opponent,
            "opponent_digest": digest,
            "output_parent": str(_pass_directory(directory, index, name)),
            "pass_id": pass_id,
            "maps": list(VALIDATION_MAPS),
            "seed_pairs": seed_pairs,
            "total_games": len(VALIDATION_MAPS) * seed_pairs * 2,
            "root": task["root"],
            "num_envs": num_envs,
            "chunk_size": chunk_size,
            "red_zone_depth": task["red_zone_depth"],
        }
        path = _run_pass(request, event_callback=event_callback)
        paths.append(str(path))
        sampling_facts[name] = _read_pass_sampling(
            path,
            task_id=task["task_id"],
            pass_id=pass_id,
        )
        rows.extend(_rows(path, pass_id=pass_id, opponent=name, kills=True))
    result = {
        **task,
        **summarize_validation(
            rows,
            maps=VALIDATION_MAPS,
            opponents=[name for name, _, _ in members],
            seed_pairs=seed_pairs,
            actual_kills=True,
            sampling_evidence=_preserve_summary_sampling(
                directory,
                validation_sampling_evidence(
                    rows,
                    facts=sampling_facts,
                    scheduled_games=len(VALIDATION_MAPS)
                    * len(members)
                    * seed_pairs
                    * 2,
                    independent_opponents=False,
                ),
            ),
        ),
        "pass_paths": paths,
        **_method_fields(identity),
    }
    summary_path = directory / "validation_summary.json"
    if not summary_path.exists() or _json(summary_path) != result:
        _publish(summary_path, result)
    if event_callback is not None:
        event_callback(
            {
                "event": "validation_complete",
                "task_id": task["task_id"],
                "result": result,
            }
        )
    return result


def validate_checkpoint(
    checkpoint: System | Policy | str | Path,
    panel: FrozenPanel | str | Path,
    *,
    output_dir: str | Path,
    purpose: str = "routine",
    seed_pairs: int | None = None,
    num_envs: int = 32,
    chunk_size: int = 128,
    event_callback: EventCallback | None = None,
    root_seed: int | None = None,
    red_zone_depth: float = DEFAULT_TDM_RED_ZONE_DEPTH,
    maps: Sequence[int] | None = None,
    system_roster: Sequence[AgentClassName] | None = None,
    opponent_roster: Sequence[AgentClassName] | None = None,
    partners: Mapping[str, System | Policy | str] | None = None,
    learner_slots: Sequence[int] | None = None,
    partner_labels: Mapping[str, str] | None = None,
) -> Record:
    """Validate a saved actor or any live System through the shared evaluator.

    Parameters
    ----------
    checkpoint : System, Policy, str or Path
        A live method, built-in name, module:function factory, actor export or
        complete learner checkpoint. Numerical values are frozen for this call;
        opaque clients remain in this process. Historical panels accept only
        saved artifacts. Live results identify the exact registered System;
        they do not claim a training step or built-in learner checkpoint.
    maps : sequence of int or None, default=None
        Distinct installed map IDs. None uses development maps 42 through 46.
    system_roster, opponent_roster : sequence of class names or None, default=None
        Physical slot order, one to five members, including repeated classes.
        None uses the canonical roster. Custom conditions require a panel made
        with opponents= and are saved in the immutable task identity.
    partners : mapping or None, default=None
        Named frozen teammates, each a System, Policy or normal method reference.
        None evaluates the candidate alone. Otherwise each partner fills the
        complement of learner_slots in a separate ordinary team() deployment.
        Every team faces the same panel, maps and seed pairs. Saved candidates
        retain their own checkpoint identity for selection.
    learner_slots : sequence of int or None, default=None
        One to four distinct Team A slots in 0 through 4, required with partners.
        At least one must be active in system_roster. Every decision keeps its
        original input and opaque memory boundaries through team().
    partner_labels : mapping or None, default=None
        Optional name-to-label values: familiar, held_out or unknown. Missing
        labels mean unknown. These labels report the caller's declared exposure;
        validation never infers it from names or training records.
    panel : FrozenPanel, str or Path
        The frozen panel manifest path, or a loaded FrozenPanel. A loaded panel
        keeps its live clients and avoids reopening factories. Identities are
        rechecked before any work.
    output_dir : str or Path
        Folder owned by one immutable task. Repeat the same call to resume it.
    purpose : str, default="routine"
        "routine", "initialization", "confirmation", or "assessment" for new
        panels.
    seed_pairs : int or None, default=None
        Positive paired seeds per map. None means 10, or 50 for confirmation.
        Each pair means two games per map and opponent, with exchanged spawn
        ends. Team A always remains the candidate.
    num_envs : int, default=32
        Positive number of games run side by side; GPU requires 32. New panels
        keep that batch on all tails with inactive padding. Historical small
        GPU tails keep their CPU recovery route.
    chunk_size : int, default=128
        Positive number of ticks per recording block.
    event_callback : callable or None, default=None
        Receives task, execution and completion records; None means no calls.
    root_seed : int or None, default=None
        None uses the panel's root for purpose. A plain integer in
        [0, 2**32 - 1] replaces a new panel's purpose root and changes the task
        ID. Assessment requires this explicit fresh root. Historical panels
        cannot change their roots.
    red_zone_depth : float, default=DEFAULT_TDM_RED_ZONE_DEPTH (5.0)
        Team Deathmatch Red Zone depth in map units for every game; 0.0 keeps
        one point per death. To score an actor under the rule it trained with,
        pass that depth, for example 0.0 for the pinned MAPPO search and
        screen models. The task records it (schema 4 for new panels, 3 for
        historical panels), so an output_dir saved under another depth, or
        before the rule, is refused before any game. A panel chosen from a
        ranking keeps its original depth; its actors may play at this task depth.

    Returns
    -------
    dict
        Complete native score, paired uncertainty, per-cell results and saved
        M8 paths. Scores are points; cells also carry mean_team_a_kills and
        mean_team_b_kills from recorded kills. New panels also report
        mean_kill_difference as a descriptive column; new checkpoint selection
        uses only mean point margin. A QMIX or PQN-VDN actor's summary carries
        its method and its
        optimizer_steps. Partner tasks add partner_results with separate native
        score summaries and declared labels. Their overall intervals keep every
        partner comparison and both spawn ends in the same map/seed block.

    Raises
    ------
    TypeError
        red_zone_depth is not a Python float.
    ValueError
        red_zone_depth breaks Core's scalar rules (not finite, negative, -0.0,
        or a positive value that is not a normal float32 number) or is wider
        than a validation map; other
        invalid settings; or identities that conflict with the saved task.
        File and method failures keep their own cause.

    Notes
    -----
    Incomplete games never select a checkpoint. The call waits for evaluation
    to finish on the active backend. It never updates a learner or actor.
    Every installed map is 20.0 map units wide. A wider depth raises
    ValueError before any file is written, so the same output_dir can be
    reused with a corrected depth.
    """
    depth = _checked_depth(red_zone_depth)
    frozen = (
        load_panel(
            panel.path,
            bindings=panel.methods if panel.schema_version == 2 else None,
            red_zone_depth=depth,
        )
        if isinstance(panel, FrozenPanel)
        else load_panel(panel, red_zone_depth=depth)
    )
    pairs = (
        (50 if purpose == "confirmation" else 10) if seed_pairs is None else seed_pairs
    )
    if frozen.schema_version == 2:
        return _validate_system_panel(
            checkpoint,
            frozen,
            output_dir=output_dir,
            purpose=purpose,
            seed_pairs=pairs,
            num_envs=num_envs,
            chunk_size=chunk_size,
            event_callback=event_callback,
            root_seed=root_seed,
            red_zone_depth=depth,
            maps=maps,
            system_roster=system_roster,
            opponent_roster=opponent_roster,
            partners=partners,
            learner_slots=learner_slots,
            partner_labels=partner_labels,
        )
    if any(
        value is not None
        for value in (
            maps,
            system_roster,
            opponent_roster,
            partners,
            learner_slots,
            partner_labels,
        )
    ):
        raise ValueError("Custom validation conditions need an opponents panel")
    if not isinstance(checkpoint, (str, Path)):
        raise ValueError("Live candidates need a panel made with opponents=")
    if root_seed is not None:
        raise ValueError("Historical panels retain their original roots")
    return _validate(
        checkpoint,
        [
            (member.name, str(member.path), member.actor_digest)
            for member in frozen.members
        ],
        panel_digest=frozen.digest,
        output_dir=output_dir,
        purpose=purpose,
        seed_pairs=pairs,
        num_envs=num_envs,
        chunk_size=chunk_size,
        event_callback=event_callback,
        red_zone_depth=depth,
    )


def _random_panel() -> FrozenPanel:
    """Describe Random through the same frozen System-panel route, without files."""
    method, identifier, registration, _ = _method_snapshot("random")
    member = PanelMember(
        "Random",
        registration_id=identifier,
        registration=registration,
        reference="random",
    )
    return FrozenPanel(
        Path("."),
        _digest({"random_deployment_panel": identifier}),
        (member,),
        True,
        schema_version=2,
        roots={**_panel_roots(None), "random": _ROOTS["random"]},
        methods=(method,),
    )


def validate_random(
    checkpoint: str | Path,
    *,
    output_dir: str | Path,
    seed_pairs: int = 10,
    root_seed: int = _ROOTS["random"],
    num_envs: int = 32,
    chunk_size: int = 128,
    event_callback: EventCallback | None = None,
    red_zone_depth: float = DEFAULT_TDM_RED_ZONE_DEPTH,
    partners: Mapping[str, System | Policy | str] | None = None,
    learner_slots: Sequence[int] | None = None,
    partner_labels: Mapping[str, str] | None = None,
    maps: Sequence[int] | None = None,
    system_roster: Sequence[AgentClassName] | None = None,
    opponent_roster: Sequence[AgentClassName] | None = None,
) -> Record:
    """Evaluate a frozen actor against Random on the five validation maps.

    Parameters
    ----------
    checkpoint : str or Path
        Exact saved actor export or complete learner checkpoint to evaluate.
    output_dir : str or Path
        Task-owned output folder. Reuse requires unchanged actor, root seed,
        game count, and other scientific settings; conflicts fail before games.
    seed_pairs : int, default=10
        Positive paired seeds per map. Both spawn sides give ten games per pair
        across maps 42-46: four pairs mean 40 games; 20 pairs mean 200 games.
    root_seed : int, default=19043001
        Plain integer in [0, 2**32 - 1] used for actual game and policy keys.
        The default preserves historical task hashes and games exactly. A
        different declared root supports fresh confirmation games; use that same
        root when verifying saved results. It is not a training seed.
    num_envs : int, default=32
        Positive worker count. GPU calls require 32; fewer than 32 unfinished
        games on resume use the existing CPU recovery route.
    chunk_size : int, default=128
        Positive number of ticks per evaluator block. Saved Random evidence
        verification requires the declared 128-tick block setting.
    event_callback : callable or None, default=None
        Receives task and execution records, or None for no callback.
    red_zone_depth : float, default=DEFAULT_TDM_RED_ZONE_DEPTH (5.0)
        Team Deathmatch Red Zone depth in map units for every game; 0.0 keeps
        one point per death. The schema 3 task records it, so a folder saved
        under another depth, or before the rule, is refused before any game.

    partners, learner_slots, partner_labels : optional
        Same team assignments and declared labels as validate_checkpoint. Each
        deployed team faces Random under the same paired game keys. None keeps
        the historical bare-actor task and identity.
    maps : sequence of int or None, default=None
        Distinct installed map IDs; None uses development maps 42 through 46.
    system_roster, opponent_roster : sequence of class names or None, default=None
        Fixed physical slot order; None uses canonical 5v5. Custom conditions
        and partners use the ordinary System-panel route with live clients and
        fixed padded batches, including recovery tails.

    Returns
    -------
    dict
        Complete scientific summary, including actual root, task and actor IDs,
        red_zone_depth, native scores (points), recorded kills per cell
        (mean_team_a_kills and mean_team_b_kills), paired-game uncertainty, and
        saved M8 pass paths. A QMIX or PQN-VDN actor's summary also carries its
        method and its optimizer_steps.

    Raises
    ------
    TypeError
        red_zone_depth is not a Python float.
    ValueError, OSError
        Settings, actor files, saved identity, or output files are invalid.

    Notes
    -----
    Defaults to canonical 5v5, K20/H300, equal map weights, and native task scores.
    Random is diagnostic only and does not become a learned panel member.
    No learner state or training key is accepted or changed. Every installed
    map is 20.0 map units wide. A wider depth raises ValueError before any
    file is written, so the same output_dir can be reused with a corrected
    depth.
    """
    root_seed = _root_seed(root_seed)
    if partners is not None or any(
        value is not None for value in (maps, system_roster, opponent_roster)
    ):
        return _validate_system_panel(
            checkpoint,
            _random_panel(),
            output_dir=output_dir,
            purpose="random",
            seed_pairs=seed_pairs,
            root_seed=root_seed,
            num_envs=num_envs,
            chunk_size=chunk_size,
            event_callback=event_callback,
            red_zone_depth=_checked_depth(red_zone_depth),
            partners=partners,
            learner_slots=learner_slots,
            partner_labels=partner_labels,
            maps=maps,
            system_roster=system_roster,
            opponent_roster=opponent_roster,
        )
    if any(value is not None for value in (learner_slots, partner_labels)):
        raise ValueError("Custom Random deployment settings need partners")
    return _validate(
        checkpoint,
        [("Random", "random", "builtin-random")],
        panel_digest="random-diagnostic-v1",
        output_dir=output_dir,
        purpose="random",
        seed_pairs=seed_pairs,
        root_seed=root_seed,
        num_envs=num_envs,
        chunk_size=chunk_size,
        event_callback=event_callback,
        red_zone_depth=_checked_depth(red_zone_depth),
    )


_RANDOM_CAPTURE_FIELDS = frozenset(
    {
        "actor_path",
        "summary_path",
        "reference_path",
        "reused_initialization",
        "elapsed_seconds",
        "training_seconds",
        "wall_seconds",
    }
)


def read_random_initialization(
    reference: str | Path,
    *,
    actor_digest: str,
    seed_pairs: int,
    root_seed: int = _ROOTS["random"],
    red_zone_depth: float | None = DEFAULT_TDM_RED_ZONE_DEPTH,
    deployment: Mapping[str, Any] | None = None,
) -> Record:
    """Read one original shared initialization result without changing any file.

    Parameters
    ----------
    reference : str or Path
        The copied original runner record, whose absolute evidence paths are
        unchanged. A relative reference is resolved from the current working
        directory. It must be an existing file reached without symbolic links.
    actor_digest : str
        The new learner's initial inference identity.
    seed_pairs : int
        Declared positive number of Random seed pairs per map (each pair plays
        both spawn sides).
    root_seed : int, default=19043001
        Expected game root, a plain integer in [0, 2**32 - 1].
    red_zone_depth : float or None, default=DEFAULT_TDM_RED_ZONE_DEPTH (5.0)
        The depth in map units that the reusing run plays under. The record
        must have been recorded at exactly that depth, so a result saved
        before the Red Zone rule is refused unless red_zone_depth is None.

    deployment : mapping or None, default=None
        Frozen validation_deployment from the owning run. It binds partner
        registrations, learner slots and custom conditions to saved M8 evidence.
        None preserves the historical bare-actor checks. No weights are rebuilt.

    Returns
    -------
    dict
        The original record, unchanged.

    Raises
    ------
    TypeError
        red_zone_depth is neither None nor a Python float.
    ValueError, OSError
        The reference is missing, follows a link, is itself a reused result,
        or its evidence is missing or does not match (another depth
        included). These errors come before any caller-owned recovery.

    Notes
    -----
    Read-only host work. All original evidence is checked through
    verify_random_result, which also requires zero training experience.
    """
    path = Path(reference).absolute()
    if path.resolve() != path or not path.is_file():
        raise ValueError("Shared Random initialization is missing or follows a link")
    record = _json(path)
    if (
        record.get("reused_initialization") is not False
        or record.get("reference_path") is not None
    ):
        raise ValueError("Shared Random initialization must be an original result")
    verify_random_result(
        record,
        actor_digest=actor_digest,
        seed_pairs=seed_pairs,
        root_seed=root_seed,
        env_steps=0,
        red_zone_depth=red_zone_depth,
        deployment=deployment,
    )
    return record


def verify_random_result(
    result: Mapping[str, Any],
    *,
    actor_digest: str,
    seed_pairs: int,
    root_seed: int = _ROOTS["random"],
    checkpoint_id: str | None = None,
    env_steps: int | None = None,
    run_id: str | None = None,
    seed: int | None = None,
    red_zone_depth: float | None = DEFAULT_TDM_RED_ZONE_DEPTH,
    deployment: Mapping[str, Any] | None = None,
) -> Record:
    """Verify one saved Random capture and return its original scientific summary.

    Parameters
    ----------
    result : mapping
        The runner record, with absolute actor and summary paths and capture
        times.
    actor_digest : str
        Expected inference identity.
    seed_pairs : int
        Expected positive number of paired seeds per map.
    root_seed : int, default=19043001
        Expected game root, a plain integer in [0, 2**32 - 1]. The saved root
        must match it.
    checkpoint_id : str or None, default=None
        When given, the exact originating learner boundary to require.
    env_steps : int or None, default=None
        When given, the exact training experience to require.
    run_id : str or None, default=None
        When given, the original training run to require.
    seed : int or None, default=None
        When given, the original training seed to require. Shared
        initialization keeps its original IDs; its caller checks the new
        initial actor identity instead and leaves out a different run's
        run_id and seed.
    red_zone_depth : float or None, default=DEFAULT_TDM_RED_ZONE_DEPTH (5.0)
        Expected Red Zone depth in map units. The task (schema 3), the M8
        options and the summary, with recorded kills, must all match it. None
        expects a record saved before the Red Zone rule (schema 1 task, no
        depth option, kills read from scores).

    deployment : mapping or None, default=None
        Frozen validation_deployment from the owning run. It binds partner
        registrations, learner slots and custom conditions to saved M8 evidence.
        None preserves the historical bare-actor checks. No weights are rebuilt.

    Returns
    -------
    dict
        The original scientific summary, without the runner's capture fields.

    Raises
    ------
    TypeError
        red_zone_depth is neither None nor a Python float.
    ValueError, OSError
        Missing or changed evidence: actor files, task, native M8 options,
        games or summary, including a record saved under another depth or
        before the rule.

    Notes
    -----
    Read-only host work. No files, learner state or keys change, and no
    policy call or new evaluation runs.
    """
    summary, _ = _verified_random_result(
        result,
        actor_digest=actor_digest,
        seed_pairs=seed_pairs,
        root_seed=root_seed,
        checkpoint_id=checkpoint_id,
        env_steps=env_steps,
        run_id=run_id,
        seed=seed,
        red_zone_depth=red_zone_depth,
        deployment=deployment,
    )
    return summary


def _verified_random_result(
    result: Mapping[str, Any],
    *,
    actor_digest: str,
    seed_pairs: int,
    root_seed: int = _ROOTS["random"],
    checkpoint_id: str | None = None,
    env_steps: int | None = None,
    run_id: str | None = None,
    seed: int | None = None,
    red_zone_depth: float | None = DEFAULT_TDM_RED_ZONE_DEPTH,
    deployment: Mapping[str, Any] | None = None,
) -> tuple[Record, list[Record]]:
    """Verify a runner's Random result against its original complete M8 evidence.

    Parameters
    ----------
    result : mapping
        One random_diagnostics.json entry, including its original actor_path and
        summary_path, optional reference_path, reuse flag and capture timings.
        A shared initialization reference must point to an original, unreused
        entry. Original task, actor, checkpoint and M8 pass identities stay intact.
    actor_digest : str
        Expected inference identity, including the input scale. Reused initial
        results must match the new learner's initial actor exactly.
    seed_pairs : int
        Expected positive number of independent seed pairs per validation map.
    root_seed : int, default=19043001
        Expected uint32 root used for game and policy keys. Fresh confirmation
        callers must supply their declared root. Shared initialization requires
        this same root and retains its original actor, task, and game identities.
    checkpoint_id, env_steps : str or None, int or None, default=None
        Optional exact originating learner boundary and experience checks. A
        shared initialization keeps its original checkpoint_id; require zero
        env_steps when checking that reference for another run.
    run_id, seed : str or None, int or None, default=None
        Optional original training-run identity checks. Shared initialization
        keeps its original run and seed; callers bind the new initial inference
        separately instead of passing a different run's identity here.
    red_zone_depth : float or None, default=DEFAULT_TDM_RED_ZONE_DEPTH (5.0)
        Expected Red Zone depth in map units. A float expects the schema 3
        task, a "red_zone_depth" M8 option equal to it, and rows and cells
        with recorded kills. None expects a record saved before the Red Zone
        rule: the schema 1 task, no depth option and kills read from scores.

    deployment : mapping or None, default=None
        Frozen validation_deployment from the owning run. It binds partner
        registrations, learner slots and custom conditions to saved M8 evidence.
        None preserves the historical bare-actor checks. No weights are rebuilt.

    Returns
    -------
    tuple of dict and list of dict
        Original scientific summary without runner capture fields, followed by
        its verified M8 episode rows (with team_a_kills and team_b_kills when a
        depth is expected). Analysis can reuse these host rows without reading
        or reducing the same files again.

    Raises
    ------
    TypeError
        red_zone_depth is neither None nor a Python float.
    ValueError, OSError
        Identity, paths, task, native game settings, completed rows or summary
        differ, including a record saved under another depth or before the
        rule. Paths must be absolute and must not traverse symbolic links.

    Notes
    -----
    Read-only host work. Verify actor file hashes and reduce saved M8 rows through
    summarize_validation. New sampling metadata is rebuilt from the saved pass's
    checked method facts; its public and conditional bounds must match. Historical
    summaries without that metadata keep their original conditional calculation.
    For a QMIX or PQN-VDN actor the M8 focal variables
    digest in a bare-actor pass is compared with the loaded greedy System's variables
    (the Q-network plus epsilon 0; for PQN-VDN the network is its parameters
    plus its frozen normalization statistics, so a statistics-only change
    breaks reuse), and the rebuilt summary carries its method and
    optimizer_steps. Deployed passes instead bind the native actor component and
    declared partner through the existing sampling sidecar and actual pass IDs.
    This verifies recorded constituents, not unavailable partner weights.
    No policy call, learner change, writer recovery or new
    evaluation occurs. This is an integrity check, not a usefulness gate.
    """
    from marl_battlegrounds.evaluation.results import load_results

    _integer(seed_pairs, "seed_pairs", minimum=1)
    root_seed = _root_seed(root_seed)
    depth = _task_depth(red_zone_depth)
    if not result.keys() >= _RANDOM_CAPTURE_FIELDS:
        raise ValueError("Random result is missing its capture evidence")
    _digest(result)

    def path(value: object, name: str) -> Path:
        """Require one existing absolute regular path without linked parents."""
        if not isinstance(value, str):
            raise ValueError(f"Random {name} must be an absolute path string")
        target = Path(value)
        if (
            not target.is_absolute()
            or target.resolve() != target
            or not target.exists()
        ):
            raise ValueError(f"Random {name} is missing or follows a symbolic link")
        return target

    for name in ("elapsed_seconds", "training_seconds", "wall_seconds"):
        value = result[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Random capture times must be finite nonnegative seconds")
    if result["training_seconds"] > result["elapsed_seconds"] + 1e-6:
        raise ValueError("Random training time exceeds its capture time")
    reused = result["reused_initialization"]
    if type(reused) is not bool:
        raise ValueError("Random initialization reuse flag must be bool")
    if reused != (result["reference_path"] is not None):
        raise ValueError("Random initialization reuse needs its exact reference")
    original: Record = {
        key: value for key, value in result.items() if key not in _RANDOM_CAPTURE_FIELDS
    }
    if reused:
        reference = _json(path(result["reference_path"], "reference_path"))
        if (
            reference.get("reused_initialization") is not False
            or reference.get("reference_path") is not None
            or result.get("env_steps") != 0
            or any(
                reference.get(name) != result[name]
                for name in ("actor_path", "summary_path")
            )
            or {
                key: value
                for key, value in reference.items()
                if key not in _RANDOM_CAPTURE_FIELDS
            }
            != original
        ):
            raise ValueError(
                "Shared Random initialization changed or chains references"
            )
        if env_steps not in (None, 0):
            raise ValueError("Shared Random initialization must have zero experience")
        return _verified_random_result(
            reference,
            actor_digest=actor_digest,
            seed_pairs=seed_pairs,
            root_seed=root_seed,
            checkpoint_id=checkpoint_id,
            env_steps=0,
            run_id=run_id,
            seed=seed,
            red_zone_depth=depth,
            deployment=deployment,
        )
    actor = _artifact(path(result["actor_path"], "actor_path"))
    if (
        actor["actor_digest"] != actor_digest
        or (checkpoint_id is not None and actor["checkpoint_id"] != checkpoint_id)
        or (env_steps is not None and actor["env_steps"] != env_steps)
        or (run_id is not None and actor["run_id"] != run_id)
        or (seed is not None and actor["seed"] != seed)
    ):
        raise ValueError("Random result differs from the expected actor or boundary")
    if deployment is not None:
        from marl_battlegrounds.training._selection_evidence import (
            _panel_evidence,  # pyright: ignore[reportPrivateUsage]
            _Snapshot,  # pyright: ignore[reportPrivateUsage]
        )

        declared = _checked_deployment(deployment)
        panel = _random_panel()
        expected = panel_task_description(
            checkpoint_id=actor["checkpoint_id"],
            actor_digest=actor_digest,
            env_steps=actor["env_steps"],
            panel=panel,
            purpose="random",
            seed_pairs=seed_pairs,
            root_seed=root_seed,
            red_zone_depth=depth,
            deployment=declared,
            **{
                key: declared[key]
                for key in ("maps", "system_roster", "opponent_roster")
                if key in declared
            },
        )
        summary_path = path(result["summary_path"], "summary_path")
        if (
            summary_path.name != "validation_summary.json"
            or _json(summary_path) != original
            or _json(path(str(summary_path.parent / "task.json"), "task")) != expected
            or any(original.get(key) != value for key, value in expected.items())
        ):
            raise ValueError("Random deployed-team task or summary differs")
        reduced, rows = _panel_evidence(
            summary_path, original, expected, actor, panel, _Snapshot()
        )
        if reduced != original:
            raise ValueError("Random summary disagrees with its saved M8 games")
        return original, rows
    expected = validation_task_description(
        checkpoint_id=actor["checkpoint_id"],
        actor_digest=actor_digest,
        env_steps=actor["env_steps"],
        panel_digest="random-diagnostic-v1",
        purpose="random",
        seed_pairs=seed_pairs,
        members=(("Random", "builtin-random"),),
        root_seed=root_seed,
        red_zone_depth=depth,
    )
    summary_path = path(result["summary_path"], "summary_path")
    directory = summary_path.parent
    if (
        summary_path.name != "validation_summary.json"
        or _json(summary_path) != original
        or _json(path(str(directory / "task.json"), "task")) != expected
        or any(original.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("Random summary or task identity changed")
    raw_paths: object = original.get("pass_paths")
    if not isinstance(raw_paths, list) or len(cast(list[Any], raw_paths)) != 1:
        raise ValueError("Random diagnostic needs its one complete M8 pass")
    paths = cast(list[Any], raw_paths)
    run_dir = path(paths[0], "pass_path")
    parent = _pass_directory(directory, 0, "Random")
    if run_dir.parent != parent or sorted(parent.glob("*/run_details.json")) != [
        run_dir / "run_details.json"
    ]:
        raise ValueError("Random M8 pass is outside its original task")
    pass_id = validation_pass_id(expected["task_id"], "Random")
    saved = load_results(run_dir, phase="validation", pass_id=pass_id)
    if saved.status != "complete" or len(saved.metadata["passes"]) != 1:
        raise ValueError("Random M8 pass is incomplete or ambiguous")
    entry = next(iter(saved.metadata["passes"].values()))
    policies, details = entry["policies"], entry["details"]
    focal, opponent = policies["team_a"], policies["team_b"]
    contract = details["evaluation_contract"]
    # A QMIX or PQN-VDN System's variables hold its network and the greedy
    # epsilon (PQN-VDN also its statistics), so M8 digests that tree; compare
    # with the verified loaded System's variables.
    expected_variables = actor["weight_digest"]
    if actor.get("method") in ("qmix", "pqn_vdn"):
        from marl_battlegrounds.evaluation.recording_identity import tree_digest
        from marl_battlegrounds.training.checkpoints import load_system

        loaded = load_system(path(result["actor_path"], "actor_path"))
        expected_variables = tree_digest(loaded.variables)
    games = len(VALIDATION_MAPS) * seed_pairs * 2
    options: Record = {
        "seed": root_seed,
        "spawn_mode": "paired",
        "score_threshold": 20,
        "max_steps": 300,
        "metrics": "priority",
        "save_replays": 0,
        "system_roster": None,
        "opponent_roster": None,
        "full_metrics_episodes": [],
        "replay_episodes": [],
    }
    if depth is not None:
        # Passes saved with the Red Zone rule record the depth they used.
        options["red_zone_depth"] = depth
    if (
        focal["checkpoint"] != actor_digest
        or focal["variables_digest"] != expected_variables
        or focal["variables_frozen"] is not True
        or opponent["name"] != "random"
        or opponent["callable_name"]
        != "marl_battlegrounds.evaluation.policy_execution._random_apply"
        or contract["version"] != 1
        or contract["schedule_kind"] != "generated"
        or contract["map_selection"] != "explicit"
        or contract["spawn_mode"] != "paired"
        or contract["options"] != options
        or [item["map_id"] for item in contract["source_choices"]]
        != list(VALIDATION_MAPS)
        or details["phase"] != "validation"
        or details["pass_id"] != pass_id
        or details["seed"] != root_seed
        or details["num_episodes"] != games
        or details["metrics"] != "priority"
        or details["chunk_size"] != 128
    ):
        raise ValueError("Random M8 actor or native evaluation settings changed")
    rows = _rows(run_dir, pass_id=pass_id, opponent="Random", kills=depth is not None)
    sampling = None
    if "sampling_evidence" in original:
        sampling = validation_sampling_evidence(
            rows,
            facts={
                "Random": _read_pass_sampling(
                    run_dir, task_id=expected["task_id"], pass_id=pass_id
                )
            },
            scheduled_games=games,
            independent_opponents=False,
        )
    reduced = {
        **expected,
        **summarize_validation(
            rows,
            maps=VALIDATION_MAPS,
            opponents=("Random",),
            seed_pairs=seed_pairs,
            actual_kills=depth is not None,
            sampling_evidence=sampling,
        ),
        "pass_paths": paths,
        **_method_fields(actor),
    }
    if reduced != original:
        raise ValueError("Random summary disagrees with its saved M8 games")
    return original, rows


def make_slot_diagnostic_schedule(
    *,
    maps: Sequence[int] = VALIDATION_MAPS,
    seed_blocks: int = 160,
    configs: Sequence[EnvConfig] | None = None,
    red_zone_depth: float | None = None,
) -> tuple[tuple[EpisodeSpec, ...], tuple[EpisodeSpec, ...]]:
    """Build four physical-side/assignment conditions per independent seed block.

    Parameters
    ----------
    maps : sequence of int, default=VALIDATION_MAPS (42-46)
        Distinct installed map IDs; it must not be empty.
    seed_blocks : int, default=160
        Positive number of independent seed blocks per map.
    configs : sequence of EnvConfig or None, default=None
        None builds canonical 5v5 K20/H300 for each map. Explicit scalar
        configs are a test and custom route: one per map, in map order, each
        with identical ordered team profiles. They keep their own rules.
    red_zone_depth : float or None, default=None
        Red Zone depth in map units for the built configs; None uses
        DEFAULT_TDM_RED_ZONE_DEPTH (5.0). It cannot be given with configs.

    Returns
    -------
    tuple of (tuple of EpisodeSpec, tuple of EpisodeSpec)
        Two schedules: focal actor on Team A, then focal actor on Team B. Each
        holds, for every map and seed block, the source spawn banks and then
        the exchanged banks (len(maps) * seed_blocks * 2 games). Global episode
        IDs are unique across both schedules; map-distinct seed IDs repeat only
        across the four conditions. Metadata names focal_team, physical_side
        and seed_block.

    Raises
    ------
    TypeError
        Core's config check refuses the type of red_zone_depth, such as an
        int.
    ValueError
        seed_blocks is not a positive integer; maps is empty or repeats an ID;
        configs does not match maps; configs and red_zone_depth are given
        together; the two teams' ordered profiles differ; a map ID is not
        installed; or red_zone_depth breaks Core's rules (not finite,
        negative, -0.0, a positive value that is not a normal float32 number,
        or wider than the map, which is 20.0 map units for every installed
        map).

    Notes
    -----
    Host-only. No game is executed and no file is written.
    """
    import numpy as np

    from marl_battlegrounds.evaluation.evaluate import EpisodeSpec
    from marl_battlegrounds.tasks import (
        _swap_spawn_banks,  # pyright: ignore[reportPrivateUsage]
        canonical_tournament_rosters,
        make_standard_team_deathmatch_config,
    )

    count = _integer(seed_blocks, "seed_blocks", minimum=1)
    identifiers = tuple(maps)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Slot maps must be nonempty and distinct")
    if configs is not None and len(configs) != len(identifiers):
        raise ValueError("Slot configs must match the map sequence")
    if configs is not None and red_zone_depth is not None:
        raise ValueError(
            "red_zone_depth cannot be combined with explicit slot configs; "
            "each config owns its rules"
        )
    depth = DEFAULT_TDM_RED_ZONE_DEPTH if red_zone_depth is None else red_zone_depth
    roster_a, roster_b = canonical_tournament_rosters()
    sources = (
        tuple(configs)
        if configs is not None
        else tuple(
            make_standard_team_deathmatch_config(
                map_id=map_id,
                team_a_roster=roster_a,
                team_b_roster=roster_b,
                red_zone_depth=depth,
            )
            for map_id in identifiers
        )
    )
    for config in sources:
        if any(
            not np.array_equal(field[:5], field[5:])
            for name, field in zip(
                config.agent_profile._fields, config.agent_profile, strict=True
            )
            if name != "team_ids"
        ):
            raise ValueError("Slot comparison needs identical ordered team profiles")
    schedules: tuple[list[EpisodeSpec], list[EpisodeSpec]] = ([], [])
    for map_index, (map_id, source) in enumerate(
        zip(identifiers, sources, strict=True)
    ):
        exchanged = _swap_spawn_banks(source)
        for block in range(count):
            seed_id = map_index * count + block + 1
            for team in (0, 1):
                for end, config in enumerate((source, exchanged)):
                    episode_id = (seed_id - 1) * 4 + team * 2 + end + 1
                    schedules[team].append(
                        EpisodeSpec(
                            episode_id,
                            config,
                            map_id,
                            seed_id=seed_id,
                            source_config=source,
                            spawn_locations=end,
                            paired_comparison_key=f"slot-{seed_id}-team-{team}",
                            metadata={
                                "focal_team": team,
                                "physical_side": team ^ end,
                                "seed_block": seed_id,
                            },
                        )
                    )
    return tuple(schedules[0]), tuple(schedules[1])


def run_slot_diagnostic(
    checkpoint: str | Path,
    opponent: str | Path,
    *,
    output_dir: str | Path,
    seed_blocks: int = 160,
    num_envs: int = 32,
    chunk_size: int = 128,
    event_callback: EventCallback | None = None,
    red_zone_depth: float = DEFAULT_TDM_RED_ZONE_DEPTH,
) -> Record:
    """Run the predeclared frozen two-model slot comparison through M8.

    Parameters
    ----------
    checkpoint : str or Path
        The focal final actor (an export or a complete learner checkpoint).
    opponent : str or Path
        The fixed development-final actor.
    output_dir : str or Path
        Folder owned by this exact task. Repeat the same call to resume
        interrupted matching passes.
    seed_blocks : int, default=160
        Independent seed blocks per validation map, at least 2. Each block
        plays four games, so the default is 3,200 games over the five maps.
    num_envs : int, default=32
        Positive number of games run side by side, as in validate_checkpoint.
    chunk_size : int, default=128
        Positive number of ticks per recording block.
    event_callback : callable or None, default=None
        Receives task, execution and completion records; None means no calls.
    red_zone_depth : float, default=DEFAULT_TDM_RED_ZONE_DEPTH (5.0)
        Red Zone depth in map units for every game; 0.0 keeps one point per
        death. To compare actors under the rule they trained with, pass that
        depth. The slot task (schema 2) records it, so a folder saved under
        another depth, or before the rule, is refused.

    Returns
    -------
    dict
        The task fields, the complete raw-pass references (pass_paths), exact
        identities and qualified-scope statistics (W/D/L game scores only).

    Raises
    ------
    TypeError
        red_zone_depth is not a Python float.
    ValueError
        red_zone_depth breaks Core's scalar rules (not finite, negative, -0.0,
        or a positive value that is not a normal float32 number) or is wider
        than a validation map; a count is invalid; an actor identity is
        invalid; or output_dir holds a different task. File and method
        failures keep their own cause.

    Notes
    -----
    Writes task.json, one M8 pass folder per focal team and
    slot_diagnostic.json under output_dir. This function does not decide when
    trained-model execution is authorized or whether either actor learned
    useful behavior; the runner owns those gates. Every installed map is 20.0
    map units wide. A wider depth raises ValueError before any file is
    written, so the same output_dir can be reused with a corrected depth.
    """
    _integer(seed_blocks, "seed_blocks", minimum=2)
    _integer(num_envs, "num_envs", minimum=1)
    _integer(chunk_size, "chunk_size", minimum=1)
    depth = _checked_depth(red_zone_depth)
    focal, other = _artifact(checkpoint), _artifact(opponent)
    directory = Path(output_dir).resolve()
    task = _task(
        directory,
        {
            "schema_version": 2,
            "purpose": "slot",
            "checkpoint_id": focal["checkpoint_id"],
            "actor_digest": focal["actor_digest"],
            "opponent_digest": other["actor_digest"],
            "env_steps": focal["env_steps"],
            "maps": list(VALIDATION_MAPS),
            "seed_blocks": seed_blocks,
            "root": _ROOTS["slot"],
            "red_zone_depth": depth,
        },
        event_callback,
    )
    rows: list[Record] = []
    paths: list[str] = []
    for team in (0, 1):
        pass_id = _digest({"task_id": task["task_id"], "focal_team": team})
        request: Record = {
            "kind": "slot",
            "task_id": task["task_id"],
            "actor": str(Path(checkpoint).resolve()),
            "actor_digest": focal["actor_digest"],
            "opponent": str(Path(opponent).resolve()),
            "opponent_digest": other["actor_digest"],
            "focal_team": team,
            "output_parent": str(directory / f"team-{team}"),
            "pass_id": pass_id,
            "maps": list(VALIDATION_MAPS),
            "seed_pairs": seed_blocks,
            "total_games": len(VALIDATION_MAPS) * seed_blocks * 2,
            "root": _ROOTS["slot"],
            "num_envs": num_envs,
            "chunk_size": chunk_size,
            "red_zone_depth": depth,
        }
        path = _run_pass(request, event_callback=event_callback)
        paths.append(str(path))
        rows.extend(
            _rows(
                path, pass_id=pass_id, opponent=other["actor_digest"], focal_team=team
            )
        )
    result = {
        **task,
        **summarize_slot_diagnostic(
            rows, maps=VALIDATION_MAPS, seed_blocks=seed_blocks
        ),
        "pass_paths": paths,
    }
    _publish(directory / "slot_diagnostic.json", result)
    if event_callback is not None:
        event_callback(
            {
                "event": "slot_diagnostic_complete",
                "task_id": task["task_id"],
                "result": result,
            }
        )
    return result


def _main() -> None:
    """Execute only a saved CPU worker request; ordinary use is through the runner."""
    parser = argparse.ArgumentParser(
        description="Complete one saved CPU validation task"
    )
    parser.add_argument("--worker", type=Path, required=True)
    arguments = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("Validation worker requires JAX_PLATFORMS=cpu")
    _execute_request(_json(arguments.worker))


if __name__ == "__main__":
    _main()
