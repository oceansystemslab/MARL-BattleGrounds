"""Run fixed training-validation tasks through the existing M8 evaluator.

Panels bind two immutable actor artifacts. Tasks preserve exact weights, game
schedules and durable M8 pass identities across interruption. The learner never
enters this module. A small resumed GPU tail uses the same evaluator in a pinned
CPU subprocess, leaving original game identities unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from marl_battlegrounds.training.analysis import (
    _atomic_text,  # pyright: ignore[reportPrivateUsage]
    _integer,  # pyright: ignore[reportPrivateUsage]
    summarize_slot_diagnostic,
    summarize_validation,
)

if TYPE_CHECKING:
    from marl_battlegrounds.core.types import EnvConfig
    from marl_battlegrounds.evaluation.evaluate import EpisodeSpec

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
    """One actual update boundary and all requested thresholds that reached it.

    requested_steps is a tuple of exact rational-upward-rounded experience targets.
    env_steps is the actual cumulative experience; update_index counts completed
    updates. The initial diagnostic has zero for each and is never selectable.
    """

    requested_steps: tuple[int, ...]
    env_steps: int
    update_index: int


def resolve_validation_schedule(
    total_env_steps: int,
    num_envs: int,
    rollout_length: int = 128,
    fractions: Sequence[float] = tuple(i / 10 for i in range(1, 11)),
) -> tuple[ValidationPoint, ...]:
    """Resolve requested fractions to completed update boundaries, including final.

    total_env_steps is positive and divisible by positive num_envs. rollout_length
    is a positive Python integer. fractions must be finite, increasing and inside
    (0,1]. Decimal strings define their exact fractions. Initial zero and final
    budget are always included; coincident rounded points are merged. Return an
    ordered tuple without changing settings, splitting a rollout or doing I/O.
    Invalid values raise ValueError.
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
    block = batch * length
    points: dict[int, list[int]] = {0: [0]}
    for fraction in amounts:
        requested = math.ceil(total * fraction)
        actual = min(total, math.ceil(requested / block) * block)
        points.setdefault(actual, []).append(requested)
    return tuple(
        ValidationPoint(
            tuple(dict.fromkeys(requests)), actual, math.ceil(actual / block)
        )
        for actual, requests in sorted(points.items())
    )


@dataclass(frozen=True)
class PanelMember:
    """One frozen Halfway/Final artifact with verified numerical and run identity."""

    name: str
    path: Path
    actor_digest: str
    checkpoint_id: str
    run_id: str
    seed: int
    env_steps: int


@dataclass(frozen=True)
class FrozenPanel:
    """Verified immutable panel manifest and ordered actor paths.

    path identifies panel.json; digest excludes storage paths so identical relocated
    actors keep their scientific identity. members are Halfway then Final. qualified
    means the declared Random diagnostics passed; it is not a competence claim.
    """

    path: Path
    digest: str
    members: tuple[PanelMember, ...]
    qualified: bool


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


def create_panel(
    halfway: str | Path,
    final: str | Path,
    *,
    output_dir: str | Path,
    diagnostics: Sequence[Mapping[str, Any]] = (),
    test_only: bool = False,
) -> FrozenPanel:
    """Freeze the two predeclared development actors and their usefulness evidence.

    halfway/final are complete artifact paths from the same run/seed, with increasing
    positive env_steps. output_dir owns an immutable panel.json. diagnostics contains
    their two complete fixed 100-game Random summaries, with exact task/checkpoint
    identities, ten pairs per map, finite scores above 0.5 and positive mean Team A
    scores on at least two maps. Verified summaries are embedded in panel.json;
    original storage paths are excluded from its scientific digest. test_only bypasses
    that learning gate for fixtures and marks the panel unqualified. Return the
    verified panel. Conflicting files or identities raise ValueError. No game runs.
    """
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
) -> Record:
    """Describe one exact fixed-map validation task without files or numerical work.

    checkpoint_id names the originating learner boundary; actor_digest identifies
    its frozen weights. env_steps counts real training transitions. panel_digest
    identifies the frozen panel, and members gives its ordered (name, weight digest)
    pairs. purpose is routine, initialization, confirmation or random; seed_pairs
    is positive. root_seed=None keeps the existing purpose-specific root. An
    explicit uint32 root is supported for Random diagnostics only; changing it
    changes actual game keys and task identity. Shared constants supply maps
    and the task schema. Existing calls retain exactly the same task identity.
    Return a new JSON-ready description including its immutable task_id. Invalid
    counts, purpose or empty/duplicate identities raise ValueError. The caller
    verifies artifact contents and the declared panel; this helper opens nothing.
    """
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
        "schema_version": 1,
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
    return {**result, "task_id": _digest(result)}


def _qualify_random(identity: Mapping[str, Any], result: Mapping[str, Any]) -> Record:
    """Check the complete declared Random diagnostic and retain portable evidence.

    identity names a checked actor and originating learner checkpoint. result is
    the summary returned by validate_random for exactly 100 games. Verify finite
    W/D/L totals, all five cells, declared seed roots and the immutable task hash.
    Return its JSON-ready scientific fields without pass storage paths. Missing,
    conflicting, nonfinite or unsuccessful evidence raises ValueError.
    """
    expected = validation_task_description(
        checkpoint_id=identity["checkpoint_id"],
        actor_digest=identity["actor_digest"],
        env_steps=identity["env_steps"],
        panel_digest="random-diagnostic-v1",
        purpose="random",
        seed_pairs=10,
        members=(("Random", "builtin-random"),),
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
                {field: value for field, value in member.items() if field != "path"}
                for member in cast(Sequence[Mapping[str, Any]], value)
            ]
            if key == "members"
            else value
            for key, value in content.items()
            if key != "panel_digest"
        }
    )


def load_panel(path: str | Path) -> FrozenPanel:
    """Verify panel.json and every member artifact without applying an actor.

    path may name the manifest or its containing directory. Member paths may be
    absolute or relative to that directory. Scientific digests are path-independent;
    moving files without changing content is permitted. Return frozen host records.
    Malformed schemas, changed artifacts or changed identities raise ValueError.
    This host read may verify numerical payload files but never runs a game.
    """
    target = Path(path).resolve()
    if target.is_dir():
        target /= "panel.json"
    content = _json(target)
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
            event_callback({"event": "validation_task", **result})
    return result


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
    paths/digests, pass/schedule fields and a task-owned output parent. No learner
    state is accepted. M8 owns scientific resume validation and all writer recovery.
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
            maps=request["maps"], seed_blocks=request["seed_pairs"]
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
    run_dir: Path, *, pass_id: str, opponent: str, focal_team: int | None = None
) -> list[Record]:
    """Join M8's complete saved rows to exact schedule conditions after resume."""
    from marl_battlegrounds.evaluation.results import load_results

    result = load_results(run_dir, phase="validation", pass_id=pass_id)
    if result.status != "complete":
        raise ValueError("Validation pass is incomplete")
    entries = result.metadata["passes"]
    entry = next(iter(entries.values()))
    declared = entry["episodes"]
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
    root_seed: int | None = None,
) -> Record:
    """Own one exact checkpoint/panel task and reduce its complete saved M8 rows."""
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
        ),
        event_callback,
    )
    rows: list[Record] = []
    paths: list[str] = []
    for index, (name, opponent, digest) in enumerate(members):
        pass_id = validation_pass_id(task["task_id"], name)
        request: Record = {
            "kind": "panel",
            "task_id": task["task_id"],
            "actor": str(Path(checkpoint).resolve()),
            "actor_digest": identity["actor_digest"],
            "opponent": opponent,
            "opponent_digest": digest,
            "output_parent": str(directory / f"opponent-{index}"),
            "pass_id": pass_id,
            "maps": list(VALIDATION_MAPS),
            "seed_pairs": seed_pairs,
            "total_games": len(VALIDATION_MAPS) * seed_pairs * 2,
            "root": task["root"],
            "num_envs": num_envs,
            "chunk_size": chunk_size,
        }
        path = _run_pass(request, event_callback=event_callback)
        paths.append(str(path))
        rows.extend(_rows(path, pass_id=pass_id, opponent=name))
    result = {
        **task,
        **summarize_validation(
            rows,
            maps=VALIDATION_MAPS,
            opponents=[name for name, _, _ in members],
            seed_pairs=seed_pairs,
        ),
        "pass_paths": paths,
    }
    _publish(directory / "validation_summary.json", result)
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
    checkpoint: str | Path,
    panel: FrozenPanel | str | Path,
    *,
    output_dir: str | Path,
    purpose: str = "routine",
    seed_pairs: int | None = None,
    num_envs: int = 32,
    chunk_size: int = 128,
    event_callback: EventCallback | None = None,
) -> Record:
    """Evaluate one exact saved actor against the immutable two-member panel.

    checkpoint identifies an actor export or complete learner checkpoint. panel is
    verified again even when supplied as FrozenPanel. output_dir is one task-owned
    directory, reused only for identical scientific conditions. purpose is routine,
    initialization or confirmation; seed_pairs defaults to 10/10/50 respectively.
    num_envs is positive and must be 32 on GPU; chunk_size defaults to 128. Optional
    event_callback receives task and execution-segment records before mutation and
    verified completion afterward. Return a JSON-ready complete result with score,
    cells, actor/checkpoint/panel identity and original M8 pass paths. Failures raise
    without treating partial games as completed validation. No learner is accepted.
    """
    if purpose not in ("routine", "initialization", "confirmation"):
        raise ValueError("Unknown checkpoint-validation purpose")
    frozen = load_panel(panel.path if isinstance(panel, FrozenPanel) else panel)
    pairs = (
        (50 if purpose == "confirmation" else 10) if seed_pairs is None else seed_pairs
    )
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

    Returns
    -------
    dict
        Complete scientific summary, including actual root, task and actor IDs,
        native scores, paired-game uncertainty, and saved M8 pass paths.

    Raises
    ------
    ValueError, OSError
        Settings, actor files, saved identity, or output files are invalid.

    Notes
    -----
    Keeps canonical 5v5, K20/H300, equal map weights, and unshaped task scores.
    Random is diagnostic only and does not become a learned panel member.
    No learner state or training key is accepted or changed.
    """
    root_seed = _root_seed(root_seed)
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
) -> Record:
    """Read one original shared initialization result without changing any file.

    reference names the copied original runner record, with unchanged absolute
    evidence paths. Relative references use the current working directory.
    actor_digest is the new learner's initial inference identity; seed_pairs is
    its declared Random game count per map and spawn pair. root_seed is the
    expected uint32 game root, default 19043001. Verify all original
    evidence through verify_random_result and require zero training experience.
    Return the original record unchanged. Missing, linked, chained or mismatched
    evidence raises ValueError or OSError before any caller-owned recovery.
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
) -> Record:
    """Verify one saved Random capture and return its original scientific summary.

    result is the runner record with absolute actor/summary paths and capture
    times. actor_digest and positive seed_pairs declare the expected inference
    and paired game count. root_seed is the expected uint32 game root, default
    19043001; a saved root must match it. Optional checkpoint_id/env_steps require
    that exact originating boundary. Optional run_id/seed bind the original run.
    Shared initialization retains its original IDs; its caller instead checks
    the new initial actor identity and omits a different run's ID/seed.
    Read and verify actor files, task, native M8 options, games and summary;
    raise ValueError or OSError for missing/changed evidence. No files, learner
    state or keys change, and no policy or new evaluation runs.
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

    Returns
    -------
    tuple of dict and list of dict
        Original scientific summary without runner capture fields, followed by
        its verified M8 episode rows. Analysis can reuse these host rows without
        reading or reducing the same files again.

    Raises
    ------
    ValueError, OSError
        Identity, paths, task, native game settings, completed rows or summary
        differ. Paths must be absolute and must not traverse symbolic links.

    Notes
    -----
    Read-only host work. Verify actor file hashes and reduce saved M8 rows through
    summarize_validation. No policy call, learner change, writer recovery or new
    evaluation occurs. This is an integrity check, not a usefulness gate.
    """
    from marl_battlegrounds.evaluation.results import load_results

    _integer(seed_pairs, "seed_pairs", minimum=1)
    root_seed = _root_seed(root_seed)
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
    expected = validation_task_description(
        checkpoint_id=actor["checkpoint_id"],
        actor_digest=actor_digest,
        env_steps=actor["env_steps"],
        panel_digest="random-diagnostic-v1",
        purpose="random",
        seed_pairs=seed_pairs,
        members=(("Random", "builtin-random"),),
        root_seed=root_seed,
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
    parent = directory / "opponent-0"
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
    if (
        focal["checkpoint"] != actor_digest
        or focal["variables_digest"] != actor["weight_digest"]
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
    rows = _rows(run_dir, pass_id=pass_id, opponent="Random")
    reduced = {
        **expected,
        **summarize_validation(
            rows, maps=VALIDATION_MAPS, opponents=("Random",), seed_pairs=seed_pairs
        ),
        "pass_paths": paths,
    }
    if reduced != original:
        raise ValueError("Random summary disagrees with its saved M8 games")
    return original, rows


def make_slot_diagnostic_schedule(
    *,
    maps: Sequence[int] = VALIDATION_MAPS,
    seed_blocks: int = 160,
    configs: Sequence[EnvConfig] | None = None,
) -> tuple[tuple[EpisodeSpec, ...], tuple[EpisodeSpec, ...]]:
    """Build four physical-side/assignment conditions per independent seed block.

    maps gives distinct installed map IDs. seed_blocks is positive. configs=None
    builds canonical 5v5 K20/H300; explicit scalar configs are a test/custom route
    and must use identical ordered team profiles. Return two schedules: focal actor
    on Team A, then focal actor on Team B. Each contains paired complete source and
    exchanged banks. Global episode IDs are unique across schedules; map-distinct
    seed IDs repeat only across the four conditions. No game is executed. Invalid
    counts, map duplicates or asymmetric profiles raise ValueError.
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
    roster_a, roster_b = canonical_tournament_rosters()
    sources = (
        tuple(configs)
        if configs is not None
        else tuple(
            make_standard_team_deathmatch_config(
                map_id=map_id, team_a_roster=roster_a, team_b_roster=roster_b
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
) -> Record:
    """Run the predeclared frozen two-model slot comparison through M8.

    checkpoint is the focal final actor and opponent is the fixed development-final
    actor. output_dir owns this exact task; interrupted matching passes resume.
    Defaults run 160 blocks on each of five validation maps, four games per block.
    num_envs/chunk_size and event_callback follow validate_checkpoint. Return the
    complete raw-pass references, exact identities and qualified-scope statistics.
    This function does not decide when trained-model execution is authorized or
    whether either actor learned useful behavior; the runner owns those gates.
    """
    _integer(seed_blocks, "seed_blocks", minimum=2)
    _integer(num_envs, "num_envs", minimum=1)
    _integer(chunk_size, "chunk_size", minimum=1)
    focal, other = _artifact(checkpoint), _artifact(opponent)
    directory = Path(output_dir).resolve()
    task = _task(
        directory,
        {
            "schema_version": 1,
            "purpose": "slot",
            "checkpoint_id": focal["checkpoint_id"],
            "actor_digest": focal["actor_digest"],
            "opponent_digest": other["actor_digest"],
            "env_steps": focal["env_steps"],
            "maps": list(VALIDATION_MAPS),
            "seed_blocks": seed_blocks,
            "root": _ROOTS["slot"],
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
