"""Run the declared recurrent MAPPO search without an attached assistant.

Run ``python -m marl_battlegrounds.training.search --help`` for preparation,
bounded calibration, detached launch, status, stop, resume and reporting.
The ordinary trainer owns learning and recovery. Validation owns opponents,
games and checkpoint selection. This host driver owns only the fixed recipes,
timing-based budget, run order and complete-recipe comparison. Import and status
use no numerical backend. Historical configuration screens are unchanged.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from marl_battlegrounds.training import _launch
from marl_battlegrounds.training._run_io import (
    append_jsonl,
    atomic_json,
    process_identity,
    read_jsonl,
    utc_now,
)

type Record = dict[str, Any]
_MANIFEST = "search_package.json"
_QUANTUM = 131_072
_MINIMUM_STEPS = 20_054_016
_DISCOVERY_SEEDS = (19_046_101, 19_046_102, 19_046_103)
_FINALIST_SEEDS = (19_046_201, 19_046_202, 19_046_203)
_ROOTS = {
    "discovery": (19_046_300, 19_046_400),
    "finalists": (19_046_500, 19_046_600),
    "alpha": (19_046_700, 19_046_701),
    "beta": (19_046_800, 19_046_801),
}
_SCRIPTS = {
    "launch": "start",
    "status": "status",
    "stop": "stop",
    "resume": "resume",
    "report": "report",
    "calibrate": "calibrate",
}


def _read(path: Path) -> Record:
    """Read one required regular JSON object through the launch owner."""
    return _launch._read(path)


def _optional(path: Path) -> Record:
    """Read a saved object, returning an empty object only when absent."""
    return _read(path) if path.exists() else {}


def _digest(value: object) -> str:
    """Hash finite canonical JSON, keeping list order and every field."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def recipes() -> list[Record]:
    """Return twelve ordered, independent recipes as changes from the reference.

    Every change is fixed before timing or learning. The first eight cover each
    main lever, the next two test higher learning rate and reuse, and the final
    two test named interactions. This is not an independent-factor analysis.
    """
    changes: tuple[Record, ...] = (
        {},
        {"actor_lr": 0.0001},
        {"entropy_coefficient": 0.003},
        {"minibatches": 1},
        {"clip_epsilon": 0.1},
        {"pinned_opponent_share": 0.3},
        {"rollout_length": 64},
        {"epochs": 2},
        {"actor_lr": 0.0005},
        {"epochs": 8},
        {"minibatches": 1, "clip_epsilon": 0.1},
        {"actor_lr": 0.0001, "entropy_coefficient": 0.003},
    )
    return [
        {"recipe_id": f"c{i:02}", "changes": dict(change)}
        for i, change in enumerate(changes)
    ]


def declaration() -> Record:
    """Expand the fixed study's settings through the ordinary config owner.

    This setup-only function lazily imports the learner config. It starts no
    training. Its imports and config construction use CPU array placement, then
    restore the caller's device default. JAX may still discover other backends
    in direct Python calls. Returned JSON fixes eight recipes, three discovery
    seeds each, 20,054,016 transitions per discovery run and twice that per fresh
    finalist. Timing must fit thirteen hours including margin and reporting.
    """
    import jax

    with jax.default_device(jax.devices("cpu")[0]):
        from marl_battlegrounds.baselines.ppo import PPOConfig
        from marl_battlegrounds.training.runner import TrainConfig, config_to_dict

        config = TrainConfig(
            num_envs=512,
            total_env_steps=_MINIMUM_STEPS,
            shaping=True,
            shaping_mode="score_delta",
            shaping_coefficient=0.01,
            pinned_opponent="tdm-alpha",
            pinned_opponent_share=0.1,
            ppo=PPOConfig(
                rollout_length=32,
                input_scale=0.01,
                spawn_frame="left",
                gamma=0.999,
                value_normalization=True,
            ),
            metrics="none",
            recording=False,
            validation_fractions=(0.25, 0.5, 0.75, 1.0),
            routine_seed_pairs=4,
            confirmation_seed_pairs=10,
        )
        base_config = config_to_dict(config)
    return {
        "schema_version": 1,
        "name": "Eight-Recipe Recurrent MAPPO Search With Alpha Validation",
        "base_config": base_config,
        "recipes": recipes()[:8],
        "tiers": [8],
        "discovery_seeds": list(_DISCOVERY_SEEDS),
        "finalist_seeds": list(_FINALIST_SEEDS),
        "calibration_seed": 19_046_000,
        "order_seed": 19_046_001,
        "roots": _ROOTS,
        "quantum": _QUANTUM,
        "minimum_steps": _MINIMUM_STEPS,
        "fixed_discovery_steps": _MINIMUM_STEPS,
        "target_seconds": 46_800,
        "numerical_stop_seconds": 46_500,
        "hard_stop_seconds": 46_800,
        "timing_margin": 1.10,
        "report_reserve_seconds": 300,
        "calibration_limit_seconds": 3_600,
        "memory_fraction": 0.85,
        "training_maps": list(range(42)),
        "validation_maps": list(range(42, 47)),
        "score_threshold": 20,
        "horizon": 300,
        "routine_captures": 5,
        "maximum_confirmation_actors": 3,
        "discovery_pairs": [4, 10],
        "finalist_pairs": [8, 20],
        "assessment_pairs": {"alpha": 40, "beta": 20},
        "selection_rule": (
            "Mean confirmed native score, mean kill difference, "
            "earlier mean selected step, recipe ID"
        ),
        "limits": (
            "Finite development search. Alpha is used in training and selection. "
            "Beta is a related familiar diagnostic. Fresh assessment games are "
            "conditional on selected actors, not unseen training seeds."
        ),
    }


def run_config(
    declared: Mapping[str, Any],
    recipe: Mapping[str, Any],
    *,
    seed: int,
    steps: int,
    panel: str | None,
    finalists: bool = False,
) -> Record:
    """Build one fresh run config without changing the declaration.

    recipe supplies declared changes; seed and steps are frozen by the caller.
    panel is an immutable panel path, or None for engineering calibration.
    finalists chooses the larger fixed validation counts, never a new learner.
    """
    result = json.loads(json.dumps(declared["base_config"]))
    result.update(seed=seed, total_env_steps=steps, validation_panel=panel)
    result["routine_seed_pairs"], result["confirmation_seed_pairs"] = declared[
        "finalist_pairs" if finalists else "discovery_pairs"
    ]
    for name, value in recipe["changes"].items():
        target = result if name == "pinned_opponent_share" else result["ppo"]
        target[name] = value
    return cast(Record, result)


def _positive(value: object, name: str) -> float:
    """Reject a missing, Boolean, nonfinite or nonpositive cost measurement."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} needs a finite positive measurement")
    return float(value)


def resolve_budgets(declared: Mapping[str, Any], costs: Mapping[str, Any]) -> Record:
    """Price fixed experience or choose a historical adaptive experience budget.

    costs contains recipe timing rows and measured validation seconds, never
    rewards or scores. Reserve every scheduled validation, recurring recovery
    save, cold/setup cost and the worst-cost eligible challenger. A 10% margin
    plus five-minute report reserve must fit the declaration's time target.
    When fixed_discovery_steps is present, it must be a positive integer at
    least minimum_steps and divisible by quantum. Use exactly that experience;
    never grow or shrink it. Declarations without this field retain the widest
    feasible tier and largest affordable budget. Return budgets and shuffled
    discovery order; raise ValueError for invalid or unaffordable fixed work.
    """
    timings: dict[str, Record] = {}
    for recipe in declared["recipes"]:
        identifier = recipe["recipe_id"]
        row = costs["recipes"][identifier]
        timings[identifier] = {
            name: _positive(row[name], f"{identifier}.{name}")
            for name in (
                "seconds_per_transition",
                "setup_seconds",
                "save_seconds",
                "report_seconds",
                "run_bytes",
                "checkpoint_bytes",
                "checkpoint_metadata_bytes",
                "actor_bytes",
                "log_bytes_per_update",
            )
        }
    validation = {
        name: _positive(costs["validation_seconds"][name], name)
        for name in (
            "alpha40",
            "alpha80",
            "alpha100",
            "alpha200",
            "alpha400",
            "beta200",
        )
    }
    validation_bytes = {
        name: _positive(costs["validation_bytes"][name], f"{name} output bytes")
        for name in validation
    }
    validation_setup = _positive(
        costs["validation_setup_seconds"], "first Alpha validation setup"
    )
    assessment_costs = {
        opponent: _positive(costs["assessment_seconds"][opponent], opponent)
        for opponent in ("alpha", "beta")
    }
    recipes_by_id = {row["recipe_id"]: row for row in declared["recipes"]}
    captures = int(declared["routine_captures"])
    shortlist = int(declared["maximum_confirmation_actors"])

    def run_cost(identifier: str, steps: int, finalists: bool) -> float:
        """Include one run's numerical work, saves, captures and confirmations."""
        row = timings[identifier]
        rollout = recipes_by_id[identifier]["changes"].get("rollout_length", 32)
        saves = math.ceil(steps / (512 * rollout * 25)) + captures
        checks = (
            captures * validation["alpha80" if finalists else "alpha40"]
            + shortlist * validation["alpha200" if finalists else "alpha100"]
        )
        return (
            row["setup_seconds"]
            + validation_setup
            + steps * row["seconds_per_transition"]
            + saves * row["save_seconds"]
            + captures * row["report_seconds"]
            + checks
        )

    def forecast(tier: int, steps: int) -> float:
        """Bound finalist costs before their result-dependent identity is known."""
        identifiers = [row["recipe_id"] for row in declared["recipes"][:tier]]
        discovery = len(declared["discovery_seeds"]) * sum(
            run_cost(identifier, steps, False) for identifier in identifiers
        )
        finalist = len(declared["finalist_seeds"]) * (
            run_cost("c00", 2 * steps, True)
            + max(
                run_cost(identifier, 2 * steps, True)
                for identifier in identifiers
                if identifier != "c00"
            )
        )
        assessment = (
            2
            * len(declared["finalist_seeds"])
            * (assessment_costs["alpha"] + assessment_costs["beta"])
        )
        return discovery + finalist + assessment

    quantum, minimum = int(declared["quantum"]), int(declared["minimum_steps"])
    fixed = declared.get("fixed_discovery_steps")
    if "fixed_discovery_steps" in declared and (
        type(fixed) is not int or fixed < minimum or fixed <= 0 or fixed % quantum
    ):
        raise ValueError(
            "fixed_discovery_steps must be a positive integer at least "
            "minimum_steps and divisible by quantum"
        )
    requested = minimum if fixed is None else cast(int, fixed)
    target = float(declared["target_seconds"])
    margin, reserve = (
        float(declared["timing_margin"]),
        float(declared["report_reserve_seconds"]),
    )
    selected = next(
        (
            int(tier)
            for tier in declared["tiers"]
            if margin * forecast(tier, requested) + reserve <= target
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            "The eight-recipe minimum does not fit the declared time budget"
        )
    if fixed is not None:
        steps = requested
    else:
        low, high = minimum // quantum, minimum // quantum + 1
        while margin * forecast(selected, high * quantum) + reserve <= target:
            high *= 2
        while low + 1 < high:
            middle = (low + high) // 2
            if margin * forecast(selected, middle * quantum) + reserve <= target:
                low = middle
            else:
                high = middle
        steps = low * quantum

    def run_bytes(identifier: str, steps: int, finalists: bool) -> float:
        """Project retained candidates, two recoveries, logs and recorded games."""
        row = timings[identifier]
        rollout = recipes_by_id[identifier]["changes"].get("rollout_length", 32)
        updates = math.ceil(steps / (512 * rollout))
        saves = math.ceil(updates / 25) + captures
        return (
            row["run_bytes"]
            + (captures + 2) * row["checkpoint_bytes"]
            + captures * row["actor_bytes"]
            + updates * row["log_bytes_per_update"]
            + saves * row["checkpoint_metadata_bytes"]
            + captures * validation_bytes["alpha80" if finalists else "alpha40"]
            + shortlist * validation_bytes["alpha200" if finalists else "alpha100"]
        )

    identifiers = [row["recipe_id"] for row in declared["recipes"][:selected]]
    projected_bytes = len(declared["discovery_seeds"]) * sum(
        run_bytes(identifier, steps, False) for identifier in identifiers
    ) + len(declared["finalist_seeds"]) * (
        run_bytes("c00", 2 * steps, True)
        + max(
            run_bytes(identifier, 2 * steps, True)
            for identifier in identifiers
            if identifier != "c00"
        )
        + 2 * (validation_bytes["alpha400"] + validation_bytes["beta200"])
    )
    order: list[Record] = []
    shuffle = random.Random(declared["order_seed"])
    for seed in declared["discovery_seeds"]:
        block = [row["recipe_id"] for row in declared["recipes"][:selected]]
        shuffle.shuffle(block)
        order.extend(
            {
                "name": f"discovery-{identifier}-s{seed}",
                "phase": "discovery",
                "recipe_id": identifier,
                "seed": seed,
                "steps": steps,
            }
            for identifier in block
        )
    return {
        "schema_version": 1,
        "tier": selected,
        "discovery_steps": steps,
        "finalist_steps": 2 * steps,
        "discovery_order": order,
        "forecast_seconds": forecast(selected, steps),
        "reserved_seconds": margin * forecast(selected, steps) + reserve,
        "timing_digest": _digest(costs),
        "declaration_digest": _digest(declared),
        "maximum_validation_games": 1500 * selected + 9600,
        "training_transitions": (3 * selected + 12) * steps,
        "projected_retained_bytes": math.ceil(projected_bytes),
        "minimum_free_bytes": math.ceil(1.25 * projected_bytes + 2_000_000_000),
    }


def rank_recipes(
    rows: Sequence[Mapping[str, Any]],
    *,
    identifiers: Sequence[str],
    seeds: Sequence[int],
) -> list[Record]:
    """Rank only recipes with exactly one completed selected actor for every seed.

    Failed or missing seeds make a recipe ineligible; no successful-only average
    is allowed. Input fields are recipe_id, seed, complete, score,
    mean_kill_difference and env_steps. Duplicate or unknown rows raise ValueError.
    Return descending score/kill means, then earlier mean step and stable ID.
    """
    grouped: dict[str, dict[int, Mapping[str, Any]]] = {
        identifier: {} for identifier in identifiers
    }
    for row in rows:
        identifier, seed = row["recipe_id"], row["seed"]
        if (
            identifier not in grouped
            or seed not in seeds
            or seed in grouped[identifier]
        ):
            raise ValueError(
                "Recipe results contain an unknown or duplicate recipe/seed"
            )
        grouped[identifier][seed] = row
    eligible: list[Record] = []
    for identifier, values in grouped.items():
        if set(values) != set(seeds) or any(
            row.get("complete") is not True for row in values.values()
        ):
            continue
        means: Record = {"recipe_id": identifier, "seeds": list(seeds)}
        for name in ("score", "mean_kill_difference", "env_steps"):
            numbers = [row.get(name) for row in values.values()]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in numbers
            ):
                raise ValueError(
                    "Complete selection results need finite scores, kills and steps"
                )
            means[name] = statistics.mean(cast(list[float], numbers))
        if not 0 <= means["score"] <= 1 or means["env_steps"] <= 0:
            raise ValueError("Recipe selection score or step is outside its range")
        eligible.append(means)
    return sorted(
        eligible,
        key=lambda row: (
            -row["score"],
            -row["mean_kill_difference"],
            row["env_steps"],
            row["recipe_id"],
        ),
    )


def _command(root: Path, action: str, *extra: str) -> list[str]:
    """Build an isolated package command with no shell interpolation."""
    return [
        str(root / ".venv/bin/python"),
        "-I",
        "-m",
        "marl_battlegrounds.training.search",
        action,
        str(root),
        *extra,
    ]


def prepare_search(
    repository: str | Path,
    destination: str | Path,
    *,
    gpu_uuid: str,
    python: str = sys.executable,
) -> Record:
    """Freeze working source, locked dependencies, Alpha panels and commands.

    repository is the current worktree; destination must be new and ignored if
    inside it. gpu_uuid must identify the authorized internal RTX 5090. This
    writes only the new package, performs CPU setup/import checks, and launches
    neither calibration nor scientific training. Failure preserves partial files.
    """
    root = Path(destination).absolute()
    if root.exists():
        raise ValueError("Search package must use a new directory")
    gpu = _launch._gpu(gpu_uuid)
    root.mkdir(parents=True)
    origin = _launch.export_working_source(Path(repository).absolute(), root / "source")
    _launch._install(root / "source", root / ".venv", python)
    runtime = _launch._runtime(root / ".venv/bin/python", root / "source")
    (root / "logs").mkdir()
    subprocess.run(
        _command(root, "initialize"),
        env=_launch._environment(None),
        check=True,
        stdin=subprocess.DEVNULL,
    )
    for name, action in _SCRIPTS.items():
        path = root / f"{name}.sh"
        path.write_text(
            "#!/usr/bin/env bash\n"
            "# Use this frozen study's own interpreter and commands.\n"
            "set -euo pipefail\nexec " + shlex.join(_command(root, action)) + ' "$@"\n'
        )
        path.chmod(0o755)
    if _launch._files(root / "source") != origin["files"]:
        raise ValueError("Package installation changed the frozen source")
    record = {
        "schema_version": 1,
        "package_path": str(root),
        "origin": origin,
        "runtime": runtime,
        "gpu": gpu,
        "declaration_sha256": _launch._hash(root / "declaration.json"),
        "panels": _launch._files(root / "panels"),
        "scripts": {
            f"{name}.sh": _launch._hash(root / f"{name}.sh") for name in _SCRIPTS
        },
    }
    atomic_json(root / _MANIFEST, record)
    return {
        "package": str(root),
        **{name: shlex.join(["bash", str(root / f"{name}.sh")]) for name in _SCRIPTS},
    }


def _initialize(root: Path) -> None:
    """Create package-owned direct panels on CPU before their hashes are frozen."""
    from marl_battlegrounds.training.validation import create_panel

    atomic_json(root / "declaration.json", declaration())
    for name, roots in _ROOTS.items():
        selection_roots = (
            (roots[0] + 10, roots[0] + 11) if name in ("alpha", "beta") else roots
        )
        create_panel(
            opponents=("tdm-beta" if name == "beta" else "tdm-alpha",),
            output_dir=root / "panels" / name,
            roots={"routine": selection_roots[0], "confirmation": selection_roots[1]},
        )


def verify_package(root: Path, *, full: bool = True) -> Record:
    """Reject changed declarations, source, panels, dependencies or hardware.

    full=False omits source hashing and CPU/runtime/device probes. Ordinary
    status never calls this function. No game, recovery or selection is run.
    """
    manifest = _read(root / _MANIFEST)
    if manifest.get("schema_version") != 1 or manifest.get("package_path") != str(root):
        raise ValueError("Search package schema or location changed")
    if (
        _launch._hash(root / "declaration.json") != manifest["declaration_sha256"]
        or _launch._files(root / "panels") != manifest["panels"]
    ):
        raise ValueError("Frozen declaration or panels changed")
    for name, digest in manifest["scripts"].items():
        if (
            name not in {f"{key}.sh" for key in _SCRIPTS}
            or _launch._hash(root / name) != digest
        ):
            raise ValueError("Frozen command changed")
    if full and (
        _launch._files(root / "source") != manifest["origin"]["files"]
        or _launch._runtime(root / ".venv/bin/python", root / "source")
        != manifest["runtime"]
        or _launch._gpu(manifest["gpu"]["uuid"]) != manifest["gpu"]
    ):
        raise ValueError("Frozen source, runtime or GPU changed")
    return manifest


def _environment(root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Use the launch owner's one-GPU environment and package-local cache."""
    return _launch._environment(
        manifest["gpu"]["uuid"],
        memory_fraction=0.85,
        compilation_cache=root / "compilation-cache",
    )


def _live_records(root: Path) -> list[Path]:
    """Find owned live controller or job identities without loading any learner."""
    paths = [
        root / "process.json",
        *root.glob("jobs/*/process.json"),
        *root.glob("calibration/*/process.json"),
    ]
    return [path for path in paths if path.is_file() and _launch._alive(_read(path))]


def _clean_job(job: Path) -> bool:
    """Require an exited successful supervisor with no stop or cleanup error."""
    record = _optional(job / "process.json")
    return (
        record.get("state") == "exited"
        and record.get("exit_code") == 0
        and not any(
            record.get(key)
            for key in ("stop_reason", "stop_signal", "error", "cleanup_error")
        )
        and not record.get("cleanup", {}).get("remaining_pids")
    )


def _run_job(
    root: Path, job: Path, *, mode: str, deadline: float, resume: bool = False
) -> None:
    """Run one owned supervisor and require its clean worker completion.

    deadline is the original absolute numerical cutoff. Each worker runs in its
    own process group under the shared launch owner. No failed job is retried
    here. Caller interruption asks the supervisor to clean up before returning.
    """
    if time.time() >= deadline:
        raise TimeoutError("Original numerical deadline reached")
    argv = _command(
        root,
        "job",
        "--job",
        str(job),
        "--mode",
        mode,
        "--deadline",
        str(deadline),
        *(["--resume-worker"] if resume else []),
    )
    with (job / "worker.log").open("ab", buffering=0) as log:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_launch._environment(None),
        )
        append_jsonl(
            job / "attempts.jsonl",
            {
                "event": "started",
                "time_utc": utc_now(),
                "mode": mode,
                "resume": resume,
                "process": process_identity(child.pid),
                "deadline": deadline,
            },
            durable=True,
        )
        try:
            code = child.wait(timeout=max(1.0, deadline - time.time()) + 30)
        except BaseException:
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=30)
            raise
    process = _read(job / "process.json")
    append_jsonl(
        job / "attempts.jsonl",
        {
            "event": "finished",
            "time_utc": utc_now(),
            "mode": mode,
            "exit_code": code,
            "process_record": process,
        },
        durable=True,
    )
    if code != 0 or not _clean_job(job):
        raise RuntimeError(f"Worker failed: {job.name}; inspect {job / 'worker.log'}")


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a declaration once or require byte-independent JSON equality."""
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"Frozen declaration changed: {path}")
    else:
        atomic_json(path, value)


def calibrate(root: Path) -> Record:
    """Run bounded engineering probes, then freeze budgets before learning.

    Reuses completed, verified calibration records after an explicit rerun of
    this command. An interrupted timing probe requires a new package: resuming
    its learner would hide part of its cold setup cost. Calibration is never a
    scientific training prefix.
    The one-hour calibration allowance starts on its first invocation and does
    not consume or reset the later study clock. Existing scientific work blocks
    calibration. Insufficient timing or disk capacity prevents launch.
    """
    manifest = verify_package(root)
    with (root / ".launch.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if _live_records(root) or (root / "study.json").exists():
            raise RuntimeError("An active or started study cannot be recalibrated")
        declared = _read(root / "declaration.json")
        clock = _optional(root / "calibration_clock.json") or {
            "started_at_seconds": time.time()
        }
        _write_once(root / "calibration_clock.json", clock)
        deadline = clock["started_at_seconds"] + declared["calibration_limit_seconds"]
        costs: Record = {
            "recipes": {},
            "validation_seconds": {},
            "validation_bytes": {},
            "assessment_seconds": {},
        }
        for recipe in declared["recipes"]:
            job = root / "calibration" / recipe["recipe_id"]
            job.mkdir(parents=True, exist_ok=True)
            length = recipe["changes"].get("rollout_length", 32)
            config = run_config(
                declared,
                recipe,
                seed=declared["calibration_seed"],
                steps=(6 + math.ceil(300 / length)) * 512 * length,
                panel=None,
            )
            _write_once(job / "config.json", config)
            if (job / "process.json").exists() and not _clean_job(job):
                raise RuntimeError(
                    "An interrupted calibration cannot establish cold setup cost; "
                    "prepare a new package before remeasuring"
                )
            if not (job / "result.json").exists():
                _run_job(
                    root,
                    job,
                    mode="calibration",
                    deadline=deadline,
                )
            row = _read(job / "result.json")
            if row.get("config_digest") != _digest(config):
                raise ValueError("Calibration result uses another configuration")
            process = _read(job / "process.json")
            startup = _positive(
                row["training_started_at_seconds"]
                - datetime.fromisoformat(process["started_at"]).timestamp(),
                "training worker process startup",
            )
            row["trainer_setup_seconds"] = row["setup_seconds"]
            row["process_startup_seconds"] = startup
            row["setup_seconds"] += startup
            costs["recipes"][recipe["recipe_id"]] = row
            if recipe["recipe_id"] == "c00":
                costs["validation_seconds"] = row["validation_seconds"]
                costs["validation_bytes"] = row["validation_bytes"]
                costs["validation_setup_seconds"] = row["validation_setup_seconds"]
        actor = _read(root / "calibration/c00/run/status.json")["final_actor"]
        for opponent, pairs in declared["assessment_pairs"].items():
            job = root / "calibration" / f"assessment-{opponent}"
            job.mkdir(exist_ok=True)
            _write_once(
                job / "request.json",
                {
                    "actor_path": actor,
                    "expected_actor": _actor_identity(actor),
                    "opponent": opponent,
                    "seed_pairs": pairs,
                    "root_seed": 19_046_090 + (opponent == "beta"),
                    "scope": "Engineering fresh-process assessment cost only",
                },
            )
            request = _read(job / "request.json")
            _check_assessment_actor(request)
            if (job / "process.json").exists() and not _clean_job(job):
                raise RuntimeError("Interrupted assessment timing needs a new package")
            if not (job / "result.json").exists():
                _run_job(root, job, mode="assessment", deadline=deadline)
            _check_assessment_summary(request, _read(job / "result.json"))
            process = _read(job / "process.json")
            costs["assessment_seconds"][opponent] = (
                datetime.fromisoformat(process["finished_at"])
                - datetime.fromisoformat(process["started_at"])
            ).total_seconds() + 1.0
        atomic_json(root / "calibration.json", costs)
        try:
            budgets = resolve_budgets(declared, costs)
        except ValueError as error:
            atomic_json(
                root / "feasibility.json",
                {"complete": False, "reason": str(error), "time_utc": utc_now()},
            )
            raise
        free = shutil.disk_usage(root).free
        if free < budgets["minimum_free_bytes"]:
            raise ValueError(
                "Insufficient free disk bytes: "
                f"need {budgets['minimum_free_bytes']}, have {free}"
            )
        _write_once(root / "budgets.json", budgets)
        _write_once(
            root / "qualified.json",
            {
                "budget_digest": _digest(budgets),
                "timing_digest": _digest(costs),
                "source_files_digest": _digest(manifest["origin"]["files"]),
            },
        )
        return budgets


def _check_budget(root: Path) -> tuple[Record, Record]:
    """Bind immutable budgets to their measured costs and prepared declaration."""
    declared, costs, budgets = (
        _read(root / "declaration.json"),
        _read(root / "calibration.json"),
        _read(root / "budgets.json"),
    )
    qualified = _read(root / "qualified.json")
    if (
        resolve_budgets(declared, costs) != budgets
        or qualified["budget_digest"] != _digest(budgets)
        or qualified["timing_digest"] != _digest(costs)
        or qualified["source_files_digest"]
        != _digest(_read(root / _MANIFEST)["origin"]["files"])
    ):
        raise ValueError("Frozen study budgets changed")
    return declared, budgets


def start_search(root: Path, *, resume: bool = False) -> Record:
    """Launch a detached controller and return immediately with its identity.

    Resume retains the first launch's deadline, recipes and seeds. Live owned
    processes block duplicate starts. stdin is closed and logs go to the package.
    Expired resume may rebuild reports but never starts more scientific work.
    """
    verify_package(root)
    declared, budgets = _check_budget(root)
    with (root / ".launch.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if _live_records(root):
            raise RuntimeError("This study still has an active owned process")
        previous = _optional(root / "study.json")
        if bool(previous) != resume:
            raise ValueError("Use resume for an existing study, start for a new study")
        if not previous:
            saved_clock = _optional(root / "launch_time.json")
            started = saved_clock.get("started_at_seconds", time.time())
            clock = {
                "started_at_seconds": started,
                "numerical_deadline": started + declared["numerical_stop_seconds"],
                "hard_deadline": started + declared["hard_stop_seconds"],
                "budget_digest": _digest(budgets),
            }
            _write_once(root / "launch_time.json", clock)
            atomic_json(
                root / "study.json",
                {
                    "schema_version": 1,
                    "status": "starting",
                    "phase": "discovery",
                    "jobs": {},
                    **clock,
                },
            )
        _check_clock(root, _read(root / "study.json"), declared, budgets)
        command = _command(root, "supervise")
        with (root / "logs/experiment.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=_launch._environment(None),
                cwd=root / "source",
            )
        record = {
            "schema_version": 1,
            "state": "starting",
            "process": process_identity(child.pid),
            "command": command,
            "mode": "resume" if resume else "start",
            "started_at": utc_now(),
        }
        atomic_json(root / "process.json", record)
        return record


def _check_clock(
    root: Path,
    study: Mapping[str, Any],
    declared: Mapping[str, Any],
    budgets: Mapping[str, Any],
) -> None:
    """Reject changed clocks or budget bindings before work or recovery."""
    clock = _read(root / "launch_time.json")
    started = _positive(clock.get("started_at_seconds"), "original launch time")
    expected = {
        "started_at_seconds": started,
        "numerical_deadline": started + declared["numerical_stop_seconds"],
        "hard_deadline": started + declared["hard_stop_seconds"],
        "budget_digest": _digest(budgets),
    }
    if clock != expected or any(
        study.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Original study clock or budgets changed")


def search_status(root: Path) -> Record:
    """Read progress without starting JAX, recovering records or changing files."""
    study = _optional(root / "study.json")
    active = study.get("active_job")
    return {
        "package": str(root),
        "study": study,
        "owned_live_records": [str(path) for path in _live_records(root)],
        "run": _optional(root / "jobs" / active / "run/status.json") if active else {},
        "remaining_seconds": max(
            0.0, study.get("numerical_deadline", time.time()) - time.time()
        ),
    }


def stop_search(root: Path) -> Record:
    """Request shutdown from exact live owners; never kill a saved group number."""
    signalled: list[int] = []
    for path in _live_records(root):
        record = _read(path)
        owner = record.get("process")
        if isinstance(owner, dict) and _launch._alive({"process": owner}):
            pid = int(cast(Record, owner)["pid"])
            os.kill(pid, signal.SIGTERM)
            signalled.append(pid)
    return {**search_status(root), "stop_requested_pids": signalled}


def _selected_row(job: Path, case: Mapping[str, Any]) -> Record:
    """Bind complete exact-budget evidence to its selected and final exports.

    Read the run status, selection and verified actor file hashes. Bind the
    selected export to its scored learner boundary and the final export to the
    last learner description, without restoring learner arrays. A different
    run, seed, inference setting, weight set or boundary raises ValueError.
    Return both verified export identities so later assessment cannot silently
    substitute another valid artifact at the same path. No files change.
    """
    status = _read(job / "run/status.json")
    selected = _read(job / "run/selection.json")
    if (
        status.get("status") != "complete"
        or status.get("env_steps") != case["steps"]
        or selected.get("complete") is not True
        or not status.get("selected_actor")
    ):
        raise ValueError("A study run lacks complete exact-budget selection evidence")
    config = _read(job / "config.json")
    actor = _actor_identity(status["selected_actor"])
    from marl_battlegrounds.training.checkpoints import read_checkpoint_description

    final = _actor_identity(status["final_actor"])
    boundary = read_checkpoint_description(status["latest_checkpoint"])
    if any(
        identity["run_id"] != status["run_id"]
        or identity["seed"] != case["seed"]
        or identity["input_scale"] != config["ppo"]["input_scale"]
        or identity["spawn_frame"] != config["ppo"]["spawn_frame"]
        for identity in (actor, final)
    ):
        raise ValueError("Selected or final actor differs from its training run")
    if any(
        actor[key] != selected[key]
        for key in ("actor_digest", "checkpoint_id", "env_steps")
    ):
        raise ValueError("Selected actor differs from its scored checkpoint")
    if (
        boundary["kind"] != "learner"
        or boundary["metadata"]["run_id"] != status["run_id"]
        or boundary["metadata"]["config"] != config
        or final["checkpoint_id"] != boundary["checkpoint_id"]
        or final["weight_digest"] != boundary["actor_digest"]
        or final["env_steps"] != boundary["counters"]["env_steps"]
        or final["env_steps"] != case["steps"]
    ):
        raise ValueError("Final actor differs from its completed learner boundary")
    return {
        **selected,
        "complete": True,
        "recipe_id": case["recipe_id"],
        "seed": case["seed"],
        "name": case["name"],
        "actor_path": status["selected_actor"],
        "actor_identity": actor,
        "final_actor": status["final_actor"],
        "final_actor_identity": final,
        "training_budget": case["steps"],
        "run_dir": str(job / "run"),
    }


def _actor_identity(path: str | Path) -> Record:
    """Read verified export bytes and the shared originating-checkpoint identity.

    The validation owner distinguishes the export's artifact_id from the
    learner's checkpoint_id. Return that full record without restoring arrays
    or changing files; missing or changed payloads raise its normal errors.
    Shared numerical imports use CPU array placement and restore the caller's
    device default even on error. Direct calls may still discover other JAX
    backends; dedicated host commands select CPU before importing JAX.
    """
    import jax

    with jax.default_device(jax.devices("cpu")[0]):
        from marl_battlegrounds.training.validation import _artifact

        return _artifact(path)


def _check_assessment_actor(request: Mapping[str, Any]) -> None:
    """Reject changed actor bytes before assessment creates or recovers a writer.

    request must contain actor_path and the expected_actor frozen at selection.
    Recheck the complete saved identity, including origin and artifact IDs.
    No evaluation or file mutation occurs; mismatch raises ValueError.
    """
    if _actor_identity(request["actor_path"]) != request["expected_actor"]:
        raise ValueError("Assessment actor differs from the frozen selection")


def _check_assessment_summary(
    request: Mapping[str, Any], summary: Mapping[str, Any]
) -> None:
    """Require a complete summary for the frozen actor and exact request fields.

    Check inference, originating checkpoint and experience before saving or
    accepting a result. Shared request/summary fields must also agree so a
    mapping merge cannot conceal conflicting evidence. Mismatch raises
    ValueError; this check reads its arguments only.
    """
    expected = request["expected_actor"]
    if (
        summary.get("complete") is not True
        or any(
            summary.get(key) != expected[key]
            for key in ("actor_digest", "checkpoint_id", "env_steps")
        )
        or any(summary[key] != request[key] for key in summary.keys() & request.keys())
    ):
        raise ValueError("Assessment summary differs from the frozen actor or request")


def _assess(root: Path, job: Path) -> None:
    """Assess one frozen actor through shared validation and publish checked evidence.

    Read job/request.json, verify current export bytes before validation can
    create or recover its writer, then check the returned complete summary.
    Only after both checks write job/result.json with its request digest.
    root supplies the frozen opponent panels. All games use batch 32 and the
    declared pair count/root; failures leave any existing evidence intact.
    """
    from marl_battlegrounds.training.validation import validate_checkpoint

    request = _read(job / "request.json")
    _check_assessment_actor(request)
    summary = validate_checkpoint(
        request["actor_path"],
        root / "panels" / request["opponent"],
        output_dir=job / "validation",
        purpose="assessment",
        seed_pairs=request["seed_pairs"],
        root_seed=request["root_seed"],
        num_envs=32,
    )
    _check_assessment_summary(request, summary)
    atomic_json(
        job / "result.json",
        {**request, **summary, "request_digest": _digest(request)},
    )


def _numerical_failure(job: Path) -> Record:
    """Read a durable failed update only when its config and worker agree.

    An empty object means no numerical failure was saved. A mismatched record
    raises ValueError before any restart; failed seeds are never retried after
    an interruption between worker exit and controller result publication.
    """
    failure = _optional(job / "failure.json")
    if failure and (
        failure.get("kind") != "numerical"
        or failure.get("config_digest") != _digest(_read(job / "config.json"))
        or not failure.get("worker")
        or failure["worker"] != _optional(job / "process.json").get("trainer")
    ):
        raise ValueError("Numerical failure does not match its config and worker")
    return failure


def _case(root: Path, study: Record, declared: Record, case: Record) -> Record:
    """Run or resume one frozen case, preserving numerical failure as evidence."""
    job = root / "jobs" / case["name"]
    job.mkdir(parents=True, exist_ok=True)
    recipe = next(
        row for row in declared["recipes"] if row["recipe_id"] == case["recipe_id"]
    )
    phase = case["phase"]
    config = run_config(
        declared,
        recipe,
        seed=case["seed"],
        steps=case["steps"],
        panel=str(root / "panels" / phase / "panel.json"),
        finalists=phase == "finalists",
    )
    _write_once(job / "case.json", case)
    _write_once(job / "config.json", config)
    failure = _numerical_failure(job)
    existing = _optional(job / "result.json")
    if existing:
        if existing.get("complete") is True:
            if not _clean_job(job):
                raise ValueError("Completed case lacks a clean worker exit")
            current = _selected_row(job, case)
            if existing != current:
                raise ValueError("Completed case selection changed")
        return existing
    study.update(active_job=case["name"], phase=phase, status="running")
    atomic_json(root / "study.json", study)
    if failure:
        result = {**case, "complete": False, "failure": failure}
    else:
        try:
            if not _clean_job(job):
                _run_job(
                    root,
                    job,
                    mode="train",
                    deadline=study["numerical_deadline"],
                    resume=(job / "run/latest_checkpoint.json").exists(),
                )
            result = _selected_row(job, case)
        except RuntimeError:
            failure = _numerical_failure(job)
            if not failure:
                raise
            result = {**case, "complete": False, "failure": failure}
    atomic_json(job / "result.json", result)
    study["jobs"][case["name"]] = {
        "complete": result["complete"],
        "result_path": str(job / "result.json"),
    }
    atomic_json(root / "study.json", study)
    return result


def execute_search(root: Path) -> Record:
    """Complete the frozen study phases through shared workers and saved evidence.

    This is the detached controller's entry point. Completed cases and assessment
    games are reused on explicit resume. All required seed results precede recipe
    choices; assessment is downstream of the immutable winner and cannot feed
    back. Infrastructure failures stop the study and keep their full error.
    """
    verify_package(root)
    declared, budgets = _check_budget(root)
    study = _read(root / "study.json")
    _check_clock(root, study, declared, budgets)
    try:
        if study.get("status") == "complete":
            return study
        if time.time() >= study["numerical_deadline"]:
            study.update(
                status="incomplete", error="Original numerical deadline reached"
            )
            return study
        discovery = [
            _case(root, study, declared, case) for case in budgets["discovery_order"]
        ]
        ranking = rank_recipes(
            discovery,
            identifiers=[
                row["recipe_id"] for row in declared["recipes"][: budgets["tier"]]
            ],
            seeds=declared["discovery_seeds"],
        )
        candidates = [row for row in ranking if row["recipe_id"] != "c00"]
        if not candidates or not any(row["recipe_id"] == "c00" for row in ranking):
            raise ValueError(
                "Reference and a challenger need all three discovery seeds"
            )
        challenger = candidates[0]["recipe_id"]
        _write_once(
            root / "discovery_selection.json",
            {
                "ranking": ranking,
                "challenger": challenger,
                "source_digest": _digest(discovery),
            },
        )
        finalist_cases: list[Record] = []
        for index, seed in enumerate(declared["finalist_seeds"]):
            order = ("c00", challenger) if index % 2 == 0 else (challenger, "c00")
            finalist_cases.extend(
                {
                    "name": f"finalists-{identifier}-s{seed}",
                    "phase": "finalists",
                    "recipe_id": identifier,
                    "seed": seed,
                    "steps": budgets["finalist_steps"],
                }
                for identifier in order
            )
        finalists = [_case(root, study, declared, case) for case in finalist_cases]
        ranking = rank_recipes(
            finalists, identifiers=("c00", challenger), seeds=declared["finalist_seeds"]
        )
        if len(ranking) != 2:
            raise ValueError("Both finalists need all three fresh completed seeds")
        selection = {
            "ranking": ranking,
            "winner": ranking[0]["recipe_id"],
            "actors": finalists,
            "source_digest": _digest(finalists),
            "assessment_used_for_selection": False,
        }
        _write_once(root / "final_selection.json", selection)
        for row in finalists:
            for opponent, pairs in declared["assessment_pairs"].items():
                job = root / "jobs" / f"assessment-{opponent}-{row['name']}"
                job.mkdir(parents=True, exist_ok=True)
                request = {
                    "actor_path": row["actor_path"],
                    "expected_actor": row["actor_identity"],
                    "recipe_id": row["recipe_id"],
                    "seed": row["seed"],
                    "opponent": opponent,
                    "seed_pairs": pairs,
                    "root_seed": declared["roots"][opponent][0],
                    "selection_digest": _digest(selection),
                }
                _write_once(job / "request.json", request)
                _check_assessment_actor(request)
                study.update(active_job=job.name, phase="assessment", status="running")
                atomic_json(root / "study.json", study)
                if not (job / "result.json").exists() or not _clean_job(job):
                    _run_job(
                        root,
                        job,
                        mode="assessment",
                        deadline=study["numerical_deadline"],
                    )
                result = _read(job / "result.json")
                _check_assessment_summary(request, result)
                if result.get("complete") is not True or result.get(
                    "request_digest"
                ) != _digest(request):
                    raise ValueError(
                        "Assessment is incomplete or belongs to another selection"
                    )
                study["jobs"][job.name] = {
                    "complete": True,
                    "result_path": str(job / "result.json"),
                }
        study.update(status="complete", phase="complete", active_job=None)
        return study
    except BaseException as error:
        study.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        study["updated_at"] = utc_now()
        atomic_json(root / "study.json", study)
        report_search(root)


def _directory_bytes(root: Path) -> int:
    """Count saved regular-file bytes without following directory links."""
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _worker(root: Path, job: Path, *, mode: str, resume: bool = False) -> None:
    """Execute numerical work only inside its supervised one-GPU child."""
    import jax

    from marl_battlegrounds.training.runner import read_config, train
    from marl_battlegrounds.training.validation import validate_checkpoint

    if (
        jax.default_backend() != "gpu"
        or len(cast(list[object], jax.devices("gpu"))) != 1
    ):
        raise ValueError("Search workers need exactly the declared GPU")
    if mode == "assessment":
        _assess(root, job)
        return
    config = read_config(job / "config.json")
    training_started_at = time.time()
    started = time.monotonic()
    try:
        result = train(
            None if resume else config,
            output_dir=None if resume else job / "run",
            resume_from=_launch._checkpoint(job, None) if resume else None,
        )
    except RuntimeError as error:
        from marl_battlegrounds.training.learner import (
            LEARNER_ERROR_NONFINITE_BATCH,
            LEARNER_ERROR_NONFINITE_UPDATE,
        )

        # Invalid actions/boundaries/history are infrastructure errors, not
        # evidence that a hyperparameter recipe learned poorly.
        if str(error) in {
            f"Learner update failed: reason {reason}"
            for reason in (
                LEARNER_ERROR_NONFINITE_BATCH,
                LEARNER_ERROR_NONFINITE_UPDATE,
            )
        }:
            atomic_json(
                job / "failure.json",
                {
                    "kind": "numerical",
                    "error": str(error),
                    "time_utc": utc_now(),
                    "config_digest": _digest(_read(job / "config.json")),
                    "worker": process_identity(),
                },
            )
        raise
    if mode != "calibration":
        return
    rows = read_jsonl(job / "run/training_updates.jsonl")
    expected = 6 + math.ceil(300 / config.ppo.rollout_length)
    if len(rows) != expected:
        raise ValueError("Calibration lacks its exact cold/warm/measured updates")
    samples = [
        _positive(row["collection_update_seconds"], "calibration update")
        for row in rows[-5:]
    ]
    events = read_jsonl(job / "run/run_events.jsonl")
    saves = [
        _positive(row["seconds"], "checkpoint save")
        for row in events
        if row.get("event") == "checkpoint_saved"
    ]
    reports = [
        _positive(row["seconds"], "report")
        for row in events
        if row.get("event") == "reports_written"
    ]
    exposure = _read(job / "run/exposure.json")
    if sum(exposure["starts_by_episode_stage"]) <= config.num_envs or not sum(
        exposure["starts_by_opponent"][2:]
    ):
        raise ValueError("Calibration missed episode resets or historical opponents")
    measured: Record = {
        "config_digest": _digest(_read(job / "config.json")),
        "training_started_at_seconds": training_started_at,
        "seconds_per_transition": statistics.median(samples)
        / (config.num_envs * config.ppo.rollout_length),
        "warm_samples_seconds": samples,
        "setup_seconds": max(
            0.001,
            time.monotonic()
            - started
            - sum(row["collection_update_seconds"] for row in rows),
        )
        + rows[0]["collection_update_seconds"],
        "save_seconds": statistics.mean(saves),
        "report_seconds": statistics.mean(reports),
        "run_bytes": _directory_bytes(job / "run"),
        "checkpoint_bytes": max(
            _directory_bytes(path)
            for path in (job / "run/checkpoints").iterdir()
            if path.is_dir()
        ),
        "checkpoint_metadata_bytes": max(
            _directory_bytes(path)
            - _directory_bytes(path / "actor")
            - _directory_bytes(path / "state")
            for path in (job / "run/checkpoints").iterdir()
            if path.is_dir()
        ),
        "actor_bytes": max(
            _directory_bytes(path)
            for path in (job / "run/actors").iterdir()
            if path.is_dir()
        ),
        "log_bytes_per_update": sum(
            (job / "run" / name).stat().st_size
            for name in ("training_updates.jsonl", "run_events.jsonl")
        )
        / len(rows),
        "memory": _read(job / "run/status.json").get("memory"),
        "scope": "Engineering timing only; excluded from scientific results",
    }
    if job.name == "c00":
        from importlib import import_module

        from marl_battlegrounds.training.checkpoints import artifact_identity

        execution = import_module("marl_battlegrounds.evaluation.evaluate")
        chunk = cast(Any, execution)._jax_system_chunk
        actors = list((job / "run/actors").iterdir())
        changed_actor = next(
            path
            for path in actors
            if path != result.final_actor
            and artifact_identity(path)["actor_digest"]
            != artifact_identity(result.final_actor)["actor_digest"]
        )
        validation: Record = {}
        validation_bytes: Record = {}
        cold: Record = {}
        reuse: Record = {}
        for opponent, games in (
            ("alpha", 40),
            ("alpha", 80),
            ("alpha", 100),
            ("alpha", 200),
            ("alpha", 400),
            ("beta", 200),
        ):
            identifier = f"{opponent}{games}"
            counts: list[int] = []
            for repeat, actor in enumerate((result.final_actor, changed_actor)):
                start = time.monotonic()
                output = job / f"timing-{identifier}-{repeat}"
                validate_checkpoint(
                    actor,
                    root / "panels" / opponent,
                    output_dir=output,
                    purpose="assessment",
                    seed_pairs=games // 10,
                    root_seed=19_046_010
                    + games
                    + (1000 if opponent == "beta" else 0)
                    + repeat,
                    num_envs=32,
                )
                (cold if repeat == 0 else validation)[identifier] = (
                    time.monotonic() - start
                )
                validation_bytes[identifier] = max(
                    validation_bytes.get(identifier, 0), _directory_bytes(output)
                )
                counts.append(chunk._cache_size())
            if counts[0] != counts[1] or counts[0] == 0:
                raise ValueError(
                    "Changing validation weights caused a new compiled program"
                )
            reuse[identifier] = counts
        measured["validation_seconds"] = validation
        measured["validation_bytes"] = validation_bytes
        measured["validation_first_seconds"] = cold
        measured["validation_setup_seconds"] = max(
            0.001, cold["alpha40"] - validation["alpha40"]
        )
        measured["validation_compilation_cache_counts"] = reuse
        measured["validation_changed_actor"] = str(changed_actor)
    atomic_json(job / "result.json", measured)


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a derived flat CSV atomically; raw experiment records stay unchanged."""
    columns = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
    )
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def report_search(root: Path) -> Record:
    """Build factual study tables and shared plots from existing records only.

    Per-seed rows, failure records and assessment are kept separate. The shared
    training analysis owns learning plots and map cells. This function never
    loads an actor, starts a game or changes a selection. Missing data stays
    missing. A selected recipe is called complete only after required assessment.
    """
    destination = root / "reports"
    destination.mkdir(exist_ok=True)
    cases: list[Record] = []
    assessment: list[Record] = []
    runs: dict[str, list[Path]] = {}
    exposure: list[Record] = []
    assessment_cells: list[Record] = []
    for job in sorted((root / "jobs").glob("*")) if (root / "jobs").exists() else []:
        if not job.is_dir():
            continue
        request = _optional(job / "case.json")
        result = _optional(job / "result.json")
        if request:
            status = _optional(job / "run/status.json")
            cases.append(
                {
                    **request,
                    **result,
                    "status": status.get("status", "not_started"),
                    "elapsed_seconds": status.get("elapsed_seconds"),
                    "wall_seconds": status.get("wall_seconds"),
                    "failure": str(result.get("failure", status.get("error", ""))),
                }
            )
            if (job / "run/run_details.json").exists():
                group = f"{request['phase']}-{request['recipe_id']}"
                runs.setdefault(group, []).append(job / "run")
                exposure.append(
                    {
                        "name": job.name,
                        "exposure": _optional(job / "run/exposure.json"),
                        "memory": status.get("memory"),
                    }
                )
        elif result:
            assessment.append({"name": job.name, **result})
            assessment_cells.extend(
                {
                    "name": job.name,
                    "recipe_id": result.get("recipe_id"),
                    "seed": result.get("seed"),
                    "opponent": result.get("opponent"),
                    **cell,
                }
                for cell in result.get("cells", [])
            )
    _csv(destination / "recipes_and_seeds.csv", cases)
    _csv(destination / "assessment.csv", assessment)
    _csv(destination / "assessment_cells.csv", assessment_cells)
    atomic_json(destination / "costs_and_exposure.json", exposure)
    analysis_result: Record = {}
    if runs:
        from marl_battlegrounds.training.analysis import analyze

        analysis_result = {
            group: analyze(paths, output_dir=destination / group)
            for group, paths in runs.items()
        }
    declared = _read(root / "declaration.json")
    recipe_rows: list[Record] = []
    for phase, seeds in (
        ("discovery", declared["discovery_seeds"]),
        ("finalists", declared["finalist_seeds"]),
    ):
        phase_rows = [row for row in cases if row["phase"] == phase]
        for recipe in declared["recipes"]:
            identifier = recipe["recipe_id"]
            rows = [row for row in phase_rows if row["recipe_id"] == identifier]
            if not rows:
                continue
            ranked = rank_recipes(rows, identifiers=[identifier], seeds=seeds)
            recipe_rows.append(
                {
                    "phase": phase,
                    "recipe_id": identifier,
                    "required_seeds": len(seeds),
                    "completed_seeds": sum(row.get("complete") is True for row in rows),
                    "eligible": bool(ranked),
                    **(ranked[0] if ranked else {}),
                    "training_seed_score_sd": statistics.stdev(
                        float(row["score"]) for row in rows
                    )
                    if ranked
                    else None,
                }
            )
    _csv(destination / "recipe_summary.csv", recipe_rows)
    study = _optional(root / "study.json")
    selection = _optional(root / "final_selection.json")
    complete = study.get("status") == "complete"
    summary = (
        "# Recurrent MAPPO Search\n\n"
        + f"Status: {study.get('status', 'not_started')}.\n\n"
    )
    summary += f"Frozen recipe choice: {selection.get('winner', 'Not selected')}. " + (
        "All required phases completed.\n\n"
        if complete
        else "Required study phases are incomplete; no completed winner is claimed.\n\n"
    )
    summary += (
        "This finite search tests declared recipes. Alpha is used in training "
        "and selection. Beta is a related diagnostic, not an untouched opponent. "
        "Assessment games never change selection. Game intervals describe fixed "
        "actors; the three training seeds show a separate and small sample of "
        "training variation. No global optimum or tactical competence is implied.\n\n"
    )
    summary += (
        "Exact methods and all defaults are in declaration.json; budgets.json "
        "records timing-only budget choice. Each job retains configurations, "
        "actor identities, recorded games and failures. The shared learning "
        "curves show native outcomes alongside training losses; "
        "costs_and_exposure.json preserves pin starts and transitions separately.\n"
    )
    (destination / "study_summary.md").write_text(summary)
    return {
        "cases": len(cases),
        "assessments": len(assessment),
        "complete": complete,
        "reports": str(destination),
        "analysis": analysis_result,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse one study command and call its shared implementation.

    argv is a command argument list, or None for the process arguments. Only
    dedicated module CLI host processes select CPU before numerical imports.
    Imported calls leave the caller's environment unchanged; numerical workers
    keep their explicit GPU environment. Return the command's exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "initialize",
            "calibrate",
            "start",
            "resume",
            "status",
            "stop",
            "report",
            "supervise",
            "control",
            "job",
            "worker",
        ),
    )
    parser.add_argument("package", type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--mode", choices=("train", "calibration", "assessment"))
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--resume-worker", action="store_true")
    args = parser.parse_args(argv)
    if __name__ == "__main__" and args.action != "worker":
        os.environ["JAX_PLATFORMS"] = "cpu"
    root = args.package.absolute()
    result: object = None
    if args.action == "prepare":
        if not args.gpu_uuid:
            parser.error("prepare requires --gpu-uuid")
        result = prepare_search(
            args.repository, root, gpu_uuid=args.gpu_uuid, python=args.python
        )
    elif args.action == "initialize":
        _initialize(root)
    elif args.action == "calibrate":
        result = calibrate(root)
    elif args.action in ("start", "resume"):
        result = start_search(root, resume=args.action == "resume")
    elif args.action == "status":
        result = search_status(root)
    elif args.action == "stop":
        result = stop_search(root)
    elif args.action == "report":
        result = report_search(root)
    elif args.action == "supervise":
        with (root / ".launch.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            clock = _read(root / "launch_time.json")
        return _launch.supervise_command(
            root,
            _command(root, "control"),
            env=_launch._environment(None),
            deadline_at=clock["hard_deadline"]
            - _launch.cleanup_reserve_seconds(
                stop_grace_seconds=_launch._NESTED_STOP_TIMEOUT_SECONDS
            ),
            stop_grace_seconds=_launch._NESTED_STOP_TIMEOUT_SECONDS,
        )
    elif args.action == "control":
        result = execute_search(root)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] == "complete" else 1
    elif args.action in ("job", "worker"):
        if args.job is None or args.mode is None:
            parser.error("job and worker require --job and --mode")
        if args.action == "worker":
            _worker(root, args.job, mode=args.mode, resume=args.resume_worker)
        else:
            command = _command(
                root,
                "worker",
                "--job",
                str(args.job),
                "--mode",
                args.mode,
                *(["--resume-worker"] if args.resume_worker else []),
            )
            return _launch.supervise_command(
                args.job,
                command,
                env=_environment(root, _read(root / _MANIFEST)),
                deadline_at=None
                if args.deadline is None
                else args.deadline - _launch.cleanup_reserve_seconds(),
            )
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
