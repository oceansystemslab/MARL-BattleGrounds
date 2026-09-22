"""Prepare and run the declared five-minute MAPPO configuration screen.

This host driver owns one fixed twelve-case experiment, its deadline and launch
commands. The ordinary trainer owns learning and recovery; M8 owns evaluation;
the shared analysis module owns reports. Importing this module or reading status
does not import JAX. Run ``python -m marl_battlegrounds.training.screen --help``
for preparation and launch commands. Preparation never starts the experiment.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import shlex
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from marl_battlegrounds.training import _launch
from marl_battlegrounds.training._run_io import (
    atomic_json,
    duration,
    process_identity,
    progress_text,
    read_jsonl,
    utc_now,
)

type Record = dict[str, Any]
_MANIFEST = "screen_package.json"
_TRAINING_SEED = 19_044_601
_CALIBRATION_SEED = 19_044_600
_ORDER_SEED = 19_044_602
_QUANTUM = 131_072
_SCRIPT_ACTIONS = {
    "launch": "start",
    "status": "status",
    "resume": "resume",
    "stop": "stop",
}


def _digest(value: object) -> str:
    """Hash finite canonical JSON without changing the supplied object."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _read_optional(path: Path) -> Record:
    """Read an existing regular JSON object; a missing record means no state."""
    return _launch._read(path) if path.exists() else {}


def screen_declaration() -> Record:
    """Return the fixed recipe and ordered cases without collecting experience.

    The lazy trainer import validates the donor settings. The returned records
    contain all defaults, three distinct seeds and the original time limits.
    Case names are stable B/T identifiers; their order is shuffled once with a
    private Python Random instance, so the shuffle does not consume a global
    random stream. Optional-library imports may initialize their own state.
    The recipe keeps the world spawn frame it was declared with, even though
    PPOConfig now defaults to "left", so the screen reproduces its original
    experiment. This function writes no files.
    """
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training.runner import TrainConfig, config_to_dict

    cases = [
        {"name": f"b{batch}-t{length}", "num_envs": batch, "rollout_length": length}
        for batch in (32, 512, 1024)
        for length in (16, 32, 64, 128)
    ]
    random.Random(_ORDER_SEED).shuffle(cases)
    base = config_to_dict(
        TrainConfig(
            seed=_TRAINING_SEED,
            curriculum=False,
            shaping=True,
            shaping_mode="score_delta",
            shaping_coefficient=0.01,
            ppo=PPOConfig(input_scale=0.01, spawn_frame="world"),
            metrics="none",
            recording=False,
            verbose=True,
        )
    )
    return {
        "schema_version": 1,
        "name": "Five-Minute MAPPO Configuration Screen",
        "cases": cases,
        "base_config": base,
        "training_seed": _TRAINING_SEED,
        "calibration_seed": _CALIBRATION_SEED,
        "order_seed": _ORDER_SEED,
        "order_algorithm": "Python Random.shuffle over ascending B then T",
        "training_seconds_per_case": 300.0,
        "max_elapsed_seconds": 7200.0,
        "numerical_stop_seconds": 7020.0,
        "calibration_limit_seconds": 1200.0,
        "timing_margin": 1.25,
        "report_reserve_seconds": 180.0,
        "validation_seed_pairs": 4,
        "validation_root_seed": 19_043_001,
        "training_maps": list(range(42)),
        "validation_maps": list(range(42, 47)),
        "score_threshold": 20,
        "horizon": 300,
        "memory_fraction": 0.85,
        "qualification": (
            "One-seed development screen; no automatic winner or extension"
        ),
    }


def calibration_config(
    declaration: Mapping[str, Any], case: Mapping[str, Any]
) -> Record:
    """Resolve cold, reset/history warmup and five timed blocks for one shape.

    Return ordinary TrainConfig JSON. The first block is cold, the next
    ceil(300/T) blocks warm at least 300 further ticks, and the last five supply
    the median. All collection is counted as engineering experience, separate
    from the scientific trials. No learner is constructed here.
    """
    batch, length = int(case["num_envs"]), int(case["rollout_length"])
    config = json.loads(json.dumps(declaration["base_config"]))
    config.update(
        seed=declaration["calibration_seed"],
        num_envs=batch,
        total_env_steps=(6 + math.ceil(300 / length)) * batch * length,
        checkpoint_env_steps=[],
        random_diagnostic_seed_pairs=None,
        random_initialization_result=None,
    )
    config["ppo"]["rollout_length"] = length
    return cast(Record, config)


def resolve_budgets(
    declaration: Mapping[str, Any], calibration: Mapping[str, Record], package: Path
) -> Record:
    """Freeze five-minute step budgets using only completed cost measurements.

    calibration maps every declared case name to its measured positive median
    block time. Return complete case settings, an optional shared experience
    anchor and the maximum diagnostic count. Invalid or infeasible timings raise
    ValueError. This pure host calculation neither inspects rewards nor writes
    files. The caller publishes all resolved configurations before any trial.
    """
    target = float(declaration["training_seconds_per_case"])
    if not math.isfinite(target) or target <= 0:
        raise ValueError("Training time target must be finite and positive")
    cases: list[Record] = []
    for shape in declaration["cases"]:
        name = shape["name"]
        raw = calibration[name]["median_update_seconds"]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(raw)
            or raw <= 0
        ):
            raise ValueError(f"Invalid calibration time for {name}")
        updates = 2 * math.floor(target / (2 * raw))
        if updates < 2:
            raise ValueError(f"Five minutes cannot fit two updates for {name}")
        cases.append(
            {
                **shape,
                "updates": updates,
                "total_env_steps": updates
                * shape["num_envs"]
                * shape["rollout_length"],
                "median_update_seconds": float(raw),
                "estimated_training_seconds": updates * raw,
            }
        )
    anchor = min(
        1_048_576, _QUANTUM * (min(row["total_env_steps"] for row in cases) // _QUANTUM)
    )
    passes = 1
    for index, case in enumerate(cases):
        total = case["total_env_steps"]
        points = sorted({total // 2, total} | ({anchor} if anchor else set()))
        config = json.loads(json.dumps(declaration["base_config"]))
        config.update(
            num_envs=case["num_envs"],
            total_env_steps=total,
            checkpoint_env_steps=points,
            checkpoint_interval_updates=math.ceil(case["updates"] / 5),
            random_diagnostic_seed_pairs=declaration["validation_seed_pairs"],
            random_initialization_result=None
            if index == 0
            else str(package / "shared" / "initialization.json"),
        )
        config["ppo"]["rollout_length"] = case["rollout_length"]
        case.update(
            checkpoint_env_steps=points,
            checkpoint_interval_updates=config["checkpoint_interval_updates"],
            config=config,
            config_path=f"configs/{case['name']}.json",
            config_digest=_digest(config),
        )
        passes += len(points)
    return {
        "schema_version": 1,
        "declaration_digest": _digest(declaration),
        "cases": cases,
        "shared_env_steps": anchor or None,
        "diagnostic_passes": passes,
        "diagnostic_games": passes * 40,
    }


def _physical_cpus() -> list[int]:
    """Select up to four allowed physical CPU cores without changing affinity."""
    selected: dict[tuple[str, str], int] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (
                (root / "physical_package_id").read_text().strip(),
                (root / "core_id").read_text().strip(),
            )
        except OSError:
            key = ("unknown", str(cpu))
        selected.setdefault(key, cpu)
        if len(selected) == 4:
            break
    return list(selected.values())


def _content_identity(interpreter: Path, source: Path) -> Record:
    """Resolve content once in the isolated CPU environment, before any launch."""
    command = (
        "import json; from marl_battlegrounds.training import "
        "prepare_training_content; print(json.dumps("
        "prepare_training_content().binding.model_dump(mode='json')))"
    )
    completed = subprocess.run(
        [str(interpreter), "-I", "-c", command],
        cwd=source,
        env=_launch._environment(None),
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    return cast(Record, json.loads(completed.stdout))


def prepare_screen(
    repository: str | Path,
    destination: str | Path,
    *,
    gpu_uuid: str,
    python: str = sys.executable,
) -> Record:
    """Copy the reviewed candidate and build a pinned package without launching.

    repository is the checkout to snapshot, including current public edits.
    destination must be new. gpu_uuid identifies the authorized internal RTX
    5090; python supplies the existing Python 3.14 interpreter to uv. Return
    absolute launch, status, resume and stop commands. Preparation verifies
    source and dependency identities and may install locked packages. It never
    changes Git, starts calibration, or replaces an existing package.
    """
    root = Path(destination).absolute()
    if root.exists():
        raise ValueError("Screen package destination must be new")
    gpu = _launch._gpu(gpu_uuid)
    declaration = screen_declaration()
    root.mkdir(parents=True)
    origin = _launch.export_working_source(Path(repository).absolute(), root / "source")
    atomic_json(root / "declaration.json", declaration)
    _launch._install(root / "source", root / ".venv", python)
    interpreter = root / ".venv/bin/python"
    runtime = _launch._runtime(interpreter, root / "source")
    content = _content_identity(interpreter, root / "source")
    if _launch._files(root / "source") != origin["files"]:
        raise ValueError("Installing the isolated environment changed source bytes")
    for name in (
        "logs",
        "configs",
        "jobs",
        "calibration",
        "shared",
        "reports",
        "compilation-cache",
    ):
        (root / name).mkdir()
    commands: Record = {}
    for name, action in _SCRIPT_ACTIONS.items():
        argv = [
            str(interpreter),
            "-I",
            "-m",
            "marl_battlegrounds.training.screen",
            action,
            str(root),
        ]
        script = root / f"{name}.sh"
        script.write_text(
            "#!/usr/bin/env bash\n"
            "# Use this frozen screen and its own Python environment.\n"
            "set -euo pipefail\nexec " + shlex.join(argv) + ' "$@"\n'
        )
        script.chmod(0o755)
        commands[name] = shlex.join(["bash", str(script)])
    manifest = {
        "schema_version": 1,
        "package_path": str(root),
        "created_at": utc_now(),
        "origin": origin,
        "runtime": runtime,
        "content": content,
        "gpu": gpu,
        "cpu_affinity": _physical_cpus(),
        "declaration_sha256": _launch._hash(root / "declaration.json"),
        "scripts": {
            f"{name}.sh": _launch._hash(root / f"{name}.sh") for name in _SCRIPT_ACTIONS
        },
    }
    atomic_json(root / _MANIFEST, manifest)
    validate_screen_package(root)
    return {"package": str(root), **commands, "log": str(root / "logs/experiment.log")}


def validate_screen_package(package: str | Path) -> Record:
    """Verify frozen source, scripts, declaration and isolated runtime read-only.

    No GPU context or recovery writer is opened. Invalid, relocated or changed
    packages raise ValueError before launch; dependencies are probed on CPU.
    Runtime verification belongs at launch/resume, not each progress display.
    """
    root = Path(package).absolute()
    manifest = _launch._read(root / _MANIFEST)
    if manifest.get("schema_version") != 1 or manifest.get("package_path") != str(root):
        raise ValueError("Screen package identity or location changed")
    if _launch._files(root / "source") != manifest["origin"]["files"]:
        raise ValueError("Screen source bytes changed")
    if _launch._hash(root / "declaration.json") != manifest["declaration_sha256"]:
        raise ValueError("Screen declaration changed")
    expected_scripts = {f"{name}.sh" for name in _SCRIPT_ACTIONS}
    if set(manifest["scripts"]) != expected_scripts:
        raise ValueError("Screen command inventory changed")
    for name, digest in manifest["scripts"].items():
        if _launch._hash(root / name) != digest:
            raise ValueError(f"Screen command changed: {name}")
    if (
        _launch._runtime(root / ".venv/bin/python", root / "source")
        != manifest["runtime"]
    ):
        raise ValueError("Screen dependencies or import location changed")
    if _launch._gpu(manifest["gpu"]["uuid"]) != manifest["gpu"]:
        raise ValueError("Selected internal GPU identity changed")
    return manifest


def _environment(root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Build the declared GPU/cache environment after removing shell overrides."""
    return _launch._environment(
        manifest["gpu"]["uuid"],
        memory_fraction=0.85,
        compilation_cache=root / "compilation-cache",
    )


def _command(root: Path, action: str, *arguments: str) -> list[str]:
    """Return an argument vector using only the package's isolated interpreter."""
    return [
        str(root / ".venv/bin/python"),
        "-I",
        "-m",
        "marl_battlegrounds.training.screen",
        action,
        str(root),
        *arguments,
    ]


def _owned_jobs(root: Path) -> list[Path]:
    """Find job process records without scanning checkpoints or actor payloads."""
    return [
        *root.glob("jobs/*/process.json"),
        *root.glob("calibration/*/attempt-*/process.json"),
    ]


def _validate_clock(root: Path, study: Record, declaration: Record) -> None:
    """Reject changed launch clocks or case membership before starting any job."""
    clock = _launch._read(root / "launch_time.json")
    started = clock.get("started_at_seconds")
    if (
        isinstance(started, bool)
        or not isinstance(started, (int, float))
        or not math.isfinite(started)
    ):
        raise ValueError("Original launch time is invalid")
    expected = {
        "started_at_seconds": started,
        "deadline_at": started + declaration["max_elapsed_seconds"],
        "stop_at": started + declaration["numerical_stop_seconds"],
    }
    if any(study.get(key) != value for key, value in expected.items()):
        raise ValueError("Original screen deadline changed")
    if set(study["cases"]) != {case["name"] for case in declaration["cases"]}:
        raise ValueError("Screen case membership changed")


def start_screen(package: str | Path, *, resume: bool = False) -> Record:
    """Launch a detached controller, or explicitly resume its original deadline.

    Validate the immutable package and block duplicate/live leftover workers.
    The first launch records its deadline before calibration. Resume never
    resets time or budgets. The detached process writes logs/experiment.log;
    return its recorded process identity and command. No shell is used.
    """
    requested_at = time.time()
    root = Path(package).absolute()
    manifest = validate_screen_package(root)
    declaration = _launch._read(root / "declaration.json")
    with (root / ".launch.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in [root / "process.json", *_owned_jobs(root)]:
            if path.exists() and _launch._alive(_launch._read(path)):
                raise RuntimeError("This screen still has an active owned process")
        previous = _read_optional(root / "study.json")
        if previous and not resume:
            raise ValueError("This screen has already started; use resume")
        if resume and not previous:
            raise ValueError("This screen has not started; use launch")
        if not previous:
            clock_path = root / "launch_time.json"
            if clock_path.exists():
                requested_at = _launch._read(clock_path)["started_at_seconds"]
            else:
                atomic_json(clock_path, {"started_at_seconds": requested_at})
            previous = {
                "schema_version": 1,
                "status": "starting",
                "phase": "calibration",
                "current_case": None,
                "started_at": datetime.fromtimestamp(requested_at, UTC).isoformat(),
                "started_at_seconds": requested_at,
                "deadline_at": requested_at + declaration["max_elapsed_seconds"],
                "stop_at": requested_at + declaration["numerical_stop_seconds"],
                "elapsed_seconds": 0.0,
                "cases": {
                    row["name"]: {
                        "status": "unstarted",
                        "run_dir": str(root / "jobs" / row["name"] / "run"),
                    }
                    for row in declaration["cases"]
                },
            }
            atomic_json(root / "study.json", previous)
        _validate_clock(root, previous, declaration)
        argv = _command(root, "supervise", *(["--resume"] if resume else []))
        with (root / "logs/experiment.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(
                argv,
                cwd=root / "source",
                env=_environment(root, manifest),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        record = {
            "schema_version": 1,
            "state": "starting",
            "started_at": utc_now(),
            "process": process_identity(child.pid),
            "command": argv,
            "mode": "resume" if resume else "start",
        }
        atomic_json(root / "process.json", record)
        return record


def screen_status(package: str | Path) -> Record:
    """Read study and active-run progress using files and process identity only.

    Return actual elapsed time, remaining deadline and the current trainer's
    already-published counters. Missing ownership is never reported as success.
    This read-only path does not initialize JAX, recover a writer or launch work.
    """
    root = Path(package).absolute()
    study = _read_optional(root / "study.json")
    process = _read_optional(root / "process.json")
    alive = bool(process) and _launch._alive(process)
    active_jobs = [
        str(path.parent)
        for path in _owned_jobs(root)
        if _launch._alive(_launch._read(path))
    ]
    active = study.get("active_job")
    current = _read_optional(Path(active) / "run/status.json") if active else {}
    state = study.get("status", "not_started")
    if (
        study
        and not alive
        and state not in ("complete", "failed", "expired", "stopped")
    ):
        state = "interrupted"
    if state == "complete" and (
        process.get("state") != "exited" or process.get("exit_code") != 0
    ):
        state = "finalizing" if alive else "interrupted"
    if process.get("stop_reason") == "deadline":
        state = "expired"
    if not alive and active_jobs:
        state = "orphaned_workers"
    elapsed = study.get("elapsed_seconds", 0.0)
    if alive and study:
        elapsed = max(0.0, time.time() - study["started_at_seconds"])
    return {
        "status": state,
        "alive": alive,
        "active_owned_jobs": active_jobs,
        "phase": study.get("phase"),
        "current_case": study.get("current_case"),
        "elapsed_seconds": elapsed,
        "remaining_deadline_seconds": max(
            0.0, study.get("deadline_at", time.time()) - time.time()
        ),
        "cases": study.get("cases", {}),
        "calibrations_complete": len(
            list((root / "calibration").glob("*/calibration.json"))
        ),
        "trials_started": any((root / "jobs").glob("*/run/run_details.json")),
        "run": current,
        "progress": progress_text(current) if current else "Waiting for run records",
        "failure": study.get("error")
        or (
            "The supervisor reached the original deadline"
            if process.get("stop_reason") == "deadline"
            else process.get("error")
        ),
        "estimated_study_remaining_seconds": _study_eta(root, study, current),
        "report": str(root / "reports/run_summary.md"),
        "log": str(root / "logs/experiment.log"),
    }


def _study_eta(root: Path, study: Mapping[str, Any], current: Record) -> float | None:
    """Estimate pending study cost from calibration; unknown costs stay unknown.

    Read only small existing JSON records. Subtract this active case's recorded
    elapsed time from its estimate and include all unstarted cases. This simple
    estimate can be wrong when workloads slow down; the hard deadline is separate.
    An overrun returns None rather than claiming zero work while a case is active.
    """
    if study.get("status") == "complete":
        return 0.0
    budgets = _read_optional(root / "budgets.json")
    estimates = budgets.get("forecast", {}).get("case_seconds")
    if not estimates:
        return None
    remaining = 0.0
    for name, case in study.get("cases", {}).items():
        if case.get("status") == "complete":
            continue
        estimate = float(estimates[name])
        if name == study.get("current_case") and current:
            estimate -= float(current.get("elapsed_seconds", 0.0))
            if estimate <= 0:
                return None
        remaining += estimate
    return remaining


def stop_screen(package: str | Path) -> Record:
    """Ask the owned controller to stop its active job and preserve partial work.

    Signal only a matching live controller identity, never a stale or unrelated
    PID. Its handler stops the fresh job supervisor. If the controller is gone,
    ask any still-live owned job supervisors to clean up their own children.
    Never signal an orphan group from its saved number. Return current status
    and requested PIDs; shutdown may still be in progress. This does not shorten
    saved budgets, delete files or start a job.
    """
    root = Path(package).absolute()
    process = _read_optional(root / "process.json")
    owner = process.get("trainer", process.get("process"))
    requested: list[int] = []
    if isinstance(owner, dict) and _launch._alive({"process": owner}):
        pid = cast(Record, owner)["pid"]
        os.kill(pid, signal.SIGTERM)
        requested.append(pid)
    else:
        for path in _owned_jobs(root):
            supervisor = _launch._read(path).get("process")
            if isinstance(supervisor, dict) and _launch._alive({"process": supervisor}):
                pid = cast(Record, supervisor)["pid"]
                os.kill(pid, signal.SIGTERM)
                requested.append(pid)
    status = screen_status(root)
    status["stop_requested_pids"] = requested
    if status["active_owned_jobs"] and not requested:
        status["stop_notice"] = (
            "Owned workers remain without a live supervisor. Restart is blocked; "
            "inspect their recorded identities before manual cleanup."
        )
    return status


def _write_study(root: Path, study: Record, **values: object) -> None:
    """Publish one phase/case transition, using the original launch clock."""
    study.update(values)
    study["elapsed_seconds"] = max(0.0, time.time() - study["started_at_seconds"])
    study["updated_at"] = utc_now()
    atomic_json(root / "study.json", study)


def screen_status_text(status: Mapping[str, Any]) -> str:
    """Explain saved study status without starting JAX or performing more reads.

    status is the mapping returned by screen_status. Speed checks and comparison
    trials have separate counts. A failed or stopped study never labels its last
    successful speed check as the completed experiment. Checkpoint and log paths
    belong to the status command, while live progress uses the shorter summary.
    """
    state = status.get("status", "not_started")
    phase = status.get("phase")
    cases = status.get("cases", {})
    done = sum(row.get("status") == "complete" for row in cases.values())
    stopped = state in (
        "failed",
        "expired",
        "stopped",
        "interrupted",
        "orphaned_workers",
    )
    label = (
        "Study Finished"
        if state == "complete"
        else "Study Stopped Before Comparison Trials"
        if stopped and not status.get("trials_started")
        else "Study Stopped"
        if stopped
        else "Checking Speed; Comparison Trials Have Not Started"
        if phase == "calibration"
        else "Study Has Not Started"
        if state == "not_started"
        else "Comparing Learning Progress"
        if phase == "training"
        else "Preparing Results"
    )
    lines = [
        label,
        f"Speed Checks Finished: {status.get('calibrations_complete', 0)} / "
        f"{len(cases) or 12} | "
        f"Comparison Trials Finished: {done} / {len(cases) or 12}",
        f"Total Time Since First Launch: {duration(status.get('elapsed_seconds'))}",
    ]
    if not stopped and state not in ("complete", "not_started"):
        lines.append(
            "Study Time Left (Estimate): "
            + duration(status.get("estimated_study_remaining_seconds"))
            + " | Time Before Stop Limit: "
            + duration(status.get("remaining_deadline_seconds"))
        )
    case = status.get("current_case")
    if case and not stopped:
        batch, length = str(case).removeprefix("b").split("-t")
        lines.append(
            f"Current Setup: {int(batch):,} Parallel Games; "
            f"{int(length)} Steps Per Collection"
        )
    current = status.get("run")
    if current and not stopped:
        view = {**current}
        if phase == "calibration":
            view["phase"] = "speed_check"
        lines.append(progress_text(view))
    if status.get("failure"):
        lines.append(f"Reason For Stopping: {status['failure']}")
    if status.get("stop_requested_pids"):
        lines.append("Stop Requested; Waiting For Running Work To Close")
    if status.get("stop_notice"):
        lines.append(str(status["stop_notice"]))
    return "\n".join(lines)


def _print_progress(root: Path, study: Mapping[str, Any], *, event: str) -> None:
    """Print the same plain-English view as status, using existing host records.

    event names the phase transition. This reads small status files and no model
    arrays. Raw optimizer and process records remain in their existing files.
    """
    status = screen_status(root)
    status["elapsed_seconds"] = max(0.0, time.time() - study["started_at_seconds"])
    stamp = utc_now()[:19].replace("T", " ")
    print(f"\n[{stamp} UTC] {event}\n{screen_status_text(status)}", flush=True)


def _run_job(
    root: Path,
    study: Record,
    job: Path,
    *,
    mode: str,
    deadline: float,
    stopped: list[bool],
    resume: bool = False,
    measure_validation: bool = False,
) -> None:
    """Own one fresh supervisor and wait while periodically reading its progress.

    The supervisor enforces deadline during arbitrary child work. A stop flag,
    stopped process record, nonzero exit or incomplete trainer status fails the
    job. The controller never launches a replacement automatically.
    """
    if stopped[0]:
        raise InterruptedError("The screen was stopped before its next job")
    if time.time() >= deadline:
        raise TimeoutError("No time remains for numerical work")
    job.mkdir(parents=True, exist_ok=True)
    argv = _command(
        root,
        "job-supervise",
        "--job",
        str(job),
        "--mode",
        mode,
        "--deadline",
        str(deadline),
    )
    if resume:
        argv.append("--resume")
    if measure_validation:
        argv.append("--measure-validation")
    _write_study(root, study, active_job=str(job))
    # Inherit the one durable study log, including worker phase changes/errors.
    environment = {
        **os.environ,
        "MARL_BGS_PROGRESS_PHASES_ONLY": "1",
        "MARL_BGS_PROGRESS_CONTEXT": (
            "Speed Check Only; These Weights Are Discarded"
            if mode == "calibration"
            else "Comparison Trial"
        ),
    }
    child = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=environment
    )
    last_display = 0.0
    stop_sent = False
    try:
        while child.poll() is None:
            if stopped[0] and not stop_sent:
                child.send_signal(signal.SIGTERM)
                stop_sent = True
            if time.monotonic() - last_display >= 10:
                _print_progress(root, study, event="Progress")
                last_display = time.monotonic()
            time.sleep(0.25)
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGTERM)
            # The shared supervisor bounds complete descendant cleanup to 14s.
            # A stuck owner remains recorded and blocks any later launch.
            child.wait(timeout=20)
    record = _read_optional(job / "process.json")
    if record.get("stop_reason") == "deadline":
        raise TimeoutError(f"Deadline reached during {job.name}")
    if stopped[0] or record.get("stop_signal") is not None:
        raise InterruptedError("The screen was stopped; no further case will start")
    if child.returncode != 0 or record.get("exit_code") != 0:
        raise RuntimeError(
            f"Job failed with exit {child.returncode}; "
            f"see {root / 'logs/experiment.log'}"
        )
    result = _launch._read(job / "run/status.json")
    if result.get("status") != "complete" or result.get("phase") != "complete":
        raise ValueError("A clean child exit did not produce a completed training run")


def _worker(
    root: Path, job: Path, *, mode: str, resume: bool, measure_validation: bool
) -> None:
    """Call the public trainer, then record calibration-only timing evidence.

    Configuration is already frozen in job/config.json. Calibration never
    becomes a scientific prefix. A single calibration actor may time 40 Random
    games; these are engineering games and are excluded from the study results.
    """
    import jax

    from marl_battlegrounds.training.runner import read_config, train

    if (
        jax.default_backend() != "gpu"
        or len(cast(list[object], jax.devices("gpu"))) != 1
    ):
        raise ValueError("The screen worker requires its one declared GPU")

    config = read_config(job / "config.json")
    result = train(
        None if resume else config,
        output_dir=job / "run",
        resume_from=_launch._checkpoint(job, None) if resume else None,
    )
    if mode != "calibration":
        return
    rows = read_jsonl(job / "run/training_updates.jsonl")
    expected = 6 + math.ceil(300 / config.ppo.rollout_length)
    if len(rows) != expected or [row["update_index"] for row in rows] != list(
        range(1, expected + 1)
    ):
        raise ValueError("Calibration does not contain its exact declared updates")
    samples = [float(row["collection_update_seconds"]) for row in rows[-5:]]
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("Calibration contains invalid timed blocks")
    exposure = _launch._read(job / "run/exposure.json")
    # Row 1 is a named pinned System, not a historical snapshot, when one is set.
    first_history_row = 1 if config.pinned_opponent is None else 2
    if sum(exposure["starts_by_episode_stage"]) <= config.num_envs or not sum(
        exposure["starts_by_opponent"][first_history_row:]
    ):
        raise ValueError(
            "Calibration did not exercise episode resets and historical opponents"
        )
    state = _launch._read(job / "run/status.json")
    saves = [
        row["seconds"]
        for row in read_jsonl(job / "run/run_events.jsonl")
        if row.get("event") == "checkpoint_saved"
    ]
    record: Record = {
        "schema_version": 1,
        "num_envs": config.num_envs,
        "rollout_length": config.ppo.rollout_length,
        "median_update_seconds": statistics.median(samples),
        "timed_seconds": samples,
        "cold_first_call_seconds": rows[0]["collection_update_seconds"],
        "overhead_seconds": max(
            0.0,
            state["elapsed_seconds"]
            - sum(row["collection_update_seconds"] for row in rows),
        ),
        "save_seconds": statistics.mean(saves) if saves else 0.0,
        "memory": state.get("memory"),
        "env_steps": result.completed_env_steps,
        "run_dir": str(result.run_dir),
        "validation_seconds": None,
        "scope": "Engineering calibration only; no model selection",
    }
    if measure_validation:
        from marl_battlegrounds.training.validation import validate_random

        started = time.monotonic()
        validate_random(
            result.final_actor, output_dir=job / "timing-validation", seed_pairs=4
        )
        record["validation_seconds"] = time.monotonic() - started
        record["engineering_validation_games"] = 40
    atomic_json(job / "calibration_result.json", record)


def _share_initialization(root: Path, first_case: str) -> None:
    """Publish the first trial's original initialization result without new IDs."""
    path = root / "jobs" / first_case / "run/random_diagnostics.json"
    values = json.loads(path.read_text())
    initial = [row for row in values if row.get("env_steps") == 0]
    if len(initial) != 1 or initial[0].get("reused_initialization"):
        raise ValueError(
            "The first trial has no unique original initialization diagnostic"
        )
    target = root / "shared/initialization.json"
    if target.exists():
        if _launch._read(target) != initial[0]:
            raise ValueError("The shared initialization result changed")
    else:
        atomic_json(target, initial[0])


def _read_calibration(root: Path, case: Record, declaration: Record) -> Record | None:
    """Recover a completed calibration once, including its publication gap.

    Completed timings must match their original attempt file, configuration and
    five raw measured rows. An incomplete attempt remains visible for an explicit
    retry. A published but unclean attempt blocks reuse instead of replacing its
    timings with a more convenient result.
    """
    directory = root / "calibration" / case["name"]
    published = directory / "calibration.json"
    result_paths = sorted(directory.glob("attempt-*/calibration_result.json"))
    if not result_paths:
        if published.exists():
            raise ValueError("Calibration lost its original attempt")
        return None
    if len(result_paths) != 1:
        raise ValueError("A case has more than one completed calibration")
    path = result_paths[0]
    result = _launch._read(path)
    process = _launch._read(path.parent / "process.json")
    if (
        process.get("state") != "exited"
        or process.get("exit_code") != 0
        or process.get("stop_signal") is not None
    ):
        raise ValueError("Published calibration has an unclean supervisor exit")
    if _launch._read(path.parent / "config.json") != calibration_config(
        declaration, case
    ):
        raise ValueError("Calibration configuration changed")
    rows = read_jsonl(path.parent / "run/training_updates.jsonl")
    expected = 6 + math.ceil(300 / case["rollout_length"])
    samples = [row["collection_update_seconds"] for row in rows[-5:]]
    if (
        len(rows) != expected
        or len(samples) != 5
        or result["timed_seconds"] != samples
        or result["median_update_seconds"] != statistics.median(samples)
        or result["num_envs"] != case["num_envs"]
        or result["rollout_length"] != case["rollout_length"]
    ):
        raise ValueError("Calibration does not match its original measurements")
    if published.exists():
        if _launch._read(published) != result:
            raise ValueError("Completed calibration changed")
    else:
        atomic_json(published, result)
    return result


def run_mappo_screen(package: str | Path, *, resume: bool = False) -> Record:
    """Run the frozen screen inside its detached controller until done or stopped.

    package was prepared and explicitly launched by the user. resume permits
    only unfinished work under the original deadline. Calibration retries are
    separate attempts until budgets are frozen; scientific runs use complete
    learner recovery. Return final study records. Failures are retained and
    re-raised after partial reporting. No additional experiment is selected.
    """
    root = Path(package).absolute()
    declaration = _launch._read(root / "declaration.json")
    study = _launch._read(root / "study.json")
    _validate_clock(root, study, declaration)
    stopped = [False]
    previous: dict[int, Any] = {}

    def request_stop(_number: int, _frame: object) -> None:
        """Remember a stop for the controller's job loop; do no signal-time I/O."""
        stopped[0] = True

    from marl_battlegrounds.training.analysis import analyze_screen

    with (root / ".study.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous[number] = signal.signal(number, request_stop)
        try:
            if study.get("status") == "complete":
                analyze_screen(root)
                return study
            if time.time() >= study["stop_at"]:
                _write_study(
                    root,
                    study,
                    status="expired",
                    phase="reporting",
                    error="Original numerical deadline has expired",
                )
                analyze_screen(root)
                return study
            budget_path = root / "budgets.json"
            if not budget_path.exists():
                calibration: dict[str, Record] = {}
                for index, case in enumerate(declaration["cases"]):
                    name = case["name"]
                    path = root / "calibration" / name
                    path.mkdir(parents=True, exist_ok=True)
                    recovered = _read_calibration(root, case, declaration)
                    if recovered is not None:
                        calibration[name] = recovered
                        continue
                    attempts = list(path.glob("attempt-*"))
                    if attempts and not resume:
                        raise ValueError(
                            "Interrupted calibration requires explicit resume"
                        )
                    job = path / f"attempt-{len(attempts) + 1}"
                    job.mkdir()
                    atomic_json(
                        job / "config.json", calibration_config(declaration, case)
                    )
                    study["cases"][name].update(status="calibrating")
                    _write_study(
                        root,
                        study,
                        status="running",
                        phase="calibration",
                        current_case=name,
                    )
                    _run_job(
                        root,
                        study,
                        job,
                        mode="calibration",
                        deadline=min(
                            study["stop_at"],
                            study["started_at_seconds"]
                            + declaration["calibration_limit_seconds"],
                        ),
                        stopped=stopped,
                        measure_validation=index == 0,
                    )
                    recovered = _read_calibration(root, case, declaration)
                    if recovered is None:
                        raise ValueError("Calibration completed without its result")
                    calibration[name] = recovered
                    study["cases"][name]["status"] = "unstarted"
                budgets = resolve_budgets(declaration, calibration, root)
                validation_seconds = next(
                    row["validation_seconds"]
                    for row in calibration.values()
                    if row.get("validation_seconds") is not None
                )
                overhead = sum(
                    row["overhead_seconds"] + row["cold_first_call_seconds"]
                    for row in calibration.values()
                )
                saves = sum(
                    8 * calibration[row["name"]]["save_seconds"]
                    for row in budgets["cases"]
                )
                projected = declaration["timing_margin"] * (
                    sum(row["estimated_training_seconds"] for row in budgets["cases"])
                    + overhead
                    + saves
                    + budgets["diagnostic_passes"] * validation_seconds
                )
                budgets["forecast"] = {
                    "remaining_seconds": projected,
                    "calibration_elapsed_seconds": time.time()
                    - study["started_at_seconds"],
                    "validation_seconds_per_pass": validation_seconds,
                    "margin": declaration["timing_margin"],
                    "report_reserve_seconds": declaration["report_reserve_seconds"],
                    "case_seconds": {
                        row["name"]: declaration["timing_margin"]
                        * (
                            row["estimated_training_seconds"]
                            + calibration[row["name"]]["overhead_seconds"]
                            + calibration[row["name"]]["cold_first_call_seconds"]
                            + 8 * calibration[row["name"]]["save_seconds"]
                            + (len(row["checkpoint_env_steps"]) + (index == 0))
                            * validation_seconds
                        )
                        for index, row in enumerate(budgets["cases"])
                    },
                }
                if (
                    time.time() + projected + declaration["report_reserve_seconds"]
                    > study["deadline_at"]
                ):
                    atomic_json(root / "infeasible_budget_forecast.json", budgets)
                    expected = (
                        budgets["forecast"]["calibration_elapsed_seconds"]
                        + projected
                        + declaration["report_reserve_seconds"]
                    )
                    raise ValueError(
                        "All speed checks finished, but no comparison trial started. "
                        "Estimated total including the timing buffer: "
                        f"{duration(expected)}. "
                        "Allowed total: "
                        f"{duration(declaration['max_elapsed_seconds'])}. "
                        "The speed measurements are saved for reuse."
                    )
                for case in budgets["cases"]:
                    atomic_json(root / case["config_path"], case["config"])
                atomic_json(budget_path, budgets)
                _write_study(
                    root,
                    study,
                    budgets_digest=_digest(budgets),
                    phase="training",
                    active_job=None,
                )
            budgets = _launch._read(budget_path)
            if budgets["declaration_digest"] != _digest(declaration):
                raise ValueError("Frozen budgets belong to a different declaration")
            if "budgets_digest" not in study:
                recovered_costs = {
                    case["name"]: _read_calibration(root, case, declaration)
                    for case in declaration["cases"]
                }
                if any(value is None for value in recovered_costs.values()):
                    raise ValueError("Unbound budgets lack completed calibrations")
                reconstructed = resolve_budgets(
                    declaration, cast(dict[str, Record], recovered_costs), root
                )
                if {
                    key: value for key, value in budgets.items() if key != "forecast"
                } != reconstructed:
                    raise ValueError("Unbound budgets differ from frozen measurements")
                _write_study(root, study, budgets_digest=_digest(budgets))
            if study["budgets_digest"] != _digest(budgets):
                raise ValueError("Frozen budgets changed")
            for index, case in enumerate(budgets["cases"]):
                name = case["name"]
                config = _launch._read(root / case["config_path"])
                if _digest(config) != case["config_digest"] or config != case["config"]:
                    raise ValueError(f"Frozen configuration changed: {name}")
                job = root / "jobs" / name
                prior = _read_optional(job / "run/status.json")
                process = _read_optional(job / "process.json")
                completed = (
                    prior.get("status") == "complete"
                    and process.get("state") == "exited"
                    and process.get("exit_code") == 0
                    and process.get("stop_signal") is None
                )
                if not completed:
                    if stopped[0]:
                        raise InterruptedError("The screen was stopped")
                    if time.time() >= study["stop_at"]:
                        raise TimeoutError("Original numerical deadline reached")
                    job.mkdir(exist_ok=True)
                    if (job / "config.json").exists() and _launch._read(
                        job / "config.json"
                    ) != config:
                        raise ValueError("A job's immutable configuration changed")
                    atomic_json(job / "config.json", config)
                    entry = study["cases"][name]
                    entry.update(status="training", phase="training")
                    entry.setdefault("started_at", utc_now())
                    _write_study(
                        root,
                        study,
                        status="running",
                        phase="training",
                        current_case=name,
                    )
                    _run_job(
                        root,
                        study,
                        job,
                        mode="trial",
                        deadline=study["stop_at"],
                        stopped=stopped,
                        resume=bool(prior),
                    )
                if index == 0:
                    _share_initialization(root, name)
                study["cases"][name].update(
                    status="complete", phase="complete", finished_at=utc_now()
                )
                _write_study(root, study, active_job=None)
                analyze_screen(root)
                _print_progress(root, study, event="Case Complete")
            _write_study(root, study, phase="reporting", current_case=None)
            report = analyze_screen(root)
            if stopped[0]:
                raise InterruptedError("The screen was stopped during reporting")
            if not report.get("complete") or report.get("evidence_errors"):
                raise ValueError(
                    "Required experiment evidence is incomplete; inspect the report"
                )
            _write_study(
                root,
                study,
                status="complete",
                phase="complete",
                finished_at=utc_now(),
                active_job=None,
            )
            analyze_screen(root)
            _print_progress(root, study, event="Experiment Complete")
            print(f"Report: {root / 'reports/run_summary.md'}", flush=True)
            return study
        except BaseException as error:
            state = (
                "expired"
                if isinstance(error, TimeoutError)
                else "stopped"
                if isinstance(error, InterruptedError) or stopped[0]
                else "failed"
            )
            current = study.get("current_case")
            if (
                current
                and study.get("phase") != "calibration"
                and study["cases"][current].get("status") != "complete"
            ):
                study["cases"][current].update(
                    status="interrupted"
                    if state in ("expired", "stopped")
                    else "failed",
                    error=str(error),
                )
            _write_study(
                root,
                study,
                status=state,
                phase="reporting",
                error=f"{type(error).__name__}: {error}",
                finished_at=utc_now(),
            )
            try:
                analyze_screen(root)
            except Exception as report_error:
                print(f"Partial report failed: {report_error}", flush=True)
            _print_progress(root, study, event=f"Experiment {state.title()}")
            raise
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the small screen CLI and call shared preparation/execution owners.

    Public actions are prepare/start/status/resume/stop. Internal supervisor,
    controller and worker actions use the same frozen package; they are not
    separate researcher workflows. Return zero on success and nonzero on error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "start",
            "status",
            "resume",
            "stop",
            "supervise",
            "run",
            "job-supervise",
            "worker",
        ),
    )
    parser.add_argument("package", type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--mode", choices=("calibration", "trial"), default="trial")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--measure-validation", action="store_true")
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable command output"
    )
    args = parser.parse_args(argv)
    root = args.package.absolute()
    if args.action == "prepare":
        if args.gpu_uuid is None:
            parser.error("prepare requires --gpu-uuid for the authorized internal card")
        print(
            json.dumps(
                prepare_screen(args.repository, root, gpu_uuid=args.gpu_uuid), indent=2
            )
        )
    elif args.action in ("start", "resume"):
        record = start_screen(root, resume=args.action == "resume")
        if args.json:
            print(json.dumps(record, indent=2))
        else:
            print(
                "Experiment Started In The Background. "
                "Closing This Terminal Will Not Stop It."
            )
        print(
            f"Watch Progress: tail -F {shlex.quote(str(root / 'logs/experiment.log'))}"
        )
    elif args.action == "status":
        status = screen_status(root)
        print(json.dumps(status, indent=2) if args.json else screen_status_text(status))
        if not args.json:
            print(f"Saved Report: {status['report']}\nFull Log: {status['log']}")
            if status.get("run", {}).get("latest_checkpoint"):
                print(f"Saved Checkpoint: {status['run']['latest_checkpoint']}")
    elif args.action == "stop":
        status = stop_screen(root)
        print(json.dumps(status, indent=2) if args.json else screen_status_text(status))
    elif args.action == "supervise":
        study = _launch._read(root / "study.json")
        command = _command(root, "run", *(["--resume"] if args.resume else []))
        deadline = (
            study["deadline_at"]
            if time.time() < study["stop_at"]
            else time.time() + 180
        )
        return _launch.supervise_command(root, command, deadline_at=deadline)
    else:
        manifest = _launch._read(root / _MANIFEST)
        os.sched_setaffinity(0, manifest["cpu_affinity"])
        if args.action == "run":
            result = run_mappo_screen(root, resume=args.resume)
            return 0 if result["status"] == "complete" else 1
        if args.job is None or not args.job.resolve().is_relative_to(root.resolve()):
            parser.error("An internal job must belong to this screen package")
        job = args.job.absolute()
        if args.action == "job-supervise":
            if args.deadline is None:
                parser.error("A job supervisor requires its saved deadline")
            command = _command(root, "worker", "--job", str(job), "--mode", args.mode)
            if args.resume:
                command.append("--resume")
            if args.measure_validation:
                command.append("--measure-validation")
            return _launch.supervise_command(job, command, deadline_at=args.deadline)
        _worker(
            root,
            job,
            mode=args.mode,
            resume=args.resume,
            measure_validation=args.measure_validation,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Stopped: {error}", flush=True)
        if len(sys.argv) > 2:
            logs = Path(sys.argv[2]) / "logs"
            if logs.is_dir():
                with (logs / "errors.log").open("a") as output:
                    output.write(f"[{utc_now()}] {sys.argv[1]}\n")
                    traceback.print_exc(file=output)
                print(f"Technical Details: {logs / 'errors.log'}", flush=True)
        raise SystemExit(1) from None
