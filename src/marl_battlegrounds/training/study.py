"""Run declared training cases through the ordinary trainer and launcher.

This module owns finite study declarations, ordered cases and saved reports.
The trainer still owns learning, validation, selection and checkpoint recovery.
Only one case worker runs at a time. Import, status and stop use no numerical
backend. Subprocesses inherit the caller's Python and device environment.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from marl_battlegrounds.training import _launch
from marl_battlegrounds.training._run_io import atomic_json, process_identity, utc_now

type Record = dict[str, Any]
type Declaration = str | Path | Mapping[str, Any]
_DECLARATION = "declaration.json"


def _read(path: Path) -> Record:
    """Read a regular JSON object through the shared launch file owner."""
    return _launch._read(path)


def _optional(path: Path) -> Record:
    """Read an existing object, returning an empty mapping only when absent."""
    return _read(path) if path.exists() else {}


def _digest(value: object) -> str:
    """Hash finite sorted JSON while preserving every list's declared order."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _object(value: object, name: str) -> Record:
    """Copy a finite JSON object without retaining caller-owned containers."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    try:
        return cast(
            Record,
            json.loads(
                json.dumps(dict(cast(Mapping[str, Any], value)), allow_nan=False)
            ),
        )
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must contain only finite JSON values") from error


def _keys(value: Record, allowed: set[str], name: str) -> None:
    """Reject unknown fields before starting any declared work."""
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")


def _duration(value: object, name: str) -> float | None:
    """Check a positive number of wall-clock seconds, or no limit."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be positive seconds or None")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite positive seconds")
    return float(value)


def _path(value: object, base: Path, name: str) -> str:
    """Resolve one declared nonempty path against the declaration's directory."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _reference(value: object, base: Path) -> object:
    """Resolve folder references; keep registered names and factories unchanged."""
    if not isinstance(value, str):
        return value
    from marl_battlegrounds.evaluation.policy_execution import policy

    try:
        policy(value)
        return value
    except ValueError:
        pass
    if (
        Path(value).is_absolute()
        or "/" in value
        or value.startswith(".")
        or (base / value).is_dir()
    ):
        return _path(value, base, "System reference")
    return value


def _config(value: object, base: Path, seed: int) -> Record:
    """Resolve reference paths and let the trainer check all method settings."""
    import jax

    with jax.default_device(jax.devices("cpu")[0]):
        from marl_battlegrounds.training.runner import config_from_dict, config_to_dict

        config = _object(value, "Case config")
        if "seed" in config:
            raise ValueError("Use the case seeds list; config.seed is not allowed")
        if "method" not in config or "total_env_steps" not in config:
            raise ValueError("Each config must declare method and total_env_steps")
        for name in ("validation_panel", "random_initialization_result"):
            if config.get(name) is not None:
                config[name] = _path(config[name], base, name)
        for name in ("pinned_opponent", "slot_diagnostic_actor"):
            if config.get(name) is not None:
                config[name] = _reference(config[name], base)
        if config.get("validation_opponents") is not None:
            config["validation_opponents"] = [
                _reference(value, base) for value in config["validation_opponents"]
            ]
        result = config_to_dict(config_from_dict({**config, "seed": seed}))
        result.pop("seed")
        return result


def _checkpoint_identity(path: str) -> Record:
    """Verify all parent payloads and pin its complete learner description."""
    import jax

    with jax.default_device(jax.devices("cpu")[0]):
        from marl_battlegrounds.training.checkpoints import read_checkpoint_details

        details = read_checkpoint_details(path)
        if details["kind"] != "learner":
            raise ValueError("A study continuation requires a full learner checkpoint")
        return {
            "checkpoint_id": details["checkpoint_id"],
            "description_sha256": _digest(details),
            "method": details["metadata"]["config"]["method"],
        }


def _identities(methods: Sequence[str]) -> Record:
    """Read source and method dependencies through the checkpoint owner."""
    from marl_battlegrounds.training.checkpoints import runtime_identity

    return {method: runtime_identity(method) for method in sorted(set(methods))}


def _file_inputs(case: Record) -> Record:
    """Pin declared external files without constructing models or factories.

    Panel and shared-result files use exact byte hashes. Actor/checkpoint
    folders use the checkpoint owner's verified description, which binds its
    payload inventory; the trainer still checks payload bytes when loading.
    This runs at declaration setup and once before each worker attempt, never
    inside a training tick. Factory code and remote service state cannot be
    frozen by a reference string and remain the provider's responsibility.
    """
    result: Record = {}

    def file(value: object, *, panel: bool = False) -> None:
        """Bind a regular file, accepting panel directories as the owner does."""
        if value is None:
            return
        path = Path(str(value))
        if panel and path.is_dir():
            path /= "panel.json"
        key = str(path)
        if key not in result:
            result[key] = {
                "kind": "file",
                "sha256": _launch._hash(path),
                "bytes": path.stat().st_size,
            }

    def artifact(value: object) -> None:
        """Bind an absolute folder reference; leave names and factories alone."""
        if not isinstance(value, str) or not Path(value).is_absolute():
            return
        if value not in result:
            from marl_battlegrounds.training.checkpoints import (
                read_checkpoint_description,
            )

            details = read_checkpoint_description(value)
            result[value] = {
                "kind": "artifact",
                "checkpoint_id": details["checkpoint_id"],
                "description_sha256": _digest(details),
            }

    config = case.get("config", {})
    file(config.get("validation_panel"), panel=True)
    file(config.get("random_initialization_result"))
    artifact(config.get("pinned_opponent"))
    artifact(config.get("slot_diagnostic_actor"))
    for reference in config.get("validation_opponents") or ():
        artifact(reference)
    validation = case.get("changes", {}).get("validation", {})
    file(validation.get("panel"), panel=True)
    for reference in validation.get("bindings") or ():
        artifact(reference)
    return result


def _normalize(config: Declaration) -> Record:
    """Freeze ordered cases, resolved trainer defaults and linked parent facts."""
    if isinstance(config, (str, Path)):
        source = Path(config).expanduser().resolve()
        declared, base = _read(source), source.parent
    else:
        declared, base = _object(config, "Study declaration"), Path.cwd()
    _keys(
        declared,
        {
            "schema_version",
            "name",
            "duration_seconds",
            "failure_policy",
            "cases",
            "finalist_of",
        },
        "study",
    )
    if (
        type(declared.get("schema_version", 1)) is not int
        or declared.get("schema_version", 1) != 1
    ):
        raise ValueError("Unsupported study schema_version")
    if not isinstance(declared.get("name"), str) or not declared["name"].strip():
        raise ValueError("Study name must be nonempty")
    failure = declared.get("failure_policy", "stop")
    if failure not in ("stop", "continue"):
        raise ValueError("failure_policy must be stop or continue")
    cases = declared.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Study cases must be a nonempty list")
    normalized: Record = {
        "schema_version": 1,
        "name": declared["name"],
        "duration_seconds": _duration(
            declared.get("duration_seconds"), "Study duration_seconds"
        ),
        "failure_policy": failure,
        "cases": [],
    }
    linked_root: Path | None = None
    if declared.get("finalist_of") is not None:
        linked_root = Path(_path(declared["finalist_of"], base, "finalist_of"))
        discovery = _read(linked_root / _DECLARATION)
        normalized["finalist_of"] = {
            "study_dir": str(linked_root),
            "declaration_sha256": _digest(discovery),
        }
    ids: set[str] = set()
    for raw in cast(list[Any], cases):
        case = _object(raw, "Study case")
        identifier = case.get("id")
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identifier) is None
        ):
            raise ValueError(
                "Case id must use 1 to 80 letters, digits, underscores or hyphens"
            )
        if identifier in ids:
            raise ValueError(f"Duplicate study case id: {identifier}")
        ids.add(identifier)
        common: Record = {
            "id": identifier,
            "duration_seconds": _duration(
                case.get("duration_seconds"), f"Case {identifier} duration_seconds"
            ),
        }
        if "config" in case:
            _keys(case, {"id", "config", "seeds", "duration_seconds"}, "training case")
            seeds = case.get("seeds")
            if not isinstance(seeds, list) or not seeds:
                raise ValueError("Case seeds must be distinct integers in [0, 2**32)")
            seeds = cast(list[Any], seeds)
            if any(
                type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds
            ) or len(set(seeds)) != len(seeds):
                raise ValueError("Case seeds must be distinct integers in [0, 2**32)")
            checked = _config(case["config"], base, seeds[0])
            normalized["cases"].append({**common, "config": checked, "seeds": seeds})
        else:
            _keys(
                case,
                {
                    "id",
                    "checkpoint",
                    "additional_env_steps",
                    "changes",
                    "duration_seconds",
                },
                "continuation case",
            )
            checkpoint = _path(case.get("checkpoint"), base, "checkpoint")
            steps = case.get("additional_env_steps")
            if type(steps) is not int or steps <= 0:
                raise ValueError("additional_env_steps must be a positive integer")
            changes = _object(case.get("changes", {}), "Continuation changes")
            _keys(
                changes,
                {
                    "validation",
                    "learning_rate",
                    "exploration",
                    "history_capture_env_steps",
                },
                "continuation changes",
            )
            if isinstance(changes.get("validation"), dict):
                validation = changes["validation"]
                if validation.get("panel") is not None:
                    validation["panel"] = _path(
                        validation["panel"], base, "validation panel"
                    )
                if isinstance(validation.get("bindings"), list):
                    validation["bindings"] = [
                        _reference(value, base) for value in validation["bindings"]
                    ]
            if linked_root is not None:
                parents = {
                    str((linked_root / "cases" / row["id"] / "run").resolve())
                    for row in _jobs(_read(linked_root / _DECLARATION))
                }
                if str(Path(checkpoint).parent.parent) not in parents:
                    raise ValueError(
                        "Finalist continuation parent must belong to the linked study"
                    )
            normalized["cases"].append(
                {
                    **common,
                    "checkpoint": checkpoint,
                    "parent_identity": _checkpoint_identity(checkpoint),
                    "additional_env_steps": steps,
                    "changes": changes,
                }
            )
    for case in normalized["cases"]:
        case["external_inputs"] = _file_inputs(case)
    jobs = _jobs(normalized)
    if len({row["id"] for row in jobs}) != len(jobs):
        raise ValueError("Expanded study case IDs conflict")
    normalized["runtime_identities"] = _identities(
        [
            row["config"]["method"]
            if row["kind"] == "train"
            else row["parent_identity"]["method"]
            for row in jobs
        ]
    )
    return normalized


def _jobs(declared: Record) -> list[Record]:
    """Expand each declared seed once, keeping case order and seed order."""
    rows: list[Record] = []
    for case in declared["cases"]:
        if "config" in case:
            for seed in case["seeds"]:
                rows.append(
                    {
                        "id": f"{case['id']}--seed-{seed}",
                        "kind": "train",
                        "config": {**case["config"], "seed": seed},
                        "duration_seconds": case["duration_seconds"],
                        "external_inputs": case.get("external_inputs", {}),
                    }
                )
        else:
            rows.append({**case, "kind": "extend"})
    return rows


def _freeze(config: Declaration) -> Record:
    """Validate one declaration on CPU without starting the caller's backend.

    One short subprocess owns trainer imports and any parent-artifact reads.
    It inherits Python paths and changes device selection only in that child.
    Temporary request/results are removed on success or failure; no study
    folder exists yet. Invalid declarations raise ValueError with the child error.
    """
    value = (
        str(Path(config).expanduser().resolve())
        if isinstance(config, (str, Path))
        else _object(config, "Study declaration")
    )
    with tempfile.TemporaryDirectory(prefix="marl-study-declaration-") as temporary:
        directory = Path(temporary)
        atomic_json(directory / "request.json", {"config": value})
        completed = subprocess.run(
            _command("_normalize", str(directory)),
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        outcome = _optional(directory / "outcome.json")
        if completed.returncode or "declaration" not in outcome:
            raise ValueError(
                outcome.get("error")
                or completed.stderr.strip()
                or "Study declaration validation failed"
            )
        return cast(Record, outcome["declaration"])


def _prepare(
    config: Declaration | None,
    output_dir: str | Path | None,
    resume_from: str | Path | None,
) -> tuple[Path, Record, bool]:
    """Check new-directory or saved-declaration rules before running cases."""
    if resume_from is not None:
        if output_dir is not None:
            raise ValueError("Use output_dir for a new study or resume_from, not both")
        root = Path(resume_from).expanduser().resolve()
        declared = _read(root / _DECLARATION)
        if config is not None and _freeze(config) != declared:
            raise ValueError(
                "Supplied study declaration differs from the saved declaration"
            )
        with _launch.launch_lock(root):
            started = any(
                (root / name).exists()
                for name in (
                    "launch_time.json",
                    "study.json",
                    "process.json",
                    "report.json",
                )
            )
            for name in ("cases", "logs"):
                folder = root / name
                if folder.is_symlink():
                    raise ValueError(
                        f"Study {name} directory must not be a symbolic link"
                    )
                if started and not folder.is_dir():
                    raise ValueError(f"Started study is missing its {name} directory")
                folder.mkdir(exist_ok=True)
        return root, declared, True
    if config is None or output_dir is None:
        raise ValueError("A new study requires config and an exact output_dir")
    root = Path(output_dir).expanduser().resolve()
    declared = _freeze(config)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Study output_dir must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    with _launch.launch_lock(root):
        if any(path.name != ".launch.lock" for path in root.iterdir()):
            raise ValueError("Study output_dir must be new or empty")
        atomic_json(root / _DECLARATION, declared)
        (root / "cases").mkdir()
        (root / "logs").mkdir()
    return root, declared, False


def _case_directory(root: Path, identifier: str) -> Path:
    """Keep every case below its declared root without following nested links."""
    directory = root / "cases" / identifier
    if directory.resolve() != directory:
        raise ValueError("Study case directory follows a link or escapes its owner")
    return directory


def _process_paths(root: Path) -> list[Path]:
    """List only the controller and declared case process records."""
    declared = _read(root / _DECLARATION)
    return [
        root / "process.json",
        root / "analysis_job/process.json",
        *(_case_directory(root, row["id"]) / "process.json" for row in _jobs(declared)),
    ]


def _clock(root: Path, declared: Record, *, create: bool) -> Record:
    """Read or create the one fixed study deadline through the launch owner."""
    duration = declared["duration_seconds"]
    if not (root / "launch_time.json").exists() and (
        any(
            (root / name).exists()
            for name in ("study.json", "process.json", "report.json")
        )
        or any((root / "cases").iterdir())
    ):
        raise ValueError("Started study is missing its original launch clock")
    return _launch.launch_clock(
        root / "launch_time.json",
        durations={} if duration is None else {"deadline_at": duration},
        binding={"declaration_sha256": _digest(declared)},
        create=create,
    )


def _command(*arguments: str) -> list[str]:
    """Use this interpreter and installed package without changing import paths."""
    return [sys.executable, "-m", "marl_battlegrounds.training.study", *arguments]


def _latest(run: Path) -> Path | None:
    """Resolve only the saved latest pointer; ordinary train verifies its bytes."""
    if run.resolve() != run:
        raise ValueError("Study run directory follows a link or escapes its case")
    pointer = _optional(run / "latest_checkpoint.json")
    if not pointer:
        return None
    identifier = pointer.get("checkpoint_id")
    relative = pointer.get("relative_path")
    if (
        not isinstance(identifier, str)
        or Path(identifier).name != identifier
        or not isinstance(relative, str)
        or Path(relative).parent != Path("checkpoints")
        or not (
            Path(relative).name == identifier
            or Path(relative).name.endswith(f"_{identifier}")
        )
    ):
        raise ValueError("Study case has an invalid latest checkpoint pointer")
    selected = run / relative
    if selected.resolve() != selected:
        raise ValueError("Study checkpoint follows a link or escapes its run")
    return selected


def _report(root: Path, declared: Record, state: Record) -> Record:
    """Save all declared outcomes, including missing, failed and unfinished work."""
    rows = [
        state["cases"].get(
            job["id"], {"id": job["id"], "status": "pending", "attempts": []}
        )
        for job in _jobs(declared)
    ]
    result = {
        "schema_version": 1,
        "study_dir": str(root),
        "declaration_sha256": _digest(declared),
        "status": state["status"],
        "cases": rows,
        "analysis": state.get("analysis", {}),
        "updated_at": utc_now(),
        "qualification": "Execution records only; no claim of learned competence",
    }
    atomic_json(root / "report.json", result)
    return result


def _finished_supervisor(directory: Path) -> None:
    """Release this controller's finished supervisor role, retaining child facts.

    supervise_command runs inside this still-live controller. Once it returns,
    that role is finished even though Python remains alive. Clear only our exact
    identity from an exited record. Saved child/group cleanup remains intact so
    surviving workers still prevent another launch.
    """
    record = _optional(directory / "process.json")
    if record.get("state") == "exited" and record.get("process") == process_identity():
        record["supervisor_identity"] = record["process"]
        record["process"] = None
        atomic_json(directory / "process.json", record)


def _stopped(directory: Path, code: int) -> bool:
    """Recognize a user or process signal without confusing it with a deadline.

    The supervisor's saved stop facts win over child exit codes. A child killed
    by a signal has a negative code; shells commonly return 128 plus the signal.
    Deadline cleanup is handled separately as a time limit.
    """
    process = _optional(directory / "process.json")
    if process.get("stop_reason") == "deadline":
        return False
    return bool(
        process.get("stop_signal")
        or process.get("stop_reason") == "signal"
        or code < 0
        or code >= 128
    )


def _execute(root: Path, *, resume: bool, handoff: bool = False) -> Record:
    """Run unfinished cases once while holding the study's launch lock."""
    with _launch.launch_lock(root, blocking=handoff):
        declared = _read(root / _DECLARATION)
        if "finalist_of" in declared:
            link = declared["finalist_of"]
            if (
                _digest(_read(Path(link["study_dir"]) / _DECLARATION))
                != link["declaration_sha256"]
            ):
                raise ValueError(
                    "Linked discovery declaration differs from its frozen identity"
                )
        live = _launch.live_process_records(_process_paths(root))
        own = process_identity()
        if any(
            path != root / "process.json" or _read(path).get("process") != own
            for path in live
        ):
            raise RuntimeError("This study still has an active owned process")
        clock = _clock(
            root, declared, create=not resume or not (root / "study.json").exists()
        )
        state = _optional(root / "study.json") or {"schema_version": 1, "cases": {}}
        state.update(status="running", **clock)
        process: Record = {
            "schema_version": 1,
            "state": "running",
            "process": own,
            "started_at": utc_now(),
        }
        atomic_json(root / "process.json", process)
        atomic_json(root / "study.json", state)
        try:
            for job in _jobs(declared):
                row = state["cases"].setdefault(
                    job["id"],
                    {
                        "id": job["id"],
                        "kind": job["kind"],
                        "status": "pending",
                        "attempts": [],
                    },
                )
                if row["status"] == "complete":
                    continue
                reserve = _launch.cleanup_reserve_seconds()
                if clock.get("deadline_at", math.inf) - reserve <= time.time():
                    state["status"] = "time_limit"
                    break
                directory = _case_directory(root, job["id"])
                directory.mkdir(exist_ok=True)
                run = directory / "run"
                if run.resolve() != run:
                    raise ValueError(
                        "Study run directory follows a link or escapes its case"
                    )
                case_clock = _launch.launch_clock(
                    directory / "launch_time.json",
                    durations={}
                    if job["duration_seconds"] is None
                    else {"deadline_at": job["duration_seconds"]},
                    binding={
                        "declaration_sha256": _digest(declared),
                        "case_id": job["id"],
                    },
                    create=not (
                        row["attempts"]
                        or run.exists()
                        or (directory / "process.json").exists()
                    ),
                )
                deadline = min(
                    clock.get("deadline_at", math.inf),
                    case_clock.get("deadline_at", math.inf),
                )
                stop_at = deadline - reserve
                if stop_at <= time.time():
                    row["status"] = "time_limit"
                    if declared["failure_policy"] == "stop":
                        state["status"] = "incomplete"
                        break
                    continue
                previous = bool(row["attempts"]) or run.exists()
                try:
                    checkpoint = _latest(run) if previous else None
                    restart_empty = (
                        resume
                        and checkpoint is None
                        and (
                            not run.exists()
                            or (run.is_dir() and not any(run.iterdir()))
                        )
                    )
                except (ValueError, OSError) as error:
                    row.update(status="failed", error=str(error))
                    if declared["failure_policy"] == "stop":
                        state["status"] = "incomplete"
                        break
                    continue
                if previous and (
                    not resume or (checkpoint is None and not restart_empty)
                ):
                    row.update(
                        status="failed",
                        error=(
                            f"No complete checkpoint is available in {run}; "
                            "the case is not restarted. Preserve any partial work. "
                            "Restore a complete checkpoint to resume, or declare "
                            "a new study in another output folder."
                        ),
                    )
                    if declared["failure_policy"] == "stop":
                        state["status"] = "incomplete"
                        break
                    continue
                if row["attempts"] and "finished_at" not in row["attempts"][-1]:
                    row["attempts"][-1].update(
                        interrupted=True,
                        recovered_at=utc_now(),
                        error="Previous controller ended before recording an outcome",
                    )
                attempt: Record = {
                    "number": len(row["attempts"]) + 1,
                    "started_at": utc_now(),
                    "resume_from": None if checkpoint is None else str(checkpoint),
                }
                row["attempts"].append(attempt)
                row.pop("error", None)
                row.update(status="running", run_dir=str(run))
                atomic_json(
                    directory / "request.json",
                    {
                        "job": job,
                        "resume_from": attempt["resume_from"],
                        "attempt": attempt["number"],
                    },
                )
                atomic_json(root / "study.json", state)
                try:
                    code = _launch.supervise_command(
                        directory,
                        _command("_worker", str(root), job["id"]),
                        deadline_at=None if math.isinf(stop_at) else stop_at,
                    )
                except Exception as error:
                    code = 1
                    attempt["error"] = f"{type(error).__name__}: {error}"
                _finished_supervisor(directory)
                attempt.update(exit_code=code, finished_at=utc_now())
                outcome = _optional(directory / "outcome.json")
                if outcome.get("request_sha256") != _digest(
                    _read(directory / "request.json")
                ):
                    outcome = {
                        "status": "failed",
                        "error": "No outcome for this exact attempt",
                    }
                saved_status = _optional(run / "status.json")
                row["run_status"] = saved_status
                row["status"] = (
                    "complete"
                    if code == 0
                    and saved_status.get("status") == "complete"
                    and outcome.get("status") == "complete"
                    else "time_limit"
                    if code == 124
                    else "failed"
                )
                if outcome.get("error"):
                    attempt["error"] = outcome["error"]
                atomic_json(root / "study.json", state)
                _report(root, declared, state)
                if _launch.live_process_records([directory / "process.json"]):
                    row["error"] = (
                        "Worker cleanup is incomplete; no further case can start"
                    )
                    state["status"] = (
                        "stopped" if _stopped(directory, code) else "incomplete"
                    )
                    break
                if _stopped(directory, code):
                    state["status"] = "stopped"
                    break
                if row["status"] != "complete" and declared["failure_policy"] == "stop":
                    state["status"] = "incomplete"
                    break
            else:
                state["status"] = (
                    "complete"
                    if all(
                        row["status"] == "complete" for row in state["cases"].values()
                    )
                    else "incomplete"
                )
            available = [
                root / "cases" / job["id"] / "run"
                for job in _jobs(declared)
                if (root / "cases" / job["id"] / "run/run_details.json").exists()
            ]
            if available and state["status"] != "stopped":
                analysis_job = root / "analysis_job"
                analysis_job.mkdir(exist_ok=True)
                code = 1
                try:
                    code = _launch.supervise_command(
                        analysis_job,
                        _command("_analyze", str(root)),
                        env={**os.environ, "JAX_PLATFORMS": "cpu"},
                    )
                    outcome = _optional(analysis_job / "outcome.json")
                    state["analysis"] = (
                        outcome
                        if code == 0 and outcome.get("status") == "complete"
                        else {
                            "status": "failed",
                            "exit_code": code,
                            "error": outcome.get(
                                "error", "Analysis did not finish this attempt"
                            ),
                        }
                    )
                except Exception as error:
                    state["analysis"] = {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                finally:
                    _finished_supervisor(analysis_job)
                if _stopped(analysis_job, code):
                    state["status"] = "stopped"
                    state["analysis"]["status"] = "stopped"
            atomic_json(root / "study.json", state)
            return _report(root, declared, state)
        except Exception as error:
            state.update(status="incomplete", error=f"{type(error).__name__}: {error}")
            atomic_json(root / "study.json", state)
            _report(root, declared, state)
            raise
        finally:
            process.update(
                state="exited",
                finished_at=utc_now(),
                controller_identity=own,
                process=None,
            )
            atomic_json(root / "process.json", process)


def run_study(
    config: Declaration | None = None,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> Record:
    """Run a declared study in order and return its saved JSON report.

    Parameters
    ----------
    config : path, mapping or None, default=None
        A schema-1 declaration with name and cases. Each fresh case has id,
        config and distinct seeds; config names method and total_env_steps.
        A continuation instead has checkpoint, additional_env_steps and optional
        changes. Defaults are frozen before work. failure_policy defaults to
        "stop"; "continue" leaves failed cases visible and moves to the next.
        Optional duration_seconds on the study and cases limits wall-clock
        work from their first launches, including downtime. finalist_of links
        a separate declaration to an existing discovery study. File-relative
        references resolve beside the declaration. A mapping uses the current
        directory. Source and method dependency identities are frozen automatically;
        workers refuse a changed install or file input. Panel/result bytes and
        actor/checkpoint descriptions are pinned automatically. Factory references
        do not freeze remote service state or external provider behavior.
        None is accepted only when resuming.
    output_dir : path or None, default=None
        Exact new or empty study folder. Required for a new study and mutually
        exclusive with resume_from. No generated parent directory is added.
        In an editable Git checkout, use an ignored folder or one outside the
        checkout so result files do not change the saved source identity.
    resume_from : path or None, default=None
        Existing study folder. Reuses its declaration and original deadlines.
        A supplied config must resolve to the same declaration. Only explicit
        resume may recover an unfinished case from its latest saved full
        checkpoint. If no checkpoint exists and the case's run folder is missing
        or empty, explicit resume may start that same case again. A nonempty run
        without a checkpoint is refused. Attempts, seeds and deadlines are kept;
        a restart never adds time or removes partial work.

    Returns
    -------
    dict
        The finite JSON report also saved as report.json, with every declared
        case and all attempts. Successful cases survive another case's failure
        or an analysis error. Ordinary training owns each run's saved selection;
        this driver makes no selection across runs and no learning claim.

    Raises
    ------
    ValueError, TypeError
        Declaration, output, saved clock or checkpoint settings are invalid.
    RuntimeError
        A live study or worker already owns this output.
    OSError
        Study files or subprocesses cannot be read or written.

    Notes
    -----
    Uses supervised subprocesses in the caller's environment, one model at a
    time. Status and import need no numerical backend. Config validation and
    analysis use CPU subprocesses without starting the caller's numerical
    backend. Numerical workers stop early enough to reserve the launcher's
    bounded cleanup time inside each saved deadline. A case with no remaining
    execution time starts no worker. Report rebuilding may finish later.
    Completed files remain available.
    """
    root, _, resume = _prepare(config, output_dir, resume_from)
    return _execute(root, resume=resume)


def start_study(
    config: Declaration | None = None,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> Record:
    """Launch run_study detached and return its saved controller identity.

    config, output_dir and resume_from have the same meaning and checks as
    run_study. The first launch fixes deadlines before the child starts; resume
    keeps them. stdin is closed and output appends to logs/study.log. The child
    inherits Python paths, device visibility and runtime settings. A pipe gate
    prevents work before its process identity is saved. Duplicate live owners
    raise RuntimeError. Declaration, file and process errors propagate.
    """
    root, declared, resume = _prepare(config, output_dir, resume_from)
    with _launch.launch_lock(root):
        if _launch.live_process_records(_process_paths(root)):
            raise RuntimeError("This study still has an active owned process")
        _clock(root, declared, create=not resume or not (root / "study.json").exists())
        command = _command("_control", str(root), *(["--resume"] if resume else []))
        return _launch.spawn_detached(
            root,
            command,
            log_path=root / "logs/study.log",
            mode="resume" if resume else "start",
        )


def study_status(output_dir: str | Path) -> Record:
    """Read study state and exact live owners without JAX or file changes.

    output_dir is an existing study folder. Returns saved declaration identity,
    clock, study state, report and live process-record paths. Missing results
    remain empty; missing or malformed declarations raise ValueError/OSError.
    It does not recover, restart or otherwise mutate an interrupted run.
    """
    root = Path(output_dir).expanduser().resolve()
    return {
        "study_dir": str(root),
        "declaration_sha256": _digest(_read(root / _DECLARATION)),
        "clock": _optional(root / "launch_time.json"),
        "study": _optional(root / "study.json"),
        "report": _optional(root / "report.json"),
        "owned_live_records": [
            str(path) for path in _launch.live_process_records(_process_paths(root))
        ],
    }


def stop_study(output_dir: str | Path) -> Record:
    """Request a stop from exact live study owners and return current status.

    output_dir names an existing study. Shared launcher checks guard PID reuse
    and signal only matching process owners. This is a request, not proof that
    cleanup has finished. Status stays backend-free and files are not removed.
    Read, permission and signalling errors propagate.
    """
    root = Path(output_dir).expanduser().resolve()
    signalled = _launch.request_stop(_process_paths(root))
    return {**study_status(root), "stop_requested_pids": signalled}


def _worker(root: Path, identifier: str) -> int:
    """Run one saved case through ordinary train or explicit continuation."""
    from marl_battlegrounds.training.runner import (
        config_from_dict,
        extend_training,
        train,
    )

    directory = _case_directory(root, identifier)
    request = _read(directory / "request.json")
    job = request["job"]
    try:
        if (directory / "run").resolve() != directory / "run":
            raise ValueError("Study run directory follows a link or escapes its case")
        declared = _read(root / _DECLARATION)
        declared_jobs = {row["id"]: row for row in _jobs(declared)}
        if job != declared_jobs.get(identifier):
            raise ValueError("Study worker request differs from its frozen declaration")
        method = (
            job["config"]["method"]
            if job["kind"] == "train"
            else job["parent_identity"]["method"]
        )
        actual_identity = _identities([method])[method]
        if actual_identity != declared["runtime_identities"][method]:
            raise ValueError(
                "Study worker source or dependencies differ from the "
                "frozen declaration. In an editable checkout, use an ignored "
                "output folder such as artifacts/ or a folder outside the checkout."
            )
        if _file_inputs(job) != job.get("external_inputs", {}):
            raise ValueError(
                "Study external file input differs from its frozen identity"
            )
        if request["resume_from"] is not None and Path(
            request["resume_from"]
        ) != _latest(directory / "run"):
            raise ValueError(
                "Study worker resume differs from its saved latest checkpoint"
            )
        if request["resume_from"] is not None:
            if job["kind"] == "train":
                result = train(
                    config_from_dict(job["config"]), resume_from=request["resume_from"]
                )
            else:
                from marl_battlegrounds.training.checkpoints import (
                    read_checkpoint_description,
                )

                child = read_checkpoint_description(request["resume_from"])
                context = child["metadata"].get("continuation", {})
                changes = _object(job["changes"], "Continuation changes")
                if isinstance(changes.get("validation"), dict):
                    changes["validation"].pop("bindings", None)
                expected = {
                    "parent_checkpoint": job["checkpoint"],
                    "parent_checkpoint_id": job["parent_identity"]["checkpoint_id"],
                    "additional_env_steps": job["additional_env_steps"],
                    "changes": changes,
                }
                if any(context.get(key) != value for key, value in expected.items()):
                    raise ValueError(
                        "Study continuation child differs from "
                        "its declared parent or extension"
                    )
                result = train(resume_from=request["resume_from"])
        elif job["kind"] == "train":
            result = train(
                config_from_dict(job["config"]), output_dir=directory / "run"
            )
        else:
            if _checkpoint_identity(job["checkpoint"]) != job["parent_identity"]:
                raise ValueError(
                    "Study continuation parent differs from its frozen identity"
                )
            result = extend_training(
                job["checkpoint"],
                additional_env_steps=job["additional_env_steps"],
                output_dir=directory / "run",
                changes=job["changes"],
            )
        atomic_json(
            directory / "outcome.json",
            {
                "status": result.status,
                "request_sha256": _digest(request),
                "runtime_identity": actual_identity,
                "run_dir": str(result.run_dir),
                "completed_env_steps": result.completed_env_steps,
                "completed_updates": result.completed_updates,
                "final_actor": str(result.final_actor),
                "selected_actor": None
                if result.selected_actor is None
                else str(result.selected_actor),
            },
        )
        return 0
    except Exception as error:
        atomic_json(
            directory / "outcome.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "request_sha256": _digest(request),
            },
        )
        return 1


def _analyze(root: Path) -> int:
    """Write ordinary saved-run analysis, preserving failures as report facts."""
    from marl_battlegrounds.training.analysis import analyze

    try:
        runs = [
            root / "cases" / job["id"] / "run"
            for job in _jobs(_read(root / _DECLARATION))
            if (root / "cases" / job["id"] / "run/run_details.json").exists()
        ]
        report = analyze(runs, output_dir=root / "analysis")
        atomic_json(
            root / "analysis_job/outcome.json", {"status": "complete", "report": report}
        )
        return 0
    except Exception as error:
        atomic_json(
            root / "analysis_job/outcome.json",
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run private subprocess roles; public commands use the package CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("_control", "_worker", "_analyze", "_normalize")
    )
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("case_id", nargs="?")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "_normalize":
        try:
            declared = _normalize(_read(args.study_dir / "request.json")["config"])
            atomic_json(args.study_dir / "outcome.json", {"declaration": declared})
            return 0
        except Exception as error:
            atomic_json(
                args.study_dir / "outcome.json",
                {"error": f"{type(error).__name__}: {error}"},
            )
            return 1
    if args.action == "_control":
        result = _execute(args.study_dir, resume=args.resume, handoff=True)
        return 0 if result["status"] == "complete" else 1
    if args.action == "_worker":
        if args.case_id is None:
            parser.error("_worker requires a case ID")
        return _worker(args.study_dir, args.case_id)
    return _analyze(args.study_dir)


if __name__ == "__main__":
    raise SystemExit(main())
