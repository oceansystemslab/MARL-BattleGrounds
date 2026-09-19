"""Own the small host files, lock and progress display for one training run.

Progress, locks and logs use the standard library only. Saved host recovery
also reads checkpoint and validation identities through their existing owners.
No helper restores model arrays, starts a backend or synchronizes a GPU.
Progress uses summaries already needed by the runner.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

if TYPE_CHECKING:
    from marl_battlegrounds.training.validation import FrozenPanel


def utc_now() -> str:
    """Return the current UTC timestamp with an explicit timezone."""
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: object) -> None:
    """Durably replace one JSON file on its local filesystem.

    path's parent must exist. value must contain finite JSON values. Write a
    unique temporary sibling, fsync its contents, replace path and fsync the
    parent. Readers see the old or complete new file. Errors propagate; cleanup
    removes only this call's unpublished temporary file.
    """
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, value: dict[str, Any], *, durable: bool = False) -> None:
    """Append one finite JSON record; optionally fsync at a lifecycle boundary.

    The caller holds the run lock. Each call closes the file, so later readers
    can see the row. Checkpoint publication separately fsyncs and hashes the
    update-log prefix. For durable lifecycle events, preserve a torn previous
    line and separate the new record so analysis can report the incomplete
    event. Ordinary update rows rely on checked checkpoint recovery instead.
    Ordinary progress display never adds a write here.
    """
    separator = ""
    if durable and path.exists() and path.stat().st_size:
        with path.open("rb") as previous:
            previous.seek(-1, os.SEEK_END)
            if previous.read(1) != b"\n":
                separator = "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(separator)
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        if durable:
            stream.flush()
            os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON-object lines; reject malformed JSON or a non-object record.

    A missing file means no records. This reader does not repair or truncate
    data; only validated checkpoint recovery may rewind the training prefix.
    A valid final JSON object is accepted even without a trailing newline.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {path}")
        records.append(cast(dict[str, Any], value))
    return records


def log_cursor(path: Path) -> dict[str, Any]:
    """Fsync a log and return its exact byte count and SHA-256 recovery prefix.

    Create an empty log when absent. Stream the file in bounded chunks instead
    of retaining its contents. The caller holds the training run lock.
    """
    with path.open("ab") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def validate_log_cursor(path: Path, cursor: dict[str, Any]) -> None:
    """Validate a saved byte prefix without changing the log.

    Reject missing/short files, unsafe counts, changed prefix bytes or a prefix
    ending inside a row. Bytes after a valid prefix are allowed for rollback.
    """
    size = cursor.get("size_bytes")
    if type(size) is not int or size < 0:
        raise ValueError("Invalid training log byte count")
    digest = hashlib.sha256()
    remaining = size
    last = b""
    with path.open("rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("Training log is shorter than its checkpoint prefix")
            digest.update(block)
            remaining -= len(block)
            last = block[-1:]
    if digest.hexdigest() != cursor.get("sha256") or (size and last != b"\n"):
        raise ValueError("Training log prefix does not match its checkpoint")


def restore_log_cursor(path: Path, cursor: dict[str, Any]) -> None:
    """Recheck and durably truncate one log after full learner validation.

    The run lock and persisted recovery marker must already be held by the
    caller. This is idempotent for the same validated checkpoint prefix.
    """
    validate_log_cursor(path, cursor)
    with path.open("r+b") as stream:
        stream.truncate(cursor["size_bytes"])
        stream.flush()
        os.fsync(stream.fileno())


def preserve_log_suffix(
    path: Path, cursor: dict[str, Any], directory: Path
) -> Path | None:
    """Preserve abandoned update bytes before validated checkpoint rewind.

    A valid saved prefix is required. Stream later bytes into a unique temporary
    file, then publish under their content hash in directory. Repeated recovery
    keeps the same evidence file. Return None when there are no later bytes.
    Even a torn final record remains visible here; no abandoned row is used as
    active scientific data. The caller holds the run lock and recovery marker.
    """
    validate_log_cursor(path, cursor)
    if path.stat().st_size == cursor["size_bytes"]:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{uuid.uuid4().hex}.tmp"
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source, temporary.open("xb") as target:
            source.seek(cursor["size_bytes"])
            while block := source.read(1024 * 1024):
                digest.update(block)
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        destination = directory / f"{digest.hexdigest()}.jsonl"
        if destination.exists():
            if (
                hashlib.sha256(destination.read_bytes()).hexdigest()
                != digest.hexdigest()
            ):
                raise ValueError("Abandoned-attempt log identity conflict")
        else:
            os.replace(temporary, destination)
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(run_dir: Path) -> Generator[None]:
    """Hold an exclusive nonblocking local-process lock until context exit.

    run_dir must exist. A second writer raises RuntimeError before changing run
    records. The persistent lock file is harmless after exit or a process crash;
    kernel ownership, not a stale PID, decides whether another run may start.
    """
    with (run_dir / ".training.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "This training run already has an active writer"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def process_identity(pid: int | None = None) -> dict[str, Any]:
    """Return PID and Linux start/boot identity, or missing start fields.

    pid defaults to this process. Reading /proc performs no signal or process
    mutation. Missing/exited processes have start_ticks=None. The boot identity
    prevents accidental PID/start reuse after a machine restart.
    """
    selected = os.getpid() if pid is None else pid
    try:
        stat = Path(f"/proc/{selected}/stat").read_text()
        start: str | None = stat.rsplit(")", 1)[1].split()[19]
        boot: str | None = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError, IndexError:
        start, boot = None, None
    return {"pid": selected, "start_ticks": start, "boot_id": boot}


def duration(seconds: float | None) -> str:
    """Format a nonnegative duration as hours/minutes/seconds or Estimating."""
    if seconds is None:
        return "Estimating"
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TrainingSpeedEstimate:
    """Estimate future speed from up to five existing warmed update timings.

    Create one instance per process attempt, including resume. The first update
    may compile and is excluded. Two later valid samples are needed before rate
    becomes available. This small host window is not learner or checkpoint
    state; it reads no clock, arrays or files.
    """

    def __init__(self) -> None:
        """Start a new attempt with no measured rate or warmed samples."""
        self._first = True
        self._samples: deque[tuple[int, float]] = deque(maxlen=5)
        self.rate: float | None = None

    def observe(self, transitions: int, seconds: float) -> None:
        """Add already measured real transitions and collection/update seconds.

        Skip this attempt's first update. Invalid or empty timings clear the
        estimate; two positive finite samples are then needed again. rate uses
        total transitions divided by total seconds in the recent window, so a
        partial final batch does not count padded rows as completed experience.
        Cumulative measured training speed is owned separately by the runner.
        """
        if self._first:
            self._first = False
            return
        if transitions <= 0 or seconds <= 0 or not math.isfinite(seconds):
            self._samples.clear()
            self.rate = None
            return
        self._samples.append((transitions, seconds))
        if len(self._samples) >= 2:
            self.rate = sum(row[0] for row in self._samples) / sum(
                row[1] for row in self._samples
            )


def progress_text(status: dict[str, Any]) -> str:
    """Format existing host records without fetching arrays or reading files.

    Missing warmed timing or future-phase measurements produce Estimating.
    estimated_transitions_per_second predicts remaining work; measured training
    throughput includes compilation and excludes validation/saving. An exhausted
    budget has zero training time left even after a fresh resume. elapsed_seconds
    carries the larger
    saved checkpoint or same-run status value, then adds the current attempt.
    Interrupted work without a durable timing record may be missing; wall_seconds
    includes the gaps between attempts.
    Fields are named by the runner. This display is not a learning verdict.
    """
    steps = int(status.get("env_steps", 0))
    total = int(status.get("total_env_steps", 0))
    rate = status.get("transitions_per_second")
    recent = status.get("recent_transitions_per_second")
    estimate = status.get("estimated_transitions_per_second")
    remaining = (
        0.0
        if total and steps >= total
        else max(0, total - steps) / estimate
        if estimate and estimate > 0
        else None
    )
    later = status.get("pending_work_seconds")
    eta = remaining + later if remaining is not None and later is not None else None
    phase = str(status.get("phase", "Starting")).replace("_", " ").capitalize()
    parts = [
        f"{phase}: {steps:,}/{total:,} environment transitions ({steps / total:.1%})"
        if total
        else f"{phase}: {steps:,} environment transitions",
        f"Updates {status.get('completed_updates', 0):,}",
        f"Elapsed {duration(status.get('elapsed_seconds'))}",
        f"Training Speed {rate:,.1f}/s" if rate else "Training Speed Estimating",
        f"Recent {recent:,.1f}/s" if recent else "Recent Estimating",
        f"Training ETA {duration(remaining)}",
        f"Whole-Run ETA {duration(eta)}",
    ]
    for name, label in (
        ("policy_loss", "Policy Loss"),
        ("value_loss", "Value Loss"),
        ("entropy", "Entropy"),
        ("task_reward_mean", "Task Reward"),
        ("shaping_mean", "Shaping"),
        ("validation_score", "Validation Score"),
    ):
        value = status.get(name)
        if value is not None:
            parts.append(f"{label} {value:.5g}")
    if status.get("latest_checkpoint"):
        parts.append(f"Checkpoint {status['latest_checkpoint']}")
    if status.get("error"):
        parts.append(f"Error {status['error']}")
    return " | ".join(parts)


class ProgressReporter:
    """Print existing summaries at most once per ten seconds or a phase change.

    enabled=False returns before clock reads or formatting. stream defaults to
    stdout. The caller supplies existing host data; this helper starts no thread,
    file writer, model call, measurement or device transfer.
    """

    def __init__(self, *, enabled: bool, stream: TextIO | None = None) -> None:
        """Set output options and start with no previous displayed phase."""
        self.enabled = enabled
        self.stream = sys.stdout if stream is None else stream
        self.last_time = float("-inf")
        self.last_phase: str | None = None

    def report(self, status: dict[str, Any], *, force: bool = False) -> None:
        """Print status when due; force also prints completion or failure."""
        if not self.enabled:
            return
        now = time.monotonic()
        phase = str(status.get("phase"))
        if force or phase != self.last_phase or now - self.last_time >= 10:
            print(progress_text(status), file=self.stream, flush=True)
            self.last_time, self.last_phase = now, phase


def validate_host_state(
    run_dir: Path,
    checkpoint_details: dict[str, Any],
    *,
    panel: FrozenPanel | None,
) -> None:
    """Reject malformed saved runner work before any writer or log rewind.

    Parameters
    ----------
    run_dir : Path
        Original run directory under the caller's exclusive run lock.
    checkpoint_details : dict
        Complete learner description already verified by restore_checkpoint.
        Its config and numerical counters must already match the resumed state.
    panel : FrozenPanel or None
        Already verified current panel, or None for a run without validation.

    Returns
    -------
    None
        Saved host counts, timings, pending capture, active ancestry, exported
        actors, validation records and selection agree. Inputs remain unchanged.

    Raises
    ------
    ValueError, OSError
        A field is missing, malformed, inconsistent or points outside this run;
        an actor or completed validation artifact is missing or changed; or a
        candidate belongs to an abandoned continuation. A pending current export
        need not exist. Ancestor descriptions remain usable after payload pruning.

    Notes
    -----
    Read-only and host-only. Existing checkpoint and selection owners verify
    artifact identities and selection rules. Numerical arrays are never restored
    or transferred, and no evaluator, writer or training backend is started.
    Call this after complete learner restore and before resume_recording.
    """
    from marl_battlegrounds.training import analysis, checkpoints, validation

    root = run_dir.resolve()
    metadata = checkpoint_details["metadata"]
    config = metadata["config"]
    value: object = metadata.get("host_state")
    if not isinstance(value, dict):
        raise ValueError("Saved host state must be a JSON object")
    host = cast(dict[str, Any], value)
    counts = (
        "env_steps",
        "completed_updates",
        "actor_decisions",
        "used_policy_samples",
        "used_value_samples",
        "validation_games",
        "saves",
    )
    times = (
        "training_seconds",
        "elapsed_seconds",
        "validation_seconds",
        "save_seconds",
    )
    required = set(counts + times) | {
        "routine_results",
        "confirmation_results",
        "actors",
        "pending",
        "selected_actor",
        "final_actor",
        "report_seconds",
        "slot_complete",
        "selection",
        "recovery_checkpoints",
    }
    stage_shapes = {
        "stage_completed": (17, 3),
        "stage_score_sums": (17, 2),
        "stage_length_sum": (17,),
        "stage_k20_count": (17,),
    }
    if not required <= set(host) or set(host) - required - stage_shapes.keys():
        raise ValueError("Saved host state fields differ from the runner schema")

    def integer(item: object, name: str) -> int:
        """Require a nonnegative host count; bool is never a count."""
        if type(item) is not int or item < 0:
            raise ValueError(f"Saved {name} must be a nonnegative integer")
        return item

    def seconds(item: object, name: str) -> float:
        """Require finite nonnegative seconds without accepting a Boolean."""
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item < 0
        ):
            raise ValueError(f"Saved {name} must be finite nonnegative seconds")
        return float(item)

    def identifier(item: object) -> str:
        """Require an immutable lowercase SHA-256 identifier."""
        if (
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
        ):
            raise ValueError("Saved artifact identity must be a SHA-256 digest")
        return item

    def read(path: Path) -> dict[str, Any]:
        """Read one run-owned JSON object without following a linked directory."""
        if (
            not path.is_relative_to(root)
            or path.resolve() != path
            or not path.is_file()
        ):
            raise ValueError("Saved work must use a regular file inside its run")
        loaded: object = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError("Saved work file must contain a JSON object")
        return cast(dict[str, Any], loaded)

    for name in counts:
        integer(host[name], name)
    for name in times:
        seconds(host[name], name)
    if host["report_seconds"] is not None:
        seconds(host["report_seconds"], "report_seconds")
    if type(host["slot_complete"]) is not bool:
        raise ValueError("Saved slot_complete must be bool")
    if any(
        host[name] > host["elapsed_seconds"] + 1e-6
        for name in times
        if name != "elapsed_seconds"
    ):
        raise ValueError("Saved phase time exceeds the complete elapsed time")
    steps, updates = host["env_steps"], host["completed_updates"]
    batch, total = config["num_envs"], config["total_env_steps"]
    epochs, length = config["ppo"]["epochs"], config["ppo"]["rollout_length"]
    if (
        {"env_steps": steps, "updates": updates} != checkpoint_details["counters"]
        or steps > total
        or steps % batch
        or updates != math.ceil(steps / (batch * length))
        or host["actor_decisions"] > steps * 5
        or host["used_policy_samples"] != host["actor_decisions"] * epochs
        or not steps * epochs <= host["used_value_samples"] <= steps * 5 * epochs
        or host["used_policy_samples"] > host["used_value_samples"]
    ):
        raise ValueError("Saved host experience or sample counts disagree")
    present_stages = set(host) & stage_shapes.keys()
    if (present_stages and present_stages != stage_shapes.keys()) or (
        updates and not present_stages
    ):
        raise ValueError("Saved stage summaries are incomplete")
    for name in present_stages:
        rows = host[name]
        shape = stage_shapes[name]
        if not isinstance(rows, list):
            raise ValueError("Saved stage summary shape differs")
        rows = cast(list[Any], rows)
        if len(rows) != shape[0]:
            raise ValueError("Saved stage summary shape differs")
        for row in rows:
            if len(shape) == 2:
                if not isinstance(row, list) or len(cast(list[Any], row)) != shape[1]:
                    raise ValueError("Saved stage summary shape differs")
                for cell in cast(list[Any], row):
                    integer(cell, name)
            else:
                integer(row, name)
    if present_stages:
        for index, outcomes in enumerate(host["stage_completed"]):
            finished = sum(outcomes)
            if (
                host["stage_length_sum"][index] < finished
                or host["stage_k20_count"][index] > finished
            ):
                raise ValueError("Saved stage completion counts disagree")
            if not finished and (
                sum(host["stage_score_sums"][index]) or host["stage_length_sum"][index]
            ):
                raise ValueError("Empty stage has completed-game measurements")
        if sum(sum(row) for row in host["stage_completed"]) > steps:
            raise ValueError("Completed games exceed real transitions")

    ancestry: dict[str, dict[str, Any]] = {}
    current = checkpoint_details
    previous_counts = checkpoint_details["counters"]
    while True:
        key = identifier(current["checkpoint_id"])
        if key in ancestry or current["kind"] != "learner":
            raise ValueError("Saved checkpoint ancestry is cyclic or invalid")
        context = current["metadata"]
        for name in ("run_id", "config", "source", "dependencies"):
            if context[name] != metadata[name]:
                raise ValueError("Saved ancestor belongs to a different experiment")
        if context.get("execution") != metadata.get("execution"):
            raise ValueError("Saved ancestor uses a different execution identity")
        for name, maximum in previous_counts.items():
            if integer(current["counters"][name], name) > maximum:
                raise ValueError("Saved ancestor follows its descendant")
        previous_counts = current["counters"]
        ancestry[key] = current
        parent = context["parent_checkpoint"]
        if parent is None:
            break
        parent = identifier(parent)
        directory = root / "checkpoints" / parent
        if directory.resolve() != directory:
            raise ValueError("Saved ancestry cannot follow a linked directory")
        current = checkpoints.read_checkpoint_description(directory)
        if current["checkpoint_id"] != parent:
            raise ValueError("Ancestor directory and description identities differ")

    recent = host["recovery_checkpoints"]
    if not isinstance(recent, list):
        raise ValueError("Saved recovery checkpoint list is invalid")
    recent = [identifier(key) for key in cast(list[Any], recent)]
    if len(recent) > 3 or len(set(recent)) != len(recent):
        raise ValueError("Saved recovery checkpoint list is invalid")
    if any(identifier(key) not in ancestry for key in recent):
        raise ValueError("Saved retention record is outside active ancestry")
    actors = host["actors"]
    if not isinstance(actors, dict):
        raise ValueError("Saved actor paths must be a mapping")
    actor_records: dict[str, dict[str, Any]] = {}
    for key, path in cast(dict[str, Any], actors).items():
        key = identifier(key)
        expected_path = root / "actors" / key
        if (
            key not in ancestry
            or path != str(expected_path)
            or expected_path.resolve() != expected_path
        ):
            raise ValueError("Saved actor path or continuation identity differs")
        actor = checkpoints.artifact_identity(expected_path)
        ancestor = ancestry[key]
        if (
            actor["metadata"].get("checkpoint_id") != key
            or actor["run_id"] != metadata["run_id"]
            or type(actor["seed"]) is not int
            or type(actor["env_steps"]) is not int
            or actor["seed"] != config["seed"]
            or actor["env_steps"] != ancestor["counters"]["env_steps"]
            or actor["actor_digest"] != ancestor["actor_digest"]
            or actor["schemas"] != ancestor["schemas"]
        ):
            raise ValueError("Saved actor identity differs from its learner boundary")
        actor_records[key] = actor

    points: set[int] = (
        {
            point.env_steps
            for point in validation.resolve_validation_schedule(
                total,
                batch,
                rollout_length=length,
                fractions=config["validation_fractions"],
            )
        }
        if panel is not None
        else set()
    )
    pending = host["pending"]
    if pending is not None:
        if (
            not isinstance(pending, dict)
            or set(cast(dict[str, Any], pending)) != {"routine"}
            or type(cast(dict[str, Any], pending)["routine"]) is not bool
        ):
            raise ValueError("Saved pending work must declare one Boolean routine flag")
        if pending["routine"] != (steps in points):
            raise ValueError("Saved pending validation differs from its schedule")
        if steps not in points | set(config["checkpoint_env_steps"]) | {0, total}:
            raise ValueError("Saved pending capture is outside its declared schedule")
    if host["final_actor"] is not None:
        if steps != total or host["final_actor"] not in actors.values():
            raise ValueError("Saved final actor is outside its completed run")
        final_id = Path(host["final_actor"]).name
        if actor_records[final_id]["env_steps"] != total:
            raise ValueError("Saved final actor has not reached the full budget")
    elif steps == total and pending is None:
        raise ValueError("Saved completed training has neither final actor nor capture")
    if host["slot_complete"] and (
        not config["slot_diagnostic"] or host["final_actor"] is None
    ):
        raise ValueError("Saved slot diagnostic completion has no declared final actor")

    seen: set[str] = set()
    routine_steps: set[int] = set()
    validation_games = 0
    for name, purpose in (
        ("routine_results", "routine"),
        ("confirmation_results", "confirmation"),
    ):
        results = host[name]
        if not isinstance(results, list) or (panel is None and results):
            raise ValueError("Saved validation results differ from the declared panel")
        for result in cast(list[Any], results):
            if not isinstance(result, dict):
                raise ValueError("Saved validation result must be an object")
            result = cast(dict[str, Any], result)
            key = identifier(result.get("checkpoint_id"))
            if key not in actor_records or panel is None:
                raise ValueError("Saved validation refers to an unknown actor")
            actor = actor_records[key]
            pairs = config[f"{purpose}_seed_pairs"]
            expected_task = validation.validation_task_description(
                checkpoint_id=key,
                actor_digest=actor["actor_digest"],
                env_steps=actor["env_steps"],
                panel_digest=panel.digest,
                purpose=purpose,
                seed_pairs=pairs,
                members=tuple(
                    (member.name, member.actor_digest) for member in panel.members
                ),
            )
            directory = root / "validation" / f"{purpose}-{key}"
            task_file = read(directory / "task.json")
            if json.dumps(task_file, sort_keys=True, allow_nan=False) != json.dumps(
                expected_task, sort_keys=True, allow_nan=False
            ) or any(
                result.get(field) != item for field, item in expected_task.items()
            ):
                raise ValueError("Saved validation task identity differs")
            if result["task_id"] in seen or result.get("complete") is not True:
                raise ValueError("Saved validation task is duplicate or incomplete")
            seen.add(result["task_id"])
            if read(directory / "validation_summary.json") != {
                field: item
                for field, item in result.items()
                if field != "elapsed_seconds"
            }:
                raise ValueError("Saved validation summary changed")
            if (
                seconds(result.get("elapsed_seconds"), "validation elapsed_seconds")
                > host["elapsed_seconds"] + 1e-6
            ):
                raise ValueError("Saved validation follows its checkpoint time")
            for field in ("score", "ci_low", "ci_high"):
                score = result.get(field)
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError("Saved validation score or interval is invalid")
            if result["ci_low"] > result["ci_high"]:
                raise ValueError("Saved validation interval is reversed")
            games = 2 * len(validation.VALIDATION_MAPS) * len(panel.members) * pairs
            if integer(result.get("games"), "validation games") != games:
                raise ValueError("Saved validation game count differs")
            paths = result.get("pass_paths")
            if not isinstance(paths, list) or len(cast(list[Any], paths)) != len(
                panel.members
            ):
                raise ValueError("Saved validation pass paths are incomplete")
            for index, (path, member) in enumerate(
                zip(cast(list[Any], paths), panel.members, strict=True)
            ):
                if not isinstance(path, str):
                    raise ValueError("Saved validation pass path must be text")
                actual_path = Path(path)
                parent = directory / f"opponent-{index}"
                if (
                    not actual_path.is_relative_to(parent)
                    or actual_path.resolve() != actual_path
                ):
                    raise ValueError("Saved validation pass is outside its task")
                pass_id = validation.validation_pass_id(result["task_id"], member.name)
                if (
                    validation._saved_run(parent, pass_id) != actual_path  # pyright: ignore[reportPrivateUsage]
                    or validation._pending(  # pyright: ignore[reportPrivateUsage]
                        actual_path,
                        pass_id=pass_id,
                        total=2 * len(validation.VALIDATION_MAPS) * pairs,
                    )
                    != 0
                ):
                    raise ValueError("Saved validation pass is missing or incomplete")
            validation_games += games
            if purpose == "routine":
                if (
                    actor["env_steps"] not in points
                    or actor["env_steps"] in routine_steps
                ):
                    raise ValueError(
                        "Saved routine validation schedule is inconsistent"
                    )
                routine_steps.add(actor["env_steps"])
    required_points = {point for point in points if point <= steps}
    if pending is not None and pending["routine"]:
        required_points.discard(steps)
    if (
        not required_points <= routine_steps
        or host["validation_games"] != validation_games
    ):
        raise ValueError("Saved completed validation coverage is incomplete")
    confirmations = host["confirmation_results"]
    if confirmations:
        if host["final_actor"] is None:
            raise ValueError("Saved confirmation has no final checkpoint")
        candidates = analysis.confirmation_candidates(
            host["routine_results"], final_checkpoint_id=Path(host["final_actor"]).name
        )
        if not {row["checkpoint_id"] for row in confirmations} <= set(candidates):
            raise ValueError("Saved confirmation includes an unselected candidate")
    if host["selection"] is None:
        if host["selected_actor"] is not None:
            raise ValueError("Saved selected actor has no selection result")
    else:
        if not confirmations or host["final_actor"] is None:
            raise ValueError("Saved selection has no completed confirmation")
        candidates = analysis.confirmation_candidates(
            host["routine_results"], final_checkpoint_id=Path(host["final_actor"]).name
        )
        if {row["checkpoint_id"] for row in confirmations} != set(candidates):
            raise ValueError("Saved selection is missing confirmation candidates")
        chosen = analysis.select_checkpoint(confirmations)
        if (
            host["selection"] != chosen
            or host["selected_actor"] != actors[chosen["checkpoint_id"]]
        ):
            raise ValueError("Saved selection or selected actor identity differs")
