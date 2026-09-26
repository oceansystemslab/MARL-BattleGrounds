"""Own the small host files, lock and progress display for one training run.

Progress, locks and logs use the standard library only. Saved host recovery
also reads checkpoint and validation identities through their existing owners.
Saved System-panel checks may load an actor to verify its recorded identity;
they run no games. Progress uses summaries already needed by the runner and
does not restore arrays or synchronize a GPU.
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
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

if TYPE_CHECKING:
    from marl_battlegrounds.training._continuation_schedules import LearnerContinuation
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
    return (
        f"{hours}h {minutes:02d}m {secs:02d}s"
        if hours
        else f"{minutes}m {secs:02d}s"
        if minutes
        else f"{secs}s"
    )


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
    """Explain progress using existing host records, with no reads or device work.

    Parameters
    ----------
    status : dict[str, Any]
        One training run's status record, as the runner reports it or a screen
        reads it back. Every key is optional; an empty dict gives a "Starting"
        line. Keys read: env_steps and total_env_steps (real environment
        transitions); transitions_per_second, recent_transitions_per_second
        and estimated_transitions_per_second (transitions per second);
        pending_work_seconds (seconds of known later work); phase;
        completed_updates; learning_blocks; elapsed_seconds; score_threshold;
        random_validation (a saved Random check; when it has games it must also
        have wins, draws and losses); validation_wins, validation_draws and
        validation_losses, or validation_score; latest_checkpoint; and error.

    Returns
    -------
    str
        Several lines joined by newlines, with Title Case labels. The phase and
        step line and the counts and time line always appear; the other lines
        appear only when their keys are present. At most one check block
        appears, chosen in this order: a saved Random check with games, then
        validation wins, draws and losses (all three present), then
        validation_score. An unknown time shows "Estimating". The status is
        not changed.

    Notes
    -----
    Steps count real environment transitions. Training speed includes the first
    compilation and excludes validation and saving. Run elapsed time excludes
    gaps between attempts; wall_seconds includes them. Unknown future costs stay
    unknown. A QMIX run in the "warmup" phase is labelled "Warmup: Filling Replay
    Before Learning", a PQN-VDN run in the "initial_collection" phase is
    labelled "Initial Random Collection", and a status with learning_blocks
    (QMIX or PQN-VDN) reports optimizer steps and learning blocks where PPO
    reports learning updates. Raw losses and signed reward averages remain in
    training_updates.jsonl; they cannot diagnose learning by their size. A
    score-threshold curriculum line says how many points (not kills) new
    training games need to win, since a Red Zone death gives 2 points;
    validation still needs 20. Saved Random results describe combat per game as
    kills and deaths and are never treated as proof of general competence.
    Host-only and standard library only.
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
    phase = str(status.get("phase", "starting"))
    labels = {
        "starting": "Starting",
        "training": "Training",
        "speed_check": "Speed Check",
        "validation": "Checking Play Against Fixed Opponents",
        "random_validation": "Checking Play Against Random",
        "saving": "Saving Progress",
        "reporting": "Writing Results",
        "warmup": "Warmup: Filling Replay Before Learning",
        "initial_collection": "Initial Random Collection",
        "complete": "Run Finished",
        "failed": "Run Stopped With An Error",
        "recovering": "Restoring Saved Progress",
    }
    label = labels.get(phase, phase.replace("_", " ").title())
    counts = (
        f"Optimizer Steps: {status.get('completed_updates', 0):,} | "
        f"Learning Blocks: {status['learning_blocks']:,} | "
        if "learning_blocks" in status
        else f"Learning Updates: {status.get('completed_updates', 0):,} | "
    )
    lines = [
        f"{label} | Steps: {steps:,} / {total:,} ({steps / total:.1%})"
        if total
        else f"{label} | Steps: {steps:,}",
        counts + f"Time Spent In This Run: {duration(status.get('elapsed_seconds'))} | "
        f"Training Time Left (Estimate): {duration(remaining)} | "
        f"Whole Run Time Left (Estimate): {duration(eta)}",
    ]
    if status.get("score_threshold") is not None:
        lines.append(
            f"New Training Games Need {status['score_threshold']} Points To Win "
            "| Validation Still Needs 20"
        )
    if rate or recent:
        speeds: list[str] = []
        if recent:
            speeds.append(f"{recent:,.0f} steps/s recently")
        if rate:
            speeds.append(f"{rate:,.0f} steps/s overall")
        lines.append(
            "Training Speed: "
            + "; ".join(speeds)
            + " (Includes Compilation; Excludes Checks And Saving)"
        )
    check = status.get("random_validation")
    if isinstance(check, dict) and check.get("games"):
        check = cast(dict[str, Any], check)
        captured_at = (
            duration(check["wall_seconds"])
            if check.get("wall_seconds") is not None
            else "Not Recorded"
        )
        lines.append(
            f"Latest Check Against Random: {check['games']:,} Games "
            f"After {check.get('env_steps', 0):,} Training Steps; "
            f"Run Time At That Check: {captured_at}"
        )
        kills, deaths, margin = (
            check.get(name)
            for name in ("mean_kills_for", "mean_kills_against", "kill_margin")
        )
        if all(value is not None for value in (kills, deaths, margin)):
            lines.append(
                f"Per Game: Our Team Kills {kills:.2f} | "
                f"Our Team Deaths {deaths:.2f} | "
                f"Kill Difference {margin:+.2f}"
            )
            change = check.get("kill_margin_change")
            if change is not None:
                lines.append(
                    "Change In Kill Difference Since Untrained: "
                    f"{change:+.2f} Per Game "
                    "(One Training Run; More Tests Needed)"
                )
        else:
            lines.append("Kills And Deaths: Not Available In This Saved Check")
        lines.append(
            f"Game Results: {check['wins']} Wins, {check['draws']} Draws, "
            f"{check['losses']} Losses | Draws Can Still Show Combat Improvement"
        )
    elif all(
        status.get(f"validation_{name}") is not None
        for name in ("wins", "draws", "losses")
    ):
        lines.append(
            f"Latest Opponent Check: {status['validation_wins']} Wins, "
            f"{status['validation_draws']} Draws, {status['validation_losses']} Losses"
        )
    elif status.get("validation_score") is not None:
        lines.append(
            "Latest Opponent Check: Average Game Score "
            f"{status['validation_score']:.3f} "
            "(Win = 1; Draw = 0.5; Loss = 0)"
        )
    if status.get("latest_checkpoint"):
        lines.append("Recovery: Saved Progress Is Available")
    if status.get("error"):
        lines.append(f"Reason For Stopping: {status['error']}")
    return "\n".join(lines)


class ProgressReporter:
    """Print existing summaries at most once per ten seconds or a phase change.

    enabled=False returns before clock reads or formatting. stream defaults to
    stdout. The caller supplies existing host data; this helper starts no thread,
    file writer, model call, measurement or device transfer. Screen workers use
    MARL_BGS_PROGRESS_PHASES_ONLY=1 to leave regular updates to their supervisor;
    MARL_BGS_PROGRESS_CONTEXT labels those workers as speed checks or trials.
    """

    def __init__(self, *, enabled: bool, stream: TextIO | None = None) -> None:
        """Set output options and start with no previous displayed phase."""
        self.enabled = enabled
        self.stream = sys.stdout if stream is None else stream
        self.last_time = float("-inf")
        self.last_phase: str | None = None
        self.phases_only = os.environ.get("MARL_BGS_PROGRESS_PHASES_ONLY") == "1"
        self.context = os.environ.get("MARL_BGS_PROGRESS_CONTEXT", "")

    def report(self, status: dict[str, Any], *, force: bool = False) -> None:
        """Print status when due; force also prints completion or failure."""
        if not self.enabled:
            return
        phase = str(status.get("phase"))
        if self.phases_only and not force and phase == self.last_phase:
            return
        now = time.monotonic()
        if force or phase != self.last_phase or now - self.last_time >= 10:
            stamp = utc_now()[:19].replace("T", " ")
            print(
                f"[{stamp} UTC] {self.context}\n{progress_text(status)}",
                file=self.stream,
                flush=True,
            )
            self.last_time, self.last_phase = now, phase


def checkpoint_ancestry(
    run_dir: Path, checkpoint_details: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Read one active chain of signed learner descriptions without model arrays.

    run_dir is the original run directory. checkpoint_details is its already
    verified latest or selected learner description, read through the checkpoint
    owner. The caller must bind that first description to the intended run.
    Return an ID-to-description mapping including the current boundary. Verify
    each parent belongs to the same run/config/source/dependencies/execution and
    does not follow its child's experience counters. Reject cycles, linked
    paths, malformed IDs or inconsistent ancestry with ValueError; missing files
    raise OSError. Pruned numerical payloads are not needed. No files change and
    no backend or numerical restore starts.
    """
    from marl_battlegrounds.training import checkpoints

    root = run_dir.resolve()
    metadata = checkpoint_details["metadata"]

    def identifier(item: object) -> str:
        """Read a lowercase SHA-256 through the existing checkpoint owner."""
        return checkpoints._digest(item, "Saved artifact identity")  # pyright: ignore[reportPrivateUsage]

    def integer(item: object, name: str) -> int:
        """Require one nonnegative ancestry counter without accepting bool."""
        if type(item) is not int or item < 0:
            raise ValueError(f"Saved {name} must be a nonnegative integer")
        return item

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

    return ancestry


def _check_validation_interval(result: Mapping[str, Any]) -> None:
    """Check finite scores and explicitly available or unsupported saved intervals.

    Historical records keep their numerical interval requirement. New sampling
    metadata may withhold public bounds only when the shared summary owner
    produces that same availability decision. Conditional bounds remain finite
    descriptive numbers and never stand in for a supported interval.
    """
    from marl_battlegrounds.training.analysis import (
        _validation_sampling_interval,  # pyright: ignore[reportPrivateUsage]
    )

    evidence = result.get("sampling_evidence")
    fields = ["score"]
    unavailable = result.get("ci_low") is None or result.get("ci_high") is None
    if evidence is None and unavailable:
        raise ValueError("Saved validation interval needs sampling evidence")
    fields.extend(
        ("conditional_ci_low", "conditional_ci_high")
        if unavailable
        else ("ci_low", "ci_high")
    )
    for field in fields:
        number = result.get(field)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or not 0 <= number <= 1
        ):
            raise ValueError("Saved validation score or interval is invalid")
    if result[fields[1]] > result[fields[2]]:
        raise ValueError("Saved validation interval is reversed")
    if evidence is not None:
        if not isinstance(evidence, Mapping):
            raise ValueError("Saved validation sampling evidence must be an object")
        base = dict(result)
        base.update(ci_low=result[fields[1]], ci_high=result[fields[2]])
        pairs = result.get("seed_pairs")
        if type(pairs) is not int or pairs < 1:
            raise ValueError("Saved validation seed_pairs must be positive")
        checked = _validation_sampling_interval(
            base,
            cast(Mapping[str, Any], evidence),
            pairs,
        )
        # Earlier closeout records used independent_blocks for the supported
        # count. Keep reading those original bytes; new records separate counts.
        historical_counts = "supported_independent_sampling_units" not in result
        if historical_counts:
            checked["independent_blocks"] = checked[
                "supported_independent_sampling_units"
            ]
        required = [
            "ci_low",
            "ci_high",
            "interval_status",
            "confidence",
            "sampling_unit",
            "interval_method",
            "independent_blocks",
            "declared_blocks",
        ]
        if not historical_counts:
            required.append("supported_independent_sampling_units")
        if unavailable:
            required.append("conditional_interval_assumption")
        if any(result.get(field) != checked[field] for field in required):
            raise ValueError(
                "Saved validation interval disagrees with sampling evidence"
            )


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
        actors, panel/Random results and selection agree. Shared initialization
        evidence is verified at its original paths. Inputs remain unchanged.

    Raises
    ------
    ValueError, OSError
        A field is missing, malformed, inconsistent or points outside this run;
        an actor or completed validation artifact is missing or changed; or a
        candidate belongs to an abandoned continuation. A pending current export
        need not exist. Ancestor descriptions remain usable after payload pruning.
        Exported weights, input scale, spawn frame and complete model schema
        must match their learner boundary and the saved training method. For
        a QMIX or PQN-VDN run every saved validation and Random result must
        name its method (``"qmix"`` or ``"pqn_vdn"``) and an integer
        ``optimizer_steps`` equal to its learner boundary's and export's
        count; PPO results carry neither.

    Notes
    -----
    Read-only and host-only. The saved method selects the host counts. PPO
    keeps used_policy_samples and used_value_samples. QMIX keeps
    completed_blocks, learning_blocks, sampled_sequences, used_td_pairs,
    used_agent_utilities and sampled_exposure (by_stage, by_source and
    by_opponent lists, each summing to used_td_pairs), all exact Python
    integers that may exceed int32, checked with the runner's fixed-block
    arithmetic: blocks of rollout_length rounds, the last one shorter, each
    learning once the stored rows reach min_buffer_size. PQN-VDN keeps
    completed_blocks, learning_blocks, used_sequences, used_td_pairs,
    used_agent_utilities, used_prefix_td_pairs and used_exposure (the same
    three marginals), checked against its offset boundaries: W = H + T
    initial rounds, then blocks of T (``_check_pqn_host_counts``); its
    validation points come from the same offset schedule. Existing checkpoint
    and selection owners verify artifact identities and selection rules.
    Saved validation tasks and Random records are rebuilt at the saved
    config's red_zone_depth (checkpoints._config_red_zone_depth), so they must
    carry that depth. System-panel records may require loading an actor to verify its
    registration; learner arrays are not restored here. No games run and no
    writer or log is changed. Child runs use their frozen future points and
    effective roots. Their local actor map stays local; explicit checked ancestor
    references may supply a selected actor or an actor for child confirmation.
    Parent games do not count as new child work. Call this after complete learner
    restore and before resume_recording.
    """
    from marl_battlegrounds.baselines.methods import method_settings_field
    from marl_battlegrounds.training import analysis, checkpoints, validation

    root = run_dir.resolve()
    metadata = checkpoint_details["metadata"]
    config = metadata["config"]
    method = config.get("method", "mappo")
    continuation = metadata.get("continuation")
    declaration = validation.saved_validation_declaration(metadata, panel)
    from marl_battlegrounds.training._continuation_schedules import (
        continuation_counts,
        learner_continuation,
    )

    learner_context = learner_continuation(
        None if continuation is None else continuation.get("learner")
    )
    inherited: dict[str, Any] = {"records": [], "actors": {}, "used_roots": []}
    if declaration is not None:
        from marl_battlegrounds.training._selection_evidence import (
            read_inherited_candidates,
        )

        inherited = read_inherited_candidates(
            continuation.get("inherited_candidates", []) if continuation else [],
            declaration=declaration,
        )
    qmix_run = method == "qmix"
    pqn_run = method == "pqn_vdn"
    settings = config[method_settings_field(method)]
    # The run's Red Zone depth sets the layout of its saved validation tasks.
    depth = checkpoints._config_red_zone_depth(config)  # pyright: ignore[reportPrivateUsage]
    schemas = checkpoints.checkpoint_schemas(method)
    if checkpoint_details["schemas"] != schemas:
        raise ValueError("Saved model schema differs from the training method")
    value: object = metadata.get("host_state")
    if not isinstance(value, dict):
        raise ValueError("Saved host state must be a JSON object")
    host = cast(dict[str, Any], value)
    samples = (
        (
            "completed_blocks",
            "learning_blocks",
            "sampled_sequences",
            "used_td_pairs",
            "used_agent_utilities",
        )
        if qmix_run
        else (
            "completed_blocks",
            "learning_blocks",
            "used_sequences",
            "used_td_pairs",
            "used_agent_utilities",
            "used_prefix_td_pairs",
        )
        if pqn_run
        else ("used_policy_samples", "used_value_samples")
    )
    counts = (
        "env_steps",
        "completed_updates",
        "actor_decisions",
        *samples,
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
    if qmix_run:
        required.add("sampled_exposure")
    if pqn_run:
        required.add("used_exposure")
    random_pairs = config.get("random_diagnostic_seed_pairs")
    if random_pairs is not None:
        required.add("random_results")
    stage_shapes = {
        "stage_completed": (17, 3),
        "stage_score_sums": (17, 2),
        "stage_length_sum": (17,),
        "stage_k20_count": (17,),
    }
    if not required <= set(host) or set(host) - required - stage_shapes.keys() - {
        "random_results"
    }:
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
    epochs, length = settings["epochs"], settings["rollout_length"]
    if qmix_run:
        _check_qmix_host_counts(
            host, checkpoint_details["counters"], config, continuation=learner_context
        )
    elif pqn_run:
        _check_pqn_host_counts(
            host, checkpoint_details["counters"], config, continuation=learner_context
        )
    elif (
        {"env_steps": steps, "updates": updates} != checkpoint_details["counters"]
        or steps > total
        or steps % batch
        or updates
        != (
            math.ceil(steps / (batch * length))
            if learner_context is None
            else continuation_counts(
                steps // batch, continuation=learner_context, rollout_length=length
            )[0]
        )
        or host["actor_decisions"] > steps * 5
        or host["used_policy_samples"] != host["actor_decisions"] * epochs
        or not steps * epochs <= host["used_value_samples"] <= steps * 5 * epochs
        or host["used_policy_samples"] > host["used_value_samples"]
    ):
        raise ValueError("Saved host experience or sample counts disagree")
    present_stages = set(host) & stage_shapes.keys()
    collected = host["completed_blocks"] if qmix_run or pqn_run else updates
    if (present_stages and present_stages != stage_shapes.keys()) or (
        collected and not present_stages
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

    ancestry = checkpoint_ancestry(root, checkpoint_details)
    if any(ancestor["schemas"] != schemas for ancestor in ancestry.values()):
        raise ValueError("Saved ancestor model differs from the training method")

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
            or actor["weight_digest"] != ancestor["actor_digest"]
            or actor["input_scale"] != settings.get("input_scale", 1.0)
            or actor["spawn_frame"] != settings.get("spawn_frame", "world")
            or actor["schemas"] != ancestor["schemas"]
            or actor["schemas"] != schemas
            or (
                (qmix_run or pqn_run)
                and actor.get("optimizer_steps") != ancestor["counters"]["updates"]
            )
        ):
            raise ValueError("Saved actor identity differs from its learner boundary")
        actor_records[key] = actor

    points: set[int] = set()
    if declaration is not None:
        points = {point["env_steps"] for point in declaration["points"]}
    elif panel is not None:
        points = {
            point.env_steps
            for point in validation.resolve_validation_schedule(
                total,
                batch,
                rollout_length=length,
                fractions=config["validation_fractions"],
                initial_rounds=(settings["memory_window"] + length if pqn_run else 0),
            )
        }
    selection_actors: dict[str, str] = {
        **{key: item["actor_path"] for key, item in inherited["actors"].items()},
        **cast(dict[str, str], actors),
    }
    random_points: set[int] = (
        {0, total, *config["checkpoint_env_steps"]}
        if random_pairs is not None
        else set()
    )
    if continuation is not None:
        random_points = {
            point for point in random_points if point > continuation["start_env_steps"]
        }
    initial_reference = config.get("random_initialization_result")
    if initial_reference is not None and continuation is None:
        initial = next(
            (item for item in ancestry.values() if item["counters"]["env_steps"] == 0),
            None,
        )
        if initial is None or random_pairs is None:
            raise ValueError(
                "Shared Random initialization has no initial learner boundary"
            )
        validation.read_random_initialization(
            initial_reference,
            actor_digest=checkpoints._inference_digest(initial),  # pyright: ignore[reportPrivateUsage]
            seed_pairs=random_pairs,
            red_zone_depth=depth,
        )
    if initial_reference is not None and continuation is not None:
        parent_context = continuation
        visited: set[str] = set()
        while True:
            parent_path = Path(parent_context["parent_checkpoint"])
            if str(parent_path) in visited or parent_path.resolve() != parent_path:
                raise ValueError("Random initialization ancestry is cyclic or linked")
            visited.add(str(parent_path))
            parent_details = checkpoints.read_checkpoint_description(parent_path)
            if parent_details["checkpoint_id"] != parent_path.name:
                raise ValueError("Random initialization ancestor identity differs")
            parent_chain = checkpoint_ancestry(
                parent_path.parent.parent, parent_details
            )
            initial = next(
                (
                    item
                    for item in parent_chain.values()
                    if item["counters"]["env_steps"] == 0
                ),
                None,
            )
            if initial is not None:
                if random_pairs is None:
                    raise ValueError(
                        "Shared Random initialization has no declared count"
                    )
                validation.read_random_initialization(
                    initial_reference,
                    actor_digest=checkpoints._inference_digest(initial),  # pyright: ignore[reportPrivateUsage]
                    seed_pairs=random_pairs,
                    red_zone_depth=depth,
                )
                break
            parent_context = parent_details["metadata"].get("continuation")
            if parent_context is None:
                raise ValueError(
                    "Shared Random initialization has no original zero boundary"
                )
    pending = host["pending"]
    if pending is not None:
        allowed_flags = (
            ({"routine", "random"},)
            if random_pairs is not None
            else ({"routine", "random"}, {"routine"})
        )
        if (
            not isinstance(pending, dict)
            or set(cast(dict[str, Any], pending)) not in allowed_flags
            or type(cast(dict[str, Any], pending)["routine"]) is not bool
            or (
                "random" in pending
                and type(cast(dict[str, Any], pending)["random"]) is not bool
            )
        ):
            raise ValueError(
                "Saved pending work must declare Boolean routine/Random flags"
            )
        if pending["routine"] != (steps in points):
            raise ValueError("Saved pending validation differs from its schedule")
        if pending.get("random", False) != (steps in random_points):
            raise ValueError(
                "Saved pending Random diagnostic differs from its schedule"
            )
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
            if panel is None or (
                key not in actor_records
                and not (purpose == "confirmation" and key in inherited["actors"])
            ):
                raise ValueError("Saved validation refers to an unknown actor")
            actor = actor_records.get(key, inherited["actors"].get(key))
            assert actor is not None
            _check_result_method(result, actor, method)
            pairs = config[f"{purpose}_seed_pairs"]
            expected_task = validation.panel_task_description(
                checkpoint_id=key,
                actor_digest=actor["actor_digest"],
                env_steps=actor["env_steps"],
                panel=panel,
                purpose=purpose,
                seed_pairs=pairs,
                red_zone_depth=depth,
            )
            if declaration is not None:
                expected_task = validation.declared_panel_task(
                    declaration,
                    panel,
                    checkpoint_id=key,
                    actor_digest=actor["actor_digest"],
                    env_steps=actor["env_steps"],
                    purpose=purpose,
                )
                pairs = expected_task["seed_pairs"]
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
            _check_validation_interval(result)
            games = 2 * len(validation.VALIDATION_MAPS) * len(panel.members) * pairs
            if integer(result.get("games"), "validation games") != games:
                raise ValueError("Saved validation game count differs")
            paths = result.get("pass_paths")
            if not isinstance(paths, list) or len(cast(list[Any], paths)) != len(
                panel.members
            ):
                raise ValueError("Saved validation pass paths are incomplete")
            loaded_actor = (
                checkpoints.load_system(selection_actors[key])
                if panel.schema_version == 2
                else None
            )
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
                if panel.schema_version == 2:
                    validation._verify_panel_pass(  # pyright: ignore[reportPrivateUsage]
                        actual_path,
                        loaded_actor,
                        panel.methods[index],
                        task=expected_task,
                        member_index=index,
                        num_envs=min(32, batch),
                    )
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
    random_steps: set[int] = set()
    results = host.get("random_results", [])
    if not isinstance(results, list) or (random_pairs is None and results):
        raise ValueError("Saved Random results differ from their enabled settings")
    for result in cast(list[Any], results):
        if not isinstance(result, dict):
            raise ValueError("Saved Random result must be an object")
        result = cast(dict[str, Any], result)
        point = integer(result.get("env_steps"), "Random env_steps")
        if point not in random_points or point in random_steps or point > steps:
            raise ValueError("Saved Random captures differ from their exact schedule")
        random_steps.add(point)
        current_actor = next(
            (actor for actor in actor_records.values() if actor["env_steps"] == point),
            None,
        )
        if current_actor is None:
            raise ValueError("Saved Random result has no current-run actor capture")
        _check_result_method(result, current_actor, method)
        reused = point == 0 and initial_reference is not None
        if result.get("reused_initialization") is not reused or result.get(
            "reference_path"
        ) != (str(Path(cast(str, initial_reference)).absolute()) if reused else None):
            raise ValueError(
                "Saved Random initialization reference differs from config"
            )
        key = current_actor["metadata"]["checkpoint_id"]
        if not reused and (
            result.get("actor_path") != actors[key]
            or result.get("summary_path")
            != str(root / "validation" / f"random-{key}" / "validation_summary.json")
        ):
            raise ValueError("Saved Random evidence is outside its current-run capture")
        validation.verify_random_result(
            result,
            actor_digest=current_actor["actor_digest"],
            seed_pairs=cast(int, random_pairs),
            checkpoint_id=None if reused else key,
            env_steps=point,
            red_zone_depth=depth,
        )
        if result["task_id"] in seen:
            raise ValueError("Saved Random task is duplicated")
        seen.add(result["task_id"])
        if (
            seconds(result["elapsed_seconds"], "Random elapsed_seconds")
            > host["elapsed_seconds"] + 1e-6
            or seconds(result["training_seconds"], "Random training_seconds")
            > host["training_seconds"] + 1e-6
        ):
            raise ValueError("Saved Random capture follows its checkpoint time")
        if not reused:
            validation_games += result["games"]
    required_random = {point for point in random_points if point <= steps}
    if pending is not None and pending.get("random"):
        required_random.discard(steps)
    if not required_random <= random_steps:
        raise ValueError("Saved Random diagnostic coverage is incomplete")
    required_points = {point for point in points if point <= steps}
    if pending is not None and pending["routine"]:
        required_points.discard(steps)
    if (
        not required_points <= routine_steps
        or host["validation_games"] != validation_games
    ):
        raise ValueError("Saved completed validation coverage is incomplete")
    confirmations = host["confirmation_results"]
    routine_results = host["routine_results"]
    if (
        declaration is not None
        and panel is not None
        and host["final_actor"] is not None
    ):
        routine_results, confirmations, _ = validation.selection_validation_results(
            routine_results,
            confirmations,
            inherited,
            declaration=declaration,
            panel=panel,
            final_checkpoint_id=Path(host["final_actor"]).name,
        )
    if confirmations:
        if host["final_actor"] is None:
            raise ValueError("Saved confirmation has no final checkpoint")
        candidates = analysis.confirmation_candidates(
            routine_results, final_checkpoint_id=Path(host["final_actor"]).name
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
            routine_results, final_checkpoint_id=Path(host["final_actor"]).name
        )
        if {row["checkpoint_id"] for row in confirmations} != set(candidates):
            raise ValueError("Saved selection is missing confirmation candidates")
        chosen = analysis.select_checkpoint(confirmations)
        if declaration is not None:
            chosen.update(
                routine_comparison=validation.validation_root_comparison(
                    routine_results,
                    allow_different_roots=declaration["allow_different_roots"],
                ),
                confirmation_comparison=validation.validation_root_comparison(
                    confirmations,
                    allow_different_roots=declaration["allow_different_roots"],
                ),
            )
        if (
            host["selection"] != chosen
            or host["selected_actor"] != selection_actors[chosen["checkpoint_id"]]
        ):
            raise ValueError("Saved selection or selected actor identity differs")


def qmix_fixed_block_counts(
    rounds: int, *, rollout_length: int, minimum: int
) -> tuple[int, int]:
    """Count a QMIX run's blocks and learning blocks for fixed-length blocks.

    Parameters
    ----------
    rounds : int
        Real rounds collected per game (env steps divided by games).
    rollout_length : int
        Rounds in each block; the last block may be shorter.
    minimum : int
        min_buffer_size: a block learns once the rows stored per game after its
        insertion reach it.

    Returns
    -------
    tuple[int, int]
        (completed blocks, learning blocks) as exact Python integers.

    Examples
    --------
    >>> qmix_fixed_block_counts(20, rollout_length=4, minimum=6)
    (5, 4)
    """
    blocks = -(-rounds // rollout_length)
    first = max(1, -(-minimum // rollout_length))
    if rounds < minimum:
        return blocks, 0
    return blocks, blocks - first + 1


def _check_qmix_host_counts(
    host: dict[str, Any],
    counters: dict[str, Any],
    config: dict[str, Any],
    *,
    continuation: LearnerContinuation | None = None,
) -> None:
    """Check a QMIX run's saved host counts with fixed-block arithmetic.

    host holds the runner's exact Python-integer totals, counters the learner
    checkpoint's saved counters and config the saved run config. Blocks are
    rollout_length rounds, the last one shorter; a block learns once the rows
    stored per game reach min_buffer_size. Every ready block takes epochs
    optimizer steps of sample_batch_size sequences with S-1 TD pairs each.
    continuation is the checked child boundary, or None for the original
    zero-based block schedule. A child retains cumulative counters across partial
    parent blocks. Raises ValueError on disagreement; reads and changes no files.
    """
    settings = config["qmix"]
    batch, total = config["num_envs"], config["total_env_steps"]
    steps, updates = host["env_steps"], host["completed_updates"]
    length, minimum = settings["rollout_length"], settings["min_buffer_size"]
    blocks, learning = qmix_fixed_block_counts(
        steps // batch, rollout_length=length, minimum=minimum
    )
    if continuation is not None:
        from marl_battlegrounds.training._continuation_schedules import (
            continuation_counts,
        )

        blocks, learning = continuation_counts(
            steps // batch,
            continuation=continuation,
            rollout_length=length,
            minimum=minimum,
        )
    pairs = host["used_td_pairs"]
    exposure = host["sampled_exposure"]
    marginals = ("by_stage", "by_source", "by_opponent")
    exposure_ok = (
        isinstance(exposure, dict)
        and set(cast(dict[str, Any], exposure)) == set(marginals)
        and all(
            isinstance(exposure[name], list)
            and len(cast(list[Any], exposure[name])) >= 1
            and all(
                type(cell) is int and cell >= 0
                for cell in cast(list[Any], exposure[name])
            )
            and sum(cast(list[int], exposure[name])) == pairs
            for name in marginals
        )
        and len(cast(list[Any], exposure["by_stage"])) == 17
        and len(cast(list[Any], exposure["by_opponent"])) == 21
    )
    if (
        {
            "env_steps": steps,
            "updates": updates,
            "completed_blocks": host["completed_blocks"],
            "learning_blocks": host["learning_blocks"],
        }
        != counters
        or steps > total
        or steps % batch
        or host["completed_blocks"] != blocks
        or host["learning_blocks"] != learning
        or updates != learning * settings["epochs"]
        or host["actor_decisions"] > steps * 5
        or host["sampled_sequences"] != updates * settings["sample_batch_size"]
        or pairs != host["sampled_sequences"] * (settings["sample_sequence_length"] - 1)
        or not pairs <= host["used_agent_utilities"] <= 5 * pairs
        or not exposure_ok
    ):
        raise ValueError("Saved host experience or sample counts disagree")


def _check_pqn_host_counts(
    host: dict[str, Any],
    counters: dict[str, Any],
    config: dict[str, Any],
    *,
    continuation: LearnerContinuation | None = None,
) -> None:
    """Check a PQN-VDN run's saved host counts with offset-block arithmetic.

    Parameters
    ----------
    host : dict
        The runner's exact Python-integer totals.
    counters : dict
        The learner checkpoint's saved counters.
    config : dict
        The saved run config, whose "pqn" block gives T, H, epochs and
        minibatches.
    continuation : LearnerContinuation or None, default=None
        Checked child block origin and cumulative counts. None uses the original
        warmup-offset schedule. A child never repeats completed warmup blocks.

    Raises
    ------
    ValueError
        Any count disagrees. With r rounds and k learned blocks: r is a
        reachable boundary (``validation._collection_boundary`` with W = H + T
        initial rounds), completed_blocks is its ordinal, k is that ordinal
        less ceil(W / T) (never below 0), updates is k * E * M, used_sequences
        is k * E * B, used_td_pairs is E * B * ((r - W) + k * (H - 1)) (0 before
        learning), used_prefix_td_pairs is k * E * B * H, utilities lie
        between 1 and 5 per pair, and each used_exposure marginal (17 stages,
        the source bank, 21 opponent rows) sums to used_td_pairs.

    Notes
    -----
    Host-only arithmetic on exact integers that may exceed int32; reads and
    changes nothing else.
    """
    from marl_battlegrounds.training.validation import (
        _collection_boundary,  # pyright: ignore[reportPrivateUsage]
    )

    settings = config["pqn"]
    batch, total = config["num_envs"], config["total_env_steps"]
    steps, updates = host["env_steps"], host["completed_updates"]
    length, window = settings["rollout_length"], settings["memory_window"]
    epochs, groups = settings["epochs"], settings["num_minibatches"]
    initial = window + length
    if steps > total or steps % batch or total % batch:
        raise ValueError("Saved host experience or sample counts disagree")
    rounds = steps // batch
    if continuation is None:
        reachable, blocks = _collection_boundary(
            rounds,
            total_env_steps=total // batch,
            num_envs=1,
            rollout_length=length,
            initial_rounds=initial,
        )
        learning = max(0, blocks - -(-initial // length))
    else:
        from marl_battlegrounds.training._continuation_schedules import (
            continuation_boundary,
        )

        reachable, blocks, learning = continuation_boundary(
            rounds,
            total_rounds=total // batch,
            continuation=continuation,
            rollout_length=length,
            initial_rounds=initial,
        )
    pairs = host["used_td_pairs"]
    expected_pairs = (
        epochs * batch * ((rounds - initial) + learning * (window - 1))
        if learning
        else 0
    )
    exposure = host["used_exposure"]
    marginals = ("by_stage", "by_source", "by_opponent")
    exposure_ok = (
        isinstance(exposure, dict)
        and set(cast(dict[str, Any], exposure)) == set(marginals)
        and all(
            isinstance(exposure[name], list)
            and len(cast(list[Any], exposure[name])) >= 1
            and all(
                type(cell) is int and cell >= 0
                for cell in cast(list[Any], exposure[name])
            )
            and sum(cast(list[int], exposure[name])) == pairs
            for name in marginals
        )
        and len(cast(list[Any], exposure["by_stage"])) == 17
        and len(cast(list[Any], exposure["by_opponent"])) == 21
    )
    if (
        {
            "env_steps": steps,
            "updates": updates,
            "completed_blocks": host["completed_blocks"],
            "learning_blocks": host["learning_blocks"],
        }
        != counters
        or reachable != rounds
        or host["completed_blocks"] != blocks
        or host["learning_blocks"] != learning
        or updates != learning * epochs * groups
        or host["actor_decisions"] > steps * 5
        or host["used_sequences"] != learning * epochs * batch
        or pairs != expected_pairs
        or host["used_prefix_td_pairs"] != learning * epochs * batch * window
        or not pairs <= host["used_agent_utilities"] <= 5 * pairs
        or not exposure_ok
    ):
        raise ValueError("Saved host experience or sample counts disagree")


def _check_result_method(
    result: dict[str, Any], actor: dict[str, Any], method: str
) -> None:
    """Match a saved validation result's method fields to its actor.

    A QMIX or PQN-VDN run's results must name that method ("qmix" or
    "pqn_vdn") and carry the actor's integer optimizer_steps (from
    artifact_identity); a PPO run's results must carry neither field.

    Parameters
    ----------
    result : dict
        One saved routine, confirmation or Random result.
    actor : dict
        The artifact identity of the actor the result evaluated.
    method : str
        The run's training method name, such as "mappo", "qmix" or "pqn_vdn".

    Raises
    ------
    ValueError
        A QMIX or PQN-VDN result names another method, or its optimizer_steps
        is not the actor's integer count; or a PPO result carries method or
        optimizer_steps.
    """
    if method in ("qmix", "pqn_vdn"):
        label = "QMIX" if method == "qmix" else "PQN-VDN"
        if result.get("method") != method:
            raise ValueError(f"Saved {label} result names another method")
        steps = result.get("optimizer_steps")
        if type(steps) is not int or steps != actor.get("optimizer_steps"):
            raise ValueError(f"Saved {label} result differs from its optimizer count")
    elif "method" in result or "optimizer_steps" in result:
        raise ValueError("Saved PPO result carries QMIX or PQN-VDN fields")
