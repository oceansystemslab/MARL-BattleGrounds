"""Run and resume one declared recurrent MAPPO experiment.

TrainConfig owns validated host settings. train joins existing collection,
learner, checkpoint and evaluation authorities; numerical work stays outside
host logging and file handling. CLI commands use these same public functions.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.training._run_io import (
    ProgressReporter,
    TrainingSpeedEstimate,
    append_jsonl,
    atomic_json,
    log_cursor,
    preserve_log_suffix,
    process_identity,
    restore_log_cursor,
    run_lock,
    utc_now,
    validate_host_state,
    validate_log_cursor,
)
from marl_battlegrounds.training.opponents import (
    _pinned_share,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.training.shaping import validate_shaping

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.run_writer import RunWriter
    from marl_battlegrounds.training.collection import (
        TrainingCarry,
        TrainingCollection,
        TrainingRollout,
    )
    from marl_battlegrounds.training.learner import LearnerState, UpdateResult
    from marl_battlegrounds.training.validation import FrozenPanel

    type _Updater = Callable[
        [LearnerState, TrainingCarry, TrainingRollout],
        tuple[LearnerState, UpdateResult],
    ]


@dataclass(frozen=True)
class TrainConfig:
    """Declare one experiment without opening files or starting a backend.

    Parameters
    ----------
    seed : int, default=42
        Root seed for model and independent collection/shuffle streams.
    method : {"mappo"}, default="mappo"
        Recurrent MAPPO is the only implemented learner.
    num_envs : int, default=32
        Fixed even environment batch, divisible by PPO groups times minibatches.
    total_env_steps : int, default=10000000
        Exact real environment transitions, divisible by num_envs. Resets and
        padding do not count. Per-lane rounds must fit a signed 32-bit integer.
    curriculum, shaping : bool, default=False
        Enable the existing 17-stage schedule and chosen team reward adjustment.
    score_threshold_curriculum : bool, default=False
        Use K1..10, K12, K15 and K20 at future resets, with 10%, nine shares
        of 1/30, 5%, 5% and 50% of requested experience. Keeps canonical 5v5
        and all training maps. Cannot be combined with curriculum=True.
        Evaluation stays K20/H300. Actual played shares can lag stage changes.
    shaping_coefficient : float, default=0.01
        Finite nonnegative shaping weight. Task reward remains separately logged.
    shaping_mode : {"potential", "score_delta"}, default="potential"
        Potential preserves the discounted task objective. Score_delta adds
        reward for team kills minus deaths without terminal cancellation; it
        deliberately changes the training objective. Used only when shaping is
        True. Evaluation always uses native task rewards and win rules.
    pinned_opponent_share : float, default=0.0
        Probability, within [0, 0.8], that each new training game meets the
        pinned first-update actor instead of drawing from the ordinary self-play
        recipe. Zero keeps 80% current weights and 20% uniform history with
        today's exact random draws. A positive share also moves the first
        history snapshot to the first completed update and drops the never-played
        100% snapshot, so slot 0 holds that near-untrained actor for the whole
        run; the remaining probability is 20% other history when any exists and
        current weights otherwise. Evaluation opponents are unaffected.
    ppo : PPOConfig, default=PPOConfig()
        Immutable donor network update settings, including rollout length,
        input scale and spawn frame. A shared random initialization result can
        be reused only by a run with the same scale and frame.
    metrics : {"priority", "none"}, default="priority"
        Existing episode metric level. No replay recording is implied.
    recording : bool, default=False
        Save episode tables through a separate training RunWriter. False skips
        recording drains; learner checkpoints and update summaries still save.
    validation_panel : str or None, default=None
        Frozen panel.json path. A demonstration requires it. Development may
        omit it, in which case no selected actor is claimed.
    validation_fractions : tuple[float, ...], default=(0.1, ..., 1.0)
        Increasing experience fractions, rounded up to completed updates.
        Initialization is diagnostic only. The final fraction must be one.
    routine_seed_pairs, confirmation_seed_pairs : int, default=10, 50
        Independent seed pairs per validation map and opponent.
    checkpoint_interval_updates : int, default=25
        Recovery-save interval. Initialization, validation and final always save.
    checkpoint_env_steps : tuple[int, ...], default=()
        Additional exact actor capture points at completed update boundaries.
    random_diagnostic_seed_pairs : int or None, default=None
        Enable fixed Random diagnostics at initialization, checkpoint_env_steps
        and the final budget. The positive count is paired seeds per each of
        five validation maps; four gives 40 games. No panel is required and no
        actor is selected from these results. None skips this optional work.
    random_initialization_result : str or None, default=None
        Original runner result copied to a JSON file for initialization reuse.
        Requires Random diagnostics. Its actor inference, task and complete M8
        evidence must match before output or recovery changes. Original evidence
        paths must remain available; later captures still run their own games.
    slot_diagnostic : bool, default=False
        Run the declared diagnostic after final training. Requires a frozen panel.
    purpose : {"development", "demonstration"}, default="development"
        Demonstration requires frozen validation and final selection.
    verbose : bool, default=True
        Print existing host summaries at most every ten seconds and phase changes.
        Adds no numerical calls or device transfers. Full-run use is qualified by
        the separate measured reporting-cost check. False skips display and extra
        ETA calculation while retaining required counters, timings and status.

    Raises
    ------
    ValueError, TypeError
        A count, option or numerical setting violates this contract. Booleans
        are not accepted as integer counts. No files are changed on rejection.
    """

    seed: int = 42
    method: Literal["mappo"] = "mappo"
    num_envs: int = 32
    total_env_steps: int = 10_000_000
    curriculum: bool = False
    score_threshold_curriculum: bool = False
    shaping: bool = False
    shaping_coefficient: float = 0.01
    shaping_mode: Literal["potential", "score_delta"] = "potential"
    pinned_opponent_share: float = 0.0
    ppo: PPOConfig = field(default_factory=PPOConfig)
    metrics: Literal["priority", "none"] = "priority"
    recording: bool = False
    validation_panel: str | None = None
    validation_fractions: tuple[float, ...] = tuple(i / 10 for i in range(1, 11))
    routine_seed_pairs: int = 10
    confirmation_seed_pairs: int = 50
    checkpoint_interval_updates: int = 25
    checkpoint_env_steps: tuple[int, ...] = ()
    random_diagnostic_seed_pairs: int | None = None
    random_initialization_result: str | None = None
    slot_diagnostic: bool = False
    purpose: Literal["development", "demonstration"] = "development"
    verbose: bool = True

    def __post_init__(self) -> None:
        """Reject invalid host settings without allocating numerical arrays."""
        if type(self.seed) is not int:
            raise TypeError("seed must be a Python integer, not bool")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        for name in (
            "num_envs",
            "total_env_steps",
            "routine_seed_pairs",
            "confirmation_seed_pairs",
            "checkpoint_interval_updates",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive Python integer")
        for name in (
            "curriculum",
            "score_threshold_curriculum",
            "shaping",
            "recording",
            "slot_diagnostic",
            "verbose",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.method != "mappo":
            raise ValueError("Only recurrent mappo is implemented")
        if self.curriculum and self.score_threshold_curriculum:
            raise ValueError("Choose team/map curriculum or score-threshold curriculum")
        if not isinstance(cast(object, self.ppo), PPOConfig):
            raise TypeError("ppo must be PPOConfig")
        if self.num_envs % 2 or self.num_envs % (
            self.ppo.groups * self.ppo.minibatches
        ):
            raise ValueError(
                "num_envs must be even and divide by PPO groups * minibatches"
            )
        if self.total_env_steps % self.num_envs:
            raise ValueError("total_env_steps must be divisible by num_envs")
        if self.total_env_steps // self.num_envs > 2**31 - 1:
            raise ValueError("per-lane rounds must fit int32")
        if self.metrics not in ("priority", "none"):
            raise ValueError("training metrics must be priority or none")
        if (
            isinstance(self.shaping_coefficient, bool)
            or not math.isfinite(self.shaping_coefficient)
            or self.shaping_coefficient < 0
        ):
            raise ValueError("shaping_coefficient must be finite and nonnegative")
        validate_shaping(
            discount=self.ppo.gamma,
            coefficient=self.shaping_coefficient,
            mode=self.shaping_mode,
        )
        _pinned_share(self.pinned_opponent_share)
        if self.purpose not in ("development", "demonstration"):
            raise ValueError("purpose must be development or demonstration")
        if self.validation_panel is not None and (
            not isinstance(cast(object, self.validation_panel), str)
            or not self.validation_panel
        ):
            raise ValueError("validation_panel must be a nonempty path string or None")
        if self.random_diagnostic_seed_pairs is not None and (
            type(self.random_diagnostic_seed_pairs) is not int
            or self.random_diagnostic_seed_pairs <= 0
        ):
            raise ValueError(
                "random_diagnostic_seed_pairs must be a positive integer or None"
            )
        if self.random_initialization_result is not None and (
            not isinstance(cast(object, self.random_initialization_result), str)
            or not self.random_initialization_result
            or self.random_diagnostic_seed_pairs is None
        ):
            raise ValueError(
                "random_initialization_result needs a path and enabled "
                "Random diagnostics"
            )
        if (
            self.purpose == "demonstration" or self.slot_diagnostic
        ) and not self.validation_panel:
            raise ValueError(
                "demonstration and slot_diagnostic require validation_panel"
            )
        fractions = self.validation_fractions
        if (
            not isinstance(cast(object, fractions), tuple)
            or not fractions
            or fractions[-1] != 1
        ):
            raise ValueError(
                "validation_fractions must be a nonempty tuple ending at 1"
            )
        previous = 0.0
        for fraction in fractions:
            if (
                isinstance(fraction, bool)
                or not math.isfinite(fraction)
                or not previous < fraction <= 1
            ):
                raise ValueError("validation_fractions must increase within (0, 1]")
            previous = fraction
        if not isinstance(cast(object, self.checkpoint_env_steps), tuple):
            raise TypeError("checkpoint_env_steps must be a tuple")
        block = self.num_envs * self.ppo.rollout_length
        previous_step = 0
        for step in self.checkpoint_env_steps:
            if (
                type(step) is not int
                or not previous_step < step <= self.total_env_steps
                or (step != self.total_env_steps and step % block)
            ):
                raise ValueError(
                    "checkpoint_env_steps must increase at complete update boundaries"
                )
            previous_step = step


@dataclass(frozen=True)
class TrainResult:
    """Return saved paths and exact counts after all declared work succeeds.

    run_dir identifies the experiment; final_actor and selected_actor identify
    frozen exports (selection is None without a panel). completed_env_steps and
    completed_updates exclude padding. status is complete only after validation,
    export and reporting finish. evidence_paths names the shared report outputs.
    This record carries no rollout, live model state or learning qualification.
    """

    run_dir: Path
    final_actor: Path
    selected_actor: Path | None
    completed_env_steps: int
    completed_updates: int
    status: Literal["complete"]
    evidence_paths: tuple[Path, ...]


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    """Return version-1 JSON-ready settings; preserve every resolved option."""
    return {"schema_version": 1, **json.loads(json.dumps(asdict(config)))}


def config_from_dict(value: dict[str, Any]) -> TrainConfig:
    """Read version-1 settings and reject unknown fields before opening a run.

    value is a JSON object with optional schema_version=1. Omitted settings
    use TrainConfig defaults; ppo is a settings object. Lists for fractions and
    extra save points become tuples. Invalid versions, fields and values raise
    ValueError or TypeError. The input dictionary is not changed.
    """
    data = dict(value)
    version = data.pop("schema_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("Unsupported training config schema_version")
    unknown = set(data) - {item.name for item in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"Unknown training config fields: {sorted(unknown)}")
    if "ppo" in data:
        if not isinstance(data["ppo"], dict):
            raise TypeError("ppo must be a JSON object")
        data["ppo"] = PPOConfig(**cast(dict[str, Any], data["ppo"]))
    for name in ("validation_fractions", "checkpoint_env_steps"):
        if name in data:
            if not isinstance(data[name], (list, tuple)):
                raise TypeError(f"{name} must be a JSON array")
            data[name] = tuple(data[name])
    return TrainConfig(**data)


def read_config(path: str | Path) -> TrainConfig:
    """Load a UTF-8 JSON training config through the shared strict validator.

    path must name a JSON object. Read/parse errors propagate; invalid settings
    raise ValueError or TypeError before training files or a backend are opened.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Training config must be a JSON object")
    return config_from_dict(cast(dict[str, Any], value))


def _preserve_reports(root: Path, attempt: str) -> None:
    """Archive existing derived reports before moving their active continuation.

    Preserve raw bytes, including malformed reports, under their content hashes
    in the old attempt folder. Fsync before the caller clears active summaries.
    Immutable evaluation tasks remain in their original directories. The caller
    holds the run lock and has validated the complete recovery checkpoint.
    """
    directory = root / "attempts" / attempt / "reports"
    for name in (
        "validation_results.json",
        "random_diagnostics.json",
        "selection.json",
        "exposure.json",
        "slot_diagnostic.json",
        "run_summary.md",
    ):
        source = root / name
        if not source.is_file():
            continue
        raw = source.read_bytes()
        directory.mkdir(parents=True, exist_ok=True)
        destination = (
            directory / f"{source.stem}-{sha256(raw).hexdigest()}{source.suffix}"
        )
        if destination.exists():
            if destination.read_bytes() != raw:
                raise ValueError("Archived report identity conflicts")
            continue
        temporary = directory / f".{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


def train(
    config: TrainConfig | None = None,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> TrainResult:
    """Run or resume one complete declared MAPPO experiment on the current device.

    Parameters
    ----------
    config : TrainConfig or None, default=None
        Required for a new run. Resume inherits saved settings when omitted;
        an explicit config must match all saved settings exactly.
    output_dir : str or Path or None, default=None
        Exact new or empty run directory. Required for new runs and mutually
        exclusive with resume_from. No generated parent run name is inserted.
    resume_from : str or Path or None, default=None
        Exact complete learner checkpoint directory. Discover the containing
        run, verify scientific/source/content facts, execution identity and
        numerical state before writer rewind. The saved compiler policy,
        backend, device kind, runtime and numerical settings must match.
        Older checkpoints without that identity remain readable but cannot
        resume. A saved config without ppo.spawn_frame means "world" whatever
        the current default is; a supplied config that disagrees is rejected.
        Failed recovery retains its explicit retry marker.

    Returns
    -------
    TrainResult
        Saved actor/report paths and exact counts after all declared work ends.
        A development run without a panel has no selected actor. Completion
        records execution, not a claim of useful learned behavior.

    Raises
    ------
    ValueError, TypeError
        Settings, content, checkpoint or saved results are invalid/incompatible.
    RuntimeError
        Another writer owns the run or numerical execution fails.
    OSError
        Required output cannot be read or published. Failure state and the last
        complete checkpoint remain available when the filesystem permits it.

    Notes
    -----
    Runs synchronously and owns its files/process lock until completion. Device
    selection belongs to the caller's environment before importing JAX. No
    budget extension, replacement seed or silent retry occurs. Validation uses
    independent frozen Systems and keys, with its own saved M8 records.
    """
    attempt_started = time.monotonic()
    started_at = utc_now()
    from marl_battlegrounds.training import checkpoints, learner, validation
    from marl_battlegrounds.training._compilation import execution_identity
    from marl_battlegrounds.training.curriculum import make_training_schedule

    if (output_dir is None) == (resume_from is None):
        raise ValueError("Supply exactly one of output_dir or resume_from")
    saved: dict[str, Any] | None = None
    checkpoint = None if resume_from is None else Path(resume_from).resolve()
    if checkpoint is not None:
        saved = checkpoints.read_checkpoint_details(checkpoint)
        if saved["kind"] != "learner":
            raise ValueError("Resume requires a complete learner checkpoint")
        inherited = config_from_dict(checkpoints.saved_training_config(saved))
        if config is not None and config_to_dict(config) != config_to_dict(inherited):
            raise ValueError("Resume config differs from the saved experiment")
        config = inherited
        root = checkpoint.parent.parent
        if (
            checkpoint.parent.name != "checkpoints"
            or not (root / "run_details.json").is_file()
        ):
            raise ValueError("Checkpoint is outside its original training run")
    else:
        if not isinstance(config, TrainConfig):
            raise TypeError("A new run requires TrainConfig")
        root = Path(str(output_dir)).resolve()
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ValueError("output_dir must be new or empty")
    assert config is not None
    panel = (
        validation.load_panel(config.validation_panel)
        if config.validation_panel
        else None
    )
    if config.purpose == "demonstration" and (panel is None or not panel.qualified):
        raise ValueError("Demonstration requires a qualified frozen validation panel")
    schedule = make_training_schedule(
        total_env_steps=config.total_env_steps,
        num_envs=config.num_envs,
        curriculum=config.curriculum,
        score_threshold_curriculum=config.score_threshold_curriculum,
        early_history_capture=config.pinned_opponent_share > 0,
    )
    # Setup performs no real action. Resume replaces all template numerical values.
    collection, state = learner.init_learner(
        schedule=schedule,
        seed=config.seed,
        ppo=config.ppo,
        shaping=config.shaping,
        shaping_coefficient=config.shaping_coefficient,
        shaping_mode=config.shaping_mode,
        metrics=config.metrics,
        recording=config.recording,
        pinned_opponent_share=config.pinned_opponent_share,
    )
    if config.random_initialization_result is not None:
        from marl_battlegrounds.evaluation.recording_identity import tree_digest

        initial_digest = checkpoints._inference_digest(  # pyright: ignore[reportPrivateUsage]
            {
                "kind": "actor",
                "actor_digest": tree_digest(state.carry.history.current_variables),
                "input_scale": config.ppo.input_scale,
                "spawn_frame": config.ppo.spawn_frame,
            }
        )
        validation.read_random_initialization(
            config.random_initialization_result,
            actor_digest=initial_digest,
            seed_pairs=cast(int, config.random_diagnostic_seed_pairs),
        )
    identity = checkpoints.runtime_identity()
    source = cast(dict[str, Any], identity["source"])
    dependencies = identity["dependencies"]
    runtime = _runtime_details(str(source["package_version"]), config.num_envs)
    execution_identity_now = execution_identity(runtime=runtime["provenance"])
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root), ExitStack() as stack:
        writer = None
        metadata: dict[str, Any]
        if saved is not None:
            assert checkpoint is not None
            restored = checkpoints.restore_checkpoint(
                checkpoint,
                collection,
                state,
                expected_metadata={
                    "config": config_to_dict(config),
                    "source": source,
                    "dependencies": dependencies,
                    "execution": execution_identity_now,
                },
                ppo=config.ppo,
            )
            metadata = dict(restored.details["metadata"])
            cursors = metadata.get("log_cursors", {})
            if set(cursors) != {"training_updates.jsonl"}:
                raise ValueError("Checkpoint has no valid training-log recovery cursor")
            validate_log_cursor(
                root / "training_updates.jsonl", cursors["training_updates.jsonl"]
            )
            run_details = json.loads((root / "run_details.json").read_text())
            if run_details["run_id"] != metadata["run_id"]:
                raise ValueError("Checkpoint belongs to a different training run")
            if run_details.get("schemas") != restored.details["schemas"]:
                raise ValueError("Run and checkpoint schema versions differ")
            if run_details.get("execution") != metadata["execution"]:
                raise ValueError("Run and checkpoint execution identities differ")
            if run_details["panel_digest"] != (None if panel is None else panel.digest):
                raise ValueError("Frozen validation panel differs from the saved run")
            state = restored.state
            host = metadata["host_state"]
            if host["env_steps"] != int(
                state.carry.progress.rounds
            ) * config.num_envs or host["completed_updates"] != int(
                state.completed_updates
            ):
                raise ValueError("Checkpoint host counters differ from learner state")
            validate_host_state(root, restored.details, panel=panel)
            if (root / "status.json").is_file():
                prior_status = json.loads((root / "status.json").read_text())
                prior_elapsed = prior_status.get("elapsed_seconds", 0.0)
                if (
                    prior_status.get("run_id") == metadata["run_id"]
                    and isinstance(prior_elapsed, (int, float))
                    and math.isfinite(prior_elapsed)
                ):
                    host["elapsed_seconds"] = max(
                        host["elapsed_seconds"], prior_elapsed
                    )
            writer = checkpoints.resume_recording(restored, root)
            if writer is not None:
                stack.callback(writer.close)
            abandoned = preserve_log_suffix(
                root / "training_updates.jsonl",
                cursors["training_updates.jsonl"],
                root / "attempts" / metadata["attempt_id"],
            )
            if abandoned is not None:
                append_jsonl(
                    root / "run_events.jsonl",
                    {
                        "event": "abandoned_updates_preserved",
                        "time_utc": utc_now(),
                        "attempt_id": metadata["attempt_id"],
                        "path": str(abandoned),
                        "restore_checkpoint": checkpoint.name,
                    },
                    durable=True,
                )
            restore_log_cursor(
                root / "training_updates.jsonl", cursors["training_updates.jsonl"]
            )
            _preserve_reports(root, metadata["attempt_id"])
            atomic_json(
                root / "validation_results.json",
                host.get("routine_results", []) + host.get("confirmation_results", []),
            )
            if config.random_diagnostic_seed_pairs is not None:
                atomic_json(root / "random_diagnostics.json", host["random_results"])
            atomic_json(root / "selection.json", host.get("selection"))
            atomic_json(root / "exposure.json", None)
            atomic_json(root / "slot_diagnostic.json", None)
            # Publish unfinished status before clearing recovery ownership. A
            # crash before the new attempt starts must not revive old completion.
            atomic_json(
                root / "status.json",
                {
                    "schema_version": 1,
                    "run_id": metadata["run_id"],
                    "attempt_id": metadata["attempt_id"],
                    "status": "incomplete",
                    "phase": "resuming",
                    "env_steps": host["env_steps"],
                    "total_env_steps": config.total_env_steps,
                    "completed_updates": host["completed_updates"],
                    "elapsed_seconds": host["elapsed_seconds"],
                    "latest_checkpoint": str(checkpoint),
                    "pending_task": host["pending"],
                    "process": process_identity(),
                    "updated_at": utc_now(),
                },
            )
            checkpoints.finish_checkpoint_recovery(restored, root)
            metadata["parent_checkpoint"] = checkpoint.name
            metadata["attempt_id"] = uuid.uuid4().hex
        else:
            metadata = {
                "run_id": uuid.uuid4().hex,
                "attempt_id": uuid.uuid4().hex,
                "parent_checkpoint": None,
                "config": config_to_dict(config),
                "source": source,
                "dependencies": dependencies,
                "execution": execution_identity_now,
                "host_state": {},
                "recording": None,
            }
            if config.recording:
                from marl_battlegrounds.evaluation.recording_identity import (
                    normalize_system_registration,
                )
                from marl_battlegrounds.evaluation.run_writer import RunWriter

                policies: dict[str, object] = {
                    name: normalize_system_registration(system, phase="training")[1]
                    for name, system in (
                        ("team_a", collection.actor),
                        ("team_b", collection.opponent),
                    )
                }
                writer = RunWriter(
                    root / "episodes", phase="training", policies=policies
                )
                stack.callback(writer.close)
                metadata["recording"] = {
                    "relative_path": writer.run_dir.relative_to(root).as_posix(),
                    "phase": "training",
                    "pass_id": "1",
                    "policies": policies,
                    "checkpoint_id": None,
                    "details": {},
                }
            atomic_json(
                root / "run_details.json",
                {
                    "schema_version": 1,
                    "run_id": metadata["run_id"],
                    "created_at": started_at,
                    "initial_setup_seconds": time.monotonic() - attempt_started,
                    "config": metadata["config"],
                    "source": source,
                    "dependencies": dependencies,
                    "schemas": checkpoints.checkpoint_schemas(),
                    "execution": execution_identity_now,
                    "runtime": runtime,
                    "content_binding": collection.binding.model_dump(mode="json"),
                    "schedule": dict(schedule.rounding_report),
                    "panel_digest": None if panel is None else panel.digest,
                    "process": process_identity(),
                },
            )
        execution = _Run(
            root,
            config,
            collection,
            state,
            metadata,
            writer,
            panel,
            checkpoint,
            attempt_started=attempt_started,
            runtime=runtime,
        )
        return execution.execute()


def _runtime_details(package_version: str, num_envs: int) -> dict[str, Any]:
    """Read M8 runtime facts and declared device selectors once per attempt.

    package_version is the verified source version and num_envs is the declared
    positive batch. Reuse M8's backend/device/precision record after setup has
    selected JAX. Environment values describe selectors, not verified physical
    GPU identities; the frozen launch package owns its UUID/PCI verification.
    No model call, timing, transfer, file write or driver query occurs here.
    """
    from marl_battlegrounds.evaluation.runtime_provenance import (
        capture_runtime_provenance,
    )

    return {
        "provenance": capture_runtime_provenance(
            package_version, num_envs=num_envs, policy_execution_included=True
        ).model_dump(mode="json"),
        "declared_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "JAX_DISABLE_JIT",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
            )
        },
    }


def _memory_snapshot() -> dict[str, Any]:
    """Read process peak RAM and JAX allocator counters after an attempt ends.

    Return optional measurements and explicit read errors. RAM is the process
    lifetime high-water mark, including earlier work in a reused Python process.
    JAX counters exclude driver/display memory and CPU worker processes. Missing
    allocator statistics are None. This performs no GPU synchronization, array
    transfer, model call, polling or file write. A failed measurement cannot hide
    the attempt's original failure. Call only after the learner loop/report work.
    """
    result: dict[str, Any] = {
        "process_peak_ram_bytes": None,
        "jax_allocator": None,
        "errors": [],
        "scope": (
            "Process lifetime, including earlier work and compilation. JAX "
            "allocator counters exclude driver/display memory and CPU workers."
        ),
    }
    try:
        import resource
        import sys

        scale = 1 if sys.platform == "darwin" else 1024
        result["process_peak_ram_bytes"] = (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale
        )
    except Exception as error:
        result["errors"].append(f"Process RAM: {type(error).__name__}: {error}")
    try:
        import jax

        result["jax_allocator"] = jax.devices()[0].memory_stats()
    except Exception as error:
        result["errors"].append(f"JAX allocator: {type(error).__name__}: {error}")
    return result


def _random_progress(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarize already verified Random games for the human progress display.

    results contains this run's completed Random captures, including an optional
    shared initialization. Return the latest capture's game counts, equal-map
    mean kills and deaths, and its change from initialization. Missing combat
    scores remain None. Capture times are the times when the actor was saved,
    not the time when its evaluation finished. Empty results return None.
    This only reads small host dictionaries; it opens no files, touches no
    device arrays and changes neither evidence nor checkpoint state.
    """
    if not results:
        return None

    def mean_scores(record: dict[str, Any]) -> tuple[float | None, float | None]:
        """Average map scores equally; missing or invalid scores stay unavailable."""
        cells = record.get("cells", [])
        if not cells:
            return None, None
        first = [cell.get("mean_team_a_score") for cell in cells]
        second = [cell.get("mean_team_b_score") for cell in cells]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in (*first, *second)
        ):
            return None, None
        return math.fsum(first) / len(cells), math.fsum(second) / len(cells)

    latest = max(results, key=lambda record: record["env_steps"])
    initial = next((record for record in results if record["env_steps"] == 0), None)
    kills, deaths = mean_scores(latest)
    margin = None if kills is None or deaths is None else kills - deaths
    initial_margin = None
    if initial is not None:
        initial_kills, initial_deaths = mean_scores(initial)
        if initial_kills is not None and initial_deaths is not None:
            initial_margin = initial_kills - initial_deaths
    return {
        "opponent": "Random",
        "task_id": latest["task_id"],
        "initial_task_id": None if initial is None else initial["task_id"],
        "env_steps": latest["env_steps"],
        "games": latest["games"],
        "wall_seconds": latest["wall_seconds"],
        "training_seconds": latest["training_seconds"],
        "score": latest["score"],
        **{
            name: sum(cell[name] for cell in latest["cells"])
            for name in ("wins", "draws", "losses")
        },
        "mean_kills_for": kills,
        "mean_kills_against": deaths,
        "kill_margin": margin,
        "initial_kill_margin": initial_margin,
        "kill_margin_change": None
        if margin is None or initial_margin is None
        else margin - initial_margin,
    }


class _Run:
    """Own one locked process attempt and its small host bookkeeping.

    The public train function validates setup/recovery first. This object holds
    one learner state and stable compiled updater; it never stores a rollout
    after its update. Files and validation are performed between numerical calls.
    """

    def __init__(
        self,
        root: Path,
        config: TrainConfig,
        collection: TrainingCollection,
        state: LearnerState,
        metadata: dict[str, Any],
        writer: RunWriter | None,
        panel: FrozenPanel | None,
        checkpoint: Path | None,
        *,
        attempt_started: float,
        runtime: dict[str, Any],
    ) -> None:
        """Attach validated owners and reconstruct host counters from a checkpoint."""
        from functools import partial

        import jax

        from marl_battlegrounds.training._compilation import training_compiler_options
        from marl_battlegrounds.training.learner import update_learner
        from marl_battlegrounds.training.validation import resolve_validation_schedule

        self.root, self.config = root, config
        self.collection, self.state = collection, state
        self.metadata, self.writer, self.panel = metadata, writer, panel
        self.runtime = runtime
        self.host = metadata["host_state"]
        defaults: dict[str, Any] = {
            "env_steps": 0,
            "completed_updates": 0,
            "actor_decisions": 0,
            "used_policy_samples": 0,
            "used_value_samples": 0,
            "training_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "routine_results": [],
            "confirmation_results": [],
            "random_results": [],
            "actors": {},
            "pending": None,
            "selected_actor": None,
            "final_actor": None,
            "validation_seconds": 0.0,
            "validation_games": 0,
            "save_seconds": 0.0,
            "saves": 0,
            "report_seconds": None,
            "slot_complete": False,
            "selection": None,
            "recovery_checkpoints": [],
        }
        for name, default in defaults.items():
            self.host.setdefault(name, default)
        self.checkpoint = checkpoint
        if (
            checkpoint is not None
            and checkpoint.name not in self.host["recovery_checkpoints"]
        ):
            self.host["recovery_checkpoints"].append(checkpoint.name)
        self.start = attempt_started
        self.prior_elapsed = self.host["elapsed_seconds"]
        details = json.loads((self.root / "run_details.json").read_text())
        self.created_at = datetime.fromisoformat(details["created_at"]).timestamp()
        self.reporter = ProgressReporter(enabled=config.verbose)
        self.speed_estimate = TrainingSpeedEstimate() if config.verbose else None
        self.updater = cast(
            "_Updater",
            jax.jit(
                partial(update_learner, ppo=config.ppo),
                compiler_options=training_compiler_options(),
            ),
        )
        self.validation_steps: set[int] = (
            {
                point.env_steps
                for point in resolve_validation_schedule(
                    config.total_env_steps,
                    config.num_envs,
                    rollout_length=config.ppo.rollout_length,
                    fractions=config.validation_fractions,
                )
            }
            if panel is not None
            else set()
        )
        self.random_steps: set[int] = (
            {0, *config.checkpoint_env_steps, config.total_env_steps}
            if config.random_diagnostic_seed_pairs is not None
            else set()
        )
        self.status: dict[str, Any] = {}
        if config.random_diagnostic_seed_pairs is not None:
            self.status["random_validation"] = _random_progress(
                self.host["random_results"]
            )

    def event(self, event_name: str, **facts: object) -> None:
        """Durably append one lifecycle event without changing scientific counters."""
        append_jsonl(
            self.root / "run_events.jsonl",
            {
                "event": event_name,
                "time_utc": utc_now(),
                "attempt_id": self.metadata["attempt_id"],
                "env_steps": self.host["env_steps"],
                **facts,
            },
            durable=True,
        )

    def set_status(self, phase: str, **facts: object) -> None:
        """Publish host counters and the requested reset-time K without device reads."""
        self.host["elapsed_seconds"] = (
            self.prior_elapsed + time.monotonic() - self.start
        )
        training_seconds = self.host["training_seconds"]
        rate = self.host["env_steps"] / training_seconds if training_seconds else None
        report = self.collection.schedule.rounding_report
        thresholds = report.get("score_thresholds", (20,))
        ends = cast(tuple[int, ...], report["cumulative_round_ends"])
        stage = min(
            sum(end * self.config.num_envs <= self.host["env_steps"] for end in ends),
            len(ends) - 1,
        )
        requested_threshold = (
            cast(tuple[int, ...], thresholds)[stage]
            if self.config.score_threshold_curriculum
            else 20
        )
        self.status.update(
            {
                "schema_version": 1,
                "run_id": self.metadata["run_id"],
                "attempt_id": self.metadata["attempt_id"],
                "process": process_identity(),
                "phase": phase,
                "status": "running",
                "total_env_steps": self.config.total_env_steps,
                "env_steps": self.host["env_steps"],
                "score_threshold": requested_threshold,
                "completed_updates": self.host["completed_updates"],
                "final_actor": self.host["final_actor"],
                "selected_actor": self.host["selected_actor"],
                "elapsed_seconds": self.host["elapsed_seconds"],
                "wall_seconds": max(0.0, time.time() - self.created_at),
                "transitions_per_second": rate,
                "estimated_transitions_per_second": None
                if self.speed_estimate is None
                else self.speed_estimate.rate,
                "latest_checkpoint": None
                if self.checkpoint is None
                else str(self.checkpoint),
                "pending_task": self.host["pending"],
                "pending_work_seconds": 0.0
                if phase == "complete"
                else self.pending_seconds()
                if self.config.verbose
                else None,
                "updated_at": utc_now(),
                **facts,
            }
        )
        atomic_json(self.root / "status.json", self.status)
        self.reporter.report(self.status)

    def pending_seconds(self) -> float | None:
        """Estimate output time from completed phase timings, never new GPU work.

        Future confirmation allows its maximum three candidates. Completed
        validation games estimate final-diagnostic cost too; the ETA is an
        approximation, not a promised finish time. Unknown phase costs give None.
        """
        if not self.host["saves"] or self.host["report_seconds"] is None:
            return None
        remaining_updates = math.ceil(
            (self.config.total_env_steps - self.host["env_steps"])
            / (self.config.num_envs * self.config.ppo.rollout_length)
        )
        estimate = (
            math.ceil(remaining_updates / self.config.checkpoint_interval_updates)
            * self.host["save_seconds"]
            / self.host["saves"]
            + self.host["report_seconds"]
        )
        random_done = {row["env_steps"] for row in self.host["random_results"]}
        random_games = (
            len(self.random_steps - random_done)
            * 10
            * (self.config.random_diagnostic_seed_pairs or 0)
        )
        if self.panel is None and not random_games:
            return float(estimate)
        if not self.host["validation_games"]:
            return None
        completed = {row["env_steps"] for row in self.host["routine_results"]}
        remaining_points = len(self.validation_steps - completed)
        games = random_games
        if self.panel is not None:
            games += (
                remaining_points
                * 5
                * len(self.panel.members)
                * 2
                * self.config.routine_seed_pairs
            )
            games += (
                max(0, 3 - len(self.host["confirmation_results"]))
                * 5
                * len(self.panel.members)
                * 2
                * self.config.confirmation_seed_pairs
            )
        if self.config.slot_diagnostic and not self.host["slot_complete"]:
            games += 3200
        return float(
            estimate
            + games * self.host["validation_seconds"] / self.host["validation_games"]
        )

    def save(self) -> Path:
        """Publish one accepted learner boundary with its exact host-log prefix."""
        from marl_battlegrounds.training.checkpoints import save_checkpoint

        self.host["elapsed_seconds"] = (
            self.prior_elapsed + time.monotonic() - self.start
        )
        self.metadata["log_cursors"] = {
            "training_updates.jsonl": log_cursor(self.root / "training_updates.jsonl")
        }
        started = time.monotonic()
        self.checkpoint = save_checkpoint(
            self.root,
            self.collection,
            self.state,
            metadata=self.metadata,
            writer=self.writer,
            ppo=self.config.ppo,
        )
        self.metadata["parent_checkpoint"] = self.checkpoint.name
        seconds = time.monotonic() - started
        self.host["save_seconds"] += seconds
        self.host["saves"] += 1
        self.event(
            "checkpoint_saved",
            checkpoint_id=self.checkpoint.name,
            seconds=seconds,
        )
        self.prune_recovery()
        return self.checkpoint

    def prune_recovery(self) -> None:
        """Keep two recent recovery payloads and every exported candidate boundary.

        Only payload directories of older checkpoints from this active attempt's
        saved recovery list are removed after a new durable pointer. Small
        ancestry descriptions, actor exports and all M8 recording bundles remain.
        """
        assert self.checkpoint is not None
        recent: list[str] = self.host["recovery_checkpoints"]
        if self.checkpoint.name not in recent:
            recent.append(self.checkpoint.name)
        while len(recent) > 2:
            identifier = recent.pop(0)
            if identifier in self.host["actors"]:
                continue
            if len(identifier) != 64 or any(
                char not in "0123456789abcdef" for char in identifier
            ):
                raise ValueError("Invalid checkpoint retention identity")
            directory = self.root / "checkpoints" / identifier
            if directory.is_symlink():
                raise ValueError("Checkpoint retention cannot follow a symlink")
            for name in ("actor", "state"):
                payload = directory / name
                if payload.is_symlink():
                    raise ValueError(
                        "Checkpoint retention cannot follow a payload symlink"
                    )
                if payload.exists():
                    shutil.rmtree(payload)
            self.event("checkpoint_payload_pruned", checkpoint_id=identifier)

    def actor(self) -> Path:
        """Export this boundary's exact actor; reuse an identical existing artifact."""
        from marl_battlegrounds.training.checkpoints import export_system

        assert self.checkpoint is not None
        destination = self.root / "actors" / self.checkpoint.name
        destination.parent.mkdir(exist_ok=True)
        export_system(
            self.state.carry.history.current_variables,
            destination,
            input_scale=self.config.ppo.input_scale,
            spawn_frame=self.config.ppo.spawn_frame,
            metadata={
                "run_id": self.metadata["run_id"],
                "seed": self.config.seed,
                "env_steps": self.host["env_steps"],
                "checkpoint_id": self.checkpoint.name,
            },
        )
        self.host["actors"][self.checkpoint.name] = str(destination)
        return destination

    def validate(self, actor: Path, purpose: str) -> dict[str, Any]:
        """Run or recover one frozen validation task and record its exact result."""
        import jax

        from marl_battlegrounds.training.validation import validate_checkpoint

        assert self.panel is not None
        self.set_status("validation")
        started = time.monotonic()
        summary = validate_checkpoint(
            actor,
            self.panel,
            output_dir=self.root / "validation" / f"{purpose}-{actor.name}",
            purpose=purpose,
            seed_pairs=self.config.routine_seed_pairs
            if purpose == "routine"
            else self.config.confirmation_seed_pairs,
            num_envs=32
            if jax.default_backend() == "gpu"
            else min(32, self.config.num_envs),
            chunk_size=128,
            event_callback=lambda record: self.event("validation_segment", **record),
        )
        key = "routine_results" if purpose == "routine" else "confirmation_results"
        if not any(item["task_id"] == summary["task_id"] for item in self.host[key]):
            summary = {
                **summary,
                "elapsed_seconds": self.prior_elapsed + time.monotonic() - self.start,
            }
            self.host[key].append(summary)
            seconds = time.monotonic() - started
            self.host["validation_seconds"] += seconds
            pairs = (
                self.config.routine_seed_pairs
                if purpose == "routine"
                else self.config.confirmation_seed_pairs
            )
            self.host["validation_games"] += 5 * len(self.panel.members) * 2 * pairs
            self.event(
                "validation_complete",
                result=summary,
                seconds=seconds,
            )
        atomic_json(
            self.root / "validation_results.json",
            self.host["routine_results"] + self.host["confirmation_results"],
        )
        self.status["validation_score"] = summary["score"]
        return summary

    def finish_pending(self) -> None:
        """Finish the saved boundary's capture/validation without another decision."""
        pending = self.host["pending"]
        if pending is None:
            return
        actor = self.actor()
        if pending.get("routine"):
            self.validate(actor, "routine")
        if pending.get("random"):
            self.validate_random(actor)
        if self.host["env_steps"] == self.config.total_env_steps:
            self.host["final_actor"] = str(actor)
        self.host["pending"] = None
        if pending.get("routine"):
            self.report()

    def validate_random(self, actor: Path) -> None:
        """Complete one fixed Random capture without changing learner or its keys.

        actor is this saved boundary's immutable export. Initialization may reuse
        the fully verified original result configured by the caller. Capture
        timings describe this run just before evaluation/reuse. Original task,
        actor, checkpoint and M8 pass identities stay unchanged on reuse.
        Publish the accumulated records and existing host game and combat means.
        Combat changes compare equal-map mean kills minus deaths with the
        verified initialization. They are early learning clues, not native wins.
        """
        import jax

        from marl_battlegrounds.training import checkpoints, validation

        pairs = self.config.random_diagnostic_seed_pairs
        assert pairs is not None
        if any(
            row["env_steps"] == self.host["env_steps"]
            for row in self.host["random_results"]
        ):
            return
        self.set_status("random_validation")
        started = time.monotonic()
        elapsed = self.prior_elapsed + started - self.start
        wall = max(0.0, time.time() - self.created_at)
        reference = (
            self.config.random_initialization_result
            if self.host["env_steps"] == 0
            else None
        )
        record: dict[str, Any]
        if reference is None:
            directory = self.root / "validation" / f"random-{actor.name}"
            summary = validation.validate_random(
                actor,
                output_dir=directory,
                seed_pairs=pairs,
                num_envs=32
                if jax.default_backend() == "gpu"
                else min(32, self.config.num_envs),
                chunk_size=128,
                event_callback=lambda record: self.event(
                    "random_validation_segment", **record
                ),
            )
            record = {
                **summary,
                "actor_path": str(actor),
                "summary_path": str(directory / "validation_summary.json"),
                "reference_path": None,
                "reused_initialization": False,
            }
            self.host["validation_games"] += summary["games"]
        else:
            identity = checkpoints.artifact_identity(actor)
            record = {
                **validation.read_random_initialization(
                    reference, actor_digest=identity["actor_digest"], seed_pairs=pairs
                ),
                "reference_path": str(Path(reference).absolute()),
                "reused_initialization": True,
            }
        record.update(
            elapsed_seconds=elapsed,
            wall_seconds=wall,
            training_seconds=self.host["training_seconds"],
        )
        self.host["random_results"].append(record)
        seconds = time.monotonic() - started
        self.host["validation_seconds"] += seconds
        atomic_json(self.root / "random_diagnostics.json", self.host["random_results"])
        self.event("random_validation_complete", result=record, seconds=seconds)
        self.set_status(
            "random_validation_complete",
            validation_score=record["score"],
            random_validation=_random_progress(self.host["random_results"]),
            **{
                f"validation_{name}": sum(cell[name] for cell in record["cells"])
                for name in ("wins", "draws", "losses")
            },
        )

    def report(self) -> tuple[Path, ...]:
        """Refresh shared reports from durable source records without new evaluation."""
        from marl_battlegrounds.training.analysis import analyze

        self.set_status("reporting")
        started = time.monotonic()
        result = analyze([self.root], output_dir=self.root)
        self.host["report_seconds"] = time.monotonic() - started
        self.event("reports_written", seconds=self.host["report_seconds"])
        return tuple(Path(path) for path in result["artifacts"].values())

    def execute(self) -> TrainResult:
        """Run exact-budget work and preserve failures for explicit resume."""
        import jax
        import numpy as np

        from marl_battlegrounds.training.analysis import (
            confirmation_candidates,
            refresh_summary,
            select_checkpoint,
        )
        from marl_battlegrounds.training.collection import (
            collect_training_rollout,
            training_summary,
        )

        self.event(
            "attempt_started",
            resumed_from=None if self.checkpoint is None else str(self.checkpoint),
            setup_seconds=time.monotonic() - self.start,
            runtime=self.runtime,
            execution=self.metadata["execution"],
        )
        try:
            if self.checkpoint is None:
                self.host["pending"] = {
                    "routine": 0 in self.validation_steps,
                    "random": 0 in self.random_steps,
                }
                self.set_status("initializing")
                self.save()
            self.finish_pending()
            if self.host["env_steps"] < self.config.total_env_steps:
                self.set_status(
                    "compiling_first_update"
                    if not self.host["completed_updates"]
                    else "training"
                )
            while self.host["env_steps"] < self.config.total_env_steps:
                started = time.monotonic()
                collected, rollout = collect_training_rollout(
                    self.collection,
                    self.state.carry,
                    length=self.config.ppo.rollout_length,
                    writer=self.writer,
                )
                # The required update-result transfer is the only learner sync.
                next_state, result = self.updater(self.state, collected, rollout)
                host_result = jax.device_get(result)
                seconds = time.monotonic() - started
                if bool(host_result.failed) or not bool(host_result.performed):
                    raise RuntimeError(
                        "Learner update failed: "
                        f"reason {int(host_result.failure_reason)}"
                    )
                self.state = next_state
                summary, metrics = host_result.summary, host_result.metrics
                self.host["env_steps"] += int(summary.real_transitions)
                self.host["completed_updates"] += 1
                self.host["actor_decisions"] += int(summary.live_actor_decisions)
                self.host["used_policy_samples"] += sum(
                    int(x) for x in np.asarray(metrics.actor_samples).flat
                )
                self.host["used_value_samples"] += sum(
                    int(x) for x in np.asarray(metrics.critic_samples).flat
                )
                for name in (
                    "stage_completed",
                    "stage_score_sums",
                    "stage_length_sum",
                    "stage_k20_count",
                ):
                    incoming = np.asarray(getattr(summary, name), dtype=object)
                    previous = np.asarray(
                        self.host.get(name, np.zeros(incoming.shape, dtype=object)),
                        dtype=object,
                    )
                    self.host[name] = (previous + incoming).tolist()
                self.host["training_seconds"] += seconds
                if self.speed_estimate is not None:
                    self.speed_estimate.observe(int(summary.real_transitions), seconds)
                self.host["elapsed_seconds"] = (
                    self.prior_elapsed + time.monotonic() - self.start
                )
                row = {
                    "schema_version": 1,
                    "attempt_id": self.metadata["attempt_id"],
                    "update_index": self.host["completed_updates"],
                    "wall_seconds": max(0.0, time.time() - self.created_at),
                    **{
                        key: self.host[key]
                        for key in (
                            "env_steps",
                            "actor_decisions",
                            "used_policy_samples",
                            "used_value_samples",
                            "training_seconds",
                            "elapsed_seconds",
                        )
                    },
                    "collection_update_seconds": seconds,
                    "task_reward_mean": float(summary.task_reward_sum)
                    / int(summary.active_samples)
                    if int(summary.active_samples)
                    else None,
                    "shaping_mean": float(summary.shaping_reward_sum)
                    / int(summary.real_transitions),
                    "policy_loss": float(
                        np.asarray(metrics.actor_loss)[
                            np.asarray(metrics.actor_samples) > 0
                        ].mean()
                    )
                    if np.any(metrics.actor_samples)
                    else None,
                    "value_loss": float(
                        np.asarray(metrics.value_loss)[
                            np.asarray(metrics.critic_samples) > 0
                        ].mean()
                    )
                    if np.any(metrics.critic_samples)
                    else None,
                    "entropy": float(
                        np.asarray(metrics.entropy)[
                            np.asarray(metrics.actor_samples) > 0
                        ].mean()
                    )
                    if np.any(metrics.actor_samples)
                    else None,
                    "learning_rate": self.config.ppo.actor_lr,
                }
                append_jsonl(self.root / "training_updates.jsonl", row)
                del rollout, collected, result, next_state
                steps = self.host["env_steps"]
                capture = (
                    steps in self.validation_steps
                    or steps in self.config.checkpoint_env_steps
                    or steps == self.config.total_env_steps
                )
                if capture:
                    self.host["pending"] = {
                        "routine": steps in self.validation_steps,
                        "random": steps in self.random_steps,
                    }
                if (
                    capture
                    or self.host["completed_updates"]
                    % self.config.checkpoint_interval_updates
                    == 0
                ):
                    self.save()
                    self.finish_pending()
                self.set_status(
                    "training",
                    recent_transitions_per_second=int(summary.real_transitions)
                    / seconds,
                    **{
                        key: row[key]
                        for key in (
                            "policy_loss",
                            "value_loss",
                            "entropy",
                            "task_reward_mean",
                            "shaping_mean",
                        )
                    },
                )
            if self.panel is not None:
                final = Path(self.host["final_actor"])
                for identifier in confirmation_candidates(
                    self.host["routine_results"], final_checkpoint_id=final.name
                ):
                    self.validate(Path(self.host["actors"][identifier]), "confirmation")
                selected = select_checkpoint(self.host["confirmation_results"])
                self.host["selection"] = selected
                self.host["selected_actor"] = self.host["actors"][
                    selected["checkpoint_id"]
                ]
                atomic_json(self.root / "selection.json", selected)
            if self.config.slot_diagnostic:
                self.run_slot_check()
            exposure = training_summary(self.collection, self.state.carry)
            exposure.update(
                {
                    name: self.host.get(name)
                    for name in (
                        "stage_completed",
                        "stage_score_sums",
                        "stage_length_sum",
                        "stage_k20_count",
                    )
                }
            )
            thresholds = cast(list[int], exposure["score_thresholds_by_episode_stage"])
            completed = self.host.get("stage_completed")
            if completed is not None:
                for row in cast(
                    list[dict[str, Any]], exposure["exposure_by_score_threshold"]
                ):
                    counts = [
                        sum(
                            completed[i][outcome]
                            for i, threshold in enumerate(thresholds)
                            if threshold == row["score_threshold"]
                        )
                        for outcome in range(3)
                    ]
                    row.update(
                        completed_games=sum(counts),
                        wins=counts[0],
                        draws=counts[1],
                        losses=counts[2],
                    )
            atomic_json(self.root / "exposure.json", exposure)
            self.set_status("finalizing")
            paths = self.report()
            if not all(path.is_file() and path.stat().st_size for path in paths):
                raise ValueError("Required final reports are missing or empty")
            memory = _memory_snapshot()
            self.set_status(
                "complete",
                status="complete",
                pending_work_seconds=0.0,
                exit_code=0,
                memory=memory,
            )
            refresh_summary([self.root], output_dir=self.root)
            self.event("attempt_complete", memory=memory)
            return TrainResult(
                self.root,
                Path(self.host["final_actor"]),
                None
                if self.host["selected_actor"] is None
                else Path(self.host["selected_actor"]),
                self.host["env_steps"],
                self.host["completed_updates"],
                "complete",
                paths,
            )
        except BaseException as error:
            memory = _memory_snapshot()
            self.event(
                "attempt_failed",
                error=f"{type(error).__name__}: {error}",
                memory=memory,
            )
            self.set_status(
                "failed",
                status="failed",
                error=str(error),
                exit_code=130 if isinstance(error, KeyboardInterrupt) else 1,
                memory=memory,
            )
            try:
                refresh_summary([self.root], output_dir=self.root)
            except Exception as report_error:
                self.event("failure_summary_unavailable", error=str(report_error))
            raise

    def run_slot_check(self) -> None:
        """Run the declared trained comparison only after the final actor exists."""
        from marl_battlegrounds.training.validation import run_slot_diagnostic

        assert self.panel is not None
        self.set_status("slot_diagnostic")
        started = time.monotonic()
        result = run_slot_diagnostic(
            Path(self.host["final_actor"]),
            self.panel.members[-1].path,
            output_dir=self.root
            / "slot_diagnostic"
            / Path(self.host["final_actor"]).name,
            event_callback=lambda record: self.event("slot_segment", **record),
        )
        atomic_json(self.root / "slot_diagnostic.json", result)
        self.host["slot_complete"] = True
        self.event("slot_diagnostic_finished", seconds=time.monotonic() - started)
