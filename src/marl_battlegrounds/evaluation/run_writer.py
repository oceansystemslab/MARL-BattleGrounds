"""Buffered scalar episode tables with explicit, durable run resumption."""

from __future__ import annotations

import csv
import fcntl
import io
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import numpy as np

from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_V3,
    SHARED_OBS_ACTOR_PROJECTION_V2,
)
from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
    METRIC_SCHEMA_ID,
    METRIC_SCHEMA_VERSION,
    PRIORITY_METRIC_NAMES,
)

if TYPE_CHECKING:
    from marl_battlegrounds.core.types import EnvConfig
    from marl_battlegrounds.environment import EpisodeInfo
    from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV3
    from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
    from marl_battlegrounds.evaluation.replay_capture import ReplayPackets
    from marl_battlegrounds.evaluation.replay_recording import ReplayCollector
    from marl_battlegrounds.evaluation.tournament_statistics import TournamentStatistics

IDENTITY_COLUMNS = (
    "run_id",
    "phase",
    "pass_id",
    "episode_id",
    "seed_id",
    "map_id",
    "config_id",
    "team_a_policy",
    "team_b_policy",
    "checkpoint_id",
    *(
        name
        for slot in range(10)
        for name in (f"agent_{slot}_class_id", f"agent_{slot}_active")
    ),
)

MATCH_COLUMNS = (
    *IDENTITY_COLUMNS,
    "block_id",
    "bootstrap_group",
    "outcome",
    *PRIORITY_METRIC_NAMES,
)
_SUMMARY_TABLES = ("tournament_results.csv", "matchup_results.csv", "map_results.csv")
_INPUT_PROJECTIONS = {
    "shared_obs": SHARED_OBS_ACTOR_PROJECTION_V2.model_dump(mode="json"),
    "no_shared_obs": NO_SHARED_OBS_ACTOR_PROJECTION_V3.model_dump(mode="json"),
}


def _json_value(value: object) -> object:
    named_fields = getattr(value, "_fields", None)
    if isinstance(value, tuple) and named_fields is not None:
        fields = cast(tuple[str, ...], named_fields)
        return {
            key: _json_value(item)
            for key, item in zip(fields, cast(tuple[object, ...], value), strict=True)
        }
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in cast(dict[object, object], value).items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item) for item in cast(list[object] | tuple[object, ...], value)
        ]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return cast(Any, value).item()
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def configuration_identity(config: EnvConfig) -> tuple[str, dict[str, object]]:
    """Identify one environment-normalized configuration by its canonical content."""
    import jax

    content = cast(dict[str, object], _json_value(jax.device_get(config)))
    return sha256(_json_bytes(content)).hexdigest(), content


class RunWriter:
    """Write every completed episode from a step or complete collected chunk.

    New runs get unique child directories. Resume is explicit and restores the
    last durable boundary; it never treats a partial CSV suffix as completed work.
    Call ``start_pass`` when appending another validation pass to the same writer.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        resume_from: str | Path | None = None,
        phase: str = "evaluation",
        pass_id: str = "1",
        policies: dict[str, object] | None = None,
        checkpoint_id: str | None = None,
        buffer_size: int = 128,
        details: dict[str, object] | None = None,
    ) -> None:
        if (output_dir is None) == (resume_from is None):
            raise ValueError(
                "supply output_dir for a new run or resume_from for an existing run"
            )
        if (
            isinstance(buffer_size, bool)
            or not isinstance(cast(object, buffer_size), int)
            or buffer_size < 1
        ):
            raise ValueError("buffer_size must be positive")
        self._closed = False
        self._failed: BaseException | None = None
        self._buffer_size = buffer_size
        self._rows: dict[str, list[list[object]]] = {
            "priority_metrics.csv": [],
            "full_metrics.csv": [],
            "match_results.csv": [],
            **{name: [] for name in _SUMMARY_TABLES},
        }
        self._headers: dict[str, tuple[str, ...]] = {
            "priority_metrics.csv": (*IDENTITY_COLUMNS, *PRIORITY_METRIC_NAMES),
            "full_metrics.csv": (*IDENTITY_COLUMNS, *FULL_METRIC_NAMES),
            "match_results.csv": MATCH_COLUMNS,
        }
        self._pending: list[tuple[str, int]] = []
        self._collector: ReplayCollector | None = None
        self._replay_ids: dict[str, int] = {}
        self._pending_replays: dict[str, dict[str, object]] = {}
        self._recording_provenance: dict[str, object] | None = None
        self._lock = -1
        if resume_from is None:
            root = Path(cast(str | Path, output_dir))
            if (root / "run_details.json").exists():
                raise ValueError(
                    "output_dir names an existing run; use resume_from instead"
                )
            root.mkdir(parents=True, exist_ok=True)
            run_id = (
                datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:12]
            )
            self.run_dir = root / run_id
            self.run_dir.mkdir()
            self._details: dict[str, Any] = {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "metric_schema_id": METRIC_SCHEMA_ID,
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "actor_input_projections": _INPUT_PROJECTIONS,
                "configurations": {},
                "passes": {},
                "tables": {},
                "details": {} if details is None else _json_value(details),
            }
        else:
            self.run_dir = Path(resume_from)
        try:
            self._lock = os.open(self.run_dir, os.O_RDONLY | os.O_DIRECTORY)
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if resume_from is not None:
                # Read after acquiring ownership: the previous writer may have
                # committed another flush immediately before releasing its lock.
                self._details = json.loads(
                    (self.run_dir / "run_details.json").read_bytes()
                )
                if (
                    self._details.get("schema_version") != 1
                    or self._details.get("metric_schema_id") != METRIC_SCHEMA_ID
                    or self._details.get("metric_schema_version")
                    != METRIC_SCHEMA_VERSION
                ):
                    raise ValueError(
                        "run schema does not match this writer "
                        "(recorded metric version "
                        f"{self._details.get('metric_schema_version')}, "
                        f"required {METRIC_SCHEMA_VERSION}); start a new run. "
                        "Existing files have not been changed."
                    )
                if self._details.get("actor_input_projections") != _INPUT_PROJECTIONS:
                    raise ValueError(
                        "run actor input contract differs from this writer; "
                        "start a new run. Existing files have not been changed."
                    )
                self._recover_tables()
            self.run_id = str(self._details["run_id"])
            self._completed = {
                (key, int(episode_id))
                for key, value in self._details["passes"].items()
                for episode_id in value["completed_episode_ids"]
            }
            self.start_pass(
                phase=phase,
                pass_id=pass_id,
                policies=policies,
                checkpoint_id=checkpoint_id,
                details=details,
            )
            self.flush()
        except BaseException:
            self._release()
            raise

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        if self._failed is not None:
            raise RuntimeError(
                "writer failed; close it and resume the durable run"
            ) from self._failed

    def _recover_tables(self) -> None:
        for filename in self._rows:
            path = self._table_path(filename)
            expected = self._details["tables"].get(filename, {}).get("durable_bytes", 0)
            if not path.exists():
                if expected:
                    raise ValueError(f"durable run table is missing: {filename}")
                continue
            if path.stat().st_size < expected:
                raise ValueError(f"durable run table is truncated: {filename}")
            with path.open("r+b") as stream:
                stream.truncate(expected)
                stream.flush()
                os.fsync(stream.fileno())
        for entry in self._details["passes"].values():
            for record in entry.get("replays", {}).values():
                relative = Path(record["path"])
                if len(relative.parts) != 2 or relative.parts[0] != "replays":
                    raise ValueError(
                        "recorded replay path is outside the run replay directory"
                    )
                path = self.run_dir / relative
                if path.parent.is_symlink() or path.is_symlink() or not path.is_file():
                    raise ValueError(
                        f"durable replay is missing or not a regular file: {relative}"
                    )
                if path.stat().st_size != record["bytes"]:
                    raise ValueError(
                        f"durable replay size differs from its boundary: {relative}"
                    )

    def _table_path(self, filename: str) -> Path:
        path = self.run_dir / filename
        if path.is_symlink():
            raise ValueError(f"run table must not be a symbolic link: {filename}")
        return path

    def start_pass(
        self,
        *,
        phase: str,
        pass_id: str,
        policies: dict[str, object] | None = None,
        checkpoint_id: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Append a clearly named pass, preserving its policy/configuration identity."""
        self._check_open()
        if not phase or not pass_id:
            raise ValueError("phase and pass_id must be nonempty")
        if self._pending or self._pending_replays:
            self.flush()
        if self._collector is not None:
            if self._collector.pending_episode_ids:
                raise ValueError(
                    "cannot change pass while selected replays are incomplete"
                )
            self._collector.close()
            self._collector = None
        key = json.dumps((phase, pass_id), separators=(",", ":"))
        identity: dict[str, object] = {
            "phase": phase,
            "pass_id": pass_id,
            "policies": {} if policies is None else _json_value(policies),
            "checkpoint_id": checkpoint_id,
            "details": {} if details is None else _json_value(details),
            "policy_state": "evolving"
            if phase == "training" and checkpoint_id is None
            else "frozen",
        }
        previous = self._details["passes"].get(key)
        execution_fields = {"runtime_provenance", "num_envs", "chunk_size"}

        def stable_details(value: object) -> dict[str, object]:
            return {
                name: item
                for name, item in cast(dict[str, object], value).items()
                if name not in execution_fields
            }

        if previous is not None:
            if any(
                (stable_details(previous.get(name, {})) != stable_details(value))
                if name == "details"
                else previous.get(name) != value
                for name, value in identity.items()
            ):
                raise ValueError("pass identity differs from the recorded run")
            previous["details"] = identity["details"]
        else:
            self._details["passes"][key] = {
                **identity,
                "completed_episode_ids": [],
                "episodes": {},
                "replays": {},
            }
        self._pass_key = key

    def register_episodes(self, schedule: Iterable[dict[str, object]]) -> None:
        """Record planned episode, map and random-stream identities before execution."""
        self._check_open()
        episodes = self._details["passes"][self._pass_key].setdefault("episodes", {})
        for specification in schedule:
            record = cast(dict[str, object], _json_value(specification))
            episode_id = record.get("episode_id")
            if (
                isinstance(episode_id, bool)
                or not isinstance(episode_id, int)
                or episode_id < 1
            ):
                raise ValueError("scheduled episode_id must be a positive integer")
            key = str(episode_id)
            if key in episodes and episodes[key] != record:
                raise ValueError(
                    f"episode {episode_id} differs from the recorded schedule"
                )
            episodes[key] = record
        self.flush()

    @property
    def completed_episode_ids(self) -> frozenset[int]:
        """Episode IDs committed by flush for the currently selected pass."""
        return frozenset(
            self._details["passes"][self._pass_key]["completed_episode_ids"]
        )

    @property
    def paths(self) -> dict[str, Path]:
        """Only files actually produced by the run, with direct descriptive names."""
        return {
            "run_details": self.run_dir / "run_details.json",
            **{
                Path(name).stem: self.run_dir / name for name in self._details["tables"]
            },
            **(
                {"replays": self.run_dir / "replays"}
                if any(
                    entry.get("replays") for entry in self._details["passes"].values()
                )
                else {}
            ),
        }

    def _replay_context(
        self, packet: ReplayPackets
    ) -> tuple[EvaluationEpisodeContextV3, RuntimeProvenanceV1]:
        from marl_battlegrounds.evaluation.recording_context import (
            build_recording_context,
            capture_recording_provenance,
        )

        entry = self._details["passes"][self._pass_key]
        details = entry["details"]
        if "runtime_provenance" not in details:
            if self._recording_provenance is None:
                self._recording_provenance = capture_recording_provenance()
            # Discovery is capture provenance, not a caller-declared pass
            # identity. Keep resume_from usable with the original arguments.
            entry["recording_provenance"] = self._recording_provenance
            details = {**details, **self._recording_provenance}
        episode_id = int(packet.episode_id)
        episode: dict[str, object] = {
            "episode_id": episode_id,
            **entry["episodes"].get(str(episode_id), {}),
        }
        context, runtime = build_recording_context(
            packet.config,
            run_id=self.run_id,
            phase=entry["phase"],
            pass_id=entry["pass_id"],
            episode=episode,
            policies=cast(
                dict[str, object], episode.get("policies", entry["policies"])
            ),
            details=details,
        )
        self._replay_ids[context.identity.episode_id] = episode_id
        return context, runtime

    def write_replay(self, packets: ReplayPackets) -> None:
        """Spool all selected transitions and publish complete single-file replays."""
        self._check_open()
        from marl_battlegrounds.evaluation.replay_io import (
            PreparedReplay,
            generated_replay_filename,
            preflight_replay_destination,
            publish_prepared_replay,
        )
        from marl_battlegrounds.evaluation.replay_recording import ReplayCollector

        try:
            if self._collector is None:
                self._collector = ReplayCollector(
                    self._replay_context, spool_dir=self.run_dir / ".replay_spool"
                )
            for replay in self._collector.write(packets):
                episode_id = self._replay_ids[replay.header.context.identity.episode_id]
                if (self._pass_key, episode_id) in self._completed:
                    raise ValueError(
                        f"replay episode {episode_id} was already completed"
                    )
                directory = self.run_dir / "replays"
                directory.mkdir(exist_ok=True)
                path = directory / generated_replay_filename(
                    replay.header.context,
                    replay.canonical_digest_sha256,
                    episode_id=episode_id,
                )
                if path.is_symlink():
                    raise ValueError("replay destination must not be a symbolic link")
                if path.exists():
                    from marl_battlegrounds.evaluation.models import (
                        canonical_json_bytes,
                    )

                    if path.read_bytes() != canonical_json_bytes(replay):
                        raise ValueError(
                            "existing replay differs from its content identity"
                        )
                else:
                    publish_prepared_replay(
                        PreparedReplay._from_capture(replay),  # pyright: ignore[reportPrivateUsage]
                        preflight_replay_destination(path),
                    )
                self._pending_replays[str(episode_id)] = {
                    "path": str(path.relative_to(self.run_dir)),
                    "canonical_digest_sha256": replay.canonical_digest_sha256,
                    "bytes": path.stat().st_size,
                }
        except BaseException as error:
            self._record_failure(error)
            raise

    def write(self, infos: EpisodeInfo) -> None:
        """Collect all completed records from arbitrary leading rollout dimensions."""
        self._check_open()
        import jax

        if infos.replay is not None:
            self.write_replay(infos.replay)
        host = jax.device_get(infos._replace(replay=None))
        completion = np.asarray(host.completed)
        leading = completion.ndim

        def flatten(value: object) -> np.ndarray[Any, Any]:
            array = np.asarray(value)
            return array.reshape((-1, *array.shape[leading:]))

        records = jax.tree.map(flatten, host)
        try:
            for index in np.flatnonzero(completion):
                self._write_record(records, int(index))
                if len(self._pending) >= self._buffer_size:
                    self.flush()
        except BaseException as error:
            self._record_failure(error)
            raise

    def _write_record(self, records: EpisodeInfo, index: int) -> None:
        import jax

        episode_id = int(records.episode_id[index])
        key = (self._pass_key, episode_id)
        if episode_id < 1:
            raise ValueError("completed episode ID must be positive")
        if key in self._completed:
            raise ValueError(f"episode {episode_id} was already written in this pass")
        if (
            self._collector is not None
            and episode_id in self._collector.pending_episode_ids
        ):
            raise ValueError(
                f"episode {episode_id} completed before its replay packets"
            )

        def select(value: object) -> object:
            return np.asarray(value)[index]

        config_id, config = configuration_identity(jax.tree.map(select, records.config))
        self._details["configurations"].setdefault(config_id, config)
        entry = self._details["passes"][self._pass_key]
        policies = entry["policies"]
        episode = entry["episodes"].get(str(episode_id), {})

        def policy_name(team: str) -> object:
            definition: object = episode.get("policies", policies).get(team, "")
            return (
                cast(dict[str, object], definition).get("name", "")
                if isinstance(definition, dict)
                else definition
            )

        identity: list[object] = [
            self.run_id,
            entry["phase"],
            entry["pass_id"],
            episode_id,
            episode.get("seed_id", ""),
            episode.get("map_id", ""),
            config_id,
            policy_name("team_a"),
            policy_name("team_b"),
            entry["checkpoint_id"] or "",
        ]
        for slot in range(10):
            identity.extend(
                (
                    int(records.class_ids[index, slot]),
                    int(records.active_mask[index, slot]),
                )
            )
        tournament = entry["phase"] == "tournament"
        if tournament:
            if "tournament_summary" in self._details:
                raise ValueError(
                    "a qualified tournament cannot accept additional matches"
                )
            block_id = episode.get("block_id")
            if (
                isinstance(block_id, bool)
                or not isinstance(block_id, int)
                or block_id < 1
            ):
                raise ValueError(
                    "tournament episodes require a registered positive block_id"
                )
            outcome = int(records.outcome[index])
            if outcome not in (1, 2, 3):
                raise ValueError("tournament outcome must be terminal code 1, 2 or 3")
            cells: list[object] = [""] * len(PRIORITY_METRIC_NAMES)
            if records.priority is not None:
                cells = self._metric_cells(
                    records.priority, index, PRIORITY_METRIC_NAMES
                )
            self._rows["match_results.csv"].append(
                [
                    *identity,
                    block_id,
                    episode.get("bootstrap_group") or "",
                    outcome,
                    *cells,
                ]
            )
        for filename, values, names in (
            ("priority_metrics.csv", records.priority, PRIORITY_METRIC_NAMES),
            ("full_metrics.csv", records.full, FULL_METRIC_NAMES),
        ):
            if tournament and filename == "priority_metrics.csv":
                continue
            if values is None:
                continue
            available = np.asarray(values.valid[index])
            cells = self._metric_cells(values, index, names)
            if not available.any():
                continue
            self._rows[filename].append([*identity, *cells])
        self._completed.add(key)
        self._pending.append(key)

    @staticmethod
    def _metric_cells(
        values: object, index: int, names: tuple[str, ...]
    ) -> list[object]:
        from marl_battlegrounds.evaluation.episode_metrics import MetricValues

        metric = cast(MetricValues, values)
        valid = np.asarray(metric.valid[index])
        numeric = np.asarray(metric.values[index])
        if numeric.shape != (len(names),) or valid.shape != numeric.shape:
            raise ValueError("measurements do not match the scalar column schema")
        if not np.isfinite(numeric[valid]).all():
            raise ValueError("available measurements contain a nonfinite value")
        cells = numeric.astype(object)
        cells[~valid] = ""
        return cast(list[object], cells.tolist())

    def write_tournament_results(self, statistics: TournamentStatistics) -> None:
        """Publish one qualified population idempotently with normal durability."""
        self._check_open()
        payload = {
            "tournament_results": statistics.tournament_results,
            "matchup_results": statistics.matchup_results,
            "map_results": statistics.map_results,
            "metadata": statistics.metadata,
        }
        digest = sha256(_json_bytes(_json_value(payload))).hexdigest()
        previous = self._details.get("tournament_summary")
        if previous is not None:
            if previous["digest"] != digest:
                raise ValueError(
                    "tournament summary differs from the qualified population"
                )
            return
        tables: dict[str, tuple[tuple[str, ...], list[list[object]]]] = {}
        for filename in _SUMMARY_TABLES:
            rows = cast(tuple[dict[str, object], ...], payload[Path(filename).stem])
            if not rows:
                raise ValueError("qualified tournament summary tables must be nonempty")
            names = tuple(rows[0])
            if any(set(row) != set(names) for row in rows):
                raise ValueError("tournament summary rows must have the same columns")
            tables[filename] = names, [[row[name] for name in names] for row in rows]
        self.flush()
        for filename, (names, rows) in tables.items():
            self._headers[filename] = names
            self._rows[filename] = rows
        self._details["tournament_summary"] = {
            "digest": digest,
            "metadata": _json_value(statistics.metadata),
        }
        self.flush()

    def flush(self) -> None:
        """Durably commit buffered rows and their completion boundary together."""
        self._check_open()
        # Flush mutates only publication boundaries; frozen schedules/configs can
        # stay shared during this synchronous operation.
        candidate = {
            **self._details,
            "tables": self._details["tables"].copy(),
            "passes": {
                key: {
                    **entry,
                    "completed_episode_ids": entry["completed_episode_ids"].copy(),
                    "replays": entry["replays"].copy(),
                }
                for key, entry in self._details["passes"].items()
            },
        }
        try:
            for filename, rows in self._rows.items():
                if not rows:
                    continue
                previous = candidate["tables"].get(
                    filename, {"durable_bytes": 0, "rows": 0}
                )
                text = io.StringIO(newline="")
                writer = csv.writer(text, lineterminator="\n")
                if previous["durable_bytes"] == 0:
                    writer.writerow(self._headers[filename])
                writer.writerows(rows)
                payload = text.getvalue().encode("utf-8")
                path = self._table_path(filename)
                with path.open("ab") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                    candidate["tables"][filename] = {
                        "durable_bytes": stream.tell(),
                        "rows": previous["rows"] + len(rows),
                    }
            for key, episode_id in self._pending:
                candidate["passes"][key]["completed_episode_ids"].append(episode_id)
            candidate["passes"][self._pass_key]["replays"].update(self._pending_replays)
            _atomic_json(self.run_dir / "run_details.json", candidate)
            os.fsync(self._lock)
        except BaseException as error:
            self._record_failure(error)
            raise
        self._details = candidate
        self._pending.clear()
        self._pending_replays.clear()
        for rows in self._rows.values():
            rows.clear()

    def record_failure(self, error: BaseException) -> None:
        """Preserve completed rows and record an escaping execution failure.

        Recording is best effort: a storage error must not replace the original
        provider, cancellation or fitting exception. Close and resume this run
        before continuing after failure.
        """
        if self._failed is error:
            return
        if not self._closed and self._failed is None:
            try:
                self.flush()
            except BaseException as storage_error:
                error.add_note(
                    "Could not preserve buffered results: "
                    f"{type(storage_error).__name__}: {storage_error}"
                )
        self._record_failure(error)

    def _record_failure(self, error: BaseException) -> None:
        if self._failed is not error:
            self._failed = error
            error.add_note(f"Run directory: {self.run_dir}")
            try:
                with (self.run_dir / "failures.jsonl").open("ab") as stream:
                    stream.write(
                        _json_bytes(
                            {
                                "type": type(error).__name__,
                                "message": str(error),
                                "pass": self._pass_key,
                                "notes": getattr(error, "__notes__", []),
                            }
                        )
                    )
                    stream.flush()
            except OSError:
                pass  # The original failure is raised even if recording also fails.

    def _release(self) -> None:
        if self._lock >= 0:
            os.close(self._lock)
            self._lock = -1

    def close(self) -> None:
        """Flush a healthy writer and always release its run lock."""
        if self._closed:
            return
        failure: BaseException | None = None
        try:
            if self._failed is None:
                self.flush()
        except BaseException as error:
            failure = error
        self._closed = True
        cleanup = [self._release]
        if self._collector is not None:
            cleanup.insert(0, self._collector.close)
        for finish in cleanup:
            try:
                finish()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(
                        "Additional run-writer cleanup failure: "
                        f"{type(error).__name__}: {error}"
                    )
        if failure is not None:
            self._record_failure(failure)
            raise failure

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        if exception is not None:
            self.record_failure(exception)
        try:
            self.close()
        except BaseException as cleanup_error:
            if exception is None:
                raise
            exception.add_note(
                "Could not close run writer: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
