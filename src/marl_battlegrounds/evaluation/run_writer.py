"""Write buffered episode tables and selected replays with explicit resume.

RunWriter owns one locked run directory. A flush records table byte lengths,
replay files and completed IDs together; resume restores that durable boundary
and discards an uncommitted table suffix. All work is on the host. Numerical
game code calls no writer; callers supply completed payloads at their boundary.
"""

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
    """Convert named tuples and NumPy containers to stable JSON-compatible values."""
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
    """Encode sorted compact UTF-8 JSON with a trailing newline; reject NaN/Infinity."""
    return (
        json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_json(path: Path, value: object) -> None:
    """Write and fsync temporary JSON, then replace the destination in one rename."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def configuration_identity(config: EnvConfig) -> tuple[str, dict[str, object]]:
    """Identify one numerical episode config by its recorded content.

    Parameters
    ----------
    config : EnvConfig
        One environment-normalized scalar EnvConfig tree. Callers should
        normalize scalar/array dtypes before comparing configuration identity.

    Returns
    -------
    tuple[str, dict[str, object]]
        (sha256_hex, content_dict). The digest covers sorted compact JSON plus a
        newline; content contains host values suitable for run_details.json.

    Raises
    ------
    ValueError
        JSON would contain NaN or Infinity.
    TypeError
        A leaf cannot be serialized.

    Notes
    -----
        Host-only: device_get may synchronize and copy numerical leaves. This
        function does not validate physical rules, infer a map or edit the config.
    """
    import jax

    content = cast(dict[str, object], _json_value(jax.device_get(config)))
    return sha256(_json_bytes(content)).hexdigest(), content


class RunWriter:
    """Own one durable run directory for episode tables and selected replays.

    Parameters
    ----------
    output_dir : str or pathlib.Path, optional
        Parent directory for a new uniquely named run. Supply this or resume_from.
    resume_from : str or pathlib.Path, optional
        Existing run directory to recover. Its metric/input schema must match.
    phase : str, default "evaluation"
        Nonempty pass label. "tournament" writes match rows and requires registered
        pairing metadata; it does not change simulator rules.
    pass_id : str, default "1"
        Nonempty identifier within phase. Reopening it must preserve saved identity.
    policies : dict, optional
        Team policy descriptions for the pass. Defaults to an empty description.
    checkpoint_id : str, optional
        Caller-supplied checkpoint identity; absent when unknown.
    details : dict, optional
        JSON-compatible pass metadata. Source discovery is added only when capture
        needs it. Resume allows runtime_provenance, num_envs and chunk_size to vary.
    buffer_size : int, default 128
        Positive completed-row count that triggers automatic flush; bool is invalid.

    Attributes
    ----------
    run_id : str
        Generated new-run identity or the identity read from the resumed directory.
    run_dir : pathlib.Path
        Directory owned and locked by this writer until close.
    completed_episode_ids : frozenset of int
        Durable completions for the current pass.
    paths : dict of str to pathlib.Path
        Only output files/directories already produced by this run.

    Raises
    ------
    ValueError
        Output/resume choice, labels, schema, pass identity or durable files are
        invalid. An existing run must be resumed explicitly.
    OSError
        The run cannot be created, accessed or exclusively locked.

    Notes
    -----
    Host-only; opening may create files or truncate uncommitted table suffixes.
    Use as a context manager or call close. A healthy close flushes; a failed writer
    must be closed and resumed before more writes. Call start_pass to append a
    distinct pass. This class does not allocate episode IDs or execute games.
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
        """Open and lock a new or resumed run, validate its schema and start the first
        pass.
        """
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
        """Reject operations after close or a recorded failure; recovery needs a new
        writer.
        """
        if self._closed:
            raise RuntimeError("writer is closed")
        if self._failed is not None:
            raise RuntimeError(
                "writer failed; close it and resume the durable run"
            ) from self._failed

    def _recover_tables(self) -> None:
        """Restore recorded durable table lengths and verify durable replay files.

        The run lock must already be held. Missing/truncated committed content or
        unsafe replay paths fail recovery; extra table suffix bytes are truncated.
        """
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
        """Resolve a run table path and reject a symbolic-link destination."""
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
        """Start or resume a named pass within this writer.

        Parameters
        ----------
        phase : str
            Nonempty phase label; the writer uses "tournament" for match rows.
        pass_id : str
            Nonempty identity within phase.
        policies : dict[str, object] | None
            Optional JSON-compatible team descriptions, default empty.
        checkpoint_id : str | None
            Optional caller-known checkpoint label, default None.
        details : dict[str, object] | None
            Optional pass metadata, default empty. Runtime provenance, batch
            size and chunk size may change when resuming; stable details must match.

        Returns
        -------
        None
            None. Pending rows are flushed and this writer selects the requested pass.

        Raises
        ------
        RuntimeError
            The writer is closed or failed.
        ValueError
            Labels are empty, selected replays are incomplete, or an
            existing pass's policy/checkpoint/stable metadata identity differs.

        Notes
        -----
            Host-only and mutating. A training pass without a checkpoint is labeled
            evolving; other pass identities are labeled frozen. This describes the
            caller's declared run and does not perform training or freeze model values.
        """
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
            """Exclude only execution placement/chunk details when comparing pass
            identity.
            """
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
        """Record the schedule metadata needed to interpret upcoming completions.

        Parameters
        ----------
        schedule : Iterable[dict[str, object]]
            Iterable of JSON-compatible dicts, each with a positive integer
            episode_id. Optional seeds, map identity and pairing metadata are kept.

        Returns
        -------
        None
            None. Records are merged into the active pass and flushed durably.

        Raises
        ------
        ValueError
            An ID is invalid or an existing ID's metadata differs.
        RuntimeError
            The writer is closed or failed.
        OSError
            Schedule publication fails.

        Notes
        -----
            Repeating identical records is allowed. This does not execute episodes,
            mark them complete or validate a promised map against future game state.
        """
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
        """Return the current pass's durable episode IDs, excluding buffered
        completions.
        """
        return frozenset(
            self._details["passes"][self._pass_key]["completed_episode_ids"]
        )

    @property
    def paths(self) -> dict[str, Path]:
        """Return fresh paths for run_details and only tables/replays already
        produced.
        """
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
        """Join a first packet to registered episode/pass facts and cached
        provenance.
        """
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
        """Collect selected transitions and publish complete replay files.

        Parameters
        ----------
        packets : ReplayPackets
            ReplayPackets for a step or chunk. Leading validity axes select
            real packet rows; transition indices must be contiguous per episode.

        Returns
        -------
        None
            None. Incomplete episodes remain in owned temporary spools. Complete
            artifacts are published below replays and queued for the next durable flush.

        Raises
        ------
        RuntimeError
            The writer or collector is closed/failed.
        ValueError
            Packet order/identity is invalid, the episode was completed,
            or an existing destination conflicts with the replay's content.
        OSError
            Temporary or replay files cannot be written.

        Notes
        -----
            Host-only: may transfer packets, discover recording provenance and create
            files. It does not mark the episode complete by itself. Supply matching
            completed EpisodeInfo through write after all selected packets arrive.
            Any collection/publication failure marks this writer failed.
        """
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
        """Buffer all completed episodes in one step or collected chunk.

        Parameters
        ----------
        infos : EpisodeInfo
            EpisodeInfo with scalar, batch or time/batch leading dimensions.
            completed admits rows. Config and selected metric leaves must share
            those dimensions; optional replay packets are collected first.

        Returns
        -------
        None
            None. Selected valid metric rows are buffered; reaching buffer_size
            flushes automatically. Explicit flush or healthy close makes the remainder
            durable, including completions that have no metric row.

        Raises
        ------
        RuntimeError
            The writer is closed or failed.
        ValueError
            A completion ID is invalid/repeated, its replay is incomplete,
            metric columns are malformed/nonfinite, or tournament metadata is invalid.
        OSError
            A triggered flush or replay write fails.

        Notes
        -----
            Transfers numerical data to the host. False completion rows are ignored.
            Unavailable metric cells are blank, not measured zeros. The writer records
            actual supplied config/slot identity; it does not reset games or generate
            IDs.
            Call once for each completion; no implicit duplicate suppression occurs.
        """
        self._check_open()
        import jax

        if infos.replay is not None:
            self.write_replay(infos.replay)
        host = jax.device_get(infos._replace(replay=None))
        completion = np.asarray(host.completed)
        leading = completion.ndim

        def flatten(value: object) -> np.ndarray[Any, Any]:
            """Flatten completion-leading axes while retaining each field's payload
            shape.
            """
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
        """Validate one completion and append its selected rows to the active pass
        buffers.
        """
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
            """Read one host record's scalar configuration leaf."""
            return np.asarray(value)[index]

        config_id, config = configuration_identity(jax.tree.map(select, records.config))
        self._details["configurations"].setdefault(config_id, config)
        entry = self._details["passes"][self._pass_key]
        policies = entry["policies"]
        episode = entry["episodes"].get(str(episode_id), {})

        def policy_name(team: str) -> object:
            """Use per-episode team metadata when present, otherwise the pass
            description.
            """
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
        """Validate metric width/finiteness and replace unavailable values with empty
        cells.
        """
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
        """Publish a qualified tournament's summary tables exactly once by content.

        Parameters
        ----------
        statistics : TournamentStatistics
            TournamentStatistics from the statistical qualification helper,
            containing nonempty population, matchup and map result tables.

        Returns
        -------
        None
            None. Flushes preceding match data, writes all three summaries and records
            their common content digest. Repeating identical statistics is a no-op.

        Raises
        ------
        ValueError
            Tables are empty/inconsistent or differ from a saved summary.
        RuntimeError
            The writer is closed or failed.
        OSError
            Summary publication fails.

        Notes
        -----
            Host-only. The caller must qualify the complete match population first;
            this writer checks table structure and content identity, not the rating
            model. Once published, additional tournament match rows are rejected.
        """
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
        """Make buffered rows, replay references and completion IDs durable together.

        Returns
        -------
        None
            None. Appends/fsyncs tables, atomically replaces run_details.json, fsyncs
            the directory, then clears successful buffers.

        Raises
        ------
        RuntimeError
            The writer is closed or failed.
        OSError
            A table, metadata or directory operation fails.

        Notes
        -----
            Host-only and synchronous. Publication failures mark the writer failed.
            The previous metadata boundary remains the recovery authority; resume
            removes table suffixes not covered by that boundary.
        """
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
        """Try to preserve completed rows and record an escaping execution failure.

        Parameters
        ----------
        error : BaseException
            Original provider, execution, cancellation or storage exception.

        Returns
        -------
        None
            None. Marks the writer failed and adds run/recovery context to error.
            Repeating the same recorded error does not duplicate this work.

        Notes
        -----
            Best effort: a storage error is added as a note rather than replacing the
            original exception. A still-healthy writer first attempts to flush buffered
            completions. Close and explicitly resume before continuing after failure.
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
        """Mark failure and append diagnostics without hiding the original error."""
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
        """Close the owned directory descriptor and release its advisory lock."""
        if self._lock >= 0:
            os.close(self._lock)
            self._lock = -1

    def close(self) -> None:
        """Flush a healthy writer and release every owned spool and run lock.

        Returns
        -------
        None
            None. Repeated calls are safe. The writer remains closed even if cleanup
            fails; already durable artifacts stay on disk.

        Raises
        ------
        OSError
            A flush or resource release fails.
        RuntimeError
            A required writer operation fails during cleanup.

        Notes
        -----
            A writer already marked failed skips normal flush. Cleanup attempts all
            resources, preserving the first error and noting later errors. Incomplete
            replay spools do not become completed replay evidence.
        """
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
        """Return this owned writer for a with block."""
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        """Record any body failure, close resources and preserve the original
        exception.
        """
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
