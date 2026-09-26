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
import math
import os
from collections import ChainMap
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
from functools import cache, partial
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, NamedTuple, cast
from uuid import uuid4

import numpy as np

from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_V4,
    SHARED_OBS_ACTOR_PROJECTION_V3,
)
from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
    METRIC_SCHEMA_ID,
    METRIC_SCHEMA_VERSION,
    PRIORITY_METRIC_NAMES,
    PRIORITY_OUTCOME_CODES,
)

if TYPE_CHECKING:
    from jax import Array

    from marl_battlegrounds.core.types import EnvConfig
    from marl_battlegrounds.environment import EpisodeInfo
    from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV4
    from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
    from marl_battlegrounds.evaluation.replay_capture import ReplayPackets
    from marl_battlegrounds.evaluation.replay_recording import ReplayCollector
    from marl_battlegrounds.evaluation.tournament_records import TournamentRecords
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

RUN_SCHEMA_VERSION = 2
EPISODE_COLUMNS = (
    *IDENTITY_COLUMNS,
    "outcome",
    "episode_length",
    "team_a_score",
    "team_b_score",
)
ASSIGNMENT_COLUMNS = (
    "run_id",
    "phase",
    "pass_id",
    "episode_id",
    "team",
    "global_slot",
    "system_id",
    "policy_id",
    "decision_start",
    "decision_stop",
)

MATCH_COLUMNS = (
    *IDENTITY_COLUMNS,
    "block_id",
    "bootstrap_group",
    "outcome",
    *PRIORITY_METRIC_NAMES,
)
_SUMMARY_TABLES = ("tournament_results.csv", "matchup_results.csv", "map_results.csv")
# New runs record the Red Zone projections (context V4). A run recorded with the
# older projections cannot be resumed by this writer; its files stay readable.
_INPUT_PROJECTIONS = {
    "shared_obs": SHARED_OBS_ACTOR_PROJECTION_V3.model_dump(mode="json"),
    "no_shared_obs": NO_SHARED_OBS_ACTOR_PROJECTION_V4.model_dump(mode="json"),
}
_VERIFICATION_CACHE_SIZE = 256
# The EnvConfig field added with Red Zone scoring; historical identities omit it.
_RED_ZONE_DEPTH_FIELD = "team_deathmatch_red_zone_depth"

type _ConfigValidationKey = tuple[str, object, tuple[tuple[str, tuple[int, ...]], ...]]


class _PreparedReplay(NamedTuple):
    """Hold validated replay inputs until the whole write call can be applied.

    packets are host rows. Contexts and config IDs belong only to new initial
    packets. provenance is newly discovered metadata, or None when not needed.
    Preparing this record creates no live spool and changes no writer metadata.
    """

    packets: ReplayPackets
    contexts: dict[int, tuple[EvaluationEpisodeContextV4, RuntimeProvenanceV1]]
    config_ids: dict[int, str]
    provenance: dict[str, object] | None


class _RecordingRows(NamedTuple):
    """View one compact host family through shared recording validators.

    IDs, action indices and lengths have one leading row axis. config_indices
    selects actual configs from the shared evidence table. Only completion
    families supply outcomes, scores and metric tables. Metric tables keep their
    own row count; completion references select their rows. This view never
    expands full metrics beside traces or starts.
    """

    episode_id: Any
    decision_step: Any
    episode_length: Any
    config: Any
    config_indices: Any
    outcome: Any = None
    team_scores: Any = None
    priority: Any = None
    full: Any = None
    episode_start_records: Any = None


def _host_row(value: object, *, index: int) -> np.ndarray[Any, Any]:
    """Select one host record row without changing its dtype or remaining axes."""
    return np.asarray(value)[index]


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


def _recording_table_headers() -> dict[str, tuple[str, ...]]:
    """Return the fixed CSV headers shared by writers and recording preflight.

    Each call returns a new mapping. Summary tables get their headers only when
    a tournament summary is written; a training checkpoint keeps them empty.
    This reads no files and creates no writer.
    """
    return {
        "episodes.csv": EPISODE_COLUMNS,
        "policy_assignments.csv": ASSIGNMENT_COLUMNS,
        "priority_metrics.csv": (*IDENTITY_COLUMNS, *PRIORITY_METRIC_NAMES),
        "full_metrics.csv": (*IDENTITY_COLUMNS, *FULL_METRIC_NAMES),
        "match_results.csv": MATCH_COLUMNS,
    }


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


def _validation_key(config: EnvConfig, identifier: str) -> _ConfigValidationKey:
    """Key an already hashed host config by content, exact tree, dtypes and shapes.

    Content identity alone omits numerical dtypes. The additional signature
    prevents a different dtype or malformed record type borrowing an earlier
    successful Core validation. No arrays are retained by this key.
    """
    import jax

    return (
        identifier,
        cast(object, jax.tree.structure(config)),
        tuple(
            (np.asarray(leaf).dtype.str, np.asarray(leaf).shape)
            for leaf in jax.tree.leaves(config)
        ),
    )


def _roster_declaration(start: Mapping[str, object]) -> tuple[int, ...] | None:
    """Normalize the optional saved roster without inventing a historical claim.

    Missing source_class_ids returns None. A present field must be a JSON list
    of ten plain integers, with classes 1..5 before zero padding in each team.
    Reject null, bool, serialized sentinel rows and malformed compact rows.
    This host wire check performs no catalog lookup or configuration work.
    """
    if "source_class_ids" not in start:
        return None
    value = start["source_class_ids"]
    if not isinstance(value, list):
        raise ValueError("source_class_ids must contain ten integer classes in 0..5")
    entries = cast(list[object], value)
    if len(entries) != 10 or any(
        type(item) is not int or not 0 <= item <= 5 for item in entries
    ):
        raise ValueError("source_class_ids must contain ten integer classes in 0..5")
    classes = tuple(cast(list[int], value))
    for team in (classes[:5], classes[5:]):
        count = sum(item != 0 for item in team)
        if any(item == 0 for item in team[:count]):
            raise ValueError("source_class_ids must use compact team prefixes")
    if not start.get("source_known"):
        raise ValueError("unknown starts cannot declare source_class_ids")
    return classes


def _same_start(previous: Mapping[str, object], current: Mapping[str, object]) -> bool:
    """Compare declared start fields symmetrically, including an absent roster.

    current is a normalized declaration without verification/result fields.
    previous may also contain those later evidence fields. A roster added or
    removed after declaration conflicts even when every other field agrees.
    """
    return _roster_declaration(previous) == _roster_declaration(current) and all(
        previous.get(key) == value
        for key, value in current.items()
        if key != "source_class_ids"
    )


@cache
def _source_relationship_check() -> Callable[
    [EnvConfig, EnvConfig, Array], tuple[Any, Any]
]:
    """Compile profile reconstruction and exact source comparison together.

    Only a new verified config/source pair uses this check. A single compiled
    call avoids dispatching a separate device operation for every config leaf.
    Both configs and the int32 (10,) roster are dynamic. Ten -1 values mean the
    original source. Invalid traced rows never return a successful relationship.
    Import numerical code only when start recording needs it.
    """
    import jax

    from marl_battlegrounds.tasks import (
        _source_config_with_class_ids,  # pyright: ignore[reportPrivateUsage]
        spawn_locations_for_source,
    )

    def compare(
        actual: EnvConfig, source: EnvConfig, classes: Array
    ) -> tuple[Array, Array]:
        """Resolve one declared profile and compare every config field and pad."""
        resolved, valid = _source_config_with_class_ids(source, classes)
        matches, choice = spawn_locations_for_source(actual, resolved)
        return matches & valid, choice

    return jax.jit(compare)


def _remember_verification[Key, Value](
    cache: dict[Key, Value], key: Key, value: Value
) -> None:
    """Retain one successful immutable fact in a bounded insertion-order cache.

    Cached facts never publish a start or certify a claim. Callers still check
    each declaration and actual config content. Eviction merely repeats a check.
    """
    if key not in cache and len(cache) >= _VERIFICATION_CACHE_SIZE:
        cache.pop(next(iter(cache)))
    cache[key] = value


def configuration_identity(
    config: EnvConfig, *, historical: bool = False
) -> tuple[str, dict[str, object]]:
    """Identify one numerical episode config by its recorded content.

    Parameters
    ----------
    config : EnvConfig
        One environment-normalized scalar EnvConfig tree. Callers should
        normalize scalar/array dtypes before comparing configuration identity.
    historical : bool, default=False
        False records every EnvConfig field. True reproduces the identity a
        config had before the Team Deathmatch Red Zone field existed: the
        team_deathmatch_red_zone_depth key is left out of the content and the
        digest. Only a depth of exactly +0.0 has a historical identity.

    Returns
    -------
    tuple[str, dict[str, object]]
        (sha256_hex, content_dict). The digest covers sorted compact JSON plus a
        newline; content contains host values suitable for run_details.json.

    Raises
    ------
    ValueError
        JSON would contain NaN or Infinity, or historical is True and the
        depth is not exactly +0.0.
    TypeError
        A leaf cannot be serialized.

    Notes
    -----
        Host-only: device_get may synchronize and copy numerical leaves. This
        function does not validate physical rules, infer a map or edit the config.
    """
    import jax

    content = cast(dict[str, object], _json_value(jax.device_get(config)))
    if historical:
        depth = content.pop(_RED_ZONE_DEPTH_FIELD)
        if (
            not isinstance(depth, float)
            or depth != 0.0
            or math.copysign(1.0, depth) < 0.0
        ):
            raise ValueError(
                "a historical configuration identity needs Red Zone depth +0.0, "
                f"not {depth!r}"
            )
    return configuration_content_identity(content), content


def configuration_content_identity(content: Mapping[str, object]) -> str:
    """Hash saved configuration content exactly as configuration_identity does.

    Parameters
    ----------
    content : Mapping[str, object]
        JSON-compatible configuration content, for example the dictionary a
        run_details.json or tournament snapshot recorded. It may be current
        (13 keys) or historical (12 keys, before the Red Zone field).

    Returns
    -------
    str
        SHA-256 hex digest of sorted compact UTF-8 JSON plus a newline.

    Raises
    ------
    ValueError
        The content contains NaN or Infinity.
    TypeError
        A value is not JSON-serializable.

    Notes
    -----
    Pure host function; nothing is validated or restored. Use it to check a
    saved identity before building any array from the content.
    """
    return sha256(_json_bytes(content)).hexdigest()


def _replay_summaries(packets: ReplayPackets) -> dict[int, tuple[int, int, int, int]]:
    """Read terminal outcome, local length and scores from authoritative packet facts.

    Transfer only these small leaves when packets are still on device. This
    performs no simulator calculation and leaves unfinished episodes absent.
    """
    import jax

    valid, ended, ids, indices, outcomes, scores = jax.device_get(
        (
            packets.valid,
            packets.done.done,
            packets.episode_id,
            packets.transition_index,
            packets.info.transition_facts.team_deathmatch_facts.outcome,
            packets.state.team_deathmatch_scores,
        )
    )
    result: dict[int, tuple[int, int, int, int]] = {}
    for index in np.flatnonzero(np.asarray(valid) & np.asarray(ended)):
        score = np.asarray(scores).reshape(-1, 2)[index]
        result[int(np.asarray(ids).reshape(-1)[index])] = (
            int(np.asarray(outcomes).reshape(-1)[index]),
            int(np.asarray(indices).reshape(-1)[index]) + 1,
            int(score[0]),
            int(score[1]),
        )
    return result


class RunWriter:
    """Own one durable run directory for episode tables and selected replays.

    Parameters
    ----------
    output_dir : str or pathlib.Path, optional
        Parent directory for a new uniquely named run. Supply this or resume_from.
    resume_from : str or pathlib.Path, optional
        Existing run directory to recover. Its metric/input schema must match.
        Saved start/source relationships are checked before table recovery.
    phase : str, default "evaluation"
        Nonempty pass label. "tournament" writes match rows and requires registered
        pairing metadata; it does not change simulator rules.
    pass_id : str, default "1"
        Nonempty identifier within phase. Reopening it must preserve saved identity.
    policies : dict, optional
        Team A/B System, Policy or serialized descriptions. Defaults to empty.
        Raw registration records available identity facts; it does not freeze
        methods. Only explicit frozen-snapshot evidence establishes that status.
    recording_checkpoint : dict[str, object] or None, default None
        Explicit earlier recording boundary returned by checkpoint_recording.
        Requires resume_from and its original System registration. Validate all
        saved identities and files before rewinding table prefixes. Pair this
        with the matching full learner checkpoint; no learner data is restored
        here. Omit it for ordinary latest-manifest recovery.
    checkpoint_id : str, optional
        Caller-supplied checkpoint label; unrelated to the recording recovery
        token and absent when unknown.
    details : dict, optional
        JSON-compatible pass metadata. Source discovery is added only when capture
        needs it. Resume allows runtime_provenance, num_envs and chunk_size to vary.
    buffer_size : int, default 128
        Positive completed-row limit; bool is invalid. Optional assignment
        intervals share a capacity of 10 * buffer_size open and closed segments.
        They also flush without a completed game, preserving whole decisions.

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
    Both resume routes first check saved roster and resolved configuration
    evidence while holding the directory lock. Ordinary resume permits valid
    pending starts; an explicit recording checkpoint requires completed evidence.
    Training callers must run their content preflight before constructing this
    writer. The generic writer does not choose training content eligibility.
    Use as a context manager or call close. A healthy close flushes; a failed writer
    must be closed and resumed before more writes. Call start_pass to append a
    distinct pass. This class does not allocate episode IDs or execute games.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        resume_from: str | Path | None = None,
        recording_checkpoint: dict[str, object] | None = None,
        phase: str = "evaluation",
        pass_id: str = "1",
        policies: dict[str, object] | None = None,
        checkpoint_id: str | None = None,
        buffer_size: int = 128,
        details: dict[str, object] | None = None,
        _expected_run_id: str | None = None,
    ) -> None:
        """Open and lock a new or resumed run, validate its schema and start the first
        pass.
        """
        if recording_checkpoint is not None and resume_from is None:
            raise ValueError("recording_checkpoint requires resume_from")
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
            "episodes.csv": [],
            "policy_assignments.csv": [],
            "priority_metrics.csv": [],
            "full_metrics.csv": [],
            "match_results.csv": [],
            **{name: [] for name in _SUMMARY_TABLES},
        }
        self._headers = _recording_table_headers()
        self._pending: list[tuple[str, int]] = []
        self._collector: ReplayCollector | None = None
        self._replay_ids: dict[str, int] = {}
        self._replay_config_ids: dict[int, str] = {}
        self._preflight_contexts: dict[
            int, tuple[EvaluationEpisodeContextV4, RuntimeProvenanceV1]
        ] = {}
        self._pending_replays: dict[str, dict[str, object]] = {}
        self._recording_provenance: dict[str, object] | None = None
        self._assignment_open: dict[tuple[int, int], list[object]] = {}
        self._source_cache: list[tuple[object, tuple[object, ...], str]] = []
        self._pending_numerical_starts: set[int] = set()
        self._collection_active = False
        self._checkpoint_hashes: dict[str, Any] | None = None
        self._checkpoint_pass: str | None = None
        self._canonical_records: TournamentRecords | None = None
        self._validated_start_configs: dict[_ConfigValidationKey, None] = {}
        self._verified_source_choices: dict[
            tuple[_ConfigValidationKey, str, tuple[int, ...] | None], int
        ] = {}
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
                "schema_version": RUN_SCHEMA_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "metric_schema_id": METRIC_SCHEMA_ID,
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "actor_input_projections": _INPUT_PROJECTIONS,
                "configurations": {},
                "systems": {},
                "source_banks": {},
                "passes": {},
                "tables": {},
                "details": {} if details is None else _json_value(details),
            }
        else:
            self.run_dir = Path(resume_from)
        restoration = None
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
                    self._details.get("schema_version") != RUN_SCHEMA_VERSION
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
                if (
                    _expected_run_id is not None
                    and self._details["run_id"] != _expected_run_id
                ):
                    raise ValueError("run_id differs from the recorded run")
                if recording_checkpoint is not None:
                    from marl_battlegrounds.evaluation.recording_checkpoint import (
                        begin_restore,
                        prepare_restore,
                    )

                    restoration = prepare_restore(
                        self,
                        recording_checkpoint,
                        phase=phase,
                        pass_id=pass_id,
                        policies=policies,
                        checkpoint_id=checkpoint_id,
                        details=details,
                    )
                    begin_restore(self, restoration)
                else:
                    if "recording_restore" in self._details:
                        raise ValueError(
                            "interrupted recording restore requires its explicit "
                            "recording_checkpoint token before recovery"
                        )
                    self._validate_saved_starts(self._details, allow_pending=True)
                    self._prepare_pass(phase, pass_id, policies, checkpoint_id, details)
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
            if restoration is None:
                self.flush()
            else:
                from marl_battlegrounds.evaluation.recording_checkpoint import (
                    finish_restore,
                )

                finish_restore(self, restoration)
        except BaseException:
            if restoration is not None:
                restoration.close()
            self._release()
            raise

    @staticmethod
    def _validate_saved_starts(
        details: Mapping[str, Any], *, allow_pending: bool, pass_key: str | None = None
    ) -> None:
        """Verify saved start claims before either resume route changes output.

        details is the selected saved run manifest. pass_key selects one pass;
        None checks all passes for ordinary resume. allow_pending permits only
        unresolved declarations with no recorded transition evidence. Token
        restore passes False. Every verified/custom start needs resolved config
        evidence, even when its optional roster field is absent.

        Check identities, wire fields, source profiles and complete spawn/config
        relationships. Repeated source/roster/resolved combinations are checked
        once. This host check can compile numerical comparisons and read device
        results, but never changes details or any file. Invalid evidence raises
        ValueError or a configuration validation error before recovery starts.
        """
        from marl_battlegrounds.evaluation.evaluation_conditions import restore_config
        from marl_battlegrounds.tasks import (
            _validate_config_choices,  # pyright: ignore[reportPrivateUsage]
        )

        configurations = details.get("configurations", {})
        banks = details.get("source_banks", {})
        passes = details.get("passes", {})
        if not all(
            isinstance(value, dict) for value in (configurations, banks, passes)
        ):
            raise ValueError(
                "saved start references require configuration, bank and pass objects"
            )
        restored: dict[str, EnvConfig] = {}
        valid_configs: set[str] = set()
        checked_banks: set[str] = set()
        relationships: dict[tuple[str, str, tuple[int, ...] | None], int] = {}

        def config(identifier: object, *, actual: bool) -> EnvConfig:
            """Restore and hash one referenced config; validate actual episode data."""
            if not isinstance(identifier, str) or identifier not in configurations:
                raise ValueError("saved start references a missing configuration")
            if identifier not in restored:
                content = configurations[identifier]
                if (
                    not isinstance(content, dict)
                    or sha256(_json_bytes(cast(dict[str, object], content))).hexdigest()
                    != identifier
                ):
                    raise ValueError(
                        "saved start configuration content differs from its ID"
                    )
                restored[identifier] = restore_config(
                    cast(dict[str, Any], content), validate=False
                )
            result = restored[identifier]
            if actual and identifier not in valid_configs:
                _validate_config_choices(
                    result, batched=False, both_spawn_choices=False
                )
                valid_configs.add(identifier)
            return result

        entries = passes.values() if pass_key is None else (passes[pass_key],)
        for entry in entries:
            starts = entry.get("episode_starts", {})
            if not isinstance(starts, dict):
                raise ValueError("saved episode_starts must be an object")
            for episode_id, start in cast(dict[str, Any], starts).items():
                if not isinstance(start, dict):
                    raise ValueError("saved start must be an object")
                start = cast(dict[str, Any], start)
                identifier = RunWriter._integer(
                    start.get("episode_id"), "saved start episode_id", minimum=1
                )
                if episode_id != str(identifier):
                    raise ValueError("saved start episode ID differs from its owner")
                RunWriter._integer(
                    start.get("reset_generation"), "saved reset_generation", minimum=0
                )
                RunWriter._integer(
                    start.get("episode_start_stage"),
                    "saved episode_start_stage",
                    minimum=-1,
                )
                source_index = RunWriter._integer(
                    start.get("source_index"), "saved source_index", minimum=-1
                )
                spawn = RunWriter._integer(
                    start.get("spawn_locations"), "saved spawn_locations", minimum=-1
                )
                if spawn not in (-1, 0, 1):
                    raise ValueError("saved spawn_locations must be -1, 0 or 1")
                if any(
                    type(start.get(field)) is not bool
                    for field in ("source_known", "authored_start")
                ):
                    raise ValueError("saved start flags must be Boolean")
                roster = _roster_declaration(start)
                known = start["source_known"]
                verification = start.get("verification")
                if verification not in ("pending", "verified", "custom"):
                    raise ValueError("saved start has an invalid verification state")
                if verification != "pending" and (verification == "verified") != (
                    known and not start["authored_start"]
                ):
                    raise ValueError(
                        "saved verification state differs from its source claim"
                    )
                evidence = (
                    entry.get("trace_config_ids", {}).get(episode_id),
                    entry.get("completed_config_ids", {}).get(episode_id),
                    entry.get("replays", {}).get(episode_id, {}).get("config_id"),
                )
                resolved_id = start.get("resolved_config_id")
                if verification == "pending":
                    if (
                        not allow_pending
                        or any(value is not None for value in evidence)
                        or identifier in entry.get("completed_episode_ids", ())
                    ):
                        raise ValueError(
                            "saved start contains unresolved first-start evidence"
                        )
                elif resolved_id is None:
                    raise ValueError(
                        "verified/custom saved start requires resolved_config_id"
                    )
                if resolved_id is not None and any(
                    value is not None and value != resolved_id for value in evidence
                ):
                    raise ValueError(
                        "saved start configuration differs from recorded "
                        "episode evidence"
                    )
                actual = (
                    None if resolved_id is None else config(resolved_id, actual=True)
                )
                table_id = start.get("source_table_id")
                if not known:
                    if table_id is not None or source_index != -1 or spawn != -1:
                        raise ValueError(
                            "unknown saved starts require unknown source references"
                        )
                    continue
                if not isinstance(table_id, str):
                    raise ValueError("saved start references a missing source")
                references = banks.get(table_id)
                if not isinstance(references, list) or not 0 <= source_index < len(
                    cast(list[object], references)
                ):
                    raise ValueError("saved start references a missing source")
                references = cast(list[Any], references)
                if table_id not in checked_banks:
                    if (
                        any(
                            not isinstance(ref, str) or ref not in configurations
                            for ref in references
                        )
                        or sha256(_json_bytes(references)).hexdigest() != table_id
                    ):
                        raise ValueError(
                            "saved source bank content/order differs from its ID"
                        )
                    checked_banks.add(table_id)
                source_id = references[source_index]
                source = config(source_id, actual=False)
                if actual is None:
                    continue
                relationship = (cast(str, resolved_id), source_id, roster)
                choice = relationships.get(relationship)
                if choice is None:
                    classes = cast(
                        "Array",
                        np.asarray(
                            (-1,) * 10 if roster is None else roster, dtype=np.int32
                        ),
                    )
                    matches, found = _source_relationship_check()(
                        actual, source, classes
                    )
                    if not bool(matches):
                        raise ValueError(
                            "saved start configuration differs from its source/roster"
                        )
                    choice = int(found)
                    relationships[relationship] = choice
                if choice != spawn:
                    raise ValueError("saved start spawn choice differs from its source")

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
        headline = "tournament_headline_metrics.csv"
        if headline in self._details["tables"] or any(
            entry.get("pass_role") == "tournament_coordinator"
            for entry in self._details["passes"].values()
        ):
            # Only tournament runs own this optional role. Training checkpoint
            # bundles keep their original exact table set.
            self._rows.setdefault(headline, [])
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
            Optional team System, Policy or serialized descriptions, default
            empty. Component order and identity must remain fixed within a pass.
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
            Host-only and mutating after identity checks. Training remains evolving.
            Other calls need explicit freezing evidence; a label is not proof that
            weights stay fixed. No method is called or frozen by registration.
        """
        self._check_open()
        key, identity, systems = self._prepare_pass(
            phase, pass_id, policies, checkpoint_id, details
        )
        if self._checkpoint_pass is not None and key != self._checkpoint_pass:
            raise ValueError(
                "recording checkpoints require one dedicated training pass"
            )
        if self._collector is not None and self._collector.pending_episode_ids:
            raise ValueError("cannot change pass while selected replays are incomplete")
        if (
            self._pending
            or self._pending_replays
            or self._assignment_open
            or any(self._rows.values())
        ):
            self.flush()
        if self._collector is not None:
            self._collector.close()
            self._collector = None
        self._details["systems"].update(systems)
        previous = self._details["passes"].get(key)
        if previous is not None:
            previous["details"] = identity["details"]
        else:
            self._details["passes"][key] = {
                **identity,
                "completed_episode_ids": [],
                "episodes": {},
                "replays": {},
                "recorded_metrics_by_episode": {},
                "completed_config_ids": {},
                "episode_starts": {},
                "trace_epochs": {},
                "trace_config_ids": {},
            }
        self._details["passes"][key].setdefault("completed_config_ids", {})
        self._pass_key = key
        self._pending_numerical_starts = {
            int(identifier)
            for identifier, start in self._details["passes"][key][
                "episode_starts"
            ].items()
            if start["verification"] == "pending"
        }

    def mark_pass_result(
        self, status: str, *, schedule_digest: str, reason: str | None = None
    ) -> None:
        """Save the current evaluator pass's explicit completion state.

        Parameters
        ----------
        status : str
            ``incomplete``, ``failed`` or ``complete``. Flush and close alone
            never mark an experiment complete.
        schedule_digest : str
            Nonempty identity of the resolved scientific schedule. It must
            match an earlier marker for this pass.
        reason : str or None, optional
            Explanation for failed or incomplete work, default None.

        Returns
        -------
        None
            Flushes prior data, checks exact scheduled completion and requested
            captures when complete, then durably saves the small status marker.

        Raises
        ------
        ValueError
            Status, identity, completion coverage or requested captures differ.
        RuntimeError
            The writer is closed or failed.
        OSError
            Durable publication fails.
        """
        self._check_open()
        if status not in {"incomplete", "failed", "complete"}:
            raise ValueError("result status must be incomplete, failed or complete")
        if not isinstance(cast(object, schedule_digest), str) or not schedule_digest:
            raise ValueError("result schedule_digest must be nonempty")
        entry = self._details["passes"][self._pass_key]
        previous = entry.get("result_state")
        if previous is not None and previous["schedule_digest"] != schedule_digest:
            raise ValueError("result schedule identity differs from the saved pass")
        self.flush()
        entry = self._details["passes"][self._pass_key]
        if status == "complete":
            self._check_result_completion(entry, schedule_digest)
        entry["result_state"] = {
            "version": 1,
            "status": status,
            "schedule_digest": schedule_digest,
            "reason": reason,
        }
        self.flush()

    def _check_result_completion(
        self, entry: Mapping[str, Any], schedule_digest: str
    ) -> None:
        """Check durable schedule/capture coverage without changing a pass or file."""
        if entry.get("pass_role") == "tournament_coordinator":
            marker = self._details.get("tournament_summary", {}).get(
                "qualification", {}
            )
            if (
                marker.get("status") != "complete"
                or marker.get("schedule_digest") != schedule_digest
            ):
                raise ValueError("tournament completion requires its qualified summary")
        else:
            declared = {int(value) for value in entry["episodes"]}
            completed = set(entry["completed_episode_ids"])
            if not declared or declared != completed:
                raise ValueError(
                    "complete result requires every scheduled episode exactly once"
                )
            details = entry["details"]
            captures = entry["recorded_metrics_by_episode"]
            for episode_id in details.get("full_metrics_episodes", ()):
                if captures.get(str(episode_id)) != "full":
                    raise ValueError(
                        "complete result is missing requested full metrics"
                    )
            for episode_id in details.get("replay_episodes", ()):
                if str(episode_id) not in entry["replays"]:
                    raise ValueError("complete result is missing a requested replay")
            mode = details.get("metrics", "priority")
            if mode != "none" and any(
                captures.get(str(value))
                not in ({"full"} if mode == "full" else {"priority", "full"})
                for value in declared
            ):
                raise ValueError("complete result is missing requested metric coverage")

    def reaffirm_tournament_result(self, schedule_digest: str) -> None:
        """Finish a resumed attempt whose complete summaries are already durable.

        Parameters
        ----------
        schedule_digest : str
            Existing complete tournament's resolved population identity. The
            owning runner must first validate the caller's scientific inputs.

        Returns
        -------
        None
            Updates only existing version-1 tournament result states after all
            pass coverage checks succeed. No game, fit, summary CSV or historical
            failure entry is changed. Already complete states cause no flush.

        Raises
        ------
        ValueError
            The saved summary, schedule identity, pass status or required
            completion/capture evidence is missing or incompatible. Historical
            summaries without new qualification cannot use this method.
        RuntimeError
            The writer is closed or failed.
        OSError
            Status publication fails; ordinary durable recovery rules apply.
        """
        self._check_open()
        summary = self._details.get("tournament_summary", {})
        marker = summary.get("qualification", {})
        if (
            marker.get("version") != 1
            or marker.get("status") != "complete"
            or marker.get("schedule_digest") != schedule_digest
            or marker.get("summary_digest") != summary.get("digest")
        ):
            raise ValueError(
                "reaffirmation requires the existing complete tournament summary"
            )
        updates: list[dict[str, Any]] = []
        for entry in self._details["passes"].values():
            if entry["phase"] != "tournament":
                continue
            state = entry.get("result_state")
            if not isinstance(state, dict) or state.get("version") != 1:
                raise ValueError(
                    "reaffirmation requires recorded tournament result states"
                )
            self._check_result_completion(entry, cast(str, state["schedule_digest"]))
            if state["status"] != "complete":
                updates.append(entry)
        for entry in updates:
            entry["result_state"] = {
                **entry["result_state"],
                "status": "complete",
                "reason": None,
            }
        if updates:
            self.flush()

    def set_tournament_coordinator(self, schedule_digest: str) -> None:
        """Mark the current zero-game pass as the tournament's status owner.

        ``schedule_digest`` is the resolved population identity. Existing
        scientific identity must match. This host call writes the role and an
        incomplete marker; only complete summary publication can finish it.
        A closed writer, populated pass or conflicting role fails before change.
        """
        self._check_open()
        entry = self._details["passes"][self._pass_key]
        if entry["episodes"] or entry["completed_episode_ids"]:
            raise ValueError("a tournament coordinator must not execute games")
        if entry.get("pass_role") not in {None, "tournament_coordinator"}:
            raise ValueError("pass already has a different role")
        previous = entry.get("result_state")
        if previous is not None and previous["schedule_digest"] != schedule_digest:
            raise ValueError("tournament schedule identity differs from the saved pass")
        entry["pass_role"] = "tournament_coordinator"
        summary = self._details.get("tournament_summary", {}).get("qualification", {})
        self.mark_pass_result(
            "complete" if summary.get("status") == "complete" else "incomplete",
            schedule_digest=schedule_digest,
        )

    def _prepare_pass(
        self,
        phase: str,
        pass_id: str,
        policies: dict[str, object] | None,
        checkpoint_id: str | None,
        details: dict[str, object] | None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Validate supplied pass facts without flushing, recovering or changing files.

        Normalize each team once. Returned registrations contain only serializable
        identity evidence, never method memory or a call to the method itself.
        Existing stable facts must match; execution placement may differ.
        """
        return _prepare_pass_identity(
            self._details, phase, pass_id, policies, checkpoint_id, details
        )

    def register_episodes(
        self,
        schedule: Iterable[dict[str, object]] | object,
        *,
        source_configs: EnvConfig | None = None,
    ) -> None:
        """Register immutable schedule facts or pending first-start declarations.

        Parameters
        ----------
        schedule : iterable of dict or EpisodeStartRecords
            Host schedule dictionaries contain a positive int32 episode_id.
            Numerical starts carry source-bank references and reset generations.
            Optional class rows declare a compact catalog roster against that
            source. Legacy None or ten -1 values mean the exact original source.
            An identical repeated declaration is allowed; a changed one fails.
        source_configs : EnvConfig or None, default None
            One immutable source config or a leading source batch. Required for
            a new referenced bank; omit it once that content identity is saved.
            Order belongs to identity. Reusing identical immutable JAX leaves
            avoids hashing. Fresh JAX leaves with matching structure are checked
            on the same device placement against the latest verified bank;
            one bool is read back. A changed device placement uses normal hashing.
            Mutable NumPy contents are hashed on every resupply.

        Returns
        -------
        None
            The whole declaration is checked, then published at one durable
            boundary. A start remains pending until write receives its actual
            first-transition configuration. This does not certify spawn balance.

        Raises
        ------
        ValueError
            IDs, shapes, source references or existing declarations disagree.
        RuntimeError
            The writer is closed or failed.
        OSError
            Durable publication fails.

        Notes
        -----
        Host-only. Never mutate or donate a live source bank. Reset generations
        do not permit reusing an episode ID for another game within one pass.
        Empty numerical starts with an unchanged saved bank are checked, then
        return without flushing other pending data. Use flush explicitly when
        that data must become durable. Mutable banks are still hashed on resupply.
        """
        self._check_open()
        from marl_battlegrounds.evaluation.recording_types import EpisodeStartRecords

        entry = self._details["passes"][self._pass_key]
        if isinstance(schedule, EpisodeStartRecords):
            bank_id, bank_refs, contents, cache = self._prepare_source_bank(
                source_configs
            )
            banks = self._details["source_banks"]
            if bank_id is not None and bank_id not in banks:
                banks = {**banks, bank_id: bank_refs}
            declarations = self._start_declarations(schedule, banks)
            for episode_id, declaration in declarations.items():
                if (
                    bank_id is not None
                    and declaration["source_known"]
                    and declaration["source_table_id"] != bank_id
                ):
                    raise ValueError(
                        "supplied source bank differs from the declared "
                        "source-table identity"
                    )
                previous = entry["episode_starts"].get(episode_id)
                if previous is None and self._episode_has_records(int(episode_id)):
                    raise ValueError("register starts before recording an episode")
                if previous is not None and not _same_start(previous, declaration):
                    raise ValueError(
                        f"episode {episode_id} differs from its recorded start"
                    )
            bank_changed = bank_id is not None and (
                bank_id not in self._details["source_banks"]
                or any(
                    self._details["configurations"].get(identifier) != content
                    for identifier, content in contents.items()
                )
            )
            if bank_id is not None:
                self._details["source_banks"][bank_id] = bank_refs
                self._details["configurations"].update(contents)
                if cache is not None:
                    self._source_cache = [cache]
            if not declarations and not bank_changed:
                return
            for episode_id, declaration in declarations.items():
                if episode_id not in entry["episode_starts"]:
                    self._pending_numerical_starts.add(int(episode_id))
                entry["episode_starts"].setdefault(
                    episode_id,
                    {
                        **declaration,
                        "verification": "pending",
                        "resolved_config_id": None,
                    },
                )
        else:
            if source_configs is not None:
                raise ValueError(
                    "source_configs accompanies numerical start records only"
                )
            episodes = entry["episodes"]
            records: dict[str, dict[str, object]] = {}
            registrations: dict[str, object] = {}
            from marl_battlegrounds.evaluation.recording_identity import (
                normalize_system_registration,
            )

            for specification in cast(Iterable[dict[str, object]], schedule):
                record = cast(dict[str, object], _json_value(specification))
                episode_id = str(
                    self._integer(
                        record.get("episode_id"), "scheduled episode_id", minimum=1
                    )
                )
                if episode_id not in episodes and self._episode_has_records(
                    int(episode_id)
                ):
                    raise ValueError(
                        "register episode declarations before recording data"
                    )
                if "policies" in record:
                    ids: dict[str, str] = {}
                    for team, definition in cast(
                        dict[str, object], record["policies"]
                    ).items():
                        identifier, normalized = normalize_system_registration(
                            definition, phase=entry["phase"]
                        )
                        registrations[identifier] = normalized
                        ids[team] = identifier
                    record["system_ids"] = ids
                if (episode_id in episodes and episodes[episode_id] != record) or (
                    episode_id in records and records[episode_id] != record
                ):
                    raise ValueError(
                        f"episode {episode_id} differs from the recorded schedule"
                    )
                records[episode_id] = record
            episodes.update(records)
            self._details["systems"].update(registrations)
        self.flush()

    def _episode_has_records(self, episode_id: int) -> bool:
        """Check for admitted evidence whose owner later labels cannot change."""
        entry = self._details["passes"][self._pass_key]
        key = str(episode_id)
        return (
            (self._pass_key, episode_id) in self._completed
            or key in entry["trace_epochs"]
            or key in entry["replays"]
            or key in self._pending_replays
            or entry["episode_starts"].get(key, {}).get("verification")
            in ("verified", "custom")
            or (
                self._collector is not None
                and episode_id in self._collector.pending_episode_ids
            )
        )

    def _prepare_source_bank(
        self,
        source_configs: EnvConfig | None,
    ) -> tuple[
        str | None,
        list[str],
        dict[str, object],
        tuple[object, tuple[object, ...], str] | None,
    ]:
        """Identify an ordered source bank without changing recorded state.

        Content hashes use configuration_identity once per source. Exact held
        immutable references skip all comparison. Fresh immutable leaves use one
        device equality check against the latest verified bank when their tree,
        shapes, dtypes and device placement match. Only its boolean result reaches
        the host. A changed placement uses normal hashing. NumPy
        is always hashed. The cache retains at most one complete source bank.
        """
        if source_configs is None:
            return None, [], {}, None
        import jax

        from marl_battlegrounds.core.types import EnvConfig
        from marl_battlegrounds.evaluation.recording_identity import (
            _source_bank_values_equal,  # pyright: ignore[reportPrivateUsage]
            ordered_source_bank_identity,
        )

        if not isinstance(cast(object, source_configs), EnvConfig):
            raise TypeError("source_configs must be an EnvConfig source or batch")
        leaves = jax.tree.leaves(source_configs)
        structure = cast(object, jax.tree.structure(source_configs))
        immutable = all(isinstance(leaf, jax.Array) for leaf in leaves)
        if immutable:
            for old_structure, old_leaves, identifier in self._source_cache:
                if old_structure != structure or len(old_leaves) != len(leaves):
                    continue
                pairs = tuple(zip(old_leaves, leaves, strict=True))
                identical = all(old is new for old, new in pairs)
                old_arrays = cast(tuple[jax.Array, ...], old_leaves)
                new_arrays = cast(tuple[jax.Array, ...], tuple(leaves))
                if identical or (
                    all(
                        old.shape == new.shape
                        and old.dtype == new.dtype
                        and old.sharding == new.sharding
                        for old, new in zip(old_arrays, new_arrays, strict=True)
                    )
                    and bool(
                        cast(
                            jax.Array, _source_bank_values_equal(old_arrays, new_arrays)
                        )
                    )
                ):
                    return (
                        identifier,
                        self._details["source_banks"][identifier],
                        {},
                        None if identical else (structure, tuple(leaves), identifier),
                    )
        identifier, references, contents = ordered_source_bank_identity(source_configs)
        cached = (structure, tuple(leaves), identifier) if immutable else None
        return identifier, references, contents, cached

    def _start_declarations(
        self, starts: object, banks: Mapping[str, Any]
    ) -> dict[str, dict[str, object]]:
        """Check numerical start shapes, IDs and references, without granting evidence.

        Invalid rows are padding; every field still has its declared shape/dtype.
        Unknown sources keep unknown indices/choices and a zero source-table ID.
        Class rows have trailing width ten. Save explicit overrides as lists;
        omit the field for legacy None or all-minus-one sentinel rows. A stored
        override requires known source ownership and compact class/zero padding.
        """
        import jax

        from marl_battlegrounds.evaluation.recording_types import EpisodeStartRecords

        if not isinstance(starts, EpisodeStartRecords):
            raise TypeError("episode_start_records must be EpisodeStartRecords")
        host = jax.device_get(starts)
        shape = np.shape(host.valid)
        arrays: dict[str, np.ndarray[Any, Any]] = {}
        for name in EpisodeStartRecords._fields:
            if name == "source_class_ids" and host.source_class_ids is None:
                continue
            array = np.asarray(getattr(host, name))
            width = (
                8
                if name == "source_table_id"
                else 10
                if name == "source_class_ids"
                else None
            )
            wanted = (*shape, width) if width is not None else shape
            dtype = (
                np.uint32
                if name == "source_table_id"
                else np.bool_
                if name in ("source_known", "authored_start", "valid")
                else np.int32
            )
            if array.shape != wanted or array.dtype != dtype:
                raise ValueError(f"start {name} has the wrong shape or dtype")
            arrays[name] = array.reshape((-1, width) if width is not None else (-1,))
        result: dict[str, dict[str, object]] = {}
        for index in np.flatnonzero(arrays["valid"]):
            episode_id = str(
                self._integer(
                    arrays["episode_id"][index], "start episode_id", minimum=1
                )
            )
            generation = self._integer(
                arrays["reset_generation"][index], "reset_generation", minimum=0
            )
            source_index = self._integer(
                arrays["source_index"][index], "source_index", minimum=-1
            )
            spawn = self._integer(
                arrays["spawn_locations"][index], "spawn_locations", minimum=-1
            )
            if spawn not in (-1, 0, 1):
                raise ValueError("spawn_locations must be -1, 0 or 1")
            stage = self._integer(
                arrays["episode_start_stage"][index], "episode_start_stage", minimum=-1
            )
            known = bool(arrays["source_known"][index])
            words = arrays["source_table_id"][index]
            table_id = words.astype(">u4").tobytes().hex() if np.any(words) else None
            if known:
                if (
                    table_id is None
                    or table_id not in banks
                    or not 0 <= source_index < len(banks[table_id])
                ):
                    raise ValueError(
                        "start source bank or source index is missing or invalid"
                    )
            elif source_index != -1 or spawn != -1 or table_id is not None:
                raise ValueError(
                    "unknown starts require unknown source index/spawn "
                    "and a zero bank ID"
                )
            declaration: dict[str, object] = {
                "episode_id": int(episode_id),
                "reset_generation": generation,
                "source_table_id": table_id,
                "source_index": source_index,
                "spawn_locations": spawn,
                "episode_start_stage": stage,
                "source_known": known,
                "authored_start": bool(arrays["authored_start"][index]),
            }
            if "source_class_ids" in arrays:
                classes = arrays["source_class_ids"][index]
                if not np.all(classes == -1):
                    declaration["source_class_ids"] = classes.tolist()
                    _roster_declaration(declaration)
            if episode_id in result and result[episode_id] != declaration:
                raise ValueError("one episode ID has conflicting start declarations")
            result[episode_id] = declaration
        return result

    def _verify_starts(
        self,
        records: EpisodeInfo | _RecordingRows,
        config_cache: dict[int, tuple[EnvConfig, str, dict[str, object]]],
        *,
        entry: dict[str, Any] | None = None,
        banks: Mapping[str, list[str]] | None = None,
        configurations: Mapping[str, object] | None = None,
    ) -> list[tuple[int, dict[str, object], str, dict[str, object]]]:
        """Join registered starts to their actual first-transition configurations.

        Return verified updates only after checking all rows. Unknown/authored
        starts remain custom. Later real rows must keep the recorded binding.
        """
        from marl_battlegrounds.tasks import (
            _config_has_batch,  # pyright: ignore[reportPrivateUsage]
            _validate_config_choices,  # pyright: ignore[reportPrivateUsage]
        )

        entry = cast(
            dict[str, Any],
            self._details["passes"][self._pass_key] if entry is None else entry,
        )
        banks = cast(
            Mapping[str, list[str]],
            self._details["source_banks"] if banks is None else banks,
        )
        configurations = cast(
            Mapping[str, object],
            (
                self._details["configurations"]
                if configurations is None
                else configurations
            ),
        )
        starts = records.episode_start_records
        updates: list[tuple[int, dict[str, object], str, dict[str, object]]] = []
        declarations = {} if starts is None else self._start_declarations(starts, banks)
        pending_bindings: dict[str, str] = {}
        for index in range(np.size(records.episode_id)):
            episode_id = str(int(records.episode_id[index]))
            start_valid = starts is not None and bool(starts.valid[index])
            bound = entry["episode_starts"].get(episode_id)
            if not start_valid and bound is None:
                continue
            config_tree, config_id, content = self._config_evidence(
                records, index, config_cache
            )
            if start_valid:
                assert starts is not None
                if str(int(starts.episode_id[index])) != episode_id:
                    raise ValueError("start episode ID must match the producing info")
                declaration = declarations[episode_id]
                if bound is None or not _same_start(bound, declaration):
                    raise ValueError(
                        "first-start evidence must match its registered declaration"
                    )
                if (
                    int(starts.episode_id[index]) != int(records.episode_id[index])
                    or int(records.decision_step[index]) != 0
                ):
                    raise ValueError(
                        "start evidence must belong to this episode's "
                        "first real transition"
                    )
                if bound["verification"] != "pending" or episode_id in pending_bindings:
                    raise ValueError(
                        "first-start evidence was already recorded for this episode"
                    )
                validation_key = _validation_key(config_tree, config_id)
                if validation_key not in self._validated_start_configs:
                    batched = _config_has_batch(config_tree, None)
                    _validate_config_choices(
                        config_tree, batched=batched, both_spawn_choices=False
                    )
                    _remember_verification(
                        self._validated_start_configs, validation_key, None
                    )
                known = bool(declaration["source_known"])
                if known:
                    refs = banks[cast(str, declaration["source_table_id"])]
                    source_id = refs[cast(int, declaration["source_index"])]
                    roster = _roster_declaration(declaration)
                    relationship_key = (validation_key, source_id, roster)
                    choice = self._verified_source_choices.get(relationship_key)
                    if choice is None:
                        source = configurations[source_id]

                        def restore(template: object, values: object) -> object:
                            """Restore source data using the actual tuple/dtype tree."""
                            if isinstance(template, tuple) and hasattr(
                                cast(object, template), "_fields"
                            ):
                                data = cast(dict[str, object], values)
                                named = cast(Any, template)
                                return type(named)(
                                    *(
                                        restore(getattr(named, k), data[k])
                                        for k in cast(tuple[str, ...], named._fields)
                                    )
                                )
                            return np.asarray(values, dtype=np.asarray(template).dtype)

                        source_tree = cast("EnvConfig", restore(config_tree, source))
                        matches, found_choice = _source_relationship_check()(
                            config_tree,
                            source_tree,
                            cast(
                                "Array",
                                np.asarray(
                                    (-1,) * 10 if roster is None else roster,
                                    dtype=np.int32,
                                ),
                            ),
                        )
                        if not bool(matches):
                            raise ValueError(
                                "actual first-start configuration differs from "
                                "its declared source/spawn choice"
                            )
                        choice = int(found_choice)
                        _remember_verification(
                            self._verified_source_choices, relationship_key, choice
                        )
                    if choice != declaration["spawn_locations"]:
                        raise ValueError(
                            "actual first-start configuration differs from "
                            "its declared source/spawn choice"
                        )
                verification = (
                    "verified"
                    if known and not declaration["authored_start"]
                    else "custom"
                )
                updates.append(
                    (
                        int(episode_id),
                        {
                            **declaration,
                            "verification": verification,
                            "resolved_config_id": config_id,
                        },
                        config_id,
                        content,
                    )
                )
                pending_bindings[episode_id] = config_id
            expected = pending_bindings.get(episode_id) or cast(
                dict[str, Any], bound or {}
            ).get("resolved_config_id")
            if expected is not None and config_id != expected:
                raise ValueError(
                    "episode configuration changed after its recorded start"
                )
            if (
                not start_valid
                and int(records.decision_step[index]) >= 0
                and bound
                and bound["verification"] == "pending"
                and episode_id not in pending_bindings
            ):
                raise ValueError(
                    "registered start needs its first-transition evidence "
                    "before later records"
                )
        return updates

    @property
    def has_pending_numerical_starts(self) -> bool:
        """Report unresolved numerical starts without scanning episode history.

        A collector must reject this state before choosing an action. Finish the
        earlier register/write handoff first. Host schedules are not pending
        numerical starts. This derived set is rebuilt when selecting a pass.
        """
        return bool(self._pending_numerical_starts)

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
    ) -> tuple[EvaluationEpisodeContextV4, RuntimeProvenanceV1]:
        """Consume one context prepared before any replay publication.

        This callback runs only during application. Missing preparation is an
        internal error; it must not discover identities after preflight ends.
        """
        context = self._preflight_contexts.pop(int(packet.episode_id), None)
        if context is None:
            raise RuntimeError("replay context was not prepared before application")
        return context

    def _prepare_replay_context(
        self, packet: ReplayPackets, provenance: dict[str, object] | None
    ) -> tuple[
        EvaluationEpisodeContextV4, RuntimeProvenanceV1, dict[str, object] | None
    ]:
        """Resolve initial replay identities without changing writer state.

        provenance holds any discovery already made in this preflight. Return
        the context, runtime and discovery for the later application boundary.
        No live collector, spool or recorded metadata is created here.
        """
        from marl_battlegrounds.evaluation.recording_context import (
            build_recording_context,
            capture_recording_provenance,
        )

        entry = self._details["passes"][self._pass_key]
        details = entry["details"]
        if "runtime_provenance" not in details:
            if provenance is None:
                provenance = self._recording_provenance
            if provenance is None:
                provenance = capture_recording_provenance()
            details = {**details, **provenance}
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
            details={
                **details,
                "system_ids": episode.get("system_ids", entry["system_ids"]),
                "systems": self._details["systems"],
            },
        )
        return context, runtime, provenance

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
        try:
            checked = self._preflight_replays(packets, {}, {})
            assert checked is not None
            self._consume_replay(checked)
        except BaseException as error:
            self._record_failure(error)
            raise

    def _consume_replay(self, prepared: _PreparedReplay) -> None:
        """Publish packets after whole-call checks; retain their summary and config.

        Direct and compact writer calls share this host-only path. Each completed
        artifact is validated and serialized once into private temporary storage.
        Only after the whole family succeeds are final files published. Memory
        holds one replay at a time. Unfinished episodes keep using their existing
        numerical spools. File failures keep the usual partial-publication limits.
        """
        from marl_battlegrounds.evaluation.replay_io import (
            PreparedReplay,
            _publish_staged_replays,  # pyright: ignore[reportPrivateUsage]
            generated_replay_filename,
            preflight_replay_destination,
            publish_prepared_replay,
        )
        from marl_battlegrounds.evaluation.replay_recording import ReplayCollector

        try:
            packets = prepared.packets
            self._preflight_contexts.update(prepared.contexts)
            self._replay_config_ids.update(prepared.config_ids)
            self._replay_ids.update(
                {
                    context.identity.episode_id: identifier
                    for identifier, (context, _) in prepared.contexts.items()
                }
            )
            if prepared.provenance is not None:
                self._recording_provenance = prepared.provenance
                self._details["passes"][self._pass_key]["recording_provenance"] = (
                    prepared.provenance
                )
            if self._collector is None:
                self._collector = ReplayCollector(
                    self._replay_context, spool_dir=self.run_dir / ".replay_spool"
                )
            summaries = _replay_summaries(packets)
            with ExitStack() as cleanup:
                staging: Path | None = None
                staged: list[tuple[Path, Path, int]] = []
                metadata: dict[str, dict[str, object]] = {}
                for replay in self._collector.write(packets):
                    episode_id = self._replay_ids[
                        replay.header.context.identity.episode_id
                    ]
                    if (self._pass_key, episode_id) in self._completed:
                        raise ValueError(
                            f"replay episode {episode_id} was already completed"
                        )
                    if staging is None:
                        staging = Path(
                            cleanup.enter_context(
                                TemporaryDirectory(
                                    prefix=".replay_publish-", dir=self.run_dir
                                )
                            )
                        )
                    path = (
                        self.run_dir
                        / "replays"
                        / generated_replay_filename(
                            replay.header.context,
                            replay.canonical_digest_sha256,
                            episode_id=episode_id,
                        )
                    )
                    private_path = staging / f"{episode_id}.marlbg-replay.json"
                    prepared_replay = PreparedReplay._from_capture(replay)  # pyright: ignore[reportPrivateUsage]
                    saved = publish_prepared_replay(
                        prepared_replay, preflight_replay_destination(private_path)
                    )
                    staged.append((private_path, path, saved.replay_byte_length))
                    metadata[str(episode_id)] = {
                        "path": str(path.relative_to(self.run_dir)),
                        "canonical_digest_sha256": replay.canonical_digest_sha256,
                        "bytes": saved.replay_byte_length,
                        "summary": summaries[episode_id],
                        "config_id": self._replay_config_ids[episode_id],
                    }
                    # Keep one completed model and byte payload at a time. Later
                    # invalid artifacts must not follow any final publication.
                    del prepared_replay, replay
                if staged:
                    (self.run_dir / "replays").mkdir(exist_ok=True)
                    _publish_staged_replays(staged)
                    self._pending_replays.update(metadata)
                    for episode_id in metadata:
                        self._replay_config_ids.pop(int(episode_id))
        except BaseException as error:
            self._record_failure(error)
            raise

    def write(self, infos: EpisodeInfo, *, policy_trace: object = None) -> None:
        """Validate a whole step or chunk, then record its selected data.

        Parameters
        ----------
        infos : EpisodeInfo
            Scalar, native-batch or time/batch records with matching numerical
            leading axes. Required lifecycle errors and optional tracking errors
            reject the whole call, even on padding. Replay keeps its own axes.
        policy_trace : PolicyTrace or None, default None
            Optional submitted-action component choices with the same leading
            shape. A scalar System's batch-one trace is also accepted for scalar
            info. IDs index each team's registered ordered components; -1 means
            unknown. No memory or learning outputs are recorded.

        Returns
        -------
        None
            Completed outcomes and selected metrics are buffered. Start evidence
            is checked before dependent output. Trace intervals also flush at
            their bounded capacity, without waiting for a game to finish.

        Raises
        ------
        ValueError
            An error flag, shape, ID, epoch, source claim, metric or completion is
            invalid. The full call is checked before consuming its replay data.
        RuntimeError
            This writer is closed or failed. Reopen through resume after failure.
        OSError
            Replay or durable table publication failed.

        Notes
        -----
        Host-only; transfers supplied numerical recording fields. Learner-only
        final inputs are excluded before transfer and are never serialized.
        A healthy flush/close
        makes pending data durable. Unavailable measurements stay blank. A
        failed call cannot be retried on this writer; earlier committed records
        remain authoritative. This does not restore a learner or provider.
        """
        self._check_open()
        import jax

        from marl_battlegrounds.evaluation.recording_types import (
            validate_recording_errors,
        )

        try:
            validate_recording_errors(infos)
            host = jax.device_get(infos._replace(replay=None, final=None))
            completion = np.asarray(host.completed)
            if completion.dtype != np.bool_:
                raise ValueError("completed must have boolean dtype")
            leading = completion.shape
            for name in (
                "episode_id",
                "outcome",
                "decision_step",
                "episode_length",
                "team_scores",
                "lifecycle_error",
            ):
                value = np.asarray(getattr(host, name))
                expected = (*leading, 2) if name == "team_scores" else leading
                dtype = np.bool_ if name == "lifecycle_error" else np.int32
                if value.shape != expected or value.dtype != dtype:
                    raise ValueError(f"info.{name} has the wrong shape or dtype")

            def flatten(value: object) -> np.ndarray[Any, Any]:
                """Check and flatten info axes, leaving each record's payload intact."""
                array = np.asarray(value)
                if array.shape[: len(leading)] != leading:
                    raise ValueError(
                        "info fields must share the completed leading shape"
                    )
                return array.reshape((-1, *array.shape[len(leading) :]))

            records = jax.tree.map(flatten, host)
            config_cache: dict[int, tuple[EnvConfig, str, dict[str, object]]] = {}
            starts = self._verify_starts(records, config_cache)
            self._check_evidence_ownership(records, config_cache)
            traces = self._prepare_traces(records, policy_trace, leading, config_cache)
            prepared: list[
                tuple[int, str, dict[str, object], list[tuple[str, list[object]]], str]
            ] = []
            seen: set[int] = set()
            for index in np.flatnonzero(completion):
                record = self._prepare_record(records, int(index), config_cache)
                episode_id = record[0]
                if episode_id in seen:
                    raise ValueError(
                        f"episode {episode_id} completed twice in one call"
                    )
                seen.add(episode_id)
                prepared.append(record)
            terminal_epochs = {
                int(records.episode_id[i]): int(records.decision_step[i])
                for i in np.flatnonzero(completion)
            }
            if any(
                trace[0] in terminal_epochs and trace[1] > terminal_epochs[trace[0]]
                for trace in traces
            ):
                raise ValueError(
                    "trace decision follows the supplied episode completion"
                )
            bindings: dict[int, str] = {}
            for episode_id, _, config_id, _ in starts:
                bindings[episode_id] = config_id
            for trace in traces:
                if bindings.get(trace[0], trace[5]) != trace[5]:
                    raise ValueError("trace configuration differs from its start")
                bindings[trace[0]] = trace[5]
            for episode_id, config_id, _, _, _ in prepared:
                if bindings.get(episode_id, config_id) != config_id:
                    raise ValueError(
                        "completion configuration differs from its trace/start"
                    )
                bindings[episode_id] = config_id
            summaries = {
                int(records.episode_id[i]): (
                    int(records.outcome[i]),
                    int(records.episode_length[i]),
                    int(records.team_scores[i, 0]),
                    int(records.team_scores[i, 1]),
                )
                for i in np.flatnonzero(completion)
            }
            replay = self._preflight_replays(infos.replay, summaries, bindings)
            self._apply_prepared(starts, traces, prepared, replay)
        except BaseException as error:
            self._record_failure(error)
            raise

    def _check_evidence_ownership(
        self,
        records: EpisodeInfo | _RecordingRows,
        config_cache: dict[int, tuple[EnvConfig, str, dict[str, object]]],
        *,
        entry: dict[str, Any] | None = None,
    ) -> None:
        """Check actual configs against every existing recorded owner, even on padding.

        An unbound row does not create an identity or durable episode ownership.
        Known rows use the same per-call config cache as starts and completions.
        No new manifest entry is published by inspecting evidence alone.
        """
        entry = cast(
            dict[str, Any],
            self._details["passes"][self._pass_key] if entry is None else entry,
        )
        for index, value in enumerate(np.asarray(records.episode_id).reshape(-1)):
            episode_id = int(value)
            key = str(episode_id)
            episode = entry["episodes"].get(key, {})
            replay = cast(
                dict[str, object],
                self._pending_replays.get(key, entry["replays"].get(key, {})),
            )
            expected = (
                episode.get("configuration_digest", episode.get("config_id")),
                entry["episode_starts"].get(key, {}).get("resolved_config_id"),
                entry["trace_config_ids"].get(key),
                entry.get("completed_config_ids", {}).get(key),
                replay.get("config_id", self._replay_config_ids.get(episode_id)),
            )
            if not any(identifier is not None for identifier in expected):
                continue
            _, actual, _ = self._config_evidence(records, index, config_cache)
            if any(
                identifier is not None and identifier != actual
                for identifier in expected
            ):
                raise ValueError(
                    "episode configuration differs from recorded episode evidence"
                )

    def _write_collected(
        self, batch: object, *, source_configs: EnvConfig | None = None
    ) -> None:
        """Validate one bounded drain, then use the ordinary writer transaction.

        Parameters
        ----------
        batch : CollectedBatch
            Private numerical families already sliced to occupied prefixes.
            Config references are local to its evidence table. No learner
            output, System memory or final learning input belongs here.
        source_configs : EnvConfig or None, default None
            Explicit source bank for new start declarations. Existing immutable
            cache checks apply; ordinary drains do not repeatedly hash it.

        Notes
        -----
        Host-only. Small errors are checked before payload transfer. Every
        family is checked before metadata or dependent records are admitted.
        A rejected drain fails this writer; prior durable drains remain valid.
        Evidence-only drains check ownership without flushing unrelated output.
        """
        self._check_open()
        import jax

        from marl_battlegrounds.evaluation.collection_types import CollectedBatch
        from marl_battlegrounds.evaluation.recording_types import (
            validate_recording_errors,
        )

        try:
            if not isinstance(batch, CollectedBatch):
                raise TypeError("collected records must be a CollectedBatch")
            errors, counts = jax.device_get((batch.errors, batch.counts))
            validate_recording_errors(errors)
            for name in errors._fields:
                value = np.asarray(getattr(errors, name))
                shape = (
                    np.shape(errors.lifecycle_error)
                    if name in ("lifecycle_error", "episode_tracking_error", "code")
                    else ()
                )
                dtype = np.bool_ if name == "lifecycle_error" else np.int32
                if value.shape != shape or value.dtype != dtype:
                    raise ValueError(
                        f"collection error {name} has the wrong shape or dtype"
                    )
            if len(np.shape(errors.lifecycle_error)) != 1:
                raise ValueError("collection errors require one lane axis")
            if np.any(np.asarray(errors.code)):
                raise ValueError(
                    "collection failed at step "
                    f"{int(errors.step)}, lane {int(errors.lane)}: "
                    f"codes {np.asarray(errors.code).tolist()}"
                )
            if any(
                np.asarray(value).shape != () or np.asarray(value).dtype != np.int32
                for value in counts
            ):
                raise ValueError("collection counts must be int32 scalars")
            sizes = {
                name: self._integer(getattr(counts, name), f"{name} count", minimum=0)
                for name in counts._fields
            }
            buffers = jax.device_get(batch.buffers)
            for name, count in sizes.items():
                family = getattr(buffers, name)
                if family is None:
                    if count:
                        raise ValueError(f"absent {name} has a nonzero count")
                    continue
                if any(
                    np.shape(leaf)[:1] != (count,) for leaf in jax.tree.leaves(family)
                ):
                    raise ValueError(f"{name} must contain exactly its occupied prefix")

            def integers(
                value: object, name: str, count: int, trailing: tuple[int, ...] = ()
            ) -> np.ndarray[Any, Any]:
                """Check a compact int32 field without coercion or clipping."""
                result = np.asarray(value)
                if result.shape != (count, *trailing) or result.dtype != np.int32:
                    raise ValueError(f"collected {name} has the wrong shape or dtype")
                return result

            evidence = buffers.evidence
            ids = integers(evidence.episode_id, "evidence IDs", sizes["evidence"])
            steps = integers(evidence.decision_step, "evidence decisions", len(ids))
            if (
                np.any(ids <= 0)
                or np.any(steps < -1)
                or len(set(ids.tolist())) != len(ids)
            ):
                raise ValueError(
                    "evidence needs unique positive IDs and valid decisions"
                )
            evidence_rows = _RecordingRows(
                ids, steps, None, evidence.config, np.arange(len(ids))
            )
            evidence_cache: dict[int, tuple[EnvConfig, str, dict[str, object]]] = {}

            def rows_for(
                row_ids: object,
                decisions: object,
                references: object,
                count: int,
                *,
                lengths: object = None,
            ) -> tuple[
                _RecordingRows, dict[int, tuple[EnvConfig, str, dict[str, object]]]
            ]:
                """Join a compact family to actual evidence without copying configs."""
                claimed = integers(row_ids, "producing IDs", count)
                epochs = integers(decisions, "producing decisions", count)
                refs = integers(references, "config references", count)
                if np.any(refs < 0) or np.any(refs >= len(ids)):
                    raise ValueError("config reference is outside the evidence table")
                if not np.array_equal(ids[refs], claimed):
                    raise ValueError(
                        "config evidence ID differs from its producing record"
                    )
                if np.any(epochs < -1):
                    raise ValueError("producing decisions must be -1 or nonnegative")
                if np.any((epochs >= 0) & ((steps[refs] < 0) | (epochs < steps[refs]))):
                    raise ValueError(
                        "real records need same-episode configuration evidence "
                        "from this decision or an earlier real decision"
                    )
                cache: dict[int, tuple[EnvConfig, str, dict[str, object]]] = {}
                for index, ref in enumerate(refs):
                    cache[index] = self._config_evidence(
                        evidence_rows, int(ref), evidence_cache
                    )
                return _RecordingRows(
                    claimed, epochs, lengths, evidence.config, refs
                ), cache

            entry = self._details["passes"][self._pass_key]
            bank_id = None
            bank_refs: list[str] = []
            contents: dict[str, object] = {}
            bank_cache = None
            declarations: dict[str, dict[str, object]] = {}
            starts: list[tuple[int, dict[str, object], str, dict[str, object]]] = []
            if buffers.starts is not None and sizes["starts"]:
                bank_id, bank_refs, contents, bank_cache = self._prepare_source_bank(
                    source_configs
                )
                banks = ChainMap(
                    {bank_id: bank_refs} if bank_id is not None else {},
                    self._details["source_banks"],
                )
                declarations = self._start_declarations(buffers.starts.records, banks)
                if len(declarations) != sizes["starts"] or not np.all(
                    buffers.starts.records.valid
                ):
                    raise ValueError(
                        "collected starts must be distinct valid declarations"
                    )
                pending: dict[str, dict[str, object]] = {}
                for identifier, declaration in declarations.items():
                    if identifier in entry[
                        "episode_starts"
                    ] or self._episode_has_records(int(identifier)):
                        raise ValueError(
                            "collected start was already registered or recorded"
                        )
                    if (
                        bank_id is not None
                        and declaration["source_known"]
                        and declaration["source_table_id"] != bank_id
                    ):
                        raise ValueError(
                            "supplied bank differs from declared source identity"
                        )
                    pending[identifier] = {
                        **declaration,
                        "verification": "pending",
                        "resolved_config_id": None,
                    }
                candidate = {
                    **entry,
                    "episode_starts": ChainMap(pending, entry["episode_starts"]),
                }
                start_rows, cache = rows_for(
                    buffers.starts.info_episode_id,
                    buffers.starts.decision_step,
                    buffers.starts.config_index,
                    sizes["starts"],
                )
                start_rows = start_rows._replace(
                    episode_start_records=buffers.starts.records
                )
                starts = self._verify_starts(
                    start_rows,
                    cache,
                    entry=candidate,
                    banks=banks,
                    configurations=ChainMap(contents, self._details["configurations"]),
                )
            verified = {
                str(identifier): declaration for identifier, declaration, _, _ in starts
            }
            candidate = {
                **entry,
                "episode_starts": ChainMap(verified, entry["episode_starts"]),
            }
            self._verify_starts(evidence_rows, evidence_cache, entry=candidate)
            self._check_evidence_ownership(
                evidence_rows, evidence_cache, entry=candidate
            )

            traces: Any = []
            if buffers.assignments is not None:
                family = buffers.assignments
                count = sizes["assignments"]
                lengths = integers(family.episode_length, "trace lengths", count)
                trace_rows, cache = rows_for(
                    family.info_episode_id,
                    family.decision_step,
                    family.config_index,
                    count,
                    lengths=lengths,
                )
                if not np.all(family.trace.valid):
                    raise ValueError("occupied assignments must all be valid")
                traces = self._prepare_traces(
                    trace_rows, family.trace, (count,), cache, entry=candidate
                )

            family = buffers.completions
            count = sizes["completions"]
            lengths = integers(family.episode_length, "completed lengths", count)
            outcomes = integers(family.outcome, "outcomes", count)
            scores = integers(family.team_scores, "scores", count, (2,))
            completion_rows, cache = rows_for(
                family.episode_id,
                family.decision_step,
                family.config_index,
                count,
                lengths=lengths,
            )
            completion_rows = completion_rows._replace(
                outcome=outcomes,
                team_scores=scores,
                priority=buffers.priority,
                full=buffers.full,
            )
            metric_refs: list[np.ndarray[Any, Any]] = []
            for name, refs in (
                ("priority", family.priority_index),
                ("full", family.full_index),
            ):
                values = integers(refs, f"{name} references", count)
                if np.any(values < -1) or np.any(values >= sizes[name]):
                    raise ValueError(f"{name} reference is outside its metric table")
                used = values[values >= 0]
                if sorted(used.tolist()) != list(range(sizes[name])):
                    raise ValueError(f"{name} rows need exactly one completion owner")
                metric_refs.append(values)
            if len(set(np.asarray(family.episode_id).tolist())) != count:
                raise ValueError("one episode completed twice in a drain")
            prepared = [
                self._prepare_record(
                    completion_rows,
                    index,
                    cache,
                    entry=candidate,
                    metric_indices=(
                        int(metric_refs[0][index]),
                        int(metric_refs[1][index]),
                    ),
                )
                for index in range(count)
            ]
            terminal = dict(
                zip(
                    np.asarray(family.episode_id).tolist(),
                    np.asarray(family.decision_step).tolist(),
                    strict=True,
                )
            )
            if any(trace[1] > terminal.get(trace[0], trace[1]) for trace in traces):
                raise ValueError("trace decision follows the supplied completion")
            bindings = {identifier: config_id for identifier, _, config_id, _ in starts}
            for trace in traces:
                if bindings.get(trace[0], trace[5]) != trace[5]:
                    raise ValueError("trace configuration differs from its start")
                bindings[trace[0]] = trace[5]
            for identifier, config_id, _, _, _ in prepared:
                if bindings.get(identifier, config_id) != config_id:
                    raise ValueError(
                        "completion configuration differs from its trace/start"
                    )
                bindings[identifier] = config_id
            summaries = {
                int(family.episode_id[index]): (
                    int(outcomes[index]),
                    int(lengths[index]),
                    int(scores[index, 0]),
                    int(scores[index, 1]),
                )
                for index in range(count)
            }
            replay = self._preflight_replays(buffers.replay, summaries, bindings)
            # All families passed. Publish touched source metadata once; the
            # shared apply authority owns ordering and any capacity flush.
            if bank_id is not None:
                self._details["source_banks"][bank_id] = bank_refs
                self._details["configurations"].update(contents)
                if bank_cache is not None:
                    self._source_cache = [bank_cache]
            self._apply_prepared(starts, traces, prepared, replay)
        except BaseException as error:
            self._record_failure(error)
            raise

    def _apply_prepared(
        self,
        starts: list[tuple[int, dict[str, object], str, dict[str, object]]],
        traces: list[
            tuple[
                int,
                int,
                np.ndarray[Any, Any],
                np.ndarray[Any, Any],
                dict[str, str],
                str,
                dict[str, object],
            ]
        ],
        completions: list[
            tuple[int, str, dict[str, object], list[tuple[str, list[object]]], str]
        ],
        replay: _PreparedReplay | None,
    ) -> None:
        """Publish a fully checked call in start, replay, trace and completion order.

        Both direct and compact writes use this boundary. Recording joins have
        passed preflight. Replay artifact construction finishes for the entire
        incoming family before final files or table rows are published. File
        failures retain normal recovery rules. Capacity flushes cannot precede
        the complete incoming replay family's validation.
        """
        entry = self._details["passes"][self._pass_key]
        for episode_id, declaration, config_id, config in starts:
            entry["episode_starts"][str(episode_id)] = declaration
            self._pending_numerical_starts.discard(episode_id)
            self._details["configurations"].setdefault(config_id, config)
        if replay is not None:
            self._consume_replay(replay)
        for trace in traces:
            self._append_trace(*trace)
        for episode_id, config_id, config, rows, coverage in completions:
            if (
                self._collector is not None
                and episode_id in self._collector.pending_episode_ids
            ):
                raise ValueError(
                    f"episode {episode_id} completed before its replay packets"
                )
            self._details["configurations"].setdefault(config_id, config)
            for filename, row in rows:
                self._rows[filename].append(row)
            entry = self._details["passes"][self._pass_key]
            entry["recorded_metrics_by_episode"][str(episode_id)] = coverage
            entry["completed_config_ids"][str(episode_id)] = config_id
            key = (self._pass_key, episode_id)
            self._completed.add(key)
            self._pending.append(key)
            if len(self._pending) >= self._buffer_size:
                self.flush()

    def _preflight_replays(
        self,
        packets: ReplayPackets | None,
        completed: dict[int, tuple[int, int, int, int]],
        bindings: dict[int, str],
    ) -> _PreparedReplay | None:
        """Check replay order, completion joins and initial identities before output.

        ReplayCollector owns packet ordering. Return contexts and config identities
        for later application, without creating a live collector, spool or writer
        metadata. Existing bindings and schedule digests must match actual configs.
        """
        import jax

        from marl_battlegrounds.evaluation.replay_recording import ReplayCollector

        entry = self._details["passes"][self._pass_key]
        for episode_id, expected_config in bindings.items():
            replay_record = self._pending_replays.get(str(episode_id))
            if replay_record is None:
                replay_record = cast(
                    dict[str, object], entry["replays"].get(str(episode_id), {})
                )
            actual_config = replay_record.get(
                "config_id", self._replay_config_ids.get(episode_id)
            )
            if actual_config is not None and actual_config != expected_config:
                raise ValueError("record configuration differs from its replay")
        for episode_id, expected in completed.items():
            recorded = self._pending_replays.get(str(episode_id))
            if recorded is None:
                recorded = cast(
                    dict[str, object], entry["replays"].get(str(episode_id), {})
                )
            if (
                "summary" in recorded
                and tuple(cast(Iterable[int], recorded["summary"])) != expected
            ):
                raise ValueError("completed summary differs from its recorded replay")

        if packets is None:
            if (
                self._collector is not None
                and set(completed) & self._collector.pending_episode_ids
            ):
                raise ValueError("episode completed before its replay packets")
            return None
        checker = self._collector
        temporary = checker is None
        if checker is None:
            checker = ReplayCollector(self._replay_context)
        try:
            host = checker.preflight(packets, completed_episode_ids=completed)
        finally:
            if temporary:
                checker.close()
        for episode_id in np.unique(
            np.asarray(host.episode_id)[np.asarray(host.valid)]
        ):
            if (self._pass_key, int(episode_id)) in self._completed:
                raise ValueError(
                    f"replay episode {int(episode_id)} was already completed"
                )
        for episode_id, actual in _replay_summaries(host).items():
            if episode_id in completed and completed[episode_id] != actual:
                raise ValueError("completed summary differs from its terminal replay")
        shape = np.shape(host.valid)

        def flatten(value: object) -> np.ndarray[Any, Any]:
            """Retain replay payload axes while flattening its own leading axes."""
            array = np.asarray(value)
            return array.reshape((-1, *array.shape[len(shape) :]))

        rows = jax.tree.map(flatten, host)
        entry = self._details["passes"][self._pass_key]
        contexts: dict[int, tuple[EvaluationEpisodeContextV4, RuntimeProvenanceV1]] = {}
        replay_configs: dict[int, str] = {}
        provenance = None
        for index in np.flatnonzero(np.asarray(rows.valid) & np.asarray(rows.initial)):
            packet = cast(
                "ReplayPackets",
                jax.tree.map(partial(_host_row, index=int(index)), rows),
            )
            episode_id = int(packet.episode_id)
            config_id, _ = configuration_identity(packet.config)
            episode = entry["episodes"].get(str(episode_id), {})
            start = entry["episode_starts"].get(str(episode_id), {})
            if start.get("verification") == "pending" and episode_id not in bindings:
                raise ValueError("replay needs verified first-start evidence")
            expected = (
                bindings.get(episode_id),
                episode.get("configuration_digest", episode.get("config_id")),
                start.get("resolved_config_id"),
                entry["trace_config_ids"].get(str(episode_id)),
            )
            if any(value is not None and value != config_id for value in expected):
                raise ValueError(
                    "replay configuration differs from its episode evidence"
                )
            context, runtime, provenance = self._prepare_replay_context(
                packet, provenance
            )
            contexts[episode_id] = context, runtime
            replay_configs[episode_id] = config_id
        return _PreparedReplay(host, contexts, replay_configs, provenance)

    @staticmethod
    def _config_evidence(
        records: EpisodeInfo | _RecordingRows,
        index: int,
        cache: dict[int, tuple[EnvConfig, str, dict[str, object]]],
    ) -> tuple[EnvConfig, str, dict[str, object]]:
        """Hash a supplied host row once for this write call.

        Start, trace and completion checks share this evidence. Each row is
        checked against its own actual configuration; no prior call, mutable
        object identity or assumed episode binding substitutes for its contents.
        The caller releases the cache when the bounded input call returns.
        """
        import jax

        if index not in cache:
            config_index = (
                int(records.config_indices[index])
                if isinstance(records, _RecordingRows)
                else index
            )
            config = jax.tree.map(
                partial(_host_row, index=config_index), records.config
            )
            identifier, content = configuration_identity(config)
            cache[index] = config, identifier, content
        return cache[index]

    def _prepare_record(
        self,
        records: EpisodeInfo | _RecordingRows,
        index: int,
        config_cache: dict[int, tuple[EnvConfig, str, dict[str, object]]],
        *,
        entry: dict[str, Any] | None = None,
        metric_indices: tuple[int, int] | None = None,
    ) -> tuple[int, str, dict[str, object], list[tuple[str, list[object]]], str]:
        """Validate one completion without mutating buffers or metadata.

        Keep exact integer outcomes separate from float32 optional measurements.
        Return its ID, config identity/content, table rows and actual metric mode.
        """
        episode_id = self._integer(
            records.episode_id[index], "completed episode ID", minimum=1
        )
        if (self._pass_key, episode_id) in self._completed:
            raise ValueError(f"episode {episode_id} was already written in this pass")
        outcome = self._integer(records.outcome[index], "outcome", minimum=1)
        if outcome not in (1, 2, 3):
            raise ValueError("outcome must be terminal code 1, 2 or 3")
        length = self._integer(
            records.episode_length[index], "episode_length", minimum=1
        )
        decision = self._integer(
            records.decision_step[index], "decision_step", minimum=0
        )
        if decision != length - 1:
            raise ValueError(
                "completed decision_step must equal episode_length minus one"
            )
        scores = np.asarray(records.team_scores[index])
        if scores.shape != (2,):
            raise ValueError("team_scores must contain Team A and Team B")
        score_a, score_b = (self._integer(v, "team score", minimum=0) for v in scores)
        config_tree, config_id, config = self._config_evidence(
            records, index, config_cache
        )
        entry = cast(
            dict[str, Any],
            self._details["passes"][self._pass_key] if entry is None else entry,
        )
        episode = entry["episodes"].get(str(episode_id), {})
        if entry["trace_epochs"].get(str(episode_id), -1) > decision:
            raise ValueError("completion precedes an already recorded trace decision")
        if (
            episode.get("configuration_digest", episode.get("config_id", config_id))
            != config_id
        ):
            raise ValueError("completion config differs from the recorded schedule")
        if entry["trace_config_ids"].get(str(episode_id), config_id) != config_id:
            raise ValueError("completion config differs from recorded trace ownership")
        bound = entry["episode_starts"].get(str(episode_id), {})
        if bound.get("resolved_config_id", config_id) not in (None, config_id):
            raise ValueError("completion config differs from its episode start")
        if (
            bound
            and bound["verification"] == "pending"
            and (
                records.episode_start_records is None
                or not np.any(
                    np.asarray(records.episode_start_records.valid)
                    & (
                        np.asarray(records.episode_start_records.episode_id)
                        == episode_id
                    )
                )
            )
        ):
            raise ValueError("completion lacks verified first-start evidence")
        if isinstance(records, _RecordingRows):
            active = np.asarray(config_tree.agent_profile.active_mask)
            classes = np.asarray(config_tree.agent_profile.class_ids)
        else:
            active = np.asarray(records.active_mask[index])
            classes = np.asarray(records.class_ids[index])
        if (
            active.shape != (10,)
            or active.dtype != np.bool_
            or classes.shape != (10,)
            or classes.dtype != np.int32
        ):
            raise ValueError(
                "recorded roster must contain ten bool active and int32 class slots"
            )
        if not np.array_equal(
            active, config_tree.agent_profile.active_mask
        ) or not np.array_equal(classes, config_tree.agent_profile.class_ids):
            raise ValueError("recorded roster differs from the actual configuration")

        def policy_name(team: str) -> object:
            """Read an episode override label, otherwise the pass label."""
            definition = episode.get("policies", entry["policies"]).get(team, "")
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
            identity.extend((int(classes[slot]), int(active[slot])))
        cells_by_mode: dict[str, list[object]] = {}
        indices = (index, index) if metric_indices is None else metric_indices
        for mode, values, names, metric_index in (
            ("priority", records.priority, PRIORITY_METRIC_NAMES, indices[0]),
            ("full", records.full, FULL_METRIC_NAMES, indices[1]),
        ):
            if values is not None and metric_index >= 0:
                cells = self._metric_cells(values, metric_index, names)
                if np.any(values.valid[metric_index]):
                    if not np.all(
                        values.valid[metric_index][: len(PRIORITY_METRIC_NAMES)]
                    ):
                        raise ValueError(
                            "completed priority measurements must all be available"
                        )
                    cells_by_mode[mode] = cells
        if "full" in cells_by_mode:
            full_prefix = cells_by_mode["full"][: len(PRIORITY_METRIC_NAMES)]
            if "priority" in cells_by_mode and full_prefix != cells_by_mode["priority"]:
                raise ValueError("priority and full measurements disagree")
            cells_by_mode.setdefault("priority", full_prefix)
        summary = {
            "episode_length": length,
            "team_a_score": score_a,
            "team_b_score": score_b,
        }
        expected_measurements = {
            **summary,
            **{
                name: int(outcome == code)
                for name, code in PRIORITY_OUTCOME_CODES.items()
            },
        }
        for mode, cells in cells_by_mode.items():
            names = PRIORITY_METRIC_NAMES if mode == "priority" else FULL_METRIC_NAMES
            for name, value in expected_measurements.items():
                cell = cells[names.index(name)]
                if cell != "" and cell != float(np.float32(value)):
                    raise ValueError(
                        f"{name} measurement differs from the required summary"
                    )
        rows: list[tuple[str, list[object]]] = []
        tournament = entry["phase"] == "tournament"
        if tournament:
            if "tournament_summary" in self._details:
                raise ValueError(
                    "a qualified tournament cannot accept additional matches"
                )
            block_id = self._integer(
                episode.get("block_id"), "registered block_id", minimum=1
            )
            cells = cells_by_mode.get(
                "priority", [""] * len(PRIORITY_METRIC_NAMES)
            ).copy()
            for name, value in summary.items():
                cells[PRIORITY_METRIC_NAMES.index(name)] = value
            rows.append(
                (
                    "match_results.csv",
                    [
                        *identity,
                        block_id,
                        episode.get("bootstrap_group") or "",
                        outcome,
                        *cells,
                    ],
                )
            )
        else:
            rows.append(
                ("episodes.csv", [*identity, outcome, length, score_a, score_b])
            )
        for mode, cells in cells_by_mode.items():
            if mode == "priority" and tournament:
                continue
            rows.append((f"{mode}_metrics.csv", [*identity, *cells]))
        coverage = (
            "full"
            if "full" in cells_by_mode
            else "priority"
            if "priority" in cells_by_mode
            else "none"
        )
        return episode_id, config_id, config, rows, coverage

    @staticmethod
    def _integer(value: object, name: str, *, minimum: int) -> int:
        """Read an int32-range host integer; never coerce floats."""
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in "iu":
            raise ValueError(f"{name} must be an integer")
        result = int(array)
        if result < minimum or result > np.iinfo(np.int32).max:
            raise ValueError(f"{name} must be between {minimum} and the int32 limit")
        return result

    def _prepare_traces(
        self,
        records: EpisodeInfo | _RecordingRows,
        trace: object,
        leading: tuple[int, ...],
        config_cache: dict[int, tuple[EnvConfig, str, dict[str, object]]],
        *,
        entry: dict[str, Any] | None = None,
    ) -> list[
        tuple[
            int,
            int,
            np.ndarray[Any, Any],
            np.ndarray[Any, Any],
            dict[str, str],
            str,
            dict[str, object],
        ]
    ]:
        """Check every trace join before output and return ordered decision groups.

        Scalar info alone accepts a batch-one trace. Each valid decision advances
        its episode's saved epoch and indexes the immutable team component table.
        """
        if trace is None:
            return []
        import jax

        from marl_battlegrounds.evaluation.policy_execution import PolicyTrace

        if not isinstance(trace, PolicyTrace):
            raise TypeError("policy_trace must be a PolicyTrace")
        host = jax.device_get(trace)
        scalar_batch = leading == () and np.shape(host.valid) == (1,)
        arrays: dict[str, np.ndarray[Any, Any]] = {}
        for name in PolicyTrace._fields:
            value = np.asarray(getattr(host, name))
            shape = (*leading, 10) if name == "policy_ids" else leading
            if scalar_batch:
                if value.shape != (1, *shape):
                    raise ValueError(
                        f"scalar policy_trace.{name} must have exactly one batch row"
                    )
                value = value[0]
            if value.shape != shape or value.dtype != (
                np.bool_ if name == "valid" else np.int32
            ):
                raise ValueError(f"policy_trace.{name} has the wrong shape or dtype")
            arrays[name] = value.reshape((-1, 10) if name == "policy_ids" else (-1,))
        entry = cast(
            dict[str, Any],
            self._details["passes"][self._pass_key] if entry is None else entry,
        )
        epochs: dict[str, int] = {}
        config_ids: dict[str, str] = {}
        result: list[
            tuple[
                int,
                int,
                np.ndarray[Any, Any],
                np.ndarray[Any, Any],
                dict[str, str],
                str,
                dict[str, object],
            ]
        ] = []
        for index in np.flatnonzero(arrays["valid"]):
            episode_id = self._integer(
                arrays["episode_id"][index], "trace episode ID", minimum=1
            )
            decision = self._integer(
                arrays["decision_step"][index], "trace decision_step", minimum=0
            )
            if episode_id != int(records.episode_id[index]) or decision != int(
                records.decision_step[index]
            ):
                raise ValueError("trace episode and action epoch must match info")
            if (self._pass_key, episode_id) in self._completed:
                raise ValueError("trace follows an already completed episode")
            previous_epoch = epochs.get(str(episode_id))
            if previous_epoch is None:
                previous_epoch = int(entry["trace_epochs"].get(str(episode_id), -1))
            if decision <= previous_epoch:
                raise ValueError("trace decisions must move forward without duplicates")
            config_tree, config_id, config = self._config_evidence(
                records, int(index), config_cache
            )
            active = np.asarray(
                config_tree.agent_profile.active_mask
                if isinstance(records, _RecordingRows)
                else records.active_mask[index]
            )
            choices = arrays["policy_ids"][index]
            if active.shape != (10,) or active.dtype != np.bool_:
                raise ValueError("trace roster must have ten boolean active slots")
            if not np.array_equal(active, config_tree.agent_profile.active_mask):
                raise ValueError("trace roster differs from the actual configuration")
            if np.any(choices[~active] != -1):
                raise ValueError("inactive slots must have policy ID -1")
            episode = entry["episodes"].get(str(episode_id), {})
            system_ids = episode.get("system_ids", entry["system_ids"])
            for slot in np.flatnonzero(active):
                team = "team_a" if slot < 5 else "team_b"
                system_id = system_ids.get(team)
                if system_id not in self._details["systems"]:
                    raise ValueError(f"trace needs a registered {team} System")
                count = len(self._details["systems"][system_id]["components"])
                if choices[slot] < -1 or choices[slot] >= count:
                    raise ValueError(
                        f"policy ID for slot {slot} is outside its component table"
                    )
            start_binding = entry["episode_starts"].get(str(episode_id), {})
            expected = (
                episode.get("configuration_digest", episode.get("config_id")),
                start_binding.get("resolved_config_id"),
            )
            if any(v is not None and v != config_id for v in expected):
                raise ValueError(
                    "trace configuration differs from recorded episode evidence"
                )
            if int(records.episode_length[index]) != decision + 1:
                raise ValueError(
                    "trace decision does not match the played episode length"
                )
            previous_config = config_ids.get(
                str(episode_id), entry["trace_config_ids"].get(str(episode_id))
            )
            if previous_config is not None and previous_config != config_id:
                raise ValueError("trace configuration changed within an episode")
            config_ids[str(episode_id)] = config_id
            epochs[str(episode_id)] = decision
            result.append(
                (episode_id, decision, choices, active, system_ids, config_id, config)
            )
        return result

    def _append_trace(
        self,
        episode_id: int,
        decision: int,
        choices: np.ndarray[Any, Any],
        active: np.ndarray[Any, Any],
        system_ids: dict[str, str],
        config_id: str,
        config: dict[str, object],
    ) -> None:
        """Append one validated decision without splitting it across durable commits.

        Open and closed segments share one capacity of ten times buffer_size.
        Flush releases all intervals, including games abandoned before completion.
        """
        capacity = 10 * self._buffer_size
        additions = 0
        for slot in np.flatnonzero(active):
            previous = self._assignment_open.get((episode_id, int(slot)))
            if (
                previous is None
                or previous[7] != int(choices[slot])
                or previous[9] != decision
            ):
                additions += 1
        if (
            len(self._assignment_open)
            + len(self._rows["policy_assignments.csv"])
            + additions
            > capacity
        ):
            self.flush()
        entry = self._details["passes"][self._pass_key]
        for slot_value in np.flatnonzero(active):
            slot = int(slot_value)
            key = (episode_id, slot)
            previous = self._assignment_open.get(key)
            policy_id = int(choices[slot])
            if (
                previous is not None
                and previous[7] == policy_id
                and previous[9] == decision
            ):
                previous[9] = decision + 1
                continue
            if previous is not None:
                self._rows["policy_assignments.csv"].append(previous)
            team = "team_a" if slot < 5 else "team_b"
            self._assignment_open[key] = [
                self.run_id,
                entry["phase"],
                entry["pass_id"],
                episode_id,
                team,
                slot,
                system_ids[team],
                policy_id,
                decision,
                decision + 1,
            ]
        entry["trace_config_ids"][str(episode_id)] = config_id
        self._details["configurations"].setdefault(config_id, config)
        entry["trace_epochs"][str(episode_id)] = decision

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
        if (
            numeric.shape != (len(names),)
            or valid.shape != numeric.shape
            or valid.dtype != np.bool_
            or numeric.dtype != np.float32
        ):
            raise ValueError("measurements do not match the scalar column schema")
        if not np.isfinite(numeric[valid]).all():
            raise ValueError("available measurements contain a nonfinite value")
        cells = numeric.astype(object)
        cells[~valid] = ""
        return cast(list[object], cells.tolist())

    def _install_tournament_plan(
        self,
        config: Mapping[str, Any],
        games: Sequence[Mapping[str, Any]],
        reuse: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Durably attach immutable canonical references under this writer's lock.

        The canonical runner validates science, source coverage and resume before
        opening this writer. This method checks exact saved bytes and the additive
        version again, writes missing immutable config/JSONL files, synchronizes
        their directory, then publishes their hashes with the existing manifest
        transaction. Unreferenced files from an interrupted attempt remain inert.
        Conflicting files/metadata fail before replacement. It returns the saved
        reference mapping and never adds table roles to a training checkpoint.
        """
        self._check_open()
        if (
            reuse.get("version") != 1
            or self._details["passes"][self._pass_key].get("phase") != "tournament"
        ):
            raise ValueError(
                "canonical references require version 1 and a tournament pass"
            )
        payloads = {
            "tournament_config.json": _json_bytes(config),
            "tournament_games.jsonl": b"".join(_json_bytes(game) for game in games),
        }
        resolved = dict(reuse)
        resolved["config_sha256"] = sha256(
            payloads["tournament_config.json"]
        ).hexdigest()
        resolved["games_sha256"] = sha256(
            payloads["tournament_games.jsonl"]
        ).hexdigest()
        previous = self._details.get("tournament_reuse")
        if previous is not None:
            if {
                key: value
                for key, value in previous.items()
                if key not in {"state", "asset_locations"}
            } != {
                key: value
                for key, value in resolved.items()
                if key not in {"state", "asset_locations"}
            }:
                raise ValueError(
                    "canonical tournament references differ from the saved run"
                )
            resolved["state"] = previous["state"]
            if "asset_locations" in previous:
                resolved["asset_locations"] = previous["asset_locations"]
        for filename, payload in payloads.items():
            path = self.run_dir / filename
            if path.is_symlink() or (path.exists() and path.read_bytes() != payload):
                raise ValueError(
                    "immutable canonical reference already has different content"
                )
        for filename, payload in payloads.items():
            path = self.run_dir / filename
            if path.exists():
                continue
            temporary = path.with_name("." + path.name + ".tmp")
            with temporary.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        os.fsync(self._lock)
        self._details["tournament_reuse"] = resolved
        self.flush()
        return resolved

    def _set_tournament_asset_locations(self, locations: Mapping[str, Any]) -> None:
        """Publish only changed canonical asset hints after runner preflight.

        Immutable config bytes and scientific hashes remain unchanged. The caller
        has verified required contents before opening this writer. Validate the
        known asset IDs and hint shapes here, then use the existing flush boundary.
        An identical map is a no-op. This method never downloads or loads models.
        """
        from marl_battlegrounds.evaluation.tournament_assets import (
            asset_location_config,
        )
        from marl_battlegrounds.evaluation.tournament_config import read_config_json

        self._check_open()
        reuse = self._details.get("tournament_reuse")
        if reuse is None or reuse.get("version") != 1:
            raise ValueError("asset hints require an installed canonical plan")
        config = read_config_json(self.run_dir / "tournament_config.json")
        asset_location_config(config, locations)
        if reuse.get("asset_locations", {}) != locations:
            reuse["asset_locations"] = _json_value(locations)
            self.flush()

    def _use_tournament_records(self, records: TournamentRecords) -> None:
        """Share the runner's original-record accessor for version-2 publication.

        The accessor must describe this run and its attached immutable game plan.
        It is borrowed only for final qualification, never serialized or used by
        numerical execution. Calling this method does not open files or flush.
        """
        self._check_open()
        if records.manifest.get("run_id") != self.run_id or not self._details.get(
            "tournament_reuse"
        ):
            raise ValueError("canonical record accessor belongs to a different run")
        reuse = self._details["tournament_reuse"]
        if (
            sha256(_json_bytes(records.config)).hexdigest() != reuse["config_sha256"]
            or sha256(b"".join(_json_bytes(game) for game in records.games)).hexdigest()
            != reuse["games_sha256"]
        ):
            raise ValueError(
                "canonical record accessor differs from the immutable run plan"
            )
        self._canonical_records = records

    def write_tournament_results(
        self,
        statistics: TournamentStatistics,
        *,
        headline: Sequence[Mapping[str, object]] | None = None,
        qualification: Mapping[str, object] | None = None,
    ) -> None:
        """Publish a qualified tournament's summary tables exactly once by content.

        Parameters
        ----------
        statistics : TournamentStatistics
            TournamentStatistics from the statistical qualification helper,
            containing nonempty population, matchup and map result tables.
        headline : tuple of dict or None, optional
            Already computed complete participant rows, default None. The
            headline reducer owns their arithmetic. None mode omits this file.
        qualification : Mapping or None, optional
            Verified population/schedule evidence, default None for legacy
            callers. New tournament runners provide version 1, the schedule
            digest, participant IDs, pairing protocol, metric mode and status.

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
        payload: dict[str, object] = {
            "tournament_results": statistics.tournament_results,
            "matchup_results": statistics.matchup_results,
            "map_results": statistics.map_results,
            "metadata": statistics.metadata,
        }
        qualified: dict[str, Any] = {}
        evidence: Mapping[str, Any] = {}
        if qualification is not None:
            supplied_evidence = qualification.get("evidence")
            if not isinstance(supplied_evidence, Mapping):
                raise ValueError(
                    "tournament qualification requires prepared game evidence"
                )
            evidence = cast(Mapping[str, Any], supplied_evidence)
            qualified = cast(
                dict[str, Any],
                _json_value(
                    {
                        key: value
                        for key, value in qualification.items()
                        if key != "evidence"
                    }
                ),
            )
            # The existing run already owns configs, systems and passes. Keep a
            # compact join identity, not another copy of that entire manifest.
            qualified["game_evidence_digest"] = sha256(
                _json_bytes(
                    _json_value(
                        {
                            "games": evidence.get("games"),
                            "match_digest": evidence.get("match_digest"),
                            "schedule_digest": evidence.get("schedule_digest"),
                        }
                    )
                )
            ).hexdigest()
            population = qualified.get("population_system_ids")
            if (
                qualified.get("version") != 1
                or qualified.get("status") != "complete"
                or not isinstance(qualified.get("schedule_digest"), str)
                or not qualified["schedule_digest"]
                or not isinstance(population, list)
                or not population
                or any(
                    not isinstance(value, str) or not value
                    for value in cast(list[object], population)
                )
                or len(set(cast(list[str], population)))
                != len(cast(list[str], population))
                or qualified.get("metrics") not in {"none", "priority", "full"}
            ):
                raise ValueError("invalid tournament completion qualification")
            if qualified["metrics"] == "none" and headline is not None:
                raise ValueError(
                    "none-mode tournaments cannot publish headline metrics"
                )
            if qualified["metrics"] != "none" and headline is None:
                raise ValueError(
                    "measured tournaments require complete headline metrics"
                )
            if headline is not None and sorted(
                str(row.get("system_id")) for row in headline
            ) != sorted(cast(list[str], population)):
                raise ValueError(
                    "headline rows must cover every participant exactly once"
                )
            payload["qualification"] = qualified
            for entry in self._details["passes"].values():
                if (
                    entry.get("pass_role") == "tournament_coordinator"
                    and entry.get("result_state", {}).get("schedule_digest")
                    != qualified["schedule_digest"]
                ):
                    raise ValueError(
                        "coordinator and tournament schedule identities differ"
                    )
        elif headline is not None:
            raise ValueError("headline metrics require tournament qualification")
        if headline is not None:
            payload["tournament_headline_metrics"] = headline
        digest = sha256(_json_bytes(_json_value(payload))).hexdigest()
        previous = self._details.get("tournament_summary")
        if previous is not None:
            if previous["digest"] != digest:
                raise ValueError(
                    "tournament summary differs from the qualified population"
                )
            return
        tables: dict[str, tuple[tuple[str, ...], list[list[object]]]] = {}
        filenames = _SUMMARY_TABLES
        if headline is not None:
            filenames = (*filenames, "tournament_headline_metrics.csv")
        for filename in filenames:
            rows = cast(tuple[dict[str, object], ...], payload[Path(filename).stem])
            if not rows:
                raise ValueError("qualified tournament summary tables must be nonempty")
            names = tuple(rows[0])
            if filename == "tournament_headline_metrics.csv":
                from marl_battlegrounds.evaluation.tournament_headlines import (
                    HEADLINE_COLUMNS,
                )

                if names != HEADLINE_COLUMNS:
                    raise ValueError("headline columns differ from the shared schema")
            if any(set(row) != set(names) for row in rows):
                raise ValueError("tournament summary rows must have the same columns")
            tables[filename] = names, [[row[name] for name in names] for row in rows]
        if qualification is not None:
            from marl_battlegrounds.evaluation.scalar_reports import iter_scalar_rows
            from marl_battlegrounds.evaluation.tournament_headlines import (
                validate_evidence,
            )

            if evidence.get("version") == 2:
                if self._canonical_records is None:
                    raise ValueError(
                        "canonical publication needs its shared record accessor"
                    )
                if self._rows["match_results.csv"]:
                    raise ValueError(
                        "flush local canonical games before summary qualification"
                    )
                matches: list[dict[str, object]] = [
                    {name: row.get(name) for name in MATCH_COLUMNS}
                    for batch in self._canonical_records.iter_rows(
                        "match_results.csv",
                        columns=(
                            *IDENTITY_COLUMNS,
                            "block_id",
                            "bootstrap_group",
                            "outcome",
                            "episode_length",
                            "team_a_score",
                            "team_b_score",
                        )
                        if qualified["metrics"] == "none"
                        else None,
                    )
                    for row in batch
                ]
            else:
                matches = [
                    dict(row)
                    for batch in iter_scalar_rows(
                        self.run_dir / "match_results.csv",
                        manifest=self._details,
                        expected_header=MATCH_COLUMNS,
                    )
                    for row in batch
                ]
                matches.extend(
                    dict(zip(MATCH_COLUMNS, row, strict=True))
                    for row in self._rows["match_results.csv"]
                )
            validate_evidence(matches, evidence)
            if evidence.get("schedule_digest") != qualified[
                "schedule_digest"
            ] or sorted(
                cast(list[str], evidence.get("population_system_ids", []))
            ) != sorted(qualified["population_system_ids"]):
                raise ValueError(
                    "tournament summary population differs from prepared evidence"
                )
        self.flush()
        for filename, (names, rows) in tables.items():
            self._headers[filename] = names
            self._rows[filename] = rows
        self._details["tournament_summary"] = {
            "digest": digest,
            "metadata": _json_value(statistics.metadata),
        }
        if qualification is not None:
            marker = {**qualified, "summary_digest": digest}
            self._details["tournament_summary"]["qualification"] = marker
            for entry in self._details["passes"].values():
                if entry.get("pass_role") == "tournament_coordinator":
                    entry["result_state"] = {
                        "version": 1,
                        "status": "complete",
                        "reason": None,
                        "schedule_digest": qualified["schedule_digest"],
                    }
        if "tournament_reuse" in self._details:
            self._details["tournament_reuse"]["state"] = "complete"
        self.flush()

    def checkpoint_recording(self) -> dict[str, object]:
        """Make the current training recording boundary durable and return its token.

        Returns
        -------
        dict[str, object]
            Versioned recording identity to save atomically with the complete
            learner carry, parameters, optimizer, RNG and stage progress. This
            token does not save learning state and is not a backup by itself.

        Raises
        ------
        ValueError
            The run does not contain exactly one training pass, a numerical
            start still lacks first-transition evidence, or collection is active.
        RuntimeError
            The writer is closed or failed.
        OSError
            Publication fails. No usable new token is returned; earlier tokens
            remain valid subject to their retained files and table prefixes.

        Notes
        -----
        Host-only. Flushes pending records and snapshots unfinished replay
        prefixes with bounded copies. Preserve each returned bundle and its
        referenced replay files while its learner checkpoint is needed. Resume
        with the explicit token and original registered System descriptions.
        """
        from marl_battlegrounds.evaluation.recording_checkpoint import create_checkpoint

        return create_checkpoint(self)

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
            The readable atomic metadata file is the recovery authority. A failure
            before its replacement leaves the old boundary; a failure after
            replacement may leave the new one. Resume removes table suffixes not
            covered by that published boundary. A failed directory fsync does not
            promise that the replacement survives a machine or power failure.
        """
        self._check_open()
        self._rows["policy_assignments.csv"].extend(self._assignment_open.values())
        self._assignment_open.clear()
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
        hashes = (
            None
            if self._checkpoint_hashes is None
            else {
                name: digest.copy() for name, digest in self._checkpoint_hashes.items()
            }
        )
        try:
            for filename, rows in self._rows.items():
                if not rows:
                    continue
                previous = candidate["tables"].get(
                    filename, {"durable_bytes": 0, "rows": 0}
                )
                path = self._table_path(filename)
                with path.open("ab") as stream:
                    for start in range(0, len(rows), self._buffer_size):
                        text = io.StringIO(newline="")
                        writer = csv.writer(text, lineterminator="\n")
                        if previous["durable_bytes"] == 0 and start == 0:
                            writer.writerow(self._headers[filename])
                        writer.writerows(rows[start : start + self._buffer_size])
                        data = text.getvalue().encode("utf-8")
                        stream.write(data)
                        if hashes is not None:
                            hashes.setdefault(filename, sha256()).update(data)
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
        self._checkpoint_hashes = hashes
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
            # Use the published boundary, never pending rows from a failed
            # flush. Diagnostics must not accidentally publish undurable data.
            try:
                path = self.run_dir / "run_details.json"
                durable = json.loads(path.read_bytes())
                changed = False
                for key, entry in durable.get("passes", {}).items():
                    if (
                        key == self._pass_key
                        or entry.get("pass_role") == "tournament_coordinator"
                    ):
                        marker = entry.get("result_state")
                        if marker is not None:
                            entry["result_state"] = {
                                **marker,
                                "status": "failed",
                                "reason": f"{type(error).__name__}: {error}",
                            }
                            changed = True
                if changed:
                    _atomic_json(path, durable)
                    os.fsync(self._lock)
            except OSError, ValueError:
                pass
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


def _prepare_pass_identity(
    manifest: Mapping[str, Any],
    phase: str,
    pass_id: str,
    policies: dict[str, object] | None,
    checkpoint_id: str | None,
    details: dict[str, object] | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Check proposed pass identity against a saved manifest without changing it.

    manifest is the already-read run description. Remaining arguments are the
    same facts accepted by RunWriter.start_pass. Return its pass key, normalized
    identity and System registrations. Changed stable facts raise ValueError;
    runtime placement, batch size and chunk size may differ. This does no I/O,
    method initialization or action call. Writer and read-only verification use
    this single identity rule.
    """
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
    )

    if (
        not isinstance(cast(object, phase), str)
        or not phase
        or not isinstance(cast(object, pass_id), str)
        or not pass_id
    ):
        raise ValueError("phase and pass_id must be nonempty strings")
    descriptions: dict[str, object] = {}
    systems: dict[str, Any] = {}
    system_ids: dict[str, str] = {}
    for team, value in (policies or {}).items():
        identifier, registration = normalize_system_registration(value, phase=phase)
        systems[identifier] = registration
        system_ids[team] = identifier
        # Legacy serialized Policy descriptions retain their exact public fields.
        descriptions[team] = (
            _json_value(cast(object, value))
            if isinstance(value, (str, dict))
            else registration
        )
    identity: dict[str, Any] = {
        "phase": phase,
        "pass_id": pass_id,
        "policies": descriptions,
        "system_ids": system_ids,
        "checkpoint_id": checkpoint_id,
        "details": {} if details is None else _json_value(details),
        "policy_state": "frozen"
        if systems and all(v["parameter_status"] == "frozen" for v in systems.values())
        else "evolving",
    }
    key = json.dumps((phase, pass_id), separators=(",", ":"))
    previous = manifest["passes"].get(key)
    execution_fields = {"runtime_provenance", "num_envs", "chunk_size"}
    if previous is not None:
        for name, value in identity.items():
            old = previous.get(name)
            if name == "details":
                value = {k: v for k, v in value.items() if k not in execution_fields}
                old = {
                    k: v
                    for k, v in cast(dict[str, Any], old or {}).items()
                    if k not in execution_fields
                }
            if old != value:
                raise ValueError("pass identity differs from the recorded run")
    return key, identity, systems
