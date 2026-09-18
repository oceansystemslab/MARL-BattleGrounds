"""Save and restore explicit recording boundaries beside learner checkpoints.

RunWriter owns locking, pass selection, CSV transactions and public methods.
This private module owns versioned bundles, file integrity and unfinished replay
prefixes. Tokens contain no learner state and cannot recover missing CSV history.
All work is host-only and begins only when checkpointing is explicitly requested.
"""

# These helpers are part of RunWriter's implementation, not external clients.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Any, BinaryIO, cast
from uuid import uuid4

import jax
import numpy as np

from marl_battlegrounds.evaluation.replay_capture import ReplayPackets
from marl_battlegrounds.evaluation.replay_recording import (
    ReplayCollector,
    validate_packet_epoch,
)

if TYPE_CHECKING:
    from _hashlib import HASH

    from marl_battlegrounds.core.types import EnvConfig
    from marl_battlegrounds.evaluation.run_writer import RunWriter

COPY_BLOCK_BYTES = 1024 * 1024
_TOKEN_KEYS = {
    "schema_version",
    "run_id",
    "pass_key",
    "checkpoint_id",
    "checkpoint_sha256",
}
_BUNDLE_KEYS = {
    "schema_version",
    "run_id",
    "pass_key",
    "host_schema_version",
    "metric_schema_id",
    "metric_schema_version",
    "actor_input_projections",
    "source",
    "replay_layout_version",
    "files",
    "tables",
    "open_replays",
}
_OPEN_REPLAY_KEYS = {
    "episode_id",
    "context_id",
    "count",
    "context",
    "runtime",
    "path",
    "bytes",
    "sha256",
    "layout",
    "config_id",
}


def _object(value: object, name: str) -> dict[str, Any]:
    """Require a JSON object, retaining its checked string-keyed values."""
    if not isinstance(value, dict) or any(
        not isinstance(k, str) for k in cast(dict[object, object], value)
    ):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    """Read an exact non-Boolean integer at or above the declared minimum."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _hex(value: object, length: int, name: str) -> str:
    """Require an exact lowercase hexadecimal identifier or digest."""
    if (
        not isinstance(value, str)
        or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None
    ):
        raise ValueError(f"{name} must contain {length} lowercase hex characters")
    return value


def _json(path: Path) -> dict[str, Any]:
    """Read an object with no duplicate keys or non-finite JSON constants."""

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        """Reject repeated keys instead of silently trusting the last value."""
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        """Reject JSON extensions such as NaN and Infinity."""
        raise ValueError(f"invalid JSON constant: {value}")

    return _object(
        json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=constant),
        path.name,
    )


def _regular(root: Path, relative: str) -> Path:
    """Resolve a known relative file beneath root without following symlinks."""
    name = Path(relative)
    if (
        name.is_absolute()
        or not name.parts
        or any(p in (".", "..") for p in name.parts)
    ):
        raise ValueError("checkpoint file path is unsafe")
    current = root
    if current.is_symlink():
        raise ValueError("checkpoint root must not be a symbolic link")
    for part in name.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("checkpoint paths must not use symbolic links")
    if not current.is_file():
        raise ValueError(f"checkpoint file is missing: {relative}")
    return current


def _sync_directory(path: Path) -> None:
    """Make directory entries durable on the supported local filesystem."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory_ancestors(path: Path) -> None:
    """Persist path and its parent entries, from deepest directory to root.

    A new run may have created several output directories. Syncing only files
    inside the run does not persist those directories' names in their parents.
    Resolve the existing local path, then sync each directory once. This work
    belongs only to a requested checkpoint, never to ordinary steps or writes.
    Filesystem errors propagate; no token may be returned before this succeeds.
    """
    directory = path.resolve(strict=True)
    for ancestor in (directory, *directory.parents):
        _sync_directory(ancestor)


def _hash_prefix(path: Path, size: int) -> HASH:
    """Hash exactly size bytes using bounded reads; reject a short prefix."""
    digest = sha256()
    if size == 0:
        return digest
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            block = stream.read(min(COPY_BLOCK_BYTES, remaining))
            if not block:
                raise ValueError(
                    f"checkpoint file is shorter than its boundary: {path.name}"
                )
            digest.update(block)
            remaining -= len(block)
    return digest


def _file_evidence(path: Path) -> dict[str, object]:
    """Describe one immutable file by its exact size and full-file hash."""
    size = path.stat().st_size
    return {"bytes": size, "sha256": _hash_prefix(path, size).hexdigest()}


def _check_file(root: Path, relative: str, evidence: object) -> Path:
    """Verify the complete file against its expected size and SHA256."""
    info = _object(evidence, "file evidence")
    if set(info) != {"bytes", "sha256"}:
        raise ValueError("file evidence has unsupported fields")
    size = _integer(info["bytes"], "file bytes")
    digest = _hex(info["sha256"], 64, "file digest")
    path = _regular(root, relative)
    if path.stat().st_size != size or _hash_prefix(path, size).hexdigest() != digest:
        raise ValueError(f"checkpoint file content differs: {relative}")
    return path


def packet_layout(packet: ReplayPackets) -> list[dict[str, object]]:
    """Describe scalar numerical leaves using installed named-record field paths.

    Return paths, exact NumPy dtype strings and shapes in JAX flatten order.
    The layout is data, not executable code or a serialized PyTreeDef.
    """
    paths = cast(
        list[tuple[tuple[Any, ...], Any]],
        jax.tree_util.tree_flatten_with_path(packet)[0],
    )
    return [
        {
            "path": [
                str(getattr(key, "name", getattr(key, "idx", ""))) for key in path
            ],
            "dtype": np.dtype(value.dtype).str,
            "shape": list(value.shape),
        }
        for path, value in paths
    ]


def _packet_template(config: EnvConfig) -> ReplayPackets:
    """Obtain current packet shapes from Core reset without running a game.

    Config contains validated reconstructed context fields. Abstract evaluation
    uses the installed Core constructors, not stored type names or new game rules.
    It does not execute a transition, draw actions or compile a GPU executable.
    """
    from marl_battlegrounds.core.env import reset
    from marl_battlegrounds.core.types import DoneFlags, Reward

    def shape(value: object) -> jax.ShapeDtypeStruct:
        """Use wrapper-normalized scalar dtypes for the abstract config tree."""
        array = np.asarray(value)
        dtype = (
            np.int32
            if array.dtype.kind in "iu"
            else (np.float32 if array.dtype.kind == "f" else np.bool_)
        )
        return jax.ShapeDtypeStruct(array.shape, dtype)

    shaped = jax.tree.map(shape, config)
    state, obs, mask, info = jax.eval_shape(
        reset, shaped, jax.ShapeDtypeStruct((2,), np.uint32)
    )
    integer = cast(jax.Array, jax.ShapeDtypeStruct((), np.int32))
    boolean = cast(jax.Array, jax.ShapeDtypeStruct((), np.bool_))
    return ReplayPackets(
        boolean,
        integer,
        integer,
        boolean,
        shaped,
        state,
        obs,
        mask,
        state,
        obs,
        mask,
        Reward(cast(jax.Array, jax.ShapeDtypeStruct((10,), np.float32))),
        DoneFlags(boolean, boolean),
        info,
        cast(jax.Array, jax.ShapeDtypeStruct((10, 10), np.bool_)),
    )


def _load_packet(stream: BinaryIO, template: ReplayPackets) -> ReplayPackets:
    """Decode one scalar structured record against the installed packet layout."""
    leaves, tree = cast(tuple[list[Any], Any], jax.tree.flatten(template))
    dtype = np.dtype(
        [(str(i), np.dtype(v.dtype), v.shape) for i, v in enumerate(leaves)]
    )
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, order, stored_dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, order, stored_dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("unsupported replay prefix NumPy record version")
    if shape != () or order or stored_dtype != dtype:
        raise ValueError("replay prefix record has an unsupported shape or dtype")
    raw = stream.read(dtype.itemsize)
    if len(raw) != dtype.itemsize:
        raise ValueError("replay prefix record is truncated")
    record = np.frombuffer(raw, dtype=dtype, count=1).reshape(())
    return cast(
        ReplayPackets, tree.unflatten([record[str(i)] for i in range(len(leaves))])
    )


def _restore_replays(
    writer: RunWriter,
    verified_paths: dict[str, Path],
    records: list[dict[str, Any]],
    saved: dict[str, Any],
) -> ReplayCollector:
    """Validate prefixes and prepare owned anonymous streams without publishing.

    Uses already checked file identities, bounded copies and one decoded packet
    at a time. On failure it closes all prepared streams. Original context and
    runtime remain unchanged. The caller checks every file before this call.
    """
    from marl_battlegrounds.evaluation.catalog import (
        build_resolved_env_config_v1,
        reconstruct_env_config_v1,
    )
    from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV3
    from marl_battlegrounds.evaluation.recording_context import restore_recording_config
    from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
    from marl_battlegrounds.evaluation.run_writer import configuration_identity

    collector = ReplayCollector(writer._replay_context)
    seen: set[int] = set()
    template: ReplayPackets | None = None
    try:
        for record in records:
            episode_id = _integer(record.get("episode_id"), "open replay episode ID", 1)
            if episode_id in seen:
                raise ValueError("checkpoint repeats an open replay episode")
            seen.add(episode_id)
            count = _integer(record.get("count"), "open replay count", 1)
            if count > np.iinfo(np.int32).max:
                raise ValueError("open replay count exceeds the transition limit")
            expected_path = f"open_replays/{episode_id}.npylog"
            if record.get("path") != expected_path:
                raise ValueError("open replay path differs from its episode ID")
            context = EvaluationEpisodeContextV3.model_validate_json(
                json.dumps(record.get("context"))
            )
            runtime = RuntimeProvenanceV1.model_validate_json(
                json.dumps(record.get("runtime"))
            )
            if record.get("context_id") != context.identity.episode_id:
                raise ValueError("open replay context identity differs")
            if context.identity.run_id != saved["run_id"]:
                raise ValueError("open replay belongs to another run")
            if template is None:
                template = _packet_template(reconstruct_env_config_v1(context))
            if record.get("layout") != packet_layout(template):
                raise ValueError("replay prefix uses an unsupported packet layout")
            path = verified_paths[expected_path]
            temporary = TemporaryFile()  # noqa: SIM115 - collector owns successful streams
            try:
                with path.open("rb") as source:
                    while block := source.read(COPY_BLOCK_BYTES):
                        temporary.write(block)
                temporary.seek(0)
                first: ReplayPackets | None = None
                for index in range(count):
                    packet = _load_packet(temporary, template)
                    if (
                        not bool(packet.valid)
                        or int(packet.episode_id) != episode_id
                        or bool(packet.done.done)
                    ):
                        raise ValueError(
                            "open replay prefix has invalid transition order"
                        )
                    validate_packet_epoch(
                        int(packet.episode_id),
                        int(packet.transition_index),
                        expected=index,
                        initial=bool(packet.initial),
                        has_transition=bool(
                            packet.info.transition_facts.has_transition
                        ),
                        completed=False,
                    )
                    if first is None:
                        first = packet
                        identifier, content = configuration_identity(packet.config)
                        if (
                            identifier != record.get("config_id")
                            or saved["configurations"].get(identifier, content)
                            != content
                        ):
                            raise ValueError(
                                "open replay config differs from its saved identity"
                            )
                        if (
                            build_resolved_env_config_v1(
                                restore_recording_config(packet.config)
                            )
                            != context.resolved_env_config
                        ):
                            raise ValueError(
                                "open replay context differs from its actual config"
                            )
                if temporary.tell() != record["bytes"] or temporary.read(1):
                    raise ValueError(
                        "open replay prefix contains trailing or missing records"
                    )
                assert first is not None
                collector.restore_stream(
                    first,
                    context=context,
                    runtime=runtime,
                    stream=temporary,
                    count=count,
                )
            except BaseException:
                temporary.close()
                raise
        return collector
    except BaseException:
        collector.close()
        raise


@dataclass
class RecordingRestore:
    """Hold fully checked restore data until RunWriter installs it.

    ``collector`` owns temporary replay streams. The constructor must call close
    if installation fails. ``details`` is the selected snapshot, ``hashes`` its
    verified CSV prefix states, and ``replays`` supplies original identity joins.
    """

    token: dict[str, object]
    details: dict[str, Any]
    collector: ReplayCollector
    hashes: dict[str, Any]
    replays: list[dict[str, Any]]

    def close(self) -> None:
        """Release prepared streams after a failed constructor or restore."""
        self.collector.close()


def create_checkpoint(writer: RunWriter) -> dict[str, object]:
    """Flush and publish a recording token for one healthy training pass.

    The caller holds the run lock. Reject an active collector or pending start
    claim before output. Returns a version-1 token only after all bundle files and
    directory entries, including new run/output ancestors, are durable on the
    supported local filesystem. I/O failures fail the writer; no learner data
    is saved, and previous bundles are retained.
    """
    from marl_battlegrounds.evaluation.run_writer import _json_bytes

    writer._check_open()
    entry = writer._details["passes"][writer._pass_key]
    if writer._collection_active:
        raise ValueError("finish collection before creating a recording checkpoint")
    if writer._pending_numerical_starts:
        raise ValueError("verify pending episode starts before checkpointing")
    if len(writer._details["passes"]) != 1 or entry["phase"] != "training":
        raise ValueError("recording checkpoints require one dedicated training pass")
    try:
        writer.flush()
        if writer._checkpoint_hashes is None:
            writer._checkpoint_hashes = {
                name: _hash_prefix(
                    writer._table_path(name),
                    writer._details["tables"].get(name, {}).get("durable_bytes", 0),
                )
                for name in writer._rows
            }
        root = writer.run_dir / "recording_checkpoints"
        if root.is_symlink():
            raise ValueError(
                "recording checkpoint directory must not be a symbolic link"
            )
        if not root.exists():
            root.mkdir()
            os.fsync(writer._lock)
        identifier = uuid4().hex
        temporary = root / f".{identifier}.tmp"
        temporary.mkdir()
        replay_dir = temporary / "open_replays"
        replay_dir.mkdir()
        with (temporary / "run_details.json").open("xb") as stream:
            stream.write(_json_bytes(writer._details))
            stream.flush()
            os.fsync(stream.fileno())
        replays = (
            []
            if writer._collector is None
            else writer._collector.checkpoint_streams(replay_dir)
        )
        for replay in replays:
            replay["config_id"] = writer._replay_config_ids[
                _integer(replay["episode_id"], "replay ID", 1)
            ]
        files: dict[str, object] = {
            "run_details.json": _file_evidence(temporary / "run_details.json"),
            **{
                str(r["path"]): {"bytes": r["bytes"], "sha256": r["sha256"]}
                for r in replays
            },
        }
        tables = {
            name: {
                "durable_bytes": writer._details["tables"]
                .get(name, {})
                .get("durable_bytes", 0),
                "rows": writer._details["tables"].get(name, {}).get("rows", 0),
                "sha256": digest.hexdigest(),
            }
            for name, digest in writer._checkpoint_hashes.items()
        }
        from marl_battlegrounds.evaluation.revision import discover_code_revision_v2

        source = {"code_revision": discover_code_revision_v2().model_dump(mode="json")}
        bundle = {
            "schema_version": 1,
            "run_id": writer.run_id,
            "pass_key": writer._pass_key,
            "host_schema_version": writer._details["schema_version"],
            "metric_schema_id": writer._details["metric_schema_id"],
            "metric_schema_version": writer._details["metric_schema_version"],
            "actor_input_projections": writer._details["actor_input_projections"],
            "source": source,
            "replay_layout_version": 1,
            "files": files,
            "tables": tables,
            "open_replays": replays,
        }
        payload = _json_bytes(bundle)
        with (temporary / "checkpoint.json").open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(replay_dir)
        _sync_directory(temporary)
        temporary.rename(root / identifier)
        _sync_directory(root)
        _sync_directory_ancestors(writer.run_dir)
        writer._checkpoint_pass = writer._pass_key
        return {
            "schema_version": 1,
            "run_id": writer.run_id,
            "pass_key": writer._pass_key,
            "checkpoint_id": identifier,
            "checkpoint_sha256": sha256(payload).hexdigest(),
        }
    except BaseException as error:
        writer._record_failure(error)
        raise


def _prefix_lines(path: Path, size: int) -> Iterator[str]:
    """Yield UTF-8 physical lines within a CSV boundary, never its later suffix.

    CSV parsing owns quoted embedded newlines. Memory holds one physical line;
    the existing fixed-width report limits ordinary row size, not run length.
    """
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            line = stream.readline(remaining)
            if not line:
                raise ValueError("checkpoint table is truncated")
            remaining -= len(line)
            if not remaining and not line.endswith(b"\n"):
                raise ValueError("checkpoint table ends inside a CSV row")
            yield line.decode("utf-8")


def _check_tables(
    writer: RunWriter, saved: dict[str, Any], tables: object
) -> dict[str, Any]:
    """Check all required CSV prefixes before rollback and rebuild hash caches."""
    table_map = _object(tables, "checkpoint tables")
    if set(table_map) != set(writer._rows):
        raise ValueError("checkpoint table roles differ from this writer")
    hashes: dict[str, Any] = {}
    for name, evidence in table_map.items():
        info = _object(evidence, "table boundary")
        if set(info) != {"durable_bytes", "rows", "sha256"}:
            raise ValueError("unsupported checkpoint table fields")
        size = _integer(info["durable_bytes"], "table byte length")
        count = _integer(info["rows"], "table row count")
        boundary = saved["tables"].get(name, {"durable_bytes": 0, "rows": 0})
        if boundary != {"durable_bytes": size, "rows": count}:
            raise ValueError("checkpoint table differs from its manifest")
        path = writer._table_path(name)
        digest = _hash_prefix(path, size)
        if digest.hexdigest() != _hex(info["sha256"], 64, "table prefix digest"):
            raise ValueError(f"recording checkpoint CSV prefix differs: {name}")
        if size:
            reader = csv.reader(_prefix_lines(path, size), strict=True)
            if next(reader, None) != list(writer._headers.get(name, ())):
                raise ValueError(f"checkpoint table header differs: {name}")
            if sum(1 for _ in reader) != count:
                raise ValueError(f"checkpoint table row boundary differs: {name}")
        elif count:
            raise ValueError("an empty checkpoint table cannot contain rows")
        hashes[name] = digest
    return hashes


def _check_manifest_references(saved: dict[str, Any], pass_key: str) -> None:
    """Check content IDs and references without re-running past game validation.

    The immutable bundle proves bytes; these checks also prove that its saved
    sources, configurations, starts, completions, traces and schedules join. Physical
    validity and source-to-spawn relationships remain the existing writer's rule.
    """
    from marl_battlegrounds.evaluation.run_writer import _json_bytes

    configurations = _object(saved.get("configurations"), "saved configurations")
    for identifier, content in configurations.items():
        _hex(identifier, 64, "configuration ID")
        if (
            sha256(_json_bytes(_object(content, "configuration"))).hexdigest()
            != identifier
        ):
            raise ValueError("checkpoint configuration content differs from its ID")
    banks = _object(saved.get("source_banks"), "saved source banks")
    for identifier, references in banks.items():
        _hex(identifier, 64, "source bank ID")
        if not isinstance(references, list) or not references:
            raise ValueError("checkpoint source bank must contain configurations")
        if any(
            not isinstance(ref, str) or ref not in configurations
            for ref in cast(list[object], references)
        ):
            raise ValueError(
                "checkpoint source bank references a missing configuration"
            )
        if (
            sha256(_json_bytes(cast(list[object], references))).hexdigest()
            != identifier
        ):
            raise ValueError("checkpoint source bank content/order differs from its ID")
    entry = saved["passes"][pass_key]
    for value in entry.get("trace_config_ids", {}).values():
        if not isinstance(value, str) or value not in configurations:
            raise ValueError("checkpoint trace references a missing configuration")
    completed = {
        str(_integer(value, "completed episode ID", 1))
        for value in entry["completed_episode_ids"]
    }
    for episode_id, value in _object(
        entry.get("completed_config_ids", {}), "completed configuration IDs"
    ).items():
        if episode_id not in completed:
            raise ValueError("checkpoint config refers to an unfinished episode")
        if not isinstance(value, str) or value not in configurations:
            raise ValueError("checkpoint completion references a missing configuration")
    for start in entry.get("episode_starts", {}).values():
        if start.get("verification") == "pending":
            raise ValueError("checkpoint contains unresolved first-start evidence")
        resolved = start.get("resolved_config_id")
        if resolved is not None and resolved not in configurations:
            raise ValueError("checkpoint start references a missing configuration")
        if start.get("source_known"):
            bank = banks.get(start.get("source_table_id"))
            index = _integer(start.get("source_index"), "start source index")
            if bank is None or index >= len(bank):
                raise ValueError("checkpoint start references a missing source")
    for episode in entry.get("episodes", {}).values():
        resolved = episode.get("configuration_digest", episode.get("config_id"))
        if resolved is not None:
            # A future dictionary schedule can declare a digest before its game
            # supplies actual config evidence. Existing writer checks own that join.
            _hex(resolved, 64, "scheduled configuration digest")


def prepare_restore(
    writer: RunWriter,
    token: object,
    *,
    phase: str,
    pass_id: str,
    policies: dict[str, object] | None,
    checkpoint_id: str | None,
    details: dict[str, object] | None,
) -> RecordingRestore:
    """Validate one explicit token and prepare replay streams before any mutation.

    Validate every referenced file and CSV prefix, pass compatibility, immutable
    schemas and replay contents. Returns owned temporary streams and selected
    metadata. No run files are truncated or rewritten here. The caller must close
    the result if later constructor work fails.
    """
    from marl_battlegrounds.evaluation.replay_io import load_replay

    supplied = _object(token, "recording checkpoint token")
    if (
        set(supplied) != _TOKEN_KEYS
        or type(supplied.get("schema_version")) is not int
        or supplied["schema_version"] != 1
    ):
        raise ValueError("unsupported recording checkpoint token")
    identifier = _hex(supplied["checkpoint_id"], 32, "checkpoint ID")
    token_digest = _hex(supplied["checkpoint_sha256"], 64, "checkpoint digest")
    current = writer._details
    if supplied["run_id"] != current["run_id"]:
        raise ValueError("recording checkpoint belongs to another run")
    marker = current.get("recording_restore")
    if marker is not None and marker != {"schema_version": 1, "token": supplied}:
        raise ValueError("interrupted restore requires its original checkpoint token")
    bundle_path = _regular(
        writer.run_dir, f"recording_checkpoints/{identifier}/checkpoint.json"
    )
    if sha256(bundle_path.read_bytes()).hexdigest() != token_digest:
        raise ValueError("recording checkpoint metadata digest differs")
    bundle = _json(bundle_path)
    if (
        set(bundle) != _BUNDLE_KEYS
        or type(bundle.get("schema_version")) is not int
        or bundle["schema_version"] != 1
        or type(bundle.get("replay_layout_version")) is not int
        or bundle["replay_layout_version"] != 1
    ):
        raise ValueError("unsupported recording checkpoint bundle version or fields")
    if (
        bundle["run_id"] != supplied["run_id"]
        or bundle["pass_key"] != supplied["pass_key"]
    ):
        raise ValueError("recording checkpoint identity differs from its token")
    root = bundle_path.parent
    files = _object(bundle["files"], "checkpoint files")
    raw_replays = bundle["open_replays"]
    if not isinstance(raw_replays, list):
        raise ValueError("open replay declarations must be a list")
    replays = [_object(item, "open replay") for item in cast(list[object], raw_replays)]
    for replay in replays:
        if set(replay) != _OPEN_REPLAY_KEYS or not isinstance(replay.get("path"), str):
            raise ValueError("open replay declaration has unsupported fields")
    wanted = {"run_details.json", *(r.get("path") for r in replays)}
    if set(files) != wanted or len(wanted) != len(replays) + 1:
        raise ValueError("checkpoint files have duplicate or extra roles")
    for replay in replays:
        if files[replay["path"]] != {
            "bytes": replay["bytes"],
            "sha256": replay["sha256"],
        }:
            raise ValueError("open replay evidence differs from its file declaration")
    verified_paths = {
        name: _check_file(root, name, evidence) for name, evidence in files.items()
    }
    saved = _json(root / "run_details.json")
    if (
        saved.get("recording_restore") is not None
        or saved.get("run_id") != current["run_id"]
    ):
        raise ValueError("checkpoint manifest has an invalid run or restore state")
    for name, bundle_name in (
        ("schema_version", "host_schema_version"),
        ("metric_schema_id", "metric_schema_id"),
        ("metric_schema_version", "metric_schema_version"),
        ("actor_input_projections", "actor_input_projections"),
    ):
        if saved.get(name) != current.get(name) or saved.get(name) != bundle.get(
            bundle_name
        ):
            raise ValueError("recording checkpoint schemas differ from this run")
    pass_key = supplied["pass_key"]
    if (
        not isinstance(pass_key, str)
        or set(saved["passes"]) != {pass_key}
        or set(current["passes"]) != {pass_key}
        or saved["passes"][pass_key]["phase"] != "training"
    ):
        raise ValueError("checkpoint restore requires the same dedicated training pass")
    _check_manifest_references(saved, pass_key)
    writer._details = saved
    try:
        selected, _, _ = writer._prepare_pass(
            phase, pass_id, policies, checkpoint_id, details
        )
        if selected != pass_key:
            raise ValueError("recording checkpoint pass differs from requested pass")
    finally:
        writer._details = current
    hashes = _check_tables(writer, saved, bundle["tables"])
    entry = saved["passes"][pass_key]
    for replay in entry["replays"].values():
        relative = replay["path"]
        if (
            not isinstance(relative, str)
            or len(Path(relative).parts) != 2
            or Path(relative).parts[0] != "replays"
        ):
            raise ValueError("completed replay path is unsafe")
        path = _regular(writer.run_dir, relative)
        if path.stat().st_size != replay["bytes"]:
            raise ValueError("completed replay byte length differs")
        loaded = load_replay(path)
        if loaded.replay.canonical_digest_sha256 != replay["canonical_digest_sha256"]:
            raise ValueError("completed replay identity differs")
    collector = _restore_replays(writer, verified_paths, replays, saved)
    try:
        collector.restore_completed(
            int(value) for value in entry["completed_episode_ids"]
        )
    except BaseException:
        collector.close()
        raise
    return RecordingRestore(
        cast(dict[str, object], supplied), saved, collector, hashes, replays
    )


def begin_restore(writer: RunWriter, prepared: RecordingRestore) -> None:
    """Publish a recoverable selected boundary before truncating newer CSV rows."""
    from marl_battlegrounds.evaluation.run_writer import _atomic_json

    writer._details = {
        **prepared.details,
        "recording_restore": {"schema_version": 1, "token": prepared.token},
    }
    _atomic_json(writer.run_dir / "run_details.json", writer._details)
    os.fsync(writer._lock)
    writer._recover_tables()


def finish_restore(writer: RunWriter, prepared: RecordingRestore) -> None:
    """Install validated replay state after pass selection and clear the marker.

    RunWriter must already have selected its saved pass and rebuilt completed IDs.
    Retry every restore failure with the same explicit token. Earlier failures
    leave a restore marker; a failure after marker replacement may already have
    removed it. The token remains the required matching learner boundary.
    The prepared collector becomes writer-owned on success.
    """
    from marl_battlegrounds.evaluation.run_writer import _atomic_json

    writer._collector = prepared.collector
    writer._collector._spool_dir = writer.run_dir / ".replay_spool"
    writer._collector._spool_dir.mkdir(exist_ok=True)
    writer._replay_ids = {
        str(row["context_id"]): int(row["episode_id"]) for row in prepared.replays
    }
    writer._replay_config_ids = {
        int(row["episode_id"]): str(row["config_id"]) for row in prepared.replays
    }
    writer._preflight_contexts.clear()
    writer._checkpoint_hashes = prepared.hashes
    writer._checkpoint_pass = writer._pass_key
    writer._details.pop("recording_restore", None)
    _atomic_json(writer.run_dir / "run_details.json", writer._details)
    os.fsync(writer._lock)
