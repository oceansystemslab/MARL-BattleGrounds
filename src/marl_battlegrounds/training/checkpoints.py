"""Save complete learner boundaries and load independent frozen MAPPO actors.

The runner owns its run lock, immutable settings and training logs. This module
owns synchronous Orbax payloads, file integrity, publication and checked numerical
restore. Restore reads everything before the optional recording recovery helper
can rewind M8 files. Actor exports need neither training content nor a learner.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp  # pyright: ignore[reportMissingTypeStubs]

from marl_battlegrounds.baselines.actions import ACTION_SCHEMA_VERSION
from marl_battlegrounds.baselines.inputs import (
    ACTOR_INPUT_SCHEMA_VERSION,
    TRAINING_STATE_SCHEMA_VERSION,
)
from marl_battlegrounds.baselines.ppo import (
    DEFAULT_PPO_CONFIG,
    PPOConfig,
    initialize_ppo,
    make_recurrent_mappo_system,
)
from marl_battlegrounds.evaluation.recording_identity import tree_digest
from marl_battlegrounds.training._compilation import execution_identity
from marl_battlegrounds.training.distributions import TRAINING_KEY_SCHEMA_VERSION

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.policy_execution import System
    from marl_battlegrounds.evaluation.run_writer import RunWriter
    from marl_battlegrounds.training.collection import TrainingCollection
    from marl_battlegrounds.training.learner import LearnerState

type Tree = Any
_DESCRIPTION = "checkpoint_details.json"
_ACTOR_DESCRIPTION = "actor_details.json"
_COPY_BYTES = 1024 * 1024
_SCHEMAS = {
    "checkpoint": 1,
    "model": "recurrent_mappo_128",
    "actor_input": ACTOR_INPUT_SCHEMA_VERSION,
    "training_state": TRAINING_STATE_SCHEMA_VERSION,
    "action": ACTION_SCHEMA_VERSION,
    "collection_keys": TRAINING_KEY_SCHEMA_VERSION,
    "learner_keys": 1,
}
_CONTEXT = {
    "run_id",
    "attempt_id",
    "parent_checkpoint",
    "config",
    "source",
    "dependencies",
}


@dataclass(frozen=True)
class RestoredCheckpoint:
    """Hold a fully checked learner before any recording or log rewind.

    state is the complete restored numerical learner. details is checked JSON
    metadata; path is the immutable checkpoint directory. This object owns no
    run lock or writer. Keep the runner's lock while installing its recovery.
    """

    state: LearnerState
    details: dict[str, Any]
    path: Path


def _json_bytes(value: object) -> bytes:
    """Encode finite JSON deterministically, including one final newline."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _object(value: object, name: str) -> dict[str, Any]:
    """Require a JSON object with string keys and copy its finite contents."""
    if not isinstance(value, dict) or any(
        not isinstance(k, str) for k in cast(dict[object, object], value)
    ):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], json.loads(_json_bytes(cast(object, value))))


def _read_json(path: Path) -> dict[str, Any]:
    """Read a finite JSON object, rejecting duplicate keys and symbolic links."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular file is missing: {path.name}")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        """Reject ambiguous repeated object keys."""
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        """Reject NaN and Infinity accepted by the default JSON reader."""
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return _object(
        json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=constant),
        path.name,
    )


def _digest(value: object, name: str) -> str:
    """Require one complete lowercase SHA256 identity."""
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA256 identity")
    return value


def _input_scale(value: object) -> float:
    """Validate one finite positive network scale through the PPO owner."""
    return float(PPOConfig(input_scale=cast(float, value)).input_scale)


def _spawn_frame(value: object) -> str:
    """Validate one spawn frame name ("world" or "left") via the PPO owner."""
    return PPOConfig(spawn_frame=cast(str, value)).spawn_frame


def _config_spawn_frame(config: object) -> str:
    """Read a learner's spawn frame; historical configs without it use "world"."""
    ppo = _object(_object(config, "Learner config").get("ppo", {}), "PPO config")
    return _spawn_frame(ppo.get("spawn_frame", "world"))


def saved_training_config(details: dict[str, Any]) -> dict[str, Any]:
    """Return a learner checkpoint's saved config with its historical meaning fixed.

    Parameters
    ----------
    details : dict
        A checkpoint description from read_checkpoint_details with
        metadata.config, the JSON training config saved with the run.

    Returns
    -------
    dict
        A new top-level and ppo dictionary. ppo.spawn_frame is filled with
        "world" when the saved config has no ppo block or no such key, because
        a checkpoint saved before the setting existed trained in raw world
        coordinates whatever the current PPOConfig default is. Every other
        saved value is returned as saved; absent keys keep taking the current
        TrainConfig and PPOConfig defaults through config_from_dict.

    Raises
    ------
    ValueError, TypeError
        metadata.config or its ppo block is not a JSON object.

    Notes
    -----
    Host-only; reads no file and changes neither details nor the saved config.
    The runner decodes a resumed run's settings through this function, so an
    explicit resume config that names a different frame is rejected by the
    runner's exact-settings comparison rather than silently adopted.
    """
    config = _object(details["metadata"]["config"], "Saved training config")
    ppo = dict(_object(config.get("ppo", {}), "Saved ppo settings"))
    ppo.setdefault("spawn_frame", "world")
    config["ppo"] = ppo
    return config


def _actor_spawn_frame(details: dict[str, Any]) -> str:
    """Read the saved spawn frame; historical descriptions without it use "world"."""
    if details["kind"] == "actor":
        return _spawn_frame(details.get("spawn_frame", "world"))
    return _config_spawn_frame(details["metadata"]["config"])


def _config_input_scale(config: object) -> float:
    """Read a learner's scale; historical configs without it use 1.0."""
    ppo = _object(_object(config, "Learner config").get("ppo", {}), "PPO config")
    return _input_scale(ppo.get("input_scale", 1.0))


def _actor_input_scale(details: dict[str, Any]) -> float:
    """Read the saved inference scale without changing historical descriptions."""
    if details["kind"] == "actor":
        return _input_scale(details.get("input_scale", 1.0))
    return _config_input_scale(details["metadata"]["config"])


def _inference_digest(details: dict[str, Any]) -> str:
    """Bind weights, scale and spawn frame; keep old identities where unchanged.

    At scale 1.0 in the world frame the identity is the raw weight digest. A
    non-default scale alone uses the version-1 envelope exactly as before, so
    every existing scaled identity is unchanged. A "left" frame adds the frame
    to a version-2 envelope, so equal weights played in different frames are
    different Systems.
    """
    scale = _actor_input_scale(details)
    frame = _actor_spawn_frame(details)
    if scale == 1.0 and frame == "world":
        return details["actor_digest"]
    envelope: dict[str, Any] = {
        "kind": "recurrent_mappo_inference",
        "version": 1,
        "actor_digest": details["actor_digest"],
        "input_scale": scale,
    }
    if frame != "world":
        envelope["version"] = 2
        envelope["spawn_frame"] = frame
    return sha256(_json_bytes(envelope)).hexdigest()


def _directory(path: Path) -> Path:
    """Reject symbolic links along an existing directory's complete path."""
    absolute = path.absolute()
    for ancestor in (*reversed(absolute.parents), absolute):
        if ancestor.is_symlink():
            raise ValueError("Checkpoint directories must not use symbolic links")
    if not absolute.is_dir():
        raise ValueError("Checkpoint directory is missing")
    return absolute


def _relative(root: Path, value: object) -> Path:
    """Resolve a relative child without traversal or symbolic links."""
    if not isinstance(value, str):
        raise ValueError("Saved file path must be a string")
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(p in (".", "..") for p in relative.parts)
    ):
        raise ValueError("Saved file path must stay below its owner")
    child = root
    for part in relative.parts:
        child /= part
        if child.is_symlink():
            raise ValueError("Saved paths must not use symbolic links")
    return child


def _sync_dir(path: Path) -> None:
    """Persist local directory entries; propagate durability failures."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object) -> None:
    """Durably replace one JSON file on its existing local filesystem."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_dir(path.parent)


def _file_hash(path: Path) -> str:
    """Hash a regular file using bounded host memory."""
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(_COPY_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, object]:
    """Describe every payload file while rejecting links and special files."""
    result: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint payload must not contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Checkpoint payload must contain regular files")
        result[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": _file_hash(path),
        }
    return result


def _sync_tree(root: Path) -> None:
    """Persist every published file and directory before exposing its name."""
    paths = sorted(root.rglob("*"))
    for path in paths:
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in reversed(paths):
        if path.is_dir():
            _sync_dir(path)
    _sync_dir(root)


def _layout(tree: Tree) -> list[dict[str, object]]:
    """Describe logical array paths without serializing Python types or trees."""
    rows: list[dict[str, object]] = []
    flattened = cast(
        list[tuple[tuple[Any, ...], Any]],
        jax.tree_util.tree_flatten_with_path(tree)[0],
    )
    for path, value in flattened:
        keys = []
        for key in path:
            if isinstance(key, jax.tree_util.GetAttrKey):
                keys.append({"field": key.name})
            elif isinstance(key, jax.tree_util.SequenceKey):
                keys.append({"index": key.idx})
            elif isinstance(
                key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)
            ):
                keys.append({"key": key.key})
            else:
                raise TypeError("Unsupported numerical checkpoint field path")
        rows.append(
            {"path": keys, "shape": list(value.shape), "dtype": str(value.dtype)}
        )
    return rows


def _is_actor(row: dict[str, object]) -> bool:
    """Identify the learner's sole actor owner by its exact record path."""
    return cast(list[object], row["path"])[:3] == [
        {"field": "carry"},
        {"field": "history"},
        {"field": "current_variables"},
    ]


def _state_payload(state: LearnerState) -> dict[str, Any]:
    """Flatten state excluding the actor and zero-element leaves.

    Orbax rejects zero-element arrays. Their exact path, shape and dtype remain
    in the checked layout; restore rebuilds their empty storage from the template.
    """
    return {
        f"leaf_{index:05d}": leaf
        for index, (leaf, row) in enumerate(
            zip(jax.tree.leaves(state), _layout(state), strict=True)
        )
        if not _is_actor(row) and leaf.size
    }


def _targets(tree: Tree, device: object | None) -> Tree:
    """Build explicit array restore targets on the caller's selected device."""
    selected = cast(Any, jax.devices()[0] if device is None else device)
    sharding = jax.sharding.SingleDeviceSharding(selected)

    def target(value: Tree) -> jax.ShapeDtypeStruct:
        """Place one abstract array on the chosen device, preserving typed keys."""
        return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding)

    return jax.tree.map(target, tree)


def _save_arrays(path: Path, tree: Tree) -> None:
    """Synchronously save numerical arrays using the installed Orbax handler."""
    with cast(Any, ocp.Checkpointer(ocp.StandardCheckpointHandler())) as checkpointer:
        checkpointer.save(path, args=ocp.args.StandardSave(tree))


def _restore_arrays(path: Path, template: Tree, device: object | None) -> Tree:
    """Restore strictly into code-owned shapes and explicit current placement."""
    with cast(Any, ocp.Checkpointer(ocp.StandardCheckpointHandler())) as checkpointer:
        saved = checkpointer.metadata(path).item_metadata.tree

        def physical(value: Tree) -> Tree:
            """Compare Orbax's physical key storage before any dtype conversion."""
            if jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
                return jax.eval_shape(jax.random.key_data, value)
            return value

        if _layout(saved) != _layout(jax.tree.map(physical, template)):
            raise ValueError("Stored payload paths, shapes or dtypes differ")
        return checkpointer.restore(
            path, args=ocp.args.StandardRestore(_targets(template, device), strict=True)
        )


def _check_context(
    metadata: object, *, require_execution: bool = False
) -> dict[str, Any]:
    """Check scientific metadata and, for new saves, the current execution policy.

    require_execution defaults to False so historical descriptions remain
    readable. True requires the current compilation/runtime identity. Missing
    old execution metadata never receives the current policy implicitly.
    """
    value = _object(metadata, "Checkpoint metadata")
    if not value.keys() >= _CONTEXT:
        raise ValueError(f"Checkpoint metadata needs {sorted(_CONTEXT)}")
    for key in ("run_id", "attempt_id"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if value["parent_checkpoint"] is not None:
        _digest(value["parent_checkpoint"], "Parent checkpoint")
    for key in ("config", "source", "dependencies"):
        if not _object(value[key], key):
            raise ValueError(f"{key} must not be empty")
    if "execution" in value:
        _object(value["execution"], "Execution identity")
    if require_execution and value.get("execution") != execution_identity():
        raise ValueError("Checkpoint needs the current execution identity")
    return value


def _collection_details(collection: TrainingCollection) -> dict[str, Any]:
    """Record reconstructible static settings and verified content identities.

    A named pinned opponent's JSON record is added under "pinned_opponent" only
    when one is pinned, so descriptions of unpinned runs keep their bytes.
    """
    details = _object(
        {
            "content_binding": collection.binding.model_dump(mode="json"),
            "schedule": dict(collection.schedule.rounding_report),
            "score_threshold_curriculum": (
                collection.schedule.score_threshold_curriculum
            ),
            "root_bits": list(collection.root_bits),
            "key_schema": collection.key_schema,
            "reward_settings": list(collection.reward_settings),
            "shaping": collection.shaping,
            "shaping_mode": collection.shaping_mode,
            "pinned_opponent_share": collection.pinned_opponent_share,
            "collect_training_state": collection.collect_training_state,
            "metrics": collection.metrics,
            "recording": collection.recording,
        },
        "Collection details",
    )
    if collection.pinned_opponent is not None:
        details["pinned_opponent"] = collection.pinned_opponent
    return details


def _check_recording(value: object, root: Path) -> dict[str, Any]:
    """Validate original recording constructor arguments before any rewind."""
    record = _object(value, "Recording settings")
    required = {
        "relative_path",
        "phase",
        "pass_id",
        "policies",
        "checkpoint_id",
        "details",
    }
    if set(record) != required or record["phase"] != "training":
        raise ValueError("Recording settings need the original dedicated training pass")
    _relative(root, record["relative_path"])
    if not isinstance(record["pass_id"], str) or not record["pass_id"]:
        raise ValueError("Recording pass ID must be nonempty")
    if record["checkpoint_id"] is not None and not isinstance(
        record["checkpoint_id"], str
    ):
        raise ValueError("Recording checkpoint label must be text or None")
    _object(record["policies"], "Original training registrations")
    _object(record["details"], "Original training pass details")
    return record


def _check_recording_registration(record: dict[str, Any], writer_path: Path) -> None:
    """Match constructor facts to the writer's already-durable registration.

    This reads the public run manifest before asking the writer to make a token.
    M8 still owns all token/table/replay checks and every recording mutation.
    """
    details = _read_json(writer_path / "run_details.json")
    key = json.dumps((record["phase"], record["pass_id"]), separators=(",", ":"))
    passes = _object(details.get("passes"), "Recorded passes")
    if set(passes) != {key}:
        raise ValueError("Checkpoint recording needs its original dedicated pass")
    saved = _object(passes[key], "Original training pass")
    for name in ("phase", "pass_id", "policies", "checkpoint_id", "details"):
        if saved.get(name) != record[name]:
            raise ValueError("Checkpoint recording registration differs")


def save_checkpoint(
    run_dir: str | Path,
    collection: TrainingCollection,
    state: LearnerState,
    *,
    metadata: dict[str, object],
    writer: RunWriter | None = None,
    ppo: PPOConfig = DEFAULT_PPO_CONFIG,
) -> Path:
    """Publish a complete update boundary, then replace its latest pointer.

    Parameters
    ----------
    run_dir : str or Path
        Existing run directory. The caller must hold its exclusive run lock.
    collection, state
        Matching stable collection descriptor and complete accepted learner.
        Initialization is allowed; no actor decision occurs during saving.
        Saving reuses the descriptor's installed-content check from setup or
        restore. It still checks the actual carried source bank and learner.
    metadata : dict
        Finite JSON with run_id, attempt_id, parent_checkpoint, config, source,
        dependencies and execution. Execution must match the current shared
        compilation/runtime owner. The runner supplies verified scientific facts.
        Optional host_state and log_cursors are stored unchanged. With recording,
        recording holds the original writer constructor facts documented below.
    writer : RunWriter or None, default None
        Required exactly when collection.recording is true. Its token is saved
        with the learner. Recording metadata contains relative_path, phase,
        pass_id, policies, checkpoint_id and details from original registration.
    ppo : PPOConfig, default DEFAULT_PPO_CONFIG
        Existing numerical learner settings used by boundary validation. Its
        input scale and spawn frame must equal the saved config's, else the
        save is refused before any file changes. The default's frame is "left".

    Returns
    -------
    Path
        Immutable content-identified checkpoint directory. All payload files and
        its latest_checkpoint.json pointer are durable on the local filesystem.

    Raises
    ------
    ValueError
        State, context, writer or recorded boundary is incompatible.
    OSError
        Saving or publication fails. Earlier checkpoints remain intact; an
        unreferenced complete directory never becomes latest implicitly.
    """
    from marl_battlegrounds.training.learner import validate_learner

    root = _directory(Path(run_dir))
    context = _check_context(metadata, require_execution=True)
    if _config_input_scale(context["config"]) != ppo.input_scale:
        raise ValueError("Checkpoint config input_scale differs from PPO settings")
    if _config_spawn_frame(context["config"]) != ppo.spawn_frame:
        raise ValueError("Checkpoint config spawn_frame differs from PPO settings")
    validate_learner(collection, state, ppo=ppo, recheck_installed_content=False)
    if collection.recording != (writer is not None):
        raise ValueError("Checkpoint writer must match collection recording")
    token = None
    if writer is not None:
        recording = _check_recording(context.get("recording"), root)
        if (
            _relative(root, recording["relative_path"]).resolve()
            != writer.run_dir.resolve()
        ):
            raise ValueError("Recording path differs from the open writer")
        _check_recording_registration(recording, writer.run_dir)
        token = writer.checkpoint_recording()
    elif context.get("recording") is not None:
        raise ValueError("Unrecorded checkpoint must not claim a recording writer")
    directory = root / "checkpoints"
    directory.mkdir(exist_ok=True)
    _directory(directory)
    temporary = directory / f".pending-{uuid4().hex}"
    temporary.mkdir()
    actor = state.carry.history.current_variables
    _save_arrays(temporary / "actor", actor)
    _save_arrays(temporary / "state", _state_payload(state))
    details: dict[str, Any] = {
        "schema_version": 1,
        "kind": "learner",
        "schemas": _SCHEMAS,
        "metadata": context,
        "collection": _collection_details(collection),
        "layout": _layout(state),
        "actor_layout": _layout(actor),
        "actor_digest": tree_digest(actor),
        "recording_token": token,
        "counters": {
            "updates": int(state.completed_updates),
            "env_steps": int(state.carry.progress.rounds)
            * collection.schedule.num_envs,
        },
        "files": _inventory(temporary),
    }
    identifier = sha256(_json_bytes(details)).hexdigest()
    details["checkpoint_id"] = identifier
    _atomic_json(temporary / _DESCRIPTION, details)
    _sync_tree(temporary)
    destination = directory / identifier
    if destination.exists():
        if read_checkpoint_details(destination) != details:
            raise ValueError("Checkpoint identity already contains different data")
        shutil.rmtree(temporary)
    else:
        temporary.rename(destination)
    _sync_dir(directory)
    _atomic_json(
        root / "latest_checkpoint.json",
        {
            "schema_version": 1,
            "checkpoint_id": identifier,
            "relative_path": f"checkpoints/{identifier}",
        },
    )
    for ancestor in (root, *root.parents):
        _sync_dir(ancestor)
    return destination


def read_checkpoint_description(path: str | Path) -> dict[str, Any]:
    """Verify one immutable description, including a pruned ancestor's record.

    path names an actor or learner directory. Check the description's own hash,
    schema, metadata and safe inventory paths without reading payload files.
    This supports ancestry after retention removes old numerical payloads. It
    does not make that ancestor restorable or prove its payload still exists.
    Use read_checkpoint_details before loading or restoring numerical data.
    """
    root = _directory(Path(path))
    name = _DESCRIPTION if (root / _DESCRIPTION).exists() else _ACTOR_DESCRIPTION
    details = _read_json(root / name)
    identifier = _digest(details.get("checkpoint_id"), "Checkpoint ID")
    unsigned = {k: v for k, v in details.items() if k != "checkpoint_id"}
    if sha256(_json_bytes(unsigned)).hexdigest() != identifier:
        raise ValueError("Checkpoint description digest differs")
    if details.get("schema_version") != 1 or details.get("schemas") != _SCHEMAS:
        raise ValueError("Checkpoint model or schema differs from this implementation")
    if details.get("kind") not in ("learner", "actor"):
        raise ValueError("Unsupported checkpoint kind")
    required = {
        "schema_version",
        "kind",
        "schemas",
        "metadata",
        "actor_layout",
        "actor_digest",
        "files",
        "checkpoint_id",
    }
    if details["kind"] == "learner":
        required |= {"collection", "layout", "recording_token", "counters"}
    else:
        # Actor exports carry these optional inference settings only when set.
        required |= {key for key in ("input_scale", "spawn_frame") if key in details}
    if set(details) != required:
        raise ValueError("Checkpoint description fields differ from its schema")
    _object(details["metadata"], "Checkpoint provenance")
    if not isinstance(details["actor_layout"], list):
        raise ValueError("Checkpoint actor layout must be an array schema")
    _digest(details.get("actor_digest"), "Actor digest")
    files = _object(details.get("files"), "Checkpoint file inventory")
    for relative in files:
        _relative(root, relative)
    if details["kind"] == "learner":
        _check_context(details.get("metadata"))
    _actor_input_scale(details)
    _actor_spawn_frame(details)
    return details


def read_checkpoint_details(path: str | Path) -> dict[str, Any]:
    """Read and verify a complete learner or actor artifact without mutation.

    path names an immutable directory. Check every payload file's bytes and the
    description through read_checkpoint_description. A pruned learner is not
    available. No learner is reconstructed and no recording is changed here.
    """
    root = _directory(Path(path))
    details = read_checkpoint_description(root)
    name = _DESCRIPTION if details["kind"] == "learner" else _ACTOR_DESCRIPTION
    actual = {k: v for k, v in _inventory(root).items() if k != name}
    if actual != details["files"]:
        raise ValueError("Checkpoint payload files are missing, changed or unexpected")
    return details


def restore_checkpoint(
    path: str | Path,
    collection: TrainingCollection,
    template: LearnerState,
    *,
    expected_metadata: Mapping[str, object],
    ppo: PPOConfig = DEFAULT_PPO_CONFIG,
    device: object | None = None,
) -> RestoredCheckpoint:
    """Fully validate and restore numerical state without changing any files.

    Parameters
    ----------
    path : str or Path
        Explicit complete learner checkpoint; never a guessed latest directory.
    collection, template
        Fresh code-built descriptor/state from the saved seed and settings. All
        numerical template values are replaced; its static structure is trusted
        only after comparison with the saved content and settings.
        Missing historical shaping_mode means potential; score_delta must be
        saved explicitly and cannot resume a potential checkpoint.
        Missing historical score_threshold_curriculum means False. New threshold
        banks and schedules must match exactly; actor-only exports remain usable
        without reconstructing a training schedule. Missing historical
        pinned_opponent_share means zero in the saved config, in the expected
        config and in the saved collection settings, so an older run compares
        equal to today's default; a positive share must be saved explicitly.
        Missing historical ppo.spawn_frame means "world" on both sides; a
        checkpoint saved in another frame must be resumed in that frame.
        Missing historical pinned_opponent means None in both configs and the
        saved collection settings. A pinned opponent must match its saved record
        exactly (reference, registration, variables digest, evidence, memory
        rule), else restore is refused before arrays are read. After arrays are
        read, and before the slower learner validation, a JAX pinned System's
        weights must match their recorded digest, and a pinned host method with
        memory refuses the resume when any of its games is unfinished, because
        that memory is never saved.
    expected_metadata : mapping
        Required config, source, dependencies and execution discovered for
        this execution. Each must equal its saved value. Additional supplied
        keys are equality assertions; attempt/parent IDs usually are omitted.
    ppo : PPOConfig, default DEFAULT_PPO_CONFIG
        Matched numerical update settings for restored-boundary validation. Its
        input scale and spawn frame must equal the checkpoint's, else restore
        is refused before arrays are read. The default's frame is "left", so a
        checkpoint trained in "world" needs the matching config passed here.
    device : jax.Device or None, default None
        Explicit target device; None uses this process's first selected device.
        Every restored array, including empty IDs, is committed to this device.
        Saved device ordinals are ignored. The target backend and device kind
        must match the checked execution identity. Use frozen actor loading for
        independent inference on another backend.

    Returns
    -------
    RestoredCheckpoint
        Checked numerical state and metadata, before writer/log recovery.

    Raises
    ------
    ValueError
        Any payload, context, content, schema, array or boundary check fails.
        No recording files or trainer logs have been changed.
    """
    from marl_battlegrounds.training.learner import validate_learner

    root = _directory(Path(path))
    details = read_checkpoint_details(root)
    if details["kind"] != "learner":
        raise ValueError("Actor exports cannot resume a learner")
    expected = _object(dict(expected_metadata), "Expected execution metadata")
    if not {"config", "source", "dependencies", "execution"} <= expected.keys():
        raise ValueError(
            "Restore requires current config, source, dependencies and execution"
        )
    if expected["execution"] != execution_identity():
        raise ValueError("Expected execution identity differs from the current runtime")
    saved_metadata = dict(details["metadata"])
    # Configs saved before pinned_opponent_share existed compare at its default.
    for record in (saved_metadata, expected):
        config_record = record.get("config")
        if isinstance(config_record, dict):
            normalized = dict(cast(dict[str, Any], config_record))
            normalized.setdefault("pinned_opponent_share", 0.0)
            normalized.setdefault("pinned_opponent", None)
            ppo_record = normalized.get("ppo")
            if isinstance(ppo_record, dict):
                # Configs saved before spawn_frame existed compare at "world".
                nested = dict(cast(dict[str, Any], ppo_record))
                nested.setdefault("spawn_frame", "world")
                normalized["ppo"] = nested
            record["config"] = normalized
    if any(saved_metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint execution metadata differs")
    if _actor_input_scale(details) != ppo.input_scale:
        raise ValueError("Checkpoint input_scale differs from PPO settings")
    if _actor_spawn_frame(details) != ppo.spawn_frame:
        raise ValueError("Checkpoint spawn_frame differs from PPO settings")
    saved_collection = _object(details["collection"], "Saved collection settings")
    saved_collection.setdefault("shaping_mode", "potential")
    saved_collection.setdefault("score_threshold_curriculum", False)
    saved_collection.setdefault("pinned_opponent_share", 0.0)
    if saved_collection.pop("pinned_opponent", None) != collection.pinned_opponent:
        raise ValueError("Checkpoint pinned opponent differs")
    current_collection = _collection_details(collection)
    current_collection.pop("pinned_opponent", None)
    if saved_collection != current_collection:
        raise ValueError("Checkpoint content or collection settings differ")
    recorded = details["metadata"].get("recording")
    if collection.recording != (recorded is not None):
        raise ValueError("Checkpoint recording declaration differs")
    if collection.recording:
        record = _check_recording(recorded, root.parent.parent)
        _check_recording_registration(
            record, _relative(root.parent.parent, record["relative_path"])
        )
    if details["layout"] != _layout(template) or details["actor_layout"] != _layout(
        template.carry.history.current_variables
    ):
        raise ValueError("Checkpoint array paths, shapes or dtypes differ")
    selected_device = cast(Any, jax.devices()[0] if device is None else device)
    runtime = expected["execution"]["runtime"]
    if (
        selected_device.platform != runtime["backend"]
        or selected_device.device_kind != runtime["device"]
    ):
        raise ValueError("Restore device differs from the checked execution identity")
    actor = _restore_arrays(
        root / "actor", template.carry.history.current_variables, selected_device
    )
    if tree_digest(actor) != details["actor_digest"]:
        raise ValueError("Restored actor digest differs")
    payload = _restore_arrays(root / "state", _state_payload(template), selected_device)
    actor_leaves = iter(jax.tree.leaves(actor))
    leaves: list[Any] = []
    for index, (row, initial) in enumerate(
        zip(details["layout"], jax.tree.leaves(template), strict=True)
    ):
        if _is_actor(row):
            leaves.append(next(actor_leaves))
        elif initial.size == 0:
            leaves.append(
                jax.device_put(np.empty(initial.shape, initial.dtype), selected_device)
            )
        else:
            leaves.append(payload[f"leaf_{index:05d}"])
    state = cast(
        "LearnerState",
        jax.tree.unflatten(cast(Any, jax.tree.structure(template)), leaves),
    )
    if _layout(state) != details["layout"]:
        raise ValueError("Restored array schema differs")
    # The pinned checks read a few restored arrays; they run before the full
    # learner validation, which re-checks the installed content and is slow.
    _check_restored_pinned_opponent(collection, state)
    validate_learner(collection, state, ppo=ppo)
    counters = {
        "updates": int(state.completed_updates),
        "env_steps": int(state.carry.progress.rounds) * collection.schedule.num_envs,
    }
    if counters != details["counters"]:
        raise ValueError("Checkpoint counters disagree with numerical state")
    return RestoredCheckpoint(state, details, root)


def _check_restored_pinned_opponent(
    collection: TrainingCollection, state: LearnerState
) -> None:
    """Apply the pinned opponent's resume rules to already restored arrays.

    A JAX pinned System's restored weights must match the digest recorded when
    the run started. A pinned host method whose memory is not saved can only
    continue when none of its games is unfinished: every lane assigned to it
    must have finished its game at this checkpoint. Raises ValueError otherwise.
    Called inside restore_checkpoint before it returns, so a refusal happens
    before any recording or log file changes. Reads small arrays; writes nothing.
    """
    record = collection.pinned_opponent
    if record is None:
        return
    digest = record.get("variables_digest")
    pinned = cast(tuple[Any, ...], state.carry.pinned_opponent)
    if digest is not None and tree_digest(pinned[0]) != digest:
        raise ValueError("Restored pinned opponent weights differ from its record")
    host = collection.host_opponent
    if host is not None and host.stateful:
        carry = state.carry
        unfinished = np.asarray(
            (carry.history.lane_snapshot == 0) & ~carry.state.done.done
        )
        if bool(unfinished.any()):
            raise ValueError(
                "The pinned host method's memory is not saved, and "
                f"{int(unfinished.sum())} of its games are unfinished at this "
                "checkpoint; ordinary resume "
                "cannot continue them without that memory. Resume from a checkpoint "
                "where none of its games is unfinished, or start a new run."
            )


def resume_recording(
    restored: RestoredCheckpoint, run_dir: str | Path
) -> RunWriter | None:
    """Mark checked recovery, then open M8 with its original token/registrations.

    The caller holds the run lock and must pass restore_checkpoint's result.
    Returns an open writer when recording was enabled, otherwise None. The
    runner then restores its logs and calls finish_checkpoint_recovery. Failure
    leaves the same checkpoint/token marker for an explicit retry; this function
    never clears it on an error or selects a different recovery checkpoint.
    """
    from marl_battlegrounds.evaluation.run_writer import RunWriter

    root = _directory(Path(run_dir))
    if (
        restored.path.parent.name != "checkpoints"
        or restored.path.parent.parent != root
    ):
        raise ValueError("Restored checkpoint belongs to a different run directory")
    token = restored.details["recording_token"]
    record = restored.details["metadata"].get("recording")
    if (record is None) != (token is None):
        raise ValueError("Recording settings and checkpoint token disagree")
    checked = None if record is None else _check_recording(record, root)
    marker = {
        "schema_version": 1,
        "checkpoint_id": restored.details["checkpoint_id"],
        "recording_token": token,
    }
    marker_path = root / "checkpoint_recovery.json"
    if marker_path.exists() and _read_json(marker_path) != marker:
        raise ValueError("Interrupted recovery requires its original checkpoint")
    _atomic_json(marker_path, marker)
    if checked is None:
        return None
    return RunWriter(
        resume_from=_relative(root, checked["relative_path"]),
        recording_checkpoint=token,
        phase=checked["phase"],
        pass_id=checked["pass_id"],
        policies=checked["policies"],
        checkpoint_id=checked["checkpoint_id"],
        details=checked["details"],
    )


def finish_checkpoint_recovery(
    restored: RestoredCheckpoint, run_dir: str | Path
) -> None:
    """Clear the matching marker after writer and trainer-log recovery succeed.

    Caller holds the run lock. A missing/different marker is an error, preventing
    accidental completion of another recovery transaction. This writes no data
    besides removing the marker and syncing its directory.
    """
    root = _directory(Path(run_dir))
    marker = root / "checkpoint_recovery.json"
    expected = {
        "schema_version": 1,
        "checkpoint_id": restored.details["checkpoint_id"],
        "recording_token": restored.details["recording_token"],
    }
    if _read_json(marker) != expected:
        raise ValueError("Recovery marker differs from the restored checkpoint")
    marker.unlink()
    _sync_dir(root)


def _actor_template() -> Tree:
    """Derive the installed fixed actor schema without allocating model weights."""
    return jax.eval_shape(initialize_ppo, jax.random.key(0)).actor_params


def export_system(
    actor_variables: Tree,
    destination: str | Path,
    *,
    metadata: dict[str, object],
    input_scale: float = 1.0,
    spawn_frame: str,
) -> Path:
    """Publish a standalone immutable sampled-MAPPO actor artifact.

    Parameters
    ----------
    actor_variables : PyTree
        Finite variables matching the installed 128-wide actor schema.
    destination : str or Path
        Exact output directory with an existing parent. Matching exports are
        reused; different weights, scale or provenance at this path are rejected.
    metadata : dict
        Finite JSON provenance with run_id, nonnegative integer seed and
        env_steps, and the originating checkpoint_id. Extra facts are retained.
    input_scale : float, default=1.0
        Finite positive factor applied before the actor's first Dense layer.
        Use the originating PPO config's value. It is part of inference identity;
        changing it does not rewrite weights or the raw input schema.
    spawn_frame : str
        Required. Spawn frame the weights were trained in: "world" or "left".
        Use the originating PPO config's value. There is no default, because
        an export writes a durable identity: a guessed frame would silently
        label world weights as left, or the reverse. "left" is written into
        the description and the inference identity; "world" writes no key, so
        world exports keep their historical description bytes.

    Returns
    -------
    Path
        Absolute immutable export directory, after durable publication.

    Raises
    ------
    ValueError
        Variables, scale, frame or provenance are invalid, or the destination
        conflicts.
    OSError
        A required directory or payload cannot be read or written.

    Notes
    -----
    Host-only. No map preparation, critic, optimizer or game state is required.
    """
    input_scale = _input_scale(input_scale)
    spawn_frame = _spawn_frame(spawn_frame)
    context = _object(metadata, "Actor provenance")
    if not {"run_id", "seed", "env_steps", "checkpoint_id"} <= context.keys():
        raise ValueError(
            "Actor provenance needs run_id, seed, env_steps and checkpoint_id"
        )
    if not isinstance(context["run_id"], str) or not context["run_id"]:
        raise ValueError("Actor run ID must be nonempty")
    for name in ("seed", "env_steps"):
        if type(context[name]) is not int or context[name] < 0:
            raise ValueError(f"Actor {name} must be a nonnegative integer")
    _digest(context["checkpoint_id"], "Origin checkpoint")
    if _layout(actor_variables) != _layout(_actor_template()):
        raise ValueError("Actor variables differ from the installed MAPPO schema")
    if not all(
        bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(actor_variables)
    ):
        raise ValueError("Actor variables must be finite")
    target = Path(destination).absolute()
    parent = _directory(target.parent)
    digest = tree_digest(actor_variables)
    if target.exists():
        saved = read_checkpoint_details(target)
        if (
            saved["kind"] == "actor"
            and saved["actor_digest"] == digest
            and _actor_input_scale(saved) == input_scale
            and _actor_spawn_frame(saved) == spawn_frame
            and saved["metadata"] == context
        ):
            return target
        raise ValueError("Actor destination already contains a different artifact")
    temporary = parent / f".{target.name}.{uuid4().hex}.tmp"
    temporary.mkdir()
    _save_arrays(temporary / "actor", actor_variables)
    details: dict[str, Any] = {
        "schema_version": 1,
        "kind": "actor",
        "schemas": _SCHEMAS,
        "metadata": context,
        "actor_layout": _layout(actor_variables),
        "actor_digest": digest,
        "input_scale": input_scale,
        "files": _inventory(temporary),
    }
    if spawn_frame != "world":
        details["spawn_frame"] = spawn_frame
    details["checkpoint_id"] = sha256(_json_bytes(details)).hexdigest()
    _atomic_json(temporary / _ACTOR_DESCRIPTION, details)
    _sync_tree(temporary)
    temporary.rename(target)
    _sync_dir(parent)
    return target


def artifact_identity(path: str | Path) -> dict[str, Any]:
    """Read verified actor identity/provenance without restoring numerical state.

    Accept a standalone export or complete learner checkpoint. Return direct
    actor_digest, weight_digest, input_scale, spawn_frame, checkpoint_id,
    schemas, run_id, seed and env_steps fields plus original metadata.
    actor_digest binds weights, inference scale and spawn frame; at scale 1.0
    in the world frame it keeps the historical weight digest. weight_digest
    always identifies only the saved variables. Neither identity proves
    competence. All file hashes are checked; this operation may read a full
    learner payload from disk. Missing historical scale means 1.0 and a missing
    frame means "world".
    """
    details = read_checkpoint_details(path)
    metadata = details["metadata"]
    actor = details["kind"] == "actor"
    return {
        "actor_digest": _inference_digest(details),
        "weight_digest": details["actor_digest"],
        "input_scale": _actor_input_scale(details),
        "spawn_frame": _actor_spawn_frame(details),
        "checkpoint_id": details["checkpoint_id"],
        "schemas": details["schemas"],
        "metadata": metadata,
        "run_id": metadata["run_id"],
        "seed": metadata.get("seed") if actor else metadata["config"].get("seed"),
        "env_steps": metadata["env_steps"]
        if actor
        else details["counters"]["env_steps"],
    }


def load_system(checkpoint: str | Path) -> System:
    """Load an exact saved actor as the existing sampled recurrent MAPPO System.

    Parameters
    ----------
    checkpoint : str or Path
        Standalone actor export or complete learner checkpoint directory.

    Returns
    -------
    System
        Frozen numerical actor variables with their saved input scale and verified
        inference digest as the checkpoint label. Each evaluator creates fresh
        recurrent memory. The
        actor samples legal masked actions; no critic or training-only input is
        loaded, and no map preparation or full run directory is needed.

    Raises
    ------
    ValueError
        Artifact files, schemas, shapes, dtypes, digest or finite arrays differ.
    OSError
        Required artifact bytes cannot be read.

    Notes
    -----
    Host-only; reads files and restores actor arrays onto the process's selected
    device. It needs the training extra but does not run an actor or learner.
    A frozen actor identity makes no claim about learned competence.
    Historical artifacts without an input scale keep their original scale 1.0
    and weight-digest identity. Learner checkpoints use config.ppo.input_scale;
    actor exports use their explicit saved scale. The saved spawn frame is
    restored the same way and is never inferred from a model's results; a
    missing frame means "world". Neither route rewrites weights.
    """
    root = _directory(Path(checkpoint))
    details = read_checkpoint_details(root)
    template = _actor_template()
    if details["actor_layout"] != _layout(template):
        raise ValueError("Actor artifact schema differs from the installed model")
    actor = _restore_arrays(root / "actor", template, None)
    if tree_digest(actor) != details["actor_digest"] or not all(
        bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(actor)
    ):
        raise ValueError("Restored actor is nonfinite or its digest differs")
    return make_recurrent_mappo_system(
        actor,
        checkpoint=_inference_digest(details),
        input_scale=_actor_input_scale(details),
        spawn_frame=_actor_spawn_frame(details),
    )


def checkpoint_schemas() -> dict[str, int | str]:
    """Return the installed checkpoint and model format identifiers.

    Returns
    -------
    dict[str, int or str]
        A fresh JSON-ready copy of the checkpoint version, model name, actor
        input, training-state and action versions, and collection/learner key
        versions. These are the same identifiers checked during restore.
        Changing the returned dictionary does not change the shared authority.

    Notes
    -----
    Reads only fixed metadata; no file access or device initialization occurs.
    """
    return dict(_SCHEMAS)


def checkpoint_dependencies() -> dict[str, str]:
    """Read installed numerical dependency versions for the runner's identity.

    Returns Python, JAX, JAXlib, NumPy, Flax, Optax and Orbax versions. Missing
    optional training packages raise PackageNotFoundError. This reads metadata
    only and does not install packages or initialize a device.
    """
    import platform

    return {
        "python": platform.python_version(),
        **{
            name: version(name)
            for name in ("jax", "jaxlib", "numpy", "flax", "optax", "orbax-checkpoint")
        },
    }


def runtime_identity() -> dict[str, object]:
    """Read the current imported-source identity and installed dependencies.

    The runner calls this at setup under a stable source snapshot, then reuses
    the result as its source/dependencies equality assertion for resume. Git or
    installed-package discovery reads files and may run read-only Git commands;
    no package is installed, device initialized or artifact written.
    """
    from marl_battlegrounds.evaluation.revision import discover_code_revision_v2

    return {
        "source": discover_code_revision_v2().model_dump(mode="json"),
        "dependencies": checkpoint_dependencies(),
    }
