"""Save complete PPO, QMIX and PQN-VDN learner boundaries and load frozen actors.

The runner owns its run lock, immutable settings and training logs. This module
owns synchronous Orbax payloads, file integrity, publication and checked numerical
restore. Restore reads everything before the optional recording recovery helper
can rewind M8 files. Actor exports need neither training content nor a learner.
All methods share the file layout; the saved model schema names the method.
A QMIX learner checkpoint's actor payload holds only the Q-network parameters;
its epsilon, targets, mixer, optimizer and replay are in the state payload.
A PQN-VDN actor payload is Flax's ``{"params", "batch_stats"}`` dictionary,
built and read by name (Orbax stores NamedTuples as dictionaries); its
epsilon, optimizer state and kept rows are in the state payload. QMIX and
PQN-VDN exports and loaded Systems are greedy (epsilon 0, first legal
maximum).

Artifacts saved before Red Zone use actor input schema 1 (5,164 features).
load_system still loads them, through the schema-1 System hooks, so they play
exactly as before and keep their identities. They cannot be exported again,
and their learner checkpoints cannot resume training in this version.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache, partial
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
    ACTOR_FEATURE_SIZE,
    ACTOR_FEATURE_SIZES,
    ACTOR_INPUT_SCHEMA_VERSION,
    TRAINING_STATE_SCHEMA_VERSION,
)
from marl_battlegrounds.baselines.methods import (
    is_ppo_method,
    method_settings_field,
    validate_training_method,
)
from marl_battlegrounds.baselines.ppo import (
    DEFAULT_PPO_CONFIG,
    PPOConfig,
    initialize_ppo,
    make_ppo_system,
    validate_ppo_method,
)
from marl_battlegrounds.baselines.pqn import (
    DEFAULT_PQN_CONFIG,
    PQN_KEY_SCHEMA_VERSION,
    PQN_NORMALIZATION_SCHEMA_VERSION,
    PQN_RECENT_WINDOW_SCHEMA_VERSION,
    PQN_TIE_RULE,
    PQNConfig,
    PQNInferenceVariables,
    _nonnegative_variances,  # pyright: ignore[reportPrivateUsage]
    make_pqn_system,
    pqn_actor_template,
)
from marl_battlegrounds.baselines.qmix import (
    DEFAULT_QMIX_CONFIG,
    QMIX_KEY_SCHEMA_VERSION,
    QMIX_REPLAY_SCHEMA_VERSION,
    QMIX_TIE_RULE,
    QMIXConfig,
    make_qmix_system,
    qmix_actor_template,
)
from marl_battlegrounds.evaluation.recording_identity import tree_digest
from marl_battlegrounds.training._compilation import execution_identity
from marl_battlegrounds.training.distributions import TRAINING_KEY_SCHEMA_VERSION

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.policy_execution import System
    from marl_battlegrounds.evaluation.run_writer import RunWriter
    from marl_battlegrounds.training.collection import TrainingCollection
    from marl_battlegrounds.training.learner import LearnerState
    from marl_battlegrounds.training.pqn_learner import PQNLearnerState
    from marl_battlegrounds.training.qmix_learner import QMIXLearnerState

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
_MODELS = {
    "mappo": "recurrent_mappo_128",
    "ippo": "recurrent_ippo_128",
    "ff_mappo": "feedforward_mappo_128x128",
    "ff_ippo": "feedforward_ippo_128x128",
    "qmix": "recurrent_qmix_256",
    "pqn_vdn": "recurrent_pqn_vdn_512",
}
_QMIX_SCHEMAS = {
    "checkpoint": 1,
    "model": _MODELS["qmix"],
    "actor_input": ACTOR_INPUT_SCHEMA_VERSION,
    "training_state": TRAINING_STATE_SCHEMA_VERSION,
    "action": ACTION_SCHEMA_VERSION,
    "collection_keys": TRAINING_KEY_SCHEMA_VERSION,
    "qmix_keys": QMIX_KEY_SCHEMA_VERSION,
    "replay": QMIX_REPLAY_SCHEMA_VERSION,
}
_QMIX_COUNTERS = {"updates", "env_steps", "completed_blocks", "learning_blocks"}
_PQN_SCHEMAS = {
    "checkpoint": 1,
    "model": _MODELS["pqn_vdn"],
    "actor_input": ACTOR_INPUT_SCHEMA_VERSION,
    "action": ACTION_SCHEMA_VERSION,
    "collection_keys": TRAINING_KEY_SCHEMA_VERSION,
    "pqn_keys": PQN_KEY_SCHEMA_VERSION,
    "recent_window": PQN_RECENT_WINDOW_SCHEMA_VERSION,
    "normalization": PQN_NORMALIZATION_SCHEMA_VERSION,
}
_PQN_COUNTERS = _QMIX_COUNTERS
# The exact schema dictionaries artifacts carried before Red Zone (actor input
# schema 1, and training state schema 1 where present), frozen. load_system
# still reads them; export and resume refuse them.
_ACTOR_INPUT_1_SCHEMAS: dict[str, dict[str, int | str]] = {
    **{
        method: {
            "checkpoint": 1,
            "model": _MODELS[method],
            "actor_input": 1,
            "training_state": 1,
            "action": 1,
            "collection_keys": 1,
            "learner_keys": 1,
        }
        for method in ("mappo", "ippo", "ff_mappo", "ff_ippo")
    },
    "qmix": {
        "checkpoint": 1,
        "model": _MODELS["qmix"],
        "actor_input": 1,
        "training_state": 1,
        "action": 1,
        "collection_keys": 1,
        "qmix_keys": 1,
        "replay": 1,
    },
    "pqn_vdn": {
        "checkpoint": 1,
        "model": _MODELS["pqn_vdn"],
        "actor_input": 1,
        "action": 1,
        "collection_keys": 1,
        "pqn_keys": 1,
        "recent_window": 1,
        "normalization": 1,
    },
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

    state is the complete restored numerical learner: a PPO LearnerState for
    the four PPO methods, a QMIXLearnerState for ``method="qmix"`` or a
    PQNLearnerState for ``method="pqn_vdn"``, as the saved model schema says.
    It is typed ``Any`` so existing PPO code keeps reading its fields without
    casts; check the method (or use isinstance) when a caller handles more
    than one. details is checked JSON metadata; path is the
    immutable checkpoint directory. This object owns no run lock or writer.
    Keep the runner's lock while installing its recovery.
    """

    state: Any
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


def _settings_block(config: object) -> dict[str, Any]:
    """Return a saved config's method settings: its "ppo", "qmix" or "pqn" object.

    config must be a JSON learner config; ``methods.method_settings_field``
    names the block. A missing block is an empty object, so historical PPO
    configs keep their defaults. Raises ValueError.
    """
    value = _object(config, "Learner config")
    name = method_settings_field(_config_method(value))
    return _object(value.get(name, {}), f"{name.upper()} config")


def _check_qmix_settings(config: object, settings: QMIXConfig) -> None:
    """Require a saved QMIX config's settings to equal the given settings.

    config is the saved JSON learner config; settings is the QMIXConfig the
    caller will save or continue with. Compares every field (for example
    gamma, q_lr, tau, the target rule, T, C, S, M and the exploration clock)
    after the same JSON round trip that config_to_dict applies. Raises
    ValueError on any difference. Reads no arrays and writes nothing.
    """
    expected = json.loads(json.dumps(asdict(settings)))
    if _settings_block(config) != expected:
        raise ValueError("Checkpoint config QMIX settings differ from the given ones")


def _check_pqn_settings(config: object, settings: PQNConfig) -> None:
    """Require a saved PQN-VDN config's settings to equal the given settings.

    config is the saved JSON learner config; settings is the PQNConfig the
    caller will save or continue with. Compares every field (T, H, epochs,
    minibatches, rates, schedules, gamma, lambda, input_scale and
    spawn_frame) after the same JSON round trip that config_to_dict applies,
    so a missing or partial "pqn" block fails. Raises ValueError. Reads no
    arrays and writes nothing.
    """
    expected = json.loads(json.dumps(asdict(settings)))
    if _settings_block(config) != expected:
        raise ValueError("Checkpoint config PQN settings differ from the given ones")


def _config_spawn_frame(config: object) -> str:
    """Read a learner's spawn frame; historical configs without it use "world".

    PPO configs keep it in "ppo", QMIX configs in "qmix", PQN-VDN configs in
    "pqn" (always present there).
    """
    return _spawn_frame(_settings_block(config).get("spawn_frame", "world"))


def _config_value_normalization(config: object) -> bool:
    """Read the saved critic setting; absence means historical raw-value training.

    config must be a JSON learner-config object. A present setting must be a
    Python bool; PPOConfig owns its validation. No saved input is changed.
    """
    ppo = _object(_object(config, "Learner config").get("ppo", {}), "PPO config")
    return PPOConfig(
        value_normalization=cast(bool, ppo.get("value_normalization", False))
    ).value_normalization


def _config_method(config: object) -> str:
    """Read a saved learner method; only historical absence means MAPPO.

    config must be a JSON object. Reject unknown method values through the
    shared method check (PPO names, "qmix" and "pqn_vdn"). No saved object is
    changed and no device work runs. Array shapes never decide which method produced a
    checkpoint.
    """
    value = _object(config, "Learner config")
    return validate_training_method(value.get("method", "mappo"))


def _schema_method(schemas: object) -> str:
    """Return the method named by one complete supported schema dictionary.

    schemas must exactly match either checkpoint_schemas for a known model
    (the current dictionary) or that model's frozen pre-Red-Zone dictionary in
    _ACTOR_INPUT_1_SCHEMAS (actor input 1, and training state 1 where the
    method has it). A mixed dictionary, such as actor input 2 with training
    state 1, matches neither and is rejected. Reject unknown models, altered
    versions/types, missing fields and extra fields with ValueError. Each
    method's own dictionary sets the types, so Boolean or float versions (such
    as a forged ``"replay": true``) do not stand in for integers. This
    host-only check reads no arrays and changes no input. Use
    _require_current_schemas where only the current dictionary is allowed.
    """
    value = _object(schemas, "Checkpoint schemas")
    for method, model in _MODELS.items():
        for expected in (checkpoint_schemas(method), _ACTOR_INPUT_1_SCHEMAS[method]):
            if (
                value.get("model") == model
                and value == expected
                and all(
                    type(value[key]) is type(item) for key, item in expected.items()
                )
            ):
                return method
    raise ValueError("Checkpoint model or schema differs from this implementation")


def _require_current_schemas(details: dict[str, Any]) -> None:
    """Refuse to resume training from a checkpoint saved before Red Zone.

    Parameters
    ----------
    details : dict
        A checkpoint description from read_checkpoint_details, whose
        "schemas" entry is a supported current or historical dictionary.

    Raises
    ------
    ValueError
        The saved schemas are the frozen pre-Red-Zone dictionary (actor input
        schema 1). The message says that this version cannot resume the run
        and that the original source environment that created it (for example
        its launch package) can; load_system still reads its actor. Also
        raised, by _schema_method, for an unsupported dictionary.

    Notes
    -----
    This is the one owner of the historical-resume refusal. restore_checkpoint
    calls it right after the checkpoint kind check, before any array is read.
    read_checkpoint_description never calls it, so launch, ancestry and
    analysis can still read old descriptions. Host-only; reads no files.
    """
    schemas = _object(details["schemas"], "Checkpoint schemas")
    if schemas != checkpoint_schemas(_schema_method(schemas)):
        raise ValueError(
            "This checkpoint was saved before Red Zone scoring (historical "
            "actor input schema 1, 5,164 features). This version cannot resume "
            "its training. Resuming it needs the original source environment "
            "that created it, for example its launch package; load_system "
            "still reads its actor."
        )


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
        A new top-level dictionary. A QMIX or PQN-VDN config keeps its
        "qmix" or "pqn" block exactly as saved and gains no "ppo" block. For
        PPO it also returns a new ppo dictionary. Missing method becomes "mappo";
        missing ppo.input_scale becomes 1.0. ppo.spawn_frame is filled with
        "world" when the saved config has no ppo block or no such key, because
        a checkpoint saved before the setting existed trained in raw world
        coordinates whatever the current PPOConfig default is. Missing
        ppo.value_normalization is filled with False because historical critics
        predicted raw reward units. Missing validation_opponents and
        slot_diagnostic_actor become None, keeping the historical panel route.
        Missing red_zone_depth becomes 0.0, because a run saved before the
        Red Zone rule scored one point per death whatever the current
        TrainConfig default (5.0) is. Every other
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
    if is_ppo_method(_config_method(config)):
        ppo = dict(_object(config.get("ppo", {}), "Saved ppo settings"))
        ppo.setdefault("input_scale", 1.0)
        ppo.setdefault("spawn_frame", "world")
        ppo.setdefault("value_normalization", False)
        config["ppo"] = ppo
        config.setdefault("method", "mappo")
    config.setdefault("validation_opponents", None)
    config.setdefault("slot_diagnostic_actor", None)
    config.setdefault("red_zone_depth", 0.0)
    return config


def _config_red_zone_depth(config: object) -> float | None:  # pyright: ignore[reportUnusedFunction]
    """Read the Red Zone depth a saved training config declares.

    Parameters
    ----------
    config : object
        A saved JSON training config (metadata.config of a checkpoint).

    Returns
    -------
    float or None
        The saved red_zone_depth in map units, or None when the config has no
        such key because it was saved before the Red Zone rule (one point per
        death). None tells validation helpers to expect the older record
        layouts, which carry no depth.

    Raises
    ------
    ValueError
        config is not a JSON object, or its red_zone_depth is not a float.
        The value's range was checked by TrainConfig when the run started.

    Notes
    -----
    Host-only; reads no file and changes nothing. Use this where the saved
    layout matters (validation task and Random record checks). A config that
    must be rebuilt as a TrainConfig uses saved_training_config, which reads a
    missing depth as 0.0 instead.
    """
    value = _object(config, "Learner config").get("red_zone_depth")
    if value is None:
        return None
    if type(value) is not float:
        raise ValueError("Saved red_zone_depth must be a float")
    return value


def _actor_spawn_frame(details: dict[str, Any]) -> str:
    """Read the saved spawn frame; historical descriptions without it use "world"."""
    if details["kind"] == "actor":
        return _spawn_frame(details.get("spawn_frame", "world"))
    return _config_spawn_frame(details["metadata"]["config"])


def _config_input_scale(config: object) -> float:
    """Read a learner's scale; historical configs without it use 1.0.

    PPO configs keep it in "ppo", QMIX configs in "qmix", PQN-VDN configs in
    "pqn".
    """
    return _input_scale(_settings_block(config).get("input_scale", 1.0))


def _actor_input_scale(details: dict[str, Any]) -> float:
    """Read the saved inference scale without changing historical descriptions."""
    if details["kind"] == "actor":
        return _input_scale(details.get("input_scale", 1.0))
    return _config_input_scale(details["metadata"]["config"])


def _inference_digest(details: dict[str, Any]) -> str:
    """Bind weights, scale and spawn frame; keep old identities where unchanged.

    For MAPPO at scale 1.0 in the world frame the identity is the raw weight digest. A
    non-default scale alone uses the version-1 envelope exactly as before, so
    every existing scaled identity is unchanged. A "left" frame adds the frame
    to a version-2 envelope, so equal weights played in different frames are
    different Systems. Other PPO methods always use the version-1 ppo_inference
    envelope with model, actor_digest, input_scale and spawn_frame. QMIX uses a
    version-1 qmix_inference envelope that also binds the greedy rule (epsilon
    0.0 and the first-legal-maximum tie rule): learner checkpoints supply these
    constants and exports their saved, checked values, so both routes give one
    identity for the same Q-network. PQN-VDN uses a version-1
    pqn_vdn_inference envelope with the same greedy fields; its actor_digest
    covers the parameters and the BatchNorm statistics, so a statistics-only
    change is a different System. A synthetic historical MAPPO identity may
    omit schemas; saved artifacts cannot omit it.
    """
    scale = _actor_input_scale(details)
    frame = _actor_spawn_frame(details)
    method = _schema_method(details.get("schemas", _SCHEMAS))
    if method == "pqn_vdn":
        exported = details["kind"] == "actor"
        return sha256(
            _json_bytes(
                {
                    "kind": "pqn_vdn_inference",
                    "version": 1,
                    "model": _MODELS[method],
                    "actor_digest": details["actor_digest"],
                    "input_scale": scale,
                    "spawn_frame": frame,
                    "epsilon": details["epsilon"] if exported else 0.0,
                    "tie_rule": details["tie_rule"] if exported else PQN_TIE_RULE,
                }
            )
        ).hexdigest()
    if method == "qmix":
        exported = details["kind"] == "actor"
        return sha256(
            _json_bytes(
                {
                    "kind": "qmix_inference",
                    "version": 1,
                    "model": _MODELS[method],
                    "actor_digest": details["actor_digest"],
                    "input_scale": scale,
                    "spawn_frame": frame,
                    "epsilon": details["epsilon"] if exported else 0.0,
                    "tie_rule": details["tie_rule"] if exported else QMIX_TIE_RULE,
                }
            )
        ).hexdigest()
    if method != "mappo":
        return sha256(
            _json_bytes(
                {
                    "kind": "ppo_inference",
                    "version": 1,
                    "model": _MODELS[method],
                    "actor_digest": details["actor_digest"],
                    "input_scale": scale,
                    "spawn_frame": frame,
                }
            )
        ).hexdigest()
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


def _is_actor(row: dict[str, object], method: str = "mappo") -> bool:
    """Identify the learner's actor payload rows by their exact record path.

    PPO saves the whole current actor tree. QMIX saves only its Q-network
    parameters (current_variables.params) and PQN-VDN only its network
    (current_variables.network: parameters and statistics); the epsilon leaf
    stays in the state payload. method defaults to MAPPO.
    """
    prefix = [{"field": "carry"}, {"field": "history"}, {"field": "current_variables"}]
    path = cast(list[object], row["path"])
    if method == "qmix":
        return path[:4] == [*prefix, {"field": "params"}]
    if method == "pqn_vdn":
        return path[:4] == [*prefix, {"field": "network"}]
    return path[:3] == prefix


def _pqn_actor_item(network: PQNInferenceVariables) -> dict[str, Tree]:
    """Return PQN-VDN's on-disk actor item, Flax's variables dictionary.

    network is the in-memory PQNInferenceVariables (arrays or shape-only
    leaves). The item is ``{"params": ..., "batch_stats": ...}``, built by
    name, because Orbax stores a NamedTuple's metadata as a dictionary. Its
    leaves come in sorted-key order (statistics first), unlike the
    NamedTuple; :func:`_pqn_network` reverses this by name.
    """
    return {"params": network.params, "batch_stats": network.batch_stats}


def _pqn_network(item: Mapping[str, Tree]) -> PQNInferenceVariables:
    """Rebuild PQNInferenceVariables from an actor item by name.

    item must hold exactly "params" and "batch_stats". The result's leaves
    are in NamedTuple order (parameters first), the order of the learner's
    full-state rows, so restore can reinsert them without swapping the two.
    Raises ValueError for other keys.
    """
    if set(item) != {"params", "batch_stats"}:
        raise ValueError("PQN-VDN actor item needs exactly params and batch_stats")
    return PQNInferenceVariables(params=item["params"], batch_stats=item["batch_stats"])


def _actor_payload(state: Tree, method: str = "mappo") -> Tree:
    """Return the tree saved in actor/.

    PPO's actor tree, QMIX's Q-network parameters, or PQN-VDN's actor item
    (``_pqn_actor_item`` of the current network).
    """
    current = state.carry.history.current_variables
    if method == "pqn_vdn":
        return _pqn_actor_item(current.network)
    return current.params if method == "qmix" else current


def _state_payload(state: Tree, method: str = "mappo") -> dict[str, Any]:
    """Flatten state excluding the actor payload and zero-element leaves.

    state is a PPO, QMIX or PQN-VDN learner, or a shape-only template of one;
    method selects the actor rows (see _is_actor). Orbax rejects zero-element
    arrays.
    Their exact path, shape and dtype remain in the checked layout; restore
    rebuilds their empty storage from the template.
    """
    return {
        f"leaf_{index:05d}": leaf
        for index, (leaf, row) in enumerate(
            zip(jax.tree.leaves(state), _layout(state), strict=True)
        )
        if not _is_actor(row, method) and leaf.size
    }


def _counters(state: Tree, games: int, method: str) -> dict[str, int]:
    """Return a learner's saved counters as exact Python integers.

    PPO records updates and env_steps. QMIX and PQN-VDN add completed_blocks
    and learning_blocks; their "updates" means optimizer steps.
    """
    counters = {
        "updates": int(state.completed_updates),
        "env_steps": int(state.carry.progress.rounds) * games,
    }
    if method in ("qmix", "pqn_vdn"):
        counters["completed_blocks"] = int(state.completed_blocks)
        counters["learning_blocks"] = int(state.learning_blocks)
    return counters


def _check_qmix_counters(counters: object, qmix: QMIXConfig, games: int) -> None:
    """Check saved QMIX counters from metadata alone, before arrays are read.

    counters must hold exactly updates, env_steps, completed_blocks and
    learning_blocks as nonnegative plain integers, env_steps a whole number of
    rounds, updates equal to learning_blocks * epochs, and a reachable
    readiness-aware block sequence. Raises ValueError.
    """
    from marl_battlegrounds.training.qmix_learner import (
        _block_counts_possible,  # pyright: ignore[reportPrivateUsage]
    )

    value = _object(counters, "Checkpoint counters")
    if set(value) != _QMIX_COUNTERS or any(
        type(item) is not int or item < 0 for item in value.values()
    ):
        raise ValueError("QMIX checkpoint counters are malformed")
    if value["env_steps"] % games or value["updates"] != (
        value["learning_blocks"] * qmix.epochs
    ):
        raise ValueError("QMIX checkpoint counters are impossible")
    if not _block_counts_possible(
        value["env_steps"] // games,
        value["completed_blocks"],
        value["learning_blocks"],
        rollout_length=qmix.rollout_length,
        minimum=qmix.min_buffer_size,
    ):
        raise ValueError("QMIX checkpoint counters are impossible")


def _check_pqn_counters(
    counters: object, pqn: PQNConfig, games: int, total_rounds: int
) -> None:
    """Check saved PQN-VDN counters from metadata alone, before arrays are read.

    Parameters
    ----------
    counters : object
        The description's counters: exactly updates, env_steps,
        completed_blocks and learning_blocks as nonnegative plain integers.
    pqn : PQNConfig
        Settings giving T, H, epochs and minibatches.
    games : int
        Number of games B; env_steps must be a whole number of rounds r.
    total_rounds : int
        The run's declared total rounds R.

    Raises
    ------
    ValueError
        A counter is malformed, or r is not a reachable collection boundary
        (``validation._collection_boundary`` with W initial rounds), or the
        blocks are not its ordinal, or learning_blocks is not
        ``max(0, ordinal - ceil(W / T))``, or updates is not
        ``learning_blocks * epochs * num_minibatches``.

    Notes
    -----
    Host-only arithmetic; totals may exceed int32 and stay exact.
    """
    from marl_battlegrounds.training.validation import (
        _collection_boundary,  # pyright: ignore[reportPrivateUsage]
    )

    value = _object(counters, "Checkpoint counters")
    if set(value) != _PQN_COUNTERS or any(
        type(item) is not int or item < 0 for item in value.values()
    ):
        raise ValueError("PQN-VDN checkpoint counters are malformed")
    rounds, remainder = divmod(value["env_steps"], games)
    if remainder or rounds > total_rounds:
        raise ValueError("PQN-VDN checkpoint counters are impossible")
    reachable, ordinal = _collection_boundary(
        rounds,
        total_env_steps=total_rounds,
        num_envs=1,
        rollout_length=pqn.rollout_length,
        initial_rounds=pqn.initial_rounds,
    )
    learning = max(0, ordinal - -(-pqn.initial_rounds // pqn.rollout_length))
    if (
        reachable != rounds
        or value["completed_blocks"] != ordinal
        or value["learning_blocks"] != learning
        or value["updates"] != learning * pqn.epochs * pqn.num_minibatches
    ):
        raise ValueError("PQN-VDN checkpoint counters are impossible")


def _check_method_settings(
    method: str, qmix: QMIXConfig | None, pqn: PQNConfig | None
) -> None:
    """Refuse settings objects that belong to another method.

    qmix may be given only for "qmix" and pqn only for "pqn_vdn". Raises
    ValueError naming the method family ("PPO checkpoints take no QMIX
    settings" keeps its historical words). Reads and writes nothing.
    """
    family = "PPO" if is_ppo_method(method) else method.upper().replace("_", "-")
    if method != "qmix" and qmix is not None:
        raise ValueError(f"{family} checkpoints take no QMIX settings")
    if method != "pqn_vdn" and pqn is not None:
        raise ValueError(f"{family} checkpoints take no PQN settings")


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
    when one is pinned, and a variables hook's identity (QMIX exploration) under
    "actor_variables_at_step" only when one is set, so descriptions of other
    runs keep their bytes.
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
    if collection.actor_variables_at_step is not None:
        details["actor_variables_at_step"] = _object(
            collection.actor_variables_at_step.identity, "Variables hook identity"
        )
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
    state: LearnerState | QMIXLearnerState | PQNLearnerState,
    *,
    metadata: dict[str, object],
    writer: RunWriter | None = None,
    ppo: PPOConfig = DEFAULT_PPO_CONFIG,
    method: str = "mappo",
    qmix: QMIXConfig | None = None,
    pqn: PQNConfig | None = None,
) -> Path:
    """Publish a complete update boundary, then replace its latest pointer.

    Parameters
    ----------
    run_dir : str or Path
        Existing run directory. The caller must hold its exclusive run lock.
    collection, state
        Matching stable collection descriptor and complete accepted learner
        (LearnerState for PPO, QMIXLearnerState for QMIX, PQNLearnerState for
        PQN-VDN). Initialization, QMIX warmup and PQN-VDN initial-collection
        boundaries are allowed; no actor decision occurs.
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
        input scale, spawn frame and value-normalization setting must equal the
        saved config's, and the state must carry matching statistics, else the
        save is refused before any file changes. The default's frame is "left".
    method : {"mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"}, \
default="mappo"
        Static learner method, matching metadata.config.method and the complete
        numerical state. Missing historical config method means MAPPO. Conflicts
        fail before obtaining a writer token or writing any payload.
    qmix : QMIXConfig or None, default=None
        QMIX settings; None means DEFAULT_QMIX_CONFIG. Used only when method
        is "qmix": every field must equal the saved config's "qmix" block,
        and validate_qmix_learner checks the state with it. It must be None
        for other methods, whose ppo argument is ignored for QMIX.
    pqn : PQNConfig or None, default=None
        PQN-VDN settings; None means DEFAULT_PQN_CONFIG. Used only when
        method is "pqn_vdn": every field must equal the saved config's "pqn"
        block, and validate_pqn_learner checks the state with it. It must be
        None for other methods.

    Returns
    -------
    Path
        Immutable content-identified checkpoint directory. All payload files and
        its latest_checkpoint.json pointer are durable on the local filesystem.
        A QMIX checkpoint's actor/ holds only the Q-network parameters; a
        PQN-VDN checkpoint's holds its network's params and batch_stats. Both
        add completed_blocks and learning_blocks to the counters.

    Raises
    ------
    ValueError
        State, context, writer or recorded boundary is incompatible.
    OSError
        Saving or publication fails. Earlier checkpoints remain intact; an
        unreferenced complete directory never becomes latest implicitly.
    """
    method = validate_training_method(method)
    root = _directory(Path(run_dir))
    context = _check_context(metadata, require_execution=True)
    if _config_method(context["config"]) != method:
        raise ValueError("Checkpoint config method differs from the selected method")
    _check_method_settings(method, qmix, pqn)
    if method == "qmix":
        from marl_battlegrounds.training.qmix_learner import validate_qmix_learner

        settings = DEFAULT_QMIX_CONFIG if qmix is None else qmix
        _check_qmix_settings(context["config"], settings)
        validate_qmix_learner(
            collection,
            cast("QMIXLearnerState", state),
            qmix=settings,
            recheck_installed_content=False,
        )
    elif method == "pqn_vdn":
        from marl_battlegrounds.training.pqn_learner import validate_pqn_learner

        chosen = DEFAULT_PQN_CONFIG if pqn is None else pqn
        _check_pqn_settings(context["config"], chosen)
        validate_pqn_learner(
            collection,
            cast("PQNLearnerState", state),
            pqn=chosen,
            recheck_installed_content=False,
        )
    else:
        from marl_battlegrounds.training.learner import validate_learner

        if _config_input_scale(context["config"]) != ppo.input_scale:
            raise ValueError("Checkpoint config input_scale differs from PPO settings")
        if _config_spawn_frame(context["config"]) != ppo.spawn_frame:
            raise ValueError("Checkpoint config spawn_frame differs from PPO settings")
        if _config_value_normalization(context["config"]) != ppo.value_normalization:
            raise ValueError(
                "Checkpoint config value_normalization differs from PPO settings"
            )
        validate_learner(
            collection,
            cast("LearnerState", state),
            ppo=ppo,
            method=method,
            recheck_installed_content=False,
        )
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
    actor = _actor_payload(state, method)
    _save_arrays(temporary / "actor", actor)
    _save_arrays(temporary / "state", _state_payload(state, method))
    details: dict[str, Any] = {
        "schema_version": 1,
        "kind": "learner",
        "schemas": checkpoint_schemas(method),
        "metadata": context,
        "collection": _collection_details(collection),
        "layout": _layout(state),
        "actor_layout": _layout(actor),
        "actor_digest": tree_digest(actor),
        "recording_token": token,
        "counters": _counters(state, collection.schedule.num_envs, method),
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

    Parameters
    ----------
    path : str or Path
        An actor or learner directory. It must exist, and no part of its path
        may be a symbolic link. The learner description
        (checkpoint_details.json) is read when present, otherwise the actor
        description (actor_details.json).

    Returns
    -------
    dict[str, Any]
        The verified description exactly as saved: schema_version (always the
        integer 1), checkpoint_id, kind ("learner" or "actor"), schemas,
        metadata, actor_layout, actor_digest and the files inventory, plus the
        kind's and method's own fields (for example a learner's collection,
        layout, recording_token and counters). Payload files are not read or
        checked.

    Raises
    ------
    ValueError
        The directory is missing or uses a symbolic link; the description is
        missing, a symbolic link, not a finite JSON object or has a repeated
        key; its hash differs from checkpoint_id; its schema, kind or field
        set is not a supported one; a QMIX or PQN-VDN rule below is broken; a
        learner's saved config names a different method than its model
        schema; or an inventory path, provenance, layout, digest, input scale
        or spawn frame is invalid.

    Notes
    -----
    This checks the description's own hash, schema, metadata and safe inventory
    paths without reading payload files. This supports ancestry after retention
    removes old numerical payloads. It does not make that ancestor restorable or
    prove its payload still exists. Use read_checkpoint_details before loading
    or restoring numerical data.

    A QMIX actor export must carry input_scale, epsilon exactly 0.0 (not -0.0),
    tie_rule "first_legal_maximum" and a nonnegative integer
    metadata.optimizer_steps; spawn_frame is present when not "world". A
    PQN-VDN export carries the same fields and always its spawn_frame. QMIX and
    PQN-VDN learners must carry exactly their four counters. Both the current
    schema dictionaries and the exact pre-Red-Zone ones (actor input schema 1)
    are accepted, so old descriptions stay readable for launch, ancestry and
    analysis. Host-only; no file is changed.
    """
    root = _directory(Path(path))
    name = _DESCRIPTION if (root / _DESCRIPTION).exists() else _ACTOR_DESCRIPTION
    details = _read_json(root / name)
    identifier = _digest(details.get("checkpoint_id"), "Checkpoint ID")
    unsigned = {k: v for k, v in details.items() if k != "checkpoint_id"}
    if sha256(_json_bytes(unsigned)).hexdigest() != identifier:
        raise ValueError("Checkpoint description digest differs")
    if type(details.get("schema_version")) is not int or details["schema_version"] != 1:
        raise ValueError("Checkpoint model or schema differs from this implementation")
    method = _schema_method(details.get("schemas"))
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
    elif method == "qmix":
        required |= {"input_scale", "epsilon", "tie_rule"}
        required |= {"spawn_frame"} if "spawn_frame" in details else set()
    elif method == "pqn_vdn":
        required |= {"input_scale", "spawn_frame", "epsilon", "tie_rule"}
    else:
        # Actor exports carry these optional inference settings only when set.
        required |= {key for key in ("input_scale", "spawn_frame") if key in details}
    if set(details) != required:
        raise ValueError("Checkpoint description fields differ from its schema")
    if method in ("qmix", "pqn_vdn") and details["kind"] == "actor":
        steps = _object(details["metadata"], "Actor provenance").get("optimizer_steps")
        if (
            type(details["epsilon"]) is not float
            or details["epsilon"] != 0.0
            or math.copysign(1.0, details["epsilon"]) < 0
            or details["tie_rule"] != QMIX_TIE_RULE
            or type(steps) is not int
            or steps < 0
        ):
            raise ValueError(
                f"{'QMIX' if method == 'qmix' else 'PQN-VDN'} export must be "
                "greedy with its optimizer count"
            )
    if method in ("qmix", "pqn_vdn") and details["kind"] == "learner":
        counters = _object(details["counters"], "Checkpoint counters")
        if set(counters) != _QMIX_COUNTERS:
            raise ValueError(
                f"{'QMIX' if method == 'qmix' else 'PQN-VDN'} checkpoint "
                "counters are malformed"
            )
    _object(details["metadata"], "Checkpoint provenance")
    if not isinstance(details["actor_layout"], list):
        raise ValueError("Checkpoint actor layout must be an array schema")
    _digest(details.get("actor_digest"), "Actor digest")
    files = _object(details.get("files"), "Checkpoint file inventory")
    for relative in files:
        _relative(root, relative)
    if details["kind"] == "learner":
        _check_context(details.get("metadata"))
        if _config_method(details["metadata"]["config"]) != method:
            raise ValueError("Checkpoint config method differs from its model schema")
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
    template: LearnerState | QMIXLearnerState | PQNLearnerState,
    *,
    expected_metadata: Mapping[str, object],
    ppo: PPOConfig = DEFAULT_PPO_CONFIG,
    method: str = "mappo",
    device: object | None = None,
    qmix: QMIXConfig | None = None,
    pqn: PQNConfig | None = None,
) -> RestoredCheckpoint:
    """Fully validate and restore numerical state without changing any files.

    Parameters
    ----------
    path : str or Path
        Explicit complete learner checkpoint; never a guessed latest directory.
    collection, template
        Fresh code-built descriptor/state from the saved seed and settings. All
        numerical template values are replaced; its static structure is trusted
        only after comparison with the saved content and settings. The template
        may be shape-only (``jax.ShapeDtypeStruct`` leaves), so a QMIX or
        PQN-VDN resume need not hold a second learner while restoring.
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
        Missing historical ppo.value_normalization means False on both sides.
        Enabled checkpoints carry three scalar statistics; disabled checkpoints
        retain their historical array paths and payload keys.
        Missing validation_opponents and slot_diagnostic_actor mean None.
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
        input scale, spawn frame and value-normalization setting must equal the
        checkpoint's, else restore
        is refused before arrays are read. The default's frame is "left", so a
        checkpoint trained in "world" needs the matching config passed here.
    method : {"mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"}, \
default="mappo"
        Static requested method. It must agree with saved/expected config and
        the model schema before any numerical arrays are restored. Missing
        historical config method means MAPPO, never a guess from array shapes.
    qmix : QMIXConfig or None, default=None
        QMIX settings (None means DEFAULT_QMIX_CONFIG), used only for "qmix".
        Scale and frame must equal the checkpoint's, every field must equal
        the saved config's "qmix" block, and the saved counters must be
        possible for these settings (``_check_qmix_counters``), all before any
        array is read. It must be None for other methods.
    pqn : PQNConfig or None, default=None
        PQN-VDN settings (None means DEFAULT_PQN_CONFIG), used only for
        "pqn_vdn", with the same metadata-only checks: scale, frame, every
        field of the saved "pqn" block, and counters that describe a reachable
        boundary of the collection's total (``_check_pqn_counters``). The
        actor item is reinserted into the full state by name (parameters,
        then statistics). It must be None for other methods.
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
        Faults that need only metadata (method, config, schema, layout
        including replay, counters) are raised before any array is read.
        A learner saved before Red Zone (actor input schema 1) is refused by
        _require_current_schemas right after the checkpoint kind check.
        No recording files or trainer logs have been changed.
    """
    method = validate_training_method(method)
    root = _directory(Path(path))
    details = read_checkpoint_details(root)
    if details["kind"] != "learner":
        raise ValueError("Actor exports cannot resume a learner")
    _require_current_schemas(details)
    if _schema_method(details["schemas"]) != method:
        raise ValueError("Checkpoint model differs from the requested method")
    expected = _object(dict(expected_metadata), "Expected execution metadata")
    if not {"config", "source", "dependencies", "execution"} <= expected.keys():
        raise ValueError(
            "Restore requires current config, source, dependencies and execution"
        )
    if _config_method(expected["config"]) != method:
        raise ValueError("Expected config method differs from the requested method")
    if expected["execution"] != execution_identity():
        raise ValueError("Expected execution identity differs from the current runtime")
    saved_metadata = dict(details["metadata"])
    # Configs saved before pinned_opponent_share existed compare at its default.
    for record in (saved_metadata, expected):
        config_record = record.get("config")
        if isinstance(config_record, dict):
            normalized = dict(cast(dict[str, Any], config_record))
            normalized.setdefault("method", "mappo")
            normalized.setdefault("pinned_opponent_share", 0.0)
            normalized.setdefault("pinned_opponent", None)
            normalized.setdefault("validation_opponents", None)
            normalized.setdefault("slot_diagnostic_actor", None)
            ppo_record = normalized.get("ppo", {})
            if is_ppo_method(normalized["method"]) and isinstance(ppo_record, dict):
                # Configs saved before spawn_frame existed compare at "world".
                nested = dict(cast(dict[str, Any], ppo_record))
                nested.setdefault("input_scale", 1.0)
                nested.setdefault("spawn_frame", "world")
                nested.setdefault("value_normalization", False)
                normalized["ppo"] = nested
            record["config"] = normalized
    if any(saved_metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint execution metadata differs")
    _check_method_settings(method, qmix, pqn)
    settings = DEFAULT_QMIX_CONFIG if qmix is None else qmix
    chosen = DEFAULT_PQN_CONFIG if pqn is None else pqn
    if method == "qmix":
        if _actor_input_scale(details) != settings.input_scale:
            raise ValueError("Checkpoint input_scale differs from QMIX settings")
        if _actor_spawn_frame(details) != settings.spawn_frame:
            raise ValueError("Checkpoint spawn_frame differs from QMIX settings")
        _check_qmix_settings(details["metadata"]["config"], settings)
        _check_qmix_counters(
            details["counters"], settings, collection.schedule.num_envs
        )
    elif method == "pqn_vdn":
        if _actor_input_scale(details) != chosen.input_scale:
            raise ValueError("Checkpoint input_scale differs from PQN settings")
        if _actor_spawn_frame(details) != chosen.spawn_frame:
            raise ValueError("Checkpoint spawn_frame differs from PQN settings")
        _check_pqn_settings(details["metadata"]["config"], chosen)
        _check_pqn_counters(
            details["counters"],
            chosen,
            collection.schedule.num_envs,
            collection.schedule.total_env_steps // collection.schedule.num_envs,
        )
    else:
        if _actor_input_scale(details) != ppo.input_scale:
            raise ValueError("Checkpoint input_scale differs from PPO settings")
        if _actor_spawn_frame(details) != ppo.spawn_frame:
            raise ValueError("Checkpoint spawn_frame differs from PPO settings")
        if (
            _config_value_normalization(details["metadata"]["config"])
            != ppo.value_normalization
        ):
            raise ValueError("Checkpoint value_normalization differs from PPO settings")
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
        _actor_payload(template, method)
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
        root / "actor", _actor_payload(template, method), selected_device
    )
    if tree_digest(actor) != details["actor_digest"]:
        raise ValueError("Restored actor digest differs")
    payload = _restore_arrays(
        root / "state", _state_payload(template, method), selected_device
    )
    # A PQN-VDN item's leaves are in sorted-key order (statistics first); the
    # full state holds the network in NamedTuple order, so rebuild it by name.
    actor_leaves = iter(
        jax.tree.leaves(_pqn_network(actor) if method == "pqn_vdn" else actor)
    )
    leaves: list[Any] = []
    for index, (row, initial) in enumerate(
        zip(details["layout"], jax.tree.leaves(template), strict=True)
    ):
        if _is_actor(row, method):
            leaves.append(next(actor_leaves))
        elif initial.size == 0:
            leaves.append(
                jax.device_put(np.empty(initial.shape, initial.dtype), selected_device)
            )
        else:
            leaves.append(payload[f"leaf_{index:05d}"])
    state = cast(
        "LearnerState | QMIXLearnerState | PQNLearnerState",
        jax.tree.unflatten(cast(Any, jax.tree.structure(template)), leaves),
    )
    if _layout(state) != details["layout"]:
        raise ValueError("Restored array schema differs")
    # The pinned checks read a few restored arrays; they run before the full
    # learner validation, which re-checks the installed content and is slow.
    _check_restored_pinned_opponent(collection, state)
    if method == "qmix":
        from marl_battlegrounds.training.qmix_learner import validate_qmix_learner

        validate_qmix_learner(
            collection, cast("QMIXLearnerState", state), qmix=settings
        )
    elif method == "pqn_vdn":
        from marl_battlegrounds.training.pqn_learner import validate_pqn_learner

        validate_pqn_learner(collection, cast("PQNLearnerState", state), pqn=chosen)
    else:
        from marl_battlegrounds.training.learner import validate_learner

        validate_learner(
            collection, cast("LearnerState", state), ppo=ppo, method=method
        )
    counters = _counters(state, collection.schedule.num_envs, method)
    if counters != details["counters"]:
        raise ValueError("Checkpoint counters disagree with numerical state")
    return RestoredCheckpoint(state, details, root)


def _check_restored_pinned_opponent(
    collection: TrainingCollection,
    state: LearnerState | QMIXLearnerState | PQNLearnerState,
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


@lru_cache(maxsize=16, typed=True)
def _actor_template(
    method: str = "mappo", actor_input_schema: int = ACTOR_INPUT_SCHEMA_VERSION
) -> Tree:
    """Cache actor shapes/dtypes for one checked method and actor input schema.

    Parameters
    ----------
    method : str, default="mappo"
        Training method: "mappo", "ippo", "ff_mappo", "ff_ippo", "qmix" or
        "pqn_vdn".
    actor_input_schema : int, default=ACTOR_INPUT_SCHEMA_VERSION
        2 for the current 5,165-feature actors, or 1 for actors saved before
        Red Zone (5,164 features).

    Returns
    -------
    PyTree
        Abstract ``jax.ShapeDtypeStruct`` actor leaves; no weights, mixer or
        optimizer arrays are allocated. The method-aware initializer is traced
        with eval_shape. For QMIX the tree is the Q-network parameters only
        (qmix_actor_template); for PQN-VDN it is the on-disk actor item,
        Flax's ``{"params", "batch_stats"}`` dictionary built from
        pqn_actor_template. The two schemas differ only in the input width,
        so the schema-1 tree is the current one with every 5,165-long input
        axis (the first Dense kernel's rows and, for PQN-VDN, the input
        BatchNorm leaves) set to 5,164; paths, order, dtypes and every other
        shape are the same. Tests prove it equals the recorded pre-Red-Zone
        layouts.

    Raises
    ------
    ValueError
        The method is unknown, or actor_input_schema is not the int 1 or 2.
        Raised before tracing.

    Notes
    -----
    Callers must not change this shared cached template; restore creates its
    own placement targets. The cache is typed, so True never reuses the
    template cached for 1 and is refused like any other non-int schema.
    """
    method = validate_training_method(method)
    if (
        type(actor_input_schema) is not int
        or actor_input_schema not in ACTOR_FEATURE_SIZES
    ):
        raise ValueError("actor_input_schema must be 1 or 2.")
    if method == "qmix":
        template = qmix_actor_template()
    elif method == "pqn_vdn":
        template = _pqn_actor_item(pqn_actor_template())
    else:
        template = jax.eval_shape(
            partial(initialize_ppo, method=validate_ppo_method(method)),
            jax.random.key(0),
        ).actor_params
    width = ACTOR_FEATURE_SIZES[actor_input_schema]
    if width == ACTOR_FEATURE_SIZE:
        return template

    def resize(leaf: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
        """Give one abstract leaf the schema's input width on its input axes."""
        shape = tuple(
            width if size == ACTOR_FEATURE_SIZE else size for size in leaf.shape
        )
        return jax.ShapeDtypeStruct(shape, leaf.dtype)

    return jax.tree.map(resize, template)


def export_system(
    actor_variables: Tree,
    destination: str | Path,
    *,
    metadata: dict[str, object],
    input_scale: float = 1.0,
    spawn_frame: str,
    method: str = "mappo",
) -> Path:
    """Publish a standalone immutable actor artifact (sampled PPO, greedy Q-values).

    Parameters
    ----------
    actor_variables : PyTree
        Finite variables matching the selected method's actor schema. For QMIX
        this is the Q-network parameters only, never epsilon or a mixer. For
        PQN-VDN it is the network's PQNInferenceVariables (parameters and
        BatchNorm statistics), never epsilon, optimizer state or kept rows.
    destination : str or Path
        Exact output directory with an existing parent. Matching exports are
        reused; different weights, scale or provenance at this path are rejected.
    metadata : dict
        Finite JSON provenance with run_id, nonnegative integer seed and
        env_steps, and the originating checkpoint_id. QMIX and PQN-VDN also
        need a nonnegative integer optimizer_steps. Extra facts are retained.
    input_scale : float, default=1.0
        Finite positive factor applied before the actor's first Dense layer.
        Use the originating learner config's value. It is part of
        inference identity; changing it does not rewrite weights or the raw
        input schema.
    spawn_frame : str
        Required. Spawn frame the weights were trained in: "world" or "left".
        Use the originating learner config's value. There is no default, because
        an export writes a durable identity: a guessed frame would silently
        label world weights as left, or the reverse. "left" is written into
        the description and the inference identity; "world" writes no key, so
        world exports keep their historical description bytes. PQN-VDN
        exports always write the frame.
    method : {"mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"}, \
default="mappo"
        Method that trained these weights. Its model schema is the export's
        sole method authority. Equal actor shapes/bytes do not make different
        methods interchangeable; an existing different-method path is rejected.
        QMIX and PQN-VDN exports also record epsilon 0.0 and tie_rule
        "first_legal_maximum": loading one gives the greedy System.

    Returns
    -------
    Path
        Absolute immutable export directory, after durable publication.

    Raises
    ------
    TypeError
        PQN-VDN variables are not a PQNInferenceVariables record.
    ValueError
        Variables, scale, frame or provenance are invalid (for PQN-VDN this
        includes a negative BatchNorm running variance), or the destination
        conflicts. Weights shaped for historical actor input schema 1 (5,164
        features, saved before Red Zone) are refused with a message saying to
        keep the original artifact, which load_system reads directly.
    OSError
        A required directory or payload cannot be read or written.

    Notes
    -----
    Host-only. No map preparation, critic, optimizer or game state is required.
    A PQN-VDN export saves the same actor item, with the same actor_digest, as
    the learner checkpoint it came from. Every TypeError or ValueError above
    is raised before any file or directory is written. Exports always record
    the current schemas.
    """
    method = validate_training_method(method)
    if method == "pqn_vdn":
        if not isinstance(cast(object, actor_variables), PQNInferenceVariables):
            raise TypeError("A PQN-VDN export takes PQNInferenceVariables")
        actor_variables = _pqn_actor_item(actor_variables)
    input_scale = _input_scale(input_scale)
    spawn_frame = _spawn_frame(spawn_frame)
    context = _object(metadata, "Actor provenance")
    if not {"run_id", "seed", "env_steps", "checkpoint_id"} <= context.keys():
        raise ValueError(
            "Actor provenance needs run_id, seed, env_steps and checkpoint_id"
        )
    if not isinstance(context["run_id"], str) or not context["run_id"]:
        raise ValueError("Actor run ID must be nonempty")
    counts = (
        ("seed", "env_steps", "optimizer_steps")
        if method in ("qmix", "pqn_vdn")
        else (
            "seed",
            "env_steps",
        )
    )
    for name in counts:
        if type(context.get(name)) is not int or context[name] < 0:
            raise ValueError(f"Actor {name} must be a nonnegative integer")
    _digest(context["checkpoint_id"], "Origin checkpoint")
    layout = _layout(actor_variables)
    if layout == _layout(_actor_template(method, 1)):
        raise ValueError(
            "These weights use historical actor input schema 1 (5,164 features). "
            "Export writes current actors only; keep the original artifact, "
            "which load_system reads directly."
        )
    if layout != _layout(_actor_template(method)):
        raise ValueError("Actor variables differ from the selected model schema")
    if not all(
        bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(actor_variables)
    ):
        raise ValueError("Actor variables must be finite")
    if method == "pqn_vdn" and not _nonnegative_variances(
        actor_variables["batch_stats"]
    ):
        raise ValueError("A PQN-VDN running variance is negative")
    target = Path(destination).absolute()
    parent = _directory(target.parent)
    digest = tree_digest(actor_variables)
    if target.exists():
        saved = read_checkpoint_details(target)
        if (
            saved["kind"] == "actor"
            and saved["schemas"] == checkpoint_schemas(method)
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
        "schemas": checkpoint_schemas(method),
        "metadata": context,
        "actor_layout": _layout(actor_variables),
        "actor_digest": digest,
        "input_scale": input_scale,
        "files": _inventory(temporary),
    }
    if method in ("qmix", "pqn_vdn"):
        details["epsilon"] = 0.0
        details["tie_rule"] = QMIX_TIE_RULE
    if spawn_frame != "world" or method == "pqn_vdn":
        details["spawn_frame"] = spawn_frame
    details["checkpoint_id"] = sha256(_json_bytes(details)).hexdigest()
    _atomic_json(temporary / _ACTOR_DESCRIPTION, details)
    _sync_tree(temporary)
    temporary.rename(target)
    _sync_dir(parent)
    return target


def artifact_identity(path: str | Path) -> dict[str, Any]:
    """Read verified actor identity/provenance without restoring numerical state.

    Parameters
    ----------
    path : str or Path
        A standalone actor export or a complete learner checkpoint directory.
        A pruned learner, whose payload files retention removed, is refused.

    Returns
    -------
    dict[str, Any]
        Direct actor_digest, weight_digest, input_scale, spawn_frame,
        checkpoint_id, schemas, run_id, seed and env_steps fields plus the
        original metadata. actor_digest binds model, weights, inference scale
        and spawn frame. MAPPO at scale 1.0 in the world frame keeps its
        historical weight digest. weight_digest always identifies only the
        saved variables. schemas identifies the method and the actor input
        schema (1 for artifacts saved before Red Zone, whose identities keep
        their original bytes). seed and env_steps come from an export's
        metadata or a learner's config and counters; seed may be None. PPO
        results carry no separate method field. QMIX and PQN-VDN results add
        "method" ("qmix" or "pqn_vdn") and the integer "optimizer_steps" (the
        export's recorded count, or a learner's saved "updates" counter),
        which validation summaries and selection use.

    Raises
    ------
    ValueError
        read_checkpoint_details rejects the artifact: its description is
        invalid (see read_checkpoint_description), or a payload file is
        missing, changed or unexpected.
    OSError
        A saved file cannot be read.

    Notes
    -----
    Neither identity proves competence. All file hashes are checked; this
    operation may read a full learner payload from disk. Missing historical
    scale means 1.0 and a missing frame means "world". Host-only: no array is
    restored and no file is changed.
    """
    details = read_checkpoint_details(path)
    metadata = details["metadata"]
    actor = details["kind"] == "actor"
    identity: dict[str, Any] = {
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
    method = _schema_method(details["schemas"])
    if method in ("qmix", "pqn_vdn"):
        identity["method"] = method
        identity["optimizer_steps"] = (
            metadata["optimizer_steps"] if actor else details["counters"]["updates"]
        )
    return identity


def load_system(checkpoint: str | Path) -> System:
    """Load an exact saved actor: a sampled PPO System or a greedy Q-value System.

    Parameters
    ----------
    checkpoint : str or Path
        Standalone actor export or complete learner checkpoint directory.

    Returns
    -------
    System
        Frozen numerical actor variables with their saved input scale and verified
        inference digest as the checkpoint label. The validated model schema
        chooses the actor architecture. Each evaluator creates fresh recurrent
        memory or empty feedforward memory. A PPO
        actor samples legal masked actions; a QMIX or PQN-VDN actor plays
        greedily (epsilon 0, first legal maximum), PQN-VDN with its saved
        BatchNorm statistics. No critic, mixer, target, replay, optimizer,
        kept row or training-only input is loaded, and no map preparation or
        full run directory is needed.

    Raises
    ------
    ValueError
        Artifact files, schemas, shapes, dtypes, digest or finite arrays differ,
        or a PQN-VDN actor has a negative BatchNorm running variance.
    OSError
        Required artifact bytes cannot be read.

    Notes
    -----
    Host-only; reads files and restores actor arrays onto the process's selected
    device. It needs the training extra but does not run an actor or learner.
    A frozen actor identity makes no claim about learned competence.
    Historical artifacts without an input scale keep their original scale 1.0
    and weight-digest identity. Learner checkpoints use their config's ppo,
    qmix or pqn input_scale;
    actor exports use their explicit saved scale. The saved spawn frame is
    restored the same way and is never inferred from a model's results; a
    missing frame means "world". Neither route rewrites weights. The saved
    actor input schema picks the template and the System route: an artifact
    saved before Red Zone (schema 1, 5,164 features) is restored with the
    schema-1 template and built with actor_input_schema=1, so it plays
    through the schema-1 hooks exactly as before. Its checkpoint label (the
    inference digest) keeps its bytes; its registration ID is new because
    the hook is new.
    """
    root = _directory(Path(checkpoint))
    details = read_checkpoint_details(root)
    method = _schema_method(details["schemas"])
    schema = cast(int, details["schemas"]["actor_input"])
    template = _actor_template(method, schema)
    if details["actor_layout"] != _layout(template):
        raise ValueError("Actor artifact schema differs from the installed model")
    actor = _restore_arrays(root / "actor", template, None)
    if tree_digest(actor) != details["actor_digest"] or not all(
        bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(actor)
    ):
        raise ValueError("Restored actor is nonfinite or its digest differs")
    if method == "pqn_vdn" and not _nonnegative_variances(actor["batch_stats"]):
        raise ValueError("A PQN-VDN running variance is negative")
    if method == "qmix":
        return make_qmix_system(
            actor,
            epsilon=0.0,
            checkpoint=_inference_digest(details),
            input_scale=_actor_input_scale(details),
            spawn_frame=_actor_spawn_frame(details),
            actor_input_schema=schema,
        )
    if method == "pqn_vdn":
        return make_pqn_system(
            _pqn_network(actor),
            epsilon=0.0,
            checkpoint=_inference_digest(details),
            input_scale=_actor_input_scale(details),
            spawn_frame=_actor_spawn_frame(details),
            actor_input_schema=schema,
        )
    return make_ppo_system(
        actor,
        method=method,
        checkpoint=_inference_digest(details),
        input_scale=_actor_input_scale(details),
        spawn_frame=_actor_spawn_frame(details),
        actor_input_schema=schema,
    )


def checkpoint_schemas(method: str = "mappo") -> dict[str, int | str]:
    """Return the installed checkpoint and selected model format identifiers.

    Parameters
    ----------
    method : {"mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"}, \
default="mappo"
        Static method. Unknown values raise ValueError through the shared
        method check. MAPPO keeps its historical keys, including learner_keys.

    Returns
    -------
    dict[str, int or str]
        A fresh JSON-ready copy of the checkpoint version, model name, actor
        input (2), training-state (2) and action versions, and collection/learner
        key versions. QMIX replaces learner_keys with qmix_keys and adds its
        replay row version. PQN-VDN has no training-state entry (it keeps no
        physical state) and adds pqn_keys, recent_window and normalization
        versions. These are the same identifiers checked during restore; new
        saves and exports write them. Artifacts saved before Red Zone carry
        the frozen dictionaries with actor input 1 (and training state 1),
        which load_system reads but export and resume refuse. Changing the
        returned dictionary does not change the shared authority.

    Notes
    -----
    Reads only fixed metadata; no file access or device initialization occurs.
    """
    if validate_training_method(method) == "qmix":
        return dict(_QMIX_SCHEMAS)
    if method == "pqn_vdn":
        return dict(_PQN_SCHEMAS)
    return {**_SCHEMAS, "model": _MODELS[validate_ppo_method(method)]}


def checkpoint_dependencies(method: str = "mappo") -> dict[str, str]:
    """Read installed numerical dependency versions for the runner's identity.

    Parameters
    ----------
    method : {"mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"}, \
default="mappo"
        Training method. QMIX also records Flashbax; PPO and PQN-VDN record
        exactly the historical set, so PPO identities keep their bytes.

    Returns
    -------
    dict[str, str]
        Python, JAX, JAXlib, NumPy, Flax, Optax and Orbax versions, plus
        Flashbax for QMIX only.

    Raises
    ------
    PackageNotFoundError
        An optional training package is missing.
    ValueError
        method is unknown.

    Notes
    -----
    Reads package metadata only; installs nothing and initializes no device.
    """
    import platform

    names = ["jax", "jaxlib", "numpy", "flax", "optax", "orbax-checkpoint"]
    if validate_training_method(method) == "qmix":
        names.append("flashbax")
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in names},
    }


def runtime_identity(method: str = "mappo") -> dict[str, object]:
    """Read the current imported-source identity and installed dependencies.

    method selects the dependency set (see checkpoint_dependencies); the
    default keeps the PPO record. The runner calls this at setup under a
    stable source snapshot, then reuses the result as its source/dependencies
    equality assertion for resume. Git or installed-package discovery reads
    files and may run read-only Git commands; no package is installed, device
    initialized or artifact written.
    """
    from marl_battlegrounds.evaluation.revision import discover_code_revision_v2

    return {
        "source": discover_code_revision_v2().model_dump(mode="json"),
        "dependencies": checkpoint_dependencies(method),
    }
