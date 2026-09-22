"""Check durable learner boundaries, exact actor loading and recording recovery.

CPU tests cover typed keys, the single stored actor, immutable attempt identity,
publication interruption, corrupt/schema/runtime rejection before recovery,
an empty update-zero recording token, strict content checks on restore and real
collect/update continuation with stable restored-array placement. These tests do
not establish GPU cost, cross-backend equality or learned competence. Actor
exports bind input scale to inference identity, preserve historical scale 1.0,
and reject learner scale mismatches before restoring arrays or changing files.
Shaping mode is saved with collection settings. Missing historical mode means
potential, and a different mode cannot reach array restore or output recovery.
The pinned opponent share is saved the same way: a missing historical key
restores at zero and a different share is rejected before arrays are restored.
The spawn frame follows the input-scale rules: an export must name its frame,
writes it only when it is not "world", a missing historical key loads as
"world", the version-1 identity envelope for a scaled world-frame actor is
pinned so existing identities cannot drift, a "left" export has its own
identity and loads as that frame,
an invalid saved frame and a learner frame mismatch are rejected before arrays
are restored, and an older learner config without the nested key resumes.
saved_training_config fixes a saved config's missing ppo.spawn_frame at "world"
without changing the checkpoint description, so the runner's resume keeps that
meaning whatever the current default is, and the runner rejects an explicit
resume config that names another frame before any file changes.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import marl_battlegrounds.training.checkpoints as checkpoints
import marl_battlegrounds.training.collection as collection_module
from marl_battlegrounds.baselines.ppo import PPOConfig, make_recurrent_mappo_system
from marl_battlegrounds.evaluation.policy_execution import apply_systems, init_systems
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
    tree_digest,
)
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.training import make_training_schedule, prepare_training_content
from marl_battlegrounds.training._compilation import (
    execution_identity,
    training_compiler_options,
)
from marl_battlegrounds.training._content import (
    PreparedTrainingContent,
    TrainingContentBinding,
)
from marl_battlegrounds.training.checkpoints import (
    artifact_identity,
    export_system,
    finish_checkpoint_recovery,
    load_system,
    read_checkpoint_details,
    restore_checkpoint,
    resume_recording,
    save_checkpoint,
)
from marl_battlegrounds.training.collection import (
    TrainingCollection,
    collect_training_rollout,
)
from marl_battlegrounds.training.learner import (
    LearnerState,
    init_learner,
    update_learner,
)

type Tree = Any
type Context = tuple[TrainingCollection, LearnerState]
PPO = PPOConfig(rollout_length=2, epochs=1, spawn_frame="world")


def _equal(left: Tree, right: Tree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module", params=[False, True], ids=["unrecorded", "recorded"])
def context(request: pytest.FixtureRequest) -> Context:
    return init_learner(
        schedule=make_training_schedule(total_env_steps=20, num_envs=4),
        seed=42,
        ppo=PPO,
        prepared=prepare_training_content(),
        metrics="none",
        recording=bool(request.param),
    )


def _metadata() -> dict[str, object]:
    return {
        "run_id": "checkpoint-test",
        "attempt_id": "first",
        "parent_checkpoint": None,
        "config": {
            "seed": 42,
            "num_envs": 4,
            "total_env_steps": 20,
            "ppo": asdict(PPO),
        },
        "source": {"scope": "test source identity"},
        "dependencies": {"scope": "test pinned dependencies"},
        "execution": execution_identity(),
        "host_state": {"selected": None},
        "log_cursors": {},
    }


def _expected(metadata: dict[str, object]) -> dict[str, object]:
    return {
        key: metadata[key] for key in ("config", "source", "dependencies", "execution")
    }


def _writer(
    root: Path, collection: TrainingCollection, metadata: dict[str, object]
) -> RunWriter | None:
    if not collection.recording:
        return None
    policies: dict[str, object] = {
        "team_a": normalize_system_registration(collection.actor, phase="training")[1],
        "team_b": normalize_system_registration(collection.opponent, phase="training")[
            1
        ],
    }
    writer = RunWriter(
        output_dir=root / "episodes", phase="training", policies=policies
    )
    metadata["recording"] = {
        "relative_path": writer.run_dir.relative_to(root).as_posix(),
        "phase": "training",
        "pass_id": "1",
        "policies": policies,
        "checkpoint_id": None,
        "details": {},
    }
    return writer


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_update_zero_roundtrip_and_recording_token(
    context: Context, tmp_path: Path
) -> None:
    collection, state = context
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)
    try:
        path = save_checkpoint(
            tmp_path, collection, state, metadata=metadata, writer=writer, ppo=PPO
        )
    finally:
        if writer is not None:
            writer.close()
    before = _files(tmp_path)
    restored = restore_checkpoint(
        path,
        collection,
        state,
        expected_metadata=_expected(metadata),
        ppo=PPO,
        device=cast(Any, jax.devices()[0]),
    )
    _equal(state, restored.state)
    for leaf in jax.tree.leaves(restored.state):
        assert isinstance(leaf, jax.Array)
        assert leaf.committed
        assert leaf.devices() == {jax.devices()[0]}
    assert restored.state.carry.env._full_ids.shape == (0,)  # pyright: ignore[reportPrivateUsage]
    assert restored.state.carry.env._replay_ids.shape == (0,)  # pyright: ignore[reportPrivateUsage]
    assert _files(tmp_path) == before
    assert str(jax.random.key_impl(restored.state.shuffle_root)) == "threefry2x32"
    assert restored.details["counters"] == {"updates": 0, "env_steps": 0}
    actor_paths = [
        row
        for row in restored.details["layout"]
        if row["path"][:3]
        == [{"field": "carry"}, {"field": "history"}, {"field": "current_variables"}]
    ]
    assert len(actor_paths) == len(
        jax.tree.leaves(state.carry.history.current_variables)
    )
    resumed = resume_recording(restored, tmp_path)
    try:
        assert (resumed is not None) == collection.recording
        assert (tmp_path / "checkpoint_recovery.json").is_file()
        finish_checkpoint_recovery(restored, tmp_path)
        assert not (tmp_path / "checkpoint_recovery.json").exists()
    finally:
        if resumed is not None:
            resumed.close()


def test_pinned_share_is_saved_and_a_missing_key_restores_at_default(
    context: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, state = context
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)
    try:
        path = save_checkpoint(
            tmp_path, collection, state, metadata=metadata, writer=writer, ppo=PPO
        )
    finally:
        if writer is not None:
            writer.close()
    details = read_checkpoint_details(path)
    assert details["collection"]["pinned_opponent_share"] == 0.0
    changed = replace(collection, pinned_opponent_share=0.1)

    def forbidden_restore(*args: object, **kwargs: object) -> None:
        pytest.fail("Changed pinned share must reject before restoring arrays")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "_restore_arrays", forbidden_restore)
        with pytest.raises(ValueError, match="collection settings"):
            restore_checkpoint(
                path, changed, state, expected_metadata=_expected(metadata), ppo=PPO
            )
    # A checkpoint saved before the key existed must restore at the default.
    del details["collection"]["pinned_opponent_share"]
    del details["checkpoint_id"]
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)  # pyright: ignore[reportPrivateUsage]
    ).hexdigest()
    (path / "checkpoint_details.json").write_text(json.dumps(details))
    historical = path.with_name(details["checkpoint_id"])
    path.rename(historical)
    restored = restore_checkpoint(
        historical,
        collection,
        state,
        expected_metadata=_expected(metadata),
        ppo=PPO,
        device=cast(Any, jax.devices()[0]),
    )
    _equal(state, restored.state)
    # Asymmetric normalization: the saved config lacks the key while the
    # expected config carries today's default; a positive expected share is
    # still rejected.
    expected = _expected(metadata)
    config = cast(dict[str, object], expected["config"])
    expected["config"] = {**config, "pinned_opponent_share": 0.0}
    restored = restore_checkpoint(
        historical,
        collection,
        state,
        expected_metadata=expected,
        ppo=PPO,
        device=cast(Any, jax.devices()[0]),
    )
    _equal(state, restored.state)
    expected["config"] = {**config, "pinned_opponent_share": 0.1}
    with pytest.raises(ValueError, match="execution metadata differs"):
        restore_checkpoint(
            historical, collection, state, expected_metadata=expected, ppo=PPO
        )


def test_saved_and_historical_shaping_mode_reject_incompatible_restore(
    context: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, state = context
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)
    try:
        path = save_checkpoint(
            tmp_path, collection, state, metadata=metadata, writer=writer, ppo=PPO
        )
    finally:
        if writer is not None:
            writer.close()
    details = read_checkpoint_details(path)
    assert details["collection"]["shaping_mode"] == "potential"
    changed = replace(collection, shaping_mode="score_delta")

    def forbidden_restore(*args: object, **kwargs: object) -> None:
        pytest.fail("Changed shaping mode must reject before restoring arrays")

    for historical in (False, True):
        if historical:
            del details["collection"]["shaping_mode"]
            del details["checkpoint_id"]
            details["checkpoint_id"] = hashlib.sha256(
                checkpoints._json_bytes(details)  # pyright: ignore[reportPrivateUsage]
            ).hexdigest()
            (path / "checkpoint_details.json").write_text(json.dumps(details))
            renamed = path.with_name(details["checkpoint_id"])
            path.rename(renamed)
            path = renamed
        before = _files(tmp_path)
        with monkeypatch.context() as patch:
            patch.setattr(checkpoints, "_restore_arrays", forbidden_restore)
            with pytest.raises(ValueError, match="collection settings"):
                restore_checkpoint(
                    path, changed, state, expected_metadata=_expected(metadata), ppo=PPO
                )
        assert _files(tmp_path) == before
        assert not (tmp_path / "checkpoint_recovery.json").exists()
    restored = restore_checkpoint(
        path, collection, state, expected_metadata=_expected(metadata), ppo=PPO
    )
    _equal(state, restored.state)
    assert _files(tmp_path) == before


def test_save_reuses_content_check_but_restore_rechecks_before_recovery(
    context: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, state = context
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)
    calls = []
    prepare = collection_module.prepare_training_content

    def checked_prepare(*, expected: TrainingContentBinding) -> PreparedTrainingContent:
        calls.append({"expected": expected})
        return prepare(expected=expected)

    monkeypatch.setattr(collection_module, "prepare_training_content", checked_prepare)
    try:
        path = save_checkpoint(
            tmp_path, collection, state, metadata=metadata, writer=writer, ppo=PPO
        )
    finally:
        if writer is not None:
            writer.close()
    assert calls == []
    before = _files(tmp_path)
    restore_checkpoint(
        path, collection, state, expected_metadata=_expected(metadata), ppo=PPO
    )
    assert calls == [{"expected": collection.binding}]

    def changed_content(**kwargs: object) -> None:
        raise ValueError("Installed content changed")

    monkeypatch.setattr(collection_module, "prepare_training_content", changed_content)
    with pytest.raises(ValueError, match="Installed content changed"):
        restore_checkpoint(
            path, collection, state, expected_metadata=_expected(metadata), ppo=PPO
        )
    assert _files(tmp_path) == before
    assert not (tmp_path / "checkpoint_recovery.json").exists()


def test_save_checks_actual_source_bank_without_rebuilding_installed_content(
    context: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, state = context
    bank = state.carry.tracking.source_configs
    assert bank is not None
    changed = state._replace(
        carry=state.carry._replace(
            tracking=replace(
                state.carry.tracking,
                source_configs=bank._replace(
                    max_steps=jnp.asarray(bank.max_steps).at[0].add(1)
                ),
            )
        )
    )
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)

    def forbidden_prepare(**kwargs: object) -> None:
        pytest.fail("Saving must reuse the descriptor's installed-content check")

    monkeypatch.setattr(
        collection_module, "prepare_training_content", forbidden_prepare
    )
    try:
        before = _files(tmp_path)
        with pytest.raises(ValueError, match="source bank"):
            save_checkpoint(
                tmp_path,
                collection,
                changed,
                metadata=metadata,
                writer=writer,
                ppo=PPO,
            )
        assert _files(tmp_path) == before
    finally:
        if writer is not None:
            writer.close()


@pytest.mark.parametrize(
    ("input_scale", "spawn_frame"),
    [(1.0, "world"), (0.01, "world"), (1.0, "left"), (0.01, "left")],
)
def test_export_load_is_independent_exact_and_immutable(
    context: Context, tmp_path: Path, input_scale: float, spawn_frame: str
) -> None:
    _, state = context
    provenance: dict[str, object] = {
        "run_id": "run",
        "seed": 42,
        "env_steps": 0,
        "checkpoint_id": "a" * 64,
    }
    with pytest.raises(TypeError, match="spawn_frame"):
        export_system(  # pyright: ignore[reportCallIssue]
            state.carry.history.current_variables,
            tmp_path / "unnamed",
            metadata=provenance,
        )
    assert not (tmp_path / "unnamed").exists()
    destination = tmp_path / "actor"
    export_system(
        state.carry.history.current_variables,
        destination,
        metadata=provenance,
        input_scale=input_scale,
        spawn_frame=spawn_frame,
    )
    actor = load_system(destination)
    identity = artifact_identity(destination)
    weight_digest = tree_digest(state.carry.history.current_variables)
    assert actor.checkpoint == identity["actor_digest"]
    assert identity["weight_digest"] == weight_digest
    assert identity["input_scale"] == input_scale
    assert identity["spawn_frame"] == spawn_frame
    assert ("spawn_frame" in read_checkpoint_details(destination)) == (
        spawn_frame != "world"
    )
    assert (actor.checkpoint == weight_digest) == (
        input_scale == 1.0 and spawn_frame == "world"
    )
    _equal(actor.variables, state.carry.history.current_variables)
    assert identity["seed"] == 42
    observations, env_state = state.carry.observations, state.carry.state
    memory = init_systems(actor, actor, observations, env_state, jax.random.key(71))
    expected = apply_systems(
        make_recurrent_mappo_system(
            state.carry.history.current_variables,
            input_scale=input_scale,
            spawn_frame=spawn_frame,
        ),
        actor,
        memory,
        observations,
        env_state,
        jax.random.key(72),
        variables_a=state.carry.history.current_variables,
    )
    actual = apply_systems(
        actor, actor, memory, observations, env_state, jax.random.key(72)
    )
    _equal(actual, expected)
    original = _files(destination)
    assert (
        export_system(
            actor.variables,
            destination,
            metadata=provenance,
            input_scale=input_scale,
            spawn_frame=spawn_frame,
        )
        == destination
    )
    assert _files(destination) == original
    with pytest.raises(ValueError, match="different artifact"):
        export_system(
            actor.variables,
            destination,
            metadata={**provenance, "env_steps": 4},
            input_scale=input_scale,
            spawn_frame=spawn_frame,
        )
    changed_scale = 0.5 if input_scale == 1.0 else 1.0
    changed_frame = "left" if spawn_frame == "world" else "world"
    with pytest.raises(ValueError, match="different artifact"):
        export_system(
            actor.variables,
            destination,
            metadata=provenance,
            input_scale=changed_scale,
            spawn_frame=spawn_frame,
        )
    with pytest.raises(ValueError, match="different artifact"):
        export_system(
            actor.variables,
            destination,
            metadata=provenance,
            input_scale=input_scale,
            spawn_frame=changed_frame,
        )
    for name, scale, frame in (
        ("other-scale", changed_scale, spawn_frame),
        ("other-frame", input_scale, changed_frame),
    ):
        other = export_system(
            actor.variables,
            tmp_path / name,
            metadata=provenance,
            input_scale=scale,
            spawn_frame=frame,
        )
        other_identity = artifact_identity(other)
        assert other_identity["actor_digest"] != identity["actor_digest"]
        assert other_identity["checkpoint_id"] != identity["checkpoint_id"]
        assert other_identity["weight_digest"] == identity["weight_digest"]
        assert other_identity["spawn_frame"] == frame
        assert load_system(other).checkpoint == other_identity["actor_digest"]


@pytest.mark.parametrize(
    "input_scale", [0.0, -1.0, float("nan"), float("inf"), True, "0.01"]
)
def test_export_rejects_invalid_input_scale_before_writing(
    tmp_path: Path, input_scale: object
) -> None:
    with pytest.raises(ValueError, match="input_scale"):
        export_system(
            None,
            tmp_path / "actor",
            metadata={},
            input_scale=cast(float, input_scale),
            spawn_frame="world",
        )
    assert not list(tmp_path.iterdir())


def test_historical_actor_default_and_invalid_saved_scale(
    context: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, state = context
    provenance: dict[str, object] = {
        "run_id": "run",
        "seed": 42,
        "env_steps": 0,
        "checkpoint_id": "a" * 64,
    }
    path = export_system(
        state.carry.history.current_variables,
        tmp_path / "legacy",
        metadata=provenance,
        spawn_frame="world",
    )
    details = read_checkpoint_details(path)
    del details["input_scale"]
    del details["checkpoint_id"]
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)  # pyright: ignore[reportPrivateUsage]
    ).hexdigest()
    (path / "actor_details.json").write_text(json.dumps(details))
    before = _files(path)
    actor = load_system(path)
    identity = artifact_identity(path)
    assert identity["actor_digest"] == tree_digest(
        state.carry.history.current_variables
    )
    assert identity["input_scale"] == 1.0
    observations, env_state = state.carry.observations, state.carry.state
    memory = init_systems(actor, actor, observations, env_state, jax.random.key(71))
    expected = apply_systems(
        collection.actor,
        actor,
        memory,
        observations,
        env_state,
        jax.random.key(72),
        variables_a=state.carry.history.current_variables,
    )
    actual = apply_systems(
        actor, actor, memory, observations, env_state, jax.random.key(72)
    )
    _equal(actual, expected)
    assert (
        export_system(actor.variables, path, metadata=provenance, spawn_frame="world")
        == path
    )
    assert _files(path) == before
    details["input_scale"] = 0.0
    del details["checkpoint_id"]
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)  # pyright: ignore[reportPrivateUsage]
    ).hexdigest()
    (path / "actor_details.json").write_text(json.dumps(details))

    def forbidden_restore(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid saved scale must reject before restoring arrays")

    monkeypatch.setattr(checkpoints, "_restore_arrays", forbidden_restore)
    before = _files(path)
    with pytest.raises(ValueError, match="input_scale"):
        load_system(path)
    assert _files(path) == before


def test_inference_identity_envelope_is_pinned_and_frames_extend_it() -> None:
    world_scaled = {"kind": "actor", "actor_digest": "a" * 64, "input_scale": 0.01}
    # Version-1 envelope digest computed from the committed formula; existing
    # scaled identities must never drift.
    assert (
        checkpoints._inference_digest(world_scaled)  # pyright: ignore[reportPrivateUsage]
        == "627796419b405ec0d11b648b9d3403c14234d7d50e693b1be0144198b873afef"
    )
    raw = {"kind": "actor", "actor_digest": "a" * 64}
    assert checkpoints._inference_digest(raw) == "a" * 64  # pyright: ignore[reportPrivateUsage]
    digests = {
        frame: checkpoints._inference_digest(  # pyright: ignore[reportPrivateUsage]
            {**world_scaled, "spawn_frame": frame}
        )
        for frame in ("world", "left")
    }
    assert digests["world"] == checkpoints._inference_digest(world_scaled)  # pyright: ignore[reportPrivateUsage]
    assert len(set(digests.values())) == 2
    unscaled_left = checkpoints._inference_digest(  # pyright: ignore[reportPrivateUsage]
        {**raw, "spawn_frame": "left"}
    )
    assert unscaled_left not in {"a" * 64, *digests.values()}


def test_historical_actor_default_frame_and_invalid_saved_frame(
    context: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, state = context
    provenance: dict[str, object] = {
        "run_id": "run",
        "seed": 42,
        "env_steps": 0,
        "checkpoint_id": "a" * 64,
    }
    path = export_system(
        state.carry.history.current_variables,
        tmp_path / "framed",
        metadata=provenance,
        input_scale=0.01,
        spawn_frame="left",
    )
    framed_identity = artifact_identity(path)
    details = read_checkpoint_details(path)
    assert details["spawn_frame"] == "left"
    del details["spawn_frame"]
    del details["checkpoint_id"]
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)  # pyright: ignore[reportPrivateUsage]
    ).hexdigest()
    (path / "actor_details.json").write_text(json.dumps(details))
    identity = artifact_identity(path)
    assert identity["spawn_frame"] == "world"
    assert identity["actor_digest"] != framed_identity["actor_digest"]
    assert identity["actor_digest"] == checkpoints._inference_digest(  # pyright: ignore[reportPrivateUsage]
        {
            "kind": "actor",
            "actor_digest": identity["weight_digest"],
            "input_scale": 0.01,
        }
    )
    assert (
        load_system(path).apply
        is make_recurrent_mappo_system({}, input_scale=0.01, spawn_frame="world").apply
    )
    details["spawn_frame"] = "up"
    del details["checkpoint_id"]
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)  # pyright: ignore[reportPrivateUsage]
    ).hexdigest()
    (path / "actor_details.json").write_text(json.dumps(details))

    def forbidden_restore(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid saved frame must reject before restoring arrays")

    monkeypatch.setattr(checkpoints, "_restore_arrays", forbidden_restore)
    before = _files(path)
    with pytest.raises(ValueError, match="spawn_frame"):
        load_system(path)
    assert _files(path) == before


def test_learner_frame_mismatch_rejects_before_files_and_old_config_resumes(
    context: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, state = context
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)
    try:
        before = _files(tmp_path)
        with pytest.raises(ValueError, match="spawn_frame"):
            save_checkpoint(
                tmp_path,
                collection,
                state,
                metadata=metadata,
                writer=writer,
                ppo=replace(PPO, spawn_frame="left"),
            )
        assert _files(tmp_path) == before
        path = save_checkpoint(
            tmp_path, collection, state, metadata=metadata, writer=writer, ppo=PPO
        )
    finally:
        if writer is not None:
            writer.close()

    def forbidden_restore(*args: object, **kwargs: object) -> None:
        pytest.fail("A different frame must reject before restoring arrays")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "_restore_arrays", forbidden_restore)
        with pytest.raises(ValueError, match="spawn_frame"):
            restore_checkpoint(
                path,
                collection,
                state,
                expected_metadata=_expected(metadata),
                ppo=replace(PPO, spawn_frame="left"),
            )
    # A learner config saved before spawn_frame existed compares at "world".
    details = read_checkpoint_details(path)
    del details["metadata"]["config"]["ppo"]["spawn_frame"]
    del details["checkpoint_id"]
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)  # pyright: ignore[reportPrivateUsage]
    ).hexdigest()
    (path / "checkpoint_details.json").write_text(json.dumps(details))
    historical = path.with_name(details["checkpoint_id"])
    path.rename(historical)
    assert artifact_identity(historical)["spawn_frame"] == "world"
    restored = restore_checkpoint(
        historical,
        collection,
        state,
        expected_metadata=_expected(metadata),
        ppo=PPO,
        device=cast(Any, jax.devices()[0]),
    )
    _equal(state, restored.state)


def test_learner_scale_load_and_config_mismatch_before_restore(
    context: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, state = context
    scaled_ppo = replace(PPO, input_scale=0.01)
    assert state.carry.tracking.source_configs is not None
    collection, state = init_learner(
        schedule=collection.schedule,
        seed=42,
        ppo=scaled_ppo,
        prepared=PreparedTrainingContent(
            binding=collection.binding,
            source_configs=state.carry.tracking.source_configs,
        ),
        metrics=collection.metrics,
        recording=collection.recording,
    )
    actor = make_recurrent_mappo_system(
        state.carry.history.current_variables,
        input_scale=scaled_ppo.input_scale,
        spawn_frame=scaled_ppo.spawn_frame,
    )
    metadata = _metadata()
    cast(dict[str, Any], metadata["config"])["ppo"] = asdict(scaled_ppo)
    writer = _writer(tmp_path, collection, metadata)
    try:
        before = _files(tmp_path)
        with pytest.raises(ValueError, match="input_scale"):
            save_checkpoint(
                tmp_path, collection, state, metadata=metadata, writer=writer, ppo=PPO
            )
        assert _files(tmp_path) == before
        path = save_checkpoint(
            tmp_path,
            collection,
            state,
            metadata=metadata,
            writer=writer,
            ppo=scaled_ppo,
        )
    finally:
        if writer is not None:
            writer.close()
    loaded = load_system(path)
    identity = artifact_identity(path)
    assert identity["input_scale"] == 0.01
    observations, env_state = state.carry.observations, state.carry.state
    memory = init_systems(actor, actor, observations, env_state, jax.random.key(71))
    _equal(
        apply_systems(
            actor, actor, memory, observations, env_state, jax.random.key(72)
        ),
        apply_systems(
            loaded, loaded, memory, observations, env_state, jax.random.key(72)
        ),
    )
    exported = export_system(
        loaded.variables,
        tmp_path / "scaled-export",
        metadata={
            "run_id": "run",
            "seed": 42,
            "env_steps": 0,
            "checkpoint_id": path.name,
        },
        input_scale=0.01,
        spawn_frame="world",
    )
    assert artifact_identity(exported)["actor_digest"] == identity["actor_digest"]
    before = _files(tmp_path)

    def forbidden_restore(*args: object, **kwargs: object) -> None:
        pytest.fail("Scale mismatch must reject before restoring arrays")

    monkeypatch.setattr(checkpoints, "_restore_arrays", forbidden_restore)
    with pytest.raises(ValueError, match="input_scale"):
        restore_checkpoint(
            path, collection, state, expected_metadata=_expected(metadata), ppo=PPO
        )
    assert _files(tmp_path) == before
    assert not (tmp_path / "checkpoint_recovery.json").exists()


@pytest.mark.parametrize(
    "mutation", ["file", "metadata", "config", "schema", "unsafe", "unexpected"]
)
def test_bad_checkpoint_is_rejected_without_recording_mutation(
    context: Context, tmp_path: Path, mutation: str
) -> None:
    collection, state = context
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)
    try:
        path = save_checkpoint(
            tmp_path, collection, state, metadata=metadata, writer=writer, ppo=PPO
        )
    finally:
        if writer is not None:
            writer.close()
    expected = _expected(metadata)
    if mutation == "file":
        path.joinpath("actor", "_METADATA").write_bytes(b"broken")
    elif mutation == "metadata":
        path.joinpath("checkpoint_details.json").write_text('{"checkpoint_id":"bad"}')
    elif mutation == "config":
        expected["config"] = {"seed": 999}
    elif mutation == "schema":
        state = state._replace(critic_memory=state.critic_memory[..., :1])
    elif mutation == "unsafe":
        child = next(p for p in (path / "actor").rglob("*") if p.is_file())
        data = child.read_bytes()
        outside = tmp_path / "outside"
        outside.write_bytes(data)
        child.unlink()
        child.symlink_to(outside)
    else:
        (path / "extra").write_text("unexpected")
    before = _files(tmp_path)
    with pytest.raises(ValueError):
        restore_checkpoint(path, collection, state, expected_metadata=expected, ppo=PPO)
    assert _files(tmp_path) == before
    assert not (tmp_path / "checkpoint_recovery.json").exists()


@pytest.mark.parametrize("refresh_expected", [False, True])
@pytest.mark.parametrize("setting", ["flags", "threefry_partitionable"])
def test_changed_execution_rejects_before_array_restore_or_writer_recovery(
    context: Context,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refresh_expected: bool,
    setting: str,
) -> None:
    collection, state = context
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)
    try:
        path = save_checkpoint(
            tmp_path, collection, state, metadata=metadata, writer=writer, ppo=PPO
        )
    finally:
        if writer is not None:
            writer.close()
    expected = _expected(metadata)

    def forbidden_restore(*args: object, **kwargs: object) -> None:
        pytest.fail("Changed execution must reject before numerical restore")

    monkeypatch.setattr(checkpoints, "_restore_arrays", forbidden_restore)
    before = _files(tmp_path)
    with ExitStack() as stack:
        if setting == "flags":
            monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=0")
        else:
            stack.enter_context(
                jax.threefry_partitionable(
                    not cast(Any, jax.config).jax_threefry_partitionable
                )
            )
        if refresh_expected:
            expected["execution"] = execution_identity()
        with pytest.raises(ValueError, match="execution"):
            restore_checkpoint(
                path, collection, state, expected_metadata=expected, ppo=PPO
            )
    assert _files(tmp_path) == before
    assert not (tmp_path / "checkpoint_recovery.json").exists()


def test_missing_execution_cannot_save_but_historical_actor_stays_readable(
    context: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, state = context
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)
    try:
        before = _files(tmp_path)
        missing = {key: value for key, value in metadata.items() if key != "execution"}
        with pytest.raises(ValueError, match="execution"):
            save_checkpoint(
                tmp_path, collection, state, metadata=missing, writer=writer, ppo=PPO
            )
        assert _files(tmp_path) == before
        path = save_checkpoint(
            tmp_path, collection, state, metadata=metadata, writer=writer, ppo=PPO
        )
    finally:
        if writer is not None:
            writer.close()
    details = read_checkpoint_details(path)
    del details["metadata"]["execution"]
    del details["metadata"]["config"]["ppo"]["input_scale"]
    del details["checkpoint_id"]
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)  # pyright: ignore[reportPrivateUsage]
    ).hexdigest()
    path.joinpath("checkpoint_details.json").write_text(json.dumps(details))
    historical = path.with_name(details["checkpoint_id"])
    path.rename(historical)
    before = _files(tmp_path)
    assert "execution" not in read_checkpoint_details(historical)["metadata"]
    assert (
        "execution"
        not in checkpoints.read_checkpoint_description(historical)["metadata"]
    )
    _equal(load_system(historical).variables, state.carry.history.current_variables)

    def forbidden_restore(*args: object, **kwargs: object) -> None:
        pytest.fail("Historical execution must reject before numerical restore")

    monkeypatch.setattr(checkpoints, "_restore_arrays", forbidden_restore)
    with pytest.raises(ValueError, match="execution"):
        restore_checkpoint(
            historical,
            collection,
            state,
            expected_metadata=_expected(metadata),
            ppo=PPO,
        )
    assert _files(tmp_path) == before
    assert not (tmp_path / "checkpoint_recovery.json").exists()


def test_directory_publication_failure_keeps_pointer_and_allows_new_attempt(
    context: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, state = context
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)
    try:
        original = save_checkpoint(
            tmp_path, collection, state, metadata=metadata, writer=writer, ppo=PPO
        )
        original_pointer = (tmp_path / "latest_checkpoint.json").read_bytes()
        publish = checkpoints._atomic_json  # pyright: ignore[reportPrivateUsage]

        def fail_pointer(path: Path, value: object) -> None:
            if path.name == "latest_checkpoint.json":
                raise OSError("injected pointer failure")
            publish(path, value)

        with monkeypatch.context() as patch:
            patch.setattr(checkpoints, "_atomic_json", fail_pointer)
            with pytest.raises(OSError, match="pointer failure"):
                save_checkpoint(
                    tmp_path,
                    collection,
                    state,
                    metadata={
                        **metadata,
                        "attempt_id": "orphan",
                        "parent_checkpoint": original.name,
                    },
                    writer=writer,
                    ppo=PPO,
                )
        assert (tmp_path / "latest_checkpoint.json").read_bytes() == original_pointer
        restored = restore_checkpoint(
            original, collection, state, expected_metadata=_expected(metadata), ppo=PPO
        )
        resumed = save_checkpoint(
            tmp_path,
            collection,
            restored.state,
            metadata={
                **metadata,
                "attempt_id": "retry",
                "parent_checkpoint": original.name,
            },
            writer=writer,
            ppo=PPO,
        )
        assert resumed != original
        assert (
            read_checkpoint_details(resumed)["metadata"]["parent_checkpoint"]
            == original.name
        )
        assert (
            json.loads((tmp_path / "latest_checkpoint.json").read_bytes())[
                "checkpoint_id"
            ]
            == resumed.name
        )
        assert (
            len(
                [
                    p
                    for p in (tmp_path / "checkpoints").iterdir()
                    if not p.name.startswith(".")
                ]
            )
            == 3
        )
    finally:
        if writer is not None:
            writer.close()


def test_complete_update_and_next_partial_block_reproduce_after_disk_restore(
    context: Context, tmp_path: Path
) -> None:
    collection, state = context
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)
    update = cast(
        Any,
        jax.jit(
            partial(update_learner, ppo=PPO),
            compiler_options=training_compiler_options(),
        ),
    )
    try:
        carry, rollout = collect_training_rollout(
            collection, state.carry, length=2, writer=writer
        )
        state, result = update(state, carry, rollout)
        assert bool(result.performed)
        path = save_checkpoint(
            tmp_path, collection, state, metadata=metadata, writer=writer, ppo=PPO
        )
        carry, rollout = collect_training_rollout(
            collection, state.carry, length=2, writer=writer
        )
        next_state, _ = update(state, carry, rollout)
        carry, last_rollout = collect_training_rollout(
            collection, next_state.carry, length=2, writer=writer
        )
        final_state, final_result = update(next_state, carry, last_rollout)
        assert int(last_rollout.real_steps) == 1
        assert bool(final_result.performed)
    finally:
        if writer is not None:
            writer.close()
    restored = restore_checkpoint(
        path, collection, state, expected_metadata=_expected(metadata), ppo=PPO
    )
    writer = resume_recording(restored, tmp_path)
    try:
        finish_checkpoint_recovery(restored, tmp_path)
        carry, restored_rollout = collect_training_rollout(
            collection, restored.state.carry, length=2, writer=writer
        )
        resumed, _ = update(restored.state, carry, restored_rollout)
        _equal(next_state, resumed)
        collector = collection_module._compiled_rollout(collection, 2)  # pyright: ignore[reportPrivateUsage]
        collection_compiles = collector._cache_size()
        update_compiles = update._cache_size()
        if not collection.recording:
            assert collection_compiles > 0
        assert update_compiles > 0
        carry, final_rollout = collect_training_rollout(
            collection, resumed.carry, length=2, writer=writer
        )
        resumed, _ = update(resumed, carry, final_rollout)
        _equal(final_state, resumed)
        _equal(last_rollout, final_rollout)
        assert collector._cache_size() == collection_compiles
        assert update._cache_size() == update_compiles
    finally:
        if writer is not None:
            writer.close()


def test_saved_training_config_fixes_a_missing_frame_at_world_without_mutation() -> (
    None
):
    saved: dict[str, Any] = {
        "metadata": {"config": {"seed": 3, "ppo": {"input_scale": 0.01}}}
    }
    fixed = checkpoints.saved_training_config(saved)
    assert fixed["ppo"] == {"input_scale": 0.01, "spawn_frame": "world"}
    assert fixed["seed"] == 3
    assert saved["metadata"]["config"] == {"seed": 3, "ppo": {"input_scale": 0.01}}
    without_block = checkpoints.saved_training_config(
        {"metadata": {"config": {"seed": 3}}}
    )
    assert without_block["ppo"] == {"spawn_frame": "world"}
    explicit = checkpoints.saved_training_config(
        {"metadata": {"config": {"ppo": {"spawn_frame": "left"}}}}
    )
    assert explicit["ppo"]["spawn_frame"] == "left"
    with pytest.raises(ValueError, match="JSON object"):
        checkpoints.saved_training_config({"metadata": {"config": {"ppo": []}}})


def test_resume_reads_a_missing_saved_frame_as_world_and_rejects_another(
    context: Context, tmp_path: Path
) -> None:
    from marl_battlegrounds.training import runner

    collection, state = context
    metadata = _metadata()
    saved_config = cast(dict[str, Any], metadata["config"])
    saved_ppo = dict(cast(dict[str, Any], saved_config["ppo"]))
    del saved_ppo["spawn_frame"]
    saved_config["ppo"] = saved_ppo
    writer = _writer(tmp_path, collection, metadata)
    try:
        path = save_checkpoint(
            tmp_path, collection, state, metadata=metadata, writer=writer, ppo=PPO
        )
    finally:
        if writer is not None:
            writer.close()
    details = checkpoints.read_checkpoint_details(path)
    assert "spawn_frame" not in details["metadata"]["config"]["ppo"]
    inherited = runner.config_from_dict(checkpoints.saved_training_config(details))
    assert inherited.ppo.spawn_frame == "world"
    explicit = replace(inherited, ppo=replace(inherited.ppo, spawn_frame="left"))
    with pytest.raises(ValueError, match="Resume config differs"):
        runner.train(explicit, resume_from=path)
