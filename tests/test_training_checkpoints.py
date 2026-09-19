"""Check durable learner boundaries, exact actor loading and recording recovery.

CPU tests cover typed keys, the single stored actor, immutable attempt identity,
publication interruption, corrupt/schema/runtime rejection before recovery,
an empty update-zero recording token, strict content checks on restore and real
collect/update continuation with stable restored-array placement. These tests do
not establish GPU cost, cross-backend equality or learned competence.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import marl_battlegrounds.training.checkpoints as checkpoints
import marl_battlegrounds.training.collection as collection_module
from marl_battlegrounds.baselines.ppo import PPOConfig
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
PPO = PPOConfig(rollout_length=2, epochs=1)


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
        "config": {"seed": 42, "num_envs": 4, "total_env_steps": 20},
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


def test_export_load_is_independent_exact_and_immutable(
    context: Context, tmp_path: Path
) -> None:
    collection, state = context
    provenance: dict[str, object] = {
        "run_id": "run",
        "seed": 42,
        "env_steps": 0,
        "checkpoint_id": "a" * 64,
    }
    destination = tmp_path / "actor"
    export_system(
        state.carry.history.current_variables, destination, metadata=provenance
    )
    actor = load_system(destination)
    assert actor.checkpoint == tree_digest(state.carry.history.current_variables)
    _equal(actor.variables, state.carry.history.current_variables)
    assert artifact_identity(destination)["seed"] == 42
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
    original = _files(destination)
    assert (
        export_system(actor.variables, destination, metadata=provenance) == destination
    )
    assert _files(destination) == original
    with pytest.raises(ValueError, match="different artifact"):
        export_system(
            actor.variables, destination, metadata={**provenance, "env_steps": 4}
        )


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
