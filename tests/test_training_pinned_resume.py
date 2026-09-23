"""Check how a named pinned opponent is saved and when a resume may continue.

A checkpoint of a run with a named pinned System stores its record in the
collection block, while an unpinned run's block keeps its earlier keys only. A
saved config without the pinned_opponent key restores against today's config,
which carries it as None. A different pinned record is refused before any
array is read and leaves every file unchanged. A JAX pinned System with
memory whose weights start committed to a device enters the carry with
uncommitted weights like the rest of the carry. This test checks placement,
not compilation counts. Saved while its games are in progress, it restores
from the files alone
and continues exactly as the uninterrupted collection; saved pinned weights
that differ from the digest recorded at setup are refused and leave every file
unchanged. A pinned host method whose
memory is not saved refuses the resume, before any file changes, when one of
its games is unfinished; a memory-free host method, and a stateful one with no
pinned game in progress, resume normally.
Mixed recurrent/feedforward learners and pinned actors keep their distinct
memory trees through a saved in-progress game and its next partial rollout.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.training_learner_helpers import equal, updater

from marl_battlegrounds.baselines.ppo import (
    PPOConfig,
    initialize_ppo,
    make_ppo_system,
)
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.training import checkpoints
from marl_battlegrounds.training._compilation import execution_identity
from marl_battlegrounds.training._content import (
    PreparedTrainingContent,
    prepare_training_content,
)
from marl_battlegrounds.training.checkpoints import restore_checkpoint, save_checkpoint
from marl_battlegrounds.training.collection import (
    TrainingCollection,
    collect_training_rollout,
)
from marl_battlegrounds.training.curriculum import make_training_schedule
from marl_battlegrounds.training.learner import LearnerState, init_learner

type Tree = Any
PPO = PPOConfig(rollout_length=2, epochs=1, spawn_frame="world")


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


def _learner(
    prepared: PreparedTrainingContent, pinned: object | None, *, method: str = "mappo"
) -> tuple[TrainingCollection, LearnerState]:
    return init_learner(
        schedule=make_training_schedule(
            total_env_steps=20, num_envs=4, early_history_capture=pinned is not None
        ),
        seed=42,
        ppo=PPO,
        method=method,
        prepared=prepared,
        metrics="none",
        pinned_opponent_share=0.5 if pinned is not None else 0.0,
        pinned_opponent=pinned,  # pyright: ignore[reportArgumentType]
    )


def _assigned(state: LearnerState, snapshot: tuple[int, ...]) -> LearnerState:
    history = state.carry.history._replace(
        lane_snapshot=jnp.asarray(snapshot, jnp.int32)
    )
    return state._replace(carry=state.carry._replace(history=history))


def _metadata(
    pinned: str | None, *, with_key: bool = True, method: str | None = None
) -> dict[str, object]:
    config: dict[str, object] = {
        "seed": 42,
        "num_envs": 4,
        "total_env_steps": 20,
        "ppo": asdict(PPO),
        "pinned_opponent_share": 0.5 if pinned is not None else 0.0,
    }
    if with_key:
        config["pinned_opponent"] = pinned
    if method is not None:
        config["method"] = method
    return {
        "run_id": "pinned-resume",
        "attempt_id": "first",
        "parent_checkpoint": None,
        "config": config,
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


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_an_unpinned_block_keeps_its_keys_and_an_old_config_restores(
    prepared: PreparedTrainingContent, tmp_path: Path
) -> None:
    collection, state = _learner(prepared, None)
    old = _metadata(None, with_key=False)
    path = save_checkpoint(tmp_path, collection, state, metadata=old, ppo=PPO)
    assert (
        "pinned_opponent" not in checkpoints.read_checkpoint_details(path)["collection"]
    )
    restore_checkpoint(
        path,
        collection,
        state,
        expected_metadata=_expected(_metadata(None)),
        ppo=PPO,
    )


def test_a_different_pinned_record_is_refused_before_arrays(
    prepared: PreparedTrainingContent,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection, state = _learner(prepared, "tdm-alpha")
    metadata = _metadata("tdm-alpha")
    path = save_checkpoint(tmp_path, collection, state, metadata=metadata, ppo=PPO)
    changed = replace(
        collection,
        pinned_opponent={**(collection.pinned_opponent or {}), "name": "other"},
    )
    before = _files(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("A changed pinned opponent must be refused before arrays")

    monkeypatch.setattr(checkpoints, "_restore_arrays", forbidden)
    with pytest.raises(ValueError, match="pinned opponent differs"):
        restore_checkpoint(
            path, changed, state, expected_metadata=_expected(metadata), ppo=PPO
        )
    assert _files(tmp_path) == before


def _weighted_init(variables: Array, inputs: SystemInput, keys: Array) -> Array:
    del variables, keys
    return jnp.zeros(inputs.valid.shape, jnp.float32)


def _weighted_apply(
    variables: Array, memory: Array, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, Array]:
    del keys
    move = jnp.broadcast_to(
        (memory.astype(jnp.int32) % 9)[:, None], (memory.shape[0], 5)
    )
    zero = jnp.zeros_like(move)
    return ActorAction(move, zero, zero), memory + variables[0] * inputs.valid


def test_a_pinned_game_in_progress_resumes_exactly_and_changed_weights_refuse(
    prepared: PreparedTrainingContent, tmp_path: Path
) -> None:
    device = cast(Any, jax.devices()[0])
    weighted = System(
        "Weighted",
        _weighted_apply,
        init=_weighted_init,
        variables=jnp.ones(3, jnp.float32, device=device),
    )
    collection, initial = _learner(prepared, weighted)
    assert not any(
        leaf.committed for leaf in jax.tree.leaves(initial.carry.pinned_opponent)
    )
    update = updater(PPO)
    after, rollout = collect_training_rollout(collection, initial.carry, length=2)
    state, result = update(initial, after, rollout)
    assert bool(result.performed)
    # One block against the pinned System leaves both of its games unfinished.
    state = _assigned(state, (-1, 0, -1, 0))
    after, rollout = collect_training_rollout(collection, state.carry, length=2)
    state, result = update(state, after, rollout)
    assert bool(result.performed)
    np.testing.assert_array_equal(np.asarray(state.carry.memory.team_b[1])[[1, 3]], 2.0)
    metadata = _metadata(None)
    for name in ("saved", "changed"):
        (tmp_path / name).mkdir()
    path = save_checkpoint(
        tmp_path / "saved", collection, state, metadata=metadata, ppo=PPO
    )
    saved = checkpoints.read_checkpoint_details(path)["collection"]
    assert saved["pinned_opponent"] == collection.pinned_opponent
    _, uninterrupted = collect_training_rollout(collection, state.carry, length=2)
    # The template is the untrained start, so every value must come from files.
    restored = restore_checkpoint(
        path, collection, initial, expected_metadata=_expected(metadata), ppo=PPO
    )
    _, resumed = collect_training_rollout(collection, restored.state.carry, length=2)
    for left, right in zip(
        jax.tree.leaves(uninterrupted), jax.tree.leaves(resumed), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    variables, template = state.carry.pinned_opponent
    changed = state._replace(
        carry=state.carry._replace(pinned_opponent=(variables * 2, template))
    )
    bad = save_checkpoint(
        tmp_path / "changed", collection, changed, metadata=metadata, ppo=PPO
    )
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="weights differ"):
        restore_checkpoint(
            bad, collection, initial, expected_metadata=_expected(metadata), ppo=PPO
        )
    assert _files(tmp_path) == before


def _host_init(variables: Tree, inputs: SystemInput, keys: Array) -> list[int]:
    del variables, keys
    return [0] * len(np.asarray(inputs.valid))


def _host_apply(
    variables: Tree, memory: Tree, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, Tree]:
    del variables, inputs, keys
    zero = np.zeros((4, 5), np.int32)
    return ActorAction(zero, zero, zero), memory  # pyright: ignore[reportArgumentType]


def test_a_host_method_with_unsaved_memory_refuses_to_resume_mid_game(
    prepared: PreparedTrainingContent, tmp_path: Path
) -> None:
    host = System("Host", _host_apply, init=_host_init, execution="host")
    collection, state = _learner(prepared, host)
    assert collection.host_opponent is not None and collection.host_opponent.stateful
    # One real update captures slot 0, so a lane may legitimately be assigned to it.
    after, rollout = collect_training_rollout(collection, state.carry, length=2)
    state, result = updater(PPO)(state, after, rollout)
    assert bool(result.performed) and int(state.carry.history.count) == 1
    metadata = _metadata(None)
    for name in ("mid-game", "idle"):
        (tmp_path / name).mkdir()
    mid_game = save_checkpoint(
        tmp_path / "mid-game",
        collection,
        _assigned(state, (-1, 0, -1, 0)),
        metadata=metadata,
        ppo=PPO,
    )
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="games are unfinished"):
        restore_checkpoint(
            mid_game, collection, state, expected_metadata=_expected(metadata), ppo=PPO
        )
    assert _files(tmp_path) == before
    idle = save_checkpoint(
        tmp_path / "idle",
        collection,
        _assigned(state, (-1, -1, -1, -1)),
        metadata=metadata,
        ppo=PPO,
    )
    restore_checkpoint(
        idle, collection, state, expected_metadata=_expected(metadata), ppo=PPO
    )


def test_a_memory_free_host_method_resumes_with_games_in_progress(
    prepared: PreparedTrainingContent,
) -> None:
    host = System("Host", _host_apply, execution="host")
    collection, state = _learner(prepared, host)
    assert collection.host_opponent is not None
    assert not collection.host_opponent.stateful
    checkpoints._check_restored_pinned_opponent(
        collection, _assigned(state, (-1, 0, -1, 0))
    )


@pytest.mark.parametrize(
    ("method", "opponent_method"), [("ff_ippo", "ippo"), ("ippo", "ff_ippo")]
)
def test_mixed_actor_memories_resume_a_pinned_game_and_partial_update(
    prepared: PreparedTrainingContent,
    tmp_path: Path,
    method: str,
    opponent_method: str,
) -> None:
    opponent = make_ppo_system(
        initialize_ppo(jax.random.key(71), PPO, method=opponent_method).actor_params,
        method=opponent_method,
        spawn_frame=PPO.spawn_frame,
    )
    collection, initial = _learner(prepared, opponent, method=method)
    update = updater(PPO, method)
    carry, rollout = collect_training_rollout(collection, initial.carry, length=2)
    state, result = update(initial, carry, rollout)
    assert bool(result.performed) and not bool(result.failed)
    state = _assigned(state, (-1, 0, -1, 0))
    carry, rollout = collect_training_rollout(collection, state.carry, length=2)
    state, result = update(state, carry, rollout)
    assert bool(result.performed) and not bool(result.failed)
    own, pinned = state.carry.memory.team_b
    if method.startswith("ff_"):
        assert own == state.carry.memory.team_a == state.critic_memory == ()
        assert pinned.shape == (4, 5, 128)
        assert np.any(np.asarray(pinned)[[1, 3]] != 0)
    else:
        assert own.shape == (4, 5, 128)
        assert pinned == ()
    metadata = _metadata(None, method=method)
    cast(dict[str, object], metadata["config"])["pinned_opponent_share"] = 0.5
    path = save_checkpoint(
        tmp_path, collection, state, metadata=metadata, ppo=PPO, method=method
    )
    restored = restore_checkpoint(
        path,
        collection,
        initial,
        expected_metadata=_expected(metadata),
        ppo=PPO,
        method=method,
    )
    uninterrupted_carry, uninterrupted_rollout = collect_training_rollout(
        collection, state.carry, length=2
    )
    resumed_carry, resumed_rollout = collect_training_rollout(
        collection, restored.state.carry, length=2
    )
    uninterrupted, result = update(state, uninterrupted_carry, uninterrupted_rollout)
    resumed, resumed_result = update(restored.state, resumed_carry, resumed_rollout)
    assert int(resumed_rollout.real_steps) == 1
    assert bool(result.performed) and bool(resumed_result.performed)
    assert not bool(result.failed) and not bool(resumed_result.failed)
    equal(
        (state, uninterrupted_rollout, uninterrupted),
        (restored.state, resumed_rollout, resumed),
    )
