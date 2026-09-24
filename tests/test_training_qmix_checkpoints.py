"""Check QMIX learner checkpoints, greedy exports and saved host counts.

Contracts checked here, all on CPU with small replay settings (buffer 12 rows
per game, minimum 6, sequences of 4, batches of 4, one epoch, hard copies every
2 steps) and two games over 20 rounds. A learner saved at initialization,
during warmup, just after a target copy and after the replay wraps restores
from disk and continues bitwise like the uninterrupted run (replay, cursors,
samples, actions, memory, optimizer, targets and counters), reusing the
compiled programs. A state after a short nonfinal block restores in a fresh
process and continues bitwise. A recorded run resumes with its writer.
Impossible QMIX counters, a wrong method, a forged Boolean replay schema, a
changed scale or other QMIX setting (such as gamma) or a layout mismatch fail
before any array is read; possible but wrong counters, wrong replay cursors,
epsilon or targets fail after loading but before any recovery file. A PPO save
refuses QMIX settings before writing anything.
The actor payload holds only the Q-network; epsilon stays in the state. A QMIX
export records epsilon 0.0 (a forged -0.0 is rejected), the tie rule and its
optimizer count, loads as the
greedy System that matches the Q-network's first legal maximum in both spawn
frames, and shares one inference identity with its learner checkpoint.
Schemas and dependencies stay exact for PPO and add QMIX's own. Saved host
totals follow fixed-block arithmetic (including totals above int32), and
saved results must carry QMIX's method and optimizer count, which PPO results
must not. These checks prove software contracts, not learning or GPU cost.
"""

# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.evaluation_fixtures import evaluation_env_config
from tests.test_training_checkpoints import (
    _files,  # pyright: ignore[reportPrivateUsage]
    _writer,  # pyright: ignore[reportPrivateUsage]
)

import marl_battlegrounds.training.checkpoints as checkpoints
from marl_battlegrounds.baselines import qmix
from marl_battlegrounds.baselines.actions import categorical_action_mask, encode_actions
from marl_battlegrounds.environment import make
from marl_battlegrounds.evaluation.policy_execution import SystemInput, SystemOutput
from marl_battlegrounds.tasks import balanced_spawn_configs
from marl_battlegrounds.training import _run_io as io_helpers
from marl_battlegrounds.training import make_training_schedule, prepare_training_content
from marl_battlegrounds.training import qmix_learner as learner
from marl_battlegrounds.training._compilation import (
    execution_identity,
    training_compiler_options,
)
from marl_battlegrounds.training._content import PreparedTrainingContent
from marl_battlegrounds.training.checkpoints import (
    artifact_identity,
    export_system,
    finish_checkpoint_recovery,
    load_system,
    restore_checkpoint,
    resume_recording,
    save_checkpoint,
)
from marl_battlegrounds.training.collection import (
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    collect_training_rollout,
)

type Tree = Any
type Update = Callable[
    [learner.QMIXLearnerState, TrainingCarry, TrainingRollout],
    tuple[learner.QMIXLearnerState, learner.QMIXUpdateResult],
]
_CONFIG = qmix.QMIXConfig(
    rollout_length=4,
    buffer_size=12,
    min_buffer_size=6,
    sample_sequence_length=4,
    sample_batch_size=4,
    epochs=1,
    update_period=2,
    eps_decay=40,
)
_GAMES, _TOTAL, _SEED = 2, 40, 19048401


def _equal(left: Tree, right: Tree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)


def _digest(tree: Tree) -> str:
    hasher = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        value = np.asarray(leaf)
        hasher.update(f"{value.dtype}{value.shape}".encode())
        hasher.update(value.tobytes())
    return hasher.hexdigest()


@lru_cache
def _content() -> PreparedTrainingContent:
    return prepare_training_content()


@lru_cache
def _learner(recording: bool = False) -> tuple[TrainingCollection, Any]:
    return learner.init_qmix_learner(
        schedule=make_training_schedule(total_env_steps=_TOTAL, num_envs=_GAMES),
        seed=_SEED,
        prepared=_content(),
        metrics="none",
        recording=recording,
        qmix=_CONFIG,
    )


@lru_cache
def _update() -> Update:
    return cast(
        Update,
        jax.jit(
            partial(learner.update_qmix_learner, qmix=_CONFIG),
            compiler_options=training_compiler_options(),
        ),
    )


def _metadata() -> dict[str, object]:
    return {
        "run_id": "qmix-checkpoint-test",
        "attempt_id": "first",
        "parent_checkpoint": None,
        "config": {
            "seed": _SEED,
            "num_envs": _GAMES,
            "total_env_steps": _TOTAL,
            "method": "qmix",
            "qmix": asdict(_CONFIG),
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


@dataclass
class _Run:
    states: list[Any]
    rollouts: list[TrainingRollout]


@lru_cache
def _uninterrupted(lengths: tuple[int, ...] = (4, 4, 4, 4, 4)) -> _Run:
    collection, state = _learner()
    run = _Run([state], [])
    for length in lengths:
        carry, rollout = collect_training_rollout(
            collection, state.carry, length=length
        )
        state, result = _update()(state, carry, rollout)
        assert bool(result.accepted) and not bool(result.failed)
        run.states.append(state)
        run.rollouts.append(rollout)
    return run


def _save(
    root: Path,
    state: learner.QMIXLearnerState,
    collection: TrainingCollection | None = None,
) -> Path:
    collection = _learner()[0] if collection is None else collection
    return save_checkpoint(
        root,
        collection,
        state,
        metadata=_metadata(),
        method="qmix",
        qmix=_CONFIG,
    )


def _resign(
    path: Path,
    change: Callable[[dict[str, Any]], None],
    name: str = "checkpoint_details.json",
) -> None:
    description = path / name
    details = json.loads(description.read_text())
    change(details)
    unsigned = {key: value for key, value in details.items() if key != "checkpoint_id"}
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(unsigned)
    ).hexdigest()
    description.write_text(json.dumps(details))


@pytest.mark.parametrize("index", [0, 1, 2, 4], ids=["init", "warmup", "copy", "wrap"])
def test_saved_boundaries_restore_and_continue_exactly(
    tmp_path: Path, index: int
) -> None:
    collection, template = _learner()
    run = _uninterrupted()
    saved = run.states[index]
    counts = (int(saved.completed_blocks), int(saved.learning_blocks))
    assert counts == [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3)][index]
    if index == 2:
        # The only update so far copied its targets at pre-step count 0.
        _equal(saved.target_q_params, saved.carry.history.current_variables.params)
    if index == 4:
        assert bool(saved.replay.is_full)
    path = _save(tmp_path, saved)
    details = checkpoints.read_checkpoint_details(path)
    assert details["schemas"] == checkpoints.checkpoint_schemas("qmix")
    assert set(details["counters"]) == {
        "updates",
        "env_steps",
        "completed_blocks",
        "learning_blocks",
    }
    assert details["actor_layout"] == checkpoints._layout(
        saved.carry.history.current_variables.params
    )
    restored = restore_checkpoint(
        path,
        collection,
        jax.eval_shape(lambda value: value, template),
        expected_metadata=_expected(_metadata()),
        method="qmix",
        qmix=_CONFIG,
    )
    _equal(restored.state, saved)
    # No leaf changes weak type, so only restore's committed placement differs.
    for before, after in zip(
        jax.tree.leaves(saved), jax.tree.leaves(restored.state), strict=True
    ):
        assert getattr(before, "weak_type", False) == getattr(after, "weak_type", False)
    update = _update()
    state = restored.state
    assert isinstance(state, learner.QMIXLearnerState)
    compiles = None
    for position in range(index, len(run.rollouts)):
        carry, rollout = collect_training_rollout(collection, state.carry, length=4)
        _equal(rollout, run.rollouts[position])
        state, _ = update(state, carry, rollout)
        _equal(state, run.states[position + 1])
        # Committed restored arrays compile once, as for PPO; then no more.
        if compiles is None:
            compiles = cast(Any, update)._cache_size()
        assert cast(Any, update)._cache_size() == compiles


_FRESH = r"""
import hashlib, json, sys
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from marl_battlegrounds.baselines import qmix
from marl_battlegrounds.training import make_training_schedule, prepare_training_content
from marl_battlegrounds.training import qmix_learner
from marl_battlegrounds.training._compilation import training_compiler_options
from marl_battlegrounds.training.checkpoints import restore_checkpoint
from marl_battlegrounds.training.collection import collect_training_rollout

def digest(tree):
    hasher = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        value = np.asarray(leaf)
        hasher.update(f"{value.dtype}{value.shape}".encode())
        hasher.update(value.tobytes())
    return hasher.hexdigest()

args = json.loads(sys.argv[1])
config = qmix.QMIXConfig(**args["qmix"])
collection, template = qmix_learner.init_qmix_learner(
    schedule=make_training_schedule(
        total_env_steps=args["total"], num_envs=args["games"]
    ),
    seed=args["seed"],
    prepared=prepare_training_content(),
    metrics="none",
    qmix=config,
)
restored = restore_checkpoint(
    args["path"],
    collection,
    template,
    expected_metadata=args["expected"],
    method="qmix",
    qmix=config,
)
update = jax.jit(
    partial(qmix_learner.update_qmix_learner, qmix=config),
    compiler_options=training_compiler_options(),
)
carry, rollout = collect_training_rollout(collection, restored.state.carry, length=4)
state, _ = update(restored.state, carry, rollout)
print(json.dumps({"state": digest(state), "rollout": digest(rollout)}))
"""


def test_short_block_boundary_resumes_in_a_fresh_process(tmp_path: Path) -> None:
    run = _uninterrupted((4, 4, 2, 4))
    short = run.states[3]
    assert (int(short.carry.progress.rounds), int(short.completed_blocks)) == (10, 3)
    assert int(run.rollouts[2].real_steps) == 2
    learner.validate_qmix_learner(
        _learner()[0], short, qmix=_CONFIG, recheck_installed_content=False
    )
    path = _save(tmp_path, short)
    arguments = {
        "qmix": asdict(_CONFIG),
        "total": _TOTAL,
        "games": _GAMES,
        "seed": _SEED,
        "path": str(path),
        "expected": _expected(_metadata()),
    }
    completed = subprocess.run(
        [sys.executable, "-c", _FRESH, json.dumps(arguments)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "state": _digest(run.states[4]),
        "rollout": _digest(run.rollouts[3]),
    }


def _forbid_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("A metadata fault must fail before any array is read")

    monkeypatch.setattr(checkpoints, "_restore_arrays", forbidden)


@pytest.mark.parametrize(
    "counters",
    [
        {"updates": 3, "env_steps": 24, "completed_blocks": 3, "learning_blocks": 3},
        {"updates": 2, "env_steps": 24, "completed_blocks": 3, "learning_blocks": 1},
        {"updates": 2, "env_steps": 25, "completed_blocks": 3, "learning_blocks": 2},
        {"updates": 2, "env_steps": 24, "completed_blocks": 3, "learning_blocks": True},
        {"updates": 2, "env_steps": 24, "completed_blocks": 3},
    ],
    ids=["readiness", "epochs", "rounds", "bool", "missing"],
)
def test_impossible_counters_fail_before_arrays_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, counters: dict[str, object]
) -> None:
    collection, template = _learner()
    path = _save(tmp_path, _uninterrupted().states[3])

    def change(details: dict[str, Any]) -> None:
        details["counters"] = counters

    _resign(path, change)
    _forbid_arrays(monkeypatch)
    with pytest.raises(ValueError, match="counters"):
        restore_checkpoint(
            path,
            collection,
            template,
            expected_metadata=_expected(_metadata()),
            method="qmix",
            qmix=_CONFIG,
        )


@pytest.mark.parametrize(
    "fault", ["method", "schema", "scale", "gamma", "layout", "ppo_args"]
)
def test_metadata_faults_fail_before_arrays_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    collection, template = _learner()
    path = _save(tmp_path, _uninterrupted().states[2])
    expected = _expected(_metadata())
    method, settings = "qmix", _CONFIG
    if fault == "method":
        method = "mappo"
    elif fault == "schema":

        def change(details: dict[str, Any]) -> None:
            details["schemas"]["replay"] = True

        _resign(path, change)
    elif fault == "scale":
        settings = replace(_CONFIG, input_scale=0.5)
    elif fault == "gamma":
        settings = replace(_CONFIG, gamma=0.9)
    elif fault == "layout":
        template = template._replace(
            replay=replace(
                template.replay,
                experience=jax.tree.map(_first_six_rows, template.replay.experience),
            )
        )
    _forbid_arrays(monkeypatch)
    before = _files(tmp_path)
    match = "QMIX settings differ" if fault == "gamma" else None
    with pytest.raises(ValueError, match=match):
        if fault == "ppo_args":
            restore_checkpoint(
                path,
                collection,
                template,
                expected_metadata=expected,
                method="mappo",
                qmix=_CONFIG,
            )
        else:
            restore_checkpoint(
                path,
                collection,
                template,
                expected_metadata=expected,
                method=method,
                qmix=settings,
            )
    assert _files(tmp_path) == before


def test_possible_but_wrong_counters_fail_after_loading(tmp_path: Path) -> None:
    collection, template = _learner()
    path = _save(tmp_path, _uninterrupted().states[2])

    # Three blocks of 2, 2 and 4 rounds could also reach 8 rounds with one
    # learning block, so only the loaded arrays reveal the forgery.
    def change(details: dict[str, Any]) -> None:
        details["counters"]["completed_blocks"] = 3

    _resign(path, change)
    with pytest.raises(ValueError, match="counters disagree"):
        restore_checkpoint(
            path,
            collection,
            template,
            expected_metadata=_expected(_metadata()),
            method="qmix",
            qmix=_CONFIG,
        )


def test_ppo_saves_refuse_qmix_settings_before_any_file(tmp_path: Path) -> None:
    collection = _learner()[0]
    metadata = _metadata()
    config = dict(cast(dict[str, Any], metadata["config"]))
    config["method"] = "mappo"
    del config["qmix"]
    metadata["config"] = config
    with pytest.raises(ValueError, match="PPO checkpoints take no QMIX settings"):
        save_checkpoint(
            tmp_path,
            collection,
            cast(Any, None),
            metadata=metadata,
            method="mappo",
            qmix=_CONFIG,
        )
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("fault", ["cursor", "epsilon", "target"])
def test_array_faults_fail_after_loading_before_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    collection, template = _learner()
    state = _uninterrupted().states[2]
    if fault == "cursor":
        state = state._replace(replay=replace(state.replay, current_index=jnp.int32(5)))
    elif fault == "epsilon":
        variables = state.carry.history.current_variables
        state = state._replace(
            carry=state.carry._replace(
                history=state.carry.history._replace(
                    current_variables=variables._replace(epsilon=jnp.float32(0.9))
                )
            )
        )
    else:
        state = state._replace(
            target_q_params=jax.tree.map(_nudge, state.target_q_params)
        )
    with monkeypatch.context() as patch:
        patch.setattr(learner, "validate_qmix_learner", _no_check)
        path = _save(tmp_path, state)
    before = _files(tmp_path)
    with pytest.raises(ValueError, match=r"cursor|exploration|targets"):
        restore_checkpoint(
            path,
            collection,
            template,
            expected_metadata=_expected(_metadata()),
            method="qmix",
            qmix=_CONFIG,
        )
    assert _files(tmp_path) == before
    assert not (tmp_path / "checkpoint_recovery.json").exists()


def test_recorded_run_resumes_with_its_writer(tmp_path: Path) -> None:
    collection, state = _learner(recording=True)
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)
    update = _update()
    try:
        for _ in range(2):
            carry, rollout = collect_training_rollout(
                collection, state.carry, length=4, writer=writer
            )
            state, _ = update(state, carry, rollout)
        path = save_checkpoint(
            tmp_path,
            collection,
            state,
            metadata=metadata,
            writer=writer,
            method="qmix",
            qmix=_CONFIG,
        )
        carry, rollout = collect_training_rollout(
            collection, state.carry, length=4, writer=writer
        )
        expected_state, _ = update(state, carry, rollout)
    finally:
        if writer is not None:
            writer.close()
    restored = restore_checkpoint(
        path,
        collection,
        state,
        expected_metadata=_expected(metadata),
        method="qmix",
        qmix=_CONFIG,
    )
    writer = resume_recording(restored, tmp_path)
    try:
        finish_checkpoint_recovery(restored, tmp_path)
        carry, resumed_rollout = collect_training_rollout(
            collection, restored.state.carry, length=4, writer=writer
        )
        assert isinstance(restored.state, learner.QMIXLearnerState)
        resumed, _ = update(restored.state, carry, resumed_rollout)
        _equal(resumed_rollout, rollout)
        _equal(resumed, expected_state)
    finally:
        if writer is not None:
            writer.close()


def _first_six_rows(value: jax.Array) -> jax.Array:
    return value[:, :6]


def _nudge(value: jax.Array) -> jax.Array:
    return value + 1e-3


def _no_check(*args: object, **kwargs: object) -> None:
    del args, kwargs


def _path(row: dict[str, object]) -> list[object]:
    return cast(list[object], row["path"])


@lru_cache
def _inputs() -> SystemInput:
    config = evaluation_env_config(team_sizes=(2, 2), max_steps=8)
    env = make(
        "tdm",
        env_config=balanced_spawn_configs(config, num_envs=2),
        num_envs=2,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(19048402))
    return env.policy_inputs(observations, state)


def test_greedy_export_matches_the_q_network_and_its_learner(tmp_path: Path) -> None:
    state = _uninterrupted().states[3]
    learner_path = _save(tmp_path, state)
    params = state.carry.history.current_variables.params
    provenance: dict[str, object] = {
        "run_id": "qmix-checkpoint-test",
        "seed": _SEED,
        "env_steps": int(state.carry.progress.rounds) * _GAMES,
        "checkpoint_id": learner_path.name,
        "optimizer_steps": int(state.completed_updates),
    }
    with pytest.raises(ValueError, match="optimizer_steps"):
        export_system(
            params,
            tmp_path / "missing",
            metadata={k: v for k, v in provenance.items() if k != "optimizer_steps"},
            spawn_frame="left",
            method="qmix",
        )
    with pytest.raises(ValueError, match="schema"):
        export_system(
            state.carry.history.current_variables,
            tmp_path / "wrong",
            metadata=provenance,
            spawn_frame="left",
            method="qmix",
        )
    export = export_system(
        params,
        tmp_path / "actor",
        metadata=provenance,
        spawn_frame="left",
        method="qmix",
    )
    details = checkpoints.read_checkpoint_details(export)
    assert details["epsilon"] == 0.0 and details["tie_rule"] == "first_legal_maximum"
    # A forged negative zero still equals 0.0 but is not the greedy record.
    forged = tmp_path / "negative-zero"
    shutil.copytree(export, forged)
    _resign(
        forged, lambda saved: saved.__setitem__("epsilon", -0.0), "actor_details.json"
    )
    with pytest.raises(ValueError, match="greedy"):
        checkpoints.read_checkpoint_details(forged)
    assert {Path(name).parts[0] for name in details["files"]} == {"actor"}
    exported, saved = artifact_identity(export), artifact_identity(learner_path)
    assert exported["actor_digest"] == saved["actor_digest"]
    assert exported["method"] == saved["method"] == "qmix"
    assert exported["optimizer_steps"] == saved["optimizer_steps"] == 2
    system = load_system(export)
    assert float(system.variables.epsilon) == 0.0
    assert system.checkpoint == exported["actor_digest"]
    inputs = _inputs()
    keys = jax.random.split(jax.random.key(3), 2)
    memory = system.init(system.variables, inputs, keys)  # pyright: ignore[reportOptionalCall]
    loaded = cast(SystemOutput, system.apply(system.variables, memory, inputs, keys))
    for frame in ("left", "world"):
        reference = qmix.make_qmix_system(params, spawn_frame=frame)
        expected = cast(
            SystemOutput, reference.apply(reference.variables, memory, inputs, keys)
        )
        indices = encode_actions(expected.actions)
        legal = categorical_action_mask(inputs.action_mask)
        assert bool(jnp.all(jnp.take_along_axis(legal, indices[..., None], axis=-1)))
        if frame == "left":
            _equal(loaded, expected)
    from_learner = load_system(learner_path)
    _equal(from_learner.variables, system.variables)


def test_schemas_and_dependencies_keep_ppo_bytes() -> None:
    ppo = {
        "checkpoint": 1,
        "actor_input": 2,
        "training_state": 1,
        "action": 1,
        "collection_keys": 1,
        "learner_keys": 1,
    }
    for method, model in (
        ("mappo", "recurrent_mappo_128"),
        ("ippo", "recurrent_ippo_128"),
        ("ff_mappo", "feedforward_mappo_128x128"),
        ("ff_ippo", "feedforward_ippo_128x128"),
    ):
        schemas = checkpoints.checkpoint_schemas(method)
        assert {
            key: value for key, value in schemas.items() if key != "actor_input"
        } == {
            **{key: value for key, value in ppo.items() if key != "actor_input"},
            "model": model,
        }
    assert checkpoints.checkpoint_schemas("qmix") == {
        **{key: value for key, value in ppo.items() if key != "learner_keys"},
        "actor_input": checkpoints.checkpoint_schemas("mappo")["actor_input"],
        "model": "recurrent_qmix_256",
        "qmix_keys": 1,
        "replay": 1,
    }
    assert "flashbax" not in checkpoints.checkpoint_dependencies()
    assert checkpoints.checkpoint_dependencies("qmix")["flashbax"] == "0.1.3"
    with pytest.raises(ValueError):
        checkpoints._schema_method(
            {**checkpoints.checkpoint_schemas("qmix"), "replay": True}
        )


def _qmix_host(steps: int, blocks: int, learning: int) -> dict[str, Any]:
    updates = learning * _CONFIG.epochs
    sequences = updates * _CONFIG.sample_batch_size
    pairs = sequences * (_CONFIG.sample_sequence_length - 1)
    host: dict[str, Any] = {
        "env_steps": steps,
        "completed_updates": updates,
        "actor_decisions": steps * 5,
        "completed_blocks": blocks,
        "learning_blocks": learning,
        "sampled_sequences": sequences,
        "used_td_pairs": pairs,
        "used_agent_utilities": pairs * 2,
        "sampled_exposure": {
            "by_stage": [pairs] + [0] * 16,
            "by_source": [pairs, 0],
            "by_opponent": [pairs] + [0] * 20,
        },
        "training_seconds": 1.0,
        "elapsed_seconds": 2.0,
        "routine_results": [],
        "confirmation_results": [],
        "actors": {},
        "pending": {"routine": False},
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
    if blocks:
        host.update(
            {
                "stage_completed": [[0, 0, 0] for _ in range(17)],
                "stage_score_sums": [[0, 0] for _ in range(17)],
                "stage_length_sum": [0] * 17,
                "stage_k20_count": [0] * 17,
            }
        )
    return host


def _qmix_description(
    root: Path, host: dict[str, Any], *, games: int, total: int
) -> dict[str, Any]:
    config = {
        "seed": 42,
        "num_envs": games,
        "total_env_steps": total,
        "method": "qmix",
        "qmix": asdict(_CONFIG),
        "validation_panel": None,
        "validation_fractions": [1.0],
        "checkpoint_env_steps": [],
        "routine_seed_pairs": 1,
        "confirmation_seed_pairs": 2,
        "slot_diagnostic": False,
    }
    details: dict[str, Any] = {
        "schema_version": 1,
        "kind": "learner",
        "schemas": checkpoints.checkpoint_schemas("qmix"),
        "metadata": {
            "run_id": "one-run",
            "attempt_id": "original",
            "parent_checkpoint": None,
            "config": config,
            "source": {"fixed": True},
            "dependencies": {"fixed": True},
            "execution": execution_identity(),
            "host_state": host,
        },
        "actor_layout": [],
        "actor_digest": "b" * 64,
        "files": {},
        "collection": {},
        "layout": [],
        "recording_token": None,
        "counters": {
            "updates": host["completed_updates"],
            "env_steps": host["env_steps"],
            "completed_blocks": host["completed_blocks"],
            "learning_blocks": host["learning_blocks"],
        },
    }
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)
    ).hexdigest()
    directory = root / "checkpoints" / details["checkpoint_id"]
    directory.mkdir(parents=True)
    io_helpers.atomic_json(directory / "checkpoint_details.json", details)
    return details


def test_saved_host_counts_follow_fixed_blocks_beyond_int32(tmp_path: Path) -> None:
    # 20 rounds in blocks of 4: blocks 2 to 5 reach the 6-row minimum.
    good = _qmix_description(
        tmp_path / "small", _qmix_host(40, 5, 4), games=2, total=40
    )
    io_helpers.validate_host_state(tmp_path / "small", good, panel=None)
    games, rounds = 1024, 2**30
    steps = games * rounds
    blocks, learning = io_helpers.qmix_fixed_block_counts(
        rounds, rollout_length=4, minimum=6
    )
    big = _qmix_host(steps, blocks, learning)
    assert big["used_td_pairs"] > 2**31
    details = _qmix_description(tmp_path / "big", big, games=games, total=steps)
    io_helpers.validate_host_state(tmp_path / "big", details, panel=None)
    for field, value in (
        ("learning_blocks", 5),
        ("used_td_pairs", 1),
        ("used_agent_utilities", 1),
        (
            "sampled_exposure",
            {"by_stage": [0] * 17, "by_source": [0], "by_opponent": [0] * 21},
        ),
    ):
        host = _qmix_host(40, 5, 4)
        host[field] = value
        bad = _qmix_description(tmp_path / field, host, games=2, total=40)
        with pytest.raises(ValueError):
            io_helpers.validate_host_state(tmp_path / field, bad, panel=None)


def test_saved_results_carry_method_fields_only_for_qmix() -> None:
    actor = {"optimizer_steps": 7}
    io_helpers._check_result_method(
        {"method": "qmix", "optimizer_steps": 7}, actor, "qmix"
    )
    io_helpers._check_result_method({"score": 0.5}, actor, "mappo")
    for result, method in (
        ({"method": "qmix", "optimizer_steps": 6}, "qmix"),
        ({"method": "qmix", "optimizer_steps": True}, "qmix"),
        ({"optimizer_steps": 7}, "qmix"),
        ({"method": "qmix"}, "mappo"),
        ({"optimizer_steps": 7}, "mappo"),
    ):
        with pytest.raises(ValueError):
            io_helpers._check_result_method(result, actor, method)


def test_actor_payload_rows_leave_epsilon_in_the_state() -> None:
    state = _uninterrupted().states[1]
    rows = checkpoints._layout(state)
    actor_rows = [row for row in rows if checkpoints._is_actor(row, "qmix")]
    assert actor_rows and all(
        _path(row)[3] == {"field": "params"} for row in actor_rows
    )
    epsilon = [
        row
        for row in rows
        if _path(row)[:4]
        == [
            {"field": "carry"},
            {"field": "history"},
            {"field": "current_variables"},
            {"field": "epsilon"},
        ]
    ]
    assert len(epsilon) == 1 and not checkpoints._is_actor(epsilon[0], "qmix")
    payload = checkpoints._state_payload(state, "qmix")
    assert len(payload) == sum(
        1
        for row, leaf in zip(rows, jax.tree.leaves(state), strict=True)
        if not checkpoints._is_actor(row, "qmix") and leaf.size
    )
