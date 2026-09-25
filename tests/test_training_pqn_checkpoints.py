"""Check PQN-VDN learner checkpoints, greedy exports and loading.

Contracts checked here, all on CPU with four games, blocks of 4 rounds, a
memory window of 2 (W = 6 initial rounds), one epoch of two minibatches and 16
rounds (the final block has 2 rounds). Learners saved at 0 rounds, at T = 4
(inside initial collection), at W = 6, at 10 (after learning) and at 14
(before the short final block), and a recorded learner saved at 10, each
restore in their own fresh process and continue one block exactly like the
uninterrupted run: the same next actions, shuffle permutations, kept rows and
memories, parameters, statistics, optimizer state, opponent snapshots and
counters. The actor payload is Flax's params and batch_stats dictionary, built
and read by name: its leaves start with the statistics while the learner's
network rows start with the parameters, and restore puts each leaf back in
place; epsilon stays in the state payload. A learner and its export share one
actor digest and inference identity, and a statistics-only change gives a
different identity. An export records epsilon 0.0 (a forged -0.0 is
rejected), the tie rule, its spawn frame even when "world", and its optimizer
count; it takes only PQNInferenceVariables, holds only actor files and loads
as the greedy System that equals the network from both spawn ends. Wrong
method, Boolean schema versions, changed settings, other methods' settings
and impossible, unreachable or overflowing counters, and a missing payload
file, fail before any array is read; negative or nonfinite statistics and
broken kept-row order or memory fail after loading and before any recovery
file; a corrupt export does not load. An actor item saved with its params and
batch_stats swapped, and a changed recording registration, fail before any
array is read. A failed pointer publication keeps the old pointer, the old
learner still restores and a new attempt saves. Checkpoint ancestry of PQN-VDN
learners lists child then parent with their four counters, and refuses a child
behind its parent, a re-signed parent and an edited parent. A pinned host
method whose games are unfinished refuses to resume a PQN-VDN learner before
any file changes (its two pinned lanes are placed by hand, because the
refusal runs before the learner check), while idle lanes resume. An export
refuses a negative BatchNorm running variance, and a forged export with one
does not load. A MAPPO learner can pin a PQN-VDN export. Loading a PQN-VDN
export never imports Flashbax, and its dependency record has none. Saved
PQN-VDN results must name the method and its optimizer count, each failure
with its own message. A network saved before Red Zone (actor input schema 1,
5,164 features) loads through the cached schema-1 PQN-VDN hook with its
inference identity bytes kept, cannot be exported again (the directory stays
unchanged), and its learner cannot resume: restore refuses it before any
array is read. The frozen pre-Red-Zone PQN-VDN schema dictionary differs from
the current one only in actor input 1. These checks prove software contracts,
not learning or GPU cost.
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
    _as_schema_1_learner,  # pyright: ignore[reportPrivateUsage]
    _files,  # pyright: ignore[reportPrivateUsage]
    _schema_1_round_trip,  # pyright: ignore[reportPrivateUsage]
    _writer,  # pyright: ignore[reportPrivateUsage]
)
from tests.test_training_pinned_resume import (
    _host_apply,  # pyright: ignore[reportPrivateUsage]
    _host_init,  # pyright: ignore[reportPrivateUsage]
)

import marl_battlegrounds.training.checkpoints as checkpoints
from marl_battlegrounds.baselines import pqn
from marl_battlegrounds.baselines.inputs import spawn_frame_flag
from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.baselines.qmix import QMIXConfig
from marl_battlegrounds.environment import make
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.tasks import balanced_spawn_configs
from marl_battlegrounds.training import _run_io as io_helpers
from marl_battlegrounds.training import make_training_schedule, prepare_training_content
from marl_battlegrounds.training import pqn_learner as learner
from marl_battlegrounds.training._compilation import (
    execution_identity,
    training_compiler_options,
)
from marl_battlegrounds.training._content import PreparedTrainingContent
from marl_battlegrounds.training.checkpoints import (
    artifact_identity,
    export_system,
    load_system,
    restore_checkpoint,
    save_checkpoint,
)
from marl_battlegrounds.training.collection import (
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    collect_training_rollout,
    scan_training_rollout,
)
from marl_battlegrounds.training.learner import init_learner

type Tree = Any
type Update = Callable[
    [learner.PQNLearnerState, TrainingCarry, TrainingRollout],
    tuple[learner.PQNLearnerState, learner.PQNUpdateResult],
]
_CONFIG = pqn.PQNConfig(rollout_length=4, memory_window=2, epochs=1, num_minibatches=2)
_GAMES, _TOTAL, _SEED = 4, 64, 19049141
_LENGTHS = (4, 2, 4, 4, 4)


def _equal(left: Tree, right: Tree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


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
    return learner.init_pqn_learner(
        schedule=make_training_schedule(total_env_steps=_TOTAL, num_envs=_GAMES),
        seed=_SEED,
        prepared=_content(),
        metrics="none",
        recording=recording,
        pqn=_CONFIG,
    )


@lru_cache
def _update() -> Update:
    planned = pqn.pqn_planned_learning_blocks(_TOTAL // _GAMES, _CONFIG)
    return cast(
        Update,
        jax.jit(
            partial(
                learner.update_pqn_learner,
                pqn=_CONFIG,
                planned_learning_blocks=planned,
            ),
            compiler_options=training_compiler_options(),
        ),
    )


def _metadata() -> dict[str, object]:
    return {
        "run_id": "pqn-checkpoint-test",
        "attempt_id": "first",
        "parent_checkpoint": None,
        "config": {
            "seed": _SEED,
            "num_envs": _GAMES,
            "total_env_steps": _TOTAL,
            "method": "pqn_vdn",
            "pqn": asdict(_CONFIG),
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
def _uninterrupted() -> _Run:
    collection, state = _learner()
    run = _Run([state], [])
    for length in _LENGTHS:
        carry, rollout = collect_training_rollout(
            collection, state.carry, length=length
        )
        state, result = _update()(state, carry, rollout)
        assert bool(result.accepted) and not bool(result.failed)
        run.states.append(state)
        run.rollouts.append(rollout)
    return run


def _save(root: Path, state: learner.PQNLearnerState) -> Path:
    return save_checkpoint(
        root,
        _learner()[0],
        state,
        metadata=_metadata(),
        method="pqn_vdn",
        pqn=_CONFIG,
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


def _permutations(state: learner.PQNLearnerState) -> str:
    return _digest(
        learner.epoch_permutations(
            state.shuffle_root, state.learning_blocks, _CONFIG, _GAMES
        )
    )


_FRESH = r"""
import hashlib, json, sys
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from marl_battlegrounds.baselines import pqn
from marl_battlegrounds.training import make_training_schedule, prepare_training_content
from marl_battlegrounds.training import pqn_learner
from marl_battlegrounds.training._compilation import training_compiler_options
from marl_battlegrounds.training.checkpoints import (
    finish_checkpoint_recovery, restore_checkpoint, resume_recording,
)
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
config = pqn.PQNConfig(**args["pqn"])
collection, fresh = pqn_learner.init_pqn_learner(
    schedule=make_training_schedule(
        total_env_steps=args["total"], num_envs=args["games"]
    ),
    seed=args["seed"],
    prepared=prepare_training_content(),
    metrics="none",
    recording=args["recording"],
    pqn=config,
)
template = jax.eval_shape(lambda value: value, fresh)
del fresh
restored = restore_checkpoint(
    args["path"],
    collection,
    template,
    expected_metadata=args["expected"],
    method="pqn_vdn",
    pqn=config,
)
writer = resume_recording(restored, args["run_dir"])
finish_checkpoint_recovery(restored, args["run_dir"])
state = restored.state
order = pqn_learner.epoch_permutations(
    state.shuffle_root, state.learning_blocks, config, args["games"]
)
planned = pqn.pqn_planned_learning_blocks(args["total"] // args["games"], config)
update = jax.jit(
    partial(
        pqn_learner.update_pqn_learner, pqn=config, planned_learning_blocks=planned
    ),
    compiler_options=training_compiler_options(),
)
carry, rollout = collect_training_rollout(
    collection, state.carry, length=args["length"], writer=writer
)
state, result = update(state, carry, rollout)
if writer is not None:
    writer.close()
print(json.dumps({
    "state": digest(state),
    "rollout": digest(rollout),
    "permutations": digest(order),
    "accepted": bool(result.accepted),
}))
"""


def _fresh(
    path: Path, run_dir: Path, length: int, *, recording: bool
) -> dict[str, Any]:
    arguments = {
        "pqn": asdict(_CONFIG),
        "total": _TOTAL,
        "games": _GAMES,
        "seed": _SEED,
        "path": str(path),
        "run_dir": str(run_dir),
        "length": length,
        "recording": recording,
        "expected": _expected(_metadata()),
    }
    completed = subprocess.run(
        [sys.executable, "-c", _FRESH, json.dumps(arguments)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_saved_boundaries_resume_in_fresh_processes(tmp_path: Path) -> None:
    run = _uninterrupted()
    assert [int(state.carry.progress.rounds) for state in run.states] == [
        0,
        4,
        6,
        10,
        14,
        16,
    ]
    cases: list[tuple[Path, Path, int, dict[str, Any], bool]] = []
    for index in range(5):
        root = tmp_path / f"boundary-{index}"
        root.mkdir()
        path = _save(root, run.states[index])
        expected = {
            "state": _digest(run.states[index + 1]),
            "rollout": _digest(run.rollouts[index]),
            "permutations": _permutations(run.states[index]),
            "accepted": True,
        }
        cases.append((path, root, _LENGTHS[index], expected, False))
    # The same boundary at 10 rounds, recorded through its writer.
    collection, state = _learner(recording=True)
    root = tmp_path / "recorded"
    root.mkdir()
    metadata = _metadata()
    writer = _writer(root, collection, metadata)
    assert writer is not None
    try:
        for length in _LENGTHS[:3]:
            carry, rollout = collect_training_rollout(
                collection, state.carry, length=length, writer=writer
            )
            state, _ = _update()(state, carry, rollout)
        path = save_checkpoint(
            root,
            collection,
            state,
            metadata=metadata,
            writer=writer,
            method="pqn_vdn",
            pqn=_CONFIG,
        )
        order = _permutations(state)
        carry, rollout = collect_training_rollout(
            collection, state.carry, length=4, writer=writer
        )
        following, _ = _update()(state, carry, rollout)
    finally:
        writer.close()
    _equal(following.recent, run.states[4].recent)
    _equal(following.opt_state, run.states[4].opt_state)
    cases.append(
        (
            path,
            root,
            4,
            {
                "state": _digest(following),
                "rollout": _digest(rollout),
                "permutations": order,
                "accepted": True,
            },
            True,
        )
    )
    for path, root, length, expected, recording in cases:
        assert _fresh(path, root, length, recording=recording) == expected


def test_the_actor_item_holds_the_network_by_name(tmp_path: Path) -> None:
    collection, template = _learner()
    state = _uninterrupted().states[3]
    network = state.carry.history.current_variables.network
    path = _save(tmp_path, state)
    details = checkpoints.read_checkpoint_details(path)
    assert details["schemas"] == checkpoints.checkpoint_schemas("pqn_vdn")
    item = checkpoints._pqn_actor_item(network)
    assert details["actor_layout"] == checkpoints._layout(item)
    rows = checkpoints._layout(state)
    actor_rows = [row for row in rows if checkpoints._is_actor(row, "pqn_vdn")]
    paths = [cast(list[Any], row["path"]) for row in actor_rows]
    assert all(path[3] == {"field": "network"} for path in paths)
    assert paths[0][4] == {"field": "params"}
    assert cast(list[Any], checkpoints._layout(item)[0]["path"])[0] == {
        "key": "batch_stats"
    }
    current = [{"field": "carry"}, {"field": "history"}, {"field": "current_variables"}]
    epsilon = [
        row
        for row in rows
        if cast(list[Any], row["path"])[:4] == [*current, {"field": "epsilon"}]
    ]
    assert len(epsilon) == 1 and not checkpoints._is_actor(epsilon[0], "pqn_vdn")
    for a, b in zip(
        jax.tree.leaves(checkpoints._pqn_network(item)),
        jax.tree.leaves(network),
        strict=True,
    ):
        assert a is b
    restored = restore_checkpoint(
        path,
        collection,
        jax.eval_shape(lambda value: value, template),
        expected_metadata=_expected(_metadata()),
        method="pqn_vdn",
        pqn=_CONFIG,
    )
    _equal(restored.state, state)


def _export(
    root: Path, network: pqn.PQNInferenceVariables, name: str, frame: str = "left"
) -> Path:
    return export_system(
        network,
        root / name,
        metadata={
            "run_id": "pqn-checkpoint-test",
            "seed": _SEED,
            "env_steps": 40,
            "checkpoint_id": "a" * 64,
            "optimizer_steps": 2,
        },
        spawn_frame=frame,
        method="pqn_vdn",
    )


def test_learner_and_export_share_one_identity_that_includes_statistics(
    tmp_path: Path,
) -> None:
    state = _uninterrupted().states[3]
    network = state.carry.history.current_variables.network
    learner_path = _save(tmp_path, state)
    export = _export(tmp_path, network, "actor")
    saved, exported = artifact_identity(learner_path), artifact_identity(export)
    assert exported["actor_digest"] == saved["actor_digest"]
    assert exported["weight_digest"] == saved["weight_digest"]
    assert exported["method"] == saved["method"] == "pqn_vdn"
    assert exported["optimizer_steps"] == saved["optimizer_steps"] == 2
    stats = cast(dict[str, Any], network.batch_stats)
    shifted = {
        **stats,
        "BatchNorm_0": {**stats["BatchNorm_0"], "var": stats["BatchNorm_0"]["var"] + 1},
    }
    changed = _export(tmp_path, network._replace(batch_stats=shifted), "changed")
    other = artifact_identity(changed)
    assert other["weight_digest"] != exported["weight_digest"]
    assert other["actor_digest"] != exported["actor_digest"]


@lru_cache
def _inputs() -> SystemInput:
    config = evaluation_env_config(team_sizes=(2, 2), max_steps=8)
    env = make(
        "tdm",
        env_config=balanced_spawn_configs(config, num_envs=2),
        num_envs=2,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(19049142))
    return env.policy_inputs(observations, state)


def test_a_greedy_export_loads_as_the_network(tmp_path: Path) -> None:
    state = _uninterrupted().states[3]
    network = state.carry.history.current_variables.network
    learner_path = _save(tmp_path, state)
    with pytest.raises(TypeError, match="PQNInferenceVariables"):
        export_system(
            checkpoints._pqn_actor_item(network),
            tmp_path / "dictionary",
            metadata={},
            spawn_frame="left",
            method="pqn_vdn",
        )
    with pytest.raises(ValueError, match="optimizer_steps"):
        export_system(
            network,
            tmp_path / "missing",
            metadata={
                "run_id": "pqn-checkpoint-test",
                "seed": _SEED,
                "env_steps": 40,
                "checkpoint_id": "a" * 64,
            },
            spawn_frame="left",
            method="pqn_vdn",
        )
    export = _export(tmp_path, network, "actor")
    details = checkpoints.read_checkpoint_details(export)
    assert details["epsilon"] == 0.0 and details["tie_rule"] == "first_legal_maximum"
    assert details["spawn_frame"] == "left"
    assert {Path(name).parts[0] for name in details["files"]} == {"actor"}
    world = checkpoints.read_checkpoint_details(
        _export(tmp_path, network, "world", frame="world")
    )
    assert world["spawn_frame"] == "world"
    forged = tmp_path / "negative-zero"
    shutil.copytree(export, forged)
    _resign(
        forged, lambda saved: saved.__setitem__("epsilon", -0.0), "actor_details.json"
    )
    with pytest.raises(ValueError, match="greedy"):
        checkpoints.read_checkpoint_details(forged)
    system = load_system(export)
    assert float(system.variables.epsilon) == 0.0
    assert system.checkpoint == artifact_identity(export)["actor_digest"]
    inputs = _inputs()
    flags = np.asarray(spawn_frame_flag(inputs.actors, "left"))
    assert flags.any() and not flags.all()
    keys = jax.random.split(jax.random.key(3), 2)
    memory = system.init(system.variables, inputs, keys)  # pyright: ignore[reportOptionalCall]
    loaded = cast(SystemOutput, system.apply(system.variables, memory, inputs, keys))
    reference = pqn.make_pqn_system(network, epsilon=0.0, spawn_frame="left")
    expected = cast(
        SystemOutput, reference.apply(reference.variables, memory, inputs, keys)
    )
    _equal(loaded, expected)
    _equal(load_system(learner_path).variables, system.variables)
    # A changed payload byte stops the export from loading.
    corrupt = tmp_path / "corrupt"
    shutil.copytree(export, corrupt)
    victim = next(
        path for path in sorted((corrupt / "actor").rglob("*")) if path.is_file()
    )
    victim.write_bytes(victim.read_bytes() + b"x")
    with pytest.raises(ValueError):
        load_system(corrupt)


def _forbid_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("A metadata fault must fail before any array is read")

    monkeypatch.setattr(checkpoints, "_restore_arrays", forbidden)


@pytest.mark.parametrize(
    "fault",
    [
        "method",
        "boolean_schema",
        "gamma",
        "scale",
        "qmix_settings",
        "counters_bool",
        "counters_unreachable",
        "counters_mismatch",
        "counters_overflow",
        "files",
    ],
)
def test_metadata_faults_fail_before_arrays_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    collection, template = _learner()
    path = _save(tmp_path, _uninterrupted().states[3])
    settings: dict[str, Any] = {"method": "pqn_vdn", "pqn": _CONFIG}
    if fault == "method":
        settings = {"method": "qmix", "qmix": QMIXConfig()}
    elif fault == "boolean_schema":
        _resign(path, lambda saved: saved["schemas"].__setitem__("normalization", True))
    elif fault == "gamma":
        settings["pqn"] = replace(_CONFIG, gamma=0.9)
    elif fault == "scale":
        settings["pqn"] = replace(_CONFIG, input_scale=0.5)
    elif fault == "qmix_settings":
        settings["qmix"] = QMIXConfig()
    elif fault == "files":
        victim = next(
            item for item in sorted((path / "state").rglob("*")) if item.is_file()
        )
        victim.unlink()
    else:
        counters = {
            "counters_bool": {"learning_blocks": True},
            "counters_unreachable": {"env_steps": 32, "completed_blocks": 3},
            "counters_mismatch": {"learning_blocks": 2},
            "counters_overflow": {"updates": 2**40},
        }[fault]
        _resign(path, lambda saved: saved["counters"].update(counters))
    _forbid_arrays(monkeypatch)
    before = _files(tmp_path)
    reason = {
        "method": "model differs",
        "boolean_schema": "schema differs",
        "gamma": "PQN settings differ",
        "scale": "input_scale differs",
        "qmix_settings": "take no QMIX settings",
        "counters_bool": "malformed",
        "files": "payload files",
    }.get(fault, "impossible")
    with pytest.raises(ValueError, match=reason):
        restore_checkpoint(
            path,
            collection,
            template,
            expected_metadata=_expected(_metadata()),
            **settings,
        )
    assert _files(tmp_path) == before


def test_a_schema_1_pqn_actor_loads_through_its_old_route_and_is_not_exported(
    tmp_path: Path,
) -> None:
    _schema_1_round_trip(tmp_path, "pqn_vdn", pqn._schema_1_pqn_actor_apply)


def test_a_pqn_learner_saved_before_red_zone_cannot_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, template = _learner()
    historical = _as_schema_1_learner(_save(tmp_path, template), "pqn_vdn")
    _forbid_arrays(monkeypatch)
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="saved before Red Zone"):
        restore_checkpoint(
            historical,
            collection,
            template,
            expected_metadata=_expected(_metadata()),
            method="pqn_vdn",
            pqn=_CONFIG,
        )
    assert _files(tmp_path) == before


def _no_check(*args: object, **kwargs: object) -> None:
    del args, kwargs


def _tampered(state: learner.PQNLearnerState, fault: str) -> learner.PQNLearnerState:
    history = state.carry.history
    variables = history.current_variables
    stats = cast(dict[str, Any], variables.network.batch_stats)
    rows = state.recent.rows
    if fault in ("variance", "nonfinite"):
        value = -1.0 if fault == "variance" else jnp.nan
        name = "var" if fault == "variance" else "mean"
        changed = {
            **stats,
            "BatchNorm_2": {
                **stats["BatchNorm_2"],
                name: stats["BatchNorm_2"][name].at[0].set(value),
            },
        }
        current = variables._replace(
            network=variables.network._replace(batch_stats=changed)
        )
        return state._replace(
            carry=state.carry._replace(
                history=history._replace(current_variables=current)
            )
        )
    if fault == "order":
        recent = state.recent._replace(
            rows=rows._replace(decision_step=rows.decision_step.at[1].add(5))
        )
    else:
        recent = state.recent._replace(
            pre_memory=state.recent.pre_memory.at[0].set(1.0),
            rows=rows._replace(episode_start=rows.episode_start.at[0].set(True)),
        )
    return state._replace(recent=recent)


@pytest.mark.parametrize("fault", ["variance", "nonfinite", "order", "memory"])
def test_array_faults_fail_after_loading_before_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    collection, template = _learner()
    state = _uninterrupted().states[3]
    with monkeypatch.context() as patch:
        patch.setattr(learner, "validate_pqn_learner", _no_check)
        path = _save(tmp_path, _tampered(state, fault))
    before = _files(tmp_path)
    reason = {
        "variance": "running variance",
        "nonfinite": "nonfinite",
        "order": "game order",
        "memory": "kept memory",
    }[fault]
    with pytest.raises(ValueError, match=reason):
        restore_checkpoint(
            path,
            collection,
            template,
            expected_metadata=_expected(_metadata()),
            method="pqn_vdn",
            pqn=_CONFIG,
        )
    assert _files(tmp_path) == before
    assert not (tmp_path / "checkpoint_recovery.json").exists()


def test_a_swapped_actor_item_fails_before_arrays_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, template = _learner()

    def swapped(network: pqn.PQNInferenceVariables) -> dict[str, Tree]:
        return {"params": network.batch_stats, "batch_stats": network.params}

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "_pqn_actor_item", swapped)
        path = _save(tmp_path, _uninterrupted().states[3])
    _forbid_arrays(monkeypatch)
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="array paths, shapes or dtypes differ"):
        restore_checkpoint(
            path,
            collection,
            template,
            expected_metadata=_expected(_metadata()),
            method="pqn_vdn",
            pqn=_CONFIG,
        )
    assert _files(tmp_path) == before


def test_a_failed_publication_keeps_the_pointer_and_a_new_attempt_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, template = _learner()
    state = _uninterrupted().states[3]
    original = _save(tmp_path, state)
    pointer = (tmp_path / "latest_checkpoint.json").read_bytes()
    publish = checkpoints._atomic_json

    def fail_pointer(path: Path, value: object) -> None:
        if path.name == "latest_checkpoint.json":
            raise OSError("injected pointer failure")
        publish(path, value)

    retry = {**_metadata(), "parent_checkpoint": original.name}
    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "_atomic_json", fail_pointer)
        with pytest.raises(OSError, match="pointer failure"):
            save_checkpoint(
                tmp_path,
                collection,
                _uninterrupted().states[4],
                metadata={**retry, "attempt_id": "orphan"},
                method="pqn_vdn",
                pqn=_CONFIG,
            )
    assert (tmp_path / "latest_checkpoint.json").read_bytes() == pointer
    restored = restore_checkpoint(
        original,
        collection,
        template,
        expected_metadata=_expected(_metadata()),
        method="pqn_vdn",
        pqn=_CONFIG,
    )
    _equal(restored.state, state)
    resumed = save_checkpoint(
        tmp_path,
        collection,
        _uninterrupted().states[4],
        metadata={**retry, "attempt_id": "retry"},
        method="pqn_vdn",
        pqn=_CONFIG,
    )
    latest = json.loads((tmp_path / "latest_checkpoint.json").read_bytes())
    assert latest["checkpoint_id"] == resumed.name != original.name


def test_a_corrupt_ancestor_is_refused(tmp_path: Path) -> None:
    collection = _learner()[0]
    parent = _save(tmp_path, _uninterrupted().states[3])
    child = save_checkpoint(
        tmp_path,
        collection,
        _uninterrupted().states[4],
        metadata={**_metadata(), "parent_checkpoint": parent.name},
        method="pqn_vdn",
        pqn=_CONFIG,
    )
    details = checkpoints.read_checkpoint_description(child)
    chain = io_helpers.checkpoint_ancestry(tmp_path, details)
    assert list(chain) == [child.name, parent.name]
    assert chain[parent.name]["counters"] == {
        "updates": 2,
        "env_steps": 40,
        "completed_blocks": 3,
        "learning_blocks": 1,
    }
    # A child whose counters fall behind its parent's is refused.
    behind = {**details, "counters": {**details["counters"], "learning_blocks": 0}}
    with pytest.raises(ValueError, match="follows its descendant"):
        io_helpers.checkpoint_ancestry(tmp_path, behind)
    copy = tmp_path / "parent-copy"
    shutil.copytree(parent, copy)
    # A re-signed parent no longer matches its directory's identity.
    _resign(parent, lambda saved: saved["counters"].update({"learning_blocks": 0}))
    with pytest.raises(ValueError, match="identities differ"):
        io_helpers.checkpoint_ancestry(tmp_path, details)
    # An edited parent no longer matches its own signed identity.
    shutil.rmtree(parent)
    shutil.copytree(copy, parent)
    description = parent / "checkpoint_details.json"
    changed = json.loads(description.read_text())
    changed["metadata"]["attempt_id"] = "forged"
    description.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        io_helpers.checkpoint_ancestry(tmp_path, details)


def test_a_host_pin_refuses_to_resume_a_pqn_learner_mid_game(tmp_path: Path) -> None:
    host = System("Host", _host_apply, init=_host_init, execution="host")
    collection, state = learner.init_pqn_learner(
        schedule=make_training_schedule(
            total_env_steps=_TOTAL, num_envs=_GAMES, early_history_capture=True
        ),
        seed=_SEED,
        prepared=_content(),
        metrics="none",
        pinned_opponent_share=0.5,
        pinned_opponent=host,
        pqn=_CONFIG,
    )
    assert collection.host_opponent is not None and collection.host_opponent.stateful
    metadata = _metadata()
    config = cast(dict[str, object], metadata["config"])
    config.update(pinned_opponent_share=0.5, pinned_opponent="Host")
    history = state.carry.history
    # A real pinned lane needs a published snapshot. The refusal runs before
    # the learner check, so two lanes are placed on slot 0 by hand.
    mid_game = state._replace(
        carry=state.carry._replace(
            history=history._replace(
                lane_snapshot=jnp.asarray((-1, 0, -1, 0), jnp.int32)
            )
        )
    )
    for name in ("mid-game", "idle"):
        (tmp_path / name).mkdir()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(learner, "validate_pqn_learner", _no_check)
        path = save_checkpoint(
            tmp_path / "mid-game",
            collection,
            mid_game,
            metadata=metadata,
            method="pqn_vdn",
            pqn=_CONFIG,
        )
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="games are unfinished"):
        restore_checkpoint(
            path,
            collection,
            state,
            expected_metadata=_expected(metadata),
            method="pqn_vdn",
            pqn=_CONFIG,
        )
    assert _files(tmp_path) == before
    idle = save_checkpoint(
        tmp_path / "idle",
        collection,
        state,
        metadata=metadata,
        method="pqn_vdn",
        pqn=_CONFIG,
    )
    restore_checkpoint(
        idle,
        collection,
        state,
        expected_metadata=_expected(metadata),
        method="pqn_vdn",
        pqn=_CONFIG,
    )


def test_a_changed_recording_registration_fails_before_arrays_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, state = _learner(recording=True)
    metadata = _metadata()
    writer = _writer(tmp_path, collection, metadata)
    assert writer is not None
    try:
        path = save_checkpoint(
            tmp_path,
            collection,
            state,
            metadata=metadata,
            writer=writer,
            method="pqn_vdn",
            pqn=_CONFIG,
        )
    finally:
        writer.close()
    record = cast(dict[str, Any], metadata["recording"])
    run_details = tmp_path / record["relative_path"] / "run_details.json"
    saved = json.loads(run_details.read_text())
    for value in saved["passes"].values():
        value["details"] = {"changed": True}
    run_details.write_text(json.dumps(saved))
    _forbid_arrays(monkeypatch)
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="registration differs"):
        restore_checkpoint(
            path,
            collection,
            state,
            expected_metadata=_expected(metadata),
            method="pqn_vdn",
            pqn=_CONFIG,
        )
    assert _files(tmp_path) == before
    assert not (tmp_path / "checkpoint_recovery.json").exists()


def _always_true(stats: Tree) -> bool:
    del stats
    return True


def test_a_negative_running_variance_blocks_export_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    network = _uninterrupted().states[3].carry.history.current_variables.network
    stats = cast(dict[str, Any], network.batch_stats)
    negative = network._replace(
        batch_stats={
            **stats,
            "BatchNorm_1": {
                **stats["BatchNorm_1"],
                "var": stats["BatchNorm_1"]["var"].at[3].set(-0.5),
            },
        }
    )
    with pytest.raises(ValueError, match="running variance is negative"):
        _export(tmp_path, negative, "refused")
    assert not (tmp_path / "refused").exists()
    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "_nonnegative_variances", _always_true)
        forged = _export(tmp_path, negative, "forged")
    with pytest.raises(ValueError, match="running variance is negative"):
        load_system(forged)


def test_counters_follow_the_offset_boundaries() -> None:
    games = 4
    for rounds, blocks, learning in (
        (0, 0, 0),
        (4, 1, 0),
        (6, 2, 0),
        (10, 3, 1),
        (14, 4, 2),
        (16, 5, 3),
    ):
        checkpoints._check_pqn_counters(
            {
                "updates": learning * 2,
                "env_steps": rounds * games,
                "completed_blocks": blocks,
                "learning_blocks": learning,
            },
            _CONFIG,
            games,
            16,
        )
    for rounds, blocks, learning, updates in (
        (8, 2, 0, 0),
        (8, 3, 1, 2),
        (12, 4, 2, 4),
        (10, 3, 1, 3),
        (18, 6, 4, 8),
    ):
        with pytest.raises(ValueError, match="impossible"):
            checkpoints._check_pqn_counters(
                {
                    "updates": updates,
                    "env_steps": rounds * games,
                    "completed_blocks": blocks,
                    "learning_blocks": learning,
                },
                _CONFIG,
                games,
                16,
            )


def test_a_mappo_learner_pins_a_pqn_export(tmp_path: Path) -> None:
    network = _uninterrupted().states[3].carry.history.current_variables.network
    export = _export(tmp_path, network, "actor")
    collection, state = init_learner(
        schedule=make_training_schedule(
            total_env_steps=32, num_envs=4, early_history_capture=True
        ),
        seed=19049143,
        prepared=_content(),
        ppo=PPOConfig(rollout_length=4, epochs=1),
        metrics="none",
        pinned_opponent_share=0.5,
        pinned_opponent=str(export),
    )
    record = collection.pinned_opponent
    assert record is not None and record["reference"] == str(export)
    carry = state.carry

    def fill(bank: jax.Array, leaf: jax.Array) -> jax.Array:
        return bank.at[0].set(leaf)

    history = carry.history._replace(
        count=jnp.int32(1),
        historical_variables=jax.tree.map(
            fill, carry.history.historical_variables, carry.history.current_variables
        ),
        lane_snapshot=jnp.asarray((-1, 0, -1, 0), jnp.int32),
    )
    scan = cast(
        Callable[[TrainingCarry], tuple[TrainingCarry, TrainingRollout]],
        jax.jit(partial(scan_training_rollout, collection, length=4)),
    )
    carry, rollout = scan(carry._replace(history=history))
    rows = rollout.transitions
    pinned = np.asarray(rows.opponent_snapshot) == 0
    assert pinned.any() and (np.asarray(rows.opponent_update)[pinned] == -2).all()
    widths = {leaf.shape[-1] for leaf in jax.tree.leaves(carry.memory.team_b)}
    assert 512 in widths


def test_loading_a_pqn_export_never_imports_flashbax(tmp_path: Path) -> None:
    network = _uninterrupted().states[3].carry.history.current_variables.network
    export = _export(tmp_path, network, "actor")
    script = (
        "import sys\n"
        "from marl_battlegrounds.training.checkpoints import "
        "checkpoint_dependencies, load_system\n"
        f"load_system({str(export)!r})\n"
        "assert 'flashbax' not in sys.modules\n"
        "print(sorted(checkpoint_dependencies('pqn_vdn')))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    names = json.loads(completed.stdout.strip().splitlines()[-1].replace("'", '"'))
    assert "flashbax" not in names and "orbax-checkpoint" in names


def test_pqn_schemas_and_result_fields() -> None:
    schemas = checkpoints.checkpoint_schemas("pqn_vdn")
    assert schemas == {
        "checkpoint": 1,
        "model": "recurrent_pqn_vdn_512",
        "actor_input": checkpoints.checkpoint_schemas("mappo")["actor_input"],
        "action": 1,
        "collection_keys": 1,
        "pqn_keys": 1,
        "recent_window": 1,
        "normalization": 1,
    }
    assert checkpoints._schema_method(schemas) == "pqn_vdn"
    with pytest.raises(ValueError):
        checkpoints._schema_method({**schemas, "recent_window": True})
    # The frozen pre-Red-Zone dictionary differs only in its actor input.
    historical = checkpoints._ACTOR_INPUT_1_SCHEMAS["pqn_vdn"]
    assert historical == {**schemas, "actor_input": 1}
    assert checkpoints._schema_method(historical) == "pqn_vdn"
    actor = {"optimizer_steps": 6}
    io_helpers._check_result_method(
        {"method": "pqn_vdn", "optimizer_steps": 6}, actor, "pqn_vdn"
    )
    for result, reason in (
        ({"method": "qmix", "optimizer_steps": 6}, "names another method"),
        ({"method": "pqn_vdn", "optimizer_steps": 5}, "differs from its optimizer"),
        ({"optimizer_steps": 6}, "names another method"),
    ):
        with pytest.raises(ValueError, match=f"PQN-VDN result {reason}"):
            io_helpers._check_result_method(result, actor, "pqn_vdn")
