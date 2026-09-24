"""Check that adding PQN-VDN left every older method's saved contracts unchanged.

The fixture ``tests/fixtures/method_contracts.json`` was captured on CPU before
PQN-VDN existed, for MAPPO, IPPO, feedforward MAPPO, feedforward IPPO and QMIX.
For each method this file recomputes and compares: the default
``config_to_dict`` bytes; the checkpoint schema dictionary; the dependency
names (Flashbax for QMIX only); the actor layout and the shape-only learner
state layout at 4 games and blocks of 4 rounds (as SHA256 digests of their
JSON); the inference identity of one fixed synthetic actor description in the
left frame; the System registration IDs of all-ones actors at scale 1.0 in
the world frame and scale 0.01 in the left frame; and the collection details
digest. It also checks the launch probe's source text. Only values that do not
depend on the run, the machine or the CPU thread count are compared.
"""

# pyright: reportPrivateUsage=false
import hashlib
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from marl_battlegrounds.baselines import ppo as ppo_module
from marl_battlegrounds.baselines import qmix as qmix_module
from marl_battlegrounds.baselines.methods import TrainingMethod
from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.baselines.qmix import QMIXConfig
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.training import _launch, checkpoints, qmix_learner
from marl_battlegrounds.training._content import PreparedTrainingContent
from marl_battlegrounds.training.curriculum import make_training_schedule
from marl_battlegrounds.training.learner import init_learner
from marl_battlegrounds.training.runner import TrainConfig, config_to_dict

type Tree = Any
_FIXTURE = Path(__file__).parent / "fixtures" / "method_contracts.json"
_METHODS = ("mappo", "ippo", "ff_mappo", "ff_ippo", "qmix")


@lru_cache
def _fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text())


@lru_cache
def _prepared() -> PreparedTrainingContent:
    from marl_battlegrounds.training import prepare_training_content

    return prepare_training_content()


def _digest(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _one(leaf: jax.ShapeDtypeStruct) -> jax.Array:
    return jnp.ones(leaf.shape, leaf.dtype)


def _ones(tree: Tree) -> Tree:
    return jax.tree.map(_one, tree)


def _learner(method: str) -> tuple[Any, Any]:
    settings = _fixture()["settings"]
    schedule = make_training_schedule(total_env_steps=64, num_envs=settings["num_envs"])
    if method == "qmix":
        return qmix_learner.init_qmix_learner(
            schedule=schedule, prepared=_prepared(), qmix=QMIXConfig(**settings["qmix"])
        )
    return init_learner(
        schedule=schedule,
        prepared=_prepared(),
        ppo=PPOConfig(**settings["ppo"]),
        method=method,
    )


def _registrations(method: str) -> dict[str, str]:
    params = _ones(checkpoints._actor_template(method))
    records: dict[str, str] = {}
    for scale, frame in ((1.0, "world"), (0.01, "left")):
        if method == "qmix":
            system = qmix_module.make_qmix_system(
                params, input_scale=scale, spawn_frame=frame
            )
        else:
            system = ppo_module.make_ppo_system(
                params, method=method, input_scale=scale, spawn_frame=frame
            )
        identifier, _ = normalize_system_registration(system, phase="evaluation")
        records[f"{scale}/{frame}"] = identifier
    return records


def _inference(method: str) -> str:
    synthetic: dict[str, object] = {
        "kind": "actor",
        "actor_digest": "0" * 64,
        "input_scale": 1.0,
        "spawn_frame": "left",
        "schemas": checkpoints.checkpoint_schemas(method),
    }
    if method == "qmix":
        synthetic.update(epsilon=0.0, tie_rule=qmix_module.QMIX_TIE_RULE)
    return checkpoints._inference_digest(synthetic)


def test_the_capture_settings_are_the_recorded_ones() -> None:
    fixture = _fixture()
    assert fixture["schema_version"] == 1
    assert set(fixture["methods"]) == set(_METHODS)
    assert fixture["launch_probe"] == _launch._PROBE
    settings = fixture["settings"]
    assert asdict(PPOConfig(**settings["ppo"])) == settings["ppo"]
    assert asdict(QMIXConfig(**settings["qmix"])) == settings["qmix"]


def _same(value: Tree) -> Tree:
    return value


@pytest.mark.parametrize("method", _METHODS)
def test_an_older_method_keeps_its_saved_contracts(method: TrainingMethod) -> None:
    expected = _fixture()["methods"][method]
    config = config_to_dict(TrainConfig(method=method))
    assert config == expected["default_config_to_dict"]
    assert checkpoints.checkpoint_schemas(method) == expected["checkpoint_schemas"]
    names = sorted(checkpoints.checkpoint_dependencies(method))
    assert names == expected["dependency_names"]
    actor = checkpoints._layout(checkpoints._actor_template(method))
    assert _digest(actor) == expected["actor_layout_sha256"]
    assert _inference(method) == expected["inference_digest"]
    assert _registrations(method) == expected["registrations"]
    collection, state = _learner(method)
    layout = checkpoints._layout(jax.eval_shape(_same, state))
    assert _digest(layout) == expected["state_layout_sha256"]
    details = checkpoints._collection_details(collection)
    assert _digest(details) == expected["collection_details_sha256"]
