"""Check that Red Zone changed older methods' saved contracts only as planned.

The fixture ``tests/fixtures/method_contracts.json`` was captured on CPU before
PQN-VDN existed, for MAPPO, IPPO, feedforward MAPPO, feedforward IPPO and QMIX,
and stays byte-identical. It holds SHA256 digests, so
``tests/fixtures/method_contracts_raw_v1.json`` holds the raw pre-Red-Zone
values behind three of them (actor layout, learner-state layout at 4 games
and blocks of 4 rounds, and collection details). This file first proves the
raw file genuine: each raw value's digest equals the committed digest. It then
requires each current value to equal the old one after exactly these
differences and nothing else: in the actor layout, only the input kernel's
rows go from 5,164 to 5,165; in the learner state, every leaf with a 5,164
axis goes to 5,165 (MAPPO 4 leaves, IPPO 7, FF-MAPPO 4, FF-IPPO 7, QMIX 5),
every leaf with a 919 axis goes to 920 (MAPPO 3, FF-MAPPO 3, QMIX 25, IPPO
and FF-IPPO none), every context_features leaf's last axis goes from 19 to 20
(2 each for the PPO methods, 3 for QMIX), and exactly three float32
team_deathmatch_red_zone_depth leaves appear, each right after its
team_deathmatch_score_threshold leaf (the env default config (4,), the state
config (4,) and the tracked source configs (42,)), so leaf counts go 336 to
339 (MAPPO, IPPO), 263 to 266 (FF) and 415 to 418 (QMIX); in the collection
details only content_binding may differ; in the checkpoint schema dictionary
only actor_input and training_state go from 1 to 2, and the frozen
pre-Red-Zone dictionary equals the recorded one. The schema-1 actor template
equals the old actor layout exactly. The default ``config_to_dict`` equals the
recorded one plus exactly "red_zone_depth": 5.0, placed right after
score_threshold_curriculum. Unchanged exactly: the dependency names
(Flashbax for QMIX only), the inference identity of one fixed synthetic actor
description in the left frame, the System registration IDs of all-ones
schema-1-shaped actors with the default hooks at scale 1.0 in the world frame
and scale 0.01 in the left frame (the factories do not check widths), and the
launch probe's source text. Only values that do not depend on the run, the
machine or the CPU thread count are compared.
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
type Row = dict[str, Any]
_FIXTURE = Path(__file__).parent / "fixtures" / "method_contracts.json"
_RAW = Path(__file__).parent / "fixtures" / "method_contracts_raw_v1.json"
_METHODS = ("mappo", "ippo", "ff_mappo", "ff_ippo", "qmix")
# The one actor input kernel whose rows grow with the actor input width.
_INPUT_KERNEL = {
    "mappo": "params/pre_torso/Dense_0/kernel",
    "ippo": "params/pre_torso/Dense_0/kernel",
    "ff_mappo": "params/torso/Dense_0/kernel",
    "ff_ippo": "params/torso/Dense_0/kernel",
    "qmix": "params/pre_torso/Dense_0/kernel",
}
# Allowed learner-state changes per method: leaves widened from 5,164 actor
# features, from 919 training-state features and from 19 context columns, and
# the three new depth leaves.
_STATE_CHANGES = {
    "mappo": {"actor": 4, "training_state": 3, "context": 2, "depth": 3},
    "ippo": {"actor": 7, "training_state": 0, "context": 2, "depth": 3},
    "ff_mappo": {"actor": 4, "training_state": 3, "context": 2, "depth": 3},
    "ff_ippo": {"actor": 7, "training_state": 0, "context": 2, "depth": 3},
    "qmix": {"actor": 5, "training_state": 25, "context": 3, "depth": 3},
}
_LEAF_COUNTS = {"mappo": 339, "ippo": 339, "ff_mappo": 266, "ff_ippo": 266, "qmix": 418}
_DEPTH_LEAVES = (
    ("carry/env/_default_config/team_deathmatch_red_zone_depth", [4]),
    ("carry/state/config/team_deathmatch_red_zone_depth", [4]),
    ("carry/tracking/source_configs/team_deathmatch_red_zone_depth", [42]),
)


@lru_cache
def _fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text())


@lru_cache
def _raw() -> dict[str, Any]:
    return json.loads(_RAW.read_text())


def _path(row: Row) -> str:
    return "/".join(str(next(iter(key.values()))) for key in row["path"])


def _widened(row: Row, old: int, new: int) -> Row:
    return {**row, "shape": [new if size == old else size for size in row["shape"]]}


def _expected_state(rows: list[Row]) -> tuple[list[Row], dict[str, int], list[Row]]:
    # Apply exactly the allowed changes to the old layout, counting each kind.
    counts = {"actor": 0, "training_state": 0, "context": 0, "depth": 0}
    expected: list[Row] = []
    added: list[Row] = []
    for row in rows:
        shape = row["shape"]
        if 5164 in shape:
            row = _widened(row, 5164, 5165)
            counts["actor"] += 1
        elif 919 in shape:
            row = _widened(row, 919, 920)
            counts["training_state"] += 1
        elif shape[-1:] == [19] and _path(row).endswith("/context_features"):
            row = {**row, "shape": [*shape[:-1], 20]}
            counts["context"] += 1
        expected.append(row)
        if _path(row).endswith("/team_deathmatch_score_threshold"):
            depth = {
                "path": [
                    *row["path"][:-1],
                    {"field": "team_deathmatch_red_zone_depth"},
                ],
                "shape": row["shape"],
                "dtype": "float32",
            }
            expected.append(depth)
            added.append(depth)
            counts["depth"] += 1
    return expected, counts, added


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
    # Schema-1-shaped weights with the default hooks: the factories do not
    # check widths, so the recorded registration IDs must stay exact.
    params = _ones(checkpoints._actor_template(method, 1))
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


def test_the_raw_pre_red_zone_values_hash_to_the_committed_digests() -> None:
    raw = _raw()
    assert raw["schema_version"] == 1
    assert raw["source_commit"] == "59c157c98962"
    assert set(raw["methods"]) == set(_METHODS)
    for method in _METHODS:
        expected = _fixture()["methods"][method]
        values = raw["methods"][method]
        assert _digest(values["actor_layout"]) == expected["actor_layout_sha256"]
        assert _digest(values["state_layout"]) == expected["state_layout_sha256"]
        details = _digest(values["collection_details"])
        assert details == expected["collection_details_sha256"]


@pytest.mark.parametrize("method", _METHODS)
def test_an_older_method_keeps_its_saved_contracts(method: TrainingMethod) -> None:
    expected = _fixture()["methods"][method]
    old = _raw()["methods"][method]
    config = config_to_dict(TrainConfig(method=method))
    # The one allowed config change: every new config states its depth.
    assert config == {**expected["default_config_to_dict"], "red_zone_depth": 5.0}
    assert list(config).index("red_zone_depth") == (
        list(config).index("score_threshold_curriculum") + 1
    )
    schemas = expected["checkpoint_schemas"]
    assert checkpoints._ACTOR_INPUT_1_SCHEMAS[method] == schemas
    current = {**schemas, "actor_input": 2, "training_state": 2}
    assert checkpoints.checkpoint_schemas(method) == current
    names = sorted(checkpoints.checkpoint_dependencies(method))
    assert names == expected["dependency_names"]
    assert (
        checkpoints._layout(checkpoints._actor_template(method, 1))
        == (old["actor_layout"])
    )
    actor = checkpoints._layout(checkpoints._actor_template(method))
    assert actor == [
        _widened(row, 5164, 5165) if _path(row) == _INPUT_KERNEL[method] else row
        for row in old["actor_layout"]
    ]
    assert sum(_path(row) == _INPUT_KERNEL[method] for row in actor) == 1
    assert _inference(method) == expected["inference_digest"]
    assert _registrations(method) == expected["registrations"]
    collection, state = _learner(method)
    layout = checkpoints._layout(jax.eval_shape(_same, state))
    widened, counts, added = _expected_state(old["state_layout"])
    assert counts == _STATE_CHANGES[method]
    assert [(_path(row), row["shape"]) for row in added] == list(_DEPTH_LEAVES)
    assert len(old["state_layout"]) + 3 == len(layout) == _LEAF_COUNTS[method]
    assert layout == widened
    details = checkpoints._collection_details(collection)
    assert details.keys() == old["collection_details"].keys()
    for key, value in details.items():
        if key != "content_binding":
            assert value == old["collection_details"][key], key
