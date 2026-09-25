"""Check the one reference loader and the pinned-opponent evidence classes.

load_method turns the four built-in names into their Policies, an exported
actor directory into its method's PPO System (or a greedy QMIX or PQN-VDN
System for such an export, which then plays a complete two-entrant
tournament), and a module:function factory
into the Policy or System it returns; it refuses a learner checkpoint
directory, an empty or unknown reference and a factory that returns anything
else, and a factory's own error keeps its type. The command line's factory
branch calls the same factory loader, load_factory, and wraps every error with
its cause kept.
pinned_opponent_evidence records the built-in Random team as installed with no
controller exposure, tdm-alpha, tdm-beta and tdm-gamma (through its recorded
Beta ancestry) as installed with known exposure that makes all eight protected
scenarios familiar, a researcher factory as
unknown exposure, and an export without its sibling learner description as a
declared export with unknown exposure.
Actors trained before Red Zone (actor input schema 1) still play exactly as
they did. The fixture ``baseline_donor/actor_input_schema_1.json`` and its
``.npz`` hold six methods' outputs captured from the pre-Red-Zone source: 14
cases (scale 1.0 in the world frame and 0.01 in the left frame, plus epsilon
0.25 for QMIX and PQN-VDN) on map 42 and on two banks centred exactly at
widths 17.3 and 20.3, three chained steps at B2. This file rebuilds the same
games on today's source (map 42 at Red Zone depths 0.0 and 5.0, the centred
banks at 5.0) and the same NumPy weights from the schema-1 template, and
checks: schema_1_actor_input reproduces the recorded 19-column inputs and
features bit for bit; the rebuilt weights with the default hooks give the
recorded registration IDs; Systems built with actor_input_schema=1 give the
recorded actions exactly and their learning outputs and final memory within
rtol 1e-5 and atol 1e-6; their registration IDs are new and pinned; on the
centred banks the schema-2 flag reads lane 0 Team A as left while the
schema-1 flag keeps the old reading, right; and the schema-1 encoding equals
the schema-2 encoding with flat feature 1128 (the depth) deleted at depths
0.0, 5.0 and 6.0.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import hashlib
import json
import sys
import types
from functools import cache
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import _cli, _method_loading
from marl_battlegrounds._method_loading import load_method
from marl_battlegrounds.baselines import ppo, pqn, qmix
from marl_battlegrounds.baselines.inputs import (
    encode_actor_inputs,
    schema_1_actor_input,
    spawn_frame_flag,
)
from marl_battlegrounds.baselines.ppo import initialize_ppo
from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.environment import make
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
    SystemInput,
    SystemOutput,
    policy,
)
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.tasks import (
    balanced_spawn_configs,
    make_standard_team_deathmatch_config,
)
from marl_battlegrounds.training import checkpoints
from marl_battlegrounds.training._content import (
    PreparedTrainingContent,
    pinned_opponent_evidence,
    prepare_training_content,
)
from marl_battlegrounds.training.checkpoints import export_system

type Tree = Any
_SCHEMA_1 = Path(__file__).parent / "fixtures" / "baseline_donor"
_ROSTER = ("mage", "warrior", "hunter", "rogue", "priest")
_DEPTH_FEATURE = 1128
# SHA256 of the 14 schema-1 Systems' registration IDs, in fixture case order,
# joined by newlines. They differ from the recorded ones because the hooks are
# new; this pins them so a change to a schema-1 hook is noticed.
_SCHEMA_1_REGISTRATIONS = (
    "0f333b55f9271dc9ebaa7fc6101055779ed36c9674ed8b625c7ee8b03e8c4f36"
)


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


@pytest.fixture(scope="module", params=("mappo", "ippo", "ff_mappo", "ff_ippo"))
def exported(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> Path:
    method = str(request.param)
    root = tmp_path_factory.mktemp("loading") / "actors"
    root.mkdir()
    return export_system(
        initialize_ppo(jax.random.key(3), method=method).actor_params,
        root / "actor",
        metadata={
            "run_id": "loading",
            "seed": 3,
            "env_steps": 0,
            "checkpoint_id": "b" * 64,
        },
        spawn_frame="left",
        method=method,
    )


def _unused_apply(
    variables: object, memory: object, inputs: SystemInput, keys: object
) -> tuple[object, object]:
    del inputs, keys
    return variables, memory


def _factory_module(monkeypatch: pytest.MonkeyPatch, value: object) -> str:
    module = types.ModuleType("pinned_loading_factory")
    module.make = lambda: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return f"{module.__name__}:make"


def test_built_in_names_exports_and_factories_resolve(
    exported: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("random", "tdm-alpha", "tdm-beta", "tdm-gamma"):
        value = load_method(name)
        assert isinstance(value, Policy) and value.name == name
    loaded = load_method(str(exported))
    assert isinstance(loaded, System) and loaded.execution == "jax"
    host = System("Host Team", _unused_apply, execution="host")
    assert load_method(_factory_module(monkeypatch, host)) is host


def test_bad_references_are_refused_with_clear_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    learner = tmp_path / "learner"
    learner.mkdir()
    (learner / "checkpoint_details.json").write_text("{}")
    with pytest.raises(ValueError, match="export the actor first"):
        load_method(str(learner))
    for text in ("", "   ", "unknown-team", "a:b:c"):
        with pytest.raises(ValueError):
            load_method(text)
    with pytest.raises(TypeError, match="System or Policy"):
        load_method(_factory_module(monkeypatch, object()))
    with pytest.raises(TypeError, match="string"):
        load_method(3)  # pyright: ignore[reportArgumentType]


def test_factory_errors_keep_their_type_and_the_cli_keeps_the_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("pinned_failing_factory")

    def fail() -> object:
        raise RuntimeError("provider unavailable")

    module.make = fail  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        load_method("pinned_failing_factory:make")
    with pytest.raises(_cli._FactoryError) as error:
        _cli._load_method("pinned_failing_factory:make")
    assert isinstance(error.value.__cause__, RuntimeError)
    assert _cli._load_method("tdm-alpha") == "tdm-alpha"
    seen: list[str] = []
    chosen = policy("random")

    def spy(reference: str) -> Policy:
        seen.append(reference)
        return chosen

    monkeypatch.setattr(_method_loading, "load_factory", spy)
    assert _cli._load_method("any_module:make") is chosen
    assert seen == ["any_module:make"]


def test_evidence_classes_for_built_ins_factories_and_unlinked_exports(
    prepared: PreparedTrainingContent, exported: Path
) -> None:
    binding = prepared.binding
    random = pinned_opponent_evidence(binding, policy("random"))
    assert random["source"] == "installed" and random["exposure"] == "none"
    assert random["familiar_scenarios"] == []
    for name in ("tdm-alpha", "tdm-beta", "tdm-gamma"):
        record = pinned_opponent_evidence(binding, policy(name))
        assert record["source"] == "installed" and record["exposure"] == "known"
        assert record["familiar_scenarios"] == list(range(1, 9))
        assert len(record["controllers"]) == 1  # pyright: ignore[reportArgumentType]
    researcher = pinned_opponent_evidence(binding, System("Researcher", _unused_apply))
    assert researcher["source"] == "researcher method"
    assert researcher["exposure"] == "unknown"
    declared = pinned_opponent_evidence(
        binding, load_method(str(exported)), export=exported
    )
    assert declared["source"] == "declared export"
    assert declared["exposure"] == "unknown"
    assert declared["familiar_scenarios"] == list(range(1, 9))


def test_a_pqn_export_loads_as_its_greedy_system(tmp_path: Path) -> None:
    from marl_battlegrounds.baselines import pqn

    network = pqn.initialize_pqn(jax.random.key(5), planned_learning_blocks=1).network
    exported = export_system(
        network,
        tmp_path / "actor",
        metadata={
            "run_id": "loading",
            "seed": 5,
            "env_steps": 0,
            "checkpoint_id": "c" * 64,
            "optimizer_steps": 0,
        },
        spawn_frame="left",
        method="pqn_vdn",
    )
    loaded = load_method(str(exported))
    assert isinstance(loaded, System) and loaded.execution == "jax"
    assert float(loaded.variables.epsilon) == 0.0
    tournament = marl_bgs.run_tournament(
        (loaded, "random"),
        maps=[47],
        episodes_per_pair=2,
        max_steps=4,
        num_envs=2,
        chunk_size=2,
        metrics="none",
    )
    assert tournament.status == "complete"
    assert len(tournament.table("tournament_rankings")["rank"]) == 2


def test_a_qmix_export_loads_as_its_greedy_system(tmp_path: Path) -> None:
    from marl_battlegrounds.baselines import qmix

    params = qmix.initialize_qmix(jax.random.key(4)).online_q
    exported = export_system(
        params,
        tmp_path / "actor",
        metadata={
            "run_id": "loading",
            "seed": 4,
            "env_steps": 0,
            "checkpoint_id": "b" * 64,
            "optimizer_steps": 0,
        },
        spawn_frame="left",
        method="qmix",
    )
    loaded = load_method(str(exported))
    assert isinstance(loaded, System) and loaded.execution == "jax"
    assert float(loaded.variables.epsilon) == 0.0
    tournament = marl_bgs.run_tournament(
        (loaded, "random"),
        maps=[47],
        episodes_per_pair=2,
        max_steps=4,
        num_envs=2,
        chunk_size=2,
        metrics="none",
    )
    assert tournament.status == "complete"
    assert len(tournament.table("tournament_rankings")["rank"]) == 2


@cache
def _schema_1_record() -> dict[str, Any]:
    return json.loads((_SCHEMA_1 / "actor_input_schema_1.json").read_text())


def _named_leaves(tree: Tree) -> list[tuple[str, Tree]]:
    leaves = cast(
        list[tuple[tuple[Any, ...], Tree]],
        jax.tree_util.tree_flatten_with_path(tree)[0],
    )
    return [(jax.tree_util.keystr(path), leaf) for path, leaf in leaves]


def _schema_1_weights(method: str, seed: int) -> Tree:
    # The capture script's NumPy rule, over the schema-1 template's leaves.
    template = checkpoints._actor_template(method, 1)
    rng = np.random.default_rng(seed)
    values: list[jax.Array] = []
    for name, leaf in _named_leaves(template):
        shape = cast(tuple[int, ...], leaf.shape)
        if name.endswith("['var']"):
            array = 0.5 + rng.random(shape)
        else:
            array = rng.standard_normal(shape) * 0.05
        values.append(jnp.asarray(array.astype(np.float32)))
    structure = cast(Any, jax.tree.structure(template))
    return jax.tree.unflatten(structure, values)


def _schema_1_system(case: dict[str, Any], weights: Tree, schema: int) -> System:
    method = str(case["method"])
    settings: dict[str, Any] = {
        "input_scale": float(case["input_scale"]),
        "spawn_frame": str(case["spawn_frame"]),
        "actor_input_schema": schema,
    }
    if method == "qmix":
        return qmix.make_qmix_system(weights, epsilon=case["epsilon"], **settings)
    if method == "pqn_vdn":
        network = pqn.PQNInferenceVariables(weights["params"], weights["batch_stats"])
        return pqn.make_pqn_system(network, epsilon=case["epsilon"], **settings)
    return ppo.make_ppo_system(weights, method=method, **settings)


def _schema_1_scene(scene: str, depth: float) -> EnvConfig:
    base = make_standard_team_deathmatch_config(
        map_id=42, team_a_roster=_ROSTER, team_b_roster=_ROSTER, red_zone_depth=depth
    )
    if scene == "map_42":
        return base
    width = {"centred_17_3": 17.3, "centred_20_3": 20.3}[scene]
    half = float(np.float32(width)) / 2
    centred = [float(np.float32(half + step)) for step in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    rows = (1.0, 3.0, 5.0, 7.0, 9.0)
    edge = float(np.float32(width - 1.0))
    pads = [
        [[x, y] for x, y in zip(centred, rows, strict=True)],
        [[edge, y] for y in rows],
    ]
    return base._replace(
        map_width=width,
        map_height=float(base.map_height),
        obstacles=jnp.zeros_like(base.obstacles),
        team_spawn_pad_positions=jnp.asarray(pads, jnp.float32),
    )


def _input_digest(inputs: SystemInput) -> str:
    digest = hashlib.sha256()
    for name, leaf in _named_leaves(inputs):
        array = np.asarray(leaf)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _schema_1_game(scene: str, depth: float) -> list[SystemInput]:
    facts = _schema_1_record()["scenes"][scene]
    config = _schema_1_scene(scene, depth)
    assert float(config.map_width) == facts["map_width"]
    assert float(config.map_height) == facts["map_height"]
    pads = np.asarray(config.team_spawn_pad_positions).tolist()
    assert pads == facts["team_spawn_pad_positions"]
    batch = balanced_spawn_configs(config, num_envs=2)
    env = make("tdm", env_config=batch, num_envs=2, metrics="none")
    observations, state = env.reset(jax.random.key(facts["reset_key_seed"]))
    steps: list[SystemInput] = []
    for step, recorded in enumerate(facts["steps"]):
        inputs = env.policy_inputs(observations, state)
        old = inputs._replace(actors=schema_1_actor_input(inputs.actors))
        assert _input_digest(old) == recorded["system_input_sha256"], (scene, step)
        features = np.asarray(encode_actor_inputs(old.actors))
        assert features.shape == (2, 5, 5164)
        features_digest = hashlib.sha256(features.tobytes()).hexdigest()
        assert features_digest == recorded["actor_features_sha256"], (scene, step)
        current = np.asarray(encode_actor_inputs(inputs.actors))
        np.testing.assert_array_equal(
            np.delete(current, _DEPTH_FEATURE, axis=-1), features
        )
        assert np.all(current[..., _DEPTH_FEATURE] == np.float32(depth))
        steps.append(inputs)
        joint = env.sample_actions(
            jax.random.key(facts["sample_action_key_seeds"][step]), state
        )
        observations, state, _, _, _ = env.step(
            jax.random.key(facts["step_key_seeds"][step]), state, joint
        )
    return steps


@pytest.fixture(scope="module")
def schema_1_games() -> dict[str, list[SystemInput]]:
    return {scene: _schema_1_game(scene, 5.0) for scene in _schema_1_record()["scenes"]}


def test_schema_1_inputs_rebuild_exactly_and_keep_their_old_spawn_side(
    schema_1_games: dict[str, list[SystemInput]],
) -> None:
    for depth in (0.0, 6.0):
        _schema_1_game("map_42", depth)
    for scene, steps in schema_1_games.items():
        actors = steps[0].actors
        old = np.asarray(spawn_frame_flag(schema_1_actor_input(actors), "left"))
        facts = _schema_1_record()["scenes"][scene]
        assert old.tolist() == facts["old_team_on_right_step_0"]
        new = np.asarray(spawn_frame_flag(actors, "left"))
        # Lane 1 exchanges banks, so its Team A starts at the right edge.
        assert new[1].all() and old[1].all()
        if scene == "map_42":
            assert not new[0].any() and not old[0].any()
        else:
            # An exactly centred bank: the exact rule reads left, the old right.
            assert not new[0].any() and old[0].all()


def test_schema_1_systems_reproduce_their_pre_red_zone_outputs(
    schema_1_games: dict[str, list[SystemInput]],
) -> None:
    record = _schema_1_record()
    base = int(record["system_key_seed_base"])
    with np.load(_SCHEMA_1 / "actor_input_schema_1.npz") as stored:
        expected = {name: stored[name] for name in stored.files}
    compared: set[str] = set()
    registrations: list[str] = []
    for case in record["cases"]:
        weights = _schema_1_weights(str(case["method"]), int(case["weight_seed"]))
        # The factories do not check width, so the default hooks reproduce the
        # recorded registration, which proves the weights were rebuilt exactly.
        current, _ = normalize_system_registration(
            _schema_1_system(case, weights, 2), phase="evaluation"
        )
        assert current == case["registration_identity"], case["arrays_prefix"]
        system = _schema_1_system(case, weights, 1)
        legacy, _ = normalize_system_registration(system, phase="evaluation")
        assert legacy != current
        registrations.append(legacy)
        # One compiled program per case, reused for every scene and step.
        apply = jax.jit(system.apply)
        init = None if system.init is None else jax.jit(system.init)
        for scene, steps in schema_1_games.items():
            memory: Tree = ()
            for step, inputs in enumerate(steps):
                keys = jax.random.split(jax.random.key(base + step), 2)
                if step == 0 and init is not None:
                    memory = cast(Tree, init(system.variables, inputs, keys))
                output = cast(
                    SystemOutput, apply(system.variables, memory, inputs, keys)
                )
                memory = output.next_memory
                prefix = f"{case['arrays_prefix']}/{scene}/step_{step}"
                for field in ActorAction._fields:
                    name = f"{prefix}/actions/{field}"
                    actual = np.asarray(getattr(output.actions, field))
                    np.testing.assert_array_equal(actual, expected[name], err_msg=name)
                    compared.add(name)
                floats = [
                    (f"{prefix}/learning{path}", leaf)
                    for path, leaf in _named_leaves(output.learning_outputs)
                ]
                if step == len(steps) - 1:
                    floats += [
                        (f"{prefix}/memory{path}", leaf)
                        for path, leaf in _named_leaves(memory)
                    ]
                for name, leaf in floats:
                    np.testing.assert_allclose(
                        np.asarray(leaf),
                        expected[name],
                        rtol=1e-5,
                        atol=1e-6,
                        err_msg=name,
                    )
                    compared.add(name)
    assert compared == set(expected)
    joined = "\n".join(registrations).encode()
    assert hashlib.sha256(joined).hexdigest() == _SCHEMA_1_REGISTRATIONS, registrations
