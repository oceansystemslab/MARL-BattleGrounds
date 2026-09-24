"""Build independent PQN-VDN calculations from the stored JaxMARL donor source.

This test-support module executes unchanged bodies from JaxMARL's
``baselines/QLearning/pqn_vdn_rnn.py`` at revision
``976aeb152cb184a5095021968bba94da96eb6394``, stored under
``tests/fixtures/baseline_donor/source/jaxmarl`` and hashed in
``pqn_source_manifest.json``: ``ScannedRNN``, ``QNetwork``, ``Transition``,
``CustomTrainState``, ``make_train``'s ``config["NUM_UPDATES"]`` and
``eps_scheduler`` assignments, ``get_greedy_actions``,
``eps_greedy_exploration``, ``train.create_agent`` and
``train._update_step._learn_epoch`` (with its nested ``_learn_phase``,
``_compute_targets``, ``_loss_fn`` and ``preprocess_transition``). It supplies
only the names those bodies read (checked from their syntax trees) and never
imports MARL-BGs' PQN implementation. The donor configuration comes from the
stored SMAX YAML plus recorded per-case overrides (PyYAML reads ``1e7`` as a
string, and ``TOTAL_TIMESTEPS`` is set so the donor's floor rule gives the
case's learning blocks).

Run as a script inside the isolated CPU environment
``/tmp/packet7-pqn-reference-env`` (the exact versions in
``pqn-reference-requirements.txt``), ``--generate DIRECTORY`` writes candidate
``pqn.npz`` and ``pqn_reference.json`` files: exact initialization digests and
the three orthogonal GRU recurrent kernels; inference and training forwards
with BatchNorm statistics; greedy choices and epsilon-greedy action counts
from the donor sampler; one captured gradient step; and two learning blocks of
two epochs and two minibatches with their permutations, losses and
parameter, statistic and optimizer fingerprints. The same bodies run inside
the test process as a second, same-stack oracle for parameter-sized
comparisons. Inputs come from recorded seeds (``jax.random`` is independent of
CPU thread count); orthogonal kernels are stored because their rounding is
not.
"""

# ruff: noqa: ANN401
# pyright: reportUnknownLambdaType=false
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import importlib.metadata
import json
import platform
import sys
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.baseline_donor_reference import (
    _source_node,  # pyright: ignore[reportPrivateUsage]
    _source_text,  # pyright: ignore[reportPrivateUsage]
    _store_tree,  # pyright: ignore[reportPrivateUsage]
)

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "baseline_donor"
_MANIFEST = "pqn_source_manifest.json"
_METADATA = "pqn_reference.json"
_DATA_FILE = "pqn.npz"
_REQUIREMENTS = "pqn-reference-requirements.txt"
_SOURCE = "baselines/QLearning/pqn_vdn_rnn.py"
_YAML = "baselines/QLearning/config/alg/pqn_vdn_rnn_smax.yaml"
DONOR_COMMIT = "976aeb152cb184a5095021968bba94da96eb6394"
FEATURES, AGENTS, ACTIONS, HIDDEN = 5164, 5, 198, 512
INIT_SEED = 17
FORWARD_SEED = 19049301
PROBABILITY_SEED = 19049302
GRADIENT_SEED = 19049303
UPDATE_SEED = 19049304
PROBABILITY_DRAWS = 20000
RATES = (0.0, 0.37, 1.0)
FORWARD_ROWS, FORWARD_ENVS = 4, 2
# Update cases: (num_envs, num_minibatches, epochs, memory_window, num_steps, blocks).
GRADIENT_CASE = {"envs": 2, "minibatches": 1, "epochs": 1, "window": 2, "steps": 2}
UPDATE_CASE = {
    "envs": 4,
    "minibatches": 2,
    "epochs": 2,
    "window": 2,
    "steps": 2,
    "blocks": 2,
}
_SUPPLIED = frozenset(
    {
        "Any",
        "TrainState",
        "chex",
        "config",
        "env",
        "jax",
        "jnp",
        "memory_transitions",
        "network",
        "nn",
        "np",
        "optax",
        "partial",
        "wrapped_env",
    }
)
_DEFINITIONS: tuple[tuple[str, ...], ...] = (
    ("ScannedRNN",),
    ("QNetwork",),
    ("Transition",),
    ("CustomTrainState",),
)
_ASSIGNMENTS = ("config[NUM_UPDATES]", "eps_scheduler")
_HELPERS: tuple[tuple[str, ...], ...] = (
    ("make_train", "get_greedy_actions"),
    ("make_train", "eps_greedy_exploration"),
)
_TRAIN_BODIES: tuple[tuple[str, ...], ...] = (
    ("make_train", "train", "create_agent"),
    ("make_train", "train", "_update_step", "_learn_epoch"),
)
_FINGERPRINT_SAMPLES = 64


def pqn_manifest() -> dict[str, Any]:
    manifest = json.loads((_FIXTURE_ROOT / _MANIFEST).read_text())
    if manifest["commit"] != DONOR_COMMIT:
        raise ValueError("The PQN manifest names a different donor revision.")
    for path in manifest["files"]:
        _source_text(path, manifest)
    return manifest


def _assignment_target(child: ast.stmt) -> str | None:
    if not isinstance(child, ast.Assign) or len(child.targets) != 1:
        return None
    target = child.targets[0]
    if isinstance(target, ast.Name):
        return target.id
    if (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and isinstance(target.slice, ast.Constant)
    ):
        return f"{target.value.id}[{target.slice.value}]"
    return None


def _assignments(text: str) -> list[ast.stmt]:
    owner = _source_node(text, ("make_train",))
    found = {
        name: child
        for child in getattr(owner, "body", [])
        if (name := _assignment_target(child)) in _ASSIGNMENTS
    }
    return [found[name] for name in _ASSIGNMENTS]


def _bound_names(node: ast.AST) -> set[str]:
    bound: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, (ast.FunctionDef, ast.ClassDef)):
            bound.add(item.name)
        elif isinstance(item, ast.arg):
            bound.add(item.arg)
        elif isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
            bound.add(item.id)
        elif isinstance(item, ast.alias):
            bound.add((item.asname or item.name).split(".")[0])
    return bound


def _free_names(nodes: list[ast.stmt]) -> set[str]:
    loaded: set[str] = set()
    bound: set[str] = set()
    for node in nodes:
        bound |= _bound_names(node)
        loaded |= {
            item.id
            for item in ast.walk(node)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
        }
    return loaded - bound - set(dir(builtins))


def _execute(namespace: dict[str, Any], nodes: list[ast.stmt]) -> None:
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    # Only enclosing packaging changes; each selected body stays intact.
    exec(compile(module, _SOURCE, "exec"), namespace)


def donor_config(overrides: dict[str, Any]) -> dict[str, Any]:
    import yaml

    config = dict(yaml.safe_load(_source_text(_YAML, pqn_manifest())))
    config.update(overrides)
    return config


def case_overrides(case: dict[str, int]) -> dict[str, Any]:
    blocks = case.get("blocks", 1)
    return {
        "NUM_ENVS": case["envs"],
        "NUM_STEPS": case["steps"],
        "MEMORY_WINDOW": case["window"],
        "NUM_EPOCHS": case["epochs"],
        "NUM_MINIBATCHES": case["minibatches"],
        "TOTAL_TIMESTEPS": blocks * case["steps"] * case["envs"],
    }


def build_pqn_namespace(overrides: dict[str, Any]) -> dict[str, Any]:
    import chex  # pyright: ignore[reportMissingTypeStubs]
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import optax  # pyright: ignore[reportMissingTypeStubs]
    from flax.training.train_state import TrainState

    manifest = pqn_manifest()
    text = _source_text(_SOURCE, manifest)
    definitions = [_source_node(text, names) for names in _DEFINITIONS]
    assignments = _assignments(text)
    helpers = [_source_node(text, names) for names in _HELPERS]
    bodies = [_source_node(text, names) for names in _TRAIN_BODIES]
    required = _free_names([*definitions, *assignments, *helpers, *bodies])
    if frozenset(required) != _SUPPLIED:
        raise ValueError(
            "Supplied donor names differ from the names the donor bodies read: "
            f"missing {sorted(required - _SUPPLIED)}, "
            f"extra {sorted(_SUPPLIED - required)}"
        )
    module = ModuleType(f"_pqn_donor_{DONOR_COMMIT[:12]}_{len(sys.modules)}")
    sys.modules[module.__name__] = module
    namespace: dict[str, Any] = module.__dict__
    namespace.update(
        {
            "Any": Any,
            "TrainState": TrainState,
            "chex": chex,
            "jax": jax,
            "jnp": jnp,
            "nn": nn,
            "np": np,
            "optax": optax,
            "partial": partial,
            "config": donor_config(overrides),
            "wrapped_env": SimpleNamespace(obs_size=FEATURES),
            "env": SimpleNamespace(num_agents=AGENTS),
            "memory_transitions": None,
        }
    )
    _execute(namespace, definitions)
    _execute(namespace, assignments)
    _execute(namespace, helpers)
    config = namespace["config"]
    # The donor's train() builds its network with these keyword arguments.
    namespace["network"] = namespace["QNetwork"](
        action_dim=ACTIONS,
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
        dueling=config.get("DUELING", False),
    )
    _execute(namespace, bodies)
    return namespace


def build_same_stack_pqn_reference(case: dict[str, int]) -> dict[str, Any]:
    return build_pqn_namespace(case_overrides(case))


def donor_learn(
    namespace: dict[str, Any], train_state: Any, window: Any, rng: Any, epochs: int
) -> tuple[Any, Any, Any]:
    import jax

    namespace["memory_transitions"] = window
    learn_epoch = namespace["_learn_epoch"]

    # The donor body reads the window as a global when traced. A fresh wrapper
    # per call stops JAX reusing an earlier trace that captured an old window.
    def epoch(carry: Any, unused: Any) -> Any:
        return learn_epoch(carry, unused)

    (train_state, rng), (loss, _) = jax.lax.scan(
        epoch, (train_state, rng), None, epochs
    )
    return train_state, rng, loss


def donor_permutations(rng: Any, epochs: int, num_envs: int) -> NDArray[np.int32]:
    import jax
    import jax.numpy as jnp

    rows: list[NDArray[np.int32]] = []
    for _ in range(epochs):
        rng, key = jax.random.split(rng)
        probe = jnp.broadcast_to(jnp.arange(num_envs), (1, 1, num_envs))
        permuted = jax.random.permutation(key, probe, axis=2)[0, 0]
        direct = jax.random.permutation(key, jnp.arange(num_envs))
        if not bool(jnp.array_equal(permuted, direct)):
            raise ValueError("Donor permutation indices changed meaning.")
        rows.append(np.asarray(direct, np.int32))
        rng, _ = jax.random.split(rng)
    return np.stack(rows)


def gradient_capture_state(namespace: dict[str, Any], train_state: Any) -> Any:
    import jax
    import jax.numpy as jnp
    import optax  # pyright: ignore[reportMissingTypeStubs]

    def init(params: Any) -> Any:
        return params

    def update(grads: Any, state: Any, params: Any = None) -> tuple[Any, Any]:
        del state, params
        zeros = jax.tree.map(lambda leaf: jnp.zeros_like(leaf), grads)  # pyright: ignore[reportUnknownArgumentType]
        return zeros, grads

    return namespace["CustomTrainState"].create(
        apply_fn=namespace["network"].apply,
        params=train_state.params,
        batch_stats=train_state.batch_stats,
        tx=optax.GradientTransformation(init, cast(Any, update)),
    )


def window_arrays(seed: int, rows: int, envs: int) -> dict[str, NDArray[Any]]:
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(jax.random.key(seed), 7)
    obs = jax.random.normal(keys[0], (rows, AGENTS, envs, FEATURES), jnp.float32)
    hidden = 0.1 * jax.random.normal(keys[1], (rows, AGENTS, envs, HIDDEN), jnp.float32)
    legal = jax.random.bernoulli(keys[2], 0.5, (rows, AGENTS, envs, ACTIONS))
    legal = legal.at[..., 0].set(True)
    scores = jax.random.uniform(keys[3], legal.shape)
    action = jnp.argmax(jnp.where(legal, scores, -1.0), axis=-1).astype(jnp.int32)
    reward = jax.random.normal(keys[4], (rows, 1, envs), jnp.float32)
    done = jnp.zeros((rows, 1, envs), jnp.float32).at[1, 0, 0].set(1.0)
    if rows > 3 and envs > 1:
        done = done.at[rows - 2, 0, envs - 1].set(1.0)
    last_done = jnp.concatenate(
        [jnp.zeros((1, 1, envs), jnp.float32).at[0, 0, envs - 1].set(1.0), done[:-1]]
    )
    return {
        "last_hs": np.asarray(hidden),
        "obs": np.asarray(obs),
        "action": np.asarray(action),
        "reward": np.asarray(reward),
        "done": np.asarray(done),
        "last_done": np.asarray(jnp.broadcast_to(last_done, (rows, AGENTS, envs))),
        "avail_actions": np.asarray(legal.astype(jnp.float32)),
        "q_vals": np.zeros((rows, AGENTS, envs, ACTIONS), np.float32),
    }


def to_transition(namespace: dict[str, Any], arrays: dict[str, NDArray[Any]]) -> Any:
    import jax.numpy as jnp

    return namespace["Transition"](
        **{key: jnp.asarray(value) for key, value in arrays.items()}
    )


def next_window(
    previous: dict[str, NDArray[Any]], new: dict[str, NDArray[Any]], steps: int
) -> dict[str, NDArray[Any]]:
    return {
        key: np.concatenate([previous[key][steps:], new[key]], axis=0)
        for key in previous
    }


def bg_layout(arrays: dict[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
    # Donor (rows, agents, envs, ...) to BG (rows, envs, agents, ...).
    return {
        "actor_features": np.swapaxes(arrays["obs"], 1, 2),
        "action_mask": np.swapaxes(arrays["avail_actions"], 1, 2).astype(bool),
        "actions": np.swapaxes(arrays["action"], 1, 2).astype(np.int32),
        "rewards": arrays["reward"][:, 0].astype(np.float32),
        "episode_start": arrays["last_done"][:, 0].astype(bool),
        "ended": arrays["done"][:, 0].astype(bool),
        "valid": np.ones(arrays["reward"][:, 0].shape, bool),
        "active": np.ones(np.swapaxes(arrays["action"], 1, 2).shape, bool),
        "initial_memory": np.swapaxes(arrays["last_hs"][0], 0, 1),
    }


def forward_inputs(seed: int) -> dict[str, NDArray[Any]]:
    import jax
    import jax.numpy as jnp

    # Arrays are built per (agent, game) and flattened agent-major, as the
    # donor flattens; episode starts are per game, shared by its five agents.
    keys = jax.random.split(jax.random.key(seed), 3)
    rows, envs = FORWARD_ROWS, FORWARD_ENVS
    obs = jax.random.normal(keys[0], (rows, AGENTS, envs, FEATURES), jnp.float32)
    hidden = 0.1 * jax.random.normal(keys[1], (AGENTS, envs, HIDDEN), jnp.float32)
    dones = np.zeros((rows, AGENTS, envs), np.float32)
    dones[0, :, 1] = 1.0
    dones[2, :, 0] = 1.0
    return {
        "obs": np.asarray(obs).reshape(rows, AGENTS * envs, FEATURES),
        "hidden": np.asarray(hidden).reshape(AGENTS * envs, HIDDEN),
        "dones": dones.reshape(rows, AGENTS * envs),
    }


def probability_inputs(seed: int) -> dict[str, NDArray[Any]]:
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(jax.random.key(seed), 2)
    q_values = np.array(jax.random.normal(keys[0], (4, ACTIONS), jnp.float32))
    legal = np.array(jax.random.bernoulli(keys[1], 0.3, (4, ACTIONS)))
    legal[:, 0] = True
    legal[1] = True
    legal[2] = False
    legal[2, 0] = True
    legal[3] = False
    legal[3, [0, 7, 150, 197]] = True
    q_values[3, 150] = q_values[3, 197] = 5.0
    return {"q_values": q_values, "legal": legal}


def recurrent_kernel_paths() -> tuple[tuple[str, ...], ...]:
    return tuple(
        ("ScannedRNN_0", "GRUCell_0", gate, "kernel") for gate in ("hr", "hz", "hn")
    )


def with_recurrent_kernels(params: Any, arrays: dict[str, NDArray[Any]]) -> Any:
    import jax.numpy as jnp

    def replace(tree: Any, path: tuple[str, ...], value: Any) -> Any:
        if not path:
            return value
        copy = dict(tree)
        copy[path[0]] = replace(tree[path[0]], path[1:], value)
        return copy

    for path in recurrent_kernel_paths():
        params = replace(params, path, jnp.asarray(arrays["init/" + "/".join(path)]))
    return params


def fingerprint(value: Any) -> dict[str, NDArray[Any]]:
    import jax

    prints: dict[str, NDArray[Any]] = {}
    flat = cast(
        list[tuple[tuple[Any, ...], Any]],
        jax.tree_util.tree_flatten_with_path(value)[0],
    )
    for path, leaf in flat:
        name = "/".join(
            str(getattr(key, "key", getattr(key, "name", getattr(key, "idx", key))))
            for key in path
        )
        array = np.asarray(leaf)
        if not np.issubdtype(array.dtype, np.floating):
            prints[f"{name}/value"] = array
            continue
        numbers = array.reshape(-1).astype(np.float64)
        picks = np.linspace(
            0, numbers.size - 1, min(_FINGERPRINT_SAMPLES, numbers.size)
        ).astype(np.int64)
        prints[f"{name}/sum"] = np.asarray(numbers.sum())
        prints[f"{name}/norm"] = np.asarray(np.sqrt(np.square(numbers).sum()))
        prints[f"{name}/sample"] = numbers[picks].astype(np.float32)
    return prints


def _leaf_digests(value: Any) -> dict[str, str]:
    import jax

    flat = cast(
        list[tuple[tuple[Any, ...], Any]],
        jax.tree_util.tree_flatten_with_path(value)[0],
    )
    return {
        jax.tree_util.keystr(path): hashlib.sha256(
            np.asarray(leaf).tobytes()
        ).hexdigest()
        for path, leaf in flat
    }


def _check_environment() -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in (_FIXTURE_ROOT / _REQUIREMENTS).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"Reference requires {name}=={expected}; found {actual}.")
        packages[name] = actual
    if sys.version_info[:2] != (3, 14):
        raise ValueError("Generate the PQN donor reference with Python 3.14.")
    return packages


def _generate_forward(
    ns: dict[str, Any], variables: Any, arrays: dict[str, Any]
) -> None:
    import jax.numpy as jnp

    inputs = forward_inputs(FORWARD_SEED)
    network = ns["network"]
    hidden, values = network.apply(
        variables,
        jnp.asarray(inputs["hidden"]),
        jnp.asarray(inputs["obs"]),
        jnp.asarray(inputs["dones"]),
        False,
    )
    arrays["forward/inference/q"] = np.asarray(values)
    arrays["forward/inference/hidden"] = np.asarray(hidden)
    (hidden, values), updates = network.apply(
        variables,
        jnp.asarray(inputs["hidden"]),
        jnp.asarray(inputs["obs"]),
        jnp.asarray(inputs["dones"]),
        True,
        mutable=["batch_stats"],
    )
    arrays["forward/train/q"] = np.asarray(values)
    arrays["forward/train/hidden"] = np.asarray(hidden)
    import jax

    _store_tree(arrays, "forward/train/batch_stats", updates["batch_stats"], jax)


def _generate_probabilities(ns: dict[str, Any], arrays: dict[str, Any]) -> None:
    import jax
    import jax.numpy as jnp

    inputs = probability_inputs(PROBABILITY_SEED)
    q_values, legal = (
        jnp.asarray(inputs["q_values"]),
        jnp.asarray(inputs["legal"], jnp.float32),
    )
    arrays["probability/greedy"] = np.asarray(ns["get_greedy_actions"](q_values, legal))
    keys = jax.random.split(jax.random.key(PROBABILITY_SEED + 1), PROBABILITY_DRAWS)
    for rate in RATES:
        explore = ns["eps_greedy_exploration"]

        def sample(key: Any, r: float = rate, explore: Any = explore) -> Any:
            return explore(key, q_values, r, legal)

        draws = jax.vmap(sample)(keys)
        counts = np.stack(
            [
                np.bincount(np.asarray(draws[:, row]), minlength=ACTIONS)
                for row in range(4)
            ]
        )
        arrays[f"probability/counts/{rate}"] = counts.astype(np.int64)


def _generate_gradient(arrays: dict[str, Any]) -> dict[str, Any]:
    import jax

    case = GRADIENT_CASE
    ns = build_pqn_namespace(case_overrides(case))
    state = ns["create_agent"](jax.random.key(INIT_SEED))
    state = state.replace(params=with_recurrent_kernels(state.params, arrays))
    rows = case["window"] + case["steps"]
    window = window_arrays(GRADIENT_SEED, rows, case["envs"])
    capture = gradient_capture_state(ns, state)
    rng = jax.random.key(GRADIENT_SEED + 1)
    arrays["gradient/permutations"] = donor_permutations(rng, 1, case["envs"])
    after, _, loss = donor_learn(ns, capture, to_transition(ns, window), rng, 1)
    arrays["gradient/loss"] = np.asarray(loss)
    for key, value in fingerprint(after.opt_state).items():
        arrays[f"gradient/grads/{key}"] = value
    for key, value in fingerprint(after.batch_stats).items():
        arrays[f"gradient/batch_stats/{key}"] = value
    return case_overrides(case)


def _generate_update(arrays: dict[str, Any]) -> dict[str, Any]:
    import jax

    case = UPDATE_CASE
    ns = build_pqn_namespace(case_overrides(case))
    arrays["update/num_updates"] = np.asarray(ns["config"]["NUM_UPDATES"])
    state = ns["create_agent"](jax.random.key(INIT_SEED))
    state = state.replace(params=with_recurrent_kernels(state.params, arrays))
    rows = case["window"] + case["steps"]
    window = window_arrays(UPDATE_SEED, rows, case["envs"])
    rng = jax.random.key(UPDATE_SEED + 100)
    for block in range(case["blocks"]):
        arrays[f"update/block{block}/permutations"] = donor_permutations(
            rng, case["epochs"], case["envs"]
        )
        state, rng, loss = donor_learn(
            ns, state, to_transition(ns, window), rng, case["epochs"]
        )
        arrays[f"update/block{block}/loss"] = np.asarray(loss)
        for group, value in (
            ("params", state.params),
            ("batch_stats", state.batch_stats),
            ("opt_state", state.opt_state),
        ):
            for key, item in fingerprint(value).items():
                arrays[f"update/block{block}/{group}/{key}"] = item
        window = next_window(
            window,
            window_arrays(UPDATE_SEED + block + 1, case["steps"], case["envs"]),
            case["steps"],
        )
    return case_overrides(case)


def generate_pqn_reference(output_dir: Path) -> None:
    import jax

    packages = _check_environment()
    if jax.default_backend() != "cpu":
        raise ValueError("Generate the PQN donor reference on CPU.")
    pqn_manifest()
    arrays: dict[str, NDArray[Any]] = {}
    ns = build_pqn_namespace(case_overrides(GRADIENT_CASE))
    state = ns["create_agent"](jax.random.key(INIT_SEED))
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    for path in recurrent_kernel_paths():
        leaf: Any = state.params
        for part in path:
            leaf = leaf[part]
        arrays["init/" + "/".join(path)] = np.asarray(leaf)
    init_digests = _leaf_digests(variables)
    optimizer_structure = str(jax.tree.structure(state.opt_state))  # pyright: ignore[reportUnknownArgumentType]
    _generate_forward(ns, variables, arrays)
    _generate_probabilities(ns, arrays)
    gradient_overrides = _generate_gradient(arrays)
    update_overrides = _generate_update(arrays)
    if "marl_battlegrounds" in sys.modules:
        raise RuntimeError("The donor reference imported MARL-BGs code.")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / _DATA_FILE
    with data_path.open("wb") as handle:
        np.savez_compressed(handle, **dict(sorted(arrays.items())))  # pyright: ignore[reportArgumentType]
    metadata = {
        "schema_version": 1,
        "donor_commit": DONOR_COMMIT,
        "source_manifest_sha256": hashlib.sha256(
            (_FIXTURE_ROOT / _MANIFEST).read_bytes()
        ).hexdigest(),
        "requirements_sha256": hashlib.sha256(
            (_FIXTURE_ROOT / _REQUIREMENTS).read_bytes()
        ).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "archive_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "array_count": len(arrays),
        "python": platform.python_version(),
        "backend": jax.default_backend(),
        "packages": packages,
        "seeds": {
            "init": INIT_SEED,
            "forward": FORWARD_SEED,
            "probability": PROBABILITY_SEED,
            "gradient": GRADIENT_SEED,
            "update": UPDATE_SEED,
        },
        "probability_draws": PROBABILITY_DRAWS,
        "rates": list(RATES),
        "init_leaf_sha256": init_digests,
        "optimizer_structure": optimizer_structure,
        "donor_config": donor_config({}),
        "overrides": {"gradient": gradient_overrides, "update": update_overrides},
        "limits": [
            "Synthetic all-active, all-valid CPU calculations, not a training run.",
            "The donor network, optimizer, loss and epoch/minibatch path run "
            "unchanged; environments, Hydra and WandB are not imported.",
            "Sampled actions are compared as counts against probabilities, not "
            "index by index.",
            "Parameter-sized results are stored as fingerprints; the same bodies "
            "also run in the test process for leaf-by-leaf comparison.",
            "No BG masking, padding, frame, reward or schedule adaptations are "
            "included.",
        ],
    }
    (output_dir / _METADATA).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def load_pqn_reference() -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    metadata = json.loads((_FIXTURE_ROOT / _METADATA).read_text())
    manifest_hash = hashlib.sha256((_FIXTURE_ROOT / _MANIFEST).read_bytes()).hexdigest()
    if manifest_hash != metadata["source_manifest_sha256"]:
        raise ValueError("The PQN donor manifest does not match the reference.")
    data_path = _FIXTURE_ROOT / _DATA_FILE
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != metadata["archive_sha256"]:
        raise ValueError("The PQN donor archive does not match its recorded hash.")
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return metadata, arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the PQN-VDN donor reference."
    )
    parser.add_argument("--generate", required=True, type=Path)
    generate_pqn_reference(parser.parse_args().generate)


if __name__ == "__main__":
    main()
