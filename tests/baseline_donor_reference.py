"""Load and rebuild the independent recurrent MAPPO numerical reference.

Ordinary tests call ``load_reference`` and ``reference_tree``. They need NumPy,
but do not import Mava, TensorFlow Probability or historical training packages.
The checked fixture was made by executing unchanged source bodies from the
pinned donor with its locked CPU dependencies. To reproduce it, run this file
with ``--generate OUTPUT_DIRECTORY`` in the separate reference environment
described in ``docs/training/source_reuse.md``. Generation never imports BG
production modules and never changes the checked fixture without that explicit
output path. Source and environment hashes are checked before execution.

``build_same_stack_reference`` returns the unchanged donor classes and functions
on the current stack. Its small categorical wrapper supplies only log probability
and entropy; the independent historical fixture checks that bridge. The returned
namespace owns ``config``, ``actor_apply_fn``, ``critic_apply_fn`` and the two
optimizer update functions required by the extracted losses/epoch. Callers set
those names before evaluating their fixed tensors. This route imports current
Flax and Optax only when requested and never imports production baseline code.
"""

from __future__ import annotations

# Extracted historical objects have no importable types in the BG environment.
# ruff: noqa: ANN401
# pyright: reportUnknownLambdaType=false
import argparse
import ast
import functools
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "baseline_donor"
_DATA_FILE = "recurrent_mappo.npz"


def load_reference() -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    metadata = json.loads((_FIXTURE_ROOT / "reference.json").read_text())
    source_hash = hashlib.sha256(
        (_FIXTURE_ROOT / "source_manifest.json").read_bytes()
    ).hexdigest()
    if source_hash != metadata["source_manifest_sha256"]:
        raise ValueError("The donor source manifest does not match the reference.")
    data_path = _FIXTURE_ROOT / _DATA_FILE
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != metadata["archive_sha256"]:
        raise ValueError(
            "The donor reference archive does not match its recorded hash."
        )
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return metadata, arrays


def reference_tree(arrays: dict[str, NDArray[Any]], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    marker = prefix.rstrip("/") + "/"
    for key, value in arrays.items():
        if not key.startswith(marker):
            continue
        parts = key[len(marker) :].split("/")
        owner = result
        for part in parts[:-1]:
            owner = owner.setdefault(part, {})
        owner[parts[-1]] = value
    if not result:
        raise KeyError(prefix)
    return result


def _source_text(relative_path: str, manifest: dict[str, Any]) -> str:
    entry = manifest["files"][relative_path]
    raw = (_FIXTURE_ROOT / entry["stored_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"Donor source hash differs: {relative_path}")
    return raw.decode()


def _source_node(text: str, names: tuple[str, ...]) -> ast.stmt:
    node: ast.AST = ast.parse(text)
    for name in names:
        candidates = getattr(node, "body", [])
        node = next(
            child
            for child in candidates
            if isinstance(child, (ast.ClassDef, ast.FunctionDef)) and child.name == name
        )
    assert isinstance(node, ast.stmt)
    return node


def _exec_source(
    namespace: dict[str, Any],
    manifest: dict[str, Any],
    path: str,
    symbols: tuple[tuple[str, ...], ...],
) -> None:
    text = _source_text(path, manifest)
    nodes = [_source_node(text, names) for names in symbols]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    # Only enclosing packaging/imports change; each selected body stays intact.
    exec(compile(module, path, "exec"), namespace)


def _check_environment() -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in (_FIXTURE_ROOT / "reference-requirements.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"Reference requires {name}=={expected}; found {actual}.")
        packages[name] = actual
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Generate the donor reference with Python 3.12.")
    return packages


def _categorical_shim(jax: Any, jnp: Any) -> tuple[SimpleNamespace, SimpleNamespace]:
    class Categorical:
        def __init__(self, logits: Any) -> None:
            self.logits = logits

        def log_prob(self, actions: Any) -> Any:
            log_probabilities = jax.nn.log_softmax(self.logits, axis=-1)
            return jnp.take_along_axis(log_probabilities, actions[..., None], axis=-1)[
                ..., 0
            ]

        def entropy(self) -> Any:
            log_probabilities = jax.nn.log_softmax(self.logits, axis=-1)
            probabilities = jax.nn.softmax(self.logits, axis=-1)
            return -jnp.sum(probabilities * log_probabilities, axis=-1)

    class TransformedDistribution:
        def __init__(self, distribution: Any, bijector: object) -> None:
            del bijector
            self.distribution = distribution

        def log_prob(self, actions: Any) -> Any:
            return self.distribution.log_prob(actions)

    return (
        SimpleNamespace(
            Categorical=Categorical, TransformedDistribution=TransformedDistribution
        ),
        SimpleNamespace(Identity=object),
    )


def build_same_stack_reference() -> dict[str, Any]:
    manifest = json.loads((_FIXTURE_ROOT / "source_manifest.json").read_text())
    return _reference_namespace(manifest, same_stack=True)


def _reference_namespace(
    manifest: dict[str, Any], *, same_stack: bool = False
) -> dict[str, Any]:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    nn = importlib.import_module("flax.linen")
    optax = importlib.import_module("optax")
    if same_stack:
        tfd, tfb = _categorical_shim(jax, jnp)
    else:
        tfd = importlib.import_module(
            "tensorflow_probability.substrates.jax.distributions"
        )
        tfb = importlib.import_module("tensorflow_probability.substrates.jax.bijectors")
    module = ModuleType("mava_pinned_reference" + ("_same_stack" if same_stack else ""))
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update(
        {
            "functools": functools,
            "jax": jax,
            "jnp": jnp,
            "np": np,
            "nn": nn,
            "orthogonal": nn.initializers.orthogonal,
            "tree": jax.tree,
            "optax": optax,
            "NamedTuple": NamedTuple,
            "tfd": tfd,
            "tfb": tfb,
            # The fixture uses ordinary arrays, never donor graph observations.
            "is_graph_observation": lambda _observation: False,
        }
    )
    sources = (
        ("mava/types.py", (("Observation",), ("ObservationGlobalState",))),
        (
            "mava/systems/ppo/types.py",
            (("Params",), ("OptStates",), ("HiddenStates",), ("RNNPPOTransition",)),
        ),
        ("mava/networks/torsos.py", (("_parse_activation_fn",), ("MLPTorso",))),
        ("mava/networks/distributions.py", (("IdentityTransformation",),)),
        ("mava/networks/heads.py", (("DiscreteActionHead",),)),
        (
            "mava/networks/base.py",
            (("ScannedRNN",), ("RecurrentActor",), ("RecurrentValueNet",)),
        ),
        ("mava/utils/multistep.py", (("calculate_gae",),)),
        (
            "mava/systems/ppo/anakin/rec_mappo.py",
            (
                ("get_learner_fn", "_update_step", "_update_epoch"),
                (
                    "get_learner_fn",
                    "_update_step",
                    "_update_epoch",
                    "_update_minibatch",
                    "_actor_loss_fn",
                ),
                (
                    "get_learner_fn",
                    "_update_step",
                    "_update_epoch",
                    "_update_minibatch",
                    "_critic_loss_fn",
                ),
            ),
        ),
    )
    for path, symbols in sources:
        _exec_source(namespace, manifest, path, symbols)
    return namespace


def _store_tree(
    arrays: dict[str, NDArray[Any]], prefix: str, value: Any, jax: Any
) -> None:
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        parts: list[str] = []
        for key in path:
            if hasattr(key, "key"):
                parts.append(str(key.key))
            elif hasattr(key, "name"):
                parts.append(str(key.name))
            else:
                parts.append(str(key.idx))
        arrays["/".join((prefix, *parts))] = np.asarray(leaf)


def generate_reference(output_dir: Path) -> None:
    packages = _check_environment()
    manifest = json.loads((_FIXTURE_ROOT / "source_manifest.json").read_text())
    for path in manifest["files"]:
        _source_text(path, manifest)
    ns = _reference_namespace(manifest)
    jax, jnp, optax = (ns[name] for name in ("jax", "jnp", "optax"))
    if jax.default_backend() != "cpu":
        raise ValueError("Reference generation requires JAX_PLATFORMS=cpu.")
    omega = importlib.import_module("omegaconf").OmegaConf
    system = omega.create(
        _source_text("mava/configs/system/ppo/rec_mappo.yaml", manifest)
    )
    architecture = omega.create(_source_text("mava/configs/arch/anakin.yaml", manifest))
    # Small CPU tensors preserve both group and minibatch axes and all five actors.
    groups, time, environments, actors = 2, 4, 4, 5
    system.rollout_length = time
    system.recurrent_chunk_size = time
    architecture.num_envs = environments
    config = omega.create({"system": system, "arch": architecture})
    ns["config"] = config
    network = omega.create(_source_text("mava/configs/network/rnn.yaml", manifest))

    def torso(settings: Any) -> Any:
        return ns["MLPTorso"](
            tuple(settings.layer_sizes),
            activation=settings.activation,
            use_layer_norm=settings.use_layer_norm,
        )

    actor = ns["RecurrentActor"](
        torso(network.actor_network.pre_torso),
        torso(network.actor_network.post_torso),
        ns["DiscreteActionHead"](198),
        hidden_state_dim=network.hidden_state_dim,
    )
    critic = ns["RecurrentValueNet"](
        torso(network.critic_network.pre_torso),
        torso(network.critic_network.post_torso),
        centralised_critic=True,
        hidden_state_dim=network.hidden_state_dim,
    )
    ns["actor_apply_fn"], ns["critic_apply_fn"] = actor.apply, critic.apply

    # Execute the exact donor optimizer construction, with its original config.
    learner_source = _source_text("mava/systems/ppo/anakin/rec_mappo.py", manifest)
    setup_node = _source_node(learner_source, ("learner_setup",))
    optimizer_nodes: list[ast.stmt] = [
        child
        for child in getattr(setup_node, "body", [])
        if isinstance(child, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in ("actor_optim", "critic_optim")
            for target in child.targets
        )
    ]
    ns["actor_lr"], ns["critic_lr"] = system.actor_lr, system.critic_lr
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=optimizer_nodes, type_ignores=[])
            ),
            "mava/systems/ppo/anakin/rec_mappo.py",
            "exec",
        ),
        ns,
    )
    actor_optim, critic_optim = ns["actor_optim"], ns["critic_optim"]
    ns["actor_update_fn"], ns["critic_update_fn"] = (
        actor_optim.update,
        critic_optim.update,
    )

    rng = np.random.default_rng(190919)
    shape = (groups, time, environments, actors)
    actor_features = jnp.asarray(rng.normal(size=(*shape, 13)).astype(np.float32))
    critic_features = jnp.asarray(rng.normal(size=(*shape, 17)).astype(np.float32))
    masks_np = rng.random((*shape, 198)) > 0.22
    masks_np[..., 0] = True
    masks = jnp.asarray(masks_np)
    starts_np = np.zeros(shape, dtype=bool)
    starts_np[:, 2, 0, :] = True
    starts = jnp.asarray(starts_np)
    actor_carry = jnp.asarray(
        rng.normal(0, 0.2, (groups, environments, actors, 128)).astype(np.float32)
    )
    critic_carry = jnp.asarray(
        rng.normal(0, 0.2, (groups, environments, actors, 128)).astype(np.float32)
    )
    observation = ns["ObservationGlobalState"](actor_features, masks, critic_features)
    example_observation = jax.tree.map(lambda value: value[0], observation)
    actor_params = actor.init(
        jax.random.PRNGKey(19), actor_carry[0], (example_observation, starts[0])
    )
    critic_params = critic.init(
        jax.random.PRNGKey(23), critic_carry[0], (example_observation, starts[0])
    )
    params = ns["Params"](actor_params, critic_params)
    opt_states = ns["OptStates"](
        actor_optim.init(actor_params), critic_optim.init(critic_params)
    )

    def forward(carry_a: Any, carry_c: Any, obs: Any, reset: Any) -> tuple[Any, ...]:
        next_a, distribution = actor.apply(actor_params, carry_a, (obs, reset))
        next_c, value = critic.apply(critic_params, carry_c, (obs, reset))
        return (
            next_a,
            next_c,
            distribution.distribution.logits,
            distribution.entropy(),
            value,
        )

    next_a, next_c, logits, entropy, values = jax.vmap(forward)(
        actor_carry, critic_carry, observation, starts
    )
    unmasked_observation = observation._replace(action_mask=jnp.ones_like(masks))
    raw_logits = jax.vmap(forward)(
        actor_carry, critic_carry, unmasked_observation, starts
    )[2]
    actions_np = np.empty(shape, dtype=np.int32)
    for index in np.ndindex(shape):
        actions_np[index] = rng.choice(np.flatnonzero(masks_np[index]))
    actions = jnp.asarray(actions_np)
    distributions = ns["tfd"].Categorical(logits=logits)
    log_probabilities = distributions.log_prob(actions)
    # Nonzero old-policy offsets exercise both PPO clipping branches.
    old_logs = log_probabilities + jnp.asarray(
        rng.normal(0, 0.3, shape).astype(np.float32)
    )
    old_values = values + jnp.asarray(rng.normal(0, 0.35, shape).astype(np.float32))
    rewards = jnp.asarray(rng.normal(0, 0.3, shape).astype(np.float32))
    final_values = jnp.asarray(rng.normal(0, 0.2, shape[2:]).astype(np.float32))
    final_values = jnp.broadcast_to(final_values, (groups, *final_values.shape))
    final_done = (
        jnp.zeros((groups, environments, actors), dtype=bool).at[:, 1, :].set(True)
    )
    carries = ns["HiddenStates"](
        jnp.broadcast_to(
            actor_carry[:, None], (groups, time, environments, actors, 128)
        ),
        jnp.broadcast_to(
            critic_carry[:, None], (groups, time, environments, actors, 128)
        ),
    )
    trajectory = ns["RNNPPOTransition"](
        starts, actions, old_values, rewards, old_logs, observation, carries
    )
    gae = jax.vmap(ns["calculate_gae"], in_axes=(0, 0, 0, None, None))
    advantages, targets = gae(
        trajectory, final_values, final_done, system.gamma, system.gae_lambda
    )
    keys = jax.random.split(jax.random.PRNGKey(29), groups)

    # These first-half minibatches have fixed membership, independent of shuffling.
    first_batch = jax.tree.map(lambda value: value[:, :, :2], trajectory)
    actor_grad_fn = jax.value_and_grad(ns["_actor_loss_fn"], has_aux=True)
    critic_grad_fn = jax.value_and_grad(ns["_critic_loss_fn"], has_aux=True)
    actor_loss, actor_grads = jax.vmap(actor_grad_fn, in_axes=(None, 0, 0, 0))(
        actor_params, first_batch, advantages[:, :, :2], keys
    )
    critic_loss, critic_grads = jax.vmap(critic_grad_fn, in_axes=(None, 0, 0))(
        critic_params, first_batch, targets[:, :, :2]
    )
    mean_actor_grads = jax.tree.map(lambda value: value.mean(0), actor_grads)
    mean_critic_grads = jax.tree.map(lambda value: value.mean(0), critic_grads)
    actor_updates, actor_opt_next = actor_optim.update(
        mean_actor_grads, opt_states.actor_opt_state
    )
    critic_updates, critic_opt_next = critic_optim.update(
        mean_critic_grads, opt_states.critic_opt_state
    )

    def broadcast_groups(value: Any) -> Any:
        return jnp.broadcast_to(value, (1, groups, *value.shape))

    state = (
        jax.tree.map(broadcast_groups, params),
        jax.tree.map(broadcast_groups, opt_states),
        jax.tree.map(lambda value: value[None], trajectory),
        advantages[None],
        targets[None],
        keys[None],
    )
    epoch = jax.vmap(
        jax.vmap(ns["_update_epoch"], in_axes=(0, None), axis_name="batch"),
        in_axes=(0, None),
        axis_name="device",
    )
    final_state, epoch_losses = jax.lax.scan(epoch, state, None, system.ppo_epochs)

    arrays: dict[str, NDArray[Any]] = {}
    for name, value in {
        "actor_features": actor_features,
        "critic_features": critic_features,
        "action_mask": masks,
        "episode_start": starts,
        "actor_carry": actor_carry,
        "critic_carry": critic_carry,
        "actions": actions,
        "old_log_probabilities": old_logs,
        "old_values": old_values,
        "rewards": rewards,
        "final_values": final_values,
        "final_done": final_done,
        "update_keys": keys,
    }.items():
        arrays[f"input/{name}"] = np.asarray(value)
    for name, value in {
        "actor_carry": next_a,
        "critic_carry": next_c,
        "masked_logits": logits,
        "raw_logits": raw_logits,
        "log_probabilities": log_probabilities,
        "entropy": entropy,
        "values": values,
        "advantages": advantages,
        "targets": targets,
    }.items():
        arrays[f"expected/{name}"] = np.asarray(value)
    trees = {
        "parameters/actor": actor_params,
        "parameters/critic": critic_params,
        "first_minibatch/actor_losses": actor_loss,
        "first_minibatch/critic_losses": critic_loss,
        "first_minibatch/actor_group_gradients": actor_grads,
        "first_minibatch/critic_group_gradients": critic_grads,
        "first_minibatch/actor_gradients": mean_actor_grads,
        "first_minibatch/critic_gradients": mean_critic_grads,
        "first_minibatch/actor_parameters": optax.apply_updates(
            actor_params, actor_updates
        ),
        "first_minibatch/critic_parameters": optax.apply_updates(
            critic_params, critic_updates
        ),
        "first_minibatch/actor_optimizer": actor_opt_next,
        "first_minibatch/critic_optimizer": critic_opt_next,
        "four_epochs/parameters": jax.tree.map(
            lambda value: value[0, 0], final_state[0]
        ),
        "four_epochs/optimizer": jax.tree.map(
            lambda value: value[0, 0], final_state[1]
        ),
        "four_epochs/losses": epoch_losses,
        "four_epochs/keys": final_state[-1],
    }
    for name, value in trees.items():
        _store_tree(arrays, name, value, jax)
    permutations = []
    current_keys = keys
    for _ in range(system.ppo_epochs):
        split_keys = jax.vmap(lambda key: jax.random.split(key, 3))(current_keys)
        permutations.append(
            jax.vmap(lambda key: jax.random.permutation(key, environments))(
                split_keys[:, 1]
            )
        )
        current_keys = split_keys[:, 0]
    arrays["input/epoch_permutations"] = np.asarray(jnp.stack(permutations))
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("The donor produced a nonfinite reference value.")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / _DATA_FILE
    cast(Callable[..., None], np.savez_compressed)(data_path, **arrays)
    metadata = {
        "schema_version": 1,
        "donor_commit": manifest["commit"],
        "donor_tree": manifest["tree"],
        "python": platform.python_version(),
        "packages": packages,
        "backend": jax.default_backend(),
        "source_manifest_sha256": hashlib.sha256(
            (_FIXTURE_ROOT / "source_manifest.json").read_bytes()
        ).hexdigest(),
        "archive_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requirements_sha256": hashlib.sha256(
            (_FIXTURE_ROOT / "reference-requirements.txt").read_bytes()
        ).hexdigest(),
        "settings": omega.to_container(config, resolve=True),
        "shapes": {
            "groups": groups,
            "time": time,
            "environments_per_group": environments,
            "actors": actors,
            "actor_features": 13,
            "critic_features": 17,
            "hidden_width": 128,
            "actions": 198,
        },
        "array_count": len(arrays),
        "limits": [
            "Synthetic all-valid CPU calculation, not a training run.",
            "No BG masking adaptations are included.",
            "Production comparisons use stored parameters; cross-version initializer "
            "identity is not claimed.",
        ],
    }
    (output_dir / "reference.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(arrays)} arrays to {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce the pinned CPU MAPPO reference."
    )
    parser.add_argument(
        "--generate", required=True, type=Path, metavar="OUTPUT_DIRECTORY"
    )
    generate_reference(parser.parse_args().generate)
