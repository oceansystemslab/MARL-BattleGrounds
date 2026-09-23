"""Load and rebuild independent pinned-source PPO numerical references.

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

Pass ``method="ippo"``, ``"ff_mappo"`` or ``"ff_ippo"`` to build the matching
current-stack reference. ``load_variants_reference(method)`` reads that method's
historical arrays. ``--methods ippo ff_mappo ff_ippo --generate DIRECTORY``
generates a separate combined archive from the additional pinned source files.
The old default generator and historical MAPPO files retain their own meaning.

Recurrent QMIX has its own route and files: ``qmix_source_manifest.json``
(bound to the MAPPO manifest by hash), ``qmix_reference.json`` and ``qmix.npz``.
``load_qmix_reference`` checks and reads them. ``build_same_stack_qmix_reference``
executes the donor's Q-network, mixer, epsilon-greedy distribution, action
selection, sample preparation, loss and ``update_q`` bodies on the current
stack; callers set ``cfg``, ``q_net``, ``mixer`` and ``opt`` (see
``reference_qmix_optimizer``) before calling them. Its categorical wrapper
supplies only probabilities, the mode and sampling, which the historical
archive checks. ``--qmix --generate DIRECTORY`` writes that archive in the
reference environment. ``update_q`` needs the nested ``device`` and ``batch``
vmap axes it averages over; with one lane each the averages change nothing.
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
_DONOR_METHODS = {
    "mappo": "rec_mappo",
    "ippo": "rec_ippo",
    "ff_mappo": "ff_mappo",
    "ff_ippo": "ff_ippo",
}
_QMIX_MANIFEST = "qmix_source_manifest.json"
_QMIX_METADATA = "qmix_reference.json"
_QMIX_DATA_FILE = "qmix.npz"
_QMIX_SYSTEM = "mava/systems/q_learning/anakin/rec_qmix.py"


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


def load_variants_reference(
    method: str,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    if method not in _DONOR_METHODS or method == "mappo":
        raise ValueError("A variants reference requires ippo, ff_mappo or ff_ippo.")
    manifest = _manifest(method)
    for path in manifest["files"]:
        _source_text(path, manifest)
    metadata = json.loads((_FIXTURE_ROOT / "variants_reference.json").read_text())
    if (
        hashlib.sha256(
            (_FIXTURE_ROOT / "variants_source_manifest.json").read_bytes()
        ).hexdigest()
        != metadata["source_manifest_sha256"]
    ):
        raise ValueError("The variants source manifest does not match the reference.")
    data_path = _FIXTURE_ROOT / "ppo_variants.npz"
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != metadata["archive_sha256"]:
        raise ValueError("The variants archive does not match its recorded hash.")
    marker = method + "/"
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {
            key.removeprefix(marker): archive[key].copy()
            for key in archive.files
            if key.startswith(marker)
        }
    return metadata["methods"][method], arrays


def _manifest(method: str) -> dict[str, Any]:
    if method not in _DONOR_METHODS:
        raise ValueError(f"Unknown donor method: {method}")
    name = (
        "source_manifest.json" if method == "mappo" else "variants_source_manifest.json"
    )
    manifest = json.loads((_FIXTURE_ROOT / name).read_text())
    if (
        method != "mappo"
        and hashlib.sha256(
            (_FIXTURE_ROOT / "source_manifest.json").read_bytes()
        ).hexdigest()
        != manifest["base_manifest_sha256"]
    ):
        raise ValueError("The historical source manifest changed.")
    return manifest


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


def build_same_stack_reference(method: str = "mappo") -> dict[str, Any]:
    return _reference_namespace(_manifest(method), same_stack=True, method=method)


def reference_optimizers(
    namespace: dict[str, Any], method: str = "mappo"
) -> tuple[Any, Any]:
    path = f"mava/systems/ppo/anakin/{_DONOR_METHODS[method]}.py"
    setup = _source_node(_source_text(path, _manifest(method)), ("learner_setup",))
    statements: list[ast.stmt] = [
        child
        for child in getattr(setup, "body", [])
        if isinstance(child, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in ("actor_optim", "critic_optim")
            for target in child.targets
        )
    ]
    namespace["actor_lr"] = namespace["config"].system.actor_lr
    namespace["critic_lr"] = namespace["config"].system.critic_lr
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
            path,
            "exec",
        ),
        namespace,
    )
    actor, critic = namespace["actor_optim"], namespace["critic_optim"]
    namespace["actor_update_fn"], namespace["critic_update_fn"] = (
        actor.update,
        critic.update,
    )
    return actor, critic


def _reference_namespace(
    manifest: dict[str, Any], *, same_stack: bool = False, method: str = "mappo"
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
    module = ModuleType(
        "mava_pinned_reference_" + method + ("_same_stack" if same_stack else "")
    )
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
            (
                ("Params",),
                ("OptStates",),
                ("HiddenStates",),
                ("RNNPPOTransition",),
                ("PPOTransition",),
            ),
        ),
        ("mava/networks/torsos.py", (("_parse_activation_fn",), ("MLPTorso",))),
        ("mava/networks/distributions.py", (("IdentityTransformation",),)),
        ("mava/networks/heads.py", (("DiscreteActionHead",),)),
        (
            "mava/networks/base.py",
            (
                ("ScannedRNN",),
                ("RecurrentActor",),
                ("RecurrentValueNet",),
                ("FeedForwardActor",),
                ("FeedForwardValueNet",),
            ),
        ),
        ("mava/utils/multistep.py", (("calculate_gae",),)),
        (
            f"mava/systems/ppo/anakin/{_DONOR_METHODS[method]}.py",
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
    if method.startswith("ff_"):
        _exec_source(
            namespace,
            manifest,
            "mava/utils/jax_utils.py",
            (("ndim_at_least",), ("merge_leading_dims",)),
        )
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


def generate_reference(output_dir: Path, method: str = "mappo") -> None:
    packages = _check_environment()
    manifest = _manifest(method)
    recurrent = not method.startswith("ff_")
    centralized = method in ("mappo", "ff_mappo")
    donor_method = _DONOR_METHODS[method]
    for path in manifest["files"]:
        _source_text(path, manifest)
    ns = _reference_namespace(manifest, method=method)
    jax, jnp, optax = (ns[name] for name in ("jax", "jnp", "optax"))
    if jax.default_backend() != "cpu":
        raise ValueError("Reference generation requires JAX_PLATFORMS=cpu.")
    omega = importlib.import_module("omegaconf").OmegaConf
    system = omega.create(
        _source_text(f"mava/configs/system/ppo/{donor_method}.yaml", manifest)
    )
    architecture = omega.create(_source_text("mava/configs/arch/anakin.yaml", manifest))
    # Small CPU tensors preserve both group and minibatch axes and all five actors.
    groups, time, environments, actors = 2, 4, 4, 5
    system.rollout_length = time
    if recurrent:
        system.recurrent_chunk_size = time
    architecture.num_envs = environments
    config = omega.create({"system": system, "arch": architecture})
    ns["config"] = config
    network = omega.create(
        _source_text(
            f"mava/configs/network/{'rnn' if recurrent else 'mlp'}.yaml", manifest
        )
    )

    def torso(settings: Any) -> Any:
        return ns["MLPTorso"](
            tuple(settings.layer_sizes),
            activation=settings.activation,
            use_layer_norm=settings.use_layer_norm,
        )

    if recurrent:
        actor = ns["RecurrentActor"](
            torso(network.actor_network.pre_torso),
            torso(network.actor_network.post_torso),
            ns["DiscreteActionHead"](198),
            hidden_state_dim=network.hidden_state_dim,
        )
        critic = ns["RecurrentValueNet"](
            torso(network.critic_network.pre_torso),
            torso(network.critic_network.post_torso),
            centralised_critic=centralized,
            hidden_state_dim=network.hidden_state_dim,
        )
    else:
        actor = ns["FeedForwardActor"](
            torso(network.actor_network.pre_torso), ns["DiscreteActionHead"](198)
        )
        critic = ns["FeedForwardValueNet"](
            torso(network.critic_network.pre_torso), centralised_critic=centralized
        )
    ns["actor_apply_fn"], ns["critic_apply_fn"] = actor.apply, critic.apply

    # Execute the original optimizer statements, not a BG optimizer helper.
    actor_optim, critic_optim = reference_optimizers(ns, method)

    rng = np.random.default_rng(190919)
    shape = (groups, time, environments, actors)
    actor_features = jnp.asarray(rng.normal(size=(*shape, 13)).astype(np.float32))
    critic_features = jnp.asarray(rng.normal(size=(*shape, 17)).astype(np.float32))
    if not centralized:
        critic_features = actor_features
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
    if recurrent:
        actor_params = actor.init(
            jax.random.PRNGKey(19), actor_carry[0], (example_observation, starts[0])
        )
        critic_params = critic.init(
            jax.random.PRNGKey(23), critic_carry[0], (example_observation, starts[0])
        )
    else:
        actor_params = actor.init(jax.random.PRNGKey(19), example_observation)
        critic_params = critic.init(jax.random.PRNGKey(23), example_observation)
    params = ns["Params"](actor_params, critic_params)
    opt_states = ns["OptStates"](
        actor_optim.init(actor_params), critic_optim.init(critic_params)
    )

    def forward(carry_a: Any, carry_c: Any, obs: Any, reset: Any) -> tuple[Any, ...]:
        if recurrent:
            next_a, distribution = actor.apply(actor_params, carry_a, (obs, reset))
            next_c, value = critic.apply(critic_params, carry_c, (obs, reset))
        else:
            distribution = actor.apply(actor_params, obs)
            value = critic.apply(critic_params, obs)
            next_a, next_c = (), ()
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
    trajectory = (
        ns["RNNPPOTransition"](
            starts, actions, old_values, rewards, old_logs, observation, carries
        )
        if recurrent
        else ns["PPOTransition"](
            starts, actions, old_values, rewards, old_logs, observation
        )
    )
    gae = jax.vmap(ns["calculate_gae"], in_axes=(0, 0, 0, None, None))
    advantages, targets = gae(
        trajectory, final_values, final_done, system.gamma, system.gae_lambda
    )
    keys = jax.random.split(jax.random.PRNGKey(29), groups)

    # These first-half minibatches have fixed membership, independent of shuffling.
    if recurrent:
        first_batch = jax.tree.map(lambda value: value[:, :, :2], trajectory)
        first_advantages, first_targets = advantages[:, :, :2], targets[:, :, :2]
    else:

        def first_rows(value: Any) -> Any:
            return value.reshape(groups, time * environments, *value.shape[3:])[
                :, : time * environments // 2
            ]

        first_batch = jax.tree.map(first_rows, trajectory)
        first_advantages, first_targets = first_rows(advantages), first_rows(targets)
    actor_grad_fn = jax.value_and_grad(ns["_actor_loss_fn"], has_aux=True)
    critic_grad_fn = jax.value_and_grad(ns["_critic_loss_fn"], has_aux=True)
    actor_loss, actor_grads = jax.vmap(actor_grad_fn, in_axes=(None, 0, 0, 0))(
        actor_params, first_batch, first_advantages, keys
    )
    critic_loss, critic_grads = jax.vmap(critic_grad_fn, in_axes=(None, 0, 0))(
        critic_params, first_batch, first_targets
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
        **(
            {"actor_carry": actor_carry, "critic_carry": critic_carry}
            if recurrent
            else {}
        ),
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
        **({"actor_carry": next_a, "critic_carry": next_c} if recurrent else {}),
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
            jax.vmap(
                lambda key: jax.random.permutation(
                    key, environments if recurrent else time * environments
                )
            )(split_keys[:, 1])
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
            (
                _FIXTURE_ROOT
                / (
                    "source_manifest.json"
                    if method == "mappo"
                    else "variants_source_manifest.json"
                )
            ).read_bytes()
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
            "critic_features": 17 if centralized else 13,
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


def generate_variants_reference(output_dir: Path, methods: list[str]) -> None:
    if len(set(methods)) != len(methods) or any(
        method not in _DONOR_METHODS or method == "mappo" for method in methods
    ):
        raise ValueError("Choose each of ippo, ff_mappo and ff_ippo at most once.")
    arrays: dict[str, NDArray[Any]] = {}
    details: dict[str, Any] = {}
    for method in methods:
        destination = output_dir / method
        generate_reference(destination, method)
        details[method] = json.loads((destination / "reference.json").read_text())
        with np.load(destination / _DATA_FILE, allow_pickle=False) as archive:
            for key in archive.files:
                arrays[f"{method}/{key}"] = archive[key].copy()
    data_path = output_dir / "ppo_variants.npz"
    cast(Callable[..., None], np.savez_compressed)(data_path, **arrays)
    metadata = {
        "schema_version": 1,
        "source_manifest_sha256": hashlib.sha256(
            (_FIXTURE_ROOT / "variants_source_manifest.json").read_bytes()
        ).hexdigest(),
        "archive_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "methods": details,
        "array_count": len(arrays),
    }
    (output_dir / "variants_reference.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def _qmix_manifest() -> dict[str, Any]:
    manifest = json.loads((_FIXTURE_ROOT / _QMIX_MANIFEST).read_text())
    if (
        hashlib.sha256(
            (_FIXTURE_ROOT / "source_manifest.json").read_bytes()
        ).hexdigest()
        != manifest["base_manifest_sha256"]
    ):
        raise ValueError("The historical source manifest changed.")
    return manifest


def load_qmix_reference() -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    manifest = _qmix_manifest()
    for path in manifest["files"]:
        _source_text(path, manifest)
    metadata = json.loads((_FIXTURE_ROOT / _QMIX_METADATA).read_text())
    if (
        hashlib.sha256((_FIXTURE_ROOT / _QMIX_MANIFEST).read_bytes()).hexdigest()
        != metadata["source_manifest_sha256"]
    ):
        raise ValueError("The QMIX source manifest does not match the reference.")
    data_path = _FIXTURE_ROOT / _QMIX_DATA_FILE
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != metadata["archive_sha256"]:
        raise ValueError("The QMIX archive does not match its recorded hash.")
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return metadata, arrays


def _eps_greedy_shim(jax: Any, jnp: Any) -> SimpleNamespace:
    class Categorical:
        def __init__(self, probs: Any) -> None:
            self.probs = probs

        @classmethod
        def _parameter_properties(
            cls, dtype: Any, num_classes: Any = None
        ) -> dict[str, Any]:
            del dtype, num_classes
            return {}

        def probs_parameter(self) -> Any:
            return self.probs

        def mode(self) -> Any:
            return jnp.argmax(self.probs, axis=-1).astype(jnp.int32)

        def sample(self, seed: Any) -> Any:
            return jax.random.categorical(seed, jnp.log(self.probs)).astype(jnp.int32)

    return SimpleNamespace(Categorical=Categorical)


def _qmix_namespace(manifest: dict[str, Any], *, same_stack: bool) -> dict[str, Any]:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    nn = importlib.import_module("flax.linen")
    tfd = (
        _eps_greedy_shim(jax, jnp)
        if same_stack
        else importlib.import_module(
            "tensorflow_probability.substrates.jax.distributions"
        )
    )
    module = ModuleType(
        "mava_pinned_reference_qmix" + ("_same_stack" if same_stack else "")
    )
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update(
        {
            "functools": functools,
            "jax": jax,
            "jnp": jnp,
            "lax": importlib.import_module("jax.lax"),
            "np": np,
            "nn": nn,
            "orthogonal": nn.initializers.orthogonal,
            "tree": jax.tree,
            "optax": importlib.import_module("optax"),
            "chex": importlib.import_module("chex"),
            "NamedTuple": NamedTuple,
            "tfd": tfd,
            # The fixture uses ordinary arrays, never donor graph observations.
            "is_graph_observation": lambda _observation: False,
        }
    )
    sources = (
        ("mava/types.py", (("ObservationGlobalState",),)),
        (
            "mava/systems/q_learning/types.py",
            (("Transition",), ("ActionSelectionState",), ("QMIXParams",)),
        ),
        ("mava/networks/torsos.py", (("_parse_activation_fn",), ("MLPTorso",))),
        ("mava/networks/distributions.py", (("MaskedEpsGreedyDistribution",),)),
        (
            "mava/networks/base.py",
            (("ScannedRNN",), ("RecQNetwork",), ("QMixingNetwork",)),
        ),
        ("mava/utils/jax_utils.py", (("add_batch_dim",), ("switch_leading_axes",))),
        (
            _QMIX_SYSTEM,
            (
                ("make_update_fns", "select_eps_greedy_action"),
                ("make_update_fns", "prep_inputs_to_scannedrnn"),
                ("make_update_fns", "q_loss_fn"),
                ("make_update_fns", "update_q"),
            ),
        ),
    )
    for path, symbols in sources:
        _exec_source(namespace, manifest, path, symbols)
    return namespace


def build_same_stack_qmix_reference() -> dict[str, Any]:
    return _qmix_namespace(_qmix_manifest(), same_stack=True)


def reference_qmix_optimizer(namespace: dict[str, Any]) -> Any:
    setup = _source_node(_source_text(_QMIX_SYSTEM, _qmix_manifest()), ("init",))
    statements: list[ast.stmt] = [
        child
        for child in getattr(setup, "body", [])
        if isinstance(child, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "opt"
            for target in child.targets
        )
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
            _QMIX_SYSTEM,
            "exec",
        ),
        namespace,
    )
    return namespace["opt"]


def qmix_reference_networks(
    namespace: dict[str, Any], network: Any, embed_dim: int, agents: int = 5
) -> tuple[Any, Any]:
    # Keyword arguments follow the donor's hydra instantiation of the YAML nodes.
    def torso(settings: Any) -> Any:
        return namespace["MLPTorso"](
            tuple(settings["layer_sizes"]),
            activation=settings["activation"],
            use_layer_norm=settings["use_layer_norm"],
        )

    q_net = namespace["RecQNetwork"](
        pre_torso=torso(network["q_network"]["pre_torso"]),
        post_torso=torso(network["q_network"]["post_torso"]),
        num_actions=198,
        hidden_state_dim=network["hidden_state_dim"],
    )
    mixer = namespace["QMixingNetwork"](
        num_actions=198,
        num_agents=agents,
        hyper_hidden_dim=network["mixer_network"]["hyper_hidden_dim"],
        embed_dim=embed_dim,
        norm_env_states=network["mixer_network"]["norm_env_states"],
    )
    namespace["q_net"], namespace["mixer"] = q_net, mixer
    return q_net, mixer


def qmix_reference_update(namespace: dict[str, Any]) -> Callable[..., Any]:
    jax = namespace["jax"]

    def run(params: Any, opt_state: Any, data: Any, t_train: Any) -> Any:
        def lift(value: Any) -> Any:
            return jax.tree.map(lambda leaf: leaf[None, None], value)

        update = jax.vmap(
            jax.vmap(namespace["update_q"], in_axes=(0, 0, 0, None), axis_name="batch"),
            in_axes=(0, 0, 0, None),
            axis_name="device",
        )
        result = update(lift(params), lift(opt_state), lift(data), t_train)
        return jax.tree.map(lambda leaf: leaf[0, 0], result)

    return run


def qmix_gradient_capture(namespace: dict[str, Any]) -> Any:
    # A zero-step optimizer whose new state is the donor's averaged gradient.
    jax, jnp, optax = (namespace[name] for name in ("jax", "jnp", "optax"))

    def init(params: Any) -> Any:
        return params

    def update(grads: Any, state: Any, params: Any = None) -> tuple[Any, Any]:
        del state, params
        return jax.tree.map(jnp.zeros_like, grads), grads

    return optax.GradientTransformation(init, update)


def _trees_equal(left: Any, right: Any, jax: Any) -> bool:
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def generate_qmix_reference(output_dir: Path) -> None:
    packages = _check_environment()
    manifest = _qmix_manifest()
    for path in manifest["files"]:
        _source_text(path, manifest)
    ns = _qmix_namespace(manifest, same_stack=False)
    jax, jnp = ns["jax"], ns["jnp"]
    if jax.default_backend() != "cpu":
        raise ValueError("Reference generation requires JAX_PLATFORMS=cpu.")
    omega = importlib.import_module("omegaconf").OmegaConf
    system = omega.create(
        _source_text("mava/configs/system/q_learning/rec_qmix.yaml", manifest)
    )
    network = omega.create(_source_text("mava/configs/network/qmix_rnn.yaml", manifest))
    architecture = omega.create(_source_text("mava/configs/arch/anakin.yaml", manifest))
    donor_defaults = omega.to_container(system, resolve=True)
    # Small CPU tensors keep five agents and the donor's real 256-wide networks.
    sequences, rows, lanes, agents, actor_width, state_width = 4, 6, 4, 5, 13, 17
    system.sample_batch_size = sequences
    system.sample_sequence_length = rows
    architecture.num_envs = lanes
    cfg = omega.create({"system": system, "network": network, "arch": architecture})
    ns["cfg"] = cfg
    q_net, mixer = qmix_reference_networks(ns, network, system.qmix_embed_dim, agents)
    opt = reference_qmix_optimizer(ns)
    width = int(network.hidden_state_dim)

    rng = np.random.default_rng(190923)
    shape = (sequences, rows, agents)
    features = rng.normal(size=(*shape, actor_width)).astype(np.float32)
    masks = rng.random((*shape, 198)) > 0.22
    masks[..., 0] = True
    # One dead actor row admits only the neutral action.
    masks[1, 2, 3] = False
    masks[1, 2, 3, 0] = True
    state_rows = rng.normal(size=(sequences, rows, state_width)).astype(np.float32)
    # The donor environment repeats the physical state for every agent.
    global_state = np.broadcast_to(state_rows[:, :, None], (*shape, state_width)).copy()
    actions = np.empty(shape, dtype=np.int32)
    for index in np.ndindex(shape):
        actions[index] = rng.choice(np.flatnonzero(masks[index]))
    rewards = rng.normal(0, 0.3, (sequences, rows, 1)).astype(np.float32)
    # term_or_trunc marks a row that starts a new episode: interior, first and last.
    starts = np.zeros((sequences, rows, 1), dtype=bool)
    starts[0, 3] = starts[2, 0] = starts[3, rows - 1] = True
    carry = jnp.asarray(
        rng.normal(0, 0.2, (sequences, agents, width)).astype(np.float32)
    )
    agent_qs = jnp.asarray(
        rng.normal(size=(sequences, rows - 1, agents)).astype(np.float32)
    )
    observation = ns["ObservationGlobalState"](
        jnp.asarray(features), jnp.asarray(masks), jnp.asarray(global_state)
    )
    data = ns["Transition"](
        observation,
        jnp.asarray(actions),
        jnp.asarray(rewards),
        jnp.asarray(starts),
        jnp.asarray(starts),
        observation,
    )

    # Initialization uses the donor's own dummy input layout.
    init_obs = ns["ObservationGlobalState"](
        jnp.zeros((1, 1, agents, actor_width)),
        jnp.ones((1, 1, agents, 198), dtype=bool),
        jnp.zeros((1, 1, agents, state_width)),
    )
    init_x = (init_obs, jnp.zeros((1, 1, 1), dtype=bool))
    init_hidden = ns["ScannedRNN"].initialize_carry((lanes, agents), width)
    dummy_qs = jnp.zeros((sequences, rows - 1, agents))
    dummy_state = jnp.zeros((sequences, rows - 1, state_width))
    q_key, target_key = jax.random.PRNGKey(31), jax.random.PRNGKey(37)
    q_params = q_net.init(q_key, init_hidden, init_x)
    mixer_params = mixer.init(q_key, dummy_qs, dummy_state)
    # A distinct target makes the Double-Q choice observable.
    target_q = q_net.init(target_key, init_hidden, init_x)
    target_mixer = mixer.init(target_key, dummy_qs, dummy_state)

    time_major = ns["switch_leading_axes"](observation)
    resets = ns["switch_leading_axes"](data.term_or_trunc)
    forward_carry, forward_q = q_net.apply(
        q_params, carry, (time_major, resets), method="get_q_values"
    )
    rates = (0.0, 0.37, 1.0)
    probabilities, modes = [], []
    for rate in rates:
        _, distribution = q_net.apply(q_params, carry, (time_major, resets), rate)
        probabilities.append(distribution.probs_parameter())
        modes.append(distribution.mode())

    q_rows = rng.normal(size=(7, 198)).astype(np.float32)
    mask_rows = rng.random((7, 198)) > 0.3
    mask_rows[:, 0] = True
    # Rows 1, 2 and 5 hold exact ties; the lowest legal index must win.
    mask_rows[1, [5, 17]] = True
    q_rows[1, [5, 17]] = q_rows[1].max() + 1
    mask_rows[2, 197] = True
    q_rows[2, [0, 197]] = q_rows[2].max() + 1
    mask_rows[3] = False
    mask_rows[3, 0] = True
    mask_rows[4] = True
    mask_rows[5] = True
    q_rows[5, [3, 90, 150]] = q_rows[5].max() + 1
    # Row 6 records the donor's finfo.min edge: only 197 is legal.
    mask_rows[6] = False
    mask_rows[6, 197] = True
    q_rows[6] = np.finfo(np.float32).min
    row_probabilities, row_modes = [], []
    for rate in rates:
        distribution = ns["MaskedEpsGreedyDistribution"](
            jnp.asarray(q_rows), rate, jnp.asarray(mask_rows)
        )
        row_probabilities.append(distribution.probs_parameter())
        row_modes.append(distribution.mode())

    mixed = mixer.apply(mixer_params, agent_qs, observation.global_state[:, :-1, 0])

    recorded: list[Any] = []

    def recorded_apply(*args: Any, **kwargs: Any) -> Any:
        recorded.append(args[3])
        return q_net.apply(*args, **kwargs)

    ns["q_net"] = SimpleNamespace(apply=recorded_apply)
    epsilon_rounds = (0, 1, 2, 17, 24999, 25000, 25001, 30000)
    next_steps, select_hidden = [], None
    lane_obs = jax.tree.map(lambda value: value[:lanes, 0], observation)
    for count in epsilon_rounds:
        selection, _action = ns["select_eps_greedy_action"](
            ns["ActionSelectionState"](
                q_params,
                carry[:lanes],
                jnp.asarray(lanes * count, dtype=jnp.int32),
                jax.random.PRNGKey(41),
            ),
            lane_obs,
            data.term_or_trunc[:lanes, 0],
        )
        next_steps.append(selection.time_steps)
        if select_hidden is None:
            select_hidden = selection.hidden_state
    ns["q_net"] = q_net

    update = qmix_reference_update(ns)
    base = ns["QMIXParams"](q_params, target_q, mixer_params, target_mixer)
    ns["opt"] = qmix_gradient_capture(ns)
    _, gradients, captured_loss = update(
        base, (q_params, mixer_params), data, jnp.int32(1)
    )
    ns["opt"] = opt
    adam = opt.init((q_params, mixer_params))
    hard = {
        count: update(base, adam, data, jnp.int32(count)) for count in (0, 1, 199, 200)
    }
    copies = {
        str(count): _trees_equal(result[0].target, result[0].online, jax)
        and _trees_equal(result[0].mixer_target, result[0].mixer_online, jax)
        for count, result in hard.items()
    }
    kept = {
        str(count): _trees_equal(result[0].target, target_q, jax)
        and _trees_equal(result[0].mixer_target, target_mixer, jax)
        for count, result in hard.items()
    }
    if copies != {"0": True, "1": False, "199": False, "200": True} or kept != {
        "0": False,
        "1": True,
        "199": True,
        "200": False,
    }:
        raise ValueError("The donor's hard target timing differs from the brief.")
    first_params, first_opt, first_loss = hard[0]
    second_params, _second_opt, second_loss = update(
        first_params, first_opt, data, jnp.int32(1)
    )
    cfg.system.hard_update = False
    soft_params, _soft_opt, soft_loss = update(base, adam, data, jnp.int32(0))
    cfg.system.hard_update = True

    arrays: dict[str, NDArray[Any]] = {}
    for name, value in {
        "actor_features": features,
        "action_mask": masks,
        "training_state": state_rows,
        "actions": actions,
        "rewards": rewards,
        "episode_start": starts,
        "carry": carry,
        "agent_qs": agent_qs,
        "q_rows": q_rows,
        "mask_rows": mask_rows,
        "epsilon_rounds": np.asarray(epsilon_rounds, dtype=np.int32),
    }.items():
        arrays[f"input/{name}"] = np.asarray(value)
    for name, value in {
        "forward_carry": forward_carry,
        "forward_q": forward_q,
        "probabilities": jnp.stack(probabilities),
        "modes": jnp.stack(modes),
        "row_probabilities": jnp.stack(row_probabilities),
        "row_modes": jnp.stack(row_modes),
        "mixer": mixed,
        "epsilon": jnp.stack(recorded),
        "epsilon_next_steps": jnp.stack(next_steps),
        "select_hidden": select_hidden,
    }.items():
        arrays[f"expected/{name}"] = np.asarray(value)
    trees = {
        "parameters/online_q": q_params,
        "parameters/target_q": target_q,
        "parameters/online_mixer": mixer_params,
        "parameters/target_mixer": target_mixer,
        "gradients/q": gradients[0],
        "gradients/mixer": gradients[1],
        "gradients/loss": captured_loss,
        "one_update/online_q": first_params.online,
        "one_update/online_mixer": first_params.mixer_online,
        "one_update/optimizer": first_opt,
        "one_update/loss": first_loss,
        "two_updates/online_mixer": second_params.mixer_online,
        "two_updates/loss": second_loss,
        "soft_update/target_q": soft_params.target,
        "soft_update/target_mixer": soft_params.mixer_target,
        "soft_update/loss": soft_loss,
    }
    for name, value in trees.items():
        _store_tree(arrays, name, value, jax)
    if not all(
        np.all(np.isfinite(value))
        for key, value in arrays.items()
        if np.issubdtype(value.dtype, np.floating) and key != "input/q_rows"
    ):
        raise ValueError("The donor produced a nonfinite reference value.")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / _QMIX_DATA_FILE
    cast(Callable[..., None], np.savez_compressed)(data_path, **arrays)
    metadata = {
        "schema_version": 1,
        "donor_commit": manifest["commit"],
        "donor_tree": manifest["tree"],
        "python": platform.python_version(),
        "packages": packages,
        "backend": jax.default_backend(),
        "source_manifest_sha256": hashlib.sha256(
            (_FIXTURE_ROOT / _QMIX_MANIFEST).read_bytes()
        ).hexdigest(),
        "archive_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requirements_sha256": hashlib.sha256(
            (_FIXTURE_ROOT / "reference-requirements.txt").read_bytes()
        ).hexdigest(),
        "settings": omega.to_container(cfg, resolve=True),
        "donor_defaults": donor_defaults,
        "rates": list(rates),
        "keys": {"online": 31, "target": 37, "selection": 41},
        "hard_update_copies": copies,
        "hard_update_keeps_target": kept,
        "shapes": {
            "sequences": sequences,
            "rows": rows,
            "lanes": lanes,
            "agents": agents,
            "actor_features": actor_width,
            "training_state": state_width,
            "hidden_width": width,
            "actions": 198,
        },
        "array_count": len(arrays),
        "limits": [
            "Synthetic all-valid CPU calculation, not a training run.",
            "Networks are built from the donor YAML with the keyword arguments "
            "hydra would pass; hydra, flashbax and environments are not imported.",
            "Sampled actions are not stored; probabilities and greedy modes are.",
            "No BG masking, frame, reward or replay adaptations are included.",
            "Production comparisons use stored parameters; cross-version "
            "initializer identity is not claimed.",
        ],
    }
    (output_dir / _QMIX_METADATA).write_text(
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--methods", nargs="+", choices=("ippo", "ff_mappo", "ff_ippo"))
    group.add_argument("--qmix", action="store_true")
    options = parser.parse_args()
    if options.qmix:
        generate_qmix_reference(options.generate)
    elif options.methods:
        generate_variants_reference(options.generate, options.methods)
    else:
        generate_reference(options.generate)
