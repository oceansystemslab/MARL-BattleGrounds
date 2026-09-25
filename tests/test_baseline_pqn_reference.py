"""Compare recurrent PQN-VDN with independent pinned JaxMARL calculations.

The saved arrays come from unchanged donor bodies (``QNetwork`` with its
``ScannedRNN``, ``create_agent``'s optimizer, ``get_greedy_actions``,
``eps_greedy_exploration`` and ``_learn_epoch`` with its loss, targets and
minibatch split) run in an isolated CPU environment at the project's exact
numerical versions. The same bodies also run inside this test process as a
second, same-stack oracle for parameter-sized results. BG production
functions never generate either oracle. Inputs come from recorded seeds;
every row and slot is valid and active and the frame is the world frame. BG
padding, inactive-slot, frame and reward adaptations have separate tests in
``test_baseline_pqn.py``.

Contracts checked here: the settings defaults equal the donor's SMAX YAML
(the SMAX reward multiplier and the donor's 128 games are not copied);
initialize_pqn equals its own construction (``_network_variables`` plus the
optimizer) exactly at the current 5,165-feature width, and that construction
at the donor's recorded 5,164 features equals the same-stack donor exactly for
parameters, BatchNorm statistics and the clipped RAdam state, with the
isolated oracle's per-leaf digests and orthogonal recurrent kernels equal;
inference-mode values and memory; training-mode values, memory and updated
running statistics; the donor's greedy choice at epsilon 0 and its sampled
action counts at epsilon 0, 0.37 and 1 against BG's epsilon-greedy
probabilities within a 5-sigma binomial bound, with no illegal draw; and one
captured minibatch step's loss, gradients before clipping and statistics
after the donor's own permutation; and two whole learning blocks (4 games, 2
epochs of 2 minibatches, H2/T2 windows, 8 clipped RAdam steps that cross
RAdam's rectification point) run through the learner's epoch loop with the
donor's permutations: per-step losses, and parameters, statistics and
optimizer state after each block against the isolated oracle's fingerprints
and the same-stack donor leaf by leaf. Float comparisons start at 2e-6 absolute
and relative; fingerprint sums use a bound scaled by each leaf's norm because
the donor flattens actors agent-major while BG flattens game-major, which
changes float reduction order only. These checks prove numerical agreement,
not learning performance or GPU cost.
"""

# Runtime donor types cannot be imported into the repository's type environment.
# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray
from tests.baseline_donor_reference import _store_tree
from tests.pqn_donor_reference import (
    ACTIONS,
    AGENTS,
    FEATURES,
    FORWARD_ENVS,
    FORWARD_ROWS,
    FORWARD_SEED,
    GRADIENT_CASE,
    GRADIENT_SEED,
    HIDDEN,
    INIT_SEED,
    PROBABILITY_DRAWS,
    PROBABILITY_SEED,
    RATES,
    UPDATE_CASE,
    UPDATE_SEED,
    bg_layout,
    build_same_stack_pqn_reference,
    donor_config,
    donor_learn,
    fingerprint,
    forward_inputs,
    gradient_capture_state,
    load_pqn_reference,
    next_window,
    probability_inputs,
    recurrent_kernel_paths,
    to_transition,
    window_arrays,
    with_recurrent_kernels,
)

from marl_battlegrounds.baselines import pqn, qmix
from marl_battlegrounds.training import pqn_learner

type Tree = Any
type RecordProperty = Callable[[str, object], None]
_TOLERANCE = 2e-6


def _leaves(tree: Tree, prefix: str = "value") -> dict[str, NDArray[Any]]:
    stored: dict[str, NDArray[Any]] = {}
    _store_tree(stored, prefix, tree, jax)
    return stored


def _compare(actual: Tree, expected: Tree, *, exact: bool = False) -> float:
    actual_leaves, expected_leaves = _leaves(actual), _leaves(expected)
    assert actual_leaves.keys() == expected_leaves.keys()
    maximum = 0.0
    for name, value in actual_leaves.items():
        target = expected_leaves[name]
        assert value.shape == target.shape, name
        assert value.dtype == target.dtype, name
        if exact or not np.issubdtype(value.dtype, np.floating):
            np.testing.assert_array_equal(value, target, err_msg=name)
        else:
            np.testing.assert_allclose(
                value, target, atol=_TOLERANCE, rtol=_TOLERANCE, err_msg=name
            )
        maximum = max(
            maximum,
            float(
                np.abs(value.astype(np.float64) - target.astype(np.float64)).max(
                    initial=0
                )
            ),
        )
    return maximum


def _fingerprints(actual: Tree, arrays: dict[str, NDArray[Any]], prefix: str) -> float:
    prints = fingerprint(actual)
    selected = {
        key.removeprefix(prefix + "/"): value
        for key, value in arrays.items()
        if key.startswith(prefix + "/")
    }
    assert prints.keys() == selected.keys()
    worst = 0.0
    for name, value in prints.items():
        target = selected[name]
        if name.endswith("/value"):
            np.testing.assert_array_equal(value, target, err_msg=name)
            continue
        if name.endswith("/sum"):
            norm = float(selected[name.removesuffix("sum") + "norm"])
            bound = _TOLERANCE * max(norm, 1.0) * 10
            error = abs(float(value) - float(target))
            assert error <= bound, (name, error, bound)
            worst = max(worst, error / max(norm, 1.0))
        else:
            np.testing.assert_allclose(
                value, target, atol=_TOLERANCE, rtol=_TOLERANCE, err_msg=name
            )
            worst = max(worst, float(np.abs(value - target).max(initial=0)))
    return worst


def _world(case: dict[str, int]) -> pqn.PQNConfig:
    return pqn.PQNConfig(
        rollout_length=case["steps"],
        memory_window=case["window"],
        epochs=case["epochs"],
        num_minibatches=case["minibatches"],
        spawn_frame="world",
    )


def _initial_state(
    config: pqn.PQNConfig, planned: int, features: int | None = None
) -> pqn.PQNTrainState:
    # initialize_pqn's construction. The donor oracles were recorded with
    # 5,164 features, so the donor proof passes that width (FEATURES).
    key = jax.random.key(INIT_SEED)
    if features is None:
        network = pqn._network_variables(key, config.input_scale)
    else:
        network = pqn._network_variables(key, config.input_scale, features=features)
    optimizer = pqn.pqn_optimizer(config, planned)
    return pqn.PQNTrainState(network, optimizer.init(network.params), jnp.int32(0))


@dataclass
class Reference:
    metadata: dict[str, Any]
    arrays: dict[str, NDArray[Any]]
    namespace: dict[str, Any]
    donor_state: Any
    bg_state: pqn.PQNTrainState
    network: pqn.PQNInferenceVariables
    donor_variables: dict[str, Tree]


@pytest.fixture(scope="module")
def reference() -> Reference:
    metadata, arrays = load_pqn_reference()
    ns = build_same_stack_pqn_reference(GRADIENT_CASE)
    donor_state = ns["create_agent"](jax.random.key(INIT_SEED))
    bg_state = _initial_state(_world(GRADIENT_CASE), 1, FEATURES)
    params = with_recurrent_kernels(bg_state.network.params, arrays)
    network = pqn.PQNInferenceVariables(params, bg_state.network.batch_stats)
    donor_variables = {
        "params": with_recurrent_kernels(donor_state.params, arrays),
        "batch_stats": donor_state.batch_stats,
    }
    return Reference(
        metadata, arrays, ns, donor_state, bg_state, network, donor_variables
    )


def test_defaults_match_the_pinned_donor_config() -> None:
    donor = donor_config({})
    config = pqn.DEFAULT_PQN_CONFIG
    assert (
        config.rollout_length,
        config.memory_window,
        config.epochs,
        config.num_minibatches,
        config.q_lr,
        config.lr_linear_decay,
        config.max_grad_norm,
        config.gamma,
        config.td_lambda,
        config.eps_start,
        config.eps_finish,
        config.eps_decay_fraction,
    ) == (
        donor["NUM_STEPS"],
        donor["MEMORY_WINDOW"],
        donor["NUM_EPOCHS"],
        donor["NUM_MINIBATCHES"],
        donor["LR"],
        donor["LR_LINEAR_DECAY"],
        donor["MAX_GRAD_NORM"],
        donor["GAMMA"],
        donor["LAMBDA"],
        donor["EPS_START"],
        donor["EPS_FINISH"],
        donor["EPS_DECAY"],
    )
    assert (pqn.PQN_HIDDEN_SIZE, donor["HIDDEN_SIZE"], donor["NUM_LAYERS"]) == (
        512,
        512,
        2,
    )
    assert donor["NORM_INPUT"] is True and donor["NORM_TYPE"] == "batch_norm"
    # Not copied: the SMAX reward multiplier and the donor's 128 games.
    assert donor["REW_SCALE"] == 10.0 and donor["NUM_ENVS"] == 128
    assert config.input_scale == 1.0 and config.spawn_frame == "left"


def test_initialization_matches_the_same_stack_donor_exactly(
    reference: Reference, record_property: RecordProperty
) -> None:
    # initialize_pqn is exactly this construction at the current width.
    config = _world(GRADIENT_CASE)
    actual = pqn.initialize_pqn(
        jax.random.key(INIT_SEED), pqn=config, planned_learning_blocks=1
    )
    _compare(actual, _initial_state(config, 1), exact=True)
    bg = reference.bg_state
    donor = reference.donor_state
    _compare(
        {"batch_stats": bg.network.batch_stats, "params": bg.network.params},
        {"batch_stats": donor.batch_stats, "params": donor.params},
        exact=True,
    )
    _compare(bg.opt_state, donor.opt_state, exact=True)
    assert (
        str(jax.tree.structure(bg.opt_state))  # pyright: ignore[reportUnknownArgumentType]
        == reference.metadata["optimizer_structure"]
    )
    # The isolated oracle's digests match every leaf except the orthogonal
    # recurrent kernels, whose rounding depends on the CPU thread count.
    kernels = {
        "['params']" + "".join(f"['{part}']" for part in path)
        for path in recurrent_kernel_paths()
    }
    stored = reference.metadata["init_leaf_sha256"]
    flat = cast(
        list[tuple[tuple[Any, ...], Any]],
        jax.tree_util.tree_flatten_with_path(
            {"params": bg.network.params, "batch_stats": bg.network.batch_stats}
        )[0],
    )
    digests = {
        jax.tree_util.keystr(path): hashlib.sha256(
            np.asarray(leaf).tobytes()
        ).hexdigest()
        for path, leaf in flat
    }
    assert digests.keys() == stored.keys()
    for name, digest in digests.items():
        if name not in kernels:
            assert digest == stored[name], name
    error = 0.0
    for path in recurrent_kernel_paths():
        leaf: Any = bg.network.params
        for part in path:
            leaf = leaf[part]
        target = reference.arrays["init/" + "/".join(path)]
        np.testing.assert_allclose(np.asarray(leaf), target, atol=_TOLERANCE, rtol=0)
        error = max(error, float(np.abs(np.asarray(leaf) - target).max()))
    record_property("maximum_absolute_error", error)


def _bg_forward(
    network: pqn.PQNInferenceVariables, inputs: dict[str, NDArray[Any]], *, train: bool
) -> tuple[Array, Array, Tree]:
    rows, games = FORWARD_ROWS, FORWARD_ENVS
    obs = inputs["obs"].reshape(rows, AGENTS, games, -1).transpose(0, 2, 1, 3)
    hidden = inputs["hidden"].reshape(AGENTS, games, HIDDEN).transpose(1, 0, 2)
    starts = inputs["dones"].reshape(rows, AGENTS, games)[:, 0].astype(bool)
    arguments = (
        jnp.asarray(hidden),
        jnp.asarray(obs),
        jnp.asarray(starts),
        jnp.ones((rows, games), bool),
        jnp.ones((rows, games, AGENTS), bool),
    )
    variables = pqn._flax_variables(network)
    if train:
        (memory, values), updates = cast(
            tuple[tuple[Array, Array], dict[str, Tree]],
            pqn.PQNNetwork().apply(
                variables, *arguments, train=True, mutable=["batch_stats"]
            ),
        )
        stats: Tree = updates["batch_stats"]
    else:
        memory, values = cast(
            tuple[Array, Array], pqn.PQNNetwork().apply(variables, *arguments)
        )
        stats = None
    donor_values = values.transpose(0, 2, 1, 3).reshape(rows, AGENTS * games, ACTIONS)
    donor_memory = memory.transpose(1, 0, 2).reshape(AGENTS * games, HIDDEN)
    return donor_values, donor_memory, stats


@pytest.mark.parametrize("train", (False, True))
def test_forward_memory_and_statistics_match_both_donor_references(
    reference: Reference, record_property: RecordProperty, train: bool
) -> None:
    inputs = forward_inputs(FORWARD_SEED)
    values, memory, stats = _bg_forward(reference.network, inputs, train=train)
    mode = "train" if train else "inference"
    arrays = reference.arrays
    error = max(
        _compare(values, jnp.asarray(arrays[f"forward/{mode}/q"])),
        _compare(memory, jnp.asarray(arrays[f"forward/{mode}/hidden"])),
    )
    network = reference.namespace["network"]
    if train:
        (donor_memory, donor_values), updates = network.apply(
            reference.donor_variables,
            jnp.asarray(inputs["hidden"]),
            jnp.asarray(inputs["obs"]),
            jnp.asarray(inputs["dones"]),
            True,
            mutable=["batch_stats"],
        )
        stored = {
            key.removeprefix("forward/train/batch_stats/"): value
            for key, value in arrays.items()
            if key.startswith("forward/train/batch_stats/")
        }
        error = max(
            error,
            _compare(_leaves(stats, "s"), {f"s/{k}": v for k, v in stored.items()}),
            _compare(stats, updates["batch_stats"]),
        )
    else:
        donor_memory, donor_values = network.apply(
            reference.donor_variables,
            jnp.asarray(inputs["hidden"]),
            jnp.asarray(inputs["obs"]),
            jnp.asarray(inputs["dones"]),
            False,
        )
    error = max(error, _compare(values, donor_values), _compare(memory, donor_memory))
    record_property("maximum_absolute_error", error)


def test_greedy_choices_and_sampled_counts_match_the_donor(
    reference: Reference,
) -> None:
    inputs = probability_inputs(PROBABILITY_SEED)
    q_values = jnp.asarray(inputs["q_values"])
    legal = jnp.asarray(inputs["legal"])
    arrays = reference.arrays
    np.testing.assert_array_equal(
        np.asarray(pqn.greedy_actions(q_values, legal)), arrays["probability/greedy"]
    )
    donor_greedy = reference.namespace["get_greedy_actions"](
        q_values, legal.astype(jnp.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(donor_greedy), arrays["probability/greedy"]
    )
    for rate in RATES:
        counts = arrays[f"probability/counts/{rate}"]
        assert counts.sum(axis=1).tolist() == [PROBABILITY_DRAWS] * 4
        np.testing.assert_array_equal(counts[~np.asarray(inputs["legal"])], 0)
        expected = np.asarray(
            qmix.action_probabilities(q_values, legal, rate), np.float64
        )
        observed = counts / PROBABILITY_DRAWS
        bound = 5 * np.sqrt(expected * (1 - expected) / PROBABILITY_DRAWS)
        assert np.all(np.abs(observed - expected) <= bound + 1 / PROBABILITY_DRAWS)


def test_one_minibatch_loss_gradients_and_statistics_match_the_donor(
    reference: Reference, record_property: RecordProperty
) -> None:
    case = GRADIENT_CASE
    arrays = reference.arrays
    window = window_arrays(GRADIENT_SEED, case["window"] + case["steps"], case["envs"])
    permutation = arrays["gradient/permutations"][0]
    layout = bg_layout(window)
    batch = pqn.PQNBatch(
        **{
            key: jnp.asarray(
                value[permutation] if key == "initial_memory" else value[:, permutation]
            )
            for key, value in layout.items()
        }
    )
    (loss, (stats, _, _)), grads = jax.value_and_grad(
        pqn._minibatch_loss, has_aux=True
    )(
        reference.network.params,
        reference.network.batch_stats,
        batch,
        _world(case),
    )
    np.testing.assert_allclose(
        float(loss), float(arrays["gradient/loss"].reshape(-1)[0]), rtol=_TOLERANCE
    )
    error = max(
        _fingerprints(grads, arrays, "gradient/grads"),
        _fingerprints(stats, arrays, "gradient/batch_stats"),
    )
    ns = reference.namespace
    donor = reference.donor_state.replace(params=reference.donor_variables["params"])
    capture = gradient_capture_state(ns, donor)
    after, _, donor_loss = donor_learn(
        ns, capture, to_transition(ns, window), jax.random.key(GRADIENT_SEED + 1), 1
    )
    np.testing.assert_allclose(
        float(loss), float(donor_loss.reshape(-1)[0]), rtol=_TOLERANCE
    )
    error = max(
        error, _compare(grads, after.opt_state), _compare(stats, after.batch_stats)
    )
    record_property("maximum_absolute_error", error)


def _selected(batch: pqn.PQNBatch, chosen: Array) -> pqn.PQNBatch:
    def games(value: Array) -> Array:
        return jnp.take(value, chosen, axis=1)

    rows = cast(pqn.PQNBatch, jax.tree.map(games, batch._replace(initial_memory=None)))
    return rows._replace(initial_memory=jnp.take(batch.initial_memory, chosen, axis=0))


def test_two_learning_blocks_match_the_donor_through_the_epoch_loop(
    reference: Reference, record_property: RecordProperty
) -> None:
    case = UPDATE_CASE
    arrays = reference.arrays
    config = _world(case)
    planned = int(arrays["update/num_updates"])
    assert planned == case["blocks"]
    start = _initial_state(config, planned, FEATURES)
    state = start._replace(
        network=pqn.PQNInferenceVariables(
            with_recurrent_kernels(start.network.params, arrays),
            start.network.batch_stats,
        )
    )
    ns = build_same_stack_pqn_reference(case)
    donor = ns["create_agent"](jax.random.key(INIT_SEED))
    donor = donor.replace(params=with_recurrent_kernels(donor.params, arrays))
    rng = jax.random.key(UPDATE_SEED + 100)
    window = window_arrays(UPDATE_SEED, case["window"] + case["steps"], case["envs"])
    worst = 0.0
    for block in range(case["blocks"]):
        prefix = f"update/block{block}"
        full = pqn.PQNBatch(
            **{key: jnp.asarray(value) for key, value in bg_layout(window).items()}
        )

        def step(
            train: pqn.PQNTrainState, chosen: Array, batch: pqn.PQNBatch = full
        ) -> tuple[pqn.PQNTrainState, pqn.PQNMetrics]:
            return pqn.update_pqn(
                train,
                _selected(batch, chosen),
                pqn=config,
                planned_learning_blocks=planned,
            )

        state, metrics = pqn_learner.run_epochs(
            state,
            jnp.asarray(arrays[f"{prefix}/permutations"]),
            step,
            num_minibatches=case["minibatches"],
        )
        donor, rng, donor_loss = donor_learn(
            ns, donor, to_transition(ns, window), rng, case["epochs"]
        )
        assert bool(jnp.all(metrics.performed)) and bool(jnp.all(metrics.finite))
        assert bool(jnp.all(metrics.grad_norm > config.max_grad_norm))
        losses = np.asarray(metrics.loss)
        np.testing.assert_allclose(
            losses, arrays[f"{prefix}/loss"].reshape(losses.shape), rtol=_TOLERANCE
        )
        np.testing.assert_allclose(
            losses, np.asarray(donor_loss).reshape(losses.shape), rtol=_TOLERANCE
        )
        worst = max(
            worst,
            _fingerprints(state.network.params, arrays, f"{prefix}/params"),
            _fingerprints(state.network.batch_stats, arrays, f"{prefix}/batch_stats"),
            _fingerprints(state.opt_state, arrays, f"{prefix}/opt_state"),
            _compare(state.network.params, donor.params),
            _compare(state.network.batch_stats, donor.batch_stats),
            _compare(state.opt_state, donor.opt_state),
        )
        window = next_window(
            window,
            window_arrays(UPDATE_SEED + block + 1, case["steps"], case["envs"]),
            case["steps"],
        )
    steps = case["blocks"] * case["epochs"] * case["minibatches"]
    assert int(state.optimizer_steps) == steps == 8
    for leaf in jax.tree.leaves(state.opt_state):
        if jnp.issubdtype(leaf.dtype, jnp.integer):
            assert int(leaf) == steps
    record_property("maximum_absolute_error", worst)
