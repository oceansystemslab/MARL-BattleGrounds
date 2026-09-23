"""Compare recurrent QMIX with independent pinned Mava calculations.

The saved arrays come from unchanged donor bodies (Q-network, mixer,
epsilon-greedy distribution, action selection, sample preparation, loss and
``update_q``) on the donor's locked CPU stack. The same bodies also run on the
current stack as a second reference. BG production functions never generate
either oracle. Inputs use four sequences of six rows, five agents, the donor's
256-wide networks, an interior, a first-row and a last-row episode start, and a
neutral-only actor row; every row and slot is valid and active, and the frame
is the world frame. BG padding, frame and reward adaptations have separate
tests in ``test_baseline_qmix.py``.

Contracts checked here: the settings defaults equal the donor YAML (the
unused ``max_grad_norm`` and ``add_agent_id`` are not copied); initialization
from one key equals the current-stack donor exactly for online and target
Q-networks, mixers and the chained Adam state, with historical layouts equal;
forward values and memory; epsilon-greedy probabilities and first-legal-maximum
greedy choices, including exact ties; the one deliberate departure, where every
legal score equals ``finfo(float32).min`` and BG chooses the legal action while
the donor chooses an illegal one; mixer values; Double-Q loss metrics and
gradients (captured with a zero-step optimizer); one Adam step, with its moments
checked before parameters; hard target copies at pre-step counts 0 and 200 but
not 1 or 199; the soft target rule; two consecutive updates; and the donor's
epsilon clock, with BG's exact ``eps_min`` plateau at the decay end.

Float comparisons use the established 2e-6 absolute and relative tolerances;
integer values, parameter paths, shapes and dtypes are exact. These checks
prove numerical agreement, not learning performance, GPU cost or new
information rights.
"""

# Runtime donor types cannot be imported into the repository's type environment.
# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
from jax import Array
from numpy.typing import NDArray
from tests.baseline_donor_reference import (
    _store_tree,
    build_same_stack_qmix_reference,
    load_qmix_reference,
    qmix_gradient_capture,
    qmix_reference_networks,
    qmix_reference_update,
    reference_qmix_optimizer,
    reference_tree,
)

from marl_battlegrounds.baselines import qmix
from marl_battlegrounds.baselines.inputs import (
    ACTOR_FEATURE_SIZE,
    TRAINING_STATE_FEATURE_SIZE,
)

type Tree = Any
type RecordProperty = Callable[[str, object], None]


def _compare(actual: Tree, expected: Tree, *, exact: bool = False) -> float:
    actual_leaves: dict[str, NDArray[Any]] = {}
    expected_leaves: dict[str, NDArray[Any]] = {}
    _store_tree(actual_leaves, "value", actual, jax)
    _store_tree(expected_leaves, "value", expected, jax)
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
                value, target, atol=2e-6, rtol=2e-6, err_msg=name
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


def _stored(actual: Tree, arrays: dict[str, NDArray[Any]], prefix: str) -> float:
    actual_leaves: dict[str, NDArray[Any]] = {}
    _store_tree(actual_leaves, prefix, actual, jax)
    selected = {
        key: value
        for key, value in arrays.items()
        if key == prefix or key.startswith(prefix + "/")
    }
    return _compare(actual_leaves, selected)


def _asarray(value: NDArray[Any]) -> Array:
    return jnp.asarray(value)


def _zero(value: Array) -> Array:
    return jnp.zeros_like(value)


def _tree(arrays: dict[str, NDArray[Any]], prefix: str) -> Tree:
    return jax.tree.map(_asarray, reference_tree(arrays, prefix))


def _time_major(value: Array) -> Array:
    return jnp.swapaxes(value, 0, 1)


def _capture(config: qmix.QMIXConfig) -> Tree:
    del config

    def init(params: Tree) -> Tree:
        return params

    def update(updates: Tree, state: Tree, params: Tree = None) -> tuple[Tree, Tree]:
        del state, params
        return jax.tree.map(_zero, updates), updates

    return optax.GradientTransformation(init, update)


@dataclass
class Reference:
    metadata: dict[str, Any]
    arrays: dict[str, NDArray[Any]]
    inputs: dict[str, Array]
    namespace: dict[str, Any]
    q_net: Any
    mixer: Any
    optimizer: Any
    donor_update: Callable[..., Any]
    data: Any
    batch: qmix.QMIXBatch
    config: qmix.QMIXConfig
    state: qmix.QMIXTrainState


def _loss_values(loss: dict[str, Array]) -> tuple[Array, Array, Array]:
    return loss["q_loss"], loss["mean_q"], loss["mean_target"]


@pytest.fixture(scope="module")
def reference() -> Reference:
    metadata, arrays = load_qmix_reference()
    inputs = {
        key.removeprefix("input/"): jnp.asarray(value)
        for key, value in arrays.items()
        if key.startswith("input/")
    }
    settings = metadata["settings"]
    ns = build_same_stack_qmix_reference()
    ns["cfg"] = SimpleNamespace(
        system=SimpleNamespace(**settings["system"]),
        network=SimpleNamespace(**settings["network"]),
        arch=SimpleNamespace(**settings["arch"]),
    )
    q_net, mixer = qmix_reference_networks(
        ns, settings["network"], settings["system"]["qmix_embed_dim"]
    )
    optimizer = reference_qmix_optimizer(ns)
    starts = inputs["episode_start"]
    sequences, rows, agents = inputs["actions"].shape
    global_state = jnp.broadcast_to(
        inputs["training_state"][:, :, None],
        (sequences, rows, agents, inputs["training_state"].shape[-1]),
    )
    observation = ns["ObservationGlobalState"](
        inputs["actor_features"], inputs["action_mask"], global_state
    )
    data = ns["Transition"](
        observation, inputs["actions"], inputs["rewards"], starts, starts, observation
    )
    first = starts[..., 0]
    # The donor marks the next row as a start; BG marks the row that ended.
    ended = jnp.concatenate((first[:, 1:], jnp.zeros((sequences, 1), bool)), axis=1)
    batch = qmix.QMIXBatch(
        inputs["actor_features"],
        inputs["action_mask"],
        inputs["actions"],
        inputs["rewards"][..., 0],
        first,
        ended,
        jnp.ones((sequences, rows), bool),
        jnp.ones((sequences, rows, agents), bool),
        inputs["training_state"],
    )
    config = qmix.QMIXConfig(
        sample_batch_size=sequences,
        sample_sequence_length=rows,
        min_buffer_size=rows,
        spawn_frame="world",
    )
    online = _tree(arrays, "parameters/online_q")
    online_mixer = _tree(arrays, "parameters/online_mixer")
    state = qmix.QMIXTrainState(
        online,
        _tree(arrays, "parameters/target_q"),
        online_mixer,
        _tree(arrays, "parameters/target_mixer"),
        qmix._optimizer(config).init((online, online_mixer)),
        jnp.int32(0),
    )
    return Reference(
        metadata,
        arrays,
        inputs,
        ns,
        q_net,
        mixer,
        optimizer,
        qmix_reference_update(ns),
        data,
        batch,
        config,
        state,
    )


def _donor_params(reference: Reference) -> Tree:
    state = reference.state
    return reference.namespace["QMIXParams"](
        state.online_q, state.target_q, state.online_mixer, state.target_mixer
    )


def _update(
    reference: Reference, state: qmix.QMIXTrainState, config: qmix.QMIXConfig
) -> tuple[qmix.QMIXTrainState, qmix.QMIXMetrics]:
    return cast(
        tuple[qmix.QMIXTrainState, qmix.QMIXMetrics],
        jax.jit(qmix.update_qmix, static_argnames="config")(
            state, reference.batch, config=config
        ),
    )


def test_defaults_match_the_pinned_donor_config(reference: Reference) -> None:
    donor = reference.metadata["donor_defaults"]
    config = qmix.DEFAULT_QMIX_CONFIG
    for name in (
        "rollout_length",
        "buffer_size",
        "min_buffer_size",
        "sample_sequence_length",
        "sample_batch_size",
        "epochs",
        "q_lr",
        "hard_update",
        "update_period",
        "tau",
        "gamma",
        "eps_min",
        "eps_decay",
    ):
        assert getattr(config, name) == donor[name], name
    assert type(config.eps_decay) is int
    assert config.input_scale == 1.0 and config.spawn_frame == "left"
    # The donor names these settings, but its optimizer and inputs ignore them.
    assert donor["max_grad_norm"] == 10 and donor["add_agent_id"] is True
    assert not hasattr(config, "max_grad_norm")
    assert donor["qmix_embed_dim"] == qmix.QMIX_EMBED_SIZE
    network = reference.metadata["settings"]["network"]
    assert network["hidden_state_dim"] == qmix.QMIX_HIDDEN_SIZE
    for torso in ("pre_torso", "post_torso"):
        assert network["q_network"][torso]["layer_sizes"] == [qmix.QMIX_HIDDEN_SIZE]
        assert network["q_network"][torso]["activation"] == "relu"
        assert network["q_network"][torso]["use_layer_norm"] is False
    mixer = network["mixer_network"]
    assert mixer["hyper_hidden_dim"] == qmix.QMIX_HYPER_HIDDEN_SIZE
    assert mixer["norm_env_states"] is True
    params = (reference.state.online_q, reference.state.online_mixer)
    donor_state = reference.optimizer.init(params)
    ours = qmix._optimizer(reference.config).init(params)
    assert jax.tree.structure(donor_state) == jax.tree.structure(ours)
    _compare(ours, donor_state, exact=True)


def test_initialization_matches_current_stack_donor_exactly(
    reference: Reference,
) -> None:
    key = jax.random.PRNGKey(19048101)
    state = qmix.initialize_qmix(key)
    ns, agents = reference.namespace, qmix.TEAM_SLOTS
    init_obs = ns["ObservationGlobalState"](
        jnp.zeros((1, 1, agents, ACTOR_FEATURE_SIZE)),
        jnp.ones((1, 1, agents, 198), bool),
        jnp.zeros((1, 1, agents, TRAINING_STATE_FEATURE_SIZE)),
    )
    hidden = ns["ScannedRNN"].initialize_carry((32, agents), qmix.QMIX_HIDDEN_SIZE)
    donor_q = reference.q_net.init(key, hidden, (init_obs, jnp.zeros((1, 1, 1), bool)))
    donor_mixer = reference.mixer.init(
        key,
        jnp.zeros((128, 19, agents)),
        jnp.zeros((128, 19, TRAINING_STATE_FEATURE_SIZE)),
    )
    for actual in (state.online_q, state.target_q):
        _compare(actual, donor_q, exact=True)
    for actual in (state.online_mixer, state.target_mixer):
        _compare(actual, donor_mixer, exact=True)
    donor_optimizer = reference_qmix_optimizer(
        {
            **reference.namespace,
            "cfg": SimpleNamespace(system=SimpleNamespace(q_lr=3e-5)),
        }
    )
    _compare(state.opt_state, donor_optimizer.init((donor_q, donor_mixer)), exact=True)
    assert state.optimizer_steps.dtype == jnp.int32 and int(state.optimizer_steps) == 0
    template = qmix.qmix_actor_template()
    assert jax.tree.structure(template) == jax.tree.structure(state.online_q)
    for left, right in zip(
        jax.tree.leaves(template), jax.tree.leaves(state.online_q), strict=True
    ):
        assert left.shape == right.shape and left.dtype == right.dtype
    small = qmix.RecurrentQNetwork().init(
        key,
        jnp.zeros((1, agents, qmix.QMIX_HIDDEN_SIZE)),
        jnp.zeros((1, 1, agents, reference.metadata["shapes"]["actor_features"])),
        jnp.zeros((1, 1, agents), bool),
        jnp.ones((1, 1, agents), bool),
    )
    small_mixer = qmix.QMixingNetwork().init(
        key,
        jnp.zeros((1, 1, agents)),
        jnp.zeros((1, 1, reference.metadata["shapes"]["training_state"])),
    )
    for actual, prefix in (
        (small, "parameters/online_q"),
        (small_mixer, "parameters/online_mixer"),
    ):
        historical = _tree(reference.arrays, prefix)
        assert jax.tree.structure(actual) == jax.tree.structure(historical)
        for left, right in zip(
            jax.tree.leaves(actual), jax.tree.leaves(historical), strict=True
        ):
            assert left.shape == right.shape and left.dtype == right.dtype


def test_forward_and_memory_match_both_donor_references(
    reference: Reference, record_property: RecordProperty
) -> None:
    inputs, ns = reference.inputs, reference.namespace
    sequences, rows, agents = inputs["actions"].shape
    resets = jnp.broadcast_to(
        _time_major(inputs["episode_start"]), (rows, sequences, agents)
    )
    memory, values = cast(
        tuple[Array, Array],
        qmix.RecurrentQNetwork().apply(
            reference.state.online_q,
            inputs["carry"],
            _time_major(inputs["actor_features"]),
            resets,
            jnp.ones((rows, sequences, agents), bool),
        ),
    )
    errors = [
        _stored(values, reference.arrays, "expected/forward_q"),
        _stored(memory, reference.arrays, "expected/forward_carry"),
    ]
    observation = ns["switch_leading_axes"](reference.data.obs)
    source = reference.q_net.apply(
        reference.state.online_q,
        inputs["carry"],
        (observation, ns["switch_leading_axes"](inputs["episode_start"])),
        method="get_q_values",
    )
    errors.append(_compare((memory, values), source))
    record_property("maximum_absolute_error", max(errors))


def test_epsilon_greedy_probabilities_and_ties_match_the_donor(
    reference: Reference, record_property: RecordProperty
) -> None:
    arrays, inputs = reference.arrays, reference.inputs
    values = jnp.asarray(arrays["expected/forward_q"])
    mask = _time_major(inputs["action_mask"])
    rows, row_masks = inputs["q_rows"], inputs["mask_rows"]
    errors: list[float] = []
    for index, rate in enumerate(reference.metadata["rates"]):
        probabilities = qmix.action_probabilities(values, mask, rate)
        errors.append(
            _compare(
                probabilities, jnp.asarray(arrays["expected/probabilities"][index])
            )
        )
        distribution = reference.namespace["MaskedEpsGreedyDistribution"](
            values, rate, mask
        )
        errors.append(_compare(probabilities, distribution.probs_parameter()))
        # Rows 0-5 are ordinary finite scores, including exact ties.
        ordinary = qmix.action_probabilities(rows[:6], row_masks[:6], rate)
        errors.append(
            _compare(
                ordinary,
                jnp.asarray(arrays["expected/row_probabilities"][index][:6]),
            )
        )
    greedy = qmix.greedy_actions(values, mask)
    _compare(greedy, jnp.asarray(arrays["expected/modes"][0]), exact=True)
    row_greedy = qmix.greedy_actions(rows, row_masks)
    np.testing.assert_array_equal(row_greedy[:6], arrays["expected/row_modes"][0][:6])
    np.testing.assert_array_equal(np.asarray(row_greedy)[[1, 2, 5]], [5, 0, 3])
    # The donor's finfo.min masking picks illegal index 0; BG picks legal 197.
    assert int(arrays["expected/row_modes"][0][6]) == 0
    assert not bool(row_masks[6, 0])
    assert int(row_greedy[6]) == 197
    edge = qmix.action_probabilities(rows[6:], row_masks[6:], 0.37)
    np.testing.assert_array_equal(edge[0, 197], 1.0)
    record_property("maximum_absolute_error", max(errors))


def test_mixer_matches_both_donor_references(
    reference: Reference, record_property: RecordProperty
) -> None:
    inputs = reference.inputs
    states = inputs["training_state"][:, :-1]
    mixed = cast(
        Array,
        qmix.QMixingNetwork().apply(
            reference.state.online_mixer, inputs["agent_qs"], states
        ),
    )
    source = reference.mixer.apply(
        reference.state.online_mixer, inputs["agent_qs"], states
    )
    errors = [
        _compare(mixed, jnp.asarray(reference.arrays["expected/mixer"][..., 0])),
        _compare(mixed, source[..., 0]),
    ]
    record_property("maximum_absolute_error", max(errors))


def test_gradients_and_loss_match_both_donor_references(
    reference: Reference,
    monkeypatch: pytest.MonkeyPatch,
    record_property: RecordProperty,
) -> None:
    monkeypatch.setattr(qmix, "_optimizer", _capture)
    start = reference.state._replace(
        opt_state=(reference.state.online_q, reference.state.online_mixer),
        optimizer_steps=jnp.int32(1),
    )
    candidate, metrics = qmix.update_qmix(
        start, reference.batch, config=reference.config
    )
    gradients = candidate.opt_state
    errors = [
        _stored(gradients[0], reference.arrays, "gradients/q"),
        _stored(gradients[1], reference.arrays, "gradients/mixer"),
    ]
    stored = {
        name: jnp.asarray(reference.arrays[f"gradients/loss/{name}"])
        for name in ("q_loss", "mean_q", "mean_target")
    }
    errors.append(
        _compare(
            (metrics.loss, metrics.mean_q, metrics.mean_target), _loss_values(stored)
        )
    )
    reference.namespace["opt"] = qmix_gradient_capture(reference.namespace)
    _, donor_gradients, donor_loss = reference.donor_update(
        _donor_params(reference),
        start.opt_state,
        reference.data,
        jnp.int32(1),
    )
    reference.namespace["opt"] = reference.optimizer
    errors.append(_compare(gradients, donor_gradients))
    errors.append(
        _compare(
            (metrics.loss, metrics.mean_q, metrics.mean_target),
            _loss_values(donor_loss),
        )
    )
    sequences, rows, agents = reference.inputs["actions"].shape
    assert int(metrics.sampled_sequences) == sequences
    assert int(metrics.used_td_pairs) == sequences * (rows - 1)
    assert int(metrics.used_agent_utilities) == sequences * (rows - 1) * agents
    assert bool(metrics.finite)
    record_property("maximum_absolute_error", max(errors))


def test_one_adam_step_matches_both_donor_references(
    reference: Reference, record_property: RecordProperty
) -> None:
    candidate, metrics = _update(reference, reference.state, reference.config)
    adam = reference.optimizer.init(
        (reference.state.online_q, reference.state.online_mixer)
    )
    donor_params, donor_optimizer, donor_loss = reference.donor_update(
        _donor_params(reference), adam, reference.data, jnp.int32(0)
    )
    # Adam moments carry the gradients, so they are checked before parameters.
    errors = [
        _stored(candidate.opt_state, reference.arrays, "one_update/optimizer"),
        _compare(candidate.opt_state, donor_optimizer),
        _stored(candidate.online_q, reference.arrays, "one_update/online_q"),
        _stored(candidate.online_mixer, reference.arrays, "one_update/online_mixer"),
        _compare(
            (candidate.online_q, candidate.online_mixer),
            (donor_params.online, donor_params.mixer_online),
        ),
        _compare(
            (metrics.loss, metrics.mean_q, metrics.mean_target),
            _loss_values(donor_loss),
        ),
    ]
    assert int(candidate.optimizer_steps) == 1
    record_property("maximum_absolute_error", max(errors))


def test_hard_target_copies_follow_the_donor_pre_step_count(
    reference: Reference,
) -> None:
    copies = reference.metadata["hard_update_copies"]
    assert copies == {"0": True, "1": False, "199": False, "200": True}
    for count, copied in copies.items():
        start = reference.state._replace(optimizer_steps=jnp.int32(int(count)))
        candidate, _ = _update(reference, start, reference.config)
        assert int(candidate.optimizer_steps) == int(count) + 1
        if copied:
            _compare(candidate.target_q, candidate.online_q, exact=True)
            _compare(candidate.target_mixer, candidate.online_mixer, exact=True)
        else:
            _compare(candidate.target_q, reference.state.target_q, exact=True)
            _compare(candidate.target_mixer, reference.state.target_mixer, exact=True)


def test_soft_target_update_matches_both_donor_references(
    reference: Reference, record_property: RecordProperty
) -> None:
    config = qmix.QMIXConfig(
        sample_batch_size=reference.config.sample_batch_size,
        sample_sequence_length=reference.config.sample_sequence_length,
        min_buffer_size=reference.config.min_buffer_size,
        hard_update=False,
        spawn_frame="world",
    )
    candidate, _ = _update(reference, reference.state, config)
    adam = reference.optimizer.init(
        (reference.state.online_q, reference.state.online_mixer)
    )
    reference.namespace["cfg"].system.hard_update = False
    try:
        donor_params, _, _ = reference.donor_update(
            _donor_params(reference), adam, reference.data, jnp.int32(0)
        )
    finally:
        reference.namespace["cfg"].system.hard_update = True
    errors = [
        _stored(candidate.target_q, reference.arrays, "soft_update/target_q"),
        _stored(candidate.target_mixer, reference.arrays, "soft_update/target_mixer"),
        _compare(
            (candidate.target_q, candidate.target_mixer),
            (donor_params.target, donor_params.mixer_target),
        ),
    ]
    record_property("maximum_absolute_error", max(errors))


def test_two_updates_match_the_current_stack_donor(
    reference: Reference, record_property: RecordProperty
) -> None:
    first, _ = _update(reference, reference.state, reference.config)
    second, metrics = _update(reference, first, reference.config)
    adam = reference.optimizer.init(
        (reference.state.online_q, reference.state.online_mixer)
    )
    params, optimizer, _ = reference.donor_update(
        _donor_params(reference), adam, reference.data, jnp.int32(0)
    )
    params, optimizer, loss = reference.donor_update(
        params, optimizer, reference.data, jnp.int32(1)
    )
    errors = [
        _compare(second.opt_state, optimizer),
        _compare(
            (
                second.online_q,
                second.target_q,
                second.online_mixer,
                second.target_mixer,
            ),
            (params.online, params.target, params.mixer_online, params.mixer_target),
        ),
        _compare(
            (metrics.loss, metrics.mean_q, metrics.mean_target), _loss_values(loss)
        ),
        _stored(second.online_mixer, reference.arrays, "two_updates/online_mixer"),
    ]
    stored = {
        name: jnp.asarray(reference.arrays[f"two_updates/loss/{name}"])
        for name in ("q_loss", "mean_q", "mean_target")
    }
    errors.append(
        _compare(
            (metrics.loss, metrics.mean_q, metrics.mean_target), _loss_values(stored)
        )
    )
    assert int(second.optimizer_steps) == 2
    record_property("maximum_absolute_error", max(errors))


def test_epsilon_clock_matches_the_donor_selection_rule(
    reference: Reference, record_property: RecordProperty
) -> None:
    arrays, inputs = reference.arrays, reference.inputs
    settings = reference.metadata["settings"]
    lanes = settings["arch"]["num_envs"]
    eps_min, eps_decay = settings["system"]["eps_min"], settings["system"]["eps_decay"]
    rounds = [int(value) for value in inputs["epsilon_rounds"]]
    compiled = cast(
        Callable[[Array, int, float, int], Array],
        jax.jit(qmix.epsilon_at, static_argnums=(1, 2, 3)),
    )
    actual = jnp.stack(
        [compiled(jnp.int32(count), lanes, eps_min, int(eps_decay)) for count in rounds]
    )
    error = _compare(actual, jnp.asarray(arrays["expected/epsilon"]))
    for count, value in zip(rounds, np.asarray(actual), strict=True):
        host = qmix.epsilon_reference(count, lanes, eps_min, int(eps_decay))
        assert abs(float(value) - host) <= 2e-6
        if lanes * count >= eps_decay:
            # The donor's float32 subtraction can leave 0.05000001 at t=100000.
            assert float(value) == np.float32(eps_min) and host == eps_min
    np.testing.assert_array_equal(
        arrays["expected/epsilon_next_steps"], [lanes * (count + 1) for count in rounds]
    )
    memory, _ = cast(
        tuple[Array, Array],
        qmix.RecurrentQNetwork().apply(
            reference.state.online_q,
            inputs["carry"][:lanes],
            inputs["actor_features"][:lanes, :1].swapaxes(0, 1),
            jnp.broadcast_to(
                inputs["episode_start"][:lanes, :1].swapaxes(0, 1),
                (1, lanes, qmix.TEAM_SLOTS),
            ),
            jnp.ones((1, lanes, qmix.TEAM_SLOTS), bool),
        ),
    )
    record_property(
        "maximum_absolute_error",
        max(error, _stored(memory, arrays, "expected/select_hidden")),
    )
