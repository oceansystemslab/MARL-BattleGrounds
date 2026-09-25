"""Check recurrent QMIX settings, actor rights, exploration and masked updates.

Contracts checked here, all on CPU and on real permitted inputs where an actor
acts. QMIXConfig and the batch-size guard reject invalid or overflowing
settings. The exploration clock starts at 1, falls linearly, is exactly
``eps_min`` from the decay end onward, stays constant when ``eps_min`` is 1,
saturates without int32 overflow when rounds times games exceeds int32, and
QMIXExploration sets epsilon while keeping shapes. The team task reward
averages configured slots only. Greedy choice is the first legal maximum for
every one of the 198 categories, for exact ties and for the case where every
legal score is ``finfo(float32).min``. The actor System returns legal
world-frame actions from both spawn ends, is greedy and key-independent at
epsilon 0, keeps inactive and dead slots neutral, breaks ties in the network
frame, keeps invalid lanes' memory, reads only each actor's own view and
memory (with a positive control), matches one lane alone to its batch row with
per-lane epsilon, and reuses one compilation when weights or epsilon change.
System identity records scale, frame and epsilon. Initialization rejects
non-Threefry and batched keys. The mixer never decreases when a utility rises.
The sampled update ignores the contents of invalid rows and inactive slots,
leaves the state unchanged when no TD pair is eligible, keeps finite
derivatives with sparse masks, and scans only over time. Parameter counts and
float32 bytes match the planned sizes (1,833,414 Q and 191,185 mixer values,
with 5,165 actor and 920 training-state features).
These checks prove software contracts, not learned skill or GPU cost.
"""

# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
import math
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config
from tests.training_learner_helpers import equal, finite

from marl_battlegrounds.baselines import qmix
from marl_battlegrounds.baselines.actions import (
    categorical_action_mask,
    encode_actions,
    mirror_action_indices,
)
from marl_battlegrounds.baselines.inputs import encode_actor_inputs, spawn_frame_flag
from marl_battlegrounds.core.types import AGENT_FEATURE_X
from marl_battlegrounds.environment import make
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.policies.input import mirror_team_view
from marl_battlegrounds.tasks import balanced_spawn_configs

type Tree = Any
type RecordProperty = Callable[[str, object], None]
_SMALL = qmix.QMIXConfig(
    sample_batch_size=3, sample_sequence_length=5, min_buffer_size=5
)


@pytest.fixture(scope="module")
def params() -> Tree:
    return qmix.initialize_qmix(jax.random.key(19048001)).online_q


@pytest.fixture(scope="module")
def actor_inputs() -> SystemInput:
    config = evaluation_env_config(team_sizes=(2, 2), max_steps=8)
    env = make(
        "tdm",
        env_config=balanced_spawn_configs(config, num_envs=2),
        num_envs=2,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(19048002))
    return env.policy_inputs(observations, state)


def _apply(
    system: System, memory: Array, inputs: SystemInput, seed: int
) -> SystemOutput:
    keys = jax.random.split(jax.random.key(seed), inputs.valid.shape[0])
    return cast(SystemOutput, system.apply(system.variables, memory, inputs, keys))


def _memory(system: System, inputs: SystemInput) -> Array:
    assert system.init is not None
    keys = jax.random.split(jax.random.key(0), inputs.valid.shape[0])
    return cast(Array, system.init(system.variables, inputs, keys))


def _lane(inputs: SystemInput, lane: int) -> SystemInput:
    def rows(value: Array) -> Array:
        return value[lane : lane + 1]

    return jax.tree.map(rows, inputs)


def _zero(value: Array) -> Array:
    return jnp.zeros_like(value)


def _scaled(value: Array) -> Array:
    return value * 0.9


def _leading(value: Array) -> Array:
    return value[None]


def _doubled(value: Array) -> Array:
    return value * 2.0


def _batch(
    key: Array, sequences: int = 3, rows: int = 5, width: int = 7
) -> qmix.QMIXBatch:
    keys = jax.random.split(key, 5)
    mask = jax.random.uniform(keys[1], (sequences, rows, 5, 198)) > 0.4
    mask = mask.at[..., 0].set(True)
    logits = jnp.where(mask, 0.0, -jnp.inf)
    actions = jax.random.categorical(keys[2], logits).astype(jnp.int32)
    starts = jnp.zeros((sequences, rows), bool).at[1, 2].set(True)
    return qmix.QMIXBatch(
        jax.random.normal(keys[0], (sequences, rows, 5, width)),
        mask,
        actions,
        jax.random.normal(keys[3], (sequences, rows)) * 0.3,
        starts,
        jnp.zeros((sequences, rows), bool).at[1, 1].set(True),
        jnp.ones((sequences, rows), bool),
        jnp.ones((sequences, rows, 5), bool),
        jax.random.normal(keys[4], (sequences, rows, 11)),
    )


def _state(batch: qmix.QMIXBatch, seed: int = 19048003) -> qmix.QMIXTrainState:
    key = jax.random.key(seed)
    width = batch.actor_features.shape[-1]
    q = qmix.RecurrentQNetwork().init(
        key,
        jnp.zeros((1, 5, qmix.QMIX_HIDDEN_SIZE)),
        jnp.zeros((1, 1, 5, width)),
        jnp.zeros((1, 1, 5), bool),
        jnp.ones((1, 1, 5), bool),
    )
    mixer = qmix.QMixingNetwork().init(
        key,
        jnp.zeros((1, 1, 5)),
        jnp.zeros((1, 1, batch.training_state.shape[-1])),
    )
    target_q = jax.tree.map(_scaled, q)
    return qmix.QMIXTrainState(
        q,
        target_q,
        mixer,
        mixer,
        qmix._optimizer(_SMALL).init((q, mixer)),
        jnp.int32(3),
    )


_compiled_update = jax.jit(qmix.update_qmix, static_argnames="config")


def _update(
    state: qmix.QMIXTrainState, batch: qmix.QMIXBatch, *, config: qmix.QMIXConfig
) -> tuple[qmix.QMIXTrainState, qmix.QMIXMetrics]:
    return cast(
        tuple[qmix.QMIXTrainState, qmix.QMIXMetrics],
        _compiled_update(state, batch, config=config),
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"rollout_length": True},
        {"rollout_length": 0},
        {"buffer_size": 8.0},
        {"epochs": -1},
        {"sample_sequence_length": 1, "min_buffer_size": 1},
        {"min_buffer_size": 19},
        {"buffer_size": 31},
        {"rollout_length": 1001},
        {"q_lr": True},
        {"q_lr": 0.0},
        {"q_lr": float("nan")},
        {"tau": 1.5},
        {"gamma": -0.1},
        {"eps_min": float("inf")},
        {"eps_decay": 2**31},
        {"hard_update": 1},
        {"spawn_frame": "right"},
        {"input_scale": 0.0},
        {"sample_batch_size": 2**30},
        {"sample_batch_size": 2**24, "epochs": 2**7},
    ),
)
def test_config_rejects_invalid_or_overflowing_settings(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        qmix.QMIXConfig(**changes)  # pyright: ignore[reportArgumentType]


def test_collected_block_guard_rejects_int32_overflow() -> None:
    qmix.validate_qmix_batch_size(1024, qmix.DEFAULT_QMIX_CONFIG)
    with pytest.raises(ValueError, match="overflow"):
        qmix.validate_qmix_batch_size(
            2**28, qmix.QMIXConfig(rollout_length=8, buffer_size=1000)
        )
    for bad in (0, True, 2.0):
        with pytest.raises(ValueError):
            qmix.validate_qmix_batch_size(bad, qmix.DEFAULT_QMIX_CONFIG)  # pyright: ignore[reportArgumentType]


def test_exploration_clock_boundaries_and_saturation() -> None:
    compiled = cast(
        Callable[[Array, int, float, int], Array],
        jax.jit(qmix.epsilon_at, static_argnums=(1, 2, 3)),
    )
    cases = (
        (0, 4, 0.05, 100, 1.0),
        (24, 4, 0.05, 100, None),
        (25, 4, 0.05, 100, 0.05),
        (26, 4, 0.05, 100, 0.05),
        (7, 3, 0.05, 20, 0.05),
        (5, 4, 1.0, 100, 1.0),
        (2**31 - 1, 1024, 0.05, 100000, 0.05),
    )
    for rounds, lanes, eps_min, decay, exact in cases:
        host = qmix.epsilon_reference(rounds, lanes, eps_min, decay)
        for value in (
            qmix.epsilon_at(jnp.int32(rounds), lanes, eps_min, decay),
            compiled(jnp.int32(rounds), lanes, eps_min, decay),
        ):
            assert value.dtype == jnp.float32 and value.shape == ()
            assert abs(float(value) - host) <= 2e-6
            if exact is not None:
                assert float(value) == np.float32(exact) and host == exact
    # One round before the end is still above the plateau.
    assert float(qmix.epsilon_at(jnp.int32(24), 4, 0.05, 100)) > 0.05
    with pytest.raises(ValueError, match="overflow"):
        qmix.epsilon_at(jnp.int32(0), 1024, 0.05, 2**31 - 1)
    with pytest.raises(ValueError, match="overflow"):
        qmix.QMIXExploration(0.05, 2**31 - 1, 2)


def test_exploration_hook_sets_epsilon_and_keeps_layout(params: Tree) -> None:
    hook = qmix.QMIXExploration(eps_min=0.05, eps_decay=100, num_envs=4)
    stored = qmix.QMIXActorVariables(params, jnp.float32(0.7))
    updated = hook(stored, jnp.int32(10))
    assert updated.params is params
    np.testing.assert_allclose(float(updated.epsilon), 1 - 0.4 * 0.95, rtol=1e-6)
    shape = jax.eval_shape(hook, stored, jnp.int32(0))
    assert jax.tree.structure(shape) == jax.tree.structure(stored)
    for left, right in zip(
        jax.tree.leaves(shape), jax.tree.leaves(stored), strict=True
    ):
        assert left.shape == right.shape and left.dtype == right.dtype
    assert hook.identity == {
        "kind": "qmix_epsilon",
        "version": 1,
        "eps_min": 0.05,
        "eps_decay": 100,
        "num_envs": 4,
    }
    with pytest.raises(ValueError):
        qmix.QMIXExploration(eps_min=True, eps_decay=100, num_envs=4)  # pyright: ignore[reportArgumentType]


def test_team_task_reward_averages_configured_slots_only() -> None:
    rewards = jnp.asarray(
        [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 9.0, 9.0, 9.0, 9.0], [2.0, 4.0, 0, 0, 0]]
    )
    active = jnp.asarray([[True] * 5, [True, False, False, False, False], [False] * 5])
    np.testing.assert_array_equal(
        qmix.team_task_reward(rewards, active), [1.0, 1.0, 0.0]
    )
    poisoned = rewards.at[1, 1:].set(jnp.nan)
    np.testing.assert_array_equal(qmix.team_task_reward(poisoned, active)[1], 1.0)


def test_greedy_choice_is_the_first_legal_maximum() -> None:
    eye = jnp.eye(198, dtype=bool)
    scores = jnp.broadcast_to(jnp.arange(198, dtype=jnp.float32)[::-1], (198, 198))
    np.testing.assert_array_equal(qmix.greedy_actions(scores, eye), jnp.arange(198))
    lifted = jnp.where(eye, 5.0, 0.0)
    np.testing.assert_array_equal(
        qmix.greedy_actions(lifted, jnp.ones((198, 198), bool)), jnp.arange(198)
    )
    tie = jnp.zeros((198,)).at[jnp.asarray([66, 88, 150])].set(2.0)
    all_legal = jnp.ones((198,), bool)
    assert int(qmix.greedy_actions(tie, all_legal)) == 66
    assert int(qmix.greedy_actions(tie, all_legal.at[66].set(False))) == 88
    # The donor's finfo.min masking would pick illegal index 0 here.
    edge_scores = jnp.full((198,), float(np.finfo(np.float32).min), jnp.float32)
    only_last = jnp.zeros((198,), bool).at[197].set(True)
    assert int(qmix.greedy_actions(edge_scores, only_last)) == 197
    for rate in (0.0, 0.25, 1.0):
        probabilities = qmix.action_probabilities(edge_scores, only_last, rate)
        np.testing.assert_array_equal(probabilities, only_last.astype(jnp.float32))
    mask = jax.random.uniform(jax.random.key(3), (4, 198)) > 0.5
    mask = mask.at[:, 0].set(True)
    probabilities = qmix.action_probabilities(
        jax.random.normal(jax.random.key(4), (4, 198)), mask, 0.3
    )
    np.testing.assert_allclose(probabilities.sum(-1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(jnp.where(mask, 0.0, probabilities), 0.0)


@pytest.mark.parametrize("frame", ("world", "left"))
def test_system_actions_are_legal_and_greedy_at_zero_epsilon(
    params: Tree, actor_inputs: SystemInput, frame: str
) -> None:
    system = qmix.make_qmix_system(params, input_scale=0.01, spawn_frame=frame)
    memory = _memory(system, actor_inputs)
    assert memory.shape == (2, 5, 256) and memory.dtype == jnp.float32
    np.testing.assert_array_equal(memory, 0)
    first = _apply(system, memory, actor_inputs, 1)
    second = _apply(system, memory, actor_inputs, 2)
    equal(first, second)
    assert first.learning_outputs == ()
    indices = encode_actions(first.actions)
    legal = categorical_action_mask(actor_inputs.action_mask)
    assert bool(jnp.all(jnp.take_along_axis(legal, indices[..., None], axis=-1)))
    np.testing.assert_array_equal(indices[:, 2:], 0)
    inputs, mask = actor_inputs.actors, actor_inputs.action_mask
    flag = jnp.zeros((2, 5), bool)
    if frame == "left":
        flag = spawn_frame_flag(inputs, frame)
        assert bool(flag.any()) and not bool(flag.all())
        inputs, mask = mirror_team_view(inputs, mask, flag)
    starts = jnp.broadcast_to(actor_inputs.episode_start[:, None], (2, 5))
    _, values = cast(
        tuple[Array, Array],
        qmix.RecurrentQNetwork(input_scale=0.01).apply(
            params,
            memory,
            encode_actor_inputs(inputs)[None],
            starts[None],
            jnp.ones((1, 2, 5), bool),
        ),
    )
    greedy = qmix.greedy_actions(values[0], categorical_action_mask(mask))
    np.testing.assert_array_equal(indices, mirror_action_indices(greedy, flag))


def test_system_explores_uniformly_over_legal_actions(
    params: Tree, actor_inputs: SystemInput
) -> None:
    system = qmix.make_qmix_system(params, epsilon=1.0, input_scale=0.01)
    memory = _memory(system, actor_inputs)
    legal = categorical_action_mask(actor_inputs.action_mask)
    seen: set[int] = set()
    for seed in range(12):
        indices = encode_actions(_apply(system, memory, actor_inputs, seed).actions)
        assert bool(jnp.all(jnp.take_along_axis(legal, indices[..., None], axis=-1)))
        np.testing.assert_array_equal(indices[:, 2:], 0)
        seen.add(int(indices[0, 0]))
    assert len(seen) > 6
    half = qmix.make_qmix_system(params, epsilon=0.5, input_scale=0.01)
    indices = encode_actions(_apply(half, memory, actor_inputs, 5).actions)
    assert bool(jnp.all(jnp.take_along_axis(legal, indices[..., None], axis=-1)))


@pytest.mark.parametrize("frame", ("world", "left"))
def test_exact_ties_break_in_the_network_frame(
    params: Tree, actor_inputs: SystemInput, frame: str
) -> None:
    tied = jax.tree.map(_zero, params)
    # Moves 3 and 4 mirror each other; both carry the neutral combat choice.
    tied["params"]["Dense_0"]["bias"] = jnp.zeros(198).at[66].set(1.0).at[88].set(1.0)
    system = qmix.make_qmix_system(tied, spawn_frame=frame)
    output = _apply(system, _memory(system, actor_inputs), actor_inputs, 3)
    indices = np.asarray(encode_actions(output.actions))
    inputs, mask = actor_inputs.actors, actor_inputs.action_mask
    flag = np.zeros((2, 5), bool)
    if frame == "left":
        flag = np.asarray(spawn_frame_flag(inputs, frame))
        inputs, mask = mirror_team_view(inputs, mask, jnp.asarray(flag))
    network_legal = np.asarray(categorical_action_mask(mask))
    checked = 0
    for lane in range(2):
        for slot in range(2):
            legal = network_legal[lane, slot]
            choice = next((k for k in (66, 88) if legal[k]), int(np.argmax(legal)))
            if flag[lane, slot] and choice in (66, 88):
                choice = 154 - choice
            assert indices[lane, slot] == choice
            checked += int(legal[66] or legal[88])
    assert checked > 0


def test_invalid_lanes_keep_memory_and_death_keeps_it_too(
    params: Tree, actor_inputs: SystemInput
) -> None:
    system = qmix.make_qmix_system(params, input_scale=0.01)
    memory = jnp.full((2, 5, 256), 0.25, jnp.float32)
    carried = actor_inputs._replace(
        episode_start=jnp.zeros((2,), bool), valid=jnp.asarray([True, False])
    )
    output = _apply(system, memory, carried, 4)
    next_memory = cast(Array, output.next_memory)
    np.testing.assert_array_equal(next_memory[1], memory[1])
    assert not np.array_equal(next_memory[0], memory[0])
    restarted = _apply(
        system, memory, carried._replace(episode_start=jnp.ones(2, bool)), 4
    )
    fresh = _apply(system, jnp.zeros_like(memory), carried, 4)
    equal(cast(Array, restarted.next_memory)[0], cast(Array, fresh.next_memory)[0])


def test_actor_reads_only_its_own_view_and_memory(
    params: Tree, actor_inputs: SystemInput
) -> None:
    system = qmix.make_qmix_system(params, input_scale=0.01)
    inputs = actor_inputs._replace(episode_start=jnp.zeros((2,), bool))
    memory = _memory(system, inputs)
    first = _apply(system, memory, inputs, 6)
    own = inputs.actors.observation

    def moved(slot: int) -> SystemInput:
        return inputs._replace(
            actors=inputs.actors._replace(
                observation=own._replace(
                    self_features=own.self_features.at[:, slot, AGENT_FEATURE_X].add(17)
                )
            )
        )

    second = _apply(system, memory.at[:, 1].set(0.3), moved(1), 6)
    for before, after in zip(first.actions, second.actions, strict=True):
        np.testing.assert_array_equal(before[:, 0], after[:, 0])
    np.testing.assert_array_equal(
        cast(Array, first.next_memory)[:, 0], cast(Array, second.next_memory)[:, 0]
    )
    # Positive control: actor 0's own view does change actor 0's memory.
    control = _apply(system, memory, moved(0), 6)
    assert not np.array_equal(
        cast(Array, first.next_memory)[:, 0], cast(Array, control.next_memory)[:, 0]
    )
    assert set(system.variables._fields) == {"params", "epsilon"}


def test_one_lane_matches_its_batch_row_with_per_lane_epsilon(
    params: Tree, actor_inputs: SystemInput
) -> None:
    system = qmix.make_qmix_system(params, input_scale=0.01)
    memory = jnp.full((2, 5, 256), 0.1, jnp.float32).at[1].set(-0.2)
    inputs = actor_inputs._replace(episode_start=jnp.zeros((2,), bool))
    keys = jax.random.split(jax.random.key(7), 2)
    rates = jnp.asarray([0.0, 0.6], jnp.float32)

    def lane(rate: Array, carry: Array, row: SystemInput, key: Array) -> SystemOutput:
        variables = qmix.QMIXActorVariables(params, rate)
        return cast(
            SystemOutput,
            system.apply(
                variables,
                carry[None],
                jax.tree.map(_leading, row),
                key[None],
            ),
        )

    together = jax.vmap(lane)(rates, memory, inputs, keys)
    for index in range(2):
        alone = cast(
            SystemOutput,
            system.apply(
                qmix.QMIXActorVariables(params, rates[index]),
                memory[index : index + 1],
                _lane(inputs, index),
                keys[index : index + 1],
            ),
        )
        for batched, single in zip(together.actions, alone.actions, strict=True):
            np.testing.assert_array_equal(batched[index], single)
        np.testing.assert_allclose(
            cast(Array, together.next_memory)[index],
            cast(Array, alone.next_memory),
            atol=2e-6,
            rtol=2e-6,
        )


def test_changing_weights_and_epsilon_reuse_one_compilation(
    params: Tree, actor_inputs: SystemInput
) -> None:
    system = qmix.make_qmix_system(params, input_scale=0.01)
    memory = _memory(system, actor_inputs)
    traces: list[int] = []

    def apply(variables: qmix.QMIXActorVariables, keys: Array) -> SystemOutput:
        traces.append(1)
        return cast(SystemOutput, system.apply(variables, memory, actor_inputs, keys))

    compiled = cast(
        Callable[[qmix.QMIXActorVariables, Array], SystemOutput], jax.jit(apply)
    )
    keys = jax.random.split(jax.random.key(8), 2)
    greedy = compiled(system.variables, keys)
    changed = qmix.QMIXActorVariables(jax.tree.map(_doubled, params), jnp.float32(1.0))
    explored = compiled(changed, keys)
    assert traces == [1]
    assert not np.array_equal(
        cast(Array, greedy.next_memory), cast(Array, explored.next_memory)
    )


def test_system_identity_records_scale_frame_and_epsilon(params: Tree) -> None:
    identities = {
        normalize_system_registration(
            qmix.make_qmix_system(
                params, epsilon=epsilon, input_scale=scale, spawn_frame=frame
            ),
            phase="evaluation",
            frozen=True,
        )[0]
        for scale, frame, epsilon in (
            (1.0, "left", 0.0),
            (0.01, "left", 0.0),
            (1.0, "world", 0.0),
            (1.0, "left", 0.5),
        )
    }
    assert len(identities) == 4
    assert qmix._q_actor_apply(1.0, "left") is qmix._q_actor_apply(1.0, "left")
    for bad in ({"epsilon": True}, {"epsilon": 1.5}, {"epsilon": float("nan")}):
        with pytest.raises(ValueError):
            qmix.make_qmix_system(params, **bad)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError):
        qmix.make_qmix_system(params, spawn_frame="right")


def test_initialization_rejects_other_key_kinds() -> None:
    with pytest.raises(ValueError, match="Threefry"):
        qmix.initialize_qmix(jax.random.key(1, impl="rbg"))
    with pytest.raises(ValueError, match="one key"):
        qmix.initialize_qmix(jax.random.split(jax.random.key(1), 2))
    typed = qmix.initialize_qmix(jax.random.key(1))
    legacy = qmix.initialize_qmix(jax.random.key_data(jax.random.key(1)))
    equal(legacy, typed)


def test_mixer_never_decreases_when_a_utility_rises() -> None:
    key = jax.random.key(19048004)
    mixer = qmix.QMixingNetwork()
    variables = mixer.init(key, jnp.zeros((1, 5)), jnp.zeros((1, 920)))
    utilities = jax.random.normal(jax.random.key(5), (64, 5)) * 3
    states = jax.random.normal(jax.random.key(6), (64, 920))

    def total(values: Array) -> Array:
        return jnp.sum(cast(Array, mixer.apply(variables, values, states)))

    gradient = cast(Array, jax.grad(total)(utilities))
    assert bool(jnp.all(gradient >= 0))
    raised = cast(Array, mixer.apply(variables, utilities.at[:, 2].add(1.0), states))
    base = cast(Array, mixer.apply(variables, utilities, states))
    assert bool(jnp.all(raised >= base))


def test_update_ignores_invalid_rows_and_inactive_slots() -> None:
    batch = _batch(jax.random.key(19048005))
    state = _state(batch)
    reference, metrics = _update(state, batch, config=_SMALL)
    assert bool(metrics.finite) and int(metrics.used_td_pairs) == 12
    # Sequence 2 loses its last row: one fewer pair, whatever that row holds.
    valid = batch.valid.at[2, 4].set(False)
    garbage = batch._replace(
        valid=valid,
        actor_features=batch.actor_features.at[2, 4].set(jnp.nan),
        action_mask=batch.action_mask.at[2, 4].set(False),
        actions=batch.actions.at[2, 4].set(197),
        rewards=batch.rewards.at[2, 4].set(jnp.nan),
        episode_start=batch.episode_start.at[2, 4].set(True),
        ended=batch.ended.at[2, 4].set(True),
        training_state=batch.training_state.at[2, 4].set(jnp.inf),
    )
    cleaned = batch._replace(valid=valid)
    first, first_metrics = _update(state, garbage, config=_SMALL)
    second, second_metrics = _update(state, cleaned, config=_SMALL)
    equal((first, first_metrics), (second, second_metrics))
    assert int(first_metrics.used_td_pairs) == 11
    assert int(first_metrics.used_agent_utilities) == 55
    assert bool(first_metrics.finite)
    assert not np.array_equal(first_metrics.loss, metrics.loss)
    # Slot 4 is not configured: its finite padding and actions change nothing.
    active = batch.active.at[..., 4].set(False)
    padded = batch._replace(active=active)
    moved = padded._replace(
        actor_features=padded.actor_features.at[..., 4, :].set(50.0),
        actions=padded.actions.at[..., 4].set(0),
    )
    third, third_metrics = _update(state, padded, config=_SMALL)
    fourth, fourth_metrics = _update(state, moved, config=_SMALL)
    equal((third, third_metrics), (fourth, fourth_metrics))
    assert int(third_metrics.used_agent_utilities) == 48
    del reference


def test_no_eligible_pair_leaves_the_state_unchanged() -> None:
    batch = _batch(jax.random.key(19048006))
    state = _state(batch)
    alternate = jnp.tile(jnp.asarray([True, False, True, False, True]), (3, 1))
    candidate, metrics = _update(state, batch._replace(valid=alternate), config=_SMALL)
    equal(candidate, state)
    assert int(metrics.used_td_pairs) == 0 and int(metrics.sampled_sequences) == 0
    assert float(metrics.loss) == 0.0 and bool(metrics.finite)


def test_sparse_masks_keep_finite_derivatives_and_time_scans() -> None:
    batch = _batch(jax.random.key(19048007))
    neutral = jnp.zeros((198,), bool).at[0].set(True)
    sparse = batch._replace(
        action_mask=batch.action_mask.at[:, ::2].set(neutral),
        actions=batch.actions.at[:, ::2].set(0),
    )
    candidate, metrics = _update(_state(sparse), sparse, config=_SMALL)
    assert bool(metrics.finite)
    finite(candidate)

    def small_update(
        state: qmix.QMIXTrainState, sample: qmix.QMIXBatch
    ) -> tuple[qmix.QMIXTrainState, qmix.QMIXMetrics]:
        return qmix.update_qmix(state, sample, config=_SMALL)

    jaxpr = jax.make_jaxpr(small_update)(_state(sparse), sparse)
    lengths: list[int] = []

    def visit(expression: Tree) -> None:
        for equation in expression.eqns:
            if equation.primitive.name == "scan":
                lengths.append(int(equation.params["length"]))
            for value in equation.params.values():
                inner = getattr(value, "jaxpr", None)
                if inner is not None:
                    visit(getattr(inner, "jaxpr", inner))

    visit(jaxpr.jaxpr)
    assert lengths and set(lengths) == {5}


def test_parameter_counts_and_bytes_match_the_planned_sizes(
    record_property: RecordProperty,
) -> None:
    state = jax.eval_shape(qmix.initialize_qmix, jax.random.key(0))

    def count(tree: Tree) -> int:
        leaves = cast(list[jax.ShapeDtypeStruct], jax.tree.leaves(tree))
        return sum(math.prod(leaf.shape) for leaf in leaves)

    q, mixer = count(state.online_q), count(state.online_mixer)
    assert (q, mixer) == (1_833_414, 191_185)
    assert 4 * (q + mixer) == 8_098_396
    # Adam keeps two float32 moments per parameter and one int32 step count.
    assert count(state.opt_state) == 2 * (q + mixer) + 1
    assert count(qmix.qmix_actor_template()) == q
    record_property("q_parameters", q)
    record_property("mixer_parameters", mixer)
