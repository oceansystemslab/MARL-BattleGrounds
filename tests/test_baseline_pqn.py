"""Check recurrent PQN-VDN settings, actor rights, normalization and masked updates.

Contracts checked here, all on CPU and on real permitted inputs where an actor
acts. PQNConfig rejects Boolean, nonfinite, out-of-range and inconsistent
settings; the batch guard rejects minibatch counts that do not divide the
games and int32 overflow; the planned-block helper rejects budgets without a
learning block (R <= H + T) before any allocation and returns
ceil((R - W) / T). The exploration clock starts at eps_start (within float32
rounding), keeps a fractional decay span, is exactly eps_finish on its
plateau (also in compiled code) and stays constant when start equals
finish; the float64 host twins of the exploration and learning-rate
schedules agree with Optax. The actor
System returns legal world-frame actions from both spawn ends, is greedy and
key-independent at epsilon 0, explores uniformly over legal actions at
epsilon 1, draws 20,000 actions at epsilon 0.37 through its own key split
within five standard deviations of the epsilon-greedy probabilities (never
an illegal one), keeps inactive slots neutral, gives a dead active slot (alive
feature 0, neutral-only mask) the neutral action at epsilon 0 and 1 while its
memory keeps moving and is not reset (with a positive control that the full
mask picks another action), breaks exact ties in the network frame, returns
the memory it received as its learning output, resets memory only at episode
starts, keeps invalid lanes' memory, holds zero memory in inactive slots,
reads only each actor's own view and memory (with a positive control),
receives no team field and, in the left frame, gives mirror-twin games the
same network-frame actions and memory from either spawn end for both teams
(the world frame, as a positive control, does not), matches one lane alone to
its batch row, changes its values when only the statistics change, and
reuses one compilation when weights, statistics or epsilon change. Training-mode
normalization equals a float64 masked-moment reference and ignores NaN
poison in padding rows and inactive slots. The lambda-return targets match
hand-computed values for an ending, a horizon draw and an ordinary cutoff.
The minibatch update skips everything when no TD pair exists. Parameter
counts and float32 bytes match the planned sizes (4,596,512 parameters and
12,378 statistics, with the 5,165-feature input). These checks prove software
contracts, not learned skill or GPU cost.
"""

# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
import math
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config
from tests.test_baseline_inputs import (
    _two_lane_reset,  # pyright: ignore[reportPrivateUsage]
)
from tests.training_learner_helpers import equal

from marl_battlegrounds.baselines import pqn, qmix
from marl_battlegrounds.baselines.actions import (
    categorical_action_mask,
    encode_actions,
    mirror_action_indices,
)
from marl_battlegrounds.baselines.inputs import encode_actor_inputs, spawn_frame_flag
from marl_battlegrounds.core.types import AGENT_FEATURE_ALIVE, AGENT_FEATURE_X
from marl_battlegrounds.environment import make
from marl_battlegrounds.evaluation.policy_execution import (
    SystemInput,
    SystemOutput,
    system_inputs,
)
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.policies.input import mirror_team_view
from marl_battlegrounds.tasks import balanced_spawn_configs

type Tree = Any
_SMALL = pqn.PQNConfig(rollout_length=4, memory_window=2, epochs=1, num_minibatches=1)


@pytest.fixture(scope="module")
def network() -> pqn.PQNInferenceVariables:
    return pqn.initialize_pqn(
        jax.random.key(19049001), planned_learning_blocks=3
    ).network


@pytest.fixture(scope="module")
def environment() -> tuple[SystemInput, SystemInput]:
    config = evaluation_env_config(team_sizes=(2, 2), max_steps=8)
    env = make(
        "tdm",
        env_config=balanced_spawn_configs(config, num_envs=2),
        num_envs=2,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(19049002))
    team_a = env.policy_inputs(observations, state)
    team_b = env.policy_inputs(observations, state, team=1)
    return team_a, team_b


def _apply(system: Any, memory: Array, inputs: SystemInput, seed: int) -> SystemOutput:  # noqa: ANN401
    keys = jax.random.split(jax.random.key(seed), inputs.valid.shape[0])
    return cast(SystemOutput, system.apply(system.variables, memory, inputs, keys))


def _memory(system: Any, inputs: SystemInput) -> Array:  # noqa: ANN401
    keys = jax.random.split(jax.random.key(0), inputs.valid.shape[0])
    return cast(Array, system.init(system.variables, inputs, keys))


def _lane(inputs: SystemInput, lane: int) -> SystemInput:
    def rows(value: Array) -> Array:
        return value[lane : lane + 1]

    return jax.tree.map(rows, inputs)


def _zeros(value: Array) -> Array:
    return jnp.zeros_like(value)


def _doubled(value: Array) -> Array:
    return value * 2.0


def _shifted(value: Array) -> Array:
    return value + 0.5


def test_config_rejects_invalid_or_inconsistent_settings() -> None:
    assert pqn.DEFAULT_PQN_CONFIG.initial_rounds == 132
    for field, value in (
        ("rollout_length", True),
        ("rollout_length", 0),
        ("memory_window", 2**31),
        ("epochs", 1.0),
        ("num_minibatches", -1),
        ("q_lr", True),
        ("q_lr", 0.0),
        ("max_grad_norm", math.inf),
        ("lr_linear_decay", 1),
        ("gamma", 1.5),
        ("td_lambda", math.nan),
        ("eps_start", True),
        ("eps_finish", -0.1),
        ("eps_decay_fraction", 0.0),
        ("eps_decay_fraction", 1.5),
        ("input_scale", 0.0),
        ("spawn_frame", "right"),
    ):
        with pytest.raises(ValueError):
            pqn.PQNConfig(**cast(dict[str, Any], {field: value}))
    with pytest.raises(ValueError, match="memory_window"):
        pqn.PQNConfig(rollout_length=4, memory_window=5)
    with pytest.raises(ValueError, match="eps_finish"):
        pqn.PQNConfig(eps_start=0.2, eps_finish=0.3)
    assert pqn.PQNConfig(memory_window=128).initial_rounds == 256


def test_batch_and_budget_guards_reject_before_allocation() -> None:
    pqn.validate_pqn_batch_size(32, pqn.DEFAULT_PQN_CONFIG)
    with pytest.raises(ValueError, match="divide"):
        pqn.validate_pqn_batch_size(24, pqn.DEFAULT_PQN_CONFIG)
    with pytest.raises(ValueError, match="actor counts"):
        pqn.validate_pqn_batch_size(
            2**24, pqn.PQNConfig(rollout_length=128, num_minibatches=1)
        )
    with pytest.raises(ValueError, match="utilities"):
        pqn.validate_pqn_batch_size(
            2**22, pqn.PQNConfig(rollout_length=16, num_minibatches=1, epochs=8)
        )
    config = pqn.PQNConfig(
        rollout_length=4, memory_window=4, epochs=1, num_minibatches=2
    )
    assert pqn.pqn_planned_learning_blocks(20, config) == 3
    assert pqn.pqn_planned_learning_blocks(21, config) == 4
    assert pqn.pqn_planned_learning_blocks(9, config) == 1
    with pytest.raises(ValueError, match="at least"):
        pqn.pqn_planned_learning_blocks(8, config)
    with pytest.raises(ValueError):
        pqn.pqn_planned_learning_blocks(True, config)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="optimizer steps"):
        pqn.pqn_planned_learning_blocks(
            2**31 - 1, pqn.PQNConfig(rollout_length=1, memory_window=1)
        )
    assert pqn.pqn_planned_learning_blocks(
        10_000_000 // 32, pqn.DEFAULT_PQN_CONFIG
    ) == math.ceil((10_000_000 // 32 - 132) / 128)


def test_exploration_and_learning_rate_schedules_and_host_twins() -> None:
    config = pqn.DEFAULT_PQN_CONFIG
    start = pqn.epsilon_at(jnp.int32(0), 25, config)
    assert start.dtype == jnp.float32
    assert abs(float(start) - 1.0) <= 1e-7
    # A span of 2.5 blocks keeps its fraction: k=1 is 40% of the way down.
    np.testing.assert_allclose(
        float(pqn.epsilon_at(jnp.int32(1), 25, config)), 1.0 - 0.99 * 0.4, rtol=1e-6
    )
    for blocks in (3, 7, 10_000):
        assert float(pqn.epsilon_at(jnp.int32(blocks), 25, config)) == np.float32(0.01)
        assert pqn.pqn_epsilon_reference(blocks, 25, config) == 0.01

    # Compiled code may divide through the reciprocal of a fractional span.
    def plateau(blocks: Array) -> Array:
        return pqn.epsilon_at(blocks, 3, config)

    compiled = cast(Callable[[Array], Array], jax.jit(plateau))
    for blocks in (1, 2, 50):
        assert float(compiled(jnp.int32(blocks))) == np.float32(0.01)
    flat = pqn.PQNConfig(eps_start=0.3, eps_finish=0.3)
    for blocks in (0, 1, 50):
        assert float(pqn.epsilon_at(jnp.int32(blocks), 25, flat)) == np.float32(0.3)
    for planned in (1, 3, 25):
        for blocks in range(0, 30):
            np.testing.assert_allclose(
                float(pqn.epsilon_at(jnp.int32(blocks), planned, config)),
                pqn.pqn_epsilon_reference(blocks, planned, config),
                atol=2e-6,
            )
    planned = 2
    schedule = optax.linear_schedule(
        config.q_lr, 1e-10, planned * config.epochs * config.num_minibatches
    )
    for count in (0, 1, 63, 127, 128, 400):
        np.testing.assert_allclose(
            float(jnp.asarray(schedule(count))),
            pqn.pqn_learning_rate(count, planned, config),
            rtol=2e-6,
            atol=1e-12,
        )
    assert pqn.pqn_learning_rate(128, planned, config) == pytest.approx(1e-10)
    constant = pqn.PQNConfig(lr_linear_decay=False)
    assert pqn.pqn_learning_rate(77, planned, constant) == constant.q_lr


@pytest.mark.parametrize("frame", ("world", "left"))
def test_system_actions_are_legal_and_greedy_at_zero_epsilon(
    network: pqn.PQNInferenceVariables,
    environment: tuple[SystemInput, SystemInput],
    frame: str,
) -> None:
    actor_inputs = environment[0]
    system = pqn.make_pqn_system(network, input_scale=0.01, spawn_frame=frame)
    memory = _memory(system, actor_inputs)
    assert memory.shape == (2, 5, 512) and memory.dtype == jnp.float32
    np.testing.assert_array_equal(memory, 0)
    first = _apply(system, memory, actor_inputs, 1)
    second = _apply(system, memory, actor_inputs, 2)
    equal(first, second)
    assert isinstance(first.learning_outputs, pqn.PQNLearningOutputs)
    equal(first.learning_outputs.pre_memory, memory)
    indices = encode_actions(first.actions)
    legal = categorical_action_mask(actor_inputs.action_mask)
    assert bool(jnp.all(jnp.take_along_axis(legal, indices[..., None], axis=-1)))
    np.testing.assert_array_equal(indices[:, 2:], 0)
    np.testing.assert_array_equal(cast(Array, first.next_memory)[:, 2:], 0)
    inputs, mask = actor_inputs.actors, actor_inputs.action_mask
    flag = jnp.zeros((2, 5), bool)
    if frame == "left":
        flag = spawn_frame_flag(inputs, frame)
        assert bool(flag.any()) and not bool(flag.all())
        inputs, mask = mirror_team_view(inputs, mask, flag)
    _, values = cast(
        tuple[Array, Array],
        pqn.PQNNetwork(input_scale=0.01).apply(
            pqn._flax_variables(network),
            memory,
            encode_actor_inputs(inputs)[None],
            actor_inputs.episode_start[None],
            actor_inputs.valid[None],
            actor_inputs.active_mask[None],
        ),
    )
    np.testing.assert_array_equal(values[0, :, 2:], 0)
    greedy = pqn.greedy_actions(values[0], categorical_action_mask(mask))
    np.testing.assert_array_equal(indices, mirror_action_indices(greedy, flag))


def test_system_explores_uniformly_over_legal_actions(
    network: pqn.PQNInferenceVariables, environment: tuple[SystemInput, SystemInput]
) -> None:
    actor_inputs = environment[0]
    system = pqn.make_pqn_system(network, epsilon=1.0, input_scale=0.01)
    memory = _memory(system, actor_inputs)
    legal = categorical_action_mask(actor_inputs.action_mask)
    seen: set[int] = set()
    for seed in range(12):
        indices = encode_actions(_apply(system, memory, actor_inputs, seed).actions)
        assert bool(jnp.all(jnp.take_along_axis(legal, indices[..., None], axis=-1)))
        np.testing.assert_array_equal(indices[:, 2:], 0)
        seen.add(int(indices[0, 0]))
    assert len(seen) > 6
    for rate in (-0.1, 1.5, math.nan, True):
        with pytest.raises(ValueError):
            pqn.make_pqn_system(network, epsilon=rate)  # pyright: ignore[reportArgumentType]


def test_system_draws_match_the_epsilon_greedy_probabilities(
    network: pqn.PQNInferenceVariables, environment: tuple[SystemInput, SystemInput]
) -> None:
    inputs = _lane(environment[0], 0)
    system = pqn.make_pqn_system(
        network, epsilon=0.37, input_scale=0.01, spawn_frame="world"
    )
    memory = _memory(system, inputs)
    _, values = cast(
        tuple[Array, Array],
        pqn.PQNNetwork(input_scale=0.01).apply(
            pqn._flax_variables(network),
            memory,
            encode_actor_inputs(inputs.actors)[None],
            inputs.episode_start[None],
            inputs.valid[None],
            inputs.active_mask[None],
        ),
    )
    legal = categorical_action_mask(inputs.action_mask)
    expected = np.asarray(qmix.action_probabilities(values[0], legal, 0.37))[0, :2]
    # 20,000 draws through the System's own key split, as ten copies of the
    # lane 2,000 wide: one compiled program, bounded memory.
    lanes, chunks = 2000, 10

    def wide(value: Array) -> Array:
        return jnp.repeat(value, lanes, axis=0)

    many = jax.tree.map(wide, inputs)
    start = jnp.repeat(memory, lanes, axis=0)
    apply = jax.jit(system.apply)
    counts = np.zeros((2, 198), np.int64)
    for chunk in range(chunks):
        keys = jax.random.split(jax.random.key(19049011 + chunk), lanes)
        output = cast(SystemOutput, apply(system.variables, start, many, keys))
        drawn = np.asarray(encode_actions(output.actions))[:, :2]
        for slot in range(2):
            counts[slot] += np.bincount(drawn[:, slot], minlength=198)
    total = lanes * chunks
    assert (counts[~np.asarray(legal[0, :2])] == 0).all()
    spread = 5 * np.sqrt(total * expected * (1 - expected))
    assert (np.abs(counts - total * expected) <= spread + 1e-9).all()


@pytest.mark.parametrize("frame", ("world", "left"))
def test_exact_ties_break_in_the_network_frame(
    network: pqn.PQNInferenceVariables,
    environment: tuple[SystemInput, SystemInput],
    frame: str,
) -> None:
    actor_inputs = environment[0]
    params = jax.tree.map(_zeros, network.params)
    params["Dense_2"]["bias"] = jnp.zeros(198).at[66].set(1.0).at[88].set(1.0)
    system = pqn.make_pqn_system(
        pqn.PQNInferenceVariables(params, network.batch_stats), spawn_frame=frame
    )
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


def test_extreme_finite_scores_still_choose_a_legal_action() -> None:
    edge = jnp.full((198,), float(np.finfo(np.float32).min), jnp.float32)
    only_last = jnp.zeros((198,), bool).at[197].set(True)
    assert int(pqn.greedy_actions(edge, only_last)) == 197
    # The donor's subtraction of 1e10 leaves every score at finfo.min.
    donor = edge - (1 - only_last.astype(jnp.float32)) * 1e10
    assert int(jnp.argmax(donor)) == 0


def test_memory_resets_only_at_episode_starts(
    network: pqn.PQNInferenceVariables, environment: tuple[SystemInput, SystemInput]
) -> None:
    actor_inputs = environment[0]
    system = pqn.make_pqn_system(network, input_scale=0.01)
    memory = jnp.full((2, 5, 512), 0.25, jnp.float32)
    carried = actor_inputs._replace(
        episode_start=jnp.zeros((2,), bool), valid=jnp.asarray([True, False])
    )
    output = _apply(system, memory, carried, 4)
    next_memory = cast(Array, output.next_memory)
    np.testing.assert_array_equal(next_memory[1], memory[1])
    assert not np.array_equal(next_memory[0, :2], memory[0, :2])
    np.testing.assert_array_equal(next_memory[0, 2:], 0)
    restarted = _apply(
        system, memory, carried._replace(episode_start=jnp.ones(2, bool)), 4
    )
    fresh = _apply(system, jnp.zeros_like(memory), carried, 4)
    equal(cast(Array, restarted.next_memory)[0], cast(Array, fresh.next_memory)[0])


def _dead_slot_zero(inputs: SystemInput) -> SystemInput:
    # Slot 0 dies: alive feature 0 and unused slot 4's neutral-only mask.
    own = inputs.actors.observation

    def neutral(leaf: Array) -> Array:
        return leaf.at[:, 0].set(leaf[:, 4])

    return inputs._replace(
        actors=inputs.actors._replace(
            observation=own._replace(
                self_features=own.self_features.at[:, 0, AGENT_FEATURE_ALIVE].set(0.0)
            )
        ),
        action_mask=jax.tree.map(neutral, inputs.action_mask),
    )


@pytest.mark.parametrize("epsilon", (0.0, 1.0))
def test_a_dead_active_slot_takes_neutral_and_keeps_its_memory(
    network: pqn.PQNInferenceVariables,
    environment: tuple[SystemInput, SystemInput],
    epsilon: float,
) -> None:
    alive = environment[0]._replace(episode_start=jnp.zeros((2,), bool))
    dead = _dead_slot_zero(alive)
    assert bool(dead.active_mask[:, 0].all())
    legal = np.asarray(categorical_action_mask(dead.action_mask))
    assert (legal[:, 0].sum(-1) == 1).all()
    neutral = int(np.argmax(legal[0, 0]))
    system = pqn.make_pqn_system(network, epsilon=epsilon, input_scale=0.01)
    memory = jnp.full((2, 5, 512), 0.25, jnp.float32)
    output = _apply(system, memory, dead, 5)
    np.testing.assert_array_equal(
        np.asarray(encode_actions(output.actions))[:, 0], neutral
    )
    # Death resets nothing: the dead slot's memory keeps moving from its old
    # value, and legality never enters the network.
    next_memory = cast(Array, output.next_memory)
    assert bool(jnp.all(jnp.any(next_memory[:, 0] != memory[:, 0], axis=-1)))
    assert bool(jnp.all(jnp.any(next_memory[:, 0] != 0, axis=-1)))
    full_mask = dead._replace(action_mask=alive.action_mask)
    control = _apply(system, memory, full_mask, 5)
    equal(cast(Array, control.next_memory), next_memory)
    if epsilon == 0.0:
        # Positive control: with its full mask the same slot picks another action.
        chosen = np.asarray(encode_actions(control.actions))[:, 0]
        assert (chosen != neutral).any()


def test_actor_reads_only_its_own_view_and_memory(
    network: pqn.PQNInferenceVariables, environment: tuple[SystemInput, SystemInput]
) -> None:
    system = pqn.make_pqn_system(network, input_scale=0.01)
    inputs = environment[0]._replace(episode_start=jnp.zeros((2,), bool))
    memory = jnp.full((2, 5, 512), 0.1, jnp.float32)
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
    control = _apply(system, memory, moved(0), 6)
    assert not np.array_equal(
        cast(Array, first.next_memory)[:, 0], cast(Array, control.next_memory)[:, 0]
    )
    assert set(system.variables._fields) == {"network", "epsilon"}


@pytest.mark.parametrize("team", (0, 1))
def test_mirror_twins_act_alike_from_either_spawn_end(
    network: pqn.PQNInferenceVariables, team: int
) -> None:
    # SystemInput carries no team or global slot field, so a team's side can
    # only reach the System through its spawn end. Lanes 0 and 1 are the same
    # game with the ends swapped: in the left frame both lanes act alike.
    assert set(SystemInput._fields) == {
        "actors",
        "action_mask",
        "active_mask",
        "episode_start",
        "valid",
    }
    env, observations, state = _two_lane_reset()
    del env
    inputs = system_inputs(observations, state, team=team)
    flag = np.asarray(spawn_frame_flag(inputs.actors, "left"))
    assert flag[1 - team].all() and not flag[team].any()
    memories: dict[str, Array] = {}
    for frame in ("left", "world"):
        system = pqn.make_pqn_system(network, input_scale=0.01, spawn_frame=frame)
        output = _apply(system, _memory(system, inputs), inputs, 13)
        memories[frame] = cast(Array, output.next_memory)
        if frame == "left":
            chosen = mirror_action_indices(
                encode_actions(output.actions), jnp.asarray(flag)
            )
            np.testing.assert_array_equal(chosen[0], chosen[1])
    np.testing.assert_allclose(
        memories["left"][0], memories["left"][1], atol=1e-4, rtol=0
    )
    # Positive control: in the world frame the two ends see different inputs.
    assert float(jnp.max(jnp.abs(memories["world"][0] - memories["world"][1]))) > 1e-2


def test_one_lane_matches_its_batch_row_and_statistics_change_its_values(
    network: pqn.PQNInferenceVariables, environment: tuple[SystemInput, SystemInput]
) -> None:
    system = pqn.make_pqn_system(network, input_scale=0.01)
    memory = jnp.full((2, 5, 512), 0.1, jnp.float32).at[1].set(-0.2)
    inputs = environment[0]._replace(episode_start=jnp.zeros((2,), bool))
    keys = jax.random.split(jax.random.key(7), 2)
    together = cast(SystemOutput, system.apply(system.variables, memory, inputs, keys))
    for index in range(2):
        alone = cast(
            SystemOutput,
            system.apply(
                system.variables,
                memory[index : index + 1],
                _lane(inputs, index),
                keys[index : index + 1],
            ),
        )
        for batched, single in zip(together.actions, alone.actions, strict=True):
            np.testing.assert_array_equal(batched[index], single[0])
        np.testing.assert_allclose(
            cast(Array, together.next_memory)[index],
            cast(Array, alone.next_memory)[0],
            atol=2e-6,
            rtol=2e-6,
        )
    shifted = pqn.make_pqn_system(
        pqn.PQNInferenceVariables(
            network.params, jax.tree.map(_shifted, network.batch_stats)
        ),
        input_scale=0.01,
    )
    changed = cast(SystemOutput, shifted.apply(shifted.variables, memory, inputs, keys))
    assert not np.array_equal(
        cast(Array, together.next_memory), cast(Array, changed.next_memory)
    )


def test_changing_weights_statistics_and_epsilon_reuse_one_compilation(
    network: pqn.PQNInferenceVariables, environment: tuple[SystemInput, SystemInput]
) -> None:
    actor_inputs = environment[0]
    system = pqn.make_pqn_system(network, input_scale=0.01)
    memory = _memory(system, actor_inputs)
    traces: list[int] = []

    def apply(variables: pqn.PQNActorVariables, keys: Array) -> SystemOutput:
        traces.append(1)
        return cast(SystemOutput, system.apply(variables, memory, actor_inputs, keys))

    compiled = cast(
        Callable[[pqn.PQNActorVariables, Array], SystemOutput], jax.jit(apply)
    )
    keys = jax.random.split(jax.random.key(8), 2)
    greedy = compiled(system.variables, keys)
    changed = pqn.PQNActorVariables(
        pqn.PQNInferenceVariables(
            jax.tree.map(_doubled, network.params),
            jax.tree.map(_shifted, network.batch_stats),
        ),
        jnp.float32(1.0),
    )
    explored = compiled(changed, keys)
    assert traces == [1]
    assert not np.array_equal(
        cast(Array, greedy.next_memory), cast(Array, explored.next_memory)
    )


def test_system_identity_records_scale_frame_statistics_and_epsilon(
    network: pqn.PQNInferenceVariables,
) -> None:
    identities = {
        normalize_system_registration(
            pqn.make_pqn_system(
                network, input_scale=scale, spawn_frame=frame, epsilon=rate
            ),
            phase="evaluation",
        )[0]
        for scale in (1.0, 0.01)
        for frame in ("world", "left")
        for rate in (0.0, 0.5)
    }
    assert len(identities) == 8
    shifted = pqn.make_pqn_system(
        pqn.PQNInferenceVariables(
            network.params, jax.tree.map(_shifted, network.batch_stats)
        )
    )
    assert (
        normalize_system_registration(shifted, phase="evaluation")[0]
        != normalize_system_registration(
            pqn.make_pqn_system(network), phase="evaluation"
        )[0]
    )


def test_initialization_rejects_other_key_kinds() -> None:
    with pytest.raises(ValueError, match="typed"):
        pqn.initialize_pqn(jax.random.PRNGKey(1), planned_learning_blocks=1)
    with pytest.raises(ValueError, match="typed"):
        pqn.initialize_pqn(
            jax.random.split(jax.random.key(1), 2), planned_learning_blocks=1
        )
    with pytest.raises(ValueError, match="Threefry"):
        pqn.initialize_pqn(jax.random.key(1, impl="rbg"), planned_learning_blocks=1)
    with pytest.raises(ValueError):
        pqn.initialize_pqn(jax.random.key(1), planned_learning_blocks=0)


def _training_forward(
    network: pqn.PQNInferenceVariables,
    features: Array,
    valid: Array,
    active: Array,
) -> tuple[Array, Tree]:
    rows, games = valid.shape
    (_, values), updates = cast(
        tuple[tuple[Array, Array], dict[str, Tree]],
        pqn.PQNNetwork().apply(
            pqn._flax_variables(network),
            jnp.zeros((games, 5, 512), jnp.float32),
            features,
            jnp.zeros((rows, games), bool),
            valid,
            active,
            train=True,
            mutable=["batch_stats"],
        ),
    )
    return values, updates["batch_stats"]


def test_training_normalization_uses_masked_moments_and_ignores_poison(
    network: pqn.PQNInferenceVariables,
) -> None:
    rows, games = 3, 2
    features = jax.random.normal(jax.random.key(11), (rows, games, 5, 5165))
    valid = jnp.ones((rows, games), bool).at[2, 1].set(False)
    active = jnp.ones((rows, games, 5), bool).at[:, 0, 3:].set(False)
    values, stats = _training_forward(network, features, valid, active)
    row = np.asarray(valid[..., None] & active).reshape(-1)
    flat = np.asarray(features, np.float64).reshape(-1, 5165)[row]
    mean, variance = flat.mean(0), np.maximum((flat**2).mean(0) - flat.mean(0) ** 2, 0)
    np.testing.assert_allclose(
        np.asarray(stats["BatchNorm_0"]["mean"]), 0.01 * mean, rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        np.asarray(stats["BatchNorm_0"]["var"]),
        0.99 + 0.01 * variance,
        rtol=2e-5,
        atol=2e-6,
    )
    excluded = ~(valid[..., None] & active)
    poisoned = jnp.where(excluded[..., None], jnp.nan, features)
    poisoned_values, poisoned_stats = _training_forward(
        network, poisoned, valid, active
    )
    equal(poisoned_stats, stats)
    keep = np.asarray(valid[..., None] & active)
    np.testing.assert_array_equal(
        np.asarray(poisoned_values)[keep], np.asarray(values)[keep]
    )
    assert bool(jnp.all(jnp.isfinite(poisoned_values)))


def test_lambda_returns_match_hand_computed_endings_and_cutoffs() -> None:
    gamma, lam = 0.9, 0.5
    rewards = jnp.asarray([[1.0], [2.0], [3.0], [4.0]])
    values = jnp.asarray([[10.0], [20.0], [30.0], [40.0]])
    valid = jnp.ones((4, 1), bool)
    none = jnp.zeros((4, 1), bool)
    targets = np.asarray(
        pqn.lambda_returns(rewards, none, values, valid, gamma=gamma, td_lambda=lam)
    )[:, 0]
    g2 = 3.0 + gamma * 40.0
    g1 = 2.0 + gamma * 30.0 + gamma * lam * (g2 - 30.0)
    g0 = 1.0 + gamma * 20.0 + gamma * lam * (g1 - 20.0)
    np.testing.assert_allclose(targets, [g0, g1, g2], rtol=1e-6)
    ended = none.at[1, 0].set(True)
    cut = np.asarray(
        pqn.lambda_returns(rewards, ended, values, valid, gamma=gamma, td_lambda=lam)
    )[:, 0]
    np.testing.assert_allclose(cut[1], 2.0, rtol=1e-6)
    np.testing.assert_allclose(
        cut[0], 1.0 + gamma * 20.0 + gamma * lam * (2.0 - 20.0), rtol=1e-6
    )
    # A padded tail: rows 0..2 are real, so row 1 bootstraps from row 2 only.
    padded = valid.at[3, 0].set(False)
    tail = np.asarray(
        pqn.lambda_returns(rewards, none, values, padded, gamma=gamma, td_lambda=lam)
    )[:, 0]
    np.testing.assert_allclose(tail[1], 2.0 + gamma * 30.0, rtol=1e-6)
    np.testing.assert_allclose(
        tail[0], 1.0 + gamma * 20.0 + gamma * lam * (tail[1] - 20.0), rtol=1e-6
    )


def _batch(key: Array, rows: int = 4, games: int = 2) -> pqn.PQNBatch:
    keys = jax.random.split(key, 4)
    mask = jax.random.uniform(keys[1], (rows, games, 5, 198)) > 0.4
    mask = mask.at[..., 0].set(True)
    actions = jax.random.categorical(keys[2], jnp.where(mask, 0.0, -jnp.inf)).astype(
        jnp.int32
    )
    return pqn.PQNBatch(
        jax.random.normal(keys[0], (rows, games, 5, 5165)),
        mask,
        actions,
        jax.random.normal(keys[3], (rows, games)) * 0.3,
        jnp.zeros((rows, games), bool).at[0].set(True),
        jnp.zeros((rows, games), bool),
        jnp.ones((rows, games), bool),
        jnp.ones((rows, games, 5), bool),
        jnp.zeros((games, 5, 512), jnp.float32),
    )


def test_update_skips_everything_without_a_pair_and_ignores_padding() -> None:
    state = pqn.initialize_pqn(jax.random.key(3), pqn=_SMALL, planned_learning_blocks=2)
    batch = _batch(jax.random.key(5))
    lonely = batch._replace(valid=jnp.zeros((4, 2), bool).at[0].set(True))
    unchanged, metrics = pqn.update_pqn(
        state, lonely, pqn=_SMALL, planned_learning_blocks=2
    )
    equal(unchanged, state)
    assert not bool(metrics.performed) and bool(metrics.finite)
    assert int(metrics.used_td_pairs) == 0 and float(metrics.loss) == 0.0
    tail = batch._replace(valid=jnp.ones((4, 2), bool).at[3].set(False))
    poisoned = tail._replace(
        actor_features=tail.actor_features.at[3].set(jnp.nan),
        rewards=tail.rewards.at[3].set(jnp.inf),
        active=tail.active.at[:, :, 4].set(False),
    )
    clean = poisoned._replace(
        actor_features=tail.actor_features.at[3].set(0.0).at[:, :, 4].set(123.0),
        rewards=tail.rewards.at[3].set(0.0),
        actions=tail.actions.at[:, :, 4].set(0),
    )
    first, first_metrics = pqn.update_pqn(
        state, poisoned, pqn=_SMALL, planned_learning_blocks=2
    )
    second, second_metrics = pqn.update_pqn(
        state, clean, pqn=_SMALL, planned_learning_blocks=2
    )
    assert bool(first_metrics.performed) and bool(first_metrics.finite)
    assert int(first_metrics.used_td_pairs) == 4
    assert int(first_metrics.used_agent_utilities) == 16
    equal(first, second)
    equal(first_metrics, second_metrics)


def test_parameter_counts_and_bytes_match_the_planned_sizes(
    network: pqn.PQNInferenceVariables,
) -> None:
    template = pqn.pqn_actor_template()
    params = sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(template.params))
    stats = sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(template.batch_stats))
    assert (params, stats) == (4_596_512, 12_378)
    assert params * 4 == 18_386_048 and stats * 4 == 49_512
    sizes = {
        "Dense_0": 2_644_992,
        "Dense_1": 262_656,
        "ScannedRNN_0": 1_574_912,
        "Dense_2": 101_574,
    }
    for name, size in sizes.items():
        assert sum(x.size for x in jax.tree.leaves(network.params[name])) == size
    state = jax.eval_shape(
        lambda: pqn.initialize_pqn(jax.random.key(0), planned_learning_blocks=3)
    )
    moments = sum(
        math.prod(leaf.shape)
        for leaf in jax.tree.leaves(state.opt_state)
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    )
    counts = [
        leaf for leaf in jax.tree.leaves(state.opt_state) if leaf.dtype == jnp.int32
    ]
    assert moments * 4 == 36_772_096 and len(counts) == 2
    constant = jax.eval_shape(
        lambda: pqn.initialize_pqn(
            jax.random.key(0),
            pqn=pqn.PQNConfig(lr_linear_decay=False),
            planned_learning_blocks=3,
        )
    )
    assert (
        len([x for x in jax.tree.leaves(constant.opt_state) if x.dtype == jnp.int32])
        == 1
    )
    history = 20 * (18_386_048 + 49_512 + 4)
    assert history == 368_711_280
