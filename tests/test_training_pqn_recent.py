"""Check PQN-VDN's compact rows, kept recent rows and learning windows.

Contracts checked here, all on CPU with four games, blocks of 4 rounds and a
memory window of 2 (so W = 6 initial rounds: a chunk of 4, then a chunk of 2).
A compact row keeps only Team A's five observer rows, their 5x5 permissions,
Team A masks and world-frame actions, the team task and shaping rewards, the
lifecycle flags and eight int32 identities: 26,008 bytes per game row, with
the planned sizes for the kept rows (4,639,744 bytes at B32, H4), a full
window (109,857,792 bytes at T128) and a block's stored memories (41,943,040
bytes). PQN collection keeps no physical state and never calls the physical
state encoder. Against a numbered host reference, the window puts the kept
real rows, then the new real rows, with no padding between them, through two
initial chunks, several normal blocks and a final block shorter than H; the
kept rows are the last H real rows with their original memories. On real
collection with three-decision episodes, the curriculum, shaping and a
frozen historical opponent, the window stays in game order across episode
joins, stored memories are zero at episode starts and in inactive slots, and
each stored memory is exactly what the actor's network produced on the
previous decision. Expanded minibatches rebuild network-frame features,
masks and actions bitwise equal to the live permitted-input path from both
spawn ends. A PQN learner pinned against a QMIX export collects PQN outputs
beside the other method's memory. The recorded route and the host route
return the same PQN outputs as the compiled route, the initial chunks are
planned as T then H rounds once each, and the recording declares each episode
start exactly once. Initial collection explores with rate 1 even when
eps_start is lower. These checks prove software contracts, not learning.
"""

# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.training_learner_helpers import synthetic_horizons

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines import pqn, qmix
from marl_battlegrounds.baselines.actions import (
    categorical_action_mask,
    encode_actions,
    mirror_action_indices,
)
from marl_battlegrounds.baselines.inputs import (
    encode_actor_inputs,
    spawn_frame_flag,
    team_obstacle_partners,
)
from marl_battlegrounds.evaluation.policy_execution import System, SystemInput
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import build_team_actor_input, mirror_team_view
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    collect_training_rollout,
    make_training_schedule,
    prepare_training_content,
    scan_training_rollout,
)
from marl_battlegrounds.training import collection as collection_module
from marl_battlegrounds.training import pqn_learner as learner
from marl_battlegrounds.training.checkpoints import export_system
from marl_battlegrounds.training.collection import _padding
from marl_battlegrounds.training.validation import _collection_boundary

type Tree = Any
type Scan = Callable[[TrainingCarry], tuple[TrainingCarry, TrainingRollout]]
_CONFIG = pqn.PQNConfig(rollout_length=4, memory_window=2, epochs=1, num_minibatches=2)
# 43 rounds: initial chunks of 4 and 2, nine blocks of 4, then one real round.
_LENGTHS = (4, 2, *(4,) * 10)
_REAL = (4, 2, *(4,) * 9, 1)


def _equal(left: Tree, right: Tree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


@lru_cache
def _scan(collection: TrainingCollection, length: int) -> Scan:
    return cast(
        Scan, jax.jit(partial(scan_training_rollout, collection, length=length))
    )


_window = cast(
    Callable[
        [learner.PQNRecent, learner.PQNRow, Array, Array],
        tuple[learner.PQNRow, Array, Array],
    ],
    jax.jit(learner._learning_window),
)
_retain = cast(
    Callable[[learner.PQNRow, Array, Array], learner.PQNRecent],
    jax.jit(partial(learner._retain_recent, pqn=_CONFIG)),
)


def _fill_slot_zero(bank: Array, leaf: Array) -> Array:
    return bank.at[0].set(leaf)


@dataclass
class _Blocks:
    collection: TrainingCollection
    start: TrainingCarry
    rollouts: list[TrainingRollout]
    windows: list[tuple[learner.PQNRow, Array, Array]]
    kept: list[learner.PQNRecent]


@pytest.fixture(scope="module")
def blocks(prepared: PreparedTrainingContent) -> _Blocks:
    collection, state = learner.init_pqn_learner(
        schedule=make_training_schedule(
            total_env_steps=172, num_envs=4, curriculum=True
        ),
        seed=19049111,
        prepared=prepared,
        shaping=True,
        metrics="none",
        pqn=_CONFIG,
    )
    state = cast(
        learner.PQNLearnerState,
        synthetic_horizons(collection, cast(Any, state), horizon=3),
    )
    carry = state.carry
    # Freeze the untrained actor in history slot 0 for games 1 and 3.
    history = carry.history._replace(
        count=jnp.int32(1),
        historical_variables=jax.tree.map(
            _fill_slot_zero,
            carry.history.historical_variables,
            carry.history.current_variables,
        ),
        captured_updates=carry.history.captured_updates.at[0].set(0),
        captured_rounds=carry.history.captured_rounds.at[0].set(0),
        captured_ids=carry.history.captured_ids.at[0].set(0),
        eligible=carry.history.eligible.at[0].set(True),
        next_capture_id=jnp.int32(1),
        last_capture_rounds=jnp.int32(0),
        lane_snapshot=jnp.asarray((-1, 0, -1, 0), jnp.int32),
    )
    carry = carry._replace(history=history)
    record = _Blocks(collection, carry, [], [], [])
    recent = state.recent
    for length in _LENGTHS:
        carry, rollout = _scan(collection, length)(carry)
        rows, memory = learner._pqn_rows(rollout.transitions)
        window = _window(recent, rows, memory, rollout.real_steps)
        recent = _retain(*window)
        record.rollouts.append(rollout)
        record.windows.append(window)
        record.kept.append(recent)
    return record


def _real_rows(blocks: _Blocks) -> tuple[learner.PQNRow, Array]:
    rows: list[learner.PQNRow] = []
    memories: list[Array] = []
    for rollout in blocks.rollouts:
        real = int(rollout.real_steps)

        def prefix(value: Array, n: int = real) -> Array:
            return value[:n]

        compact, memory = learner._pqn_rows(rollout.transitions)
        rows.append(cast(learner.PQNRow, jax.tree.map(prefix, compact)))
        memories.append(memory[:real])
    return cast(
        learner.PQNRow, jax.tree.map(lambda *v: jnp.concatenate(v), *rows)
    ), jnp.concatenate(memories)


def test_compact_rows_keep_only_permitted_team_a_facts(blocks: _Blocks) -> None:
    collection, carry = blocks.collection, blocks.start
    assert not collection.collect_training_state
    assert collection.actor_variables_at_step is None

    def compact(values: TrainingCarry) -> learner.PQNRow:
        return learner._pqn_rows(_padding(collection, values))[0]

    spec = jax.eval_shape(compact, carry)
    leaves = cast(list[jax.ShapeDtypeStruct], jax.tree.leaves(spec))
    per_row = sum(math.prod(leaf.shape[1:]) * leaf.dtype.itemsize for leaf in leaves)
    assert per_row == 26_008
    memory_row = 5 * 512 * 4
    assert 4 * 32 * (per_row + memory_row) == 4_639_744
    assert 132 * 32 * per_row == 109_857_792
    assert 128 * 32 * memory_row == 41_943_040
    assert "training_state" not in learner.PQNRow._fields
    for leaf in jax.tree.leaves(spec.observation):
        assert leaf.shape[:2] == (4, 5)
    assert spec.source_availability.shape == (4, 5, 5)
    for rollout in blocks.rollouts:
        rows = rollout.transitions
        assert rows.training_state is None and rollout.final_training_state is None
        assert isinstance(rows.learning_outputs, pqn.PQNLearningOutputs)
        assert rows.learning_outputs.pre_memory.shape == (
            rows.valid.shape[0],
            4,
            5,
            512,
        )


def test_collection_never_encodes_physical_state(
    prepared: PreparedTrainingContent, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object) -> Array:
        raise AssertionError("PQN collection encoded physical state")

    monkeypatch.setattr(collection_module, "encode_training_state", refuse)
    collection, state = learner.init_pqn_learner(
        schedule=make_training_schedule(total_env_steps=40, num_envs=4),
        seed=19049112,
        prepared=prepared,
        metrics="none",
        pqn=_CONFIG,
    )
    # A fresh jit wrapper traces the whole step with the patched encoder.
    scan = cast(Scan, jax.jit(partial(scan_training_rollout, collection, length=2)))
    carry, rollout = scan(state.carry)
    assert int(rollout.real_steps) == 2 and int(carry.progress.rounds) == 2


def _numbered(start: int, real: int, capacity: int) -> tuple[learner.PQNRow, Array]:
    games = 3
    values = (np.arange(capacity)[:, None] + start) * 10 + np.arange(games)
    valid = np.arange(capacity)[:, None] < real
    number = jnp.asarray(np.where(valid, values, -1), jnp.int32)
    fields = [number] * len(learner.PQNRow._fields)
    fields[learner.PQNRow._fields.index("valid")] = jnp.asarray(valid).repeat(games, 1)
    row = learner.PQNRow(*cast(list[Any], fields))
    memory = jnp.asarray(np.where(valid, values + 0.5, 0.0), jnp.float32)
    return row, memory


def test_window_and_kept_rows_match_a_numbered_host_reference() -> None:
    recent = learner.PQNRecent(
        _numbered(0, 0, 2)[0], jnp.zeros((2, 3), jnp.float32), jnp.int32(0)
    )
    history: list[int] = []
    for real, capacity in zip(_REAL, _LENGTHS, strict=True):
        row, memory = _numbered(len(history), real, capacity)
        joined, joined_memory, count = _window(recent, row, memory, jnp.int32(real))
        kept = history[-2:]
        new = list(range(len(history), len(history) + real))
        expected = [*kept, *new]
        assert int(count) == len(expected)
        numbers = np.asarray(joined.episode_id)
        valid = np.asarray(joined.valid)
        np.testing.assert_array_equal(numbers[: len(expected), 0] // 10, expected)
        assert valid[: len(expected)].all() and not valid[len(expected) :].any()
        np.testing.assert_array_equal(
            np.asarray(joined_memory)[: len(expected)], numbers[: len(expected)] + 0.5
        )
        assert (numbers[len(expected) :] == -1).all()
        history.extend(new)
        recent = _retain(joined, joined_memory, count)
        size = min(2, len(history))
        assert int(recent.size) == size
        tail = np.asarray(recent.rows.episode_id)[:size, 0] // 10
        np.testing.assert_array_equal(tail, history[-size:])
        np.testing.assert_array_equal(
            np.asarray(recent.pre_memory)[:size],
            np.asarray(recent.rows.episode_id)[:size] + 0.5,
        )
    assert len(history) == 43


def test_real_windows_keep_game_order_and_the_latest_rows(blocks: _Blocks) -> None:
    rows, memory = _real_rows(blocks)
    assert [int(r.real_steps) for r in blocks.rollouts] == list(_REAL)
    total = 0
    crossings = 0
    for rollout, (joined, joined_memory, count), recent in zip(
        blocks.rollouts, blocks.windows, blocks.kept, strict=True
    ):
        kept = min(2, total)
        total += int(rollout.real_steps)
        assert int(count) == kept + int(rollout.real_steps)
        span = slice(total - int(count), total)

        def expected(value: Array, window: slice = span) -> Array:
            return value[window]

        def actual(value: Array, n: int = int(count)) -> Array:
            return value[:n]

        _equal(jax.tree.map(actual, joined), jax.tree.map(expected, rows))
        np.testing.assert_array_equal(joined_memory[: int(count)], memory[span])
        assert not bool(joined.valid[int(count) :].any())
        assert bool(learner._sequence_valid(joined, jnp.int32(0)))
        ended = np.asarray(joined.ended[: int(count) - 1])
        start = np.asarray(joined.episode_start[1 : int(count)])
        crossings += int((ended & start).sum())
        size = min(2, total)
        assert int(recent.size) == size
        last = slice(total - size, total)

        def newest(value: Array, window: slice = last) -> Array:
            return value[window]

        def front(value: Array, n: int = size) -> Array:
            return value[:n]

        _equal(jax.tree.map(front, recent.rows), jax.tree.map(newest, rows))
        np.testing.assert_array_equal(recent.pre_memory[:size], memory[last])
    assert crossings > 0 and total == 43
    stages = set(np.asarray(rows.episode_stage).ravel().tolist())
    assert len(stages) >= 2
    assert bool((np.asarray(rows.active).sum(-1) < 5).any())
    assert bool((np.asarray(rows.opponent_snapshot) == 0).any())
    assert bool((np.asarray(rows.opponent_snapshot) == -1).any())


def test_stored_memories_are_the_actor_memories_before_each_decision(
    blocks: _Blocks,
) -> None:
    rows, memory = _real_rows(blocks)
    carry = blocks.start
    first = blocks.rollouts[0]
    starts = np.asarray(rows.episode_start)
    active = np.asarray(rows.active)
    stored = np.asarray(memory)
    assert (stored[starts] == 0).all() and (stored[~active] == 0).all()
    np.testing.assert_array_equal(
        stored[0],
        np.where(starts[0][:, None, None], 0.0, np.asarray(first.initial_memory)),
    )
    variables = carry.history.current_variables.network
    network = pqn.PQNNetwork(input_scale=_CONFIG.input_scale)
    batch = learner._expand_minibatch(rows, memory[0], _CONFIG)
    checked = 0
    for t in range(stored.shape[0] - 1):
        following, _ = cast(
            tuple[Array, Array],
            network.apply(
                pqn._flax_variables(variables),
                memory[t],
                batch.actor_features[t : t + 1],
                batch.episode_start[t : t + 1],
                batch.valid[t : t + 1],
                batch.active[t : t + 1],
                train=False,
            ),
        )
        keep = ~starts[t + 1]
        np.testing.assert_allclose(
            np.asarray(following)[keep], stored[t + 1][keep], rtol=0, atol=1e-6
        )
        checked += int(keep.sum())
    assert checked > 0


@pytest.mark.parametrize("frame", ("left", "world"))
def test_expanded_minibatches_equal_the_live_permitted_inputs(
    blocks: _Blocks, frame: str
) -> None:
    rollout = blocks.rollouts[2]
    rows, memory = learner._pqn_rows(rollout.transitions)
    chosen = jnp.asarray((2, 0), jnp.int32)

    def select(value: Array) -> Array:
        return jnp.take(value, chosen, axis=1)

    config = pqn.PQNConfig(
        rollout_length=4,
        memory_window=2,
        epochs=1,
        num_minibatches=2,
        spawn_frame=frame,
    )
    batch = learner._expand_minibatch(
        cast(learner.PQNRow, jax.tree.map(select, rows)), memory[0][chosen], config
    )
    transitions: Any = jax.tree.map(select, rollout.transitions)
    live = jax.vmap(
        jax.vmap(build_team_actor_input, in_axes=(0, None)), in_axes=(0, None)
    )(transitions.observations, 0)
    mask = transitions.action_mask
    actions = encode_actions(
        ActorAction(*(head[..., :5] for head in transitions.actions))
    )
    if frame == "left":
        flag = spawn_frame_flag(live, frame)
        assert bool(flag.any()) and not bool(flag.all())
        live, mask = mirror_team_view(
            live, mask, flag, obstacle_partners=team_obstacle_partners(live)
        )
        actions = mirror_action_indices(actions, flag)
    np.testing.assert_array_equal(batch.actor_features, encode_actor_inputs(live))
    np.testing.assert_array_equal(batch.action_mask, categorical_action_mask(mask))
    np.testing.assert_array_equal(batch.actions, actions)
    np.testing.assert_array_equal(
        batch.rewards,
        qmix.team_task_reward(transitions.task_rewards, transitions.active)
        + transitions.shaping_reward,
    )
    np.testing.assert_array_equal(batch.initial_memory, memory[0][chosen])
    assert batch.actor_features.shape == (4, 2, 5, 5165)
    assert bool(
        jnp.all(jnp.take_along_axis(batch.action_mask, batch.actions[..., None], -1))
    )


def test_a_pqn_learner_pinned_against_a_qmix_export_collects_pqn_rows(
    prepared: PreparedTrainingContent, tmp_path: Path
) -> None:
    export = export_system(
        qmix.initialize_qmix(jax.random.key(19049113)).online_q,
        tmp_path / "qmix",
        metadata={
            "run_id": "pinned",
            "seed": 19049113,
            "env_steps": 0,
            "checkpoint_id": "c" * 64,
            "optimizer_steps": 0,
        },
        spawn_frame="left",
        method="qmix",
    )
    collection, state = learner.init_pqn_learner(
        schedule=make_training_schedule(
            total_env_steps=40, num_envs=4, early_history_capture=True
        ),
        seed=19049114,
        prepared=prepared,
        metrics="none",
        pinned_opponent_share=0.5,
        pinned_opponent=str(export),
        pqn=_CONFIG,
    )
    assert collection.pinned_opponent is not None
    carry = state.carry
    history = carry.history._replace(
        lane_snapshot=jnp.asarray((-1, -2, -1, -2), jnp.int32),
    )
    carry, rollout = _scan(collection, 4)(carry._replace(history=history))
    rows = rollout.transitions
    assert isinstance(rows.learning_outputs, pqn.PQNLearningOutputs)
    assert carry.memory.team_a.shape == (4, 5, 512)
    widths = {leaf.shape[-1] for leaf in jax.tree.leaves(carry.memory.team_b)}
    assert {512, qmix.QMIX_HIDDEN_SIZE} <= widths
    snapshot = np.asarray(rows.opponent_snapshot)
    np.testing.assert_array_equal(snapshot, np.tile((-1, -2, -1, -2), (4, 1)))
    np.testing.assert_array_equal(np.asarray(rows.opponent_update)[snapshot == -2], -2)
    compact, memory = learner._pqn_rows(rows)
    assert bool(learner._behavior_valid(rollout))
    assert compact.valid.shape == (4, 4) and memory.shape == (4, 4, 5, 512)


def _writer_policies(collection: TrainingCollection) -> dict[str, object]:
    return {"team_a": collection.actor, "team_b": collection.opponent}


def test_recorded_initial_chunks_match_the_compiled_route_once(
    prepared: PreparedTrainingContent, tmp_path: Path
) -> None:
    chunks: list[int] = []
    rounds = 0
    while rounds < _CONFIG.initial_rounds:
        chunks.append(min(_CONFIG.rollout_length, _CONFIG.initial_rounds - rounds))
        rounds += chunks[-1]
        assert _collection_boundary(
            rounds,
            total_env_steps=27,
            num_envs=1,
            rollout_length=4,
            initial_rounds=_CONFIG.initial_rounds,
        ) == (rounds, len(chunks))
    assert chunks == [4, 2]
    collection, state = learner.init_pqn_learner(
        schedule=make_training_schedule(total_env_steps=108, num_envs=4),
        seed=19049115,
        prepared=prepared,
        metrics="none",
        recording=True,
        pqn=_CONFIG,
    )
    state = cast(
        learner.PQNLearnerState,
        synthetic_horizons(collection, cast(Any, state), horizon=3),
    )
    direct = state.carry
    recorded_carry = state.carry
    starts = 0
    with marl_bgs.RunWriter(
        tmp_path, phase="training", policies=_writer_policies(collection)
    ) as writer:
        for length in chunks:
            direct, expected = _scan(collection, length)(direct)
            recorded_carry, recorded = collect_training_rollout(
                collection, recorded_carry, length=length, writer=writer
            )
            _equal(recorded, expected)
            _equal(recorded_carry, direct)
            assert isinstance(
                recorded.transitions.learning_outputs, pqn.PQNLearningOutputs
            )
            rows = recorded.transitions
            starts += int(np.asarray(rows.episode_start & rows.valid).sum())
        run_dir = writer.run_dir
    details = json.loads((run_dir / "run_details.json").read_text())
    declared = next(iter(details["passes"].values()))["episode_starts"]
    assert len(declared) == starts > 4


def _host_init(variables: Tree, inputs: SystemInput, keys: Array) -> list[int]:
    del variables, keys
    return [0] * len(np.asarray(inputs.valid))


def _host_apply(
    variables: Tree, memory: list[int], inputs: SystemInput, keys: Array
) -> tuple[ActorAction, list[int]]:
    del variables, keys
    move = np.broadcast_to(
        (np.asarray(memory, np.int32) % 9)[:, None], (len(memory), 5)
    ).astype(np.int32)
    zero = np.zeros_like(move)
    valid = np.asarray(inputs.valid)
    following = [count + int(flag) for count, flag in zip(memory, valid, strict=True)]
    return ActorAction(move, zero, zero), following  # pyright: ignore[reportArgumentType]


def _jax_init(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    del variables, keys
    return jnp.zeros(inputs.valid.shape, jnp.int32)


def _jax_apply(
    variables: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, Array]:
    del variables, keys
    move = jnp.broadcast_to((memory % 9)[:, None], (memory.shape[0], 5))
    zero = jnp.zeros_like(move)
    return ActorAction(move, zero, zero), memory + inputs.valid.astype(jnp.int32)


def _pinned(
    prepared: PreparedTrainingContent, system: System
) -> tuple[TrainingCollection, TrainingCarry]:
    collection, state = learner.init_pqn_learner(
        schedule=make_training_schedule(
            total_env_steps=40, num_envs=4, early_history_capture=True
        ),
        seed=19049116,
        prepared=prepared,
        metrics="none",
        pinned_opponent_share=0.5,
        pinned_opponent=system,
        pqn=_CONFIG,
    )
    carry = state.carry
    history = carry.history._replace(
        lane_snapshot=jnp.asarray((-1, -2, -1, -2), jnp.int32),
    )
    return collection, carry._replace(history=history)


def test_the_host_route_returns_the_same_pqn_outputs_as_its_jax_twin(
    prepared: PreparedTrainingContent,
) -> None:
    host = System("Host Counter", _host_apply, init=_host_init, execution="host")
    twin = System("Jax Counter", _jax_apply, init=_jax_init)
    host_collection, host_carry = _pinned(prepared, host)
    jax_collection, jax_carry = _pinned(prepared, twin)
    assert host_collection.host_opponent is not None
    for length in (4, 2):
        host_carry, host_rollout = collect_training_rollout(
            host_collection, host_carry, length=length
        )
        jax_carry, jax_rollout = collect_training_rollout(
            jax_collection, jax_carry, length=length
        )
        outputs = host_rollout.transitions.learning_outputs
        assert isinstance(outputs, pqn.PQNLearningOutputs)
        _equal(outputs, jax_rollout.transitions.learning_outputs)
        _equal(host_rollout.transitions.actions, jax_rollout.transitions.actions)
        pinned = np.asarray(host_rollout.transitions.opponent_snapshot) == -2
        assert pinned.any()


def test_initial_collection_explores_at_rate_one_below_eps_start(
    prepared: PreparedTrainingContent,
) -> None:
    config = pqn.PQNConfig(
        rollout_length=4,
        memory_window=2,
        epochs=1,
        num_minibatches=2,
        eps_start=0.5,
        eps_finish=0.1,
    )
    collection, state = learner.init_pqn_learner(
        schedule=make_training_schedule(total_env_steps=40, num_envs=4),
        seed=19049117,
        prepared=prepared,
        metrics="none",
        pqn=config,
    )
    variables = state.carry.history.current_variables
    assert isinstance(variables, pqn.PQNActorVariables)
    assert float(variables.epsilon) == 1.0
    learner.validate_pqn_learner(collection, state, pqn=config)
