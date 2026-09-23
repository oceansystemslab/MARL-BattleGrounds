"""Check QMIX's collection hook and compact replay on real training decisions.

Contracts checked here, all on CPU. The optional variables hook sets the shared
current variables before every real decision from the rounds completed so far:
Team A and every current Team B lane act with the set value, a historical
opponent keeps its captured value, and the carry keeps the stored variables.
Setup rejects a hook that changes the variable tree or lacks a JSON identity,
and checkpoint collection details record the identity only when a hook is set.
Replay rows keep only Team A's five observer rows and 5x5 permissions, the
team mean task reward and the shaping reward separately. The default layout is
29,664 bytes per game row with a strong int32 write index. Guarded insertion
stores exactly the real prefix of each block (never padding), in order, across
block seams and ring wrap; readiness starts at the minimum. Samples are
consecutive stored rows in time order, drawn with replacement; the newest row
of a game is only ever the last row of a sequence; episodes join only after an
ended row. The same key gives the same sample after a device round trip.
Expanded samples rebuild network-frame features, masks and actions bitwise
equal to the live permitted-input path in both spawn frames. These checks use
short synthetic horizons after verified content admission and prove software
contracts, not learning or GPU cost.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.test_training_collection import (
    _synthetic,  # pyright: ignore[reportPrivateUsage]
)

from marl_battlegrounds.baselines import qmix
from marl_battlegrounds.baselines.actions import (
    categorical_action_mask,
    mirror_action_indices,
)
from marl_battlegrounds.baselines.inputs import (
    encode_actor_inputs,
    spawn_frame_flag,
    team_obstacle_partners,
)
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import build_team_actor_input, mirror_team_view
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    checkpoints,
    init_training_collection,
    make_training_schedule,
    prepare_training_content,
    refresh_opponents,
    scan_training_rollout,
)
from marl_battlegrounds.training import qmix_learner as replay

# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false

type Tree = Any
type Context = tuple[TrainingCollection, TrainingCarry]
_SMALL = qmix.QMIXConfig(
    rollout_length=4,
    buffer_size=10,
    min_buffer_size=6,
    sample_sequence_length=4,
    sample_batch_size=24,
)


def _clock_memory(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    del variables, keys
    return jnp.zeros(inputs.active_mask.shape, jnp.float32)


def _clock_apply(
    variables: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    del keys
    zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
    seen = jnp.broadcast_to(variables["clock"], inputs.active_mask.shape)
    return SystemOutput(
        ActorAction(zero, zero, zero), jnp.where(inputs.valid[:, None], seen, memory)
    )


@dataclass(frozen=True)
class _ClockHook:
    offset: float = 0.5

    def __call__(self, variables: Tree, completed_rounds: Array) -> Tree:
        return {"clock": completed_rounds.astype(jnp.float32) + self.offset}

    @property
    def identity(self) -> dict[str, object]:
        return {"kind": "test_clock", "version": 1}


@dataclass(frozen=True)
class _ShapeChangingHook:
    def __call__(self, variables: Tree, completed_rounds: Array) -> Tree:
        return {"clock": jnp.zeros((2,), jnp.float32) + completed_rounds}

    @property
    def identity(self) -> dict[str, object]:
        return {"kind": "bad"}


def _clock_system() -> System:
    return System(
        "Clock Reader",
        _clock_apply,
        init=_clock_memory,
        variables={"clock": jnp.float32(-5)},
    )


_insert = cast(
    Callable[[replay.ReplayState, TrainingRollout], replay.ReplayState],
    jax.jit(partial(replay._insert_rollout, qmix=_SMALL)),
)
_sample = cast(
    Callable[[replay.ReplayState, Array], replay.QMIXReplayRow],
    jax.jit(partial(replay._sample_rows, qmix=_SMALL, num_envs=2)),
)


def _concatenate(*values: Array) -> Array:
    return jnp.concatenate(values)


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


@lru_cache
def _scan(
    collection: TrainingCollection, length: int
) -> Callable[[TrainingCarry], tuple[TrainingCarry, TrainingRollout]]:
    def scan(values: TrainingCarry) -> tuple[TrainingCarry, TrainingRollout]:
        return scan_training_rollout(collection, values, length=length)

    return cast(
        Callable[[TrainingCarry], tuple[TrainingCarry, TrainingRollout]], jax.jit(scan)
    )


def test_hook_sets_current_values_for_both_teams_only(
    prepared: PreparedTrainingContent,
) -> None:
    actor = _clock_system()
    collection, carry = init_training_collection(
        actor,
        actor.variables,
        schedule=make_training_schedule(total_env_steps=16, num_envs=2),
        prepared=prepared,
        metrics="none",
        actor_variables_at_step=_ClockHook(),
    )
    for rounds in range(2):
        carry, rollout = _scan(collection, 1)(carry)
        assert int(rollout.real_steps) == 1
        np.testing.assert_array_equal(carry.memory.team_a, rounds + 0.5)
        np.testing.assert_array_equal(carry.memory.team_b, rounds + 0.5)
        np.testing.assert_array_equal(carry.history.current_variables["clock"], -5)
    history, event = refresh_opponents(
        carry.history,
        {"clock": jnp.float32(777)},
        completed_rounds=carry.progress.rounds,
        update_index=jnp.int32(1),
        schedule=carry.schedule,
    )
    assert bool(event.created) and not bool(history.error)
    carry = carry._replace(
        history=history._replace(lane_snapshot=jnp.asarray([0, -1], jnp.int32))
    )
    carry, _ = _scan(collection, 1)(carry)
    np.testing.assert_array_equal(carry.memory.team_a, 2.5)
    # Lane 0 meets the frozen snapshot, which keeps its captured value.
    np.testing.assert_array_equal(carry.memory.team_b[0], 777)
    np.testing.assert_array_equal(carry.memory.team_b[1], 2.5)
    np.testing.assert_array_equal(carry.history.current_variables["clock"], 777)
    details = checkpoints._collection_details(collection)
    assert details["actor_variables_at_step"] == {"kind": "test_clock", "version": 1}


def test_hook_setup_rejects_changed_trees_and_missing_identity(
    prepared: PreparedTrainingContent,
) -> None:
    actor = _clock_system()
    schedule = make_training_schedule(total_env_steps=16, num_envs=2)
    with pytest.raises(ValueError, match="tree, shapes and dtypes"):
        init_training_collection(
            actor,
            actor.variables,
            schedule=schedule,
            prepared=prepared,
            metrics="none",
            actor_variables_at_step=_ShapeChangingHook(),
        )

    def plain(variables: Tree, completed_rounds: Array) -> Tree:
        del completed_rounds
        return variables

    with pytest.raises(TypeError, match="identity"):
        init_training_collection(
            actor,
            actor.variables,
            schedule=schedule,
            prepared=prepared,
            metrics="none",
            actor_variables_at_step=plain,  # pyright: ignore[reportArgumentType]
        )
    collection, _ = init_training_collection(
        actor, actor.variables, schedule=schedule, prepared=prepared, metrics="none"
    )
    assert collection.actor_variables_at_step is None
    assert "actor_variables_at_step" not in checkpoints._collection_details(collection)


def _qmix_context(prepared: PreparedTrainingContent, frame: str) -> Context:
    params = qmix.initialize_qmix(jax.random.key(19048101)).online_q
    system = qmix.make_qmix_system(params, epsilon=1.0, spawn_frame=frame)
    context = init_training_collection(
        system,
        system.variables,
        schedule=make_training_schedule(total_env_steps=28, num_envs=2),
        prepared=prepared,
        shaping=True,
        collect_training_state=True,
        metrics="none",
        actor_variables_at_step=qmix.QMIXExploration(0.05, 20, 2),
    )
    return _synthetic(context, horizon=3)


@dataclass
class _Blocks:
    context: Context
    rollouts: list[TrainingRollout]
    replays: list[replay.ReplayState]


@pytest.fixture(scope="module")
def blocks(prepared: PreparedTrainingContent) -> _Blocks:
    context = _qmix_context(prepared, "left")
    collection, carry = context
    state = replay._init_replay(collection, carry, _SMALL)
    rollouts: list[TrainingRollout] = []
    replays: list[replay.ReplayState] = [state]
    # 14 rounds: blocks of 4, 4, 4 and a padded final block of 2.
    for _ in range(4):
        carry, rollout = _scan(collection, 4)(carry)
        state = _insert(state, rollout)
        rollouts.append(rollout)
        replays.append(state)
    return _Blocks(context, rollouts, replays)


def _real_rows(blocks: _Blocks) -> replay.QMIXReplayRow:
    rows: list[replay.QMIXReplayRow] = []
    for rollout in blocks.rollouts:
        real = int(rollout.real_steps)

        def prefix(value: Array, n: int = real) -> Array:
            return value[:n]

        compact = replay._replay_rows(rollout.transitions)
        rows.append(cast(replay.QMIXReplayRow, jax.tree.map(prefix, compact)))
    return cast(replay.QMIXReplayRow, jax.tree.map(_concatenate, *rows))


def _equal(left: Tree, right: Tree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_replay_layout_keeps_only_team_a_rows(blocks: _Blocks) -> None:
    collection, carry = blocks.context
    spec = replay._row_spec(collection, carry)
    leaves = cast(list[jax.ShapeDtypeStruct], jax.tree.leaves(spec))
    per_row = sum(math.prod(leaf.shape[1:]) * leaf.dtype.itemsize for leaf in leaves)
    assert per_row == 29_664
    assert 32 * 1000 * per_row == 949_248_000
    for leaf in jax.tree.leaves(spec.observation):
        assert leaf.shape[:2] == (2, 5)
    assert spec.source_availability.shape == (2, 5, 5)
    assert spec.training_state.shape == (2, 919)

    def default_replay(values: TrainingCarry) -> replay.ReplayState:
        return replay._init_replay(collection, values, qmix.DEFAULT_QMIX_CONFIG)

    layout = jax.eval_shape(default_replay, carry)
    assert layout.current_index.dtype == jnp.int32
    assert not layout.current_index.weak_type
    for leaf in jax.tree.leaves(layout.experience):
        assert leaf.shape[:2] == (2, 1000)
    empty = blocks.replays[0]
    assert not empty.current_index.weak_type and int(empty.current_index) == 0
    assert all(bool(jnp.all(leaf == 0)) for leaf in jax.tree.leaves(empty.experience))


def test_insertion_stores_exactly_the_real_prefix_in_order(blocks: _Blocks) -> None:
    real = [int(rollout.real_steps) for rollout in blocks.rollouts]
    assert real == [4, 4, 4, 2]
    assert not bool(blocks.rollouts[-1].transitions.valid[2:].any())
    stored = 0
    for count, state in zip(real, blocks.replays[1:], strict=True):
        stored += count
        assert int(state.current_index) == stored % 10
        assert bool(state.is_full) == (stored >= 10)
        assert bool(replay._replay_ready(state, _SMALL)) == (stored >= 6)
    rows = _real_rows(blocks)
    final = blocks.replays[-1]
    # The last ten real rows sit in ring order starting at the write index.
    order = (jnp.arange(10) + final.current_index) % 10

    def last_ten(value: Array) -> Array:
        return value[4:]

    def time_major(value: Array) -> Array:
        return jnp.swapaxes(value[:, order], 0, 1)

    newest = jax.tree.map(last_ten, rows)
    kept = jax.tree.map(time_major, final.experience)
    _equal(kept, newest)
    assert bool(jnp.all(final.experience.valid))
    for lane in range(2):
        steps = np.asarray(rows.decision_step[:, lane])
        ids = np.asarray(rows.episode_id[:, lane])
        ended = np.asarray(rows.ended[:, lane])
        starts = np.asarray(rows.episode_start[:, lane])
        for t in range(1, len(steps)):
            if ended[t - 1]:
                assert starts[t] and steps[t] == 0 and ids[t] != ids[t - 1]
            else:
                assert ids[t] == ids[t - 1] and steps[t] == steps[t - 1] + 1
        assert ended.any()


def test_task_and_shaping_rewards_are_stored_separately(blocks: _Blocks) -> None:
    rollout = blocks.rollouts[0]
    rows = replay._replay_rows(rollout.transitions)
    transitions = rollout.transitions
    expected = qmix.team_task_reward(transitions.task_rewards, transitions.active)
    np.testing.assert_array_equal(rows.task_reward, expected)
    np.testing.assert_array_equal(rows.shaping_reward, transitions.shaping_reward)
    np.testing.assert_array_equal(
        rows.actions[..., 0],
        transitions.actions.move[..., 0] * 22
        + transitions.actions.select_target[..., 0] * 2
        + transitions.actions.use_ultimate[..., 0],
    )


def test_samples_are_ordered_repeatable_and_leave_the_newest_row_last(
    blocks: _Blocks,
) -> None:
    final = blocks.replays[-1]
    rows = _sample(final, jax.random.key(5))
    assert rows.valid.shape == (24, 4) and bool(jnp.all(rows.valid))
    steps, ids = np.asarray(rows.decision_step), np.asarray(rows.episode_id)
    ended, starts = np.asarray(rows.ended), np.asarray(rows.episode_start)
    for m in range(24):
        for t in range(1, 4):
            if ended[m, t - 1]:
                assert starts[m, t] and steps[m, t] == 0
            else:
                assert ids[m, t] == ids[m, t - 1] and steps[m, t] == steps[m, t - 1] + 1
    newest = final.experience.episode_id[:, (int(final.current_index) - 1) % 10]
    newest_step = final.experience.decision_step[:, (int(final.current_index) - 1) % 10]
    for m in range(24):
        for t in range(3):
            assert not any(
                ids[m, t] == int(newest[lane]) and steps[m, t] == int(newest_step[lane])
                for lane in range(2)
            )
    starts_seen = {(int(ids[m, 0]), int(steps[m, 0])) for m in range(24)}
    # Two games with seven possible starts each: 24 draws must repeat some.
    assert len(starts_seen) < 24
    restored = jax.device_put(jax.device_get(final))
    again = _sample(restored, jax.random.key(5))
    _equal(rows, again)
    other = _sample(final, jax.random.key(6))
    assert not np.array_equal(np.asarray(other.decision_step), steps)


@pytest.mark.parametrize("frame", ("left", "world"))
def test_expanded_samples_equal_the_live_permitted_inputs(
    prepared: PreparedTrainingContent, blocks: _Blocks, frame: str
) -> None:
    rollout = blocks.rollouts[0]
    rows = replay._replay_rows(rollout.transitions)
    config = qmix.QMIXConfig(
        rollout_length=4,
        buffer_size=10,
        min_buffer_size=6,
        sample_sequence_length=4,
        sample_batch_size=24,
        spawn_frame=frame,
    )
    batch = replay._expand_sample(rows, config)
    transitions = rollout.transitions
    live = jax.vmap(
        jax.vmap(build_team_actor_input, in_axes=(0, None)), in_axes=(0, None)
    )(transitions.observations, 0)
    mask = transitions.action_mask
    actions = rows.actions
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
    np.testing.assert_array_equal(batch.rewards, rows.task_reward + rows.shaping_reward)
    assert bool(
        jnp.all(jnp.take_along_axis(batch.action_mask, batch.actions[..., None], -1))
    )
    del prepared
