"""Qualify compiled public workflows with 32 environments on CPU or CUDA.

Compare native batches with external vmap and scan with explicit steps. Check
dynamic System values, episode memory, partial resets and selected no-file replay
capture. Every environment execution, including references, uses 32 lanes.
These checks prove correctness for this workload, not simulation throughput.
"""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import Action, DoneFlags, EnvConfig, Reward
from marl_battlegrounds.environment import EnvironmentState, EpisodeInfo
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
    SystemState,
    apply_systems,
    init_systems,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

type StepResult = tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]
type Step = Callable[[Array, EnvironmentState, Action], StepResult]

_BATCH = 32


def _config(max_steps: int = 3) -> EnvConfig:
    roster = ("warrior", "mage", "priest", "rogue", "hunter")
    return make_standard_team_deathmatch_config(
        map_id=12,
        max_steps=max_steps,
        team_a_roster=roster,
        team_b_roster=roster,
    )


def _assert_tree_equal(actual: object, expected: object) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        a, b = np.asarray(left), np.asarray(right)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


def test_native_and_external_batch_match_through_jit_and_scan() -> None:
    config = _config()
    native = marl_bgs.make("tdm", num_envs=_BATCH, metrics="none")
    scalar = marl_bgs.make("tdm", metrics="none")
    ids = jnp.arange(1, _BATCH + 1, dtype=jnp.int32)
    root = jax.random.key(17)
    keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(root, ids)
    native_obs, native_state = native.reset(root, config, episode_id=ids)

    def reset_lane(
        key: Array, episode_id: Array
    ) -> tuple[Observations, EnvironmentState]:
        return scalar.reset(key, config, episode_id=episode_id)

    external_obs, external_state = cast(
        tuple[Observations, EnvironmentState], jax.jit(jax.vmap(reset_lane))(keys, ids)
    )
    _assert_tree_equal(native_obs, external_obs)
    _assert_tree_equal(native_state.core_state, external_state.core_state)
    zeros = jnp.zeros((_BATCH, 10), jnp.int32)
    actions = Action(jnp.ones_like(zeros), zeros, zeros)
    step_keys = jax.random.split(jax.random.key(18), 4)
    external_step = cast(Step, jax.jit(jax.vmap(scalar.step)))

    def advance(
        state: EnvironmentState, key: Array
    ) -> tuple[EnvironmentState, PolicyTree]:
        obs, state, reward, done, info = native.step(key, state, actions)
        return state, (obs, reward, done, info.completed)

    @jax.jit
    def scan(state: EnvironmentState, keys: Array) -> PolicyTree:
        return jax.lax.scan(advance, state, keys)

    scanned, history = cast(PolicyTree, scan(native_state, step_keys))
    for tick in range(4):
        lane_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
            step_keys[tick], ids
        )
        obs, external_state, reward, done, info = external_step(
            lane_keys, external_state, actions
        )

        def at_tick(value: Array, index: int = tick) -> Array:
            return value[index]

        _assert_tree_equal(
            jax.tree.map(at_tick, history),
            (obs, reward, done, info.completed),
        )
    _assert_tree_equal(scanned.core_state, external_state.core_state)
    np.testing.assert_array_equal(scanned.core_state.step_count, 3)
    np.testing.assert_array_equal(
        history[3],
        np.broadcast_to(np.array([False, False, True, False])[:, None], (4, _BATCH)),
    )
    assert scanned.priority is None and scanned.full is None


def test_dynamic_system_scan_preserves_memory_and_partial_resets() -> None:
    env = marl_bgs.make("tdm", num_envs=_BATCH, metrics="none")

    def broadcast(value: ArrayLike) -> Array:
        array = jnp.asarray(value)
        return jnp.broadcast_to(array, (_BATCH, *array.shape))

    config = jax.tree.map(broadcast, _config())
    lengths = np.tile(np.array([1, 3], np.int32), _BATCH // 2)
    config = config._replace(max_steps=jnp.asarray(lengths))
    observations, state = env.reset(jax.random.key(20), config)
    traces: list[int] = []

    def initialize(variables: PolicyTree, inputs: SystemInput, keys: Array) -> Array:
        del variables, keys
        return jnp.zeros(inputs.valid.shape, jnp.int32)

    def advance(
        variables: Array, memory: Array, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        del keys
        zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
        return SystemOutput(
            ActorAction(inputs.active_mask.astype(jnp.int32) * variables, zero, zero),
            memory + inputs.valid.astype(jnp.int32),
            learning_outputs=variables * inputs.valid,
        )

    method = System("counter", advance, init=initialize)
    memory = init_systems(method, method, observations, state, jax.random.key(21))

    @jax.jit
    def run(
        initial: tuple[Observations, EnvironmentState, SystemState],
        weight: Array,
        key: Array,
    ) -> PolicyTree:
        traces.append(1)

        def step(carry: PolicyTree, tick: Array) -> tuple[PolicyTree, PolicyTree]:
            obs, current, mem = carry
            actor_key, step_key, reset_key = jax.random.split(
                jax.random.fold_in(key, tick), 3
            )
            actions, mem, outputs = apply_systems(
                method,
                method,
                mem,
                obs,
                current,
                actor_key,
                variables_a=weight,
                variables_b=weight + 1,
            )
            _, following, _, done, info = env.step(step_key, current, actions)
            next_obs, following_reset = env.reset_done(reset_key, following)
            history = (
                mem.team_a,
                mem.team_b,
                following.core_state.step_count,
                following_reset.core_state.step_count,
                done.done,
                info.completed,
                following.episode_id,
                following_reset.episode_id,
                mem.policy_trace.decision_step,
                outputs,
                actions.move,
            )
            return (next_obs, following_reset, mem), history

        return jax.lax.scan(step, initial, jnp.arange(4, dtype=jnp.int32))

    # Changed carried memory belongs to the same active episodes. The next reset
    # clears it only in the lanes that completed a game.
    offsets = jnp.arange(_BATCH, dtype=jnp.int32) + 4
    changed = memory._replace(team_a=offsets, team_b=offsets + 100)
    results = (
        (
            cast(
                PolicyTree,
                run((observations, state, memory), jnp.int32(1), jax.random.key(22)),
            ),
            1,
            0,
            0,
        ),
        (
            cast(
                PolicyTree,
                run((observations, state, changed), jnp.int32(2), jax.random.key(23)),
            ),
            2,
            np.asarray(offsets),
            np.asarray(offsets + 100),
        ),
    )
    expected_steps = np.stack([tick % lengths + 1 for tick in range(4)])
    completed = expected_steps == lengths
    for result, weight, offset_a, offset_b in results:
        final, history = result
        fresh = np.arange(4)[:, None] >= lengths
        np.testing.assert_array_equal(
            history[0], expected_steps + np.where(fresh, 0, offset_a)
        )
        np.testing.assert_array_equal(
            history[1], expected_steps + np.where(fresh, 0, offset_b)
        )
        np.testing.assert_array_equal(history[2], expected_steps)
        np.testing.assert_array_equal(
            history[3], np.where(completed, 0, expected_steps)
        )
        np.testing.assert_array_equal(history[4], completed)
        np.testing.assert_array_equal(history[5], completed)
        np.testing.assert_array_equal(history[6] != history[7], completed)
        np.testing.assert_array_equal(history[8], expected_steps - 1)
        np.testing.assert_array_equal(history[9][0], weight)
        np.testing.assert_array_equal(history[9][1], weight + 1)
        np.testing.assert_array_equal(history[10][..., :5], weight)
        np.testing.assert_array_equal(history[10][..., 5:], weight + 1)
        assert not np.any(final[1].lifecycle_error)
    assert traces == [1]


def test_selected_replay_evaluation_needs_no_metrics_or_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = marl_bgs.evaluate(
        "random",
        "random",
        num_episodes=64,
        maps=[_config()],
        num_envs=_BATCH,
        metrics="none",
        replay_episodes=[2, 33],
        chunk_size=16,
        spawn_mode="default",
    )
    assert result.metadata["num_envs"] == _BATCH
    assert result.completed_episode_ids == tuple(range(1, 65))
    assert result.priority_metrics == result.full_metrics == {}
    assert result.paths is None
    assert len(result.replays) == 2
    for replay, episode_id in zip(result.replays, (2, 33), strict=True):
        assert replay.header.context.identity.episode_id.endswith(
            f"episode-{episode_id}"
        )
        assert replay.header.context.identity.run_id == result.metadata["run_id"]
        assert replay.header.context.seed_protocol.episode_seed == episode_id
        assert len(replay.frames) == 4
        assert len(replay.transitions) == 3
        assert (
            replay.header.context.policy_assignments[0].assignment_status == "assigned"
        )
    assert list(tmp_path.iterdir()) == []
