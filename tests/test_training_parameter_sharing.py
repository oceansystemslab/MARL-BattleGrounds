"""Check grouped actors through real collection, learner updates and history.

CPU tests use the public repeated-class and unequal-roster curriculum setup.
Every built-in method must update its owned actor groups, retain absent local
optimizer counts, pass saved-boundary validation and run a mixed current/history
collection. The history assignment is a numerical test fixture; it does not
claim an episode reset or a new sampling rule. These checks prove integration,
not learned behavior or GPU speed.
"""

# pyright: reportPrivateUsage=false
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.baselines import ppo, pqn, qmix
from marl_battlegrounds.evaluation.policy_execution import SystemOutput, system_inputs
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    learner,
    make_training_schedule,
    pqn_learner,
    prepare_training_content,
    qmix_learner,
    scan_training_rollout,
)
from marl_battlegrounds.training.collection import _failed

type Tree = Any


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


def _row(tree: Tree, lane: int) -> Tree:
    def take(value: Array) -> Array:
        return value[lane : lane + 1]

    return jax.tree.map(take, tree)


@pytest.mark.parametrize(
    ("method", "sharing"),
    [
        (name, "class")
        for name in ("mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn")
    ]
    + [("ff_ippo", "none")],
)
def test_grouped_learners_update_and_collect_mixed_frozen_history(
    prepared: PreparedTrainingContent, method: str, sharing: str
) -> None:
    schedule = make_training_schedule(
        total_env_steps=20,
        num_envs=2,
        curriculum=[
            {
                "share": 1.0,
                "maps": [0],
                "rosters": {
                    "system": ["mage", "mage", "priest"],
                    "opponent": ["warrior", "hunter"],
                },
            }
        ],
    )
    settings: Tree
    state: Tree
    updater: Callable[..., Tree]
    validate: Callable[..., None]
    options: dict[str, Any] = dict(
        schedule=schedule,
        seed=42,
        prepared=prepared,
        metrics="none",
        keep_past=1,
        history_capture_capacity=2,
    )
    if method == "qmix":
        settings = qmix.QMIXConfig(
            rollout_length=2,
            buffer_size=4,
            min_buffer_size=2,
            sample_sequence_length=2,
            sample_batch_size=2,
            epochs=1,
            parameter_sharing=sharing,
        )
        collection, state = qmix_learner.init_qmix_learner(**options, qmix=settings)
        updater = partial(qmix_learner.update_qmix_learner, qmix=settings)
        validate = partial(qmix_learner.validate_qmix_learner, qmix=settings)
    elif method == "pqn_vdn":
        settings = pqn.PQNConfig(
            rollout_length=2,
            memory_window=1,
            epochs=1,
            num_minibatches=1,
            parameter_sharing=sharing,
        )
        planned = pqn.pqn_planned_learning_blocks(10, settings)
        collection, state = pqn_learner.init_pqn_learner(**options, pqn=settings)
        updater = partial(
            pqn_learner.update_pqn_learner,
            pqn=settings,
            planned_learning_blocks=planned,
        )
        validate = partial(pqn_learner.validate_pqn_learner, pqn=settings)
    else:
        settings = ppo.PPOConfig(
            rollout_length=2,
            epochs=1,
            minibatches=1,
            groups=1,
            parameter_sharing=sharing,
        )
        collection, state = learner.init_learner(**options, ppo=settings, method=method)
        updater = partial(learner.update_learner, ppo=settings, method=method)
        validate = partial(learner.validate_learner, ppo=settings, method=method)
    assert not collection.vectorize_opponent_lanes
    validate(collection, state)
    update = cast(Callable[..., Tree], jax.jit(updater))
    for length in (2, 1, 2) if method == "pqn_vdn" else (2,):
        collect = cast(
            Callable[..., Tree],
            jax.jit(partial(scan_training_rollout, collection, length=length)),
        )
        carry, rollout = collect(state.carry)
        state, result = update(state, carry, rollout)
        assert not bool(result.failed)
    assert int(state.completed_updates) == 1
    validate(collection, state)
    opt = (
        state.actor_opt_state
        if method not in {"qmix", "pqn_vdn"}
        else (state.opt_state[0] if method == "qmix" else state.opt_state)
    )
    counts = [
        np.asarray(value)
        for value in jax.tree.leaves(opt)
        if jnp.issubdtype(value.dtype, jnp.integer)
    ]
    assert counts and all(value.shape == (5,) for value in counts)
    assert all(
        np.count_nonzero(value) == (3 if sharing == "none" else 2) for value in counts
    )
    expected = np.zeros(5, np.int32)
    if sharing == "none":
        expected[:3] = 1
    else:
        classes = np.asarray(state.carry.source_class_ids)[:, :3]
        expected[np.unique(classes) - 1] = 1
    for value in counts:
        np.testing.assert_array_equal(value, expected)
    if method == "qmix":
        unused = int(np.flatnonzero(expected == 0)[0])

        def changed_target(value: Array) -> Array:
            return value.at[unused].add(0.1)

        bad_target = state._replace(
            target_q_params=jax.tree.map(changed_target, state.target_q_params)
        )
        with pytest.raises(ValueError, match="targets differ"):
            validate(collection, bad_target)
    history = state.carry.history
    slots = np.flatnonzero(np.asarray(history.captured_ids) >= 0)
    assert len(slots) > 0
    slot = int(slots[0])

    def change_snapshot(bank: Array) -> Array:
        return bank.at[slot].multiply(jnp.asarray(0.9, bank.dtype))

    history = history._replace(
        lane_snapshot=jnp.asarray([slot, -1], jnp.int32),
        historical_variables=jax.tree.map(
            change_snapshot, history.historical_variables
        ),
    )
    carry = state.carry._replace(history=history)
    inputs = system_inputs(carry.observations, carry.state, team=1)
    keys = jax.random.split(jax.random.key(94), 2)
    mixed = cast(
        SystemOutput,
        jax.jit(collection.opponent.apply)(history, carry.memory.team_b, inputs, keys),
    )
    for lane in range(2):

        def frozen(bank: Array) -> Array:
            return bank[slot]

        variables = (
            jax.tree.map(frozen, history.historical_variables)
            if lane == 0
            else history.current_variables
        )
        expected = cast(
            SystemOutput,
            collection.actor.apply(
                variables,
                _row(carry.memory.team_b, lane),
                _row(inputs, lane),
                keys[lane : lane + 1],
            ),
        )
        for actual, wanted in zip(
            jax.tree.leaves((_row(mixed.actions, lane), _row(mixed.next_memory, lane))),
            jax.tree.leaves((expected.actions, expected.next_memory)),
            strict=True,
        ):
            np.testing.assert_allclose(actual, wanted, rtol=3e-5, atol=2e-6)
    collect_mixed = cast(
        Callable[..., Tree],
        jax.jit(partial(scan_training_rollout, collection, length=1)),
    )
    after, rollout = collect_mixed(carry)
    assert not bool(_failed(after))
    assert int(rollout.real_steps) == 1
    assert set(np.asarray(rollout.transitions.opponent_snapshot).ravel()) == {
        -1,
        int(history.captured_ids[slot]),
    }


def test_grouped_pqn_count_admission_keeps_local_clocks_consistent() -> None:
    check = cast(
        Callable[[Tree, Array], Array],
        jax.jit(partial(pqn_learner._count_leaves_equal, grouped=True)),
    )
    local = jnp.asarray([3, 0, 1, 0, 0], jnp.int32)
    assert bool(check((local, local), jnp.int32(3)))
    for bad in (local.at[0].set(4), local.at[1].set(-1)):
        assert not bool(check((bad, bad), jnp.int32(3)))
    assert not bool(check((local, local.at[2].set(0)), jnp.int32(3)))
    largest = jnp.asarray([np.iinfo(np.int32).max] * 5, jnp.int32)
    assert bool(check((largest, largest), jnp.int32(np.iinfo(np.int32).max)))
