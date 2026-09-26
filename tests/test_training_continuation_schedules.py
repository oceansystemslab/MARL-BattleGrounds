"""Check declared child learner schedules without a training run.

These CPU checks cover all six method names, partial block clocks, remaining
PQN warmup, repeated extensions, original decay horizons, exact terminal-rate
call intervals, strict future changes and unchanged small RAdam state trees.
They prove schedule and optimizer wiring, not learning or GPU cost.
"""

from dataclasses import asdict, replace
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds.baselines.pqn import PQNConfig, pqn_optimizer
from marl_battlegrounds.training._continuation_schedules import (
    ContinuationExploration,
    LearnerContinuation,
    LinearSchedule,
    continuation_boundary,
    continuation_counts,
    continuation_state,
    learner_continuation,
    resolve_learner_continuation,
)

_SETTINGS = {
    "rollout_length": 4,
    "memory_window": 2,
    "epochs": 2,
    "num_minibatches": 2,
    "minibatches": 2,
    "min_buffer_size": 6,
    "lr_linear_decay": True,
    "q_lr": 0.00025,
}


def _pqn(
    *,
    rounds: int = 10,
    blocks: int = 3,
    learning: int = 1,
    total: int = 16,
    changes: dict[str, Any] | None = None,
    parent: dict[str, object] | None = None,
) -> dict[str, object]:
    return resolve_learner_continuation(
        method="pqn_vdn",
        settings=_SETTINGS,
        counters={
            "env_steps": rounds * 4,
            "updates": learning * 4,
            "completed_blocks": blocks,
            "learning_blocks": learning,
        },
        num_envs=4,
        total_env_steps=total * 4,
        original_total_env_steps=16 * 4,
        changes=changes or {},
        parent=parent,
    )


@pytest.mark.parametrize("method", ["mappo", "ippo", "ff_mappo", "ff_ippo", "qmix"])
def test_partial_fork_keeps_the_actual_block_origin(method: str) -> None:
    context = LearnerContinuation(method, 7, 2, 1 if method == "qmix" else 2)
    assert continuation_boundary(
        8, total_rounds=14, continuation=context, rollout_length=4, minimum=6
    ) == (11, 3, 2 if method == "qmix" else 3)
    assert continuation_boundary(
        14, total_rounds=14, continuation=context, rollout_length=4, minimum=6
    ) == (14, 4, 3 if method == "qmix" else 4)
    assert continuation_counts(
        7, continuation=context, rollout_length=4, minimum=6
    ) == (2, 1 if method == "qmix" else 2)


def test_pqn_partial_parent_does_not_merge_an_already_learned_block() -> None:
    context = LearnerContinuation("pqn_vdn", 13, 4, 2, 3)
    assert continuation_boundary(
        14, total_rounds=20, continuation=context, rollout_length=4, initial_rounds=6
    ) == (17, 5, 3)
    assert continuation_boundary(
        20, total_rounds=20, continuation=context, rollout_length=4, initial_rounds=6
    ) == (20, 6, 4)


def test_remaining_warmup_can_end_partway_then_continue_once() -> None:
    first = LearnerContinuation("pqn_vdn", 4, 1, 0, 3)
    assert continuation_boundary(
        5, total_rounds=5, continuation=first, rollout_length=4, initial_rounds=6
    ) == (5, 2, 0)
    second = replace(first, start_rounds=5, start_blocks=2)
    assert continuation_boundary(
        6, total_rounds=11, continuation=second, rollout_length=4, initial_rounds=6
    ) == (6, 3, 0)
    assert continuation_boundary(
        7, total_rounds=11, continuation=second, rollout_length=4, initial_rounds=6
    ) == (10, 4, 1)
    assert continuation_counts(
        11, continuation=second, rollout_length=4, initial_rounds=6
    ) == (5, 2)


def test_pqn_keeps_original_horizon_and_distinguishes_the_last_boundary() -> None:
    first = _pqn(total=16)
    context = learner_continuation(first)
    assert context is not None
    assert context.pqn_planned_learning_blocks == 3
    assert continuation_counts(
        16, continuation=context, rollout_length=4, initial_rounds=6
    ) == (5, 3)
    with pytest.raises(ValueError, match="terminal learning rate"):
        _pqn(total=19)
    with pytest.raises(ValueError, match="terminal learning rate"):
        _pqn(rounds=16, blocks=5, learning=3, total=17)


def test_explicit_future_rule_and_permission_survive_a_second_extension() -> None:
    first = _pqn(
        total=20,
        changes={
            "learning_rate": {"kind": "linear", "q_lr": 0.002, "optimizer_steps": 4}
        },
    )
    context = learner_continuation(first)
    assert context is not None and context.pqn_rate == LinearSchedule(
        4, 0.002, 1e-10, 4
    )
    assert context.allow_terminal_rate
    second = learner_continuation(
        _pqn(rounds=20, blocks=6, learning=4, total=25, parent=first)
    )
    assert second is not None
    assert second.pqn_rate == context.pqn_rate
    assert second.pqn_planned_learning_blocks == 3
    assert second.start_rounds == 20
    assert second.start_blocks == 6
    assert second.allow_terminal_rate
    keep = _pqn(total=20, changes={"learning_rate": {"kind": "keep_terminal_rate"}})
    kept = learner_continuation(keep)
    assert kept is not None and kept.allow_terminal_rate


@pytest.mark.parametrize(
    "changes",
    [
        {"rollout_length": 8},
        {"learning_rate": {"kind": "constant", "q_lr": True}},
        {"learning_rate": {"kind": "linear", "q_lr": 0.1, "optimizer_steps": 0}},
        {"learning_rate": {"kind": "keep_terminal_rate", "q_lr": 0.1}},
        {
            "exploration": {
                "kind": "linear",
                "epsilon": 1,
                "end_epsilon": 0.1,
                "env_steps": 3,
            }
        },
        {"exploration": {"kind": "constant", "epsilon": float("nan")}},
    ],
)
def test_invalid_amendments_fail_without_arrays(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _pqn(changes=changes)


def test_no_optimizer_calls_do_not_need_terminal_permission() -> None:
    value = _pqn(rounds=4, blocks=1, learning=0, total=5)
    context = learner_continuation(value)
    assert context is not None and not context.allow_terminal_rate


def test_qmix_exploration_avoids_a_large_absolute_transition_product() -> None:
    context = resolve_learner_continuation(
        method="qmix",
        settings=_SETTINGS,
        counters={
            "env_steps": 3_000_000_512,
            "updates": 10,
            "completed_blocks": 6,
            "learning_blocks": 5,
        },
        num_envs=1024,
        total_env_steps=3_000_000_512 + 4096,
        changes={
            "exploration": {
                "kind": "linear",
                "epsilon": 0.6,
                "end_epsilon": 0.1,
                "env_steps": 2048,
            }
        },
    )
    parsed = learner_continuation(context)
    assert parsed is not None and parsed.exploration is not None
    hook = ContinuationExploration(parsed.exploration, 1024)
    compiled = jax.jit(hook.rate)
    for offset, expected in ((0, 0.6), (1, 0.35), (2, 0.1), (100, 0.1)):
        np.testing.assert_allclose(
            cast(jax.Array, compiled(jnp.int32(3_000_000_512 // 1024 + offset))),
            expected,
            rtol=2e-6,
        )


@pytest.mark.parametrize("decay", [False, True])
def test_future_pqn_rate_keeps_the_original_optimizer_tree_and_moments(
    decay: bool,
) -> None:
    config = PQNConfig(epochs=1, num_minibatches=1, lr_linear_decay=decay)
    params = {"q": jnp.asarray([0.2, -0.3], jnp.float32)}
    grads = {"q": jnp.asarray([0.4, -0.8], jnp.float32)}
    original = pqn_optimizer(config, 8)
    state = original.init(params)
    for _ in range(3):
        _, state = original.update(grads, state, params)
    rule = LinearSchedule(3, 0.002, 1e-10, 4)
    future = pqn_optimizer(
        config, 8, learning_rate=rule.at_array, optimizer_count=jnp.int32(3)
    )
    assert jax.tree.structure(future.init(params)) == jax.tree.structure(state)
    delta_old, next_old = original.update(grads, state, params)
    delta_new, next_new = future.update(grads, state, params)
    for old, new in zip(
        jax.tree.leaves(next_old), jax.tree.leaves(next_new), strict=True
    ):
        np.testing.assert_array_equal(old, new)
    old_rate = config.q_lr if not decay else (config.q_lr - 1e-10) * (1 - 3 / 8) + 1e-10
    np.testing.assert_allclose(
        delta_new["q"], delta_old["q"] * (0.002 / old_rate), rtol=2e-6
    )


def test_schedule_round_trip_and_host_device_values_agree() -> None:
    rule = LinearSchedule(3, 0.002, 1e-10, 4)
    compiled = jax.jit(rule.at_array)
    for count in (0, 3, 4, 6, 7, 100):
        np.testing.assert_allclose(
            cast(jax.Array, compiled(jnp.int32(count))), rule.at(count), rtol=2e-6
        )
    context = LearnerContinuation("pqn_vdn", 10, 3, 1, 3, pqn_rate=rule)
    assert learner_continuation({"schema_version": 1, **asdict(context)}) == context


@pytest.mark.parametrize("decay", [False, True])
def test_original_terminal_rate_requires_permission_even_before_the_horizon(
    decay: bool,
) -> None:
    with pytest.raises(ValueError, match="terminal learning rate"):
        resolve_learner_continuation(
            method="pqn_vdn",
            settings={**_SETTINGS, "q_lr": 1e-10, "lr_linear_decay": decay},
            counters={
                "env_steps": 40,
                "updates": 4,
                "completed_blocks": 3,
                "learning_blocks": 1,
            },
            num_envs=4,
            total_env_steps=64,
            original_total_env_steps=64,
            changes={},
        )


def test_inactive_history_thresholds_never_request_or_consume_a_snapshot() -> None:
    from marl_battlegrounds.training.curriculum import make_training_schedule
    from marl_battlegrounds.training.opponents import (
        init_opponent_history,
        refresh_opponents,
    )

    schedule = make_training_schedule(total_env_steps=8, num_envs=2).arrays._replace(
        history_threshold_rounds=jnp.asarray([1] + [0] * 19, jnp.int32),
        history_threshold_count=jnp.int32(1),
    )
    variables = {"weight": jnp.asarray([1.0], jnp.float32)}
    history = init_opponent_history(variables, num_envs=2)
    first, capture = refresh_opponents(
        history,
        variables,
        completed_rounds=jnp.int32(1),
        update_index=jnp.int32(1),
        schedule=schedule,
    )
    assert bool(capture.created) and int(first.count) == 1
    np.testing.assert_array_equal(capture.threshold_mask, [True] + [False] * 19)
    second, absent = refresh_opponents(
        first,
        variables,
        completed_rounds=jnp.int32(2),
        update_index=jnp.int32(2),
        schedule=schedule,
    )
    assert not bool(absent.created) and int(second.count) == 1
    np.testing.assert_array_equal(
        first.historical_variables["weight"], second.historical_variables["weight"]
    )


@pytest.mark.parametrize("method", ["qmix", "pqn_vdn"])
@pytest.mark.parametrize("committed", [False, True])
def test_changed_epsilon_keeps_saved_placement_and_other_state(
    method: str, committed: bool
) -> None:
    class Variables(NamedTuple):
        epsilon: jax.Array
        parameters: jax.Array

    class History(NamedTuple):
        current_variables: Variables
        historical_variables: Variables

    class Progress(NamedTuple):
        rounds: jax.Array

    class Carry(NamedTuple):
        history: History
        progress: Progress

    class State(NamedTuple):
        carry: Carry
        learning_blocks: jax.Array

    epsilon = jnp.float32(0.9)
    if committed:
        epsilon = jax.device_put(epsilon, cast(Any, jax.devices("cpu")[0]))
    parameters = jnp.asarray([1.0, 2.0], jnp.float32)
    frozen = Variables(jnp.float32(0.8), parameters)
    state = State(
        Carry(History(Variables(epsilon, parameters), frozen), Progress(jnp.int32(10))),
        jnp.int32(3),
    )
    origin = 20 if method == "qmix" else 2
    rule = LinearSchedule(origin, 0.37, 0.04, 17)
    context = LearnerContinuation(method, 10, 3, 3, exploration=rule)
    updated = continuation_state(state, context, num_envs=2)
    current = updated.carry.history.current_variables
    assert current.epsilon.committed == committed
    assert current.epsilon.sharding == epsilon.sharding
    np.testing.assert_allclose(
        current.epsilon, rule.at(20 if method == "qmix" else 3), rtol=2e-6
    )
    assert current.parameters is parameters
    assert updated.carry.history.historical_variables is frozen
    assert updated.learning_blocks is state.learning_blocks
    assert continuation_state(state, None, num_envs=2) is state
