"""Check frozen training opponents, update boundaries and native System execution.

CPU proofs cover independent opponent streams, the 80/20 reset distribution,
immutable complete actor variables, shared result normalization, selected memory
resets, stochastic initialization limits and untrained MAPPO on public inputs.
These tests do not train a learner or establish GPU cost or learned competence.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.baselines.ppo import (
    initialize_ppo,
    make_recurrent_mappo_system,
)
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import TASK_MODE_TDM
from marl_battlegrounds.environment import Environment, EnvironmentState, make
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
    apply_systems,
    independent_policies,
    init_systems,
    policy,
    shared_policy,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.training.curriculum import make_training_schedule
from marl_battlegrounds.training.distributions import TrainingStream, training_keys
from marl_battlegrounds.training.opponents import (
    OpponentHistory,
    SnapshotEvent,
    assign_opponents,
    init_opponent_history,
    make_opponent_system,
    refresh_opponents,
)

type Tree = Any


def _jit[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    return cast(Callable[P, R], jax.jit(function))


def _variables(weight: int = 2, inference: int = 1) -> dict[str, Array]:
    return {"weight": jnp.int32(weight), "inference": jnp.int32(inference)}


def _tree_equal(actual: Tree, expected: Tree) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, right)


def _row(tree: Tree, lane: int) -> Tree:
    def row(value: Array) -> Array:
        return value[lane : lane + 1]

    return jax.tree.map(row, tree)


def _stored(tree: Tree, slot: int) -> Tree:
    def row(value: Array) -> Array:
        return value[slot]

    return jax.tree.map(row, tree)


def _plus_one(value: Array) -> Array:
    return value + 1


def _small_random(key: Array) -> Array:
    return jax.random.randint(key, (5,), 0, 2)


def _uniform(key: Array) -> Array:
    return jax.random.uniform(key, (5,))


def _custom_reset(old: Tree, new: Tree, mask: Array) -> Tree:
    del old, mask
    return new


def _scalar_memory(weights: Tree, view: SystemInput, rng: Array) -> Array:
    del weights, view, rng
    return jnp.int32(0)


def _initial(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    del keys
    return jnp.broadcast_to(
        variables["weight"] * 10 + variables["inference"], inputs.active_mask.shape
    )


def _apply(
    variables: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    draw = jax.vmap(_small_random)(keys)
    moves = (variables["weight"] + variables["inference"] + draw) % 9
    legal = jnp.take_along_axis(inputs.action_mask.move_mask, moves[..., None], -1)[
        ..., 0
    ]
    moves = jnp.where(legal & inputs.valid[:, None], moves, 0)
    zero = jnp.zeros_like(moves)
    return SystemOutput(
        ActorAction(moves, zero, zero),
        memory + inputs.valid[:, None].astype(jnp.int32),
        learning_outputs={"probability": jnp.ones(inputs.active_mask.shape)},
        policy_ids=jnp.zeros(inputs.active_mask.shape, jnp.int32),
    )


def _actor() -> System:
    return System(
        "Tiny Actor",
        _apply,
        variables=_variables(),
        init=_initial,
        components=({"name": "Tiny Actor"},),
    )


@pytest.fixture(scope="module")
def public_setup() -> tuple[Environment, Observations, EnvironmentState]:
    env = make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(2, 2), max_steps=20),
        num_envs=4,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(40))
    return env, observations, state


def _bank(*, batch: int = 4, count: int = 3) -> OpponentHistory:
    history = init_opponent_history(_variables(), num_envs=batch)
    return history._replace(
        count=jnp.int32(count),
        historical_variables={
            "weight": jnp.arange(20, dtype=jnp.int32),
            "inference": jnp.arange(20, dtype=jnp.int32) + 5,
        },
    )


def test_empty_bank_has_no_draw_and_reserves_only_capacity_axis() -> None:
    variables = {"weights": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)}
    history = init_opponent_history(variables, num_envs=4)
    assert history.current_variables["weights"] is variables["weights"]
    assert history.historical_variables["weights"].shape == (20, 2, 3)
    assert not np.any(history.historical_variables["weights"])
    np.testing.assert_array_equal(history.lane_snapshot, -np.ones(4))
    np.testing.assert_array_equal(history.threshold_to_snapshot, -np.ones(20))
    assigned = _jit(assign_opponents)(
        history, jnp.ones(4, jnp.bool_), jax.random.split(jax.random.key(1), 4)
    )
    _tree_equal(assigned, history)
    assert not bool(history.error)


@pytest.mark.parametrize("count", [1, 3, 20])
@pytest.mark.parametrize("legacy", [False, True])
def test_reset_draws_match_independent_scalar_key_oracle(
    count: int, legacy: bool
) -> None:
    history = _bank(count=count)
    root = jax.random.PRNGKey(3) if legacy else jax.random.key(3)
    keys = training_keys(root, jnp.arange(4, dtype=jnp.int32), stream="opponent")
    mask = jnp.asarray((True, False, True, True))
    result = _jit(assign_opponents)(history, mask, keys)
    expected: list[int] = []
    for selected, key in zip(mask, keys, strict=True):
        chance, slot = jax.random.split(key)
        expected.append(
            int(jax.random.randint(slot, (), 0, count))
            if bool(selected) and bool(jax.random.bernoulli(chance, 0.2))
            else -1
        )
    np.testing.assert_array_equal(result.lane_snapshot, expected)


def test_seeded_frequencies_are_consistent_with_eighty_twenty_and_uniform_history() -> (
    None
):
    history = _bank(batch=20_000, count=4)
    result = _jit(assign_opponents)(
        history,
        jnp.ones(20_000, jnp.bool_),
        jax.random.split(jax.random.key(91), 20_000),
    )
    counts = np.bincount(np.asarray(result.lane_snapshot) + 1, minlength=5)
    assert 0.78 < counts[0] / 20_000 < 0.82
    assert np.all((counts[1:] / 20_000 > 0.04) & (counts[1:] / 20_000 < 0.06))


def test_one_reset_key_or_mask_cannot_change_other_assignments_or_streams() -> None:
    history = _bank()
    root = jax.random.key(120)
    generations = jnp.zeros(4, jnp.int32)
    reset = jnp.ones(4, jnp.bool_)
    keys = training_keys(root, generations, stream="opponent")
    first = assign_opponents(history, reset, keys)
    altered = assign_opponents(history, reset, keys.at[2].set(jax.random.key(900)))
    np.testing.assert_array_equal(
        first.lane_snapshot[jnp.array([0, 1, 3])],
        altered.lane_snapshot[jnp.array([0, 1, 3])],
    )
    masked = assign_opponents(first, reset.at[2].set(False), keys)
    _tree_equal(masked, first)
    advanced = generations.at[2].add(1)
    streams: tuple[TrainingStream, ...] = ("map", "roster", "action")
    for stream in streams:
        extra = {"decision_step": jnp.zeros(4, jnp.int32)} if stream == "action" else {}
        before = training_keys(root, generations, stream=stream, **extra)
        after = training_keys(root, advanced, stream=stream, **extra)
        np.testing.assert_array_equal(
            jax.random.key_data(before)[jnp.asarray([0, 1, 3])],
            jax.random.key_data(after)[jnp.asarray([0, 1, 3])],
        )


def test_refresh_merges_tied_thresholds_and_preserves_all_prior_slots() -> None:
    schedule = make_training_schedule(total_env_steps=12, num_envs=4, curriculum=False)
    history = init_opponent_history(_variables(), num_envs=4)
    refresh = _jit(refresh_opponents)
    for step in range(1, 4):
        previous = history
        history, event = refresh(
            history,
            _variables(step + 2, step + 5),
            completed_rounds=jnp.int32(step),
            update_index=jnp.int32(step),
            schedule=schedule.arrays,
        )
        assert bool(event.created)
        assert int(event.slot) == step - 1
        assert int(event.rounds) == step
        assert int(event.update_index) == step
        assert int(history.count) == step
        for name in history.historical_variables:
            np.testing.assert_array_equal(
                history.historical_variables[name][: step - 1],
                previous.historical_variables[name][: step - 1],
            )
        np.testing.assert_array_equal(history.lane_snapshot, previous.lane_snapshot)
    np.testing.assert_array_equal(
        history.threshold_to_snapshot,
        np.ceil(np.arange(1, 21) * 3 / 20).astype(int) - 1,
    )
    np.testing.assert_array_equal(history.captured_rounds[:3], [1, 2, 3])
    assert int(history.count) == 3


def test_each_five_percent_update_can_fill_twenty_unique_slots() -> None:
    schedule = make_training_schedule(total_env_steps=80, num_envs=4, curriculum=False)
    history = init_opponent_history(_variables(), num_envs=4)

    def update(
        prior: OpponentHistory, index: Array
    ) -> tuple[OpponentHistory, SnapshotEvent]:
        return refresh_opponents(
            prior,
            {"weight": index, "inference": index + 7},
            completed_rounds=index,
            update_index=index,
            schedule=schedule.arrays,
        )

    def updates(value: OpponentHistory) -> tuple[OpponentHistory, SnapshotEvent]:
        return jax.lax.scan(update, value, jnp.arange(1, 21, dtype=jnp.int32))

    history, events = _jit(updates)(history)
    assert int(history.count) == 20
    assert not bool(history.error)
    np.testing.assert_array_equal(events.slot, np.arange(20))
    np.testing.assert_array_equal(history.threshold_to_snapshot, np.arange(20))
    np.testing.assert_array_equal(
        history.historical_variables["weight"], np.arange(1, 21)
    )


def test_valid_update_without_capture_and_final_partial_boundary() -> None:
    schedule = make_training_schedule(total_env_steps=404, num_envs=4, curriculum=False)
    history = init_opponent_history(_variables(), num_envs=4)
    history, event = refresh_opponents(
        history,
        _variables(4),
        completed_rounds=jnp.int32(1),
        update_index=jnp.int32(1),
        schedule=schedule.arrays,
    )
    assert not bool(event.created)
    assert (int(event.slot), int(event.rounds), int(event.update_index)) == (-1, -1, -1)
    assert not np.any(event.threshold_mask)
    assert int(history.current_variables["weight"]) == 4
    history, event = refresh_opponents(
        history,
        _variables(8),
        completed_rounds=jnp.int32(101),
        update_index=jnp.int32(2),
        schedule=schedule.arrays,
    )
    assert bool(event.created)
    assert int(history.count) == 1
    assert int(event.rounds) == 101
    assert np.all(event.threshold_mask)
    np.testing.assert_array_equal(history.threshold_to_snapshot, np.zeros(20))


@pytest.mark.parametrize("rounds,index", [(0, 1), (101, 1), (1, 0), (1, 2), (-1, 1)])
def test_invalid_update_is_sticky_and_preserves_weights_assignments_and_timing(
    rounds: int, index: int
) -> None:
    history = _bank()
    schedule = make_training_schedule(total_env_steps=400, num_envs=4, curriculum=False)
    failed, event = _jit(refresh_opponents)(
        history,
        _variables(8),
        completed_rounds=jnp.int32(rounds),
        update_index=jnp.int32(index),
        schedule=schedule.arrays,
    )
    assert bool(failed.error)
    assert not bool(event.created)
    _tree_equal(failed._replace(error=history.error), history)
    later, _ = refresh_opponents(
        failed,
        _variables(6),
        completed_rounds=jnp.int32(1),
        update_index=jnp.int32(1),
        schedule=schedule.arrays,
    )
    _tree_equal(later, failed)


@pytest.mark.parametrize("slot,count", [(3, 3), (-2, 3), (0, 0), (-1, 21), (-1, -1)])
def test_invalid_assignment_is_preserved_and_cannot_be_repaired(
    slot: int, count: int
) -> None:
    history = _bank(count=count)._replace(lane_snapshot=jnp.full(4, slot, jnp.int32))
    result = _jit(assign_opponents)(
        history, jnp.ones(4, jnp.bool_), jax.random.split(jax.random.key(2), 4)
    )
    assert bool(result.error)
    _tree_equal(result._replace(error=history.error), history)


@pytest.mark.parametrize("form", ["two", "three", "named"])
@pytest.mark.parametrize("assignment", [(-1, -1, -1, -1), (0, 2, 1, 0), (0, -1, 2, -1)])
def test_wrapper_forms_match_independent_selected_application_and_initialization(
    public_setup: tuple[Environment, Observations, EnvironmentState],
    form: str,
    assignment: tuple[int, ...],
) -> None:
    env, observations, state = public_setup
    inputs = env.policy_inputs(observations, state, team=1)
    keys = jax.random.split(jax.random.key(501), 4)

    def apply(variables: Tree, memory: Array, view: SystemInput, rng: Array) -> Tree:
        output = _apply(variables, memory, view, rng)
        if form == "two":
            return output.actions, output.next_memory
        if form == "three":
            return output.actions, output.next_memory, output.policy_ids
        return output

    actor = replace(_actor(), apply=apply)
    wrapped = make_opponent_system(actor)
    history = _bank()._replace(lane_snapshot=jnp.array(assignment, jnp.int32))
    assert wrapped.init is not None
    memory = _jit(wrapped.init)(history, inputs, keys)
    actual = _jit(wrapped.apply)(history, memory, inputs, keys)
    assert actual.learning_outputs == ()
    assert actual.policy_ids is None
    for lane, slot in enumerate(assignment):
        variables = (
            history.current_variables
            if slot == -1
            else _stored(history.historical_variables, slot)
        )
        expected_memory = _initial(variables, _row(inputs, lane), keys[lane : lane + 1])
        _tree_equal(_row(memory, lane), expected_memory)
        expected = _apply(
            variables, expected_memory, _row(inputs, lane), keys[lane : lane + 1]
        )
        _tree_equal(_row(actual.actions, lane), expected.actions)
        _tree_equal(_row(actual.next_memory, lane), expected.next_memory)


def test_current_refresh_preserves_frozen_inference_state_and_continuing_memory(
    public_setup: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = public_setup
    inputs = env.policy_inputs(observations, state, team=1)
    actor = _actor()
    wrapped = make_opponent_system(actor)
    history = _bank()._replace(lane_snapshot=jnp.asarray((0, -1, 1, -1), jnp.int32))
    keys = jax.random.split(jax.random.key(60), 4)
    memory = jnp.arange(20, dtype=jnp.int32).reshape(4, 5)
    apply = _jit(wrapped.apply)
    before = apply(history, memory, inputs, keys)
    schedule = make_training_schedule(total_env_steps=400, num_envs=4, curriculum=False)
    changed, _ = refresh_opponents(
        history,
        _variables(5, 3),
        completed_rounds=jnp.int32(2),
        update_index=jnp.int32(1),
        schedule=schedule.arrays,
    )
    after = apply(changed, memory, inputs, keys)
    np.testing.assert_array_equal(
        after.actions.move[jnp.asarray([0, 2])],
        before.actions.move[jnp.asarray([0, 2])],
    )
    assert not np.array_equal(
        after.actions.move[jnp.asarray([1, 3])],
        before.actions.move[jnp.asarray([1, 3])],
    )
    np.testing.assert_array_equal(after.next_memory, memory + 1)
    np.testing.assert_array_equal(after.next_memory, before.next_memory)


def test_weight_history_count_and_assignments_change_without_retracing(
    public_setup: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = public_setup
    inputs = env.policy_inputs(observations, state, team=1)
    wrapped = make_opponent_system(_actor())
    traces: list[int] = []

    def apply(history: OpponentHistory, memory: Array, keys: Array) -> SystemOutput:
        traces.append(1)
        return wrapped.apply(history, memory, inputs, keys)

    compiled = _jit(apply)
    history = _bank(count=1)
    memory = jnp.zeros((4, 5), jnp.int32)
    keys = jax.random.split(jax.random.key(99), 4)
    compiled(history, memory, keys)
    compiled(
        history._replace(
            count=jnp.int32(3),
            current_variables=_variables(7),
            lane_snapshot=jnp.asarray((1, -1, 2, 0), jnp.int32),
            historical_variables=jax.tree.map(_plus_one, history.historical_variables),
        ),
        memory,
        jax.random.split(jax.random.key(100), 4),
    )
    assert len(traces) == 1


@pytest.mark.parametrize("kind", ["host", "reset", "shared", "independent", "object"])
def test_wrapper_rejects_unsupported_systems_at_setup(kind: str) -> None:
    actor = _actor()
    if kind == "host":
        actor = replace(actor, execution="host")
    elif kind == "reset":
        actor = replace(actor, reset_memory=_custom_reset)
    elif kind == "shared":
        actor = shared_policy(policy("random"))
    elif kind == "independent":
        actor = independent_policies((policy("random"), policy("random")))
    else:
        actor = replace(actor, variables={"wrong": object()})
    with pytest.raises((TypeError, ValueError)):
        make_opponent_system(actor)


def test_all_current_executes_one_full_batch_call_and_mixed_executes_assigned_lanes(
    public_setup: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = public_setup
    inputs = env.policy_inputs(observations, state, team=1)
    calls: list[int] = []

    def record(valid: Array) -> None:
        calls.append(valid.shape[0])

    def observed(
        variables: Tree, memory: Array, view: SystemInput, keys: Array
    ) -> SystemOutput:
        jax.debug.callback(record, view.valid)
        return _apply(variables, memory, view, keys)

    wrapped = make_opponent_system(replace(_actor(), apply=observed))
    run = _jit(wrapped.apply)
    history = _bank()
    keys = jax.random.split(jax.random.key(311), 4)
    memory = jnp.zeros((4, 5), jnp.int32)
    jax.block_until_ready(run(history, memory, inputs, keys))
    jax.effects_barrier()
    assert calls == [4]
    calls.clear()
    history = history._replace(lane_snapshot=jnp.asarray((0, -1, 1, 2), jnp.int32))
    jax.block_until_ready(run(history, memory, inputs, keys))
    jax.effects_barrier()
    assert calls == [1, 1, 1, 1]


def test_public_death_and_respawn_keep_frozen_assignment_and_recurrent_memory() -> None:
    config = evaluation_env_config(
        team_sizes=(1, 1),
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=20,
        max_steps=20,
    )._replace(
        spawn_shield_duration_steps=0,
        team_respawn_wave_period_step_count=jnp.asarray((2, 2), jnp.int32),
    )
    initial, *_ = core.reset(config, jax.random.key(50))
    authored = initial._replace(
        step_count=jnp.int32(5),
        agent_positions=initial.agent_positions.at[0]
        .set(jnp.asarray((4.0, 4.0)))
        .at[5]
        .set(jnp.asarray((6.5, 4.0))),
        current_health=initial.current_health.at[0].set(1.0).at[5].set(1.0),
        team_respawn_wave_countdowns=jnp.asarray((1, 1), jnp.int32),
    )
    prepared = core.initialize_scenario_state(authored, config)
    env = make("tdm", env_config=config, metrics="none")
    observations, state = env.reset(jax.random.key(51), initial=prepared[:3])

    def fight(
        variables: Tree, memory: Array, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        output = _apply(variables, memory, inputs, keys)
        zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
        targets = jnp.where(inputs.action_mask.select_target_mask[..., 6], 6, 0)
        return SystemOutput(ActorAction(zero, targets, zero), output.next_memory)

    actor = replace(_actor(), apply=fight)
    wrapped = make_opponent_system(actor)
    history = _bank(batch=1)._replace(lane_snapshot=jnp.zeros(1, jnp.int32))
    memory = init_systems(
        actor, wrapped, observations, state, jax.random.key(52), variables_b=history
    )
    for decision in range(3):
        before = memory.team_b
        actions, memory, _ = apply_systems(
            actor,
            wrapped,
            memory,
            observations,
            state,
            jax.random.key(60 + decision),
            variables_b=history,
        )
        np.testing.assert_array_equal(memory.team_b, before + 1)
        observations, state, *_ = env.step(
            jax.random.key(70 + decision), state, actions
        )
        if decision < 2:
            np.testing.assert_array_equal(
                state.core_state.alive_mask[jnp.asarray((0, 5))],
                (decision == 1, decision == 1),
            )
        assert int(history.lane_snapshot[0]) == 0
        assert int(state.reset_generation) == 0


def test_existing_policy_adapter_route_still_runs(
    public_setup: tuple[Environment, Observations, EnvironmentState],
) -> None:
    _, observations, state = public_setup
    actor = shared_policy(policy("random"))
    memory = init_systems(actor, actor, observations, state, jax.random.key(8))
    actions, _, _ = apply_systems(
        actor, actor, memory, observations, state, jax.random.key(9)
    )
    assert actions.move.shape == (4, 10)


def test_static_shape_key_memory_and_variable_failures_are_clear(
    public_setup: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = public_setup
    inputs = env.policy_inputs(observations, state, team=1)
    history = _bank()
    keys = jax.random.split(jax.random.key(2), 4)
    with pytest.raises(TypeError, match="Python integer"):
        init_opponent_history(_variables(), num_envs=True)
    with pytest.raises(ValueError, match="positive"):
        init_opponent_history(_variables(), num_envs=0)
    with pytest.raises(TypeError, match="array leaves"):
        init_opponent_history({"weight": None}, num_envs=4)
    with pytest.raises(ValueError, match="shape"):
        assign_opponents(history, jnp.ones(3, jnp.bool_), keys)
    with pytest.raises(TypeError, match="dtype"):
        assign_opponents(history, jnp.ones(4, jnp.int32), keys)
    with pytest.raises(ValueError, match="Threefry"):
        assign_opponents(
            history,
            jnp.ones(4, jnp.bool_),
            jax.random.split(jax.random.key(1, impl="rbg"), 4),
        )
    wrapped = make_opponent_system(_actor())
    assert wrapped.init is not None
    with pytest.raises(ValueError, match="snapshot ID"):
        wrapped.init(
            history._replace(lane_snapshot=jnp.full(4, 19, jnp.int32)), inputs, keys
        )
    with pytest.raises(ValueError, match="match the actor tree"):
        wrapped.init(
            history._replace(current_variables={"weight": jnp.int32(2)}), inputs, keys
        )
    bad = make_opponent_system(replace(_actor(), init=_scalar_memory))
    assert bad.init is not None
    with pytest.raises(ValueError, match="leading lane axis"):
        bad.init(history, inputs, keys)


def test_public_partial_reset_keeps_current_and_historical_memories_separate(
    public_setup: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, _, initial = public_setup
    observations, state = env.reset(
        jax.random.key(12),
        initial.config._replace(max_steps=jnp.asarray((1, 3, 3, 3), jnp.int32)),
    )
    actor = _actor()
    wrapped = make_opponent_system(actor)
    history = _bank()._replace(lane_snapshot=jnp.asarray((0, -1, 1, -1), jnp.int32))
    memory = init_systems(
        actor, wrapped, observations, state, jax.random.key(13), variables_b=history
    )
    actions, memory, learning = apply_systems(
        actor,
        wrapped,
        memory,
        observations,
        state,
        jax.random.key(14),
        variables_b=history,
    )
    assert learning[1] == ()
    assert learning[0]
    assert not np.array_equal(memory.team_a[0], memory.team_b[0])
    observations, state, *_ = env.step(jax.random.key(15), state, actions)
    np.testing.assert_array_equal(state.done.done, (True, False, False, False))
    before = memory
    observations, state = env.reset_done(jax.random.key(16), state)
    history = history._replace(lane_snapshot=history.lane_snapshot.at[0].set(2))
    _, memory, _ = apply_systems(
        actor,
        wrapped,
        before,
        observations,
        state,
        jax.random.key(17),
        variables_b=history,
    )
    np.testing.assert_array_equal(memory.team_b[0], np.full(5, 28))
    np.testing.assert_array_equal(memory.team_b[1:], before.team_b[1:] + 1)
    np.testing.assert_array_equal(
        memory.policy_trace.policy_ids[:, 5:], -np.ones((4, 5))
    )


def test_m8_stochastic_initializer_keeps_declared_global_episode_id_limit(
    public_setup: tuple[Environment, Observations, EnvironmentState],
) -> None:
    _, observations, state = public_setup

    def stochastic(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
        del variables, inputs
        return jax.vmap(_uniform)(keys)

    actor = replace(_actor(), init=stochastic)
    wrapped = make_opponent_system(actor)
    history = init_opponent_history(actor.variables, num_envs=4)
    roots = training_keys(
        jax.random.key(19), jnp.ones(4, jnp.int32), stream="initialization"
    )
    first = init_systems(
        actor, wrapped, observations, state, roots, variables_b=history
    )
    changed = state._replace(episode_id=state.episode_id.at[2].add(1))
    second = init_systems(
        actor, wrapped, observations, changed, roots, variables_b=history
    )
    np.testing.assert_array_equal(
        first.team_b[jnp.asarray([0, 1, 3])], second.team_b[jnp.asarray([0, 1, 3])]
    )
    assert not np.array_equal(first.team_b[2], second.team_b[2])


def test_untrained_mappo_matches_public_permitted_inputs_and_keeps_rows_private(
    public_setup: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = public_setup
    actor = make_recurrent_mappo_system(initialize_ppo(jax.random.key(30)).actor_params)
    wrapped = make_opponent_system(actor)
    history = init_opponent_history(actor.variables, num_envs=4)
    inputs = env.policy_inputs(observations, state, team=1)
    keys = jax.random.split(jax.random.key(31), 4)
    assert actor.init is not None and wrapped.init is not None
    memory = actor.init(actor.variables, inputs, keys)
    expected = _jit(actor.apply)(actor.variables, memory, inputs, keys)
    actual = _jit(wrapped.apply)(history, memory, inputs, keys)
    _tree_equal(actual.actions, expected.actions)
    np.testing.assert_allclose(
        actual.next_memory, expected.next_memory, atol=2e-6, rtol=2e-6
    )
    assert actual.learning_outputs == ()

    def historical(leaf: Array) -> Array:
        return leaf.at[0].set(leaf[0] + 0.02)

    bank = jax.tree.map(historical, history.historical_variables)
    history = history._replace(
        count=jnp.int32(1),
        historical_variables=bank,
        lane_snapshot=jnp.asarray((0, -1, 0, -1), jnp.int32),
    )
    actual = _jit(wrapped.apply)(history, memory, inputs, keys)
    for lane in range(4):
        variables = history.current_variables if lane % 2 else _stored(bank, 0)
        expected = cast(
            SystemOutput,
            actor.apply(
                variables, _row(memory, lane), _row(inputs, lane), keys[lane : lane + 1]
            ),
        )
        _tree_equal(_row(actual.actions, lane), expected.actions)
        np.testing.assert_allclose(
            _row(actual.next_memory, lane), expected.next_memory, atol=2e-6, rtol=2e-6
        )
        selected = np.asarray(actual.actions.move[lane])
        assert np.all(
            np.asarray(inputs.action_mask.move_mask[lane])[np.arange(5), selected]
        )
