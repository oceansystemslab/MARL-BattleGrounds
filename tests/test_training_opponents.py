"""Check frozen training opponents, update boundaries and native System execution.

CPU proofs cover independent opponent streams, the 80/20 reset distribution,
the separate permanent pin, rolling-copy retirement and stable capture IDs,
actual capture spacing, dynamic copy weights, zero-capacity play, immutable
complete actor variables, shared result normalization, selected memory resets,
stochastic initialization limits and untrained MAPPO on public inputs. These
tests do not train a learner or establish GPU cost or learned competence.
"""

from collections.abc import Callable
from dataclasses import replace
from functools import partial
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
    publication_valid,
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


def _bank(*, batch: int = 4, count: int = 3, pin: bool = False) -> OpponentHistory:
    history = init_opponent_history(_variables(), num_envs=batch, pin_first_update=pin)
    return history._replace(
        count=jnp.int32(count),
        eligible=jnp.arange(21) < count,
        captured_ids=jnp.where(jnp.arange(21) < count, jnp.arange(21), -1).astype(
            jnp.int32
        ),
        next_capture_id=jnp.int32(max(count, 0)),
        pinned_variables=_variables(99, 7) if pin else None,
        pinned_update=jnp.int32(1 if pin else -1),
        historical_variables={
            "weight": jnp.arange(21, dtype=jnp.int32),
            "inference": jnp.arange(21, dtype=jnp.int32) + 5,
        },
    )


def test_empty_bank_has_no_draw_and_reserves_only_capacity_axis() -> None:
    variables = {"weights": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)}
    history = init_opponent_history(variables, num_envs=4)
    assert history.current_variables["weights"] is variables["weights"]
    assert history.historical_variables["weights"].shape == (21, 2, 3)
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


def test_pinned_share_reaches_the_traced_program_only_when_positive() -> None:
    # The default and an explicit 0.0 trace one program by construction. That
    # this program is the pre-change program rests on the diff and on the
    # package's GPU equivalence job, not on this test; the guarded half is that
    # a positive share changes the traced program.
    history = _bank(count=3)
    mask = jnp.asarray((True, False, True, True))
    keys = jax.random.split(jax.random.key(7), 4)
    default = str(jax.make_jaxpr(assign_opponents)(history, mask, keys))
    explicit = str(
        jax.make_jaxpr(partial(assign_opponents, pinned_share=0.0))(history, mask, keys)
    )
    pinned = str(
        jax.make_jaxpr(partial(assign_opponents, pinned_share=0.1))(history, mask, keys)
    )
    assert default == explicit
    assert default != pinned


@pytest.mark.parametrize("count", [1, 2, 4, 20])
@pytest.mark.parametrize("legacy", [False, True])
def test_pinned_draws_match_independent_scalar_key_oracle(
    count: int, legacy: bool
) -> None:
    history = _bank(count=count, pin=True)
    root = jax.random.PRNGKey(5) if legacy else jax.random.key(5)
    keys = training_keys(root, jnp.arange(4, dtype=jnp.int32), stream="opponent")
    mask = jnp.asarray((True, True, False, True))
    result = _jit(partial(assign_opponents, pinned_share=0.1))(history, mask, keys)
    expected: list[int] = []
    for selected, key in zip(mask, keys, strict=True):
        chance_key, slot_key = jax.random.split(key)
        chance = np.float32(jax.random.uniform(chance_key))
        if not bool(selected):
            expected.append(-1)
        elif chance < np.float32(0.1):
            expected.append(-2)
        elif chance < np.float32(0.1 + 0.2):
            expected.append(int(jax.random.randint(slot_key, (), 0, count)))
        else:
            expected.append(-1)
    np.testing.assert_array_equal(result.lane_snapshot, expected)
    assert not bool(result.error)


def test_seeded_pinned_frequencies_match_declared_shares() -> None:
    draw = _jit(partial(assign_opponents, pinned_share=0.1))
    result = draw(
        _bank(batch=20_000, count=4, pin=True),
        jnp.ones(20_000, jnp.bool_),
        jax.random.split(jax.random.key(93), 20_000),
    )
    shares = np.bincount(np.asarray(result.lane_snapshot) + 2, minlength=6) / 20_000
    assert 0.085 < shares[0] < 0.115
    assert 0.68 < shares[1] < 0.72
    assert np.all((shares[2:] > 0.04) & (shares[2:] < 0.06))
    single = draw(
        _bank(batch=20_000, count=1, pin=True),
        jnp.ones(20_000, jnp.bool_),
        jax.random.split(jax.random.key(94), 20_000),
    )
    values = np.asarray(single.lane_snapshot)
    assert set(np.unique(values).tolist()) <= {-2, -1, 0}
    assert 0.68 < np.mean(values == -1) < 0.72


def test_empty_bank_draws_nothing_even_with_a_pinned_share() -> None:
    history = init_opponent_history(_variables(), num_envs=4)
    assigned = _jit(partial(assign_opponents, pinned_share=0.1))(
        history, jnp.ones(4, jnp.bool_), jax.random.split(jax.random.key(3), 4)
    )
    _tree_equal(assigned, history)


@pytest.mark.parametrize(
    "share,error",
    [
        (True, TypeError),
        ("0.1", TypeError),
        (jnp.float32(0.1), TypeError),
        (0.9, ValueError),
        (-0.1, ValueError),
        (float("nan"), ValueError),
    ],
)
def test_pinned_share_contract_rejects_bool_range_and_nonpython_values(
    share: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        assign_opponents(
            _bank(),
            jnp.ones(4, jnp.bool_),
            jax.random.split(jax.random.key(2), 4),
            pinned_share=cast(Any, share),
        )


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
        history.historical_variables["weight"][:20], np.arange(1, 21)
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
        _bank(count=3)._replace(
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


def test_rotation_protects_both_teams_and_keeps_stable_capture_ids() -> None:
    schedule = make_training_schedule(total_env_steps=400, num_envs=4, curriculum=False)
    arrays = schedule.arrays._replace(
        history_threshold_rounds=jnp.pad(
            jnp.arange(10, 61, 10, dtype=jnp.int32), (0, 14)
        ),
        history_threshold_count=jnp.int32(6),
    )
    history = init_opponent_history(
        _variables(), num_envs=4, keep_past=2, minimum_capture_rounds=10
    )
    refresh = _jit(refresh_opponents)
    for index, rounds in enumerate((10, 20, 30), 1):
        if index == 2:
            history = history._replace(
                lane_snapshot=jnp.asarray((0, -1, -1, -1), jnp.int32)
            )
        history, event = refresh(
            history,
            _variables(rounds, rounds + 1),
            completed_rounds=jnp.int32(rounds),
            update_index=jnp.int32(index),
            schedule=arrays,
        )
        assert bool(event.created) and int(event.capture_id) == index - 1
    np.testing.assert_array_equal(history.eligible, (False, True, True))
    assert int(history.historical_variables["weight"][0]) == 10
    assert int(history.lane_snapshot[0]) == 0
    previous = history
    history, event = refresh(
        history,
        _variables(40),
        completed_rounds=jnp.int32(40),
        update_index=jnp.int32(4),
        schedule=arrays,
    )
    assert not bool(event.created) and not bool(history.error)
    assert int(history.current_variables["weight"]) == 40
    _tree_equal(history.historical_variables, previous.historical_variables)
    history = history._replace(lane_snapshot=jnp.full(4, -1, jnp.int32))
    partners = jnp.asarray((True, False, False))
    assert bool(
        publication_valid(
            history,
            completed_rounds=jnp.int32(50),
            update_index=jnp.int32(5),
            schedule=arrays,
            live_slots=partners,
        )
    )
    history, event = refresh(
        history,
        _variables(50),
        completed_rounds=jnp.int32(50),
        update_index=jnp.int32(5),
        schedule=arrays,
        live_slots=partners,
    )
    assert not bool(event.created) and not bool(history.error)
    history, event = refresh(
        history,
        _variables(60, 61),
        completed_rounds=jnp.int32(60),
        update_index=jnp.int32(6),
        schedule=arrays,
        live_slots=jnp.zeros(3, jnp.bool_),
    )
    assert bool(event.created) and int(event.capture_id) == 3 and int(event.slot) == 0
    assert int(event.rounds) == 60 and int(event.update_index) == 6
    assert int(history.historical_variables["inference"][0]) == 61
    np.testing.assert_array_equal(history.eligible, (True, False, True))
    np.testing.assert_array_equal(history.captured_ids, (3, 1, 2))
    np.testing.assert_array_equal(history.threshold_to_snapshot[:6], (0, 1, 2, 3, 3, 3))
    assert history.historical_variables["weight"].shape == (3,)
    assert int(history.count) == 3


def test_recurring_capture_uses_actual_time_and_keeps_newest_window() -> None:
    schedule = make_training_schedule(total_env_steps=400, num_envs=4, curriculum=False)
    arrays = schedule.arrays._replace(history_threshold_count=jnp.int32(0))
    history = init_opponent_history(
        _variables(),
        num_envs=4,
        keep_past=2,
        minimum_capture_rounds=10,
        capture_interval_rounds=10,
    )
    refresh = _jit(refresh_opponents)
    captures: list[tuple[int, int]] = []
    for index, rounds in enumerate(range(8, 97, 8), 1):
        history, event = refresh(
            history,
            _variables(rounds, rounds + 1),
            completed_rounds=jnp.int32(rounds),
            update_index=jnp.int32(index),
            schedule=arrays,
        )
        assert not bool(history.error)
        if bool(event.created):
            captures.append((int(event.capture_id), int(event.rounds)))
        assert np.count_nonzero(history.eligible) <= 2
    assert captures == [(0, 16), (1, 32), (2, 48), (3, 64), (4, 80), (5, 96)]
    assert set(np.asarray(history.captured_ids)[np.asarray(history.eligible)]) == {4, 5}
    assert set(np.asarray(history.captured_rounds)[np.asarray(history.eligible)]) == {
        80,
        96,
    }
    assert history.historical_variables["weight"].shape == (3,)
    assert int(history.next_capture_id) == 6


def test_explicit_capture_waits_for_minimum_gap_without_rejecting_updates() -> None:
    schedule = make_training_schedule(total_env_steps=80, num_envs=4, curriculum=False)
    arrays = schedule.arrays._replace(
        history_threshold_rounds=jnp.pad(jnp.asarray((1, 2), jnp.int32), (0, 18)),
        history_threshold_count=jnp.int32(2),
    )
    history = init_opponent_history(
        _variables(), num_envs=4, keep_past=1, minimum_capture_rounds=10
    )
    events: list[bool] = []
    for index, rounds in enumerate((1, 2, 11), 1):
        history, event = _jit(refresh_opponents)(
            history,
            _variables(rounds),
            completed_rounds=jnp.int32(rounds),
            update_index=jnp.int32(index),
            schedule=arrays,
        )
        events.append(bool(event.created))
        assert not bool(history.error)
        assert int(history.current_variables["weight"]) == rounds
    assert events == [True, False, True]
    np.testing.assert_array_equal(history.threshold_to_snapshot[:2], (0, 1))
    np.testing.assert_array_equal(history.captured_rounds, (1, 11))


def test_zero_history_preserves_separate_first_update_pin_and_executes_it(
    public_setup: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = public_setup
    schedule = make_training_schedule(total_env_steps=80, num_envs=4, curriculum=False)
    history = init_opponent_history(
        _variables(), num_envs=4, keep_past=0, pin_first_update=True
    )
    refresh = _jit(refresh_opponents)
    for index in (1, 2):
        history, event = refresh(
            history,
            _variables(index * 3, index),
            completed_rounds=jnp.int32(index),
            update_index=jnp.int32(index),
            schedule=schedule.arrays,
        )
        assert not bool(event.created) and not bool(history.error)
    assert history.historical_variables["weight"].shape == (0,)
    _tree_equal(history.pinned_variables, _variables(3, 1))
    assert int(history.pinned_update) == 1 and int(history.next_capture_id) == 0
    keys = jax.random.split(jax.random.key(77), 4)
    chosen = _jit(partial(assign_opponents, pinned_share=0.8))(
        history, jnp.ones(4, jnp.bool_), keys
    )
    assert not bool(chosen.error)
    assert set(np.asarray(chosen.lane_snapshot)) <= {-2, -1}
    history = history._replace(lane_snapshot=jnp.asarray((-2, -1, -2, -1), jnp.int32))
    wrapped = make_opponent_system(_actor())
    assert wrapped.init is not None
    inputs = env.policy_inputs(observations, state, team=1)
    memory = _jit(wrapped.init)(history, inputs, keys)
    actual = _jit(wrapped.apply)(history, memory, inputs, keys)
    for lane in range(4):
        variables = (
            history.pinned_variables if lane % 2 == 0 else history.current_variables
        )
        expected = _apply(
            variables, _row(memory, lane), _row(inputs, lane), keys[lane : lane + 1]
        )
        _tree_equal(_row(actual.actions, lane), expected.actions)
    empty = init_opponent_history(_variables(), num_envs=4, keep_past=0)
    empty, event = refresh(
        empty,
        _variables(9),
        completed_rounds=jnp.int32(1),
        update_index=jnp.int32(1),
        schedule=schedule.arrays,
    )
    selected = assign_opponents(empty, jnp.ones(4, jnp.bool_), keys, past_share=1.0)
    assert not bool(event.created) and not bool(selected.error)
    np.testing.assert_array_equal(selected.lane_snapshot, -np.ones(4))


def test_dynamic_copy_weights_change_choices_without_retracing() -> None:
    history = _bank(count=3)
    keys = jax.random.split(jax.random.key(87), 4)
    traces: list[int] = []

    def draw(value: OpponentHistory, weights: Array) -> OpponentHistory:
        traces.append(1)
        return assign_opponents(
            value,
            jnp.ones(4, jnp.bool_),
            keys,
            past_share=jnp.float32(1),
            past_weights=weights,
        )

    compiled = _jit(draw)
    weights = jnp.zeros(21, jnp.float32)
    first = compiled(history, weights.at[1].set(1))
    second = compiled(history, weights.at[2].set(1))
    retired = compiled(
        history._replace(eligible=history.eligible.at[1].set(False)),
        weights.at[1].set(1),
    )
    assert len(traces) == 1
    np.testing.assert_array_equal(first.lane_snapshot, np.ones(4))
    np.testing.assert_array_equal(second.lane_snapshot, np.full(4, 2))
    np.testing.assert_array_equal(retired.lane_snapshot, -np.ones(4))
    invalid = compiled(history, weights.at[1].set(float("nan")))
    assert bool(invalid.error)
    _tree_equal(invalid._replace(error=history.error), history)


def test_external_pin_and_reused_slots_have_distinct_stable_counter_rows() -> None:
    from marl_battlegrounds.training.opponents import (
        opponent_counter_row,
        opponent_identity,
    )

    history = init_opponent_history(
        _variables(), num_envs=4, keep_past=1, external_pin=True
    )
    schedule = make_training_schedule(total_env_steps=80, num_envs=4).arrays
    for index in range(1, 6):
        history, event = refresh_opponents(
            history,
            _variables(index),
            completed_rounds=jnp.int32(index),
            update_index=jnp.int32(index),
            schedule=schedule,
        )
        assert bool(event.created) and not bool(history.error)
    assert history.historical_variables["weight"].shape == (2,)
    slot = int(np.flatnonzero(np.asarray(history.captured_ids) == 4)[0])
    history = history._replace(
        lane_snapshot=jnp.asarray([-1, -2, slot, slot], jnp.int32)
    )
    identities, versions = _jit(opponent_identity)(history)
    np.testing.assert_array_equal(identities, [-1, -2, 4, 4])
    np.testing.assert_array_equal(versions, [5, -2, 5, 5])
    np.testing.assert_array_equal(opponent_counter_row(identities), [0, 1, 6, 6])
    empty = init_opponent_history(
        _variables(), num_envs=4, keep_past=0, external_pin=True
    )
    chosen = assign_opponents(
        empty,
        jnp.ones(4, jnp.bool_),
        jax.random.split(jax.random.key(5), 4),
        pinned_share=0.8,
        past_share=0.0,
    )
    assert not bool(chosen.error)
    assert -2 in np.asarray(chosen.lane_snapshot)


def test_named_order_counts_only_actual_starts_and_updates_future_choices() -> None:
    from marl_battlegrounds.training.opponents import (
        configure_opponent_selection,
        opponent_selection_names,
        select_opponents,
    )

    assert opponent_selection_names({"self": 2, "script": 1}) == ("self", "script")
    assert opponent_selection_names(["script", "self", "script"]) == ("script", "self")
    invalid_values: tuple[object, ...] = (
        {},
        [],
        {"self": -1},
        {"self": 0},
        {"self": float("nan")},
    )
    for invalid in invalid_values:
        with pytest.raises(ValueError):
            opponent_selection_names(invalid)
    history = init_opponent_history(
        _variables(), num_envs=4, keep_past=0, external_pin=True
    )
    settings = configure_opponent_selection(
        ("left", "right"), history, ["left", "self", "right"]
    )
    choose = _jit(select_opponents)
    keys = jax.random.split(jax.random.key(7), 4)
    history, settings = choose(history, settings, jnp.ones(4, jnp.bool_), keys)
    np.testing.assert_array_equal(history.lane_snapshot, [-2, -1, -2, -2])
    np.testing.assert_array_equal(settings.choices, [0, 0, 1, 0])
    assert int(settings.game_starts) == 4
    history, settings = choose(
        history, settings, jnp.asarray([False, True, False, True]), keys
    )
    np.testing.assert_array_equal(history.lane_snapshot, [-2, -1, -2, -2])
    np.testing.assert_array_equal(settings.choices, [0, 0, 1, 1])
    assert int(settings.game_starts) == 6
    before = history, settings
    history, settings = choose(history, settings, jnp.zeros(4, jnp.bool_), keys)
    _tree_equal((history, settings), before)
    changed = configure_opponent_selection(
        ("left", "right"), history, ["right", "self"], previous=settings
    )
    history, changed = choose(
        history, changed, jnp.asarray([True, False, False, False]), keys
    )
    assert int(changed.choices[0]) == 1
    assert int(changed.game_starts) == 7
    weighted = configure_opponent_selection(
        ("left", "right"), history, {"right": 1}, previous=changed
    )
    history, weighted = choose(history, weighted, jnp.ones(4, jnp.bool_), keys)
    np.testing.assert_array_equal(history.lane_snapshot, [-2] * 4)
    np.testing.assert_array_equal(weighted.choices, [1] * 4)


def test_generic_past_weights_bind_capture_ids_and_new_copies_default_uniform() -> None:
    from marl_battlegrounds.training.opponents import (
        configure_opponent_selection,
        select_opponents,
    )

    history = init_opponent_history(
        _variables(), num_envs=2, keep_past=2, capture_interval_rounds=1
    )
    schedule = make_training_schedule(total_env_steps=20, num_envs=2).arrays
    for index in (1, 2):
        history, event = refresh_opponents(
            history,
            _variables(index),
            completed_rounds=jnp.int32(index),
            update_index=jnp.int32(index),
            schedule=schedule,
        )
        assert bool(event.created)
    settings = configure_opponent_selection((), history, {"past": 1}, past={0: 1, 1: 0})
    keys = jax.random.split(jax.random.key(91), 2)
    selected, _ = _jit(select_opponents)(
        history, settings, jnp.ones(2, jnp.bool_), keys
    )
    np.testing.assert_array_equal(selected.captured_ids[selected.lane_snapshot], [0, 0])
    history, event = refresh_opponents(
        history,
        _variables(3),
        completed_rounds=jnp.int32(3),
        update_index=jnp.int32(3),
        schedule=schedule,
    )
    assert int(event.capture_id) == 2
    selected, _ = _jit(select_opponents)(
        history, settings, jnp.ones(2, jnp.bool_), keys
    )
    # ID 0 retired, ID 1 has zero weight, and new ID 2 gets the default weight.
    np.testing.assert_array_equal(selected.captured_ids[selected.lane_snapshot], [2, 2])
    with pytest.raises(ValueError, match="eligible"):
        configure_opponent_selection((), history, {"past": 1}, past={0: 1})


def test_all_external_games_skip_native_actor_work(
    public_setup: tuple[Environment, Observations, EnvironmentState],
) -> None:
    from marl_battlegrounds.evaluation.policy_execution import system_inputs

    _, observations, state = public_setup
    calls: list[int] = []

    def note(value: Array) -> None:
        calls.append(int(value))

    def counted(
        variables: Tree, memory: Array, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        jax.debug.callback(note, jnp.sum(inputs.valid))
        return _apply(variables, memory, inputs, keys)

    actor, pinned = replace(_actor(), apply=counted), _actor()
    history = init_opponent_history(
        actor.variables, num_envs=4, keep_past=0, external_pin=True
    )._replace(lane_snapshot=jnp.full(4, -2, jnp.int32))
    wrapper = make_opponent_system(actor, pinned=pinned)
    inputs = system_inputs(observations, state, team=1)
    keys = jax.random.split(jax.random.key(99), 4)
    assert wrapper.init is not None
    variables = (history, pinned.variables, ())
    memory = wrapper.init(variables, inputs, keys)
    output = _jit(wrapper.apply)(variables, memory, inputs, keys)
    jax.block_until_ready(output)
    jax.effects_barrier()
    assert calls == []
    _tree_equal(output.next_memory[0], memory[0])
    expected = pinned.apply(pinned.variables, memory[1], inputs, keys)
    _tree_equal(output.actions, expected.actions)
