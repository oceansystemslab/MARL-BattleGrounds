"""Check raw JAX Systems, learning outputs, adapters and episode memory.

These CPU cases exercise scalar/native shapes, supplied information rights,
one-call result provenance, dynamic gradients and compiled reset trajectories.
They preserve the existing scalar Policy contract and do not qualify GPU speed.
"""

from dataclasses import FrozenInstanceError
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import TASK_MODE_TDM
from marl_battlegrounds.environment import Environment, EnvironmentState, make
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyExecution,
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
    SystemState,
    apply_policies,
    apply_systems,
    independent_policies,
    init_systems,
    initial_policy_carry,
    policy,
    shared_policy,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput, Observations


def _setup(
    *, batch: int | None = 2, max_steps: int = 8, sizes: tuple[int, int] = (2, 3)
) -> tuple[Environment, Observations, EnvironmentState]:
    env = make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=sizes, max_steps=max_steps),
        num_envs=batch,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(10))
    return env, observations, state


def _idle(inputs: SystemInput) -> ActorAction:
    zero = jnp.zeros(inputs.active_mask.shape, dtype=jnp.int32)
    return ActorAction(zero, zero, zero)


def _stateless(
    variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, PolicyTree]:
    del variables, keys
    return _idle(inputs), memory


def _counter_init(variables: PolicyTree, inputs: SystemInput, keys: Array) -> Array:
    del variables, keys
    return jnp.zeros(inputs.valid.shape, dtype=jnp.int32)


def _counter_apply(
    variables: PolicyTree, memory: Array, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, Array]:
    del variables, keys
    return _idle(inputs), memory + inputs.valid.astype(jnp.int32)


@pytest.mark.parametrize("form", ["two", "three", "named"])
@pytest.mark.parametrize("batch", [None, 1, 3])
def test_result_forms_keep_shapes_and_separate_learning_trees(
    form: str, batch: int | None
) -> None:
    _, observations, state = _setup(batch=batch)
    size = 1 if batch is None else batch

    def choose(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> PolicyTree:
        del variables, keys
        action = _idle(inputs)
        ids = jnp.where(inputs.active_mask, 0, -1).astype(jnp.int32)
        if form == "two":
            return action, memory
        if form == "three":
            return action, memory, ids
        return SystemOutput(
            action,
            memory,
            learning_outputs={"value": jnp.ones((size,), jnp.float32)},
            policy_ids=ids,
        )

    first = System("first", choose, components=({"name": "actor"},))
    second = System("second", _stateless)
    initial = init_systems(first, second, observations, state, jax.random.key(11))
    assert initial.team_a == initial.team_b == ()
    assert not np.any(initial.policy_trace.valid)
    actions, memory, outputs = apply_systems(
        first, second, initial, observations, state, jax.random.key(12)
    )
    assert actions.move.shape == ((10,) if batch is None else (batch, 10))
    assert actions.move.dtype == jnp.int32
    assert memory.policy_trace.policy_ids.shape == (size, 10)
    assert memory.policy_trace.episode_id.shape == (size,)
    assert outputs[1] == ()
    if form == "named":
        np.testing.assert_array_equal(outputs[0]["value"], np.ones(size))
    else:
        assert outputs[0] == ()
    expected_ids = np.full((size, 10), -1, dtype=np.int32)
    if form != "two":
        expected_ids[:, :2] = 0
    np.testing.assert_array_equal(memory.policy_trace.policy_ids, expected_ids)


def test_result_is_immutable_and_learning_outputs_keep_gradients_and_rng() -> None:
    _, observations, state = _setup()

    def choose(
        weights: Array, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        def sample(key: Array) -> Array:
            return jax.random.randint(key, (5,), 0, 2)

        choices = jax.vmap(sample)(keys)
        choices = jnp.where(inputs.active_mask, choices, 0)
        zero = jnp.zeros_like(choices)
        ids = jnp.where(inputs.active_mask, choices, -1)
        return SystemOutput(
            ActorAction(choices, zero, zero),
            memory,
            learning_outputs={"sample": choices, "loss": weights**2},
            policy_ids=ids,
        )

    first = System("learning", choose, components=({"name": "a"}, {"name": "b"}))
    second = System("idle", _stateless)
    initial = init_systems(first, second, observations, state, jax.random.key(4))

    def loss(weights: Array) -> Array:
        actions, memory, outputs = apply_systems(
            first,
            second,
            initial,
            observations,
            state,
            jax.random.key(5),
            variables_a=weights,
        )
        del actions, memory
        return outputs[0]["loss"].sum()

    weights = jnp.asarray((2.0, -3.0), jnp.float32)
    gradient = cast(Array, jax.jit(jax.grad(loss))(weights))
    np.testing.assert_array_equal(gradient, 2 * weights)
    actions, memory, outputs = apply_systems(
        first,
        second,
        initial,
        observations,
        state,
        jax.random.key(5),
        variables_a=weights,
    )
    np.testing.assert_array_equal(actions.move[:, :5], outputs[0]["sample"])
    np.testing.assert_array_equal(
        memory.policy_trace.policy_ids[:, :2], outputs[0]["sample"][:, :2]
    )
    result = SystemOutput(ActorAction(*(jnp.int32(0),) * 3), ())
    attribute = "next_memory"
    with pytest.raises((AttributeError, FrozenInstanceError)):
        setattr(result, attribute, jnp.int32(4))


@pytest.mark.parametrize("name", ["random", "tdm-alpha", "tdm-beta"])
def test_legacy_policy_application_keeps_supplied_keys_and_scalar_results(
    name: str,
) -> None:
    env, observations, state = _setup(batch=None)
    original = policy(name)
    keys = jax.random.split(jax.random.key(13), 10)
    empty = initial_policy_carry((), 1)
    actions, _, _ = apply_policies(
        original.apply,
        original.apply,
        original.variables,
        original.variables,
        empty,
        empty,
        observations,
        state.action_mask,
        keys,
    )
    expected = []
    for team in (0, 1):
        inputs = env.policy_inputs(observations, state, team=team)
        for actor in range(5):

            def row(value: Array, index: int = actor) -> Array:
                return value[0, index]

            actor_input = jax.tree.map(row, inputs.actors)
            mask = jax.tree.map(row, inputs.action_mask)
            action, _ = original.apply(
                original.variables, (), actor_input, mask, keys[5 * team + actor]
            )
            expected.append(action)
    expected_action = jax.tree.map(lambda *values: jnp.stack(values), *expected)
    for actual, wanted in zip(actions, expected_action, strict=True):
        np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("execution", ["jax", "host"])
def test_shared_and_independent_adapters_preserve_distinct_initial_memories(
    execution: PolicyExecution,
) -> None:
    env = make(
        "tdm",
        map_id=12,
        team_a_roster=("mage", "mage"),
        team_b_roster=("mage", "priest", "mage"),
        num_envs=2,
        metrics="none",
        balance_spawn_locations=False,
    )
    observations, state = env.reset(jax.random.key(10))

    def actor(
        variables: PolicyTree,
        memory: PolicyTree,
        inputs: ActorInput,
        mask: PolicyTree,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        del inputs, mask, key
        zero = jnp.int32(0)

        def increment(value: Array) -> Array:
            return value + 1

        return ActorAction(variables["move"], zero, zero), jax.tree.map(
            increment, memory
        )

    first_policy = Policy(
        "first", actor, {"move": jnp.int32(1)}, jnp.int32(7), execution=execution
    )
    second_policy = Policy(
        "second",
        actor,
        {"move": jnp.int32(2)},
        {"vector": jnp.asarray((10, 20))},
        execution=execution,
    )
    first = independent_policies((first_policy, second_policy))
    second = shared_policy(first_policy)
    initial = init_systems(first, second, observations, state, jax.random.key(2))
    np.testing.assert_array_equal(initial.team_a[0], (7, 7))
    np.testing.assert_array_equal(initial.team_a[1]["vector"], ((10, 20), (10, 20)))
    np.testing.assert_array_equal(initial.team_b, np.full((2, 5), 7))
    actions, memory, outputs = apply_systems(
        first, second, initial, observations, state, jax.random.key(3)
    )
    np.testing.assert_array_equal(actions.move[:, :5], ((1, 2, 0, 0, 0),) * 2)
    np.testing.assert_array_equal(memory.team_a[0], (8, 8))
    np.testing.assert_array_equal(memory.team_a[1]["vector"], ((11, 21), (11, 21)))
    assert outputs == ((), ())
    assert first.variables[0] is first_policy.variables
    assert first.variables[1] is second_policy.variables
    assert second.variables is first_policy.variables
    observations, state, *_ = env.step(jax.random.key(4), state, actions)
    observations, state = env.reset(
        jax.random.key(5), state=state, reset_mask=jnp.asarray((True, False))
    )
    _, memory, _ = apply_systems(
        first, second, memory, observations, state, jax.random.key(6)
    )
    np.testing.assert_array_equal(memory.team_a[0], (8, 9))
    np.testing.assert_array_equal(memory.team_a[1]["vector"], ((11, 21), (12, 22)))
    np.testing.assert_array_equal(memory.team_b, ((8,) * 5, (9,) * 5))


def test_independent_adapter_rejects_wrong_roster_size_and_mixed_modes() -> None:
    _, observations, state = _setup()
    original = policy("random")
    with pytest.raises(ValueError):
        independent_policies(
            (original, Policy("host", original.apply, execution="host"))
        )
    first = independent_policies((original,))
    with pytest.raises(ValueError):
        init_systems(
            first, shared_policy(original), observations, state, jax.random.key(0)
        )


@pytest.mark.parametrize("execution", ["jax", "host"])
def test_independent_adapter_rejects_changed_roster_before_action_calls(
    execution: PolicyExecution,
) -> None:
    env, observations, state = _setup()
    calls = []

    def actor(
        variables: PolicyTree,
        memory: PolicyTree,
        inputs: ActorInput,
        mask: PolicyTree,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        del variables, inputs, mask, key
        calls.append(True)
        zero = jnp.int32(0)
        return ActorAction(zero, zero, zero), memory

    original = Policy("actor", actor, execution=execution)
    first = independent_policies((original, original))
    second = shared_policy(original)
    memory = init_systems(first, second, observations, state, jax.random.key(0))
    observations, state = env.reset(
        jax.random.key(1),
        evaluation_env_config(team_sizes=(3, 3), max_steps=8),
        state=state,
        reset_mask=jnp.asarray((True, False)),
    )
    np.testing.assert_array_equal(
        state.config.agent_profile.active_mask.sum(axis=1), (6, 5)
    )
    with pytest.raises(ValueError, match=r"independent|roster"):
        apply_systems(first, second, memory, observations, state, jax.random.key(2))
    assert calls == []


def test_generation_resets_reused_ids_once_and_preserves_continuing_memory() -> None:
    env, observations, state = _setup()
    system = System("counter", _counter_apply, init=_counter_init)
    memory = init_systems(system, system, observations, state, jax.random.key(6))
    for index in range(2):
        actions, memory, _ = apply_systems(
            system, system, memory, observations, state, jax.random.key(20 + index)
        )
        observations, state, *_ = env.step(jax.random.key(30 + index), state, actions)
    np.testing.assert_array_equal(memory.team_a, (2, 2))
    old_ids = state.episode_id
    observations, state = env.reset(
        jax.random.key(7),
        state=state,
        reset_mask=jnp.asarray((True, False)),
        episode_id=old_ids,
    )
    actions, memory, _ = apply_systems(
        system, system, memory, observations, state, jax.random.key(8)
    )
    np.testing.assert_array_equal(memory.team_a, (1, 3))
    np.testing.assert_array_equal(memory.team_b, (1, 3))
    np.testing.assert_array_equal(memory.policy_trace.decision_step, (0, 2))
    np.testing.assert_array_equal(memory.policy_trace.episode_id, old_ids)
    observations, state, *_ = env.step(jax.random.key(9), state, actions)
    for _ in range(2):
        observations, state = env.reset(
            jax.random.key(7),
            state=state,
            reset_mask=jnp.asarray((True, False)),
            episode_id=old_ids,
        )
    _, memory, _ = apply_systems(
        system, system, memory, observations, state, jax.random.key(8)
    )
    np.testing.assert_array_equal(memory.team_a, (1, 4))


def test_initialization_keys_repeat_for_reused_ids_without_reusing_memory() -> None:
    env, observations, state = _setup()

    def initialize(variables: PolicyTree, inputs: SystemInput, keys: Array) -> Array:
        del variables, inputs

        def sample(key: Array) -> Array:
            return jax.random.uniform(key, ())

        return jax.vmap(sample)(keys)

    def advance(
        variables: PolicyTree, memory: Array, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, Array]:
        del variables, keys
        return _idle(inputs), memory + inputs.valid

    system = System("random memory", advance, init=initialize)
    memory = init_systems(system, system, observations, state, jax.random.key(4))
    original = memory.team_a
    assert not np.array_equal(memory.team_a, memory.team_b)
    actions, memory, _ = apply_systems(
        system, system, memory, observations, state, jax.random.key(5)
    )
    observations, state, *_ = env.step(jax.random.key(6), state, actions)
    observations, state = env.reset(
        jax.random.key(7),
        state=state,
        reset_mask=jnp.asarray((True, False)),
        episode_id=state.episode_id,
    )
    _, memory, _ = apply_systems(
        system, system, memory, observations, state, jax.random.key(8)
    )
    np.testing.assert_array_equal(memory.team_a, original + jnp.asarray((1, 2)))


def test_custom_reset_supports_memory_without_a_leading_batch_axis() -> None:
    env, observations, state = _setup(batch=3)

    def initialize(variables: PolicyTree, inputs: SystemInput, keys: Array) -> Array:
        del variables, keys
        return jnp.zeros((2, inputs.valid.size), jnp.int32)

    def reset(memory: Array, fresh_memory: Array, reset_mask: Array) -> Array:
        return jnp.where(reset_mask[None, :], fresh_memory, memory)

    def advance(
        variables: PolicyTree, memory: Array, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, Array]:
        del variables, keys
        return _idle(inputs), memory + inputs.valid[None, :]

    system = System("transposed", advance, init=initialize, reset_memory=reset)
    memory = init_systems(system, system, observations, state, jax.random.key(0))
    actions, memory, _ = apply_systems(
        system, system, memory, observations, state, jax.random.key(1)
    )
    observations, state, *_ = env.step(jax.random.key(2), state, actions)
    observations, state = env.reset(
        jax.random.key(3), state=state, reset_mask=jnp.asarray((False, True, False))
    )
    _, memory, _ = apply_systems(
        system, system, memory, observations, state, jax.random.key(4)
    )
    np.testing.assert_array_equal(memory.team_a, ((2, 1, 2), (2, 1, 2)))


def test_authored_start_death_and_respawn_keep_episode_memory_and_local_trace() -> None:
    config = evaluation_env_config(
        team_sizes=(1, 1),
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=20,
        max_steps=20,
    )._replace(
        spawn_shield_duration_steps=0,
        team_respawn_wave_period_step_count=jnp.asarray((2, 2), jnp.int32),
    )
    initial_core, *_ = core.reset(config, jax.random.key(10))
    authored = initial_core._replace(
        step_count=jnp.int32(5),
        agent_positions=initial_core.agent_positions.at[0]
        .set(jnp.asarray((4.0, 4.0)))
        .at[5]
        .set(jnp.asarray((6.5, 4.0))),
        current_health=initial_core.current_health.at[0].set(1.0).at[5].set(1.0),
        team_respawn_wave_countdowns=jnp.asarray((1, 1), jnp.int32),
    )
    prepared = core.initialize_scenario_state(authored, config)
    env = make("tdm", env_config=config, metrics="none")
    observations, state = env.reset(jax.random.key(11), initial=prepared[:3])

    def choose(
        variables: PolicyTree, memory: Array, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, Array]:
        del variables, keys
        idle = _idle(inputs)
        target = jnp.where(inputs.active_mask & (memory[:, None] == 0), 6, 0)
        return idle._replace(select_target=target), memory + inputs.valid

    system = System("one attack", choose, init=_counter_init)
    memory = init_systems(system, system, observations, state, jax.random.key(12))
    for decision in range(3):
        actions, memory, _ = apply_systems(
            system, system, memory, observations, state, jax.random.key(20 + decision)
        )
        np.testing.assert_array_equal(memory.team_a, (decision + 1,))
        np.testing.assert_array_equal(memory.team_b, (decision + 1,))
        np.testing.assert_array_equal(memory.policy_trace.decision_step, (decision,))
        observations, state, *_ = env.step(
            jax.random.key(30 + decision), state, actions
        )
        if decision < 2:
            np.testing.assert_array_equal(
                state.core_state.alive_mask[jnp.asarray((0, 5))],
                (decision == 1, decision == 1),
            )
    assert int(state.episode_id) == 1
    assert int(state.reset_generation) == 0


def test_scan_reuses_dynamic_weights_and_keeps_padding_out_of_memory() -> None:
    env, observations, state = _setup(max_steps=2)
    traces: list[int] = []

    def advance(
        variables: Array, memory: Array, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        del keys
        return SystemOutput(
            _idle(inputs),
            memory + inputs.valid.astype(jnp.int32),
            learning_outputs=variables * inputs.valid,
        )

    system = System("counter", advance, init=_counter_init)
    initial = init_systems(system, system, observations, state, jax.random.key(2))

    @jax.jit
    def run(weights: Array, key: Array, starting_memory: SystemState) -> PolicyTree:
        traces.append(1)

        def step(carry: PolicyTree, tick: Array) -> tuple[PolicyTree, PolicyTree]:
            obs, current, memory = carry
            policy_key, step_key = jax.random.split(jax.random.fold_in(key, tick))
            actions, memory, outputs = apply_systems(
                system,
                system,
                memory,
                obs,
                current,
                policy_key,
                variables_a=weights,
                variables_b=weights + 1,
            )
            obs, current, *_ = env.step(step_key, current, actions)
            return (obs, current, memory), (memory.policy_trace.valid, outputs)

        return jax.lax.scan(
            step, (observations, state, starting_memory), jnp.arange(4, dtype=jnp.int32)
        )

    changed_memory = initial._replace(
        team_a=jnp.asarray((4, 9), jnp.int32),
        team_b=jnp.asarray((5, 8), jnp.int32),
    )
    first = cast(PolicyTree, run(jnp.float32(2), jax.random.key(3), initial))
    second = cast(PolicyTree, run(jnp.float32(5), jax.random.key(4), changed_memory))
    assert traces == [1]
    for result, weight, final_memory in ((first, 2, (2, 2)), (second, 5, (6, 11))):
        np.testing.assert_array_equal(result[0][2].team_a, final_memory)
        np.testing.assert_array_equal(
            result[1][0], ((True, True),) * 2 + ((False, False),) * 2
        )
        np.testing.assert_array_equal(result[1][1][0][:2], np.full((2, 2), weight))
        np.testing.assert_array_equal(result[1][1][0][2:], np.zeros((2, 2)))


def test_compiled_scan_partially_resets_memory_after_retaining_terminal_data() -> None:
    env, _, state = _setup()
    config = state.config._replace(max_steps=jnp.asarray((1, 3), jnp.int32))
    observations, state = env.reset(jax.random.key(1), config)

    def advance(
        variables: PolicyTree, memory: Array, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        del variables, keys
        actions = _idle(inputs)._replace(move=inputs.active_mask.astype(jnp.int32))
        return SystemOutput(
            actions,
            memory + inputs.valid.astype(jnp.int32),
            learning_outputs=inputs.actors.observation.self_features[..., :2],
        )

    system = System("moving counter", advance, init=_counter_init)
    initial_memory = init_systems(
        system, system, observations, state, jax.random.key(2)
    )

    @jax.jit
    def run(
        initial: tuple[Observations, EnvironmentState, SystemState], key: Array
    ) -> PolicyTree:
        def step(
            carry: tuple[Observations, EnvironmentState, SystemState], tick: Array
        ) -> tuple[tuple[Observations, EnvironmentState, SystemState], PolicyTree]:
            obs, current, memory = carry
            policy_key, step_key, reset_key = jax.random.split(
                jax.random.fold_in(key, tick), 3
            )
            actions, memory, learning = apply_systems(
                system, system, memory, obs, current, policy_key
            )
            terminal_obs, following, _, done, _ = env.step(step_key, current, actions)
            next_obs, next_state = env.reset_done(reset_key, following)
            history = {
                "terminal_observations": terminal_obs,
                "reset_observations": next_obs,
                "done": done.done,
                "terminal_id": following.episode_id,
                "next_id": next_state.episode_id,
                "step_count": following.core_state.step_count,
                "next_step_count": next_state.core_state.step_count,
                "memory_a": memory.team_a,
                "memory_b": memory.team_b,
                "trace": memory.policy_trace,
                "decision_positions": learning[0],
            }
            return (next_obs, next_state, memory), history

        return jax.lax.scan(step, initial, jnp.arange(4, dtype=jnp.int32))

    final, history = cast(
        PolicyTree, run((observations, state, initial_memory), jax.random.key(3))
    )
    expected_memory = ((1, 1), (1, 2), (1, 3), (1, 1))
    np.testing.assert_array_equal(history["memory_a"], expected_memory)
    np.testing.assert_array_equal(history["memory_b"], expected_memory)
    np.testing.assert_array_equal(history["step_count"], expected_memory)
    np.testing.assert_array_equal(
        history["trace"].decision_step, ((0, 0), (0, 1), (0, 2), (0, 0))
    )
    np.testing.assert_array_equal(
        history["done"], ((True, False), (True, False), (True, True), (True, False))
    )
    np.testing.assert_array_equal(
        history["next_step_count"], ((0, 1), (0, 2), (0, 0), (0, 1))
    )
    np.testing.assert_array_equal(history["trace"].episode_id, history["terminal_id"])
    assert np.all(history["trace"].valid)
    np.testing.assert_array_equal(
        history["terminal_id"] != history["next_id"], history["done"]
    )
    terminal = history["terminal_observations"]
    reset = history["reset_observations"]
    for tick in range(4):
        for lane in range(2):
            if history["done"][tick, lane]:
                assert not np.array_equal(
                    terminal.observation.self_features[tick, lane, 0, :2],
                    reset.observation.self_features[tick, lane, 0, :2],
                )
            else:
                for before, after in zip(
                    jax.tree.leaves(terminal), jax.tree.leaves(reset), strict=True
                ):
                    np.testing.assert_array_equal(before[tick, lane], after[tick, lane])
    np.testing.assert_array_equal(
        history["decision_positions"][1:],
        reset.observation.self_features[:-1, :, :5, :2],
    )
    np.testing.assert_array_equal(final[1].cumulative_transition_count, (4, 4))
    _, continued = cast(PolicyTree, run(final, jax.random.key(4)))
    np.testing.assert_array_equal(continued["memory_a"][0], (1, 2))
    np.testing.assert_array_equal(continued["memory_b"][0], (1, 2))


def test_compiled_bad_component_ids_remain_visible_for_later_rejection() -> None:
    _, observations, state = _setup()

    def choose(
        variables: Array, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        del keys
        return SystemOutput(
            _idle(inputs),
            memory,
            policy_ids=jnp.full(inputs.active_mask.shape, variables, jnp.int32),
        )

    first = System("ids", choose, components=({"name": "only"},))
    second = System("idle", _stateless)
    memory = init_systems(first, second, observations, state, jax.random.key(0))

    @jax.jit
    def run(value: Array) -> Array:
        _, updated, _ = apply_systems(
            first,
            second,
            memory,
            observations,
            state,
            jax.random.key(1),
            variables_a=value,
        )
        return updated.policy_trace.policy_ids

    ids = cast(Array, run(jnp.int32(7)))
    np.testing.assert_array_equal(ids[:, :2], np.full((2, 2), 7))


def test_actual_system_input_keeps_restricted_sources_and_no_runner_fields() -> None:
    _, observations, state = _setup()
    observation = observations.observation._replace(
        objective_features=jnp.ones_like(observations.observation.objective_features)
    )
    observations = observations._replace(
        observation=observation,
        source_availability=jnp.zeros_like(observations.source_availability),
    )

    def inspect(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        del variables, keys
        assert SystemInput._fields == (
            "actors",
            "action_mask",
            "active_mask",
            "episode_start",
            "valid",
        )
        return SystemOutput(_idle(inputs), memory, learning_outputs=inputs.actors)

    system = System("inspect", inspect)
    memory = init_systems(system, system, observations, state, jax.random.key(0))
    _, _, outputs = apply_systems(
        system, system, memory, observations, state, jax.random.key(1)
    )
    for actor in outputs:
        assert not np.any(actor.source_availability)
        for leaf in actor.source_bank:
            assert not np.any(leaf)


@pytest.mark.parametrize(
    "kind",
    ["tuple_length", "action_shape", "action_dtype", "action_int64", "bare_learning"],
)
def test_malformed_system_results_fail_without_action_repair(kind: str) -> None:
    _, observations, state = _setup()

    def malformed(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> PolicyTree:
        del variables, keys
        action = _idle(inputs)
        if kind == "tuple_length":
            return (action,)
        if kind == "action_shape":
            return ActorAction(*(jnp.int32(0),) * 3), memory
        if kind == "action_dtype":
            return ActorAction(*(jnp.zeros((2, 5), jnp.float32),) * 3), memory
        if kind == "action_int64":
            wide = cast(Array, np.full((2, 5), 2**32 + 1, np.int64))
            return ActorAction(*(wide,) * 3), memory
        return action, memory, {"log_probability": jnp.zeros((2, 5))}

    first = System("bad", malformed)
    second = System("idle", _stateless)
    memory = init_systems(first, second, observations, state, jax.random.key(0))
    with pytest.raises((TypeError, ValueError)):
        apply_systems(first, second, memory, observations, state, jax.random.key(1))
