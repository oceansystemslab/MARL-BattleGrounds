"""Check stable host System batches, opaque memory and mixed JAX execution.

These CPU tests use local fake providers. They check key formats, compilation
reuse, permitted NumPy inputs, selected resets, retained outputs and errors.
They do not call a network service or qualify GPU or provider performance.
"""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.environment import Environment, EnvironmentState, make
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
    apply_systems,
    init_systems,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations


class _Session:
    def __init__(self, label: str) -> None:
        self.label = label
        self.decisions = 0

    def __array__(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("opaque sessions must not become arrays")


def _setup(*, max_steps: int = 8) -> tuple[Environment, Observations, EnvironmentState]:
    env = make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(2, 3), max_steps=max_steps),
        num_envs=2,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(1))
    return env, observations, state


def _jax_idle(
    variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
) -> SystemOutput:
    del variables, keys
    zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
    return SystemOutput(
        ActorAction(zero, zero, zero),
        memory,
        learning_outputs={"valid": inputs.valid},
    )


def _host_idle(inputs: SystemInput) -> ActorAction:
    zero = cast(Array, np.zeros(inputs.active_mask.shape, np.int32))
    return ActorAction(zero, zero, zero)


@pytest.mark.parametrize("legacy_keys", [False, True])
@pytest.mark.parametrize("host_team", [0, 1])
def test_mixed_system_preserves_numpy_inputs_keys_and_one_stable_host_batch(
    legacy_keys: bool,
    host_team: int,
) -> None:
    env, observations, state = _setup()
    seen: list[tuple[tuple[int, ...], object]] = []
    expected = env.policy_inputs(observations, state, team=host_team)

    def host(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        del variables
        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(inputs))
        assert isinstance(keys, Array)
        assert keys.shape == ((2, 2) if legacy_keys else (2,))
        if legacy_keys:
            assert keys.dtype == jnp.uint32
        else:
            assert jnp.issubdtype(keys.dtype, jax.dtypes.prng_key)
        for actual, wanted in zip(
            jax.tree.leaves(inputs), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_array_equal(actual, wanted)
        seen.append((inputs.valid.shape, jax.random.key_data(keys)))
        return SystemOutput(
            _host_idle(inputs),
            memory,
            learning_outputs=("provider result", tuple(inputs.valid.tolist())),
        )

    host_system = System("host", host, execution="host")
    jax_system = System("jax", _jax_idle)
    first, second = (
        (host_system, jax_system) if host_team == 0 else (jax_system, host_system)
    )
    key = jax.random.PRNGKey(11) if legacy_keys else jax.random.key(11)
    memory = init_systems(first, second, observations, state, key)
    assert seen == []
    actions, memory, outputs = apply_systems(
        first, second, memory, observations, state, key
    )
    assert len(seen) == 1
    assert seen[0][0] == (2,)
    assert outputs[host_team] == ("provider result", (True, True))
    np.testing.assert_array_equal(outputs[1 - host_team]["valid"], (True, True))
    assert actions.move.shape == (2, 10)
    assert memory.team_a == memory.team_b == ()


@pytest.mark.parametrize("per_lane", [False, True])
@pytest.mark.parametrize("legacy_init", [False, True])
def test_mixed_reset_reuses_compilation_with_different_init_and_action_key_formats(
    per_lane: bool, legacy_init: bool
) -> None:
    env, observations, state = _setup()
    traced: list[int] = []

    def initialize(variables: PolicyTree, inputs: SystemInput, keys: Array) -> Array:
        del variables, inputs

        def sample(key: Array) -> Array:
            return jax.random.uniform(key, ())

        return jax.vmap(sample)(keys)

    def numerical(
        variables: Array, memory: Array, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        traced.append(1)
        zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
        return SystemOutput(
            ActorAction(zero, zero, zero),
            memory + variables * inputs.valid,
            learning_outputs={
                "start_memory": memory,
                "keys": jax.random.key_data(keys),
            },
        )

    def host(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, PolicyTree]:
        del variables, keys
        return _host_idle(inputs), memory

    first = System("host", host, execution="host")
    second = System("jax", numerical, variables=jnp.float32(1), init=initialize)
    init_key = jax.random.PRNGKey(2) if legacy_init else jax.random.key(2)
    action_key = jax.random.key(3) if legacy_init else jax.random.PRNGKey(3)
    next_action_key = jax.random.key(6) if legacy_init else jax.random.PRNGKey(6)
    if per_lane:
        init_key = jax.random.split(init_key, 2)
        action_key = jax.random.split(action_key, 2)
        next_action_key = jax.random.split(next_action_key, 2)
    memory = init_systems(first, second, observations, state, init_key)
    actions, advanced, learning = apply_systems(
        first, second, memory, observations, state, action_key
    )
    np.testing.assert_array_equal(learning[1]["start_memory"], memory.team_b)
    previous_keys = learning[1]["keys"]
    observations, state, *_ = env.step(jax.random.key(4), state, actions)
    observations, state = env.reset(
        jax.random.key(5), state=state, reset_mask=jnp.asarray((True, False))
    )
    fresh = init_systems(first, second, observations, state, init_key)
    _, updated, learning = apply_systems(
        first,
        second,
        advanced,
        observations,
        state,
        next_action_key,
        variables_b=jnp.float32(2),
    )
    np.testing.assert_array_equal(learning[1]["start_memory"][0], fresh.team_b[0])
    np.testing.assert_array_equal(learning[1]["start_memory"][1], advanced.team_b[1])
    np.testing.assert_array_equal(updated.team_b, learning[1]["start_memory"] + 2)
    assert not np.array_equal(learning[1]["keys"], previous_keys)
    assert traced == [1]


def test_default_host_reset_keeps_sessions_attached_to_continuing_lanes() -> None:
    env, observations, state = _setup()
    initialized: list[tuple[bool, ...]] = []
    calls: list[tuple[str | None, ...]] = []

    def initialize(
        variables: PolicyTree, inputs: SystemInput, keys: Array
    ) -> list[_Session | None]:
        del variables, keys
        selected = tuple(bool(value) for value in inputs.valid)
        initialized.append(selected)
        return [
            _Session(f"init-{len(initialized)}-lane-{lane}") if valid else None
            for lane, valid in enumerate(selected)
        ]

    def host(
        variables: PolicyTree,
        memory: list[_Session | None],
        inputs: SystemInput,
        keys: Array,
    ) -> SystemOutput:
        del variables, keys
        calls.append(
            tuple(value.label if value is not None else None for value in memory)
        )
        for index, valid in enumerate(inputs.valid):
            if valid:
                session = memory[index]
                assert session is not None
                session.decisions += 1
        retained = tuple(
            (value.label, value.decisions) if value is not None else None
            for value in memory
        )
        return SystemOutput(_host_idle(inputs), memory, learning_outputs=retained)

    first = System("sessions", host, init=initialize, execution="host")
    second = System("idle", _jax_idle)
    memory = init_systems(first, second, observations, state, jax.random.key(2))
    assert initialized == [(True, True)]
    assert calls == []
    actions, memory, first_outputs = apply_systems(
        first, second, memory, observations, state, jax.random.key(3)
    )
    retained = first_outputs[0]
    continuing = memory.team_a[1]
    replaced = memory.team_a[0]
    observations, state, *_ = env.step(jax.random.key(4), state, actions)
    observations, state = env.reset(
        jax.random.key(5),
        state=state,
        reset_mask=jnp.asarray((True, False)),
        episode_id=state.episode_id,
    )
    _, memory, outputs = apply_systems(
        first, second, memory, observations, state, jax.random.key(6)
    )
    assert initialized == [(True, True), (True, False)]
    assert len(calls) == 2
    assert memory.team_a[0] is not replaced
    assert memory.team_a[1] is continuing
    assert memory.team_a[0].decisions == 1
    assert memory.team_a[1].decisions == 2
    assert retained == (("init-1-lane-0", 1), ("init-1-lane-1", 1))
    assert outputs[0] == (("init-2-lane-0", 1), ("init-1-lane-1", 2))


def test_custom_host_reset_receives_opaque_layout_only_at_reset_boundaries() -> None:
    env, observations, state = _setup()
    resets: list[tuple[bool, ...]] = []

    def initialize(
        variables: PolicyTree, inputs: SystemInput, keys: Array
    ) -> PolicyTree:
        del variables, keys
        return {
            "sessions": [
                _Session(str(index)) if valid else None
                for index, valid in enumerate(inputs.valid)
            ],
            "description": "opaque provider state",
        }

    def reset(memory: PolicyTree, fresh: PolicyTree, selected: object) -> PolicyTree:
        mask = np.asarray(selected)
        resets.append(tuple(bool(value) for value in mask))
        return {
            "sessions": [
                fresh["sessions"][index] if value else memory["sessions"][index]
                for index, value in enumerate(mask)
            ],
            "description": memory["description"],
        }

    def host(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, PolicyTree]:
        del variables, keys
        assert memory["description"] == "opaque provider state"
        for index, valid in enumerate(inputs.valid):
            if valid:
                memory["sessions"][index].decisions += 1
        return _host_idle(inputs), memory

    first = System(
        "custom", host, init=initialize, reset_memory=reset, execution="host"
    )
    second = System("idle", _jax_idle)
    memory = init_systems(first, second, observations, state, jax.random.key(2))
    continuing = memory.team_a["sessions"][1]
    for index in range(2):
        actions, memory, _ = apply_systems(
            first, second, memory, observations, state, jax.random.key(3 + index)
        )
        observations, state, *_ = env.step(jax.random.key(5 + index), state, actions)
    assert resets == []
    observations, state = env.reset(
        jax.random.key(7), state=state, reset_mask=jnp.asarray((True, False))
    )
    _, memory, _ = apply_systems(
        first, second, memory, observations, state, jax.random.key(8)
    )
    assert resets == [(True, False)]
    assert memory.team_a["sessions"][1] is continuing
    assert continuing.decisions == 3
    assert memory.team_a["sessions"][0].decisions == 1


@pytest.mark.parametrize("host_team", [0, 1])
def test_all_invalid_skips_host_calls_and_does_not_reuse_learning_outputs(
    host_team: int,
) -> None:
    env, observations, state = _setup(max_steps=1)
    calls: list[int] = []

    def host(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        del variables, keys
        calls.append(1)
        return SystemOutput(_host_idle(inputs), memory, learning_outputs="decision one")

    provider = System("host", host, execution="host")
    numerical = System("jax", _jax_idle)
    first, second = (provider, numerical) if host_team == 0 else (numerical, provider)
    memory = init_systems(first, second, observations, state, jax.random.key(2))
    actions, memory, previous = apply_systems(
        first, second, memory, observations, state, jax.random.key(3)
    )
    observations, state, *_ = env.step(jax.random.key(4), state, actions)
    assert np.all(state.done.done)
    padding, memory, outputs = apply_systems(
        first, second, memory, observations, state, jax.random.key(5)
    )
    assert calls == [1]
    assert previous[host_team] == "decision one"
    assert outputs[host_team] == ()
    np.testing.assert_array_equal(outputs[1 - host_team]["valid"], (False, False))
    assert not np.any(memory.policy_trace.valid)
    assert memory.team_a == memory.team_b == ()
    for leaf in padding:
        assert not np.any(leaf[:, host_team * 5 : (host_team + 1) * 5])


def test_partial_invalid_batch_keeps_stable_rows_and_opaque_memory() -> None:
    env, observations, state = _setup()
    config = state.config._replace(max_steps=jnp.asarray((1, 3), jnp.int32))
    observations, state = env.reset(jax.random.key(1), config)
    seen: list[tuple[bool, ...]] = []

    def initialize(
        variables: PolicyTree, inputs: SystemInput, keys: Array
    ) -> PolicyTree:
        del variables, keys
        return [_Session(str(index)) for index in range(inputs.valid.size)]

    def host(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, PolicyTree]:
        del variables, keys
        seen.append(tuple(bool(value) for value in inputs.valid))
        for index, valid in enumerate(inputs.valid):
            if valid:
                memory[index].decisions += 1
        return _host_idle(inputs), memory

    first = System("host", host, init=initialize, execution="host")
    second = System("jax", _jax_idle)
    memory = init_systems(first, second, observations, state, jax.random.key(2))
    for index in range(3):
        actions, memory, _ = apply_systems(
            first, second, memory, observations, state, jax.random.key(3 + index)
        )
        observations, state, *_ = env.step(jax.random.key(6 + index), state, actions)
    assert seen == [(True, True), (False, True), (False, True)]
    assert [value.decisions for value in memory.team_a] == [1, 3]
    assert [value.label for value in memory.team_a] == ["0", "1"]


@pytest.mark.parametrize("where", ["init", "apply", "reset"])
def test_host_failure_propagates_without_retry_or_environment_advance(
    where: str,
) -> None:
    env, observations, state = _setup()
    calls: list[str] = []

    def initialize(
        variables: PolicyTree, inputs: SystemInput, keys: Array
    ) -> PolicyTree:
        del variables, keys
        calls.append("init")
        if where == "init":
            raise TimeoutError("fake provider failed")
        return [_Session(str(index)) for index in range(inputs.valid.size)]

    def reset(memory: PolicyTree, fresh: PolicyTree, selected: object) -> PolicyTree:
        del memory, fresh, selected
        calls.append("reset")
        raise TimeoutError("fake provider failed")

    def host(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, PolicyTree]:
        del variables, keys
        calls.append("apply")
        if where == "apply":
            raise TimeoutError("fake provider failed")
        return _host_idle(inputs), memory

    first = System(
        "failing", host, init=initialize, reset_memory=reset, execution="host"
    )
    second = System("idle", _jax_idle)
    if where == "init":
        with pytest.raises(TimeoutError, match="fake provider failed"):
            init_systems(first, second, observations, state, jax.random.key(2))
        assert calls == ["init"]
        return
    memory = init_systems(first, second, observations, state, jax.random.key(2))
    if where == "reset":
        observations, state = env.reset(
            jax.random.key(3), state=state, reset_mask=jnp.asarray((True, False))
        )
    before = np.asarray(state.core_state.step_count).copy()
    with pytest.raises(TimeoutError, match="fake provider failed"):
        apply_systems(first, second, memory, observations, state, jax.random.key(4))
    np.testing.assert_array_equal(state.core_state.step_count, before)
    assert calls.count(where) == 1


@pytest.mark.parametrize(
    "kind", ["unknown", "negative", "dtype", "int64", "shape", "undeclared"]
)
def test_host_component_ids_reject_bad_values_before_returning_actions(
    kind: str,
) -> None:
    _, observations, state = _setup()

    def host(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        del variables, keys
        shape = (2,) if kind == "shape" else (2, 5)
        dtype = np.float32 if kind == "dtype" else np.int32
        value = -2 if kind == "negative" else (2 if kind == "unknown" else 0)
        if kind == "int64":
            dtype = np.int64
            value = 2**32
        ids = np.full(shape, value, dtype=dtype)
        return SystemOutput(_host_idle(inputs), memory, policy_ids=cast(Array, ids))

    components = None if kind == "undeclared" else ({"name": "only"},)
    first = System("ids", host, execution="host", components=components)
    second = System("idle", _jax_idle)
    memory = init_systems(first, second, observations, state, jax.random.key(2))
    with pytest.raises((TypeError, ValueError)):
        apply_systems(first, second, memory, observations, state, jax.random.key(3))
