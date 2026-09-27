"""Check the shared System runner boundary without changing the simulator.

CPU cases compare exact legacy actor keys/actions, frozen numerical snapshots,
dynamic compilation reuse, refill initialization and opaque mixed host memory.
They also check that evaluation discards unused learning values inside the JAX
boundary and preserves the original public raw-System behavior.
"""

# These cases exercise the private boundary shared by public runners.
# pyright: reportPrivateUsage=false

from collections.abc import Callable
from dataclasses import replace
from operator import itemgetter
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.core.types import Action
from marl_battlegrounds.environment import Environment, EnvironmentState, make
from marl_battlegrounds.evaluation import policy_execution as execution
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, _stack_configs
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
    SystemState,
    _apply_system_pair,
    _init_system_pair,
    apply_policy_batch,
    independent_policies,
    shared_policy,
)
from marl_battlegrounds.evaluation.system_evaluation import (
    freeze_evaluation_method,
    prepare_evaluation_system,
    replace_initialization_roots,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations

type _PairResult = tuple[Action, SystemState, tuple[PolicyTree, PolicyTree]]


@pytest.mark.parametrize("device_sources", (False, True))
def test_config_batch_matches_numerical_composition_and_transfers_unique_sources(
    monkeypatch: pytest.MonkeyPatch, device_sources: bool
) -> None:
    first = evaluation_env_config(team_sizes=(2, 3), max_steps=4)
    second = evaluation_env_config(team_sizes=(3, 2), max_steps=7)
    if device_sources:

        def to_array(value: object) -> Array:
            return jnp.asarray(value)

        first, second = jax.tree.map(to_array, (first, second))
    specs = tuple(
        EpisodeSpec(index + 1, config)
        for index, config in enumerate((first, second, first))
    )
    expected = jax.tree.map(
        lambda *values: jnp.stack(tuple(jnp.asarray(value) for value in values)),
        *(spec.env_config for spec in specs),
    )
    original = jax.device_get
    transfers: list[object] = []

    def record(value: object) -> object:
        transfers.append(value)
        return original(value)

    monkeypatch.setattr(jax, "device_get", record)
    actual = _stack_configs(specs)
    assert len(transfers) == 2
    assert transfers[0] is first and transfers[1] is second
    for got, wanted in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert got.dtype == wanted.dtype and got.shape == wanted.shape
        np.testing.assert_array_equal(got, wanted)
    np.testing.assert_array_equal(
        actual.agent_profile.active_mask[0], first.agent_profile.active_mask
    )
    np.testing.assert_array_equal(
        actual.agent_profile.active_mask[1], second.agent_profile.active_mask
    )


def _setup() -> tuple[Environment, Observations, EnvironmentState]:
    env = make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(2, 3), max_steps=4),
        num_envs=2,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(1))
    return env, observations, state


def _assert_tree(actual: object, expected: object) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for first, second in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if hasattr(first, "dtype") and jnp.issubdtype(first.dtype, jax.dtypes.prng_key):
            first, second = jax.random.key_data(first), jax.random.key_data(second)
        np.testing.assert_array_equal(first, second)


def _roots(root: Array, seeds: Array, decisions: Array, stream: int) -> Array:
    def lane(seed: Array, step: Array) -> Array:
        return jax.random.fold_in(
            jax.random.fold_in(jax.random.fold_in(root, stream), seed), step
        )

    return jax.vmap(lane)(seeds, decisions)


def _actor_keys(roots: Array) -> Array:
    def lane(root: Array) -> Array:
        def actor(slot: Array) -> Array:
            return jax.random.fold_in(root, slot)

        return jax.vmap(actor)(jnp.arange(10, dtype=jnp.uint32))

    return jax.vmap(lane)(roots)


def _sample_actor(
    variables: PolicyTree,
    memory: PolicyTree,
    actor: PolicyTree,
    mask: PolicyTree,
    key: Array,
) -> tuple[ActorAction, PolicyTree]:
    del actor
    logits = jnp.where(mask.move_mask, variables * jnp.arange(9), -jnp.inf)
    move = jax.random.categorical(key, logits).astype(jnp.int32)
    zero = jnp.asarray(0, jnp.int32)
    return ActorAction(move, zero, zero), memory + 1


def _idle(inputs: SystemInput) -> ActorAction:
    zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
    return ActorAction(zero, zero, zero)


def _initialize(variables: PolicyTree, inputs: SystemInput, keys: Array) -> Array:
    del variables

    def draw(key: Array) -> Array:
        return jax.random.bits(key, dtype=jnp.uint32)

    return jax.vmap(draw)(keys).astype(jnp.int32) % 1000 + jnp.zeros(
        inputs.valid.shape, jnp.int32
    )


def _recurrent(
    variables: PolicyTree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    del keys
    return SystemOutput(
        _idle(inputs),
        memory + jnp.asarray(variables, jnp.int32) * inputs.valid,
        learning_outputs={"unused": jnp.sin(jnp.arange(4096) + variables)},
        policy_ids=jnp.where(inputs.active_mask, 0, -1).astype(jnp.int32),
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_shared_adapter_preserves_exact_legacy_actor_actions_and_memory(
    legacy: bool,
) -> None:
    _, observations, state = _setup()
    first = Policy("first", _sample_actor, jnp.asarray(0.0), jnp.asarray(3))
    second = Policy("second", _sample_actor, jnp.asarray(0.2), jnp.asarray(7))
    a, va, ta = prepare_evaluation_system(freeze_evaluation_method(first))
    b, vb, tb = prepare_evaluation_system(freeze_evaluation_method(second))
    root = jax.random.PRNGKey(19) if legacy else jax.random.key(19)
    seeds = jnp.asarray([107, 23], jnp.uint32)
    roots = _roots(root, seeds, jnp.zeros(2, jnp.int32), 2)
    actor_keys = _actor_keys(roots)
    memory = _init_system_pair(
        a, b, va, vb, (ta, tb), observations, state, _roots(root, seeds, seeds * 0, 3)
    )
    expected = apply_policy_batch(
        first.apply,
        second.apply,
        va,
        vb,
        memory.team_a,
        memory.team_b,
        observations,
        state.action_mask,
        actor_keys,
        ~state.done.done,
    )
    compiled = cast(
        Callable[..., _PairResult],
        jax.jit(
            _apply_system_pair,
            static_argnums=(0, 1),
            static_argnames=("keep_learning_outputs",),
        ),
    )
    action, updated, learning = compiled(
        a,
        b,
        va,
        vb,
        memory,
        observations,
        state,
        roots,
        actor_keys=actor_keys,
        keep_learning_outputs=False,
    )
    _assert_tree(action, expected[0])
    # The legacy helper calls inactive capacity slots; System ownership freezes them.
    for count, previous, actual, reference in (
        (2, memory.team_a, updated.team_a, expected[1]),
        (3, memory.team_b, updated.team_b, expected[2]),
    ):
        active = itemgetter((slice(None), slice(None, count)))
        inactive = itemgetter((slice(None), slice(count, None)))
        _assert_tree(jax.tree.map(active, actual), jax.tree.map(active, reference))
        _assert_tree(jax.tree.map(inactive, actual), jax.tree.map(inactive, previous))
    assert learning == ((), ())
    np.testing.assert_array_equal(updated.policy_trace.decision_step, [0, 0])
    np.testing.assert_array_equal(updated.policy_trace.valid, [True, True])


def test_independent_adapter_uses_supplied_global_actor_keys_in_roster_order() -> None:
    env, observations, state = _setup()
    entries = [
        Policy(str(index), _sample_actor, jnp.asarray(index / 10), jnp.asarray(index))
        for index in range(3)
    ]
    first, second = independent_policies(entries[:2]), independent_policies(entries)
    a, va, ta = prepare_evaluation_system(first)
    b, vb, tb = prepare_evaluation_system(second)
    roots = jax.random.split(jax.random.key(33), 2)
    keys = _actor_keys(roots)
    memory = _init_system_pair(a, b, va, vb, (ta, tb), observations, state, roots)
    actual, updated, _ = _apply_system_pair(
        a, b, va, vb, memory, observations, state, roots, actor_keys=keys
    )
    for team, count, values, old, new in (
        (0, 2, va, memory.team_a, updated.team_a),
        (1, 3, vb, memory.team_b, updated.team_b),
    ):
        inputs = env.policy_inputs(observations, state, team=team)
        for slot in range(count):
            expected, next_memory = jax.vmap(_sample_actor, in_axes=(None, 0, 0, 0, 0))(
                values[slot],
                old[slot],
                jax.tree.map(itemgetter((slice(None), slot)), inputs.actors),
                jax.tree.map(itemgetter((slice(None), slot)), inputs.action_mask),
                keys[:, team * 5 + slot],
            )
            _assert_tree(
                ActorAction(*(head[:, team * 5 + slot] for head in actual)), expected
            )
            _assert_tree(new[slot], next_memory)


@pytest.mark.parametrize("legacy", [False, True])
def test_refill_changes_only_selected_roots_and_resets_only_new_memory(
    legacy: bool,
) -> None:
    env, observations, state = _setup()
    system = System(
        "recurrent",
        _recurrent,
        jnp.asarray(1),
        init=_initialize,
        components=({"name": "actor"},),
    )
    descriptor, variables, template = prepare_evaluation_system(system)
    root = jax.random.PRNGKey(8) if legacy else jax.random.key(8)
    seeds = jnp.asarray([11, 12], jnp.uint32)
    roots = _roots(root, seeds, seeds * 0, 3)
    memory = _init_system_pair(
        descriptor,
        descriptor,
        variables,
        variables,
        (template, template),
        observations,
        state,
        roots,
    )
    _, advanced, _ = _apply_system_pair(
        descriptor,
        descriptor,
        variables,
        variables,
        memory,
        observations,
        state,
        _roots(root, seeds, seeds * 0, 2),
        keep_learning_outputs=False,
    )
    mask = jnp.asarray([True, False])
    observations, reset = env.reset(jax.random.key(5), state=state, reset_mask=mask)
    new_roots = _roots(root, jnp.asarray([91, 92], jnp.uint32), seeds * 0, 3)
    replaced = cast(
        SystemState, jax.jit(replace_initialization_roots)(advanced, new_roots, mask)
    )
    _assert_tree(replaced.init_key[0], new_roots[0])
    _assert_tree(replaced.init_key[1], roots[1])
    _assert_tree(replaced.team_a, advanced.team_a)
    _, actual, outputs = _apply_system_pair(
        descriptor,
        descriptor,
        variables,
        variables,
        replaced,
        observations,
        reset,
        _roots(root, seeds, seeds * 0, 2),
        keep_learning_outputs=False,
    )
    fresh = _init_system_pair(
        descriptor,
        descriptor,
        variables,
        variables,
        (template, template),
        observations,
        reset,
        new_roots,
    )
    for new, initial, old in (
        (actual.team_a, fresh.team_a, advanced.team_a),
        (actual.team_b, fresh.team_b, advanced.team_b),
    ):
        assert int(new[0]) == int(initial[0]) + 1
        assert int(new[1]) == int(old[1]) + 1
    assert outputs == ((), ())


def test_separate_parameter_snapshots_reuse_static_call_structure() -> None:
    _, observations, state = _setup()
    system = System(
        "recurrent",
        _recurrent,
        np.asarray(1, np.int32),
        init=_initialize,
        components=({"name": "actor"},),
    )
    first = freeze_evaluation_method(system)
    second = freeze_evaluation_method(
        replace(system, variables=np.asarray(4, np.int32))
    )
    a, va, ta = prepare_evaluation_system(first)
    b, vb, tb = prepare_evaluation_system(second)
    assert a == b
    roots = jax.random.split(jax.random.key(8), 2)
    memory = _init_system_pair(a, a, va, va, (ta, ta), observations, state, roots)
    compiled = cast(
        Callable[..., _PairResult],
        jax.jit(
            _apply_system_pair,
            static_argnums=(0, 1),
            static_argnames=("keep_learning_outputs",),
        ),
    )
    cast(Any, compiled).clear_cache()
    first_output = compiled(
        a, a, va, va, memory, observations, state, roots, keep_learning_outputs=False
    )
    second_output = compiled(
        b,
        b,
        vb,
        vb,
        memory._replace(adapter_templates=(tb, tb)),
        observations,
        state,
        roots,
        keep_learning_outputs=False,
    )
    np.testing.assert_array_equal(second_output[1].team_a - first_output[1].team_a, 3)
    assert cast(Any, compiled)._cache_size() == 1
    assert first_output[2] == second_output[2] == ((), ())


class _Opaque:
    def __array__(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("opaque values must not be converted")

    def __deepcopy__(self, memo: object) -> object:
        raise AssertionError("opaque values must not be copied")


def test_freezing_preserves_policy_kind_adapter_evidence_and_opaque_host_leaves() -> (
    None
):
    weights = np.asarray([1.0, 2.0], np.float32)
    template = np.asarray(4, np.int32)
    original = Policy("named", _sample_actor, weights, template, checkpoint="v1")
    frozen_policy = freeze_evaluation_method(original)
    frozen_adapter = freeze_evaluation_method(shared_policy(original))
    opaque = _Opaque()
    host = System(
        "provider",
        _recurrent,
        {"weights": weights, "session": opaque},
        execution="host",
    )
    frozen_host = freeze_evaluation_method(host)
    weights[:] = 9
    template[...] = 8
    assert isinstance(frozen_policy, Policy)
    assert isinstance(frozen_adapter, System)
    assert isinstance(frozen_host, System)
    assert frozen_policy.apply is original.apply
    assert frozen_policy.checkpoint == "v1"
    np.testing.assert_array_equal(frozen_policy.variables, [1.0, 2.0])
    assert int(frozen_policy.initial_carry) == 4
    _assert_tree(frozen_adapter.variables, frozen_adapter._policies[0].variables)
    assert int(frozen_adapter._policies[0].initial_carry) == 4
    np.testing.assert_array_equal(frozen_host.variables["weights"], [1.0, 2.0])
    assert not frozen_host.variables["weights"].flags.writeable
    assert frozen_host.variables["session"] is opaque


@pytest.mark.parametrize("host_team", [0, 1])
def test_mixed_evaluation_calls_host_once_and_drops_learning_inside_jax_boundary(
    host_team: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, observations, state = _setup()
    calls: list[tuple[object, Array]] = []
    retained = _Opaque()

    def host(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        assert isinstance(inputs.valid, np.ndarray)
        calls.append((variables, keys))
        return SystemOutput(_idle(inputs), memory, learning_outputs=retained)

    first = System("host", host, {"provider": retained}, execution="host")
    second = System(
        "jax",
        _recurrent,
        jnp.asarray(1),
        init=_initialize,
        components=({"name": "actor"},),
    )
    methods = (first, second) if host_team == 0 else (second, first)
    a, va, ta = prepare_evaluation_system(freeze_evaluation_method(methods[0]))
    b, vb, tb = prepare_evaluation_system(freeze_evaluation_method(methods[1]))
    roots = jax.random.split(jax.random.key(9), 2)
    memory = _init_system_pair(a, b, va, vb, (ta, tb), observations, state, roots)
    original = execution._mixed_jax_apply
    returned: list[object] = []

    def inspect(*args: PolicyTree, **kwargs: PolicyTree) -> SystemOutput:
        value = cast(SystemOutput, original(*args, **kwargs))
        returned.append(value.learning_outputs)
        return value

    monkeypatch.setattr(execution, "_mixed_jax_apply", inspect)
    _, updated, learning = _apply_system_pair(
        a,
        b,
        va,
        vb,
        memory,
        observations,
        state,
        roots,
        keep_learning_outputs=False,
    )
    assert len(calls) == 1
    assert cast(dict[str, object], calls[0][0])["provider"] is retained
    assert returned == [()]
    assert learning == ((), ())
    done = state._replace(done=state.done._replace(terminated=jnp.ones(2, jnp.bool_)))
    _, after_padding, learning = _apply_system_pair(
        a,
        b,
        va,
        vb,
        updated,
        observations,
        done,
        roots,
        keep_learning_outputs=False,
    )
    assert len(calls) == 1
    assert learning == ((), ())
    assert not np.any(after_padding.policy_trace.valid)


def test_bad_explicit_actor_keys_and_refill_shapes_fail_before_methods() -> None:
    _, observations, state = _setup()
    policy = Policy("sample", _sample_actor, jnp.asarray(0.0), jnp.asarray(0))
    descriptor, variables, template = prepare_evaluation_system(policy)
    roots = jax.random.split(jax.random.key(9), 2)
    memory = _init_system_pair(
        descriptor,
        descriptor,
        variables,
        variables,
        (template, template),
        observations,
        state,
        roots,
    )
    with pytest.raises(ValueError, match="actor_keys"):
        _apply_system_pair(
            descriptor,
            descriptor,
            variables,
            variables,
            memory,
            observations,
            state,
            roots,
            actor_keys=roots,
        )
    with pytest.raises(ValueError, match="refill roots"):
        replace_initialization_roots(memory, roots, jnp.asarray([True]))


def test_evaluation_roster_setup_checks_both_teams_without_calling_methods() -> None:
    from marl_battlegrounds.evaluation.system_evaluation import (
        validate_evaluation_rosters,
    )

    def forbidden(*args: object) -> PolicyTree:
        del args
        pytest.fail("roster validation must not execute a method")

    first = independent_policies(tuple(Policy(str(i), forbidden) for i in range(2)))
    second = independent_policies(tuple(Policy(str(i), forbidden) for i in range(3)))
    a, _, _ = prepare_evaluation_system(first)
    b, _, _ = prepare_evaluation_system(second)
    config = evaluation_env_config(team_sizes=(2, 3))
    validate_evaluation_rosters(a, b, config)
    with pytest.raises(ValueError, match="active prefix"):
        validate_evaluation_rosters(b, a, config)
