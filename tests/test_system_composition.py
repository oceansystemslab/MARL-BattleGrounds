"""Check ordinary team/pool composition without changing actor or memory identity.

CPU tests follow public reset/apply/step paths, fixed Policy slot streams, nested
ownership, per-game pool choice, custom memory and resource cleanup. Compilation
counts cover same-shaped value changes; these are not GPU speed measurements.
"""

# Private executor calls here verify evaluator-owned physical-slot keys.
# pyright: reportPrivateUsage=false
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.environment import Environment, EnvironmentState
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
    SystemInput,
    SystemOutput,
    SystemState,
    _apply_system_pair,
    _execution,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.tasks import AgentClassName

type Tree = Any


def _game(
    roster: Sequence[AgentClassName] = ("mage", "warrior", "hunter", "rogue", "priest"),
) -> tuple[Environment, Observations, EnvironmentState]:
    env = marl_bgs.make(
        "tdm", map_id=0, num_envs=2, team_a_roster=roster, max_steps=3, metrics="none"
    )
    obs, state = env.reset(jax.random.key(91))
    return env, obs, state


def _zero(inputs: SystemInput) -> ActorAction:
    return ActorAction(
        *(jnp.zeros(inputs.active_mask.shape, jnp.int32) for _ in range(3))
    )


def _init(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    del variables

    def draw(key: Array) -> Array:
        return jax.random.randint(key, (), 0, 100)

    return jax.vmap(draw)(keys)


def _act(
    variables: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    def draw(key: Array) -> Array:
        return jax.random.randint(key, (5,), 0, 9)

    draws = jax.vmap(draw)(keys)
    zero = jnp.zeros_like(draws)
    return SystemOutput(
        ActorAction(draws, zero, zero),
        memory + inputs.valid,
        learning_outputs={"value": variables * jnp.ones(inputs.valid.shape)},
    )


def _scalar(
    variables: Tree, memory: Tree, actor: Tree, masks: Tree, key: Array
) -> tuple[ActorAction, Tree]:
    del actor, masks
    return ActorAction(
        jax.random.randint(key, (), 0, 9),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    ), memory + variables


def _equal(first: Tree, second: Tree) -> None:
    for one, two in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(one, two)


def _singleton_pool(method: System) -> System:
    return marl_bgs.pool({method: 1})


@pytest.mark.parametrize("wrapped", [marl_bgs.team, _singleton_pool])
@pytest.mark.parametrize("key", [jax.random.key(4), jax.random.PRNGKey(4)])
def test_singleton_preserves_system_initialization_actions_and_learning(
    wrapped: Callable[[System], System], key: Array
) -> None:
    env, obs, state = _game()
    method = System("Counter", _act, variables=jnp.array(3.0), init=_init)
    other = marl_bgs.shared_policy(marl_bgs.policy("random"))
    composed = wrapped(method)
    old = marl_bgs.init_systems(method, other, obs, state, key)
    nested = marl_bgs.init_systems(composed, other, obs, state, key)
    _equal(old.team_a, nested.team_a.members[0])
    action, old, values = marl_bgs.apply_systems(method, other, old, obs, state, key)
    composed_action, nested, composed_values = marl_bgs.apply_systems(
        composed, other, nested, obs, state, key
    )
    _equal(action, composed_action)
    _equal(old.team_a, nested.team_a.members[0])
    _equal(values[0], composed_values[0][0])
    _equal(env.step(key, state, action), env.step(key, state, composed_action))


@pytest.mark.parametrize("by_class", [False, True])
def test_singleton_team_keeps_partial_incoming_ownership(by_class: bool) -> None:
    _, obs, state = _game(("mage", "warrior", "mage", "rogue", "priest"))

    def report_control(
        variables: Tree, memory: Array, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        output = _act(variables, memory, inputs, keys)
        return SystemOutput(
            output.actions, output.next_memory, learning_outputs=inputs.controlled_mask
        )

    member = System("Counter", report_control, jnp.array(3.0), init=_init)
    singleton = marl_bgs.team(member, slots=["mage"] if by_class else None)
    outer_slots: Sequence[Sequence[int | str] | int | str] = [
        "mage",
        ["warrior", "rogue", "priest"],
    ]
    direct = marl_bgs.team(member, "random", slots=outer_slots)
    nested = marl_bgs.team(singleton, "random", slots=outer_slots)
    other = marl_bgs.shared_policy(marl_bgs.policy("random"))
    key = jax.random.key(17)
    outputs: list[Tree] = []
    choose = jax.jit(marl_bgs.apply_systems, static_argnums=(0, 1))
    for method in (direct, nested):
        memory = marl_bgs.init_systems(method, other, obs, state, key)
        outputs.append(choose(method, other, memory, obs, state, key))
    _equal(outputs[0][0], outputs[1][0])
    _equal(outputs[0][1].team_a.members[0], outputs[1][1].team_a.members[0].members[0])
    _equal(outputs[0][1].policy_trace, outputs[1][1].policy_trace)
    _equal(outputs[0][2][0][0], outputs[1][2][0][0][0])
    np.testing.assert_array_equal(
        outputs[1][2][0][0][0], [[True, False, True, False, False]] * 2
    )


@pytest.mark.parametrize("nested", [False, True])
def test_singleton_missing_coverage_fails_before_member_initialization(
    nested: bool,
) -> None:
    _, obs, state = _game(("mage", "warrior", "mage", "rogue", "priest"))

    def forbidden(*args: Tree) -> Tree:
        del args
        pytest.fail("invalid singleton coverage must fail before member work")

    bad = marl_bgs.team(System("Unused", forbidden, init=forbidden), slots=[0])
    if nested:
        bad = marl_bgs.team(
            bad, "random", slots=["mage", ["warrior", "rogue", "priest"]]
        )
    other = marl_bgs.shared_policy(marl_bgs.policy("random"))
    with pytest.raises(ValueError, match="exactly once"):
        marl_bgs.init_systems(bad, other, obs, state, jax.random.key(17))


def test_singleton_keeps_invalid_lane_control_mask_unchanged() -> None:
    env, obs, state = _game(("mage", "warrior", "mage", "rogue", "priest"))
    inputs = env.policy_inputs(obs, state)._replace(
        valid=jnp.array([True, False]),
        controlled_mask=jnp.array(
            [[True, False, True, False, False], [True, True, True, True, True]]
        ),
    )

    def report_control(
        variables: Tree, memory: Tree, current: SystemInput, keys: Array
    ) -> SystemOutput:
        del variables, keys
        return SystemOutput(
            _zero(current), memory, learning_outputs=current.controlled_mask
        )

    method = marl_bgs.team(System("Reporter", report_control), slots=["mage"])
    keys = jax.random.split(jax.random.key(18), 2)
    assert method.init is not None
    memory = method.init(method.variables, inputs, keys)
    result = cast(
        SystemOutput, jax.jit(method.apply)(method.variables, memory, inputs, keys)
    )
    np.testing.assert_array_equal(
        result.learning_outputs[0], [[True, False, True, False, False]] * 2
    )


def test_none_control_matches_explicit_active_without_clearing_rows() -> None:
    env, obs, state = _game()
    inputs = env.policy_inputs(obs, state)
    old_constructor = SystemInput(*inputs[:5])
    assert old_constructor.controlled_mask is None
    method = marl_bgs.team(System("Counter", _act, jnp.array(2.0), init=_init))
    keys = jax.random.split(jax.random.key(1), 2)
    assert method.init is not None
    memory = method.init(method.variables, inputs, keys)
    _equal(
        method.apply(method.variables, memory, inputs, keys),
        method.apply(method.variables, memory, old_constructor, keys),
    )


@pytest.mark.parametrize("exact_keys", [False, True])
def test_five_policies_keep_physical_variables_memory_and_keys(
    exact_keys: bool,
) -> None:
    _, obs, state = _game()
    policies = tuple(
        Policy(
            str(i), _scalar, variables=jnp.array(i + 1), initial_carry=jnp.array(i * 10)
        )
        for i in range(5)
    )
    direct = marl_bgs.independent_policies(policies)
    composed = marl_bgs.team(*policies)
    other = marl_bgs.shared_policy(marl_bgs.policy("random"))
    key = jax.random.key(19)
    actor_keys = jax.random.split(key, (2, 10)) if exact_keys else None
    outputs: list[Tree] = []
    for method in (direct, composed):
        memory = marl_bgs.init_systems(method, other, obs, state, key)
        outputs.append(
            _apply_system_pair(
                _execution(method),
                _execution(other),
                method.variables,
                other.variables,
                memory,
                obs,
                state,
                key,
                actor_keys=actor_keys,
            )
        )
    _equal(outputs[0][0], outputs[1][0])
    for index in range(5):
        np.testing.assert_array_equal(
            outputs[1][1].team_a.members[index][:, index], outputs[0][1].team_a[index]
        )
        expected = np.full((2, 5), index * 10)
        expected[:, index] += index + 1
        np.testing.assert_array_equal(outputs[1][1].team_a.members[index], expected)
    _equal(outputs[0][1].policy_trace.policy_ids, outputs[1][1].policy_trace.policy_ids)


def test_nonprefix_existing_independent_system_keeps_original_slots() -> None:
    _, obs, state = _game()
    policies = tuple(
        Policy(
            str(i), _scalar, variables=jnp.array(i + 1), initial_carry=jnp.array(i * 10)
        )
        for i in range(5)
    )
    member = marl_bgs.independent_policies(policies)
    method = marl_bgs.team(member, "random", slots=[[3, 4], [0, 1, 2]])
    other = marl_bgs.shared_policy(marl_bgs.policy("random"))
    key = jax.random.key(7)
    memory = marl_bgs.init_systems(method, other, obs, state, key)
    _, memory, _ = marl_bgs.apply_systems(method, other, memory, obs, state, key)
    for index, carry in enumerate(memory.team_a.members[0]):
        np.testing.assert_array_equal(
            carry, index * 10 + (index + 1 if index >= 3 else 0)
        )
    np.testing.assert_array_equal(
        memory.policy_trace.policy_ids[:, 3:5], [[3, 4], [3, 4]]
    )


def test_nested_team_coverage_uses_incoming_mask_and_repeated_classes() -> None:
    env, obs, state = _game(("mage", "mage", "hunter", "rogue", "priest"))
    inner = marl_bgs.team("random", "random", slots=[0, 1])
    method = marl_bgs.team(inner, "random", slots=["mage", [2, 3, 4]])
    other = marl_bgs.shared_policy(marl_bgs.policy("random"))
    key = jax.random.key(3)
    memory = marl_bgs.init_systems(method, other, obs, state, key)
    actions, memory, _ = marl_bgs.apply_systems(method, other, memory, obs, state, key)
    result = env.step(key, state, actions)
    assert result[1].episode_id.shape == (2,)
    np.testing.assert_array_equal(
        memory.policy_trace.policy_ids[:, :5], [[0, 1, 2, 2, 2]] * 2
    )
    for slots in ([0, [1, 2, 3]], [[0, 1], [1, 2, 3, 4]]):
        bad = marl_bgs.team("random", "random", slots=slots)
        with pytest.raises(ValueError, match="exactly once"):
            marl_bgs.init_systems(bad, other, obs, state, key)
    short = marl_bgs.team(
        marl_bgs.independent_policies(policies=[marl_bgs.policy("random")]),
        "random",
        slots=[[3], [0, 1, 2, 4]],
    )
    with pytest.raises(ValueError, match="independent_policies"):
        marl_bgs.init_systems(short, other, obs, state, key)


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"random": 0},
        {"random": -1},
        {"random": float("nan")},
        {"random": float("inf")},
    ],
)
def test_invalid_pool_weights_fail_at_construction(weights: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="shares"):
        marl_bgs.pool(weights)


def test_pool_choice_survives_weight_change_until_selected_game_reset() -> None:
    env, obs, state = _game()
    method = marl_bgs.pool(
        {
            System("First", _act, jnp.array(1.0), init=_init): 1,
            System("Second", _act, jnp.array(2.0), init=_init): 0,
        }
    )
    other = marl_bgs.shared_policy(marl_bgs.policy("random"))
    key = jax.random.key(5)
    memory = marl_bgs.init_systems(method, other, obs, state, key)
    np.testing.assert_array_equal(memory.team_a.choice, [0, 0])
    new_variables = method.variables._replace(weights=jnp.asarray([0.0, 1.0]))
    _, memory, values = marl_bgs.apply_systems(
        method, other, memory, obs, state, key, variables_a=new_variables
    )
    np.testing.assert_array_equal(memory.team_a.choice, [0, 0])
    np.testing.assert_array_equal(values[0][1]["value"], 0)
    previous = memory.team_a.members[0]
    obs, state = env.reset(key, state=state, reset_mask=jnp.asarray([True, False]))
    _, memory, _ = marl_bgs.apply_systems(
        method, other, memory, obs, state, key, variables_a=new_variables
    )
    np.testing.assert_array_equal(memory.team_a.choice, [1, 0])
    assert int(memory.team_a.members[0][1]) == int(previous[1]) + 1
    np.testing.assert_array_equal(
        memory.policy_trace.policy_ids[:, :5], [[1] * 5, [0] * 5]
    )


def test_host_full_batch_custom_memory_unused_members_and_resource_cleanup() -> None:
    env, obs, state = _game()
    calls: list[np.ndarray] = []
    lifecycle: list[str] = []

    def initialize(variables: Tree, inputs: SystemInput, keys: Array) -> Tree:
        return {"lanes": [0] * len(inputs.valid), "opaque": object()}

    def reset(old: Tree, new: Tree, selected: Array) -> Tree:
        return {
            "lanes": [
                new["lanes"][i] if selected[i] else old["lanes"][i]
                for i in range(len(selected))
            ],
            "opaque": old["opaque"],
        }

    def apply(
        variables: Tree, memory: Tree, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, Tree]:
        assert isinstance(inputs.actors.observation.self_features, np.ndarray)
        assert inputs.active_mask.shape == (2, 5)
        assert np.all(inputs.active_mask)
        assert np.all(inputs.actors.observation.self_features[..., 6] > 0)
        calls.append(np.array(inputs.controlled_mask))
        updated = {
            "lanes": [
                old + int(valid)
                for old, valid in zip(memory["lanes"], inputs.valid, strict=True)
            ],
            "opaque": memory["opaque"],
        }
        return _zero(inputs), updated

    def scope(recording: bool) -> AbstractContextManager[None]:
        lifecycle.append("validated")

        @contextmanager
        def opened() -> Generator[None]:
            lifecycle.append("entered")
            try:
                yield
            finally:
                lifecycle.append("closed")

        return opened()

    host = System(
        "Host",
        apply,
        init=initialize,
        reset_memory=reset,
        execution="host",
        resource_scope=scope,
    )

    def unused(
        variables: Tree, memory: Tree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        pytest.fail("unused action")

    inactive = System("Unused", unused, execution="host")
    method = marl_bgs.team(
        marl_bgs.pool({host: 1, inactive: 0}), "random", slots=[[3, 4], [0, 1, 2]]
    )
    other = marl_bgs.shared_policy(marl_bgs.policy("random"))
    key = jax.random.key(5)
    memory = marl_bgs.init_systems(method, other, obs, state, key)
    opaque = memory.team_a.members[0].members[0]["opaque"]
    assert method.resource_scope is not None
    with pytest.raises(RuntimeError, match="stop"), method.resource_scope(False):
        actions, memory, _ = marl_bgs.apply_systems(
            method, other, memory, obs, state, key
        )
        env.step(key, state, actions)
        raise RuntimeError("stop")
    assert lifecycle == ["validated", "entered", "closed"]
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [[False, False, False, True, True]] * 2)
    assert memory.team_a.members[0].members[0]["opaque"] is opaque
    assert memory.team_a.members[0].members[0]["lanes"] == [1, 1]


def test_same_shaped_weights_and_member_variables_reuse_compilation() -> None:
    _, obs, state = _game()
    method = marl_bgs.pool(
        {
            System("One", _act, jnp.array(1.0), init=_init): 1,
            System("Two", _act, jnp.array(2.0), init=_init): 0,
        }
    )
    other = marl_bgs.shared_policy(marl_bgs.policy("random"))
    key = jax.random.key(1)
    memory = marl_bgs.init_systems(method, other, obs, state, key)
    traces: list[int] = []

    @jax.jit
    def run(variables: Tree, carry: SystemState) -> Tree:
        traces.append(1)

        def step(previous: SystemState, unused: None) -> tuple[SystemState, Tree]:
            _, following, outputs = marl_bgs.apply_systems(
                method, other, previous, obs, state, key, variables_a=variables
            )
            return following, outputs

        return jax.lax.scan(step, carry, None, length=2)

    first = cast(Tree, run(method.variables, memory))
    second = cast(
        Tree,
        run(
            method.variables._replace(
                members=(jnp.array(7.0), jnp.array(8.0)), weights=jnp.array([0.0, 1.0])
            ),
            memory,
        ),
    )
    jax.block_until_ready((first, second))
    assert len(traces) == 1
    np.testing.assert_array_equal(second[0].team_a.choice, first[0].team_a.choice)
    assert float(second[1][0][0]["value"][0, 0]) == 7.0


def test_compositions_use_ordinary_saved_evaluation(tmp_path: Path) -> None:
    method = marl_bgs.team("random", "tdm-alpha", slots=[[0, 1], [2, 3, 4]])
    opponent = marl_bgs.pool({"random": 1, "tdm-alpha": 0})
    result = marl_bgs.evaluate(
        method,
        opponent,
        maps=[0],
        num_episodes=2,
        num_envs=2,
        max_steps=1,
        metrics="none",
        output_dir=tmp_path,
    )
    assert len(result.completed_episode_ids) == 2
    assert result.paths is not None
    assert result.paths["episodes"].is_file()
    assert len(result.table("episodes")["episode_id"]) == 2
    assert method.components is not None
    assert method.components[0]["name"] == "random"


def test_composition_example_runs_all_public_workflows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.systems import composition_examples

    composition_examples()
    output = capsys.readouterr().out
    assert "Mixture games:" in output
    assert "Fixed opponent:" in output
    assert "Tournament completed:" in output


def test_restricted_policy_columns_skip_unowned_model_work() -> None:
    _, obs, state = _game()
    calls: list[int] = []

    def note(value: Array) -> None:
        calls.append(int(value))

    def counted(
        variables: Tree, memory: Tree, actor: Tree, masks: Tree, key: Array
    ) -> tuple[ActorAction, Tree]:
        # This diagnostic callback observes actual execution, not tracing.
        jax.debug.callback(note, variables)
        return _scalar(variables, memory, actor, masks, key)

    policies = tuple(
        Policy(str(index), counted, jnp.array(index + 1), jnp.array(0))
        for index in range(5)
    )
    method = marl_bgs.team(*policies)
    other = marl_bgs.shared_policy(marl_bgs.policy("random"))
    key = jax.random.key(31)
    memory = marl_bgs.init_systems(method, other, obs, state, key)
    result = marl_bgs.apply_systems(method, other, memory, obs, state, key)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert sorted(calls) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_supplied_pool_choices_initialize_the_chosen_members_and_keep_live_games() -> (
    None
):
    def initialize(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
        del keys
        return jnp.where(inputs.valid, variables, 0).astype(jnp.int32)

    env, obs, state = _game()
    method = marl_bgs.pool(
        {
            System("First", _act, jnp.int32(11), init=initialize): 1.0,
            System("Second", _act, jnp.int32(22), init=initialize): 0.0,
        }
    )
    other = marl_bgs.shared_policy(marl_bgs.policy("random"))
    selected = method.variables._replace(choices=jnp.asarray([1, 0], jnp.int32))
    key = jax.random.key(160)
    memory = marl_bgs.init_systems(method, other, obs, state, key, variables_a=selected)
    np.testing.assert_array_equal(memory.team_a.choice, [1, 0])
    np.testing.assert_array_equal(memory.team_a.members[0], [0, 11])
    np.testing.assert_array_equal(memory.team_a.members[1], [22, 0])
    next_choices = selected._replace(choices=jnp.asarray([0, 1], jnp.int32))
    _, memory, _ = marl_bgs.apply_systems(
        method, other, memory, obs, state, key, variables_a=next_choices
    )
    np.testing.assert_array_equal(memory.team_a.choice, [1, 0])
    obs, state = env.reset(key, state=state, reset_mask=jnp.asarray([True, False]))
    _, memory, _ = marl_bgs.apply_systems(
        method, other, memory, obs, state, key, variables_a=next_choices
    )
    np.testing.assert_array_equal(memory.team_a.choice, [0, 0])
    assert int(memory.team_a.members[0][0]) == 12
    with pytest.raises(ValueError, match="outside its members"):
        marl_bgs.init_systems(
            method,
            other,
            obs,
            state,
            key,
            variables_a=selected._replace(choices=jnp.asarray([2, 0], jnp.int32)),
        )
