"""Check policy adapters, changing numerical inputs and source-data filtering."""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import (
    TASK_MODE_TDM,
    TEAM_A_ID,
    TEAM_B_ID,
    Action,
    ActionMask,
    EnvConfig,
    Observation,
)
from marl_battlegrounds.environment import make
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyApply,
    PolicyExecution,
    PolicyTree,
    apply_policies,
    freeze_variables,
    initial_policy_carry,
    policy,
    select_policy_carry,
)
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.input import (
    ActorInput,
    build_actor_input,
    build_observations,
)
from marl_battlegrounds.policies.no_shared_obs import execute_no_shared_obs_team_policy
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.policies.reactive_tdm_alpha import reactive_tdm_alpha_policy
from marl_battlegrounds.policies.reactive_tdm_beta import reactive_tdm_beta_policy
from marl_battlegrounds.policies.reactive_tdm_gamma import reactive_tdm_gamma_policy
from marl_battlegrounds.policies.shared_obs import (
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)


@pytest.mark.parametrize("execution", ["jax", "host"])
def test_every_source_subset_is_redacted_at_the_actual_policy_boundary(
    execution: PolicyExecution,
) -> None:
    config = evaluation_env_config(team_sizes=(5, 5))
    _, observation, mask, _ = core.reset(config, jax.random.key(0))
    # Nonzero synthetic objective rows make omitted objective redaction visible.
    observation = observation._replace(
        objective_features=jnp.broadcast_to(
            jnp.arange(1, 11, dtype=jnp.float32)[:, None, None], (10, 8, 12)
        )
    )
    compact = build_observations(observation, config)
    keys = jax.random.split(jax.random.key(12), 10)

    def inspect(
        variables: PolicyTree,
        memory: PolicyTree,
        actor: ActorInput,
        action_mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        del variables, memory, action_mask, key
        assert ActorInput._fields == (
            "observation",
            "source_bank",
            "source_availability",
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        return ActorAction(zero, zero, zero), actor

    run = cast(
        Callable[..., tuple[Action, ActorInput, ActorInput]],
        (
            jax.jit(
                apply_policies,
                static_argnums=(0, 1),
                static_argnames=("execution_a", "execution_b"),
            )
            if execution == "jax"
            else apply_policies
        ),
    )
    unmasked = build_shared_obs_sensor_source_bank(observation)
    for bits in range(32):
        selected = (jnp.asarray(bits) & (1 << (jnp.arange(10) % 5))) != 0
        availability = compact.source_availability & selected[None, :]
        _, team_a, team_b = run(
            inspect,
            inspect,
            (),
            (),
            (),
            (),
            compact._replace(source_availability=availability),
            mask,
            keys,
            execution_a=execution,
            execution_b=execution,
        )
        for team_index, delivered in enumerate((team_a, team_b)):
            start = team_index * 5
            expected = np.asarray(availability[start : start + 5, start : start + 5])
            np.testing.assert_array_equal(delivered.source_availability, expected)
            for actual_leaf, raw_leaf in zip(
                delivered.source_bank, unmasked, strict=True
            ):
                raw = np.broadcast_to(
                    np.asarray(raw_leaf[team_index]), np.asarray(actual_leaf).shape
                )
                actual = np.asarray(actual_leaf)
                np.testing.assert_array_equal(actual[expected], raw[expected])
                assert not actual[~expected].any()


def _assert_tree(actual: object, expected: object) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def _inputs() -> tuple[EnvConfig, Observation, ActionMask, Array]:
    config = evaluation_env_config(
        team_sizes=(3, 2),
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=20,
    )
    state, _, _, _ = core.reset(config, jax.random.key(0))
    state = state._replace(
        current_health=state.current_health.at[1].set(0),
        alive_mask=state.alive_mask.at[1].set(False),
    )
    _, observation, mask, _ = core.initialize_scenario_state(state, config)
    return config, observation, mask, jax.random.split(jax.random.key(13), 10)


@pytest.mark.parametrize("name", ["random", "tdm-alpha", "tdm-beta", "tdm-gamma"])
def test_named_adapters_match_existing_controllers_with_dead_and_inactive_slots(
    name: str,
) -> None:
    config, observation, mask, keys = _inputs()
    adapted = policy(name)
    run = cast(
        Callable[..., tuple[Action, PolicyTree, PolicyTree]],
        jax.jit(
            apply_policies,
            static_argnums=(0, 1),
        ),
    )
    actual, memory_a, memory_b = run(
        adapted.apply,
        adapted.apply,
        (),
        (),
        (),
        (),
        build_observations(observation, config),
        mask,
        keys,
    )
    if name == "random":
        execute = cast(Callable[..., ActorAction], execute_no_shared_obs_team_policy)
        first = execute(observation, mask, keys, random_policy, TEAM_A_ID)
        second = execute(observation, mask, keys, random_policy, TEAM_B_ID)
    else:
        execute = cast(Callable[..., ActorAction], execute_shared_obs_team_policy)
        original = {
            "tdm-alpha": reactive_tdm_alpha_policy,
            "tdm-beta": reactive_tdm_beta_policy,
            "tdm-gamma": reactive_tdm_gamma_policy,
        }[name]
        bank = build_shared_obs_sensor_source_bank(observation)
        availability = build_observations(observation, config).source_availability
        first = execute(
            observation, mask, keys, bank, availability, original, TEAM_A_ID
        )
        second = execute(
            observation, mask, keys, bank, availability, original, TEAM_B_ID
        )
    _assert_tree(actual, build_joint_action_from_actor_actions(first, second))
    assert memory_a == memory_b == ()


def test_host_actor_receives_exact_authorized_reference_and_both_teams_same_epoch() -> (
    None
):
    config, observation, mask, keys = _inputs()
    reference = build_actor_input(observation, config)
    seen: list[int] = []

    def inspect(
        variables: PolicyTree,
        memory: PolicyTree,
        actor: ActorInput,
        action_mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        del variables, key
        slot = len(seen)  # Test-only call order, outside the delivered input.
        seen.append(int(actor.observation.self_ally_index))
        assert not hasattr(actor, "global_slot")

        def row(value: Array) -> Array:
            return value[slot]

        expected = jax.tree.map(row, reference)
        _assert_tree(actor, expected)
        _assert_tree(action_mask, jax.tree.map(row, mask))
        for leaf in actor.source_bank:
            assert not np.any(np.asarray(leaf)[~np.asarray(actor.source_availability)])
        zero = jnp.asarray(0, jnp.int32)
        return ActorAction(zero, zero, zero), memory

    apply_policies(
        inspect,
        inspect,
        (),
        (),
        (),
        (),
        build_observations(observation, config),
        mask,
        keys,
        execution_a="host",
        execution_b="host",
    )
    assert seen == [0, 1, 2, 0, 0, 0, 1, 0, 0, 0]
    compact_bytes = sum(
        leaf.nbytes for leaf in jax.tree.leaves(build_observations(observation, config))
    )
    expanded_bytes = sum(leaf.nbytes for leaf in jax.tree.leaves(reference))
    assert compact_bytes < expanded_bytes / 2


def test_dynamic_variables_and_recurrent_memory_do_not_retrace_or_share_actors() -> (
    None
):
    config, observation, mask, keys = _inputs()
    traces: list[int] = []

    def apply(
        variables: PolicyTree,
        memory: PolicyTree,
        actor: ActorInput,
        action_mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        del action_mask, key
        traces.append(1)
        zero = jnp.asarray(0, jnp.int32)
        return (
            ActorAction(variables, zero, zero),
            memory + actor.observation.self_ally_index + 1,
        )

    run = cast(
        Callable[..., tuple[Action, PolicyTree, PolicyTree]],
        jax.jit(
            apply_policies,
            static_argnums=(0, 1),
        ),
    )
    initial = initial_policy_carry(jnp.asarray(0, jnp.int32), 2)
    first, carry_a, carry_b = run(
        apply,
        apply,
        jnp.int32(0),
        jnp.int32(0),
        initial[0],
        initial[0],
        build_observations(observation, config),
        mask,
        keys,
    )
    trace_count = len(traces)
    second, next_a, next_b = run(
        apply,
        apply,
        jnp.int32(1),
        jnp.int32(2),
        carry_a,
        carry_b,
        build_observations(observation, config),
        mask,
        keys,
    )
    assert len(traces) == trace_count
    np.testing.assert_array_equal(first.move, np.zeros(10))
    np.testing.assert_array_equal(second.move, [1] * 5 + [2] * 5)
    np.testing.assert_array_equal(next_a, 2 * np.asarray([1, 2, 3, 1, 1]))
    np.testing.assert_array_equal(next_b, 2 * np.asarray([1, 2, 1, 1, 1]))
    batched = jnp.stack((next_a, next_b))
    reset = select_policy_carry(jnp.asarray([True, False]), initial, batched)
    np.testing.assert_array_equal(reset[0], np.zeros(5))
    np.testing.assert_array_equal(reset[1], next_b)


def test_recurrent_mixture_scan_and_partial_reset_reuse_compilation() -> None:
    env = make("tdm", num_envs=2, metrics="none")

    def stack(first: Array, second: Array) -> Array:
        return jnp.stack((first, second))

    configs = jax.tree.map(
        stack,
        evaluation_env_config(team_sizes=(3, 2), max_steps=1),
        evaluation_env_config(team_sizes=(1, 5), max_steps=3),
    )
    traces: list[int] = []

    def mixture(
        variables: PolicyTree,
        memory: PolicyTree,
        actor: ActorInput,
        mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        del key
        choice = variables[actor.observation.self_ally_index % 2]
        move = jnp.where(mask.move_mask[choice], choice, 0).astype(jnp.int32)
        zero = jnp.int32(0)
        return ActorAction(move, zero, zero), memory + 1

    @jax.jit
    def run(key: Array, settings: EnvConfig, weights: Array, memory: Array) -> object:
        traces.append(1)
        observations, state = env.reset(key, settings, episode_id=jnp.asarray([11, 12]))

        def advance(
            carry: PolicyTree, step_key: Array
        ) -> tuple[PolicyTree, PolicyTree]:
            obs, current, memory_a, memory_b = carry
            finished = current.done.done
            obs, current = env.reset(
                step_key,
                settings,
                episode_id=current.episode_id + finished.astype(jnp.int32) * 10,
                state=current,
                reset_mask=finished,
            )
            memory_a = select_policy_carry(finished, jnp.zeros_like(memory_a), memory_a)
            memory_b = select_policy_carry(finished, jnp.zeros_like(memory_b), memory_b)
            actions, next_a, next_b = jax.vmap(
                apply_policies, in_axes=(None, None, None, None, 0, 0, 0, 0, 0)
            )(
                mixture,
                mixture,
                weights,
                weights[::-1],
                memory_a,
                memory_b,
                obs,
                current.action_mask,
                jax.random.split(step_key, (2, 10)),
            )
            result = env.step(step_key, current, actions)
            return (result[0], result[1], next_a, next_b), (result[4].completed, next_a)

        return jax.lax.scan(
            advance, (observations, state, memory, memory), jax.random.split(key, 5)
        )

    initial_memory = jnp.zeros((2, 5), dtype=jnp.int32)
    first = cast(
        PolicyTree, run(jax.random.key(1), configs, jnp.asarray([0, 1]), initial_memory)
    )
    changed = jax.tree.map(
        stack,
        evaluation_env_config(team_sizes=(5, 1), max_steps=1),
        evaluation_env_config(team_sizes=(2, 3), max_steps=3),
    )
    second = cast(
        PolicyTree,
        run(jax.random.key(9), changed, jnp.asarray([2, 3]), initial_memory + 7),
    )
    assert traces == [1]
    expected_completed = [
        [True, False],
        [True, False],
        [True, True],
        [True, False],
        [True, False],
    ]
    for result in (first, second):
        np.testing.assert_array_equal(result[1][0], expected_completed)
        np.testing.assert_array_equal(result[0][2], [[1] * 5, [2] * 5])
        np.testing.assert_array_equal(result[0][3], [[1] * 5, [2] * 5])
        np.testing.assert_array_equal(result[0][1].core_state.step_count, [1, 2])
        assert result[0][1].priority is None
        assert result[0][1].full is None


def test_mutable_variables_are_snapshotted_and_unknown_controller_fails() -> None:
    mutable = np.asarray([2.0], dtype=np.float32)
    frozen = freeze_variables({"weights": mutable})
    carry = initial_policy_carry(mutable, 2)
    mutable[0] = 9
    np.testing.assert_array_equal(frozen["weights"], [2])
    np.testing.assert_array_equal(carry, np.full((2, 5, 1), 2))
    with pytest.raises(ValueError, match="unknown policy"):
        policy("unavailable")
    with pytest.raises(ValueError, match="name"):
        Policy("", policy("random").apply)


@pytest.mark.parametrize("kind", ["type", "shape", "dtype"])
def test_malformed_provider_action_is_rejected_before_core(kind: str) -> None:
    config, observation, mask, keys = _inputs()

    def malformed(*args: object) -> tuple[object, tuple[()]]:
        del args
        zero = jnp.asarray(0, jnp.int32)
        if kind == "type":
            return (zero, zero, zero), ()
        invalid = jnp.zeros(1, jnp.int32) if kind == "shape" else jnp.asarray(0.0)
        return ActorAction(invalid, zero, zero), ()

    with pytest.raises(TypeError, match=r"ActorAction|int32 scalar"):
        apply_policies(
            cast(PolicyApply, malformed),
            policy("random").apply,
            (),
            (),
            (),
            (),
            build_observations(observation, config),
            mask,
            keys,
            execution_a="host",
        )
