"""Public actor adapters, dynamic variables and the single redaction barrier."""

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
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyApply,
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
from marl_battlegrounds.policies.shared_obs import (
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)


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


@pytest.mark.parametrize("name", ["random", "tdm-alpha", "tdm-beta"])
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
        original = (
            reactive_tdm_alpha_policy
            if name == "tdm-alpha"
            else reactive_tdm_beta_policy
        )
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
        slot = int(actor.global_slot)
        seen.append(slot)

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
    assert seen == list(range(10))
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
        return ActorAction(variables, zero, zero), memory + actor.global_slot + 1

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
    np.testing.assert_array_equal(next_a, 2 * np.arange(1, 6))
    np.testing.assert_array_equal(next_b, 2 * np.arange(6, 11))
    batched = jnp.stack((next_a, next_b))
    reset = select_policy_carry(jnp.asarray([True, False]), initial, batched)
    np.testing.assert_array_equal(reset[0], np.zeros(5))
    np.testing.assert_array_equal(reset[1], next_b)


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
