"""Causal, RNG, completion, and policy-integration proof for the M7 rollout."""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.core.env import initialize_scenario_state, reset
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MOVE_STAY,
    TASK_MODE_TDM,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.evaluation.rollout import ReferenceRolloutResult, rollout
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.no_shared_obs import NoSharedObsPolicy
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.policies.scripted import (
    team_deathmatch_no_shared_obs_policy,
)

_FIRST_ENEMY_TARGET = 1 + MAX_AGENTS_PER_TEAM


def _assert_tree_arrays_exact(actual: object, expected: object) -> None:
    """Require identical PyTree structures, shapes, dtypes, and values."""
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        expected
    )
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        actual_array = np.asarray(actual_leaf)
        expected_array = np.asarray(expected_leaf)
        assert actual_array.shape == expected_array.shape
        assert actual_array.dtype == expected_array.dtype
        np.testing.assert_array_equal(actual_array, expected_array)


def _first_supported_action(
    observation: Observation,
    action_mask: ActionMask,
    key: Array,
) -> ActorAction:
    """Choose the first exact movement and combat support entries."""
    del observation, key
    combat_flat = jnp.argmax(
        action_mask.select_target_use_ultimate_joint_mask.reshape(-1)
    )
    target, ultimate = jnp.unravel_index(
        combat_flat,
        action_mask.select_target_use_ultimate_joint_mask.shape,
    )
    return ActorAction(
        move=jnp.argmax(action_mask.move_mask).astype(jnp.int32),
        select_target=target.astype(jnp.int32),
        use_ultimate=ultimate.astype(jnp.int32),
    )


def _last_supported_action(
    observation: Observation,
    action_mask: ActionMask,
    key: Array,
) -> ActorAction:
    """Choose the last exact movement and combat support entries."""
    del observation, key
    move = (
        action_mask.move_mask.shape[0] - 1 - jnp.argmax(jnp.flip(action_mask.move_mask))
    )
    combat_support = action_mask.select_target_use_ultimate_joint_mask
    combat_flat = (
        combat_support.size - 1 - jnp.argmax(jnp.flip(combat_support.reshape(-1)))
    )
    target, ultimate = jnp.unravel_index(combat_flat, combat_support.shape)
    return ActorAction(
        move=move.astype(jnp.int32),
        select_target=target.astype(jnp.int32),
        use_ultimate=ultimate.astype(jnp.int32),
    )


def _attack_first_enemy(
    observation: Observation,
    action_mask: ActionMask,
    key: Array,
) -> ActorAction:
    """Use Basic on the first enemy when legal, otherwise remain inert."""
    del observation, key
    can_attack = action_mask.select_target_use_ultimate_joint_mask[
        _FIRST_ENEMY_TARGET, 0
    ]
    return ActorAction(
        move=jnp.asarray(MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.where(can_attack, _FIRST_ENEMY_TARGET, 0).astype(jnp.int32),
        use_ultimate=jnp.asarray(0, dtype=jnp.int32),
    )


def _key_bits(key: Array) -> Array:
    """Return one deterministic scalar code for an actor key."""
    return jax.random.bits(key, (), dtype=jnp.uint32).astype(jnp.int32)


def _without_first_row(leaf: Array) -> Array:
    """Drop the first stacked rollout row from one PyTree leaf."""
    return leaf[1:]


def _without_last_row(leaf: Array) -> Array:
    """Drop the last stacked rollout row from one PyTree leaf."""
    return leaf[:-1]


def _team_a_key_policy(
    observation: Observation,
    action_mask: ActionMask,
    key: Array,
) -> ActorAction:
    """Expose Team A actor-key identity through submitted action values."""
    del observation, action_mask
    value = _key_bits(key)
    return ActorAction(move=value, select_target=value, use_ultimate=value)


def _team_b_key_policy(
    observation: Observation,
    action_mask: ActionMask,
    key: Array,
) -> ActorAction:
    """Expose Team B policy identity while retaining its actor-key identity."""
    del observation, action_mask
    value = jnp.bitwise_not(_key_bits(key))
    return ActorAction(move=value, select_target=value, use_ultimate=value)


def _tdm_config(
    *,
    team_sizes: tuple[int, int] = (2, 2),
    score_threshold: int = 100,
    max_steps: int = 3,
) -> EnvConfig:
    """Return a deterministic TDM configuration without reset shields."""
    return evaluation_env_config(
        team_sizes=team_sizes,
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=score_threshold,
        max_steps=max_steps,
    )._replace(spawn_shield_duration_steps=0)


def _reset_rollout_inputs(
    config: EnvConfig,
) -> tuple[EnvState, Observation, ActionMask]:
    """Return the three public rollout inputs produced by reset."""
    state, observation, action_mask, _ = reset(config, jax.random.key(0))
    return state, observation, action_mask


def _threshold_start(
    config: EnvConfig,
    *,
    step_count: int = 0,
) -> tuple[EnvState, Observation, ActionMask]:
    """Place one low-health opponent inside the first Team A Mage's Basic range."""
    state, _, _, _ = reset(config, jax.random.key(0))
    positions = state.agent_positions
    positions = positions.at[0].set(jnp.asarray((4.0, 4.0), dtype=jnp.float32))
    positions = positions.at[MAX_AGENTS_PER_TEAM].set(
        jnp.asarray((6.5, 4.0), dtype=jnp.float32)
    )
    state = state._replace(
        step_count=jnp.asarray(step_count, dtype=jnp.int32),
        agent_positions=positions,
        current_health=state.current_health.at[MAX_AGENTS_PER_TEAM].set(1.0),
    )
    state, observation, action_mask, _ = initialize_scenario_state(state, config)
    return state, observation, action_mask


def _run(
    config: EnvConfig,
    policy_a: NoSharedObsPolicy,
    policy_b: NoSharedObsPolicy,
    *,
    key: Array | None = None,
    inputs: tuple[EnvState, Observation, ActionMask] | None = None,
) -> object:
    """Run the public C1 entry point from reset or a curated valid start."""
    state, observation, action_mask = (
        _reset_rollout_inputs(config) if inputs is None else inputs
    )
    return rollout(
        config,
        state,
        observation,
        action_mask,
        jax.random.key(17) if key is None else key,
        policy_a,
        policy_b,
        execution_information_mode="no_shared_obs",
    )


def _unpack(
    history: object,
) -> tuple[
    tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
    tuple[EnvState, ActionMask, Action],
]:
    """Give the fixed C1 history tuple a readable local type boundary."""
    result = cast(ReferenceRolloutResult, history)
    return result.successors, result.currents


def _assert_real_actions_are_masked(history: object) -> None:
    """Require every submitted real action to belong to its stored current mask."""
    successors, currents = _unpack(history)
    _, _, _, _, _, infos = successors
    _, current_masks, submitted = currents
    valid = infos.transition_facts.has_transition

    move_supported = jnp.take_along_axis(
        current_masks.move_mask,
        submitted.move[..., None],
        axis=-1,
    )[..., 0]
    transition_index = jnp.arange(submitted.move.shape[0])[:, None]
    actor_index = jnp.arange(MAX_AGENT_SLOTS)[None, :]
    combat_supported = current_masks.select_target_use_ultimate_joint_mask[
        transition_index,
        actor_index,
        submitted.select_target,
        submitted.use_ultimate,
    ]
    assert bool(jnp.all(jnp.logical_or(~valid[:, None], move_supported)))
    assert bool(jnp.all(jnp.logical_or(~valid[:, None], combat_supported)))


def _assert_submitted_equals_accepted(history: object) -> None:
    """Require zero core rejection for every real transition."""
    successors, currents = _unpack(history)
    _, _, _, _, _, infos = successors
    _, _, submitted = currents
    acceptance = infos.transition_facts.action_acceptance_facts
    valid = np.asarray(infos.transition_facts.has_transition, dtype=np.bool_)
    for submitted_leaf, accepted_leaf in zip(
        jax.tree_util.tree_leaves(submitted),
        jax.tree_util.tree_leaves(acceptance.accepted_joint_action),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(submitted_leaf)[valid],
            np.asarray(accepted_leaf)[valid],
        )
    assert not bool(
        jnp.any(
            jnp.logical_and(
                valid[:, None],
                acceptance.submitted_action_tuple_is_out_of_domain_by_actor,
            )
        )
    )
    assert not bool(
        jnp.any(
            jnp.logical_and(
                valid[:, None],
                acceptance.in_domain_move_action_is_rejected_by_actor,
            )
        )
    )
    assert not bool(
        jnp.any(
            jnp.logical_and(
                valid[:, None],
                acceptance.in_domain_combat_action_pair_is_rejected_by_actor,
            )
        )
    )


def test_rollout_is_exactly_reproducible_and_preserves_fixed_history_contract() -> None:
    """One root key reproduces every stored frame, action, fact, and dtype."""
    config = _tdm_config(max_steps=3)
    inputs = _reset_rollout_inputs(config)
    first = _run(
        config,
        _first_supported_action,
        _last_supported_action,
        inputs=inputs,
    )
    repeated = _run(
        config,
        _first_supported_action,
        _last_supported_action,
        inputs=inputs,
    )
    _assert_tree_arrays_exact(first, repeated)
    _assert_real_actions_are_masked(first)
    _assert_submitted_equals_accepted(first)

    successors, currents = _unpack(first)
    successor_states, _, rewards, done_flags, successor_masks, infos = successors
    current_states, current_masks, submitted = currents
    assert successor_states.step_count.shape == (config.max_steps,)
    assert current_states.step_count.shape == (config.max_steps,)
    assert rewards.rewards.shape == (config.max_steps, MAX_AGENT_SLOTS)
    assert submitted.move.shape == (config.max_steps, MAX_AGENT_SLOTS)
    assert submitted.move.dtype == jnp.int32
    assert submitted.select_target.dtype == jnp.int32
    assert submitted.use_ultimate.dtype == jnp.int32
    assert done_flags.terminated.dtype == jnp.bool_
    assert done_flags.truncated.dtype == jnp.bool_
    assert infos.transition_facts.has_transition.dtype == jnp.bool_
    _assert_tree_arrays_exact(
        jax.tree.map(_without_first_row, current_states),
        jax.tree.map(_without_last_row, successor_states),
    )
    _assert_tree_arrays_exact(
        jax.tree.map(_without_first_row, current_masks),
        jax.tree.map(_without_last_row, successor_masks),
    )


@pytest.mark.parametrize(
    "root_key",
    (
        pytest.param(jax.random.key(23), id="typed"),
        pytest.param(jax.random.PRNGKey(23), id="legacy"),
    ),
)
def test_rng_tree_preserves_global_actor_slots_and_team_policy_identity(
    root_key: Array,
) -> None:
    """Epoch/team splitting assigns each policy its stable five-slot key block."""
    config = _tdm_config(max_steps=3)
    history = _run(
        config,
        _team_a_key_policy,
        _team_b_key_policy,
        key=root_key,
    )
    _, (_, _, submitted) = _unpack(history)

    expected_rows: list[Array] = []
    for episode_key in jax.random.split(root_key, config.max_steps):
        _, team_key = jax.random.split(episode_key)
        actor_keys = jax.random.split(team_key, MAX_AGENT_SLOTS)
        actor_codes = jax.vmap(_key_bits)(actor_keys)
        expected_rows.append(
            jnp.concatenate(
                (
                    actor_codes[:MAX_AGENTS_PER_TEAM],
                    jnp.bitwise_not(actor_codes[MAX_AGENTS_PER_TEAM:]),
                )
            )
        )
    expected = jnp.stack(expected_rows)
    assert bool(jnp.array_equal(submitted.move, expected))
    assert bool(jnp.array_equal(submitted.select_target, expected))
    assert bool(jnp.array_equal(submitted.use_ultimate, expected))


def test_typed_and_legacy_root_keys_produce_identical_trajectories() -> None:
    """JAX key representation does not change the C1 stochastic trajectory."""
    config = _tdm_config(max_steps=2)
    inputs = _reset_rollout_inputs(config)
    typed = _run(
        config,
        random_policy,
        random_policy,
        key=jax.random.key(31),
        inputs=inputs,
    )
    legacy = _run(
        config,
        random_policy,
        random_policy,
        key=jax.random.PRNGKey(31),
        inputs=inputs,
    )
    _assert_tree_arrays_exact(typed, legacy)


def test_current_mask_governs_action_and_terminal_successor_is_retained() -> None:
    """The accepted lethal action uses m_t even though m_t+1 rejects its target."""
    config = _tdm_config(team_sizes=(1, 1), score_threshold=1, max_steps=3)
    inputs = _threshold_start(config)
    history = _run(
        config,
        _attack_first_enemy,
        _first_supported_action,
        inputs=inputs,
    )
    successors, currents = _unpack(history)
    successor_states, _, _, done_flags, successor_masks, infos = successors
    current_states, current_masks, submitted = currents
    accepted = infos.transition_facts.action_acceptance_facts.accepted_joint_action

    assert bool(current_masks.select_target_use_ultimate_joint_mask[0, 0, 6, 0])
    assert int(submitted.select_target[0, 0]) == _FIRST_ENEMY_TARGET
    assert int(accepted.select_target[0, 0]) == _FIRST_ENEMY_TARGET
    assert not bool(successor_masks.select_target_use_ultimate_joint_mask[0, 0, 6, 0])
    assert not bool(successor_states.alive_mask[0, MAX_AGENTS_PER_TEAM])
    assert bool(done_flags.terminated[0])
    assert not bool(done_flags.truncated[0])
    assert bool(
        jnp.array_equal(
            infos.transition_facts.has_transition,
            jnp.asarray((True, False, False), dtype=jnp.bool_),
        )
    )
    assert int(jnp.sum(infos.transition_facts.has_transition)) == 1
    assert int(current_states.step_count[0]) == 0
    assert int(successor_states.step_count[0]) == 1
    assert bool(jnp.all(current_states.step_count[1:] == 1))
    assert bool(jnp.all(successor_states.step_count[1:] == 1))
    assert bool(jnp.all(submitted.move[1:] == 0))
    assert bool(jnp.all(submitted.select_target[1:] == 0))
    assert bool(jnp.all(submitted.use_ultimate[1:] == 0))
    assert bool(jnp.all(infos.transition_facts.transition_start_step_count[1:] == -1))


@pytest.mark.parametrize(
    (
        "score_threshold",
        "max_steps",
        "threshold_start",
        "expected_terminated",
        "expected_truncated",
    ),
    (
        pytest.param(100, 2, False, False, True, id="horizon-only"),
        pytest.param(1, 3, True, True, False, id="threshold-only"),
        pytest.param(1, 1, True, True, True, id="coincident"),
    ),
)
def test_completion_bases_are_independent_and_retain_the_last_real_successor(
    score_threshold: int,
    max_steps: int,
    threshold_start: bool,
    expected_terminated: bool,
    expected_truncated: bool,
) -> None:
    """Threshold, horizon, and coincident completion preserve exact final truth."""
    config = _tdm_config(
        team_sizes=(1, 1),
        score_threshold=score_threshold,
        max_steps=max_steps,
    )
    inputs = (
        _threshold_start(config) if threshold_start else _reset_rollout_inputs(config)
    )
    history = _run(
        config,
        _attack_first_enemy if threshold_start else _first_supported_action,
        _first_supported_action,
        inputs=inputs,
    )
    successors, _ = _unpack(history)
    successor_states, _, _, done_flags, _, infos = successors
    valid = np.asarray(infos.transition_facts.has_transition, dtype=np.bool_)
    final_real_index = int(np.flatnonzero(valid)[-1])

    assert bool(done_flags.terminated[final_real_index]) is expected_terminated
    assert bool(done_flags.truncated[final_real_index]) is expected_truncated
    assert bool(done_flags.done[final_real_index])
    assert int(successor_states.step_count[final_real_index]) == final_real_index + 1
    assert np.array_equal(valid, np.arange(max_steps) <= final_real_index)


@pytest.mark.parametrize(
    "policy",
    (
        pytest.param(random_policy, id="random-valid"),
        pytest.param(
            team_deathmatch_no_shared_obs_policy,
            id="scripted-team-deathmatch",
        ),
    ),
)
def test_mask_valid_policies_run_every_fixed_topology_without_rejection(
    policy: Callable[[Observation, ActionMask, Array], ActorAction],
) -> None:
    """Random-valid and scripted actors integrate from 1v1 through 5v5."""
    for team_size in range(1, MAX_AGENTS_PER_TEAM + 1):
        config = _tdm_config(team_sizes=(team_size, team_size), max_steps=1)
        history = _run(config, policy, policy)
        _assert_real_actions_are_masked(history)
        _assert_submitted_equals_accepted(history)


def test_disabled_jit_and_public_compiled_rollout_are_exactly_equal() -> None:
    """The selected compiled scan preserves eager public semantics exactly."""
    config = _tdm_config(max_steps=2)
    inputs = _reset_rollout_inputs(config)
    compiled = _run(
        config,
        _first_supported_action,
        _last_supported_action,
        inputs=inputs,
    )
    with jax.disable_jit():
        eager = _run(
            config,
            _first_supported_action,
            _last_supported_action,
            inputs=inputs,
        )
    _assert_tree_arrays_exact(eager, compiled)
