"""Legality, determinism, and core-integration proof for random-valid policy."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MOVE_STAY,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    TEAM_A_ID,
    TEAM_B_ID,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.no_shared_obs import (
    execute_no_shared_obs_team_policy,
)
from marl_battlegrounds.policies.random_valid import random_policy


def _assert_tree_arrays_exact(actual: object, expected: object) -> None:
    """Require identical PyTree structures, dtypes, shapes, and values."""
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
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        np.testing.assert_array_equal(actual_array, expected_array)


def _local_action_mask(
    *,
    move_support: tuple[int, ...],
    combat_support: tuple[tuple[int, int], ...],
) -> ActionMask:
    """Build one exact local support and derive only its marginal conveniences."""
    move_mask = jnp.zeros((NUM_MOVE_ACTIONS,), dtype=jnp.bool_)
    for move_action in move_support:
        move_mask = move_mask.at[move_action].set(True)

    joint_mask = jnp.zeros((NUM_TARGET_ACTIONS, NUM_ULTIMATE_ACTIONS), dtype=jnp.bool_)
    for select_target_action, use_ultimate_action in combat_support:
        joint_mask = joint_mask.at[select_target_action, use_ultimate_action].set(True)

    return ActionMask(
        move_mask=move_mask,
        select_target_mask=jnp.any(joint_mask, axis=-1),
        use_ultimate_mask=jnp.any(joint_mask, axis=0),
        select_target_use_ultimate_joint_mask=joint_mask,
    )


def _first_recipient_row(leaf: jax.Array) -> jax.Array:
    """Remove only the public recipient axis from one observation leaf."""
    return leaf[0]


@pytest.fixture(scope="module")
def actor_observation() -> Observation:
    """Return one complete rank-reduced actor observation from the public reset."""
    config = evaluation_env_config(team_sizes=(1, 1))
    _, observation, _, _ = reset(config, jax.random.key(0))
    return cast(Observation, jax.tree.map(_first_recipient_row, observation))


def test_random_policy_samples_only_the_exact_non_cartesian_support(
    actor_observation: Observation,
) -> None:
    """Bounded seeds never produce a movement or combat pair outside its mask."""
    action_mask = _local_action_mask(
        move_support=(MOVE_STAY, 3, 8),
        combat_support=((1, 0), (6, 1)),
    )
    misleading_marginals = action_mask._replace(
        select_target_mask=jnp.ones((NUM_TARGET_ACTIONS,), dtype=jnp.bool_),
        use_ultimate_mask=jnp.ones((NUM_ULTIMATE_ACTIONS,), dtype=jnp.bool_),
    )
    actor_keys = jax.random.split(jax.random.key(31), 512)
    mapped_policy = jax.vmap(random_policy, in_axes=(None, None, 0))

    actions = mapped_policy(actor_observation, action_mask, actor_keys)
    actions_with_misleading_marginals = mapped_policy(
        actor_observation,
        misleading_marginals,
        actor_keys,
    )

    assert bool(jnp.all(action_mask.move_mask[actions.move]))
    assert bool(
        jnp.all(
            action_mask.select_target_use_ultimate_joint_mask[
                actions.select_target, actions.use_ultimate
            ]
        )
    )
    _assert_tree_arrays_exact(actions_with_misleading_marginals, actions)
    for leaf in jax.tree_util.tree_leaves(actions):
        assert leaf.shape == (512,)
        assert leaf.dtype == jnp.int32


@pytest.mark.parametrize(
    ("move_support", "combat_support", "expected_action"),
    (
        ((8,), ((10, 1),), (8, 10, 1)),
        ((MOVE_STAY,), ((0, 0),), (MOVE_STAY, 0, 0)),
    ),
)
def test_singleton_and_nonacting_supports_return_the_only_legal_action(
    actor_observation: Observation,
    move_support: tuple[int, ...],
    combat_support: tuple[tuple[int, int], ...],
    expected_action: tuple[int, int, int],
) -> None:
    """Singleton supports are deterministic, including dead/inactive no-op rows."""
    action_mask = _local_action_mask(
        move_support=move_support,
        combat_support=combat_support,
    )
    action = random_policy(actor_observation, action_mask, jax.random.key(37))
    expected = ActorAction(
        move=jnp.asarray(expected_action[0], dtype=jnp.int32),
        select_target=jnp.asarray(expected_action[1], dtype=jnp.int32),
        use_ultimate=jnp.asarray(expected_action[2], dtype=jnp.int32),
    )
    _assert_tree_arrays_exact(action, expected)
    for leaf in jax.tree_util.tree_leaves(action):
        assert leaf.shape == ()
        assert leaf.dtype == jnp.int32


def test_random_policy_repeats_under_eager_jit_and_legacy_key_forms(
    actor_observation: Observation,
) -> None:
    """A fixed input/key is pure across supported execution and key layouts."""
    action_mask = _local_action_mask(
        move_support=(0, 2, 4, 6, 8),
        combat_support=((0, 0), (2, 0), (7, 1)),
    )
    typed_key = jax.random.key(41)
    legacy_key = jax.random.key_data(typed_key)

    eager = random_policy(actor_observation, action_mask, typed_key)
    repeated = random_policy(actor_observation, action_mask, typed_key)
    compiled = cast(
        ActorAction,
        jax.jit(random_policy)(actor_observation, action_mask, typed_key),
    )
    legacy = random_policy(actor_observation, action_mask, legacy_key)

    _assert_tree_arrays_exact(repeated, eager)
    _assert_tree_arrays_exact(compiled, eager)
    _assert_tree_arrays_exact(legacy, eager)


def test_random_policy_preserves_float32_sampling_and_int32_action_abi_with_x64(
    actor_observation: Observation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global x64 mode must not widen policy sampling or submitted actions."""
    action_mask = _local_action_mask(
        move_support=(0, 2, 4, 6, 8),
        combat_support=((0, 0), (2, 0), (7, 1)),
    )
    categorical = jax.random.categorical
    sampled_logits_dtypes: list[str] = []

    def record_categorical(key: jax.Array, logits: jax.Array) -> jax.Array:
        sampled_logits_dtypes.append(str(logits.dtype))
        return categorical(key, logits)

    monkeypatch.setattr(jax.random, "categorical", record_categorical)

    with jax.enable_x64(True):
        action = random_policy(actor_observation, action_mask, jax.random.key(42))

    assert sampled_logits_dtypes == ["float32", "float32"]
    for leaf in jax.tree_util.tree_leaves(action):
        assert leaf.shape == ()
        assert leaf.dtype == jnp.int32


def test_random_policy_fixed_team_execution_is_mask_legal_and_core_accepted() -> None:
    """One public observe-choose-assemble-step path accepts every submitted head."""
    config = evaluation_env_config(team_sizes=(1, 1), max_steps=10)
    state, observation, action_mask, _ = reset(config, jax.random.key(43))
    actor_keys = jax.random.split(jax.random.key(47), MAX_AGENT_SLOTS)

    team_a_actions = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            observation,
            action_mask,
            actor_keys,
            random_policy,
            TEAM_A_ID,
        ),
    )
    team_b_actions = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            observation,
            action_mask,
            actor_keys,
            random_policy,
            TEAM_B_ID,
        ),
    )
    submitted_action = build_joint_action_from_actor_actions(
        team_a_actions, team_b_actions
    )

    global_slots = jnp.arange(MAX_AGENT_SLOTS, dtype=jnp.int32)
    assert bool(jnp.all(action_mask.move_mask[global_slots, submitted_action.move]))
    assert bool(
        jnp.all(
            action_mask.select_target_use_ultimate_joint_mask[
                global_slots,
                submitted_action.select_target,
                submitted_action.use_ultimate,
            ]
        )
    )
    inactive_slots = jnp.asarray((1, 2, 3, 4, 6, 7, 8, 9), dtype=jnp.int32)
    assert bool(jnp.all(submitted_action.move[inactive_slots] == MOVE_STAY))
    assert bool(jnp.all(submitted_action.select_target[inactive_slots] == 0))
    assert bool(jnp.all(submitted_action.use_ultimate[inactive_slots] == 0))

    _, _, _, _, _, info = step(
        config,
        state,
        action_mask,
        submitted_action,
        jax.random.key(53),
    )
    acceptance = info.transition_facts.action_acceptance_facts
    _assert_tree_arrays_exact(acceptance.submitted_joint_action, submitted_action)
    _assert_tree_arrays_exact(acceptance.accepted_joint_action, submitted_action)
    assert not bool(
        jnp.any(acceptance.submitted_action_tuple_is_out_of_domain_by_actor)
    )
    assert not bool(jnp.any(acceptance.in_domain_move_action_is_rejected_by_actor))
    assert not bool(
        jnp.any(acceptance.in_domain_combat_action_pair_is_rejected_by_actor)
    )
