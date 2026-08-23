"""Fixed-slot and information-boundary proof for policy action assembly."""

import ast
import inspect
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

import marl_battlegrounds.policies.actor as actor_module
from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import (
    CONTEXT_FEATURES,
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBJECTIVE_SLOTS,
    MAX_OBSTACLE_SLOTS,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_TEAMS,
    NUM_ULTIMATE_ACTIONS,
    OBJECTIVE_FEATURES,
    OBSTACLE_FEATURES,
    SELF_FEATURES,
    TEAM_A_ID,
    TEAM_B_ID,
    UNIT_FEATURES,
    Action,
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


def _recipient_coded_leaf(leaf: Array) -> Array:
    """Broadcast a distinct global-slot code across every recipient row."""
    recipient_codes = jnp.arange(MAX_AGENT_SLOTS, dtype=jnp.int32)
    if jnp.issubdtype(leaf.dtype, jnp.bool_):
        typed_codes = (recipient_codes % 2).astype(jnp.bool_)
    else:
        typed_codes = recipient_codes.astype(leaf.dtype)
    broadcast_shape = (MAX_AGENT_SLOTS,) + (1,) * (leaf.ndim - 1)
    return jnp.broadcast_to(typed_codes.reshape(broadcast_shape), leaf.shape)


def _recipient_coded_policy_inputs() -> tuple[Observation, ActionMask]:
    """Return complete policy inputs whose leaves identify their recipient row."""
    config = evaluation_env_config(team_sizes=(2, 3))
    _, observation, action_mask, _ = reset(config, jax.random.key(0))
    return (
        cast(Observation, jax.tree.map(_recipient_coded_leaf, observation)),
        cast(ActionMask, jax.tree.map(_recipient_coded_leaf, action_mask)),
    )


def _shape_checking_slot_policy(
    observation: Observation,
    action_mask: ActionMask,
    key: Array,
) -> ActorAction:
    """Assert the complete rank-reduced contract and encode row, mask, and key."""
    assert observation.self_features.shape == (SELF_FEATURES,)
    assert observation.ally_unit_features.shape == (
        MAX_AGENTS_PER_TEAM,
        UNIT_FEATURES,
    )
    assert observation.enemy_unit_features.shape == (
        MAX_AGENTS_PER_TEAM,
        UNIT_FEATURES,
    )
    assert observation.map_obstacle_features.shape == (
        MAX_OBSTACLE_SLOTS,
        OBSTACLE_FEATURES,
    )
    assert observation.objective_features.shape == (
        MAX_OBJECTIVE_SLOTS,
        OBJECTIVE_FEATURES,
    )
    assert observation.context_features.shape == (CONTEXT_FEATURES,)
    assert observation.ally_visibility_mask.shape == (MAX_AGENTS_PER_TEAM,)
    assert observation.enemy_visibility_mask.shape == (MAX_AGENTS_PER_TEAM,)

    previous_actions = observation.previous_timestep_actions
    assert previous_actions.ally_previous_timestep_move_actions_one_hot.shape == (
        MAX_AGENTS_PER_TEAM,
        NUM_MOVE_ACTIONS,
    )
    assert previous_actions.enemy_previous_timestep_move_actions_one_hot.shape == (
        MAX_AGENTS_PER_TEAM,
        NUM_MOVE_ACTIONS,
    )
    assert (
        previous_actions.ally_previous_timestep_select_target_actions_one_hot.shape
        == (
            MAX_AGENTS_PER_TEAM,
            NUM_TARGET_ACTIONS,
        )
    )
    assert (
        previous_actions.enemy_previous_timestep_select_target_actions_one_hot.shape
        == (
            MAX_AGENTS_PER_TEAM,
            NUM_TARGET_ACTIONS,
        )
    )
    assert (
        previous_actions.ally_previous_timestep_use_ultimate_actions_one_hot.shape
        == (
            MAX_AGENTS_PER_TEAM,
            NUM_ULTIMATE_ACTIONS,
        )
    )
    assert (
        previous_actions.enemy_previous_timestep_use_ultimate_actions_one_hot.shape
        == (
            MAX_AGENTS_PER_TEAM,
            NUM_ULTIMATE_ACTIONS,
        )
    )

    spawn_lifecycle = observation.spawn_lifecycle
    assert spawn_lifecycle.spawn_pad_positions_by_agent_by_team.shape == (
        NUM_TEAMS,
        MAX_AGENTS_PER_TEAM,
        ENVIRONMENT_DIMENSIONS,
    )
    assert spawn_lifecycle.spawn_shield_actual_durations_by_agent_by_team.shape == (
        NUM_TEAMS,
        MAX_AGENTS_PER_TEAM,
    )
    assert spawn_lifecycle.spawn_shield_configured_duration_by_agent.shape == ()
    assert spawn_lifecycle.spawn_shield_speed_by_agent.shape == ()
    assert spawn_lifecycle.respawn_wave_period_step_count_by_agent_by_team.shape == (
        NUM_TEAMS,
    )
    assert spawn_lifecycle.respawn_wave_countdowns_by_agent_by_team.shape == (
        NUM_TEAMS,
    )
    assert spawn_lifecycle.active_mask_by_agent_by_team.shape == (
        NUM_TEAMS,
        MAX_AGENTS_PER_TEAM,
    )
    assert spawn_lifecycle.alive_mask_by_agent_by_team.shape == (
        NUM_TEAMS,
        MAX_AGENTS_PER_TEAM,
    )
    assert spawn_lifecycle.class_ids_by_agent_by_team.shape == (
        NUM_TEAMS,
        MAX_AGENTS_PER_TEAM,
    )
    assert spawn_lifecycle.class_ids_by_agent_by_team.dtype == jnp.int32

    assert action_mask.move_mask.shape == (NUM_MOVE_ACTIONS,)
    assert action_mask.select_target_mask.shape == (NUM_TARGET_ACTIONS,)
    assert action_mask.use_ultimate_mask.shape == (NUM_ULTIMATE_ACTIONS,)
    assert action_mask.select_target_use_ultimate_joint_mask.shape == (
        NUM_TARGET_ACTIONS,
        NUM_ULTIMATE_ACTIONS,
    )

    for leaf in jax.tree_util.tree_leaves(observation):
        if jnp.issubdtype(leaf.dtype, jnp.bool_):
            assert leaf.dtype == jnp.bool_
        elif jnp.issubdtype(leaf.dtype, jnp.integer):
            assert leaf.dtype == jnp.int32
        else:
            assert leaf.dtype == jnp.float32
    for leaf in jax.tree_util.tree_leaves(action_mask):
        assert leaf.dtype == jnp.bool_

    return ActorAction(
        move=spawn_lifecycle.class_ids_by_agent_by_team[0, 0],
        select_target=action_mask.move_mask[0].astype(jnp.int32),
        use_ultimate=jax.random.bits(key, (), dtype=jnp.uint32).astype(jnp.int32),
    )


def _perturb_nonfocal_rows(leaf: Array, focal_global_slot: int) -> Array:
    """Change every recipient row except the focal row without changing shape."""
    if jnp.issubdtype(leaf.dtype, jnp.bool_):
        perturbed = jnp.logical_not(leaf)
    else:
        perturbed = leaf + jnp.asarray(100, dtype=leaf.dtype)
    preserve_focal = (jnp.arange(MAX_AGENT_SLOTS) == focal_global_slot).reshape(
        (MAX_AGENT_SLOTS,) + (1,) * (leaf.ndim - 1)
    )
    return jnp.where(preserve_focal, leaf, perturbed)


def _actor_key_bits(actor_key: Array) -> Array:
    """Encode one actor key as a deterministic scalar test value."""
    return jax.random.bits(actor_key, (), dtype=jnp.uint32).astype(jnp.int32)


def test_actor_action_and_joint_assembly_preserve_exact_fixed_slot_values() -> None:
    """Scalar and batched actions retain shape, dtype, ordering, and raw values."""
    scalar_action = ActorAction(
        move=jnp.asarray(3, dtype=jnp.int32),
        select_target=jnp.asarray(7, dtype=jnp.int32),
        use_ultimate=jnp.asarray(1, dtype=jnp.int32),
    )
    for leaf in jax.tree_util.tree_leaves(scalar_action):
        assert leaf.shape == ()
        assert leaf.dtype == jnp.int32

    team_a_actions = ActorAction(
        move=jnp.asarray((-5, -4, -3, -2, -1), dtype=jnp.int32),
        select_target=jnp.asarray((21, 22, 23, 24, 25), dtype=jnp.int32),
        use_ultimate=jnp.asarray((31, 32, 33, 34, 35), dtype=jnp.int32),
    )
    team_b_actions = ActorAction(
        move=jnp.asarray((10, 11, 12, 13, 14), dtype=jnp.int32),
        select_target=jnp.asarray((40, 41, 42, 43, 44), dtype=jnp.int32),
        use_ultimate=jnp.asarray((50, 51, 52, 53, 54), dtype=jnp.int32),
    )
    expected = Action(
        move=jnp.asarray((-5, -4, -3, -2, -1, 10, 11, 12, 13, 14), dtype=jnp.int32),
        select_target=jnp.asarray(
            (21, 22, 23, 24, 25, 40, 41, 42, 43, 44), dtype=jnp.int32
        ),
        use_ultimate=jnp.asarray(
            (31, 32, 33, 34, 35, 50, 51, 52, 53, 54), dtype=jnp.int32
        ),
    )

    eager = build_joint_action_from_actor_actions(team_a_actions, team_b_actions)
    compiled = cast(
        Action,
        jax.jit(build_joint_action_from_actor_actions)(team_a_actions, team_b_actions),
    )
    _assert_tree_arrays_exact(eager, expected)
    _assert_tree_arrays_exact(compiled, expected)


@pytest.mark.parametrize(
    ("team_identity", "start_global_slot"),
    ((TEAM_A_ID, 0), (TEAM_B_ID, MAX_AGENTS_PER_TEAM)),
)
@pytest.mark.parametrize("use_legacy_keys", (False, True))
def test_no_shared_obs_executor_slices_complete_rows_and_keys_in_global_order(
    team_identity: int,
    start_global_slot: int,
    use_legacy_keys: bool,
) -> None:
    """Each team receives exactly five complete actor rows and matching keys."""
    observation, action_mask = _recipient_coded_policy_inputs()
    typed_keys = jax.random.split(jax.random.key(17), MAX_AGENT_SLOTS)
    actor_keys = jax.random.key_data(typed_keys) if use_legacy_keys else typed_keys

    with jax.disable_jit():
        eager = cast(
            ActorAction,
            execute_no_shared_obs_team_policy(
                observation,
                action_mask,
                actor_keys,
                _shape_checking_slot_policy,
                team_identity,
            ),
        )
    compiled = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            observation,
            action_mask,
            actor_keys,
            _shape_checking_slot_policy,
            team_identity,
        ),
    )

    team_slice = slice(start_global_slot, start_global_slot + MAX_AGENTS_PER_TEAM)
    expected = ActorAction(
        move=jnp.arange(
            start_global_slot,
            start_global_slot + MAX_AGENTS_PER_TEAM,
            dtype=jnp.int32,
        ),
        select_target=(
            jnp.arange(
                start_global_slot,
                start_global_slot + MAX_AGENTS_PER_TEAM,
                dtype=jnp.int32,
            )
            % 2
        ),
        use_ultimate=jax.vmap(_actor_key_bits)(actor_keys[team_slice]),
    )
    _assert_tree_arrays_exact(eager, expected)
    _assert_tree_arrays_exact(compiled, expected)
    for leaf in jax.tree_util.tree_leaves(compiled):
        assert leaf.shape == (MAX_AGENTS_PER_TEAM,)
        assert leaf.dtype == jnp.int32


def test_no_shared_obs_executor_is_focally_independent_of_other_rows_and_keys() -> None:
    """A focal result depends on only its own row, local mask, and actor key."""
    focal_global_slot = 2
    observation, action_mask = _recipient_coded_policy_inputs()
    actor_keys = jax.random.split(jax.random.key(23), MAX_AGENT_SLOTS)
    alternate_keys = jax.random.split(jax.random.key(29), MAX_AGENT_SLOTS)

    def perturb_leaf(leaf: Array) -> Array:
        return _perturb_nonfocal_rows(leaf, focal_global_slot)

    changed_observation = cast(Observation, jax.tree.map(perturb_leaf, observation))
    changed_action_mask = cast(ActionMask, jax.tree.map(perturb_leaf, action_mask))
    changed_actor_keys = alternate_keys.at[focal_global_slot].set(
        actor_keys[focal_global_slot]
    )

    baseline = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            observation,
            action_mask,
            actor_keys,
            _shape_checking_slot_policy,
            TEAM_A_ID,
        ),
    )
    changed = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            changed_observation,
            changed_action_mask,
            changed_actor_keys,
            _shape_checking_slot_policy,
            TEAM_A_ID,
        ),
    )

    for baseline_leaf, changed_leaf in zip(
        jax.tree_util.tree_leaves(baseline),
        jax.tree_util.tree_leaves(changed),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(baseline_leaf[focal_global_slot]),
            np.asarray(changed_leaf[focal_global_slot]),
        )
    assert not bool(jnp.array_equal(baseline.move, changed.move))


def test_actor_assembler_dependency_boundary_excludes_policy_input_authority() -> None:
    """The mode-neutral assembler imports no observation or regime concepts."""
    module_tree = ast.parse(inspect.getsource(actor_module))
    imported_modules: set[str] = set()
    imported_core_type_names: set[str] = set()
    for node in ast.walk(module_tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
            if node.module == "marl_battlegrounds.core.types":
                imported_core_type_names.update(alias.name for alias in node.names)

    assert imported_core_type_names == {"Action"}
    forbidden_roots = {
        "marl_battlegrounds.evaluation",
        "marl_battlegrounds.policies.no_shared_obs",
        "marl_battlegrounds.scenarios",
        "marl_battlegrounds.training",
    }
    assert not any(
        imported == forbidden or imported.startswith(f"{forbidden}.")
        for imported in imported_modules
        for forbidden in forbidden_roots
    )
