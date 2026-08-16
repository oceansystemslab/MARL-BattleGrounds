"""Exact contract and real-core parity tests for canonical simulator axes."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds.core.axis_mappings import (
    GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION,
    GLOBAL_RECIPIENT_SLOT_INDEX_BY_ACTOR_AND_TARGET_ACTION,
    GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW,
    GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW,
    MOVEMENT_ACTION_NAME_BY_ID,
    TARGET_ACTION_NAME_BY_ID,
    TEAM_A_END,
    TEAM_A_START,
    TEAM_B_END,
    TEAM_B_START,
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION,
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY,
    global_slot_to_target_action,
    observation_relation_and_row,
    target_action_to_global_slot,
)
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import initialize_scenario_state, reset, step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_NORTH,
    MOVE_SOUTHWEST,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    Action,
    EnvConfig,
)

_EXPECTED_MOVEMENT_NAMES = (
    "Stay",
    "North",
    "South",
    "East",
    "West",
    "Northeast",
    "Northwest",
    "Southeast",
    "Southwest",
)
_EXPECTED_TARGET_NAMES = (
    "Target None",
    "Ally 0",
    "Ally 1",
    "Ally 2",
    "Ally 3",
    "Ally 4",
    "Enemy 0",
    "Enemy 1",
    "Enemy 2",
    "Enemy 3",
    "Enemy 4",
)
_EXPECTED_TEAM_A_TARGET_ROW = (None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
_EXPECTED_TEAM_B_TARGET_ROW = (None, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4)


def _real_core_mapping_parity_config() -> EnvConfig:
    """Build an asymmetric public config with both team blocks and padding."""
    requested_classes = jnp.asarray(
        (
            MAGE_CLASS_ID,
            PRIEST_CLASS_ID,
            WARRIOR_CLASS_ID,
            HUNTER_CLASS_ID,
            ROGUE_CLASS_ID,
            HUNTER_CLASS_ID,
            ROGUE_CLASS_ID,
            MAGE_CLASS_ID,
            WARRIOR_CLASS_ID,
            PRIEST_CLASS_ID,
        ),
        dtype=jnp.int32,
    )
    profile = resolve_agent_profile(
        requested_classes,
        jnp.asarray((2, 1), dtype=jnp.int32),
    )
    y_coordinates = jnp.linspace(
        2.0,
        10.0,
        MAX_AGENTS_PER_TEAM,
        dtype=jnp.float32,
    )
    spawn_pads = jnp.stack(
        (
            jnp.stack(
                (jnp.full_like(y_coordinates, 2.0), y_coordinates),
                axis=-1,
            ),
            jnp.stack(
                (jnp.full_like(y_coordinates, 18.0), y_coordinates),
                axis=-1,
            ),
        ),
        axis=0,
    )
    return EnvConfig(
        task_mode=0,
        team_deathmatch_score_threshold=0,
        max_steps=100,
        map_width=20.0,
        map_height=12.0,
        obstacles=jnp.zeros(
            (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
            dtype=jnp.float32,
        ),
        agent_profile=profile,
        ordinary_movement_distance_scale=0.1,
        team_spawn_pad_positions=spawn_pads,
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray((5, 7), dtype=jnp.int32),
    )


def _assert_tree_arrays_exact(actual: object, expected: object) -> None:
    """Require identical PyTree structure, dtypes, shapes, and values."""
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


def test_fixed_team_boundaries_and_action_names_are_exact() -> None:
    assert (TEAM_A_START, TEAM_A_END) == (0, MAX_AGENTS_PER_TEAM)
    assert (TEAM_B_START, TEAM_B_END) == (
        MAX_AGENTS_PER_TEAM,
        MAX_AGENT_SLOTS,
    )
    assert MOVEMENT_ACTION_NAME_BY_ID == _EXPECTED_MOVEMENT_NAMES
    assert TARGET_ACTION_NAME_BY_ID == _EXPECTED_TARGET_NAMES
    assert len(MOVEMENT_ACTION_NAME_BY_ID) == NUM_MOVE_ACTIONS
    assert len(TARGET_ACTION_NAME_BY_ID) == NUM_TARGET_ACTIONS


def test_unit_direction_vectors_have_exact_values_shapes_and_dtypes() -> None:
    inverse_square_root_of_two = float(np.float32(1 / np.sqrt(2.0)))
    expected_host_vectors = (
        (0.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (inverse_square_root_of_two, inverse_square_root_of_two),
        (-inverse_square_root_of_two, inverse_square_root_of_two),
        (inverse_square_root_of_two, -inverse_square_root_of_two),
        (-inverse_square_root_of_two, -inverse_square_root_of_two),
    )

    assert expected_host_vectors == UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION
    assert isinstance(UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION, tuple)
    assert all(
        isinstance(row, tuple) for row in UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION
    )
    assert UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY.shape == (
        NUM_MOVE_ACTIONS,
        2,
    )
    assert UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY),
        np.asarray(expected_host_vectors, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.linalg.norm(
            np.asarray(UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY)[1:],
            axis=-1,
        ),
        np.ones((NUM_MOVE_ACTIONS - 1,), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_target_and_observation_mappings_have_exact_v1_rows() -> None:
    expected_target_rows = (
        *(_EXPECTED_TEAM_A_TARGET_ROW for _ in range(MAX_AGENTS_PER_TEAM)),
        *(_EXPECTED_TEAM_B_TARGET_ROW for _ in range(MAX_AGENTS_PER_TEAM)),
    )
    expected_ally_rows = (
        *((0, 1, 2, 3, 4) for _ in range(MAX_AGENTS_PER_TEAM)),
        *((5, 6, 7, 8, 9) for _ in range(MAX_AGENTS_PER_TEAM)),
    )
    expected_enemy_rows = (
        *((5, 6, 7, 8, 9) for _ in range(MAX_AGENTS_PER_TEAM)),
        *((0, 1, 2, 3, 4) for _ in range(MAX_AGENTS_PER_TEAM)),
    )

    assert expected_target_rows == GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION
    assert expected_ally_rows == GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW
    assert expected_enemy_rows == GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW
    assert isinstance(GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION, tuple)
    assert all(
        isinstance(row, tuple)
        for row in GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION
    )
    assert GLOBAL_RECIPIENT_SLOT_INDEX_BY_ACTOR_AND_TARGET_ACTION.shape == (
        MAX_AGENT_SLOTS,
        NUM_TARGET_ACTIONS,
    )
    assert GLOBAL_RECIPIENT_SLOT_INDEX_BY_ACTOR_AND_TARGET_ACTION.dtype == jnp.int32
    np.testing.assert_array_equal(
        np.asarray(GLOBAL_RECIPIENT_SLOT_INDEX_BY_ACTOR_AND_TARGET_ACTION),
        np.asarray(
            tuple(
                tuple(-1 if value is None else value for value in row)
                for row in expected_target_rows
            ),
            dtype=np.int32,
        ),
    )


@pytest.mark.parametrize("actor_global_slot", range(MAX_AGENT_SLOTS))
def test_target_rows_align_with_ally_then_enemy_observation_rows(
    actor_global_slot: int,
) -> None:
    target_row = GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION[actor_global_slot]
    ally_row = GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW[actor_global_slot]
    enemy_row = GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW[actor_global_slot]

    assert target_row[0] is None
    assert target_row[1 : 1 + MAX_AGENTS_PER_TEAM] == ally_row
    assert target_row[1 + MAX_AGENTS_PER_TEAM :] == enemy_row
    assert set(ally_row).isdisjoint(enemy_row)
    assert set(ally_row + enemy_row) == set(range(MAX_AGENT_SLOTS))
    assert actor_global_slot in ally_row


@pytest.mark.parametrize("actor_global_slot", range(MAX_AGENT_SLOTS))
def test_target_mapping_round_trips_every_global_slot(
    actor_global_slot: int,
) -> None:
    assert global_slot_to_target_action(actor_global_slot, None) == 0
    assert target_action_to_global_slot(actor_global_slot, 0) is None

    for target_global_slot in range(MAX_AGENT_SLOTS):
        target_action = global_slot_to_target_action(
            actor_global_slot,
            target_global_slot,
        )
        assert 1 <= target_action < NUM_TARGET_ACTIONS
        assert (
            target_action_to_global_slot(actor_global_slot, target_action)
            == target_global_slot
        )


@pytest.mark.parametrize("observer_global_slot", range(MAX_AGENT_SLOTS))
def test_every_candidate_has_one_exact_relation_local_row(
    observer_global_slot: int,
) -> None:
    ally_slots = GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW[observer_global_slot]
    enemy_slots = GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW[observer_global_slot]

    for candidate_global_slot in range(MAX_AGENT_SLOTS):
        relation, row = observation_relation_and_row(
            observer_global_slot,
            candidate_global_slot,
        )
        mapped_slots = ally_slots if relation == "ally" else enemy_slots
        assert mapped_slots[row] == candidate_global_slot


@pytest.mark.parametrize(
    ("function_name", "arguments"),
    (
        pytest.param("forward", (-1, None), id="forward-actor-below-domain"),
        pytest.param(
            "forward",
            (MAX_AGENT_SLOTS, None),
            id="forward-actor-above-domain",
        ),
        pytest.param("forward", (0, -1), id="target-below-domain"),
        pytest.param(
            "forward",
            (0, MAX_AGENT_SLOTS),
            id="target-above-domain",
        ),
        pytest.param("inverse", (-1, 0), id="inverse-actor-below-domain"),
        pytest.param(
            "inverse",
            (MAX_AGENT_SLOTS, 0),
            id="inverse-actor-above-domain",
        ),
        pytest.param("inverse", (0, -1), id="action-below-domain"),
        pytest.param(
            "inverse",
            (0, NUM_TARGET_ACTIONS),
            id="action-above-domain",
        ),
        pytest.param("relation", (-1, 0), id="relation-observer-below-domain"),
        pytest.param(
            "relation",
            (MAX_AGENT_SLOTS, 0),
            id="relation-observer-above-domain",
        ),
        pytest.param("relation", (0, -1), id="relation-candidate-below-domain"),
        pytest.param(
            "relation",
            (0, MAX_AGENT_SLOTS),
            id="relation-candidate-above-domain",
        ),
    ),
)
def test_mapping_helpers_reject_values_outside_fixed_domains(
    function_name: str,
    arguments: tuple[int, int | None],
) -> None:
    if function_name == "forward":
        with pytest.raises(ValueError):
            global_slot_to_target_action(*arguments)
    elif function_name == "inverse":
        target_action = arguments[1]
        assert target_action is not None
        with pytest.raises(ValueError):
            target_action_to_global_slot(arguments[0], target_action)
    else:
        candidate_global_slot = arguments[1]
        assert candidate_global_slot is not None
        with pytest.raises(ValueError):
            observation_relation_and_row(arguments[0], candidate_global_slot)


@pytest.mark.parametrize("invalid_value", (True, False, 1.0, "1"))
def test_mapping_helpers_reject_non_integer_slot_and_action_types(
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError):
        global_slot_to_target_action(invalid_value, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        global_slot_to_target_action(0, invalid_value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        target_action_to_global_slot(0, invalid_value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        observation_relation_and_row(invalid_value, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        observation_relation_and_row(0, invalid_value)  # type: ignore[arg-type]


def test_real_core_mapping_consumers_preserve_eager_jit_and_trace_contracts() -> None:
    """Exercise canonical mappings through one real public transition.

    This is the behavior-preserving extraction proof: the fixed tables must
    still drive movement, actor-relative combat routing, observer relation
    rows, accepted-action history, and lifecycle projection under both eager
    and traced execution. The trace must close over the exact canonical JAX
    payloads without introducing a host callback.
    """
    config = _real_core_mapping_parity_config()
    key = jax.random.PRNGKey(17)
    reset_state, _, _, _ = reset(config, key)
    scenario_positions = jnp.zeros(
        (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        dtype=jnp.float32,
    )
    scenario_positions = scenario_positions.at[0].set(jnp.asarray((8.0, 5.0)))
    scenario_positions = scenario_positions.at[1].set(jnp.asarray((4.0, 8.0)))
    scenario_positions = scenario_positions.at[5].set(jnp.asarray((10.0, 5.0)))
    scenario_state = reset_state._replace(agent_positions=scenario_positions)
    start_state, _, start_mask, _ = initialize_scenario_state(
        scenario_state,
        config,
    )

    submitted_action = Action(
        move=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
        .at[0]
        .set(MOVE_NORTH)
        .at[5]
        .set(MOVE_SOUTHWEST),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
        .at[0]
        .set(global_slot_to_target_action(0, 5))
        .at[5]
        .set(global_slot_to_target_action(5, 0)),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )
    assert bool(start_mask.select_target_use_ultimate_joint_mask[0, 6, 0])
    assert bool(start_mask.select_target_use_ultimate_joint_mask[5, 6, 0])

    eager_result = step(
        config,
        start_state,
        start_mask,
        submitted_action,
        key,
    )
    jitted_result = cast(
        object,
        jax.jit(step)(
            config,
            start_state,
            start_mask,
            submitted_action,
            key,
        ),
    )
    _assert_tree_arrays_exact(jitted_result, eager_result)

    next_state, next_observation, _, _, _, info = eager_result
    facts = info.transition_facts
    accepted = facts.action_acceptance_facts.accepted_joint_action
    np.testing.assert_array_equal(
        np.asarray(accepted.move),
        np.asarray(submitted_action.move),
    )
    np.testing.assert_array_equal(
        np.asarray(accepted.select_target),
        np.asarray(submitted_action.select_target),
    )

    expected_has_recipient = np.zeros((MAX_AGENT_SLOTS,), dtype=np.bool_)
    expected_has_recipient[[0, 5]] = True
    expected_recipient_slots = np.full((MAX_AGENT_SLOTS,), -1, dtype=np.int32)
    expected_recipient_slots[0] = 5
    expected_recipient_slots[5] = 0
    combat_facts = facts.combat_transition_facts
    np.testing.assert_array_equal(
        np.asarray(combat_facts.combat_effect_has_recipient_by_source),
        expected_has_recipient,
    )
    np.testing.assert_array_equal(
        np.asarray(combat_facts.combat_effect_recipient_global_slot_by_source),
        expected_recipient_slots,
    )

    expected_displacement = np.zeros(
        (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        dtype=np.float32,
    )
    for actor_slot in (0, 5):
        move_action = int(np.asarray(accepted.move)[actor_slot])
        expected_displacement[actor_slot] = (
            np.asarray(UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY)[move_action]
            * float(np.asarray(config.agent_profile.base_movement_speeds)[actor_slot])
            * config.ordinary_movement_distance_scale
        )
    np.testing.assert_allclose(
        np.asarray(facts.physical_facts.ordinary_movement_phase_displacement_by_agent),
        expected_displacement,
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(next_state.agent_positions - start_state.agent_positions),
        np.asarray(facts.physical_facts.ordinary_movement_phase_displacement_by_agent),
    )

    for observer_slot, candidate_slot in ((0, 5), (5, 0), (0, 1)):
        relation, relation_row = observation_relation_and_row(
            observer_slot,
            candidate_slot,
        )
        if relation == "ally":
            unit_features = next_observation.ally_unit_features
            visibility = next_observation.ally_visibility_mask
        else:
            unit_features = next_observation.enemy_unit_features
            visibility = next_observation.enemy_visibility_mask
        assert bool(visibility[observer_slot, relation_row])
        np.testing.assert_allclose(
            np.asarray(
                unit_features[
                    observer_slot,
                    relation_row,
                    (AGENT_FEATURE_X, AGENT_FEATURE_Y),
                ]
            ),
            np.asarray(next_state.agent_positions[candidate_slot]),
            rtol=0.0,
            atol=0.0,
        )

    previous_actions = next_observation.previous_timestep_actions
    for observer_slot, actor_slot in ((0, 0), (0, 5), (5, 0), (5, 5)):
        relation, relation_row = observation_relation_and_row(
            observer_slot,
            actor_slot,
        )
        if relation == "ally":
            move_rows = previous_actions.ally_previous_timestep_move_actions_one_hot
            target_rows = (
                previous_actions.ally_previous_timestep_select_target_actions_one_hot
            )
        else:
            move_rows = previous_actions.enemy_previous_timestep_move_actions_one_hot
            target_rows = (
                previous_actions.enemy_previous_timestep_select_target_actions_one_hot
            )

        accepted_move = int(np.asarray(accepted.move)[actor_slot])
        accepted_actor_target = int(np.asarray(accepted.select_target)[actor_slot])
        global_target = target_action_to_global_slot(
            actor_slot,
            accepted_actor_target,
        )
        observer_relative_target = global_slot_to_target_action(
            observer_slot,
            global_target,
        )
        assert int(np.argmax(np.asarray(move_rows[observer_slot, relation_row]))) == (
            accepted_move
        )
        assert float(np.sum(np.asarray(move_rows[observer_slot, relation_row]))) == 1.0
        assert (
            int(np.argmax(np.asarray(target_rows[observer_slot, relation_row])))
            == observer_relative_target
        )
        assert (
            float(np.sum(np.asarray(target_rows[observer_slot, relation_row]))) == 1.0
        )

    active_mask = np.asarray(config.agent_profile.active_mask)
    lifecycle_active = np.asarray(
        next_observation.spawn_lifecycle.active_mask_by_agent_by_team
    )
    np.testing.assert_array_equal(
        lifecycle_active[0],
        np.stack((active_mask[TEAM_A_START:TEAM_A_END], active_mask[TEAM_B_START:])),
    )
    np.testing.assert_array_equal(
        lifecycle_active[5],
        np.stack((active_mask[TEAM_B_START:], active_mask[TEAM_A_START:TEAM_A_END])),
    )
    np.testing.assert_array_equal(
        lifecycle_active[2],
        np.zeros((2, MAX_AGENTS_PER_TEAM), dtype=np.bool_),
    )
    for inactive_observer in (2, 3, 4, 6, 7, 8, 9):
        np.testing.assert_array_equal(
            np.asarray(
                previous_actions.ally_previous_timestep_move_actions_one_hot[
                    inactive_observer
                ]
            ),
            np.zeros((MAX_AGENTS_PER_TEAM, NUM_MOVE_ACTIONS), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(
                previous_actions.enemy_previous_timestep_move_actions_one_hot[
                    inactive_observer
                ]
            ),
            np.zeros((MAX_AGENTS_PER_TEAM, NUM_MOVE_ACTIONS), dtype=np.float32),
        )

    traced_step = jax.make_jaxpr(step)(
        config,
        start_state,
        start_mask,
        submitted_action,
        key,
    )
    traced_array_constants = tuple(
        np.asarray(constant)
        for constant in traced_step.consts
        if hasattr(constant, "shape")
    )
    assert any(
        constant.dtype == np.float32
        and np.array_equal(
            constant,
            np.asarray(UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY),
        )
        for constant in traced_array_constants
    )
    assert any(
        constant.dtype == np.int32
        and np.array_equal(
            constant,
            np.asarray(GLOBAL_RECIPIENT_SLOT_INDEX_BY_ACTOR_AND_TARGET_ACTION),
        )
        for constant in traced_array_constants
    )
    assert "callback" not in str(traced_step).lower()
