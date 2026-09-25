"""Check movement and target mappings, actor-to-slot rows and the spawn side rule.

The tests compare the mapping tables with real Core transitions so a table can
be checked against the behavior that consumes it.

The spawn side rule (spawn_bank_on_right) says a bank is on the right when its
five float32 pad x values add up to more than five float32 half-widths. The
tests check Core eagerly, under jit and under jit(vmap), and the host copy
evaluation.models.red_zone_team_on_right, against an exact Fraction reference
written here. The cases are the 52 installed maps in both bank orders, custom
and swapped banks, exactly centred banks at widths 17.3, 20.3 and 1.0 (Left),
cancellation banks, changes to unused rows that keep the exact sum, widths up
to the float32 maximum with every internal component finite, near ties within
three float32 steps and seeded random banks, each in all 120 pad orders. On
200,000 random banks at ordinary widths (1 to 1,000), Core eager, jit and
jit(vmap) match exact float64 sums on every bank, and the host copy matches
them on every 100th bank (2,000 banks). The centred, custom and extreme-width
banks also run through the compiled scoring classifier red_zone_death_mask
under jit, so each new death at the right edge or the left edge counts as a
Red Zone death exactly when that edge is its own team's side. They also check
the leading and trailing shape contract, the host strip bounds
red_zone_x_range, and that the policy helper team_on_right uses the Core rule
while the frozen pre-Red-Zone formula still reads the 17.3 and 20.3 centred
banks as right.
"""
# pyright: reportPrivateUsage=false

import itertools
from collections.abc import Callable, Sequence
from fractions import Fraction
from functools import cache
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

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
    _spawn_side_expansion,
    global_slot_to_target_action,
    observation_relation_and_row,
    spawn_bank_on_right,
    target_action_to_global_slot,
)
from marl_battlegrounds.core.config import resolve_agent_profile, validate_env_config
from marl_battlegrounds.core.env import (
    initialize_scenario_state,
    red_zone_death_mask,
    reset,
    step,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    CONTEXT_FEATURE_MAP_WIDTH,
    CONTEXT_FEATURES,
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
from marl_battlegrounds.evaluation import models
from marl_battlegrounds.policies.input import (
    ActorInput,
    _historical_team_on_right,
    build_actor_input,
    team_on_right,
)
from marl_battlegrounds.tasks import (
    canonical_tournament_rosters,
    list_tdm_maps,
    make_standard_team_deathmatch_config,
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
        team_deathmatch_red_zone_depth=0.0,
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


_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_SMALLEST_SUBNORMAL = float(np.finfo(np.float32).smallest_subnormal)
_FLOAT32_LARGEST_SUBNORMAL = float(
    np.nextafter(np.float32(np.finfo(np.float32).tiny), np.float32(0))
)
# Every order of the five pad rows, as (120, 5) row indices.
_PAD_ORDERS = np.asarray(
    tuple(itertools.permutations(range(MAX_AGENTS_PER_TEAM))), dtype=np.int32
)
# JAX adds a traceback note after the message under jit, so only the start is
# anchored.
_SIDE_SHAPE_ERROR = r"^team_spawn_pad_positions must end with shape \(5, 2\)\."

type _Side = Callable[[Array, Array], Array]
type _Expansion = Callable[[Array, Array], list[Array]]
type _ActorSide = Callable[[ActorInput], Array]
_compiled_side = cast(_Side, jax.jit(spawn_bank_on_right))
_compiled_side_by_bank = cast(_Side, jax.jit(jax.vmap(spawn_bank_on_right)))
_compiled_expansion = cast(_Expansion, jax.jit(_spawn_side_expansion))
_compiled_death_mask = cast(
    Callable[[EnvConfig, Array, Array], Array], jax.jit(red_zone_death_mask)
)


def _f32(value: float) -> float:
    return float(np.float32(value))


def _float32_steps(value: float, steps: int) -> float:
    moved = np.float32(value)
    direction = np.float32(np.inf if steps > 0 else -np.inf)
    for _ in range(abs(steps)):
        moved = np.nextafter(moved, direction)
    return float(moved)


def _exact_on_right(map_width: float, pad_x: Sequence[float]) -> bool:
    # Independent reference: compare the exact rational sum of the five float32
    # pad x values with five float32 half-widths.
    half_width = Fraction(_f32(map_width)) / 2
    total = sum((Fraction(_f32(x)) for x in pad_x), Fraction(0))
    return total > MAX_AGENTS_PER_TEAM * half_width


def _exact_sides(banks: np.ndarray, widths: np.ndarray) -> np.ndarray:
    return np.asarray(
        tuple(
            _exact_on_right(float(width), tuple(float(x) for x in bank[:, 0]))
            for bank, width in zip(banks, widths, strict=True)
        ),
        dtype=np.bool_,
    )


def _bank(pad_x: Sequence[float]) -> np.ndarray:
    x = np.asarray(pad_x, dtype=np.float32)
    y = np.arange(1, MAX_AGENTS_PER_TEAM + 1, dtype=np.float32)
    return np.stack((x, y), axis=-1)


def _centred_spread(map_width: float) -> tuple[float, ...]:
    half_width = np.float32(map_width) * np.float32(0.5)
    return tuple(float(half_width + np.float32(offset)) for offset in (-2, -1, 0, 1, 2))


def _assert_side_rule_matches_exact_reference(
    banks: np.ndarray, widths: np.ndarray
) -> np.ndarray:
    # banks: float32 (N, 5, 2); widths: float32 (N,). Core eager, jit and
    # jit(vmap) over all 120 pad orders, and the host copy in stored and
    # reversed order, must all equal the exact reference. Returns it, (N,).
    expected = _exact_sides(banks, widths)
    banks_array = jnp.asarray(banks, dtype=jnp.float32)
    widths_array = jnp.asarray(widths, dtype=jnp.float32)
    eager = spawn_bank_on_right(banks_array, widths_array)
    assert eager.shape == expected.shape and eager.dtype == jnp.bool_
    np.testing.assert_array_equal(np.asarray(eager), expected)
    np.testing.assert_array_equal(
        np.asarray(_compiled_side(banks_array, widths_array)), expected
    )
    every_order = banks[:, _PAD_ORDERS].reshape(-1, MAX_AGENTS_PER_TEAM, 2)
    order_widths = np.repeat(widths, len(_PAD_ORDERS))
    np.testing.assert_array_equal(
        np.asarray(
            _compiled_side_by_bank(jnp.asarray(every_order), jnp.asarray(order_widths))
        ).reshape(len(banks), len(_PAD_ORDERS)),
        np.repeat(expected[:, None], len(_PAD_ORDERS), axis=1),
    )
    for order in (slice(None), slice(None, None, -1)):
        host = tuple(
            models.red_zone_team_on_right(
                float(width), tuple(float(x) for x in bank[order, 0])
            )
            for bank, width in zip(banks, widths, strict=True)
        )
        np.testing.assert_array_equal(np.asarray(host, dtype=np.bool_), expected)
    return expected


def _assert_expansion_finite(banks: np.ndarray, widths: np.ndarray) -> None:
    # Every component of the exact sum must stay finite in every pad order;
    # an overflow anywhere in the chain would leave inf or NaN behind.
    every_order = banks[:, _PAD_ORDERS].reshape(-1, MAX_AGENTS_PER_TEAM, 2)
    order_widths = jnp.asarray(np.repeat(widths, len(_PAD_ORDERS)))
    for components in (
        _spawn_side_expansion(jnp.asarray(banks), jnp.asarray(widths)),
        _compiled_expansion(jnp.asarray(every_order), order_widths),
    ):
        assert len(components) == 2 * MAX_AGENTS_PER_TEAM
        for component in components:
            assert bool(jnp.all(jnp.isfinite(component)))


@cache
def _installed_map_banks() -> tuple[np.ndarray, np.ndarray]:
    team_a, team_b = canonical_tournament_rosters()
    configs = tuple(
        make_standard_team_deathmatch_config(
            map_id=info.map_id, team_a_roster=team_a, team_b_roster=team_b
        )
        for info in list_tdm_maps()
    )
    banks = np.stack(
        tuple(np.asarray(config.team_spawn_pad_positions) for config in configs)
    )
    widths = np.asarray(tuple(config.map_width for config in configs), np.float32)
    banks.flags.writeable = False
    widths.flags.writeable = False
    return banks, widths


def test_spawn_side_rule_is_exact_on_every_installed_map_in_both_bank_orders() -> None:
    banks, widths = _installed_map_banks()
    assert banks.shape == (52, 2, MAX_AGENTS_PER_TEAM, 2)
    for ordered_banks, first_team_on_right in ((banks, False), (banks[:, ::-1], True)):
        expected = _assert_side_rule_matches_exact_reference(
            ordered_banks.reshape(-1, MAX_AGENTS_PER_TEAM, 2), np.repeat(widths, 2)
        ).reshape(52, 2)
        # Every installed map has Team A's bank on the left and Team B's on the right.
        np.testing.assert_array_equal(expected[:, 0], first_team_on_right)
        np.testing.assert_array_equal(expected[:, 1], not first_team_on_right)
        np.testing.assert_array_equal(
            np.asarray(
                spawn_bank_on_right(
                    jnp.asarray(ordered_banks), jnp.asarray(widths)[:, None]
                )
            ),
            expected,
        )


_WIDTH_3E38 = _f32(3e38)
_HALF_FLOAT32_MAX = _FLOAT32_MAX / 2


@pytest.mark.parametrize(
    ("map_width", "pad_x", "on_right"),
    (
        (17.3, _centred_spread(17.3), False),
        (20.3, _centred_spread(20.3), False),
        (1.0, _centred_spread(1.0), False),
        (20.0, (0.5, 19.5, 0.5, 19.5, 10.0), False),
        (20.0, (1.0, 19.0, 1.0, 19.0, 10.0 + 2.0**-20), True),
        (20.0, (2.0, 2.0, 2.0, 2.0, 2.0), False),
        (20.0, (18.0, 18.0, 18.0, 18.0, 18.0), True),
        (12.0, (2.0, 2.0, 2.0, 2.0, 2.0), False),
        (12.0, (10.0, 10.0, 10.0, 10.0, 10.0), True),
        (30.0, (3.0, 4.0, 5.0, 6.0, 7.0), False),
        (20.0, (1.0, 19.0, 1.0, 19.0, 11.0), True),
        (_WIDTH_3E38, (_f32(2.9e38),) * 5, True),
        (_FLOAT32_MAX, (_FLOAT32_MAX,) * 5, True),
        (_FLOAT32_MAX, (0.5,) * 5, False),
        (_FLOAT32_MAX, (_HALF_FLOAT32_MAX,) * 5, False),
        (
            _FLOAT32_MAX,
            (_float32_steps(_HALF_FLOAT32_MAX, 1),) + (_HALF_FLOAT32_MAX,) * 4,
            True,
        ),
    ),
    ids=(
        "centred-width-17.3",
        "centred-width-20.3",
        "centred-width-1",
        "cancellation-centred",
        "cancellation-one-step-right",
        "custom-left",
        "custom-right",
        "small-map-left-bank",
        "small-map-right-bank",
        "custom-both-teams-left",
        "custom-uneven-right",
        "width-3e38-pads-2.9e38",
        "float32-max-pads-at-max",
        "float32-max-pads-at-half-unit",
        "float32-max-centred",
        "float32-max-centred-plus-one-step",
    ),
)
def test_spawn_side_rule_is_exact_on_centred_custom_and_extreme_banks(
    map_width: float, pad_x: tuple[float, ...], on_right: bool
) -> None:
    banks = _bank(pad_x)[None]
    widths = np.asarray((map_width,), dtype=np.float32)
    expected = _assert_side_rule_matches_exact_reference(banks, widths)
    assert bool(expected[0]) is on_right
    _assert_expansion_finite(banks, widths)
    every_order = banks[0][_PAD_ORDERS]
    host = {
        models.red_zone_team_on_right(map_width, tuple(float(x) for x in bank[:, 0]))
        for bank in every_order
    }
    assert host == {on_right}

    # Both teams use this bank (Team B in reverse row order). Slots 0 and 5
    # die at the right edge x = width, slots 1 and 6 at the left edge x = 0,
    # and slot 2 stands at the right edge alive. The strips are a quarter of
    # the width deep, so each edge is inside only its own side's strip.
    width32 = float(np.float32(map_width))
    config = _real_core_mapping_parity_config()._replace(
        map_width=map_width,
        team_deathmatch_red_zone_depth=width32 / 4,
        team_spawn_pad_positions=jnp.asarray(
            np.stack((banks[0], banks[0][::-1])), dtype=jnp.float32
        ),
    )
    positions = np.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), np.float32)
    positions[[0, 2, 5], 0] = width32
    newly_dead = np.isin(np.arange(MAX_AGENT_SLOTS), (0, 1, 5, 6))
    np.testing.assert_array_equal(
        np.asarray(
            _compiled_death_mask(
                config, jnp.asarray(positions), jnp.asarray(newly_dead)
            )
        ),
        np.isin(np.arange(MAX_AGENT_SLOTS), (0, 5) if on_right else (1, 6)),
    )


def test_exactly_centred_banks_are_centred_in_exact_arithmetic() -> None:
    for map_width in (17.3, 20.3, 1.0):
        half_width = Fraction(_f32(map_width)) / 2
        pads = tuple(Fraction(_f32(x)) for x in _centred_spread(map_width))
        assert sum(pads, Fraction(0)) == MAX_AGENTS_PER_TEAM * half_width


def _ordered_float32_offset_sum(map_width: float, pad_x: Sequence[float]) -> float:
    # The rejected Revision 2 rule: add (x - h) in stored order in float32.
    half_width = np.float32(map_width) * np.float32(0.5)
    total = np.float32(0.0)
    for x in pad_x:
        total = np.float32(total + np.float32(np.float32(x) - half_width))
    return float(total)


def test_spawn_side_rule_ignores_pad_order_and_unused_rows_that_keep_the_sum() -> None:
    # Two used rows and three unused rows at width 20. Both variants add up to
    # exactly 50 (five half-widths), so the bank is centred and reads Left.
    used = (8.502655982971191, 12.912826538085938)
    variants = (
        (*used, 14.935379028320312, 6.7151198387146, 6.934018611907959),
        (*used, 1.79768705368042, 19.184789657592773, 7.602040767669678),
    )
    for pad_x in variants:
        assert all(_f32(x) == x for x in pad_x)
        assert sum((Fraction(x) for x in pad_x), Fraction(0)) == 50
    # The stored-order float32 offset sum gives opposite sides for them.
    assert [_ordered_float32_offset_sum(20.0, pad_x) > 0 for pad_x in variants] == [
        False,
        True,
    ]
    expected = _assert_side_rule_matches_exact_reference(
        np.stack(tuple(_bank(pad_x) for pad_x in variants)),
        np.full((len(variants),), 20.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(expected, False)

    # Both variants are valid configs whose active agents reset to the same
    # places; Core reads one side for both.
    base = _real_core_mapping_parity_config()
    positions: list[np.ndarray] = []
    for pad_x in variants:
        config = base._replace(
            team_spawn_pad_positions=base.team_spawn_pad_positions.at[0, :, 0].set(
                jnp.asarray(pad_x, dtype=jnp.float32)
            )
        )
        validate_env_config(config)
        state, _, _, _ = reset(config, jax.random.PRNGKey(3))
        active = np.asarray(config.agent_profile.active_mask)
        positions.append(np.asarray(state.agent_positions)[active])
        np.testing.assert_array_equal(
            np.asarray(
                spawn_bank_on_right(config.team_spawn_pad_positions, config.map_width)
            ),
            (False, True),
        )
    np.testing.assert_array_equal(positions[0], positions[1])


def test_spawn_side_rule_decides_near_ties_within_three_float32_steps() -> None:
    bases: list[tuple[float, tuple[float, ...]]] = [
        (map_width, (_f32(map_width) / 2,) * MAX_AGENTS_PER_TEAM)
        for map_width in (17.3, 20.3, 20.0, 12.1, 1.0, _WIDTH_3E38, _FLOAT32_MAX)
    ]
    bases += [
        (map_width, _centred_spread(map_width))
        for map_width in (17.3, 20.3, 20.0, 12.1, 1.0)
    ]
    near_ties: list[tuple[float, tuple[float, ...]]] = []
    for map_width, pad_x in bases:
        near_ties.append((map_width, pad_x))
        for pad in range(MAX_AGENTS_PER_TEAM):
            for steps in (-3, -2, -1, 1, 2, 3):
                moved = list(pad_x)
                moved[pad] = _float32_steps(moved[pad], steps)
                near_ties.append((map_width, tuple(moved)))
        for up, down in itertools.permutations(range(MAX_AGENTS_PER_TEAM), 2):
            for up_steps, down_steps in itertools.product((1, 2, 3), repeat=2):
                moved = list(pad_x)
                moved[up] = _float32_steps(moved[up], up_steps)
                moved[down] = _float32_steps(moved[down], -down_steps)
                near_ties.append((map_width, tuple(moved)))
    banks = np.stack(tuple(_bank(pad_x) for _, pad_x in near_ties))
    widths = np.asarray(tuple(map_width for map_width, _ in near_ties), np.float32)
    expected = _assert_side_rule_matches_exact_reference(banks, widths)
    assert bool(np.any(expected)) and not bool(np.all(expected))
    _assert_expansion_finite(banks, widths)
    # The old float32 mean gets some of these wrong, so the set is sharp enough.
    old_mean = np.asarray(
        jnp.mean(jnp.asarray(banks[..., 0]), axis=-1) > jnp.asarray(widths) / 2
    )
    assert bool(np.any(old_mean != expected))


def test_spawn_side_rule_matches_exact_reference_on_seeded_random_banks() -> None:
    rng = np.random.default_rng(20260924)
    log_float32_max = float(np.log10(_FLOAT32_MAX))
    # Widths spread evenly in log scale from 1 to the float32 maximum. The
    # edge group uses the top of that range, where an unscaled sum overflows.
    widths = np.minimum(
        np.concatenate(
            (
                10.0 ** rng.uniform(0.0, log_float32_max, 1600),
                10.0 ** rng.uniform(36.0, log_float32_max, 400),
            )
        ),
        _FLOAT32_MAX,
    ).astype(np.float32)
    shares = np.empty((2000, MAX_AGENTS_PER_TEAM))
    # 800 banks spread across the map.
    shares[:800] = rng.uniform(0.0, 1.0, (800, MAX_AGENTS_PER_TEAM))
    # 800 banks near the centre, from far off down to exact ties.
    shares[800:1600] = 0.5 * (
        1.0
        + rng.uniform(-1.0, 1.0, (800, MAX_AGENTS_PER_TEAM))
        * 2.0 ** -rng.integers(0, 40, (800, MAX_AGENTS_PER_TEAM))
    )
    # 400 banks with every pad near the same edge.
    near_edge = rng.uniform(0.0, 1.0, (400, MAX_AGENTS_PER_TEAM)) * (
        2.0 ** -rng.integers(0, 30, (400, 1))
    )
    shares[1600:] = np.where(
        rng.integers(0, 2, (400, 1)) == 1, 1.0 - near_edge, near_edge
    )
    width_column = widths.astype(np.float64)[:, None]
    pad_x = np.clip(shares * width_column, 0.0, width_column).astype(np.float32)
    assert bool(np.all((pad_x >= 0) & (pad_x <= widths[:, None])))
    banks = np.stack(
        (pad_x, np.broadcast_to(np.arange(1, 6, dtype=np.float32), pad_x.shape)),
        axis=-1,
    )
    expected = _assert_side_rule_matches_exact_reference(banks, widths)
    assert 0.2 < float(np.mean(expected)) < 0.8
    _assert_expansion_finite(banks, widths)


def test_spawn_side_rule_matches_exact_sums_on_200000_ordinary_banks() -> None:
    rng = np.random.default_rng(20260925)
    count = 200_000
    widths = rng.uniform(1.0, 1000.0, count).astype(np.float32)
    width64 = widths.astype(np.float64)[:, None]
    # Half the banks anywhere on the map, half within a few float32 steps of
    # the centre, where rounding decides the side.
    anywhere = rng.uniform(0.5, 1.0, (count // 2, MAX_AGENTS_PER_TEAM))
    near = 0.5 + rng.uniform(-1.0, 1.0, (count // 2, MAX_AGENTS_PER_TEAM)) * 2.0**-22
    shares = np.concatenate((anywhere, near))
    pad_x = np.clip(shares * width64, 0.5, width64 - 0.5).astype(np.float32)
    pad_x[:, 0] = np.where(rng.integers(0, 2, count) == 1, pad_x[:, 0], 0.5)
    banks = np.stack(
        (pad_x, np.broadcast_to(np.float32(1.0), pad_x.shape)), axis=-1
    ).astype(np.float32)
    # Pads in [0.5, 1000] and five of them need at most 38 bits, so these
    # float64 sums and 5 * (width / 2) are exact: an independent exact reference.
    pad_sums = np.sum(pad_x.astype(np.float64), axis=1)
    half_widths = widths.astype(np.float64) / 2.0
    exact_right = np.asarray(pad_sums > 5.0 * half_widths, dtype=np.bool_)
    banks_array, widths_array = jnp.asarray(banks), jnp.asarray(widths)
    for core in (
        spawn_bank_on_right(banks_array, widths_array),
        _compiled_side(banks_array, widths_array),
        _compiled_side_by_bank(banks_array, widths_array),
    ):
        assert core.shape == (count,) and core.dtype == jnp.bool_
        assert np.array_equal(np.asarray(core), exact_right)
    # The pure-Python host copy on a fixed sample of 2,000 banks.
    sample = slice(None, None, 100)
    host = tuple(
        models.red_zone_team_on_right(float(width), tuple(float(x) for x in bank))
        for bank, width in zip(pad_x[sample], widths[sample], strict=True)
    )
    assert len(host) == 2000
    assert np.array_equal(np.asarray(host, dtype=np.bool_), exact_right[sample])
    assert 0.2 < float(np.mean(exact_right)) < 0.8


@pytest.mark.parametrize(
    ("pad_shape", "width_shape", "side_shape"),
    (
        ((5, 2), (), ()),
        ((2, 5, 2), (), (2,)),
        ((7, 5, 2), (7,), (7,)),
        ((7, 2, 5, 2), (7, 1), (7, 2)),
    ),
    ids=("one-bank", "both-teams", "batch", "batch-of-both-teams"),
)
def test_spawn_side_rule_keeps_the_leading_shape(
    pad_shape: tuple[int, ...],
    width_shape: tuple[int, ...],
    side_shape: tuple[int, ...],
) -> None:
    pads = jnp.broadcast_to(jnp.asarray(_bank((1.0, 2.0, 3.0, 4.0, 19.0))), pad_shape)
    widths = jnp.full(width_shape, 20.0, dtype=jnp.float32)
    for side in (spawn_bank_on_right(pads, widths), _compiled_side(pads, widths)):
        assert side.shape == side_shape and side.dtype == jnp.bool_
        assert not bool(jnp.any(side))


@pytest.mark.parametrize(
    "pad_shape",
    ((2, 5, 3), (2, 4, 2), (5,), (10, 2), (2, 5, 2, 1)),
    ids=("three-coordinates", "four-pads", "flat", "ten-pads", "extra-axis"),
)
def test_spawn_side_rule_rejects_a_wrong_trailing_shape(
    pad_shape: tuple[int, ...],
) -> None:
    pads = jnp.ones(pad_shape, dtype=jnp.float32)
    with pytest.raises(ValueError, match=_SIDE_SHAPE_ERROR):
        spawn_bank_on_right(pads, 20.0)
    with pytest.raises(ValueError, match=_SIDE_SHAPE_ERROR):
        _compiled_side(pads, jnp.asarray(20.0))


@pytest.mark.parametrize(
    ("map_width", "depth"),
    (
        (20.0, 5.0),
        (17.3, 5.5),
        (20.3, 0.1),
        (12.0, 12.000000000000002),
        (12.1, 12.1),
        (20.0, 1e-8),
        (_WIDTH_3E38, 1e38),
        (_FLOAT32_MAX, _FLOAT32_MAX),
    ),
    ids=(
        "default",
        "fractional-width",
        "thin",
        "rounds-to-width",
        "depth-equal-to-float32-width",
        "collapsed-right",
        "huge",
        "float32-max",
    ),
)
def test_host_red_zone_x_range_returns_the_float32_bounds_core_uses(
    map_width: float, depth: float
) -> None:
    width32 = np.float32(map_width)
    depth32 = np.float32(depth)
    left = models.red_zone_x_range(map_width, depth, False)
    right = models.red_zone_x_range(map_width, depth, True)
    assert left == (0.0, float(depth32))
    assert right == (float(np.float32(width32 - depth32)), float(width32))
    assert all(type(bound) is float for bound in left + right)
    assert right[0] <= right[1]


def test_host_red_zone_x_range_keeps_collapsed_and_full_width_strips() -> None:
    assert models.red_zone_x_range(20.0, 1e-8, True) == (20.0, 20.0)
    width = _f32(12.1)
    assert models.red_zone_x_range(12.1, 12.1, True) == (0.0, width)
    assert models.red_zone_x_range(12.1, 12.1, False) == (0.0, width)
    assert models.red_zone_x_range(12.0, 12.000000000000002, True) == (0.0, 12.0)


@pytest.mark.parametrize(
    "depth",
    (
        0.0,
        -1.0,
        _FLOAT32_SMALLEST_SUBNORMAL,
        _FLOAT32_LARGEST_SUBNORMAL,
        float(np.nextafter(np.float32(12), np.float32(np.inf))),
        12.5,
        float("nan"),
        float("inf"),
    ),
    ids=(
        "zero",
        "negative",
        "smallest-subnormal",
        "largest-subnormal",
        "next-float32-above-width",
        "above-width",
        "nan",
        "infinity",
    ),
)
def test_host_red_zone_x_range_rejects_depths_core_rejects(depth: float) -> None:
    for on_right in (False, True):
        with pytest.raises(
            ValueError, match="normal float32 value in \\(0, map_width\\]"
        ):
            models.red_zone_x_range(12.0, depth, on_right)


@cache
def _base_actor_input() -> ActorInput:
    config = _real_core_mapping_parity_config()
    _, observation, _, _ = reset(config, jax.random.PRNGKey(0))
    return build_actor_input(observation, config)


def _actor_view(own: np.ndarray, enemy: np.ndarray, widths: np.ndarray) -> ActorInput:
    # Lay banks out as actors see them: pads (L, 2, 5, 2) with the own team
    # first, and the map width in each context row (L, 20).
    base = _base_actor_input()
    context = np.zeros((*widths.shape, CONTEXT_FEATURES), dtype=np.float32)
    context[..., CONTEXT_FEATURE_MAP_WIDTH] = widths
    lifecycle = base.observation.spawn_lifecycle._replace(
        spawn_pad_positions_by_agent_by_team=jnp.asarray(
            np.stack((own, enemy), axis=-3)
        )
    )
    observation = base.observation._replace(
        spawn_lifecycle=lifecycle, context_features=jnp.asarray(context)
    )
    return base._replace(observation=observation)


def test_policy_side_helper_uses_the_core_rule_and_old_actors_keep_the_mean() -> None:
    map_banks, map_widths = _installed_map_banks()
    own = map_banks.reshape(-1, MAX_AGENTS_PER_TEAM, 2)
    enemy = map_banks[:, ::-1].reshape(-1, MAX_AGENTS_PER_TEAM, 2)
    widths = np.repeat(map_widths, 2)
    expected = _exact_sides(own, widths)
    actors = _actor_view(own, enemy, widths)
    compiled_team_on_right = cast(_ActorSide, jax.jit(team_on_right))
    for side in (team_on_right(actors), compiled_team_on_right(actors)):
        np.testing.assert_array_equal(np.asarray(side), expected)
    # The old mean formula agreed with the exact rule on every installed map.
    np.testing.assert_array_equal(
        np.asarray(_historical_team_on_right(actors)), expected
    )

    centred_widths = np.asarray((17.3, 20.3), dtype=np.float32)
    centred = np.stack(
        tuple(_bank(_centred_spread(float(width))) for width in centred_widths)
    )
    centred_actors = _actor_view(centred, centred[:, ::-1], centred_widths)
    np.testing.assert_array_equal(np.asarray(team_on_right(centred_actors)), False)
    np.testing.assert_array_equal(
        np.asarray(
            spawn_bank_on_right(jnp.asarray(centred), jnp.asarray(centred_widths))
        ),
        False,
    )
    # Actors trained on input schema 1 keep the old answer: right.
    for side in (
        _historical_team_on_right(centred_actors),
        cast(_ActorSide, jax.jit(_historical_team_on_right))(centred_actors),
    ):
        np.testing.assert_array_equal(np.asarray(side), True)


@pytest.mark.parametrize("swap_banks", (False, True), ids=("stored", "swapped"))
def test_policy_side_helper_reads_each_actor_s_own_bank_from_real_observations(
    swap_banks: bool,
) -> None:
    config = make_standard_team_deathmatch_config(
        map_id=42,
        team_a_roster=("mage", "priest", "hunter"),
        team_b_roster=("rogue", "warrior"),
    )
    if swap_banks:
        config = config._replace(
            team_spawn_pad_positions=config.team_spawn_pad_positions[::-1]
        )
    _, observation, _, _ = reset(config, jax.random.PRNGKey(5))
    actors = build_actor_input(observation, config)
    team_sides = np.asarray(
        spawn_bank_on_right(config.team_spawn_pad_positions, config.map_width)
    )
    np.testing.assert_array_equal(team_sides, (swap_banks, not swap_banks))
    active = np.asarray(config.agent_profile.active_mask)
    expected = np.where(active, np.repeat(team_sides, MAX_AGENTS_PER_TEAM), np.False_)
    np.testing.assert_array_equal(np.asarray(team_on_right(actors)), expected)
