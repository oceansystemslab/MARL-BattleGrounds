"""Synthetic Scenario 5 pursuit, inheritance, and same-epoch information proof."""

from collections.abc import Callable
from math import cos, radians, sin
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scripts.dev.visual_debugger.authoring_compiler import CompiledDevScenarioV1
from tests.scenario_controller_fixtures import load_scenario_1

from marl_battlegrounds.core.geometry import GEOMETRY_TOLERANCE
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_MAP_WIDTH,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MOVE_EAST,
    MOVE_NORTHEAST,
    MOVE_STAY,
    MOVE_WEST,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TEAM_B_ID,
    WARRIOR_CLASS_ID,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.reactive_common import (
    _body_aware_move,  # pyright: ignore[reportPrivateUsage]
    _body_bypass_moves,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.policies.reactive_tdm import (
    reactive_tdm_controller_descriptor,
    reactive_tdm_policy,
)
from marl_battlegrounds.policies.scenario_5 import (
    scenario_5_controller_descriptor,
    scenario_5_policy,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsPolicy,
    SharedObsSensorSourceBankV1,
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)

_POLICY = cast(SharedObsPolicy, jax.jit(scenario_5_policy))
_BASELINE = cast(SharedObsPolicy, jax.jit(reactive_tdm_policy))
_TEAM = cast(Callable[..., ActorAction], execute_shared_obs_team_policy)
_BYPASS = cast(Callable[..., Array], jax.jit(_body_bypass_moves))


@pytest.fixture(scope="module")
def scenario() -> CompiledDevScenarioV1:
    # Reuse an existing approved shape/template fixture, never the authoring store.
    return load_scenario_1()


def _scalar(tree: object, slot: int = 8) -> object:
    def select(leaf: Array) -> Array:
        return leaf[slot]

    return jax.tree.map(select, tree)


def _row(
    template: Array,
    xy: tuple[float, float],
    hp: float = 40,
    class_id: int = WARRIOR_CLASS_ID,
) -> Array:
    return (
        template.at[AGENT_FEATURE_ACTIVE]
        .set(1)
        .at[AGENT_FEATURE_ALIVE]
        .set(1)
        .at[AGENT_FEATURE_CLASS_ID]
        .set(class_id)
        .at[AGENT_FEATURE_CURRENT_HEALTH]
        .set(hp)
        .at[AGENT_FEATURE_X]
        .set(xy[0])
        .at[AGENT_FEATURE_Y]
        .set(xy[1])
    )


def _observation(scenario: CompiledDevScenarioV1) -> Observation:
    base = cast(Observation, _scalar(scenario.observation))
    own = (
        _row(base.self_features, (10, 5), class_id=ROGUE_CLASS_ID)
        .at[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
        .set(1)
        .at[AGENT_FEATURE_RADIUS]
        .set(0.5)
        .at[AGENT_FEATURE_BASIC_INTERACTION_RADIUS]
        .set(1.5)
    )
    return base._replace(
        self_features=own,
        ally_unit_features=jnp.zeros_like(base.ally_unit_features).at[3].set(own),
        enemy_unit_features=jnp.zeros_like(base.enemy_unit_features),
        ally_visibility_mask=jnp.array([False, False, False, True, False]),
        enemy_visibility_mask=jnp.zeros(5, dtype=jnp.bool_),
        map_obstacle_features=jnp.zeros_like(base.map_obstacle_features),
        context_features=base.context_features.at[CONTEXT_FEATURE_MAP_WIDTH]
        .set(24)
        .at[CONTEXT_FEATURE_MAP_HEIGHT]
        .set(10),
    )


def _enemy(
    obs: Observation,
    row: int,
    xy: tuple[float, float],
    hp: float = 40,
    class_id: int = WARRIOR_CLASS_ID,
) -> Observation:
    return obs._replace(
        enemy_unit_features=obs.enemy_unit_features.at[row].set(
            _row(obs.self_features, xy, hp, class_id)
        ),
        enemy_visibility_mask=obs.enemy_visibility_mask.at[row].set(True),
    )


def _mask(*pairs: tuple[int, int], moves: tuple[int, ...] | None = None) -> ActionMask:
    joint = jnp.zeros((11, 2), dtype=jnp.bool_).at[0, 0].set(True)
    for target, ultimate in pairs:
        joint = joint.at[target, ultimate].set(True)
    movement = jnp.ones(9, dtype=jnp.bool_)
    if moves is not None:
        movement = jnp.zeros(9, dtype=jnp.bool_).at[jnp.array((0, *moves))].set(True)
    return ActionMask(movement, joint.any(axis=1), joint.any(axis=0), joint)


def _bank(scenario: CompiledDevScenarioV1) -> SharedObsSensorSourceBankV1:
    def zeros(leaf: Array) -> Array:
        return jnp.zeros_like(leaf)

    return jax.tree.map(
        zeros, build_shared_obs_sensor_source_bank(scenario.observation)
    )


def _act(
    scenario: CompiledDevScenarioV1,
    obs: Observation,
    mask: ActionMask | None = None,
    *,
    policy: SharedObsPolicy = _POLICY,
) -> ActorAction:
    return policy(
        obs,
        _mask() if mask is None else mask,
        jax.random.key(0),
        _bank(scenario),
        jnp.zeros(10, dtype=jnp.bool_),
        jnp.int32(8),
    )


def _exact(actual: object, expected: object) -> None:
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    "class_id", [MAGE_CLASS_ID, WARRIOR_CLASS_ID, HUNTER_CLASS_ID, PRIEST_CLASS_ID]
)
@pytest.mark.parametrize(
    "condition", ["ordinary", "no_combat", "stunned", "dead", "inactive"]
)
def test_nonrogue_actions_are_exactly_inherited(
    scenario: CompiledDevScenarioV1, class_id: int, condition: str
) -> None:
    obs = _enemy(_observation(scenario), 0, (11.2, 5), 10, PRIEST_CLASS_ID)
    own = obs.self_features.at[AGENT_FEATURE_CLASS_ID].set(class_id)
    if condition in ("dead", "inactive"):
        own = own.at[
            AGENT_FEATURE_ALIVE if condition == "dead" else AGENT_FEATURE_ACTIVE
        ].set(0)
    obs = obs._replace(
        self_features=own,
        ally_unit_features=obs.ally_unit_features.at[3].set(own),
    )
    mask = _mask((0, 1), (4, 0), (4, 1), (6, 0), (6, 1))
    if condition in ("no_combat", "stunned"):
        mask = _mask(moves=() if condition == "stunned" else None)
    _exact(_act(scenario, obs, mask), _act(scenario, obs, mask, policy=_BASELINE))


@pytest.mark.parametrize("prey_class", [PRIEST_CLASS_ID, HUNTER_CLASS_ID])
def test_priority_pursuit_and_nearby_low_health_combat_are_independent(
    scenario: CompiledDevScenarioV1,
    prey_class: int,
) -> None:
    obs = _enemy(_observation(scenario), 0, (9, 5), 1)
    obs = _enemy(obs, 4, (13, 5), 40, prey_class)
    result = _act(scenario, obs, _mask((6, 0), (6, 1)))
    assert tuple(map(int, result)) == (MOVE_EAST, 6, 1)
    assert int(_act(scenario, obs, policy=_BASELINE).move) == MOVE_WEST


@pytest.mark.parametrize("prey_class", [PRIEST_CLASS_ID, HUNTER_CLASS_ID])
def test_priority_health_slot_ties_and_reselection_have_no_memory(
    scenario: CompiledDevScenarioV1,
    prey_class: int,
) -> None:
    obs = _enemy(_observation(scenario), 0, (13, 5), 30, prey_class)
    obs = _enemy(obs, 1, (7, 5), 20, prey_class)
    assert int(_act(scenario, obs).move) == MOVE_WEST
    tied = _enemy(obs, 0, (13, 5), 20, prey_class)
    assert int(_act(scenario, tied).move) == MOVE_EAST
    disappeared = tied._replace(
        enemy_visibility_mask=tied.enemy_visibility_mask.at[0].set(False)
    )
    assert int(_act(scenario, disappeared).move) == MOVE_WEST
    assert int(_act(scenario, tied).move) == MOVE_EAST


@pytest.mark.parametrize("prey_class", [PRIEST_CLASS_ID, HUNTER_CLASS_ID])
@pytest.mark.parametrize("kind", ["hidden", "dead", "inactive", "zero_health"])
def test_ineligible_priority_enemy_is_neither_prey_nor_blocker(
    scenario: CompiledDevScenarioV1, prey_class: int, kind: str
) -> None:
    obs = _enemy(_observation(scenario), 0, (13, 5), 20, prey_class)
    expected = _act(scenario, obs)
    changed = _enemy(obs, 1, (10.7, 5), 1, prey_class)
    if kind == "hidden":
        changed = changed._replace(
            enemy_visibility_mask=changed.enemy_visibility_mask.at[1].set(False)
        )
    else:
        field = {
            "dead": AGENT_FEATURE_ALIVE,
            "inactive": AGENT_FEATURE_ACTIVE,
            "zero_health": AGENT_FEATURE_CURRENT_HEALTH,
        }[kind]
        changed = changed._replace(
            enemy_unit_features=changed.enemy_unit_features.at[1, field].set(0)
        )
    _exact(_act(scenario, changed), expected)


def test_priest_priority_returns_after_hunter_fallback(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _enemy(_observation(scenario), 0, (7, 5), 1, HUNTER_CLASS_ID)
    obs = _enemy(obs, 4, (13, 5), 40, PRIEST_CLASS_ID)
    assert int(_act(scenario, obs).move) == MOVE_EAST
    hidden = obs._replace(
        enemy_visibility_mask=obs.enemy_visibility_mask.at[4].set(False)
    )
    assert int(_act(scenario, hidden).move) == MOVE_WEST
    dead = obs._replace(
        enemy_unit_features=obs.enemy_unit_features.at[4, AGENT_FEATURE_ALIVE].set(0)
    )
    assert int(_act(scenario, dead).move) == MOVE_WEST
    assert int(_act(scenario, obs).move) == MOVE_EAST


def test_no_priest_or_hunter_movement_uses_tdm_without_body_screening(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _enemy(_observation(scenario), 0, (13, 5), 20)
    obs = obs._replace(
        ally_unit_features=obs.ally_unit_features.at[0].set(
            _row(obs.self_features, (11, 5))
        ),
        ally_visibility_mask=obs.ally_visibility_mask.at[0].set(True),
    )
    assert int(_act(scenario, obs).move) == MOVE_EAST
    _exact(_act(scenario, obs), _act(scenario, obs, policy=_BASELINE))
    no_enemy = obs._replace(enemy_visibility_mask=jnp.zeros(5, dtype=jnp.bool_))
    assert tuple(map(int, _act(scenario, no_enemy))) == (MOVE_EAST, 0, 0)
    _exact(_act(scenario, no_enemy), _act(scenario, no_enemy, policy=_BASELINE))


@pytest.mark.parametrize("prey_class", [PRIEST_CLASS_ID, HUNTER_CLASS_ID])
def test_prey_contact_is_permitted_but_an_attacked_blocker_is_avoided(
    scenario: CompiledDevScenarioV1,
    prey_class: int,
) -> None:
    obs = _enemy(_observation(scenario), 4, (11, 5), 40, prey_class)
    assert int(_act(scenario, obs).move) == MOVE_EAST
    obs = _enemy(obs, 4, (13, 5), 40, prey_class)
    ally_blocked = obs._replace(
        ally_unit_features=obs.ally_unit_features.at[0].set(
            _row(obs.self_features, (11.5, 5))
        ),
        ally_visibility_mask=obs.ally_visibility_mask.at[0].set(True),
    )
    assert int(_act(scenario, ally_blocked).move) != MOVE_EAST
    obs = _enemy(obs, 0, (11.5, 5), 1)
    result = _act(scenario, obs, _mask((6, 0), (6, 1)))
    assert int(result.move) not in (MOVE_EAST, MOVE_STAY)
    assert (int(result.select_target), int(result.use_ultimate)) == (6, 1)
    # Retain the old specialist's useful generic mask/detour proof after its
    # retirement: safe retreat is allowed, and a blocked-only choice is Stay.
    assert int(_act(scenario, obs, _mask(moves=(MOVE_WEST,))).move) == MOVE_WEST
    assert int(_act(scenario, obs, _mask(moves=(MOVE_EAST,))).move) == MOVE_STAY
    assert int(_act(scenario, obs, _mask(moves=())).move) == MOVE_STAY


@pytest.mark.parametrize("overlap", [0.0, 0.5 * GEOMETRY_TOLERANCE])
def test_glancing_contact_has_an_inclusive_45_degree_boundary(
    scenario: CompiledDevScenarioV1,
    overlap: float,
) -> None:
    own = _observation(scenario).self_features
    bodies = jnp.stack([_row(own, (1 - overlap, 0))])
    angles = [0, 44, 45, 46, 90, 180, -44, -45]
    endpoints = jnp.array(
        [(0.5 * cos(radians(angle)), 0.5 * sin(radians(angle))) for angle in angles],
        dtype=jnp.float32,
    )
    args = (jnp.zeros(2), endpoints, jnp.float32(0.5), bodies, jnp.array([True]))
    expected = [False, False, True, True, True, True, False, True]
    np.testing.assert_array_equal(_body_bypass_moves(*args), expected)
    np.testing.assert_array_equal(_BYPASS(*args), expected)


@pytest.mark.parametrize("speed", [1.0, 1e-3, 1e-6, 1e-8])
def test_tiny_head_on_moves_do_not_acquire_angle_tolerance(
    scenario: CompiledDevScenarioV1,
    speed: float,
) -> None:
    own = _observation(scenario).self_features
    bodies = jnp.stack([_row(own, (1, 0))])
    endpoints = jnp.tile(jnp.array([speed, 0], dtype=jnp.float32), (8, 1))
    args = (jnp.zeros(2), endpoints, jnp.float32(0.5), bodies, jnp.array([True]))
    np.testing.assert_array_equal(_body_bypass_moves(*args), np.zeros(8, dtype=bool))
    np.testing.assert_array_equal(_BYPASS(*args), np.zeros(8, dtype=bool))


@pytest.mark.parametrize("speed,allowed", [(1e-6, True), (1e-5, False)])
def test_a_tiny_positive_gap_is_clear_until_the_segment_reaches_contact(
    scenario: CompiledDevScenarioV1,
    speed: float,
    allowed: bool,
) -> None:
    own = _observation(scenario).self_features
    bodies = jnp.stack([_row(own, (1 + 0.5 * GEOMETRY_TOLERANCE, 0))])
    endpoints = jnp.tile(jnp.array([speed, 0], dtype=jnp.float32), (8, 1))
    args = (jnp.zeros(2), endpoints, jnp.float32(0.5), bodies, jnp.array([True]))
    np.testing.assert_array_equal(_body_bypass_moves(*args), np.full(8, allowed))
    np.testing.assert_array_equal(_BYPASS(*args), np.full(8, allowed))


@pytest.mark.parametrize("body_y,allowed", [(0.6, False), (0.8, True), (1.1, True)])
def test_new_contact_uses_first_contact_normal_and_clear_paths_pass(
    scenario: CompiledDevScenarioV1,
    body_y: float,
    allowed: bool,
) -> None:
    own = _observation(scenario).self_features
    bodies = jnp.stack([_row(own, (2, body_y))])
    endpoints = jnp.tile(jnp.array([3.0, 0.0]), (8, 1))
    args = (jnp.zeros(2), endpoints, jnp.float32(0.5), bodies, jnp.array([True]))
    # With y=0.8 the start-center angle is below 45 degrees, but the incidence
    # at the actual first contact is above it. With y=1.1 there is no contact.
    np.testing.assert_array_equal(_body_bypass_moves(*args), np.full(8, allowed))
    np.testing.assert_array_equal(_BYPASS(*args), np.full(8, allowed))


@pytest.mark.parametrize("overlap", [2 * GEOMETRY_TOLERANCE, 0.25])
def test_existing_overlap_requires_a_nondeepening_outward_escape(
    scenario: CompiledDevScenarioV1,
    overlap: float,
) -> None:
    own = _observation(scenario).self_features
    bodies = jnp.stack([_row(own, (1 - overlap, 0))])
    endpoints = jnp.array(
        [
            [-0.5, 0],
            [0, 0.5],
            [0.5, 0.5],
            [3, 0],
            [0, 0],
            [-0.5, 0.5],
            [0, -0.5],
            [0.5, -0.5],
        ],
        dtype=jnp.float32,
    )
    args = (jnp.zeros(2), endpoints, jnp.float32(0.5), bodies, jnp.array([True]))
    # Finishing beyond the blocker cannot excuse passing deeper through it.
    expected = [True, True, False, False, False, True, True, False]
    np.testing.assert_array_equal(_body_bypass_moves(*args), expected)
    np.testing.assert_array_equal(_BYPASS(*args), expected)


def test_every_admitted_body_must_allow_the_candidate(
    scenario: CompiledDevScenarioV1,
) -> None:
    own = _observation(scenario).self_features
    bodies = jnp.stack([_row(own, (2, 0.8)), _row(own, (2, 0))])
    endpoints = jnp.tile(jnp.array([3.0, 0.0]), (8, 1))
    args = (jnp.zeros(2), endpoints, jnp.float32(0.5), bodies)
    np.testing.assert_array_equal(
        _BYPASS(*args, jnp.array([True, False])), np.ones(8, dtype=bool)
    )
    np.testing.assert_array_equal(
        _BYPASS(*args, jnp.array([True, True])), np.zeros(8, dtype=bool)
    )
    np.testing.assert_array_equal(
        _BYPASS(*args, jnp.array([False, False])), np.ones(8, dtype=bool)
    )


def test_body_aware_movement_admits_glancing_contact(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _observation(scenario)
    bodies = jnp.stack([_row(obs.self_features, (11, 5))])
    args = (
        obs,
        _mask(moves=(MOVE_NORTHEAST,)),
        jnp.array([13, 7]),
        bodies,
        jnp.array([True]),
    )
    assert int(_body_aware_move(*args)) == MOVE_NORTHEAST


@pytest.mark.parametrize("own_y,expected", [(5.0, MOVE_NORTHEAST), (9.5, MOVE_STAY)])
def test_glancing_screen_uses_wall_projected_displacement(
    scenario: CompiledDevScenarioV1,
    own_y: float,
    expected: int,
) -> None:
    obs = _observation(scenario)
    own = _row(obs.self_features, (10, own_y), class_id=ROGUE_CLASS_ID)
    obs = obs._replace(
        self_features=own,
        ally_unit_features=obs.ally_unit_features.at[3]
        .set(own)
        .at[0]
        .set(_row(own, (11, own_y))),
        ally_visibility_mask=obs.ally_visibility_mask.at[0].set(True),
    )
    obs = _enemy(obs, 4, (13, own_y), 40, PRIEST_CLASS_ID)
    # At the north boundary, a nominal northeast move projects to head-on east.
    assert int(_act(scenario, obs, _mask(moves=(MOVE_NORTHEAST,))).move) == expected


@pytest.mark.parametrize("distance,legal", [(1.49, True), (1.5, True), (1.51, False)])
def test_combat_is_basic_radius_bounded_even_without_a_priest(
    scenario: CompiledDevScenarioV1, distance: float, legal: bool
) -> None:
    obs = _enemy(_observation(scenario), 0, (10 + distance, 5), 1)
    result = _act(scenario, obs, _mask((6, 0), (6, 1)))
    assert int(result.select_target) == (6 if legal else 0)
    assert int(result.use_ultimate) == int(legal)


def test_combat_health_slot_ties_ultimate_priority_and_exact_mask(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _enemy(_observation(scenario), 0, (11.4, 5), 10)
    obs = _enemy(obs, 1, (8.6, 5), 10)
    basic = _act(scenario, obs, _mask((6, 0), (7, 0)))
    assert (int(basic.select_target), int(basic.use_ultimate)) == (6, 0)
    ultimate = _act(scenario, obs, _mask((6, 0), (7, 1)))
    assert (int(ultimate.select_target), int(ultimate.use_ultimate)) == (7, 1)
    assert int(_act(scenario, obs).select_target) == 0


@pytest.mark.parametrize("field", [AGENT_FEATURE_ACTIVE, AGENT_FEATURE_ALIVE])
def test_rogue_lifecycle_and_stun_force_canonical_noops(
    scenario: CompiledDevScenarioV1, field: int
) -> None:
    obs = _enemy(_observation(scenario), 0, (11, 5), 1, PRIEST_CLASS_ID)
    stopped = obs._replace(self_features=obs.self_features.at[field].set(0))
    assert tuple(map(int, _act(scenario, stopped, _mask((6, 0), (6, 1))))) == (0, 0, 0)
    assert tuple(map(int, _act(scenario, obs, _mask(moves=())))) == (0, 0, 0)
    # Reappearing living actors immediately use the same current rules.
    assert tuple(map(int, _act(scenario, obs, _mask((6, 0), (6, 1))))) == (
        MOVE_EAST,
        6,
        1,
    )


@pytest.mark.parametrize("prey_class", [PRIEST_CLASS_ID, HUNTER_CLASS_ID])
def test_shared_priority_sighting_guides_movement_without_bypassing_own_mask(
    scenario: CompiledDevScenarioV1,
    prey_class: int,
) -> None:
    obs = _enemy(_observation(scenario), 0, (9, 5), 1)
    bank = _bank(scenario)
    prey = _row(obs.self_features, (13, 5), 40, prey_class)
    bank = bank._replace(
        unit_features_by_sensor_source_and_global_slot=(
            bank.unit_features_by_sensor_source_and_global_slot.at[5, 4].set(prey)
        ),
        unit_visibility_by_sensor_source_and_global_slot=(
            bank.unit_visibility_by_sensor_source_and_global_slot.at[5, 4].set(True)
        ),
    )
    unavailable = jnp.zeros(10, dtype=jnp.bool_)
    available = unavailable.at[5].set(True)
    mask = _mask((6, 0))
    args = (obs, mask, jax.random.key(0), bank)
    local = _POLICY(*args, unavailable, jnp.int32(8))
    shared = _POLICY(*args, available, jnp.int32(8))
    assert int(local.move) == MOVE_WEST
    assert tuple(map(int, shared)) == (MOVE_EAST, 6, 0)
    # Neither another source nor a hidden row can change that admitted sighting.
    hostile = bank._replace(
        unit_features_by_sensor_source_and_global_slot=(
            bank.unit_features_by_sensor_source_and_global_slot.at[0].set(999)
        ),
        unit_visibility_by_sensor_source_and_global_slot=(
            bank.unit_visibility_by_sensor_source_and_global_slot.at[0].set(True)
        ),
    )
    _exact(
        shared,
        _POLICY(obs, mask, jax.random.key(999), hostile, available, jnp.int32(8)),
    )
    hidden = bank._replace(
        unit_visibility_by_sensor_source_and_global_slot=(
            bank.unit_visibility_by_sensor_source_and_global_slot.at[5, 4].set(False)
        ),
    )
    _exact(
        local, _POLICY(obs, mask, jax.random.key(0), hidden, available, jnp.int32(8))
    )


def test_determinism_eager_jit_team_parity_and_descriptor_freshness(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs, mask = scenario.observation, scenario.action_mask
    bank = build_shared_obs_sensor_source_bank(obs)
    availability = build_default_shared_obs_information_availability(
        scenario.config.agent_profile.active_mask,
        scenario.config.agent_profile.team_ids,
    )
    keys = jax.random.split(jax.random.key(0), 10)
    team = _TEAM(obs, mask, keys, bank, availability, scenario_5_policy, TEAM_B_ID)
    for slot in range(5, 10):
        args = (
            cast(Observation, _scalar(obs, slot)),
            cast(ActionMask, _scalar(mask, slot)),
            keys[slot],
            bank,
            availability[slot],
            jnp.int32(slot),
        )
        eager = scenario_5_policy(*args)
        _exact(eager, _POLICY(*args))
        _exact(eager, _scalar(team, slot - 5))
        _exact(eager, _POLICY(*args[:2], jax.random.key(99), *args[3:]))
        for leaf in eager:
            assert leaf.shape == ()
            assert leaf.dtype == jnp.int32
    descriptor = scenario_5_controller_descriptor()
    assert descriptor["version"] == 2
    assert descriptor["policy_id"] == "scenario-5-pressure-controller"
    assert descriptor["inherited_controller"] == reactive_tdm_controller_descriptor()
    descriptor["version"] = 999
    cast(dict[str, object], descriptor["inherited_controller"])["version"] = 999
    fresh = scenario_5_controller_descriptor()
    assert fresh["version"] == 2
    assert fresh["inherited_controller"] == reactive_tdm_controller_descriptor()
