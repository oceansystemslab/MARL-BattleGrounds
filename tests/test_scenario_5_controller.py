"""Synthetic Scenario 5 pursuit, inheritance, and same-epoch information proof."""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scripts.dev.visual_debugger.authoring_compiler import CompiledDevScenarioV1
from tests.scenario_controller_fixtures import load_scenario_1

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


def test_priest_pursuit_and_nearby_low_health_combat_are_independent(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _enemy(_observation(scenario), 0, (9, 5), 1)
    obs = _enemy(obs, 4, (13, 5), 40, PRIEST_CLASS_ID)
    result = _act(scenario, obs, _mask((6, 0), (6, 1)))
    assert tuple(map(int, result)) == (MOVE_EAST, 6, 1)
    assert int(_act(scenario, obs, policy=_BASELINE).move) == MOVE_WEST


def test_priest_health_slot_ties_and_reselection_have_no_memory(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _enemy(_observation(scenario), 0, (13, 5), 30, PRIEST_CLASS_ID)
    obs = _enemy(obs, 1, (7, 5), 20, PRIEST_CLASS_ID)
    assert int(_act(scenario, obs).move) == MOVE_WEST
    tied = _enemy(obs, 0, (13, 5), 20, PRIEST_CLASS_ID)
    assert int(_act(scenario, tied).move) == MOVE_EAST
    disappeared = tied._replace(
        enemy_visibility_mask=tied.enemy_visibility_mask.at[0].set(False)
    )
    assert int(_act(scenario, disappeared).move) == MOVE_WEST
    assert int(_act(scenario, tied).move) == MOVE_EAST


@pytest.mark.parametrize("kind", ["hidden", "dead", "inactive", "zero_health"])
def test_ineligible_priest_is_neither_prey_nor_blocker(
    scenario: CompiledDevScenarioV1, kind: str
) -> None:
    obs = _enemy(_observation(scenario), 0, (13, 5), 20, PRIEST_CLASS_ID)
    expected = _act(scenario, obs)
    changed = _enemy(obs, 1, (10.7, 5), 1, PRIEST_CLASS_ID)
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


def test_no_priest_movement_uses_tdm_without_body_screening(
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


def test_priest_contact_is_permitted_but_an_attacked_blocker_is_avoided(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _enemy(_observation(scenario), 4, (11, 5), 40, PRIEST_CLASS_ID)
    assert int(_act(scenario, obs).move) == MOVE_EAST
    obs = _enemy(obs, 4, (13, 5), 40, PRIEST_CLASS_ID)
    obs = _enemy(obs, 0, (11.5, 5), 1)
    result = _act(scenario, obs, _mask((6, 0), (6, 1)))
    assert int(result.move) not in (MOVE_EAST, MOVE_STAY)
    assert (int(result.select_target), int(result.use_ultimate)) == (6, 1)


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


def test_shared_priest_sighting_guides_movement_without_bypassing_own_mask(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _enemy(_observation(scenario), 0, (9, 5), 1)
    bank = _bank(scenario)
    priest = _row(obs.self_features, (13, 5), 40, PRIEST_CLASS_ID)
    bank = bank._replace(
        unit_features_by_sensor_source_and_global_slot=(
            bank.unit_features_by_sensor_source_and_global_slot.at[5, 4].set(priest)
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
    assert descriptor["version"] == 1
    assert descriptor["policy_id"] == "scenario-5-pressure-controller"
    assert descriptor["inherited_controller"] == reactive_tdm_controller_descriptor()
    descriptor["version"] = 999
    cast(dict[str, object], descriptor["inherited_controller"])["version"] = 999
    fresh = scenario_5_controller_descriptor()
    assert fresh["version"] == 1
    assert fresh["inherited_controller"] == reactive_tdm_controller_descriptor()
