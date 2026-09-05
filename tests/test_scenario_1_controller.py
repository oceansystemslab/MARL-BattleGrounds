"""Reactive MRP rules and accepted Scenario 1 and Scenario 2 witnesses."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scripts.dev.visual_debugger.authoring_compiler import (
    CompiledDevScenarioV1,
    compile_dev_scenario,
)
from scripts.dev.visual_debugger.authoring_models import DevScenarioDraftV1
from tests.scenario_controller_fixtures import load_scenario_1

from marl_battlegrounds.core.axis_mappings import (
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY,
)
from marl_battlegrounds.core.env import initialize_scenario_state, step
from marl_battlegrounds.core.geometry import (
    has_clear_line_of_sight,
    project_movement_with_geometry,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    AGENT_FEATURE_MAX_HEALTH,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_MAP_WIDTH,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_NORTHEAST,
    MOVE_NORTHWEST,
    MOVE_SOUTH,
    MOVE_SOUTHEAST,
    MOVE_SOUTHWEST,
    MOVE_STAY,
    MOVE_WEST,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TEAM_B_ID,
    ActionMask,
    DoneFlags,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.scenario_controllers import (
    scenario_1_controller_descriptor,
    scenario_1_policy,
)
from marl_battlegrounds.policies.scenario_controllers.common import refine_movement
from marl_battlegrounds.policies.shared_obs import (
    SharedObsPolicy,
    SharedObsSensorSourceBankV1,
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)

_POLICY = cast(SharedObsPolicy, jax.jit(scenario_1_policy))
_REFINE = cast(
    Callable[[Observation, ActionMask, Array], Array], jax.jit(refine_movement)
)
_STEP = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(step),
)


def _zeros(leaf: Array) -> Array:
    return jnp.zeros_like(leaf)


def _ones(leaf: Array) -> Array:
    return jnp.ones_like(leaf)


@pytest.fixture(scope="module")
def scenario() -> CompiledDevScenarioV1:
    return load_scenario_1()


def _scalar(tree: object, slot: int) -> object:
    def take(leaf: Array) -> Array:
        return leaf[slot]

    return jax.tree.map(take, tree)


def _row(
    template: Array,
    *,
    xy: tuple[float, float],
    hp: float = 50,
    max_hp: float = 100,
) -> Array:
    return (
        template.at[AGENT_FEATURE_ACTIVE]
        .set(1)
        .at[AGENT_FEATURE_ALIVE]
        .set(1)
        .at[AGENT_FEATURE_X]
        .set(xy[0])
        .at[AGENT_FEATURE_Y]
        .set(xy[1])
        .at[AGENT_FEATURE_CURRENT_HEALTH]
        .set(hp)
        .at[AGENT_FEATURE_MAX_HEALTH]
        .set(max_hp)
    )


def _observation(scenario: CompiledDevScenarioV1, class_id: int) -> Observation:
    base = cast(Observation, _scalar(scenario.observation, 9))
    own = _row(base.self_features, xy=(10, 5), hp=60)
    own = (
        own.at[AGENT_FEATURE_CLASS_ID]
        .set(class_id)
        .at[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
        .set(1)
        .at[AGENT_FEATURE_RADIUS]
        .set(0.5)
    )
    return base._replace(
        self_features=own,
        ally_unit_features=jnp.zeros_like(base.ally_unit_features).at[4].set(own),
        enemy_unit_features=jnp.zeros_like(base.enemy_unit_features),
        ally_visibility_mask=jnp.array([False, False, False, False, True]),
        enemy_visibility_mask=jnp.zeros(5, dtype=jnp.bool_),
        map_obstacle_features=jnp.zeros_like(base.map_obstacle_features),
        context_features=base.context_features.at[CONTEXT_FEATURE_MAP_WIDTH]
        .set(20)
        .at[CONTEXT_FEATURE_MAP_HEIGHT]
        .set(10),
    )


def _visible(
    observation: Observation,
    relation: str,
    row: int,
    xy: tuple[float, float],
    *,
    hp: float = 10,
    max_hp: float = 100,
) -> Observation:
    features = _row(observation.self_features, xy=xy, hp=hp, max_hp=max_hp)
    if relation == "ally":
        return observation._replace(
            ally_unit_features=observation.ally_unit_features.at[row].set(features),
            ally_visibility_mask=observation.ally_visibility_mask.at[row].set(True),
        )
    return observation._replace(
        enemy_unit_features=observation.enemy_unit_features.at[row].set(features),
        enemy_visibility_mask=observation.enemy_visibility_mask.at[row].set(True),
    )


def _mask(*pairs: tuple[int, int], move: int | None = None) -> ActionMask:
    joint = jnp.zeros((11, 2), dtype=jnp.bool_).at[0, 0].set(True)
    for target, ultimate in pairs:
        joint = joint.at[target, ultimate].set(True)
    moves = jnp.ones(9, dtype=jnp.bool_)
    if move is not None:
        moves = jnp.zeros_like(moves).at[0].set(True).at[move].set(True)
    return ActionMask(moves, joint.any(axis=1), joint.any(axis=0), joint)


def _empty_bank(scenario: CompiledDevScenarioV1) -> SharedObsSensorSourceBankV1:
    return jax.tree.map(
        _zeros, build_shared_obs_sensor_source_bank(scenario.observation)
    )


def _act(
    scenario: CompiledDevScenarioV1,
    observation: Observation,
    mask: ActionMask | None = None,
) -> ActorAction:
    return _POLICY(
        observation,
        mask if mask is not None else _mask(),
        jax.random.key(0),
        _empty_bank(scenario),
        jnp.zeros(10, dtype=jnp.bool_),
        jnp.int32(9),
    )


def _assert_exact(actual: object, expected: object) -> None:
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    ("ally_x", "enemy_x", "self_lowest", "expected"),
    [
        (14.0, 8.0, False, MOVE_EAST),
        (13.0, 8.0, False, MOVE_EAST),
        (11.0, 8.0, False, MOVE_EAST),
        (11.0, None, False, MOVE_WEST),
        (11.5, None, False, MOVE_WEST),
        (11.501, None, False, MOVE_STAY),
        (13.0, None, False, MOVE_STAY),
        (13.001, None, False, MOVE_EAST),
        (11.0, 8.0, True, MOVE_EAST),
        (11.0, None, True, MOVE_STAY),
    ],
)
def test_priest_ordered_movement_branches(
    scenario: CompiledDevScenarioV1,
    ally_x: float,
    enemy_x: float | None,
    self_lowest: bool,
    expected: int,
) -> None:
    obs = _observation(scenario, PRIEST_CLASS_ID)
    obs = _visible(obs, "ally", 0, (ally_x, 5), hp=100 if self_lowest else 10)
    if enemy_x is not None:
        obs = _visible(obs, "enemy", 0, (enemy_x, 5))
    assert int(_act(scenario, obs).move) == expected


@pytest.mark.parametrize("health", [29.0, 30.0, 30.001])
def test_priest_ultimate_threshold_and_basic_fallback(
    scenario: CompiledDevScenarioV1, health: float
) -> None:
    obs = _visible(
        _observation(scenario, PRIEST_CLASS_ID), "ally", 0, (12, 5), hp=health
    )
    action = _act(scenario, obs, _mask((1, 0), (1, 1), (5, 0)))
    assert int(action.select_target) == 1
    assert int(action.use_ultimate) == int(health <= 30)
    assert int(_act(scenario, obs, _mask((5, 0))).select_target) == 5


def test_priest_hp_max_hp_and_slot_ties_and_full_health_healing(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _observation(scenario, PRIEST_CLASS_ID)
    obs = _visible(obs, "ally", 0, (12, 5), hp=10, max_hp=100)
    obs = _visible(obs, "ally", 1, (8, 5), hp=10, max_hp=50)
    obs = _visible(obs, "ally", 2, (10, 7), hp=10, max_hp=50)
    action = _act(scenario, obs, _mask((1, 0), (2, 0), (3, 0)))
    assert int(action.select_target) == 2
    # The movement tie chooses row 1 too; that ally lies in the desired band.
    assert int(action.move) == MOVE_STAY
    full = _observation(scenario, PRIEST_CLASS_ID)
    full = full._replace(
        ally_unit_features=full.ally_unit_features.at[
            4, AGENT_FEATURE_CURRENT_HEALTH
        ].set(100)
    )
    assert int(_act(scenario, full, _mask((5, 0))).select_target) == 5


@pytest.mark.parametrize("class_id", [MAGE_CLASS_ID, ROGUE_CLASS_ID])
@pytest.mark.parametrize("distance", [1.0, 2.0, 3.0])
def test_enemy_distance_intent_and_no_enemy_stay(
    scenario: CompiledDevScenarioV1, class_id: int, distance: float
) -> None:
    obs = _observation(scenario, class_id)
    assert int(_act(scenario, obs).move) == MOVE_STAY
    obs = _visible(obs, "enemy", 0, (10 + distance, 5))
    expected = MOVE_EAST
    if class_id == MAGE_CLASS_ID:
        expected = (
            MOVE_WEST if distance < 2 else MOVE_STAY if distance == 2 else MOVE_EAST
        )
    assert int(_act(scenario, obs).move) == expected


def test_rogue_nearest_movement_but_lowest_health_legal_combat(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _observation(scenario, ROGUE_CLASS_ID)
    obs = _visible(obs, "enemy", 0, (11, 5), hp=50)
    obs = _visible(obs, "enemy", 1, (8, 5), hp=10)
    obs = _visible(obs, "enemy", 2, (10, 6), hp=1)
    action = _act(scenario, obs, _mask((6, 0), (7, 0), (7, 1)))
    assert tuple(int(x) for x in action) == (MOVE_EAST, 7, 1)
    basic = _act(scenario, obs, _mask((6, 0), (7, 0)))
    assert tuple(int(x) for x in basic) == (MOVE_EAST, 7, 0)


def test_mage_burst_requires_own_legal_basic_target(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _visible(_observation(scenario, MAGE_CLASS_ID), "enemy", 0, (13, 5))
    assert tuple(int(x) for x in _act(scenario, obs, _mask((0, 1)))) == (
        MOVE_EAST,
        0,
        0,
    )
    assert tuple(int(x) for x in _act(scenario, obs, _mask((6, 0), (0, 1)))) == (
        MOVE_EAST,
        0,
        1,
    )
    assert int(_act(scenario, obs, _mask((6, 0))).select_target) == 6


@pytest.mark.parametrize("field", [AGENT_FEATURE_ACTIVE, AGENT_FEATURE_ALIVE])
def test_lifecycle_and_stun_masks_override_preferences(
    scenario: CompiledDevScenarioV1, field: int
) -> None:
    obs = _visible(_observation(scenario, ROGUE_CLASS_ID), "enemy", 0, (12, 5))
    assert tuple(int(x) for x in _act(scenario, obs, _mask(move=MOVE_STAY))) == (
        0,
        0,
        0,
    )
    obs = obs._replace(self_features=obs.self_features.at[field].set(0))
    assert tuple(int(x) for x in _act(scenario, obs, _mask((6, 0), (6, 1)))) == (
        0,
        0,
        0,
    )


def test_hidden_dead_inactive_and_zero_health_candidates_cannot_win(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _observation(scenario, ROGUE_CLASS_ID)
    for row in range(5):
        obs = _visible(obs, "enemy", row, (11, 5), hp=1 if row < 4 else 40)
    obs = obs._replace(
        enemy_visibility_mask=obs.enemy_visibility_mask.at[0].set(False),
        enemy_unit_features=obs.enemy_unit_features.at[1, AGENT_FEATURE_ALIVE]
        .set(0)
        .at[2, AGENT_FEATURE_ACTIVE]
        .set(0)
        .at[3, AGENT_FEATURE_CURRENT_HEALTH]
        .set(0),
    )
    assert (
        int(_act(scenario, obs, _mask(*[(6 + i, 0) for i in range(5)])).select_target)
        == 10
    )


def test_unavailable_shared_rows_and_unused_history_cannot_change_action(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _observation(scenario, ROGUE_CLASS_ID)
    bank = _empty_bank(scenario)
    no_sources = jnp.zeros(10, dtype=jnp.bool_)
    expected = _POLICY(obs, _mask(), jax.random.key(0), bank, no_sources, jnp.int32(9))
    hostile = jax.tree.map(_ones, bank)
    changed = obs._replace(
        previous_timestep_actions=jax.tree.map(_ones, obs.previous_timestep_actions)
    )
    actual = _POLICY(
        changed, _mask(), jax.random.key(77), hostile, no_sources, jnp.int32(9)
    )
    _assert_exact(actual, expected)
    sighting = _row(obs.self_features, xy=(13, 5))
    bank = bank._replace(
        unit_features_by_sensor_source_and_global_slot=bank.unit_features_by_sensor_source_and_global_slot.at[
            5, 0
        ].set(sighting),
        unit_visibility_by_sensor_source_and_global_slot=bank.unit_visibility_by_sensor_source_and_global_slot.at[
            5, 0
        ].set(True),
    )
    actual = _POLICY(
        obs, _mask(), jax.random.key(0), bank, no_sources.at[5].set(True), jnp.int32(9)
    )
    assert tuple(int(x) for x in actual) == (MOVE_EAST, 0, 0)


def test_movement_allows_clipping_rejects_effectively_blocked_and_preserves_stay(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _observation(scenario, ROGUE_CLASS_ID)
    for start, expected in [(19.0, MOVE_EAST), (19.45, MOVE_STAY), (19.5, MOVE_STAY)]:
        adjusted = obs._replace(
            self_features=obs.self_features.at[AGENT_FEATURE_X].set(start)
        )
        assert (
            int(_REFINE(adjusted, _mask(move=MOVE_EAST), jnp.array([1.0, 0.0])))
            == expected
        )
    assert int(_REFINE(obs, _mask(), jnp.zeros(2))) == MOVE_STAY
    assert int(_REFINE(obs, _mask(move=MOVE_STAY), jnp.array([1.0, 0.0]))) == MOVE_STAY


def test_movement_projection_matches_isolated_candidates_and_ignores_bodies(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = cast(Observation, _scalar(scenario.observation, 5))
    own = obs.self_features
    origin = own[jnp.array([AGENT_FEATURE_X, AGENT_FEATURE_Y])]
    speed = own[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
    radii = jnp.full(MAX_AGENT_SLOTS, own[AGENT_FEATURE_RADIUS])
    active = jnp.arange(MAX_AGENT_SLOTS) == 0
    no_bodies = jnp.zeros(MAX_AGENT_SLOTS, dtype=jnp.bool_)
    for move in range(1, 9):
        direction = UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY[move]
        isolated = project_movement_with_geometry(
            jnp.broadcast_to(origin, (MAX_AGENT_SLOTS, 2)),
            radii,
            jnp.zeros((MAX_AGENT_SLOTS, 2)).at[0].set(direction * speed),
            active,
            active,
            obs.context_features[CONTEXT_FEATURE_MAP_WIDTH],
            obs.context_features[CONTEXT_FEATURE_MAP_HEIGHT],
            obs.map_obstacle_features,
            no_bodies,
            no_bodies,
            agent_agent_overlap_projection_passes=0,
        )[0]
        expected = (
            move
            if float(np.linalg.norm(np.asarray(isolated - origin)))
            >= 0.1 * float(speed)
            else MOVE_STAY
        )
        assert int(_REFINE(obs, _mask(move=move), direction)) == expected
    bodies = obs._replace(
        ally_unit_features=jnp.ones_like(obs.ally_unit_features) * 999,
        enemy_unit_features=jnp.ones_like(obs.enemy_unit_features) * 999,
    )
    _assert_exact(
        _REFINE(obs, _mask(), jnp.array([1.0, 0.0])),
        _REFINE(bodies, _mask(), jnp.array([1.0, 0.0])),
    )


@pytest.mark.parametrize(
    ("obstacle", "direct", "refined"),
    [
        ([OBSTACLE_TYPE_PILLAR, 11.5, 5, 0.5, 0, 0, 0, 1], MOVE_EAST, MOVE_EAST),
        ([OBSTACLE_TYPE_PILLAR, 11, 5, 0.5, 0, 0, 0, 1], MOVE_STAY, MOVE_NORTHEAST),
        ([OBSTACLE_TYPE_WALL, 11, 5, 0, 1, 8, 0, 1], MOVE_STAY, MOVE_NORTHEAST),
    ],
)
def test_static_obstacle_contact_clipping_and_slide_alternatives(
    scenario: CompiledDevScenarioV1,
    obstacle: list[float],
    direct: int,
    refined: int,
) -> None:
    obs = _observation(scenario, ROGUE_CLASS_ID)
    obs = obs._replace(
        map_obstacle_features=obs.map_obstacle_features.at[0].set(jnp.array(obstacle))
    )
    east = jnp.array([1.0, 0.0])
    assert int(_REFINE(obs, _mask(move=MOVE_EAST), east)) == direct
    assert int(_REFINE(obs, _mask(), east)) == refined


def _team_b(
    scenario: CompiledDevScenarioV1, obs: Observation, mask: ActionMask, key: int = 0
) -> ActorAction:
    profile = scenario.config.agent_profile
    availability = build_default_shared_obs_information_availability(
        profile.active_mask, profile.team_ids
    )
    return cast(
        ActorAction,
        execute_shared_obs_team_policy(
            obs,
            mask,
            jax.random.split(jax.random.key(key), 10),
            build_shared_obs_sensor_source_bank(obs),
            availability,
            scenario_1_policy,
            TEAM_B_ID,
        ),
    )


def test_scalar_eager_jit_team_parity_and_key_invariance(
    scenario: CompiledDevScenarioV1,
) -> None:
    profile = scenario.config.agent_profile
    av = build_default_shared_obs_information_availability(
        profile.active_mask, profile.team_ids
    )
    bank = build_shared_obs_sensor_source_bank(scenario.observation)
    team = _team_b(scenario, scenario.observation, scenario.action_mask)
    _assert_exact(
        team, _team_b(scenario, scenario.observation, scenario.action_mask, 77)
    )
    args = (
        cast(Observation, _scalar(scenario.observation, 9)),
        cast(ActionMask, _scalar(scenario.action_mask, 9)),
        jax.random.key(0),
        bank,
        av[9],
        jnp.int32(9),
    )
    eager = scenario_1_policy(*args)
    _assert_exact(eager, _POLICY(*args))
    _assert_exact(eager, _scalar(team, 4))


@pytest.mark.parametrize(
    ("heal_target", "survivor", "hp"), [(3, 2, 2.575), (4, 3, 20.0)]
)
def test_reactive_controller_reproduces_both_accepted_witnesses(
    scenario: CompiledDevScenarioV1, heal_target: int, survivor: int, hp: float
) -> None:
    state, obs, mask = (
        scenario.initial_state,
        scenario.observation,
        scenario.action_mask,
    )
    expected_moves = [
        [MOVE_EAST, 0, 0, MOVE_SOUTH, MOVE_NORTH],
        [MOVE_SOUTHWEST, 0, 0, MOVE_EAST, 0],
    ]
    expected_targets = [[10, 0, 0, 8, 1], [0, 0, 0, 8, 0]]
    reward = None
    for tick in range(3):
        b = _team_b(scenario, obs, mask)
        if tick < 2:
            np.testing.assert_array_equal(b.move, expected_moves[tick])
            np.testing.assert_array_equal(b.select_target, expected_targets[tick])
        a = ActorAction(
            jnp.array(
                [0, 0, MOVE_EAST, MOVE_SOUTH, 0]
                if tick == 0
                else [0, 0, MOVE_NORTHEAST, 0, 0]
                if tick == 1
                else [0] * 5,
                dtype=jnp.int32,
            ),
            jnp.array(
                [0, 0, 10, 0, heal_target]
                if tick == 0
                else [0, 0, 6, 0, 0]
                if tick == 1
                else [0, 0, 6, 9, 0],
                dtype=jnp.int32,
            ),
            jnp.array(
                [0, 0, 1, 0, 0]
                if tick == 0
                else [0] * 5
                if tick == 1
                else [0, 0, 0, 1, 0],
                dtype=jnp.int32,
            ),
        )
        joint = build_joint_action_from_actor_actions(a, b)
        slots = jnp.arange(10)
        assert bool(jnp.all(mask.move_mask[slots, joint.move]))
        assert bool(
            jnp.all(
                mask.select_target_use_ultimate_joint_mask[
                    slots, joint.select_target, joint.use_ultimate
                ]
            )
        )
        if tick == 2:
            assert not bool(mask.select_target_use_ultimate_joint_mask[5, 9, 0])
            assert int(b.select_target[0]) == 8
            assert int(b.select_target[3]) == (9 if heal_target == 3 else 8)
            np.testing.assert_array_equal(b.move, [MOVE_EAST, 0, 0, MOVE_NORTHEAST, 0])
        np.testing.assert_array_equal(b.use_ultimate, [0, 0, 0, 0, 0])
        state, obs, reward, done, mask, _ = _STEP(
            scenario.config, state, mask, joint, jax.random.key(0)
        )
        assert bool(done.done) == (tick == 2)
    np.testing.assert_array_equal(state.team_deathmatch_scores, [20, 19])
    assert reward is not None
    assert float(reward.rewards[2]) == 1.0
    assert float(state.current_health[survivor]) == pytest.approx(hp, abs=1e-5)
    assert int(state.stun_durations[9, 1]) == 2


def test_descriptor_is_fresh_and_contains_frozen_constants() -> None:
    first = scenario_1_controller_descriptor()
    assert first["policy_id"] == "scenario-1-pressure-controller"
    assert first["version"] == 1
    cast(dict[str, object], first["movement"])["minimum_stride_fraction_inclusive"] = 99
    assert (
        cast(dict[str, object], scenario_1_controller_descriptor()["movement"])[
            "minimum_stride_fraction_inclusive"
        ]
        == 0.1
    )


def test_not_trapping_priest_changes_next_decision_reactively(
    scenario: CompiledDevScenarioV1,
) -> None:
    b = _team_b(scenario, scenario.observation, scenario.action_mask)
    # Counterfactual to the witness: Hunter basics Mage instead of trapping Priest.
    a = ActorAction(
        jnp.array([0, 0, MOVE_EAST, MOVE_SOUTH, 0], dtype=jnp.int32),
        jnp.array([0, 0, 6, 0, 3], dtype=jnp.int32),
        jnp.zeros(5, dtype=jnp.int32),
    )
    _, obs, _, done, mask, _ = _STEP(
        scenario.config,
        scenario.initial_state,
        scenario.action_mask,
        build_joint_action_from_actor_actions(a, b),
        jax.random.key(0),
    )
    assert not bool(done.done)
    next_b = _team_b(scenario, obs, mask)
    assert int(next_b.use_ultimate[4]) == 1
    assert int(next_b.select_target[4]) == 1


def test_rules_continue_to_final_wave_without_unsupported_class_decisions(
    scenario: CompiledDevScenarioV1,
) -> None:
    # Larger K isolates the five-transition lifecycle from early score termination.
    config = scenario.config._replace(team_deathmatch_score_threshold=100)
    state, obs, mask, _ = initialize_scenario_state(scenario.initial_state, config)
    manual = ActorAction(*(jnp.zeros(5, dtype=jnp.int32) for _ in range(3)))
    for tick in range(5):
        assert not bool(jnp.any(state.alive_mask[6:8]))
        b = _team_b(scenario, obs, mask)
        for action_head in b:
            np.testing.assert_array_equal(action_head[1:3], [0, 0])
        joint = build_joint_action_from_actor_actions(manual, b)
        slots = jnp.arange(10)
        assert bool(jnp.all(mask.move_mask[slots, joint.move]))
        assert bool(
            jnp.all(
                mask.select_target_use_ultimate_joint_mask[
                    slots, joint.select_target, joint.use_ultimate
                ]
            )
        )
        state, obs, _, done, mask, _ = _STEP(
            config, state, mask, joint, jax.random.key(0)
        )
        assert bool(done.done) == (tick == 4)
    assert bool(jnp.all(state.alive_mask[6:8]))


def test_reactive_controller_reproduces_scenario_2_cover_and_healing_witness() -> None:
    fixture = Path(__file__).parent / "fixtures" / "scenario_2_r12.json"
    draft = DevScenarioDraftV1.model_validate_json(fixture.read_text(encoding="utf-8"))
    scenario = compile_dev_scenario(draft)
    assert draft.revision == 12
    assert scenario.semantic_digest == (
        "1b2e2d391053a0de15c7f292dc70f6db679ed9e62e35ec1f11dcc68bb9f1c2cc"
    )
    state, obs, mask = (
        scenario.initial_state,
        scenario.observation,
        scenario.action_mask,
    )
    # Warrior/Priest: Charge/self-heal, basic/heal, no-combat/Ultimate, basic/no-combat.
    # Target actions are observer-relative; Team A's 9 is Rogue-B and 6 is Mage-B.
    moves = [
        [0, MOVE_SOUTH, 0, 0, MOVE_WEST],
        [0, MOVE_SOUTHEAST, 0, 0, MOVE_NORTHWEST],
        [0, MOVE_NORTHWEST, 0, 0, MOVE_STAY],
        [0, MOVE_STAY, 0, 0, MOVE_STAY],
    ]
    targets = [[0, 9, 0, 0, 5], [0, 9, 0, 0, 2], [0, 0, 0, 0, 2], [0, 6, 0, 0, 0]]
    ultimates = [[0, 1, 0, 0, 0], [0] * 5, [0, 0, 0, 0, 1], [0] * 5]
    expected_b_moves = [
        [MOVE_WEST, 0, 0, MOVE_WEST, 0],
        [MOVE_SOUTH, 0, 0, 0, 0],
        [MOVE_SOUTH, 0, 0, 0, 0],
        [MOVE_NORTH, 0, 0, 0, 0],
    ]
    expected_scores = [[18, 19], [19, 19], [19, 19], [20, 19]]
    expected_warrior_hp = [15.9387493, 4.8774986, 185.8162537, 166.7550049]
    for tick in range(4):
        if tick == 2:
            # Priest is protected by cover, not by being outside Mage's basic range.
            mage, priest = state.agent_positions[5], state.agent_positions[4]
            distance = float(jnp.linalg.norm(mage - priest))
            assert distance == pytest.approx(2.982093, abs=1e-5)
            assert distance < float(
                scenario.config.agent_profile.basic_interaction_radii[5]
            )
            assert not bool(
                has_clear_line_of_sight(mage, priest, scenario.config.obstacles)
            )
            assert not bool(mask.select_target_use_ultimate_joint_mask[5, 10, 0])
            assert bool(mask.select_target_use_ultimate_joint_mask[4, 2, 1])
        # These are expected outputs only; Team B is generated afresh from this epoch.
        b = _team_b(scenario, obs, mask)
        np.testing.assert_array_equal(b.move, expected_b_moves[tick])
        np.testing.assert_array_equal(
            b.select_target, [7, 0, 0, 10 if tick == 0 else 0, 0]
        )
        np.testing.assert_array_equal(b.use_ultimate, [0] * 5)
        a = ActorAction(
            jnp.array(moves[tick], dtype=jnp.int32),
            jnp.array(targets[tick], dtype=jnp.int32),
            jnp.array(ultimates[tick], dtype=jnp.int32),
        )
        joint = build_joint_action_from_actor_actions(a, b)
        slots = jnp.arange(MAX_AGENT_SLOTS)
        assert bool(jnp.all(mask.move_mask[slots, joint.move]))
        assert bool(
            jnp.all(
                mask.select_target_use_ultimate_joint_mask[
                    slots, joint.select_target, joint.use_ultimate
                ]
            )
        )
        state, obs, reward, done, mask, _ = _STEP(
            scenario.config, state, mask, joint, jax.random.key(0)
        )
        np.testing.assert_array_equal(
            state.team_deathmatch_scores, expected_scores[tick]
        )
        assert float(state.current_health[1]) == pytest.approx(
            expected_warrior_hp[tick], abs=1e-5
        )
        assert float(state.current_health[4]) == 4.0
        assert bool(done.terminated) == (tick == 3)
        assert not bool(done.truncated)
        assert float(reward.rewards[1]) == (1.0 if tick == 3 else 0.0)
    assert int(state.step_count) == 299
