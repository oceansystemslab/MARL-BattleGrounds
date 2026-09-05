"""Scenario 3 steering, information boundaries, and real body-blocking trajectories."""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scripts.dev.visual_debugger.authoring_compiler import CompiledDevScenarioV1
from tests.scenario_controller_fixtures import (
    SCENARIO_3_SEMANTIC_DIGEST,
    load_scenario_3,
)

from marl_battlegrounds.core.env import initialize_scenario_state, step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_NORTHWEST,
    MOVE_SOUTH,
    MOVE_SOUTHWEST,
    MOVE_STAY,
    MOVE_WEST,
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
from marl_battlegrounds.policies.scenario_3 import (
    _body_clear_moves,  # pyright: ignore[reportPrivateUsage]
    scenario_3_controller_descriptor,
    scenario_3_policy,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsPolicy,
    SharedObsSensorSourceBankV1,
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)

_POLICY = cast(SharedObsPolicy, jax.jit(scenario_3_policy))
_TEAM = cast(Callable[..., ActorAction], execute_shared_obs_team_policy)
_STEP = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(step),
)


@pytest.fixture(scope="module")
def scenario() -> CompiledDevScenarioV1:
    return load_scenario_3()


def _scalar(tree: object, slot: int = 8) -> object:
    def select(leaf: Array) -> Array:
        return leaf[slot]

    return jax.tree.map(select, tree)


def _zeros(leaf: Array) -> Array:
    return jnp.zeros_like(leaf)


def _ones(leaf: Array) -> Array:
    return jnp.ones_like(leaf)


def _row(own: Array, xy: tuple[float, float], hp: float = 10) -> Array:
    return (
        own.at[AGENT_FEATURE_X]
        .set(xy[0])
        .at[AGENT_FEATURE_Y]
        .set(xy[1])
        .at[AGENT_FEATURE_CURRENT_HEALTH]
        .set(hp)
        .at[AGENT_FEATURE_ACTIVE]
        .set(1)
        .at[AGENT_FEATURE_ALIVE]
        .set(1)
    )


def _observation(scenario: CompiledDevScenarioV1) -> Observation:
    obs = cast(Observation, _scalar(scenario.observation))
    own = _row(obs.self_features, (10, 5), 47)
    return obs._replace(
        self_features=own,
        ally_unit_features=jnp.zeros_like(obs.ally_unit_features).at[3].set(own),
        enemy_unit_features=jnp.zeros_like(obs.enemy_unit_features),
        ally_visibility_mask=jnp.array([False, False, False, True, False]),
        enemy_visibility_mask=jnp.zeros(5, dtype=jnp.bool_),
        map_obstacle_features=jnp.zeros_like(obs.map_obstacle_features),
    )


def _enemy(
    obs: Observation, row: int, xy: tuple[float, float], hp: float
) -> Observation:
    return obs._replace(
        enemy_unit_features=obs.enemy_unit_features.at[row].set(
            _row(obs.self_features, xy, hp)
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
    return jax.tree.map(
        _zeros, build_shared_obs_sensor_source_bank(scenario.observation)
    )


def _act(
    scenario: CompiledDevScenarioV1,
    obs: Observation,
    mask: ActionMask | None = None,
    *,
    key: int = 0,
) -> ActorAction:
    return _POLICY(
        obs,
        _mask() if mask is None else mask,
        jax.random.key(key),
        _bank(scenario),
        jnp.zeros(10, dtype=jnp.bool_),
        jnp.int32(8),
    )


def _exact(left: object, right: object) -> None:
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_pursuit_uses_lowest_health_while_combat_uses_legal_nearby_target(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _enemy(_enemy(_observation(scenario), 0, (9, 5), 65), 1, (13, 5), 1)
    result = _act(scenario, obs, _mask((6, 0), (6, 1)))
    assert tuple(map(int, result)) == (MOVE_EAST, 6, 1)
    # The same prey wins combat priority as soon as it becomes a legal target.
    obs = _enemy(obs, 1, (11.2, 5), 1)
    result = _act(scenario, obs, _mask((6, 0), (6, 1), (7, 0), (7, 1)))
    assert int(result.select_target) == 7
    assert int(result.use_ultimate) == 1


@pytest.mark.parametrize("distance,legal", [(1.49, True), (1.5, True), (1.51, False)])
def test_combat_enforces_basic_radius_even_with_an_ultimate_mask(
    scenario: CompiledDevScenarioV1, distance: float, legal: bool
) -> None:
    obs = _enemy(_observation(scenario), 0, (10 + distance, 5), 1)
    result = _act(scenario, obs, _mask((6, 0), (6, 1)))
    assert int(result.use_ultimate) == int(legal)
    assert int(result.select_target) == (6 if legal else 0)


def test_health_slot_ties_masks_and_cooldown_fallback(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _enemy(_enemy(_observation(scenario), 0, (11.4, 5), 10), 1, (8.6, 5), 10)
    basic = _act(scenario, obs, _mask((6, 0), (7, 0)))
    assert int(basic.select_target) == 6
    assert int(basic.use_ultimate) == 0
    ultimate = _act(scenario, obs, _mask((6, 0), (7, 0), (7, 1)))
    assert tuple(map(int, ultimate))[1:] == (7, 1)
    assert int(_act(scenario, obs, _mask()).select_target) == 0


@pytest.mark.parametrize("kind", ["hidden", "dead", "inactive", "zero_health"])
def test_ineligible_enemy_cannot_become_prey_or_blocker(
    scenario: CompiledDevScenarioV1, kind: str
) -> None:
    obs = _enemy(_observation(scenario), 0, (13, 5), 10)
    expected = _act(scenario, obs)
    changed = _enemy(obs, 1, (10.7, 5), 1)
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


def test_other_bodies_are_avoided_but_prey_and_self_are_exempt(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _enemy(_observation(scenario), 0, (13, 5), 1)
    assert int(_act(scenario, obs).move) == MOVE_EAST
    # A living ally between actor and prey is as much a blocker as an enemy.
    blocked = obs._replace(
        ally_unit_features=obs.ally_unit_features.at[0].set(
            _row(obs.self_features, (11.5, 5), 100)
        ),
        ally_visibility_mask=obs.ally_visibility_mask.at[0].set(True),
    )
    assert int(_act(scenario, blocked).move) != MOVE_EAST
    prey_at_contact = _enemy(obs, 0, (11, 5), 1)
    assert int(_act(scenario, prey_at_contact).move) == MOVE_EAST


def test_segment_intersection_tangency_and_initial_overlap_escape(
    scenario: CompiledDevScenarioV1,
) -> None:
    own = cast(Observation, _scalar(scenario.observation)).self_features
    bodies = jnp.stack([_row(own, (1, 0))])
    body_mask = jnp.array([True])
    # The first endpoint is clear, but reaching it crosses the body.
    endpoints = jnp.array([[3.0, 0.0], [0.0, 2.0], [-1.0, 0.0]], dtype=jnp.float32)
    clear = _body_clear_moves(
        jnp.array([0.0, 0.0]), endpoints, jnp.float32(0.5), bodies, body_mask
    )
    np.testing.assert_array_equal(clear, [False, True, True])
    # Starting in an overlap: an outward or sideways escape is allowed, inward isn't.
    bodies = jnp.stack([_row(own, (0.5, 0))])
    endpoints = jnp.array([[-1.0, 0.0], [0.0, 1.0], [1.5, 0.0]], dtype=jnp.float32)
    clear = _body_clear_moves(
        jnp.array([0.0, 0.0]), endpoints, jnp.float32(0.5), bodies, body_mask
    )
    np.testing.assert_array_equal(clear, [True, True, False])


def test_safe_detour_can_retreat_and_blocked_moves_stay(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _enemy(_observation(scenario), 0, (13, 5), 1)
    assert int(_act(scenario, obs, _mask(moves=(MOVE_WEST,))).move) == MOVE_WEST
    assert int(_act(scenario, obs, _mask(moves=())).move) == MOVE_STAY
    blocked = obs._replace(
        ally_unit_features=obs.ally_unit_features.at[0].set(
            _row(obs.self_features, (11, 5))
        ),
        ally_visibility_mask=obs.ally_visibility_mask.at[0].set(True),
    )
    assert int(_act(scenario, blocked, _mask(moves=(MOVE_EAST,))).move) == MOVE_STAY


@pytest.mark.parametrize(
    "field,value",
    [
        (AGENT_FEATURE_ACTIVE, 0),
        (AGENT_FEATURE_ALIVE, 0),
        (AGENT_FEATURE_CLASS_ID, MAGE_CLASS_ID),
    ],
)
def test_dead_inactive_and_nonrogue_agents_noop(
    scenario: CompiledDevScenarioV1, field: int, value: int
) -> None:
    obs = _enemy(_observation(scenario), 0, (11, 5), 1)
    obs = obs._replace(self_features=obs.self_features.at[field].set(value))
    assert tuple(map(int, _act(scenario, obs, _mask((6, 0), (6, 1))))) == (0, 0, 0)


def test_no_enemy_stays_and_stun_mask_overrides_preferences(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs = _observation(scenario)
    assert tuple(map(int, _act(scenario, obs))) == (0, 0, 0)
    obs = _enemy(obs, 0, (11, 5), 1)
    assert tuple(map(int, _act(scenario, obs, _mask(moves=())))) == (0, 0, 0)


def test_determinism_eager_jit_team_parity_and_unavailable_information(
    scenario: CompiledDevScenarioV1,
) -> None:
    obs, mask = scenario.observation, scenario.action_mask
    bank = build_shared_obs_sensor_source_bank(obs)
    availability = build_default_shared_obs_information_availability(
        scenario.config.agent_profile.active_mask,
        scenario.config.agent_profile.team_ids,
    )
    keys = jax.random.split(jax.random.key(0), 10)
    team = _TEAM(obs, mask, keys, bank, availability, scenario_3_policy, TEAM_B_ID)
    for slot in range(5, 10):
        scalar_obs = cast(Observation, _scalar(obs, slot))
        scalar_mask = cast(ActionMask, _scalar(mask, slot))
        args = (
            scalar_obs,
            scalar_mask,
            keys[slot],
            bank,
            availability[slot],
            jnp.int32(slot),
        )
        eager = scenario_3_policy(*args)
        _exact(eager, _POLICY(*args))
        _exact(eager, _scalar(team, slot - 5))
        _exact(
            eager,
            _POLICY(
                scalar_obs,
                scalar_mask,
                jax.random.key(99),
                bank,
                availability[slot],
                jnp.int32(slot),
            ),
        )
    scalar_obs = cast(Observation, _scalar(obs))
    scalar_mask = cast(ActionMask, _scalar(mask))
    unavailable = jnp.zeros(10, dtype=jnp.bool_)
    ones = jax.tree.map(_ones, bank)
    _exact(
        _POLICY(scalar_obs, scalar_mask, keys[8], bank, unavailable, jnp.int32(8)),
        _POLICY(scalar_obs, scalar_mask, keys[8], ones, unavailable, jnp.int32(8)),
    )


def _run(
    scenario: CompiledDevScenarioV1, warrior_moves: tuple[int, ...]
) -> list[tuple[ActorAction, EnvState]]:
    state, obs, mask = (
        scenario.initial_state,
        scenario.observation,
        scenario.action_mask,
    )
    availability = build_default_shared_obs_information_availability(
        scenario.config.agent_profile.active_mask,
        scenario.config.agent_profile.team_ids,
    )
    result: list[tuple[ActorAction, EnvState]] = []
    for tick, warrior_move in enumerate(warrior_moves):
        b = _TEAM(
            obs,
            mask,
            jax.random.split(jax.random.key(tick), 10),
            build_shared_obs_sensor_source_bank(obs),
            availability,
            scenario_3_policy,
            TEAM_B_ID,
        )
        a = ActorAction(
            jnp.zeros(5, dtype=jnp.int32).at[1].set(warrior_move),
            jnp.zeros(5, dtype=jnp.int32),
            jnp.zeros(5, dtype=jnp.int32),
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
        state, obs, _, _, mask, _ = _STEP(
            scenario.config, state, mask, joint, jax.random.key(tick)
        )
        result.append((b, state))
    return result


def test_actual_stationary_blocker_detour_kills_hunter_before_first_wave(
    scenario: CompiledDevScenarioV1,
) -> None:
    assert scenario.semantic_digest == SCENARIO_3_SEMANTIC_DIGEST
    run = _run(scenario, (MOVE_STAY,) * 4)
    assert [int(b.move[3]) for b, _ in run] == [
        MOVE_NORTHWEST,
        MOVE_NORTHWEST,
        MOVE_SOUTHWEST,
        MOVE_SOUTH,
    ]
    assert [int(b.select_target[3]) for b, _ in run] == [7, 7, 0, 8]
    for index, (b, state) in enumerate(run):
        np.testing.assert_array_equal(b.move[jnp.array([0, 1, 2, 4])], 0)
        assert bool(state.alive_mask[2]) == (index < 3)
        assert int(state.step_count) == index + 1
        assert int(state.team_respawn_wave_countdowns[0]) == 3 - index
    final = run[-1][1]
    assert float(final.current_health[8]) == 47
    assert float(final.current_health[1]) == pytest.approx(44.6, abs=1e-5)
    np.testing.assert_array_equal(final.team_deathmatch_scores, [0, 1])


def test_blocker_changes_and_moving_counterplay_use_observed_successors(
    scenario: CompiledDevScenarioV1,
) -> None:
    still = _run(scenario, (MOVE_STAY,) * 3)
    moving = _run(scenario, (MOVE_NORTH, MOVE_STAY, MOVE_STAY))
    assert int(still[0][0].move[3]) == int(moving[0][0].move[3])
    assert any(
        int(a.move[3]) != int(b.move[3])
        for (a, _), (b, _) in zip(still[1:], moving[1:], strict=True)
    )
    shifted = scenario.initial_state._replace(
        agent_positions=scenario.initial_state.agent_positions.at[1, 1].set(5.4)
    )
    _, obs, mask, _ = initialize_scenario_state(shifted, scenario.config)
    bank = build_shared_obs_sensor_source_bank(obs)
    availability = build_default_shared_obs_information_availability(
        scenario.config.agent_profile.active_mask,
        scenario.config.agent_profile.team_ids,
    )
    action = _TEAM(
        obs,
        mask,
        jax.random.split(jax.random.key(0), 10),
        bank,
        availability,
        scenario_3_policy,
        TEAM_B_ID,
    )
    assert int(action.move[3]) == MOVE_SOUTHWEST


def test_nonrogue_respawn_and_descriptor_freshness(
    scenario: CompiledDevScenarioV1,
) -> None:
    run = _run(scenario, (MOVE_STAY,) * 6)
    b, _ = run[-1]
    for field in b:
        np.testing.assert_array_equal(field[jnp.array([0, 1, 2, 4])], 0)
    first = scenario_3_controller_descriptor()
    first["version"] = 99
    fresh = scenario_3_controller_descriptor()
    assert fresh["version"] == 1
    assert fresh["policy_id"] == "scenario-3-pressure-controller"
