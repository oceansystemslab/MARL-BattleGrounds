"""Check movement around wall ends and clearance in real paired-agent steps."""

from collections.abc import Callable
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
from scripts.dev.visual_debugger.authoring_models import DevWallV1, new_scenario_draft
from tests.scenario_controller_fixtures import load_scenario_1
from tests.test_reactive_tdm import (
    _mask,  # pyright: ignore[reportPrivateUsage]
    _observation,  # pyright: ignore[reportPrivateUsage]
    _scalar,  # pyright: ignore[reportPrivateUsage]
)

from marl_battlegrounds.core.env import initialize_scenario_state, step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_NORTHEAST,
    MOVE_NORTHWEST,
    MOVE_SOUTH,
    MOVE_SOUTHEAST,
    MOVE_SOUTHWEST,
    MOVE_STAY,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    WARRIOR_CLASS_ID,
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
from marl_battlegrounds.policies.reactive_common import (
    _body_aware_move,  # pyright: ignore[reportPrivateUsage]
    _wall_steering,  # pyright: ignore[reportPrivateUsage]
    refine_movement,
)

_STEER = cast(Callable[..., tuple[Array, Array, Array, Array]], jax.jit(_wall_steering))
_REFINE = cast(Callable[..., Array], jax.jit(refine_movement))
_BODY_MOVE = cast(Callable[..., Array], jax.jit(_body_aware_move))
_STEP = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(step),
)


@pytest.fixture(scope="module")
def scenario() -> CompiledDevScenarioV1:
    return load_scenario_1()


def _wall_observation(
    scenario: CompiledDevScenarioV1,
    position: tuple[float, float] = (5.0, 4.664643),
    wall: tuple[float, ...] = (2, 6, 5, 0, 1, 3, 0, 1),
) -> Observation:
    observation = _observation(scenario, WARRIOR_CLASS_ID)
    return observation._replace(
        self_features=observation.self_features.at[AGENT_FEATURE_X]
        .set(position[0])
        .at[AGENT_FEATURE_Y]
        .set(position[1]),
        map_obstacle_features=observation.map_obstacle_features.at[0].set(
            jnp.asarray(wall, dtype=jnp.float32)
        ),
    )


@pytest.mark.parametrize(
    ("position", "goal", "direction", "active", "end"),
    [
        ((5.0, 4.664643), (10.0, 5.5), (0.0, -1.0), True, -1),
        ((7.0, 4.664643), (2.0, 5.5), (0.0, -1.0), True, -1),
        ((5.0, 2.8), (10.0, 5.5), (1.0, 0.0), True, -1),
        ((6.0, 2.8), (10.0, 5.5), (1.0, 0.0), True, -1),
        ((7.0, 2.8), (2.0, 5.5), (-1.0, 0.0), True, -1),
        ((6.0, 2.8), (2.0, 5.5), (-1.0, 0.0), True, -1),
        ((7.0, 2.8), (10.0, 5.5), (3.0, 2.7), False, 0),
        ((5.0, 2.8), (2.0, 5.5), (-3.0, 2.7), False, 0),
    ],
)
def test_mirrored_south_turn_and_far_face_release(
    scenario: CompiledDevScenarioV1,
    position: tuple[float, float],
    goal: tuple[float, float],
    direction: tuple[float, float],
    active: bool,
    end: int,
) -> None:
    observation = _wall_observation(scenario, position)
    args = observation, jnp.asarray(goal, dtype=jnp.float32), jnp.bool_(True)
    eager = _wall_steering(*args)
    compiled = _STEER(*args)
    for left, right in zip(eager, compiled, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(compiled[0], direction, atol=1e-6)
    assert bool(compiled[1]) is active
    assert float(compiled[2]) == end
    assert compiled[0].shape == (2,)
    assert compiled[0].dtype == jnp.float32


@pytest.mark.parametrize(
    ("wall", "position", "expected", "end", "active"),
    [
        ((2, 2.5, 1.5, 0, 1, 3, 0, 1), (1.5, 1.5), MOVE_NORTH, 1, True),
        ((2, 2.5, 1.5, 0, 1, 3, 0, 1), (1.5, 3.8), MOVE_EAST, 0, False),
        ((2, 2.5, 1.5, 0, 1, 3, 0, 1), (2.5, 3.8), MOVE_EAST, 0, False),
        ((2, 6, 2.5, 0, 1, 3, 0, 1), (5.0, 2.5), MOVE_SOUTH, -1, True),
        ((2, 6, 4.5, 0, 1, 9, 0, 1), (5.0, 4.5), MOVE_NORTH, 1, True),
        ((2, 6, 5, 0, 1, 10, 0, 1), (5.0, 5.0), MOVE_STAY, 0, True),
    ],
)
def test_bounds_choose_north_only_when_south_cannot_fit(
    scenario: CompiledDevScenarioV1,
    wall: tuple[float, ...],
    position: tuple[float, float],
    expected: int,
    end: int,
    active: bool,
) -> None:
    observation = _wall_observation(scenario, position, wall)
    goal = jnp.asarray((10.0, position[1]), dtype=jnp.float32)
    direction, steering, selected_end, _ = _STEER(observation, goal, True)
    assert bool(steering) is active
    assert float(selected_end) == end
    if active and end == 0:
        np.testing.assert_array_equal(direction, jnp.zeros(2))
    assert (
        int(_REFINE(observation, _mask(), goal - jnp.asarray(position), approach=True))
        == expected
    )


@pytest.mark.parametrize(
    "wall",
    [
        (2, 6, 5, 0, 1, 3, 0, 1),
        (2, 6, 5, 0, 3, 1, np.pi / 2, 1),
        (2, 6, 5, 0, 1, 3, np.pi, 1),
        (2, 6, 5, 0, 3, 1, -np.pi / 2, 1),
        (2, 5.55, 5, 0, 0.1, 3, 0, 1),
    ],
)
def test_equivalent_vertical_and_thin_walls_share_south_preference(
    scenario: CompiledDevScenarioV1, wall: tuple[float, ...]
) -> None:
    observation = _wall_observation(scenario, wall=wall)
    direction, active, end, _ = _STEER(observation, jnp.array([10.0, 5.5]), True)
    np.testing.assert_array_equal(direction, jnp.array([0.0, -1.0]))
    assert bool(active)
    assert float(end) == -1


@pytest.mark.parametrize(
    "wall",
    [
        (OBSTACLE_TYPE_PILLAR, 6, 5, 0.5, 0, 0, 0, 1),
        (OBSTACLE_TYPE_WALL, 6, 5, 0, 3, 1, 0, 1),
        (OBSTACLE_TYPE_WALL, 6, 5, 0, 1, 3, np.pi / 4, 1),
        (OBSTACLE_TYPE_WALL, 6, 5, 0, 1, 3, 0, 0),
    ],
)
def test_pillars_horizontal_oblique_and_inactive_rows_do_not_override(
    scenario: CompiledDevScenarioV1, wall: tuple[float, ...]
) -> None:
    observation = _wall_observation(scenario, wall=wall)
    goal = jnp.array([10.0, 5.5])
    direction, active, end, _ = _STEER(observation, goal, True)
    np.testing.assert_array_equal(direction, goal - observation.self_features[:2])
    assert not bool(active)
    assert float(end) == 0


@pytest.mark.parametrize("speed", [1.0, 0.2, 0.01])
def test_geometric_end_strip_does_not_shrink_with_slowed_speed(
    scenario: CompiledDevScenarioV1, speed: float
) -> None:
    observation = _wall_observation(scenario, (3.5, 1.1))
    observation = observation._replace(
        self_features=observation.self_features.at[
            AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED
        ].set(speed)
    )
    direction, active, end, _ = _STEER(observation, jnp.array([10.0, 5.5]), True)
    np.testing.assert_array_equal(direction, jnp.array([1.0, 0.0]))
    assert bool(active)
    assert float(end) == -1


@pytest.mark.parametrize("position", [(2.5, 4.5), (5.0, 0.75), (5.0, 8.0)])
def test_positions_outside_local_wall_envelope_keep_their_goal(
    scenario: CompiledDevScenarioV1, position: tuple[float, float]
) -> None:
    observation = _wall_observation(scenario, position)
    goal = jnp.array([10.0, 5.5])
    direction, active, end, _ = _STEER(observation, goal, True)
    np.testing.assert_array_equal(direction, goal - observation.self_features[:2])
    assert not bool(active)
    assert float(end) == 0


@pytest.mark.parametrize(
    "feature",
    [AGENT_FEATURE_ACTIVE, AGENT_FEATURE_ALIVE, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED],
)
def test_lifecycle_and_zero_speed_do_not_activate_steering(
    scenario: CompiledDevScenarioV1, feature: int
) -> None:
    observation = _wall_observation(scenario)
    observation = observation._replace(
        self_features=observation.self_features.at[feature].set(0)
    )
    goal = jnp.array([10.0, 5.5])
    direction, active, end, _ = _STEER(observation, goal, True)
    np.testing.assert_array_equal(direction, goal - observation.self_features[:2])
    assert not bool(active)
    assert float(end) == 0


def test_retreat_zero_intent_and_exact_movement_masks_remain_authoritative(
    scenario: CompiledDevScenarioV1,
) -> None:
    observation = _wall_observation(scenario)
    origin = observation.self_features[:2]
    goal = jnp.array([10.0, 5.5])
    direction, active, end, _ = _STEER(observation, goal, False)
    np.testing.assert_array_equal(direction, goal - origin)
    assert not bool(active)
    assert float(end) == 0
    direction, active, end, _ = _STEER(observation, origin, True)
    np.testing.assert_array_equal(direction, jnp.zeros(2))
    assert not bool(active)
    assert float(end) == 0
    assert int(_REFINE(observation, _mask(), jnp.zeros(2), approach=True)) == MOVE_STAY
    assert (
        int(_REFINE(observation, _mask(move=MOVE_NORTH), goal - origin, approach=True))
        == MOVE_STAY
    )
    assert (
        int(_REFINE(observation, _mask(move=MOVE_SOUTH), goal - origin, approach=True))
        == MOVE_SOUTH
    )
    assert (
        int(_REFINE(observation, _mask(move=MOVE_NORTH), goal - origin)) == MOVE_NORTH
    )


def test_nearest_wall_then_fixed_slot_decides_competing_ends(
    scenario: CompiledDevScenarioV1,
) -> None:
    observation = _wall_observation(scenario, (5.0, 3.0))
    lower_wall = jnp.array([2, 6, 1.5, 0, 1, 3, 0, 1], dtype=jnp.float32)
    observation = observation._replace(
        map_obstacle_features=observation.map_obstacle_features.at[1].set(lower_wall)
    )
    assert float(_STEER(observation, jnp.array([10.0, 3.0]), True)[2]) == 1
    tied = observation._replace(
        self_features=observation.self_features.at[AGENT_FEATURE_Y].set(3.25)
    )
    assert float(_STEER(tied, jnp.array([10.0, 3.25]), True)[2]) == -1
    reordered = tied._replace(
        map_obstacle_features=tied.map_obstacle_features.at[:2].set(
            tied.map_obstacle_features[jnp.array([1, 0])]
        )
    )
    assert float(_STEER(reordered, jnp.array([10.0, 3.25]), True)[2]) == 1


@pytest.mark.parametrize("from_west", [True, False], ids=["west_face", "east_face"])
def test_wall_steering_preserves_specialist_shoulder_screening(
    scenario: CompiledDevScenarioV1, from_west: bool
) -> None:
    x = 5.0 if from_west else 7.0
    observation = _wall_observation(scenario, (x, 4.664643))
    goal = jnp.array([10.0 if from_west else 2.0, 5.5])
    blocker = observation.self_features.at[AGENT_FEATURE_Y].add(-1)
    bodies = jnp.zeros_like(scenario.observation.self_features).at[0].set(blocker)
    no_bodies = jnp.zeros(10, dtype=jnp.bool_)
    assert int(_BODY_MOVE(observation, _mask(), goal, bodies, no_bodies)) == MOVE_SOUTH
    body_mask = no_bodies.at[0].set(True)
    assert int(_BODY_MOVE(observation, _mask(), goal, bodies, body_mask)) == (
        MOVE_SOUTHWEST if from_west else MOVE_SOUTHEAST
    )
    assert (
        int(_BODY_MOVE(observation, _mask(move=MOVE_SOUTH), goal, bodies, body_mask))
        == MOVE_STAY
    )


def test_specialist_prey_on_expanded_east_face_keeps_east_exit(
    scenario: CompiledDevScenarioV1,
) -> None:
    observation = _wall_observation(scenario)
    bodies = jnp.zeros_like(scenario.observation.self_features)
    direction, active, end, _ = _STEER(
        observation,
        jnp.array([7.0, 5.5]),
        True,
        bodies=bodies,
        body_mask=jnp.zeros(10, dtype=jnp.bool_),
    )
    assert bool(active)
    assert float(end) == -1
    np.testing.assert_array_equal(direction, jnp.array([0.0, -1.0]))


def test_specialist_equal_phase_alignment_prefers_closer_prey_endpoint(
    scenario: CompiledDevScenarioV1,
) -> None:
    observation = _wall_observation(
        scenario,
        (3.5, 1.5),
        (2, 2.5, 1.5, 0, 1, 3, 0, 1),
    )
    action_mask = _mask()._replace(
        move_mask=jnp.zeros(9, dtype=jnp.bool_)
        .at[MOVE_NORTHEAST]
        .set(True)
        .at[MOVE_NORTHWEST]
        .set(True)
    )
    move = _BODY_MOVE(
        observation,
        action_mask,
        jnp.array([0.0, 1.5]),
        jnp.zeros_like(scenario.observation.self_features),
        jnp.zeros(10, dtype=jnp.bool_),
    )
    assert int(move) == MOVE_NORTHWEST


@pytest.fixture(scope="module")
def physical_wall_scenario() -> CompiledDevScenarioV1:
    draft = new_scenario_draft("isolated_wall_steering")
    wall = DevWallV1(
        object_id="vertical_wall", center_x=6.0, center_y=5.0, width=1.0, height=3.0
    )
    return compile_dev_scenario(
        draft.model_copy(
            update={
                "content": draft.content.model_copy(
                    update={
                        "embedded_map": draft.content.embedded_map.model_copy(
                            update={"obstacles": (wall,)}
                        )
                    }
                )
            }
        )
    )


@pytest.mark.parametrize(
    "slots", [(1, 2), (0, 2)], ids=["warrior_hunter", "mage_hunter"]
)
@pytest.mark.parametrize("from_west", [True, False], ids=["west_face", "east_face"])
def test_paired_steering_clears_the_actual_wall_without_opposing_body_motion(
    physical_wall_scenario: CompiledDevScenarioV1,
    slots: tuple[int, int],
    from_west: bool,
) -> None:
    config = physical_wall_scenario.config
    x = 5.0 if from_west else 7.0
    authored = physical_wall_scenario.initial_state._replace(
        agent_positions=physical_wall_scenario.initial_state.agent_positions.at[
            slots[0]
        ]
        .set(jnp.array([x, 4.664643]))
        .at[slots[1]]
        .set(jnp.array([x, 5.664643]))
    )
    state, observation, mask, _ = initialize_scenario_state(authored, config)
    goal = jnp.array([10.0 if from_west else 2.0, 5.5], dtype=jnp.float32)
    no_combat = jnp.zeros(5, dtype=jnp.int32)
    manual_b = ActorAction(no_combat, no_combat, no_combat)
    cleared = np.zeros(2, dtype=bool)
    for tick in range(12):
        moves = jnp.zeros(5, dtype=jnp.int32)
        for slot in slots:
            own_observation = cast(Observation, _scalar(observation, slot))
            own_mask = cast(ActionMask, _scalar(mask, slot))
            move = _REFINE(
                own_observation,
                own_mask,
                goal - own_observation.self_features[:2],
                approach=True,
            )
            if tick == 0:
                assert int(move) == MOVE_SOUTH
            moves = moves.at[slot].set(move)
        action = build_joint_action_from_actor_actions(
            ActorAction(moves, no_combat, no_combat), manual_b
        )
        previous = state
        state, observation, _, _, mask, info = _STEP(
            config, state, mask, action, jax.random.key(tick)
        )
        if tick == 0:
            displacement = np.asarray(state.agent_positions - previous.agent_positions)
            assert np.all(displacement[list(slots), 1] < -0.1)
            np.testing.assert_allclose(displacement[list(slots), 0], 0.0, atol=1e-5)
        for submitted, accepted in zip(
            action,
            info.transition_facts.action_acceptance_facts.accepted_joint_action,
            strict=True,
        ):
            np.testing.assert_array_equal(submitted, accepted)
        xs = np.asarray(state.agent_positions)[list(slots), 0]
        cleared |= xs >= 7.0 - 1e-5 if from_west else xs <= 5.0 + 1e-5
        if bool(np.all(cleared)):
            break
    assert bool(np.all(cleared)), np.asarray(state.agent_positions)[list(slots)]
