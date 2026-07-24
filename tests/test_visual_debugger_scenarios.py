"""Exact public-trajectory integration tests for every debugger scenario."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.control import (
    create_session,
    submit_next_script_frame,
)
from scripts.dev.visual_debugger.diagnostics import derive_selected_target_facts
from scripts.dev.visual_debugger.model import (
    ActorCommand,
    DebuggerScenario,
    DebuggerSession,
)
from scripts.dev.visual_debugger.scenarios import (
    SCENARIOS,
    get_scenario,
    list_scenarios,
)

from marl_battlegrounds.core.config import validate_env_config
from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_STAY,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    SLOW_CHANNEL_HUNTER_BASIC,
    SLOW_CHANNEL_ROGUE_POISON,
    SLOW_CHANNEL_WARRIOR_CHARGE,
    STUN_CHANNEL_HUNTER_TRAP,
    STUN_CHANNEL_ROGUE_POISON,
    STUN_CHANNEL_WARRIOR_CHARGE,
    WARRIOR_CLASS_ID,
)


def _session(name: str) -> tuple[DebuggerScenario, DebuggerSession]:
    scenario = get_scenario(name)
    return scenario, create_session(
        scenario,
        seed=0,
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )


def test_all_scenario_configs_validate_and_reset() -> None:
    assert tuple(SCENARIOS) == (
        "arena_5v5",
        "acceptance_lane_lab",
        "basic_support",
        "ultimate_showcase",
        "aura_crossfire",
        "status_stack",
    )
    assert len(list_scenarios()) == 6
    for scenario in list_scenarios():
        config = scenario.build_config()
        validate_env_config(config)
        state, observation, action_mask, _ = reset(config, jax.random.key(17))

        assert config.max_steps == 300
        assert config.ordinary_movement_distance_scale == 1.0
        assert int(state.step_count) == 0
        assert not bool(state.has_previous_timestep_joint_action)
        assert observation.self_features.shape[0] == MAX_AGENT_SLOTS
        assert action_mask.select_target_use_ultimate_joint_mask.shape == (
            MAX_AGENT_SLOTS,
            11,
            2,
        )
        assert bool(
            jnp.array_equal(
                state.current_health,
                config.agent_profile.max_health,
            )
        )


def test_scenario_registry_exact_maps_rosters_positions_modes_and_frames() -> None:
    expected_configs = {
        "arena_5v5": (
            (18.0, 12.0),
            tuple(range(10)),
            (
                MAGE_CLASS_ID,
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
                ROGUE_CLASS_ID,
                PRIEST_CLASS_ID,
                MAGE_CLASS_ID,
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
                ROGUE_CLASS_ID,
                PRIEST_CLASS_ID,
            ),
            (
                (3.0, 2.0),
                (3.0, 4.0),
                (3.0, 6.0),
                (3.0, 8.0),
                (3.0, 10.0),
                (15.0, 10.0),
                (15.0, 8.0),
                (15.0, 6.0),
                (15.0, 4.0),
                (15.0, 2.0),
            ),
            "interactive",
            0,
        ),
        "acceptance_lane_lab": (
            (16.0, 12.0),
            (0, 5),
            (HUNTER_CLASS_ID, MAGE_CLASS_ID),
            ((3.0, 6.0), (12.0, 6.0)),
            "interactive",
            0,
        ),
        "basic_support": (
            (14.0, 12.0),
            (0, 1, 2, 5, 6, 7),
            (
                MAGE_CLASS_ID,
                HUNTER_CLASS_ID,
                PRIEST_CLASS_ID,
                MAGE_CLASS_ID,
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
            ),
            (
                (4.0, 3.0),
                (4.0, 6.0),
                (4.0, 9.0),
                (9.0, 3.0),
                (9.0, 6.0),
                (9.0, 9.0),
            ),
            "scripted",
            0,
        ),
        "ultimate_showcase": (
            (16.0, 12.0),
            tuple(range(10)),
            (
                MAGE_CLASS_ID,
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
                ROGUE_CLASS_ID,
                PRIEST_CLASS_ID,
                MAGE_CLASS_ID,
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
                ROGUE_CLASS_ID,
                PRIEST_CLASS_ID,
            ),
            (
                (3.0, 2.0),
                (5.0, 5.0),
                (5.0, 8.0),
                (8.0, 5.0),
                (3.0, 10.0),
                (9.0, 5.0),
                (9.0, 8.0),
                (10.0, 3.0),
                (12.0, 8.0),
                (13.0, 10.0),
            ),
            "scripted",
            0,
        ),
        "aura_crossfire": (
            (14.0, 12.0),
            (0, 1, 2, 5, 6, 7),
            (
                MAGE_CLASS_ID,
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
                MAGE_CLASS_ID,
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
            ),
            (
                (4.0, 5.0),
                (4.0, 7.0),
                (5.5, 6.0),
                (10.0, 5.0),
                (10.0, 7.0),
                (8.5, 6.0),
            ),
            "scripted",
            2,
        ),
        "status_stack": (
            (14.0, 12.0),
            (0, 1, 2, 5, 6),
            (
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
                ROGUE_CLASS_ID,
                HUNTER_CLASS_ID,
                PRIEST_CLASS_ID,
            ),
            (
                (3.0, 6.0),
                (5.0, 4.0),
                (8.0, 5.0),
                (8.0, 6.0),
                (8.0, 8.0),
            ),
            "scripted",
            5,
        ),
    }
    for name, (
        map_size,
        active_slots,
        class_ids,
        positions,
        mode,
        controlled_slot,
    ) in expected_configs.items():
        scenario = get_scenario(name)
        config = scenario.build_config()
        observed_active = tuple(
            int(slot)
            for slot in np.flatnonzero(
                np.asarray(config.agent_profile.active_mask, dtype=bool)
            )
        )
        assert (config.map_width, config.map_height) == map_size
        assert observed_active == active_slots
        assert (
            tuple(int(config.agent_profile.class_ids[slot]) for slot in active_slots)
            == class_ids
        )
        np.testing.assert_allclose(
            np.asarray(config.initial_agent_positions)[list(active_slots)],
            positions,
        )
        assert scenario.mode == mode
        assert scenario.default_controlled_slot == controlled_slot

    expected_frames = {
        "arena_5v5": (),
        "acceptance_lane_lab": (
            ((0, MOVE_EAST, 5, 0),),
            ((0, MOVE_EAST, 5, 0),),
            ((0, MOVE_EAST, 5, 0),),
            ((0, MOVE_EAST, 5, 0),),
            ((0, MOVE_EAST, 5, 1),),
            ((0, MOVE_STAY, 5, 1),),
        ),
        "basic_support": (
            (
                (0, MOVE_STAY, 5, 0),
                (1, MOVE_STAY, 6, 0),
                (7, MOVE_STAY, 2, 0),
            ),
            ((2, MOVE_STAY, 2, 0), (7, MOVE_STAY, 2, 0)),
        ),
        "ultimate_showcase": (
            ((5, MOVE_STAY, 2, 0),),
            (
                (0, MOVE_STAY, None, 1),
                (1, MOVE_STAY, 7, 1),
                (2, MOVE_STAY, 6, 1),
                (3, MOVE_STAY, 5, 1),
                (4, MOVE_STAY, 2, 1),
            ),
            ((2, MOVE_STAY, 6, 0),),
        ),
        "aura_crossfire": (((2, MOVE_STAY, 7, 0), (7, MOVE_STAY, 2, 0)),),
        "status_stack": (
            (
                (0, MOVE_NORTH, 5, 1),
                (1, MOVE_STAY, 5, 1),
                (2, MOVE_STAY, 5, 1),
                (6, MOVE_STAY, 5, 0),
            ),
            (
                (1, MOVE_STAY, 5, 0),
                (5, MOVE_EAST, None, 0),
                (6, MOVE_STAY, 5, 0),
            ),
            ((5, MOVE_EAST, None, 0),),
            ((5, MOVE_EAST, None, 0),),
        ),
    }
    for name, expected in expected_frames.items():
        observed = tuple(
            tuple(
                (
                    command.actor_global_slot,
                    command.move_action,
                    command.target_global_slot,
                    command.use_ultimate,
                )
                for command in frame.commands
            )
            for frame in get_scenario(name).frames
        )
        assert observed == expected

    # Keep the constructor contract visible in this registry-level test.
    assert ActorCommand(0) == ActorCommand(0, MOVE_STAY, None, 0)


def test_every_scenario_frame_has_unique_active_actors_and_targets() -> None:
    for scenario in list_scenarios():
        config = scenario.build_config()
        active = np.asarray(config.agent_profile.active_mask, dtype=bool)
        for frame in scenario.frames:
            actor_slots = [command.actor_global_slot for command in frame.commands]
            assert len(actor_slots) == len(set(actor_slots))
            for command in frame.commands:
                assert active[command.actor_global_slot]
                if command.target_global_slot is not None:
                    assert active[command.target_global_slot]


def test_arena_5v5_exact_map_roster_positions_obstacles_and_initial_facts() -> None:
    scenario, session = _session("arena_5v5")
    config = session.config
    expected_positions = np.asarray(
        (
            (3, 2),
            (3, 4),
            (3, 6),
            (3, 8),
            (3, 10),
            (15, 10),
            (15, 8),
            (15, 6),
            (15, 4),
            (15, 2),
        ),
        dtype=np.float32,
    )

    assert scenario.mode == "interactive"
    assert scenario.frames == ()
    assert scenario.default_controlled_slot == 0
    assert (config.map_width, config.map_height) == (18.0, 12.0)
    np.testing.assert_array_equal(session.state.agent_positions, expected_positions)
    np.testing.assert_array_equal(
        session.state.current_health,
        (80, 200, 100, 100, 100, 80, 200, 100, 100, 100),
    )
    first, second = np.asarray(config.obstacles[:2])
    assert int(first[OBSTACLE_FEATURE_TYPE]) == OBSTACLE_TYPE_PILLAR
    assert float(first[OBSTACLE_FEATURE_X]) == 9.0
    assert float(first[OBSTACLE_FEATURE_Y]) == 3.0
    assert float(first[OBSTACLE_FEATURE_RADIUS]) == pytest.approx(0.9)
    assert float(first[OBSTACLE_FEATURE_ACTIVE]) == 1.0
    assert int(second[OBSTACLE_FEATURE_TYPE]) == OBSTACLE_TYPE_WALL
    assert float(second[OBSTACLE_FEATURE_X]) == 9.0
    assert float(second[OBSTACLE_FEATURE_Y]) == pytest.approx(7.8)
    assert float(second[OBSTACLE_FEATURE_WIDTH]) == 3.0
    assert float(second[OBSTACLE_FEATURE_HEIGHT]) == 0.5
    assert float(second[OBSTACLE_FEATURE_THETA]) == pytest.approx(0.45)

    mage_auras = np.asarray(
        session.observation.self_features[
            :,
            AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
        ]
    )
    warrior_auras = np.asarray(
        session.observation.self_features[
            :,
            AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
        ]
    )
    np.testing.assert_allclose(mage_auras[[0, 1, 5, 6]], 1.15)
    np.testing.assert_allclose(warrior_auras[[0, 1, 2, 5, 6, 7]], 0.85)
    np.testing.assert_array_equal(
        session.action_mask.select_target_use_ultimate_joint_mask[0, 0],
        (True, True),
    )
    np.testing.assert_array_equal(
        session.action_mask.select_target_use_ultimate_joint_mask[5, 0],
        (True, True),
    )
    for target_action, expected in {
        2: (False, True),
        3: (True, True),
        4: (True, True),
        5: (True, True),
    }.items():
        np.testing.assert_array_equal(
            session.action_mask.select_target_use_ultimate_joint_mask[
                4,
                target_action,
            ],
            expected,
        )


def test_acceptance_lane_lab_geometry_visibility_and_mask_trajectory() -> None:
    scenario, session = _session("acceptance_lane_lab")
    expected = (
        (9.0, True, False, False, False, False, False, False),
        (8.0, True, False, False, False, False, False, False),
        (7.0, True, False, False, False, False, False, False),
        (6.0, True, True, True, False, False, False, False),
        (5.0, True, True, True, True, False, True, False),
        (4.0, True, True, True, True, True, True, True),
    )
    for frame_index, facts_expected in enumerate(expected):
        facts = derive_selected_target_facts(
            config=session.config,
            state=session.state,
            observation=session.observation,
            action_mask=session.action_mask,
            controlled_global_slot=0,
            target_global_slot=5,
        )
        assert facts is not None
        (
            distance,
            los,
            visible,
            observation_range,
            basic_range,
            ultimate_range,
            lane_0,
            lane_1,
        ) = facts_expected
        assert facts.center_distance == pytest.approx(distance)
        assert facts.has_clear_line_of_sight is los
        assert facts.observer_visible is visible
        assert facts.inside_observation_radius is observation_range
        assert facts.inside_basic_radius is basic_range
        assert facts.inside_ultimate_radius is ultimate_range
        assert facts.lane_0_available is lane_0
        assert facts.lane_1_available is lane_1
        if frame_index < len(scenario.frames):
            session = submit_next_script_frame(session)


def test_acceptance_lane_lab_reference_trajectory() -> None:
    _, session = _session("acceptance_lane_lab")
    for expected_step, expected_x in enumerate((4, 5, 6, 7, 8), start=1):
        session = submit_next_script_frame(session)
        transition = session.last_transition
        assert transition is not None
        actor = transition.actor_transitions[0]
        assert int(session.state.step_count) == expected_step
        assert float(session.state.agent_positions[0, 0]) == pytest.approx(expected_x)
        assert actor.movement_accepted
        assert not actor.combat_pair_accepted
        assert actor.accepted_target_action == 0
        assert actor.accepted_use_ultimate == 0
        np.testing.assert_array_equal(
            np.asarray(session.state.current_health)[[0, 5]],
            (100.0, 80.0),
        )

    session = submit_next_script_frame(session)
    transition = session.last_transition
    assert transition is not None
    actor = transition.actor_transitions[0]
    assert actor.movement_accepted and actor.combat_pair_accepted
    assert (actor.accepted_target_action, actor.accepted_use_ultimate) == (6, 1)
    assert int(session.state.ultimate_cooldowns[0]) == 30
    assert int(session.state.stun_durations[5, STUN_CHANNEL_HUNTER_TRAP]) == 4
    assert (
        float(
            session.observation.self_features[
                5,
                AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
            ]
        )
        == 0.0
    )


def test_basic_support_reference_trajectory() -> None:
    _, session = _session("basic_support")
    session = submit_next_script_frame(session)
    np.testing.assert_allclose(
        np.asarray(session.state.current_health)[[2, 5, 6]],
        (92.0, 66.2, 193.2),
        atol=1e-5,
    )
    assert int(session.state.slow_durations[2, SLOW_CHANNEL_HUNTER_BASIC]) == 1
    assert int(session.state.slow_durations[6, SLOW_CHANNEL_HUNTER_BASIC]) == 1
    assert bool(jnp.all(session.state.ultimate_cooldowns == 0))

    session = submit_next_script_frame(session)
    assert float(session.state.current_health[2]) == pytest.approx(92.0)
    assert int(session.state.slow_durations[2, SLOW_CHANNEL_HUNTER_BASIC]) == 1
    assert int(session.state.slow_durations[6, SLOW_CHANNEL_HUNTER_BASIC]) == 0
    assert int(session.state.priest_blessing_of_freedom_slow_floor_durations[2]) == 1
    transition = session.last_transition
    assert transition is not None
    kinds = {activation.kind for activation in transition.accepted_activations}
    assert kinds == {"basic_heal", "basic_damage"}


def test_ultimate_showcase_reference_trajectory() -> None:
    _, session = _session("ultimate_showcase")
    session = submit_next_script_frame(session)
    assert float(session.state.current_health[2]) == pytest.approx(86.2)

    session = submit_next_script_frame(session)
    np.testing.assert_allclose(
        session.state.agent_positions[1],
        (9.0715, 3.3714),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(session.state.current_health)[[2, 7]],
        (100.0, 84.0),
        atol=1e-5,
    )
    np.testing.assert_array_equal(session.state.ultimate_cooldowns[:5], 30)
    assert int(session.state.mage_burst_damage_amplification_durations[0]) == 5
    assert int(session.state.slow_durations[7, SLOW_CHANNEL_WARRIOR_CHARGE]) == 5
    assert int(session.state.stun_durations[7, STUN_CHANNEL_WARRIOR_CHARGE]) == 1
    assert int(session.state.stun_durations[6, STUN_CHANNEL_HUNTER_TRAP]) == 4
    assert int(session.state.slow_durations[5, SLOW_CHANNEL_ROGUE_POISON]) == 5
    assert int(session.state.stun_durations[5, STUN_CHANNEL_ROGUE_POISON]) == 1
    assert int(session.state.rogue_poison_anti_heal_durations[5]) == 4
    transition = session.last_transition
    assert transition is not None
    assert {activation.kind for activation in transition.accepted_activations} == {
        "mage_burst",
        "warrior_charge",
        "hunter_trap",
        "rogue_poison",
        "holy_word",
    }

    session = submit_next_script_frame(session)
    assert float(session.state.current_health[6]) == pytest.approx(193.2)
    assert int(session.state.stun_durations[6, STUN_CHANNEL_HUNTER_TRAP]) == 0
    assert int(session.state.slow_durations[6, SLOW_CHANNEL_HUNTER_BASIC]) == 1
    np.testing.assert_array_equal(session.state.ultimate_cooldowns[:5], 29)
    assert int(session.state.mage_burst_damage_amplification_durations[0]) == 4
    assert int(session.state.slow_durations[7, SLOW_CHANNEL_WARRIOR_CHARGE]) == 4
    assert int(session.state.stun_durations[7, STUN_CHANNEL_WARRIOR_CHARGE]) == 0
    assert int(session.state.slow_durations[5, SLOW_CHANNEL_ROGUE_POISON]) == 4
    assert int(session.state.stun_durations[5, STUN_CHANNEL_ROGUE_POISON]) == 0
    assert int(session.state.rogue_poison_anti_heal_durations[5]) == 3
    transition = session.last_transition
    assert transition is not None
    trap = next(
        status
        for status in transition.status_transitions
        if status.global_slot == 6 and status.status_kind == "stun_hunter_trap"
    )
    assert trap.change == "trap_broken"


def test_aura_crossfire_reference_trajectory() -> None:
    _, session = _session("aura_crossfire")
    self_features = np.asarray(session.observation.self_features)
    np.testing.assert_allclose(
        self_features[
            [0, 1, 2, 5, 6, 7],
            AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
        ],
        1.15,
    )
    np.testing.assert_allclose(
        self_features[
            [0, 1, 2, 5, 6, 7],
            AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
        ],
        0.85,
    )
    session = submit_next_script_frame(session)
    np.testing.assert_allclose(
        np.asarray(session.state.current_health)[[2, 7]],
        (92.18, 92.18),
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        np.asarray(session.state.slow_durations)[
            [2, 7],
            SLOW_CHANNEL_HUNTER_BASIC,
        ],
        (1, 1),
    )


def test_status_stack_reference_trajectory() -> None:
    _, session = _session("status_stack")
    session = submit_next_script_frame(session)
    np.testing.assert_allclose(session.state.agent_positions[0], (7.0, 7.0))
    assert float(session.state.current_health[5]) == pytest.approx(92.0)
    np.testing.assert_array_equal(session.state.slow_durations[5], (5, 0, 5))
    np.testing.assert_array_equal(session.state.stun_durations[5], (1, 4, 1))
    assert int(session.state.rogue_poison_anti_heal_durations[5]) == 4
    assert int(session.state.priest_blessing_of_freedom_slow_floor_durations[5]) == 1
    np.testing.assert_array_equal(session.state.ultimate_cooldowns[:3], 30)
    assert (
        float(
            session.observation.self_features[
                5,
                AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
            ]
        )
        == 0.0
    )

    session = submit_next_script_frame(session)
    np.testing.assert_allclose(session.state.agent_positions[5], (8.0, 6.0))
    assert float(session.state.current_health[5]) == pytest.approx(88.0)
    np.testing.assert_array_equal(session.state.slow_durations[5], (4, 1, 4))
    np.testing.assert_array_equal(session.state.stun_durations[5], (0, 0, 0))
    assert int(session.state.rogue_poison_anti_heal_durations[5]) == 3
    assert int(session.state.priest_blessing_of_freedom_slow_floor_durations[5]) == 1
    np.testing.assert_array_equal(session.state.ultimate_cooldowns[:3], 29)
    assert float(
        session.observation.self_features[
            5,
            AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
        ]
    ) == pytest.approx(0.85)

    session = submit_next_script_frame(session)
    np.testing.assert_allclose(
        session.state.agent_positions[5],
        (8.85, 6.0),
        atol=1e-5,
    )
    np.testing.assert_array_equal(session.state.slow_durations[5], (3, 0, 3))
    assert int(session.state.rogue_poison_anti_heal_durations[5]) == 2
    assert int(session.state.priest_blessing_of_freedom_slow_floor_durations[5]) == 0
    np.testing.assert_array_equal(session.state.ultimate_cooldowns[:3], 28)
    assert float(
        session.observation.self_features[
            5,
            AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
        ]
    ) == pytest.approx(0.25)

    session = submit_next_script_frame(session)
    np.testing.assert_allclose(
        session.state.agent_positions[5],
        (9.10, 6.0),
        atol=1e-5,
    )
    np.testing.assert_array_equal(session.state.slow_durations[5], (2, 0, 2))
    assert int(session.state.rogue_poison_anti_heal_durations[5]) == 1
    np.testing.assert_array_equal(session.state.ultimate_cooldowns[:3], 27)
