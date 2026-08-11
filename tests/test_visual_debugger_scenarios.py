"""Exact public-trajectory integration tests for every debugger scenario."""

import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.control import (
    build_scripted_joint_action,
    create_session,
    submit_next_script_frame,
)
from scripts.dev.visual_debugger.model import (
    ActorCommand,
    DebuggerScenario,
    DebuggerSession,
)
from scripts.dev.visual_debugger.scenarios import (
    RESEARCHER_SCENARIOS,
    SCENARIOS,
    STRESS_SCENARIOS,
    cycle_scenario_name,
    get_scenario,
    iter_scenario_summaries,
    list_scenarios,
)
from scripts.dev.visual_debugger.targeting import global_slot_to_target_action
from tests.visual_debugger_fixtures import (
    debugger_test_launch_specification,
    rejection_lane_scenario,
    submit_fixture_frame,
)

from marl_battlegrounds.core.config import validate_env_config
from marl_battlegrounds.core.env import initialize_scenario_state
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_STAY,
    MOVE_WEST,
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
from marl_battlegrounds.evaluation.models import (
    AbilityActivatedEventV1,
    RecipientHealthResolutionEventV1,
    StatusLifecycleEventBaseV1,
)


def _session(name: str) -> tuple[DebuggerScenario, DebuggerSession]:
    scenario = get_scenario(name)
    return scenario, create_session(
        scenario,
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )


def _canonical_ability_signatures(
    session: DebuggerSession,
) -> tuple[tuple[str, int, int | None], ...]:
    view = session.incoming_evaluation_view
    assert view is not None
    return tuple(
        (
            event.ability_component,
            event.source_global_slot,
            event.recipient_global_slot,
        )
        for event in view.transition.events
        if isinstance(event, AbilityActivatedEventV1)
    )


def _canonical_status_event_types(
    session: DebuggerSession,
    *,
    global_slot: int,
    status_id: str,
) -> tuple[str, ...]:
    view = session.incoming_evaluation_view
    assert view is not None
    return tuple(
        event.event_type
        for event in view.transition.events
        if isinstance(event, StatusLifecycleEventBaseV1)
        and event.recipient_global_slot == global_slot
        and event.status_id == status_id
    )


def test_all_scenario_configs_validate_and_initialize_authored_state() -> None:
    researcher_names = (
        "arena_5v5",
        "basic_support",
        "ultimate_showcase",
        "aura_crossfire",
        "status_stack",
        "team_focus_crossfire",
        "mirrored_ultimates",
    )
    stress_names = (
        "moving_basic_crossfire",
        "moving_focus_crossfire",
        "charge_convergence",
        "trap_lifecycle",
        "max_status_stack",
    )
    assert tuple(RESEARCHER_SCENARIOS) == researcher_names
    assert tuple(STRESS_SCENARIOS) == stress_names
    assert set(RESEARCHER_SCENARIOS).isdisjoint(STRESS_SCENARIOS)
    assert tuple(SCENARIOS) == researcher_names + stress_names
    assert tuple(scenario.name for scenario in list_scenarios()) == researcher_names
    assert (
        tuple(scenario.name for scenario in list_scenarios(include_stress=True))
        == researcher_names + stress_names
    )
    assert all(scenario.audience == "researcher" for scenario in list_scenarios())
    assert all(scenario.audience == "stress" for scenario in STRESS_SCENARIOS.values())
    assert tuple(iter_scenario_summaries()) == tuple(
        f"{scenario.name:<22} {scenario.mode:<11} {scenario.description}"
        for scenario in RESEARCHER_SCENARIOS.values()
    )
    assert tuple(iter_scenario_summaries(include_stress=True))[
        -len(stress_names) :
    ] == tuple(
        f"{scenario.name:<22} {scenario.mode:<11} {scenario.description}"
        for scenario in STRESS_SCENARIOS.values()
    )
    assert cycle_scenario_name("mirrored_ultimates", 1) == "arena_5v5"
    assert cycle_scenario_name("charge_convergence", 1) == "trap_lifecycle"
    assert (
        cycle_scenario_name("charge_convergence", 1, include_stress=True)
        == "trap_lifecycle"
    )

    for scenario in list_scenarios(include_stress=True):
        config, authored_state = scenario.build_scenario()
        validate_env_config(config)
        state, observation, action_mask, info = initialize_scenario_state(
            authored_state,
            config,
        )

        assert config.max_steps == 300
        assert config.ordinary_movement_distance_scale == 1.0
        assert config.spawn_shield_duration_steps == 3
        assert config.spawn_shield_movement_speed == 2.0
        assert state is authored_state
        assert int(state.step_count) == 0
        assert not bool(state.has_previous_timestep_joint_action)
        assert observation.self_features.shape[0] == MAX_AGENT_SLOTS
        assert action_mask.select_target_use_ultimate_joint_mask.shape == (
            MAX_AGENT_SLOTS,
            11,
            2,
        )
        shield_facts = info.transition_facts.spawn_shield_facts
        assert not bool(jnp.any(shield_facts.was_active_at_transition_start_by_agent))
        assert not bool(jnp.any(shield_facts.expired_at_transition_end_by_agent))
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
                (7.0, 3.0),
                (7.0, 6.0),
                (7.0, 9.0),
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
                (7.0, 6.0),
                (8.0, 8.0),
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
                (5.5, 4.4),
                (8.0, 5.0),
                (8.0, 6.0),
                (8.0, 8.0),
            ),
            "scripted",
            5,
        ),
        "team_focus_crossfire": (
            (16.0, 12.0),
            (0, 1, 2, 3, 5, 6, 7, 8),
            (
                MAGE_CLASS_ID,
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
                ROGUE_CLASS_ID,
                WARRIOR_CLASS_ID,
                PRIEST_CLASS_ID,
                PRIEST_CLASS_ID,
                PRIEST_CLASS_ID,
            ),
            (
                (6.0, 6.0),
                (7.0, 5.0),
                (8.0, 3.0),
                (8.0, 4.6),
                (8.0, 6.0),
                (10.5, 6.0),
                (9.8, 8.0),
                (6.2, 8.0),
            ),
            "scripted",
            2,
        ),
        "mirrored_ultimates": (
            (18.0, 14.0),
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
                (6.0, 5.0),
                (6.6, 9.0),
                (7.3, 12.0),
                (4.0, 11.5),
                (15.0, 2.0),
                (10.0, 5.0),
                (9.4, 9.0),
                (8.7, 12.0),
                (12.0, 11.5),
            ),
            "scripted",
            0,
        ),
        "moving_basic_crossfire": (
            (14.0, 12.0),
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
                (5.0, 4.0),
                (5.0, 8.5),
                (5.0, 6.0),
                (8.7, 8.5),
                (3.0, 4.5),
                (7.0, 6.0),
                (6.3, 8.5),
                (7.0, 4.0),
                (10.0, 8.5),
                (9.0, 5.5),
            ),
            "scripted",
            0,
        ),
        "moving_focus_crossfire": (
            (16.0, 12.0),
            (0, 1, 2, 3, 5, 6, 7, 8),
            (
                MAGE_CLASS_ID,
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
                ROGUE_CLASS_ID,
                WARRIOR_CLASS_ID,
                PRIEST_CLASS_ID,
                PRIEST_CLASS_ID,
                PRIEST_CLASS_ID,
            ),
            (
                (6.02, 4.02),
                (6.52, 6.0),
                (6.02, 7.98),
                (9.48, 6.0),
                (8.0, 6.0),
                (9.98, 4.02),
                (9.98, 7.98),
                (8.0, 8.8),
            ),
            "scripted",
            2,
        ),
        "charge_convergence": (
            (14.0, 12.0),
            (0, 1, 5),
            (WARRIOR_CLASS_ID, WARRIOR_CLASS_ID, WARRIOR_CLASS_ID),
            ((3.0, 4.0), (3.0, 8.0), (8.0, 6.0)),
            "scripted",
            0,
        ),
        "trap_lifecycle": (
            (12.0, 12.0),
            tuple(range(10)),
            (
                HUNTER_CLASS_ID,
                HUNTER_CLASS_ID,
                HUNTER_CLASS_ID,
                HUNTER_CLASS_ID,
                HUNTER_CLASS_ID,
                WARRIOR_CLASS_ID,
                WARRIOR_CLASS_ID,
                WARRIOR_CLASS_ID,
                WARRIOR_CLASS_ID,
                WARRIOR_CLASS_ID,
            ),
            (
                (3.0, 2.0),
                (3.0, 4.0),
                (3.0, 6.0),
                (3.0, 8.0),
                (4.5, 3.0),
                (6.0, 2.0),
                (6.0, 4.0),
                (6.0, 6.0),
                (6.0, 8.0),
                (6.0, 10.0),
            ),
            "scripted",
            0,
        ),
        "max_status_stack": (
            (16.0, 12.0),
            (0, 1, 5, 6, 7, 8),
            (
                MAGE_CLASS_ID,
                PRIEST_CLASS_ID,
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
                HUNTER_CLASS_ID,
                ROGUE_CLASS_ID,
            ),
            (
                (8.0, 6.0),
                (10.5, 7.0),
                (8.0, 1.0),
                (10.0, 4.0),
                (11.0, 6.0),
                (8.0, 7.4),
            ),
            "scripted",
            0,
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
        config, state = scenario.build_scenario()
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
            np.asarray(state.agent_positions)[list(active_slots)],
            positions,
        )
        assert scenario.mode == mode
        assert scenario.default_controlled_slot == controlled_slot
        assert scenario.audience == (
            "stress" if name in STRESS_SCENARIOS else "researcher"
        )

    expected_frames = {
        "arena_5v5": (),
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
                (5, MOVE_STAY, None, 0),
                (6, MOVE_STAY, 5, 0),
            ),
            ((5, MOVE_EAST, None, 0),),
            ((5, MOVE_EAST, None, 0),),
        ),
        "team_focus_crossfire": (
            ((2, MOVE_STAY, 5, 0),),
            ((2, MOVE_STAY, 5, 0),),
            (
                (0, MOVE_STAY, 5, 0),
                (1, MOVE_STAY, 5, 0),
                (2, MOVE_STAY, 5, 0),
                (3, MOVE_STAY, 5, 0),
                (6, MOVE_STAY, 5, 0),
                (7, MOVE_STAY, 5, 0),
                (8, MOVE_STAY, 5, 0),
            ),
            (
                (3, MOVE_STAY, 5, 1),
                (6, MOVE_STAY, 5, 0),
                (7, MOVE_STAY, 5, 0),
                (8, MOVE_STAY, 5, 0),
            ),
            (
                (6, MOVE_STAY, 5, 0),
                (7, MOVE_STAY, 5, 0),
                (8, MOVE_STAY, 5, 0),
            ),
            (
                (6, MOVE_STAY, 5, 1),
                (7, MOVE_STAY, 5, 1),
                (8, MOVE_STAY, 5, 1),
            ),
        ),
        "mirrored_ultimates": (
            ((0, MOVE_NORTH, None, 1), (5, MOVE_NORTH, None, 1)),
            ((1, MOVE_STAY, 6, 1), (6, MOVE_STAY, 1, 1)),
            ((2, MOVE_NORTH, 7, 1), (7, MOVE_NORTH, 2, 1)),
            ((3, MOVE_EAST, 8, 1), (8, MOVE_EAST, 3, 1)),
            ((4, MOVE_NORTH, 3, 1), (9, MOVE_NORTH, 8, 1)),
        ),
        "moving_basic_crossfire": (
            (
                (0, MOVE_NORTH, 5, 0),
                (1, MOVE_EAST, 6, 0),
                (2, MOVE_NORTH, 7, 0),
                (3, MOVE_EAST, 8, 0),
                (4, MOVE_NORTH, 0, 0),
                (5, MOVE_NORTH, 0, 0),
                (6, MOVE_EAST, 1, 0),
                (7, MOVE_NORTH, 2, 0),
                (8, MOVE_EAST, 3, 0),
                (9, MOVE_NORTH, 5, 0),
            ),
            (
                (0, MOVE_SOUTH, 5, 0),
                (1, MOVE_WEST, 6, 0),
                (2, MOVE_SOUTH, 7, 0),
                (3, MOVE_WEST, 8, 0),
                (4, MOVE_SOUTH, 0, 0),
                (5, MOVE_SOUTH, 0, 0),
                (6, MOVE_WEST, 1, 0),
                (7, MOVE_SOUTH, 2, 0),
                (8, MOVE_WEST, 3, 0),
                (9, MOVE_SOUTH, 5, 0),
            ),
        ),
        "moving_focus_crossfire": (
            (
                (0, MOVE_EAST, 5, 0),
                (1, MOVE_EAST, 5, 0),
                (2, MOVE_EAST, 5, 0),
                (3, MOVE_EAST, 5, 0),
                (5, MOVE_EAST, None, 0),
                (6, MOVE_EAST, 5, 0),
                (7, MOVE_EAST, 5, 0),
                (8, MOVE_EAST, 5, 0),
            ),
        ),
        "charge_convergence": (
            (
                (0, MOVE_STAY, 5, 1),
                (1, MOVE_STAY, 5, 1),
                (5, MOVE_STAY, 0, 1),
            ),
        ),
        "trap_lifecycle": (
            (
                (0, MOVE_STAY, 5, 1),
                (1, MOVE_STAY, 6, 1),
                (2, MOVE_STAY, 7, 1),
                (3, MOVE_STAY, 8, 1),
            ),
            ((0, MOVE_STAY, 5, 0),),
            (),
            ((4, MOVE_STAY, 6, 1),),
            ((2, MOVE_STAY, 7, 0),),
        ),
        "max_status_stack": (
            (
                (0, MOVE_STAY, None, 1),
                (1, MOVE_STAY, 0, 0),
                (5, MOVE_STAY, 0, 1),
                (6, MOVE_STAY, 0, 1),
                (7, MOVE_STAY, 0, 0),
                (8, MOVE_STAY, 0, 1),
            ),
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
    for scenario in list_scenarios(include_stress=True):
        config, _ = scenario.build_scenario()
        active = np.asarray(config.agent_profile.active_mask, dtype=bool)
        for frame in scenario.frames:
            actor_slots = [command.actor_global_slot for command in frame.commands]
            assert len(actor_slots) == len(set(actor_slots))
            for command in frame.commands:
                assert active[command.actor_global_slot]
                if command.target_global_slot is not None:
                    assert active[command.target_global_slot]


def test_every_registered_scripted_command_is_legal_and_accepted() -> None:
    for scenario in list_scenarios(include_stress=True):
        if not scenario.frames:
            continue
        _, session = _session(scenario.name)
        for frame in scenario.frames:
            step_before = int(session.state.step_count)
            action = build_scripted_joint_action(session.evaluation_context, frame)
            for command in frame.commands:
                slot = command.actor_global_slot
                target_action = global_slot_to_target_action(
                    slot,
                    command.target_global_slot,
                )
                assert 0 <= command.move_action < session.action_mask.move_mask.shape[1]
                assert (
                    0
                    <= target_action
                    < (
                        session.action_mask.select_target_use_ultimate_joint_mask.shape[
                            1
                        ]
                    )
                )
                assert command.use_ultimate in (0, 1)
                assert bool(session.action_mask.move_mask[slot, command.move_action])
                assert bool(
                    session.action_mask.select_target_use_ultimate_joint_mask[
                        slot,
                        target_action,
                        command.use_ultimate,
                    ]
                )

            submitted = submit_next_script_frame(session)
            view = submitted.incoming_evaluation_view
            assert view is not None
            acceptance = view.transition.facts.action_acceptance_facts
            assert int(submitted.state.step_count) == step_before + 1
            assert (
                submitted.next_script_frame_index == session.next_script_frame_index + 1
            )
            for retained_head, expected_head in zip(
                (
                    acceptance.submitted_joint_action.move,
                    acceptance.submitted_joint_action.select_target,
                    acceptance.submitted_joint_action.use_ultimate,
                ),
                action,
                strict=True,
            ):
                np.testing.assert_array_equal(retained_head, expected_head)
            for accepted_head, expected_head in zip(
                (
                    acceptance.accepted_joint_action.move,
                    acceptance.accepted_joint_action.select_target,
                    acceptance.accepted_joint_action.use_ultimate,
                ),
                action,
                strict=True,
            ):
                np.testing.assert_array_equal(accepted_head, expected_head)
            for command in frame.commands:
                actor_slot = command.actor_global_slot
                expected_target = int(action.select_target[actor_slot])
                assert acceptance.accepted_joint_action.move[actor_slot] == (
                    command.move_action
                )
                assert acceptance.accepted_joint_action.select_target[actor_slot] == (
                    expected_target
                )
                assert acceptance.accepted_joint_action.use_ultimate[actor_slot] == (
                    command.use_ultimate
                )
                assert not (
                    acceptance.submitted_action_tuple_is_out_of_domain_by_actor[
                        actor_slot
                    ]
                )
                assert not acceptance.in_domain_move_action_is_rejected_by_actor[
                    actor_slot
                ]
                assert not (
                    acceptance.in_domain_combat_action_pair_is_rejected_by_actor[
                        actor_slot
                    ]
                )
            session = submitted


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
        2: (False, False),
        3: (False, True),
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


def test_rejection_boundary_is_preserved_as_a_test_only_fixture() -> None:
    scenario = rejection_lane_scenario()
    session = create_session(
        scenario,
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    first, second = scenario.frames
    first_action = build_scripted_joint_action(session.evaluation_context, first)
    assert bool(session.action_mask.move_mask[0, MOVE_EAST])
    assert not bool(session.action_mask.select_target_use_ultimate_joint_mask[0, 6, 0])

    rejected = submit_fixture_frame(session, first)
    rejected_view = rejected.incoming_evaluation_view
    assert rejected_view is not None
    rejected_facts = rejected_view.transition.facts.action_acceptance_facts
    assert not rejected_facts.in_domain_move_action_is_rejected_by_actor[0]
    assert rejected_facts.in_domain_combat_action_pair_is_rejected_by_actor[0]
    assert int(first_action.select_target[0]) == 6
    assert (
        rejected_facts.accepted_joint_action.select_target[0],
        rejected_facts.accepted_joint_action.use_ultimate[0],
    ) == (0, 0)
    assert bool(rejected.action_mask.select_target_use_ultimate_joint_mask[0, 6, 0])

    accepted = submit_fixture_frame(rejected, second)
    accepted_view = accepted.incoming_evaluation_view
    assert accepted_view is not None
    accepted_facts = accepted_view.transition.facts.action_acceptance_facts
    assert not accepted_facts.in_domain_move_action_is_rejected_by_actor[0]
    assert not accepted_facts.in_domain_combat_action_pair_is_rejected_by_actor[0]
    assert (
        accepted_facts.accepted_joint_action.select_target[0],
        accepted_facts.accepted_joint_action.use_ultimate[0],
    ) == (6, 0)


def test_basic_support_reference_trajectory() -> None:
    _, session = _session("basic_support")
    session = submit_next_script_frame(session)
    np.testing.assert_allclose(
        np.asarray(session.state.current_health)[[2, 5, 6]],
        (94.0, 65.05, 194.9),
        atol=1e-5,
    )
    assert int(session.state.slow_durations[2, SLOW_CHANNEL_HUNTER_BASIC]) == 1
    assert int(session.state.slow_durations[6, SLOW_CHANNEL_HUNTER_BASIC]) == 1
    assert bool(jnp.all(session.state.ultimate_cooldowns == 0))

    session = submit_next_script_frame(session)
    assert float(session.state.current_health[2]) == pytest.approx(96.0)
    assert int(session.state.slow_durations[2, SLOW_CHANNEL_HUNTER_BASIC]) == 1
    assert int(session.state.slow_durations[6, SLOW_CHANNEL_HUNTER_BASIC]) == 0
    assert int(session.state.priest_blessing_of_freedom_slow_floor_durations[2]) == 1
    view = session.incoming_evaluation_view
    assert view is not None
    event_types = {event.event_type for event in view.transition.events}
    assert {"source_damage_output", "source_healing_output"}.issubset(event_types)


def test_ultimate_showcase_reference_trajectory() -> None:
    _, session = _session("ultimate_showcase")
    session = submit_next_script_frame(session)
    assert float(session.state.current_health[2]) == pytest.approx(85.05)

    session = submit_next_script_frame(session)
    np.testing.assert_allclose(
        session.state.agent_positions[1],
        (9.0715, 3.3714),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(session.state.current_health)[[2, 5, 6, 7]],
        (100.0, 44.0, 191.5, 80.0),
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
    assert _canonical_ability_signatures(session) == (
        ("ultimate", 0, None),
        ("ultimate", 1, 7),
        ("ultimate", 2, 6),
        ("ultimate", 3, 5),
        ("ultimate", 4, 2),
    )

    session = submit_next_script_frame(session)
    assert float(session.state.current_health[6]) == pytest.approx(186.4)
    assert int(session.state.stun_durations[6, STUN_CHANNEL_HUNTER_TRAP]) == 0
    assert int(session.state.slow_durations[6, SLOW_CHANNEL_HUNTER_BASIC]) == 1
    np.testing.assert_array_equal(session.state.ultimate_cooldowns[:5], 29)
    assert int(session.state.mage_burst_damage_amplification_durations[0]) == 4
    assert int(session.state.slow_durations[7, SLOW_CHANNEL_WARRIOR_CHARGE]) == 4
    assert int(session.state.stun_durations[7, STUN_CHANNEL_WARRIOR_CHARGE]) == 0
    assert int(session.state.slow_durations[5, SLOW_CHANNEL_ROGUE_POISON]) == 4
    assert int(session.state.stun_durations[5, STUN_CHANNEL_ROGUE_POISON]) == 0
    assert int(session.state.rogue_poison_anti_heal_durations[5]) == 3
    assert _canonical_status_event_types(
        session,
        global_slot=6,
        status_id="hunter_trap_stun",
    ) == ("status_broken_by_damage",)


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
        (94.135, 94.135),
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
    assert float(session.state.current_health[5]) == pytest.approx(42.0)
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
    assert float(session.state.current_health[5]) == pytest.approx(40.0)
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


def test_team_focus_crossfire_reference_trajectory() -> None:
    _, session = _session("team_focus_crossfire")

    session = submit_next_script_frame(session)
    first_view = session.incoming_evaluation_view
    assert first_view is not None
    assert _canonical_ability_signatures(session) == (("basic", 2, 5),)
    first_net = next(
        event.realized_net_health_change
        for event in first_view.transition.events
        if isinstance(event, RecipientHealthResolutionEventV1)
        and event.recipient_global_slot == 5
    )
    assert first_net == pytest.approx(
        first_view.successor_frame.snapshot.current_health[5]
        - first_view.start_frame.snapshot.current_health[5]
    )

    session = submit_next_script_frame(session)
    assert _canonical_ability_signatures(session) == (("basic", 2, 5),)
    second_view = session.incoming_evaluation_view
    assert second_view is not None
    assert second_view.successor_frame.simulator_step_count == (
        first_view.successor_frame.simulator_step_count + 1
    )

    session = submit_next_script_frame(session)
    assert _canonical_ability_signatures(session) == (
        ("basic", 0, 5),
        ("basic", 1, 5),
        ("basic", 2, 5),
        ("basic", 3, 5),
        ("basic", 6, 5),
        ("basic", 7, 5),
        ("basic", 8, 5),
    )

    session = submit_next_script_frame(session)
    poison_view = session.incoming_evaluation_view
    assert poison_view is not None
    assert poison_view.start_frame.snapshot.rogue_poison_anti_heal_durations[5] == 0
    assert poison_view.successor_frame.snapshot.rogue_poison_anti_heal_durations[5] > 0
    assert _canonical_ability_signatures(session) == (
        ("ultimate", 3, 5),
        ("basic", 6, 5),
        ("basic", 7, 5),
        ("basic", 8, 5),
    )

    session = submit_next_script_frame(session)
    anti_heal_view = session.incoming_evaluation_view
    assert anti_heal_view is not None
    assert anti_heal_view.start_frame.snapshot.rogue_poison_anti_heal_durations[5] > 0
    assert _canonical_ability_signatures(session) == (
        ("basic", 6, 5),
        ("basic", 7, 5),
        ("basic", 8, 5),
    )

    assert float(session.state.current_health[5]) == pytest.approx(183.3725)
    health_before_holy_words = float(session.state.current_health[5])
    session = submit_next_script_frame(session)
    assert _canonical_ability_signatures(session) == (
        ("ultimate", 6, 5),
        ("ultimate", 7, 5),
        ("ultimate", 8, 5),
    )
    final_view = session.incoming_evaluation_view
    assert final_view is not None
    warrior = next(
        event
        for event in final_view.transition.events
        if isinstance(event, RecipientHealthResolutionEventV1)
        and event.recipient_global_slot == 5
    )
    assert warrior.transition_start_health == pytest.approx(health_before_holy_words)
    assert warrior.health_after_combat_resolution == pytest.approx(
        float(session.config.agent_profile.max_health[5])
    )
    assert warrior.realized_net_health_change == pytest.approx(
        warrior.health_after_combat_resolution - health_before_holy_words
    )


def test_mirrored_ultimates_reference_trajectory() -> None:
    _, session = _session("mirrored_ultimates")
    expected = (
        (("ultimate", 0, None), ("ultimate", 5, None)),
        (("ultimate", 1, 6), ("ultimate", 6, 1)),
        (("ultimate", 2, 7), ("ultimate", 7, 2)),
        (("ultimate", 3, 8), ("ultimate", 8, 3)),
        (("ultimate", 4, 3), ("ultimate", 9, 8)),
    )

    for frame_index, expected_signatures in enumerate(expected):
        session = submit_next_script_frame(session)
        assert _canonical_ability_signatures(session) == expected_signatures
        assert int(session.state.step_count) == frame_index + 1

    assert int(session.state.mage_burst_damage_amplification_durations[0]) > 0
    assert int(session.state.mage_burst_damage_amplification_durations[5]) > 0
    assert int(session.state.stun_durations[7, STUN_CHANNEL_HUNTER_TRAP]) > 0
    assert int(session.state.stun_durations[2, STUN_CHANNEL_HUNTER_TRAP]) > 0
    assert int(session.state.rogue_poison_anti_heal_durations[3]) > 0
    assert int(session.state.rogue_poison_anti_heal_durations[8]) > 0


def test_moving_basic_crossfire_reference_trajectory() -> None:
    _, session = _session("moving_basic_crossfire")
    expected = (
        ("basic", 0, 5),
        ("basic", 1, 6),
        ("basic", 2, 7),
        ("basic", 3, 8),
        ("basic", 4, 0),
        ("basic", 5, 0),
        ("basic", 6, 1),
        ("basic", 7, 2),
        ("basic", 8, 3),
        ("basic", 9, 5),
    )

    for _ in get_scenario("moving_basic_crossfire").frames:
        before = np.asarray(session.state.agent_positions).copy()
        session = submit_next_script_frame(session)
        assert _canonical_ability_signatures(session) == expected
        after = np.asarray(session.state.agent_positions)
        assert all(
            not np.array_equal(before[global_slot], after[global_slot])
            for global_slot in range(MAX_AGENT_SLOTS)
        )


def test_moving_focus_crossfire_reference_trajectory() -> None:
    _, session = _session("moving_focus_crossfire")
    involved_slots = (0, 1, 2, 3, 5, 6, 7, 8)
    before = np.asarray(session.state.agent_positions).copy()
    session = submit_next_script_frame(session)

    assert _canonical_ability_signatures(session) == (
        ("basic", 0, 5),
        ("basic", 1, 5),
        ("basic", 2, 5),
        ("basic", 3, 5),
        ("basic", 6, 5),
        ("basic", 7, 5),
        ("basic", 8, 5),
    )
    after = np.asarray(session.state.agent_positions)
    assert all(
        not np.array_equal(before[global_slot], after[global_slot])
        for global_slot in involved_slots
    )
    pair_distances = tuple(
        float(np.linalg.norm(after[left] - after[right]))
        for index, left in enumerate(involved_slots)
        for right in involved_slots[index + 1 :]
    )
    assert min(pair_distances) > 1.25


def test_charge_convergence_reference_trajectory() -> None:
    _, session = _session("charge_convergence")
    before_positions = np.asarray(session.state.agent_positions).copy()
    session = submit_next_script_frame(session)

    assert _canonical_ability_signatures(session) == (
        ("ultimate", 0, 5),
        ("ultimate", 1, 5),
        ("ultimate", 5, 0),
    )
    after_positions = np.asarray(session.state.agent_positions)
    for source_slot in (0, 1, 5):
        assert np.all(np.isfinite(after_positions[source_slot]))
        assert not np.array_equal(
            after_positions[source_slot],
            before_positions[source_slot],
        )


def test_trap_lifecycle_reference_trajectory() -> None:
    _, session = _session("trap_lifecycle")

    session = submit_next_script_frame(session)
    assert _canonical_ability_signatures(session) == (
        ("ultimate", 0, 5),
        ("ultimate", 1, 6),
        ("ultimate", 2, 7),
        ("ultimate", 3, 8),
    )
    for target_slot in (5, 6, 7, 8):
        assert _canonical_status_event_types(
            session,
            global_slot=target_slot,
            status_id="hunter_trap_stun",
        ) == ("status_applied",)

    session = submit_next_script_frame(session)
    assert _canonical_status_event_types(
        session,
        global_slot=5,
        status_id="hunter_trap_stun",
    ) == ("status_broken_by_damage",)

    session = submit_next_script_frame(session)
    view = session.incoming_evaluation_view
    assert view is not None
    accepted = view.transition.facts.action_acceptance_facts.accepted_joint_action
    for accepted_head in (accepted.move, accepted.select_target, accepted.use_ultimate):
        np.testing.assert_array_equal(
            accepted_head,
            np.zeros((MAX_AGENT_SLOTS,), dtype=np.int32),
        )

    session = submit_next_script_frame(session)
    assert _canonical_ability_signatures(session) == (("ultimate", 4, 6),)
    assert _canonical_status_event_types(
        session,
        global_slot=6,
        status_id="hunter_trap_stun",
    ) == ("status_broken_by_damage", "status_applied")

    session = submit_next_script_frame(session)
    view = session.incoming_evaluation_view
    assert view is not None
    assert _canonical_status_event_types(
        session,
        global_slot=7,
        status_id="hunter_trap_stun",
    ) == ("status_aged_to_zero",)
    assert _canonical_status_event_types(
        session,
        global_slot=8,
        status_id="hunter_trap_stun",
    ) == ("status_aged_to_zero",)
    assert not any(
        isinstance(event, StatusLifecycleEventBaseV1)
        and event.recipient_global_slot == 7
        and event.event_type == "status_broken_by_damage"
        for event in view.transition.events
    )


def test_max_status_stack_reference_trajectory() -> None:
    _, session = _session("max_status_stack")
    session = submit_next_script_frame(session)

    assert _canonical_ability_signatures(session) == (
        ("ultimate", 0, None),
        ("basic", 1, 0),
        ("ultimate", 5, 0),
        ("ultimate", 6, 0),
        ("basic", 7, 0),
        ("ultimate", 8, 0),
    )
    assert tuple(int(value) for value in session.state.stun_durations[0]) == (
        1,
        4,
        1,
    )
    assert tuple(int(value) for value in session.state.slow_durations[0]) == (
        5,
        1,
        5,
    )
    assert int(session.state.rogue_poison_anti_heal_durations[0]) == 4
    assert int(session.state.priest_blessing_of_freedom_slow_floor_durations[0]) == 1
    assert int(session.state.mage_burst_damage_amplification_durations[0]) == 5
    assert int(session.state.stun_durations[0, STUN_CHANNEL_HUNTER_TRAP]) > 0
