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

from marl_battlegrounds.core.config import (
    CANONICAL_PRODUCT_MOVEMENT_SCALE,
    validate_env_config,
)
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
    STUN_CHANNEL_HUNTER_TRAP,
    WARRIOR_CLASS_ID,
)
from marl_battlegrounds.evaluation.models import (
    AbilityActivatedEventV1,
    ActionRejectedEventV1,
    RecipientHealthResolutionEventV1,
    SourceDamageOutputEventV1,
    StatusLifecycleEventBaseV1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    build_evaluation_battlefield_scene_v2,
)
from marl_battlegrounds.rendering.scene import BattlefieldSceneV2


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


def _catalog_status_duration(session: DebuggerSession, status_id: str) -> int:
    """Read one configured duration from the episode's immutable catalog."""
    return next(
        row.duration_steps
        for row in session.evaluation_context.static_mechanics_catalog.status_channels
        if row.status_id == status_id
    )


def _catalog_aura_multiplier(session: DebuggerSession, aura_id: str) -> float:
    """Read one configured per-emitter multiplier from the episode catalog."""
    return next(
        row.per_emitter_multiplier
        for row in session.evaluation_context.static_mechanics_catalog.aura_mechanics
        if row.aura_id == aura_id
    )


def _researcher_scene(session: DebuggerSession) -> BattlefieldSceneV2:
    """Build the normalized Scene V2 served for the current debugger epoch."""
    return build_evaluation_battlefield_scene_v2(
        session.evaluation_context,
        session.current_evaluation_frame,
        transition_view=session.incoming_evaluation_view,
        status_source_evidence_state=session.status_source_evidence_state,
    )


def _assert_health_matches_researcher_scene(
    session: DebuggerSession,
    global_slots: tuple[int, ...],
) -> None:
    """Join public health-resolution events to successor Scene V2 health."""
    view = session.incoming_evaluation_view
    assert view is not None
    scene = _researcher_scene(session)
    health_by_slot = {row.global_slot: row.current_health for row in scene.agents}
    resolution_by_slot = {
        event.recipient_global_slot: event
        for event in view.transition.events
        if isinstance(event, RecipientHealthResolutionEventV1)
    }
    assert set(global_slots).issubset(resolution_by_slot)
    for global_slot in global_slots:
        resolution = resolution_by_slot[global_slot]
        assert resolution.health_after_combat_resolution == pytest.approx(
            health_by_slot[global_slot]
        )
        assert resolution.realized_net_health_change == pytest.approx(
            resolution.health_after_combat_resolution
            - resolution.transition_start_health
        )
    np.testing.assert_allclose(
        np.asarray(session.state.current_health)[list(global_slots)],
        tuple(health_by_slot[slot] for slot in global_slots),
    )


def _assert_effective_speed_matches_researcher_scene(
    session: DebuggerSession,
    global_slot: int,
) -> None:
    """Tie effective-speed checks to the normalized Scene V2 value."""
    scene = _researcher_scene(session)
    agent = next(row for row in scene.agents if row.global_slot == global_slot)
    assert float(
        session.observation.self_features[
            global_slot,
            AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
        ]
    ) == pytest.approx(agent.effective_movement_speed)


def _assert_durable_mechanics_match_researcher_scene(
    session: DebuggerSession,
    global_slots: tuple[int, ...],
) -> None:
    """Join status/cooldown truth to Scene V2 without owning tuning numbers."""
    scene_by_slot = {row.global_slot: row for row in _researcher_scene(session).agents}
    for global_slot in global_slots:
        state_durations = (
            *tuple(int(value) for value in session.state.slow_durations[global_slot]),
            *tuple(int(value) for value in session.state.stun_durations[global_slot]),
            int(session.state.rogue_poison_anti_heal_durations[global_slot]),
            int(session.state.mage_burst_damage_amplification_durations[global_slot]),
            int(
                session.state.priest_blessing_of_freedom_slow_floor_durations[
                    global_slot
                ]
            ),
        )
        expected_statuses = tuple(
            (channel, duration)
            for channel, duration in enumerate(state_durations)
            if duration > 0
        )
        scene_agent = scene_by_slot[global_slot]
        observed_statuses = tuple(
            sorted(
                (row.status_channel, row.remaining_duration)
                for row in scene_agent.statuses
            )
        )
        assert observed_statuses == expected_statuses
        assert scene_agent.ultimate_cooldown_remaining == int(
            session.state.ultimate_cooldowns[global_slot]
        )


def _scene_status_ids(session: DebuggerSession, global_slot: int) -> tuple[str, ...]:
    """Read stable status identity, without mirroring volatile durations."""
    agent = next(
        row
        for row in _researcher_scene(session).agents
        if row.global_slot == global_slot
    )
    return tuple(row.status_id for row in agent.statuses)


def _scene_status_durations(
    session: DebuggerSession,
    global_slot: int,
) -> dict[str, int]:
    """Read normalized durable status values keyed by stable catalog identity."""
    agent = next(
        row
        for row in _researcher_scene(session).agents
        if row.global_slot == global_slot
    )
    return {row.status_id: row.remaining_duration for row in agent.statuses}


def _expected_status_durations_after_transition(
    session: DebuggerSession,
    *,
    global_slot: int,
    previous: dict[str, int],
) -> dict[str, int]:
    """Derive timer aging and lifecycle edges from catalog/event authority."""
    expected = {
        status_id: duration - 1
        for status_id, duration in previous.items()
        if duration > 1
    }
    view = session.incoming_evaluation_view
    assert view is not None
    for event in view.transition.events:
        if not (
            isinstance(event, StatusLifecycleEventBaseV1)
            and event.recipient_global_slot == global_slot
        ):
            continue
        if event.event_type in ("status_applied", "status_refreshed_or_extended"):
            expected[event.status_id] = _catalog_status_duration(
                session,
                event.status_id,
            )
        else:
            expected.pop(event.status_id, None)
    return expected


def _positive_scene_cooldown_slots(session: DebuggerSession) -> tuple[int, ...]:
    """Read cooldown ownership from normalized successor scene truth."""
    return tuple(
        row.global_slot
        for row in _researcher_scene(session).agents
        if row.ultimate_cooldown_remaining > 0
    )


def _expected_trap_lifecycle_by_slot(
    *,
    duration: int,
    transition: int,
) -> dict[int, tuple[str, ...]]:
    """Derive the authored five-frame Trap story from its catalog duration."""
    expected: dict[int, tuple[str, ...]] = {
        global_slot: () for global_slot in (5, 6, 7, 8)
    }
    if transition == 2:
        if duration == 1:
            return {
                global_slot: ("status_aged_to_zero",) for global_slot in (5, 6, 7, 8)
            }
        expected[5] = ("status_broken_by_damage",)
    elif transition == 3 and duration == 2:
        for global_slot in (6, 7, 8):
            expected[global_slot] = ("status_aged_to_zero",)
    elif transition == 4:
        if duration == 3:
            expected[6] = ("status_aged_to_zero", "status_applied")
            expected[7] = ("status_aged_to_zero",)
            expected[8] = ("status_aged_to_zero",)
        elif duration >= 4:
            expected[6] = ("status_broken_by_damage", "status_applied")
        else:
            expected[6] = ("status_applied",)
    elif transition == 5:
        if duration == 1:
            expected[6] = ("status_aged_to_zero",)
        elif duration == 4:
            expected[7] = ("status_aged_to_zero",)
            expected[8] = ("status_aged_to_zero",)
        elif duration >= 5:
            expected[7] = ("status_broken_by_damage",)
    return expected


def _assert_catalog_derived_trap_lifecycle(
    session: DebuggerSession,
    *,
    transition: int,
) -> None:
    """Prove exact Trap causality without fixing its duration to four ticks."""
    expected = _expected_trap_lifecycle_by_slot(
        duration=_catalog_status_duration(session, "hunter_trap_stun"),
        transition=transition,
    )
    for global_slot, event_types in expected.items():
        assert (
            _canonical_status_event_types(
                session,
                global_slot=global_slot,
                status_id="hunter_trap_stun",
            )
            == event_types
        )


def test_all_scenario_configs_validate_and_initialize_authored_state() -> None:
    researcher_names = (
        "arena_5v5",
        "basic_support",
        "ultimate_showcase",
        "aura_crossfire",
        "stacked_team_auras",
        "status_stack",
        "team_focus_crossfire",
        "mirrored_ultimates",
        "death_respawn_cycle",
        "recovery_refresh_cycle",
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
    assert cycle_scenario_name("recovery_refresh_cycle", 1) == "arena_5v5"
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
        assert (
            config.ordinary_movement_distance_scale == CANONICAL_PRODUCT_MOVEMENT_SCALE
        )
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
        if scenario.name not in {
            "death_respawn_cycle",
            "recovery_refresh_cycle",
        }:
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
            (18.0, 12.0),
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
                (6.0, 3.0),
                (6.0, 6.0),
                (6.0, 9.0),
                (9.0, 3.0),
                (9.0, 6.0),
                (9.0, 9.0),
            ),
            "scripted",
            0,
        ),
        "ultimate_showcase": (
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
                (4.0, 2.0),
                (6.0, 5.0),
                (6.0, 8.0),
                (9.0, 5.0),
                (4.0, 10.0),
                (8.0, 6.0),
                (9.0, 8.0),
                (11.0, 3.0),
                (13.0, 8.0),
                (14.0, 10.0),
            ),
            "scripted",
            0,
        ),
        "aura_crossfire": (
            (18.0, 12.0),
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
                (6.0, 5.0),
                (6.0, 7.0),
                (7.5, 6.0),
                (12.0, 5.0),
                (12.0, 7.0),
                (10.5, 6.0),
            ),
            "scripted",
            2,
        ),
        "stacked_team_auras": (
            (18.0, 12.0),
            tuple(range(10)),
            (
                MAGE_CLASS_ID,
                MAGE_CLASS_ID,
                WARRIOR_CLASS_ID,
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
                MAGE_CLASS_ID,
                MAGE_CLASS_ID,
                WARRIOR_CLASS_ID,
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
            ),
            (
                (6.0, 5.0),
                (6.0, 7.0),
                (7.5, 4.5),
                (7.5, 7.5),
                (7.5, 6.0),
                (12.0, 5.0),
                (12.0, 7.0),
                (10.5, 4.5),
                (10.5, 7.5),
                (10.5, 6.0),
            ),
            "scripted",
            4,
        ),
        "status_stack": (
            (18.0, 12.0),
            (0, 1, 2, 5, 6),
            (
                WARRIOR_CLASS_ID,
                HUNTER_CLASS_ID,
                ROGUE_CLASS_ID,
                HUNTER_CLASS_ID,
                PRIEST_CLASS_ID,
            ),
            (
                (5.0, 6.0),
                (7.5, 4.4),
                (10.0, 5.0),
                (10.0, 6.0),
                (10.0, 8.0),
            ),
            "scripted",
            5,
        ),
        "team_focus_crossfire": (
            (18.0, 12.0),
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
                (7.0, 6.0),
                (8.0, 5.0),
                (9.0, 3.0),
                (9.0, 4.6),
                (9.0, 6.0),
                (11.5, 6.0),
                (10.8, 8.0),
                (7.2, 8.0),
            ),
            "scripted",
            2,
        ),
        "mirrored_ultimates": (
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
                (3.0, 1.0),
                (6.0, 4.0),
                (6.6, 8.0),
                (7.3, 11.0),
                (4.0, 10.5),
                (15.0, 1.0),
                (10.0, 4.0),
                (9.4, 8.0),
                (8.7, 11.0),
                (12.0, 10.5),
            ),
            "scripted",
            0,
        ),
        "death_respawn_cycle": (
            (18.0, 12.0),
            (0, 1, 5),
            (MAGE_CLASS_ID, HUNTER_CLASS_ID, ROGUE_CLASS_ID),
            ((13.0, 1.5), (13.0, 3.0), (14.5, 2.25)),
            "scripted",
            5,
        ),
        "recovery_refresh_cycle": (
            (18.0, 12.0),
            (0, 1, 2, 3, 4, 5, 6, 7),
            (
                ROGUE_CLASS_ID,
                ROGUE_CLASS_ID,
                HUNTER_CLASS_ID,
                HUNTER_CLASS_ID,
                MAGE_CLASS_ID,
                HUNTER_CLASS_ID,
                PRIEST_CLASS_ID,
                WARRIOR_CLASS_ID,
            ),
            (
                (8.0, 4.5),
                (8.0, 5.5),
                (6.7, 7.0),
                (7.5, 8.0),
                (8.5, 7.0),
                (9.2, 5.0),
                (13.0, 9.0),
                (9.5, 7.5),
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
        "stacked_team_auras": (((4, MOVE_STAY, 9, 0), (9, MOVE_STAY, 4, 0)),),
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
        "death_respawn_cycle": (
            ((0, MOVE_STAY, 5, 0), (1, MOVE_STAY, 5, 0)),
            (),
            ((5, MOVE_WEST, 0, 1),),
            ((5, MOVE_WEST, 0, 0),),
            ((5, MOVE_STAY, 0, 0),),
            ((5, MOVE_STAY, 0, 0),),
            ((5, MOVE_STAY, 0, 0),),
        ),
        "recovery_refresh_cycle": (
            (
                (0, MOVE_STAY, 5, 1),
                (2, MOVE_STAY, 7, 1),
                (6, MOVE_STAY, 6, 1),
            ),
            (
                (1, MOVE_STAY, 5, 1),
                (3, MOVE_STAY, 7, 1),
                (4, MOVE_STAY, 7, 0),
            ),
            (),
            (),
            (),
            (),
            (),
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


def test_every_registered_scripted_command_matches_authored_acceptance() -> None:
    expected_rejections = {
        ("death_respawn_cycle", "due-wave-respawn"): (
            (5, "movement"),
            (5, "combat_pair"),
        ),
        ("death_respawn_cycle", "shielded-movement-and-rejection"): (
            (5, "combat_pair"),
        ),
        ("death_respawn_cycle", "shield-countdown-two-to-one"): ((5, "combat_pair"),),
        ("death_respawn_cycle", "shield-expiry"): ((5, "combat_pair"),),
        ("recovery_refresh_cycle", "application-recovery-and-readiness"): (
            (6, "combat_pair"),
        ),
    }
    for scenario in list_scenarios(include_stress=True):
        if not scenario.frames:
            continue
        _, session = _session(scenario.name)
        for frame in scenario.frames:
            expected_frame_rejections = expected_rejections.get(
                (scenario.name, frame.label),
                (),
            )
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
                move_is_legal = bool(
                    session.action_mask.move_mask[slot, command.move_action]
                )
                combat_pair_is_legal = bool(
                    session.action_mask.select_target_use_ultimate_joint_mask[
                        slot,
                        target_action,
                        command.use_ultimate,
                    ]
                )
                assert move_is_legal == (
                    (slot, "movement") not in expected_frame_rejections
                )
                assert combat_pair_is_legal == (
                    (slot, "combat_pair") not in expected_frame_rejections
                )

            submitted = submit_next_script_frame(session)
            view = submitted.incoming_evaluation_view
            assert view is not None
            acceptance = view.transition.facts.action_acceptance_facts
            observed_rejections = tuple(
                (event.actor_global_slot, event.rejection_component)
                for event in view.transition.events
                if isinstance(event, ActionRejectedEventV1)
            )
            assert observed_rejections == expected_frame_rejections
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
            for command in frame.commands:
                actor_slot = command.actor_global_slot
                expected_target = int(action.select_target[actor_slot])
                move_rejected = (actor_slot, "movement") in expected_frame_rejections
                combat_rejected = (
                    actor_slot,
                    "combat_pair",
                ) in expected_frame_rejections
                assert acceptance.accepted_joint_action.move[actor_slot] == (
                    MOVE_STAY if move_rejected else command.move_action
                )
                assert acceptance.accepted_joint_action.select_target[actor_slot] == (
                    0 if combat_rejected else expected_target
                )
                assert acceptance.accepted_joint_action.use_ultimate[actor_slot] == (
                    0 if combat_rejected else command.use_ultimate
                )
                assert not (
                    acceptance.submitted_action_tuple_is_out_of_domain_by_actor[
                        actor_slot
                    ]
                )
                assert (
                    bool(
                        acceptance.in_domain_move_action_is_rejected_by_actor[
                            actor_slot
                        ]
                    )
                    == move_rejected
                )
                assert (
                    bool(
                        acceptance.in_domain_combat_action_pair_is_rejected_by_actor[
                            actor_slot
                        ]
                    )
                    == combat_rejected
                )
            session = submitted


def test_researcher_scenarios_cover_every_canonical_event_kind() -> None:
    canonical_event_kinds = {
        "action_rejected",
        "ability_activated",
        "source_damage_output",
        "source_healing_output",
        "recipient_health_resolution",
        "combat_countdown_reset",
        "health_regenerated",
        "cooldown_started",
        "cooldown_ready",
        "charge_phase_displacement",
        "ordinary_movement_phase_displacement",
        "agent_died",
        "lethal_damage_contribution",
        "status_aged_to_zero",
        "status_broken_by_damage",
        "status_applied",
        "status_refreshed_or_extended",
        "status_cleared_by_new_death",
        "spawn_shield_expired",
        "respawn_wave_occurred",
        "agent_respawned",
    }
    observed_event_kinds: set[str] = set()
    for scenario in RESEARCHER_SCENARIOS.values():
        if not scenario.frames:
            continue
        _, session = _session(scenario.name)
        for _ in scenario.frames:
            session = submit_next_script_frame(session)
            view = session.incoming_evaluation_view
            assert view is not None
            observed_event_kinds.update(
                event.event_type for event in view.transition.events
            )

    assert observed_event_kinds == canonical_event_kinds


def test_death_respawn_cycle_reference_trajectory() -> None:
    _, session = _session("death_respawn_cycle")

    session = submit_next_script_frame(session)
    first_view = session.incoming_evaluation_view
    assert first_view is not None
    assert [
        event.event_type
        for event in first_view.transition.events
        if event.event_type
        in {
            "agent_died",
            "lethal_damage_contribution",
            "status_cleared_by_new_death",
        }
    ] == [
        "agent_died",
        "lethal_damage_contribution",
        "lethal_damage_contribution",
        "status_cleared_by_new_death",
        "status_cleared_by_new_death",
        "status_cleared_by_new_death",
    ]
    corpse = next(
        row for row in _researcher_scene(session).agents if row.global_slot == 5
    )
    assert (corpse.life_state, corpse.current_health, corpse.statuses) == (
        "corpse",
        0.0,
        (),
    )

    session = submit_next_script_frame(session)
    waiting = next(
        row for row in _researcher_scene(session).agents if row.global_slot == 5
    )
    assert waiting.life_state == "corpse"
    assert session.incoming_evaluation_view is not None
    assert session.incoming_evaluation_view.transition.events == ()

    session = submit_next_script_frame(session)
    respawn_view = session.incoming_evaluation_view
    assert respawn_view is not None
    assert tuple(event.event_type for event in respawn_view.transition.events) == (
        "action_rejected",
        "action_rejected",
        "respawn_wave_occurred",
        "agent_respawned",
    )
    respawned = next(
        row for row in _researcher_scene(session).agents if row.global_slot == 5
    )
    assert (
        respawned.life_state,
        respawned.current_health,
        respawned.spawn_shield_remaining,
        respawned.respawned_on_incoming_transition,
    ) == ("alive", respawned.max_health, 3, True)

    expected_shield_remaining = (2, 1, 0)
    for expected_remaining in expected_shield_remaining:
        session = submit_next_script_frame(session)
        incoming_view = session.incoming_evaluation_view
        assert incoming_view is not None
        shielded = next(
            row for row in _researcher_scene(session).agents if row.global_slot == 5
        )
        assert shielded.spawn_shield_remaining == expected_remaining
        assert any(
            event.event_type == "action_rejected"
            for event in incoming_view.transition.events
        )
    incoming_view = session.incoming_evaluation_view
    assert incoming_view is not None
    assert any(
        event.event_type == "spawn_shield_expired"
        for event in incoming_view.transition.events
    )

    session = submit_next_script_frame(session)
    incoming_view = session.incoming_evaluation_view
    assert incoming_view is not None
    assert _canonical_ability_signatures(session) == (("basic", 5, 0),)
    assert not any(
        event.event_type == "action_rejected"
        for event in incoming_view.transition.events
    )


def test_recovery_refresh_cycle_reference_trajectory() -> None:
    _, session = _session("recovery_refresh_cycle")

    session = submit_next_script_frame(session)
    first_view = session.incoming_evaluation_view
    assert first_view is not None
    assert {event.event_type for event in first_view.transition.events}.issuperset(
        {
            "action_rejected",
            "health_regenerated",
            "cooldown_ready",
            "status_applied",
        }
    )
    assert _scene_status_durations(session, 5) == {
        "rogue_poison_stun": 1,
        "rogue_poison_slow": 5,
        "rogue_poison_anti_heal": 4,
    }
    assert _scene_status_durations(session, 7) == {"hunter_trap_stun": 4}

    session = submit_next_script_frame(session)
    second_view = session.incoming_evaluation_view
    assert second_view is not None
    assert _canonical_status_event_types(
        session,
        global_slot=5,
        status_id="rogue_poison_stun",
    ) == ("status_aged_to_zero", "status_applied")
    assert _canonical_status_event_types(
        session,
        global_slot=5,
        status_id="rogue_poison_slow",
    ) == ("status_applied", "status_refreshed_or_extended")
    assert _canonical_status_event_types(
        session,
        global_slot=7,
        status_id="hunter_trap_stun",
    ) == ("status_broken_by_damage", "status_applied")

    for _ in range(5):
        session = submit_next_script_frame(session)
    assert _scene_status_ids(session, 5) == ()
    assert _scene_status_ids(session, 7) == ()
    assert _canonical_status_event_types(
        session,
        global_slot=5,
        status_id="rogue_poison_slow",
    ) == ("status_aged_to_zero",)


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
    catalog = session.evaluation_context.static_mechanics_catalog
    np.testing.assert_array_equal(
        session.state.current_health,
        tuple(
            catalog.class_mechanics[roster.class_id].maximum_health
            for roster in session.evaluation_context.roster
        ),
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
    np.testing.assert_allclose(
        mage_auras[[0, 1, 5, 6]],
        _catalog_aura_multiplier(session, "mage_damage_amplification"),
    )
    np.testing.assert_allclose(
        warrior_auras[[0, 1, 2, 5, 6, 7]],
        _catalog_aura_multiplier(session, "warrior_damage_mitigation"),
    )
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
    _assert_health_matches_researcher_scene(session, (2, 5, 6))
    _assert_durable_mechanics_match_researcher_scene(session, (2, 5, 6))
    assert _canonical_ability_signatures(session) == (
        ("basic", 0, 5),
        ("basic", 1, 6),
        ("basic", 7, 2),
    )
    assert "hunter_basic_slow" in _scene_status_ids(session, 2)
    assert "hunter_basic_slow" in _scene_status_ids(session, 6)
    assert bool(jnp.all(session.state.ultimate_cooldowns == 0))

    session = submit_next_script_frame(session)
    _assert_health_matches_researcher_scene(session, (2,))
    _assert_durable_mechanics_match_researcher_scene(session, (2, 5, 6))
    assert _canonical_ability_signatures(session) == (
        ("basic", 2, 2),
        ("basic", 7, 2),
    )
    assert {
        "hunter_basic_slow",
        "priest_blessing_of_freedom_movement_floor",
    }.issubset(_scene_status_ids(session, 2))
    view = session.incoming_evaluation_view
    assert view is not None
    event_types = {event.event_type for event in view.transition.events}
    assert {"source_damage_output", "source_healing_output"}.issubset(event_types)


def test_ultimate_showcase_reference_trajectory() -> None:
    _, session = _session("ultimate_showcase")
    session = submit_next_script_frame(session)
    _assert_health_matches_researcher_scene(session, (2,))
    assert _canonical_ability_signatures(session) == (("basic", 5, 2),)

    positions_before = np.asarray(session.state.agent_positions).copy()
    session = submit_next_script_frame(session)
    assert not np.array_equal(session.state.agent_positions[1], positions_before[1])
    _assert_health_matches_researcher_scene(session, (2, 5, 6, 7))
    _assert_durable_mechanics_match_researcher_scene(session, tuple(range(8)))
    assert _canonical_ability_signatures(session) == (
        ("ultimate", 0, None),
        ("ultimate", 1, 7),
        ("ultimate", 2, 6),
        ("ultimate", 3, 5),
        ("ultimate", 4, 2),
    )
    assert _positive_scene_cooldown_slots(session) == (0, 1, 2, 3, 4)
    assert "mage_burst_damage_amplification" in _scene_status_ids(session, 0)
    assert {
        "warrior_charge_stun",
        "warrior_charge_slow",
    }.issubset(_scene_status_ids(session, 7))
    assert "hunter_trap_stun" in _scene_status_ids(session, 6)
    assert {
        "rogue_poison_stun",
        "rogue_poison_slow",
        "rogue_poison_anti_heal",
    }.issubset(_scene_status_ids(session, 5))

    session = submit_next_script_frame(session)
    _assert_health_matches_researcher_scene(session, (6,))
    _assert_durable_mechanics_match_researcher_scene(session, tuple(range(8)))
    assert _canonical_ability_signatures(session) == (("basic", 2, 6),)
    expected_trap_event = (
        ("status_aged_to_zero",)
        if _catalog_status_duration(session, "hunter_trap_stun") == 1
        else ("status_broken_by_damage",)
    )
    assert (
        _canonical_status_event_types(
            session,
            global_slot=6,
            status_id="hunter_trap_stun",
        )
        == expected_trap_event
    )


def test_aura_crossfire_reference_trajectory() -> None:
    _, session = _session("aura_crossfire")
    self_features = np.asarray(session.observation.self_features)
    np.testing.assert_allclose(
        self_features[
            [0, 1, 2, 5, 6, 7],
            AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
        ],
        _catalog_aura_multiplier(session, "mage_damage_amplification"),
    )
    np.testing.assert_allclose(
        self_features[
            [0, 1, 2, 5, 6, 7],
            AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
        ],
        _catalog_aura_multiplier(session, "warrior_damage_mitigation"),
    )
    session = submit_next_script_frame(session)
    _assert_health_matches_researcher_scene(session, (2, 7))
    assert _canonical_ability_signatures(session) == (
        ("basic", 2, 7),
        ("basic", 7, 2),
    )
    assert "hunter_basic_slow" in _scene_status_ids(session, 2)
    assert "hunter_basic_slow" in _scene_status_ids(session, 7)
    np.testing.assert_array_equal(
        np.asarray(session.state.slow_durations)[
            [2, 7],
            SLOW_CHANNEL_HUNTER_BASIC,
        ],
        (
            _catalog_status_duration(session, "hunter_basic_slow"),
            _catalog_status_duration(session, "hunter_basic_slow"),
        ),
    )


def test_stacked_team_auras_reference_trajectory() -> None:
    _, session = _session("stacked_team_auras")
    mage_per_emitter = _catalog_aura_multiplier(
        session,
        "mage_damage_amplification",
    )
    warrior_per_emitter = _catalog_aura_multiplier(
        session,
        "warrior_damage_mitigation",
    )
    assert mage_per_emitter == pytest.approx(1.15)
    assert warrior_per_emitter == pytest.approx(0.85)
    expected_mage_aggregate = mage_per_emitter**2
    expected_warrior_aggregate = warrior_per_emitter**2
    assert expected_mage_aggregate == pytest.approx(1.3225)
    assert expected_warrior_aggregate == pytest.approx(0.7225)
    expected_mage_by_slot = (
        expected_mage_aggregate,
        expected_mage_aggregate,
        mage_per_emitter,
        mage_per_emitter,
        expected_mage_aggregate,
        expected_mage_aggregate,
        expected_mage_aggregate,
        mage_per_emitter,
        mage_per_emitter,
        expected_mage_aggregate,
    )
    expected_warrior_by_slot = (
        warrior_per_emitter,
        warrior_per_emitter,
        warrior_per_emitter,
        warrior_per_emitter,
        expected_warrior_aggregate,
        warrior_per_emitter,
        warrior_per_emitter,
        warrior_per_emitter,
        warrior_per_emitter,
        expected_warrior_aggregate,
    )
    self_features = np.asarray(session.observation.self_features)
    np.testing.assert_allclose(
        self_features[
            :,
            AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
        ],
        expected_mage_by_slot,
    )
    np.testing.assert_allclose(
        self_features[
            :,
            AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
        ],
        expected_warrior_by_slot,
    )
    np.testing.assert_allclose(
        self_features[
            [4, 9],
            AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
        ],
        expected_mage_aggregate,
    )
    np.testing.assert_allclose(
        self_features[
            [4, 9],
            AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
        ],
        expected_warrior_aggregate,
    )
    assert np.all(
        self_features[
            :,
            AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
        ]
        != 1.0
    )
    assert np.all(
        self_features[
            :,
            AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
        ]
        != 1.0
    )

    scene = _researcher_scene(session)
    assert tuple(
        (field.source_global_slot, field.aura_id) for field in scene.aura_fields
    ) == (
        (0, "mage_damage_amplification"),
        (1, "mage_damage_amplification"),
        (2, "warrior_damage_mitigation"),
        (3, "warrior_damage_mitigation"),
        (5, "mage_damage_amplification"),
        (6, "mage_damage_amplification"),
        (7, "warrior_damage_mitigation"),
        (8, "warrior_damage_mitigation"),
    )
    for agent in scene.agents:
        modifiers = {row.aura_id: row for row in agent.aura_modifiers}
        assert tuple(modifiers) == (
            "mage_damage_amplification",
            "warrior_damage_mitigation",
        )
        assert all(row.multiplier != 1.0 for row in modifiers.values())
        assert all(not hasattr(row, "source_global_slot") for row in modifiers.values())
        assert modifiers["mage_damage_amplification"].multiplier == pytest.approx(
            expected_mage_by_slot[agent.global_slot]
        )
        assert modifiers["warrior_damage_mitigation"].multiplier == pytest.approx(
            expected_warrior_by_slot[agent.global_slot]
        )
        if agent.global_slot in (4, 9):
            assert modifiers["mage_damage_amplification"].multiplier == pytest.approx(
                expected_mage_aggregate
            )
            assert modifiers["warrior_damage_mitigation"].multiplier == pytest.approx(
                expected_warrior_aggregate
            )

    session = submit_next_script_frame(session)
    _assert_health_matches_researcher_scene(session, (4, 9))
    assert _canonical_ability_signatures(session) == (
        ("basic", 4, 9),
        ("basic", 9, 4),
    )
    view = session.incoming_evaluation_view
    assert view is not None
    damage_events = tuple(
        event
        for event in view.transition.events
        if isinstance(event, SourceDamageOutputEventV1)
    )
    assert tuple(
        (
            event.source_global_slot,
            event.recipient_global_slot,
            event.mage_damage_aura_covering_emitter_global_slots,
            event.warrior_mitigation_aura_covering_emitter_global_slots,
        )
        for event in damage_events
    ) == (
        (4, 9, (0, 1), (7, 8)),
        (9, 4, (5, 6), (2, 3)),
    )
    for event in damage_events:
        assert event.source_modified_damage_output == pytest.approx(
            event.raw_damage_output * expected_mage_aggregate
        )
        assert event.recipient_damage_modifier == pytest.approx(
            expected_warrior_aggregate
        )
    assert "hunter_basic_slow" in _scene_status_ids(session, 4)
    assert "hunter_basic_slow" in _scene_status_ids(session, 9)


def test_status_stack_reference_trajectory() -> None:
    _, session = _session("status_stack")
    active_slots = (0, 1, 2, 5, 6)
    expected_statuses: dict[str, int] = {}
    position_before = np.asarray(session.state.agent_positions[0]).copy()
    session = submit_next_script_frame(session)
    expected_statuses = _expected_status_durations_after_transition(
        session,
        global_slot=5,
        previous=expected_statuses,
    )
    assert _scene_status_durations(session, 5) == expected_statuses
    assert not np.array_equal(session.state.agent_positions[0], position_before)
    _assert_health_matches_researcher_scene(session, (5,))
    _assert_durable_mechanics_match_researcher_scene(session, active_slots)
    _assert_effective_speed_matches_researcher_scene(session, 5)
    assert _canonical_ability_signatures(session) == (
        ("ultimate", 0, 5),
        ("ultimate", 1, 5),
        ("ultimate", 2, 5),
        ("basic", 6, 5),
    )
    assert {
        "warrior_charge_slow",
        "rogue_poison_slow",
        "warrior_charge_stun",
        "hunter_trap_stun",
        "rogue_poison_stun",
        "rogue_poison_anti_heal",
        "priest_blessing_of_freedom_movement_floor",
    } == set(_scene_status_ids(session, 5))
    assert _positive_scene_cooldown_slots(session) == (0, 1, 2)

    position_before = np.asarray(session.state.agent_positions[5]).copy()
    session = submit_next_script_frame(session)
    expected_statuses = _expected_status_durations_after_transition(
        session,
        global_slot=5,
        previous=expected_statuses,
    )
    assert _scene_status_durations(session, 5) == expected_statuses
    np.testing.assert_array_equal(session.state.agent_positions[5], position_before)
    _assert_health_matches_researcher_scene(session, (5,))
    _assert_durable_mechanics_match_researcher_scene(session, active_slots)
    _assert_effective_speed_matches_researcher_scene(session, 5)
    assert _canonical_ability_signatures(session) == (
        ("basic", 1, 5),
        ("basic", 6, 5),
    )
    assert {
        "hunter_basic_slow",
        "priest_blessing_of_freedom_movement_floor",
    }.issubset(_scene_status_ids(session, 5))

    for _ in range(2):
        session = submit_next_script_frame(session)
        expected_statuses = _expected_status_durations_after_transition(
            session,
            global_slot=5,
            previous=expected_statuses,
        )
        assert _scene_status_durations(session, 5) == expected_statuses
        assert np.all(np.isfinite(session.state.agent_positions[5]))
        view = session.incoming_evaluation_view
        assert view is not None
        assert not any(
            isinstance(event, RecipientHealthResolutionEventV1)
            for event in view.transition.events
        )
        _assert_durable_mechanics_match_researcher_scene(session, active_slots)
        _assert_effective_speed_matches_researcher_scene(session, 5)


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

    _assert_health_matches_researcher_scene(session, (5,))
    health_before_holy_words = anti_heal_view.successor_frame.snapshot.current_health[5]
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
        session.evaluation_context.static_mechanics_catalog.class_mechanics[
            WARRIOR_CLASS_ID
        ].maximum_health
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
        direct_statuses = (
            {
                0: {"mage_burst_damage_amplification"},
                5: {"mage_burst_damage_amplification"},
            },
            {
                1: {"warrior_charge_stun", "warrior_charge_slow"},
                6: {"warrior_charge_stun", "warrior_charge_slow"},
            },
            {2: {"hunter_trap_stun"}, 7: {"hunter_trap_stun"}},
            {
                3: {
                    "rogue_poison_stun",
                    "rogue_poison_slow",
                    "rogue_poison_anti_heal",
                },
                8: {
                    "rogue_poison_stun",
                    "rogue_poison_slow",
                    "rogue_poison_anti_heal",
                },
            },
            {},
        )[frame_index]
        for global_slot, status_ids in direct_statuses.items():
            assert status_ids.issubset(_scene_status_ids(session, global_slot))


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
    resolved_slots = session.evaluation_context.resolved_env_config.slot_mechanics
    pair_clearances = tuple(
        float(np.linalg.norm(after[left] - after[right]))
        - resolved_slots[left].body_radius
        - resolved_slots[right].body_radius
        for index, left in enumerate(involved_slots)
        for right in involved_slots[index + 1 :]
    )
    assert min(pair_clearances) > 0.0


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
    _assert_durable_mechanics_match_researcher_scene(session, (5, 6, 7, 8))

    session = submit_next_script_frame(session)
    assert _canonical_ability_signatures(session) == (("basic", 0, 5),)
    assert _canonical_status_event_types(
        session,
        global_slot=5,
        status_id="hunter_basic_slow",
    ) == ("status_applied",)
    _assert_catalog_derived_trap_lifecycle(session, transition=2)
    _assert_durable_mechanics_match_researcher_scene(session, (5, 6, 7, 8))

    session = submit_next_script_frame(session)
    view = session.incoming_evaluation_view
    assert view is not None
    accepted = view.transition.facts.action_acceptance_facts.accepted_joint_action
    for accepted_head in (accepted.move, accepted.select_target, accepted.use_ultimate):
        np.testing.assert_array_equal(
            accepted_head,
            np.zeros((MAX_AGENT_SLOTS,), dtype=np.int32),
        )
    _assert_catalog_derived_trap_lifecycle(session, transition=3)
    _assert_durable_mechanics_match_researcher_scene(session, (5, 6, 7, 8))

    session = submit_next_script_frame(session)
    assert _canonical_ability_signatures(session) == (("ultimate", 4, 6),)
    _assert_catalog_derived_trap_lifecycle(session, transition=4)
    _assert_durable_mechanics_match_researcher_scene(session, (5, 6, 7, 8))

    session = submit_next_script_frame(session)
    assert _canonical_ability_signatures(session) == (("basic", 2, 7),)
    assert _canonical_status_event_types(
        session,
        global_slot=7,
        status_id="hunter_basic_slow",
    ) == ("status_applied",)
    _assert_catalog_derived_trap_lifecycle(session, transition=5)
    _assert_durable_mechanics_match_researcher_scene(session, (5, 6, 7, 8))


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
    assert tuple(int(value) for value in session.state.stun_durations[0]) == tuple(
        _catalog_status_duration(session, status_id)
        for status_id in (
            "warrior_charge_stun",
            "hunter_trap_stun",
            "rogue_poison_stun",
        )
    )
    assert tuple(int(value) for value in session.state.slow_durations[0]) == tuple(
        _catalog_status_duration(session, status_id)
        for status_id in (
            "warrior_charge_slow",
            "hunter_basic_slow",
            "rogue_poison_slow",
        )
    )
    assert int(session.state.rogue_poison_anti_heal_durations[0]) == (
        _catalog_status_duration(session, "rogue_poison_anti_heal")
    )
    assert int(session.state.priest_blessing_of_freedom_slow_floor_durations[0]) == (
        _catalog_status_duration(
            session,
            "priest_blessing_of_freedom_movement_floor",
        )
    )
    assert int(session.state.mage_burst_damage_amplification_durations[0]) == (
        _catalog_status_duration(session, "mage_burst_damage_amplification")
    )
    assert int(session.state.stun_durations[0, STUN_CHANNEL_HUNTER_TRAP]) > 0
