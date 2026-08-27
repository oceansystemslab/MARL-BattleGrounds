"""Focused public-contract proofs for the interactive Combat Debugger arena."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from scripts.dev.visual_debugger.control import (
    create_session,
    set_pending_movement,
    submit_interactive,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.presentation_protocol import (
    LiveNoSharedObsAuthorizedPresentationFrameV1,
    LiveOracleAuthorizedPresentationFrameV1,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.service import DebuggerService
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.types import (
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_MAP_WIDTH,
    MOVE_EAST,
)


def _arena_session(*, controlled_global_slot: int = 0) -> DebuggerSession:
    return create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=controlled_global_slot,
        show_ranges=True,
        verbose_logging=False,
    )


def _public_lifecycle_session() -> DebuggerSession:
    """Install distinct public lifecycle states without changing body visibility."""
    registered = get_scenario("arena_5v5")
    config, state = registered.build_scenario()
    authored_state = state._replace(
        # Team B slot zero is dead with its public team wave due.  Team A slot
        # one and Team B slot one remain alive with distinct shield durations.
        alive_mask=state.alive_mask.at[5].set(False),
        current_health=state.current_health.at[5].set(0.0),
        spawn_shield_durations=(state.spawn_shield_durations.at[1].set(1).at[6].set(2)),
        team_respawn_wave_countdowns=jnp.asarray((3, 0), dtype=jnp.int32),
    )
    scenario = replace(
        registered,
        build_scenario=lambda: (config, authored_state),
    )
    return create_session(
        scenario,
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=0,
        show_ranges=True,
        verbose_logging=False,
    )


def test_live_oracle_and_agent_publish_the_same_public_arena_configuration() -> None:
    """Fog may filter bodies, but it cannot change public map mechanics."""
    session = _arena_session()
    oracle_result = DebuggerService(
        session,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        session_id="arena-public-config",
    ).current_presentation()
    agent_result = DebuggerService(
        session,
        view_mode="pov",
        preset="analysis",
        include_stress=False,
        session_id="arena-public-config",
    ).current_presentation()

    assert oracle_result.outcome == agent_result.outcome == "response"
    oracle = oracle_result.payload
    agent = agent_result.payload
    assert type(oracle) is LiveOracleAuthorizedPresentationFrameV1
    assert type(agent) is LiveNoSharedObsAuthorizedPresentationFrameV1

    oracle_scene = oracle.current_endpoint.scene
    agent_scene = agent.current_endpoint.parts.scene
    assert (oracle_scene.map.width, oracle_scene.map.height) == (20.0, 10.0)
    assert (agent_scene.map.width, agent_scene.map.height) == (
        oracle_scene.map.width,
        oracle_scene.map.height,
    )
    assert tuple(
        (
            row.kind,
            row.center,
            row.radius,
            row.width,
            row.height,
            row.theta,
        )
        for row in agent_scene.map.obstacles
    ) == tuple(
        (
            row.kind,
            row.center,
            row.radius,
            row.width,
            row.height,
            row.theta,
        )
        for row in oracle_scene.map.obstacles
    )
    oracle_mechanics_by_class = {
        row.class_id: row for row in oracle_scene.class_mechanics
    }
    assert tuple(agent_scene.class_mechanics) == tuple(
        oracle_mechanics_by_class[row.class_id] for row in agent_scene.class_mechanics
    )
    assert {row.class_id for row in agent_scene.class_mechanics} == {
        row.class_id for row in agent_scene.agents
    }
    assert agent_scene.spawn_shield_mechanics == oracle_scene.spawn_shield_mechanics
    assert agent.researcher_space.class_mechanics == oracle_scene.class_mechanics
    assert oracle.source.source_frame_index == agent.source.source_frame_index == 0
    assert (
        oracle.source.source_simulator_step_count
        == agent.source.source_simulator_step_count
        == 0
    )


def test_agent_fog_preserves_public_lifecycle_without_hidden_body_geometry() -> None:
    """Lifecycle is public policy input; fog removes identities and bodies only."""
    session = _public_lifecycle_session()
    oracle_result = DebuggerService(
        session,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        session_id="arena-public-lifecycle",
    ).current_presentation()
    agent_result = DebuggerService(
        session,
        view_mode="pov",
        preset="analysis",
        include_stress=False,
        session_id="arena-public-lifecycle",
    ).current_presentation()

    assert oracle_result.outcome == agent_result.outcome == "response"
    oracle = oracle_result.payload
    agent = agent_result.payload
    assert type(oracle) is LiveOracleAuthorizedPresentationFrameV1
    assert type(agent) is LiveNoSharedObsAuthorizedPresentationFrameV1
    oracle_scene = oracle.current_endpoint.scene
    agent_scene = agent.current_endpoint.parts.scene

    oracle_pads = {
        (row.team_id, row.team_local_slot): row for row in oracle_scene.spawn_pads
    }
    agent_pads = {
        (row.team_id, row.team_local_slot): row for row in agent_scene.spawn_pads
    }
    assert set(agent_pads) == set(oracle_pads)
    for key, oracle_pad in oracle_pads.items():
        agent_pad = agent_pads[key]
        assert (
            agent_pad.position,
            agent_pad.configured_active,
            agent_pad.currently_alive,
            agent_pad.spawn_shield_remaining,
        ) == (
            oracle_pad.position,
            oracle_pad.configured_active,
            oracle_pad.currently_alive,
            oracle_pad.spawn_shield_remaining,
        )

    # These fields are deliberately public in each actor's o_t, independently
    # of local body visibility.  Assert the authorized scene copies that exact
    # actor-relative input before converting its team axis to absolute IDs.
    lifecycle = session.observation.spawn_lifecycle
    controlled_slot = session.controlled_global_slot
    controlled_team_id = session.evaluation_context.roster[
        controlled_slot
    ].configured_team_id
    for (team_id, team_local_slot), pad in agent_pads.items():
        actor_relative_team = 0 if team_id == controlled_team_id else 1
        np.testing.assert_array_equal(
            np.asarray(pad.position),
            np.asarray(
                lifecycle.spawn_pad_positions_by_agent_by_team[
                    controlled_slot,
                    actor_relative_team,
                    team_local_slot,
                ]
            ),
        )
        assert pad.configured_active == bool(
            lifecycle.active_mask_by_agent_by_team[
                controlled_slot,
                actor_relative_team,
                team_local_slot,
            ]
        )
        assert pad.currently_alive == bool(
            lifecycle.alive_mask_by_agent_by_team[
                controlled_slot,
                actor_relative_team,
                team_local_slot,
            ]
        )
        assert pad.spawn_shield_remaining == int(
            lifecycle.spawn_shield_actual_durations_by_agent_by_team[
                controlled_slot,
                actor_relative_team,
                team_local_slot,
            ]
        )

    assert (
        tuple(
            (row.team_id, row.period_steps, row.countdown_steps)
            for row in agent_scene.respawn_waves
        )
        == tuple(
            (row.team_id, row.period_steps, row.countdown_steps)
            for row in oracle_scene.respawn_waves
        )
        == ((1, 5, 3), (2, 5, 0))
    )
    for wave in agent_scene.respawn_waves:
        actor_relative_team = 0 if wave.team_id == controlled_team_id else 1
        assert wave.period_steps == int(
            lifecycle.respawn_wave_period_step_count_by_agent_by_team[
                controlled_slot,
                actor_relative_team,
            ]
        )
        assert wave.countdown_steps == int(
            lifecycle.respawn_wave_countdowns_by_agent_by_team[
                controlled_slot,
                actor_relative_team,
            ]
        )

    visible_ids = {row.public_agent_id for row in agent_scene.agents}
    hidden_ids = {row.public_agent_id for row in oracle_scene.agents} - visible_ids
    assert {"5", "6"}.issubset(hidden_ids)
    for pad in agent_scene.spawn_pads:
        oracle_assignee = oracle_pads[
            (pad.team_id, pad.team_local_slot)
        ].assigned_public_agent_id
        if oracle_assignee in hidden_ids:
            assert pad.assigned_public_agent_id is None
            assert pad.assigned_presentation_key is None

    hidden_dead = agent_pads[(2, 0)]
    hidden_shielded = agent_pads[(2, 1)]
    assert (
        hidden_dead.assigned_public_agent_id,
        hidden_dead.currently_alive,
        hidden_dead.spawn_shield_remaining,
    ) == (None, False, 0)
    assert (
        hidden_shielded.assigned_public_agent_id,
        hidden_shielded.currently_alive,
        hidden_shielded.spawn_shield_remaining,
    ) == (None, True, 2)
    assert hidden_dead.position != oracle_scene.agents[5].position
    assert hidden_shielded.position != oracle_scene.agents[6].position


def _boundary_session(*, map_width: float) -> DebuggerSession:
    registered = get_scenario("arena_5v5")
    config, state = registered.build_scenario()
    config = config._replace(map_width=map_width)
    positions = state.agent_positions.at[5].set(
        jnp.asarray((19.5, 1.0), dtype=jnp.float32)
    )
    scenario = replace(
        registered,
        build_scenario=lambda: (
            config,
            state._replace(agent_positions=positions),
        ),
    )
    return create_session(
        scenario,
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=5,
        show_ranges=True,
        verbose_logging=False,
    )


def test_public_map_width_precedes_and_explains_boundary_limited_movement() -> None:
    """The decision input exposes the bound before that bound affects transition."""
    arena = _boundary_session(map_width=20.0)
    wider_counterfactual = _boundary_session(map_width=21.0)

    for session, expected_width in (
        (arena, 20.0),
        (wider_counterfactual, 21.0),
    ):
        active = np.asarray(session.config.agent_profile.active_mask, dtype=bool)
        context = np.asarray(session.observation.context_features)
        np.testing.assert_array_equal(
            context[active, CONTEXT_FEATURE_MAP_WIDTH],
            np.full((int(active.sum()),), expected_width, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            context[active, CONTEXT_FEATURE_MAP_HEIGHT],
            np.full((int(active.sum()),), 10.0, dtype=np.float32),
        )
        assert bool(session.action_mask.move_mask[5, MOVE_EAST])

    for arena_mask, wider_mask in zip(
        arena.action_mask,
        wider_counterfactual.action_mask,
        strict=True,
    ):
        np.testing.assert_array_equal(arena_mask, wider_mask)

    arena_successor = submit_interactive(set_pending_movement(arena, MOVE_EAST))
    wider_successor = submit_interactive(
        set_pending_movement(wider_counterfactual, MOVE_EAST)
    )

    for successor, expected_width in (
        (arena_successor, 20.0),
        (wider_successor, 21.0),
    ):
        incoming = successor.incoming_evaluation_view
        assert incoming is not None
        acceptance = incoming.transition.facts.action_acceptance_facts
        assert int(acceptance.submitted_joint_action.move[5]) == MOVE_EAST
        assert int(acceptance.accepted_joint_action.move[5]) == MOVE_EAST
        assert int(successor.state.step_count) == 1
        assert (
            float(successor.observation.context_features[5, CONTEXT_FEATURE_MAP_WIDTH])
            == expected_width
        )
        assert (
            float(successor.observation.context_features[5, CONTEXT_FEATURE_MAP_HEIGHT])
            == 10.0
        )

    radius = float(arena.config.agent_profile.agent_radii[5])
    assert radius == 0.5
    assert float(arena.state.agent_positions[5, 0]) == 19.5
    assert float(wider_counterfactual.state.agent_positions[5, 0]) == 19.5
    assert float(arena_successor.state.agent_positions[5, 0]) == (
        arena.config.map_width - radius
    )
    assert float(wider_successor.state.agent_positions[5, 0]) == (
        wider_counterfactual.config.map_width - radius
    )
