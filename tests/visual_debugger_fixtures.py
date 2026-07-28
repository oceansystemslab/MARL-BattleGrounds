"""Test-only visual-debugger scenarios that intentionally exercise rejection."""

from collections.abc import Sequence

import jax.numpy as jnp
from scripts.dev.visual_debugger.control import (
    build_scripted_joint_action,
    submit_joint_action,
)
from scripts.dev.visual_debugger.model import (
    ActorCommand,
    DebuggerScenario,
    DebuggerSession,
    ScenarioFrame,
)

from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_STAY,
    NEUTRAL_CLASS_ID,
    OBSTACLE_FEATURES,
    EnvConfig,
)


def rejection_lane_scenario() -> DebuggerScenario:
    """Return an unregistered boundary fixture with one rejection then acceptance."""
    roster = jnp.full((MAX_AGENT_SLOTS,), NEUTRAL_CLASS_ID, dtype=jnp.int32)
    roster = roster.at[0].set(HUNTER_CLASS_ID)
    roster = roster.at[5].set(MAGE_CLASS_ID)
    profile = resolve_agent_profile(
        roster,
        jnp.asarray((1, 1), dtype=jnp.int32),
    )
    positions = jnp.zeros(
        (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        dtype=jnp.float32,
    )
    positions = positions.at[0].set(jnp.asarray((3.0, 6.0), dtype=jnp.float32))
    positions = positions.at[5].set(jnp.asarray((7.0, 6.0), dtype=jnp.float32))
    config = EnvConfig(
        max_steps=20,
        map_width=12.0,
        map_height=12.0,
        obstacles=jnp.zeros(
            (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
            dtype=jnp.float32,
        ),
        agent_profile=profile,
        initial_agent_positions=positions,
        ordinary_movement_distance_scale=1.0,
    )
    frames = (
        ScenarioFrame(
            "rejected-basic-with-movement",
            "Movement is accepted while the out-of-range Basic is rejected.",
            (ActorCommand(0, MOVE_EAST, 5, 0),),
        ),
        ScenarioFrame(
            "accepted-basic-after-approach",
            "The same Basic is accepted from the next decision epoch.",
            (ActorCommand(0, MOVE_STAY, 5, 0),),
        ),
    )
    return DebuggerScenario(
        name="rejection_lane_fixture",
        title="Test-only movement and combat rejection boundary",
        description="Not exposed by the debugger launcher.",
        mode="scripted",
        build_config=lambda: config,
        frames=frames,
        default_controlled_slot=0,
    )


def submit_fixture_frame(
    session: DebuggerSession,
    frame: ScenarioFrame,
) -> DebuggerSession:
    """Submit one test-only frame without consulting the user scenario registry."""
    action = build_scripted_joint_action(session.config, frame)
    report_slots: Sequence[int] = sorted(
        command.actor_global_slot for command in frame.commands
    )
    return submit_joint_action(
        session,
        action,
        submission_kind="scripted",
        report_actor_slots=tuple(report_slots),
    )
