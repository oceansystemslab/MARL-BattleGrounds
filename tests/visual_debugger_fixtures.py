"""Test-only visual-debugger scenarios that intentionally exercise rejection."""

from collections.abc import Sequence

import jax
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
from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_STAY,
    NEUTRAL_CLASS_ID,
    OBSTACLE_FEATURES,
    EnvConfig,
)


def _spawn_pad_positions(map_width: float, map_height: float) -> jax.Array:
    """Return valid fixed pads independent of the authored combat layout."""
    y_coordinates = jnp.linspace(
        1.5,
        map_height - 1.5,
        MAX_AGENTS_PER_TEAM,
        dtype=jnp.float32,
    )
    return jnp.stack(
        (
            jnp.stack(
                (jnp.full_like(y_coordinates, 1.5), y_coordinates),
                axis=-1,
            ),
            jnp.stack(
                (
                    jnp.full_like(y_coordinates, map_width - 1.5),
                    y_coordinates,
                ),
                axis=-1,
            ),
        ),
        axis=0,
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
        ordinary_movement_distance_scale=1.0,
        team_spawn_pad_positions=_spawn_pad_positions(12.0, 12.0),
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
    )
    state, _, _, _ = reset(config, jax.random.key(0))
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
        build_scenario=lambda: (config, state._replace(agent_positions=positions)),
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
