"""Launch a deterministic geometry scene for renderer/manual-control debugging."""

from __future__ import annotations

import argparse
import os
from importlib import import_module
from importlib.util import find_spec
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    CLASS_NEUTRAL,
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_FEATURES,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.rendering import render_geometry, run_manual_control

type RendererMode = Literal["manual", "static"]
_EFFECTIVELY_UNBOUNDED_MAX_STEPS = 2_147_483_647


def main() -> None:
    """Open the debug renderer in manual-control or static mode."""
    args = _parse_args()
    mode = _resolve_mode(args)
    _require_matplotlib()

    config = _debug_config()
    state = _debug_state()

    if mode == "static":
        _run_static(config, state)
        return

    _run_manual(
        config,
        state,
        controlled_slot=args.controlled_slot,
        step_interval_ms=args.step_interval_ms,
    )


def _parse_args() -> argparse.Namespace:
    """Parse the small dev-harness CLI."""
    parser = argparse.ArgumentParser(
        description="Open a deterministic MARL-BattleGrounds geometry debug scene.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--manual",
        action="store_true",
        help="run the interactive manual-control renderer",
    )
    mode_group.add_argument(
        "--static",
        action="store_true",
        help="render one static geometry snapshot",
    )
    parser.add_argument(
        "--controlled-slot",
        type=int,
        default=int(os.environ.get("CONTROLLED_SLOT", "0")),
        help="global agent slot controlled in manual mode",
    )
    parser.add_argument(
        "--step-interval-ms",
        type=int,
        default=int(os.environ.get("STEP_INTERVAL_MS", "50")),
        help="manual-control timestep interval",
    )
    return parser.parse_args()


def _resolve_mode(args: argparse.Namespace) -> RendererMode:
    """Resolve CLI and environment mode selection."""
    if args.manual:
        return "manual"
    if args.static:
        return "static"

    mode = os.environ.get("MODE", "manual").lower()
    if mode == "manual":
        return "manual"
    if mode == "static":
        return "static"

    raise SystemExit(f"Unsupported MODE={mode!r}; expected 'manual' or 'static'.")


def _empty_obstacles() -> Array:
    """Create a padded all-inactive obstacle table."""
    return jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _pillar_obstacle() -> Array:
    """Create one active pillar row for collision/LOS inspection."""
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(5.2)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(4.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_RADIUS].set(0.85)
    obstacle = obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)
    return obstacle


def _wall_obstacle() -> Array:
    """Create one active rotated wall row for geometry inspection."""
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_WALL)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(7.6)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(3.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_WIDTH].set(2.6)
    obstacle = obstacle.at[OBSTACLE_FEATURE_HEIGHT].set(0.45)
    obstacle = obstacle.at[OBSTACLE_FEATURE_THETA].set(0.55)
    obstacle = obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)
    return obstacle


def _debug_config() -> EnvConfig:
    """Create the static map contract for the debug scene."""
    obstacles = _empty_obstacles()
    obstacles = obstacles.at[0].set(_pillar_obstacle())
    obstacles = obstacles.at[1].set(_wall_obstacle())

    return EnvConfig(
        team_size=2,
        max_steps=_EFFECTIVELY_UNBOUNDED_MAX_STEPS,
        map_width=12.0,
        map_height=8.0,
        default_agent_radius=0.5,
        default_movement_speed=0.35,
        default_observation_radius=8.0,
        default_basic_interaction_radius=6.0,
        default_ultimate_interaction_radius=9.0,
        obstacles=obstacles,
    )


def _debug_state() -> EnvState:
    """Create a fixed initial state for manual geometry checks."""
    positions = jnp.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
    positions = positions.at[0].set(jnp.array((2.0, 2.0), dtype=jnp.float32))
    positions = positions.at[1].set(jnp.array((3.0, 2.7), dtype=jnp.float32))
    positions = positions.at[MAX_AGENTS_PER_TEAM].set(
        jnp.array((10.0, 6.0), dtype=jnp.float32)
    )
    positions = positions.at[MAX_AGENTS_PER_TEAM + 1].set(
        jnp.array((8.8, 5.3), dtype=jnp.float32)
    )

    active_mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=bool)
    active_mask = active_mask.at[0].set(True)
    active_mask = active_mask.at[1].set(True)
    active_mask = active_mask.at[MAX_AGENTS_PER_TEAM].set(True)
    active_mask = active_mask.at[MAX_AGENTS_PER_TEAM + 1].set(True)

    team_ids = jnp.concatenate(
        (
            jnp.zeros((MAX_AGENTS_PER_TEAM,), dtype=jnp.int32),
            jnp.ones((MAX_AGENTS_PER_TEAM,), dtype=jnp.int32),
        ),
        axis=0,
    )

    return EnvState(
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=positions,
        agent_radii=jnp.full((MAX_AGENT_SLOTS,), 0.5, dtype=jnp.float32),
        team_ids=team_ids,
        class_ids=jnp.full((MAX_AGENT_SLOTS,), CLASS_NEUTRAL, dtype=jnp.int32),
        movement_speeds=jnp.full((MAX_AGENT_SLOTS,), 0.35, dtype=jnp.float32),
        observation_radii=jnp.full((MAX_AGENT_SLOTS,), 8.0, dtype=jnp.float32),
        basic_interaction_radii=jnp.full((MAX_AGENT_SLOTS,), 6.0, dtype=jnp.float32),
        ultimate_interaction_radii=jnp.full((MAX_AGENT_SLOTS,), 9.0, dtype=jnp.float32),
        active_mask=active_mask,
        alive_mask=active_mask,
    )


def _require_matplotlib() -> None:
    """Fail with an actionable message if optional viz dependencies are absent."""
    if find_spec("matplotlib.pyplot") is None:
        raise SystemExit(
            "Matplotlib is required for this dev renderer. "
            "Install the optional visualization extra with `uv sync --extra viz`."
        )


def _run_static(config: EnvConfig, state: EnvState) -> None:
    """Open a single static render frame."""
    result = render_geometry(config, state)
    print("Opened static geometry renderer. Close the window to exit.")
    pyplot = import_module("matplotlib.pyplot")
    pyplot.show()
    _ = result


def _run_manual(
    config: EnvConfig,
    state: EnvState,
    *,
    controlled_slot: int,
    step_interval_ms: int,
) -> None:
    """Open the manual-control renderer."""
    print(
        "Opened manual geometry renderer. "
        f"Controlled slot: {controlled_slot}. "
        "Use WASD/QEZC; no input means stay."
    )
    run_manual_control(
        config,
        state,
        jax.random.key(0),
        controlled_slot=controlled_slot,
        step_interval_ms=step_interval_ms,
    )


if __name__ == "__main__":
    main()
