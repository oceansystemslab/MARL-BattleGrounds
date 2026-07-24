"""Deterministic ordinary-reset scenarios for the Milestone 5 debugger."""

from collections.abc import Iterator

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.config import resolve_agent_profile, validate_env_config
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_STAY,
    NEUTRAL_CLASS_ID,
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
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    EnvConfig,
)
from scripts.dev.visual_debugger.model import (
    ActorCommand,
    DebuggerScenario,
    ScenarioFrame,
)

_MAX_STEPS = 300
_MOVEMENT_SCALE = 1.0


def _empty_obstacles() -> Array:
    return jnp.zeros(
        (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
        dtype=jnp.float32,
    )


def _pillar(x: float, y: float, radius: float) -> Array:
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(x)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(y)
    obstacle = obstacle.at[OBSTACLE_FEATURE_RADIUS].set(radius)
    return obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)


def _wall(
    x: float,
    y: float,
    width: float,
    height: float,
    theta: float,
) -> Array:
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_WALL)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(x)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(y)
    obstacle = obstacle.at[OBSTACLE_FEATURE_WIDTH].set(width)
    obstacle = obstacle.at[OBSTACLE_FEATURE_HEIGHT].set(height)
    obstacle = obstacle.at[OBSTACLE_FEATURE_THETA].set(theta)
    return obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)


def _config(
    *,
    map_width: float,
    map_height: float,
    team_sizes: tuple[int, int],
    class_ids: tuple[int, ...],
    active_positions: dict[int, tuple[float, float]],
    obstacles: Array | None = None,
) -> EnvConfig:
    if len(class_ids) != MAX_AGENT_SLOTS:
        msg = f"class_ids must contain {MAX_AGENT_SLOTS} fixed-slot values."
        raise ValueError(msg)

    profile = resolve_agent_profile(
        jnp.asarray(class_ids, dtype=jnp.int32),
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )
    positions = jnp.zeros(
        (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        dtype=jnp.float32,
    )
    for global_slot, position in active_positions.items():
        positions = positions.at[global_slot].set(
            jnp.asarray(position, dtype=jnp.float32)
        )

    config = EnvConfig(
        max_steps=_MAX_STEPS,
        map_width=map_width,
        map_height=map_height,
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
        agent_profile=profile,
        initial_agent_positions=positions,
        ordinary_movement_distance_scale=_MOVEMENT_SCALE,
    )
    validate_env_config(config)
    return config


def _arena_5v5_config() -> EnvConfig:
    obstacles = _empty_obstacles()
    obstacles = obstacles.at[0].set(_pillar(9.0, 3.0, 0.9))
    obstacles = obstacles.at[1].set(_wall(9.0, 7.8, 3.0, 0.5, 0.45))
    roster = (
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
    )
    positions = {
        0: (3.0, 2.0),
        1: (3.0, 4.0),
        2: (3.0, 6.0),
        3: (3.0, 8.0),
        4: (3.0, 10.0),
        5: (15.0, 10.0),
        6: (15.0, 8.0),
        7: (15.0, 6.0),
        8: (15.0, 4.0),
        9: (15.0, 2.0),
    }
    return _config(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions=positions,
        obstacles=obstacles,
    )


def _basic_support_config() -> EnvConfig:
    roster = (
        MAGE_CLASS_ID,
        HUNTER_CLASS_ID,
        PRIEST_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _config(
        map_width=14.0,
        map_height=12.0,
        team_sizes=(3, 3),
        class_ids=roster,
        active_positions={
            0: (4.0, 3.0),
            1: (4.0, 6.0),
            2: (4.0, 9.0),
            5: (7.0, 3.0),
            6: (7.0, 6.0),
            7: (7.0, 9.0),
        },
    )


def _ultimate_showcase_config() -> EnvConfig:
    roster = (
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
    )
    return _config(
        map_width=16.0,
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions={
            0: (3.0, 2.0),
            1: (5.0, 5.0),
            2: (5.0, 8.0),
            3: (8.0, 5.0),
            4: (3.0, 10.0),
            5: (7.0, 6.0),
            6: (8.5, 8.0),
            7: (10.0, 3.0),
            8: (12.0, 8.0),
            9: (13.0, 10.0),
        },
    )


def _aura_crossfire_config() -> EnvConfig:
    roster = (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _config(
        map_width=14.0,
        map_height=12.0,
        team_sizes=(3, 3),
        class_ids=roster,
        active_positions={
            0: (4.0, 5.0),
            1: (4.0, 7.0),
            2: (5.5, 6.0),
            5: (10.0, 5.0),
            6: (10.0, 7.0),
            7: (8.5, 6.0),
        },
    )


def _status_stack_config() -> EnvConfig:
    roster = (
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        HUNTER_CLASS_ID,
        PRIEST_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _config(
        map_width=14.0,
        map_height=12.0,
        team_sizes=(3, 2),
        class_ids=roster,
        active_positions={
            0: (3.0, 6.0),
            1: (5.2, 4.2),
            2: (8.0, 5.0),
            5: (8.0, 6.0),
            6: (8.0, 8.0),
        },
    )


_BASIC_SUPPORT_FRAMES = (
    ScenarioFrame(
        "three-basics",
        "Mage and Hunters apply simultaneous Basic effects.",
        (
            ActorCommand(0, MOVE_STAY, 5, 0),
            ActorCommand(1, MOVE_STAY, 6, 0),
            ActorCommand(7, MOVE_STAY, 2, 0),
        ),
    ),
    ScenarioFrame(
        "simultaneous-heal-damage",
        "Priest self-heal and Hunter damage produce zero net health.",
        (
            ActorCommand(2, MOVE_STAY, 2, 0),
            ActorCommand(7, MOVE_STAY, 2, 0),
        ),
    ),
)

_ULTIMATE_SHOWCASE_FRAMES = (
    ScenarioFrame(
        "mage-basic",
        "The opposing Mage damages the allied Hunter before the showcase.",
        (ActorCommand(5, MOVE_STAY, 2, 0),),
    ),
    ScenarioFrame(
        "five-ultimates",
        "All five Team A classes activate their Ultimate simultaneously.",
        (
            ActorCommand(0, MOVE_STAY, None, 1),
            ActorCommand(1, MOVE_STAY, 7, 1),
            ActorCommand(2, MOVE_STAY, 6, 1),
            ActorCommand(3, MOVE_STAY, 5, 1),
            ActorCommand(4, MOVE_STAY, 2, 1),
        ),
    ),
    ScenarioFrame(
        "break-trap",
        "Hunter Basic breaks the active Trap and applies its slow.",
        (ActorCommand(2, MOVE_STAY, 6, 0),),
    ),
)

_AURA_CROSSFIRE_FRAMES = (
    ScenarioFrame(
        "reciprocal-hunter-basics",
        "Both Hunters fire through simultaneous Mage and Warrior auras.",
        (
            ActorCommand(2, MOVE_STAY, 7, 0),
            ActorCommand(7, MOVE_STAY, 2, 0),
        ),
    ),
)

_STATUS_STACK_FRAMES = (
    ScenarioFrame(
        "stack",
        "Charge, Trap, Poison, and Freedom land on one recipient.",
        (
            ActorCommand(0, MOVE_NORTH, 5, 1),
            ActorCommand(1, MOVE_STAY, 5, 1),
            ActorCommand(2, MOVE_STAY, 5, 1),
            ActorCommand(6, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "break-and-refresh",
        "Hunter Basic breaks Trap while the stunned recipient stays still.",
        (
            ActorCommand(1, MOVE_STAY, 5, 0),
            ActorCommand(5, MOVE_STAY, None, 0),
            ActorCommand(6, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "freedom-speed",
        "The recipient moves with the current Freedom speed floor.",
        (ActorCommand(5, MOVE_EAST, None, 0),),
    ),
    ScenarioFrame(
        "stacked-slow-speed",
        "The recipient moves under the remaining multiplicative slows.",
        (ActorCommand(5, MOVE_EAST, None, 0),),
    ),
)

SCENARIOS: dict[str, DebuggerScenario] = {
    "arena_5v5": DebuggerScenario(
        name="arena_5v5",
        title="5v5 geometry and combat laboratory",
        description=(
            "Interactive LOS, visibility, range, relation, and mask inspection."
        ),
        mode="interactive",
        build_config=_arena_5v5_config,
        frames=(),
        default_controlled_slot=0,
    ),
    "basic_support": DebuggerScenario(
        name="basic_support",
        title="Basic damage and support",
        description="Scripted simultaneous Basic damage, healing, and passives.",
        mode="scripted",
        build_config=_basic_support_config,
        frames=_BASIC_SUPPORT_FRAMES,
        default_controlled_slot=0,
    ),
    "ultimate_showcase": DebuggerScenario(
        name="ultimate_showcase",
        title="Five-class Ultimate showcase",
        description="Scripted activation and lifecycle of all class Ultimates.",
        mode="scripted",
        build_config=_ultimate_showcase_config,
        frames=_ULTIMATE_SHOWCASE_FRAMES,
        default_controlled_slot=0,
    ),
    "aura_crossfire": DebuggerScenario(
        name="aura_crossfire",
        title="Aura crossfire",
        description="Scripted reciprocal Basics under both aura families.",
        mode="scripted",
        build_config=_aura_crossfire_config,
        frames=_AURA_CROSSFIRE_FRAMES,
        default_controlled_slot=2,
    ),
    "status_stack": DebuggerScenario(
        name="status_stack",
        title="Status composition and lifecycle",
        description="Scripted stacked control, mitigation, break, and movement.",
        mode="scripted",
        build_config=_status_stack_config,
        frames=_STATUS_STACK_FRAMES,
        default_controlled_slot=5,
    ),
}


def get_scenario(name: str) -> DebuggerScenario:
    """Return a registered scenario or raise a user-facing value error."""
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        choices = ", ".join(SCENARIOS)
        msg = f"unknown scenario {name!r}; choose one of: {choices}."
        raise ValueError(msg) from exc


def list_scenarios() -> tuple[DebuggerScenario, ...]:
    """Return scenarios in stable launcher order."""
    return tuple(SCENARIOS.values())


def cycle_scenario_name(current_name: str, direction: int) -> str:
    """Return the adjacent scenario name in stable cyclic order."""
    if direction not in (-1, 1):
        msg = f"scenario direction must be -1 or 1; got {direction}."
        raise ValueError(msg)
    names = tuple(SCENARIOS)
    try:
        current_index = names.index(current_name)
    except ValueError as exc:
        msg = f"unknown current scenario {current_name!r}."
        raise ValueError(msg) from exc
    return names[(current_index + direction) % len(names)]


def iter_scenario_summaries() -> Iterator[str]:
    """Yield stable one-line scenario summaries without importing Matplotlib."""
    for scenario in SCENARIOS.values():
        yield f"{scenario.name:<22} {scenario.mode:<11} {scenario.description}"
