"""Deterministic authored-state scenarios for the visual debugger."""

from collections.abc import Iterator

import jax
import jax.numpy as jnp
from jax import Array

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
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_STAY,
    MOVE_WEST,
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
    EnvState,
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


def _spawn_pad_positions(map_width: float, map_height: float) -> Array:
    """Return two unobstructed edge formations for later wave respawns."""
    y_coordinates = jnp.linspace(
        1.5,
        map_height - 1.5,
        MAX_AGENTS_PER_TEAM,
        dtype=jnp.float32,
    )
    team_a = jnp.stack(
        (jnp.full_like(y_coordinates, 1.5), y_coordinates),
        axis=-1,
    )
    team_b = jnp.stack(
        (jnp.full_like(y_coordinates, map_width - 1.5), y_coordinates),
        axis=-1,
    )
    return jnp.stack((team_a, team_b), axis=0)


def _scenario(
    *,
    map_width: float,
    map_height: float,
    team_sizes: tuple[int, int],
    class_ids: tuple[int, ...],
    active_positions: dict[int, tuple[float, float]],
    obstacles: Array | None = None,
) -> tuple[EnvConfig, EnvState]:
    """Build a validated respawn configuration and authored debugger state."""
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
        task_mode=0,
        team_deathmatch_score_threshold=0,
        max_steps=_MAX_STEPS,
        map_width=map_width,
        map_height=map_height,
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
        agent_profile=profile,
        ordinary_movement_distance_scale=_MOVEMENT_SCALE,
        team_spawn_pad_positions=_spawn_pad_positions(map_width, map_height),
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray((5, 5), dtype=jnp.int32),
    )
    initial_state, _, _, _ = reset(config, jax.random.key(0))
    return config, initial_state._replace(agent_positions=positions)


def _arena_5v5_scenario() -> tuple[EnvConfig, EnvState]:
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
    return _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions=positions,
        obstacles=obstacles,
    )


def _basic_support_scenario() -> tuple[EnvConfig, EnvState]:
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
    return _scenario(
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


def _ultimate_showcase_scenario() -> tuple[EnvConfig, EnvState]:
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
    return _scenario(
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
            6: (8.0, 8.0),
            7: (10.0, 3.0),
            8: (12.0, 8.0),
            9: (13.0, 10.0),
        },
    )


def _aura_crossfire_scenario() -> tuple[EnvConfig, EnvState]:
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
    return _scenario(
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


def _status_stack_scenario() -> tuple[EnvConfig, EnvState]:
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
    return _scenario(
        map_width=14.0,
        map_height=12.0,
        team_sizes=(3, 2),
        class_ids=roster,
        active_positions={
            0: (3.0, 6.0),
            1: (5.5, 4.4),
            2: (8.0, 5.0),
            5: (8.0, 6.0),
            6: (8.0, 8.0),
        },
    )


def _team_focus_crossfire_scenario() -> tuple[EnvConfig, EnvState]:
    roster = (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        NEUTRAL_CLASS_ID,
        WARRIOR_CLASS_ID,
        PRIEST_CLASS_ID,
        PRIEST_CLASS_ID,
        PRIEST_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _scenario(
        map_width=16.0,
        map_height=12.0,
        team_sizes=(4, 4),
        class_ids=roster,
        active_positions={
            0: (6.0, 6.0),
            1: (7.0, 5.0),
            2: (8.0, 3.0),
            3: (8.0, 4.6),
            5: (8.0, 6.0),
            6: (10.5, 6.0),
            7: (9.8, 8.0),
            8: (6.2, 8.0),
        },
    )


def _moving_focus_crossfire_scenario() -> tuple[EnvConfig, EnvState]:
    """Return a radially separated focus-fire fixture for minimum-view review."""
    roster = (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        NEUTRAL_CLASS_ID,
        WARRIOR_CLASS_ID,
        PRIEST_CLASS_ID,
        PRIEST_CLASS_ID,
        PRIEST_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _scenario(
        map_width=16.0,
        map_height=12.0,
        team_sizes=(4, 4),
        class_ids=roster,
        active_positions={
            0: (6.02, 4.02),
            1: (6.52, 6.0),
            2: (6.02, 7.98),
            3: (9.48, 6.0),
            5: (8.0, 6.0),
            6: (9.98, 4.02),
            7: (9.98, 7.98),
            8: (8.0, 8.8),
        },
    )


def _mirrored_ultimates_scenario() -> tuple[EnvConfig, EnvState]:
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
    return _scenario(
        map_width=18.0,
        map_height=14.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions={
            0: (3.0, 2.0),
            1: (6.0, 5.0),
            2: (6.6, 9.0),
            3: (7.3, 12.0),
            4: (4.0, 11.5),
            5: (15.0, 2.0),
            6: (10.0, 5.0),
            7: (9.4, 9.0),
            8: (8.7, 12.0),
            9: (12.0, 11.5),
        },
    )


def _moving_basic_crossfire_scenario() -> tuple[EnvConfig, EnvState]:
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
    return _scenario(
        map_width=14.0,
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions={
            0: (5.0, 4.0),
            1: (5.0, 8.5),
            2: (5.0, 6.0),
            3: (8.7, 8.5),
            4: (3.0, 4.5),
            5: (7.0, 6.0),
            6: (6.3, 8.5),
            7: (7.0, 4.0),
            8: (10.0, 8.5),
            9: (9.0, 5.5),
        },
    )


def _charge_convergence_scenario() -> tuple[EnvConfig, EnvState]:
    roster = (
        WARRIOR_CLASS_ID,
        WARRIOR_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        WARRIOR_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _scenario(
        map_width=14.0,
        map_height=12.0,
        team_sizes=(2, 1),
        class_ids=roster,
        active_positions={
            0: (3.0, 4.0),
            1: (3.0, 8.0),
            5: (8.0, 6.0),
        },
    )


def _trap_lifecycle_scenario() -> tuple[EnvConfig, EnvState]:
    roster = (
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
    )
    return _scenario(
        map_width=12.0,
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions={
            0: (3.0, 2.0),
            1: (3.0, 4.0),
            2: (3.0, 6.0),
            3: (3.0, 8.0),
            4: (4.5, 3.0),
            5: (6.0, 2.0),
            6: (6.0, 4.0),
            7: (6.0, 6.0),
            8: (6.0, 8.0),
            9: (6.0, 10.0),
        },
    )


def _max_status_stack_scenario() -> tuple[EnvConfig, EnvState]:
    roster = (
        MAGE_CLASS_ID,
        PRIEST_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _scenario(
        map_width=16.0,
        map_height=12.0,
        team_sizes=(2, 4),
        class_ids=roster,
        active_positions={
            0: (8.0, 6.0),
            1: (10.5, 7.0),
            5: (8.0, 1.0),
            6: (10.0, 4.0),
            7: (11.0, 6.0),
            8: (8.0, 7.4),
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
        "Priest self-heal exceeds simultaneous Hunter damage by 2 HP.",
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


_TEAM_FOCUS_CROSSFIRE_FRAMES = (
    ScenarioFrame(
        "first-hunter-basic",
        "The Hunter opens repeated fire on the opposing Warrior.",
        (ActorCommand(2, MOVE_STAY, 5, 0),),
    ),
    ScenarioFrame(
        "repeated-hunter-basic",
        "The same Hunter repeats its Basic against the same recipient.",
        (ActorCommand(2, MOVE_STAY, 5, 0),),
    ),
    ScenarioFrame(
        "focus-fire-and-healing",
        "Four attackers focus one Warrior while three Priests heal it.",
        (
            ActorCommand(0, MOVE_STAY, 5, 0),
            ActorCommand(1, MOVE_STAY, 5, 0),
            ActorCommand(2, MOVE_STAY, 5, 0),
            ActorCommand(3, MOVE_STAY, 5, 0),
            ActorCommand(6, MOVE_STAY, 5, 0),
            ActorCommand(7, MOVE_STAY, 5, 0),
            ActorCommand(8, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "poison-and-same-epoch-healing",
        "Poison lands while three precommitted Priest Basics still heal.",
        (
            ActorCommand(3, MOVE_STAY, 5, 1),
            ActorCommand(6, MOVE_STAY, 5, 0),
            ActorCommand(7, MOVE_STAY, 5, 0),
            ActorCommand(8, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "current-anti-heal",
        "Three Priest Basics heal under the now-current Poison anti-heal.",
        (
            ActorCommand(6, MOVE_STAY, 5, 0),
            ActorCommand(7, MOVE_STAY, 5, 0),
            ActorCommand(8, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "three-holy-words",
        "All three Priests use Holy Word into the allied Warrior's health cap.",
        (
            ActorCommand(6, MOVE_STAY, 5, 1),
            ActorCommand(7, MOVE_STAY, 5, 1),
            ActorCommand(8, MOVE_STAY, 5, 1),
        ),
    ),
)

_MIRRORED_ULTIMATES_FRAMES = (
    ScenarioFrame(
        "mirrored-bursts",
        "Both Mages activate Burst simultaneously.",
        (
            ActorCommand(0, MOVE_NORTH, None, 1),
            ActorCommand(5, MOVE_NORTH, None, 1),
        ),
    ),
    ScenarioFrame(
        "reciprocal-charges",
        "The opposing Warriors Charge one another.",
        (
            ActorCommand(1, MOVE_STAY, 6, 1),
            ActorCommand(6, MOVE_STAY, 1, 1),
        ),
    ),
    ScenarioFrame(
        "reciprocal-traps",
        "The opposing Hunters Trap one another.",
        (
            ActorCommand(2, MOVE_NORTH, 7, 1),
            ActorCommand(7, MOVE_NORTH, 2, 1),
        ),
    ),
    ScenarioFrame(
        "reciprocal-poisons",
        "The opposing Rogues Poison one another.",
        (
            ActorCommand(3, MOVE_EAST, 8, 1),
            ActorCommand(8, MOVE_EAST, 3, 1),
        ),
    ),
    ScenarioFrame(
        "mirrored-holy-words",
        "Each Priest uses Holy Word on its allied damaged Rogue.",
        (
            ActorCommand(4, MOVE_NORTH, 3, 1),
            ActorCommand(9, MOVE_NORTH, 8, 1),
        ),
    ),
)

_MOVING_BASIC_CROSSFIRE_FRAMES = (
    ScenarioFrame(
        "north-east-crossfire",
        "Every class pair moves north or east through reciprocal Basics and healing.",
        (
            ActorCommand(0, MOVE_NORTH, 5, 0),
            ActorCommand(1, MOVE_EAST, 6, 0),
            ActorCommand(2, MOVE_NORTH, 7, 0),
            ActorCommand(3, MOVE_EAST, 8, 0),
            ActorCommand(4, MOVE_NORTH, 0, 0),
            ActorCommand(5, MOVE_NORTH, 0, 0),
            ActorCommand(6, MOVE_EAST, 1, 0),
            ActorCommand(7, MOVE_NORTH, 2, 0),
            ActorCommand(8, MOVE_EAST, 3, 0),
            ActorCommand(9, MOVE_NORTH, 5, 0),
        ),
    ),
    ScenarioFrame(
        "south-west-crossfire",
        "The same pairs reverse south or west and repeat their completed effects.",
        (
            ActorCommand(0, MOVE_SOUTH, 5, 0),
            ActorCommand(1, MOVE_WEST, 6, 0),
            ActorCommand(2, MOVE_SOUTH, 7, 0),
            ActorCommand(3, MOVE_WEST, 8, 0),
            ActorCommand(4, MOVE_SOUTH, 0, 0),
            ActorCommand(5, MOVE_SOUTH, 0, 0),
            ActorCommand(6, MOVE_WEST, 1, 0),
            ActorCommand(7, MOVE_SOUTH, 2, 0),
            ActorCommand(8, MOVE_WEST, 3, 0),
            ActorCommand(9, MOVE_SOUTH, 5, 0),
        ),
    ),
)

_MOVING_FOCUS_CROSSFIRE_FRAMES = (
    ScenarioFrame(
        "eastbound-focus-fire-and-healing",
        "Four attackers, three healers, and their shared recipient move east together.",
        (
            ActorCommand(0, MOVE_EAST, 5, 0),
            ActorCommand(1, MOVE_EAST, 5, 0),
            ActorCommand(2, MOVE_EAST, 5, 0),
            ActorCommand(3, MOVE_EAST, 5, 0),
            ActorCommand(5, MOVE_EAST, None, 0),
            ActorCommand(6, MOVE_EAST, 5, 0),
            ActorCommand(7, MOVE_EAST, 5, 0),
            ActorCommand(8, MOVE_EAST, 5, 0),
        ),
    ),
)

_CHARGE_CONVERGENCE_FRAMES = (
    ScenarioFrame(
        "three-converging-charges",
        "Two Warriors Charge one target while that target Charges one back.",
        (
            ActorCommand(0, MOVE_STAY, 5, 1),
            ActorCommand(1, MOVE_STAY, 5, 1),
            ActorCommand(5, MOVE_STAY, 0, 1),
        ),
    ),
)

_TRAP_LIFECYCLE_FRAMES = (
    ScenarioFrame(
        "four-trap-applications",
        "Four Hunters Trap four adjacent Warriors.",
        (
            ActorCommand(0, MOVE_STAY, 5, 1),
            ActorCommand(1, MOVE_STAY, 6, 1),
            ActorCommand(2, MOVE_STAY, 7, 1),
            ActorCommand(3, MOVE_STAY, 8, 1),
        ),
    ),
    ScenarioFrame(
        "exact-trap-break",
        "A Hunter Basic breaks a Trap with more than one tick remaining.",
        (ActorCommand(0, MOVE_STAY, 5, 0),),
    ),
    ScenarioFrame(
        "neutral-aging-transition",
        "A canonical neutral transition ages the remaining Traps.",
        (),
    ),
    ScenarioFrame(
        "trap-reapplication",
        "The fifth Hunter reapplies Trap to the second target.",
        (ActorCommand(4, MOVE_STAY, 6, 1),),
    ),
    ScenarioFrame(
        "ambiguous-end-and-expiry",
        "Damage accompanies one duration-one ending while another expires.",
        (ActorCommand(2, MOVE_STAY, 7, 0),),
    ),
)

_MAX_STATUS_STACK_FRAMES = (
    ScenarioFrame(
        "maximum-compatible-status-stack",
        "All compatible control, slow, modifier, and Burst channels land together.",
        (
            ActorCommand(0, MOVE_STAY, None, 1),
            ActorCommand(1, MOVE_STAY, 0, 0),
            ActorCommand(5, MOVE_STAY, 0, 1),
            ActorCommand(6, MOVE_STAY, 0, 1),
            ActorCommand(7, MOVE_STAY, 0, 0),
            ActorCommand(8, MOVE_STAY, 0, 1),
        ),
    ),
)


RESEARCHER_SCENARIOS: dict[str, DebuggerScenario] = {
    "arena_5v5": DebuggerScenario(
        name="arena_5v5",
        title="5v5 geometry and combat laboratory",
        description=(
            "Interactive LOS, visibility, range, relation, and mask inspection."
        ),
        mode="interactive",
        build_scenario=_arena_5v5_scenario,
        frames=(),
        default_controlled_slot=0,
        audience="researcher",
    ),
    "basic_support": DebuggerScenario(
        name="basic_support",
        title="Basic damage and support",
        description="Scripted simultaneous Basic damage, healing, and passives.",
        mode="scripted",
        build_scenario=_basic_support_scenario,
        frames=_BASIC_SUPPORT_FRAMES,
        default_controlled_slot=0,
        audience="researcher",
    ),
    "ultimate_showcase": DebuggerScenario(
        name="ultimate_showcase",
        title="Five-class Ultimate showcase",
        description="Scripted activation and lifecycle of all class Ultimates.",
        mode="scripted",
        build_scenario=_ultimate_showcase_scenario,
        frames=_ULTIMATE_SHOWCASE_FRAMES,
        default_controlled_slot=0,
        audience="researcher",
    ),
    "aura_crossfire": DebuggerScenario(
        name="aura_crossfire",
        title="Aura crossfire",
        description="Scripted reciprocal Basics under both aura families.",
        mode="scripted",
        build_scenario=_aura_crossfire_scenario,
        frames=_AURA_CROSSFIRE_FRAMES,
        default_controlled_slot=2,
        audience="researcher",
    ),
    "status_stack": DebuggerScenario(
        name="status_stack",
        title="Status composition and lifecycle",
        description="Scripted stacked control, mitigation, break, and movement.",
        mode="scripted",
        build_scenario=_status_stack_scenario,
        frames=_STATUS_STACK_FRAMES,
        default_controlled_slot=5,
        audience="researcher",
    ),
    "team_focus_crossfire": DebuggerScenario(
        name="team_focus_crossfire",
        title="Focus fire and coordinated healing",
        description=(
            "Repeated and simultaneous damage, healing, Poison, and Holy Word."
        ),
        mode="scripted",
        build_scenario=_team_focus_crossfire_scenario,
        frames=_TEAM_FOCUS_CROSSFIRE_FRAMES,
        default_controlled_slot=2,
        audience="researcher",
    ),
    "mirrored_ultimates": DebuggerScenario(
        name="mirrored_ultimates",
        title="Mirrored five-class Ultimates",
        description="Reciprocal and mirrored activation of all Ultimate families.",
        mode="scripted",
        build_scenario=_mirrored_ultimates_scenario,
        frames=_MIRRORED_ULTIMATES_FRAMES,
        default_controlled_slot=0,
        audience="researcher",
    ),
}


STRESS_SCENARIOS: dict[str, DebuggerScenario] = {
    "moving_basic_crossfire": DebuggerScenario(
        name="moving_basic_crossfire",
        title="Moving Basic crossfire",
        description="Reciprocal Basics and healing across moving successor anchors.",
        mode="scripted",
        build_scenario=_moving_basic_crossfire_scenario,
        frames=_MOVING_BASIC_CROSSFIRE_FRAMES,
        default_controlled_slot=0,
        audience="stress",
    ),
    "moving_focus_crossfire": DebuggerScenario(
        name="moving_focus_crossfire",
        title="Moving focus crossfire",
        description="Moving focus fire and healing converge on one recipient.",
        mode="scripted",
        build_scenario=_moving_focus_crossfire_scenario,
        frames=_MOVING_FOCUS_CROSSFIRE_FRAMES,
        default_controlled_slot=2,
        audience="stress",
    ),
    "charge_convergence": DebuggerScenario(
        name="charge_convergence",
        title="Converging Charge routes",
        description="Three simultaneous reciprocal and shared-target Charges.",
        mode="scripted",
        build_scenario=_charge_convergence_scenario,
        frames=_CHARGE_CONVERGENCE_FRAMES,
        default_controlled_slot=0,
        audience="stress",
    ),
    "trap_lifecycle": DebuggerScenario(
        name="trap_lifecycle",
        title="Trap lifecycle stress",
        description=(
            "Exact application, damage break, reapplication, and age-to-zero "
            "status lifecycle."
        ),
        mode="scripted",
        build_scenario=_trap_lifecycle_scenario,
        frames=_TRAP_LIFECYCLE_FRAMES,
        default_controlled_slot=0,
        audience="stress",
    ),
    "max_status_stack": DebuggerScenario(
        name="max_status_stack",
        title="Maximum status density",
        description="All nine compatible status channels on one recipient.",
        mode="scripted",
        build_scenario=_max_status_stack_scenario,
        frames=_MAX_STATUS_STACK_FRAMES,
        default_controlled_slot=0,
        audience="stress",
    ),
}


SCENARIOS: dict[str, DebuggerScenario] = {
    **RESEARCHER_SCENARIOS,
    **STRESS_SCENARIOS,
}


def get_scenario(name: str) -> DebuggerScenario:
    """Return a registered scenario or raise a user-facing value error."""
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        choices = ", ".join(SCENARIOS)
        msg = f"unknown scenario {name!r}; choose one of: {choices}."
        raise ValueError(msg) from exc


def list_scenarios(
    *,
    include_stress: bool = False,
) -> tuple[DebuggerScenario, ...]:
    """Return scenarios in stable launcher order, optionally including stress cases."""
    registry = SCENARIOS if include_stress else RESEARCHER_SCENARIOS
    return tuple(registry.values())


def cycle_scenario_name(
    current_name: str,
    direction: int,
    *,
    include_stress: bool = False,
) -> str:
    """Return the adjacent allowed scenario name in stable cyclic order."""
    if direction not in (-1, 1):
        msg = f"scenario direction must be -1 or 1; got {direction}."
        raise ValueError(msg)
    registry = (
        SCENARIOS
        if include_stress or current_name in STRESS_SCENARIOS
        else RESEARCHER_SCENARIOS
    )
    names = tuple(registry)
    try:
        current_index = names.index(current_name)
    except ValueError as exc:
        msg = f"unknown current scenario {current_name!r}."
        raise ValueError(msg) from exc
    return names[(current_index + direction) % len(names)]


def iter_scenario_summaries(
    *,
    include_stress: bool = False,
) -> Iterator[str]:
    """Yield stable one-line scenario summaries without importing Matplotlib."""
    for scenario in list_scenarios(include_stress=include_stress):
        yield f"{scenario.name:<22} {scenario.mode:<11} {scenario.description}"
