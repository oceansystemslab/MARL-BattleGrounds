"""Deterministic authored-state scenarios for the visual debugger."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.config import (
    CANONICAL_PRODUCT_MOVEMENT_SCALE,
    resolve_agent_profile,
)
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
    SLOW_CHANNEL_WARRIOR_CHARGE,
    STUN_CHANNEL_HUNTER_TRAP,
    WARRIOR_CLASS_ID,
    EnvConfig,
    EnvState,
)
from scripts.dev.visual_debugger.model import (
    ActorCommand,
    DebuggerScenario,
    ScenarioFrame,
)
from scripts.dev.visual_debugger.scenario_catalog import SCENARIO_CATALOG_BY_NAME
from scripts.dev.visual_debugger.scenario_catalog import (
    iter_scenario_summaries as iter_scenario_summaries,
)

_MAX_STEPS = 300


def _registered_scenario(
    name: str,
    *,
    build_scenario: Callable[[], tuple[EnvConfig, EnvState]],
    frames: tuple[ScenarioFrame, ...],
) -> DebuggerScenario:
    """Bind one live constructor to the catalog's canonical launch metadata."""
    metadata = SCENARIO_CATALOG_BY_NAME[name]
    return DebuggerScenario(
        name=metadata.name,
        title=metadata.title,
        description=metadata.description,
        mode=metadata.mode,
        build_scenario=build_scenario,
        frames=frames,
        default_controlled_slot=metadata.default_controlled_slot,
        audience=metadata.audience,
    )


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
        max_steps=_MAX_STEPS,
        map_width=map_width,
        map_height=map_height,
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
        agent_profile=profile,
        ordinary_movement_distance_scale=CANONICAL_PRODUCT_MOVEMENT_SCALE,
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
        map_width=18.0,
        map_height=12.0,
        team_sizes=(3, 3),
        class_ids=roster,
        active_positions={
            0: (6.0, 3.0),
            1: (6.0, 6.0),
            2: (6.0, 9.0),
            5: (9.0, 3.0),
            6: (9.0, 6.0),
            7: (9.0, 9.0),
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
        map_width=18.0,
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions={
            0: (4.0, 2.0),
            1: (6.0, 5.0),
            2: (6.0, 8.0),
            3: (9.0, 5.0),
            4: (4.0, 10.0),
            5: (8.0, 6.0),
            6: (9.0, 8.0),
            7: (11.0, 3.0),
            8: (13.0, 8.0),
            9: (14.0, 10.0),
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
        map_width=18.0,
        map_height=12.0,
        team_sizes=(3, 3),
        class_ids=roster,
        active_positions={
            0: (6.0, 5.0),
            1: (6.0, 7.0),
            2: (7.5, 6.0),
            5: (12.0, 5.0),
            6: (12.0, 7.0),
            7: (10.5, 6.0),
        },
    )


def _stacked_team_auras_scenario() -> tuple[EnvConfig, EnvState]:
    roster = (
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
    )
    return _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions={
            0: (6.0, 5.0),
            1: (6.0, 7.0),
            2: (7.5, 4.5),
            3: (7.5, 7.5),
            4: (7.5, 6.0),
            5: (12.0, 5.0),
            6: (12.0, 7.0),
            7: (10.5, 4.5),
            8: (10.5, 7.5),
            9: (10.5, 6.0),
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
        map_width=18.0,
        map_height=12.0,
        team_sizes=(3, 2),
        class_ids=roster,
        active_positions={
            0: (5.0, 6.0),
            1: (7.5, 4.4),
            2: (10.0, 5.0),
            5: (10.0, 6.0),
            6: (10.0, 8.0),
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
        map_width=18.0,
        map_height=12.0,
        team_sizes=(4, 4),
        class_ids=roster,
        active_positions={
            0: (7.0, 6.0),
            1: (8.0, 5.0),
            2: (9.0, 3.0),
            3: (9.0, 4.6),
            5: (9.0, 6.0),
            6: (11.5, 6.0),
            7: (10.8, 8.0),
            8: (7.2, 8.0),
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
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions={
            0: (3.0, 1.0),
            1: (6.0, 4.0),
            2: (6.6, 8.0),
            3: (7.3, 11.0),
            4: (4.0, 10.5),
            5: (15.0, 1.0),
            6: (10.0, 4.0),
            7: (9.4, 8.0),
            8: (8.7, 11.0),
            9: (12.0, 10.5),
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


def _death_respawn_cycle_scenario() -> tuple[EnvConfig, EnvState]:
    """Start immediately before a multi-source lethal and later shielded respawn."""
    roster = (
        MAGE_CLASS_ID,
        HUNTER_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        ROGUE_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    config, state = _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(2, 1),
        class_ids=roster,
        active_positions={
            0: (13.0, 1.5),
            1: (13.0, 3.0),
            5: (14.5, 2.25),
        },
    )
    return config, state._replace(
        current_health=state.current_health.at[5].set(5.0),
        slow_durations=state.slow_durations.at[
            5,
            SLOW_CHANNEL_WARRIOR_CHARGE,
        ].set(3),
        stun_durations=state.stun_durations.at[
            5,
            STUN_CHANNEL_HUNTER_TRAP,
        ].set(3),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[5].set(3)
        ),
        team_respawn_wave_countdowns=jnp.asarray((4, 2), dtype=jnp.int32),
    )


def _recovery_refresh_cycle_scenario() -> tuple[EnvConfig, EnvState]:
    """Start before recovery, cooldown-ready, refresh, and reapply trajectories."""
    roster = (
        ROGUE_CLASS_ID,
        ROGUE_CLASS_ID,
        HUNTER_CLASS_ID,
        HUNTER_CLASS_ID,
        MAGE_CLASS_ID,
        HUNTER_CLASS_ID,
        PRIEST_CLASS_ID,
        WARRIOR_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    config, state = _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(5, 3),
        class_ids=roster,
        active_positions={
            0: (8.0, 4.5),
            1: (8.0, 5.5),
            2: (6.7, 7.0),
            3: (7.5, 8.0),
            4: (8.5, 7.0),
            5: (9.2, 5.0),
            6: (13.0, 9.0),
            7: (9.5, 7.5),
        },
    )
    return config, state._replace(
        current_health=state.current_health.at[6].set(50.0),
        ultimate_cooldowns=state.ultimate_cooldowns.at[6].set(1),
        steps_until_out_of_combat=state.steps_until_out_of_combat.at[6].set(0),
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
        "Hunter Basic breaks the active Freezing Trap and applies its slow.",
        (ActorCommand(2, MOVE_STAY, 6, 0),),
    ),
)

_AURA_CROSSFIRE_FRAMES = (
    ScenarioFrame(
        "reciprocal-hunter-basics",
        "Both Hunters fire through simultaneous Sorcerer\u2019s Empowerment and "
        "Guardian\u2019s Barrier auras.",
        (
            ActorCommand(2, MOVE_STAY, 7, 0),
            ActorCommand(7, MOVE_STAY, 2, 0),
        ),
    ),
)

_STACKED_TEAM_AURAS_FRAMES = (
    ScenarioFrame(
        "reciprocal-stacked-aura-basics",
        "Both Hunters fire through two same-team Mage and Warrior emitters.",
        (
            ActorCommand(4, MOVE_STAY, 9, 0),
            ActorCommand(9, MOVE_STAY, 4, 0),
        ),
    ),
)

_STATUS_STACK_FRAMES = (
    ScenarioFrame(
        "stack",
        "Charge, Freezing Trap, Crippling Poison, and Freedom land on one recipient.",
        (
            ActorCommand(0, MOVE_NORTH, 5, 1),
            ActorCommand(1, MOVE_STAY, 5, 1),
            ActorCommand(2, MOVE_STAY, 5, 1),
            ActorCommand(6, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "break-and-refresh",
        "Hunter Basic breaks Freezing Trap while the stunned recipient stays still.",
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
        "Crippling Poison lands while three precommitted Priest Basics still heal.",
        (
            ActorCommand(3, MOVE_STAY, 5, 1),
            ActorCommand(6, MOVE_STAY, 5, 0),
            ActorCommand(7, MOVE_STAY, 5, 0),
            ActorCommand(8, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "current-anti-heal",
        "Three Priest Basics heal under the now-current Crippling Poison anti-heal.",
        (
            ActorCommand(6, MOVE_STAY, 5, 0),
            ActorCommand(7, MOVE_STAY, 5, 0),
            ActorCommand(8, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "three-holy-words",
        "All three Priests use Holy Word: Salvation into the allied Warrior's "
        "health cap.",
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
        "The opposing Hunters use Freezing Trap on one another.",
        (
            ActorCommand(2, MOVE_NORTH, 7, 1),
            ActorCommand(7, MOVE_NORTH, 2, 1),
        ),
    ),
    ScenarioFrame(
        "reciprocal-poisons",
        "The opposing Rogues use Crippling Poison on one another.",
        (
            ActorCommand(3, MOVE_EAST, 8, 1),
            ActorCommand(8, MOVE_EAST, 3, 1),
        ),
    ),
    ScenarioFrame(
        "mirrored-holy-words",
        "Each Priest uses Holy Word: Salvation on its allied damaged Rogue.",
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
        "Four Hunters use Freezing Trap on four adjacent Warriors.",
        (
            ActorCommand(0, MOVE_STAY, 5, 1),
            ActorCommand(1, MOVE_STAY, 6, 1),
            ActorCommand(2, MOVE_STAY, 7, 1),
            ActorCommand(3, MOVE_STAY, 8, 1),
        ),
    ),
    ScenarioFrame(
        "exact-trap-break",
        "A Hunter Basic breaks a Freezing Trap with more than one tick remaining.",
        (ActorCommand(0, MOVE_STAY, 5, 0),),
    ),
    ScenarioFrame(
        "neutral-aging-transition",
        "A canonical neutral transition ages the remaining Freezing Traps.",
        (),
    ),
    ScenarioFrame(
        "trap-reapplication",
        "The fifth Hunter reapplies Freezing Trap to the second target.",
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

_DEATH_RESPAWN_CYCLE_FRAMES = (
    ScenarioFrame(
        "multi-source-lethal",
        "Mage and Hunter contributions kill a statused Rogue.",
        (
            ActorCommand(0, MOVE_STAY, 5, 0),
            ActorCommand(1, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "corpse-waits-for-wave",
        "The corpse remains out until its team's next due wave.",
        (),
    ),
    ScenarioFrame(
        "due-wave-respawn",
        "A corpse submission is rejected while the due wave restores it.",
        (ActorCommand(5, MOVE_WEST, 0, 1),),
    ),
    ScenarioFrame(
        "shielded-movement-and-rejection",
        "The protected Rogue may move while combat intent is rejected.",
        (ActorCommand(5, MOVE_WEST, 0, 0),),
    ),
    ScenarioFrame(
        "shield-countdown-two-to-one",
        "The invulnerable shield remains visible with one tick left.",
        (ActorCommand(5, MOVE_STAY, 0, 0),),
    ),
    ScenarioFrame(
        "shield-expiry",
        "Transition-start protection still rejects combat before expiring.",
        (ActorCommand(5, MOVE_STAY, 0, 0),),
    ),
    ScenarioFrame(
        "first-unshielded-interaction",
        "The Rogue's first post-shield Basic is accepted normally.",
        (ActorCommand(5, MOVE_STAY, 0, 0),),
    ),
)

_RECOVERY_REFRESH_CYCLE_FRAMES = (
    ScenarioFrame(
        "application-recovery-and-readiness",
        "Crippling Poison and Freezing Trap apply while a damaged Priest "
        "regenerates and becomes ready.",
        (
            ActorCommand(0, MOVE_STAY, 5, 1),
            ActorCommand(2, MOVE_STAY, 7, 1),
            ActorCommand(6, MOVE_STAY, 6, 1),
        ),
    ),
    ScenarioFrame(
        "refresh-and-break-reapplication",
        "Crippling Poison refreshes while Mage damage breaks and Hunter reapplies "
        "Freezing Trap.",
        (
            ActorCommand(1, MOVE_STAY, 5, 1),
            ActorCommand(3, MOVE_STAY, 7, 1),
            ActorCommand(4, MOVE_STAY, 7, 0),
        ),
    ),
    ScenarioFrame("age-one", "Current effects age without new applications.", ()),
    ScenarioFrame("age-two", "Current effects continue their public duration.", ()),
    ScenarioFrame(
        "age-three", "Freezing Trap reaches its final active decision epoch.", ()
    ),
    ScenarioFrame("trap-expiry", "Freezing Trap ages from one tick to zero.", ()),
    ScenarioFrame(
        "poison-expiry", "Crippling Poison slow ages from one tick to zero.", ()
    ),
)


RESEARCHER_SCENARIOS: dict[str, DebuggerScenario] = {
    "arena_5v5": _registered_scenario(
        "arena_5v5",
        build_scenario=_arena_5v5_scenario,
        frames=(),
    ),
    "basic_support": _registered_scenario(
        "basic_support",
        build_scenario=_basic_support_scenario,
        frames=_BASIC_SUPPORT_FRAMES,
    ),
    "ultimate_showcase": _registered_scenario(
        "ultimate_showcase",
        build_scenario=_ultimate_showcase_scenario,
        frames=_ULTIMATE_SHOWCASE_FRAMES,
    ),
    "aura_crossfire": _registered_scenario(
        "aura_crossfire",
        build_scenario=_aura_crossfire_scenario,
        frames=_AURA_CROSSFIRE_FRAMES,
    ),
    "stacked_team_auras": _registered_scenario(
        "stacked_team_auras",
        build_scenario=_stacked_team_auras_scenario,
        frames=_STACKED_TEAM_AURAS_FRAMES,
    ),
    "status_stack": _registered_scenario(
        "status_stack",
        build_scenario=_status_stack_scenario,
        frames=_STATUS_STACK_FRAMES,
    ),
    "team_focus_crossfire": _registered_scenario(
        "team_focus_crossfire",
        build_scenario=_team_focus_crossfire_scenario,
        frames=_TEAM_FOCUS_CROSSFIRE_FRAMES,
    ),
    "mirrored_ultimates": _registered_scenario(
        "mirrored_ultimates",
        build_scenario=_mirrored_ultimates_scenario,
        frames=_MIRRORED_ULTIMATES_FRAMES,
    ),
    "death_respawn_cycle": _registered_scenario(
        "death_respawn_cycle",
        build_scenario=_death_respawn_cycle_scenario,
        frames=_DEATH_RESPAWN_CYCLE_FRAMES,
    ),
    "recovery_refresh_cycle": _registered_scenario(
        "recovery_refresh_cycle",
        build_scenario=_recovery_refresh_cycle_scenario,
        frames=_RECOVERY_REFRESH_CYCLE_FRAMES,
    ),
}


STRESS_SCENARIOS: dict[str, DebuggerScenario] = {
    "moving_basic_crossfire": _registered_scenario(
        "moving_basic_crossfire",
        build_scenario=_moving_basic_crossfire_scenario,
        frames=_MOVING_BASIC_CROSSFIRE_FRAMES,
    ),
    "moving_focus_crossfire": _registered_scenario(
        "moving_focus_crossfire",
        build_scenario=_moving_focus_crossfire_scenario,
        frames=_MOVING_FOCUS_CROSSFIRE_FRAMES,
    ),
    "charge_convergence": _registered_scenario(
        "charge_convergence",
        build_scenario=_charge_convergence_scenario,
        frames=_CHARGE_CONVERGENCE_FRAMES,
    ),
    "trap_lifecycle": _registered_scenario(
        "trap_lifecycle",
        build_scenario=_trap_lifecycle_scenario,
        frames=_TRAP_LIFECYCLE_FRAMES,
    ),
    "max_status_stack": _registered_scenario(
        "max_status_stack",
        build_scenario=_max_status_stack_scenario,
        frames=_MAX_STATUS_STACK_FRAMES,
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
