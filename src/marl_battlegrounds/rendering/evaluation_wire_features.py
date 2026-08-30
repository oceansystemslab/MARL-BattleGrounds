"""Version-bound feature columns consumed by offline evaluation rendering.

These values name columns in the already-published V1 evaluation observation
wire record.  They intentionally do not import the live simulator module.  A
future observation layout change must add a new evaluation/scene version rather
than silently changing loaded-replay presentation.

The decoder in this module is deliberately lossless with respect to the facts
needed by authorized presentation.  It validates exact wire booleans and
integer-valued counters, but it does not infer identities, visibility, status
sources, or simulator configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final

from marl_battlegrounds.evaluation.wire_shapes import SELF_FEATURES_V1

AGENT_FEATURE_X_V1: Final = 0
AGENT_FEATURE_Y_V1: Final = 1
AGENT_FEATURE_RADIUS_V1: Final = 2
AGENT_FEATURE_TEAM_ID_V1: Final = 3
AGENT_FEATURE_ACTIVE_V1: Final = 4
AGENT_FEATURE_ALIVE_V1: Final = 5
AGENT_FEATURE_CLASS_ID_V1: Final = 6
AGENT_FEATURE_BASE_MOVEMENT_SPEED_V1: Final = 7
AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1: Final = 8
AGENT_FEATURE_OBSERVATION_RADIUS_V1: Final = 9
AGENT_FEATURE_BASIC_INTERACTION_RADIUS_V1: Final = 10
AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS_V1: Final = 11
AGENT_FEATURE_CURRENT_HEALTH_V1: Final = 12
AGENT_FEATURE_MAX_HEALTH_V1: Final = 13
AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1: Final = 14

AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION_V1: Final = 15
AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION_V1: Final = 16
AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION_V1: Final = 17
AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1: Final = 18
AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER_V1: Final = 19
AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER_V1: Final = 20
AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1: Final = 21
AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION_V1: Final = 22
AGENT_FEATURE_STUN_ROGUE_POISON_DURATION_V1: Final = 23
AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION_V1: Final = 24
AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1: Final = 25
AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1: Final = 26
AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1: Final = 27
AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1: Final = 28
AGENT_STATUS_FEATURE_START_V1: Final = 15
AGENT_STATUS_FEATURE_STOP_V1: Final = 29

AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT_V1: Final = 29
AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1: Final = 30
AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1: Final = 31

AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE_V1: Final = 32
AGENT_FEATURE_CAPABILITY_BASIC_HEALING_V1: Final = 33
AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION_V1: Final = 34
AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION_V1: Final = 35
AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION_V1: Final = 36
AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION_V1: Final = 37
AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1: Final = 38
AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER_V1: Final = 39
AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER_V1: Final = 40
AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION_V1: Final = 41
AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION_V1: Final = 42
AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION_V1: Final = 43
AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_DURATION_V1: Final = 44
AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1: Final = 45
AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1: Final = 46
AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER_V1: Final = 47
AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1: Final = 48
AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1: Final = 49
AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS_V1: Final = 50
AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1: Final = 51
AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS_V1: Final = 52
AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1: Final = 53
AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING_V1: Final = 54
AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE_V1: Final = 55
AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS_V1: Final = 56
AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP_V1: Final = 57

# Scientific status channels are not contiguous in presentation order.  These
# tuples name the exact V1 wire columns without importing simulator constants.
AGENT_STATUS_REMAINING_DURATION_COLUMN_BY_CHANNEL_V1: Final = (
    AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION_V1,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION_V1,
    AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION_V1,
    AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1,
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION_V1,
    AGENT_FEATURE_STUN_ROGUE_POISON_DURATION_V1,
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION_V1,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1,
    AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1,
)
AGENT_STATUS_ACTIVE_MAGNITUDE_COLUMN_BY_CHANNEL_V1: Final = (
    AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER_V1,
    AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER_V1,
    None,
    None,
    None,
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1,
    None,
    AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1,
)
AGENT_STATUS_CAPABILITY_DURATION_COLUMN_BY_CHANNEL_V1: Final = (
    AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1,
    AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1,
)
AGENT_STATUS_CAPABILITY_MAGNITUDE_COLUMN_BY_CHANNEL_V1: Final = (
    AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1,
    AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER_V1,
    AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER_V1,
    None,
    None,
    None,
    AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER_V1,
    AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1,
)

CONTEXT_FEATURE_MAP_WIDTH_V1: Final = 2
CONTEXT_FEATURE_MAP_HEIGHT_V1: Final = 3
OBSTACLE_FEATURE_TYPE_V1: Final = 0
OBSTACLE_FEATURE_X_V1: Final = 1
OBSTACLE_FEATURE_Y_V1: Final = 2
OBSTACLE_FEATURE_RADIUS_V1: Final = 3
OBSTACLE_FEATURE_WIDTH_V1: Final = 4
OBSTACLE_FEATURE_HEIGHT_V1: Final = 5
OBSTACLE_FEATURE_THETA_V1: Final = 6
OBSTACLE_FEATURE_ACTIVE_V1: Final = 7


def _wire_bool(value: float, *, name: str) -> bool:
    if type(value) is not float or value not in (0.0, 1.0):
        raise ValueError(f"{name} must be the exact wire float 0.0 or 1.0.")
    return value == 1.0


def _wire_int(value: float, *, name: str, minimum: int = 0) -> int:
    if type(value) is not float or not isfinite(value) or not value.is_integer():
        raise ValueError(f"{name} must be an integer-valued finite wire float.")
    decoded = int(value)
    if decoded < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return decoded


def _finite(value: float, *, name: str, minimum: float | None = None) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite Python float.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodedAgentFeatureRowV1:
    """Typed scalar projection of one exact 58-column V1 agent row."""

    position: tuple[float, float]
    radius: float
    team_id: int
    configured_active: bool
    alive: bool
    class_id: int
    base_movement_speed: float
    effective_movement_speed: float
    observation_radius: float
    basic_interaction_radius: float
    ultimate_interaction_radius: float
    current_health: float
    maximum_health: float
    ultimate_cooldown_remaining: int
    status_remaining_duration_by_channel: tuple[int, ...]
    status_active_magnitude_by_channel: tuple[float | None, ...]
    steps_until_out_of_combat: int
    mage_aura_damage_multiplier: float
    warrior_aura_damage_multiplier: float
    basic_raw_damage: float
    basic_raw_healing: float
    ultimate_cooldown_steps: int
    status_capability_duration_by_channel: tuple[int, ...]
    status_capability_magnitude_by_channel: tuple[float | None, ...]
    mage_aura_radius: float
    mage_aura_per_emitter_multiplier: float
    warrior_aura_radius: float
    warrior_aura_per_emitter_multiplier: float
    ultimate_raw_healing: float
    ultimate_raw_damage: float
    out_of_combat_delay_steps: int
    out_of_combat_health_regeneration_fraction_per_step: float


def decode_agent_feature_row_v1(
    row: tuple[float, ...],
) -> DecodedAgentFeatureRowV1:
    """Decode one frozen V1 row without consulting simulator or Oracle state."""
    if type(row) is not tuple or len(row) != SELF_FEATURES_V1:
        raise ValueError(
            f"agent feature row must be an exact {SELF_FEATURES_V1}-value tuple."
        )
    for column, value in enumerate(row):
        _finite(value, name=f"agent feature column {column}")

    position = (
        row[AGENT_FEATURE_X_V1],
        row[AGENT_FEATURE_Y_V1],
    )
    radius = _finite(row[AGENT_FEATURE_RADIUS_V1], name="radius", minimum=0.0)
    maximum_health = _finite(
        row[AGENT_FEATURE_MAX_HEALTH_V1],
        name="maximum health",
        minimum=0.0,
    )
    current_health = _finite(
        row[AGENT_FEATURE_CURRENT_HEALTH_V1],
        name="current health",
        minimum=0.0,
    )
    if current_health > maximum_health:
        raise ValueError("current health must not exceed maximum health.")

    status_remaining = tuple(
        _wire_int(row[column], name=f"status channel {channel} remaining duration")
        for channel, column in enumerate(
            AGENT_STATUS_REMAINING_DURATION_COLUMN_BY_CHANNEL_V1
        )
    )
    status_active_magnitude = tuple(
        None
        if column is None
        else _finite(
            row[column],
            name=f"status channel {channel} active magnitude",
        )
        for channel, column in enumerate(
            AGENT_STATUS_ACTIVE_MAGNITUDE_COLUMN_BY_CHANNEL_V1
        )
    )
    status_capability_duration = tuple(
        _wire_int(row[column], name=f"status channel {channel} capability duration")
        for channel, column in enumerate(
            AGENT_STATUS_CAPABILITY_DURATION_COLUMN_BY_CHANNEL_V1
        )
    )
    status_capability_magnitude = tuple(
        None
        if column is None
        else _finite(
            row[column],
            name=f"status channel {channel} capability magnitude",
        )
        for channel, column in enumerate(
            AGENT_STATUS_CAPABILITY_MAGNITUDE_COLUMN_BY_CHANNEL_V1
        )
    )

    return DecodedAgentFeatureRowV1(
        position=position,
        radius=radius,
        team_id=_wire_int(row[AGENT_FEATURE_TEAM_ID_V1], name="team ID", minimum=0),
        configured_active=_wire_bool(
            row[AGENT_FEATURE_ACTIVE_V1],
            name="configured active",
        ),
        alive=_wire_bool(row[AGENT_FEATURE_ALIVE_V1], name="alive"),
        class_id=_wire_int(row[AGENT_FEATURE_CLASS_ID_V1], name="class ID"),
        base_movement_speed=_finite(
            row[AGENT_FEATURE_BASE_MOVEMENT_SPEED_V1],
            name="base movement speed",
            minimum=0.0,
        ),
        effective_movement_speed=_finite(
            row[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1],
            name="effective movement speed",
            minimum=0.0,
        ),
        observation_radius=_finite(
            row[AGENT_FEATURE_OBSERVATION_RADIUS_V1],
            name="observation radius",
            minimum=0.0,
        ),
        basic_interaction_radius=_finite(
            row[AGENT_FEATURE_BASIC_INTERACTION_RADIUS_V1],
            name="basic interaction radius",
            minimum=0.0,
        ),
        ultimate_interaction_radius=_finite(
            row[AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS_V1],
            name="ultimate interaction radius",
            minimum=0.0,
        ),
        current_health=current_health,
        maximum_health=maximum_health,
        ultimate_cooldown_remaining=_wire_int(
            row[AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1],
            name="ultimate cooldown remaining",
        ),
        status_remaining_duration_by_channel=status_remaining,
        status_active_magnitude_by_channel=status_active_magnitude,
        steps_until_out_of_combat=_wire_int(
            row[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT_V1],
            name="steps until out of combat",
        ),
        mage_aura_damage_multiplier=_finite(
            row[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1],
            name="Mage aura aggregate multiplier",
            minimum=0.0,
        ),
        warrior_aura_damage_multiplier=_finite(
            row[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1],
            name="Warrior aura aggregate multiplier",
            minimum=0.0,
        ),
        basic_raw_damage=_finite(
            row[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE_V1],
            name="basic raw damage capability",
            minimum=0.0,
        ),
        basic_raw_healing=_finite(
            row[AGENT_FEATURE_CAPABILITY_BASIC_HEALING_V1],
            name="basic raw healing capability",
            minimum=0.0,
        ),
        ultimate_cooldown_steps=_wire_int(
            row[AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION_V1],
            name="ultimate cooldown capability",
        ),
        status_capability_duration_by_channel=status_capability_duration,
        status_capability_magnitude_by_channel=status_capability_magnitude,
        mage_aura_radius=_finite(
            row[AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS_V1],
            name="Mage aura radius capability",
            minimum=0.0,
        ),
        mage_aura_per_emitter_multiplier=_finite(
            row[AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1],
            name="Mage aura multiplier capability",
            minimum=0.0,
        ),
        warrior_aura_radius=_finite(
            row[AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS_V1],
            name="Warrior aura radius capability",
            minimum=0.0,
        ),
        warrior_aura_per_emitter_multiplier=_finite(
            row[AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1],
            name="Warrior aura multiplier capability",
            minimum=0.0,
        ),
        ultimate_raw_healing=_finite(
            row[AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING_V1],
            name="ultimate raw healing capability",
            minimum=0.0,
        ),
        ultimate_raw_damage=_finite(
            row[AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE_V1],
            name="ultimate raw damage capability",
            minimum=0.0,
        ),
        out_of_combat_delay_steps=_wire_int(
            row[AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS_V1],
            name="out-of-combat delay capability",
        ),
        out_of_combat_health_regeneration_fraction_per_step=_finite(
            row[
                AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP_V1
            ],
            name="out-of-combat regeneration capability",
            minimum=0.0,
        ),
    )


__all__ = [
    "AGENT_FEATURE_ACTIVE_V1",
    "AGENT_FEATURE_ALIVE_V1",
    "AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION_V1",
    "AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1",
    "AGENT_FEATURE_BASE_MOVEMENT_SPEED_V1",
    "AGENT_FEATURE_BASIC_INTERACTION_RADIUS_V1",
    "AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE_V1",
    "AGENT_FEATURE_CAPABILITY_BASIC_HEALING_V1",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS_V1",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS_V1",
    "AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS_V1",
    "AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1",
    "AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION_V1",
    "AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE_V1",
    "AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING_V1",
    "AGENT_FEATURE_CLASS_ID_V1",
    "AGENT_FEATURE_CURRENT_HEALTH_V1",
    "AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1",
    "AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION_V1",
    "AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1",
    "AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1",
    "AGENT_FEATURE_MAX_HEALTH_V1",
    "AGENT_FEATURE_OBSERVATION_RADIUS_V1",
    "AGENT_FEATURE_RADIUS_V1",
    "AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION_V1",
    "AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION_V1",
    "AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION_V1",
    "AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER_V1",
    "AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION_V1",
    "AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER_V1",
    "AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION_V1",
    "AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER_V1",
    "AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT_V1",
    "AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION_V1",
    "AGENT_FEATURE_STUN_ROGUE_POISON_DURATION_V1",
    "AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION_V1",
    "AGENT_FEATURE_TEAM_ID_V1",
    "AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1",
    "AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS_V1",
    "AGENT_FEATURE_X_V1",
    "AGENT_FEATURE_Y_V1",
    "AGENT_STATUS_ACTIVE_MAGNITUDE_COLUMN_BY_CHANNEL_V1",
    "AGENT_STATUS_CAPABILITY_DURATION_COLUMN_BY_CHANNEL_V1",
    "AGENT_STATUS_CAPABILITY_MAGNITUDE_COLUMN_BY_CHANNEL_V1",
    "AGENT_STATUS_FEATURE_START_V1",
    "AGENT_STATUS_FEATURE_STOP_V1",
    "AGENT_STATUS_REMAINING_DURATION_COLUMN_BY_CHANNEL_V1",
    "CONTEXT_FEATURE_MAP_HEIGHT_V1",
    "CONTEXT_FEATURE_MAP_WIDTH_V1",
    "OBSTACLE_FEATURE_ACTIVE_V1",
    "OBSTACLE_FEATURE_HEIGHT_V1",
    "OBSTACLE_FEATURE_RADIUS_V1",
    "OBSTACLE_FEATURE_THETA_V1",
    "OBSTACLE_FEATURE_TYPE_V1",
    "OBSTACLE_FEATURE_WIDTH_V1",
    "OBSTACLE_FEATURE_X_V1",
    "OBSTACLE_FEATURE_Y_V1",
    "DecodedAgentFeatureRowV1",
    "decode_agent_feature_row_v1",
]
