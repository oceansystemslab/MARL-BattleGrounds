"""Pure combat catalog support for the JAX-native simulator."""

import jax.numpy as jnp
from jax import Array

# Class catalog row order: neutral, Mage, Warrior, Hunter, Rogue, Priest.

MAX_HEALTH_BY_CLASS = jnp.asarray(
    [
        0.0,  # neutral
        80.0,  # mage
        200.0,  # warrior
        100.0,  # hunter
        100.0,  # rogue
        100.0,  # priest
    ],
    dtype=jnp.float32,
)


MOVEMENT_SPEED_BY_CLASS = jnp.asarray(
    [
        0.0,  # neutral
        1.0,  # mage
        1.0,  # warrior
        1.0,  # hunter
        1.25,  # rogue
        1.0,  # priest
    ],
    dtype=jnp.float32,
)


BODY_RADIUS_BY_CLASS = jnp.asarray(
    [
        0.0,  # neutral
        0.5,  # mage
        0.5,  # warrior
        0.5,  # hunter
        0.5,  # rogue
        0.5,  # priest
    ],
    dtype=jnp.float32,
)


BASIC_INTERACTION_RADIUS_BY_CLASS = jnp.asarray(
    [
        0.0,  # neutral
        5.0,  # mage
        0.5,  # warrior
        5.5,  # hunter
        0.5,  # rogue
        5.0,  # priest
    ],
    dtype=jnp.float32,
)


BASIC_DAMAGE_BY_CLASS = jnp.asarray(
    [
        0.0,  # neutral
        12.0,  # mage
        8.0,  # warrior
        8.0,  # hunter
        12.0,  # rogue
        0.0,  # priest
    ],
    dtype=jnp.float32,
)


BASIC_HEALING_BY_CLASS = jnp.asarray(
    [
        0.0,  # neutral
        0.0,  # mage
        0.0,  # warrior
        0.0,  # hunter
        0.0,  # rogue
        8.0,  # priest
    ],
    dtype=jnp.float32,
)


# Mage Burst is a no-target self-buff, so its interaction radius is zero.
ULTIMATE_INTERACTION_RADIUS_BY_CLASS = jnp.asarray(
    [
        0.0,  # neutral
        0.0,  # mage
        6.0,  # warrior
        4.0,  # hunter
        1.0,  # rogue
        6.0,  # priest
    ],
    dtype=jnp.float32,
)


ULTIMATE_COOLDOWN_BY_CLASS = jnp.asarray(
    [
        0,  # neutral
        30,  # mage
        30,  # warrior
        30,  # hunter
        30,  # rogue
        30,  # priest
    ],
    dtype=jnp.int32,
)

OBSERVATION_RADIUS_BY_CLASS = jnp.asarray(
    [
        0,  # neutral
        6,  # mage
        6,  # warrior
        6,  # hunter
        6,  # rogue
        6,  # priest
    ],
    dtype=jnp.float32,
)

# Global status rules.

# The floor keeps stacked slows from becoming a hard stun.
GLOBAL_SLOW_FLOOR = 0.20


# Hunter mechanics.

HUNTER_BASIC_SLOW_MULTIPLIER = 0.85
HUNTER_BASIC_SLOW_DURATION_TICKS = 1

# Hunter trap has a longer stun duration because the trap can be broken.
HUNTER_TRAP_STUN_DURATION_TICKS = 4


# Warrior mechanics.

WARRIOR_CHARGE_SLOW_MULTIPLIER = 0.50
WARRIOR_CHARGE_SLOW_DURATION_TICKS = 5
WARRIOR_CHARGE_STUN_DURATION_TICKS = 1

# Warrior mitigation is the defensive counterpart to Mage amplification.
WARRIOR_MITIGATION_AURA_RADIUS = 2.0
WARRIOR_MITIGATION_AURA_MULTIPLIER = 0.85


# Rogue mechanics.

ROGUE_POISON_SLOW_MULTIPLIER = 0.50
ROGUE_POISON_SLOW_DURATION_TICKS = 5

ROGUE_POISON_STUN_DURATION_TICKS = 1

ROGUE_POISON_ANTI_HEAL_MULTIPLIER = 0.50
ROGUE_POISON_ANTI_HEAL_DURATION_TICKS = 4


# Mage mechanics.

MAGE_DAMAGE_AURA_RADIUS = 2.0
MAGE_DAMAGE_AURA_MULTIPLIER = 1.15

# Mage Burst duration keeps the buff interruptible by hard control.
MAGE_ULT_DAMAGE_DURATION_TICKS = 5
MAGE_ULT_DAMAGE_MULTIPLIER = 1.50


# Priest mechanics.

PRIEST_ULT_HEAL_AMOUNT = 75.0

# Priest healing grants temporary slow protection.
# While active, slows cannot reduce the target below this fraction of base speed.
PRIEST_HEAL_SPEED_FLOOR = 0.85
PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS = 1

# Catalog access helpers.


def get_max_health_by_class_ids(class_ids: int | Array) -> Array:
    """Return max health values for one or more class IDs."""
    return MAX_HEALTH_BY_CLASS[class_ids]


def get_movement_speed_by_class_ids(class_ids: int | Array) -> Array:
    """Return movement speed values for one or more class IDs."""
    return MOVEMENT_SPEED_BY_CLASS[class_ids]


def get_body_radius_by_class_ids(class_ids: int | Array) -> Array:
    """Return body radius values for one or more class IDs."""
    return BODY_RADIUS_BY_CLASS[class_ids]


def get_basic_interaction_radius_by_class_ids(class_ids: int | Array) -> Array:
    """Return basic interaction radii for one or more class IDs."""
    return BASIC_INTERACTION_RADIUS_BY_CLASS[class_ids]


def get_basic_damage_by_class_ids(class_ids: int | Array) -> Array:
    """Return basic damage values for one or more class IDs."""
    return BASIC_DAMAGE_BY_CLASS[class_ids]


def get_basic_healing_by_class_ids(class_ids: int | Array) -> Array:
    """Return basic healing values for one or more class IDs."""
    return BASIC_HEALING_BY_CLASS[class_ids]


def get_ultimate_interaction_radius_by_class_ids(class_ids: int | Array) -> Array:
    """Return ultimate interaction radii for one or more class IDs."""
    return ULTIMATE_INTERACTION_RADIUS_BY_CLASS[class_ids]


def get_ultimate_cooldown_by_class_ids(class_ids: int | Array) -> Array:
    """Return ultimate cooldown values for one or more class IDs."""
    return ULTIMATE_COOLDOWN_BY_CLASS[class_ids]


def get_observation_radius_by_class_ids(class_ids: int | Array) -> Array:
    """Return observation radii for one or more class IDs."""
    return OBSERVATION_RADIUS_BY_CLASS[class_ids]


def _build_slow_multipliers(slow_durations: Array) -> Array:
    # if a duration is > 0, then the multiplier is active.
    class_slow_multipliers = jnp.asarray(
        [
            WARRIOR_CHARGE_SLOW_MULTIPLIER,
            HUNTER_BASIC_SLOW_MULTIPLIER,
            ROGUE_POISON_SLOW_MULTIPLIER,
        ]
    )[None, :]

    slow_multipliers = jnp.where(slow_durations > 0, class_slow_multipliers, 1.0)

    return slow_multipliers


def _build_priest_blessing_of_freedom_slow_floor_fractions(
    priest_blessing_of_freedom_slow_floor_durations: Array,
) -> Array:
    return jnp.where(
        priest_blessing_of_freedom_slow_floor_durations > 0,
        PRIEST_HEAL_SPEED_FLOOR,
        0.0,
    ).astype(jnp.float32)


def derive_status_magnitudes(
    slow_durations: Array,
    rogue_poison_anti_heal_durations: Array,
    priest_blessing_of_freedom_slow_floor_durations: Array,
) -> tuple[Array, Array, Array]:
    """Derive the fixed-strength status payloads consumed by current mechanics.

    Multiplicative effects use ``1.0`` while inactive. The Priest movement-floor
    fraction uses ``0.0`` while inactive because absence is not a multiplier.
    The return order is slow multipliers, Rogue anti-heal multipliers, then
    Priest slow-floor fractions. Inputs and outputs retain the simulator's fixed
    slot and source-channel shapes so this helper remains safe under eager, JIT,
    and scanned execution. Add other derived payloads only when a production
    mechanic consumes them.
    """

    # if a duration is > 0, then the multiplier is active.
    slow_multipliers = _build_slow_multipliers(slow_durations)

    # Rogue anti-heal
    rogue_poison_anti_heal_multipliers = jnp.where(
        rogue_poison_anti_heal_durations > 0, ROGUE_POISON_ANTI_HEAL_MULTIPLIER, 1.0
    ).astype(jnp.float32)

    priest_blessing_of_freedom_slow_floor_fractions = (
        _build_priest_blessing_of_freedom_slow_floor_fractions(
            priest_blessing_of_freedom_slow_floor_durations
        )
    )

    return (
        slow_multipliers,
        rogue_poison_anti_heal_multipliers,
        priest_blessing_of_freedom_slow_floor_fractions,
    )


def derive_effective_movement_speeds_from_durations(
    base_movement_speeds: Array,
    slow_durations: Array,
    priest_blessing_of_freedom_slow_floor_durations: Array,
) -> Array:

    effective_movement_multipliers = jnp.maximum(
        jnp.prod(_build_slow_multipliers(slow_durations), axis=-1),
        _build_priest_blessing_of_freedom_slow_floor_fractions(
            priest_blessing_of_freedom_slow_floor_durations
        ),
    )

    return base_movement_speeds * jnp.maximum(
        effective_movement_multipliers, GLOBAL_SLOW_FLOOR
    )
