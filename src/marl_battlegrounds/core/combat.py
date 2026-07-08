"""Pure combat catalog support for the JAX-native simulator."""

import jax.numpy as jnp

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


BASIC_RANGE_BY_CLASS = jnp.asarray(
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


# Mage Burst is a no-target self-buff, so its range is zero.
ULTIMATE_RANGE_BY_CLASS = jnp.asarray(
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
