"""Pure combat catalog support for the JAX-native simulator."""

import jax.numpy as jnp
from jax import Array

# Class catalog row order: neutral, Mage, Warrior, Hunter, Rogue, Priest.

# Ultimate target-relation modes. Zero remains the inert neutral catalog value;
# no-target is distinct because Mage Burst is a real self-buff action.
NO_ULTIMATE_MODE = 0
ONLY_NONE_TARGET_ULTIMATE_MODE = 1
ONLY_ALLY_TARGET_ULTIMATE_MODE = 2
ONLY_ENEMY_TARGET_ULTIMATE_MODE = 3

# Global status rules.

# The floor keeps stacked slows from becoming a hard stun.
GLOBAL_SLOW_FLOOR = 0.20

# Hunter mechanics.

HUNTER_BASIC_SLOW_MULTIPLIER = 0.85
HUNTER_BASIC_SLOW_DURATION_TICKS = 1

# Hunter trap has a longer stun duration because the trap can be broken.
HUNTER_TRAP_STUN_DURATION_TICKS = 4


# Warrior mechanics.

WARRIOR_CHARGE_SLOW_DURATION_TICKS = 5
WARRIOR_CHARGE_STUN_DURATION_TICKS = 1

# Warrior mitigation is the defensive counterpart to Mage amplification.
WARRIOR_DAMAGE_MITIGATION_AURA_RADIUS = 2.0
WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER = 0.85
# Duplicate emitters multiply before the effective aura value reaches this floor.
WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER_FLOOR = (
    WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER
    * WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER
)
WARRIOR_CHARGE_SLOW_MULTIPLIER = 0.50


# Rogue mechanics.

ROGUE_POISON_SLOW_MULTIPLIER = 0.50
ROGUE_POISON_SLOW_DURATION_TICKS = 5

ROGUE_POISON_STUN_DURATION_TICKS = 1

ROGUE_POISON_ANTI_HEAL_MULTIPLIER = 0.50
ROGUE_POISON_ANTI_HEAL_DURATION_TICKS = 4


# Mage mechanics.

MAGE_DAMAGE_AURA_RADIUS = 2.0
MAGE_DAMAGE_AURA_MULTIPLIER = 1.15
# Duplicate emitters multiply before the effective aura value reaches this ceiling.
MAGE_DAMAGE_AURA_MULTIPLIER_CEILING = (
    MAGE_DAMAGE_AURA_MULTIPLIER * MAGE_DAMAGE_AURA_MULTIPLIER
)

# Mage Burst duration keeps the buff interruptible by hard control.
MAGE_BURST_DAMAGE_DURATION_TICKS = 5
MAGE_BURST_DAMAGE_MULTIPLIER = 1.50


# Priest mechanics.

# Priest healing grants temporary slow protection.
# While active, slows cannot reduce the target below this fraction of base speed.
PRIEST_HEAL_SPEED_FLOOR = 0.85
PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS = 1


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


BASE_MOVEMENT_SPEED_BY_CLASS = jnp.asarray(
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

ULTIMATE_DAMAGE_BY_CLASS = jnp.asarray(
    [
        0.0,  # neutral
        0.0,  # mage
        16.0,  # warrior
        0.0,  # hunter
        0.0,  # rogue
        0.0,  # priest
    ],
    dtype=jnp.float32,
)


ULTIMATE_HEALING_BY_CLASS = jnp.asarray(
    [
        0.0,  # neutral
        0.0,  # mage
        0.0,  # warrior
        0.0,  # hunter
        0.0,  # rogue
        75.0,  # priest
    ],
    dtype=jnp.float32,
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

ULTIMATE_TARGET_MODE_BY_CLASS = jnp.asarray(
    [
        NO_ULTIMATE_MODE,  # neutral
        ONLY_NONE_TARGET_ULTIMATE_MODE,  # mage
        ONLY_ENEMY_TARGET_ULTIMATE_MODE,  # warrior
        ONLY_ENEMY_TARGET_ULTIMATE_MODE,  # hunter
        ONLY_ENEMY_TARGET_ULTIMATE_MODE,  # rogue
        ONLY_ALLY_TARGET_ULTIMATE_MODE,  # priest
    ],
    dtype=jnp.int32,
)


# Catalog access helpers.


def get_max_health_by_class_ids(class_ids: int | Array) -> Array:
    """Return max health values for one or more class IDs."""
    return MAX_HEALTH_BY_CLASS[class_ids]


def get_base_movement_speed_by_class_ids(class_ids: int | Array) -> Array:
    """Return movement speed values for one or more class IDs."""
    return BASE_MOVEMENT_SPEED_BY_CLASS[class_ids]


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


def get_ultimate_target_mode_by_class_ids(class_ids: int | Array) -> Array:
    """Return ultimate target-relation modes for one or more class IDs."""
    return ULTIMATE_TARGET_MODE_BY_CLASS[class_ids]


def get_ultimate_damage_by_class_ids(class_ids: int | Array) -> Array:
    """Return ultimate damage values for one or more class IDs."""
    return ULTIMATE_DAMAGE_BY_CLASS[class_ids]


def get_ultimate_healing_by_class_ids(class_ids: int | Array) -> Array:
    """Return ultimate healing values for one or more class IDs."""
    return ULTIMATE_HEALING_BY_CLASS[class_ids]


def _build_slow_multipliers(slow_durations: Array) -> Array:
    """Return per-source slow multipliers selected by active durations."""
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
    """Return the active Priest movement-speed floor for each agent slot."""
    return jnp.where(
        priest_blessing_of_freedom_slow_floor_durations > 0,
        PRIEST_HEAL_SPEED_FLOOR,
        0.0,
    ).astype(jnp.float32)


def build_rogue_poison_anti_heal_multipliers(
    rogue_poison_anti_heal_durations: Array,
) -> Array:
    """Return per-slot Rogue Poison healing multipliers from durations."""
    return jnp.where(
        rogue_poison_anti_heal_durations > 0, ROGUE_POISON_ANTI_HEAL_MULTIPLIER, 1.0
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
    slow_multipliers = _build_slow_multipliers(slow_durations)

    rogue_poison_anti_heal_multipliers = build_rogue_poison_anti_heal_multipliers(
        rogue_poison_anti_heal_durations
    )

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


def derive_effective_movement_speeds(
    slow_durations: Array,
    priest_freedom_slow_floor_durations: Array,
    stun_durations: Array,
    base_movement_speeds: Array,
    active_and_alive_mask: Array,
    ordinary_movement_distance_scale: float,
) -> Array:
    """Derive the currently actuated voluntary speed for every fixed slot.

    Slow sources compose multiplicatively before the global and Blessing of
    Freedom floors apply. The episode's ordinary-movement distance scale then
    converts catalog speed into per-decision voluntary displacement. Inactive,
    dead, or currently stunned actors expose exactly zero effective speed. The
    returned ``float32`` vector has shape ``(MAX_AGENT_SLOTS,)`` and is shared
    by movement actuation and the policy-facing observation contract.
    """
    effective_movement_multipliers = jnp.maximum(
        jnp.prod(_build_slow_multipliers(slow_durations), axis=-1),
        _build_priest_blessing_of_freedom_slow_floor_fractions(
            priest_freedom_slow_floor_durations
        ),
    )

    active_alive_not_stunned_mask = jnp.logical_and(
        active_and_alive_mask,
        jnp.all(stun_durations == 0, axis=-1),
    )

    adjusted_movement_speeds = (
        base_movement_speeds
        * jnp.maximum(effective_movement_multipliers, GLOBAL_SLOW_FLOOR)
        * ordinary_movement_distance_scale
    )

    return jnp.where(
        active_alive_not_stunned_mask,
        adjusted_movement_speeds,
        jnp.zeros_like(adjusted_movement_speeds),
    ).astype(jnp.float32)
