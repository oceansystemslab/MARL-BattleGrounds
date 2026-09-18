"""Own class catalogs and derive current status strengths for Core.

Catalog rows are neutral, Mage, Warrior, Hunter, Rogue and Priest. Lookup
helpers return JAX arrays without validating IDs. Configuration builders
resolve and validate those values before reset. Status helpers read stored
durations and derive strengths; they do not apply damage or advance time.

``derive_effective_movement_speeds`` is shared by physical movement and
observations. Keeping it here gives both paths the same current speed.
Catalog arrays use float32 for magnitudes and int32 for counts and modes."""

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

MAGE_DAMAGE_AMPLIFICATION_AURA_RADIUS = 2.0
MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER = 1.15
# Duplicate emitters multiply before the effective aura value reaches this ceiling.
MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER_CEILING = (
    MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER
    * MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER
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
        1.3,  # rogue
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
        3.0,  # mage
        1.5,  # warrior
        3.5,  # hunter
        1.5,  # rogue
        3.0,  # priest
    ],
    dtype=jnp.float32,
)


BASIC_DAMAGE_BY_CLASS = jnp.asarray(
    [
        0.0,  # neutral
        13.0,  # mage
        8.0,  # warrior
        6.0,  # hunter
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
        5.75,  # warrior
        3.0,  # hunter
        1.5,  # rogue
        5.75,  # priest
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
        20.0,  # warrior
        10.0,  # hunter
        36.0,  # rogue
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
        jnp.max(MAX_HEALTH_BY_CLASS),  # priest
    ],
    dtype=jnp.float32,
)

OBSERVATION_RADIUS_BY_CLASS = jnp.asarray(
    [
        0,  # neutral
        6,  # mage
        6,  # warrior
        6.5,  # hunter
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

OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS = jnp.asarray(
    [
        0,  # neutral
        5,  # mage
        5,  # warrior
        5,  # hunter
        3,  # rogue
        5,  # priest
    ],
    dtype=jnp.int32,
)

# These remain independent tuning surfaces even though every active class
# currently shares the same regeneration rate.
OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS = jnp.asarray(
    [
        0.0,  # neutral
        0.04,  # mage
        0.04,  # warrior
        0.04,  # hunter
        0.04,  # rogue
        0.04,  # priest
    ],
    dtype=jnp.float32,
)


# Catalog access helpers.


def get_max_health_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog maximum health for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Float32 values with the input shape, or scalar shape () for one ID.
        Values use health units. Neutral entries are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return MAX_HEALTH_BY_CLASS[class_ids]


def get_base_movement_speed_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog base movement speed for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Float32 values with the input shape, or scalar shape () for one ID.
        Values use world units per step before the episode movement scale. Neutral
        entries are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return BASE_MOVEMENT_SPEED_BY_CLASS[class_ids]


def get_body_radius_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog body radius for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Float32 values with the input shape, or scalar shape () for one ID.
        Values use world units. Neutral entries are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return BODY_RADIUS_BY_CLASS[class_ids]


def get_basic_interaction_radius_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog Basic interaction radius for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Float32 values with the input shape, or scalar shape () for one ID.
        Values use world units. Neutral entries are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return BASIC_INTERACTION_RADIUS_BY_CLASS[class_ids]


def get_basic_damage_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog Basic damage for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Float32 values with the input shape, or scalar shape () for one ID.
        Values use health units before status and aura modifiers. Neutral entries
        are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return BASIC_DAMAGE_BY_CLASS[class_ids]


def get_basic_healing_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog Basic healing for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Float32 values with the input shape, or scalar shape () for one ID.
        Values use health units before recipient modifiers and health clipping.
        Neutral entries are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return BASIC_HEALING_BY_CLASS[class_ids]


def get_ultimate_interaction_radius_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog Ultimate interaction radius for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Float32 values with the input shape, or scalar shape () for one ID.
        Values use world units; Mage's no-target Ultimate has radius zero. Neutral
        entries are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return ULTIMATE_INTERACTION_RADIUS_BY_CLASS[class_ids]


def get_ultimate_cooldown_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog Ultimate cooldown for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Int32 values with the input shape, or scalar shape () for one ID.
        Values use steps. Neutral entries are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return ULTIMATE_COOLDOWN_BY_CLASS[class_ids]


def get_observation_radius_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog observation radius for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Float32 values with the input shape, or scalar shape () for one ID.
        Values use world units. Neutral entries are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return OBSERVATION_RADIUS_BY_CLASS[class_ids]


def get_ultimate_target_mode_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog Ultimate target mode for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Int32 values with the input shape, or scalar shape () for one ID.
        Values use mode IDs: 0 inert, 1 no target, 2 ally, 3 enemy. Neutral entries
        are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return ULTIMATE_TARGET_MODE_BY_CLASS[class_ids]


def get_ultimate_damage_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog Ultimate damage for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Float32 values with the input shape, or scalar shape () for one ID.
        Values use health units before status and aura modifiers. Neutral entries
        are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return ULTIMATE_DAMAGE_BY_CLASS[class_ids]


def get_ultimate_healing_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog Ultimate healing for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Float32 values with the input shape, or scalar shape () for one ID.
        Values use health units before recipient modifiers and health clipping.
        Neutral entries are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return ULTIMATE_HEALING_BY_CLASS[class_ids]


def get_ooc_delay_steps_by_class_ids(class_ids: int | Array) -> Array:
    """Return catalog out-of-combat recovery delay for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Int32 values with the input shape, or scalar shape () for one ID.
        Values use steps used to reset the recovery countdown. Neutral entries are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS[class_ids]


def get_ooc_health_regen_fraction_per_step_by_class_ids(
    class_ids: int | Array,
) -> Array:
    """Return catalog out-of-combat recovery fraction for the supplied class IDs.

    Parameters
    ----------
    class_ids : int or jax.Array
        Class IDs in 0..5. Zero is neutral; 1..5 are Mage, Warrior, Hunter,
        Rogue and Priest. Array inputs use integer dtype and may have any shape.

    Returns
    -------
    jax.Array
        Float32 values with the input shape, or scalar shape () for one ID.
        Values use fraction of maximum health recovered per eligible step. Neutral
        entries are zero.

    Notes
    -----
    This is a pure JAX table lookup. Callers validate IDs before execution;
    out-of-range indexing is not a supported validation mechanism.
    """
    return OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS[class_ids]


def _build_slow_multipliers(slow_durations: Array) -> Array:
    """Read active slow strengths in Warrior, Hunter and Rogue channel order.

    Input durations are int32 (10, 3). The matching multiplier is selected
    when a duration is positive; an inactive channel contributes 1.0.
    The fixed channel table broadcasts over slots and is not stored in state.
    """
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
    """Read the Priest speed floor for each current duration.

    Int32 durations (10,) produce float32 fractions (10,): the catalog
    floor while positive, otherwise 0.0. Zero means no extra floor.
    """
    return jnp.where(
        priest_blessing_of_freedom_slow_floor_durations > 0,
        PRIEST_HEAL_SPEED_FLOOR,
        0.0,
    ).astype(jnp.float32)


def build_rogue_poison_anti_heal_multipliers(
    rogue_poison_anti_heal_durations: Array,
) -> Array:
    """Read the healing multiplier for each current Rogue Poison duration.

    Parameters
    ----------
    rogue_poison_anti_heal_durations : jax.Array
        Nonnegative int32 remaining durations (10,) in steps.

    Returns
    -------
    jax.Array
        Float32 (10,): the catalog anti-heal multiplier where duration is
        positive, otherwise 1.0 so unmodified healing is preserved.

    Notes
    -----
    This pure JAX helper reads durations without decrementing them. Map it
    across games with vmap when working with a native environment batch.
    """
    return jnp.where(
        rogue_poison_anti_heal_durations > 0, ROGUE_POISON_ANTI_HEAL_MULTIPLIER, 1.0
    ).astype(jnp.float32)


def derive_status_magnitudes(
    slow_durations: Array,
    rogue_poison_anti_heal_durations: Array,
    priest_blessing_of_freedom_slow_floor_durations: Array,
) -> tuple[Array, Array, Array]:
    """Read current status strengths from the stored remaining durations.

    Parameters
    ----------
    slow_durations : jax.Array
        Int32 (10, 3) nonnegative durations in Warrior, Hunter, Rogue order.
    rogue_poison_anti_heal_durations : jax.Array
        Int32 (10,) nonnegative anti-heal durations in steps.
    priest_blessing_of_freedom_slow_floor_durations : jax.Array
        Int32 (10,) nonnegative Priest movement-floor durations in steps.

    Returns
    -------
    tuple of jax.Array
        Slow multipliers (10, 3), Rogue healing multipliers (10,), then
        Priest speed-floor fractions (10,). In the supported float32 Core
        path all outputs are float32. Inactive multipliers are 1.0;
        an absent extra movement floor is 0.0.

    Notes
    -----
    This reads the current decision's strengths without changing durations.
    State owns durations; this catalog owns fixed strengths. The helper is
    pure JAX and supports jit and vmap. Add a derived payload only when a
    production mechanic consumes it.
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
    spawn_shield_durations: Array,
    base_movement_speeds: Array,
    spawn_shield_movement_speed: Array | float,
    active_and_alive_mask: Array,
    ordinary_movement_distance_scale: float,
) -> Array:
    """Return the voluntary distance each fixed slot can move this step.

    Slow channels multiply before the global and Priest minimum-speed
    fractions apply. The ordinary movement scale then converts catalog
    speed to distance per step. A positive spawn shield instead selects
    the absolute shield speed and bypasses those adjustments.

    Parameters
    ----------
    slow_durations : jax.Array
        Nonnegative int32 (10, 3) source-specific remaining slow steps.
    priest_freedom_slow_floor_durations : jax.Array
        Nonnegative int32 (10,) remaining Priest speed-floor steps.
    stun_durations : jax.Array
        Nonnegative int32 (10, 3) source-specific remaining stun steps.
    spawn_shield_durations : jax.Array
        Nonnegative int32 (10,) remaining protected movement steps.
    base_movement_speeds : jax.Array
        Float32 (10,) class-catalog movement speeds.
    spawn_shield_movement_speed : float or jax.Array
        Positive finite scalar shielded movement distance per step.
    active_and_alive_mask : jax.Array
        Bool (10,) indicating actors currently able to participate.
    ordinary_movement_distance_scale : float
        Episode scale in (0.0, 1.0], supplied as a dynamic scalar under JAX.

    Returns
    -------
    jax.Array
        Float32 (10,) movement distances in world units. Dead and unused
        slots are zero. Unshielded stunned actors are also zero.

    Notes
    -----
    Movement and observations share this helper so the visible speed agrees
    with actuation. Inputs describe the same current decision. Durations are
    not advanced here. This is pure JAX; use vmap for separate games.
    """
    effective_movement_multipliers = jnp.maximum(
        jnp.prod(_build_slow_multipliers(slow_durations), axis=-1),
        _build_priest_blessing_of_freedom_slow_floor_fractions(
            priest_freedom_slow_floor_durations
        ),
    )

    adjusted_movement_speeds = (
        base_movement_speeds
        * jnp.maximum(effective_movement_multipliers, GLOBAL_SLOW_FLOOR)
        * ordinary_movement_distance_scale
    )

    stun_adjusted_movement_speeds = jnp.where(
        jnp.all(stun_durations == 0, axis=-1),
        adjusted_movement_speeds,
        jnp.zeros_like(adjusted_movement_speeds),
    ).astype(jnp.float32)

    spawn_shield_adjusted_movement_speeds = jnp.where(
        spawn_shield_durations > 0,
        spawn_shield_movement_speed,
        stun_adjusted_movement_speeds,
    ).astype(jnp.float32)

    # Derive from the sole stored counter so movement and visible effective
    # speed cannot disagree about whether the shield override is active.
    return jnp.where(
        active_and_alive_mask,
        spawn_shield_adjusted_movement_speeds,
        jnp.zeros_like(spawn_shield_adjusted_movement_speeds),
    )
