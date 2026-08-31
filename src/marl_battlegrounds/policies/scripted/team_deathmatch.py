"""Score one complete TDM action from authorized same-epoch facts.

Exact masks own legality and invalid facts neutralize only dependent values.
No transition or new observation occurs between combat and movement selection;
both use the same facts. The regime-independent core remains task-owned until a
second task proves an identical ``common.py`` extraction contract.
"""

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.axis_mappings import (
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY,
)
from marl_battlegrounds.core.geometry import (
    disc_overlaps_obstacle,
    has_clear_line_of_sight,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION,
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER,
    AGENT_FEATURE_BASE_MOVEMENT_SPEED,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE,
    AGENT_FEATURE_CAPABILITY_BASIC_HEALING,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS,
    AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION,
    AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION,
    AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION,
    AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION,
    AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION,
    AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION,
    AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE,
    AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    AGENT_FEATURE_MAX_HEALTH,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION,
    AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER,
    AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION,
    AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER,
    AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER,
    AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT,
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION,
    AGENT_FEATURE_STUN_ROGUE_POISON_DURATION,
    AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING,
    AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MOVE_STAY,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    ActionMask,
)
from marl_battlegrounds.policies.actor import ActorAction

POLICY_ID = "scripted/team_deathmatch"
POLICY_SEMANTIC_VERSION = 1
TASK_HEAD_VERSION = 1
SEMANTIC_PROFILE_ID = "scripted-common-v1"
NUMERIC_PROFILE_ID = "scripted-common-f32-v1"
TRACE_ONTOLOGY_VERSION = 1

FOCAL_SHIELD_UNKNOWN = -1
FOCAL_SHIELD_KNOWN_FALSE = 0
FOCAL_SHIELD_KNOWN_TRUE = 1

# Reason IDs are stable within TRACE_ONTOLOGY_VERSION and describe the
# authority that produced the returned support, basis, or movement choice.
DIRECT_SCORE = 0
EFFECT_INERT_NOOP = 1
MAGE_BURST_TRIGGER = 2
WARRIOR_CHARGE_TRIGGER = 3
HUNTER_TRAP_EMERGENCY = 4
HUNTER_TRAP_PRIEST_CROWD = 5
HUNTER_TRAP_NO_PRIEST_CROWD = 6
ROGUE_POISON_SUBSTITUTION = 7
PRIEST_HOLY_WORD_TRIGGER = 8

MOVE_DIRECT_SCORE = 0
CHARGE_TO_STAY = 1
STAY_DEADBAND = 2
MIN_RISK_FALLBACK = 3


class TeamDeathmatchProfile(NamedTuple):
    """Immutable host profile with explicit JAX dtypes at every use site."""

    combat_weights: tuple[tuple[float, ...], ...]
    movement_weights: tuple[tuple[float, ...], ...]
    basic_range_bands: tuple[tuple[float, float], ...]
    formation_band: tuple[float, float]
    quality_falloff: float
    damage_potency_fraction: float
    impact_mix: tuple[float, float]
    urgency_mix: tuple[float, float]
    excess_weight: float
    duration_base: float
    duration_scale: float
    duration_horizon: float
    slow_value_scale: float
    offensive_counter_value: float
    countered_by_multiplier: float
    role_class_values: tuple[float, ...]
    counters: tuple[tuple[int, ...], ...]
    countered_by: tuple[tuple[int, ...], ...]
    recovery_at_one: float
    recovery_after_one: float
    crowding_distance_factor: float
    obstruction_weights: tuple[float, float, float]
    aura_weight: float
    residual_risk_weight: float
    crowding_weight: float
    obstruction_weight: float
    alive_context_weight: float
    risk_ceiling: float
    stay_deadband: float
    aura_cover_cap: float
    mage_burst_crowd: int
    warrior_charge_health: float
    warrior_charge_trap_max: float
    warrior_mage_burst_min: float
    hunter_emergency_health: float
    hunter_crowd: int
    priest_holy_word_health: float
    trap_preserve_duration: float
    tdm_task_value: float


TEAM_DEATHMATCH_PROFILE = TeamDeathmatchProfile(
    combat_weights=(
        (0.30, 0.30, 0.15, 0.15, 0.10),
        (0.15, 0.20, 0.35, 0.20, 0.10),
        (0.20, 0.20, 0.20, 0.30, 0.10),
        (0.25, 0.35, 0.10, 0.10, 0.20),
        (0.30, 0.30, 0.25, 0.10, 0.05),
    ),
    movement_weights=(
        (0.40, 0.20, 0.15),
        (0.20, 0.40, 0.10),
        (0.45, 0.15, 0.15),
        (0.45, 0.05, 0.20),
        (0.20, 0.40, 0.15),
    ),
    basic_range_bands=(
        (0.75, 0.95),
        (0.35, 0.80),
        (0.80, 0.95),
        (0.25, 0.75),
        (0.60, 0.90),
    ),
    formation_band=(0.50, 1.00),
    quality_falloff=0.25,
    damage_potency_fraction=0.20,
    impact_mix=(0.50, 0.50),
    urgency_mix=(0.50, 0.50),
    excess_weight=0.15,
    duration_base=0.50,
    duration_scale=0.50,
    duration_horizon=5.0,
    slow_value_scale=0.50,
    offensive_counter_value=0.12,
    countered_by_multiplier=1.25,
    role_class_values=(0.06, 0.00, 0.02, 0.04, 0.08),
    counters=(
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 1, 0, 0, 0),
        (1, 0, 0, 0, 1),
        (0, 0, 1, 0, 0),
    ),
    countered_by=(
        (0, 0, 0, 1, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 0, 1),
        (0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0),
    ),
    recovery_at_one=0.50,
    recovery_after_one=0.25,
    crowding_distance_factor=2.0,
    obstruction_weights=(1.00, 0.75, 0.50),
    aura_weight=0.05,
    residual_risk_weight=0.30,
    crowding_weight=0.05,
    obstruction_weight=0.10,
    alive_context_weight=0.10,
    risk_ceiling=0.70,
    stay_deadband=0.03,
    aura_cover_cap=2.0,
    mage_burst_crowd=2,
    warrior_charge_health=80.0,
    warrior_charge_trap_max=1.0,
    warrior_mage_burst_min=3.0,
    hunter_emergency_health=20.0,
    hunter_crowd=2,
    priest_holy_word_health=30.0,
    trap_preserve_duration=1.0,
    tdm_task_value=0.0,
)


class PolicyFacts(NamedTuple):
    """Adapter-authorized same-epoch scalar actor facts."""

    focal_features: Array  # (58,) float32
    ally_features: Array  # (5, 58) float32
    enemy_features: Array  # (5, 58) float32
    obstacles: Array  # (MAX_OBSTACLE_SLOTS, 8) float32
    ally_visible: Array  # (5,) bool
    enemy_visible: Array  # (5,) bool
    own_active: Array  # (5,) bool
    enemy_active: Array  # (5,) bool
    own_alive: Array  # (5,) bool
    enemy_alive: Array  # (5,) bool
    own_spawn_shields: Array  # (5,)
    enemy_spawn_shields: Array  # (5,)
    own_class_ids: Array  # (5,) int32
    enemy_class_ids: Array  # (5,) int32
    own_configured_count: Array  # scalar int32
    enemy_configured_count: Array  # scalar int32
    map_width: Array  # scalar float32
    map_height: Array  # scalar float32
    focal_shield_state: Array  # scalar int32


class ScriptedTrace(NamedTuple):
    """Selected-path evidence whose reason IDs define basis/component meaning."""

    combat_target: Array  # scalar int32
    combat_use_ultimate: Array  # scalar int32
    movement_action: Array  # scalar int32
    combat_selection_basis_value: Array  # scalar float32
    movement_selection_basis_value: Array  # scalar float32
    combat_selection_basis_components: Array  # (8,) float32
    movement_selection_basis_components: Array  # (9,) float32
    combat_reason_id: Array  # scalar int32
    movement_reason_id: Array  # scalar int32
    fired_guards: Array  # (10,) bool
    combat_peer_count: Array  # scalar int32
    movement_peer_count: Array  # scalar int32


_F32 = jnp.float32
_I32 = jnp.int32
_F32_MAX = jnp.asarray(jnp.finfo(jnp.float32).max, dtype=_F32)
_NEG_INF = jnp.asarray(-jnp.inf, dtype=_F32)


# Private scalar and causal helpers ---


def _clip01(value: Array) -> Array:
    """Clamp a scalar or array to the unit interval."""
    return jnp.clip(value, _F32(0.0), _F32(1.0))


def _class_index(class_id: Array) -> Array:
    """Map a public class ID to its zero-based profile row."""
    return jnp.clip(class_id.astype(_I32) - 1, 0, 4)


def _center(features: Array) -> Array:
    """Read x-y center coordinates from one or more feature rows."""
    return features[..., AGENT_FEATURE_X : AGENT_FEATURE_Y + 1]


def _distance(center_a: Array, center_b: Array) -> Array:
    """Return Euclidean distance between two centers."""
    return cast(Array, jnp.linalg.norm(center_a - center_b))


def _distances(centers: Array, center: Array) -> Array:
    """Return Euclidean distance from each row center to one center."""
    return cast(Array, jnp.linalg.norm(centers - center, axis=1))


def _has_clear_line_of_sight(
    center_a: Array,
    center_b: Array,
    obstacles: Array,
) -> Array:
    """Call the shared LOS predicate under explicit float32 execution."""
    with jax.enable_x64(False):
        return has_clear_line_of_sight(center_a, center_b, obstacles)


def _finite_nonnegative(value: Array) -> Array:
    """Return whether a numeric payload is finite and nonnegative."""
    return jnp.isfinite(value) & (value >= _F32(0.0))


def _finite_nonnegative_product(*factors: Array) -> tuple[Array, Array]:
    """Multiply finite nonnegative factors without overflowing float32."""
    product = jnp.asarray(1.0, dtype=_F32)
    valid = jnp.asarray(True)
    for factor in factors:
        factor = jnp.asarray(factor, dtype=_F32)
        factor_valid = _finite_nonnegative(factor)
        safe_factor = jnp.where(factor_valid, factor, _F32(0.0))
        needs_limit = safe_factor > _F32(1.0)
        divisor = jnp.where(needs_limit, safe_factor, _F32(1.0))
        step_valid = (
            valid & factor_valid & (~needs_limit | (product <= _F32_MAX / divisor))
        )
        candidate = product * jnp.where(step_valid, safe_factor, _F32(0.0))
        step_valid = step_valid & jnp.isfinite(candidate)
        product = jnp.where(
            step_valid,
            candidate,
            _F32(0.0),
        )
        valid = step_valid
    return product.astype(_F32), valid


def _safe_ratio(numerator: Array, denominator: Array, valid: Array) -> Array:
    """Divide valid lanes and return neutral zero for every invalid lane."""
    ratio_valid = (
        valid
        & jnp.isfinite(numerator)
        & jnp.isfinite(denominator)
        & (denominator > _F32(0.0))
    )
    safe_denominator = jnp.where(ratio_valid, denominator, _F32(1.0))
    ratio = numerator / safe_denominator
    ratio_valid = ratio_valid & jnp.isfinite(ratio)
    return jnp.where(ratio_valid, ratio, _F32(0.0))


def _sum_components(components: Array) -> Array:
    """Sum left-to-right in float32 so literal peer equality stays stable."""
    values = jax.lax.optimization_barrier(components.astype(_F32))
    total = values[..., 0]
    for component in range(1, components.shape[-1]):
        total = jax.lax.optimization_barrier(
            (total + values[..., component]).astype(_F32)
        )
    return total


def _health_need(features: Array) -> Array:
    """Return normalized missing-health need, or zero for invalid health."""
    hp = features[..., AGENT_FEATURE_CURRENT_HEALTH]
    max_hp = features[..., AGENT_FEATURE_MAX_HEALTH]
    valid = (
        jnp.isfinite(hp)
        & jnp.isfinite(max_hp)
        & (hp >= _F32(0.0))
        & (max_hp > _F32(0.0))
    )
    return jnp.where(
        valid,
        _F32(1.0) - _clip01(_safe_ratio(hp, max_hp, valid)),
        _F32(0.0),
    )


def _duration_factor(
    duration: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> Array:
    """Map one positive status duration to its bounded control value."""
    p = profile
    value = _F32(p.duration_base) + _F32(p.duration_scale) * _clip01(
        duration / _F32(p.duration_horizon)
    )
    return jnp.where(jnp.isfinite(duration) & (duration > 0), value, _F32(0.0))


def _control_values_valid(
    slow_durations: Array,
    slow_multipliers: Array,
    stun_durations: Array,
    freedom_duration: Array,
    freedom_floor: Array,
) -> Array:
    """Validate the complete slow, stun, and Freedom payload set."""
    active_slow = slow_durations > 0
    slow_valid = (
        jnp.all(jnp.isfinite(slow_durations))
        & jnp.all(slow_durations >= 0)
        & jnp.all(
            ~active_slow | (jnp.isfinite(slow_multipliers) & (slow_multipliers >= 0))
        )
    )
    stun_valid = jnp.all(jnp.isfinite(stun_durations)) & jnp.all(stun_durations >= 0)
    freedom_active = freedom_duration > 0
    freedom_valid = (
        jnp.isfinite(freedom_duration)
        & (freedom_duration >= 0)
        & (
            ~freedom_active
            | (jnp.isfinite(freedom_floor) & (freedom_floor >= _F32(0.0)))
        )
    )
    return slow_valid & stun_valid & freedom_valid


def _normalized_control(
    slow_durations: Array,
    slow_multipliers: Array,
    stun_durations: Array,
    freedom_duration: Array,
    freedom_floor: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array]:
    """Return bounded combined-control and slow-only values."""

    def duration_factor(duration: Array) -> Array:
        """Apply the bound profile's duration normalization."""
        return _duration_factor(duration, profile)

    active_slow = slow_durations > 0
    product = jnp.prod(jnp.where(active_slow, slow_multipliers, _F32(1.0)))
    freedom_active = freedom_duration > 0
    valid_floor = ~freedom_active | (jnp.isfinite(freedom_floor) & (freedom_floor >= 0))
    speed_fraction = jnp.maximum(
        product,
        jnp.where(freedom_active & valid_floor, freedom_floor, _F32(0.0)),
    )
    slow_duration = jnp.max(jax.vmap(duration_factor)(slow_durations))
    slow_value = _clip01(
        _F32(profile.slow_value_scale) * (_F32(1.0) - speed_fraction) * slow_duration
    )
    stun_value = jnp.max(jax.vmap(duration_factor)(stun_durations))
    valid = (
        _control_values_valid(
            slow_durations,
            slow_multipliers,
            stun_durations,
            freedom_duration,
            freedom_floor,
        )
        & valid_floor
    )
    return jnp.where(valid, jnp.maximum(stun_value, slow_value), _F32(0.0)), jnp.where(
        valid, slow_value, _F32(0.0)
    )


def _status_arrays(features: Array) -> tuple[Array, Array, Array, Array, Array]:
    """Collect the three slow, three stun, and Freedom status channels."""
    slow_durations = jnp.stack(
        (
            features[AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION],
            features[AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION],
            features[AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION],
        )
    )
    slow_multipliers = jnp.stack(
        (
            features[AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER],
            features[AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER],
            features[AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER],
        )
    )
    stun_durations = jnp.stack(
        (
            features[AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION],
            features[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION],
            features[AGENT_FEATURE_STUN_ROGUE_POISON_DURATION],
        )
    )
    return (
        slow_durations,
        slow_multipliers,
        stun_durations,
        features[AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION],
        features[AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION],
    )


def _control_input_valid(features: Array) -> Array:
    """Validate the recipient facts needed for control valuation."""
    base_speed = features[AGENT_FEATURE_BASE_MOVEMENT_SPEED]
    return (
        jnp.isfinite(base_speed)
        & (base_speed > _F32(0.0))
        & _control_values_valid(*_status_arrays(features))
    )


def _fresh_effects(
    source: Array, source_class: Array, use_ultimate: Array
) -> tuple[Array, Array, Array, Array, Array]:
    """Decode one class action into its fresh successor-control payloads."""
    zero3 = jnp.zeros((3,), dtype=_F32)
    slow_durations = zero3
    slow_multipliers = jnp.ones((3,), dtype=_F32)
    stun_durations = zero3

    is_warrior_ult = (source_class == WARRIOR_CLASS_ID) & use_ultimate
    is_hunter_basic = (source_class == HUNTER_CLASS_ID) & ~use_ultimate
    is_hunter_ult = (source_class == HUNTER_CLASS_ID) & use_ultimate
    is_rogue_ult = (source_class == ROGUE_CLASS_ID) & use_ultimate
    is_priest_basic = (source_class == PRIEST_CLASS_ID) & ~use_ultimate

    slow_durations = slow_durations.at[0].set(
        jnp.where(
            is_warrior_ult,
            source[AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION],
            0,
        )
    )
    slow_durations = slow_durations.at[1].set(
        jnp.where(
            is_hunter_basic,
            source[AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION],
            0,
        )
    )
    slow_durations = slow_durations.at[2].set(
        jnp.where(
            is_rogue_ult,
            source[AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION],
            0,
        )
    )
    slow_multipliers = slow_multipliers.at[0].set(
        jnp.where(
            is_warrior_ult,
            source[AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER],
            1,
        )
    )
    slow_multipliers = slow_multipliers.at[1].set(
        jnp.where(
            is_hunter_basic,
            source[AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER],
            1,
        )
    )
    slow_multipliers = slow_multipliers.at[2].set(
        jnp.where(
            is_rogue_ult,
            source[AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER],
            1,
        )
    )
    stun_durations = stun_durations.at[0].set(
        jnp.where(
            is_warrior_ult,
            source[AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION],
            0,
        )
    )
    stun_durations = stun_durations.at[1].set(
        jnp.where(
            is_hunter_ult,
            source[AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION],
            0,
        )
    )
    stun_durations = stun_durations.at[2].set(
        jnp.where(
            is_rogue_ult,
            source[AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION],
            0,
        )
    )
    freedom_duration = jnp.where(
        is_priest_basic,
        source[AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION],
        0,
    )
    freedom_floor = jnp.where(
        is_priest_basic,
        source[AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION],
        0,
    )
    return (
        slow_durations,
        slow_multipliers,
        stun_durations,
        freedom_duration,
        freedom_floor,
    )


def _control_components(
    recipient: Array,
    raw_damage: Array,
    fresh_slow_durations: Array,
    fresh_slow_multipliers: Array,
    fresh_stun_durations: Array,
    fresh_freedom_duration: Array,
    fresh_freedom_floor: Array,
    freedom_candidate: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array]:
    """Return ordinary current exposure and incremental first-successor control.

    Current exposure excludes existing Hunter Trap. The candidate comparison
    ages old statuses, breaks aged Trap on positive current raw damage, then
    max-merges fresh source-local effects. Freedom gains zero value while any
    successor stun remains.
    """
    (
        slow_durations,
        slow_multipliers,
        stun_durations,
        freedom_duration,
        freedom_floor,
    ) = _status_arrays(recipient)
    exposed_stuns = stun_durations.at[1].set(_F32(0.0))
    current_exposure, _ = _normalized_control(
        slow_durations,
        slow_multipliers,
        exposed_stuns,
        freedom_duration,
        freedom_floor,
        profile,
    )

    # Compare the deterministic first successor: age old effects, let current
    # damage break old Trap, then max-refresh the source-local fresh effects.
    aged_slow = jnp.maximum(slow_durations - _F32(1.0), _F32(0.0))
    aged_stun = jnp.maximum(stun_durations - _F32(1.0), _F32(0.0))
    aged_freedom = jnp.maximum(freedom_duration - _F32(1.0), _F32(0.0))
    broken_stun = aged_stun.at[1].set(
        jnp.where(
            _finite_nonnegative(raw_damage) & (raw_damage > _F32(0.0)),
            _F32(0.0),
            aged_stun[1],
        )
    )

    with_slow = jnp.maximum(aged_slow, fresh_slow_durations)
    use_fresh_slow = fresh_slow_durations > aged_slow
    with_slow_multipliers = jnp.where(
        use_fresh_slow, fresh_slow_multipliers, slow_multipliers
    )
    with_stun = jnp.maximum(broken_stun, fresh_stun_durations)
    with_freedom = jnp.maximum(aged_freedom, fresh_freedom_duration)
    with_freedom_floor = jnp.where(
        fresh_freedom_duration > aged_freedom, fresh_freedom_floor, freedom_floor
    )

    without_control, without_slow = _normalized_control(
        aged_slow,
        slow_multipliers,
        aged_stun,
        aged_freedom,
        freedom_floor,
        profile,
    )
    with_control, with_slow_value = _normalized_control(
        with_slow,
        with_slow_multipliers,
        with_stun,
        with_freedom,
        with_freedom_floor,
        profile,
    )
    offensive_gain = jnp.maximum(with_control - without_control, _F32(0.0))
    freedom_gain = jnp.where(
        jnp.any(with_stun > 0),
        _F32(0.0),
        jnp.maximum(without_slow - with_slow_value, _F32(0.0)),
    )
    successor_gain = jnp.where(freedom_candidate, freedom_gain, offensive_gain)
    valid = _control_input_valid(recipient) & _control_values_valid(
        fresh_slow_durations,
        fresh_slow_multipliers,
        fresh_stun_durations,
        fresh_freedom_duration,
        fresh_freedom_floor,
    )
    return jnp.where(valid, current_exposure, _F32(0.0)), jnp.where(
        valid, successor_gain, _F32(0.0)
    )


def _post_damage(source: Array, recipient: Array, raw_damage: Array) -> Array:
    """Return offered post-modifier damage before current-HP clipping."""
    burst_duration = source[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION]
    burst_value = source[
        AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER
    ]
    burst = jnp.where(burst_duration > 0, burst_value, _F32(1.0))
    mage_aura = source[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER]
    mitigation = recipient[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER]
    offered, product_valid = _finite_nonnegative_product(
        raw_damage,
        burst,
        mage_aura,
        mitigation,
    )
    valid = _finite_nonnegative(burst_duration) & (
        (raw_damage == _F32(0.0)) | product_valid
    )
    return jnp.where(valid, offered, _F32(0.0))


def _damage_inputs_valid(source: Array, recipient: Array, raw_damage: Array) -> Array:
    """Validate raw damage and only modifiers active in its current chain."""
    burst_duration = source[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION]
    burst_multiplier = source[
        AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER
    ]
    burst = jnp.where(burst_duration > 0, burst_multiplier, _F32(1.0))
    _, product_valid = _finite_nonnegative_product(
        raw_damage,
        burst,
        source[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER],
        recipient[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER],
    )
    return (
        _finite_nonnegative(raw_damage)
        & _finite_nonnegative(burst_duration)
        & ((raw_damage == _F32(0.0)) | product_valid)
    )


def _post_healing(recipient: Array, raw_healing: Array) -> Array:
    """Return offered post-modifier healing before current-HP clipping."""
    poison_duration = recipient[AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION]
    poison_value = recipient[AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER]
    poison = jnp.where(poison_duration > 0, poison_value, _F32(1.0))
    offered, product_valid = _finite_nonnegative_product(raw_healing, poison)
    valid = _finite_nonnegative(poison_duration) & product_valid
    return jnp.where(valid, offered, _F32(0.0))


def _healing_inputs_valid(recipient: Array, raw_healing: Array) -> Array:
    """Validate raw healing and any currently active anti-heal payload."""
    poison_duration = recipient[AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION]
    poison_value = recipient[AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER]
    poison = jnp.where(poison_duration > 0, poison_value, _F32(1.0))
    _, product_valid = _finite_nonnegative_product(raw_healing, poison)
    return _finite_nonnegative(poison_duration) & product_valid


def _ally_dynamic_valid(facts: PolicyFacts) -> Array:
    """Return valid current private rows for own-team candidates."""
    return (
        facts.ally_visible
        & facts.own_active
        & facts.own_alive
        & (facts.own_spawn_shields == 0)
    )


def _enemy_dynamic_valid(facts: PolicyFacts) -> Array:
    """Return valid current private rows for opponent candidates."""
    return _enemy_visible_living(facts) & (facts.enemy_spawn_shields == 0)


def _enemy_visible_living(facts: PolicyFacts) -> Array:
    """Return visible active/living enemies for exact crowd predicates."""
    return facts.enemy_visible & facts.enemy_active & facts.enemy_alive


def _lane_threat(
    source: Array,
    source_valid: Array,
    source_class: Array,
    recipient: Array,
    recipient_valid: Array,
    recipient_effect_available: Array,
    use_ultimate: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array]:
    """Return one capability-and-reach threat upper bound and its validity."""
    raw_damage = jnp.where(
        use_ultimate,
        source[AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE],
        source[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE],
    )
    radius = jnp.where(
        use_ultimate,
        source[AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS],
        source[AGENT_FEATURE_BASIC_INTERACTION_RADIUS],
    )
    fresh = _fresh_effects(source, source_class, use_ultimate)
    fresh_valid = _control_values_valid(*fresh)
    fresh_slow = jnp.any((fresh[0] > 0) & (fresh[1] < _F32(1.0)))
    fresh_stun = jnp.any(fresh[2] > 0)
    has_fresh_control = fresh_valid & (fresh_slow | fresh_stun)
    damage_valid = _damage_inputs_valid(source, recipient, raw_damage)
    has_effect = (damage_valid & (raw_damage > 0)) | has_fresh_control
    geometry_valid = (
        jnp.all(jnp.isfinite(_center(source)))
        & jnp.all(jnp.isfinite(_center(recipient)))
        & jnp.isfinite(radius)
        & (radius > 0)
    )
    distance = _distance(_center(source), _center(recipient))
    cooldown_ready = source[AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING] == 0
    hp = recipient[AGENT_FEATURE_CURRENT_HEALTH]
    max_hp = recipient[AGENT_FEATURE_MAX_HEALTH]
    health_valid = (
        jnp.isfinite(hp)
        & jnp.isfinite(max_hp)
        & (hp >= _F32(0.0))
        & (max_hp > _F32(0.0))
    )
    recipient_fact_valid = health_valid & _control_input_valid(recipient)
    lane_valid = (
        source_valid
        & recipient_valid
        & recipient_fact_valid
        & geometry_valid
        & (distance <= radius)
        & has_effect
        & (~use_ultimate | cooldown_ready)
    )

    damage = _post_damage(source, recipient, raw_damage)
    damage_potency = _clip01(
        _safe_ratio(
            jnp.minimum(damage, jnp.maximum(hp, _F32(0.0))),
            _F32(profile.damage_potency_fraction) * max_hp,
            health_valid & damage_valid,
        )
    )
    _, control_gain = _control_components(
        recipient,
        raw_damage,
        *fresh,
        jnp.asarray(False),
        profile,
    )
    control_gain = jnp.where(fresh_valid, control_gain, _F32(0.0))
    lane_effect = jnp.maximum(
        jnp.where(health_valid, damage_potency, _F32(0.0)), control_gain
    )
    return (
        jnp.where(lane_valid & recipient_effect_available, lane_effect, _F32(0.0)),
        lane_valid,
    )


def _threat_tensor(
    facts: PolicyFacts,
    focal_center: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return same-epoch capability-and-evaluated-reach threat.

    Threat uses observed ready capabilities, never predicted actions.
    Countered-by can amplify an already-positive soft value but cannot create
    threat.
    """
    recipients = jnp.concatenate(
        (facts.focal_features[jnp.newaxis, :], facts.ally_features), axis=0
    )
    recipients = recipients.at[0, AGENT_FEATURE_X : AGENT_FEATURE_Y + 1].set(
        focal_center
    )
    focal_lifecycle = (facts.focal_features[AGENT_FEATURE_ACTIVE] > 0) & (
        facts.focal_features[AGENT_FEATURE_ALIVE] > 0
    )
    # Known true gives valid zero threat; known false uses the ordinary formula.
    # Unknown is conservatively unshielded for focal threat/risk only, with no
    # duration or expiry inference.
    focal_effect_available = facts.focal_shield_state != FOCAL_SHIELD_KNOWN_TRUE
    ally_valid = facts.ally_visible & facts.own_active & facts.own_alive
    ally_effect_available = (
        facts.ally_visible
        & facts.own_active
        & facts.own_alive
        & (facts.own_spawn_shields == 0)
    )
    recipient_valid = jnp.concatenate((focal_lifecycle[jnp.newaxis], ally_valid))
    recipient_effect_available = jnp.concatenate(
        (focal_effect_available[jnp.newaxis], ally_effect_available)
    )

    enemy_stuns = facts.enemy_features[
        :,
        AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION : (
            AGENT_FEATURE_STUN_ROGUE_POISON_DURATION + 1
        ),
    ]
    enemy_stun_valid = jnp.all(jnp.isfinite(enemy_stuns) & (enemy_stuns >= 0), axis=1)
    enemy_stunned = jnp.any(enemy_stuns > 0, axis=1)
    source_valid = (
        facts.enemy_visible
        & facts.enemy_active
        & facts.enemy_alive
        & (facts.enemy_spawn_shields == 0)
        & enemy_stun_valid
        & ~enemy_stunned
    )

    def enemy_threat(
        source: Array, valid: Array, class_id: Array
    ) -> tuple[Array, Array]:
        """Build one enemy's threat row across focal and ally recipients."""

        def recipient_threat(
            recipient: Array, recipient_ok: Array, recipient_available: Array
        ) -> tuple[Array, Array]:
            """Reduce one enemy-recipient pair across Basic and Ultimate lanes."""
            basic, basic_valid = _lane_threat(
                source,
                valid,
                class_id,
                recipient,
                recipient_ok,
                recipient_available,
                jnp.asarray(False),
                profile,
            )
            ultimate, ultimate_valid = _lane_threat(
                source,
                valid,
                class_id,
                recipient,
                recipient_ok,
                recipient_available,
                jnp.asarray(True),
                profile,
            )
            return jnp.maximum(basic, ultimate), basic_valid | ultimate_valid

        return jax.vmap(recipient_threat)(
            recipients, recipient_valid, recipient_effect_available
        )

    base, threat_valid = jax.vmap(enemy_threat)(
        facts.enemy_features,
        source_valid,
        facts.enemy_class_ids,
    )
    class_index = _class_index(facts.focal_features[AGENT_FEATURE_CLASS_ID])
    enemy_class_index = _class_index(facts.enemy_class_ids)
    countered_by = jnp.asarray(profile.countered_by, dtype=_F32)
    authored_edge = countered_by[class_index, enemy_class_index] > 0
    soft = jnp.where(
        (base > 0) & authored_edge[:, jnp.newaxis],
        _clip01(base * _F32(profile.countered_by_multiplier)),
        base,
    )
    base = jnp.where(threat_valid, base, _F32(0.0))
    soft = jnp.where(threat_valid, soft, _F32(0.0))
    credible = _clip01(
        _F32(1.0) - jnp.prod(_F32(1.0) - jnp.where(threat_valid, base, 0), axis=0)
    )
    soft_credible = _clip01(
        _F32(1.0) - jnp.prod(_F32(1.0) - jnp.where(threat_valid, soft, 0), axis=0)
    )
    return base, soft, threat_valid, credible, soft_credible


def _candidate_rows(facts: PolicyFacts) -> tuple[Array, Array, Array, Array]:
    """Assemble target-none, ally, and enemy rows in action-target order."""
    zero = jnp.zeros_like(facts.focal_features)
    rows = jnp.concatenate(
        (zero[jnp.newaxis, :], facts.ally_features, facts.enemy_features), axis=0
    )
    classes = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=_I32),
            facts.own_class_ids.astype(_I32),
            facts.enemy_class_ids.astype(_I32),
        )
    )
    visible = jnp.concatenate(
        (jnp.array([False]), facts.ally_visible, facts.enemy_visible)
    )
    lifecycle = jnp.concatenate(
        (
            jnp.array([False]),
            facts.own_active & facts.own_alive,
            facts.enemy_active & facts.enemy_alive,
        )
    )
    return rows, classes, visible, lifecycle


# Combat scoring ---


def _combat_workspaces_impl(
    facts: PolicyFacts,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Build fixed combat components, utilities, payloads, guards, and threat."""
    p = profile
    focal = facts.focal_features
    focal_class = focal[AGENT_FEATURE_CLASS_ID].astype(_I32)
    rows, classes, visible, lifecycle = _candidate_rows(facts)
    target_shields = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=facts.own_spawn_shields.dtype),
            facts.own_spawn_shields,
            facts.enemy_spawn_shields,
        )
    )
    # These rows gate valuation from observable lifecycle and shield facts. The
    # exact pair mask remains the sole combat-legality authority.
    candidate_valid = visible & lifecycle & (target_shields == 0)
    target_id = jnp.arange(NUM_TARGET_ACTIONS, dtype=_I32)[:, jnp.newaxis]
    ultimate = jnp.arange(NUM_ULTIMATE_ACTIONS, dtype=_I32)[jnp.newaxis, :]
    enemy_target = target_id >= 6
    ally_target = (target_id >= 1) & (target_id <= 5)
    basic = ultimate == 0
    use_ultimate = ultimate == 1

    basic_damage = focal[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE]
    ultimate_damage = focal[AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE]
    basic_healing = focal[AGENT_FEATURE_CAPABILITY_BASIC_HEALING]
    ultimate_healing = focal[AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING]
    raw_damage = jnp.where(
        enemy_target,
        jnp.where(basic, basic_damage, ultimate_damage),
        _F32(0.0),
    )
    raw_healing = jnp.where(
        ally_target,
        jnp.where(basic, basic_healing, ultimate_healing),
        _F32(0.0),
    )

    def post_basic_damage(recipient: Array) -> Array:
        """Apply the focal Basic damage chain to one candidate row."""
        return _post_damage(focal, recipient, basic_damage)

    def post_ultimate_damage(recipient: Array) -> Array:
        """Apply the focal Ultimate damage chain to one candidate row."""
        return _post_damage(focal, recipient, ultimate_damage)

    def post_basic_healing(recipient: Array) -> Array:
        """Apply the focal Basic healing chain to one candidate row."""
        return _post_healing(recipient, basic_healing)

    def post_ultimate_healing(recipient: Array) -> Array:
        """Apply the focal Ultimate healing chain to one candidate row."""
        return _post_healing(recipient, ultimate_healing)

    def basic_damage_valid(recipient: Array) -> Array:
        """Validate the focal Basic damage chain for one candidate row."""
        return _damage_inputs_valid(focal, recipient, basic_damage)

    def ultimate_damage_valid(recipient: Array) -> Array:
        """Validate the focal Ultimate damage chain for one candidate row."""
        return _damage_inputs_valid(focal, recipient, ultimate_damage)

    def basic_healing_valid(recipient: Array) -> Array:
        """Validate the focal Basic healing chain for one candidate row."""
        return _healing_inputs_valid(recipient, basic_healing)

    def ultimate_healing_valid(recipient: Array) -> Array:
        """Validate the focal Ultimate healing chain for one candidate row."""
        return _healing_inputs_valid(recipient, ultimate_healing)

    post_damage_by_target = jax.vmap(post_basic_damage)(rows)
    post_ultimate_damage_by_target = jax.vmap(post_ultimate_damage)(rows)
    post_healing_by_target = jax.vmap(post_basic_healing)(rows)
    post_ultimate_healing_by_target = jax.vmap(post_ultimate_healing)(rows)
    basic_damage_valid_by_target = jax.vmap(basic_damage_valid)(rows)
    ultimate_damage_valid_by_target = jax.vmap(ultimate_damage_valid)(rows)
    basic_healing_valid_by_target = jax.vmap(basic_healing_valid)(rows)
    ultimate_healing_valid_by_target = jax.vmap(ultimate_healing_valid)(rows)
    post_damage = jnp.where(
        enemy_target,
        jnp.where(
            basic,
            post_damage_by_target[:, None],
            post_ultimate_damage_by_target[:, None],
        ),
        0,
    )
    post_healing = jnp.where(
        ally_target,
        jnp.where(
            basic,
            post_healing_by_target[:, None],
            post_ultimate_healing_by_target[:, None],
        ),
        0,
    )
    damage_lane_valid = jnp.where(
        basic,
        basic_damage_valid_by_target[:, None],
        ultimate_damage_valid_by_target[:, None],
    )
    healing_lane_valid = jnp.where(
        basic,
        basic_healing_valid_by_target[:, None],
        ultimate_healing_valid_by_target[:, None],
    )
    payload_lane_valid = jnp.where(
        enemy_target,
        damage_lane_valid,
        jnp.where(ally_target, healing_lane_valid, True),
    )
    payload = jnp.where(candidate_valid[:, None], post_damage + post_healing, 0)

    hp = rows[:, AGENT_FEATURE_CURRENT_HEALTH][:, None]
    max_hp = rows[:, AGENT_FEATURE_MAX_HEALTH][:, None]
    missing_hp = jnp.maximum(max_hp - hp, _F32(0.0))
    need = jnp.where(enemy_target, hp, jnp.where(ally_target, missing_hp, 0))
    current_hp_valid = jnp.isfinite(hp) & (hp >= 0) & candidate_valid[:, None]
    health_valid = current_hp_valid & jnp.isfinite(max_hp) & (max_hp > 0)
    realized = jnp.where(
        health_valid & (need > 0), jnp.minimum(payload, need), _F32(0.0)
    )
    positive_need = health_valid & (need > 0)
    coverage = _clip01(_safe_ratio(realized, need, positive_need))
    potency = _clip01(
        _safe_ratio(
            realized,
            _F32(p.damage_potency_fraction) * max_hp,
            health_valid,
        )
    )
    impact = _clip01(_F32(p.impact_mix[0]) * potency + _F32(p.impact_mix[1]) * coverage)
    excess = _clip01(
        _safe_ratio(
            jnp.maximum(payload - realized, _F32(0.0)),
            payload,
            health_valid & (payload > 0),
        )
    )
    health_need = jax.vmap(_health_need)(rows)[:, None]
    damage_urgency = _clip01(
        _F32(p.urgency_mix[0]) * health_need + _F32(p.urgency_mix[1]) * (payload >= hp)
    )
    urgency = jnp.where(
        health_valid & payload_lane_valid,
        jnp.where(enemy_target, damage_urgency, jnp.where(ally_target, health_need, 0)),
        0,
    )

    target_distance = _distances(_center(rows), _center(focal))[:, None]
    radius = jnp.where(
        basic,
        focal[AGENT_FEATURE_BASIC_INTERACTION_RADIUS],
        focal[AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS],
    )
    geometry_valid = (
        candidate_valid[:, None]
        & jnp.all(jnp.isfinite(_center(rows)), axis=1)[:, None]
        & jnp.all(jnp.isfinite(_center(focal)))
        & jnp.isfinite(radius)
        & (radius > 0)
    )
    access = _clip01(_safe_ratio(radius - target_distance, radius, geometry_valid))

    def target_control(recipient: Array, target_index: Array) -> tuple[Array, Array]:
        """Build current and successor control for both lanes of one target."""

        def lane_control(lane: Array) -> tuple[Array, Array]:
            """Build control values for one target-action lane."""
            fresh = _fresh_effects(focal, focal_class, lane == 1)
            freedom = (focal_class == PRIEST_CLASS_ID) & (lane == 0)
            current, gain = _control_components(
                recipient,
                jnp.where(
                    target_index >= 6,
                    jnp.where(lane == 0, basic_damage, ultimate_damage),
                    0,
                ),
                *fresh,
                freedom,
                profile,
            )
            return current, gain

        return jax.vmap(lane_control)(jnp.arange(2, dtype=_I32))

    current_control, successor_gain = jax.vmap(target_control)(
        rows, jnp.arange(NUM_TARGET_ACTIONS, dtype=_I32)
    )
    successor_gain = jnp.where(candidate_valid[:, None], successor_gain, 0)
    effect_positive = (realized > 0) | (successor_gain > 0)
    current_exposure = jnp.where(enemy_target & effect_positive, current_control, 0)
    control = jnp.where(
        effect_positive, jnp.maximum(current_exposure, successor_gain), 0
    )

    _, soft_threat, threat_valid, _, soft_credible = _threat_tensor(
        facts, _center(focal), profile
    )
    enemy_threat = jnp.max(jnp.where(threat_valid, soft_threat, 0), axis=1)
    mapped_threat = jnp.zeros((NUM_TARGET_ACTIONS,), dtype=_F32)
    mapped_threat = mapped_threat.at[1:6].set(soft_credible[1:])
    mapped_threat = mapped_threat.at[6:11].set(enemy_threat)
    threat_value = jnp.where(effect_positive, mapped_threat[:, None], 0)

    focal_index = _class_index(focal_class)
    target_index = _class_index(classes)[:, None]
    counters = jnp.asarray(p.counters, dtype=_F32)
    counter_edge = counters[focal_index, target_index]
    eligible_counter_lane = basic | ((focal_class == WARRIOR_CLASS_ID) & use_ultimate)
    counter_effect = (realized > 0) | (successor_gain > 0)
    counter_value = jnp.where(
        enemy_target & eligible_counter_lane & counter_effect & (counter_edge > 0),
        _F32(p.offensive_counter_value),
        0,
    )
    role_values = jnp.asarray(p.role_class_values, dtype=_F32)[target_index]
    rogue_role = jnp.where(
        (focal_class == ROGUE_CLASS_ID) & enemy_target & basic,
        role_values,
        0,
    )
    priest_role = jnp.where(
        (focal_class == PRIEST_CLASS_ID) & ally_target & basic & (realized > 0),
        role_values,
        0,
    )
    class_value = counter_value + rogue_role + priest_role

    combat_weights = jnp.asarray(p.combat_weights, dtype=_F32)
    weights = combat_weights[focal_index]
    weights = jnp.where(
        (focal_class >= MAGE_CLASS_ID) & (focal_class <= PRIEST_CLASS_ID),
        weights,
        jnp.zeros((5,), dtype=_F32),
    )
    # The eight trace slots are the only combat-score owners. TDM contributes
    # literal zero in the final slot and no hidden task urgency exists.
    components = jnp.stack(
        (
            weights[0] * impact,
            weights[1] * urgency,
            weights[2] * threat_value,
            weights[3] * control,
            weights[4] * access,
            -_F32(p.excess_weight) * excess,
            class_value,
            jnp.full_like(impact, _F32(p.tdm_task_value)),
        ),
        axis=-1,
    ).astype(_F32)
    utility = _sum_components(components)
    noop = (target_id == 0) & basic
    components = jnp.where(noop[..., None], _F32(0.0), components)
    utility = jnp.where(noop, _F32(0.0), utility)

    target_trap = rows[:, AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION][:, None]
    # Preserve an aged Trap only for valid positive damage that would not
    # currently finish the recipient. Duration one is deliberately attackable.
    trap_suppressed = (
        damage_lane_valid
        & current_hp_valid
        & _finite_nonnegative(raw_damage)
        & (raw_damage > 0)
        & (post_damage < hp)
        & _finite_nonnegative(target_trap)
        & (target_trap > _F32(p.trap_preserve_duration))
    )
    return (
        components,
        utility.astype(_F32),
        raw_damage.astype(_F32),
        raw_healing.astype(_F32),
        post_damage.astype(_F32),
        realized.astype(_F32),
        successor_gain.astype(_F32),
        trap_suppressed,
        soft_threat,
    )


def _combat_workspaces(
    facts: PolicyFacts,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Build combat workspaces without allowing global x64 widening."""
    with jax.enable_x64(False):
        return _combat_workspaces_impl(facts, profile)


def _ordinary_combat_support(mask: Array, trap_suppressed: Array) -> Array:
    """Admit mask-legal Basic pairs not suppressed by aged-Trap preservation."""
    basic_lane = jnp.arange(NUM_ULTIMATE_ACTIONS, dtype=_I32) == _I32(0)
    return mask & basic_lane[jnp.newaxis, :] & ~trap_suppressed


def _maximum_combat_support(admitted: Array, utility: Array) -> Array:
    """Retain every admitted candidate exactly equal to the maximum utility."""
    maximum = jnp.max(jnp.where(admitted, utility, _NEG_INF))
    return admitted & (utility == maximum)


def _visible_enemies_in_focal_basic(facts: PolicyFacts) -> Array:
    """Return visible living enemies inside the focal Basic radius."""
    radius = facts.focal_features[AGENT_FEATURE_BASIC_INTERACTION_RADIUS]
    geometry_valid = (
        jnp.all(jnp.isfinite(_center(facts.focal_features)))
        & jnp.all(jnp.isfinite(_center(facts.enemy_features)), axis=1)
        & jnp.isfinite(radius)
        & (radius > 0)
    )
    distance = _distances(_center(facts.enemy_features), _center(facts.focal_features))
    return _enemy_visible_living(facts) & geometry_valid & (distance <= radius)


# Class-specific combat support ---


def _mage_combat(
    facts: PolicyFacts,
    mask: Array,
    ordinary: Array,
    utility: Array,
    post_damage: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array, Array, Array]:
    """Hard-select Burst at the positive-count, crowd, and no-cover thresholds."""
    p = profile
    visible_in_range = jnp.sum(_visible_enemies_in_focal_basic(facts), dtype=_I32)
    count_valid = facts.enemy_configured_count > 0
    crowd_threshold = jnp.minimum(
        _I32(p.mage_burst_crowd), facts.enemy_configured_count.astype(_I32)
    )
    enemy_hp = facts.enemy_features[:, AGENT_FEATURE_CURRENT_HEALTH]
    raw_basic_damage = facts.focal_features[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE]

    def basic_damage_is_valid(recipient: Array) -> Array:
        """Validate the focal Basic damage chain for one enemy row."""
        return _damage_inputs_valid(
            facts.focal_features,
            recipient,
            raw_basic_damage,
        )

    basic_damage_valid = jax.vmap(basic_damage_is_valid)(facts.enemy_features)
    covering_basic = jnp.any(
        ordinary[6:, 0]
        & _enemy_dynamic_valid(facts)
        & basic_damage_valid
        & _finite_nonnegative(enemy_hp)
        & (post_damage[6:, 0] > _F32(0.0))
        & (post_damage[6:, 0] >= enemy_hp)
    )
    trigger = (
        mask[0, 1]
        & count_valid
        & (visible_in_range >= crowd_threshold)
        & ~covering_basic
    )
    trigger_support = jnp.zeros_like(mask).at[0, 1].set(True)
    support = jnp.where(
        trigger, trigger_support, _maximum_combat_support(ordinary, utility)
    )
    basis = jnp.where(trigger, jnp.zeros_like(utility), utility)
    reason = jnp.where(trigger, _I32(MAGE_BURST_TRIGGER), _I32(DIRECT_SCORE)).astype(
        _I32
    )
    return support, basis, reason, trigger


def _nonfocal_geometry_proved(
    facts: PolicyFacts,
    target: Array,
    target_radius: Array,
    use_ally_radius: Array,
) -> Array:
    """Prove nonfocal ally geometry without decoding an ally-row identity.

    Rogue uses each visible active/living ally's Basic radius; Priest uses the
    target's Basic radius. If the focal could qualify, a second row is required.
    This is geometry, not a follow-up damage proof.
    """
    ally_radius = facts.ally_features[:, AGENT_FEATURE_BASIC_INTERACTION_RADIUS]
    radius = jnp.where(use_ally_radius, ally_radius, target_radius)
    distance = _distances(_center(facts.ally_features), _center(target))
    valid = (
        facts.ally_visible
        & facts.own_active
        & facts.own_alive
        & jnp.all(jnp.isfinite(_center(facts.ally_features)), axis=1)
        & jnp.all(jnp.isfinite(_center(target)))
        & jnp.isfinite(radius)
        & (radius > 0)
    )
    qualifying_count = jnp.sum(valid & (distance <= radius), dtype=_I32)

    focal_radius = jnp.where(
        use_ally_radius,
        facts.focal_features[AGENT_FEATURE_BASIC_INTERACTION_RADIUS],
        target_radius,
    )
    focal_distance = _distance(_center(facts.focal_features), _center(target))
    focal_qualifies = (
        jnp.all(jnp.isfinite(_center(facts.focal_features)))
        & jnp.all(jnp.isfinite(_center(target)))
        & jnp.isfinite(focal_radius)
        & (focal_radius > 0)
        & (focal_distance <= focal_radius)
    )
    required_count = jnp.where(focal_qualifies, _I32(2), _I32(1))
    return qualifying_count >= required_count


def _warrior_combat(
    facts: PolicyFacts,
    mask: Array,
    ordinary: Array,
    utility: Array,
    trap_suppressed: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array, Array, Array]:
    """Admit the four exact Charge predicates and rank qualifiers ordinarily."""
    p = profile

    def target_qualifies(target: Array, class_id: Array) -> Array:
        """Evaluate one target against the directional Charge predicates."""
        rogue = (class_id == ROGUE_CLASS_ID) & _nonfocal_geometry_proved(
            facts,
            target,
            target[AGENT_FEATURE_BASIC_INTERACTION_RADIUS],
            jnp.asarray(True),
        )
        priest = (class_id == PRIEST_CLASS_ID) & _nonfocal_geometry_proved(
            facts,
            target,
            target[AGENT_FEATURE_BASIC_INTERACTION_RADIUS],
            jnp.asarray(False),
        )
        burst_duration = target[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION]
        mage = (
            (class_id == MAGE_CLASS_ID)
            & _finite_nonnegative(burst_duration)
            & (burst_duration >= _F32(p.warrior_mage_burst_min))
        )
        configured_counter_class = facts.enemy_active & (
            (facts.enemy_class_ids == MAGE_CLASS_ID)
            | (facts.enemy_class_ids == PRIEST_CLASS_ID)
        )
        # Absence means no configured-active Mage/Priest slot. Death or respawn
        # does not make either configured class absent.
        absence = ~jnp.any(configured_counter_class) & (
            (class_id == WARRIOR_CLASS_ID)
            | (class_id == ROGUE_CLASS_ID)
            | (class_id == HUNTER_CLASS_ID)
        )
        return rogue | priest | mage | absence

    predicate = jax.vmap(target_qualifies)(facts.enemy_features, facts.enemy_class_ids)
    target_trap = facts.enemy_features[:, AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION]
    focal_hp = facts.focal_features[AGENT_FEATURE_CURRENT_HEALTH]
    health_gate = _finite_nonnegative(focal_hp) & (
        focal_hp >= _F32(p.warrior_charge_health)
    )
    target_valid = _enemy_dynamic_valid(facts)
    trap_valid = target_valid & _finite_nonnegative(target_trap)
    support = jnp.zeros_like(mask)
    support = support.at[6:, 1].set(
        mask[6:, 1]
        & ~trap_suppressed[6:, 1]
        & health_gate
        & trap_valid
        & (target_trap <= _F32(p.warrior_charge_trap_max))
        & predicate
    )
    trigger = jnp.any(support)
    charge_maximum = jnp.max(jnp.where(support, utility, _NEG_INF))
    charge_support = support & (utility == charge_maximum)
    final_support = jnp.where(
        trigger, charge_support, _maximum_combat_support(ordinary, utility)
    )
    reason = jnp.where(
        trigger, _I32(WARRIOR_CHARGE_TRIGGER), _I32(DIRECT_SCORE)
    ).astype(_I32)
    return final_support, utility, reason, trigger


def _hunter_combat_impl(
    facts: PolicyFacts,
    mask: Array,
    ordinary: Array,
    utility: Array,
    trap_suppressed: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array, Array, Array]:
    """Use the first legal emergency, Priest-crowd, or no-Priest Trap branch."""
    p = profile
    in_basic = _visible_enemies_in_focal_basic(facts)
    legal_trap = mask[6:, 1] & ~trap_suppressed[6:, 1]
    focal_center = _center(facts.focal_features)

    def emergency_qualifies(enemy: Array, class_id: Array) -> Array:
        """Return whether one enemy currently threatens the emergency Hunter."""
        basic_radius = enemy[AGENT_FEATURE_BASIC_INTERACTION_RADIUS]
        ultimate_radius = enemy[AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS]
        distance = _distance(_center(enemy), focal_center)
        fresh = _fresh_effects(enemy, class_id, jnp.asarray(True))
        ultimate_damage = enemy[AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE]
        fresh_control = _control_values_valid(*fresh) & (
            jnp.any((fresh[0] > 0) & (fresh[1] < _F32(1.0))) | jnp.any(fresh[2] > 0)
        )
        ultimate_effect = (
            _finite_nonnegative(ultimate_damage) & (ultimate_damage > 0)
        ) | fresh_control
        valid_geometry = jnp.all(jnp.isfinite(_center(enemy))) & jnp.all(
            jnp.isfinite(focal_center)
        )
        basic_reach = (
            jnp.isfinite(basic_radius) & (basic_radius > 0) & (distance <= basic_radius)
        )
        ultimate_reach = (
            ultimate_effect
            & jnp.isfinite(ultimate_radius)
            & (ultimate_radius > 0)
            & (distance <= ultimate_radius)
        )
        return valid_geometry & (basic_reach | ultimate_reach)

    capability_reach = jax.vmap(emergency_qualifies)(
        facts.enemy_features, facts.enemy_class_ids
    )
    valid_target = _enemy_dynamic_valid(facts)
    emergency_qualifier = legal_trap & valid_target & capability_reach
    base_threat, _, threat_valid, _, _ = _threat_tensor(facts, focal_center, profile)
    emergency_value = jnp.where(threat_valid[:, 0], base_threat[:, 0], _F32(0.0))
    emergency_max = jnp.max(jnp.where(emergency_qualifier, emergency_value, _NEG_INF))
    emergency_support_rows = emergency_qualifier & (emergency_value == emergency_max)
    focal_hp = facts.focal_features[AGENT_FEATURE_CURRENT_HEALTH]
    emergency = (
        _finite_nonnegative(focal_hp)
        & (focal_hp < _F32(p.hunter_emergency_health))
        & jnp.any(emergency_qualifier)
    )

    priest_qualifier = (
        legal_trap
        & valid_target
        & in_basic
        & (facts.enemy_class_ids == PRIEST_CLASS_ID)
    )
    priest_crowd = (jnp.sum(in_basic, dtype=_I32) >= _I32(p.hunter_crowd)) & jnp.any(
        priest_qualifier
    )
    priest_max = jnp.max(jnp.where(priest_qualifier, utility[6:, 1], _NEG_INF))
    priest_support_rows = priest_qualifier & (utility[6:, 1] == priest_max)

    configured_priest = jnp.any(
        facts.enemy_active & (facts.enemy_class_ids == PRIEST_CLASS_ID)
    )
    raw_hp = facts.enemy_features[:, AGENT_FEATURE_CURRENT_HEALTH]
    no_priest_qualifier = (
        legal_trap & valid_target & in_basic & _finite_nonnegative(raw_hp)
    )
    no_priest = (
        ~configured_priest
        & (facts.enemy_configured_count >= _I32(p.hunter_crowd))
        & (jnp.sum(in_basic, dtype=_I32) >= _I32(p.hunter_crowd))
        & jnp.any(no_priest_qualifier)
    )
    hp_max = jnp.max(jnp.where(no_priest_qualifier, raw_hp, _NEG_INF))
    no_priest_support_rows = no_priest_qualifier & (raw_hp == hp_max)

    branch = jnp.where(
        emergency,
        _I32(1),
        jnp.where(
            priest_crowd,
            _I32(2),
            jnp.where(no_priest, _I32(3), _I32(0)),
        ),
    ).astype(_I32)

    ordinary_support = _maximum_combat_support(ordinary, utility)
    emergency_support = jnp.zeros_like(mask).at[6:, 1].set(emergency_support_rows)
    priest_support = jnp.zeros_like(mask).at[6:, 1].set(priest_support_rows)
    no_priest_support = jnp.zeros_like(mask).at[6:, 1].set(no_priest_support_rows)
    emergency_basis = jnp.zeros_like(utility).at[6:, 1].set(emergency_value)
    no_priest_basis = jnp.zeros_like(utility).at[6:, 1].set(raw_hp)
    support = jnp.where(
        branch == _I32(1),
        emergency_support,
        jnp.where(
            branch == _I32(2),
            priest_support,
            jnp.where(branch == _I32(3), no_priest_support, ordinary_support),
        ),
    )
    basis = jnp.where(
        branch == _I32(1),
        emergency_basis,
        jnp.where(branch == _I32(3), no_priest_basis, utility),
    )
    reason = jnp.where(
        branch == _I32(1),
        _I32(HUNTER_TRAP_EMERGENCY),
        jnp.where(
            branch == _I32(2),
            _I32(HUNTER_TRAP_PRIEST_CROWD),
            jnp.where(
                branch == _I32(3),
                _I32(HUNTER_TRAP_NO_PRIEST_CROWD),
                _I32(DIRECT_SCORE),
            ),
        ),
    ).astype(_I32)
    return support, basis, reason, branch > _I32(0)


def _hunter_combat(
    facts: PolicyFacts,
    mask: Array,
    ordinary: Array,
    utility: Array,
    trap_suppressed: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array, Array, Array]:
    """Run Hunter support construction without global x64 widening."""
    with jax.enable_x64(False):
        return _hunter_combat_impl(
            facts,
            mask,
            ordinary,
            utility,
            trap_suppressed,
            profile,
        )


def _rogue_combat(
    mask: Array,
    ordinary: Array,
    utility: Array,
) -> tuple[Array, Array, Array, Array]:
    """Rank Basics against no-op before same-target post-draw Poison."""
    basic_targets = ordinary[:, 0] & (jnp.arange(NUM_TARGET_ACTIONS, dtype=_I32) > 0)
    no_op = ordinary[0, 0]
    has_basic = jnp.any(basic_targets)
    maximum = jnp.max(jnp.where(basic_targets, utility[:, 0], _NEG_INF))
    best_basics = basic_targets & (utility[:, 0] == maximum)
    keep_basics = has_basic & (~no_op | (maximum >= _F32(0.0)))
    keep_no_op = no_op & (~has_basic | (maximum <= _F32(0.0)))
    support = jnp.zeros_like(mask)
    support = support.at[:, 0].set(jnp.where(keep_basics, best_basics, False))
    support = support.at[0, 0].set(keep_no_op)
    return support, utility, jnp.asarray(DIRECT_SCORE, dtype=_I32), jnp.asarray(False)


def _priest_combat_impl(
    facts: PolicyFacts,
    mask: Array,
    ordinary: Array,
    utility: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array, Array, Array]:
    """Apply raw-HP Holy Word priority: Priest, Mage, Rogue, Hunter, Warrior."""
    p = profile
    recipient_hp = facts.ally_features[:, AGENT_FEATURE_CURRENT_HEALTH]
    recipient_class = facts.own_class_ids
    class_valid = (recipient_class >= MAGE_CLASS_ID) & (
        recipient_class <= PRIEST_CLASS_ID
    )
    qualifier = (
        mask[1:6, 1]
        & _ally_dynamic_valid(facts)
        & _finite_nonnegative(recipient_hp)
        & (recipient_hp < _F32(p.priest_holy_word_health))
        & class_valid
    )
    priority = jnp.where(
        facts.own_class_ids == PRIEST_CLASS_ID,
        _I32(0),
        jnp.where(
            facts.own_class_ids == MAGE_CLASS_ID,
            _I32(1),
            jnp.where(
                facts.own_class_ids == ROGUE_CLASS_ID,
                _I32(2),
                jnp.where(
                    facts.own_class_ids == HUNTER_CLASS_ID,
                    _I32(3),
                    _I32(4),
                ),
            ),
        ),
    ).astype(_I32)
    winning_priority = jnp.min(jnp.where(qualifier, priority, _I32(5)))
    holy_rows = qualifier & (priority == winning_priority)
    trigger = jnp.any(qualifier)
    holy_support = jnp.zeros_like(mask).at[1:6, 1].set(holy_rows)
    support = jnp.where(
        trigger, holy_support, _maximum_combat_support(ordinary, utility)
    )
    basis = jnp.where(trigger, jnp.zeros_like(utility), utility)
    reason = jnp.where(
        trigger, _I32(PRIEST_HOLY_WORD_TRIGGER), _I32(DIRECT_SCORE)
    ).astype(_I32)
    return support, basis, reason, trigger


def _priest_combat(
    facts: PolicyFacts,
    mask: Array,
    ordinary: Array,
    utility: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array, Array, Array]:
    """Run Priest support construction without global x64 widening."""
    with jax.enable_x64(False):
        return _priest_combat_impl(facts, mask, ordinary, utility, profile)


# Movement scoring ---


def _band_quality(
    distance: Array,
    scale: Array,
    band: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> Array:
    """Score a normalized distance against one inclusive desired band."""
    valid = jnp.isfinite(distance) & jnp.isfinite(scale) & (scale > 0)
    normalized = _safe_ratio(distance, scale, valid)
    band_distance = jnp.maximum(
        jnp.maximum(band[0] - normalized, _F32(0.0)),
        normalized - band[1],
    )
    quality = _clip01(_F32(1.0) - band_distance / _F32(profile.quality_falloff))
    return jnp.where(valid, quality, _F32(0.0))


def _movement_range_quality(
    facts: PolicyFacts,
    endpoints: Array,
    selected_target: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> Array:
    """Score each intended endpoint against the selected target's Basic band."""
    rows, _, visible, lifecycle = _candidate_rows(facts)
    target = rows[selected_target]
    target_valid = (
        (selected_target > 0) & visible[selected_target] & lifecycle[selected_target]
    )
    distance = _distances(endpoints, _center(target))
    focal_class = facts.focal_features[AGENT_FEATURE_CLASS_ID].astype(_I32)
    band = jnp.asarray(profile.basic_range_bands, dtype=_F32)[_class_index(focal_class)]
    radius = facts.focal_features[AGENT_FEATURE_BASIC_INTERACTION_RADIUS]

    def band_quality(value: Array) -> Array:
        """Score one endpoint distance against the focal class band."""
        return _band_quality(value, radius, band, profile)

    quality = jax.vmap(band_quality)(distance)
    return jnp.where(
        target_valid & jnp.all(jnp.isfinite(_center(target))), quality, _F32(0.0)
    )


def _formation_quality(
    facts: PolicyFacts,
    endpoints: Array,
    selected_target: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> Array:
    """Score endpoints around a selected ally or identity-safe cohort anchor."""
    ally_valid = (
        facts.ally_visible
        & facts.own_active
        & facts.own_alive
        & jnp.all(jnp.isfinite(_center(facts.ally_features)), axis=1)
    )
    selected_ally = (selected_target >= 1) & (selected_target <= 5)
    selected_index = jnp.clip(selected_target - 1, 0, 4)
    selected_valid = selected_ally & ally_valid[selected_index]
    selected_center = _center(facts.ally_features[selected_index])

    count = jnp.sum(ally_valid, dtype=_I32) - _I32(1)
    nonfocal_sum = jnp.sum(
        jnp.where(ally_valid[:, None], _center(facts.ally_features), 0), axis=0
    ) - _center(facts.focal_features)
    cohort_valid = count > 0
    cohort_center = nonfocal_sum / jnp.maximum(count, _I32(1)).astype(_F32)
    anchor = jnp.where(selected_valid, selected_center, cohort_center)
    anchor_valid = selected_valid | cohort_valid

    focal_class = facts.focal_features[AGENT_FEATURE_CLASS_ID].astype(_I32)
    support_scale = jnp.where(
        focal_class == MAGE_CLASS_ID,
        facts.focal_features[
            AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS
        ],
        jnp.where(
            focal_class == WARRIOR_CLASS_ID,
            facts.focal_features[
                AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS
            ],
            facts.focal_features[AGENT_FEATURE_BASIC_INTERACTION_RADIUS],
        ),
    )
    distance = _distances(endpoints, anchor)
    band = jnp.asarray(profile.formation_band, dtype=_F32)

    def band_quality(value: Array) -> Array:
        """Score one endpoint distance against the formation band."""
        return _band_quality(value, support_scale, band, profile)

    quality = jax.vmap(band_quality)(distance)
    return jnp.where(anchor_valid, quality, _F32(0.0))


def _focal_endpoint_risk(
    facts: PolicyFacts,
    endpoint: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array]:
    """Return base mechanics risk and Countered-By-refined residual risk.

    Only mechanics risk may reject an endpoint. Residual risk is a weighted
    score cost.
    """
    base, soft, valid, _, _ = _threat_tensor(facts, endpoint, profile)
    hard = _clip01(
        _F32(1.0) - jnp.prod(_F32(1.0) - jnp.where(valid[:, 0], base[:, 0], 0))
    )
    residual = _clip01(
        _F32(1.0) - jnp.prod(_F32(1.0) - jnp.where(valid[:, 0], soft[:, 0], 0))
    )
    return hard, residual


def _recovery_quality(
    facts: PolicyFacts,
    mechanics_risk: Array,
    selected_target: Array,
    selected_ultimate: Array,
    raw_damage: Array,
    raw_healing: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> Array:
    """Score selected-pair current recovery opportunity for each endpoint risk."""
    rows, _, _, _ = _candidate_rows(facts)
    recipient = rows[selected_target]
    recipient_countdown = recipient[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT]
    chosen_damage = raw_damage[selected_target, selected_ultimate]
    chosen_healing = raw_healing[selected_target, selected_ultimate]
    damage_valid = _finite_nonnegative(chosen_damage)
    healing_valid = _finite_nonnegative(chosen_healing)
    recipient_countdown_valid = (selected_target == 0) | _finite_nonnegative(
        recipient_countdown
    )
    positive_damage = damage_valid & (chosen_damage > 0)
    positive_healing = healing_valid & (chosen_healing > 0)
    participates = positive_damage | (
        positive_healing & recipient_countdown_valid & (recipient_countdown > 0)
    )
    participates_valid = damage_valid & jnp.where(
        positive_damage,
        True,
        healing_valid & (~positive_healing | recipient_countdown_valid),
    )
    countdown = facts.focal_features[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT]
    poison_duration = facts.focal_features[
        AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION
    ]
    poison_value = facts.focal_features[AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER]
    antiheal = jnp.where(poison_duration > 0, poison_value, _F32(1.0))
    opportunity = jnp.where(
        countdown == 0,
        jnp.where(participates, _F32(0.0), antiheal),
        jnp.where(
            countdown == 1,
            _F32(profile.recovery_at_one),
            jnp.where(
                countdown > 1,
                _F32(profile.recovery_after_one),
                _F32(0.0),
            ),
        ),
    )
    antiheal_valid = (
        jnp.isfinite(poison_duration)
        & (poison_duration >= 0)
        & ((poison_duration == 0) | (jnp.isfinite(poison_value) & (poison_value >= 0)))
    )
    current_opportunity_valid = participates_valid & (participates | antiheal_valid)
    valid = _finite_nonnegative(countdown) & jnp.where(
        countdown == 0,
        current_opportunity_valid,
        True,
    )
    quality = _clip01(
        _health_need(facts.focal_features) * opportunity * (_F32(1.0) - mechanics_risk)
    )
    return jnp.where(valid, quality, _F32(0.0))


def _friendly_crowding(
    facts: PolicyFacts,
    endpoint: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> Array:
    """Measure visible nonfocal ally pressure around one intended endpoint."""
    focal_radius = facts.focal_features[AGENT_FEATURE_RADIUS]
    ally_radius = facts.ally_features[:, AGENT_FEATURE_RADIUS]
    radius_sum = focal_radius + ally_radius
    distance = _distances(_center(facts.ally_features), endpoint)
    valid = (
        facts.ally_visible
        & facts.own_active
        & facts.own_alive
        & jnp.all(jnp.isfinite(_center(facts.ally_features)), axis=1)
        & jnp.isfinite(focal_radius)
        & jnp.isfinite(ally_radius)
        & (focal_radius > 0)
        & (ally_radius > 0)
        & (radius_sum > 0)
    )
    pressure = _clip01(
        _safe_ratio(
            _F32(profile.crowding_distance_factor) * radius_sum - distance,
            radius_sum,
            valid,
        )
    )
    self_radius_sum = _F32(2.0) * focal_radius
    self_distance = _distance(_center(facts.focal_features), endpoint)
    self_valid = jnp.isfinite(self_radius_sum) & (self_radius_sum > 0)
    self_pressure = _clip01(
        _safe_ratio(
            _F32(profile.crowding_distance_factor) * self_radius_sum - self_distance,
            self_radius_sum,
            self_valid,
        )
    )
    denominator = jnp.maximum(facts.own_configured_count - 1, 1).astype(_F32)
    crowding = _clip01((jnp.sum(pressure) - self_pressure) / denominator)
    overall_valid = (
        (facts.own_configured_count > 1)
        & jnp.isfinite(self_radius_sum)
        & (self_radius_sum > 0)
    )
    return jnp.where(overall_valid, crowding, _F32(0.0))


def _obstruction(
    facts: PolicyFacts,
    endpoint: Array,
    selected_target: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> Array:
    """Score preference-only obstruction evidence, never movement legality."""
    radius = facts.focal_features[AGENT_FEATURE_RADIUS]
    center = _center(facts.focal_features)
    endpoint_finite = jnp.all(jnp.isfinite(endpoint))
    center_finite = jnp.all(jnp.isfinite(center))
    endpoint_predicate_valid = (
        endpoint_finite
        & jnp.isfinite(radius)
        & (radius > 0)
        & jnp.isfinite(facts.map_width)
        & jnp.isfinite(facts.map_height)
        & (facts.map_width > 0)
        & (facts.map_height > 0)
    )
    outside = (
        (endpoint[0] < radius)
        | (endpoint[0] > facts.map_width - radius)
        | (endpoint[1] < radius)
        | (endpoint[1] > facts.map_height - radius)
    )
    overlaps = jnp.any(
        jax.vmap(disc_overlaps_obstacle, in_axes=(None, None, 0))(
            endpoint, radius, facts.obstacles
        )
    )
    endpoint_blocked = endpoint_predicate_valid & (outside | overlaps)
    segment_blocked = (
        endpoint_finite
        & center_finite
        & ~_has_clear_line_of_sight(center, endpoint, facts.obstacles)
    )

    rows, _, visible, lifecycle = _candidate_rows(facts)
    target = rows[selected_target]
    target_valid = (
        (selected_target > 0)
        & visible[selected_target]
        & lifecycle[selected_target]
        & endpoint_finite
        & jnp.all(jnp.isfinite(_center(target)))
    )
    target_blocked = target_valid & ~_has_clear_line_of_sight(
        endpoint, _center(target), facts.obstacles
    )
    weights = jnp.asarray(profile.obstruction_weights, dtype=_F32)
    return jnp.maximum(
        weights[0] * endpoint_blocked.astype(_F32),
        jnp.maximum(
            weights[1] * segment_blocked.astype(_F32),
            weights[2] * target_blocked.astype(_F32),
        ),
    )


def _mage_ally_damage_value(
    facts: PolicyFacts,
    ally_index: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> Array:
    """Return one ally's best current Basic-damage impact on visible enemies."""
    ally = facts.ally_features[ally_index]
    raw_damage = ally[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE]
    radius = ally[AGENT_FEATURE_BASIC_INTERACTION_RADIUS]
    source_stuns = ally[
        AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION : (
            AGENT_FEATURE_STUN_ROGUE_POISON_DURATION + 1
        )
    ]
    source_stun_valid = jnp.all(
        jnp.isfinite(source_stuns) & (source_stuns >= _F32(0.0))
    )
    source_stunned = jnp.any(source_stuns > 0)
    source_valid = (
        facts.ally_visible[ally_index]
        & facts.own_active[ally_index]
        & facts.own_alive[ally_index]
        & (facts.own_spawn_shields[ally_index] == 0)
        & source_stun_valid
        & ~source_stunned
        & _finite_nonnegative(raw_damage)
        & (raw_damage > 0)
        & jnp.isfinite(radius)
        & (radius > 0)
    )

    def target_impact(enemy: Array, valid: Array, shield: Array) -> Array:
        """Score the ally's current Basic impact on one enemy row."""
        distance = _distance(_center(ally), _center(enemy))
        geometry = (
            jnp.all(jnp.isfinite(_center(ally)))
            & jnp.all(jnp.isfinite(_center(enemy)))
            & (distance <= radius)
        )
        hp = enemy[AGENT_FEATURE_CURRENT_HEALTH]
        max_hp = enemy[AGENT_FEATURE_MAX_HEALTH]
        health_valid = (
            jnp.isfinite(hp) & jnp.isfinite(max_hp) & (hp >= 0) & (max_hp > 0)
        )
        payload = _post_damage(ally, enemy, raw_damage)
        realized = jnp.minimum(payload, jnp.maximum(hp, 0))
        coverage = _clip01(_safe_ratio(realized, hp, health_valid & (hp > 0)))
        potency = _clip01(
            _safe_ratio(
                realized,
                _F32(profile.damage_potency_fraction) * max_hp,
                health_valid,
            )
        )
        impact = _clip01(
            _F32(profile.impact_mix[0]) * potency
            + _F32(profile.impact_mix[1]) * coverage
        )
        return jnp.where(
            source_valid & valid & (shield == 0) & geometry & health_valid,
            impact,
            0,
        )

    values = jax.vmap(target_impact)(
        facts.enemy_features,
        facts.enemy_visible & facts.enemy_active & facts.enemy_alive,
        facts.enemy_spawn_shields,
    )
    return jnp.max(values)


def _marginal_aura_quality(
    facts: PolicyFacts,
    endpoint: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> Array:
    """Score weak current-input potential for marginal aura coverage."""
    focal_class = facts.focal_features[AGENT_FEATURE_CLASS_ID].astype(_I32)
    focal_lifecycle = (facts.focal_features[AGENT_FEATURE_ACTIVE] > 0) & (
        facts.focal_features[AGENT_FEATURE_ALIVE] > 0
    )
    focal_eligible = focal_lifecycle & (
        facts.focal_shield_state == FOCAL_SHIELD_KNOWN_FALSE
    )
    row_valid = (
        facts.ally_visible
        & facts.own_active
        & facts.own_alive
        & (facts.own_spawn_shields == 0)
        & jnp.all(jnp.isfinite(_center(facts.ally_features)), axis=1)
    )

    focal_aura_radius = jnp.where(
        focal_class == MAGE_CLASS_ID,
        facts.focal_features[
            AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS
        ],
        facts.focal_features[
            AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS
        ],
    )
    emitter_in_range = (
        row_valid
        & jnp.isfinite(focal_aura_radius)
        & (focal_aura_radius > 0)
        & (_distances(_center(facts.ally_features), endpoint) <= focal_aura_radius)
    )

    def ally_damage_value(index: Array) -> Array:
        """Score one ally's current damage value for Mage aura coverage."""
        return _mage_ally_damage_value(facts, index, profile)

    damage_value = jax.vmap(ally_damage_value)(jnp.arange(5, dtype=_I32))
    emitter_q = jnp.where(
        focal_class == MAGE_CLASS_ID,
        emitter_in_range * damage_value,
        emitter_in_range.astype(_F32),
    )
    emitter_denominator = jnp.maximum(facts.own_configured_count - 1, 1).astype(_F32)
    emitter_quality = _clip01(
        (jnp.sum(emitter_q) - jnp.max(emitter_q)) / emitter_denominator
    )

    def cover_for_class(class_id: int, radius_feature: int) -> Array:
        """Measure capped coverage by one visible allied aura class."""
        radius = facts.ally_features[:, radius_feature]
        covered = (
            row_valid
            & (facts.own_class_ids == class_id)
            & jnp.isfinite(radius)
            & (radius > 0)
            & (_distances(_center(facts.ally_features), endpoint) <= radius)
        )
        return _clip01(
            jnp.minimum(
                jnp.sum(covered, dtype=_F32),
                _F32(profile.aura_cover_cap),
            )
            / _F32(profile.aura_cover_cap)
        )

    mage_cover = cover_for_class(
        MAGE_CLASS_ID, AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS
    )
    warrior_cover = cover_for_class(
        WARRIOR_CLASS_ID,
        AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS,
    )
    beneficiary_quality = jnp.maximum(mage_cover, warrior_cover)
    emitter_class = (focal_class == MAGE_CLASS_ID) | (focal_class == WARRIOR_CLASS_ID)
    quality = jnp.where(emitter_class, emitter_quality, beneficiary_quality)
    return jnp.where(focal_eligible, quality, _F32(0.0))


def _movement_workspaces_impl(
    facts: PolicyFacts,
    selected_target: Array,
    selected_ultimate: Array,
    raw_damage: Array,
    raw_healing: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array, Array, Array]:
    """Score intended endpoints without projection or reconstructed legality."""
    p = profile
    focal = facts.focal_features
    speed = focal[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
    endpoint_inputs_valid = (
        jnp.all(jnp.isfinite(_center(focal))) & jnp.isfinite(speed) & (speed >= 0)
    )
    endpoints = _center(focal) + speed * UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY
    endpoint_valid = endpoint_inputs_valid & jnp.all(jnp.isfinite(endpoints), axis=1)
    endpoints = jnp.where(endpoint_valid[:, None], endpoints, jnp.zeros_like(endpoints))

    range_quality = _movement_range_quality(facts, endpoints, selected_target, profile)
    formation = _formation_quality(facts, endpoints, selected_target, profile)

    def endpoint_risk(endpoint: Array) -> tuple[Array, Array]:
        """Build hard and residual threat for one intended endpoint."""
        return _focal_endpoint_risk(facts, endpoint, profile)

    mechanics_risk, residual_risk = jax.vmap(endpoint_risk)(endpoints)
    range_quality = jnp.where(endpoint_valid, range_quality, 0)
    formation = jnp.where(endpoint_valid, formation, 0)
    mechanics_risk = jnp.where(endpoint_valid, mechanics_risk, 0)
    residual_risk = jnp.where(endpoint_valid, residual_risk, 0)
    recovery = _recovery_quality(
        facts,
        mechanics_risk,
        selected_target,
        selected_ultimate,
        raw_damage,
        raw_healing,
        profile,
    )

    def endpoint_crowding(endpoint: Array) -> Array:
        """Build friendly-crowding cost for one intended endpoint."""
        return _friendly_crowding(facts, endpoint, profile)

    def endpoint_obstruction(endpoint: Array) -> Array:
        """Build obstruction cost for one intended endpoint."""
        return _obstruction(facts, endpoint, selected_target, profile)

    def endpoint_aura(endpoint: Array) -> Array:
        """Build marginal aura value for one intended endpoint."""
        return _marginal_aura_quality(facts, endpoint, profile)

    crowding = jax.vmap(endpoint_crowding)(endpoints)
    obstruction = jax.vmap(endpoint_obstruction)(endpoints)
    aura = jax.vmap(endpoint_aura)(endpoints)
    recovery = jnp.where(endpoint_valid, recovery, 0)
    crowding = jnp.where(endpoint_valid, crowding, 0)
    obstruction = jnp.where(endpoint_valid, obstruction, 0)
    aura = jnp.where(endpoint_valid, aura, 0)

    configured_valid = (facts.own_configured_count > 0) & (
        facts.enemy_configured_count > 0
    )
    own_alive = jnp.sum(facts.own_active & facts.own_alive, dtype=_F32)
    enemy_alive = jnp.sum(facts.enemy_active & facts.enemy_alive, dtype=_F32)
    own_fraction = _safe_ratio(
        own_alive,
        facts.own_configured_count.astype(_F32),
        facts.own_configured_count > 0,
    )
    enemy_fraction = _safe_ratio(
        enemy_alive,
        facts.enemy_configured_count.astype(_F32),
        facts.enemy_configured_count > 0,
    )
    alive_gap = jnp.clip(
        own_fraction - enemy_fraction,
        _F32(-1.0),
        _F32(1.0),
    )
    alive_context = jnp.where(
        configured_valid,
        _F32(p.alive_context_weight)
        * alive_gap
        * (range_quality - jnp.maximum(formation, recovery)),
        _F32(0.0),
    )

    focal_class = focal[AGENT_FEATURE_CLASS_ID].astype(_I32)
    weights = jnp.asarray(p.movement_weights, dtype=_F32)[_class_index(focal_class)]
    weights = jnp.where(
        (focal_class >= MAGE_CLASS_ID) & (focal_class <= PRIEST_CLASS_ID),
        weights,
        jnp.zeros((3,), dtype=_F32),
    )
    # The nine trace slots are the only movement-score owners. Obstruction and
    # residual risk are preferences; neither reconstructs movement legality.
    components = jnp.stack(
        (
            weights[0] * range_quality,
            weights[1] * formation,
            weights[2] * recovery,
            _F32(p.aura_weight) * aura,
            -_F32(p.residual_risk_weight) * residual_risk,
            -_F32(p.crowding_weight) * crowding,
            -_F32(p.obstruction_weight) * obstruction,
            alive_context,
            jnp.full((9,), _F32(p.tdm_task_value)),
        ),
        axis=-1,
    ).astype(_F32)
    utility = _sum_components(components)
    return components, utility, mechanics_risk, endpoints


def _movement_workspaces(
    facts: PolicyFacts,
    selected_target: Array,
    selected_ultimate: Array,
    raw_damage: Array,
    raw_healing: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[Array, Array, Array, Array]:
    """Build movement workspaces without allowing global x64 widening."""
    with jax.enable_x64(False):
        return _movement_workspaces_impl(
            facts,
            selected_target,
            selected_ultimate,
            raw_damage,
            raw_healing,
            profile,
        )


# Exact selection and trace realization ---


def _complete_decision(
    facts: PolicyFacts,
    action_mask: ActionMask,
    combat_key: Array,
    movement_key: Array,
    pair_mask: Array,
    combat_components: Array,
    raw_damage: Array,
    raw_healing: Array,
    trap_suppressed: Array,
    combat_support: Array,
    combat_basis: Array,
    branch_reason: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[ActorAction, ScriptedTrace]:
    """Sample combat peers, then pair-conditioned movement peers, and trace both."""
    focal_class = facts.focal_features[AGENT_FEATURE_CLASS_ID].astype(_I32)
    # Nonempty support is an upstream mask contract. Zero/-inf logits make this
    # single draw uniform over literal float32 peers without invented fallback.
    combat_logits = jnp.where(combat_support, _F32(0.0), _NEG_INF)
    flat_combat = jax.random.categorical(combat_key, jnp.ravel(combat_logits))
    sampled_target, sampled_ultimate = jnp.unravel_index(
        flat_combat, (NUM_TARGET_ACTIONS, NUM_ULTIMATE_ACTIONS)
    )
    sampled_target = sampled_target.astype(_I32)
    sampled_ultimate = sampled_ultimate.astype(_I32)

    # Poison substitutes after the combat draw, consumes no key, and preserves
    # the sampled target selected by ordinary Rogue Basic utility.
    poison_substitution = (
        (focal_class == ROGUE_CLASS_ID)
        & (sampled_target > 0)
        & (sampled_ultimate == 0)
        & pair_mask[sampled_target, 1]
        & ~trap_suppressed[sampled_target, 1]
    )
    selected_target = sampled_target
    selected_ultimate = jnp.where(poison_substitution, _I32(1), sampled_ultimate)
    combat_reason = jnp.where(
        poison_substitution,
        _I32(ROGUE_POISON_SUBSTITUTION),
        jnp.where(
            (branch_reason == DIRECT_SCORE)
            & (sampled_target == 0)
            & (sampled_ultimate == 0),
            _I32(EFFECT_INERT_NOOP),
            branch_reason,
        ),
    )
    # Trigger traces are reason-defined. Some hard reducers use a zero basis;
    # Hunter emergency/no-Priest branches retain only their reducer basis.
    zero_component_reason = (
        (combat_reason == MAGE_BURST_TRIGGER)
        | (combat_reason == HUNTER_TRAP_EMERGENCY)
        | (combat_reason == HUNTER_TRAP_NO_PRIEST_CROWD)
        | (combat_reason == PRIEST_HOLY_WORD_TRIGGER)
    )
    selected_combat_components = jnp.where(
        zero_component_reason,
        jnp.zeros((8,), dtype=_F32),
        combat_components[sampled_target, sampled_ultimate],
    ).astype(_F32)
    selected_combat_basis = jnp.where(
        zero_component_reason,
        _F32(0.0),
        combat_basis[sampled_target, sampled_ultimate],
    ).astype(_F32)
    selected_combat_basis = jnp.where(
        combat_reason == HUNTER_TRAP_EMERGENCY,
        combat_basis[sampled_target, sampled_ultimate],
        selected_combat_basis,
    ).astype(_F32)
    selected_combat_basis = jnp.where(
        combat_reason == HUNTER_TRAP_NO_PRIEST_CROWD,
        combat_basis[sampled_target, sampled_ultimate],
        selected_combat_basis,
    ).astype(_F32)
    combat_peer_count = jnp.sum(combat_support, dtype=_I32)

    movement_components, movement_utility, mechanics_risk, _ = _movement_workspaces(
        facts,
        selected_target,
        selected_ultimate,
        raw_damage,
        raw_healing,
        profile,
    )
    move_mask = action_mask.move_mask.astype(jnp.bool_)
    # The exact move mask owns legality. Risk rejects only values strictly over
    # the ceiling; an all-rejected set retains every exact minimum-risk peer.
    rejected = move_mask & (mechanics_risk > _F32(profile.risk_ceiling))
    post_risk = move_mask & ~rejected
    minimum_risk_fallback = ~jnp.any(post_risk)
    minimum_risk = jnp.min(jnp.where(move_mask, mechanics_risk, _F32(jnp.inf)))
    minimum_support = move_mask & (mechanics_risk == minimum_risk)
    risk_support = jnp.where(minimum_risk_fallback, minimum_support, post_risk)

    # The inclusive Stay deadband replaces the winning non-Stay support and
    # carries no state across decisions.
    nonstay = risk_support & (jnp.arange(9, dtype=_I32) != MOVE_STAY)
    max_nonstay = jnp.max(jnp.where(nonstay, movement_utility, _NEG_INF))
    stay_supported = risk_support[MOVE_STAY]
    deadband_condition = (
        stay_supported
        & jnp.any(nonstay)
        & (max_nonstay - movement_utility[MOVE_STAY] <= _F32(profile.stay_deadband))
    )
    maximum_movement = jnp.max(jnp.where(risk_support, movement_utility, _NEG_INF))
    direct_movement_support = risk_support & (movement_utility == maximum_movement)
    stay_support = (jnp.arange(9, dtype=_I32) == MOVE_STAY) & move_mask
    deadband_support = jnp.where(
        deadband_condition, stay_support, direct_movement_support
    )
    deadband_changed = deadband_condition & jnp.any(direct_movement_support & nonstay)

    # Charge is the only combat result that forces canonical Stay.
    charge_selected = combat_reason == WARRIOR_CHARGE_TRIGGER
    movement_support = jnp.where(charge_selected, stay_support, deadband_support)
    movement_reason = jnp.where(
        charge_selected,
        _I32(CHARGE_TO_STAY),
        jnp.where(
            deadband_changed,
            _I32(STAY_DEADBAND),
            jnp.where(
                minimum_risk_fallback,
                _I32(MIN_RISK_FALLBACK),
                _I32(MOVE_DIRECT_SCORE),
            ),
        ),
    )
    movement_logits = jnp.where(movement_support, _F32(0.0), _NEG_INF)
    selected_movement = jax.random.categorical(movement_key, movement_logits).astype(
        _I32
    )
    # Charge-to-Stay reports zero movement basis/components. Every other reason
    # reports the selected movement candidate's weighted computation.
    selected_movement_components = jnp.where(
        charge_selected,
        jnp.zeros((9,), dtype=_F32),
        movement_components[selected_movement],
    ).astype(_F32)
    selected_movement_basis = jnp.where(
        charge_selected,
        _F32(0.0),
        movement_utility[selected_movement],
    ).astype(_F32)
    movement_peer_count = jnp.sum(movement_support, dtype=_I32)

    trap_guard = jnp.any(pair_mask & trap_suppressed)
    risk_guard = ~charge_selected & jnp.any(rejected)
    fallback_guard = ~charge_selected & minimum_risk_fallback
    fired_guards = jnp.asarray(
        (
            trap_guard,
            combat_reason == MAGE_BURST_TRIGGER,
            combat_reason == WARRIOR_CHARGE_TRIGGER,
            (combat_reason == HUNTER_TRAP_EMERGENCY)
            | (combat_reason == HUNTER_TRAP_PRIEST_CROWD)
            | (combat_reason == HUNTER_TRAP_NO_PRIEST_CROWD),
            poison_substitution,
            combat_reason == PRIEST_HOLY_WORD_TRIGGER,
            charge_selected,
            risk_guard,
            fallback_guard,
            ~charge_selected & deadband_changed,
        ),
        dtype=jnp.bool_,
    )

    action = ActorAction(
        move=selected_movement.astype(_I32),
        select_target=selected_target.astype(_I32),
        use_ultimate=selected_ultimate.astype(_I32),
    )
    trace = ScriptedTrace(
        combat_target=action.select_target,
        combat_use_ultimate=action.use_ultimate,
        movement_action=action.move,
        combat_selection_basis_value=selected_combat_basis,
        movement_selection_basis_value=selected_movement_basis,
        combat_selection_basis_components=selected_combat_components,
        movement_selection_basis_components=selected_movement_components,
        combat_reason_id=combat_reason.astype(_I32),
        movement_reason_id=movement_reason.astype(_I32),
        fired_guards=fired_guards,
        combat_peer_count=combat_peer_count.astype(_I32),
        movement_peer_count=movement_peer_count.astype(_I32),
    )
    return action, trace


def _decide_team_deathmatch(
    facts: PolicyFacts,
    action_mask: ActionMask,
    key: Array,
    profile: TeamDeathmatchProfile = TEAM_DEATHMATCH_PROFILE,
) -> tuple[ActorAction, ScriptedTrace]:
    """Dispatch one action-and-trace decision from same-epoch authorized facts."""
    # Split unconditionally and consume one categorical draw per head, including
    # singleton support. The complete action is precommitted from this snapshot.
    combat_key, movement_key = jax.random.split(key, 2)
    # The environment-provided joint mask is the sole combat-legality authority;
    # policy guards and exact class triggers may only narrow its support.
    pair_mask = action_mask.select_target_use_ultimate_joint_mask.astype(jnp.bool_)
    (
        combat_components,
        combat_utility,
        raw_damage,
        raw_healing,
        post_damage,
        _,
        _,
        trap_suppressed,
        _,
    ) = _combat_workspaces(facts, profile)
    ordinary = _ordinary_combat_support(pair_mask, trap_suppressed)

    def finish(
        branch: tuple[Array, Array, Array, Array],
    ) -> tuple[ActorAction, ScriptedTrace]:
        """Realize one class branch through the common selection path."""
        support, basis, reason, _ = branch
        return _complete_decision(
            facts,
            action_mask,
            combat_key,
            movement_key,
            pair_mask,
            combat_components,
            raw_damage,
            raw_healing,
            trap_suppressed,
            support,
            basis,
            reason,
            profile,
        )

    def neutral_branch(operand: None) -> tuple[ActorAction, ScriptedTrace]:
        """Use ordinary scoring for an unsupported or neutral focal class ID."""
        del operand
        branch = (
            _maximum_combat_support(ordinary, combat_utility),
            combat_utility,
            jnp.asarray(DIRECT_SCORE, dtype=_I32),
            jnp.asarray(False),
        )
        return finish(branch)

    def mage_branch(operand: None) -> tuple[ActorAction, ScriptedTrace]:
        """Build and realize Mage combat support."""
        del operand
        return finish(
            _mage_combat(
                facts,
                pair_mask,
                ordinary,
                combat_utility,
                post_damage,
                profile,
            )
        )

    def warrior_branch(operand: None) -> tuple[ActorAction, ScriptedTrace]:
        """Build and realize Warrior combat support."""
        del operand
        return finish(
            _warrior_combat(
                facts,
                pair_mask,
                ordinary,
                combat_utility,
                trap_suppressed,
                profile,
            )
        )

    def hunter_branch(operand: None) -> tuple[ActorAction, ScriptedTrace]:
        """Build and realize Hunter combat support."""
        del operand
        return finish(
            _hunter_combat(
                facts,
                pair_mask,
                ordinary,
                combat_utility,
                trap_suppressed,
                profile,
            )
        )

    def rogue_branch(operand: None) -> tuple[ActorAction, ScriptedTrace]:
        """Build and realize Rogue combat support."""
        del operand
        return finish(_rogue_combat(pair_mask, ordinary, combat_utility))

    def priest_branch(operand: None) -> tuple[ActorAction, ScriptedTrace]:
        """Build and realize Priest combat support."""
        del operand
        return finish(
            _priest_combat(facts, pair_mask, ordinary, combat_utility, profile)
        )

    focal_class = facts.focal_features[AGENT_FEATURE_CLASS_ID].astype(_I32)
    dispatch_index = jnp.where(
        (focal_class >= MAGE_CLASS_ID) & (focal_class <= PRIEST_CLASS_ID),
        focal_class,
        _I32(0),
    ).astype(_I32)
    return cast(
        tuple[ActorAction, ScriptedTrace],
        jax.lax.switch(
            dispatch_index,
            (
                neutral_branch,
                mage_branch,
                warrior_branch,
                hunter_branch,
                rogue_branch,
                priest_branch,
            ),
            None,
        ),
    )


# Public task-policy API ---


def decide_team_deathmatch_with_profile(
    facts: PolicyFacts,
    action_mask: ActionMask,
    key: Array,
    profile: TeamDeathmatchProfile,
) -> tuple[ActorAction, ScriptedTrace]:
    """Run an explicit profile under x64-disabled float32 execution."""
    with jax.enable_x64(False):
        return _decide_team_deathmatch(facts, action_mask, key, profile)


def decide_team_deathmatch(
    facts: PolicyFacts,
    action_mask: ActionMask,
    key: Array,
) -> tuple[ActorAction, ScriptedTrace]:
    """Choose one action and trace with the canonical immutable TDM profile."""
    return decide_team_deathmatch_with_profile(
        facts,
        action_mask,
        key,
        TEAM_DEATHMATCH_PROFILE,
    )


__all__ = (
    "FOCAL_SHIELD_KNOWN_FALSE",
    "FOCAL_SHIELD_KNOWN_TRUE",
    "FOCAL_SHIELD_UNKNOWN",
    "NUMERIC_PROFILE_ID",
    "POLICY_ID",
    "POLICY_SEMANTIC_VERSION",
    "SEMANTIC_PROFILE_ID",
    "TASK_HEAD_VERSION",
    "TEAM_DEATHMATCH_PROFILE",
    "TRACE_ONTOLOGY_VERSION",
    "PolicyFacts",
    "ScriptedTrace",
    "TeamDeathmatchProfile",
    "decide_team_deathmatch",
)
