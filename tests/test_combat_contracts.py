"""Combat contract tests for Milestone 5 class setup."""

import subprocess
import sys
from collections.abc import Callable
from importlib import import_module
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

import marl_battlegrounds.core.combat as combat
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_BASE_MOVEMENT_SPEED,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_OBSERVATION_RADIUS,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_TEAM_ID,
    AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    CLASS_NEUTRAL,
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    NEUTRAL_CLASS_ID,
    NUM_CLASSES,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    SELF_FEATURES,
    SLOW_CHANNEL_HUNTER_BASIC,
    SLOW_CHANNEL_ROGUE_POISON,
    SLOW_CHANNEL_WARRIOR_CHARGE,
    STUN_CHANNEL_HUNTER_TRAP,
    UNIT_FEATURES,
    WARRIOR_CLASS_ID,
    EnvConfig,
    EnvState,
    Observation,
)

_CONFIG_DEFAULT_5_CLASS_MIRROR: Array = jnp.asarray(
    [
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
    ],
    dtype=jnp.int32,
)

_FORBIDDEN_COMBAT_IMPORT_MODULES: tuple[str, ...] = (
    "marl_battlegrounds.core.config",
    "marl_battlegrounds.core.env",
    "marl_battlegrounds.core.geometry",
    "marl_battlegrounds.rendering",
    "marl_battlegrounds.rendering.scene_geometry",
)

_FLOAT_CLASS_CATALOG_NAMES: tuple[str, ...] = (
    "MAX_HEALTH_BY_CLASS",
    "BASE_MOVEMENT_SPEED_BY_CLASS",
    "BODY_RADIUS_BY_CLASS",
    "BASIC_INTERACTION_RADIUS_BY_CLASS",
    "BASIC_DAMAGE_BY_CLASS",
    "BASIC_HEALING_BY_CLASS",
    "ULTIMATE_INTERACTION_RADIUS_BY_CLASS",
    "ULTIMATE_DAMAGE_BY_CLASS",
    "ULTIMATE_HEALING_BY_CLASS",
    "OBSERVATION_RADIUS_BY_CLASS",
)

_INTEGER_CLASS_CATALOG_NAMES: tuple[str, ...] = (
    "ULTIMATE_COOLDOWN_BY_CLASS",
    "ULTIMATE_TARGET_MODE_BY_CLASS",
)

_CLASS_CATALOG_NAMES: tuple[str, ...] = (
    *_FLOAT_CLASS_CATALOG_NAMES,
    *_INTEGER_CLASS_CATALOG_NAMES,
)

_CLASS_CATALOG_HELPER_NAMES: tuple[tuple[str, str], ...] = (
    ("MAX_HEALTH_BY_CLASS", "get_max_health_by_class_ids"),
    ("BASE_MOVEMENT_SPEED_BY_CLASS", "get_base_movement_speed_by_class_ids"),
    ("BODY_RADIUS_BY_CLASS", "get_body_radius_by_class_ids"),
    (
        "BASIC_INTERACTION_RADIUS_BY_CLASS",
        "get_basic_interaction_radius_by_class_ids",
    ),
    ("BASIC_DAMAGE_BY_CLASS", "get_basic_damage_by_class_ids"),
    ("BASIC_HEALING_BY_CLASS", "get_basic_healing_by_class_ids"),
    (
        "ULTIMATE_INTERACTION_RADIUS_BY_CLASS",
        "get_ultimate_interaction_radius_by_class_ids",
    ),
    ("ULTIMATE_COOLDOWN_BY_CLASS", "get_ultimate_cooldown_by_class_ids"),
    ("ULTIMATE_DAMAGE_BY_CLASS", "get_ultimate_damage_by_class_ids"),
    ("ULTIMATE_HEALING_BY_CLASS", "get_ultimate_healing_by_class_ids"),
    ("OBSERVATION_RADIUS_BY_CLASS", "get_observation_radius_by_class_ids"),
    ("ULTIMATE_TARGET_MODE_BY_CLASS", "get_ultimate_target_mode_by_class_ids"),
)

_CatalogHelper = Callable[[int | Array], Array]

# Test helpers ---


def _empty_obstacles() -> Array:
    """Return a zero-filled obstacle feature table."""
    return jnp.zeros(shape=(MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _config(
    team_size: int = 3, class_ids: Array = _CONFIG_DEFAULT_5_CLASS_MIRROR
) -> EnvConfig:
    """Return a deterministic combat-contract test config."""
    profile = resolve_agent_profile(
        class_ids, jnp.asarray((team_size, team_size), dtype=jnp.int32)
    )
    positions = jnp.asarray(
        (
            (0.5, 0.5),
            (2.5, 0.5),
            (4.5, 0.5),
            (6.5, 0.5),
            (8.5, 0.5),
            (0.5, 7.5),
            (2.5, 7.5),
            (4.5, 7.5),
            (6.5, 7.5),
            (8.5, 7.5),
        ),
        dtype=jnp.float32,
    )
    return EnvConfig(
        max_steps=1000,
        map_width=20.0,
        map_height=12.0,
        obstacles=_empty_obstacles(),
        agent_profile=profile,
        ordinary_movement_distance_scale=1.0,
        team_spawn_pad_positions=positions.reshape(
            (2, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS)
        ),
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
    )


def _canonical_class_ids() -> tuple[int, ...]:
    """Return neutral plus M5 canonical class IDs in catalog row order."""
    return (
        CLASS_NEUTRAL,
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
    )


def _catalog_array(name: str) -> Array:
    """Return a named class catalog as a typed JAX array."""
    return cast(Array, getattr(combat, name))


def _catalog_helper(name: str) -> _CatalogHelper:
    """Return a named class catalog lookup helper."""
    return cast(_CatalogHelper, getattr(combat, name))


def test_basic_interaction_radii_match_the_approved_combat_catalog() -> None:
    """The approved class-specific Basic ranges stay exact."""
    expected = jnp.asarray(
        (0.0, 3.0, 1.5, 3.5, 1.5, 3.0),
        dtype=jnp.float32,
    )

    _assert_array_equal(
        combat.BASIC_INTERACTION_RADIUS_BY_CLASS,
        expected,
        "BASIC_INTERACTION_RADIUS_BY_CLASS",
    )


def _expected_active_mask(team_size: int) -> Array:
    """Return the fixed-slot active mask for a symmetric two-team task."""
    indices = jnp.arange(MAX_AGENT_SLOTS)

    team_0_active = indices < team_size
    team_1_active = (indices >= MAX_AGENTS_PER_TEAM) & (
        indices < MAX_AGENTS_PER_TEAM + team_size
    )

    return jnp.logical_or(team_0_active, team_1_active)


def _expected_resolved_class_ids(class_ids: Array, team_size: int) -> Array:
    """Return class IDs after reset neutralizes inactive padded slots."""
    active_mask = _expected_active_mask(team_size)
    return jnp.where(active_mask, class_ids, NEUTRAL_CLASS_ID).astype(jnp.int32)


def _assert_array_equal(actual: Array, expected: Array, name: str) -> None:
    """Assert exact JAX array equality with a readable failure name."""
    assert actual.shape == expected.shape, name
    assert bool(jnp.array_equal(actual, expected)), name


def _assert_float_array_close(actual: Array, expected: Array, name: str) -> None:
    """Assert floating-point JAX array equality with a readable failure name."""
    assert actual.shape == expected.shape, name
    assert bool(jnp.allclose(actual, expected)), name


def _assert_reset_combat_state_is_inert(state: EnvState) -> None:
    """Assert dynamic combat effect fields start inert at reset."""
    _assert_array_equal(
        state.ultimate_cooldowns,
        jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        "ultimate_cooldowns",
    )
    _assert_array_equal(
        state.slow_durations,
        jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32),
        "slow_durations",
    )
    _assert_array_equal(
        state.stun_durations,
        jnp.zeros((MAX_AGENT_SLOTS, NUM_STUN_CHANNELS), dtype=jnp.int32),
        "stun_durations",
    )
    _assert_array_equal(
        state.rogue_poison_anti_heal_durations,
        jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        "anti_heal_durations",
    )
    _assert_array_equal(
        state.mage_burst_damage_amplification_durations,
        jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        "damage_amplification_durations",
    )
    _assert_array_equal(
        state.priest_blessing_of_freedom_slow_floor_durations,
        jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        "blessing_of_freedom_durations",
    )


def _assert_self_features_project_state(
    obs: Observation, state: EnvState, config: EnvConfig
) -> None:
    """Assert self rows project dynamic state and the resolved profile."""
    profile = config.agent_profile
    assert obs.self_features.shape == (MAX_AGENT_SLOTS, SELF_FEATURES)

    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_X : AGENT_FEATURE_Y + 1],
        state.agent_positions,
        "obs.self_features.positions",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_RADIUS],
        profile.agent_radii,
        "obs.self_features.radius",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_TEAM_ID],
        profile.team_ids.astype(jnp.float32),
        "obs.self_features.team_id",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_ACTIVE],
        profile.active_mask.astype(jnp.float32),
        "obs.self_features.active",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_ALIVE],
        state.alive_mask.astype(jnp.float32),
        "obs.self_features.alive",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_CLASS_ID],
        profile.class_ids.astype(jnp.float32),
        "obs.self_features.class_id",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_BASE_MOVEMENT_SPEED],
        profile.base_movement_speeds,
        "obs.self_features.movement_speed",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_OBSERVATION_RADIUS],
        profile.observation_radii,
        "obs.self_features.observation_radius",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_BASIC_INTERACTION_RADIUS],
        profile.basic_interaction_radii,
        "obs.self_features.basic_interaction_radius",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS],
        profile.ultimate_interaction_radii,
        "obs.self_features.ultimate_interaction_radius",
    )


# Tests ---


def test_class_id_constants_are_exact_and_contiguous() -> None:
    assert CLASS_NEUTRAL == 0
    assert NEUTRAL_CLASS_ID == CLASS_NEUTRAL
    assert MAGE_CLASS_ID == 1
    assert WARRIOR_CLASS_ID == 2
    assert HUNTER_CLASS_ID == 3
    assert ROGUE_CLASS_ID == 4
    assert PRIEST_CLASS_ID == 5

    assert _canonical_class_ids() == tuple(range(NUM_CLASSES))


def test_num_classes_matches_neutral_and_canonical_class_rows() -> None:
    class_ids = _canonical_class_ids()

    assert len(class_ids) == NUM_CLASSES
    assert len(set(class_ids)) == NUM_CLASSES
    assert max(class_ids) == NUM_CLASSES - 1


def test_combat_module_imports() -> None:
    combat_module = import_module("marl_battlegrounds.core.combat")

    assert combat_module.__name__ == "marl_battlegrounds.core.combat"


def test_combat_module_import_does_not_pull_transition_or_rendering_modules() -> None:
    script = f"""
import sys

import marl_battlegrounds.core.combat  # noqa: F401

for module_name in {_FORBIDDEN_COMBAT_IMPORT_MODULES!r}:
    if module_name in sys.modules:
        raise SystemExit("unexpected import: " + module_name)
"""

    result = subprocess.run(
        (sys.executable, "-c", script),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_class_catalogs_are_fixed_shape_and_dtype_explicit() -> None:
    for name in _FLOAT_CLASS_CATALOG_NAMES:
        catalog = _catalog_array(name)

        assert catalog.shape == (NUM_CLASSES,)
        assert catalog.dtype == jnp.float32

    for name in _INTEGER_CLASS_CATALOG_NAMES:
        catalog = _catalog_array(name)

        assert catalog.shape == (NUM_CLASSES,)
        assert catalog.dtype == jnp.int32


def test_class_catalog_neutral_rows_are_inert() -> None:
    for name in _CLASS_CATALOG_NAMES:
        catalog = _catalog_array(name)

        assert bool(catalog[NEUTRAL_CLASS_ID] == 0), name


def test_all_canonical_class_rows_are_reachable_by_class_id() -> None:
    class_ids = jnp.asarray(_canonical_class_ids(), dtype=jnp.int32)

    for name in _CLASS_CATALOG_NAMES:
        selected_rows = _catalog_array(name)[class_ids]

        assert selected_rows.shape == (NUM_CLASSES,)


def test_basic_damage_and_healing_catalogs_encode_role_families() -> None:
    damage_class_ids = (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
    )

    assert bool(combat.BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID] > 0)
    assert bool(combat.BASIC_DAMAGE_BY_CLASS[PRIEST_CLASS_ID] == 0)

    for class_id in damage_class_ids:
        assert bool(combat.BASIC_DAMAGE_BY_CLASS[class_id] > 0)
        assert bool(combat.BASIC_HEALING_BY_CLASS[class_id] == 0)


def test_rogue_movement_speed_role_distinction_is_represented() -> None:
    non_rogue_class_ids = jnp.asarray(
        (MAGE_CLASS_ID, WARRIOR_CLASS_ID, HUNTER_CLASS_ID, PRIEST_CLASS_ID),
        dtype=jnp.int32,
    )

    assert bool(
        jnp.all(
            combat.BASE_MOVEMENT_SPEED_BY_CLASS[ROGUE_CLASS_ID]
            > combat.BASE_MOVEMENT_SPEED_BY_CLASS[non_rogue_class_ids]
        )
    )


def test_warrior_basic_and_ultimate_interaction_radii_are_separate() -> None:
    assert bool(
        combat.ULTIMATE_INTERACTION_RADIUS_BY_CLASS[WARRIOR_CLASS_ID]
        > combat.BASIC_INTERACTION_RADIUS_BY_CLASS[WARRIOR_CLASS_ID]
    )


def test_mage_burst_is_representable_as_no_target_self_buff() -> None:
    assert bool(combat.ULTIMATE_INTERACTION_RADIUS_BY_CLASS[MAGE_CLASS_ID] == 0)
    assert combat.MAGE_BURST_DAMAGE_DURATION_TICKS > 0
    assert combat.MAGE_BURST_DAMAGE_MULTIPLIER > 1.0


def test_ultimate_target_modes_are_distinct_and_class_aligned() -> None:
    """Prove neutral, no-target, ally, and enemy semantics cannot collapse."""
    modes = (
        combat.NO_ULTIMATE_MODE,
        combat.ONLY_NONE_TARGET_ULTIMATE_MODE,
        combat.ONLY_ALLY_TARGET_ULTIMATE_MODE,
        combat.ONLY_ENEMY_TARGET_ULTIMATE_MODE,
    )
    expected_catalog = jnp.asarray(
        (
            combat.NO_ULTIMATE_MODE,
            combat.ONLY_NONE_TARGET_ULTIMATE_MODE,
            combat.ONLY_ENEMY_TARGET_ULTIMATE_MODE,
            combat.ONLY_ENEMY_TARGET_ULTIMATE_MODE,
            combat.ONLY_ENEMY_TARGET_ULTIMATE_MODE,
            combat.ONLY_ALLY_TARGET_ULTIMATE_MODE,
        ),
        dtype=jnp.int32,
    )

    assert len(set(modes)) == len(modes)
    assert bool(jnp.array_equal(combat.ULTIMATE_TARGET_MODE_BY_CLASS, expected_catalog))


def test_status_mechanic_defaults_are_scalar_parameters() -> None:
    slow_multiplier_names = (
        "HUNTER_BASIC_SLOW_MULTIPLIER",
        "WARRIOR_CHARGE_SLOW_MULTIPLIER",
        "ROGUE_POISON_SLOW_MULTIPLIER",
    )
    duration_names = (
        "HUNTER_BASIC_SLOW_DURATION_TICKS",
        "HUNTER_TRAP_STUN_DURATION_TICKS",
        "WARRIOR_CHARGE_SLOW_DURATION_TICKS",
        "WARRIOR_CHARGE_STUN_DURATION_TICKS",
        "ROGUE_POISON_SLOW_DURATION_TICKS",
        "ROGUE_POISON_STUN_DURATION_TICKS",
        "ROGUE_POISON_ANTI_HEAL_DURATION_TICKS",
    )

    assert 0.0 < combat.GLOBAL_SLOW_FLOOR <= 1.0
    assert 0.0 < combat.ROGUE_POISON_ANTI_HEAL_MULTIPLIER <= 1.0

    for name in slow_multiplier_names:
        value = getattr(combat, name)

        assert isinstance(value, float), name
        assert 0.0 < value <= 1.0, name

    for name in duration_names:
        value = getattr(combat, name)

        assert isinstance(value, int), name
        assert value > 0, name


def test_derive_status_magnitudes_returns_neutral_fixed_shape_contract() -> None:
    slow_multipliers, anti_heal_multipliers, floor_fractions = (
        combat.derive_status_magnitudes(
            jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32),
            jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
            jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        )
    )

    assert slow_multipliers.shape == (MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS)
    assert slow_multipliers.dtype == jnp.float32
    assert bool(jnp.all(slow_multipliers == 1.0))

    assert anti_heal_multipliers.shape == (MAX_AGENT_SLOTS,)
    assert anti_heal_multipliers.dtype == jnp.float32
    assert bool(jnp.all(anti_heal_multipliers == 1.0))

    assert floor_fractions.shape == (MAX_AGENT_SLOTS,)
    assert floor_fractions.dtype == jnp.float32
    assert bool(jnp.all(floor_fractions == 0.0))


@pytest.mark.parametrize(
    ("channel", "expected_multiplier"),
    (
        (SLOW_CHANNEL_WARRIOR_CHARGE, combat.WARRIOR_CHARGE_SLOW_MULTIPLIER),
        (SLOW_CHANNEL_HUNTER_BASIC, combat.HUNTER_BASIC_SLOW_MULTIPLIER),
        (SLOW_CHANNEL_ROGUE_POISON, combat.ROGUE_POISON_SLOW_MULTIPLIER),
    ),
)
def test_derive_status_magnitudes_activates_each_slow_source_independently(
    channel: int,
    expected_multiplier: float,
) -> None:
    active_slot = 3
    slow_durations = (
        jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32)
        .at[active_slot, channel]
        .set(2)
    )

    slow_multipliers, _, _ = combat.derive_status_magnitudes(
        slow_durations,
        jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )
    expected = (
        jnp.ones((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.float32)
        .at[active_slot, channel]
        .set(expected_multiplier)
    )

    assert bool(jnp.array_equal(slow_multipliers, expected))


@pytest.mark.parametrize(
    ("duration_input", "output_index", "active_value", "inactive_value"),
    (
        (
            "rogue",
            1,
            combat.ROGUE_POISON_ANTI_HEAL_MULTIPLIER,
            1.0,
        ),
        (
            "priest",
            2,
            combat.PRIEST_HEAL_SPEED_FLOOR,
            0.0,
        ),
    ),
)
def test_derive_status_magnitudes_activates_each_scalar_source_independently(
    duration_input: str,
    output_index: int,
    active_value: float,
    inactive_value: float,
) -> None:
    active_slot = 6
    durations = {
        name: jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
        for name in ("rogue", "priest")
    }
    durations[duration_input] = durations[duration_input].at[active_slot].set(1)

    derived = combat.derive_status_magnitudes(
        jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32),
        durations["rogue"],
        durations["priest"],
    )
    expected = (
        jnp.full((MAX_AGENT_SLOTS,), inactive_value, dtype=jnp.float32)
        .at[active_slot]
        .set(active_value)
    )

    assert bool(jnp.array_equal(derived[output_index], expected))


def test_derive_status_magnitudes_matches_jit_for_mixed_active_durations() -> None:
    slow_durations = jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32)
    slow_durations = slow_durations.at[0, SLOW_CHANNEL_WARRIOR_CHARGE].set(5)
    slow_durations = slow_durations.at[2, SLOW_CHANNEL_HUNTER_BASIC].set(1)
    slow_durations = slow_durations.at[7, SLOW_CHANNEL_ROGUE_POISON].set(3)
    rogue_durations = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32).at[7].set(4)
    priest_durations = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32).at[9].set(1)

    eager = combat.derive_status_magnitudes(
        slow_durations,
        rogue_durations,
        priest_durations,
    )
    jitted = cast(
        tuple[Array, Array, Array],
        jax.jit(combat.derive_status_magnitudes)(
            slow_durations,
            rogue_durations,
            priest_durations,
        ),
    )

    for eager_output, jitted_output in zip(eager, jitted, strict=True):
        assert bool(jnp.array_equal(eager_output, jitted_output))


@pytest.mark.parametrize(
    "movement_scale",
    (
        pytest.param(1.0, id="legacy-compatible"),
        pytest.param(0.1, id="calibrated"),
    ),
)
def test_effective_movement_speed_aligns_status_and_participation_control(
    movement_scale: float,
) -> None:
    """Prove one derived speed represents every current voluntary-speed gate."""
    base_movement_speeds = jnp.full((MAX_AGENT_SLOTS,), 2.0, dtype=jnp.float32)
    active_and_alive_mask = jnp.arange(MAX_AGENT_SLOTS) < 5
    slow_durations = jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32)
    slow_durations = slow_durations.at[1, SLOW_CHANNEL_HUNTER_BASIC].set(1)
    slow_durations = slow_durations.at[2:5, :].set(1)
    freedom_durations = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    freedom_durations = freedom_durations.at[2].set(1).at[4].set(1)
    stun_durations = (
        jnp.zeros((MAX_AGENT_SLOTS, NUM_STUN_CHANNELS), dtype=jnp.int32)
        .at[4, STUN_CHANNEL_HUNTER_TRAP]
        .set(1)
    )

    effective_speeds = combat.derive_effective_movement_speeds(
        slow_durations,
        freedom_durations,
        stun_durations,
        jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        base_movement_speeds,
        2.0,
        active_and_alive_mask,
        movement_scale,
    )
    compiled_speeds = cast(
        Array,
        jax.jit(combat.derive_effective_movement_speeds)(
            slow_durations,
            freedom_durations,
            stun_durations,
            jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
            base_movement_speeds,
            2.0,
            active_and_alive_mask,
            movement_scale,
        ),
    )
    expected_without_calibration = jnp.asarray(
        (
            2.0,
            2.0 * combat.HUNTER_BASIC_SLOW_MULTIPLIER,
            2.0 * combat.PRIEST_HEAL_SPEED_FLOOR,
            2.0
            * combat.WARRIOR_CHARGE_SLOW_MULTIPLIER
            * combat.HUNTER_BASIC_SLOW_MULTIPLIER
            * combat.ROGUE_POISON_SLOW_MULTIPLIER,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        dtype=jnp.float32,
    )
    expected = expected_without_calibration * movement_scale

    assert effective_speeds.shape == (MAX_AGENT_SLOTS,)
    assert effective_speeds.dtype == jnp.float32
    assert bool(jnp.allclose(effective_speeds, expected))
    assert bool(jnp.array_equal(compiled_speeds, effective_speeds))


def test_passive_and_support_defaults_are_scalar_parameters() -> None:
    assert combat.MAGE_DAMAGE_AURA_RADIUS > 0.0
    assert combat.MAGE_DAMAGE_AURA_MULTIPLIER > 1.0

    assert combat.WARRIOR_DAMAGE_MITIGATION_AURA_RADIUS > 0.0
    assert 0.0 < combat.WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER < 1.0

    assert 0.0 < combat.PRIEST_HEAL_SPEED_FLOOR <= 1.0
    assert isinstance(combat.PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS, int)
    assert combat.PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS > 0


def test_class_catalog_helpers_preserve_vector_shape_dtype_and_row_alignment() -> None:
    class_ids = jnp.asarray(_canonical_class_ids(), dtype=jnp.int32)

    for catalog_name, helper_name in _CLASS_CATALOG_HELPER_NAMES:
        catalog = _catalog_array(catalog_name)
        selected_values = _catalog_helper(helper_name)(class_ids)

        assert selected_values.shape == class_ids.shape
        assert selected_values.dtype == catalog.dtype
        assert jnp.array_equal(selected_values, catalog[class_ids])


def test_class_catalog_helpers_preserve_batched_class_id_shape() -> None:
    class_ids = jnp.asarray(
        (
            (NEUTRAL_CLASS_ID, MAGE_CLASS_ID, WARRIOR_CLASS_ID),
            (HUNTER_CLASS_ID, ROGUE_CLASS_ID, PRIEST_CLASS_ID),
        ),
        dtype=jnp.int32,
    )

    for catalog_name, helper_name in _CLASS_CATALOG_HELPER_NAMES:
        catalog = _catalog_array(catalog_name)
        selected_values = _catalog_helper(helper_name)(class_ids)

        assert selected_values.shape == class_ids.shape
        assert selected_values.dtype == catalog.dtype
        assert jnp.array_equal(selected_values, catalog[class_ids])


def test_class_catalog_helpers_accept_scalar_class_ids() -> None:
    for catalog_name, helper_name in _CLASS_CATALOG_HELPER_NAMES:
        catalog = _catalog_array(catalog_name)
        selected_value = _catalog_helper(helper_name)(MAGE_CLASS_ID)

        assert selected_value.shape == ()
        assert selected_value.dtype == catalog.dtype
        assert bool(selected_value == catalog[MAGE_CLASS_ID])


@pytest.mark.parametrize(
    ("class_ids", "team_size"),
    (
        pytest.param(
            jnp.asarray(
                [
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
                ],
                dtype=jnp.int32,
            ),
            3,
            id="default-3v3-neutralizes-padded-non-neutral-slots",
        ),
        pytest.param(
            jnp.asarray(
                [
                    MAGE_CLASS_ID,
                    PRIEST_CLASS_ID,
                    HUNTER_CLASS_ID,
                    ROGUE_CLASS_ID,
                    WARRIOR_CLASS_ID,
                    ROGUE_CLASS_ID,
                    HUNTER_CLASS_ID,
                    WARRIOR_CLASS_ID,
                    PRIEST_CLASS_ID,
                    MAGE_CLASS_ID,
                ],
                dtype=jnp.int32,
            ),
            1,
            id="1v1-mage-vs-rogue",
        ),
        pytest.param(
            jnp.asarray(
                [
                    HUNTER_CLASS_ID,
                    MAGE_CLASS_ID,
                    PRIEST_CLASS_ID,
                    ROGUE_CLASS_ID,
                    WARRIOR_CLASS_ID,
                    ROGUE_CLASS_ID,
                    WARRIOR_CLASS_ID,
                    MAGE_CLASS_ID,
                    HUNTER_CLASS_ID,
                    PRIEST_CLASS_ID,
                ],
                dtype=jnp.int32,
            ),
            2,
            id="2v2-hunter-mage-vs-rogue-warrior",
        ),
        pytest.param(
            jnp.asarray(
                [
                    HUNTER_CLASS_ID,
                    HUNTER_CLASS_ID,
                    MAGE_CLASS_ID,
                    ROGUE_CLASS_ID,
                    PRIEST_CLASS_ID,
                    WARRIOR_CLASS_ID,
                    PRIEST_CLASS_ID,
                    ROGUE_CLASS_ID,
                    HUNTER_CLASS_ID,
                    MAGE_CLASS_ID,
                ],
                dtype=jnp.int32,
            ),
            2,
            id="duplicate-class-team-supported",
        ),
        pytest.param(
            _CONFIG_DEFAULT_5_CLASS_MIRROR,
            5,
            id="canonical-mirrored-5v5",
        ),
    ),
)
def test_reset_produces_correct_active_class_ids_and_class_stats(
    class_ids: Array,
    team_size: int,
) -> None:
    config = _config(team_size=team_size, class_ids=class_ids)
    state, obs, action_mask, _ = reset(config, jax.random.key(42))

    expected_active_mask = _expected_active_mask(team_size)
    expected_class_ids = _expected_resolved_class_ids(class_ids, team_size)

    profile = config.agent_profile
    _assert_array_equal(profile.active_mask, expected_active_mask, "active_mask")
    _assert_array_equal(state.alive_mask, expected_active_mask, "alive_mask")
    _assert_array_equal(profile.class_ids, expected_class_ids, "class_ids")

    _assert_float_array_close(
        profile.agent_radii,
        combat.BODY_RADIUS_BY_CLASS[expected_class_ids],
        "agent_radii",
    )
    _assert_float_array_close(
        profile.base_movement_speeds,
        combat.BASE_MOVEMENT_SPEED_BY_CLASS[expected_class_ids],
        "movement_speeds",
    )
    _assert_float_array_close(
        profile.observation_radii,
        combat.OBSERVATION_RADIUS_BY_CLASS[expected_class_ids],
        "observation_radii",
    )
    _assert_float_array_close(
        profile.basic_interaction_radii,
        combat.BASIC_INTERACTION_RADIUS_BY_CLASS[expected_class_ids],
        "basic_interaction_radii",
    )
    _assert_float_array_close(
        profile.ultimate_interaction_radii,
        combat.ULTIMATE_INTERACTION_RADIUS_BY_CLASS[expected_class_ids],
        "ultimate_interaction_radii",
    )
    _assert_float_array_close(
        profile.max_health,
        combat.MAX_HEALTH_BY_CLASS[expected_class_ids],
        "max_health",
    )
    _assert_float_array_close(
        state.current_health,
        profile.max_health,
        "current_health",
    )

    _assert_reset_combat_state_is_inert(state)
    _assert_self_features_project_state(obs, state, config)

    inactive_mask = jnp.logical_not(expected_active_mask)
    assert bool(jnp.all(action_mask.move_mask[inactive_mask, 0]))
    assert bool(jnp.all(jnp.sum(action_mask.move_mask[inactive_mask], axis=-1) == 1))
    assert bool(jnp.all(action_mask.select_target_mask[inactive_mask, 0]))
    assert bool(
        jnp.all(jnp.sum(action_mask.select_target_mask[inactive_mask], axis=-1) == 1)
    )
    assert bool(jnp.all(action_mask.use_ultimate_mask[inactive_mask, 0]))
    assert bool(
        jnp.all(jnp.sum(action_mask.use_ultimate_mask[inactive_mask], axis=-1) == 1)
    )
    assert bool(
        jnp.all(action_mask.select_target_use_ultimate_joint_mask[inactive_mask, 0, 0])
    )
    assert bool(
        jnp.all(
            jnp.sum(
                action_mask.select_target_use_ultimate_joint_mask[inactive_mask],
                axis=(-2, -1),
            )
            == 1
        )
    )


def test_reset_neutralizes_inactive_slots_even_when_supplied_classes_non_neutral() -> (
    None
):
    class_ids = jnp.asarray(
        [
            MAGE_CLASS_ID,
            WARRIOR_CLASS_ID,
            HUNTER_CLASS_ID,
            ROGUE_CLASS_ID,
            PRIEST_CLASS_ID,
            ROGUE_CLASS_ID,
            PRIEST_CLASS_ID,
            MAGE_CLASS_ID,
            WARRIOR_CLASS_ID,
            HUNTER_CLASS_ID,
        ],
        dtype=jnp.int32,
    )
    team_size = 1

    config = _config(team_size=team_size, class_ids=class_ids)
    state, obs, _, _ = reset(config, jax.random.key(42))

    expected_active_mask = _expected_active_mask(team_size)
    expected_class_ids = _expected_resolved_class_ids(class_ids, team_size)
    inactive_mask = jnp.logical_not(expected_active_mask)

    profile = config.agent_profile
    _assert_array_equal(profile.class_ids, expected_class_ids, "class_ids")

    assert bool(jnp.all(profile.class_ids[inactive_mask] == NEUTRAL_CLASS_ID))
    assert bool(jnp.all(profile.agent_radii[inactive_mask] == 0.0))
    assert bool(jnp.all(profile.base_movement_speeds[inactive_mask] == 0.0))
    assert bool(jnp.all(profile.observation_radii[inactive_mask] == 0.0))
    assert bool(jnp.all(profile.basic_interaction_radii[inactive_mask] == 0.0))
    assert bool(jnp.all(profile.ultimate_interaction_radii[inactive_mask] == 0.0))
    assert bool(jnp.all(profile.max_health[inactive_mask] == 0.0))
    assert bool(jnp.all(state.current_health[inactive_mask] == 0.0))

    _assert_self_features_project_state(obs, state, config)


def test_reset_starts_ultimate_remaining_cooldowns_at_zero_not_catalog_durations() -> (
    None
):
    config = _config(team_size=5)
    state, _, _, _ = reset(config, jax.random.key(42))

    assert bool(jnp.all(state.ultimate_cooldowns == 0))

    active_mask = _expected_active_mask(team_size=5)
    active_class_ids = config.agent_profile.class_ids[active_mask]
    catalog_cooldowns = combat.ULTIMATE_COOLDOWN_BY_CLASS[active_class_ids]

    assert bool(jnp.any(catalog_cooldowns > 0))


def test_reset_observation_radius_is_class_catalog_derived() -> None:
    class_ids = jnp.asarray(
        [
            MAGE_CLASS_ID,
            WARRIOR_CLASS_ID,
            HUNTER_CLASS_ID,
            ROGUE_CLASS_ID,
            PRIEST_CLASS_ID,
            PRIEST_CLASS_ID,
            ROGUE_CLASS_ID,
            HUNTER_CLASS_ID,
            WARRIOR_CLASS_ID,
            MAGE_CLASS_ID,
        ],
        dtype=jnp.int32,
    )

    config = _config(team_size=5, class_ids=class_ids)
    _, obs, _, _ = reset(config, jax.random.key(42))

    expected_observation_radii = combat.OBSERVATION_RADIUS_BY_CLASS[class_ids]

    _assert_float_array_close(
        config.agent_profile.observation_radii,
        expected_observation_radii,
        "profile.observation_radii",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_OBSERVATION_RADIUS],
        expected_observation_radii,
        "obs.self_features.observation_radius",
    )


def test_observation_relation_unit_feature_tables_keep_agent_schema_width() -> None:
    config = _config(team_size=3)
    state, obs, _, _ = reset(config, jax.random.key(42))

    assert obs.ally_unit_features.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
        UNIT_FEATURES,
    )
    assert obs.enemy_unit_features.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
        UNIT_FEATURES,
    )
    assert UNIT_FEATURES == SELF_FEATURES

    _assert_self_features_project_state(obs, state, config)
