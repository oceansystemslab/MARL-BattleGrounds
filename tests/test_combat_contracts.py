"""Combat contract tests for Milestone 5 class setup."""

import subprocess
import sys
from collections.abc import Callable
from importlib import import_module
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

import marl_battlegrounds.core.combat as combat_catalog
from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import (
    CLASS_NEUTRAL,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_OBSTACLE_SLOTS,
    NEUTRAL_CLASS_ID,
    NUM_CLASSES,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    EnvConfig,
)

_FORBIDDEN_COMBAT_IMPORT_MODULES: tuple[str, ...] = (
    "marl_battlegrounds.core.env",
    "marl_battlegrounds.core.geometry",
    "marl_battlegrounds.rendering",
    "marl_battlegrounds.rendering.geometry",
    "marl_battlegrounds.rendering.manual_control",
)

_FLOAT_CLASS_CATALOG_NAMES: tuple[str, ...] = (
    "MAX_HEALTH_BY_CLASS",
    "MOVEMENT_SPEED_BY_CLASS",
    "BODY_RADIUS_BY_CLASS",
    "BASIC_RANGE_BY_CLASS",
    "BASIC_DAMAGE_BY_CLASS",
    "BASIC_HEALING_BY_CLASS",
    "ULTIMATE_RANGE_BY_CLASS",
)

_INTEGER_CLASS_CATALOG_NAMES: tuple[str, ...] = ("ULTIMATE_COOLDOWN_BY_CLASS",)

_CLASS_CATALOG_NAMES: tuple[str, ...] = (
    *_FLOAT_CLASS_CATALOG_NAMES,
    *_INTEGER_CLASS_CATALOG_NAMES,
)

_CLASS_CATALOG_HELPER_NAMES: tuple[tuple[str, str], ...] = (
    ("MAX_HEALTH_BY_CLASS", "get_max_health_by_class_ids"),
    ("MOVEMENT_SPEED_BY_CLASS", "get_movement_speed_by_class_ids"),
    ("BODY_RADIUS_BY_CLASS", "get_body_radius_by_class_ids"),
    ("BASIC_RANGE_BY_CLASS", "get_basic_range_by_class_ids"),
    ("BASIC_DAMAGE_BY_CLASS", "get_basic_damage_by_class_ids"),
    ("BASIC_HEALING_BY_CLASS", "get_basic_healing_by_class_ids"),
    ("ULTIMATE_RANGE_BY_CLASS", "get_ultimate_range_by_class_ids"),
    ("ULTIMATE_COOLDOWN_BY_CLASS", "get_ultimate_cooldown_by_class_ids"),
)

_CatalogHelper = Callable[[int | Array], Array]

# Test helpers ---


def _empty_obstacles() -> Array:
    """Return a zero-filled obstacle feature table."""
    return jnp.zeros(shape=(MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _config() -> EnvConfig:
    """Return a deterministic combat-contract test config."""
    return EnvConfig(
        team_size=3,
        max_steps=1000,
        map_width=20.0,
        map_height=12.0,
        default_agent_radius=0.5,
        default_movement_speed=1.0,
        default_observation_radius=8.0,
        default_basic_interaction_radius=6.0,
        default_ultimate_interaction_radius=9.0,
        obstacles=_empty_obstacles(),
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
    return cast(Array, getattr(combat_catalog, name))


def _catalog_helper(name: str) -> _CatalogHelper:
    """Return a named class catalog lookup helper."""
    return cast(_CatalogHelper, getattr(combat_catalog, name))


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


def test_reset_initializes_all_slots_to_neutral_class_id() -> None:
    state, _, _, _ = reset(_config(), jax.random.key(42))

    assert state.class_ids.shape == (MAX_AGENT_SLOTS,)
    assert state.class_ids.dtype == jnp.int32
    assert jnp.all(state.class_ids == CLASS_NEUTRAL)


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

    assert bool(combat_catalog.BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID] > 0)
    assert bool(combat_catalog.BASIC_DAMAGE_BY_CLASS[PRIEST_CLASS_ID] == 0)

    for class_id in damage_class_ids:
        assert bool(combat_catalog.BASIC_DAMAGE_BY_CLASS[class_id] > 0)
        assert bool(combat_catalog.BASIC_HEALING_BY_CLASS[class_id] == 0)


def test_rogue_movement_speed_role_distinction_is_represented() -> None:
    non_rogue_class_ids = jnp.asarray(
        (MAGE_CLASS_ID, WARRIOR_CLASS_ID, HUNTER_CLASS_ID, PRIEST_CLASS_ID),
        dtype=jnp.int32,
    )

    assert bool(
        jnp.all(
            combat_catalog.MOVEMENT_SPEED_BY_CLASS[ROGUE_CLASS_ID]
            > combat_catalog.MOVEMENT_SPEED_BY_CLASS[non_rogue_class_ids]
        )
    )


def test_warrior_basic_and_charge_ranges_are_separately_represented() -> None:
    assert bool(
        combat_catalog.ULTIMATE_RANGE_BY_CLASS[WARRIOR_CLASS_ID]
        > combat_catalog.BASIC_RANGE_BY_CLASS[WARRIOR_CLASS_ID]
    )


def test_mage_burst_is_representable_as_no_target_self_buff() -> None:
    assert bool(combat_catalog.ULTIMATE_RANGE_BY_CLASS[MAGE_CLASS_ID] == 0)
    assert combat_catalog.MAGE_ULT_DAMAGE_DURATION_TICKS > 0
    assert combat_catalog.MAGE_ULT_DAMAGE_MULTIPLIER > 1.0


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

    assert 0.0 < combat_catalog.GLOBAL_SLOW_FLOOR <= 1.0
    assert 0.0 < combat_catalog.ROGUE_POISON_ANTI_HEAL_MULTIPLIER <= 1.0

    for name in slow_multiplier_names:
        value = getattr(combat_catalog, name)

        assert isinstance(value, float), name
        assert 0.0 < value <= 1.0, name

    for name in duration_names:
        value = getattr(combat_catalog, name)

        assert isinstance(value, int), name
        assert value > 0, name


def test_passive_and_support_defaults_are_scalar_parameters() -> None:
    assert combat_catalog.MAGE_DAMAGE_AURA_RADIUS > 0.0
    assert combat_catalog.MAGE_DAMAGE_AURA_MULTIPLIER > 1.0

    assert combat_catalog.WARRIOR_MITIGATION_AURA_RADIUS > 0.0
    assert 0.0 < combat_catalog.WARRIOR_MITIGATION_AURA_MULTIPLIER < 1.0

    assert combat_catalog.PRIEST_ULT_HEAL_AMOUNT > 0.0
    assert 0.0 < combat_catalog.PRIEST_HEAL_SPEED_FLOOR <= 1.0
    assert isinstance(combat_catalog.PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS, int)
    assert combat_catalog.PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS > 0


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
